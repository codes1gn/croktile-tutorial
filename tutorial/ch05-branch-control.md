# Warp Specialization: Concurrent Regions and Events

Every programming language that targets a parallel machine must answer one question: **what happens when different threads need to do different things?**

On a CPU, the answer is trivial — each core has its own program counter, so each thread runs whatever code it likes. On a GPU, the answer is deeply constrained. The SIMT (Single Instruction, Multiple Thread) execution model groups threads into **warps** of 32 that share a single instruction pointer. When a branch sends some threads one way and other threads another way, the hardware cannot simply fork — it must **serialize** the divergent paths and mask inactive threads. This is the fundamental tension of GPU programming languages: the hardware wants uniformity, but real programs need heterogeneity.

CUDA exposes exactly one control-flow primitive for this: `if`. All threads evaluate the condition; threads where it is false are masked (deactivated); both paths execute sequentially within the warp. This is **predicated execution** — simple, universal, and sometimes ruinously expensive. A warp that diverges on an `if` runs both sides, throwing away half its throughput on each.

But predicated execution is not enough. Consider a matmul kernel where one group of threads should continuously fetch tiles from global memory while another group continuously multiplies tiles on the tensor cores. This is not a data-dependent branch — it is two **structurally different programs** that happen to share an address space. Trying to express this with `if` means one program pauses while the other runs. There is no overlap, no pipeline.

Croqtile introduces two additional control-flow primitives to fill this gap:

- **`inthreads.async`** — **structured concurrent regions**: compile-time partitioning of threads into groups that run **different programs simultaneously**. The compiler generates separate instruction streams; the hardware schedules them independently.
- **`shared event` / `wait` / `trigger`** — **inter-region signaling**: lightweight synchronization tokens that let concurrent regions communicate safely.

Together with `if`, these three primitives cover the full spectrum of control flow in a GPU kernel: data-dependent branching, structural program composition, and inter-program coordination.

## Predicated execution with `if`

Croqtile's `if` behaves like its C counterpart:

```choreo
if (tile_id < total_tiles) {
  // body executes only when the condition is true
}
```

All threads in scope evaluate the condition. Threads where it is false skip the body. Within a single warp, if some threads take the branch and others do not, the hardware **serializes** the two paths — threads on the skipped side sit idle while the taken side runs, then vice versa. This is **warp divergence**, and it is the price of runtime flexibility.

**When to use `if`:** data-dependent decisions that cannot be resolved at compile time. Bounds checks, partial-tile guards, conditional accumulation. The condition can depend on runtime values — loop indices, input data, tile coordinates.

**Cost model:** divergence within a warp serializes both paths. Divergence across warps (where all threads in each warp agree) costs nothing — the hardware simply skips the not-taken path. The practical rule: keep `if` conditions **warp-uniform** (all 32 threads agree) whenever possible.

## Structured concurrent regions with `inthreads.async`

`inthreads.async` solves a fundamentally different problem than `if`. Instead of asking "should this thread execute this code?" at runtime, it says "this group of threads runs this program, that group runs that program" at **compile time**.

```choreo
parallel p1 by 2 : group-4 {

  inthreads.async (p1 == 0) {
    // program A: only warpgroup 0 compiles and runs this
  }

  inthreads.async (p1 == 1) {
    // program B: only warpgroup 1 compiles and runs this
  }
}
```

The distinction from `if` is structural, not just performance:

| | `if` (predicated execution) | `inthreads.async` (structured concurrency) |
|---|---|---|
| **Resolution** | Runtime — every thread evaluates the condition | Compile time — thread assignment is fixed |
| **Instruction streams** | One program; divergent threads masked | Separate programs per region |
| **Execution** | Serial within a warp if divergent | Concurrent across warpgroups |
| **PL analogy** | `if`/`else` in any language | `async`/`spawn` in structured concurrency (Trio, Go goroutines, Cilk) |
| **GPU analogy** | SPMD with masking | MPMD within a single kernel launch |

**Why "structured"?** The regions are lexically scoped — the compiler knows at parse time which threads belong to which region. There is no dynamic spawn, no unbounded concurrency. Each `inthreads.async` block is a static partition. This is what makes it amenable to compile-time analysis: the compiler can allocate registers differently for each region, emit different instruction schedules, and verify that shared resources are used safely.

**The `.async` modifier.** Without `.async`, `inthreads` would execute regions sequentially — thread subsets take turns. The `.async` suffix is the concurrency modifier: it tells the compiler and hardware that the regions may overlap in time. This is analogous to the `async` keyword in structured concurrency frameworks — it marks a region as independently schedulable.

The figure below shows the effect. The top timeline shows a single warpgroup alternating between DMA and MMA (sequential, no overlap). The bottom shows two warpgroups with `inthreads.async` — the producer's DMA and the consumer's MMA overlap in time:

![Uniform vs structured-concurrent execution: sequential alternation vs overlapping regions](../assets/images/ch05/fig1_role_comparison_dark.png#only-dark)
![Uniform vs structured-concurrent execution: sequential alternation vs overlapping regions](../assets/images/ch05/fig1_role_comparison_light.png#only-light)

*Top: one warpgroup alternates DMA and MMA — each waits for the other. Bottom: `inthreads.async` partitions into two concurrent programs — DMA and MMA overlap, roughly halving wall-clock time.*

### The canonical pattern: 1 producer + 1 consumer

The most common use of `inthreads.async` is the **1P1C** (one producer, one consumer) split for matmul:

```choreo
parallel p1 by 2 : group-4 {

  inthreads.async (p1 == 0) {
    // producer: only warpgroup 0 runs this
    // issue DMA / TMA loads, fill shared memory
  }

  inthreads.async (p1 == 1) {
    // consumer: only warpgroup 1 runs this
    // run MMA on shared memory, accumulate results
  }
}
```

**`parallel p1 by 2 : group-4`** — Two warpgroups (128 threads each), indexed by `p1`.

**`inthreads.async (p1 == 0)`** — Warpgroup 0 compiles and runs the producer body; warpgroup 1 never sees this code.

**`inthreads.async (p1 == 1)`** — Warpgroup 1 runs the consumer body. The two blocks are separate programs sharing an address space.

But sharing an address space is exactly what makes this dangerous. Without coordination, the consumer might read a buffer before the producer has finished writing it. This is where events come in.

## Inter-region signaling with events

When `inthreads.async` creates concurrent regions, those regions need a way to communicate. Croqtile provides **events** — lightweight synchronization tokens declared in shared memory:

```choreo
shared event full;
shared event empty;
```

Events have two operations:

- **`trigger name`** — signal that a condition is met (e.g., "data is ready")
- **`wait name`** — block until the corresponding `trigger` fires

The producer calls `trigger full` after writing a tile to signal "data ready." The consumer calls `wait full` before reading, blocking until the signal arrives. Symmetrically, the consumer triggers `empty` after finishing its read (the buffer can be reused), and the producer waits on `empty` before writing the next tile.

This is a **credit-based bounded buffer** protocol — the same pattern used in operating systems (semaphores), network flow control (TCP window), and hardware (warp barriers). `full` is the "data available" credit; `empty` is the "buffer free" credit.

### Event arrays for multi-stage pipelines

For pipelines with multiple buffered stages, declare event arrays:

```choreo
shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
```

Each physical buffer slot gets its own `full`/`empty` pair. The ring index `stage = iv_k % MATMUL_STAGES` maps the unbounded K iteration to a fixed number of physical slots. With four stages, the producer can run several tiles ahead before blocking on `wait empty`.

### Bootstrap protocol

The consumer must **seed** the `empty` credits before the K-loop starts:

```choreo
foreach {s} in [MATMUL_STAGES] {
  trigger empty[s];
}
```

Without this bootstrap, the producer's first `wait empty[0]` blocks forever — a deadlock, not a mysterious MMA bug. This is a common pitfall: every bounded buffer protocol requires initial credits.

[Chapter 6](ch06-synchronization.md) develops the full double-buffered and multi-stage pipeline kernels that put these primitives to work. The examples there compose `inthreads.async`, events, `swap`, and `mma.commit` into complete, runnable matmul pipelines.

## A 1P1C matmul skeleton

Here is how the three primitives sit together inside a Hopper matmul. Event-based synchronization is omitted — [Chapter 6](ch06-synchronization.md) adds the full pipeline protocol. Focus on the program structure:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;
        }
      }

      inthreads.async (p1 == 1) {
        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load lhs_load_s.chunkat(_, iv_warp);
            mb = mma.load rhs_load_s.chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**Producer `foreach`** — Walks K with `cdiv(K, MATMUL_TILE_K)` steps; warpgroup 0 issues `dma.copy` into shared memory.

**Consumer `mma` path** — Warpgroup 1 never touches those DMAs; it reads shared memory, accumulates in `mc`, and writes the result.

**Missing coordination** — Both sides loop over K independently. The consumer assumes each K-slab is ready when it reads. Making that assumption correct requires events ([Chapter 6](ch06-synchronization.md)).

## Persistent scheduling and the `if` guard

In Chapters 3–4, the grid grew with the problem: roughly one block per output tile. For large matrices that means large launch counts, and the last wave of blocks often leaves SMs partially idle — **tail underutilization**.

A **persistent kernel** fixes the launch size (often near the SM count) and lets each block iterate over multiple tiles:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  int total_tiles = cdiv(M, MATMUL_WARP_M) * cdiv(N, MATMUL_WARP_N);

  parallel block_id by NUM_SMS : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)] {
      tile_id = tile_iter # block_id;

      if (tile_id < total_tiles) {
        block_m = tile_id / cdiv(N, MATMUL_WARP_N);
        block_n = tile_id % cdiv(N, MATMUL_WARP_N);

        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k) => rhs_load_s;

          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            parallel p by 1 : group-4 {
              ma = mma.load lhs_load_s.chunkat(_, iv_warp);
              mb = mma.load rhs_load_s.chunkat(_, iv_warp);
              mma.row.row mc, ma, mb;
            }
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**`parallel block_id by NUM_SMS : block`** — Fixed worker count.

**`tile_id = tile_iter # block_id`** — Composes iteration with block index to stripe across tiles.

**`if (tile_id < total_tiles)`** — The `if` guard: a runtime predicate that skips the body for padding iterations. This is exactly the use case `if` is designed for — a data-dependent decision, not a structural partition.

![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_dark.png#only-dark)
![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_light.png#only-light)

### Data-dependent vs persistent grids

| Aspect | One block per tile | Persistent (`NUM_SMS` blocks) |
|--------|-------------------|-------------------------------|
| Grid size | Grows with problem | Fixed |
| Tail utilization | Last wave may leave SMs idle | All SMs stay busy |
| Extra constructs | Minimal | `total_tiles`, `tile_iter # block_id`, `if` |
| Complexity | Lower | Higher |

## `parallel.async` and `stream s`: host-level concurrency

Everything above runs inside a kernel. Sometimes you need concurrency at the **host level**: launch a grid without blocking the CPU, or pin different grids to different CUDA streams so they can execute concurrently on the GPU.

```choreo
parallel.async {px, py} by [grid_m, grid_n] : block {
  stream s;
  // kernel body
}
```

**`parallel.async`** returns control to the host immediately — the kernel is enqueued but the host does not wait for completion. This is the Croqtile equivalent of `cudaLaunchKernel` with a non-default stream.

**`stream s`** pins the kernel to CUDA stream `s`. Multiple `parallel.async` blocks with different streams can overlap on the GPU if there are enough SMs.

This is **host orchestration**, orthogonal to in-kernel control flow. It does not replace `inthreads.async` for thread partitioning or `if` for runtime predicates — it decides *when* and *where* a grid runs relative to other grids.

## New syntax

| Syntax | Meaning |
|--------|---------|
| `if (expr) { ... }` | Predicated execution — runtime conditional, divergent threads masked |
| `inthreads.async (condition)` | Structured concurrent region — compile-time thread partitioning |
| `shared event name` | Declare a synchronization token in shared memory |
| `shared event name[N]` | Declare N synchronization tokens |
| `trigger name` | Signal that a condition is met |
| `wait name` | Block until the corresponding `trigger` fires |
| `tile_id = tile_iter # block_id` | Compose indices for tile striping |
| `int total_tiles = expr` | Local integer variable |
| `parallel.async ... : block` | Non-blocking kernel launch |
| `stream s` | Bind kernel to CUDA stream `s` |

## Chapter summary

| Concept | Primitive | When to use |
|---------|-----------|-------------|
| Predicated execution | `if` | Data-dependent decisions (bounds, conditions) |
| Structured concurrency | `inthreads.async` | Compile-time thread partitioning (producer/consumer, heterogeneous roles) |
| Inter-region signaling | `shared event` / `wait` / `trigger` | Coordination between concurrent regions |
| Host concurrency | `parallel.async` / `stream s` | Multi-kernel overlap, non-blocking launch |
| Persistent scheduling | `if` + `foreach` + `#` | Fixed grid size, tile striping with padding guard |

The 1P1C skeleton above is incomplete: without `wait` / `trigger`, the consumer can read before the producer has finished writing. [Chapter 6](ch06-synchronization.md) adds the full synchronization protocols — `swap` for single-schedule double buffering, events for multi-warpgroup pipelines — so the pipeline runs safely and at full throughput.
