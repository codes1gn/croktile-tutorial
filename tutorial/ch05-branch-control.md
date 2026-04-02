# Warp Specialization and Control Flow

Chapter 4 walked through the MMA lifecycle on Hopper: `mma.load` from shared, `mma.row.row` on tensor cores, accumulation in registers, then `mma.store` and `dma.copy` back to global memory. That pipeline is effective. It also has a structural limitation: while the tensor cores are busy multiplying, the memory system often idles, because every thread in the block runs the same kernel body. Nobody is left to issue the next bulk load while MMA is in flight.

Real pipelines are not uniform like that. A producer loads data; a consumer turns it into math. On a CPU you might use threads and queues. On a GPU, the classic CUDA approach is to keep one kernel but add careful synchronization so different warps play different roles. Croktile makes the split **structural**: `inthreads.async` assigns different straight-line programs to different warpgroups, so the hardware can overlap DMA with MMA without every thread executing both paths. When the schedule itself needs a decision — for example, skipping tiles past the end of the problem — you use an ordinary **`if`**, which is a **runtime** branch, not a role split.

This chapter has one thread: how to give warpgroups different jobs, and when to guard work with conditionals. The second half covers **persistent kernels**, where a fixed pool of blocks stripes across many output tiles and `if` prevents out-of-bounds stores.

![1P1C timeline: producer DMA and consumer MMA overlap on a shared time axis](../assets/images/ch05/fig1_role_split_dark.png#only-dark)
![1P1C timeline: producer DMA and consumer MMA overlap on a shared time axis](../assets/images/ch05/fig1_role_split_light.png#only-light)

## `inthreads.async`: Structural Split, Not a Runtime Branch

`inthreads.async (condition)` means: only threads for which `condition` is true **have** this block in their program at all. It is **not** “every thread evaluates the condition and some skip the body,” which is what an `if` does. The distinction matters for how you think about the hardware:

- **Structural split** (`inthreads.async`): two (or more) separate straight-line bodies, compiled for different subsets of threads. Producer warpgroups and consumer warpgroups are different programs that happen to share an address space.
- **Runtime branch** (`if`): one program; every live thread tests the predicate; some execute the taken branch and some do not.

In traditional GPU programming, the entire block usually shares one kernel. Overlap between load and math then depends on instruction-level interleaving or hand-rolled warp specialization with barriers and atomics. Croktile’s `inthreads.async` pushes the role boundary into the language so the split is explicit and the bodies stay simple.

The canonical pattern is **one producer + one consumer (1P1C)** for matmul: one warpgroup issues DMA (or TMA) into shared memory; another runs MMA on those tiles. The skeleton without synchronization looks like this:

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

**`parallel p1 by 2 : group-4`** — Two warpgroups, four warps each (128 threads per warpgroup), indexed by `p1`.

**`inthreads.async (p1 == 0)`** — Only warpgroup 0 compiles and executes the producer body; it is not an empty trip for everyone else.

**`inthreads.async (p1 == 1)`** — Only warpgroup 1 runs the consumer body. The two blocks are not branches of one loop; they are two roles.

Compare this with Chapter 3’s `parallel`, where every thread runs the same body. Here the parallel index **selects** which job description applies. The hardware can schedule TMA work on the producer warpgroup while WGMMA runs on the consumer — overlap in time, not time-slicing one instruction stream.

## A 1P1C Matmul Skeleton

Here is how the split sits inside a Hopper matmul. Events, wait, and trigger are omitted on purpose; [Chapter 6](ch06-synchronization.md) adds synchronization. Focus on who does what:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        // Producer: walk K, load tiles into shared
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;
        }
      }

      inthreads.async (p1 == 1) {
        // Consumer: walk K, MMA on loaded tiles
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

**`parallel {block_m, block_n} by [...] : block`** — Same tile grid as earlier chapters; `cdiv` (ceiling division) counts tiles along M and N when dimensions are not exact multiples.

**Producer `foreach`** — Walks the K dimension with `cdiv(K, MATMUL_TILE_K)` steps; only the producer issues `dma.copy` into `lhs_load_s` and `rhs_load_s`.

**Consumer `mma.fill` / `mma.row.row` / `mma.store`** — The consumer never issues those DMA fills; it only reads shared, accumulates in `mc`, and writes the result tile.

**Missing coordination** — The two sides both loop over K independently here. The consumer assumes each K-slab is ready when it reads it; making that true is synchronization (Chapter 6).

## `if` Guards: Runtime Conditional Execution

Sometimes you need a predicate every thread evaluates. Croktile’s `if` behaves like C:

```choreo
if (tile_id < total_tiles) {
  // only execute this body when the condition is true
}
```

**`if (tile_id < total_tiles)`** — All threads in the scope test the condition; threads where it is false skip the body. That is the opposite of `inthreads.async`: one program, divergent execution.

This shows up most often in **persistent kernels**, where the number of loop iterations can leave some blocks with a “padding” iteration that does not correspond to a real tile.

## Persistent Kernels

In Chapters 3–4, the grid grew with the problem: roughly one block per output tile. For large matrices that can mean huge launch counts. The GPU runs blocks in **waves**; the last wave often leaves SMs partially idle — **tail underutilization**.

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

**`int total_tiles`** — Linear count of output tiles; product of the per-axis tile counts, each computed with `cdiv` so partial tiles are counted.

**`parallel block_id by NUM_SMS : block`** — Fixed worker count; `block_id` names which persistent worker this is, not a single output tile.

**`foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)]`** — Each block steps through its share of iterations; the ceiling may add one extra iteration for some blocks.

**`tile_id = tile_iter # block_id`** — Composes iteration with block index to stripe across the linear tile list (same `#` operator as Chapter 2, used here for scheduling).

**`block_m` / `block_n`** — Map linear `tile_id` to 2D tile coordinates using division and modulus with `cdiv(N, MATMUL_WARP_N)` as the row width in tiles.

**`if (tile_id < total_tiles)`** — Skips TMA, MMA, and store when the stripe walks past the last real tile. Without this guard, you would read and write out of bounds.

The inner K-loop and MMA body match the non-persistent style from Chapter 4. Only the **wrapper** changed: fixed launch, striping, and a guard. Partial tiles at domain boundaries still need masks or epilogue handling in production kernels; `cdiv` is how you size grids and loops when divisibility is not guaranteed.

![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_dark.png#only-dark)
![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_light.png#only-light)

## Choosing Between Data-Dependent and Persistent Grids

| Aspect | One block per tile | Persistent (`NUM_SMS` blocks) |
|--------|-------------------|-------------------------------|
| Grid size | Grows with problem | Fixed |
| Tail utilization | Last wave may leave SMs idle | All SMs stay busy |
| Extra constructs | Minimal | `total_tiles`, `tile_iter # block_id`, `if` |
| Complexity | Lower | Higher |

Neither layout changes the mathematical result by itself; both can match modulo floating-point associativity. Persistent scheduling tends to pay off when `total_tiles` is much larger than the number of SMs — typical for large GEMMs.

## `parallel.async` and `stream s`: Non-Blocking Launch

Everything above runs inside the kernel. Sometimes the control you need is at the **host level**: launch a grid without blocking the host thread, or pin different grids to different CUDA streams so they can execute concurrently on the GPU.

```choreo
parallel.async {px, py} by [grid_m, grid_n] : block {
  stream s;
  // kernel body
}
```

**`parallel.async`** returns control to the host immediately — the kernel is enqueued but the host does not wait for it to finish. This is the Croktile equivalent of `cudaLaunchKernel` with a non-default stream.

**`stream s`** inside the block body pins the kernel to CUDA stream `s`. Multiple `parallel.async` blocks with different streams can overlap on the GPU if there are enough SMs. Without `stream s`, the default stream serializes launches.

This is **host orchestration**, not in-kernel control flow. It does not replace `inthreads.async` for warp specialization or `if` for runtime predicates — it decides *when* and *where* a grid runs relative to other grids.

## Chapter Summary

| Topic | Takeaway |
|-------|----------|
| Uniform vs specialized | Same kernel for every thread maximizes simplicity; role splits overlap memory and compute. |
| `inthreads.async` | Structural: different bodies for different threads — not a shared `if`. |
| `if` | Runtime: every thread evaluates the condition; false threads skip the body. |
| Persistent kernels | Fixed `NUM_SMS` blocks, linear tile ids, striping with `#`, guard with `if`. |
| `parallel.async` / `stream s` | Host-side async launch and stream assignment — orthogonal to in-kernel specialization. |
| `cdiv` | Ceiling division for tile counts and loop bounds (used throughout; no separate recipe needed). |

**New syntax**

| Syntax | Meaning |
|--------|---------|
| `inthreads.async (condition)` | Only threads satisfying `condition` include this block — structural role split |
| `if (expr) { ... }` | Runtime conditional — skip body when `expr` is false |
| `tile_id = tile_iter # block_id` | Compose iteration index with block index for tile striping |
| `int total_tiles = expr` | Local integer in a Croktile function |
| `parallel.async ... : block` | Non-blocking async kernel launch |
| `stream s` | Bind kernel body to CUDA stream `s` |

Producer and consumer still need a shared notion of “ready” before the 1P1C skeleton is safe. [Chapter 6](ch06-synchronization.md) adds **events**, **swap**, and **pipeline** patterns so the two sides can overlap in time without racing on shared memory.
