# Synchronization: Pipelines, Events, and Double Buffering

Chapter 5 carved the matmul into two jobs: a **producer** warpgroup that issues loads into shared memory, and a **consumer** warpgroup that runs MMA on those tiles. The skeleton was honest about the split, but it quietly assumed something impossible: that the consumer could treat shared memory as if it updated the moment the producer wrote it. Real hardware does not work that way. Without coordination, the consumer might read a tile mid-write, or the producer might stomp a buffer still in use — classic **data races**.

This chapter is one story: **how to make pipelined execution safe.** Croktile gives you **events** so separate roles can signal readiness, **`swap` / `rotate`** so double- or multi-buffering stays legible, **`dma.copy.async`** so loads can overlap with compute, and the **prologue / steady-state / epilogue** pattern to structure the K-loop. We start from the single-threaded case (same program counter loads and computes), then add events when producer and consumer truly diverge.

## Why You Cannot "Just Overlap" Without Coordination

Walk the K loop of a tiled matmul with **one** staging buffer. Each iteration must: copy the A- and B-tiles into shared memory, wait until those copies are visible, then run MMA on that tile. If you tried to start the *next* iteration's copies while MMA still reads the same buffer, you would **overwrite** bytes the tensor cores are consuming. No amount of wishful scheduling fixes that; you need a happens-before relationship between "bytes landed" and "MMA reads them."

In hand-written CUDA, people enforce that with **barriers** (`__syncthreads`), **atomics**, or **CUDA events** wired between streams. The bookkeeping explodes quickly: you track which phase owns which buffer, which fence clears which hazard, and you hope every path through the loop matches. Croktile narrows the design space: **events** for cross-role signaling, **`swap`** for rotating buffer *names* without copying data, and explicit **`wait` / `trigger`** so the credit flow stays visible in the source.

The picture below contrasts a strict **load-then-compute** staircase with **double-buffered** overlap: same logical work, less idle time on the memory or math side when the pipeline fills.

![Sequential vs double-buffered K-tile timelines (schematic)](../assets/images/ch06/fig1_pipeline_timeline_dark.png#only-dark)
![Sequential vs double-buffered K-tile timelines (schematic)](../assets/images/ch06/fig1_pipeline_timeline_light.png#only-light)

## Double Buffering with `swap`

Give the K-loop **two** logical buffers. While MMA drains buffer 0, DMA fills buffer 1 with the next tile. After the math step, **swap** the handles: what was "next" becomes "current," and the freed slot is ready for the following load. Croktile spells this with `dma.copy.async` (non-blocking copy), `dma.any` (a placeholder future), `swap` (exchange futures), and a three-phase loop.

```choreo
__co__ auto matmul(s32 [M, K] lhs, s32 [K, N] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel {px, py} by [8, 16] : block
    parallel {qx, qy} by [16, 16] : thread {

    with tile_k in 16 {
      // Prologue: start loading tile 0
      lf0 = dma.copy lhs.chunkat(px, tile_k) => shared;
      rf0 = dma.copy rhs.chunkat(tile_k, py) => shared;

      // Placeholder futures for buffer 1
      lf1 = dma.any;
      rf1 = dma.any;

      // Steady state: load next tile while computing on current
      foreach tile_k(1:) {
        lf1 = dma.copy lhs.chunkat(px, tile_k) => shared;
        rf1 = dma.copy rhs.chunkat(tile_k, py) => shared;

        foreach k in [256 / #tile_k]
          output.at(px#qx, py#qy) += lf0.data.at(qx, k) * rf0.data.at(k, qy);

        swap(lf0, lf1);
        swap(rf0, rf1);
      }

      // Epilogue: compute on the last loaded tile
      foreach k in [256 / #tile_k]
        output.at(px#qx, py#qy) += lf0.data.at(qx, k) * rf0.data.at(k, qy);
    }
  }

  return output;
}
```

### **`with tile_k in 16`**

```choreo
with tile_k in 16 {
```

Opens a **scoped region** and binds `tile_k` as a tile axis with extent 16. Inside the block, `tile_k` is the chunk index for `chunkat` along K, and `#tile_k` is 16 — "within this scope, K is divided into 16 tiles."

### **`dma.any`: placeholder futures**

```choreo
lf1 = dma.any;
rf1 = dma.any;
```

`dma.any` is a future that does not yet represent a transfer. It exists so the type system has something to `swap` against on the first steady-state iteration. Before any use of `lf1.data`, a real `dma.copy` has been assigned.

### **`foreach tile_k(1:)`: sliced iteration**

```choreo
foreach tile_k(1:) {
```

`(1:)` means tile indices `1, 2, …` through the end. Tile 0 was loaded in the prologue into `lf0`/`rf0`.

### **The three phases**

**Prologue.** Issue loads for tile 0 into `lf0`/`rf0`. No compute yet.

**Steady state.** For each later tile: start loads into `lf1`/`rf1`, compute on `lf0`/`rf0` from the previous iteration, then `swap` so names track the active buffers. New copies land in `lf1`/`rf1` **before** the compute reads `lf0`/`rf0`, so you never read a buffer being overwritten.

**Epilogue.** After the last swap, `lf0`/`rf0` hold the final tile; one more compute pass drains them.

### **`swap`: names, not bytes**

`swap(lf0, lf1)` exchanges **future handles**. Shared-memory contents stay where the hardware placed them; only Croktile-level names rotate. The same idiom in CUDA is often a `^ 1` buffer index or a boolean phase; here the intent is explicit. For triple buffering, `rotate(f0, f1, f2)` cycles three handles in one step.

*Note: this example uses `s32` with scalar accumulation and `with tile_k` scoping — a deliberately simplified kernel style to isolate the `swap` mechanism. The same `swap` pattern applies verbatim to FP16/MMA kernels from Chapter 4; only the compute body changes.*

### **`auto` return type**

`__co__ auto matmul(...)` lets Croktile infer the result type from `return output`, which keeps the signature aligned with shape expressions.

## Events: When Producer and Consumer Are Different Programs

`swap` works when **one** thread group interleaves loads and MMA in a single schedule. Warp specialization (Chapter 5) puts loading and computing on **different** warpgroups with different program counters. They cannot share a `swap` schedule line-by-line; they need **events** — named synchronization in shared scope.

```choreo
shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
```

`wait event_name` blocks until that event has been signaled; `trigger event_name` wakes waiters. For the 1P1C matmul, a common convention is:

- `full[s]` — stage `s` has been filled; the consumer may read it.
- `empty[s]` — the consumer has released stage `s`; the producer may overwrite it.

Staging for tiles that many threads share still lives in **`=> shared`**; Chapter 2 and Chapter 3 already covered **local** vs **shared** placement — the new ingredient here is *who* waits on *which* event, not the memory space alone.

### **1P1C kernel with events**

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
    shared f16 [MATMUL_STAGES * MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_STAGES * MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait empty[stage];
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)
            => lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0);
          dma.copy rhs.chunkat(block_n, iv_k)
            => rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0);
          trigger full[stage];
        }
      }

      inthreads.async (p1 == 1) {
        mc = mma.fill.f16 0.0f;
        foreach {s} in [MATMUL_STAGES] {
          trigger empty[s];
        }
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait full[stage];
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mb = mma.load rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
          mma.commit;
          trigger empty[stage];
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**Ring index.** `stage = iv_k % MATMUL_STAGES` maps the unbounded K iteration to a fixed number of physical slots — double buffering generalized to N buffers. With four stages, the producer can run several tiles ahead before blocking on `wait empty`.

**Producer path.** For each `iv_k`, `wait empty[stage]` acquires a free slot, the `dma.copy` lines fill `lhs_load_s`/`rhs_load_s` at that `stage`, then `trigger full[stage]` hands the slot to the consumer. If you drop the initial `trigger empty` bootstrap in the consumer prologue, the first `wait empty` never completes — a deadlock, not a mysterious MMA bug.

**Consumer path.** The loop `foreach {s} in [MATMUL_STAGES] { trigger empty[s]; }` runs **before** the K-loop so every stage starts with an **empty** credit. Otherwise the producer would wait forever on the first tile. Then each `iv_k`: `wait full[stage]`, MMA over that stage, `mma.commit`, `trigger empty[stage]` to release the slot. Skipping `mma.commit` or signaling `empty` before the math is really done risks reuse while operands are still live — another form of undersynchronization.

**`mma.commit`.** Hopper WGMMA overlaps issue and accumulation; `mma.commit` is the fence that completes one K-slab's contribution to `mc` before that stage's shared buffer may be logically reused. Treat it as required glue between "done with this stage" and `trigger empty`.

## Credit Flow for One Stage

The diagram matches the code: bootstrap grants empty credits; the producer waits on `empty`, fills, signals `full`; the consumer waits on `full`, computes, signals `empty`. When `iv_k` wraps modulo `MATMUL_STAGES`, the same physical stage re-enters the cycle — the ring is safe because `wait`/`trigger` serialize access, not because modulo arithmetic is magic.

![Event credit flow for one pipeline stage](../assets/images/ch06/fig2_event_credit_flow_dark.png#only-dark)
![Event credit flow for one pipeline stage](../assets/images/ch06/fig2_event_credit_flow_light.png#only-light)

If something looks wrong after you edit a pipeline, verify **event order and trip counts** before you chase MMA layout: producer and consumer must use the same `cdiv(K, MATMUL_TILE_K)` loop, and too few stages shifts pressure to `wait full` when the consumer outruns the producer.

## New Syntax

| Syntax | Meaning |
|--------|---------|
| `shared event name[N]` | Declare N named synchronization events in shared scope |
| `wait event` | Block until `event` has been signaled |
| `trigger event` | Signal `event`, waking any waiters |
| `dma.copy.async src => dst` | Non-blocking copy (returns immediately) |
| `dma.any` | Placeholder future (no transfer in flight yet) |
| `swap(f0, f1)` | Exchange two future handles without copying data |
| `rotate(f0, f1, f2)` | Cycle three future handles |
| `with tile_k in N { ... }` | Scoped tile axis binding with extent N |
| `foreach tile_k(1:)` | Iterate starting from index 1 |
| `mma.commit` | Fence between pipeline stages for WGMMA |
| `__co__ auto fn(...)` | Return type inferred from `return` statement |

## Summary

| Idea | Role |
|------|------|
| Data races | Unsynchronized overlap lets loads clobber MMA operands or read partial tiles. |
| CUDA-style fixes | Barriers, atomics, and manual event wiring work but scale poorly in complexity. |
| `swap` / `rotate` | Rotate **futures** so double- or multi-buffering stays explicit in one program. |
| `shared event` | Coordinate **different** warpgroups with `wait` / `trigger` and a credit discipline. |
| Bootstrap `empty` | Required so the producer's first `wait empty` can succeed. |
| `mma.commit` | Separates completed math for one K-slab from reuse of shared staging. |

The pipeline is now safe for split roles. The [next chapter](ch07-advanced-movement.md) moves on to hardware-accelerated **TMA**, **swizzled** shared layouts, and **`view` / `from`** for irregular access — the same synchronization ideas, with richer movement primitives underneath.
