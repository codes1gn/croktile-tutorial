# Synchronization in Practice: Pipelines, Buffers, and Events

Chapter 5 introduced three control-flow primitives: **`if`** for predicated execution, **`inthreads.async`** for structured concurrent regions, and **events** (`shared event` / `wait` / `trigger`) for inter-region signaling. This chapter puts them to work. You will see two progressively complex kernel patterns, each demonstrating how the primitives compose to solve real synchronization problems.

The first pattern — **double buffering with `swap`** — uses a single thread group that interleaves loading and computing within one program. No `inthreads.async`, no events: just two buffer handles and a rotation. The second pattern — **the full 1P1C event pipeline** — splits loading and computing into separate concurrent programs with `inthreads.async` and coordinates them with event arrays.

Both patterns solve the same underlying problem: **you cannot read a buffer while someone is writing to it**. They differ in how they structure the solution.

## Double buffering with `swap`

Give the K-loop **two** logical buffers. While the math drains buffer 0, DMA fills buffer 1 with the next tile. After the math step, **swap** the handles: what was "next" becomes "current," and the freed slot is ready for the following load.

Croqtile spells this with `dma.copy.async` (non-blocking copy), `dma.any` (a placeholder future), `swap` (exchange future handles), and a three-phase loop:

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

### The three phases

**Prologue.** Issue loads for tile 0 into `lf0`/`rf0`. No compute yet — the first tile must land before anything can multiply it.

**Steady state.** For each subsequent tile: start loads into `lf1`/`rf1`, compute on `lf0`/`rf0` from the previous iteration, then `swap` so names track the active buffers. New copies land in `lf1`/`rf1` **before** the compute reads `lf0`/`rf0`, so you never read a buffer being overwritten.

**Epilogue.** After the last swap, `lf0`/`rf0` hold the final tile; one more compute pass drains them.

### `swap`: names, not bytes

`swap(lf0, lf1)` exchanges **future handles** — the Croqtile-level names that refer to buffers. Shared-memory contents stay where the hardware placed them; only the names rotate. In CUDA, the same idiom is often a `^ 1` buffer index or a boolean phase variable. For triple buffering, `rotate(f0, f1, f2)` cycles three handles in one step.

### `with tile_k in 16`

Opens a scoped region and binds `tile_k` as a tile axis with extent 16. Inside the block, `tile_k` is the chunk index for `chunkat` along K, and `#tile_k` is 16.

### `dma.any`: placeholder futures

`dma.any` creates a future that does not yet represent a transfer. It gives the type system something to `swap` against on the first steady-state iteration. Before any use of `lf1.data`, a real `dma.copy` has been assigned.

### `foreach tile_k(1:)`: sliced iteration

`(1:)` means tile indices `1, 2, ...` through the end. Tile 0 was loaded in the prologue.

### `__co__ auto` return type

`__co__ auto matmul(...)` lets the compiler infer the return type from `return output`.

*This example uses `s32` with scalar accumulation — a simplified style to isolate the `swap` mechanism. The same pattern applies to FP16/MMA kernels from Chapter 4.*

## Why you cannot "just overlap" without events

The `swap` pattern works because one thread group controls both loading and computing — it knows the order. Warp specialization ([Chapter 5](ch05-branch-control.md)) puts them on **different** warpgroups with different program counters. They cannot share a `swap` schedule; they need a signaling mechanism.

The picture below contrasts a strict load-then-compute staircase with double-buffered overlap: same logical work, less idle time.

![Sequential vs double-buffered K-tile timelines (schematic)](../assets/images/ch06/fig1_pipeline_timeline_dark.png#only-dark)
![Sequential vs double-buffered K-tile timelines (schematic)](../assets/images/ch06/fig1_pipeline_timeline_light.png#only-light)

## The full 1P1C event pipeline

This kernel combines `inthreads.async` (from Chapter 5) with event arrays to build a complete multi-stage pipeline. The producer and consumer run as separate concurrent programs, coordinated entirely through `wait` / `trigger`:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s[MATMUL_STAGES];
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s[MATMUL_STAGES];
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {
      inthreads.async (p1 == 0) {
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait empty[stage];
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)
            => lhs_load_s[stage];
          dma.copy rhs.chunkat(block_n, iv_k)
            => rhs_load_s[stage];
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
            ma = mma.load lhs_load_s[stage].chunkat(_, iv_warp);
            mb = mma.load rhs_load_s[stage].chunkat(_, iv_warp);
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

### Walking through the kernel

**Ring index.** `stage = iv_k % MATMUL_STAGES` maps the unbounded K iteration to a fixed number of physical buffer slots — double buffering generalized to N buffers.

**Producer path.** For each `iv_k`, `wait empty[stage]` acquires a free slot. The `dma.copy` lines fill `lhs_load_s` / `rhs_load_s` at that stage. Then `trigger full[stage]` hands the slot to the consumer.

**Consumer bootstrap.** The loop `foreach {s} in [MATMUL_STAGES] { trigger empty[s]; }` runs **before** the K-loop so every stage starts with an `empty` credit. Without this, the producer blocks forever on its first `wait empty` — a deadlock.

**Consumer path.** Each `iv_k`: `wait full[stage]` blocks until the producer has filled that slot, then MMA over the tile, `mma.commit`, and `trigger empty[stage]` to release the slot for reuse.

**`mma.commit`.** Hopper WGMMA overlaps instruction issue and accumulation. `mma.commit` is the fence that completes one K-slab's contribution to `mc` before that stage's shared buffer may be reused. Omitting it risks reading stale data — the MMA might still be consuming operands when the producer overwrites the buffer.

### Credit flow for one stage

The diagram matches the code: bootstrap grants empty credits; the producer waits on `empty`, fills, signals `full`; the consumer waits on `full`, computes, signals `empty`. When `iv_k` wraps modulo `MATMUL_STAGES`, the same physical stage re-enters the cycle.

![Event credit flow for one pipeline stage](../assets/images/ch06/fig2_event_credit_flow_dark.png#only-dark)
![Event credit flow for one pipeline stage](../assets/images/ch06/fig2_event_credit_flow_light.png#only-light)

### Debugging tip

If something looks wrong after editing a pipeline, verify **event order and trip counts** before chasing MMA layout bugs: producer and consumer must use the same `cdiv(K, MATMUL_TILE_K)` loop bound, and too few stages shifts pressure to `wait full` when the consumer outruns the producer.

## New syntax

| Syntax | Meaning |
|--------|---------|
| `dma.copy.async src => dst` | Non-blocking copy (returns immediately) |
| `dma.any` | Placeholder future (no transfer in flight yet) |
| `swap(f0, f1)` | Exchange two future handles without copying data |
| `rotate(f0, f1, f2)` | Cycle three future handles |
| `with tile_k in N { ... }` | Scoped tile axis binding with extent N |
| `foreach tile_k(1:)` | Iterate starting from index 1 |
| `mma.commit` | Fence between pipeline stages for WGMMA |
| `__co__ auto fn(...)` | Return type inferred from `return` statement |

## Summary

| Pattern | Primitives used | When to use |
|---------|----------------|-------------|
| `swap` double buffering | `dma.copy`, `dma.any`, `swap`, `with` | Single thread group interleaving load and compute |
| 1P1C event pipeline | `inthreads.async`, `shared event[]`, `wait`, `trigger`, `mma.commit` | Separate producer/consumer warpgroups with multi-stage pipeline |

Both patterns achieve the same goal — overlapping memory and compute — but at different levels of complexity. Start with `swap` for simpler kernels; graduate to events when you need warp specialization.

The [next chapter](ch07-advanced-movement.md) moves on to hardware-accelerated **TMA**, **swizzled** shared layouts, and **`view` / `from`** for irregular access — the same synchronization patterns from this chapter, with richer data movement primitives underneath.
