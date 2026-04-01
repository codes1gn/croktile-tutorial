# Synchronization: Pipelines, Events, and Double Buffering

Chapter 5 split the matmul into a producer (loads data) and a consumer (runs MMA). But the skeleton cheated: it assumed the consumer could read shared memory the instant the producer wrote it. In reality, you need **synchronization** — a way for the producer to say "this buffer is ready" and the consumer to wait until it is.

This chapter introduces the primitives that make pipelined execution safe: **events** for signaling between roles, **`swap`** and **`rotate`** for double- and multi-buffering, **`dma.copy.async`** for non-blocking transfers, and the **prologue / steady-state / epilogue** pattern that overlaps memory movement with compute.

## The Problem: Sequential Load-Then-Compute

Picture the K loop from a naive tiled matmul. Each iteration:

1. Copy the A-tile and B-tile into shared memory.
2. Wait for the copies to finish.
3. Run MMA on the loaded tiles.

Steps 2 and 3 cannot overlap with the **next** iteration's step 1 if you hold only one buffer — you would overwrite data that the MMA is still reading. So the timeline is a staircase: load, compute, load, compute, with one side of the machine always idle.

## Double Buffering with `swap`

The fix is two buffers. While the MMA reads buffer 0, the producer fills buffer 1 with the next tile. When the MMA finishes, you **swap** the buffers: what was "next" becomes "current," and the now-free slot is ready for the following load.

Croktile spells this with `dma.copy.async` (non-blocking copy), `dma.any` (placeholder future), `swap` (exchange future handles), and a three-phase loop.

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

Here is what each new construct does.

## `with tile_k in 16`

```choreo
with tile_k in 16 {
```

This opens a **scoped region** and binds `tile_k` as a tile axis with extent 16. Inside the block, `tile_k` serves as the chunk index for `chunkat` along the K dimension, and `#tile_k` gives its extent (16). Think of it as "within this scope, K is divided into 16 tiles."

## `dma.any`: Placeholder Futures

```choreo
lf1 = dma.any;
rf1 = dma.any;
```

`dma.any` creates a future that does not yet represent any real transfer. It exists so the type system has something to `swap` with on the first iteration. Before any code reads `lf1.data`, a real `dma.copy` will have been assigned to it.

## `foreach tile_k(1:)`: Sliced Iteration

```choreo
foreach tile_k(1:) {
```

The `(1:)` slice means "iterate `tile_k` starting at 1, through the remaining tiles." Tile 0 was already handled by the prologue — the initial loads into `lf0` and `rf0`. So the steady-state loop runs for tiles 1, 2, ..., 15.

## The Three Phases

**Prologue.** Load tile 0 into `lf0`/`rf0`. No compute yet — the buffers are being filled.

**Steady state.** For each subsequent tile: start loading into `lf1`/`rf1` (the "next" buffers), then compute on `lf0`/`rf0` (the "current" buffers that were filled on the previous iteration). After computing, `swap(lf0, lf1)` exchanges the handles — what was "next" becomes "current" for the following iteration.

**Epilogue.** After the last swap, `lf0`/`rf0` hold the final tile. One more compute pass drains it.

The ordering matters: new copies are assigned to `lf1`/`rf1` **before** the compute reads `lf0`/`rf0`. This keeps the dependence story clear: you never read from a buffer that is simultaneously being overwritten.

## `swap`: Names, Not Data

`swap(lf0, lf1)` exchanges **future handles**, not tensor data. After a swap, the name `lf0` refers to whatever asynchronous operation `lf1` referred to before, and vice versa. The data already staged in shared memory stays where the hardware put it; only the Croktile-level handles rotate.

In hand-written CUDA, the same idea appears as toggling between two `__shared__` arrays with a `^ 1` index or a boolean phase variable. Croktile makes the intent visible at the language level.

For triple buffering, use `rotate(f0, f1, f2)` instead of two `swap` calls — it cycles three handles in one operation.

## `auto` Return Type

The kernel signature uses `__co__ auto matmul(...)`: the `auto` return type tells Croktile to infer the result type from `return output`. This keeps the header aligned with shape expressions and avoids repeating literal dimensions.

## Events: Synchronization Between Roles

Double buffering with `swap` works when one set of threads does both loading and computing — they interleave steps on the same program counter. Warp specialization (Chapter 5) assigns loading and computing to **different** warpgroups. Those groups need a different coordination mechanism: **events**.

```choreo
shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
```

`shared event` declares named synchronization barriers in shared scope. `wait event_name` blocks until the event has been signaled; `trigger event_name` signals it. In the 1P1C matmul:

- `full[s]` means stage `s` has been filled by the producer — the consumer may read it.
- `empty[s]` means the consumer is done with stage `s` — the producer may overwrite it.

Here is the full 1P1C kernel with events:

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

**Ring indexing.** `stage = iv_k % MATMUL_STAGES` maps the unbounded K timeline onto a fixed number of physical slots — like double buffering generalized to N buffers. With `MATMUL_STAGES = 4`, the producer can run up to 4 K-tiles ahead of the consumer before it has to wait for a slot to be freed.

**Producer flow.** For each `iv_k`: wait until `empty[stage]` says the slot is free, copy tiles into that stage's region of `lhs_load_s` and `rhs_load_s`, then `trigger full[stage]` to tell the consumer the data is ready.

**Consumer flow.** Before the K loop, `trigger empty[s]` for all stages bootstraps credits — every slot starts logically empty, which prevents the producer from deadlocking on the first `wait empty`. Then for each `iv_k`: `wait full[stage]`, run MMA on that stage's data, `mma.commit`, then `trigger empty[stage]` to return the slot.

**`mma.commit`.** This marks a logical boundary after the MMA sequence for one K-slab. WGMMA on Hopper overlaps operand fetch, issue, and accumulation aggressively; `mma.commit` is the fence that says "fold this stage's partial products into `mc` before the shared buffer is reused." Treat it as mandatory glue between "done with this stage's math" and "signal empty."

## Credit Flow for One Stage

1. Consumer pre-triggers `empty[stage]` (bootstrap).
2. Producer passes `wait empty[stage]` — the slot is free.
3. Producer fills shared memory for this K-tile, then `trigger full[stage]`.
4. Consumer passes `wait full[stage]` — the data is ready.
5. Consumer runs MMA, commits, then `trigger empty[stage]` — the slot is free again.
6. When `iv_k` wraps around (modulo `MATMUL_STAGES`), the cycle repeats.

The ring is not magic — `wait`/`trigger` on `full`/`empty` are what make `iv_k % MATMUL_STAGES` safe.

## Shared vs Local for Staging

Chapter 2 used `=> local` for thread-private copies. Pipelining a tile that many threads read almost always points to `=> shared`. The DMA futures track which asynchronous transfer owns which shared buffer; the `swap` or ring index keeps the bookkeeping straight.

A useful rule: if every thread in the block reads overlapping regions of the same staged tile, `=> shared`. If each thread consumes a disjoint piece with no cross-thread reuse, `=> local` keeps data closer.

## Pitfalls

- **Deadlock.** Removing the initial `trigger empty[s]` loop leaves the producer's first `wait empty` waiting forever.
- **Undersynchronized reuse.** Dropping `mma.commit` or mis-ordering `trigger empty` relative to loads risks stale data.
- **Mismatched trip counts.** Producer and consumer must both use `foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)]`; asymmetric loops leak or orphan events.
- **Too few stages.** If the consumer is faster than the producer, you stall on `wait full`; more stages increase run-ahead (at the cost of shared memory).

When a pipeline edit breaks correctness, check **event order** before you suspect MMA layout — specialization bugs are usually synchronization bugs.

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

The pipeline is now safe: producer and consumer overlap through events, the ring index cycles shared buffers, and `mma.commit` keeps the accumulator coherent. The [next chapter](ch07-advanced-movement.md) pushes data movement further with hardware-accelerated TMA, swizzled shared-memory layouts, and `view`/`from` for irregular access patterns.
