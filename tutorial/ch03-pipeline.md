# Overlapping Compute and DMA: Pipeline Patterns

In the previous chapter you expressed matrix multiplication with explicit DMA copies: each K-tile of the operands is brought into fast memory, multiplied into the accumulator, and then the next tile is fetched. That version is easy to read and correct, but it leaves a familiar hole in the schedule. While the GPU waits for the next DMA to finish, the arithmetic units sit idle; while the GPU is busy multiplying, the memory subsystem could be prefetching the *following* tile. This chapter shows how Choreo expresses **software pipelining** so data movement and computation overlap, using the same matmul example with **double buffering** in shared memory.

## The Problem: Sequential DMA and Compute

Picture the inner K-loop from a naive tiled matmul expressed with DMA. For each tile index along K you:

1. Issue copies for the A-tile and B-tile.
2. Wait until those copies complete (implicit in how you use the futures).
3. Run the dot-product updates for that tile.

Steps 2 and 3 cannot overlap with the *next* tile's step 1 if you only hold one staging buffer: you would overwrite data that is still being read. So the timeline looks like a staircase: copy, compute, copy, compute, with one side of the machine always idle.

**Double buffering** fixes that by keeping two slots: **buffer 0** holds the tile you are computing on *now*, while **buffer 1** is filled with the *next* tile in parallel. When a stage finishes, you **swap** the roles of the two buffers and repeat. Choreo makes this pattern explicit with named DMA futures, a prologue, a steady-state loop, and an epilogue—plus a few syntax pieces you have not seen yet.

## The Idea in One Diagram

Think of K as a sequence of tile indices `0, 1, …, T-1`.

- **Prologue:** Start DMA loads for tile `0` into buffer 0 (no compute yet—nothing valid to multiply).
- **Steady state:** For each `t` from `1` to `T-1`, start loads for tile `t` into buffer 1 while computing on buffer 0, then swap so buffer 1 becomes "current" and buffer 0 becomes "next."
- **Epilogue:** After the last swap, buffer 0 holds tile `T-1`; run the final multiply-accumulate on it.

The code below follows exactly that story. It also places tiles in **shared** memory so an entire thread block can reuse the same staged data—natural for a block-wide matmul tile.

## Full Example: Pipelined Matmul

The following program matches the Choreo end-to-end test `matmul-pipelined-2.co`: square-ish block and thread tiling, `K = 256`, and a K-tile size of `16`. The interesting part is the `with tile_k` region and the `foreach tile_k(1:)` loop.

```choreo
#define M 128
#define N 256
#define K 256

__co__ auto matmul(s32 [M, K] lhs, s32 [K, N] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel {px, py} by [8, 16] : block
    parallel {qx, qy} by [16, 16] : thread {
    with tile_k in 16 {
      lf0 = dma.copy lhs.chunkat(px, tile_k) => shared;
      rf0 = dma.copy rhs.chunkat(tile_k, py) => shared;

      lf1 = dma.any;
      rf1 = dma.any;

      foreach tile_k(1:) {
        lf1 = dma.copy lhs.chunkat(px, tile_k) => shared;
        rf1 = dma.copy rhs.chunkat(tile_k, py) => shared;

        foreach k in [256 / #tile_k]
          output.at(px#qx, py#qy) += lf0.data.at(qx, k) * rf0.data.at(k, qy);

        swap(lf0, lf1);
        swap(rf0, rf1);
      }

      foreach k in [256 / #tile_k]
        output.at(px#qx, py#qy) += lf0.data.at(qx, k) * rf0.data.at(k, qy);
    }
  }

  return output;
}

int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(M, K);
  auto rhs = choreo::make_spandata<choreo::s32>(K, N);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);
  auto res = matmul(lhs.view(), rhs.view());
  // verification against a scalar reference loop...
  std::cout << "Test Passed\n" << std::endl;
}
```

The host side is unchanged in spirit from earlier chapters: allocate `spandata`, fill inputs, call `matmul` with `.view()`, and check results. The new skills are all in the `__co__` body.

**Running the bundled test.** The Choreo tree ships this kernel as `tests/gpu/end2end/matmul-pipelined-2.co`. Your local checkout may use a different driver command, but the idea is the same as in Chapter 1: invoke the compiler on the `.co` file, then run the generated host binary so it prints `Test Passed` after the reference check. Pipelining does not change the observable result—only the schedule under the hood—so the verification loop in `main` stays a straightforward triply nested scalar matmul against `res`.

## Parallel Granularity: `: block` and `: thread`

Two nested `parallel` constructs carve the output matrix:

```choreo
parallel {px, py} by [8, 16] : block
  parallel {qx, qy} by [16, 16] : thread {
```

The annotations **`: block`** and **`: thread`** tell Choreo how each loop maps to the GPU hierarchy. The outer loop is a 8×16 grid of **thread blocks**; the inner loop is a 16×16 arrangement of **threads** within each block. Together they cover the logical output indices: expressions like `px#qx` and `py#qy` combine block indices with in-block thread indices to address the full output element this thread owns. You can read `#` as "compose block and thread coordinates."

This structure matters for pipelining because **shared memory is scoped to a block**: one pair `(px, py)` corresponds to one block, and all threads in that block cooperate on the same double-buffered tiles in shared memory.

## Binding the Tile Size: `with tile_k in 16`

The line:

```choreo
with tile_k in 16 {
```

does two jobs at once. It opens a scoped region (like the `with index in [...]` you saw earlier), and it **binds the name `tile_k` to the tiling factor 16**. Inside the block, `tile_k` is not just a number—it is the **chunk index** for `chunkat` along the K dimension, and Choreo can relate the extent of each chunk to that constant.

The inner arithmetic uses `256 / #tile_k` for the number of elements along K inside a tile: `#tile_k` turns the bound symbol into a compile-time numeric value for shape expressions. That keeps the inner `foreach k` loop consistent with the physical tile size without scattering literal `16` everywhere.

## Staging the First Tile (Prologue)

Immediately inside `with tile_k`:

```choreo
lf0 = dma.copy lhs.chunkat(px, tile_k) => shared;
rf0 = dma.copy rhs.chunkat(tile_k, py) => shared;
```

These are the **first** DMA operations: they start loading the K-tile indexed `0` (the initial value of the tile iterator) into two futures, `lf0` and `rf0`. The destination **`=> shared`** means the copied data lands in **shared memory** visible to all threads in the block, not in per-thread **local** storage.

Why shared? For matmul, each output element needs a row fragment of A and a column fragment of B for the same K-tile. Those fragments are reused across threads in the block; shared memory is the usual place to stage that reuse. Local memory would still be possible for some algorithms, but you would be trading different occupancy and access patterns. Here, shared is the right default mental model: **one double-buffered A-tile and one double-buffered B-tile per block**, wide enough for the thread tile to index into.

## Placeholder Futures: `dma.any`

The next lines introduce **uninitialized** futures:

```choreo
lf1 = dma.any;
rf1 = dma.any;
```

**`dma.any`** declares a slot that will later hold a real DMA operation. At this point there is no copy in flight for `lf1` / `rf1`; they exist so that the type system and the swap pattern have something to exchange with `lf0` / `rf0` on the first iteration.

If this feels like declaring uninitialized variables in C++, the analogy is fair—except these are **future handles** for asynchronous memory operations, and you assign real `dma.copy` operations to them before relying on their `.data`. The steady-state loop does exactly that.

## The Steady-State Loop: `foreach tile_k(1:)`

The heart of the pipeline is:

```choreo
foreach tile_k(1:) {
  lf1 = dma.copy lhs.chunkat(px, tile_k) => shared;
  rf1 = dma.copy rhs.chunkat(tile_k, py) => shared;

  foreach k in [256 / #tile_k]
    output.at(px#qx, py#qy) += lf0.data.at(qx, k) * rf0.data.at(k, qy);

  swap(lf0, lf1);
  swap(rf0, rf1);
}
```

**`foreach tile_k(1:)`** means: iterate the `tile_k` index **starting at 1**, running through the remaining tiles along K. The slice notation `(1:)` is "from 1 to the end," so tile index `0` is handled only by the prologue copies into `lf0`/`rf0`.

On each iteration:

1. **`lf1` / `rf1` assignments** start DMA for the **next** K-tile (`tile_k` is 1, 2, …) into shared memory.
2. The **`foreach k`** nest performs the multiply-accumulate using **`lf0` and `rf0`**—the buffers that were filled on the *previous* iteration (or by the prologue, when `tile_k == 1`).

So while the hardware works on bringing in the new tile through `lf1`/`rf1`, the program (conceptually) consumes the **previous** tile already resident behind `lf0`/`rf0`. That is the overlap you wanted.

3. **`swap(lf0, lf1)`** and **`swap(rf0, rf1)`** exchange the two futures **without** copying tensor data. After a swap, what used to be "next" becomes "current" for the following iteration's compute, and the old "current" slot is ready to be overwritten with a new `dma.copy` on the next trip through the loop.

By the time the loop exits, the roles have been rotated so that the **last** tile you need for the dot product sits behind whichever pair is currently `lf0`/`rf0`. That is why one more compute nest appears after the loop.

Notice the **ordering** inside the loop body: new copies are assigned to `lf1`/`rf1` *before* the compute reads `lf0`/`rf0`. That keeps the dependence story clear for both you and the compiler—the "previous" tile is read only after the "next" transfers are set up to land in the alternate buffer, and the swap at the bottom prepares the names for the next index.

## Epilogue: Drain the Last Buffer

When `foreach tile_k(1:)` finishes, there is no additional tile to prefetch, but you still owe one multiply-accumulate pass for the final staged data:

```choreo
foreach k in [256 / #tile_k]
  output.at(px#qx, py#qy) += lf0.data.at(qx, k) * rf0.data.at(k, qy);
```

This is the mirror of the prologue: you **compute once more** without issuing another DMA. If you trace a small example with three K-tiles, you will see exactly two swaps and three compute phases—prologue load, two overlapped iterations, epilogue compute.

## Tracing the Schedule for `K = 256`, `tile_k = 16`

With a K extent of 256 and tiles of length 16, there are `256 / 16 = 16` tiles along the contraction. Index them `0 … 15`.

The prologue kicks off loads for tile **0** into `lf0` and `rf0`. Nothing is multiplied yet—the hardware is still filling those buffers.

The loop `foreach tile_k(1:)` runs for `tile_k ∈ {1,…,15}`—fifteen iterations. On iteration `t` (where `t` is the value of `tile_k`):

- The assignments to `lf1` and `rf1` start loads for tile **`t`**.
- The inner `foreach k` uses **`lf0` / `rf0`**, which still describe tile **`t - 1`** from the point of view of the *previous* assignment (or the prologue when `t == 1`).
- After the multiply-accumulate, **`swap`** exchanges names so the futures that now hold tile `t` become the new `lf0`/`rf0`, and the slots you will overwrite on the next iteration become `lf1`/`rf1`.

After the fifteenth steady-state iteration, `lf0`/`rf0` refer to the buffers containing tile **15**. No tile **16** exists, so the code does not issue another DMA; the epilogue runs one last `foreach k` over those buffers and completes the dot product.

Counting stages: one prologue, fifteen overlapped iterations, one epilogue—sixteen compute phases total, matching sixteen K-tiles. The pattern scales the same way for any `K` divisible by the tile size: the number of steady-state iterations is `(K / tile_k) - 1`.

## What `swap` Does (and Does Not Do)

**`swap(lf0, lf1)`** is about **names and futures**, not about memcpy. After a swap, the identifier `lf0` refers to whatever asynchronous operation and backing buffer `lf1` referred to before, and vice versa. The data already staged in shared memory stays where the hardware put it; only the Choreo-level handles rotate.

That distinction matters when you read the loop: you are not "moving" tensors between variables—you are **re-pointing** the program at the correct in-flight copy and the correct `.data` view for the next iteration. The same idea appears in hand-written CUDA when programmers toggle between two `__shared__` arrays with a `^ 1` index or a boolean phase variable; Choreo makes the intent visible at the language level.

Because `dma.any` creates an inert placeholder, the **first** steady-state iteration must assign real copies to `lf1`/`rf1` before any `.data` read from those names in later code paths. The structure of the listing guarantees that: placeholders appear only once, immediately before the loop.

## Generalizing Beyond Fixed `256`

The example hard-codes `256` in `[256 / #tile_k]` to match the test file's `K`. In your own kernels, you will usually write that divisor in terms of the same symbolic `K` you used in the tensor shapes—so the inner loop's trip count always tracks the actual contraction length divided by the tile size. The pipeline structure does not change: prologue, `foreach tile_k(1:)`, epilogue, with `#tile_k` still supplying the compile-time tile length for the `foreach k` bounds.

If `K` were not an exact multiple of the tile size, you would need a strategy for a partial final tile (padding, a separate tail loop, or a different tile binding). The tutorial stays on the clean divisibility case so the overlap story stays front and center.

## Return Type Inference: `auto`

The kernel is declared as:

```choreo
__co__ auto matmul(s32 [M, K] lhs, s32 [K, N] rhs) {
```

Using **`auto`** as the return type tells Choreo to infer the result tensor type from `return output;`. Here `output` has shape `[lhs.span(0), rhs.span(1)]`, i.e. `[M, N]` for these fixed `M`, `N`. Explicit `s32 [M, N] matmul(...)` would also work; `auto` keeps the header aligned with shape expressions that depend on spans and keeps refactors less noisy if you generalize dimensions later.

## Shared vs Local: When to Prefer Which

**Local** (`=> local`) is closest to registers or thread-private scratch: great when each thread's working set is independent and you do not need cross-thread reuse within the block.

**Shared** (`=> shared`) is block-scoped and fast when threads collaborate on the same tile—classic for matmul, reductions, and stencils.

Pipelining does not *require* shared memory, but double buffering a **single** logical tile that many threads read almost always points to shared staging. The DMA futures (`lf0`, `rf1`, …) track **which** asynchronous transfer owns which shared buffer; the `swap` keeps that bookkeeping straight without duplicating tensor storage in the source code.

A useful rule of thumb: if every thread in the block reads overlapping regions of the same staged tile, **`=> shared`** is the first choice. If each thread consumes a disjoint piece of memory with no cross-thread reuse, **`=> local`** keeps data closer to the lane and can simplify the dependency picture—at the cost of duplicating loads or using a different decomposition. Matmul's reuse pattern is the textbook case for shared staging plus a 2-wide pipeline.

## How `chunkat` Interacts With the Pipeline

The expressions `lhs.chunkat(px, tile_k)` and `rhs.chunkat(tile_k, py)` select the **block's** slice of the operands for the current K-tile. The block indices `px` and `py` pin which output "tile" of the full matrices this CTA is responsible for; `tile_k` walks the shared contraction dimension.

Double buffering does not change **what** is copied—only **when** copies overlap with compute. Each `dma.copy` still describes the same logical tile of global memory; the futures merely alternate **which shared buffer** receives the next tile. That is why the `chunkat` arguments inside the steady-state loop match the prologue apart from `tile_k` advancing each time.

## Putting It Together

You can read the whole `with tile_k` region as a structured pipeline:

| Phase | What happens |
|--------|----------------|
| Prologue | `dma.copy` tile 0 → `lf0`/`rf0` |
| Steady state | For `tile_k = 1…`, `dma.copy` into `lf1`/`rf1` while computing on `lf0`/`rf0`, then `swap` |
| Epilogue | Compute on the final `lf0`/`rf0` after the last swap |

The new syntax—**`with tile_k in 16`**, **`foreach tile_k(1:)`**, **`dma.any`**, **`swap`**, **`=> shared`**, **`: block` / `: thread`**, and **`auto`**—exists to make that schedule explicit and checkable. The compiler can see which futures are live across iterations and align generated code with the overlap you intended.

## What to Take Away

Sequential DMA-then-compute leaves either memory or math idle; pipelining hides transfer latency behind useful work. Double buffering needs two names for two buffers, and `swap` rotates those names without moving data. The placeholder future `dma.any` lets the swap pattern type-check from the very first iteration, while `foreach tile_k(1:)` separates the prologue (tile 0 only loaded) from the steady state (load next, compute previous). Staging tiles with `=> shared` keeps them visible to the whole block—a natural fit for tiled matmul.

In the next chapter you will push data movement further toward hardware-specific fast paths (TMA and swizzled layouts). The mental model you built here—prologue, steady state, epilogue, and explicit futures—carries over directly: you are still describing **who** moves **which** tile **when**, only with richer primitives underneath.
