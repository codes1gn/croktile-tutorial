# Parallelism: Mapping Work to Hardware

Chapters 1 and 2 used `parallel` to spread work across the GPU — but without telling Croktile *how* to organize those instances on the hardware. The compiler chose defaults, and for element-wise operations those defaults are fine. Matrix multiply is different: it has multiple tiling levels that map naturally to different levels of the GPU hierarchy, and getting the mapping right is the difference between toy throughput and real throughput.

This chapter introduces **space specifiers** — the annotations that map logical parallel axes to CUDA thread blocks, threads, warps, and warpgroups. By the end, you will have a tiled matrix multiply that uses `dma.copy => shared` and nested `parallel` to get data reuse across threads.

## `parallel` and `foreach` Revisited

Both appeared in earlier chapters, but it is worth being precise about what each one means:

- `parallel p by N` — creates `N` independent instances that execute concurrently. The compiler maps them to GPU execution units. Used when iterations are independent.
- `foreach i in [N]` — runs `N` iterations **sequentially** within the context that contains it. Used when later iterations depend on earlier ones (like a K-dimension reduction loop in a matmul).

Both support `#p` (extent) and `#` compose, and both can be multi-dimensional (`parallel {a, b} by [M, N]` / `foreach {i, j} in [M, N]`).

## Space Specifiers: Where Parallelism Lives

A GPU is not a flat bag of threads. It has a hierarchy: **thread blocks** (also called CTAs) that live on a streaming multiprocessor, **warps** of 32 threads that execute in lockstep, and on newer hardware, **warpgroups** of 128 threads (four warps) that cooperate on wide instructions.

Croktile lets you map your logical parallel dimensions to this hierarchy with **space specifiers** after the extent:

```choreo
parallel p by 16 : block     // 16 thread blocks
parallel q by 32 : thread    // 32 threads within a block
parallel w by 4  : group     // 4 warps (one per group)
parallel g by 2  : group-4   // 2 warpgroups of 4 warps each (128 threads per group)
```

When you write `parallel p by 16 : block`, you are telling Croktile: "create 16 instances of this body, and map each instance to a separate thread block on the GPU." When you write `parallel q by 32 : thread`, you are saying: "within whatever block we are already in, create 32 threads."

If you omit the specifier entirely — just `parallel p by 8` — Croktile picks a reasonable default. Explicit specifiers are how you control the mapping when performance matters.

The hierarchy matters for data sharing. **Shared memory** (`=> shared`) is visible to all threads in the same block but not across blocks. So if you want multiple threads to read the same DMA-loaded tile, you need those threads in the same block and the tile in shared memory. **Local memory** (`=> local`) is private to one thread. **Global memory** is visible everywhere but slow.

## Nested Parallelism

Real GPU programs nest these levels. A common pattern:

```choreo
parallel {px, py} by [8, 16] : block
  parallel {qx, qy} by [16, 16] : thread {
    // each (px, py) block has 16 × 16 = 256 threads
    // each thread is identified by (qx, qy) within its block
  }
```

The outer `parallel` creates an 8 × 16 = 128 block grid. The inner `parallel` creates 256 threads per block. Total: 128 × 256 = 32,768 threads. The braces `{px, py}` and `{qx, qy}` introduce **multi-dimensional** parallel indices in one declaration — shorthand for what would otherwise be two nested `parallel` lines.

## Matrix Multiply: Putting It Together

With `parallel`, `dma.copy`, `chunkat`, and `#`, a tiled matrix multiply is within reach. This is the first program in the tutorial that does something GPUs are famous for.

The plan: multiply a `[128, 256]` matrix by a `[256, 256]` matrix to get a `[128, 256]` result. The output is tiled into a grid of blocks, and within each block, threads cooperate to load K-strips of `lhs` and `rhs` into shared memory, then compute partial products.

**The scalar matmul (no DMA, `parallel` and `.at()` only).**

Start with `parallel` and direct global access, no explicit data movement:

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel p by 16, q by 64 {
    foreach index = {m, n, k} in [128 / #p, 256 / #q, 256]
      output.at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n);
  }

  return output;
}
```

This is dense — here is what each piece does.

**Output shape from operand dimensions.** The line `s32 [lhs.span(0), rhs.span(1)] output` builds the output shape from the inputs: `lhs.span(0)` is 128 (rows of the left matrix) and `rhs.span(1)` is 256 (columns of the right matrix). This is the `span(i)` syntax from Chapter 2 — pick one dimension instead of copying the whole shape.

**Multi-axis parallel.** `parallel p by 16, q by 64` declares two parallel indices at once with a comma. This creates a 16 × 64 = 1024-way parallel grid. Each `(p, q)` pair owns a tile of the output matrix.

**Named tuple destructuring.** `foreach index = {m, n, k} in [128 / #p, 256 / #q, 256]` introduces three nested loop indices — `m`, `n`, `k` — bound to a tuple called `index`. The trip counts are:

- `128 / #p = 128 / 16 = 8` rows per tile
- `256 / #q = 256 / 64 = 4` columns per tile
- `256` for the full contraction dimension

**Composed global indices.** `p#m` and `q#n` compose the tile index with the within-tile offset to form a global position (outer # inner, same convention as Chapter 2). `p` selects which of the 16 row-tiles, `m` runs 0..7 within that tile, so `p#m` covers the full 0..127 range across all tiles.

**The arithmetic.** `output.at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n)` is the textbook dot product: for each output element, sum products along the K dimension. Every `.at()` call here reads from global memory — which is where DMA comes in.

**Adding DMA: tiles in shared memory.**

The scalar version works but every multiply reads from global memory. Bringing the K-tiles into shared memory first allows all threads in a block to reuse them:

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel {px, py} by [8, 16] : block {
    foreach {tile_k} in [16] {
      lhs_load = dma.copy lhs.chunkat(px, tile_k) => shared;
      rhs_load = dma.copy rhs.chunkat(tile_k, py) => shared;

      parallel {qx, qy} by [16, 16] : thread {
        foreach k in [256 / #tile_k]
          output.at(px#qx, py#qy) += lhs_load.data.at(qx, k) * rhs_load.data.at(k, qy);
      }
    }
  }

  return output;
}
```

What changed:

1. The outer `parallel` now uses `: block` with brace-form indices `{px, py}`. An 8 × 16 grid of blocks covers the output.

2. A `foreach` over `tile_k` walks the K dimension in 16 steps. Each step copies one strip of `lhs` and one strip of `rhs` into `shared` memory with `dma.copy ... => shared`. The destination is `shared` instead of `local` because all threads in the block need to read the same tile.

3. An inner `parallel {qx, qy} by [16, 16] : thread` creates 256 threads within each block. Each thread owns one output element within the block's tile.

4. The arithmetic reads from `lhs_load.data` and `rhs_load.data` — the shared-memory copies — instead of from global `lhs` and `rhs`.

The composed indices work in two layers now: `px#qx` composes block index `px` with thread index `qx` to form the global row; `py#qy` does the same for columns.

**Dimension arithmetic.** With `[8, 16]` blocks, each block owns `128/8 = 16` rows and `256/16 = 16` columns. The inner `[16, 16]` threads subdivide that into one element per thread. Along K, 16 tiles of `256/16 = 16` elements each cover the full contraction. For each `tile_k`, `chunkat(px, tile_k)` selects the rows owned by block `px` and the K-range owned by `tile_k`; `chunkat(tile_k, py)` selects the K-range and the columns owned by block `py`.

**Host code.** The host side uses the same `make_spandata` / `.view()` pattern:

```choreo
int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(128, 256);
  auto rhs = choreo::make_spandata<choreo::s32>(256, 256);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);

  auto res = matmul(lhs.view(), rhs.view());

  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      int ref = 0;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += lhs[i][k] * rhs[k][j];
      choreo::choreo_assert(ref == res[i][j], "values are not equal.");
    }

  std::cout << "Test Passed\n" << std::endl;
}
```

Nothing here is new — `make_spandata` creates the buffers, `matmul(lhs.view(), rhs.view())` calls the Croktile function, and the triple loop verifies against a scalar reference. The host program stays boring so the Croktile function stays the star.

## Warp Groups and `: group-4`

On Hopper-class GPUs (compute capability 9.0+), NVIDIA introduced **warpgroup** cooperation: four warps (128 threads) issuing wide matrix instructions together. Croktile calls this `: group-4`:

```choreo
parallel p by 1 : group-4 {
  // body executes as a warpgroup of 128 threads
}
```

Even when the parallel count is 1 (one warpgroup instance), the `: group-4` annotation tells the compiler that this body's 128 threads form a cooperative unit for instructions like WGMMA. You will see this in Chapter 4 when we add tensor-core operations.

You can also launch **multiple** warpgroups per block:

```choreo
parallel p1 by 2 : group-4 {
  // two warpgroups, indexed by p1 = 0 and p1 = 1
}
```

This is how you scale a block's work along one dimension without adding more blocks — both warpgroups share the same shared memory but maintain independent accumulators. Chapter 4 will put this to work.

## How `shared` Enables Reuse

The DMA matmul uses `=> shared` instead of `=> local` for a reason: in the inner loop, every thread in the `[16, 16]` grid reads from the **same** `lhs_load.data` and `rhs_load.data`. Thread `(qx=3, qy=7)` reads `lhs_load.data.at(3, k)` — the same row that thread `(qx=3, qy=0)` reads. If each thread had its own private copy in `local`, the same data would be loaded 16 times (once per column thread). Shared memory stores it once for the whole block.

The rule of thumb: if multiple threads in a block need the same data, copy it into `shared`. If each thread's working set is independent, `local` keeps data closer and avoids shared-memory bank contention.

## Tracing One Output Element

Pick global position `(row=37, col=50)` in the output. Which block and thread own it?

Blocks partition 128 rows into 8 tiles of 16: `px = 37 / 16 = 2`, offset `qx = 37 % 16 = 5`. Columns partition 256 into 16 tiles of 16: `py = 50 / 16 = 3`, offset `qy = 50 % 16 = 2`. So block `(2, 3)`, thread `(5, 2)`.

For that thread, the K-loop runs 16 iterations. On iteration `tile_k = 0`, `dma.copy` loads `lhs` rows 32..47 and K columns 0..15 into shared `lhs_load`, and `rhs` K rows 0..15 and columns 48..63 into shared `rhs_load`. The inner `foreach k in [16]` accumulates `lhs_load.data.at(5, k) * rhs_load.data.at(k, 2)` for k = 0..15 into `output.at(37, 50)`. Then `tile_k = 1` loads the next K-strip, and so on. After all 16 iterations, `output.at(37, 50)` holds the complete dot product — same answer as the scalar reference.

## Summary of New Syntax

| Syntax | Meaning |
|--------|---------|
| `parallel p by N` | N-way parallelism, index `p` runs 0..N-1 concurrently |
| `parallel {px, py} by [M, N]` | Multi-dimensional parallel (Cartesian product) |
| `parallel p by N : block` | Map to CUDA thread blocks |
| `parallel q by N : thread` | Map to threads within a block |
| `parallel w by N : group` | Map to warps (32 threads each) |
| `parallel g by N : group-4` | Map to warpgroups (128 threads each) |
| `=> shared` | DMA destination: block-scoped shared memory |
| `foreach index = {m, n, k} in [a, b, c]` | Named tuple destructuring in a `foreach` |
| `parallel p by A, q by B` | Comma-separated parallel axes |

The tiled DMA matmul you built in this chapter is the structural backbone of every high-performance GPU kernel. The next chapter replaces the scalar `.at()` arithmetic in the inner loop with **tensor-core** operations — hardware-accelerated matrix multiply that processes a 16×16×16 tile in a single instruction.
