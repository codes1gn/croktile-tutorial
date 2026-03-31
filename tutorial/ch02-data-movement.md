# Data Movement Basics: Moving Data Blocks as a Whole

In Chapter 1, you added two arrays element by element. The tileflow program carved the tensors into chunks, copied each chunk into local memory with `dma.copy`, called a tiny CUDA kernel, and wrote the result back. That pattern is a perfect introduction to Choreo because the arithmetic is trivial: every output element depends only on the matching input elements in the same place.

Matrix multiplication is different. Each output element is a dot product along an entire *inner* dimension, so a naïve kernel touches a lot of global data unless you reorganize how work and memory line up. This chapter uses plain matrix multiply — first without explicit DMA, then with `dma.copy` and tiling — to show how Choreo expresses **element-level indexing**, **tile index composition**, and **nested parallelism**. By the end, you will have a mental model for why GPUs move data in blocks and how Choreo makes that explicit.

## The Scalar Matmul: Parallelism Without DMA

We start with the smallest complete matmul in the Choreo test suite: no DMA, no device kernel call in the tileflow — just `parallel`, `foreach`, and arithmetic on spanned tensors. It is not how you would ship production GEMM, but it is the cleanest place to learn `.at()`, `#`, and `span`.

**Function shape and output allocation.**

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32[lhs.span(0), rhs.span(1)] output;
  // ...
  return output;
}
```

The signature says the function returns a `128 × 256` matrix of 32-bit integers. The left-hand side `lhs` is `128 × 256`; the right-hand side `rhs` is `256 × 256`, so the classic contraction dimension is the shared `256`.

The line that allocates `output` is worth a slow read. Instead of repeating literal sizes, it ties the output shape to the *actual* extents of the operands:

- `lhs.span(0)` is the size of `lhs` along its first axis — here, `128` (rows of the result).
- `rhs.span(1)` is the size of `rhs` along its second axis — here, `256` (columns of the result).

You will see `lhs.span` without an index when the full shape is implied (as in Chapter 1). The indexed form `span(i)` picks out one dimension when you are building a tensor whose rank does not match the operand's rank, or when you want to be explicit about which axis you mean.

**Parallel over a 2D grid of tiles.**

```choreo
  parallel p by 16, q by 64 {
```

This declares two **parallel indices** at once: `p` and `q`. Think of them as a logical 2D launch grid. The first axis is split into 16 tiles; the second into 64. Choreo uses the comma form `p by 16, q by 64` when you want a Cartesian product of parallel dimensions without introducing extra braces.

A quick sanity check on the numbers never hurts. With `#p = 16` and `#q = 64`, the `foreach` bounds `128 / #p` and `256 / #q` evaluate to `8` and `4`. So each `(p, q)` tile owns an `8 × 4` block of output elements, and there are `16 × 64 = 1024` such tiles in parallel — enough to cover the full `128 × 256` result without gaps or overlaps, because `16 × 8 = 128` and `64 × 4 = 256`. When you write your own tilings, this kind of multiplication check is how you catch an off-by-one factor in the parallel grid before you ever run the program.

**The triple loop with named indices.**

```choreo
    foreach index = {m, n, k} in [128 / #p , 256 / #q, 256]
      output.at(m#p, n#q) += lhs.at(m#p, k) * rhs.at(k, n#q);
```

The `foreach` iterates three nested indices, but instead of three separate `in` clauses, it **names a tuple** of indices: `{m, n, k}`. The `index =` part binds that tuple to the loop variable `index` (handy if you later need the whole thing); the important part for reading the body is that `m`, `n`, and `k` are in scope inside the loop.

The trip counts are symbolic fractions of the parallel grid:

- `128 / #p` — how many row steps each `p`-tile owns along the result rows.
- `256 / #q` — how many column steps each `q`-tile owns.
- `256` — the full inner dimension (the contraction length).

Here `#p` and `#q` mean "the **extent** of the parallel index" — the number of tiles along that axis. So `128 / #p` is exactly the height of one tile in rows, and `256 / #q` is the width of one tile in columns. The inner `k` runs over the whole `K` dimension; this version does not yet break `K` into chunks in memory.

**`.at()` and the `#` composition operator.**

Until now, you mostly saw whole chunks (`lhs.chunkat(...)`) and device kernels that consumed flat pointers. For matmul we need **element-level** addressing on spanned values. That is what `.at(...)` is for: it takes one index per dimension and refers to a single scalar inside the tensor view.

The interesting indices are the ones that mix **intra-tile** offsets with **which tile** you are in:

- `m#p` — row index in the full matrix: offset `m` within tile column `p`.
- `n#q` — column index: offset `n` within tile column `q`.

Read `#` as **compose**: "local offset within tile" glued to "parallel tile id." So `m` runs from `0` up to (but not including) the per-tile row count, and `p` selects which horizontal band of rows you are responsible for; `m#p` is the global row. Likewise `n#q` is the global column.

The multiply-accumulate is the textbook formula:

\[
C_{ij} \mathrel{+}= \sum_k A_{ik} B_{kj}
\]

In Choreo, for a fixed `(m, n, k)` inside tile `(p, q)`:

- `lhs.at(m#p, k)` takes row `m#p` and column `k` from the left matrix.
- `rhs.at(k, n#q)` takes row `k` and column `n#q` from the right matrix.
- `output.at(m#p, n#q)` accumulates into the result cell at that global row and column.

No temporary buffers, no `dma.copy` — the compiler still has to implement loads and stores under the hood, but the **tileflow** you wrote is entirely in terms of global tensor indices.

**Host side: owning buffers with `make_spandata`.**

Chapter 1 used `choreo::make_spanview` to wrap existing C arrays. For tests that do not need a preallocated stack array, Choreo can allocate for you:

```choreo
int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(128, 256);
  auto rhs = choreo::make_spandata<choreo::s32>(256, 256);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);

  auto res = matmul(lhs.view(), rhs.view());
  // verification...
  std::cout << "Test Passed\n" << std::endl;
}
```

`make_spandata` builds a **dense owning buffer** with the given shape. It is the host-side counterpart to the `spanned_data` return type from Chapter 1: you can fill it, query `.shape()`, and pass a non-owning `view()` into a `__co__` function so the tileflow sees a normal spanned input.

The verification loop is ordinary C++: compare `res` against a reference triple loop over `lhs` and `rhs`. The point of this chapter is the tileflow; the host code stays boring on purpose.

**What this version teaches — and what it omits.**

You have now seen:

- Multi-dimensional `parallel` with comma-separated axes.
- `foreach` with **named destructuring** `index = {m, n, k}` and a single `in [..., ..., ...]` shape.
- `.at` for scalar indexing into spanned tensors.
- `#` for combining loop indices with parallel tile ids.
- `span(0)` / `span(1)` for extracting individual dimension sizes into a new tensor's shape.
- `make_spandata` for owned host tensors and `.view()` to borrow them into a kernel.

What you have *not* seen yet is any control over **where** those `lhs.at` / `rhs.at` reads land in the memory hierarchy. For large matrices on real hardware, reading every `k` step from far-away global memory inside the innermost loop is exactly what you want to fix next. That is where DMA and explicit local buffers enter the story.

**Complete scalar matmul (tileflow + host).**

For reference, here is the full scalar program as it appears in the Choreo tests (minus the `// REQUIRES` / `// RUN` harness lines). Read it top to bottom once: the tileflow fits on a handful of lines, and the host is almost entirely verification.

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32[lhs.span(0), rhs.span(1)] output;

  parallel p by 16, q by 64 {
    foreach index = {m, n, k} in [128 / #p , 256 / #q, 256]
      output.at(m#p, n#q) += lhs.at(m#p, k) * rhs.at(k, n#q);
  }
  return output;
}

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

Notice the symmetry between the tileflow and the reference: the triple `foreach` is the same `i, j, k` contraction as the host's `ref` loop, only written in terms of tiles and composed indices. That is a useful pattern when you are debugging — if the host reference passes and the tileflow fails, you can usually narrow the bug to indexing (`#`, `.at`, or trip counts) rather than to the arithmetic itself.

**Tracing a single output cell.**

Pick a concrete output position, say global row `37` and column `50`. In the scalar program, there is exactly one pair `(p, q)` and one pair `(m, n)` such that `m#p = 37` and `n#q = 50`. Because rows are split across `16` tiles of height `8`, tile `p = 37 / 8 = 4` and within-tile offset `m = 37 % 8 = 5`. Similarly `50 / 4 = 12` so `q = 12`, and `n = 50 % 4 = 2`. For that fixed `(p, q, m, n)`, the loop over `k` runs from `0` to `255`, and each iteration adds one product to the same `output.at(37, 50)`. Nothing in this version says *how* the hardware should cache those loads — it only states the dependence chain. The DMA version below will make the *reuse* of `lhs` and `rhs` values across `k` explicit by loading contiguous K-slabs into `local` first.

## The DMA Matmul: Blocks Along K

The file `matmul-dma.co` keeps the same mathematical contract — `128 × 256` times `256 × 256` → `128 × 256` — but changes the **orchestration**. Instead of indexing global `lhs` and `rhs` for every `(k, qx, qy)`, it loads **rectangular tiles** of `lhs` and `rhs` into local memory once per `tile_k`, then performs the arithmetic against those local views.

**Why move data in blocks?**

GPUs hide memory latency by keeping many threads busy and by exploiting fast **on-chip** storage (shared memory, registers, L2). If each multiply goes straight to device DRAM, you spend most of your time waiting on loads. A standard pattern is:

1. Cooperatively load a **tile** of `A` and a **tile** of `B` that are relevant to the same slice of `K`.
2. Compute partial dot products from those tiles while they are still hot in fast memory.
3. Advance `tile_k` and repeat until the full inner dimension is covered.

Choreo makes step 1 explicit with `dma.copy ... => local`. You still write the arithmetic with `.at`, but the tensors you read in the inner loop are the **loaded** locals, not the global spanned parameters.

**Top-level parallel grid over output tiles.**

```choreo
  parallel {px, py} by [8, 16] {
```

This is the **brace form** for multi-dimensional parallelism: one parallel index tuple `{px, py}` with a matching tile count vector `[8, 16]`. The grid is `8 × 16` tiles. Together with the inner `parallel {qx, qy} by [16, 16]`, the example carves the `128 × 256` output into a hierarchy: coarse tiles `(px, py)` and finer sub-tiles `(qx, qy)` inside each coarse tile.

If you are used to CUDA `blockIdx` / `threadIdx`, think of `px, py` as a logical block over the output matrix and `qx, qy` as a finer partition inside that block — Choreo keeps the algebra of indices explicit instead of folding everything into one flat thread id.

**Outer `foreach` over K-tiles.**

```choreo
    foreach {tile_k} in [16] {
      lhs_load = dma.copy lhs.chunkat(px, tile_k) => local;
      rhs_load = dma.copy rhs.chunkat(tile_k , py) => local;
```

There are `16` steps along the `tile_k` axis (the test's chosen factorization). For each step:

- `lhs.chunkat(px, tile_k)` selects the **rows** owned by `px` and the **K-range** owned by `tile_k`. Intuitively, it is the strip of `lhs` that participates in this K-tile for those output rows.
- `rhs.chunkat(tile_k, py)` selects the **K-range** for `tile_k` and the **columns** owned by `py` — the strip of `rhs` that lines up with the same contraction chunk.

Both copies target `local`, i.e. fast memory visible to the compute that will consume them. The assignments `lhs_load =` and `rhs_load =` bind **futures** (the DMA operations in flight). The nested `parallel` below shows how the inner compute waits on those implicitly and reads through `.data`.

Spacing in `chunkat(tile_k , py)` is only stylistic; Choreo ignores the extra space.

**Nested `parallel` and the inner `k` loop.**

```choreo
      parallel {qx, qy} by [16, 16] {
        foreach k in [256 / #tile_k] {
          output.at(px#qx, py#qy) += lhs_load.data.at(qx, k) * rhs_load.data.at(k, qy);
        }
      }
```

A second `parallel` nest subdivides each `(px, py)` tile. Indices `qx` and `qy` are composed with `px` and `py` exactly like `m#p` in the scalar version: `px#qx` is the global output row; `py#qy` is the global output column.

Inside, `k` runs over the **K extent of this loaded tile**. The trip count is `256 / #tile_k`: total inner size divided by the number of `tile_k` steps. Each `tile_k` iteration loads another slab of `K`; within that slab, `k` indexes the position inside the slab.

Crucially, the loads use **`lhs_load.data` and `rhs_load.data`**. The `dma.copy` future's `.data` member is the local spanned buffer you can index with `.at`. The global `lhs` / `rhs` parameters are not touched in the innermost statement — the hardware (via generated code) already brought the relevant rectangles into `local`.

Nested `parallel` expresses **two levels of parallelism**: orchestration across the output grid, and finer-grained work inside each tile. On a GPU target, the compiler maps these to warps, blocks, and shared memory in ways that depend on the backend; the tileflow you write stays at the level of *what* is parallel, not *which* hardware register holds which tid.

**Dimension arithmetic for the DMA nest.**

It helps to read the DMA example as a stack of factors. The output has shape `128 × 256`. The outer parallel grid is `[8, 16]` along `(px, py)`, so each coarse tile spans `128 / 8 = 16` rows and `256 / 16 = 16` columns of the result. The inner parallel grid is `[16, 16]` along `(qx, qy)`, which subdivides that `16 × 16` block into `16 × 16` sub-cells of size `1 × 1`. In other words, at this particular choice of numbers, each inner `(qx, qy)` pair is responsible for exactly one output element inside the coarse `(px, py)` tile — a common didactic layout because you can read `px#qx` and `py#qy` as "row and column in the global matrix" without an extra scaling factor in your head.

Along `K`, the outer `foreach {tile_k} in [16]` fixes `#tile_k = 16`, so each DMA slice covers `256 / 16 = 16` consecutive `k` values. The inner `foreach k in [256 / #tile_k]` therefore runs `k = 0 … 15` inside each loaded tile. For each `tile_k`, the `chunkat` expressions pull the matching `lhs` rows for `px` and `rhs` columns for `py` — aligned strips so that `lhs_load.data.at(qx, k) * rhs_load.data.at(k, qy)` is a legal multiply for every `k` in that slab. After all `tile_k` iterations complete, every output has seen contributions from the full `256`-long contraction, just as in the scalar program.

You do not have to memorize these exact factors. What you should internalize is the **pattern**: outer parallelism over output regions, an outer sequential or tiled loop over `K` that triggers DMA, inner parallelism over the fine structure of the tile, and an inner `k` loop whose length is the **current K-tile height**, not the full `256` every time.

**Complete DMA matmul (tileflow + host).**

Here is the full `matmul-dma.co` body for side-by-side comparison with the scalar version. The tileflow is longer, but the `main` function is still mostly setup and verification.

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32[lhs.span(0), rhs.span(1)] output;
  parallel {px, py} by [8, 16] {
    foreach {tile_k} in [16] {
      lhs_load = dma.copy lhs.chunkat(px, tile_k) => local;
      rhs_load = dma.copy rhs.chunkat(tile_k , py) => local;
      parallel {qx, qy} by [16, 16] {
        foreach k in [256 / #tile_k] {
          output.at(px#qx, py#qy) += lhs_load.data.at(qx, k) * rhs_load.data.at(k, qy);
        }
      }
    }
  }

  return output;
}

int main() {
  choreo::s32 a[128][256] = {0};
  choreo::s32 b[256][256] = {0};
  auto lhs_data = choreo::make_spanview<2, choreo::s32>((int*)a, {128, 256});
  auto rhs_data = choreo::make_spanview<2, choreo::s32>((int*)b, {256, 256});
  lhs_data.fill_random(-10, 10);
  rhs_data.fill_random(-10, 10);

  auto res = matmul(lhs_data, rhs_data);

  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      int ref = 0;
      for (size_t k = 0; k < lhs_data.shape()[1]; ++k)
        ref += a[i][k] * b[k][j];
      choreo::choreo_assert(ref == res[i][j], "values are not equal.");
    }

  std::cout << "Test Passed\n" << std::endl;
}
```

Comparing the two `__co__` functions line by line is instructive. Both allocate `output` the same way. Both ultimately implement `+= lhs * rhs` into `output.at(...)`. The scalar version states *which global elements* participate in each product; the DMA version states *which tiles were copied* before the product, then uses `.data` to read the copy. That is the whole conceptual step from Chapter 1's "copy a chunk, call a kernel" to "copy a chunk, stay in tileflow and index the chunk."

**Host program: views into stack arrays.**

The DMA variant's `main` in the test suite uses `make_spanview` with stack-backed storage:

```choreo
int main() {
  choreo::s32 a[128][256] = {0};
  choreo::s32 b[256][256] = {0};
  auto lhs_data = choreo::make_spanview<2, choreo::s32>((int*)a, {128, 256});
  auto rhs_data = choreo::make_spanview<2, choreo::s32>((int*)b, {256, 256});
  lhs_data.fill_random(-10, 10);
  rhs_data.fill_random(-10, 10);

  auto res = matmul(lhs_data, rhs_data);
  // verification against a and b...
}
```

That is the same pattern as Chapter 1: you own the memory (`a`, `b`), and the spanview tells Choreo the rank (`2`) and element type. You could equally allocate with `make_spandata` and pass `.view()` — the tileflow only cares that the arguments are spanned tensors with consistent shapes.

**Scalar vs DMA: same math, different movement story.**

Both programs implement the same multiply-accumulate. The scalar version is a **reference choreography**: easy to read, direct `.at` on globals. The DMA version is a **movement-aware** choreography: it states *which rectangles* of `lhs` and `rhs` live in `local` for each `tile_k`, and it nests parallelism so that sub-tiles `(qx, qy)` cooperate under each `(px, py)`.

When you optimize further (pipelines, TMA, tensor cores), you will keep this structure: outer loops over tiles and memory stages, inner loops over compute, explicit `.data` views after copies. Chapter 1 gave you DMA around a device kernel; this chapter shows DMA feeding **inline** tileflow arithmetic — same primitives, richer loop nest.

**How this extends Chapter 1.**

In Chapter 1, `dma.copy` always fed a `__device__` kernel through raw pointers (`lhs_load.data`, `rhs_load.data`, `l1_out`). The matmul DMA example skips the kernel call and uses the same `.data` handles directly in tileflow expressions. Nothing magic changed about DMA — you still get a future, you still read the payload through `.data` — only the *consumer* of that payload is different.

Similarly, Chapter 1 introduced `chunkat` on 3D tensors for element-wise add. Here, `chunkat` takes **two** indices because each operand is a matrix and the tiling axes are the output row tile, the output column tile, and the `K` tile. The rule of thumb is that `chunkat` always lists indices in the same order as the dimensions you are carving; the meaning of each slot follows from the shape of the tensor in the signature.

If you are tempted to merge the scalar and DMA styles — for example, using `.at` on globals for some loops and `.data.at` for others in one kernel — Choreo will let you express that, but readability usually suffers. Pick one level of abstraction per loop nest: either trust the compiler for global accesses, or stage explicitly with `dma.copy` and stay on `.data` until you leave the tile.

## New Syntax and Concepts (Chapter 2)

| Topic | What to remember |
|--------|------------------|
| `.at(i, j, ...)` | Element-level indexing on spanned tensors; one argument per dimension. |
| `m#p`, `py#qy` | `#` composes an offset within a tile with a parallel tile index to form a global index. |
| `lhs.span(0)`, `rhs.span(1)` | Read individual dimension sizes from an operand's shape when building another tensor. |
| `parallel p by A, q by B` | Comma-separated parallel axes (Cartesian product). |
| `parallel {px, py} by [8, 16]` | Tuple parallel indices with a matching list of tile counts. |
| Nested `parallel` | Outer grid over coarse tiles, inner grid over sub-tiles — hierarchy of parallelism. |
| `dma.copy x.chunkat(...) => local` | Explicit bulk move of a tensor tile into local memory. |
| `future.data.at(...)` | After `dma.copy`, use `.data` to get the local spanned buffer for `.at` indexing. |
| `make_spandata<T>(...)` | Host-side owning buffer; use `.view()` to pass into `__co__` functions. |
| `foreach index = {m, n, k} in [a, b, c]` | Named destructuring: multiple loop indices introduced in one `foreach` head. |
| `128 / #p`, `256 / #tile_k` | `#name` as the extent of parallel index or tile axis in symbolic trip counts. |

That table recaps what this chapter added relative to Chapter 1. If you are comparing to CUDA tutorials, the scalar program is like writing the arithmetic in one place with global pointers; the DMA program is like bringing shared-memory tiles into scope before the inner loops. Choreo's advantage is that both levels are **one language** — the same `.at` notation, the same `parallel` / `foreach` vocabulary — so you can refactor from one to the other without rewriting your host code or inventing ad-hoc macros.

## What's Next

You can now express matrix multiply with explicit tile loads and nested parallelism. The next chapter builds on this by **overlapping** DMA with compute — multiple buffers, pipelined `tile_k` steps, and the habit of thinking in terms of what is in flight while arithmetic runs. Bring the mental model from this chapter: global tensors, `chunkat`, `dma.copy => local`, and `.data` for the pieces you actually compute on.
