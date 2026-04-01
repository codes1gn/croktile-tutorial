# Tensor cores: the `mma` operations

Chapter 3's tiled matmul computes each output element with scalar multiplies and adds — one product at a time, accumulated through a `foreach k` loop. That is how a CPU would do it. Modern GPUs have dedicated hardware called **tensor cores** that multiply a small matrix tile (commonly 16×16×16 for FP16) in a single macro-operation, producing orders of magnitude more throughput than scalar FMA on the same silicon.

Croktile exposes tensor cores through four operations that form a fixed lifecycle: **fill**, **load**, **multiply**, **store**. This chapter replaces the scalar inner loop from Chapter 3 with that lifecycle, first on SM86 (Ampere — one warp per MMA) and then on SM90 (Hopper — warpgroup WGMMA).

## The MMA Lifecycle

Every tensor-core matmul follows the same rhythm:

1. **`mma.fill 0.0`** — create a register-resident accumulator tile, initialized to zero.
2. **`mma.load`** — load an operand tile from shared memory into MMA operand registers.
3. **`mma.row.row mc, ma, mb`** — multiply operand tiles and accumulate: C += A × B.
4. **`mma.store mc, output_s`** — write the accumulator back to shared memory.

You loop steps 2–3 over K (loading the next K-slice of A and B each iteration, accumulating into the same `mc`), then do step 4 once to flush the result. The names `mc`, `ma`, `mb` are opaque register tiles — you never declare their sizes or lane mappings. The compiler handles that based on the target architecture.

## SM86 (Ampere): One Warp, One MMA Tile

On SM86, tensor-core MMA is scoped to a **single warp** (32 threads). In Croktile, that corresponds to `parallel ... : group` — one cooperative thread group the size of one warp.

Here is a complete FP16 matmul kernel for SM86. The tile sizes match the Croktile benchmark defaults: all `MATMUL_*` constants are 16, so one block tile equals one MMA tile along M and N.

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_TILE_N)] : block {
    shared f16 [MATMUL_TILE_M, MATMUL_TILE_N] output_s;

    parallel {warp_m, warp_n} by [cdiv(MATMUL_TILE_M, MATMUL_MMA_M), cdiv(MATMUL_TILE_N, MATMUL_MMA_N)] : group {
      mc = mma.fill 0.0;

      foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
        lhs_load_s = dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => shared;
        rhs_load_s = dma.copy rhs.subspan(MATMUL_TILE_N, MATMUL_TILE_K).at(block_n, iv_k) => shared;

        foreach iv_warp_k in [cdiv(MATMUL_TILE_K, MATMUL_MMA_K)] {
          ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k);
          mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k);
          mma.row.row mc, ma, mb;
        }
      }

      mma.store mc, output_s.subspan(MATMUL_MMA_M, MATMUL_MMA_N).at(warp_m, warp_n);
    }

    dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_TILE_N).at(block_m, block_n);
  }
}
```

Reading from the outside in:

**Grid and `void` return.** The function returns `void` — results are written in-place to the `output` parameter instead of being returned. This is the standard pattern for GPU kernels that accept destination pointers. `cdiv(M, MATMUL_TILE_M)` is ceiling division — the number of tiles along M, rounding up for partial tiles.

**Block-level structure.** Each block owns a `MATMUL_TILE_M × MATMUL_TILE_N` region of the output. `output_s` is shared memory staging for the result tile. Indices `block_m` and `block_n` select which region.

**Warp-level MMA.** Inside the block, `parallel {warp_m, warp_n} ... : group` enumerates warps. With tile sizes all equal to 16, the extents are 1×1 — a single warp handles the entire block tile. If you widened the block tile to 32×32 with MMA still 16×16, you would have 2×2 = 4 warps, each with its own accumulator.

**The K loop.** For each K-tile, `dma.copy` stages A and B strips into shared memory. The `subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k)` syntax creates a view with explicit tile extents and selects the tile at position `(block_m, iv_k)` — the same idea as `chunkat`, but with the tile shape spelled out. Chapter 7 covers `subspan` vs `chunkat` in more detail.

**Loading MMA operands.** `ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k)` loads the warp's A-operand tile from the shared-memory staging buffer into MMA registers. The `chunkat` selects the M × K slice for this warp and this inner K step.

**The multiply-accumulate.** `mma.row.row mc, ma, mb` is the tensor-core instruction. The `row.row` suffix states the **layout contract**: both A and B operands are interpreted as row-major. Choosing the wrong variant is not a performance hint — it is a correctness bug. The hardware interprets register bits differently depending on this choice.

**Store.** After the K loop, `mma.store mc, output_s.subspan(...).at(warp_m, warp_n)` writes the warp's accumulated tile from register into its sub-rectangle of shared memory. Then `dma.copy output_s => output.subspan(...).at(block_m, block_n)` copies the full block tile to global memory.

## SM90 (Hopper): WGMMA and Warp Groups

Hopper introduces **Warpgroup Matrix Multiply-Accumulate (WGMMA)**: the same C += A × B idea, but issued cooperatively by **four warps** (128 threads). In Croktile, that wider cooperation appears as `: group-4` instead of `: group`.

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;

  mc = mma.fill.f16 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
      parallel p by 1 : group-4 {
        ma = mma.load lhs_load_s.chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
  mma.store mc, output_s;
  dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

The lifecycle is identical: fill, load, multiply, store. What changes:

| Aspect | SM86 (Ampere) | SM90 (Hopper) |
|--------|---------------|---------------|
| Thread scope | One warp — `: group` | Four warps — `: group-4` |
| Accumulator init | `mma.fill 0.0` | `mma.fill.f16 0.0f` (precision suffix) |
| Global → shared | `dma.copy` | Same (or TMA — see Chapter 7) |
| Core math | `mma.row.row mc, ma, mb` | Same mnemonic, wider hardware |
| Store | `mma.store` into per-warp tile | `mma.store` into per-warpgroup tile |

The underscore in `chunkat(_, iv_warp)` means "no tiling on the first dimension — keep its full extent." Only K is subdivided for each MMA slice; the full M (or N) side is already in shared memory for this block.

**`mma.fill.f16` vs `mma.fill 0.0`.** On Hopper, you sometimes want to specify the accumulator precision explicitly — `.f16` for FP16, `.f32` for FP32. A common pattern is FP16 operands with FP32 accumulators for numerical stability on large K. The SM86 version uses the generic `mma.fill 0.0` which infers the type.

## Multi-Warpgroup MMA

Chapter 3 introduced `parallel p1 by 2 : group-4` — two warpgroups in one block. Here is how that works with MMA: both groups share the same B tile but load different rows of A:

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_TILE_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
  shared f16 [MATMUL_TILE_M, MATMUL_WARP_N] output_s;

  mc = mma.fill.f32 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    parallel p1 by 2 : group-4 {
      foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
        ma = mma.load lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(p1, 0).chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  parallel p1 by 2 : group-4 {
    mma.store mc, output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0);
  }
  dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

With `MATMUL_TILE_M = 128` and `MATMUL_WARP_M = 64`, the block tile is 128 rows tall and split between two warpgroups of 64 rows each. `p1` selects which half: `lhs_load_s.subspan(MATMUL_WARP_M, ...).at(p1, 0)` gives warpgroup 0 the top 64 rows and warpgroup 1 the bottom 64. Both read the same `rhs_load_s` — the reuse win.

The store mirrors the load: `output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0)` ensures each warpgroup writes to its own half of the output staging buffer.

## What Croktile Handles For You

In raw CUDA, tensor-core programming means declaring `wmma::fragment` types with specific shapes, calling `load_matrix_sync`, `mma_sync`, and `store_matrix_sync`, and manually tracking row-major vs column-major variants and `ldmatrix` staging. Croktile pushes that below the surface. `mc`, `ma`, and `mb` are logical MMA tiles; the compiler maps them to the correct register layouts for your target.

You still must choose: consistent layouts (`mma.row.row` must match how your data is actually stored), tile sizes that divide the hardware MMA geometry (16 for this FP16 path on SM86), and a thread hierarchy (`: group` vs `: group-4`) that matches the ISA. Croktile does not remove those constraints — it makes them readable and keeps the register mapping out of your way.

## Debugging MMA Kernels

When you get a wrong result, suspect **layout first** (row vs column major, and whether `rhs` is `[N, K]` vs `[K, N]`), then **indexing** (which `block_m`, `block_n`, and K slice you attached with `.at` / `chunkat`), then **async ordering** if you introduced asynchronous copies. The common mistake is mislabeling `mma.row.row` when your data is actually column-major, or using `chunkat` indices that do not align with the MMA tile geometry.

## New Syntax

| Syntax | Meaning |
|--------|---------|
| `mc = mma.fill 0.0` | Initialize an MMA accumulator tile to zero |
| `ma = mma.load src.chunkat(...)` | Load operand tile from shared into MMA registers |
| `mma.row.row mc, ma, mb` | C += A × B on tensor cores (row-major operands) |
| `mma.store mc, dst` | Write accumulator tile from registers to shared |
| `mma.fill.f16 0.0f` | Accumulator with explicit FP16 precision |
| `mma.fill.f32 0.0f` | Accumulator with FP32 precision (for mixed-precision) |
| `cdiv(a, b)` | Ceiling division: number of tiles rounding up |
| `__co__ void fn(...)` | Kernel that writes results in-place (no return value) |
| `subspan(M, K).at(i, j)` | View with explicit tile extents, selected by index |
| `chunkat(_, iv_warp)` | `_` wildcard: no tiling on that dimension |

The inner loop is now hardware-accelerated, but loads and compute still take turns — while tensor cores multiply, the memory system idles, and vice versa. The [next chapter](ch05-branch-control.md) introduces warp specialization and conditional control so different threads can play different roles: one group loading data while another computes.
