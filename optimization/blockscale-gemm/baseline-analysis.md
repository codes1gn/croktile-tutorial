# Baseline: Block Scaling and Where 397.9 TFLOPS Comes From

Before optimizing, we need to understand what block-scaled GEMM actually does, how the baseline kernel is wired, and what the first throughput numbers mean in context.

## Why Block Scaling Exists

FP8 E4M3 trades dynamic range and mantissa precision for half the operand footprint of FP16 and a path to high tensor-core throughput. A naive GEMM that simply casts to FP8 and accumulates in low precision loses information: values that differ greatly in magnitude within a long K reduction cannot all be represented faithfully after quantization.

Per-block scaling repairs this without reverting to FP16 weights. The idea is local:

1. Partition K into blocks (aligned with TILE_K = 128)
2. For each (row, block) on the left and (column, block) on the right, store a single FP32 scale factor
3. During the dot product, scale each block's contribution: `s_a × s_b × <ã, b̃>`
4. Accumulate in FP16 so the reduction stays useful

This is the same family of tricks used in MXFP8-style training and inference: FP8 for density, scales for fidelity. The verification tolerances reflect this contract — close numerically, not ULP-perfect.

## The Baseline Kernel

In Croktile, the fused path is expressed as `mma.row.row.scale` — the blockscale analogue of `mma.row.row` from [Chapter 4](../../tutorial/ch04-mma.md). Same tiling discipline, extra operands for scales.

The baseline kernel in `blockscale_gemm_dyn_sm90.co` follows the dense GEMM skeleton — TMA in, WGMMA-shaped loads, accumulator in registers — except the inner MMA consumes scale factors for the active K slice alongside operand fragments.

**One K-tile iteration (baseline control flow):**

1. **TMA** `lhs` and `rhs` subspans of shape (WARP_M, TILE_K) and (WARP_N, TILE_K) into shared memory with swizzle matching TILE_K
2. **Inner loop** over WARP_K chunks: MMA load fragments from SMEM, then `mma.row.row.scale` with views into `scale_lhs` and `scale_rhs` indexed by block_m, block_n, and iv_k
3. After all K tiles, **store** accumulator to SMEM and TMA out to global output

Every K iteration touches fresh FP8 data **and** the corresponding scale columns. That coupling is what distinguishes profiling blockscale from profiling vanilla GEMM.

### Baseline Geometry

| Field | Value |
| ----- | ----- |
| Tile | M64 × N128 × K32 (four K32 steps per TILE_K=128) |
| Swizzle | 128 on TMA |
| WARP_M | 64 (fixed for E4M3 WGMMA constraints) |
| Parallelism | CTA grid over M and N tile indices |

### Scale Tensor Shapes

| Tensor | Shape | Indexing |
| ------ | ----- | -------- |
| `scale_lhs` | `[M, K/128]` | One scale per M-row per K-block |
| `scale_rhs` | `[N/block_n, K/128]` | Aligned to block_n and K-block |

The device loop `iv_k` steps K-tiles, and `mma.row.row.scale` consumes the matching scale slice. Variants that transpose how scales are stored (`transposed_scale`) trade TMA friendliness against index arithmetic in the inner loop.

## Measured Baseline Throughput

| Shape | TFLOPS | Efficiency vs 3026 peak |
| ----- | ------ | ----------------------- |
| 2048³ | 314.2 | 10.4% |
| 4096³ | **397.9** | **13.2%** |

### Why 13% of Peak Is Not a Bug

The 3026 TFLOPS figure is a tensor-core marketing peak under favorable assumptions. Block-scaled GEMM:

- Issues **more global traffic per FLOP** than dense GEMM (matrix + scales)
- Uses fused MMA that does not pack identically to the simplest FP8×FP8→FP32 throughput tests
- Runs at problem sizes where L2 and wave quantization matter

So 13% is a lower bound on quality, not an upper bound on ambition. A plain dense FP8 GEMM approaches much higher fractions of peak because operand traffic dominates and fused scale paths are absent. Block-scaled intentionally pays those costs for accuracy. The optimization question is: how much of the lost throughput can we reclaim with better overlap and memory hints?

## Where the Cycles Go

The published README compresses 71 iterations into four shipped snapshots. The search was guided by four concerns:

**1. Schedule overlap.** The consumer finishes WGMMA, does scale_accumulator work, and only then drives the next K-tile TMA. The gap between "WGMMA done" and "next TMA started" is idle bandwidth. iter049 attacks this.

**2. Tile width.** N128 tiles finish K-pipeline steps quickly but launch many CTAs along N. On large N, wave quantization and per-CTA overhead hurt. iter051 doubles to N256.

**3. Cache behavior.** At 4096³, RHS panels are large. TMA traffic does not always stick in L2. iter053 adds L2 promotion hints.

**4. Scale latency.** Scale loads inside a tight WGMMA loop can stall the consumer if load latency is not hidden. iter066 prefetches scales before the hot loop.

## Source Variants

Under `blockscale_gemm_v2/`, additional `.co` files factor the scale path differently:

| Variant | What it does |
| ------- | ------------ |
| `rhs_scale_dma_smem` | Stage RHS scales via TMA into shared memory |
| `scale_dma_smem` | Stage all scales via TMA into shared memory |
| `transposed_scale` | Change scale layout for coalescing vs index cost |
| `tileN` | Tile along N explicitly in the Croktile structure |

These align with the theme that scale DMA is an alternative when register pressure or load scheduling hurts WGMMA issue interval.

**Verification:** 512 coprime-stride samples over M×N, each compared against a full FP32 reference dot with blockscale factors. That keeps iteration velocity high while catching scale indexing and MMA mistakes.

Next: [Optimization patterns](pattern-optimizations.md) — the four steps from 397.9 to 621 TFLOPS.
