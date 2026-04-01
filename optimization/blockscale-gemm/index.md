# How to Optimize a Croktile Block-Scaled FP8 GEMM: a Worklog

This walkthrough follows FP8 E4M3 matrix multiply with **per-block scaling** on Hopper (SM90a), measured on H800 PCIe (114 SMs). Operands are E4M3; the accumulator stays in FP16. Along K, each block aligned with 128-element tiles carries FP32 scale factors so inner products stay useful after quantization — FP8 for density, scales for fidelity.

## Results Summary

| Step | Kernel | TFLOPS @2048³ | TFLOPS @4096³ | Δ vs baseline @4k |
| ---- | ------ | ------------- | ------------- | ----------------- |
| 0 | Baseline (M64N128K32) | 314.2 | 397.9 | — |
| 1 | TMA overlap with scale accumulation (iter049) | **380** | — | +21% @2k |
| 2 | N256 WGMMA (iter051) | 372 | 602 | +51% |
| 3 | N256 + L2 256B promotion (iter053) | — | 610 | +53% |
| 4 | N256 + L2 + prefetch scale_a (iter066) | — | **621** | **+56%** |

Reference peak: **3026 TFLOPS** (FP8 tensor on H800 PCIe). Efficiency vs peak stays modest because block-scaled GEMM pays extra scale traffic and fused math compared with a plain FP8 GEMM. The interesting part is relative gain from scheduling, tile geometry, cache hints, and scale prefetch.

## What Makes Block-Scaled GEMM Different

Plain FP8 trades dynamic range for bandwidth. Block scaling repairs this without reverting to FP16 weights: partition K into blocks (aligned with TILE_K = 128), and for each (row, block) on the left and (column, block) on the right, store a single FP32 scale factor. During the dot product, each block's contribution is scaled consistently so the FP16 accumulator approximates a higher-precision reference.

This means every K-tile iteration pulls **matrix data and scale metadata**. The Croktile surface expresses this as `mma.row.row.scale` instead of a plain `mma.row.row` — same tiling discipline, extra operands for scales, and the same pressure to hide TMA latency behind math.

## Read Order

1. [Baseline and block-scaling background](baseline-analysis.md) — why per-block scales exist, how the baseline kernel is wired, and the first throughput numbers in context
2. [Optimization patterns](pattern-optimizations.md) — TMA overlap, N256 tiles, L2 promotion, scale prefetch, with TFLOPS at each step

## Compile and Run

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co \
  -o /tmp/bs.cute.result && bash /tmp/bs.cute.result --execute
```

Shipped winner harnesses with `run.sh` live under `croktile/benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_aitune_2026-03-22_iter{049,051,053,066}/`. Full iteration history (71 iterations): `README_blockscale_gemm_e4m3_aitune_2026-03-22.md`.
