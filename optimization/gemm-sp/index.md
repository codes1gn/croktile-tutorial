# How to Optimize a Croktile Sparse GEMM: FP16 and E4M3 Worklog

This case study walks **structured 2:4** sparse GEMM on Hopper (SM90a), measured on H800 PCIe (114 SMs). One sparsity pattern, one metadata story, two math paths: **FP16** and **FP8 E4M3** (accumulating to FP16). You run the same logical problem under both — what changes is which concern binds first.

**Problem:** 4096 × 8192 × 8192 GEMM with NVIDIA 2:4 structured sparsity — groups of four weights along K, two kept, two zero, with hardware metadata selecting the active pair.

## FP16 Summary

| Step | Kernel | TFLOPS | Δ vs baseline |
| ---- | ------ | ------ | ------------- |
| 0 | Baseline: 1p1c, TK64, 2-stage, swizzle 64 | 368 | — |
| 1 | Best `.co`: 1p2c + 3-stage (iter120) | 434 | +18% |
| 2 | Hand `.cu`: inner unroll 24 + FTZ (iter137) | 543 | +47% |
| 3 | TK128, TMA metadata, split RHS TMA (iter143) | **655** | **+78%** |

## E4M3 Summary

| Step | Kernel | TFLOPS | Δ vs baseline |
| ---- | ------ | ------ | ------------- |
| 0 | Baseline: 1p1c, swizzle 128/128, prepack, 2-stage | 671 | — |
| 1 | TMA metadata staging (iter001) | 759 | +13% |
| 2 | Early empty + merged barrier (iter016) | 772 | +15% |
| 3 | Software pipeline + warpgroup_wait (iter023) | 811 | +21% |
| 4 | 1p2c warp specialization (iter036) | 897 | +34% |
| 5 | 3-stage pipeline (iter040) | 1090 | +62% |
| 6 | Early empty arrive (iter068) | **1127** | **+68%** |

For scale: FP8 peak on this GPU is 3026 TFLOPS; FP16 dense peak is 1513 TFLOPS. Sparse kernels are not compared to these as "efficiency of dense GEMM" — the metadata overhead is real work. These anchor whether your numbers are in the right ballpark.

## What Makes Sparse Different

Dense GEMM teaches TMA + WGMMA rhythm. Sparse GEMM teaches **not to starve MMA while waiting on metadata**. The 2:4 pattern compresses the sparse operand 2× along K, but **metadata** — the indices telling hardware which two of four elements are live — rides as a separate data plane alongside operand traffic. When operand TMA and WGMMA are in decent shape, metadata often becomes the first-class bottleneck.

## Read Order

1. [Baseline analysis](baseline-analysis.md) — what 2:4 implies, why baselines land at 368 / 671, and why metadata sits on the critical path
2. [Optimization patterns](pattern-optimizations.md) — step-by-step with TFLOPS at each move: sync tuning, metadata delivery, 1p2c, pipeline depth, inner loop geometry
3. [AI-tune last mile](aitune-last-mile.md) — the `.co` ceiling vs `.cu` breakthrough on FP16, and why E4M3 automation carried most of the distance

## Source Files

Iteration tables: `croktile/benchmark/performance/gemm_sp/README_gemm_sp_f16_aitune_2026-03-25.md` and `README_e4m3_aitune_2026-03-21.md`. Kernel artifacts sit under `benchmark/performance/gemm_sp/`.
