# Performance Tuning Demos

In this part, we iteratively optimize three Croktile GEMM kernels on **H800 PCIe** (SM90a, 114 SMs). The goal is not to replace cuBLAS, but to deeply understand the performance characteristics of Hopper GPUs through the lens of Croktile's abstractions — warp specialization, TMA pipelining, tile geometry, and compiler flag tuning.

Each case study starts from a correct baseline, measures it against hardware limits, then changes one axis at a time and watches what moves. The optimization arc mirrors what you would do on a real kernel: profile, hypothesize, change, re-measure.

## Case Studies at a Glance

### [Dense GEMM FP16](matmul-f16/index.md)

| Step | Kernel | TFLOPS @8192³ | vs cuBLAS (~380) |
| ---- | ------ | ------------- | ---------------- |
| 0 | Baseline (1p1c, WN=128, 4-stage) | 208.7 | 55% |
| 1 | WN=176, STAGES=2 (iter046) | 242.0 | 64% |
| 2 | WN=176, STAGES=3 (iter048) | 354.1 | 93% |
| 3 | 1p2c split-output, WN=128 (iter050) | ~375.0 | 99% |
| 4 | 1p2c split-output, WN=152, non-persistent (iter057) | **382.5** | **101%** |
| 5 | WN=160, K-unroll, wait-depth tuning (iter061) | 380.6 | 100% |

### [Sparse GEMM: FP16 and FP8 E4M3](gemm-sp/index.md)

**FP16 (4096 × 8192 × 8192, 2:4 structured sparsity):**

| Step | Kernel | TFLOPS | Δ vs baseline |
| ---- | ------ | ------ | ------------- |
| 0 | Baseline (1p1c, TK64, 2-stage) | 368 | — |
| 1 | Best `.co`: 1p2c + 3-stage (iter120) | 434 | +18% |
| 2 | Hand `.cu`: inner unroll 24 + FTZ (iter137) | 543 | +47% |
| 3 | TK128, TMA metadata, split RHS TMA (iter143) | **655** | **+78%** |

**E4M3 (same shape):**

| Step | Kernel | TFLOPS | Δ vs baseline |
| ---- | ------ | ------ | ------------- |
| 0 | Baseline (1p1c, swizzle 128/128, 2-stage) | 671 | — |
| 1 | TMA metadata staging (iter001) | 759 | +13% |
| 2 | Early empty + merged barrier (iter016) | 772 | +15% |
| 3 | Software pipeline + warpgroup_wait (iter023) | 811 | +21% |
| 4 | 1p2c (iter036) | 897 | +34% |
| 5 | 3-stage pipeline (iter040) | 1090 | +62% |
| 6 | Early empty arrive (iter068) | **1127** | **+68%** |

### [Block-Scaled GEMM FP8](blockscale-gemm/index.md)

| Step | Kernel | TFLOPS @4096³ | Δ vs baseline |
| ---- | ------ | ------------- | ------------- |
| 0 | Baseline (M64N128K32) | 397.9 | — |
| 1 | TMA overlap with scale accumulation (iter049) | 380 @2k | +21% @2k |
| 2 | N256 WGMMA (iter051) | 602 | +51% |
| 3 | N256 + L2 256B promotion (iter053) | 610 | +53% |
| 4 | N256 + L2 + prefetch scale_a (iter066) | **621** | **+56%** |

## Before You Start

Skim [Setting Up: TimerOption, TFLOPS, and HW Efficiency](setup-profiling.md) for how timing, TFLOPS, and hardware efficiency are computed — every case study uses the same measurement harness. You should also be comfortable with the Part I tutorial through at least [Chapter 7 (advanced movement)](../tutorial/ch07-advanced-movement.md).
