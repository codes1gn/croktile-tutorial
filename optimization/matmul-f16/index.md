# How to Optimize a Croktile FP16 GEMM for cuBLAS-like Performance: a Worklog

In this walkthrough, we iteratively optimize a Hopper (SM90a) half-precision GEMM written in Croktile from **208 TFLOPS** to **382+ TFLOPS** on H800 PCIe — within spitting distance of cuBLAS on the same hardware. The path is deliberate: measure first, identify the bottleneck from TFLOPS and occupancy arithmetic, then apply exactly one optimization and re-measure.

Matrix multiplication on GPUs is the most important algorithm in deep learning. So how much work is it to write a performant Croktile SGEMM from a correct baseline? Here is the summary:

| Step | Kernel | TFLOPS @8192³ | vs cuBLAS (~380) |
| ---- | ------ | ------------- | ---------------- |
| 0 | Baseline: 1p1c, WN=128, 4-stage | 208.7 | 55% |
| 1 | Tile geometry: WN=176, STAGES=2 | 242.0 | 64% |
| 2 | Pipeline depth: WN=176, STAGES=3 | 354.1 | 93% |
| 3 | Split-output 1p2c, WN=128 | ~375.0 | 99% |
| 4 | Split-output 1p2c, WN=152, non-persistent | **382.5** | **101%** |
| 5 | WN=160, K-unroll, wgmma-wait-depth | 380.6 | 100% |

The +83% from baseline to best came entirely from Croktile function geometry, output staging, and compiler flags — no mixed precision, no split-K, no CUDA Graph capture.

## Prerequisites

You should already be comfortable with:

- [Chapter 4 (MMA)](../../tutorial/ch04-mma.md) — WGMMA and why `tma.copy` and `mma.load.swiz` agree on layout
- [Chapter 5 (branch and control)](../../tutorial/ch05-branch-control.md) — warp specialization and persistent kernels
- [Chapter 6 (synchronization)](../../tutorial/ch06-synchronization.md) — pipelining
- [Chapter 7 (advanced movement)](../../tutorial/ch07-advanced-movement.md) — TMA and swizzle

## Read Order

1. [Baseline analysis](baseline-analysis.md) — where 208.7 TFLOPS comes from, hardware limits calculation, and why the kernel is schedule-bound
2. [Optimization patterns](pattern-optimizations.md) — each step with the code change, measurement, and explanation
3. [AI-tune last mile](aitune-last-mile.md) — shipped checkpoints, repro commands, the WN sweep, and the occupancy cliff

## Source Files

Full iteration tables: `croktile/benchmark/performance/matmul/README_matmul_f16_aitune_2026-03-23.md`. Representative sources:

- `matmul_f16_dyn_sm90.co` — dynamic baseline
- `matmul_f16_dyn_sm90_warpspec_1p1c.co` — 1p1c teaching kernel
- `matmul_f16_dyn_sm90_warpspec_1p2c.co` — 1p2c split-output
- Dated `*_iter048_*` through `*_iter061_*` builds
