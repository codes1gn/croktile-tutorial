# Performance Tuning Demos

In this part, we iteratively optimize three Croktile GEMM kernels on H800 PCIe (SM90a, 114 SMs). Each is written as a continuous worklog: start from a correct baseline, measure against hardware limits, change one thing, re-measure, and tell the story of why each optimization works.

Before diving in, skim [Setting Up: TimerOption, TFLOPS, and HW Efficiency](setup-profiling.md) for how timing and efficiency are computed — every story uses the same harness.

## [Dense GEMM FP16](dense-gemm-fp16.md)

Half-precision matmul from **208 → 382 TFLOPS** (+83%), matching cuBLAS. Tile geometry, pipeline depth, split-output 1p2c, and the WN=168 occupancy cliff.

## [Sparse GEMM: FP16 and E4M3](sparse-gemm.md)

Structured 2:4 sparse GEMM at 4096 × 8192 × 8192. FP16: **368 → 655 TFLOPS** (+78%). E4M3: **671 → 1127 TFLOPS** (+68%). Metadata delivery, the `.co` vs `.cu` boundary, and the 3-stage discontinuity.

## [Block-Scaled GEMM FP8](blockscale-gemm-fp8.md)

FP8 E4M3 with per-block scaling: **397 → 621 TFLOPS** (+56%). TMA overlap with scale accumulation, N256 tiles, L2 promotion, and scale prefetch.
