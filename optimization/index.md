# Performance Tuning Demos

When we say **performance tuning** in this tutorial, we mean the same thing you would do on a real kernel: start from something that runs correctly, measure it with a stable harness, read the numbers against hardware limits, then change one axis at a time and watch what moves. The Choreo matmul benchmarks all share that rhythm—warmup and repeat counts, TFLOPS from FLOPs and mean time, efficiency vs a documented peak—so the case studies below are comparable experiments, not one-off demos.

Before you open a specific kernel, skim how timing and reporting work in the tree:

- [Setting Up: TimerOption, TFLOPS, and HW Efficiency](setup-profiling.md)

## Case Studies

### [Dense GEMM FP16](matmul-f16/index.md)

Half-precision matrix multiply from about **208 TFLOPS** to **382+ TFLOPS** on H800 PCIe. Warp specialization, multi-stage pipelining, split-output, persistent kernels, and tile scheduling.

### [Sparse GEMM: FP16 and FP8 E4M3](gemm-sp/index.md)

Structured 2:4 sparse GEMM at **4096×8192×8192**: FP16 **368 → 655 TFLOPS** (iter143), FP8 E4M3 **671 → 1127 TFLOPS** (iter068). Metadata and TMA staging, warp specialization, three-stage pipelines, barrier tuning, and the `.co`-to-`.cu` boundary.

### [Block-Scaled GEMM FP8](blockscale-gemm/index.md)

FP8 E4M3 with per-block scaling: baseline **314.2 TFLOPS @2048³** and **397.9 @4096³**, best shipped kernel **iter066** at **621 @4096³** (**+56%**). Scale DMA staging, warp-specialized blockscale pipelines, and transposed scale layouts.
