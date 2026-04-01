# Dense GEMM FP16: optimization case study

You take a Hopper (SM90a) half-precision GEMM from the Croktile benchmark baseline to throughput in the same band as cuBLAS on **H800 PCIe** (114 SMs). The path is deliberate: measure first, infer the limiter from TFLOPS and occupancy math, then pull in tutorial ideas—[warp specialization](../../tutorial/ch06-warpspec.md), [pipelining](../../tutorial/ch03-pipeline.md), [persistent tiling](../../tutorial/ch07-persistent.md)—only when the numbers justify them.

**Anchor numbers** (8192³ unless noted): baseline **208.7** TFLOPS (1p1c, WN=128, four stages); best shipped **382.5** TFLOPS (1p2c split-output, WN=152, non-persistent); **+83%** vs. that baseline on the same problem. Marketing peak for this class is often quoted around **1513** TFLOPS FP16 tensor; **cuBLAS** on this stack sits near **~380** TFLOPS—that is the bar the tuned kernels aim at.

**Read order**

1. [Baseline and profiling](baseline-analysis.md) — where **208.7** comes from and why it is schedule-bound, not “missing WGMMA.”
2. [Optimization patterns](pattern-optimizations.md) — how WN, stage depth, 1p2c split-output, launch mode, and compiler flags line up with the measured jumps.
3. [AI-tune last mile](aitune-last-mile.md) — shipped checkpoints (iter048, iter050, iter057, iter061), repro commands, and the WN sweep / occupancy cliff.

Full iteration tables: `croktile/benchmark/performance/matmul/README_matmul_f16_aitune_2026-03-23.md`. Representative sources: `matmul_f16_dyn_sm90.co`, `matmul_f16_dyn_sm90_warpspec_1p1c.co`, `matmul_f16_dyn_sm90_warpspec_1p2c.co`, and dated `*_iter048_*`, `*_iter050_*`, `*_iter057_*`, `*_iter061_*` builds.

**Before you start** — You should already be comfortable with [TMA and swizzle](../../tutorial/ch04-tma-swizzle.md) and [WGMMA](../../tutorial/ch05-mma.md): why **`tma.copy`** and **`mma.load.swiz`** agree on layout.

**Method** — Each step: quote TFLOPS at a fixed size, name the limiter (occupancy, pipeline bubbles, output contention), change one thing, re-measure.
