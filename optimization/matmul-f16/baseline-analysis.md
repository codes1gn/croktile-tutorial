# Baseline and profiling

This page is about where the kernel sits before you change structure, and how you read that from measurements. Unless stated otherwise, the numbers come from the 2026-03-23 AI-tune log; the headline baseline is **8192³**.

## What “baseline” means here

On main, the reference point is **208.7 TFLOPS** at **8192³** with **1p1c** warp specialization (one TMA producer warpgroup, one WGMMA consumer warpgroup), **WN = 128** (`MATMUL_WARP_N`), and **STAGES = 4** for the K-direction operand ring—the same pipelining idea as [Chapter 3](../../tutorial/ch03-pipeline.md), with roles split as in [Chapter 6](../../tutorial/ch06-warpspec.md). The Croktile sources that embody this are **`matmul_f16_dyn_sm90.co`** and **`matmul_f16_dyn_sm90_warpspec_1p1c.co`**. Operand layout (swizzle, `MATMUL_TILE_K`, `MATMUL_SWIZ`) follows [Chapter 4](../../tutorial/ch04-tma-swizzle.md); the baseline is not broken on addressing. It is **under-scheduled** relative to what the SM can take once tile width, stage depth, and how you stage the output line up with the problem size.

For a square GEMM of side **S**, useful work is **2S³** multiply-adds (one FMA ≈ two flops), so **TFLOPS ≈ 2S³ / t × 10⁻¹²**. With **S** fixed, TFLOPS tracks inverse runtime. That is why the study moves between **2048³**, **4096³**, and **8192³**: grid waves, L2 footprint, and tail behavior change; the flop definition does not.

## Why 208.7 TFLOPS still leaves a large gap

**208.7** is a lot of throughput in the abstract, but it sits far below both the **~1513** TFLOPS theoretical FP16 tensor peak and the **~380** TFLOPS you get from **cuBLAS** on the same machine. That gap is what you expect when latency hiding, occupancy, and pipeline depth are off: WGMMA can be “on” while the schedule still wastes cycles. Against **~380** TFLOPS, the baseline is roughly **55% of cuBLAS**—enough to treat the kernel as **schedule-bound**, not as if the hardware were idle for lack of math.

You do not need a single magic counter to argue that. You combine **throughput vs. problem size** (same kernel at 2048³, 4096³, 8192³—a kernel that looks fine small can be SMEM- or wave-limited large), **shared memory vs. occupancy** (each extra stage multiplies operand staging; Hopper’s **228 KB** per-SM shared budget means residency can step discretely), and **role balance in 1p1c** (consumer stuck on **`wait full`** or producer on **`wait empty`** means a bubble-limited pipeline). Warp specialization fixes *who* does TMA vs. WGMMA; it does not by itself size the ring or the tile.

Early tuning at **2048³** (Phase 1) stresses inner-loop efficiency without paying the full cost of **8192³** every iteration. There the 1p1c family lands near **204** TFLOPS with WN=128 and four stages—same story as the big cube, on a faster edit-measure loop.

## Waves, occupancy, and why size matters

With a normal data-parallel grid, block count scales with **M/WM** and **N/WN**. Hardware runs that in waves across the **114** SMs. If per-block time is uneven or occupancy drops so fewer blocks run together, the last wave leaves SMs idle. [Persistent kernels](../../tutorial/ch07-persistent.md) target that grid-level tail. This baseline used a conventional grid; later, **non-persistent** still won at **8192³** once inner-block throughput dominated—persistence is not always the win when the block schedule is already the bottleneck.

## What Phase 1 and Phase 2 previews teach you

At **2048³**, two small experiments already show SMEM and lowering matter:

- **iter004**: WN=256, STAGES=2 → **208.9** TFLOPS. Fewer stages shrink the operand ring; footprint and residency move together.
- **iter023**: adds **`ptx-barrier`**, **`stmatrix`**, and **subspan** work → **214.3** TFLOPS (~**+5%** over the Phase-1 baseline). That is compiler and operand-path quality on top of the Croktile function, not a new algorithm.

Phase 2 is where the profile story shifts: **iter046** hits **242** TFLOPS at **2048³** (WN=176, STAGES=2); **iter048** keeps **WN=176** but uses **three** stages and jumps to **354.1** TFLOPS. Same tile width, different stage count—that is the clearest log lesson that **pipeline depth must be tuned with** tile width **and** revisited when you change problem size. **iter050** then validates **1p2c split-output** at **4096³** (~**375** TFLOPS); **iter057** carries that to **382.5** TFLOPS at **8192³**. Once TFLOPS at 2048³ crosses **350+**, you are not arguing about “turning WGMMA on” anymore; you are arguing about occupancy, output staging, and the large-cube grid.

## Matching the baseline when you reproduce

The teaching kernel **`matmul_f16_dyn_sm90_warpspec_1p1c.co`** is the clearest **1p1c** illustration in the repo, not the throughput champion. For apples-to-apples against the tune artifacts, use the **dynamic** benchmark **`matmul_f16_dyn_sm90.co`** for the **208.7** configuration. If your H800 number drifts from **208.7**, check clocks, that timing runs without accidental verification overhead, and that **M, N, K** are really **8192** (grid shape is sensitive).

**`MATMUL_WARP_M` / `MATMUL_WARP_N`**, **`MATMUL_TILE_K`**, **`MATMUL_STAGES`**, and **`MATMUL_SWIZ`** are the vocabulary: change any of them and recompute shared memory before you interpret TFLOPS—a small **WN** bump can land on the wrong side of the budget.

## Correctness as a gate

Published rows assume agreement with a reference. When TFLOPS jumps **+50%** in one step, you ask first whether the math changed, not whether Hopper “likes” the kernel. Use **`CROKTILE_SKIP_VERIFY`** only when you trust correctness; otherwise treat failures as hard stops—a fast wrong kernel sends the search backward.

If you have Nsight Compute, consumer stalls at **`wait full`** and producer stalls at **`wait empty`** line up with the same mental model as [Chapter 6](../../tutorial/ch06-warpspec.md). This writeup leans on TFLOPS as the public evidence; the mapping to counters is optional but consistent.

## Takeaway

The baseline was already a Hopper-style TMA + WGMMA warp-specialized pipeline. It gave **208.7** TFLOPS at **8192³** because tile width, stage count, output staging, and launch mode did not match that size’s occupancy and contention—not because the code ignored the architecture. The next page walks the structural and flag changes that close most of the distance to **cuBLAS**.

Next: [Optimization patterns](pattern-optimizations.md).
