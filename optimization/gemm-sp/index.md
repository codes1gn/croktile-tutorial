# Sparse GEMM: FP16 and E4M3 (2:4)

This case study walks **structured 2:4** sparse GEMM on Hopper (SM90a), measured on **H800 PCIe** (114 SMs). You run the same logical problem in **FP16** and **FP8 E4M3** (accumulate to FP16): one sparsity pattern, one metadata story, two different math paths and ceilings.

**Problem:** \(4096 \times 8192 \times 8192\) GEMM with NVIDIA **2:4** on the sparse operand—groups of four weights along **K**, two kept, two zero, with hardware metadata selecting the active pair.

**Results (TFLOPS on this harness and shape)**

- **FP16:** 368 → **655** (+78%), best at **iter143**
- **E4M3:** 671 → **1127** (+68%), best at **iter068**

For scale only: FP8 peak on this class of GPU is **3026 TFLOPS**, FP16 dense peak **1513 TFLOPS**. Sparse kernels are not compared to those as “efficiency of dense GEMM”; they anchor whether your numbers are in the right ballpark.

**Read next**

1. [Baseline analysis](baseline-analysis.md) — what 2:4 implies, why baselines land at 368 / 671, and why **metadata** sits on the critical path.
2. [Optimization patterns](pattern-optimizations.md) — sync and warpgroup tuning, metadata delivery, 1p2c and pipeline depth, inner loop and tile geometry, with TFLOPS ladders woven through.
3. [AI-tune and the last mile](aitune-last-mile.md) — the **`.co` / `.cu` boundary** on FP16 (plateau at iter120, breakthrough to iter143) versus E4M3, where automated sweeps carry most of the distance to iter068.

Iteration tables and short change blurbs live in `croktile/benchmark/performance/gemm_sp/README_gemm_sp_f16_aitune_2026-03-25.md` and `README_e4m3_aitune_2026-03-21.md`. Kernel artifacts sit under `benchmark/performance/gemm_sp/` (`.co` variants and per-iteration `.cu` trees).
