# Block-Scaled GEMM FP8 (E4M3): Case Study

This walkthrough follows **FP8 E4M3 matrix multiply with per-block scaling** on **Hopper SM90a**, measured on **H800 PCIe** (114 SMs). Operands are **e4m3**; the accumulator stays in **FP16**. Along **K**, each block aligned with **128**-element tiles carries **FP32 scale factors** for left and right operands so inner products stay useful after quantization without giving up FP8 bandwidth.

**Reference peak (headline)**

| Metric | Value |
|--------|--------|
| H800 PCIe FP8 tensor peak | **3026 TFLOPS** |
| Problem shapes (reported) | **2048³**, **4096³** |

**End-to-end results (AI-tune 2026-03-22)**

| Variant | TFLOPS @2048³ | TFLOPS @4096³ | HW eff @4096³ | Notes |
|---------|---------------|---------------|---------------|--------|
| Baseline (`blockscale_gemm_dyn_sm90.co`, M64N128K32) | 314.2 | 397.9 | 13.2% | Starting warp layout and tiling |
| iter049 | **380** | — | — | **+21% @2k** — TMA overlap around scale accumulation |
| iter051 | 372 | 602 | 19.9% | N256 WGMMA — doubled math per tile |
| iter053 | — | 610 | 20.2% | N256 + **L2 256B promotion** on RHS TMA |
| **iter066** | — | **621** | **20.5%** | **+56% @4k vs baseline** — N256 + L2 + **prefetch `scale_a`** |

Efficiency vs **3026 TFLOPS** stays modest because blockscaled GEMM pays extra scale traffic and fused math compared with a plain FP8 GEMM; the interesting part is **relative gain** from scheduling, tile geometry, cache hints, and scale prefetch.

**How to read the series**

1. [Baseline and block-scaling background](baseline-analysis.md) — why per-block scales exist, how the baseline kernel is wired, and what the first throughput numbers mean in context.
2. [Optimization patterns](pattern-optimizations.md) — TMA overlap, N256 tiles, L2 promotion, scale prefetch, and how sources under `benchmark/performance/blockscale_gemm/` and `blockscale_gemm_v2/` explore scale **DMA to shared memory** and layout variants.

**Compile and run (Cute backend example)**

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co \
  -o /tmp/bs.cute.result && bash /tmp/bs.cute.result --execute
```

Shipped winner harnesses with `run.sh` live under `croktile/benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_aitune_2026-03-22_iter{049,051,053,066}/`. Summary tables and iteration notes: `croktile/benchmark/performance/blockscale_gemm_v2/README_blockscale_gemm_e4m3_aitune_2026-03-22.md`.
