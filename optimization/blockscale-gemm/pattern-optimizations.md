# Optimization Patterns: Block-Scaled GEMM FP8

This page follows the patterns that moved blockscale GEMM E4M3 from the baseline (`blockscale_gemm_dyn_sm90.co`) to the best shipped kernel (**iter066**), using the **2026-03-22** AI-tune log as the source of truth. Targets are **SM90a**; headline peak stays **3026 TFLOPS** on H800 PCIe. Timing and environment knobs are documented in [setup-profiling.md](../setup-profiling.md).

## Results ladder (2048³ and 4096³)

| Iter | TFLOPS @2048³ | TFLOPS @4096³ | Δ vs baseline @4k | Primary lever |
|------|---------------|---------------|-------------------|---------------|
| baseline | 314.2 | 397.9 | — | M64N128K32 reference |
| iter049 | **380** | — | — | **TMA overlap** with **scale accumulation** |
| iter051 | 372 | 602 | +51% | **N256 WGMMA** (M64N256K32) |
| iter053 | — | 610 | +53% | **N256** + **L2 256B promotion** on **RHS TMA** |
| **iter066** | — | **621** | **+56%** | **N256** + **L2** + **prefetch `scale_a`** (before WGMMA loop) |

**Best @2048³** is **iter049** (**+21%** vs baseline). **N256** trades small-cube grid density for large-cube throughput, so **iter051** sits slightly below baseline at **2048³** while winning at **4096³**.

The four sections below follow that ladder: first you fix the schedule so TMA does not wait on scale work (**iter049**), then you widen the math tile (**iter051**), then you help the RHS stick in cache (**iter053**), and finally you pull scale loads ahead of the hot loop (**iter066**). Each step assumes you are already on the warp-specialized 1p1c template described in [baseline-analysis.md](baseline-analysis.md).

## Pattern 1: TMA overlap after WGMMA (iter049)

The consumer finishes **WGMMA**, does **scale_accumulator** work, and only then drives the next **K**-tile TMA, so TMA sits idle across the handoff. **iter049** issues the **next** **K**-block’s TMA loads as soon as the WGMMA wait completes, overlapping memory latency with scale-related math that does not need the new operands yet.

**Outcome:** **380 TFLOPS** at **2048³** (**+21%** over **314.2**), still in the **M64N128K32** tile class.

This is the same scheduling instinct as [Chapter 3: Pipelining](../../tutorial/ch03-pipeline.md): move independent work so the longest-latency piece (TMA) starts earlier. Blockscale adds **scale_accumulator** as a third phase beside load and MMA; **iter049** shows that phase can share the bubble with TMA.

## Pattern 2: N256 WGMMA — double math per tile (iter051)

**N128** tiles finish **K**-pipeline steps quickly but launch many CTAs along **N**; on large **N**, wave quantization and per-CTA overhead hurt. **iter051** moves to **M64N256K32**—double the **N** extent of the WGMMA tile per CTA. The README notes about **40 KB** shared memory for operand staging.

**Outcome:** **602 TFLOPS** at **4096³**. **2048³** falls to **372 TFLOPS** (vs **380** on **iter049**) because fewer blocks cover **N**; the grid is coarser and occupancy trades differently.

Wider **N** is the same knob as in dense FP16 tuning: more math per block, fewer blocks, heavier SMEM. For blockscale, RHS and **scale_rhs** footprint grows with **N**; you still need enough SMEM headroom if the pipeline stages TMA.

## Pattern 3: L2 promotion on RHS TMA (iter053)

At **4096³**, RHS panels are large; TMA traffic does not always stick in L2 the way you want. **iter053** sets **`CU_TENSOR_MAP_L2_PROMOTION_L2_256B`** on the RHS tensor map so Hopper promotes lines into L2 with a **256B** granularity policy.

**Outcome:** **610 TFLOPS** at **4096³** (**+8** over **iter051**).

You are not changing the math; you are nudging the memory system. The hint biases the cache toward reuse of RHS data across **K** iterations and CTAs that share spatial locality. It pairs naturally with wider **N** from **iter051**, which increases per-CTA RHS volume and makes L2 behavior matter more.

## Pattern 4: Prefetch per-row `scale_a` before WGMMA (iter066)

Scale loads can stall the consumer if they sit inside a tight WGMMA loop with short II. **iter066** prefetches per-row **`scale_a`** into registers **before** the inner WGMMA body so load latency hides behind independent setup or prior WGMMA work.

**Outcome:** **621 TFLOPS** at **4096³**; **+56.1%** vs baseline **397.9**.

By the time you reach **iter066**, the kernel is already doing heavy WGMMA and wide **N**; the remaining slack often sits in operand latency. Blockscale makes scales first-class operands—treat them like any other latency-bound input. Software prefetch, double-buffering, or DMA-to-SMEM (see the **v2** variants below) are all design axes when registers or scheduling bind.

## Choreo source variants

Under **`blockscale_gemm_v2/`**, additional **`.co`** files factor the scale path differently than register-immediate style in **`blockscale_gemm_dyn_sm90.co`**: **`rhs_scale_dma_smem`** and **`scale_dma_smem`** stage scales via TMA into shared memory; **`transposed_scale`** changes layout for coalescing vs index cost; **`tileN`** tiles along **N** explicitly in the Choreo structure. Those align with the README theme that **scale DMA** is an alternative when register pressure or load scheduling hurts WGMMA II.

Warp-specialized entry points include **`blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co`** (1p1c template) and **`..._m2048_n2048_k2048.co`** for launch and register tuning. **`blockscale_gemm_e4m3_dyn_sm90_warpspec_persis_1p1c.co`** connects to [Chapter 7: Persistent kernels](../../tutorial/ch07-persistent.md) for grid behavior across tiles.

## Compile flags (Cute + warp specialization)

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co \
  -o /tmp/bs.cute.result && bash /tmp/bs.cute.result --execute
```

**`-arch=sm_90a`** selects Hopper (WGMMA, TMA). **`--use-warpspec`** matches 1p1c producer/consumer lowering. **`--stmatrix`** enables store-matrix paths for accumulator writeback where the pipeline expects them. Pre-generated **iter049 / iter051 / iter053 / iter066** trees ship with **`run.sh`** for bit-identical reproduction of the README numbers.

## Takeaways

1. Blockscale adds a **scale critical path**; **iter049** shows scheduling (TMA vs **scale_accumulator**) matters as much as tile size.
2. **N256** (**iter051**) is the large-cube win: more FLOP per CTA, cost at small cubes.
3. **L2 promotion** (**iter053**) and **scale prefetch** (**iter066**) are late percentage gains on an already strong kernel—where memory hierarchy and operand latency dominate.
4. **`blockscale_gemm/`** and **`blockscale_gemm_v2/`** document alternative scale movement (DMA SMEM, transposed layouts) for future tuning when register or layout limits bind.

Full iteration history: **`ai-tune/2026-03-22/blockscale_gemm_v2`** (71 iterations). Summary: `choreo/benchmark/performance/blockscale_gemm_v2/README_blockscale_gemm_e4m3_aitune_2026-03-22.md`.
