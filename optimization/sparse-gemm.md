# How to Optimize a Croktile 2:4 Sparse GEMM: FP16 and E4M3 Worklog

In this post, I'll walk the optimization of structured 2:4 sparse GEMM on Hopper (SM90a), measured on H800 PCIe (114 SMs). One sparsity pattern, one metadata story, two math paths. The FP16 kernel goes from 368 to 655 TFLOPS; the E4M3 kernel goes from 671 to 1127 TFLOPS. The same ideas help both, but which concern binds first is different.

Dense GEMM teaches TMA + WGMMA rhythm. Sparse GEMM teaches **not to starve MMA while waiting on metadata**. That distinction is what makes this walkthrough worth reading separately from the [dense FP16 story](dense-gemm-fp16.md).

**FP16 (4096 × 8192 × 8192, 2:4 structured sparsity):**

| Step | Kernel | TFLOPS | Δ vs baseline |
| ---- | ------ | ------ | ------------- |
| 0 | Baseline: 1p1c, TK64, 2-stage | 368 | — |
| 1 | Best `.co`: 1p2c + 3-stage (iter120) | 434 | +18% |
| 2 | Hand `.cu`: inner unroll 24 + FTZ (iter137) | 543 | +47% |
| 3 | TK128, TMA metadata, split RHS TMA (iter143) | **655** | **+78%** |

**E4M3 (same shape):**

| Step | Kernel | TFLOPS | Δ vs baseline |
| ---- | ------ | ------ | ------------- |
| 0 | Baseline: 1p1c, swizzle 128/128, 2-stage | 671 | — |
| 1 | TMA metadata staging (iter001) | 759 | +13% |
| 2 | Early empty + merged barrier (iter016) | 772 | +15% |
| 3 | Software pipeline + warpgroup_wait (iter023) | 811 | +21% |
| 4 | 1p2c (iter036) | 897 | +34% |
| 5 | 3-stage pipeline (iter040) | 1090 | +62% |
| 6 | Early empty arrive (iter068) | **1127** | **+68%** |

## What 2:4 Structured Sparsity Costs You

Along the sparse axis (K in the weight-like operand), every four consecutive values keep two nonzeros; the other two are zero. Hardware uses **metadata** — small index arrays that tell the sparse MMA path which lanes are live — so the core fetches packed nonzeros instead of pretending the matrix is dense.

![2:4 sparsity pattern and metadata](images/SparsityPattern_ManimCE_v0.19.1_dark.png#only-dark)
![2:4 sparsity pattern and metadata](images/SparsityPattern_ManimCE_v0.19.1_light.png#only-light)

You get 2× compression along K on the sparse side. The tradeoff is explicit: **metadata traffic** and **instruction overhead** ride alongside operand traffic. Metadata is small per tile, but it is touched every K iteration. Scalar loads that miss L2 behave like pointer chasing next to wide TMA — that is why vectorizing and hoisting metadata loads later produces measurable gains.

---

## FP16 Baseline: 368 TFLOPS

The starting kernel uses 1p1c warp specialization, swizzle 64 on LHS packing, TK=64, and a 2-stage operand ring. At 368 TFLOPS the schedule is not broken — it is **shallow** and **metadata-conservative**. TK64 and two stages leave little slack to hide metadata latency next to the math path.

## E4M3 Baseline: 671 TFLOPS

The E4M3 baseline reflects stronger FP8-oriented choices from the start: 1p1c, swizzle 128/128 on both sides, prepacked sparse operand, and 2-stage pipeline. 671 TFLOPS is roughly 22% of the 3026 TFLOPS FP8 peak — a reasonable starting point before deep staging.

---

## Synchronization and Warpgroup Tuning

Before widening tiles or adding stages, check that warpgroup-level waits are not over-serialized. It is the cheapest thing to fix.

**Fine-grained waits.** Producer and consumer warpgroups coordinate through async proxies and barriers. Coarse waits leave lanes idle while data is already ready. Tightening to `warpgroup_wait<1>` — the smallest sufficient wait depth — gives ~+4% on FP16 and is part of the E4M3 iter023 jump (811 TFLOPS, combining software pipelining with fine-grained waits).

**MMA batch configuration.** Hopper WGMMA splits work across batches of K fragments. A poor split underfeeds tensor cores relative to operand delivery. `--wgmma-split-batch` gives ~+5% on FP16. If Nsight shows WGMMA issue slots gapping while SMEM is ready, revisit batching before blaming TMA.

**Early empty, merged barriers, early arrive (E4M3).** Async pipelines use empty/full phases; late signals steal overlap. iter016 (772 TFLOPS) uses early empty plus a merged barrier. iter068 (1127 TFLOPS) refines who signals when with early empty arrive. Above ~900 TFLOPS on E4M3, sync polish is worth double-digit TFLOPS.

---

## Metadata Delivery: Where Sparse Diverges from Dense

This is the section that does not exist in the [dense FP16 story](dense-gemm-fp16.md). Metadata is a second operand plane that you must keep fed alongside matrix data.

![Metadata delivery: scalar vs TMA-staged](images/MetadataBottleneck_ManimCE_v0.19.1_dark.png#only-dark)
![Metadata delivery: scalar vs TMA-staged](images/MetadataBottleneck_ManimCE_v0.19.1_light.png#only-light)

**Read-only cache path.** Forcing `__ldg`-style loads on metadata gives ~+0.5% on FP16. Small, but it establishes consistency across tiles.

**Vectorization and hoisting.** Three changes that form one story — how metadata reaches registers before MMA consumes it:

| Change | FP16 Δ | What it does |
| ------ | ------ | ------------ |
| L2/128B-oriented grouping | +0.7% | Align metadata to cache line boundaries |
| `uint2` metadata vectorization | +8% | Load 2× metadata per instruction |
| Hoisted `__ldg` metadata | +7% | Move metadata loads before the K inner loop |

These percents are local (each step vs the previous edit). They do not multiply cleanly because interactions matter — hoisting weighs more after vectorization.

**TMA metadata staging — the strongest move.** Put metadata on the same async machinery as operands. Let TMA prefetch metadata tiles into shared memory on the producer side, instead of scalar loads inside the K loop.

On E4M3, this is **iter001 (759 TFLOPS)** — the very first optimization, +13% over baseline. On FP16, TMA-backed metadata arrives as part of the iter143 bundle alongside TK128 and split RHS TMA — it was harder to introduce on FP16 because the `.co` compiler could not express the TMA descriptor work.

**Risks you own:** wrong metadata for a repacked operand is a silent numerical bug. Run host checks on small sizes when you change load paths. When TK changes, diff metadata offsets and fragment boundaries.

---

## 1p2c and the 3-Stage Discontinuity

In 1p1c, one producer warpgroup issues all TMA and often absorbs setup work that steals issue slots from the consumer. 1p2c adds a second consumer warpgroup — more math capacity to keep up with data delivery.

**FP16 (iter120, 434 TFLOPS):** Best `.co` outcome — 1p2c + 3-stage. About +9% over the prior step in the chain.

**E4M3 (iter036, 897 TFLOPS):** 1p2c alone, before the 3-stage change. +34% over baseline.

Then comes the 3-stage discontinuity:

![E4M3: the 3-stage discontinuity at iter040](images/ThreeStagePipelineJump_ManimCE_v0.19.1_dark.png#only-dark)
![E4M3: the 3-stage discontinuity at iter040](images/ThreeStagePipelineJump_ManimCE_v0.19.1_light.png#only-light)

**E4M3 (iter040, 1090 TFLOPS):** This is the breakthrough — +62% vs baseline. The move past 1000 TFLOPS. This is not gradual improvement; it is a **step function** that says the pipeline went from "producer regularly stalls" to "producer stays ahead." Three stages lets the producer run ahead of the consumer, hiding TMA and metadata latency behind math.

**SMEM and occupancy.** Pushing from two to three stages increases SMEM footprint. If occupancy collapses, the math gains vanish. The E4M3 jump at 3-stage says the SM had enough headroom — metadata staging and warp specialization had already trimmed per-block SMEM before depth was added. Stop pursuing deeper pipelines when three independent mutations no longer move TFLOPS.

---

## Inner Loop, Epilogue, and Tile Geometry

**`stmatrix`.** Store-matrix paths for accumulators give ~+2% on FP16. Above 1000 TFLOPS on E4M3, epilogue matters less unless the profile says store is hot.

**Inner unroll and FTZ (FP16 iter137, 543 TFLOPS).** Compiler-generated `.co` schedules may not unroll the inner K loop enough to overlap address math, metadata prefetch, and WGMMA. Hand `.cu` at iter137 used **unroll 24** and **FTZ** to cut denorm penalties. This is the strongest "organic" `.cu` before iter143, and the first move that required leaving the `.co` world.

**TK128, TMA metadata, split RHS TMA (FP16 iter143, 655 TFLOPS).** TK64 keeps K tiles small, inflating trip count and metadata traffic per unit work. iter143 combines three structural changes: TK128 (halve K-loop trips), TMA metadata (async metadata plane), and split RHS TMA (bandwidth tracks consumer demand). Result: **+78% over baseline**. This is not a polishing pass — it is structural memory-system work that the `.co` compiler could not express.

E4M3 already used 128/128 swizzle from the baseline. The parallel is iter001 metadata + iter040 depth, not a literal copy of FP16 knobs. Do not copy FP16 swizzle 64 onto E4M3 128/128 without validation — bank conflict behavior changes.

---

## The `.co` Plateau and the `.cu` Breakthrough

The FP16 story has a sharp boundary between what automation can find and what needs human CUDA:

```
368 ─── .co automation ───> 434 (iter120, +18%)  ═══ CEILING ═══
                                                      │
434 ─── hand .cu ─────────> 543 (iter137, unroll+FTZ)
543 ─── hand .cu ─────────> 655 (iter143, TK128+TMA meta+split RHS)
```

That is +18% from baseline to best `.co`, then +51% from iter120 to iter143 once you have CUDA-level control. The second leg is **different expressiveness**: Croktile's `.co` compiler chooses loop nests, register allocation, and async-proxy placement, but sparse GEMM couples operand TMA, metadata, and WGMMA batching with warpgroup barriers. When the compiler serializes metadata consumption with MMA in a way no single pragma fixes, you need `.cu` surface area.

E4M3 has no analogous cliff. The search stays in automation territory from 671 to 1127, with 3-stage and barrier work as the headline structural wins. The baseline's stronger TMA and layout choices mean the compiler had less to trip over.

---

## What Transferred Between Dtypes

Copy **causal structure**, not parameter equality:

| Pattern | FP16 | E4M3 |
| ------- | ---- | ---- |
| Metadata on TMA plane | iter143 (late, `.cu`) | iter001 (early, automated) |
| Fine warpgroup sync | Early in chain | iter023 |
| 1p2c | iter120 | iter036 |
| 3-stage depth | iter120 (bundled) | iter040 (the >1000 jump) |
| Barrier micro-optimization | Secondary | iter016, iter068 |

If FP16 is stuck below ~450 TFLOPS, attack metadata vectorization, `__ldg`, 1p2c + 3-stage, and `warpgroup_wait` first. If E4M3 is already ~850+ TFLOPS, barrier/early-empty/arrive and stage tuning often beat more operand widening.

---

## Conclusion

Sparse GEMM optimization follows the same arc as dense — schedule first, widen second, cache-tune third — but adds **metadata as a second operand plane**. The metadata story is what makes 2:4 sparse kernels harder than dense: you can have WGMMA fed and ready while the metadata load path is still pointer-chasing through L2. Once metadata is on the TMA plane and the pipeline has enough depth, the same 1p2c and barrier polish from dense applies.

The FP16 case taught that `.co` automation has limits — the last +51% needed hand `.cu`. The E4M3 case taught that with stronger baseline choices, automation can carry you from 671 to 1127 without leaving the `.co` world.

Iteration tables: `README_gemm_sp_f16_aitune_2026-03-25.md` and `README_e4m3_aitune_2026-03-21.md`. Kernel artifacts: `benchmark/performance/gemm_sp/`.
