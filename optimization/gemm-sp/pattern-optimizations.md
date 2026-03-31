# Optimization Patterns: One Campaign, Two Ladders

You are not memorizing a catalog—you are following a single thread: **profile, form a bottleneck hypothesis, change one family of things, re-measure**. The patterns below grouped the work from the 2026-03 AI-tune logs on **4096 × 8192 × 8192** 2:4 sparse GEMM. What helps FP16 usually helps E4M3; the difference is **which** concern binds first.

**TFLOPS ladders (anchors, not every micro-step)**

- **FP16:** 368 (baseline) → **434** (iter120) → **543** (iter137) → **655** (iter143)
- **E4M3:** 671 → **759** → **772** → **811** → **897** → **1090** → **1127**

The FP16 README also lists a **percent chain** (each step vs the kernel immediately before that edit). Those percents are **local**; they do not multiply cleanly to **368 → 655** because interactions matter—**hoisting** weighs more after **vectorization**, and so on. Always record **absolute TFLOPS** next to any percent so regressions stay visible.

---

## 1. Synchronization and warpgroup tuning

**Fine-grained warpgroup waits.** Producer and consumer warpgroups coordinate through async proxies and barriers. Coarse waits leave lanes idle while data is already ready. On FP16, tightening to something like **`warpgroup_wait<1>`** (smallest sufficient wait depth) showed up as an early gain on the order of **+4%** vs the then-current kernel. On E4M3, **iter023 (811 TFLOPS)** combines **software pipelining** with that same idea: the profile pointed at **sync-induced bubbles**, and narrowing wait granularity recovered cycles without changing tile shape. Before you widen tiles or add stages, check that warpgroup-level waits are not overserialized.

**MMA batch configuration.** Hopper **WGMMA** splits work across batches of K fragments; a poor split underfeeds tensor cores relative to operand delivery. Flags such as **`--wgmma-split-batch`** (as logged in the FP16 chain, on the order of **+5%**) reshape how batches map to instructions. The intent transfers to E4M3 even if exact flag spelling varies by build: if Nsight shows WGMMA issue slots gapping while shared memory is ready, revisit **batching** before you blame TMA alone.

**Early empty, merged barriers, early arrive (E4M3-heavy).** Async pipelines use empty/full phases; late signals or over-synchronized barriers steal overlap. **iter016 (772 TFLOPS)** uses **early empty** plus a **merged barrier** to cut round trips on proxy state. **iter068 (1127 TFLOPS)** refines **who** signals **when** with **early empty arrive**. Above roughly **900 TFLOPS** on E4M3, that kind of sync polish is worth double-digit TFLOPS—use Nsight **warp stall** views on barrier-related reasons. FP16 README emphasizes metadata and TMA more than this exact vocabulary, but **warpgroup_wait** and stage tuning play the same structural role.

---

## 2. Metadata delivery

**Read-only cache path (`__ldg`).** Metadata often lives in global memory and is touched every K tile. Scalar loads that miss behave like pointer chasing next to wide TMA. Forcing a read-only, L2-friendly path with **`__ldg`-style** loads bought about **+0.5%** on FP16—the baseline already cached somewhat, but **consistency** across tiles mattered.

**L2-friendly layout, `uint2`, hoisting.** Small metadata arrays per tile still need **128B-line** discipline if you do not want L2 thrash across CTAs. **L2 / 128B-oriented** grouping logged near **+0.7%**; **`uint2` metadata** near **+8%**; **hoisted `__ldg` metadata** near **+7%** on FP16. Treat those three as one story: **how** metadata reaches registers **before** MMA consumes it—vector width, alignment, and software pipelining on the **metadata plane**.

**TMA metadata staging.** The stronger move is to put metadata on the **same async machinery** as operands where the toolchain allows. On E4M3, **iter001 (759 TFLOPS)** is the headline jump from **TMA metadata staging**. On FP16, **TMA-backed metadata** is part of the **iter143** mix alongside TK128 and split RHS TMA.

**Risk you own:** wrong metadata for a repacked operand is a silent numerical bug. Run host checks against a dense reference on small sizes when you change load paths; when **TK** changes, diff metadata **offsets** and fragment boundaries. **3-stage** can overflow SMEM—print shared usage and watch the occupancy cliff. **1p2c** needs event ordering consistent with your warp-spec model ([Ch6](../../tutorial/ch06-warpspec.md)). **TK128** without matching TMA descriptors invites bank conflicts; diff descriptor setup against TK64.

---

## 3. Structural changes: 1p2c and pipeline depth

**1p2c and multi-stage rings.** In **1p1c**, one producer warpgroup issues all TMA and often absorbs setup work that steals issue slots from the consumer’s steady `wgmma` stream. **1p2c** adds a second consumer warpgroup (exact role split follows the Choreo schedule) and pairs with **3-stage** (or deeper) operand buffering so producers run **ahead** of consumers.

On FP16, **iter120 (434 TFLOPS)** is the best **`.co`** outcome: **1p2c + 3-stage**, a large structural step (~**+9%** class vs the prior step in the logged chain). On E4M3, **iter036 (897 TFLOPS)** marks **1p2c**; **iter040 (1090 TFLOPS)** is the **3-stage** breakthrough—about **+62%** vs the **671** baseline in the README narrative, and the move into **>1000 TFLOPS** territory. **Depth without producer throughput** tends to fail; **1p2c without enough stages** still bubbles.

**Shared memory and occupancy.** Pushing from two to three stages increases SMEM footprint. If occupancy falls off a cliff, math gains can vanish—watch shared usage prints and profiler **warps active**. Stop pursuing deeper pipelines when SMEM exceeds what your cluster config allows or when three independent mutations in the same family no longer move TFLOPS—then borrow ideas from the other dtype’s milestones (for example, if FP16 is metadata-stuck, study **iter001** on E4M3 for TMA-meta layout).

**iter040 as a bundled change.** Stage depth rarely lands in isolation; **iter040** is an example where a **bundled** 3-stage change was justified because producer staging and metadata path were already compatible.

**Automation guardrails.** Fixed clocks (where policy allows) or wide repeat counts, pinned host memory if the harness uses async copies, and **one** change per run when debugging versus **bundled** changes when sweeping a known-good neighborhood keep noise from masquerading as signal.

---

## 4. Inner loop, epilogue, and tile geometry

**Store path (`stmatrix`).** Accumulators still have to exit through shared or registers with bank-safe stores. **`stmatrix`** where the toolchain supports it aligned store paths with Hopper preferences for about **+2%** on FP16. When the kernel is math- and sync-bound above **1000 TFLOPS** on E4M3, epilogue tricks matter less unless the profile says **store** is hot—if metadata is still the long pole, skip this until it is not.

**Inner unroll and FTZ (FP16 iter137).** Compiler-generated **`.co`** schedules may not unroll the inner K loop enough to overlap address math, metadata prefetch, and `wgmma`. Hand **`.cu`** at **iter137 (543 TFLOPS)** used **unroll 24** plus **FTZ** to cut denorm edge cases on the data you had. That is the strongest **“organic” `.cu`** before **iter143**. Inner unroll still applies philosophically to E4M3; FTZ is less central when inputs are E4M3 and the accum path is already constrained. Gate **FTZ** behind a benchmark flag and document ULP impact if accuracy matters.

**TK128, TMA metadata, split RHS TMA (FP16 iter143).** **TK64** keeps K tiles small, which can inflate trip count and metadata traffic per unit work. **iter143 at 655 TFLOPS** combines **TK128**, **TMA metadata**, and **split RHS TMA** so bandwidth tracks consumer demand—**+78%** vs the **368** baseline overall. E4M3 already used **128/128** swizzle from the baseline; the parallel is **iter001** metadata plus **iter040** depth, not a literal copy of every FP16 knob.

**Swizzle and dtype.** Do not copy FP16 **swizzle 64** onto E4M3 **128/128** or vice versa without validation—bank conflict behavior changes. **Accum dtype** (E4M3 into FP16 accum) shifts register pressure relative to pure FP16 sparse. Choreo **`.co`** sources and AI-tune flags (e.g. **`--wgmma-split-batch`**) are the same CUDA ideas under different spelling—treat README flag names as “reshape WGMMA batching,” exact tokens may differ by revision.

---

## Closing discipline

Before you end a tuning session, sanity-check: metadata **vectorized**, **hoisted**, and ideally **TMA-staged** where possible; **stage count** matched to **producer rate** with **1p2c** justified by profiled slack; **warpgroup waits** and empty/full signaling **minimal** without races; **TK** changes forced a joint pass on **swizzle, TMA, and metadata**; and you know where **`.co` plateaus** so you do not burn days on micro-tweaks that need **`.cu`** expressiveness ([aitune-last-mile](aitune-last-mile.md)).

**If FP16 is stuck below ~450 TFLOPS** on this class of GPU, the logs suggest attacking **metadata vectorization**, **`__ldg`**, **1p2c + 3-stage**, and **`warpgroup_wait`** before the heaviest TMA-metadata experiments. **If E4M3 is already ~850+ TFLOPS**, **barrier / early empty / arrive** and **stage** tuning often beat more operand widening—math is fed enough that **sync** dominates.

Patterns from [dense matmul](../matmul-f16/pattern-optimizations.md)—**1p2c**, multi-stage rings, tile search—still apply, but sparse adds **metadata as a second operand plane**. Dense work teaches **TMA + WGMMA rhythm**; sparse work teaches **not to starve MMA while waiting on meta**. Keep one profiling preset that highlights DRAM and another that highlights **L1/L2 on small meta arrays**.

For TFLOPS definitions and harness detail, see [setup-profiling](../setup-profiling.md). TMA swizzle vocabulary lives in [Ch4](../../tutorial/ch04-tma-swizzle.md); warp specialization in [Ch6](../../tutorial/ch06-warpspec.md). [Ch10](../../tutorial/ch10-cpp-inline-macros.md) touches gemm_sp-style producer/consumer asymmetry.
