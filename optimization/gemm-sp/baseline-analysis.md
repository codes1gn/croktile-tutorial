# Baseline Analysis: Why 2:4 Sits Where It Does

You are optimizing **structured 2:4** sparse GEMM at **4096 × 8192 × 8192**. Along the sparse axis (here, **K** in the weight-like operand), every four consecutive values keep **two** nonzeros; the other two are zero. Hardware uses **metadata** to tell the sparse MMA path which lanes are live, so the core fetches **packed** nonzeros instead of pretending the matrix is dense.

That regularity is what makes tiling and TMA predictable: you get a **2×** compression along **K** on the sparse side in terms of stored weights. The tradeoff is explicit: **metadata traffic** and **instruction overhead** ride next to operand traffic. Once operand TMA and WGMMA are in decent shape, **metadata** often becomes a first-class bottleneck—extra loads on the path from global memory (or staging) to the MMA interface, not a free sideband.

When you profile, you want to separate **tensor math**, **TMA / shared staging**, **metadata** (latency, coalescing, cache behavior), and **synchronization** between producer and consumer warpgroups. Sparse work teaches you to never let MMA sit idle waiting on meta while DRAM looks fine.

---

**FP16 baseline: 368 TFLOPS.** The starting kernel is already using Hopper in earnest: **1p1c** (one TMA producer warpgroup, one consumer), **swizzle 64** on the lhs packing / shared layout, **TK 64**, and a **2-stage** operand pipeline. At 368 TFLOPS the schedule is not broken; it is **shallow** and **metadata-conservative** relative to what the SM can sustain when stages, warp mix, and vectorized or TMA-backed metadata line up. TK64 and two stages leave little slack to hide metadata latency next to the math path.

Compare 368 TFLOPS to the **1513 TFLOPS** FP16 dense peak on this GPU class only for order of magnitude. Sparse effective FLOPs per stored element differ from dense, and metadata is real work. The question that matters is whether time goes to **MMA**, **TMA**, or **metadata plus barriers**.

**E4M3 baseline: 671 TFLOPS.** Here the baseline already reflects stronger FP8-oriented choices: still **1p1c**, but **swizzle 128 / 128** on lhs and rhs, a **prepacked** sparse operand, and **2-stage** pipelining. Roughly **22%** of the **3026 TFLOPS** FP8 headline peak is a reasonable sanity check for a first structured-sparse implementation before you push **deep staging** and **producer–consumer overlap**. The math ceiling is higher than FP16, so **sync and pipeline bubbles** tend to show up sooner in relative terms even when absolute TFLOPS looks strong.

---

**Why metadata shows up in profiles.** Symptoms include consumers with decent `wgmma` issue rates but **gaps between fragments**, extra L1/L2 traffic from scalar or poorly grouped loads compared to operand TMA, and **serial dependence** when metadata is read **inside** the K loop without prefetch into registers or shared. The FP16 AI-tune chain attacks that with `__ldg`, **`uint2` vectorization**, **hoisting**, L2-friendly grouping, and eventually **TMA-backed metadata**. On E4M3, **TMA metadata staging** appears as early as **iter001 (759 TFLOPS)**—the same idea at a higher baseline.

**Vocabulary you will see on the next pages:** **1p1c / 1p2c** is one producer versus one producer with **two** consumer warpgroups (warp specialization). **TK** is the tile size along **K** in the inner steady state. **2-stage / 3-stage** counts buffered operand slots in the async ring. **Prepack** means the sparse operand is already in the hardware-packed 2:4 layout.

---

**Milestones preview (same story, two dtypes).** On FP16, the README anchors **368** (baseline) → **434** at **iter120** (best `.co`: 1p2c + 3-stage) → **543** at **iter137** (hand `.cu`: inner unroll 24 and FTZ) → **655** at **iter143** (TK128, TMA metadata, split RHS TMA). The step from 434 to 655 is the **`.co` versus `.cu` gap** in miniature; [aitune-last-mile](aitune-last-mile.md) unpacks that.

On E4M3, the ladder runs **671** (baseline) → **759** (iter001, TMA metadata) → **772** (iter016, early empty + merged barrier) → **811** (iter023, software pipeline + `warpgroup_wait<1>`) → **897** (iter036, 1p2c) → **1090** (iter040, **3-stage**) → **1127** (iter068, early empty **arrive**). The jump into **>1000 TFLOPS** at iter040 is the signature of **pipeline depth** once operands **and** metadata prefetch far enough ahead; the last tens of TFLOPS are **sync polish** on top of that.

**Pipeline depth versus tile width.** More stages hide TMA and metadata latency but cost shared memory and registers; if occupancy collapses, extra stages can hurt. For this shape, the measured E4M3 jump at **3-stage** says the SM had enough headroom—likely because metadata staging and warp specialization had already trimmed bubbles. Widening **TK** (FP16 **TK128** at iter143) amortizes inner-loop and epilogue overhead per K step but forces you to revisit **swizzle**, **TMA descriptors**, and **metadata** layout together—never TK in isolation.

---

**Shape and measurement.** At **4096 × 8192 × 8192**, **N** and **K** are large relative to **M**, so wave packing on **114 SMs** can make grid efficiency sensitive to tile choices. When you compare iterations, watch for CTA count changes that move tail-wave behavior; the cleanest A/B tests are edits that touch **only** inner-loop sync or scheduling while the grid stays the same. TFLOPS here follow the harness in [setup-profiling](../setup-profiling.md) for 2:4 (nonzeros only); keep the formula fixed so before/after runs stay comparable.

Timings can wobble with L2 state or clock behavior; prefer a **median** over many repeats and, when you need tight comparisons, controlled clocks. A few TFLOPS of noise at **1100+** is sub‑percent—small relative to iter040→iter068, so methodology matters for last-mile claims.

**Correctness.** Throughput is useless if metadata disagrees with packed indices. Keep host checks enabled while you churn **TK** and **unroll**—those are the edits most likely to misalign 2:4 silently.

**Dense case study.** The [dense FP16 matmul](../matmul-f16/index.md) story tops out near cuBLAS-class throughput at **8192³**. Sparse FP16 here reaches **655 TFLOPS** at a different shape and FLOP definition; the habits still rhyme—**stages**, **1p2c**, **swizzle**, **TMA** first, micro-optimizations after you know where the profile points.

When you are done with baselines, continue to [optimization patterns](pattern-optimizations.md) for the pattern narrative tied to those TFLOPS numbers, and [aitune-last-mile](aitune-last-mile.md) for where automation hands off to hand-authored CUDA on FP16.
