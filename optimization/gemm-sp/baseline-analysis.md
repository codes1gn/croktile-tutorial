# Baseline: Why 2:4 Sits Where It Does

Before we optimize anything, we need to understand what 2:4 structured sparsity actually costs and why the baselines land at 368 (FP16) and 671 (E4M3) TFLOPS.

## What 2:4 Structured Sparsity Means

Along the sparse axis (K in the weight-like operand), every four consecutive values keep **two** nonzeros; the other two are zero. The hardware uses **metadata** to tell the sparse MMA path which lanes are live, so the core fetches **packed** nonzeros instead of pretending the matrix is dense.

This gives you 2× compression along K on the sparse side in terms of stored weights. The tradeoff is explicit: **metadata traffic** and **instruction overhead** ride alongside operand traffic. You are not getting sparsity for free — you are trading matrix bandwidth for metadata bandwidth.

## The FP16 Baseline: 368 TFLOPS

The starting kernel uses Hopper in earnest:

| Parameter | Value |
| --------- | ----- |
| Warp spec | 1p1c (one TMA producer, one WGMMA consumer) |
| Swizzle | 64 on LHS packing / shared layout |
| TK | 64 |
| Pipeline | 2-stage operand ring |

At 368 TFLOPS the schedule is not broken — it is **shallow** and **metadata-conservative**. TK64 and two stages leave little slack to hide metadata latency next to the math path. The pipeline works; it just does not run far enough ahead.

Compare 368 to the 1513 TFLOPS FP16 dense peak only for order of magnitude — sparse effective FLOPs per stored element differ from dense, and metadata is real work. The question that matters is whether time goes to MMA, TMA, or metadata-plus-barriers.

## The E4M3 Baseline: 671 TFLOPS

The E4M3 baseline reflects stronger FP8-oriented choices from the start:

| Parameter | Value |
| --------- | ----- |
| Warp spec | 1p1c |
| Swizzle | 128/128 on LHS and RHS |
| Operand format | Prepacked sparse |
| Pipeline | 2-stage |

671 TFLOPS is roughly 22% of the 3026 TFLOPS FP8 peak — a reasonable starting point before pushing deep staging and producer-consumer overlap. The math ceiling is higher than FP16, so sync and pipeline bubbles show up sooner in relative terms even when absolute TFLOPS looks strong.

## Why Metadata Shows Up in Profiles

When you profile a 2:4 sparse GEMM, the symptoms of metadata bottlenecks look like this:

- Consumers show decent `wgmma` issue rates but **gaps between fragments**
- Extra L1/L2 traffic from scalar or poorly grouped loads compared to operand TMA
- **Serial dependence** when metadata is read inside the K loop without prefetch into registers or shared memory

The optimization chains on both dtypes attack these symptoms with different tools: `__ldg` read-only cache paths, `uint2` vectorization, hoisting, L2-friendly grouping, and eventually TMA-backed metadata. On E4M3, TMA metadata staging appears as early as iter001 (759 TFLOPS). On FP16, it is part of the iter143 mix alongside TK128 and split RHS TMA.

## Lower Bounding What the Kernel Does

For a 4096 × 8192 × 8192 GEMM with 2:4 sparsity:

```
Effective FLOPs: 2 × 4096 × 8192 × 8192 × (2/4) ≈ 275 GFLOP
                 (only 2 of 4 elements contribute per group)
Packed operand:  8192 × (8192/2) × element_size
                 (2× compression on sparse axis)
Metadata:        proportional to K/4 groups per row
```

The metadata is small per tile, but it is touched every K iteration. Scalar loads that miss L2 behave like pointer chasing next to wide TMA — that is why vectorizing and hoisting metadata loads produces measurable gains.

## Milestones Preview

Both ladders follow the same structural arc, with different dominant bottlenecks:

**FP16:** 368 → 434 (iter120, best `.co`: 1p2c + 3-stage) → 543 (iter137, hand `.cu`: inner unroll + FTZ) → **655** (iter143, TK128 + TMA metadata + split RHS TMA). The 434 → 655 jump is the `.co` vs `.cu` gap in miniature — [AI-tune last mile](aitune-last-mile.md) unpacks that.

**E4M3:** 671 → 759 (TMA metadata) → 772 (early empty + merged barrier) → 811 (software pipeline + `warpgroup_wait<1>`) → 897 (1p2c) → **1090** (3-stage) → **1127** (early empty arrive). The jump past 1000 TFLOPS at iter040 is the signature of pipeline depth — once operands and metadata prefetch far enough ahead, the math path stays fed.

## Shape and Measurement Notes

At 4096 × 8192 × 8192, N and K are large relative to M, so wave packing on 114 SMs makes grid efficiency sensitive to tile choices. The cleanest A/B tests are edits that touch only inner-loop sync or scheduling while the grid shape stays the same.

Timings can wobble with L2 state or clock behavior. A few TFLOPS of noise at 1100+ is sub-percent — small relative to the iter040 → iter068 gap, but it means methodology matters for last-mile claims. Prefer median over many repeats and, when you need tight comparisons, controlled clocks.

**Correctness:** Throughput is useless if metadata disagrees with packed indices. Keep host checks enabled while churning TK and unroll — those are the edits most likely to misalign 2:4 silently.

Next: [Optimization patterns](pattern-optimizations.md) — step-by-step through each optimization with measurements.
