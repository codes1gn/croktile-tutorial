# Optimization Patterns: One Campaign, Two Ladders

You are not memorizing a catalog — you are following a single thread: **profile, form a bottleneck hypothesis, change one family of things, re-measure.** What helps FP16 usually helps E4M3; the difference is which concern binds first.

Here are the two ladders side by side for reference:

**FP16:** 368 → 434 → 543 → **655**

**E4M3:** 671 → 759 → 772 → 811 → 897 → 1090 → **1127**

## Step 1: Synchronization and Warpgroup Tuning

### Fine-Grained Warpgroup Waits

Producer and consumer warpgroups coordinate through async proxies and barriers. Coarse waits leave lanes idle while data is already ready. The fix: tighten to `warpgroup_wait<1>` — the smallest sufficient wait depth.

**FP16 result:** ~+4% over the then-current kernel. Not a dramatic number, but it establishes that the pipeline has sync-induced bubbles worth recovering.

**E4M3 result (iter023, 811 TFLOPS):** Combines software pipelining with fine-grained waits. The profile pointed at sync bubbles between consumer warpgroup batches — narrowing wait granularity recovered cycles without changing tile shape. Up from 772 TFLOPS (+5%).

Before you widen tiles or add stages, check that warpgroup-level waits are not over-serialized. It is the cheapest thing to fix.

### MMA Batch Configuration

Hopper WGMMA splits work across batches of K fragments. A poor split underfeeds tensor cores relative to operand delivery. Flags like `--wgmma-split-batch` reshape how batches map to instructions.

**FP16 result:** ~+5% from batch reshaping. If Nsight shows WGMMA issue slots gapping while shared memory is ready, revisit batching before you blame TMA.

### Early Empty, Merged Barriers, Early Arrive (E4M3)

Async pipelines use empty/full phases. Late signals or over-synchronized barriers steal overlap.

| Iteration | Change | TFLOPS | Δ |
| --------- | ------ | ------ | - |
| iter016 | Early empty + merged barrier | 772 | +15% vs baseline |
| iter068 | Early empty **arrive** | 1127 | +68% vs baseline |

Above ~900 TFLOPS on E4M3, this kind of sync polish is worth double-digit TFLOPS. Use Nsight warp stall views filtered on barrier-related reasons to find these.

## Step 2: Metadata Delivery

This is where sparse GEMM diverges from dense. Metadata is a second operand plane that you must keep fed alongside the matrix data.

### Read-Only Cache Path (`__ldg`)

Metadata lives in global memory and is touched every K tile. Scalar loads that miss behave like pointer chasing next to wide TMA. Forcing a read-only, L2-friendly path with `__ldg`-style loads:

**FP16 result:** ~+0.5%. The baseline already cached somewhat, but consistency across tiles mattered.

### Vectorization and Hoisting

Small metadata arrays per tile still need 128B-line discipline to avoid L2 thrash across CTAs. Three changes that form one story — how metadata reaches registers before MMA consumes it:

| Change | FP16 Δ | What it does |
| ------ | ------ | ------------ |
| L2/128B-oriented grouping | +0.7% | Align metadata to cache line boundaries |
| `uint2` metadata vectorization | +8% | Load 2× metadata per instruction |
| Hoisted `__ldg` metadata | +7% | Move metadata loads before the K inner loop |

These percents are local (each step vs the previous edit). They do not multiply cleanly to the full 368 → 655 because interactions matter — hoisting weighs more after vectorization.

### TMA Metadata Staging

The strongest move: put metadata on the **same async machinery** as operands. Instead of scalar loads inside the K loop, let TMA prefetch metadata tiles into shared memory on the producer side.

**E4M3 (iter001, 759 TFLOPS):** This is the headline jump — TMA metadata staging from day one. +13% over the 671 baseline.

**FP16 (iter143, 655 TFLOPS):** TMA-backed metadata arrives as part of the final bundle alongside TK128 and split RHS TMA. It was harder to introduce on FP16 because the compiler-generated `.co` schedule could not express the TMA descriptor work — that is the `.co` vs `.cu` story from the [AI-tune page](aitune-last-mile.md).

### Risks You Own

- Wrong metadata for a repacked operand is a **silent numerical bug**. Run host checks against a dense reference on small sizes when you change load paths.
- When TK changes, diff metadata offsets and fragment boundaries.
- 3-stage can overflow SMEM — print shared usage and watch the occupancy cliff.
- TK128 without matching TMA descriptors invites bank conflicts; diff descriptor setup against TK64.

## Step 3: 1p2c and Pipeline Depth

### Why 1p2c Helps

In 1p1c, one producer warpgroup issues all TMA and often absorbs setup work that steals issue slots from the consumer's steady `wgmma` stream. 1p2c adds a second consumer warpgroup — more math capacity to keep up with the data delivery.

**FP16 (iter120, 434 TFLOPS):** Best `.co` outcome — 1p2c + 3-stage. A ~+9% jump vs the prior step in the chain.

**E4M3 (iter036, 897 TFLOPS):** 1p2c alone, before the 3-stage change. +34% over baseline.

### Pipeline Depth: The 3-Stage Discontinuity

Adding a third operand buffer slot lets the producer run ahead of the consumer, hiding TMA and metadata latency behind math.

**E4M3 (iter040, 1090 TFLOPS):** This is the breakthrough — +62% vs the 671 baseline. The move into >1000 TFLOPS territory. This is not gradual improvement; it is a **discontinuity** that says the pipeline went from "producer regularly stalls" to "producer stays ahead."

**FP16 (iter120):** 3-stage pairs with 1p2c at the same step. The combined change was justified because producer staging and metadata path were already compatible.

### SMEM and Occupancy

Pushing from two to three stages increases SMEM footprint. If occupancy collapses, math gains vanish. Watch shared usage prints and profiler active-warps counts. Stop pursuing deeper pipelines when:

- SMEM exceeds what your cluster config allows
- Three independent mutations in the same family no longer move TFLOPS

Then borrow ideas from the other dtype's milestones — if FP16 is metadata-stuck, study iter001 on E4M3 for TMA metadata layout.

## Step 4: Inner Loop, Epilogue, and Tile Geometry

### Store Path (`stmatrix`)

Accumulators exit through shared or registers with bank-safe stores. `stmatrix` where the toolchain supports it aligned store paths with Hopper preferences:

**FP16 result:** ~+2%. When the kernel is math- and sync-bound above 1000 TFLOPS on E4M3, epilogue tricks matter less unless the profile says store is hot.

### Inner Unroll and FTZ (FP16 iter137, 543 TFLOPS)

Compiler-generated `.co` schedules may not unroll the inner K loop enough to overlap address math, metadata prefetch, and `wgmma`. Hand `.cu` at iter137 used **unroll 24** plus **FTZ** (flush-to-zero) to cut denorm edge cases:

```
# Key changes in iter137 hand-written .cu:
- Inner K loop unrolled 24× (vs compiler default)
- FTZ enabled to avoid denorm penalties
- Result: 543 TFLOPS (+47% vs baseline)
```

This is the strongest "organic" `.cu` before iter143. Inner unroll applies philosophically to E4M3; FTZ is less central when inputs are E4M3 and the accumulator path is already constrained.

### TK128, TMA Metadata, Split RHS TMA (FP16 iter143, 655 TFLOPS)

TK64 keeps K tiles small, which inflates trip count and metadata traffic per unit work. iter143 combines three structural changes:

| Change | Purpose |
| ------ | ------- |
| TK128 | Halve K-loop trip count, amortize inner-loop overhead |
| TMA metadata | Metadata on the async plane alongside operands |
| Split RHS TMA | Bandwidth tracks consumer demand on the RHS side |

**Result:** 655 TFLOPS — +78% vs baseline. This is not a polishing pass; it is structural memory-system work that the `.co` compiler could not express.

E4M3 already used 128/128 swizzle from the baseline. The parallel is iter001 metadata plus iter040 depth, not a literal copy of every FP16 knob. Do not copy FP16 swizzle 64 onto E4M3 128/128 without validation — bank conflict behavior changes.

## What Transferred Between Dtypes

Copy **causal structure**, not parameter equality:

| Pattern | FP16 | E4M3 |
| ------- | ---- | ---- |
| Metadata on TMA plane | iter143 | iter001 |
| Fine warpgroup sync | early chain | iter023 |
| 1p2c | iter120 | iter036 |
| 3-stage depth | iter120 (bundled) | iter040 |
| Barrier micro-optimization | secondary | iter016, iter068 |

## Closing Checklist

Before you end a tuning session, verify:

- [ ] Metadata vectorized, hoisted, and ideally TMA-staged
- [ ] Stage count matched to producer rate with 1p2c justified by profiled slack
- [ ] Warpgroup waits and empty/full signaling minimal without races
- [ ] TK changes forced a joint pass on swizzle, TMA, and metadata
- [ ] You know where `.co` plateaus so you do not burn days on micro-tweaks that need `.cu` expressiveness

If FP16 is stuck below ~450 TFLOPS, the logs suggest attacking metadata vectorization, `__ldg`, 1p2c + 3-stage, and `warpgroup_wait` first. If E4M3 is already ~850+ TFLOPS, barrier/early-empty/arrive and stage tuning often beat more operand widening.

Next: [AI-tune last mile](aitune-last-mile.md) — the `.co` ceiling, the `.cu` breakthrough, and workflow for each dtype.
