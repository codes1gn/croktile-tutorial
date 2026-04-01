# Optimization Patterns: 208.7 → 382.5 TFLOPS

The through-line is one rule: change the thing the numbers tell you is slow, then re-measure. Here is the sequence that closed the gap to cuBLAS, grouped by what each step actually fixed.

## Step 1: Tile Geometry — WN=176, STAGES=2 (iter046)

**The problem.** The baseline uses WN=128 with 4 stages. Four stages means a large operand ring that eats SMEM, leaving little room for concurrent CTAs. At 2048³, this gives ~204 TFLOPS.

**The change.** Widen the N tile to 176 (more math per staged K-slab) and drop to 2 stages to keep SMEM in budget:

```
MATMUL_WARP_N = 176    # was 128
MATMUL_STAGES = 2      # was 4
```

Let's calculate the new SMEM footprint:

```
SMEM ≈ STAGES × (WM × TK + WN × TK) × 2B
     = 2 × (64 × 64 + 176 × 64) × 2
     = 2 × 15360 × 2
     ≈ 60 KB
```

At 60 KB per block, the SM can hold **3 blocks** within its 228 KB budget — up from 2 with the baseline's ~96 KB. More concurrent blocks means better latency hiding across CTAs.

**Result:** **242 TFLOPS** at 2048³ (+18% over baseline). The wider tile does more math per loaded panel, and the smaller ring freed occupancy. But 2 stages leave the pipeline shallow — TMA latency is not fully hidden.

### Why WN Matters: Arithmetic Intensity

The intuition is the same as in CPU matmul optimization. A wider N tile means each thread block computes more output elements per byte loaded from GMEM into SMEM. With WN=128 and TK=64:

```
Bytes loaded per K-slab: (WM + WN) × TK × 2B = (64 + 128) × 64 × 2 = 24,576 B
FLOPs per K-slab:        2 × WM × WN × TK = 2 × 64 × 128 × 64 = 1,048,576
Arithmetic intensity:    1,048,576 / 24,576 ≈ 42.7 FLOPs/B
```

With WN=176:

```
Bytes loaded per K-slab: (64 + 176) × 64 × 2 = 30,720 B
FLOPs per K-slab:        2 × 64 × 176 × 64 = 1,441,792
Arithmetic intensity:    1,441,792 / 30,720 ≈ 46.9 FLOPs/B
```

The wider tile increased arithmetic intensity by ~10%, but the bigger win was freeing SMEM for occupancy.

## Step 2: Pipeline Depth — WN=176, STAGES=3 (iter048)

**The problem.** At 2 stages, the producer finishes loading the next K-slab and stalls on `wait empty` — the consumer has not freed the previous buffer yet. The pipeline has a bubble.

**The change.** Add one more stage:

```
MATMUL_STAGES = 3      # was 2
```

New SMEM:

```
SMEM ≈ 3 × (64 × 64 + 176 × 64) × 2
     = 3 × 15360 × 2
     ≈ 90 KB
```

At 90 KB, 228 KB still fits **2 blocks per SM**. The extra stage lets the producer run one K-slab ahead of the consumer, hiding TMA latency behind WGMMA compute.

**Result:** **354.1 TFLOPS** at 2048³ — a **+46%** jump from the previous step.

This is not linear scaling from one extra buffer. It is the signature of a **bubble-limited** schedule: the extra stage bought producer-consumer concurrency, not more math. The pipeline went from "producer stalls every iteration" to "producer stays ahead."

### The Catch: Stages × Problem Size Interaction

Three stages help at 2048³ but can hurt at 8192³. The larger grid amplifies occupancy effects — each extra stage is bytes that could evict concurrent blocks. When you change problem size by 4×, re-sweep STAGES. This is why later steps revisit the WN/STAGES balance for the big cube.

## Step 3: Split-Output 1p2c (iter050)

**The problem.** With a single consumer warpgroup (1p1c), there is one `output_s` tile in shared memory for accumulator staging. As WN grows, output contention becomes the bottleneck — the consumer serializes on writing to this shared tile, and SMEM traffic on the accumulator path eats into throughput.

**The change.** Switch to **1p2c split-output**: one producer, two consumer warpgroups, each with a private slice of the output staging area. The Croktile surface for this lives in `matmul_f16_dyn_sm90_warpspec_1p2c.co`.

This trades slightly higher SMEM (two output slices instead of one) for less contention and a better instruction mix. The parameters for validation at 4096³:

```
Warp spec:   1p2c split-output
MATMUL_WARP_N = 128
MATMUL_STAGES = 2
```

**Result:** **~375 TFLOPS** at 4096³.

If split-output had regressed at 4096³, the pattern would not have been trusted at 8192³. Validating at an intermediate size is part of the method — it is cheaper to discover that output staging hurts before committing to the expensive large-cube experiment.

### How to Spot Output Contention

You rarely see output contention in a single profiler counter. The heuristic that correlated in this study: TFLOPS rose when moving from 1p1c to 1p2c **only** with split-output enabled — implying the consumer side was serialized on `output_s` traffic, not on the math path.

## Step 4: The Best Headline — iter057

**The change.** Carry split-output to the full 8192³ problem with tuned WN and non-persistent launch:

```
Warp spec:   1p2c split-output
MATMUL_WARP_N = 152
Launch:      non-persistent (conventional grid)
```

**Result:** **382.5 TFLOPS** at 8192³ — **+83%** over the 208.7 baseline, matching cuBLAS.

[Chapter 5](../../tutorial/ch05-branch-control.md) covers persistent kernels that fix grid-level tail underuse. But when inner-block SMEM and pipeline choices already cap throughput, persistence cannot recover what occupancy lost. At 8192³ with the split-output tile, wave quantization was acceptable and the inner block was already the bottleneck — a conventional grid won.

## Step 5: WN Sweep and K-Unroll — iter061

**The problem.** After split-output, the question becomes: is WN=152 actually optimal for 8192³, or did we inherit it from smaller-cube experiments?

**The change.** Phase 3 swept WN at 8192³ with K-unroll and `--wgmma-wait-depth`:

```
MATMUL_WARP_N = 160
K-unroll:    enabled
--wgmma-wait-depth=N (tuned to match stage count)
```

**Result:** **380.6 TFLOPS** at 8192³ — 1.9 TFLOPS below iter057 at the big cube, but a stronger cross-size story (100.5% of cuBLAS at 2048³, 80.7% at 8192³ when measured against a different cuBLAS baseline).

### The WN=168 Occupancy Cliff

The sweep also found a **hard failure**: WN=168 pushed shared memory past 228 KB, forcing **1 CTA per SM** instead of 2. Throughput fell off a cliff — not a few percent, but a catastrophic loss of latency hiding.

```
WN=160: SMEM ≈ 114.7 KB → 2 CTAs/SM ✓ → 380.6 TFLOPS
WN=168: SMEM >  228  KB → 1 CTA/SM  ✗ → significant regression
```

You catch this kind of threshold by computing `STAGES × tile_dimensions × element_size` and comparing against the per-SM budget, not by guessing from WN alone.

## Compiler Flags: The Last Layer

With function structure settled, how the compiler lowers Croktile to PTX matters. The shipped builds share a common flag bundle:

| Flag | Purpose |
| ---- | ------- |
| `--use-warpspec` | Warp-specialized codegen for producer/consumer split |
| `--stmatrix` | STSM-style shared-memory matrix setup |
| `--hoist-offset` / `--hoist-scale` | Hoist address arithmetic and scale factors out of inner loops |
| `--ptx-barrier` | Barrier instructions for async producer/consumer sync |
| `--tma-cluster-aware` | Bias TMA lowering for cluster and multicast on SM90 |
| `--wgmma-wait-depth=N` | Expose WGMMA pipeline wait depth as a tunable |

**iter023** showed flags matter: +5% at 2048³ from `--ptx-barrier` and `--stmatrix`. But the lesson from the full log is **order of operations**: freeze flags while sweeping WN and STAGES, then unfreeze only after split-output lands. Over-tuning flags while SMEM is on the wrong side of 228 KB is a common failure mode.

## The Arc from Baseline to Best

The optimization followed a dependency chain, not a free-form search:

1. **Baseline** (208.7) — correct roles but wrong balance of stages and WN for the problem's occupancy.
2. **Step 1** — tune WN and STAGES jointly at 2048³ → modest gains (242).
3. **Step 2** — add one pipeline stage → large jump from bubble elimination (354.1).
4. **Step 3–4** — split-output 1p2c removes accumulator contention → cuBLAS-class throughput (375 → 382.5).
5. **Step 5** — WN sweep at 8192³ discovers the occupancy cliff, settles on 380.6 as the robust choice.

The largest single structural win was not a compiler flag — it was **1p2c split-output** moving TFLOPS into the 370–382 band. Flags like `--stmatrix` matter, but they cannot recover serialization on `output_s` if two consumers share one accumulator tile. When you face a similar ceiling in your own kernel, check whether the output path is the bottleneck before reaching for instruction-level levers.

Next: [AI-tune last mile](aitune-last-mile.md) — shipped checkpoints, exact repro commands, and the full timeline.
