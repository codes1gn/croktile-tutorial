# How to Optimize a Croktile FP16 GEMM for cuBLAS-like Performance: a Worklog

In this post, I'll iteratively optimize a Hopper (SM90a) half-precision matrix multiply written in Croktile. My goal is not to build a cuBLAS replacement, but to deeply understand the performance characteristics of H800 GPUs through the lens of Croktile's abstractions — warp specialization, TMA pipelining, tile geometry, and compiler-flag tuning. You can find all kernel sources under `benchmark/performance/matmul/` in the Croktile repository.

Matrix multiplication on GPUs may currently be the most important algorithm that exists, considering it makes up almost all the FLOPs during training and inference of large deep-learning models. So how much work is it to push a correct Croktile SGEMM from "it runs" to "it matches cuBLAS"? Starting from a baseline and step-by-step applying optimizations, we get within 101% of cuBLAS:

| Step | Kernel | TFLOPS @8192³ | vs cuBLAS (~380) |
| ---- | ------ | ------------- | ---------------- |
| 0 | Baseline: 1p1c, WN=128, 4-stage | 208.7 | 55% |
| 1 | Tile geometry: WN=176, STAGES=2 | 242.0 | 64% |
| 2 | Pipeline depth: WN=176, STAGES=3 | 354.1 | 93% |
| 3 | Split-output 1p2c, WN=128 | ~375.0 | 99% |
| 4 | Split-output 1p2c, WN=152, non-persistent | **382.5** | **101%** |
| 5 | WN=160, K-unroll, wgmma-wait-depth | 380.6 | 100% |

## Step 0: The Baseline

In Croktile's programming model, a matmul kernel is a `__co__` function that describes operand flow through TMA, shared memory staging, and WGMMA accumulation. The Croktile compiler transpiles this into Hopper-native PTX with warp specialization, pipelining, and swizzled addressing. The important knobs are:

- `MATMUL_WARP_N` — the N extent of the WGMMA tile (how wide each block's output is)
- `MATMUL_STAGES` — operand ring slots along K (how deep the async pipeline is)
- Warp specialization mode — 1p1c (one producer, one consumer) or 1p2c

The baseline kernel `matmul_f16_dyn_sm90.co` uses **1p1c** warp specialization (one TMA producer warpgroup, one WGMMA consumer warpgroup — roles from [Chapter 5](../tutorial/ch05-branch-control.md)), **WN=128**, and **4 pipeline stages** (the pipelining idea from [Chapter 6](../tutorial/ch06-synchronization.md)). Compile and run:

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/matmul/matmul_f16_dyn_sm90.co \
  -o /tmp/matmul.cute.result && bash /tmp/matmul.cute.result --execute
```

Result: **208.7 TFLOPS** at 8192³. Not bad in the abstract — but far from what the hardware can deliver.

![Baseline 1p1c pipeline with 4-stage ring](images/BaselineKernel_ManimCE_v0.19.1_dark.png#only-dark)
![Baseline 1p1c pipeline with 4-stage ring](images/BaselineKernel_ManimCE_v0.19.1_light.png#only-light)

### Lower Bounding the Fastest Possible Runtime

For a square GEMM of side S = 8192, total work is `2 × S³ ≈ 1.1 TFLOP` (each output element does S multiply-adds, which counts as 2 FLOPs each — one multiply, one add, usually fused into a single FMA instruction). The H800 PCIe advertises ~1513 TFLOPS FP16 tensor peak and ~3.35 TB/s HBM3 bandwidth.

If we hit peak tensor throughput, the calculation takes ~0.7 ms. The minimum memory transfers (two FP16 input matrices + one output, assuming perfect reuse: `3 × 8192² × 2B ≈ 384 MB`) take ~0.15 ms at peak bandwidth. So this kernel is firmly **compute-bound** in the ideal case — we need ~5× more time for compute than memory.

But cuBLAS only reaches ~380 TFLOPS on this stack, about 25% of the theoretical tensor peak. That gap reflects real-world overhead: scheduling, synchronization, pipeline bubbles, occupancy, instruction mix. So **380 TFLOPS** — not 1513 — is our practical target, and our baseline is at **55%** of it.

### Why 208.7 Is Schedule-Bound, Not Broken

The kernel uses TMA, WGMMA, and warp specialization correctly. It is **under-scheduled**: tile width, stage depth, and output staging do not match this problem size's occupancy and contention profile.

You can see this without a profiler by combining three observations. First, **throughput vs problem size**: the same kernel gives ~204 TFLOPS at 2048³ and 208.7 at 8192³ — consistent numbers suggest the bottleneck is inside the block (scheduling), not at the grid level (wave quantization). Second, **shared memory vs occupancy**: with WN=128 and 4 stages, SMEM footprint is about 96 KB per block — Hopper's 228 KB per-SM budget allows 2 CTAs/SM, but just barely. Third, **role balance in 1p1c**: with one producer and one consumer, if the consumer stalls on `wait_full` or the producer on `wait_empty`, you have a bubble-limited pipeline. Warp specialization assigns roles; it does not by itself size the ring to eliminate bubbles.

### Occupancy Arithmetic

This calculation will guide every subsequent optimization:

```
SMEM per block ≈ STAGES × (WM × TK + WN × TK) × sizeof(fp16)
               = 4 × (64 × 64 + 128 × 64) × 2B
               = 4 × 12288 × 2 ≈ 96 KB
```

At 96 KB, the SM's 228 KB budget fits **2 blocks**. Any increase in SMEM — wider WN, more stages, output staging — can tip this to 1 block/SM. That is a step-function loss of latency hiding, not a gentle degradation. Every optimization below interacts through this arithmetic.

---

## Step 1: Tile Geometry — WN=176, STAGES=2

**The problem.** Four stages eat SMEM, leaving little room for concurrent CTAs. The pipeline is correctly structured but oversized for the occupancy budget.

**The change.** Widen the N tile to 176 (more math per staged K-slab) and drop to 2 stages:

```
MATMUL_WARP_N = 176    # was 128
MATMUL_STAGES = 2      # was 4
```

New SMEM:

```
SMEM ≈ 2 × (64 × 64 + 176 × 64) × 2B
     = 2 × 15360 × 2 ≈ 60 KB
```

At 60 KB per block, the SM can hold **3 blocks** — up from 2. More concurrent blocks means better latency hiding across CTAs.

**Why WN matters — arithmetic intensity.** A wider N tile means each block computes more output elements per byte loaded from GMEM into SMEM. With WN=128: `AI = 2 × 64 × 128 × 64 / ((64 + 128) × 64 × 2) ≈ 42.7 FLOPs/B`. With WN=176: `AI ≈ 46.9 FLOPs/B`. The 10% intensity gain is useful, but the bigger win was freeing SMEM for occupancy.

**Result:** **242 TFLOPS** at 2048³ (+18%). But 2 stages leave the pipeline shallow — TMA latency is not fully hidden.

---

## Step 2: Pipeline Depth — STAGES=3

**The problem.** At 2 stages, the producer finishes loading the next K-slab and stalls on `wait_empty` — the consumer has not freed the previous buffer yet.

**The change.** Add one more stage:

```
MATMUL_STAGES = 3      # was 2, keeping WN=176
```

New SMEM: `3 × 15360 × 2 ≈ 90 KB` — still fits 2 blocks in 228 KB.

The extra stage lets the producer run one K-slab ahead of the consumer, hiding TMA latency behind WGMMA compute.

![3-stage pipeline: producer runs ahead](images/Step2ThreeStage_ManimCE_v0.19.1_dark.png#only-dark)
![3-stage pipeline: producer runs ahead](images/Step2ThreeStage_ManimCE_v0.19.1_light.png#only-light)

**Result:** **354.1 TFLOPS** at 2048³ — a **+46%** jump from the previous step.

This is the largest single jump in the entire optimization and it is not from more math — it is the signature of a **bubble-limited** schedule. The extra stage bought producer-consumer concurrency. The pipeline went from "producer stalls every iteration" to "producer stays ahead."

### The catch: stages × problem size interaction

Three stages help at 2048³ but can hurt at 8192³ because the larger grid amplifies occupancy effects. Extra stages are bytes that could evict concurrent blocks. When you change problem size by 4×, re-sweep STAGES. This is why later steps revisit the WN/STAGES balance.

---

## Step 3: Split-Output 1p2c

**The problem.** With a single consumer warpgroup (1p1c), there is one `output_s` tile in shared memory for accumulator staging. As WN grows, **output contention** becomes the bottleneck — the consumer serializes on writing to this shared tile, and SMEM traffic on the accumulator path eats into throughput.

**The change.** Switch to **1p2c split-output**: one producer, two consumer warpgroups, each with a private slice of the output staging area. Source: `matmul_f16_dyn_sm90_warpspec_1p2c.co`.

![Split-output 1p2c architecture](images/SplitOutput1p2c_ManimCE_v0.19.1_dark.png#only-dark)
![Split-output 1p2c architecture](images/SplitOutput1p2c_ManimCE_v0.19.1_light.png#only-light)

This trades slightly higher SMEM (two output slices) for less contention. Validating at 4096³ first — if split-output had regressed at this intermediate size, it would not have been trusted at the expensive 8192³.

**Result:** **~375 TFLOPS** at 4096³.

You rarely see output contention in a single profiler counter. The heuristic that correlated: TFLOPS rose when moving from 1p1c to 1p2c **only** with split-output enabled — implying the consumer side was serialized on `output_s`, not on the math path.

---

## Step 4: The Best Headline — iter057

**The change.** Carry split-output to the full 8192³ problem with tuned WN and non-persistent launch:

```
Warp spec:        1p2c split-output
MATMUL_WARP_N:    152
Launch:           non-persistent (conventional grid)
```

[Chapter 5](../tutorial/ch05-branch-control.md) covers persistent kernels that fix grid-level tail underuse. But when inner-block SMEM and pipeline choices already cap throughput, persistence cannot recover what occupancy lost. At 8192³ with the split-output tile, wave quantization was acceptable and the inner block was already the bottleneck — a conventional grid won.

**Result:** **382.5 TFLOPS** at 8192³ — **+83%** over the 208.7 baseline, matching cuBLAS.

---

## Step 5: WN Sweep and the Occupancy Cliff — iter061

After split-output unlocked cuBLAS-class throughput, the question becomes: is WN=152 optimal for 8192³, or did we inherit it from smaller experiments? Phase 3 swept WN at 8192³ with K-unroll and `--wgmma-wait-depth`:

```
MATMUL_WARP_N:    160
K-unroll:         enabled
--wgmma-wait-depth=N (tuned to match stage count)
```

**Result:** **380.6 TFLOPS** — 1.9 TFLOPS below iter057 at 8192³, but a stronger cross-size story (100.5% of cuBLAS at 2048³).

The sweep also found a **hard failure** at WN=168:

![The WN=168 occupancy cliff](images/OccupancyCliff_ManimCE_v0.19.1_dark.png#only-dark)
![The WN=168 occupancy cliff](images/OccupancyCliff_ManimCE_v0.19.1_light.png#only-light)

At WN=168, shared memory exceeds 228 KB. Residency drops from 2 blocks to 1 block per SM. Throughput falls off a cliff — not a few percent, but a catastrophic loss of latency hiding. You catch this by computing `STAGES × tile_dimensions × element_size` and comparing against the 228 KB budget, not by guessing.

---

## Compiler Flags: The Last Layer

With function structure settled, how the compiler lowers Croktile to PTX matters. The shipped builds share a common flag bundle:

| Flag | Purpose |
| ---- | ------- |
| `--use-warpspec` | Warp-specialized codegen for producer/consumer split |
| `--stmatrix` | STSM-style shared-memory matrix setup |
| `--hoist-offset` / `--hoist-scale` | Hoist address arithmetic out of inner loops |
| `--ptx-barrier` | Barrier instructions for async sync |
| `--tma-cluster-aware` | Bias TMA lowering for SM90 cluster/multicast |
| `--wgmma-wait-depth=N` | Expose WGMMA pipeline wait depth as a tunable |

Flags matter — iter023 showed +5% at 2048³ from `--ptx-barrier` and `--stmatrix` alone. But the lesson from the full log is **order of operations**: freeze flags while sweeping WN and STAGES, unfreeze only after split-output lands. Over-tuning flags while SMEM is wrong is a common failure mode.

---

## Shipped Checkpoints and Reproduction

From the Croktile repo root after `make build`:

```bash
./croktile -gs -t cute -arch=sm_90a \
  --use-warpspec --stmatrix --hoist-offset --hoist-scale \
  --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/<INPUT>.co \
  -o /tmp/run.cute.result && bash /tmp/run.cute.result --execute
```

| Checkpoint | `.co` filename | TFLOPS | Size | Configuration |
| ---------- | -------------- | ------ | ---- | ------------- |
| iter048 | `..._iter048_s3_wn176_best.co` | 354.1 | 2048³ | 1p1c, WN=176, 3-stage |
| iter050 | `..._iter050_1p2c_splitout.co` | ~375 | 4096³ | 1p2c split-output, WN=128 |
| iter057 | `..._iter057_1p2c_so_wn152_nonpersis.co` | **382.5** | 8192³ | 1p2c split-output, WN=152 |
| iter061 | `..._iter061_1p2c_so_wn160_kunroll.co` | 380.6 | 8192³ | 1p2c split-output, WN=160, K-unroll |

**Choosing:** iter057 for peak 8192³ headline. iter061 for one binary that behaves well across sizes (100.5% cuBLAS at 2048³, 80.7% at 8192³).

Harness defaults: `CROKTILE_TIMING_WARMUP=10`, `CROKTILE_TIMING_REPEAT=500`. Build from the same revision as the `.co` file to avoid codegen drift. Use `-arch=sm_90a`. When comparing to external cuBLAS figures, note driver version and clock behavior.

---

## Conclusion

Writing this took ~65 iterations across three phases. Phase 1 spent 38 iterations improving by +5%. Phase 2 spent 14 iterations improving by +83%. Phase 3 refined the last few TFLOPS and discovered the WN=168 failure. Power laws are everywhere.

The largest single structural win was not a compiler flag — it was **1p2c split-output** moving TFLOPS into the 370–382 band. Flags like `--stmatrix` matter, but they cannot recover serialization on `output_s` if two consumers share one accumulator tile. When you face a similar ceiling in your own kernel, check whether the output path is the bottleneck before reaching for instruction-level levers.

The +83% came entirely from Croktile function geometry, output staging, and compiler flags — no mixed precision, no split-K, no CUDA Graph capture.

Full iteration tables: `README_matmul_f16_aitune_2026-03-23.md`. Sources: `matmul_f16_dyn_sm90.co`, `matmul_f16_dyn_sm90_warpspec_1p1c.co`, `matmul_f16_dyn_sm90_warpspec_1p2c.co`, and dated `*_iter048_*` through `*_iter061_*` builds.
