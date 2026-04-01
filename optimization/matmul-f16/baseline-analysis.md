# Baseline: 208.7 TFLOPS and Why That Leaves a Large Gap

Before we change anything, we need to understand where the baseline sits and what "fast" even means on this hardware. The numbers come from the 2026-03-23 AI-tune log; the headline problem is **8192³**.

## The Baseline Kernel

The starting point is `matmul_f16_dyn_sm90.co` with these parameters:

- **1p1c** warp specialization — one TMA producer warpgroup, one WGMMA consumer warpgroup (the roles from [Chapter 5](../../tutorial/ch05-branch-control.md))
- **WN = 128** (`MATMUL_WARP_N`) — the N extent of the WGMMA tile
- **STAGES = 4** — four operand ring slots along K (the pipelining idea from [Chapter 6](../../tutorial/ch06-synchronization.md))
- Operand layout with swizzle matching [Chapter 7](../../tutorial/ch07-advanced-movement.md)

Compile and run:

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/matmul/matmul_f16_dyn_sm90.co \
  -o /tmp/matmul.cute.result && bash /tmp/matmul.cute.result --execute
```

Result: **208.7 TFLOPS** at 8192³.

## Lower Bounding the Fastest Possible Runtime

For a square GEMM of side S = 8192:

- **Total FLOPs:** `2 × S³ = 2 × 8192³ ≈ 1.1 TFLOP` (each output element does S multiply-adds, counting as 2 FLOPs each)
- **Total data to read (minimum):** `3 × S² × 2B = 3 × 8192² × 2B ≈ 384 MB` (two FP16 input matrices + one output, assuming perfect reuse)
- **Total data to store:** `S² × 2B ≈ 128 MB`

The H800 PCIe is advertised with:

| Resource | Peak |
| -------- | ---- |
| FP16 tensor core throughput | ~1513 TFLOPS |
| Global memory bandwidth | ~3.35 TB/s (HBM3) |

If we could hit peak tensor throughput, the 1.1 TFLOP calculation would take **~0.7 ms**. The memory transfers (384 + 128 = 512 MB) would take **~0.15 ms** at peak bandwidth. So the kernel is firmly **compute-bound** in the ideal case — we need ~5x more time for compute than for memory.

But **cuBLAS** on this stack only reaches ~380 TFLOPS, which is ~25% of the theoretical tensor peak. That gap between 1513 and 380 reflects real-world overhead: scheduling, synchronization, pipeline bubbles, occupancy limits, and instruction mix. So 380 TFLOPS — not 1513 — is our practical target.

## Why 208.7 Is Schedule-Bound, Not Broken

Our baseline sits at **55% of cuBLAS**. The kernel is not broken — it uses TMA, WGMMA, and warp specialization correctly. It is **under-scheduled**: the tile width, stage depth, and output staging do not match the problem size's occupancy and contention characteristics.

You can see this without a single profiler counter by combining three observations:

**1. Throughput vs problem size.** Running the same kernel at different sizes:

| Size | TFLOPS | Notes |
| ---- | ------ | ----- |
| 2048³ | ~204 | Small grid, inner-loop dominated |
| 4096³ | ~206 | Similar — inner-loop dominates |
| 8192³ | 208.7 | Slightly better wave utilization |

Consistent ~204–208 across sizes suggests the bottleneck is inside the block (pipeline/scheduling), not at the grid level (wave quantization).

**2. Shared memory vs occupancy.** Each stage multiplies operand staging. With WN=128 and 4 stages, the SMEM footprint determines how many CTAs an SM can hold simultaneously. On Hopper, the per-SM shared budget is **228 KB**. If your per-block SMEM pushes past the threshold for 2 CTAs/SM down to 1 CTA/SM, you lose half your latency hiding — a step-function loss.

**3. Role balance in 1p1c.** With one producer and one consumer, if the consumer is stuck on `wait full` (waiting for TMA data) or the producer on `wait empty` (waiting for the consumer to free a buffer), you have a bubble-limited pipeline. Warp specialization assigns roles; it does not by itself size the ring or tile to eliminate bubbles.

## Occupancy Arithmetic

Let's do the occupancy calculation that will guide our optimizations. The H800 SM90a has:

| Resource | Per SM |
| -------- | ------ |
| Max warps | 64 |
| Max threads | 2048 |
| Max shared memory | 228 KB |
| Max registers | 65536 |

With WN=128, STAGES=4, and the operand staging for FP16 TMA:

```
SMEM per block ≈ STAGES × (WM × TK + WN × TK) × 2B
               = 4 × (64 × 64 + 128 × 64) × 2
               = 4 × 12288 × 2
               ≈ 96 KB
```

At 96 KB per block, 228 KB allows **2 blocks per SM** — but just barely. Any increase in SMEM (wider WN, more stages, output staging area) can tip this to 1 block/SM.

The occupancy story is what makes every subsequent optimization interact: changing WN affects SMEM, which affects how many blocks fit, which determines how well we hide latency. This is why the optimization path is a dependency chain, not a free-form search.

## What Early Experiments Show

Two quick experiments at 2048³ already hint at the structure of the problem:

**iter004:** WN=256, STAGES=2 → **208.9 TFLOPS.** Fewer stages shrink the operand ring, freeing SMEM for the wider tile. But the wider tile does not help — the pipeline is too shallow to hide TMA latency.

**iter023:** Same geometry + `--ptx-barrier`, `--stmatrix`, subspan refinements → **214.3 TFLOPS (+5%).** This is compiler and operand-path quality on top of the same Croktile structure, not a new algorithm. It shows that lowering matters, but cannot fix a schedule-bound kernel.

## Matching the Baseline

The teaching kernel `matmul_f16_dyn_sm90_warpspec_1p1c.co` is the clearest 1p1c illustration, not the throughput champion. For apples-to-apples against the tune artifacts, use `matmul_f16_dyn_sm90.co` for the 208.7 configuration.

If your number drifts from 208.7, check:

- GPU clocks (fixed or thermal throttling?)
- Timing without accidental verification overhead
- M, N, K are really 8192 (grid shape is sensitive)
- `CROKTILE_TIMING_WARMUP=10`, `CROKTILE_TIMING_REPEAT=500`

The vocabulary for what follows: `MATMUL_WARP_M` / `MATMUL_WARP_N`, `MATMUL_TILE_K`, `MATMUL_STAGES`, `MATMUL_SWIZ`. Change any of them and recompute shared memory before you interpret TFLOPS — a small WN bump can land on the wrong side of the 228 KB budget.

Next: [Optimization patterns](pattern-optimizations.md) — the step-by-step path from 208.7 to 382.5 TFLOPS.
