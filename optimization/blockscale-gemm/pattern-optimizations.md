# Optimization Patterns: 397.9 → 621 TFLOPS

Four steps, each addressing a different bottleneck. The ladder builds on the warp-specialized 1p1c template from [baseline analysis](baseline-analysis.md).

| Step | Kernel | TFLOPS @2k | TFLOPS @4k | Primary lever |
| ---- | ------ | ---------- | ---------- | ------------- |
| 0 | Baseline (M64N128K32) | 314.2 | 397.9 | — |
| 1 | iter049 | **380** | — | TMA overlap |
| 2 | iter051 | 372 | 602 | N256 WGMMA |
| 3 | iter053 | — | 610 | L2 256B promotion |
| 4 | **iter066** | — | **621** | **Prefetch scale_a** |

## Step 1: TMA Overlap After WGMMA (iter049)

**The problem.** In the baseline, the consumer finishes WGMMA, does `scale_accumulator` work, and only then starts the next K-tile's TMA. The TMA pipeline sits idle across that handoff — scale accumulation and TMA are serialized when they could overlap.

**The change.** Issue the next K-block's TMA loads as soon as the WGMMA wait completes, so memory latency hides behind scale-related math that does not need the new operands yet.

This is the same scheduling instinct as [Chapter 6 (synchronization)](../../tutorial/ch06-synchronization.md): move independent work so the longest-latency piece (TMA) starts earlier. Block-scaled GEMM adds `scale_accumulator` as a third phase beside load and MMA — iter049 shows that phase can share the bubble with TMA.

**Result:** **380 TFLOPS** at 2048³ (+21% over 314.2). Still in the M64N128K32 tile class — no structural geometry change, just better scheduling within the existing tile.

## Step 2: N256 WGMMA — Double the Math Per Tile (iter051)

**The problem.** N128 tiles finish K-pipeline steps quickly but launch many CTAs along N. On large N, wave quantization and per-CTA overhead hurt. Each CTA does relatively little math before the grid overhead (launch, synchronization, epilogue) cuts in.

**The change.** Move to M64N256K32 — double the N extent of the WGMMA tile per CTA.

Let's check the SMEM impact:

```
Operand staging (N256):
  LHS: WM × TK × sizeof(e4m3) = 64 × 128 × 1B = 8 KB
  RHS: WN × TK × sizeof(e4m3) = 256 × 128 × 1B = 32 KB
  Total ≈ 40 KB per stage
```

At ~40 KB, this is workable on Hopper but reduces headroom for extra pipeline stages.

**Result:** **602 TFLOPS** at 4096³ (+51% over baseline). But 2048³ drops to **372 TFLOPS** (vs 380 on iter049) — fewer blocks cover N, the grid is coarser, and occupancy trades differently at the smaller size.

This is the same WN tradeoff as in the dense FP16 case: more math per block, fewer blocks, heavier SMEM. For block-scaled GEMM, RHS and `scale_rhs` footprint grows with N — you need enough SMEM headroom if the pipeline stages TMA.

### The Size-Dependent Tradeoff

| | iter049 (N128) | iter051 (N256) |
|---|----------------|----------------|
| 2048³ | **380** | 372 |
| 4096³ | — | **602** |

N256 trades small-cube grid density for large-cube throughput. If your workload is always 2048³, iter049 wins. For 4096³ and above, N256 is the clear choice.

## Step 3: L2 Promotion on RHS TMA (iter053)

**The problem.** At 4096³, RHS panels are large. TMA traffic does not always stick in L2 the way you want — lines get evicted before they can be reused across K iterations or neighboring CTAs.

**The change.** Set `CU_TENSOR_MAP_L2_PROMOTION_L2_256B` on the RHS tensor map. This Hopper cache hint promotes lines into L2 with a 256B granularity policy.

**Result:** **610 TFLOPS** at 4096³ (+8 TFLOPS over iter051, +53% over baseline).

You are not changing the math — you are nudging the memory system. The hint biases the cache toward reuse of RHS data across K iterations and CTAs that share spatial locality. It pairs naturally with N256, which increased per-CTA RHS volume and made L2 behavior matter more.

This is a percentage-point gain on an already strong kernel. But it is also nearly free: one flag on the TMA descriptor, zero change to the compute path. When you have already tuned tile geometry and scheduling, cache hints are the next lever.

## Step 4: Prefetch `scale_a` Before WGMMA (iter066)

**The problem.** Scale loads inside a tight WGMMA loop with short issue interval can stall the consumer. The per-row `scale_a` loads are latency-bound — they compete with WGMMA for issue slots and cannot overlap with anything if they sit in the critical path.

**The change.** Prefetch per-row `scale_a` into registers **before** the inner WGMMA body, so load latency hides behind independent setup or prior WGMMA work.

**Result:** **621 TFLOPS** at 4096³ — **+56%** vs the 397.9 baseline.

By this point, the kernel is already doing heavy WGMMA and wide N. The remaining slack sits in operand latency. Block-scaled GEMM makes scales first-class operands — treat them like any other latency-bound input. Software prefetch, double-buffering, or DMA-to-SMEM (see the `v2` variants in [baseline analysis](baseline-analysis.md)) are all design axes when registers or scheduling bind.

## Source Variants Under `blockscale_gemm_v2/`

The `v2` folder explores alternative scale movement strategies:

| Variant | Approach |
| ------- | -------- |
| `rhs_scale_dma_smem` / `scale_dma_smem` | Stage scales via TMA into shared memory |
| `transposed_scale` | Change scale layout for coalescing vs index cost |
| `tileN` | Tile along N explicitly in Croktile structure |
| `..._warpspec_persis_1p1c.co` | Persistent kernel variant ([Ch5](../../tutorial/ch05-branch-control.md)) |

These align with the theme that scale DMA is an alternative when register pressure or load scheduling hurts WGMMA issue interval.

## Compile and Run

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co \
  -o /tmp/bs.cute.result && bash /tmp/bs.cute.result --execute
```

Pre-generated iter049/051/053/066 trees ship with `run.sh` for bit-identical reproduction. Full iteration history: `ai-tune/2026-03-22/blockscale_gemm_v2` (71 iterations).

## Takeaways

1. **Scale scheduling matters as much as tile size.** iter049 (+21%) shows that TMA vs `scale_accumulator` overlap is a first-order concern, not a micro-optimization.
2. **N256 is the large-cube win** — iter051 (+51% @4k). It costs at small cubes.
3. **L2 promotion and scale prefetch are late percentage gains** on an already strong kernel — iter053 and iter066 target memory hierarchy and operand latency, not raw WGMMA width.
4. **Scale DMA variants** (`blockscale_gemm_v2/`) document alternative scale movement for future tuning when register or layout limits bind.

The arc is the same as in dense and sparse: schedule first, widen second, cache-tune third. Block scaling adds scale tensors as first-class operands that need the same latency-hiding discipline as matrix data.
