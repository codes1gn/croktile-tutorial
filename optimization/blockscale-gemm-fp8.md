# How to Optimize a Croktile Block-Scaled FP8 GEMM: a Worklog

In this post, I'll optimize an FP8 E4M3 matrix multiply with **per-block scaling** on Hopper (SM90a), measured on H800 PCIe (114 SMs). Operands are E4M3, the accumulator stays in FP16, and along K every 128-element tile carries FP32 scale factors so inner products stay useful after quantization. This is the same family of tricks used in MXFP8-style training: FP8 for density, scales for fidelity.

What makes block-scaled GEMM interesting for optimization is that **scales are first-class operands**. Every K-tile iteration pulls matrix data **and** scale metadata. The Croktile surface expresses this as `mma.row.row.scale` instead of a plain `mma.row.row` — same tiling discipline, extra operands, same pressure to hide TMA latency, but with a third critical path (scale fetch) that does not exist in dense or 2:4 sparse GEMM.

| Step | Kernel | TFLOPS @2k | TFLOPS @4k | Δ vs baseline @4k |
| ---- | ------ | ---------- | ---------- | ----------------- |
| 0 | Baseline: M64N128K32 | 314.2 | 397.9 | — |
| 1 | TMA overlap with scale accumulation (iter049) | **380** | — | +21% @2k |
| 2 | N256 WGMMA (iter051) | 372 | 602 | +51% |
| 3 | N256 + L2 256B promotion (iter053) | — | 610 | +53% |
| 4 | N256 + L2 + prefetch scale_a (iter066) | — | **621** | **+56%** |

## The Baseline Kernel

The baseline `blockscale_gemm_dyn_sm90.co` follows the dense GEMM skeleton — TMA in, WGMMA-shaped loads, accumulator in registers — except the inner MMA is `mma.row.row.scale`, which ties WGMMA execution on E4M3 operands to the scale factors for the active K slice.

![Block scaling concept](images/BlockScaleConcept_ManimCE_v0.19.1_dark.png#only-dark)
![Block scaling concept](images/BlockScaleConcept_ManimCE_v0.19.1_light.png#only-light)

**Baseline control flow, one K-tile at a time:**

1. **TMA** lhs and rhs subspans of shape (WM, TK) and (WN, TK) into SMEM with swizzle matching TK
2. **Inner loop** over WARP_K chunks: MMA load fragments from SMEM, then `mma.row.row.scale` with views into `scale_lhs` and `scale_rhs` indexed by block_m, block_n, and iv_k
3. After all K tiles, **store** accumulator to SMEM and TMA out to global output

The baseline uses M64 × N128 × K32 per warpgroup inner steps, TILE_K=128 (four K32 steps per tile), swizzle 128 on TMA. Readers coming from [Chapter 4 (MMA)](../tutorial/ch04-mma.md) should treat `mma.row.row.scale` as the blockscale analogue of `mma.row.row`.

### Measured baseline

| Shape | TFLOPS | vs 3026 peak |
| ----- | ------ | ------------ |
| 2048³ | 314.2 | 10.4% |
| 4096³ | 397.9 | 13.2% |

13% of FP8 peak is not a bug — block-scaled GEMM issues more global traffic per FLOP than dense (matrix + scales), and `mma.row.row.scale` does not pack identically to the simplest FP8×FP8→FP32 throughput tests. But 13% leaves a large software margin, which is what we are here to close.

---

## Step 1: TMA Overlap with Scale Accumulation — iter049

**The problem.** In the baseline, the consumer finishes WGMMA, does `scale_accumulator` work, and only then starts the next K-tile's TMA. TMA sits idle across the handoff — scale accumulation and TMA are serialized when they could overlap.

![TMA overlap scheduling](images/TMAOverlap_ManimCE_v0.19.1_dark.png#only-dark)
![TMA overlap scheduling](images/TMAOverlap_ManimCE_v0.19.1_light.png#only-light)

**The change.** Issue the next K-block's TMA loads as soon as the WGMMA wait completes, so memory latency hides behind scale-related math that does not need the new operands yet. This is the same instinct as [Chapter 6 (synchronization)](../tutorial/ch06-synchronization.md): move independent work so the longest-latency piece starts earlier.

Block-scaled GEMM adds `scale_accumulator` as a third phase beside load and MMA. iter049 shows that third phase can share the bubble with TMA.

**Result:** **380 TFLOPS** at 2048³ (+21% over 314.2). No structural geometry change — just better scheduling within the existing tile.

---

## Step 2: N256 WGMMA — Double the Math Per Tile — iter051

**The problem.** N128 tiles finish K-pipeline steps quickly but launch many CTAs along N. On large N, wave quantization and per-CTA overhead hurt.

**The change.** Move to M64N256K32 — double the N extent.

![N128 vs N256 tile layout](images/N256VsN128_ManimCE_v0.19.1_dark.png#only-dark)
![N128 vs N256 tile layout](images/N256VsN128_ManimCE_v0.19.1_light.png#only-light)

SMEM impact: `WN=256 → (64 + 256) × 128 × 1B = 40 KB` per stage for operand staging. Workable on Hopper, but reduces headroom for extra pipeline stages.

**Result:** **602 TFLOPS** at 4096³ (+51% over baseline). But 2048³ drops to 372 TFLOPS (vs 380 on iter049) — fewer blocks cover N, the grid is coarser. N256 trades small-cube grid density for large-cube throughput.

This is the same WN tradeoff as in the [dense FP16 story](dense-gemm-fp16.md): more math per block, fewer blocks, heavier SMEM. For block-scaled GEMM, RHS **and** `scale_rhs` footprint both grow with N.

---

## Step 3: L2 Promotion on RHS TMA — iter053

**The problem.** At 4096³, RHS panels are large. TMA traffic does not always stick in L2 — lines get evicted before they can be reused across K iterations or neighboring CTAs.

**The change.** Set `CU_TENSOR_MAP_L2_PROMOTION_L2_256B` on the RHS tensor map. This Hopper cache hint promotes lines into L2 with 256B granularity.

**Result:** **610 TFLOPS** at 4096³ (+8 TFLOPS over iter051). Nearly free: one flag on the TMA descriptor, zero change to the compute path. When tile geometry and scheduling are already tuned, cache hints are the next lever.

---

## Step 4: Prefetch `scale_a` — iter066

**The problem.** Scale loads inside a tight WGMMA loop with short issue interval can stall the consumer. Per-row `scale_a` loads compete with WGMMA for issue slots.

**The change.** Prefetch `scale_a` into registers **before** the inner WGMMA body, so load latency hides behind independent setup or prior WGMMA work.

**Result:** **621 TFLOPS** at 4096³ — **+56%** vs the 397.9 baseline.

By this point, the kernel is doing heavy WGMMA and wide N. Remaining slack sits in operand latency. Block-scaled GEMM makes scales first-class operands — treat them like any other latency-bound input. Software prefetch, double-buffering, or DMA-to-SMEM are all design axes when registers or scheduling bind.

![Full optimization ladder](images/OptimizationLadder_ManimCE_v0.19.1_dark.png#only-dark)
![Full optimization ladder](images/OptimizationLadder_ManimCE_v0.19.1_light.png#only-light)

---

## Source Variants and Alternative Approaches

Under `blockscale_gemm_v2/`, several `.co` files explore alternative scale movement:

| Variant | Approach |
| ------- | -------- |
| `rhs_scale_dma_smem` / `scale_dma_smem` | Stage scales via TMA into shared memory |
| `transposed_scale` | Change scale layout for coalescing vs index cost |
| `tileN` | Tile along N explicitly in Croktile structure |
| `..._warpspec_persis_1p1c.co` | Persistent kernel variant ([Ch5](../tutorial/ch05-branch-control.md)) |

These document that scale DMA to SMEM is a viable alternative when register pressure or load scheduling hurts WGMMA issue interval.

## Reproduction

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co \
  -o /tmp/bs.cute.result && bash /tmp/bs.cute.result --execute
```

Pre-generated iter049/051/053/066 trees ship with `run.sh` for bit-identical reproduction. Full iteration history: 71 iterations in `README_blockscale_gemm_e4m3_aitune_2026-03-22.md`.

---

## Conclusion

Block-scaled GEMM follows the same arc as dense and sparse: schedule first, widen second, cache-tune third. But it adds **scale tensors as first-class operands** that need the same latency-hiding discipline as matrix data.

The largest single structural win was **N256** (iter051, +51% at 4k) — more math per CTA, fewer CTAs. But the first optimization, iter049 (+21% at 2k), showed that **scale scheduling** matters as much as tile geometry: the gap between "WGMMA done" and "next TMA started" was idle bandwidth that scale accumulation could fill.

L2 promotion (iter053) and scale prefetch (iter066) are late percentage gains on an already strong kernel — they target memory hierarchy and operand latency, not raw WGMMA width. When you have tuned geometry and scheduling, these are the levers that remain.
