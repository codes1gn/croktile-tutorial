# AI-tune: shipped kernels and reproduction

This page is about **how** the final numbers were produced—which **`.co`** files shipped, how to run them, and what Phase 3 learned about **WN** and occupancy—without replaying all 65 iterations. Branch detail: `origin/ai-tune/2026-03-23/matmul_f16`; full tables: `croktile/benchmark/performance/matmul/README_matmul_f16_aitune_2026-03-23.md`.

**Hardware** — H800 PCIe, **SM90a**, **114** SMs. Compare tuned kernels to **cuBLAS** (~**380** TFLOPS on this stack), not the **~1513** TFLOPS marketing peak. The README quotes **iter061** as **80.7%** of cuBLAS at **8192³** and **100.5%** at **2048³**—the smaller cube favors the chosen tile and stage mix, so short-cube efficiency can read above the library while **8192³** stays the stress case.

## Shipped checkpoints

**iter048** — **354.1** TFLOPS at **2048³**; **1p1c**, **WN=176**, **STAGES=3**. Shows the three-stage operand ring paying off on a mid-size cube before you scale the grid.

**iter050** — **~375** TFLOPS at **4096³**; **1p2c split-output**, **WN=128**, **STAGES=2**. Validates split-output between **2048³** and the final **8192³** push.

**iter057** — **382.5** TFLOPS at **8192³**; **1p2c split-output**, **WN=152**, **non-persistent**. Best headline in the study vs. the **208.7** main baseline.

**iter061** — **380.6** TFLOPS at **8192³**; **1p2c split-output**, **WN=160**, **K-unroll** and Phase-3 wait-depth tuning. Trades **1.9** TFLOPS vs. iter057 for a stronger cross-size story (the **100.5% / 80.7%** cuBLAS pair above). Shared flag bundle; difference is **WN**, **K-unroll**, and **`wgmma-wait-depth`**.

## Build and run

From the Croktile repo root after `make build`, use the same **croktile** flags for each artifact; only the input **`.co`** changes:

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/<INPUT>.co \
  -o /tmp/run.cute.result && bash /tmp/run.cute.result --execute
```

Concrete inputs:

- **iter057** (best **8192³** TFLOPS): `matmul_f16_aitune_2026-03-23_matmul_f16_iter057_1p2c_so_wn152_nonpersis.co`
- **iter061**: `matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co`
- **iter050**: `matmul_f16_aitune_2026-03-23_matmul_f16_iter050_1p2c_splitout.co`
- **iter048**: `matmul_f16_aitune_2026-03-23_matmul_f16_iter048_s3_wn176_best.co`

Phase 3 added **`--wgmma-wait-depth=N`** to the README invocations where recorded; **`N`** couples to **STAGES** and **WN**—reproduce with the exact command from the README unless you are sweeping **`N`** deliberately.

**Harness** — Defaults in the log use **`CROKTILE_TIMING_WARMUP`** (10) and **`CROKTILE_TIMING_REPEAT`** (500). Keep verify on for apples-to-apples TFLOPS; set **`CROKTILE_SKIP_VERIFY=1`** only when you already trust correctness. If noise dominates, raise **`CROKTILE_TIMING_REPEAT`** before you chase shorter warmup.

## iter057 vs. iter061

Pick **iter057** when **8192³** is the only number that matters (peak headline, strong scaling). Pick **iter061** when one binary should behave well on both small and large cubes without retuning **`MATMUL_*`** per job. Both are **1p2c split-output** with the same compiler bundle.

## Phase 3: WN sweep and the occupancy cliff

After split-output unlocked cuBLAS-class throughput, Phase 3 swept **WN** at **8192³**. **WN=168** failed: shared memory went past **228 KB**, residency dropped to **one block per SM**, and throughput fell off a cliff—latency hiding across CTAs disappears, so the hit is large, not a few percent. **iter061** at **WN=160** sits near **114.7 KB** with **two CTAs/SM**. You catch that kind of threshold by measuring and by precomputing bytes per block, not by guessing from **WN** alone.

Condensed timeline: Phase 1 (iterations ~001–038) — **1p1c** at **2048³**, **214.3** TFLOPS after SMEM and lowering tweaks. Phase 2 (~043–057) — split-output and multi-size validation, **382.5** at **8192³** (**iter057**). Phase 3 (~061–065) — **WN** sweep, **380.6** (**iter061**), **`wgmma-wait-depth`**, **WN=168** failure.

## Reproducibility

Build **`./croktile`** from the same revision as the **`.co`** file or expect codegen drift. Use **`-arch=sm_90a`** for this GPU class. When you compare to external cuBLAS figures, note driver and library version and GPU clock behavior—thermal or power caps can narrow **~380** TFLOPS references slightly.

## Where the files live

Dated filenames (`matmul_f16_aitune_2026-03-23_*`) keep artifacts stable in history. Beside them: **`matmul_f16_dyn_sm90.co`** (dynamic baseline), **`matmul_f16_dyn_sm90_warpspec_1p1c.co`**, **`matmul_f16_dyn_sm90_warpspec_1p2c.co`**. For regressions, **`diff -u`** on **`MATMUL_*`** and **`parallel`** structure—those dominate SMEM and roles.

Concepts tie back to [index](index.md), [baseline](baseline-analysis.md), and [patterns](pattern-optimizations.md). The parent [optimization index](../index.md) lists this case next to sparse and block-scaled GEMMs; dense FP16 keeps the data path minimal so the schedule story stays visible.
