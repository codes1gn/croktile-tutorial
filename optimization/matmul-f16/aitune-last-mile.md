# AI-Tune: Shipped Kernels and Reproduction

This page covers the **how**: which `.co` files shipped, how to run them, and what the Phase 3 WN sweep revealed about occupancy. Branch detail: `origin/ai-tune/2026-03-23/matmul_f16`; full tables: `croktile/benchmark/performance/matmul/README_matmul_f16_aitune_2026-03-23.md`.

**Hardware:** H800 PCIe, SM90a, 114 SMs. Compare tuned kernels to cuBLAS (~380 TFLOPS), not the ~1513 TFLOPS marketing peak.

## Shipped Checkpoints

| Checkpoint | TFLOPS | Size | Configuration |
| ---------- | ------ | ---- | ------------- |
| **iter048** | 354.1 | 2048³ | 1p1c, WN=176, STAGES=3 |
| **iter050** | ~375 | 4096³ | 1p2c split-output, WN=128, STAGES=2 |
| **iter057** | **382.5** | 8192³ | 1p2c split-output, WN=152, non-persistent |
| **iter061** | 380.6 | 8192³ | 1p2c split-output, WN=160, K-unroll, wgmma-wait-depth |

**iter057** is the peak headline. **iter061** trades 1.9 TFLOPS at 8192³ for a stronger cross-size story: 100.5% of cuBLAS at 2048³ and 80.7% at 8192³.

## Build and Run

From the Croktile repo root after `make build`, use the same flags for each artifact — only the input `.co` changes:

```bash
./croktile -gs -t cute -arch=sm_90a \
  --use-warpspec --stmatrix --hoist-offset --hoist-scale \
  --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/<INPUT>.co \
  -o /tmp/run.cute.result && bash /tmp/run.cute.result --execute
```

Concrete `.co` filenames:

```
# iter057 — best 8192³ TFLOPS
matmul_f16_aitune_2026-03-23_matmul_f16_iter057_1p2c_so_wn152_nonpersis.co

# iter061 — best cross-size robustness
matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co

# iter050 — split-output validation at 4096³
matmul_f16_aitune_2026-03-23_matmul_f16_iter050_1p2c_splitout.co

# iter048 — 3-stage at 2048³
matmul_f16_aitune_2026-03-23_matmul_f16_iter048_s3_wn176_best.co
```

Phase 3 added `--wgmma-wait-depth=N` where recorded — `N` couples to STAGES and WN, so reproduce with the exact command from the README unless you are sweeping `N` deliberately.

**Harness defaults:** `CROKTILE_TIMING_WARMUP=10`, `CROKTILE_TIMING_REPEAT=500`. Keep verification on for apples-to-apples TFLOPS. Set `CROKTILE_SKIP_VERIFY=1` only when you already trust correctness. If noise dominates, raise `CROKTILE_TIMING_REPEAT` before chasing shorter warmup.

## Choosing Between iter057 and iter061

| | iter057 | iter061 |
|---|---------|---------|
| Best for | Peak 8192³ headline | One binary across sizes |
| WN | 152 | 160 |
| Extra | non-persistent | K-unroll, wgmma-wait-depth |
| 8192³ | **382.5** | 380.6 |
| 2048³ | good | **100.5% of cuBLAS** |

Both are 1p2c split-output with the same compiler flag bundle.

## Phase 3: The WN Sweep and the Occupancy Cliff

After split-output unlocked cuBLAS-class throughput, Phase 3 asked: what is the optimal WN for 8192³? The sweep found a sharp boundary:

| WN | SMEM (approx) | CTAs/SM | TFLOPS @8192³ |
| -- | ------------- | ------- | ------------- |
| 152 | ~108 KB | 2 | 382.5 |
| 160 | ~114.7 KB | 2 | 380.6 |
| **168** | **> 228 KB** | **1** | **cliff** |

At WN=168, shared memory exceeded 228 KB. Residency dropped from 2 blocks to 1 block per SM. Latency hiding across CTAs disappeared, and throughput fell catastrophically — not a few percent, but a large, sharp regression.

You catch this kind of threshold by precomputing bytes per block:

```
SMEM = STAGES × (WM × TK + WN × TK) × sizeof(fp16) + output_staging
```

and checking against the 228 KB per-SM budget. Do not guess from WN alone.

## Condensed Timeline

| Phase | Iterations | Key Milestone | TFLOPS |
| ----- | ---------- | ------------- | ------ |
| Phase 1 | ~001–038 | 1p1c at 2048³ + SMEM/lowering tweaks | 214.3 |
| Phase 2 | ~043–057 | Split-output, multi-size validation | **382.5** @8192³ |
| Phase 3 | ~061–065 | WN sweep, K-unroll, wgmma-wait-depth | 380.6 @8192³ |

65 iterations total. Phase 1 took ~38 iterations to improve by +5%. Phase 2 took ~14 iterations to improve by +83%. Phase 3 refined the last few TFLOPS and discovered the WN=168 failure. Power laws are everywhere.

## Reproducibility Notes

- Build `./croktile` from the same revision as the `.co` file, or expect codegen drift.
- Use `-arch=sm_90a` for this GPU class.
- When comparing to external cuBLAS figures, note driver version, library version, and GPU clock behavior — thermal or power caps can narrow ~380 TFLOPS references slightly.

## Where the Files Live

Dated filenames (`matmul_f16_aitune_2026-03-23_*`) keep artifacts stable in history. Beside them:

- `matmul_f16_dyn_sm90.co` — dynamic baseline
- `matmul_f16_dyn_sm90_warpspec_1p1c.co` — 1p1c teaching kernel
- `matmul_f16_dyn_sm90_warpspec_1p2c.co` — 1p2c split-output template

For regressions, `diff -u` on `MATMUL_*` parameters and `parallel` structure — those dominate SMEM and roles.

Concepts tie back to [baseline analysis](baseline-analysis.md) and [optimization patterns](pattern-optimizations.md). The parent [optimization index](../index.md) lists this case alongside sparse and block-scaled GEMMs.
