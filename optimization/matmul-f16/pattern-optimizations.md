# Optimization patterns

The through-line here is one rule: change the thing the profiler tells you is slow, then re-measure. What follows is the sequence that moved dense FP16 GEMM from **208.7** to **382.5** TFLOPS at **8192³** on H800 PCIe, grouped by the bottleneck each pattern addresses.

## Tile geometry and shared-memory budget

The first knobs you reach for are **WN** (`MATMUL_WARP_N`, the N extent of the WGMMA tile) and **STAGES** (the number of operand ring slots along K). Both interact with shared-memory footprint, and that footprint determines how many CTAs an SM can hold.

**WN** controls arithmetic intensity per staged K-slab. Wider tiles do more math per loaded panel, but every extra column adds bytes to the operand and sometimes accumulator staging areas. At **2048³**, **iter046** moved to **WN=176** with **STAGES=2** and reached **242** TFLOPS, roughly +13% over the early baseline family. Later at **8192³**, a Phase 3 sweep found **WN=160** with K-unroll (iter061) at **380.6** TFLOPS while **WN=168** fell off a cliff: SMEM exceeded **228 KB**, forcing one CTA per SM instead of two. That is a step-function loss, not a gentle degradation. **WN=160** sat near **114.7 KB** with two CTAs/SM. When you sweep WN, step by multiples of 8 and recompute SMEM after every candidate.

**STAGES** deepens latency hiding along K but linearly scales operand buffer size. The baseline used four stages with WN=128. **iter004** tried **STAGES=2** with **WN=256** and landed at **208.9** TFLOPS at **2048³** — fewer stages freed enough SMEM for the wider tile. The largest single jump in the log came from stages alone: **iter048** kept **WN=176** and moved from **STAGES=2** to **STAGES=3**, jumping from **242** to **354.1** TFLOPS at **2048³**. That is not linear scaling from one extra buffer — it is the signature of a bubble-limited schedule where the extra stage bought producer-consumer concurrency, not more math.

The catch: three stages help at **2048³** but hurt at **8192³** because the larger grid amplifies occupancy effects. Extra stages are not free latency hiding; they are bytes that evict concurrent blocks. When you change problem size by 4x, re-sweep STAGES.

**WN and STAGES interact** — fixing WN from a 2048³ experiment and later discovering STAGES is wrong at 8192³ means you revisit WN anyway. The optimization log's arc (iter046 → iter048 → iter050 → iter057 → iter061) reflects that coupling: the 8192³ winners used different WN values than the 2048³ winners because split-output and non-persistent launch changed the binding constraint.

Multiply `MATMUL_STAGES × MATMUL_TILE_K × tile_dimensions` to get bytes; that product must fit under the per-SM shared budget after accounting for output staging. When TFLOPS move +5% without changing WN or STAGES (as in **iter023**, which added **`stmatrix`**, **`ptx-barrier`**, and subspan refinements for **214.3** TFLOPS at **2048³**), you are fixing addressing or operand setup — the space between TMA and WGMMA, not the GEMM graph itself.

## Split-output 1p2c

Once WN grows beyond a threshold, **output contention** becomes the limiter. In 1p1c (one producer, one consumer), a single `output_s` tile serves one consumer warpgroup. Adding a second consumer with the same shared output creates serialization on that tile — SMEM traffic and synchronization on the accumulator path eat into the throughput you gained from wider tiles.

**Split-output 1p2c** gives each consumer a private slice of the output staging area. You trade slightly higher SMEM for less contention and a better instruction mix. **iter050** validated this at **4096³** (~**375** TFLOPS, 1p2c split-output, WN=128, STAGES=2) before committing to the large-cube experiment. If split-output had regressed at 4096³, the pattern would not have been trusted at 8192³.

**iter057** carried split-output to its conclusion: 1p2c split-output, WN=152, **non-persistent** launch → **382.5** TFLOPS at **8192³**, the best headline in the study. The non-persistent win is problem-specific — [persistent kernels](../../tutorial/ch07-persistent.md) fix grid-level tail underuse, but when inner-block SMEM and pipeline choices already cap throughput, persistence cannot recover what occupancy lost. At 8192³ with the split-output tile, wave quantization was acceptable and the inner block was already the bottleneck, so a conventional grid won.

You rarely see output contention in a single profiler counter. The heuristic that correlated in this study: TFLOPS rose when moving from 1p1c to 1p2c at larger WN, but **only** with split-output enabled — implying the consumer side was serialized on `output_s` traffic.

Reference kernels: `matmul_f16_dyn_sm90_warpspec_1p2c.co` and the shipped `*_iter050_*` / `*_iter057_*` variants.

## Compiler flags and instruction-level overlap

With tileflow structure settled, the last layer is how the compiler lowers it. The shipped builds share a common flag bundle:

- **`--use-warpspec`** — warp-specialized codegen for the producer/consumer split.
- **`--stmatrix`** — STSM-style shared-memory matrix setup where legal.
- **`--hoist-offset`** / **`--hoist-scale`** — hoist address arithmetic and scale factors out of inner loops.
- **`--ptx-barrier`** — barrier instructions compatible with async producer/consumer synchronization.
- **`--tma-cluster-aware`** — bias TMA lowering for cluster and multicast on SM90.
- **`--wgmma-wait-depth=N`** — expose WGMMA pipeline wait depth as a tunable, added during Phase 3 to match stage count and issue rate.

**iter023** showed these matter: +5% at 2048³ from `ptx-barrier` and `stmatrix` on top of tile tweaks. But the lesson from the full log is order of operations: freeze flags while sweeping WN and STAGES, then unfreeze only after split-output lands. Over-tuning flags while SMEM is on the wrong side of **228 KB** is a common failure mode.

When comparing iterations, hold everything except the knob under test: same `CHOREO_TIMING_*`, same verification state, same arch (`sm_90a`). The README tables stay trustworthy because harness defaults held stable across 65 iterations.

## The arc from baseline to best

The optimization followed a dependency chain, not a free-form search:

1. **Baseline** (208.7 TFLOPS at 8192³) — correct roles (1p1c) but stage count and WN mismatched to the problem's occupancy reality.
2. **Phase 1** — at 2048³, reduce SMEM or improve lowering → modest gains (214.3 TFLOPS).
3. **Phase 2** — jointly tune WN and STAGES, then move to 1p2c split-output → large jumps (354.1 at 2048³, ~375 at 4096³, 382.5 at 8192³).
4. **Phase 3** — WN sweep at 8192³ with K-unroll and `wgmma-wait-depth` → 380.6 (iter061) and discovery of the WN=168 occupancy cliff.

The study stayed inside dense FP16 GEMM with TMA-staged operands and WGMMA accumulation — no mixed precision, no split-K across CTAs, no CUDA Graph capture. The +83% came entirely from tileflow geometry, output staging, and compiler flags.

The largest single structural win was not a flag — it was **1p2c split-output** moving TFLOPS into the 370–382 band. Flags like `--stmatrix` matter, but they cannot recover serialization on `output_s` if two consumers share one accumulator tile. When you face a similar ceiling in your own kernel, check whether the output path is the bottleneck before reaching for instruction-level levers.

Next: [AI-tune last mile](aitune-last-mile.md) for repro commands and shipped kernel details.
