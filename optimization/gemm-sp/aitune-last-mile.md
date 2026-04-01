# AI-Tune and the Last Mile: `.co` vs `.cu`

AI-tune here means high-volume compile-measure over a structured neighborhood of kernels. You mutate a small set of knobs — stages, warp split, swizzle, metadata load style, flags, unroll, TK, TMA descriptors — build, run the harness, and record TFLOPS. The search is cheap compared to human hypothesis latency. **Correctness** (2:4 metadata aligned with packed operands) and **interpretation** (sync vs TMA vs occupancy) stay on you.

## FP16: Where `.co` Stops and `.cu` Takes Over

The FP16 story has a clear ceiling on compiler-generated `.co`:

| Checkpoint | TFLOPS | Type | Key change |
| ---------- | ------ | ---- | ---------- |
| Baseline | 368 | `.co` | 1p1c, TK64, 2-stage |
| **iter120** | **434** | **`.co`** | **1p2c + 3-stage (best `.co`)** |
| iter137 | 543 | `.cu` | Inner unroll 24 + FTZ |
| **iter143** | **655** | **`.cu`** | **TK128, TMA metadata, split RHS TMA** |

That is roughly +18% from baseline to best `.co` (368 → 434), then +51% from iter120 to iter143 (434 → 655) once you have CUDA-level control. The second leg is not a polishing pass — it is **different expressiveness**.

### Why the `.co` Path Plateaus

Croktile's `.co` compiler still chooses loop nests, register allocation, and async-proxy placement. Sparse GEMM couples operand TMA, metadata (scalar, vector, or TMA), and WGMMA batching with warpgroup barriers. When the compiler serializes metadata consumption with MMA in a way no single pragma fixes, you need `.cu` surface area for:

- Manual inner-loop unroll (iter137: unroll 24)
- Explicit metadata prefetch into registers
- TMA descriptor changes for metadata (iter143)
- Split RHS TMA to match consumer demand

That is the gap between iter120 and iter137/iter143.

### What Nsight Should Show

If you are validating with Nsight Compute:

- **iter137** should show inner-loop issue improvements — fewer stalls between WGMMA fragments
- **iter143** should show shifted DRAM/L2 balance and better behavior on the metadata stream after TK128 and TMA metadata

## E4M3: Automation Carries Most of the Distance

E4M3 starts stronger and stays in the automation story longer:

| Checkpoint | TFLOPS | Key change |
| ---------- | ------ | ---------- |
| Baseline | 671 | 1p1c, swizzle 128/128, prepack, 2-stage |
| iter001 | 759 | TMA metadata staging |
| iter016 | 772 | Early empty + merged barrier |
| iter023 | 811 | Software pipeline + warpgroup_wait<1> |
| iter036 | 897 | 1p2c |
| **iter040** | **1090** | **3-stage pipeline (+62% vs baseline)** |
| iter068 | **1127** | Early empty arrive (+68% vs baseline) |

There is no headline `.co` plateau vs `.cu` cliff in this table. The search spends more iterations on pipeline and barrier polish because the baseline already brought strong TMA and layout choices. The jump past 1000 TFLOPS at iter040 is the signature of pipeline depth; the last tens of TFLOPS are sync polish.

## What Transferred Between Dtypes

Copy causal structure, not parameter equality:

| Pattern | FP16 appearance | E4M3 appearance |
| ------- | --------------- | --------------- |
| Metadata on TMA plane | iter143 (late, `.cu`) | iter001 (early, automated) |
| Fine warpgroup sync | Early % chain | iter023 |
| 1p2c | iter120 (with 3-stage) | iter036 |
| 3-stage depth | iter120 (bundled) | iter040 (the >1000 jump) |
| Barrier micro-optimization | Secondary | iter016, iter068 |

## Workflow That Matches the Logs

1. **Freeze** problem size (4096 × 8192 × 8192) and build flags
2. **One family** of edits per run when debugging; **bundle** when exploring a known-good neighborhood (as with 3-stage)
3. **Establish** baseline TFLOPS; keep artifact paths under `benchmark/performance/gemm_sp/`
4. **Run AI-tune or manual grid** on `.co` until TFLOPS flatten — on FP16, near iter120 class
5. **Export or edit `.cu`** for unroll, explicit metadata prefetch, TK moves, and split TMA
6. **Re-run numerical checks** after every risky change — sparse metadata bugs stay silent until a bitwise compare fails

### Regression Hazards

Legal schedules can still break 2:4 invariants:

- Metadata chunks misaligned to K tile boundaries
- Double consumption of a packed fragment under unroll
- Keep a small deterministic self-check in CI
- When TFLOPS jumps unexpectedly, treat it as suspect until checks pass

## Efficiency vs Peak (Sanity Only)

| Dtype | Best TFLOPS | Dense peak | Ratio |
| ----- | ----------- | ---------- | ----- |
| FP16 | 655 | 1513 | ~43% |
| E4M3 | 1127 | 3026 | ~37% |

Similar band, different dominant bottleneck (sync at the top end on E4M3, metadata expressiveness on FP16). These ratios are not a grade — they confirm both dtypes are in the right ballpark for sparse 2:4 on this hardware.

## Summary

AI-tune compresses calendar time on the long tail: it finds iter120, iter040, and iter068 faster than ad-hoc guessing. On FP16, the last mile belongs to engineers when compiler schedules cap TFLOPS — that is iter120 → iter143. On E4M3, the same machinery covers 671 → 1127 with 3-stage and barrier work as the headline structural wins. Use `.co` for breadth and `.cu` for depth where the logs say you must; measure every step on the same harness so the ladders stay comparable.

Iteration labels and TFLOPS rows are anchored in `README_gemm_sp_f16_aitune_2026-03-25.md` and `README_e4m3_aitune_2026-03-21.md`. If your Croktile tree advances, match patterns to your regenerated READMEs.
