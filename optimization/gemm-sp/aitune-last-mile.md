# AI-Tune and the Last Mile: `.co` vs `.cu`

AI-tune here means **high-volume compile–measure** over a structured neighborhood of kernels Choreo already understands. You mutate a small set of knobs—stages, warp split, swizzle, metadata load style, flags, unroll, **TK**, TMA descriptors—build under `benchmark/performance/gemm_sp/`, run the standard harness, and record TFLOPS. The search is cheap compared to human hypothesis latency; **correctness** (2:4 metadata aligned with packed operands) and **interpretation** (sync vs TMA vs occupancy) stay on you.

---

## FP16: where `.co` stops and `.cu` takes over

The documented FP16 sweep hits a clear ceiling on **compiler-generated `.co`**: **iter120 at 434 TFLOPS** with **1p2c + 3-stage**—already a major schedule rewrite from the **368 TFLOPS** baseline (1p1c, swizzle 64, TK64, 2-stage), but still short of what hand-authored CUDA eventually reaches.

From there the story splits. **iter137 at 543 TFLOPS** is the strongest **“organic” `.cu`** in the log: **inner unroll 24** and **FTZ** expose ILP and trim denorm edge cases without yet rewriting the full memory hierarchy story. **iter143 at 655 TFLOPS** adds **TK128**, **TMA-backed metadata**, and **split RHS TMA**—structural memory-system work, not loop cosmetics.

Roughly **+18%** baseline → best `.co` (368→434), then **~+51%** from iter120 to iter143 (434→655) once you have **CUDA-level** control over scheduling and descriptors. The second leg is not a polishing pass; it is **different expressiveness**. Choreo’s `.co` path still chooses loop nests, register allocation, and much async-proxy placement. Sparse GEMM couples **operand TMA**, **metadata** (scalar, vector, or TMA), and **WGMMA** batching with **warpgroup** barriers. When the compiler **serializes** metadata consumption with MMA in a way no single pragma fixes, **manual unroll**, **explicit prefetch**, and **TMA descriptor** work need **`.cu`** surface area—which is exactly the gap between **iter120** and **iter137 / iter143**.

If you are validating in Nsight, **iter137** should show inner-loop issue improvements; **iter143** should show shifted DRAM/L2 balance or better behavior on the metadata stream after **TK128** and **TMA meta**.

---

## E4M3: automation carries most of the distance

E4M3 starts at **671 TFLOPS** with **1p1c**, **swizzle 128/128**, **prepack**, and **2-stage**—already close to serious FP8 sparse scheduling. The ladder **671 → 759 → 772 → 811 → 897 → 1090 → 1127** (iter001 through iter068) stays largely in the same automation story: metadata on the TMA plane at **iter001**, barrier simplification at **iter016**, software pipeline plus fine warpgroup wait at **iter023**, **1p2c** at **iter036**, the **3-stage** discontinuity at **iter040 (1090 TFLOPS**, on the order of **+62%** vs baseline in the README), and **early empty arrive** at **iter068** for **+68%** overall. There is no headline **`.co` plateau vs `.cu` cliff** in the same table as FP16; the search spends more iterations on **pipeline** and **barrier** polish because the baseline already brought strong TMA and layout choices.

---

## What transferred between dtypes

You should copy **causal structure**, not parameter equality. **Metadata on the TMA plane** shows up as **iter143** on FP16 and **iter001** on E4M3. **Fine warpgroup sync** shows up as **`warpgroup_wait<1>`** early in the FP16 percent chain and **iter023** on E4M3. **1p2c** lands at **iter120** (with 3-stage) on FP16 and **iter036** on E4M3. **3-stage depth** is the **iter040** story on E4M3 and pairs with **1p2c** at **iter120** on FP16. Barrier micro-optimization is secondary in the FP16 milestone table but central to **iter016** and **iter068** on E4M3.

---

## Workflow that matches the logs

Freeze problem size (**4096×8192×8192**) and build flags; change **one family** of edits per run when you are debugging, bundle when you are exploring a known-good neighborhood (as with **3-stage**). Establish baseline TFLOPS and keep the artifact path under `benchmark/performance/gemm_sp/`. Run AI-tune or a manual grid on **`.co`** until TFLOPS flatten—on FP16, near **iter120** class—then export or edit **`.cu`** for **unroll**, explicit meta prefetch, **TK** moves, and **split TMA**. Re-run **numerical** checks after every risky change; sparse metadata bugs stay silent until a bitwise or tight tolerance compare fails.

**Regression hazards:** legal schedules can still break 2:4 invariants—metadata chunks misaligned to **K** tile boundaries, double consumption of a packed fragment under **unroll**. Keep a small deterministic self-check in CI; when TFLOPS jumps unexpectedly, treat it as suspect until checks pass.

**Efficiency vs peak (sanity only):** **655 / 1513 ≈ 43%** for FP16 best vs dense FP16 peak; **1127 / 3026 ≈ 37%** for E4M3 best vs FP8 peak—similar band, different dominant bottleneck (sync at the top end on E4M3).

---

## Summary

AI-tune **compresses calendar time** on the long tail: it finds **iter120**, **iter040**, and **iter068** faster than ad-hoc guessing. On **FP16**, the **last mile** still belongs to engineers when compiler schedules cap TFLOPS—that is **iter120 → iter143**. On **E4M3**, the same machinery covers **671 → 1127** with **3-stage** and **barrier** work as the headline structural wins. Use **`.co` for breadth** and **`.cu` for depth** where the logs say you must; measure every step on the **same harness** so the ladders stay comparable.

Iteration labels and TFLOPS rows are anchored in `README_gemm_sp_f16_aitune_2026-03-25.md` and `README_e4m3_aitune_2026-03-21.md`; if your Choreo tree advances, match patterns to **your** regenerated READMEs.
