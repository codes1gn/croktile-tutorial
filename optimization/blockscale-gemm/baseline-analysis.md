# Baseline and Block-Scaling Background

This section defines **block-scaled GEMM** in the sense used by the Croktile benchmarks, ties it to **FP8 E4M3** limitations, and describes the **baseline kernel** whose throughput anchors the AI-tune story. Unless noted, problem sizes are **2048³** and **4096³** cubes; hardware is **H800 PCIe** with **3026 TFLOPS** FP8 peak.

## Why block scaling exists

**FP8 E4M3** trades dynamic range and mantissa precision for **half the operand footprint** of FP16 and a path to high tensor-core throughput. A naive GEMM that simply casts to FP8 and accumulates in low precision **loses information**: values that differ greatly in magnitude within a long **K** reduction cannot all be represented faithfully after quantization.

**Per-block scaling** repairs this without reverting to FP16 weights throughout. The idea is local: partition **K** into blocks (in these kernels, aligned with **TILE_K = 128**). For each **(row, block)** on the left and matching **(column, block)** on the right, store a **single FP32 scale** (or a small factor pair) that rescales the FP8 payloads in that block. During the dot product, contributions from block **b** are scaled consistently so that the **reduction in FP16** (or a widened internal path) approximates a higher-precision reference.

Conceptually, if `\tilde{a}` and `\tilde{b}` are quantized blocks and `s_a`, `s_b` are their scales, the block’s contribution to the inner product scales like `s_a s_b \langle \tilde{a}, \tilde{b} \rangle`. The exact layout of `scale_lhs` and `scale_rhs` in memory depends on the kernel; the benchmarks use **global** tensors sized with **`DIV_BLK_K`** (and **`DIV_BLK_N`** where the right-hand scale grid is two-dimensional) so that each **K-tile iteration** can index the correct factors.

Accumulation **order** still matters for **finite** **FP16**: the blockscaled formulation is **not** bitwise identical to **FP32** reference unless you widen internally. The shipped verification tolerances reflect that contract—**close** numerically, **not** **ULP-perfect**.

This is the same family of tricks used in **MXFP8**-style training and inference stacks: **FP8 for density**, **scales for fidelity**.

## Croktile surface: `mma.row.row.scale`

In Croktile, the fused path is expressed as a single MMA form that consumes **fragments from shared memory** (after **TMA** loads with **swizzle**) **and** the **per-tile scale pointers** appropriate to the current **M**, **N**, and **K** block.

The baseline device code in `benchmark/performance/blockscale_gemm/blockscale_gemm_dyn_sm90.co` follows the dense GEMM skeleton—**TMA** in, **WGMMA-shaped** loads, accumulator in registers—except the inner MMA is **`mma.row.row.scale`** instead of a plain **`mma.row.row`**. That opcode ties **WGMMA** execution on **E4M3** operands to the **scale factors** for the active **K** slice, keeping the **FP16** accumulator contract described in the shipped README (verification uses **FP32** reference dots with blockscale factors, **FP16** output, tolerances **`base_tol=0.5`**, **`rel_tol=0.01`**).

**One K-tile at a time (baseline control flow)**:

1. **TMA** **`lhs`** and **`rhs`** **subspans** of shape **(WARP_M, TILE_K)** and **(WARP_N, TILE_K)** into **`lhs_load_s`** / **`rhs_load_s`** with **swizzle** matching **TILE_K**.
2. **Inner** loop over **WARP_K** chunks along **K** inside the tile: **MMA load** fragments from **SMEM**, then **`mma.row.row.scale`** with **views** into **`scale_lhs`** and **`scale_rhs`** indexed by **`block_m`**, **`block_n`**, and **`iv_k`**.
3. After all **K** tiles, **store** the **accumulator tile** to **SMEM** and **TMA** out to **global** **`output`**.

Every **iteration** on **`iv_k`** touches **fresh** **FP8** data **and** the **corresponding** **scale** **columns**—that **coupling** is what distinguishes **profiling** **blockscale** from **profiling** **vanilla** **GEMM**.

Readers coming from [Chapter 5: MMA](../../tutorial/ch05-mma.md) should treat **`mma.row.row.scale`** as the blockscale analogue of the usual **`mma.row.row`**: same tiling discipline, **extra operands** for scales, and the same pressure to **hide TMA latency** behind math.

## Baseline kernel geometry

The reference baseline uses **M64 × N128 × K32** per warpgroup inner steps with **TILE_K = 128** along **K** (four **K32** steps per tile), **swizzle 128** on **TMA**, and **`MATMUL_WARP_M = 64`** fixed for **E4M3 WGMMA** constraints in the source comments.

| Field | Baseline choice |
|-------|-----------------|
| Tile label | **M64N128K32** (per README) |
| Operand staging | `lhs_load_s`, `rhs_load_s` in **shared memory** |
| Parallelism | **CTA grid** over **M** and **N** tile indices |

Representative **`.co`** entry points:

| Role | Path |
|------|------|
| Baseline (this study) | `benchmark/performance/blockscale_gemm/blockscale_gemm_dyn_sm90.co` |
| Warp-specialized 1p1c | `benchmark/performance/blockscale_gemm/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co` |
| Tuned **`__launch_bounds__` / regs | `blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c_m2048_n2048_k2048.co` |

The **v2** folder adds experiments (**`tileN`**, **`rhs_scale_dma_smem`**, **`scale_dma_smem`**, **`transposed_scale`**) that change **how** scales reach the MMA path—**registers vs TMA-to-SMEM**—without changing the high-level problem.

## Measured baseline throughput

| Shape | TFLOPS | Efficiency vs 3026 TFLOPS peak |
|-------|--------|----------------------------------|
| 2048³ | **314.2** | 10.4% |
| 4096³ | **397.9** | **13.2%** |

For comparison, a **plain** **dense FP8 GEMM** on the same class of hardware can approach **much higher** fractions of peak because **operand** traffic dominates and **fused scale** paths are absent. **Blockscale** intentionally **pays** those costs for **accuracy**; tuning then asks how much of the **lost** throughput can be **reclaimed** with **better overlap** and **memory hints**—the subject of [pattern-optimizations.md](pattern-optimizations.md).

These figures are **not** a statement that the baseline is “wrong”—they show that **blockscaled FP8** is **more than matmul**: every **K-tile** pulls **matrix data and scale metadata**, and **`mma.row.row.scale`** ties the math to that metadata. Compared with a highly tuned **plain FP8 GEMM**, **extra loads and dependent instructions** shrink the fraction of cycles spent in “pure” FMA throughput.

Still, **13%** of headline peak leaves a large **software margin**. The optimization thread asks: **where do cycles go**—**TMA bubbles**, **scale fetch latency**, **too small an N tile**, or **cache behavior on large RHS panels**—and can **warp specialization**, **overlap**, **wider N**, **L2 hints**, and **prefetch** recover them?

## Scale tensor shapes (conceptual)

The benchmark entry points pass **FP8 matrices** and **FP32 scale** tensors. Naming matches the **`.co`** kernels:

| Tensor | Typical role | Indexing intuition |
|--------|--------------|--------------------|
| `scale_lhs` | Per **M**-row scale per **K**-block | **`[M, DIV_BLK_K]`** — one column per **K** block |
| `scale_rhs` | Per **N**-column (and possibly **M**-tile) scale per **K**-block | **`[DIV_BLK_N, DIV_BLK_K]`** or layouts aligned to **block_n** |

**`DIV_BLK_K`** is **`K / 128`** when **K** is a multiple of the tile; the device loop **`iv_k`** steps **K-tiles**, and **`mma.row.row.scale`** consumes the **slice of scales** matching **`iv_k`**. Variants that **transpose** how scales are stored (**`transposed_scale`**) trade **TMA friendliness** against **index arithmetic** in the inner loop.

## What the baseline is not measuring

The **3026 TFLOPS** figure is a **tensor-core marketing peak** under favorable assumptions. **Blockscaled GEMM**:

- Issues **more global traffic per FLOP** than dense GEMM (matrix + scales).
- Uses **fused MMA** that may not **pack** identically to the simplest **FP8×FP8→FP32** throughput tests.
- Runs at **problem sizes** where **L2** and **wave quantization** matter; **2048³** and **4096³** are useful for development but are not the largest cubes in the full benchmark matrix.

So **13% of peak** is a **lower bound on quality**, not an upper bound on ambition. The case study treats **relative speedups** (baseline → iter066) as the primary signal.

## Profiling narrative (how the thread was guided)

The published README compresses **71 iterations** into four **shipped** snapshots. The search combined:

1. **Size sweeps** — a change that helps **4096³** can **hurt 2048³** if it **halves the CTA count** (**N256** story).
2. **Occupancy vs shared memory** — widening **N** to **256** **doubles** math per tile but pushes **SMEM** to about **40 KB** for operand staging in the winning configuration; that is still workable on Hopper but **reduces headroom** for extra pipeline stages.
3. **Latency hiding** — **TMA** completion, **WGMMA** completion, and **scale accumulation** are **different critical paths**. **iter049** explicitly **overlaps** the **next** **K-block’s TMA** with **scale_accumulator** work after **WGMMA** (**+21%** at **2048³**).
4. **Memory hierarchy hints** — **iter053** and **iter066** target **RHS** reuse and **scale** fetch latency, not raw **WGMMA** width.

This is the same **“throughput + resource arithmetic”** style as [Dense GEMM FP16 baseline notes](../matmul-f16/baseline-analysis.md), with **scale tensors** added to the balance sheet.

## Relation to warp specialization

The baseline path can be written as **single-group** Croktile; production-style kernels in this tree often use **1p1c** warp specialization (**one producer warpgroup for TMA**, **one consumer for WGMMA**), as in [Chapter 6: Warp specialization](../../tutorial/ch06-warpspec.md). The **AI-tune winners (iter049–iter066)** are generated in that **warp-specialized** family; gains reported in the README are **on top of** an already structured pipeline, not on top of a trivial scalar loop.

**Verification** (from the README): **512** **coprime-stride** samples over **M×N**, each compared against a **full FP32** reference dot with blockscale factors. That keeps iteration velocity high while catching **scale indexing** and **MMA** mistakes.

## Summary

Block scaling makes **FP8 GEMM numerically viable** by attaching **FP32 factors** to **K-blocks**. The **Croktile** baseline realizes that as **`mma.row.row.scale`** with **TMA-swizzled** operands and lands near **314 / 398 TFLOPS** at **2048³ / 4096³**. The next document walks **concrete schedule and memory changes** that push **4096³** to **621 TFLOPS** (**+56%** vs baseline) while staying in the same **ISA and framework** family.
