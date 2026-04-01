# Enable Tensor Cores in One Primitive: the `mma` Operations

Chapter 4 showed Tensor Memory Accelerator (TMA) copies: wide, hardware-assisted movement from global memory into layouts that later compute expects. The natural follow-up is what happens after those tiles land in shared memory: **tensor cores** turn them into math. The lifecycle is always **fill → load → multiply → store**, whether you are on Ampere or Hopper. Croktile hides PTX fragment types so you think in **tiles**, not lanes.

This chapter covers **Matrix Multiply-Accumulate (MMA)** and how Croktile spells it with **`mma.*`**. You will see **SM86** (Ampere-class hardware such as RTX 3090 / A40), where each warp owns a **16×16×16** FP16 tile, and contrast that with **SM90** (Hopper) **WGMMA** as in `matmul_f16_dyn_sm90.co` alongside TMA—same story at a high level, different thread scope and memory details.

## Tensor cores and the MMA lifecycle

A **tensor core** is not a general-purpose ALU. It is dedicated hardware that, in one **macro-operation** from your point of view, multiplies a small matrix tile and folds the result into a running accumulator—the **multiply-accumulate** pattern behind dense linear algebra.

The unit is deeply pipelined internally. For your mental model, what matters is that you issue **one MMA instruction** per 16×16×16 on this FP16 path instead of thousands of scalar FMAs in source code.

On Ampere FP16 tensor cores, a common native tile is **M×N×K = 16×16×16**. Thirty-two threads in a warp cooperate; the hardware maps operand **fragments** (register subsets per lane) across lanes and performs the dot products implied by the tile.

Compared with independent scalar multiplies and adds on generic CUDA cores, tensor cores deliver much higher throughput on the tile shapes they support. That is why high-performance GEMM kernels are structured as nested loops of **global → shared → register MMA** around those fixed geometries.

Croktile keeps you at the **tile** level: you name accumulator and operand tiles, load from shared, issue MMA, and store—without hand-writing PTX fragment layouts or matching CUDA C++ `fragment` types by hand.

For matrices **A** (M×K), **B** (K×N), and **C** (M×N), you want **C += A B**. No GPU holds all of **A** and **B** on-chip, so real kernels **tile** M, N, and K.

**Block tiles** carve the output into rectangles each CUDA block will eventually own. **Warp tiles** (on Ampere MMA) carve those rectangles so each warp works on 16×16 outputs at a time, accumulating along K in steps of 16.

Croktile makes those levels explicit in the `__co__` function: outer **`parallel … : block`** for the block grid, inner **`parallel … : group`** for warps, and **`foreach`** loops over K that line up with **`MATMUL_TILE_K`** and **`MATMUL_MMA_K`**.

Every tensor-core matmul in Croktile follows the same rhythm. You **initialize** the accumulator with **`mma.fill`**.

You **loop over K**, possibly nested across block tile and warp sub-tile. Each iteration brings the next slices of **A** and **B** into **shared memory**—here with **`dma.copy`**; Chapter 4 showed TMA on Hopper.

You **load** operand tiles from shared into MMA operand registers with **`mma.load`**, then **accumulate** with **`mma.row.row`** (or another layout-specific variant) so **C += A × B** runs on tensor cores. Finally you **store** with **`mma.store`** to shared, and later (outside the warp loop) back to global.

The accumulator **`mc`** is an **opaque register-resident tile**: you do not manually size `wmma::fragment` types or choose lane mappings. The compiler maps **`mc`**, **`ma`**, and **`mb`** to the correct MMA register file layout for your target architecture.

Conceptually, one **`mma.row.row`** applies to **fixed** M, N, and K extents baked into the instruction (16×16×16 here). A full matrix multiply has a much larger **K**.

The kernel therefore **streams** along K: each iteration brings another **K-slab** into shared, loads 16×K_sub chunks into **`ma`** and **`mb`**, and **adds** the new partial product into the **same** **`mc`**. That is standard blocked GEMM—Croktile simply names the stages (`dma.copy`, `mma.load`, `mma.row.row`) instead of hiding them inside a single opaque library call.

If this feels familiar after Chapter 4, that is intentional. TMA answered “how does this slab land in shared?” MMA answers “now that it is in shared, how does a warp turn it into tensor-core math?” The two chapters are two halves of one pipeline.

## SM86 Ampere: one warp, one 16×16×16 tile

On **SM86**, the synchronous MMA path (`mma.sync`-class instructions under the hood) is scoped to a **single warp** (32 threads). In the Croktile function, that corresponds to a **`parallel`** region annotated **`: group`**—one cooperative thread group the size of one warp, not four warps as on Hopper WGMMA.

The outer **`parallel {block_m, block_n}`** launches one CUDA thread block per output tile of size **`MATMUL_TILE_M × MATMUL_TILE_N`**. Indices **`block_m`** and **`block_n`** select which **M** and **N** stripes of the result matrix this block owns.

Think of **`block_m`** as “which row bundle of **C**” and **`block_n`** as “which column bundle of **C**.” Together they pinpoint the **top-left corner** of the tile this block must eventually write, modulo **`.at(block_m, iv_k)`**-style addressing into **lhs** and **rhs** as K advances.

**`shared f16 [MATMUL_TILE_M, MATMUL_TILE_N] output_s`** is the block’s scratchpad for the completed partial result of that tile. Warps write their MMA results into disjoint sub-rectangles of **`output_s`**; a final **`dma.copy`** from **`output_s`** to global **`output`** (shown in the async listing below) completes the write-back, typically issued cooperatively by the whole block as in the full benchmark.

Using shared memory as an intermediate for **C** is typical: MMA instructions write register tiles, and **`mma.store`** lands those tiles in a space the **entire block** can see. You can then perform vectorized or coalesced global stores once the tile is complete and consistent.

Inside the block, **`parallel {warp_m, warp_n} by […] : group`** enumerates warps. With **`MATMUL_TILE_M = MATMUL_MMA_M = 16`** and the same for N, the **`by`** extents are **1×1**: a single warp handles the entire block tile.

If you widen the block tile to multiples of 16, **`warp_m`** and **`warp_n`** count how many **independent** 16×16 output patches exist; each patch gets its own warp and its own **`mc`**. **`: group`** means “one warp’s worth of threads”—the right granularity for pre-Hopper **per-warp** MMA on Ampere.

If you used **`: group-4`** here on SM86, you would be describing four-warpgroup cooperation the older ISA does not treat as a single MMA issuer in this style.

**`mc = mma.fill 0.0`** creates a fresh **accumulator tile** in MMA accumulator registers, initialized to zero (floating-point zero compatible with FP16 accumulation). Variable **`mc`** holds that tile for the rest of the warp’s K-loop.

This is the Croktile spelling of “zero my **C** fragment before the K dimension sweep.” Nothing in your source names individual lanes; **`mc`** is the whole 16×16 accumulator as a first-class value.

**`foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)]`** walks the **K** dimension in steps of **`MATMUL_TILE_K`**. **`dma.copy` … `=> shared`** stages a **K-slab** of **`lhs`** and **`rhs`** into shared buffers **`lhs_load_s`** and **`rhs_load_s`**.

This is ordinary async-capable DMA-style copy, not TMA (TMA is the Hopper-era accelerator from Chapter 4). On Ampere matmuls you still get excellent performance with well-placed **`dma.copy`** / **`dma.copy.async`**; you are not “missing out” on tensor cores by skipping TMA—you are only skipping Hopper’s dedicated bulk loader.

The inner **`foreach iv_warp_k in [cdiv(MATMUL_TILE_K, MATMUL_MMA_K)]`** subdivides the K slab into **16-element** chunks along K, matching the **K=16** depth of one **`mma.sync.m16n16k16.f16`**-class operation.

For each K chunk, **`ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k)`** loads the warp’s **A** operand tile from shared memory into MMA **A** registers. The **`chunkat`** selects the **M × 16** slab of the staged **lhs** tile that corresponds to this warp’s row offset and the current K slice.

**`mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k)`** loads the **B** operand tile similarly from the staged **rhs** tile. **`mma.row.row mc, ma, mb`** performs **C += A × B** using tensor cores.

The **`row.row`** suffix states the **layout contract**: both **`ma`** and **`mb`** are interpreted as **row-major** operand tiles for this instruction variant. Other Croktile MMA variants exist for different matrix layouts; picking the wrong one is a correctness bug, not a silent transpose.

After the loops, **`mma.store mc, output_s.subspan(MATMUL_MMA_M, MATMUL_MMA_N).at(warp_m, warp_n)`** writes the warp’s accumulated **C** tile from accumulator registers into the correct sub-rectangle of shared memory. Equivalently, benchmarks often write **`mma.store mc, output_s.chunkat(warp_m, warp_n)`** when the chunking API matches the MMA tile shape directly.

The kernel below sketches the structure. Tile sizes match the benchmark defaults: **`MATMUL_MMA_*`** and **`MATMUL_TILE_*`** all equal **16**, so one block tile equals one MMA tile along M and N. Global tensors use **row-major** **`lhs[M, K]`** and **rhs stored as `[N, K]`** so that each row of **`rhs`** is a length-**K** vector (a common layout for this style of kernel).

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_TILE_N)] : block {
    shared f16 [MATMUL_TILE_M, MATMUL_TILE_N] output_s;
    parallel {warp_m, warp_n} by [cdiv(MATMUL_TILE_M, MATMUL_MMA_M), cdiv(MATMUL_TILE_N, MATMUL_MMA_N)] : group {
      mc = mma.fill 0.0;
      foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
        lhs_load_s = dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => shared;
        rhs_load_s = dma.copy rhs.subspan(MATMUL_TILE_N, MATMUL_TILE_K).at(block_n, iv_k) => shared;
        foreach iv_warp_k in [cdiv(MATMUL_TILE_K, MATMUL_MMA_K)] {
          ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k);
          mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k);
          mma.row.row mc, ma, mb;
        }
      }
      mma.store mc, output_s.subspan(MATMUL_MMA_M, MATMUL_MMA_N).at(warp_m, warp_n);
    }
  }
}
```

This is the conceptual core of **`matmul_f16_dyn_sm86.co`** in the Croktile benchmark tree. The shipping file adds **asynchronous** **`dma.copy.async`**, explicit **`wait`** barriers, and uses **`chunkat`** for output staging—good for overlapping memory traffic with math.

For learning MMA, the synchronous copy version above is easier to read; the lifecycle is identical. For completeness, the MMA portion of **`matmul_f16_dyn_sm86.co`** as it appears in the repository follows—note **`dma.copy.async`**, **`wait`**, and **`chunkat`** on the store target:

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_TILE_N)] : block {
  shared f16 [MATMUL_TILE_M, MATMUL_TILE_N] output_s;
  parallel {warp_m, warp_n} by [cdiv(MATMUL_TILE_M, MATMUL_MMA_M), cdiv(MATMUL_TILE_N, MATMUL_MMA_N)] : group {
    mc = mma.fill 0.0;
    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      lhs_load_s = dma.copy.async lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => shared;
      rhs_load_s = dma.copy.async rhs.subspan(MATMUL_TILE_N, MATMUL_TILE_K).at(block_n, iv_k) => shared;
      wait lhs_load_s, rhs_load_s;

      foreach {iv_warp_k} in [cdiv(MATMUL_TILE_K, MATMUL_MMA_K)] {
        ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k);
        mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k);
        mma.row.row mc, ma, mb;
      }
    }
    mma.store mc, output_s.chunkat(warp_m, warp_n);
  }
  dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_TILE_N).at(block_m, block_n);
}
```

The **`wait`** synchronizes the in-flight copies before **`mma.load`** reads shared memory. That is classic software pipelining; you can extend it once you are comfortable with the bare **`mma`** sequence.

The tutorial uses 16×16×16 tiles for clarity. The actual **`matmul_f16_dyn_sm86.co`** comments note aggressive register usage (on the order of **255 registers per thread** in optimized builds), which caps how many warps you can pack into a block.

When you scale tile sizes up, always check occupancy and register limits. Croktile makes the dataflow obvious, but the hardware still enforces physics.

## SM90 Hopper: WGMMA and warp groups

Hopper introduces **Warpgroup Matrix Multiply Accumulate (WGMMA)**: the same **C += A × B** idea, but the instruction is issued cooperatively by **four warps** (**128 threads**), and operand tiles are larger in the **M** and/or **N** dimensions subject to ISA rules.

In Croktile, that wider cooperation appears as **`: group-4`** instead of **`: group`**. A canonical excerpt from **`matmul_f16_dyn_sm90.co`** looks like the following (abbreviated to the MMA body; TMA loads were covered in Chapter 4):

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
  mc = mma.fill.f16 0.0f;
  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    tma.copy.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    tma.copy.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s;

    foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
      parallel p by 1 : group-4 {
        ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.chunkat(_, iv_warp);
        mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
  mma.store mc, output_s;
  tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

| Aspect | SM86 (Ampere, this chapter) | SM90 (Hopper, Chapter 4 + above) |
|--------|-----------------------------|-----------------------------------|
| Thread scope | **One warp** — `: group` | **Four warps** — `: group-4` |
| Global → shared | **`dma.copy`** / **`dma.copy.async`** | **`tma.copy.swiz<N>`** (TMA + swizzle metadata) |
| Operand loads | **`mma.load`** from staged shared layout | **`mma.load.swiz<N>`** — operands match **swizzled** shared layout from TMA |
| Accumulator setup | **`mma.fill 0.0`** | **`mma.fill.f16 …`** here; other kernels may use **`mma.fill.f32`** for FP32 accumulators with FP16 operands |
| Core math op | **`mma.row.row mc, ma, mb`** | Same mnemonic—WGMMA is a wider, warpgroup-scoped family under the hood |
| Store | **`mma.store`** into a per-warp tile carve | **`mma.store`** into shared sized for the **warpgroup** output |

Read this side by side with the SM86 version. The **lifecycle** is intentionally parallel: **fill**, **load operands**, **accumulate along K**, **store**.

What changes is **how many threads participate in one MMA**, how data must be **laid out in shared memory** (Hopper’s **`.swiz<128>`**-style loads expect a particular swizzle pattern, commonly tied to **`2 * TILE_K`** constraints in the benchmark headers), and whether global→shared uses **DMA copies** or **TMA**.

So Chapter 4’s **`tma.copy.swiz`** and this chapter’s **`mma.load.swiz`** are a matched pair: the first establishes the shared-memory pattern; the second asserts that operands are read **consistent with that pattern** when the warpgroup issues WGMMA.

Warpgroup MMA can widen the **accumulator** relative to the **operands**. A frequent pattern on Hopper-class kernels is to keep **FP16** (or narrower) for **`ma`** and **`mb`** while accumulating into **FP32** registers for **numerical stability** on large **K**.

The Croktile surface may expose that as a different **`mma.fill.***` precision or related typing on the **`mc`** handle; the exact spelling depends on the generator version and target. The benchmark you are reading (`matmul_f16_dyn_sm90.co`) uses **`mma.fill.f16 0.0f`** to match its end-to-end FP16 story—always read the **header constants** next to the kernel when you port ideas between projects.

If you see **`mma.fill.f32`** in other kernels, think: “same **`mma.row.row`** math, wider **C** register file.” The **load** instructions still feed **FP16** tiles in the common case; it is the **accumulator** that gained headroom.

## Throughput, register tiles, and what Croktile handles for you

It is tempting to imagine the inner K loop as “heavy work.” On a well-tuned kernel, the opposite is closer to the truth: **tensor cores are so fast** that the kernel often becomes **memory bound**.

What usually limits you is how quickly you can **`dma.copy`** or **TMA** the next K-slab into shared, and how cleanly you **pipeline** those copies with **`mma.load`**. That is why the repository’s SM86 example reaches for **`dma.copy.async`** and explicit **`wait`**: the arithmetic in **`mma.row.row`** is not the bottleneck you optimize first; **latency hiding** is.

When you profile, interpret high **tensor core utilization** with care. Croktile makes the **instruction sequence** legible; occupancy, **L2** traffic, and **bank conflicts** in shared still decide whether you are feeding those units fast enough.

If you expand tiles, watch **shared footprint** and **register spills**—both show up as sudden cliffs in effective TFLOPS.

In raw CUDA C++, WMMA and PTX require you to declare **`fragment`** types with specific shapes and manually **`load_matrix_sync`**, **`mma_sync`**, and **`store_matrix_sync`**, keeping track of **row-major versus column-major** variants and **`ldmatrix`** staging patterns.

Croktile pushes that detail below the surface. **`mc`**, **`ma`**, and **`mb`** are **logical MMA tiles** in accumulator and operand register files. **`mma.load`** and **`mma.store`** connect **shared-memory slices** to those registers using the same **chunking** / **`subspan`** vocabulary you already use for DMA and TMA.

**`mma.row.row`** selects the **instruction semantic** (here: row-major times row-major into the accumulator). You still must choose **consistent layouts** (correct **`mma.*`** variant for your data), tile sizes that **divide** the hardware MMA geometry (**16** on SM86 for this FP16 path), and a **thread hierarchy** (**`: group`** versus **`: group-4`**) that matches the ISA.

Croktile does not remove those constraints—it makes them **readable** and keeps the **register mapping** out of your way.

Before you treat a kernel as “done,” skim this list. **Shapes**: do **`MATMUL_TILE_*`** and **`MATMUL_MMA_*`** line up so every **`chunkat`** / **`subspan`** is a whole number of MMA tiles? **Layouts**: does **`mma.row.row`** (or your chosen variant) match how **`lhs`** and **`rhs`** are actually stored in global memory?

**Synchronization**: after async copies, did you **`wait`** (or equivalent) before **`mma.load`**? **Scope**: is **`parallel … : group`** used for SM86-style per-warp MMA, and **`group-4`** only where WGMMA is intended?

It helps to name what you are *not* writing. In PTX-flavored workflows you might thread **A** and **B** through **`ldmatrix`** with a specific **layout** bit, then feed the result into **`mma.sync`**.

Croktile’s **`mma.load`** is the composite “get a tile from shared into the operand register file” step; **`mma.row.row`** is the “multiply with this major-ness assumption” step; **`mma.store`** reverses the path for **C**. That decomposition mirrors how experts think about kernels—memory → math → memory—even when the underlying SASS bundles several micro-ops.

When you debug a wrong result, suspect **layout first** (row versus column major, and whether **rhs** is **`[N,K]`** versus **`[K,N]`**), then **indexing** (which **`block_m`**, **`block_n`**, and K slice you attached with **`.at`** / **`chunkat`**), then **async ordering** if you introduced **`dma.copy.async`**.

The **`mma.row.row`** token is read as **“treat the left operand as row-major, the right operand as row-major.”** Croktile may offer additional MMA variants for other major orders or transposed views; they exist because the hardware instruction set exposes **multiple encodings** for the same mathematical multiply, each assuming a particular **register packing**.

If you have ever fought **`layout_t`** arguments in WMMA, this is the same decision with a choreographic name. Choosing correctly is not a performance hint—it is **required for correctness**.

The **global** tensor layout, the **shared** staging pattern, and the **MMA** variant must agree. When in doubt, draw a 4×4 **toy matrix** on paper, mark how elements sit in memory, and check which **row** or **column** interpretation your **`chunkat`** slices preserve.

With **`MATMUL_MMA_M = MATMUL_MMA_N = 16`**, each **`mma.store`** commits **256** output elements worth of **C** for that warp (in FP16). Each **`mma.row.row`** fuses a **16×16** update that consumed a **16×16×16** contraction along K for that step.

Walking **`iv_k`** and **`iv_warp_k`** simply **sums** those contractions into the same **`mc`** until the full **K** dimension for the block tile is covered.

If you shrink **`MATMUL_TILE_K`** below **16**, you still need **`MATMUL_MMA_K`** to remain **16** for this instruction shape—the inner loop’s trip count would become fractional unless you pad or switch ISA. That is why tutorials and benchmarks usually keep **tile K** as a multiple of **16** for FP16 MMA on Ampere.

## Looking ahead

On **SM86**, tensor cores show up as a **per-warp** pipeline: **`mma.fill`** zeros **`mc`**, **`mma.load`** pulls **`ma`** / **`mb`** from shared, **`mma.row.row`** does **C += AB**, and **`mma.store`** writes **`mc`** back to shared under **`parallel … : group`**. **`dma.copy`** or **`dma.copy.async`** with **`wait`** stages K-slabs from global memory. On **Hopper**, the same **fill → load → multiply → store** rhythm pairs **TMA** with **`mma.load.swiz<N>`** and **`parallel … : group-4`** for WGMMA.

**`mc = mma.fill 0.0`** (or **`mma.fill.f16`** / **`mma.fill.f32`** where the kernel demands) holds the accumulator for the full K sweep. **`mma.row.row mc, ma, mb`** is row-major **C += A × B** on tensor cores; **`mma.store mc, output_s.…`** hands **C** to shared for global write-back. **`: group`** is one warp—the SM86 MMA scope; **`: group-4`** is four warps on SM90 and is not a drop-in on Ampere-style kernels. **`mc`**, **`ma`**, and **`mb`** remain **opaque** logical tiles; the compiler maps lanes and register counts.

Widen **`MATMUL_TILE_M`** / **`MATMUL_TILE_N`** to 32 with MMA still 16×16×16 and reconcile warp count inside **`parallel {warp_m, warp_n} … : group`** with the register-pressure comments in **`matmul_f16_dyn_sm86.co`**. Diff that file against **`matmul_f16_dyn_sm90.co`** for “same lifecycle, different mechanism” versus Hopper-only lines, and try omitting **`wait`** in a scratch copy to see verification fail—**memory visibility** before **`mma.load`**. For reference, read **`croktile/benchmark/performance/matmul/matmul_f16_dyn_sm86.co`** and **`matmul_f16_dyn_sm90.co`** end to end (compare **`tma.copy.swiz`** with **`mma.load.swiz`**). On new matmul **`.co`** files, skim **`mma.`** first, then **`dma.copy`** versus **`tma.copy`**, then **`group`** versus **`group-4`**. Later topics—persistent kernels, warp specialization, FP8 and other dtypes in the same tree—reuse this skeleton. **Croktile describes orchestration**: you still allocate **`global`** buffers, launch, and validate like any CUDA program; carry the Croktile **recipe** into the next chapter. Newer GPUs may change thread scope, swizzles, and copy engines, but **fill → load → multiply → store** remains the backbone.
