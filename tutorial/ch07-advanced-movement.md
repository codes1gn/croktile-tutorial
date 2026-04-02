# Advanced Data Movement: TMA, Swizzle, and Irregular Access

Chapter 6 closed the loop on **safe pipelining**: producers issue `dma.copy` into shared-memory stages, consumers wait on **events**, and the K-tile loop overlaps loads with MMA while **double- or multi-buffering** keeps each phase from stepping on the other. Every one of those transfers was **software-driven**: warps cooperated, each lane helped compute addresses, and the program issued loads the way ordinary CUDA does—just expressed more cleanly in Croktile.

This chapter stays on one thread—**advanced data movement**—but changes the mechanism. **`dma.copy`** is still the right mental model for older GPUs and for stores that remain DMA-sized, but on **Hopper (SM90)** you will usually replace ingress with **`tma.copy`**. The **Tensor Memory Accelerator (TMA)** is a dedicated unit that accepts a **tensor descriptor** and performs multi-dimensional tile movement with **negligible thread work**, instead of burning warps on address arithmetic. Alongside TMA, **swizzle** rearranges how columns land in shared memory so a warp’s simultaneous accesses do not all map to the **same 4-byte bank**—the root of **bank conflicts**, where the hardware serializes what should have been a parallel read. Finally, Croktile’s **`view` / `from`**, **strided `subspan`**, **`.zfill`**, and **`span_as`** cover **irregular and ragged** access when tiles are not neat multiples of the tensor or when windows start at arbitrary offsets.

![Software DMA vs TMA: cooperative thread loads vs descriptor-driven hardware tensor copy](../assets/images/ch07/fig1_tma_vs_dma_dark.png#only-dark)
![Software DMA vs TMA: cooperative thread loads vs descriptor-driven hardware tensor copy](../assets/images/ch07/fig1_tma_vs_dma_light.png#only-light)

## `tma.copy`: Hardware Tensor Movement

The surface syntax mirrors `dma.copy`:

```choreo
tma.copy lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => lhs_load_s;
```

**Same arrow, new engine.** The line looks like `dma.copy`, but the work shifts from **software** to **hardware** as follows.

Same **source expression**, same **`=>` destination** form. The difference is **who does the work**:

- **DMA path.** Threads **cooperate** to cover the tile; each lane participates in **address math** and load issue. Throughput scales with how well you keep those loads **bank-friendly**—often a fight on its own.
- **TMA path.** One logical **descriptor-based** operation describes the tensor slice; the **TMA** expands that into the correct **multi-dimensional** addressing and moves the **whole tile** as a unit. Producer warps can **overlap** other work or disappear into a thinner launch pattern because the **hardware**, not a full warp’s worth of threads, owns the transfer semantics.

**What this buys you.** You still **synchronize** on the **whole tile** (with events or the same pipeline discipline as Chapter 6), but you drop the **per-thread** load choreography for operand ingress. The compiler builds the **tensor descriptor** from your `__co__` signature and global layouts; in typical kernels you only write `tma.copy` where you once wrote `dma.copy`.

## Swizzle and Bank Conflicts

Shared memory is **striped into banks** (32 banks, 4 bytes per bank on common paths). When **multiple lanes in a warp** touch **different addresses that map to the same bank** in the same cycle, the hardware **serializes** those accesses—a **bank conflict**. Dense **row-major** tiles often put **consecutive columns** where a warp’s consecutive lanes want them, which can create **2-way, 4-way, or worse** conflicts and **cut effective bandwidth** sharply.

**Swizzle** applies a **fixed XOR-style remapping** to column indices within each row so that the layout threads **actually read** spreads accesses across banks. Croktile exposes it on the copy and on the MMA load so **ingress and math agree**:

```choreo
tma.copy.swiz<3> src => dst;
```

**Ingress.** The copy lands bytes in shared memory using swizzle pattern **`N`**.

```choreo
ma = mma.load.swiz<3> lhs_load_s.chunkat(_, iv_warp);
```

**MMA read path.** Operand loads must use the **same** **`swiz<N>`** so addresses match the staged layout.

**Swizzle levels.** The template argument sets the **granularity**: `swiz<0>` is identity, then **64 B, 128 B, and 256 B** XOR patterns for `<1>`, `<2>`, and `<3>`. Larger granularities defeat **wider** conflict patterns but require **tile extents** that line up with that granularity.

**Matching rule.** The `<N>` on **`tma.copy.swiz<N>`** must match **`mma.load.swiz<N>`**. If you load with plain `mma.load` from **`swiz<3>`** data, addresses disagree and you read **garbage**. The compiler does **not** enforce the pairing—it is a **correctness invariant** you maintain. (As introduced in [Chapter 4's MMA variant table](ch04-mma.md#load-variants), `mma.load.swiz<N>` is part of the MMA load family.)

![Bank conflicts without swizzle vs XOR swizzle spreading warp lanes across banks](../assets/images/ch07/fig2_swizzle_dark.png#only-dark)
![Bank conflicts without swizzle vs XOR swizzle spreading warp lanes across banks](../assets/images/ch07/fig2_swizzle_light.png#only-light)

## TMA in a Pipelined Matmul

The pipeline **skeleton** from Chapter 6 is unchanged: **ring of stages**, **`wait` / `trigger` on events**, **MMA commit**, and a **consumer** that drains tiles while a **producer** fills the next slot. Here the producer swaps **`dma.copy`** for **`tma.copy.swiz<3>`** and the consumer swaps **`mma.load`** for **`mma.load.swiz<3>`** on the staged buffers. The example below is a **Hopper FP16** matmul with the same **1P1C** split (`parallel p1 by 2 : group-4`) you have seen before:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
    shared f16 [MATMUL_STAGES * MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_STAGES * MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait empty[stage];
          tma.copy.swiz<3> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)
            => lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0);
          tma.copy.swiz<3> rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k)
            => rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0);
          trigger full[stage];
        }
      }

      inthreads.async (p1 == 1) {
        mc = mma.fill.f16 0.0f;
        foreach {s} in [MATMUL_STAGES] { trigger empty[s]; }
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait full[stage];
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load.swiz<3> lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mb = mma.load.swiz<3> rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
          mma.commit;
          trigger empty[stage];
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**Pipeline parity.** Compared to the Chapter 6 **`dma.copy`** version, only the **ingress** and **operand load** lines change; **events**, **staging indices**, and **commit** stay the same. **TMA** removes cooperative **per-thread** address work on the loads; **swizzle** aligns shared layout with **MMA** access patterns. The **writeback** to global memory still uses **`dma.copy`** here—choose TMA or DMA for stores according to your target and compiler support.

## Handling Irregular Access

Uniform tiling with **`chunkat`** and **`subspan(...).at(...)`** covers many kernels. Real workloads also need **windows at arbitrary offsets**, **strides between tiles**, **partial tiles** at boundaries, and **layout reinterpretation**—the subsections below collect those tools under one heading.

### Arbitrary-offset windows: `view` and `from`

**`view(M, N).from(row, col)`** defines an **`M × N`** rectangle starting at **`(row, col)`** in the underlying tensor—**no** requirement that the origin aligns to a precomputed tile grid.

```choreo
patch = matrix.view(16, 16).from(37, 50);
```

**Fixed window, free origin.** This is a **`[16, 16]`** slice starting at row **`37`**, column **`50`**—alignment is **not** required (use **`.zfill`** if the window crosses the tensor edge).

**When to use it.** **`chunkat`** needs the tensor divided evenly into a fixed number of chunks; **`view(...).from(...)`** does not. Prefer **`chunkat`** for **regular** tiling and **`view` / `from`** when the window is **ragged** or **runtime-positioned**.

```choreo
expert_lhs = lhs.view(expert_M, K).from(expert_offset, 0);
dma.copy expert_lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => shared;
```

**MoE-style GEMM.** In **mixture-of-experts** stacks, each expert’s token batch often starts at a **dynamic row** `expert_offset`. Slicing with **`view` / `from`** rewires the operand **before** the rest of the pipeline—DMA or TMA, MMA, events—**unchanged**.

### Strided tiles: `.subspan`, `.step`, and `.at`

**`subspan(M, K).at(i, j)`** selects the tile anchored at logical tile indices **`(i, j)`** with extent **`[M, K]`**. Adding **`.step(sM, sK)`** spaces tiles **`sM`** rows and **`sK`** columns apart instead of packing them contiguously:

```choreo
matrix.subspan(16, 16).step(32, 32).at(i, j);
```

**Strided tiling.** Neighboring tile indices advance by **`(sM, sK)`** in **global** coordinates, not necessarily by the tile size.

**What `.step` means.** Tile **`(0, 0)`** still starts at **`(0, 0)`**, but tile **`(1, 0)`** starts at **`(32, 0)`** and **`(0, 1)`** at **`(0, 32)`**. Omitting **`.step`** uses a step equal to the **tile size** along each axis—the **packed** case.

**Typical uses:** skipping **padding or guard bands**, **overlapping** stencils where the step is **smaller** than the extent, or matching an **outer layout** that is not dense tile-major.

### Zero-padding: `.zfill`

When **`M` or `K`** is **not** a multiple of the tile size, the **last** tile along an axis is **partial**. Reading past the tensor’s edge is **undefined** unless you explicitly **pad**.

```choreo
tma.copy.swiz<3> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k).zfill
  => lhs_load_s;
```

**Semantics.** **`.zfill`** applies to the **source** side of a copy: out-of-range elements are **written as zero** in the destination tile. Zeros **contribute nothing** to a GEMM accumulation, so the **MMA loop** can stay **uniform** while remaining **mathematically** correct for **partial** edges.

### Layout reinterpretation: `span_as`

**`span_as`** **reinterprets** a buffer’s linear storage as another **shape** with the **same element count**—**no** copy.

```choreo
flat_buffer.span_as([rows, cols])
```

**View-only reshape.** Element count is preserved; only the **logical** rank changes.

```choreo
strip_load = dma.copy data.chunkat(tile) => shared;
tile_2d = strip_load.data.span_as([tile_m, tile_k]);
ma = mma.load tile_2d.chunkat(_, iv_warp);
```

**1D staging to 2D MMA.** **`span_as`** exposes the loaded strip as a **matrix** for **`chunkat`** without an extra copy.

**Contract.** **`rows * cols`** must equal the **span length** of the underlying storage or the compiler **rejects** the program.

## Chapter Summary

| Idea | Role in the “advanced movement” story |
|------|--------------------------------------|
| **`dma.copy` (Ch. 6)** | Software-driven pipelined loads—threads + address math; baseline for comparison. |
| **`tma.copy` / `tma.copy.swiz<N>`** | Descriptor-driven **Hopper** ingress; hardware **multi-dimensional** tile fetch with **minimal thread overhead**. |
| **Swizzle + `mma.load.swiz<N>`** | Align **shared layout** with **MMA** reads; avoid **bank conflicts** via **XOR** remapping—**matching** `N` on copy and load. |
| **`view` / `from`** | **Arbitrary-offset** rectangular windows for **ragged** or **runtime** slice origins. |
| **`.subspan(...).step(...).at(...)`** | **Strided** tiling—overlap, padding skips, or non-packed layouts. |
| **`.zfill`** | **Safe partial tiles** by zero-filling out-of-bounds elements on copy. |
| **`span_as`** | **Zero-copy** shape **reinterpretation** for staging buffers. |

## New Syntax (quick reference)

| Syntax | Meaning |
|--------|---------|
| `tma.copy src => dst` | TMA hardware tensor copy |
| `tma.copy.swiz<N> src => dst` | TMA copy with swizzle mode `N` (0–3) |
| `mma.load.swiz<N> src` | MMA operand load consistent with swizzle `N` |
| `tensor.view(M, N).from(r, c)` | Arbitrary-offset `M × N` window |
| `.subspan(M, K).step(sM, sK).at(i, j)` | Strided tile selection |
| `.zfill` | Zero-fill out-of-bounds elements on the copy source |
| `span_as([dims])` | Reinterpret linear storage as a shaped tensor |
| `parallel.async ... : block` | Non-blocking async kernel launch (see [Ch 5](ch05-branch-control.md)) |
| `stream s` | Bind the kernel body to CUDA stream `s` (see [Ch 5](ch05-branch-control.md)) |

The [next chapter](ch08-cpp-interop.md) steps past pure Croktile choreography into **C++ interop**: **register hints**, **preprocessor guards**, and **inline PTX** when you need to **drop down** to the metal beside generated code.
