# Advanced Data Movement: TMA, Swizzle, and Irregular Access

Chapter 2 introduced `dma.copy` as Croqtile's general data movement primitive — a way to move rectangular tiles between memory spaces using a simple `src => dst` arrow syntax. Under the hood, DMA copies are **software-driven**: every thread in the warpgroup participates in address computation and load issuance. The hardware sees dozens of individual load instructions, one per thread, that collectively assemble a tile in shared memory.

NVIDIA's Hopper GPUs (SM90+) add a second mechanism: the **Tensor Memory Accelerator (TMA)**. TMA is a physical hardware unit sitting near the L2 cache / shared memory interface. Instead of threads cooperatively issuing loads, a single thread issues one **descriptor-based** instruction, and the TMA engine handles everything: multi-dimensional address computation, boundary clamping, and the actual data transfer.

A common misconception is that TMA moves data faster than DMA. It does not — both paths ultimately travel the same HBM → L2 → shared-memory data highway at the same bandwidth. The advantage is elsewhere: TMA has its own **dedicated engine** that runs independently of the SM's instruction pipeline. Once the descriptor-based instruction is issued, the TMA hardware performs the transfer in the background while the issuing thread (and all other threads in the warpgroup) are free to execute compute instructions. With `dma.copy`, the threads that participate in address math and load issuance are occupied for the duration of the transfer; with `tma.copy`, they can overlap the transfer with MMA or other work. In a warp-specialized pipeline (Chapter 5–6), this is the difference between a producer that blocks on loads and a producer that fires-and-forgets.

The two paths differ in interface, not in throughput:

- **`dma.copy`** — Threads cooperatively issue loads. The programmer controls nothing about address patterns — Croqtile handles coalescing automatically. Flexible: works on any CUDA GPU since it compiles to standard load instructions.
- **`tma.copy`** — One descriptor-based instruction. The TMA hardware expands it into multi-dimensional addressing, applies swizzle, and handles boundary clamping. Hopper (SM90+) only. The descriptor is built by the compiler from your `__co__` signature and global layout.

Croqtile exposes TMA through the same arrow syntax as DMA — `tma.copy` instead of `dma.copy`. The rest of this chapter covers TMA, swizzle, the irregular-access utilities for real-world edge cases, and how Croqtile's design philosophy trades generality for guaranteed performance.

## `tma.copy`: hardware tensor movement

The surface syntax mirrors `dma.copy`:

```choreo
tma.copy lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => lhs_load_s;
```

Same source expression, same `=>` destination form. The difference is **who does the work**:

![TMA descriptor to hardware tile fetch: descriptor fields, TMA unit, and SMEM tile](../assets/images/ch07/fig3_tma_descriptor_dark.png#only-dark)
![TMA descriptor to hardware tile fetch: descriptor fields, TMA unit, and SMEM tile](../assets/images/ch07/fig3_tma_descriptor_light.png#only-light)

- **DMA path.** Threads cooperate to cover the tile; each lane participates in address math and load issue. Throughput scales with how well you keep those loads bank-friendly.
- **TMA path.** One descriptor-based operation describes the tensor slice; the TMA hardware expands it into multi-dimensional addressing and moves the whole tile as a unit. Producer threads can overlap other work because the hardware, not a warp's worth of threads, owns the transfer.

**What this buys you.** You still synchronize on the whole tile (with events or the pipeline discipline from Chapter 6), but you drop the per-thread load choreography. The compiler builds the tensor descriptor from your `__co__` signature and global layouts.

![Software DMA vs TMA: cooperative thread loads vs descriptor-driven hardware tensor copy](../assets/images/ch07/fig1_tma_vs_dma_dark.png#only-dark)
![Software DMA vs TMA: cooperative thread loads vs descriptor-driven hardware tensor copy](../assets/images/ch07/fig1_tma_vs_dma_light.png#only-light)

## Swizzle and bank conflicts

Shared memory is striped into **32 banks** (4 bytes per bank). When multiple lanes in a warp touch different addresses that map to the same bank in the same cycle, the hardware **serializes** those accesses — a **bank conflict**. Dense row-major tiles often create 2-way, 4-way, or worse conflicts that cut effective bandwidth.

**Swizzle** applies a fixed XOR-style remapping to column indices within each row, spreading accesses across banks. Croqtile exposes it on **both DMA and TMA**, with the same syntax and the same effect:

```choreo
dma.copy.swiz<3> src => dst;       // software DMA with swizzle
tma.copy.swiz<3> src => dst;       // hardware TMA with swizzle
```

The copy lands bytes in shared memory using swizzle pattern `N`. The MMA read path must use the same `swiz<N>` so addresses match the staged layout:

```choreo
ma = mma.load.swiz<3> lhs_load_s.chunkat(_, iv_warp);
```

Swizzle is not a TMA-specific feature. In Croqtile, `dma.copy.swiz<N>` and `tma.copy.swiz<N>` produce identical shared-memory layouts. The difference is only in how the data gets there (thread-cooperative loads vs. descriptor-based hardware transfer), not in how the data is arranged once it arrives.

**Swizzle levels.** The template argument sets the granularity: `swiz<0>` is identity, then 64B, 128B, and 256B XOR patterns for `<1>`, `<2>`, and `<3>`. Larger granularities defeat wider conflict patterns but require tile extents that line up with that granularity.

**Matching rule.** The `<N>` on the copy must match `mma.load.swiz<N>`. If you load with plain `mma.load` from `swiz<3>` data, addresses disagree and you read garbage. The compiler does not enforce the pairing — it is a correctness invariant you maintain. (As introduced in [Chapter 4](ch04-mma.md#new-syntax), `mma.load.swiz<N>` is part of the MMA load family.)

![Bank conflicts without swizzle vs XOR swizzle spreading warp lanes across banks](../assets/images/ch07/fig2_swizzle_dark.png#only-dark)
![Bank conflicts without swizzle vs XOR swizzle spreading warp lanes across banks](../assets/images/ch07/fig2_swizzle_light.png#only-light)

## Why the restricted interface works: expressiveness vs performance

In raw CUDA, a programmer implementing data movement has enormous freedom: arbitrary pointer arithmetic, variable-stride access, hand-computed bank-conflict avoidance, custom swizzle formulas. This flexibility is a double-edged sword. The space of possible data movement patterns is vast, but the subset that actually performs well on GPU hardware is narrow — it requires coalesced global loads, conflict-free shared-memory access, and correct swizzle alignment. Getting any of these wrong silently degrades throughput by 2–32×.

Croqtile takes the opposite approach: it restricts the expressible patterns to those that are **guaranteed to be in the performance sweet spot**. When you write `dma.copy` or `tma.copy`, the compiler automatically handles coalesced access, bank-conflict-free layout, and swizzle alignment. There is no way to accidentally write a strided, uncoalesced global load or a bank-conflicted shared-memory layout — the syntax simply does not allow it.

![Expressiveness vs performance: CUDA's wide range includes many slow patterns; Croqtile's restricted range maps entirely to the performance sweet spot](../assets/images/ch07/fig8_expressiveness_dark.png#only-dark)
![Expressiveness vs performance: CUDA's wide range includes many slow patterns; Croqtile's restricted range maps entirely to the performance sweet spot](../assets/images/ch07/fig8_expressiveness_dark.png#only-light)

TMA's descriptor-based interface is an extreme instance of this philosophy: it supports only a few tile-aligned transfer patterns, but those patterns are exactly the ones the hardware is optimized for. Croqtile's `dma.copy` follows the same principle — it generates only patterns that are coalesced and conflict-free, even though the underlying `LDG`/`STS` instructions can express far more (including many slow patterns). The trade is explicit: you give up the ability to write arbitrary data movement, and in return, every data movement you write is fast.

## TMA in a pipelined matmul

The pipeline skeleton from Chapter 6 is unchanged: ring of stages, `wait` / `trigger` on events, MMA commit, consumer drains tiles while producer fills the next slot. Here the producer swaps `dma.copy` for `tma.copy.swiz<3>` and the consumer swaps `mma.load` for `mma.load.swiz<3>`:

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

Compared to the Chapter 6 `dma.copy` version, only the ingress and operand load lines change; events, staging indices, and commit stay the same. The writeback to global memory still uses `dma.copy` — choose TMA or DMA for stores according to your target.

## Handling irregular access

Uniform tiling with `chunkat` and `subspan(...).at(...)` covers many kernels. Real workloads also need windows at arbitrary offsets, strides between tiles, partial tiles at boundaries, and layout reinterpretation. The subsections below collect those tools.

### Arbitrary-offset windows: `view` and `from`

`view(M, N).from(row, col)` defines an `M x N` rectangle starting at `(row, col)` — no requirement that the origin aligns to a precomputed tile grid.

```choreo
patch = matrix.view(16, 16).from(37, 50);
```

This is a `[16, 16]` slice starting at row 37, column 50. Alignment is not required.

![chunkat (aligned grid) vs view/from (arbitrary offset window)](../assets/images/ch07/fig4_view_from_dark.png#only-dark)
![chunkat (aligned grid) vs view/from (arbitrary offset window)](../assets/images/ch07/fig4_view_from_light.png#only-light)

**When to use it.** `chunkat` needs the tensor divided evenly; `view(...).from(...)` does not. Prefer `chunkat` for regular tiling and `view` / `from` when the window is ragged or runtime-positioned.

```choreo
expert_lhs = lhs.view(expert_M, K).from(expert_offset, 0);
dma.copy expert_lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => shared;
```

In mixture-of-experts stacks, each expert's token batch starts at a dynamic row `expert_offset`. Slicing with `view` / `from` rewires the operand before the rest of the pipeline — DMA, MMA, events — continues unchanged.

### Strided tiles: `.subspan`, `.step`, and `.at`

`subspan(M, K).at(i, j)` selects the tile at logical tile indices `(i, j)` with extent `[M, K]`. Adding `.step(sM, sK)` spaces tiles `sM` rows and `sK` columns apart instead of packing them contiguously:

```choreo
matrix.subspan(16, 16).step(32, 32).at(i, j);
```

![Packed tiling vs strided tiling with .step](../assets/images/ch07/fig5_subspan_step_dark.png#only-dark)
![Packed tiling vs strided tiling with .step](../assets/images/ch07/fig5_subspan_step_light.png#only-light)

Tile `(0,0)` starts at `(0,0)`, but tile `(1,0)` starts at `(32,0)` and `(0,1)` at `(0,32)`. Omitting `.step` uses a step equal to the tile size — the packed case.

**Typical uses:** skipping padding or guard bands, overlapping stencils where the step is smaller than the extent, or matching an outer layout that is not dense tile-major.

### Zero-padding: `.zfill`

When `M` or `K` is not a multiple of the tile size, the last tile along an axis is partial. Reading past the tensor's edge is undefined unless you explicitly pad.

```choreo
tma.copy.swiz<3> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k).zfill
  => lhs_load_s;
```

`.zfill` applies to the source side of a copy: out-of-range elements are written as zero in the destination tile. Zeros contribute nothing to a GEMM accumulation, so the MMA loop stays uniform while remaining mathematically correct for partial edges.

![.zfill: zero-padding partial tiles at the tensor boundary](../assets/images/ch07/fig6_zfill_dark.png#only-dark)
![.zfill: zero-padding partial tiles at the tensor boundary](../assets/images/ch07/fig6_zfill_light.png#only-light)

### Layout reinterpretation: `span_as`

`span_as` reinterprets a buffer's linear storage as another shape with the same element count — no copy.

```choreo
flat_buffer.span_as([rows, cols])
```

Element count is preserved; only the logical rank changes.

```choreo
strip_load = dma.copy data.chunkat(tile) => shared;
tile_2d = strip_load.data.span_as([tile_m, tile_k]);
ma = mma.load tile_2d.chunkat(_, iv_warp);
```

This exposes a loaded 1D strip as a matrix for `chunkat` without an extra copy. `rows * cols` must equal the span length of the underlying storage, or the compiler rejects the program.

![span_as: zero-copy shape reinterpretation from 1D to 2D](../assets/images/ch07/fig7_span_as_dark.png#only-dark)
![span_as: zero-copy shape reinterpretation from 1D to 2D](../assets/images/ch07/fig7_span_as_light.png#only-light)

## Chapter summary

| Concept | Syntax | Role |
|---------|--------|------|
| Software DMA (Ch. 2, 6) | `dma.copy` / `dma.copy.swiz<N>` | Thread-cooperative tile transfer; works on all CUDA GPUs |
| Hardware TMA | `tma.copy` / `tma.copy.swiz<N>` | Descriptor-driven Hopper ingress; dedicated engine enables async overlap |
| Swizzle | `.swiz<N>` on copy + `mma.load.swiz<N>` | Bank-conflict-free SMEM layout; same effect for DMA and TMA |
| Expressiveness trade | — | Croqtile restricts patterns to guarantee coalesced, conflict-free transfers |
| Arbitrary windows | `view(M,N).from(r,c)` | Ragged or runtime-positioned slices |
| Strided tiling | `.subspan().step().at()` | Non-packed layouts, overlapping stencils |
| Partial tiles | `.zfill` | Zero-fill out-of-bounds elements |
| Shape reinterpretation | `span_as([dims])` | Zero-copy reshape for staging buffers |

## New syntax

| Syntax | Meaning |
|--------|---------|
| `tma.copy src => dst` | TMA hardware tensor copy (Hopper SM90+) |
| `tma.copy.swiz<N> src => dst` | TMA copy with swizzle mode `N` (0–3) |
| `dma.copy.swiz<N> src => dst` | DMA copy with swizzle mode `N` (0–3); same layout as TMA |
| `mma.load.swiz<N> src` | MMA operand load consistent with swizzle `N` |
| `tensor.view(M, N).from(r, c)` | Arbitrary-offset `M x N` window |
| `.subspan(M, K).step(sM, sK).at(i, j)` | Strided tile selection |
| `.zfill` | Zero-fill out-of-bounds elements on copy source |
| `span_as([dims])` | Reinterpret linear storage as shaped tensor |

The [next chapter](ch08-cpp-interop.md) steps past pure Croqtile into **C++ interop**: `__device__` functions, **register hints**, **preprocessor guards**, and **inline PTX** when you need to drop down to the metal.
