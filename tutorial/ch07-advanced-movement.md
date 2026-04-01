# Advanced Data Movement: TMA, Swizzle, and Irregular Access

Every data movement so far has used `dma.copy` — a software-driven transfer where threads construct addresses, issue loads, and shuffle bytes into shared memory. On Hopper (SM90), NVIDIA added a dedicated hardware unit for this job: the **Tensor Memory Accelerator (TMA)**. TMA handles address computation, layout translation, and multi-dimensional tiling in hardware, freeing the producer warpgroup to do other work (or not exist at all).

This chapter replaces `dma.copy` with `tma.copy` and introduces **swizzle modes** that rearrange shared-memory layout to avoid bank conflicts. It also covers the data-access tools for kernels whose tiles are not perfectly regular grids: **`view`/`from`** for arbitrary-offset windows, **`.subspan`/`.step`/`.at`** for strided access, **`.zfill`** for out-of-bounds padding, and **`span_as`** for layout reinterpretation.

## `tma.copy`: Hardware Tensor Movement

The syntax looks almost identical to `dma.copy`:

```choreo
tma.copy lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => lhs_load_s;
```

Same source expression, same `=> destination` form. But the semantics differ:

- **No thread cooperation needed.** DMA requires threads to collectively load a tile (each thread loads its portion). TMA issues a single descriptor-based copy from one thread (or even from the hardware pipeline itself).
- **Multi-dimensional addressing.** TMA understands tensor layouts natively — it translates the `.subspan(...).at(...)` expression into a hardware descriptor without generating per-element address arithmetic.
- **Bulk completion.** TMA commits an entire tile atomically; you synchronize on the whole tile, not individual elements.

The tradeoff is that TMA requires a **tensor descriptor** on the host side, which the compiler generates from the `__co__` function signature. For most kernels, this is invisible — the compiler handles it and you write `tma.copy` where you would have written `dma.copy`.

## Swizzle Modes

Shared memory on NVIDIA GPUs is divided into **banks** — 32 banks, each 4 bytes wide. When multiple threads in a warp access different addresses that map to the same bank, the accesses are serialized: a **bank conflict**. In a dense tile load where thread `t` reads column `t`, the access pattern often creates 2-way or 4-way conflicts that halve or quarter effective bandwidth.

**Swizzle** rearranges the column indices within each row so that threads accessing consecutive columns hit different banks. Croktile's notation:

```choreo
tma.copy.swiz<3> src => dst;
```

The `<3>` parameter controls the swizzle granularity: `swiz<0>` is no swizzle, `swiz<1>` is 64-byte, `swiz<2>` is 128-byte, and `swiz<3>` is 256-byte XOR-based remapping. Larger granularity eliminates conflicts over wider access patterns but requires the tile shape to be a multiple of the granularity.

When you load operands with swizzled layout, you must read them back with the **matching** swizzle mode:

```choreo
ma = mma.load.swiz<3> lhs_load_s.chunkat(_, iv_warp);
```

`mma.load.swiz<3>` tells the MMA operand load that the data in shared memory was laid out with `swiz<3>` XOR pattern. Using `mma.load` (unswizzled) on `swiz<3>` data reads garbage — the addresses don't match.

**Consistent swizzle rule:** the `<N>` parameter on `tma.copy.swiz<N>` must equal the `<N>` on `mma.load.swiz<N>`. The compiler does not enforce this — it is a correctness invariant you maintain.

## TMA in a Pipelined Matmul

Here is a Hopper FP16 matmul using TMA for loads and swizzled shared memory. It builds on the 1P1C pipeline from Chapter 6:

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

Compared to Chapter 6's `dma.copy` version, the producer replaces `dma.copy ... =>` with `tma.copy.swiz<3> ... =>`, and the consumer replaces `mma.load` with `mma.load.swiz<3>`. The pipeline structure — events, ring buffer, commit — is unchanged. TMA + swizzle is a drop-in for DMA that eliminates per-thread address math and bank conflicts.

## `view` and `from`: Arbitrary-Offset Windows

All examples so far tile along aligned boundaries: `chunkat(i)` picks the i-th equal chunk, `subspan(M, K).at(block_m, iv_k)` picks a tile at position `(block_m, iv_k)`. But some kernels need a window that starts at an arbitrary offset, not necessarily a tile-aligned position.

`view(M, N).from(row, col)` creates a rectangular view of size `M × N` starting at position `(row, col)`:

```choreo
patch = matrix.view(16, 16).from(37, 50);
```

This gives you a `[16, 16]` window into `matrix` starting at row 37, column 50 — no alignment requirement. If `(37 + 16)` exceeds the matrix height, you are reading out of bounds unless you use `.zfill`.

The distinction matters: `chunkat` requires the tensor to be evenly divisible by the tile count; `view().from()` does not. Use `chunkat` for uniform tiling, `view().from()` for ragged or offset access.

A real use case: **Mixture-of-Experts (MoE) GEMM**, where each expert processes a different number of tokens. The expert's slice of the input matrix starts at an offset determined at runtime, not at a tile boundary:

```choreo
expert_lhs = lhs.view(expert_M, K).from(expert_offset, 0);
dma.copy expert_lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => shared;
```

Here `expert_offset` is a runtime value — the starting row of expert `e`'s token batch. `view().from()` cuts a dynamically positioned window before the rest of the pipeline (DMA, MMA, events) runs unchanged.

## `.subspan`, `.step`, `.at`: Strided Tile Access

You have already seen `subspan(M, K).at(i, j)` — define a tile extent and select by tile index. Adding `.step` introduces a stride between tiles:

```choreo
matrix.subspan(16, 16).step(32, 32).at(i, j)
```

This selects tiles of size `[16, 16]`, but **separated by 32 rows and 32 columns** instead of being packed contiguously. Tile `(0, 0)` starts at `(0, 0)`, tile `(1, 0)` starts at `(32, 0)`, tile `(0, 1)` starts at `(0, 32)`. The `.step` controls how far apart tiles are in each dimension.

Without `.step`, tiles are packed: the step equals the tile size. `.step` is useful when:

- **You skip over padding or guard bands** between tiles.
- **Tiles have overlap** (step smaller than extent), as in stencil or convolution patterns.
- **Tiles are strided along an outer dimension** for memory layout reasons.

## `.zfill`: Zero-Padding Out-of-Bounds Tiles

When the tensor dimensions are not multiples of the tile size, the last tile along each axis is **partial** — it extends past the tensor boundary. Reading past the end is undefined behavior in CUDA.

`.zfill` tells the DMA or TMA copy to fill out-of-bounds elements with zero instead of reading garbage:

```choreo
tma.copy.swiz<3> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k).zfill
  => lhs_load_s;
```

The `.zfill` modifier goes after the source expression. The hardware (or the generated code for DMA) checks which elements of the requested tile fall outside the tensor and writes zero for those elements, leaving the valid elements unchanged. The MMA loop proceeds as normal — the zeros contribute nothing to the accumulation, which is the correct behavior for partial tiles in GEMM.

## `mma.row.row.scale`: Block Dequantization

Chapter 4 introduced `mma.row.row mc, ma, mb` for FP16 × FP16. When working with **FP8** formats (`f8_e4m3` or `f8_e5m2`), the accumulation is in FP32 but the operands lack dynamic range. Block-scaled quantization stores one scaling factor per tile:

```choreo
mma.row.row.scale mc, ma, mb, sc_a, sc_b;
```

After the matrix multiply, each element of `mc` is multiplied by the corresponding scale factors `sc_a` and `sc_b`. This fuses dequantization into the MMA instruction — no separate pass needed.

The scale factors `sc_a` and `sc_b` are loaded alongside operands A and B, typically from a separate metadata tensor. The compiler handles the broadcast from per-tile scale to per-element result.

## `span_as`: Layout Reinterpretation

Sometimes the same buffer needs to appear with different logical shapes at different points. `span_as` reinterprets the layout without copying:

```choreo
flat_buffer.span_as([rows, cols])
```

This treats the underlying storage of `flat_buffer` as a 2D tensor of shape `[rows, cols]`. The element count must match — `rows * cols == flat_buffer.span(0)` — or the compiler will reject it.

A common pattern: loading a 1D strip from global memory and then reinterpreting it as a 2D tile for MMA:

```choreo
strip_load = dma.copy data.chunkat(tile) => shared;
tile_2d = strip_load.data.span_as([tile_m, tile_k]);
ma = mma.load tile_2d.chunkat(_, iv_warp);
```

## `parallel.async` and `stream s`

For fire-and-forget kernel launches, Croktile provides:

```choreo
parallel.async {px, py} by [grid_m, grid_n] : block {
  stream s;
  // kernel body
}
```

`parallel.async` launches the grid without blocking the host. The `stream s` directive assigns the launch to a CUDA stream; multiple `parallel.async` blocks on different streams run concurrently. This is the Croktile equivalent of launching a kernel on a non-default stream.

## New Syntax

| Syntax | Meaning |
|--------|---------|
| `tma.copy src => dst` | TMA hardware-accelerated tensor copy |
| `tma.copy.swiz<N> src => dst` | TMA copy with swizzle mode (0–3) |
| `mma.load.swiz<N> src` | MMA operand load matching swizzle mode |
| `tensor.view(M, N).from(r, c)` | Arbitrary-offset rectangular window |
| `.subspan(M, K).step(sM, sK).at(i, j)` | Strided tile access with explicit step |
| `.zfill` | Zero-fill out-of-bounds elements on copy |
| `mma.row.row.scale mc, ma, mb, sc_a, sc_b` | MMA with per-tile block dequantization |
| `span_as([dims])` | Reinterpret layout without copying |
| `parallel.async ... : block` | Non-blocking (async) grid launch |
| `stream s` | Assign kernel body to a CUDA stream |

With TMA and swizzle, the memory pipeline runs faster and the code is shorter — no per-thread address arithmetic, no manual bank-conflict avoidance. `view`/`from` and `.zfill` handle the messy edges of real workloads where tiles do not divide evenly and inputs are not aligned.

The [next chapter](ch08-cpp-interop.md) leaves Croktile's high-level abstractions and shows how to drop into raw C++ when you need to — register hints, preprocessor guards, and inline PTX.
