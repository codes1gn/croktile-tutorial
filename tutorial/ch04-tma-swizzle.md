# High-Performance Data Movement: TMA and Swizzle

In Chapters 2 and 3, you moved data with **DMA** — explicit copies that software orchestrates tile by tile. That model is clear and portable, but on modern NVIDIA GPUs it leaves performance on the table. Starting with the **Hopper** architecture (compute capability 9.0, or **SM90**), NVIDIA added a dedicated hardware path for bulk, multi-dimensional loads and stores: the **Tensor Memory Accelerator**, or **TMA**.

This chapter introduces TMA in Choreo, together with **swizzle** — a layout trick that keeps shared-memory accesses free of **bank conflicts** when tensor cores and wide loads expect a particular byte pattern. The running example is a real **FP16 matrix multiply** tileflow from the Choreo benchmark suite. MMA operations (`mma.load`, `mma.row.row`, and friends) appear here because they sit in the same loop as TMA; Chapter 5 unpacks tensor-core programming in depth.

## From software copies to hardware tensor movement

On pre-Hopper GPUs, getting a tile from global memory into shared memory usually meant many threads each loading a few elements, or a vectorized loop you wrote yourself. The program pays for every instruction that participates in that copy.

**TMA** moves that work into a **hardware unit**. You give it a description of a multi-dimensional tensor in memory (a *tensor map* or descriptor), and it issues the right sequence of memory transactions to copy a **rectangular tile** in one logical operation. Warps can keep doing math — or wait on a lightweight barrier — while TMA works. That is the performance story: **fewer instructions spent on movement**, better overlap, and a path that lines up with what tensor memory and WGMMA expect on Hopper.

Choreo exposes this through **`tma.copy`** and variants like **`tma.copy.swiz<N>`**. The **swizzle** parameter tells the hardware (and the compiler) how to **remap** the destination layout in shared memory so that later column-like or K-dimension accesses do not hammer the same shared memory **bank** repeatedly.

Shared memory is split into **32 banks**. Successive 4-byte words map to successive banks. If a warp reads a row and every lane reads a consecutive element, you get a nice spread across banks. If a warp reads a **column** of a row-major `f16` matrix, many lanes land on the same bank — **bank conflicts** that serialize accesses. **Swizzling** applies a fixed permutation of how addresses map to banks (often described in bytes: 32, 64, or **128**), so that the access pattern your MMA or TMA consumer uses sees fewer conflicts. For FP16 kernels, a common choice is **`swiz<128>`** when the tile width in K matches the recipe (in the benchmark, `MATMUL_TILE_K` is 64 elements of `f16` → 128 bytes per row fragment, and the build asserts `MATMUL_SWIZ == 2 * MATMUL_TILE_K`).

The important idea: **`tma.copy.swiz<N>`** is not just a copy; it is **copy + agreed-upon SMEM layout** so that **`mma.load.swiz<N>`** on the same buffer sees a consistent byte pattern.

In Chapters 2 and 3, **`dma.copy`** expressed the same intent — move a tile between memory kinds — but the generated path was software-driven (threads participating in the copy, possibly pipelined with `inthreads` and events). **`tma.copy`** is the Hopper-shaped spelling: the Choreo compiler maps it to TMA descriptors and instructions for SM90-family targets. You still think in terms of source fragment and destination buffer, but the mechanism is different. When a kernel uses **`mma.load.swiz`** and WGMMA-style **`parallel ... : group-4`**, pairing TMA loads with those intrinsics is the expected layout; swapping in generic DMA for the same SMEM buffers would break the assumptions the MMA side makes about byte-level placement. **DMA** teaches tiling and orchestration; **TMA** is the next gear when you commit to Hopper tensor pipelines.

## Reference kernel: FP16 matmul tileflow (SM90)

The following is the tileflow core of the dynamic FP16 matrix multiply from the Choreo Hopper benchmark (`matmul_f16_dyn_sm90.co`). Compile-time constants:

- `MATMUL_WARP_M = 64`, `MATMUL_WARP_N = 128`, `MATMUL_TILE_K = 64`, `MATMUL_WARP_K = 16`
- `MATMUL_SWIZ = 128`
- `M`, `N`, and `K` are **symbolic** — actual sizes come from the host at runtime.

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
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
}
```

A smaller, test-oriented version lives in `choreo/tests/gpu/end2end/matmul_f16_dynamic.co` with simpler numbers (`WARP_M = 64`, `WARP_N = 64`, `TILE_K = 64`, `WARP_K = 16`, literal `128` for swizzle) and is easier to grep and run under the test harness. The ideas are identical; the benchmark widens the N tile to 128 for throughput.

## What one thread block does

It helps to narrate a single output tile before worrying about the whole grid.

**Grid and dynamic shapes.** Pick indices **`block_m`** and **`block_n`**. This CUDA block owns the **`MATMUL_WARP_M × MATMUL_WARP_N`** region of **`output`** at logical position `(block_m, block_n)` in the block grid — rows roughly `block_m * MATMUL_WARP_M` through `(block_m + 1) * MATMUL_WARP_M - 1`, columns analogously. The function signature says **`global f16 [M, K] lhs`**: **`f16`** is half precision, **`M`** and **`K`** are **symbolic dimensions** whose concrete values come from the host, and **`global`** marks device global memory. Using symbolic sizes means one Choreo function handles any problem shape without recompilation. The return type **`void`** means results are written **in place** through the **`output`** parameter, matching how many GPU kernels accept destination pointers explicitly.

**Ceiling division.** **`cdiv(K, MATMUL_TILE_K)`** is \(\lceil K / \text{TILE\_K} \rceil\) — one more tile when K is not a multiple of the tile size. The grid uses `cdiv(M, MATMUL_WARP_M)` and `cdiv(N, MATMUL_WARP_N)` for the same reason. When the last tile is partial, real kernels pair `cdiv` grid sizing with predicate masks, zero-padding, or epilogue cleanup so out-of-bounds threads do not read or write garbage.

**Shared buffers.** Inside `parallel … : block`, two SMEM allocations stage operand tiles:

```choreo
shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
```

These are explicit shared-memory arenas with fixed compile-time extents. TMA writes into them (`=> lhs_load_s`), and later **`mma.load.swiz`** reads the same SMEM with the same swizzle tag so the layout contract holds. The accumulator path uses a separate SMEM tile for the result before copying to global:

```choreo
shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
mma.store mc, output_s;
tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
```

The end-to-end flow: **TMA in → tensor cores accumulate → `mma.store` to SMEM → TMA out to global.**

**The accumulator.** **`mc = mma.fill.f16 0.0f`** creates an MMA accumulator state in register-resident tiles, zeroed for FP16 accumulation. **`mc`** is not ordinary SMEM; it persists across the K loop and represents the block's running output tile. Chapter 5 says more about its scope and type.

**The K loop.** **`foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)]`** iterates over K-tiles from global memory. For each `iv_k`, two **`tma.copy.swiz<MATMUL_SWIZ>`** lines pull operand slabs from global into SMEM. The LHS load uses **`lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)`** — a view with explicit tile extents, then tile selection by block index and K index. The RHS load uses **`rhs.chunkat(block_n, iv_k)`** — regular tiling along block and K indices. Both are views over the same underlying `global` data; they differ in how explicitly you state the tile shape (more on this choice in the next section).

**The inner MMA loop.** Inside each loaded K-tile, **`iv_warp`** runs from 0 to `cdiv(MATMUL_TILE_K, MATMUL_WARP_K) - 1`. With TILE_K=64 and WARP_K=16, that is **four** MMA slices along K per outer K-tile. For each slice, the warp group loads **`ma`** and **`mb`** from SMEM with **`mma.load.swiz<MATMUL_SWIZ>`** and issues **`mma.row.row mc, ma, mb`** to fold the partial product into the accumulator.

Hopper **WGMMA** expects a **warp group** of four warps (128 threads) to cooperate on one MMA. Choreo annotates that with **`parallel p by 1 : group-4`**. Even though the parallel count is 1 in the `p` dimension, the team executing the body is four warps. Chapter 5 goes deeper; for this chapter, treat **`: group-4`** as "schedule this body as a Hopper warp group."

In **`lhs_load_s.chunkat(_, iv_warp)`**, the underscore **`_`** says "no tiling selector on the first dimension — keep its full extent." The entire `MATMUL_WARP_M` (or `MATMUL_WARP_N`) side is already in SMEM for this block; only K is subdivided for each MMA slice. Writing `chunkat(some_m, iv_warp)` would incorrectly imply an extra partition along M inside SMEM.

Notice two nested notions of K: the **outer** loop `iv_k` stages `MATMUL_TILE_K` elements at a time from global — sized for memory and descriptor efficiency — while the **inner** loop `iv_warp` feeds the MMA in thinner `MATMUL_WARP_K` slivers. The outer loop is about **what fits through TMA and SMEM**; the inner loop is about **what one WGMMA instruction eats per step**.

Chapter 3 taught you to overlap DMA with compute using software queues. On Hopper, the same load → math → store rhythm continues, but TMA and async barriers let hardware participate in that overlap more aggressively than a naive thread loop. Do not worry if the MMA opcodes feel opaque on first read; the data-movement story you need from this chapter is: **TMA filled `lhs_load_s` / `rhs_load_s` with swizzle N; MMA consumed those buffers with the same N; then results left via SMEM and TMA.**

The end-to-end test `matmul_f16_dynamic.co` uses **`mma.fill.f32`** with **`mma.row.row`** and FP16 tensors — a slightly different accumulator for the test harness than the `mma.fill.f16` benchmark variant. For this chapter, treat both as "initialize accumulator, accumulate, store"; Chapter 5 unifies the precision and layout rules.

## Choosing between subspan and chunkat

Both carve global tensors into tiles, but they emphasize different things.

**`subspan(M, K).at(i, j)`** creates a **view** with explicit tile extents, then selects which tile. Use it when you want to spell out the logical tile size — especially when matching a fixed MMA or TMA recipe:

```choreo
lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)
output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n)
```

Read it as: "take a `MATMUL_WARP_M × MATMUL_TILE_K` windowing pattern on `lhs`, then pick the window at `(block_m, iv_k)`."

**`chunkat(i, j)`** assumes a regular decomposition: divide each dimension into equal chunks according to the tile shape implied by the surrounding type and indexing. Use it when the default even tiling is obvious:

```choreo
rhs.chunkat(block_n, iv_k)
```

Prefer **`subspan().at()`** when you are matching hardware recipes from a spreadsheet or an existing CUTLASS/CUTE kernel — the view extents are literal in your source, easy to cross-check against `shared` sizes. Prefer **`chunkat`** when the decomposition is uniform and already clear from context. Both are views over the same data; the compiler still proves access coherence for the target.

## Constraints, pitfalls, and what's next

The `// REQUIRES: TARGET-SM_90` line on the test file is a hint: TMA + WGMMA + `group-4` needs a **Hopper-class** target (typically `sm_90a` in your Choreo invocation). The benchmark source statically asserts relationships between constants — for example that `MATMUL_SWIZ` equals `2 * MATMUL_TILE_K` for this FP16 recipe, and that `MATMUL_SWIZ` is one of 32, 64, or 128. Those checks mirror hardware constraints: swizzle mode, element size, and tile width must agree or the descriptor setup is invalid. If you change `TILE_K` without changing `swiz`, expect either a compile-time error or a runtime failure depending on how far the bad combination gets.

Treat `MATMUL_WARP_K = 16` for `f16` the same way: it is tied to the MMA instruction shape on SM90, not an arbitrary tunable. Copy working triples `(TILE_K, SWIZ, WARP_K)` from known-good kernels before you experiment. Later chapters revisit autotuning; for now, start from known-good constants.

When reading or writing a Hopper-style Choreo kernel, watch for:

- **Mismatched swizzle tags.** If `tma.copy.swiz<128>` feeds a buffer but `mma.load.swiz<64>` reads it, tensor cores interpret the wrong byte layout. Keep the template argument identical per SMEM allocation.
- **`subspan` extents disagreeing with `shared` sizes.** The global view's tile shape and the `shared f16 [...]` declaration must describe the same element counts in each dimension. Off-by-one or transposition shows up as silent wrong answers or illegal TMA shapes.
- **Forgetting that `void` means no return value.** Host code must pass a writable `output` view; you cannot assign the result of `matmul(...)` to a fresh `spanned_data` unless you wrap a helper.
- **Assuming `_` means "any index."** It means omit this chunk axis, not "infer a loop variable." The surrounding `foreach` and `parallel` still determine which threads run.

When you read this kernel, ask five questions: Where does data live (`global` vs `shared` vs MMA accumulators)? How big is each tile (`WARP_M`, `WARP_N`, `TILE_K`, `WARP_K`, and how `cdiv` counts loops)? How do global indices map to blocks? Is the SMEM layout consistent (same `swiz<N>` on `tma.copy.swiz` and `mma.load.swiz`)? Who executes the MMA body (`: group-4` for WGMMA)?

In [Chapter 5](ch05-mma.md), the focus shifts to MMA itself: what the accumulators represent, how layouts connect to PTX, and how to reason about correctness and performance without drowning in hardware manuals.
