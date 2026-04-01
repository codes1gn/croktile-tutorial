# Beyond chunkat: view and from for Irregular Access

So far in this tutorial, tiling has been **regular**: you carved tensors with **`chunkat`**, **`subspan`**, and **`.at(block_m, block_n)`** so that each parallel unit owned a predictable, evenly spaced tile. The **block index** and **chunk index** told you *which* rectangle you were touching, and the **upper bounds** of those indices implied a **fixed grid** over the whole matrix.

That model is exactly what you want for a dense GEMM. It is not always what you want for **grouped** or **ragged** work. In a **Mixture of Experts (MoE)** layer, each expert may process a **different number of tokens**. The tokens for all experts still live in one big buffer (for example, all rows of **`lhs`**), but the **starting row** for expert *e* is not `e * fixed_tile_height`. It is whatever the router assigned — and you learn that only at **runtime**, typically from an **offset array**.

This chapter introduces **`view`** and **`from`**: a way to say “give me an **`M × N`** window into this tensor, anchored at **absolute** row and column offsets.” Together with **`dma.copy` variants that zero-fill** out-of-bounds regions, **`mma.scale`** for block-wise dequantization, and a few launch details (**`parallel.async`**, **`stream`**), you will read a real **FP8 MoE GEMM** Croktile function from Croktile’s benchmark tree and connect it back to the **regular** tiling you already know.

Along the way you will see why **`chunkat` / `subspan().at()`** assume a **regular** decomposition and when that breaks; how **`.zfill`** on a swizzled DMA handles **partial final tiles** without reading past valid data; how **`mma.scale`** applies **per-block scale factors** after FP8 tensor-core accumulation; what **`parallel.async`** and a **`stream`** parameter imply for **asynchronous** kernel launch; how **`expert_offsets`** drives **variable-length segments** along the token (row) dimension; and where **`__cpp__`** fits as an escape hatch (full treatment in Chapter 10).

**Prerequisites:** Chapters 2–5 (**DMA**, **tiling**, **TMA/swizzle**, **MMA**) and **Chapter 8** (**multi-warpgroup** context) give the vocabulary this chapter reuses. You do **not** need MoE training beyond “experts can get **different** token counts.”

## The limitation of even tiling

Picture the left-hand side of a batched matmul as **`lhs`**, shaped **`[M, K]`** in logical rows and columns. In Chapters 4–8, a thread block often selected its rows with something morally equivalent to **`block_m * TILE_M`**: the map from **block id → row range** was a **closed form**.

MoE grouping breaks that assumption. After routing, you might store tokens in **`lhs`** in **expert-major** order, but the **count** of tokens per expert varies. A compact representation is a **prefix-sum style** offset table **`expert_offsets`**, length **`E + 1`**, where expert **`e`** owns token rows **`[expert_offsets[e], expert_offsets[e+1])`**. The **length** of that half-open interval is **`seg_length`**, and it is **not** generally divisible by your favorite **`MATMUL_WARP_M`**.

You still want to reuse the **same micro-kernel** — fixed **`MATMUL_WARP_M`**, **`MATMUL_WARP_N`**, **`MATMUL_TILE_K`** — because hardware likes predictable tile sizes. So you iterate **`iv_m`** over **`cdiv(seg_length, MATMUL_WARP_M)`** and, on the **last** iteration, you might only have **`TILE_M < MATMUL_WARP_M`** valid rows. The **DMA** still moves a full **`MATMUL_WARP_M × MATMUL_TILE_K`** logical tile into **`sA`**, but part of that tile is **padding**. That is where **zero-fill** semantics come in.

None of this is visible if you only ever write **`lhs.chunkat(block_m, iv_k)`**. **`chunkat`** ties **tile shape** to **how many chunks** the dimension is divided into. Here, the **origin** along **`M`** is **`seg_start + iv_m * MATMUL_WARP_M`**, which is **data-dependent** and **not** a simple chunk index into a uniform grid.

The **performance** story is the usual one: you **amortize** instruction and **shared-memory** setup across **many** **K** panels and **many** output **N** blocks, while the **ragged** dimension costs you only **partial** work on the **last** **`iv_m`** tile per expert. The **irregularity** is **localized** to **how many** **`iv_m`** iterations run and **how** **zfill** behaves on the **tail**, not to a **different** inner **MMA** recipe per expert. That localization is what makes MoE GEMM a good first ragged example: you change addressing, not math.

## `view` and `from`: shape first, then origin

The pattern you need is:

```choreo
tensor.view(M, N).from(row, col)
```

- **`view(M, N)`** — “I want a **logical** **`M × N`** fragment of this tensor’s element space.” It does **not** by itself say *where* that fragment lives.
- **`from(row, col)`** — “Anchor that **`M × N`** window so its **top-left** element aligns with **`(row, col)`** in the tensor’s indexing coordinates.”

Think of **`view`** as declaring the **footprint** of the operation (what size the copy or MMA store expects) and **`from`** as the **base offset** into the backing storage. This is **absolute** indexing in **element** coordinates along each dimension, analogous to slicing a dense matrix at **`[row : row+M, col : col+N]`** in NumPy terms — except Croktile keeps the spelling explicit for code generation and for hardware paths that care about **swizzle** and **layout**.

Contrast with **`subspan(H, W).at(i, j)`**, which also picks a rectangle, but is usually read as “take a **subspan** of the **parent’s** shape, then place it at **tile indices** **`(i, j)`** in a **tiling plan**.” **`chunkat`** goes further: it **derives** chunk shape from **how many** parallel chunks you declared. **`view` / `from`** decouple **window size** from **how the global tensor was partitioned for parallelism**; **`from`** can be **any expression** the compiler can see, including **`seg_start + iv_m * MATMUL_WARP_M`**.

You can read **`tensor.view(M, N).from(r, c)`** as producing a **value** that **participates** in whatever operation needs a shaped fragment: **DMA**, **TMA**, **MMA store**, or a **load** expression, depending on context. The **compiler** still has to prove (or assume) that **`r, c`** and **`M, N`** stay within the **logical tensor bounds** when you are **not** using **zfill**; with **zfill**, the contract shifts: **in-bounds** elements come from memory, **out-of-bounds** lanes in the **source window** become **zero** in the destination fragment the MMA consumes.

Keep **swizzle** in mind from Chapter 4: **`view`** does not turn off layout rules. When you pair **`view`** with **`dma.copy.swiz<128>`**, the **`128`** still describes how the destination **`sA`** is **organized** for later **`mma.load.swiz<128>`**. **`view` / `from`** only relocates **which global elements** feed that pipeline.

If you are **porting** from **pointer**-style CUDA, think of **`from(r, c)`** as the **base** in **element** units and **`view(M, N)`** as the **extent** the **hardware** **transaction** **names** to the **compiler**. Croktile keeps both **explicit** so **later** passes can **prove** **overlap**, **alias**, and **alignment** **facts** that **raw** **arithmetic** often **hides**.

## `expert_offsets` and the `TILE_M` expression

The kernel begins each expert’s work by **loading** two adjacent entries from **`expert_offsets`**:

```choreo
s32 seg_start = expert_offsets.at(eid);
s32 seg_end = expert_offsets.at(eid + 1);
s32 seg_length = seg_end - seg_start;
```

**`expert_offsets`** is a **`global s32 [EXPERTS1]`** vector — the benchmark name **`EXPERTS1`** is **`E + 1`** slots so the last expert can use **`eid + 1`** without overrunning. **`at(eid)`** is ordinary **indexing** into that vector; it is **not** a tile idiom like **`chunkat`**. The result is a plain **`s32`** you can **add** to **`iv_m * MATMUL_WARP_M`** inside **`from`**.

Inside the **`foreach {iv_m}`** loop, **`TILE_M`** is the **minimum** of the **full** warp height and whatever **rows remain** in the segment:

```choreo
TILE_M = (MATMUL_WARP_M < (seg_length - iv_m * MATMUL_WARP_M) )? MATMUL_WARP_M : seg_length - iv_m * MATMUL_WARP_M;
```

That is a **symbolic** way to spell “this tile’s **valid** row count.” The **DMA** destination uses **`sA.subspan(TILE_M, MATMUL_TILE_K)`** so the **typed** destination shape the rest of the pipeline sees matches **valid** data height, while the **source** **`view`** stays **`MATMUL_WARP_M × MATMUL_TILE_K`** so the **hardware copy** stays **uniform** across iterations. **zfill** reconciles the mismatch when **`TILE_M < MATMUL_WARP_M`**.

When you reason about **correctness**, treat **`TILE_M`** as the **only** row count downstream MMA and stores should **trust** as **non-padding**. The **physical** **`sA`** allocation is still **`MATMUL_WARP_M`** rows deep; **zfill** is what makes the **unused** tail rows **safe** to **accumulate** as zeros. Host code that builds **`expert_offsets`** should keep segments **contiguous** and **within** **`[0, M)`** so **`seg_start + seg_length ≤ M`** for valid experts.

On CPU, **assert** **`expert_offsets[0] == 0`**, **`expert_offsets[e] ≤ expert_offsets[e+1]`**, and **`expert_offsets[E] ≤ M`** (or equals **`M`** if **`lhs`** rows are **exactly** the **routed** tokens). **Drift** between **`M`** and the **final** offset **shows** up as **out-of-bounds** **global** access or as **silent** **reads** of **uninitialized** **padding** if the **buffer** is **oversized**.

## Reference kernel: FP8 MoE GEMM (Croktile function excerpt)

The following is the **`__co__` core** of **`moe_gemm_kernel_bf16`** from **`croktile/benchmark/performance/moe_gemm/moe_gemm_v1_fp8_bf16.co`**, slightly trimmed of comments for readability. Read it **top to bottom**: **launch** nesting first, then **segment** setup, then the **`iv_m`** loop over **ragged** **M**, the **`iv_k`** **K**-panel loop, and finally **scale** and **store** tied to the **same** **origins** as **load**.

Types **`f8_e4m3`**, **`bf16`**, and **`f32`** are the FP8 operand, BF16 output, and FP32 accumulator/scale types used in this benchmark. **`f8_e4m3`** is an 8-bit floating type (4 exponent bits, 3 mantissa bits) used for **narrow** operands; **`bf16`** is the **output** element type in this kernel. **`f32`** shows up for **accumulators** and **scales**. The surrounding C++ in the same file wraps pointers with **`crok::make_spanview`** and passes a **`cudaStream_t`** through as **`stream s`** on the **`__co__`** entry point — the DSL names the stream so launches participate in the right **async** queue.

```choreo
__co__ void moe_gemm_kernel_bf16(global f8_e4m3 [M, K] lhs,
                   global f32 [M, DIV_BLK_K] scale_a,
                   global f8_e4m3 [EXPERT_N, K] rhs,
                   global f32 [EXPERT_DIV_BLK_N, DIV_BLK_K] scale_b,
                   global s32 [EXPERTS1] expert_offsets,
                   global bf16 [M, N] output, stream s) {

  parallel.async {eid, block_n} by [EXPERTS, cdiv(N, MATMUL_WARP_N)] : block
  parallel by 1 : group-4
  parallel t by 128 : thread {
    shared f8_e4m3 [MATMUL_WARP_M, MATMUL_TILE_K] sA;
    shared f8_e4m3 [MATMUL_WARP_N, MATMUL_TILE_K] sB;

    s32 seg_start = expert_offsets.at(eid);
    s32 seg_end = expert_offsets.at(eid + 1);
    s32 seg_length = seg_end - seg_start;
    __cpp__("  if (seg_end - seg_start <= 0) return;\n\n");

    foreach {iv_m} in [cdiv(seg_length, MATMUL_WARP_M)] {
      TILE_M = (MATMUL_WARP_M < (seg_length - iv_m * MATMUL_WARP_M) )? MATMUL_WARP_M : seg_length - iv_m * MATMUL_WARP_M;
      mc = mma.fill.f32 0.0f;
      foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
        dma.copy.swiz<128>.zfill
          lhs.view(MATMUL_WARP_M, MATMUL_TILE_K).from(seg_start + iv_m * MATMUL_WARP_M, iv_k * MATMUL_TILE_K)
            => sA.subspan(TILE_M, MATMUL_TILE_K);
        tma.copy.swiz<128>
          rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(eid # block_n, iv_k) => sB;

        foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
          ma = mma.load.swiz<128> sA.chunkat(_, iv_warp);
          mb = mma.load.swiz<128> sB.chunkat(_, iv_warp);
          mma.row.row mc, ma, mb;
        }

        sc_a = scale_a.view(MATMUL_WARP_M, 1).from(seg_start + iv_m * MATMUL_WARP_M, iv_k);
        sc_b = scale_b.at(eid # block_n, iv_k);
        mma.scale mc, sc_a, sc_b;
      }
      mma.store mc, output.view(TILE_M, MATMUL_WARP_N).from(seg_start + iv_m * MATMUL_WARP_M, block_n * MATMUL_WARP_N);
    }
  }
}
```

The listing is **dense** on purpose: every **annotation** (**`swiz<128>`**, **`.zfill`**, **`view` / `from`**, **`#`**) is doing **real** work for **correctness** or **performance**. If you **strip** modifiers while **debugging**, change **one** axis at a time — for example, prove **ragged** **M** with **FP8** **disabled** before you chase **swizzle** bugs.

The outer **`parallel.async {eid, block_n} by [EXPERTS, cdiv(N, MATMUL_WARP_N)] : block`** launches one CUDA block per **(expert, N-tile)** pair. Expert **`eid`** determines **which token segment** of **`lhs`** and **`scale_a`** you care about; **`block_n`** steps along **N** in **`MATMUL_WARP_N`**-wide strips, the same way your dense matmul tiled **N**. Notice the **two-dimensional** launch maps cleanly onto **MoE × output columns**: every expert that receives **at least one token** still has **full** **`N`** coverage because **`block_n`** spans **`cdiv(N, MATMUL_WARP_N)`** independently of **`seg_length`**. An expert with **zero** tokens should **return immediately** so those blocks do no useless global traffic; the benchmark uses **`__cpp__`** for that guard.

**`parallel.async`** opts into **asynchronous** grid behavior relative to the **synchronous** **`parallel`** grids you saw in earlier chapters: the DSL does **not** insert the same **implicit** synchronization you might rely on when every block of a plain **`parallel`** launch is assumed to **serialize** with the next **phase** of host or device work in a particular way. In practice you use **`parallel.async`** when the **runtime** or **compiler** strategy should treat this kernel as part of a **larger** async graph — for example, **back-to-back** launches on the same **CUDA stream**, **overlapped** **H2D** copies, or **producer/consumer** relationships with other kernels.

The **`stream s`** parameter on **`moe_gemm_kernel_bf16`** threads that idea into the **entry point**: the host passes a **`cudaStream_t`**, Croktile names it **`s`**, and generated launch code can associate **this** kernel with **that** queue. You still write **one** `__co__` body; the **stream** is part of the **ABI** between C++ driver code and the **`__co__`** function, like **`global`** pointers for tensors.

Do not confuse **`stream s`** with **GPU-wide** **async** DMA inside the kernel body. Here **`stream`** affects **how the host schedules** the kernel relative to other work; **`parallel.async`** affects **how the Croktile launch** is classified. Together they are the **plumbing** that lets an MoE GEMM sit beside **attention**, **norm**, or **all-to-all** without **accidentally** serializing everything on the **default** stream.

Inside the block, **`seg_start`**, **`seg_end`**, and **`seg_length`** come straight from **`expert_offsets`**. Those are ordinary **`s32`** values in the Croktile function; they exist so the generated kernel can **skip** or **special-case** empty experts. The snippet uses **`__cpp__`** to inject a raw C++ **early return** when the segment is empty — a blunt but effective hook when control flow is easier to express in C++ than in the DSL surface. Chapter 10 goes deeper on **`__cpp__`** and macro patterns; here, treat it as “escape to the metal for one line of guard code.”

```choreo
__cpp__("  if (seg_end - seg_start <= 0) return;\n\n");
```

The string is **C++** text **spliced** into the generated function. You are responsible for **braces**, **semicolons**, and **indentation** that match the surrounding emission. It is easy to misuse; prefer Croktile control flow when you can, and reserve **`__cpp__`** for **small**, **auditable** guards like this one until you have read Chapter 10.

The middle of the launch stack matches Chapters 4–8:

```choreo
parallel by 1 : group-4
parallel t by 128 : thread {
```

That is the same **warpgroup** pattern introduced with **Hopper**-style **WGMMA** in Chapters 4–8: **one** **group-4** warpgroup and **128** threads in the **inner** **parallel**. This chapter does not change **how** tensor cores are **addressed**; it changes **where** **global** data is **anchored** along **M** for the **LHS** and **scales**. If you are **debugging** an MoE port, **first** verify **offsets** and **`TILE_M`**, **then** suspect **MMA** layout — beginners often flip that order.

The crucial **LHS** DMA is:

```choreo
dma.copy.swiz<128>.zfill
  lhs.view(MATMUL_WARP_M, MATMUL_TILE_K).from(seg_start + iv_m * MATMUL_WARP_M, iv_k * MATMUL_TILE_K)
    => sA.subspan(TILE_M, MATMUL_TILE_K);
```

- **`lhs.view(MATMUL_WARP_M, MATMUL_TILE_K)`** — logical source tile size matches the **full** warp tile height and **K-panel** width (here **64 × 128** in the benchmark’s defines).
- **`.from(seg_start + iv_m * MATMUL_WARP_M, iv_k * MATMUL_TILE_K)`** — rows start at the **expert’s** base row plus **`iv_m`** times the fixed vertical stride; columns march along **K** in **`MATMUL_TILE_K`** steps, exactly like a dense GEMM’s **`iv_k`** loop.
- **`=> sA.subspan(TILE_M, MATMUL_TILE_K)`** — shared memory receives only **`TILE_M`** rows of **meaningful** height for this iteration. When **`TILE_M < MATMUL_WARP_M`**, the tail tile is **short**.

**`.zfill`** on **`dma.copy.swiz<128>`** tells the copy path to **zero** any **out-of-bounds** source positions that the **`MATMUL_WARP_M × MATMUL_TILE_K`** window would otherwise try to read — for example, rows beyond **`seg_length`** when the last **`iv_m`** round only has **`TILE_M`** valid rows. Zeros in **`sA`** correspond to **zero contributions** in the matmul, which is what you want for padding. Without **zfill**, you would either need a narrower DMA for the tail (complicating the inner MMA loop) or illegal global reads.

Why not shrink the **`view`** height to **`TILE_M`** on the last tile? You **could** imagine a **variable-shaped** copy into **`sA`**, but then the **inner** **`mma.load`** loop is harder to keep **uniform**: tensor-core **microkernels** want **fixed** fragment dimensions across **all** **`iv_m`** iterations. The **zfill** pattern keeps the **inner** MMA structure **identical** for every **`iv_m`**; only the **destination subspan** and the **final store** **`view(TILE_M, …)`** reflect the **ragged** height.

The **RHS** side stays **regular** in **M** for this kernel: **`rhs`** is indexed with **`subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(eid # block_n, iv_k)`**, combining expert id and **N** block into the row dimension (the benchmark’s **`EXPERT_N`** layout). The **`#`** operator (Chapter 2 / docs on **ubound**) still appears: **`eid # block_n`** packs **bounded** launch indices into the row coordinate expected by **`rhs`**’s storage layout. TMA + swizzle match Chapter 4’s story; only the **LHS** needed **`view` / `from`** because of **ragged** token ranges along **M**.

You might wonder why the **LHS** uses **`dma.copy.swiz<128>.zfill`** while the **RHS** uses **`tma.copy.swiz<128>`**. The benchmark is making a **practical** choice: **TMA** is an excellent default for **dense**, **descriptor-friendly** tiles, but **ragged** **M** with **zfill** is expressed here through the **DMA** path with **swizzle** and **zero-fill** modifiers. The **important** invariant is still that **shared memory** matches what **`mma.load.swiz<128>`** expects — the **same** **128-byte** swizzle story as Chapter 4.

When you port patterns to your own kernels, treat this as **evidence** that Croktile lets you **mix** **movement** primitives inside **one** loop nest: **not** every tensor **leg** must use the **same** instruction family, as long as **layouts** and **barriers** agree. If your target **only** exposes **zfill** on **DMA**, or your **TMA** descriptor setup cannot **cheaply** express **dynamic** **row** bases per expert, the **LHS** side may stay on **DMA** even after you **upgrade** other legs to **TMA**.

After the loads, the **K** dimension is still traversed in **`MATMUL_WARP_K`** chunks with **`chunkat(_, iv_warp)`** on **`sA`** and **`sB`**, and **`mma.row.row`** accumulates into **`mc`** as in Chapter 5. Nothing conceptually new there — shared buffers are still **regular** once filled.

FP8 matmul is not done when the tensor cores finish. **`scale_a`** and **`scale_b`** hold **per-block dequantization** factors (the benchmark uses **`DIV_BLK_K`** and expert-scoped **`DIV_BLK_N`** dimensions on the host side). The **scales** are **blocked** to match how **FP8** **tiles** map to **accumulator** lanes; wiring the wrong **stride** in **`from`** or **`.at`** shows up as **systematic** **numeric** drift, not as **out-of-bounds** faults. For each **`(iv_m, iv_k)`** panel:

```choreo
sc_a = scale_a.view(MATMUL_WARP_M, 1).from(seg_start + iv_m * MATMUL_WARP_M, iv_k);
sc_b = scale_b.at(eid # block_n, iv_k);
mma.scale mc, sc_a, sc_b;
```

**`view(MATMUL_WARP_M, 1)`** takes a **column-shaped** scale strip aligned with the same **row origin** as the **`lhs`** tile; **`.from(..., iv_k)`** picks the **K-block** column of scales. **`sc_b`** uses the familiar **`.at`** on the RHS scale tensor. **`mma.scale`** then applies those factors to the **FP32 accumulator** **`mc`** so the FP8 dot products land in the right numeric range before the final store.

The store mirrors the load:

```choreo
mma.store mc, output.view(TILE_M, MATMUL_WARP_N).from(seg_start + iv_m * MATMUL_WARP_M, block_n * MATMUL_WARP_N);
```

**`view(TILE_M, MATMUL_WARP_N)`** matches the **actual** accumulated tile height for this **`iv_m`** iteration; **`from`** places it in the **global output** at the expert’s rows and the block’s **N** offset. No **`zfill`** is needed on a store: you only write **`TILE_M`** valid rows.

The **movement → MMA → store** ordering in the excerpt follows the **same** **pattern** as dense **GEMM** in earlier chapters: **loads** into **`sA`** / **`sB`** **precede** **`mma.load`**, **accumulation** spans the **full** **`iv_k`** loop, and **`mma.store`** runs **once** per **`(iv_m, block_n)`** after **all** **K** **panels** for that **output** tile **finish**. If you **hoist** **stores** or **fuse** **loops**, **re-check** that **every** **store** still **sees** the **fully** **reduced** **accumulator** for its **logical** **rows**.

## Choosing between `chunkat` / `subspan().at()` and `view().from()`

Use the table as a **decision aid**, not a **dogma**. **`chunkat`** remains the **default** for **dense** **GEMM**-shaped loops where **parallel** indices **line up** with **tensor** storage. Reach for **`view` / `from`** when **storage** is **flat** but **semantics** are **segmented**, or when **padding** is **cheaper** than **forking** the **inner** **MMA** for a **tail** shape.

| Aspect | **`chunkat` / `subspan().at()`** | **`view(M,N).from(r,c)`** |
|--------|----------------------------------|---------------------------|
| **Typical use** | Dense grids: divide **`M`**, **`N`**, **`K`** into **equal** tiles indexed by block or loop counters | **Windows** at **arbitrary** offsets: ragged segments, dynamic slicing, padding-friendly fixed DMA size |
| **How tile size is chosen** | From **upper bounds** of parallel indices (**`#block_m`**, etc.) or explicit **`subspan`** | **Explicit** **`M`, `N`** in **`view`** |
| **How position is chosen** | **Tile indices** (**`block_m`**, **`iv_k`**, …) imply a **regular** map | **`from(r,c)`** can use **runtime** values (**`seg_start + …`**) |
| **Interaction with partial tiles** | Often handled with **masking** or **separate** tail kernels | Combine **`view`** (full hardware tile) with **`.zfill`** and **`subspan`** on destination (valid rows only) |
| **Best when indices are…** | **Affine** in block ids (**`block_m * TILE`**) | **Indirect** or **prefix-sum** style (**`offsets[e]`**) |
| **Typical paired ops** | **`tma.copy`**, **`dma.copy`**, **`chunkat`** on both sides | **`dma.copy…zfill`**, **`mma.store`** with **`view(TILE_M, …)`** |

Neither table row is “better”; they answer different questions. Most production kernels use **both**: **regular** tiling where the data is regular, **`view` / `from`** where the abstraction is **logical rows** in a flat buffer and the hardware still wants **fixed-size** tiles.

When you reach for **`view` / `from`**, you are usually solving one of these: an **indirect** or **variable** base along one dimension with **fixed** micro-tile size on the hardware; **padding** a tail tile to full width or height without second inner kernels; or alignment of **scale** or **bias** tensors with the same row/column origin as a data tile (**`scale_a`** tracks **`lhs`** rows in the MoE kernel). If every dimension is evenly divisible and **block id → offset** is a simple formula, **`chunkat`** and **`subspan().at()`** stay shorter and clearer. When offsets live in global memory as data, **`view` / `from`** is usually the faithful spelling.

Refactors between the two styles are mechanical once you write down three invariants: tile shape, origin formula, and partial-tile policy (**mask**, tail kernel, or **zfill**). If you cannot state those three lines on a whiteboard, the bug you are hunting is usually not the choice of **`chunkat`** versus **`view`**.

## Closing

**`chunkat`** and **`subspan().at()`** shine when your parallelism is a **Cartesian product** of **even** divisions. MoE-style **grouped GEMM** breaks that along the **token** axis: experts own **variable-length** row ranges, and those ranges start at **runtime** positions from **`expert_offsets`**.

**`.view(M, N).from(r, c)`** separates **tile shape** from **tile origin**. Pair it with **`dma.copy...zfill`** when the last tile is **shorter** than the hardware tile but you still want to **fill** shared memory with a **well-shaped** fragment for tensor cores. **`mma.scale`** applies **block scales** after FP8 accumulation. **`parallel.async`** and **`stream`** tie the kernel into the broader **async** CUDA model. In one sentence: **regular** tilings index **which** tile; **`view` / `from`** index **where** that tile’s **top-left** lives in **global** memory, which is what **ragged** **batched** problems require.

**Empty segments** — If **`seg_length == 0`**, any load from **`lhs`** at **`seg_start`** is invalid. The **`__cpp__`** guard exists for that reason; if you remove it, you need an equivalent DSL-side early exit. **Non-monotonic offsets** — **`expert_offsets`** should be non-decreasing; if host code violates that, **`seg_length`** can go negative unless you clamp or validate on the CPU. **Out-of-range `eid`** — The launch grid assumes **`eid ∈ [0, EXPERTS)`**; host metadata and **`make_spanview`** lengths must stay consistent with **`parallel ... by [EXPERTS, ...]`**.

**Forgetting zfill** — If you use a **`view`** of full height but drop **zfill**, the tail **`iv_m`** iteration can read past the logical end of the expert’s tokens unless **`seg_start + iv_m * MATMUL_WARP_M + MATMUL_WARP_M ≤ M`** always holds by construction. **Scale alignment** — **`scale_a.view(…).from(…)`** must use the same row origin as the **`lhs`** tile for that **`iv_m`**. An off-by-one-block error in **K** (**`iv_k` vs `iv_k * DIV_BLK_K`**) is a common integration bug when wiring host blocking constants.

If you are learning by editing, log **`seg_start`**, **`seg_length`**, and **`TILE_M`** from the host for a small **E** and compare to what you expect from routing. Temporarily replace **`view` / `from`** on **`lhs`** with a dense **`subspan().at(block_m, iv_k)`** kernel on uniform segments to isolate ragged behavior from FP8 scale behavior. Read the **`moe_gemm_v1_fp8_bf16.co`** host path to see how **`EXPERT_N`**, **`DIV_BLK_K`**, and **`DIV_BLK_N`** are derived from **`m`**, **`n`**, **`k`**, and **`block_size_*`**. Those steps mirror how engineers bisect performance and correctness issues in real grouped GEMM bring-up. The reference implementation remains **`croktile/benchmark/performance/moe_gemm/moe_gemm_v1_fp8_bf16.co`** if you want to diff your experiments against a known-good baseline.

Chapter 10 zooms in on **`__cpp__`**, macros, and other **escape hatches** for the corners of real kernels that resist pure DSL expression — so you can **graduate** from **one-line** guards to **structured** **interop** without losing Croktile-level clarity.

Until then, treat **`view` / `from`** as the primary tool whenever parallel indices stop predicting memory locations by closed form. The MoE GEMM is representative: grouped batch dimensions, sparse activations, and variable-length sequences all reuse the same recipe with different offset tables.

When you reuse this pattern outside MoE, rename **`expert_offsets`** mentally to whatever table defines segment starts (batch pointers, CSR row pointers, bucket heads, and so on). The Croktile spelling stays **`view` / `from`** as long as one axis is ragged and hardware still wants fixed tiles.
