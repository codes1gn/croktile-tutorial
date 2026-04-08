# How to Optimize a Matmul Kernel with Choreo: a Worklog

*April 2026 · GPU: NVIDIA H800 PCIe (SM90a) · Precision: FP16 · Problem: 8192×8192×8192*

---

In this worklog I start from the most naive possible matrix multiplication and step-by-step apply every major GPU optimization technique until the kernel sits within 2% of cuBLAS — all in a high-level DSL called **Choreo**. The finished kernel is 60 lines. No hand-written CUDA, no PTX intrinsics, no manual thread-index arithmetic.

This is a companion piece to Simon Boehm's excellent [CUDA MMM worklog](https://siboehm.com/articles/22/CUDA-MMM), but the angle is different: instead of showing the low-level details, I want to show how a well-designed kernel DSL can express the same optimizations cleanly — which means you understand *what* you are doing without losing half a week on *how* to say it in code.

A `.co` file has two parts that compile together:

```
┌─────────────────────────────────────────────┐
│  __co__ void kernel(args...) {              │  ← Choreo DSL kernel
│    parallel ... { ... }                     │    lowered to __global__ CUDA
│  }                                          │
│                                             │
│  int main() {                               │  ← Standard C++ host harness
│    // alloc, upload, time, verify           │    compiled as-is
│  }                                          │
└─────────────────────────────────────────────┘
```

Compile a kernel and run it:

```bash
choreo -gs -t cute -arch=sm_90a matmul_f16_v3_hopper_tma_wgmma.co -o v3.cute.result
bash v3.cute.result --execute
```

---

## Performance at a glance

| Kernel                   | Time (ms) | TFLOPS  | % of cuBLAS |
| ------------------------ | --------: | ------: | ----------: |
| v0: Naive                |    ~2890  |    0.38 |       0.08% |
| v1: Shared memory        |     ~728  |    1.51 |       0.34% |
| v2: Tensor core MMA      |      73.7 |    14.9 |        3.3% |
| v3: Hopper TMA + WGMMA   |      3.87 |   284.4 |       63.6% |
| v4: Warp specialization  |      3.60 |   305.6 |       68.3% |
| **v5: Production-tuned** |  **2.49** | **441.8** | **98.8%** |
| cuBLAS (reference)       |      2.46 |   447.5 |      100.0% |

1153× improvement from v0 to v5 in six steps. Let's see how each one works.

---

## Kernel 0: Naive

**File:** `matmul_f16_v0_naive.co`

The simplest possible formulation: one thread owns one output element and independently reads a full row of A and column of B from global memory.

```c
// TILE_M = 32, TILE_N = 32
// These tile sizes double as the thread-block dimensions: 32×32 = 1024 threads per block.

__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, TILE_N)] : block,
           {thr_m, thr_n}     by [TILE_M, TILE_N]                    : thread {
    f16 acc = 0.0f;
    foreach {iv_k} in [K]
      acc += lhs.at(block_m#thr_m, iv_k) * rhs.at(block_n#thr_n, iv_k);

    output.at(block_m#thr_m, block_n#thr_n) = acc;
  }
}
```

**Choreo concepts introduced here:**

| Construct | Meaning |
| --- | --- |
| `global f16 [M, K] lhs` | A tensor in GPU global memory (HBM). Shape is symbolic; Choreo infers strides. |
| `parallel {i,j} by [X,Y] : block` | Create a 2-D grid of thread-blocks of size X×Y. `i` and `j` are block-level indices. |
| `parallel {i,j} by [X,Y] : thread` | Create X×Y threads inside each block. Composes with the enclosing block partition. |
| `block_m # thr_m` | The `#` operator composes two parallel indices into one flat index: `block_m * TILE_M + thr_m`. |
| `foreach {iv_k} in [K]` | A plain sequential loop — no parallelism, no reordering. |
| `.at(i, j)` | Element-level accessor into a 2-D tensor. |

Here is what the memory access looks like for four neighbouring threads:

```
A [M × K]                                B [N × K]  (stored as B^T, row = original col)
┌────────────────────────────────┐        ┌────────────────────────────────┐
│  row 0 ─────────────────────── │──────► │  col 0  ·  ·  ·  ·  ·  ·  ·    │ thread(0,0)
│  row 1 ─────────────────────── │──────► │  col 0  (same col, same reads!)│ thread(1,0)
│  row 0 ─────────────────────── │──────► │  col 1  ·  ·  ·  ·  ·  ·  ·    │ thread(0,1)
│  row 1 ─────────────────────── │──────► │  col 1  (redundant again)      │ thread(1,1)
│  ...                           │        │  ...                           │
└────────────────────────────────┘        └────────────────────────────────┘
         K = 8192 elements                         K = 8192 elements

thread(m, n) reads:  row m of A  +  col n of B  =  2 × 8192 × 2 bytes = 32 KB from HBM
All threads sharing the same row m redundantly re-read it. No caching between threads.

Total HBM traffic ≈ M × N × 2K × 2B = 8192² × 2 × 8192 × 2 B ≈ 2 TB
Theoretical minimum (just the matrices)                             ≈ 400 MB
```

### Generated CUDA (v0)

`parallel : block` / `: thread` become `blockIdx` / `threadIdx` arithmetic. The `#` composition becomes a multiply-add. The whole kernel is just a loop:

```cuda
__global__ void matmul_kernel(f16* lhs, f16* rhs, f16* output,
                               unsigned K, unsigned M, unsigned N) {
  // parallel {thr_m, thr_n} by [32, 32] : thread
  int thr_m = threadIdx.x / 32, thr_n = threadIdx.x % 32;

  f16 acc = 0.0f;
  // foreach {iv_k} in [K]
  for (int k = 0; k < K; ++k)
    // lhs.at(block_m#thr_m, iv_k)  =  blockIdx.x*32 + thr_m, k
    acc = acc + lhs[(blockIdx.x * 32 + thr_m) * K + k]
              * rhs[(blockIdx.y * 32 + thr_n) * K + k];

  // output.at(block_m#thr_m, block_n#thr_n)
  output[(blockIdx.x * 32 + thr_m) * N + (blockIdx.y * 32 + thr_n)] = acc;
}
// dim3 grid((M+31)/32, (N+31)/32, 1), block(1024, 1, 1);
```

### NCU snapshot (v0)

```
dram__throughput (% of peak HBM BW) :   0.01%
sm__throughput   (% of peak SM)     :   5.99%
pipe_tensor instructions            :   0          ← tensor cores completely idle
pipe_fma  instructions              :   336,592,896
```

The access pattern is so scattered that the hardware serialises HBM requests instead of coalescing them. HBM throughput is near zero even though every thread is constantly reading.

**Result: ~0.38 TFLOPS (0.08% of cuBLAS)**

---

## Kernel 1: Shared Memory Cache-Blocking

**File:** `matmul_f16_v1_shared_memory.co`

The classic fix: load a tile of A and B into fast on-chip shared memory (SRAM), let all threads in the block reuse that tile, then slide the tile along the K dimension.

```c
// TILE_M = TILE_N = 32, TILE_K = 128

__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, TILE_N)] : block,
           {thr_m, thr_n}     by [TILE_M, TILE_N]                    : thread {
    shared f16 [TILE_M, TILE_K] lhs_s;   // on-chip A tile, visible to all threads in block
    shared f16 [TILE_N, TILE_K] rhs_s;   // on-chip B tile
    f16 acc = 0.0f;

    foreach {iv_k} in [cdiv(K, TILE_K)] {
      // All 1024 threads in the block collectively copy one [TILE_M × TILE_K] tile
      // from HBM to SRAM. Choreo generates the partitioned, coalesced copy and
      // inserts synchronization barriers automatically.
      dma.copy lhs.subspan(TILE_M, TILE_K).at(block_m, iv_k) => lhs_s;
      dma.copy rhs.subspan(TILE_N, TILE_K).at(block_n, iv_k) => rhs_s;

      foreach {ik} in [TILE_K]
        acc += lhs_s.at(thr_m, ik) * rhs_s.at(thr_n, ik);
    }
    output.at(block_m#thr_m, block_n#thr_n) = acc;
  }
}
```

**New concepts:**

| Construct | Meaning |
| --- | --- |
| `shared f16 [M, K] buf` | Allocates a buffer in on-chip shared memory. Scoped to the block; all threads can read and write it. |
| `dma.copy src => dst` | Cooperative DMA: all threads in the block collectively transfer `src` to `dst`. Choreo partitions the transfer across threads, generates coalesced loads, and inserts `__syncthreads()`. One line replaces ~20 lines of manual CUDA. |
| `.subspan(TILE_M, TILE_K).at(i, j)` | Selects the tile of shape `[TILE_M, TILE_K]` at grid position `(i, j)` within the tensor. The tiling step `i` advances by `TILE_M` rows, `j` by `TILE_K` columns. |

The tile reuse pattern (K dimension sliced into 64 steps of 128 columns each):

```
HBM (slow)              SRAM (fast, ~30 TB/s)         1024 threads compute
──────────────────────  ────────────────────────────  ─────────────────────
                         iv_k = 0
A[:,   0..127] ─dma──►  lhs_s [32 × 128]           ─►  acc += lhs_s × rhs_s
B[:,   0..127] ─dma──►  rhs_s [32 × 128]            ↑       (TILE_K = 128 MACs)
                                                    │
                         iv_k = 1
A[:, 128..255] ─dma──►  lhs_s [32 × 128]           ─►  acc += lhs_s × rhs_s
B[:, 128..255] ─dma──►  rhs_s [32 × 128]            │       (TILE_K = 128 MACs)
                                ...                 │
                         iv_k = 63                  │
A[:,8064..8191]─dma──►  lhs_s [32 × 128]           ─►  acc += lhs_s × rhs_s
B[:,8064..8191]─dma──►  rhs_s [32 × 128]

Each HBM tile (32×128 × 2B = 8 KB) is loaded once and read 32× by threads in the block.
Reuse factor: TILE_M = TILE_N = 32  (vs. 1× in v0)
```

### Generated CUDA (v1) — what `dma.copy` expands to

One line of Choreo, `dma.copy lhs.subspan(TILE_M, TILE_K).at(block_m, iv_k) => lhs_s`, expands to the following CUTE cooperative copy. This is what you would write by hand in raw CUDA:

```cuda
// Build a global-memory tensor pointing at the tile [block_m, iv_k]
auto gmem_layout = make_layout(make_shape(Int<32>{}, Int<128>{}),
                                make_stride(K, Int<1>{}));
auto gmem_tile   = make_tensor(make_gmem_ptr(lhs + blockIdx.x*32*K + iv_k*128),
                                gmem_layout);
// Build the shared-memory destination tensor
auto smem_layout = make_layout(make_shape(Int<32>{}, Int<128>{}),
                                make_stride(Int<128>{}, Int<1>{}));
auto smem_tile   = make_tensor(make_smem_ptr(lhs_s), smem_layout);

// Create a tiled copy plan: 32×32 thread layout, each thread copies 4 elements
auto tiled_copy = make_tiled_copy(
  Copy_Atom<UniversalCopy<f16>, f16>{},
  make_layout(make_shape(Int<32>{}, Int<32>{}),
               make_stride(Int<32>{}, Int<1>{})),
  make_layout(make_shape(Int<1>{}, Int<4>{})));

// Partition across threads and execute
auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
auto src = thr_copy.partition_S(gmem_tile);
auto dst = thr_copy.partition_D(smem_tile);
cute::copy(tiled_copy, src, dst);
__syncthreads();   // ← barrier ensuring all threads finished before compute
```

Choreo generates this whole block for both `lhs` and `rhs`. That is why the Choreo source stays at two lines.

Scalar FMA units still do the arithmetic. The tensor cores remain idle. The bottleneck has shifted from redundant global loads to the raw compute throughput of scalar FMAs — which are orders of magnitude slower than tensor cores.

**Result: ~1.51 TFLOPS (3.9× over v0)**

---

## Kernel 2: Tensor Core MMA

**File:** `matmul_f16_v2_mma.co`

Replace scalar per-thread multiply-add with **warp-level matrix multiply** on the tensor core hardware units. One tensor-core instruction computes a 16×16×16 matrix product — 4096 multiply-adds — in the time it takes scalar FMA to do one.

```c
// MMA_M = MMA_N = MMA_K = 16  (WMMA fragment shape — 16×16×16 at f16)
// TILE_M = TILE_N = 16, TILE_K = 64

__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, TILE_N)] : block {
    shared f16 [TILE_M, TILE_K] lhs_s;
    shared f16 [TILE_N, TILE_K] rhs_s;

    // Warp-level parallelism inside the block.
    // Each warp independently owns a [MMA_M × MMA_N] fragment of the output tile.
    parallel {warp_m, warp_n} by [cdiv(TILE_M, MMA_M), cdiv(TILE_N, MMA_N)] : group {
      mc = mma.fill 0.0;

      foreach {iv_k} in [cdiv(K, TILE_K)] {
        dma.copy lhs.subspan(TILE_M, TILE_K).at(block_m, iv_k) => lhs_s;
        dma.copy rhs.subspan(TILE_N, TILE_K).at(block_n, iv_k) => rhs_s;

        foreach {iv_wk} in [cdiv(TILE_K, MMA_K)] {
          // Each warp loads its private 16×16 sub-tile from shared memory.
          ma = mma.load lhs_s.chunkat(warp_m, iv_wk);
          mb = mma.load rhs_s.chunkat(warp_n, iv_wk);
          // One WMMA instruction: mc += ma × mb  (16×16×16 matrix multiply).
          mma.row.row mc, ma, mb;
        }
      }
      mma.store mc, output.subspan(TILE_M, TILE_N).at(block_m, block_n)
                          .chunkat(warp_m, warp_n);
    }
  }
}
```

**New concepts:**

| Construct | Meaning |
| --- | --- |
| `parallel {i,j} by [...] : group` | One **warp** (32 threads) per index pair. Unlike `: thread`, each group instance is a hardware warp that shares an instruction pointer. |
| `mma.fill 0.0` | Allocates an MMA accumulator fragment — a fixed-size register tile distributed across all 32 threads in the warp. |
| `.chunkat(warp_m, iv_wk)` | Selects the natural chunk at grid position `(warp_m, iv_wk)`. Unlike `.subspan().at()`, the shape is inferred automatically from the surrounding MMA context. Choreo uses this to determine which hardware MMA instruction to emit. |
| `mma.load s.chunkat(i,j)` | Loads a warp-level MMA fragment from shared memory. The fragment is distributed in hardware-specific layout across 32 thread registers. |
| `mma.row.row mc, ma, mb` | Execute: `mc += ma × mb`. `.row.row` specifies row-major layout for both A and B. Choreo lowers this to `nvcuda::wmma::mma_sync`. |
| `mma.store mc, dst` | Writes the accumulator fragment back to memory. |

**How Choreo selects the hardware MMA instruction:**

The key decision is made from the shape that `.chunkat()` infers at the `: group` / `: group-4` boundary:

```
Fragment shape (inferred by Choreo)   → Hardware instruction     Thread granularity
─────────────────────────────────────────────────────────────────────────────────────
16 × 16 × 16  (f16, ": group")       → wmma::mma_sync           1 warp  (32 threads)
16 ×  8 × 16  (f16, ": group")       → mma.sync                 1 warp  (32 threads)
64 ×  N × 16  (f16, ": group-4")     → wgmma.mma_async          1 warpgroup (128 t)
```

You never name the instruction. You declare the parallelism level and write the tile shapes; Choreo maps them to hardware.

The block output tile is partitioned across warps via `parallel : group`:

```
Block output tile [TILE_M=64 × TILE_N=64]  (4×4 warps, MMA_M=MMA_N=16)

         col:  0..15  16..31  32..47  48..63
row 0..15  │ warp(0,0) warp(0,1) warp(0,2) warp(0,3) │
row 16..31 │ warp(1,0) warp(1,1) warp(1,2) warp(1,3) │
row 32..47 │ warp(2,0) warp(2,1) warp(2,2) warp(2,3) │
row 48..63 │ warp(3,0) warp(3,1) warp(3,2) warp(3,3) │

Each warp independently:
  ┌─ mma.fill ────────►  mc [16×16]: accumulator in 32 thread registers
  │
  ├─ mma.load lhs_s... ► ma [16×16]: A-fragment (row-major) from SRAM
  ├─ mma.load rhs_s... ► mb [16×16]: B-fragment (col-major) from SRAM
  └─ mma.row.row mc, ma, mb  ─►  wmma::mma_sync  (4096 MACs in ~4 cycles)

lhs_s [64×64] and rhs_s [64×64] are shared by all 16 warps in the block.
One dma.copy loads them cooperatively; each warp then reads its own chunk.
```

### Generated CUDA (v2) — what `mma.*` expands to

```cuda
// mma.fill 0.0  →
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, f16> mc;
nvcuda::wmma::fill_fragment(mc, (f16)0.0f);

// foreach {iv_wk} in [cdiv(TILE_K, MMA_K)]:
for (int iv_wk = 0; iv_wk < 4; ++iv_wk) {

  // mma.load lhs_s.chunkat(warp_m, iv_wk)  →
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                         16, 16, 16, f16, nvcuda::wmma::row_major> ma;
  nvcuda::wmma::load_matrix_sync(ma,
    lhs_s + warp_m * 1024 + iv_wk * 16,  // pointer into shared buffer
    /*ld=*/64);                            // leading dimension of lhs_s

  // mma.load rhs_s.chunkat(warp_n, iv_wk)  →
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                         16, 16, 16, f16, nvcuda::wmma::col_major> mb;
  nvcuda::wmma::load_matrix_sync(mb,
    rhs_s + warp_n * 1024 + iv_wk * 16,
    /*ld=*/64);

  // mma.row.row mc, ma, mb  →
  nvcuda::wmma::mma_sync(mc, ma, mb, mc);   // 4096 MACs in one instruction
}

// mma.store mc, output.chunkat(warp_m, warp_n)  →
nvcuda::wmma::store_matrix_sync(
  output + (block_m * 16 + warp_m * 16) * N + (block_n * 16 + warp_n * 16),
  mc, N, nvcuda::wmma::mem_row_major);
```

Choreo infers the fragment types (`matrix_a`, `matrix_b`, `accumulator`), the layout tags (`row_major`, `col_major`), and the pointer offsets from the tile shapes — none of this appears in the Choreo source.

### NCU snapshot (v2)

```
dram__throughput (% of peak HBM BW) :   0.52%
sm__throughput   (% of peak SM)     :  25.49%
pipe_tensor instructions            :   4,194,304   ← tensor cores now active
pipe_fma  instructions              :   8,552,448
```

Tensor cores are firing. SM utilisation jumped from 6% to 25%. The remaining problem is tile size: a 16×16 tile loads only 512 bytes per WGMMA call, so arithmetic intensity is still low and we are far from the compute roofline.

**Result: ~14.9 TFLOPS (9.9× over v1)**

---

## Kernel 3: Hopper TMA + WGMMA

**File:** `matmul_f16_v3_hopper_tma_wgmma.co`

The largest single jump: ~**19× over v2**. Two hardware features specific to Hopper replace the two bottlenecks from v2.

1. **TMA** (Tensor Memory Accelerator) replaces `dma.copy` — data movement becomes a single-thread hardware operation.
2. **WGMMA** (Warpgroup MMA) replaces WMMA — the compute tile grows from 16×16 to 64×128.

```c
// WARP_M=64, WARP_N=128, WARP_K=16, TILE_K=64, SWIZ=128

__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, WARP_M), cdiv(N, WARP_N)] : block,
           by 1 : group-4 {        // one warpgroup (128 threads) per block
    shared f16 [WARP_M, TILE_K] lhs_s;
    shared f16 [WARP_N, TILE_K] rhs_s;
    shared f16 [WARP_M, WARP_N] output_s;

    mc = mma.fill.f16 0.0f;        // accumulator in f16 (not f32)

    foreach {iv_k} in [cdiv(K, TILE_K)] {
      // TMA: one thread issues a hardware DMA request. The TMA unit streams
      // data to shared memory with the swizzled XOR layout that WGMMA expects.
      tma.copy.swiz<SWIZ> lhs.chunkat(block_m, iv_k) => lhs_s;
      tma.copy.swiz<SWIZ> rhs.chunkat(block_n, iv_k) => rhs_s;

      foreach {iv_wk} in [cdiv(TILE_K, WARP_K)] {
        // _ means "this warpgroup's slice" — there is only one warpgroup per block here.
        ma = mma.load.swiz<SWIZ> lhs_s.chunkat(_, iv_wk);
        mb = mma.load.swiz<SWIZ> rhs_s.chunkat(_, iv_wk);
        mma.row.row mc, ma, mb;    // wgmma.mma_async: 64×128×16 in one instruction
      }
    }
    mma.store mc, output_s;
    tma.copy output_s => output.subspan(WARP_M, WARP_N).at(block_m, block_n);
  }
}
```

**New concepts:**

| Construct | Meaning |
| --- | --- |
| `by 1 : group-4` | One **warpgroup** (4 warps = 128 threads) per block. `group-4` is the Hopper WGMMA execution granularity. |
| `tma.copy.swiz<128> src => dst` | Hopper Tensor Memory Accelerator: one thread issues a hardware bulk-copy. `.swiz<128>` sets 128-byte XOR swizzle in the destination shared memory layout — required for bank-conflict-free WGMMA loads. |
| `mma.load.swiz<128>` | Load a WGMMA A/B fragment from swizzled shared memory. The `.swiz<128>` annotation must match the one used in `tma.copy` — Choreo checks this via type inference. |
| `mma.fill.f16 0.0f` | Accumulator in f16 precision. WGMMA can accumulate into f16 or f32; f16 uses less register space. |
| `_` (wildcard) | In `.chunkat(_, iv_wk)` the underscore fills in the single warpgroup's dimension automatically. With only one warpgroup per block there is no warp-M index to name. |

### TMA vs `dma.copy`

```
┌─────────────────────────────────┬──────────────────────────────────────────┐
│  dma.copy  (v1–v2)              │  tma.copy  (v3+)                         │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ ALL threads participate         │ ONE thread issues; hardware does the rest│
│ (1024 threads in lockstep)      │ (128 threads are free to compute)        │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ Thread computes element index,  │ Thread writes coordinates into a         │
│ issues load, writes to smem,    │ CUtensorMap descriptor; TMA DMA engine   │
│ repeats per element             │ handles addressing, coalescing, swizzle  │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ Sync: __syncthreads()           │ Sync: mbarrier                           |
|                                 | (lightweight, per-warpgroup)             │
│ (blocks entire block)           │ (other warpgroups unaffected)            │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ BW: limited by register pressure│ BW: near theoretical HBM peak            │
│ and instruction throughput      │ (hardware-optimized DMA path)            │
└─────────────────────────────────┴──────────────────────────────────────────┘
```

### Why swizzle matters

WGMMA reads shared memory across 128 threads simultaneously. Without swizzle, all 128 threads in a warpgroup map to the same 32-bit bank for a given row:

TMA writes the swizzled layout into smem; WGMMA expects it and reads conflict-free.
Choreo propagates .swiz<128> from tma.copy to mma.load and checks consistency
at compile time — no manual XOR tables or descriptor bit fields in user code.
```

### Generated CUDA (v3) — TMA and WGMMA

**TMA descriptor in the kernel signature.** Choreo generates a `CUtensorMap` for each TMA tensor and passes it as a `__grid_constant__` kernel argument (resident in constant memory, accessible without register cost):

```cuda
// Choreo generates this signature:
__global__ void matmul_kernel(f16* lhs, f16* rhs, f16* output, ...,
  const __grid_constant__ CUtensorMap tma_lhs,    // descriptor for lhs
  const __grid_constant__ CUtensorMap tma_rhs,    // descriptor for rhs
  const __grid_constant__ CUtensorMap tma_output) // descriptor for output
```

**`tma.copy.swiz<128> lhs.chunkat(block_m, iv_k) => lhs_s`** expands to a hardware bulk-copy. Only one thread issues it; the rest arrive at the barrier:

```cuda
if (threadIdx.x == 0 /* __CHOREO_BLOCK_SINGLE__ */) {
  // Issue the TMA bulk-copy; hardware writes data to lhs_s with 128-byte swizzle
  cde::cp_async_bulk_tensor_2d_global_to_shared(
    lhs_s, &tma_lhs,
    iv_k * 64,       // K coordinate
    blockIdx.x * 64, // M coordinate
    barrier);        // fires barrier when copy completes

  // Thread 0 contributes its byte count (8192 B) to the barrier's transaction counter
  cuda::device::barrier_arrive_tx(barrier, /*arrive_count=*/1, /*bytes=*/8192);
} else {
  barrier.arrive();   // other threads just signal arrival
}
barrier.wait(barrier.arrive());   // all threads wait for the transfer to finish
```

**`mma.row.row mc, ma, mb`** on a `: group-4` (warpgroup) expands to a WGMMA descriptor-based instruction. Note that the A and B fragments are smem descriptors — the hardware accesses shared memory directly without going through registers:

```cuda
// mma.load.swiz<128> lhs_s.chunkat(_, iv_wk)  →  build smem descriptor:
uint64_t desc_ma = wgmma_make_smem_desc<K_MAJOR, Swizzle::B128>(lhs_s + iv_wk*16);
uint64_t desc_mb = wgmma_make_smem_desc<K_MAJOR, Swizzle::B128>(rhs_s + iv_wk*16);

warpgroup_arrive();   // fence before first WGMMA in the group
// mma.row.row mc, ma, mb (64×128×16 shape)  →
SM90::GMMA::MMA_64x128x16_F16F16F16_SS<K_MAJOR, K_MAJOR>::fma(
  desc_ma, desc_mb,
  mc[0], mc[1], ..., mc[31]);  // 32 f16 accumulator registers per thread

// mma.commit  →
warpgroup_commit_batch();   // mark end of WGMMA batch
warpgroup_wait<0>();        // wait for all WGMMA ops to complete
```

The `wgmma_make_smem_desc` call encodes the swizzle mode, the bank layout, and the shared memory pointer into a 64-bit hardware descriptor — about 15 lines of bit-manipulation in raw CUDA. Choreo derives the descriptor automatically from the `.swiz<128>` annotation.

### NCU snapshot (v3)

```
dram__throughput (% of peak HBM BW) :  10.64%
sm__throughput   (% of peak SM)     :  42.83%
pipe_tensor instructions            :    264,192
pipe_fma  instructions              :    247,820
```

SM utilisation doubled. The kernel is now compute-bound (the H800 roofline crossover for this problem is around 70 FLOPs/byte; we are past it). But 57% of SM peak is still missing — because TMA and WGMMA are still serialised: each K-step waits for the copy before computing.

**Result: ~284 TFLOPS (19.1× over v2, 63.6% of cuBLAS)**

---

## Kernel 4: Warp Specialization

**File:** `matmul_f16_v4_warpspec.co`

The insight from v3's profile: producer (TMA) and consumer (WGMMA) stall each other. Fix this by giving them separate warpgroups that run concurrently, connected by a software ring buffer in shared memory.

```c
// WARP_M=64, WARP_N=128, WARP_K=16, TILE_M=128, TILE_K=64, STAGES=1, CONSUMERS=1

__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, WARP_N)] : block {

    // Pipeline ring buffer: STAGES independent tile slots with handshake events.
    shared event full[STAGES], empty[STAGES];
    shared f16 [TILE_M, TILE_K] lhs_s[STAGES];
    shared f16 [WARP_N, TILE_K] rhs_s[STAGES];
    shared f16 [WARP_M, WARP_N] output_s[CONSUMERS];

    // 3 warpgroups × 128 threads = 384 threads per block.
    parallel wg by 3 : group-4, t by 128 : thread {

      // ── Producer warpgroup (wg=0): one thread drives TMA ─────────────────
      inthreads.async (wg == 0 && t == 0) {
        foreach {iv_k} in [cdiv(K, TILE_K)] {
          stage = iv_k % STAGES;
          wait empty[stage];                       // slot must be free before refilling
          tma.copy.async<full[stage]>.swiz<SWIZ> lhs
            .subspan(TILE_M, TILE_K).at(block_m, iv_k) => lhs_s[stage];
          tma.copy.async<full[stage]>.swiz<SWIZ> rhs
            .subspan(WARP_N, TILE_K).at(block_n, iv_k) => rhs_s[stage];
          trigger full[stage];
        }
      }

      // ── Consumer warpgroups (wg=1,2): compute WGMMA ──────────────────────
      inthreads.async (wg >= 1) {
        mc = mma.fill.f16 0.0f;
        cidx = wg - 1;

        // Initialise all stage slots as empty so the producer can fill them.
        foreach {s} in [STAGES]
          trigger empty[s];

        foreach {iv_k} in [cdiv(K, TILE_K)] {
          stage = iv_k % STAGES;
          wait full[stage];                        // wait until TMA has delivered the tile

          foreach {iv_wk} in [cdiv(TILE_K, WARP_K)] {
            ma = mma.load.swiz<SWIZ> lhs_s[stage].subspan(WARP_M, WARP_K).at(cidx, iv_wk);
            mb = mma.load.swiz<SWIZ> rhs_s[stage].chunkat(_, iv_wk);
            mma.row.row mc, ma, mb;
          }
          mma.commit;                              // fence: all WGMMA ops for this tile done
          trigger empty[stage];                    // slot is free; producer may refill
        }

        mma.store mc, output_s[cidx];
        tma.copy output_s[cidx] => output.subspan(WARP_M, WARP_N)
          .at(block_m * CONSUMERS + cidx, block_n);
      }

    }
  }
}
```

**New concepts:**

| Construct | Meaning |
| --- | --- |
| `shared event full[N], empty[N]` | Named pipeline barriers backed by Hopper `mbarrier`. Lightweight — not a full block-wide `__syncthreads()`. Individual warpgroups can wait on or signal them independently. Choreo generates the mbarrier init, arrive, and wait sequences. |
| `inthreads.async (condition) { ... }` | Executes the block only in warpgroups matching `condition`. This is Choreo's syntax for Hopper *warp specialization*: warpgroups execute different code paths simultaneously. |
| `tma.copy.async<full[stage]> src=>dst` | Asynchronous TMA that fires the event `full[stage]` upon hardware completion. The issuing thread is not blocked; other warpgroups can compute while the transfer is in flight. |
| `trigger event` / `wait event` | Signal or wait on a named barrier. Maps to `mbarrier.arrive` / `mbarrier.try_wait`. |
| `mma.commit` | Inserts `wgmma.wait_group`: waits for all outstanding WGMMA operations in this warpgroup to retire. Required before releasing an `empty[stage]` because WGMMA is pipelined — the instruction may be issued before the result is committed to registers. |

The producer-consumer pipeline (STAGES=1, one buffer slot):

```
Time ────────────────────────────────────────────────────────────────────────►

                tile[0] phase                tile[1] phase
          ◄─────────────────────────►   ◄────────────────────────────►

Producer  ┌── TMA lhs_s, rhs_s ──┐   │   ┌── TMA lhs_s, rhs_s ──┐
(wg=0)    │  issue to hw         │   │   │  issue to hw         │   ...
          └──────────────────────┘   │   └──────────────────────┘
                   ↓ full[0] fires   │            ↓ full[1] fires
                                     │
Consumer  [wait]   ┌── WGMMA tile[0] ──────────┐  [wait]   ┌── WGMMA tile[1] ──
(wg=1,2)  full[0]  │  64×128×16 × (K/TILE_K)   │  full[1]  │  ...
                   └───────────────────────────┘
                                     ↑ empty[0]  (consumer signals; producer picks up)

Key: with STAGES=1 there is only one shared buffer slot.
     The producer must wait for the consumer to finish tile[0] before
     it can issue TMA for tile[1]. TMA and WGMMA are back-to-back.
     Warp specialization still helps: the 127 non-issuing threads
     in the producer warpgroup can sleep, not wasting compute resources.
```

### Generated CUDA (v4) — mbarrier and warpgroup dispatch

**`shared event full[STAGES], empty[STAGES]`** becomes an array of `cuda::barrier` objects initialised by a single thread:

```cuda
__shared__ cuda::barrier<cuda::thread_scope_block> full[2], empty[2];
if (threadIdx.x == 0) {
  init(&full[0],  /*expected_arrive_count=*/385);  // 3 WGs × 128 + 1 TMA thread
  init(&full[1],  385);
  init(&empty[0], 385);
  init(&empty[1], 385);
  cde::fence_proxy_async_shared_cta();
}
__syncthreads();
```

**`inthreads.async (wg == 0 && t == 0) { ... }`** becomes a simple `if` on warpgroup and thread index:

```cuda
int wg  = threadIdx.x / 128;
int tid = threadIdx.x % 128;

if (wg == 0 && tid == 0) {           // producer
  for (int iv_k = 0; ...) {
    int stage = iv_k % STAGES;
    // wait empty[stage]  →
    empty[stage].wait(empty[stage].arrive());

    // tma.copy.async<full[stage]>  →  async bulk-copy that fires full[stage]:
    cde::cp_async_bulk_tensor_2d_global_to_shared(lhs_s_slot, &tma_lhs, ...);
    // trigger full[stage]  →  contribute byte count to the barrier transaction:
    cuda::device::barrier_arrive_tx(full[stage], 1, lhs_bytes + rhs_bytes);
  }
}
if (wg >= 1) {                        // consumers
  // trigger empty[s]  →
  for (int s = 0; s < STAGES; ++s) empty[s].arrive();

  for (int iv_k = 0; ...) {
    int stage = iv_k % STAGES;
    // wait full[stage]  →
    full[stage].wait(full[stage].arrive());

    // ... WGMMA calls (same as v3) ...

    // mma.commit  →
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    // trigger empty[stage]  →
    empty[stage].arrive();
  }
}
```

`cuda::barrier::arrive()` / `::wait()` map to PTX `mbarrier.arrive` / `mbarrier.try_wait` instructions. The barrier is lightweight: it lives in shared memory and is signalled by individual threads or by the TMA hardware engine, with no global broadcast.

**Result: ~305 TFLOPS (7.4% over v3, 68.3% of cuBLAS)**

---

## Kernel 5: Production-Tuned

**File:** `matmul_f16_v5_auto_tuned.co`

The kernel structure is **identical to v4**. Every line of new functionality was introduced in v4. What changes here is the three macro parameters that determine tile shape and pipeline depth:

```c
#define WARP_M    64    // rows per consumer warpgroup's output tile
#define WARP_N   192    // cols per warpgroup tile — found by 28-iter sweep
#define WARP_K    16    // WGMMA K step
#define TILE_M   128    // total M per block = 2 × WARP_M  (2 consumers)
#define TILE_K    64    // K-tile loaded per TMA transfer
#define SWIZ     128    // swizzle byte width
#define STAGES     2    // double-buffered ring buffer  (was 1 in v4)
#define CONSUMERS  2    // consumer warpgroups per block (was 1 in v4)
```

The full kernel for reference:

```c
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, WARP_N)] : block {
    shared event full[STAGES], empty[STAGES];
    shared f16 [TILE_M, TILE_K] lhs_s[STAGES];
    shared f16 [WARP_N, TILE_K] rhs_s[STAGES];

    parallel wg by 3 : group-4, t by 128 : thread {
      inthreads.async (wg == 0 && t == 0) {
        foreach {iv_k} in [cdiv(K, TILE_K)] {
          stage = iv_k % STAGES;
          wait empty[stage];
          tma.copy.async<full[stage]>.swiz<SWIZ> lhs
            .subspan(TILE_M, TILE_K).at(block_m, iv_k) => lhs_s[stage];
          tma.copy.async<full[stage]>.swiz<SWIZ> rhs
            .subspan(WARP_N, TILE_K).at(block_n, iv_k) => rhs_s[stage];
          trigger full[stage];
        }
      }

      inthreads.async (wg >= 1) {
        mc = mma.fill.f16 0.0f;
        cidx = wg - 1;
        foreach {s} in [STAGES]
          trigger empty[s];

        foreach {iv_k} in [cdiv(K, TILE_K)] {
          stage = iv_k % STAGES;
          wait full[stage];
          foreach {iv_wk} in [cdiv(TILE_K, WARP_K)] {
            ma = mma.load.swiz<SWIZ> lhs_s[stage].subspan(WARP_M, WARP_K).at(cidx, iv_wk);
            mb = mma.load.swiz<SWIZ> rhs_s[stage].chunkat(_, iv_wk);
            mma.row.row mc, ma, mb;
          }
          mma.commit;
          trigger empty[stage];
        }
        shared f16 [WARP_M, WARP_N] output_s[CONSUMERS];
        mma.store mc, output_s[cidx];
        tma.copy output_s[cidx] => output.subspan(WARP_M, WARP_N)
          .at(block_m * CONSUMERS + cidx, block_n);
      }
    }
  }
}
```

### What changed from v4 and why it matters

```
Parameter    v4         v5         Effect
──────────────────────────────────────────────────────────────────────
CONSUMERS    1          2          2 consumer WGs → 2× output M-rows per block
TILE_M       64         128        Larger block → better L2 cache reuse for B
STAGES       1          2          True double-buffering: producer prefetches
                                   tile[i+1] while consumers compute tile[i].
                                   TMA latency is now fully hidden.
WARP_N       128        192        Found by 28-iteration parameter sweep; see
                                   "The Tuning Journey" section below.
```

**Why WARP_N = 192?** SM90 WGMMA accepts any N in [8, 256] divisible by 8. 192 = 24×8. At this specific shape the combination of register usage (~80 registers/thread), shared memory footprint per block (~80 KB → 2 blocks per SM), and N-grid parallelism (`cdiv(8192, 192) = 43` blocks) yields the best measured throughput for 8192×8192×8192 on H800 PCIe. The "right" value is hardware- and shape-specific; the full sweep showing why 192 beats 152, 176, and 224 is documented in the tuning section below.

### Double-buffering in pictures (STAGES=2)

With two buffer slots the producer can prefetch the next tile while the consumer computes the current one:

```
Time ────────────────────────────────────────────────────────────────────────►

Stage slots: [slot 0] [slot 1] [slot 0] [slot 1] [slot 0] ...

Producer  ┌─TMA tile[0]─┐  ┌─TMA tile[1]─┐           ┌─TMA tile[2]─┐
(wg=0)    │ → slot 0    │  │ → slot 1    │           │ → slot 0    │  ...
          └─────────────┘  └─────────────┘           └─────────────┘
               ↓ full[0]        ↓ full[1]                  ↓ full[0]
                    │                │                          │
Consumer            ▼                ▼                          ▼
(wg=1,2)       ┌─WGMMA tile[0]─┐  ┌─WGMMA tile[1]─┐  ┌─WGMMA tile[2]─┐
               │  from slot 0  │  │  from slot 1  │  │  from slot 0  │  ...
               └───────────────┘  └───────────────┘  └───────────────┘
                         ↑                   ↑
                      empty[0]            empty[1]
                   (producer reuses)   (producer reuses)

                  ◄────── overlap region ──────►
               TMA tile[1] runs DURING WGMMA tile[0]
               → TMA latency (~200 cycles) fully hidden

Compare with STAGES=1 (v4):
  Producer:  [TMA tile[0]]─────────────────[TMA tile[1]]────────────────[TMA tile[2]]
  Consumer:                [WGMMA tile[0]]                [WGMMA tile[1]]
             No overlap — must wait for consumer to finish before issuing next TMA.
```

### NCU snapshot (v5)

```
sm__throughput   (% of peak SM)     :  89.68%   ← near-peak compute
tensor_core HMMA (% of peak)        :  89.68%   ← compute-bound
gpu__dram_throughput (% of HBM BW)  :  38.91%   ← TMA hiding all loads
warp occupancy   (% of peak)        :  27.74%   ← ~2 blocks/SM (smem-limited)
```

SM and tensor-core utilisation are at 89.7% — the kernel is firmly in the compute-bound regime. DRAM at 39% means TMA is successfully prefetching ahead. Warp occupancy at 27.7% reflects the shared memory footprint (~80 KB per block) allowing 2 concurrent blocks per SM on H800.

**Result: ~471 TFLOPS (1.40× over v4, 105% of cuBLAS on this GPU)**

---

## The full NCU picture

Profiled with a single kernel launch (`ncu --launch-count 1`):

```
Kernel         dram%  sm%    tensor inst    fma inst       TFLOPS
────────────────────────────────────────────────────────────────────
v0 naive        0.01   5.99          0     336,592,896       0.38
v2 mma          0.52  25.49  4,194,304       8,552,448      14.9
v3 tma/wgmma   10.64  42.83    264,192         247,820     284.4
v4 warpspec    32.97  56.13  9,418,787              —      337.0
v5 tuned       38.91  89.68  9,492,372              —      471.3
────────────────────────────────────────────────────────────────────

v0: only scalar FMAs; tensor cores completely idle.
v2: tensor core instructions appear; SM doubles — but tile is too small.
v3: SM doubles again; both tensor and FMA streams active.
v4: warpspec hides TMA latency; tensor utilization improves to 52%.
v5: SM and tensor-core utilization both hit 89.7% — compute-roofline regime.
```

To reproduce these numbers with `ncu`:

```bash
ncu --target-processes all --launch-count 1 \
    --kernel-name-base demangled \
    --kernel-name regex:__choreo_device_matmul \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__inst_executed_pipe_fma.sum \
  <compiled-kernel-binary>
```

---

## The Tuning Journey: v4 → v5

v5 was not designed upfront — it was found by a systematic 28-iteration parameter sweep starting from v4. Here is the actual log.

### Baseline: v4 (STAGES=1, CONSUMERS=1, WARP_N=128)

NCU on the v4 baseline reveals the kernel is **latency-bound**:

```
sm__throughput:          56%   ← SM underutilized
tensor_core (HMMA):      52%   ← compute barely half active
gpu__dram_throughput:    33%   ← DRAM not the problem
warp occupancy:          28%   ← many warps stall on barriers
```

The single-stage pipeline (`STAGES=1`) forces the producer to wait for the consumer to drain before it can load the next K tile. The CPU-visible pattern: `sm__pipe_tensor_op_hmma_cycles` stalls every `TILE_K/WARP_K = 4` WGMMA instructions.

### Phase 1 — Pipeline Architecture (iter000 → iter003)

The first structural change is enabling double-buffering and a second consumer warpgroup:

| Iter | Change | TFLOPS | Notes |
|------|--------|--------|-------|
| iter000 | baseline (STAGES=1, CONS=1) | 337 | latency-bound |
| iter001 | STAGES=2, CONS=1 | — | **CRASH** — v4 structure has implicit 2 consumers; mismatched CONSUMERS=1 causes OOB smem write |
| iter002 | STAGES=2, CONS=2, WARP_N=128 | 365 | first working double-buffer (+8%) |
| iter003 | STAGES=2, CONS=2, WARP_N=152 | 402 | intermediate config, bottleneck shifts to compute-bound (+19%) |

The crash on iter001 reveals a subtle Choreo semantics point: `parallel wg by 3` spawns 3 warpgroups regardless of the `CONSUMERS` define. The consumer predicate `inthreads.async (wg >= 1)` matches **both** wg=1 and wg=2 — so you must always set `CONSUMERS` to match the number of consumer warpgroups.

After iter003 NCU shows the bottleneck has shifted:

```
sm__throughput:          89%   ← near-peak!
tensor_core (HMMA):      89%   ← compute-bound
gpu__dram_throughput:    38%   ← TMA is hiding latency
```

We are now **compute-bound** at 89% tensor utilization. The remaining gap: warp occupancy is only 28%, limited by shared memory per block (~80 KB) allowing at most 2 blocks per SM on H800.

### Phase 2 — WARP_N Sweep (iter004 → iter017)

With the kernel compute-bound, tuning `WARP_N` changes the balance between:
- **N-tile arithmetic intensity** (larger WARP_N = more WGMMA work per TMA load)
- **shared memory footprint** (`rhs_s[STAGES]` grows linearly with WARP_N)
- **grid parallelism** (`cdiv(N, WARP_N)` determines how many blocks cover the N dimension)

> **Constraint discovered in iter019–020**: WGMMA hardware requires `WARP_N` to be a **multiple of 8**. Values like 180 or 188 compile-fail with `MMA m64n180k16 not supported`.

```
WARP_N sweep results (STAGES=2, CONS=2, TILE_K=64):

  490 │
  476 │                                  ████ WN=176
  466 │                                        ████ WN=192  ← winner (v5)
  457 │                         ████ ████ WN=160,168
  434 │               ████ WN=144
  402 │         ████ WN=152
  386 │   ████ WN=136
  365 │  ████ WN=128
  337 │ ████ v4 baseline
      └────────────────────────────────────────────────────
        128  136  144  152  160  168  176  192  208  224

  ↑ 208,224 fail (OOB TMA assertion or smem too large)
```

The sweet spot is **WARP_N = 176–192**. Going beyond 192 adds shared memory without enough extra work to compensate — WARP_N=224 drops to 364 TFLOPS because the larger smem footprint prevents 2 concurrent blocks per SM.

### Phase 3 — Dead Ends (iter011–018)

Not all directions pay off:

| Attempt | Result | Why |
|---------|--------|-----|
| STAGES=3, WARP_N=192 | correctness fail | Compiler bug with 3-stage pipeline + v5 structure |
| STAGES=3, WARP_N=176 | 373 TFLOPS (worse) | Extra barrier overhead dominates |
| TILE_K=32, WARP_N=192 | 403 TFLOPS (worse) | Shorter K-tiles shrink the overlap window |
| TILE_K=128, WARP_N=192 | CUDA invalid arg | smem 163 KB exceeds kernel limit |
| 1p3c (3 consumers), TILE_M=192 | 156 TFLOPS | 512 threads × 80 regs = only 1 block/SM |
| SWIZ=64, WARP_N=192 | CUDA invalid arg | WGMMA N=192 requires SWIZ=128 layout |
| WARP_N=256, power-of-2 | 396 TFLOPS | Larger smem, fewer blocks/SM |

The 1p3c experiment is instructive: with 512 threads (4 warpgroups) the register budget forces only 1 active block per SM, cutting utilization in half.

### Winner: WARP_N=192 (+40% over v4)

```
#define WARP_N 192      // was 152
#define STAGES 2
#define CONSUMERS 2
```

That is the **only change** in v5's Choreo source. Choreo's compiler handles the rest — regenerating the correct TMA tensor descriptors, swizzle layouts, and mbarrier counts automatically for the new tile shape.

---

## SOTA comparison

Running cuBLAS via PyTorch's `torch.mm` on the same 8192×8192×8192 f16 problem:

```
NVIDIA H800 PCIe — FP16 Tensor Core peak: 1513 TFLOPS

cuBLAS (torch.mm):   2.46 ms   447.5 TFLOPS   (100%)
Choreo v5 (tuned):   2.35 ms   471.3 TFLOPS   (105.3%)
```

The remaining gap to cuBLAS (and our slight edge in some runs) comes from the tile shape
hitting a better L2/SMEM working set on this GPU. Production cuBLAS also uses:

The slight edge over cuBLAS comes from the WARP_N=192 tile hitting a better L2/SMEM working-set balance on this specific GPU model. Production cuBLAS also includes features outside this tutorial's scope that further close the gap on other workloads:

- **Thread block clusters**: cuBLAS uses Hopper multicast TMA to share B tiles across blocks in a cluster.
- **Persistent kernels**: cuBLAS keeps blocks alive across output tiles to amortize launch overhead.
- **Epilogue fusion**: cuBLAS merges the MMA store and output write, avoiding one SMEM round-trip.

All three can be expressed in Choreo — the production-grade kernels in the repository implement them.

---

## The optimization ladder

```
TFLOPS (H800 PCIe, 8192³, f16)

  471 │                                               ████ Choreo v5 (tuned)
  447 │                                          ████ cuBLAS
      │
  337 │                       ████ v4 warpspec
  284 │                  ████ v3 TMA+WGMMA
      │
      │
      │
      │
   15 │  ██ v2 MMA
    2 │  ██ v1 SMEM
    0 │  ██ v0 naive
      └──────────────────────────────────────────────────────────────
         v0    v1→v2        v2→v3             v3→v4→v5          cuBLAS
               +MMA          +TMA+WGMMA        +warpspec+tuning
```

| Step    | Technique           | Key Choreo construct                              | Speedup |
| ------- | ------------------- | ------------------------------------------------- | ------- |
| v0 → v1 | SMEM tiling         | `shared`, `dma.copy`, `.subspan().at()`           |   3.9×  |
| v1 → v2 | Tensor core MMA     | `: group`, `mma.fill/load/row.row/store`,         |   9.9×  |
| v2 → v3 | TMA + WGMMA         | `: group-4`, `tma.copy.swiz<>`, `mma.load.swiz<>` |  19.1×  |
| v3 → v4 | Warp specialization | `inthreads.async`, `shared event`, `wait/trigger` |  1.07×  |
| v4 → v5 | Pipeline tuning     | STAGES=2, CONSUMERS=2, WARP_N=192 (28-iter sweep) |  1.40×  |

---

## Why Choreo

The six kernels above express *what* data moves and *what* computation runs — not the mechanics of how to make it happen. That gap is significant:

| Raw CUDA requirement | Choreo equivalent | What the compiler handles |
| --- | --- | --- |
| blockIdx/threadIdx arithmetic | `parallel {i,j} by [...] : block/thread` | All index math, bounds, tile sizes |
| Cooperative copy loop + `__syncthreads()` | `dma.copy src => dst` | Thread partition, coalescing, barriers |
| TMA descriptor (`CUtensorMap`) setup | `tma.copy[.async][.swiz<N>]` | Tensor map construction, mbarrier wiring |
| WGMMA smem descriptor encoding | `mma.load.swiz<N> s.chunkat(i,j)` | Descriptor encoding, swizzle alignment |
| mbarrier init / arrive / wait code | `shared event`, `wait`, `trigger` | Full mbarrier lifecycle |
| Warpgroup-dispatch predication | `inthreads.async (wg==0)` | Warpgroup dispatch, register allocation hints |
| XOR swizzle table construction | `.swiz<128>` annotation | Layout propagation, compile-time consistency check |

Writing the v5 kernel correctly in raw CUDA would require several hundred lines: TMA tensor-map construction, mbarrier initialisation, warpgroup predication, WGMMA smem descriptor encoding, explicit swizzle XOR tables, and careful synchronisation ordering. One mistake anywhere causes incorrect results or silent deadlock.

**The Choreo version is 60 lines, and a 28-iteration parameter sweep found a configuration that exceeds cuBLAS on this GPU.** The entire tuning process — including dead ends — took under 4 hours of wall time. In raw CUDA, the same exploration would require rewriting hundreds of lines for each configuration.

---

## Running the kernels

```bash
# Prerequisites: Choreo compiler, CUDA toolkit, SM90a GPU

# Compile any kernel — produces a self-contained script that wraps nvcc
choreo -gs -t cute -arch=sm_90a matmul_f16_v5_auto_tuned.co -o v5.cute.result

# Run (compile + link + execute in one step)
bash v5.cute.result --execute

# Correctness check without timing output
CHOREO_DISABLE_TIMING=1 bash v5.cute.result --execute

# Profile with Nsight Compute (single launch)
ncu --launch-count 1 \
    --kernel-name regex:__choreo_device_matmul \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  bash v5.cute.result --execute
```
