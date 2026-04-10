# How to Optimize a Matmul Kernel with Croqtile: a Worklog

*April 2026 · GPU: NVIDIA H800 PCIe (SM90a) · Precision: FP16 · Problem: 8192×8192×8192*

---

In this worklog I start from the most naive possible matrix multiplication and step-by-step apply every major GPU optimization technique until the kernel exceeds cuBLAS — all in a high-level DSL called **Croqtile**. The finished kernel is 60 lines. No hand-written CUDA, no PTX intrinsics, no manual thread-index arithmetic.

This is a companion piece to Simon Boehm's excellent [CUDA MMM worklog](https://siboehm.com/articles/22/CUDA-MMM), but the angle is different: instead of showing the low-level details, I want to show how a well-designed kernel DSL can express the same optimizations cleanly — which means you understand *what* you are doing without losing half a week on *how* to say it in code.

!!! tip "Download source code"
    All kernel files from this tutorial are available as a single archive:
    **[matmul_tutorial_kernels.tar.gz](assets/matmul_tutorial_kernels.tar.gz)**

Compile and run any kernel:

```bash
croqtile -gs -t cute -arch=sm_90a kernel.co -o kernel.cute.result
bash kernel.cute.result --execute
```

---

## Performance at a glance

| Kernel | Time (ms) | TFLOPS | % of cuBLAS |
| --- | ---: | ---: | ---: |
| v0: Naive | ~2890 | 0.38 | 0.08% |
| v1: Shared memory | ~728 | 1.51 | 0.34% |
| v2: Hopper TMA + WGMMA | 3.87 | 284.4 | 63.6% |
| v3: Warp specialization | 3.26 | 337.0 | 75.3% |
| **v4: Production-tuned** | **2.35** | **471.3** | **105.3%** |
| cuBLAS (reference) | 2.46 | 447.5 | 100.0% |

1240× improvement from v0 to v4 in five steps. Let's see how each one works.

---

## Kernel 0: Naive

**File:** `matmul_f16_v0_naive.co`

The simplest possible formulation: one thread owns one output element and independently reads a full row of A and column of B from global memory.

```choreo
// TILE_M = 32, TILE_N = 32
// 32×32 = 1024 threads per block.

__co__ void matmul(
    global f16 [M, K] lhs,
    global f16 [N, K] rhs,
    global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, TILE_N)] : block,
           {thr_m, thr_n} by [TILE_M, TILE_N] : thread {
    f16 acc = 0.0f;
    foreach {iv_k} in [K]
      acc += lhs.at(block_m#thr_m, iv_k) * rhs.at(block_n#thr_n, iv_k);

    output.at(block_m#thr_m, block_n#thr_n) = acc;
  }
}
```

**Croqtile concepts introduced here:**

| Construct | Meaning |
| --- | --- |
| `global f16 [M, K] lhs` | A tensor in GPU global memory (HBM). Shape is symbolic; Croqtile infers strides. |
| `parallel {i,j} by [X,Y] : block` | Create a 2-D grid of thread-blocks of size X×Y. `i` and `j` are block-level indices. |
| `parallel {i,j} by [X,Y] : thread` | Create X×Y threads inside each block. Composes with the enclosing block partition. |
| `block_m # thr_m` | The `#` operator composes two parallel indices into one flat index: `block_m * TILE_M + thr_m`. |
| `foreach {iv_k} in [K]` | A plain sequential loop — no parallelism, no reordering. |
| `.at(i, j)` | Element-level accessor into a 2-D tensor. |

![Memory access pattern for v0: four threads redundantly read the same rows and columns](assets/img/v0_memory_access.png#only-dark)
![Memory access pattern for v0: four threads redundantly read the same rows and columns](assets/img/v0_memory_access_light.png#only-light)

### Generated CUDA (v0)

`parallel : block` / `: thread` become `blockIdx` / `threadIdx` arithmetic. The `#` composition becomes a multiply-add. The whole kernel is just a loop:

```cuda
__global__ void matmul_kernel(f16* lhs, f16* rhs, f16* output,
                               unsigned K, unsigned M, unsigned N) {
  int thr_m = threadIdx.x / 32, thr_n = threadIdx.x % 32;

  f16 acc = 0.0f;
  for (int k = 0; k < K; ++k)
    acc = acc + lhs[(blockIdx.x * 32 + thr_m) * K + k]
              * rhs[(blockIdx.y * 32 + thr_n) * K + k];

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

```choreo
// TILE_M = TILE_N = 32, TILE_K = 128

__co__ void matmul(
    global f16 [M, K] lhs,
    global f16 [N, K] rhs,
    global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, TILE_N)] : block,
           {thr_m, thr_n} by [TILE_M, TILE_N] : thread {
    shared f16 [TILE_M, TILE_K] lhs_s;
    shared f16 [TILE_N, TILE_K] rhs_s;
    f16 acc = 0.0f;

    foreach {iv_k} in [cdiv(K, TILE_K)] {
      dma.copy lhs.subspan(TILE_M, TILE_K)
        .at(block_m, iv_k) => lhs_s;
      dma.copy rhs.subspan(TILE_N, TILE_K)
        .at(block_n, iv_k) => rhs_s;

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
| `dma.copy src => dst` | Cooperative DMA: all threads in the block collectively transfer `src` to `dst`. Croqtile partitions the transfer across threads, generates coalesced loads, and inserts `__syncthreads()`. One line replaces ~20 lines of manual CUDA. |
| `.subspan(TILE_M, TILE_K).at(i, j)` | Selects the tile of shape `[TILE_M, TILE_K]` at grid position `(i, j)` within the tensor. |

![Tile reuse pattern: K dimension is sliced into steps; each tile is loaded once and reused by all threads in the block](assets/img/v1_tile_reuse.png#only-dark)
![Tile reuse pattern: K dimension is sliced into steps; each tile is loaded once and reused by all threads in the block](assets/img/v1_tile_reuse_light.png#only-light)

### Generated CUDA (v1) — what `dma.copy` expands to

One line of Croqtile expands to ~25 lines of CUTE layout construction, thread-partitioned copy, and synchronisation. The essential pattern:

```cuda
// dma.copy ... .at(block_m, iv_k) => lhs_s  →

auto gmem_tile = make_tensor(
    make_gmem_ptr(lhs + ...), make_layout(...));
auto smem_tile = make_tensor(
    make_smem_ptr(lhs_s),     make_layout(...));
auto tiled_copy = make_tiled_copy(
    Copy_Atom<...>{}, thread_layout, val_layout);

cute::copy(tiled_copy,
    thr_copy.partition_S(gmem_tile),  // src slice
    thr_copy.partition_D(smem_tile)); // dst slice
__syncthreads();
```

Croqtile generates this for both `lhs` and `rhs` — two lines become ~50 lines of CUTE.

Scalar FMA units still do the arithmetic — the tensor cores remain idle. The bottleneck has shifted from redundant global loads to the raw compute throughput of scalar FMAs, which are orders of magnitude slower than tensor cores.

**Result: ~1.51 TFLOPS (3.9× over v0)**

---

## Kernel 2: Hopper TMA + WGMMA

**File:** `matmul_f16_v2_hopper_tma_wgmma.co`

The largest single jump: **~188× over v1**. Two hardware features specific to Hopper (SM90a) replace both bottlenecks at once:

1. **TMA** (Tensor Memory Accelerator) replaces `dma.copy` — data movement becomes a single-thread hardware operation instead of a cooperative thread-wide copy.
2. **WGMMA** (Warpgroup MMA) replaces scalar FMA — the compute tile is 64×128×16, executing on dedicated tensor core hardware at throughput orders of magnitude beyond scalar arithmetic.

```choreo
// WARP_M=64, WARP_N=128, WARP_K=16, TILE_K=64, SWIZ=128

__co__ void matmul(
    global f16 [M, K] lhs,
    global f16 [N, K] rhs,
    global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, WARP_M), cdiv(N, WARP_N)] : block,
      by 1 : group-4 {
    shared f16 [WARP_M, TILE_K] lhs_s;
    shared f16 [WARP_N, TILE_K] rhs_s;
    shared f16 [WARP_M, WARP_N] output_s;

    mc = mma.fill.f16 0.0f;
    foreach {iv_k} in [cdiv(K, TILE_K)] {
      tma.copy.swiz<SWIZ> lhs.chunkat(block_m, iv_k) => lhs_s;
      tma.copy.swiz<SWIZ> rhs.chunkat(block_n, iv_k) => rhs_s;

      foreach {iv_wk} in [cdiv(TILE_K, WARP_K)] {
        ma = mma.load.swiz<SWIZ> lhs_s.chunkat(_, iv_wk);
        mb = mma.load.swiz<SWIZ> rhs_s.chunkat(_, iv_wk);
        mma.row.row mc, ma, mb;
      }
    }
    mma.store mc, output_s;
    tma.copy output_s =>
      output.subspan(WARP_M, WARP_N)
        .at(block_m, block_n);
  }
}
```

**New concepts:**

| Construct | Meaning |
| --- | --- |
| `by 1 : group-4` | One **warpgroup** (4 warps = 128 threads) per block. `group-4` is the Hopper WGMMA execution granularity. |
| `tma.copy.swiz<128> src => dst` | Hopper Tensor Memory Accelerator: one thread issues a hardware bulk-copy. `.swiz<128>` sets 128-byte XOR swizzle in the destination layout — required for bank-conflict-free WGMMA loads. |
| `mma.load.swiz<128>` | Load a WGMMA A/B fragment from swizzled shared memory. The `.swiz<128>` on `mma.load` is the **source of truth** — Croqtile propagates it to `tma.copy` so the DMA engine writes the layout that WGMMA expects. |
| `mma.fill.f16 0.0f` | Accumulator in f16 precision. WGMMA can accumulate into f16 or f32; f16 uses less register space. |
| `.chunkat(_, iv_wk)` | The underscore fills in the single warpgroup's dimension automatically. |

### How Croqtile selects the hardware MMA instruction

The key decision is made from the shape that `.chunkat()` infers at the `: group` / `: group-4` boundary:

| Fragment shape (inferred) | Hardware instruction | Thread granularity |
| --- | --- | --- |
| 16 × 16 × 16 (f16, `: group`) | `wmma::mma_sync` | 1 warp (32 threads) |
| 16 × 8 × 16 (f16, `: group`) | `mma.sync` | 1 warp (32 threads) |
| 64 × N × 16 (f16, `: group-4`) | `wgmma.mma_async` | 1 warpgroup (128 threads) |

You never name the instruction. You declare the parallelism level and write the tile shapes; Croqtile maps them to hardware.

### TMA vs `dma.copy`

|   | `dma.copy` (v1) | `tma.copy` (v2+) |
| --- | --- | --- |
| **Who moves data** | ALL threads participate (1024 threads in lockstep) | ONE thread issues; hardware does the rest (128 threads free to compute) |
| **Addressing** | Thread computes element index, issues load, writes to smem, repeats per element | Thread writes coordinates into a `CUtensorMap` descriptor; TMA DMA engine handles addressing, coalescing, swizzle |
| **Sync** | `__syncthreads()` — blocks entire block | `mbarrier` — lightweight, per-warpgroup; other warpgroups unaffected |
| **Bandwidth** | Limited by register pressure and instruction throughput | Near theoretical HBM peak (hardware-optimized DMA path) |

### Why swizzle matters

WGMMA reads shared memory across 128 threads simultaneously. Without swizzle, all 128 threads in a warpgroup map to the same 32-bit bank for a given row — a 32-way bank conflict that serialises every load.

The `.swiz<128>` annotation on `mma.load` declares the swizzle layout that WGMMA requires. Croqtile propagates this requirement **backwards** to `tma.copy`, so the TMA engine writes data into smem in the swizzled format that WGMMA expects. The compiler checks consistency at compile time — no manual XOR tables or descriptor bit fields in user code.

### Generated CUDA (v2) — TMA and WGMMA

The two Croqtile lines below map to very different hardware paths. Here is the representative output, condensed:

```cuda
// ── tma.copy.swiz<128> ... => lhs_s ─────────────
//    Kernel receives CUtensorMap descriptors as
//    __grid_constant__ args. Only thread 0 issues
//    the hardware copy; others arrive at the barrier.

if (threadIdx.x == 0) {
  cde::cp_async_bulk_tensor_2d_global_to_shared(
      lhs_s, &tma_lhs,
      iv_k * 64, blockIdx.x * 64, barrier);
  cuda::device::barrier_arrive_tx(
      barrier, 1, /*bytes=*/8192);
} else { barrier.arrive(); }
barrier.wait(barrier.arrive());

// ── mma.row.row mc, ma, mb  (: group-4 → WGMMA) ─
//    Builds 64-bit smem descriptors encoding swizzle
//    + bank layout, then issues async 64×128×16 MMA.

uint64_t desc_a = wgmma_make_smem_desc<Swizzle::B128>(
    lhs_s + iv_wk * 16);
uint64_t desc_b = wgmma_make_smem_desc<Swizzle::B128>(
    rhs_s + iv_wk * 16);
warpgroup_arrive();
SM90::GMMA::MMA_64x128x16_F16F16F16_SS::fma(
    desc_a, desc_b, mc[0..31]);
warpgroup_commit_batch();
warpgroup_wait<0>();
```

### NCU snapshot (v2)

```
dram__throughput (% of peak HBM BW) :  10.64%
sm__throughput   (% of peak SM)     :  42.83%
pipe_tensor instructions            :    264,192
pipe_fma  instructions              :    247,820
```

SM utilisation jumped from 6% to 43%. The kernel is past the compute roofline crossover for this problem (~70 FLOPs/byte). But 57% of SM peak is missing — because TMA and WGMMA are serialised: each K-step waits for the copy to finish before computing.

**Result: ~284 TFLOPS (188× over v1, 63.6% of cuBLAS)**

---

## Kernel 3: Warp Specialization

**File:** `matmul_f16_v3_warpspec.co`

The insight from v2's profile: producer (TMA) and consumer (WGMMA) stall each other. Fix this by giving them separate warpgroups that run concurrently, connected by a software ring buffer in shared memory.

```choreo
// WARP_M=64, WARP_N=128, WARP_K=16
// TILE_M=128, TILE_K=64, STAGES=1, CONSUMERS=1

__co__ void matmul(
    global f16 [M, K] lhs,
    global f16 [N, K] rhs,
    global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, WARP_N)] : block {
    shared event full[STAGES], empty[STAGES];
    shared f16 [TILE_M, TILE_K] lhs_s[STAGES];
    shared f16 [WARP_N, TILE_K] rhs_s[STAGES];
    shared f16 [WARP_M, WARP_N] output_s[CONSUMERS];

    // 3 warpgroups × 128 threads = 384 threads/block
    parallel wg by 3 : group-4, t by 128 : thread {
      // ── Producer (wg=0): one thread drives TMA ──
      inthreads.async (wg == 0 && t == 0) {
        foreach {iv_k} in [cdiv(K, TILE_K)] {
          stage = iv_k % STAGES;
          wait empty[stage];
          tma.copy.async<full[stage]>.swiz<SWIZ>
            lhs.subspan(TILE_M, TILE_K).at(block_m, iv_k) => lhs_s[stage];
          tma.copy.async<full[stage]>.swiz<SWIZ>
            rhs.subspan(WARP_N, TILE_K).at(block_n, iv_k) => rhs_s[stage];
          trigger full[stage];
        }
      }

      // ── Consumers (wg>=1): compute WGMMA ──
      inthreads.async (wg >= 1) {
        cidx = wg - 1;
        foreach {s} in [STAGES]
          trigger empty[s];

        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, TILE_K)] {
          stage = iv_k % STAGES;
          wait full[stage];

          foreach {iv_wk} in [cdiv(TILE_K, WARP_K)] {
            ma = mma.load.swiz<SWIZ> 
                  lhs_s[stage].subspan(WARP_M, WARP_K).at(cidx, iv_wk);
            mb = mma.load.swiz<SWIZ> rhs_s[stage].chunkat(_, iv_wk);
            mma.row.row mc, ma, mb;
          }
          trigger empty[stage];
        }

        mma.store mc, output_s[cidx];
        tma.copy output_s[cidx] =>
          output.subspan(WARP_M, WARP_N).at(block_m * CONSUMERS + cidx, block_n);
      }

    }
  }
}
```

**New concepts:**

| Construct | Meaning |
| --- | --- |
| `shared event full[N], empty[N]` | Named pipeline barriers backed by Hopper `mbarrier`. Individual warpgroups can wait on or signal them independently. Croqtile generates the mbarrier init, arrive, and wait sequences. |
| `inthreads.async (condition) { ... }` | Executes the block only in warpgroups matching `condition`. This is Croqtile's syntax for Hopper *warp specialization*: warpgroups execute different code paths simultaneously. |
| `tma.copy.async<full[stage]> src=>dst` | Asynchronous TMA that fires the event `full[stage]` upon hardware completion. The issuing thread is not blocked. |
| `trigger event` / `wait event` | Signal or wait on a named barrier. Maps to `mbarrier.arrive` / `mbarrier.try_wait`. |


### Generated CUDA (v3) — warpgroup dispatch

`shared event` becomes `cuda::barrier` arrays; `inthreads.async` becomes a plain `if` on warpgroup index. The core dispatch:

```cuda
__shared__ cuda::barrier<cuda::thread_scope_block>
    full[STAGES], empty[STAGES];
int wg = threadIdx.x / 128;

if (wg == 0 && threadIdx.x % 128 == 0) {      // producer
  for (int iv_k = 0; ...) {
    empty[iv_k % STAGES].wait(...);            // wait empty
    cp_async_bulk_tensor_2d_global_to_shared(
        lhs_s, &tma_lhs, ...);
    barrier_arrive_tx(
        full[iv_k % STAGES], 1, bytes);        // trigger full
  }
}
if (wg >= 1) {                                 // consumers
  for (int iv_k = 0; ...) {
    full[iv_k % STAGES].wait(...);             // wait full
    /* WGMMA (same as v2) */
    empty[iv_k % STAGES].arrive();             // trigger empty
  }
}
```

**Result: ~337 TFLOPS (1.19× over v2, 75.3% of cuBLAS)**

---

## Kernel 4: Production-Tuned

**File:** `matmul_f16_v4_auto_tuned.co`

The kernel structure is **identical to v3**. Every line of new functionality was introduced in v3. What changes here are the macro parameters that determine tile shape and pipeline depth:

```choreo
#define WARP_M    64    // rows per consumer warpgroup's output tile
#define WARP_N   192    // cols per warpgroup tile — found by 28-iter sweep
#define WARP_K    16    // WGMMA K step
#define TILE_M   128    // total M per block = 2 × WARP_M  (2 consumers)
#define TILE_K    64    // K-tile loaded per TMA transfer
#define SWIZ     128    // swizzle byte width
#define STAGES     2    // double-buffered ring buffer  (was 1 in v3)
#define CONSUMERS  2    // consumer warpgroups per block (was 1 in v3)
```

The full kernel is identical to v3 except for these defines — same 60 lines of Croqtile.

### What changed from v3 and why it matters

| Parameter | v3 | v4 | Effect |
| --- | ---: | ---: | --- |
| `CONSUMERS` | 1 | 2 | 2 consumer WGs → 2× output M-rows per block |
| `TILE_M` | 64 | 128 | Larger block → better L2 cache reuse for B |
| `STAGES` | 1 | 2 | True double-buffering: producer prefetches tile[i+1] while consumers compute tile[i]. TMA latency is now fully hidden. |
| `WARP_N` | 128 | 192 | Found by 28-iteration parameter sweep; see the tuning journey below. |

**Why WARP_N = 192?** SM90 WGMMA accepts any N in [8, 256] divisible by 8. 192 = 24×8. At this shape the combination of register usage (~80 registers/thread), shared memory footprint per block (~80 KB → 2 blocks per SM), and N-grid parallelism (`cdiv(8192, 192) = 43` blocks) yields the best measured throughput for 8192³ on H800 PCIe. The full sweep is documented in the tuning section below.

### Double-buffering in pictures (STAGES=2)

With two buffer slots the producer can prefetch the next tile while the consumer computes the current one:

![Double-buffering timeline: producer prefetches into the alternate slot while consumers compute from the current slot](assets/img/double_buffering_timeline.png#only-dark)
![Double-buffering timeline: producer prefetches into the alternate slot while consumers compute from the current slot](assets/img/double_buffering_timeline_light.png#only-light)

### NCU snapshot (v4)

```
sm__throughput   (% of peak SM)     :  89.68%   ← near-peak compute
tensor_core HMMA (% of peak)        :  89.68%   ← compute-bound
gpu__dram_throughput (% of HBM BW)  :  38.91%   ← TMA hiding all loads
warp occupancy   (% of peak)        :  27.74%   ← ~2 blocks/SM (smem-limited)
```

SM and tensor-core utilisation are at 89.7% — the kernel is firmly in the compute-bound regime. DRAM at 39% means TMA is successfully prefetching ahead. Warp occupancy at 27.7% reflects the shared memory footprint (~80 KB per block) allowing 2 concurrent blocks per SM on H800.

**Result: ~471 TFLOPS (1.40× over v3, 105% of cuBLAS on this GPU)**

---

## The full NCU picture

Profiled with a single kernel launch (`ncu --launch-count 1`):

| Kernel | dram% | sm% | tensor inst | fma inst | TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 naive | 0.01 | 5.99 | 0 | 336,592,896 | 0.38 |
| v1 smem | 0.02 | 6.12 | 0 | 336,592,896 | 1.51 |
| v2 tma/wgmma | 10.64 | 42.83 | 264,192 | 247,820 | 284.4 |
| v3 warpspec | 32.97 | 56.13 | 9,418,787 | — | 337.0 |
| v4 tuned | 38.91 | 89.68 | 9,492,372 | — | 471.3 |

- **v0→v1**: same scalar FMAs; shared memory fixes redundant reads but compute is unchanged.
- **v2**: tensor core instructions appear; SM jumps from 6% to 43% — but TMA and WGMMA serialise.
- **v3**: warpspec decouples producer/consumer; tensor utilization improves to 56%.
- **v4**: double-buffer + WARP_N=192 → both SM and tensor utilization hit 89.7%, compute-roofline regime.

To reproduce these numbers:

```bash
ncu --target-processes all --launch-count 1 \
    --kernel-name-base demangled \
    --kernel-name regex:__croqtile_device_matmul \
    --metrics \
      sm__throughput.avg.pct_of_peak_sustained_elapsed,\
      dram__throughput.avg.pct_of_peak_sustained_elapsed,\
      smsp__inst_executed_pipe_tensor.sum,\
      smsp__inst_executed_pipe_fma.sum \
    <compiled-kernel-binary>
```

---

## The Tuning Journey: v3 → v4

v4 was not designed upfront — it was found by a systematic 28-iteration parameter sweep starting from v3. Here is the actual log. This is the core of the tutorial: the techniques above give you a structurally correct kernel, but the last 40% comes from tuning.

### Baseline: v3 (STAGES=1, CONSUMERS=1, WARP_N=128)

NCU on the v3 baseline reveals the kernel is **latency-bound**:

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
|------|--------|-------:|-------|
| iter000 | baseline (STAGES=1, CONS=1) | 337 | latency-bound |
| iter001 | STAGES=2, CONS=1 | — | **CRASH** — `parallel wg by 3` spawns 3 warpgroups regardless of `CONSUMERS`; mismatch causes OOB smem write |
| iter002 | STAGES=2, CONS=2, WARP_N=128 | 365 | first working double-buffer (+8%) |
| iter003 | STAGES=2, CONS=2, WARP_N=152 | 402 | intermediate config, bottleneck shifts to compute-bound (+19%) |

The crash on iter001 reveals a subtle Croqtile semantics point: `parallel wg by 3` always spawns 3 warpgroups. The consumer predicate `inthreads.async (wg >= 1)` matches **both** wg=1 and wg=2 — so you must always set `CONSUMERS` to match the actual number of consumer warpgroups.

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

!!! warning "Hardware constraint"
    WGMMA requires `WARP_N` to be a **multiple of 8**. Values like 180 or 188 compile-fail with `MMA m64n180k16 not supported`. Discovered during iter019–020.

![WARP_N sweep results: performance peaks at WARP_N=192, drops beyond 208 due to smem limits](assets/img/warpn_sweep.png#only-dark)
![WARP_N sweep results: performance peaks at WARP_N=192, drops beyond 208 due to smem limits](assets/img/warpn_sweep_light.png#only-light)

The sweet spot is **WARP_N = 176–192**. Going beyond 192 adds shared memory without enough extra work to compensate — WARP_N=224 drops to 364 TFLOPS because the larger smem footprint prevents 2 concurrent blocks per SM.

### Phase 3 — Dead Ends (iter011–018)

Not all directions pay off:

| Attempt | Result | Why |
|---------|--------|-----|
| STAGES=3, WARP_N=192 | correctness fail | Compiler bug with 3-stage pipeline |
| STAGES=3, WARP_N=176 | 373 TFLOPS (worse) | Extra barrier overhead dominates |
| TILE_K=32, WARP_N=192 | 403 TFLOPS (worse) | Shorter K-tiles shrink the overlap window |
| TILE_K=128, WARP_N=192 | CUDA invalid arg | smem 163 KB exceeds kernel limit |
| 1p3c (3 consumers), TILE_M=192 | 156 TFLOPS | 512 threads × 80 regs = only 1 block/SM |
| SWIZ=64, WARP_N=192 | CUDA invalid arg | WGMMA N=192 requires SWIZ=128 layout |
| WARP_N=256 | 396 TFLOPS | Larger smem, fewer blocks/SM |

The 1p3c experiment is instructive: with 512 threads (4 warpgroups) the register budget forces only 1 active block per SM, cutting utilization in half.

### Winner: WARP_N=192 (+40% over v3)

```choreo
#define WARP_N 192
#define STAGES 2
#define CONSUMERS 2
```

That is the **only change** in v4's Croqtile source. The compiler handles the rest — regenerating the correct TMA tensor descriptors, swizzle layouts, and mbarrier counts automatically for the new tile shape.

---

## SOTA comparison

Running cuBLAS via PyTorch's `torch.mm` on the same 8192×8192×8192 f16 problem:

*NVIDIA H800 PCIe — FP16 Tensor Core peak: 1513 TFLOPS*

| Implementation | Time (ms) | TFLOPS | % |
| --- | ---: | ---: | ---: |
| cuBLAS (`torch.mm`) | 2.46 | 447.5 | 100% |
| **Croqtile v4 (tuned)** | **2.35** | **471.3** | **105.3%** |

The slight edge over cuBLAS comes from the WARP_N=192 tile hitting a better L2/SMEM working-set balance on this specific GPU model. Production cuBLAS also includes features outside this tutorial's scope:

- **Thread block clusters**: cuBLAS uses Hopper multicast TMA to share B tiles across blocks in a cluster.
- **Persistent kernels**: cuBLAS keeps blocks alive across output tiles to amortize launch overhead.
- **Epilogue fusion**: cuBLAS merges the MMA store and output write, avoiding one SMEM round-trip.

All three can be expressed in Croqtile — the production-grade kernels in the repository implement them.

---

## The optimization ladder

![Optimization ladder: TFLOPS progression from v0 (0.38) through v4 (471), compared to cuBLAS (447)](assets/img/optimization_ladder.png#only-dark)
![Optimization ladder: TFLOPS progression from v0 (0.38) through v4 (471), compared to cuBLAS (447)](assets/img/optimization_ladder_light.png#only-light)

| Step | Technique | Key Croqtile construct | Speedup |
| --- | --- | --- | ---: |
| v0 → v1 | SMEM tiling | `shared`, `dma.copy`, `.subspan().at()` | 3.9× |
| v1 → v2 | TMA + WGMMA | `: group-4`, `tma.copy.swiz<>`, `mma.load.swiz<>` | 188× |
| v2 → v3 | Warp specialization | `inthreads.async`, `shared event`, `wait/trigger` | 1.19× |
| v3 → v4 | Pipeline tuning | STAGES=2, CONSUMERS=2, WARP_N=192 (28-iter sweep) | 1.40× |

---

## Why Croqtile

The five kernels above express *what* data moves and *what* computation runs — not the mechanics of how to make it happen. That gap is significant:

| Raw CUDA requirement | Croqtile equivalent | What the compiler handles |
| --- | --- | --- |
| blockIdx/threadIdx arithmetic | `parallel {i,j} by [...] : block/thread` | All index math, bounds, tile sizes |
| Cooperative copy loop + `__syncthreads()` | `dma.copy src => dst` | Thread partition, coalescing, barriers |
| TMA descriptor (`CUtensorMap`) setup | `tma.copy[.async][.swiz<N>]` | Tensor map construction, mbarrier wiring |
| WGMMA smem descriptor encoding | `mma.load.swiz<N> s.chunkat(i,j)` | Descriptor encoding, swizzle alignment |
| mbarrier init / arrive / wait code | `shared event`, `wait`, `trigger` | Full mbarrier lifecycle |
| Warpgroup-dispatch predication | `inthreads.async (wg==0)` | Warpgroup dispatch, register allocation hints |
| XOR swizzle table construction | `.swiz<128>` on `mma.load` | Backward propagation to `tma.copy`, compile-time consistency check |

Writing the v4 kernel correctly in raw CUDA would require several hundred lines: TMA tensor-map construction, mbarrier initialisation, warpgroup predication, WGMMA smem descriptor encoding, explicit swizzle XOR tables, and careful synchronisation ordering. One mistake anywhere causes incorrect results or silent deadlock.

**The Croqtile version is 60 lines, and a 28-iteration parameter sweep found a configuration that exceeds cuBLAS on this GPU.** The entire tuning process — including dead ends — took under 4 hours of wall time. In raw CUDA, the same exploration would require rewriting hundreds of lines for each configuration.

---

## Running the kernels

```bash
# Compile — produces a self-contained script wrapping nvcc
croqtile -gs -t cute -arch=sm_90a \
    matmul_f16_v4_auto_tuned.co -o v4.cute.result

# Run (compile + link + execute in one step)
bash v4.cute.result --execute

# Correctness check without timing output
CHOREO_DISABLE_TIMING=1 bash v4.cute.result --execute

# Profile with Nsight Compute (single launch)
ncu --launch-count 1 \
    --kernel-name regex:__croqtile_device_matmul \
    --metrics \
      sm__throughput.avg.pct_of_peak_sustained_elapsed,\
      dram__throughput.avg.pct_of_peak_sustained_elapsed \
    bash v4.cute.result --execute
```
