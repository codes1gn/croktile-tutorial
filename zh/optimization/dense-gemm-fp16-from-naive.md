# 如何用 Croqtile 一步步优化矩阵乘法内核

*2026 年 4 月 · GPU: NVIDIA H800 PCIe (SM90a) · 精度: FP16 · 问题规模: 8192×8192×8192*

---

在这篇教程中，我从最简单的矩阵乘法开始，逐步应用 GPU 上的每一项主要优化技术，直到内核性能超越 cuBLAS —— 全部使用一种名为 **Croqtile** 的高层 DSL 完成。最终的内核仅 60 行代码，无需手写 CUDA，无需 PTX 内联汇编，也无需手动计算线程索引。

本教程参考了 Simon Boehm 出色的 [CUDA MMM 系列文章](https://siboehm.com/articles/22/CUDA-MMM)，但角度不同：我不展示底层细节，而是展示一个精心设计的内核 DSL 如何清晰地表达同样的优化 —— 这意味着你能理解自己在*做什么*，而无需在*如何编码*上耗费大量时间。

!!! tip "下载源代码"
    本教程所有内核文件打包为单一压缩包：
    **[matmul_tutorial_kernels.tar.gz](assets/matmul_tutorial_kernels.tar.gz)**

编译和运行任意内核：

```bash
croqtile -gs -t cute -arch=sm_90a kernel.co -o kernel.cute.result
bash kernel.cute.result --execute
```

---

## 性能一览

| 内核 | 时间 (ms) | TFLOPS | 相对 cuBLAS |
| --- | ---: | ---: | ---: |
| v0: 朴素版 | ~2890 | 0.38 | 0.08% |
| v1: 共享内存 | ~728 | 1.51 | 0.34% |
| v2: Hopper TMA + WGMMA | 3.87 | 284.4 | 63.6% |
| v3: Warp 特化 | 3.26 | 337.0 | 75.3% |
| **v4: 生产级调优** | **2.35** | **471.3** | **105.3%** |
| cuBLAS（参考基线） | 2.46 | 447.5 | 100.0% |

从 v0 到 v4，五步实现 1240 倍性能提升。下面逐一讲解每一步的原理。

---

## 内核 0：朴素版

**文件：** `matmul_f16_v0_naive.co`

最简单的实现方式：每个线程负责一个输出元素，独立从全局内存读取 A 的一整行和 B 的一整列。

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

**此处引入的 Croqtile 概念：**

| 构造 | 含义 |
| --- | --- |
| `global f16 [M, K] lhs` | GPU 全局内存 (HBM) 中的张量。形状为符号式，Croqtile 自动推断步长。 |
| `parallel {i,j} by [X,Y] : block` | 创建 X×Y 大小的二维线程块网格。`i`、`j` 是块级索引。 |
| `parallel {i,j} by [X,Y] : thread` | 在每个块内创建 X×Y 个线程，与外层的块级划分组合。 |
| `block_m # thr_m` | `#` 运算符将两个并行索引组合为一个扁平索引：`block_m * TILE_M + thr_m`。 |
| `foreach {iv_k} in [K]` | 普通顺序循环 —— 无并行，无重排。 |
| `.at(i, j)` | 二维张量的元素级访问器。 |

![v0 的内存访问模式：四个线程冗余地读取相同的行和列](assets/img/v0_memory_access.png#only-dark)
![v0 的内存访问模式：四个线程冗余地读取相同的行和列](assets/img/v0_memory_access_light.png#only-light)

### 生成的 CUDA 代码 (v0)

`parallel : block` / `: thread` 变成 `blockIdx` / `threadIdx` 运算。`#` 组合变成乘加运算。整个内核就是一个循环：

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

### NCU 快照 (v0)

```
dram__throughput (% of peak HBM BW) :   0.01%
sm__throughput   (% of peak SM)     :   5.99%
pipe_tensor instructions            :   0          ← Tensor Core 完全空闲
pipe_fma  instructions              :   336,592,896
```

由于访问模式极度分散，硬件无法合并 HBM 请求，HBM 吞吐量几乎为零——尽管每个线程都在不断读取数据。

**结果：~0.38 TFLOPS（cuBLAS 的 0.08%）**

---

## 内核 1：共享内存分块

**文件：** `matmul_f16_v1_shared_memory.co`

经典解决方案：将 A 和 B 的一个分块加载到快速的片上共享内存 (SRAM)，让块内所有线程复用该分块，然后沿 K 维度滑动分块窗口。

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

**新概念：**

| 构造 | 含义 |
| --- | --- |
| `shared f16 [M, K] buf` | 在片上共享内存中分配缓冲区。作用域为线程块；所有线程均可读写。 |
| `dma.copy src => dst` | 协作式 DMA：块内所有线程协同将 `src` 传输到 `dst`。Croqtile 自动在线程间分配传输任务、生成合并访问、并插入 `__syncthreads()`。一行代码替代约 20 行手写 CUDA。 |
| `.subspan(TILE_M, TILE_K).at(i, j)` | 在张量内选取形状为 `[TILE_M, TILE_K]`、网格位置为 `(i, j)` 的分块。 |

![分块复用模式：K 维度被切分为多步；每个分块加载一次，被块内所有线程复用](assets/img/v1_tile_reuse.png#only-dark)
![分块复用模式：K 维度被切分为多步；每个分块加载一次，被块内所有线程复用](assets/img/v1_tile_reuse_light.png#only-light)

### 生成的 CUDA 代码 (v1) —— `dma.copy` 展开为什么

一行 Croqtile 展开为约 25 行 CUTE 布局构造、线程分区拷贝和同步代码。核心模式如下：

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

Croqtile 为 `lhs` 和 `rhs` 各生成一份 —— 两行代码变成约 50 行 CUTE。

标量 FMA 单元仍在执行运算 —— Tensor Core 保持空闲。瓶颈已从冗余的全局加载转移到标量 FMA 的原始计算吞吐量，后者比 Tensor Core 慢几个数量级。

**结果：~1.51 TFLOPS（相比 v0 提升 3.9 倍）**

---

## 内核 2：Hopper TMA + WGMMA

**文件：** `matmul_f16_v2_hopper_tma_wgmma.co`

最大的单次跳跃：**相比 v1 提升约 188 倍**。Hopper (SM90a) 特有的两项硬件特性同时解决了两个瓶颈：

1. **TMA**（张量内存加速器）取代 `dma.copy` —— 数据搬运变成单线程发出的硬件操作，而非全体线程的协作拷贝。
2. **WGMMA**（线程组矩阵乘累加）取代标量 FMA —— 计算分块为 64×128×16，在专用 Tensor Core 硬件上执行，吞吐量远超标量运算。

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

**新概念：**

| 构造 | 含义 |
| --- | --- |
| `by 1 : group-4` | 每个块分配一个 **warpgroup**（4 个 warp = 128 个线程）。`group-4` 是 Hopper WGMMA 的执行粒度。 |
| `tma.copy.swiz<128> src => dst` | Hopper 张量内存加速器：单线程发出硬件批量拷贝。`.swiz<128>` 设置目标布局的 128 字节 XOR swizzle —— WGMMA 加载所需的无 bank 冲突布局。 |
| `mma.load.swiz<128>` | 从 swizzle 布局的共享内存加载 WGMMA A/B 片段。`mma.load` 上的 `.swiz<128>` 是**基准声明** —— Croqtile 将其反向传播到 `tma.copy`，使 DMA 引擎按 WGMMA 期望的布局写入数据。 |
| `mma.fill.f16 0.0f` | f16 精度累加器。WGMMA 可在 f16 或 f32 中累加；f16 使用更少的寄存器。 |
| `.chunkat(_, iv_wk)` | 下划线自动填充单个 warpgroup 的维度。 |

### Croqtile 如何选择硬件 MMA 指令

关键决策来自 `.chunkat()` 在 `: group` / `: group-4` 边界推断出的形状：

| 推断的片段形状 | 硬件指令 | 线程粒度 |
| --- | --- | --- |
| 16 × 16 × 16 (f16, `: group`) | `wmma::mma_sync` | 1 个 warp（32 线程） |
| 16 × 8 × 16 (f16, `: group`) | `mma.sync` | 1 个 warp（32 线程） |
| 64 × N × 16 (f16, `: group-4`) | `wgmma.mma_async` | 1 个 warpgroup（128 线程） |

你无需指定指令名称。声明并行层级和分块形状，Croqtile 自动映射到硬件。

### TMA 与 `dma.copy` 对比

|   | `dma.copy` (v1) | `tma.copy` (v2+) |
| --- | --- | --- |
| **谁搬运数据** | 所有线程参与（1024 个线程同步执行） | 单个线程发出；硬件完成其余工作（128 个线程可自由计算） |
| **寻址方式** | 线程计算元素索引、发出加载、写入 smem，逐元素重复 | 线程将坐标写入 `CUtensorMap` 描述符；TMA DMA 引擎处理寻址、合并和 swizzle |
| **同步机制** | `__syncthreads()` —— 阻塞整个线程块 | `mbarrier` —— 轻量级、按 warpgroup 独立；其他 warpgroup 不受影响 |
| **带宽** | 受限于寄存器压力和指令吞吐 | 接近理论 HBM 峰值（硬件优化的 DMA 路径） |

### 为什么 swizzle 重要

WGMMA 同时通过 128 个线程读取共享内存。如果不使用 swizzle，同一 warpgroup 中 128 个线程对于给定行会映射到相同的 32 位 bank —— 产生 32 路 bank 冲突，将每次加载串行化。

`mma.load` 上的 `.swiz<128>` 声明了 WGMMA 所需的 swizzle 布局。Croqtile 将此需求**反向传播**到 `tma.copy`，使 TMA 引擎以 WGMMA 期望的 swizzle 格式写入 smem 数据。编译器在编译时检查一致性 —— 用户代码中无需手动编写 XOR 表或描述符位字段。

### 生成的 CUDA 代码 (v2) —— TMA 和 WGMMA

以下两行 Croqtile 代码映射到截然不同的硬件路径。精简后的代表性输出：

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

### NCU 快照 (v2)

```
dram__throughput (% of peak HBM BW) :  10.64%
sm__throughput   (% of peak SM)     :  42.83%
pipe_tensor instructions            :    264,192
pipe_fma  instructions              :    247,820
```

SM 利用率从 6% 跃升到 43%。该内核已越过此问题规模的计算屋顶线交叉点（约 70 FLOPs/byte）。但仍有 57% 的 SM 峰值缺失 —— 因为 TMA 和 WGMMA 是串行的：每个 K 步都要等待拷贝完成后才能计算。

**结果：~284 TFLOPS（相比 v1 提升 188 倍，cuBLAS 的 63.6%）**

---

## 内核 3：Warp 特化

**文件：** `matmul_f16_v3_warpspec.co`

v2 的性能剖析揭示了问题：生产者 (TMA) 和消费者 (WGMMA) 互相阻塞。解决方案是让它们在不同的 warpgroup 中并发运行，通过共享内存中的软件环形缓冲区连接。

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
      // ── 生产者 (wg=0)：单线程驱动 TMA ──
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

      // ── 消费者 (wg>=1)：执行 WGMMA 计算 ──
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

**新概念：**

| 构造 | 含义 |
| --- | --- |
| `shared event full[N], empty[N]` | 由 Hopper `mbarrier` 支持的命名管道屏障。各 warpgroup 可独立等待或发信号。Croqtile 自动生成 mbarrier 的初始化、到达和等待序列。 |
| `inthreads.async (condition) { ... }` | 仅在满足 `condition` 的 warpgroup 中执行该代码块。这是 Croqtile 的 Hopper *warp 特化*语法：warpgroup 同时执行不同的代码路径。 |
| `tma.copy.async<full[stage]> src=>dst` | 异步 TMA，在硬件完成时触发事件 `full[stage]`。发出指令的线程不会被阻塞。 |
| `trigger event` / `wait event` | 信号或等待命名屏障。分别映射到 `mbarrier.arrive` / `mbarrier.try_wait`。 |


### 生成的 CUDA 代码 (v3) —— warpgroup 调度

`shared event` 变成 `cuda::barrier` 数组；`inthreads.async` 变成基于 warpgroup 索引的 `if` 判断。核心调度逻辑：

```cuda
__shared__ cuda::barrier<cuda::thread_scope_block>
    full[STAGES], empty[STAGES];
int wg = threadIdx.x / 128;

if (wg == 0 && threadIdx.x % 128 == 0) {      // 生产者
  for (int iv_k = 0; ...) {
    empty[iv_k % STAGES].wait(...);            // wait empty
    cp_async_bulk_tensor_2d_global_to_shared(
        lhs_s, &tma_lhs, ...);
    barrier_arrive_tx(
        full[iv_k % STAGES], 1, bytes);        // trigger full
  }
}
if (wg >= 1) {                                 // 消费者
  for (int iv_k = 0; ...) {
    full[iv_k % STAGES].wait(...);             // wait full
    /* WGMMA（同 v2） */
    empty[iv_k % STAGES].arrive();             // trigger empty
  }
}
```

**结果：~337 TFLOPS（相比 v2 提升 1.19 倍，cuBLAS 的 75.3%）**

---

## 内核 4：生产级调优

**文件：** `matmul_f16_v4_auto_tuned.co`

内核结构与 v3 **完全相同**。v3 引入了所有新功能。这里变化的只是决定分块形状和管道深度的宏参数：

```choreo
#define WARP_M    64    // 每个消费者 warpgroup 的输出分块行数
#define WARP_N   192    // 每个 warpgroup 的列分块 — 28 轮搜索得出
#define WARP_K    16    // WGMMA K 步长
#define TILE_M   128    // 每个块的总 M = 2 × WARP_M（2 个消费者）
#define TILE_K    64    // 每次 TMA 传输的 K 分块
#define SWIZ     128    // swizzle 字节宽度
#define STAGES     2    // 双缓冲环形缓冲区（v3 中为 1）
#define CONSUMERS  2    // 每个块的消费者 warpgroup 数（v3 中为 1）
```

完整内核与 v3 相同，仅更改这些宏定义 —— 同样 60 行 Croqtile。

### v3 到 v4 的变化及影响

| 参数 | v3 | v4 | 效果 |
| --- | ---: | ---: | --- |
| `CONSUMERS` | 1 | 2 | 2 个消费者 WG → 每个块输出 2 倍的 M 行 |
| `TILE_M` | 64 | 128 | 更大的块 → B 矩阵更好的 L2 缓存复用 |
| `STAGES` | 1 | 2 | 真正的双缓冲：生产者预取 tile[i+1]，消费者同时计算 tile[i]。TMA 延迟完全隐藏。 |
| `WARP_N` | 128 | 192 | 通过 28 轮参数搜索找到；参见下方调优历程。 |

**为什么 WARP_N = 192？** SM90 WGMMA 接受 [8, 256] 范围内 8 的倍数作为 N。192 = 24×8。在此形状下，寄存器使用量（约 80 个寄存器/线程）、每块共享内存占用（约 80 KB → 每 SM 2 个块）和 N 维网格并行度（`cdiv(8192, 192) = 43` 个块）的组合在 H800 PCIe 上产生了 8192³ 问题的最佳实测吞吐量。完整搜索过程记录在下方的调优章节中。

### 双缓冲图解 (STAGES=2)

使用两个缓冲槽位后，生产者可以在消费者计算当前分块时预取下一个分块：

![双缓冲时间线：生产者在交替槽位预取，消费者同时从当前槽位计算](assets/img/double_buffering_timeline.png#only-dark)
![双缓冲时间线：生产者在交替槽位预取，消费者同时从当前槽位计算](assets/img/double_buffering_timeline_light.png#only-light)

### NCU 快照 (v4)

```
sm__throughput   (% of peak SM)     :  89.68%   ← 接近峰值计算
tensor_core HMMA (% of peak)        :  89.68%   ← 计算受限
gpu__dram_throughput (% of HBM BW)  :  38.91%   ← TMA 隐藏了所有加载
warp occupancy   (% of peak)        :  27.74%   ← 约 2 块/SM（受 smem 限制）
```

SM 和 Tensor Core 利用率达到 89.7% —— 内核已处于计算受限区域。DRAM 为 39% 意味着 TMA 成功进行了预取。Warp 占用率 27.7% 反映了共享内存占用（每块约 80 KB），使得 H800 上每 SM 只能容纳 2 个并发块。

**结果：~471 TFLOPS（相比 v3 提升 1.40 倍，cuBLAS 的 105%）**

---

## 完整 NCU 数据

使用单次内核启动进行分析（`ncu --launch-count 1`）：

| 内核 | dram% | sm% | tensor 指令数 | fma 指令数 | TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 朴素版 | 0.01 | 5.99 | 0 | 336,592,896 | 0.38 |
| v1 共享内存 | 0.02 | 6.12 | 0 | 336,592,896 | 1.51 |
| v2 tma/wgmma | 10.64 | 42.83 | 264,192 | 247,820 | 284.4 |
| v3 warpspec | 32.97 | 56.13 | 9,418,787 | — | 337.0 |
| v4 调优版 | 38.91 | 89.68 | 9,492,372 | — | 471.3 |

- **v0→v1**：相同的标量 FMA；共享内存消除了冗余读取但计算未变。
- **v2**：Tensor Core 指令出现；SM 从 6% 跃升到 43% —— 但 TMA 和 WGMMA 串行执行。
- **v3**：warp 特化解耦生产者/消费者；Tensor 利用率提升至 56%。
- **v4**：双缓冲 + WARP_N=192 → SM 和 Tensor 利用率均达 89.7%，计算屋顶线区域。

复现这些数据：

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

## 调优历程：v3 → v4

v4 不是预先设计的 —— 它是从 v3 开始，通过系统化的 28 轮参数搜索找到的。以下是实际记录。这是本教程的核心：上述技术为你提供了结构正确的内核，但最后 40% 的性能提升来自调优。

### 基线：v3 (STAGES=1, CONSUMERS=1, WARP_N=128)

v3 基线的 NCU 分析显示内核处于**延迟受限**状态：

```
sm__throughput:          56%   ← SM 利用不足
tensor_core (HMMA):      52%   ← 计算仅半数活跃
gpu__dram_throughput:    33%   ← DRAM 不是问题
warp occupancy:          28%   ← 大量 warp 在屏障上阻塞
```

单级管道（`STAGES=1`）迫使生产者在消费者消耗完数据前等待，才能加载下一个 K 分块。CPU 可见的表现：`sm__pipe_tensor_op_hmma_cycles` 每 `TILE_K/WARP_K = 4` 条 WGMMA 指令就会暂停。

### 阶段 1 —— 管道架构 (iter000 → iter003)

第一个结构性变化是启用双缓冲和第二个消费者 warpgroup：

| 迭代 | 变更 | TFLOPS | 备注 |
|------|--------|-------:|-------|
| iter000 | 基线 (STAGES=1, CONS=1) | 337 | 延迟受限 |
| iter001 | STAGES=2, CONS=1 | — | **崩溃** — `parallel wg by 3` 始终生成 3 个 warpgroup，不受 `CONSUMERS` 影响；不匹配导致 smem 越界写入 |
| iter002 | STAGES=2, CONS=2, WARP_N=128 | 365 | 首个可用的双缓冲版本（+8%） |
| iter003 | STAGES=2, CONS=2, WARP_N=152 | 402 | 中间配置，瓶颈转为计算受限（+19%） |

iter001 的崩溃揭示了一个微妙的 Croqtile 语义点：`parallel wg by 3` 始终生成 3 个 warpgroup。消费者谓词 `inthreads.async (wg >= 1)` 同时匹配 wg=1 和 wg=2 —— 因此必须始终设置 `CONSUMERS` 与实际消费者 warpgroup 数量一致。

iter003 之后 NCU 显示瓶颈已转移：

```
sm__throughput:          89%   ← 接近峰值！
tensor_core (HMMA):      89%   ← 计算受限
gpu__dram_throughput:    38%   ← TMA 隐藏了延迟
```

我们现在处于**计算受限**状态，Tensor 利用率 89%。剩余差距：warp 占用率仅 28%，受共享内存限制（每块约 80 KB），H800 上每 SM 最多容纳 2 个块。

### 阶段 2 —— WARP_N 搜索 (iter004 → iter017)

在内核计算受限时，调整 `WARP_N` 改变以下因素之间的平衡：

- **N 分块算术强度**（更大的 WARP_N = 每次 TMA 加载更多 WGMMA 计算量）
- **共享内存占用**（`rhs_s[STAGES]` 随 WARP_N 线性增长）
- **网格并行度**（`cdiv(N, WARP_N)` 决定覆盖 N 维度所需的块数）

!!! warning "硬件约束"
    WGMMA 要求 `WARP_N` 为 **8 的倍数**。如 180 或 188 等值在编译时失败：`MMA m64n180k16 not supported`。在 iter019–020 中发现。

![WARP_N 搜索结果：性能在 WARP_N=192 时达到峰值，超过 208 后因 smem 限制下降](assets/img/warpn_sweep.png#only-dark)
![WARP_N 搜索结果：性能在 WARP_N=192 时达到峰值，超过 208 后因 smem 限制下降](assets/img/warpn_sweep_light.png#only-light)

最佳区间是 **WARP_N = 176–192**。超过 192 后，增加的共享内存无法被额外的计算量补偿 —— WARP_N=224 降至 364 TFLOPS，因为更大的 smem 占用阻止了每 SM 运行 2 个并发块。

### 阶段 3 —— 死胡同 (iter011–018)

并非所有方向都有回报：

| 尝试 | 结果 | 原因 |
|---------|--------|-----|
| STAGES=3, WARP_N=192 | 正确性失败 | 3 级管道的编译器 bug |
| STAGES=3, WARP_N=176 | 373 TFLOPS（更差） | 额外的屏障开销占主导 |
| TILE_K=32, WARP_N=192 | 403 TFLOPS（更差） | 更短的 K 分块缩小了重叠窗口 |
| TILE_K=128, WARP_N=192 | CUDA invalid arg | smem 163 KB 超出内核限制 |
| 1p3c（3 消费者），TILE_M=192 | 156 TFLOPS | 512 线程 × 80 寄存器 = 每 SM 仅 1 个块 |
| SWIZ=64, WARP_N=192 | CUDA invalid arg | WGMMA N=192 需要 SWIZ=128 布局 |
| WARP_N=256 | 396 TFLOPS | 更大 smem，更少块/SM |

1p3c 实验很有启发性：512 个线程（4 个 warpgroup）的寄存器预算迫使每 SM 仅有 1 个活跃块，利用率减半。

### 获胜者：WARP_N=192（相比 v3 提升 40%）

```choreo
#define WARP_N 192
#define STAGES 2
#define CONSUMERS 2
```

这是 v4 Croqtile 源码中**唯一的变更**。编译器处理其余一切 —— 为新的分块形状自动重新生成正确的 TMA 张量描述符、swizzle 布局和 mbarrier 计数。

---

## SOTA 对比

在相同的 8192×8192×8192 f16 问题上，通过 PyTorch 的 `torch.mm` 运行 cuBLAS：

*NVIDIA H800 PCIe — FP16 Tensor Core 峰值：1513 TFLOPS*

| 实现方案 | 时间 (ms) | TFLOPS | 百分比 |
| --- | ---: | ---: | ---: |
| cuBLAS (`torch.mm`) | 2.46 | 447.5 | 100% |
| **Croqtile v4（调优版）** | **2.35** | **471.3** | **105.3%** |

Croqtile 相对 cuBLAS 的微小优势来自 WARP_N=192 分块在该特定 GPU 型号上命中了更好的 L2/SMEM 工作集平衡。生产级 cuBLAS 还包含本教程范围之外的特性：

- **线程块集群**：cuBLAS 使用 Hopper 多播 TMA 在集群内的块间共享 B 分块。
- **持久内核**：cuBLAS 使块在输出分块间保持存活，以摊销启动开销。
- **Epilogue 融合**：cuBLAS 合并 MMA 存储和输出写入，避免一次 SMEM 往返。

这三项都可以在 Croqtile 中表达 —— 代码库中的生产级内核实现了它们。

---

## 优化阶梯

![优化阶梯：TFLOPS 从 v0 (0.38) 到 v4 (471) 的进展，与 cuBLAS (447) 对比](assets/img/optimization_ladder.png#only-dark)
![优化阶梯：TFLOPS 从 v0 (0.38) 到 v4 (471) 的进展，与 cuBLAS (447) 对比](assets/img/optimization_ladder_light.png#only-light)

| 步骤 | 技术 | 关键 Croqtile 构造 | 加速比 |
| --- | --- | --- | ---: |
| v0 → v1 | SMEM 分块 | `shared`、`dma.copy`、`.subspan().at()` | 3.9× |
| v1 → v2 | TMA + WGMMA | `: group-4`、`tma.copy.swiz<>`、`mma.load.swiz<>` | 188× |
| v2 → v3 | Warp 特化 | `inthreads.async`、`shared event`、`wait/trigger` | 1.19× |
| v3 → v4 | 管道调优 | STAGES=2, CONSUMERS=2, WARP_N=192（28 轮搜索） | 1.40× |

---

## 为什么选择 Croqtile

以上五个内核表达的是*什么*数据在移动、*什么*计算在执行 —— 而非如何实现的机制。这个差距非常显著：

| 原始 CUDA 需求 | Croqtile 等价表达 | 编译器处理的内容 |
| --- | --- | --- |
| blockIdx/threadIdx 运算 | `parallel {i,j} by [...] : block/thread` | 所有索引运算、边界、分块大小 |
| 协作拷贝循环 + `__syncthreads()` | `dma.copy src => dst` | 线程分区、合并访问、屏障 |
| TMA 描述符 (`CUtensorMap`) 设置 | `tma.copy[.async][.swiz<N>]` | 张量映射构造、mbarrier 连接 |
| WGMMA smem 描述符编码 | `mma.load.swiz<N> s.chunkat(i,j)` | 描述符编码、swizzle 对齐 |
| mbarrier 初始化 / arrive / wait 代码 | `shared event`、`wait`、`trigger` | 完整的 mbarrier 生命周期 |
| Warpgroup 调度谓词 | `inthreads.async (wg==0)` | Warpgroup 调度、寄存器分配提示 |
| XOR swizzle 表构造 | `mma.load` 上的 `.swiz<128>` | 反向传播到 `tma.copy`、编译时一致性检查 |

用原始 CUDA 正确编写 v4 内核需要数百行代码：TMA 张量映射构造、mbarrier 初始化、warpgroup 谓词、WGMMA smem 描述符编码、显式的 swizzle XOR 表，以及精心的同步顺序。任何一处错误都会导致结果不正确或静默死锁。

**Croqtile 版本仅 60 行，28 轮参数搜索找到了在该 GPU 上超越 cuBLAS 的配置。** 整个调优过程 —— 包括死胡同 —— 耗时不到 4 小时。用原始 CUDA 进行同样的探索，需要为每个配置重写数百行代码。

---

## 运行内核

```bash
# 编译 — 生成包装 nvcc 的自包含脚本
croqtile -gs -t cute -arch=sm_90a \
    matmul_f16_v4_auto_tuned.co -o v4.cute.result

# 运行（编译 + 链接 + 执行一步完成）
bash v4.cute.result --execute

# 正确性检查（不输出计时信息）
CHOREO_DISABLE_TIMING=1 bash v4.cute.result --execute

# 使用 Nsight Compute 分析（单次启动）
ncu --launch-count 1 \
    --kernel-name regex:__croqtile_device_matmul \
    --metrics \
      sm__throughput.avg.pct_of_peak_sustained_elapsed,\
      dram__throughput.avg.pct_of_peak_sustained_elapsed \
    bash v4.cute.result --execute
```
