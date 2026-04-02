# Warp 特化：并发区域与事件

凡面向并行机器的编程语言都必须回答一个问题：**当不同线程需要做不同事情时，会发生什么？**

在 CPU 上，答案很直接——每个核心自有程序计数器，因而每条线程可执行任意代码。在 GPU 上，答案受到严格约束。**SIMT**（Single Instruction, Multiple Thread）执行模型将线程编成每 **warp** 32 条、共享单一指令指针。当分支使一部分线程走一路、另一部分走另一路时，硬件不能简单分叉——必须对分歧路径**串行化**，并对非活跃线程做屏蔽。这便是 GPU 编程语言的根本张力：硬件偏好均匀性，而真实程序需要异质性。

CUDA 对此只暴露一种控制流原语：`if`。全体线程求值条件；条件为假的线程被屏蔽（停用）；在 warp 内两侧路径顺序执行。这是**谓词执行**——简单、通用，有时代价极高。在 `if` 上发生分歧的 warp 会执行两侧，在每一侧都浪费约一半吞吐。

但谓词执行并不够用。考虑矩阵乘内核：一组线程应持续从全局内存取 tile，另一组应持续在 tensor core 上对 tile 做乘法。这不是数据相关分支——而是两个**结构不同的程序**恰好共享同一地址空间。若用 `if` 表达，则一个程序在另一个运行时只能暂停。没有重叠，没有流水线。

鳄霸（Croqtile）引入另外两种控制流原语以填补这一空白：

- **`inthreads.async`** —— **结构化并发区域**：在编译期将线程划分为若干组，使其**同时运行不同程序**。编译器生成独立指令流；硬件独立调度。
- **`shared event` / `wait` / `trigger`** —— **区域间信令**：轻量级同步令牌，使并发区域可安全通信。

与 `if` 一起，这三种原语覆盖 GPU 内核中的完整控制流谱系：数据相关分支、结构化程序组合，以及程序间协调。

## 以 `if` 实现谓词执行

鳄霸（Croqtile）的 `if` 行为与其 C 语言对应物类似：

```choreo
if (tile_id < total_tiles) {
  // body executes only when the condition is true
}
```

作用域内全体线程求值条件。条件为假的线程跳过 body。在单一 warp 内，若部分线程走分支、部分不走，硬件对两路径**串行化**——被跳过一侧的线程空闲，待执行侧跑完后再反过来。这便是 **warp 分歧**，也是运行时灵活性所付出的代价。

**何时使用 `if`：** 无法在编译期解析的数据相关决策。边界检查、部分 tile 保护、条件累加。条件可依赖运行时值——循环索引、输入数据、tile 坐标。

**代价模型：** warp 内分歧会使两路径串行。跨 warp 的分歧（每个 warp 内全体一致）无额外代价——硬件直接跳过未取路径。实用规则：在可能时使 `if` 条件**warp 一致**（32 条线程一致）。

## 以 `inthreads.async` 实现结构化并发区域

`inthreads.async` 解决的问题与 `if` 根本不同。它不在运行时问「这条线程是否执行这段代码？」，而在**编译期**声明「这组线程跑这个程序，那组跑那个程序」。

```choreo
parallel p1 by 2 : group-4 {

  inthreads.async (p1 == 0) {
    // program A: only warpgroup 0 compiles and runs this
  }

  inthreads.async (p1 == 1) {
    // program B: only warpgroup 1 compiles and runs this
  }
}
```

与 `if` 的区别是结构性的，不仅是性能：

| | `if`（谓词执行） | `inthreads.async`（结构化并发） |
|---|---|---|
| **解析时机** | 运行时——每条线程求值条件 | 编译时——线程指派固定 |
| **指令流** | 单一程序；分歧线程被屏蔽 | 每区域独立程序 |
| **执行** | warp 内分歧则串行 | 跨 warpgroup 并发 |
| **PL 类比** | 任意语言中的 `if`/`else` | 结构化并发中的 `async`/`spawn`（Trio、Go goroutine、Cilk） |
| **GPU 类比** | 带屏蔽的 SPMD | 单次 launch 内的 MPMD |

**何谓「结构化」？** 区域按词法作用域界定——编译器在解析时即知哪些线程属于哪一区域。无动态 spawn、无无界并发。每个 `inthreads.async` 块是静态划分。这使其适于编译期分析：编译器可为各区域分配不同寄存器、发出不同指令调度，并验证共享资源使用安全。

**`.async` 修饰符。** 若无 `.async`，`inthreads` 将按顺序执行各区域——线程子集轮流。`.async` 后缀为并发修饰符：告知编译器与硬件各区域可在时间上重叠。这与结构化并发框架中的 `async` 关键字类似——将区域标为可独立调度。

下图示意其效果。上方时间线表示单一 warpgroup 在 DMA 与 MMA 之间交替（串行、无重叠）。下方表示两个 warpgroup 使用 `inthreads.async`——生产者 DMA 与消费者 MMA 在时间上重叠：

![Uniform vs structured-concurrent execution: sequential alternation vs overlapping regions](../assets/images/ch05/fig1_role_comparison_dark.png#only-dark)
![Uniform vs structured-concurrent execution: sequential alternation vs overlapping regions](../assets/images/ch05/fig1_role_comparison_light.png#only-light)

*上图：单一 warpgroup 在 DMA 与 MMA 之间交替——彼此等待。下图：`inthreads.async` 划分为两个并发程序——DMA 与 MMA 重叠，墙钟时间大致减半。*

### 典型模式：1 个生产者 + 1 个消费者

`inthreads.async` 最常见的用法是矩阵乘的 **1P1C**（one producer, one consumer）划分：

```choreo
parallel p1 by 2 : group-4 {

  inthreads.async (p1 == 0) {
    // producer: only warpgroup 0 runs this
    // issue DMA / TMA loads, fill shared memory
  }

  inthreads.async (p1 == 1) {
    // consumer: only warpgroup 1 runs this
    // run MMA on shared memory, accumulate results
  }
}
```

**`parallel p1 by 2 : group-4`** —— 两个 warpgroup（各 128 条线程），由 `p1` 索引。

**`inthreads.async (p1 == 0)`** —— warpgroup 0 编译并执行生产者 body；warpgroup 1 永不见到此代码。

**`inthreads.async (p1 == 1)`** —— warpgroup 1 执行消费者 body。两块为共享地址空间下的独立程序。

但共享地址空间恰恰使问题危险。若无协调，消费者可能在生产者写完之前读取缓冲区。此处即需要 event。

## 以 event 实现区域间信令

当 `inthreads.async` 创建并发区域时，这些区域需要通信方式。鳄霸（Croqtile）提供 **event**——在共享内存中声明的轻量级同步令牌：

```choreo
shared event full;
shared event empty;
```

Event 有两种操作：

- **`trigger name`** —— 表明某条件已满足（例如「数据已就绪」）
- **`wait name`** —— 阻塞直至对应 `trigger` 发生

生产者在将 tile 写入后调用 `trigger full`，表示「数据就绪」。消费者在读前调用 `wait full`，阻塞直至信号到达。对称地，消费者读完之后触发 `empty`（缓冲区可复用），生产者在写下一 tile 前 `wait empty`。

这是**基于信用的有界缓冲区**协议——与操作系统（信号量）、网络流控（TCP 窗口）及硬件（warp barrier）所用模式相同。`full` 为「数据可用」信用；`empty` 为「缓冲区空闲」信用。

### 多阶段流水线的 event 数组

对具有多级缓冲的流水线，声明 event 数组：

```choreo
shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
```

每个物理缓冲槽位自有 `full`/`empty` 对。环形索引 `stage = iv_k % MATMUL_STAGES` 将无界的 K 迭代映射到固定数量的物理槽。四阶段时，生产者在因 `wait empty` 阻塞前可超前运行若干 tile。

### 自举协议

消费者必须在 K 循环开始前对 `empty` 信用**播种**：

```choreo
foreach {s} in [MATMUL_STAGES] {
  trigger empty[s];
}
```

若无此自举，生产者首次 `wait empty[0]` 将永远阻塞——死锁，而非神秘的 MMA bug。这是常见陷阱：凡有界缓冲区协议都需初始信用。

[第 6 章](ch06-synchronization.md) 给出完整的双缓冲与多阶段流水线内核，将这些原语付诸实践。该章示例将 `inthreads.async`、event、`swap` 与 `mma.commit` 组合为可完整运行的矩阵乘流水线。

## 1P1C 矩阵乘骨架

以下展示三种原语在 Hopper 矩阵乘中的组合方式。基于 event 的同步省略——[第 6 章](ch06-synchronization.md) 补充完整流水线协议。请关注程序结构：

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;
        }
      }

      inthreads.async (p1 == 1) {
        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load lhs_load_s.chunkat(_, iv_warp);
            mb = mma.load rhs_load_s.chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**生产者 `foreach`** —— 以 `cdiv(K, MATMUL_TILE_K)` 步遍历 K；warpgroup 0 向共享内存发出 `dma.copy`。

**消费者 `mma` 路径** —— warpgroup 1 不触碰这些 DMA；它读共享内存，在 `mc` 中累加，并写回结果。

**缺失的协调** —— 两侧独立遍历 K。消费者在读时假定每个 K 条带已就绪。要使该假定成立，需要 event（见 [第 6 章](ch06-synchronization.md)）。

## 持久调度与 `if` 守卫

在第 3–4 章中，grid 随问题规模增长：大致每个输出 tile 一个 block。对大矩阵意味着 launch 次数多，最后一波 block 常使 SM 部分空闲——**尾部利用率不足**。

**持久内核**将 launch 规模固定（常接近 SM 数量），并使每个 block 迭代多个 tile：

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  int total_tiles = cdiv(M, MATMUL_WARP_M) * cdiv(N, MATMUL_WARP_N);

  parallel block_id by NUM_SMS : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)] {
      tile_id = tile_iter # block_id;

      if (tile_id < total_tiles) {
        block_m = tile_id / cdiv(N, MATMUL_WARP_N);
        block_n = tile_id % cdiv(N, MATMUL_WARP_N);

        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k) => rhs_load_s;

          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            parallel p by 1 : group-4 {
              ma = mma.load lhs_load_s.chunkat(_, iv_warp);
              mb = mma.load rhs_load_s.chunkat(_, iv_warp);
              mma.row.row mc, ma, mb;
            }
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**`parallel block_id by NUM_SMS : block`** —— 固定工作者数量。

**`tile_id = tile_iter # block_id`** —— 将迭代与 block 索引组合，在 tile 间条带化分配。

**`if (tile_id < total_tiles)`** —— `if` 守卫：运行时谓词，对填充迭代跳过 body。这正是 `if` 的设计用途——数据相关决策，而非结构划分。

![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_dark.png#only-dark)
![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_light.png#only-light)

### 数据相关 grid 与持久 grid

| 方面 | 每 tile 一 block | 持久（`NUM_SMS` 个 block） |
|--------|-------------------|-------------------------------|
| Grid 规模 | 随问题增长 | 固定 |
| 尾部利用率 | 末波可能使 SM 空闲 | 各 SM 保持忙碌 |
| 额外构造 | 最少 | `total_tiles`、`tile_iter # block_id`、`if` |
| 复杂度 | 较低 | 较高 |

## `parallel.async` 与 `stream s`：主机侧并发

上文均在 kernel 内。有时需要在**主机侧**并发：启动 grid 而不阻塞 CPU，或将不同 grid 绑定到不同 CUDA `stream`，使其在 GPU 上并发执行。

```choreo
parallel.async {px, py} by [grid_m, grid_n] : block {
  stream s;
  // kernel body
}
```

**`parallel.async`** 立即将控制权交还主机——kernel 已入队，主机不等待完成。这相当于在非默认 `stream` 上调用 `cudaLaunchKernel` 的鳄霸（Croqtile）写法。

**`stream s`** 将 kernel 绑定到 CUDA `stream s`。若 SM 充足，不同 `stream` 的多个 `parallel.async` 块可在 GPU 上重叠。

这是**主机编排**，与内核内控制流正交。它不替代用于线程划分的 `inthreads.async` 或用于运行时谓词的 `if`——它决定相对其他 grid **何时、在何处**运行某一 grid。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `if (expr) { ... }` | 谓词执行——运行时条件，分歧线程被屏蔽 |
| `inthreads.async (condition)` | 结构化并发区域——编译期线程划分 |
| `shared event name` | 在共享内存中声明同步令牌 |
| `shared event name[N]` | 声明 N 个同步令牌 |
| `trigger name` | 表明某条件已满足 |
| `wait name` | 阻塞直至对应 `trigger` 发生 |
| `tile_id = tile_iter # block_id` | 为 tile 条带化组合索引 |
| `int total_tiles = expr` | 局部整型变量 |
| `parallel.async ... : block` | 非阻塞 kernel launch |
| `stream s` | 将 kernel 绑定到 CUDA `stream s` |

## 本章小结

| 概念 | 原语 | 适用情形 |
|---------|-----------|-------------|
| 谓词执行 | `if` | 数据相关决策（边界、条件） |
| 结构化并发 | `inthreads.async` | 编译期线程划分（生产者/消费者、异构角色） |
| 区域间信令 | `shared event` / `wait` / `trigger` | 并发区域间协调 |
| 主机并发 | `parallel.async` / `stream s` | 多 kernel 重叠、非阻塞 launch |
| 持久调度 | `if` + `foreach` + `#` | 固定 grid 规模、带填充守卫的 tile 条带化 |

上文 1P1C 骨架不完整：若无 `wait` / `trigger`，消费者可能在生产者写完之前读取。[第 6 章](ch06-synchronization.md) 补充完整同步协议——单调度表双缓冲用 `swap`，多 warpgroup 流水线用 event——使流水线安全且满吞吐运行。
