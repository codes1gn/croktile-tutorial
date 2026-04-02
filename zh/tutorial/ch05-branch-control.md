# Warp 特化与控制流

第 4 章在 Hopper 上走通了 MMA 生命周期：从共享内存 `mma.load`、`mma.row.row` 在张量核心上运算、寄存器累加，再到 `mma.store` 与 `dma.copy` 写回全局内存。该流水线有效，但它存在结构性局限：张量核心忙于乘法时，内存子系统往往空闲，因为 block 内每条线程执行同一内核体。没有剩余线程在 MMA 执行期间发起下一批 bulk load。

真实流水线并非如此均匀：生产者加载数据，消费者将其转化为运算。在 CPU 上可能用线程与队列；在 GPU 上，经典 CUDA 做法是保留单一内核并辅以谨慎同步，使不同 warp 扮演不同角色。鳄霸（Croktile）将这一划分变为**结构性**的：`inthreads.async` 为不同 warpgroup 指派不同的直线型程序，硬件从而可在不令每条线程同时执行两条路径的前提下重叠 DMA 与 MMA。当调度本身需要判断——例如跳过问题末尾之外的 tile——则使用普通的 **`if`**，它是**运行时**分支，而非角色划分。

本章只有一条主线：如何赋予 warpgroup 不同任务，以及何时用条件语句保护工作。后半部分介绍**持久化内核（persistent kernel）**：固定大小的 block 池在多个输出 tile 上条带化遍历，并用 `if` 防止越界写。

![1P1C 时间线：生产者 DMA 与消费者 MMA 在同一时间轴上重叠](../assets/images/ch05/fig1_role_split_dark.png#only-dark)
![1P1C 时间线：生产者 DMA 与消费者 MMA 在同一时间轴上重叠](../assets/images/ch05/fig1_role_split_light.png#only-light)

## `inthreads.async`：结构性划分，非运行时分支

`inthreads.async (condition)` 的含义是：仅当 `condition` 为真的那些线程，其程序中**才包含**该代码块。它**不是**“每条线程都求值条件，部分跳过 body”——那是 `if` 的行为。该区别影响你对硬件的理解：

- **结构性划分**（`inthreads.async`）：两条（或更多）独立的直线型 body，为不同线程子集编译。生产者 warpgroup 与消费者 warpgroup 是共享地址空间的不同程序。
- **运行时分支**（`if`）：单一程序；每条活跃线程测试谓词；部分执行所取分支，部分不执行。

在传统 GPU 编程中，整个 block 通常共享一个内核。加载与运算之间的重叠则依赖指令级交错，或手工展开的 warp 特化配合屏障与原子操作。鳄霸的 `inthreads.async` 将角色边界纳入语言，使划分显式且各 body 保持简单。

矩阵乘法的典型模式是 **一生产者 + 一消费者（1P1C）**：一个 warpgroup 向共享内存发起 DMA（或 TMA）；另一个对相应 tile 运行 MMA。下面是无同步的骨架：

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

**`parallel p1 by 2 : group-4`** — 两个 warpgroup，每个四个 warp（每个 warpgroup 128 条线程），由 `p1` 索引。

**`inthreads.async (p1 == 0)`** — 仅 warpgroup 0 编译并执行生产者 body；对其他线程并非空转一次。

**`inthreads.async (p1 == 1)`** — 仅 warpgroup 1 执行消费者 body。两段并非同一循环的两个分支，而是两种角色。

与第 3 章的 `parallel` 对比：那里每条线程执行相同 body。此处并行索引**选择**适用哪份任务描述。硬件可在生产者 warpgroup 上调度 TMA 工作，同时在消费者上运行 WGMMA——时间上的重叠，而非对单条指令流做时间片轮转。

## 1P1C 矩阵乘法骨架

下面展示该划分如何嵌入 Hopper 矩阵乘法。事件、等待与触发有意省略；[第 6 章](ch06-synchronization.md) 补充同步。请关注分工：

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        // Producer: walk K, load tiles into shared
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;
        }
      }

      inthreads.async (p1 == 1) {
        // Consumer: walk K, MMA on loaded tiles
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

**`parallel {block_m, block_n} by [...] : block`** — 与前几章相同的 tile 网格；`cdiv`（向上取整除法）在维度非整除时统计沿 M、N 的 tile 数。

**生产者 `foreach`** — 以 `cdiv(K, MATMUL_TILE_K)` 步遍历 K 维；仅生产者向 `lhs_load_s` 与 `rhs_load_s` 发出 `dma.copy`。

**消费者 `mma.fill` / `mma.row.row` / `mma.store`** — 消费者从不执行上述 DMA 填充；只读共享内存，在 `mc` 中累加并写回结果 tile。

**缺失的协调** — 两侧此处各自独立遍历 K。消费者假定读取每一 K 条带时已就绪；使该假设成立属于同步问题（第 6 章）。

## `if` 守卫：运行时条件执行

有时需要每条线程都参与的谓词。鳄霸的 `if` 行为与 C 一致：

```choreo
if (tile_id < total_tiles) {
  // only execute this body when the condition is true
}
```

**`if (tile_id < total_tiles)`** — 作用域内所有线程测试条件；为假则跳过 body。这与 `inthreads.async` 相反：单一程序，发散执行。

该模式最常见于**持久化内核**：循环迭代次数可能使部分 block 多出一次不对应真实 tile 的“填充”迭代。

## 持久化内核

在第 3–4 章中，网格随问题规模增长：大致每个输出 tile 一个 block。对大矩阵而言 launch 次数可能极大。GPU 以**波（wave）**运行 block；最后一波常使 SM 部分空闲——**尾部利用率不足（tail underutilization）**。

**持久化内核**将 launch 规模固定（常接近 SM 数量），并让每个 block 遍历多个 tile：

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

**`int total_tiles`** — 输出 tile 的线性计数；各轴 tile 数之积，每轴用 `cdiv` 计算以包含部分 tile。

**`parallel block_id by NUM_SMS : block`** — 固定 worker 数量；`block_id` 表示该持久 worker 的编号，而非单个输出 tile。

**`foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)]`** — 每个 block 走过其份额的迭代；向上取整可能为部分 block 增加一次额外迭代。

**`tile_id = tile_iter # block_id`** — 将迭代索引与 block 索引组合，在线性 tile 列表上条带化（与第 2 章相同的 `#` 运算符，此处用于调度）。

**`block_m` / `block_n`** — 将线性 `tile_id` 映射为二维 tile 坐标，用除法与取模，`cdiv(N, MATMUL_WARP_N)` 为沿 tile 的行宽。

**`if (tile_id < total_tiles)`** — 当条带越过最后一个真实 tile 时跳过 TMA、MMA 与存储。若无此守卫，将读写越界。

内层 K 循环与 MMA body 与第 4 章非持久化风格一致。仅**外层**变化：固定 launch、条带化与守卫。域边界上的部分 tile 在生产内核中仍需掩码或收尾处理；`cdiv` 用于在无法保证整除时确定网格与循环规模。

![持久化内核：条带化 tile、block 着色，以及针对填充的 if 守卫](../assets/images/ch05/fig2_persistent_kernel_dark.png#only-dark)
![持久化内核：条带化 tile、block 着色，以及针对填充的 if 守卫](../assets/images/ch05/fig2_persistent_kernel_light.png#only-light)

## 数据相关网格与持久化网格的选择

| 方面 | 每 tile 一个 block | 持久化（`NUM_SMS` 个 block） |
|--------|-------------------|-------------------------------|
| 网格规模 | 随问题增长 | 固定 |
| 尾部利用率 | 最后一波可能使 SM 空闲 | 各 SM 保持忙碌 |
| 额外构造 | 最少 | `total_tiles`、`tile_iter # block_id`、`if` |
| 复杂度 | 较低 | 较高 |

两种布局本身不会改变数学结果；在浮点结合律意义下均可一致。当 `total_tiles` 远大于 SM 数量时——大 GEMM 常见——持久化调度往往更划算。

## `parallel.async` 与 `stream s`：非阻塞启动

上文均在**内核**内部运行。有时你需要的控制在**宿主端**：在不阻塞宿主线程的情况下启动网格，或将不同网格绑定到不同 CUDA 流，使它们在 GPU 上并发执行。

```choreo
parallel.async {px, py} by [grid_m, grid_n] : block {
  stream s;
  // kernel body
}
```

**`parallel.async`** 立即将控制权交还宿主——内核已入队，但宿主不等待其完成。这相当于在鳄霸（Croktile）中用非默认流调用 `cudaLaunchKernel` 的**非阻塞** launch。

**`stream s`** 在 block 体内将内核绑定到 CUDA 流 `s`。若有足够 SM，使用不同流的多个 `parallel.async` 块可在 GPU 上重叠。若无 `stream s`，默认流会使各次 launch 串行。

这是**宿主端编排**，而非内核内控制流。它不能替代用于 **warp 特化**的 `inthreads.async` 或用于运行时谓词的 `if`——它决定的是相对其他网格，*何时*、*在何处*运行某一网格。

## 本章小结

| 主题 | 要点 |
|-------|----------|
| 均匀 vs 特化 | 每条线程同一内核最简单；角色划分可重叠内存与计算。 |
| `inthreads.async` | 结构性：不同线程不同 body——并非共享的 `if`。 |
| `if` | 运行时：每条线程求值条件；为假则跳过 body。 |
| 持久化内核 | 固定 `NUM_SMS` 个 block，线性 tile id，用 `#` 条带化，用 `if` 守卫。 |
| `parallel.async` / `stream s` | 宿主端异步 launch 与流绑定——与内核内特化正交。 |
| `cdiv` | 向上取整除法，用于 tile 计数与循环边界（全书通用；无单独配方）。 |

**新语法**

| 语法 | 含义 |
|--------|---------|
| `inthreads.async (condition)` | 仅满足 `condition` 的线程包含该代码块——结构性角色划分 |
| `if (expr) { ... }` | 运行时条件——`expr` 为假时跳过 body |
| `tile_id = tile_iter # block_id` | 组合迭代索引与 block 索引以实现 tile 条带化 |
| `int total_tiles = expr` | 鳄霸函数中的局部整数 |
| `parallel.async ... : block` | 非阻塞异步内核 launch |
| `stream s` | 将内核 body 绑定到 CUDA 流 `s` |

要使 1P1C 骨架安全，生产者与消费者仍需对“就绪”有共同约定。[第 6 章](ch06-synchronization.md) 增加 **event**、**swap** 与 **pipeline** 模式，使两侧可在时间上重叠而不在共享内存上竞态。
