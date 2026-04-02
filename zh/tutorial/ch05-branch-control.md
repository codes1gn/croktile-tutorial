# 分支与控制：Warp 角色与持久内核

迄今为止，块内每个线程执行相同的代码：加载相同的 tile、执行相同的 MMA、写回相同的结果。这种一致性清晰，但代价是：张量核心忙于乘法时，存储系统本可预取下一块 tile——却无人发起，因为每个线程都卡在 `mma.row.row` 指令上。

本章介绍两种有意打破该一致性的控制流。**warp 特化**（`inthreads.async`）为不同的 warpgroup 分配不同角色——一部分加载数据，另一部分计算。**条件守卫**（`if`）在 tile 索引越界时跳过工作，这对在可变数量的 tile 上运行固定块池的**持久内核**至关重要。

## `inthreads.async`：划分角色

`inthreads.async (condition)` 的含义是：「仅当 `condition` 为真时，对应线程才执行该代码块。」它并非运行时上每个线程都求值、部分跳过的分支，而是一种**结构性划分**，会生成两套（或更多）各自直线式的程序，每种角色一套，可在不同硬件单元上并发执行。

典型模式是 **1 生产者 + 1 消费者（1P1C）** 矩阵乘法：

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

`parallel p1 by 2 : group-4` 创建两个 warpgroup，各含 128 个线程。第一个 `inthreads.async` 块仅在 warpgroup 0（生产者）上运行；第二个仅在 warpgroup 1（消费者）上运行。由于两个函数体在结构上相互独立，硬件可在一个 warpgroup 上调度 TMA 加载，同时在另一个上运行 WGMMA——真正的重叠，而非交错执行。

与第 3 章中的 `parallel` 对比：那里每个线程执行相同的函数体。此处 `inthreads.async` 根据并行索引**区分**不同函数体。可将其理解为「每个 warpgroup 有不同的任务描述」。

## 1P1C 矩阵乘法骨架

下面展示 `inthreads.async` 如何嵌入 Hopper 矩阵乘法。同步细节（事件、wait、trigger）有意省略——第 6 章将补充。请重点关注角色划分：

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

生产者从不接触 `mc`、`mma.load` 或 `mma.row.row`；消费者从不发出用于填充共享内存的 `dma.copy`。每个函数体都是对 K 的干净、直线式循环。缺失的一环——消费者在读取之前如何得知 tile 已就绪——属于同步问题，将在第 6 章讨论。

## `if` 守卫：条件执行

有时需要普通的条件分支。鳄霸的 `if` 与 C 语言类似：

```choreo
if (tile_id < total_tiles) {
  // only execute this body when the condition is true
}
```

这是**运行时**检查，而非像 `inthreads.async` 那样的结构性角色划分。每个线程都会对条件求值；条件为假时跳过函数体。

何处需要？主要在**持久内核**中：固定数量的块在可变数量的 tile 上迭代，部分块可能多一次迭代，却没有实际工作可做。

## 持久内核

在第 3–4 章中，网格规模与问题规模成比例：每个输出 tile 对应一块。对大矩阵而言，可能意味着数十万个块。GPU 以**波次**调度它们，最后一波往往使大量 SM 空闲——即**尾波利用率不足**。

**持久内核**扭转这一做法：启动**固定**数量的块（通常与 SM 数量一致），每块在多个 tile 上循环：

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

此处三者协同工作：

**固定启动。** `parallel block_id by NUM_SMS : block` 恰好创建 `NUM_SMS` 个块（例如在 H800 PCIe 上为 114）。索引 `block_id` 并不绑定单个输出 tile——它标识你是哪一个持久工作线程。

**Tile 条带化。** `tile_id = tile_iter # block_id` 将迭代次数与块索引组合，得到唯一的 tile id。块 `b` 处理 tile `b`、`b + NUM_SMS`、`b + 2 * NUM_SMS`，依此类推——在线性化的 tile 列表上以步长 `NUM_SMS` 遍历。`#` 组合运算符与第 2 章相同，此处用于调度算术而非张量索引。

**线性到二维映射。** `block_m = tile_id / cdiv(N, MATMUL_WARP_N)` 与 `block_n = tile_id % cdiv(N, MATMUL_WARP_N)` 由线性 id 恢复二维 tile 位置，与在 C 中将扁平数组索引转为行、列的方式相同。

**`if` 守卫。** 由于 `foreach {tile_iter}` 执行 `cdiv(total_tiles, NUM_SMS)` 次迭代，部分块可能多一次迭代，`tile_id` 可能超过 `total_tiles`。`if` 对这些填充迭代跳过全部 TMA、MMA 与存储工作。若无此守卫，将发生越界索引。

内层函数体——K 循环、MMA、存储——与第 4 章非持久版本相同。仅**外层**改变：固定并行、tile 循环、组合与守卫。

## `cdiv`：向上取整除法

`cdiv(a, b)` 计算 \\(\lceil a / b \rceil\\)——在维度未必整除时所需的 tile 数量。它随处可见：网格范围（`cdiv(M, MATMUL_WARP_M)`）、循环边界（`cdiv(K, MATMUL_TILE_K)`）以及持久迭代次数（`cdiv(total_tiles, NUM_SMS)`）。

当最后一块 tile 为部分有效（有效元素少于 tile 大小时），实际内核会将 `cdiv` 与谓词掩码、零填充或 epilogue 清理配合使用。本教程保持整除的干净情形，但 `cdiv` 正是书写不假定完美整除的 tile 计数的方式。

## 在数据相关网格与持久网格之间选择

| 方面 | 每 tile 一块 | 持久（`NUM_SMS` 块） |
|--------|-------------------|-------------------------------|
| 网格规模 | 随问题增长 | 固定 |
| 尾波利用率 | 最后一波可能使 SM 空闲 | 各 SM 保持忙碌 |
| 额外构造 | 最少 | `total_tiles`、`tile_iter # block_id`、`if` |
| 复杂度 | 较低 | 较高 |

二者均不改变正确性；在浮点结合律意义下输出相同。当 `total_tiles` 远大于 SM 数量时——大矩阵问题中常见——持久布局更有优势。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `inthreads.async (condition)` | 仅满足 `condition` 的线程执行该块——结构性角色划分 |
| `if (expr) { ... }` | 运行时条件——当 `expr` 为假时跳过函数体 |
| `cdiv(a, b)` | 向上取整除法 |
| `tile_id = tile_iter # block_id` | 将迭代索引与块索引组合以实现 tile 条带化 |
| `int total_tiles = expr` | 鳄霸函数中的局部整型变量 |

warp 特化划分角色；`if` 守卫边界情况。但 1P1C 骨架中的生产者与消费者仍各自独立运行 K 循环而互不协调——消费者假定 tile 已就绪。[下一章](ch06-synchronization.md) 将介绍事件、swap 与流水线模式，使生产者与消费者可在时间上安全重叠。
