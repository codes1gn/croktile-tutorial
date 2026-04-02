# 并行性：将工作映射到硬件

第 1、2 章使用 `parallel` 在 GPU 上铺开工作——但没有告诉鳄霸（Croktile）*如何*在硬件上组织这些实例。编译器采用默认策略，对逐元素运算而言这些默认足够好。矩阵乘法则不同：它具有多个分块层级，自然对应 GPU 层次结构的不同层级，映射是否得当，决定了吞吐量是「玩具级」还是「实战级」。

本章介绍**空间限定符**——将逻辑并行轴映射到 CUDA thread block、线程、warp 与 warpgroup 的注解。读到最后，你将得到一个使用 `dma.copy => shared` 与嵌套 `parallel`、在线程间复用数据的分块矩阵乘。

## 再谈 `parallel` 与 `foreach`

二者在前文均已出现，但有必要精确说明各自含义：

- `parallel p by N` —— 创建 `N` 个彼此独立、并发执行的实例。编译器将它们映射到 GPU 执行单元。适用于各次迭代相互独立的情形。
- `foreach i in [N]` —— 在包含它的上下文中**顺序**执行 `N` 次迭代。适用于后一次迭代依赖前一次的情形（例如矩阵乘中沿 K 维的归约循环）。

二者均支持 `#p`（范围）与 `#` 组合，且均可为多维（`parallel {a, b} by [M, N]` / `foreach {i, j} in [M, N]`）。

## 空间限定符：并行性落在何处

GPU 并非扁平的一袋线程。它具有层次结构：驻留在流式多处理器（SM）上的 **thread block**（亦称 CTA）、以锁步执行的 32 线程 **warp**，以及在新硬件上由 128 线程（四个 warp）协作执行宽指令的 **warpgroup**。

鳄霸（Croktile）允许你在范围之后用**空间限定符**将逻辑并行维映射到上述层次：

```choreo
parallel p by 16 : block     // 16 thread blocks
parallel q by 32 : thread    // 32 threads within a block
parallel w by 4  : group     // 4 warps (one per group)
parallel g by 2  : group-4   // 2 warpgroups of 4 warps each (128 threads per group)
```

当你写 `parallel p by 16 : block` 时，是在告诉鳄霸：「为该体创建 16 个实例，并将每个实例映射到 GPU 上各自独立的 thread block。」当你写 `parallel q by 32 : thread` 时，则是在说：「在已经所处的 block 内，创建 32 个线程。」

若完全省略限定符——仅写 `parallel p by 8`——鳄霸会选取合理的默认映射。在性能关键时，显式空间限定符才是你控制映射的手段。

层次结构关系到数据共享。**共享内存**（`=> shared`）对同一块内的所有线程可见，但不可跨 block。因此若希望多个线程读取同一块 DMA 加载的分块，需要这些线程位于同一 block，且分块置于 shared memory。**本地内存**（`=> local`）仅对单个线程私有。**全局内存**处处可见，但较慢。

## 嵌套并行

真实 GPU 程序会嵌套这些层级。一种常见模式为：

```choreo
parallel {px, py} by [8, 16] : block
  parallel {qx, qy} by [16, 16] : thread {
    // each (px, py) block has 16 × 16 = 256 threads
    // each thread is identified by (qx, qy) within its block
  }
```

外层 `parallel` 产生 8 × 16 = 128 个 block 的网格。内层 `parallel` 在每个 block 内产生 256 个线程。总计：128 × 256 = 32,768 个线程。花括号 `{px, py}` 与 `{qx, qy}` 在一次声明中引入**多维**并行下标——相当于否则需要两行嵌套 `parallel` 的简写。

## 矩阵乘：综合起来

借助 `parallel`、`dma.copy`、`chunkat` 与 `#`，分块矩阵乘已触手可及。这是本教程中首个真正体现 GPU「看家本领」的程序。

计划如下：将 `[128, 256]` 矩阵与 `[256, 256]` 矩阵相乘，得到 `[128, 256]` 结果。输出划分为 block 网格；在每个 block 内，线程协作将 `lhs` 与 `rhs` 的 K 条带加载到 shared memory，再计算部分积。

**标量矩阵乘（无 DMA，仅 `parallel` 与 `.at()`）。**

从 `parallel` 与直接全局访问起步，无显式数据搬运：

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel p by 16, q by 64 {
    foreach index = {m, n, k} in [128 / #p, 256 / #q, 256]
      output.at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n);
  }

  return output;
}
```

信息很密——下面说明各部分作用。

**由操作数维度确定输出形状。** 行 `s32 [lhs.span(0), rhs.span(1)] output` 根据输入构造输出形状：`lhs.span(0)` 为 128（左矩阵行数），`rhs.span(1)` 为 256（右矩阵列数）。这是第 2 章中的 `span(i)` 语法——选取单维而非复制整个形状。

**多轴并行。** `parallel p by 16, q by 64` 用逗号一次声明两个并行下标。由此得到 16 × 64 = 1024 路并行网格。每个 `(p, q)` 对拥有输出矩阵的一块分块。

**命名元组解构。** `foreach index = {m, n, k} in [128 / #p, 256 / #q, 256]` 引入三个嵌套循环下标——`m`、`n`、`k`——并绑定到名为 `index` 的元组。各维趟数为：

- `128 / #p = 128 / 16 = 8`，每块行数
- `256 / #q = 256 / 64 = 4`，每块列数
- `256`，完整收缩维

**组合后的全局下标。** `p#m` 与 `q#n` 将分块下标与块内偏移组合成全局位置（外层 # 内层，约定与第 2 章相同）。`p` 从 16 个行分块中选其一，`m` 在该分块内取 0..7，故 `p#m` 覆盖所有分块上的完整 0..127 行范围。

**算术。** `output.at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n)` 即教科书式的点积：对每个输出元素，沿 K 维求乘积之和。此处每次 `.at()` 均从全局内存读取——这正是需要 DMA 之处。

**加入 DMA：分块置于 shared memory。**

标量版本可以工作，但每次乘法都从全局内存读取。先将 K 方向分块载入 shared memory，可使 block 内所有线程复用这些数据：

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel {px, py} by [8, 16] : block {
    foreach {tile_k} in [16] {
      lhs_load = dma.copy lhs.chunkat(px, tile_k) => shared;
      rhs_load = dma.copy rhs.chunkat(tile_k, py) => shared;

      parallel {qx, qy} by [16, 16] : thread {
        foreach k in [256 / #tile_k]
          output.at(px#qx, py#qy) += lhs_load.data.at(qx, k) * rhs_load.data.at(k, qy);
      }
    }
  }

  return output;
}
```

变化要点：

1. 外层 `parallel` 现采用花括号形式下标 `{px, py}` 并带 `: block`。8 × 16 的 block 网格覆盖输出。

2. 对 `tile_k` 的 `foreach` 沿 K 维走 16 步。每步用 `dma.copy ... => shared` 将 `lhs` 与 `rhs` 各一条带拷入 `shared` memory。目标为 `shared` 而非 `local`，因为 block 内所有线程需读取同一块分块。

3. 内层 `parallel {qx, qy} by [16, 16] : thread` 在每个 block 内创建 256 个线程。每个线程在该 block 的分块内负责一个输出元素。

4. 算术从 `lhs_load.data` 与 `rhs_load.data`——即 shared memory 中的副本——读取，而非从全局的 `lhs` 与 `rhs`。

组合下标现分两层：`px#qx` 将 block 下标 `px` 与线程下标 `qx` 组合成全局行；`py#qy` 对列同理。

**维度算术。** 在 `[8, 16]` 个 block 下，每个 block 拥有 `128/8 = 16` 行与 `256/16 = 16` 列。内层 `[16, 16]` 个线程将该区域再细分为每线程一个元素。沿 K 方向，16 个各含 `256/16 = 16` 个元素的分块覆盖完整收缩维。对每个 `tile_k`，`chunkat(px, tile_k)` 选取 block `px` 所拥有的行与 `tile_k` 所拥有的 K 区间；`chunkat(tile_k, py)` 选取 K 区间与 block `py` 所拥有的列。

**主机端代码。** 主机侧仍采用相同的 `make_spandata` / `.view()` 模式：

```choreo
int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(128, 256);
  auto rhs = choreo::make_spandata<choreo::s32>(256, 256);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);

  auto res = matmul(lhs.view(), rhs.view());

  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      int ref = 0;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += lhs[i][k] * rhs[k][j];
      choreo::choreo_assert(ref == res[i][j], "values are not equal.");
    }

  std::cout << "Test Passed\n" << std::endl;
}
```

这里没有新概念——`make_spandata` 创建缓冲区，`matmul(lhs.view(), rhs.view())` 调用鳄霸（Croktile）函数，三重循环与标量参考实现对照校验。主机程序保持平淡，以便鳄霸函数成为焦点。

## Warp group 与 `: group-4`

在 Hopper 级 GPU（计算能力 9.0+）上，NVIDIA 引入了 **warpgroup** 协作：四个 warp（128 个线程）共同发射宽矩阵指令。鳄霸（Croktile）将这一层次记为 `: group-4`：

```choreo
parallel p by 1 : group-4 {
  // body executes as a warpgroup of 128 threads
}
```

即使并行计数为 1（一个 warpgroup 实例），`: group-4` 注解仍告知编译器：该体的 128 个线程构成协作单元，以执行 WGMMA 等指令。第 4 章在加入张量核心运算时会用到这一点。

也可在每个 block 内启动**多个** warpgroup：

```choreo
parallel p1 by 2 : group-4 {
  // two warpgroups, indexed by p1 = 0 and p1 = 1
}
```

这是在某一维上扩展 block 内工作量而不增加 block 数目的方式——两个 warpgroup 共享同一块 shared memory，但保持独立的累加器。第 4 章将据此展开。

## `shared` 如何促成复用

DMA 版矩阵乘选用 `=> shared` 而非 `=> local` 的原因在于：在内层循环中，`[16, 16]` 网格上的每个线程都从**同一**份 `lhs_load.data` 与 `rhs_load.data` 读取。线程 `(qx=3, qy=7)` 读取 `lhs_load.data.at(3, k)`——与线程 `(qx=3, qy=0)` 所读为同一行。若每线程在 `local` 中各持私有一份副本，同一数据会被加载 16 次（每个列线程各一次）。Shared memory 则为整个 block 只存一份。

经验法则：若 block 内多个线程需要相同数据，将其拷入 `shared`。若每线程工作集相互独立，`local` 使数据更贴近计算并避免 shared memory 的 bank 争用。

## 追踪单个输出元素

在输出中取全局位置 `(row=37, col=50)`。它由哪个 block、哪个线程负责？

行方向将 128 行分为 8 个各含 16 行的分块：`px = 37 / 16 = 2`，偏移 `qx = 37 % 16 = 5`。列方向将 256 分为 16 个各含 16 列的分块：`py = 50 / 16 = 3`，偏移 `qy = 50 % 16 = 2`。故为 block `(2, 3)`、线程 `(5, 2)`。

对该线程而言，K 循环执行 16 次迭代。在迭代 `tile_k = 0` 时，`dma.copy` 将 `lhs` 的第 32..47 行与 K 列 0..15 载入 shared 的 `lhs_load`，将 `rhs` 的 K 行 0..15 与列 48..63 载入 shared 的 `rhs_load`。内层 `foreach k in [16]` 将 `lhs_load.data.at(5, k) * rhs_load.data.at(k, 2)` 在 k = 0..15 上累加到 `output.at(37, 50)`。接着 `tile_k = 1` 加载下一条 K 条带，依此类推。16 次迭代结束后，`output.at(37, 50)` 保存完整点积——与标量参考结果一致。

## 新语法小结

| 语法 | 含义 |
|--------|---------|
| `parallel p by N` | N 路并行，下标 `p` 并发取 0..N-1 |
| `parallel {px, py} by [M, N]` | 多维并行（笛卡尔积） |
| `parallel p by N : block` | 映射到 CUDA thread block |
| `parallel q by N : thread` | 映射到 block 内线程 |
| `parallel w by N : group` | 映射到 warp（每 warp 32 线程） |
| `parallel g by N : group-4` | 映射到 warpgroup（每组 128 线程） |
| `=> shared` | DMA 目标：block 作用域的 shared memory |
| `foreach index = {m, n, k} in [a, b, c]` | 在 `foreach` 中的命名元组解构 |
| `parallel p by A, q by B` | 逗号分隔的并行轴 |

本章构建的分块 DMA 矩阵乘是每一个高性能 GPU 内核的结构骨架。下一章将内层循环中的标量 `.at()` 算术替换为**张量核心（tensor-core）**运算——以单条指令处理 16×16×16 分块的硬件加速矩阵乘。
