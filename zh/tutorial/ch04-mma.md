# Tensor core：`mma` 操作

第 3 章的分块矩阵乘法用标量乘加逐元素计算每个输出——一次一个乘积，通过 `foreach k` 循环累加。这是 CPU 的典型做法。现代 GPU 配备称为 **tensor core** 的专用硬件，可在单次宏操作中完成小块矩阵乘法（FP16 常见为 16×16×16），在同一片硅上相较标量 FMA 可获得数量级更高的吞吐。

鳄霸通过四个操作暴露 tensor core，构成固定生命周期：**fill**、**load**、**multiply**、**store**。本章用该生命周期取代第 3 章的标量内层循环，先在 SM86（Ampere——每个 MMA 一个 warp）上实现，再在 SM90（Hopper——warpgroup WGMMA）上实现。

## MMA 生命周期

每个 tensor core 矩阵乘法遵循相同节奏：

1. **`mma.fill 0.0`** — 创建驻留在寄存器中的累加器块，初始化为零。
2. **`mma.load`** — 将操作数块从共享内存加载到 MMA 操作数寄存器。
3. **`mma.row.row mc, ma, mb`** — 将操作数块相乘并累加：C += A × B。
4. **`mma.store mc, output_s`** — 将累加器写回共享内存。

在 K 上循环步骤 2–3（每次迭代加载 A、B 的下一 K 切片，向同一 `mc` 累加），然后执行一次步骤 4 刷出结果。名称 `mc`、`ma`、`mb` 为不透明寄存器块——你从不声明其大小或 lane 映射。编译器根据目标架构处理。

## SM86（Ampere）：一个 warp，一块 MMA 块

在 SM86 上，tensor core MMA 作用域为**单个 warp**（32 线程）。在鳄霸中对应 `parallel ... : group`——大小为一个 warp 的协作线程组。

以下为 SM86 的完整 FP16 矩阵乘法核。块大小与鳄霸基准默认一致：所有 `MATMUL_*` 常数为 16，故一个 block 块在 M、N 方向各等于一块 MMA 块。

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_TILE_N)] : block {
    shared f16 [MATMUL_TILE_M, MATMUL_TILE_N] output_s;

    parallel {warp_m, warp_n} by [cdiv(MATMUL_TILE_M, MATMUL_MMA_M), cdiv(MATMUL_TILE_N, MATMUL_MMA_N)] : group {
      mc = mma.fill 0.0;

      foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
        lhs_load_s = dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => shared;
        rhs_load_s = dma.copy rhs.subspan(MATMUL_TILE_N, MATMUL_TILE_K).at(block_n, iv_k) => shared;

        foreach iv_warp_k in [cdiv(MATMUL_TILE_K, MATMUL_MMA_K)] {
          ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k);
          mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k);
          mma.row.row mc, ma, mb;
        }
      }

      mma.store mc, output_s.subspan(MATMUL_MMA_M, MATMUL_MMA_N).at(warp_m, warp_n);
    }

    dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_TILE_N).at(block_m, block_n);
  }
}
```

自外向内阅读：

**网格与 `void` 返回。** 函数返回 `void`——结果就地写入 `output` 参数，而非作为返回值。这是接受目标指针的 GPU 核的惯用模式。`cdiv(M, MATMUL_TILE_M)` 为**向上取整除法**——沿 M 方向的块数，对部分块向上取整。

**Block 级结构。** 每个 block 拥有输出中 `MATMUL_TILE_M × MATMUL_TILE_N` 大小的区域。`output_s` 为结果的共享内存暂存区。索引 `block_m` 与 `block_n` 选取对应区域。

**Warp 级 MMA。** 在 block 内，`parallel {warp_m, warp_n} ... : group` 枚举 warp。当块大小均为 16 时，范围为 1×1——单个 warp 处理整个 block 块。若将 block 块扩至 32×32 而 MMA 仍为 16×16，则有 2×2 = 4 个 warp，各自持有独立累加器。

**K 循环。** 对每个 K 块，`dma.copy` 将 A、B 条带暂存到共享内存。`subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k)` 语法创建带显式块范围的视图，并选取位置 `(block_m, iv_k)` 处的块——与 `chunkat` 思想相同，但显式写出块形状。第 7 章更详细地讨论 `subspan` 与 `chunkat`。

**加载 MMA 操作数。** `ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k)` 将该 warp 的 A 操作数块从共享内存暂存缓冲加载到 MMA 寄存器。`chunkat` 选取该 warp、该内层 K 步的 M × K 切片。

**乘加。** `mma.row.row mc, ma, mb` 即 tensor core 指令。后缀 `row.row` 说明**布局约定**：A、B 操作数均按行主序解释。选错变体不是性能提示——而是正确性错误。硬件会根据该选择以不同方式解释寄存器位。

**Store。** K 循环结束后，`mma.store mc, output_s.subspan(...).at(warp_m, warp_n)` 将该 warp 的累加块从寄存器写入其在共享内存中的子矩形。随后 `dma.copy output_s => output.subspan(...).at(block_m, block_n)` 将整个 block 块拷贝到全局内存。

## SM90（Hopper）：WGMMA 与 warp group

Hopper 引入 **Warpgroup Matrix Multiply-Accumulate（WGMMA）**：同样是 C += A × B，但由**四个 warp**（128 线程）协同发射。在鳄霸中，这种更宽的协作表现为 `: group-4` 而非 `: group`。

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;

  mc = mma.fill.f16 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
      parallel p by 1 : group-4 {
        ma = mma.load lhs_load_s.chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
  mma.store mc, output_s;
  dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

生命周期相同：fill、load、multiply、store。变化如下：

| 方面 | SM86（Ampere） | SM90（Hopper） |
|--------|---------------|---------------|
| 线程范围 | 一个 warp — `: group` | 四个 warp — `: group-4` |
| 累加器初始化 | `mma.fill 0.0` | `mma.fill.f16 0.0f`（精度后缀） |
| Global → shared | `dma.copy` | 相同（或 TMA——见第 7 章） |
| 核心运算 | `mma.row.row mc, ma, mb` | 相同助记符，硬件更宽 |
| Store | `mma.store` 写入 per-warp 块 | `mma.store` 写入 per-warpgroup 块 |

`chunkat(_, iv_warp)` 中的下划线表示「第一维不做分块——保持其完整范围」。仅沿 K 为每个 MMA 切片细分；M（或 N）侧整块已在共享内存中供该 block 使用。

**`mma.fill.f16` 与 `mma.fill 0.0`。** 在 Hopper 上有时需显式指定累加器精度——`.f16` 对应 FP16，`.f32` 对应 FP32。常见模式为 FP16 操作数配 FP32 累加器，以在大 K 下提高数值稳定性。SM86 版本使用泛型 `mma.fill 0.0`，由编译器推断类型。

## 多 warpgroup MMA

第 3 章介绍了 `parallel p1 by 2 : group-4`——一个 block 内两个 warpgroup。其与 MMA 的配合方式如下：两组共享同一块 B，但加载 A 的不同行：

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_TILE_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
  shared f16 [MATMUL_TILE_M, MATMUL_WARP_N] output_s;

  mc = mma.fill.f32 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    parallel p1 by 2 : group-4 {
      foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
        ma = mma.load lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(p1, 0).chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  parallel p1 by 2 : group-4 {
    mma.store mc, output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0);
  }
  dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

当 `MATMUL_TILE_M = 128` 且 `MATMUL_WARP_M = 64` 时，block 块高 128 行，在两个 warpgroup 之间各分 64 行。`p1` 选取哪一半：`lhs_load_s.subspan(MATMUL_WARP_M, ...).at(p1, 0)` 使 warpgroup 0 得到上 64 行，warpgroup 1 得到下 64 行。二者读取同一份 `rhs_load_s`——即复用收益。

Store 与 load 对称：`output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0)` 保证每个 warpgroup 写入输出暂存缓冲各自的一半。

## 鳄霸替你处理的部分

在原始 CUDA 中，tensor core 编程意味着声明特定形状的 `wmma::fragment`、调用 `load_matrix_sync`、`mma_sync`、`store_matrix_sync`，并手动跟踪行主序与列主序变体及 `ldmatrix` 暂存。鳄霸将这些下沉到抽象之下。`mc`、`ma`、`mb` 为逻辑 MMA 块；编译器将其映射到目标架构的正确寄存器布局。

你仍需自行选择：一致的布局（`mma.row.row` 须与实际数据存储方式一致）、能整除硬件 MMA 几何的块大小（SM86 上该 FP16 路径为 16），以及与 ISA 匹配的线程层次（`: group` 与 `: group-4`）。鳄霸并未消除这些约束——而是使其可读，并让你不必亲自处理寄存器映射。

## 调试 MMA 核

若结果错误，首先怀疑**布局**（行主序与列主序，以及 `rhs` 为 `[N, K]` 还是 `[K, N]`），其次**索引**（`block_m`、`block_n` 以及用 `.at` / `chunkat` 绑定的 K 切片），若引入了异步拷贝再查**异步次序**。常见错误是数据实际为列主序却标成 `mma.row.row`，或 `chunkat` 索引与 MMA 块几何不对齐。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `mc = mma.fill 0.0` | 将 MMA 累加器块初始化为零 |
| `ma = mma.load src.chunkat(...)` | 将操作数块从共享内存加载到 MMA 寄存器 |
| `mma.row.row mc, ma, mb` | 在 tensor core 上执行 C += A × B（行主序操作数） |
| `mma.store mc, dst` | 将累加器块从寄存器写回共享内存 |
| `mma.fill.f16 0.0f` | 显式 FP16 精度的累加器 |
| `mma.fill.f32 0.0f` | FP32 精度的累加器（用于混合精度） |
| `cdiv(a, b)` | 向上取整除法：块数向上取整 |
| `__co__ void fn(...)` | 就地写入结果的核（无返回值） |
| `subspan(M, K).at(i, j)` | 带显式块范围的视图，按索引选取 |
| `chunkat(_, iv_warp)` | `_` 通配符：该维不分块 |

内层循环现已由硬件加速，但加载与计算仍交替进行——tensor core 做乘法时存储子系统空闲，反之亦然。[下一章](ch05-branch-control.md)介绍 warp 特化与条件控制，使不同线程承担不同角色：一组搬运数据，另一组计算。
