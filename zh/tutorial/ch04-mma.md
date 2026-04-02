# 张量收缩：`mma` 语法

第 3 章的分块矩阵乘法以内缩方式计算内层收缩，其方式与 CPU 类似：一个 `foreach k` 循环，每次迭代将两个标量相乘并累加到单个输出元素上。这样做可行，但也是使用现代加速器最慢的方式，因为它忽略了专为这一任务设计的硬件单元。

这里的任务是**二维张量收缩**——运算 C += A × B，其中 A、B、C 均为小而固定形状的矩阵 tile。它是每个 GEMM、每个以 im2col 矩阵乘法表示的卷积、每个注意力头中 QK^T 的内核。该运算如此核心，以至于硬件厂商会构建专用单元，用一条宏指令执行它：NVIDIA 称之为**张量核心**，Google 称之为 **MXU**，Intel 有 **AMX tile**，定制的领域专用加速器（DSA）也有各自的变体。tile 尺寸各不相同（NVIDIA 上 FP16 为 16×16×16，TPU 上为 128×128 脉动阵列，AMX 上为 16×64），但数学形状处处相同：取 A 的一块 tile、B 的一块 tile，相乘，并累加到 C 中。

![二维张量收缩：A[M,K] × B[K,N] → C[M,N]，以及不同的硬件实现](../assets/images/ch04/fig1_tensor_contraction_dark.png#only-dark)
![二维张量收缩：A[M,K] × B[K,N] → C[M,N]，以及不同的硬件实现](../assets/images/ch04/fig1_tensor_contraction_light.png#only-light)

*同一数学运算——在 tile 形操作数上做 C += A × B——在不同加速器上映射到不同的硬件实现。*

对程序员而言，难点不在于数学，而在于**寄存器布局**。在 GPU 张量核心上，tile 并非连续存放在单个线程的寄存器中。它是**碎片化**的：warp 中的 32 个线程各自拥有 tile 的零散片段，而具体的分散模式取决于数据类型、架构代际，以及操作数是 A、B 还是 C。编写原始 CUDA 意味着声明 `wmma::fragment` 对象、调用 `load_matrix_sync` 以正确模式将共享内存中的 tile 分布到各 lane、发出 `mma_sync`，再调用 `store_matrix_sync` 重组输出。布局一旦出错——例如把列主序 tile 载入行主序 fragment——结果会在不知不觉中错误。

![GPU 张量核心的寄存器布局：线程拥有 tile 的零散片段](../assets/images/ch04/fig2_register_loading_dark.png#only-dark)
![GPU 张量核心的寄存器布局：线程拥有 tile 的零散片段](../assets/images/ch04/fig2_register_loading_light.png#only-light)

*简化视图：warp 中的 32 个线程如何拥有 MMA tile 的零散寄存器片段。具体模式因架构而异，且有意保持不透明。*

鳄霸（Croktile）的设计完全绕开了这一复杂性。它不暴露架构相关的 fragment 类型，而是提供**四种抽象操作**，作用于不透明的寄存器 tile：**fill**、**load**、**multiply** 和 **store**。无论由哪种硬件后端执行，这些操作都描述同一套二维收缩工作流。编译器为目标架构处理 fragment 布局、lane 映射与指令选择——你描述的是*做何种*收缩，而非*如何*分散寄存器。

![鳄霸的四步 MMA 语法：fill、load、multiply、store](../assets/images/ch04/fig3_mma_syntax_dark.png#only-dark)
![鳄霸的四步 MMA 语法：fill、load、multiply、store](../assets/images/ch04/fig3_mma_syntax_light.png#only-light)

*四步 MMA 语法是一种抽象接口——并非硬编码到 GPU 张量核心。任何支持二维 tile 收缩的领域专用加速器（DSA）都可以映射到这些操作。*

## 四步 MMA 语法

每个张量收缩内核都遵循同一节奏：

1. **`mma.fill 0.0`** —— 在寄存器中分配累加器 tile `mc` 并置零。
2. **`mma.load`** —— 将操作数 tile 从共享内存载入不透明的 MMA 寄存器 `ma` 和 `mb`。
3. **`mma.row.row mc, ma, mb`** —— 发出收缩：**C += A × B** 写入 `mc`。
4. **`mma.store mc, dst`** —— 将 `mc` 从寄存器写回共享内存。

你在 K 上循环执行第 2–3 步（每次迭代载入下一段 K 切片，向同一 `mc` 累加），然后执行一次第 4 步以刷出完成的输出 tile。名称 `mc`、`ma`、`mb` 均为不透明寄存器 tile——你不声明逐 lane 的布局；编译器根据目标与你的布局选择（此处为 `row.row`）推导它们。

## SM86（Ampere）：一个 warp，一块 MMA tile

在 SM86 上，张量核心的 MMA 作用域为**单个 warp**（32 个线程）。在鳄霸中，这对应 `: group`。

![Ampere 与 Hopper 的 MMA 协作范围](../assets/images/ch04/fig4_sm86_vs_sm90_dark.png#only-dark)
![Ampere 与 Hopper 的 MMA 协作范围](../assets/images/ch04/fig4_sm86_vs_sm90_light.png#only-light)

*SM86：每个 MMA 一个 warp。SM90：四个 warp（`: group-4`）协作执行 WGMMA。*

下面是一个适用于 SM86 的完整 FP16 矩阵乘法内核。tile 尺寸与鳄霸基准的默认设置一致：所有 `MATMUL_*` 常量均为 16，因此沿 M 和 N 方向，一个 block tile 等于一块 MMA tile。

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

**`__co__ void` 与原地输出。** 内核无返回值；结果经 `output` 写出。这与常见的 GPU 模式一致：通过全局指针写入。

**block 网格。** `cdiv(M, MATMUL_TILE_M)` 为向上取整除法——沿 M 方向有多少块 tile，含部分 tile。`block_m` 与 `block_n` 选定本 block 负责的输出 tile。

**warp 网格与 `mma.fill`。** `parallel {warp_m, warp_n} ... : group` 将 MMA tile 映射到 warp。当各维均为 16 时，范围为 1×1——单个 warp 覆盖整块 block tile。更宽的 block tile 会增加 warp，每个 warp 自有其 `mc`。

**K 循环与 DMA。** 每个 `iv_k` 阶段通过带 `subspan(...).at(...)` 的 `dma.copy` 将 A、B 条带拉入共享内存。第 7 章会更深入讨论 `subspan` 与 `chunkat` 的区别。

**操作数载入。** `mma.load` 将该 warp 的 tile 从共享内存移入 `ma` / `mb`。`chunkat(warp_m, iv_warp_k)` 选取本 warp 及内层 K 步对应的 M×K 切片。

**收缩。** `mma.row.row mc, ma, mb` 即张量核心的乘加。**`row.row` 是一种布局契约**：两个操作数均按行主序解释。选错变体是正确性错误，而非性能提示。

**存储与收尾。** K 完成后，`mma.store` 将 `mc` 写入 `output_s` 中该 warp 的子矩形，随后 `dma.copy` 将整个 block tile 送到全局内存。

## SM90（Hopper）：WGMMA 与 warpgroup

Hopper 增加了 **Warpgroup 矩阵乘加（WGMMA）**：同一收缩，但由**四个 warp**（128 个线程）协作发出。在鳄霸中，这一更宽的作用域用 `: group-4` 表示，而非 `: group`。

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

**同样的四步。** fill、load、multiply、store——心智模型不变；变的是协作范围。

**`mma.fill.f16`。** Hopper 上常显式写出累加器精度——`.f16`、`.f32` 等。FP16 操作数配 FP32 累加是长 K 时的常见模式。SM86 通常使用更短的 `mma.fill 0.0` 并依赖推断。

**`parallel p by 1 : group-4`。** 一个 warpgroup（四个 warp）执行内层载入与 MMA。助记符 `mma.row.row` 与 Ampere 一致，但硬件发射更宽。

**`chunkat(_, iv_warp)`。** `_` 表示「此处不对该维做分块」——对本 block 而言，共享内存中已驻留完整的 M（或 N）范围；仅 K 按 MMA 切片细分。

| 方面 | SM86（Ampere） | SM90（Hopper） |
|--------|---------------|---------------|
| 线程范围 | 一个 warp —— `: group` | 四个 warp —— `: group-4` |
| 累加器初始化 | `mma.fill 0.0` | `mma.fill.f16 0.0f`（精度后缀） |
| 全局 → 共享 | `dma.copy` | 相同（TMA 见第 7 章） |
| 核心数学 | `mma.row.row mc, ma, mb` | 相同助记符，硬件更宽 |
| 存储 | `mma.store` 写入每个 warp 的 tile | `mma.store` 写入 warpgroup tile |

## 多 warpgroup MMA

第 3 章介绍了 `parallel p1 by 2 : group-4`——一个 block 内有两个 warpgroup。配合 MMA 时，两组可共享同一块 B tile，同时载入 A 的不同行：

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

**用 `p1` 切分 M。** 当 `MATMUL_TILE_M = 128` 且 `MATMUL_WARP_M = 64` 时，block 跨 128 行；`p1` 选择上或下 64 行条带。`lhs_load_s.subspan(MATMUL_WARP_M, ...).at(p1, 0)` 为每个 warpgroup 提供各自的 A 行；两者共用同一 `rhs_load_s`。

**对称存储。** `mma.store` 的目标是 `output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0)`，使每个 warpgroup 写入暂存缓冲区的一半，随后一次 `dma.copy` 发出合并后的 tile。

## MMA 变体

上文示例在稠密 FP16 tile 上使用 `mma.row.row`。鳄霸（Croktile）的 MMA 语法实际上是一族沿多个轴变化的操作。下面是完整图景——部分变体在后续章节出现；前向引用标明位置。

### 布局变体：`mma.<A>.<B>`

两段后缀声明操作数 A 与 B 的**存储布局契约**。选错变体是正确性错误——硬件对每种组合以不同方式解释寄存器位。

| 变体 | 操作数 A | 操作数 B | 典型用途 |
|---------|-----------|-----------|-------------|
| `mma.row.row mc, ma, mb` | 行主序 | 行主序 | 标准 GEMM（本教程全部示例） |
| `mma.row.col mc, ma, mb` | 行主序 | 列主序 | B 以转置存储（基准中常见） |
| `mma.col.row mc, ma, mb` | 列主序 | 行主序 | A 以转置存储 |
| `mma.col.col mc, ma, mb` | 列主序 | 列主序 | 两者皆转置 |

实践中，`mma.row.row` 与 `mma.row.col` 覆盖大多数内核。布局必须与 `dma.copy` 或 `tma.copy` 之后数据在共享内存中的实际排布一致——没有自动转置。

### 稀疏变体：`.sp`

结构化稀疏（Ampere 及更高架构上的 2:4 模式）除标准 A、B、C 外还使用**元数据操作数** `me`：

```choreo
mma.row.row.sp mc, ma, mb, me;
```

`me` 从单独的元数据张量载入，编码 A 中哪些元素非零。硬件跳过与零的乘积，在相同 tile 尺寸下吞吐大约翻倍。任意布局组合均可：`mma.row.col.sp` 等。

### 缩放变体：融合与独立

对 **FP8** 量化操作数（`f8_e4m3`、`f8_e5m2`），需要按 tile 缩放：

```choreo
mma.row.row.scale mc, ma, mb, sc_a, sc_b;
```

它将乘加与按 tile 反量化融合——收缩后每个结果元素由 `sc_a` 与 `sc_b` 缩放。无需单独的缩放遍。

或者，缩放也可以是常规 MMA 之后的**独立语句**：

```choreo
mma.row.row mc, ma, mb;
mma.scale mc, sc_a, sc_b;
```

独立的 `mma.scale` 在收缩完成后应用缩放。该形式出现在部分 MoE 与混合精度内核中，此时缩放来源与标准融合路径不同。

### fill 精度变体

| 变体 | 含义 |
|---------|---------|
| `mma.fill 0.0` | 推断精度（SM86 默认） |
| `mma.fill.f16 0.0f` | 显式 FP16 累加器 |
| `mma.fill.f32 0.0f` | 显式 FP32 累加器（混合精度常见） |

FP16 操作数配 FP32 累加是 K 较大时的标准选择——可避免部分和数值溢出。

### Load 变体 { #load-variants }

| 变体 | 含义 | 详见 |
|---------|---------|-------------|
| `mma.load src` | 从共享内存的普通载入 | 本章 |
| `mma.load.swiz<N> src` | 带 swizzle 模式的载入，以匹配共享内存布局 | [第 7 章](ch07-advanced-movement.md) |

当共享内存按 swizzle 模式排布（以避免 bank 冲突）时，MMA 载入必须使用**匹配**的 swizzle 模式。`<N>` 参数须在 `tma.copy.swiz<N>`（或 `dma.copy.swiz<N>`）与 `mma.load.swiz<N>` 之间一致——不一致会读到垃圾数据。

### store 变体

| 变体 | 含义 |
|---------|---------|
| `mma.store mc, dst` | 标准写入共享内存 |
| `mma.store.transp mc, dst` | 转置的输出布局 |

`mma.store.transp` 以行与列互换的方式写出累加器。当下一级需要列主序输出时很有用。

### 同步

| 变体 | 含义 | 详见 |
|---------|---------|-------------|
| `mma.commit` | WGMMA 流水线阶段之间的栅栏 | [第 6 章](ch06-synchronization.md) |

在生产者与消费者 warpgroup 重叠的流水线内核中，`mma.commit` 标记「已读完本 K 条带的操作数」与「可以安全复用共享内存缓冲区」之间的边界。它是事件驱动流水线中的必要粘合剂。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `mc = mma.fill 0.0` | 将累加器 tile 初始化为零 |
| `mma.fill.f16 0.0f` / `mma.fill.f32 0.0f` | 显式精度的累加器 |
| `ma = mma.load src.chunkat(...)` | 将操作数 tile 从共享内存载入 MMA 寄存器 |
| `mma.load.swiz<N> src` | 带 swizzle 模式的载入（见[第 7 章](ch07-advanced-movement.md)） |
| `mma.row.row mc, ma, mb` | C += A × B（行主序操作数） |
| `mma.row.col mc, ma, mb` | C += A × B（A 行主序，B 列主序） |
| `mma.row.row.sp mc, ma, mb, me` | 带元数据操作数的稀疏 MMA |
| `mma.row.row.scale mc, ma, mb, sc_a, sc_b` | 融合的 MMA + 按 tile 反量化 |
| `mma.scale mc, sc_a, sc_b` | 独立的 MMA 后缩放 |
| `mma.store mc, dst` | 将累加器写入共享内存 |
| `mma.store.transp mc, dst` | 转置写出累加器 |
| `mma.commit` | WGMMA 的流水线阶段栅栏（见[第 6 章](ch06-synchronization.md)） |
| `cdiv(a, b)` | 向上取整除法 |
| `__co__ void fn(...)` | 原地写出结果的内核 |
| `subspan(M, K).at(i, j)` | 显式 tile 范围的视图，按索引 |
| `chunkat(_, iv_warp)` | `_` 通配符：该维不做分块 |

## 本章小结

| 主题 | 要点 |
|-------|----------|
| 二维张量收缩 | tile 形操作数上的 C += A × B——通用的内层内核 |
| 硬件多样性 | GPU 张量核心、TPU MXU、Intel AMX、定制 DSA 均实现此运算；tile 尺寸与寄存器布局各异 |
| 寄存器碎片化 | 线程拥有零散片段；原始 CUDA 需手动管理 lane |
| 鳄霸（Croktile）的抽象 | 四步操作：**fill → load → multiply → store**；编译器处理 fragment 布局 |
| SM86 | `: group` —— 32 线程 warp，单一 MMA 作用域 |
| SM90 | `: group-4` —— 128 线程 warpgroup，WGMMA |
| 布局契约 | `mma.row.row`、`mma.row.col` 等——须与共享内存中的数据一致 |
| 变体维度 | 布局、稀疏（`.sp`）、缩放（`.scale` / `mma.scale`）、swizzle（`.swiz`）、转置存储 |

收缩很快，但载入与计算仍须轮流进行——张量核心做乘法时，存储子系统空闲。[下一章](ch05-branch-control.md)介绍 **warp 特化与条件控制**，使不同线程承担不同角色：一组载入数据，另一组计算。
