# 高级数据搬运：TMA、Swizzle 与不规则访问

第 6 章完成了**安全流水线**的闭环：生产者通过 `dma.copy` 向共享内存各级发起传输，消费者在 **event** 上等待，K-tile 循环让载入与 MMA 重叠，而**双缓冲或多缓冲**确保各阶段互不覆盖。但所有这些传输都是**软件驱动**的——warp 协作完成，每条 lane 参与地址运算，程序按照普通 CUDA 的方式发起载入——只是在鳄霸（Croktile）中表达得更清晰。

本章保持同一条主线——**高级数据搬运**——但改变了底层机制。**`dma.copy`** 在旧架构和某些 DMA 大小的 store 上仍然是正确的思维模型，但在 **Hopper（SM90）** 上，你通常会把 ingress 替换为 **`tma.copy`**。**Tensor Memory Accelerator (TMA)** 是一个专用硬件单元，它接受一个**张量描述符**，以**几乎零线程开销**的方式完成多维 tile 移动，而不再用 warp 来承担地址运算。与 TMA 一起，**swizzle** 对列索引做了一次**固定 XOR 重映射**，使得一个 warp 的同时访问不会全部落在**同一个 4 字节 bank** 上——这正是 **bank conflict** 的根源，硬件不得不把本该并行的读取串行化。最后，鳄霸的 **`view` / `from`**、**`subspan` 步长**、**`.zfill`** 和 **`span_as`** 覆盖了 tile 并非张量整除倍数或窗口起点任意偏移时的**不规则和锯齿状**访问。

![软件 DMA 与 TMA 对比：线程协作载入 vs 描述符驱动的硬件张量拷贝](../assets/images/ch07/fig1_tma_vs_dma_dark.png#only-dark)
![软件 DMA 与 TMA 对比：线程协作载入 vs 描述符驱动的硬件张量拷贝](../assets/images/ch07/fig1_tma_vs_dma_light.png#only-light)

## `tma.copy`：硬件张量搬运

表面语法与 `dma.copy` 几乎一致：

```choreo
tma.copy lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => lhs_load_s;
```

**相同的箭头，全新的引擎。** 这行代码看起来像 `dma.copy`，但工作的承担者从**软件**转向了**硬件**：

- **DMA 路径。** 线程**协作**覆盖 tile：每条 lane 参与**地址运算**和载入发射。吞吐量取决于你能否让这些载入**避开 bank 冲突**——这本身就是一场持续的战斗。
- **TMA 路径。** 一个逻辑上**基于描述符**的操作描述了张量切片；**TMA** 将其展开为正确的**多维**寻址，并将**整个 tile** 作为一个单元搬运。生产者 warp 可以**重叠**其他工作，甚至可以用更精简的启动模式——因为**硬件**而非整个 warp 的线程拥有传输语义。

**好处。** 你仍然通过**事件**或与第 6 章相同的流水线纪律在**整个 tile** 上**同步**，但不再需要操作数 ingress 的**逐线程**载入编排。编译器根据 `__co__` 签名和全局布局构建**张量描述符**；在典型内核中你只需把原来写 `dma.copy` 的地方改写为 `tma.copy`。

## Swizzle 与 Bank Conflict

共享内存被**条带化为 bank**（32 个 bank，常见路径上每 bank 4 字节）。当一个 warp 中**多条 lane** 在同一周期内触及**映射到同一 bank 的不同地址**时，硬件会**串行化**这些访问——即 **bank conflict**。稠密的**行优先** tile 中，连续的列往往恰好落在 warp 的连续 lane 所需的位置，容易产生 **2-way、4-way 乃至更严重**的冲突，大幅**削减有效带宽**。

**Swizzle** 对每行内的列索引施加一个**固定 XOR 重映射**，使线程**实际读取**的布局将访问分散到不同 bank。鳄霸在 copy 和 MMA load 上同时暴露了这一选项，确保 **ingress 与数学运算一致**：

```choreo
tma.copy.swiz<3> src => dst;
```

**Ingress。** copy 按 swizzle 模式 **`N`** 把数据落入共享内存。

```choreo
ma = mma.load.swiz<3> lhs_load_s.chunkat(_, iv_warp);
```

**MMA 读取路径。** 操作数载入必须使用**相同的** **`swiz<N>`**，否则地址不匹配。

**Swizzle 级别。** 模板参数控制**粒度**：`swiz<0>` 为恒等映射，`<1>`、`<2>`、`<3>` 分别对应 **64 B、128 B 和 256 B** 的 XOR 模式。更大的粒度可以消除**更宽**的冲突模式，但要求 **tile 尺寸**与该粒度对齐。

**匹配规则。** **`tma.copy.swiz<N>`** 的 `<N>` 必须与 **`mma.load.swiz<N>`** 一致。如果你用 plain `mma.load` 去读 **`swiz<3>`** 布局的数据，地址不匹配，读出来的就是**垃圾**。编译器**不会**强制这一配对——这是你自行维护的**正确性不变量**。(如[第 4 章 MMA 变体表](ch04-mma.md#load-variants)所述，`mma.load.swiz<N>` 属于 MMA 加载家族。)

![无 swizzle 的 bank 冲突 vs XOR swizzle 将 warp lane 分散到各 bank](../assets/images/ch07/fig2_swizzle_dark.png#only-dark)
![无 swizzle 的 bank 冲突 vs XOR swizzle 将 warp lane 分散到各 bank](../assets/images/ch07/fig2_swizzle_light.png#only-light)

## TMA 流水线矩阵乘法

流水线**骨架**与第 6 章相同：**stage 环**、**`wait` / `trigger` 事件**、**MMA commit**，以及**消费者**在**生产者**填充下一个槽位时消耗 tile。这里生产者将 **`dma.copy`** 替换为 **`tma.copy.swiz<3>`**，消费者将 **`mma.load`** 替换为 **`mma.load.swiz<3>`**。下面是一个使用相同 **1P1C** 分工（`parallel p1 by 2 : group-4`）的 **Hopper FP16** 矩阵乘法示例：

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
    shared f16 [MATMUL_STAGES * MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_STAGES * MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait empty[stage];
          tma.copy.swiz<3> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)
            => lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0);
          tma.copy.swiz<3> rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k)
            => rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0);
          trigger full[stage];
        }
      }

      inthreads.async (p1 == 1) {
        mc = mma.fill.f16 0.0f;
        foreach {s} in [MATMUL_STAGES] { trigger empty[s]; }
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait full[stage];
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load.swiz<3> lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mb = mma.load.swiz<3> rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
          mma.commit;
          trigger empty[stage];
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**流水线对等性。** 与第 6 章 **`dma.copy`** 版本相比，只有 **ingress** 和**操作数载入**行发生了变化；**event**、**staging 索引**和 **commit** 保持不变。**TMA** 去除了载入上的协作式**逐线程**地址运算；**swizzle** 将共享内存布局与 **MMA** 访问模式对齐。**writeback** 到全局内存此处仍使用 **`dma.copy`**——根据目标架构和编译器支持选择 TMA 或 DMA 做 store。

## 处理不规则访问

使用 **`chunkat`** 和 **`subspan(...).at(...)`** 的均匀 tiling 可以覆盖许多内核。实际工作负载还需要**任意偏移的窗口**、**tile 间的步长**、边界处的**部分 tile**以及**布局重解释**——以下小节将这些工具集中在一个标题下。

### 任意偏移窗口：`view` 和 `from`

**`view(M, N).from(row, col)`** 定义了一个 **`M × N`** 矩形，起始于底层张量的 **`(row, col)`** 位置——**不要求**原点与预先计算的 tile 网格对齐。

```choreo
patch = matrix.view(16, 16).from(37, 50);
```

**固定窗口，自由原点。** 这是一个起始于第 **`37`** 行、第 **`50`** 列的 **`[16, 16]`** 切片——**不要求**对齐（如果窗口越过张量边缘，使用 **`.zfill`**）。

**何时使用。** **`chunkat`** 需要张量被均匀分为固定数量的 chunk；**`view(...).from(...)`** 则不需要。**规则** tiling 用 **`chunkat`**，窗口是**锯齿状**或**运行时定位**的场景用 **`view` / `from`**。

```choreo
expert_lhs = lhs.view(expert_M, K).from(expert_offset, 0);
dma.copy expert_lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => shared;
```

**MoE 风格 GEMM。** 在**混合专家**架构中，每个专家的 token 批次通常起始于一个**动态行** `expert_offset`。使用 **`view` / `from`** 切片后，流水线的其余部分——DMA 或 TMA、MMA、event——**无需修改**。

### 步长 tile：`.subspan`、`.step` 和 `.at`

**`subspan(M, K).at(i, j)`** 选取锚定在逻辑 tile 索引 **`(i, j)`** 处、尺寸为 **`[M, K]`** 的 tile。添加 **`.step(sM, sK)`** 使 tile 按 **`sM`** 行、**`sK`** 列的间距排列，而非紧密排列：

```choreo
matrix.subspan(16, 16).step(32, 32).at(i, j);
```

**步长 tiling。** 相邻 tile 索引在**全局**坐标中推进 **`(sM, sK)`**，不一定等于 tile 大小。

**`.step` 的含义。** Tile **`(0, 0)`** 仍从 **`(0, 0)`** 开始，但 tile **`(1, 0)`** 从 **`(32, 0)`** 开始，**`(0, 1)`** 从 **`(0, 32)`** 开始。省略 **`.step`** 时步长等于各轴的 **tile 大小**——即**紧密排列**情形。

**典型用途：** 跳过**填充或保护带**、step 小于 extent 的**重叠** stencil、或匹配非紧密 tile-major 的**外层布局**。

### 零填充：`.zfill`

当 **`M` 或 `K`** 不是 tile 大小的**整数倍**时，沿某轴的**最后一个** tile 是**部分**的。越过张量边界的读取是**未定义行为**，除非你显式**填充**。

```choreo
tma.copy.swiz<3> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k).zfill
  => lhs_load_s;
```

**语义。** **`.zfill`** 作用于 copy 的**源端**：超出范围的元素在目标 tile 中被**写为零**。零对 GEMM 累加**无贡献**，因此 **MMA 循环**可以保持**统一**，同时在数学上对**部分**边界仍然**正确**。

### 布局重解释：`span_as`

**`span_as`** 将一个 buffer 的线性存储**重解释**为另一个具有**相同元素数**的**形状**——**不做**拷贝。

```choreo
flat_buffer.span_as([rows, cols])
```

**仅视图的 reshape。** 元素数量保持不变；只有**逻辑** rank 改变。

```choreo
strip_load = dma.copy data.chunkat(tile) => shared;
tile_2d = strip_load.data.span_as([tile_m, tile_k]);
ma = mma.load tile_2d.chunkat(_, iv_warp);
```

**1D staging 到 2D MMA。** **`span_as`** 将已载入的条带暴露为一个**矩阵**供 **`chunkat`** 使用，无需额外拷贝。

**契约。** **`rows * cols`** 必须等于底层存储的 **span 长度**，否则编译器**拒绝**该程序。

## 本章小结

| 概念 | 在"高级数据搬运"中的角色 |
|------|--------------------------------------|
| **`dma.copy`（第 6 章）** | 软件驱动的流水线载入——线程 + 地址运算；作为对比基线。 |
| **`tma.copy` / `tma.copy.swiz<N>`** | 基于描述符的 **Hopper** ingress；硬件**多维** tile 搬运，**极少线程开销**。 |
| **Swizzle + `mma.load.swiz<N>`** | 使**共享内存布局**与 **MMA** 读取对齐；通过 **XOR** 重映射避免 **bank conflict**——copy 和 load 上的 `N` **必须匹配**。 |
| **`view` / `from`** | **任意偏移**的矩形窗口，用于**锯齿状**或**运行时**切片起点。 |
| **`.subspan(...).step(...).at(...)`** | **步长** tiling——重叠、跳过填充或非紧密布局。 |
| **`.zfill`** | 在 copy 中对越界元素**零填充**，安全处理**部分 tile**。 |
| **`span_as`** | **零拷贝**的形状**重解释**，用于 staging buffer。 |

## 新语法速查

| 语法 | 含义 |
|--------|---------|
| `tma.copy src => dst` | TMA 硬件张量拷贝 |
| `tma.copy.swiz<N> src => dst` | 带 swizzle 模式 `N`（0–3）的 TMA 拷贝 |
| `mma.load.swiz<N> src` | 与 swizzle `N` 一致的 MMA 操作数载入 |
| `tensor.view(M, N).from(r, c)` | 任意偏移的 `M × N` 窗口 |
| `.subspan(M, K).step(sM, sK).at(i, j)` | 步长 tile 选取 |
| `.zfill` | copy 源端越界元素零填充 |
| `span_as([dims])` | 将线性存储重解释为指定形状的张量 |
| `parallel.async ... : block` | 非阻塞异步内核启动（见[第 5 章](ch05-branch-control.md)） |
| `stream s` | 将 kernel body 绑定到 CUDA stream `s`（见[第 5 章](ch05-branch-control.md)） |

[下一章](ch08-cpp-interop.md)从纯鳄霸编排跨入 **C++ 互操作**：**寄存器提示**、**预处理器保护**和**内联 PTX**——当你需要在生成代码旁边**降到底层**时的手段。
