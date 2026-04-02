# 数据搬运：从逐元素到数据块

第 1 章以单个元素的粒度表达计算：选取位置 `(i, j)`，读取两个输入，相加，写入结果。这是思考 SIMD 风格编程的自然方式——也正是大多数 CUDA 和 GPU 教程所教授的思维模型。你编写一个逐元素内核，每个元素启动一个线程，每个线程完成自己的微小任务。

问题在于，硬件实际上并不是这样工作的。GPU 不会一次从内存中取出一个 32 位整数。它一次取出连续的数据块——128 字节、256 字节，有时更多——在单次事务中完成，并在任何算术运算触及之前，将这些数据块通过多级缓存和片上缓冲的层次结构进行流转。逐元素编程模型与逐块硬件现实之间存在根本性的不匹配。弥合这一鸿沟——以数据块为单位思考、管理内存层级、建立传输通路——是 GPU 编程对新手而言困难的最大原因。

鳄霸正是围绕这一洞察而设计的。它不强迫你逐元素思考然后寄希望于编译器或硬件自行推算出块结构，而是直接提供**数据块级原语**：用 `chunkat` 命名张量的一个矩形分块，用 `dma.copy` 在内存层级之间搬运它，然后在原地对其进行操作。编程模型与硬件的实际行为相匹配。

本章将第 1 章的逐元素加法改写为使用这些块级原语。数学完全相同——`lhs` 的每个元素仍然与 `rhs` 对应元素相加——但代码现在显式描述了哪些数据块搬运到哪里，计算作用于整个分块而非单个标量。

![逐元素 vs 数据块编程模型对比](../assets/images/ch02/fig1_element_vs_block_dark.png#only-dark)
![逐元素 vs 数据块编程模型对比](../assets/images/ch02/fig1_element_vs_block_light.png#only-light)

*左：逐元素视角——每个线程独立取出一个元素。右：数据块视角——一次 DMA 搬运整个分块。*

<details>
<summary>动画版</summary>

<div markdown>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-dark">
  <source src="/croktile-tutorial/assets/videos/ch02/anim1_element_vs_block_dark.mp4" type="video/mp4" />
</video>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-light">
  <source src="/croktile-tutorial/assets/videos/ch02/anim1_element_vs_block_light.mp4" type="video/mp4" />
</video>
</div>

</details>

## 分块逐元素加法

以下是同一个加法，改写为以 16 个元素为一个分块进行数据搬运。输入是长度为 128 的一维向量，这样分块算术保持简单：

```choreo
__co__ s32 [128] tiled_add(s32 [128] lhs, s32 [128] rhs) {
  s32 [lhs.span] output;

  parallel tile by 8 {
    lhs_load = dma.copy lhs.chunkat(tile) => local;
    rhs_load = dma.copy rhs.chunkat(tile) => local;

    foreach i in [128 / #tile]
      output.at(tile # i) = lhs_load.data.at(i) + rhs_load.data.at(i);
  }

  return output;
}

int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(128);
  auto rhs = choreo::make_spandata<choreo::s32>(128);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);

  auto res = tiled_add(lhs.view(), rhs.view());

  for (int i = 0; i < 128; ++i)
    choreo::choreo_assert(lhs[i] + rhs[i] == res[i], "values are not equal.");

  std::cout << "Test Passed\n" << std::endl;
}
```

保存为 `tiled_add.co`，编译并运行：

```bash
croktile tiled_add.co -o tiled_add
./tiled_add
```

同样输出 `Test Passed`。结果与第 1 章的版本完全相同——数学没有变化，只是数据在内存中搬运的方式变了。

![分块加法：加载、计算、存储流程](../assets/images/ch02/fig2_tiled_add_dark.png#only-dark)
![分块加法：加载、计算、存储流程](../assets/images/ch02/fig2_tiled_add_light.png#only-light)

*tile = 2 的分块加法：DMA 将两个操作数的分块加载到本地内存，逐元素加法在分块上运行，结果写回输出向量。*

<details>
<summary>动画版</summary>

<div markdown>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-dark">
  <source src="/croktile-tutorial/assets/videos/ch02/anim2_tiled_add_dark.mp4" type="video/mp4" />
</video>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-light">
  <source src="/croktile-tutorial/assets/videos/ch02/anim2_tiled_add_light.mp4" type="video/mp4" />
</video>
</div>

</details>

以下是新引入的内容。

## `chunkat`：将张量切分为分块

```choreo
lhs.chunkat(tile)
```

`chunkat` 沿每个维度将张量等分为不重叠的矩形块。这里 `lhs` 的形状为 `[128]`，`parallel` 声明了 8 个分块，因此每个块宽 `128 / 8 = 16` 个元素。参数 `tile` 是块索引——你想要 8 块中的第几块。当 `tile` 为 0 时，得到元素 0 到 15；当 `tile` 为 3 时，得到元素 48 到 63；以此类推。

对于二维张量，`chunkat` 每个维度接受一个索引：

```choreo
matrix.chunkat(row_tile, col_tile)
```

各维度独立划分。如果 `matrix` 形状为 `[64, 128]`，并且你声明 `parallel {r, c} by [4, 8]`，那么 `matrix.chunkat(r, c)` 给你一个在分块位置 `(r, c)` 的 `[16, 16]` 子块。鳄霸根据张量形状和各轴的分块数自动计算块大小。

关键要记住的是：`chunkat` 不复制数据。它是一个**视图**——描述你所指的原始张量中哪个矩形区域。实际的数据搬运发生在 `dma.copy` 中。

![chunkat 二维分块选择语义](../assets/images/ch02/fig3_chunkat_dark.png#only-dark)
![chunkat 二维分块选择语义](../assets/images/ch02/fig3_chunkat_light.png#only-light)

*一个 [64, 128] 张量被分为 4 × 8 个分块。`chunkat(1, 3)` 选中行分块 1、列分块 3 处的 [16, 16] 子张量。*

<details>
<summary>动画版</summary>

<div markdown>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-dark">
  <source src="/croktile-tutorial/assets/videos/ch02/anim3_chunkat_dark.mp4" type="video/mp4" />
</video>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-light">
  <source src="/croktile-tutorial/assets/videos/ch02/anim3_chunkat_light.mp4" type="video/mp4" />
</video>
</div>

</details>

## `dma.copy`：内存层级间的批量传输

```choreo
lhs_load = dma.copy lhs.chunkat(tile) => local;
```

这会将 `chunkat(tile)` 选中的分块从 `lhs` 所在的位置（默认为全局设备内存）复制到 `local` 内存——靠近计算单元的快速片上存储。结果 `lhs_load` 是一个 **DMA future**：代表正在进行（或已完成）传输的句柄。

`=> local` 部分是**目标内存限定符**。我们在介绍完其余语法后，将在下文详细讨论内存限定符。

![dma.copy——批量内存传输](../assets/images/ch02/fig_dma_copy_dark.png#only-dark)
![dma.copy——批量内存传输](../assets/images/ch02/fig_dma_copy_light.png#only-light)

*`dma.copy` 将一个分块从全局内存传输到本地内存，并返回一个 DMA future 句柄。*

## Future 与 `.data`

`dma.copy` 之后，变量 `lhs_load` 不是张量——而是追踪传输过程的 **future**。要获取实际数据，使用 `.data`：

```choreo
lhs_load.data.at(i)
```

`lhs_load.data` 是本地内存中一个带形状的张量，其形状与被复制的分块相同。你用 `.at(i)` 索引它，和第 1 章中索引 `lhs` 的方式完全一样——只是现在你从快速内存而非全局内存读取。

为什么要这层间接？因为在后续章节中，你将发出**异步**执行的复制——硬件在你的程序执行其他工作的同时开始搬运数据。future 让你可以引用"传输完成时将在那里的数据"而无需立即阻塞。目前，复制是同步的，`.data` 在 `dma.copy` 那行之后立即有效，但模式是相同的。

![Future 与 .data](../assets/images/ch02/fig_future_data_dark.png#only-dark)
![Future 与 .data](../assets/images/ch02/fig_future_data_light.png#only-light)

*DMA future 追踪传输过程。通过 `.data` 访问已复制的数据，它给你一个位于本地内存中的带形状张量。*

## `#` 组合运算符

观察输出的索引方式：

```choreo
output.at(tile # i)
```

`#` 运算符将**分块索引** `tile` 与**局部偏移** `i` 组合，生成 `output` 中的**全局索引**。规则是**外层 # 内层**：高层索引在左，分块内的元素偏移在右。由于 `tile` 选择的是 16 个元素中的哪一块，`i` 在该块内从 0 到 15，因此 `tile # i` 给出在完整 128 元素向量中的位置：具体为 `tile * 16 + i`。

你需要 `#` 是因为 `lhs_load.data.at(i)` 使用的是**局部**索引（分块内的位置），而 `output.at(...)` 使用的是**全局**索引（完整输出张量中的位置）。组合运算符桥接了两个坐标系。将 `tile # i` 读作"分块 `tile` 中的第 `i` 个元素"。

这里 `#` 运算符出现在一个维度上。在第 3 章中，当用并行索引 `p` 和 `q` 对二维矩阵分块时，模式是 `output.at(p#m, q#n)`——相同的思路，只是更多轴。

![# 组合运算符](../assets/images/ch02/fig_compose_dark.png#only-dark)
![# 组合运算符](../assets/images/ch02/fig_compose_light.png#only-light)

*`tile # i` 将分块索引 2 与局部偏移 3 组合，生成全局索引 35。规则始终是外层 # 内层。*

## `#` 范围运算符

在内层循环中：

```choreo
foreach i in [128 / #tile]
```

`#tile` 表示"tile 轴的**范围**"——该维度有多少个分块。这里 `#tile` 是 8（因为 `parallel tile by 8` 声明了 8 个分块），所以 `128 / #tile` 是 16——每个分块中的元素数。这就是内层循环的迭代次数：访问一个分块内的每个元素位置。

`#` 符号在鳄霸中有双重用途：在表达式中作为名称前缀（`#tile`）表示该索引的**范围**；在两个名称之间（`tile # i`）表示**组合**。上下文可以区分——作为范围的 `#` 总是出现为前缀，作为组合的 `#` 总是出现为两个操作数之间的中缀运算符。

![# 范围运算符](../assets/images/ch02/fig_extent_dark.png#only-dark)
![# 范围运算符](../assets/images/ch02/fig_extent_light.png#only-light)

*前缀 `#tile` 给出范围（计数 = 8）。中缀 `tile # i` 组合索引。上下文消除歧义。*

## `span(i)`：选取单个维度

第 1 章用 `lhs.span` 复制输入的*完整*形状。有时你只需要其中一个维度。`lhs.span(0)` 给出第一个轴的大小，`rhs.span(1)` 给出第二个轴的大小，以此类推。当你的输出与输入具有不同的秩时这很重要——例如，矩阵乘法中输出形状 `[M, N]` 来自 `lhs.span(0)` 和 `rhs.span(1)`：

```choreo
s32 [lhs.span(0), rhs.span(1)] output;
```

在这个一维示例中还不需要 `span(i)`，但当你开始对二维张量分块时它就变得重要了。

![span(i)——选取单个维度](../assets/images/ch02/fig_span_dark.png#only-dark)
![span(i)——选取单个维度](../assets/images/ch02/fig_span_light.png#only-light)

*`lhs.span` 复制完整形状。`lhs.span(0)` 只选取第一个维度，`lhs.span(1)` 选取第二个。*

## 二维分块加法

同样的模式适用于矩阵。以下是一个 `[64, 128]` 的加法，分为 `[4, 8]` 块，每块大小 `[16, 16]`：

```choreo
__co__ s32 [64, 128] tiled_add_2d(s32 [64, 128] lhs, s32 [64, 128] rhs) {
  s32 [lhs.span] output;

  parallel {tr, tc} by [4, 8] {
    lhs_load = dma.copy lhs.chunkat(tr, tc) => local;
    rhs_load = dma.copy rhs.chunkat(tr, tc) => local;

    foreach {i, j} in [64 / #tr, 128 / #tc]
      output.at(tr # i, tc # j) = lhs_load.data.at(i, j) + rhs_load.data.at(i, j);
  }

  return output;
}
```

每个构造都自然地推广到更高维度：`chunkat(tr, tc)` 接受两个分块索引，`foreach {i, j}` 引入两个内层索引，`tr # i` 和 `tc # j` 沿各轴组合（外层 # 内层），而 `#tr` / `#tc` 给出分块计数（分别为 4 和 8），因此内层循环边界计算为 `16` 和 `16`。

宿主代码与之前的模式相同——`make_spandata<choreo::s32>(64, 128)`、`.view()`、用嵌套循环验证。

## 内存限定符：数据存放在哪里

每个 `dma.copy` 以 `=> local`、`=> shared` 或 `=> global` 结尾。这些是**内存限定符**——鳄霸对 GPU 物理内存层次的抽象。理解它们的含义以及对应的硬件，值得作一番说明。

### 抽象

现代 GPU 拥有多个存储层级，每个层级在大小、速度和可见范围上各不相同。编写原生 CUDA 迫使你手动管理这些层级：用 `cudaMalloc` 分配全局缓冲区，用显式大小声明 `__shared__` 数组，并寄希望于编译器将局部变量映射到寄存器。这是学习曲线中最陡峭的部分之一。

鳄霸的内存限定符是一种有意的简化。你无需担心 CUDA 的 `__shared__` 声明、寄存器压力或缓存行对齐，只需告诉 `dma.copy` *将数据放在哪里*。编译器处理其余一切——分配大小、bank conflict 避免、寄存器溢出——让你专注于数据流。

### 三个层级

- **`global`** ——全设备内存。在物理芯片上这是 HBM（高带宽内存）或 GDDR——容量大（数十 GB）但访问相对较慢。所有线程块的所有线程都能看到它。输入和输出张量最初位于此处。在 CUDA 中，对应 `cudaMalloc` 分配的内存。

- **`shared`** ——线程块范围的片上 SRAM。每个流式多处理器（SM）有一块快速、低延迟的内存池（通常 100–228 KB），同一线程块内的所有线程可见。在 CUDA 中，对应 `__shared__` 内存。当多个线程需要读取同一分块时使用——矩阵乘法、规约和模板计算中的标准模式。我们将在下一章使用 `shared`。

- **`local`** ——线程私有存储。映射到 SM 上的寄存器和每线程暂存空间——最快的层级，但仅对拥有它的线程可见。在 CUDA 中，这是编译器分配到寄存器的局部变量。适用于没有共享需求的情况，或每个线程处理自己独立切片的场景。

### 映射到 GPU 硬件

下图展示了这三个限定符如何对应物理 GPU 布局。注意其层次结构：数据从 global（大、慢）经 shared（中、快）到 local（小、最快）。鳄霸的 `dma.copy` 让你直接表达这些搬运。

![内存限定符 → GPU 硬件](../assets/images/ch02/fig_memory_hierarchy_dark.png#only-dark)
![内存限定符 → GPU 硬件](../assets/images/ch02/fig_memory_hierarchy_light.png#only-light)

*鳄霸的 `global`、`shared` 和 `local` 限定符直接映射到 GPU 硬件层级：HBM/DRAM、每 SM 共享内存和每线程寄存器。*

### 如何选择限定符

选择不会改变鳄霸函数的语义——只影响性能。你可以将每个 `=> local` 替换为 `=> shared`，程序仍然会产生相同的结果，只是速度特征不同。经验法则：

| 场景 | 使用 | 原因 |
|-----------|-----|-----|
| 每个分块独立，无需共享 | `local` | 最快，无需同步 |
| 多个线程协作处理同一分块 | `shared` | 对线程块内所有线程可见 |
| 将结果写回输出张量 | `global` | 输出必须全设备可见 |

目前示例使用 `local`，因为每个分块独立运行。第 3 章在多个线程需要读取同一分块进行分块矩阵乘法时，将引入 `shared`。

## 与第 1 章相比有何变化

数学上没有任何变化。`lhs` 的每个元素仍然与 `rhs` 的对应元素相加。变化的是内存访问的**粒度**：不再是 128 次单独的读写，而是对每个输入张量发出 8 次批量复制，每次搬运 16 个连续元素到快速内存。计算循环随后完全在本地数据上运行。

新增词汇表：

| 语法 | 含义 |
|--------|---------|
| `dma.copy src => local` | 将 `src` 批量复制到 local（或 `shared` / `global`）内存 |
| `tensor.chunkat(i)` | `tensor` 第 `i` 个等分块的视图 |
| `tensor.chunkat(i, j)` | 二维分块中位置 `(i, j)` 的块视图 |
| `future.data.at(...)` | 通过 DMA future 访问已复制的数据 |
| `tile # i` | 将分块索引 `tile` 与局部偏移 `i` 组合为全局索引（外层 # 内层） |
| `#tile` | tile 轴的范围（分块数） |
| `lhs.span(0)` | `lhs` 沿第一个维度的大小 |
| `local` / `shared` / `global` | DMA 目标的内存层级限定符 |

所有这些都与第 1 章的 `parallel` 和 `.at()` 自然组合。[下一章](ch03-parallelism.md)将深入 `parallel`——将其映射到 CUDA 线程块、warp 和 warpgroup——并利用本章的一切来构建分块矩阵乘法。
