# Hello Croktile：从零到运行内核

你即将编写、编译并运行一个完整的鳄霸程序——对两个小矩阵进行逐元素加法。它故意不做任何花哨的事情：没有分块、没有 DMA、没有复杂的线程层次结构。那些都会在后面的章节中出现。现在唯一的目标是了解一个鳄霸程序的组成部分，以及它们如何连接在一起。

## 鳄霸程序的两个部分

每个鳄霸程序恰好由两个部分组成：

1. **鳄霸函数**（`__co__`）——内核逻辑，以 `__co__` 前缀标记。它描述在带形状的张量上的计算。编译器将其转译为可在 GPU 上运行的代码。
2. **宿主程序**——标准 C++，负责准备数据、调用鳄霸函数并检查结果。

两个部分都位于同一个 `.co` 文件中。编译器通过 `__co__` 前缀来区分它们。

## 完整示例：逐元素加法

以下是完整程序。它对两个 `[4, 8]` 的 32 位整数矩阵进行逐元素加法：

```choreo
__co__ s32 [4, 8] ele_add(s32 [4, 8] lhs, s32 [4, 8] rhs) {
  s32 [lhs.span] output;

  parallel {i, j} by [4, 8]
    output.at(i, j) = lhs.at(i, j) + rhs.at(i, j);

  return output;
}

int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(4, 8);
  auto rhs = choreo::make_spandata<choreo::s32>(4, 8);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);

  auto res = ele_add(lhs.view(), rhs.view());

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      choreo::choreo_assert(lhs[i][j] + rhs[i][j] == res[i][j],
                          "values are not equal.");

  std::cout << "Test Passed\n" << std::endl;
}
```

将此代码保存为 `ele_add.co`，编译并运行：

```bash
croktile ele_add.co -o ele_add
./ele_add
```

你应该看到 `Test Passed`。现在让我们仔细看看每个部分。

## 鳄霸函数

这是程序的核心。以下逐段解析。

**函数签名。**

```choreo
__co__ s32 [4, 8] ele_add(s32 [4, 8] lhs, s32 [4, 8] rhs) {
```

`__co__` 前缀将此标记为鳄霸函数。与普通 C++ 函数不同，签名中携带完整的形状信息：`s32 [4, 8]` 表示"一个形状为 4 × 8、元素类型为 `s32`（有符号 32 位整数）的二维张量"。参数和返回类型都遵循此约定——`类型 [形状] 名称`。编译器利用这些形状在编译期验证每个索引操作是否在界内。

鳄霸支持以下元素类型：`s32`（有符号 32 位整数）、`f16`（半精度浮点）、`bf16`（brain float）、`f32`（单精度浮点）以及 `f8_e4m3` / `f8_e5m2`（8 位浮点）。目前 `s32` 是最容易推理的类型。

**输出缓冲区声明。**

```choreo
s32 [lhs.span] output;
```

这会分配输出张量。表达式 `lhs.span` 从 `lhs` 复制完整的形状，因此 `output` 自动具有形状 `[4, 8]`。如果你后续改变输入形状，输出会随之调整——这种模式使鳄霸函数更易于泛化。

**计算。**

```choreo
parallel {i, j} by [4, 8]
  output.at(i, j) = lhs.at(i, j) + rhs.at(i, j);
```

`parallel {i, j} by [4, 8]` 创建 4 × 8 = 32 个并行实例，每个元素位置一个。每个实例以其自身的 `(i, j)` 对执行循环体。`.at(i, j)` 访问器在位置 `(i, j)` 索引张量，加法直接执行——一个元素，一个线程。

如果你写过 CUDA，可能会想：这些实例是 CUDA 线程？线程块？简短的回答是，不带任何标注的 `parallel` 让编译器自行决定。在底层，这里的 32 个实例会变成单个线程块中的 32 个 CUDA 线程——但鳄霸在此层级刻意隐藏了这种映射。在[第 3 章](ch03-parallelism.md)中，你将学会使用 `: block` 和 `: thread` 等**空间限定符**来显式控制映射，嵌套多个 `parallel` 轴来构建线程块-线程层次结构，以及将它们映射到 warp 和 warpgroup。现在，只需将 `parallel` 理解为"在 GPU 上并发运行所有实例"，然后继续前进。

**返回值。**

```choreo
return output;
```

`__co__` 函数返回其结果张量。签名中的返回类型（`s32 [4, 8]`）必须与实际返回的内容匹配。宿主端的调用者接收到一个 `choreo::spanned_data`——一个带有形状元数据的拥有型缓冲区，可以用 `[i][j]` 进行索引。

## 宿主程序

宿主部分是纯 C++，搭配少量鳄霸 API 调用：

```choreo
auto lhs = choreo::make_spandata<choreo::s32>(4, 8);
auto rhs = choreo::make_spandata<choreo::s32>(4, 8);
lhs.fill_random(-10, 10);
rhs.fill_random(-10, 10);
```

`choreo::make_spandata<T>(dims...)` 在宿主端创建一个拥有型张量缓冲区。元素类型作为模板参数传递，维度作为函数参数传递。`fill_random` 用给定范围内的值填充缓冲区。

```choreo
auto res = ele_add(lhs.view(), rhs.view());
```

从 C++ 调用 `__co__` 函数就像普通函数调用一样。`.view()` 方法从拥有型 `spanned_data` 生成非拥有型 `spanned_view`——这就是你将宿主端张量传入鳄霸函数而不转移所有权的方式。返回值 `res` 是一个拥有型 `choreo::spanned_data`。

`main` 的其余部分是普通的验证：遍历每个元素并断言相等。宿主程序故意很无聊——有趣的工作发生在 `__co__` 函数中。

## 构建与编译

鳄霸文件使用 `.co` 扩展名。编译器的使用方式类似 `gcc` 或 `clang`：

```bash
croktile ele_add.co                          # 生成 a.out
croktile ele_add.co -o ele_add               # 指定输出文件名
croktile -es -t cuda ele_add.co -o out.cu    # 仅输出 CUDA 源码
```

`-es` 标志在转译后停止，让你检查生成的 CUDA 代码。`-t` 标志选择目标平台。

要查看所有可用选项，运行 `croktile --help`：

```bash
croktile --help
```

这会打印完整的编译器标志列表——输出命名、目标选择、详细度和诊断选项。值得浏览一遍，这样你在需要时知道有什么可用的。

![终端：编译并运行鳄霸](../assets/images/ch01/compile_and_run_dark.png#only-dark)
![终端：编译并运行鳄霸](../assets/images/ch01/compile_and_run_light.png#only-light)

*编译并运行一个鳄霸程序，以及 `--help` 输出。*

<details>
<summary>动画版</summary>

<div markdown>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-dark">
  <source src="/croktile-tutorial/assets/videos/ch01/compile_and_run_dark.mp4" type="video/mp4" />
</video>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-light">
  <source src="/croktile-tutorial/assets/videos/ch01/compile_and_run_light.mp4" type="video/mp4" />
</video>
</div>

</details>

## 目前你掌握了什么

现在你已经掌握了每个鳄霸程序的骨架：一个带有类型化、具形状参数和返回值的 `__co__` 函数；用于逐元素计算的 `parallel` 和 `.at()`；用于从输入推导输出形状的 `lhs.span`；以及宿主端创建和传递张量的 `make_spandata` / `.view()` API。

这个示例可以工作，但它一次只处理一个元素，对数据如何在内存层次中移动没有任何控制。在真实硬件上这很重要——GPU 通过批量搬运数据而非逐元素搬运来达到峰值吞吐。[下一章](ch02-data-movement.md)将引入 `dma.copy` 和 `chunkat` 来表达块级数据搬运。
