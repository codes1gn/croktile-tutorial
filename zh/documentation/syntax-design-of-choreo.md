# Choreo 语法设计

## 引言
Choreo 是一种嵌入式领域专用语言（eDSL），旨在简化加速器硬件上的数据搬运编排，减轻编写高性能 kernel 的工程师的日常负担，包括需受硬件能力约束的 tiling 策略等数据搬运复杂性。借助 Choreo，编程过程更易上手，工程师可更高效地优化这些关键环节。

## 将 Choreo 嵌入 C++
Choreo 为嵌入 C++ 的 DSL 代码。Choreo 编译器将 Choreo 函数做 source-to-source 翻译为 C++，并隐式包含 `choreo.h`，使翻译结果能与其余 C++ 代码协同工作。下例展示用法。

```
// some C++ code

__co__ void choreo_function() {
  // choreo code
}

// another C++ code
void foo() {
  choreo_function();
}

```
注意：Choreo 函数以关键字 `\__co\__` 为前缀；函数体内的代码均由 Choreo 编译器翻译。其余 C++ 代码按声明名调用 Choreo 函数。

此外，引入若干 Choreo 专有类型，以保证 Choreo 函数实参与调用方一致。因若干重要编程要素尚未介绍，相关细节延至数据类型说明之后再述。

## 变量与类型
Choreo 引入四类基本类型范畴：scalar-type、spanned-type、integer-tuple-type（ituple-type）与 bounded-type，各司其职。

- **Scalar Type**：满足程序控制需求，由 *Integer Type* 与 *Boolean Type* 构成。
- **Spanned Type**：表示用于计算的数据类型（通常为张量）。除引用原始数据外，*spanned type* 还将数据与由多维范围表示的形状相关联；该「多维范围」称为 `mdspan`，对 tiling 等场景很有用，后文即将介绍。
- **Integer Tuple（I-Tuple）Type**：表示一组整数值；常见用法是为多维数据引入各维上界。
- **Bounded (Integer/ITuple) Type**：用于简化数据（子区域）引用的特殊类型，并与循环构造配合以迭代处理数据。

四类中，*scalar* 与 *ituple* 较易接受，因其可对应通用语言中的元素。而 *spanned type* 与 *bounded type* 具有领域特异性，下文详述。

### Scalar Types
如前所述，Choreo 的 Scalar types 包含 *Integer Type* 与 *Boolean Type*。*Integer Type* 类似 C++ 的 `int` 或 `int32_t`，为有符号 32 位整数，取值范围为 $-2^{31}$ 至 $2^{31}-1$。下例展示其用于数据与函数声明：
```
int a;
__co__ int foo(int b);
```
支持整数算术、移位与逻辑运算，语法与 C++ 内建运算一致。
Choreo 不提供无符号标量整数，亦不提供 8/16/64 位标量整数等价物；该设计动机在于：程序控制未必需要这些类型，通常 32 位有符号整数已足够。

Choreo 中 *Boolean Type* 类似 C++ 的 `bool`；布尔上的运算及与整数之间的转换，与 C++ 一致。

### Spanned Types
Spanned type 为**复合类型**，由 **fundamental type** 与 **multi-dimensional-span（mdspan）type** 组成。*fundamental type* 与 *mdspan* 均非**完整类型**，即二者单独均无法为带存储的数据定型。在 Choreo 中称之为 *partial types*。

将类型设计为二者组合，是为了能单独操纵 *mdspan*。在 ML 场景中，数据多为多维组织；程序员常将数据分块并在存储层次间搬运，因而操纵多维数据的 *shape*（由 *mdspan* 表示）至关重要。

下文将说明如何定义这些 partial types，以及如何将其组合为完整类型，以声明/定义多维数据。

#### Partial Type：**mdspan**
与多数类型系统不同，*mdspan* 作为类型实体出现时仅为 partial。称其为 *partial*，是因为 Choreo 程序无法将数据单独定义为 *mdspan* 类型；但 *mdspan* 本身可单独定义。该设计基于如下观察：loop tiling/blocking 主要关心多维数据的形状，因此 Choreo 允许程序员在不论关联何种 fundamental type 的情况下操纵 *mdspan*，以简化与形状相关的编程。

同时，choreo 编译器可对 *mdspan* 做类型检查，以尽早（在适用时于执行前）发现代码错误。

在 Choreo 中，mdspan 由 `[` 与 `]` 括起，维上界以 `,` 分隔。例如
```
mdspan sp : [7, 8];           // defined a mdspan of 2-dimensions
mdspan<4> mds : [a, 3, 4, 1]; // 'a' is an existing integer
d : [c, 4, 28];               // 'd' is not explicit annotated. Type is deduced.
```
如图所示，可用 *mdspan* 关键字显式定义 mdspan，并可选用尖括号中的总维数；亦可无类型标注定义 mdspan，如上例中的 `d`。（注意：`mds` 与 `d` 属于**依赖类型**，因其类型依赖于 `a`、`b` 等求值；程序员可为 `a`、`b` 提供运行时取值，但需付出若干运行时检查代价。）

对 C++ 程序员而言，可将 mdspan 视作多维数组的性状（trait），描述多层范围；例如上例中 `sp` 定义两层范围，分别为 0..6 与 0..7。

定义 mdspan 后，可通过 `()` 运算符取得各维整型尺寸。
```
sp : [7, 8];
int b = sp(0) + sp(1);   // 'b' equals to 15 (7 + 8)
```
上例中，`sp(0)` 表示 mdspan `sp` 第一维的尺寸；各维尺寸本质上为整型，可用于整数算术。

由此可从已有 mdspan 派生新的 mdspan，如下：
```
sp : [6, 8];
spn : [1, sp(0)/2, sp(1)/4]; // define a new mdspan from the existing one.
```
该机制便于在 Choreo 代码中对多维 span 做 tiling。鉴于高性能 kernel 构造中 tiling 不可或缺，Choreo 提供语法糖以进一步简化：
```
sp : [6, 8];
spn : sp [1, (0)/2, (1)/4];  // spn is defined as [1, 3, 2]
```
语义与前一写法相同，语法更简洁，体现 Choreo 的设计取向之一：在可行时用最少代码实现功能。

**注意：mdspan 仅可定义，不得修改已有 mdspan；且同一 mdspan 只能定义一次。**

除上述按**逐维**方式定义 mdspan 外，Choreo 还支持其他定义方式；在介绍 *i-tuple* 时再说明。

#### 完整定型（Fully-Typing）
单独 *mdspan* 无法用于定义计算数据。在 Choreo 函数中，数据定义须**完整定型**，由 fundamental type 与 *mdspan* 共同组成。下例说明：
```
ndims : [20, 15];
f32 [10, 10] d0;
f16 [ndims] d1;
```
Choreo 支持的 fundamental types 包括：

- Unsigned Integers：*u8/u16/u32*
- Signed Integers：*s8/s16/s32*
- Floating-points：*f16/bf16/f32*

注意：Choreo 中 `s32` 与 `int` 不同。`s32` 为 fundamental type，不能用于 fully typing。

#### 存储限定符（Storage Qualifier）
Spanned 类型数据在 Choreo 中通常较大；程序员可将其在加速器不同 memory hierarchy 间搬运以充分利用硬件。

在 choreo 中定义三种 storage qualifier 以标注所定义数据：

- **global**，
- **shared**，
- 以及 **local**。

用法如下：
```
ndims : [20, 15];
local f32 [10, 10] d0;
shared f16 [ndims] d1;
```

默认情况下，若无 storage qualifier，则所定义数据视为来自 *global* memory。

### I-Tuple Types
整数元组为整数的无序集合；如前所述，常用作（下标）索引。

定义 i-tuple 时，将元素置于 `{` 与 `}` 之间。例如

```
ituple index = {5, 4, 3, 2, 1};  // It defines a tuple of 5 elements
index = {a, b};  // 'a' and 'b' are existing integers
```

### 对 *mdspan* 与 *i-tuple* 的运算
Choreo 允许对 *i-tuple* 与 *mdspan* 进行特殊运算。下例对 mdspan 施加固定 tiling：
```
sp : [6, 8];
tiling_factor = {3, 2};
spn : sp / tiling_factor;   // spn is defined as [2, 4];

```
Choreo 中可通过此类 **Tuple-Span Operations** 定义 mdspan。支持的运算包括：

- *mdspan* $/$ *i-tuple*
- *mdspan* $+$ *i-tuple*
- *mdspan* $\%$ *i-tuple*
- *mdspan* $*$ *i-tuple*
- *mdspan* $-$ *i-tuple*

这些运算本质上均可通过 mdspan 的*逐维*定义实现；*tuple-span operations* 有助于写出更可读的代码，亦为 Choreo 所追求的目标。

### Bounded Types
Bounded types 包含 **Bounded Scalar** 与 **Bounded ITuple**。Bounded Scalar 取值范围为 $[0, ub)$，其中 `ub` 为上界。因此若将整数 `p` 设为 bounded，须同时关联具体上界。该关联须在 `parallel-by` 与 `with-in` 等 *Control Structures* 内显式编写，后文介绍。

类似地，*ITuple* 为一组 *Integer*，亦可关联一组上界。具体而言，Choreo 中 *Bounded ITuple* 与 *mdspan* 值关联，以确定一组上界。后续章节将给出详细语法。




## 控制结构
<!-- 
Choreo follows C++ to involve 'if-else' blocks to handle branches inside programs. However, it has significant difference with C++ on parallelization, loop, etc.
-->

Choreo 在并行化、循环等方面与 C++ 有显著差异。


### 并行区域：`parallel-by` 块
在 CPU 等系统上，可通过异步线程实现并行执行。而在 Choreo 中，采用单指令多数据（SPMD）模型作为并行化手段，类似部分 OpenMP parallel 指令及 OpenCL/CUDA 等并行语言。

但构造并行区域的语法不同：采用 C 风格花括号，将并行执行的代码置于 `parallel-by` 块内。

```
parallel p by 6 {
  // SPMD code
}
```

上例展示如何用 Choreo 关键字 `parallel` 与 `by` 创建并行区域：假定有 6 个执行线程，各线程执行相同的 SPMD 代码，但 `p` 取值不同。若熟悉 CUDA 编程，可将 `p` 类比为 `thread index`；若更熟悉顺序 C/C++，也可将 `p` 视为迭代 6 次的循环变量（但各次迭代的执行次序任意）。

除并行外，`parallel-by` 还有一层含义：语句中的 `p` 为与上界 $[0, 6)$ 关联的整数，称为 **bounded integer**，而非普通整数。在 `chunkat` 等特殊运算中（后文说明），需要 *bounded-integer* 才能正确工作，因为上界对其计算必不可少。

### `with-in` 块与 `where` 子句
与 `parallel-by` 类似，`with-in` 也可将 *i-tuple* 绑定到 *mdspan*。下例说明：
```
with index in [10, 10] {
  // index is ituple with 2 elements
}
```
此处 `index` 为含 2 个元素的 *i-tuple*。有时希望为两元素命名，可使用如下语法：
```
with {x, y} in [10, 10] {...}
```
或同时命名 *i-tuple* 与其元素：
```
with index = {x, y} in [10, 10] {...}
```
此类情形下，称 `index` 为 **bounded ituple**。

`with-in` 表面类似 `parallel-by`，但行为不同：重要区别之一是 `with-in` **不**隐含并行；块内代码顺序执行，仅用于创建 *bounded-ituple*。

此外，可追加 `where` 子句。例如：
```
with {m, n} in [M, N], {n_p, k} in [N_P, K] where n_p <-> n {
  // matmul implements with m,n,K. n_p is no long useful.
}
```
该片段要求在所有迭代中 `n` 与 `n_p` 取值相同；因而在 `with-in` 块内，需要 `n_p` 处可用 `n` 替换，反之亦然。该机制对编写多种 AI kernel 很有用；应使用运算符 `<->` 建立此类关系。

### `foreach` 块
由 `with-in` 定义 *bounded-ituple* 之后，可对其或 bounded-integer 进行循环。在 Choreo 中写法简洁：

```
with x in [10] {
  foreach x {
    // do something with each x
  }
}
```

### `upper-bound` 运算

### 异步运算：DMA 语句
除并行执行外，Choreo 允许一种固定形式的异步运算：DMA 语句。

概念上，DMA 语句与 SPMD 代码**异步**执行，类似 CPU 异步线程，但行为受 DMA 配置约束。（CPU 对异步线程的编程自由度更高。）

下例展示基本 DMA 语句：
```
global f32 [10] g_data;
local f32 [10] l_data;
f = dma.copy.async g_data => l_data;
// ... async operations
wait f;         // explicit wait
```
此处利用数据搬运引擎（DTE）发起线性拷贝，将位于 global memory 的数据搬至 local memory 的 `data1`。可注意到对 `f` 的赋值：其为异步 DMA 实体的句柄，在 Choreo 中称为 DMA 操作的 `future`。程序员可写显式同步语句 `wait`，使当前线程在 `future` 完成前阻塞；否则线程与 DMA 并行继续执行。

有时显式定义临时数据较为繁琐，Choreo 提供更简语法：
```
global f32 [10] data;
f = dma.copy data => local;
... f.data;  // retrieve the 'local' data from the future
```
此处不必指定 DMA 的目标具体地址，仅需指定目标 memory 类型。这在许多场景下有利：程序员常希望避免自行管理 scratchpad memory（SPM）。本例中 local memory 分配由编译器负责；要取得搬运后的数据，只需调用 `future` 的 `data` 成员。此外，该 dma 未标记 `.async`，故无需也不能对 `f` 执行 `wait`。

DMA 操作细节繁复，编程须谨慎，应参考 DMA 手册做决策。尽管如此，Choreo 编译器仍提供大量静态与运行时检查，帮助避免常见错误。

### Bounded-ituple/integer 与 `chunkat` 运算
`chunkat` 作用于 spanned 数据，在已有数据上构造新的 *mdspan*；在部分系统中亦称 `subview`。因 `chunkat` 接受 bounded-ituple 与 bounded-integer 作为参数，在 Choreo 中命名不同。
```
global f32 [6, 10, 100] data;
parallel p by 6 {
  with index = {x, y} in [10, 10] {
    // for every data move, the stride into 'data' is
    //    stride = p*1000 + x * 100 + y * 10
    //
    // for each chunk, the dimensioned size for the movement is {1, 1, 10}
    f = dma.copy data.chunkat(p, index) => local;
  }
}
```
上例展示 `chunkat` 的典型用法：spanned 数据类型为 `f32 [6, 10, 100]`；`chunkat` 接收 *integer* `p` 与 *ituple* `index` 两个参数，假定将数据划分为 $6 \times 10 \times 10$ 块，对应 `p` 与 `index` 所关联的范围；每次取一块并搬运至 local 存储，沿最低维包含 10 个连续元素。


## 函数调用与调用 Choreo 函数
Choreo 函数不得调用另一 Choreo 函数；但在 Choreo 函数内调用 C++ kernel 属常见用法。

```
void bar() {...}    // C++ kernel function
__co__ void foo() {
  parallel p by 6 {
    call bar();     // Call the C++ function
  }
}

```
上例使用 Choreo 关键字 `call` 调用既有 C++ 函数 `bar`，较为直观。

## Choreo 与 C++ 函数间的参数传递
向 Choreo 传参或反向传递时，需包含 Choreo 头文件 `choreo.h`。通常将原始 C++ 指针与维信息结合，构造 Choreo spanned 数据。下例说明：

```
#include "choreo.h"

void bar(const float* data, unsigned size) {}

__co__ void foo(f32 mdspan<2> d) {
  parallel p by 6 {
    call bar(d, |d|);     // Call the C++ function
  }
}

void foobar(float* a) {
  foo(choreo::make_spanview<2>(a, {1, 2}));
}
```
示例中使用 choreo 工具函数（模板）`make_spanview` 包装数据，不发生拷贝。在 Choreo 函数 `foo` 中，参数 `f32 mdspan<2>` 为对应实体；因存在隐式转换，可将 `d` 直接作为 C++ 函数 `bar` 的第一实参；运算 `|d|` 得到 spanned 数据 `d` 的总元素个数，用于调用 `bar`。

同样，*Scalar Type* 数据也可与 Choreo 函数互传；但 `ituple` 仅在 Choreo 函数内部使用。

## 小结
Choreo 为 SPMD 编程提供新思路：偏好 C++ 风格写法，并嵌入 C++；其核心在于减轻底层编程负担，尤其是通过 DMA 在各 memory 层之间编排数据搬运。有时亦被称为 dataflow 编程 DSL。我们开发该工具以支撑日常构建高性能 kernel 的工作，希望程序员少纠缠语言构造细节，更多关注高层概念。

若您觉得其行为符合预期，欢迎反馈以便持续改进。
