## 概述
Choreo 的 *tileflow 程序*描述数据如何移动，因而声明或定义数据与缓冲区是基础。本节将介绍相关语法。

## *Spanned*：数据与缓冲区
Tileflow 程序的主要任务是通过移动来操作大规模数据集。在部分术语体系中，Choreo 函数的输入与输出分别称为「输入数据」与「输出数据」；其余存储则统称为「缓冲区」。这在概念上从函数视角区分了「外部」与「内部」内存。然而，「数据」与「缓冲区」均指存储位置。在 Choreo 中，二者均类型化为 **spanned**，表示其为与 *mdspan* 所表示形状相关联的数据/缓冲区。下文称其为 *spanned data*，或简称 *spanned*。

## 定义 *Spanned Data*
定义 *spanned data* 需要元素类型与形状。例如：

```
f32 [32, 16] data;
```

该语句定义形状为 `[32, 16]`、元素类型为 `f32`（*IEEE-754* 单精度浮点）的缓冲区。亦可使用具名 `mdspan` 定义缓冲区：

```
shape : [72, 14];
s32 [shape] data0;              // using the named mdspan `shape`
u32 [shape(0), shape(1)] data1; // same shape as the above
u16 [shape + 6] data2;          // shaped as [78, 20]
s16 [shape, 8] data3;           // shaped as [72, 14, 8]
```

此处使用具名 *mdspan* `shape` 定义名为 `data0` 的二维缓冲区，元素为 32 位整数。注意具名 *mdspan* `shape` 置于 `[]` 内。此外，允许在 *mdspan* 上使用 *element-of*、算术运算或拼接以指定缓冲区形状。

在 Choreo 术语中，元素类型称为**基本类型（fundamental type）**。Choreo 支持多种基本类型，包括：

- **无符号整数**：`u8`、`u16`、`u32`
- **有符号整数**：`s8`、`s16`、`s32`
- **浮点**：`f16`、`bf16`、`f32`

前缀 `u` 与 `s` 分别表示 unsigned 与 signed，数字后缀表示类型的位宽。`f16` 指 *IEEE-754* 标准的*半精度浮点*，`bf16` 指 16 位*二进制浮点*，常用于机器学习场景。

注意 `s32` 与 `int` 不同。二者占用存储相同，但在 Choreo 中 `s32` 不能单独用于定义编程实体。例如：

```choreo
s32 a;  // error: the fundamental type can not be used for variable definition alone
```

这将导致编译期失败。因此，`s32` 不能像 `int` 那样用于程序控制，二者之间亦无法进行类型转换。

## 存储说明符
在实际代码中，部分缓冲区声明必须指定**存储说明符（storage specifier）**，例如：

```choreo
global f32 [32, 7, 2] a;
shared u8 [512, 144] b;
local u8 [72, 1024] c;
```

由于 Choreo 在异构上下文中处理存储，未带存储说明符的缓冲区定义默认采用主机程序的存储类型，即 CPU 内存。其余存储说明符由目标平台定义。例如，面向 GPU 硬件的 *Cuda/Cute* 支持：

- *global*：设备全局存储。
- *shared*：设备线程块共享存储。
- *local*：线程私有存储。

其他目标可能有不同定义。此外，不同存储类型的缓冲区在声明上存在限制，将于后续章节讨论。

## 初始化
可在声明处初始化缓冲区。例如：

```choreo
local s32 [17, 6] b1 {0};       // elements are initialized to 0
shared f32 [128, 16] b2 {3.14f}; // elements are initialized to 3.14f
```

*spanned data* 的*初始化表达式*语法直观：在变量名后以括号将初值括起。但其功能有限——始终将所有元素设为同一固定值。

## 声明参数与返回值
*spanned data* 在主机与 tileflow 程序之间传递，因此 choreo 函数可将 *spanned data* 作为参数。下列代码展示一例：

```choreo
__co__ f16 [7, 8] foo(f32 [16, 17, 5] input) {...}
```

语法与变量定义类似，但不可对已有 *mdspan* 进行操作，且不允许存储说明符或初始化。

对 *spanned* 参数而言，一个有用的内建成员函数为 `.span`，其提供该 *spanned data* 关联的 *mdspan*。下列代码演示如何将其用于缓冲区声明：

```choreo
__co__ auto foo(f32 [16, 17, 5] input) {
  f32 [input.span / {4, 1, 5} ] buffer;  // declare a buffer with tiled shape
  // ...
}
```

## 缓冲区生存期管理
在 *choreo 函数*内部分配的存储之生存期由 Choreo 编译器管理，以实现高效使用。若缓冲区生存期互不重叠，编译器会尽可能复用缓冲区。因此 Choreo 用户通常无需自行管理缓冲区。

## 简要小结
本节介绍了如何在 Choreo tileflow 程序中定义与管理数据/缓冲区，包括声明语法、初始化、存储说明符，以及由编译器实现的缓冲区生存期高效管理。这为进入下一核心主题——动态形状支持——奠定了基础。
