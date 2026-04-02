## 概述
宏长期以来是 C/C++ 编程的基础能力。为同时顺畅地操作 C++ 与 *tileflow* 程序，宏展开不可或缺。本节展示 Choreo 的宏处理能力。

## 类对象宏（Object-Like Macros）
Choreo 预处理器支持 C/C++ **类对象宏（object-like macros）**，以连接 host 代码与 tileflow 代码。*类对象宏* 仅做纯文本替换，不接受参数。下例说明：

```choreo
#define M 256
#define N 32
#define K 64

__co__ auto matmul(f32 [M, N] lhs, f32 [N, K] rhs) { /*...*/ }

void foo() {
  choreo::f32 a[M][K];
  choreo::f32 b[N][K];
  // ...
  auto res = matmul(choreo::make_spanview<2>(a, {M, K}),
                    choreo::make_spanview<2>(b, {N, K}));
}

```
在代码片段中，Choreo 函数 `matmul` 的输入并非动态形状；Choreo 预处理器在 Choreo 编译之前将 `M`、`N`、`K` 替换为 `256`、`32`、`64`，从而生成静态形状输入的 tileflow 函数。由此，程序员可使不同代码路径保持一致并便于维护。

注意：截至目前，Choreo **尚不支持**带参宏。因此如下代码：
```cpp
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
```
无法生效。

## 注释
Choreo 支持 C 风格注释，即 `/*...*/` 或 `//...`，依托 choreo 预处理器已提供的能力。

## 条件编译
Choreo 亦支持 C/C++ 中大量使用的条件编译，包括 `#if` / `#ifdef` / `#ifndef` / `#else` / `#endif`。下例展示用法：

```choreo
#define PATH0
// some host code
__co__ foo() {
#ifdef PATH0
// some code related to PATH0
#else
// other code
#endif
}

// host control
#ifdef PATH0
// ...
#else
// ...
#endif
```
由此，host 与 choreo 代码可在同一套预处理流程中受控。

注意：choreo 预处理器的能力仍在增强；完全复刻 C 预处理的可能性不大。除非确有必要，Choreo 不会引入既有 C 特性。

## 与 C++ 预处理的差异
关于 choreo 预处理的重要一点是：其触发时机远早于 C++ 预处理。choreo 编译流程如下所示：

```
chore-preprocessing -> choreo compilation -> c/c++ preprocessing -> c/c++ compilation
```

choreo 预处理的主要目标是使 host/device 宏作为整体协同工作，但该流程有时会导致行为差异。从实现角度，Choreo 预处理器仅替换/条件编译 **tileflow 函数内部**的代码，其余预处理交给 C++ 预处理器，从而将 Choreo 预处理限制在较小范围内。

## 预定义宏
为模拟目标 native 编译，choreo 预处理还会从目标平台读取内置宏。例如，生成 CUDA/Cute 代码时全局定义 `__CUDA__`，而 `__CUDA_ARCH__` 仅在 CUDA/Cute device 代码编译时设置。
因此，这些宏既可用于 tileflow 函数，也可用于 host 代码。
