## 概述

## Choreo 函数输入
### 输入参数
对 Choreo 函数而言，仅接受两类输入参数：*spanned data* 与 *integer*。

与任何 C++ 程序类似，参数可以命名也可以不命名：

```choreo
__co__ void foo(f32 [7] a, int) {...}
```

本例中第二个参数未命名，因而在 Choreo 函数内无法引用。该代码仍然有效。


### C++ Host：带 shape 的数据实参
C/C++ 指针/数组与 Choreo *spanned data* 之间一个显而易见的差异在于：*spanned data* 具有 shape。因此调用 Choreo 函数时，需要将指针或数组进行转换。

一种直接做法是使用 `choreo.h` 中提供的 Choreo API。尽管在 Choreo 编译环境中不必显式 `#include choreo.h`，每个 Choreo-C++ 程序都会包含该头文件。

```choreo
__co__ void foo(f32 [7, 16] input, f32 [7, 16] output) {...}

void entry(float * input, float* output) {
  // foo(input, output); <-- result in error since missing shape

  foo(choreo::make_spanned<2>(input, {7, 16}),
      choreo::make_spanned<2>(output, {7, 16}));
}
```

上例使用 `choreo::make_spanned` 函数模板，以模板参数指定 rank，并以类型化指针与 shape 列表作为函数参数。这样即为指针附加了 shape，从而构成合法的 Choreo 函数实参。由 `choreo.h` 可知，其得到 `choreo::spanned_view` 对象，仅包装 C/C++ 指针/数组与基于数组的维度列表。为便于理解，其简化定义类似：

```cpp
template <typename T, size_t Rank>
class spanned_view {
  T* ptr;                 // typed pointer to the data
  size_t dims[Rank];      // ranked dimensions
};
```

也可使用变量作为维度构造带 shape 的数据，例如：

```choreo
__co__ void foo(f32 [7, 16] input) {...}

void entry(float * input, float* output, int M, int N) {
  foo(choreo::make_spanned<2>(input, {M, N})),
}
```
本例中 `input` 的 shape 为运行时变量。若传入错误参数亦无需过度担心：进入 tileflow program 时 Choreo 会检查是否 `M == 7` 且 `N == 16`。**若不满足，程序将立即终止**，因为对 Choreo 函数中形状的一切假设均不成立。

关于 *spanned data* 输入的一个重要说明是：它们 **指向输入数据/buffer 的引用**。这意味着调用 Choreo 函数时不会发生数据拷贝，尽管从语言表面看可能类似 copy-semantics。

## Choreo 函数输出
### 返回类型与类型推导
与输入类似，*spanned data* 与 *integer* 均为合法输出类型；若不返回任何内容，也可使用 *void*。

除显式标注返回类型外，Choreo 支持对返回类型进行推断。例如：

```choreo
__co__ auto foo() {
  f32 [7, 16] result;
  // ...
  return result;
}
```
本代码将返回类型标注为 `auto`，与 C++ 中关键字相同。Choreo 可推断返回类型为 `f32 [7, 16]`。因此建议尽可能使用 `auto`。

### C++ Host：带 shape 的返回

接收 Choreo 函数输出的 C++ host 代码实际接收的是 buffer。语义上，返回须为拷贝而非引用，否则将返回被调用函数内分配对象的引用。例如：

```choreo
__co__ auto foo() {
  f32 [7, 16] output;
  // ...
  return output;  // semantically copy
}

void entry() {
  auto result = foo(); // move the 'spanned_data' to caller
  // ...
}
```

本例中 `output` 在语义上拷贝给调用方。然而在 Choreo 的实现中并不发生拷贝。返回类型为 `choreo::spanned_data`，其简化定义如下：

```cpp
template <typename T, size_t Rank>
class spanned_data {
  std::unique_ptr<T[]> ptr; // move only pointers
  size_t dims[Rank];        // ranked dimensions
};
```

与 `choreo::spanned_view` 相反，`choreo::spanned_data` 拥有其所指向的 buffer。因此 **从 Choreo 函数返回时实际上不会发生拷贝**。在编译器术语中，这称为 *Return Value Optimization*。

可使用 `.rank()`、`.shape()` 等成员函数查询 Choreo 返回值的 shape 相关信息，从而在无需过多了解 Choreo 函数细节的情况下完成集成。

## 更多问题
与 C++ 函数类似，Choreo 函数可在源码中任意定义。一个常见问题是 **function signature**：程序员可能希望将模块与定义了 Choreo 函数的代码链接。

答案是：Choreo 函数遵循 C++ calling convention，因而为标准 C++ 函数并采用 C++ name mangling。注意 **Choreo 函数不能使用 `extern "C"`**，因为其参数类型为基于 C++ template 的对象（spanned data）。

以下特性尚待支持：

- 仅有声明而无定义的 Choreo 函数。这将允许从另一模块调用 Choreo 函数。
- 带 `inline` 的 Choreo 函数。这将允许在头文件中使用 Choreo 函数。

