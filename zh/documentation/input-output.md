## 概述
本节介绍 Choreo 函数的参数与返回值约定，以及 C++ 主机侧如何传递带形状的数据并接收结果。

## Choreo 函数的输入
### 输入参数
对于 Choreo 函数，仅接受两类输入参数：*spanned data* 与 *integer*。

与任意 C++ 程序类似，参数可以具名或匿名：

```choreo
__co__ void foo(f32 [7] a, int) {...}
```

本例中第二个参数未命名，因而在 Choreo 函数内无法引用；该代码仍然有效。


### C++ 主机：带形状的数据实参
C/C++ 指针/数组与 Choreo *spanned data* 之间的一个明显差异在于，*spanned data* 带有形状。因此，调用 Choreo 函数时有必要将指针或数组转换为带形状的形式。

一种直接方法是使用 `choreo.h` 中提供的 Choreo API。尽管在 Choreo 编译环境中不必显式 `#include choreo.h`，该头文件会包含于所有 Choreo-C++ 程序中。

```choreo
__co__ void foo(f32 [7, 16] input, f32 [7, 16] output) {...}

void entry(float * input, float* output) {
  // foo(input, output); <-- result in error since missing shape

  foo(choreo::make_spanned<2>(input, {7, 16}),
      choreo::make_spanned<2>(output, {7, 16}));
}
```

上例使用函数模板 `choreo::make_spanned`，以秩为模板实参，并将类型化指针与形状列表作为函数参数。由此为指针附加形状，从而构成合法的 Choreo 函数实参。由 `choreo.h` 可知，其得到 `choreo::spanned_view` 对象，该类仅包装 C/C++ 指针/数组与基于数组的形状列表。为便于理解，其简化定义类似如下：

```cpp
template <typename T, size_t Rank>
class spanned_view {
  T* ptr;                 // typed pointer to the data
  size_t dims[Rank];      // ranked dimensions
};
```

亦可使用变量作为维度以构造带形状的数据，例如：

```choreo
__co__ void foo(f32 [7, 16] input) {...}

void entry(float * input, float* output, int M, int N) {
  foo(choreo::make_spanned<2>(input, {M, N})),
}
```
本例中 `input` 的形状由运行时变量决定。若参数错误亦无需过度担忧：进入 tileflow 程序时，Choreo 会检查 `M == 7` 与 `N == 16` 是否成立。**若不成立，程序将立即终止**，因为 Choreo 函数中关于形状的全部假设均不成立。

关于 *spanned data* 输入须注意：它们**指向输入数据/缓冲区的引用**。这意味着调用 Choreo 函数时不会发生数据拷贝，尽管从语言表面看可能类似按值拷贝语义。

## Choreo 函数的输出
### 返回类型与类型推导
与输入类似，*spanned data* 与 *integer* 均为合法输出类型；若无返回值，亦允许使用 *void*。

除显式标注返回类型外，Choreo 支持对返回值进行类型推断。例如：

```choreo
__co__ auto foo() {
  f32 [7, 16] result;
  // ...
  return result;
}
```
该代码将返回类型标注为 `auto`，与 C++ 中关键字用法相同。Choreo 可将返回类型推断为 `f32 [7, 16]`。因此，在可行时建议使用 `auto`。

### C++ 主机：形状与返回

接收 Choreo 函数输出的 C++ 主机代码实际收到的是缓冲区。语义上，返回值须按拷贝而非引用传递，否则将返回被调函数内分配对象的引用。例如：

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

本例中，`output` 语义上被拷贝给调用方。然而在 Choreo 的实现中，实际不发生拷贝。返回类型为 `choreo::spanned_data`，其简化定义如下：

```cpp
template <typename T, size_t Rank>
class spanned_data {
  std::unique_ptr<T[]> ptr; // move only pointers
  size_t dims[Rank];        // ranked dimensions
};
```

与 `choreo::spanned_view` 相反，`choreo::spanned_data` 拥有其所指向的缓冲区，故**从 Choreo 函数返回时实际上不发生拷贝**。用编译器术语表述，这称为*返回值优化（Return Value Optimization）*。

可使用 `.rank()`、`.shape()` 等成员函数查询 Choreo 返回值的形状相关信息，从而在不必过深了解 Choreo 函数细节的情况下完成编程。

## 更多问题
与 C++ 函数类似，Choreo 函数可在源码中任意定义。一个常见疑问涉及**函数签名**，尤其当程序员希望将模块与定义了 Choreo 函数的翻译单元链接时。

答案是：Choreo 函数遵循 C++ 调用约定，因而为标准 C++ 函数并采用 C++ 名称修饰（name mangling）。注意 **`extern "C"`** 不能用于 Choreo 函数，因其参数类型为基于 C++ 模板的对象（spanned data）。

以下特性尚待支持：

- 仅有声明而无定义的 Choreo 函数：从而可从另一模块调用 Choreo 函数。
- 使用 `inline` 的 Choreo 函数：从而可在头文件中使用 Choreo 函数。

