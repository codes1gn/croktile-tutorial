## 概述
本节将介绍 Choreo 中的 *anonymous dimension* 与 *symbolic dimension*，以及它们如何支撑某些系统所需的 dynamic shape。

## 固定形状与动态形状
固定形状的数据常见于高性能计算 kernel，可在已知维度下进行积极优化。然而，部分场景需要处理运行时维度决定的形状，即 dynamic shape。这要求 kernel 具备通用性以处理不同形状的输入——许多机器学习框架均强调该能力。

## 带 **Anonymous Dimension** 的 *mdspan*
Choreo 支持取值未知的形状维度，与许多高层机器学习语言类似。最简单的方法是在 `mdspan` 中使用 **Anonymous Dimension**：

```choreo
__co__ auto foo(s32 [?, 1, 2] input) { ... }
```

此处，问号 `?` 表示编译期未知值，可假设为任意正整数。但该值必须在运行时提供，如下例所示：

```cpp
void bar(int * data, int dim0) {
  foo(choreo::make_spanview<3>(data, {dim0, 1, 2}));
}
```

本例中，`input` 的第一维由 `foo` 的 *spanned* 输入提供，其值在执行时由 `bar` 的调用方传入。可假定该值通过读取配置文件、在机器学习框架中由形状推导等方式确定。

*anonymous dimension* 在此类情形下可以正常工作。但在某些场景中，它并不充分。

## 更进一步：**Symbolic Dimension**

Choreo 中支持 dynamic shape 的另一途径是使用 **Symbolic Dimension**。*Symbolic dimension* 具名，但在编译期同样未知。示例如下：

```choreo
__co__ auto Matmul(s32 [M, K] lhs, s32 [K, N] rhs) {
  int tile_m = 32, tile_m = 8, tile_k = 16;

  s32 [M / tile_m, K / tile_k] tiled_lhs;
  s32 [K / tile_k, N / tile_n] tiled_rhs;

  // ...
}
```
代码中，各维度为 *spanned* 输入赋予符号名（本例为 `M, K, N`）。与 *anonymous dimension* 相比，*symbolic dimension* 方式更清晰地描述形状之间的关系，代码直观且易于维护。此外，借助额外信息还可提升代码安全性。下文对此作进一步说明。

## 代码安全性的提升
使用 *symbolic dimension* 能提升代码安全性的原因在于，Choreo 编译器可施加更全面的检查。例如，考虑 `Matmul` 函数的 *anonymous dimension* 版本：

```choreo
__co__ auto Matmul(s32 [?, ?] lhs, s32 [?, ?] rhs) {
  int tile_m = 32, tile_m = 8, tile_k = 16;

  s32 [lhs.span / {tile_m, tile_k}] tiled_lhs;  // 'mdspan' and 'ituple': rank must be same
  s32 [rhs.span / {tile_k, tile_n}] tiled_rhs;  // 'tiled_rhs' can not have a zero dimension

  // ...
}
```

Choreo 在编译期与运行期均进行检查以保证安全：

- **Compile-time Checks**：Choreo 校验 rank 一致性。本例中，`lhs.span` 的 rank 为 `2`，与 `{tile_m, tile_k}` 的 rank 一致，故无问题。
- **Runtime Checks**：Choreo 生成代码在运行期校验维度。例如，在 `tiled_lhs` 的声明中，其形状不得出现维度为 `0` 的情况，否则将产生无效的零大小 buffer。Choreo 的 *transpilation* 过程生成如下 *target host code*：

```cpp
void __choreo_transpiled_Matmul(choreo::span_view<2, choreo::s32> lhs,
                                choreo::span_view<2, choreo::s32> rhs) {
  choreo_assert(lhs.shape()[0] / 32 > 0,
                "the 0 dimension of 'lhs' may result in spanned data "
                "with a dimension value of 0.");
  // ...
}
```

此处，若条件不满足，`choreo_abort` 函数将终止程序执行。该早期检查有助于 Choreo 及时发现问题。

对于 *symbolic dimension* 版本，Choreo 编译器在 *target host code* 中生成额外检查：

```cpp
void __choreo_transpiled_Matmul(choreo::span_view<2, choreo::s32> lhs,
                                choreo::span_view<2, choreo::s32> rhs) {
  // additional check happens only when using symbolic dimensions
  choreo_assert(lhs.shape()[1] == rhs.shape()[0],
                "dimension 'K' is not consistent.");

  choreo_assert(lhs.shape()[0] / 32 > 0,
                "the 0 dimension of 'lhs' may result in spanned data "
                "with a dimension value of 0.");
  // ...
}
```

除 *anonymous dimension* 版本已执行的检查外，*symbolic dimension* 版本还校验 `lhs` 与 `rhs` 之间维度的一致性，从而使代码更安全。

因此，建议程序员使用 symbolic dimension，不仅为了易用，亦为了增强代码安全性。

## 代价与局限
为支持 dynamic shape，Choreo 编译器在编译期必须采取保守策略。该限制使部分编译期检查无法完成，部分检查需改在运行期执行。尽管实现力求将这些检查置于 host 代码中以降低开销，**与固定尺寸形状的代码相比，它们仍然更为保守**。程序员应留意此点，必要时可添加用户级断言以进一步增强安全性。

*anonymous dimension* 与 *symbolic dimension* 的一个显著局限是，二者**仅能应用于参数列表中的 *spanned data***。在 Choreo 函数内声明的 buffer 的维度不能设为动态。例如，以下代码将产生编译期错误：

```choreo
__co__ void foo() { f32 [M] d;} // compile-time error: can not applied to the spanned buffer
```

## 小结
本节讨论了 Choreo 中的 *anonymous dimension* 与 *symbolic dimension*，二者支撑特定场景所需的 dynamic shape。与 *anonymous dimension* 相比，使用 *symbolic dimension* 可通过额外编译期检查提升代码安全性，故在易用性与安全性方面均为推荐做法。
