## 概述
在本节中，你将学习 Choreo 中的*匿名维度（anonymous dimension）*与*符号维度（symbolic dimension）*，以及它们如何支持特定系统所需的动态形状。

## 固定形状与动态形状
固定形状的数据常用于高性能计算核函数，便于在已知维度下进行积极优化。然而，某些场景需要处理运行时方可确定的维度，即动态形状。这要求核函数具备足够的通用性以处理形状各异的输入，亦是诸多机器学习框架所强调的特性。

## 带**匿名维度**的 *mdspan*
Choreo 支持取值未知的形状维度，与许多高层机器学习语言类似。最简单的方法是在 `mdspan` 中使用**匿名维度（Anonymous Dimension）**：

```choreo
__co__ auto foo(s32 [?, 1, 2] input) { ... }
```

此处问号 `?` 表示编译期未知值；可假设为任意正整数，但该值必须在运行时提供，如下例所示：

```cpp
void bar(int * data, int dim0) {
  foo(choreo::make_spanview<3>(data, {dim0, 1, 2}));
}
```

本例中，`input` 的第一维由 `foo` 的 *spanned* 输入提供，其值在执行时来自 `bar` 的调用方。可设想该值由读取配置文件、机器学习框架中的形状推导等方式确定。

*匿名维度*在此情形下可以正常工作；但在某些场景中，它并不充分。

## 更进一步：**符号维度（Symbolic Dimension）**

Choreo 支持动态形状的另一种途径是使用**符号维度（Symbolic Dimension）**。*符号维度*具名，但在编译期同样未知。示例如下：

```choreo
__co__ auto Matmul(s32 [M, K] lhs, s32 [K, N] rhs) {
  int tile_m = 32, tile_m = 8, tile_k = 16;

  s32 [M / tile_m, K / tile_k] tiled_lhs;
  s32 [K / tile_k, N / tile_n] tiled_rhs;

  // ...
}
```
在代码中，*spanned* 输入的每一维均赋予符号名（本例为 `M, K, N`）。与*匿名维度*相比，*符号维度*方式更清晰地描述形状之间的关系，代码直观且易于维护；此外，还可利用额外信息提升代码安全性。下文进一步说明。

## 提升代码安全性
使用*符号维度*能够提升代码安全性的原因在于，Choreo 编译器可施加更全面的检查。例如，考虑 `Matmul` 函数的*匿名维度*版本：

```choreo
__co__ auto Matmul(s32 [?, ?] lhs, s32 [?, ?] rhs) {
  int tile_m = 32, tile_m = 8, tile_k = 16;

  s32 [lhs.span / {tile_m, tile_k}] tiled_lhs;  // 'mdspan' and 'ituple': rank must be same
  s32 [rhs.span / {tile_k, tile_n}] tiled_rhs;  // 'tiled_rhs' can not have a zero dimension

  // ...
}
```

Choreo 在编译期与运行期均进行检查以确保安全：

- **编译期检查**：Choreo 验证秩一致性。本例中，`lhs.span` 的秩为 `2`，与 `{tile_m, tile_k}` 的秩一致，故无问题。
- **运行期检查**：Choreo 生成代码以在运行时校验维度。例如在 `tiled_lhs` 的声明中，其形状不得出现维度 `0`，否则将产生无效的零大小缓冲区。Choreo 的*转译*过程会生成如下*目标主机代码*：

```cpp
void __choreo_transpiled_Matmul(choreo::span_view<2, choreo::s32> lhs,
                                choreo::span_view<2, choreo::s32> rhs) {
  choreo_assert(lhs.shape()[0] / 32 > 0,
                "the 0 dimension of 'lhs' may result in spanned data "
                "with a dimension value of 0.");
  // ...
}
```

此处，`choreo_assert` 在条件不满足时将中止程序执行；该早期检查有助于 Choreo 及时发现问题。

对于*符号维度*版本，Choreo 编译器在*目标主机代码*中生成额外检查：

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

除*匿名维度*版本已执行的检查外，*符号维度*版本还校验 `lhs` 与 `rhs` 之间维度的一致性，从而使代码更为安全。

因此，建议程序员使用符号维度，不仅出于易用性，亦为提高代码安全性。

## 代价与限制
为支持动态形状，Choreo 编译器在编译期必须采取保守策略。该限制使得部分编译期检查无法完成，而需在运行期执行。尽管实现力求将此类检查置于主机代码中以降低开销，与固定大小形状相比，其**仍更为保守**。程序员应留意此点，并在必要时添加用户级断言以进一步增强安全性。

*匿名维度*与*符号维度*的一个重要限制是，二者**仅能应用于参数列表中的 *spanned data***。在 Choreo 函数内部声明的缓冲区之维度不能设为动态。例如，下列代码将导致编译期错误：

```choreo
__co__ void foo() { f32 [M] d;} // compile-time error: can not applied to the spanned buffer
```

## 简要小结
本节讨论了 Choreo 中的*匿名维度*与*符号维度*，二者用于满足特定场景下的动态形状需求。与*匿名维度*相比，使用*符号维度*可通过额外的编译期检查提升代码安全性，故在易用性与安全性方面均为推荐做法。
