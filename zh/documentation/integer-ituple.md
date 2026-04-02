## 概述
在本节中，你将学习如何在 Choreo 代码中定义*整数（integer）*与整数元组（*i-tuple*），并理解其用法。

## 用于程序控制的整数
在高性能计算核函数中，整数通常用于程序控制而非算术运算。在 Choreo 的 *tileflow 程序*中，整数**仅**用于循环控制与数组索引。鉴于这些场景中取值范围有限，Choreo 为该领域专用场景提供单一整数类型——*32 位有符号整数*——以简化设计。

在 Choreo 中，**integer** 类型变量的定义方式与 C/C++ 类似：

```choreo
int a = 1;
```

由于 Choreo 可从*初始化表达式*推断类型，亦可采用另一种风格定义整数变量：

```choreo
a = 1;
```

与 C/C++ 类似，可在同一行定义多个整数：

```choreo
int a = 1, b = 2, c = 3;
d = 4, e = 5;
```

在 Choreo 中，*mdspan* 类型的形状不能用作函数参数；但整数可从主机传入 Tileflow 程序，亦可从 Tileflow 程序传入设备：

```choreo
__co__ void foo(int a) {
  // ...
  call kernel(a, 3);
}
```

与 C/C++ 不同，Choreo 不允许未初始化的整数声明，也不允许对整数重新赋值。例如：

```choreo
int a;  // error: declaration without initialization
int b = 1;
b = 3;  // error: re-assignment
```

这保证了**整数必须在声明时初始化且不可重新赋值**。在某些语境下，这些整数被称为「**不可变（immutable）**」或「**常量（constant）**」。

## 范围组与整数组
在 Choreo 中，*mdspan* 可用于表示多维形状。然而如前所述，*mdspan* 表示 **Multi-Dimensional Span**，其元素为 *Dimensional Span*，隐含取值的范围。

例如，*mdspan* `[7, 14]` 表示两个范围组成的组：`[0, 7)`、`[0, 14)`。因此 *mdspan* 并不表示整数取值的列表。然而，将整数分组在若干场景下十分有用：

- 考虑两个 mdspan 的差量时，需要多个整数；在高层语义中，该差量可能反映调整形状维度时的填充差异。
- 对形状进行分块时，多个分块因子为整数，而非范围。

这些情况促使 Choreo 引入用于分组整数取值的类型。

## 将整数组作为整数元组（I-Tuple）
### 定义 I-Tuple
在 Choreo 中，多个整数可组合为**整数元组（integer-tuple）**，简称 **i-tuple**（或 **ituple**）。可使用关键字 `ituple` 定义 *i-tuple*。下列代码展示用法：

```
ituple a = {1, 2, 3};
b = {4, 5, 6};  // utilize the type inference
```

由于 Choreo 可从*初始化表达式*推断类型，程序员常可省略 `ituple` 关键字。然而，若无显式类型标注，*ituple* 变量定义可能与 *mdspan* 定义相似，尤其对 Choreo 尚不熟悉的读者。二者区分如下：

- *ituple* 的*初始化表达式*在赋值运算符 `=` 之后（与整数类似），而 *mdspan* 在 `:` 之后初始化；
- *ituple* 的*初始化表达式*由 `{}` 括起，而非 *mdspan* 所用的 `[]`。

与 `mdspan` 类似，可在编译期对 *ituple* 强制进行秩检查：

```
ituple<3> a = {1, 2};  // error: inconsistent rank
```

### I-Tuple 上的运算
对 *i-tuple* 的运算与对 *mdspan* 类似。可使用*元素访问*操作 `()` 取得元素值，亦可整体使用 *ituple*：

```choreo
a = {3, 4};
b = {a(0), 1, a(1)};  // '()' to retrieve the element value
c = a {(0), (1), 2};  // syntax sugar
d = {a, 5, 6};        // concatenate
e = a + 1;            // addition is applied elementwise
```

注意，与 *integer* 与 *mdspan* 类似，**ituple 变量必须初始化且不可重新赋值**。

实践中，*ituple* 常与 *mdspan* 联用。例如：

```choreo
shape : [7, 18, 28];
tiling_factors = {1, 2, 4};
tiled_shape : shape / tiling_factors;
padded_shape : shape + {2, 0, 2};
```

在该代码中，mdspan 类型的 `shape` 除以 ituple 类型的分块因子，得到 `tiled_shape`。此外，`padded_shape` 由初始 `shape` 与匿名 *ituple* 相加推导。注意 *mdspan* 与 *ituple* 相加时，二者秩须一致，否则将报错。

```choreo
shape : [7, 8, 9] + {1, 2}; // error: inconsistent rank
```

### 整数与 I-Tuple 的求值
与 mdspan 类似，*integer* 与 *i-tuple* 的运行时开销极小。只要可能，其值在编译期求值，故程序员通常可忽略其代价。

### 前瞻：有界 I-Tuple
*I-tuple* 在实际 Choreo 代码中并不频繁出现；但其变体——**有界 i-tuple（bounded i-tuple）**——将 *i-tuple* 与 *mdspan* 绑定以指示可能取值范围，对构造 Choreo 循环至关重要。*有界 i-tuple* 将于后续章节详述。

## 简要小结
本节介绍了如何在 Choreo 中定义 *integer* 与 *i-tuple*。*Integer* 遵循 C/C++ 语法；*i-tuple* 的语法与 *mdspan* 类似，但可与 *mdspan* 运算以推导新值。

二者**均需要*初始化表达式*且不可重新赋值**，因而表现为**常量**或**不可变**值。事实上，Choreo tileflow 程序中定义的变量均为不可变，特定类型除外。因此，**在多数情况下赋值运算符 `=` 用于变量初始化**，程序员须留意这一点。
