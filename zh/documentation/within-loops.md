## 概述

本节将介绍如何使用 `with-in` 语句构造 *bounded variable*，以及如何使用 `foreach` 语句创建循环。

## Choreo 中的循环

在 C++ 中，循环通常由 `for`、`while` 与 `do-while` 块构成。这些循环包含终止条件，并常涉及与循环控制相关的变量操作。程序员可自由定义退出条件、步进方式等，从而获得灵活的循环结构。

相比之下，Choreo 作为面向数据搬运的领域特定语言，支持的循环结构更为受限。当前 Choreo 中的循环构造形式如下：

```choreo
foreach index {
  // loop body
}
```

其中，`index` 为 *bounded variable*（上一节已介绍）。若 `index` 的上界为 `6`，则上述代码等价于：

```cpp
for (int index = 0; index < 6; ++index) {
  // loop body
}
```

尽管 *bounded variable* 看似受限，但对数据搬运任务已足够——通常不会使用负下标或越界下标搬运数据。

在 Choreo 中，`parallel-by` 语句内定义的 *bounded variable* 不可变，且不能用于 `foreach` 语句。程序员应使用 `with-in` 语句定义适用于循环的 *bounded variable*。

## **With-In** 块

### 定义 *Bounded Variable*：

`with-in` 语句可定义 *bounded integer* 或 *bounded ituple*。语法上与 `parallel-by` 类似，但仅定义 *bounded variable*，并不启动多实例执行。示例如下：

```choreo
with x in 128 {
  // 'x' is a bounded integer
}
with y in [512] {
  // 'y' is a bounded ituple
}
with index in [10, 10] {
  // 'index' is a bounded ituple
}
```

此处，`x` 表示上界为 `128` 的 bounded integer；`y` 与 `index` 均为 bounded ituple，上界分别为 `512` 与 `10, 10`。与 `parallel-by` 类似，可为 ituple 元素命名以增强可读性，或同时声明 bounded ituple 与对应的 bounded integer，如下所示：

```choreo
with {x, y} in [10, 10] {
  // x and y are explicitly named elements of the ituple
}
with index = {x, y} in [10, 10] {
  // 'x', 'y', and 'index' can be used within the block
}
```

与 `parallel-by` 相同，多个 `with-in` 声明可使用逗号分隔语法合并，例如：

```choreo
with index = {x, y} in [10, 10], idx in [100, 10] { }
```

如此可定义两个 bounded ituple。注意：`index` 与 `idx` 均只能在其后的 `with-in` 块内引用。

## `foreach` 块

### 基本语法

`with-in` 语句定义 *bounded variable*，随后可用 `foreach` 语句对其迭代。基本语法如下：

```choreo
with index in [6] {
  foreach index {
    // Perform operations with each index
  }
}
```

本例中，`foreach` 块迭代 6 次，**iteration variable（迭代变量）** `index` 的取值从 `0` 递增到 `5`。

亦可对 *bounded ituple* 进行迭代。例如：

```choreo
with index = {x, y} in [6, 17] {
  foreach index { }
}
```

该代码定义 *bounded ituple* `index`。对 `index` 迭代等价于如下嵌套循环结构：

```cpp
for (int x = 0; x < 6; ++x)
  for (int y = 0; y < 17; ++y) { }
```

注意循环嵌套顺序：*bounded ituple* 内 *bounded variable* 的 **从左到右** 顺序对应循环的 **从外到内** 嵌套。该顺序对正确构造代码行为有时至关重要。

此外，该规则同样适用于 `foreach` 语句之后的 *逗号分隔 bounded variable 列表*：

```choreo
with index = {x, y} in [6, 17], iv in [128] {
  foreach iv, index { }
}
```
本代码定义两个 bounded variable。随后的 `foreach` 块等价于多级嵌套循环：

```cpp
for (int iv = 0; iv < 128; ++iv)
  for (int x = 0; x < 6; ++x)
    for (int y = 0; y < 17; ++y) { }
```

### 语法糖


```choreo
foreach x in 128 { }
foreach idx in [10, 20] { }
foreach {y, z} in [8, 16] { }
```

上述代码等价于：

```choreo
with x in 128 {
  foreach x { }
}

with idx in [10, 20] {
  foreach idx { }
}

with {y, z} in [8, 16] {
  foreach y, z { }
}
```

### 从 Bounded Integer 派生循环

某些场景（如流水线化数据搬运）需要调整循环迭代次数。在 Choreo 中，可在 `foreach` 语句中通过从 *bounded integer* 派生循环实现。例如：

```choreo
with {x, y} in [6, 17] {
  foreach x, y(1::) { }
}
```

此处使用 **Range Expression（范围表达式）** `y(1::)` 派生循环，对应等价的 C/C++ 代码为：

```cpp
for (int x = 0; x < 6; ++x)
  for (int y = 1; y < 17; ++y) { }
```

如代码所示，`y` 循环从 `1` 开始。*range expression* 由 *bounded variable* 及紧随其后的括号内三个冒号分隔的整数值构成，形式如下：

```
  bounded-variable(lower-offset:upper-offset:stride)
```

由此从 `bounded-variable` 派生循环，其中 `bounded-variable` 作为循环的 *iteration variable*。具体而言：

- *iteration variable* 的初值为 `lower-offset` 加上 `bounded-variable` 的下界。由于 Choreo 中所有 bounded integer 的下界均为 `0`，`lower-offset` 即设定循环初值。
- 当 *iteration variable* 大于或等于 `bounded-variable` 的上界加上 `upper-offset` 时循环终止。`upper-offset` 通常为负值。
- 每次迭代结束时，*iteration variable* 按 `stride` 递增。

因此，在上例中 `y(1:-1:2)` 产生类似 `for (y = 0 + 1; y < 17 - 1; y += 2)` 的循环。若 *range expression* 中某字段未指定，则采用默认值：`lower-offset` 与 `upper-offset` 默认为 `0`，`stride` 默认为 `1`。

注意：*range expression* 仅适用于 *bounded variable*。对 *bounded ituple* 使用 *range expression* 会在编译期报错。

## Bounded Variable 的取值

理解与 bounded variable 相关的取值至关重要。bounded variable 有两个关键取值：

- **Current Value（当前值）**：
    - 在 `foreach` 语句内，当前值由循环迭代决定，因为 bounded variable 充当迭代变量。
    - 在 `foreach` 之外，bounded variable 的当前值恒为 `0`。

- **Upper-Bound Value（上界值）**：在 `with-in` 或 `parallel-by` 语句中指定。

程序员在 `foreach` 之外使用 bounded variable 的当前值时可能遇到问题，尤其在循环结束之后。按定义，bounded variable 的当前值仅在 `foreach` 循环内可变。以下代码说明该点：

```choreo
with x in 6 {
  // x's current value is 0
  foreach x {
    // x's current value is either 0, 1, 2, ..., 5
  }
  // x's current value is 0, NOT 6
}
```

## 小结
本节说明如何在 Choreo 中使用 `with-in` 与 `foreach` 语句定义并迭代 *bounded variable*——这对数据搬运任务至关重要。我们介绍了这些构造的语法与行为，包括如何应用 range 操作以修改循环迭代，以及理解 bounded variable 的当前值与上界值的重要性。

循环派生部分对实现多缓冲数据搬运尤为重要，亦是构建高性能 kernel 的基础，将在后续优化章节中进一步介绍。
