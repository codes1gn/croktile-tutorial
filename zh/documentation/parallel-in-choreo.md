## 概述
为充分利用并行硬件，在 Choreo 中编写并行代码至关重要。本节将引导你使用 `parallel-by` 块在 Choreo 中实现并行代码。


## 并行执行块
在 Choreo 中，并行运行的代码置于 *parallel execution block（并行执行块）* 中，用以表明存在多个相同代码实例同时执行。

### 基本语法：`parallel-by`

`parallel-by` 块的基本语法如下：

```choreo
parallel p by 6 {
  // SPMD (Single Program, Multiple Data) code
}
```
将 `{}` 内的代码称为 SPMD 风格的 **parallel execution block（并行执行块）**，各代码实例称为 **parallel thread（并行线程）**。
本例中：

- **`parallel`**：该关键字启动 *parallel execution block*。
- **`p`**：**parallel variable（并行变量）**。其值为整数，用于标识当前 *parallel thread*。各 *parallel thread* 执行相同代码，但 *parallel variable* `p` 的取值不同。
- **`by 6`**：表示共有 6 个 *parallel thread*。因此，*parallel variable* `p` 在不同线程上取值范围为 `0` 至 `5`。

某些场景下，程序员在简单并行构造中并不需要显式的 *parallel variable*，可以省略：

```choreo
parallel by 2 { ... }
```

上述写法会启动两个并行线程执行。但两个线程无法针对不同数据工作，而 SPMD 程序在 data-parallel 处理中通常需要这一点。

### 多级并行

在 Choreo 中可定义多级并行。示例如下：

```choreo
parallel p by 2 {
  // parallel-level-0
  parallel q by 12 {
    // parallel-level-1
  }
}
```

该代码定义两级并行，取值分别为 `2` 与 `12`，即总共启动 `2 x 12 = 24` 个并行线程完成该任务。每个线程由 `p` 与 `q` 的唯一组合标识。在 Choreo 中，将 `p` 所在层级称为 *parallel-level-0*，将 `q` 所在层级称为 *parallel-level-1*。熟悉 CUDA 的读者可将 *parallel-level-0* 对应 CUDA 的 **grid** 层级，*parallel-level-1* 对应 CUDA 的 **block** 层级。

多级并行概念源于具有层次化存储的硬件架构。程序员可观察到的一个关键差异是 **synchronization cost（同步开销）**。例如，在 CUDA 中，同一 **block** 内线程的同步开销远低于跨 **grid** 的线程。该层次化设计促使程序员将最需要同步（分歧最大）的代码放在 **block** 层级，以提升整体性能。

在 Choreo 语法中，可在单行中定义多级并行：

```choreo
parallel p by 2, q by 12 { ... }
```

其与上一示例的并行化效果相同，采用 **逗号分隔** 的 *parallel-by* 表达式，假定两级并行之间无中间代码。

### 子级并行

如 *CUDA* 等目标允许将单一并行层级划分为最多三个子级。在 Choreo 中，对此类语法使用 `ituple` 与 `mdspan`：

```choreo
parallel {px, py, pz} by [1, 1, 2] {
  // higher synchronization cost
  // ...
  parallel t_index = {qx, qy} by [3, 4] {
    // lower synchronization cost
    // ...
  }
  // ...
}
```

本例定义两级并行。第一级（*level-0*）细分为三个子级，由 `{px, py, pz}` 形式的 `ituple` 标注；第二级（*level-1*）细分为两个子级，由 `{qx, qy}` 形式的 `ituple` 标注，或整体记为 `t_index`。对每个子级，并行数量与 `mdspan` 中对应维度的取值一致。

子级概念源自 *CUDA* 的 GPU workload 管理，常涉及二维或三维数据。注意：`ituple` 中第一个元素表示 **least significant parallel variable（最低有效并行变量）**。Choreo 采用该顺序以与 *CUDA* 开发者习惯一致，而非出于内部一致性考虑。

### Bounded Variable（有界变量）

在 `parallel-by` 块中，`parallel` 关键字之后定义的变量称为 **Bounded Variable（有界变量）**。具体而言：

- 若变量为 *integer*，则称为 **bounded integer（有界整数）**。
- 若变量为 *ituple*，则称为 **bounded ituple（有界 ituple）**。

该术语表明变量不仅具有取值，还具有定义好的上下界。例如，在语句 `parallel p by 6` 中，`p` 为 *bounded integer*，其界为 `[0, 6)`，其中 `6` 为不包含的上界。

有界变量的概念将在后续章节中进一步展开，并详细说明其用法。

## 含义与推论

### 异构性：转译为 Kernel Launch

对于 *CUDA/Cute* 等目标，`parallel-by` 块不仅指定并行线程数量，还界定 host 与 device 代码的边界，利用 Choreo 管理异构性的能力。具体而言：

- **`parallel-by` 块之外**的代码转译为 **host code**。
- **块内**的代码转译为 **device code**，如下所示：

```choreo
__co__ void foo(...) {
  // Generate host code
  // ...
  parallel p by 6 { // Kernel launch
    // Generate device code
    // ...
  }
  // Generate host code
  // ...
}
```

熟悉 *CUDA* 的读者可将此类比为在 `parallel-by` 语句处发生 **Kernel Launch**。不过，Choreo 以统一的 SPMD 并行模型抽象这些细节，使开发者可专注于算法而无需顾虑底层异构细节。

### 对存储说明符的限制

由于 Choreo 中的 `parallel-by` 构造与硬件相关（包括异构性与存储层次），它受目标平台约束。

例如，对于 *CUDA/Cute* 目标，*spanned* buffer 必须在 *parallel execution block* 内标注恰当的 storage specifier。原因是目标平台仅允许 device 代码分配 *scratchpad memory*，如 *shared* 与 *local* memory。反之，*parallel execution block* 之外的代码只能声明 *global* 或默认 host memory，由目标平台允许 host 代码管理。

此外，*shared* 与 *local* memory 类型的生命周期仅持续于 kernel launch 期间。因此，无法在 *parallel execution block* 之外引用它们。该约束保证这些内存类型的生命周期得到正确管理。

### 小结

本节探讨了 Choreo 在异构计算环境中处理并行性的方式，重点包括 `parallel-by` 块语法以及多级与子级并行的管理；亦说明了 Choreo 如何抽象 kernel launch 并施加内存管理约束，从而有效利用并行硬件。

然而，并行虽能提升效率，循环仍不可或缺。下一节将介绍如何在 Choreo 中构造循环。
