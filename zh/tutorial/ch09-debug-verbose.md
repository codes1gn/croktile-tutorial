# 调试与详细输出：打印、RTTI 与 GDB

内核能够编译并启动且无报错，但结果却是错的。接下来怎么办？盯着 warp 特化流水线代码指望 bug 自己跳出来并非可行策略。你需要在出错之处观察内核**实际**在做什么——中间值、索引计算、张量形状。

本章介绍鳄霸内置的检查手段：用于设备端输出的 **`print`** 与 **`println`**，用于静态形状查询的**编译期打印变体**（`print!`/`println!`），使鳄霸类型对调试器可见的 **debug RTTI**，以及借助 **`cuda-gdb`** 逐步缩小问题范围的实用流程。

## `print` 与 `println`

最简单的调试工具。`print` 将其参数写入标准输出；`println` 行为相同，但会追加换行。二者均可在 `__co__` 函数内使用，并在生成的 CUDA 中发出设备端 `printf` 调用：

```choreo
__co__ void inspect(s32 [4, 8] data) {
  foreach i in [data.span] {
    println("element ", i, " = ", data.at(i));
  }
}
```

每次 `println` 调用接受逗号分隔的字符串字面量与表达式混合。字符串按原样打印；表达式求值后按其运行时值打印。各线程之间的输出顺序**非确定**——GPU 的 printf 缓冲区异步刷新，因此不同线程的行会不可预测地交错。

针对某一检查——例如瓦片索引 `(3, 5)` 是否算对了部分和——请用条件保护打印：

```choreo
parallel {px, py} by [8, 16] : block
  parallel {qx, qy} by [16, 16] : thread {
    // ... compute ...
    if (px == 0 && py == 0 && qx == 3 && qy == 5) {
      println("partial sum = ", accum);
    }
  }
```

若无此保护，每个线程都会打一行——对大网格而言可达成千上万行，其中多数并非你关心的信息。

## `print!` 与 `println!`：编译期查询

带感叹号的变体（`print!`、`println!`）在**编译期**执行，而非运行时。它们打印到编译器输出，适用于在不启动内核的情况下检查形状、范围与类型信息：

```choreo
__co__ void check_shapes(f32 [3, 2] b) {
  print!("shape of b: ");
  println!("b.span = ", b.span);
}
```

编译该文件时，编译器会输出：

```
shape of b: b.span = {3, 2}
```

无需 GPU。字符串字面量会拼接（`"wor" "ld!"` 变为 `"world!"`），便于由片段拼装诊断信息。可用 `print!` 在完整内核启动之前，确认 `chunkat` 与 `subspan` 是否产生你预期的瓦片尺寸，以免浪费时间。

## Debug RTTI 与 GDB

当 `print`/`println` 不足以定位问题——你需要单步执行、检查寄存器状态，或查看复杂数据结构——鳄霸支持 **debug RTTI**（Runtime Type Information），使其类型对 `cuda-gdb` 可见。

使用 `-g -O0` 编译以启用 debug symbols 并关闭优化：

```bash
croktile kernel.co -g -O0 -o kernel_debug
```

生成代码中包含鳄霸类型的 RTTI 结构：

| 鳄霸类型 | GDB 类型 | 字段 |
|--------------|----------|--------|
| 带形状张量（`s32 [M, N]`） | `choreo::rtti::spanned<int, 2>` | `.span.data[]`（维度），`.stride.data[]`（步长），`.data`（指针） |
| 索引元组 | `choreo::rtti::bounded_ituple<N>` | `.data[]`（取值），`.ub[]`（上界） |
| 整数元组 | `choreo::rtti::ituple<N>` | `.data[]`（取值） |
| 多维 span | `choreo::rtti::mdspan<N>` | `.data[]`（范围） |

在调试编译的内核上进行的 GDB 会话示例：

```bash
cuda-gdb -q ./kernel_debug
(gdb) break my_kernel
(gdb) run
(gdb) ptype __dbg_lhs
type = struct choreo::rtti::spanned<int, 2>
(gdb) print __dbg_lhs.span.data[0]
$1 = 32
(gdb) print __dbg_lhs.span.data[1]
$2 = 64
(gdb) print __dbg_lhs.data != 0
$3 = true
```

变量名上的 `__dbg_` 前缀由编译器生成——它使鳄霸变量与生成的 C++ 中间表示一并可被 GDB 检视。你可以检查张量维度、步长与数据指针，并在生成代码中单步执行，观察 `foreach` 循环的哪一次迭代产生了错误值。

## 调试流程

当内核结果错误时，请**系统性地**缩小问题范围：

**1. 检查形状。** 在编译期使用 `println!`，确认所有 `chunkat`、`subspan` 与 `span` 表达式是否产生你预期的瓦片尺寸。形状类错误最为常见——范围不一致会在不知不觉中读写到错误的内存区域。

**2. 检查单个瓦片。** 在 K 循环内加入受保护的 `println`，仅对块 `(0, 0)` 与某一个线程触发。在每次 K 迭代后打印累加器值。将该输出元素与手算参考值对比。

**3. 检查同步。** 若仅在 K 较大时数值错误，应怀疑事件顺序。在生产者与消费者两侧打印 `iv_k` 与 `stage`，确认二者访问同一序列。常见 bug：消费者的 `trigger empty[stage]` 触发过早，使生产者在消费者仍在读取时覆盖缓冲区。

**4. 检查布局。** 若结果呈某种错误模式（例如每第 16 个元素正确、其余偏移），应怀疑 `mma.row.row` 中行主序与列主序不一致，或 `tma.copy.swiz<N>` 与 `mma.load.swiz<N>` 之间的 swizzle 模式不一致。

**5. 指针类问题用 GDB。** 若 `println` 显示的数值不合理（NaN、巨大整数、本应为非零却为零），请使用 `-g -O0` 编译并在 `cuda-gdb` 中单步执行。检查张量指针（`__dbg_x.data`）是否指向有效内存。

## 性能说明

`print`/`println` 代价很高。每次调用都经全局 printf 缓冲区串行化，会严重扭曲吞吐测量结果。debug RTTI 结构也会增加寄存器压力并阻碍编译器优化。基准测试前应移除所有打印并去掉 `-g -O0` 选项。

一种折中做法：将打印置于 `#ifdef DEBUG_PRINT` 宏之后（第 8 章），以便从命令行开关而无需改源码：

```choreo
#ifdef DEBUG_PRINT
  println("tile_k=", iv_k, " accum=", mc);
#endif
```

使用 `croktile kernel.co -DDEBUG_PRINT` 编译以启用，生产运行则不带该宏。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `print(expr, ...)` | 设备端 printf（无换行） |
| `println(expr, ...)` | 设备端 printf（带换行） |
| `print!("str", ...)` | 编译期打印（无换行，无需启动） |
| `println!("str", ...)` | 编译期打印（带换行，无需启动） |
| `-g -O0` | 用于 debug symbols 与关闭优化的编译选项 |
| `cuda-gdb` | 用于单步调试 GPU 内核的 NVIDIA 调试器 |

至此，鳄霸教程告一段落。你从第 1 章的单元素加法出发，学习了如何按块搬运数据（第 2 章）、在数千线程上并行（第 3 章）、启用张量核心（第 4 章）、划分 warp 角色（第 5 章）、对加载与计算做流水线（第 6 章）、使用硬件 TMA 并处理不规则访问（第 7 章）、在需要时回落到原始 C++（第 8 章），以及出错时如何调试（第 9 章）。

最有成效的下一步是打开 `croktile/benchmark/` 目录下的某个基准内核，将其各区域对应到本章内容，修改一个常量，重新编译并测量。小而刻意的修改优于大规模重写——当某种配置不再合法时，`#error` 守卫会在你追逐 GPU 上莫名其妙的错误答案之前很久就告诉你。
