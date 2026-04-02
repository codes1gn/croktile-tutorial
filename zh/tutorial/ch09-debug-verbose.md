# 调试与详细输出：打印、断言与 GDB

GPU 内核是**不透明系统**：成千上万线程并发执行，共享内存在主机侧不可见，且当结果错误时，没有栈回溯能指向出错行。这种不透明性并非 GPU 独有——凡是程序员无法直接观察中间状态的系统都会面临同样挑战：分布式系统（丢失的消息在哪？）、嵌入式固件（哪个中断处理程序损坏了寄存器？）、优化编译器（哪趟优化破坏了语义？）。

通用的调试纪律在各地相同：**系统地缩小搜索空间**，从廉价检查到昂贵检查。先做静态分析与编译期断言（几乎无成本，可捕获整类缺陷）。再转向有针对性的运行时探针（成本低，能定位问题）。仅当更廉价的手段已缩小嫌疑范围时，才动用交互式调试器。

Croqtile（鳄霸）在每一层都提供工具：**编译期形状打印**（`print!`、`println!`），可在不启动内核的情况下核对分块维度；**运行时断言**（`assert`）用于不变量检查；**带条件的设备端 `println`** 用于在特定线程上做运行时检视；面向 `cuda-gdb` 的 **debug RTTI**；以及 **详细模式**（`-v`），用于查看编译器在底层实际调用了哪些外部命令。

## `print!` 与 `println!`：编译期检视

带感叹号的变体（`print!`、`println!`）在**编译期**执行，而非运行时。它们输出到编译器日志，适用于在不启动内核的情况下检视形状、范围与类型信息：

```choreo
__co__ void check_shapes(f32 [3, 2] b) {
  print!("shape of b: ");
  println!("b.span = ", b.span);
}
```

编译该文件时，编译器会输出：

```
shape of b: b.span = [3, 2]
```

无需 GPU。字符串字面量会被拼接（`"wor" "ld!"` 变为 `"world!"`）。在投入完整内核启动之前，可用 `print!` / `println!` 验证 `chunkat` 与 `subspan` 是否产生你期望的分块尺寸。

## `print` 与 `println`：运行时设备端输出

不带感叹号时，`print` 与 `println` 会在生成的 CUDA 中发出设备端 `printf` 调用：

```choreo
__co__ void inspect(s32 [4, 8] data) {
  foreach i in [data.span] {
    println("element ", i, " = ", data.at(i));
  }
}
```

每次调用接受由逗号分隔的字符串字面量与表达式混合。各线程之间的输出顺序**不确定**——GPU 的 `printf` 缓冲区异步刷新。

**对输出加条件。** 针对特定检查，请用条件守卫打印：

```choreo
parallel {px, py} by [8, 16] : block
  parallel {qx, qy} by [16, 16] : thread {
    // ... compute ...
    if (px == 0 && py == 0 && qx == 3 && qy == 5) {
      println("partial sum = ", accum);
    }
  }
```

若无守卫，则每个线程一行——对大网格会产生成千上万行输出。

## `assert`：运行时不变量检查

Croqtile 的内建 `assert` 在运行时检查不变量；若失败则中止内核并附带消息：

```choreo
assert(stage < MATMUL_STAGES, "stage index out of bounds");
```

在设备上，这会编译为先 `printf` 消息再 `abort`。可用断言尽早捕获越界索引、空指针及其他「本不应发生」的情形。

## 详细模式：`-v`

向编译器传入 `-v`（或 `--verbose`）可查看其调用的外部命令——哪个预处理器、哪个代码生成器、哪次 `nvcc` 调用：

```bash
croqtile kernel.co -v -o kernel
```

当你怀疑编译器向 `nvcc` 传错了标志，或需要查看生成文件的确切路径以便手动检查时，这很有用。

## 运行时检查：`-rtc`

编译器支持通过 `-rtc`（或 `--runtime-check`）进行分级运行时检查：

```bash
croqtile kernel.co -rtc=high -o kernel
```

级别：`none`、`entry`（默认）、`low`、`medium`、`high`、`all`。每一级是上一级的超集。开发阶段可使用 `high` 或 `all`，生产环境使用 `none`。

## 调试大量依赖 MMA 的内核

若错误答案来自 Tensor Core 路径，应**先怀疑布局**——行主序与列主序，以及右操作数在内存中是 `[N, K]` 还是 `[K, N]`。再检查**索引**（你用 `.at` / `chunkat` 绑定的 `block_m`、`block_n` 与 K 分块）。若引入了异步拷贝或拆分了生产者与消费者 warp，再检查**异步顺序**。

常见错误包括在分阶段数据实际为列主序时误标 `mma.row.row`，或使用与 MMA 分块几何不对齐的 `chunkat` 索引。若使用打乱加载（`tma.copy.swiz` / `mma.load.swiz`），失配往往表现为误差中的规则模式（例如每第十六个元素正确，其余错位）。

## Debug RTTI 与 `cuda-gdb`

当 `print` / `println` 仍不足时，Croqtile 支持 debug RTTI（Runtime Type Information），使其类型对 `cuda-gdb` 可见。

使用 `-g` 编译以启用调试符号：

```bash
croqtile kernel.co -g -o kernel_debug
```

生成代码包含 Croqtile 类型的 RTTI 结构：

| Croqtile 类型 | GDB 类型 | 字段 |
|--------------|----------|------|
| 带形状张量（`s32 [M, N]`） | `choreo::rtti::spanned<int, 2>` | `.span.data[]`（维度），`.stride.data[]`（步长），`.data`（指针） |
| 索引元组 | `choreo::rtti::bounded_ituple<N>` | `.data[]`（取值），`.ub[]`（上界） |
| 整数元组 | `choreo::rtti::ituple<N>` | `.data[]`（取值） |
| 多维 span | `choreo::rtti::mdspan<N>` | `.data[]`（范围） |

**示例会话：**

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

变量名上的 `__dbg_` 前缀由编译器生成——它使 Croqtile 变量与生成的 C++ 中间代码一并可见。

## 综合应用

![Debugging workflow: shapes -> one tile -> sync -> layout -> GDB](../assets/images/ch09/fig1_debug_workflow_dark.png#only-dark)
![Debugging workflow: shapes -> one tile -> sync -> layout -> GDB](../assets/images/ch09/fig1_debug_workflow_light.png#only-light)

由廉价到昂贵依次排查：`println!`（编译期形状）→ 带条件的 `println` 只看一块（运行时数值）→ 事件/阶段打印（同步）→ 模式分析（布局）→ `cuda-gdb`（指针类缺陷）。运行时打印用 `#ifdef` 守卫，使之在生产构建中消失：

```choreo
#ifdef DEBUG_PRINT
  println("tile_k=", iv_k, " accum=", mc);
#endif
```

```bash
croqtile kernel.co -DDEBUG_PRINT -o kernel    # 调试构建
croqtile kernel.co -o kernel                   # 生产构建
```

## 小结

| 工具 | 层级 | 代价 |
|------|------|------|
| `print!` / `println!` | 编译期 | 几乎无——无需启动内核 |
| `assert(expr, "msg")` | 运行时 | 低——违反时中止 |
| `print` / `println` | 运行时 | 中——经 `printf` 串行化 |
| `-rtc=high` | 运行时 | 中——边界检查 |
| `-v` / `--verbose` | 编译器 | 几乎无——显示子进程调用 |
| `-g` + `cuda-gdb` + RTTI | 运行时 | 高——调试符号，无优化 |

你从第 1 章的元素级加法起步，依次学习了数据搬运（第 2 章）、并行（第 3 章）、Tensor Core（第 4 章）、控制流（第 5 章）、流水线（第 6 章）、TMA（第 7 章）、C++ 互操作（第 8 章）以及调试（本章）。下一步可打开 `croqtile/benchmark/` 目录中的基准内核，将其各区域对应到本章内容，修改一个常量，重新构建并测量。
