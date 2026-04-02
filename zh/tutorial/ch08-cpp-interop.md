# C++ 互操作：内联代码与预处理器

鳄霸处理分块、流水线和内存层级，让你不必在 CUDA 中逐一拼写。大多数情况下这正是你想要的。但偶尔你需要 DSL 没有封装的东西：某条特定的 PTX 指令、一个守卫空片段的提前返回，或者在 `__co__` 函数体与宿主 C++ 之间共享的编译期常量。

本章介绍两条逃生通道：**`__cpp__`**——将原始 C++ 逐字注入生成的输出；以及**鳄霸预处理器**——`#define`、`#if`/`#ifdef`/`#else`/`#endif` 和 `#error`——用于宏和条件编译。

## `__cpp__`：逐字 C++ 注入

`__cpp__` 接受一个字符串字面量，将其逐字粘贴到生成的 CUDA 或 C++ 文件中。你放入其中的内容必须在拼接点合法——正确的花括号、分号、类型和作用域。

两种形式：

- **`__cpp__("...")`** ——普通字符串，适合简短的单行语句。
- **`__cpp__(R"(...)")`** ——原始字符串字面量，非常适合 `asm volatile("...")` 等含大量引号的片段，避免对每个 `"` 进行转义的痛苦。

编译器不会解析或改写内容。鳄霸的符号在字符串内部不可见——只有生成的 C++ 输出中存在的名称才有效。

## 寄存器提示：`setmaxnreg`

这是典型的使用场景。在 warp 特化流水线（第 5 章）中，生产者 warpgroup 需要很少的寄存器（它主要发出 TMA 加载），而消费者需要很多（它持有 MMA 累加器）。NVIDIA 的 PTX `setmaxnreg` 指令在角色之间重新分配寄存器预算：

```choreo
parallel p1 by 2 : group-4 {
  inthreads.async (p1 == 0) {
    __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
    // producer: register-light, decrease to 40
    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      // ... TMA loads ...
    }
  }

  inthreads.async (p1 == 1) {
    __cpp__(R"(asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");)");
    // consumer: register-heavy, increase to 216
    mc = mma.fill.f16 0.0f;
    // ... WGMMA compute ...
  }
}
```

具体数值（40、216）因内核不同而调整。`.dec`/`.inc` 形式与硬件寄存器分配器协作，使两个角色能够共存而不会发生不必要的溢出。将此提示放在每个 `inthreads.async` 分支的顶部，位于重循环之前。

## 提前返回与守卫

MoE 风格的 GEMM 内核处理大小可变的专家片段。某些启动可能对给定专家宽度为零。一行 `__cpp__` 注入即可避免对空范围执行内核的其余部分：

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

变量名（`seg_end`、`seg_start`）必须与周围生成代码中的声明匹配。如果你重命名了一个鳄霸参数且代码生成输出发生变化，过时的 `__cpp__` 字符串将在编译时出错——这比悄无声息的错误结果要好。

## 预处理器：`#define` 与条件编译

鳄霸的预处理器在主编译阶段之前运行。它支持：

| 指令 | 作用 |
|-----------|------|
| `#define NAME value` | 对象式宏：文本替换 |
| `#if` / `#elif` / `#else` / `#endif` | 条件包含 |
| `#ifdef` / `#ifndef` | "已定义或未定义"的简写 |
| `#error message` | 强制产生一个带消息的编译期错误 |

宏在同一 `.co` 文件中的 `__co__` 区域和宿主 C++ 中均会展开，因此一个 `#define` 即可使分块几何在两个世界中保持一致。

## 以宏定义分块几何

生产级矩阵乘法文件将所有分块维度集中定义：

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

这些名称出现在 `parallel ... by [cdiv(M, MATMUL_WARP_M), ...]`、共享内存声明、`foreach` 边界以及宿主端验证中。修改一个 define，所有使用点随之更新。

**限制：** 预处理器不支持函数式宏。`#define MAX(a, b) ...` 不会带参数展开。需要参数化表达式时，请在纯 C++ 中使用 `constexpr` 函数。

## 用 `#error` 进行编译期断言

库文件将 `#if` 与 `#error` 配合使用以强制执行配置约束：

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif
```

当有人不兼容地修改了 swizzle 宽度或 warp 分块大小时，构建会立即失败并给出清晰的消息，而不是产生非法内核。将 `#error` 守卫视为硬件约束的活文档。

## 条件代码路径

根据宏选择整个代码区域：

```choreo
#define PATH0

__co__ foo() {
  // ...
  #ifdef PATH0
    // path 0 code
  #else
    // path 1 code
  #endif
}
```

预处理器保留一个分支并丢弃另一个，在鳄霸解析 `__co__` 函数体之前完成。这是编译期消除，不是运行时 `if`——生成的程序中没有死代码。

你也可以从命令行驱动变体：`croktile kernel.co -DMATMUL_TILE_K=128` 可以覆盖或定义宏而无需编辑源文件。这就是基准测试套件在不复制文件的情况下扫描分块大小的方式。

## `__cpp__` 字符串中的宏

一个诱人的错误：

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

这**不会**工作。预处理器不在字符串字面量内展开宏——生成的汇编将包含标识符 `PRODUCER_MAXNREG` 而非数字 `40`，PTX 会拒绝它。

在实际中，团队在 `__cpp__` 字符串内使用数值字面量，并在字符串外部使用 `#if`/`#error` 守卫来强制一致性：

```choreo
#define PRODUCER_MAXNREG 40
#if PRODUCER_MAXNREG > 50
#error "Producer maxnreg too high for this tile config"
#endif

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
}
```

## 阅读生产级 `.co` 文件

当你打开一个基准测试内核（如 `blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co`）时，从上到下阅读：

1. **宏与 `#error` 守卫** ——契约：允许的分块大小、swizzle 规则、架构标志。
2. **宿主端设置** ——缓冲区、启动配置、计时；普通 C++。
3. **`__co__` 函数** ——编排：`parallel`、`foreach`、TMA/MMA、`inthreads.async`、事件。将每个区域映射回前面的章节。
4. **`__cpp__` 孤岛** ——通常只有几行。在每处暂停，辨别硬件得到了什么而 DSL 没有表达。

这个顺序可以防止你在了解哪些常量可以修改之前，就迷失在 warp 特化循环中。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `__cpp__("...")` | 注入逐字 C++（普通字符串） |
| `__cpp__(R"(...)")` | 注入逐字 C++（原始字符串字面量） |
| `#define NAME value` | 对象式宏 |
| `#if expr` / `#elif` / `#else` / `#endif` | 条件编译 |
| `#ifdef NAME` / `#ifndef NAME` | 测试宏是否已定义 |
| `#error "message"` | 编译期断言失败 |

这些逃生通道完成了闭环：鳄霸使日常内核保持可读和结构化，而 `__cpp__` 和预处理器处理位于 DSL 抽象层之下的硬件特定细节。请谨慎使用——一条 PTX 提示、一个守卫、一条 pragma——让鳄霸函数掌控其他一切。

[下一章](ch09-debug-verbose.md)将介绍工作流的另一面：当你的内核能编译但产生错误结果时该怎么办。
