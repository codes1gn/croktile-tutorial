# C++ 互操作：`__device__` 函数、内联代码与预处理器

每一种高级语言都需要一条通往更底层机制的逃生通道。Python 有 `ctypes` 与 C 扩展，Rust 有 `unsafe` 块与 `extern "C"`，Java 有 JNI。原因总是一样的：无论抽象多么富有表现力，总有某些硬件特性、某些遗留库、某些性能攸关的内建能力处在该语言域之外。

鳄霸（Croqtile）的互操作叙事包含三个部分：

1. **`__device__` 函数** — 标准 CUDA device 函数，与 `__co__` 函数共存于同一 `.co` 文件中。编译器对它们原样透传。
2. **`__cpp__`** — 将逐字 C++ 或 PTX 注入到生成代码中，用于 DSL 无法发出的硬件相关指令。
3. **预处理器** — `#define`、`#if`、`#ifdef`、`#error`，用于编译期配置，与 C/C++ 代码库中 `#define` 所扮演的角色相同。

至此，本教程已堆叠起一整座抽象栈：tile、并行、MMA、流水线、事件、TMA。那是你希望在鳄霸中长期栖居的世界。本章讨论的是你需要暂时跨出去的那些时刻。

## `.co` 文件如何编译

`.co` 文件混合三类代码。鳄霸编译器对每类代码区别对待——理解这些边界对于搞清哪些事能做、哪些事不能做十分关键。考虑下面这个骨架：

```choreo
__device__ __forceinline__ float fast_rsqrt(float x) {   // ①
  return __frsqrt_rn(x);
}

__co__ void my_kernel(f32 [M, N] input, f32 [M, N]& output) {   // ②
  parallel {bm, bn} by [cdiv(M, 64), cdiv(N, 64)] : block {
    // ... Croqtile 编排：parallel, dma, mma, events ...
    __cpp__("asm volatile(\"setmaxnreg.dec.sync.aligned.u32 40;\");");   // ③
  }
}

int main() {   // ④
  auto in = choreo::make_spandata<choreo::f32>(M, N);
  auto out = my_kernel(in.view());
}
```

编译器将所有内容合并为一个 `.cu` 中间文件，然后交由 `nvcc` 处理：

![.co compilation: Croqtile compiler transforms __co__ and passes through __device__ and host C++, merging everything into one .cu file for nvcc](../assets/images/ch08/fig1_compilation_flow_dark.png#only-dark)
![.co compilation: Croqtile compiler transforms __co__ and passes through __device__ and host C++, merging everything into one .cu file for nvcc](../assets/images/ch08/fig1_compilation_flow_light.png#only-light)

各区域的处理方式：

- **① `__device__` 函数** 被**逐字拷贝**到生成的 `.cu` 文件。鳄霸编译器不解析其函数体、不改写、不管理其寄存器分配。它们出现在中间源码中时与你书写时完全一致。这是与 `__co__` 函数的关键区别：`__device__` 函数是恰好住在 `.co` 文件中的纯 CUDA 代码。
- **② `__co__` 函数** 是鳄霸的核心。编译器将它们变换为 `__global__` CUDA kernel：`parallel` 变为 grid/block launch 维度，`dma.copy` 变为协作 load 序列（或 TMA 描述符发射），`mma` 变为 `wgmma` 指令，`shared` 声明变为带计算尺寸的 `__shared__` 分配，事件变为 barrier/mbarrier 同步。生成代码通常为几十行鳄霸对应数百行 CUDA。
- **③ `__cpp__` 字符串** 在它们出现于 `__co__` 主体中的位置被逐字符拼接到生成的 CUDA。编译器不解析其内容——它们在生成代码的拼接点处必须合法。
- **④ Host 代码**（`main()` 及任何其他无注解函数）作为普通 C++ 透传，进入 host 编译路径。

四个区域合并进一个 `.cu` 文件。`nvcc` 将 device 代码（`__global__`、`__device__`）编译为 GPU 机器码，将 host 代码编译为 CPU 代码，最终链接为单一二进制。运行 `croqtile kernel.co -v`（第 9 章）可查看编译器发出的完整 `nvcc` 命令。

## `__device__` 与普通 C++ inline 的区别

`__device__` 函数与普通 C++ 函数都被原样透传，但它们落入不同的编译路径——这一区别至关重要。

`__device__` 函数编译为 **GPU 机器码**。它可以使用 CUDA 内建（`__shfl_xor_sync`、`__frsqrt_rn`、`atomicAdd`），访问 `__shared__` 内存，在 `__global__` kernel 内部运行。`__co__` 主体通过 `call` 关键字调用它，生成的 `__global__` kernel 在设备端直接调用。

普通 C++ 函数（无 `__device__` 或 `__co__` 注解）编译为 **CPU 代码**。它在 host 上运行——建立缓冲区、启动 kernel、打印结果。你不能在 `__co__` 主体内调用 host 函数；编译器会拒绝，因为生成的 `__global__` kernel 运行在 GPU 上。

边界是严格的：`__device__` 位于 `nvcc` 切分的 device 侧，host C++ 位于 host 侧。`__co__` 函数是桥梁——编译器将其变换为 `__global__` kernel（device 侧）并生成 host 侧的启动封装。

## `__device__` 函数：CUDA 与 Croqtile 并存

当算法需要在线程或 warp 粒度上运作的辅助函数 —— 例如自定义归约、排序网络、特殊数学函数 —— 将其写成标准 CUDA `__device__` 函数往往是自然之选。鳄霸无需管理这些；它们就是普通 CUDA。

`__co__` 函数可使用 `call` 关键字调用 `__device__` 函数：

```choreo
template <int K>
__device__ void warp_topk(f32* vals, s32* idxs);

template <typename T>
__device__ __forceinline__ T SHFL_XOR(T var, int lane_mask, int width) {
  return __shfl_xor_sync(uint32_t(-1), var, lane_mask, width);
}

__co__ void moe_topk(f32 [N_TOKENS, N_EXPERTS] scores,
                     s32 [N_TOKENS, K]& topk_ids,
                     f32 [N_TOKENS, K]& topk_scores,
                     int N_BLOCK) {
  parallel n by N_BLOCK : block {
    foreach m in [ |scores.span| / N_THREAD / N_BLOCK ] {
      shared_scores = dma.copy scores.chunkat(n#m, _) => shared;

      parallel gid by [N_THREAD / 32] : group {
        parallel tid by 32 : thread {
          score = shared_scores.data.span_as(|shared_scores.data.span|).at(gid # tid);

          local s32 [K] frag_idx{-1};
          local f32 [K] frag_val{-1.0f};
          frag_idx.at(0) = tid;
          frag_val.at(0) = score;

          call warp_topk<8>(frag_val, frag_idx);

          inthreads (tid == 0) {
            foreach k in K {
              topk_ids.at(n#m, gid#k) = frag_idx.at(k);
              topk_scores.at(n#m, gid#k) = frag_val.at(k);
            }
          }
        }
      }
    }
  }
}
```

**`__device__ void warp_topk<K>`** — 使用 shuffle 指令实现 warp 级 top-K 选择的模板 device 函数。它操作原始指针（`f32*`、`s32*`），而非 Croqtile spans。

**`__device__ T SHFL_XOR`** — warp shuffle 内建封装。`__forceinline__` 与 `__shfl_xor_sync` 为标准 CUDA —— Croqtile 原样透传。

**`call warp_topk<8>(...)`** — `call` 关键字在 `__co__` 主体内调用 `__device__` 函数。参数按指针传递；编译器负责在 Croqtile span 与原始指针之间完成地址转换。

**`inthreads (tid == 0)`** — 每个 warp 中只有线程 0 写回结果。注意此处使用不带 `.async` 的 `inthreads` —— 为顺序线程过滤，而非并发区域。

这一模式 —— Croqtile 负责并行、tiling 与内存编排；`__device__` 函数负责 per-warp 算法 —— 在 MoE（mixture-of-experts）top-K、自定义归约与专用数学等生产级 kernel 中十分常见。

## `__cpp__`：逐字 C++ 注入

`__cpp__` 接收一个字符串字面量，并逐字符粘贴到生成的 CUDA 文件中。置于该处的任何内容在拼接点都必须合法。鳄霸编译器既不解析也不改写其内容。

**两种形式：**

- **`__cpp__("...")`** — 普通字符串；最适合短单行。
- **`__cpp__(R"(...)")`** — 原始字符串字面量；在需要 `asm volatile` 且为每个 `"` 转义会很痛苦时使用。

### 寄存器提示：`setmaxnreg`

典型的 `__cpp__` 用例是 warp 特化流水线中的寄存器再分配（[第 5 章](ch05-branch-control.md)）。producer warpgroup 寄存器占用较轻（多为 TMA load），而 consumer 较重（MMA 累加器）。PTX 的 `setmaxnreg` 指令用于移动寄存器预算：

```choreo
parallel p1 by 2 : group-4 {
  inthreads.async (p1 == 0) {
    __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      // ... TMA loads ...
    }
  }

  inthreads.async (p1 == 1) {
    __cpp__(R"(asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");)");
    mc = mma.fill.f16 0.0f;
    // ... WGMMA compute ...
  }
}
```

**放置位置** — 将提示放在每个 `inthreads.async` 分支的顶部、重循环之前。

### 提前返回与守卫

MoE 风格 kernel 常处理可变长度的 expert 段；某些 launch 宽度为零：

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

标识符须与周围生成代码所声明的一致。

### `__cpp__` 字符串内的宏

常见误区：以为预处理器会在字符串字面量内展开宏。

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

这会失败 —— 预处理器不会在字符串内展开。在 `__cpp__` 中使用数字字面量，并在其外使用 `#if` / `#error` 以强制一致性：

```choreo
#define PRODUCER_MAXNREG 40
#if PRODUCER_MAXNREG > 50
#error "Producer maxnreg too high for this tile config"
#endif

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
}
```

## 预处理器

鳄霸的预处理器在主编译遍之前运行。宏在同一 `.co` 文件的 `__co__` 区域与 host C++ 中都会展开，因此一份定义即可使 tile 几何与 host 侧检查保持一致。

| 指令 | 作用 |
|-----------|------|
| `#define NAME value` | 类对象宏：文本替换 |
| `#if` / `#elif` / `#else` / `#endif` | 条件包含 |
| `#ifdef` / `#ifndef` | 宏是否已定义的简写 |
| `#error message` | 以消息强制编译期失败 |

不支持函数式宏（`#define MAX(a, b) ...`）。请在普通 C++ 中使用 `constexpr` 辅助函数。

### 以宏表示 tile 几何

生产级 matmul 源码通常在文件顶部集中定义 tile 尺寸：

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

相同名字出现在 `parallel`、共享内存声明、`foreach` 边界以及 host 侧校验中。

### 使用 `#error` 的编译期断言

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif
```

将这些守卫视为对硬件约束的文档化说明。

### 条件代码路径

```choreo
#define PATH0

__co__ foo() {
  #ifdef PATH0
    // path 0 code
  #else
    // path 1 code
  #endif
}
```

预处理器在鳄霸解析 `__co__` 主体之前保留一分支并丢弃另一分支。命令行定义：`croqtile kernel.co -DMATMUL_TILE_K=128`。

## 如何阅读生产级 `.co` 文件

打开基准 kernel 时，自上而下阅读：

1. **宏与 `#error` 守卫** — 契约：允许的 tile 尺寸、swizzle 规则、架构标志。
2. **`__device__` 函数** — Croqtile 原样透传的辅助算法（top-K、归约、shuffle 封装等）。
3. **Host 设置** — 缓冲区、launch 配置、计时；普通 C++。
4. **`__co__` 函数** — 编排：`parallel`、`foreach`、TMA/MMA、`inthreads.async`、事件。将每个区域映射回先前章节。
5. **`__cpp__` 岛** — 通常寥寥数行。在每一处停顿，追问硬件收到了哪些 DSL 未显式写出的内容。

## 本章小结

| 语法 | 功能 | 运行位置 |
|--------|------|---------|
| `__device__ fn()` | 标准 CUDA device 函数；逐字透传到 `.cu` | GPU |
| `call fn(args)` | 从 `__co__` 主体内调用 `__device__` 函数 | GPU |
| `__cpp__("...")` | 在拼接点注入逐字 C++（普通字符串） | GPU |
| `__cpp__(R"(...)")` | 注入逐字 C++（原始字符串；用于 `asm volatile`） | GPU |
| `setmaxnreg` | 通过 `__cpp__` 做寄存器再分配：producer 上 `dec`，consumer 上 `inc` | GPU |
| `#define NAME value` | 类对象宏；在 `__co__` 与 host 区域间共享 | 预处理器 |
| `#if` / `#ifdef` / `#error` | 条件编译与编译期断言 | 预处理器 |
| `-DNAME=value` | 命令行定义宏，用于 tile 尺寸扫描 | 预处理器 |

[下一章](ch09-debug-verbose.md)转向工作流的另一面：kernel 已能编译但结果错误时该如何处理 —— 调试、verbose 模式与系统化缩小问题范围。
