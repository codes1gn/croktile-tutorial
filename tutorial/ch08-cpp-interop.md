# C++ Interop: Device Functions, Inline Code, and the Preprocessor

Every high-level language needs an escape hatch to the layer below. Python has `ctypes` and C extensions. Rust has `unsafe` blocks and `extern "C"`. Java has JNI. The reason is always the same: no matter how expressive the abstraction, some hardware feature, some legacy library, some performance-critical intrinsic lives outside the language's domain.

Croqtile's interoperability story has three parts:

1. **`__device__` functions** — Standard CUDA device functions that coexist with `__co__` functions in the same `.co` file. The compiler passes them through unchanged.
2. **`__cpp__`** — Inject verbatim C++ or PTX into the generated code for hardware-specific instructions the DSL does not emit.
3. **The preprocessor** — `#define`, `#if`, `#ifdef`, `#error` for compile-time configuration, the same role `#define` plays in C/C++ codebases.

So far the tutorial has built a stack of abstractions: tiles, parallelism, MMA, pipelines, events, TMA. That is the Croqtile you want to live in. This chapter is about the times you need to step outside.

## How `.co` files compile

A `.co` file mixes three kinds of code. The Croqtile compiler treats each differently — and the difference matters for understanding what you can and cannot do at each boundary. Consider this skeleton:

```choreo
__device__ __forceinline__ float fast_rsqrt(float x) {   // ①
  return __frsqrt_rn(x);
}

__co__ void my_kernel(f32 [M, N] input, f32 [M, N]& output) {   // ②
  parallel {bm, bn} by [cdiv(M, 64), cdiv(N, 64)] : block {
    // ... Croqtile orchestration: parallel, dma, mma, events ...
    __cpp__("asm volatile(\"setmaxnreg.dec.sync.aligned.u32 40;\");");   // ③
  }
}

int main() {   // ④
  auto in = choreo::make_spandata<choreo::f32>(M, N);
  auto out = my_kernel(in.view());
}
```

The compiler splits this into a single intermediate `.cu` file, then hands it to `nvcc`:

![.co compilation: Croqtile compiler transforms __co__ and passes through __device__ and host C++, merging everything into one .cu file for nvcc](../assets/images/ch08/fig1_compilation_flow_dark.png#only-dark)
![.co compilation: Croqtile compiler transforms __co__ and passes through __device__ and host C++, merging everything into one .cu file for nvcc](../assets/images/ch08/fig1_compilation_flow_dark.png#only-light)

What happens to each region:

- **① `__device__` functions** are copied into the generated `.cu` file **verbatim**. The Croqtile compiler does not parse their bodies, does not rewrite them, does not manage their register allocation. They appear in the intermediate source exactly as you wrote them. This is the key distinction from `__co__` functions: a `__device__` function is pure CUDA that happens to live in a `.co` file.
- **② `__co__` functions** are the heart of Croqtile. The compiler transforms them into `__global__` CUDA kernels: `parallel` becomes grid/block launch dimensions, `dma.copy` becomes cooperative load sequences (or TMA descriptor issues), `mma` becomes `wgmma` instructions, `shared` declarations become `__shared__` allocations with computed sizes, events become barrier/mbarrier synchronization. The generated code is typically hundreds of lines of CUDA for a few dozen lines of Croqtile.
- **③ `__cpp__` strings** are spliced character-for-character into the generated CUDA at the point where they appear in the `__co__` body. The compiler does not parse their contents — they must be valid at the splice point in the generated code.
- **④ Host code** (`main()` and any other non-annotated functions) is passed through as ordinary C++ and ends up in the host compilation path.

All four regions merge into one `.cu` file. `nvcc` compiles the device code (`__global__`, `__device__`) into GPU machine code, compiles the host code into CPU code, and links them into a single binary. Run `croqtile kernel.co -v` (Chapter 9) to see the exact `nvcc` command the compiler issues.

## `__device__` vs ordinary C++ inline

Both `__device__` functions and ordinary C++ functions are passed through unchanged, but they land in different compilation paths — and the distinction matters.

A `__device__` function compiles to **GPU machine code**. It can use CUDA intrinsics (`__shfl_xor_sync`, `__frsqrt_rn`, `atomicAdd`), access `__shared__` memory, and run inside a `__global__` kernel. The `__co__` body calls it with the `call` keyword, and the generated `__global__` kernel invokes it directly on the device.

An ordinary C++ function (no `__device__` or `__co__` annotation) compiles to **CPU code**. It runs on the host — setting up buffers, launching kernels, printing results. You cannot call a host function from inside a `__co__` body; the compiler would reject it because the generated `__global__` kernel runs on the GPU.

The boundary is strict: `__device__` lives on the device side of the `nvcc` split, host C++ lives on the host side. The `__co__` function is the bridge — the compiler transforms it into a `__global__` kernel (device side) and generates a host-side wrapper that launches it.

## `__device__` functions: CUDA alongside Croqtile

When an algorithm needs a helper function that operates at the warp or thread level — a custom reduction, a sorting network, a special math function — writing it as a standard CUDA `__device__` function is often the natural choice. Croqtile does not need to manage these; they are plain CUDA.

A `__co__` function can call a `__device__` function using the `call` keyword:

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

**`__device__ void warp_topk<K>`** — A templated device function implementing warp-level top-K selection using shuffle instructions. It operates on raw pointers (`f32*`, `s32*`), not Croqtile spans.

**`__device__ T SHFL_XOR`** — A warp shuffle intrinsic wrapper. The `__forceinline__` and `__shfl_xor_sync` are standard CUDA — Croqtile passes them through.

**`call warp_topk<8>(...)`** — The `call` keyword invokes a `__device__` function from within a `__co__` body. Arguments are passed by pointer; the compiler handles the address translation between Croqtile spans and raw pointers.

**`inthreads (tid == 0)`** — Only thread 0 in each warp writes back results. Note this uses `inthreads` without `.async` — a sequential thread filter, not a concurrent region.

This pattern — Croqtile handles the parallelism, tiling, and memory orchestration; `__device__` functions handle the per-warp algorithm — is common in production kernels like MoE (mixture-of-experts) top-K, custom reductions, and specialized math.

## `__cpp__`: verbatim C++ injection

`__cpp__` takes a string literal and pastes it, character for character, into the generated CUDA file. Whatever you place there must be valid at the splice point. The Croqtile compiler does not parse or rewrite the contents.

**Two forms:**

- **`__cpp__("...")`** — Ordinary string; best for short one-liners.
- **`__cpp__(R"(...)")`** — Raw string literal; use for `asm volatile` where escaping every `"` would be painful.

### Register hints: `setmaxnreg`

The canonical `__cpp__` use case is register redistribution in warp-specialized pipelines ([Chapter 5](ch05-branch-control.md)). The producer warpgroup is register-light (mostly TMA loads), while the consumer is register-heavy (MMA accumulators). PTX's `setmaxnreg` instruction moves the register budget:

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

**Placement** — Put the hint at the top of each `inthreads.async` branch, before the heavy loop.

### Early returns and guards

MoE-style kernels often process variable-sized expert segments; some launches have zero width:

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

The identifiers must match what the surrounding generated code declares.

### Macros inside `__cpp__` strings

A common mistake: assuming the preprocessor expands macros inside string literals.

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

This fails — the preprocessor does not expand inside strings. Use numeric literals inside `__cpp__` and `#if` / `#error` outside to enforce consistency:

```choreo
#define PRODUCER_MAXNREG 40
#if PRODUCER_MAXNREG > 50
#error "Producer maxnreg too high for this tile config"
#endif

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
}
```

## The preprocessor

Croqtile's preprocessor runs before the main compiler pass. Macros expand in both `__co__` regions and host C++ in the same `.co` file, so one definition keeps tile geometry and host-side checks aligned.

| Directive | Role |
|-----------|------|
| `#define NAME value` | Object-like macro: textual replacement |
| `#if` / `#elif` / `#else` / `#endif` | Conditional inclusion |
| `#ifdef` / `#ifndef` | Shorthand for whether a macro is defined |
| `#error message` | Force a compile-time failure with a message |

Function-like macros (`#define MAX(a, b) ...`) are not supported. Use `constexpr` helpers in ordinary C++.

### Tile geometry as macros

Production matmul sources centralize tile dimensions at the top:

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

The same names appear in `parallel`, shared-memory declarations, `foreach` bounds, and host-side verification.

### Compile-time assertions with `#error`

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif
```

Treat these guards as documentation of hardware constraints.

### Conditional code paths

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

The preprocessor keeps one branch and discards the other before Croqtile parses the `__co__` body. Command-line defines: `croqtile kernel.co -DMATMUL_TILE_K=128`.

## Reading a production `.co` file

When you open a benchmark kernel, read top down:

1. **Macros and `#error` guards** — The contract: allowed tile sizes, swizzle rules, architecture flags.
2. **`__device__` functions** — Helper algorithms (top-K, reductions, shuffle wrappers) that Croqtile passes through.
3. **Host setup** — Buffers, launch configuration, timing; ordinary C++.
4. **The `__co__` function** — Orchestration: `parallel`, `foreach`, TMA/MMA, `inthreads.async`, events. Map each region back to earlier chapters.
5. **`__cpp__` islands** — Usually a handful of lines. Pause on each and ask what the hardware receives that the DSL does not spell.

## Chapter summary

| Syntax | What it does | Runs on |
|--------|--------------|---------|
| `__device__ fn()` | Standard CUDA device function; passed through unchanged to `.cu` | GPU |
| `call fn(args)` | Invoke a `__device__` function from a `__co__` body | GPU |
| `__cpp__("...")` | Inject verbatim C++ at the splice point (ordinary string) | GPU |
| `__cpp__(R"(...)")` | Inject verbatim C++ (raw string; use for `asm volatile`) | GPU |
| `setmaxnreg` | Register redistribution via `__cpp__`: `dec` on producer, `inc` on consumer | GPU |
| `#define NAME value` | Object-like macro; shared across `__co__` and host regions | Preprocessor |
| `#if` / `#ifdef` / `#error` | Conditional compilation and compile-time assertions | Preprocessor |
| `-DNAME=value` | Define macros from the command line for tile sweeps | Preprocessor |

The [next chapter](ch09-debug-verbose.md) turns to the other side of the workflow: what to do when the kernel compiles but the output is wrong — debugging, verbose modes, and systematic narrowing.
