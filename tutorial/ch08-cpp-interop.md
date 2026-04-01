# C++ Interop: Inline Code and the Preprocessor

Croktile handles tiles, pipelines, and memory levels so you don't have to spell them in CUDA. Most of the time that is exactly right. But every so often you need something the DSL does not wrap: a specific PTX instruction, an early return guarding an empty segment, or a compile-time constant shared between the `__co__` body and host C++.

This chapter covers the two escape hatches: **`__cpp__`**, which injects raw C++ verbatim into the generated output, and the **Croktile preprocessor** — `#define`, `#if`/`#ifdef`/`#else`/`#endif`, and `#error` — for macros and conditional compilation.

## `__cpp__`: Verbatim C++ Injection

`__cpp__` takes a string literal and pastes it, character for character, into the generated CUDA or C++ file. Whatever you put inside must be valid at the splice point — correct braces, semicolons, types, and scope.

Two forms:

- **`__cpp__("...")`** — ordinary string, good for short one-liners.
- **`__cpp__(R"(...)")`** — raw string literal, ideal for `asm volatile("...")` and other quote-heavy fragments where escaping every `"` would be painful.

The compiler does not parse or rewrite the contents. Croktile symbols are not visible inside the string — only names that exist in the generated C++ output.

## Register Hints: `setmaxnreg`

The canonical use case. In warp-specialized pipelines (Chapter 5), the producer warpgroup needs few registers (it mostly issues TMA loads) while the consumer needs many (it holds MMA accumulators). NVIDIA's PTX `setmaxnreg` instruction redistributes the register budget between roles:

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

The exact numbers (40, 216) are tuned per kernel. The `.dec`/`.inc` forms cooperate with the hardware register allocator so both roles can coexist without unnecessary spilling. Place the hint at the top of each `inthreads.async` branch, before the heavy loop.

## Early Returns and Guards

MoE-style GEMM kernels process variable-sized expert segments. Some launches may have zero width for a given expert. A one-line `__cpp__` injection avoids running the rest of the kernel on empty ranges:

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

The variable names (`seg_end`, `seg_start`) must match what the surrounding generated code declares. If you rename a Croktile parameter and the codegen changes its output, stale `__cpp__` strings break at compile time — which is better than silently wrong results.

## The Preprocessor: `#define` and Conditionals

Croktile's preprocessor runs before the main compiler pass. It understands:

| Directive | Role |
|-----------|------|
| `#define NAME value` | Object-like macro: textual replacement |
| `#if` / `#elif` / `#else` / `#endif` | Conditional inclusion |
| `#ifdef` / `#ifndef` | Shorthand for "defined or not" |
| `#error message` | Force a compile-time error with a message |

Macros expand in both `__co__` regions and host C++ in the same `.co` file, so a single `#define` keeps tile geometry consistent across both worlds.

## Tile Geometry as Macros

Production matmul files centralize every tile dimension:

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

These names appear in `parallel ... by [cdiv(M, MATMUL_WARP_M), ...]`, shared memory declarations, `foreach` bounds, and host-side verification. Change one define, and every use site updates.

**Limitation:** the preprocessor does not support function-like macros. `#define MAX(a, b) ...` will not expand with arguments. Use `constexpr` functions in plain C++ where you need parameterized expressions.

## Compile-Time Assertions with `#error`

Libraries pair `#if` with `#error` to enforce configuration constraints:

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif
```

When someone changes a swizzle width or warp tile incompatibly, the build fails immediately with a clear message instead of producing an illegal kernel. Treat `#error` guards as living documentation of hardware constraints.

## Conditional Code Paths

Select entire regions based on a macro:

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

The preprocessor keeps one branch and discards the other before Croktile parses the `__co__` body. This is compile-time elimination, not a runtime `if` — no dead code in the generated program.

You can also drive variants from the command line: `croktile kernel.co -DMATMUL_TILE_K=128` overrides or defines macros without editing the source. This is how benchmark suites sweep over tile sizes without duplicating files.

## Macros Inside `__cpp__` Strings

A tempting mistake:

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

This does **not** work. The preprocessor does not expand macros inside string literals — the generated asm will contain the identifier `PRODUCER_MAXNREG`, not the number `40`, and PTX will reject it.

In practice, teams type numeric literals inside `__cpp__` strings and use `#if`/`#error` guards outside the strings to enforce consistency:

```choreo
#define PRODUCER_MAXNREG 40
#if PRODUCER_MAXNREG > 50
#error "Producer maxnreg too high for this tile config"
#endif

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
}
```

## Reading a Production `.co` File

When you open a benchmark kernel like `blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co`, read top down:

1. **Macros and `#error` guards** — the contract: allowed tile sizes, swizzle rules, architecture flags.
2. **Host setup** — buffers, launch config, timing; ordinary C++.
3. **The `__co__` function** — orchestration: `parallel`, `foreach`, TMA/MMA, `inthreads.async`, events. Map each region back to earlier chapters.
4. **`__cpp__` islands** — usually a handful of lines. Pause on each and identify what the hardware gets that the DSL does not spell.

That order prevents getting lost in a warp-specialized loop before you know which constants you can change.

## New Syntax

| Syntax | Meaning |
|--------|---------|
| `__cpp__("...")` | Inject verbatim C++ (ordinary string) |
| `__cpp__(R"(...)")` | Inject verbatim C++ (raw string literal) |
| `#define NAME value` | Object-like macro |
| `#if expr` / `#elif` / `#else` / `#endif` | Conditional compilation |
| `#ifdef NAME` / `#ifndef NAME` | Test whether a macro is defined |
| `#error "message"` | Compile-time assertion failure |

These escape hatches close the loop: Croktile keeps everyday kernels readable and structured, while `__cpp__` and the preprocessor handle the hardware-specific details that sit below the DSL's abstraction level. Use them sparingly — one PTX hint, one guard, one pragma — and let the Croktile function own everything else.

The [next chapter](ch09-debug-verbose.md) covers the other side of the workflow: what to do when your kernel compiles but produces wrong results.
