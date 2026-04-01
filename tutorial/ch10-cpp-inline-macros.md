# C++ Inline and Macros: Interfacing with Low-Level Control

Croktile is a high-level DSL: you describe tiles, pipelines, and memory levels, and the compiler generates the tedious parts of CUDA-style kernels. Most of the time that is exactly what you want.

Sometimes, though, you need to reach *below* the abstraction — to issue a specific PTX instruction, to insert a guard the Croktile layer does not express cleanly, or to parameterize a family of kernels with compile-time constants and static checks.

This final chapter covers the two main escape hatches: **`__cpp__`**, which injects raw C++ (including inline assembly and PTX) verbatim into generated code, and **the Croktile preprocessor** — **`#define`**, **`#if` / `#ifdef` / `#else` / `#endif`**, and **`#error`** for macros and conditional compilation. Neither mechanism replaces good Croktile design; they complement it when the hardware or the build system demands something the DSL does not spell out directly.

If you read [Async Pipelining: inthreads, Events, and Warp Roles](ch06-warpspec.md), you already saw how **different warps** can own **different stages** of a pipeline. This chapter is the footnote engineers reach for when that pattern must also respect **register-file limits** and **ISA-level** tuning: **`setmaxnreg`** does not appear as a first-class Croktile keyword, but it *does* appear in shipping SM90 GEMMs via **`__cpp__`**.

The matmul snippets in [Enable Tensor Cores in One Primitive: the `mma` Operations](ch05-mma.md) used names like **`MATMUL_TILE_K`** for a reason — those names are almost always **`#define`d** once at the top of real kernels so every slice of the program agrees on geometry.

## `__cpp__`: Verbatim C++ in Generated Code

**`__cpp__`** takes a **string literal** containing C++ source. The compiler splices that string into the generated output **as written** — no rewriting, no Croktile semantics on the inside. Whatever you put there must be valid in the surrounding C++ context (scopes, types, semicolons) and legal for your target (CUDA device code, host code, etc.).

The usual spellings are **`__cpp__("…")`** for ordinary strings (good for one or a few lines, as long as escaping quotes and newlines is tolerable) and **`__cpp__(R"(…)")`** for a C++ **raw string literal**: everything between **`R"(`** and **`)"`** is copied literally, including double quotes and newlines, without backslash escapes.

For PTX wrapped in **`asm volatile("…")`**, the inner assembly string contains **double quotes**; a raw string avoids the pain of **`\"`** on every line.

Multi-line injections work the same way: either embed **`\n`** in a normal string (as in the MoE-style guard below) or use a raw string that spans multiple lines inside **`R"( … )"`**. The compiler does not re-indent or reformat the string; whitespace inside the literal is preserved.

It helps to have a concrete mental model of what **verbatim** means at compile time. Croktile parses the `__co__` body around your **`__cpp__`** call sites, emits C++ (or CUDA) that implements the Croktile function, and **splices** each **`__cpp__`** argument into the generated text **exactly** as if you had typed it yourself in the output file.

There is no automatic bridge from Croktile symbols to C++ names unless the code generator already created those names and your string spells them correctly.

That is why the early-return example names **`seg_end`** and **`seg_start`** explicitly: those identifiers must match the generated declarations in that function. If you rename something in the Croktile function and the generator changes its output, a stale **`__cpp__`** string can become a **hard compile error** in generated code — which is still preferable to silently wrong machine code, but it means **`__cpp__`** fragments deserve the same review attention as hand-written CUDA.

On recent NVIDIA architectures, **warp-specialized** pipelines split work across warps or warpgroups: one side **produces** data (TMA loads, pointers, counters) and another **consumes** it (WGMMA, large register-resident accumulators). The producer needs **few** registers; the consumer needs **many**. The hardware exposes **dynamic register budget** hints via PTX such as **`setmaxnreg.dec`** and **`setmaxnreg.inc`**, which redistribute how many registers each warp is allowed to use at a given program point.

In a Croktile function, you typically place these hints at the start of an **`inthreads.async`** branch that corresponds to producer vs consumer. The following pattern is adapted from high-performance GEMM kernels in the Croktile benchmark tree (for example block-scale GEMM on SM90):

```choreo
parallel p1 by 2 : group-4 {
  inthreads.async (p1 == 0) {
    __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 104;");)");
    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      // ... producer TMA loads ...
    }
  }

  inthreads.async (p1 == 1) {
    __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 120;");)");
    mc = mma.fill.f16 0.0f;
    // ... consumer WGMMA compute ...
  }
}
```

Here each branch picks a **different** nominal register count (`104` vs `120`) so the compiler and hardware can schedule register pressure differently for the producer and consumer roles. The exact numbers are tuned per kernel, tile shape, and instruction mix.

The tutorial point is **where** the hook lives: inside the async thread specialization, before the heavy loop body.

Another kernel variant (`gemm_sp` style) pushes the asymmetry further: aggressively **decrease** the producer budget and **increase** the consumer budget so the MMA side keeps more accumulators live:

```choreo
inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
  // producer: needs few registers, so decrease to 40
}

inthreads.async (p1 == 1) {
  __cpp__(R"(asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");)");
  // consumer: needs many registers for accumulators, so increase to 216
}
```

Read these snippets as **documentation of intent**: the producer thread role is register-light; the consumer is register-heavy. The **`.dec`** / **`.inc`** forms cooperate with the GPU’s register allocator so both sides can coexist without spilling the wrong side first.

If you tried to write the same thing with an ordinary string, every embedded **`"`** inside the **`asm`** operand would need escaping, and multi-line assembly quickly becomes unreadable. **Raw string literals** are the standard C++ idiom for injecting **quote-heavy** or **multi-line** fragments; **`__cpp__`** simply forwards that string to the generated file.

**`__cpp__`** is not limited to inline assembly. Any C++ statement or snippet that makes sense at the splice point is fair game. For example, MoE-style GEMM code paths sometimes inject a **host-visible or device-visible early return** when a segment is empty, roughly along the lines of:

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

That is plain C++ control flow dropped into the generated function: if your indices say there is nothing to do, skip the rest. The exact variable names and types must match what the surrounding generated code actually declares — **`__cpp__`** does not magically know your Croktile symbols; it only pastes text.

In **MoE GEMM**-style kernels, experts are processed in **segments**; some launches may have **zero width** for a given segment after routing. An early return avoids touching uninitialized ranges or issuing empty loops. The guard is trivial C++, but expressing “return from this generated function now” might not map cleanly to a single Croktile statement in every codegen path — hence a one-line **`__cpp__`** injection.

Treat **`__cpp__`** as **sharp**. You bypass Croktile’s usual checks on whatever is inside the string.

You are responsible for **correctness** on the target SM, **synchronization** with async pipelines, and **ABI** compatibility with neighboring generated code. Prefer keeping injected fragments **small** and **localized** — one asm blob, one guard, one pragma — and let the Croktile function own structure and data movement.

When a single PTX hint or guard unlocks measurable performance (as with **`setmaxnreg`** in warp-specialized GEMM), **`__cpp__`** is the supported way to carry that detail into an otherwise declarative kernel.

## The Croktile Preprocessor: Macros and Conditionals

Croktile ships a **preprocessor** that runs over your source before the main compiler pass. It understands a familiar subset of C-style directives:

| Directive | Role |
|-----------|------|
| `#define NAME value` | Object-like macro: textual replacement |
| `#if` / `#elif` / `#else` / `#endif` | Conditional inclusion |
| `#ifdef` / `#ifndef` | Shorthand for “macro defined or not” |
| `#error message` | Force a compile-time error with a message |

Macros expand in **both** Croktile function (**`__co__`**) regions and ordinary host or device C++ in the same file, so you can share numeric constants and feature flags across the whole program.

The preprocessor evaluates **integer constant expressions** in **`#if`** conditions using the same names you **`#define`**.

For nested configuration (architecture × precision × feature flags), chain **`#elif`** branches the way you would in C++; only the winning branch is visible to the Croktile parser afterward.

**Object-like macros** excel when the same numeric literal must appear in **dozens** of places (tile extents, **`cdiv`** bounds, shared memory sizing), when you want **`#error`** to enforce relationships between those literals (swizzle vs tile K, warp M vs WGMMA hardware expectations), or when you generate **many kernel variants** from one source file by varying defines on the **compiler command line** (`-DMATMUL_TILE_K=128`) without editing the body.

When a value is truly local to one C++ function and never crosses into the `__co__` body, ordinary **`constexpr`** variables or templates in host/device C++ are often clearer. The tutorial’s matmul files skew toward macros because **`__co__` and C++ share** those numbers; a single **`#define`** avoids drift between the two worlds.

Typical matmul and GEMM sources centralize tile geometry in macros:

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
```

Those names then appear in array extents, **`parallel … by […]`** bounds, and shared-memory declarations. One change to the macro updates every use site.

**Important limitation:** Croktile’s preprocessor does **not** support **function-like** macros. You cannot write **`#define MAX(a, b) …`** and expect it to expand with arguments. Stick to **object-like** **`#define NAME replacement`** or use **`constexpr`** / inline functions in plain C++ where you need functions.

Libraries use **`#if`** together with **`#error`** to **assert configuration constraints** at compile time. If someone changes a swizzle width or warp tile incompatibly, the build fails immediately with a clear message instead of producing a wrong or illegal kernel:

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif

```

This pattern documents **why** those equalities must hold (layout assumptions, ISA constraints) and enforces them in CI.

You can select whole regions of code based on a defined macro — useful for experimental paths, architecture branches, or debug instrumentation:

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

The preprocessor keeps **one** branch and discards the other before Croktile parses the `__co__` body. That is different from a runtime **`if`**: no dead branch remains in the generated program for the disabled path.

Earlier tutorial tracks (for example **Lv1** optimization material) lean heavily on macros to **parameterize** kernels: tile sizes, unroll factors, and feature toggles become compile-time constants so you can sweep a search space or generate variants without duplicating entire files. The same **`#define` / `#if`** machinery you saw in matmul benchmarks is the glue for those experiments.

You might hope to write a register count once as a macro and reuse it inside **`__cpp__`**:

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

That snippet is **not** what you want: inside a string literal (including a raw string), the C-style preprocessor does **not** replace **`PRODUCER_MAXNREG`** with `40`.

The generated asm would contain the **identifier** **`PRODUCER_MAXNREG`**, which is not a valid immediate for PTX.

In shipping kernels, immediates inside **`R"(asm …)"`** are almost always **numeric literals** typed out (`40`, `216`, `104`, …). To keep them consistent with macro-defined tile geometry, teams typically document the relationship in comments, use **`#if` / `#error`** so illegal combinations fail at compile time, or generate the **`.co`** from a template or script when a single source of truth is mandatory across many asm sites.

Use **`#define` freely for Croktile function geometry** (sizes that appear **outside** string literals). Treat **`__cpp__` strings** as opaque text the preprocessor will not reach into.

When you review or write an injection, check **scope**: is this string valid C++ at the exact insertion point, with correct braces and semicolons? Check **identifiers**: do all names match what codegen emits for this kernel version?

Ask whether the instruction is legal on the **SM target** you compile for, and whether a fallback path would need **`#if`** at the Croktile level. For **synchronization**, especially with async pipelines from Chapter 6, confirm this asm interacts safely with producer/consumer ordering and barriers. Prefer **minimal surface**: express most of the logic in the Croktile function and reserve **`__cpp__`** for the one line the DSL does not cover.

If any answer is uncertain, prototype in a small CUDA file first, then port the proven fragment into **`__cpp__`**.

Reach for **`__cpp__`** when you need **exact** control over a line of C++, inline assembly, or PTX in the **generated** output — register hints, targeted guards, intrinsics the DSL does not wrap. Reach for **`#define`** and conditionals when you want **compile-time** parameterization and **validation** shared across the `__co__` body and C++ — tile dimensions, algorithm variants, and static **`#error`** checks.

Together they close the loop: Croktile keeps everyday kernels readable and structured, while **`__cpp__`** and the preprocessor let expert paths stay **honest** (asserted constraints) and **complete** (hardware-specific details where they matter).

## How to read a production `.co` file

When you open something like **`blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c_m2048_n2048_k2048.co`** or another entry in the Croktile benchmark tree, read **top down**.

Start with **macros and `#error` guards**. They encode the contract: allowed tile sizes, swizzle rules, and architecture flags. If you change a define, the next **`#if`** might stop the build; treat that as documentation, not annoyance.

Next is **host setup** — buffers, launch configuration, timing; often ordinary C++.

Then scan the **`__co__` function** for the orchestration story: **`parallel`**, **`foreach`**, TMA/MMA, **`inthreads.async`**, events. Map each region back to the chapters in this tutorial.

Finally, find **`__cpp__` islands** — usually a handful of lines. Pause on each one and ask what the hardware gets that the DSL does not spell out (register hints, returns, intrinsics).

That order prevents getting lost in the middle of a warp-specialized loop before you know which **`MATMUL_*`** constants you are allowed to change.

**`__cpp__("…")`** and **`__cpp__(R"(…)")`** splice string contents verbatim into generated code; raw strings are ideal for **`asm volatile("…")`** and other quote-heavy snippets. Warp-specialized GEMM examples use **`setmaxnreg.dec`** / **`setmaxnreg.inc`** in producer vs consumer **`inthreads.async`** branches to tune register pressure for TMA-heavy vs MMA-heavy roles.

Other **`__cpp__`** uses include simple C++ control flow such as early **`return`** when a segment range is empty (as in MoE-style GEMM sources).

Croktile’s preprocessor supports **`#define`**, **`#if` / `#ifdef` / `#ifndef` / `#else` / `#endif`**, and **`#error`**; macros apply to `__co__` and C++ regions alike, with **no function-like macros** — only object-like **`#define NAME value`**. **`#error`** under **`#if`** encodes compile-time requirements on configuration constants, a pattern ubiquitous in matmul benchmarks.

That finishes the Croktile DSL tutorial track at the boundary between declarative Croktile orchestration and the low-level reality of production GPU kernels. You started from a running program ([Hello Croktile](ch01-hello-choreo.md)), learned how data moves and pipelines overlap, saw TMA and swizzle, and connected MMA and WGMMA to tile shapes.

You specialized warps for async roles, iterated persistent tiles, scaled across warpgroups, and shaped irregular access with **`view`** / **`from`**. This chapter is the **pressure valve**: when the model is right but the last percent needs a PTX hint or a static configuration guard, you know which door to open.

From here, the most productive next step is to read a full benchmark kernel end to end — **macros at the top**, **`__co__` function in the middle**, and the occasional **`__cpp__` hook** where the hardware demands it — then change one constant or one guard, rebuild, and measure.

Small, deliberate edits beat large rewrites; the toolchain and the **`#error`** lines will tell you when a configuration stops being legal long before you chase mysterious numerical wrong answers on the GPU.
