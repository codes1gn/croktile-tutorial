# What is Good in CrokTile?

Croktile is a C++ embedded DSL for writing high-performance GPU and DSA kernels. You write Croktile functions in `.co` files alongside your existing C++ code, and the compiler transpiles them into efficient target code (CUDA today, more backends planned) with full interoperability to CUDA, CuTe, and any C++ library.

Four design pillars set Croktile apart from raw CUDA, CuTe, or Triton.

## Easy to Use

You work on tensors and tiles, not raw buffers and pointers. DMA transfers, memory allocation, and synchronization that take dozens of lines of CUDA boilerplate reduce to a single `dma.copy ... => ...` statement — roughly 40% of the equivalent CUDA code. Every construct compiles down to the same PTX you would write by hand: zero-cost abstraction, no runtime overhead, no hidden allocations.

## Compile-Time Safety

353 compile-time diagnostic checks across 7 compiler modules catch shape mismatches, tiling errors, and DMA violations before any code runs on the GPU. In development builds, 1,319 runtime assertions guard every transfer and memory access. Together they eliminate entire classes of bugs that would otherwise need `cuda-memcheck` and hours of printf debugging.

## Dynamic Shapes

First-class symbolic dimensions let you write a kernel once and run it on any shape without recompilation. The compiler derives packed-K, metadata columns, grid dimensions, and shared-memory size automatically. Static and runtime memory are unified — no template metaprogramming, no boilerplate.

## Born for AI Tuning

Compact, structured syntax keeps entire kernels inside AI context windows (30–60 lines vs. hundreds in CUDA/CuTe). Structured error messages and well-documented CLI arguments let autonomous agents compile, profile, and iterate without human intervention. In practice, AI agents have pushed FP16 matmul from 671 to 1,127 TFLOPS in a single session.
