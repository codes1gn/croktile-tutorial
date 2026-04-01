# Croktile Tutorial

Welcome to the Croktile tutorial. This guide walks you through writing high-performance GPU kernels using Croktile, starting from scratch and building up to production-grade patterns.

Each chapter introduces a small set of new concepts by evolving a running example. By the end, you will have encountered every major Croktile construct in a concrete, working program. For detailed syntax design and language reference, see the [Coding Reference](../documentation/index.md).

## Chapters

0. [Installation: Setting Up the Croktile Compiler](ch00-installation.md)
1. [Hello Croktile: From Zero to Running Kernel](ch01-hello-croktile.md)
2. [Data Movement: Tiles Instead of Elements](ch02-data-movement.md)
3. [Parallelism: Mapping Work to Hardware](ch03-parallelism.md)
4. [Tensor Cores: The `mma` Operations](ch04-mma.md)
5. [Branch and Control: Warp Roles and Persistent Kernels](ch05-branch-control.md)
6. [Synchronization: Pipelines, Events, and Double Buffering](ch06-synchronization.md)
7. [Advanced Data Movement: TMA, Swizzle, and Irregular Access](ch07-advanced-movement.md)
8. [C++ Interop: Inline Code and the Preprocessor](ch08-cpp-interop.md)
9. [Debug and Verbose: Printing, RTTI, and GDB](ch09-debug-verbose.md)

## Prerequisites

- Basic C++ knowledge (functions, pointers, arrays)
- Familiarity with GPU programming concepts (threads, blocks, shared memory)
- A working Croktile compiler (see [Chapter 0](ch00-installation.md))
