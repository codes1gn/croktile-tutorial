# Croktile Coding Reference

Welcome to the Croktile coding reference. This section provides exhaustive syntax and semantics documentation for **Croktile**, a C++ embedded DSL for high-performance kernel programming featuring easy-to-use syntax with zero-cost abstractions, comprehensive compile-time safety, first-class dynamic shapes, and an AI-tuning-friendly design.

## What You Will Find Here

This reference covers every aspect of programming with Croktile:

1. **Program structure**: How Croktile programs are organized — the `__co__` function, host-side APIs, and C++ interoperability.

2. **Shaped data**: Croktile's type system for multi-dimensional tensors — shapes, MDSpan, spanned data, symbolic dimensions, and dynamic shapes.

3. **Loop and parallelism**: SPMD parallelism, loop control, tiling, and iteration constructs.

4. **Data movement**: DMA statements, tiling with `chunkat`, TMA, swizzle, and advanced data movement patterns.

5. **C++ embeddings**: Input/output conventions, `__cpp__` inline blocks, macros, and preprocessor integration.

6. **MPMD programming**: Thread masking, events, async execution, and warp specialization.

7. **Optimization patterns**: Tileflow optimizations, async DMA patterns, multi-buffering, and performance tuning techniques.
