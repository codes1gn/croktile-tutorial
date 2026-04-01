# Croktile

Documentation and tutorials for **Croktile**, a C++ embedded DSL for high-performance kernel programming, featuring easy-to-use syntax with zero-cost abstractions, comprehensive compile-time safety checks, first-class dynamic shapes and symbolic dimensions, and an AI-tuning-friendly design from the ground up.

## Documentation Structure

### Part I — [Tutorial](tutorial/index.md)

A step-by-step walkthrough for newcomers. Starts with the simplest possible kernel and progressively introduces Croktile syntax through increasingly sophisticated matmul variants. No prior Croktile experience required — just basic C++ and GPU programming concepts.

### Part II — [Performance Tuning Demos](optimization/index.md)

Case-oriented performance engineering. Each section profiles a real kernel family, identifies bottlenecks, and applies optimization patterns to push toward hardware peak. Covers profiling, benchmarking, compiler flags, and AI-tuning workflows.

### Part III — [Coding Reference](documentation/index.md)

Exhaustive syntax and semantics reference. Every Croktile construct — shapes, DMA, loops, parallelism, events, memory, macros, and more — is documented in detail.

### Part IV — [Design Rationales](advanced/index.md)

Deep-dive articles on Croktile's design rationale, tradeoffs, and frontier challenges. *(Planned)*
