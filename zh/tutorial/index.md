# 鳄霸教程

欢迎来到鳄霸教程。本指南将引导你从零开始，使用鳄霸编写高性能 GPU 内核，并逐步深入到生产级模式。

每章通过推进一个贯穿的示例，引入少量新概念。学完全部章节后，你将接触到鳄霸的所有主要语言构造，并在具体的、可运行的程序中理解它们。详细的语法设计和语言参考请参阅[编程参考](../documentation/index.md)。

## 章节目录

0. [安装：搭建鳄霸编译器](ch00-installation.md)
1. [Hello Croktile：从零到运行内核](ch01-hello-croktile.md)
2. [数据搬运：从逐元素到数据块](ch02-data-movement.md)
3. [并行性：将工作映射到硬件](ch03-parallelism.md)
4. [张量核心：`mma` 操作](ch04-mma.md)
5. [分支与控制：Warp 角色与持久内核](ch05-branch-control.md)
6. [同步：流水线、事件与双缓冲](ch06-synchronization.md)
7. [高级数据搬运：TMA、Swizzle 与不规则访问](ch07-advanced-movement.md)
8. [C++ 互操作：内联代码与预处理器](ch08-cpp-interop.md)
9. [调试与诊断：打印、RTTI 与 GDB](ch09-debug-verbose.md)

## 前置要求

- 基本的 C++ 知识（函数、指针、数组）
- 熟悉 GPU 编程概念（线程、线程块、共享内存）
- 可用的鳄霸编译器（参见[第 0 章](ch00-installation.md)）
