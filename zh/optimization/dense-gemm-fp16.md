# 如何将鳄霸 FP16 GEMM 优化到接近 cuBLAS 的性能：一份工作日志

本文中，我将以迭代方式优化一个在鳄霸中编写的 Hopper（SM90a）半精度矩阵乘。目标并非打造 cuBLAS 的替代品，而是借助鳄霸的抽象——warp 特化、TMA 流水线、分块几何与编译器标志调参——深入理解 H800 GPU 的性能特征。可在鳄霸仓库的 `benchmark/performance/matmul/` 下找到全部内核源码。

就当前而言，GPU 上的矩阵乘法或许是最重要的算法之一：大规模深度学习模型在训练与推理中，几乎全部的 FLOPs 都消耗在矩阵乘上。那么，把一个「能跑」的正确鳄霸 SGEMM 推到「与 cuBLAS 相当」需要多少工作量？从基线出发，逐步施加优化，我们可达到 cuBLAS 的 101%：

| 步骤 | 内核 | TFLOPS @8192³ | 相对 cuBLAS（约 380） |
| ---- | ---- | ------------- | -------------------- |
| 0 | 基线：1p1c，WN=128，4 级 | 208.7 | 55% |
| 1 | 分块几何：WN=176，STAGES=2 | 242.0 | 64% |
| 2 | 流水线深度：WN=176，STAGES=3 | 354.1 | 93% |
| 3 | 分离输出 1p2c，WN=128 | ~375.0 | 99% |
| 4 | 分离输出 1p2c，WN=152，非持久化 | **382.5** | **101%** |
| 5 | WN=160，K-unroll，wgmma-wait-depth | 380.6 | 100% |

## 步骤 0：基线

在鳄霸编程模型中，矩阵乘内核是一个描述操作数经 TMA、共享内存暂存与 WGMMA 累加流动的 `__co__` 函数。鳄霸编译器将其转译为带 warp 特化、流水线与 swizzle 寻址的 Hopper 原生 PTX。关键旋钮包括：

- `MATMUL_WARP_N` —— WGMMA 分块的 N 向范围（每个 block 输出有多宽）
- `MATMUL_STAGES` —— 沿 K 的操作数环形槽位数（异步流水线深度）
- Warp 特化模式 —— 1p1c（一个生产者、一个消费者）或 1p2c

基线内核 `matmul_f16_dyn_sm90.co` 采用 **1p1c** warp 特化（一个 TMA 生产者 warpgroup、一个 WGMMA 消费者 warpgroup——角色定义见 [Chapter 5](../tutorial/ch05-branch-control.md)），**WN=128**，以及 **4 级流水线**（流水线思想见 [Chapter 6](../tutorial/ch06-synchronization.md)）。编译并运行：

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/matmul/matmul_f16_dyn_sm90.co \
  -o /tmp/matmul.cute.result && bash /tmp/matmul.cute.result --execute
```

结果：在 8192³ 上为 **208.7 TFLOPS**。抽象地看并不差——但离硬件能力尚远。

![Baseline 1p1c pipeline with 4-stage ring](images/BaselineKernel_ManimCE_v0.19.1_dark.png#only-dark)
![Baseline 1p1c pipeline with 4-stage ring](images/BaselineKernel_ManimCE_v0.19.1_light.png#only-light)

### 最快可能运行时间的下界

对于边长 S = 8192 的方阵 GEMM，总工作量为 `2 × S³ ≈ 1.1 TFLOP`（每个输出元素做 S 次乘加；每次乘加计为 2 FLOPs——一次乘法、一次加法，通常融合为单条 FMA 指令）。H800 PCIe 标称约 1513 TFLOPS FP16 张量峰值与约 3.35 TB/s HBM3 带宽。

若达到张量吞吐峰值，计算约需 0.7 ms。最小访存量（两个 FP16 输入矩阵加一个输出，假设完美复用：`3 × 8192² × 2B ≈ 384 MB`）在峰值带宽下约需 0.15 ms。因而在理想情形下，该内核明确处于 **compute-bound（计算受限）** ——计算所需时间约为访存的约 5 倍。

但 cuBLAS 在此栈上仅约 **380 TFLOPS**，约为理论张量峰值的 25%。这一差距反映真实开销：调度、同步、流水线气泡、占用率、指令混合等。因此 **380 TFLOPS**——而非 1513——才是我们的实际目标；基线仅为其 **55%**。

### 为何 208.7 是调度受限，而非实现错误

内核对 TMA、WGMMA 与 warp 特化的使用是正确的。它是 **under-scheduled（调度不足）**：分块宽度、级数深度与输出暂存与当前问题规模下的占用率及争用特征不匹配。

无需分析器，综合三点即可看出。第一，**吞吐与问题规模**：同一内核在 2048³ 约 **204 TFLOPS**、在 8192³ 为 208.7——数值一致表明瓶颈在 block 内部（调度），而非 grid 层面（wave quantization）。第二，**共享内存与占用率**：WN=128 且 4 级时，每 block 的 SMEM 占用约 96 KB——Hopper 每 SM 228 KB 预算下勉强可容纳 2 个 CTA/SM。第三，**1p1c 中的角色平衡**：单生产者单消费者时，若消费者在 `wait_full` 上停顿、或生产者在 `wait_empty` 上停顿，流水线即受气泡限制。Warp 特化分配角色；它本身并不会把环形缓冲区尺寸调到消除气泡。

### 占用率算术

以下计算将指导后续每一项优化：

```
SMEM per block ≈ STAGES × (WM × TK + WN × TK) × sizeof(fp16)
               = 4 × (64 × 64 + 128 × 64) × 2B
               = 4 × 12288 × 2 ≈ 96 KB
```

在 96 KB 时，SM 的 228 KB 预算可容纳 **2 个 block**。SMEM 的任何增加——更宽的 WN、更多级数、输出暂存——都可能使每 SM 仅余 1 个 block。这是延迟隐藏的阶跃式损失，而非温和退化。下文每一项优化都通过这一算术相互牵制。

---

## 步骤 1：分块几何 —— WN=176，STAGES=2

**问题。** 四级占用大量 SMEM，留给并发 CTA 的空间很小。流水线结构正确，但对占用率预算而言过深。

**改动。** 将 N 向分块加宽至 176（每个已分段的 K 板层上算更多数学），并降为 2 级：

```
MATMUL_WARP_N = 176    # was 128
MATMUL_STAGES = 2      # was 4
```

新的 SMEM：

```
SMEM ≈ 2 × (64 × 64 + 176 × 64) × 2B
     = 2 × 15360 × 2 ≈ 60 KB
```

每 block 约 60 KB 时，SM 可容纳 **3 个 block**——由 2 个增加。更多并发 block 意味着跨 CTA 更好的延迟隐藏。

**为何 WN 重要——算术强度。** 更宽的 N 分块意味着每从 GMEM 载入 SMEM 一字节，每个 block 计算更多输出元素。WN=128 时：`AI = 2 × 64 × 128 × 64 / ((64 + 128) × 64 × 2) ≈ 42.7 FLOPs/B`。WN=176 时：`AI ≈ 46.9 FLOPs/B`。约 10% 的强度提升有用，但更大收益来自腾出 SMEM 以改善占用率。

**结果：** 在 2048³ 上为 **242 TFLOPS**（+18%）。但 2 级使流水线偏浅——TMA 延迟未能充分隐藏。

---

## 步骤 2：流水线深度 —— STAGES=3

**问题。** 在 2 级时，生产者加载完下一块 K 板层后会在 `wait_empty` 上阻塞——消费者尚未释放上一缓冲区。

**改动。** 增加一级：

```
MATMUL_STAGES = 3      # was 2, keeping WN=176
```

新 SMEM：`3 × 15360 × 2 ≈ 90 KB`——在 228 KB 内仍可放 2 个 block。

多出的这一级使生产者相对消费者可超前运行一个 K 板层，在 WGMMA 计算背后隐藏 TMA 延迟。

![3-stage pipeline: producer runs ahead](images/Step2ThreeStage_ManimCE_v0.19.1_dark.png#only-dark)
![3-stage pipeline: producer runs ahead](images/Step2ThreeStage_ManimCE_v0.19.1_light.png#only-light)

**结果：** 在 2048³ 上为 **354.1 TFLOPS**——相对上一步 **+46%**。

这是整段优化中单次最大跃升，且并非来自更多数学运算——而是 **bubble-limited（受气泡限制）** 调度的典型特征。多一级换来了生产者–消费者并发。流水线从「每轮迭代生产者都停」变为「生产者保持超前」。

### 隐患：级数 × 问题规模的相互作用

三级在 2048³ 上有利，但在 8192³ 上可能有害，因为更大 grid 会放大占用率效应。多出来的级数是字节，可能挤掉并发 block。问题规模改变 4× 时，应重新扫描 STAGES。这也是后续步骤会再次调整 WN/STAGES 平衡的原因。

---

## 步骤 3：分离输出 1p2c

**问题。** 单一消费者 warpgroup（1p1c）时，共享内存中只有一个 `output_s` 分块用于累加器暂存。随着 WN 增大，**输出争用**成为瓶颈——消费者在写入该共享分块时串行，累加路径上的 SMEM 流量侵蚀吞吐。

**改动。** 切换为 **1p2c 分离输出**：一个生产者、两个消费者 warpgroup，各自拥有输出暂存区的私有条带。源码：`matmul_f16_dyn_sm90_warpspec_1p2c.co`。

![Split-output 1p2c architecture](images/SplitOutput1p2c_ManimCE_v0.19.1_dark.png#only-dark)
![Split-output 1p2c architecture](images/SplitOutput1p2c_ManimCE_v0.19.1_light.png#only-light)

代价是略高的 SMEM（两片输出），换来更低争用。先在 4096³ 上验证——若分离输出在这一中间尺寸上回退，则不会信任代价更高的 8192³。

**结果：** 在 4096³ 上约 **375 TFLOPS**。

单一分析器计数很少直接显示输出争用。与之相关的经验法则是：仅当启用分离输出时，从 1p1c 换到 1p2c 后 TFLOPS 才上升——说明消费者侧在 `output_s` 上串行，而非在数学路径上。

---

## 步骤 4：最佳标题数字 —— iter057

**改动。** 将分离输出带到完整 8192³ 问题，并调优 WN、采用非持久化 launch：

```
Warp spec:        1p2c split-output
MATMUL_WARP_N:    152
Launch:           non-persistent (conventional grid)
```

[Chapter 5](../tutorial/ch05-branch-control.md) 讨论了通过持久化内核缓解 grid 层面尾波利用率不足的方案。但当 block 内 SMEM 与流水线选择已限制吞吐时，持久化无法挽回占用率已造成的损失。在 8192³ 与分离输出分块下，wave quantization 可接受，且瓶颈已在 block 内部——常规网格更优。

**结果：** 在 8192³ 上为 **382.5 TFLOPS**——相对 208.7 基线 **+83%**，与 cuBLAS 持平。

---

## 步骤 5：WN 扫描与占用率断崖 —— iter061

在分离输出已达到 cuBLAS 量级吞吐之后，问题变为：对 8192³ 而言 WN=152 是否最优，抑或只是从小规模实验继承而来？第三阶段在 8192³ 上扫描 WN，并配合 K-unroll 与 `--wgmma-wait-depth`：

```
MATMUL_WARP_N:    160
K-unroll:         enabled
--wgmma-wait-depth=N (tuned to match stage count)
```

**结果：** **380.6 TFLOPS**——在 8192³ 上比 iter057 低 1.9 TFLOPS，但跨尺寸表现更强（在 2048³ 上为 cuBLAS 的 100.5%）。

扫描还发现了 **WN=168 时的硬失败**：

![The WN=168 occupancy cliff](images/OccupancyCliff_ManimCE_v0.19.1_dark.png#only-dark)
![The WN=168 occupancy cliff](images/OccupancyCliff_ManimCE_v0.19.1_light.png#only-light)

在 WN=168 时，共享内存超过 228 KB。驻留从每 SM 2 个 block 降为 1 个。吞吐断崖式下跌——不是几个百分点，而是延迟隐藏的灾难性丧失。应通过计算 `STAGES × tile_dimensions × element_size` 并与 228 KB 预算比较来发现，而非凭感觉。

---

## 编译器标志：最后一层

函数结构确定之后，编译器如何将鳄霸 lower 到 PTX 仍然重要。发布构建共用一套标志组合：

| 标志 | 作用 |
| ---- | ---- |
| `--use-warpspec` | 针对生产者/消费者划分的 warp 特化代码生成 |
| `--stmatrix` | STSM 风格的共享内存矩阵设置 |
| `--hoist-offset` / `--hoist-scale` | 将地址运算从内层循环中 hoist 出 |
| `--ptx-barrier` | 异步同步用的 barrier 指令 |
| `--tma-cluster-aware` | 面向 SM90 cluster/multicast 偏置 TMA lowering |
| `--wgmma-wait-depth=N` | 将 WGMMA 流水线等待深度暴露为可调参数 |

标志很重要——iter023 仅 `--ptx-barrier` 与 `--stmatrix` 就在 2048³ 上带来约 +5%。但完整日志的教训是 **操作顺序**：扫描 WN 与 STAGES 时冻结标志；仅在分离输出落地后再解冻。在 SMEM 仍错误时过度调标志是常见失败模式。

---

## 发布检查点与复现

在鳄霸仓库根目录执行 `make build` 之后：

```bash
./croktile -gs -t cute -arch=sm_90a \
  --use-warpspec --stmatrix --hoist-offset --hoist-scale \
  --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/<INPUT>.co \
  -o /tmp/run.cute.result && bash /tmp/run.cute.result --execute
```

| 检查点 | `.co` 文件名 | TFLOPS | 规模 | 配置 |
| ------ | -------------- | ------ | ---- | ---- |
| iter048 | `..._iter048_s3_wn176_best.co` | 354.1 | 2048³ | 1p1c，WN=176，3 级 |
| iter050 | `..._iter050_1p2c_splitout.co` | ~375 | 4096³ | 1p2c 分离输出，WN=128 |
| iter057 | `..._iter057_1p2c_so_wn152_nonpersis.co` | **382.5** | 8192³ | 1p2c 分离输出，WN=152 |
| iter061 | `..._iter061_1p2c_so_wn160_kunroll.co` | 380.6 | 8192³ | 1p2c 分离输出，WN=160，K-unroll |

**选择：** 8192³ 峰值标题取 iter057。需要单个二进制在多种尺寸上表现均衡时取 iter061（2048³ 为 cuBLAS 的 100.5%，8192³ 为 80.7%）。

Harness 默认：`CROKTILE_TIMING_WARMUP=10`，`CROKTILE_TIMING_REPEAT=500`。与 `.co` 文件使用同一 revision 构建，以避免 codegen drift。使用 `-arch=sm_90a`。与外部 cuBLAS 数据对比时，注明驱动版本与时钟行为。

---

## 结论

撰写本文对应约 65 次迭代、分三个阶段。第一阶段 38 次迭代约提升 +5%。第二阶段 14 次迭代约提升 +83%。第三阶段打磨最后若干 TFLOPS，并发现 WN=168 的失败。幂律处处可见。

单次最大的结构性收益并非来自某条编译器标志，而是 **1p2c 分离输出** 将 TFLOPS 推入 370–382 区间。`--stmatrix` 等标志重要，但若两个消费者共享同一累加器分块，它们无法消除 `output_s` 上的串行化。若在自己的内核中遇到类似天花板，在动用指令级手段之前，先检查输出路径是否为瓶颈。

+83% 完全来自鳄霸的函数几何、输出暂存与编译器标志——无混合精度、无 split-K、无 CUDA Graph capture。

完整迭代表：`README_matmul_f16_aitune_2026-03-23.md`。源码：`matmul_f16_dyn_sm90.co`、`matmul_f16_dyn_sm90_warpspec_1p1c.co`、`matmul_f16_dyn_sm90_warpspec_1p2c.co`，以及带日期的 `*_iter048_*` 至 `*_iter061_*` 构建。
