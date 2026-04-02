# 环境搭建：TimerOption、TFLOPS 与硬件效率

案例研究中的每个内核均采用相同的测量流水线。本文将相关代码集中展示一次，以便各篇分步讲解能够聚焦于不同内核之间的差异。

## 对内核计时

`choreo::timing` 将内核包在预热/重复循环中，将预热排除在平均值之外，并返回**平均耗时（毫秒）**：

```cpp
int warmup = 10;
int repeat = 500;
const char* warmup_env = std::getenv("CROKTILE_TIMING_WARMUP");
const char* repeat_env = std::getenv("CROKTILE_TIMING_REPEAT");
if (warmup_env) { int value = std::atoi(warmup_env); if (value >= 0) warmup = value; }
if (repeat_env) { int value = std::atoi(repeat_env); if (value > 0) repeat = value; }

choreo::TimerOption topt;
topt.warmup = warmup;
topt.repeat = repeat;

auto avg_ms = choreo::timing([&]() {
  matmul(lhs_d, rhs_d, res_d);
  cudaDeviceSynchronize();
}, topt);

std::cout << "Timing avg ms: " << avg_ms << "\n";
```

lambda 中包含 `cudaDeviceSynchronize()`，以便在停止计时前 GPU 已完成工作。预热可使缓存、TLB 及稳态行为趋于稳定。

## 计算 TFLOPS

对稠密 `C = A × B`，若 `A`、`B` 的形状分别为 `(M, K)` 与 `(K, N)`，则 `M × N` 个输出元素各自需要 `K` 次乘加。一次乘加计为两个 FLOP（一次乘法与一次加法——通常对应单条 FMA 指令）：

```
FLOPs = 2 × M × N × K
```

对 8192³ 的 GEMM：`2 × 8192³ = 1,099,511,627,776 ≈ 1.1 TFLOP`，即每次内核调用的计算量。

结合测得的平均时间：

```cpp
double flops = 2.0 * double(M) * double(N) * double(K);
double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
std::cout << "TFLOPS: " << tflops << "\n";
```

稀疏变体按非零元素的有效乘加次数统计。若某基准对 FLOP 的定义不同（例如将 MAC 计为一次运算），应与该主机端程序的公式保持一致。

## 硬件效率

若无上限参照，单独的 TFLOPS 并无意义。基准测试将效率输出为文档所载 GPU 峰值的一个比例：

```cpp
double eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
std::cout << "HW efficiency: " << eff << "%\n";
```

**H800 PCIe** 的参考峰值：

| 常量 | 取值 | 用途 |
| -------- | ----- | --- |
| `H800_PCIE_PEAK_F16_TFLOPS` | 1513 TFLOPS | FP16 稠密 |
| `H800_PCIE_PEAK_F8_TFLOPS` | 3026 TFLOPS | FP8 稠密 |

上述为理论峰值；实际内核很少达到 100%。在案例研究中，我们在同一硬件上与 **cuBLAS** 对比（8192³ 下 FP16 稠密约 380 TFLOPS）作为实际可达的上限。修改前后应使用同一峰值常量——比较的是增量，而非绝对分数。

## 环境变量

| 变量 | 默认值 | 作用 |
| -------- | ------- | ------ |
| `CROKTILE_TIMING_WARMUP` | `10` | 预热迭代次数（设为 0 则关闭预热） |
| `CROKTILE_TIMING_REPEAT` | `500` | 计时的迭代次数（必须大于 0） |
| `CROKTILE_DISABLE_TIMING` | 未设置 | 设为 `1` 则完全跳过计时 |
| `CROKTILE_SKIP_VERIFY` | 未设置 | 设为 `1` 则跳过数值校验 |

仅在确信正确性可靠时使用 `CROKTILE_SKIP_VERIFY=1`。错误但很快的内核会使优化搜索南辕北辙——在更改数据布局、精度或分块策略之后，务必重新启用校验。

## 编译与运行

性能相关的 `.co` 文件通过鳄霸驱动构建。典型调用如下：

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/matmul/matmul_f16_dyn_sm90.co \
  -o /tmp/matmul.cute.result && bash /tmp/matmul.cute.result --execute
```

常用的 SM90 标志包括：`--use-warpspec`、`--stmatrix`、`--hoist-offset`、`--hoist-scale`、`--ptx-barrier`、`--tma-cluster-aware`、`--wgmma-wait-depth=N`。具体语义见鳄霸 CLI 帮助。请从所要复现的基准中复制命令行配方，再每次只改动一个标志。
