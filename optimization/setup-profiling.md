# Setting Up: TimerOption, TFLOPS, and HW Efficiency

Every kernel in the case studies uses the same measurement pipeline. This page shows the code once so the walkthroughs can focus on what changes between kernels.

## Timing a Kernel

`choreo::timing` wraps the kernel in a warmup/repeat loop, excludes warmup from the average, and returns the **mean elapsed time in milliseconds**:

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

The lambda includes `cudaDeviceSynchronize()` so the GPU finishes before the timer stops. Warmup lets caches, TLBs, and steady-state behavior settle.

## Computing TFLOPS

For dense `C = A × B` with shapes `(M, K)` and `(K, N)`, each of the `M × N` output elements requires `K` multiply-add pairs. A multiply-add counts as two FLOPs (one multiply, one add — often a single FMA instruction):

```
FLOPs = 2 × M × N × K
```

For an 8192³ GEMM: `2 × 8192³ = 1,099,511,627,776 ≈ 1.1 TFLOP` of work per kernel call.

With the measured average time:

```cpp
double flops = 2.0 * double(M) * double(N) * double(K);
double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
std::cout << "TFLOPS: " << tflops << "\n";
```

Sparse variants count effective multiply-adds for nonzeros. If a benchmark documents FLOPs differently (e.g. MACs as one op), align with that host program's formula.

## Hardware Efficiency

TFLOPS alone is meaningless without a ceiling. The benchmarks print efficiency as a fraction of the documented GPU peak:

```cpp
double eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
std::cout << "HW efficiency: " << eff << "%\n";
```

Reference peaks for **H800 PCIe**:

| Constant | Value | Use |
| -------- | ----- | --- |
| `H800_PCIE_PEAK_F16_TFLOPS` | 1513 TFLOPS | FP16 dense |
| `H800_PCIE_PEAK_F8_TFLOPS` | 3026 TFLOPS | FP8 dense |

These are theoretical peaks; real kernels rarely hit 100%. In the case studies, we compare against **cuBLAS** on the same hardware (~380 TFLOPS for FP16 dense at 8192³) as a practical ceiling. Use the same peak constant before and after a change — you are comparing deltas, not final grades.

## Environment Variables

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `CROKTILE_TIMING_WARMUP` | `10` | Warmup iterations (0 disables) |
| `CROKTILE_TIMING_REPEAT` | `500` | Timed iterations (must be > 0) |
| `CROKTILE_DISABLE_TIMING` | unset | Set to `1` to skip timing entirely |
| `CROKTILE_SKIP_VERIFY` | unset | Set to `1` to skip numerical verification |

Use `CROKTILE_SKIP_VERIFY=1` only when you trust correctness. A fast wrong kernel sends the optimization search backward — always re-enable verification after changing data layout, precision, or tiling.

## Compile and Run

Performance `.co` files are built through the Croktile driver. A typical invocation:

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/matmul/matmul_f16_dyn_sm90.co \
  -o /tmp/matmul.cute.result && bash /tmp/matmul.cute.result --execute
```

Common SM90 flags: `--use-warpspec`, `--stmatrix`, `--hoist-offset`, `--hoist-scale`, `--ptx-barrier`, `--tma-cluster-aware`, `--wgmma-wait-depth=N`. Exact semantics live in Croktile's CLI help. Copy the recipe from the benchmark you are reproducing, then vary one flag at a time.
