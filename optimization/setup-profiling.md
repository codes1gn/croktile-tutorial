# Setting Up: TimerOption, TFLOPS, and HW Efficiency

This page is the shared measurement contract for the optimization walkthroughs. You will see the same ideas in every matmul host program: Croktile wraps the kernel in `crok::timing` with a `crok::TimerOption`, prints mean milliseconds, then derives TFLOPS and sometimes efficiency vs peak. Once that pipeline is familiar, you can read each case study as an A/B story on the same yardstick.

## How `crok::timing` Works

`TimerOption` carries **`warmup`** and **`repeat`**. Warmup runs are executed but excluded from the average so caches, TLBs, and steady-state behavior settle first. The timed runs are averaged; the reported value is **mean elapsed time in milliseconds**.

Host code often reads overrides from the environment, then times a lambda that launches the kernel and synchronizes:

```cpp
int warmup = 10;
int repeat = 500;
const char* warmup_env = std::getenv("CROKTILE_TIMING_WARMUP");
const char* repeat_env = std::getenv("CROKTILE_TIMING_REPEAT");
if (warmup_env) { int value = std::atoi(warmup_env); if (value >= 0) warmup = value; }
if (repeat_env) { int value = std::atoi(repeat_env); if (value > 0) repeat = value; }
crok::TimerOption topt;
topt.warmup = warmup;
topt.repeat = repeat;
auto avg_ms = crok::timing([&]() { matmul(lhs_d, rhs_d, res_d); cudaDeviceSynchronize(); }, topt);
std::cout << "Timing avg ms: " << avg_ms << "\n";
```

Include whatever belongs in a fair wall-clock slice—typically the launch plus `cudaDeviceSynchronize()` so the GPU has finished before the timer stops.

## From Average Milliseconds to TFLOPS

For dense `C = A × B` with shapes `(M, K)` and `(K, N)`:

`FLOPs = 2 * M * N * K`

Each output element does `K` multiply-add pairs, hence the factor of two. Sparse variants count **effective** multiply-adds for nonzeros; if a benchmark documents FLOPs differently (e.g. MACs counted as one op), align with that host program rather than forcing this formula everywhere.

With average time `avg_ms`:

`TFLOPS = (FLOPs / (avg_ms / 1000.0)) / 1e12`

```cpp
double flops = 2.0 * double(M) * double(N) * double(K);
double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
std::cout << "TFLOPS: " << tflops << "\n";
```

## Hardware Efficiency (Percent of Peak)

Throughput needs a ceiling. Many benchmarks print efficiency as TFLOPS divided by a documented GPU peak for that precision, times 100:

```cpp
double eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
std::cout << "HW efficiency: " << eff << "%\n";
```

Reference peaks used in Croktile matmul benchmarks for **H800 PCIe**:

| Constant | Value (TFLOPS) | Use |
| -------- | ------------- | --- |
| `H800_PCIE_PEAK_F16_TFLOPS` | 1513 | FP16 dense and similar |
| `H800_PCIE_PEAK_F8_TFLOPS` | 3026 | FP8 dense and similar |

Those numbers are theoretical peaks; real kernels rarely sit at 100%. Use the same peak constant before and after a change so you are comparing apples to apples, not treating the percentage as a final grade.

## Environment, Compile, and Run

| Variable | Default | Effect |
| -------- | ------- | ------ |
| `CROKTILE_TIMING_WARMUP` | `10` | Warmup iterations (non-negative; `0` disables warmup). |
| `CROKTILE_TIMING_REPEAT` | `500` | Timed iterations (must be positive to take effect). |
| `CROKTILE_DISABLE_TIMING` | unset | Set to `1` to skip timing (compile or correctness only). |
| `CROKTILE_SKIP_VERIFY` | unset | Set to `1` to skip numerical verification (faster iteration when you trust the math). |

Verification compares the kernel against a reference and adds host work. For timing-focused runs after layout or precision are stable, `export CROKTILE_SKIP_VERIFY=1` keeps the measurement focused on the device path. Turn verification back on when you change data layout, precision, or tiling.

Performance `.co` files are built through the Croktile driver: pass **`-gs`** (generate script), **`-t cute`**, **`-arch`**, and codegen flags; **`-o`** points at a shell script (often under `/tmp`); run that script with **`--execute`** to compile and run. Keep flags fixed when you are hunting regressions—change one knob at a time.

```bash
./croktile -gs -t cute -arch=sm_90a --use-warpspec --stmatrix \
  benchmark/performance/matmul/matmul_f16_dyn_sm90.co \
  -o /tmp/matmul.cute.result && bash /tmp/matmul.cute.result --execute
```

SM90-class matmul command lines often include flags such as **`--use-warpspec`**, **`--stmatrix`**, **`--hoist-offset`**, **`--hoist-scale`**, **`--ptx-barrier`**, **`--tma-cluster-aware`**, and **`--wgmma-wait-depth=N`**. Exact semantics live in Croktile’s CLI help; copy the recipe from the benchmark you are reproducing, then vary flags deliberately.

With warmup/repeat, TFLOPS, efficiency, env vars, and the compile-and-run loop in hand, the case studies read as one measurement harness and many controlled kernel experiments.
