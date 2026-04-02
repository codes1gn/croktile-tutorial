# 如何优化鳄霸 2:4 稀疏 GEMM：FP16 与 E4M3 工作记录

本文将梳理在 Hopper（SM90a）上对结构化 2:4 稀疏 GEMM 的优化过程，测量环境为 H800 PCIe（114 个 SM）。同一稀疏模式、同一套元数据叙事、两条数学路径。FP16 内核由 368 TFLOPS 提升至 655 TFLOPS；E4M3 内核由 671 TFLOPS 提升至 1127 TFLOPS。核心思路对二者均适用，但何者首先成为瓶颈有所不同。

稠密 GEMM 教会我们 TMA 与 WGMMA 的节奏配合。稀疏 GEMM 则强调：**在等待元数据时勿使 MMA 挨饿**。这一区分使本文有别于 [稠密 FP16 叙事](dense-gemm-fp16.md)，值得单独阅读。

**FP16（4096 × 8192 × 8192，2:4 结构化稀疏）：**

| 步骤 | 内核 | TFLOPS | 相对基线 Δ |
| ---- | ------ | ------ | ------------- |
| 0 | 基线：1p1c，TK64，2 级流水 | 368 | — |
| 1 | 最佳 `.co`：1p2c + 3 级流水（iter120） | 434 | +18% |
| 2 | 手写 `.cu`：内层展开 24 + FTZ（iter137） | 543 | +47% |
| 3 | TK128，TMA 元数据，拆分 RHS TMA（iter143） | **655** | **+78%** |

**E4M3（同形状）：**

| 步骤 | 内核 | TFLOPS | 相对基线 Δ |
| ---- | ------ | ------ | ------------- |
| 0 | 基线：1p1c，swizzle 128/128，2 级流水 | 671 | — |
| 1 | TMA 元数据分级（iter001） | 759 | +13% |
| 2 | Early empty + 合并 barrier（iter016） | 772 | +15% |
| 3 | 软件流水线 + warpgroup_wait（iter023） | 811 | +21% |
| 4 | 1p2c（iter036） | 897 | +34% |
| 5 | 3 级流水（iter040） | 1090 | +62% |
| 6 | Early empty arrive（iter068） | **1127** | **+68%** |

## 2:4 结构化稀疏的代价

沿稀疏轴（类权重操作数上的 K），每四个连续元素保留两个非零；另两个为零。硬件借助**元数据**——指明稀疏 MMA 路径上哪些 lane 有效的小型索引数组——使核心获取打包后的非零，而非按稠密矩阵处理。

![2:4 sparsity pattern and metadata](images/SparsityPattern_ManimCE_v0.19.1_dark.png#only-dark)
![2:4 sparsity pattern and metadata](images/SparsityPattern_ManimCE_v0.19.1_light.png#only-light)

稀疏侧沿 K 可获得 2× 压缩。代价是明确的：**元数据流量**与**指令开销**与操作数流量并存。每块元数据体量不大，但每个 K 迭代都会触及。若标量加载未命中 L2，其行为便如同在宽 TMA 旁进行指针追逐——因此后续对元数据加载做向量化与提升（hoisting）可带来可测收益。

---

## FP16 基线：368 TFLOPS

起始内核采用 1p1c warp 特化、LHS 打包 swizzle 64、TK=64 与 2 级操作数环形缓冲。在 368 TFLOPS 下调度并非「坏掉」，而是**偏浅**且**偏保守于元数据**。TK64 与两级流水在数学路径旁留给隐藏元数据延迟的余地很小。

## E4M3 基线：671 TFLOPS

E4M3 基线自始即采用更偏 FP8 的配置：1p1c、两侧 swizzle 128/128、预打包稀疏操作数与 2 级流水线。671 TFLOPS 约为 FP8 峰值 3026 TFLOPS 的 22%——在深度分级之前是合理的起点。

---

## 同步与 warpgroup 调参

在加宽分块或增加流水级数之前，应确认 warpgroup 级等待未被过度串行化。这是成本最低的修复项。

**细粒度等待。** 生产者与消费者 warpgroup 通过异步代理与 barrier 协同。粗粒度等待会在数据已就绪时仍使 lane 空闲。收紧至 `warpgroup_wait<1>`——满足需求的最小等待深度——在 FP16 上约 +4%，亦是 E4M3 iter023 跃迁（811 TFLOPS，结合软件流水线与细粒度等待）的组成部分。

**MMA 批次配置。** Hopper WGMMA 将工作按 K 片段批次划分。划分不佳会使张量核心相对操作数供给「吃不饱」。`--wgmma-split-batch` 在 FP16 上约 +5%。若 Nsight 显示 WGMMA 发射槽出现空隙而 SMEM 已就绪，在归咎 TMA 之前应先复查批次划分。

**Early empty、合并 barrier、early arrive（E4M3）。** 异步流水线使用空/满相位；过晚的信号会侵占重叠空间。iter016（772 TFLOPS）采用 early empty 加合并 barrier。iter068（1127 TFLOPS）进一步用 early empty arrive 细化信号时机。在 E4M3 上超过约 900 TFLOPS 后，同步层面的打磨可换来两位数的 TFLOPS 收益。

---

## 元数据供给：稀疏与稠密的分歧点

本节在 [稠密 FP16 叙事](dense-gemm-fp16.md) 中不存在。元数据是第二块操作数平面，必须与矩阵数据一并持续喂饱。

![Metadata delivery: scalar vs TMA-staged](images/MetadataBottleneck_ManimCE_v0.19.1_dark.png#only-dark)
![Metadata delivery: scalar vs TMA-staged](images/MetadataBottleneck_ManimCE_v0.19.1_light.png#only-light)

**只读缓存路径。** 对元数据强制采用 `__ldg` 风格加载，在 FP16 上约 +0.5%。幅度不大，但有助于在各分块间建立一致策略。

**向量化与提升。** 三项改动构成同一叙事——元数据如何在 MMA 消费之前进入寄存器：

| 改动 | FP16 Δ | 作用 |
| ------ | ------ | ------------ |
| 面向 L2/128B 的分组 | +0.7% | 将元数据对齐到缓存行边界 |
| `uint2` 元数据向量化 | +8% | 每条指令加载 2× 元数据 |
| 提升后的 `__ldg` 元数据 | +7% | 将元数据加载移到 K 内层循环之前 |

上述百分比均为局部（各步相对前一次编辑）。它们不能简单相乘，因为存在交互——向量化之后，提升的权重更大。

**TMA 元数据分级——最强一招。** 将元数据置于与操作数相同的异步机制上。由生产侧通过 TMA 将元数据分块预取到共享内存，而非在 K 循环内做标量加载。

在 E4M3 上，这对应 **iter001（759 TFLOPS）**——即第一项优化，相对基线 +13%。在 FP16 上，TMA 支撑的元数据作为 iter143 组合的一部分出现，与 TK128、拆分 RHS TMA 一并引入；在 FP16 上更难先行引入，因 `.co` 编译器无法表达 TMA 描述符相关代码。

**需自担的风险：** 若与重打包操作数对应的元数据错误，会产生静默数值错误。在改变加载路径时，应对小规模尺寸做主机端校验。TK 变化时，需比对元数据偏移与片段边界。

---

## 1p2c 与 3 级流水的阶跃

在 1p1c 中，单个生产者 warpgroup 发出全部 TMA，并常吸收侵占消费者发射槽的设置工作。1p2c 增加第二个消费者 warpgroup——更强的数学吞吐以跟上数据供给。

**FP16（iter120，434 TFLOPS）：** 最佳 `.co` 结果——1p2c + 3 级流水。相对链上前一步约 +9%。

**E4M3（iter036，897 TFLOPS）：** 单独启用 1p2c，尚未改 3 级流水。相对基线 +34%。

随后出现 3 级流水的阶跃：

![E4M3: the 3-stage discontinuity at iter040](images/ThreeStagePipelineJump_ManimCE_v0.19.1_dark.png#only-dark)
![E4M3: the 3-stage discontinuity at iter040](images/ThreeStagePipelineJump_ManimCE_v0.19.1_light.png#only-light)

**E4M3（iter040，1090 TFLOPS）：** 此为突破——相对基线 +62%。跨越 1000 TFLOPS。这并非渐进改进，而是**阶跃函数**，表明流水线从「生产者经常停顿」转为「生产者持续领先」。三级流水使生产者可跑在消费者之前，在数学背后隐藏 TMA 与元数据延迟。

**SMEM 与占用率。** 从两级推到三级会增加 SMEM 占用。若占用率崩溃，数学收益将化为乌有。E4M3 在 3 级上的跃迁表明 SM 仍有裕量——元数据分级与 warp 特化在加深深度之前已降低每 block 的 SMEM。当三种独立改动均不再推动 TFLOPS 时，应停止追求更深流水线。

---

## 内层循环、尾声与分块几何

**`stmatrix`。** 累加器的存矩阵路径在 FP16 上约 +2%。在 E4M3 超过 1000 TFLOPS 后，除非剖析显示存储为热点，尾声重要性下降。

**内层展开与 FTZ（FP16 iter137，543 TFLOPS）。** 编译器生成的 `.co` 调度可能对 K 内层循环展开不足，无法重叠地址运算、元数据预取与 WGMMA。iter137 的手写 `.cu` 使用 **展开 24** 与 **FTZ** 以降低非正规数（denorm）代价。这是在 iter143 之前最关键的一步手写 `.cu`，也是首次需要离开 `.co` 世界的一步。

**TK128、TMA 元数据、拆分 RHS TMA（FP16 iter143，655 TFLOPS）。** TK64 使 K 分块偏小，抬高迭代次数与单位工作量的元数据流量。iter143 合并三项结构改动：TK128（减半 K 循环趟数）、TMA 元数据（异步元数据平面）、拆分 RHS TMA（带宽贴合消费者需求）。结果：**相对基线 +78%**。这不是细部打磨——而是 `.co` 编译器无法表达的结构化存储体系工作。

E4M3 自基线已采用 128/128 swizzle。对应关系是 iter001 元数据 + iter040 深度，而非字面复制 FP16 旋钮。未经验证勿将 FP16 的 swizzle 64 套到 E4M3 的 128/128 上——bank 冲突行为会变化。

---

## .co 平台期与 .cu 突破

FP16 叙事在自动化所能触及与需要人工 CUDA 之间有一条清晰界线：

```
368 ─── .co automation ───> 434 (iter120, +18%)  ═══ CEILING ═══
                                                      │
434 ─── hand .cu ─────────> 543 (iter137, unroll+FTZ)
543 ─── hand .cu ─────────> 655 (iter143, TK128+TMA meta+split RHS)
```

即从基线到最佳 `.co` 为 +18%，而从 iter120 到 iter143，一旦具备 CUDA 级控制力则为 +51%。第二段依赖**不同的可表达性**：鳄霸的 `.co` 编译器负责循环嵌套、寄存器分配与异步代理放置，但稀疏 GEMM 将操作数 TMA、元数据与 WGMMA 批次及 warpgroup barrier 耦合在一起。当编译器以单一 pragma 无法修复的方式将元数据消费与 MMA 串行化时，需要 `.cu` 层面的施展空间。

E4M3 没有类似的断崖。从 671 到 1127 的搜索始终停留在自动化领地，3 级流水与 barrier 工作带来最突出的结构性收益。基线更强的 TMA 与布局选择意味着编译器本就不易绊脚。

---

## 数据类型之间可迁移的内容

复制**因果结构**，而非参数逐项相等：

| 模式 | FP16 | E4M3 |
| ------- | ---- | ---- |
| 元数据置于 TMA 平面 | iter143（较晚，`.cu`） | iter001（较早，自动化） |
| 细 warpgroup 同步 | 链上前段 | iter023 |
| 1p2c | iter120 | iter036 |
| 3 级深度 | iter120（捆绑） | iter040（>1000 的跃迁） |
| Barrier 微优化 | 次要 | iter016、iter068 |

若 FP16 卡在约 450 TFLOPS 以下，应优先攻克元数据向量化、`__ldg`、1p2c + 3 级流水与 `warpgroup_wait`。若 E4M3 已达约 850+ TFLOPS，barrier / early-empty / arrive 与级数调参往往胜过继续加宽操作数。

---

## 结论

稀疏 GEMM 优化与稠密遵循同一弧线——先调度、再加宽、最后调缓存——但多出**作为第二操作数平面的元数据**。元数据叙事正是 2:4 稀疏内核难于稠密之处：WGMMA 可能已喂饱就绪，而元数据加载路径仍在 L2 中指针追逐。一旦元数据上了 TMA 平面且流水线深度足够，稠密中的 1p2c 与 barrier 打磨同样适用。

FP16 案例表明 `.co` 自动化存在边界——最后 +51% 需要手写 `.cu`。E4M3 案例则表明，在更强基线选择下，自动化可将性能从 671 带到 1127，而无需离开 `.co` 世界。

迭代表：`README_gemm_sp_f16_aitune_2026-03-25.md` 与 `README_e4m3_aitune_2026-03-21.md`。内核产物：`benchmark/performance/gemm_sp/`。
