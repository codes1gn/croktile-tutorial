# 同步：流水线、事件与双缓冲

第 5 章将矩阵乘剖成两份工作：作为**生产者**的 warpgroup 负责向共享内存发起加载，作为**消费者**的 warpgroup 则在这些 tile 上执行 MMA。该骨架如实体现了这一划分，却暗中假设了一件不可能的事：消费者可以像共享内存一写入就立刻更新那样对待它。真实硬件并非如此。若无协调，消费者可能在写入尚未完成时就读取 tile，或生产者可能踩到仍在使用中的缓冲区——典型的**数据竞争**。

本章讲述同一条主线：**如何使流水线执行安全。**鳄霸（Croktile）提供 **event**，使不同角色能够就就绪状态发信号；提供 **`swap` / `rotate`**，使双缓冲或多缓冲在源码中仍清晰可读；提供 **`dma.copy.async`**，使加载可与计算重叠；并提供**序幕（prologue）/ 稳态（steady-state）/ 收尾（epilogue）**模式以组织 K 循环。我们从单线程情形（同一程序计数器上既加载又计算）出发，再在生产者与消费者真正分叉时引入 event。

## 为何不能不经协调就「单纯重叠」

用**一个**暂存缓冲区遍历分块矩阵乘的 K 循环时，每次迭代必须：将 A、B 的 tile 拷贝到共享内存，等待这些拷贝对后续读可见，再在该 tile 上运行 MMA。若试图在 MMA 仍读取**同一**缓冲区时启动*下一轮*迭代的拷贝，就会**覆盖**张量核心正在消费的字节。无论调度如何一厢情愿都无法消除该问题；你需要在「字节已落盘」与「MMA 读取它们」之间建立 happens-before 关系。

手写 CUDA 时，人们用**屏障**（`__syncthreads`）、**原子操作**或跨流线的 **CUDA event** 来强制这一点。簿记很快爆炸：你要跟踪哪个阶段拥有哪个缓冲区、哪道栅栏消除哪类冒险，并指望循环的每条路径都对齐。鳄霸收窄设计空间：**event** 用于跨角色发信号，**`swap`** 用于旋转缓冲区*名称*而不搬运数据，显式的 **`wait` / `trigger`** 则使信用流在源码中保持可见。

下图对比严格的**先加载后计算**阶梯与**双缓冲**重叠：逻辑工作量相同，但在流水线填满时，存储侧或算术侧的空闲时间更少。

![顺序执行与双缓冲 K-tile 时间线（示意图）](../assets/images/ch06/fig1_pipeline_timeline_dark.png#only-dark)
![顺序执行与双缓冲 K-tile 时间线（示意图）](../assets/images/ch06/fig1_pipeline_timeline_light.png#only-light)

## 使用 `swap` 的双缓冲

为 K 循环赋予**两个**逻辑缓冲区。当 MMA 正在排空缓冲区 0 时，DMA 将下一块 tile 填入缓冲区 1。数学步骤结束后，**交换**句柄：原先的「下一」变为「当前」，腾出的槽位即可用于后续加载。鳄霸用 `dma.copy.async`（非阻塞拷贝）、`dma.any`（占位 future）、`swap`（交换 future）以及三阶段循环来表达这一点。

```choreo
__co__ auto matmul(s32 [M, K] lhs, s32 [K, N] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel {px, py} by [8, 16] : block
    parallel {qx, qy} by [16, 16] : thread {

    with tile_k in 16 {
      // Prologue: start loading tile 0
      lf0 = dma.copy lhs.chunkat(px, tile_k) => shared;
      rf0 = dma.copy rhs.chunkat(tile_k, py) => shared;

      // Placeholder futures for buffer 1
      lf1 = dma.any;
      rf1 = dma.any;

      // Steady state: load next tile while computing on current
      foreach tile_k(1:) {
        lf1 = dma.copy lhs.chunkat(px, tile_k) => shared;
        rf1 = dma.copy rhs.chunkat(tile_k, py) => shared;

        foreach k in [256 / #tile_k]
          output.at(px#qx, py#qy) += lf0.data.at(qx, k) * rf0.data.at(k, qy);

        swap(lf0, lf1);
        swap(rf0, rf1);
      }

      // Epilogue: compute on the last loaded tile
      foreach k in [256 / #tile_k]
        output.at(px#qx, py#qy) += lf0.data.at(qx, k) * rf0.data.at(k, qy);
    }
  }

  return output;
}
```

### **`with tile_k in 16`**

```choreo
with tile_k in 16 {
```

打开**作用域区域**，并将 `tile_k` 绑定为范围为 16 的 tile 轴。在该块内，`tile_k` 是沿 K 方向 `chunkat` 的分块索引，`#tile_k` 为 16——即「在此作用域内，K 被划分为 16 个 tile」。

### **`dma.any`：占位 future**

```choreo
lf1 = dma.any;
rf1 = dma.any;
```

`dma.any` 是尚不代表任何传输的 future。它的存在使得类型系统在稳态的第一次迭代时有可与 `swap` 交换的对象。在任何使用 `lf1.data` 之前，已赋值为真实的 `dma.copy`。

### **`foreach tile_k(1:)`：切片迭代**

```choreo
foreach tile_k(1:) {
```

`(1:)` 表示 tile 索引从 `1, 2, …` 直至末尾。tile 0 已在序幕中加载到 `lf0`/`rf0`。

### **三阶段**

**序幕。**向 `lf0`/`rf0` 发起 tile 0 的加载。尚无计算。

**稳态。**对每一块后续 tile：向 `lf1`/`rf1` 启动加载，在上一轮迭代的 `lf0`/`rf0` 上计算，再 `swap`，使名称跟踪活动缓冲区。新的拷贝在计算读取 `lf0`/`rf0` **之前**就已落到 `lf1`/`rf1`，因而永远不会在缓冲区正被覆盖时去读它。

**收尾。**最后一次 `swap` 之后，`lf0`/`rf0` 持有最后一块 tile；再执行一轮计算将其排空。

### **`swap`：交换的是名称，不是字节**

`swap(lf0, lf1)` 交换的是 **future 句柄**。共享内存中的内容仍留在硬件放置的位置；仅鳄霸层面的名称在轮换。CUDA 中的同类写法常是 `^ 1` 的缓冲区下标或布尔相位；此处意图是显式的。对于三缓冲，`rotate(f0, f1, f2)` 一步轮换三个句柄。

*注：本示例使用 `s32` 与标量累加以及 `with tile_k` 作用域——刻意简化的内核风格，用于隔离 `swap` 机制。同样的 `swap` 模式原样适用于第 4 章的 FP16/MMA 内核；变化的仅是计算部分。*

### **`auto` 返回类型**

`__co__ auto matmul(...)` 允许鳄霸从 `return output` 推断结果类型，使签名与形状表达式一致。

## Event：当生产者与消费者是不同程序时

当**单一**线程组在单一调度中交错加载与 MMA 时，`swap` 可用。Warp 特化（第 5 章）将加载与计算放在**不同**的 warpgroup 上，且程序计数器不同。它们无法逐行共享 `swap` 调度；需要 **event**——共享作用域中的具名同步。

```choreo
shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
```

`wait event_name` 阻塞直至该 event 被置位；`trigger event_name` 唤醒等待者。对于 1P1C 矩阵乘，常见约定为：

- `full[s]` — 阶段 `s` 已填满；消费者可以读取。
- `empty[s]` — 消费者已释放阶段 `s`；生产者可以覆盖。

多线程共享的 tile 暂存仍放在 **`=> shared`** 中；第 2、3 章已讨论 **local** 与 **shared** 的放置——此处新要素是*谁*在*哪个* event 上等待，而不仅是存储空间本身。

### **带 event 的 1P1C 内核**

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
    shared f16 [MATMUL_STAGES * MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_STAGES * MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait empty[stage];
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k)
            => lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0);
          dma.copy rhs.chunkat(block_n, iv_k)
            => rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0);
          trigger full[stage];
        }
      }

      inthreads.async (p1 == 1) {
        mc = mma.fill.f16 0.0f;
        foreach {s} in [MATMUL_STAGES] {
          trigger empty[s];
        }
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          stage = iv_k % MATMUL_STAGES;
          wait full[stage];
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mb = mma.load rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
          mma.commit;
          trigger empty[stage];
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**环形索引。**`stage = iv_k % MATMUL_STAGES` 将无界的 K 迭代映射到固定数量的物理槽位——双缓冲推广到 N 缓冲。若有四个阶段，生产者可在阻塞于 `wait empty` 之前多跑若干块 tile。

**生产者路径。**对每个 `iv_k`，`wait empty[stage]` 取得空闲槽，`dma.copy` 行在对应 `stage` 填充 `lhs_load_s`/`rhs_load_s`，随后 `trigger full[stage]` 将该槽交给消费者。若省略消费者序幕中的初始 `trigger empty` 引导，则第一次 `wait empty` 永远完不成——这是死锁，而非神秘的 MMA 错误。

**消费者路径。**循环 `foreach {s} in [MATMUL_STAGES] { trigger empty[s]; }` 在 K 循环**之前**运行，使每个阶段以 **empty** 信用开始。否则生产者会在第一块 tile 上永远等待。随后每个 `iv_k`：`wait full[stage]`，在该阶段上执行 MMA，`mma.commit`，`trigger empty[stage]` 以释放槽位。跳过 `mma.commit` 或在运算真正完成前发 `empty` 信号，可能导致操作数仍有效时槽被复用——又一种欠同步。

**`mma.commit`。**Hopper 的 WGMMA 将发射与累加重叠；`mma.commit` 是在该阶段的共享缓冲区可被逻辑复用之前，完成该 K 条带对 `mc` 的贡献的栅栏。应将其视为「该阶段结束」与 `trigger empty` 之间必需的粘合剂。

## 单阶段的信用流

下图与代码一致：引导阶段授予 empty 信用；生产者等待 `empty`，填满后发 `full`；消费者等待 `full`，计算后发 `empty`。当 `iv_k` 按 `MATMUL_STAGES` 取模回绕时，同一物理阶段重新进入循环——环形安全是因为 `wait`/`trigger` 串行化访问，而非因为取模算术本身魔法般可靠。

![单级流水线的事件信用流](../assets/images/ch06/fig2_event_credit_flow_dark.png#only-dark)
![单级流水线的事件信用流](../assets/images/ch06/fig2_event_credit_flow_light.png#only-light)

若编辑流水线后行为异常，在追查 MMA 布局之前，请先核对 **event 顺序与触发次数**：生产者与消费者必须使用相同的 `cdiv(K, MATMUL_TILE_K)` 循环，阶段过少则会在消费者快于生产者时把压力压到 `wait full` 上。

## 新语法

| 语法 | 含义 |
|--------|--------|
| `shared event name[N]` | 在 shared 作用域声明 N 个具名同步 event |
| `wait event` | 阻塞直至 `event` 被置位 |
| `trigger event` | 置位 `event`，唤醒等待者 |
| `dma.copy.async src => dst` | 非阻塞拷贝（立即返回） |
| `dma.any` | 占位 future（尚无在途传输） |
| `swap(f0, f1)` | 交换两个 future 句柄，不拷贝数据 |
| `rotate(f0, f1, f2)` | 轮换三个 future 句柄 |
| `with tile_k in N { ... }` | 绑定范围为 N 的作用域 tile 轴 |
| `foreach tile_k(1:)` | 从索引 1 开始迭代 |
| `mma.commit` | WGMMA 流水线阶段之间的栅栏 |
| `__co__ auto fn(...)` | 由 `return` 语句推断返回类型 |

## 小结

| 要点 | 作用 |
|------|------|
| 数据竞争 | 未同步的重叠会使加载破坏 MMA 操作数，或读到未完成的 tile。 |
| CUDA 式修复 | 屏障、原子与手工 event 接线可行，但复杂度扩展性差。 |
| `swap` / `rotate` | 旋转 **future**，使双缓冲或多缓冲在单一程序中显式可见。 |
| `shared event` | 用 `wait` / `trigger` 与信用纪律协调**不同** warpgroup。 |
| 引导 `empty` | 必需，以使生产者的第一次 `wait empty` 能够成功。 |
| `mma.commit` | 将某一 K 条带的已完成数学与共享暂存的复用分离。 |

流水线现已对拆分角色安全。[下一章](ch07-advanced-movement.md)将讨论硬件加速的 **TMA**、**swizzled** 共享布局，以及 **`view` / `from`** 的不规则访问——同步思想相同，底层搬运原语更丰富。
