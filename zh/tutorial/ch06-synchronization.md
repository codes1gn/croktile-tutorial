# 同步：流水线、事件与双缓冲

第 5 章将矩阵乘分解为生产者（加载数据）与消费者（执行 MMA）。但该骨架隐含了一个不切实际的假设：消费者能在生产者写入共享内存的瞬间就读取其中的数据。实际上需要**同步**——一种让生产者宣告「该缓冲区已就绪」，并使消费者等待至该时机的机制。

本章介绍使流水线执行安全所需的基元：用于在角色间发信号的 **event**、用于双缓冲与多缓冲的 **`swap`** 与 **`rotate`**、用于非阻塞传输的 **`dma.copy.async`**，以及将数据搬运与计算重叠的**序幕阶段 / 稳态阶段 / 收尾阶段**模式。

## 问题：顺序的「先加载后计算」

设想朴素分块矩阵乘中的 K 循环。每次迭代：

1. 将 A 分块与 B 分块拷贝到共享内存。
2. 等待拷贝完成。
3. 对已加载的分块执行 MMA。

若仅持有一个缓冲区，则步骤 2、3 无法与**下一次**迭代的步骤 1 重叠——否则会覆盖 MMA 仍在读取的数据。因此时间线呈阶梯状：加载、计算、加载、计算，机器总有一侧处于空闲。

## 借助 `swap` 的双缓冲

解决方法是使用两个缓冲区。当 MMA 读取缓冲区 0 时，生产者向缓冲区 1 填入下一块分块。MMA 结束后，**交换（`swap`）**缓冲区：原先的「下一」块变为「当前」，空出的槽位即可用于后续加载。

鳄霸用语言层面的 `dma.copy.async`（非阻塞拷贝）、`dma.any`（占位 future）、`swap`（交换 future 句柄）以及三阶段循环来表达这一过程。

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

下面说明各新构造的作用。

## `with tile_k in 16`

```choreo
with tile_k in 16 {
```

该语句开启**作用域区域**，并将 `tile_k` 绑定为跨度为 16 的分块轴。块内 `tile_k` 用作沿 K 维 `chunkat` 的分块索引，`#tile_k` 给出其跨度（16）。可理解为：「在此作用域内，K 被划分为 16 个分块。」

## `dma.any`：占位 Future

```choreo
lf1 = dma.any;
rf1 = dma.any;
```

`dma.any` 创建一个尚不代表任何真实传输的 future。其目的在于使类型系统在首次迭代时有可与 `swap` 交换的对象。在任何代码读取 `lf1.data` 之前，真实的 `dma.copy` 会被赋给该句柄。

## `foreach tile_k(1:)`：切片迭代

```choreo
foreach tile_k(1:) {
```

切片 `(1:)` 表示「从索引 1 起迭代 `tile_k`，直至剩余各分块」。分块 0 已由序幕阶段处理——即向 `lf0`、`rf0` 的初始加载。因此稳态阶段循环覆盖分块 1、2、…、15。

## 三个阶段

**序幕阶段。** 将分块 0 加载到 `lf0`/`rf0`。此时尚无计算——缓冲区正在填充。

**稳态阶段。** 对每个后续分块：开始向 `lf1`/`rf1`（「下一」缓冲区）加载，继而在 `lf0`/`rf0`（上一轮迭代已填充的「当前」缓冲区）上计算。计算完成后，`swap(lf0, lf1)` 交换句柄——原先的「下一」在下一轮迭代中成为「当前」。

**收尾阶段。** 最后一次 `swap` 之后，`lf0`/`rf0` 持有最后一个分块；再执行一轮计算将其消费完毕。

顺序至关重要：新的拷贝在计算读取 `lf0`/`rf0` **之前**赋给 `lf1`/`rf1`。这使依赖关系清晰：绝不会在缓冲区正被覆盖的同时从中读取。

## `swap`：交换的是名字，而非数据

`swap(lf0, lf1)` 交换的是 **future 句柄**，而非张量数据。交换后，名称 `lf0` 指向原先 `lf1` 所指的异步操作，反之亦然。已暂存于共享内存中的数据仍留在硬件放置的位置；仅在鳄霸语言层面旋转句柄。

手写 CUDA 中，同一思路常体现为在两个 `__shared__` 数组间用 `^ 1` 下标或布尔相位变量切换。鳄霸在语言层面显式表达这一意图。

若采用三缓冲，可用 `rotate(f0, f1, f2)` 替代两次 `swap`——在一次操作中循环三个句柄。

## `auto` 返回类型

核函数签名使用 `__co__ auto matmul(...)`：`auto` 返回类型令鳄霸根据 `return output` 推断结果类型，使声明与形状表达式一致，并避免重复字面维度。

## Event：角色之间的同步

借助 `swap` 的双缓冲适用于**同一组**线程交错执行加载与计算的情形——它们在同一个程序计数器上交织步骤。Warp 特化（第 5 章）将加载与计算分配给**不同**的 warpgroup，这些组需要另一种协调机制：**event**。

```choreo
shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
```

`shared event` 在共享作用域声明具名同步屏障。`wait event_name` 在 event 被信号化之前阻塞；`trigger event_name` 对其发信号。在 1P1C 矩阵乘中：

- `full[s]` 表示阶段 `s` 已被生产者填满——消费者可以读取。
- `empty[s]` 表示消费者已用完阶段 `s`——生产者可以覆盖。

以下为带 event 的完整 1P1C 核：

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

**环形索引。** `stage = iv_k % MATMUL_STAGES` 将无界的 K 时间线映射到固定数量的物理槽位——可视为双缓冲推广到 N 个缓冲区。当 `MATMUL_STAGES = 4` 时，生产者最多可比消费者超前运行 4 个 K 分块，直至必须等待某槽位被释放。

**生产者流程。** 对每个 `iv_k`：等待 `empty[stage]` 表明槽位空闲，将分块拷贝至该阶段在 `lhs_load_s` 与 `rhs_load_s` 中的区域，然后 `trigger full[stage]` 通知消费者数据已就绪。

**消费者流程。** 在 K 循环之前，对所有阶段执行 `trigger empty[s]` 以完成**引导（bootstrap）**——每个槽位在逻辑上初始为空，避免生产者在首次 `wait empty` 上死锁。随后对每个 `iv_k`：`wait full[stage]`，在该阶段的数据上运行 MMA，`mma.commit`，再 `trigger empty[stage]` 归还槽位。

**`mma.commit`。** 它标定单个 K 板条 MMA 序列之后的逻辑边界。Hopper 上的 WGMMA 会积极重叠操作数取数、发射与累加；`mma.commit` 是栅栏，表示「在共享缓冲区被复用之前，将该阶段的局部乘积累入 `mc`」。应将其视为「该阶段数学完成」与「发 empty 信号」之间不可或缺的衔接。

## 单阶段的信用流

1. 消费者预先 `trigger empty[stage]`（引导）。
2. 生产者通过 `wait empty[stage]`——槽位空闲。
3. 生产者填充该 K 分块的共享内存，然后 `trigger full[stage]`。
4. 消费者通过 `wait full[stage]`——数据就绪。
5. 消费者执行 MMA、commit，再 `trigger empty[stage]`——槽位再次空闲。
6. 当 `iv_k` 环绕（对 `MATMUL_STAGES` 取模）时，周期重复。

环形结构并非魔法——对 `full`/`empty` 的 `wait`/`trigger` 才使 `iv_k % MATMUL_STAGES` 安全可用。

## 暂存用 Shared 与 Local

第 2 章对线程私有拷贝使用 `=> local`。若流水线中的分块被多个线程读取，几乎总是指向 `=> shared`。DMA future 跟踪哪次异步传输拥有哪块共享缓冲区；`swap` 或环形索引负责理清簿记。

实用规则：若块内每个线程读取同一暂存分块的重叠区域，用 `=> shared`。若每个线程消费互不重叠、无跨线程复用的片段，`=> local` 使数据更靠近计算核心。

## 常见陷阱

- **死锁。** 若去掉初始的 `trigger empty[s]` 循环，生产者的首次 `wait empty` 将永远等待。
- **同步不足导致的复用。** 省略 `mma.commit`，或相对加载错误排序 `trigger empty`，可能导致读到陈旧数据。
- **迭代次数不一致。** 生产者与消费者都必须使用 `foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)]`；不对称循环会导致 event 泄漏或悬空。
- **阶段数过少。** 若消费者快于生产者，会在 `wait full` 上停顿；增加阶段数可提高超前运行深度（以共享内存为代价）。

当对流水线的修改破坏正确性时，在怀疑 MMA 布局之前先检查 **event 顺序**——特化相关的缺陷通常实为同步问题。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `shared event name[N]` | 在共享作用域声明 N 个具名同步 event |
| `wait event` | 阻塞直至 `event` 已被信号化 |
| `trigger event` | 对 `event` 发信号，唤醒等待者 |
| `dma.copy.async src => dst` | 非阻塞拷贝（立即返回） |
| `dma.any` | 占位 future（尚无传输在进行） |
| `swap(f0, f1)` | 交换两个 future 句柄，不拷贝数据 |
| `rotate(f0, f1, f2)` | 循环轮换三个 future 句柄 |
| `with tile_k in N { ... }` | 绑定跨度为 N 的分块轴的作用域 |
| `foreach tile_k(1:)` | 从索引 1 开始迭代 |
| `mma.commit` | WGMMA 流水线阶段之间的栅栏 |
| `__co__ auto fn(...)` | 由 `return` 语句推断返回类型 |

流水线现已安全：生产者与消费者通过 event 重叠，环形索引在共享缓冲区上循环，`mma.commit` 保持累加器一致。[下一章](ch07-advanced-movement.md)将借助硬件加速的 TMA、打乱（swizzle）的共享内存布局，以及用于不规则访问模式的 `view`/`from`，进一步深入数据搬运。
