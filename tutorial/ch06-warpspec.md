# Async Pipelining: inthreads, Events, and Warp Roles

Chapter 3’s double buffering interleaves producer and consumer work on the **same threads**: two named slots, a prologue, a steady-state loop with **`swap`**, and an epilogue. Warp specialization assigns those roles to **different warpgroups** so loads and math **truly overlap in time**, coordinated by **shared events** instead of swapping buffer aliases. The **1 producer + 1 consumer (1P1C)** kernel in **`matmul_f16_dyn_sm90_warpspec_1p1c.co`** is the canonical example.

You still need **TMA** and **swizzle** from Chapter 4 and **WGMMA** with **`: group-4`** from Chapter 5. This chapter connects them with **shared events**, **`inthreads.async`**, **async TMA completion** wired to events, and **`mma.commit`**. The tileflow below matches the benchmark header:

- **`MATMUL_WARP_M = 64`**, **`MATMUL_WARP_N = 128`** — warpgroup output tile (WGMMA footprint along M and N).
- **`MATMUL_TILE_K = 64`** — K depth of each staged slab in shared memory per pipeline stage.
- **`MATMUL_WARP_K = 16`** — K step per inner **`mma.row.row`** (hardware slice along K).
- **`MATMUL_SWIZ = 128`** — swizzle metadata for **TMA** and **`mma.load.swiz`** (here **`2 * MATMUL_TILE_K`**, as the benchmark asserts).
- **`MATMUL_STAGES = 4`** — ring-buffer slots in shared memory (producer fills, consumer drains).

Treat the numbers as tuning knobs you would sweep in a real project; the transferable idea is **events**, **stages**, and **split roles**.

The header in the same benchmark file defines the constants and assertions (including **`MATMUL_SWIZ == 2 * MATMUL_TILE_K`**) that this chapter assumes.

## From swap-based double buffering to split roles

In Chapter 3, **double buffering** meant computing on **`lf0` / `rf0`** while prefetching into **`lf1` / `rf1`**, then **swap**ping names. You never read a slot that was simultaneously being overwritten.

**Warp specialization** keeps that safety story — each **stage** is either **empty** (safe for the producer to write) or **full** (safe for the consumer to read) — but **assigns** those actions to **different warpgroups**. The producer never runs **`mma.row.row`**; the consumer never issues **`tma.copy`**. They coordinate only through **events** and the shared arrays they agree on.

On **NVIDIA Hopper**, a block can host **multiple warpgroups**; **WGMMA** already used **four warps** (**`: group-4`**) in Chapter 5. The 1P1C kernel launches **two** such groups (**`parallel p1 by 2 : group-4`**) and uses **`p1`** as a role index. Choreo expresses that with **`inthreads.async (condition)`**: only the warpgroup for which **`condition`** holds runs the block — here **`p1 == 0`** or **`p1 == 1`**. That is **role specialization** in the parallel structure, not a scattered runtime **`if`** around unrelated code. A plain **`if (p1 == 0)`** inside one body could diverge every instruction; **`inthreads.async (…)`** keeps **two straight-line loop nests**, one per role, which is the idiomatic spell when you read **`inthreads.async`** in repository **`.co`** files.

## Shared memory and the event arrays

Before roles split, the block declares staging for **all** pipeline slots:

```choreo
shared event full[MATMUL_STAGES], empty[MATMUL_STAGES];
shared f16 [MATMUL_STAGES * MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
shared f16 [MATMUL_STAGES * MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
```

**`shared event`** declares **named synchronization events** in shared scope. **`wait event_name`** blocks until another participant has **signaled** that event; **`trigger event_name`** wakes waiters. **`full[s]`** means stage **`s`** has been filled by TMA and is ready for WGMMA. **`empty[s]`** means the consumer has finished with stage **`s`** and the producer may overwrite it. The **full/empty** naming is conventional and matches many CPU producer–consumer queues.

**`lhs_load_s`** and **`rhs_load_s`** are **`MATMUL_STAGES`** copies of the same logical shape — a **ring** of **A** and **B** slabs. Physically, **`lhs_load_s`** uses leading dimension **`MATMUL_STAGES * MATMUL_WARP_M`** and **`rhs_load_s`** uses **`MATMUL_STAGES * MATMUL_WARP_N`**, so each **`stage`** owns a contiguous **`MATMUL_WARP_M × MATMUL_TILE_K`** or **`MATMUL_WARP_N × MATMUL_TILE_K`** slab. Index **`stage`**, from **`iv_k % MATMUL_STAGES`**, picks which slab row you address with **`.at(stage, 0)`**.

In the **`swap(lf0, lf1)`** world, two names alias two buffers and you exchange them each iteration; here **names stay fixed** and you **cycle `stage`**, while **`full` / `empty`** carry the “which side is current” handshake. The structural difference from Chapter 3 is **concurrency**: swap pipelining **interleaves** on one thread set; 1P1C **overlaps** TMA and WGMMA **across warpgroups** — your profiler can show both busy in the same block.

**`output_s`** is a single warpgroup-sized tile: the consumer accumulates the full K sweep into **`mc`**, **`mma.store`** once, then **`tma.copy`** to global. Only operand staging is multi-buffered.

## Full tileflow: 1P1C matmul

Below is the **`__co__ void matmul`** body from **`matmul_f16_dyn_sm90_warpspec_1p1c.co`**, trimmed to tileflow (no host harness).

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
          tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0);
          tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0);
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
            ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
          mma.commit;
          trigger empty[stage];
        }
        mma.store mc, output_s;
        tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

The outer **`parallel {block_m, block_n} … : block`** is unchanged: one CUDA block per **`MATMUL_WARP_M × MATMUL_WARP_N`** tile. **`parallel p1 by 2 : group-4`** creates two warpgroups with **`p1 ∈ {0, 1}`**; each **`inthreads.async`** body is a **different program** for a different warpgroup.

In the repo file, the comment block above **`matmul`** labels producer versus consumer. Trace one **`iv_k`** through both bodies (**`stage`**, **`wait`**, **`tma.copy.async`**, **`wait full`**, inner **`mma`**, **`mma.commit`**, **`trigger empty`**). Once one lap is clear, picture **`iv_k+1`** issuing on the producer while the consumer is still in the WGMMA sequence for an older **`stage`**: that overlap is the pipeline you built.

## Producer warpgroup

The **`p1 == 0`** region is the **producer**:

```choreo
inthreads.async (p1 == 0) {
  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    stage = iv_k % MATMUL_STAGES;
    wait empty[stage];
    tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0);
    tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> rhs.chunkat(block_n, iv_k) => rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0);
    trigger full[stage];
  }
}
```

**`stage = iv_k % MATMUL_STAGES`** is **ring indexing**: after **`MATMUL_STAGES`** steps along **`iv_k`**, you reuse the same shared slice, like double buffering generalized to N slots. **`wait empty[stage]`** means the consumer has released that slot; without it, TMA could overwrite data the consumer still reads.

The two **`tma.copy.async<full[stage]>.swiz<…>`** lines start **asynchronous TMA** from global into the **`stage`** slice. The **`<full[stage]>`** annotation ties transfer **completion** to the **`full[stage]`** event: Choreo’s contract is that hardware completion of this async transfer participates in signaling **`full`** (exact micro-model is target-specific).

**`trigger full[stage]`** appears **after** both copies are **issued**. Together with the **`<full[stage]>`** completion semantics, that yields a clear story for the consumer: **`wait full[stage]`** means the **A/B** slab at **`stage`** is consistent for this **`iv_k`**.

The producer never touches **`mc`**, **`mma.load`**, or **`mma.row.row`**; it keeps the ring filled ahead of the consumer, modulo **`MATMUL_STAGES`**.

## Consumer warpgroup

The **`p1 == 1`** region is the **consumer**:

```choreo
inthreads.async (p1 == 1) {
  mc = mma.fill.f16 0.0f;
  foreach {s} in [MATMUL_STAGES] {
    trigger empty[s];
  }
  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    stage = iv_k % MATMUL_STAGES;
    wait full[stage];
    foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
      ma = mma.load.swiz<MATMUL_SWIZ> lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
      mb = mma.load.swiz<MATMUL_SWIZ> rhs_load_s.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(stage, 0).chunkat(_, iv_warp);
      mma.row.row mc, ma, mb;
    }
    mma.commit;
    trigger empty[stage];
  }
  mma.store mc, output_s;
  tma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

Before the main K loop, **`foreach {s} in [MATMUL_STAGES] { trigger empty[s]; }`** **bootstraps** credits: every slot starts logically empty, but the producer blocks on **`wait empty[stage]`** until those events exist. Pre-triggering all **`empty[s]`** avoids deadlock on the first TMA wave — like initializing per-slot credits.

In the steady state, **`wait full[stage]`** waits until TMA and signaling mean **`stage`** holds a consistent **A/B** slab for this **`iv_k`**. The inner **`foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)]`** runs **`MATMUL_TILE_K / MATMUL_WARP_K`** WGMMA steps (here **64/16 = 4**): each pass uses **`chunkat(_, iv_warp)`** on both staged tensors, **`mma.load.swiz<MATMUL_SWIZ>`**, then **`mma.row.row mc, ma, mb`**, same Chapter 5 pattern but reading **`at(stage, 0)`**. All passes accumulate into the same **`mc`** for this **`iv_k`** before **`mma.commit`**.

**`mma.commit`** ends the **pipeline stage** from the accumulator’s perspective. WGMMA overlaps operand movement and math deeply; without this boundary, you risk **semantic overlap** between micro-operations the hardware still treats as in flight for the prior stage and the next round of **`wait full`** / **`mma.load`** assumptions about **shared** contents.

**`commit`** is the Choreo-level fence that folds this stage’s partial products into **`mc`** before you reuse the slot. **`trigger empty[stage]`** returns the slot to the producer for a future **`iv_k`** when the ring wraps.

After all K tiles, **`mma.store`** and **`tma.copy`** write the result tile to global, as in the non-specialized Hopper kernel, except the K loop lives entirely in the consumer.

If you change **`MATMUL_TILE_K`**, keep **`cdiv(MATMUL_TILE_K, MATMUL_WARP_K)`** integral and the swizzle rule (**`MATMUL_SWIZ == 2 * MATMUL_TILE_K`** on this FP16 path) so **TMA** and **`mma.load.swiz`** stay matched.

## Why four stages and one K-tile handshake

**`MATMUL_STAGES = 4`** is a **latency-hiding** knob: more stages let the producer **run ahead** while the consumer drains older slots, until shared memory or bookkeeping stops paying off. The **`%`** operator maps an unbounded logical K timeline onto a **fixed** number of physical slots; reuse of slot 0 is safe only because **`wait empty` / `trigger empty`** serialize reuse, not because of modulo alone. Fewer stages (e.g. **2**) shrink shared footprint and simplify control but give less slack to hide TMA latency; more stages increase run-ahead until capacity flattens. **`4`** is a typical hand-tuned compromise; an auto-tuner might search stage count next to tile K.

For a single **`iv_k`**, the credit flow looks like this:

1. The consumer has already signaled **`empty[stage]`** (bootstrap or previous lap), so the producer **`wait empty[stage]`** succeeds.
2. The producer launches async TMA into **`lhs_load_s.at(stage, …)`** and **`rhs_load_s.at(stage, …)`**, tied to **`full[stage]`**, and **`trigger full[stage]`** participates in waking the consumer.
3. The consumer passes **`wait full[stage]`**, runs the inner WGMMA sequence on that **`stage`**, **`mma.commit`**, then **`trigger empty[stage]`**.
4. When **`iv_k`** advances enough that **`iv_k % MATMUL_STAGES`** equals this **`stage`** again, the ring wraps and the cycle repeats.

**`empty`** credits are spent when the producer claims a slot and replenished when the consumer finishes; **`full`** means data is ready. The modulo index only picks **which physical slot** joins that exchange.

This mirrors **producer–consumer queues** on CPUs: credits guard reuse of a fixed pool of buffers. The ring is not magic — **`wait` / `trigger`** on **`full` / `empty`** are what make **`iv_k % MATMUL_STAGES`** safe.

## `mma.commit` and pipeline boundaries

WGMMA overlaps operand fetch, issue, and accumulation more aggressively than a one-**`mma`**-at-a-time picture suggests. **`mma.commit`** marks a **logical boundary** after the **`mma.row.row`** sequence for one staged K slab (one **`iv_k`** worth of operand data at **`stage`**).

Before **`trigger empty[stage]`**, you want two properties. **Accumulator coherence:** partial sums from this stage’s loads are folded into **`mc`** so the next **`wait full` / `mma.load`** round does not race in-flight WGMMA from the previous use of overlapping register state. **Shared reuse safety:** once **`empty`** fires, the consumer will not read **`lhs_load_s` / `rhs_load_s`** at **`stage`** again for the completed **`iv_k`**, so TMA may overwrite that slice.

Exact ISA mapping is target-dependent. Treat **`mma.commit`** as **mandatory glue** between “done with this stage’s math” and “signal **empty**.” Dropping it in experiments is a stress test, not a performance knob, until you validate on your chip and toolchain.

Chapter 3’s **DMA futures** and async **wait** on copy handles tie completion to **specific operations**. **Shared events** are **named barriers**: any cooperating participant may **`wait`** or **`trigger`**. Here **`full` / `empty`** describe **buffer lifecycle**, while **`tma.copy.async<full[stage]>`** still connects **hardware completion** to **`full[stage]`**. See **`documentation/events.md`** for more idioms.

## Single-role Hopper matmul vs 1P1C (same primitives, different schedule)

| Aspect | **`matmul_f16_dyn_sm90.co`** (Chapter 5 excerpt) | **`matmul_f16_dyn_sm90_warpspec_1p1c.co`** (this chapter) |
|--------|---------------------------------------------------|-----------------------------------------------------------|
| **Warpgroups in the block** | One **`parallel … : group-4`** body for load + MMA | **`parallel p1 by 2 : group-4`** — **two** warpgroups, **`p1`** selects **role** |
| **Who issues TMA** | Same warpgroup that **WGMMA**-computes | Only **`p1 == 0`** |
| **Who runs `mma.row.row`** | Same warpgroup after **sync** TMA | Only **`p1 == 1`** |
| **Operand buffering** | One **lhs** / **rhs** slab per **block** (per K iter in excerpt) | **`MATMUL_STAGES`** **ring** of slabs in **shared** |
| **Synchronization** | **Implicit** / **wait**-style pairing with copies (in full benchmarks) | Explicit **`shared event`** arrays **`full[]`**, **`empty[]`** |
| **Overlap model** | **Interleaved** steps on **one** warpgroup | **Concurrent** **producer** and **consumer** warpgroups |

Neither layout changes the **math**; both implement blocked GEMM with **TMA** + **WGMMA**. Warp specialization changes **throughput** by changing **who** can be busy in the same cycle.

The single-role kernel is simpler to read and sometimes enough when memory latency is not the limiter. The 1P1C layout pays for extra shared memory, event arrays, and duplicated loop structure with the chance to **saturate** both **TMA** and **WGMMA** engines inside one block. Neither choice relaxes **swizzle** or **shape** rules from Chapters 4–5.

## Pitfalls

- **Deadlock:** Removing the initial **`trigger empty[s]`** loop can leave the producer’s first **`wait empty`** waiting forever.
- **Undersynchronized reuse:** Dropping **`mma.commit`** or mis-ordering **`trigger empty`** relative to loads risks stale swizzled data or broken accumulator coherence — treat **`commit`** as part of the WGMMA stage **contract**.
- **Swizzle mismatch:** **`MATMUL_SWIZ`** must stay consistent with **`tma.copy…swiz`** and **`mma.load.swiz`** (Chapters 4–5). Warp specialization does not relax layout rules.
- **Role drift:** Both **`inthreads.async`** bodies must agree on **`stage`** indexing and **`.at(stage, 0)`**; one-side typos yield wrong answers or rare races.
- **Too few stages:** If TMA often completes after the consumer needs the next **`full`**, you stall; raising **`MATMUL_STAGES`** (when shared memory allows) increases producer run-ahead.
- **Mismatched trip counts:** Producer and consumer both use **`foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)]`**; asymmetric loops leak or orphan events.

When a pipeline edit breaks correctness, re-check **event order** before you suspect MMA layout — specialization bugs are usually **synchronization** bugs. The benchmark **`main`** prints timing, TFLOPS, and a host numerical check on **C**.

## Looking ahead

You have walked the **1P1C** pattern end to end: one warpgroup keeps **TMA** in flight while another **consumes** with **WGMMA**, coordinated by **`full` / `empty`** instead of Chapter 3’s **`swap`**. On the producer, the steady rhythm is **`wait empty` → async TMA → `trigger full`**; on the consumer, **`wait full` → inner WGMMA → `mma.commit` → `trigger empty`**. **`iv_k % MATMUL_STAGES`** maps a long K sweep onto a small shared ring without changing the math — only **who** runs **when**.

Relative to Chapters 3–5, the new vocabulary is **`inthreads.async (condition)`** for **warpgroup roles**, **`shared event`** arrays, **`wait` / `trigger`**, **`tma.copy.async<event>.swiz<N>`** (completion tied to **`event`**), and **`mma.commit`** as the fence between WGMMA pipeline stages. **`grep`** **`inthreads.async`** in the benchmark tree when you want more working examples of the same idea.

For source and harness, open **`choreo/benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co`**. The swap baseline lives in **`croktile-tutorial/tutorial/ch03-pipeline.md`**. For event semantics beyond this matmul, see **`croktile-tutorial/documentation/events.md`**.

Chapter 7 (persistent kernels) and Chapter 8 (multi-warpgroup scaling) build on the same tileflow: you already know how to **fill**, **compute**, and **store**; warp specialization adds **which warpgroup** owns each phase and how credits move around the ring.

If you want hands-on checks, compare wall-clock and TFLOPS against **`matmul_f16_dyn_sm90.co`** on the same **M, N, K** after you align timing flags — gains track problem size, memory bandwidth, and stage count, not a universal speedup.

In a scratch copy of the warpspec file, try **`MATMUL_STAGES = 2`** and watch occupancy and stalls in your tooling. In a disposable branch, delete the **`trigger empty`** prologue once to see the deadlock, then restore it.

You may enable **`CHOREO_SKIP_VERIFY`** for faster iteration while editing, but run a full host check on **C** before you trust TFLOPS — a green verify is your first signal that **`full` / `empty`** ordering still matches the ring index math.

Underneath, Choreo is still **orchestration**: which warpgroup **waits** on which **event**, which async copy fills which shared slice, and where **`mma.commit`** closes a WGMMA stage.

**TMA** and **WGMMA** stay the Chapter 4–5 primitives; **roles** and **events** are what let memory traffic and math **overlap across warpgroups** instead of **interleaving** on one program counter. When **`stage`**, **`full`**, and **`empty`** feel automatic, you are in good shape for Chapters 7 and 8.
