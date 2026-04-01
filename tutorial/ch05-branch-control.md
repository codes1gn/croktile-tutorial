# Branch and Control: Warp Roles and Persistent Kernels

Up to now, every thread in a block runs the same code. They all load the same tiles, they all run the same MMA, they all store the same result. That uniformity is clean, but it has a cost: while tensor cores are busy multiplying, the memory system could be prefetching the next tile — except nobody is asking it to, because every thread is stuck in the `mma.row.row` instruction.

This chapter introduces two forms of control flow that break that uniformity on purpose. **Warp specialization** (`inthreads.async`) assigns different roles to different warpgroups — one loads data while another computes. **Conditional guards** (`if`) let you skip work when a tile index falls out of bounds, which is central to **persistent kernels** that run a fixed pool of blocks across a variable number of tiles.

## `inthreads.async`: Splitting Roles

`inthreads.async (condition)` says: "only the threads for which `condition` is true execute this block." It is not a runtime branch that every thread evaluates and some skip — it is a **structural split** that creates two (or more) straight-line programs, one per role, which can run concurrently on different hardware units.

The canonical pattern is the **1 producer + 1 consumer (1P1C)** matmul:

```choreo
parallel p1 by 2 : group-4 {

  inthreads.async (p1 == 0) {
    // producer: only warpgroup 0 runs this
    // issue DMA / TMA loads, fill shared memory
  }

  inthreads.async (p1 == 1) {
    // consumer: only warpgroup 1 runs this
    // run MMA on shared memory, accumulate results
  }
}
```

`parallel p1 by 2 : group-4` creates two warpgroups of 128 threads each. The first `inthreads.async` block runs only on warpgroup 0 (the producer); the second runs only on warpgroup 1 (the consumer). Because the two bodies are structurally independent, the hardware can schedule TMA loads on one warpgroup while WGMMA runs on the other — true overlap, not interleaving.

Compare this with Chapter 3's `parallel`, where every thread runs the same body. Here, `inthreads.async` **differentiates** the bodies based on the parallel index. Think of it as "each warpgroup has a different job description."

## A 1P1C Matmul Skeleton

Here is how `inthreads.async` fits into a Hopper matmul. The synchronization details (events, wait, trigger) are deliberately omitted — Chapter 6 will add them. Focus on the role split:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        // Producer: walk K, load tiles into shared
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;
        }
      }

      inthreads.async (p1 == 1) {
        // Consumer: walk K, MMA on loaded tiles
        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load lhs_load_s.chunkat(_, iv_warp);
            mb = mma.load rhs_load_s.chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

The producer never touches `mc`, `mma.load`, or `mma.row.row`. The consumer never issues `dma.copy` to fill shared. Each body is a clean, straight-line loop over K. The missing piece — how the consumer knows a tile is ready before reading it — is synchronization, which comes in Chapter 6.

## `if` Guards: Conditional Execution

Sometimes you need a plain old conditional. Croktile's `if` works like C:

```choreo
if (tile_id < total_tiles) {
  // only execute this body when the condition is true
}
```

This is a **runtime** check, not a structural role split like `inthreads.async`. Every thread evaluates the condition; threads where it is false skip the body.

Where do you need this? Primarily in **persistent kernels**, where a fixed number of blocks iterate over a variable number of tiles, and some blocks may have one extra iteration with no real work to do.

## Persistent Kernels

In Chapters 3–4, the grid size was proportional to the problem: one block per output tile. For a big matrix, that can mean hundreds of thousands of blocks. The GPU schedules them in **waves**, and the last wave often leaves many SMs idle — **tail underutilization**.

A **persistent kernel** flips this: you launch a **fixed** number of blocks (typically matching the SM count), and each block loops over multiple tiles:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  int total_tiles = cdiv(M, MATMUL_WARP_M) * cdiv(N, MATMUL_WARP_N);

  parallel block_id by NUM_SMS : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)] {
      tile_id = tile_iter # block_id;

      if (tile_id < total_tiles) {
        block_m = tile_id / cdiv(N, MATMUL_WARP_N);
        block_n = tile_id % cdiv(N, MATMUL_WARP_N);

        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k) => rhs_load_s;

          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            parallel p by 1 : group-4 {
              ma = mma.load lhs_load_s.chunkat(_, iv_warp);
              mb = mma.load rhs_load_s.chunkat(_, iv_warp);
              mma.row.row mc, ma, mb;
            }
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

Three new ideas work together here:

**Fixed launch.** `parallel block_id by NUM_SMS : block` creates exactly `NUM_SMS` blocks (for example, 114 on an H800 PCIe). The index `block_id` is not tied to a single output tile — it names which persistent worker you are.

**Tile striping.** `tile_id = tile_iter # block_id` composes the iteration count with the block index to produce a unique tile id. Block `b` processes tiles `b`, `b + NUM_SMS`, `b + 2 * NUM_SMS`, and so on — a stride-`NUM_SMS` walk through the linearized tile list. The `#` compose operator is the same one from Chapter 2, applied here to scheduling arithmetic instead of tensor indexing.

**Linear-to-2D mapping.** `block_m = tile_id / cdiv(N, MATMUL_WARP_N)` and `block_n = tile_id % cdiv(N, MATMUL_WARP_N)` recover the 2D tile position from the linear id, the same way you would convert a flat array index to row and column in C.

**The `if` guard.** Because `foreach {tile_iter}` runs `cdiv(total_tiles, NUM_SMS)` iterations and some blocks may have one more than needed, `tile_id` can exceed `total_tiles`. The `if` skips all TMA, MMA, and store work for those padding iterations. Without it, you would index out of bounds.

The inner body — K loop, MMA, store — is identical to the non-persistent version from Chapter 4. Only the **wrapper** changed: fixed parallel, tile loop, compose, and guard.

## `cdiv`: Ceiling Division

`cdiv(a, b)` computes \\(\lceil a / b \rceil\\) — the number of tiles when the dimension might not be evenly divisible. It appears everywhere: grid extents (`cdiv(M, MATMUL_WARP_M)`), loop bounds (`cdiv(K, MATMUL_TILE_K)`), and persistent iteration counts (`cdiv(total_tiles, NUM_SMS)`).

When the last tile is partial (fewer valid elements than the tile size), real kernels pair `cdiv` with predicate masks, zero-padding, or epilogue cleanup. The tutorial stays on the clean-divisibility case, but `cdiv` is how you write tile counts that do not assume perfect division.

## Choosing Between Data-Dependent and Persistent Grids

| Aspect | One block per tile | Persistent (`NUM_SMS` blocks) |
|--------|-------------------|-------------------------------|
| Grid size | Grows with problem | Fixed |
| Tail utilization | Last wave may leave SMs idle | All SMs stay busy |
| Extra constructs | Minimal | `total_tiles`, `tile_iter # block_id`, `if` |
| Complexity | Lower | Higher |

Neither changes correctness; both produce the same output modulo floating-point associativity. Persistent layout pays off when `total_tiles` greatly exceeds the SM count, which is common for large matrix problems.

## New Syntax

| Syntax | Meaning |
|--------|---------|
| `inthreads.async (condition)` | Only threads satisfying `condition` execute this block — structural role split |
| `if (expr) { ... }` | Runtime conditional — skip body when `expr` is false |
| `cdiv(a, b)` | Ceiling division |
| `tile_id = tile_iter # block_id` | Compose iteration index with block index for tile striping |
| `int total_tiles = expr` | Local integer variable in a Croktile function |

Warp specialization splits roles; `if` guards edge cases. But the producer and consumer in the 1P1C skeleton still run their K loops independently with no coordination — the consumer assumes the tile is ready. The [next chapter](ch06-synchronization.md) introduces events, swap, and pipeline patterns so producer and consumer can safely overlap in time.
