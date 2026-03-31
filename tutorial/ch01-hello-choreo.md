# Hello Choreo: From Zero to Running Kernel

In this chapter, you will write and run your very first Choreo program — a parallel element-wise addition of two 3D arrays. Along the way, you will learn how a Choreo program is structured and meet the core concepts you will build on throughout this tutorial.

## What Choreo Does, in One Sentence

Choreo is a C++ embedded DSL that lets you describe *how data moves* between memory levels on heterogeneous hardware (think: host RAM, GPU shared memory, local registers), while keeping your actual compute logic in plain C++ or CUDA. The Choreo compiler takes your description and transpiles it into efficient, target-specific code — so you focus on *what* to move and *where*, not on the low-level plumbing.

## The Three Parts of a Choreo Program

Every Choreo program has three parts. If you have written CUDA before, two of them will look familiar:

1. **Host Program** — standard C++ running on the CPU. It prepares data, calls the Choreo function, and checks results.
2. **Device Program** — the compute kernel running on the GPU (or other accelerator). This is where the actual arithmetic happens.
3. **Tileflow Program** — the Choreo-specific part, marked with `__co__`. This is the *orchestrator*: it describes how data tiles are carved from inputs, moved to the device, handed to the compute kernel, and moved back.

The key insight is that in traditional CUDA, you write one monolithic kernel that handles both data movement and computation. In Choreo, these concerns are separated. The tileflow program handles data movement; the device program handles computation. This separation makes the data orchestration explicit, composable, and much easier to optimize.

## A Complete Example: Element-Wise Addition

Look at the full program. It adds two `[6, 17, 128]` arrays of 32-bit integers element by element:

```choreo
// Device Program
__device__ void kernel(int * a, int * b, int * c, int n) {
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// Tileflow Program
__co__ s32 [6, 17, 128] ele_add(s32 [6, 17, 128] lhs, s32 [6, 17, 128] rhs) {
  s32 [lhs.span] output;

  parallel p by 6 {
    with index in [17, 4] {
      foreach index {
        lhs_load = dma.copy lhs.chunkat(p, index) => local;
        rhs_load = dma.copy rhs.chunkat(p, index) => local;

        local s32 [lhs_load.span] l1_out;

        call kernel(lhs_load.data, rhs_load.data, l1_out, |lhs_load.span|);

        dma.copy l1_out => output.chunkat(p, index);
      }
    }
  }
  return output;
}

// Host Program
int main() {
  choreo::s32 a[6][17][128] = {0};
  choreo::s32 b[6][17][128] = {0};

  std::fill_n(&a[0][0][0], sizeof(a) / sizeof(a[0][0][0]), 1);
  std::fill_n(&b[0][0][0], sizeof(b) / sizeof(b[0][0][0]), 2);

  auto res = ele_add(choreo::make_spanview<3>(&a[0][0][0], {6, 17, 128}),
                     choreo::make_spanview<3>(&b[0][0][0], {6, 17, 128}));

  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j)
      for (size_t k = 0; k < res.shape()[2]; ++k)
        if (a[i][j][k] + b[i][j][k] != res[i][j][k]) {
          std::cerr << "result does not match.\n";
          abort();
        }

  std::cout << "Test Passed\n" << std::endl;
}
```

Save this as `ele_add.co` and compile it:

```bash
choreo ele_add.co -o ele_add
./ele_add
```

You should see `Test Passed`. Now walk through each part.

## The Host Program: Preparing and Verifying Data

The `main` function is plain C++ — nothing Choreo-specific except two API calls:

```choreo
choreo::s32 a[6][17][128] = {0};
choreo::s32 b[6][17][128] = {0};

std::fill_n(&a[0][0][0], sizeof(a) / sizeof(a[0][0][0]), 1);
std::fill_n(&b[0][0][0], sizeof(b) / sizeof(b[0][0][0]), 2);

auto res = ele_add(choreo::make_spanview<3>(&a[0][0][0], {6, 17, 128}),
                   choreo::make_spanview<3>(&b[0][0][0], {6, 17, 128}));
```

`choreo::make_spanview<3>` wraps a raw pointer with shape information, producing a `spanned_view` — Choreo's way of saying "here is a pointer to data, and it has this multi-dimensional shape." The template parameter `3` is the number of dimensions (rank). The shape `{6, 17, 128}` means 6 slices of 17 rows of 128 elements, with the most-significant dimension first (row-major order, like C arrays).

The return value `res` is a `choreo::spanned_data`, which *owns* its buffer (unlike `spanned_view`, which just points at existing memory). You can index into it with `res[i][j][k]` and query its shape with `res.shape()`.

## The Device Program: Just a Kernel

```choreo
__device__ void kernel(int * a, int * b, int * c, int n) {
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}
```

This is a standard CUDA device function. It takes three pointers and a length, and adds the first two arrays into the third. Nothing about data movement — the tileflow program handles that.

In Choreo programs, the device function operates on whatever data chunk the tileflow program has placed into local memory. It does not need to know the global shape, the tiling strategy, or the memory level it is reading from. This is the core of Choreo's separation of concerns.

## The Tileflow Program: Line by Line

This is the heart of the program. Here is a step-by-step walkthrough.

**Function signature.**

```choreo
__co__ s32 [6, 17, 128] ele_add(s32 [6, 17, 128] lhs, s32 [6, 17, 128] rhs) {
```

The `__co__` prefix marks this as a Choreo function. Unlike regular C++ functions, Choreo functions carry shape information in their signature: both inputs and the return value have the shape `[6, 17, 128]` with element type `s32` (signed 32-bit integer).

**Output buffer.**

```choreo
s32 [lhs.span] output;
```

This declares the output buffer. The expression `lhs.span` copies the shape from `lhs`, so `output` automatically has shape `[6, 17, 128]`. This is a convenient pattern — if the input shape changes, the output follows.

**Parallel execution.**

```choreo
parallel p by 6 {
  ...
}
```

`parallel p by 6` launches 6 parallel threads of execution. The variable `p` ranges from 0 to 5 and identifies each thread. If you are coming from CUDA, think of `p` as a block index — the code inside runs concurrently across 6 instances.

**Tiling with `with` and `foreach`.**

```choreo
with index in [17, 4] {
  foreach index {
    ...
  }
}
```

The `with ... in` block binds the symbol `index` to two *tiling factors*: 17 and 4. Then `foreach index` iterates over all combinations — conceptually equivalent to:

```cpp
for (int x = 0; x < 17; x++)
  for (int y = 0; y < 4; y++) { ... }
```

So each parallel thread runs 17 × 4 = 68 iterations.

**DMA: Moving Data.**

```choreo
lhs_load = dma.copy lhs.chunkat(p, index) => local;
```

This is the DMA statement — the operation that actually moves data. Let's unpack it:

- `dma.copy` initiates a direct data transfer.
- `lhs.chunkat(p, index)` is the *source*. The `chunkat` expression tiles `lhs` according to the parallel index `p` and loop index `index`. Since `lhs` has shape `[6, 17, 128]` and the tiling factors are `6`, `17`, and `4`, each chunk has size `1 × 1 × 32` (that is, `6/6`, `17/17`, `128/4`).
- `=> local` is the *destination* — a buffer in the device's local memory, automatically allocated by the compiler.
- `lhs_load` is the *future* of this DMA operation. It carries information about where the data landed, and you use it to access the data afterward.

The same pattern is used for `rhs`. After both DMA copies complete, the data chunk is available in local memory.

**Calling the device kernel.**

```choreo
local s32 [lhs_load.span] l1_out;
call kernel(lhs_load.data, rhs_load.data, l1_out, |lhs_load.span|);
```

`local s32 [lhs_load.span] l1_out` allocates a local buffer for the output, with the same shape as the loaded chunk.

The `call` statement invokes the device function `kernel`. Notice the accessors:
- `lhs_load.data` extracts the raw pointer from the DMA future.
- `|lhs_load.span|` computes the total number of elements in the chunk (the "span volume").

**Storing results back.**

```choreo
dma.copy l1_out => output.chunkat(p, index);
```

This reverses the direction: the result in local memory is copied back into the corresponding chunk of `output`.

## Build and Compile

Choreo files use the `.co` extension. The compiler works like `gcc` or `clang`:

```bash
choreo ele_add.co                          # produces a.out
choreo ele_add.co -o ele_add               # specify output name
choreo -es -t cuda ele_add.co -o out.cu    # emit CUDA source only
```

The `-es` flag stops after transpilation, letting you inspect the generated CUDA code. The `-t` flag selects the target platform.

## What You Have Learned

In this chapter, you have seen the three-part structure of a Choreo program (host, device, and tileflow); the `__co__` prefix for Choreo functions, with typed, shaped inputs and outputs; `parallel ... by` for launching parallel execution; and `with ... in` and `foreach` for tiling loops. You have also seen how `dma.copy ... => ...` moves data between memory levels, how `chunkat` carves data into tiles based on parallel and loop indices, how `call` invokes device functions on local data, and how `make_spanview` connects host arrays to Choreo's type system.

In the [next chapter](ch02-data-movement.md), we move from element-wise operations to matrix multiplication, and see how these same concepts apply to a more realistic kernel.
