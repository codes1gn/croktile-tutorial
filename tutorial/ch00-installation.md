# Ch 0: Installing Croktile

Before writing your first kernel, you need the Croktile compiler on your machine. This chapter walks through system dependencies, building from source, and verifying the installation.

## System Requirements

You need a C++17-capable compiler and two parser-generator tools:

| Dependency | Minimum Version |
|---|---|
| GCC | 8.1+ (or Clang 6+) |
| Bison | 3.8+ |
| Flex | 2.6.4+ |
| CUDA Toolkit | 11.8+ (for GPU targets) |

Most Linux distributions ship Flex and an older Bison. If your system Bison is below 3.8, the setup step below will fetch a compatible version automatically.

## Build from Source

Clone the repository and run the automated setup:

```bash
git clone https://github.com/codes1gn/croktile.git
cd croktile
make setup
```

`make setup` downloads the required versions of Flex, Bison, FileCheck, and GoogleTest if they are not already present. Once setup finishes, build the compiler:

```bash
make
```

Then run the test suite to confirm everything works:

```bash
make test
```

If all tests pass, the `croktile` binary is ready in your build directory. Add it to your `PATH` or invoke it by its full path.

## Verify the Installation

Create a minimal `.co` file to confirm the compiler runs:

```choreo
__co__ s32 [4] identity(s32 [4] input) {
  return input;
}

int main() {
  crok::s32 data[4] = {1, 2, 3, 4};
  auto result = identity(crok::make_spanview<1>(&data[0], {4}));
  for (int i = 0; i < 4; ++i)
    if (data[i] != result[i]) { std::cerr << "FAIL\n"; return 1; }
  std::cout << "OK\n";
}
```

Compile and run:

```bash
croktile verify.co -o verify
./verify
```

You should see `OK`. If you do, the compiler, linker, and runtime are all working.

## Compiler Usage

The `croktile` command works like `gcc` or `clang`:

```bash
croktile program.co                     # compile and link → a.out
croktile program.co -o my_kernel        # specify output name
croktile -es -t cuda program.co -o out.cu  # emit CUDA source only
croktile -E program.co                  # preprocess only
```

Key flags:

| Flag | Effect |
|---|---|
| `-o <file>` | Set output filename |
| `-t <platform>` | Select target platform (e.g. `cuda`) |
| `-es` | Emit target source code without compiling it |
| `-E` | Preprocess only (expand macros, strip `#if 0` blocks) |
| `-c` | Compile without linking |
| `-S` | Emit assembly |
| `--help` | Show all options |
| `--help-hidden` | Show advanced/internal options |

## Development Utilities

The Makefile includes shortcuts for running the bundled test suites:

```bash
make help                              # list all available targets
make sample-test                       # run all sample operator tests
make sample-test-operator OPERATOR=add # test a specific operator
```

These are useful when you modify Croktile itself or want to verify a specific operator family.

With the compiler installed and verified, you are ready to write your first real Croktile program in [Chapter 1](ch01-hello-choreo.md).
