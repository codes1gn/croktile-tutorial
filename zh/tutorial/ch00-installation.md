# 第 0 章：安装鳄霸

在编写你的第一个内核之前，需要先将鳄霸编译器安装到你的机器上。本章将介绍系统依赖、从源码构建以及验证安装。

## 系统要求

你需要一个支持 C++17 的编译器和两个解析器生成工具：

| 依赖项 | 最低版本 |
|---|---|
| GCC | 8.1+（或 Clang 6+） |
| Bison | 3.8+ |
| Flex | 2.6.4+ |
| CUDA Toolkit | 11.8+（用于 GPU 目标） |

大多数 Linux 发行版自带 Flex 和较旧版本的 Bison。如果你系统的 Bison 低于 3.8，下面的安装步骤会自动获取兼容版本。

## 从源码构建

克隆仓库并运行自动化安装：

```bash
git clone https://github.com/codes1gn/croktile.git
cd croktile
make setup-core
```

`make setup-core` 会拉取 git 子模块，并在未安装的情况下下载所需版本的 Flex、Bison、FileCheck 和 GoogleTest。安装完成后，构建编译器：

```bash
make
```

然后运行测试套件以确认一切正常：

```bash
make test
```

如果所有测试通过，`croktile` 二进制文件已就绪，位于构建目录中。将其添加到 `PATH` 或使用完整路径调用。

## 验证安装

创建一个最简 `.co` 文件以确认编译器可以运行：

```choreo
__co__ s32 [4] identity(s32 [4] input) {
  s32 [input.span] output;
  parallel i by 4
    output.at(i) = input.at(i);
  return output;
}

int main() {
  auto input = choreo::make_spandata<choreo::s32>(4);
  input[0] = 1; input[1] = 2; input[2] = 3; input[3] = 4;
  auto result = identity(input.view());
  for (int i = 0; i < 4; ++i)
    if (input[i] != result[i]) { std::cerr << "FAIL\n"; return 1; }
  std::cout << "OK\n";
}
```

编译并运行：

```bash
croktile verify.co -o verify
./verify
```

你应该看到 `OK`。如果看到了，说明编译器、链接器和运行时都已正常工作。

## 编译器用法

`croktile` 命令的使用方式类似 `gcc` 或 `clang`：

```bash
croktile program.co                     # 编译并链接 → a.out
croktile program.co -o my_kernel        # 指定输出文件名
croktile -es -t cuda program.co -o out.cu  # 仅输出 CUDA 源码
croktile -E program.co                  # 仅预处理
```

主要标志：

| 标志 | 作用 |
|---|---|
| `-o <file>` | 设置输出文件名 |
| `-t <platform>` | 选择目标平台（如 `cuda`） |
| `-es` | 输出目标源码但不编译 |
| `-E` | 仅预处理（展开宏，剥离 `#if 0` 块） |
| `-c` | 仅编译不链接 |
| `-S` | 输出汇编 |
| `--help` | 显示所有选项 |
| `--help-hidden` | 显示高级/内部选项 |

## 开发工具

Makefile 中包含了运行自带测试套件的快捷方式：

```bash
make help                              # 列出所有可用目标
make sample-test                       # 运行所有样例算子测试
make sample-test-operator OPERATOR=add # 测试特定算子
```

当你修改鳄霸本身或想要验证特定算子族时，这些快捷方式非常有用。

编译器安装并验证完毕后，你已准备好在[第 1 章](ch01-hello-croktile.md)中编写你的第一个真正的鳄霸程序。
