# C++路线-llama.cpp的取舍

> **一句话**：不是"更快"，是不需要那一整套运行时。
> **前置**：先读 `01-用什么语言写推理引擎-12个真实引擎的证据.md` 建立"12 个引擎里 10 个 Python、只有 llama.cpp 和 Mooncake 是 C++"这个事实基础；再读 `02-自回归解码为什么是访存瓶颈.md`，本篇第 2.5 节会用它的结论拆穿一个流传最广的误解。

## 0. 这一篇要解决什么问题

前两篇（`01-用什么语言写推理引擎-12个真实引擎的证据.md`、`02-Python为什么还没被淘汰.md`）已经把"为什么 12 个引擎里 10 个是 Python"讲透了：热路径在 GPU 上，Python 只负责发指令，只要发指令的速度跟得上 GPU 消费指令的速度，Python 就是免费的。这个判据成立的前提是——**这个引擎运行在一台有 GPU、有 Python、有 torch 的机器上，服务的是很多并发用户**。

llama.cpp 不满足这个前提。它是 12 个引擎里唯二主语言是 C++ 的（另一个是 Mooncake，但 Mooncake 根本不是推理引擎，是 KV 缓存的传输与存储层，`00-总览与学习路径.md:51-52`），把它单独拎出来写一篇，不是因为"C++ 阵营"有多大，而是因为它是这份语言构成表里**唯一一个不是在优化"同一个问题"的引擎**。

本篇的立论要放在最前面：**vLLM/SGLang 解的是"数据中心里一堆 GPU 服务一堆并发用户"；llama.cpp 解的是"一台没有 GPU 的笔记本、一部手机、一块树莓派上把模型跑起来"。两者的约束不同，所以最优语言不同。** 这不是"同一个问题的两个不同答案"，是两个不同问题各自的最优解——把 llama.cpp 的选择拿去和 vLLM 比"谁的语言选型更对"，问的从一开始就不是同一个问题。

本篇要诚实地回答四件事，不站队：

1. C++ 在这条路上**换来了什么**——四样具体的能力，每样都要在源码里找到证据（`## 2`）；
2. **付出了什么代价**——这是本篇最有价值的部分，代价要写得和收益一样具体（`## 5`）；
3. 拆穿一个最常见的误解：**"C++ 比 Python 快，所以引擎该用 C++"是错的推理**——llama.cpp 选 C++ 不是因为它让矩阵乘更快，而是因为它不需要那一整套运行时（`## 2.5`）；
4. 给出一张"什么时候该走这条路、什么时候不该"的判据（`## 5`）。

## 1. 先看现象（可跑的代码/可算的数）

### 1.1 语言构成表里真正的"C++阵营"只有一家

`_PLAN.md` §3.2 的统计（`_PLAN.md:123-124`）：

| 引擎 | 总行 | 主语言 | 占比 | 前四语言 |
|---|---:|---|---:|---|
| llama.cpp | 992,377 | **C++** | 44% | C++ 44%, C/C++ header 22%, C 7%, Python 7% |
| Mooncake | 548,895 | **C++** | 54% | C++ 54%, C/C++ header 21%, Python 11%, Markdown 6% |

两家主语言都是 C++，但只有 llama.cpp 是推理引擎——它接收请求、加载模型、执行前向、吐出 token。Mooncake 不做这四件事里的任何一件，它做的是把 KV 缓存在多机之间搬运和落盘（`00-总览与学习路径.md:41` 标注"它不是引擎，是 KV 传输层"），是 vLLM/SGLang 这类引擎在做分布式 KV 复用时可能依赖的基础设施组件，不是一个独立对外提供推理服务的引擎。所以准确的说法不是"12 个引擎里 2 个选了 C++"，而是**"12 个真正的推理引擎里，11 个选 Python，1 个选 C++"**——llama.cpp 是彻底的少数派，这正是它值得单独拆开讲的原因：少数派要么是错的，要么是在解一个不一样的问题。本篇的立场是后者。

### 1.2 分发方式已经在暴露答案

打开 llama.cpp 的 README 看它建议用户怎么用它，比看任何一行源码都更快看出它的设计目标（`llama.cpp:README.md:24-26`）：

```
- Visit https://llama.app and follow the instructions
- Run with Docker - see our Docker documentation
- Download pre-built binaries from the releases page
- Build from source by cloning this repository
```

以及它给出的最短上手路径（`llama.cpp:README.md:32-34`）：

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF
```

对比一下装 vLLM 要做的事：`pip install vllm`，需要一个能装 CUDA 版 torch 的 Python 环境，需要目标机器上有能被 CUDA 驱动到的 GPU。llama.cpp 给出的第一条路径是"下载一个预编译好的可执行文件"——不需要 Python 解释器，不需要 pip 依赖树，不需要 CUDA。这个分发方式上的差异，就是全篇要讲的"换来了什么"的第一个可见信号，`## 2` 会把它拆到源码级别。

### 1.3 三个后面要核的数字，先摆出来

`## 3` 会带你自己在源码里数出这三个数字，这里先预告，作为 `## 2` 论证的引子：

- **ggml 的后端目录数**：`ggml/src/` 下有 **18 个** `ggml-*` 后端目录（CPU、CUDA、Metal、Vulkan、HIP、SYCL、CANN……），每一个都是独立实现——这是"可移植性"这项收益的代价面，`## 2.4`、`## 5` 都会用到；
- **架构分支数**：`src/llama-arch.h` 里 `enum llm_arch` 有 **148 个**架构条目（`llama.cpp:src/llama-arch.h:13`），每一个都要在 `src/llama-model.cpp` 里至少被 `switch (arch)` 命中一次；
- **新模型支持要走的两道工序**：`conversion/` 目录下有 **87 个**按模型家族拆开的 Python 转换脚本，`src/models/` 目录下有 **151 个**按模型家族拆开的 C++ 图构建文件——这两个数字是 `## 5` "新模型支持要手写"这条代价的直接证据。

这三个数字不是从哪篇博客抄来的，是我自己在 `../VLLM-SGlang-研究-推进/_src/llama.cpp/` 这份 clone 上用 `find`/`grep`/`ls` 数出来的，`## 3` 给出原始命令，读者应该自己跑一遍复核，而不是相信这段文字。

## 2. 原理

C++ 在 llama.cpp 这条路上换来的东西，可以拆成四条，每条都要在源码里找到落地的证据，而不是停在"C++ 能做到"这种抽象描述上。

### 2.1 无运行时依赖

`## 1.2` 已经看到现象了：README 建议的第一条路径是下载一个预编译二进制。这件事能成立，根子在于 llama.cpp 从底层张量库（ggml）开始就没有引入任何需要在目标机器上单独安装的运行时——不需要 Python 解释器、不需要 torch、不需要 CUDA Toolkit（CPU-only 编译时）。编译产物是一个静态或动态链接过的可执行文件，`llama serve` 这一行命令背后不存在"先启动一个解释器，再 import 一堆包"这个步骤。

这条收益的代价面在 `ggml/CMakeLists.txt` 里看得很清楚：每一个硬件后端都是一个**编译期开关**，默认关闭（`llama.cpp:ggml/CMakeLists.txt:199`）：

```
option(GGML_CUDA  "ggml: use CUDA"  OFF)
```

`GGML_HIP`（`llama.cpp:ggml/CMakeLists.txt:215`）同样默认 `OFF`。这意味着"无依赖"不是一句口号，是一个真实的构建策略：你编译出来的那个二进制，只包含你显式打开的后端代码，CPU-only 编译出来的产物里连 CUDA 头文件都没链接进去。这是"单文件分发"能成立的直接原因——**依赖不是被隐藏了，是被编译期决策从源头上移除了**。

### 2.2 对内存布局的完全控制

Python 路线加载一个模型，做的事情是：反序列化 safetensors → 构造 torch Tensor → 拷贝到 GPU 显存，这中间有 Python 对象、张量元数据、可能的 dtype 转换。llama.cpp 选择了一条完全不同的路：**GGUF 文件格式 + mmap 加载 + 量化权重按位摆放**，三件事拼在一起，让"加载模型"这个动作退化成"把一个文件映射进地址空间"。

GGUF 是一个自定义的二进制容器格式，核心结构是 `gguf_context`（`llama.cpp:ggml/src/gguf.cpp:217`）：

```cpp
struct gguf_context {
    uint32_t version = GGUF_VERSION;
    std::vector<struct gguf_kv> kv;
    std::vector<struct gguf_tensor_info> info;
    ...
};
```

它把模型的超参数（`kv`，key-value 元数据）和每个张量的 offset/shape/dtype（`info`）都编码在文件头里，张量数据本身按 `ALIGNMENT` 对齐紧跟其后（`gguf_tensor_info::offset` 的注释写明"offset from start of `data`, must be a multiple of `ALIGNMENT`"）。这不是随便选的设计——**对齐布局是为了让 mmap 之后，张量数据可以直接被当成一段连续内存使用，不需要额外拷贝或反序列化**。

`llama_mmap` 就是把这件事落地的地方（`llama.cpp:src/llama-mmap.cpp:457`）：

```cpp
addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
```

以及它的构造函数入口（`llama.cpp:src/llama-mmap.cpp:620`）：

```cpp
llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa) : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}
```

`mmap` 把整个模型文件映射进虚拟地址空间，由操作系统按需换页——权重"加载"这个动作在用户态代码里几乎不做任何工作，真正的磁盘 I/O 被推迟到第一次访问某段内存的那一刻，由内核处理。这跟 Python 路线"先整个反序列化进内存/显存"是完全不同的哲学：**内存布局的每一个字节都是被设计过的，加载不是一个需要写代码去做的步骤，而是文件格式设计的自然结果。**

量化权重的字节布局同样是手工设计的。以最常见的 4-bit 量化 `block_q4_0` 为例（`llama.cpp:ggml/src/ggml-common.h:194-199`）：

```cpp
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");
```

32 个权重值打包成一个 block：一个 fp16 的缩放因子 `d`，加上 16 字节的 4-bit 量化值（每字节存两个 nibble）。这个 struct 的内存布局就是文件里的字节布局，`static_assert` 强制编译期保证两者一致——**C++ 在这里换来的是字节级别的可控性：量化格式怎么设计、怎么摆，完全由这份代码决定，不需要绕过任何张量库的抽象层。**

### 2.3 能下沉到 CPU SIMD

llama.cpp 的主场是没有 GPU 的机器，这意味着**算力全部要靠 CPU 出**，而 CPU 上能不能喂饱算力，很大程度上取决于有没有用到向量指令集。`ggml/src/ggml-cpu/` 下按硬件架构拆出了独立目录：`arch/x86/`、`arch/arm/`，每个目录里都有专门为该架构手写的量化/矩阵乘代码路径。

x86 侧的 AVX512 分支，在处理 `block_q4_0` 打包时有专门代码（`llama.cpp:ggml/src/ggml-cpu/arch/x86/quants.c:139`）：

```c
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);
    bytes = _mm256_or_si256(bytes, bytes_srli_4);
    return _mm256_cvtepi16_epi8(bytes);
```

ARM 侧对应的是 NEON 路径（`llama.cpp:ggml/src/ggml-cpu/arch/arm/quants.c:26`）：

```c
#if defined(__ARM_NEON)
```

这类 `#if defined(__ARM_NEON)` / `#if __AVX512F__` 的条件编译在 `ggml-cpu/arch/x86/quants.c`、`repack.cpp`、`ggml-cpu.c`、`ops.cpp`、`vec.cpp` 等多个文件里反复出现——每一种量化格式在每一种指令集下都有单独手写的内层循环。**这是 C++（以及它能直接嵌入的架构相关 intrinsics/汇编）换来的第二样东西：能摸到硬件最底层的向量化能力，不用等一个上层框架（比如 PyTorch 的 CPU 后端）把这条路径实现好。** Python 生态不是不能触达 SIMD（numpy 背后的 BLAS 本身就是手写 SIMD 代码），但那要求先装上一整套运行时；llama.cpp 把这条路径直接编进了那个无依赖的二进制里。

### 2.4 可移植性：同一份计算图，多种后端

前三条容易让人以为 C++ 换来的是"针对某一种硬件抠到极致"，但 llama.cpp 真正的架构设计恰恰相反：**它先定义了一层与硬件无关的后端接口，再让 18 种硬件各自实现这层接口**，计算图本身只写一次。

接口定义是 `ggml_backend_i`（`llama.cpp:ggml/src/ggml-backend-impl.h:105`），核心是一组函数指针，其中最关键的是 `graph_compute`：

```cpp
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void (*free)(ggml_backend_t backend);
    ...
    enum ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    ...
};
```

一个具体后端（CPU/CUDA/Metal/Vulkan/……）就是给这组函数指针填上自己的实现，打包进 `ggml_backend` 结构体（`llama.cpp:ggml/src/ggml-backend-impl.h:142`）：

```cpp
struct ggml_backend {
    ggml_guid_t guid;
    struct ggml_backend_i iface;
    ggml_backend_dev_t device;
    void * context;
};
```

而调用方——模型的前向计算图构建代码——从头到尾不知道自己最终会被哪个后端执行。真正做"同一张图，可能横跨多个异构后端一起算"这件事的是调度器 `ggml_backend_sched`（`llama.cpp:ggml/src/ggml-backend.cpp:1792` 的 `ggml_backend_sched_new`），它的典型用法在头文件的注释示例里写得很直接（`llama.cpp:ggml/include/ggml-backend.h:278`）：

```cpp
sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);
```

一份计算图，同时喂给"GPU1、GPU2、CPU"这三个后端，调度器负责把图里的节点分派到合适的后端上执行、在需要的地方插入跨设备拷贝。这就是为什么同一份 llama.cpp 源码能不改一行模型代码，就跑在 `## 1.3` 数出来的 **18 个 `ggml-*` 后端目录**对应的硬件上——每一个都是 `ggml_backend_i` 接口的一份独立实现，但它们背后共享同一份 `ggml_cgraph`。README 自己维护了一张"Supported backends"表，把每个后端目录对应的目标设备写得很清楚（`llama.cpp:README.md:68-88`）：

| Backend | Target devices |
| --- | --- |
| CUDA | Nvidia GPU |
| HIP | AMD GPU |
| Metal | Apple Silicon |
| Vulkan | GPU（跨厂商） |
| CANN | Ascend NPU（华为昇腾，`llama.cpp:docs/backend/CANN.md:17`） |
| MUSA | Moore Threads GPU（摩尔线程，`llama.cpp:README.md:79`） |
| Hexagon [In Progress] | Snapdragon（高通骁龙移动芯片，`llama.cpp:docs/backend/snapdragon/README.md:1`） |
| OpenCL | Adreno GPU（骁龙内置 GPU） |
| SYCL | Intel GPU（`llama.cpp:docs/backend/SYCL.md:22` 讲的是 Intel oneAPI） |
| IBM zDNN | IBM Z & LinuxONE 大型机 |
| ZenDNN | AMD CPU |
| VirtGPU | 虚拟机里的 GPU 直通（`llama.cpp:docs/backend/VirtGPU.md:1-4`） |
| WebGPU | 浏览器 |
| BLAS/BLIS | 所有平台的 CPU |
| RPC | 所有平台（通过网络转发到远端后端） |

这张表比"能跑在 CPU/CUDA/Metal 上"这句笼统的话更能说明问题——**手机芯片（Snapdragon/Adreno）、大型机（IBM Z）、虚拟机直通（VirtGPU）、国产 GPU（Moore Threads）都在这张表里**，这正是 `## 0` 开篇那句话的具体所指："一台没有 GPU 的笔记本、一部手机、一块树莓派"不是修辞，是这张表里真实存在的目标设备。这也是为什么 `## 2.1` 那条"不依赖 PyTorch"的决策代价高但值得付——PyTorch 官方 wheel 覆盖不到 IBM Z 大型机或者某些嵌入式 NPU，llama.cpp 选择自己维护后端抽象，换来的是能触达这些 PyTorch 生态覆盖不到的角落。

**这是可移植性的真实来源：不是"C++ 天生可移植"，是这层后端抽象把"计算图长什么样"和"这台机器怎么执行它"彻底解耦了。**

### 2.5 题眼：这四条里没有一条叫"更快"

把 2.1～2.4 放在一起看会发现一件事：**无运行时依赖、内存布局控制、SIMD 下沉、后端可移植——没有一条是"C++ 让矩阵乘算得比 Python 快"。** 这不是疏漏，是因为这个理由本来就不成立，值得单独拆开讲清楚，因为它是关于"为什么用 C++"这个问题上最常见、也最误导人的一句话。

`02-自回归解码为什么是访存瓶颈.md` 已经证明了一件事：单条请求做 decode，从第一步开始就已经站在访存瓶颈的地板上（`01-地基/02-自回归解码为什么是访存瓶颈.md:60`）——每一步的耗时由"要从显存/内存搬多少字节的权重和 KV"决定，不是由"发起这次计算调用的语言是什么"决定。`02-Python为什么还没被淘汰.md` 从另一个方向证明了同一件事：GPU 的 kernel launch 是异步的，Python 只负责把指令塞进队列，只要塞得比 GPU 消费得快，Python 解释器的存在感就是零。

这条判据对 vLLM（Python + CUDA kernel）成立，对 llama.cpp（C++ + CUDA/Metal/Vulkan kernel）**同样成立**——当 llama.cpp 跑在 GPU 后端上时，真正做浮点运算的仍然是 GPU 上的 kernel，跟 Python 路线调用的 FlashAttention/cuBLAS 内核在算法层面没有本质差异，C++ 语言本身并没有让这些 kernel 算得更快。C++ 在这条链路上省下的，是 CPU 侧组织这次调用的胶白代码开销——而这部分开销在 GPU decode 场景下本来就不是瓶颈，省下它对最终吞吐的影响可以忽略。

真正用得上"C++ 更快"这个说法的场景，是 `## 2.3` 讲的那个——**没有 GPU、纯 CPU 推理**。这时候瓶颈变成了 CPU 内存带宽和 CPU 算力，`T_gpu` 这个概念本身就不存在了，`02-Python为什么还没被淘汰.md` 里"Python 被 GPU 计算掩盖"这套判据也随之失效——CPU 上跑 Python 解释器的调度开销、对象创建开销，是要跟"真正做量化矩阵乘的那几百微秒"顺序叠加的，没有异步执行模型帮忙盖住。但即便在这个场景下，真正起作用的也不是"C++ 这门语言比 Python 快"，而是"手写的 AVX512/NEON 内层循环比 numpy 在没有对应量化格式支持时能达到的效率更高"——**这件事完全可以用 C 或者 Rust 达成同样的效果，C++ 在这里不是必要条件，只是 llama.cpp 实际选择的语言。** 真正的必要条件是"能编译成一个不依赖任何运行时的二进制、能直接嵌入架构相关 intrinsics"——C、C++、Rust 都满足，llama.cpp 选了 C++，是工程选择，不是语言性能上的必然。

**所以 C++ 在 llama.cpp 里的价值，准确的表述是：不需要那一整套运行时。** 这跟"更快"是两件事——一台没有 GPU 的树莓派上，决定能不能跑起来的从来不是"每次矩阵乘快多少纳秒"，而是"这台机器上到底能不能装下 Python + torch 那一整套东西、装不下的话还有没有别的路"。llama.cpp 的存在，回答的是后一个问题。

### 2.6 收益-代价对照表：先记住这张表，`## 5` 会逐条展开

| 收益（`## 2.1`～`## 2.4`） | 对应的代价 | 详见 |
|---|---|---|
| 无运行时依赖：单文件分发，不需要 Python/torch/CUDA | 每一种硬件后端都要自己实现，`## 3` 数出来的 18 个 `ggml-*` 目录都是自己维护的 | `## 4.3`、`## 5.1` 决策二 |
| 内存布局完全控制：GGUF + mmap + 量化 block 手工摆放 | 文件格式和量化格式要自己设计、自己维护版本兼容；别的引擎读它时状态是"实验性" | `## 4.2`、`## 4.4`、`## 6` 误区四 |
| CPU SIMD 下沉：AVX2/AVX512/NEON 手写路径 | 只有在没有 GPU 的场景下才是决定性收益；GPU 场景下跟 Python 路线打平，见 `## 2.5` | `## 2.5`、`## 6` 误区一 |
| 后端可移植性：同一份计算图，18 种后端各自实现 `ggml_backend_i` | 新架构要同时手写 Python 转换器和 C++ 图构建，两道工序都不能省 | `## 3`、`## 4.1`、`## 5.1` 决策三 |

这张表就是本篇标题"取舍"两个字的具体内容——**每一行左边的收益，都能在右边找到对应的代价，没有一条是白拿的。** `## 3` 开始，把表里提到的每个数字自己核一遍。

## 3. 自己动手：去 `_src/llama.cpp/` 里自己核这些结论

`## 1.3`、`## 2` 里出现的每一个数字，都可以在 `../VLLM-SGlang-研究-推进/_src/llama.cpp/` 这份 clone 上自己跑出来，不要相信这篇文章写的数字，跑一遍最可靠。

**后端目录数**：

```bash
cd ../VLLM-SGlang-研究-推进/_src/llama.cpp
find ggml/src -maxdepth 1 -type d -name "ggml-*" | wc -l
```

我跑出来的结果是 `18`，具体目录：

```
ggml-blas  ggml-cann  ggml-cpu  ggml-cuda  ggml-et  ggml-hexagon  ggml-hip
ggml-metal ggml-musa  ggml-opencl ggml-openvino ggml-rpc ggml-sycl
ggml-virtgpu ggml-vulkan ggml-webgpu ggml-zdnn ggml-zendnn
```

**架构分支数**（`enum llm_arch` 里的条目数，用 `awk` 只统计 `enum llm_arch {` 到对应 `};` 之间的行，避免数到文件里其他地方重复出现的 `LLM_ARCH_` 字样）：

```bash
awk '/enum llm_arch \{/,/\};/' src/llama-arch.h | grep -c "LLM_ARCH_"
```

我跑出来的结果是 `148`。同一份架构，在 `src/llama-model.cpp` 里至少要被 `switch (arch)` 命中一次（工厂函数在 `llama.cpp:src/llama-model.cpp:42-43`），常常还要在第二个、第三个 `switch (arch)`（比如决定用哪种 KV 缓存实现的 `create_memory`，`llama.cpp:src/llama-model.cpp:2103-2107`）里再命中一次：

```bash
grep -c "case LLM_ARCH_" src/llama-model.cpp
```

我跑出来的结果是 `319`——远大于 148，说明平均每个架构在这个文件里要被 `case` 命中两次以上。

**新模型支持要走的两道工序**：

```bash
ls conversion/*.py | grep -v "__init__\|base.py" | wc -l    # Python 转换脚本，按模型家族拆分
ls src/models/*.cpp | wc -l                                 # C++ 图构建文件，按模型家族拆分
```

我跑出来的结果分别是 `87` 和 `151`。这两个数字会在 `## 4`、`## 5` 里反复用到——它们是"新模型支持要手写"这条代价最直接的证据：一个新架构要先在 `conversion/` 下有一个 Python 类把 HF checkpoint 转换成 GGUF 字节布局（比如 `llama.cpp:conversion/llama.py:17` 的 `@ModelBase.register(...)` 装饰器），再在 `src/models/` 下有一个 C++ 文件手写运行时真正执行的计算图（比如 `llama.cpp:src/models/llama.cpp:94` 的 `build_arch_graph`），两道工序都要走完，模型才算真正"支持"了。

**训练生态的现状**（用来核实"C++ 路线天然只做推理"这条说法，不是凭印象下结论）：

```bash
find . -iname "*finetune*" -not -path "*/.git/*"
cat examples/training/README.md | head -5
```

跑出来只有一个文件 `examples/training/finetune.cpp`（100 行），README 第一句话自己写的是（`llama.cpp:examples/training/README.md:4`）：

```
So far finetuning is technically functional (for FP32 models and limited
hardware setups) but the code is very much WIP.
```

——"目前可用但仅限 FP32、有限硬件配置，代码还很 WIP"，这是项目自己的定性，不是我的推断，`## 5` 会用这条证据支撑"训练生态在 Python"这条代价。

**并发模型的现状**（用来核实"llama.cpp 不支持批量并发"这句常见说法，`## 5.1` 决策四、`## 6` 误区二会用到）：

```bash
grep -n "Continuous batching" tools/server/README.md
grep -n "cont-batching" tools/server/README.md
grep -n "struct server_slot {" tools/server/server-context.cpp
grep -n "initializing, n_slots" tools/server/server-context.cpp
```

跑出来能看到 `README.md` 第 13 行把 "Continuous batching" 列为 server 的特性之一，第 176 行有 `-cb, --cont-batching` 选项说明"默认开启"；`server-context.cpp` 第 196 行是 `struct server_slot {` 的定义，第 1231 行是初始化时打印的 `"initializing, n_slots = %d, n_ctx_slot = %d, ..."` 日志——这条日志本身就说明了并发规模在初始化阶段就被 `n_parallel` 定死，不是运行时动态扩展的。这四条命令加起来，把"支持并发"和"并发模型跟 vLLM 不一样"这两件事都能自己核实一遍，不用停留在"听说 llama.cpp 只能单用户"这种道听途说上。

## 4. 同一件事，Python 路线怎么做

`## 2`、`## 3` 看的是 llama.cpp 一家的源码，容易产生一种错觉：GGUF、量化 block、手写图构建，看起来是"应该这么做的标准做法"。把 Python 路线（以 vLLM 为例，源码在 `../VLLM-SGlang-研究-推进/_src/vllm/`）摆在旁边做同一件事，才能看清楚 llama.cpp 每一分代价换来的到底是什么，代价才有意义——单独讲一条路线的成本没有参照系。

### 4.1 加新模型：注册表指向已有实现，vs 两道手写工序

vLLM 支持一个新架构，核心动作是往一个字典里加一行。`_TEXT_GENERATION_MODELS`（`vllm:vllm/model_executor/models/registry.py:72`）：

```python
_TEXT_GENERATION_MODELS = {
    "AfmoeForCausalLM": ("afmoe", "AfmoeForCausalLM"),
    "ApertusForCausalLM": ("apertus", "ApertusForCausalLM"),
    "ArceeForCausalLM": ("arcee", "ArceeForCausalLM"),
    ...
```

这是一个"HF 架构字符串 → (模块名, 类名)"的映射表，真正加载时才按需 `import`（惰性导入，避免把所有模型的依赖一次性拉进内存）。表里类似这样的条目一共有 394 处（自己数的：`grep -c '": (' vllm/model_executor/models/registry.py`）。这个数字比 llama.cpp 的 148 个架构分支更大，但背后的工作量完全不是一回事——**vLLM 的注册表指向的是已经存在的实现**：模型的超参数解析直接复用 HF `transformers` 的 `PretrainedConfig`（`vllm:vllm/transformers_utils/config.py:18` 的 `from transformers import GenerationConfig, PretrainedConfig`），很多新架构在 HF 官方发布权重的同时就已经有了参考实现，vLLM 这边要写的往往是一层把 HF 的 `nn.Module` 前向适配成 vLLM 内部张量并行/KV 缓存接口的薄封装，不需要从零手写一遍矩阵乘和注意力的计算图。

对照 `## 3` 数出来的两个数字：llama.cpp 支持一个新架构，需要 `conversion/` 下一个 Python 转换类（负责把 HF checkpoint 的权重重新摆放成 GGUF 的字节布局）**和** `src/models/` 下一个 C++ 图构建文件（负责运行时真正执行的计算图，因为运行时不能依赖 Python 在场）——两道工序都要走完，87 加 151 这两个文件数不是巧合，是"每个架构都要在两种语言里各实现一遍"的直接产物。llama.cpp 的 `conversion/` 里也有子类化复用的手法（比如 `llama.cpp:conversion/llama.py:365` 的 `ArceeModel(LlamaModel)`），这说明"复用相近架构"不是 Python 独有的技巧，但它只能省下**权重转换**那一步；`src/models/` 里的图构建文件，只有当新架构的计算图跟已有架构完全一致时才能省，这在实践里比"超参数格式相似"苛刻得多。

### 4.2 加载权重：直接读 safetensors，vs 自己设计并维护一套二进制格式

vLLM 加载权重直接用 `safetensors` 这个 HF 生态的事实标准格式（`vllm:vllm/model_executor/model_loader/weight_utils.py:26`）：

```python
from safetensors.torch import load, load_file, safe_open, save_file
```

`load_file` 拿到的就是可以直接送进 GPU 的张量，不需要一个专门的转换步骤、不需要引擎自己设计文件格式。`## 2.2` 已经讲清楚了 GGUF + 量化 block 换来的东西（mmap 直接可用的内存布局、字节级可控的量化格式），但反过来看，这也是一份要自己维护的负担：GGUF 的版本演进（`llama.cpp:ggml/src/gguf.cpp:506` 那行 `if (ok && ctx->version > GGUF_VERSION)` 的版本检查）、每种量化格式的 block 结构（`ggml-common.h` 里几十个 `typedef struct` 各自定义了自己的 `static_assert`），都是 llama.cpp 团队自己设计、自己保证向后兼容的东西。vLLM 不需要背这份负担，因为它选择直接吃 HF 生态已经定下来的标准格式。

### 4.3 支持新硬件：继承 PyTorch 已经写好的后端，vs 自己实现 18 个

这是两条路线代价差异最大的一处。vLLM/SGLang 的模型执行层构建在 PyTorch 之上，PyTorch 自己维护了 CUDA、ROCm（对应 AMD 的 HIP）、XPU 等设备后端的底层实现——vLLM 要支持一种新硬件，很多时候只需要 PyTorch 那边已经支持了这块设备，vLLM 这层写的是设备无关的 Python/Triton/CUDA 代码，不需要重新实现一遍"怎么在这块硬件上做矩阵乘"。

llama.cpp 从 `## 2.1` 那条决策的起点就拒绝了这条捷径——不依赖 PyTorch 这类运行时，意味着也不能"借用"PyTorch 已经写好的硬件后端。`## 2.4` 里的 18 个 `ggml-*` 目录，每一个都是 ggml 团队自己实现的：`ggml-cuda` 要自己封装 cuBLAS/写 CUDA kernel，`ggml-hip` 要自己适配 AMD 的 HIP API，`ggml-vulkan` 要自己写 Vulkan compute shader。**这不是团队选择"重复造轮子"，是"不依赖运行时"这条决策的必然代价——一旦放弃依赖 PyTorch 的设备抽象层，硬件适配的工作量就必须自己扛，没有生态可以分摊。**

vLLM 这边"继承"这件事在源码里也能看到落点——它自己的硬件平台抽象层直接建在 PyTorch 的设备体系之上，比如 `CudaPlatformBase`（`vllm:vllm/platforms/cuda.py:208`）这个类描述的是"CUDA 这一类设备在 vLLM 里怎么被识别、怎么选内存分配策略"，而不是"怎么在 CUDA 上从头实现一次矩阵乘"——后者已经是 PyTorch/cuBLAS/FlashAttention 解决好的问题。vLLM 的平台层要做的是**适配**，llama.cpp 的后端目录要做的是**实现**，这是两条路线在"支持新硬件"这件事上工作量差一个数量级的根本原因。

### 4.4 一个交叉验证：GGUF 在 Python 路线里的真实地位

`## 2.2` 讲 GGUF 是 llama.cpp 自己设计的格式，但它并非完全没有外溢——vLLM 也能读 GGUF 文件，不过看它自己文档的定性就知道这不是一等公民（`vllm:docs/features/quantization/gguf.md:4`）：

```
Please note that GGUF support in vLLM is highly experimental and
under-optimized at the moment, it might be incompatible with other features.
```

以及（`vllm:docs/features/quantization/gguf.md:7`）：

```
GGUF support has migrated to OOT vllm-gguf-plugin.
```

——"高度实验性"、"已经迁移到一个树外（out-of-tree）插件"。这条交叉验证反而印证了本节的论点：GGUF 确实已经流出了 llama.cpp 的边界，说明它不是不可跨越的技术壁垒；但它在 Python 路线里的位置是"需要单独装插件、状态标注为实验性"的外来客人，不是 safetensors 那种原生格式。**格式设计时的目标环境决定了它的自然栖息地——GGUF 为"无依赖二进制 + mmap 直接可用"设计，这正是 C++ 路线的核心约束，搬进一个本来就依赖 PyTorch 运行时的环境里，那份约束带来的收益大部分用不上，只剩下"多一种要维护的加载路径"这个成本。

## 5. 设计决策与代价（为什么这样 / 不这样会怎样 / 什么时候可以不这样）

### 5.1 六条核心决策

**决策一：用 GGUF + mmap 加载权重，而不是 safetensors/state_dict 反序列化**

- **为什么这样**：`## 2.2` 讲清楚了——mmap 让"加载"退化成"映射进地址空间"，操作系统按需换页，没有 GPU 显存也没有大内存的机器上，模型文件甚至不需要真正一次性"读进"内存就能开始跑（虽然会慢，但不会直接因为内存不够而失败）；量化 block 的字节布局和文件布局是同一份东西，不需要额外的解包步骤。
- **不这样会怎样**：退回 safetensors/state_dict 反序列化，需要先把整个文件读进内存构造出 Python/C++ 对象，再拷贝一遍到最终存放的位置，在内存紧张的目标机器上这一份额外拷贝可能就是"跑得起来"和"跑不起来"的区别。
- **什么时候可以不这样**：目标机器本来就有大显存 GPU、内存完全不是瓶颈，或者模型权重经常需要动态更新（比如边训练边推理），这种场景下 safetensors/state_dict 这种更贴近 Python 对象模型、更容易和训练代码共享的格式反而更省事——这正是 vLLM 选它的原因（`## 4.2`）。

**决策二：自己实现 18 个硬件后端，而不是依赖 PyTorch 已经写好的设备抽象**

- **为什么这样**：`## 2.1` 那条"无运行时依赖"是全篇的地基决策，一旦依赖 PyTorch，分发就不再是一个单文件，`GGML_CUDA`/`GGML_HIP` 这些编译期开关默认 `OFF`（`llama.cpp:ggml/CMakeLists.txt:199,215`）背后的逻辑就不成立了；反过来自己维护后端抽象（`## 2.4` 的 `ggml_backend_i`），换来的是能覆盖 PyTorch 官方 wheel 覆盖不到的目标——旧款 CPU 的 AVX512 兼容路径、某些嵌入式设备。
- **不这样会怎样**：依赖 PyTorch 的设备后端，能直接获得 CUDA/ROCm 生态里质量最高、迭代最快的那份实现（`## 4.3`），但要接受 PyTorch 本身的体积和依赖树，而且很多边缘设备（树莓派、老旧笔记本的核显）根本没有官方 wheel。
- **什么时候可以不这样**：目标平台就锁定在 x86_64 + 主流 NVIDIA/AMD GPU 这几种确定组合，且不介意运行时依赖——这种情况下直接基于 PyTorch 写，比自己维护 18 个后端目录的工程量小得多，这正是 10 家 Python 路线引擎的选择。

**决策三：新架构必须同时手写 Python 转换器和 C++ 图构建，两道工序**

- **为什么这样**：`conversion/*.py` 只负责一件离线的事——把 HF checkpoint 的权重重新摆放成 GGUF 的字节布局，跑一次就结束；`src/models/*.cpp` 负责的是运行时真正执行前向的计算图，必须能编译进那个无依赖二进制里，不能依赖 Python 在场。两件事目标不同，天然分成两道工序（`## 3` 数出的 87 + 151 个文件）。
- **不这样会怎样**：如果只做权重转换不写图构建，GGUF 文件是转出来了，但没有代码知道怎么把这些张量摆进一张能执行的计算图；反过来只写图构建不做标准化转换，就没有一条能批量从 HF 生态拿到输入的稳定路径——两道工序缺一个，"支持一个新模型"这句话都不成立。
- **什么时候可以不这样**：如果项目目标从来不是"支持任意 HF 上的新模型"，而是"就服务好一个自研的、架构固定的模型"，这条流水线可以大幅简化，甚至跳过通用的转换脚本，直接写一次性的权重导出代码——这不是 llama.cpp 的场景（它的价值恰恰是能追上 HF 生态里层出不穷的新架构），但对一个内部专用引擎是合理的简化。

**决策四：并发用启动时固定数量的 slot，而不是运行时动态分页 KV**

- **为什么这样**：server 侧的并发实现是启动时按 `n_parallel` 分配固定数量的 `server_slot`（`llama.cpp:tools/server/server-context.cpp:196,1231`，日志打印 `n_slots = %d, n_ctx_slot = %d`），每个 slot 拿到一块大小固定的上下文。这个设计跟单机单用户/小规模并发的主场景是匹配的——不需要处理跨请求共享物理 KV 块这种复杂度，内存布局在启动那一刻就能确定，跟 `## 2.1`"不依赖复杂运行时"的哲学是一致的。**需要说清楚的是，llama.cpp 的 server 确实支持 continuous batching**（`llama.cpp:tools/server/README.md:13` 把它列为特性，`:176` 的 `-cb, --cont-batching` 默认开启）——"它不支持批量并发"是一句不准确的话，准确的说法是它的并发上限在启动时由 `n_parallel` 固定下来，不是 vLLM 那种运行时按块动态增长、跨请求复用物理 KV 空间的模型。
- **不这样会怎样**：换成 `03-分页KV与块管理.md` 里讲的那套动态分页机制，需要引入块表、引用计数、跨请求共享与驱逐策略，复杂度陡增——而 llama.cpp 的目标场景（一台机器、几个并发请求）本来就用不上这套机制想解决的问题（大量并发请求下的显存碎片与利用率）。
- **什么时候可以不这样**：如果打算用 llama.cpp 去扛"数据中心里成百上千并发用户"这种场景，固定 slot 模型会先于其他因素成为瓶颈——slot 数量在启动时就封顶了并发上限，KV 显存不能跨请求动态复用，这时候该换的是为这个场景设计的引擎（vLLM/SGLang），而不是把 `n_parallel` 一路调大。

**决策五：自己写 ggml 这个张量库，而不是包一层现成的 C++ 推理运行时**

- **为什么这样**：GGUF 的量化 block 布局（`## 2.2`）是 ggml 团队自己定义的格式，不是套用哪个现成张量库的既有格式；自己写张量库能保证`## 2.1`那条"无运行时依赖"的红线不被打破——引入任何一个第三方 C++ 推理库，都是新增一条依赖链，哪怕它本身也是 C++ 写的。
- **不这样会怎样**：包一层 ONNX Runtime 这类现成推理运行时，能省下自己实现算子和后端抽象层的工作量，但要接受那个运行时自己的依赖树、构建系统、以及它是否原生支持你需要的量化格式和后端组合——量化格式往往是这类通用运行时覆盖最弱的一块。
- **什么时候可以不这样**：如果目标平台本来就对某个现成的 C++ 推理运行时有良好支持，且不需要 llama.cpp 独有的量化格式和 GGUF 生态（比如已经有大量模型转换好的 GGUF 权重可以直接下载），直接用现成运行时可能比自己维护 18 个后端目录、151 个图构建文件更省力——这是一个"要不要自己造轮子"的工程判断，不是"C++ 路线必须自己造张量库"这条铁律。

**决策六：对外暴露纯 C ABI（`llama.h`），而不是直接暴露 C++ 类接口**

- **为什么这样**：库的实现是 C++，但公共头文件 `llama.h` 一开始就包在 `extern "C"` 里（`llama.cpp:include/llama.h:52`）——这意味着对外暴露的是没有 C++ name mangling、没有 C++ ABI 稳定性问题的纯 C 函数签名。这是 `## 2.1` "无运行时依赖"这条哲学的自然延伸：不仅运行时不依赖 Python/CUDA，连"调用它的代码用什么语言写"都不必须是 C++——任何能链接 C 函数的语言，理论上都能直接调用这个库。
- **不这样会怎样**：如果对外暴露的是 C++ 类和模板，调用方必须用兼容的编译器版本和 C++ ABI（不同编译器、甚至同一编译器不同版本，C++ 的 ABI 都可能不兼容），内部实现细节的改动很容易间接破坏对外接口，把使用者绑死在特定的编译器/版本组合上。
- **什么时候可以不这样**：如果这个库注定只会被同一个代码库内部使用，不需要作为可被其他语言/项目链接的组件对外分发，直接暴露 C++ 接口能省下设计和维护这层 C 封装的工作量——这是很多不打算做成通用库、只服务单一产品的内部代码的合理选择。

### 5.2 判据表：什么时候该走这条路，什么时候不该

| 场景特征 | 结论 | 理由 |
|---|---|---|
| 端侧部署（笔记本/手机/树莓派）、目标机器没有 GPU | **该走这条路** | `## 2.1` 的无运行时依赖是分发能成立的前提，`## 2.3` 的 CPU SIMD 下沉是没有 GPU 时唯一能指望的算力来源 |
| 离线分发、二进制交付、不能假设目标机器装了 Python/CUDA | **该走这条路** | `## 1.2` 的分发方式差异是最直接的判据——一行 `pip install` 做不到的场景，正是这条路的主场 |
| 单用户/低并发、强隐私要求（数据不能出本机） | **该走这条路** | `## 5.1` 决策四的固定 slot 模型完全够用，本地跑还顺带满足"数据不出机器"的隐私约束 |
| 多租户在线服务、高并发、要压榨 GPU 集群吞吐 | **不该走这条路** | `## 5.1` 决策四讲清楚了固定 slot 模型的并发上限；这正是 vLLM/SGLang 那套动态分页 + 连续批处理解决的问题，见 `03-分页KV与块管理.md`、`04-连续批处理.md` |
| 需要第一时间跟进最新模型架构、模型迭代频率高 | **不该走这条路（或接受滞后）** | `## 4.1`、`## 5.1` 决策三讲清楚了新架构要走两道手写工序，`_PLAN.md` 记录的 148 个架构里未来还会持续增长，这份滞后是结构性的 |
| 团队没有 C++ 工程能力，或团队本来就在 Python/PyTorch 生态里 | **不该走这条路** | `## 2.2`～`## 2.4` 的每一条收益都要求团队能读写 C++、理解内存布局和后端抽象；`02-Python为什么还没被淘汰.md` 已经证明 Python 在 GPU 场景下不是瓶颈，硬换语言换不来收益，只会付出团队生产力的代价 |

这张表读法跟 `02-Python为什么还没被淘汰.md` 5.2 节的判据表是同一个方法论：**先问约束是什么，再问语言选型**。约束是"数据中心 GPU 集群服务并发用户"，答案几乎总是 Python；约束是"没有 GPU 的单机/端侧部署"，答案才轮到 C++ 这条路，而且轮到它的理由从来不是"更快"，是`## 2.5`那句话——不需要那一整套运行时。

## 6. 常见错误与踩坑

**误区一："C++ 比 Python 快，所以推理引擎该用 C++"**

`## 2.5` 已经拆穿过一次，这里从选型决策的角度再钉一遍：这句话默认了"引擎变快的杠杆在语言执行速度上"，但 `02-自回归解码为什么是访存瓶颈.md` 证明了单条请求的 decode 从第一步开始就站在访存带宽的地板上，`02-Python为什么还没被淘汰.md` 证明了 GPU 场景下 Python 的调度开销会被异步 kernel launch 掩盖——两条证据合起来说明，在 GPU 场景下换语言换不来热路径的速度提升，llama.cpp 自己在 GPU 后端上跑的仍然是同一批 CUDA/Metal/Vulkan kernel（`## 2.4`）。这条误区的正确改法是：不要问"这门语言快不快"，要问"这个场景下瓶颈到底在哪"——瓶颈在带宽，语言选型改变不了带宽；瓶颈在"要不要背一整套运行时"，这才是语言选型能真正起作用的地方。

**误区二："llama.cpp 不支持并发/批处理，只能单用户"**

这句话在 llama.cpp 早期版本或许更接近事实，但现在不成立——server 明确把 continuous batching 列为特性（`llama.cpp:tools/server/README.md:13`），`-cb, --cont-batching` 默认开启（`:176`），`server_slot` 数组按 `n_parallel` 分配多个槽位并行处理请求（`llama.cpp:tools/server/server-context.cpp:196,1231`）。准确的表述不是"不支持"，是 `## 5.1` 决策四讲的那句话：**它的并发上限在启动时由 `n_parallel` 固定，不是运行时动态分页扩展的**。把"设计重心不在这里"说成"不支持"，会让人误判它完全无法用于任何多用户场景（现实是"能用，但槽位数量和显存都是启动时定死的"）；反过来把它和 vLLM 的动态分页混为一谈，又会在真正需要高并发弹性的场景里选错引擎——两种误判都源于没有去 `tools/server/` 下实际确认现状。

**误区三："新模型支持慢，是 llama.cpp 团队不给力/不上心"**

`## 3`、`## 4.1` 数出来的数字直接反驳这句话：87 个 `conversion/*.py` 转换文件、151 个 `src/models/*.cpp` 图构建文件，覆盖 148 个架构分支——这是**两道手写工序 × 148 个架构**的工作量，不是团队懒。对照 vLLM 的 `vllm:vllm/model_executor/models/registry.py:72`，一个新架构常常只需要往字典里加一行、指向一个复用了 HF `transformers` 已有实现的薄封装，两条路线要做的工作量在数量级上就不一样。这不是"C++团队效率低"，是`## 2.1`那条决策——拒绝依赖 PyTorch/HF 生态——的结构性代价：省下的是运行时依赖，付出的是每个新架构都要重新手写两遍。

**误区四："GGUF/量化 block 是 C++ 独有的黑科技，别的引擎学不会"**

`## 4.4` 已经给出了反例：vLLM 也能读 GGUF 文件，说明它不是不可跨越的技术壁垒。但反过来把这条反例理解成"GGUF 和 C++ 路线没有特殊关系"同样是错的——vLLM 自己的文档把这条能力标注成"highly experimental...migrated to OOT plugin"（`vllm:docs/features/quantization/gguf.md:4,7`），这说明 GGUF 在 Python 生态里是外来客人，需要单独装插件、状态不稳定，跟它在 llama.cpp 里作为原生格式、被整个加载路径（mmap + 量化 block）围绕它设计出来的地位完全不是一回事。准确的说法是：**格式本身可以被移植，但格式设计时假设的运行环境（无依赖二进制、mmap 直接可用）移植不过去**，脱离了这个环境，格式的大部分收益也跟着消失，只剩下"能读"这一层最基础的兼容性。

**误区五："选了 C++ 路线 = 自动获得更高的工程质量、更少的 bug"**

这条误区把"语言更底层"和"代码更可靠"划了等号，但 `## 3` 里 llama.cpp 自己的训练示例文档反而是反例——`examples/training/README.md` 自己承认"the code is very much WIP"（`llama.cpp:examples/training/README.md:4`）。更结构性的原因是 `## 4.1` 讲的那条：Python 路线的新架构常常复用 HF 官方已经经过大量用户验证的参考实现，C++ 路线的每个新架构都要重新手写图构建、重新验证数值正确性，不能像 Python 路线那样借到"这段代码已经被无数人跑过"的信任积累。**C++ 在这条路上真正换来的是分发性和可移植性，不是正确性保障**——一份手写的 C++ 图构建代码和一份手写的 Python `nn.Module`，在"这段代码本身有没有 bug"这个维度上没有哪种语言天然更安全，llama.cpp 能保持较高质量，靠的是活跃的社区和持续的测试投入，不是"用了 C++"这件事本身。

五条误区合起来指向同一条方法论：**"选 C++ 换来了什么"这个问题的每一个正确答案，都要能在源码里找到具体落点**（`## 2` 的四条收益、`## 5` 的五条代价都是这么写的）；反过来，任何一句关于 llama.cpp 的笼统判断——无论是吹捧"更快"还是贬低"功能不全"——不去 `tools/server/`、`conversion/`、`src/models/` 里确认一下现状，都经不起推敲。

## 7. 自测题与延伸阅读

**自测题（闭卷回答，答案都在本篇里）**：

1. llama.cpp 和 Mooncake 的主语言都是 C++，但本篇说"真正的 C++ 推理引擎阵营只有 llama.cpp 一家"，理由是什么？Mooncake 实际上是什么？
2. `## 2` 列出的四条 C++ 带来的收益——无运行时依赖、内存布局控制、CPU SIMD 下沉、后端可移植——里，有没有一条叫"运算更快"？如果没有，llama.cpp 在 GPU 后端上执行 decode，真正做浮点运算的是谁？
3. GGUF + mmap 把"加载模型"这个动作变成了什么？这跟 safetensors 反序列化在内存使用上的本质区别是什么？
4. `block_q4_0` 这个 struct 用 `static_assert` 强制检查大小，这条检查在保证什么？如果这个 struct 的字节布局和 GGUF 文件里的实际布局对不上，会发生什么？
5. `ggml_backend_sched_new` 能同时接收 `backend_gpu`、`backend_gpu2`、`backend_cpu` 三个后端参数，这说明 llama.cpp 的后端抽象解决的是"针对某一种硬件抠到极致"，还是"同一份计算图能在多种硬件上执行"？两者有什么区别？
6. llama.cpp 支持一个新架构需要走哪两道工序？分别对应仓库里的哪两个目录？`## 3` 里数出来的文件数分别是多少？
7. vLLM 的 `_TEXT_GENERATION_MODELS` 注册表里加一行能省掉 llama.cpp 那两道工序里的哪一部分？为什么不能把两道工序都省掉？
8. llama.cpp 的 server 到底支不支持并发请求？如果支持，它的并发模型和 vLLM 的动态分页模型本质区别在哪一个环节？
9. GGUF 能不能在 vLLM 里读取？vLLM 自己的文档怎么形容这个功能的状态？这条事实反驳的是哪一句误解，又印证了本篇的哪个论点？
10. `examples/training/README.md` 里那句"the code is very much WIP"，反驳的是选型层面的哪条误区？它和"C++ 路线天然只做推理"这条判断是什么关系？
11. 如果要在一台没有 GPU 的树莓派上跑一个本地问答助手，`## 5.2` 的判据表会给出什么结论？如果同一个模型要同时给上千个并发用户提供在线服务，结论会怎么变？
12. "C++ 比 Python 快，所以推理引擎该用 C++"这句话错在哪一步推理？把它改成一句站得住的表述，应该怎么说？

**延伸阅读（本库内）**：

- [[01-用什么语言写推理引擎-12个真实引擎的证据]] —— 本篇开头引用的事实基础，12 个引擎完整语言构成表和"Mooncake 不是推理引擎"这条结论的原始出处
- [[02-Python为什么还没被淘汰]] —— `## 2.5`、`## 6` 误区一借用的 `T_gpu`/`T_py` 判据框架完整版，以及 5.2 节判据表的姊妹版本
- [[02-自回归解码为什么是访存瓶颈]] —— `## 2.5` 拆穿"C++ 更快"这条误解时依赖的核心结论：decode 从第一步开始就在访存带宽的地板上
- [[03-分页KV与块管理]] —— `## 5.1` 决策四、`## 5.2` 判据表里"该换 vLLM/SGLang"那一行指向的具体机制，llama.cpp 固定 slot 模型的对照组
- [[06-选型决策表]] —— 本篇 `## 5.2` 判据表是它在"端侧/无 GPU"这个分支上的完整展开

双链：[[01-用什么语言写推理引擎-12个真实引擎的证据]] [[02-Python为什么还没被淘汰]] [[02-自回归解码为什么是访存瓶颈]] [[03-分页KV与块管理]] [[06-选型决策表]]
