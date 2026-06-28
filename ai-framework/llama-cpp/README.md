# llama.cpp 与 GGUF 量化推理

> 用纯 C/C++ 在 CPU/消费级 GPU 上跑大模型：GGUF 格式 + 块量化（K-quant / I-quant）+ mmap 加载。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-inference/README]] · [[llm-inference/vllm/README]]

## 阅读地图

| 小节 | 你会得到什么 | 何时回来看 |
|------|--------------|-----------|
| 0. 一句话锚点 | llama.cpp 到底解决什么问题 | 第一次接触 |
| 1. 地基/前置 | 量化、bpw、perplexity 的最小知识 | 看不懂 Q4_K_M 时 |
| 2. GGUF 文件格式 | 为什么不是 .bin/.safetensors | 转换模型时 |
| 3. 块量化原理 | Q4_0 / K-quant / I-quant 怎么压 | 选量化方法时 |
| 4. 命名解码 | Q4_K_M 的每个字母含义 | 看到一堆后缀懵了 |
| 5. 量化档位对照表（原文真料） | 体积 vs 精度损失的硬数据 | 做取舍决策时 |
| 6. 推理执行：mmap + n_gpu_layers | 24G 显存怎么跑 70B | 显存不够时 |
| 实操 | 编译/安装/起服务（原文命令） | 动手时 |
| 常见坑 | 别人踩过的雷 | 报错时 |

## 0. 一句话锚点

**llama.cpp 是一个用纯 C/C++ 写的大模型推理引擎，核心卖点是「把权重量化成 4 比特左右、用 GGUF 单文件 + mmap 加载、在 CPU 或消费级显卡上把 7B~70B 模型跑起来」。** 配套的 `llama-cpp-python` 提供 Python 绑定与 OpenAI 兼容的 HTTP server。

参考：
- https://github.com/ggerganov/llama.cpp
- https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file

与 vLLM/TensorRT-LLM 的分工：vLLM 面向数据中心 A100/H100 的高吞吐服务（PagedAttention、连续批处理），llama.cpp 面向**单机、低显存、CPU 也能跑**的本地部署。两者解决的是光谱的两端。

## 1. 地基/前置

### 1.1 什么是量化（一句话）

权重原本是 FP16（每个数 16 bit）。量化就是用更少的比特（4 bit、2 bit）近似表示这些数，体积按比例缩小。

设一组权重 $w_i$，FP16 占 16 bit/个。最朴素的「对称线性量化」：

$$q_i = \mathrm{round}\!\left(\frac{w_i}{s}\right), \quad s = \frac{\max_i |w_i|}{2^{b-1}-1}$$

其中 $b$ 是比特数，$s$ 是缩放因子（scale）。反量化 $\hat w_i = s \cdot q_i$。比特越少，$s$ 越粗，误差越大。

### 1.2 bpw（bits per weight）

GGUF 量化档位常标 "2.06 bpw"、"4.50 bpw"。bpw = **平均每个权重占多少比特**，它**不是整数**，因为还要摊上 scale、min、查找表等元数据。例如 Q4_0 名义 4 bit，实际约 4.5 bpw。

模型体积估算：

$$\text{体积} \approx \text{参数量} \times \frac{\text{bpw}}{8} \text{ 字节}$$

8B 模型 @ 4.5 bpw $\approx 8\times10^9 \times 4.5/8 \approx 4.5$ GB —— 正好对上原文表里 Q4_K_M 的 4.58G。

### 1.3 perplexity（困惑度，ppl）

衡量量化「掉了多少精度」的指标。ppl 越低越好；量化会让 ppl 变大。原文表里的 `+0.1754 ppl @ Llama-3-8B` 就是「相对 FP16，困惑度上升了 0.1754」。

$$\mathrm{ppl} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x_i \mid x_{<i})\right)$$

直觉：ppl 是「模型平均在多少个候选词里纠结」。+0.02 几乎无感，+3.5（如 Q2_K）则明显变蠢。

## 2. GGUF 文件格式：为什么不用 .safetensors

GGUF（GPT-Generated Unified Format）是 llama.cpp 的原生格式，取代了老的 GGML。它把**一切打进一个文件**：

```
 GGUF 单文件
┌──────────────────────────────────────────────┐
│ Header  魔数 "GGUF" + 版本号                   │
├──────────────────────────────────────────────┤
│ Metadata (KV 键值对)                           │
│   general.architecture = "llama"               │
│   llama.context_length = 8192                  │
│   llama.embedding_length = 4096                │
│   tokenizer.ggml.tokens = [...]   ← 词表内嵌   │
│   tokenizer.ggml.merges = [...]                │
├──────────────────────────────────────────────┤
│ Tensor Info  每个张量的名字/形状/量化类型/偏移  │
├──────────────────────────────────────────────┤
│ Tensor Data  量化后的权重（按块存）             │
│   [block0][block1][block2]...                  │
└──────────────────────────────────────────────┘
```

**为什么是单文件 + 内嵌 tokenizer**：
- 部署只需 copy 一个 `.gguf`，不用 config.json + tokenizer.json + 多个 shard 一起搬。
- 张量数据按字节对齐排布，可以用 **mmap** 直接映射进地址空间（见第 6 节），省一次显存/内存拷贝。
- 元数据自描述：架构、上下文长度、RoPE 参数全在 KV 里，加载器不依赖外部 config。

> 涉及到的架构元数据（context_length、attention、RoPE）背后的原理见 [[llm-algo/transformer/模型架构]] 与 [[llm-algo/旋转编码RoPE]]。

## 3. 块量化原理：Q4_0 → K-quant → I-quant 三代

llama.cpp 不对整层用一个 scale，而是**分块（block）**量化，典型块大小 = 32 个权重。块越小，scale 越贴合局部分布，误差越小，但元数据开销越大。

### 3.1 第一代：Q4_0 / Q4_1（朴素块量化）

每 32 个权重一块：

```
Q4_0 一个块 = 1 个 scale(fp16) + 32 个 4-bit 量化值
┌────────┬──┬──┬──┬──┬──┬──┬──┬──┐
│ scale  │q0│q1│q2│q3│..│..│..│q31│
│ 16 bit │4b│4b│...           ...│
└────────┴──┴──┴──┴──┴──┴──┴──┴──┘
 实际占用 = 16 + 32×4 = 144 bit / 32 个权重 = 4.5 bpw
```

- **Q4_0**：只存 scale（对称量化，假设零点为 0）。
- **Q4_1**：额外存一个 min（非对称量化 $\hat w = s\cdot q + m$），更准但更大。看原文表：Q4_0 4.34G / +0.4685 ppl，Q4_1 4.78G / +0.4511 ppl —— 多花体积换更小损失。

### 3.2 第二代：K-quant（Q*_K，超级块）

K-quant 引入「**super-block**」两级 scale：一个超级块（256 权重）含多个子块，子块 scale 自己也被量化。不同张量用不同精度（注意力/FFN 关键层给更高比特），所以同一档位里 attention.wv、ffn_down 等会被「混搭」。

后缀 S/M/L = Small/Medium/Large，表示混合策略里高精度层用得多还是少：

```
Q4_K_M：大部分层 Q4_K，部分关键层（如 ffn_down、attn_v）用 Q6_K
        → 体积略大、ppl 损失更小
Q4_K_S：更激进，更多层留 Q4_K → 体积小一点、ppl 略差
```

对照原文：Q4_K_S 4.37G / +0.2689，Q4_K_M 4.58G / +0.1754 —— M 多 0.2G 换损失减半。

> 「不同层给不同精度」的混合量化思想，与 [[llm-compression/quantization/GPTQ]] 的逐层误差补偿、[[llm-compression/quantization/量化基础]] 同源。MoE 模型的专家层量化另见 [[llm-algo/moe/README]]。

### 3.3 第三代：I-quant（IQ*，码本量化）

IQ 系列（IQ2_XXS、IQ3_S、IQ4_NL…）用 **codebook / 非线性映射**，借鉴 QuIP# 思路，在 2~3 bit 极低比特下显著优于 K-quant。代价：推理时要查表，CPU 上可能更慢，且部分需要 imatrix（重要性矩阵）校准。

```
比特预算光谱（bpw 越小越省，越右越准）
 1.56   2.06    2.5    3.0    3.44   4.25   4.5    4.58    5.33   6.14  7.96  16
  │      │       │      │      │      │      │      │       │      │     │    │
 IQ1_S  IQ2_XXS IQ2_S IQ3_XXS IQ3_S IQ4_XS IQ4_NL Q4_K_M  Q5_K_M Q6_K  Q8_0 F16
  └─ 几乎不可用 ─┘└── 凑合 ──┘└──── 甜点区（推荐）────┘└─ 近无损 ─┘└无损┘
```

TQ1_0 / TQ2_0 是**三值化（ternary）**，对应 1.58-bit BitNet 类模型，权重只取 {-1,0,+1}。

## 4. 命名解码：把 Q4_K_M 拆成原子

```
   Q   4   _K   _M
   │   │    │    └─ 混合策略大小：S(small)/M(medium)/L(large)
   │   │    └────── 量化代际：K = K-quant 超级块（无 K = 第一代 Q4_0/Q4_1）
   │   └─────────── 名义比特数：4 → 约 4 bit
   └─────────────── Quantized（量化）

   IQ3_XXS：I=I-quant，3=约3bit，XXS=extra-extra-small（最省）
   F16 / BF16 / F32：未量化的浮点基线（BF16 指数位多，数值范围大）
```

记忆口诀：**字母越靠后（M>S）体积越大但越准；I 系列同比特更准但更慢；带 K 的是主流甜点。**

## 5. 量化档位对照表（原文真料，逐档精度/体积）

下表来自 llama.cpp 源码 `examples/quantize/quantize.cpp` 的 `QUANT_OPTIONS`（数据基于 Llama-3-8B / Mistral-7B 实测）。**官方推荐：Q4_K_M、Q5_K_S、Q5_K_M。**

源码：https://github.com/ggerganov/llama.cpp/blob/3e693197724c31d53a9b69018c2f1bd0b93ebab2/examples/quantize/quantize.cpp#L18

```cpp
static const std::vector<struct quant_option> QUANT_OPTIONS = {
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 4.78G, +0.4511 ppl @ Llama-3-8B",  },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 5.21G, +0.1316 ppl @ Llama-3-8B",  },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 5.65G, +0.1062 ppl @ Llama-3-8B",  },
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization",            },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization",            },
    { "IQ2_S",    LLAMA_FTYPE_MOSTLY_IQ2_S,    " 2.5  bpw quantization",            },
    { "IQ2_M",    LLAMA_FTYPE_MOSTLY_IQ2_M,    " 2.7  bpw quantization",            },
    { "IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S,    " 1.56 bpw quantization",            },
    { "IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M,    " 1.75 bpw quantization",            },
    { "TQ1_0",    LLAMA_FTYPE_MOSTLY_TQ1_0,    " 1.69 bpw ternarization",           },
    { "TQ2_0",    LLAMA_FTYPE_MOSTLY_TQ2_0,    " 2.06 bpw ternarization",           },
    { "Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K,     " 2.96G, +3.5199 ppl @ Llama-3-8B",  },
    { "Q2_K_S",   LLAMA_FTYPE_MOSTLY_Q2_K_S,   " 2.96G, +3.1836 ppl @ Llama-3-8B",  },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization",            },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization",            },
    { "IQ3_M",    LLAMA_FTYPE_MOSTLY_IQ3_M,    " 3.66 bpw quantization mix",        },
    { "Q3_K",     LLAMA_FTYPE_MOSTLY_Q3_K_M,   "alias for Q3_K_M"                   },
    { "IQ3_XS",   LLAMA_FTYPE_MOSTLY_IQ3_XS,   " 3.3 bpw quantization",             },
    { "Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S,   " 3.41G, +1.6321 ppl @ Llama-3-8B",  },
    { "Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.74G, +0.6569 ppl @ Llama-3-8B",  },
    { "Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L,   " 4.03G, +0.5562 ppl @ Llama-3-8B",  },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
    { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
    { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   "alias for Q4_K_M",                  },
    { "Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S,   " 4.37G, +0.2689 ppl @ Llama-3-8B",  },
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 4.58G, +0.1754 ppl @ Llama-3-8B",  },
    { "Q5_K",     LLAMA_FTYPE_MOSTLY_Q5_K_M,   "alias for Q5_K_M",                  },
    { "Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S,   " 5.21G, +0.1049 ppl @ Llama-3-8B",  },
    { "Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 5.33G, +0.0569 ppl @ Llama-3-8B",  },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 6.14G, +0.0217 ppl @ Llama-3-8B",  },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 7.96G, +0.0026 ppl @ Llama-3-8B",  },
    { "F16",      LLAMA_FTYPE_MOSTLY_F16,      "14.00G, +0.0020 ppl @ Mistral-7B",  },
    { "BF16",     LLAMA_FTYPE_MOSTLY_BF16,     "14.00G, -0.0050 ppl @ Mistral-7B",  },
    { "F32",      LLAMA_FTYPE_ALL_F32,         "26.00G              @ 7B",          },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing",  },
};
```

### 怎么读这张表（决策提炼）

| 你的处境 | 选什么 | 依据（表里数据） |
|----------|--------|------------------|
| 显存够、要质量 | Q6_K / Q8_0 | +0.0217 / +0.0026 ppl，几乎无损 |
| 主流甜点（最常用） | **Q4_K_M** | 4.58G，+0.1754，官方"recommended" |
| 再省 0.2G | Q4_K_S | 4.37G，+0.2689，损失翻倍可接受 |
| 质量优先一点 | Q5_K_M | 5.33G，+0.0569，损失再砍 2/3 |
| 显存极度紧张 | IQ3_M / IQ2_M | 极低 bpw，靠 I-quant 兜底 |
| **不要碰** | Q2_K | +3.5199 ppl，模型明显变蠢 |

> 经验法则：**只要装得下，Q4_K_M 起步；能上 Q5_K_M / Q6_K 就上。** 低于 3 bit 只在万不得已（显存实在不够）时用 I-quant。

延伸阅读（原文给的参考）：
- 量化 PR（值得读）：https://github.com/ggerganov/llama.cpp/pull/1684
- GGUF/GGML 格式 + Q4_0/Q4_1/Q4_K/Q4_K_M 区别：https://blog.csdn.net/weixin_42426841/article/details/142706753
- LLM 量化大比拼：https://zhuanlan.zhihu.com/p/8936080946
- llama.cpp 量化方法简介：https://zhuanlan.zhihu.com/p/12729759086

GGUF 量化工具生态：ctransformers、llama.cpp 本身的 `llama-quantize`。

## 6. 推理执行：mmap + n_gpu_layers 混合卸载

### 6.1 mmap：为什么加载这么快、内存这么省

llama.cpp 默认用 `mmap` 把 GGUF 文件**映射**进进程地址空间，而不是 `read` 到堆里：

```
传统 read：              mmap：
磁盘 → 内核缓冲 → 用户堆   磁盘 ←→ 页缓存 ←→ 进程虚拟地址
（拷贝两次、占双份内存）   （按需缺页加载、多进程共享同一份）
```

好处：启动快（不必先读完整文件）、多个进程共享只读权重、OS 自动按访问换页。这也是「单文件 + 字节对齐」格式设计的回报。

### 6.2 n_gpu_layers：把多少层放显卡

模型按 Transformer 层切分，可以**前若干层放 GPU、其余留 CPU**。`n_gpu_layers=N` 表示把 N 层卸载到 GPU。

```
n_gpu_layers = 20（共 32 层为例）
┌─────────── GPU 显存 ───────────┐ ┌──── CPU 内存 ────┐
│ layer 0..19（含 KV cache 一部分）│ │ layer 20..31     │
└────────────────────────────────┘ └──────────────────┘
   算得快                              算得慢，但省显存
```

- `n_gpu_layers=0`：纯 CPU。
- `n_gpu_layers=-1`（或足够大）：全部上 GPU。
- 显存不够时调小，直到「装得下」为止——这是 llama.cpp 能在 24G 卡上跑 70B（量化版）的关键。

> KV cache 同样吃显存且随上下文线性增长，原理见 [[llm-optimizer/kv-cache]]；注意力加速见 [[llm-optimizer/FlashAttention]]；整体推理流水线见 [[llm-inference/README]]、解码采样见 [[llm-inference/解码策略]]。

## 实操：编译 / 安装 / 起服务（原文命令保留）

### 安装 llama-cpp-python（含 server）

macOS Metal（Apple GPU）后端编译安装：

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'
```

说明：`CMAKE_ARGS` 控制后端编译开关。`-DGGML_METAL=on` 开 Apple Metal；对应地，CUDA 用 `-DGGML_CUDA=on`，CPU+OpenBLAS 用 `-DGGML_BLAS=on`。`[server]` 这个 extra 装的是 OpenAI 兼容 HTTP 服务依赖。

### 启动 OpenAI 兼容服务（加载 GGUF 模型）

```bash
export MODEL=/Users/liguodong/model/qwen2/qwen2-0_5b-instruct-q2_k.gguf
python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers 1
```

逐项解释：
- `MODEL` 指向一个 GGUF 文件，文件名 `...-q2_k.gguf` 表明它是 **Q2_K** 量化的 Qwen2-0.5B-instruct（这是个 demo 档位，正式用建议 Q4_K_M 以上，见第 5 节）。
- `--n_gpu_layers 1`：只卸载 1 层到 GPU（小模型/试跑），见 6.2。
- 起来后即暴露 `/v1/chat/completions` 等 OpenAI 风格端点，可直接用 openai SDK 指向本地。

> 同样是 OpenAI 兼容 server，vLLM 的服务化与连续批处理对比见 [[llm-inference/vllm/README]]；PD 分离架构见 [[llm-inference/PD分离]]。

## 常见问题 / 坑

| 现象 | 原因 | 解法 |
|------|------|------|
| 选了 Q2_K 后模型胡言乱语 | +3.5 ppl，2-bit 损失过大 | 换 Q4_K_M 及以上 |
| `pip install` 没用上 GPU | 没传 `CMAKE_ARGS` 后端开关 | 重装并带 `-DGGML_CUDA=on`/`-DGGML_METAL=on`，加 `--no-cache-dir` |
| 显存 OOM | `n_gpu_layers` 太大 / 上下文太长 | 调小 `--n_gpu_layers`；缩短 ctx（KV cache 随上下文线性涨） |
| `[server]` 命令找不到 | 只装了核心包没装 server extra | `pip install 'llama-cpp-python[server]'` |
| IQ 系列推理偏慢 | 码本/查表 + 部分需 imatrix | CPU 场景优先 K-quant；要极低比特再上 IQ |
| 模型名后缀看不懂 | Q/数字/K/S-M-L 多层含义 | 见第 4 节命名解码 |
| 不同来源同名 gguf 体积不一 | 混合量化策略 + imatrix 校准差异 | 认准来源，参考表内 bpw/ppl |
| 改了 `CMAKE_ARGS` 没生效 | pip 用了旧缓存轮子 | 必须加 `--no-cache-dir` 强制重编 |

## 🔗 跳转链接

- 枢纽地图：[[00-知识地图]]
- 模型架构基础：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 推理加速：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理引擎：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 量化邻居：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练/微调对照：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架对照：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 硬件底座：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测/估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
