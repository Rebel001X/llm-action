# TensorRT-LLM：NVIDIA 官方 LLM 推理引擎

> 用 Python API 把 LLM「编译」成 TensorRT 引擎，在 NVIDIA GPU 上榨干每一个 SM 的推理速度。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-compression/quantization/fp8]] · [[ai-infra/算力/GPU工作原理]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
| :-- | :-- | :-- |
| 0 | TensorRT-LLM 一句话是什么 | 编译式推理 |
| 1 | 前置：TensorRT / 算子融合 / SM 架构 | 地基 |
| 2 | 整体架构：从 Python 定义到引擎执行 | build→engine→runtime |
| 3 | 三层 API：functional / layers / models | einsum/Attention/GPTAttention |
| 4 | Batch Manager（In-flight Batching） | 连续批处理 |
| 5 | 量化全景：FP8 / INT8 / INT4 / SmoothQuant | 精度 vs 速度 |
| 6 | 精度支持矩阵：哪代卡支持哪种精度 | SM70~SM90 |
| 7 | 模型 × 量化支持矩阵 | LLaMA/GPT/... |
| 实操 | 参考链接与典型工作流 | build & serve |
| 坑 | 常见问题对照表 | 排错 |

## 0. 一句话锚点

> **TensorRT-LLM = 用 PyTorch 风格的 Python API 描述 LLM → 编译成一个针对特定 GPU/特定 shape 高度优化的 TensorRT 引擎（.engine 文件）→ 用 C++/Python runtime 或 Triton 后端高速执行推理。**

它和 vLLM 的根本区别是「**编译式**」：vLLM 在运行时即时调度 PyTorch 算子，TensorRT-LLM 则**提前（AOT）把整个网络融合、选核、固化**成一个二进制引擎。换性能的代价是：引擎和 GPU 型号、并行度、batch/seq 上限强绑定，换卡或换 shape 通常要重新 build。

```
 vLLM（运行时解释）            TensorRT-LLM（提前编译）
 ┌───────────┐               ┌───────────┐
 │ HF 权重    │               │ HF 权重    │
 └────┬──────┘               └────┬──────┘
      │ 直接加载                   │ trtllm-build（一次，几分钟~几十分钟）
      ▼                           ▼
 ┌───────────┐               ┌──────────────┐
 │ PyTorch 图 │  每步解释执行  │ .engine 二进制│ 已融合/已选核
 └───────────┘               └──────┬───────┘
                                    │ runtime 加载（毫秒级）
                                    ▼
                              超低延迟推理
```

## 1. 地基：你得先有的三个概念

**① TensorRT 是什么。** TensorRT 是 NVIDIA 的通用深度学习**推理优化器 + 运行时**。它接收一张计算图，做四件事：算子融合（fusion）、精度校准（FP16/INT8/FP8）、为当前 GPU **自动选最快的 kernel**（kernel auto-tuning）、显存复用规划。产物是 `.engine`。TensorRT-LLM 是在 TensorRT 之上、专门为 Transformer/LLM 写的一套**插件 + Python 前端**——因为通用 TensorRT 不懂 KV-Cache、不懂 in-flight batching、不懂 RoPE/GQA 这些 LLM 专属结构。

**② 算子融合为什么快。** GPU 推理大量时间花在「把数据从显存搬到 SM、算一下、再写回显存」。把 `matmul→bias→gelu` 三个算子融成一个 kernel，中间结果留在寄存器/共享内存，省掉两次显存往返。LLM 里最关键的融合就是 **Attention 融合**（QKV 投影 + softmax + KV-Cache 读写一体化），TensorRT-LLM 用 `GPTAttention` 插件实现，对应思想见 [[llm-optimizer/FlashAttention]]。

**③ SM 架构代号。** 后文矩阵里的 SM70/75/80/86/89/90 是 GPU 的「计算能力（Compute Capability）」：

| 架构 | SM | 代表卡 | 新增能力 |
| :-- | :-- | :-- | :-- |
| Volta | SM70 | V100 | 第一代 Tensor Core（FP16） |
| Turing | SM75 | T4 | INT8 Tensor Core |
| Ampere | SM80/86 | A100 / A10 | **BF16**、TF32 |
| Ada Lovelace | SM89 | L40S / 4090 | **FP8** |
| Hopper | SM90 | H100 / H800 | FP8 + Transformer Engine |

记住一条主线：**FP8 是 Ada(SM89)/Hopper(SM90) 才有的硬件能力**，这解释了后面精度矩阵里 FP8 那一列为什么只有最后两行是 Y。详见 [[ai-infra/算力/GPU工作原理]] 与 [[ai-infra/ai-hardware/硬件对比]]。

## 2. 整体架构：从 Python 定义到引擎执行

> 原文核心定位（保留）：TensorRT-LLM 提供易用的 **Python API** 定义 LLM 并构建 **TensorRT 引擎**，在 NVIDIA GPU 上高效推理；包含 **Python 和 C++ 运行时**执行引擎；并含一个**与 NVIDIA Triton 推理服务集成的后端**，为 LLM 服务的生产化提供保障。模型可在**单 GPU 到多节点多 GPU（张量并行/流水线并行）**各种配置上执行。

把这段话拆成三个阶段：

```
 阶段 A：定义/转换          阶段 B：编译            阶段 C：服务
 ┌────────────────┐      ┌──────────────┐      ┌─────────────────┐
 │ Python API 搭网络│ ──▶ │ trtllm-build  │ ──▶ │ Python/C++ runtime│
 │ 或转换 HF 权重   │      │ 生成 .engine  │      │ 或 Triton 后端     │
 └────────────────┘      └──────┬───────┘      └─────────────────┘
                                │ 此时已确定：
                                │  - 目标 GPU 架构（选核）
                                │  - TP/PP 并行度
                                │  - max_batch / max_input_len / max_output_len
                                │  - 量化精度（FP16/FP8/INT8/INT4）
```

**为什么 build 时要固定这些参数？** 因为 TensorRT 要为「**这个 shape 范围 + 这块卡**」选最优 kernel 并预分配显存。`max_batch_size`、`max_input_len`、`max_output_len` 决定 KV-Cache 和激活的显存上界；超出上界的请求引擎跑不了。这就是 TensorRT-LLM 上手门槛比 vLLM 高的根因——你得**提前想清楚 serving 的 shape 包络**。

**并行（TP/PP）发生在 build 阶段。** 张量并行（TP）把每个权重矩阵按行/列切到多卡，每层算完做一次 AllReduce；流水线并行（PP）把不同层放到不同卡。TensorRT-LLM 在 build 时就把通信算子（AllReduce/AllGather）编进引擎，依赖 NCCL。原理见 [[llm-inference/大模型推理张量并行]]、[[ai-infra/网络/集合通信原语]] 与 [[ai-infra/网络/InfiniBand]]。

## 3. 三层 Python API：和 PyTorch 同构

> 原文（保留并解释）：Python API 架构与 PyTorch 类似——**functional 模块**含 `einsum / softmax / matmul / view` 等函数；**layers 模块**捆绑构建块如 `Attention / MLP / 整个 Transformer 层`；**models 模块**含特定于模型的组件如 `GPTAttention / BertAttention`；自带 LLaMA、Bloom 等预定义模型，可修改扩展。

```
          ┌─────────────────────────────────────────┐
 models   │ LLaMAForCausalLM / GPTAttention / BertAttention │  ← 整模型，可直接转 HF 权重
          ├─────────────────────────────────────────┤
 layers   │ Attention · MLP · TransformerLayer        │  ← 积木块
          ├─────────────────────────────────────────┤
functional│ einsum · softmax · matmul · view ...       │  ← 最底层算子，对标 torch.nn.functional
          └─────────────────────────────────────────┘
```

类比理解：

| TensorRT-LLM | PyTorch 对应 | 作用 |
| :-- | :-- | :-- |
| functional | `torch.nn.functional` | 单算子（会被编译成 TRT layer） |
| layers | `torch.nn.Module` 积木 | 组装一层 Transformer，含 [[llm-algo/旋转编码RoPE]] |
| models | `transformers` 里的模型类 | 端到端模型 + 权重映射 |

**为什么要分这三层？** 让你既能「拿 LLaMA 直接 build」，也能「改一层 Attention 做实验」，还能「用 functional 从零拼一个新结构」。整层 Transformer 的拼装逻辑对应 [[llm-algo/transformer/模型架构]]；MoE 类模型对应 [[llm-algo/moe/README]]。

## 4. Batch Manager：In-flight Batching（连续批处理）

原文有 `## Batch Manager` 标题，这里补全其原理——它是 TensorRT-LLM **吞吐的灵魂**。

**问题：静态批处理（static batching）浪费严重。** 一个 batch 里有的请求只生成 10 个 token、有的要 500 个。静态批必须等整批最慢的请求结束才能释放，短请求早早算完却被「锁」在 batch 里空转。

**解法：In-flight Batching（也叫 continuous batching）。** 在**每个解码 step 边界**动态进出请求：谁生成完 EOS 就立刻踢出、释放 KV-Cache；空出的槽位立刻塞进等待队列里的新请求。

```
 step:    t0    t1    t2    t3    t4
 req A   [██]  [██]  [done]
 req B   [██]  [██]  [██]  [██]  [██]
 req C         [新进]  [██]  [done]
 req D                       [新进] [██]
          ↑ A 完成后槽位立即被 C/D 复用，GPU 不空转
```

效果：GPU 利用率从静态批的常常 <50% 拉到接近饱和，吞吐成倍提升。这和 vLLM 的 continuous batching 是同一思想，区别是 TensorRT-LLM 的 batch manager 用 C++ 实现并与 Triton 后端深度集成。配合 PagedKV / KV-Cache 复用见 [[llm-optimizer/kv-cache]] 与 [[llm-inference/KV-Cache优化]]；进一步的 prefill/decode 分离见 [[llm-inference/PD分离]] 与 [[llm-inference/分离式推理架构]]。

## 5. 量化全景：FP8 / INT8 / INT4 / SmoothQuant

> 原文（保留）：为最大化性能并减少内存占用，TensorRT-LLM 支持不同量化模式；支持**仅 INT4/INT8 权重量化（weight-only）**以及 **SmoothQuant 技术的完整实现**。原文另有 `## FP8` 标题。

先建立坐标轴——量化命名 `WxAy` 表示**权重 x bit、激活 y bit**：

| 模式 | 权重 | 激活 | 本质 | 适用 |
| :-- | :-- | :-- | :-- | :-- |
| FP16/BF16 | 16 | 16 | 不量化（基线） | 精度优先 |
| **FP8** | 8 | 8 | 浮点 8 位（E4M3），Hopper 硬件原生 | H100 上又快又准 |
| **W8A8 SQ** | INT8 | INT8 | SmoothQuant：把激活的离群值「搬」到权重，使两者都好量化 | 通用提速 |
| **W8A16** | INT8 | FP16 | weight-only INT8，激活仍 FP16 | 省显存、精度稳 |
| **W4A16** | INT4 | FP16 | weight-only INT4 | 显存吃紧 |
| W4A16 AWQ / GPTQ | INT4 | FP16 | 带校准的 INT4，质量更好 | 4bit 首选 |

**Weight-only 为什么单独成一类？** LLM 推理 decode 阶段是**访存瓶颈（memory-bound）**：每生成一个 token 都要把全部权重从显存读一遍。把权重压到 INT4，显存读取量降到 1/4，decode 直接变快——而激活保持 FP16，精度损失小。这是「以解量化的算力换访存带宽」的划算交易。

**SmoothQuant（W8A8）解决什么。** LLM 激活里存在少数**离群值（outlier）**，幅度比普通值大几十倍，直接 INT8 量化激活会被离群值「撑满」量程，普通值精度全丢。SmoothQuant 用一个逐通道缩放因子 $s$，把激活 $X$ 缩小、权重 $W$ 等比放大，保持 $X W = (X/s)(sW)$ 不变，让**激活和权重都落入好量化的范围**。数学上：

$$Y = X W = \big(X \cdot \mathrm{diag}(s)^{-1}\big)\big(\mathrm{diag}(s)\, W\big),\quad s_j = \frac{\max_i |X_{ij}|^{\alpha}}{\max_i |W_{ij}|^{1-\alpha}}$$

迁移强度 $\alpha$ 通常取 0.5，平衡激活与权重的量化难度。

**FP8 vs INT8 数值直觉。** FP8-E4M3（1 符号 + 4 指数 + 3 尾数）是浮点，动态范围远大于 INT8。同样 8 bit，FP8 能同时表达 0.001 和 100（靠指数），INT8 在固定量程下只能均匀切分。所以 Hopper 上 **FP8 往往比 INT8 精度更好且无需复杂校准**——前提是卡支持（见下节）。量化基础与各方法详见 [[llm-compression/quantization/量化基础]]、[[llm-compression/quantization/GPTQ]]、[[llm-compression/quantization/fp8]] 与 [[llm-compression/README]]。

**手算一笔显存账（7B 模型）。** 70 亿参数：
- FP16：$7\text{e9} \times 2\text{B} = 14\,\text{GB}$
- INT8（W8A16）：$7\text{e9} \times 1\text{B} = 7\,\text{GB}$
- INT4（W4A16）：$7\text{e9} \times 0.5\text{B} = 3.5\,\text{GB}$

这就是为什么 24GB 的 4090 跑不动 FP16 的 70B，但 INT4 量化后 70B（约 35GB）也只差一点点——量化把「装得下/装不下」直接改写。更多估算见 [[docs/transformer内存估算]]。

## 6. 精度支持矩阵：哪代卡支持哪种精度（原文保留）

行 = GPU 架构，列 = 精度。`Y` 支持，`N` 不支持。

|                              | FP32  | FP16  | BF16  | FP8  | INT8 | INT4 |
| :--------------------------- | :---- | :---- | :---- | :--- | :--- | :--- |
| Volta (SM70)                 | Y     | Y     | N     | N    | Y    | Y    |
| Turing (SM75)                | Y     | Y     | N     | N    | Y    | Y    |
| Ampere (SM80, SM86)          | Y     | Y     | Y     | N    | Y    | Y    |
| Ada-Lovelace (SM89)          | Y     | Y     | Y     | Y    | Y    | Y    |
| Hopper (SM90)                | Y     | Y     | Y     | Y    | Y    | Y    |

**怎么读这张表（每一列的「为什么」）：**
- **BF16 列**：SM80 起才有 Y。BF16 是 Ampere 引入的——它指数位和 FP32 一样宽（8 位），训练/推理不易溢出，所以新卡都偏好 BF16 over FP16。
- **FP8 列**：只有 SM89/SM90 是 Y，正是第 1 节说的硬件铁律——FP8 Tensor Core 是 Ada/Hopper 专属。在 A100（SM80）上**无法**用 FP8，强行配置只会报错或回退。
- **INT8/INT4 列**：全 Y，因为 weight-only 量化主要靠软件 + 通用整数算力，老卡也能省显存（虽然 V100 没有 INT4 Tensor Core，吞吐增益不如新卡）。

**实战决策：** 有 H100/H800 → 首选 **FP8**（精度几乎无损、速度最快）；A100 → 没有 FP8，用 **W8A8 SmoothQuant** 或 **W4A16 AWQ**；消费卡显存紧 → **W4A16 GPTQ/AWQ**。

## 7. 模型 × 量化支持矩阵（原文保留）

`Y` 支持，`.` 暂不支持。列含义同上（SQ=SmoothQuant）。

| Model                       | FP32 | FP16 | BF16 | FP8  | W8A8 SQ | W8A16 | W4A16 | W4A16 AWQ | W4A16 GPTQ |
| :-------------------------- | :--: | :--: | :--: | :--: | :-----: | :---: | :---: | :-------: | :--------: |
| Baichuan                    | Y    | Y    | Y    | .    | .       | Y     | Y     | .         | .          |
| BERT                        | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| BLOOM                       | Y    | Y    | Y    | .    | Y       | Y     | Y     | .         | .          |
| ChatGLM                     | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| ChatGLM-v2                  | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| Falcon                      | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| GPT                         | Y    | Y    | Y    | Y    | Y       | Y     | Y     | .         | .          |
| GPT-J                       | Y    | Y    | Y    | Y    | Y       | Y     | Y     | Y         | .          |
| GPT-NeMo                    | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| GPT-NeoX                    | Y    | Y    | Y    | .    | .       | .     | .     | .         | Y          |
| LLaMA                       | Y    | Y    | Y    | .    | Y       | Y     | Y     | Y         | Y          |
| LLaMA-v2                    | Y    | Y    | Y    | Y    | Y       | Y     | Y     | Y         | Y          |
| OPT                         | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| SantaCoder                  | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |
| StarCoder                   | Y    | Y    | Y    | .    | .       | .     | .     | .         | .          |

**怎么用这张表：** 选模型前先查它支持哪些量化。例如 **LLaMA-v2** 这一行几乎全 Y——FP8、SmoothQuant、INT8、AWQ、GPTQ 全支持，是上手 TensorRT-LLM 量化的最佳样本。而 **BERT** 只有 FP/BF 系列，因为它是 encoder、不是自回归 decode，量化收益和适配优先级都低。注意「卡支持 FP8」（第 6 节）和「模型实现支持 FP8」（本节）要**同时满足**才能跑 FP8。

## 实操：参考链接与典型工作流

**官方与文档（原文保留）：**
- 项目主页：https://github.com/NVIDIA/TensorRT-LLM
- 官方文档：https://nvidia.github.io/TensorRT-LLM/index.html
- Triton TRT-LLM 后端模型配置：https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/model_config.md
- 性能基准（perf-overview）：https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md
- 性能优化最佳实践：https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html

**典型工作流（三步）：**

```
① 转权重 / 量化   HF 权重 ──▶ TRT-LLM checkpoint（这一步决定量化模式，如 fp8/int4_awq）
② 编译引擎        trtllm-build：指定目标卡、TP/PP、max_batch、max_input_len、max_output_len
③ 起服务          Triton + tensorrtllm_backend（生产）  或  Python/C++ runtime（自研）
```

每一步的「为什么」回看：① 见第 5 节量化选型；② 见第 2 节为何要固定 shape 包络；③ 见 [[llm-inference/README]] 的服务化全景。性能调优（开 in-flight batching、调 KV-Cache 显存比例、选并行度）以「性能优化最佳实践」链接为准。

## 常见问题 / 坑

| 现象 | 根因 | 处理 |
| :-- | :-- | :-- |
| 配 FP8 报错/回退 | 卡不是 SM89/SM90（如 A100） | 查第 6 节矩阵，A100 改用 W8A8 SQ 或 W4A16 |
| 引擎换卡跑不了 | engine 与 GPU 架构强绑定（选核固定） | 在目标卡上重新 `trtllm-build` |
| 请求 input/output 超长报错 | 超过 build 时的 `max_input_len/max_output_len` | 重新 build 提高上限，或截断请求 |
| 升级 TRT-LLM 后旧引擎失效 | 引擎格式与版本绑定 | 用新版本重新 build，不要跨版本复用 .engine |
| 量化后精度掉太多 | 选了 weight-only INT4 无校准 | 改用 AWQ/GPTQ（带校准）或 W8A8 SmoothQuant |
| 某模型某量化跑不通 | 该模型未实现该量化路径 | 查第 7 节模型×量化矩阵，换支持的组合 |
| 吞吐上不去/GPU 闲 | 未开 in-flight batching 或 batch 上限太小 | Triton 后端开连续批，调大 max_batch_size |
| 多卡 TP 卡住 | NCCL/网络拓扑问题 | 检查 [[ai-infra/网络/InfiniBand]] 与集合通信配置 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 推理同类：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/分离式推理架构]] · [[llm-inference/大模型推理张量并行]]
- 优化原理：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 量化压缩：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 模型结构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 硬件与网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 框架对照：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 训练/对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 评测/估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
