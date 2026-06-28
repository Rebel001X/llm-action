# 大模型压缩工具生态(LLM Compression Tools)

> 把"算法论文"变成"能在生产环境跑的权重"的那一层软件：量化 / 剪枝 / 蒸馏的开源工具盘点与选型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/README]] [[llm-compression/quantization/量化基础]] [[llm-compression/quantization/大模型量化概述]] [[llm-inference/vllm/README]]

## 阅读地图

| 你想知道的 | 跳到 |
|---|---|
| 这些工具到底解决什么问题 | [1. 地基](#1-地基为什么需要专门的压缩工具) |
| 工具在整条链路里站在哪 | [2. 工具在链路中的位置](#2-工具在整条链路里站在哪) |
| 量化工具有哪几类、怎么分 | [3. 量化工具的四大流派](#3-量化工具的四大流派) |
| llm-compressor 怎么用 | [4. llm-compressor(vLLM 官方)](#4-llm-compressorvllm-官方) |
| GPTQModel / AutoAWQ 区别 | [5. GPTQModel--autoawq--bitsandbytes](#5-gptqmodel--autoawq--bitsandbytes) |
| 厂商工具链(NVIDIA/华为/百度) | [6. 厂商工具链](#6-厂商工具链) |
| 我该选哪个 | [7. 选型决策树](#7-选型决策树) |
| 典型工作流长啥样 | [8. 典型工作流](#8-典型工作流w8a8-量化为例) |
| 常见坑 | [常见问题/坑](#常见问题坑) |

## 0. 一句话锚点

**压缩工具 = "算法实现 + 校准流水线 + 序列化格式 + 推理后端对接"四件套的打包。** 你给它一个 FP16 模型和一点校准数据，它吐出一个体积更小、精度可控、且能被某个推理引擎(vLLM / TensorRT-LLM / llama.cpp …)直接加载的权重包。选错工具，最常见的后果不是"精度差",而是"压出来的模型你的推理引擎根本不认"。

## 1. 地基：为什么需要专门的压缩工具

模型压缩的**算法**(GPTQ、AWQ、SmoothQuant、剪枝、蒸馏)本身往往只是一篇论文 + 一段参考代码。但要把它用到真实的百亿千亿模型上，你需要解决一堆"脏活"：

```
论文里的一个公式  ──────►  生产可用的压缩权重
        │
        ├── 1) 怎么遍历几百层、逐层套算法？(逐层 hook / 分块)
        ├── 2) 校准数据(calibration)从哪来、喂多少？
        ├── 3) 中间激活值显存放不下怎么办？(offload / 分块)
        ├── 4) 量化后的 scale/zero-point 怎么存？(序列化格式)
        ├── 5) 推理引擎认不认这个格式？(kernel 对接)
        └── 6) 怎么测精度没掉太多？(eval harness)
```

工具的价值就是把上面 1~6 全部封装好。**它解决的核心问题是"工程化落地"**：让你不必为每个新模型重写一遍校准与序列化代码。

> 关键名词速记(详见 [[llm-compression/quantization/量化基础]])：
> - **PTQ**(训练后量化)：拿现成模型 + 少量校准数据直接量化，几分钟到几小时，主流工具都属此类。
> - **QAT**(量化感知训练)：训练时模拟量化误差，精度更高但要重训，工具更重(见 [[llm-compression/quantization/llm-qat/README]])。
> - **W8A8 / W4A16 / W4A8**：权重(Weight)/激活(Activation)的比特数。`W4A16` = 权重 4bit、激活仍 FP16，是最常见的"省显存"配置。

## 2. 工具在整条链路里站在哪

压缩工具是**离线**步骤，处在"训练/微调完成"和"上线推理"之间，它的产物要被推理引擎消费：

```
   训练 / 微调            ┌──── 压缩工具(离线一次) ────┐         推理上线
 ┌───────────┐          │  ┌────────┐   ┌─────────┐ │      ┌──────────────┐
 │ FP16/BF16 │  权重 ──►│  │ 校准    │──►│ 量化/剪枝 │ ├──►   │ vLLM / TRT-LLM│──► 用户
 │  base 模型 │          │  │ 数据集  │   │  算法核心 │ │ 权重 │ llama.cpp ... │
 └───────────┘          │  └────────┘   └─────────┘ │ 包   └──────────────┘
                        └────────────┬───────────────┘
                                     │ 序列化成某种格式
                          (safetensors + 量化元数据 / GGUF / TRT engine)
                                     ▼
                          这个"格式"决定了下游引擎认不认
```

**最重要的一条工程铁律**：压缩工具与推理引擎是**配对**关系。`llm-compressor` 产出的 `compressed-tensors` 格式天然被 vLLM 认；GGUF 格式服务于 `llama.cpp`；TensorRT engine 只能被 TensorRT-LLM 跑。**先确定推理后端，再倒推选压缩工具**——这是选型的第一性原则。

## 3. 量化工具的四大流派

按"谁出的、对接谁"可以把当前主流量化工具分成四类：

```
              ┌─────────────────────── 量化工具生态 ──────────────────────┐
              │                                                          │
  推理引擎自带 │   厂商硬件配套      算法社区库          通用训练框架插件     │
 ┌──────────┐ │ ┌────────────┐  ┌──────────────┐   ┌────────────────┐   │
 │llm-       │ │ │TensorRT-   │  │GPTQModel     │   │bitsandbytes    │   │
 │compressor │ │ │Model-Opt   │  │(GPTQ 系)     │   │(HF 即时量化)    │   │
 │(→vLLM/    │ │ │(→TRT-LLM)  │  │AutoAWQ(AWQ系)│   │PEFT/QLoRA      │   │
 │ SGLang)   │ │ │PaddleSlim  │  │              │   │                │   │
 │           │ │ │(→Paddle)   │  │              │   │                │   │
 │           │ │ │昇腾 量化工具 │  │              │   │                │   │
 └──────────┘ │ └────────────┘  └──────────────┘   └────────────────┘   │
              └──────────────────────────────────────────────────────────┘
```

| 流派 | 代表 | 定位 | 适合谁 |
|---|---|---|---|
| 推理引擎官方 | `llm-compressor`(vLLM) | 一站式压缩→直接被 vLLM/SGLang 加载 | 用 vLLM 部署的团队(首选) |
| 厂商硬件配套 | NVIDIA TensorRT-Model-Optimizer、华为昇腾量化、`PaddleSlim` | 绑定自家硬件/引擎，精度与 kernel 调优最好 | 锁定某硬件栈 |
| 算法社区库 | `GPTQModel`、`AutoAWQ` | 专精某一算法家族，更新快、模型覆盖广 | 想要最新算法/特定模型支持 |
| 训练框架插件 | `bitsandbytes`、`PEFT`(QLoRA) | 微调时即时低比特，省显存 | 做 QLoRA 微调 |

> 注：上述工具的**确切版本号、CLI 参数名、默认 bit 配置**请以各自官方文档/仓库 README 为准，下文只讲"机制与用途",不背具体默认值。

## 4. llm-compressor(vLLM 官方)

`https://github.com/vllm-project/llm-compressor` —— vLLM 生态的官方压缩库，本仓库在 [[llm-compression/llm-compressor/README]] 有专门记录。它的设计哲学是**"配方(recipe)驱动"**：你不直接调底层算法，而是声明一段配方，描述"对哪些层、用什么算法、压到几比特"。

```
  你的 Python 脚本
      │
      │  recipe = [ SmoothQuant(...), GPTQ(scheme="W4A16"), ... ]
      ▼
 ┌────────────────── llm-compressor 引擎 ──────────────────┐
 │  oneshot(model, dataset, recipe)                        │
 │     │                                                   │
 │     ├─► 加载 HF 模型                                      │
 │     ├─► 跑校准数据，收集每层激活统计                        │
 │     ├─► 按 recipe 顺序施加 modifier(算法步骤)              │
 │     │      SmoothQuant → 平滑激活离群值                    │
 │     │      GPTQ        → 逐层最小化量化误差                 │
 │     └─► 序列化为 compressed-tensors 格式(safetensors)     │
 └─────────────────────────────────────────────────────────┘
      ▼
  save_pretrained → 一个能被 vLLM 直接 `llm = LLM(path)` 加载的目录
```

**核心概念**：
- **Modifier(修饰器)**：一个压缩"动作",如 `GPTQModifier`、`SmoothQuantModifier`、稀疏化 modifier(剪枝见 [[llm-compression/llm-compressor/剪枝]])。多个 modifier 串成 recipe 顺序执行。
- **Scheme(方案)**：用字符串声明目标精度，如 `W4A16`(权重4bit对称)、`W8A8`(权重激活都8bit)、`FP8`。本仓库 [[llm-compression/llm-compressor/量化方案]] 有展开。
- **oneshot vs train**：`oneshot` 是纯 PTQ(无梯度，快);也支持带训练的压缩(QAT 类)。
- **compressed-tensors 格式**：vLLM 原生识别的量化权重容器，把 scale、zero-point、量化后整数权重一起存进 safetensors。**这是它最大的护城河**——压完即可上线，无需格式转换。

**适合场景**：你已经用 vLLM/SGLang 部署，想要 W4A16、W8A8、FP8、或结构化稀疏，且希望"压缩→部署"零摩擦。

## 5. GPTQModel / AutoAWQ / bitsandbytes

这三个是算法社区里最常被点名的"专科"工具。

### 5.1 GPTQModel(GPTQ 家族)
- 本仓库见 [[llm-compression/gptqmodel/README]]。是早期 `AutoGPTQ` 的活跃继任者，专注 **GPTQ** 算法(逐层用近似二阶信息最小化量化误差，详见 [[llm-compression/quantization/GPTQ]])。
- 主打 **W4A16 / W3 / W2** 等低比特**仅权重**量化，省显存效果显著(4bit 约省 4x 权重显存)。
- 产物可被 vLLM、TGI、Transformers 等多家加载，模型覆盖面广、更新快。

### 5.2 AutoAWQ(AWQ 家族)
- 专注 **AWQ**(Activation-aware Weight Quantization)：通过激活分布找出"重要权重通道"并保护它们，对 4bit 权重量化精度通常很友好。
- 同样是 W4A16 路线，与 GPTQ 是"互为备选"的两条技术路线。实践中常两个都跑一遍，比 perplexity/下游精度选优。

### 5.3 bitsandbytes(即时量化)
- 不是离线产出文件，而是在 `from_pretrained(..., load_in_4bit=True)` 时**即时**把权重量化进显存，是 **QLoRA** 微调的基础设施(配合 PEFT)。
- 优点：零额外步骤、与 HF 无缝；缺点：推理吞吐通常不如专门编译的 W4 kernel。**它更偏"省显存训练",不是"高吞吐部署"。**

```
   省显存训练 ◄──────────────────────────► 高吞吐部署
  bitsandbytes      GPTQModel/AutoAWQ      llm-compressor
   (即时4bit)        (离线W4A16文件)        (→vLLM原生格式)
       │                  │                      │
   QLoRA 微调        通用部署文件            vLLM 专精流水线
```

## 6. 厂商工具链

当你锁定了某家硬件/推理栈，用厂商自家的工具往往能拿到**最好的 kernel 调优和精度**，代价是**绑定**。

| 工具 | 厂商/绑定 | 主要能力 | 仓库内参考 |
|---|---|---|---|
| TensorRT-Model-Optimizer | NVIDIA → TensorRT-LLM | PTQ/QAT、FP8/INT8/INT4、稀疏、蒸馏；选量化方法有官方指南 | [[llm-inference/tensorrt/README]] |
| PaddleSlim | 百度 → PaddlePaddle | 量化/剪枝/蒸馏/NAS 一体 | [[llm-compression/PaddleSlim/README]] |
| 昇腾量化工具 | 华为 → 昇腾 NPU | 面向 NPU 的量化与算子适配 | [[ai-infra/算力/昇腾NPU]] |

NVIDIA 的两份官方资料(原文件里就引用了)很值得读：
- 选量化方法指南：`https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_choosing_quant_methods.html`(讲什么场景选 FP8 / INT8 / INT4 / AWQ)。
- DeepSeek 压缩示例：`https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/deepseek/README.md`(MoE 大模型量化的实战配方，可对照 [[llm-compression/quantization/moe模型量化]])。

> TensorRT-Model-Optimizer 的产物通常还要再经 TensorRT-LLM 编译成 engine 才能跑，**比"压完即用"的 llm-compressor 多一步编译**，但 NVIDIA 卡上吞吐和延迟一般更优。

## 7. 选型决策树

```
                你要压缩一个 LLM
                        │
        ┌───────────────┼────────────────────┐
        ▼               ▼                     ▼
   目的是微调省显存？   目的是部署省显存/提吞吐？  绑定了某家硬件？
        │               │                     │
   bitsandbytes    ┌────┴─────┐          ┌─────┴──────┐
   + PEFT(QLoRA)   ▼          ▼          ▼            ▼
                用 vLLM?    用llama.cpp? NVIDIA卡   昇腾/Paddle
                  │          │            │            │
            llm-compressor  转 GGUF   TensorRT-     昇腾量化/
            (W4A16/W8A8/FP8)(社区工具) Model-Opt     PaddleSlim
                  │
            想要最新算法/特定模型支持？
                  │
            GPTQModel / AutoAWQ
            (产物再喂给 vLLM)
```

判断顺序固定为：**目的(训练还是部署) → 推理后端 → 硬件 → 算法**。把"算法选择"放到最后，因为它最容易换。

## 8. 典型工作流(W8A8 量化为例)

下面是用 llm-compressor 做 PTQ 的**逻辑流程**(参数名以官方为准，此处讲含义)：

```
1) 准备
   - base 模型(HF safetensors)
   - 校准数据：从目标领域/通用语料抽 N 条样本(通常几十~几百条即可)
            ▼
2) 写 recipe
   recipe = [
     SmoothQuant(平滑激活离群值, 让 W8A8 的激活更好量化),
     QuantizationModifier(scheme="W8A8", 选对称/逐通道等)
   ]
            ▼
3) oneshot(model, dataset, recipe)
   - 前向跑校准数据 → 收集每层激活的 max/分布
   - 逐层计算 scale / zero-point
   - 用整数权重替换原权重
            ▼
4) 评估(别跳过!)
   - 用 lm-eval / perplexity 对比压缩前后
   - 精度掉太多 → 回去调 recipe(换算法、加校准数据、改对称性)
            ▼
5) save → 上线
   - 保存为 compressed-tensors 目录
   - vLLM: LLM("/path/to/compressed") 直接加载
```

**校准数据怎么选**(高频问题)：
- 量太少 → scale 估计不稳，离群层精度崩。
- 分布偏离上线场景 → 实际推理精度比离线 eval 差。
- 经验：用**与上线流量同分布**的几十到几百条样本，激活类量化(W8A8)比纯权重量化(W4A16)对校准数据更敏感。

> 各算法的取舍(W4A16 省显存最狠但激活仍 FP16、W8A8 计算也加速但需校准激活、FP8 在新卡上兼顾精度与速度)详见 [[llm-compression/quantization/大模型量化概述]] 与 [[llm-compression/quantization/fp8]]。

## 常见问题/坑

| 现象 / 坑 | 根因 | 应对 |
|---|---|---|
| 压完 vLLM/TRT 加载报"未知量化格式" | 工具产出格式与推理引擎不配对 | 先定后端再选工具；llm-compressor↔vLLM、GGUF↔llama.cpp |
| 离线 eval 还行，上线精度掉 | 校准数据分布 ≠ 线上分布 | 用同分布样本校准；增大校准条数 |
| W4A16 精度可接受，换 W4A8/W8A8 就崩 | 激活含离群值(outlier),直接量化误差大 | 先 SmoothQuant 平滑激活，再量化(见 [[llm-compression/quantization/SmoothQuant]]) |
| 量化时显存 OOM | 中间激活/Hessian 统计占显存 | 开 offload、减小校准 batch、分块逐层处理 |
| MoE 模型量化精度异常 | 专家路由不均、各专家分布差异大 | 用 MoE 专用配方，参考 [[llm-compression/quantization/moe模型量化]] 与 NVIDIA DeepSeek 示例 |
| bitsandbytes 4bit 推理慢 | 即时量化非高吞吐 kernel | 部署改用 GPTQModel/AutoAWQ/llm-compressor 的离线 W4 kernel |
| 不同工具同 4bit 精度差异大 | GPTQ vs AWQ 算法路线不同 | 两条路线都跑，按下游指标选优 |
| 引用了某 CLI 参数默认值结果对不上 | 版本迭代默认值会变 | 一律以当前版本官方文档/`--help` 为准,不要背 |

## 🔗 跳转链接

- 总览：[[00-知识地图]] · [[llm-compression/README]] · [[llm-compression/大模型压缩综述]]
- 量化基础与概述：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/大模型量化概述]]
- 具体算法：[[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/SmoothQuant]] · [[llm-compression/quantization/fp8]] · [[llm-compression/quantization/moe模型量化]]
- 工具专页：[[llm-compression/llm-compressor/README]] · [[llm-compression/llm-compressor/量化方案]] · [[llm-compression/llm-compressor/剪枝]] · [[llm-compression/gptqmodel/README]] · [[llm-compression/PaddleSlim/README]]
- 推理后端：[[llm-inference/vllm/README]] · [[llm-inference/tensorrt/README]]
- 硬件：[[ai-infra/算力/昇腾NPU]]
