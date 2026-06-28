# 量化工具对比

> 一句话定位：把"量化算法(GPTQ/AWQ/SmoothQuant…)"落到能跑的代码上，靠的是一批量化**工具/库**——它们负责"读模型→喂校准数据→跑量化算法→存成某种权重格式→交给推理引擎加载"。本文横向对比 AutoGPTQ / GPTQModel / AutoAWQ / llm-compressor / TensorRT-Model-Optimizer / bitsandbytes 六大主流工具，讲清各自的算法、硬件、导出格式与选型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-compression/quantization/GPTQ]] [[llm-compression/llm-compressor/README]] [[llm-compression/gptqmodel/README]]

## 阅读地图

| 你想知道 | 跳到 |
| --- | --- |
| 量化工具到底是干嘛的、和"算法"啥区别 | §0 锚点 |
| 一次量化的通用五步骤（读→校准→量化→打包→落盘） | §1 地基 |
| 六个工具的总览对照大表 | §2 总览 |
| AutoGPTQ：祖师爷与它的退场 | §3 |
| GPTQModel：AutoGPTQ 的精神继承者 | §4 |
| AutoAWQ：激活感知路线的代表 | §5 |
| llm-compressor：vLLM 官方一站式压缩栈 | §6 |
| TensorRT-Model-Optimizer：NVIDIA 全家桶 | §7 |
| bitsandbytes：训练/QLoRA 友好的"即开即用" | §8 |
| 三大维度对比：算法 / 硬件 / 导出格式 | §9 |
| 一次调用全流程 ASCII（以 GPTQModel/llm-compressor 为例） | §10 流程 |
| 我该选哪个？决策树 | §11 选型 |
| 常见坑与误解 | §12 FAQ |

## 0. 一句话锚点

- **量化算法 ≠ 量化工具。** 算法（GPTQ、AWQ、SmoothQuant、RTN）是"怎么把 FP16 变成 INT4 且少掉精度"的**数学规则**；工具是"把这套规则工程化、能在真实大模型上一键跑完并存盘"的**软件**。
- 一个工具通常 = **算法实现 + 校准数据管线 + 权重打包格式 + 推理 kernel（或对接外部引擎）**。
- 选工具的本质三问：**(1) 它支持我要的算法吗？(2) 它产出的格式我的推理引擎(vLLM/TRT-LLM/transformers)能加载吗？(3) 它支持我的硬件吗（NVIDIA / AMD / CPU / 昇腾）？**

## 1. 地基：任何量化工具的"通用五步"

不管哪个工具，PTQ（训练后量化）的骨架几乎一样。把它拆到最原子：

```
              ┌────────────────────────────────────────────────────┐
              │              量化工具的通用五步骤                     │
              └────────────────────────────────────────────────────┘
  ① 读模型        ② 喂校准数据       ③ 跑量化算法        ④ 打包权重      ⑤ 落盘/导出
 ┌─────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────┐   ┌──────────┐
 │FP16 权重 │ →  │ 几十~几百条 │ →  │ GPTQ/AWQ/RTN │ →  │ 4bit 压包 │ → │ safetensor│
 │(HF/本地) │    │ 文本前向传播│    │ 求 scale/zero│    │+scale/zp │   │ +量化config│
 └─────────┘    └────────────┘    └──────────────┘    └──────────┘   └──────────┘
   只读权重       记录每层激活        逐层最优重建        INT4 不是直接      推理引擎按
   不改图         的分布(校准)         /误差补偿         存的,要按 kernel    格式反量化加载
                                                       约定 bit-pack
```

逐步说"为什么"：

- **① 读模型**：通常直接吃 HuggingFace `transformers` 的模型对象。工具要知道哪些层是 `Linear`（可量化）、哪些必须保留 FP16（如 `lm_head`、`LayerNorm`、router）。
- **② 校准（calibration）**：除了纯 RTN，GPTQ/AWQ/SmoothQuant 都需要**少量真实文本**做前向，统计每层**激活的分布**——因为"哪些权重重要"取决于它乘的激活有多大。校准集一般几十到几百条、各几百~2048 token。**校准数据的分布要贴近部署场景**（代码模型用代码、中文模型用中文）。
- **③ 跑算法**：核心数学发生在这里。RTN 直接就近取整；GPTQ 用 Hessian 做逐列误差补偿（见 [[llm-compression/quantization/GPTQ]]）；AWQ 先按激活幅度给每个通道找缩放再 RTN。
- **④ 打包（pack）**：INT4 不能一个数占一个字节存——那不省内存。要把 8 个 4-bit 数 **bit-pack 进一个 int32**，并和 scale / zero-point 一起存。**不同工具/不同 kernel 的 pack 布局不同**，这正是"格式不兼容"的根因。
- **⑤ 落盘**：存成 `safetensors` + 一份量化元信息（bits、group_size、对称与否、算法名、哪些层被跳过）。推理时引擎读这份元信息，用对应的反量化 kernel 还原。

> 记住这张图，后面六个工具的差异，几乎都可以定位到"它在第③步用什么算法、第④步用什么格式、第⑤步交给谁加载"。

## 2. 六大工具总览对照表

> 下表是"机制级"的稳定认知。**精确的版本号、CLI 默认值请以各项目官方 README / 源码为准**——这类细节迭代很快，本文不写死。

| 工具 | 主算法 | 定位 | 产出格式 | 主要硬件 | 谁来推理 |
| --- | --- | --- | --- | --- | --- |
| **AutoGPTQ** | GPTQ(W4/W3/W8) | 早期 GPTQ 事实标准，现已**基本停更/归档** | GPTQ-pack safetensors | NVIDIA(主) | transformers / vLLM / TGI |
| **GPTQModel** | GPTQ + 多种变体 | AutoGPTQ 的**活跃继任者**，ModelCloud 维护 | 兼容 GPTQ 格式（含 marlin 等 kernel） | NVIDIA / 部分 CPU / 多后端 | transformers / vLLM |
| **AutoAWQ** | AWQ(激活感知 W4) | AWQ 的事实实现 | AWQ-pack safetensors | NVIDIA(主) | vLLM / transformers |
| **llm-compressor** | GPTQ / AWQ / SmoothQuant / RTN / W8A8-INT8 / FP8 / 2:4 稀疏 | vLLM 官方**一站式压缩栈** | **compressed-tensors** 格式 | NVIDIA(主) | **vLLM 原生** |
| **TensorRT-Model-Optimizer** | PTQ(INT8/INT4/FP8/NVFP4) + QAT + 稀疏 + 蒸馏 | NVIDIA 官方**部署全家桶** | 量化 ckpt → 编译进 **TensorRT-LLM 引擎** | **NVIDIA 专属**（含最新卡 FP8/FP4） | TensorRT-LLM |
| **bitsandbytes** | LLM.int8() / NF4 / FP4（RTN 类，**on-the-fly**） | 训练/微调友好，**QLoRA 标配** | 不单独落盘量化权重，**加载时即时量化** | NVIDIA(主)，CPU/多后端推进中 | transformers / PEFT |

一句话区分三条技术路线：

```
路线 A：权重重建型(需校准, 高压缩高精度)   →  AutoGPTQ / GPTQModel / AutoAWQ / llm-compressor(GPTQ,AWQ)
路线 B：即时 RTN 型(免校准, 上手快)         →  bitsandbytes(NF4/int8)
路线 C：引擎一体型(算法→编译→部署绑死引擎)  →  TensorRT-Model-Optimizer(→TRT-LLM) / llm-compressor(→vLLM)
```

## 3. AutoGPTQ —— GPTQ 的"祖师爷"，但已退场

- **它是什么**：最早把 GPTQ 论文工程化、能在 HF 模型上一键跑 W4/W3 量化并被 `transformers` 直接加载的库。曾经是社区 GPTQ 量化模型的事实生产工具，HuggingFace 上海量 `-GPTQ` 模型由它产出。
- **架构核心模块**：
  - `quantize()`：逐层抓激活 → 算 Hessian → GPTQ 逐列补偿（算法细节见 [[llm-compression/quantization/GPTQ]]）。
  - `QuantLinear`：量化后的线性层，封装了 bit-pack 的权重 + scale/zero + 反量化 kernel（早期 cuda/triton kernel，后来引入 exllama / marlin 加速 kernel）。
- **关键参数**：`bits`（4/3/8）、`group_size`（如 128，越小越精越占空间）、`desc_act`（是否按激活重要性排序量化列，开了更准但更慢、且对某些 kernel 不友好）、`sym`（对称量化）。
- **现状**：项目**已基本停止维护 / 归档**，对新模型结构、新硬件、新 kernel 跟进乏力。**新项目不建议再用**，迁移到 GPTQModel。

## 4. GPTQModel —— AutoGPTQ 的活跃继任者

- **它是什么**：由 ModelCloud 主导、**接棒 AutoGPTQ** 的活跃项目（见 [[llm-compression/gptqmodel/README]]）。API 与 AutoGPTQ 高度相似（迁移成本低），但在三方面明显更强。
- **它解决什么**（相对 AutoGPTQ 的痛点）：
  1. **新模型支持快**：持续跟进 Llama/Qwen/DeepSeek/MoE 等新架构。
  2. **更快的 kernel**：集成 Marlin / 优化 triton kernel，推理吞吐显著优于老 cuda kernel。
  3. **格式互通**：产出仍是社区通用的 GPTQ-pack 格式，`transformers` / vLLM 可直接加载；并支持与 AutoAWQ 等格式互转/兼容。
- **关键参数**：与 AutoGPTQ 同源——`bits / group_size / desc_act / sym`，外加可选的不同 kernel/format 后端选择。
- **定位**：**今天要做纯 GPTQ-W4 权重量化、且想要最好维护与最快 kernel，首选 GPTQModel。**

```
AutoGPTQ (归档)  ──传承API/格式──►  GPTQModel (活跃, 新kernel+新模型)
        │                                  │
        └────── 都产出 GPTQ-pack 格式 ──────┘  ← transformers / vLLM 直接加载
```

## 5. AutoAWQ —— 激活感知量化的代表

- **它是什么**：AWQ（Activation-aware Weight Quantization）算法的事实实现。
- **AWQ 与 GPTQ 的本质区别**（一句话）：GPTQ 靠 Hessian 做**逐列误差补偿**；AWQ 不补偿，而是**先给每个权重通道乘一个缩放系数**——让"乘了大激活"的重要通道在量化时占更宽的动态范围，再做普通 RTN。它不需要反传，**校准更轻、量化更快**，对很多模型精度与 GPTQ 相当。
- **核心模块**：搜索每层的 per-channel scale（用一小批校准激活，使量化后输出误差最小）→ 应用 scale → RTN 量化 → pack 成 AWQ 格式。
- **关键参数**：`w_bit`（一般 4）、`q_group_size`（如 128）、`zero_point`（是否非对称）。
- **现状**：AutoAWQ 主仓维护趋缓，**其能力正被 llm-compressor 吸收**（llm-compressor 内置 AWQ modifier），新流程可优先考虑用 llm-compressor 跑 AWQ。
- **适用**：追求**量化速度快 + W4 精度好 + vLLM 部署**的场景。

## 6. llm-compressor —— vLLM 官方一站式压缩栈

- **它是什么**：vLLM 生态官方的模型压缩库（见 [[llm-compression/llm-compressor/README]]），把多种算法统一到**一个框架、一套配方（recipe）**里，产出 **compressed-tensors** 格式，被 **vLLM 原生高效加载**。
- **解决什么**：以前 GPTQ 用 AutoGPTQ、AWQ 用 AutoAWQ、INT8 又用别的库，格式各异、和推理引擎对接零散。llm-compressor 用统一抽象（`Modifier` + `recipe`）把这些算法整合，**一个 API 切换算法**，且天然对齐 vLLM。
- **核心抽象**：
  - **Modifier**：一个"压缩动作"，如 `GPTQModifier` / `AWQModifier` / `SmoothQuantModifier` / `QuantizationModifier`（RTN/FP8）/ 稀疏 modifier。
  - **Recipe**：把若干 modifier 按顺序组合的配方（例如先 SmoothQuant 再 GPTQ，实现 W8A8）。
  - **compressed-tensors**：统一的权重保存格式，记录量化方案、被压缩张量、稀疏掩码等，vLLM 直读。
- **覆盖最广的方案矩阵**：**W4A16（GPTQ/AWQ）、W8A8-INT8（SmoothQuant+GPTQ）、W8A8-FP8、KV-cache 量化、2:4 结构化稀疏**——这是它相对单算法库最大的优势。
- **定位**：**目标推理引擎是 vLLM、且想在一个工具里覆盖从 W4 到 FP8 到稀疏的全部需求 → 首选 llm-compressor。**

## 7. TensorRT-Model-Optimizer —— NVIDIA 的部署全家桶

- **它是什么**：NVIDIA 官方的模型优化库（常简称 ModelOpt / TRT-ModelOpt），覆盖**量化(PTQ/QAT) + 稀疏 + 蒸馏 + 投机解码**，产出的量化模型最终**编译进 TensorRT-LLM 引擎**部署。
- **解决什么**：要榨干 **NVIDIA 硬件**（尤其 Hopper/Blackwell 上的 **FP8 / NVFP4** 张量核），并走 TRT-LLM 极致推理路径时，ModelOpt 是官方且最贴硬件的工具。
- **核心能力**：
  - **PTQ**：INT8 / INT4(AWQ) / **FP8** / **NVFP4（4-bit 浮点，新一代卡）**。
  - **QAT**：量化感知训练，挽回 PTQ 在低 bit 下的精度损失。
  - **导出**：量化后的 checkpoint → TensorRT-LLM 构建引擎（`.engine`），获得 kernel 融合、In-flight batching 等极致优化。
- **代价**：**强绑定 NVIDIA + TRT-LLM**，工具链相对重，可移植性弱（产物不是通用 safetensors，而是引擎/特定 ckpt）。
- **定位**：**生产环境就是 NVIDIA 数据中心卡、要用 FP8/FP4、走 TRT-LLM → 选 ModelOpt。**

## 8. bitsandbytes —— 训练/QLoRA 友好的"即开即用"

- **它是什么**：一个 CUDA 量化算子库，提供 **LLM.int8()**（8-bit 推理）与 **NF4 / FP4**（4-bit）量化，深度集成进 `transformers` 与 `PEFT`。
- **最大不同点**：**它是"即时量化"，不产出独立的量化权重文件。** 你照常加载 FP16 模型，传 `load_in_4bit=True` / `load_in_8bit=True`，它在**加载时把权重 RTN 量化进显存**，反量化在前向时由 kernel 临时完成。
- **NF4（4-bit NormalFloat）**：QLoRA 的核心——一种为"近似正态分布的权重"设计的非均匀 4-bit 编码，配合 double quantization（连 scale 都再量化一次）进一步省显存。
- **关键参数**：`load_in_4bit / load_in_8bit`、`bnb_4bit_quant_type`（`nf4` / `fp4`）、`bnb_4bit_use_double_quant`、`bnb_4bit_compute_dtype`（计算用 bf16）。
- **优缺点**：
  - 优点：**零校准、一行参数即开**；**唯一原生支持"4-bit 基座 + LoRA 微调（QLoRA）"** 的主流方案；适合显存吃紧的微调/实验。
  - 缺点：纯 RTN，**W4 推理精度通常不如 GPTQ/AWQ**；推理吞吐一般不如专门的 W4 kernel（marlin 等）。**它强在训练，不强在极致推理。**
- **定位**：**要在消费级卡上微调大模型（QLoRA）或快速跑个 8-bit demo → bitsandbytes。要部署高吞吐推理 → 用 GPTQModel/AutoAWQ/llm-compressor。**

## 9. 三大维度横向对比

### 9.1 按"支持算法"

| 算法 | AutoGPTQ | GPTQModel | AutoAWQ | llm-compressor | TRT-ModelOpt | bitsandbytes |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| RTN(就近取整) | ○ | ○ | ○(底层) | ✓ | ✓ | ✓(即时) |
| GPTQ(Hessian 补偿) | ✓ | ✓✓ | ✗ | ✓ | ✗ | ✗ |
| AWQ(激活感知) | ✗ | △ | ✓ | ✓ | ✓ | ✗ |
| SmoothQuant | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| W8A8-INT8 | ✗ | △ | ✗ | ✓ | ✓ | ✓(int8 推理) |
| FP8 / NVFP4 | ✗ | ✗ | ✗ | ✓(FP8) | ✓✓(FP8/FP4) | ✗ |
| 2:4 稀疏 | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| QAT(量化感知训练) | ✗ | ✗ | ✗ | △ | ✓ | △(QLoRA) |

> ✓✓=该工具的强项/最佳实现，✓=支持，△=部分/借助外部，○=底层用到，✗=不支持。**具体能力以官方文档为准。**

### 9.2 按"硬件"

```
            消费级 N卡   数据中心 N卡(Hopper/Blackwell)   AMD ROCm   CPU      昇腾/其它
AutoGPTQ      ✓               ✓(无FP8专长)               △(社区)    △        ✗
GPTQModel     ✓               ✓                          △          ✓(部分)   △(推进)
AutoAWQ       ✓               ✓                          △          ✗        ✗
llm-compressor✓               ✓(FP8)                     △(随vLLM)  △        △(随vLLM)
TRT-ModelOpt  △               ✓✓(FP8/FP4 专属)           ✗          ✗        ✗
bitsandbytes  ✓✓(QLoRA)       ✓                          △(推进)    △(推进)   ✗
```

要点：**TRT-ModelOpt 是唯一能吃满最新 N 卡 FP8/FP4 张量核的；bitsandbytes 在消费级卡微调最舒服；其余 W4 工具消费级/数据中心 N 卡都 OK，非 N 卡支持普遍偏弱。**

### 9.3 按"导出格式 / 谁来加载"

```
工具                产出格式                      推理加载方
─────────────────────────────────────────────────────────────
AutoGPTQ      →   GPTQ-pack safetensors    →   transformers / vLLM / TGI
GPTQModel     →   GPTQ-pack(+marlin)       →   transformers / vLLM
AutoAWQ       →   AWQ-pack safetensors     →   vLLM / transformers
llm-compressor→   compressed-tensors       →   vLLM(原生最佳)
TRT-ModelOpt  →   量化ckpt→TRT-LLM .engine  →   TensorRT-LLM(绑死)
bitsandbytes  →   (无独立文件, 即时量化)     →   transformers / PEFT
```

**格式即站队**：选 vLLM 部署 → GPTQ/AWQ/compressed-tensors 都行，llm-compressor 最顺；选 TRT-LLM 极致性能 → 必须走 ModelOpt；只是想加载现成 `-GPTQ`/`-AWQ` HF 模型 → transformers 直接读。

## 10. 一次量化的完整调用流（以 GPTQ 路线为例）

下面这张图对 GPTQModel / llm-compressor(GPTQModifier) 的内部流程都成立：

```
用户配置(bits=4, group_size=128, calib=512条)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  for 每个 Transformer 层 (逐层, 省显存):                    │
│   ┌──────────────────────────────────────────────────┐   │
│   │ 1) 注入 hook, 喂校准 batch 前向 → 抓该层输入激活 X   │   │
│   │ 2) 累积 Hessian  H = X·Xᵀ  (GPTQ 的二阶信息)        │   │
│   │ 3) 对该层每个 Linear:                               │   │
│   │      逐列量化 → 算量化误差 → 用 H 把误差补偿到未量化列 │   │
│   │      (group_size 决定多少列共享一组 scale)          │   │
│   │ 4) 得到 INT4 权重 + per-group scale/zero            │   │
│   │ 5) bit-pack: 8 个 int4 → 1 个 int32                 │   │
│   └──────────────────────────────────────────────────┘   │
│   该层量化完, 释放激活, 进入下一层                          │
└──────────────────────────────────────────────────────────┘
        │
        ▼
保存: safetensors(packed权重+scale) + 量化config(bits/group_size/algo/skipped层)
        │
        ▼
推理引擎(vLLM/transformers)读config → 选对应反量化kernel(marlin等) → 上线
```

数值直觉（为什么 W4 省内存）：一个 7B 模型 FP16 约 $7\times10^9 \times 2\text{B} \approx 14\,\text{GB}$；W4 后权重位宽降到 4-bit，约 $14\,\text{GB} \times \tfrac{4}{16} = 3.5\,\text{GB}$，再加 scale/zero 的少量开销，实测约 **4 GB 出头**——这就是 W4 让 7B 跑进消费级显卡的原因。group_size=128 意味着每 128 个权重共享一组 scale/zero，scale 开销约占 $\tfrac{16\text{bit}}{128}\approx 0.125$ bit/权重，可忽略；group_size 越小越精但 scale 开销越大。

## 11. 选型决策树

```
开始
 │
 ├─ 我要"微调/QLoRA"(不是纯推理)？
 │     └─ 是 ──────────────────────────► bitsandbytes (NF4 + LoRA)
 │
 ├─ 部署引擎是 TensorRT-LLM、且要 FP8/FP4 榨干新 N 卡？
 │     └─ 是 ──────────────────────────► TensorRT-Model-Optimizer
 │
 ├─ 部署引擎是 vLLM，且想"一个工具覆盖 W4/W8A8/FP8/稀疏"？
 │     └─ 是 ──────────────────────────► llm-compressor
 │
 ├─ 只要纯 GPTQ-W4 权重量化、要最好维护与最快 kernel？
 │     └─ 是 ──────────────────────────► GPTQModel (别再用 AutoGPTQ)
 │
 ├─ 要 AWQ 路线(量化快、校准轻)？
 │     └─ 是 ──────► AutoAWQ 或 llm-compressor(AWQModifier)
 │
 └─ 只是想加载别人量化好的 -GPTQ/-AWQ 模型？
       └─ 直接用 transformers / vLLM 加载, 无需量化工具
```

一句话经验：

- **生产 + vLLM** → llm-compressor（最对齐、方案最全）。
- **生产 + TRT-LLM + 最新 N 卡** → TensorRT-Model-Optimizer。
- **纯 GPTQ W4 + 通用** → GPTQModel。
- **训练/微调/省显存实验** → bitsandbytes。
- **AutoGPTQ / AutoAWQ** → 视为"历史/被吸收"，新项目优先用上面的继任者。

## 12. 常见问题（FAQ）

| 问题 | 解答 |
| --- | --- |
| GPTQ 和 AWQ 哪个精度高？ | 二者在 W4 上普遍接近，模型/任务不同各有胜负；AWQ 量化更快、校准更轻，GPTQ 在极低 bit(W3) 上常更稳。**别迷信单一榜单，按你的模型实测。** |
| 为什么我的 `-GPTQ` 模型在 A 引擎能跑、B 引擎报错？ | pack 布局/kernel 约定不同。换工具产出的格式要确认目标引擎支持的 kernel（如 marlin 要满足 group_size、对称等约束）。 |
| `desc_act=True` 要不要开？ | 开了按激活重要性排序量化、精度略升，但更慢，且某些快 kernel(如部分 marlin 路径)不支持。**追求速度可关，追求精度可开后实测。** |
| 校准数据要多少、用什么？ | 一般几十到几百条、各几百~2048 token；**分布要贴近部署场景**（中文模型别只用英文维基）。太少欠拟合统计、太多收益递减还慢。 |
| bitsandbytes 量化后能不能存成 safetensors 部署？ | 它是即时量化，不产出标准 W4 部署权重；要高吞吐部署请改用 GPTQModel/AWQ/llm-compressor 重新量化。 |
| FP8 和 INT4 怎么选？ | FP8 精度损失极小、对新 N 卡有硬件加速，适合**精度敏感 + 有 Hopper/Blackwell**；INT4 压得更狠、省内存更多，适合**显存极度受限**。FP8 走 llm-compressor/ModelOpt，INT4 走 GPTQ/AWQ 系。 |
| 这些工具的具体 API/默认值我能照抄本文吗？ | **机制可信，精确签名/默认值请以官方文档与源码为准**——它们迭代很快，本文刻意不写死版本号与 CLI 默认。 |

## 🔗 跳转链接

- [[00-知识地图]] —— 全局导航
- [[llm-compression/quantization/GPTQ]] —— GPTQ 算法的最底层推导（Hessian / 逐列补偿）
- [[llm-compression/llm-compressor/README]] —— llm-compressor 一站式压缩栈详解
- [[llm-compression/gptqmodel/README]] —— GPTQModel（AutoGPTQ 继任者）详解
- 相关：[[llm-compression/quantization/量化基础]]、[[llm-compression/quantization/SmoothQuant]]、[[llm-compression/quantization/大模型量化概述]]、[[llm-compression/quantization/fp8]]
