# TensorRT Model Optimizer

> NVIDIA 官方的「模型压缩工具箱」：把训练好的大模型做量化 / 稀疏 / 蒸馏 / 投机解码，再无缝导出到 TensorRT-LLM 或 TensorRT 上跑出极致推理性能。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-compression/quantization/量化基础]] [[llm-inference/DeepSpeed-Inference]]

## 阅读地图

| 节 | 你会学到 | 关键词 |
| --- | --- | --- |
| 0 | 一句话锚点 | ModelOpt = 压缩前端 + TRT-LLM 后端 |
| 1 | 地基/前置 | 为什么要量化、PTQ vs QAT、FP8/INT4 |
| 2 | 它解决什么问题 | 训练框架到推理引擎的「压缩断层」 |
| 3 | 整体架构 | 模型 → 量化/稀疏/蒸馏 → 导出 → TRT-LLM |
| 4 | 核心机制：量化（PTQ/QAT） | 校准、伪量化、FP8/INT4/AWQ |
| 5 | 核心机制：稀疏化 | 2:4 结构化稀疏、Sparse Tensor Core |
| 6 | 核心机制：蒸馏 + 投机解码 | KD、Medusa/EAGLE |
| 7 | 导出与 TRT-LLM 协作 | 统一 checkpoint、unified export |
| 8 | 端到端完整流程 | 一条流水线串起来 |
| 9 | 与同类对比 / 何时用 / 权衡 | bitsandbytes、AutoAWQ、llm-compressor |
| 数值例子 | INT8/FP8/INT4 手算压缩比与误差 | 位宽、scale、显存 |
| FAQ | 常见坑 | 精度掉点、导出失败 |

- 代码：https://github.com/NVIDIA/TensorRT-Model-Optimizer
- 文档：https://nvidia.github.io/TensorRT-Model-Optimizer/
- 量化方法最佳实践：https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_choosing_quant_methods.html

---

## 0. 一句话锚点

**TensorRT Model Optimizer（简称 ModelOpt）** 是一个 Python 库，它**不负责推理本身**，而是负责「**把模型变小变快的那一步**」——量化（quantization）、稀疏（sparsity）、蒸馏（distillation）、投机解码（speculative decoding）。处理完后，它产出一个可以被 **TensorRT-LLM** 或 **TensorRT** 直接吃下去的、带量化信息的 checkpoint。

记忆口诀：

```
       [ 训练好的 FP16/BF16 模型 ]
                 │
                 ▼
   ┌──────────────────────────────┐
   │   TensorRT Model Optimizer   │  ← 压缩前端（本文主角）
   │  量化 / 稀疏 / 蒸馏 / 投机解码  │
   └──────────────────────────────┘
                 │  导出统一 checkpoint
                 ▼
   ┌──────────────────────────────┐
   │  TensorRT-LLM / TensorRT     │  ← 推理后端（实际跑模型）
   └──────────────────────────────┘
                 │
                 ▼
         [ 低延迟 / 高吞吐推理 ]
```

一句话：**ModelOpt 管「压」，TRT-LLM 管「跑」**。

---

## 1. 地基 / 前置：为什么需要量化与压缩

在看工具之前，先把最底层的概念拆到原子。

### 1.1 模型为什么大、为什么慢

一个权重张量在内存里是一堆浮点数。常见数据类型：

| 类型 | 位宽 | 一个数占字节 | 说明 |
| --- | --- | --- | --- |
| FP32 | 32 bit | 4 B | 训练默认精度 |
| FP16 / BF16 | 16 bit | 2 B | 推理常用半精度 |
| FP8 (E4M3) | 8 bit | 1 B | Hopper/Ada 原生支持 |
| INT8 | 8 bit | 1 B | 经典定点量化 |
| INT4 | 4 bit | 0.5 B | 权重量化极限常用值 |

一个 7B（70 亿参数）模型，FP16 下显存占用 ≈ $7\times10^9 \times 2\,\text{B} = 14\,\text{GB}$。换成 INT4，权重部分 ≈ $7\times10^9 \times 0.5\,\text{B} = 3.5\,\text{GB}$。**显存直接降到约 1/4**。

推理慢的两大瓶颈：**访存瓶颈（memory-bound，自回归每出一个 token 都要把全部权重从显存搬一遍，权重越小搬运越快）** 与 **算力瓶颈（compute-bound，低位宽 INT8/FP8 能用专用 Tensor Core，吞吐是 FP16 的数倍）**。所以「量化」一箭双雕：**省显存 + 提速**。

### 1.2 量化的本质：把浮点映射到整数（或低位浮点）

整数对称量化的核心公式。设权重 $w$ 的最大绝对值为 $\alpha$（量化范围），目标位宽 $b$：

$$
s = \frac{\alpha}{2^{b-1}-1}, \qquad q = \text{round}\!\left(\frac{w}{s}\right), \qquad \hat{w} = q \cdot s
$$

- $s$ 叫 **scale（缩放因子）**，是连接「浮点世界」和「整数世界」的桥。
- $q$ 是存下来的整数。
- $\hat{w}$ 是反量化（dequantize）后近似还原的浮点，与原始 $w$ 之间的差就是 **量化误差**。

直观图（INT4 把连续浮点切成 16 个台阶）：

```
浮点 w  ─────────────────────────►
        -α                       +α
         │   实际是连续值          │
         ▼                        ▼
  量化 q: -7 -6 .. -1  0  1 .. 6  7   ← 只剩 16 个离散台阶 (4bit)
         └──┬──┘            └──┬──┘
         台阶宽度 = s（量化粒度）
```

**关键洞察**：量化误差来自「把连续值塞进离散台阶」。台阶越少（位宽越低）误差越大；范围 $\alpha$ 选得越准，误差越小。怎么选 $\alpha$ 就是各种量化算法（AWQ、SmoothQuant 等）拼的地方。详见 [[llm-compression/quantization/量化基础]]。

### 1.3 PTQ vs QAT（两条路线，必背）

```
              ┌─────────── PTQ (Post-Training Quantization) ───────────┐
训练好的模型 ─┤  喂少量校准数据 → 统计激活范围 → 直接定 scale          │  快、无需训练、可能掉点
              └────────────────────────────────────────────────────────┘

              ┌─────────── QAT (Quantization-Aware Training) ──────────┐
训练好的模型 ─┤  插入「伪量化」节点 → 继续微调几步 → 让权重适应量化误差 │  慢、需数据与算力、精度更高
              └────────────────────────────────────────────────────────┘
```

- **PTQ（训练后量化）**：不改训练，只拿几百条样本「校准（calibration）」，统计激活值分布来确定每层的 scale。几分钟到几十分钟搞定。ModelOpt 的主战场。
- **QAT（量化感知训练）**：在前向里插入「伪量化（fake quantization）」算子，让梯度反向传播时模型「知道」自己会被量化，从而学到对量化更鲁棒的权重。需要训练循环和标注/无标注数据。

---

## 2. 它解决什么问题：训练框架与推理引擎之间的「压缩断层」

没有 ModelOpt 之前，工程师面对的困境：

```
PyTorch/HF 训练好模型
   │
   │  想上 TensorRT-LLM 加速，但 TRT-LLM 需要量化后的权重
   ▼
 ？？？  ← 断层：量化算法散落在各处，格式各不相同
   │       AWQ 一套格式、SmoothQuant 一套、FP8 又一套
   ▼
 手动转换、调试 scale、对齐 layer 命名……极易出错
```

**ModelOpt 把这条断层补上**：它提供统一的量化/稀疏 API（一两行 `mtq.quantize`），并保证产出的 checkpoint 能被 TRT-LLM **一键加载**。等于在「PyTorch 训练」与「TensorRT-LLM 部署」之间架了一座官方的、对齐过的桥。

它要解决的三类核心需求：
1. **降显存**：让大模型塞进更小的卡 / 让单卡塞更多并发。
2. **提速降本**：用 FP8/INT4/INT8 + 稀疏吃满 Tensor Core。
3. **保精度**：用先进算法（AWQ、SmoothQuant、QAT、KD）把掉点压到可接受范围。

---

## 3. 整体架构

```
┌───────────────────────────────────────────────────────────────────┐
│                  TensorRT Model Optimizer (Python)                  │
│                                                                     │
│  输入：PyTorch / Hugging Face / Megatron / NeMo 模型                 │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ 量化     │  │ 稀疏化   │  │ 蒸馏     │  │ 投机解码          │    │
│  │ mtq.*    │  │ mts.*    │  │ mtd.*    │  │ (Medusa/EAGLE)   │    │
│  │ PTQ/QAT  │  │ 2:4 等   │  │ KD       │  │                  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘    │
│       └─────────────┴─────────────┴───────────────-─┘              │
│                            │                                        │
│                   统一中间表示 (带 scale / 稀疏掩码的模型)             │
│                            │                                        │
│                      mto.save / 导出器                              │
└────────────────────────────┼──────────────────────────────────────┘
                             │ Unified / 量化 checkpoint
            ┌────────────────┴─────────────────┐
            ▼                                  ▼
   ┌─────────────────┐                ┌──────────────────┐
   │  TensorRT-LLM   │                │   TensorRT       │
   │  (LLM 推理)     │                │  (CNN/视觉/通用)  │
   └─────────────────┘                └──────────────────┘
```

四大功能模块（API 命名以官方文档为准，常见前缀）：
- **`modelopt.torch.quantization` (mtq)**：量化。
- **`modelopt.torch.sparsity` (mts)**：稀疏。
- **`modelopt.torch.distill` (mtd)**：蒸馏。
- **`modelopt.torch.opt` (mto)**：保存/恢复模型压缩状态。
- 投机解码（speculative decoding）相关模块用于训练 Medusa/EAGLE 头。

> 具体导入名与子模块以官方文档为准，上面给的是结构记忆。

---

## 4. 核心机制一：量化（PTQ / QAT）

这是 ModelOpt 用得最多的能力。

### 4.1 PTQ 的内部数据流

```
                          ┌── 前向跑校准数据 ──┐
  原模型 (FP16)           │                    │
     │   mtq.quantize(    │   每层观察激活      │
     │     model,         ▼   值的分布范围 α    │
     │     config,    ┌───────────────────┐    │
     │     forward_loop)│ 插入 Quantizer    │    │
     ▼   ─────────────►│ 节点(伪量化)       │◄───┘
  量化后模型            │ 统计 amax → scale  │
     │                 └───────────────────┘
     ▼
  权重/激活带上 scale，可导出
```

关键步骤拆解：
1. **插桩**：`mtq.quantize` 把每个 Linear/MatMul 包上「Quantizer」节点。
2. **校准（calibration）**：用户提供一个 `forward_loop`（把几百条样本喂进模型跑前向），ModelOpt 在此过程中统计每层激活的最大绝对值 `amax`。
3. **定 scale**：根据 `amax` 和位宽算出每层（或每通道、每 block）的 scale。
4. **完成**：模型此时处于「伪量化」状态，可直接评估精度，也可导出。

### 4.2 ModelOpt 支持的量化「配方」

| 配方 | 权重 | 激活 | 适用硬件 | 特点 |
| --- | --- | --- | --- | --- |
| FP8 | FP8 | FP8 | Hopper(H100)/Ada/Blackwell | 掉点极小，首选 |
| INT8 SmoothQuant | INT8 | INT8 | 通用 | 平滑激活离群值再量化 |
| INT4 AWQ | INT4 | FP16 | 通用 | 权重 4bit、激活半精度，省显存 |
| INT4 (W4A8) | INT4 | FP8/INT8 | 新架构 | 更激进 |
| NVFP4 | FP4 | FP4 | Blackwell | 4bit 浮点，新一代 |

> 具体配方名以 `mtq.*_CFG` 官方常量为准。

**两个必懂算法（解决「INT4 掉点」）**：

- **SmoothQuant**：激活里常有「离群值（outlier）」，几个通道数值特别大，导致 scale 被撑大、其他值精度变差。SmoothQuant 把激活的「难度」按公式 $\hat{X} = X / s,\ \hat{W} = W \cdot s$ 迁移一部分到权重上（权重好量化），让激活和权重都变得平滑、易量化。

- **AWQ（Activation-aware Weight Quantization）**：观察发现「重要权重」由激活幅度决定。AWQ 给重要通道乘一个保护性缩放，量化时这些通道误差更小，从而在 INT4 下保住精度。

### 4.3 QAT：让模型「适应」量化

```
   插入伪量化节点的模型 ──► for batch: 前向(带伪量化) → loss.backward()(STE 直通) → step()
                          几百~几千步微调 ──► 对量化更鲁棒的权重 → 精度逼近原始
```

伪量化在前向里做 `round`，但 `round` 不可导。QAT 用 **STE（Straight-Through Estimator，直通估计）**：反向时把 `round` 的梯度近似为 1，直接传过去，使训练可进行。ModelOpt 用 `mtq.quantize` 完成插桩后，接你自己的训练循环即可做 QAT。

---

## 5. 核心机制二：稀疏化（Sparsity）

稀疏 = 把一部分权重直接置 0，跳过这些乘法。

NVIDIA Ampere 起的 Tensor Core 支持 **2:4 结构化稀疏**：每连续 4 个权重里强制 2 个为 0。硬件能识别这种规则结构，吃下后矩阵乘吞吐近似翻倍。

```
稠密权重 (4个一组):   [ 0.8  -0.3   0.5   0.1 ]
                         │     │      │     │
       保留幅度最大的 2 个，其余清零
                         ▼
2:4 稀疏:             [ 0.8   0     0.5   0  ]
                       └─保留─┘     └保留┘
   → Sparse Tensor Core 只算非零，吞吐 ≈ ×2
```

ModelOpt 提供 `mts.sparsify`（命名以官方为准）做 2:4 稀疏，通常配合少量微调恢复精度，再与量化叠加，进一步压缩。

---

## 6. 核心机制三：蒸馏 + 投机解码

### 6.1 知识蒸馏（Knowledge Distillation, KD）

```
   [ Teacher 大模型(高精度,慢) ] ──软标签/logits──► [ Student 小模型(小,快) ]
        学生学习老师的「软概率分布」，比只学硬标签学到更多「暗知识」
```

蒸馏损失常用 KL 散度：$L_{KD} = T^2 \cdot \text{KL}\big(\sigma(z_t/T)\,\|\,\sigma(z_s/T)\big)$，其中 $T$ 是温度，$\sigma$ 是 softmax，$z_t/z_s$ 是师/生 logits。ModelOpt 的 `mtd` 模块把蒸馏封装成可插拔接口，常用于「量化后掉点 → 用原模型当老师蒸回来」。

### 6.2 投机解码（Speculative Decoding）

自回归生成一次只出一个 token，慢。投机解码让一个**小的草稿模型/草稿头**一次猜多个 token，再由大模型一次性并行验证：

```
草稿头一次提议: t1 t2 t3 t4 (便宜) → 大模型并行验证: ✔ ✔ ✘ → 接受 t1 t2，从 t3 重来
              一步验证多 token → 总步数变少 → 更快
```

ModelOpt 支持训练 **Medusa**、**EAGLE** 等草稿头，产出的模型同样可导出给 TRT-LLM 加速。

---

## 7. 导出与 TRT-LLM 协作

这是 ModelOpt 存在的「闭环」所在。

```
ModelOpt 处理完的模型
   │  导出 unified / 量化 checkpoint
   ▼
[ 统一 checkpoint：权重 + 每层 scale + 量化配置 + (可选)稀疏掩码 + 草稿头 ]
   │  TRT-LLM 读取 → build
   ▼
[ TRT-LLM build：算子融合 / 选 FP8·INT4 kernel / In-flight batching / KV cache 量化 → .engine ]
   │
   ▼
[ trtllm-serve / triton 部署，对外提供推理 ]
```

协作要点：
- **格式对齐**：ModelOpt 导出的 checkpoint 字段（scale、量化方式、layer 命名）与 TRT-LLM 的加载器一一对应，免去手动转换。
- **统一导出（unified export）**：新版倾向产出一个与 Hugging Face 结构兼容、又带量化信息的目录，TRT-LLM、vLLM、SGLang 等生态可读（具体兼容范围以官方文档为准）。
- **职责分离**：ModelOpt 决定「用什么精度、scale 多少」；TRT-LLM 决定「用哪个 CUDA kernel、怎么排批」。两者解耦但对齐。

对比 [[llm-inference/DeepSpeed-Inference]]：DeepSpeed-Inference 是「推理引擎 + 张量并行」一体方案，自己也做核融合；而 ModelOpt 只做压缩前端，把推理交给 TRT-LLM。前者偏「训练框架自带推理」，后者偏「专用推理引擎 + 专用压缩工具」的 NVIDIA 全家桶组合。

---

## 8. 端到端完整流程（把上面串起来）

以「FP16 的 LLM → INT4/FP8 → TRT-LLM 部署」为例，伪代码（API 以官方文档为准）：

```python
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

# 1. 加载原始模型 (HF/PyTorch)
model = load_my_model()

# 2. 定义校准前向循环
def forward_loop(model):
    for batch in calib_dataloader:   # 几百条样本即可
        model(batch)

# 3. PTQ 量化（选 FP8 或 INT4_AWQ 配方）
model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

# 4. (可选) 评估精度，掉点大就上 QAT 或换配方
# 5. (可选) QAT：接训练循环再微调若干步
# 6. 保存压缩状态
mto.save(model, "model_opt.pth")

# 7. 导出统一 checkpoint 给 TRT-LLM
#    export_tensorrt_llm_checkpoint(...) 或 unified export
```

流程总览图：

```
[加载模型] → [选配方 FP8/INT4] → [校准/PTQ] → [评估]
                          掉点大? ─是─► [QAT/换配方/蒸馏] ─┐
                          │ 否                            │
                          ▼ ◄──────────────────────────────┘
                  [导出 checkpoint] → [TRT-LLM build → .engine] → [部署服务]
```

---

## 9. 与同类对比 / 何时用 / 权衡

### 9.1 横向对比

| 工具 | 主要能力 | 后端 | 定位 |
| --- | --- | --- | --- |
| **ModelOpt** | 量化+稀疏+蒸馏+投机 | TRT-LLM / TRT | NVIDIA 官方全栈压缩 |
| bitsandbytes | INT8/NF4 量化 | PyTorch/HF | 训练/推理即插即用 |
| AutoAWQ / AutoGPTQ | INT4 权重量化 | vLLM/HF | 社区轻量量化 |
| llm-compressor (vLLM) | 量化+稀疏 | vLLM | vLLM 生态压缩前端 |
| TensorRT(原生) | 引擎构建+部分量化 | TRT | 通用推理引擎 |

ModelOpt 的差异化：**功能最全（四合一）+ 与 TRT-LLM 深度对齐 + 支持最新硬件特性（FP8/NVFP4/2:4 稀疏）**。代价是绑定 NVIDIA 生态。

### 9.2 何时用 / 不用

- **该用**：目标硬件是 NVIDIA 数据中心卡（H100/H200/L40S/Blackwell），要上 TRT-LLM，追求极致延迟/吞吐；需要 FP8/INT4 且想少掉点。
- **不必用**：只在 CPU / 非 NVIDIA 硬件部署；只想快速验证、用 bitsandbytes 几行就够；后端是 vLLM 且已用 llm-compressor。

### 9.3 权衡

| 收益 | 代价 |
| --- | --- |
| 显存大幅下降 | 可能掉点（需校准/QAT 找回） |
| 推理提速降本 | 绑定 NVIDIA + TRT-LLM 生态 |
| 一站式（量化+稀疏+蒸馏） | 学习成本、配方选择需经验 |
| 支持最新硬件特性 | 老卡（无 FP8 Tensor Core）收益有限 |

---

## 数值例子 / 对照（手算）

**例：7B 模型，各精度下的权重显存与单台阶误差**

权重数量 $N = 7\times10^9$。显存 = $N \times$ 每数字节数：

| 精度 | 每数字节 | 显存 | 相对 FP16 |
| --- | --- | --- | --- |
| FP16 | 2 B | $7\text{e}9\times2 = 14\,\text{GB}$ | 1.0× |
| FP8 | 1 B | $7\text{e}9\times1 = 7\,\text{GB}$ | 0.5× |
| INT8 | 1 B | $7\,\text{GB}$ | 0.5× |
| INT4 | 0.5 B | $7\text{e}9\times0.5 = 3.5\,\text{GB}$ | 0.25× |

**量化台阶误差手算**：设某层权重范围 $\alpha = 4.0$。

- INT8（$b=8$）：$s = \dfrac{4.0}{2^{7}-1} = \dfrac{4.0}{127} \approx 0.0315$。台阶宽 0.0315，最大舍入误差 $\le s/2 \approx 0.0157$。
- INT4（$b=4$）：$s = \dfrac{4.0}{2^{3}-1} = \dfrac{4.0}{7} \approx 0.571$。台阶宽 0.571，最大舍入误差 $\le 0.286$。

**结论**：INT4 的台阶宽是 INT8 的约 18 倍（$0.571/0.0315$），误差也成比例放大——这正是为何 INT4 需要 AWQ/SmoothQuant/QAT 这类「保精度」技术，而 INT8/FP8 往往直接 PTQ 就够。

**单个数量化演示**：$w = 0.9$，INT4，$s=0.571$：
$$
q = \text{round}(0.9 / 0.571) = \text{round}(1.576) = 2,\quad \hat{w} = 2 \times 0.571 = 1.142
$$
误差 $|\hat w - w| = |1.142 - 0.9| = 0.242$，确实在台阶半宽 0.286 内。

---

## 常见问题

| 问题 | 解答 |
| --- | --- |
| ModelOpt 自己能跑推理吗？ | 不能。它只做压缩，推理交给 TRT-LLM / TensorRT。 |
| PTQ 掉点太多怎么办？ | 换更高位宽配方（INT4→FP8）、用 AWQ/SmoothQuant、上 QAT、或蒸馏找回。 |
| 校准数据要多少？ | 通常几百条代表性样本即可，重在分布贴近真实流量。 |
| FP8 在所有卡都能用吗？ | 不能。需 Hopper/Ada/Blackwell 等带 FP8 Tensor Core 的架构；老卡用 INT8/INT4。 |
| 量化后必须导出给 TRT-LLM 吗？ | 不必。也可在 PyTorch 内用伪量化评估；但要实打实加速需导出到引擎。 |
| 2:4 稀疏一定提速吗？ | 需 Ampere 及以后的 Sparse Tensor Core，且要恢复微调避免掉点。 |
| 和 vLLM 能配合吗？ | 新版统一导出对部分生态兼容，具体以官方文档为准。 |
| 导出失败/字段对不上？ | 多为版本不匹配；保持 ModelOpt 与 TRT-LLM 版本对齐，以官方文档为准。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-compression/quantization/量化基础]] — 量化数学原理、对称/非对称、per-channel 等基础
- [[llm-inference/DeepSpeed-Inference]] — 另一类「引擎自带推理 + 压缩」方案，可对照本文 NVIDIA 全家桶思路
- 官方代码：https://github.com/NVIDIA/TensorRT-Model-Optimizer
- 官方文档：https://nvidia.github.io/TensorRT-Model-Optimizer/
- 量化方法最佳实践：https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_choosing_quant_methods.html
