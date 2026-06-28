# 大模型量化（Quantization）总览

> 用低比特整数/浮点表示权重与激活，换取显存、带宽、算力的数倍节省——量化是 LLM 落地推理的"第一性价比"压缩手段。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]] · [[llm-inference/vllm/README]]

## 阅读地图

| 小节 | 你会得到什么 | 适合谁 |
|------|--------------|--------|
| 0. 一句话锚点 | 量化到底在干什么 | 所有人 |
| 1. 地基 | 定点表示、scale/zero-point、对称/非对称 | 入门 |
| 2. 为什么能量化 | 权重冗余 + 推理是带宽瓶颈 | 想懂"为什么有效" |
| 3. PTQ vs QAT | 两条主线的取舍 | 选型 |
| 4. 离群值之战 | LLM.int8 / SmoothQuant / AWQ / GPTQ 的核心矛盾 | 看懂方法图谱 |
| 5. 方法图谱 | 原文 10+ 篇论文归位 | 查阅 |
| 6. 硬件真相 | 为什么"INT4×FP16 不一定更快" | 避坑 |
| 实操 | 量化工具与命令清单 | 动手 |
| 常见坑 | 表格速查 | 调试 |

## 0. 一句话锚点

量化 = 把一个高精度张量 $X$（FP16/FP32）用一个**低比特整数** $X_q$ 加一个**缩放因子** $s$（可能还有零点 $z$）来近似：

$$X \approx s \cdot (X_q - z),\qquad X_q = \mathrm{clip}\Big(\mathrm{round}(X/s) + z,\ q_{min},\ q_{max}\Big)$$

FP16 一个数占 2 字节，INT8 占 1 字节，INT4 占 0.5 字节。把 70B 模型从 FP16（140 GB）压到 INT4（约 35 GB），**单卡就能放下**——这就是量化对 LLM 的核心价值。

## 1. 地基：定点表示与三个旋钮

```
   原始浮点轴 (FP16)                量化后整数格点 (INT8, 256 个台阶)
  ───┬────┬────┬────┬───►        ───●──●──●──●──●──●──●──●───►
   -2.3      0      4.1            -128 ...   0   ...   127
        连续、无限精度                 离散、只有 256 个值
        ↑ round 到最近格点，误差 ≤ s/2 ↑
```

三个旋钮决定一种量化方案：

| 旋钮 | 选项 | 影响 |
|------|------|------|
| **比特数** | INT8 / INT4 / FP8 / INT3 | 越低越省，误差越大 |
| **对称 vs 非对称** | 对称：$z=0$；非对称：$z\neq 0$ | 权重常对称；激活（含 ReLU 后非负）常非对称 |
| **粒度** | per-tensor / per-channel / per-group(group_size=128) | 越细越准，元数据开销越大 |

**对称 INT8 的 scale**（per-tensor 为例）：
$$s = \frac{\max(|X|)}{127}$$
若某权重张量绝对值最大是 $0.81$，则 $s = 0.81/127 \approx 0.00638$，权重 $0.42$ 量化为 $\mathrm{round}(0.42/0.00638)=66$，反量化回 $66\times0.00638=0.4211$，误差 $0.0011$。

> 关键直觉：**离群值（outlier）会撑大 max，从而撑大 $s$，让其余正常值的台阶变粗、误差变大**。这正是第 4 节所有方法要解决的根本矛盾。

## 2. 为什么 LLM 能被量化、且应该被量化

两条独立的理由叠加：

```
理由一：权重有冗余                理由二：推理是"内存带宽瓶颈"
┌──────────────────┐            decode 阶段每生成 1 token，
│ FP16 的 11 位尾数 │            都要把【全部权重】从 HBM 读进
│ 远超神经网络对    │            计算单元一次：
│ 精度的真实需求    │              time ≈ 模型字节数 / HBM带宽
└──────────────────┘            权重砍半 → 读取砍半 → decode 提速
        ↓                                 ↓
  改 INT8/INT4 精度损失很小        带宽减半 ≈ 吞吐近翻倍
```

decode 阶段算术强度极低（一次矩阵×向量），GPU 算力用不满，瓶颈是**把权重搬进来**。所以 weight-only 量化（只量权重、激活仍 FP16）即使要在 kernel 里反量化回 FP16 再算，**整体仍然变快**——因为省下的是搬运时间。参见 [[llm-optimizer/kv-cache]]、[[llm-inference/PD分离]]。

## 3. 两条主线：PTQ 与 QAT

> 原文要点：在深度神经网络上应用量化策略有两种常见方法。

- **训练后量化（PTQ, Post Training Quantization）**：先把模型训练至收敛，再降低权重精度。量化操作相比训练**代价小得多**，只需少量校准数据（calibration set）统计激活分布即可。LLM 量化绝大多数走这条路。
- **量化感知训练（QAT, Quantization Aware Training）**：在预训练或进一步微调期间就**模拟量化**（前向插入伪量化节点，反向用 STE 直通梯度）。QAT 能获得**更好的性能**，但需要**额外的计算资源**，还需要**有代表性的训练数据**。

```
PTQ:  [预训练完的FP16模型] → 校准(几百条样本统计分布) → 量化权重 → 直接部署
       便宜、快、无需标注数据，几小时搞定

QAT:  [模型] → 插入伪量化 → 重新训练/微调(感知量化误差) → 导出低比特
       贵、慢、要数据，但精度更接近原模型；极低比特(≤4bit)时优势明显
```

| 维度 | PTQ | QAT |
|------|-----|-----|
| 需要训练数据 | 少量校准（无标签） | 代表性训练集 |
| 算力成本 | 低（分钟~小时） | 高（一轮训练） |
| 精度 | 8bit 几乎无损；4bit 有损 | 同比特下更优 |
| 代表方法 | GPTQ / AWQ / SmoothQuant / LLM.int8 | LLM-QAT |

## 4. 离群值之战：LLM 量化方法的核心矛盾

LLM 激活值里存在**系统性离群值**（少数维度数值极大，可达正常值的百倍）。直接 per-tensor INT8 量化激活，会因离群值撑大 scale 而精度崩溃。四类主流方法是对这个矛盾的不同回答：

```
                       激活有离群值，权重相对平滑
                                 │
      ┌──────────────┬───────────┴───────────┬──────────────┐
   LLM.int8()     SmoothQuant              AWQ            GPTQ
   "隔离"         "迁移"                  "保护"         "补偿"
      │              │                      │              │
 离群值那几列      把激活的尺度             识别1%重要        逐列量化，用
 单独走FP16        数学等价地搬到          权重通道，         二阶Hessian信息
 其余走INT8        权重上，让两边          按激活幅度          补偿已量化列
 (混合精度)        都好量化               缩放保护            带来的误差
```

- **LLM.int8()**（Tim Dettmers）：把含离群值的少数特征维度**拆出来用 FP16 算**，其余 99%+ 走 INT8，再相加。零精度损失，但混合精度有额外开销。
- **SmoothQuant**：通过等价变换 $\hat X = X\,\mathrm{diag}(s)^{-1}$、$\hat W = \mathrm{diag}(s)\,W$，把激活的"难量化"按通道**迁移**一部分到权重，使两者都变得平滑、可做 W8A8。已集成在 FasterTransformer。
- **AWQ**（Activation-aware Weight Quantization）：观察到**不是所有权重同等重要**，按对应激活幅度找出约 1% 的"重要通道"并按比例放大保护，4bit weight-only 几乎无损。
- **GPTQ**：逐列贪心量化权重，每量化一列就用 Hessian 逆信息**更新未量化列来补偿误差**，是 4bit weight-only 的事实标准。详见 [[llm-compression/quantization/GPTQ]]。

## 5. 方法图谱（原文论文归位）

### Post Training Quantization (PTQ)

| 方法 | 一句话 | 资源 |
|------|--------|------|
| **ZeroQuant** | 大规模 Transformer 的高效低成本 PTQ，**集成在 DeepSpeed** | deepspeed.ai/tutorials/model-compression |
| **SmoothQuant** | 激活→权重难度迁移，W8A8；集成在 FasterTransformer | github.com/mit-han-lab/smoothquant |
| **GPTQ** | 二阶误差补偿的 4bit weight-only 标准 | 见 [[llm-compression/quantization/GPTQ]] |
| **AWQ** | 激活感知，保护 1% 重要权重通道 | github.com/mit-han-lab/llm-awq |
| **LLM.int8()** | 离群值隔离的混合精度 INT8（Tim Dettmers） | zhuanlan.zhihu.com/p/586406082 |
| **OWQ** | 从激活离群值学到的权重量化经验 | github.com/xvyaward/owq |
| **SpQR** | 稀疏-量化表示，近无损权重压缩（Tim Dettmers） | arxiv.org/pdf/2306.03078.pdf |
| **RPTQ** | 重排序的 PTQ | github.com/hahnyuan/RPTQ4LLM |
| **OliVe** | 离群值-牺牲者成对量化 | — |
| **Outlier Suppression+** | 离群值抑制改进 | github.com/wimh966/outlier_suppression |

### Quantization Aware Training (QAT)

| 方法 | 一句话 | 资源 |
|------|--------|------|
| **LLM-QAT** | Data-Free QAT，用模型自生成数据做 QAT，无需原始训练集 | github.com/facebookresearch/LLM-QAT |

## 6. 硬件真相：理论最优 ≠ 实际更快

> 原文原话（务必记住）：**由于 GPU 内核对某些类型的矩阵乘法（例如 INT4 × FP16）缺乏支持，并非以上所有方法都会加速实际的推理过程。** 理论上的最优量化策略与实际在硬件内核上的表现存在客观差距。

```
你以为：                          实际：
权重 INT4 → 直接 INT4×FP16 算       GPU 没有 INT4×FP16 的 Tensor Core 指令
→ 又省显存又提速                     ↓
                                  kernel 里要先把 INT4 反量化(dequant)回 FP16
                                  → 再用 FP16×FP16 算
                                  → 省了"搬运"(显存/带宽)，但多了"反量化"开销
```

所以要分清两类收益：
- **省显存/带宽**：几乎所有量化都能拿到（存储变小是确定的）。
- **省算力（更快的 GEMM）**：只有当硬件**原生支持该数据类型的矩阵乘**时才拿得到，例如 INT8×INT8（SmoothQuant W8A8 走得通）、FP8×FP8（Hopper/Ada 原生支持，见 [[llm-compression/quantization/fp8]]）。

选型口诀：**weight-only 低比特（GPTQ/AWQ）主要赚带宽，适合 decode 显存吃紧；W8A8/FP8 才同时赚算力，适合 prefill 计算密集。** 配合 [[llm-inference/PD分离]] 与 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] 理解 prefill/decode 差异。

## 实操：量化工具与命令清单（原文真料）

> 以下链接与定位均来自原文，是当前主流的可用工具入口。

**通用量化框架**

- vLLM 官方量化器 **llm-compressor**：https://github.com/vllm-project/llm-compressor/ —— 产出可直接被 vLLM 加载的 GPTQ/AWQ/FP8/INT8 权重，是 [[llm-inference/vllm/README]] 生态的首选。
- **NVIDIA TensorRT-Model-Optimizer**
  - 方法选择指南：https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_choosing_quant_methods.html
  - LLM PTQ 示例：https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_ptq

**算法官方实现**

- SmoothQuant：https://github.com/mit-han-lab/smoothquant （已集成进 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)）
- AWQ：https://github.com/mit-han-lab/llm-awq
- LLM-QAT：https://github.com/facebookresearch/LLM-QAT
- OWQ：https://github.com/xvyaward/owq
- RPTQ：https://github.com/hahnyuan/RPTQ4LLM
- Outlier Suppression+：https://github.com/wimh966/outlier_suppression
- ZeroQuant（DeepSpeed 模型压缩教程）：https://www.deepspeed.ai/tutorials/model-compression/

**TensorRT QAT 工具链与入门资料**

- NVIDIA TensorRT TF Quantization Toolkit（QAT）：https://docs.nvidia.com/deeplearning/tensorrt/tensorflow-quantization-toolkit/docs/docs/qat.html
- Awesome-LLM-Compression（论文/工具大全）：https://github.com/HuangOwen/Awesome-LLM-Compression
- HF bitsandbytes 8bit 矩阵乘简介（中文）：https://huggingface.co/blog/zh/hf-bitsandbytes-integration
- Weight Quantization 入门（8-bit）：https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c
- 闲话模型压缩之量化篇：https://blog.csdn.net/jinzhuojun/article/details/106955059
- 当下常用大型 transformer 效率优化方案：https://zhuanlan.zhihu.com/p/604118644
- Lilian Weng 推理优化：https://lilianweng.github.io/posts/2023-01-10-inference-optimization/
- Neural Network Weight Quantization 入门：https://www.analyticsvidhya.com/blog/2025/01/neural-network-weight-quantization/

**最小心智模型（伪流程）**

```
PTQ weight-only (GPTQ/AWQ) 典型流程：
1) 加载 FP16 模型 + 一小份校准集(128~512 条文本)
2) 前向跑校准集，统计每层激活分布 / Hessian
3) 逐层量化权重(group_size=128, 4bit, 对称)
4) 导出量化权重 + 每组的 scale
5) 用 vLLM / TensorRT 加载，kernel 内反量化推理
```

## 常见问题 / 坑

| 现象 | 根因 | 对策 |
|------|------|------|
| INT8 量化后精度大跌 | 激活离群值撑大 scale | 用 LLM.int8 隔离 / SmoothQuant 迁移 |
| 4bit weight-only 没变快反而变慢 | GPU 无 INT4×FP16 原生 GEMM，反量化有开销且 batch 大时算力瓶颈 | 仅在 decode/小 batch 用；大 batch 考虑 W8A8/FP8 |
| per-tensor 精度不够 | 粒度太粗，一个 scale 管整张量 | 改 per-channel 或 per-group(128) |
| 校准集选不好，部分任务掉点 | 校准分布与真实分布不匹配 | 用贴近目标域的代表性样本做校准 |
| 量化模型显存省了但吞吐没提升 | prefill 是计算密集，weight-only 只省带宽不省算力 | 该阶段上 FP8/INT8 W8A8 |
| QAT 训不动 / 不收敛 | 伪量化梯度问题或学习率过大 | STE + 较小学习率，从 FP16 权重 warm start |
| 4bit 后激活也想量化 | weight-only 不够省时 | 走 W4A8 / W8A8，但需硬件与 kernel 支持 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 压缩家族：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 推理落地：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 推理优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 模型结构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 硬件/算力：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 网络：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 训练/对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 评测/估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
