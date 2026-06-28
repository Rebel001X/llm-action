# 分布式并行训练（Distribution Parallelism）

> 把一个装不下、算不动的大模型，按"数据/张量/层/专家"四个维度切开，分摊到成百上千张卡上协同训练。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/InfiniBand]]

## 阅读地图

| 你想知道 | 看哪一节 |
|---|---|
| 一句话先建立直觉 | [§0 锚点](#0-一句话锚点) |
| 为什么单卡装不下、为什么必须并行 | [§1 地基](#1-地基为什么必须并行) |
| 数据并行 DP 到底切什么、怎么同步 | [§2 数据并行](#2-数据并行-dp切样本) |
| ZeRO 怎么把显存进一步压下来 | [§3 ZeRO](#3-zero把-dp-的冗余显存切掉) |
| 张量并行 TP 怎么切一个矩阵乘 | [§4 张量并行](#4-张量并行-tp切单层矩阵) |
| 流水线并行 PP 的气泡是什么 | [§5 流水线并行](#5-流水线并行-pp切层) |
| 专家并行 EP / 序列并行 SP | [§6 EP-SP](#6-专家并行-ep--序列并行-sp) |
| 三/四维并行怎么组合（3D/4D） | [§7 多维并行](#7-多维并行3d4d-组合) |
| 自动并行：搜索最优切法 | [§8 自动并行](#8-自动并行从手工到搜索)（**原文论文清单在此**） |
| 原始论文清单 + 参考链接 | [实操/论文清单](#论文清单与参考链接原文真料) |
| 容易踩的坑 | [常见坑](#常见问题与坑) |

## 0. 一句话锚点

- **核心问题**：模型参数、激活、优化器状态加起来远超单卡显存（80GB），而且算力也不够 → 必须把工作量切给多卡。
- **切的本质**：所有并行方式都只是**张量切分维度不同**而已（原文 *Exploring Hidden Dimensions* 的洞见）。
  - 切 batch 维 = 数据并行；切权重维 = 张量并行；切层 = 流水线并行；切专家 = 专家并行。
- **三个轴**：省**显存** vs 省**算力** vs 省**通信**，三者互相博弈，没有银弹。

## 1. 地基：为什么必须并行

### 1.1 单卡装不下：训练显存账

训练一个参数量 $\Phi$ 的模型，用 Adam + FP16 混合精度，显存占用大致为：

$$M_{\text{model}} = \underbrace{2\Phi}_{\text{fp16 权重}} + \underbrace{2\Phi}_{\text{fp16 梯度}} + \underbrace{4\Phi + 4\Phi + 4\Phi}_{\text{Adam: fp32 权重/一阶/二阶矩}} = 16\Phi \text{ 字节}$$

**数值手算**：7B 模型 → $16 \times 7\times10^9 = 112\,\text{GB}$，单张 80GB A100 直接 OOM，**还没算激活值**。175B → $2.8\,\text{TB}$。这就是为什么必须并行。详见 [[docs/transformer内存估算]]。

### 1.2 算不动：算力账

一次前向 + 反向的浮点运算量约 $6\Phi$ FLOPs/token（前向 $2\Phi$、反向 $4\Phi$）。训 1T token 的 7B 模型 ≈ $6\times7\times10^9\times10^{12}=4.2\times10^{22}$ FLOPs。单卡 A100 BF16 算力约 $312$ TFLOPS，理想下也要 $4.2\times10^{22}/(3.12\times10^{14}) \approx 1.3\times10^8\,\text{s}$ ≈ **4.3 年**。必须用千卡级集群把时间压到天级。

### 1.3 前置：集合通信原语

并行的代价是**通信**。三条命脉原语（详见 [[ai-infra/网络/集合通信原语]]）：

```
AllReduce :  每卡一份数据 → 求和 → 结果广播回每卡   (DP 同步梯度)
              g0  g1  g2  g3  ──►  (g0+g1+g2+g3) 复制到每卡
AllGather :  每卡一片 → 拼成完整副本到每卡          (ZeRO 取回权重)
ReduceScatter: 求和后每卡只留自己负责的那一片      (ZeRO 同步梯度)
关系：AllReduce = ReduceScatter + AllGather
```

带宽决定一切：卡内 NVLink 数百 GB/s，跨机走 [[ai-infra/网络/InfiniBand]] 才几十~几百 GB/s → **通信量大的并行方式要尽量留在机内**。

## 2. 数据并行 DP：切样本

**思想**：每张卡保存**完整模型副本**，把一个 batch 切成 N 份，各算各的，反向后用 AllReduce 同步梯度，保证各卡权重一致。

```
       global batch = 32
   ┌──────┬──────┬──────┬──────┐
   │ b0..7│ b8..15│b16..23│b24..31│   ← 切 batch 维
   ▼      ▼       ▼       ▼
 [GPU0] [GPU1]  [GPU2]  [GPU3]      每卡一份完整权重 W
   │      │       │       │
 反向 g0  g1      g2      g3
   └──────┴───AllReduce───┴──────┐
              ḡ = (g0+g1+g2+g3)/4   每卡用同一个平均梯度更新
```

- **为什么有效**：等价于用 4 倍 batch 单卡训练，吞吐近线性扩展。
- **致命缺点**：每卡都存完整 $16\Phi$，**显存完全没省**，所以纯 DP 只能训"单卡装得下"的模型。这正是 ZeRO 要解决的。
- 历史渊源：原文 *One weird trick* 早就指出——**卷积层数据大、参数小 → 适合数据并行；全连接层参数大、数据小 → 适合模型并行**。这就是"按层选并行方式"的最早直觉。

## 3. ZeRO：把 DP 的冗余显存切掉

DP 里 N 张卡存了 N 份一模一样的优化器状态/梯度/权重，纯属浪费。**ZeRO（Zero Redundancy Optimizer，DeepSpeed 提出）** 把这三样沿 DP 维度切开，每卡只存 $1/N$：

| 阶段 | 切谁 | 每卡显存 | 额外通信 |
|---|---|---|---|
| ZeRO-1 | 优化器状态 | $4\Phi + 12\Phi/N$ | 同 DP |
| ZeRO-2 | + 梯度 | $2\Phi + 14\Phi/N$ | 同 DP（ReduceScatter 替 AllReduce） |
| ZeRO-3 | + 参数 | $16\Phi/N$ | 多一次参数 AllGather |

```
ZeRO-3 计算到第 L 层时：
  AllGather → 临时拼出第 L 层完整权重 → 算完立刻丢弃 → 下一层再 AllGather
  显存换通信：N 越大每卡越省，但通信次数线性增加
```

详见 [[ai-framework/deepspeed/README]]。本质：ZeRO 用通信换显存，是 DP 与 TP 之间的"光滑过渡"。

## 4. 张量并行 TP：切单层矩阵

**思想**：把**一层内部**的大矩阵乘法切到多卡（Megatron-LM 提出，详见 [[ai-framework/megatron-lm/README]]）。以 MLP 的 $Y = \text{GeLU}(XA)$，再 $Z = YB$ 为例：

### 4.1 列切 + 行切的精妙配合

```
A 按列切: A = [A1 | A2]      B 按行切: B = [B1]
                                          [B2]
GPU0:  Y1 = GeLU(X·A1)   GPU1:  Y2 = GeLU(X·A2)
       Z1 = Y1·B1               Z2 = Y2·B2
                  AllReduce
              Z = Z1 + Z2   ◄── 整个 MLP 块只需 1 次 AllReduce
```

- **为什么 A 列切、B 行切**：GeLU 是逐元素非线性，$\text{GeLU}(XA)$ 必须先在每卡算出**完整的某几列** $Y_i$，列切恰好让每卡独立算；接着 B 行切让 $Y_iB_i$ 可直接相加。**前向一次 AllReduce、反向一次**，把通信压到最低。
- **代价**：通信量大（每个 token 都要 AllReduce），所以 TP 通常**只在一台机内 8 卡走 NVLink**，跨机会被带宽拖死。
- Attention 同理：按**注意力头**切分到不同卡，天然并行。与 [[llm-optimizer/FlashAttention]]、[[llm-optimizer/kv-cache]] 配合时要注意头维切分。

## 5. 流水线并行 PP：切层

**思想**：把模型按**层**纵向切成若干段（stage），每段放一台/几台机器，数据像流水线一样逐段传递。

```
stage0: 层0-7   stage1: 层8-15   stage2: 层16-23   stage3: 层24-31
 GPU0  ───激活──►  GPU1  ───►  GPU2  ───►  GPU3
                       反向梯度逆向传回
```

### 5.1 气泡（Bubble）问题与微批

朴素 PP 一次只有一个 stage 在干活，其余空转。**把 batch 切成 m 个 micro-batch 流水起来**（GPipe / 1F1B）才能填满流水线：

```
朴素（m=1）：              GPipe 流水（m=4）：
GPU0 ■░░░░░░░             GPU0 ■■■■░░░▓▓▓▓
GPU1 ░■░░░░░░             GPU1 ░■■■■░░░▓▓▓
GPU2 ░░■░░░░░             GPU2 ░░■■■■░░░▓▓
GPU3 ░░░■░░░░             GPU3 ░░░■■■■░░░▓
  ■前向 ▓反向 ░气泡(空转)
```

气泡占比 $\approx \dfrac{p-1}{m+p-1}$（$p$=stage 数，$m$=micro-batch 数）。**数值**：$p=4, m=4 \to 3/7 \approx 43\%$ 浪费；增到 $m=32 \to 3/35 \approx 8.6\%$。所以 **micro-batch 越多气泡越小**，但激活显存也越大，需权衡。1F1B 调度进一步降低激活峰值显存。

- **优点**：跨 stage 只传**激活值**（小），通信量远小于 TP → 适合**跨机**。
- **缺点**：气泡、负载均衡（每段层数/算力要均匀）。

## 6. 专家并行 EP / 序列并行 SP

### 6.1 专家并行 EP（MoE 专用）

MoE 模型（详见 [[llm-algo/moe/README]]）每层有很多专家，但每个 token 只路由到 Top-K 个。**把不同专家放到不同卡**，token 通过 **All-to-All** 通信发到对应专家所在的卡：

```
token ──Gating──► 选中 expert_3, expert_7
      All-to-All 把 token 发到 expert 所在卡 ──► 计算 ──► All-to-All 收回
```

EP 让总参数量爆炸增长而单卡算力不变，是 DeepSeek/Mixtral 等的关键。痛点：负载不均（热门专家堵车）、All-to-All 通信重。

### 6.2 序列并行 SP

把**序列长度维度**切到多卡，专门解决长上下文下 **LayerNorm/Dropout 激活值**和注意力的显存爆炸。常与 TP 配合（Megatron-SP），把 TP 没切到的那部分激活也切掉。配合 [[llm-algo/旋转编码RoPE]] 处理超长上下文。

## 7. 多维并行：3D/4D 组合

单一并行都有上限，真实千卡训练是**组合拳**。核心原则——**把通信量大的并行放在带宽高的地方**：

```
通信量：  TP/SP  >  EP  >  DP/ZeRO  >  PP
带宽：    机内NVLink ──────────────► 跨机IB
放置：    TP放机内8卡 │ PP跨机少量传激活 │ DP在最外层

典型 3D 布局（64 GPU = TP4 × PP2 × DP8）：
┌─────────── DP 组 (×8) ───────────┐
│  ┌── PP stage0 ──┐ ┌── stage1 ──┐│
│  │ TP[0 1 2 3]   │ │ TP[0 1 2 3]││  每个 TP 组在同一台机内
│  └───────────────┘ └───────────┘│
└──────────────────────────────────┘
```

实际框架（Megatron-LM）按 `tensor-model-parallel-size` / `pipeline-model-parallel-size` / 数据并行隐式三个参数配置，乘积 = 总卡数。MoE 再叠加 EP 即 4D。

## 8. 自动并行：从手工到搜索

手工配 3D 并行需要专家经验，学界一直想**自动搜出最优切法**。原文给出的这条研究脉络极有价值，把它的内在逻辑串起来：

```
洞见(切分维度等价) → 形式化(构型搜索) → 代价模型 → 模拟器 → 联合优化
   Hidden Dim         构型空间         Tofu/cost   FlexFlow     Unity
```

- **核心洞见**：数据并行、模型并行只是**张量在不同维度切分**罢了（sample/channel/width/length 都能切），找最优并行 = 在**构型（configuration）空间里搜索**最优构型 → 一个**搜索问题**。
- **代价模型**：用 cost model 衡量每个构型优劣，并对搜索空间**剪枝**。
- **执行模拟器**：FlexFlow 用 execution simulator 让 cost model 更准。
- **算子 vs 张量切分**：Tofu 关注 **operator 划分**，其它工作关注 **tensor 划分**，二者等价；但关注 tensor 更优——无需用户改 operator 实现。
- **联合优化**：Unity 把代数变换（图替代）与并行化统一表示，多级搜索同时优化计算/并行/通信。

这些思想最终下沉到 vLLM 等推理框架的并行配置中（[[llm-inference/vllm/README]]、[[llm-inference/PD分离]]）。

## 实操 / 论文清单与参考链接（原文真料）

> 以下为原文保留的奠基性论文清单与注解，是理解自动并行的第一手脉络，原样保留并补充串讲见 §8。

- **One weird trick for parallelizing convolutional neural networks**
  - 不同的层适合用不同的并行方式：卷积层数据比参数大，适合**数据并行**；全连接层参数比数据大，适合**模型并行**。

- **Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks**
  - 在抽象上更进一步：数据并行、模型并行都只是**张量切分方式的不同**，有的切数据有的切模型；对多维张量，在不同维度（sample/channel/width/length）切分效果也不同。
  - 不同的切分方式都是一种**构型（configuration）**，不同构型导致不同效果，寻找最优并行 = 在构型空间里**搜索最优构型**，问题形式化成搜索问题。
  - 引入**代价模型**衡量每个构型优劣，提出一系列对搜索空间**剪枝**的策略，实现了原型系统。

- **BEYOND DATA AND MODEL PARALLELISM FOR DEEP NEURAL NETWORKS（FlexFlow）**
  - 主要提出 **execution simulator** 来完善 cost model。

- **Supporting Very Large Models using Automatic Dataflow Graph Partitioning（Tofu）**
  - 提出一套 DSL 方便开发者描述张量划分策略，使用类似 poly 的 integer interval analysis 描述并行策略，搜索算法上也有很多特色工作。
  - Tofu 与其它工作的不同在于：它关注 **operator 的划分**，其它工作关注 **tensor 的划分**，二者等价。
  - 作者观点：关注 tensor 划分更好——不需要用户修改 operator 实现；Tofu 需在 DSL 里描述 operator 实现方式。

- **Mesh-TensorFlow: Deep Learning for Supercomputers**
  - 作者与 GShard 几乎重叠，Mesh-TensorFlow 可看作 GShard 的前身。
  - 核心理念 **beyond batch splitting**：数据并行是 batch splitting，模型并行是张量其它维度的切分。把集群加速卡抽象成 **mesh 结构**，提出把张量切分并映射到 mesh 的办法。

- **Unity: Accelerating DNN Training Through Joint Opt of Algebraic Transform and Parallelization**
  - 链接：https://zhuanlan.zhihu.com/p/560247608
  - 在 FlexFlow、TASO、MetaFlow 基础上，提出在并行计算图（PCG）中代数变换和并行化的**统一表示（OP，Operator）**与**共优化（图替代，Substitution）**方法，可同时考虑分布式训练中的计算、并行和通信过程。对共优化，Unity 使用**多级搜索算法**高效搜索性能最好的图替代组合及相应硬件放置策略。

### 参考链接（原文保留）

- 个人论文笔记：https://github.com/DicardoX/Individual_Paper_Notes
- MLSys 阅读清单：https://jeongseob.github.io/readings_mlsys.html
- 分布式方法综述：https://paperswithcode.com/methods/category/distributed-methods

## 常见问题与坑

| 坑 | 现象 | 原因 / 对策 |
|---|---|---|
| 纯 DP 还是 OOM | 加卡也装不下大模型 | DP 不省显存，每卡仍存完整 $16\Phi$ → 上 ZeRO-3 或 TP/PP |
| TP 跨机巨慢 | 多机 TP 吞吐崩 | TP 每 token 都 AllReduce，跨机带宽不够 → TP 只放**机内 8 卡 NVLink** |
| PP 利用率低 | GPU 大量空转 | 气泡占比 $\frac{p-1}{m+p-1}$ → 增大 micro-batch 数 m，或用 1F1B |
| PP 负载不均 | 某 stage 是瓶颈 | 各 stage 层数/算力不均 → 按算力而非层数均分（embedding/lm_head 单独算） |
| MoE 专家堵车 | 部分卡满载部分空 | 路由热点 → 加负载均衡 loss，限制专家容量 capacity factor |
| 全开并行反而慢 | 通信吃满 | 维度组合错位 → 遵循"通信大的并行放高带宽处"：TP机内 / PP/DP 跨机 |
| 激活显存爆炸 | 长序列 OOM | 激活随 batch×seq 增长 → 序列并行 SP + 激活重计算（gradient checkpointing） |
| 数值不一致 | 各 DP 卡权重漂移 | 梯度未正确 AllReduce / 随机种子不同步 → 检查 reduce 与 seed 一致性 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 框架实现：[[ai-framework/megatron-lm/README]]（TP/PP 标杆）· [[ai-framework/deepspeed/README]]（ZeRO）· [[ai-framework/pytorch/README]]（DDP/FSDP）
- 训练：[[llm-train/README]] · [[llm-train/peft/PEFT-API]]
- 对齐：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 模型结构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 优化算子：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 压缩量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 硬件与网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测与估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
