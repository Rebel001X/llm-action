# FlexFlow：SOAP 自动并行搜索

> FlexFlow 把"怎么切模型才最快"变成一个**成本最小化搜索问题**：在 SOAP 四维空间里用 MCMC 随机游走，靠一个比真机快三个数量级的**执行模拟器**当裁判，自动找出最优并行策略。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 节 | 内容 | 你会得到 |
|----|------|----------|
| 0 | 一句话锚点 | FlexFlow 到底解决什么问题 |
| 1 | 地基：DP/MP 为什么不够 | 理解"更多并行维度"的动机 |
| 2 | SOAP 搜索空间 | 四个并行维度各切什么 |
| 3 | 总体架构 | operator graph + device topology + optimizer |
| 4 | 执行模拟器 | 为什么能快 1000 倍（delta simulation） |
| 5 | MCMC 搜索算法 | NP-hard 空间里怎么启发式找解 |
| 6 | 程序接口与可移植性 | 设备拓扑当一等公民 |
| 7 | 数值示例 | 手算一次切分维度的乘法空间 |
| 实操 | 原论文要点 / 参考链接 | 保留原始真料 |
| 坑 | 常见问题表 | 模拟器假设何时失效 |

---

## 0. 一句话锚点

> **FlexFlow = SOAP 搜索空间 + 执行模拟器（裁判）+ MCMC 搜索（探索）。**

斯坦福团队发表于 **MLSys 2019**，原论文标题：**《Beyond Data and Model Parallelism for Deep Neural Networks》**。

核心贡献（原文四点）：

1. 相比 data-parallel 与 model-parallel，提出**更多维度**的 split 方案：**SOAP**（Sample / Operator / Attribute / Parameter）四个维度。
2. 在四维之上，提出一种在候选空间中**搜索**的方案。
3. 提出一个更轻量的 **simulator**，能更快速地对候选 split 策略做 evaluate，相比"直接执行"的方案**提升了 3 个数量级**。
4. 实现总体框架 **FlexFlow**。

实验结果：在两个 GPU 集群、六个真实 DNN 基准上，**即使把搜索时间算进去**，FlexFlow 也能把训练吞吐提升到 SOTA 的 **3.8 倍**，并改善可扩展性。

---

## 1. 地基：为什么 DP / MP 不够

手写并行策略时，工程师通常只在两个极端里二选一：

- **数据并行（DP）**：每张卡放整份模型，切 batch（样本）。通信量 = 梯度全量 AllReduce，模型大时显存撑不住。
- **模型并行（MP）**：按算子/层切模型。切法五花八门，人工很难找到通信与负载平衡的最优点。

问题在于：**这两种策略只是巨大策略空间里的两个角落**。原文一句话点破——"已有的系统都是在 SOAP 的子集中划分的"。

```
            真实最优策略可能在这片中间地带
  DP 角落 ●──────────────────────────────● MP 角落
          ╲   人工几乎从不探索的广阔空间   ╱
           ╲                            ╱
            ╲──────── SOAP 空间 ───────╱
```

> 直觉：把"切哪一维、切几份"全部放开当变量，让机器自动搜，往往能找到人想不到的混合切法。

---

## 2. SOAP：四个并行维度

FlexFlow 的切分是**针对每个 op 的 output tensor**来切的——选择 output tensor 的多个维度。原文与 MindSpore 的切分 config 描述基本一致。

| 维度 | 英文 | 切什么 | 类比已有并行 |
|------|------|--------|--------------|
| **S**ample | Sample | input 的 **batch** 维 | 数据并行 |
| **A**ttribute | Attribute | tensor 的属性维，如 **height / width** | 空间/序列切分 |
| **P**arameter | Parameter | tensor 的参数维，如 **in-channel / out-channel** | 张量并行 |
| **O**perator | Operator | **op 与 op 之间**的切分维度 | 流水/模型并行 |

```
   一个 op 的 output tensor（以卷积为例）
   ┌───────────────────────────────────┐
   │  [ N (Sample) , C (Parameter) ,    │
   │    H (Attribute), W (Attribute) ]  │
   └───────────────────────────────────┘
        ▲          ▲           ▲
     切 batch    切通道      切空间
     =数据并行  =张量并行   =空间并行
   而 Operator 维 = 不同 op 分到不同设备（横向跨 op）
```

> 关键观察（原文）：虽然把 tensor 分成多个维度，**实际上 S/A/P 都属于 tensor 本身的维度**；只有 Operator 维是 op 之间的切分。这一点与 MindSpore 一致。
>
> 价值：把"切 batch、切通道、切空间、分配 op"统一成一组可组合的切分变量，**每个 op 可以独立选不同切法**——这才是比 DP/MP 表达力强的根源。

---

## 3. FlexFlow 总体架构

FlexFlow 接收两个输入，产出一个并行策略，交给 runtime 执行：

```
  ┌────────────────┐      ┌────────────────────┐
  │ Operator Graph │      │  Device Topology   │
  │  (计算图)       │      │   (设备拓扑)        │
  │ node = op      │      │ node = device      │
  │ edge = tensor  │      │ edge = connection  │
  └───────┬────────┘      └─────────┬──────────┘
          │                         │ (边带 带宽/延迟 标签)
          └──────────┬──────────────┘
                     ▼
        ┌────────────────────────────┐
        │     Execution Optimizer     │  ← 核心部件
        │  ┌───────────┐ ┌──────────┐ │
        │  │ MCMC 搜索  │↔│ Simulator│ │
        │  │ (探索空间) │ │ (裁判)    │ │
        │  └───────────┘ └──────────┘ │
        └─────────────┬──────────────┘
                      ▼
              最优 Parallelization Strategy
                      ▼
                  ┌────────┐
                  │ Runtime │  执行 split 方案
                  └────────┘
```

- **Operator graph**：计算图描述，op 作为 node，tensor 作为 edge。
- **Device topology**：描述真实设备的拓扑关系，device 作为 node，connection 作为 edge；**边上带宽/延迟有标签**。
- **Execution Optimizer**：FlexFlow 核心，用于搜索最优 split 方案，内部是"搜索算法 + 模拟器"协作。
- **Runtime**：执行被选中的 split 方案。

> 设计哲学（原文）：FlexFlow 把模拟器当**预言机（oracle）**，将并行优化问题转化为**成本最小化问题**——最小化预测执行时间。好处是**不必显式编码相互依赖的权衡**（如"减少数据传输 vs 平衡负载"），只盯整体执行时间。

---

## 4. 执行模拟器（Simulator）——为什么能快 1000 倍

这是 FlexFlow 最核心的部件，负责对 proposed strategy 做 evaluate，得到 candidate 的性能数据。

### 4.1 不真跑，而是模拟跑

直接在硬件上跑一轮训练来测时间，在 SOAP 这么大的空间里**代价太高**。模拟器照常构建执行 timeline，但当某个 op 要在 device 上执行时，**不真算**，而是从**上一次执行相同 input-size 的记录里直接取执行时间**。

模拟器依赖两个事实（原文）：

1. 很多 DNN 模型只用**少数几种**不同的 op。
2. 同种 op 对**相同 input-size** 的执行时间基本不变，且**与 input-data 内容无关**（只取决于数据 size 与硬件）。

```
  真实 profile (慢)              模拟 (快)
  ┌──────────────┐             ┌──────────────────┐
  │ 每个策略都    │             │ 每种 (op,size) 只 │
  │ 真跑一轮训练  │   ──►       │ 测一次，建查找表   │
  │ 测墙钟时间    │             │ 之后查表组装timeline│
  └──────────────┘             └──────────────────┘
        1×                          ~1000× 更快
```

> 因此模拟器对每种数据大小，**用一个 op 的实测计算时间来代表同类 op**，再把这些估算拼成不同并行策略的总时间。

### 4.2 Delta Simulation：增量模拟

模拟器还用了 **delta simulate** 算法：新策略往往只是在旧策略上改动一小处，于是**基于上一次模拟的结果做增量更新**，而非从头重算整条 timeline。

```
  策略 t       策略 t+1 (只改了 op_5 的切法)
  ┌─┬─┬─┬─┬─┐   ┌─┬─┬─┬─┬─┐
  │1│2│3│4│5│   │1│2│3│4│5'│  ← 只有 op_5 变
  └─┴─┴─┴─┴─┘   └─┴─┴─┴─┴─┘
  全量重算 = 慢   只重算受影响子图 + 下游 = 快
```

相比已有方法，delta simulation 有两个优势（原文）：**更快、所需资源更少**。论文报告**模拟器预测准确率很高**。

---

## 5. MCMC 搜索算法

遍历整个解空间是 **NP-hard**：原文指出"通过从最小完工时间（makespan）轻松减少来找最佳并行化策略是 NP 困难的"，且可能的策略数量与 op 数量呈**指数关系**，穷举不可行。

FlexFlow 用**成本最小化搜索程序**启发式探索，返回找到的最优策略。具体采用 **MCMC（马尔可夫链蒙特卡洛）** 随机方法。

### 5.1 为什么要 MCMC

- 蒙特卡洛：随机生成变量 $X_i$ 的值，用 $X_i$ 模拟计算，得到目标问题的解。
- 当 $X_i$ 的概率分布很复杂、不能用简单均匀分布转换得到时，就需要 **MCMC** 来生成 $X_i$ 的序列。

回到搜索：用 MCMC 随机采样生成随机的 $X_i$，每个 $X_i$ 即一个**候选策略**。

### 5.2 接受准则（直觉）

MCMC 用 Metropolis-Hastings 风格的接受概率，让搜索**偏好低成本策略**，又不至于卡在局部最优：

$$
p(\text{accept}) = \min\!\Big(1,\ \exp\big(\beta \cdot (\text{cost}_{\text{old}} - \text{cost}_{\text{new}})\big)\Big)
$$

- 新策略更快（cost 更小）→ 指数项 $>0$ → 概率 1，**总是接受**。
- 新策略更慢 → 以一定概率接受，**允许爬出局部最优**。
- $\beta$ 越大越"贪心"。这里 cost 由**模拟器**给出，所以每步只花模拟时间而非真机时间。

```
  cost
   │  ╲          ╱╲          搜索轨迹（接受好的，偶尔接受差的）
   │   ╲   局部 ╱  ╲    ●
   │    ╲_____╱    ╲  ╱ ╲___●  全局更优
   │                ╲╱      ╲___
   └──────────────────────────────► 策略空间
       MCMC 随机游走 + 模拟器打分
```

---

## 6. 程序接口与可移植性

与多数深度学习框架不同，FlexFlow 把**设备拓扑结构**当成一等公民：

- 用设备拓扑结构描述所有可用设备及其关联，拓扑的"边"带 **带宽** 与 **延迟** 标签。
- 给定一个计算图 + 一个设备拓扑，FlexFlow **自动**找到合适的并行策略。

两大优势（原文）：

1. **易于编程的接口**：用户不必手写并行切分。
2. **可移植性**：换硬件时自动选择高效策略，无需重写并行代码。

---

## 7. 数值示例：策略空间为什么是指数

设一个 op 的 output tensor 有 4 个可切维度（N、C、H、W），每维可切成 $1,2,4$ 份中的一种（受设备数约束），单 op 的切分组合数：

$$
3 \times 3 \times 3 \times 3 = 81 \text{ 种}
$$

若计算图有 $K$ 个 op，且每个 op 独立选择切法，则总策略数约：

$$
81^{K}
$$

- $K = 10$ 个 op：$81^{10} \approx 1.2 \times 10^{19}$ 种。
- 这就是"与 op 数呈指数关系"的来源——**穷举不可能**，必须靠模拟器 + MCMC。

> 对照感受成本差异：若真机评估一个策略要 1 分钟，评估 $10^{6}$ 个策略需约 **694 天**；模拟器快 1000 倍 → 约 **17 小时**。这正是"3 个数量级"带来的可行性。

---

## 实操：原论文要点与参考链接（保留原始真料）

### 原 paper

- 标题：**Beyond Data and Model Parallelism for Deep Neural Networks**（FlexFlow，斯坦福，MLSys 2019）

### 论文摘要（原文）

训练 DNN 的计算要求已增长到并行训练成为标准做法。现有系统通常只用数据或模型并行，往往导致并行化性能不佳。本文定义了更全面的并行化策略搜索空间 **SOAP**（sample / operation / attribute / parameter），并提出 **FlexFlow**：用 SOAP 空间的引导随机搜索为特定并行机寻找快速并行化策略。为加速搜索，FlexFlow 引入一种新颖的**执行模拟器**，能准确预测策略性能，比"必须执行每个策略"的先前方法快**三个数量级**。在两个 GPU 集群、六个真实 DNN 基准上评估，**即使包含搜索时间**，FlexFlow 也能把训练吞吐提升到 SOTA 的 **3.8 倍**并改善可扩展性。

### §6 Execution Optimizer（原文要点）

- 以 operator graph 和 device topology 为输入，自动找到有效并行化策略。
- 用模拟器作预言机，把并行优化转为**成本最小化**（最小化预测执行时间），避免显式编码相互依赖的优化权衡。
- 从 makespan 的归约证明：找最佳策略是 **NP-hard**；策略数与 op 数呈指数 → 穷举困难。
- 用成本最小化搜索程序启发式探索，返回找到的最优策略。

### 参考链接

- 更多维度的深度神经网络并行策略：<https://diandiangu.github.io/2020/07/20/FlexFlow/>
- 读论文《FlexFlow-Beyond Data and Model Parallelism for Deep Neural Networks》：<https://zhuanlan.zhihu.com/p/464355830>

---

## 常见问题 / 坑

| 问题 / 坑 | 说明 | 应对 |
|-----------|------|------|
| 模拟器假设"op 执行时间与 input-data 无关"何时失效？ | 数据相关分支（如稀疏/动态控制流、可变长序列、MoE 路由）会让同 size 不同 data 的耗时差很大 | 对这类 op 单独 profile 或退化为真机评估 |
| 同 size 时间"基本不变"对哪些 op 成立？ | 卷积、稠密 GEMM 等规则算子成立；论文明确"在大多数模型中假设成立" | 不规则算子需校准查找表 |
| 为什么不用穷举？ | $K$ 个 op 时策略数约 $81^K$，指数爆炸，NP-hard | 用 MCMC 启发式 + 模拟器打分 |
| MCMC 会卡在局部最优吗？ | 单纯贪心会，但 MH 接受准则允许以概率接受更差解 | 调 $\beta$ 控制探索/利用平衡 |
| SOAP 与张量并行/数据并行什么关系？ | DP=只切 Sample；TP≈切 Parameter；它们都是 SOAP 的**子集**角落 | FlexFlow 自动覆盖更广的混合切法 |
| delta simulation 为什么省资源？ | 新策略多为旧策略小改，只重算受影响子图而非整图 | 适合 MCMC 这种"每步小改"的搜索 |
| 设备拓扑为什么要带宽/延迟标签？ | 同样切法在 NVLink 与 PCIe / 跨机网络上成本天差地别 | 拓扑感知才能真实预测通信成本（见集合通信原语） |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 上游框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]
- 模型与架构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]]
- 训练与对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 硬件与网络（拓扑/通信成本来源）：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]]
- 性能评估：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
- 推理侧并行参考：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]]
