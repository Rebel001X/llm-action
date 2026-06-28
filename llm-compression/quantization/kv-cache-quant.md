# KV Cache 量化（KV-Cache Quantization）

> 把推理时随生成而膨胀的 Key/Value 缓存压成 INT4/INT2，让长上下文与大 batch 不再被显存卡死。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-optimizer/kv-cache]] · [[llm-compression/quantization/量化基础]] · [[llm-algo/旋转编码RoPE]] · [[llm-optimizer/FlashAttention]] · [[llm-inference/vllm/README]]

## 阅读地图

| 小节 | 你将搞懂 | 关键词 |
|------|----------|--------|
| 0 锚点 | 一句话说清 KV Cache 量化为什么独特 | 在线量化 / 推理瓶颈 |
| 1 地基 | KV Cache 是什么、为什么会成为瓶颈 | 自回归 / 显存墙 |
| 2 在线 vs 离线 | KV 量化跟权重量化的本质差异 | runtime / 延时占用 |
| 3 Key≠Value | 为什么 Key 比 Value 难量化（核心难点） | softmax / 分布差异 |
| 4 per-channel Key | KVQuant/KIVI 的招牌 recipe | 沿通道分组 / 异常值 |
| 5 RoPE 顺序 | KVQuant 为什么"先量化再 RoPE" | 旋转编码 / 反量化 |
| 6 KIVI | per-channel + FP16 residual 怎么落地 | autoregressive / 残差 |
| 7 IntactKV | 保留首 token KV 无损 | pivot token / massive activation |
| 8 QAQ | 混合 bit + 保留 outlier | 微分误差 / 高阶项 |
| 实操 | 各方案 recipe 对照 | 部署选型 |
| 坑 | 常见误区 | — |

## 0. 一句话锚点

> **KV Cache 量化 = 在推理过程中，把每一步新生成的 Key/Value 立刻就地压成低比特存起来；它是"在线一次性量化"，省的是显存，付的是延时。**

记住一个反差：权重量化是"离线量化一次、永久复用"；KV 量化是"在线量化、用完即弃"。**计算时通常仍反量化回 FP16 算**——存得省，算得准。

## 1. 地基：KV Cache 是什么、为什么会爆

### 1.1 KV Cache 的由来

Transformer 自回归解码时，每生成一个新 token 都要对**前面所有 token** 做注意力。若每步都重算所有历史 token 的 Key、Value，复杂度是 $O(n^2)$ 的重复劳动。于是缓存下来：

```
       生成第 t 步                需要的注意力
  q_t · [k_1 k_2 ... k_t]^T  ──►  对历史所有 K 算分
        softmax 后乘 [v_1 ... v_t]

  k_1..k_{t-1}, v_1..v_{t-1}  ←── 已经算过，不用重算
        ↑ 缓存它们 = KV Cache
```

代价是显存：每存一个 token 的 KV，占用为

$$
\text{bytes} = 2 \times L \times H \times d_{head} \times \text{batch} \times \text{seqlen} \times \text{precision}
$$

其中 $2$ 是 K 和 V，$L$ 层数，$H$ 头数。

### 1.2 数值手算：KV Cache 怎么就成了主瓶颈

以 LLaMA-7B（$L{=}32$, $H{=}32$, $d_{head}{=}128$，FP16=2 byte）单序列为例，单 token 的 KV：

$$
2 \times 32 \times 32 \times 128 \times 2 = 524288 \text{ byte} \approx 0.5\text{ MB/token}
$$

那么：
- 序列 **128K**：$0.5\text{MB} \times 128000 \approx 64\text{ GB}$ —— 一张 A100(80G) 几乎被 KV 吃满。
- 序列 **32K**：约 16 GB —— 而 7B 模型权重若已量化到 INT4 只占约 3.5 GB，**KV 反而成了大头**。

> 原文要点：**长序列和大 batch 下，激活内存（主要是 KV Cache）成为主要瓶颈**；尤其当模型权重已经量化到低精度后，对 LLaMA-7B，序列长到 128K 时 KV 缓存就是主瓶颈，**即便只有 32K，量化权重后 KV 也已是主瓶颈**。

这正是 KV 量化的动机：权重压完了，下一个该压的就是 KV。

## 2. 在线量化 vs 离线量化：KV 量化的特殊性

| 维度 | 权重量化（离线） | KV Cache 量化（在线） |
|------|------------------|----------------------|
| 量化时机 | 部署前一次性量化好，存盘 | 推理中边生成边量化 |
| 复用性 | 量化一次，永久复用 | 用完即弃，下个 batch 重来 |
| 资源占用 | 量化开销不计入推理 | **占用在线显存 + 增加延时** |
| 误差容忍 | 静态校准充分优化 | 在线，校准窗口受限 |

> 原文原话：KV Cache 量化"与离线参数一次性量化好可以不停使用不同，KV Cache 的数据是随时生成的，随后马上量化、存储和后续使用；虽然也是一次性量化，但这个一次性是在**整个推理过程中**的，是占有一部分**在线资源和延时**的。"

```
离线权重量化:  [校准]→[量化]→存盘  ===(部署后从不再变)===> 推理
在线 KV 量化:  生成 k_t → 立刻量化 → 存 KV → 后续步反量化使用
                 └──────── 每步都在发生，吃延时 ────────┘
```

设计含义：KV 量化算法必须**轻量**（kernel 要快），否则省下的显存被反复量化/反量化的延时吃掉。这也是"**值计算仍用 FP16**"的原因——只在存储环节降精度。

## 3. 核心难点：Key 比 Value 难量化得多

这是 KV 量化区别于普通激活量化的**第一性发现**，QAQ、KIVI、KVQuant 等多篇工作都独立证实：

> **Key cache 的量化难度远高于 Value cache。**

### 3.1 为什么？——softmax 把两者放在了非线性的两侧

```
   q · K^T  ──►  [ softmax ]  ──►  attn 权重 · V
      ↑                              ↑
   Key 在 softmax 之前            Value 在 softmax 之后
   (输入 / 非线性放大区)          (输出 / 已被归一化)
```

- **Key** 进入 softmax 的指数 $e^{x}$ 之前，分数上的小扰动经指数会被**非线性放大**；且 Key 上存在沿某些通道集中的**异常值（outlier）**，量化误差被放大后直接污染注意力分布。
- **Value** 在 softmax 之后被加权求和，分布更平滑、更"线性"，量化误差被归一化权重平摊掉。

> 原文：Key cache 与 value cache 分布完全不同，"这两块和 softmax 函数直接相关，一个在 softmax 前一个在后，softmax 又是非线性的，带来区别就很正常。"

### 3.2 QAQ 的数学视角：Key 误差含更高阶项

> QAQ 用**微分**表达量化前后误差：直观看，Key cache 推导出的误差式子**显著含有更高阶次**，因此需要更小心的量化策略（论文 Figure 1 给出问题定义）。

直觉：若把误差 $\Delta$ 对输出做泰勒展开，Value 路径基本是一阶（线性求和），而 Key 路径经过 softmax 的指数会引入 $\Delta^2$、$\Delta^3$ 等高阶项——同样的量化误差，Key 侧的输出偏差更大、更"震荡"。结论：**不能对 K、V 用同一套量化策略**。

## 4. 招牌 recipe：per-channel 量化 Key、per-token 量化 Value

KVQuant / KIVI 的核心配方：

> **KVQuant：per-channel 的 Key + per-token 的 Value 这样的组合。**

### 4.1 per-token 和 per-channel 在量化谁

KV Cache 张量形状约为 `[token, channel(=head×dim)]`。量化分组沿哪个维度走，决定每组共享一个 scale：

```
            channel 维 (沿头/特征) ───────────────►
   token   ┌───────────────────────────────────┐
    维      │  c0  c1  c2  c3  ...  cN           │
    │       │  ·   ·   ·   ·                     │
    ▼       │  ·   ·   ·   ·                     │
            └───────────────────────────────────┘

  per-token  量化:  每"一行"(一个 token)共享一个 scale  → 横着切
  per-channel 量化: 每"一列"(一个 channel)共享一个 scale → 竖着切
```

### 4.2 为什么 Key 要 per-channel

Key 的异常值**沿通道聚集**——某几个固定通道总是出现巨大值。

- 若用 **per-token**：一个 token 行里既有异常通道又有正常通道，巨大的异常值撑大整行 scale，正常通道被"陪绑"，精度崩。
- 若用 **per-channel**：把异常值**关进它自己那一列**，scale 大就大它一个，不污染其它通道。

```
  per-token (按行) 量化 Key:           per-channel (按列) 量化 Key:
    行内有 outlier → scale 被拉爆        outlier 独占一列 → 只放大该列 scale
    [ 2  3  9999  4 ] scale=2500         列c2专属 scale，其它列 scale 正常
       正常值被压成 0/1，全毁              正常列保持精度
```

> 原文：per-channel 量化 Key 是 KIVI/KVQuant 里"蛮重要的 recipe"。要点是**沿哪个 dimension 组 block**，而不是 block size 多大多小；**异常值的分布**正是 per-channel 的 motivation。

### 4.3 per-channel 在自回归下的麻烦 → FP16 residual

per-channel（沿通道）量化需要**跨多个 token** 统计该通道的范围。但自回归是逐 token 生成的——新 token 还没来，怎么提前定通道 scale？

> 原文：因为 per-channel quant 需要跨多个 token，所以怎么在 autoregressive 背景下 quant 是个挑战；**KIVI 搞 FP16 residual 一定程度上就是为了方便这个**。

机制（见第 6 节）：把最近的一小段 token 用 FP16 暂存（residual），攒够一个 group 再整体 per-channel 量化进 KV Cache。

## 5. RoPE 的顺序坑：KVQuant 为什么"先量化、再 RoPE"

[[llm-algo/旋转编码RoPE]] 会把 Key 按位置做旋转。旋转会**混合不同通道**，破坏 Key 原本"异常值沿固定通道聚集"的良好结构，让 per-channel 量化失效。

> 原文：KVQuant 采用的方法是**在 Key 前面的 RoPE 之前就完成量化**，实际使用时则**反量化之后再做一次 RoPE**。

```
  KVQuant 的 Key 路径:
    pre-RoPE Key ──► [量化存储]  (此时通道结构干净，per-channel 友好)
                          │
                    使用时 反量化
                          │
                          ▼
                    再施加 RoPE ──► 参与注意力
```

对照其它方案对 Key 难点的不同打法：

| 方案 | 对付 Key 难量化的核心手段 |
|------|--------------------------|
| KVQuant | RoPE **之前**量化，用时反量化后再 RoPE；per-channel Key + per-token Value |
| QAQ | **混合 bit 数** + **全精度保留 outlier** |
| KIVI | K、V 用**两种不同的分组粒度**（per-channel K / per-token V）+ FP16 residual |
| IntactKV | 保留**首 token / pivot token 的 KV 无损** |

## 6. KIVI：per-channel Key + FP16 residual 落地

KIVI 把第 3、4 节的认知工程化：

- **K 用 per-channel，V 用 per-token**——两种不同分组粒度对付两种不同分布。
- **FP16 residual**：滑动保留最近若干 token 的全精度 KV，凑满一个 group 后再 per-channel 量化沉淀进低比特缓存。既解决了"自回归下无法跨 token 提前定 scale"的难题，又让最近 token（注意力中往往最关键）保有高精度。

```
  KV Cache 布局 (KIVI):
   [ ───── 已量化的历史 KV (INT, per-channel K / per-token V) ───── | residual: 最近 R 个 token (FP16) ]
                                                                          ↑
                                            攒满 group_size 后整体量化, 并入左侧
```

## 7. IntactKV：保留关键词元（pivot token）的 KV 无损

### 7.1 现象：pivot token 上的超大异常值

LLM 中**某些特殊 token（尤其首 token / 起始符）** 会承载远超其它 token 的巨大激活值（massive activation）。这些 token 像注意力的"锚点"，量化误差落在它们身上会全局性地毁掉注意力分布。

> 原文：IntactKV 中 pivot token 的现象，在同期工作 *Massive Activations in Large Language Models* 中有更细致研究——详细分析了 pivot token 上超大 outlier 的**来源、作用和重要性**。

### 7.2 做法：让 pivot token 的 KV 完全不量化 + 端到端校准

- **保留首 token / pivot token 的 KV cache 无损**（这些位置不量化）。
- 校准用与其它 PTQ 方法相同的 **MSE loss**，但**不逐层优化，而是端到端优化所有层的 MSE**。
- 因为 IntactKV 是在**已量化好的模型上**做校准，内存开销很小。

> 原文实测：**7B 模型在单卡 H800 上只需训练 10 分钟**。

### 7.3 与生态的关系

> 原文：KVQuant 最新版也采用了与 IntactKV 相同的思路——**保持首 token 的 KV cache 无损**。IntactKV 还**同时支持权重量化和激活值量化**，与目前多种 LLM 主流量化方法兼容。

```
  IntactKV 思想:
    token:   [BOS]  the   cat   sat  ...
    KV存储:  FP16   INT   INT   INT       ← 只放过 pivot/首 token
             ↑保无损   ↑其余照常量化
```

## 8. QAQ：混合精度 + 保留异常值

> 原文：QAQ 采用**混合 bit 数 + 全精度保留 outlier** 的量化策略；并用**微分表达量化误差**，发现 Key 的误差式含更高阶次，需更小心的策略（问题定义见论文 Figure 1）。

直觉对照（同样基于"Key 难、Value 易"的认知，但工程手段不同）：

```
  QAQ:      正常值 → 低 bit;  outlier → 保留全精度(FP16);  K、V 不同 bit 预算
  KVQuant:  RoPE 前量化 + per-channel K / per-token V + 保留首 token
  KIVI:     per-channel K / per-token V + FP16 residual
  IntactKV: pivot token KV 无损 + 端到端 MSE 校准
```

## 实操：四大方案 recipe 速查

> 下表把原文出现的全部真实 recipe 整理为选型表（不涉及未在原文出现的命令/版本/参数）。

| 方案 | Key 策略 | Value 策略 | RoPE 处理 | 关键 trick | 原文标注的实测/特性 |
|------|----------|------------|-----------|-----------|---------------------|
| **KVQuant** | per-channel | per-token | RoPE **前**量化，用时反量化后再 RoPE | 最新版保留首 token KV 无损 | — |
| **KIVI** | per-channel | per-token | — | FP16 residual（解决自回归 per-channel） | per-channel for Key 是重要 recipe |
| **IntactKV** | 保留 pivot/首 token 无损 | 同左 | — | 端到端（非逐层）MSE 校准；兼容权重/激活量化 | **7B 单卡 H800 训练约 10 min** |
| **QAQ** | 混合 bit + 保留 outlier 全精度 | 较低 bit | — | 微分量化误差分析（Figure 1） | Key 误差含更高阶项 |

通用工程约束（原文锚点）：
- **存储降精度，计算回 FP16**：KV Cache 值计算还是用 FP16 是比较常见的方式。
- **量化是在线发生的**：占用在线显存与延时，kernel 必须轻量。

## 常见问题 / 坑

| 现象 / 误区 | 真相 | 依据 |
|------------|------|------|
| "K、V 用同一套量化就行" | 错。Key 在 softmax 前、非线性放大，难度远高于 Value，需分别处理 | §3 |
| "per-channel 就是 block 更小" | 错。per-channel 在乎**沿哪个维度**组 block，而非 block size 大小 | §4.2 |
| "先 RoPE 再量化 Key 没问题" | RoPE 混合通道、破坏 Key 异常值结构 → per-channel 失效；应 RoPE 前量化 | §5 |
| "per-channel 在自回归里直接能用" | per-channel 需跨多 token 统计，新 token 没来无法定 scale → 需 FP16 residual 缓冲 | §4.3 / §6 |
| "首 token 没什么特殊" | pivot/首 token 携带 massive activation，量化它会全局毁掉注意力 → 保无损 | §7 |
| "KV 量化省显存却没代价" | 它是在线量化，占用在线资源 + 增加延时；省的是显存换的是时间 | §2 |
| "Value 也要 per-channel" | Value 分布平滑、softmax 后更线性，per-token 即足够，省 scale 开销 | §3.1 / §4 |

## 🔗 跳转链接

枢纽：[[00-知识地图]]

- 架构基础：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 推理优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 量化家族：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练 / 对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 硬件 / 网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测 / 估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
