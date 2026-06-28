# 大模型核心算法与模型家族（AI-Algo）

> 一句话定位：把 GPT / BERT / LLaMA / GLM / Bloom / Codex 这些"名字"还原成同一套**算法零件**（Transformer Block + 注意力 + 位置编码 + 归一化 + 训练目标），看清谁换了哪个零件、为什么换。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]]、[[llm-algo/旋转编码RoPE]]、[[llm-algo/mlp]]、[[llm-algo/moe/README]]、[[llm-optimizer/FlashAttention]]、[[llm-optimizer/kv-cache]]、[[llm-inference/解码策略]]

## 阅读地图

| 节 | 你会得到什么 | 关键产出 |
|----|--------------|----------|
| 0 | 一句话锚点：所有 LLM = 同一个 Block 堆 N 次 | 心智模型 |
| 1 | 地基：Token → Embedding → Block → LMHead 的数据流 | 形状追踪 |
| 2 | 注意力原子：QKV / Softmax / 因果掩码 / MHA·MQA·GQA | 手算一遍 |
| 3 | 位置编码：绝对 vs 相对 vs RoPE | 谁能外推 |
| 4 | FFN/MLP 与 MoE：算力都烧在这 | FLOPs 占比 |
| 5 | 归一化与残差：Pre-LN/Post-LN/RMSNorm | 为什么能训深 |
| 6 | 三种训练目标：AR / MLM / Prefix-LM | GPT·BERT·GLM 之分水岭 |
| 7 | 模型家族对照：GPT2/BERT/Bloom/LLaMA/GLM/Codex | 换了哪个零件 |
| 8 | 数值手算：参数量 / FLOPs / KV-Cache 显存 | 拿计算器对答案 |
| 9 | 常见问题表 + 跳转 | 串联全仓 |

---

## 0. 一句话锚点

> **一切自回归大模型，本质都是把同一个 Transformer Decoder Block 复制 $N$ 层、堆起来，再在末端接一个"预测下一个 token"的线性分类头。** 所谓 GPT、LLaMA、Bloom 的差别，只是这个 Block 内部的 4 个零件（注意力变体、位置编码、激活、归一化）和**训练目标**的不同排列组合。

记住这张"换零件"对照心法，后面 7 节都是在填它：

```
                 ┌───────────────── Transformer Block (堆 N 次) ─────────────────┐
 token ids ─► Embedding ─►│  Norm → Attention(QKV) → +残差 → Norm → FFN/MoE → +残差 │─► ... ─► Norm ─► LMHead ─► logits
                 └──────────────────────────────────────────────────────────────┘
                              ▲位置编码          ▲MHA/MQA/GQA      ▲MLP/MoE
   可替换零件：  [训练目标 AR/MLM/Prefix] [PosEnc 绝对/RoPE/ALiBi] [Norm LN/RMSNorm] [激活 GeLU/SwiGLU]
```

---

## 1. 地基：一条数据是怎么穿过模型的

设词表大小 $V$、隐藏维度 $d$、层数 $L$、序列长 $s$、批大小 $b$。一次前向的**形状流水线**：

```
输入文本 "我 爱 大 模 型"
  │ tokenizer（BPE/SentencePiece）
  ▼
token ids        [b, s]                 ← 整数，例如 [[101, 2769, ...]]
  │ 查 Embedding 表 E∈[V, d]
  ▼
hidden states    [b, s, d]              ← 浮点向量
  │ ×L 个 Block（形状不变！只做信息混合）
  ▼
hidden states    [b, s, d]
  │ 末端 Norm + LMHead  W_lm∈[d, V]（常与 E 权重共享=weight tying）
  ▼
logits           [b, s, V]              ← 每个位置对全词表打分
  │ softmax → 取下一个 token
  ▼
"我 爱 大 模 型 的 训 练 ..."
```

关键直觉：**Block 不改形状**（始终 `[b,s,d]`），它只是把每个位置的向量"按上下文重新混合"。深度 $L$ 决定混合多少轮。

> 🔗 嵌入/词表/权重共享细节见 [[llm-algo/transformer/模型架构]]。

---

## 2. 注意力原子：信息在 token 之间流动的唯一通道

### 2.1 单头自注意力（Scaled Dot-Product Attention）

三步：用三个线性层把每个 token 向量投影成 Query / Key / Value，让 Query 去和所有 Key 算相似度（点积），归一化成权重后对 Value 加权求和。

$$
\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V
$$

- $Q=XW_Q,\ K=XW_K,\ V=XW_V$，形状 $[s, d_k]$。
- $\sqrt{d_k}$ 缩放：防止点积随维度变大导致 softmax 饱和、梯度消失。
- $M$ 是**因果掩码**：未来位置填 $-\infty$，softmax 后变 0 —— 这是自回归"不能偷看未来"的算法实现。

```
因果掩码 M（s=4，✓=可见 ×=屏蔽为 -∞）:
        k1  k2  k3  k4
   q1 [  ✓   ×   ×   × ]   位置1只看自己
   q2 [  ✓   ✓   ×   × ]
   q3 [  ✓   ✓   ✓   × ]
   q4 [  ✓   ✓   ✓   ✓ ]   位置4能看全部历史
        下三角（含对角）保留
```

### 2.2 多头：把 d 切成 h 份并行算

```
  X[s,d] ──split──►  head1[s,d/h]  head2[s,d/h] ... headh[s,d/h]
                        │Attn          │Attn          │Attn
                        ▼              ▼              ▼
                     out1           out2          outh
                        └──── concat ────►[s,d] ──W_O──► 输出[s,d]
```

多头 = 让模型在不同子空间里同时学"语法关系/指代关系/位置关系"。

### 2.3 MHA → MQA → GQA：为推理省 KV-Cache

推理时每生成一个 token 都要缓存历史的 K、V。MHA 每头各一份 KV，显存大；MQA 让所有头**共享一份** KV；GQA 折中（每组头共享）。

```
MHA: Q头8  K头8  V头8     KV 显存基准 ×1
GQA: Q头8  K头2  V头2     KV 显存 ×1/4   ← LLaMA2-70B 用
MQA: Q头8  K头1  V头1     KV 显存 ×1/8   ← 推理最省，质量略降
```

> 🔗 KV-Cache 的显存账与优化见 [[llm-optimizer/kv-cache]]、[[llm-inference/KV-Cache优化]]；把 softmax 分块、省显存且更快的核见 [[llm-optimizer/FlashAttention]]。

---

## 3. 位置编码：注意力本身"看不见顺序"

点积注意力对 token 顺序**置换不变**（打乱输入，输出只是跟着打乱），所以必须额外注入位置信息。三代方案：

| 方案 | 代表 | 注入方式 | 长度外推 |
|------|------|----------|----------|
| 绝对（Sinusoidal/Learned） | 原始 Transformer、GPT2、BERT | 直接加到 Embedding | 差（超训练长度崩） |
| 相对偏置（ALiBi） | Bloom | 在注意力分数上按距离减线性偏置 | 好 |
| 旋转（RoPE） | LLaMA、GLM | 把 Q/K 按位置**旋转**一个角度 | 好（可插值扩展） |

RoPE 核心：把每个 2 维子向量按位置 $m$ 旋转角 $m\theta$，使得点积 $\langle q_m, k_n\rangle$ 只依赖**相对距离** $m-n$：

```
位置 m 的向量(成对维度) 旋转矩阵:
  [ cos(mθ)  -sin(mθ) ] [x1]
  [ sin(mθ)   cos(mθ) ] [x2]
   ↑ 不同维度用不同 θ_i = 10000^(-2i/d)：低维转得快(细粒度)，高维转得慢(粗粒度)
```

> 🔗 RoPE 的完整推导、长度插值（NTK/线性）见 [[llm-algo/旋转编码RoPE]]。

---

## 4. FFN / MLP 与 MoE：算力的大头都在这

每个 Block 里，注意力之后跟一个逐位置前馈网络（Position-wise FFN）。**它占了模型 ~2/3 的参数和 FLOPs**。

```
标准 FFN（升维→激活→降维）：
  x[d] ──W1──► [d_ff]（通常 d_ff=4d）──激活──► [d_ff] ──W2──► [d]
              GPT2/BERT 用 GeLU；LLaMA 用 SwiGLU（多一个门控 W3）
```

SwiGLU：$\text{FFN}(x)=\big(\text{SiLU}(xW_1)\odot xW_3\big)W_2$，比单纯 GeLU 效果好，故 LLaMA 把 $d_{ff}$ 调成 $\frac{8}{3}d$ 以对齐参数量。

**MoE（混合专家）**：把一个大 FFN 换成 $E$ 个小 FFN（专家），每个 token 只激活其中 top-$k$ 个。参数量爆涨但每 token 计算量几乎不变——"知识容量"与"推理算力"解耦。

```
       ┌─ Router（小线性层打分，选 top-2）─┐
 token─┤  Expert1  Expert2 ... Expert8     │  只有被选中的 2 个专家计算
       └──────────► 加权求和 ◄─────────────┘
 总参数 ∝ E×单专家；激活参数 ∝ k×单专家（k≪E）
```

> 🔗 MLP 细节见 [[llm-algo/mlp]]；MoE 路由/负载均衡/并行见 [[llm-algo/moe/README]]。

---

## 5. 归一化与残差：为什么能堆几十层不崩

- **残差连接** $x + \text{Sublayer}(x)$：给梯度一条"高速公路"，让 $L$ 很深时梯度不消失。
- **LayerNorm vs RMSNorm**：LN 减均值除标准差再仿射；RMSNorm 省掉减均值，只按均方根缩放，更快、效果相当（LLaMA 用 RMSNorm）。

$$
\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}\cdot g
$$

- **Pre-LN vs Post-LN**：归一化放在子层"前"还是"后"，决定训练稳定性。

```
Post-LN（原始 Transformer）          Pre-LN（GPT2/LLaMA 主流）
  x ─► Sublayer ─► (+x) ─► Norm        x ─► Norm ─► Sublayer ─► (+x)
  深层时残差被 Norm 压缩，难训          残差始终"干净"，可稳定训百层
```

经验法则：**想训得深、训得稳，用 Pre-LN + RMSNorm**。

---

## 6. 三种训练目标：GPT / BERT / GLM 的真正分水岭

模型家族的最大区别**不是结构，是预测什么**。

```
①自回归 AR（GPT/LLaMA/Bloom/Codex，Decoder-only，因果掩码）
  我 爱 大 [?]            目标：从左到右预测下一个词 → 天生会"生成"

②掩码语言 MLM（BERT，Encoder-only，双向可见）
  我 爱 [MASK] 模型       目标：还原被遮的词，能看左右上下文 → 擅长"理解/分类"

③前缀/空白填充 Prefix-LM & GLM
  前缀双向可见 + 续写部分单向        GLM：自回归空白填充，统一理解与生成
```

| 目标 | 掩码 | 擅长 | 代表 |
|------|------|------|------|
| AR（Causal LM） | 因果（下三角） | 文本生成、对话、代码 | GPT2、LLaMA、Bloom、Codex、CodeGeeX |
| MLM | 无（全可见） | 句向量、分类、抽取 | BERT |
| Prefix-LM / 空白填充 | 前缀双向+续写单向 | 兼顾理解与生成 | GLM、ChatGLM |

> 🔗 对话模型还要叠加对齐阶段（SFT→RLHF/DPO），见 [[llm-alignment/RLHF]]、[[llm-alignment/DPO]]；解码采样策略见 [[llm-inference/解码策略]]、[[autoregressive-lm-decoding-methods]]。

---

## 7. 模型家族对照：换了哪个零件

| 模型 | 结构 | 目标 | 位置编码 | 归一化 | 激活 | 备注 |
|------|------|------|----------|--------|------|------|
| **GPT2** | Decoder-only | AR | 学习式绝对 | Pre-LN | GeLU | 自回归生成范式奠基 |
| **BERT** | Encoder-only | MLM+NSP | 学习式绝对 | Post-LN | GeLU | 理解任务王者 |
| **Bloom** | Decoder-only | AR | **ALiBi** | LN+embedding-norm | GeLU | 多语言、176B，见经验贴 |
| **LLaMA/LLaMA2** | Decoder-only | AR | **RoPE** | **RMSNorm**(Pre) | **SwiGLU** | 开源 SOTA 基座，LLaMA2 用 **GQA** |
| **GLM/ChatGLM** | Prefix 变体 | 空白填充 | RoPE(2D) | Post→DeepNorm | GeLU | 统一理解+生成，中英双语 |
| **Codex/CodeGeeX** | Decoder-only | AR(代码语料) | 绝对/RoPE | Pre-LN | GeLU | 在 GPT 类基座上用代码语料续训 |

读法：**横向看差异都集中在 4~5 个零件格子上**，主干（堆 Block + 注意力 + FFN）完全一致。这就是为什么一套训练/推理框架能跑遍所有家族。

> 🔗 训练这些基座的工程见 [[llm-train/README]]、[[ai-framework/megatron-lm/README]]、[[ai-framework/deepspeed/README]]；176B 实战教训见 [[Bloom-176B训练经验]]、[[GLM-130B训练经验]]。

---

## 数值手算：把抽象参数算成具体数字

以一个**典型 7B 配置**为锚：$d=4096,\ L=32,\ h=32,\ d_{ff}=11008,\ V=32000,\ s=2048$。

### (A) 参数量估算

每层主要参数：
- 注意力 $W_Q,W_K,W_V,W_O$：$4d^2 = 4\times4096^2 \approx 6.71\times10^7$
- FFN（SwiGLU 三矩阵）：$3\times d\times d_{ff}=3\times4096\times11008\approx1.35\times10^8$
- 每层合计 $\approx 6.71\times10^7+1.35\times10^8 \approx 2.02\times10^8$

总计：
$$
L\times2.02\times10^8 + \underbrace{V\times d}_{\text{embedding}} = 32\times2.02\times10^8 + 32000\times4096 \approx 6.47\times10^9+1.31\times10^8 \approx 6.6\text{B} \;\checkmark
$$
（验证 7B 量级合理；注意 FFN 占每层 $\approx67\%$，印证第 4 节"算力大头在 FFN"。）

### (B) 一次前向 FLOPs（粗估法则）

经验公式：**前向 FLOPs $\approx 2\times N_{\text{params}}\times s$**（每个参数对每个 token 做一次乘加=2 FLOPs）。训练含反向再 $\times3$。

$$
\text{前向} \approx 2\times6.6\times10^9\times2048 \approx 2.70\times10^{13}\ \text{FLOPs} = 27\ \text{TFLOPs/序列}
$$

> 🔗 含注意力 $O(s^2)$ 项的精确推导见 [[llm-base/FLOPS]]、[[llm-algo/FLOPs]]。

### (C) 推理 KV-Cache 显存

每 token 每层缓存 K、V 各一份：$2\times d$ 个值。FP16 每值 2 字节。

$$
\text{KV显存} = 2(\text{KV}) \times L \times d \times s \times 2(\text{bytes}) \times b
$$
$$
= 2\times32\times4096\times2048\times2\times1 \approx 2.15\times10^9\ \text{B} \approx 2.0\ \text{GiB}（batch=1, 满 2048 长度）
$$

若用 **GQA（KV 头数 8，对比 Q 头 32）**，KV 维度降为 $d/4$：

$$
2.0\ \text{GiB}\times\frac{1}{4}=0.5\ \text{GiB}
$$

—— 这就是 LLaMA2-70B 选 GQA 的直接动机：长上下文 + 大 batch 时 KV-Cache 才是显存瓶颈，而非权重。

> 🔗 与权重/激活显存合并的完整账见 [[transformer内存估算]]。

---

## 常见问题

| 问题 | 一句话答案 |
|------|------------|
| 为什么注意力要除 $\sqrt{d_k}$？ | 点积方差随 $d_k$ 线性增大，会把 softmax 推到饱和区、梯度趋零；缩放后方差归一。 |
| GPT 和 BERT 结构差很多吗？ | 几乎不差，核心差别是**训练目标+掩码**（AR 因果 vs MLM 双向），不是模块。 |
| 为什么 LLaMA 不用绝对位置编码？ | RoPE 把相对位置编进点积、可做长度插值外推；绝对编码超训练长度即崩。 |
| MoE 是不是更费算力？ | 不。总参数大但每 token 只激活 top-$k$ 专家，**激活算力≈稠密小模型**。 |
| 推理瓶颈是权重还是 KV-Cache？ | 短序列看权重；**长上下文/大 batch 时 KV-Cache 主导**，故有 MQA/GQA。 |
| Pre-LN 一定比 Post-LN 好？ | 深层训练更稳，是大模型主流；但 Post-LN 收敛后表征略强，浅层仍可用。 |
| Codex/CodeGeeX 是新架构吗？ | 不是，就是 GPT 类 Decoder-only 在代码语料上续训/微调。 |
| 为什么 FFN 升维到 4d？ | 给非线性变换更大的"工作空间"，是容量与算力的经验甜点（SwiGLU 改 $\frac{8}{3}d$）。 |

---

## 🔗 跳转链接

- 总图：[[00-知识地图]]
- 主干结构：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/moe/README]]
- 位置编码：[[llm-algo/旋转编码RoPE]]
- 注意力加速：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]]
- 计算量与显存：[[llm-algo/FLOPs]] · [[llm-base/FLOPS]] · [[transformer内存估算]]
- 解码与服务：[[llm-inference/解码策略]] · [[autoregressive-lm-decoding-methods]] · [[llm-inference/README]] · [[llm-inference/大模型推理张量并行]]
- 训练与框架：[[llm-train/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[Bloom-176B训练经验]] · [[GLM-130B训练经验]]
- 微调与对齐：[[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 量化压缩：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 底层算力：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/ai-hardware/CUDA]]
