# LLaMA 系列总览

> Meta 开源的稠密 Decoder-only 大语言模型家族，用「小参数 + 大数据 + 干净架构」把开源 LLM 推到接近闭源 SOTA，并成为整个开源生态的事实基座。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/llama/模型架构]] [[llm-algo/qwen/README]]

## 阅读地图

| 节 | 主题 | 你将学到 | 难度 |
|----|------|----------|------|
| 0 | 一句话锚点 | LLaMA 到底是什么、为什么重要 | ★ |
| 1 | 地基/前置 | Decoder-only、Transformer、缩放定律 | ★★ |
| 2 | 演进史 | LLaMA1→2→3→3.1 每代改了什么 | ★★ |
| 3 | 架构四件套 | RoPE / RMSNorm / SwiGLU / GQA 原子拆解 | ★★★ |
| 4 | 数据与规模 | token 量、上下文、词表、参数配置 | ★★ |
| 5 | 训练与对齐 | 预训练→SFT→RLHF/DPO 流程 | ★★ |
| 6 | 开源影响与生态 | 协议、衍生模型、推理框架 | ★★ |
| 7 | 与 Qwen 等同类对比 | 横向定位 | ★★ |
| 8 | 数值例子 | 参数量/显存/KV Cache 手算 | ★★★ |
| 9 | 常见问题 | 易错点速查 | ★ |

## 0. 一句话锚点

LLaMA（Large Language Model Meta AI）是 Meta 从 2023 年起开源的一系列**稠密**（非 MoE）**Decoder-only** 大语言模型。它的历史地位不在于「最大」，而在于：用**远小于 GPT-3 的参数量**（最小 7B vs 175B），靠**喂更多 token**（1T~15T），训出**逼近甚至超过**当时闭源模型的效果，并把权重开放给社区。一句话：**LLaMA = 开源世界的「Linux 内核」级基座**。

## 1. 地基/前置

要看懂 LLaMA，先要有三个底层概念。不假设你记得，逐个拆到原子。

### 1.1 什么是 Decoder-only Transformer

Transformer 原始论文是「Encoder-Decoder」（翻译用）。现代生成式大模型几乎都只保留 **Decoder** 一半，叫 **Decoder-only**。它的唯一任务是**自回归预测下一个 token**：给定前面所有词，预测下一个词的概率分布。

```
输入: "中国 的 首都 是"
                │
        ┌───────▼────────┐
        │  Decoder 堆叠   │  (N 层完全相同的 DecoderLayer)
        └───────┬────────┘
                │  输出每个位置对全词表的概率
                ▼
        预测: "北京" (概率最高)
```

"自回归"= 把上一步预测的词拼回输入，再预测下一个，循环生成。

### 1.2 一层 DecoderLayer 长什么样

LLaMA 的每一层（对应代码里的 `LlamaDecoderLayer`）由两个子模块组成，各带一个**残差连接**和**归一化**：

```
       x ──────────────┐ (残差)
       │               │
   ┌───▼────┐          │
   │RMSNorm │          │
   └───┬────┘          │
   ┌───▼────────┐      │
   │ Self-Attn  │ (含 RoPE + GQA)
   │ LlamaAttention    │
   └───┬────────┘      │
       │               │
       +◄──────────────┘
       │
       x' ─────────────┐ (残差)
       │               │
   ┌───▼────┐          │
   │RMSNorm │          │
   └───┬────┘          │
   ┌───▼────────┐      │
   │  SwiGLU MLP│ (LlamaMLP)
   └───┬────────┘      │
       │               │
       +◄──────────────┘
       ▼
     输出 (送入下一层)
```

注意 LLaMA 用 **Pre-Norm**（归一化放在子模块**前面**），比原始 Transformer 的 Post-Norm 训练更稳定。这一层里就藏着我们要讲的四件套：RoPE、RMSNorm、SwiGLU、GQA。

### 1.3 缩放定律（为什么是"小模型大数据"）

DeepMind 的 Chinchilla 论文指出：固定算力预算下，**参数量 $N$ 和训练 token 数 $D$ 应同比例增长**，经验上 $D \approx 20N$ 较优。GPT-3 是 175B 参数只喂 300B token（严重"欠训练"）。LLaMA 反其道而行：**参数压小、token 拉满**，于是 7B 模型也能很强，且推理便宜。这正是 LLaMA 能"以小搏大"的理论基础。

## 2. 演进史：LLaMA 1 → 2 → 3 → 3.1

每一代的核心改动总结成一张演进图（数字以官方为准，下表为公开口径概数）：

```
LLaMA 1 (2023.02)         LLaMA 2 (2023.07)        LLaMA 3 (2024.04)        LLaMA 3.1 (2024.07)
─────────────────         ─────────────────        ─────────────────       ──────────────────
7/13/33/65B               7/13/70B                 8/70B                    8/70/405B
1.0~1.4T token            2.0T token               15T+ token              15T+ token
ctx 2048                  ctx 4096                 ctx 8192                ctx 128K
仅研究许可                可商用(Llama2协议)       可商用                  可商用
MHA(65B用?)               GQA(仅34/70B)            GQA(全系)               GQA(全系)
无Chat官方版              首发 Llama-2-Chat        Instruct 版            Instruct + 工具调用
                          RLHF(PPO)                                       蒸馏(405B→8/70B)
                                                  词表32K→128K            首个开源405B级
```

逐代要点：

- **LLaMA 1**：证明"开源可逼近 GPT-3"。但许可证只允许研究，且只放给申请者（后被泄露到社区，反而引爆了开源浪潮）。
- **LLaMA 2**：里程碑式**开放商用**。引入官方 **Chat 版**（SFT + RLHF），并在大模型上用 **GQA** 降推理成本。安全对齐（红队、helpfulness/safety 双奖励）做得很重。
- **LLaMA 3**：词表从 32K SentencePiece 升级到 **128K tiktoken-BPE**，编码效率大涨（同样文本更少 token）；训练数据扩到 **15T+ token**；全系标配 GQA。8B 模型即超过上一代 13B。
- **LLaMA 3.1**：首次开源 **405B** 级别（接近 GPT-4 档），上下文拉到 **128K**，并用 405B 作"教师"**蒸馏**提升 8B/70B。补齐了多语言、长上下文、工具调用能力。

> 后续还有 3.2（含轻量端侧 1B/3B 与视觉版）、3.3（70B 高效版）等，本文聚焦 1~3.1 主干，细节以官方 Model Card 为准。

## 3. 架构四件套（原子拆解 + ASCII）

LLaMA 相对原始 Transformer 的四个关键工程选择。每个都讲"为什么"。

### 3.1 RoPE：旋转位置编码

**问题**：Self-Attention 本身对词序无感（打乱输入注意力分数不变），必须注入"位置"信息。

**老办法**：把一个位置向量**加**到词向量上（绝对位置编码）。缺点：外推差、不直接表达"相对距离"。

**RoPE 思路**：不"加"位置，而是按位置把 Q、K 向量在二维平面里**旋转一个角度**。位置 $m$ 越大转得越多。两个词做点积时，结果只依赖它们的**相对位置差** $m-n$，天然表达相对距离。

对一对维度 $(x_1,x_2)$，在位置 $m$ 的旋转：

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

不同维度对用不同频率 $\theta_i = 10000^{-2i/d}$，低频维度转得慢（管长距离），高频维度转得快（管局部）。对应代码 `LlamaRotaryEmbedding`。

```
位置m=0   位置m=1   位置m=2     向量像钟表指针，位置越大转角越大
  →         ↗         ↑         点积只看两指针夹角(相对位置)
 0°        45°       90°        → 天生支持相对位置 + 易外推
```

LLaMA 3.1 把上下文从 8K 扩到 128K，靠的就是对 RoPE 频率做 **scaling**（改 base / NTK 插值），让旋转角在长序列上不"绕飞"。

### 3.2 RMSNorm：均方根归一化

**LayerNorm**（原始用法）：先减均值、再除标准差、再缩放平移，含 $\mu$、$\sigma$、$\gamma$、$\beta$ 四套统计/参数。

**RMSNorm**：去掉"减均值"和偏置 $\beta$，只用**均方根**做缩放：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

**为什么**：实验发现"减均值"贡献小，去掉后**计算更省、训练同样稳**。少了一次求均值和一组偏置参数。

```
LayerNorm:  x -> 减均值μ -> 除标准差σ -> *γ +β   (4步)
RMSNorm:    x ->         -> 除RMS    -> *γ        (2步, 更快)
```

### 3.3 SwiGLU：门控前馈网络

**普通 MLP**：`Linear -> ReLU -> Linear`。

**SwiGLU**：把中间激活换成"门控"形式，用三个线性层：

$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}})\big)\,W_{\text{down}}$$

其中 $\text{SiLU}(z)=z\cdot\sigma(z)$（也叫 Swish），$\odot$ 是逐元素相乘（门控）。对应代码 `LlamaMLP` 的 `gate_proj / up_proj / down_proj`。

**为什么**：门控让网络能"选择性放行"信息，表达力强、收敛好。代价是参数从 2 个矩阵变 3 个，所以 LLaMA 把中间维度从 $4d$ 缩到约 $\frac{8}{3}d$ 来保持总参数量不变。

```
        ┌──gate_proj──► SiLU ─┐
   x ───┤                     ⊙ ──► down_proj ──► out
        └──up_proj───────────►┘
          (门控分支与值分支逐元素相乘)
```

### 3.4 GQA：分组查询注意力

**问题**：自回归生成要缓存历史的 K、V（KV Cache）。标准**多头注意力 MHA** 每个 Q 头都配独立的 K/V 头，Cache 巨大，是长上下文的显存/带宽瓶颈。

**两个极端**：
- **MHA**：$h$ 个 Q 头，$h$ 个 KV 头（质量好，Cache 大）。
- **MQA**：$h$ 个 Q 头，但**只 1 个** KV 头（Cache 最小，质量略降）。

**GQA 折中**：把 Q 头分成 $g$ 组，**每组共享一个 KV 头**。LLaMA 2/3 大模型多用 8 个 KV 组。KV Cache 缩小 $h/g$ 倍，质量几乎不掉。

```
MHA (8Q,8KV)        GQA (8Q,2KV)         MQA (8Q,1KV)
Q1..Q8              Q1Q2Q3Q4 Q5Q6Q7Q8    Q1..Q8
│ │ │ ...│            └──┬──┘   └──┬──┘     └───┬───┘
K1K2..K8               KV组1    KV组2          KV共享
KV最大              KV缩4倍               KV最小
```

## 4. 数据与规模

### 4.1 训练数据

- **来源**：公开网页爬取（CommonCrawl 为主）、代码（GitHub）、维基百科、书籍、论文（ArXiv）、StackExchange 等。**全部公开数据**，不含 Meta 用户私有数据（官方口径）。
- **清洗**：去重、质量过滤、去毒、PII 处理。LLaMA 3 强调用"数据质量分类器"做精选。
- **规模**：1→2→3 大致是 1.4T → 2T → 15T+ token，数据是逐代最大变量。

### 4.2 关键超参（公开概数，以官方 Model Card 为准）

| 模型 | 层数 | 隐藏维 $d$ | 头数 | KV头 | 词表 | ctx |
|------|------|-----------|------|------|------|-----|
| LLaMA2-7B | 32 | 4096 | 32 | 32(MHA) | 32K | 4096 |
| LLaMA2-70B | 80 | 8192 | 64 | 8(GQA) | 32K | 4096 |
| LLaMA3-8B | 32 | 4096 | 32 | 8(GQA) | 128K | 8192 |
| LLaMA3.1-405B | 126 | 16384 | 128 | 8(GQA) | 128K | 128K |

词表 32K→128K 是 LLaMA 3 的大改：更细的 token 切分 + 更高编码效率，对中文等非英语语言收益尤其明显。

## 5. 训练与对齐流程

LLaMA 的"基座 → 助手"分三阶段，理解这条流水线对用好它至关重要。

```
[海量公开文本]
     │  ① 预训练 (自监督, next-token)  耗算力 99%
     ▼
[Base 模型]  会"续写"但不会"听话"
     │  ② SFT 指令微调 (人工/合成 问答对)
     ▼
[SFT 模型]  会回答指令
     │  ③ 偏好对齐 (LLaMA2: RLHF-PPO; 新版多用 DPO/拒绝采样)
     ▼
[Instruct/Chat 模型]  更有用 + 更安全
```

- **②SFT**：用高质量「指令-回答」对，把续写模型"调教"成助手。
- **③对齐**：LLaMA 2 用经典 **RLHF**——训一个奖励模型打分，再用 **PPO** 优化策略；同时训 helpfulness 和 safety 两个奖励。新版更多用 **DPO**（直接偏好优化，省掉奖励模型与 PPO，更稳更省）和拒绝采样。

## 6. 开源影响与生态

### 6.1 协议要点

LLaMA 1 仅限研究；**LLaMA 2 起开放商用**，但用的是**自定义社区许可**（非标准 Apache/MIT）：核心限制是**月活超 7 亿的超大厂需单独获 Meta 授权**，且不得用 LLaMA 输出去训练**竞争性**模型（部分限制在后续版本放宽）。商用前务必读当代 License 原文。

### 6.2 它点燃了什么

LLaMA（尤其权重泄露 + 2 代商用）几乎以一己之力**启动了开源 LLM 大爆发**：

```
                    LLaMA Base
                        │
   ┌──────────┬─────────┼──────────┬───────────┐
  Alpaca    Vicuna   Code Llama   中文继训      量化/推理
 (指令微调) (对话)   (代码专精)  (Chinese-LLaMA) (llama.cpp/GGUF)
                        │
              成为微调/RAG/Agent 的默认底座
```

- **微调生态**：LoRA/QLoRA + LLaMA 几乎是开源微调的标准组合。
- **推理生态**：`llama.cpp`（CPU/端侧 GGUF 量化）、vLLM、TensorRT-LLM、Ollama 都把 LLaMA 当一等公民。
- **派生模型**：Alpaca、Vicuna、Code Llama、各种中文继训版、行业垂类模型，数量以万计。
- **架构事实标准**：RoPE+RMSNorm+SwiGLU+GQA 这套"LLaMA 配方"被后来者（包括 Qwen、Mistral 等）广泛沿用，HF 里大量模型直接复用 `modeling_llama.py` 路径。

## 7. 与同类横向对比

| 维度 | LLaMA (Meta) | Qwen (阿里) |
|------|--------------|-------------|
| 架构基底 | RoPE+RMSNorm+SwiGLU+GQA | 同款配方（受 LLaMA 启发） |
| 强项语言 | 英语为主，3 代起多语改善 | 中文/中英双强 |
| 形态 | 长期稠密为主 | 稠密 + MoE 双线 |
| 多模态 | 3.2 起补视觉 | 早期即有 VL 系列 |
| 协议 | 自定义社区许可(有大厂限制) | 多为 Apache 2.0(更宽松) |

要点：两者**架构同源**，差异主要在**数据语言侧重、是否走 MoE、协议宽松度**。中文场景常把 Qwen 作主选、LLaMA 作对照。详见 [[llm-algo/qwen/README]]。

## 8. 数值例子 / 手算

### 例 1：估算 7B 模型参数量

设 $d=4096$，层数 $L=32$，词表 $V=32000$。

- 词嵌入：$V\times d = 32000\times4096 \approx 1.31\times10^8$
- 每层注意力（Q/K/V/O 四个 $d\times d$ 矩阵，MHA）：$4d^2 = 4\times4096^2 \approx 6.71\times10^7$
- 每层 MLP（SwiGLU 三矩阵，中间维约 $\frac{8}{3}d\approx11008$）：$3\times d\times11008 \approx 1.35\times10^8$
- 每层合计 $\approx 2.0\times10^8$，乘 32 层 $\approx 6.5\times10^9$
- 加词嵌入与输出头 $\approx \mathbf{6.7\times10^9 \approx 7B}$ ✅ 与命名吻合。

### 例 2：推理显存（权重部分）

参数量 7B，不同精度每参数字节数不同：

$$\text{显存} \approx N \times \text{bytes/参数}$$

- FP16/BF16（2 字节）：$7\times10^9\times2 = 14\,\text{GB}$ → 单张 24G 卡可跑。
- INT8（1 字节）：约 **7 GB**。
- INT4（0.5 字节）：约 **3.5 GB** → 消费级显卡甚至高端手机可跑（这就是 `llama.cpp` 的魔力）。

### 例 3：GQA 省了多少 KV Cache

KV Cache 大小 $\approx 2 \times L \times n_{kv} \times d_{head} \times \text{seq} \times \text{bytes}$（2 是 K 和 V）。

以 70B 类配置：$L=80$，$d_{head}=128$，seq=4096，BF16(2B)：

- **MHA（64 KV头）**：$2\times80\times64\times128\times4096\times2 \approx 1.37\times10^{10}\,\text{B}\approx 13.7\,\text{GB}$
- **GQA（8 KV头）**：$\frac{8}{64}$ 倍 $\approx \mathbf{1.7\,\text{GB}}$

→ KV Cache 直接缩 **8 倍**，这就是长上下文（128K）能跑得动的关键。

## 9. 常见问题

| 问题 | 答案 |
|------|------|
| LLaMA 是 MoE 吗？ | 不是，主干是**稠密**模型（每个 token 走全部参数）。 |
| 为什么 7B 能打 175B 的 GPT-3？ | 缩放定律：参数小但**喂了几倍的 token**，训练更充分。 |
| RoPE 和绝对位置编码区别？ | RoPE 靠**旋转**注入位置，点积只依赖**相对距离**，外推更好。 |
| RMSNorm 比 LayerNorm 少了啥？ | 少了**减均值**和**偏置 β**，更快且同样稳。 |
| GQA 会掉点吗？ | 几乎不掉，但 KV Cache 缩 $h/g$ 倍，是性价比之选。 |
| LLaMA 2 可以随便商用吗？ | 可商用，但**月活>7亿大厂需授权**，且不得训竞品模型，须读 License。 |
| LLaMA 3 最大改动？ | 词表 32K→128K、数据 15T+、全系 GQA、3.1 上下文 128K + 开源 405B。 |
| 中文场景选谁？ | 中文优先常选 Qwen，LLaMA 作英文/架构对照。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图入口
- [[llm-algo/llama/模型架构]] — LLaMA 架构与 HF `modeling_llama.py` 逐类拆解（RoPE/Attention/MLP/DecoderLayer）
- [[llm-algo/qwen/README]] — 同源架构的中文强模型 Qwen 对照
- 代码参考：`transformers/models/llama/modeling_llama.py`（`LlamaRotaryEmbedding`/`LlamaAttention`/`LlamaMLP`/`LlamaDecoderLayer`/`LlamaModel`/`LlamaForCausalLM`）
