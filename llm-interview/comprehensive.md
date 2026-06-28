# 面试·综合大题
> 把"训练→推理"全链路、系统设计、scaling、长上下文、显存优化串成一张可背诵的答题地图。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-interview/base]] [[llm-interview/llm-algo]] [[llm-interview/llm-eval]]

## 阅读地图

| 节 | 主题 | 一句话 | 必背产出 |
|----|------|--------|----------|
| 0 | 一句话锚点 | 综合题考"全链路 + 数量级估算" | 4 个公式卡 |
| 1 | 地基 | Transformer 参数/FLOPs/显存三公式 | $N$、$6N$、$C=6ND$ |
| 2 | 显存优化 | 训练 OOM / 推理 OOM 的解法树 | 两棵解法树 |
| 3 | 训练全链路 | 数据→预训练→SFT→对齐→评估 | 五段流水线 |
| 4 | 推理全链路 | Prefill/Decode→KV Cache→批处理→并行 | 两阶段模型 |
| 5 | Scaling Law | Chinchilla 最优配比 $D\approx20N$ | 配比手算 |
| 6 | 长上下文 | 外推 + 稀疏/线性 + KV 压缩 | 三类方法 |
| 7 | 系统设计大题 | 给定约束做端到端方案 | STAR 答题框架 |
| 8 | 数值手算合集 | 7B 训练/推理全套数字 | 逐数表 |
| 9 | 高频追问 | 陷阱与踩点 | 问答表 |

## 0. 一句话锚点

综合大题 = **把一个开放约束（"给你 N 张卡，训/部署一个 M 参数的模型"）翻译成数量级估算 + 取舍决策**。
评分点只有三类：①能不能列出 **参数量/FLOPs/显存** 三个公式并代数；②能不能画出 **训练或推理的数据流**；③遇到瓶颈能不能给出 **解法树**（而不是只背一个名词）。

四张公式卡（先死记，后文逐一推）：

```
①参数量   N ≈ 12 · L · d²          (Transformer 主干, 忽略词表/偏置)
②训练FLOPs C ≈ 6 · N · D           (N=参数, D=训练token数)
③前向FLOPs/token ≈ 2N              (反向约2倍前向, 故 fwd+bwd=6N)
④KV Cache  = 2 · L · d · b · s · p_bytes  (2=K和V)
```

## 1. 地基：三条命脉公式（逐步推导）

### 1.1 参数量 $N$

一层 Transformer 的可训练权重分两块：

- **Attention**：$W_Q,W_K,W_V,W_O$ 各是 $d\times d$ → $4d^2$。
- **FFN**：升维 $d\to 4d$ 再降维 $4d\to d$ → $d\cdot4d + 4d\cdot d = 8d^2$。

单层合计 $4d^2+8d^2 = 12d^2$。$L$ 层：

$$N \approx 12\,L\,d^2$$

> 为什么忽略词表？嵌入矩阵 $V\times d$（如 $5\text{万}\times4096\approx2\text{亿}$）在 7B 里占比小；估算量级时可略，精算时加上。

**手算（7B 类配置，$L=32,\ d=4096$）**：
$$12\times32\times4096^2 = 384\times1.68\times10^7 \approx 6.44\times10^9 \approx 6.4\text{B}$$
加词表 $\approx 7$B，对得上。

### 1.2 训练计算量 $C=6ND$

每个 token 走一次前向：每个权重做一次"乘+加"，约 $2N$ FLOPs（乘法+加法各算 1）。反向传播要算激活梯度和权重梯度，约为前向的 2 倍 → $4N$。前向+反向 $= 6N$ FLOPs/token。训练 $D$ 个 token：

$$C \approx 6\,N\,D \ \text{(FLOPs)}$$

**手算**：7B 模型训 1.4T token → $6\times7\times10^9\times1.4\times10^{12}=5.88\times10^{22}$ FLOPs。
一张 A100 约 $312$ TFLOPS（BF16），实际利用率（MFU）约 40% → $1.25\times10^{14}$ FLOPS 有效。
单卡耗时 $=5.88\times10^{22}/1.25\times10^{14}=4.7\times10^8$ s $\approx 14.9$ 年 → 1024 卡 ≈ **5.3 天**。这是"几张卡训多久"类题的标准算法。

### 1.3 显存四大占用

训练显存 = **模型权重 + 梯度 + 优化器状态 + 激活值**。以 Adam + 混合精度为例（每参数字节）：

```
权重(fp16)        2 B
梯度(fp16)        2 B
Adam一阶动量(fp32) 4 B
Adam二阶动量(fp32) 4 B
权重fp32副本       4 B
---------------------------
合计              16 B/参数   ← "16倍参数量"是必背常数
```

**手算**：7B → $7\times10^9\times16 = 112$ GB（还不含激活）。一张 80GB A100 装不下，**所以必须并行/优化**——这就引出第 2 节。

## 2. 显存优化：两棵"解法树"（对应原文档的训练/推理）

### 2.1 训练 OOM 解法树

```
训练显存不够
├─ 砍"模型态"(权重+梯度+优化器=16N)
│   ├─ ZeRO-1 切优化器状态   → 16N → 8N + 8N/G
│   ├─ ZeRO-2 再切梯度       → 进一步均摊
│   ├─ ZeRO-3 再切权重(=FSDP)→ 每卡只 16N/G
│   └─ 张量/流水并行(TP/PP)  → 把单层也切到多卡
├─ 砍"激活态"
│   ├─ 梯度检查点(重计算)    → 激活 O(L)→O(√L), 换20%算力
│   ├─ 序列并行(SP)         → 沿seq维切LayerNorm/Dropout激活
│   └─ 更小micro-batch + 梯度累积 → 等效大batch不增显存
├─ 降精度
│   ├─ BF16/FP16混合精度    → 权重梯度2B
│   └─ 8-bit Adam          → 优化器状态 8N→2N
└─ Offload
    └─ ZeRO-Offload/Infinity → 优化器/参数下放CPU/NVMe
```

> ZeRO-3 显存 ≈ $16N/G + \text{激活}$。$G$=数据并行卡数。配 offload 可在少量 GPU 上训大模型，代价是 PCIe/NVMe 带宽变瓶颈。

**手算（7B，ZeRO-3，G=8）**：模型态 $112/8=14$ GB/卡，加激活+碎片约 20–30 GB/卡 → 单张 40GB 卡可训。

### 2.2 推理 OOM 解法树

```
推理显存不够 / 吞吐不够
├─ 砍权重
│   ├─ 量化 INT8/INT4(GPTQ/AWQ)  → 权重 2B→1B/0.5B
│   └─ 张量并行 TP               → 权重切到多卡
├─ 砍 KV Cache (长序列/大batch的真凶)
│   ├─ MQA/GQA                  → KV头数↓, cache↓数倍
│   ├─ KV量化(INT8/FP8)          → cache字节减半
│   ├─ PagedAttention(vLLM)     → 消除碎片, 利用率↑
│   ├─ 滑窗/StreamingLLM         → 只留近窗+attention sink
│   └─ KV淘汰(H2O/SnapKV)        → 丢低注意力token
└─ 调度
    ├─ Continuous batching       → 解码级动态拼批, 吞吐数倍
    ├─ Chunked prefill           → 长prompt切块, 不堵解码
    └─ PD分离                     → prefill/decode拆集群各自优化
```

**KV Cache 手算（关键）**：公式 $\text{KV}=2\cdot L\cdot d\cdot b\cdot s\cdot p$。
7B（$L=32,d=4096$），batch=1，序列 $s=2048$，FP16（2B）：
$$2\times32\times4096\times1\times2048\times2 = 2.1\times10^9\,\text{B}\approx 2\text{GB}$$
batch=32 → **64 GB**！这就是为什么大 batch 长序列必须 GQA + 分页 + 量化。
GQA 把 KV 头从 32 压到 8（1/4）→ KV 直接降到 16 GB。

## 3. 训练全链路（五段流水线 + 数据流）

```
原始语料 ──清洗/去重/质量分类──► 预训练数据(T级token)
                                      │ 自回归 next-token, C=6ND
                                      ▼
                                 Base 模型(会续写, 不会听话)
                                      │ SFT: (instruction, response) 对
                                      ▼
                                 SFT 模型(会按指令格式回答)
                                      │ 对齐: RM打分 → PPO/DPO/GRPO
                                      ▼
                                 对齐模型(有用+无害+诚实)
                                      │ 评测: MMLU/GSM8K/人类偏好/红队
                                      ▼
                                 上线(+持续RLHF/在线学习)
```

逐段要点（面试踩点）：

1. **数据**：去重（MinHash/SimHash）> 质量过滤 > 配比（代码/多语/数学提分）。"垃圾进垃圾出"，质量 > 数量。
2. **预训练**：损失 = 交叉熵 $-\sum \log p(x_t\mid x_{<t})$；监控 loss 平滑下降、无 spike；学习率 warmup + cosine 衰减。
3. **SFT**：只在 response 段算 loss（mask 掉 prompt）；样本要"少而精"，多样性 > 数量。
4. **对齐**：RM（奖励模型）学人类偏好；PPO 在线但难调，**DPO** 把 RL 转成分类损失更稳，**GRPO**（DeepSeek）去掉 value 网络省显存。可追问"为什么需要 KL 约束"——防止 reward hacking、防策略跑偏。
5. **评估**：见 [[llm-interview/llm-eval]]，区分自动指标 / 人类偏好 / 安全红队。

## 4. 推理全链路（两阶段模型 + 性能拆解）

推理分两个性质完全不同的阶段：

```
Prompt: "中国的首都是"  →  ┌──────────┐
                            │ Prefill  │ 并行处理全部prompt token
所有prompt token一次过      │ 算力受限 │ (compute-bound, GPU吃满)
                            └────┬─────┘ 产出首token + 写满KV Cache
                                 ▼
                            ┌──────────┐
"北"→"京"→"。"→<eos>        │  Decode  │ 逐token自回归
每步只算1个新token          │ 访存受限 │ (memory-bound, 搬KV为主)
                            └──────────┘ 每步读全部KV+权重
```

- **首 token 延迟 TTFT** 由 prefill 决定（算力）；**吐字速度 TPOT** 由 decode 决定（访存带宽）。
- decode 是 **memory-bound**：每生成 1 token 要把全部权重 + KV 从 HBM 搬一遍，算术强度低 → 优化方向是**减少搬运**（量化权重、压 KV）和**摊薄搬运**（批处理，让一次权重搬运服务多请求）。

**Roofline 手算**：decode 单 token 计算 $\approx2N=1.4\times10^{10}$ FLOPs，需搬权重 $14$ GB（fp16）。A100 带宽 $2$ TB/s → 搬运耗时 $14/2000=7$ ms，算力耗时 $1.4\times10^{10}/3.12\times10^{14}=0.045$ ms。**搬运是算力的 150 倍**，铁证 memory-bound → 上 INT4 把权重搬运减到 1/4，吐字快约 4 倍。

并行策略对照：

| 策略 | 切什么 | 通信 | 用途 |
|------|--------|------|------|
| 数据并行 DP | 复制整模型,切 batch | AllReduce 梯度 | 训练扩 batch |
| 张量并行 TP | 切单层矩阵(列/行) | 每层 AllReduce | 单机多卡放大模型 |
| 流水并行 PP | 切层(stage) | 点对点激活 | 跨机, 注意 bubble |
| 专家并行 EP | 切 MoE 专家 | All2All 路由 | MoE 大模型 |
| 序列并行 SP | 切 seq 维激活 | 配合 TP | 省激活显存 |

## 5. Scaling Law：算力一定怎么分配

核心问题：**固定算力预算 $C$，参数 $N$ 和数据 $D$ 怎么配最优？**
Kaplan(2020) 偏大模型；Chinchilla(2022) 修正为**两者等比扩**：

$$N_{opt}\propto C^{0.5},\quad D_{opt}\propto C^{0.5},\quad \boxed{D \approx 20\,N}$$

直觉：$C=6ND$ 是一条等算力双曲线，loss $L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}$ 在约束下的最优点要求两项边际收益相等 → token 数约为参数 20 倍。

**手算**：算力够训一个 Chinchilla-最优 70B → 需 $D\approx20\times70\text{B}=1.4$T token。
反过来 LLaMA 系列"**过度训练**"（小模型喂超量 token，如 7B 喂 2T，$D/N\approx285$）——牺牲训练最优换**推理便宜**，因为部署时小模型省显存省钱。这是高频追问点："Chinchilla 最优 ≠ 部署最优"。

## 6. 长上下文：三类正交手段

把"上下文从 4K 拉到 128K+"拆成三个独立问题：

### 6.1 位置外推（让模型"看得懂"远位置）

- **RoPE 旋转位置编码**：把位置信息编进 Q/K 的旋转角度，相对位置天然可外推。
- **NTK-aware / YaRN**：放大 RoPE 低频波长（"插值/外推混合"），训练 4K 推理 32K 不崩。
- **位置插值 PI**：把超长位置线性压回训练范围。

### 6.2 降低注意力复杂度（让模型"算得动"）

朴素注意力 $O(s^2)$，$s=128$K 时 $s^2=1.6\times10^{10}$，平方爆炸。

```
Full O(s²)        Sliding Window O(s·w)    Linear/SSM O(s)
■■■■              ■_ _ _                   状态递推, 无显式s×s矩阵
■■■■   →每个token  ■■_ _      →Mamba/RetNet  hₜ=A·hₜ₋₁+B·xₜ
■■■■    只看近w窗   _■■■_
■■■■               _ _■■
```

### 6.3 控制 KV Cache（让模型"放得下"）

长序列真正的显存杀手是 KV（见 2.2 手算，128K 单序列 KV 就 ~128GB 量级）。手段：GQA、KV 量化、StreamingLLM（attention sink + 滑窗）、H2O/SnapKV（按注意力分淘汰）。

> 答题诀窍：被问"如何支持长上下文"，**一定分这三层**回答（看得懂 / 算得动 / 放得下），而不是只甩一个 RoPE。

## 7. 系统设计大题：STAR 答题框架

遇到"设计一个支持 X 的 LLM 服务/训练系统"，按 **约束→估算→架构→瓶颈→取舍** 五步走：

```
①澄清约束  模型多大? QPS? 延迟SLA(TTFT/TPOT)? 上下文长度? 预算/卡数?
②数量级估算 用§1四公式: 显存够吗? 单卡放得下吗? KV多大? 吞吐上限?
③画架构    数据流图: 网关→路由→prefill集群→decode集群→KV存储
④定瓶颈    decode访存? KV显存? 长尾延迟? 用Roofline/§2解法树定位
⑤取舍      列2-3个方案对比(成本/延迟/质量), 给推荐 + 理由
```

**例题：设计支持 70B、1000 QPS、TTFT<1s、32K 上下文的推理服务。**
- 估算：70B 权重 fp16=140GB → 单卡放不下 → **TP=4**（A100 80G）或 INT8 量化到 ~70GB+TP=2。
- KV：32K×batch 很大 → 必上 **GQA + PagedAttention + KV INT8**。
- 吞吐：1000 QPS 靠 **continuous batching** 摊薄；长 prompt 用 **chunked prefill** 防堵。
- 架构：**PD 分离**——prefill 集群（算力型卡）与 decode 集群（带宽型卡）分开扩缩容。
- 取舍：量化省卡但掉点（评测验证 <1% 才上）；副本数按 P99 延迟和峰值 QPS 定。

## 8. 数值手算合集（7B 全套，可直接背）

| 量 | 公式 | 代数 | 结果 |
|----|------|------|------|
| 参数量 | $12Ld^2$ | $12\cdot32\cdot4096^2$ | ≈6.4B(+词表≈7B) |
| 训练显存(Adam混合精度) | $16N$ | $16\times7\text{B}$ | 112 GB(不含激活) |
| 训练算力(1.4T tok) | $6ND$ | $6\cdot7\text{B}\cdot1.4\text{T}$ | $5.9\times10^{22}$ FLOPs |
| 单卡训练时长 | $C/(\text{TFLOPS}\cdot\text{MFU})$ | $/312\text{T}/0.4$ | ≈15年(1024卡≈5天) |
| 推理权重(fp16) | $2N$ | $2\times7\text{B}$ | 14 GB |
| 推理权重(INT4) | $0.5N$ | $0.5\times7\text{B}$ | 3.5 GB |
| KV/序列(s=2048,fp16) | $2Ldsp$ | 见§2.2 | ≈2 GB |
| KV(GQA,头1/4) | 上×1/4 | | ≈0.5 GB |
| decode访存:算力 | 带宽/算力roofline | 7ms : 0.045ms | ≈150:1 |
| Chinchilla最优tok | $20N$ | $20\times7\text{B}$ | 140B tok |

> 背这张表，几乎所有"估算类"综合题都能现场推。关键是 **记公式不记结果**，到题里换数就行。

## 对照/复杂度表

| 维度 | 训练 | 推理-Prefill | 推理-Decode |
|------|------|-------------|-------------|
| 瓶颈性质 | 算力(大batch) | 算力 | 访存带宽 |
| 显存大头 | 优化器+激活 | 权重+KV写入 | 权重+KV读取 |
| 关键优化 | ZeRO/重计算/混合精度 | chunked prefill | 量化+GQA+批处理 |
| 复杂度 | $O(6ND)$ | $O(s^2 d)$/序列 | $O(s d)$/步 |
| 衡量指标 | MFU/吞吐 | TTFT | TPOT/吞吐 |

## 常见问题/高频追问

| 问题 | 踩点答案 | 陷阱/追问 |
|------|----------|----------|
| 训练 7B 要多少显存 | 16N≈112GB+激活 | 别忘激活和碎片；问优化器拆解 |
| 为什么 decode 慢 | memory-bound，每步搬全部权重+KV | 追问 roofline/算术强度 |
| KV Cache 怎么算 | $2Ldbsp$，逐数代入 | 易漏"2"(K和V)和batch维 |
| 长上下文怎么做 | 看得懂/算得动/放得下三层 | 只答 RoPE 会被追问显存 |
| ZeRO 三级区别 | 切优化器→梯度→权重 | ZeRO-3≈FSDP；问通信代价 |
| Chinchilla 最优是部署最优吗 | 不是，过训小模型省推理 | $D/N$=20 vs LLaMA~285 |
| 大 batch 推理为何能提吞吐 | 摊薄权重搬运(memory-bound) | 但增 TTFT，trade-off |
| PPO vs DPO vs GRPO | 在线RL/转分类/去value网 | 为何要 KL 约束 |
| MFU 是什么 | 有效FLOPs/峰值，常0.3-0.5 | 通信/bubble/访存拉低它 |
| 量化掉点怎么办 | AWQ保护显著权重+评测把关 | 权重vs激活vsKV量化区别 |
| Prefill 堵 decode 怎么破 | chunked prefill + PD分离 | 长尾延迟与公平性 |
| TP 和 PP 怎么选 | TP机内(高带宽NVLink)，PP跨机 | PP有bubble，需1F1B调度 |

## 🔗 跳转链接

> 🔗 相关：[[00-知识地图]] [[llm-interview/base]] [[llm-interview/llm-algo]] [[llm-interview/llm-eval]]
