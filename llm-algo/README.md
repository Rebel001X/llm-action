# 大模型算法与架构超参数解读（LLM-Algo）

> 一句话定位：从 `config.json` 的每一个超参数出发，反推 Transformer 解码器的形状、显存与算力，把 GPT2 / Bloom / LLaMA / ChatGLM 放在同一张表里横向对比。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]] · [[docs/transformer内存估算]]

## 阅读地图

| 你想知道的 | 看哪一节 | 一句话回答 |
| --- | --- | --- |
| 这些超参数到底是什么 | §1 地基 | 词表、层数、头数、隐藏维、FFN 维、序列长度六大旋钮 |
| 为什么 FFN 通常是 hidden 的 4 倍 | §2 FFN | GPT 系传统比例；LLaMA 用 SwiGLU 后约 2.7×4096≈11008 |
| MHA / MQA / GQA 区别 | §3 KV 头数 | `num_key_value_heads` 一个参数决定三种注意力 |
| 怎么估参数量/显存 | §4 数值手算 | 参数≈12·L·h² 量级；KV Cache 由 kv 头数决定 |
| 各模型具体数字 | §5 模型对比表 | 原文两张真实对照表 |
| 配置从哪来 | §6 配置来源 | 原文 HuggingFace config.json 链接 |
| 易踩的坑 | 常见问题 | seq_length vs max_position、n_inner 默认值等 |

## 0. 一句话锚点

**一个大模型的“身材”由 `config.json` 里六个超参数完全确定**：词表大小 `vocab_size`、层数 `num_hidden_layers`、注意力头数 `num_attention_heads`、KV 头数 `num_key_value_heads`、隐藏维 `hidden_size`、FFN 中间维 `intermediate_size`，再加一个上下文长度 `max_position_embeddings`。读懂这七个数字，就能反推参数量、显存、KV Cache 与算力，无需打开模型权重。

可视化（强烈推荐边读边看，把下面的数字“点亮”成立体结构图）：

- https://bbycroft.net/llm
- https://zhuanlan.zhihu.com/p/644815089

## 1. 地基：七个旋钮决定一个 Decoder

现代 LLM（GPT/LLaMA/Bloom/ChatGLM）都是 **Decoder-Only Transformer**：把 N 个相同的 Block 堆叠起来。一个 Block 的形状由下面这些数字决定。

```
                输入 token ids  (batch, seq_length)
                        │
              ┌─────────▼─────────┐
              │ Embedding         │  形状: [vocab_size, hidden_size]
              └─────────┬─────────┘
                        │  x: (batch, seq, hidden)
   ┌────────────────────▼────────────────────┐
   │  ×  num_hidden_layers 个相同的 Block     │
   │  ┌───────────────────────────────────┐  │
   │  │ Self-Attention                    │  │
   │  │   Q: hidden→hidden  (n_head 份)   │  │
   │  │   K,V: hidden→ kv 维 (kv_head 份) │  │
   │  └───────────────┬───────────────────┘  │
   │  ┌───────────────▼───────────────────┐  │
   │  │ FFN: hidden→intermediate→hidden   │  │
   │  └───────────────────────────────────┘  │
   └────────────────────┬────────────────────┘
                        │
              ┌─────────▼─────────┐
              │ LM Head           │  形状: [hidden_size, vocab_size]
              └───────────────────┘
                        │
                logits (batch, seq, vocab_size)
```

七个旋钮逐一拆解：

| 超参数（别名） | 含义 | 直觉 |
| --- | --- | --- |
| `vocab_size` | 词表大小 | 分词器能识别多少个 token；决定 Embedding/LM Head 大小 |
| `num_hidden_layers`（n_layer, num_layers） | Block 堆叠层数 | 越深越能表达复杂函数，但越难训、越慢 |
| `num_attention_heads`（n_head） | 注意力头数 | 把 hidden 切成多少个“视角”并行做 attention |
| `num_key_value_heads` | KV 头数 | 见 §3，决定 MHA/MQA/GQA |
| `hidden_size`（n_embd, n_embed） | 隐藏维 = 残差流宽度 | 模型的“主干带宽”，一切张量的核心维度 |
| `intermediate_size`（ffn_hidden_size, n_inner） | FFN 中间层维度 | FFN 的“放大-压缩”宽度，通常远大于 hidden |
| `max_position_embeddings`（n_positions, n_ctx, seq_length） | 最大上下文长度 | 一次能看多少 token |

**为什么要关心这些**：训练/推理框架（Megatron、DeepSpeed、vLLM）启动时都从 `config.json` 读这些值来切张量并行、分配显存、建 KV Cache。读不懂 config 就调不动框架。

## 2. FFN 宽度：为什么是 4 倍，又为什么 LLaMA 是 11008

经典 Transformer（GPT2/Bloom）里 **FFN 中间维 = 4 × hidden_size**。原文表里写得很直白：

- GPT2 Medium：`ffn = 4*n_embd = 4*1024 = 4096`
- Bloom-7b1：`ffn = 4 * hidden_size = 4*4096 = 16384`

```
hidden=4096 ──W1(4096→16384)──> 16384 ──激活──> 16384 ──W2(16384→4096)──> 4096
              先放大 4 倍                                  再压回来
```

**为什么放大**：FFN 是 Transformer 里真正“存知识”的地方，更宽的中间层 = 更大的 key-value 记忆容量。4× 是经验上算力/效果的甜点。

**LLaMA 为什么是 11008 而不是 16384**：LLaMA 用了 **SwiGLU**（门控激活），它需要三个矩阵（gate / up / down）而不是两个。为保持总算力与 4× 标准 FFN 相当，把中间维缩小为约 `4 × 2/3 = 8/3 ≈ 2.667` 倍，并对齐到 256 的倍数：

$$\text{intermediate} \approx \frac{2}{3}\times 4 \times h = \frac{8}{3}\times 4096 \approx 10923 \xrightarrow{\text{对齐到256倍数}} 11008$$

所以 LLaMA-7B 表里 `intermediate_size=11008`、ChatGLM2 是 `13696`，都不是整 4 倍——看到非整倍数，基本就是门控 FFN 的指纹。

## 3. KV 头数：一个参数玩转 MHA / MQA / GQA

这是原文最有价值的一段说明（**原文保留**）：

> key_value 头数：This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group.

翻译成一张决策表：

| `num_key_value_heads` 取值 | 注意力类型 | KV Cache 大小 | 代表模型 |
| --- | --- | --- | --- |
| `= num_attention_heads` | MHA（多头） | 最大（每个 Q 头独享 K/V） | LLaMA-7B（32=32）、GPT2、Bloom |
| `= 1` | MQA（多查询） | 最小（所有 Q 头共享 1 组 K/V） | PaLM、Falcon 部分版本 |
| `1 < kv < heads` | GQA（分组查询） | 居中 | LLaMA-2-70B（8 组）、ChatGLM2 |

```
 MHA (kv=heads=8)        GQA (kv=2)             MQA (kv=1)
 Q0 Q1 ... Q7            Q0..Q3   Q4..Q7        Q0 Q1 ... Q7
 │  │      │              \  |  /   \  |  /       \  \    /  /
 K0 K1 ... K7              KV组0     KV组1          单组 KV
 8 套 K/V                 2 套 K/V                1 套 K/V
 KV Cache 最大            KV Cache 1/4            KV Cache 1/8
```

**为什么要 GQA/MQA**：推理时 **KV Cache 显存正比于 KV 头数**。长上下文场景下 KV Cache 往往比模型权重还大，把 32 个 KV 头压成 8 组（LLaMA-2-70B）能把这部分显存砍到 1/4，几乎不掉精度。这正是 §4 显存估算的关键变量。

**原文表里的细节**：原文“模型对比”表里 GPT2/Bloom/LLaMA-7B 的 `num_key_value_heads` 写 N/A（早期 config 没这个字段，等价 MHA）；而“LLaMA”表里 LLaMA-2-7B 明确标 `32`（=heads，即 MHA）、LLaMA-2-70B 标 `8`（GQA 8 组）。看到 70B 用 8、7B 用 32，就是 GQA 只对大模型才划算的体现。

转换技巧（原文）：把 MHA 权重转成 GQA 时，每个分组的 K/V 头由组内原始头 **mean-pooling（取平均）** 得到，便于无损初始化再微调。

## 4. 数值手算：从 config 反推参数量与 KV Cache

**(a) 参数量量级估算**（忽略 embedding/bias，只看主干）：每层参数约 `12·h²`（attention 4h² + FFN 8h²，标准 4× FFN）。

以 LLaMA-7B 为例：`L=32, h=4096`

$$N \approx 12 \times L \times h^2 = 12 \times 32 \times 4096^2 \approx 6.44\times 10^9 \approx 6.4\text{B}$$

再加上 Embedding `vocab·h = 32000×4096 ≈ 0.13B` 与 LM Head 共享/不共享，凑到约 6.7B——和“7B”吻合。（LLaMA 用 SwiGLU 三矩阵，FFN 实际略大，结果更接近 6.7B。）

**(b) KV Cache 显存**（推理时每个 token 占用），fp16=2 字节：

$$\text{KV Cache} = 2 \times L \times \underbrace{(\text{kv\_heads}\times d_{head})}_{\text{kv 维}} \times 2\text{字节} \times \text{seq} \times \text{batch}$$

LLaMA-2-7B（MHA，kv=32，d_head=128，L=32），单条 seq=2048：

$$2 \times 32 \times (32\times128) \times 2 \times 2048 \approx 1.07\text{ GB / 条}$$

若换成 GQA kv=8（如 70B 的策略），这一项直接降到 **1/4**。这就是为什么长上下文一定要 GQA。详见 [[llm-optimizer/kv-cache]]。

**(c) 单 token 前向算力**：约 `2N` FLOPs（N=参数量）。7B 模型生成 1 个 token ≈ 14 GFLOPs，再乘以你要生成的 token 数即可估推理成本。

## 5. 模型对比（原文真实数据，原样保留）

### 5.1 跨家族对比

| 模型 | GPT2 Medium（345M） | Bloom-7b1 | LLaMA-7B | LLaMA2-7B | ChatGLM-6B | ChatGLM2-6B |
| --- | --- | --- | --- | --- | --- | --- |
| 词表大小（vocab_size） | 50257 | 250880 | 32000 | 32000 | 130528 | 65024 |
| Transformer层（n_layer, num_layers, num_hidden_layers） | 24 | 30 | 32 | 32 | 28 | 28 |
| 注意力头数（num_attention_heads, n_head） | 16 | 32 | 32 | 32 | 32 | 32 |
| key_value头数（num_key_value_heads） | N/A | N/A | N/A | N/A | N/A | N/A |
| 隐藏层大小（hidden_size） | 1024(n_embd) | 4096(n_embed) | 4096 | 4096 | 4096 | 4096 |
| 前馈神经网络的隐藏层大小（ffn_hidden_size, intermediate_size,n_inner） | 4*n_embd | 4 * hidden_size | 11008 | 11008 | 16384 | 13696 |
| seq_length, n_ctx | 1024 | 2048 | 2048(max_position_embeddings) | 2048(max_position_embeddings) | 2048 | 32768 |
| n_positions,max_position_embeddings,n_embed | 1024(default) | 2048(4096,bloomz-7b1-hf) | 2048 | 2048(4096,llama2-chat-hf) | hidden_size | hidden_size |

**读表小贴士**：

- **词表差异巨大**：Bloom 是多语言模型，词表 25 万；GPT2 英文为主 5 万；ChatGLM 中英混合 13 万/6.5 万。词表越大，Embedding 与 LM Head 越占参数。
- **ChatGLM2 上下文 32768**：相比 ChatGLM 的 2048 大幅扩展，靠 RoPE + 长度外推，见 [[llm-algo/旋转编码RoPE]]。
- **FFN 列看激活函数**：写 `4*n_embd` 的是经典 FFN；写 11008/13696 的是 SwiGLU 门控（§2）。

### 5.2 LLaMA 家族纵向缩放

| 模型 | LLaMA-7B | LLaMA-2-7B | LLaMA-13B | LLaMA-2-13B | LLaMA-30B | LLaMA-65B | LLaMA-2-70B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 词表大小（vocab_size） | 32000 | 32000 | 32000 | 32000 | 32000 | 32000 | 32000 |
| Transformer层（n_layer, num_layers, num_hidden_layers） | 32 | 32 | 40 | 40 | 60 | 80 | 80 |
| 注意力头数（num_attention_heads, n_head） | 32 | 32 | 40 | 40 | 52 | 64 | 64 |
| key_value头数（num_key_value_heads） | N/A | 32 | N/A | 40 | N/A | N/A | 8 |
| 隐藏层大小（hidden_size） | 4096 | 4096 | 5120 | 5120 | 6656 | 8192 | 8192 |
| 前馈神经网络的隐藏层大小（ffn_hidden_size, intermediate_size,n_inner） | 11008 | 11008 | 13824 | 13824 | 17920 | 22016 | 28672 |
| seq_length, n_ctx | 2048(max_position_embeddings) | 2048(max_position_embeddings) | 2048 | N/A | 2048 |  | N/A |
| n_positions,max_position_embeddings,n_embed | 2048 | 2048(4096,llama2-chat-hf) | N/A | 4096 | N/A | N/A | 4096 |

**缩放规律（怎么把 7B 放大到 70B）**：

```
 规模 ↑   层数 L ↑   头数 ↑   隐藏维 h ↑   FFN ↑     KV头数
 7B       32        32       4096        11008     32 (MHA)
 13B      40        40       5120        13824     40 (MHA)
 70B      80        64       8192        28672     8  (GQA!)
          ↑ 同步增大深度与宽度          ↑ 词表恒定 32000
```

- **词表恒定 32000**：同一分词器贯穿整个家族，方便权重迁移与蒸馏。
- **深度与宽度协同放大**：层数 32→80、隐藏维 4096→8192，符合“计算最优”缩放律。
- **只有 70B 用 GQA(kv=8)**：印证 §3——大模型 KV Cache 压力大，GQA 收益最高。
- **head_dim 始终≈128**：`hidden/heads = 4096/32 = 5120/40 = 8192/64 = 128`，刻意保持每头维度恒定，利于硬件高效。

## 6. 实操：配置文件来源（原文链接，原样保留）

所有上表数字均可在 HuggingFace 的 `config.json` 中逐项核对（这是“不编造”的根据）：

- https://huggingface.co/gpt2-medium/resolve/main/config.json
- https://huggingface.co/bigscience/bloom-7b1/blob/main/config.json
- https://huggingface.co/bigscience/bloomz-7b1-mt/blob/main/config.json
- https://huggingface.co/yahma/llama-7b-hf/blob/main/config.json
- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json
- https://huggingface.co/THUDM/chatglm2-6b-32k
- https://huggingface.co/THUDM/chatglm-6b
- https://huggingface.co/decapoda-research/llama-13b-hf

**核对工作流**（自己验证一个模型的形状）：

1. 打开对应 `config.json`，找 `hidden_size` / `num_hidden_layers` / `num_attention_heads` / `num_key_value_heads` / `intermediate_size` / `max_position_embeddings`。
2. 用 §4 公式 `N≈12·L·h²` 估参数量，与模型名里的 B 数对比。
3. 看 `num_key_value_heads` 是否等于 `num_attention_heads`，判定 MHA/GQA/MQA。
4. 看 `intermediate_size` 是否是 hidden 的整 4 倍，判定是否门控 FFN。

**原文说明（保留）**：

- 通常 `seq_length` 与 `max_position_embeddings` 相等。
- `key_value头数` 字段的完整定义见 §3 引用。

## 常见问题 / 坑

| 坑 | 现象 | 原因 / 正解 |
| --- | --- | --- |
| seq_length 与 max_position 混淆 | 估 KV Cache 用错长度 | 二者通常相等（原文已说明），但部分模型（如 bloomz-7b1-hf）config 里写 4096 而原始 2048，以实际 config 为准 |
| N/A 当成“没有 KV 头” | 误判注意力类型 | 早期 config 无 `num_key_value_heads` 字段，缺省即 `=num_attention_heads`，是 MHA 不是“没有” |
| FFN 用错倍数 | 参数量估算偏大 | LLaMA/ChatGLM2 是门控 FFN，约 8/3·4 倍且对齐 256，不是整 4 倍（§2） |
| `n_inner` 默认 None | GPT2 系算 FFN 卡住 | GPT2 config 里 `n_inner` 默认 None，实际取 `4*n_embd`（原文已注明） |
| 忽略词表对参数量的影响 | 小模型估算误差大 | Bloom 词表 25 万、ChatGLM 13 万，Embedding+LMHead 占参数显著，不能只算主干 |
| GQA 转换乱拼 K/V | 转换后精度暴跌 | 须按组对原始头 **mean-pooling**（原文转换说明），不是随机丢弃 |
| 把 30B/65B 当存在 LLaMA-2 版本 | 找不到权重 | 原文表中 LLaMA-30B/65B 属 LLaMA-1；LLaMA-2 只有 7B/13B/70B |

## 🔗 跳转链接

- 知识枢纽：[[00-知识地图]]
- 架构原理：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]] · [[docs/transformer内存估算]]
- 注意力优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理部署：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 压缩量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 硬件与网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
