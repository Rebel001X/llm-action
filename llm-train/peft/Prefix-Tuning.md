# Prefix-Tuning / P-Tuning
> 在每个 Transformer 层的 K/V 前面拼接一小段「可训练的虚拟 token」，冻结整个主干，只训这几个前缀向量。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-train/peft/Prompt-Tuning]] [[llm-train/peft/PEFT-API]]

## 阅读地图

| 节 | 你会得到什么 | 适合谁 |
|----|--------------|--------|
| 0  | 一句话锚点 | 想 30 秒抓住本质 |
| 1  | 地基：注意力里的 K/V、冻结主干是什么意思 | 忘了 Attention 的人 |
| 2  | 核心机制：前缀如何进每一层的注意力（ASCII） | 想搞懂数据流 |
| 3  | 手算：前缀如何改变 softmax 与输出 | 想要逐数验证 |
| 4  | reparameterization（MLP 重参数化）为什么必须 | 训练不收敛的人 |
| 5  | 参数量逐数计算 + 显存账 | 要估算资源 |
| 6  | Prompt-Tuning / Prefix-Tuning / P-Tuning v1 / v2 谱系 | 容易搞混的人 |
| 7  | 与全量微调对比 | 选型决策 |
| 8  | 代码：PEFT 里的 PrefixEncoder | 上手实现 |
| 9  | 面试问答清单 | 备面 |
| 表 | 对照/复杂度表 + 高频追问 | 速查 |

## 0. 一句话锚点

> **全量微调改的是「权重 $W$」，Prefix-Tuning 改的是「输入给注意力的上下文」。**
> 它在每一层的 Key 和 Value 序列最前面，偷偷插进去 $L$ 个**凭空学出来的**向量（不对应任何真实词），让后面所有真实 token 都能「注意到」这些前缀，从而把模型的行为往任务方向掰。主干一个参数都不动。

类比：你不能改老师的大脑（冻结主干），但你可以在每节课开头塞给他一张**只有他能看到的小抄**（前缀）。小抄是你反复打磨学出来的，老师每次回答前都会瞄一眼。

三层递进，记住这三句就够：
1. **离散 prompt（人写的提示词）** → 受限于词表里真实存在的词，调起来像「猜口令」。
2. **soft prompt（连续可学的虚拟 token）** → 不再受词表约束，但 Prompt-Tuning 只在输入层加，信号传到深层会衰减。
3. **deep prompt（每层都加，即 Prefix / P-Tuning v2）** → 在每一层注入，表达力最强，这是本文主角。

## 1. 地基/前置

### 1.1 注意力里到底有哪些矩阵
一个标准自注意力层，输入是序列 $X \in \mathbb{R}^{n \times d}$（$n$ 个 token，每个 $d$ 维）。它做三件事：

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V \qquad (W_Q,W_K,W_V \in \mathbb{R}^{d\times d})$$

然后：

$$\text{Attn}(Q,K,V) = \underbrace{\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)}_{\text{注意力权重 }A,\ n\times n} V$$

- $Q$（query）：「我想找什么」
- $K$（key）：「我有什么标签」
- $V$（value）：「我携带的内容」
- 第 $i$ 个 token 的输出 = 它的 query 和**所有** key 算相似度 → softmax 成权重 → 对**所有** value 加权求和。

**关键洞察**：序列里每多一个 (key, value) 对，每个真实 token 的输出就会被它影响。Prefix-Tuning 就是钻这个空子。

### 1.2 「冻结主干」是什么意思
冻结 = `requires_grad = False`。所有 $W_Q, W_K, W_V, W_O$、FFN、Embedding、LayerNorm 全部不更新，前向照常算，反向时**梯度流过它们但不在它们上累积**，只更新前缀参数。这意味着：
- 优化器状态（Adam 的 m、v）只为前缀那一点参数分配 → 省显存。
- 一个底座模型可以挂多套前缀，按任务热插拔。

## 2. 核心机制：前缀如何进每一层

Prefix-Tuning 为**每一层** $\ell$ 准备两个可训练张量：
$$P^{(\ell)}_K \in \mathbb{R}^{L \times d},\qquad P^{(\ell)}_V \in \mathbb{R}^{L \times d}$$
其中 $L$ = 前缀长度（虚拟 token 数，常见 10/20/30）。前向时，把它们**拼到当前层算出的 K、V 前面**：

$$\tilde K = [\,P^{(\ell)}_K\,;\,K\,]\in\mathbb{R}^{(L+n)\times d},\qquad \tilde V = [\,P^{(\ell)}_V\,;\,V\,]\in\mathbb{R}^{(L+n)\times d}$$

注意：**Query 不加前缀**。前缀只扩充被「注意」的对象，不新增「发问者」，所以输出序列长度还是 $n$，不影响下游形状。

```
            真实输入 token：  x1  x2  x3 ... xn
                              │   │   │       │
        ┌─────────────────────────────────────────────┐
 第ℓ层  │  Q = X·Wq      (只来自真实 token, L 个前缀不进 Q) │
        │                                               │
        │  K = [ Pk(ℓ) | x1Wk x2Wk ... xnWk ]           │
        │        └─前缀─┘ └────── 真实 ──────┘            │
        │  V = [ Pv(ℓ) | x1Wv x2Wv ... xnWv ]           │
        │                                               │
        │  Attn 权重矩阵 A 形状: n × (L+n)               │
        │      每个真实 query 都能看到 L 个前缀 + n 个真实   │
        └─────────────────────────────────────────────┘
                              │
                          第ℓ层输出 (长度仍为 n)
                              │  ↓ 进入第 ℓ+1 层，重复（每层有独立前缀）
```

**「每层都加」是 Prefix-Tuning 区别于 Prompt-Tuning 的命门**：
- Prompt-Tuning：只在**最底层 embedding** 前面拼几个软 token，靠它们一路传上去影响。表达力弱，模型小的时候几乎学不动。
- Prefix-Tuning / P-Tuning v2：**每一层都有自己的前缀**，直接注入到该层的 K/V，相当于在深处不断「续写小抄」，表达力强得多。

## 3. 手算：前缀如何改变 softmax 与输出

设一个迷你层：$d=2$，前缀长度 $L=1$，真实 token 数 $n=2$。某个真实 query $q=[1,\,0]$。

**未加前缀时**，假设两个真实 key：
$$k_1=[1,0],\ k_2=[0,1]$$
打分（先不除 $\sqrt{d}$ 简化）：$q\cdot k_1 = 1,\ q\cdot k_2 = 0$。
softmax：$\dfrac{[e^1,\,e^0]}{e^1+e^0}=\dfrac{[2.718,\,1]}{3.718}=[0.731,\,0.269]$。
设 $v_1=[10,0],\ v_2=[0,10]$，输出 $=0.731\cdot[10,0]+0.269\cdot[0,10]=[7.31,\,2.69]$。

**加入一个前缀** $p_K=[1,0]$（学出来的，恰好和 $q$ 同向）、$p_V=[0,100]$（携带很强的「方向 2」信号）：
现在 key 序列 = $[p_K, k_1, k_2]$，打分 $= [q\cdot p_K,\ 1,\ 0] = [1,\,1,\,0]$。
softmax：$\dfrac{[e^1, e^1, e^0]}{e^1+e^1+e^0}=\dfrac{[2.718,2.718,1]}{6.436}=[0.4224,\,0.4224,\,0.1554]$。
输出 $=0.4224\cdot[0,100]+0.4224\cdot[10,0]+0.1554\cdot[0,10]$
$=[4.224,\ 42.24+1.554]=[4.224,\,43.79]$。

**对比**：输出从 $[7.31, 2.69]$ 被前缀一把拽到了 $[4.22, 43.79]$——「方向 2」被前缀的高 value 强行放大。这说明：
1. **前缀的 key**（$p_K$）决定「真实 token 多大程度去看这条小抄」（抢 softmax 概率质量）。
2. **前缀的 value**（$p_V$）决定「看了之后被注入什么内容」。
3. 主干一个权重没变，仅靠注入 $(p_K,p_V)$ 就改写了输出分布——这就是 Prefix-Tuning 的全部魔法。

## 4. reparameterization（重参数化）：为什么不能直接训前缀

原始论文发现：**直接把 $P_K, P_V$ 当自由参数用梯度去优化，训练极不稳定、对学习率敏感、性能差**。原因：前缀向量维度高、相互之间又要协调，直接优化的损失面很崎岖。

解决：不直接学 $P$，而是学一个**低维种子** $P'\in\mathbb{R}^{L\times d'}$，再用一个小 MLP 把它「膨胀」出来：

$$P^{(\ell)}_{K},\,P^{(\ell)}_{V} = \text{MLP}_\theta\big(P'\big),\qquad \text{MLP}: \mathbb{R}^{d'} \to \mathbb{R}^{2 \cdot \text{num\_layers}\cdot d}$$

```
   小种子 P'              重参数化 MLP                  各层前缀
 [L × d']  ──►  Linear(d'→H) ─ Tanh ─ Linear(H→2·num_layers·d)  ──► reshape
                                                                  ┌ Pk(1),Pv(1)
                                                                  ├ Pk(2),Pv(2)
                                                                  ├   ...
                                                                  └ Pk(N),Pv(N)
```

- MLP 提供了**参数共享 + 平滑约束**，让优化好走。
- **训练时**才需要这个 MLP；**推理时**可以把它跑一遍、把生成的 $P$ 缓存下来，丢掉 MLP（所以推理零额外计算开销，只多了 $L$ 个 K/V）。
- P-Tuning v1 用的是 **LSTM + MLP** 当 prompt encoder；Prefix-Tuning 用 MLP；P-Tuning v2 实测发现**重参数化在大模型/某些任务上未必更好**，干脆做成可选项。

## 5. 参数量逐数计算 + 显存账

记号：层数 $N$、隐藏维 $d$、前缀长 $L$。**可训练前缀参数**（不含训练期 MLP）：

$$\#\text{params} = \underbrace{2}_{K\text{ 和 }V} \times N \times L \times d$$

**逐数代入**（以 GPT-2 medium：$N=24,\ d=1024,\ L=20$）：
$$2 \times 24 \times 20 \times 1024 = 983{,}040 \approx 0.98\text{M}$$

GPT-2 medium 总参数约 **355M**，于是占比：
$$\frac{0.98\text{M}}{355\text{M}} \approx 0.28\%$$

换 LLaMA-7B（$N=32,\ d=4096,\ L=20$）：
$$2 \times 32 \times 20 \times 4096 = 5{,}242{,}880 \approx 5.24\text{M}\ \Rightarrow\ \frac{5.24}{7000}\approx 0.075\%$$

**显存账（粗算，fp16，每参数 2 字节；Adam 还要 m、v 两份 fp32 各 4 字节 + master 4 字节）**：
- 全量微调 7B：优化器状态 ≈ $7\text{B}\times(4+4+4)\text{B} = 84\text{GB}$（外加权重+激活，单卡放不下）。
- Prefix 7B：可训练仅 5.24M，优化器状态 ≈ $5.24\text{M}\times 12\text{B}\approx 63\text{MB}$。**冻结主干的权重仍要常驻**（约 14GB fp16），但**梯度/优化器状态从 84GB 级降到几十 MB**，这才是省显存的真相——省的是状态，不是底座。

## 6. 谱系：Prompt-Tuning / Prefix-Tuning / P-Tuning v1 / v2

```
              加在哪              加几层      训练辅助           典型场景
Prompt-Tuning  embedding 前        仅第1层    无                超大模型+生成
Prefix-Tuning  每层 K/V 前         每一层     MLP 重参数化       NLG(生成)
P-Tuning v1    embedding 前(连续)  仅第1层    LSTM/MLP encoder   NLU(理解,GPT做分类)
P-Tuning v2    每层 K/V 前         每一层     可选重参数化        NLU 全规模(对标微调)
```

- **Prompt-Tuning（Lester 2021）**：最简，只在输入 embedding 拼软 token，规模够大（10B+）才追平微调。
- **P-Tuning v1（Liu 2021）**：为「让 GPT 做 NLU」而生，把离散 prompt 换成连续可学的虚拟 token，用 LSTM 编码它们之间的依赖；但只加在输入层，**中小模型、难任务（序列标注）上掉点**。
- **Prefix-Tuning（Li & Liang 2021）**：面向**生成（NLG）**，每层注入 K/V 前缀 + MLP 重参数化。
- **P-Tuning v2（Liu 2022）**：把 Prefix-Tuning 的「每层前缀」思想搬到 **NLU**，去掉对 verbalizer 的依赖（改用普通分类头），在 **330M~10B 全规模**上稳定追平全量微调，是这条线的集大成者。

> 一句话记忆：**v1 → v2 的最大进化 = 从「只在输入层」升级为「每一层都加前缀（deep prompt）」。**

## 7. 与全量微调对比

| 维度 | 全量微调 (Full FT) | Prefix-Tuning / P-Tuning v2 |
|------|--------------------|------------------------------|
| 改什么 | 所有权重 $W$ | 只改注入的 K/V 前缀 |
| 可训练参数 | 100% | ~0.1%–1% |
| 优化器状态 | 巨大（如 7B→84GB） | 极小（几十 MB） |
| 多任务部署 | 每任务存一份全模型 | 共享底座 + 每任务存小前缀 |
| 灾难性遗忘 | 易（动了底座） | 轻（底座没动） |
| 小数据/低资源 | 易过拟合 | 更稳 |
| 极限精度 | 上限最高 | 大模型上可追平，小模型略逊 |
| 推理额外开销 | 无 | 每层 K/V 多 $L$ 个（KV-cache 略增） |

## 8. 代码：PEFT 里的 PrefixEncoder

```python
from peft import PrefixEncoder, PrefixTuningConfig

config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,        # 前缀长度 L
    token_dim=768,               # 隐藏维 d
    num_transformer_submodules=1,# encoder/decoder 子模块数(seq2seq 常为 2)
    num_attention_heads=12,
    num_layers=12,               # 层数 N，决定要生成多少层前缀
    encoder_hidden_size=768,     # 重参数化 MLP 的隐藏维 H
)

prefix_encoder = PrefixEncoder(config)
# PrefixEncoder 内部: 一个 Embedding(L, d) 当种子 P'，
# 若 prefix_projection=True，再接 MLP 把 P' 投影成 (L, 2·N·d) 的各层 K/V 前缀。
```

实际用时通常 `model = get_peft_model(base_model, config)`，PEFT 会自动把生成的前缀在每层前向时塞进 `past_key_values`（即 KV-cache 的最前面），无需改动模型代码。参数管理与 `save_pretrained` 等见 [[llm-train/peft/PEFT-API]]。

## 9. 面试问答清单

**Q1：Prefix-Tuning 和 Prompt-Tuning 最本质的区别？**
踩点：① Prompt 只在**输入 embedding 层**加软 token，Prefix 在**每一层的 K/V** 加；② 因此 Prefix 表达力更强、对模型规模不敏感，Prompt 要超大模型才行；③ Prefix 一般配 MLP 重参数化。
追问「为什么每层加更强？」→ 影响在深层不断被「续写」，而 Prompt 的信号要靠残差一路传上去，会被稀释。

**Q2：为什么前缀只加在 K、V，不加在 Q？**
加 Q 会凭空多出 $L$ 个「发问者」，导致输出序列长度从 $n$ 变 $n+L$，破坏下游形状，且这些前缀 query 的输出没有监督意义。加在 K/V 只是**扩充被注意的上下文**，输出仍是 $n$ 个真实 token。

**Q3：为什么要 reparameterization？不重参数会怎样？**
直接优化高维前缀，损失面崎岖、训练不稳、对学习率极敏感。用小种子 + MLP 膨胀，提供平滑与参数共享。**推理时**可把 MLP 跑一遍缓存 $P$ 再丢掉，零额外计算。

**Q4：Prefix-Tuning 省的是什么显存？底座权重省了吗？**
省的是**梯度 + 优化器状态**（Adam 的 m/v），从全量的几十 GB 降到几十 MB。**底座权重仍要常驻**（冻结不更新但要前向）。所以它不是「模型变小」，是「需要存梯度的参数变小」。

**Q5：P-Tuning v1 → v2 解决了什么？**
v1 只在输入层加，序列标注等难任务、中小模型上掉点。v2 改成**每层 deep prompt**（即 Prefix 思想用于 NLU），并去掉 verbalizer 依赖，在 330M~10B 全规模稳定追平微调。

**Q6：推理时 Prefix-Tuning 有额外开销吗？**
有一点点：每层 KV-cache 前面多 $L$ 个 token，注意力打分多算 $n\times L$ 次。但没有额外的层/矩阵乘大块，相对很小；MLP 在推理期已被缓存丢弃。

## 对照/复杂度表

| 方法 | 可训练参数量级 | 注意力 K/V 序列长 | 训练辅助网络 | 推理额外计算 |
|------|----------------|-------------------|--------------|--------------|
| Full FT | $O(\text{模型全参})$ | $n$ | 无 | 无 |
| Prompt-Tuning | $L\times d$ | 第1层 $n{+}L$ | 无 | 极小 |
| Prefix-Tuning | $2NLd$ | 每层 $n{+}L$ | MLP(训练期) | 极小(已缓存) |
| P-Tuning v1 | $\approx L\times d$ + LSTM | 第1层 $n{+}L$ | LSTM+MLP | 小 |
| P-Tuning v2 | $2NLd$ | 每层 $n{+}L$ | 可选 MLP | 极小 |

复杂度（单层注意力）：原本 $O(n^2 d)$ → 加前缀后 $O(n(n+L)d)$，因 $L\ll n$ 通常可忽略。

**一张图收束「改哪里」的差异**：
```
                输入X ─► [Embedding] ─► 层1 ─► 层2 ─► ... ─► 层N ─► 头
 Prompt-Tuning:          ▲软token                                      只这一处
 P-Tuning v1:            ▲软token(LSTM编码)                            只这一处
 Prefix / v2:                       ▲KV  ▲KV  ...  ▲KV               每一层都有
 LoRA:                              ↑ΔW  ↑ΔW  ...  ↑ΔW   (改权重增量,不改上下文)
 Full FT:        全部权重都更新
```

## 常见问题/高频追问

| 问题 | 一句话答案 |
|------|------------|
| 前缀长 $L$ 怎么选？ | 常用 10–30；太长收益递减且占 KV-cache，生成任务可稍长 |
| 前缀向量初始化？ | 可随机，也可用真实词 embedding（如任务相关词）初始化更稳 |
| 能和 LoRA 一起用吗？ | 可以，二者改的是不同位置（前缀改上下文，LoRA 改权重增量），PEFT 支持叠加 |
| 为什么生成任务偏爱 Prefix？ | 自回归每步都看得到前缀，控制生成风格/任务很自然（原论文做 table-to-text、摘要）|
| 训练只更新前缀，梯度怎么回传？ | 前向用了冻结主干，反向梯度照常**穿过**主干（不累积），最终落到前缀/MLP 上 |
| 和 Adapter 比？ | Adapter 在层间插小瓶颈模块（改结构）；Prefix 不改结构，只改 K/V 上下文 |

## 🔗 跳转链接
- [[00-知识地图]] — 全局索引，PEFT 在大图中的位置
- [[llm-train/peft/Prompt-Tuning]] — 只在输入层的「轻量版」，理解差异的对照组
- [[llm-train/peft/PEFT-API]] — `get_peft_model` / `PrefixEncoder` / 保存加载的工程接口
