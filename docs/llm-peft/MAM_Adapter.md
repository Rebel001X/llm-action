# MAM Adapter（Mix-And-Match Adapter，混合并匹配适配器）

> 一句话定位：MAM Adapter 把"在注意力子层用 **并行 Prefix-Tuning**"与"在 FFN 子层用 **缩放的并行 Adapter（Scaled Parallel Adapter）**"两个最优组件**混搭拼装**，是论文《Towards a Unified View of Parameter-Efficient Transfer Learning》(He et al., ICLR 2022) 在统一视角下手工搜出的"最佳配方"。📍 导航：[[00-知识地图]]
>
> 🔗 相关：[[llm-train/peft/Prefix-Tuning]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-peft/LoRA-FA]] · [[llm-peft/ReLoRA]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-train/README]]

---

## 阅读地图

| 节 | 你将学到 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点：MAM = 并行Prefix(attn) + 缩放并行Adapter(ffn) | 配方 |
| 1 | 地基：Transformer 两个子层 + 三种 PEFT 前置 | Adapter/Prefix/LoRA |
| 2 | 统一视角：把 Adapter/Prefix/LoRA 写成同一个 $\Delta h$ 公式 | 统一框架 |
| 3 | 四个设计维度：插入形式/组合函数/插入位置/作用函数 | 设计空间 |
| 4 | 三大发现：并行 > 串行、FFN > Attn、缩放很关键 | 消融结论 |
| 5 | MAM Adapter 的精确结构与数据流 ASCII 图 | 拼装 |
| 6 | 配置逐行解读（仓库 options.py 对应） | 超参 |
| 7 | 数值手算：可训练参数量、内存、FLOPs | 估算 |
| 8 | 前向传播伪代码 | 实现 |
| - | 常见问题 + 跳转链接 | FAQ |

---

## 0. 一句话锚点

**MAM Adapter 不是一个新模块，而是一张"配方表"。** 它的全部内容是：

```
注意力(Attention)子层  ──> 用「并行 Prefix-Tuning」，瓶颈维 bn = 30
前馈(FFN)子层         ──> 用「缩放的并行 Adapter」，瓶颈维 bn = 512，缩放 s = 4
```

为什么是这两个？因为论文先把所有 PEFT 方法塞进**同一个数学框架**，再沿四个"旋钮"做网格消融，最后把每个子层上**各自最好**的那个旋钮位置拼起来——这就是 **M**ix-**A**nd-**M**atch。

---

## 1. 地基 / 前置：先把 Transformer 一层拆到原子

一个标准 Transformer Encoder/Decoder 层有两个**子层（sub-layer）**，每个子层都是 `输入 → 子函数 → 残差相加 → LayerNorm`：

```
        x ──────────────┐ (残差)
        │               │
   ┌────▼─────┐         │
   │ Multi-Head│        │
   │ Attention │        │  ← 子层 A：注意力
   └────┬─────┘         │
        │               │
        ▼               │
       (+)◄─────────────┘
        │
   [LayerNorm]
        │
        x' ─────────────┐ (残差)
        │               │
   ┌────▼─────┐         │
   │   FFN     │        │  ← 子层 B：前馈
   │ W2·σ(W1·) │        │
   └────┬─────┘         │
        │               │
        ▼               │
       (+)◄─────────────┘
        │
   [LayerNorm]
```

- **注意力子层**：$\text{Attn}(Q,K,V)=\text{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}\big)V$。详见 [[llm-algo/transformer/模型架构]]。
- **FFN 子层**：$\text{FFN}(x)=W_2\,\sigma(W_1 x)$，中间维度通常 $4d$。详见 [[llm-algo/mlp]]。

**三个 PEFT 前置方法**（MAM 就是从它们里挑零件）：

| 方法 | 在哪里加 | 加什么 | 富笔记 |
|------|----------|--------|--------|
| **Adapter** | 子层之后（串行） | 一个降维-非线性-升维瓶颈 MLP | 本文 §2 |
| **Prefix-Tuning** | 注意力的 K、V 前面 | 拼接 $l$ 个可学习"前缀"向量 | [[llm-train/peft/Prefix-Tuning]] |
| **LoRA** | 权重矩阵旁边 | 低秩 $BA$ 增量 | [[llm-peft/LoRA-FA]] |

> 关键观察（论文的核心洞见）：**Prefix-Tuning 其实是一种作用在注意力上的"并行 Adapter"。** 看下一节怎么把它们写成同一个公式。

---

## 2. 统一视角：一个 $\Delta h$ 公式装下所有 PEFT

论文把每种 PEFT 对隐状态的修改都写成对**某个隐藏向量 $h$ 的增量 $\Delta h$**。设输入到被修改模块的向量为 $x$，原输出为 $h$：

**(a) Adapter（串行）**：在子层输出后接瓶颈，
$$\Delta h = f\big(h\, W_{\text{down}}\big)\,W_{\text{up}},\qquad W_{\text{down}}\in\mathbb{R}^{d\times r},\;W_{\text{up}}\in\mathbb{R}^{r\times d}$$
其中 $r\ll d$ 是**瓶颈维（bottleneck dim）**，$f$ 是非线性（如 ReLU/GELU）。

**(b) LoRA**：
$$\Delta h = s\cdot x\, W_{\text{down}}\,W_{\text{up}},\qquad s=\tfrac{\alpha}{r}$$
没有非线性 $f$，且**直接乘在输入 $x$ 上**（并行支路），还带一个**缩放 $s$**。

**(c) Prefix-Tuning**：在注意力里拼接 $l$ 个前缀键值 $P_k,P_v\in\mathbb{R}^{l\times d}$。论文做代数变换证明：
$$
h \;=\; (1-\lambda)\,\underbrace{\text{Attn}(xW_q,\,KW_k,\,VW_v)}_{\text{原注意力}}
\;+\;\lambda\,\underbrace{\text{Attn}(xW_q,\,P_k,\,P_v)}_{\text{对前缀的注意力}}
$$
$$
\Rightarrow\quad \Delta h \;=\; \lambda\cdot\Big[\,\text{softmax}(xW_q P_k^\top)\,P_v \;-\; h\,\Big]
$$
**右边那一项本质上又是一个"低秩、并行、带门控 $\lambda$"的修改**——形状上等价于一个**作用在注意力上的并行 Adapter**，瓶颈维就是前缀长度 $l$。

> 三个看似无关的方法，被压成同一个模板：
> $$\boxed{\;\Delta h \;=\; s\cdot f\!\big(x\,W_{\text{down}}\big)\,W_{\text{up}}\;}$$
> 它们只是在四个旋钮上取了不同档位。

```
统一模板：    x ──► W_down(d×r) ──► f(·) ──► W_up(r×d) ──► × s ──► Δh
                     降维            非线性     升维          缩放
 Adapter:   f=ReLU, s=1, 串行(作用在 h 上), 加在 FFN/Attn 后
 LoRA:      f=Id,   s=α/r, 并行(作用在 x 上), 加在权重旁
 Prefix:    f=softmax门控, 并行, 作用在 Attention 的 K/V 上
```

---

## 3. 四个设计维度（PEFT 的"旋钮空间"）

论文把整个设计空间拆成**四个正交旋钮**，MAM 就是在这四维里搜出来的点：

```
┌─────────────────────────────────────────────────────────────┐
│ 旋钮①  Functional Form  作用函数  : 用不用非线性 f？          │
│        - Adapter 有 ReLU；LoRA 没有                           │
├─────────────────────────────────────────────────────────────┤
│ 旋钮②  Insertion Form   插入形式  : 串行(sequential) 还是     │
│        并行(parallel)？                                        │
│        - 串行：Δh 作用在子层"输出 h"上                         │
│        - 并行：Δh 与子层"并排"，都吃同一个输入 x               │
├─────────────────────────────────────────────────────────────┤
│ 旋钮③  Modified Repr.   修改位置  : 改注意力(attn) 还是 FFN？ │
├─────────────────────────────────────────────────────────────┤
│ 旋钮④  Composition Fn   组合函数  : 怎么把 Δh 合回去？        │
│        - simple add:  h ← h + Δh                              │
│        - scaled add:  h ← h + s·Δh   (LoRA 风格，带缩放)      │
│        - gated add:   h ← (1-λ)h + λΔh (Prefix 风格，门控)    │
└─────────────────────────────────────────────────────────────┘
```

**串行 vs 并行**的图（最重要的一个旋钮）：

```
  串行 Adapter (原始 Houlsby)            并行 Adapter (MAM 用的)
  ─────────────────────────             ─────────────────────────
       x                                     x ──────────┬─────────┐
       │                                     │           │         │
   ┌───▼───┐                             ┌───▼───┐   ┌───▼────┐    │
   │ 子层   │                             │ 子层   │   │Adapter │    │
   │(FFN)  │                             │(FFN)  │   │ 瓶颈   │    │
   └───┬───┘                             └───┬───┘   └───┬────┘    │
       │                                     │           │(Δh)     │
   ┌───▼────┐                                └────►(+)◄──┘         │
   │Adapter │  ← 必须等子层算完                    │                │
   │ 瓶颈   │     才能算 Adapter                   ▼                │
   └───┬────┘     (有数据依赖)              h = FFN(x)+Δh(x)        │
       │                                   两支路可并行计算          │
       ▼
   h = Adapter(FFN(x))
```

并行的好处不止精度：两条支路**没有数据依赖**，可以并行/重叠计算（参见 [[llm-optimizer/计算通信重叠]] 的思想）。

---

## 4. 三大消融发现（MAM 的"为什么"）

论文沿四个旋钮做了大量 XSum/MT/SQuAD 实验，得到三条可复用的结论：

| # | 发现 | 直觉解释 |
|---|------|----------|
| 1 | **并行 > 串行**。并行 Adapter 一致优于串行 Adapter。 | 并行支路直接吃原始输入 $x$，信息更"干净"，且不破坏主干。 |
| 2 | **小参数预算下：改 Attention 更划算；大参数预算下：改 FFN 更划算**。 | FFN 容量大（中间维 $4d$），能吸收更多知识，但要给够瓶颈维。 |
| 3 | **缩放（scaled add）的组合函数很关键**，缩放系数 $s$（如 4）显著提升表现。 | 控制 $\Delta h$ 相对主干的"音量"，避免初期扰动过大。 |

把三条拼起来就得到配方：
- **Attention 子层**：参数预算给得少（前缀只 30 维）→ 选 **并行 Prefix**（= 并行 attn-adapter）。
- **FFN 子层**：参数预算给得多（瓶颈 512 维）→ 选 **缩放的并行 Adapter**，$s=4$。

这就是 **M**ix-**A**nd-**M**atch：不同子层用各自最优的零件。

---

## 5. MAM Adapter 的精确结构与数据流

一层 Transformer 在 MAM 下长这样（★ 标记的是新增的可训练支路，主干全部冻结）：

```
                        输入 x
                          │
        ┌─────────────────┼───────────────────────────┐
        │ (冻结主干)       │             ★ 并行 Prefix    │
        │                 │             (attn bn=30)    │
   ┌────▼─────┐    ┌──────▼──────┐                      │
   │Multi-Head│    │ 拼接前缀 K/V │  P_k,P_v ∈ ℝ^{30×d}  │
   │Attention │◄───┤ 到注意力里   │  (经 MLP 重参数化)    │
   └────┬─────┘    └─────────────┘                      │
        │   h = (1-λ)·Attn(x) + λ·Attn_to_prefix(x)     │
       (+)◄── 残差 x                                     │
        │                                               │
   [LayerNorm]                                           │
        │                                               │
        x' ─────────────┬──────────────────────┐        │
        │ (冻结主干)      │        ★ 缩放并行 Adapter │     │
   ┌────▼─────┐         │        (ffn bn=512, s=4) │     │
   │   FFN     │    ┌────▼─────┐                   │     │
   │ W2σ(W1·) │     │ W_down   │ 512维              │     │
   └────┬─────┘     │   ↓ f    │ (init=lora 风格)   │     │
        │           │ W_up ×4  │                   │     │
        │           └────┬─────┘ Δh = 4·f(x'·W_d)·W_u   │
        │                │                          │     │
       (+)◄──────────────┴── 残差 x' ───────────────┘     │
        │   FFN_out = FFN(x') + Δh(x') + x'                │
   [LayerNorm]                                             │
        │                                                  │
        输出 ◄────────────────────────────────────────────┘
```

**注意两条支路的差别**：
- Attention 支路用 **gated add**（门控 $\lambda$，Prefix 天然带），瓶颈仅 30。
- FFN 支路用 **scaled add**（缩放 $s=4$，LoRA 风格），瓶颈 512，且 `init=lora`（$W_{\text{up}}$ 初始化为 0、$W_{\text{down}}$ 高斯，保证训练初始 $\Delta h=0$，不破坏预训练）。

---

## 6. 配置逐行解读

仓库 `unify-parameter-efficient-tuning/petl/options.py` 中 MAM 的典型配置（即原文件给出的片段）：

```bash
# ----- MAM adapter -----
attn_mode="prefix"            # 注意力子层用 Prefix-Tuning
attn_option="concat"          # 把前缀 concat 到 K/V（标准 prefix 做法）
attn_composition="add"        # 组合函数：加（门控加由 prefix 内部实现）
attn_bn=30                    # ★ attention 瓶颈维 = 前缀长度 l = 30

ffn_mode="adapter"            # FFN 子层用 Adapter
ffn_option="parallel"         # ★ 并行插入（不是串行！这是关键）
ffn_adapter_layernorm_option="none"   # adapter 内不加 LayerNorm
ffn_adapter_init_option="lora"        # ★ 用 LoRA 式初始化：W_up=0,起点 Δh=0
ffn_adapter_scalar="4"        # ★ 缩放 s = 4（scaled parallel adapter）
ffn_bn=512                    # ★ FFN 瓶颈维 = 512（比 attn 大很多）
```

逐项对应到 §3 的四个旋钮：

| 配置项 | 旋钮 | 取值含义 |
|--------|------|----------|
| `attn_mode=prefix` | ③位置 + ②形式 | attn 上的并行 adapter |
| `attn_bn=30` | 参数预算 | attn 瓶颈小（少参数即可） |
| `ffn_option=parallel` | ②插入形式 | 并行（发现1：并行>串行） |
| `ffn_bn=512` | 参数预算 | FFN 瓶颈大（发现2：FFN 给够维度） |
| `ffn_adapter_scalar=4` | ④组合函数 | scaled add（发现3：缩放关键） |
| `ffn_adapter_init_option=lora` | ①作用函数初始化 | 初始 Δh=0，安全启动 |

> 参数细节以官方仓库 `petl/options.py` 与 `petl/petl_factory.py` 为准（不同 commit 默认值可能不同）。

---

## 7. 数值手算

设模型隐藏维 $d=1024$，层数 $L=12$（如 BART-large 的 encoder/decoder 合计也可按此类推）。

### 7.1 可训练参数量

**FFN 并行 Adapter（每层）**：$W_{\text{down}}\in\mathbb{R}^{d\times r_f}$、$W_{\text{up}}\in\mathbb{R}^{r_f\times d}$，$r_f=512$：
$$
P_{\text{ffn}} = 2\,d\,r_f = 2\times 1024\times 512 = 1{,}048{,}576 \approx 1.05\text{M}/\text{层}
$$

**Attention 并行 Prefix（每层）**：前缀 $l=30$，需为 K、V 各一组前缀，原始 Prefix-Tuning 还用一个重参数化 MLP（瓶颈 $r_a=30$ 量级）。按"直接前缀 + K/V 两组 × $d$"粗算下界：
$$
P_{\text{attn}} \approx 2\,l\,d = 2\times 30\times 1024 = 61{,}440 \approx 0.06\text{M}/\text{层}
$$

**单层合计**：$P_{\text{layer}}\approx 1.05\text{M}+0.06\text{M}=1.11\text{M}$。

**全模型**（$L=12$）：
$$
P_{\text{total}} \approx 12\times 1.11\text{M} \approx 13.3\text{M}
$$

对一个 ~400M 的 BART-large，这是约 **3.3%** 的可训练占比（实际论文报告 MAM 用约 **6.7%** 参数即逼近全量微调，量级一致——差异来自重参数化 MLP 与是否含 cross-attention）。

> 核心直觉：FFN 支路吃掉 ~95% 的可训练参数（512 远大于 30），印证"大预算给 FFN"。

### 7.2 显存增量

冻结主干 → **优化器状态（Adam）只为可训练参数维护 $m,v$ 两份**。以 fp32、参数量 $13.3\text{M}$：
$$
\text{优化器+梯度+参数} \approx 13.3\text{M}\times (4+4+4)\,\text{B} = 13.3\text{M}\times 12\,\text{B}\approx 160\,\text{MB}
$$
相比全量微调 400M 参数需 $400\text{M}\times 12\text{B}=4.8\text{GB}$，**节省约 30×**。激活内存上，并行 Adapter 比串行更省（可与主干计算重叠），但仍需缓存中间激活（这正是 [[llm-peft/LoRA-FA]] 进一步优化的点）。

### 7.3 前向 FLOPs 增量

FFN 并行 Adapter 每 token 的乘加：$2\times(d\,r_f + r_f\,d)=4\,d\,r_f$。设序列 $T=512$：
$$
\text{FLOPs}_{\text{ffn,层}} \approx 2\times 4\,d\,r_f\,T = 8\times 1024\times 512\times 512 \approx 2.1\,\text{GFLOPs}
$$
（前面 $\times2$ 是一次乘+一次加。）相对原 FFN 的 $2\times 2\,(4d)\,d\,T\approx 8.6$ GFLOPs/层，Adapter 增量约 **+24%**；但因瓶颈维远小于 $4d$，整体推理代价仍很轻。Prefix 支路只在注意力里多 $l=30$ 个 K/V，增量可忽略。

---

## 8. 前向传播伪代码

```python
# 主干 W_q,W_k,W_v,W_o,W1,W2 全部 requires_grad=False（冻结）
def mam_layer(x):
    # ---- 子层 A：注意力 + 并行 Prefix ----
    Pk, Pv = reparam_mlp(prefix_tokens)      # ★ 可训练，l=30
    K = cat([Pk, x @ W_k]); V = cat([Pv, x @ W_v])
    attn = softmax((x @ W_q) @ K.T / sqrt(dk)) @ V   # 含门控 λ（前缀内部）
    x = layernorm(x + attn @ W_o)            # 残差 + LN

    # ---- 子层 B：FFN + 缩放并行 Adapter ----
    ffn_out = (relu(x @ W1)) @ W2            # 冻结主干 FFN
    delta   = (f(x @ W_down)) @ W_up         # ★ 并行支路, r=512
    delta   = SCALAR * delta                 # ★ s = 4 (scaled add)
    x = layernorm(x + ffn_out + delta)       # 残差 + 两支路相加 + LN
    return x

# 初始化（ffn_adapter_init_option="lora"）：
#   W_up   <- 0          → 训练起点 delta=0，不破坏预训练
#   W_down <- 高斯
```

关键点：`delta` 与 `ffn_out` **并行**（都吃同一个 `x`），训练初始为 0（LoRA 式 init），缩放 4 控制其相对幅度。

---

## 常见问题

| 问题 | 回答 |
|------|------|
| MAM Adapter 是一个新网络层吗？ | 不是。它是"在 attn 用并行 Prefix + 在 FFN 用缩放并行 Adapter"的**组合配方**，零件都来自已有方法。 |
| 为什么 Prefix 能算作"注意力上的并行 Adapter"？ | 代数变换后，前缀注意力等价于一个低秩、并行、带门控 $\lambda$ 的 $\Delta h$，形状与并行 adapter 一致（§2c）。 |
| 为什么 attn 瓶颈 30、FFN 瓶颈 512 差这么多？ | 发现2：小预算改 attn 更划算，大预算改 FFN 更划算。所以把"大维度"留给 FFN。 |
| `parallel` 和串行 Adapter 有何本质差别？ | 串行作用在子层**输出**上、有数据依赖；并行与子层**并排**吃同一输入，精度更高且可重叠计算（发现1）。 |
| `scalar=4` 是干嘛的？ | scaled-add 组合函数的缩放 $s$，控制 $\Delta h$ 音量，发现3 表明它对效果关键。 |
| `init_option=lora` 是什么意思？ | 用 LoRA 式初始化（$W_{\text{up}}=0$），保证训练起点 $\Delta h=0$，安全地从预训练权重出发。 |
| 和 LoRA / Prefix / Adapter 单用比？ | MAM 在相近甚至更少参数下，效果优于任一单方法，逼近全量微调——因为它"取各家之长"。 |
| 推理有额外开销吗？ | FFN 支路约 +24% FLOPs（瓶颈小，绝对量轻）；Prefix 支路几乎可忽略。无法像 LoRA 那样完全合并进主干。 |

---

## 🔗 跳转链接

- 顶层：[[00-知识地图]]
- 同族 PEFT：[[llm-train/peft/Prefix-Tuning]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-peft/LoRA-FA]] · [[llm-peft/ReLoRA]]
- Transformer 结构地基：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/FLOPs]]
- 训练框架与并行：[[llm-train/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 计算重叠（并行支路思想）：[[llm-optimizer/计算通信重叠]]

---

> 原始仓库参考（参数与逻辑以官方为准）：
> - 配置项 `petl/options.py`：`unify-parameter-efficient-tuning`
> - `ffn_mode` 前向逻辑：`src/transformers/models/bart/modeling_bart.py`
> - `ffn_adapter_init_option` 初始化：`petl/petl_factory.py`
> - 启动脚本：`exps/run_xsum.sh`
> - 论文：He et al., *Towards a Unified View of Parameter-Efficient Transfer Learning*, ICLR 2022.
