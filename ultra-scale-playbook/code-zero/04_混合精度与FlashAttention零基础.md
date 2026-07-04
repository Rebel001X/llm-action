# 零基础(四)· 混合精度训练 与 FlashAttention

> 这两样是"压榨单卡"的两大杀手锏,叠加在并行之上。读完你能说清 bf16/fp16/fp8 怎么选、loss scaling 为什么要、以及 FlashAttention 为什么能把注意力从"显存杀手"变成"省显存"。
>
> 配套:`../book-guide/09_深入GPU...md`、`../projects/05_context_parallel_ring_attention`(同源的在线 softmax)。

---

## 第一部分:混合精度训练 Mixed Precision

### 🔢 浮点格式速览

| 格式 | 位数 | 指数/尾数 | 特点 |
|---|---|---|---|
| fp32 | 32 | 8 / 23 | 范围大精度高,慢、占显存 |
| tf32 | 19(存 32) | 8 / 10 | Ampere+ 张量核默认,范围同 fp32、精度降 |
| **fp16** | 16 | **5** / 10 | 范围小(易溢出/下溢),需 loss scaling |
| **bf16** | 16 | **8** / 7 | 范围同 fp32(不易溢出),精度低——**训练首选** |
| fp8 | 8 | E4M3/E5M2 | Hopper+,极致吞吐,需精细缩放 |

> 💡 **fp16 vs bf16**:两者都 16 位,但 bf16 保留了 fp32 的 8 位指数 → **动态范围一样大,几乎不溢出**,所以现代大模型训练几乎都用 bf16(省去 loss scaling 的麻烦)。

### ⚖️ 为什么"混合"

全用 fp16/bf16 会丢精度导致训练发散。混合精度的做法:
- **前向/反向**用低精度(bf16)算 → 快、省显存、喂张量核。
- **保留一份 fp32 主参数(master weights)**,优化器在 fp32 上更新 → 保住精度。
- fp16 时还需 **loss scaling**:把 loss 乘一个大数,防止小梯度在 fp16 里下溢成 0,更新前再除回来。bf16 范围大,通常不需要。

```python
import torch
scaler = torch.cuda.amp.GradScaler()          # fp16 才需要
for x, y in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):   # 前向用 bf16
        loss = model(x, y)
    scaler.scale(loss).backward()             # 缩放 loss 再反向
    scaler.step(opt); scaler.update()
```

### 🧾 显存账
混合精度 + Adam 每参数:fp16 参数 2 + fp16 梯度 2 + **fp32 主参数 4 + Adam m 4 + v 4** = **16 B**。别以为"用了 fp16 就省一半"——优化器那 12 字节还在(见 `03_Transformer显存...md`)。

---

## 第二部分:FlashAttention

### 😱 朴素注意力的问题:显存 O(s²)

```
S = Q @ Kᵀ / √d      # (s, s) —— 物化一个 s×s 的大矩阵!
A = softmax(S)       # 又一个 s×s
O = A @ V
```
序列 s=8192 时,一个头的 `S` 就是 `8192² × 2字节 ≈ 128MB`,多头多层直接爆显存;而且反复读写 HBM,**访存受限 memory-bound**。

### 💡 FlashAttention 的两招

1. **分块 tiling**:把 Q/K/V 切成小块,放进片上极快的 SRAM 算,算完只写最终输出回 HBM,**不物化整个 s×s**。
2. **在线 softmax online softmax**:不需要先看到整行再做 softmax。分块处理,维护 running max `m`、running sum `l`、running output `O`,来一块用 `exp(m_old − m_new)` 把旧结果重缩放再并入。数学上和普通 softmax **完全等价**,但显存从 O(s²) 降到 **O(s)**。

$$m_{new}=\max(m, \text{rowmax}(S_{blk})),\quad \alpha=e^{m-m_{new}}$$
$$l\leftarrow \alpha l+\text{sum}(e^{S_{blk}-m_{new}}),\quad O\leftarrow \alpha O + e^{S_{blk}-m_{new}}V_{blk},\quad\text{最后 } O/=l$$

### 🧪 纯 numpy 验证在线 softmax 与普通 softmax 等价(可跑)

```python
import numpy as np

def softmax_attn(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    S = S - S.max(1, keepdims=True)
    A = np.exp(S); A /= A.sum(1, keepdims=True)
    return A @ V

def flash_attn(Q, K, V, block=4):
    d = Q.shape[-1]; s = K.shape[0]
    O = np.zeros_like(Q); m = np.full(Q.shape[0], -np.inf); l = np.zeros(Q.shape[0])
    for j in range(0, s, block):                      # 逐块扫 K/V
        Kb, Vb = K[j:j+block], V[j:j+block]
        S = Q @ Kb.T / np.sqrt(d)                     # (sq, block)
        m_new = np.maximum(m, S.max(1))
        alpha = np.exp(m - m_new)
        p = np.exp(S - m_new[:, None])
        l = alpha * l + p.sum(1)
        O = alpha[:, None] * O + p @ Vb
        m = m_new
    return O / l[:, None]

np.random.seed(0)
Q, K, V = (np.random.randn(6, 8) for _ in range(3))
print("最大偏差:", np.abs(softmax_attn(Q, K, V) - flash_attn(Q, K, V)).max())
# → 约 1e-16,证明在线 softmax 与普通 softmax 逐元素等价
```

### 🌊 FA1 → FA2 → FA3 → FlashDecoding
- **FA2**:减少非矩阵乘 FLOPs、沿序列维并行,约 2× FA1。
- **FA3**:Hopper 专属(WGMMA + TMA 异步 + fp8)。
- **FlashDecoding**:decode 时 q 长度=1,沿 KV 维拆分并行,拉满占用率。

> 🔗 **和上下文并行的关系**:Ring Attention(`../projects/05`)用的就是同一个在线 softmax,只不过 FlashAttention 在**单卡 SRAM 内**分块,Ring Attention 在**多卡序列维**分块。

## ⚠️ 常见坑
- 用 fp16 不做 loss scaling → 小梯度下溢、训练不动;换 bf16 或加 scaler。
- 忘了 fp32 主参数 → 精度崩。
- 在线 softmax 忘了用 `exp(m_old−m_new)` 重缩放旧的 O、l → 结果错。

## 📌 速查
- 训练首选 **bf16**;混合精度仍是 16 B/param;FlashAttention = tiling + 在线 softmax,显存 O(s)。

## 🔗 延伸
- 理论:`../book-guide/09_深入GPU_融合_线程_混合精度_FlashAttention_融合核.md`
- 实战:`../projects/05_context_parallel_ring_attention`(在线 softmax 的多卡版)
