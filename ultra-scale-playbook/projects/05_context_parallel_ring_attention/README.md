# 项目 05 · 从零实现上下文并行 CP / Ring Attention

> 对应《Ultra-Scale Playbook》**第 5 章(上下文并行)**。
>
> 当序列长到 128k、1M token,连注意力的激活都放不下(注意力显存随 `s²` 爆炸)。**上下文并行**沿**序列维**把 Q/K/V 切到各卡,用 **Ring Attention** 让 K/V 块在环上逐跳传递、边收边算,**永不物化完整的 S×S 注意力矩阵**,显存从 O(S²) 降到 O(S)。

---

## 🎯 核心:在线 softmax(和 FlashAttention 同源)

每个 rank 持有一段 Q_i,在环上依次收到各个 K/V 块。对每个新块,维护 running max `m`、running sum `l`、running output `O`:

$$m_{new}=\max(m,\ \text{rowmax}(S)),\quad \alpha=e^{m-m_{new}}$$
$$l \leftarrow \alpha l + \text{rowsum}(e^{S-m_{new}}),\quad O \leftarrow \alpha O + e^{S-m_{new}}V$$

最后 `O /= l`。初始 `m=-∞` → `α=0`,自动忽略空状态。数学上与"先算完整 softmax 再乘 V"**逐元素等价**——这正是测试验证的。

```mermaid
flowchart LR
    subgraph Ring[K/V 在环上逐跳传递]
      R0[rank0<br/>Q0] --> R1[rank1<br/>Q1] --> R2[rank2<br/>Q2] --> R3[rank3<br/>Q3] --> R0
    end
    note[每 rank 只存 O(S/P) 的 K/V<br/>走完一圈见过全部 K/V]
```

## ⚖️ Zig-Zag 负载均衡
因果注意力下,靠后的 query 块要看更多 key → 各 rank 工作量不均(rank P-1 做 P 块,rank 0 只做 1 块)。**Zig-Zag** 把序列位置交错分配(每个 rank 拿"一前一后"两段),把计算量摊平。`zigzag_indices` 给出这个重排(测试验证它是合法双射)。

## 📁 文件
| 文件 | 作用 |
|---|---|
| `ring_attention.py` | 在线 softmax 累加、Ring Attention(含因果)、zigzag 重排(单进程核心) |
| `test_ring.py` | 15 个 pytest:ring==全注意力(因果/非因果)、在线 softmax 两块累加、首 token 只看自己、zigzag 合法性 |
| `run_demo.py` | **真·多进程**:gloo 起 4 进程,K/V 用 `batch_isend_irecv` 环上逐跳,校验偏差 ~1e-16 |

## ▶️ 如何运行
```bash
python -m pytest -q      # 15 passed —— ring attention 逐元素对拍全注意力
python run_demo.py       # 4 进程 gloo ring,K/V 环上流转,偏差 ~2.2e-16
```

## 💡 面试高频
- "长序列显存瓶颈在哪、怎么解?" → 注意力激活 O(s²);上下文并行 + Ring Attention 沿序列切,在线 softmax 累加,显存 O(s/P)。
- "Ring Attention 和 FlashAttention 关系?" → 都用在线 softmax + 分块;FlashAttention 在**单卡片上(SRAM)**分块,Ring Attention 在**多卡序列维**分块,K/V 走网络环。
- "因果掩码下为什么要 zigzag?" → 摊平各 rank 计算量,避免尾部 rank 成为瓶颈。

## ⚠️ 常见坑
- 环上通信用阻塞 send/recv 容易死锁 → 用 `batch_isend_irecv`(非阻塞)。
- 在线 softmax 忘了用 `exp(m_old - m_new)` 重缩放旧的 O 和 l → 结果错。
- 整块被因果完全屏蔽时(全 -inf)未做保护 → NaN(本项目用 `nan_to_num` + `m_safe` 处理)。

## 🔗 延伸
- 理论:`../../book-guide/04_上下文并行_CP_RingAttention_ZigZag.md`
- 同源算子:`../../book-guide/09_深入GPU...md`(FlashAttention)、CUDA 版见仓库 `cuda-mastery`(如有)
- 上一个:`../04_pipeline_parallel`;下一个:`../06_collectives_from_scratch`
