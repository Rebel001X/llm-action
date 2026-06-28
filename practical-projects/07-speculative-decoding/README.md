# 从零实现投机采样（Speculative Decoding）

一个 **CPU 几秒就能跑通** 的 toy 实战：用两个字符级语言模型（小的 draft + 大的 target）从零实现投机采样，量化加速比，并用蒙特卡洛实验验证它是 **无损** 的（输出分布与 target 自回归完全一致）。

> 对应理论文档：[`../../llm-inference/README.md`](../../llm-inference) 第 5.5 节「解码加速：投机解码（Speculative Decoding）」。

---

## 一、演示什么原理

大模型（target）自回归解码是 **串行** 的：一次前向只产出 1 个 token，访存/算力利用率低、延迟高。

投机采样（Leviathan et al., 2022）的解法：

1. **草稿（draft）**：用一个便宜的小模型一口气“猜” `k` 个 token，记录每个位置的 draft 分布 `q`。
2. **验证（verify）**：让大模型 target **并行** 对这 `k` 个候选位置一次性打分，得到分布 `p`。这是加速的来源——**一次 target 前向能确认多个 token**。
3. **接受/拒绝（accept-reject）**：对每个候选 token `x`，以概率 `min(1, p(x)/q(x))` 接受；
   - 第一个被拒的位置，从 **修正分布** `norm((p - q)_+)` 重采一个 token，并丢弃其后所有候选；
   - 若 `k` 个全部被接受，再用 target 在下一位的分布 **免费多采 1 个**（每轮最多前进 `k+1` 个 token）。

**关键性质（无损）**：经过“接受概率 `min(1,p/q)` + 修正分布 `(p-q)_+`”，每个被产出的 token 都严格服从 target 分布 `p`。小模型只影响 **速度**，不影响 **分布**（arXiv:2211.17192 定理 1）。

> 为什么用统计字符模型而非 Transformer？因为接受准则与修正分布是 **与模型无关** 的概率算法。用可枚举的小模型，能在 CPU 上严格验证“分布等价”，教学信号最干净；真实工程把 draft/target 换成两个 Transformer，算法一字不变。

---

## 二、怎么跑

环境：Python 3.13 / numpy 2.3 / torch 2.12（CPU）。本脚本仅依赖 numpy（matplotlib 可选）。

```bash
cd practical-projects/07-speculative-decoding
python speculative_decoding.py
```

约 1.5 秒跑完，并在当前目录生成接受长度直方图 `accept_length_hist.png`（matplotlib 缺失则自动降级为纯文本，不会崩）。

---

## 三、预期输出（真实运行摘录）

```
[autoregressive] target_forwards=200 (one per token)
[autoregressive] sample: 'fox jumps over the lazy dog . the lazy dog . the lazy dog . '
----------------------------------------------------------------------
[speculative]  k=4  rounds=84
[speculative]  target_forwards=84  draft_forwards=333
[speculative]  avg_accept_len=1.393 (tokens accepted per target forward, before bonus)
----------------------------------------------------------------------
[speedup] target forwards: AR=200 -> SPEC=84
[speedup] tokens per target forward = 2.381
[speedup] relative speedup (fewer target forwards) = 2.38x
----------------------------------------------------------------------
[lossless check] TV(speculative , autoregressive) = 0.0084
[lossless check] TV(speculative , target_true)    = 0.0036
[lossless check] TV(autoregress , target_true)    = 0.0056
[lossless check] speculative matches target within sampling noise: YES
----------------------------------------------------------------------
SUMMARY
  - speculative produced 200 tokens using only 84 target forwards
  - speedup vs pure autoregressive: 2.38x (target forwards)
  - distribution lossless vs target: PASS
```

怎么看“成功”：

- **加速**：产出同样的 200 个 token，target 前向次数从 `200` 降到 `84`，即 **2.38x** 加速（每次 target 前向平均产出 2.38 个 token）。
- **无损**：投机采样与 target 真实分布的总变差距离 `TV = 0.0036`，**比基线自回归自身的采样噪声 `0.0056` 还小**，说明二者在统计上不可区分 → `PASS`。
- **接受长度直方图**：能直观看到“每轮接受 0/1/2/3/4 个 token”的分布；draft 越准，长接受占比越高、加速越明显。

---

## 四、对应 llm-action 文档

- [`../../llm-inference`](../../llm-inference) — 大模型推理目录总览。
- [`../../llm-inference/README.md`](../../llm-inference/README.md) **第 5.5 节** 「解码加速：投机解码（Speculative Decoding）」—— 本项目的理论出处。
- [`../../llm-inference/KV-Cache优化.md`](../../llm-inference/KV-Cache优化.md) — 投机采样与 KV-Cache 配合使用，验证阶段需对接受/回退的 token 维护正确的 KV。

---

## 五、社区参考

- [Speculative Decoding (arXiv:2211.17192) — Leviathan et al., 2022](https://arxiv.org/abs/2211.17192)
- [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

---

## 六、局限 / 与真实工程的差异

- **模型是 toy 统计 n-gram，不是 Transformer**：本项目用字符级 n-gram 充当 draft/target，目的是把“接受-拒绝 + 修正分布”讲清楚并可严格验证。真实系统中 draft/target 是两个 Transformer（或同一模型的 Medusa/EAGLE 头、n-gram 提示等）。
- **“并行验证”是概念上的**：脚本里 target 用循环逐位求分布，但每轮只计 **1 次** target 前向以反映真实代价。真实推理是把 `k+1` 个位置 **拼成一个 batch 一次前向**，并配合 attention mask；本脚本未实现批量 kernel。
- **没有真正的 KV-Cache 管理**：真实工程最难的部分之一是接受/回退时正确回滚 KV-Cache（接受的前缀复用、被拒位置之后丢弃）。这里序列短、直接重算，未涉及缓存回滚。
- **加速比是“target 前向次数”的代理**：真实加速还取决于 draft 自身开销、batch/显存带宽、`k` 的取值、draft 与 target 的对齐度（接受率）。本脚本忽略 draft 计算成本，因此报告的是“理想化上界”量级的趋势，而非端到端墙钟时间。
- **规模极小**：词表 28、语料几千字符、`n_new=200`，只为快速跑通与可复现；接受率受 draft 质量主导，换更难的语料接受长度会下降。
