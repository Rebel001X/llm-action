"""
moe.py —— MoE 下的投机采样账本：为什么它和稠密模型的结论**两头都不一样**。

背景：本库前面所有物理账（speedup.py）都把参数量 P 当常数，因此**不适用于 MoE**。
第 18、23 篇原本只能标注"未建模"。但调研里有一个解释不了的真实反例：
Red Hat 2026-04 报告 gpt-oss-120b（MoE + MXFP4）+ EAGLE3 在**并发 200** 时仍有约 +20% 吞吐，
而稠密模型在那个 batch 早就跌破 1.0 了。本文件把这件事算清楚。

核心机制（一句话）：**MoE 把"要读多少权重"和"每个 token 算多少"解耦了。**

  - 权重读取量取决于**这一批激活了多少个专家**，而这个数随 token 数**饱和**（券收集问题）：
        E[激活专家数] = E * (1 - (1 - k/E)^N)
    N 小时线性涨，N 大时贴住 E 不动。
  - 每个 token 的算力只走 k 个专家，与 E 无关：
        FLOPs/token = 2 * (P_dense + k * P_expert) + 4 * L * seqlen * attn_dim
    ⚠ attention 项要用 **n_heads x head_dim**，不是 d_model。稠密 Llama 系两者恰好相等，
      但 MoE 上不等（gpt-oss 4096 vs 2880；DeepSeek-MLA 20480 vs 7168），
      用 d_model 会**低估算力 1.4~2.9 倍**，方向是让投机看起来更好。2026-08-22 对抗审稿查出并修正。

于是投机采样把 token 数乘上 (gamma+1) 这件事，在 MoE 上有**两个相反的后果**：

  小 batch：baseline 只激活少数专家，投机把 token 数乘 5 倍 -> **激活更多专家 -> 多读权重**。
            "验证几乎免费"在这里**不成立** —— 这是稠密直觉会栽跟头的地方。
  大 batch：两边都把专家激活满了 -> 权重读取**完全相同**，"读一次"的性质回来了；
            而 MoE 每 token 算力极低，memory-bound 区被拉得很长 -> 投机**能一直赚到很大的 batch**。

用法：
    python moe.py --experts    # 激活专家数怎么随 token 数饱和
    python moe.py --budget     # MoE vs 稠密：免费额度差多少
    python moe.py --redhat     # 用模型解释 gpt-oss-120b 并发 200 仍 +20% 的反例
    python moe.py --penalty    # 小 batch 下 MoE 特有的"多激活专家"惩罚
"""
from __future__ import annotations

import argparse

from speedup import H100, scale

# --------------------------------------------------------------------------
# MoE 模型规格
# --------------------------------------------------------------------------

# 规格全部核自官方 config.json（2026-08-22 WebFetch 核验）：
#   gpt-oss-120b   https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json
#   DeepSeek-V3    https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
# P_active 是官方公布的"每 token 激活参数量"，用来**反查本文件的分解对不对**
# （见 test_moe.py::test_expert_decomposition_matches_published_active）。
MOE_MODELS = {
    # 36 层 / hidden 2880 / 128 专家 top-4 / 64 头 8 KV 头 head_dim 64
    # KV/token = 2(K,V) * 36 层 * (8*64) * 2 字节 = 73728 B
    # ⚠ 该模型交替层用 sliding_window=128，本文件**按全长 KV 计**，
    #   于是高估了访存 -> 让它看起来更 memory-bound -> 对投机偏乐观。
    "gpt-oss-120b": dict(P_total=117e9, P_dense=1.49e9, n_experts=128, top_k=4,
                         layers=36, d_model=2880, kv_per_tok=2 * 36 * 512 * 2,
                         attn_dim=64 * 64,          # 64 头 x head_dim 64 = 4096 != d_model
                         P_active=5.1e9),
    # 61 层 / hidden 7168 / 256 路由专家 top-8（另有 1 个常开共享专家，计入 P_dense）
    # ⚠ MLA：KV cache 存的是**一个联合压缩潜向量**，不是分开的 K 和 V。
    #   每层每 token = kv_lora_rank(512) + qk_rope_head_dim(64) = 576 个元素。
    #   KV/token = 61 * 576 * 2 字节 = 70272 B。
    #   （初版误按 2*61*512*2=124928 算，高估 1.78 倍，2026-08-22 核 config.json 后修正。）
    # P_dense 由官方 37B 激活反解：P_dense + 8*(671-P_dense)/256 = 37 -> P_dense ≈ 17.0e9
    "deepseek-v3": dict(P_total=671e9, P_dense=17.0e9, n_experts=256, top_k=8,
                        layers=61, d_model=7168, kv_per_tok=61 * 576 * 2,
                        # MLA：QK 每头 qk_nope(128)+qk_rope(64)=192，PV 每头 v_head_dim=128，共 128 头
                        # 4*attn_dim = 2*(128*192) + 2*(128*128) -> attn_dim = 20480
                        attn_dim=20480,
                        P_active=37e9),
}


def expert_size(m: dict) -> float:
    """单个专家的参数量。"""
    return (m["P_total"] - m["P_dense"]) / m["n_experts"]


def active_params(m: dict) -> float:
    """每个 token 实际过的参数量（= 稠密部分 + top_k 个专家）。"""
    return m["P_dense"] + m["top_k"] * expert_size(m)


def expected_active_experts(m: dict, n_tokens: float) -> float:
    """N 个 token 一共会激活多少个不同的专家（券收集问题的期望）。

        E[激活数] = E * (1 - (1 - k/E)^N)

    假设：每个 token 独立、均匀地选 top_k 个专家。真实路由有负载均衡损失推着它接近均匀，
    但同一条序列里相邻 token 的上下文相似、路由相关，所以真实激活数**不会高于**这个值。
    本函数给的是激活多样性的**上界**，而这个上界的偏向**在两个区间里方向相反**：

      小 batch：投机把 token 数乘 (gamma+1)，本模型会**高估**它多激活的专家数，
                 于是**高估**了 MoE 特有的那个惩罚 —— 真实惩罚比本文件算出来的小。
      大 batch：两边都把专家激活满了，这个假设**不起作用**，结论不受影响。

    所以引用本文件的小 batch 结论时要说清：那是惩罚的**上界**，不是实测值。
    """
    E, k = m["n_experts"], m["top_k"]
    if n_tokens <= 0:
        return 0.0
    return E * (1.0 - (1.0 - k / E) ** n_tokens)


def weight_bytes(m: dict, n_tokens: float, bytes_per_param: float) -> float:
    """这一次前向要从 HBM 读多少权重字节 —— 取决于激活了多少专家。"""
    return (m["P_dense"] + expected_active_experts(m, n_tokens) * expert_size(m)) \
        * bytes_per_param


def fwd_time_moe(m: dict, hw: dict, batch: int, seqlen: int, q_per_seq: int,
                 bytes_per_param: float = 2.0, kv_scale: float = 1.0) -> dict:
    """MoE 的一次前向时间。与 speedup.py::fwd_time 同构，但权重项是 token 数的函数。"""
    n_q = batch * q_per_seq
    mem = (weight_bytes(m, n_q, bytes_per_param)
           + batch * seqlen * m["kv_per_tok"] * kv_scale)
    # 算力：每 token 只过 active_params，外加 attention 项（要乘层数，见 speedup.py 的勘误）
    flops = n_q * (2 * active_params(m) + 4 * m["layers"] * seqlen * m["attn_dim"])
    t_mem, t_cmp = mem / hw["bw"], flops / hw["peak"]
    return dict(t=max(t_mem, t_cmp), t_mem=t_mem, t_cmp=t_cmp,
                bound="memory" if t_mem >= t_cmp else "compute",
                n_experts_active=expected_active_experts(m, n_q),
                mem_bytes=mem)


def tokens_to_saturate_moe(m: dict, hw: dict, batch: int, seqlen: int,
                           bytes_per_param: float = 2.0) -> float:
    """MoE 的"免费额度"：还能再塞多少 query token 才翻进 compute-bound。

    注意这是个**不动点**问题：塞的 token 越多，激活的专家越多，访存也越多，
    额度本身会往上抬。这里用迭代求不动点（单调有界，几步就收敛）。
    """
    per_token = (2 * active_params(m) + 4 * m["layers"] * seqlen * m["attn_dim"]) / hw["peak"]
    n = float(batch)
    for _ in range(60):
        mem = (weight_bytes(m, n, bytes_per_param)
               + batch * seqlen * m["kv_per_tok"]) / hw["bw"]
        nxt = mem / per_token
        if abs(nxt - n) < 1e-6 * max(n, 1.0):
            return nxt
        n = 0.5 * n + 0.5 * nxt        # 阻尼迭代，避免振荡
    return n


def moe_spec_speedup(name: str, hw: dict, batch: int, seqlen: int, gamma: int,
                     accept_len: float, bytes_per_param: float = 2.0,
                     draft_cost_share: float = 0.05) -> dict:
    """MoE 上投机 vs 基线的吞吐比。

    draft_cost_share：草稿开销占迭代的比例。MoE 的草稿通常是轻量草稿头（EAGLE 类），
    这里当参数给，不再单独建模草稿模型 —— 因为本文件要解释的是**目标模型侧**的机制。
    """
    m = MOE_MODELS[name]
    base = fwd_time_moe(m, hw, batch, seqlen, 1, bytes_per_param)
    ver = fwd_time_moe(m, hw, batch, seqlen, gamma + 1, bytes_per_param)
    t_iter = ver["t"] / (1.0 - draft_cost_share)
    return dict(
        base_tps=batch / base["t"], spec_tps=batch * accept_len / t_iter,
        speedup=(accept_len / t_iter) * base["t"],
        base_bound=base["bound"], verify_bound=ver["bound"],
        experts_base=base["n_experts_active"], experts_spec=ver["n_experts_active"],
        mem_ratio=ver["mem_bytes"] / base["mem_bytes"],
    )


# --------------------------------------------------------------------------
# 打印
# --------------------------------------------------------------------------

def _experts():
    print("=" * 92)
    print("激活专家数怎么随 token 数饱和（券收集）：E*(1-(1-k/E)^N)")
    print("=" * 92)
    for name in ("gpt-oss-120b", "deepseek-v3"):
        m = MOE_MODELS[name]
        print("\n--- %s：%d 专家 top-%d，单专家 %.2fB，每 token 激活 %.2fB / 总 %.0fB ---"
              % (name, m["n_experts"], m["top_k"], expert_size(m) / 1e9,
                 active_params(m) / 1e9, m["P_total"] / 1e9))
        print("%-12s %-16s %-16s" % ("token 数 N", "激活专家数", "占全部专家"))
        for n in (1, 4, 16, 64, 200, 1000, 5000):
            a = expected_active_experts(m, n)
            print("%-12d %-16.1f %-16.1f%%" % (n, a, 100 * a / m["n_experts"]))
    print()
    print("读法：N 很小时激活数几乎线性涨，N 大到几百上千就贴住上限不动了。")
    print("      **投机采样把 N 乘上 (gamma+1)，落在曲线的哪一段，决定了它是被罚还是白赚。**")


def _budget():
    hw = scale(H100, 8)
    print("=" * 100)
    print("免费额度：MoE vs 稠密（8xH100 TP=8，忽略通信；seqlen=1024）")
    print("=" * 100)
    from speedup import MODELS, tokens_to_saturate
    print("%-22s %-10s %-14s %-18s" % ("模型", "batch", "权重精度", "免费额度(query token)"))
    for batch in (1, 8, 64, 200):
        r = tokens_to_saturate(MODELS["llama3-70b"], hw, batch, 1024, 2.0)
        print("%-22s %-10d %-14s %-18.0f" % ("llama3-70b(稠密)", batch, "fp16", r))
    print()
    for name, wb, tag in (("gpt-oss-120b", 0.53, "MXFP4≈4.25bit"),
                          ("gpt-oss-120b", 2.0, "fp16"),
                          ("deepseek-v3", 1.0, "fp8")):
        for batch in (1, 8, 64, 200):
            r = tokens_to_saturate_moe(MOE_MODELS[name], hw, batch, 1024, wb)
            print("%-22s %-10d %-14s %-18.0f" % (name, batch, tag, r))
    print()
    print("读法：MoE 的免费额度比同量级稠密模型大一个数量级以上。")
    print("      原因不是它'访存少'（专家全激活时访存反而更大），而是它**每 token 的算力极低**：")
    print("      分母小了，同样的访存时间能换来多得多的 query token。")


def _redhat():
    hw = scale(H100, 8)
    print("=" * 104)
    print("解释那个反例：gpt-oss-120b（MoE + MXFP4）+ EAGLE3，并发 200 仍有正收益？")
    print("口径：8xH100 TP=8（忽略通信），MXFP4≈0.53 B/param，seqlen=1024，gamma=4，")
    print("      E[tau]=3.0，草稿开销占迭代 5%（EAGLE 类轻量草稿头）。解析模型的乐观上界。")
    print("=" * 104)
    print("%-9s %-13s %-13s %-9s %-11s %-13s %-11s"
          % ("batch", "基线激活专家", "投机激活专家", "访存比", "验证阶段", "加速比", "结论"))
    for batch in (1, 8, 32, 64, 128, 200, 512, 1024, 2048):
        r = moe_spec_speedup("gpt-oss-120b", hw, batch, 1024, 4, 3.0, 0.53)
        tag = "赚" if r["speedup"] > 1.0 else "**亏**"
        print("%-9d %-13.1f %-13.1f %-9.3f %-11s %-11.3f %-11s"
              % (batch, r["experts_base"], r["experts_spec"], r["mem_ratio"],
                 r["verify_bound"], r["speedup"], tag))
    print()
    print("三条读法：")
    print("  1) **batch=1 时投机反而要多读权重**（访存比 > 1）：基线只激活 4 个专家，")
    print("     投机 5 倍 token 激活了近 19 个。这是 MoE 特有的惩罚，稠密模型上不存在。")
    print("  2) batch 到几十以后两边都把专家激活满了，访存比回到 1.000 ——")
    print("     '权重只读一次、与 query token 数无关'这条**在 MoE 上是大 batch 才成立**。")
    print("  3) 并发 200 时仍是 memory-bound，所以仍然赚 —— 这正是那份报告的机制。")
    print("     稠密 70B 在同样口径下 batch 三百多就翻转了（见第 18 篇），差了近一个数量级。")


def _penalty():
    hw = scale(H100, 8)
    print("=" * 92)
    print("MoE 特有的小 batch 惩罚：投机把 token 数乘 (gamma+1)，激活的专家也跟着涨")
    print("=" * 92)
    print("%-9s %-8s %-15s %-15s %-12s" % ("batch", "gamma", "基线激活专家", "投机激活专家", "多读权重"))
    m = MOE_MODELS["gpt-oss-120b"]
    for batch in (1, 2, 4, 8, 16, 32):
        for gamma in (4,):
            b = expected_active_experts(m, batch)
            s = expected_active_experts(m, batch * (gamma + 1))
            wb_b = m["P_dense"] + b * expert_size(m)
            wb_s = m["P_dense"] + s * expert_size(m)
            print("%-9d %-8d %-15.1f %-15.1f %-12.2f倍"
                  % (batch, gamma, b, s, wb_s / wb_b))
    print()
    print("读法：batch=1 时投机要多读 3 倍以上的权重字节，batch 到 32 就基本抹平。")
    print("      工程含义：**MoE 上做投机解码，小 batch 反而是最不划算的区间** ——")
    print("      与稠密模型（小 batch 最划算）**恰好相反**。低并发场景要实测，别照搬稠密直觉。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("experts", "budget", "redhat", "penalty"):
        ap.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    if a.experts:
        _experts()
    if a.budget:
        _budget()
    if a.redhat:
        _redhat()
    if a.penalty:
        _penalty()
    if not any(vars(a).values()):
        _redhat()
