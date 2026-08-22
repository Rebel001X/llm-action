"""
speedup.py —— 投机采样的**物理账**：roofline + KV 流量 + batch 崩塌点。

Leviathan 的理想加速比 E[tau]/(gamma*c+1) 把"目标模型验证 gamma+1 个 token 与
生成 1 个 token 同价"当成了公理。这条只在 **memory-bound 区**成立。
本文件把这句话变成可算的数：

    单次前向时间 T = max( 访存字节 / 带宽 ,  浮点数 / 峰值算力 )
      访存字节 = 权重 P*b  +  KV 读 batch*seqlen*kv_per_token
      浮点数   = n_query_tokens * (2*P  +  4*layers*seqlen*d_model)

关键结构：**权重只读一次，与 query token 数无关**（这是投机采样能成立的全部理由）；
**算力却与 query token 数成正比**（这是它在大 batch 下崩塌的全部理由）。

三条铁律里的铁律二（加速比不许裸奔）就是从这里来的：同一个 alpha、同一个 gamma，
换个 batch 或换个 seqlen，结论可以从"提速 2.5 倍"翻转成"降速 40%"。

口径声明（本文件所有数字）：
  - 硬件 H100 SXM5：HBM3 带宽 3.35 TB/s，BF16 **稠密**峰值 989.5 TFLOPS
    （不是含稀疏的 1979 —— 峰值算力不许裸奔，见 AI 芯片对比库的同名铁律）
  - 精度 fp16/bf16 权重与 KV
  - 只算 GEMM 与 attention 的主项，忽略 norm/softmax/kernel launch/调度开销
    => 因此本模型给出的是**乐观上界**，真实系统的崩塌点只会来得更早

用法：
    python speedup.py --roofline   # 屋脊点与"验证 k 个 token 要多久"
    python speedup.py --batch      # batch 扫描：什么时候投机开始亏
    python speedup.py --breakeven  # 保本所需的接受长度
    python speedup.py --quant      # 量化与投机采样是叠加还是竞争
    python speedup.py --chunked    # chunked prefill 与投机采样抢同一份免费额度
"""
from __future__ import annotations

import argparse

# ---------------------------------------------------------------- 硬件与模型

H100 = dict(name="H100 SXM5", bw=3.35e12, peak=989.5e12, hbm=80e9)  # B/s, FLOP/s(BF16稠密), B

def scale(hw: dict, n_gpu: int) -> dict:
    """张量并行的一阶近似：带宽/算力/容量都乘 n_gpu，**忽略通信开销**。

    忽略通信让本模型继续保持"乐观上界"的性质：真实 TP 有 all-reduce，
    而 all-reduce 恰恰是 decode 阶段占比很高的一项，所以真实崩塌点比这里更早。
    """
    return dict(name="%s x%d" % (hw["name"], n_gpu), bw=hw["bw"] * n_gpu,
                peak=hw["peak"] * n_gpu, hbm=hw["hbm"] * n_gpu, n_gpu=n_gpu)


def feasible(model: dict, hw: dict, batch: int, seqlen: int,
             bytes_per_param: float = 2.0, draft: dict | None = None) -> tuple[bool, float]:
    """这套 (batch, seqlen) 在显存里放得下吗？返回 (是否可行, 占用 GB)。

    权重 + KV cache（+ 草稿模型的权重与 KV）。忽略激活与碎片，因此仍是乐观上界。
    """
    need = model["P"] * bytes_per_param + batch * seqlen * model["kv_per_tok"]
    if draft is not None:
        need += draft["P"] * bytes_per_param + batch * seqlen * draft["kv_per_tok"]
    return need <= hw["hbm"], need / 1e9

MODELS = {
    "llama3-8b":   dict(P=8.03e9,  layers=32, d_model=4096, kv_per_tok=2 * 32 * 1024 * 2),
    "llama3-70b":  dict(P=70.6e9,  layers=80, d_model=8192, kv_per_tok=2 * 80 * 1024 * 2),
    "llama3.2-1b": dict(P=1.24e9,  layers=16, d_model=2048, kv_per_tok=2 * 16 * 512 * 2),
    "eagle-head":  dict(P=0.6e9,   layers=1,  d_model=4096, kv_per_tok=2 * 1 * 1024 * 2),
}


def fwd_time(model: dict, hw: dict, batch: int, seqlen: int, q_per_seq: int,
             bytes_per_param: float = 2.0, kv_scale: float = 1.0) -> dict:
    """一次前向的时间（秒）与它是被什么卡住的。

    q_per_seq: 每条序列这次喂进去几个 query token
               （baseline decode = 1；投机验证 = gamma+1；draft 起草 = 1）
    """
    n_q = batch * q_per_seq
    mem_bytes = (model["P"] * bytes_per_param
                 + batch * seqlen * model["kv_per_tok"] * kv_scale)
    # attention 的浮点数必须乘层数 L：每层每个 query token 是 QK^T 与 PV 各 2*seqlen*d_model。
    # 早期版本漏了 L（由写第 03 篇的过程中查出），s=1024 时该项只占 2P 的 0.02% 无关痛痒，
    # 但 s=32768 时占到 60% 以上 —— 长上下文的结论会被它改写，所以必须算对。
    flops = n_q * (2 * model["P"] + 4 * model["layers"] * seqlen * model["d_model"])
    t_mem, t_cmp = mem_bytes / hw["bw"], flops / hw["peak"]
    return dict(t=max(t_mem, t_cmp), t_mem=t_mem, t_cmp=t_cmp,
                bound="memory" if t_mem >= t_cmp else "compute",
                intensity=flops / mem_bytes)


def spec_throughput(target: str, draft: str, hw: dict, batch: int, seqlen: int,
                    gamma: int, accept_len: float, wbytes: float = 2.0,
                    kv_scale: float = 1.0) -> dict:
    """投机采样 vs 朴素解码的吞吐（tokens/s，全 batch 合计）。

    accept_len = 一次迭代的期望产出 token 数 E[tau]（含赠品 token），1 <= E[tau] <= gamma+1。
    草稿是**串行**跑 gamma 次前向；目标模型跑 1 次、每条序列喂 gamma+1 个 query token。
    """
    tm, dm = MODELS[target], MODELS[draft]
    kw = dict(bytes_per_param=wbytes, kv_scale=kv_scale)
    base = fwd_time(tm, hw, batch, seqlen, 1, **kw)
    t_draft = sum(fwd_time(dm, hw, batch, seqlen + i, 1, **kw)["t"] for i in range(gamma))
    verify = fwd_time(tm, hw, batch, seqlen, gamma + 1, **kw)
    t_iter = t_draft + verify["t"]
    return dict(
        base_tps=batch / base["t"],
        spec_tps=batch * accept_len / t_iter,
        speedup=(accept_len / t_iter) * base["t"],
        base_bound=base["bound"], verify_bound=verify["bound"],
        t_draft_share=t_draft / t_iter,
    )


def breakeven_accept_len(target, draft, hw, batch, seqlen, gamma) -> float:
    """要不亏本，E[tau] 至少得多大。"""
    tm, dm = MODELS[target], MODELS[draft]
    base = fwd_time(tm, hw, batch, seqlen, 1)["t"]
    t_draft = sum(fwd_time(dm, hw, batch, seqlen + i, 1)["t"] for i in range(gamma))
    t_verify = fwd_time(tm, hw, batch, seqlen, gamma + 1)["t"]
    return (t_draft + t_verify) / base


# ---------------------------------------------------------------- 打印

def _roofline():
    hw = H100
    print("=" * 92)
    print("屋脊点与「验证 k 个 token 要多久」  硬件=%s  BW=%.2f TB/s  峰值(稠密)=%.1f TFLOPS"
          % (hw["name"], hw["bw"] / 1e12, hw["peak"] / 1e12))
    print("=" * 92)
    print("屋脊点 = 峰值/带宽 = %.1f FLOP/Byte\n" % (hw["peak"] / hw["bw"]))
    for name in ("llama3.2-1b", "llama3-8b", "llama3-70b"):
        m = MODELS[name]
        print("--- %s (P=%.2fB, KV=%.0f KiB/token) ---"
              % (name, m["P"] / 1e9, m["kv_per_tok"] / 1024))
        print("%-8s %-8s %-11s %-11s %-11s %-9s %-10s"
              % ("batch", "seqlen", "q/seq", "t_mem(ms)", "t_cmp(ms)", "bound", "T(ms)"))
        for batch, seqlen in ((1, 1024), (1, 32768), (64, 1024), (256, 1024), (256, 32768)):
            for q in (1, 5):
                r = fwd_time(m, hw, batch, seqlen, q)
                print("%-8d %-8d %-11d %-11.3f %-11.3f %-9s %-10.3f"
                      % (batch, seqlen, q, r["t_mem"] * 1e3, r["t_cmp"] * 1e3, r["bound"], r["t"] * 1e3))
        print()
    print("读法：q/seq 从 1 变 5（验证 5 个 token），只要还在 memory 区，T 几乎不动 —— 验证是白送的。")
    print("      一旦进入 compute 区（大 batch 或长序列），T 就跟着 q/seq 线性涨 —— 白送结束。")


def _batch():
    hw = scale(H100, 8)   # 70B fp16 权重 141 GB，单卡 80 GB 放不下；这里用 TP=8 以便扫到大 batch
    print("=" * 108)
    print("batch 扫描：投机采样什么时候开始亏  (target=llama3-70b, draft=llama3.2-1b, gamma=4, TP=8)")
    print("口径：8xH100（带宽/算力/容量按 8 倍线性放大，忽略通信），fp16，E[tau]=3.0 固定，只改 batch")
    print("=" * 108)
    for seqlen in (1024, 16384):
        print("\n--- seqlen = %d ---" % seqlen)
        print("%-8s %-13s %-13s %-10s %-13s %-12s %-12s"
              % ("batch", "基线 tok/s", "投机 tok/s", "加速比", "验证阶段", "草稿占迭代", "显存占用"))
        for batch in (1, 16, 64, 128, 256, 384, 512, 768, 1024):
            ok, gb = feasible(MODELS["llama3-70b"], hw, batch, seqlen,
                              draft=MODELS["llama3.2-1b"])
            if not ok:
                print("%-8d %-13s %-13s %-10s %-13s %-12s %-12s"
                      % (batch, "-", "-", "-", "-", "-", "%.0fGB 装不下" % gb))
                continue
            r = spec_throughput("llama3-70b", "llama3.2-1b", hw, batch, seqlen, 4, 3.0)
            flag = "  <== 亏了" if r["speedup"] < 1.0 else ""
            print("%-8d %-13.1f %-13.1f %-10.3f %-13s %-12.1f%% %-12s%s"
                  % (batch, r["base_tps"], r["spec_tps"], r["speedup"],
                     r["verify_bound"], r["t_draft_share"] * 100, "%.0fGB" % gb, flag))
    print()
    print("读法一：同一个 E[tau]=3.0，seqlen=1024 时 bs=1 加速 2.8x，bs 大到某处翻转成减速。")
    print("        翻转的原因不是接受率变了，是**验证前向从 memory-bound 掉进了 compute-bound**。")
    print("读法二：seqlen=16384 那一段**不翻转** —— 长上下文的 KV 读把前向死死钉在 memory 区，")
    print("        验证依旧近乎白送。但它换来另一个约束：显存装不下大 batch（见最后一列）。")
    print("        所以「投机解码在大 batch 下没用」这句流行说法是有条件的，条件是**短上下文**。")
    print("读法三：任何不同时报 batch 和 seqlen 的加速比数字都不可用（铁律二）。")


def _breakeven():
    hw = scale(H100, 4)
    print("=" * 96)
    print("保本线：E[tau] 至少要多大，投机才不亏  (target=llama3-70b, gamma=4)")
    print("=" * 96)
    for draft in ("llama3.2-1b", "eagle-head"):
        print("\n--- draft = %s (P=%.2fB) ---" % (draft, MODELS[draft]["P"] / 1e9))
        print("%-10s" % "batch", "".join("%14s" % ("seqlen=%d" % s) for s in (1024, 8192, 32768)))
        for batch in (1, 8, 32, 64, 128, 256):
            cells = []
            for seqlen in (1024, 8192, 32768):
                ok, gb = feasible(MODELS["llama3-70b"], hw, batch, seqlen,
                                  draft=MODELS[draft])
                if not ok:
                    cells.append("%14s" % "装不下")
                    continue
                be = breakeven_accept_len("llama3-70b", draft, hw, batch, seqlen, 4)
                cells.append("%14s" % ("%.2f%s" % (be, "!" if be > 5 else "")))
            print("%-10d" % batch, "".join(cells))
    print()
    print("读法：gamma=4 时 E[tau] 的上限是 5（全接受+赠品）。表里带 ! 的格子（>5）意味着")
    print("      **无论接受率多高都不可能保本** —— 这就是「投机解码在大 batch 下必须关掉」的由来。")
    print("      对比两个 draft 能看出：草稿越轻（eagle-head 0.6B vs 1.24B），保本线越低、可用区间越宽。")


def tokens_to_saturate(model: dict, hw: dict, batch: int, seqlen: int,
                      wbytes: float = 2.0, kv_scale: float = 1.0) -> float:
    """这一批还能再塞多少个 query token 才会从 memory-bound 翻进 compute-bound。

    用途：chunked prefill 会把 prefill 的 chunk 混进同一次前向。
    如果 chunk 大小已经超过这个数，那"投机验证几乎免费"的前提就当场失效 ——
    因为免费的额度已经被 prefill chunk 吃光了。
    """
    mem = (model["P"] * wbytes + batch * seqlen * model["kv_per_tok"] * kv_scale) / hw["bw"]
    per_token = (2 * model["P"] + 4 * model["layers"] * seqlen * model["d_model"]) / hw["peak"]
    return mem / per_token


def _max_feasible_batch(target, draft, hw, seqlen, wbytes=2.0, bmax=8192) -> int:
    """显存装得下的最大 batch（二分）。给报表用，免得报出物理上开不起来的 batch。"""
    lo, hi = 1, bmax
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(MODELS[target], hw, mid, seqlen, wbytes, MODELS[draft])[0]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def crossover_batch(target, draft, hw, seqlen, gamma, accept_len,
                    wbytes=2.0, kv_scale=1.0, bmax=8192,
                    require_feasible: bool = False) -> int | None:
    """二分找加速比跌破 1.0 的最小 batch（找不到返回 None）。

    require_feasible=True 时，只在**显存装得下**的 batch 范围内找 —— 否则会报出
    物理上根本开不起来的 batch。初版没有这个开关，于是 `--quant` 的长上下文行
    全部打印 ">8192"，而同口径下最大可行 batch 只有一两百（2026-08-22 对抗审稿指出）。
    """
    if require_feasible:
        lo_f, hi_f = 1, bmax
        while lo_f < hi_f:
            mid = (lo_f + hi_f + 1) // 2
            if feasible(MODELS[target], hw, mid, seqlen, wbytes, MODELS[draft])[0]:
                lo_f = mid
            else:
                hi_f = mid - 1
        bmax = lo_f
    lo, hi = 1, bmax
    if spec_throughput(target, draft, hw, hi, seqlen, gamma, accept_len,
                       wbytes, kv_scale)["speedup"] >= 1.0:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if spec_throughput(target, draft, hw, mid, seqlen, gamma, accept_len,
                           wbytes, kv_scale)["speedup"] < 1.0:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _chunked():
    """chunked prefill 与投机采样抢的是同一份"免费额度"。"""
    hw = scale(H100, 8)
    print("=" * 96)
    print("chunked prefill x 投机采样：免费额度只有一份，谁先来谁吃掉")
    print("口径：target=llama3-70b，8xH100 TP=8，fp16，解析模型的乐观上界")
    print("=" * 96)
    print("%-10s %-10s %-22s %-22s %-16s"
          % ("batch", "seqlen", "本批总免费额度(token)", "投机需要(γ=4)", "留给prefill chunk"))
    for batch in (1, 8, 32, 64, 128):
        for seqlen in (1024, 8192):
            cap = tokens_to_saturate(MODELS["llama3-70b"], hw, batch, seqlen)
            need = batch * 5
            left = cap - need
            print("%-10d %-10d %-22.0f %-22d %-16s"
                  % (batch, seqlen, cap, need,
                     "%.0f" % left if left > 0 else "**已透支 %.0f**" % (-left)))
    print()
    print("读法：中间那列是「这一批在翻进 compute-bound 之前还能吃下多少个 query token」。")
    print("  1) 投机解码要占掉 batch*(γ+1) 个 —— 这是它买信息的开销。")
    print("  2) 剩下的才是能留给 prefill chunk 的。而主流引擎的 chunk 大小通常是 512~2048，")
    print("     对照最后一列可以看到：**中等 batch 下，一个 chunk 就能把额度吃光**。")
    print("  3) 两者同时开启时，谁都不是免费的：prefill chunk 把前向推进 compute 区，")
    print("     投机验证的 (γ+1) 倍算力就要按全价付钱（见第 18 篇的渐近线）。")
    print()
    print("这不是说两者不能共存，而是说**它们的收益不叠加**，调优时必须放在一张账本上算。")


def _quant():
    """量化与投机采样：叠加还是竞争？

    结论（本函数负责把它算出来）：**竞争**。两者薅的是同一份"访存冗余"。
    权重量化把访存项砍小 -> 前向更早离开 memory-bound 区 -> 投机的可用 batch 区间缩小。
    """
    hw = scale(H100, 8)
    print("=" * 104)
    print("量化 x 投机采样：抢的是同一份访存冗余（target=llama3-70b, draft=llama3.2-1b, gamma=4, E[tau]=3.0）")
    print("口径：8xH100 TP=8（忽略通信），解析模型的乐观上界，非硬件实测")
    print("=" * 104)
    print("%-14s %-9s %-13s %-13s %-11s %-14s"
          % ("权重精度", "KV精度", "seqlen", "基线tok/s(bs=1)", "bs=1加速比", "跌破1.0的batch"))
    for wname, wb in (("fp16 (2B)", 2.0), ("fp8  (1B)", 1.0), ("int4 (0.5B)", 0.5)):
        for kvname, kvs in (("fp16", 1.0), ("fp8", 0.5)):
            for seqlen in (1024, 16384):
                r1 = spec_throughput("llama3-70b", "llama3.2-1b", hw, 1, seqlen, 4, 3.0, wb, kvs)
                # 只在**显存装得下**的 batch 范围内找交叉点 —— 否则会报出物理上开不起来的 batch。
                # 初版没做这件事，长上下文六行全打印 ">8192"，而同口径最大可行 batch 只有一两百
                # （2026-08-22 对抗审稿指出的口径裸奔）。
                cb = crossover_batch("llama3-70b", "llama3.2-1b", hw, seqlen, 4, 3.0, wb, kvs,
                                     require_feasible=True)
                bmax = crossover_batch.__globals__["_max_feasible_batch"](
                    "llama3-70b", "llama3.2-1b", hw, seqlen, wb)
                tag = str(cb) if cb else "无（可行区间内不翻转，最大可行 batch=%d）" % bmax
                print("%-14s %-9s %-13d %-13.1f %-11.3f %-14s"
                      % (wname, kvname, seqlen, r1["base_tps"], r1["speedup"], tag))
    print()
    print("读法（三条，第一条与「量化和投机可以叠加」的直觉相反）：")
    print("  1) **权重量化会缩小投机采样的可用区间**：seqlen=1024 下，fp16 的交叉点在 batch 三百多，")
    print("     换成 int4 之后交叉点大幅左移。原因：量化把访存项砍小，前向更早进入 compute-bound，")
    print("     而「验证几乎免费」恰恰只在 memory-bound 区成立。**两者薅的是同一份冗余。**")
    print("  2) 但量化让**基线本身变快**（同一列的基线 tok/s 随精度下降而上升）。")
    print("     所以正确的比较不是「投机加速比有没有变小」，而是「两条路各自的绝对吞吐哪个高」。")
    print("  3) KV 量化在长上下文下影响更大（它砍的是 batch*seqlen 那一项），")
    print("     同样是把前向往 compute 区推 —— 长上下文原本是投机采样最舒服的区间，")
    print("     KV 量化会把这份舒服吃掉一部分。")
    print()
    print("注意：本模型只算访存与算力主项。真实量化还会带来反量化开销、")
    print("      kernel 效率差异与精度损失，这些都没算进来。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofline", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--breakeven", action="store_true")
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--chunked", action="store_true")
    a = ap.parse_args()
    if a.roofline:
        _roofline()
    if a.batch:
        _batch()
    if a.breakeven:
        _breakeven()
    if a.quant:
        _quant()
    if a.chunked:
        _chunked()
    if not (a.roofline or a.batch or a.breakeven or a.quant or a.chunked):
        _roofline()
