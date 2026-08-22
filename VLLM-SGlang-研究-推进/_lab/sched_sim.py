"""sched_sim.py —— 调度策略的**差分模拟器**：量化「缓存友好」与「公平性」之间的兑换率。

════════════════════════════════════════════════════════════════════════
它回答的唯一问题
════════════════════════════════════════════════════════════════════════
SGLang 的招牌调度策略 LPM（longest prefix match，最长前缀优先）把「缓存友好」
直接做进了排序。代价是什么？**无共享前缀的请求会不会被饿死？**
以及：LMDeploy 那套 age credit 能不能在几乎不损失命中率的前提下把公平性救回来？

  * `sglang:python/sglang/srt/managers/schedule_policy.py:200` 起是 `CacheAwarePolicy`
    枚举（`LPM = "lpm"`、`DFS_WEIGHT = "dfs-weight"`），
    `:207` 起是 `CacheAgnosticPolicy`（`FCFS`、`RANDOM`）。
    **全文件搜不到 starvation / aging 相关机制** —— 这是本模拟器的出发点。
  * `lmdeploy:lmdeploy/pytorch/paging/scheduler.py:158` docstring 原文
    "Prefer smaller long prompts, with age credit to avoid starvation."，
    公式在 `:162`-`:163`：`age_credit = int(wait_age // seconds_per_chunk)`，
    然后 `age_adjusted_chunks = estimated_long_chunks - age_credit`。
    —— 也就是**等得越久，排序分数越低（越优先）**。本模拟器照这个形状建 `lpm_aged`。

════════════════════════════════════════════════════════════════════════
它**不是**什么
════════════════════════════════════════════════════════════════════════
* 不测性能。步进是抽象步，不是毫秒；没有 GPU、没有算子、没有显存带宽。
* 不建模 chunked prefill 的细节、不建模抢占、不建模 overlap 调度。
* 所以它只能回答**排序策略之间的相对关系**：谁的命中率高、谁让谁等得久。
  "SGLang 比 vLLM 快多少"这类问题这里一个字都答不了。

用法:
    python sched_sim.py             # 跑全部实验，落 out/sched_sim.json
    python sched_sim.py --selftest
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from common import OUT
from prefix_sim import Rng, SglangRadixCache


@dataclass
class Req:
    rid: int
    arrive: int
    tokens: list[int]
    out_len: int
    has_shared_prefix: bool          # 这条请求是否属于"有共享前缀"的那一族
    start: int = -1                  # 开始 prefill 的步
    done: int = -1
    hit: int = 0

    @property
    def wait(self) -> int:
        return (self.start - self.arrive) if self.start >= 0 else -1


def gen_arrivals(n_req: int, n_hot_families: int, hot_ratio_pct: int,
                 prefix_len: int, seed: int) -> list[Req]:
    """造到达序列。

    `hot_ratio_pct` 比例的请求来自 `n_hot_families` 个共享前缀族（缓存友好），
    其余是无共享前缀的独立请求（缓存不友好）——**后者正是可能被 LPM 饿死的那批**。
    到达时刻用 LCG 造，同 seed 逐位可复现。
    """
    r = Rng(seed)
    fams = [[7000 + f * 10000 + i for i in range(prefix_len)] for f in range(n_hot_families)]
    reqs: list[Req] = []
    t = 0
    for k in range(n_req):
        t += r.randint(0, 2)                      # 到达间隔 0~2 步
        if r.randint(1, 100) <= hot_ratio_pct:
            body = r.choice(fams) + [r.randint(1, 999) for _ in range(r.randint(8, 32))]
            shared = True
        else:
            body = [r.randint(20000, 90000) for _ in range(r.randint(120, 260))]
            shared = False
        reqs.append(Req(rid=k, arrive=t, tokens=body,
                        out_len=r.randint(16, 64), has_shared_prefix=shared))
    return reqs


# ────────────────────────────────────────────────── 排序策略

def score_fcfs(req: Req, hit: int, now: int) -> tuple:
    """先来先服务：只看到达时刻。"""
    return (req.arrive, req.rid)


def score_lpm(req: Req, hit: int, now: int) -> tuple:
    """最长前缀优先：命中越多越靠前（SGLang 的 LPM 形状）。**完全不看等待时长。**"""
    return (-hit, req.rid)


def score_lpm_aged(req: Req, hit: int, now: int, credit_per_step: int = 8) -> tuple:
    """LPM + age credit（照 LMDeploy `scheduler.py:162` 的形状）。

    等待每满 `credit_per_step` 步，就给这条请求补上相当于 `credit_per_step` 个
    命中 token 的分数。等得越久，排序分越低（越优先）—— 与 LMDeploy 的
    `age_adjusted_chunks = estimated_long_chunks - age_credit` 同构。
    """
    age = now - req.arrive
    credit = (age // credit_per_step) * credit_per_step
    return (-(hit + credit), req.rid)


def make_lpm_aged(credit_per_step: int, tiebreak: str = "arrival"):
    """LPM + age credit，**并列时的 tiebreak 可切换** —— 这是本模拟器最重要的一个旋钮。

    实验发现（见 `06-拓展方向/04-拓展优先级-亲笔.md`）：
    age credit 的公平性收益**大部分不来自"等得久就加分"这条公式本身**，
    而来自「粗粒度量化（T 大）把很多请求压成同一个分数 → 并列 → tiebreak 生效」。
    把 tiebreak 从"到达序"换成"反 FCFS"，T 保持 32 不变，
    公平差中位数就从 70.5 弹回 111.0 —— 真正在防饥饿的是 tiebreak，不是 credit。

    tiebreak:
      "arrival"  —— 按 rid（到达序），等价于并列时退回 FCFS
      "anti"     —— 按 -hit（命中多的优先），用来做对照、证明上面那条
    """
    def f(req: Req, hit: int, now: int) -> tuple:
        credit = ((now - req.arrive) // credit_per_step) * credit_per_step
        second = req.rid if tiebreak == "arrival" else -hit
        return (-(hit + credit), second)
    return f


POLICIES = {
    "fcfs": score_fcfs,
    "lpm": score_lpm,
    "lpm_aged": score_lpm_aged,
}


# ────────────────────────────────────────────────── 引擎（抽象到只剩调度）

@dataclass
class SimResult:
    policy: str = ""
    n_req: int = 0
    steps: int = 0
    prompt_tokens: int = 0
    hit_tokens: int = 0
    hit_rate: float = 0.0
    wait_p50: int = 0
    wait_p90: int = 0
    wait_p99: int = 0
    wait_max: int = 0
    wait_max_shared: int = 0        # 有共享前缀那一族的最大等待
    wait_max_unshared: int = 0      # **无共享前缀那一族的最大等待 —— 饥饿看这个**
    wait_p50_shared: int = 0
    wait_p50_unshared: int = 0
    wait_p90_unshared: int = 0      # 比 max 稳，用它当饥饿的主指标
    fairness_gap: int = 0           # 无共享 p90 − 有共享 p90，越大越不公平
    starved: int = 0                # 等待超过阈值的请求数
    starved_unshared: int = 0
    completed: int = 0
    detail: dict = field(default_factory=dict)


def simulate(reqs_proto: list[Req], policy, *, token_budget: int = 2048,
             max_running: int = 16, cache_tokens: int = 32768,
             starve_threshold: int = 200, max_steps: int = 20000) -> SimResult:
    """一步一步跑：每步先从等待队列按策略挑一批做 prefill，再让在跑的请求各出一个 token。"""
    reqs = [Req(r.rid, r.arrive, list(r.tokens), r.out_len, r.has_shared_prefix)
            for r in reqs_proto]
    cache = SglangRadixCache(page_size=1, capacity_tokens=cache_tokens)
    scorer = POLICIES[policy] if isinstance(policy, str) else policy
    policy_name = policy if isinstance(policy, str) else getattr(policy, "_name", "custom")

    waiting: list[Req] = []
    running: list[Req] = []
    pending = sorted(reqs, key=lambda r: r.arrive)
    idx = 0
    step = 0
    res = SimResult(policy=policy_name, n_req=len(reqs))

    while step < max_steps:
        while idx < len(pending) and pending[idx].arrive <= step:
            waiting.append(pending[idx]); idx += 1

        if waiting and len(running) < max_running:
            # 探测每条等待请求当前能命中多少（不改缓存状态）
            probed = [(r, _probe(cache, r.tokens)) for r in waiting]
            probed.sort(key=lambda pr: scorer(pr[0], pr[1], step))
            budget = token_budget
            admitted: list[Req] = []
            for r, hit in probed:
                if len(running) + len(admitted) >= max_running:
                    break
                cost = len(r.tokens) - hit
                if cost > budget:
                    continue
                budget -= cost
                r.start = step
                r.hit = cache.process(r.tokens)     # 真正插入缓存
                admitted.append(r)
            for r in admitted:
                waiting.remove(r)
                running.append(r)

        for r in list(running):
            r.out_len -= 1
            if r.out_len <= 0:
                r.done = step
                running.remove(r)
                res.completed += 1

        if idx >= len(pending) and not waiting and not running:
            break
        step += 1

    res.steps = step
    done = [r for r in reqs if r.start >= 0]
    res.prompt_tokens = sum(len(r.tokens) for r in done)
    res.hit_tokens = sum(r.hit for r in done)
    res.hit_rate = round(res.hit_tokens / res.prompt_tokens, 4) if res.prompt_tokens else 0.0

    waits = sorted(r.wait for r in done)
    if waits:
        res.wait_p50 = waits[len(waits) // 2]
        res.wait_p90 = waits[int(len(waits) * 0.9) - 1]
        res.wait_p99 = waits[int(len(waits) * 0.99) - 1]
        res.wait_max = waits[-1]
    sh = [r.wait for r in done if r.has_shared_prefix]
    un = [r.wait for r in done if not r.has_shared_prefix]
    res.wait_max_shared = max(sh) if sh else 0
    res.wait_max_unshared = max(un) if un else 0
    def _pct(xs, q):
        if not xs:
            return 0
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * q))]
    res.wait_p50_shared = _pct(sh, 0.5)
    res.wait_p50_unshared = _pct(un, 0.5)
    res.wait_p90_unshared = _pct(un, 0.9)
    res.fairness_gap = _pct(un, 0.9) - _pct(sh, 0.9)
    res.starved = sum(1 for w in waits if w > starve_threshold)
    res.starved_unshared = sum(1 for w in un if w > starve_threshold)
    res.detail = {"n_shared": len(sh), "n_unshared": len(un),
                  "never_started": len(reqs) - len(done)}
    return res


def _probe(cache: SglangRadixCache, toks: list[int]) -> int:
    """只看能命中多少，不改树 —— 调度排序阶段不该有副作用。"""
    node, key, hit = cache.root, tuple(toks), 0
    while key and key[0] in node.children:
        child = node.children[key[0]]
        m = cache._match_len(child.key, key)
        if m == 0:
            break
        hit += m
        key = key[m:]
        if m < len(child.key):
            break
        node = child
    return hit


# ────────────────────────────────────────────────── 实验

SEEDS = [11, 23, 37, 53, 71, 97, 113, 131]     # 8 个种子；单种子的百分位噪声太大

# 缓存容量是**决定 LPM 值不值**的关键自变量，必须扫：
# 容量充裕时 LPM 一分命中率都不多赚，只是重新分配延迟。
CAPACITIES = [32768, 4096, 1024, 512]

SCENARIOS = [
    # (名字, 请求数, 热族数, 热请求占比%, 共享前缀长度)
    ("half_shared",   150, 2, 50, 128),
    ("mostly_shared", 150, 2, 80, 128),   # 少数派最容易被饿死
    ("few_shared",    150, 2, 20, 128),
]

POLICY_ARMS = [
    ("fcfs", None),
    ("lpm", None),
    ("aged_T32", None),          # 粗量化 + 到达序 tiebreak
    ("aged_T8", None),
    ("aged_T1", None),           # 细量化 ⇒ 几乎无并列 ⇒ tiebreak 失效
    ("aged_T32_antitie", None),  # **对照**：只把 tiebreak 换掉，T 不变
]


def _arm(name):
    """名字 → scorer。`_antitie` 后缀表示把并列时的 tiebreak 换成反 FCFS。"""
    if name.startswith("aged_T"):
        anti = name.endswith("_antitie")
        t = int(name.replace("_antitie", "").split("T")[1])
        return make_lpm_aged(t, tiebreak="anti" if anti else "arrival")
    return name


def _mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return round(m, 4), round(var ** 0.5, 4)


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    out = []
    for name, n, fam, ratio, plen in SCENARIOS:
        for cap in CAPACITIES:
            row = {"scenario": name, "n_req": n, "hot_families": fam,
                   "shared_ratio_pct": ratio, "prefix_len": plen,
                   "cache_tokens": cap, "seeds": len(SEEDS), "policies": {}}
            for arm, _ in POLICY_ARMS:
                hits, gaps, p90un, p90sh = [], [], [], []
                for sd in SEEDS:
                    proto = gen_arrivals(n, fam, ratio, plen, sd)
                    r = simulate(proto, _arm(arm), cache_tokens=cap)
                    hits.append(r.hit_rate); gaps.append(r.fairness_gap)
                    p90un.append(r.wait_p90_unshared)
                    p90sh.append(r.wait_p90 if not r.wait_p90_unshared else
                                 r.wait_p90_unshared - r.fairness_gap)
                hm, hs = _mean_std(hits)
                gm, gs = _mean_std(gaps)
                um, us = _mean_std(p90un)
                row["policies"][arm] = {
                    "hit_rate_mean": hm, "hit_rate_std": hs,
                    "fairness_gap_mean": gm, "fairness_gap_std": gs,
                    "unshared_p90_wait_mean": um, "unshared_p90_wait_std": us,
                }
            out.append(row)

    payload = {
        "disclaimer": ("**策略模拟，不是性能实测。** 步进是抽象步不是毫秒；"
                       "无 GPU、无算子、无显存带宽；不建模 chunked prefill/抢占/overlap。"
                       "只能回答排序策略之间的相对关系。"),
        "modeled_from": {
            "sglang_policy_enum": "python/sglang/srt/managers/schedule_policy.py:200",
            "sglang_cache_agnostic": "python/sglang/srt/managers/schedule_policy.py:207",
            "lmdeploy_age_credit_doc": "lmdeploy/pytorch/paging/scheduler.py:158",
            "lmdeploy_age_credit_formula": "lmdeploy/pytorch/paging/scheduler.py:162",
        },
        "results": out,
    }
    fp = OUT / "sched_sim.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {fp}   （{len(SEEDS)} 个种子取均值±标准差）")
    print()
    hdr = (f"{'场景':<15}{'容量':>7}  {'策略':<14}{'命中率':>20}"
           f"{'公平差(无共享p90−共享p90)':>26}")
    print(hdr)
    print("-" * 96)
    for row in out:
        for pol, m in row["policies"].items():
            print(f"{row['scenario']:<15}{row['cache_tokens']:>7}  {pol:<14}"
                  f"{m['hit_rate_mean']:>13.4f}±{m['hit_rate_std']:<6.4f}"
                  f"{m['fairness_gap_mean']:>16.1f}±{m['fairness_gap_std']:<8.1f}")
        print()
    return 0


def selftest() -> int:
    ok = True

    # 1) 到达序列可复现
    a = gen_arrivals(30, 2, 50, 64, 11)
    b = gen_arrivals(30, 2, 50, 64, 11)
    if [(x.arrive, x.tokens, x.out_len) for x in a] != [(y.arrive, y.tokens, y.out_len) for y in b]:
        print("FAIL 到达序列不可复现"); ok = False

    # 2) 三个策略在**无任何共享前缀**时必须给出相同命中率（都是 0）
    proto = gen_arrivals(40, 1, 0, 64, 11)          # hot_ratio=0 → 全是独立请求
    rates = {p: simulate(proto, p).hit_rate for p in POLICIES}
    if set(rates.values()) != {0.0}:
        print(f"FAIL 无共享前缀时命中率应全为 0 -> {rates}"); ok = False

    # 3) 有共享前缀时，LPM 的命中率不应低于 FCFS（这是它存在的理由）
    proto2 = gen_arrivals(150, 2, 60, 128, 11)
    r_fcfs, r_lpm = simulate(proto2, "fcfs"), simulate(proto2, "lpm")
    if r_lpm.hit_rate < r_fcfs.hit_rate:
        print(f"FAIL LPM 命中率不该低于 FCFS -> lpm={r_lpm.hit_rate} fcfs={r_fcfs.hit_rate}")
        ok = False

    # 4) **核心断言（多种子）**：有缓存压力时 LPM 的命中率必须明显高于 FCFS，
    #    且 LPM 的公平差必须明显大于 FCFS —— 这两条一起才叫"用公平换命中"。
    #    **单种子会被百分位噪声骗**（我第一版就是这么被骗的），所以这里必须多种子。
    hs_l, hs_f, g_l, g_f = [], [], [], []
    for sd in (11, 23, 37, 53):
        pr = gen_arrivals(150, 2, 60, 128, sd)
        rl = simulate(pr, "lpm", cache_tokens=512)
        rf = simulate(pr, "fcfs", cache_tokens=512)
        hs_l.append(rl.hit_rate); hs_f.append(rf.hit_rate)
        g_l.append(rl.fairness_gap); g_f.append(rf.fairness_gap)
    if sum(hs_l) / 4 <= sum(hs_f) / 4:
        print(f"FAIL 有缓存压力时 LPM 命中率应高于 FCFS -> {hs_l} vs {hs_f}"); ok = False
    if sum(g_l) / 4 <= sum(g_f) / 4:
        print(f"FAIL LPM 的公平差应大于 FCFS（这正是它的代价）-> {g_l} vs {g_f}"); ok = False

    # 5) 缓存充裕时 LPM 相对 FCFS **不该**有命中率优势 —— 排序只是重分配延迟
    pr = gen_arrivals(150, 2, 60, 128, 11)
    if simulate(pr, "lpm", cache_tokens=65536).hit_rate !=        simulate(pr, "fcfs", cache_tokens=65536).hit_rate:
        print("FAIL 缓存充裕时两策略命中率应相同"); ok = False

    # 6) **tiebreak 机制**：T 不变、只换 tiebreak，公平性必须明显变差 ——
    #    这条钉住"真正在防饥饿的是 tiebreak 而不是 credit 公式"这个结论。
    g_tie, g_anti = [], []
    for sd in (11, 23, 37, 53, 71, 97):
        pr = gen_arrivals(150, 2, 50, 128, sd)
        g_tie.append(simulate(pr, _arm("aged_T32"), cache_tokens=512).fairness_gap)
        g_anti.append(simulate(pr, _arm("aged_T32_antitie"), cache_tokens=512).fairness_gap)
    if sum(g_tie) / len(g_tie) >= sum(g_anti) / len(g_anti):
        print(f"FAIL 到达序 tiebreak 应比反 FCFS 更公平 -> {g_tie} vs {g_anti}"); ok = False

    # 7) 所有请求最终都应被服务完（否则统计口径不可信）
    for p in POLICIES:
        r = simulate(proto2, p)
        if r.detail["never_started"] != 0:
            print(f"FAIL {p} 有 {r.detail['never_started']} 条请求从未开始"); ok = False
        if r.completed != r.n_req:
            print(f"FAIL {p} 完成数 {r.completed} != 请求数 {r.n_req}"); ok = False

    # 8) _probe 不能有副作用
    c = SglangRadixCache(page_size=1, capacity_tokens=9999)
    c.process(list(range(50)))
    before = (c.n_nodes, c.n_tokens)
    _probe(c, list(range(50)))
    if (c.n_nodes, c.n_tokens) != before:
        print("FAIL _probe 改动了缓存状态"); ok = False

    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
