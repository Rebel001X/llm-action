"""structured.py —— 结构化输出：把语法变成一个**每步作用在 logits 上的掩码**。

需求很实在：让模型输出**保证**能被 `json.loads()` 解析，而不是"通常可以"。
做法只有一句话：

    **每一步，把当前语法状态下不合法的 token 全部置成 -inf，然后照常采样。**

模型没有被改，改的只是候选集。所以结果**一定**合法 ——
不是"提示词写得好所以合法"，是数学上不可能不合法。

但这件事真正难的地方不在语法，在 **token 边界与语法边界对不齐**：
词表里的一个 token 可能横跨"字符串收尾"和"冒号"。
所以不能按 token 写规则，必须**在字符层面写规则，再把每个 token 拆开走一遍**，
为每个语法状态预先算出「哪些 token 整体可走」。
这张表就是 outlines / xgrammar 这类库在做的事，本文件叫 `build_index()`。

还有一个白赚的优化：**当语法只剩一条路可走时，根本不用问模型**（jump-forward）。
比如刚吐完键名的收尾引号，接下来必然是 `:`，直接吐出去即可。

**和投机解码的关键区别**（这两件事很容易被混为一谈）：

    投机解码**不改变**输出分布，是精确的重写；
    结构化输出**故意改变**分布 —— 它就是要把不合法的那部分概率剪掉再归一化。
    **前者是等价变换，后者是有意的干预。别把两者的正确性标准混用。**

跑法：
    python structured.py --selftest    # 12 项断言
    python structured.py               # 状态表 + 约束生成 + jump-forward 省了多少次调用
"""
from __future__ import annotations

import json
import string
import zlib

import numpy as np

# ── 一个极小的字节级词表（故意让两个 token 横跨语法边界）──────────────────
# `":` 与 `,"` 是本文件的主角：它们各自跨了两个语法元素，
# **正是它们让"按 token 写规则"这条路走不通。**
VOCAB: list[str] = [
    "",        # 0  EOS
    "{",       # 1
    "}",       # 2
    '"',       # 3
    ":",       # 4
    ",",       # 5
    '":',      # 6  <- 横跨：字符串收尾 + 冒号
    ',"',      # 7  <- 横跨：逗号 + 下一个键的开头引号
    "name",    # 8
    "age",     # 9
    "Ada",     # 10
    "42",      # 11
    "x",       # 12
]

LETTERS = set(string.ascii_letters)
DIGITS = set(string.digits)
ALPHABET = sorted({c for t in VOCAB for c in t})

# 文法：{ "键" : 值 (, "键" : 值)* }   键是字母串，值是字母串或数字
# **三个上界让语言变成有限的** —— 没有它们，随机采样可能永远不收口。
# 真实场景里这些上界通常来自 JSON Schema（字段固定、类型固定），是同一回事。
MAX_WORD, MAX_DIGIT, MAX_PAIR = 5, 3, 2

# 状态是一个三元组 (种类, 计数, 已完成的键值对数)
Start = ("start", 0, 0)


# ═════════════════════════════════════════════════ 一、字符级文法

def step_char(st: tuple, ch: str) -> tuple | None:
    """吃一个**字符**，返回新状态；不合法返回 None。

    规则写在字符层面，是因为**只有字符层面的规则才是文法本身**；
    token 层面的合法性是由它推出来的结果，不是可以独立定义的东西。
    """
    kind, n, p = st
    if kind == "start":
        return ("kq", 0, 0) if ch == "{" else None
    if kind == "kq":                              # 等键名的开头引号
        return ("kb", 0, p) if ch == '"' else None
    if kind == "kb":                              # 键名内部
        if ch in LETTERS and n < MAX_WORD:
            return ("kb", n + 1, p)
        if ch == '"' and n >= 1:
            return ("colon", 0, p)
        return None
    if kind == "colon":
        return ("val", 0, p) if ch == ":" else None
    if kind == "val":
        if ch == '"':
            return ("vs", 0, p)
        if ch in DIGITS:
            return ("vn", 1, p)
        return None
    if kind == "vs":                              # 字符串值内部
        if ch in LETTERS and n < MAX_WORD:
            return ("vs", n + 1, p)
        if ch == '"' and n >= 1:
            return ("av", 0, p + 1)
        return None
    if kind == "vn":                              # 数字值内部
        if ch in DIGITS and n < MAX_DIGIT:
            return ("vn", n + 1, p)
        if ch == "," and p + 1 < MAX_PAIR:
            return ("kq", 0, p + 1)
        if ch == "}":
            return ("done", 0, p + 1)
        return None
    if kind == "av":                              # 一个键值对刚结束
        if ch == "," and p < MAX_PAIR:
            return ("kq", 0, p)
        if ch == "}":
            return ("done", 0, p)
        return None
    return None                                   # done 之后不再接受任何字符


def step_token(st: tuple, tok: str) -> tuple | None:
    """吃一个 **token**：拆成字符逐个走。中途任何一步不合法，整个 token 就不合法。

    **这就是 token 边界问题的全部解法。** 一个 token 合不合法**取决于当前状态**，
    不是它自身的属性 —— `selftest()` 用 `":` 在两个不同状态下的结果把这条钉住。
    """
    if tok == "":
        return st if st[0] == "done" else None    # EOS 只在终态合法
    cur: tuple | None = st
    for ch in tok:
        cur = step_char(cur, ch)
        if cur is None:
            return None
    return cur


# ═════════════════════════════════════════════════ 二、索引：状态 -> 合法 token 掩码

def all_states() -> list[tuple]:
    """从起始状态 BFS 枚举所有可达状态。**真实语法编译器做的就是这件事。**"""
    seen, frontier, order = {Start}, [Start], [Start]
    while frontier:
        nxt = []
        for st in frontier:
            for ch in ALPHABET:
                to = step_char(st, ch)
                if to is not None and to not in seen:
                    seen.add(to)
                    order.append(to)
                    nxt.append(to)
        frontier = nxt
    return order


def build_index(vocab: list[str] = VOCAB) -> dict[tuple, np.ndarray]:
    """为每个状态预算一张 bool 掩码。**成本 O(状态数 × 词表大小 × token 长度)，只做一次。**

    真实系统里词表几万到几十万、状态可能上千，这张表因此不小 ——
    **它是结构化输出的主要内存与预处理开销**，也是"换个语法要重新编译一次"的原因。
    """
    return {st: np.array([step_token(st, t) is not None for t in vocab])
            for st in all_states()}


def apply_mask(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """把不合法的位置置 -inf。**必须在采样之前、作用在 logits 上。**

    "先采样再检查、不合法就重试"是另一回事：它慢、没有终止保证，
    而且**改变分布的方式不可控**。掩码是唯一能给出"一定合法"的做法。
    """
    if not allowed.any():
        # **必须炸出来。** 全 -inf 会让 softmax 变成 nan，再被 argmax 取成 0 ——
        # 于是模型开始吐垃圾，而全程没有任何报错。
        raise ValueError("当前状态没有任何合法 token —— 语法或索引有 bug")
    return np.where(allowed, logits, -np.inf)


def forced_token(allowed: np.ndarray) -> int | None:
    """只剩一个合法 token 时返回它 —— **这一步不用问模型**（jump-forward）。

    代价要说清楚：这样吐出去的切分**未必是模型自己会选的切分**，
    于是"模型看到的历史"与"真实生成过程"出现口径差（token healing 问题）。
    多数场景无所谓，但要拿 logprob 做打分时，这里会咬人。
    """
    ids = np.flatnonzero(allowed)
    return int(ids[0]) if len(ids) == 1 else None


# ═════════════════════════════════════════════════ 三、带约束的生成

def generate(logits_fn, seed: int = 0, max_tokens: int = 40,
             jump_forward: bool = True, vocab: list[str] = VOCAB,
             index: dict | None = None) -> dict:
    """按语法约束生成。`logits_fn(state, text)` 返回一整排 logits（模拟模型）。

    返回 `{"text", "tokens", "model_calls", "jumped", "state"}`。
    **`model_calls` 是最有意思的那个输出** —— 它量化了 jump-forward 省下的调用数。

    ⚠️ **随机性按「(seed, 已生成文本)」派生，而不是一个从头走到尾的 Generator。**
    这不是洁癖，是被 jump-forward 逼出来的：被跳过的那些步**不消耗随机数**，
    共用一个有状态的 rng 时，开关 jump-forward 会让两条轨迹从第一次跳过之后
    彻底发散 —— 于是根本没法验证"跳过的都是没得选的步"。
    **我第一版就是这么写的，自检直接打脸：产出文本不同、调用次数反而更多。**
    派生式随机让"同 seed ⇒ 同轨迹"成立，两者才可比。
    """
    idx = index or build_index(vocab)
    st, text, toks = Start, "", []
    calls = jumped = 0
    for _ in range(max_tokens):
        if st[0] == "done":
            break
        allowed = idx[st]
        if jump_forward:
            f = forced_token(allowed)
            if f is not None:
                jumped += 1
                toks.append(f)
                text += vocab[f]
                st = step_token(st, vocab[f])
                continue
        calls += 1
        lg = apply_mask(np.asarray(logits_fn(st, text), np.float64), allowed)
        e = np.exp(lg - lg.max())
        # **必须用确定性哈希。** Python 的内置 `hash()` 对字符串是逐进程加盐的，
        # 用它会让"同 seed ⇒ 同轨迹"只在单次进程内成立，换个进程就不一样。
        srng = np.random.default_rng([seed, zlib.crc32(text.encode()), len(toks)])
        t = int(srng.choice(len(lg), p=e / e.sum()))
        toks.append(t)
        text += vocab[t]
        st = step_token(st, vocab[t])
    return {"text": text, "tokens": toks, "model_calls": calls,
            "jumped": jumped, "state": st}


# ═════════════════════════════════════════════════ 四、自检

def selftest() -> int:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        ok &= bool(cond)

    idx = build_index()
    states = all_states()

    chk(len(states) > 10 and Start in idx,
        f"BFS 枚举出 {len(states)} 个可达状态（有界文法 ⇒ 状态有限）")
    chk(idx[Start].sum() == 1 and VOCAB[int(np.flatnonzero(idx[Start])[0])] == "{",
        "起始状态只有 '{' 合法 —— **第一步就不用问模型**")

    # ---- 横跨语法边界的 token：合法性是状态相关的 ----
    chk(step_token(("kb", 1, 0), '":') == ("val", 0, 0),
        "token '\":' 在键名内部合法，逐字符走后落在 val")
    chk(step_token(("kq", 0, 0), '":') is None,
        "**同一个 token 换个状态就不合法** —— 合法性是状态的属性，不是 token 的属性")
    chk(step_token(("av", 0, 1), ',"') == ("kb", 0, 1),
        "token ',\"' 横跨「逗号 + 下一个键的开头引号」，同样能走通")

    # ---- 约束生成一定合法 ----
    def rand_logits(st, tx):
        h = zlib.crc32(f"{st}|{tx}".encode())
        return np.random.default_rng(h).normal(0, 3, len(VOCAB))

    outs = [generate(rand_logits, seed=s, index=idx) for s in range(500)]
    reached = sum(1 for o in outs if o["state"][0] == "done")
    valid = 0
    for o in outs:
        try:
            json.loads(o["text"])
            valid += 1
        except json.JSONDecodeError:
            pass
    chk(reached == 500 and valid == 500,
        f"500 次随机采样：{reached} 次走到终态、{valid} 次能被 json.loads 解析"
        f" —— **100%，这是保证不是概率**")

    # ---- 不加约束的对照 ----
    bad = 0
    for s in range(500):
        r = np.random.default_rng(s)
        txt = "".join(VOCAB[int(r.integers(1, len(VOCAB)))] for _ in range(6))
        try:
            json.loads(txt)
        except json.JSONDecodeError:
            bad += 1
    chk(bad > 490,
        f"同一个词表不加约束随机吐 6 个 token：{bad}/500 无法解析")

    # ---- jump-forward ----
    a = generate(rand_logits, seed=3, jump_forward=True, index=idx)
    b = generate(rand_logits, seed=3, jump_forward=False, index=idx)
    chk(a["jumped"] > 0 and a["model_calls"] < b["model_calls"]
        and a["model_calls"] + a["jumped"] == b["model_calls"],
        f"jump-forward 把模型调用从 {b['model_calls']} 降到 {a['model_calls']}"
        f"（正好跳过 {a['jumped']} 步，加起来对得上）")
    chk(a["state"][0] == "done" and b["state"][0] == "done",
        "开/关 jump-forward 都走到终态")
    chk(a["text"] == b["text"],
        f"**同一个 rng 下开关 jump-forward 产出同一段文本**（{a['text']}）"
        f" —— 被跳过的那些步本来就只有一个选择")

    # ---- 掩码的语义 ----
    probs_state = ("val", 0, 0)
    lg = np.zeros(len(VOCAB))
    masked = apply_mask(lg, idx[probs_state])
    e = np.exp(masked - masked.max())
    probs = e / e.sum()
    chk(abs(probs[idx[probs_state]].sum() - 1.0) < 1e-12
        and probs[~idx[probs_state]].sum() == 0.0,
        "掩码后概率**全部**落在合法集合上并重新归一化"
        " —— **这是有意改变分布，与投机解码的等价变换不是一回事**")

    try:
        apply_mask(lg, np.zeros(len(VOCAB), bool))
        chk(False, "空合法集合应当抛异常")
    except ValueError:
        chk(True, "**没有任何合法 token 时明确报错**（静默的话 softmax 直接变 nan）")

    print("  全部通过" if ok else "  有失败项")
    return 0 if ok else 1


# ═════════════════════════════════════════════════ 五、演示

def _demo() -> None:
    idx = build_index()
    states = all_states()
    print(f"一、BFS 枚举出 {len(states)} 个可达状态；每个状态下有多少 token 合法"
          f"（词表 {len(VOCAB)} 个）")
    print(f"  {'状态':<20}{'合法数':>8}  合法 token")
    forced = 0
    for st in states:
        toks = [repr(VOCAB[i]) for i in np.flatnonzero(idx[st])]
        if len(toks) == 1:
            forced += 1
        if st[0] in ("start", "kq", "colon", "val", "av", "done") or st[1] <= 1:
            print(f"  {str(st):<20}{len(toks):>8}  " + " ".join(toks[:8]))
    print(f"  其中 {forced}/{len(states)} 个状态只有唯一选择 —— "
          f"**这些就是 jump-forward 能白赚的地方。**")

    def rand_logits(st, tx):
        h = zlib.crc32(f"{st}|{tx}".encode())
        return np.random.default_rng(h).normal(0, 3, len(VOCAB))

    print("\n二、约束生成 8 条（模型给的是纯噪声 logits）")
    for s in range(8):
        o = generate(rand_logits, seed=s, index=idx)
        try:
            json.loads(o["text"])
            v = "可解析"
        except json.JSONDecodeError:
            v = "**不可解析**"
        print(f"  {o['text']:<30} 模型调用 {o['model_calls']:>2} 次，"
              f"跳过 {o['jumped']:>2} 次   {v}")
    print("  模型输出是纯噪声，结果照样全部合法 —— **合法性来自掩码，不来自模型。**")

    print("\n三、jump-forward 省下的调用（200 条平均）")
    on = [generate(rand_logits, seed=s, jump_forward=True, index=idx)
          for s in range(200)]
    off = [generate(rand_logits, seed=s, jump_forward=False, index=idx)
           for s in range(200)]
    mo = float(np.mean([o["model_calls"] for o in on]))
    mf = float(np.mean([o["model_calls"] for o in off]))
    same = sum(1 for x, y in zip(on, off) if x["text"] == y["text"])
    print(f"  开：平均 {mo:.2f} 次   关：平均 {mf:.2f} 次   省 {1 - mo / mf:.0%}")
    print(f"  两者产出完全相同的条数：{same}/200 —— **跳过的都是本来就没得选的步**")
    print("  代价：这样吐出的 token 切分未必是模型自己会选的，")
    print("        要拿 logprob 做打分时这个口径差会咬人（token healing）。")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    _demo()
