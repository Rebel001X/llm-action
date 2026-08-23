"""sampling.py —— 采样、停止条件、流式输出。

到 `engine.py` 为止，引擎每一步都取 `argmax`，这叫**贪心解码**。
贪心是确定性的，所以前面「优化不许改变输出」那一整套逐位对比才能成立。
但一个真能用的服务必须再补三件事，而这三件**全都不是数学问题，是工程问题**：

  1. **采样**：temperature / top-k / top-p / 重复惩罚。
     这里的坑不在公式，在**算子的施加顺序**和**负 logit 的符号**。
  2. **停止条件**：max_new / EOS / 停止字符串。
     前两个是整数比较；第三个难，因为**停止字符串会跨 token 边界**。
  3. **流式输出**：不能一拿到 token 就发。
     两个原因逼你缓一手 —— **UTF-8 多字节字符会跨 token 边界**，
     以及**已发出去的字符收不回来**，所以可能构成停止串前缀的尾巴必须先扣住。

**本文件的立场**：贪心不是"采样的对立面"，是 `temperature → 0` 的退化情形；
流式不是"把结果切碎发出去"，是**一个需要显式状态机的增量协议**。

跑法：
    python sampling.py --selftest     # 13 项断言
    python sampling.py                # 采样配置对比 + 流式演示

⚠️ 本文件不产任何性能数字。它讲的是正确性。
"""
from __future__ import annotations

import codecs
from dataclasses import dataclass, field

import numpy as np

import minigpt as M


# ═════════════════════════════════════════════════ 一、采样参数

@dataclass
class SamplingParams:
    """一次请求的采样配置。**默认值刻意等价于贪心**，方便和前面的测试对齐。"""
    temperature: float = 0.0         # 0 = 贪心（argmax）
    top_k: int = 0                   # 0 = 关闭
    top_p: float = 1.0               # 1.0 = 关闭
    repetition_penalty: float = 1.0  # 1.0 = 关闭
    seed: int | None = None
    max_new: int = 32
    eos: int | None = None
    stop: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0 or self.top_k == 1


# ═════════════════════════════════════════════════ 二、四个算子

def apply_repetition_penalty(logits: np.ndarray, prev: list[int],
                             penalty: float) -> np.ndarray:
    """重复惩罚（CTRL 论文的形式）。

    **这个函数是本文件最容易写错的一个，错因是符号。**
    直觉写法是「出现过的 token 一律 `logit / penalty`」。
    但 logit 可以是负的：`-2.0 / 1.2 = -1.67`，**比原来更大**，
    于是"惩罚"把重复 token 的概率**推高**了 —— 越惩罚越复读。

    正确做法必须分支：正的除、负的乘。
    """
    if penalty == 1.0 or not prev:
        return logits
    out = logits.copy()
    idx = np.unique(np.asarray(prev, dtype=np.int64))
    v = out[idx]
    out[idx] = np.where(v > 0, v / penalty, v * penalty)
    return out


def apply_repetition_penalty_naive(logits: np.ndarray, prev: list[int],
                                   penalty: float) -> np.ndarray:
    """**故意写错的版本**，只留给自检当反例，不要在别处调用。

    它对负 logit 的处理是反的，`selftest()` 里有一条断言专门钉这个差异。
    """
    if penalty == 1.0 or not prev:
        return logits
    out = logits.copy()
    idx = np.unique(np.asarray(prev, dtype=np.int64))
    out[idx] = out[idx] / penalty
    return out


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    """`logits / t`。

    t → 0 时最大值被无限放大 ⇒ 退化成 argmax；t → ∞ 时所有 logit 趋同 ⇒ 均匀分布。
    **所以贪心不是采样的反面，是它的一个端点。**
    """
    if t <= 0:
        raise ValueError("temperature<=0 请走贪心分支，不要在这里除零")
    return logits / t


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """只留最大的 k 个，其余置 -inf。k<=0 或 k>=vocab 时是空操作。"""
    n = logits.shape[-1]
    if k <= 0 or k >= n:
        return logits
    thresh = np.partition(logits, n - k)[n - k]
    return np.where(logits < thresh, -np.inf, logits)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """核采样：按概率降序累加，保留**累计概率首次达到 p 的那个最小集合**。

    两个必须注意的点：
      * **永远保留至少一个 token**。判据写成「它*之前*的累计 < p」而不是
        「含它的累计 <= p」，第一个 token 的"之前累计"恒为 0，所以必然留下。
        写成后者时，只要 top-1 概率本身就大于 p，集合就空了，采样直接崩。
      * 这个函数吃的是 **logits**，softmax 在函数内部做。
        它看到的是"当前这一步的分布"，所以**前面已经被 top-k 置成 -inf 的位置
        不参与归一化** —— 这正是下面 `filter_order_matters` 要演示的事。
    """
    if p >= 1.0:
        return logits
    order = np.argsort(logits)[::-1]
    probs = M.softmax(logits[order][None, :].astype(np.float32))[0]
    cum_before = np.cumsum(probs) - probs
    keep_sorted = cum_before < p
    keep_sorted[0] = True                      # 兜底：至少一个
    keep = np.zeros_like(logits, dtype=bool)
    keep[order[keep_sorted]] = True
    return np.where(keep, logits, -np.inf)


def filter_order_matters(logits: np.ndarray, k: int, p: float) -> tuple[int, int]:
    """返回「先 k 后 p」与「先 p 后 k」两种顺序各自留下几个 token。

    两者**不一定相等**：top-k 先砍掉尾巴之后，剩下的概率会被重新归一化，
    于是 top-p 的累计和涨了，截断点就往前挪。
    `selftest()` 用一个具体分布把这个差异钉住。

    这不是学术细节 —— 不同实现的默认顺序不一样，
    **同样的 (k, p) 在两个引擎上可以给出不同的候选集**。
    """
    kp = np.isfinite(top_p_filter(top_k_filter(logits, k), p)).sum()
    pk = np.isfinite(top_k_filter(top_p_filter(logits, p), k)).sum()
    return int(kp), int(pk)


# ═════════════════════════════════════════════════ 三、采样器

class Sampler:
    """一条请求的采样器。

    **每条请求持有自己的随机数发生器**，这不是洁癖：
    如果整个引擎共用一个全局 rng，那么某条请求这一步抽到什么，
    就取决于**同一批里还有谁**、以及它们的顺序 ——
    于是同样的 prompt + 同样的 seed，在不同负载下给出不同结果，
    而且**不会报错**。`selftest()` 里有一条断言演示这个失败模式。
    """

    def __init__(self, params: SamplingParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)

    def __call__(self, logits: np.ndarray, prev: list[int]) -> int:
        p = self.p
        logits = apply_repetition_penalty(logits, prev, p.repetition_penalty)
        if p.is_greedy:
            return int(np.argmax(logits))
        logits = apply_temperature(logits, p.temperature)
        logits = top_k_filter(logits, p.top_k)
        logits = top_p_filter(logits, p.top_p)
        probs = M.softmax(logits[None, :].astype(np.float32))[0]
        probs = probs / probs.sum()            # rng.choice 对归一化很敏感
        return int(self.rng.choice(len(probs), p=probs))


# ═════════════════════════════════════════════════ 四、玩具字节级词表

# 真实 BPE 词表的 token 是**字节串**，不是字符串 —— 一个 token 可以是半个汉字。
# 这里用 12 个 token 的玩具词表把两个真实难点复现出来：
#   * id 4 + id 5 合起来才是 "你"（E4 BD | A0），单独 decode 任一个都会抛异常；
#   * id 7 + id 8 合起来才是 "STOP"，停止串跨了 token 边界。
TOY_VOCAB: list[bytes] = [
    b"",                # 0  EOS
    b"Hello",           # 1
    b" world",          # 2
    b"!",               # 3
    b"\xe4\xbd",        # 4  "你" 的前 2 字节
    b"\xa0",            # 5  "你" 的第 3 字节
    b"\xe5\xa5\xbd",    # 6  "好"
    b"ST",              # 7
    b"OP",              # 8  与 7 拼成 "STOP"
    b" and",            # 9
    b" more",           # 10
    b"\n",              # 11
]


def toy_decode(ids: list[int]) -> str:
    """一次性解码（非流式）—— 作为流式输出的正确性参照。"""
    return b"".join(TOY_VOCAB[i] for i in ids).decode("utf-8")


# ═════════════════════════════════════════════════ 五、流式输出的状态机

class StreamGuard:
    """增量解码 + 停止串检测。**流式输出的全部难点都在这个类里。**

    它扣住两种还不能发的东西：

      1. **不完整的 UTF-8 字节序列。** 交给 `codecs` 的增量解码器：
         喂进去半个字符它就先攒着，凑齐了才吐出来。
         自己写 `bytes.decode()` 的话，半个汉字会直接抛 `UnicodeDecodeError`；
         而 `errors="replace"` 更糟 —— **它会把半个汉字变成一个永久的 U+FFFD**，
         等后半截到了也补不回来了。
      2. **可能是停止串前缀的尾巴。** 停止串 "STOP" 可能由 "ST"+"OP" 两个 token 拼出来。
         已经发给用户的字符收不回来，所以必须扣住最后 `max(len(stop))-1` 个字符，
         确认它凑不出停止串了再发。

    代价很直白：**流式输出天然带一点延迟**，延迟长度等于最长停止串减一。
    这是协议要求，不是实现不好。
    """

    def __init__(self, stop: tuple[str, ...] = ()):
        self.stop = tuple(s for s in stop if s)
        self.hold = max((len(s) for s in self.stop), default=1) - 1
        self.dec = codecs.getincrementaldecoder("utf-8")()
        self.text = ""       # 已解码出的全部文本（含扣住没发的尾巴）
        self.emitted = 0     # 已经发给用户的字符数
        self.stopped = False
        self.stop_hit: str | None = None

    def push(self, token_id: int) -> str:
        """喂一个 token，返回**这一刻可以安全发出去**的文本（可能是空串）。"""
        if self.stopped:
            return ""
        self.text += self.dec.decode(TOY_VOCAB[token_id])
        for s in self.stop:
            i = self.text.find(s)
            if i >= 0:
                out = self.text[self.emitted:i]
                self.emitted = max(i, self.emitted)
                self.stopped, self.stop_hit = True, s
                return out
        safe = len(self.text) - self.hold
        if safe <= self.emitted:
            return ""
        out = self.text[self.emitted:safe]
        self.emitted = safe
        return out

    def flush(self) -> str:
        """生成正常结束（EOS / max_new）时调用，把扣住的尾巴放出来。

        **忘了调用它是流式实现最常见的 bug**：输出总是少最后一两个字，
        而且短输出比长输出更容易暴露，因为扣住的比例更大。
        """
        if self.stopped:
            return ""
        out = self.text[self.emitted:]
        self.emitted = len(self.text)
        return out


# ═════════════════════════════════════════════════ 六、停止条件

def should_stop(out: list[int], params: SamplingParams,
                guard: StreamGuard) -> str | None:
    """返回停止原因（`"stop"` / `"eos"` / `"length"`），没停就返回 None。

    **停止原因必须回给调用方**，不能只回一句"完了"：
    客户端要靠它区分"模型说完了"和"被截断了"，这两种情况的重试策略完全不同。
    OpenAI 协议里的 `finish_reason` 就是这个东西。
    """
    if guard.stopped:
        return "stop"
    if params.eos is not None and out and out[-1] == params.eos:
        return "eos"
    if len(out) >= params.max_new:
        return "length"
    return None


# ═════════════════════════════════════════════════ 七、自检

def selftest() -> int:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, 64).astype(np.float32)

    # ---- 采样算子 ----
    chk(Sampler(SamplingParams(temperature=0.0, seed=1))(logits, [])
        == int(logits.argmax()),
        "temperature=0 等价于 argmax（贪心是采样的退化情形）")

    chk(Sampler(SamplingParams(temperature=5.0, top_k=1, seed=1))(logits, [])
        == int(logits.argmax()),
        "top_k=1 无论温度多高都等价于贪心")

    a = [Sampler(SamplingParams(temperature=1.0, seed=7))(logits, []) for _ in range(5)]
    b = [Sampler(SamplingParams(temperature=1.0, seed=7))(logits, []) for _ in range(5)]
    chk(a == b, "同 seed 同 logits => 同结果（采样必须可复现）")

    chk(int(np.isfinite(top_p_filter(logits, 1e-9)).sum()) == 1,
        "top_p 极小时仍保留恰好 1 个候选（永远不许把集合筛空）")

    # ---- 施加顺序 ----
    demo = np.log(np.array([0.87, 0.08, 0.05], np.float32))
    kp, pk = filter_order_matters(demo, k=2, p=0.9)
    chk(kp != pk,
        f"先 k 后 p 留 {kp} 个、先 p 后 k 留 {pk} 个 —— **顺序会改变候选集**")

    # ---- 重复惩罚的符号 ----
    neg = np.array([-2.0, 1.0, 0.5], np.float32)
    good = apply_repetition_penalty(neg, [0], 1.2)
    bad = apply_repetition_penalty_naive(neg, [0], 1.2)
    chk(good[0] < neg[0], "正确实现：负 logit 被惩罚后**更小**")
    chk(bad[0] > neg[0],
        f"朴素实现把 {neg[0]:.1f} 变成 {bad[0]:.3f} —— **越惩罚越复读**，且不报错")

    # ---- 每请求 rng（批不变性）----
    s1 = Sampler(SamplingParams(temperature=1.0, seed=42))
    s2 = Sampler(SamplingParams(temperature=1.0, seed=42))
    other = Sampler(SamplingParams(temperature=1.0, seed=999))
    solo = [s1(logits, []) for _ in range(4)]
    batched = []
    for _ in range(4):
        other(logits, [])          # 同批里的另一条请求先抽一次
        batched.append(s2(logits, []))
    chk(solo == batched,
        "每请求独立 rng => 同批里有别人也不影响自己（批不变性）")

    g_alone = np.random.default_rng(42)
    v1 = [int(g_alone.integers(0, 64)) for _ in range(3)]
    g_batch = np.random.default_rng(42)
    v2 = []
    for _ in range(3):
        g_batch.integers(0, 64)    # 冒充同批里的另一条请求
        v2.append(int(g_batch.integers(0, 64)))
    chk(v1 != v2,
        "反例：共用一个全局 rng 时，同批里多一条请求就会改变自己的输出")

    # ---- 流式：UTF-8 跨 token ----
    g = StreamGuard()
    parts = [g.push(t) for t in [1, 4, 5, 6, 3]]
    parts.append(g.flush())
    chk(parts[1] == "", "半个汉字（E4 BD）被扣住，不会发出去也不会抛异常")
    chk("".join(parts) == toy_decode([1, 4, 5, 6, 3]),
        "流式各片拼起来 == 一次性解码（这是流式唯一的正确性标准）")
    chk(all(chr(0xFFFD) not in x for x in parts),
        "全程不出现 U+FFFD（errors='replace' 会在这里制造永久乱码）")

    # ---- 流式：停止串跨 token ----
    g2 = StreamGuard(stop=("STOP",))
    got = "".join(g2.push(t) for t in [1, 2, 7, 8, 10])
    chk(g2.stopped and g2.stop_hit == "STOP",
        "停止串由 'ST'+'OP' 跨 token 拼出，仍被检出")
    chk("STOP" not in got and got == "Hello world",
        f"停止串本身及其之后一个字都没发出去（实得 {got!r}）")

    print("  全部通过" if ok else "  有失败项")
    return 0 if ok else 1


# ═════════════════════════════════════════════════ 八、演示

def _demo() -> None:
    cfg = M.Config(n_layer=2, n_head=2, d_model=32, vocab=64, max_seq=128, seed=1)
    w = M.init_weights(cfg)
    prompt = [3, 14, 15, 62]

    def gen(params: SamplingParams) -> list[int]:
        s = Sampler(params)
        out: list[int] = []
        cache = M.KVCache(cfg)
        logits = M.forward_cached(cfg, w, list(prompt), cache)
        for _ in range(params.max_new):
            nxt = s(logits, out)
            out.append(nxt)
            logits = M.forward_cached(cfg, w, [nxt], cache)
        return out

    print("采样配置对比（同一个 prompt，各生成 16 个 token）")
    print(f"  {'配置':<30}{'不同 token 数':>14}{'最长连续重复':>14}")
    table = [
        ("贪心 (temperature=0)", SamplingParams(max_new=16)),
        ("t=0.7", SamplingParams(temperature=0.7, seed=0, max_new=16)),
        ("t=1.0", SamplingParams(temperature=1.0, seed=0, max_new=16)),
        ("t=1.0 top_k=5", SamplingParams(temperature=1.0, top_k=5, seed=0, max_new=16)),
        ("t=1.0 top_p=0.9", SamplingParams(temperature=1.0, top_p=0.9, seed=0,
                                           max_new=16)),
        ("贪心 + 重复惩罚 1.3", SamplingParams(repetition_penalty=1.3, max_new=16)),
    ]
    for name, pr in table:
        o = gen(pr)
        run = best = 1
        for i in range(1, len(o)):
            run = run + 1 if o[i] == o[i - 1] else 1
            best = max(best, run)
        print(f"  {name:<30}{len(set(o)):>14}{best:>14}")
    print("\n  ! 这是 2 层 32 维**随机权重**的玩具模型，只用来演示算子的作用方向，")
    print("     **不代表任何真实模型的采样质量**。")

    print("\n流式输出演示（停止串 'STOP'，词表见 TOY_VOCAB）")
    g = StreamGuard(stop=("STOP",))
    for t in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        piece = g.push(t)
        print(f"  token {t:>2}  {TOY_VOCAB[t]!r:<18} -> 发出 {piece!r}"
              + ("   <- 命中停止串，之后全部丢弃" if g.stopped else ""))
        if g.stopped:
            break
    print(f"  finish_reason = {'stop' if g.stopped else 'length'}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    _demo()
