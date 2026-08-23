"""采样 / 停止条件 / 流式输出 / HTTP 服务层的回归测试。

单独一个文件，和 `test_lab.py` 分开：那边测的是**"优化不许改变输出"**，
是纯确定性的逐位对比；这边测的东西带随机性和并发，判据不一样 ——

    **采样层的判据是「同 seed 可复现」与「批不变性」，不是「输出恒等」。**

跑法： cd _lab && python -m pytest tests -q
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sampling as S  # noqa: E402
import serve as SV  # noqa: E402


@pytest.fixture(scope="module")
def logits():
    return np.random.default_rng(0).normal(0, 2, 64).astype(np.float32)


# ────────────────────────────────────────── 一、采样算子

def test_temperature_zero_is_argmax(logits):
    """贪心不是采样的反面，是 temperature -> 0 的退化情形。"""
    assert S.Sampler(S.SamplingParams(temperature=0.0))(logits, []) \
        == int(logits.argmax())


def test_top_k_one_is_argmax_regardless_of_temperature(logits):
    for t in (0.1, 1.0, 100.0):
        assert S.Sampler(S.SamplingParams(temperature=t, top_k=1, seed=3))(
            logits, []) == int(logits.argmax())


def test_same_seed_reproduces(logits):
    a = [S.Sampler(S.SamplingParams(temperature=1.0, seed=11))(logits, [])
         for _ in range(6)]
    b = [S.Sampler(S.SamplingParams(temperature=1.0, seed=11))(logits, [])
         for _ in range(6)]
    assert a == b


def test_top_p_never_empties_the_candidate_set(logits):
    """判据必须是「它*之前*的累计 < p」。写成「含它的累计 <= p」时，
    只要 top-1 概率大于 p，候选集就空了，采样直接崩。"""
    for p in (1e-9, 1e-3, 0.1, 0.5):
        assert np.isfinite(S.top_p_filter(logits, p)).sum() >= 1


def test_filter_order_changes_the_candidate_set():
    """**先 k 后 p 与先 p 后 k 不等价。**

    top-k 砍掉尾巴后剩下的概率会被重新归一化，top-p 的累计和涨了，
    截断点就往前挪。同样的 (k, p) 在不同实现上可以给出不同候选集。
    """
    demo = np.log(np.array([0.87, 0.08, 0.05], np.float32))
    kp, pk = S.filter_order_matters(demo, k=2, p=0.9)
    assert (kp, pk) == (1, 2)


# ────────────────────────────────────────── 二、重复惩罚的符号

def test_repetition_penalty_lowers_negative_logits():
    """朴素写法 `logit / penalty` 会把负 logit 抬高 —— 越惩罚越复读，且不报错。"""
    x = np.array([-2.0, 1.0, 0.5], np.float32)
    good = S.apply_repetition_penalty(x, [0], 1.2)
    bad = S.apply_repetition_penalty_naive(x, [0], 1.2)
    assert good[0] < x[0] < bad[0]


def test_repetition_penalty_lowers_positive_logits_too():
    x = np.array([-2.0, 1.0, 0.5], np.float32)
    assert S.apply_repetition_penalty(x, [1], 1.2)[1] < x[1]


def test_penalty_one_is_identity(logits):
    assert np.array_equal(S.apply_repetition_penalty(logits, [1, 2], 1.0), logits)


# ────────────────────────────────────────── 三、批不变性

def test_per_request_rng_is_batch_invariant(logits):
    """同一条请求的输出**不许**取决于同批里还有谁。

    共用全局 rng 时这条会挂，而且不报错 —— 只是同样的 seed 在不同负载下
    给出不同结果，复现 bug 时怎么也对不上。
    """
    mine = S.Sampler(S.SamplingParams(temperature=1.0, seed=42))
    solo = [mine(logits, []) for _ in range(5)]

    mine2 = S.Sampler(S.SamplingParams(temperature=1.0, seed=42))
    noisy = S.Sampler(S.SamplingParams(temperature=1.0, seed=7))
    batched = []
    for _ in range(5):
        noisy(logits, [])
        batched.append(mine2(logits, []))
    assert solo == batched


# ────────────────────────────────────────── 四、流式输出

def test_streaming_concatenation_equals_bulk_decode():
    """**流式唯一的正确性标准**：各片拼起来 == 一次性解码。"""
    ids = [1, 4, 5, 6, 3, 9, 10]
    g = S.StreamGuard()
    out = "".join(g.push(t) for t in ids) + g.flush()
    assert out == S.toy_decode(ids)


def test_partial_utf8_is_held_back_not_mangled():
    """半个汉字（E4 BD）必须被扣住。

    `bytes.decode()` 会抛异常；`errors="replace"` 更糟 ——
    它把半个汉字变成一个**永久的** U+FFFD，后半截到了也补不回来。
    """
    g = S.StreamGuard()
    assert g.push(1) == "Hello"
    assert g.push(4) == ""              # 扣住，不发也不炸
    assert g.push(5) == "你"            # 凑齐才发
    assert chr(0xFFFD) not in g.text


def test_stop_string_straddling_token_boundary_is_caught():
    """停止串由 'ST'+'OP' 两个 token 拼出，仍必须检出。"""
    g = S.StreamGuard(stop=("STOP",))
    got = "".join(g.push(t) for t in [1, 2, 7, 8, 10])
    assert g.stopped and g.stop_hit == "STOP"
    assert got == "Hello world" and "STOP" not in got


def test_nothing_after_the_stop_string_is_emitted():
    g = S.StreamGuard(stop=("!",))
    got = "".join(g.push(t) for t in [1, 3, 2, 2]) + g.flush()
    assert got == "Hello"


def test_flush_releases_the_held_tail():
    """忘了 flush 是流式最常见的 bug：输出总是少最后一两个字。"""
    g = S.StreamGuard(stop=("STOP",))       # hold = 3
    streamed = "".join(g.push(t) for t in [1, 3])
    assert streamed != "Hello!"             # 尾巴被扣着
    assert streamed + g.flush() == "Hello!"


def test_no_stop_strings_still_streams_everything():
    ids = [1, 2, 3]
    g = S.StreamGuard()
    assert "".join(g.push(t) for t in ids) + g.flush() == S.toy_decode(ids)


# ────────────────────────────────────────── 五、停止原因

def test_finish_reason_length_vs_eos():
    p = S.SamplingParams(max_new=3, eos=0)
    g = S.StreamGuard()
    assert S.should_stop([1, 2], p, g) is None
    assert S.should_stop([1, 2, 3], p, g) == "length"
    assert S.should_stop([1, 0], p, g) == "eos"


def test_finish_reason_stop_wins_over_length():
    """三种停止同时成立时，'stop' 优先 —— 客户端要据此决定要不要续写。"""
    p = S.SamplingParams(max_new=2, eos=0)
    g = S.StreamGuard(stop=("Hello",))
    g.push(1)
    assert S.should_stop([1, 1], p, g) == "stop"


# ────────────────────────────────────────── 六、HTTP 服务层

@pytest.fixture(scope="module")
def server():
    httpd, loop = SV.serve("127.0.0.1", 0, max_queue=8, n_blocks=512,
                           block_size=8, max_running=4)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", loop
    httpd.shutdown()
    loop.shutdown()


def _post(base, body, raw=False):
    req = urllib.request.Request(base + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    r = urllib.request.urlopen(req, timeout=60).read()
    return r.decode() if raw else json.loads(r)


def test_health_and_models(server):
    base, _ = server
    assert json.loads(urllib.request.urlopen(base + "/health",
                                             timeout=10).read())["status"] == "ok"
    m = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=10).read())
    assert m["data"][0]["id"] == "minigpt-toy"


def test_completion_reports_length_finish_reason(server):
    base, _ = server
    r = _post(base, {"prompt": "Hello", "max_tokens": 6})
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["completion_tokens"] == 6


def test_stream_and_nonstream_agree(server):
    """**同一请求，流式各片拼起来必须等于非流式整段。**

    这条挂了通常不是模型的问题，是 StreamGuard 的 flush 漏了，
    或者服务端把扣住的尾巴丢了。
    """
    base, _ = server
    body = {"prompt": "Hello", "max_tokens": 6, "seed": 5}
    full = _post(base, body)["choices"][0]["text"]
    raw = _post(base, {**body, "stream": True}, raw=True)
    pieces = [json.loads(ln[6:])["choices"][0]["text"]
              for ln in raw.splitlines()
              if ln.startswith("data: ") and ln != "data: [DONE]"]
    assert "".join(pieces) == full
    assert raw.rstrip().endswith("data: [DONE]")


def test_ttft_and_tpot_are_reported_separately(server):
    """TTFT 归排队和 prefill 管，TPOT 归 decode 管。合成一个数等于没量。"""
    base, _ = server
    t = _post(base, {"prompt": "Hello world", "max_tokens": 8})["timings"]
    assert t["ttft_ms"] > 0 and t["tpot_ms"] > 0


def test_oversized_request_rejected_before_admission(server):
    """必须在准入**之前**拒。放进去再撑爆，已占的块和已算的 prefill 全白费。"""
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, {"prompt": "Hello", "max_tokens": 10 ** 6})
    assert e.value.code == 400


def test_bad_json_is_400_not_500(server):
    base, _ = server
    req = urllib.request.Request(base + "/v1/completions", data=b"{not json",
                                 headers={"content-type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400


def test_concurrent_requests_leak_no_blocks(server):
    """**并发下块必须全部归还。** 漏还不会报错，只会表现为内存只涨不跌。"""
    base, loop = server
    got: list[dict] = []
    ts = [threading.Thread(target=lambda: got.append(
        _post(base, {"prompt": "Hello world", "max_tokens": 5})))
        for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)
    deadline = time.time() + 15
    while loop.depth() > 0 and time.time() < deadline:
        time.sleep(0.05)
    assert len(got) == 8
    assert loop.engine.pool.alloc.n_used == 0


def test_metrics_endpoint_accounts_for_blocks(server):
    base, _ = server
    m = json.loads(urllib.request.urlopen(base + "/metrics", timeout=10).read())
    assert m["completed"] > 0
    assert "NOT comparable" in m["note"]      # 玩具数字不许被当成引擎数字
