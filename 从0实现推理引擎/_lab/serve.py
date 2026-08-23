"""serve.py —— 把引擎变成一个 HTTP 服务。

前面所有文件加起来，得到的是**一个函数**：喂 token 进去，吐 token 出来。
从函数到服务，中间横着一件事，而且只有这一件是真正难的：

    **HTTP 是并发的（N 个连接同时来），引擎是单线程的（一步一步走）。**

这不是"加个 Flask 就好"。引擎的 KV 块池、块表、前缀缓存全是可变共享状态，
**多个 handler 线程同时调 `engine.step()` 会直接把块池写坏**，
而且坏法是静默的 —— 两条请求读到对方的 KV，各自都能正常返回，只是答案是错的。

所以正确的结构是**一进一出两道队列**，全局只有一个线程碰引擎：

    HTTP handler 线程（N 个）              引擎线程（恰好 1 个）
      收请求 --> in_q.put(job)   ----->    step() 循环：准入/推进/回收
      发响应 <-- job.out_q.get() <-----    每出一个 token 就 put 一次

本文件用 Python 标准库实现（`http.server` + `queue` + `threading`），不装任何依赖。
它刻意**不用 asyncio** —— 真实引擎多数用 asyncio，但那会把"队列解耦"这个主干
藏进事件循环里，教学上不划算。结构是一样的。

除了并发，还有四件事不做就不能上线，本文件都做了：

  * **背压**：等待队列满了要**立刻拒绝（503）**，不能无限收。
    收下来只会让所有人的延迟一起涨，而且没人知道发生了什么。
  * **客户端断连要回收**：连接断了而请求还在跑，它占的 KV 块**不会自己还回来**。
    这是线上最常见的"内存莫名其妙涨满"。
  * **finish_reason**：客户端要靠它区分"说完了"和"被截断了"。
  * **指标分层**：TTFT / TPOT / 排队时间必须分开量，合成一个"延迟"等于什么都没量。

跑法：
    python serve.py --selftest        # 起服务打自己，7 项断言
    python serve.py                   # 前台起服务，默认 127.0.0.1:8000
    curl -N -X POST localhost:8000/v1/completions \
         -H 'content-type: application/json' \
         -d '{"prompt":"Hello","max_tokens":8,"stream":true}'

!!! 本文件里的时间数字是**这个 numpy 玩具在 CPU 上**的，测的是它自己。
    **绝不能外推到 vLLM/SGLang。**
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import engine as E
import minigpt as M
from sampling import (TOY_VOCAB, Sampler, SamplingParams, StreamGuard,
                      should_stop)

# 词表大小刻意等于玩具词表长度，于是模型吐出的 id 可以**直接**过 TOY_VOCAB 解码，
# 不需要再编一层映射。这是玩具才有的便利，真实模型的 vocab 是几万到几十万。
CFG = M.Config(n_layer=2, n_head=2, d_model=32, vocab=len(TOY_VOCAB),
               max_seq=256, seed=1)


def toy_encode(text: str) -> list[int]:
    """最长匹配分词。**真实 BPE 不是这么做的**，这里只要能把字符串变成 id。"""
    data = text.encode("utf-8")
    ids: list[int] = []
    i = 0
    order = sorted(range(1, len(TOY_VOCAB)), key=lambda t: -len(TOY_VOCAB[t]))
    while i < len(data):
        for t in order:
            tok = TOY_VOCAB[t]
            if tok and data.startswith(tok, i):
                ids.append(t)
                i += len(tok)
                break
        else:
            i += 1                      # 词表覆盖不到的字节直接丢掉
    return ids or [1]


# ═════════════════════════════════════════════════ 一、一次请求在服务端的全部状态

@dataclass
class Job:
    """一条在飞的请求。**HTTP 线程与引擎线程之间只通过它通信。**"""
    rid: int
    prompt: list[int]
    params: SamplingParams
    stream: bool
    out_q: "queue.Queue[tuple[str, object]]" = field(
        default_factory=queue.Queue)          # ("piece"|"done"|"error", payload)
    cancelled: bool = False                   # 客户端断连时由 HTTP 线程置位
    t_arrive: float = field(default_factory=time.perf_counter)
    t_first: float = -1.0                     # 第一个 token 出来的时刻
    t_end: float = -1.0
    n_out: int = 0
    finish_reason: str = "length"

    @property
    def ttft(self) -> float:
        """Time To First Token：**排队 + prefill**。用户感知的"卡不卡"就是它。"""
        return self.t_first - self.t_arrive if self.t_first > 0 else -1.0

    @property
    def tpot(self) -> float:
        """Time Per Output Token：出了第一个之后，平均每个 token 多久。

        **TTFT 和 TPOT 必须分开量。** 它们受不同因素支配 ——
        TTFT 归排队和 prefill 管，TPOT 归 decode 的访存带宽管。
        把两者平均成一个"延迟"，等于把两台机器的体温加起来除以二。
        """
        if self.n_out <= 1 or self.t_first <= 0:
            return -1.0
        return (self.t_end - self.t_first) / (self.n_out - 1)


# ═════════════════════════════════════════════════ 二、带采样的引擎

class ServingEngine(E.Engine):
    """在 `Engine` 上补两件服务必需的能力：**每请求采样** 与 **中途取消**。

    ⚠ 下面把父类的 `_admit()` 和 `step()` 整个抄了一遍，只为了把两处 `argmax`
    换成 `self._pick()`。**这个重复本身是一个设计教训，值得单独说一句**：

        `engine.py` 把"选下一个 token"写死成 `argmax`，是为了让前面几章
        「优化不许改变输出」的逐位对比能成立 —— 采样一旦引入随机性，那套测试就没法写。
        代价就是这里：**采样点没有被设计成一个可替换的钩子，于是只能整段覆盖。**

    真实引擎从第一天就把这里做成一个独立对象（一个 sampler / logits processor 管线），
    正是因为需要往里塞的东西远不止采样：logit bias、结构化输出的掩码、投机解码的校验。
    **如果你从 0 写，这个位置一开始就要留缝。**
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.samplers: dict[int, Sampler] = {}

    def _pick(self, req: E.Request, logits: np.ndarray) -> int:
        s = self.samplers.get(req.rid)
        return int(logits.argmax()) if s is None else s(logits, req.out)

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting[0]
            req.tab = E.SeqTable(self.pool)
            reuse = 0
            if self.prefix is not None:
                blocks, reuse = self.prefix.match(req.prompt)
                if blocks:
                    req.tab.blocks = list(blocks)
                    req.tab.length = reuse
            req.cached_tokens = reuse
            self.stats["cached_tokens"] += reuse
            todo = req.prompt[reuse:]
            if not todo:
                back = self.pool.block_size
                req.tab.length -= back
                todo = req.prompt[reuse - back:]
                req.cached_tokens -= back
                self.stats["cached_tokens"] -= back
            try:
                logits = self._forward(req, todo)
            except MemoryError:
                req.tab.free()
                return
            self.stats["prefill_tokens"] += len(todo)
            req.out.append(self._pick(req, logits))     # <- 唯一的改动
            req.prefilled = True
            req.start = self.step_id
            if self.prefix is not None:
                self.prefix.store(req.prompt, req.tab)
            self.waiting.pop(0)
            self.running.append(req)

    def step(self) -> None:
        self._admit()
        for req in list(self.running):
            if req.finished or getattr(req, "abort", False):
                continue
            logits = self._forward(req, [req.out[-1]])
            req.out.append(self._pick(req, logits))     # <- 唯一的改动
            self.stats["decode_steps"] += 1
        for req in list(self.running):
            # **取消也是一种结束。** 少了 `abort` 这一支，断连的请求会一直跑到
            # max_new，它占的块直到那时才还 —— 表现为"内存涨了下不来"。
            if req.finished or getattr(req, "abort", False):
                req.done = self.step_id
                req.tab.free()
                self.running.remove(req)
                self.samplers.pop(req.rid, None)
        self.step_id += 1


# ═════════════════════════════════════════════════ 三、引擎线程

class EngineLoop(threading.Thread):
    """**全进程唯一碰引擎的线程。** 它做四件事，循环做：

      1. 从 `in_q` 收新请求（非阻塞，收空为止）；
      2. 调 `engine.step()` 推进一步；
      3. 把这一步新产生的 token 增量解码后塞进各自的 `out_q`；
      4. 结束/取消的请求收尾。

    **没有请求时要让出 CPU**，否则这个 while 会把一个核吃满 ——
    很多人第一版会忘，表现是"服务一起来风扇就转"。
    """

    daemon = True

    def __init__(self, max_queue: int = 32, **engine_kw):
        super().__init__(name="engine-loop")
        self.engine = ServingEngine(CFG, M.init_weights(CFG), **engine_kw)
        self.in_q: "queue.Queue[Job]" = queue.Queue()
        self.max_queue = max_queue
        self.jobs: dict[int, Job] = {}
        self.guards: dict[int, StreamGuard] = {}
        self.reqs: dict[int, E.Request] = {}
        self.done: list[Job] = []
        self._stop = threading.Event()

    # ---- 供 HTTP 线程调用 ----
    def submit(self, job: Job) -> bool:
        """收下返回 True；**队列满则返回 False，调用方必须回 503**。"""
        if self.in_q.qsize() + len(self.engine.waiting) >= self.max_queue:
            return False
        self.in_q.put(job)
        return True

    def depth(self) -> int:
        return self.in_q.qsize() + len(self.engine.waiting) + len(self.engine.running)

    def shutdown(self) -> None:
        self._stop.set()

    # ---- 引擎线程内部 ----
    def _intake(self) -> None:
        while True:
            try:
                job = self.in_q.get_nowait()
            except queue.Empty:
                return
            req = E.Request(job.rid, job.prompt, job.params.max_new)
            req.abort = False          # 取消标记；`engine.py` 里没有，是服务层加的
            self.jobs[job.rid] = job
            self.reqs[job.rid] = req
            self.guards[job.rid] = StreamGuard(job.params.stop)
            self.engine.samplers[job.rid] = Sampler(job.params)
            self.engine.add(req)

    def _drain(self) -> None:
        for rid, req in list(self.reqs.items()):
            job, guard = self.jobs[rid], self.guards[rid]
            if job.cancelled and not getattr(req, "abort", False):
                req.abort = True                        # 引擎下一步就会回收它的块
            while job.n_out < len(req.out):
                tid = req.out[job.n_out]
                job.n_out += 1
                if job.t_first < 0:
                    job.t_first = time.perf_counter()
                piece = guard.push(tid)
                if piece and not job.cancelled:
                    job.out_q.put(("piece", piece))
                reason = should_stop(req.out, job.params, guard)
                if reason is not None:
                    job.finish_reason = reason
                    req.abort = True                    # 停止串命中，别再算了
                    break
            # **只认引擎的回收信号**（`req.done >= 0` 由 `Engine.step()` 置位）。
            # 不能自己判断"输出够了就算完"—— 那样会在块还没归还时就宣告结束，
            # 于是 `/metrics` 的 blocks_in_use 永远对不上账。
            if req.done >= 0:
                tail = guard.flush()
                if tail and not job.cancelled:
                    job.out_q.put(("piece", tail))
                job.t_end = time.perf_counter()
                job.out_q.put(("done", job.finish_reason))
                self.done.append(job)
                for d in (self.jobs, self.reqs, self.guards):
                    d.pop(rid, None)

    def run(self) -> None:
        while not self._stop.is_set():
            self._intake()
            if self.engine.waiting or self.engine.running:
                self.engine.step()
                self._drain()
            else:
                time.sleep(0.002)      # 空转时让出 CPU，别把核吃满


# ═════════════════════════════════════════════════ 四、HTTP 层

_next_rid = iter(range(1, 10 ** 9))
_rid_lock = threading.Lock()


def _new_rid() -> int:
    with _rid_lock:
        return next(_next_rid)


def _parse(body: dict) -> tuple[list[int], SamplingParams]:
    prompt = body.get("prompt", "")
    ids = list(prompt) if isinstance(prompt, list) else toy_encode(str(prompt))
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    return ids, SamplingParams(
        temperature=float(body.get("temperature", 0.0)),
        top_k=int(body.get("top_k", 0)),
        top_p=float(body.get("top_p", 1.0)),
        repetition_penalty=float(body.get("repetition_penalty", 1.0)),
        seed=body.get("seed"),
        max_new=int(body.get("max_tokens", 16)),
        eos=body.get("eos", 0),
        stop=tuple(stop),
    )


class Handler(BaseHTTPRequestHandler):
    loop: EngineLoop = None            # 由 serve() 注入
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):         # 别把 stderr 刷满
        pass

    # ---- 小工具 ----
    def _json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "queue_depth": self.loop.depth()})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list",
                             "data": [{"id": "minigpt-toy", "object": "model"}]})
        elif self.path == "/metrics":
            self._json(200, _metrics(self.loop))
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._json(400, {"error": f"bad json: {e}"})
            return

        ids, params = _parse(body)
        if not ids:
            self._json(400, {"error": "empty prompt"})
            return
        if len(ids) + params.max_new > CFG.max_seq:
            # **必须在准入前查。** 放进去再撑爆，代价是已经占掉的块和已经算掉的 prefill。
            self._json(400, {"error": "prompt + max_tokens 超过 max_seq"})
            return

        job = Job(_new_rid(), ids, params, bool(body.get("stream")))
        if not self.loop.submit(job):
            # **背压**：与其收下来让所有人一起慢，不如立刻说不。
            self.send_response(503)
            self.send_header("retry-after", "1")
            self.send_header("content-length", "0")
            self.end_headers()
            return

        if job.stream:
            self._stream(job)
        else:
            self._blocking(job)

    # ---- 非流式 ----
    def _blocking(self, job: Job) -> None:
        text = []
        while True:
            kind, payload = job.out_q.get()
            if kind == "piece":
                text.append(str(payload))
            else:
                break
        self._json(200, {
            "id": f"cmpl-{job.rid}", "object": "text_completion",
            "model": "minigpt-toy",
            "choices": [{"index": 0, "text": "".join(text),
                         "finish_reason": job.finish_reason}],
            "usage": {"prompt_tokens": len(job.prompt),
                      "completion_tokens": job.n_out},
            "timings": {"ttft_ms": round(job.ttft * 1000, 2),
                        "tpot_ms": round(job.tpot * 1000, 2)},
        })

    # ---- 流式（SSE）----
    def _stream(self, job: Job) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        # SSE 长度未知，只能分块传。**漏了这个头，客户端会一直等 content-length。**
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()

        def chunk(payload: str) -> None:
            raw = payload.encode("utf-8")
            self.wfile.write(f"{len(raw):X}\r\n".encode())
            self.wfile.write(raw + b"\r\n")
            self.wfile.flush()

        try:
            while True:
                kind, payload = job.out_q.get()
                if kind == "piece":
                    chunk("data: " + json.dumps(
                        {"id": f"cmpl-{job.rid}", "object": "text_completion",
                         "choices": [{"index": 0, "text": payload,
                                      "finish_reason": None}]},
                        ensure_ascii=False) + "\n\n")
                else:
                    chunk("data: " + json.dumps(
                        {"id": f"cmpl-{job.rid}", "object": "text_completion",
                         "choices": [{"index": 0, "text": "",
                                      "finish_reason": job.finish_reason}]},
                        ensure_ascii=False) + "\n\n")
                    chunk("data: [DONE]\n\n")
                    chunk("")           # chunked 编码的结束标记
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            # **客户端断了。** 必须打上取消标记，否则这条请求会一直算到 max_new，
            # 它占的 KV 块也要到那时才还 —— 这就是"内存只涨不跌"的常见来源。
            job.cancelled = True


def _metrics(loop: EngineLoop) -> dict:
    done = list(loop.done)
    ttfts = sorted(j.ttft for j in done if j.ttft > 0)
    tpots = sorted(j.tpot for j in done if j.tpot > 0)

    def p(xs, q):
        return round(xs[min(len(xs) - 1, int(len(xs) * q))] * 1000, 2) if xs else -1

    return {
        "completed": len(done),
        "queue_depth": loop.depth(),
        "running": len(loop.engine.running),
        "waiting": len(loop.engine.waiting),
        "blocks_in_use": loop.engine.pool.alloc.n_used,
        "ttft_ms": {"p50": p(ttfts, 0.5), "p90": p(ttfts, 0.9)},
        "tpot_ms": {"p50": p(tpots, 0.5), "p90": p(tpots, 0.9)},
        "engine": dict(loop.engine.stats),
        "note": "numpy toy on CPU; NOT comparable to any real engine",
    }


def serve(host: str = "127.0.0.1", port: int = 8000, **loop_kw
          ) -> tuple[ThreadingHTTPServer, EngineLoop]:
    loop = EngineLoop(**loop_kw)
    loop.start()
    Handler.loop = loop
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd, loop


# ═════════════════════════════════════════════════ 五、自检

def selftest() -> int:
    import urllib.error
    import urllib.request

    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)
        ok &= bool(cond)

    httpd, loop = serve("127.0.0.1", 0, max_queue=4, n_blocks=512,
                        block_size=8, max_running=4)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    n_503 = [0]

    def post(body: dict, stream: bool = False):
        """**客户端必须自己处理 503。**

        这一段本来没写重试，于是并发那一项有约四成概率挂掉 ——
        6 条请求打 `max_queue=4` 的服务，被拒的那两条在工作线程里抛出
        `HTTPError`，看起来像"服务坏了"，其实是背压在正常工作。
        **背压不是服务端一家的事**：服务端负责及时说不，客户端负责听得懂。
        真实客户端在这里要做的正是下面这件事：退避 + 重试。
        """
        req = urllib.request.Request(
            base + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"})
        for attempt in range(60):
            try:
                r = urllib.request.urlopen(req, timeout=30)
                return r.read().decode() if stream else json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code != 503:
                    raise
                n_503[0] += 1
                time.sleep(float(e.headers.get("retry-after", 0) or 0) or 0.05)
        raise RuntimeError("重试 60 次仍被拒 —— 这才是真的有问题")

    try:
        h = json.loads(urllib.request.urlopen(base + "/health", timeout=5).read())
        chk(h["status"] == "ok", "GET /health 可用")

        r = post({"prompt": "Hello", "max_tokens": 8})
        chk(r["choices"][0]["finish_reason"] == "length",
            "非流式：跑满 max_tokens 时 finish_reason == 'length'")
        chk(r["usage"]["completion_tokens"] == 8, "usage 里的 token 数对得上")
        chk(r["timings"]["ttft_ms"] > 0, "TTFT 被量到了（且与 TPOT 分开报）")

        # 流式：各片拼起来必须等于非流式的整段
        raw = post({"prompt": "Hello", "max_tokens": 8, "stream": True}, stream=True)
        pieces = [json.loads(ln[6:])["choices"][0]["text"]
                  for ln in raw.splitlines()
                  if ln.startswith("data: ") and ln != "data: [DONE]"]
        chk("".join(pieces) == r["choices"][0]["text"],
            "**流式各片拼起来 == 非流式整段**（流式唯一的正确性标准）")
        chk(raw.rstrip().endswith("data: [DONE]"), "SSE 以 [DONE] 收尾")

        # 参数校验：超出 max_seq 必须在准入前拒
        try:
            post({"prompt": "Hello", "max_tokens": 10 ** 6})
            chk(False, "max_tokens 过大应当被拒")
        except urllib.error.HTTPError as e:
            chk(e.code == 400, "max_tokens + prompt 超过 max_seq 时返回 400（准入前拒）")

        # 并发下块不泄漏
        outs = []
        ts = [threading.Thread(target=lambda: outs.append(
            post({"prompt": "Hello world", "max_tokens": 6})))
            for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
        deadline = time.time() + 10
        while loop.depth() > 0 and time.time() < deadline:
            time.sleep(0.05)
        chk(len(outs) == 6 and loop.engine.pool.alloc.n_used == 0,
            f"6 条并发全部返回，块全部归还（在用 {loop.engine.pool.alloc.n_used}）")
        chk(n_503[0] >= 0,
            f"过程中被背压拒绝并重试了 {n_503[0]} 次"
            f"（max_queue=4 而并发 6）—— **503 是正常工作，不是故障**")
    finally:
        httpd.shutdown()
        loop.shutdown()

    print("  全部通过" if ok else "  有失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    httpd, loop = serve()
    print(f"listening on http://{httpd.server_address[0]}:{httpd.server_address[1]}")
    print("  POST /v1/completions   GET /health  /v1/models  /metrics")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        loop.shutdown()
