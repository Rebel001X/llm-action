"""engine.py —— 把前面几件拼成一个能连续服务的引擎：连续批处理 + 前缀缓存。

到这里为止我们有了：
  * `minigpt.py` —— 能算的模型 + KV 缓存
  * `paged.py`   —— 分页 KV，把"必须物理连续"这个假设拆掉
本文件补上最后两件，一个引擎就成型了：

  1. **连续批处理（continuous batching）**
     朴素做法是「攒一批 → 一起跑到全部结束 → 再攒下一批」，
     于是**最慢的那条拖着所有人**，而且新请求必须等整批结束才能进来。
     连续批处理把调度粒度从"一批"降到"一步"：**每一步都重新决定这一步跑哪些请求**，
     谁结束了就立刻让位，新来的立刻能挤进去。

  2. **前缀缓存（prefix caching）**
     两条请求如果开头的 token 完全一样，那它们前面这段的 K/V 也**逐位相同**
     （因为 K/V 只依赖该位置及之前的输入 —— 见 `minigpt.py` 的推导）。
     既然一样，就没必要各算一遍、各存一份。
     做法：按块算内容哈希，哈希相同的块直接复用同一个物理块。

⚠️ 本文件里的耗时数字是**这个 numpy 玩具引擎在 CPU 上的实测**，测的是它自己。
   **绝不能外推到 vLLM/SGLang** —— 那需要 GPU，本机没有。
"""
from __future__ import annotations

import time

import numpy as np

import minigpt as M
from minigpt import Config
from paged import PagedKVCache, SeqTable


# ────────────────────────────────────────────────── 前缀缓存

class PrefixCache:
    """按块的内容哈希做前缀共享。

    **哈希必须是链式的**：`hash(父块哈希, 本块token)`。
    为什么不能只 hash 本块的 token —— 那样 "AB|CD" 和 "XY|CD" 的第二块会撞成同一个 key，
    可它们的 K/V 完全不同（K/V 依赖**全部**历史）。**这会静默给出错误答案。**
    真实引擎（vLLM）也是链式的，见隔壁研究库对 `kv_cache_utils.py` 的解剖。
    """

    def __init__(self, pool: PagedKVCache):
        self.pool = pool
        self.table: dict[tuple, int] = {}          # 块哈希 -> 物理块号
        self.hits = 0
        self.lookups = 0

    def block_hashes(self, tokens: list[int]) -> list[tuple]:
        """只对**满块**算哈希 —— 不满的块内容还会变，不能进缓存。"""
        bs = self.pool.block_size
        out, parent = [], None
        for i in range(0, len(tokens) - bs + 1, bs):
            parent = (parent, tuple(tokens[i:i + bs]))
            out.append(parent)
        return out

    def match(self, tokens: list[int]) -> tuple[list[int], int]:
        """返回 (可复用的物理块列表, 命中的 token 数)。

        **一旦某块 miss 就必须停** —— 链式哈希决定了后面的块不可能命中。
        """
        blocks: list[int] = []
        for h in self.block_hashes(tokens):
            self.lookups += 1
            b = self.table.get(h)
            if b is None:
                break
            blocks.append(b)
            self.pool.alloc.incref(b)              # 复用即加引用，防止被别人释放掉
            self.hits += 1
        return blocks, len(blocks) * self.pool.block_size

    def store(self, tokens: list[int], tab: SeqTable) -> None:
        """把这条请求的满块登记进缓存，供后来者复用。"""
        for i, h in enumerate(self.block_hashes(tokens)):
            if h not in self.table and i < len(tab.blocks):
                self.table[h] = tab.blocks[i]
                self.pool.alloc.incref(tab.blocks[i])   # 缓存自己也持有一份引用


# ────────────────────────────────────────────────── 请求与引擎

class Request:
    def __init__(self, rid: int, prompt: list[int], max_new: int, arrive: int = 0):
        self.rid, self.prompt, self.max_new, self.arrive = rid, prompt, max_new, arrive
        self.out: list[int] = []
        self.tab: SeqTable | None = None
        self.prefilled = False
        self.cached_tokens = 0
        self.start = -1
        self.done = -1

    @property
    def finished(self) -> bool:
        return len(self.out) >= self.max_new


class Engine:
    """一个能连续服务的最小引擎。

    每一步（`step()`）做三件事：
      1. **准入**：在预算允许的前提下，从等待队列放请求进来（做 prefill）；
      2. **推进**：让所有在跑的请求各出一个 token（decode）；
      3. **回收**：结束的请求释放块、腾出位置。

    真实引擎在这三步里还有抢占、chunked prefill、overlap 调度等等，
    本文件**刻意不做** —— 教学库要的是把主干露出来。
    """

    def __init__(self, cfg: Config, w: dict, n_blocks: int = 512, block_size: int = 16,
                 max_running: int = 8, enable_prefix_cache: bool = True):
        self.cfg, self.w = cfg, w
        self.pool = PagedKVCache(cfg, n_blocks=n_blocks, block_size=block_size)
        self.prefix = PrefixCache(self.pool) if enable_prefix_cache else None
        self.max_running = max_running
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.step_id = 0
        self.stats = {"prefill_tokens": 0, "decode_steps": 0, "cached_tokens": 0}

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    # ---- 前向：单条请求走分页路径 ----
    def _forward(self, req: Request, new_tokens: list[int]) -> np.ndarray:
        cfg, w, tab = self.cfg, self.w, req.tab
        T = len(new_tokens)
        p0 = tab.length
        tab.begin(T)
        x = w["wte"][new_tokens] + w["wpe"][p0:p0 + T]
        causal = np.triu(np.full((T, T), -1e9, np.float32), k=1)
        for li, lw in enumerate(w["layers"]):
            h = M.layernorm(x, lw["ln1_g"], lw["ln1_b"])
            q = (h @ lw["wq"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)
            kn = (h @ lw["wk"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)
            vn = (h @ lw["wv"]).reshape(T, cfg.n_head, cfg.d_head).transpose(1, 0, 2)
            tab.append(li, kn, vn)
            k_all, v_all = tab.gather(li)
            scores = q @ k_all.transpose(0, 2, 1) / np.sqrt(cfg.d_head)
            scores[:, :, p0:] += causal
            o = (M.softmax(scores) @ v_all).transpose(1, 0, 2).reshape(T, cfg.d_model)
            x = x + o @ lw["wo"]
            h = M.layernorm(x, lw["ln2_g"], lw["ln2_b"])
            x = x + M.gelu(h @ lw["w1"]) @ lw["w2"]
        tab.commit()
        x = M.layernorm(x, w["lnf_g"], w["lnf_b"])
        return x[-1] @ w["wte"].T

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting[0]
            req.tab = SeqTable(self.pool)

            # 前缀缓存：能复用多少就少算多少
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
                # 整段 prompt 都命中了。**必须回退一个块**：
                # 最后一个 token 的 logits 还没算过，没有它就没法开始 decode。
                back = self.pool.block_size
                req.tab.length -= back
                todo = req.prompt[reuse - back:]
                req.cached_tokens -= back
                self.stats["cached_tokens"] -= back
            try:
                logits = self._forward(req, todo)
            except MemoryError:
                req.tab.free()
                return                              # 块不够，这一步先不放它进来
            self.stats["prefill_tokens"] += len(todo)
            req.out.append(int(logits.argmax()))
            req.prefilled = True
            req.start = self.step_id
            if self.prefix is not None:
                self.prefix.store(req.prompt, req.tab)
            self.waiting.pop(0)
            self.running.append(req)

    def step(self) -> None:
        self._admit()
        for req in list(self.running):
            if req.finished:
                continue
            logits = self._forward(req, [req.out[-1]])
            req.out.append(int(logits.argmax()))
            self.stats["decode_steps"] += 1
        for req in list(self.running):
            if req.finished:
                req.done = self.step_id
                req.tab.free()
                self.running.remove(req)
        self.step_id += 1

    def run(self, max_steps: int = 10000) -> dict:
        t0 = time.perf_counter()
        while (self.waiting or self.running) and self.step_id < max_steps:
            self.step()
        return {**self.stats, "steps": self.step_id,
                "wall_seconds": round(time.perf_counter() - t0, 3),
                "prefix_hits": self.prefix.hits if self.prefix else 0,
                "blocks_in_use": self.pool.alloc.n_used}


# ────────────────────────────────────────────────── 自检

def selftest() -> int:
    ok = True
    cfg = Config(n_layer=2, n_head=2, d_model=32, vocab=64, max_seq=128, seed=1)
    w = M.init_weights(cfg)
    shared = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    p1 = shared + [30, 31]
    p2 = shared + [40, 41]

    # 参照：不用引擎，直接朴素生成
    def ref(prompt, n):
        out = []
        toks = list(prompt)
        for _ in range(n):
            nxt = int(M.forward_naive(cfg, w, toks).argmax())
            out.append(nxt)
            toks.append(nxt)
        return out

    want1, want2 = ref(p1, 6), ref(p2, 6)

    # 1) **最重要的命题：批处理不许改变单条请求的输出。**
    for use_pc in (False, True):
        e = Engine(cfg, w, n_blocks=256, block_size=4, max_running=4,
                   enable_prefix_cache=use_pc)
        r1, r2 = Request(1, p1, 6), Request(2, p2, 6)
        e.add(r1); e.add(r2)
        e.run()
        if r1.out != want1 or r2.out != want2:
            print(f"FAIL 批处理改变了输出 (prefix_cache={use_pc})\n"
                  f"  r1 得到 {r1.out}\n  r1 期望 {want1}\n"
                  f"  r2 得到 {r2.out}\n  r2 期望 {want2}")
            ok = False

    # 2) **前缀缓存不许改变输出，但必须真的命中。**
    e = Engine(cfg, w, n_blocks=256, block_size=4, max_running=4, enable_prefix_cache=True)
    e.add(Request(1, p1, 4)); e.add(Request(2, p2, 4))
    st = e.run()
    if st["prefix_hits"] == 0:
        print("FAIL 两条请求共享 16 个 token 前缀，应当有命中"); ok = False
    if st["cached_tokens"] <= 0:
        print(f"FAIL 复用 token 数应 > 0 -> {st['cached_tokens']}"); ok = False

    # 3) 开前缀缓存必须**减少** prefill 的 token 数（这是它唯一的收益来源）
    def prefill_of(pc: bool) -> int:
        en = Engine(cfg, w, n_blocks=256, block_size=4, max_running=4, enable_prefix_cache=pc)
        en.add(Request(1, p1, 4)); en.add(Request(2, p2, 4))
        return en.run()["prefill_tokens"]
    off, on = prefill_of(False), prefill_of(True)
    if not on < off:
        print(f"FAIL 前缀缓存应减少 prefill token：关={off} 开={on}"); ok = False

    # 4) 连续批处理：先来的短请求应当**先结束**，不必等长的那条
    e = Engine(cfg, w, n_blocks=512, block_size=4, max_running=4)
    short, long_ = Request(1, p1, 2), Request(2, p2, 12)
    e.add(short); e.add(long_)
    e.run()
    if not (short.done < long_.done):
        print(f"FAIL 短请求应先结束：short={short.done} long={long_.done}"); ok = False

    # 5) 所有请求结束后，块必须全部归还（不许泄漏）
    #    注：开了前缀缓存时缓存自己持有引用，所以这里用不开缓存的引擎验
    e = Engine(cfg, w, n_blocks=64, block_size=4, max_running=2, enable_prefix_cache=False)
    for i in range(6):
        e.add(Request(i, [1, 2, 3, 4, 5, 6, 7, 8], 3))
    e.run()
    if e.pool.alloc.n_used != 0:
        print(f"FAIL 全部结束后应无占用，实际 {e.pool.alloc.n_used}"); ok = False

    # 6) **链式哈希**：前缀不同、后一块 token 相同时不许撞 key
    pool = PagedKVCache(cfg, n_blocks=32, block_size=2)
    pc = PrefixCache(pool)
    h_ab = pc.block_hashes([1, 2, 5, 6])
    h_xy = pc.block_hashes([3, 4, 5, 6])
    if h_ab[1] == h_xy[1]:
        print("FAIL 链式哈希失效：不同前缀的同内容块撞了 key（会静默给出错答案）"); ok = False

    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())

    cfg = Config(n_layer=4, n_head=4, d_model=128, vocab=512, max_seq=512)
    w = M.init_weights(cfg)
    sysp = list(range(100, 164))                    # 64 token 的共享 system prompt

    print("同一批请求，开/关前缀缓存的对比（本玩具引擎在 CPU 上的实测）")
    print("⚠️ 这些数字只说明这个 numpy 玩具的行为，**不可外推到真实引擎**。\n")
    print(f"{'前缀缓存':>10}{'prefill token':>15}{'复用 token':>12}"
          f"{'命中块':>9}{'步数':>7}{'墙钟秒':>9}")
    for pc in (False, True):
        e = Engine(cfg, w, n_blocks=1024, block_size=16, max_running=8,
                   enable_prefix_cache=pc)
        for i in range(8):
            e.add(Request(i, sysp + [200 + i, 201 + i], 8))
        st = e.run()
        print(f"{'开' if pc else '关':>10}{st['prefill_tokens']:>15,}"
              f"{st['cached_tokens']:>12,}{st['prefix_hits']:>9}"
              f"{st['steps']:>7}{st['wall_seconds']:>9.3f}")
    print("\n8 条请求共享同一段 64-token 的 system prompt："
          "\n开缓存后，重复的那一段只在第一条请求上算了一次。")
