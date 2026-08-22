"""prefix_sim.py —— 前缀缓存策略的**差分模拟器**：把 vLLM 与 SGLang 的匹配/驱逐策略
如实重实现在纯 Python 里，喂同一批请求 trace，量化它们在什么工作负载下给出不同结果。

════════════════════════════════════════════════════════════════════════
这个模拟器**不是什么**（先说清楚，免得被误读）
════════════════════════════════════════════════════════════════════════
* 它**不测性能**。没有显存、没有 kernel、没有拷贝开销。本机无 GPU，本库不产实测数字。
* 它**不是引擎**。只建模「给定 token 序列，能命中多少前缀 / 淘汰谁」这一层决策。
* 因此它的输出只能回答**策略层面的问题**：同一条 trace 下，两家的命中 token 数、
  驱逐次数、结构维护操作数差多少，以及**差异出现在什么工作负载上**。
  任何"谁更快"的结论都不能从这里得出。

════════════════════════════════════════════════════════════════════════
建模依据（每条行为都对应真实源码，行号可被 _verify.py 核）
════════════════════════════════════════════════════════════════════════
vLLM（`vllm` @ 7ca49fbe）
  * 块哈希是**链式**的：`hash((parent_block_hash, block_token_ids, extra_keys))`
    —— `vllm/v1/core/kv_cache_utils.py:620`。因此命中必须是从头开始的连续块链，
    中间断一块，后面全部作废。
  * Phase 1：从头逐块查表，miss 即 break —— `vllm/v1/core/single_type_kv_cache_manager.py:734`
    的注释原文 "A missing block implies every later block misses too (chained hashes)"。
  * Phase 2（**细粒度模式**，`alignment_tokens < block_size` 时才有）：在第一个不满块
    **内部**按 alignment 边界从高到低回探，能再多命中一截
    —— 同文件 `:742` 起。**这一条推翻了"vLLM 只能按块粒度命中"的通常说法**，
    本模拟器把它做成开关 `fine_grained`，好量化它到底值多少。
  * 驱逐：`BlockPool.free_blocks` 把释放的块分两路 —— 无 hash 的走 LIFO（GPU 局部性），
    带 hash 的走 FIFO（缓存寿命）；且释放一个请求时先把块列表**倒序**
    —— `vllm/v1/core/block_pool.py:719` 起，源码注释原文见该处。

SGLang（`sglang` @ 15a43983）
  * 基数树匹配，且 `RadixKey.match` **向下取整到 page_size**
    —— `python/sglang/srt/mem_cache/radix_cache.py:181`。所以 page_size>1 时
    它同样不是 token 粒度。
  * 部分匹配时**分裂节点** `_split_node` —— 同文件 `:705`。
  * 驱逐：对**可驱逐叶子**建小顶堆，弹最低优先级（默认 LRU 按 last_access_time），
    整节点释放；父节点变成无子且未上锁时再入堆 —— 同文件 `:593`。
  * `lock_ref` 保护在用节点不被驱逐 —— 同文件 `:623`。

用法:
    python prefix_sim.py                # 跑全部实验，落 out/prefix_sim.json
    python prefix_sim.py --selftest
"""
from __future__ import annotations

import heapq
import json
import sys
from dataclasses import dataclass, field

from common import OUT

# ──────────────────────────────────────────────────────────── 工作负载

class Rng:
    """自带线性同余随机数 —— 不依赖 random 模块的实现细节，保证跨机器逐位可复现。"""

    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF or 1

    def next(self) -> int:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s

    def randint(self, lo: int, hi: int) -> int:
        return lo + self.next() % (hi - lo + 1) if hi > lo else lo

    def choice(self, seq):
        return seq[self.next() % len(seq)]


def gen_workload(kind: str, n_req: int, seed: int = 7) -> list[list[int]]:
    """造请求 trace。每条请求是一串 token id。

    四种工作负载，覆盖「共享前缀」这个自变量的四种典型形态：
      shared_system  —— 长 system prompt 完全共享，后接短用户问题（企业对话服务）
      tree_branch    —— 树状分叉：同一前缀展开多个分支（Agent/搜索/best-of-k）
      no_share       —— 几乎无共享（每条随机），前缀缓存应当**接近零收益**
      hot_cold       —— 少量热前缀反复命中 + 大量冷请求冲刷（真实线上流量的样子）
    """
    r = Rng(seed)
    out: list[list[int]] = []

    if kind == "shared_system":
        sys_prompt = [1000 + i for i in range(512)]
        for _ in range(n_req):
            out.append(sys_prompt + [r.randint(1, 999) for _ in range(r.randint(8, 64))])

    elif kind == "tree_branch":
        root = [2000 + i for i in range(128)]
        mids = [[3000 + b * 100 + i for i in range(96)] for b in range(4)]
        for _ in range(n_req):
            out.append(root + r.choice(mids) + [r.randint(1, 999) for _ in range(r.randint(8, 48))])

    elif kind == "misaligned_share":
        # 共享前缀 517 token —— **故意不是 block_size(16) 的整数倍**。
        # 块粒度匹配只能命中 512，剩下 5 个 token 每条请求都要重算。
        # 这是"块粒度 vs token 粒度"唯一真正会分叉的形态。
        sys_prompt = [5000 + i for i in range(517)]
        for _ in range(n_req):
            out.append(sys_prompt + [r.randint(1, 999) for _ in range(r.randint(8, 64))])

    elif kind == "no_share":
        for _ in range(n_req):
            out.append([r.randint(1, 50000) for _ in range(r.randint(200, 600))])

    elif kind == "hot_cold":
        hot = [[4000 + h * 1000 + i for i in range(256)] for h in range(3)]
        for k in range(n_req):
            if k % 3 == 0:                       # 三分之一命中热前缀
                out.append(r.choice(hot) + [r.randint(1, 999) for _ in range(r.randint(8, 40))])
            else:                                # 其余是冷流量，负责冲刷缓存
                out.append([r.randint(1, 50000) for _ in range(r.randint(150, 400))])
    else:
        raise ValueError(kind)
    return out


# ──────────────────────────────────────────────────────────── vLLM 侧

@dataclass
class VllmStats:
    hit_tokens: int = 0
    req_tokens: int = 0
    lookups: int = 0            # 哈希表查询次数（Phase1 + Phase2）
    fine_grained_extra: int = 0  # Phase2 额外多命中的 token 数
    evictions: int = 0          # 被驱逐的块数
    peak_cached_blocks: int = 0


class VllmBlockCache:
    """vLLM 的链式块哈希前缀缓存。

    `capacity_blocks` 是缓存池能容纳的块数；超了就按 FIFO 驱逐带 hash 的块
    （对应 `BlockPool.free_blocks` 里 "FIFO reuse of cached blocks for LRU
    eviction behavior" 那一路）。
    """

    def __init__(self, block_size: int = 16, capacity_blocks: int = 2048,
                 hash_block_size: int | None = None, evict: str = "freequeue") -> None:
        # evict="freequeue" 是 vLLM 的**真实行为**，比看起来复杂：
        #   * 空闲队列是驱逐候选队列，从**队头**驱逐；
        #   * 命中一个块时 `touch()` 把它从队列里**摘掉**（block_pool.py:702、:714）；
        #   * 请求结束释放时，带 hash 的块**追加到队尾**（block_pool.py:737 附近，
        #     注释原文 "FIFO reuse of cached blocks for LRU eviction behavior"），
        #     且释放前会把请求的块列表**倒序**，让尾部块排在更靠前被驱逐。
        #   合起来 ⇒ 实际顺序是「按最近一次释放时间」，**近似 LRU 而不是 FIFO**。
        # evict="insert_fifo" 是我第一版写错的稻草人（按首次插入排序，命中不改顺序），
        #   保留下来只为量化"把 vLLM 误建模成 FIFO 会高估多少差距"。
        # evict="lru" 是受控对照：按最近一次**访问**时间。
        self.bs = block_size
        # hash_block_size < block_size 时进入 Phase2 细粒度模式
        self.hbs = hash_block_size or block_size
        assert self.bs % self.hbs == 0, "block_size 必须是 hash_block_size 的整数倍"
        self.fine_grained = self.hbs < self.bs
        self.cap = capacity_blocks
        self.evict_policy = evict
        self.table: dict[tuple, int] = {}       # block_hash -> 逻辑块 id
        # 用 dict 当有序集合：插入保序、删除 O(1)。队头＝最先被驱逐。
        self.queue: dict[tuple, None] = {}
        self.fifo: list[tuple] = []             # 仅 insert_fifo 稻草人用
        self.last_use: dict[tuple, int] = {}    # 仅 lru 对照组用
        self.clock = 0
        self.st = VllmStats()
        self._next_id = 0

    def _chain(self, toks: list[int]) -> list[tuple]:
        """按 **hash 单元** 切并算链式哈希 —— 注意这条链**只算一次**。

        关键（第一版我在这里搞错过）：vLLM 不是维护两条独立的链。
        `resolve_block_hashes` 保留的是 hash 粒度的原始列表，
        块级视图 `BlockHashListWithBlockSize(block_hashes, alignment_tokens, block_size)`
        只是在这条链上**按块边界取样**（见
        `vllm/v1/core/single_type_kv_cache_manager.py:722`）。
        当年按 block 和按 hash 各算一条链，两条链的父哈希不同，
        Phase2 永远命中不了 —— 那是模拟器的 bug，不是 vLLM 的行为。
        """
        hs: list[tuple] = []
        parent = None
        u = self.hbs
        for i in range(0, len(toks) - u + 1, u):
            parent = (parent, tuple(toks[i:i + u]))
            hs.append(parent)
        return hs

    def process(self, toks: list[int]) -> int:
        st = self.st
        self.clock += 1
        st.req_tokens += len(toks)
        scale = self.bs // self.hbs
        fine = self._chain(toks)
        n_full_blocks = len(fine) // scale

        # Phase 1：块级视图＝在 fine 链上按块边界取样，逐块查，miss 即停
        hit_blocks = 0
        for i in range(n_full_blocks):
            st.lookups += 1
            h = fine[(i + 1) * scale - 1]
            if h not in self.table:
                break
            self.last_use[h] = self.clock
            self.queue.pop(h, None)      # touch(): 命中即从驱逐候选队列摘掉
            hit_blocks += 1
        hit = hit_blocks * self.bs

        # Phase 2：细粒度模式下，进第一个不满块内部，按 hash 单元从高到低回探
        if self.fine_grained:
            start = hit_blocks * scale
            stop = min(start + scale - 1, len(fine))
            for idx in range(stop - 1, start - 1, -1):
                st.lookups += 1
                if fine[idx] in self.table:
                    self.queue.pop(fine[idx], None)
                    self.last_use[fine[idx]] = self.clock
                    extra = (idx + 1 - start) * self.hbs
                    hit += extra
                    st.fine_grained_extra += extra
                    break

        st.hit_tokens += hit

        # 写回：细粒度模式下连不满块的内部边界一起缓存
        # （对应 block_pool.py 的 cache_full_blocks + cache_partial_block）
        keep = len(fine) if self.fine_grained else n_full_blocks * scale
        used = fine[:keep]
        for h in used:
            if h not in self.table:
                self.table[h] = self._next_id
                self._next_id += 1
                self.fifo.append(h)
            self.last_use[h] = self.clock
        # 请求结束＝释放：把用到的块**倒序**追加到队尾
        # （倒序对应 single_type_kv_cache_manager 释放前反转块列表，
        #   让尾部、最不可能被共享的块排在更靠前被驱逐）
        for h in reversed(used):
            self.queue.pop(h, None)
            self.queue[h] = None

        cap_entries = self.cap * scale
        if self.evict_policy == "freequeue":          # vLLM 真实行为
            while len(self.table) > cap_entries and self.queue:
                old = next(iter(self.queue))
                del self.queue[old]
                if self.table.pop(old, None) is not None:
                    st.evictions += 1
        elif self.evict_policy == "insert_fifo":      # 稻草人：按首次插入
            while len(self.table) > cap_entries and self.fifo:
                old = self.fifo.pop(0)
                if self.table.pop(old, None) is not None:
                    self.queue.pop(old, None)
                    st.evictions += 1
        else:                                          # 受控对照：按最近访问
            while len(self.table) > cap_entries:
                old = min(self.table, key=lambda k: self.last_use.get(k, 0))
                self.table.pop(old, None)
                self.last_use.pop(old, None)
                self.queue.pop(old, None)
                st.evictions += 1
        st.peak_cached_blocks = max(st.peak_cached_blocks, len(self.table) // scale)
        return hit


# ──────────────────────────────────────────────────────────── SGLang 侧

@dataclass
class SgNode:
    key: tuple = ()
    parent: "SgNode | None" = None
    children: dict = field(default_factory=dict)
    last_access: int = 0
    lock_ref: int = 0

    @property
    def n_tokens(self) -> int:
        return len(self.key)


@dataclass
class SgStats:
    hit_tokens: int = 0
    req_tokens: int = 0
    splits: int = 0             # 节点分裂次数（基数树独有的维护开销）
    descends: int = 0           # 下行步数（≈ 匹配复杂度）
    evictions: int = 0          # 被驱逐的节点数
    evicted_tokens: int = 0
    peak_nodes: int = 0


class SglangRadixCache:
    """SGLang 的基数树前缀缓存。

    `capacity_tokens` 是可缓存 token 总量；超了就按「可驱逐叶子的 LRU」驱逐，
    父节点在变成无子且未上锁后重新入堆 —— 对应 `radix_cache.py:593` 的 evict()。
    """

    def __init__(self, page_size: int = 1, capacity_tokens: int = 32768) -> None:
        self.page = page_size
        self.cap = capacity_tokens
        self.root = SgNode()
        self.n_tokens = 0
        self.n_nodes = 1
        self.clock = 0
        self.st = SgStats()

    # match 向下取整到 page —— radix_cache.py:181 的 "rounded down to page_size"
    def _match_len(self, a: tuple, b: tuple) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return (i // self.page) * self.page if self.page > 1 else i

    def _split(self, child: SgNode, at: int) -> SgNode:
        self.st.splits += 1
        new = SgNode(key=child.key[:at], parent=child.parent,
                     last_access=child.last_access, lock_ref=child.lock_ref)
        child.parent.children[child.key[0]] = new
        child.key = child.key[at:]
        child.parent = new
        new.children[child.key[0]] = child
        self.n_nodes += 1
        return new

    def process(self, toks: list[int]) -> int:
        self.clock += 1
        st = self.st
        st.req_tokens += len(toks)

        key = tuple(toks)
        node = self.root
        node.last_access = self.clock
        hit = 0
        while key and key[0] in node.children:
            st.descends += 1
            child = node.children[key[0]]
            child.last_access = self.clock
            m = self._match_len(child.key, key)
            if m == 0:
                break
            if m < len(child.key):
                node = self._split(child, m)
                hit += m
                key = key[m:]
                break
            hit += m
            key = key[m:]
            node = child
        st.hit_tokens += hit

        # 插入剩余部分
        if key:
            leaf = SgNode(key=key, parent=node, last_access=self.clock)
            node.children[key[0]] = leaf
            self.n_nodes += 1
            self.n_tokens += len(key)

        self._evict_if_needed()
        st.peak_nodes = max(st.peak_nodes, self.n_nodes)
        return hit

    def _evictable_leaves(self) -> list[SgNode]:
        out, stack = [], [self.root]
        while stack:
            x = stack.pop()
            if x is not self.root and not x.children and x.lock_ref == 0:
                out.append(x)
            stack.extend(x.children.values())
        return out

    def _evict_if_needed(self) -> None:
        if self.n_tokens <= self.cap:
            return
        st = self.st
        heap = [(x.last_access, id(x), x) for x in self._evictable_leaves()]
        heapq.heapify(heap)
        while self.n_tokens > self.cap and heap:
            _, _, x = heapq.heappop(heap)
            if x.parent is None or x.children:
                continue
            self.n_tokens -= x.n_tokens
            st.evicted_tokens += x.n_tokens
            st.evictions += 1
            x.parent.children.pop(x.key[0], None)
            self.n_nodes -= 1
            p = x.parent
            if p is not self.root and not p.children and p.lock_ref == 0:
                heapq.heappush(heap, (p.last_access, id(p), p))


# ──────────────────────────────────────────────────────────── 实验

def run_case(kind: str, n_req: int, block_size: int, page_size: int,
             cap_tokens: int, hash_block_size: int | None, seed: int) -> dict:
    wl = gen_workload(kind, n_req, seed)
    v = VllmBlockCache(block_size=block_size,
                       capacity_blocks=max(1, cap_tokens // block_size),
                       hash_block_size=hash_block_size)
    # 受控对照组：只把驱逐策略换成 LRU，匹配方式完全不变。
    # 它与 v 的差＝**驱逐策略贡献**；它与 SGLang 的差＝**匹配方式贡献**。
    v_lru = VllmBlockCache(block_size=block_size,
                           capacity_blocks=max(1, cap_tokens // block_size),
                           hash_block_size=hash_block_size, evict="lru")
    # 稻草人组：把 vLLM 误建模成"按首次插入的 FIFO"会得出什么 —— 用来量化我自己的建模误差
    v_straw = VllmBlockCache(block_size=block_size,
                             capacity_blocks=max(1, cap_tokens // block_size),
                             hash_block_size=hash_block_size, evict="insert_fifo")
    s = SglangRadixCache(page_size=page_size, capacity_tokens=cap_tokens)
    for toks in wl:
        v.process(toks)
        v_lru.process(toks)
        v_straw.process(toks)
        s.process(toks)
    vt, sg, vl, vs = v.st, s.st, v_lru.st, v_straw.st
    v_rate = vt.hit_tokens / vt.req_tokens if vt.req_tokens else 0.0
    s_rate = sg.hit_tokens / sg.req_tokens if sg.req_tokens else 0.0
    vl_rate = vl.hit_tokens / vl.req_tokens if vl.req_tokens else 0.0
    vs_rate = vs.hit_tokens / vs.req_tokens if vs.req_tokens else 0.0
    return {
        "workload": kind, "n_req": n_req,
        "block_size": block_size, "page_size": page_size,
        "hash_block_size": hash_block_size or block_size,
        "capacity_tokens": cap_tokens,
        "total_tokens": vt.req_tokens,
        "vllm": {"hit_tokens": vt.hit_tokens, "hit_rate": round(v_rate, 4),
                 "lookups": vt.lookups, "fine_grained_extra": vt.fine_grained_extra,
                 "evicted_blocks": vt.evictions,
                 "peak_cached_blocks": vt.peak_cached_blocks},
        "sglang": {"hit_tokens": sg.hit_tokens, "hit_rate": round(s_rate, 4),
                   "descends": sg.descends, "splits": sg.splits,
                   "evicted_nodes": sg.evictions, "evicted_tokens": sg.evicted_tokens,
                   "peak_nodes": sg.peak_nodes},
        "vllm_lru_control": {"hit_rate": round(vl_rate, 4),
                             "evicted_blocks": vl.evictions},
        "vllm_strawman_insert_fifo": {"hit_rate": round(vs_rate, 4),
                                      "note": "不是 vLLM 的行为，用来量化误建模的代价"},
        "modeling_error_pp": round((v_rate - vs_rate) * 100, 2),
        "delta_hit_tokens": sg.hit_tokens - vt.hit_tokens,
        "delta_hit_rate_pp": round((s_rate - v_rate) * 100, 2),
        # 归因拆解：总差 = 驱逐策略贡献 + 匹配方式贡献
        "attribution": {
            "eviction_policy_pp": round((vl_rate - v_rate) * 100, 2),
            "matching_policy_pp": round((s_rate - vl_rate) * 100, 2),
        },
    }


EXPERIMENTS = [
    ("misaligned_share", 200, 16, 1, 65536, None),   # 共享前缀不块对齐：块粒度会掉 5 token
    ("misaligned_share", 200, 16, 1, 65536, 4),      # 同上，vLLM 开细粒度（hash 单元 4）
    ("misaligned_share", 200, 16, 1, 65536, 1),      # 极限：hash 单元 1 = 逐 token
    # (工作负载, 请求数, vLLM block_size, SGLang page_size, 容量 token, vLLM hash_block_size)
    ("shared_system", 200, 16, 1, 65536, None),
    ("shared_system", 200, 16, 16, 65536, None),   # page_size 拉齐到 block_size
    ("shared_system", 200, 16, 1, 65536, 4),       # vLLM 开细粒度（hash 单元 4）
    ("tree_branch",   200, 16, 1, 65536, None),
    ("tree_branch",   200, 16, 1, 65536, 4),
    ("no_share",      200, 16, 1, 65536, None),
    ("hot_cold",      300, 16, 1, 16384, None),    # 容量收紧，逼出驱逐策略差异
    ("hot_cold",      300, 16, 1, 4096,  None),    # 更紧
]


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    results = [run_case(k, n, bs, ps, cap, hbs, seed=7)
               for (k, n, bs, ps, cap, hbs) in EXPERIMENTS]
    payload = {
        "disclaimer": ("这是**策略模拟**不是性能实测：无显存、无 kernel、无拷贝开销。"
                       "只回答「同一条 trace 下两家能命中多少前缀、淘汰谁」。"
                       "任何'谁更快'的结论都不能从这里得出。"),
        "modeled_from": {
            "vllm_chained_hash": "vllm/v1/core/kv_cache_utils.py:620",
            "vllm_phase1_break": "vllm/v1/core/single_type_kv_cache_manager.py:734",
            "vllm_phase2_fine_grained": "vllm/v1/core/single_type_kv_cache_manager.py:742",
            "vllm_eviction_two_queues": "vllm/v1/core/block_pool.py:719",
            "sglang_match_page_aligned": "python/sglang/srt/mem_cache/radix_cache.py:181",
            "sglang_split_node": "python/sglang/srt/mem_cache/radix_cache.py:705",
            "sglang_evict_leaf_lru": "python/sglang/srt/mem_cache/radix_cache.py:593",
        },
        "results": results,
    }
    fp = OUT / "prefix_sim.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {fp}\n")
    hdr = (f"{'工作负载':<17}{'blk':>4}{'page':>5}{'hash':>5}{'容量':>7}"
           f"{'vLLM真实':>9}{'稻草人FIFO':>11}{'vLLM+LRU':>10}{'SGLang':>8}"
           f"{'总差pp':>8}{'└驱逐pp':>9}{'└匹配pp':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        a = r["attribution"]
        print(f"{r['workload']:<17}{r['block_size']:>4}{r['page_size']:>5}"
              f"{r['hash_block_size']:>5}{r['capacity_tokens']:>7}"
              f"{r['vllm']['hit_rate']:>9.4f}"
              f"{r['vllm_strawman_insert_fifo']['hit_rate']:>11.4f}"
              f"{r['vllm_lru_control']['hit_rate']:>10.4f}"
              f"{r['sglang']['hit_rate']:>8.4f}{r['delta_hit_rate_pp']:>8.2f}"
              f"{a['eviction_policy_pp']:>9.2f}{a['matching_policy_pp']:>9.2f}")
    return 0


# ──────────────────────────────────────────────────────────── 自检

def selftest() -> int:
    ok = True

    # 1) 链式哈希：中间改一个 token，后面所有块都必须失配
    v = VllmBlockCache(block_size=4, capacity_blocks=999)
    base = list(range(20))
    v.process(base)
    changed = base[:8] + [999] + base[9:]
    hit = v.process(changed)
    if hit != 8:
        print(f"FAIL 链式哈希：改第 9 个 token 后应只命中前 8 个，实得 {hit}"); ok = False

    # 2) 基数树：同一条 trace 下，page=1 时能命中到 token 粒度
    s = SglangRadixCache(page_size=1, capacity_tokens=999999)
    s.process(base)
    hit_s = s.process(changed)
    if hit_s != 8:
        print(f"FAIL 基数树命中 -> {hit_s}"); ok = False

    # 3) **两者差异的核心场景**：共享前缀长度不是 block_size 的整数倍
    #    vLLM 只能命中向下取整到块的部分；SGLang page=1 能命中到 token
    v2 = VllmBlockCache(block_size=16, capacity_blocks=999)
    s2 = SglangRadixCache(page_size=1, capacity_tokens=999999)
    a = list(range(100))
    b = list(range(100))[:70] + [777] * 30      # 共享 70 个 token（70 不是 16 的倍数）
    v2.process(a); s2.process(a)
    hv, hs = v2.process(b), s2.process(b)
    if hv != 64:
        print(f"FAIL vLLM 应命中 64（70 向下取整到 16 的倍数），实得 {hv}"); ok = False
    if hs != 70:
        print(f"FAIL SGLang page=1 应命中 70，实得 {hs}"); ok = False

    # 4) page_size 拉到 16 后，SGLang 也退化成块粒度 —— 证明"树 ≠ 自动 token 粒度"
    s3 = SglangRadixCache(page_size=16, capacity_tokens=999999)
    s3.process(a)
    hs3 = s3.process(b)
    if hs3 != 64:
        print(f"FAIL page=16 时 SGLang 应同样只命中 64，实得 {hs3}"); ok = False

    # 5) vLLM 细粒度模式应当把那 6 个 token 找回来
    v3 = VllmBlockCache(block_size=16, capacity_blocks=999, hash_block_size=2)
    v3.process(a)
    hv3 = v3.process(b)
    if hv3 != 70:
        print(f"FAIL 细粒度(hash 单元 2)应命中 70，实得 {hv3}"); ok = False

    # 6) 无共享前缀时两边都应接近 0
    v4 = VllmBlockCache(block_size=16, capacity_blocks=999)
    s4 = SglangRadixCache(page_size=1, capacity_tokens=999999)
    for t in ([1] * 50, [2] * 50, [3] * 50):
        v4.process(t); s4.process(t)
    if v4.st.hit_tokens != 0 or s4.st.hit_tokens != 0:
        print(f"FAIL 无共享应 0 命中 -> vllm={v4.st.hit_tokens} sglang={s4.st.hit_tokens}")
        ok = False

    # 7) 驱逐要真的发生，且不把树/表清空
    s5 = SglangRadixCache(page_size=1, capacity_tokens=200)
    for i in range(20):
        s5.process([i * 1000 + j for j in range(100)])
    if s5.st.evictions == 0:
        print("FAIL 容量收紧后应发生驱逐"); ok = False
    if s5.n_tokens > 200:
        print(f"FAIL 驱逐后仍超容量 -> {s5.n_tokens}"); ok = False

    # 8) 随机数发生器可复现
    if [Rng(7).next() for _ in range(3)] != [Rng(7).next() for _ in range(3)]:
        print("FAIL Rng 不可复现"); ok = False

    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
