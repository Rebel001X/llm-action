"""paged.py —— 分页 KV 缓存：把「一条请求的 KV 必须物理连续」这个假设拆掉。

为什么需要它（`minigpt.py` 里那个朴素 `KVCache` 的三宗罪）：
  1. 必须按 max_seq 预分配 → 一条只用 20 token 的请求也占着 2048 token 的位置（**内碎片**）；
  2. 一条请求的 KV 必须物理连续 → 显存里明明有空位却凑不出连续段（**外碎片**）；
  3. 请求之间完全不共享 → 同一段 system prompt 被算了 N 遍（**无法共享**）。

分页的做法和操作系统虚拟内存一模一样：
  **逻辑上连续（token 0..n）、物理上分散（若干固定大小的块）**，中间用一张**块表**映射。
  于是：按需分配（碎片降到最多一个块）、可以让两条请求的块表指向同一个物理块（共享）。

这个文件只做**机制**，不做性能：块是 numpy 数组，分配器是一个 Python list。
但它足以让「分页不改变输出」这条正确性命题可被测试。
"""
from __future__ import annotations

import numpy as np

from minigpt import Config


class BlockAllocator:
    """最小块分配器。空闲块用一个栈管着，分配 O(1)、释放 O(1)。

    真实引擎（vLLM）在这里要复杂得多：块带 hash、带引用计数、
    释放时按「是否被缓存」分两条队列。那些是**前缀缓存**要用的，
    见 `engine.py` 的 `PrefixCache`。这里先只做最朴素的版本。
    """

    def __init__(self, n_blocks: int):
        self.n_blocks = n_blocks
        self.free: list[int] = list(range(n_blocks - 1, -1, -1))  # 栈顶是小编号
        self.ref: dict[int, int] = {}

    def alloc(self) -> int:
        if not self.free:
            raise MemoryError("没有空闲块了 —— 真实引擎在这里会触发抢占/回退")
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def incref(self, b: int) -> None:
        self.ref[b] = self.ref.get(b, 0) + 1

    def decref(self, b: int) -> None:
        """引用计数归零才真的还回去 —— 这是**块共享**能成立的前提。"""
        self.ref[b] -= 1
        if self.ref[b] == 0:
            del self.ref[b]
            self.free.append(b)

    @property
    def n_free(self) -> int:
        return len(self.free)

    @property
    def n_used(self) -> int:
        return self.n_blocks - len(self.free)


class PagedKVCache:
    """物理 KV 池 + 每条请求一张块表。

    物理布局：`k[layer]` 形状 (n_blocks, block_size, n_head, d_head)。
    **为什么把 block 放最外维**：这样一个块在内存里是连续的，
    真实引擎的 attention kernel 就能按块做 gather。这里用不上，但布局要对得上。
    """

    def __init__(self, cfg: Config, n_blocks: int = 256, block_size: int = 16):
        self.cfg, self.block_size = cfg, block_size
        self.alloc = BlockAllocator(n_blocks)
        shape = (n_blocks, block_size, cfg.n_head, cfg.d_head)
        self.k = [np.zeros(shape, np.float32) for _ in range(cfg.n_layer)]
        self.v = [np.zeros(shape, np.float32) for _ in range(cfg.n_layer)]

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k) + sum(a.nbytes for a in self.v)


class SeqTable:
    """一条请求的块表：逻辑位置 → (物理块号, 块内偏移)。"""

    def __init__(self, pool: PagedKVCache):
        self.pool = pool
        self.blocks: list[int] = []
        self.length = 0        # 已提交的 token 数
        self._pending = 0      # 本步正在写、尚未提交的 token 数

    def _ensure(self, n_tokens: int) -> None:
        """按需扩容 —— **只在写满当前块时才要新块**，这就是内碎片被压到 ≤1 块的原因。"""
        need = (self.length + n_tokens + self.pool.block_size - 1) // self.pool.block_size
        while len(self.blocks) < need:
            self.blocks.append(self.pool.alloc.alloc())

    def begin(self, n_new: int) -> None:
        """本步要写 n_new 个 token：先按需扩容，并把它们标成"未提交"。

        为什么要分 begin/commit 两步（第一版我没分，直接在最后一层推进 length，
        结果前面几层 `gather` 拿到的长度是旧的，attention 掩码形状对不上就崩了）：
        **每一层都要写自己的 K/V，但"序列长了 n 个"这件事只该发生一次。**
        """
        self._ensure(n_new)
        self._pending = n_new

    def commit(self) -> None:
        self.length += self._pending
        self._pending = 0

    def append(self, layer: int, kn: np.ndarray, vn: np.ndarray) -> None:
        """写入本步 T 个新 token 在第 layer 层的 K/V。kn/vn 形状 (n_head, T, d_head)。"""
        T = kn.shape[1]
        bs = self.pool.block_size
        for i in range(T):
            pos = self.length + i
            b, off = self.blocks[pos // bs], pos % bs
            self.pool.k[layer][b, off] = kn[:, i, :]
            self.pool.v[layer][b, off] = vn[:, i, :]

    def gather(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """把散在各块里的 K/V 收成 (n_head, length, d_head)。

        ⚠️ 真实引擎**不做这一步** —— 它把块表直接交给 attention kernel，
        由 kernel 边算边取。这里 gather 是为了让教学代码能复用 numpy 的矩阵乘，
        **代价是多一次拷贝**，所以本文件不谈性能。
        """
        bs, L = self.pool.block_size, self.length + self._pending
        cfg = self.pool.cfg
        ks = np.empty((cfg.n_head, L, cfg.d_head), np.float32)
        vs = np.empty((cfg.n_head, L, cfg.d_head), np.float32)
        for pos in range(L):
            b, off = self.blocks[pos // bs], pos % bs
            ks[:, pos, :] = self.pool.k[layer][b, off]
            vs[:, pos, :] = self.pool.v[layer][b, off]
        return ks, vs

    def free(self) -> None:
        for b in self.blocks:
            self.pool.alloc.decref(b)
        self.blocks.clear()
        self.length = self._pending = 0

    def fragmentation(self) -> dict:
        """内碎片：分配了但没用上的槽位。**分页的核心卖点就是把它压到 < 1 块。**"""
        cap = len(self.blocks) * self.pool.block_size
        return {"capacity": cap, "used": self.length, "internal_waste": cap - self.length}


def selftest() -> int:
    import minigpt as M
    ok = True
    cfg = Config(n_layer=2, n_head=2, d_model=32, vocab=64, max_seq=64, seed=1)
    w = M.init_weights(cfg)
    prompt = [3, 14, 15, 62, 6, 35, 8, 9, 7, 12, 41, 5]

    # 1) **本文件最重要的命题：分页不许改变输出。**
    #    用块大小 4（故意小，逼出多块）跑一遍，和连续版逐位比。
    pool = PagedKVCache(cfg, n_blocks=64, block_size=4)
    tab = SeqTable(pool)
    got = _forward_paged(cfg, w, prompt, tab)
    want = M.forward_naive(cfg, w, prompt)
    if not np.allclose(got, want, atol=1e-4):
        print(f"FAIL 分页改变了输出：最大差 {np.abs(got - want).max():.3e}"); ok = False

    # 2) 逐 token 增量喂，结果也必须一致
    tab2 = SeqTable(PagedKVCache(cfg, n_blocks=64, block_size=4))
    for t in prompt:
        got2 = _forward_paged(cfg, w, [t], tab2)
    if not np.allclose(got2, want, atol=1e-4):
        print(f"FAIL 分页增量不一致：最大差 {np.abs(got2 - want).max():.3e}"); ok = False

    # 3) 块大小不该影响结果（1/2/4/8/16 都要一致）
    for bsz in (1, 2, 8, 16):
        t = SeqTable(PagedKVCache(cfg, n_blocks=128, block_size=bsz))
        g = _forward_paged(cfg, w, prompt, t)
        if not np.allclose(g, want, atol=1e-4):
            print(f"FAIL block_size={bsz} 改变了输出"); ok = False

    # 4) 内碎片必须 < 一个块 —— 这是分页的核心卖点
    f = tab.fragmentation()
    if f["internal_waste"] >= pool.block_size:
        print(f"FAIL 内碎片应小于一个块 -> {f}"); ok = False

    # 5) 释放后块要全部回到空闲池（否则就是泄漏）
    p2 = PagedKVCache(cfg, n_blocks=16, block_size=4)
    t2 = SeqTable(p2)
    _forward_paged(cfg, w, prompt, t2)
    used = p2.alloc.n_used
    t2.free()
    if p2.alloc.n_used != 0:
        print(f"FAIL 释放后应归零，实际仍占 {p2.alloc.n_used}（之前用了 {used}）"); ok = False

    # 6) 块用尽必须**明确报错**，不许静默乱写
    p3 = PagedKVCache(cfg, n_blocks=1, block_size=2)
    t3 = SeqTable(p3)
    try:
        _forward_paged(cfg, w, prompt, t3)
        print("FAIL 块用尽时应抛 MemoryError"); ok = False
    except MemoryError:
        pass

    # 7) 引用计数：两张表指向同一个块时，只有都释放才归还
    p4 = PagedKVCache(cfg, n_blocks=8, block_size=4)
    b = p4.alloc.alloc()
    p4.alloc.incref(b)
    p4.alloc.decref(b)
    if p4.alloc.n_used != 1:
        print("FAIL 引用计数未归零就不该归还"); ok = False
    p4.alloc.decref(b)
    if p4.alloc.n_used != 0:
        print("FAIL 引用计数归零后应归还"); ok = False

    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _forward_paged(cfg: Config, w: dict, new_tokens: list[int],
                   tab: SeqTable) -> np.ndarray:
    """和 `minigpt.forward_cached` 逻辑相同，只是 K/V 走块表。

    **对比这两个函数是本篇教学的核心**：算法一个字没变，
    变的只是「历史 K/V 从哪儿取」。
    """
    import minigpt as M
    T = len(new_tokens)
    p0 = tab.length
    tab.begin(T)                                   # 先占位，再逐层写
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

    tab.commit()                                   # 全部层写完，序列才真的变长
    x = M.layernorm(x, w["lnf_g"], w["lnf_b"])
    return x[-1] @ w["wte"].T


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    cfg = Config()
    print("分页 KV：内碎片随块大小的变化（序列长度 100）")
    print(f"{'block_size':>12}{'块数':>8}{'容量':>8}{'浪费':>8}{'浪费率':>9}")
    for bsz in (1, 4, 16, 64, 256):
        n = (100 + bsz - 1) // bsz
        cap = n * bsz
        print(f"{bsz:>12}{n:>8}{cap:>8}{cap - 100:>8}{(cap - 100) / cap * 100:>8.1f}%")
    print("\n块越大 → 块表越短、kernel 越好写，但内碎片越大。这就是 block_size 的取舍。")
