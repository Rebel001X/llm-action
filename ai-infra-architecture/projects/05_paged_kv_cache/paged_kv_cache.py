"""
paged_kv_cache.py —— PagedAttention 式「分页 KV Cache 管理器」核心实现

对应架构文档 ../../07_KVCache深入_PagedAttention_前缀缓存_量化压缩.md。

一句话:把 KV Cache 从「每序列连续预留 max_seq_len」改成「操作系统式的分页管理」——
固定大小的 block(页)+ 每序列一张 block table(页表)+ 引用计数(ref count)。
由此得到三个杀手级收益:
  1) 消除**外部碎片**(external fragmentation):任何空闲 block 都能被任何序列用。
  2) 把**内部碎片**(internal fragmentation)锁死在「每序列最多浪费 1 个 block」。
  3) **前缀共享**(prefix sharing):相同前缀的序列共享同一批物理 block,
     真正要写时才复制 —— **写时复制 copy-on-write(COW)**。

本文件不依赖 GPU,用一个 numpy int 数组当「物理 KV 显存」的替身
(每个 token 存一个 id,代表它那一格 KV 向量),因此 COW / 共享 / 碎片
全部是**可验证的真实行为**,而不是打印几个数字糊弄人。

术语对照(面试要能中英互译):
  block / page            —— 页(固定大小的 KV 存储单元,单位是 token 数)
  block table / page table—— 页表(逻辑块号 → 物理块号)
  block allocator         —— 块分配器(管空闲链表 + 引用计数)
  ref count               —— 引用计数(一个物理块被几个序列引用)
  copy-on-write (COW)     —— 写时复制(共享块被写前先拷贝出私有副本)
  internal fragmentation  —— 内部碎片(块内没填满的空槽)
  external fragmentation  —— 外部碎片(空闲空间总量够但不连续,无法满足请求)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import numpy as np

# 一格 KV 尚未写入时的填充值(-1 表示空槽,方便断言「没越界写脏数据」)
EMPTY = -1


class OutOfMemory(RuntimeError):
    """物理 block 耗尽时抛出,对应真实引擎的『KV cache 满 → 触发抢占/换出』。"""


# ────────────────────────────────────────────────────────────────────────────
# 1. BlockAllocator —— 块分配器:空闲链表 + 引用计数
# ────────────────────────────────────────────────────────────────────────────
class BlockAllocator:
    """
    管理 `num_blocks` 个固定大小物理块的「分配 / 引用 / 释放」。

    数据结构:
      · self._free      —— 空闲块号栈(free list),O(1) 出入
      · self.ref_count  —— 每个物理块的引用计数;>0 表示在用,==0 表示空闲
    这就是操作系统物理页帧分配器的极简版:引用计数支撑「多序列共享同一物理页」。
    """

    def __init__(self, num_blocks: int):
        assert num_blocks > 0
        self.num_blocks = num_blocks
        # 用列表当栈;初始全部空闲(倒序 push,pop 时从小号开始拿,输出更直观)
        self._free: List[int] = list(range(num_blocks - 1, -1, -1))
        self.ref_count: List[int] = [0] * num_blocks

    # 分配一个全新的物理块(引用计数 0 → 1)
    def allocate(self) -> int:
        if not self._free:
            raise OutOfMemory(
                f"物理块耗尽:共 {self.num_blocks} 块已全部占用(真实引擎此时会抢占/换出)"
            )
        b = self._free.pop()
        assert self.ref_count[b] == 0, "拿到的空闲块引用计数必须为 0(否则分配器已损坏)"
        self.ref_count[b] = 1
        return b

    # 增加引用(共享 / fork 时用):某物理块又被一个序列引用
    def incref(self, block: int) -> None:
        assert self.ref_count[block] > 0, "只能对『在用』的块 incref"
        self.ref_count[block] += 1

    # 释放一个引用;引用计数归 0 时真正回收到空闲链表。返回是否『刚刚变空闲』
    def free(self, block: int) -> bool:
        assert self.ref_count[block] > 0, "重复释放 / 释放未分配块 → 分配器 bug"
        self.ref_count[block] -= 1
        if self.ref_count[block] == 0:
            self._free.append(block)
            return True
        return False

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free


# ────────────────────────────────────────────────────────────────────────────
# 2. Sequence —— 一条请求的逻辑视图:token 列表 + block table(页表)
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class Sequence:
    seq_id: int
    token_ids: List[int] = field(default_factory=list)   # 逻辑 token 序列
    block_table: List[int] = field(default_factory=list) # 逻辑块 i → 物理块号
    # 到目前为止「所有已填满 block」的链式哈希(用于前缀共享的块级匹配)
    prefix_hash: int = 0

    def __len__(self) -> int:
        return len(self.token_ids)


# ────────────────────────────────────────────────────────────────────────────
# 3. PagedKVCacheManager —— 顶层管理器
# ────────────────────────────────────────────────────────────────────────────
class PagedKVCacheManager:
    """
    分页 KV Cache 管理器。对外暴露 5 个核心动作:
      · add_sequence(seq_id, prompt_ids)  加入一条请求(prefill:填 prompt 的 KV)
      · append_token(seq_id, token_id)    decode:追加 1 个 token 的 KV
      · fork(parent, child)               分裂(并行采样/beam search:先共享,后 COW)
      · free_sequence(seq_id)             释放整条序列的 KV(引用计数递减)
      · 各种 stats():利用率 / 碎片率 / 共享节省

    物理存储 self.store 是 (num_blocks, block_size) 的 int64 数组,当作 KV 显存的替身。
    """

    def __init__(self, num_blocks: int, block_size: int,
                 enable_prefix_sharing: bool = True):
        assert block_size > 0 and num_blocks > 0
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_sharing = enable_prefix_sharing

        self.alloc = BlockAllocator(num_blocks)
        # 物理 KV 显存替身:每格存一个 token id,-1 表示空槽
        self.store = np.full((num_blocks, block_size), EMPTY, dtype=np.int64)
        # 每个物理块已填的槽数(≤ block_size)。满块 == block_size,尾块可能不满。
        self.filled: List[int] = [0] * num_blocks

        # 前缀缓存:块级链式哈希 → 物理块号(仅缓存『已填满』的块)
        self.prefix_cache: Dict[int, int] = {}
        self.hash_of_block: Dict[int, int] = {}   # 反查:物理块 → 其哈希(释放时清缓存)

        self.sequences: Dict[int, Sequence] = {}

        # 统计计数器
        self.blocks_saved_by_prefix = 0   # 因前缀命中而少分配的块数
        self.cow_count = 0                # 发生写时复制的次数

    # ── 内部:块级链式哈希 ──────────────────────────────────────────────
    def _chain_hash(self, prev_hash: int, block_tokens: Tuple[int, ...]) -> int:
        """
        vLLM 式前缀哈希:一个块的身份 = f(它前面所有 token 的哈希, 本块 token)。
        这样只有**从头到本块完全相同**的两条序列,才会得到相同哈希 → 才允许共享。
        避免了「中间某块碰巧相同就错误共享」的语义 bug。
        """
        return hash((prev_hash, block_tokens))

    # ── 内部:写一个 token 到某物理块的下一个空槽 ──────────────────────
    def _write_slot(self, pblock: int, token_id: int) -> None:
        slot = self.filled[pblock]
        assert slot < self.block_size, "越界写:块已满还往里写 → block table 逻辑错误"
        self.store[pblock, slot] = token_id
        self.filled[pblock] = slot + 1

    # ── add_sequence:prefill 阶段,一次性把 prompt 的 KV 装进分页 ────────
    def add_sequence(self, seq_id: int, prompt_ids: Seq[int]) -> Sequence:
        """
        把一条新请求的 prompt 写入分页 KV。**逐『满块』尝试前缀共享**:
        对每个已填满的块算链式哈希,若前缀缓存命中 → 直接引用现有物理块(省显存);
        否则分配新块、写入、登记进缓存。最后不满的尾块永远是私有的(不参与共享)。
        """
        assert seq_id not in self.sequences, f"seq_id {seq_id} 已存在"
        tokens = [int(t) for t in prompt_ids]
        seq = Sequence(seq_id)
        self.sequences[seq_id] = seq

        bs = self.block_size
        n_full = len(tokens) // bs
        prev = 0
        for i in range(n_full):
            blk = tuple(tokens[i * bs:(i + 1) * bs])
            prev = self._chain_hash(prev, blk)
            if self.enable_prefix_sharing and prev in self.prefix_cache:
                # 前缀命中:共享现有物理块,只需 +1 引用,一个字节都不用新写
                pblock = self.prefix_cache[prev]
                self.alloc.incref(pblock)
                self.blocks_saved_by_prefix += 1
            else:
                pblock = self.alloc.allocate()
                self.store[pblock, :] = blk
                self.filled[pblock] = bs
                if self.enable_prefix_sharing:
                    self.prefix_cache[prev] = pblock
                    self.hash_of_block[pblock] = prev
            seq.block_table.append(pblock)
            seq.token_ids.extend(blk)
        seq.prefix_hash = prev  # 记录『所有满块』的链式哈希,供 decode 续算

        # 处理不满的尾部(私有块,不共享,因为还会被继续写)
        rem = tokens[n_full * bs:]
        if rem:
            pblock = self.alloc.allocate()
            for t in rem:
                self._write_slot(pblock, t)
            seq.block_table.append(pblock)
            seq.token_ids.extend(rem)
        return seq

    # ── append_token:decode 阶段,追加 1 个 token 的 KV(含 COW) ────────
    def append_token(self, seq_id: int, token_id: int) -> None:
        """
        给序列追加一个新 token 的 KV。三种情况:
          A) 无块 / 尾块已满 → 分配一个新块,写到第 0 槽。
          B) 尾块未满但被共享(ref>1)→ **写时复制 COW**:拷一份私有副本再写。
          C) 尾块未满且独占 → 直接写。
        写满一个块后,把它登记进前缀缓存,供后续同前缀的请求共享。
        """
        seq = self.sequences[seq_id]
        token_id = int(token_id)
        bs = self.block_size
        last = seq.block_table[-1] if seq.block_table else None

        if last is None or self.filled[last] == bs:
            # A:需要一个新块
            last = self.alloc.allocate()
            seq.block_table.append(last)
        elif self.alloc.ref_count[last] > 1:
            # B:尾块是共享的,写前必须复制,避免污染其他序列(这就是 COW)
            newb = self.alloc.allocate()
            self.store[newb, :] = self.store[last, :]
            self.filled[newb] = self.filled[last]
            self.alloc.free(last)            # 旧块引用 -1(仍被别的序列引用,不回收)
            seq.block_table[-1] = newb
            last = newb
            self.cow_count += 1

        self._write_slot(last, token_id)
        seq.token_ids.append(token_id)

        # 若本块刚好写满 → 算它的链式哈希,登记进前缀缓存
        if self.filled[last] == bs:
            blk = tuple(int(x) for x in self.store[last, :].tolist())
            h = self._chain_hash(seq.prefix_hash, blk)
            seq.prefix_hash = h
            if self.enable_prefix_sharing and h not in self.prefix_cache:
                self.prefix_cache[h] = last
                self.hash_of_block[last] = h

    # ── fork:并行采样 / beam search,先整段共享,后各自 COW ──────────────
    def fork(self, parent_id: int, child_id: int) -> Sequence:
        """
        从 parent 分裂出 child:**共享 parent 的全部物理块**(包括不满的尾块),
        只增加引用计数,零拷贝。之后谁先写尾块,谁就触发 COW 拿到私有副本。
        这正是「一个 prompt 采样 N 条候选」时省显存的关键(N 条只存 1 份 prompt)。
        """
        assert child_id not in self.sequences
        p = self.sequences[parent_id]
        child = Sequence(child_id, list(p.token_ids), list(p.block_table), p.prefix_hash)
        for b in child.block_table:
            self.alloc.incref(b)
        self.sequences[child_id] = child
        return child

    # ── free_sequence:释放整条序列 ────────────────────────────────────
    def free_sequence(self, seq_id: int) -> None:
        """引用计数逐块 -1;归 0 的块回收到空闲链表,并从前缀缓存中逐出、清空内容。"""
        seq = self.sequences.pop(seq_id)
        for b in seq.block_table:
            became_free = self.alloc.free(b)
            if became_free:
                h = self.hash_of_block.pop(b, None)
                if h is not None and self.prefix_cache.get(h) == b:
                    del self.prefix_cache[h]
                self.filled[b] = 0
                self.store[b, :] = EMPTY

    # ── 读取:把一条序列的逻辑 token 从物理块「gather」回来 ────────────────
    def read_tokens(self, seq_id: int) -> List[int]:
        """
        按 block table 把散落在各物理块里的 token 拼回逻辑顺序。
        用于测试『COW 后各序列内容互不污染』——这才证明分页寻址是对的。
        """
        seq = self.sequences[seq_id]
        out: List[int] = []
        bs = self.block_size
        remaining = len(seq.token_ids)
        for b in seq.block_table:
            take = min(bs, remaining)
            out.extend(int(x) for x in self.store[b, :take].tolist())
            remaining -= take
        return out

    # ── 统计 ──────────────────────────────────────────────────────────
    def stored_slots(self) -> int:
        """所有『在用』物理块里实际写了 token 的槽数之和(共享块只算一次)。"""
        return sum(self.filled[b] for b in range(self.num_blocks)
                   if self.alloc.ref_count[b] > 0)

    def num_used_blocks(self) -> int:
        return self.alloc.num_used

    def pool_occupancy(self) -> float:
        """整个 block 池被占用的比例 = 已分配块 / 总块。反映『显存压力』。"""
        return self.num_used_blocks() / self.num_blocks

    def internal_fragmentation(self) -> float:
        """
        内部碎片率 = 1 - 已写槽数 / 已分配槽数。
        分页方案下每序列最多浪费 1 个尾块 → 该值天然很小(块越小越小)。
        """
        used = self.num_used_blocks()
        if used == 0:
            return 0.0
        allocated_slots = used * self.block_size
        return 1.0 - self.stored_slots() / allocated_slots

    def utilization(self) -> float:
        """有效利用率 = 已写槽数 / 已分配槽数(= 1 - 内部碎片率)。"""
        used = self.num_used_blocks()
        if used == 0:
            return 1.0
        return self.stored_slots() / (used * self.block_size)

    # 一条序列若用『连续预留 max_seq_len』的朴素方案,需要多少槽
    @staticmethod
    def naive_reserved_slots(num_seqs: int, max_seq_len: int) -> int:
        """朴素连续分配:每条序列都必须按最坏长度 max_seq_len 预留一整段连续显存。"""
        return num_seqs * max_seq_len

    def snapshot(self) -> dict:
        """打一份可读的状态快照,方便 demo 打印 / 调试。"""
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "used_blocks": self.num_used_blocks(),
            "free_blocks": self.alloc.num_free,
            "stored_slots": self.stored_slots(),
            "pool_occupancy": round(self.pool_occupancy(), 4),
            "internal_fragmentation": round(self.internal_fragmentation(), 4),
            "utilization": round(self.utilization(), 4),
            "blocks_saved_by_prefix": self.blocks_saved_by_prefix,
            "cow_count": self.cow_count,
            "num_sequences": len(self.sequences),
        }
