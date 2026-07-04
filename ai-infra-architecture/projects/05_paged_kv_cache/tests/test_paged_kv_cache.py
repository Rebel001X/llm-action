"""
test_paged_kv_cache.py —— 分页 KV Cache 管理器的正确性金标准(pytest)。

覆盖 5 类不变量(invariants):
  1) 分配/释放不泄漏:任何操作序列后,释放干净则空闲块回到满额、引用计数全 0。
  2) 不越界:块内写不超过 block_size;读回的 token 与写入完全一致。
  3) 前缀共享确实省块:相同前缀的多条序列用的物理块 < 各自独立所需之和。
  4) 碎片率计算正确:构造已知场景,断言精确数值。
  5) 写时复制 COW 正确:fork 后各写各的,内容互不污染,且不泄漏。

运行:python -m pytest -q
"""
import os
import sys

import pytest

# 让 tests/ 能 import 到上一层的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paged_kv_cache import (  # noqa: E402
    BlockAllocator,
    OutOfMemory,
    PagedKVCacheManager,
    EMPTY,
)


# ────────────────────────────── 工具 ──────────────────────────────
def fresh(num_blocks=64, block_size=4, sharing=True):
    return PagedKVCacheManager(num_blocks, block_size, enable_prefix_sharing=sharing)


def assert_no_leak(mgr: PagedKVCacheManager):
    """释放完所有序列后:引用计数全 0、空闲块满额、前缀缓存清空。"""
    assert mgr.alloc.num_free == mgr.num_blocks
    assert mgr.num_used_blocks() == 0
    assert all(rc == 0 for rc in mgr.alloc.ref_count)
    assert mgr.prefix_cache == {}
    assert mgr.hash_of_block == {}


# ────────────────────────── 1. 分配器基础 ──────────────────────────
def test_allocator_allocate_free_roundtrip():
    a = BlockAllocator(4)
    assert a.num_free == 4 and a.num_used == 0
    blocks = [a.allocate() for _ in range(4)]
    assert len(set(blocks)) == 4          # 分出的块互不重复
    assert a.num_free == 0 and a.num_used == 4
    for b in blocks:
        a.free(b)
    assert a.num_free == 4 and all(rc == 0 for rc in a.ref_count)


def test_allocator_out_of_memory_raises():
    a = BlockAllocator(2)
    a.allocate(); a.allocate()
    with pytest.raises(OutOfMemory):
        a.allocate()


def test_allocator_refcount_shared_block_not_freed_until_zero():
    a = BlockAllocator(2)
    b = a.allocate()
    a.incref(b)                            # 两个「引用者」
    assert a.ref_count[b] == 2
    assert a.free(b) is False              # 第一次 free:仍被引用,不回收
    assert a.num_free == 1
    assert a.free(b) is True               # 第二次 free:归 0,回收
    assert a.num_free == 2


# ────────────────────────── 2. 不越界 / 读回一致 ──────────────────────────
def test_add_sequence_layout_and_readback():
    mgr = fresh(block_size=4)
    toks = list(range(10))                 # 10 个 token → 2 满块 + 1 尾块(2)
    mgr.add_sequence(1, toks)
    seq = mgr.sequences[1]
    assert len(seq.block_table) == 3       # ceil(10/4) = 3 块
    assert mgr.read_tokens(1) == toks      # gather 回来完全一致
    # 尾块只填了 2 个,其余是 EMPTY(证明没越界写脏)
    tail = seq.block_table[-1]
    assert mgr.filled[tail] == 2
    assert mgr.store[tail, 2] == EMPTY and mgr.store[tail, 3] == EMPTY


def test_append_never_overflows_block():
    mgr = fresh(block_size=4)
    mgr.add_sequence(1, [0, 1, 2])         # 尾块填了 3
    for t in range(3, 20):
        mgr.append_token(1, t)             # 一路 decode
    assert mgr.read_tokens(1) == list(range(20))
    # 每个在用块 filled ≤ block_size(不越界)
    for b in range(mgr.num_blocks):
        assert mgr.filled[b] <= mgr.block_size


def test_append_allocates_new_block_on_boundary():
    mgr = fresh(block_size=4)
    mgr.add_sequence(1, [0, 1, 2, 3])      # 正好 1 满块,无尾块
    assert len(mgr.sequences[1].block_table) == 1
    mgr.append_token(1, 4)                 # 触发新块分配
    assert len(mgr.sequences[1].block_table) == 2
    assert mgr.read_tokens(1) == [0, 1, 2, 3, 4]


# ────────────────────────── 3. 前缀共享省块 ──────────────────────────
def test_prefix_sharing_saves_blocks():
    mgr = fresh(block_size=4, sharing=True)
    prompt = list(range(16))               # 4 个满块,完全相同的前缀
    mgr.add_sequence(1, prompt)
    used_after_first = mgr.num_used_blocks()
    assert used_after_first == 4
    # 第二条一模一样的前缀:应全部共享,不再新增物理块
    mgr.add_sequence(2, prompt)
    assert mgr.num_used_blocks() == used_after_first          # 一块都没多用
    assert mgr.blocks_saved_by_prefix == 4                    # 省了 4 块
    # 内容正确:两条读回都对
    assert mgr.read_tokens(1) == prompt
    assert mgr.read_tokens(2) == prompt


def test_prefix_sharing_partial_common_prefix():
    mgr = fresh(block_size=4, sharing=True)
    a = list(range(12))                    # 3 满块
    b = list(range(8)) + [99, 98, 97, 96]  # 前 2 块相同,第 3 块不同
    mgr.add_sequence(1, a)                 # 用 3 块
    mgr.add_sequence(2, b)                 # 前 2 块共享,只需新分配第 3 块
    assert mgr.num_used_blocks() == 4      # 3 + 1(而非 3 + 3)
    assert mgr.blocks_saved_by_prefix == 2
    assert mgr.read_tokens(2) == b         # 分叉后内容仍正确


def test_sharing_off_uses_more_blocks_than_sharing_on():
    prompt = list(range(16))
    on = fresh(sharing=True); on.add_sequence(1, prompt); on.add_sequence(2, prompt)
    off = fresh(sharing=False); off.add_sequence(1, prompt); off.add_sequence(2, prompt)
    assert off.num_used_blocks() > on.num_used_blocks()
    assert off.num_used_blocks() == 8 and on.num_used_blocks() == 4


# ────────────────────────── 4. 碎片率计算 ──────────────────────────
def test_fragmentation_exact_values():
    mgr = fresh(block_size=4, sharing=False)
    # 一条长度 10 的序列:3 块 = 12 槽,写了 10 → 碎片 2/12
    mgr.add_sequence(1, list(range(10)))
    assert mgr.num_used_blocks() == 3
    assert mgr.stored_slots() == 10
    assert mgr.internal_fragmentation() == pytest.approx(2 / 12)
    assert mgr.utilization() == pytest.approx(10 / 12)


def test_fragmentation_full_blocks_zero_frag():
    mgr = fresh(block_size=4, sharing=False)
    mgr.add_sequence(1, list(range(8)))    # 正好 2 满块,零碎片
    assert mgr.internal_fragmentation() == 0.0
    assert mgr.utilization() == 1.0


def test_paged_beats_naive_utilization():
    # 变长序列下,分页利用率应远高于「连续预留 max_len」的朴素方案
    mgr = fresh(num_blocks=256, block_size=8, sharing=False)
    lens = [5, 9, 13, 40, 100, 7]
    for i, n in enumerate(lens):
        mgr.add_sequence(i, list(range(n)))
    total_tokens = sum(lens)
    naive = PagedKVCacheManager.naive_reserved_slots(len(lens), max(lens))
    naive_util = total_tokens / naive
    assert mgr.utilization() > 0.85        # 分页:碎片 < 1 块/序列
    assert naive_util < 0.6                # 朴素:被最长序列拖成大碎片
    assert mgr.utilization() > naive_util


# ────────────────────────── 5. 写时复制 COW ──────────────────────────
def test_fork_shares_then_cow_diverges():
    mgr = fresh(block_size=4, sharing=True)
    mgr.add_sequence(1, [1, 2, 3])         # 尾块填 3(未满,可写)
    used_before = mgr.num_used_blocks()
    mgr.fork(1, 2)                         # 共享全部块,零新增
    assert mgr.num_used_blocks() == used_before
    tail1 = mgr.sequences[1].block_table[-1]
    tail2 = mgr.sequences[2].block_table[-1]
    assert tail1 == tail2                  # fork 后尾块是同一物理块
    assert mgr.alloc.ref_count[tail1] == 2

    # 父序列写 → 触发 COW,拿到私有尾块
    mgr.append_token(1, 100)
    assert mgr.cow_count == 1
    assert mgr.sequences[1].block_table[-1] != tail2   # 父已分叉到新块
    assert mgr.sequences[2].block_table[-1] == tail2   # 子仍指原块

    # 子序列写不同的 token → 各自独立,互不污染
    mgr.append_token(2, 200)
    assert mgr.read_tokens(1) == [1, 2, 3, 100]
    assert mgr.read_tokens(2) == [1, 2, 3, 200]


def test_fork_full_prefix_no_cow_on_shared_full_block():
    # fork 后若只在各自新块上写(不碰共享满块),不应发生 COW
    mgr = fresh(block_size=4, sharing=True)
    mgr.add_sequence(1, [0, 1, 2, 3])      # 正好 1 满块
    mgr.fork(1, 2)
    mgr.append_token(1, 4)                 # 满块 → 直接分配新块,而非 COW
    mgr.append_token(2, 5)
    assert mgr.cow_count == 0
    assert mgr.read_tokens(1) == [0, 1, 2, 3, 4]
    assert mgr.read_tokens(2) == [0, 1, 2, 3, 5]


# ────────────────────────── 6. 不泄漏(综合) ──────────────────────────
def test_free_shared_prefix_no_premature_recycle():
    mgr = fresh(block_size=4, sharing=True)
    prompt = list(range(8))
    mgr.add_sequence(1, prompt)
    mgr.add_sequence(2, prompt)            # 共享 2 块
    shared = list(mgr.sequences[1].block_table)
    mgr.free_sequence(1)                   # 释放其一
    # 共享块仍被 seq2 引用,不能回收
    for b in shared:
        assert mgr.alloc.ref_count[b] == 1
    assert mgr.read_tokens(2) == prompt    # seq2 内容完好
    mgr.free_sequence(2)
    assert_no_leak(mgr)


def test_no_leak_after_mixed_workload():
    mgr = fresh(num_blocks=128, block_size=4, sharing=True)
    sys_prompt = list(range(20))
    # 5 条共享系统提示 + 各自 decode 若干步 + fork 一些
    for sid in range(5):
        mgr.add_sequence(sid, sys_prompt + [1000 + sid])
        for t in range(sid + 3):
            mgr.append_token(sid, 5000 + t)
    mgr.fork(0, 100)
    mgr.fork(1, 101)
    for t in range(4):
        mgr.append_token(100, 7000 + t)
        mgr.append_token(101, 8000 + t)
    # 全部释放
    for sid in list(mgr.sequences.keys()):
        mgr.free_sequence(sid)
    assert_no_leak(mgr)


def test_out_of_memory_when_pool_exhausted():
    mgr = fresh(num_blocks=3, block_size=4, sharing=False)
    mgr.add_sequence(1, list(range(12)))   # 用光 3 块
    with pytest.raises(OutOfMemory):
        mgr.append_token(1, 99)            # 尾块满 → 想分配第 4 块 → OOM
