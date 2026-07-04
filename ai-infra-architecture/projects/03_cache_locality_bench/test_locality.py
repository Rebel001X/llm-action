"""
test_locality.py —— 验证实测抓住了缓存/局部性效应(阈值留足余量,抗噪声)。
运行:python -m pytest -q
"""
from locality_bench import (
    contiguous_vs_transpose, stride_effect, working_set_bandwidth, ELEMS_PER_LINE,
)


def test_cache_line_is_8_float64():
    assert ELEMS_PER_LINE == 8            # 64B / 8B = 8 个 float64 一条缓存行


def test_transpose_slower_than_contiguous():
    tc, tt = contiguous_vs_transpose(2500)
    # 跨步(转置)读缓存不友好,应明显更慢(实测 ~4x,这里只要求 > 1.5x)
    assert tt > tc * 1.5


def test_stride_increases_time_across_cache_line():
    se = stride_effect(m=1_000_000)
    # 步长越过缓存行(8)后每有用元素多拖一条缓存行 → 变慢
    assert se[16] > se[2] * 1.5
    assert se[32] > se[1] * 3
    # 大体单调(允许小噪声):stride 8 慢于 stride 1
    assert se[8] > se[1]


def test_dram_slower_than_cache():
    ws = working_set_bandwidth(sizes_kb=(256, 262144), total_bytes=1_000_000_000)
    # 落到 DRAM(256MB)的流式带宽应明显低于装在 cache(256KB)里
    assert ws[262144] < ws[256]
