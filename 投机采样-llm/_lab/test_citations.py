"""引用核验的离线回归测试。

**这一条是本库最强的一道约束**：正文里每新增一个 arXiv 编号，
如果没有先跑过 `python citecheck.py --fetch` 联网核实，本测试就会红。
也就是说 —— **编造一个 arXiv 编号，测试会当场抓住。**
"""
import json
from pathlib import Path

import pytest

import citecheck

CACHE = Path(__file__).resolve().parent / "_citecache.json"


@pytest.fixture(scope="module")
def data():
    assert CACHE.exists(), "缺 _citecache.json，先跑 python citecheck.py --fetch"
    return json.loads(CACHE.read_text(encoding="utf-8")), citecheck.collect()


def test_every_cited_arxiv_id_has_been_verified(data):
    """全库每个 arXiv 编号都必须在缓存里 —— 新增引用必须先联网核实。"""
    cache, refs = data
    missing = sorted(set(refs) - set(cache))
    assert not missing, ("这些编号还没核实过（跑 python citecheck.py --fetch）：%s"
                         % missing[:10])


def test_no_fabricated_arxiv_ids(data):
    """**核心不变量：库里不存在编造的 arXiv 编号。**

    arXiv 的 Atom API 对不存在的编号不会返回带 abs/<id> 的 entry，
    所以"在缓存里且有官方标题"即等价于"这个编号真实存在"。
    """
    cache, refs = data
    r = citecheck.check(cache, refs)
    assert r["missing"] == [], r["missing"]
    for aid in refs:
        assert cache[aid]["title"], aid
        assert cache[aid]["published"], aid


def test_cache_covers_at_least_the_known_corpus(data):
    """防止有人把 md 删空后让测试"通过"。"""
    cache, refs = data
    assert len(refs) >= 140, len(refs)
    assert len(cache) >= len(refs)


def test_stated_year_month_matches_official_v1(data):
    """正文写的年月必须是**官方 v1** 的年月，不是编号前四位推出来的。

    这条抓过一次真错：早前用编号 YYMM 自动补年月，而**月底投稿会拿到下个月的编号**，
    于是 7 个编号推错（如 2604.09562 的 v1 其实是 2026-02）。已按官方日期改正。
    """
    cache, refs = data
    bad = []
    for aid, places in refs.items():
        implied = citecheck.implied_ym(aid)
        real = cache[aid]["published"][:7]
        if implied == real:
            continue                       # 一致时无从区分，跳过
        for fname, lineno, ctx in places:
            if implied in ctx and real not in ctx:
                bad.append((fname, lineno, aid, implied, real))
    assert not bad, "这些地方写的是编号推出来的年月而非官方 v1：%s" % bad[:6]
