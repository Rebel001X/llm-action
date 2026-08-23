"""
citecheck.py —— 引用核验：把全库每一个 arXiv 编号拿去问官方 API，逐条核对。

为什么要有这个：**编造引用是本库最严重的失败模式**。
`verify.py` 能保证"测试引用不虚"，但保证不了"论文引用不虚" ——
一个不存在的 arXiv 编号、一个张冠李戴的标题，自检器一个都抓不到。

此前只做过人工抽样（4 个 + 35 个 / 共 146 个）。本文件把它做成**全量、确定性、可复跑**的：

  1. 从正文与 _research 抽出全部 arXiv 编号；
  2. 批量查 export.arxiv.org 的 Atom API（官方元数据，不经过任何摘要模型）；
  3. 逐条核三件事：
       a) 编号**存在**吗？
       b) v1 的年月与**编号前四位 YYMM 一致**吗？（arXiv 编号本身就编码了提交年月）
       c) 正文在该编号附近写的标题，与**官方标题**对得上吗？
  4. 结果落盘 `_citecache.json`，之后可离线复跑（测试就是这么用的）。

**为什么不用 WebFetch**：本库踩过的坑 —— WebFetch 的摘要模型会编造内容
（编过一张 12 行表格、把 2026 读成 2024）。这里直接解析官方 Atom XML，不经过模型。

用法：
    python citecheck.py --fetch    # 联网抓取并写缓存（几分钟，对 arXiv 友好地限速）
    python citecheck.py            # 用缓存离线核验并打印报告
    python citecheck.py --stale    # 只列出缓存里缺失的编号
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / "_citecache.json"
API = "http://export.arxiv.org/api/query?id_list=%s&max_results=%d"
NS = {"a": "http://www.w3.org/2005/Atom"}

ARXIV_PAT = re.compile(r"(?:arXiv[:\s]*|arxiv\.org/(?:abs|pdf|html)/)(\d{4}\.\d{4,5})", re.I)
STOP = {"a", "an", "the", "of", "for", "with", "and", "to", "in", "on", "via",
        "is", "are", "by", "from", "at", "as", "we", "our", "its"}


def md_files() -> list:
    return sorted(ROOT.glob("*.md")) + sorted((ROOT / "_research").glob("*.md"))


def collect() -> dict:
    """{arxiv_id: [(文件名, 行号, 上下文), ...]}"""
    out: dict = {}
    for p in md_files():
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, raw in enumerate(lines):
            for m in ARXIV_PAT.finditer(raw):
                ctx = " ".join(lines[max(0, i - 2): i + 3])
                out.setdefault(m.group(1), []).append((p.name, i + 1, ctx))
    return out


def fetch(ids: list, batch: int = 25, pause: float = 3.0) -> dict:
    """批量查 arXiv Atom API。对 arXiv 友好：小批 + 间隔。"""
    meta = {}
    for k in range(0, len(ids), batch):
        chunk = ids[k:k + batch]
        url = API % (",".join(chunk), len(chunk))
        try:
            xml = urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:                              # noqa: BLE001
            print("  !! 批次 %d 抓取失败：%s" % (k // batch + 1, e))
            continue
        root = ET.fromstring(xml)
        got = 0
        for e in root.findall("a:entry", NS):
            eid = e.findtext("a:id", "", NS)
            m = re.search(r"abs/(\d{4}\.\d{4,5})", eid)
            if not m:
                continue                                    # 查不到时 arXiv 会回一条占位 entry
            meta[m.group(1)] = {
                "title": " ".join((e.findtext("a:title", "", NS) or "").split()),
                "published": e.findtext("a:published", "", NS)[:10],
                "updated": e.findtext("a:updated", "", NS)[:10],
                "authors": [a.findtext("a:name", "", NS)
                            for a in e.findall("a:author", NS)][:6],
            }
            got += 1
        print("  批次 %2d/%d：请求 %2d 条，拿到 %2d 条"
              % (k // batch + 1, (len(ids) + batch - 1) // batch, len(chunk), got))
        time.sleep(pause)
    return meta


def implied_ym(aid: str) -> str:
    return "20%s-%s" % (aid[:2], aid[2:4])


def title_tokens(t: str) -> set:
    return {w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", t.lower())
            if w not in STOP}


def lead_name(title: str) -> str | None:
    """标题的**主名**：冒号前那段里最长的实词，多半就是系统名（TurboSpec / EAGLE / DFlash）。

    比"全标题实词命中率"灵敏得多 —— 正文引用时用的是简称，全标题的词本来就不会出现；
    但如果连**主名**都不出现，就值得怀疑"编号张冠李戴"了。
    """
    head = title.split(":")[0]
    ws = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", head)
    if not ws:
        return None
    ws.sort(key=len, reverse=True)
    w = ws[0].lower()
    return None if w in {"speculative", "efficiency", "accelerating", "decoding",
                         "inference", "language", "models", "breaking"} else w


def check(cache: dict, refs: dict) -> dict:
    """返回 {missing, date_mismatch, title_suspect, ok}"""
    missing, date_bad, title_bad, lead_bad, ok = [], [], [], [], []
    for aid, places in sorted(refs.items()):
        m = cache.get(aid)
        if not m:
            missing.append((aid, places[0][0]))
            continue
        if m["published"][:7] != implied_ym(aid):
            date_bad.append((aid, implied_ym(aid), m["published"]))
        # 标题核对**只查正文章节**，不查 _research。
        # 理由：调研笔记末尾是成片的 URL 清单，那里本来就不写标题，
        # 拿"标题实词有没有出现"去判它，报出来的全是噪声（实测 44 -> 12）。
        prose = [c for f, _, c in places if not f.startswith("RS-")]
        if not prose:
            ok.append(aid)
            continue
        blob = " ".join(prose).lower()
        ln = lead_name(m["title"])
        if ln and ln not in blob:
            lead_bad.append((aid, m["title"][:56], ln, prose and places[0][0]))
        toks = title_tokens(m["title"])
        if toks:
            hit = sum(1 for w in toks if w in blob) / len(toks)
            if hit < 0.34:
                title_bad.append((aid, m["title"][:64], round(hit, 2), places[0][0]))
            else:
                ok.append(aid)
        else:
            ok.append(aid)
    return dict(missing=missing, date=date_bad, title=title_bad,
                lead=lead_bad, ok=ok)


def _report(refs: dict, cache: dict):
    r = check(cache, refs)
    print("=" * 92)
    print("引用核验（全量，数据来自 export.arxiv.org 官方 Atom API，不经过任何摘要模型）")
    print("=" * 92)
    print("全库 arXiv 编号 %d 个，缓存命中 %d 个" % (len(refs), len(cache)))
    print()
    print("① 编号不存在 / 未抓到：%d 个" % len(r["missing"]))
    for aid, f in r["missing"]:
        print("   MISSING  arXiv:%s   首现于 %s" % (aid, f))
    print("② v1 年月与编号编码不符：%d 个 —— **这是 arXiv 的正常现象，不是错**" % len(r["date"]))
    print("   （月底投稿会拿到下个月的编号。列在这里是为了提醒：**别拿编号前四位当年月**，"
          "要用官方 published。本库正文写的年月已按官方 v1 逐条改正。）")
    for aid, want, got in r["date"]:
        print("   DATE     arXiv:%s   编号隐含 %s  官方 v1 %s" % (aid, want, got))
    print("③ **标题主名**（系统名）在正文附近找不到：%d 个 —— 这一条最灵敏，优先看" % len(r["lead"]))
    for aid, t, ln, f in r["lead"]:
        print("   LEAD?    arXiv:%s  主名 '%s' 未出现  官方《%s》  见 %s" % (aid, ln, t, f))
    print("④ 全标题实词命中率 <34%%：%d 个（灵敏度低、噪声多，仅供参考）" % len(r["title"]))
    for aid, t, hit, f in r["title"]:
        print("   TITLE?   arXiv:%s  命中%.0f%%  官方《%s》  见 %s" % (aid, hit * 100, t, f))
    print()
    # 判决只看**存在性**：编号查不到 = 可能编造，这是唯一的硬失败。
    # ② 的日期差是 arXiv 编号规则本身造成的，拿它当失败会天天喊狼来了。
    print("=> 存在性（有没有编造的编号）：%s"
          % ("**全部通过**" if not r["missing"] else "**否，有 %d 个查不到**" % len(r["missing"])))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="联网抓取并写缓存")
    ap.add_argument("--stale", action="store_true", help="只列缓存里缺的编号")
    a = ap.parse_args()

    refs = collect()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    if a.stale:
        miss = [i for i in sorted(refs) if i not in cache]
        print("缓存缺 %d 个：%s" % (len(miss), ", ".join(miss)))
        return

    if a.fetch:
        todo = [i for i in sorted(refs) if i not in cache]
        print("需抓取 %d 个（已缓存 %d 个）" % (len(todo), len(cache)))
        if todo:
            cache.update(fetch(todo))
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1,
                                        sort_keys=True), encoding="utf-8")
            print("缓存已写入 %s（共 %d 条）" % (CACHE.name, len(cache)))

    _report(refs, cache)


if __name__ == "__main__":
    main()
