# -*- coding: utf-8 -*-
"""确定性小语料 + 词级词表（对应原书第 7–8 章）。

语料是一批**结构强、可学习**的良性短句（颜色 + 动物 + 动作 + 地点的组合），
词级 LSTM 很容易学到"上一个词 → 下一个词"的转移，几 epoch 就能收敛，
从而让贪心生成（temperature=0）稳定可复现。
"""
from __future__ import annotations

PAD, UNK = "<pad>", "<unk>"

# 用固定模板确定性地铺开一批句子——不含任何随机性，保证可复现。
_SUBJECTS = ["the red bird", "the blue fish", "the green frog", "the yellow duck"]
_ACTIONS = ["flew over", "swam past", "jumped across", "walked around"]
_PLACES = ["the quiet hill", "the wide river", "the green field", "the old bridge"]


def build_sentences() -> list[str]:
    """确定性地生成一批短句（4x4x4 = 64 句），每句结构一致、便于学习。"""
    out = []
    for s in _SUBJECTS:
        for a in _ACTIONS:
            for p in _PLACES:
                out.append(f"{s} {a} {p}")
    return out


def build_vocab(sentences: list[str]) -> tuple[dict[str, int], list[str]]:
    """建立 word->id 词表；0 号是 <pad>，1 号是 <unk>。返回 (word2idx, idx2word)。"""
    words: list[str] = [PAD, UNK]
    seen = set(words)
    for sent in sentences:
        for w in sent.split():
            if w not in seen:
                seen.add(w)
                words.append(w)
    word2idx = {w: i for i, w in enumerate(words)}
    return word2idx, words


def encode(tokens: list[str], word2idx: dict[str, int]) -> list[int]:
    """把词列表转成 id 列表，未登录词落到 <unk>。"""
    unk = word2idx[UNK]
    return [word2idx.get(w, unk) for w in tokens]


def decode(ids: list[int], idx2word: list[str]) -> list[str]:
    """把 id 列表转回词列表。"""
    return [idx2word[i] for i in ids]
