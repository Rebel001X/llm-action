# -*- coding: utf-8 -*-
"""把语料切成 (输入窗口, 下一个词) 的监督样本（对应原书第 8 章 windowing 思路）。

原书用不断增长的 n-gram 前缀来预测下一个词；这里用**定长滑动窗口**（更干净、
形状固定、便于批处理）：对每个句子，取长度为 window 的连续片段作为输入，
紧随其后的那个词作为标签。
"""
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from corpus import build_sentences, build_vocab, encode


def make_dataset(window: int = 3):
    """返回 (dataset, word2idx, idx2word)。

    dataset 里每个样本是 (X[window], y)，X/y 都是 long 张量。
    """
    sentences = build_sentences()
    word2idx, idx2word = build_vocab(sentences)

    X_rows, y_rows = [], []
    for sent in sentences:
        ids = encode(sent.split(), word2idx)
        # 定长滑动窗口：需要至少 window+1 个 token 才能取到一个样本
        for i in range(len(ids) - window):
            X_rows.append(ids[i : i + window])
            y_rows.append(ids[i + window])

    X = torch.tensor(X_rows, dtype=torch.long)  # (N, window)
    y = torch.tensor(y_rows, dtype=torch.long)  # (N,)
    return TensorDataset(X, y), word2idx, idx2word
