# -*- coding: utf-8 -*-
"""词级文本生成模型：Embedding + LSTM + Linear（对应原书第 7–8 章）。

结构与原书一致：
    token ids ──nn.Embedding──> 词向量序列 ──nn.LSTM──> 隐状态序列
             ──取最后一个时间步──> nn.Linear ──> 每个词的 logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TextGen(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window)
        emb = self.embed(x)              # (batch, window, embed_dim)
        out, _ = self.lstm(emb)          # (batch, window, hidden)
        last = out[:, -1, :]             # 取最后一个时间步 (batch, hidden)
        return self.fc(last)             # (batch, vocab_size) —— 下一个词的 logits
