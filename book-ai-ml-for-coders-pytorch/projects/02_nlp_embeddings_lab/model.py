# -*- coding: utf-8 -*-
"""
model.py —— SentimentNet：Embedding + 均值池化 + Linear

复刻书里第 6 章的核心结构：
    nn.Embedding(vocab, dim)  ->  对序列做"掩码均值池化"  ->  Linear(dim, 1)

关键点：均值池化时必须 mask 掉 <pad>，否则大量补零会把句向量往 0 拉，
弄脏真正有意义的词向量。这里用 (id != PAD) 生成 mask，只对真实词求平均。

输出 1 个 logit，配 BCEWithLogitsLoss 做二分类（正面/负面）。
"""

import torch
import torch.nn as nn

from tokenizer import PAD_ID


class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, pad_id=PAD_ID):
        """
        参数
        ----
        vocab_size : 词表大小（含 <pad>/<oov>）
        embed_dim  : 词向量维度，小语料用 8~16 足够
        pad_id     : padding 的 id，用于 Embedding 的 padding_idx 与掩码
        """
        super().__init__()
        self.pad_id = pad_id
        # padding_idx=pad_id 让 <pad> 的向量恒为 0 且不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        x : LongTensor, 形状 (batch, seq_len)，元素是词 id
        返回 : FloatTensor, 形状 (batch,) 的 logit
        """
        # (batch, seq_len, embed_dim)
        emb = self.embedding(x)

        # 掩码：真实词为 1，<pad> 为 0，形状 (batch, seq_len, 1)
        mask = (x != self.pad_id).unsqueeze(-1).type_as(emb)

        # 只对真实词求和，再除以真实词个数 -> 掩码均值池化
        summed = (emb * mask).sum(dim=1)               # (batch, embed_dim)
        counts = mask.sum(dim=1).clamp(min=1.0)        # (batch, 1)，避免除 0
        pooled = summed / counts                        # (batch, embed_dim)

        logit = self.fc(pooled).squeeze(-1)             # (batch,)
        return logit
