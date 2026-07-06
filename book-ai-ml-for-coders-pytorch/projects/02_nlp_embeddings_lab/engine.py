# -*- coding: utf-8 -*-
"""
engine.py —— 训练 / 评估 / 单句推理

把"分词器 + 模型"打包成一个易用的 SentimentClassifier：
    clf = SentimentClassifier()
    clf.fit(train_texts, train_labels, val_texts, val_labels)
    clf.classify("i love this amazing movie")   # -> 正类概率

训练很小很快（几百样本、几个 epoch、embed_dim=16），CPU 秒级完成。
"""

import numpy as np
import torch
import torch.nn as nn

from tokenizer import SimpleTokenizer
from model import SentimentNet


def _to_tensor(sequences):
    """List[List[int]] -> LongTensor。"""
    return torch.tensor(sequences, dtype=torch.long)


class SentimentClassifier:
    def __init__(self, num_words=None, embed_dim=16, maxlen=12, seed=0):
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        self.seed = seed
        self.tokenizer = SimpleTokenizer(num_words=num_words)
        self.model = None

    # ---------- 内部工具 ----------
    def _encode(self, texts):
        """文本 -> padding 后的 LongTensor。"""
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = self.tokenizer.pad_sequences(seqs, maxlen=self.maxlen)
        return _to_tensor(padded)

    # ---------- 训练 ----------
    def fit(self, train_texts, train_labels,
            val_texts=None, val_labels=None,
            epochs=5, lr=0.05, batch_size=32, verbose=False):
        """在训练集上拟合，返回每个 epoch 的 (train_loss, val_acc) 历史。"""
        # 固定随机种子，保证词表 & 权重初始化可复现
        torch.manual_seed(self.seed)

        # 1) 建词表（只在训练集上 fit，避免信息泄露）
        self.tokenizer.fit_on_texts(train_texts)

        # 2) 建模型
        self.model = SentimentNet(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.embed_dim,
        )

        X = self._encode(train_texts)
        y = torch.tensor(train_labels, dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        n = X.shape[0]
        history = []
        for epoch in range(epochs):
            self.model.train()
            # 每个 epoch 用固定种子打乱，保证可复现
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + epoch))
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X[idx], y[idx]
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.shape[0]
            epoch_loss /= n

            val_acc = None
            if val_texts is not None and val_labels is not None:
                val_acc = self.evaluate(val_texts, val_labels)
            history.append((epoch_loss, val_acc))
            if verbose:
                msg = f"epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}"
                if val_acc is not None:
                    msg += f"  val_acc={val_acc:.3f}"
                print(msg)
        return history

    # ---------- 评估 ----------
    @torch.no_grad()
    def evaluate(self, texts, labels):
        """返回准确率（accuracy）。"""
        self.model.eval()
        X = self._encode(texts)
        logits = self.model(X)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        y = torch.tensor(labels, dtype=torch.long)
        return (preds == y).float().mean().item()

    # ---------- 单句推理 ----------
    @torch.no_grad()
    def classify(self, text):
        """返回该句为'正面'的概率（0~1 之间的 float）。"""
        self.model.eval()
        X = self._encode([text])
        logit = self.model(X)
        prob = torch.sigmoid(logit).item()
        return prob

    @torch.no_grad()
    def classify_batch(self, texts):
        """批量版本，返回 numpy 概率数组。"""
        self.model.eval()
        X = self._encode(texts)
        probs = torch.sigmoid(self.model(X)).cpu().numpy()
        return probs


# 便捷函数：一步训练出分类器（供 run_demo / 测试调用）
def train_classifier(train, val, epochs=5, lr=0.05, embed_dim=16,
                     maxlen=12, seed=0, verbose=False):
    """
    train, val : 分别是 (texts, labels) 元组
    返回训练好的 SentimentClassifier 与 history。
    """
    tr_texts, tr_labels = train
    va_texts, va_labels = val
    clf = SentimentClassifier(embed_dim=embed_dim, maxlen=maxlen, seed=seed)
    history = clf.fit(tr_texts, tr_labels, va_texts, va_labels,
                      epochs=epochs, lr=lr, verbose=verbose)
    return clf, history
