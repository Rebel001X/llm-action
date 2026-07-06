# -*- coding: utf-8 -*-
"""
test_nlp.py —— 端到端离线测试（CPU、秒级、确定性）

覆盖：
  (a) 分词器往返 & padding 形状；
  (b) 模型前向输出形状；
  (c) 训练后验证集准确率 > 0.8；
  (d) classify 对明显正/负句给出正确倾向。
"""

import os
import sys

import torch

# 允许直接 pytest 时找到上级目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import SimpleTokenizer, PAD_ID, OOV_ID  # noqa: E402
from model import SentimentNet  # noqa: E402
from data import load_sentiment_data  # noqa: E402
from engine import train_classifier  # noqa: E402


# ---------- (a) 分词器往返 & padding ----------
def test_tokenizer_roundtrip_and_padding():
    tok = SimpleTokenizer()
    texts = ["I love this GREAT movie!", "This is terrible, awful."]
    tok.fit_on_texts(texts)

    # 特殊 token 固定 id
    assert tok.word_index["<pad>"] == PAD_ID
    assert tok.word_index["<oov>"] == OOV_ID
    assert tok.vocab_size >= 2

    seqs = tok.texts_to_sequences(texts)
    # 去掉标点后 "i love this great movie" -> 5 个 token
    assert len(seqs[0]) == 5

    # 未登录词映射到 <oov>
    unseen = tok.texts_to_sequences(["zzz unknownword"])
    assert unseen[0] == [OOV_ID, OOV_ID]

    # padding 形状：统一到 maxlen=8
    padded = tok.pad_sequences(seqs, maxlen=8)
    assert all(len(row) == 8 for row in padded)
    # post padding 末尾应是 <pad>
    assert padded[0][-1] == PAD_ID

    # 往返：sequences_to_texts 能还原有意义的词（跳过 pad）
    back = tok.sequences_to_texts(seqs)
    assert "love" in back[0] and "great" in back[0]


def test_padding_truncation():
    tok = SimpleTokenizer()
    tok.fit_on_texts(["a b c d e f"])
    seqs = tok.texts_to_sequences(["a b c d e f"])
    padded = tok.pad_sequences(seqs, maxlen=3, truncating="post")
    assert len(padded[0]) == 3  # 超长被截断


# ---------- (b) 模型前向形状 ----------
def test_model_forward_shape():
    torch.manual_seed(0)
    batch, seq_len, vocab = 4, 7, 20
    model = SentimentNet(vocab_size=vocab, embed_dim=8)
    x = torch.randint(0, vocab, (batch, seq_len))
    out = model(x)
    assert out.shape == (batch,)  # 每个样本 1 个 logit

    # 掩码有效性：<pad> 不应改变句向量的均值池化结果
    x2 = x.clone()
    # 在末尾拼一列全 pad，输出应基本不变
    x_pad = torch.cat([x2, torch.full((batch, 3), PAD_ID)], dim=1)
    out_pad = model(x_pad)
    assert torch.allclose(out, out_pad, atol=1e-5)


# ---------- (c) 训练后验证集准确率 > 0.8 ----------
def test_training_reaches_high_accuracy():
    train, val = load_sentiment_data(n_per_class=120, val_ratio=0.2, seed=0)
    clf, history = train_classifier(train, val, epochs=5, seed=0)
    val_acc = history[-1][1]
    assert val_acc is not None and val_acc > 0.8, f"val_acc={val_acc}"


# ---------- (d) classify 情感倾向正确 ----------
def test_classify_polarity():
    train, val = load_sentiment_data(n_per_class=120, val_ratio=0.2, seed=0)
    clf, _ = train_classifier(train, val, epochs=5, seed=0)

    pos_prob = clf.classify("i love this amazing wonderful movie")
    neg_prob = clf.classify("this is a terrible awful horrible film")

    assert pos_prob > 0.5, f"正面句概率应 >0.5，实际 {pos_prob}"
    assert neg_prob < 0.5, f"负面句概率应 <0.5，实际 {neg_prob}"
    assert pos_prob > neg_prob  # 正面句概率明显高于负面句
