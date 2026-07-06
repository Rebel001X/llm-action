# -*- coding: utf-8 -*-
import os
import sys

import torch

# 让测试能 import 到项目根目录下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import build_sentences, build_vocab, PAD, UNK
from dataset import make_dataset
from model import TextGen
from engine import train
from generate import generate

WINDOW = 3


def test_vocab_and_dataset_shapes():
    ds, word2idx, idx2word = make_dataset(window=WINDOW)
    # 0/1 号必须是 pad/unk
    assert word2idx[PAD] == 0 and word2idx[UNK] == 1
    assert len(word2idx) == len(idx2word)
    assert len(ds) > 0
    X, y = ds[0]
    assert X.shape == (WINDOW,)
    assert y.ndim == 0  # 标签是标量
    assert X.dtype == torch.long and y.dtype == torch.long


def test_model_forward_shape():
    ds, word2idx, idx2word = make_dataset(window=WINDOW)
    model = TextGen(vocab_size=len(idx2word))
    X = torch.stack([ds[i][0] for i in range(4)])  # (4, WINDOW)
    logits = model(X)
    assert logits.shape == (4, len(idx2word))


def test_training_reduces_loss():
    ds, word2idx, idx2word = make_dataset(window=WINDOW)
    model = TextGen(vocab_size=len(idx2word))
    history = train(model, ds, epochs=30, seed=0)
    # 末期损失应显著低于初期（这份结构化语料很容易学）
    assert history[-1] < history[0] * 0.7


def test_greedy_generation_is_deterministic():
    ds, word2idx, idx2word = make_dataset(window=WINDOW)
    model = TextGen(vocab_size=len(idx2word))
    train(model, ds, epochs=20, seed=0)
    seed_text = "the red bird"
    a = generate(model, word2idx, idx2word, seed_text, n=6, window=WINDOW, temperature=0.0)
    b = generate(model, word2idx, idx2word, seed_text, n=6, window=WINDOW, temperature=0.0)
    # temperature=0 贪心：同一 seed 两次调用必须完全一致
    assert a == b
    assert a.startswith(seed_text)
    # 生成结果应比种子长（确实往后续写了 6 个词）
    assert len(a.split()) == len(seed_text.split()) + 6
