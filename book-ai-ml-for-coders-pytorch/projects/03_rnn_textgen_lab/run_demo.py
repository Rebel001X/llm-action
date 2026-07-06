# -*- coding: utf-8 -*-
"""端到端演示：建数据 → 训练 LSTM → 自回归生成文本。

跑法：
    python run_demo.py
"""
from __future__ import annotations

from dataset import make_dataset
from model import TextGen
from engine import train
from generate import generate

WINDOW = 3


def main():
    ds, word2idx, idx2word = make_dataset(window=WINDOW)
    print(f"词表大小 = {len(idx2word)}，样本数 = {len(ds)}")

    model = TextGen(vocab_size=len(idx2word))
    history = train(model, ds, epochs=40, seed=0)
    print(f"训练损失：首 {history[0]:.3f} → 末 {history[-1]:.3f}")

    for seed_text in ["the red bird", "the blue fish", "the green frog"]:
        text = generate(model, word2idx, idx2word, seed_text, n=6, window=WINDOW, temperature=0.0)
        print(f"  种子 [{seed_text}] → {text}")


if __name__ == "__main__":
    main()
