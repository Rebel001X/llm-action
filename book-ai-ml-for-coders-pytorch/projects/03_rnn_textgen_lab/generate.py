# -*- coding: utf-8 -*-
"""自回归文本生成（对应原书第 8 章"复合预测滚雪球"）。

从种子词出发，反复"预测下一个词 → 接回输入 → 再预测"，就是 LLM 解码的玩具版。
- temperature == 0：贪心（argmax），**完全确定性、可复现**；
- temperature > 0：按温度缩放后采样，更有多样性（用传入 seed 固定随机性）。
"""
from __future__ import annotations

import torch

from corpus import encode, decode


@torch.no_grad()
def generate(model, word2idx, idx2word, seed_text: str, n: int = 6,
             window: int = 3, temperature: float = 0.0, sample_seed: int = 0) -> str:
    """生成 n 个词并把整段文本（含种子）拼成字符串返回。"""
    model.eval()
    tokens = encode(seed_text.split(), word2idx)

    for _ in range(n):
        # 取最近 window 个 token 作为上下文；不足则前面补 <pad>(0)
        ctx = tokens[-window:]
        if len(ctx) < window:
            ctx = [0] * (window - len(ctx)) + ctx
        x = torch.tensor([ctx], dtype=torch.long)   # (1, window)
        logits = model(x)[0]                         # (vocab,)

        if temperature <= 0:
            nxt = int(torch.argmax(logits).item())   # 贪心：确定性
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            g = torch.Generator().manual_seed(sample_seed)
            nxt = int(torch.multinomial(probs, 1, generator=g).item())
        tokens.append(nxt)

    return " ".join(decode(tokens, idx2word))
