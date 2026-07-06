# -*- coding: utf-8 -*-
"""
data.py —— 确定性合成的"正/负情感"小语料

不联网、不下载任何真实数据集。用固定模板 + 情感词组合出上百条句子，
保证：
  1) 完全可复现（固定随机种子 numpy Generator(0)）；
  2) 线性可分性强（正例只含好词、负例只含坏词），
     这样一个小 Embedding 模型几个 epoch 就能到 >0.8 准确率。

对应书里第 5-6 章：先有语料，再分词、再做嵌入情感分类。
真实场景请把 build_corpus() 换成读取 IMDB / Sarcasm / 自己的 CSV。
"""

import numpy as np

# 情感词：正面 / 负面。测试与 demo 会直接引用这些词造句。
POSITIVE_WORDS = [
    "great", "love", "amazing", "wonderful", "excellent", "fantastic",
    "brilliant", "awesome", "perfect", "delightful", "superb", "enjoyable",
]
NEGATIVE_WORDS = [
    "terrible", "hate", "awful", "horrible", "worst", "disappointing",
    "boring", "dreadful", "bad", "pathetic", "annoying", "useless",
]

# 中性填充词（两类都会用到，制造噪声但不改变极性）
NEUTRAL_WORDS = [
    "the", "this", "that", "movie", "book", "film", "story", "product",
    "it", "was", "is", "really", "very", "so", "quite", "and", "a",
]

# 句子模板，{s} 放情感词、{n} 放中性词
TEMPLATES = [
    "the {n} was {s}",
    "this {n} is {s} and {s}",
    "i think it was {s}",
    "a {s} {n} really {s}",
    "{s} {n} {s} experience",
    "it was so {s} the {n}",
    "what a {s} {n}",
    "really {s} and {s} {n}",
]


def _make_examples(rng, sentiment_words, label, n):
    """用模板 + 给定极性的情感词生成 n 条 (text, label) 样本。"""
    examples = []
    for _ in range(n):
        template = TEMPLATES[rng.integers(len(TEMPLATES))]
        # 模板里可能有多个 {s}/{n}，逐个填
        text = template
        while "{s}" in text:
            word = sentiment_words[rng.integers(len(sentiment_words))]
            text = text.replace("{s}", word, 1)
        while "{n}" in text:
            word = NEUTRAL_WORDS[rng.integers(len(NEUTRAL_WORDS))]
            text = text.replace("{n}", word, 1)
        examples.append((text, label))
    return examples


def build_corpus(n_per_class=120, seed=0):
    """
    生成平衡的正/负语料。

    返回
    ----
    texts  : List[str]
    labels : List[int]   1=正面, 0=负面
    """
    rng = np.random.default_rng(seed)
    pos = _make_examples(rng, POSITIVE_WORDS, 1, n_per_class)
    neg = _make_examples(rng, NEGATIVE_WORDS, 0, n_per_class)
    data = pos + neg

    # 用固定种子打乱，保证顺序可复现
    idx = rng.permutation(len(data))
    data = [data[i] for i in idx]

    texts = [t for t, _ in data]
    labels = [y for _, y in data]
    return texts, labels


def train_val_split(texts, labels, val_ratio=0.2, seed=0):
    """把语料切成训练/验证两份（确定性）。"""
    rng = np.random.default_rng(seed)
    n = len(texts)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = set(idx[:n_val].tolist())

    tr_texts, tr_labels, va_texts, va_labels = [], [], [], []
    for i in range(n):
        if i in val_idx:
            va_texts.append(texts[i])
            va_labels.append(labels[i])
        else:
            tr_texts.append(texts[i])
            tr_labels.append(labels[i])
    return (tr_texts, tr_labels), (va_texts, va_labels)


def load_sentiment_data(n_per_class=120, val_ratio=0.2, seed=0):
    """一步到位：生成语料并切分。返回 (train, val) 两个 (texts, labels) 元组。"""
    texts, labels = build_corpus(n_per_class=n_per_class, seed=seed)
    return train_val_split(texts, labels, val_ratio=val_ratio, seed=seed)
