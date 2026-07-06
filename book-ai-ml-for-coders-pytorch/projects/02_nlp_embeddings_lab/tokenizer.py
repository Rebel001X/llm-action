# -*- coding: utf-8 -*-
"""
tokenizer.py —— 极简分词器 SimpleTokenizer

复刻《AI and Machine Learning for Coders》第 5 章的分词流程：
    小写化 -> 去标点 -> 按空格切 -> 建词表(vocab) -> 文本转序列 -> padding

对应 Keras 里的 Tokenizer + texts_to_sequences + pad_sequences，
这里用纯 Python 手写，方便看清每一步在做什么。

词表里预留两个特殊 token：
    <pad>  -> id 0，用来把不等长的句子补齐到同样长度
    <oov>  -> id 1，"out of vocabulary"，训练时没见过的词都映射到它
"""

import re
from collections import Counter

# 两个特殊 token 的字符串与固定 id
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"
PAD_ID = 0
OOV_ID = 1


def basic_clean(text):
    """小写化 + 去掉标点，只保留字母/数字/空格。返回清洗后的字符串。"""
    text = text.lower()
    # 把非 a-z0-9 的字符替换成空格（等价于"去标点"）
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def basic_tokenize(text):
    """清洗后按空白切词，返回 token 列表。"""
    return basic_clean(text).split()


class SimpleTokenizer:
    """
    最小可用的分词器。

    典型用法：
        tok = SimpleTokenizer(num_words=None)
        tok.fit_on_texts(train_texts)             # 统计词频、建词表
        seqs = tok.texts_to_sequences(texts)      # 文本 -> id 序列
        padded = tok.pad_sequences(seqs, maxlen=10)  # 补齐/截断

    参数
    ----
    num_words : int | None
        只保留词频最高的前 num_words 个词（不含特殊 token）。
        None 表示全部保留。对应 Keras Tokenizer 的 num_words。
    """

    def __init__(self, num_words=None):
        self.num_words = num_words
        # word -> id 与 id -> word 两张表
        self.word_index = {PAD_TOKEN: PAD_ID, OOV_TOKEN: OOV_ID}
        self.index_word = {PAD_ID: PAD_TOKEN, OOV_ID: OOV_TOKEN}
        self.word_counts = Counter()
        self._fitted = False

    @property
    def vocab_size(self):
        """词表大小（含 <pad> 与 <oov>）。"""
        return len(self.word_index)

    def fit_on_texts(self, texts):
        """在语料上统计词频并构建词表。"""
        for text in texts:
            self.word_counts.update(basic_tokenize(text))

        # 按词频从高到低排序（同频按字母序，保证结果确定、可复现）
        ordered = sorted(self.word_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if self.num_words is not None:
            ordered = ordered[: self.num_words]

        # 从 id=2 开始给普通词分配 id（0/1 已被特殊 token 占用）
        next_id = 2
        for word, _count in ordered:
            if word not in self.word_index:
                self.word_index[word] = next_id
                self.index_word[next_id] = word
                next_id += 1

        self._fitted = True
        return self

    def texts_to_sequences(self, texts):
        """把一批文本转成 id 序列列表；未登录词映射为 <oov>。"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit_on_texts 再转换文本。")
        sequences = []
        for text in texts:
            seq = [self.word_index.get(tok, OOV_ID) for tok in basic_tokenize(text)]
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        """把 id 序列还原成文本（跳过 <pad>）。主要用于调试/往返测试。"""
        texts = []
        for seq in sequences:
            words = [self.index_word.get(int(i), OOV_TOKEN) for i in seq if int(i) != PAD_ID]
            texts.append(" ".join(words))
        return texts

    def pad_sequences(self, sequences, maxlen=None, padding="post", truncating="post"):
        """
        把不等长的 id 序列补齐/截断到同一长度，返回 List[List[int]]。

        参数
        ----
        maxlen : int | None
            目标长度。None 时取这批序列里最长的那条。
        padding : "post" | "pre"
            在末尾还是开头补 <pad>。
        truncating : "post" | "pre"
            超长时从末尾还是开头截断。
        """
        if maxlen is None:
            maxlen = max((len(s) for s in sequences), default=0)
            maxlen = max(maxlen, 1)  # 避免全空时 maxlen=0

        padded = []
        for seq in sequences:
            seq = list(seq)
            # 截断
            if len(seq) > maxlen:
                if truncating == "post":
                    seq = seq[:maxlen]
                else:
                    seq = seq[-maxlen:]
            # 补齐
            if len(seq) < maxlen:
                pad = [PAD_ID] * (maxlen - len(seq))
                seq = seq + pad if padding == "post" else pad + seq
            padded.append(seq)
        return padded
