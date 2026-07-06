# -*- coding: utf-8 -*-
"""
knowledge.py —— 迷你知识库

这是 RAG 里"私有数据"的角色（第 18 章：LLM 训练集之外、需要靠检索补上的事实）。
书里用的是作者自己那本冷门科幻小说 PDF；这里换成一小份关于
**PyTorch 与本书各章节** 的明确事实（16 条），每条都自成一句、关键词清晰，
方便离线确定性嵌入把它们检索出来。

想换成你自己的数据？把 FACTS 换成你的段落即可（见 README「换成真实数据」）。
"""

from __future__ import annotations

from typing import Dict, List

# 每条事实：id 唯一，text 是一句可被单独检索/引用的事实。
FACTS: List[Dict[str, str]] = [
    {"id": "f01", "text": "张量（Tensor）是 PyTorch 的核心数据结构，可在 CPU 或 GPU 上做自动微分运算。"},
    {"id": "f02", "text": "本书第 2 章用 FashionMNIST 数据集入门计算机视觉，它包含 10 类灰度服饰图像。"},
    {"id": "f03", "text": "卷积神经网络（CNN）通过卷积核在图像中检测边缘、纹理等局部特征，对应本书第 3 章。"},
    {"id": "f04", "text": "PyTorch 用 Dataset 定义样本、用 DataLoader 做批量与打乱，对应本书第 4 章的数据管理。"},
    {"id": "f05", "text": "自然语言处理的第一步是分词并把词编码成数字索引（token），对应本书第 5 章。"},
    {"id": "f06", "text": "Embedding 嵌入把离散的词映射为稠密向量，让语义相近的词在向量空间中彼此靠近，对应第 6 章。"},
    {"id": "f07", "text": "循环神经网络 RNN 与 LSTM 能建模序列的时间依赖，LSTM 用门控缓解梯度消失，对应第 7 章。"},
    {"id": "f08", "text": "用语言模型逐字/逐词预测可以生成文本，本书第 8 章讲机器学习文本生成。"},
    {"id": "f09", "text": "时间序列数据具有趋势、季节性与噪声，本书第 9、10、11 章讲序列与时间序列建模。"},
    {"id": "f10", "text": "PyTorch 模型可用 TorchServe 或 Flask 部署成 HTTP 服务对外提供推理，对应第 13 章。"},
    {"id": "f11", "text": "Hugging Face Hub 是一个模型中心，可直接下载并复用第三方预训练模型，对应本书第 14 章。"},
    {"id": "f12", "text": "Transformer 架构以自注意力为核心，transformers 库提供 pipeline 一行调用预训练模型，对应第 15 章。"},
    {"id": "f13", "text": "用自定义数据微调（fine-tuning）或提示微调（prompt-tuning）可让 LLM 适配你的任务，对应本书第 16 章。"},
    {"id": "f14", "text": "Ollama 可在本地部署和服务开源大模型，默认监听 http://localhost:11434，对应本书第 17 章。"},
    {"id": "f15", "text": "RAG 检索增强生成：先按余弦相似度检索相关片段，再把片段拼进 prompt 交给 LLM 生成答案，对应第 18 章。"},
    {"id": "f16", "text": "余弦相似度用向量夹角衡量语义相似度，取值范围是 -1 到 1，1 表示方向完全一致最相似。"},
]


def get_facts() -> List[Dict[str, str]]:
    """返回知识库事实列表（返回副本，避免调用方误改）。"""
    return [dict(f) for f in FACTS]
