# -*- coding: utf-8 -*-
"""
玩具图文嵌入生成器 (toy paired embeddings)
=========================================

真实的 CLIP 嵌入需要跑一个几亿参数的模型 + 一堆图片,离线拿不到。
这里用一个「共享潜在语义 + 各自模态噪声」的生成过程,凭空造出
一批「配对好」的 (图嵌入, 文嵌入),既能秒级运行又能真实复现出
跨模态检索的所有现象 (归一化重要、噪声让检索变难、温度不改排序……)。

生成模型 (为什么这样造):
    1. 先为第 i 个「概念」采一个共享语义向量 z_i  ~ N(0, I)。
       它代表「一只猫在沙发上」这类语义,图和文都围绕它。
    2. 图嵌入  img_i = z_i + eps_img   (加一点图像模态特有的噪声)
       文嵌入  txt_i = z_i + eps_txt   (加一点文本模态特有的噪声)
    共享的 z_i 让「第 i 图」和「第 i 文」天然最像 ->
    完美情况下自检索 recall@1 = 1。模态噪声 sigma 越大,越难对上。
"""

from __future__ import annotations

import numpy as np


def make_paired_embeddings(
    n: int = 200,
    dim: int = 64,
    modality_sigma: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 n 对 (图嵌入, 文嵌入),维度 dim。

    参数
        n              : 概念/样本对数量。
        dim            : 嵌入维度。
        modality_sigma : 图/文各自模态噪声的标准差。越大越难检索。
        seed           : 随机种子,保证可复现。

    返回
        (img, txt): 两个 (n, dim) 矩阵,第 i 行互为正确配对。
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=(n, dim))          # 共享语义
    img = z + rng.normal(0.0, modality_sigma, size=(n, dim))
    txt = z + rng.normal(0.0, modality_sigma, size=(n, dim))
    return img, txt


# 一组便于测试引用的小型「干净」样本 (模态噪声很小 -> 几乎完美对齐)
def make_easy_pairs(n: int = 50, dim: int = 32, seed: int = 42):
    return make_paired_embeddings(n=n, dim=dim, modality_sigma=0.05, seed=seed)
