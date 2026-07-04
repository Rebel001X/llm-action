# -*- coding: utf-8 -*-
"""
clip_lab.py — CLIP 式对比学习的最小可跑内核(纯 torch / CPU / 离线)
================================================================

本文件实现一个"玩具版 CLIP":
  1) 一批**图像特征**(image features)与一批**文本特征**(text features);
  2) 两个小型投影编码器 ImageEncoder / TextEncoder,把两种模态各自
     映射到同一个**共享嵌入空间**(shared embedding space);
  3) 对称的 **InfoNCE 对比损失**(带**可学习温度** temperature);
  4) 一个玩具数据生成器,构造"天生配对"的图文特征;
  5) 训练循环 + 检索指标(recall@1)。

设计原则(第一性原理):
  - CLIP 的本质是"让配对的 (图, 文) 在嵌入空间里更近,让非配对的更远"。
  - "更近/更远"用**余弦相似度**度量(先 L2 归一化,再点积)。
  - "让配对更近"这件事,被写成一个**分类问题**:在一个 batch 内,
    对第 i 张图,它的正确文本就是第 i 条文本(对角线),其余 N-1 条是负样本。
    于是"图->文检索"是一个 N 分类,"文->图检索"也是一个 N 分类,
    两个方向的交叉熵取平均,就是 CLIP 的损失。

只依赖 torch,不需要 GPU、网络或任何预训练权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. 玩具数据:构造"天生配对"的图 / 文原始特征
# =============================================================================
def make_toy_pairs(
    n_pairs: int = 64,
    img_dim: int = 32,
    txt_dim: int = 24,
    n_concepts: int = 8,
    noise: float = 0.35,
    seed: int = 0,
):
    """生成一批配对的(图像特征, 文本特征)。

    造数据的核心思路(为什么这样造):
      - 我们先随机造 ``n_concepts`` 个"概念原型"(比如"猫""狗""飞机"...),
        每个概念在**图像侧**有一个原型向量 ``img_proto``、在**文本侧**
        有另一个原型向量 ``txt_proto``。两侧原型维度不同、数值也不同 ——
        这模拟了"图像和文本是两种完全不同的模态,原始特征不可直接比较"。
      - 第 i 个样本随机抽一个概念 c,它的图像特征 = 该概念的图像原型 + 噪声,
        文本特征 = 该概念的文本原型 + 噪声。
      - 于是"第 i 张图"和"第 i 条文本"共享同一个概念 c —— 它们是**配对**的,
        但因为两侧原型不同,模型必须**学会一个投影**才能把它们对齐。

    参数:
      n_pairs   : 样本对数量(= batch 内的 N)。
      img_dim   : 图像原始特征维度。
      txt_dim   : 文本原始特征维度(故意与 img_dim 不同)。
      n_concepts: 概念(类别)数;越小,不同样本越容易"撞概念"->越难区分。
      noise     : 加在原型上的高斯噪声强度;越大越难。
      seed      : 随机种子,保证可复现。

    返回:
      img_feats : (n_pairs, img_dim)  图像原始特征
      txt_feats : (n_pairs, txt_dim)  文本原始特征
      labels    : (n_pairs,)          每个样本所属的概念 id(仅用于分析/可视化)
    """
    g = torch.Generator().manual_seed(seed)

    # 两侧的概念原型:形状分别是 (n_concepts, img_dim) 和 (n_concepts, txt_dim)。
    # 原型之间要"彼此拉开",所以直接用标准正态采样即可(不同概念大概率不共线)。
    img_proto = torch.randn(n_concepts, img_dim, generator=g)
    txt_proto = torch.randn(n_concepts, txt_dim, generator=g)

    # 为每个样本随机分配一个概念 id。
    labels = torch.randint(0, n_concepts, (n_pairs,), generator=g)

    # 按概念 id 取出对应原型,再各自叠加噪声,得到最终的原始特征。
    img_feats = img_proto[labels] + noise * torch.randn(n_pairs, img_dim, generator=g)
    txt_feats = txt_proto[labels] + noise * torch.randn(n_pairs, txt_dim, generator=g)

    return img_feats, txt_feats, labels


# =============================================================================
# 2. 两个模态各自的小型投影编码器
# =============================================================================
class ImageEncoder(nn.Module):
    """把图像原始特征投影到共享嵌入空间(一个两层 MLP)。"""

    def __init__(self, in_dim: int, embed_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),  # 升维,给非线性留空间
            nn.GELU(),                  # 非线性激活(比 ReLU 更平滑,CLIP/ViT 常用)
            nn.Linear(hidden, embed_dim),  # 投影到共享空间
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # 注意:这里**不**做归一化,归一化留给损失/相似度函数统一处理


class TextEncoder(nn.Module):
    """把文本原始特征投影到共享嵌入空间(结构同图像侧,但参数独立)。"""

    def __init__(self, in_dim: int, embed_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 3. 相似度与 InfoNCE 对比损失
# =============================================================================
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """按最后一维做 L2 归一化,把向量放到单位球面上。

    归一化之后,两个向量的**点积**就等于它们的**余弦相似度** ∈ [-1, 1]。
    这一步是 CLIP 的关键:让相似度只关心"方向",不关心"长度"。
    """
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def similarity_matrix(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """计算图 vs 文的相似度矩阵(logits),形状 (N_img, N_txt)。

    步骤:
      1) 各自 L2 归一化 -> 余弦相似度;
      2) 除以温度 temperature(等价于乘以 logit_scale = 1/temperature),
         得到用于交叉熵的 logits。

    温度的作用(⚠️高频考点):
      - temperature 越**小**(如 0.01),logits 被放得越大,softmax 越"尖锐",
        模型对"最像的那个"惩罚/奖励越极端 —— 对难负样本更敏感,但也更易过拟合/不稳。
      - temperature 越**大**(如 1.0),softmax 越"平缓",区分度下降,训练更温和但更慢。
      - CLIP 原文用**可学习**的 logit_scale,并把它 clamp 到 <=100(即 temperature>=0.01)。
    """
    img_n = l2_normalize(img_emb)
    txt_n = l2_normalize(txt_emb)
    # (N,d) @ (d,N) -> (N,N);sim[i,j] = cos(图 i, 文 j)
    sim = img_n @ txt_n.t()
    return sim / temperature


def info_nce_loss(sim: torch.Tensor) -> torch.Tensor:
    """对称 InfoNCE 损失(CLIP 的损失函数)。

    输入:
      sim : (N, N) 的相似度 logits,sim[i,j] = 图 i 与 文 j 的(缩放后)相似度。
            **约定对角线 sim[i,i] 是配对**(正样本)。

    做法:
      - 图->文方向:把 sim 的**每一行**看作一个 N 分类的 logits,
        第 i 行的正确类别是 i(对角线)。用交叉熵。
      - 文->图方向:把 sim 的**每一列**看作一个 N 分类;等价于对 sim.t() 按行做交叉熵。
      - 两个方向的交叉熵取平均 -> 对称损失。

    为什么是"分类":InfoNCE = 在 1 个正样本 + (N-1) 个负样本里,
    用 softmax 把概率质量推给正样本。最小化它 <=> 最大化配对的互信息下界。
    """
    n = sim.shape[0]
    # 目标标签就是 0,1,2,...,N-1,即"第 i 个查询的正确答案是第 i 个"。
    targets = torch.arange(n, device=sim.device)
    loss_i2t = F.cross_entropy(sim, targets)       # 每行一个 N 分类(图找文)
    loss_t2i = F.cross_entropy(sim.t(), targets)   # 每列一个 N 分类(文找图)
    return 0.5 * (loss_i2t + loss_t2i)


# =============================================================================
# 4. 完整模型:两塔 + 可学习温度
# =============================================================================
class CLIPModel(nn.Module):
    """双塔 CLIP:图像塔 + 文本塔 + 可学习温度(logit_scale)。"""

    def __init__(
        self,
        img_dim: int,
        txt_dim: int,
        embed_dim: int = 16,
        hidden: int = 64,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(img_dim, embed_dim, hidden)
        self.text_encoder = TextEncoder(txt_dim, embed_dim, hidden)
        # CLIP 的技巧:不直接学 temperature,而是学 log(1/temperature)=log_logit_scale,
        # 这样温度恒为正、且以"乘性"方式变化更稳定。
        init_logit_scale = torch.log(torch.tensor(1.0 / init_temperature))
        self.log_logit_scale = nn.Parameter(init_logit_scale)

    @property
    def temperature(self) -> torch.Tensor:
        """当前等效温度 = 1 / exp(log_logit_scale)。

        detach:这是"只读"查询,不希望在 float(model.temperature) 时
        触发 autograd 警告,也不应把它接进计算图。
        """
        return 1.0 / self.log_logit_scale.detach().exp()

    def encode(self, img_feats: torch.Tensor, txt_feats: torch.Tensor):
        """返回(未归一化的)图/文嵌入。"""
        return self.image_encoder(img_feats), self.text_encoder(txt_feats)

    def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor):
        """前向:返回 (相似度矩阵 logits, 损失)。"""
        img_emb, txt_emb = self.encode(img_feats, txt_feats)
        # clamp:防止 log_logit_scale 学飞导致温度过小(<0.01)而数值爆炸,复刻 CLIP 官方做法。
        logit_scale = self.log_logit_scale.clamp(max=torch.log(torch.tensor(100.0))).exp()
        img_n = l2_normalize(img_emb)
        txt_n = l2_normalize(txt_emb)
        sim = (img_n @ txt_n.t()) * logit_scale  # (N,N) logits
        loss = info_nce_loss(sim)
        return sim, loss


# =============================================================================
# 5. 检索指标:recall@1
# =============================================================================
def recall_at_1(sim: torch.Tensor) -> float:
    """给定相似度矩阵(约定对角线为配对),计算双向 recall@1 的平均。

    recall@1 的含义:对每个查询,取相似度最高的那个候选,
    如果它正好是配对(对角线),就算命中。命中率就是 recall@1。
    """
    n = sim.shape[0]
    targets = torch.arange(n, device=sim.device)
    i2t = (sim.argmax(dim=1) == targets).float().mean()   # 每行最大值是否落在对角线
    t2i = (sim.argmax(dim=0) == targets).float().mean()   # 每列最大值是否落在对角线
    return float(0.5 * (i2t + t2i))


# =============================================================================
# 6. 训练循环
# =============================================================================
def train_clip(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    embed_dim: int = 16,
    hidden: int = 64,
    init_temperature: float = 0.07,
    lr: float = 5e-3,
    steps: int = 300,
    seed: int = 0,
    verbose: bool = False,
):
    """在给定的配对特征上训练一个玩具 CLIP,返回 (model, history)。

    history 是一个 dict,记录逐步的:
      - 'loss'       : InfoNCE 损失
      - 'recall'     : 双向 recall@1
      - 'temperature': 当前等效温度
    以及初末的相似度矩阵,供 run_demo 画热图。
    """
    torch.manual_seed(seed)  # 固定编码器的初始化,保证可复现
    model = CLIPModel(
        img_dim=img_feats.shape[1],
        txt_dim=txt_feats.shape[1],
        embed_dim=embed_dim,
        hidden=hidden,
        init_temperature=init_temperature,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "recall": [], "temperature": []}

    # 记录"训练前"的相似度矩阵(用归一化后的余弦相似度,范围 [-1,1],便于画热图)。
    model.eval()
    with torch.no_grad():
        img_emb, txt_emb = model.encode(img_feats, txt_feats)
        sim_before = (l2_normalize(img_emb) @ l2_normalize(txt_emb).t()).clone()
    model.train()

    for step in range(steps):
        opt.zero_grad()
        _, loss = model(img_feats, txt_feats)  # 全 batch 梯度下降(玩具规模,直接全量)
        loss.backward()
        opt.step()

        # 记录指标(用不带温度缩放的纯余弦相似度评估检索,更直观)。
        with torch.no_grad():
            img_emb, txt_emb = model.encode(img_feats, txt_feats)
            cos = l2_normalize(img_emb) @ l2_normalize(txt_emb).t()
            history["loss"].append(float(loss))
            history["recall"].append(recall_at_1(cos))
            history["temperature"].append(float(model.temperature))

        if verbose and (step % 50 == 0 or step == steps - 1):
            print(
                f"step {step:4d} | loss {loss.item():.4f} "
                f"| recall@1 {history['recall'][-1]:.3f} "
                f"| temp {history['temperature'][-1]:.4f}"
            )

    # 记录"训练后"的相似度矩阵。
    model.eval()
    with torch.no_grad():
        img_emb, txt_emb = model.encode(img_feats, txt_feats)
        sim_after = (l2_normalize(img_emb) @ l2_normalize(txt_emb).t()).clone()

    history["sim_before"] = sim_before
    history["sim_after"] = sim_after
    return model, history


if __name__ == "__main__":
    # 直接运行本文件:跑一遍最小训练,打印关键指标,方便快速自检。
    imgs, txts, labels = make_toy_pairs(n_pairs=64, seed=0)
    model, hist = train_clip(imgs, txts, steps=300, verbose=True)
    print("\n[训练前] recall@1 =", recall_at_1(hist["sim_before"]))
    print("[训练后] recall@1 =", recall_at_1(hist["sim_after"]))
    print("[训练后] 等效温度 =", float(model.temperature))
