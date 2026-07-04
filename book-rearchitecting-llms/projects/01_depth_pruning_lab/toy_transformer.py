# -*- coding: utf-8 -*-
"""
toy_transformer.py — 一个「玩具版」多层 Transformer(纯 torch, CPU 可跑)。

设计目标(对齐《Rearchitecting LLMs》第 4 章 深度剪枝):
  - 结构刻意做成和 Qwen3DecoderLayer 同构:每个 block = Pre-Norm + Attention + 残差 + Pre-Norm + MLP + 残差。
  - block 之间维度完全一致([B, S, D] 进 [B, S, D] 出),这样才能拿「输入激活 vs 输出激活」比相似度。
  - 所有 block 结构相同、可堆叠,`self.layers` 是一个 ModuleList —— 深度剪枝就是「重建一个更短的 layers 列表」。

术语(和书里三级术语对齐):
  - block(块)  = 一个 TransformerBlock,对应 HF 里的 model.layers[i]
  - module(模块)= block 内部的 attn / mlp
  - layer(层)  = 单个 nn.Linear / nn.LayerNorm 计算操作

⚠️ 本文件不联网、不下载任何权重,随机初始化即可 —— 我们要验证的是「剪枝的机械正确性 + 重要性排序的确定性」,
   不是「模型说人话」。这正是「本机可跑的实战 lab」和「跑真 Qwen3」的分工。
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# 配置对象:把「模型有多大」集中成一个 dataclass,方便复现和测试。
# --------------------------------------------------------------------------------------
@dataclass
class ToyConfig:
    vocab_size: int = 256      # 词表大小(玩具级,够跑通即可)
    d_model: int = 64          # 隐藏维 D(对应 Qwen3 的 1024,这里缩到 64 让 CPU 秒跑)
    n_layers: int = 12         # block 数量(对应 Qwen3-0.6B 的 28)
    n_heads: int = 4           # 注意力头数,需整除 d_model
    d_ff: int = 256            # MLP 中间维(通常是 d_model 的 4 倍左右)
    max_seq: int = 128         # 支持的最大序列长度(位置编码用)
    seed: int = 0              # 随机种子:固定它,重要性排序才「确定」

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"


# --------------------------------------------------------------------------------------
# 一个 Transformer block(= 书里说的「block / Qwen3DecoderLayer」)。
# 结构:Pre-Norm 架构(现代 LLM 主流),即 LayerNorm 放在子层「之前」。
# --------------------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        # --- Attention 模块的四个投影(对应 Qwen3 的 q/k/v/o_proj) ---
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # --- MLP 模块(SwiGLU 风格:gate / up / down,和 Qwen3MLP 同构) ---
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

        # --- 两个 Pre-Norm(对应 input_layernorm / post_attention_layernorm) ---
        self.input_layernorm = nn.LayerNorm(cfg.d_model)
        self.post_attention_layernorm = nn.LayerNorm(cfg.d_model)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        """标准因果多头自注意力。x: [B, S, D] -> [B, S, D]。"""
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # 把 [B,S,D] 投影后拆成多头 [B,H,S,Dh]
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        # 缩放点积注意力分数 [B,H,S,S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        # 因果掩码:第 i 个 token 只能看 <= i 的 token(下三角保留,上三角设 -inf)
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                       # [B,H,S,Dh]
        out = out.transpose(1, 2).contiguous().view(B, S, D)  # 合并多头回 [B,S,D]
        return self.o_proj(out)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU MLP:down( SiLU(gate(x)) * up(x) )。"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm 残差写法:x = x + sublayer(norm(x))。
        # 正因为有「+ x」残差,输入和输出维度一致,block 才是「在原信息上做增量变换」——
        # 这也是「输入输出相似度」能当重要性指标的物理基础:改动小 => 相似度高 => 该 block 打酱油。
        x = x + self._attn(self.input_layernorm(x))
        x = x + self._mlp(self.post_attention_layernorm(x))
        return x


# --------------------------------------------------------------------------------------
# 整个玩具模型:embedding -> N 个 block -> final norm -> lm_head。
# 关键:self.layers 是 ModuleList,深度剪枝直接替换它。
# --------------------------------------------------------------------------------------
class ToyTransformer(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        # 固定种子,保证「随机初始化」也是可复现的 —— 重要性排序的确定性依赖它。
        torch.manual_seed(cfg.seed)

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    @property
    def n_layers(self) -> int:
        """当前 block 数(剪枝后会变小)。始终以 len(self.layers) 为准。"""
        return len(self.layers)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        """
        input_ids: [B, S] 的 long 张量。
        返回:带 .logits(和可选 .loss)的小对象,模仿 HF 的输出接口。
        """
        B, S = input_ids.shape
        assert S <= self.cfg.max_seq, f"序列长 {S} 超过 max_seq={self.cfg.max_seq}"

        pos = torch.arange(S, device=input_ids.device)
        x = self.embed_tokens(input_ids) + self.pos_embed(pos)[None, :, :]

        for block in self.layers:      # 逐 block 前向;剪枝后这个循环自然变短
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)       # [B, S, vocab]

        loss = None
        if labels is not None:
            # 语言模型:用第 t 个位置预测第 t+1 个 token(错位一格)。
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,     # -100 位置(如 padding)不计入 loss
            )
        return ToyOutput(logits=logits, loss=loss)


@dataclass
class ToyOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


# --------------------------------------------------------------------------------------
# 便捷构造 + 造点玩具数据。
# --------------------------------------------------------------------------------------
def build_toy_model(cfg: ToyConfig | None = None, vary_blocks: bool = False) -> ToyTransformer:
    """
    构造玩具模型。

    vary_blocks=False:纯随机初始化。此时各 block 的 BI 都很小且接近(残差主导),
                      适合验证「机械正确性 + 确定性」,但柱状图会偏平。
    vary_blocks=True: 给不同 block 的子层权重乘上一个「确定性的振幅曲线」,
                      人为制造「有的 block 干实事、有的打酱油」的重要性梯度,
                      让 run_demo 的柱状图更像书里图 4.6 那样有明显高低。
                      这只是为了「可视化好看」,不改变任何剪枝逻辑。
    """
    cfg = cfg or ToyConfig()
    model = ToyTransformer(cfg)
    if vary_blocks:
        _apply_importance_profile(model)
    model.eval()  # 我们只做推理/观测,不训练
    return model


def _apply_importance_profile(model: ToyTransformer) -> None:
    """
    给每个 block 的 attn/mlp 输出投影乘一个确定性系数,制造重要性高低差。

    振幅曲线:首尾块高(模拟「基础表征 + 任务精修」),中间挑几个压得很低(模拟冗余块)。
    系数越大 => 该 block 对残差流的改动越大 => 输入输出余弦越低 => BI 越高。
    全程用固定值,保证可复现、排序确定。
    """
    n = model.n_layers
    # 基线放大系数:让首尾块「响亮」,中段整体偏低。
    amp = [1.0] * n
    for i in range(n):
        if i < 3 or i >= n - 2:
            amp[i] = 3.0            # 首 3 + 尾 2:高重要性(保护区)
        else:
            amp[i] = 0.6            # 中段:偏冗余
    # 再手动把中段的某几块压到极低(明确的「打酱油块」,便于观察被剪中)。
    for i in (n // 2, n // 2 + 2):
        if 0 <= i < n:
            amp[i] = 0.15
    with torch.no_grad():
        for i, block in enumerate(model.layers):
            block.o_proj.weight.mul_(amp[i])
            block.down_proj.weight.mul_(amp[i])


def make_toy_batch(cfg: ToyConfig, batch: int = 2, seq: int = 16, seed: int = 123) -> torch.Tensor:
    """造一批可复现的随机 token id,形状 [batch, seq]。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (batch, seq), generator=g)
