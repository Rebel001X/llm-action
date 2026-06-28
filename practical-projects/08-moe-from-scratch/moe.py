"""
从零实现稀疏 MoE(Mixture of Experts)层:top-k 路由 + 负载均衡辅助损失。

这个脚本用 PyTorch(CPU)在一个 toy 分类任务上,从零搭建一个稀疏混合专家层,
完整展示现代 MoE(GShard / Switch Transformer / Mixtral)中最核心的三件事:

  1. Router / Gating(门控路由)
     - 一个线性层为每个 token 给出 N 个专家的打分(logits),softmax 成概率;
     - 取概率最高的 top-k 个专家,只让这 k 个专家参与计算(稀疏激活),
       从而在"参数量大"的同时保持"每个 token 的计算量小"。

  2. 加权组合(weighted combine)
     - 被选中的 k 个专家各自前向,输出按 router 给出的(归一化后)门控权重加权求和。

  3. 负载均衡辅助损失(load-balancing auxiliary loss)
     - 朴素训练时 router 容易"赢者通吃":少数专家被反复选中、其余专家饿死,
       既浪费容量又难训练。我们加入 Switch Transformer 式的辅助损失,
       鼓励"被路由到各专家的样本比例 f_i"与"各专家的平均路由概率 P_i"都趋于均匀。

脚本会在训练过程中打印:
  - 任务交叉熵 loss(应当下降);
  - 每个专家被路由到的 token 占比(应当从不均衡逐渐趋于均衡);
  - 分布不均衡度指标 imbalance(max/min 占比之比,应当下降)。

设计目标:CPU 几十秒跑完、自包含、无外部数据集/网络依赖,作为教学 demo。
所有 print 仅用 ASCII(兼容 Windows GBK 控制台),中文只出现在注释/文档里。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 可复现:固定随机种子
# ---------------------------------------------------------------------------
SEED = 0
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. 单个专家(Expert):就是一个普通的两层 MLP(FFN)
#    在真实的 Transformer-MoE 里,专家就是替换掉 FFN 子层的那个前馈网络。
# ---------------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 经典 FFN: Linear -> GELU -> Linear
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 2. 稀疏 MoE 层:router 选 top-k 专家 + 加权组合 + 负载均衡辅助损失
# ---------------------------------------------------------------------------
class SparseMoE(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_experts: int, top_k: int):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k

        # router/gating:把每个 token 映射到 num_experts 个打分(无偏置更接近论文实现)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # num_experts 个独立专家
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor):
        """
        x: [num_tokens, d_model]
        返回:
          y         : [num_tokens, d_model]  MoE 层输出
          aux_loss  : 标量,负载均衡辅助损失
          route_frac: [num_experts] 本 batch 每个专家被选中的 token 占比(用于打印观察)
        """
        num_tokens = x.shape[0]

        # ---- (a) 路由打分 ----
        # 原理:router 为每个 token 对每个专家打分,softmax 得到路由概率分布。
        router_logits = self.router(x)                       # [T, E]
        router_probs = F.softmax(router_logits, dim=-1)      # [T, E],每行和为 1

        # ---- (b) top-k 选择 ----
        # 原理:稀疏激活——每个 token 只交给概率最高的 top_k 个专家处理。
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # [T, k]
        # 把 top-k 的门控权重重新归一化,使被选中的 k 个权重之和为 1
        # (这样组合输出是一个凸组合,数值更稳定,也是 Mixtral 的做法)
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # ---- (c) 各专家前向 + 加权组合 ----
        # 教学实现:为清晰起见按"专家"循环,把路由到该专家的 token 聚到一起算。
        # (真实高性能实现会用 dispatch/combine 矩阵或分组 gemm,这里以可读性优先。)
        y = torch.zeros_like(x)
        for e in range(self.num_experts):
            # 找出本 batch 中把专家 e 选进 top-k 的 token,以及它对应的 slot 位置
            sel_token, sel_slot = torch.where(topk_idx == e)  # 两个 1D 索引张量
            if sel_token.numel() == 0:
                continue  # 这个专家这一步没被任何 token 选中
            expert_in = x[sel_token]                      # 该专家要处理的 token
            expert_out = self.experts[e](expert_in)       # 专家前向
            weight = topk_weights[sel_token, sel_slot].unsqueeze(-1)  # 对应门控权重
            # 把加权输出累加回这些 token 的输出位置(同一 token 的多个专家会累加)
            y.index_add_(0, sel_token, expert_out * weight)

        # ---- (d) 负载均衡辅助损失(Switch Transformer 式) ----
        # 原理:
        #   f_i = 路由到专家 i 的 token 比例(由"硬"的 top-k 选择统计,不可导)
        #   P_i = 专家 i 在所有 token 上的平均路由概率(可导)
        #   aux = E * sum_i f_i * P_i
        # 直觉:当某专家既"被选得多(f_i 大)"又"平均概率高(P_i 大)"时惩罚增大,
        #       梯度会压低它、抬高冷门专家,从而把负载推向均匀(理想最小值=1)。
        #   f_i 用于加权(detach,不回传),真正的梯度通路在可导的 P_i 上。
        # one_hot: 每个 token 的 top-k 选择展开成 [T, E] 的 0/1 命中矩阵
        expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).sum(dim=1)  # [T, E]
        tokens_per_expert = expert_mask.sum(dim=0).float()          # [E] 每个专家命中的(token*slot)数
        route_frac = tokens_per_expert / tokens_per_expert.sum()    # 归一化成占比(打印用)
        f_i = expert_mask.float().mean(dim=0)                       # [E] 平均命中率(每 token 期望)
        P_i = router_probs.mean(dim=0)                              # [E] 平均路由概率(可导)
        aux_loss = self.num_experts * torch.sum(f_i.detach() * P_i)

        return y, aux_loss, route_frac.detach()


# ---------------------------------------------------------------------------
# 3. 一个把 MoE 当作核心层的极小分类模型
# ---------------------------------------------------------------------------
class MoEClassifier(nn.Module):
    def __init__(self, d_in, d_model, d_hidden, num_experts, top_k, num_classes):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.moe = SparseMoE(d_model, d_hidden, num_experts, top_k)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = F.gelu(self.embed(x))
        h_moe, aux_loss, route_frac = self.moe(h)
        h = h + h_moe  # 残差连接(和 Transformer 里 FFN 子层一致)
        logits = self.head(h)
        return logits, aux_loss, route_frac


# ---------------------------------------------------------------------------
# 4. 构造一个 toy 分类任务(自包含,无需任何数据集)
#    设计成"天然适合分工"的任务:数据由 num_classes 个高斯簇生成,
#    我们期望 router 学会把不同簇(子任务)分给不同专家,即"专家分化"。
# ---------------------------------------------------------------------------
def make_toy_data(num_samples, d_in, num_classes, seed=0):
    g = torch.Generator().manual_seed(seed)
    # 每个类别一个随机簇中心(彼此拉开),样本 = 中心 + 噪声
    # 簇中心拉开适中、噪声偏大 -> 类别之间有重叠,任务非平凡(loss 不会瞬间到 0)
    centers = torch.randn(num_classes, d_in, generator=g) * 2.0
    labels = torch.randint(0, num_classes, (num_samples,), generator=g)
    noise = torch.randn(num_samples, d_in, generator=g) * 1.5
    x = centers[labels] + noise
    return x, labels


def fmt_frac(frac: torch.Tensor) -> str:
    """把每个专家的占比格式化成 ASCII 字符串,如 [0.30 0.05 ...]。"""
    return "[" + " ".join(f"{v:0.2f}" for v in frac.tolist()) + "]"


def imbalance_ratio(frac: torch.Tensor) -> float:
    """不均衡度 = 最大占比 / 最小占比;完全均衡时为 1.0,越大越不均衡。"""
    mn = float(frac.min())
    mx = float(frac.max())
    return mx / mn if mn > 1e-9 else float("inf")


# ---------------------------------------------------------------------------
# 5. 训练循环 + 对照实验(有/无负载均衡损失)
# ---------------------------------------------------------------------------
def train(num_experts=8, top_k=2, aux_weight=0.0, steps=600, log_every=100, tag=""):
    torch.manual_seed(SEED)  # 每次实验同一初始化,保证可比

    # --- 超参(toy 规模,CPU 几秒级)---
    # 故意把任务调得"不那么轻松"(类别多、噪声大、模型小),
    # 这样 task_loss 会有一个肉眼可见的下降过程,而不是一两步就到 0。
    d_in = 16
    d_model = 24
    d_hidden = 48
    num_classes = 10
    num_samples = 4096
    batch_size = 256
    lr = 2e-3

    x_all, y_all = make_toy_data(num_samples, d_in, num_classes, seed=SEED)
    model = MoEClassifier(d_in, d_model, d_hidden, num_experts, top_k, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []  # 记录 (step, task_loss, acc, route_frac, imbalance)
    g = torch.Generator().manual_seed(SEED)

    for step in range(1, steps + 1):
        # 随机取一个 batch
        idx = torch.randint(0, num_samples, (batch_size,), generator=g)
        xb, yb = x_all[idx], y_all[idx]

        logits, aux_loss, route_frac = model(xb)
        task_loss = F.cross_entropy(logits, yb)
        # 总损失 = 任务损失 + aux_weight * 负载均衡损失
        loss = task_loss + aux_weight * aux_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                acc = (logits.argmax(-1) == yb).float().mean().item()
            history.append(
                (step, task_loss.item(), acc, route_frac.clone(),
                 imbalance_ratio(route_frac))
            )

    # ---- 打印该实验的训练轨迹(全英文,兼容 GBK 控制台)----
    print(f"--- experiment: {tag} "
          f"(num_experts={num_experts}, top_k={top_k}, aux_weight={aux_weight}) ---")
    print(f"{'step':>5} | {'task_loss':>9} | {'acc':>5} | {'imbalance':>9} | route_frac")
    for step, tl, acc, frac, imb in history:
        print(f"{step:>5} | {tl:>9.4f} | {acc:>5.2f} | {imb:>9.2f} | {fmt_frac(frac)}")
    final = history[-1]
    print(f"summary[{tag}]: final_task_loss={final[1]:.4f} "
          f"final_acc={final[2]:.2f} final_imbalance={final[4]:.2f}")
    print()
    return history


def main():
    print("=" * 70)
    print("Sparse MoE from scratch: top-k routing + load-balancing aux loss")
    print("=" * 70)
    print()

    # 实验 A:不加负载均衡损失 —— 观察 router 趋向"赢者通吃",负载不均衡
    hist_no_aux = train(aux_weight=0.0, tag="no_aux")

    # 实验 B:加入负载均衡损失 —— 观察负载被推向均衡,且任务仍学得好
    hist_aux = train(aux_weight=0.01, tag="with_aux")

    # ---- 结论对比:辅助损失是否真的改善了负载均衡 ----
    imb_no = hist_no_aux[-1][4]
    imb_aux = hist_aux[-1][4]
    loss_no = hist_no_aux[-1][1]
    loss_aux = hist_aux[-1][1]
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"final imbalance  : no_aux={imb_no:.2f}  with_aux={imb_aux:.2f}  "
          f"(lower is more balanced)")
    print(f"final task_loss  : no_aux={loss_no:.4f}  with_aux={loss_aux:.4f}  "
          f"(both should be low -> task still learned)")
    if imb_aux < imb_no:
        print("RESULT: PASS -- load-balancing aux loss reduced expert imbalance.")
    else:
        print("RESULT: CHECK -- aux loss did not reduce imbalance this run.")

    # ---- 可选:画一张"各专家负载随训练变化"的图(失败则跳过,不崩)----
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面后端,直接存 png
        import matplotlib.pyplot as plt

        steps_x = [h[0] for h in hist_aux]
        fracs = torch.stack([h[3] for h in hist_aux])  # [num_logs, num_experts]
        plt.figure(figsize=(7, 4))
        for e in range(fracs.shape[1]):
            plt.plot(steps_x, fracs[:, e].tolist(), label=f"expert {e}")
        plt.axhline(1.0 / fracs.shape[1], ls="--", c="k", lw=1, label="uniform")
        plt.xlabel("step")
        plt.ylabel("route fraction")
        plt.title("Per-expert load over training (with aux loss)")
        plt.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        out = "expert_load.png"
        plt.savefig(out, dpi=120)
        print(f"[plot] saved per-expert load curve to {out}")
    except Exception as ex:  # noqa: BLE001 - 教学脚本,画图失败不应中断
        print(f"[plot] skipped (matplotlib unavailable: {type(ex).__name__})")


if __name__ == "__main__":
    main()
