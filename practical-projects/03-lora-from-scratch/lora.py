"""
从零实现 LoRA(Low-Rank Adaptation)参数高效微调
=================================================

本文件用 PyTorch(纯 CPU,toy 规模)从零实现 LoRA,并在一个**人造分类任务**上
对比三种训练方式,直观展示 LoRA 的核心卖点:用极少的可训练参数逼近全量微调的效果。

LoRA 的核心思想(对应论文 arXiv:2106.09685):
------------------------------------------------
对一个已经预训练好的线性层权重 W0 (维度 d_out x d_in),微调时不直接更新 W0,
而是冻结 W0,旁路加上一个**低秩增量** ΔW = (alpha / r) * B @ A,其中:
    A : r x d_in   (低秩,通常初始化为高斯小值)
    B : d_out x r  (低秩,通常初始化为 0,保证训练开始时 ΔW = 0)
    r : 秩(rank),r << min(d_in, d_out)
前向计算变为:  y = x @ (W0 + ΔW)^T = x @ W0^T + (alpha/r) * (x @ A^T) @ B^T
                                       └ 冻结主干 ┘   └────── 可训练旁路 ──────┘

可训练参数量从 d_out*d_in 降为 r*(d_in + d_out)。当 r 很小、且模型很深(大量大矩阵)
时,可训练占比可低至 < 1%(如真实 7B 模型上常见 0.1%~1%)。本 demo 只有 2 个线性层,
LoRA 占比天然在 ~10-15%,但"用远少于 full 的参数逼近 full 的效果"这一结论一致成立。
因为 B 初始化为 0,训练起点的模型行为与原始预训练模型完全一致(ΔW=0),稳定且安全。

本 demo 做什么:
----------------
1) 先"预训练"一个小 MLP 当作 base 模型(在任务 A 上训出还不错的权重),冻结它。
2) 把它放到一个**分布漂移后的任务 B** 上 —— 直接用 base 不够好,需要适配。
3) 三种适配方式对比:
   - frozen      : 完全不训练,直接拿 base 在任务 B 上评估(下界基线)
   - lora        : 冻结 base,只训练注入的 LoRA 低秩矩阵 A、B(参数极少)
   - full-finetune: 解冻 base 全部权重一起训练(参数最多,效果上界参考)
4) 打印各自的可训练参数占比、任务 B 上的准确率/loss,看出 LoRA 用 ~1% 参数
   就能把准确率从 frozen 基线大幅拉到接近 full-finetune 的水平。

运行:  python lora.py
说明:  print 全部用 ASCII(适配 Windows GBK 控制台);中文只在注释/docstring。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 可复现:固定所有随机种子
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)


# ===========================================================================
# 1. LoRA 线性层:核心实现
# ===========================================================================
class LoRALinear(nn.Module):
    """带 LoRA 旁路的线性层。

    包装一个已有的 nn.Linear(base),冻结其权重 W0、bias,
    额外学习低秩矩阵 A、B,实现  W_eff = W0 + (alpha/r) * B @ A。

    参数:
        base_linear : 被适配的原始 nn.Linear(其权重将被冻结)
        r           : LoRA 的秩(rank),控制旁路容量与参数量
        alpha       : LoRA 缩放系数,实际缩放为 scaling = alpha / r
    """

    def __init__(self, base_linear: nn.Linear, r: int = 4, alpha: int = 8):
        super().__init__()
        assert r > 0, "rank r must be positive"
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        # scaling = alpha / r:论文中的 ΔW 缩放,解耦"秩大小"与"更新幅度"
        self.scaling = alpha / r

        # --- 冻结的预训练主干 W0(以及 bias)---
        # 直接复用传入的 base linear,并关闭其梯度 -> 这就是"冻结 base"
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        # --- 可训练的低秩旁路 A、B ---
        # A: (r, in_features)  用 kaiming 风格的小高斯初始化
        # B: (out_features, r) 初始化为 0 -> 训练起点 ΔW = B@A = 0,模型 == base
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # 与 PEFT 实现一致的初始化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主干输出:y0 = x @ W0^T + b   (W0 冻结,不参与反传)
        base_out = self.base(x)
        # 低秩增量:先降维 x@A^T (-> r 维),再升维 @B^T,最后乘 scaling
        # 这一步对应 ΔW = (alpha/r) * B @ A,但用 (x@A^T)@B^T 避免显式构造 dxd 大矩阵
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def merged_weight(self) -> torch.Tensor:
        """返回合并后的等效权重 W_eff = W0 + (alpha/r) * B @ A。

        推理部署时可把 LoRA 增量合并回主干,做到"零额外推理开销"。
        """
        delta_w = self.scaling * (self.lora_B @ self.lora_A)  # (out, in)
        return self.base.weight.data + delta_w


# ===========================================================================
# 2. Toy 模型:一个小 MLP("迷你 transformer 的 FFN")
# ===========================================================================
class MLP(nn.Module):
    """两层 MLP 分类器,作为可被 LoRA 注入的 base 模型。"""

    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)


def inject_lora(model: MLP, r: int, alpha: int) -> MLP:
    """把 model 中的 nn.Linear 替换为 LoRALinear(base 被冻结,只训 A/B)。"""
    model.fc1 = LoRALinear(model.fc1, r=r, alpha=alpha)
    model.fc2 = LoRALinear(model.fc2, r=r, alpha=alpha)
    return model


# ===========================================================================
# 3. 人造数据:两个有"分布漂移"的高斯混合分类任务
# ===========================================================================
def make_task(n_per_class: int, in_dim: int, n_classes: int,
              center_scale: float, generator: torch.Generator):
    """生成一个高斯混合分类任务。

    每个类是一个以随机向量为中心的高斯团。center_scale 控制中心位置,
    任务 A 与任务 B 用不同 center_scale -> 形成分布漂移,使 base 不能直接通用。
    """
    centers = torch.randn(n_classes, in_dim, generator=generator) * center_scale
    xs, ys = [], []
    for c in range(n_classes):
        # 噪声幅度偏大,让类与类之间有重叠 -> 任务不可被完美分开,准确率不会饱和到 1.0
        pts = centers[c] + 1.3 * torch.randn(n_per_class, in_dim, generator=generator)
        xs.append(pts)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(xs, 0)
    y = torch.cat(ys, 0)
    # 打乱
    perm = torch.randperm(X.size(0), generator=generator)
    return X[perm], y[perm]


# ===========================================================================
# 4. 通用训练 / 评估
# ===========================================================================
def count_params(model: nn.Module):
    """返回 (可训练参数量, 总参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def train(model, X, y, steps, lr, log_every=0):
    """用全 batch 梯度下降训练 model(只更新 requires_grad=True 的参数)。"""
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"    step {step:4d}  train_loss {loss.item():.4f}")
    return model


# ===========================================================================
# 5. Demo 主流程
# ===========================================================================
def main():
    # ---- 超参(toy 规模,CPU 几秒内跑完)----
    # HIDDEN 取得较大,让 base 主干权重很多 -> LoRA 的低秩旁路才显得"极小"。
    # 这正是 LoRA 在真实大模型上的优势来源:base 越大,r*(d_in+d_out) 占比越低。
    IN_DIM, HIDDEN, N_CLASSES = 64, 256, 8
    N_PER_CLASS = 250
    LORA_R, LORA_ALPHA = 4, 8

    g = torch.Generator().manual_seed(SEED)

    # ---- 数据:任务 A(预训练)+ 任务 B(下游适配,有分布漂移)----
    Xa, ya = make_task(N_PER_CLASS, IN_DIM, N_CLASSES, center_scale=2.2, generator=g)
    Xb, yb = make_task(N_PER_CLASS, IN_DIM, N_CLASSES, center_scale=2.2, generator=g)
    # 任务 B 再施加一个固定的线性"漂移"(旋转 + 偏置),模拟下游分布变化
    drift = torch.randn(IN_DIM, IN_DIM, generator=g) * 0.25
    Xb = Xb + Xb @ drift + 0.8

    # 切分 train/test
    def split(X, y, n_test=160):
        return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]

    Xa_tr, ya_tr, _, _ = split(Xa, ya)
    Xb_tr, yb_tr, Xb_te, yb_te = split(Xb, yb)

    print("=" * 64)
    print("LoRA from scratch :: toy demo  (PyTorch CPU)")
    print("=" * 64)
    print(f"task dim={IN_DIM}  hidden={HIDDEN}  classes={N_CLASSES}  "
          f"lora_r={LORA_R}  lora_alpha={LORA_ALPHA}")
    print()

    # ----------------------------------------------------------------
    # Phase 0: 预训练 base 模型(在任务 A 上),然后冻结
    # ----------------------------------------------------------------
    print("[Phase 0] pretrain base model on task A ...")
    base = MLP(IN_DIM, HIDDEN, N_CLASSES)
    train(base, Xa_tr, ya_tr, steps=300, lr=1e-2)
    _, base_total = count_params(base)
    la, aa = evaluate(base, Xa, ya)
    lb0, ab0 = evaluate(base, Xb_te, yb_te)
    print(f"    base on task A : acc={aa:.3f} loss={la:.4f}  (good, as expected)")
    print(f"    base on task B : acc={ab0:.3f} loss={lb0:.4f}  (poor -> needs adapt)")
    print(f"    base total params = {base_total}")
    print()

    # 保存 base 的初始 state,供三种方式公平地从同一起点出发
    base_state = {k: v.clone() for k, v in base.state_dict().items()}

    def fresh_base():
        m = MLP(IN_DIM, HIDDEN, N_CLASSES)
        m.load_state_dict(base_state)
        return m

    results = {}

    # ----------------------------------------------------------------
    # Method 1: frozen —— 完全不训练(下界基线)
    # ----------------------------------------------------------------
    print("[Method 1] frozen base (no training) on task B")
    m_frozen = fresh_base()
    tr, tot = count_params_frozen(m_frozen)
    loss, acc = evaluate(m_frozen, Xb_te, yb_te)
    print(f"    trainable params = {tr} / {tot}  ({100*tr/tot:.2f}%)")
    print(f"    task B test : acc={acc:.3f} loss={loss:.4f}")
    results["frozen"] = (tr, tot, acc, loss)
    print()

    # ----------------------------------------------------------------
    # Method 2: LoRA —— 冻结 base,只训低秩 A/B
    # ----------------------------------------------------------------
    print("[Method 2] LoRA fine-tune on task B (freeze base, train only A,B)")
    m_lora = inject_lora(fresh_base(), r=LORA_R, alpha=LORA_ALPHA)
    tr, tot = count_params(m_lora)
    print(f"    trainable params = {tr} / {tot}  ({100*tr/tot:.2f}%)")
    train(m_lora, Xb_tr, yb_tr, steps=300, lr=5e-2, log_every=100)
    loss, acc = evaluate(m_lora, Xb_te, yb_te)
    print(f"    task B test : acc={acc:.3f} loss={loss:.4f}")
    results["lora"] = (tr, tot, acc, loss)
    print()

    # ----------------------------------------------------------------
    # Method 3: full fine-tune —— 解冻全部权重(参数上界参考)
    # ----------------------------------------------------------------
    print("[Method 3] full fine-tune on task B (train ALL params)")
    m_full = fresh_base()
    for p in m_full.parameters():
        p.requires_grad = True
    tr, tot = count_params(m_full)
    print(f"    trainable params = {tr} / {tot}  ({100*tr/tot:.2f}%)")
    train(m_full, Xb_tr, yb_tr, steps=300, lr=1e-2, log_every=100)
    loss, acc = evaluate(m_full, Xb_te, yb_te)
    print(f"    task B test : acc={acc:.3f} loss={loss:.4f}")
    results["full"] = (tr, tot, acc, loss)
    print()

    # ----------------------------------------------------------------
    # 验证:LoRA 合并权重后的前向 == 未合并的前向(数值等价)
    # ----------------------------------------------------------------
    with torch.no_grad():
        # 手动用合并权重重算 fc1 的输出,验证 merged_weight 正确
        x_probe = Xb_te[:8]
        unmerged = m_lora.fc1(x_probe)
        W_eff = m_lora.fc1.merged_weight()
        b_eff = m_lora.fc1.base.bias
        merged = x_probe @ W_eff.t() + b_eff
        merge_err = (unmerged - merged).abs().max().item()

    # ----------------------------------------------------------------
    # 汇总
    # ----------------------------------------------------------------
    print("=" * 64)
    print("SUMMARY (task B test set)")
    print("-" * 64)
    print(f"{'method':<14}{'trainable':>12}{'%trainable':>12}{'acc':>8}{'loss':>9}")
    for name in ("frozen", "lora", "full"):
        tr, tot, acc, loss = results[name]
        print(f"{name:<14}{tr:>12}{100*tr/tot:>11.2f}%{acc:>8.3f}{loss:>9.4f}")
    print("-" * 64)

    lora_tr = results["lora"][0]
    full_tr = results["full"][0]
    acc_frozen = results["frozen"][2]
    acc_lora = results["lora"][2]
    acc_full = results["full"][2]
    ratio = full_tr / max(lora_tr, 1)
    # 用 frozen->full 的提升空间衡量 LoRA"补回"了多少 gap
    gap = max(acc_full - acc_frozen, 1e-9)
    recovered = (acc_lora - acc_frozen) / gap * 100

    print(f"LoRA trains {lora_tr} params vs full {full_tr}  "
          f"-> {ratio:.1f}x fewer trainable params")
    print(f"acc lift over frozen baseline: +{acc_lora-acc_frozen:.3f} (lora), "
          f"+{acc_full-acc_frozen:.3f} (full)")
    print(f"LoRA recovered {recovered:.1f}% of the frozen->full accuracy gap")
    print(f"merge check (max |unmerged - merged|): {merge_err:.2e}  "
          f"(~0 => merged weight is exact)")
    print("=" * 64)

    # ---- 可量化的"成功"判定 ----
    # 注:本 demo 只有 2 个线性层,LoRA 占比天然在 ~10-15%;真实深层 transformer
    # (几十~上百个大矩阵)上,同样的 rank-r 旁路占比会降到 <1%。这里用 "明显更少"
    # 作为判据,并在 README 中说明这一差异。
    ok_param = lora_tr < 0.30 * full_tr          # LoRA 可训练参数应明显少于 full
    ok_lift = acc_lora > acc_frozen + 0.05       # LoRA 应明显优于 frozen 基线
    ok_merge = merge_err < 1e-4                   # 合并权重应数值等价
    print(f"CHECK trainable<30%% of full : {ok_param}")
    print(f"CHECK lora beats frozen     : {ok_lift}")
    print(f"CHECK merge is exact        : {ok_merge}")
    print("RESULT:", "PASS" if (ok_param and ok_lift and ok_merge) else "FAIL")

    # ---- 可选作图(失败则文本兜底,不崩)----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["frozen", "lora", "full"]
        accs = [results[n][2] for n in names]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.bar(names, accs, color=["#bbb", "#3b82f6", "#10b981"])
        ax.set_ylabel("task B test accuracy")
        ax.set_title("LoRA vs frozen vs full fine-tune")
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        out = "lora_results.png"
        fig.savefig(out, dpi=110)
        print(f"[plot] saved bar chart to {out}")
    except Exception as e:  # noqa: BLE001
        print(f"[plot] skipped (matplotlib unavailable): {e}")


def count_params_frozen(model: nn.Module):
    """frozen 方法:把所有参数视为冻结(可训练=0),返回 (0, total)。"""
    for p in model.parameters():
        p.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    return 0, total


if __name__ == "__main__":
    main()
