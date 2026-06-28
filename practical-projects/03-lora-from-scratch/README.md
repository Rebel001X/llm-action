# 从零实现 LoRA 参数高效微调(CPU 可跑通)

用 PyTorch 从零实现 **LoRA(Low-Rank Adaptation)**,并在一个人造分类任务上,直观对比
**冻结(frozen)/ LoRA / 全量微调(full fine-tune)** 三种方式的「可训练参数量」与「效果」。
全程纯 CPU、toy 规模,几秒钟跑完,不依赖任何外部数据集 / 网络 / 预训练权重。

> 这是 `llm-action` 仓库少有的「动手可跑」实战代码,配合理论文档食用更佳。

---

## 一、演示什么原理

对一个已经预训练好的线性层权重 `W0`(维度 `d_out x d_in`),微调时**不直接更新 W0**,
而是冻结 `W0`,旁路加上一个**低秩增量**:

```
W_eff = W0 + ΔW,   ΔW = (alpha / r) * B @ A
```

- `A : r x d_in`  —— 低秩矩阵,小高斯初始化(kaiming)
- `B : d_out x r` —— 低秩矩阵,**初始化为 0**,保证训练起点 `ΔW = 0`(模型行为 == 原始 base,稳定)
- `r` —— 秩(rank),`r << min(d_in, d_out)`
- `alpha / r` —— 缩放系数,解耦「秩大小」与「更新幅度」

前向计算高效实现为(不显式构造 `d×d` 大矩阵):

```
y = x @ W0^T + (alpha/r) * (x @ A^T) @ B^T
    └ 冻结主干 ┘   └────────── 可训练低秩旁路 ──────────┘
```

**核心收益**:可训练参数从 `d_out*d_in` 降为 `r*(d_in + d_out)`。模型越深、矩阵越大,
这个旁路占比越小(真实 7B 模型上常见 0.1%~1%)。

demo 的流程:
1. 先「预训练」一个小 MLP 作为 base 模型(在任务 A 上),然后**冻结**;
2. 把它放到一个**有分布漂移**的下游任务 B 上(直接用 base 效果很差);
3. 三种方式适配任务 B:`frozen`(不训,下界) / `lora`(只训 A、B) / `full`(全量,上界参考);
4. 打印各自的可训练参数占比与任务 B 准确率/loss,并验证「LoRA 增量可合并回主干」数值等价。

---

## 二、怎么跑

环境:Python 3.13 / numpy 2.3 / torch 2.12 (CPU)。

```bash
cd practical-projects/03-lora-from-scratch
python lora.py
```

跑完会在当前目录生成柱状图 `lora_results.png`(matplotlib 不可用时自动跳过、不报错)。

---

## 三、预期输出(真实跑通摘录)

```
[Phase 0] pretrain base model on task A ...
    base on task A : acc=1.000 loss=0.0000  (good, as expected)
    base on task B : acc=0.156 loss=15.1037  (poor -> needs adapt)
    base total params = 18696

SUMMARY (task B test set)
----------------------------------------------------------------
method           trainable  %trainable     acc     loss
frozen                   0       0.00%   0.156  15.1037
lora                  2336      11.11%   0.994   0.0056
full                 18696     100.00%   1.000   0.0000
----------------------------------------------------------------
LoRA trains 2336 params vs full 18696  -> 8.0x fewer trainable params
acc lift over frozen baseline: +0.837 (lora), +0.844 (full)
LoRA recovered 99.3% of the frozen->full accuracy gap
merge check (max |unmerged - merged|): 3.81e-05  (~0 => merged weight is exact)
================================================================
CHECK trainable<30% of full : True
CHECK lora beats frozen     : True
CHECK merge is exact        : True
RESULT: PASS
```

**结论一眼看穿**:

- frozen 直接拿 base 用,任务 B 上只有 **0.156** 准确率(分布漂移,base 失效);
- LoRA 只训练 **2336 个参数(比 full 少 8 倍)**,把准确率拉到 **0.994**,
  **补回了 frozen→full 准确率差距的 99.3%** —— 用极少参数逼近全量微调;
- `merge check ≈ 3.8e-5`,证明 `W_eff = W0 + (alpha/r)·B@A` 合并回主干后前向**数值等价**,
  对应「LoRA 部署时可合并、零额外推理开销」。

---

## 四、对应 llm-action 文档

- LoRA / QLoRA 原理:[`../../llm-train/peft/LoRA-QLoRA.md`](../../llm-train/peft/LoRA-QLoRA.md)
- PEFT 总览与 API:[`../../llm-train/peft`](../../llm-train/peft) · [`../../llm-train/peft/PEFT-API.md`](../../llm-train/peft/PEFT-API.md)
- 其它 PEFT 方法对照:[`../../llm-train/peft/Prefix-Tuning.md`](../../llm-train/peft/Prefix-Tuning.md) · [`../../llm-train/peft/Prompt-Tuning.md`](../../llm-train/peft/Prompt-Tuning.md)
- 工程化 LoRA 微调示例:[`../../llm-train/alpaca-lora`](../../llm-train/alpaca-lora) · [`../../llm-train/qlora`](../../llm-train/qlora) · [`../../llm-train/chatglm-lora`](../../llm-train/chatglm-lora)

---

## 五、社区参考

- [LoRA 论文 arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- [HuggingFace PEFT](https://github.com/huggingface/peft)

---

## 六、局限 / 与真实工程的差异

1. **占比不是 <1%**:本 demo 只有 2 个线性层,LoRA 占比天然在 ~10%。真实 LoRA 的「<1%」
   来自**深层 transformer**(几十~上百个大权重矩阵,且通常只把 LoRA 注入 q/v 投影),
   同样的 rank-r 旁路占比会摊薄到极小。这里追求的是「同一结论、最小可跑」的演示。
2. **任务是人造高斯混合**,不是真实文本;base 也不是真预训练大模型,只是「在任务 A 上训出的小 MLP」。
   目的是隔离出 LoRA 机制本身,避免下载权重 / 数据,保证 CPU 秒级可复现。
3. **注入位置简化**:demo 直接替换全部 `nn.Linear`。真实工程通常按 `target_modules`
   只注入注意力的部分投影,并涉及 `lora_dropout`、按层不同 rank 等更精细的配置(见 PEFT)。
4. **未含量化**:没有 QLoRA 的 4-bit NF4 量化主干 + LoRA 训练(显存大杀器),这部分见
   `../../llm-train/qlora` 与 `LoRA-QLoRA.md`。
5. **优化器状态**:LoRA 省的不只是参数,还有 Adam 的优化器状态(m、v)显存。demo 规模太小,
   未专门量化这块收益。
6. 任务较易、训练步数充足时 loss 会迅速降到 ~0;这是 toy 任务的特征,真实任务不会这么干净。

> 一句话:这份代码教的是 **LoRA「冻结主干 + 低秩可训练旁路 + 可合并」** 的机制骨架,
> 是理解 PEFT/QLoRA 工程实现的最小起点。
