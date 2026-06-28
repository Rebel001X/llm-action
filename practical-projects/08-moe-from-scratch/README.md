# 从零实现 MoE(top-k 路由 + 负载均衡)

一个 **CPU 几十秒就能跑通** 的稀疏混合专家(Mixture of Experts, MoE)最小实现。
用纯 PyTorch 把现代 MoE(GShard / Switch Transformer / Mixtral)里最核心的三件事讲清楚:
**门控路由(router)→ top-k 稀疏激活 → 负载均衡辅助损失**。

> 教学定位:不追求性能,只追求"看得懂、跑得通、看得出原理"。
> 代码里中文注释密集,所有 `print` 仅用 ASCII(兼容 Windows GBK 控制台)。

---

## 一、演示什么原理

稀疏 MoE 的思路是:用很多个"专家"(每个专家就是一个 FFN)堆出巨大的参数量,
但每个 token 只激活其中很少几个专家,从而做到 **参数量大、单 token 计算量小**。

本项目实现并可视化三个关键机制:

1. **Router / Gating(门控路由)**
   一个线性层为每个 token 给出 N 个专家的打分,`softmax` 成路由概率。

2. **top-k 稀疏激活 + 加权组合**
   每个 token 只选概率最高的 top-k(本例 k=2)个专家参与计算,
   输出按(重新归一化后的)门控权重加权求和。

3. **负载均衡辅助损失(load-balancing aux loss)**
   朴素训练时 router 容易"赢者通吃":少数专家被反复选中、其余专家**饿死**(占比变 0)。
   加入 Switch Transformer 式辅助损失
   `aux = E * Σ_i f_i · P_i`(`f_i`=被路由占比,`P_i`=平均路由概率),
   把各专家负载推向均匀。

为了让"分工"自然发生,toy 任务用 10 个高斯簇生成(类别间有重叠),
我们期望 router 学会把不同簇分给不同专家(**专家分化**),并通过辅助损失避免负载塌缩。

---

## 二、怎么跑

环境:Python 3.13 / numpy 2.3 / torch 2.12 (CPU)。无需任何外部数据集 / 网络。

```bash
cd practical-projects/08-moe-from-scratch
python moe.py
```

脚本会跑两个对照实验:`no_aux`(不加负载均衡损失)与 `with_aux`(加),
并在最后给出结论;若装了 matplotlib,会额外存一张 `expert_load.png`
(各专家负载随训练变化曲线),没装也不会报错。

---

## 三、预期输出(真实跑通,CPU ~22s)

```text
--- experiment: no_aux (num_experts=8, top_k=2, aux_weight=0.0) ---
 step | task_loss |   acc | imbalance | route_frac
    1 |    2.5922 |  0.10 |     15.00 | [0.02 0.06 0.05 0.23 0.13 0.03 0.15 0.32]
  100 |    0.0264 |  0.99 |       inf | [0.00 0.00 0.08 0.18 0.30 0.07 0.17 0.19]
  600 |    0.0021 |  1.00 |       inf | [0.01 0.00 0.11 0.17 0.30 0.14 0.13 0.15]
summary[no_aux]: final_task_loss=0.0021 final_acc=1.00 final_imbalance=inf

--- experiment: with_aux (num_experts=8, top_k=2, aux_weight=0.01) ---
 step | task_loss |   acc | imbalance | route_frac
    1 |    2.5922 |  0.10 |     15.00 | [0.02 0.06 0.05 0.23 0.13 0.03 0.15 0.32]
  100 |    0.0302 |  0.99 |      2.97 | [0.09 0.11 0.13 0.15 0.21 0.07 0.11 0.12]
  200 |    0.0118 |  1.00 |      2.09 | [0.14 0.10 0.13 0.18 0.18 0.10 0.09 0.08]
  300 |    0.0078 |  1.00 |      1.69 | [0.10 0.10 0.15 0.15 0.17 0.12 0.10 0.11]
  600 |    0.0019 |  1.00 |      1.24 | [0.12 0.11 0.14 0.12 0.12 0.13 0.13 0.12]
summary[with_aux]: final_task_loss=0.0019 final_acc=1.00 final_imbalance=1.24

CONCLUSION
final imbalance  : no_aux=inf  with_aux=1.24  (lower is more balanced)
final task_loss  : no_aux=0.0021  with_aux=0.0019  (both should be low -> task still learned)
RESULT: PASS -- load-balancing aux loss reduced expert imbalance.
```

**怎么看懂这几行(成功信号):**

- **任务在学**:`task_loss` 从 2.59(acc≈0.10,10 类随机水平)降到 ≈0.002(acc=1.00)。
- **不加 aux → 负载塌缩**:`no_aux` 里有 2 个专家占比变成 `0.00`(被饿死),
  `imbalance`(最大占比 / 最小占比)永远是 `inf`。
- **加 aux → 趋于均衡**:`with_aux` 里 `imbalance` 单调下降
  `2.97 → 2.09 → 1.69 → 1.45 → 1.24`,各专家占比都逼近理想均匀值 `1/8 = 0.125`,
  **而且任务 loss 一样低** —— 说明均衡不是靠牺牲任务效果换来的。

`route_frac` 是当前 batch 里每个专家被路由到的 token 占比;
对比两个实验从 `[..0.00 0.00..]` 到 `[..0.12 0.13..]` 的变化,就能直观看到"从不均衡到均衡"。

---

## 四、对应 llm-action 文档

- MoE 原理:[`../../llm-algo/moe`](../../llm-algo/moe)
- Mixtral(稀疏 MoE 的代表作):[`../../llm-algo/mixtral`](../../llm-algo/mixtral)
- 普通 FFN / MLP(专家的本体):[`../../llm-algo/mlp.md`](../../llm-algo/mlp.md)
- Transformer 架构(MoE 替换的是 FFN 子层):[`../../llm-algo/transformer.md`](../../llm-algo/transformer.md)

---

## 五、社区参考

- [Switch Transformers: Scaling to Trillion Parameter Models (arXiv:2101.03961)](https://arxiv.org/abs/2101.03961)
  —— 本项目负载均衡辅助损失 `aux = E·Σ f_i·P_i` 即来自此文(Sec. 2.2)。
- [GShard (arXiv:2006.16668)](https://arxiv.org/abs/2006.16668) —— top-2 路由 + 专家并行的奠基工作。
- [Mixtral of Experts (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088)
  —— top-2 稀疏 MoE 的开源代表;本项目"top-k 权重重新归一化"的做法与之一致。

---

## 六、局限 / 与真实工程的差异

| 维度 | 本教学实现 | 真实工程(Switch / Mixtral / Megatron-MoE 等) |
| --- | --- | --- |
| 计算方式 | 按专家 Python for 循环、`index_add_` 组合,可读优先 | dispatch/combine 矩阵或分组 GEMM,GPU 上批量并行 |
| 容量限制 | 无 expert capacity / token dropping | 设每专家容量上限,溢出 token 被丢弃或走残差 |
| 并行 | 单进程 CPU | 专家并行(EP)+ all-to-all 跨设备通信 |
| 路由稳定性 | 仅 aux loss | 还有 router z-loss、noisy top-k、jitter 等 |
| 规模 | 8 专家 / d=24 / toy 高斯数据 | 数十~数百专家、真实语料、十亿~万亿参数 |
| 任务 | 高斯簇分类(易收敛) | 语言建模,loss 不会到 0 |

简言之:本项目把 MoE 的"思想骨架"完整跑通,但刻意省略了让它在大规模上高效、稳定所需的
**容量管理、跨设备 all-to-all 通信、以及多种正则项**。要把它放进真实 Transformer,
需要把 for 循环换成批量 dispatch、加上 expert capacity,并接入专家并行框架。
