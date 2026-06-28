# MoE 并行框架（Expert Parallelism 的工程实现）

> 把 Mixture-of-Experts（MoE）模型在多卡/多机上跑起来的核心机制——专家并行（EP）、All-to-All 通信、容量与负载均衡——以及 DeepSpeed-MoE / Megatron / ColossalAI / Paddle 等框架如何落地。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/moe/README]] · [[llm-algo/mlp]] · [[ai-infra/网络/集合通信原语]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[llm-optimizer/计算通信重叠]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|---|---|---|
| 0 | 一句话锚点：MoE 并行 = "稀疏激活 + 把专家摊到多卡" | EP / All-to-All |
| 1 | 地基：Dense MLP → MoE 层、Router、Top-k | Gate / 稀疏 |
| 2 | 为什么要专家并行（显存账） | 参数量爆炸 |
| 3 | Expert Parallel 的数据流（4 步 + ASCII） | dispatch/combine |
| 4 | All-to-All 通信量逐数手算 | $2bsh$ |
| 5 | 容量因子 capacity factor 与 token 丢弃 | drop / pad |
| 6 | 负载均衡 loss 与路由抖动 | aux loss |
| 7 | EP × DP × TP × PP 五维混合并行 | 3D+EP |
| 8 | 框架对照：DeepSpeed / Megatron / Colossal / Paddle | API |
| 数值 | 一个 Mixtral 量级的端到端手算 | 显存/通信 |
| FAQ | 易错点速查 | — |

---

## 0. 一句话锚点

**MoE 并行**：每个 MoE 层有 $E$ 个专家（各是一个独立 FFN），每个 token 只被路由到其中 Top-$k$ 个（通常 $k{=}1$ 或 $2$）。于是**参数量随 $E$ 线性增长，但每 token 的计算量几乎不变**——这就是"稀疏激活"。

工程上的核心矛盾：$E$ 个专家的参数放不进一张卡，于是把它们**摊到不同 GPU**（Expert Parallelism, EP）。但 token 在哪张卡、专家在哪张卡是错位的，必须用 **All-to-All** 把 token "寄送"到专家所在卡、算完再"寄回"。

> 记忆锚：**Dense = 人人算所有层；MoE = 每个 token 挑几个专家算；MoE 并行 = 专家分散在各卡，token 坐 All-to-All 这趟"快递"往返。**

---

## 1. 地基：从 Dense MLP 到 MoE 层

Transformer 的 FFN（MLP）子层：$y = W_2\,\sigma(W_1 x)$，隐藏维 $h$，中间维 $d_{ff}=4h$。

MoE 把这一个 FFN 替换成 $E$ 个**结构相同、参数不同**的 FFN（专家），加一个**路由器 Router / Gate**：

```
                 ┌──────────── MoE 层 ────────────┐
   token x ──►  Router(Wg)  ──► softmax ──► Top-k 选择
   (维度 h)        │                          │
                  │      g = softmax(Wg·x)    │  选出 k 个专家 id + 权重
                  ▼                          ▼
            ┌─────────────────────────────────────────┐
            │  Expert_0  Expert_1  ...  Expert_{E-1}   │  每个 = 一个 FFN
            └─────────────────────────────────────────┘
                  │  只激活被选中的 k 个
                  ▼
   y = Σ_{i∈Topk} g_i · Expert_i(x)     ← 加权求和
```

- **Router**：一个小线性层 $W_g\in\mathbb{R}^{h\times E}$，输出每个专家的打分 $g=\text{softmax}(W_g x)$。
- **Top-k**：取分数最高的 $k$ 个专家，$k{=}2$ 是 GShard/Switch 之后的常见配置（Switch 用 $k{=}1$）。
- **输出**：被选专家输出的加权和（权重 = router 分数）。

> 详细的门控、Switch/GShard 区别见 [[llm-algo/moe/README]]；FFN 本身见 [[llm-algo/mlp]]。

---

## 2. 为什么必须专家并行：先算一笔显存账

设 $h{=}4096$，$d_{ff}{=}4h{=}16384$，单个专家 FFN 参数：

$$P_{expert}=2\,h\,d_{ff}=2\times4096\times16384\approx1.34\times10^8\approx0.134\text{B}$$

若 $E{=}64$ 个专家、共 32 层全是 MoE：

$$P_{moe}=64\times0.134\text{B}\times32\approx 274\text{B 参数}$$

FP16 下仅专家权重 $\approx274\text{B}\times2\text{B}=548\text{ GB}$，加上 Adam 优化器状态（fp32 动量+方差+主权重，约 $\times$ 8 字节/参数）$\approx 2.2$ TB——**单卡 80GB 装不下任何一层全部专家**。

**结论**：必须把 $E$ 个专家拆到多张卡上。这就是 Expert Parallelism。

```
  EP=4 时，64 个专家平均分到 4 张卡：
  GPU0: E0..E15   GPU1: E16..E31   GPU2: E32..E47   GPU3: E48..E63
  每卡只存 16 个专家 → 单卡专家显存 548GB/4 ≈ 137GB（仍需配合 ZeRO/TP，见第7节）
```

---

## 3. Expert Parallel 的数据流：dispatch → 算 → combine

EP 的并行组：`ep_size` 张卡构成一个 expert-parallel group，$E$ 个专家**均匀分布**在组内（每卡 $E/\text{ep\_size}$ 个）。每张卡同时也持有自己那份 token（来自 DP/序列切分）。

四步数据流（核心是两次 All-to-All）：

```
 ┌── 每卡本地有 N 个 token，各自被 Router 选了目标专家 ──┐
 │                                                        │
 │  Step1 Gate+Permute：按"目标专家所在的卡"对本地       │
 │        token 重排分桶（local sort by dest-rank）       │
 │                                                        │
 │  Step2 All-to-All (dispatch)：把 token 发往专家所在卡  │
 │        ┌────────┐                                      │
 │  GPU0 ─┤ a2a    ├─► token 落到"专家在哪卡"             │
 │  GPU1 ─┤        ├─►                                    │
 │  GPU2 ─┤ 每卡把 ├─► 收到的都是"我负责的专家"的 token   │
 │  GPU3 ─┤各桶发出├─►                                    │
 │        └────────┘                                      │
 │                                                        │
 │  Step3 Expert 计算：本地专家对收到的 token 做 FFN      │
 │                                                        │
 │  Step4 All-to-All (combine)：把算完的结果原路寄回      │
 │        → unpermute 还原到 token 原始顺序               │
 └────────────────────────────────────────────────────────┘
```

要点：

1. **两次 All-to-All**：一次发去（dispatch / scatter），一次收回（combine / gather）。
2. **dispatch 前要 permute**：把 token 按目标 rank 连续排好，All-to-All 才能按桶发送。
3. **combine 后要 unpermute**：恢复 token 在序列中的原始位置，加权求和。
4. Router 打分、Top-k、加权求和在**本地**完成，只有 token 的特征向量上 All-to-All。

> All-to-All 是集合通信原语之一，详见 [[ai-infra/网络/集合通信原语]]：每个 rank 给其它每个 rank 都发一块不同数据，是 MoE 通信开销的大头。

---

## 4. All-to-All 通信量：逐数手算

设：批 $b$、序列 $s$、隐藏维 $h$、Top-$k$、专家并行度 `ep`，数据类型 2 字节（fp16/bf16）。

**每个 MoE 层、每次前向**：每个 token 被复制 $k$ 份发往 $k$ 个专家，dispatch 与 combine 各一次 All-to-All，每次传输的张量大小：

$$V_{a2a}=b\cdot s\cdot k\cdot h\ \text{（元素数）}$$

一层前向的双向 All-to-All 字节数（dispatch+combine）：

$$\text{Bytes}_{fwd}=2\times (b\,s\,k\,h)\times 2\text{B}=4\,k\,b\,s\,h\ \text{字节}$$

> 注：这是"逻辑传输量"。All-to-All 在 `ep` 张卡间分摊，单卡实际**收发**约为 $V_{a2a}\cdot\frac{ep-1}{ep}$，近似 $V_{a2a}$。

**代入数值**：$b{=}1$（每卡 micro-batch）、$s{=}4096$、$h{=}4096$、$k{=}2$：

$$V_{a2a}=1\times4096\times2\times4096=3.36\times10^7\ \text{元素}=33.5\text{M}$$
$$\text{Bytes}_{fwd}=4\times2\times1\times4096\times4096\times1\text{?}$$

逐项展开更清楚（一次 All-to-All 的字节）：

$$b\,s\,k\,h\times2\text{B}=1\times4096\times2\times4096\times2=6.7\times10^7\ \text{B}\approx 64\text{ MiB}$$

dispatch+combine 两次：$\approx128$ MiB / 层 / 前向。反向再来一遍（梯度方向相反的 All-to-All）：再 $\approx128$ MiB。

**与 TP 的 AllReduce 对比**：张量并行每层 2 次 AllReduce，每次 $\approx 2bsh$ 字节；MoE 的 All-to-All 量级相近，但**模式更差**——All-to-All 对网络对分带宽（bisection bandwidth）极敏感，跨机时容易成为瓶颈，所以 EP 通常优先放在 **NVLink 同机内**（见第 7 节）。

> 把通信藏到计算后面的技巧见 [[llm-optimizer/计算通信重叠]]。

---

## 5. 容量因子 capacity factor 与 token 丢弃

All-to-All 是**固定形状**通信（编译期要定缓冲区大小），但路由是动态的——某些专家可能被分到远多于平均的 token（热点专家）。框架用 **capacity（容量）** 给每个专家设一个固定上限：

$$C=\left\lceil \text{cf}\times\frac{k\cdot b\cdot s}{E}\right\rceil$$

- $\frac{k\,b\,s}{E}$：理想均匀时每个专家应得的 token 数。
- $\text{cf}$（capacity factor，常用 1.0 / 1.25 / 2.0）：留出余量。

```
  E=8, k=2, b*s=1024 tokens, cf=1.25
  理想每专家 = 2*1024/8 = 256
  容量 C = ceil(1.25 * 256) = 320

  专家 #3 实际被路由 380 个 token：
  ┌──────────────── 容量 320 ────────────────┐ 丢弃 60
  │ token token token ... token              │  ✗ ✗ ... ✗   ← dropped（残差直通）
  └───────────────────────────────────────────┘
  专家 #5 实际被路由 100 个 token：
  ┌──────────────── 容量 320 ────────────────┐
  │ token ... token | pad pad pad ... pad     │  ← 用 0 填充，浪费算力
  └───────────────────────────────────────────┘
```

- **溢出（drop）**：超过 $C$ 的 token 不进专家，直接走残差（输出 = 输入），训练初期常见。
- **不足（pad）**：少于 $C$ 时补零，浪费计算但保证形状固定。
- **cf↑** → 丢得少、padding 浪费多、显存/通信涨；**cf↓** → 省资源、丢得多、精度掉。
- 推理时常用 **dropless / no-drop**（如 Megatron 的 `--moe-token-dropless` / grouped-GEMM）做变长计算，避免丢 token。

---

## 6. 负载均衡 loss：逼专家被"雨露均沾"

如果不加约束，Router 会塌缩到只用少数专家（其它专家学不到东西）。Switch/GShard 引入**辅助负载均衡损失（auxiliary loss）**：

$$\mathcal{L}_{aux}=\alpha\cdot E\cdot\sum_{i=1}^{E} f_i\cdot P_i$$

- $f_i$：本 batch 实际被路由到专家 $i$ 的 token 比例。
- $P_i$：专家 $i$ 的平均 router 概率（softmax 输出均值）。
- $\alpha$：权重（典型 $10^{-2}$）。当某专家又被频繁选（$f_i$ 大）又被高概率打分（$P_i$ 大）时，$\mathcal{L}_{aux}$ 升高，梯度把负载推向均匀。

```
  不均衡（坏）            均衡（好）
  E0 ████████ 80%        E0 ██ 12%
  E1 █ 5%                E1 ██ 13%
  E2 0%                  E2 ██ 12%
  E3 █ 15%               E3 ██ 13%   ...
  → aux loss 高，被惩罚   → aux loss 低
```

- DeepSeek-MoE 系列改用 **bias-based / loss-free 负载均衡**（给每个专家一个可调偏置项，按负载动态加减），减少 aux loss 对主任务的干扰。
- 路由还常加**噪声（noisy top-k）**或 **z-loss** 稳定 logits 数值范围。

---

## 7. EP × DP × TP × PP：多维混合并行

真实大模型把 MoE 的 **EP** 与稠密的 3D 并行叠加。各维职责：

| 维度 | 切什么 | 通信原语 | 放哪 |
|---|---|---|---|
| DP（数据并行） | 切 batch，复制权重 | AllReduce（梯度） | 跨机可 |
| TP（张量并行） | 切单层权重矩阵 | AllReduce（每层2次） | 同机 NVLink |
| PP（流水并行） | 切层 | P2P send/recv | 跨机可 |
| **EP（专家并行）** | 切专家 | **All-to-All** | 优先同机 NVLink |

注意 **EP 与 DP 共享同一组 GPU**：MoE 层用 EP（专家分散），非 MoE 部分（attention、router、norm）仍用 DP/TP。框架通过两套不同的 process group 切换。

```
  16 卡布局示例：TP=2, EP=4, DP=2（DP×EP 复用同一物理卡）
  ┌── 节点0（8卡 NVLink） ──┐  ┌── 节点1（8卡） ──┐
  │ TP组: (g0,g1) (g2,g3)... │  │ ...               │
  │ EP组: g0 g2 g4 g6 一组   │  │ All-to-All 优先   │
  │       专家分散其中       │  │ 留在节点内        │
  └──────────────────────────┘  └───────────────────┘
   规则：把 All-to-All（EP）和 AllReduce（TP）尽量塞进
         机内高带宽域；DP/PP 的 AllReduce/P2P 走机间。
```

- **ZeRO + EP**：DeepSpeed 把非专家参数用 ZeRO-1/2/3 切，专家参数用 EP 切，两者正交。
- **专家梯度同步**：同一专家若被 DP 复制了多份，专家梯度需在 **expert-data-parallel** 组内 AllReduce（与普通 DP 的组不同）。这是 MoE 训练最易配错的点。

> 3D 并行原理见 [[ai-framework/megatron-lm/README]] 与 [[ai-framework/deepspeed/README]]；张量并行细节见 [[llm-inference/大模型推理张量并行]]。

---

## 8. 框架对照（机制为主，不写死版本与命令）

| 框架 | MoE 实现要点 | 入口/概念 |
|---|---|---|
| **DeepSpeed-MoE** | `deepspeed.moe.layer.MoE` 包住 FFN；`ep_size` 设专家并行度；与 ZeRO 正交；PR-MoE/残差专家；提供蒸馏 MoS | 见官方 tutorial，参数以官方为准 |
| **Megatron-LM** | `--num-experts / --expert-model-parallel-size`；grouped-GEMM 做 dropless；与 TP/PP/SP 深度集成；token permutation 内核 | 参数名以官方为准 |
| **ColossalAI** | `MoeLayer` + 集成的 expert-parallel context；面向集成进自有模型 | 见集成教程 |
| **PaddlePaddle** | `paddle.distributed` MoE 接口，`launch --gpus` 起多卡 | 见分布式训练文档 |
| **Tutel / FastMoE** | 高性能 All-to-All 内核、动态容量、JIT 调度（常被上层框架借用） | 内核库 |

DeepSpeed 启动示意（来自官方示例，命令默认值以官方为准）：

```
deepspeed --num_gpus=8 cifar10_deepspeed.py --moe --ep-world-size 2 \
          --num-experts 4 --top-k 1 --noisy-gate-policy RSample
# ep-world-size = 专家并行度；num-experts 总专家数；top-k 每 token 选几个
```

Paddle 启动（原文件保留）：

```
python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir logs train_moe.py
```

参考链接：
- DeepSpeed MoE：https://www.deepspeed.ai/tutorials/mixture-of-experts/
- DeepSpeedExamples：https://github.com/microsoft/DeepSpeedExamples/blob/master/training/cifar/run_ds_moe.sh
- ColossalAI MoE：https://colossalai.org/zh-Hans/docs/advanced_tutorials/integrate_mixture_of_experts_into_your_model/
- Paddle MoE：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/moe_cn.html

---

## 数值手算：Mixtral-8x7B 量级端到端

设定（贴近 Mixtral-8x7B）：层数 $L{=}32$、$h{=}4096$、$d_{ff}{=}14336$、专家数 $E{=}8$、$k{=}2$、序列 $s{=}4096$、每卡 micro-batch $b{=}1$、bf16（2 字节）。EP=8（每卡 1 个专家）。

**① 单专家参数**（SwiGLU 有 3 个矩阵）：

$$P_{e}=3\times h\times d_{ff}=3\times4096\times14336\approx1.76\times10^8\approx0.176\text{B}$$

**② 全模型专家参数**：$E\times L\times P_e=8\times32\times0.176\text{B}\approx45\text{B}$。加 attention 等约 $47\text{B}$ 总参数——但每 token 只激活 $\approx13\text{B}$（$k{=}2$ 个专家 + 共享部分）。

**③ 每 token 每 MoE 层 FLOPs**（仅激活 2 个专家，前向）：

$$\text{FLOPs}=2_{(乘加)}\times k\times3 h d_{ff}=2\times2\times3\times4096\times14336\approx7.05\times10^8$$

对比 dense-8x（8 个专家全算）会是 $4\times$ 这么多——稀疏激活省下的就是这部分。

**④ 一层 All-to-All 通信（dispatch，单向，bf16）**：

$$b\,s\,k\,h\times2\text{B}=1\times4096\times2\times4096\times2=6.71\times10^7\text{B}\approx 64\text{ MiB}$$

dispatch+combine = $128$ MiB/层/前向；反向再 $128$ MiB；全模型 32 层一次前向 All-to-All $\approx 32\times128=4$ GiB。

**⑤ 容量与丢弃**：$E{=}8,k{=}2,b s{=}4096$，cf=1.0 →

$$C=\lceil 1.0\times\tfrac{2\times4096}{8}\rceil=1024\ \text{token/专家}$$

若某专家路由到 1300 个 → 丢 $1300{-}1024{=}276$ 个（约 6.7% 该专家 token 走残差）。把 cf 提到 1.5 → $C{=}1536$，几乎不丢，但 dispatch 缓冲 + padding 显存涨 50%。

> 直观结论：MoE 用 ~3.6× 的总参数（47B vs 13B 激活），换来与 13B dense 相近的**单 token 计算量**，代价是 All-to-All 通信 + 容量管理 + 负载均衡的工程复杂度。FLOPs 估算方法见 [[llm-algo/FLOPs]]。

---

## 常见问题

| 问题 | 答案 |
|---|---|
| EP 和 TP 能不能都设大？ | 能但都要 All-reduce/All-to-All；优先把两者塞进同机 NVLink，跨机带宽会成瓶颈 |
| All-to-All 为什么比 AllReduce 难？ | 它要求**对分带宽**充足，跨机交换机易拥塞；AllReduce 有 ring/tree 优化更友好 |
| 容量因子设多大？ | 训练常 1.0~1.25，推理用 dropless；越大越不丢但越费显存/带宽 |
| token 被丢了模型还能用吗？ | 能，丢弃 token 走残差（恒等映射），少量丢弃对收敛影响有限 |
| 专家梯度在哪同步？ | 在 **expert-data-parallel 组**（与普通 DP 组不同），最易配错 |
| 为什么要负载均衡 loss？ | 否则路由塌缩到少数专家，多数专家学不到东西、算力浪费 |
| 推理时 MoE 的瓶颈？ | All-to-All 延迟 + 专家权重的显存/带宽（每层要读不同专家） |
| EP=1 是什么？ | 不做专家并行，所有专家在本卡（小模型/调试），无 All-to-All |
| grouped-GEMM 解决什么？ | 把变长的各专家 token 批量做成一次分组矩阵乘，避免 padding 浪费、支持 dropless |

---

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-algo/moe/README]] — MoE 算法原理（门控/Top-k/Switch/GShard）
- [[llm-algo/mlp]] — FFN/MLP 子层，专家的本体
- [[llm-algo/FLOPs]] — FLOPs 与计算量估算
- [[llm-algo/transformer/模型架构]] — MoE 嵌在哪一层
- [[ai-infra/网络/集合通信原语]] — All-to-All / AllReduce 原理
- [[llm-optimizer/计算通信重叠]] — 把 All-to-All 藏到计算后面
- [[ai-framework/deepspeed/README]] — DeepSpeed-MoE + ZeRO
- [[ai-framework/megatron-lm/README]] — Megatron MoE + 3D 并行
- [[llm-inference/大模型推理张量并行]] — TP 与 EP 的混合
- [[llm-inference/README]] — 推理侧 MoE 部署
