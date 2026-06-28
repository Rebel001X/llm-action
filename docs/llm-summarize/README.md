# 大模型实践总结（落地经验汇总）

> 本目录是「从 0 到 1 把大模型跑起来」的实战经验合集：AI 集群 → 分布式训练 → 参数高效微调 → 推理加速 → 评估 → 网络通信 → 生态选型。本文是该目录的总览与导航，把散落在各篇里的硬核结论用「底层原理 + 数值手算 + ASCII 图」串成一条主线。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[llm-inference/README]] · [[llm-algo/transformer/模型架构]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]]

## 阅读地图

| 节 | 主题 | 你会得到 | 富笔记跳转 |
|----|------|---------|-----------|
| 0 | 一句话锚点 | 整个目录在讲什么 | — |
| 1 | 地基：内存墙 vs 通信墙 | 为什么必须分布式 | [[ai-infra/算力/GPU工作原理]] |
| 2 | AI 集群与加速卡 | A800/H800 特供差异手算 | [[ai-infra/ai-hardware/CUDA]] |
| 3 | 模型架构三分法 | Encoder/Decoder/Enc-Dec 选型 | [[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] |
| 4 | 分布式并行 | DP/TP/PP/ZeRO 通信量手算 | [[llm-inference/大模型推理张量并行]] · [[llm-optimizer/计算通信重叠]] |
| 5 | 显存优化 | 重计算/Offload/混合精度 | [[transformer内存估算]] |
| 6 | 参数高效微调 PEFT | LoRA/Prefix/Prompt 显存账 | [[llm-train/peft/Prefix-Tuning]] · [[llm-train/peft/Prompt-Tuning]] |
| 7 | 推理加速 | KV-Cache/量化/FlashAttention | [[llm-inference/KV-Cache优化]] · [[llm-optimizer/FlashAttention]] |
| 8 | RLHF/对齐 | 为什么 RLHF 有效 | [[llm-alignment/RLHF]] · [[llm-alignment/DPO]] |
| 9 | 评估 | benchmark/Arena/Elo | [[llm-inference/解码策略]] |
| 10 | 网络通信 | NVLink/RDMA/NCCL | [[ai-infra/网络/集合通信原语]] |
| — | 数值手算汇总 | 一张表把所有账算清 | — |

## 0. 一句话锚点

> **大模型工程 = 在「内存墙」与「通信墙」两堵墙之间，用并行 + 显存优化 + 低精度，把一个装不下、算不快的模型塞进有限的 GPU 集群里训练与推理。**

下文每一节都围绕「这堵墙怎么撞上的、用什么招拆掉、拆的代价（通信量/显存）是多少」展开。

## 1. 地基：两堵墙——内存墙与通信墙

单卡放不下、多卡通信慢，是大模型一切技术的根源。先建立直觉。

```
            ┌──────────────── 一次训练 step ────────────────┐
   数据 →  [前向计算] → [反向计算] → [优化器更新] → 下一 step
            │            │             │
            │  需要存：    │ 需要存：      │ 需要存：
            ▼            ▼             ▼
        激活值(Act)   梯度(Grad)    优化器状态(Opt)
         ↑撞内存墙↑                  ↑这部分最大↑
         多卡之间交换梯度/参数 ───────► 撞通信墙
```

**内存墙**：模型参数量增长（亿→千亿→万亿）远快于单卡显存增长（V100 32G → A100 80G → H100 80G）。
**通信墙**：参数放不下要切到多卡，多卡每步都要交换梯度/参数，网络带宽成为瓶颈。

> 这两堵墙是 [[ai-infra/算力/GPU工作原理]] 里「算力涨得快、显存/带宽涨得慢」的直接后果。

## 2. AI 集群与加速卡

主流仍是 NVIDIA GPU，分三系列：Tesla（A100/H100，训练推理）、GeForce（RTX 4090，消费级）、Quadro（专业可视化）。国内有特供版 A800/H800，**算力相同，差在卡间互联带宽**。

```
  A100 NVLink 总带宽 600 GB/s  ┐
  A800 NVLink 总带宽 400 GB/s  ├─ 特供版砍的是「卡间互联」，不是算力
  H100 NVLink 总带宽 900 GB/s  │
  H800 NVLink 总带宽 400 GB/s  ┘
```

### 数值手算：为什么 A800 的张量并行更吃亏？

张量并行（TP）每层要做 2 次 AllReduce，通信量正比于激活大小、反比于 NVLink 带宽。设单次需传输 $D$ 字节：

$$T_{\text{comm}} = \frac{2(N-1)}{N}\cdot\frac{D}{B}$$

其中 $N$=并行卡数、$B$=NVLink 带宽。同样 8 卡、同样 $D$：

- A100：$B=600$ GB/s → 通信耗时 $\propto 1/600$
- A800：$B=400$ GB/s → 通信耗时 $\propto 1/400$ = **慢 1.5×**

> 结论：算力（GEMM 部分）一样，但通信密集的 TP 在 A800 上更慢 → 国内集群更倾向**少用 TP、多用 PP+DP**。详见 [[llm-inference/大模型推理张量并行]]。

**集群规划经验**：GPU 服务器最好买**偶数台**（2/4/8）。例：OPT-66B 共 64 层，流水线并行只能切成 8 或 16 段，3 台 24 卡反而切不整（只能用 16 或 8 卡）。

国产加速卡：华为昇腾 910/310（达芬奇架构）、海光 DCU、寒武纪思元、百度昆仑芯、阿里含光 800。坑较多，时间紧建议优先 NVIDIA。

## 3. 模型架构三分法

Transformer 开创了 MLP/CNN/RNN 之后的第四大类模型，基于它分三类：

```
   ┌─ Encoder-only  (自编码)   理解类：分类/抽取   BERT, RoBERTa
   │
Transformer ─┼─ Decoder-only  (自回归)   生成类：文本生成    GPT, LLaMA, OPT, Bloom
   │
   └─ Enc-Decoder (seq2seq)   条件生成：翻译/摘要  T5, BART, GLM
```

- **Encoder-only**：破坏句子让模型填补 → 擅长理解。
- **Decoder-only**：把自己上一步输出当下一步输入 → 擅长生成（**当今主流**）。
- **Enc-Decoder**：编码器输出喂给解码器 → 擅长「输入 A 生成 B」。

> MoE（稀疏混合专家）让参数量从亿→万亿而 FLOPs 不爆炸，原理见 [[llm-algo/moe/README]]；Transformer 内部结构见 [[llm-algo/transformer/模型架构]] 与 [[llm-algo/mlp]]，位置编码见 [[llm-algo/旋转编码RoPE]]。

## 4. 分布式并行（拆内存墙/通信墙的主力）

```
 数据并行 DP      每卡完整模型，切数据，反向后 AllReduce 梯度
 张量并行 TP      把一层的矩阵横/竖切到多卡，层内 AllReduce 激活
 流水线并行 PP    把不同层放不同卡，像流水线一样传 activation
 ZeRO            DP 的外衣 + 把 Opt/Grad/Param 切片分散，逻辑上是 DP
 3D 混合并行      DP × TP × PP 同时用，万亿模型标配
```

### 三种并行的通信对象一图看懂

```
   DP(切数据)         TP(切层内)          PP(切层间)
  GPU0  GPU1        GPU0 │ GPU1        GPU0: 层1-8
   │     │           └─AllReduce─┘     GPU1: 层9-16
   └AllReduce梯度┘     (每层2次)         GPU0→GPU1 传激活(P2P)
```

### 数值手算：DP 的梯度 AllReduce 通信量

模型 $P$=7B 参数，FP16 梯度（2 字节/参数）。Ring-AllReduce 每卡收发量约 $2P\cdot\frac{N-1}{N}$：

- 梯度总字节 $= 7\times10^9 \times 2 = 14$ GB
- 8 卡：每卡通信量 $\approx 2\times14\times\frac{7}{8} = 24.5$ GB/step
- 若 NVLink 400 GB/s：$24.5/400 \approx 61$ ms 纯通信/step

> 这解释了为什么要 [[llm-optimizer/计算通信重叠]]——让这 61ms 通信藏在反向计算后面，而不是干等。集合通信原语（AllReduce/AllGather/ReduceScatter）见 [[ai-infra/网络/集合通信原语]]。

**ZeRO 三级**（在「逻辑数据并行」下达到「模型并行省显存」效果）：

```
ZeRO-1: 切优化器状态(Opt)         省最多，通信几乎不变
ZeRO-2: 再切梯度(Grad)            通信略增
ZeRO-3: 再切参数(Param)          省到极致，通信量翻倍(需AllGather参数)
```

> 框架对照：ZeRO/FSDP 看 [[ai-framework/deepspeed/README]]，TP/PP 看 [[ai-framework/megatron-lm/README]]。

## 5. 显存优化技术（不加卡也能省）

```
 重计算 Recompute   前向不存激活，反向时重算 → 时间换空间
 Offload            参数/激活在 GPU↔CPU 内存间横跳 → 通信换显存
 混合精度 BF16/FP16 权重/激活用 16bit → 显存减半 + 提速 2~4×
```

- **重计算**（Activation/Gradient Checkpointing）：把存激活的显存 $O(L)$ 降到 $O(\sqrt{L})$，代价是多一次前向（约 +30% 算力）。
- **Offload**：ZeRO-Offload/Infinity，把不活跃的参数甩到 CPU 内存甚至 NVMe。
- **混合精度**：BF16 动态范围大、不易溢出；FP16 输入 > 65504 会溢出成 Inf。

### 数值手算：7B 模型训练显存账（Adam + 混合精度）

| 项 | 公式（每参数字节） | 7B 合计 |
|----|------|--------|
| FP16 参数 | 2 | 14 GB |
| FP16 梯度 | 2 | 14 GB |
| FP32 参数副本(Adam) | 4 | 28 GB |
| Adam 一阶动量 m | 4 | 28 GB |
| Adam 二阶动量 v | 4 | 28 GB |
| **小计（不含激活）** | **16** | **≈112 GB** |

> 单张 80G 卡装不下 → 必须 ZeRO/并行。激活显存另算，正是 [[transformer内存估算]] 详细推导的部分；这也是「重计算」要省的目标。

## 6. 参数高效微调（PEFT）

全量微调一个 7B 要 112GB 显存，普通人玩不起 → 冻结主干，只训一小撮参数。

```
   原始权重 W (冻结，不更新)
        │
        ├── LoRA:      W + B·A   只训低秩 A(r×d)、B(d×r)
        ├── Prefix:    在每层 KV 前拼可学习的虚拟 token
        ├── Prompt:    只在输入层拼可学习 prompt（Prefix 简化版）
        └── Adapter:   层间插 down→非线性→up 小瓶颈结构
```

### 数值手算：LoRA 省了多少？

一个 $d=4096$ 的方阵全量微调要训 $4096^2 \approx 16.8$M 参数。LoRA 秩 $r=8$：

$$\text{LoRA 参数} = d\times r + r\times d = 2\times4096\times8 = 65536 \approx 0.066\text{M}$$

→ 可训参数降到 **0.39%**，优化器状态（Adam 占大头）随之降 256×。

> 显存账：LoRA 把第 5 节那张「112GB」表里随参数走的 Opt/Grad 砍到几乎为零，只剩冻结的 14GB FP16 权重 + 少量激活。Prefix/Prompt 原理见 [[llm-train/peft/Prefix-Tuning]] 与 [[llm-train/peft/Prompt-Tuning]]，训练全景见 [[llm-train/README]]。

**两个代价要记住**：PEFT 通常①推理略变慢（多一条旁路计算）②精度略低于全量微调。

## 7. 推理加速

推理是模型投产的「最后一公里」，三个层面发力：

```
 算法层  蒸馏、量化(INT8/INT4/FP8)
 软件层  计算图融合、KV-Cache、FlashAttention、PagedAttention
 硬件层  FP8 (H 系列原生支持，兼 fp16 稳定性 + int8 速度)
```

### KV-Cache：自回归生成的省算神器

```
   不缓存：生成第 t 个 token 要重算前 t-1 个的 K,V  → O(t²) 浪费
   缓存：  把历史 K,V 存下，每步只算新 token 的 K,V → O(t)
```

### 数值手算：KV-Cache 显存

LLaMA-7B：层数 $L=32$、隐藏维 $d=4096$、FP16。每 token 每层缓存 K 和 V 各 $d$ 个数：

$$\text{每 token} = 2 \times L \times d \times 2\text{字节} = 2\times32\times4096\times2 = 512\text{ KB}$$

序列长 2048、batch 16：$512\text{KB}\times2048\times16 \approx 16$ GB——**KV-Cache 比模型权重还吃显存**。

> 这正是 [[llm-inference/KV-Cache优化]] 和 [[llm-optimizer/kv-cache]] 要压缩的对象（MQA/GQA/量化 KV）。FlashAttention 用分块 + 不落盘中间矩阵把注意力显存从 $O(n^2)$ 降到 $O(n)$，见 [[llm-optimizer/FlashAttention]]。量化基础见 [[llm-compression/quantization/量化基础]]，FP8 见 [[llm-compression/quantization/fp8]]。解码采样策略见 [[llm-inference/解码策略]]。

经典框架：FasterTransformer（NV，融合非 GEMM 算子 + 去 padding + FP8/INT8）、TurboTransformers（腾讯，变长序列 + smart batching）。

## 8. RLHF / 对齐：为什么有效？

```
   基座模型 ──SFT──► 指令模型 ──RLHF(PPO)──► 对齐模型
                                    ▲
                            人类「比较」奖励
```

OpenAI 实验：人类更偏爱 RLHF 模型的输出。一个被广泛接受的直觉是**「比较」比「生成」容易**——

> 让你写一首好俳句很难；但给你两首让你选哪首更好，容易得多。RLHF 正是利用这种「评判 < 生成」的不对称性，用人类的判断力造出更好的模型。

代价：RLHF 有时会**降低熵**——输出更单调、多样性下降。

> 偏好优化原理见 [[llm-alignment/RLHF]]；不用强化学习、直接用偏好数据做的 [[llm-alignment/DPO]] 是更轻量的替代。开源工具：DeepSpeed-Chat、ColossalChat。

## 9. 大模型评估

学术 benchmark（如 HELM）在聊天机器人上失效，原因有三：

```
 ① 聊得好不好太主观，难量化
 ② 测试集可能早被训练数据「看过」(数据污染)
 ③ 真实对话的任务在 benchmark 里根本不存在
```

→ **Chatbot Arena**：放弃固定 benchmark，两两对战、人工打分、用 **Elo 分数**排名。

评估方法谱系：人工评估 / GPT-4 自动评估 / 指标评估（BLEU-4、ROUGE-L）/ Arena。工具：OpenAI evals（写 prompt 模板）、PandaLM（训一个打分模型）。

> 影响模型水平的三因素（Scaling Laws）：**计算量、数据量、参数量**——任一指数增长且不成瓶颈时，loss 线性下降；外加第四因素**数据质量**（GPT-4 造的数据微调远好于 GPT-3）。

## 10. AI 集群网络通信

```
  机器内              机器间
  ┌─────────┐         ┌──────────────┐
  共享内存(CPU↔CPU)    TCP/IP
  PCIe(CPU↔GPU)        RDMA ┬ InfiniBand
  NVLink(GPU↔GPU)           ├ iWARP
                            └ RoCE v2
```

- **机器内**：NVLink（GPU 直连，最快）> PCIe > 共享内存。
- **机器间**：RDMA（远程直接内存访问，绕过 CPU/内核）远快于普通 TCP/IP。
- 通信库：NCCL（NV）、Gloo（FB）、OpenMPI、HCCL（华为）。
- 监控：nvbandwidth（测带宽）、DCGM（健康/遥测）。

> 集合通信算法（Ring/Tree AllReduce 等）原子级讲解见 [[ai-infra/网络/集合通信原语]]。

## 数值手算汇总（一张表收口）

| 场景 | 关键量 | 手算 | 结论 |
|------|-------|------|------|
| 7B 训练显存 | Adam 混合精度 | 16 B/参数 ×7B | ≈112 GB，单卡装不下 |
| DP 梯度通信 | Ring-AllReduce | 2×14GB×7/8 | ≈24.5 GB/step/卡 |
| 通信耗时 | ÷NVLink 400GB/s | 24.5/400 | ≈61 ms/step 纯通信 |
| LoRA 省参 | r=8, d=4096 | 65536/16.8M | 仅 0.39% 可训参数 |
| KV-Cache | LLaMA-7B,2048,bs16 | 512KB×2048×16 | ≈16 GB，比权重还大 |
| A800 vs A100 | NVLink 带宽比 | 400 vs 600 | TP 通信慢 1.5× |

## 常见问题

| 问题 | 答案 |
|------|------|
| 为什么不能单机单卡训千亿模型？ | 撞内存墙：112GB+ 显存远超单卡 80GB |
| 国内为什么少用张量并行？ | A800/H800 NVLink 带宽被砍，TP 通信密集吃亏 |
| GPU 服务器为什么买偶数台？ | PP 切分要整除层数，奇数台常切不整 |
| LoRA 为什么省这么多？ | 优化器状态（Adam 占 12B/参数）只随可训参数走 |
| KV-Cache 为什么重要？ | 把自回归从 O(t²) 降到 O(t)，但显存可能超过权重 |
| RLHF 为什么有效？ | 利用「评判比生成容易」的不对称性 |
| BF16 vs FP16 选哪个？ | BF16 动态范围大不易溢出，训练首选；FP16 >65504 溢出 |
| 学术 benchmark 为什么不够？ | 主观 + 数据污染 + 覆盖不全 → 改用 Arena+Elo |
| 多机多卡反复加载模型出错？ | 共享存储读写延迟 + 模型/卡未正确设置（见本目录坑点） |

## 实践坑点（血泪经验）

- **HF Transformers 多机多卡**：反复加载模型、设置卡时要小心，否则可能保存模型不成功/不完整。
- **共享存储**：一个进程写入文件后，另一进程可能无法马上读到（NFS 缓存一致性问题）。
- **环境搭建**：注意旧环境 python/pip/virtualenv/setuptools 版本，能联网建议直接用 Docker。
- **底层库升级**：遇到 GLIBC 等提示需升级时务必慎重，可能导致系统宕机。
- **训练先小后大**：先用 OPT-125m/2.7b 跑通流程再上 13b/30b，便于排查。
- **框架选型省钱**：同模型不同框架资源消耗差异大（HF+DeepSpeed 训 OPT-30B 比 Alpa 省不少）。

> 本目录其他细分文章：`大模型实践总结.md`（全景长文）、`领域大模型.md`、`金融大模型.md`、`文档大模型.md`、`distribution_dl_roadmap.md`（分布式学习路线）。

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 架构原理：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]] · [[llm-algo/FLOPs]]
- 训练与微调：[[llm-train/README]] · [[llm-train/peft/Prefix-Tuning]] · [[llm-train/peft/Prompt-Tuning]]
- 并行与框架：[[llm-inference/大模型推理张量并行]] · [[llm-optimizer/计算通信重叠]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 推理优化：[[llm-inference/README]] · [[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[llm-inference/解码策略]]
- 压缩与精度：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 对齐：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 硬件与网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]] · [[ai-infra/网络/集合通信原语]]
- 内存估算：[[transformer内存估算]]
