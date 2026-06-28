# llm-action / docs 知识总览（Master Index）

> 一句话定位：本页是 `docs/` 整棵目录树的**总导航与知识地图**——把"基础 → 训练并行 → 推理优化 → PEFT 微调 → 国产化 → 实践总结"六大板块串成一条可学可查的主线。
> 📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[llm-inference/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/集合通信原语]]

---

## 阅读地图

| 板块 | 你要解决的问题 | 本仓核心目录 | 配套富笔记 |
| --- | --- | --- | --- |
| ① LLM 基础 | 数据类型/FLOPs/显存/GPU 监控 | `llm-base/` | [[llm-algo/FLOPs]] · [[ai-infra/算力/GPU工作原理]] |
| ② 分布式并行 | DP/TP/PP/MoE/自动并行 怎么切 | `llm-base/distribution-parallelism/` | [[llm-inference/大模型推理张量并行]] |
| ③ 分布式训练 | 混合精度/大模型训练经验 | `llm-base/distribution-training/` | [[llm-train/README]] |
| ④ 推理优化 | KV-Cache/投机采样/vLLM/服务框架 | `llm-inference/` | [[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]] |
| ⑤ FlashAttention | 注意力如何省显存、提带宽 | `flash-attention/` | [[llm-optimizer/FlashAttention]] |
| ⑥ PEFT 微调 | LoRA 家族/Adapter/Prefix | `llm-peft/` | [[llm-train/peft/Prefix-Tuning]] · [[llm-train/peft/Prompt-Tuning]] |
| ⑦ 国产化 & 总结 | 昇腾等硬件迁移 + 落地经验 | `llm_localization/` · `llm-summarize/` | [[llm-inference/README]] |

> 用法：**横向**按板块查资料；**纵向**按"先基础后并行、先训练后推理、先原理后落地"学。

---

## 0. 一句话锚点

LLM 工程的全部复杂度，都来自一个矛盾：**模型参数量 $N$ 与序列长度 $S$ 在涨，而单卡显存与单卡算力是有限的**。
于是衍生出三条主线，本 `docs/` 正是围绕它们组织：

- **算得下**：FLOPs / 显存估算，决定要几张卡 → 板块①。
- **放得下、训得快**：DP/TP/PP/MoE 并行 + 混合精度 → 板块②③。
- **跑得快、服务得起**：KV-Cache / FlashAttention / 投机采样 / vLLM → 板块④⑤。

```
            ┌───────────────────────────────────────────────┐
            │   矛盾: 模型/序列在涨, 单卡显存&算力有限       │
            └───────────────────────────────────────────────┘
                     │              │               │
            ┌────────▼───┐   ┌──────▼──────┐   ┌────▼────────┐
            │ 算得下      │   │ 训得快/放得下│   │ 跑得快       │
            │ FLOPs/显存  │   │ 并行+混合精度│   │ KV/Flash/投机│
            │ 板块①      │   │ 板块②③     │   │ 板块④⑤     │
            └────────────┘   └─────────────┘   └─────────────┘
                     └──────────► 板块⑥⑦ 微调 / 国产化 / 落地 ◄─┘
```

---

## 1. 地基/前置：开始之前要懂的 5 个常数

学并行与优化前，先把这 5 个"物理常数"刻进脑子，后面所有估算都靠它们：

| 符号 | 含义 | 典型值 | 出处 |
| --- | --- | --- | --- |
| $N$ | 模型参数量 | 7B / 70B / 175B | `llm-base/FLOPS.md` |
| $P$ | 单参数字节数 | FP32=4, FP16/BF16=2, FP8=1 | `llm-base/机器学习中常用的数据类型.md` |
| $C_{fwd}$ | 单 token 前向 FLOPs | $\approx 2N$ | `llm-base/FLOPS.md` |
| $C_{train}$ | 单 token 训练 FLOPs | $\approx 6N$ | [[llm-algo/FLOPs]] |
| $BW$ | 卡间/卡内带宽 | NVLink ~数百 GB/s, PCIe 数十 GB/s | [[ai-infra/算力/GPU工作原理]] |

```
 一个参数在不同精度下占的字节
 FP32 ████  4B
 FP16 ██    2B      <- 训练主力(配 FP32 master)
 BF16 ██    2B      <- 大模型训练首选, 指数位同 FP32
 FP8  █     1B      <- 推理/前沿训练, 见 [[llm-compression/quantization/fp8]]
 INT8 █     1B      <- 推理量化
 INT4 ▌     0.5B    <- 极限推理量化
```

> 关键直觉：**训练 ≈ 前向 ×3**（前向 1 + 反向 2）。所以 $C_{train}\approx 6N$/token，这是所有训练算力估算的根。

---

## 2. 板块①：LLM 基础（`llm-base/`）

把"一块 GPU 到一个集群"之间的所有概念补齐。

```
llm-base/
├── ai-algo.md ················ AI 算法概览
├── FLOPS.md ·················· FLOPs 与计算量估算 ★必读
├── 机器学习中常用的数据类型.md  FP32/FP16/BF16/FP8/INT8
├── 分布式训练加速技术.md ······ ZeRO/重计算/Offload 总览
├── 监控类: nvidia-smi(.md/-dmon) / dcgmi / monitor / NVIDIA-Nsight-Systems
├── 环境类: a800-env-install / h800-env-install / conda / singularity / slurm
└── scenes/ ··················· CV / 多模态 场景
```

- **必读起点**：`FLOPS.md` —— 训练算力、训练耗时、MFU 全靠它。
- **监控工具**：定位"卡是不是真在干活"——`nvidia-smi`（瞬时）→ `dcgmi`/`dmon`（持续）→ `Nsight Systems`（时间线级 profiling，看 kernel 与通信重叠）。
- 🔗 配套：[[llm-algo/FLOPs]] · [[ai-infra/ai-hardware/CUDA]] · [[ai-infra/算力/GPU工作原理]]

---

## 3. 板块②：分布式并行（`distribution-parallelism/`）

并行的本质是**沿不同维度切张量**。把一个 GEMM $Y = XW$ 想象成可切的矩形：

```
数据并行 DP        张量并行 TP(列切)     流水并行 PP
切 batch 维         切 W 的列            切 layer 维
┌──┬──┬──┐         ┌────┐ 卡0           层1┐
│b0│b1│b2│ 各卡    │W列 │ ──┐           层2├ 卡A
│整模型│整模型│    │切片 │   concat       层3┘
└──┴──┴──┘         └────┘ 卡1           层4┐
通信: AllReduce梯度  通信: AllReduce激活   层5├ 卡B
                                          层6┘
                                        通信: P2P 边界激活
```

| 子目录 | 切什么 | 通信原语 | 富笔记 |
| --- | --- | --- | --- |
| `data-parallelism/` | batch | AllReduce(梯度) | [[ai-infra/网络/集合通信原语]] |
| `tensor-parallel/` | 单层权重(行/列) | AllReduce(激活) | [[llm-inference/大模型推理张量并行]] |
| `pipeline-parallelism/` | layer 段 | P2P(边界激活) | [[ai-framework/megatron-lm/README]] |
| `moe-parallel/` | 专家 | All-to-All(token) | [[llm-algo/moe/README]] |
| `multidimensional-hybrid-parallel/` | 3D/4D 混合 | 组合 | [[ai-framework/deepspeed/README]] |
| `auto-parallel/` | 自动搜索(Alpa/GSPMD/Mesh-TF/FlexFlow/Galvatron) | 由编译器决定 | — |

**通信量手算（TP，单个 Transformer 层，张量并行度 $t$）**：Megatron 风格每层前向 2 次、反向 2 次 AllReduce，每次 AllReduce 通信量 $\approx 2\cdot\frac{(t-1)}{t}\cdot M$（$M$=激活张量字节）。

> 例：batch×seq×hidden = $b{=}8,\ s{=}2048,\ h{=}8192$，FP16 → $M = 8\times2048\times8192\times2\text{B}\approx 268\text{MB}$。$t{=}8$ 时单次 AllReduce 实际传输 $2\times\frac{7}{8}\times268\approx 469\text{MB}$。这就是 TP **必须放在 NVLink 同机内**的原因——跨机 PCIe/IB 撑不住这么频繁的大块 AllReduce。

---

## 4. 板块③：分布式训练（`distribution-training/`）

并行解决"放得下"，混合精度与 ZeRO 解决"训得起"。

```
混合精度(AMP) 一步:
  FP16 前向 ─► FP16 反向 ─► 梯度 unscale ─► FP32 master 权重更新
       ▲ loss scaling 防下溢                 ▲ 保精度
  显存里同时存: FP16权重 + FP16梯度 + FP32(master权重/动量/方差)
```

**Adam 训练单参数显存手算**（混合精度，经典 ZeRO 论文口径）：

| 项 | 字节/参数 |
| --- | --- |
| FP16 权重 | 2 |
| FP16 梯度 | 2 |
| FP32 master 权重 | 4 |
| FP32 动量 $m$ | 4 |
| FP32 方差 $v$ | 4 |
| **合计** | **16** |

> 例：$N=7\text{B}$ → 仅优化器状态 + 权重 + 梯度 = $7\times10^9\times16\text{B}=112\text{GB}$，**单张 80GB 卡放不下** → 必须 ZeRO 切分或并行。这正是 `分布式训练加速技术.md` 与 [[ai-framework/deepspeed/README]] 的动机。
>
> 文件夹内还有 `Bloom-176B`/`GLM-130B`/`OPT-175B` 三份**一手训练经验**，是踩坑实录，强烈建议在动手前读。

🔗 配套：[[llm-train/README]] · [[llm-compression/quantization/fp8]]（前沿 FP8 训练）

---

## 5. 板块④：推理优化（`llm-inference/`）

推理与训练的瓶颈不同：**训练受算力(compute-bound)，自回归解码受显存带宽(memory-bound)**。

```
自回归解码两阶段:
 Prefill(算所有prompt token, 算力密集)
   ┌──────────────┐
   │ t1 t2 t3 t4  │ ──► 写满 KV-Cache
   └──────────────┘
 Decode(逐 token, 每步只算1个, 反复读全部KV → 带宽密集)
   t5 → t6 → t7 ...  每步都把整份 KV 从显存搬进 SM
```

| 文件 | 主题 | 富笔记 |
| --- | --- | --- |
| `KV-Cache.md` | 缓存历史 K/V,免重算 | [[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]] |
| `flexflow/投机采样.md` | 小模型起草+大模型验证 | [[llm-inference/解码策略]] |
| `autoregressive-lm-decoding-methods.md`(在 llm-base) | greedy/beam/top-p | [[llm-inference/解码策略]] |
| `vllm.md` / `DeepSpeed-Inference.md` | PagedAttention/推理引擎 | [[llm-inference/README]] |
| `LLM服务框架对比.md` / `llm推理框架.md` | 选型 | — |
| `llm推理优化技术.md` | 总纲 | [[llm-optimizer/计算通信重叠]] |

**KV-Cache 显存手算**（单序列）：$\text{KV}=2\times L\times s\times h\times P$。

> 例：Llama-7B 类 $L=32,\ h=4096,\ \text{FP16}\,P=2$，序列 $s=2048$ →
> $2\times32\times2048\times4096\times2\text{B}\approx 2.1\text{GB}$ / 单条序列。
> 并发 32 条就是 ~67GB——**KV 而非权重才是高并发服务的显存杀手**，所以才有 PagedAttention / KV 量化 / GQA。

---

## 6. 板块⑤：FlashAttention（`flash-attention/`）

朴素注意力把 $S\times S$ 的注意力矩阵**整个写回 HBM**，显存 $O(S^2)$ 且带宽爆炸。FlashAttention 用**分块 + online softmax**，在 SRAM 内边算边累加，永不落地大矩阵。

```
朴素:  Q,K,V ─► S=QKᵀ(写HBM, O(S²)) ─► softmax(读写HBM) ─► O
Flash: 分块 ─► for 每块Kj,Vj:
              在SRAM算 Sij, 用 online-softmax 更新 (m,l,O)
       从不在 HBM 落 S 全矩阵 ─► 显存 O(S), HBM 读写量大降
```

- **省的是 IO 不是 FLOPs**：FLOPs 几乎不变，但 HBM 访问从 $O(S^2)$ 降到 $O(S^2/\text{blocksize})$ 的有效搬运，长序列加速显著。
- 细节（online softmax 的 $m,\ell$ 递推、反向重计算）见 `flash-attention/FlashAttention.md`。
- 🔗 配套：[[llm-optimizer/FlashAttention]] · [[ai-infra/ai-hardware/CUDA]]

---

## 7. 板块⑥：PEFT 参数高效微调（`llm-peft/`）

全参微调一个 70B 要存一整套优化器状态（见第 4 节，×16B/参数）。PEFT 只训极小一部分参数。

```
LoRA 思想: 冻结 W, 旁路加低秩增量
   h = W x + (B A) x        其中 A: r×d, B: d×r,  r ≪ d
   只训 A,B → 可训参数从 d² 降到 2dr
```

| 文件 | 方法 | 富笔记 |
| --- | --- | --- |
| `LoRA-FA.md` | 冻结 A 只训 B,再省显存 | [[llm-train/peft/Prefix-Tuning]] |
| `ReLoRA.md` | 反复合并低秩→等效高秩 | — |
| `MAM_Adapter.md` | 统一 Adapter/Prefix/LoRA 视角 | [[llm-train/peft/Prompt-Tuning]] |

**LoRA 参数量手算**：$d=4096,\ r=8$ → 每个被改的矩阵可训参数 $2dr=2\times4096\times8\approx 6.5\text{万}$，相比 $d^2\approx1678\text{万}$ **降到约 0.4%**。这就是单卡能微调大模型的根本原因。

🔗 配套：[[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]

---

## 8. 板块⑦：国产化 & 实践总结（`llm_localization/` · `llm-summarize/`）

- **国产化**：把上述训练/推理栈从 NVIDIA 迁到国产 NPU（如昇腾）时的差异点——通信库、算子、混合精度行为不同。原理通用，**具体命令/版本以官方文档为准**（本仓不编造默认值）。
- **总结（`llm-summarize/`）**：`大模型实践总结*.md`、`领域大模型.md`（金融/文档）、`distribution_dl_roadmap.md`——一线落地经验与选型标准（如"13B 以下、单卡 4090 可跑、中文优先"）。

```
落地决策树(简化):
  要私有部署? ──否──► 直接调 API
       │是
  显存够全参微调? ──否──► PEFT(LoRA) 板块⑥
       │是
  单卡放得下? ──否──► 并行(TP/PP/ZeRO) 板块②③
       │是
  延迟敏感? ──是──► KV量化/投机采样/FlashAttn 板块④⑤
```

🔗 配套：[[llm-inference/README]] · [[llm-train/README]]

---

## 数值手算汇总（速查）

| 量 | 公式 | 7B 示例 |
| --- | --- | --- |
| 推理权重显存(FP16) | $2N$ | $14\text{GB}$ |
| 训练全套显存(Adam混精) | $16N$ | $112\text{GB}$ |
| 单 token 训练 FLOPs | $\approx 6N$ | $4.2\times10^{10}$ |
| KV-Cache(单序列) | $2Lsh\cdot P$ | $\approx 2.1\text{GB}(s{=}2048)$ |
| LoRA 可训参数占比 | $2r/d$ | $\approx0.4\%(r{=}8)$ |
| TP 单次 AllReduce 量 | $2\frac{t-1}{t}M$ | $469\text{MB}(t{=}8)$ |

> 训练一个 7B 模型跑 1T tokens 的算力：$6N\times \text{tokens}=6\times7\times10^9\times10^{12}\approx4.2\times10^{22}$ FLOPs。
> 一张算力 $312\,\text{TFLOPS}$(FP16) 的卡若 $\text{MFU}=40\%$，有效 $\approx125\,\text{TFLOPS}$ → 单卡约 $3.4\times10^8$ 秒 ≈ **10.6 年**，所以必须上千卡并行——这一句话串起了整个 `docs/`。

---

## 常见问题

| 问题 | 答 |
| --- | --- |
| 训练显存为什么是 16N？ | 权重2+梯度2+master4+动量4+方差4（FP16训练+FP32 Adam 状态） |
| 为什么 TP 要同机、PP 可跨机？ | TP 通信频繁且大块(AllReduce 激活)，PP 仅边界 P2P，对带宽不敏感 |
| 推理为什么 memory-bound？ | decode 每步只算 1 token 却要读整份权重+KV，算力闲、带宽满 |
| KV-Cache 和权重谁更吃显存？ | 高并发/长序列下 KV 远超权重，故有 PagedAttention/GQA/KV 量化 |
| FlashAttention 省 FLOPs 吗？ | 不省 FLOPs，省的是 HBM 读写(IO)，长序列才显著 |
| LoRA 为什么能单卡微调大模型？ | 可训参数降到 ~0.4%，优化器状态随之锐减 |
| 国产化能照搬命令吗？ | 原理通用，**具体命令/版本以官方为准** |

---

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 基础/FLOPs/架构：[[llm-algo/FLOPs]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/旋转编码RoPE]] · [[llm-algo/moe/README]]
- 训练/并行：[[llm-train/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[llm-inference/大模型推理张量并行]]
- 推理优化：[[llm-inference/README]] · [[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]] · [[llm-inference/解码策略]] · [[llm-optimizer/计算通信重叠]]
- 注意力/量化：[[llm-optimizer/FlashAttention]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 微调/对齐：[[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 硬件/网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]] · [[ai-infra/网络/集合通信原语]]
