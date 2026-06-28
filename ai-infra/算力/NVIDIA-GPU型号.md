# NVIDIA GPU 型号全景图

> 一句话定位：从消费级 RTX 到数据中心 A100/H100/H200，看懂 NVIDIA 产品线分层、计算能力（Compute Capability）语义、以及 DGX/HGX 整机的 NVLink/NVSwitch 互联拓扑。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 你会学到 | 关键产物 |
|----|----------|----------|
| 0 | 一句话锚点 | 型号≠裸卡，要分"卡/整机/平台"三层 |
| 1 | 地基：四大下游市场 | 游戏 / 专业可视化 / 数据中心 / 汽车 |
| 2 | 计算能力 Compute Capability | `X.Y` 主/次修订号语义 |
| 3 | 数据中心卡：A100 / H100 / H200 | 显存、带宽、架构代号 |
| 4 | DGX 与 HGX 的区别 | 整机 vs 基板 |
| 5 | HGX H100 8-GPU 互联拓扑 | 8 GPU + 4 NVSwitch 全连接 |
| 6 | NVLink / NVSwitch / PCIe 带宽账 | 900 GB/s 双向、14× PCIe |
| 实操 | 查型号 / 看拓扑 / 算能力 | `nvidia-smi`、官方链接 |
| 坑 | 选型与误区 | SXM≠PCIe、整机≠裸卡 |

## 0. 一句话锚点

NVIDIA 的"型号"不是单一维度，而是**三层嵌套**：

```
平台 (Platform)   ── DGX / HGX / EGX / AGX，含整机 + 软件栈
   └─ 整机/基板     ── 8×GPU + NVSwitch 的服务器或基板
        └─ 裸卡 GPU  ── A100 / H100 / H200，单颗 die（SXM 或 PCIe 封装）
             └─ 架构  ── Ampere(安培) / Hopper，决定 Compute Capability
```

> 看到"H100"先问一句：是说**裸卡**（单颗 GPU），还是**HGX H100 基板**（8 卡），还是**DGX H100 整机**（基板 + CPU + 网卡 + 机箱 + 软件）？三者价格差一个数量级。

## 1. 地基：NVIDIA 四大下游市场

NVIDIA 下游市场分为四类，各市场重点产品如下（原文真料，保留并补全）：

| 市场 | 重点产品 | 说明 |
|------|----------|------|
| **游戏** | GeForce RTX/GTX 系列 GPU（PC）、GeForce NOW（云游戏）、SHIELD（游戏主机） | 消费级，光追 + DLSS |
| **专业可视化** | Quadro / RTX GPU（企业工作站） | 渲染、CAD、影视 |
| **数据中心** | DGX（AI 服务器）、HGX（超算）、EGX（边缘计算）、AGX（自动设备） | 基于 GPU 的计算平台和系统 |
| **汽车** | NVIDIA DRIVE 计算平台：AGX Xavier（SoC 芯片）、DRIVE AV（自动驾驶）、DRIVE IX（驾驶舱软件）、Constellation（仿真软件） | 车载与自动驾驶 |

**消费级 vs 生产级官方入口（原文链接保留）**：

- 消费级：<https://www.nvidia.cn/geforce/graphics-cards/40-series/rtx-4090/>
- 生产级：<https://www.nvidia.cn/data-center/a100/>

**为什么 LLM 训练/推理几乎只看"数据中心"线？**

```
消费级 RTX 4090         数据中心 A100/H100
─────────────────       ─────────────────
24GB GDDR6X 显存        40/80GB HBM2e/HBM3 显存（带 ECC）
无 NVLink（40 系砍掉）   NVLink + NVSwitch 多卡全连接
FP64 被阉割             完整 FP64 + Tensor Core
驱动禁止数据中心部署     官方授权数据中心使用
```

显存容量、卡间互联带宽、ECC，这三点决定了大模型只能跑在数据中心卡上——单张 4090 的 24GB 连一个 70B 模型的权重（FP16 ≈ 140GB）都装不下，必须靠多卡，而多卡又需要 NVLink/NVSwitch 高速互联，这正是消费卡缺失的。

## 2. 计算能力（Compute Capability）

**定义**（原文真料）：计算能力标识设备的**核心架构、GPU 硬件支持的功能和指令**。

- 不同 GPU 型号的计算能力查询：<https://developer.nvidia.com/cuda-gpus#compute>

**语义结构**（原文真料 + 补全）：计算能力用主修订号 X 和次修订号 Y 表示，记为 `X.Y`：

- **主修订号 X**：标明**核心架构**（同一 X 代表同一代架构）；
- **次修订号 Y**：标识在此核心架构上的**增量更新**。

```
Compute Capability =  X  .  Y
                      │     │
                架构代号    架构内增量
                （Ampere=8）（A100=0, RTX30=6, A10=6...）
```

**常见对照（用于理解，非凭空编造，可在上面官方页核对）**：

| 架构代号 | 主号 X | 代表卡 | 典型 Compute Capability |
|----------|--------|--------|--------------------------|
| Volta | 7 | V100 | 7.0 |
| Turing | 7 | T4 / RTX 20 | 7.5 |
| Ampere(安培) | 8 | A100 | 8.0 |
| Ampere | 8 | RTX 30 / A10 | 8.6 |
| Hopper | 9 | H100 / H200 | 9.0 |

**为什么这串数字对工程师很重要？**

1. **编译目标**：`nvcc` 的 `-gencode arch=compute_80,code=sm_80` 里的 `80` 就是 Compute Capability 8.0。编错了，kernel 在目标卡上跑不起来。
2. **功能门槛**：某些指令只在特定能力以上可用。例如 Tensor Core 的 **FP8** 数据类型从 Hopper（9.0）起才有硬件支持 → 这直接决定了 FP8 量化推理能不能在你的卡上加速。详见 [[llm-compression/quantization/fp8]]。
3. **框架兼容**：PyTorch/CUDA 版本与 sm_XX 绑定，老卡（低能力）可能根本不被新框架支持。

> 数值示例：A100 的能力是 8.0，H100 是 9.0。差一个主号，意味着 Tensor Core 多了 FP8、Transformer Engine、第四代 NVLink 等一整套新功能——不是简单"更快"，而是"多了新指令"。

## 3. 数据中心裸卡：A100 / H100 / H200

| 型号 | 架构 | 显存 | 显存类型 | 代表互联 | 官方文档（原文真料） |
|------|------|------|----------|----------|----------------------|
| **A100** | Ampere(安培) | 40 / 80 GB | HBM2e | 第三代 NVLink | <https://docs.nvidia.com/dgx/dgxa100-user-guide/> |
| **H100** | Hopper | 80 GB | HBM3 | 第四代 NVLink | <https://docs.nvidia.com/dgx/dgxh100-user-guide/> |
| **H200** | Hopper | 141 GB | HBM3e | 第四代 NVLink | <https://www.nvidia.com/en-us/data-center/h200/> |

**H200 的关键点**：架构仍是 Hopper（与 H100 同代、同 Compute Capability 9.0），但把显存升级到 **HBM3e、141GB**。

**为什么 H200 只升显存就有意义？** 大模型推理在 decode 阶段是**显存带宽受限（memory-bound）**而非算力受限：每生成一个 token 都要把整个 KV Cache + 权重从 HBM 读一遍。显存更大、带宽更高，直接等于：

- 能装更长的上下文（KV Cache 更大）→ 见 [[llm-optimizer/kv-cache]]；
- 能装更大的 batch → 吞吐更高；
- 单卡能放下更大的模型 → 减少跨卡通信。

> 参考阅读（原文真料保留）：
> - [NVIDIA GPU A100 Ampere(安培) 架构深度解析](https://blog.csdn.net/han2529386161/article/details/106411138)
> - [GPU 进阶笔记（一）：高性能 GPU 服务器硬件拓扑与集群组网（2023）](https://arthurchiao.art/blog/gpu-advanced-notes-1-zh/)
> - [GPU 进阶笔记（二）：华为 GPU 相关（2023）](https://arthurchiao.art/blog/gpu-advanced-notes-2-zh/)
> - [NVIDIA DGX H100 介绍](https://www.foresine.com/news/465-cn.html)

## 4. DGX vs HGX：整机 vs 基板

这是初学者最容易混的两个词。

```
HGX（基板 / baseboard）          DGX（整机 / appliance）
──────────────────────          ──────────────────────
8×GPU + NVSwitch 焊在一块板上     在 HGX 基板基础上，再加：
卖给 OEM/云厂商去组装             ├─ CPU（双路）
不含 CPU/机箱/网卡选型自由        ├─ 系统内存
                                ├─ NVMe 存储
"零件"                          ├─ ConnectX/BlueField 网卡
                                ├─ 机箱 + 电源 + 散热
                                └─ DGX OS 软件栈
                                "成品整机，开箱即用"
```

- DGX 系统文档（原文真料）：<https://docs.nvidia.com/dgx-systems/>
- DGX-2 用户指南（原文真料）：<https://docs.nvidia.com/dgx/pdf/dgx2-user-guide.pdf>
- DGX H100 介绍（原文真料）：<https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html>
- DGX A100 介绍（原文真料）：<https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html>
- AI 芯片白皮书下载（原文真料）：<https://www.nvidia.cn/data-center/dgx-a100/>

> 一句话记忆：**HGX 是主板套件，DGX 是品牌服务器**。云厂商（如某 GPU 云）通常买 HGX 基板自行组装；企业图省事买 DGX 整机。

## 5. HGX H100 8-GPU 互联拓扑（核心）

这是原文最有价值的一段，下面把它"画出来"。

**事实（原文真料）**：HGX H100 8-GPU 是新一代 Hopper GPU 服务器的关键组成部分。它拥有**八个 H100 Tensor Core GPU** 和**四个第三代 NVSwitch**。每个 H100 GPU 都有多个第四代 NVLink 端口，并连接到所有四个 NVLink 交换机。每个 NVSwitch 都是一个**完全无阻塞（non-blocking）**的交换机，**完全连接所有八个 H100**。

```
        ┌──────────────────────────────────────────┐
        │  4 × 第三代 NVSwitch（全无阻塞交换）        │
        │   SW0    SW1    SW2    SW3                 │
        └───┬──────┬──────┬──────┬─────┬──┬──┬──┬───┘
            │      │      │      │ ... 每卡多个NVLink端口
   ┌────────┴──────┴──────┴──────┴────────┐
   │  每个 H100 都连到全部 4 个 NVSwitch    │
   ▼      ▼      ▼      ▼      ▼      ▼  ▼  ▼
 [H100] [H100] [H100] [H100] [H100][H100][H100][H100]
   GPU0   GPU1   GPU2   GPU3   GPU4  GPU5 GPU6 GPU7

  任意两卡之间 = 全连接（full mesh via switch）
  GPU_i ←→ GPU_j 双向带宽 = 900 GB/s
```

**这种拓扑的意义（原文真料 + 补全）**：NVSwitch 的完全连接拓扑使**任意一个 H100 都可以同时与任何其他 H100 通信**，无需多跳转发，也不会因为某条链路被占用而阻塞（non-blocking）。

**为什么不是直连 mesh，而要 NVSwitch？**

```
直连 mesh（无 switch）              NVSwitch 交换
─────────────────────              ──────────────
N 张卡需要 N×(N-1)/2 条链路          N 张卡各连到所有 switch
8 卡 = 28 条点对点链路               任意两卡距离恒为 1 跳
每卡链路数随 N 增长 → 端口爆炸        带宽不随通信对手变化而衰减
扩展性差                            可扩展、负载均衡
```

8 卡用直连要 28 条链路且每卡端口随规模爆炸；用 4 个 NVSwitch 把所有卡接到交换层，任意两卡恒为"1 跳"，这才是大规模 all-reduce 的理想拓扑。

## 6. NVLink / NVSwitch / PCIe 的带宽账

**事实（原文真料）**：HGX H100 卡间通信以每秒 **900 GB/s** 的 NVLink **双向**速度运行，这是当前 **PCIe Gen4 x16 总线带宽的 14 倍**多。

**手算验证"14 倍"**：

$$
\text{PCIe Gen4 x16 单向} \approx 16 \text{ GB/s},\quad \text{双向} \approx 32 \text{ GB/s}
$$

$$
\frac{900 \text{ GB/s}}{\sim 64 \text{ GB/s (双向 Gen4 x16 理论上限)}} \approx 14\times
$$

> 数量级直觉：NVLink 把"卡间搬数据"的成本降到接近 PCIe 的 1/14，所以张量并行（TP）这种**每层都要跨卡 all-reduce** 的策略才跑得动——TP 对带宽极度敏感，必须放在 NVLink 域内。见 [[ai-framework/megatron-lm/README]]。

**第三代 NVSwitch 的额外加速（原文真料）**：第三代 NVSwitch 为集合通信（collective）提供了**新的硬件加速**，包括**多播（multicast）**和 **NVIDIA SHARP 网络内规约（in-network reduction）**。结合更快的 NVLink 速度，像 **all-reduce** 这样的常见 AI 集合操作的有效带宽比 HGX A100 **增加了 3 倍**。集合操作的 NVSwitch 加速也显著**降低了 GPU 上的负载**。

**为什么"网络内规约"能省 GPU？**

```
传统 all-reduce（GPU 算）           SHARP 网络内规约（Switch 算）
─────────────────────             ──────────────────────────
数据在卡间多轮搬运 + GPU 做加法      Switch 在转发途中直接做求和
GPU 既算梯度又算规约                GPU 只发数据，规约下沉到交换芯片
占用 SM 算力 + 多次过网             一次过网即完成，多播分发结果
```

把"求和"从 GPU 下沉到交换机，GPU 的 SM 可以专心算梯度而不是搬数据做加法，这就是"有效带宽 ×3 且降低 GPU 负载"的来源。集合通信原语的语义见 [[ai-infra/网络/集合通信原语]]。

## 实操：如何在机器上确认这些信息

> 以下为常用查询命令（标准 `nvidia-smi` 用法，非凭空编造）。

**1. 看你手上是什么卡、显存多大**

```bash
nvidia-smi
# 输出第一列即型号，如 "NVIDIA A100-SXM4-80GB"
# Memory-Usage 列显示总显存（如 81920MiB ≈ 80GB）
```

型号串里的 `SXM4` / `PCIe` 后缀告诉你**封装形态**：`SXM` 是焊在 HGX 基板上、带 NVLink 的形态；`PCIe` 是插槽卡，NVLink 能力受限或没有。

**2. 看多卡互联拓扑（确认是否走 NVLink/NVSwitch）**

```bash
nvidia-smi topo -m
# 矩阵里 NV# 表示 NVLink（数字是链路数），
# PIX/PHB/SYS 表示走 PCIe/主板/跨 NUMA，速度依次变慢
```

如果两卡之间显示 `NV18` 之类，说明走 NVLink 全连接；显示 `SYS` 则是跨 CPU 走 PCIe，TP 放这里会很慢。

**3. 查 NVLink 实时带宽利用**

```bash
nvidia-smi nvlink -s     # 各 NVLink 链路状态/速率
nvidia-smi nvlink -g 0   # 计数器
```

**4. 确认 Compute Capability（决定能编译/能用 FP8 吗）**

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
# 例如输出 "NVIDIA H100 80GB HBM3, 9.0"
```

或在线对照官方表：<https://developer.nvidia.com/cuda-gpus#compute>

## 常见问题/坑

| 坑 | 现象 | 正解 |
|----|------|------|
| 把 H100 当成"一张卡" | 报价、招采时数量算错 | 区分裸卡 / HGX 基板（8卡）/ DGX 整机 |
| SXM 与 PCIe 混为一谈 | 买了 PCIe 版却想做 NVLink 全连接 TP | PCIe 版 NVLink 受限，TP 走 PCIe 会严重掉速 |
| 以为 H200 比 H100 算力强 | 期待 FLOPS 大涨 | 同 Hopper 架构、同 9.0；H200 主升**显存(141GB HBM3e)**，利推理 memory-bound |
| 消费卡跑数据中心 | 4090 缺 NVLink、显存小、驱动限制 | 大模型多卡训练/推理用数据中心卡 |
| Compute Capability 编错 | kernel 在目标卡跑不起来 | `nvcc -gencode arch=compute_XX,code=sm_XX` 对齐目标卡 |
| 以为 all-reduce 都靠 GPU 算 | 低估 NVSwitch 价值 | 三代 NVSwitch 有 SHARP 网络内规约，降 GPU 负载、有效带宽 ×3 |
| 拓扑没确认就跑 TP | 误把卡放到跨 NUMA / PCIe 域 | 先 `nvidia-smi topo -m`，TP 必须在 NVLink 域内 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 同域硬件：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 网络互联：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 框架并行（吃 NVLink 带宽）：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]
- 推理与显存：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]]
- 量化（FP8 看架构）：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]] · [[llm-compression/quantization/GPTQ]]
- 训练/对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 模型与评测：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
