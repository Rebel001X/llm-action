# 昇腾 PEFT 参数高效微调

> 在昇腾 NPU 上用 PEFT(LoRA/QLoRA/Adapter 等)只训练少量参数即可微调大模型，省显存、省算力、可多任务复用。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[llm-train/README]] [[llm-algo/transformer/模型架构]]

## 阅读地图

| 小节 | 你会得到什么 | 谁该重点看 |
| --- | --- | --- |
| 0. 一句话锚点 | PEFT 到底解决什么问题 | 所有人 |
| 1. 地基:昇腾栈定位 + 对标英伟达 | NPU↔GPU、CANN↔CUDA、torch_npu↔CUDA后端 的迁移心智图 | 从 GPU 迁移者 |
| 2. PEFT 家族与原理 | LoRA/QLoRA/Adapter/Prefix 的机制与 ASCII 图 | 算法/调参 |
| 3. PEFT 在昇腾的执行链路 | 一条前向反向如何落到 Cube/Vector 单元 | 想懂底层 |
| 4. 实操流程(不含具体命令) | 从 GPU 脚本搬到 NPU 的标准步骤 | 工程落地 |
| 5. 迁移要点与常见坑 | 算子缺失、dtype、量化、显存的真实坑 | 踩坑救援 |
| 常见问题 | 速查 FAQ | 救火 |

---

## 0. 一句话锚点

**全参微调(Full Fine-tuning)** 要更新模型的全部权重，7B 模型仅优化器状态(Adam 的一阶/二阶动量)就要十几 GB，显存压力极大。
**PEFT(Parameter-Efficient Fine-Tuning,参数高效微调)** 的核心思想：**冻结绝大部分预训练权重，只在旁路插入并训练极少量(常 <1%)的新参数**。这样：

- 显存占用大幅下降(优化器只为少量可训练参数分配状态)；
- 训练更快、更稳，不易灾难性遗忘；
- 产物是几十 MB 的"补丁"(adapter),一个底座可挂多个任务补丁。

在昇腾上，PEFT 的价值被进一步放大：国产卡常见单卡显存与互联带宽相对受限，**用 LoRA/QLoRA 把全参微调换成低秩微调，往往是"能不能在现有卡上跑起来"的关键**。

---

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 PEFT 处在软件栈哪一层

PEFT 本身是**框架层之上的训练算法/库**(Hugging Face `peft` 库),它不直接碰硬件，而是改写模型结构(注入 LoRA 层),再依赖下方的框架与算子库把计算落到 NPU。

```
┌─────────────────────────────────────────────┐
│  训练脚本 (transformers Trainer / 自定义循环) │  ← 你写的代码
├─────────────────────────────────────────────┤
│  PEFT 库 (LoRA/QLoRA/Adapter 注入)           │  ← 本文主角:只标记少量参数可训练
├─────────────────────────────────────────────┤
│  框架层  PyTorch + torch_npu  (或 MindSpore) │  ← 把算子派发到 NPU
├─────────────────────────────────────────────┤
│  CANN (算子库 AOL + 图引擎 GE + 运行时)       │  ← 对标 CUDA + cuDNN + cuBLAS
├─────────────────────────────────────────────┤
│  驱动 Driver / 固件 Firmware                  │
├─────────────────────────────────────────────┤
│  昇腾 NPU 硬件 (达芬奇 Cube + Vector 单元)     │  ← 对标 GPU 的 Tensor Core/CUDA Core
└─────────────────────────────────────────────┘
```

关键点：**PEFT 库是硬件无关的**。它在昇腾上能跑，靠的是 `torch_npu` 这块"适配垫片"——它把 PyTorch 的算子调用重定向到 CANN。所以"在昇腾上做 PEFT"几乎等价于"在昇腾上把 PyTorch 训练流程跑通，再叠加 peft 库"。

### 1.2 昇腾 ↔ 英伟达 生态对照表(迁移心智图)

| 维度 | 英伟达生态 | 昇腾生态 | 说明 |
| --- | --- | --- | --- |
| 加速芯片 | GPU (A100/H100) | NPU (昇腾 910B 等) | 一个是 SIMT，一个是达芬奇 Cube+Vector |
| 计算核心 | CUDA Core / Tensor Core | Vector 单元 / Cube 单元 | Cube 专攻矩阵乘(LoRA 的 BA 乘法都在这跑) |
| 底层计算平台 | CUDA | CANN | 编程框架 + 运行时 |
| 算子库 | cuDNN / cuBLAS | CANN 算子库 (AOL/AscendCL) | 提供 matmul、softmax、attention 等 |
| 设备字符串 | `cuda` | `npu` | 代码里 `to("cuda")` → `to("npu")` |
| PyTorch 后端 | 原生 CUDA 后端 | `torch_npu` 插件 | import 它即可让 `.npu()` 生效 |
| 集合通信 | NCCL | HCCL | 多卡 LoRA/分布式微调的 allreduce 走 HCCL |
| 训练大套件 | Megatron-LM | MindFormers / ModelLink | 大并行场景；小规模 PEFT 常直接用 transformers+peft |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE / vllm-ascend | 微调完的 adapter 合并后部署 |
| 量化工具 | GPTQ / AWQ / bitsandbytes | msmodelslim / 昇腾量化方案 | QLoRA 的 4bit 量化在昇腾上路径不同(见坑) |
| 混合精度 | AMP (fp16/bf16) | torch_npu AMP (bf16 优先) | 910B 对 bf16 友好 |

> **一句话迁移心智图**：把 `cuda` 想成 `npu`、`NCCL` 想成 `HCCL`、`cuDNN/cuBLAS` 想成 `CANN 算子库`，PEFT 库本身几乎不用改，改的是它脚下那层。

---

## 2. PEFT 家族与原理

### 2.1 LoRA:低秩适配(最主流)

LoRA(Low-Rank Adaptation)的洞见：微调时权重的**变化量 ΔW 是低秩的**。于是不直接更新原权重 W，而是把 ΔW 分解成两个小矩阵 **B·A**：

```
        原始权重 W (冻结, d×d)
            │
   x ───────┼──────────────►  W·x
            │                    +
            │   ┌──────┐  ┌──────┐
            └──►│  A   │─►│  B   │──►  (BA)·x · (α/r)
                │ d×r  │  │ r×d  │
                └──────┘  └──────┘
                 可训练     可训练      r ≪ d, 通常 r=8/16/32
            输出 = W·x + (α/r)·B·A·x
```

- 只训练 A、B 两个瘦矩阵，参数量从 d² 降到 2·d·r；
- `r` 是秩(rank),`α` 是缩放系数，二者控制补丁的"容量"和"强度"；
- 推理时可把 BA 合并进 W(`merge`),零额外延迟。

**昇腾视角**：`W·x` 和 `B·(A·x)` 都是矩阵乘，全部落在 **Cube 单元**;Cube 对规整大矩阵乘最高效，而 LoRA 的 A、B 很"瘦"(r 维很小),所以 LoRA 增量计算的开销相对主干极小。

### 2.2 QLoRA:量化 + LoRA(省显存王者)

QLoRA = 把冻结的底座权重**量化到 4bit** 存储 + 在其上做 LoRA。前向时 4bit 权重反量化回 bf16 参与计算，梯度只流向 LoRA 的 A/B。

> **昇腾注意**:GPU 上 QLoRA 依赖 `bitsandbytes` 的 NF4 内核，而 **bitsandbytes 的 CUDA 内核在昇腾上不可直接用**。昇腾侧的"4bit 微调"路径与 GPU 不同，需依赖昇腾官方提供的量化方案/适配(参见 [[msmodelslim]]),不能照搬 GPU 的 `load_in_4bit`。具体可用方案与版本以华为昇腾官方文档(Ascend 社区)为准。

### 2.3 其他成员一览

| 方法 | 注入位置 | 训练参数 | 直觉 |
| --- | --- | --- | --- |
| LoRA | 注意力/FFN 的线性层旁路 | A、B 低秩矩阵 | 学权重的低秩增量 |
| QLoRA | 同 LoRA + 底座 4bit | A、B | LoRA + 极致省显存 |
| Adapter | 每层后插小 MLP | 瓶颈 MLP | 串联小模块 |
| Prefix/P-Tuning | 注意力前缀/输入嵌入 | 虚拟 token | 学"软提示" |
| (IA)³ | 激活逐元素缩放 | 缩放向量 | 极少参数 |

---

## 3. PEFT 在昇腾的执行链路

一次 LoRA 训练 step 在昇腾上的旅程：

```
前向:
  input ──► Embedding ──► [冻结 W·x  (Cube)]
                            +  [A·x → B(...)  (Cube, 低秩)]  ──► 各层 ──► loss
反向:
  loss ──► 梯度只回流到 LoRA 的 A、B
            (冻结权重 requires_grad=False, 不分配梯度/动量)
  ↓
  优化器 (AdamW) 只为 A、B 维护一/二阶动量 ──► 显存省在这
  ↓
多卡时: A、B 的梯度做 allreduce  ──►  走 HCCL (非 NCCL)
```

要点：

1. **冻结即省显存**:`requires_grad=False` 的权重不产生梯度张量，优化器也不为其建动量缓冲，这是 PEFT 省显存的根因(不只是少算，是少存)。
2. **图模式(可选)**:MindSpore/部分 torch_npu 场景可走图编译(GE),把算子融合成图、减少下发开销；动态图(eager)更易调试。PEFT 调参阶段建议先 eager 跑通。
3. **集合通信走 HCCL**:多卡微调时梯度同步用 HCCL 的环/树算法，对应 GPU 的 NCCL allreduce。

---

## 4. 实操流程(讲含义,不写具体命令)

> 凡涉及确切命令、包名、版本号、镜像、路径，一律以**华为昇腾官方文档(Ascend 社区)为准**。下面只讲"每步为什么、依赖什么、易错点"。

1. **准备硬件与 CANN 环境**
   含义:装好昇腾驱动/固件,再装与之匹配的 CANN(toolkit + kernels)。
   依赖:CANN 版本必须与驱动/固件、与 torch_npu 版本**三方对齐**——这是最常见的环境地雷。

2. **安装 PyTorch + torch_npu**
   含义:装社区版 PyTorch,再装**版本严格对应**的 `torch_npu` 插件,代码里 `import torch_npu` 后 `.npu()` 才生效。
   易错点:torch 与 torch_npu 版本号必须配套,错配会出现算子找不到或 import 报错。

3. **安装训练栈:transformers + peft (+ accelerate)**
   含义:peft 库本身硬件无关,正常 pip 安装即可。
   注意:**不要照搬 GPU 教程里的 `bitsandbytes`**(QLoRA 4bit)——它的 CUDA 内核在昇腾不可用。

4. **改设备:`cuda` → `npu`**
   含义:把脚本里的 `device="cuda"`、`.cuda()`、`torch.cuda.xxx` 改成 `npu` 对应写法。
   提示:很多代码用 `accelerate`/`device_map`,设备识别可自动化,但仍需确认其识别到 NPU。

5. **配 LoRA(target_modules 是关键)**
   含义:用 `LoraConfig` 指定 `r`、`alpha`、`dropout` 和 **`target_modules`**(给哪些线性层加旁路,如 `q_proj/k_proj/v_proj/o_proj`)。
   易错点:`target_modules` 写错(模型层名不匹配)会导致"可训练参数为 0"或注入到错误层。

6. **混合精度:优先 bf16**
   含义:910B 系列对 bf16 友好,数值更稳;fp16 在某些算子上易溢出。

7. **跑通 → 多卡 → 调优**
   含义:先单卡 eager 跑通小步数,确认 loss 正常下降;再上多卡(HCCL);最后考虑图模式/算子融合提速。

8. **合并与部署**
   含义:训练完用 `merge_and_unload` 把 LoRA 合并进底座,或保留 adapter 单独加载;部署交给 MindIE / vllm-ascend。

---

## 5. 迁移要点与常见坑

| 坑 | 现象 | 根因 | 处理思路 |
| --- | --- | --- | --- |
| 版本三件套错配 | import 报错 / 算子缺失 | 驱动·CANN·torch_npu 未对齐 | 严格按官方版本配套表,三方一起定版 |
| 直接用 bitsandbytes | QLoRA 4bit 报错/不生效 | bnb 是 CUDA 内核 | 改用昇腾官方量化方案,不照搬 `load_in_4bit` |
| 可训练参数为 0 | loss 不降、训练无效 | `target_modules` 层名不匹配 | 打印 `print_trainable_parameters()` 核对 |
| 算子未覆盖 | "operator not implemented" | 某冷门算子昇腾尚未支持 | 换等价实现/升级 CANN/反馈社区 |
| fp16 溢出 NaN | loss 变 NaN | 部分算子 fp16 动态范围不足 | 优先 bf16 |
| `.cuda()` 残留 | 张量没上 NPU/报设备错 | 脚本写死 cuda | 全局替换为 npu,或用 device 变量 |
| 多卡不通信 | 卡住/超时 | HCCL 环境/网络未配好 | 检查 HCCL 配置与卡间互联([[HCCL]]) |
| 显存仍爆 | OOM | 序列长/batch 大,或没真省到 | 减 batch、开梯度检查点、确认底座已冻结 |

**性能调优思路(机制层面)**：

- **让 Cube 吃饱**:矩阵规整、batch 合理时 Cube 利用率高;LoRA 的瘦矩阵本身算力占比小,瓶颈常在主干。
- **减少下发开销**:eager 下每个算子单独下发,kernel 多时主机侧成为瓶颈;成熟后可试图模式/算子融合。
- **梯度检查点**:用算力换显存,长序列时显著降峰值。
- **通信与计算重叠**:多卡时让 HCCL allreduce 与反向计算重叠,掩盖通信延迟。

---

## 常见问题

| 问题 | 回答 |
| --- | --- |
| 昇腾上 PEFT 库要改吗? | 几乎不用。peft 硬件无关,改的是下面的 torch_npu / 设备字符串。 |
| LoRA 和 QLoRA 选哪个? | 显存够选 LoRA(简单稳);显存紧张才上 QLoRA,但昇腾 4bit 路径需用官方方案。 |
| 为什么不能用 bitsandbytes? | 它的内核是 CUDA 写的,昇腾没有对应实现。 |
| `r` 设多大? | 常 8/16/32;越大容量越强但参数越多,从小试起。 |
| 多卡微调梯度怎么同步? | 走 HCCL(对标 NCCL)做 allreduce。 |
| 训练完怎么部署? | merge 进底座或保留 adapter,交 MindIE / vllm-ascend 推理。 |
| 优先 fp16 还是 bf16? | 910B 优先 bf16,数值更稳、更不易溢出。 |
| 怎么确认 LoRA 真注入了? | 调 `print_trainable_parameters()`,看可训练参数是否非 0 且占比很小。 |

---

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-infra/算力/昇腾NPU]]
- [[ai-infra/ai-hardware/AI芯片软件生态]]
- [[ai-infra/ai-hardware/CUDA]]
- [[ai-infra/网络/NCCL]]
- [[ai-infra/网络/集合通信原语]]
- [[ai-framework/megatron-lm/README]]
- [[ai-framework/huggingface-transformers/README]]
- [[llm-compression/quantization/量化基础]]
- [[llm-inference/README]]
- [[llm-train/README]]
- [[llm-algo/transformer/模型架构]]
- 本仓相关:[[msmodelslim]] · [[mindie/README]] · [[vllm-ascend/README]] · [[modellink/README]] · [[HCCL]] · [[达芬奇架构]]
