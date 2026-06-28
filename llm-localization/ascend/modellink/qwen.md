# ModelLink 训练 Qwen(昇腾)

> 在昇腾 NPU 上用 ModelLink(AscendSpeed)做 Qwen 系列大模型的预训练 / 微调:它是华为对标 Megatron-LM 的国产化大模型训练套件。📍 导航:[[00-知识地图]]
> 🔗 相关:[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-framework/megatron-lm/README]] [[llm-train/README]] [[llm-algo/transformer/模型架构]]

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0. 一句话锚点 | ModelLink 是谁、训 Qwen 干嘛 | ModelLink / Megatron / 昇腾 |
| 1. 地基与对照表 | 昇腾训练栈定位 + 昇腾↔英伟达迁移心智图 | NPU / CANN / torch_npu / AscendSpeed |
| 2. 整体流程 | 从克隆到训练的 5 步流水线 | 权重转换 / 数据预处理 / 启动脚本 |
| 3. 并行机制 | TP/PP/DP + 序列并行怎么在 NPU 上跑 | 张量/流水/数据并行 |
| 4. 权重转换原理 | HF ↔ Megatron 权重为什么要切分重排 | ckpt convert |
| 5. Qwen 结构适配 | Qwen 相对 LLaMA 改了哪些点 | QKV bias / GQA / RoPE |
| 6. 迁移要点与坑 | 从 GPU+Megatron 搬到 NPU 要改什么 | 算子 / 精度 / 亲和 |
| 常见问题 | 速查 | FAQ |

## 0. 一句话锚点

**ModelLink(仓库现也整合进 MindSpeed / AscendSpeed 体系)是华为昇腾官方的"大模型分布式训练加速套件",定位等价于英伟达世界的 Megatron-LM。** 它在 PyTorch + `torch_npu` 之上,把 Megatron-LM 的张量并行、流水线并行、序列并行等能力适配到昇腾 NPU,并预置了一批主流模型(Qwen、LLaMA、Baichuan、GLM 等)的转换脚本与训练配置。

训练 Qwen,本质就是:**把 HuggingFace 格式的 Qwen 权重转换成 Megatron 切分格式 → 把语料预处理成索引化的 bin/idx → 用分布式启动脚本拉起多卡训练**。

> ⚠️ 本文聚焦"机制与流程含义"。所有精确命令、commit 哈希、whl 包名、CANN/torch_npu 版本号,**一律以华为昇腾官方文档(Ascend 社区 / Gitee ModelLink 仓库 README)为准**,本文不写死。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层(ModelLink 在哪一层)

```
┌─────────────────────────────────────────────────────────┐
│  套件层   ModelLink / MindSpeed (训练)   MindIE (推理)    │  ← 本文在这
├─────────────────────────────────────────────────────────┤
│  框架层   PyTorch + torch_npu 插件   |  MindSpore        │
├─────────────────────────────────────────────────────────┤
│  加速库   AscendSpeed / Apex(NPU)  分布式 & 融合算子      │
├─────────────────────────────────────────────────────────┤
│  CANN 层  GE 图引擎 / ACL / 算子库(AOL) / HCCL 通信库     │  ← 对标 CUDA
├─────────────────────────────────────────────────────────┤
│  驱动固件 Driver + Firmware(NPU 设备管理)                │
├─────────────────────────────────────────────────────────┤
│  硬件层   Ascend 910 系列 NPU(达芬奇架构 Cube/Vector)    │  ← 对标 GPU
└─────────────────────────────────────────────────────────┘
```

ModelLink 不直接碰硬件:它调用 PyTorch 算子 → `torch_npu` 把算子下发给 CANN → CANN 编译/调度到达芬奇核 → 多卡间用 HCCL 做集合通信。它的价值在"上层":并行切分策略、训练循环、权重格式、模型适配。

### 1.2 昇腾 ↔ 英伟达生态对照表(迁移心智图)

| 层次 | 昇腾(Ascend) | 英伟达(NVIDIA) | 说明 |
|---|---|---|---|
| 加速硬件 | NPU(910B 等) | GPU(A100/H100) | 计算单元不同:Cube/Vector vs CUDA Core/Tensor Core |
| 底层软件栈 | CANN | CUDA | 编程模型 + 运行时 + 编译 |
| 算子库 | AOL / 融合算子 | cuDNN / cuBLAS | 高性能卷积/矩阵/注意力等 |
| 集合通信 | HCCL | NCCL | AllReduce/AllGather 等多卡通信 |
| 设备插件 | torch_npu | (PyTorch 原生 CUDA) | 把 PyTorch 接到加速器 |
| 混合精度 | Apex(NPU 版) | Apex / AMP | fp16/bf16 训练 |
| **训练套件** | **ModelLink / MindSpeed** | **Megatron-LM** | **本文主角,大模型分布式训练** |
| 训练框架(国产) | MindFormers + MindSpore | — | 另一条纯国产技术路线 |
| 推理引擎 | MindIE | TensorRT-LLM / vLLM | 部署侧 |
| 量化工具 | msModelSlim | GPTQ / AWQ 工具 | 压缩侧 |
| 模型与数据 | ModelScope(魔搭) | HuggingFace Hub | 权重/数据来源 |

> 记住一句:**ModelLink ≈ "昇腾上的 Megatron-LM"**。ModelLink 早期甚至直接复用了 Megatron-LM 的部分源码(把 `megatron/` 目录拷进来),再用 AscendSpeed 替换/补丁掉与硬件相关的部分。所以读 Megatron 代码的经验几乎可以直接迁移。

## 2. 整体训练流程(从克隆到出 loss 的 5 步)

```
   HuggingFace Qwen 权重            原始语料(jsonl/文本)
          │                                │
   ① ckpt 转换                       ② 数据预处理
   HF → Megatron 切分格式            tokenizer → bin/idx 索引
          │                                │
          └──────────┬─────────────────────┘
                     ▼
            ③ 编写/选择训练脚本(设定 TP/PP/DP、超参、模型结构)
                     ▼
            ④ 分布式启动(torchrun/分布式拉起,HCCL 建链)
                     ▼
            ⑤ 训练循环出 loss → 周期保存 Megatron ckpt
                     ▼
            (训完再 ⑥ Megatron → HF 反向转换,交给推理引擎)
```

各步**含义**(具体命令以官方为准):

1. **环境与代码准备**:克隆 ModelLink 仓库,拷入对应 commit 的 Megatron-LM 源码,创建 `logs/model_from_hf/dataset/ckpt` 等目录;新建 conda 环境,安装 `torch`、`torch_npu`、`apex`(NPU 版)、`AscendSpeed` 及 `requirements`。**坑**:torch 与 torch_npu 版本必须严格配对,且要与已装的 CANN 版本匹配,错配会在 import 阶段或首个算子下发时报错。

2. **权重转换(① HF→Megatron)**:把魔搭/HF 下载的 Qwen 权重,按目标 TP/PP 切分重排成 Megatron 的 `mp_rank_xx` 目录结构(详见第 4 节)。

3. **数据预处理(②)**:用 Qwen 的 tokenizer 把语料切成 token,落成 Megatron 的 `.bin`(token 流)+ `.idx`(样本偏移索引)二进制格式,训练时内存映射读取,避免反复 tokenize。

4. **训练启动(③④)**:训练脚本里设定模型超参(层数、hidden、heads、是否 GQA、词表大小)、并行度(TP×PP×DP = 总卡数)、batch/序列长度/学习率,以及 NPU 相关开关(融合算子、确定性计算等)。通过分布式启动器在多卡/多机拉起,HCCL 负责建链与通信。

5. **反向转换(⑥)**:训练产物是 Megatron 格式 ckpt,要部署时需再转回 HF 格式,交给 MindIE / vLLM-Ascend 等推理侧。

## 3. 并行机制:TP / PP / DP / SP 在 NPU 上怎么跑

ModelLink 沿用 Megatron 的 3D 并行,只是底层 AllReduce 走 HCCL 而非 NCCL。

```
  全局卡数 = TP × PP × DP
  ┌── TP(张量并行)── 把一个矩阵乘按列/行切到组内多卡,前后插 AllReduce/AllGather
  │     例:Attention 的 QKV、FFN 的两个线性层
  ├── PP(流水线并行)── 把不同 Transformer 层放到不同卡,micro-batch 流水
  │     例:48 层切 4 段,每段 12 层,用 1F1B 调度填满气泡
  ├── DP(数据并行)── 不同卡跑不同数据,梯度 AllReduce 同步
  └── SP(序列并行)── 在 TP 组内沿序列维切 LayerNorm/Dropout,省激活显存
```

- **TP 组内通信最重**(每层多次 AllReduce),所以 TP 一般限制在单机内(910 多卡间高速互联),跨机走 PP/DP。这点和 GPU 上 Megatron 的拓扑感知是同一思路,只是把"NVLink 域"换成昇腾的"片间高速互联域"。
- **重计算(activation recomputation)**:用算力换显存,长序列/大模型几乎必开。
- **HCCL 的环算法**:AllReduce 在环形拓扑上分 Reduce-Scatter + AllGather 两阶段,带宽利用率与 NCCL 的 ring 思路一致(原理详见 [[ai-infra/网络/集合通信原语]])。

## 4. 权重转换原理:为什么不能直接 load HF 权重

HF 的 Qwen 权重是"单卡整权重 + HF 命名",Megatron 要的是"按 TP/PP 切好 + Megatron 命名"。转换脚本做三件事:

```
HF 权重 (qkv 合一/分离, gate_proj/up_proj 分开)
        │  ① 改名映射:HF 层名 → Megatron 层名
        │  ② 张量切分:按 TP 度沿正确维度切(列并行切列/行并行切行)
        │  ③ 算子合并:QKV 三个权重 cat 成一个、gate+up 合成一个
        ▼
Megatron 切分权重  iter_xxx/mp_rank_00 ~ mp_rank_(TP*PP-1)/
```

关键细节:
- **切分维度不能错**。列并行层(如 FFN 第一层、QKV)沿输出维切;行并行层(如 FFN 第二层、Attention 输出投影)沿输入维切。切错会"能加载但 loss 不收敛"。
- **GQA(分组查询注意力)**:Qwen2 用 GQA,K/V 头数 < Q 头数。切 TP 时要保证每张卡分到的 KV 头数整除,否则报错或精度异常。
- **改 TP/PP 度要重新转换**:Megatron ckpt 与并行度绑定,想从 TP=8 改成 TP=4,必须重跑转换(或用 ckpt 重切工具)。

## 5. Qwen 结构适配:相对 LLaMA 改了什么

ModelLink 对每个模型有专门配置,Qwen 与 LLaMA 同属 decoder-only,但有几处必须在脚本/转换里对齐:

| 特性 | Qwen 的做法 | 训练脚本/转换需注意 |
|---|---|---|
| QKV 偏置 | Qwen1/2 注意力 QKV 带 bias(LLaMA 无) | 转换要保留并正确切分 bias |
| 归一化 | RMSNorm | 选对 norm 类型 |
| 位置编码 | RoPE | base/维度对齐,长序列看是否扩展 |
| 激活 | SwiGLU(gate+up) | gate_proj/up_proj 合并切分 |
| 注意力 | Qwen2 用 GQA | num_key_value_heads 与 TP 协调 |
| 词表 | 较大(15万级) | embedding 列并行切分、padding 对齐 |

**机制提醒**:Qwen 早期 QKV 带 bias 是和 LLaMA 最容易踩的差异点,转换脚本若按 LLaMA 模板套用、漏掉 bias,会出现"权重加载不报错但首步 loss 异常大"。

## 迁移要点 / 注意事项与坑

从 **GPU + Megatron-LM** 搬到 **NPU + ModelLink**,主要改动与高频坑:

1. **设备与后端替换**:`cuda` → `npu`,通信后端 `nccl` → `hccl`,引入 `import torch_npu`。多数已被套件封装,但自写代码/补丁要注意。
2. **版本铁三角**:**驱动固件 ↔ CANN ↔ torch_npu** 三者版本必须匹配,再加上 ModelLink/AscendSpeed 要求的特定 Megatron commit。任一错配都可能"装上但跑不起来"。版本对应表以官方文档为准。
3. **算子覆盖与回退**:个别 CUDA 上的融合算子在 NPU 上可能用昇腾自研融合算子(如 FlashAttention 的 NPU 实现)或回退到小算子组合。开启 NPU 融合注意力通常能显著提速;若某算子不支持会报"算子未注册/未实现"。
4. **精度**:优先 **bf16**(910 系列对 bf16 友好),fp16 易溢出。混合精度 loss scale、确定性计算开关行为可能和 GPU 略有差异,复现实验前先对齐随机性配置。
5. **亲和性调优(机制层)**:绑定 NPU 与 CPU/网卡的 NUMA 亲和、合理设置 HCCL 环境、开启序列并行与重计算、把 TP 限制在单机内——这些是性能调优的主要抓手,思路与 GPU 一致,只是参数名换成昇腾的。
6. **数据格式可复用**:Megatron 的 bin/idx 数据格式两边通用,数据预处理脚本迁移成本低。
7. **转换-训练-反转换闭环**:别忘了训完要转回 HF 才能进推理引擎;转换脚本的并行度参数要与训练时一致。

## 常见问题

| 问题 | 答案 |
|---|---|
| ModelLink 和 Megatron-LM 什么关系? | 昇腾对标版,早期直接复用 Megatron 源码 + AscendSpeed 补丁,API/思路高度一致 |
| ModelLink 和 MindFormers 选哪个? | ModelLink 走 PyTorch+torch_npu 路线(迁移成本低);MindFormers 走纯国产 MindSpore 路线 |
| 为什么不能直接 load HF 的 Qwen 权重? | 需按 TP/PP 切分重排 + 改名 + 合并 QKV/gate-up,见第 4 节 |
| TP 应该设多大? | 通信最重,一般限制在单机内卡数;跨机用 PP/DP |
| Qwen 转换最易踩的坑? | 漏掉 QKV bias、GQA 的 KV 头数与 TP 不整除、切分维度搞反 |
| 用 fp16 还是 bf16? | 910 系列优先 bf16,fp16 易溢出 |
| 具体命令和版本去哪查? | 华为昇腾官方文档 / Gitee ModelLink 仓库 README(本文不写死) |
| 训完怎么部署? | Megatron→HF 反向转换后,交给 MindIE / vLLM-Ascend |

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
