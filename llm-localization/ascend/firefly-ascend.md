# Firefly 在昇腾 NPU 上的训练/微调

> Firefly 是一个开源中文大模型「全流程训练/微调」框架；本文讲它如何跑在昇腾(Ascend)NPU 上，以及从 CUDA/GPU 迁过来要改什么。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-framework/huggingface-transformers/README]] [[llm-train/README]]

## 阅读地图

| 你想知道的 | 看哪一节 |
| --- | --- |
| Firefly 是什么、它在昇腾栈的哪一层 | §0、§1 |
| 昇腾生态 ↔ 英伟达生态怎么一一对应 | §1 对照表 |
| 一次 NPU 微调从环境到落盘的整体流程 | §2 |
| Firefly 训练流水线在昇腾上跑通的内部链路 | §3 |
| torch_npu / Adapter 到底替换了什么 | §4 |
| 从 CUDA 迁到昇腾要改哪些代码 | §5 迁移要点 |
| 容易踩的坑(算子缺失/精度/显存/通信) | §6 注意事项与坑 |
| 常见问题速查 | 常见问题表 |

## 0. 一句话锚点

**Firefly = 「数据 → SFT/全参/LoRA/QLoRA → 评测 → 导出」的训练编排层；昇腾化的本质，是把它底下的 PyTorch 计算后端从 CUDA 换成 `torch_npu`(CANN)，上层训练逻辑尽量不动。** 你写的还是 HuggingFace `Trainer` 风格代码，变的是「算子在谁家硅片上执行」。

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层(自下而上)

```
┌─────────────────────────────────────────────┐
│  训练编排层   Firefly(SFT/LoRA/QLoRA/全参)   │ ← 本文主角
├─────────────────────────────────────────────┤
│  生态库       HF Transformers / PEFT / TRL   │
│               DeepSpeed / accelerate         │
├─────────────────────────────────────────────┤
│  框架后端     PyTorch + torch_npu(Adapter)   │ ← 关键的「换硅片」层
├─────────────────────────────────────────────┤
│  异构计算架构 CANN(算子库 + 图引擎 + Runtime)│ ← 对标 CUDA
├─────────────────────────────────────────────┤
│  集合通信     HCCL                            │ ← 对标 NCCL
├─────────────────────────────────────────────┤
│  驱动/固件    Ascend Driver + Firmware        │
├─────────────────────────────────────────────┤
│  硬件         昇腾 NPU(达芬奇 Cube/Vector)   │ ← 对标 GPU
└─────────────────────────────────────────────┘
```

Firefly 本身**不感知硬件**——它调 HuggingFace 生态；真正决定「跑在 NPU 还是 GPU」的是底下那层 `torch_npu`。所以「Firefly 昇腾化」并不是改 Firefly 大量源码，而是：**让 PyTorch 的设备后端从 cuda 变成 npu**，再把 Firefly 里硬编码 `cuda` 的零星地方擦掉。

### 1.2 昇腾 ↔ 英伟达 生态对照表(迁移心智图)

| 能力 / 角色 | 英伟达世界 | 昇腾世界 | 在 Firefly 微调里扮演 |
| --- | --- | --- | --- |
| 加速硬件 | GPU(SM/Tensor Core) | NPU(达芬奇 Cube/Vector 单元) | 真正算矩阵乘的硅片 |
| 异构计算平台 | CUDA | CANN | 提供 Runtime + 算子编译 |
| 高性能算子库 | cuDNN / cuBLAS | CANN 内置算子(AOL/aclnn 等) | 卷积/矩阵乘/归一化的底层实现 |
| 框架后端插件 | PyTorch 原生 CUDA 后端 | `torch_npu`(PyTorch Adapter) | 把 `tensor.to('npu')` 接到 CANN |
| 集合通信 | NCCL | HCCL | 多卡数据并行的 AllReduce |
| 训练大模型套件 | Megatron-LM | MindFormers / ModelLink | 大规模并行训练(非 Firefly 路线) |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 微调后部署上线 |
| 量化工具 | GPTQ / AWQ / bitsandbytes | msModelSlim | QLoRA 的低比特权重来源 |
| 混合精度类型 | FP16 / BF16 | FP16 / BF16(NPU 同样支持) | 训练吞吐与显存平衡 |
| 设备字符串 | `"cuda"` / `"cuda:0"` | `"npu"` / `"npu:0"` | 代码里最高频要改的点 |
| 显存查询 | `nvidia-smi` | `npu-smi info` | 看占用与卡间负载 |

**一句话记法**：把脑子里所有 `cuda → npu`、`NCCL → HCCL`、`cuDNN/cuBLAS → CANN 算子`、`bitsandbytes → msModelSlim`、`nvidia-smi → npu-smi` 各替换一遍，Firefly 的训练心智模型几乎不变。

## 2. 一次 NPU 微调的整体流程(讲含义，不背命令)

下面是「在昇腾上用 Firefly 微调」从零到产出的**步骤含义与依赖关系**。每一步**为什么要做**讲清楚；凡涉及精确命令/镜像名/版本号，一律以华为昇腾官方文档(Ascend 社区)为准。

```
[1] 准备昇腾基础环境
     Driver + Firmware + CANN(toolkit/kernels) + torch_npu
            │  依赖：版本必须三方匹配(见坑①)
            ▼
[2] 进入容器并 source 环境变量
     source .../set_env.sh  → 让 shell 找到 CANN 的库和工具
            │
            ▼
[3] 拉起 Firefly 代码 + 安装 Python 依赖
     transformers / peft / accelerate / deepspeed(可选)
            │  注意：bitsandbytes 等 CUDA 专属库要替换或绕过(坑③)
            ▼
[4] 准备数据集 + 写训练配置(json/yaml)
     指定 base model / 数据路径 / LoRA 超参 / batch / 精度
            │
            ▼
[5] 启动训练(单卡 or 多卡 HCCL)
     import torch_npu → 设备走 npu → 算子下发到 CANN
            │
            ▼
[6] 落盘 checkpoint / 合并 LoRA 权重 / 导出
            │
            ▼
[7] (可选)用 MindIE 部署推理 上线
```

文件顶部那几段 `conda activate llm-dev` + `source set_env.sh` + `sh run_xxx_npu.sh` 正好对应 **[2][3][5]**：先激活 conda 环境、再 source CANN 环境变量(否则找不到 NPU)、最后跑昇腾版训练脚本。`run_all_npu.sh` 是全参/SFT、`run_lora_npu.sh` 是 LoRA。

> 关键依赖点：**[2] 的 `source set_env.sh` 不能省**。它把 CANN 的算子库、`ASCEND_HOME`、Runtime 路径注入当前 shell。少了它，`import torch_npu` 会找不到底层库 → 后面所有训练步骤直接报错。这是新手最常见的「没报硬件错却起不来」的根因。

## 3. Firefly 训练流水线在昇腾上跑通的内部链路

```
 Firefly train.py
      │ 1. load_dataset()           ← CPU 侧，与硬件无关
      │ 2. AutoModelForCausalLM     ← 权重先在 CPU/host
      │ 3. model.to("npu")          ← ★ 这一步把权重搬上 NPU
      ▼
 HuggingFace Trainer.train()
      │ forward / backward
      ▼
 PyTorch 算子 dispatch
      │ 设备是 npu → 不走 CUDA kernel
      ▼
 torch_npu(PyTorch Adapter)
      │ 把 aten 算子翻译成 CANN 调用
      ▼
 CANN(算子编译 + Runtime)
      │ Cube 单元算 MatMul / Vector 单元算 Norm/激活
      ▼
 达芬奇 NPU 执行 ──(多卡时)── HCCL AllReduce 同步梯度
```

要点：

- **达芬奇架构**把矩阵乘交给 **Cube 单元**(类比 Tensor Core)、把逐元素/归约交给 **Vector 单元**。Transformer 里的 QKV/FFN 大矩阵乘吃满 Cube，是 NPU 算力发挥的关键。
- **图模式 vs 单算子模式**：CANN 既支持「单算子(eager)逐个下发」也支持「整图编译优化」。训练初期常用 eager 便于调试；追吞吐时可探索图模式，但要确认所用算子在图模式下都被支持(坑④)。
- **多卡**用 HCCL(对标 NCCL)做 AllReduce 同步梯度。Firefly 用 `accelerate`/`deepspeed` 拉起多进程时，通信后端要选 HCCL 而非 NCCL。

## 4. torch_npu / Adapter 到底替换了什么

`torch_npu` 不是「新框架」，而是给原生 PyTorch **挂了一个 NPU 设备后端**。它的职责：

1. 注册 `"npu"` 设备类型，让 `tensor.to("npu")`、`torch.device("npu")` 合法。
2. 把 PyTorch 的 `aten` 算子(MatMul、Softmax、LayerNorm...)路由到 CANN 对应实现。
3. 暴露 `torch_npu.npu.xxx` 系列接口(类比 `torch.cuda.xxx`)：内存、流、同步等。

所以 Firefly 昇腾化的「最小改动」其实就两类：
- 在程序入口 `import torch_npu`(否则 npu 后端没注册);
- 把所有 `cuda` 字样换成 `npu`(设备、amp、显存查询)。

## 5. 迁移要点：从 CUDA 版 Firefly 改到昇腾(代码层面)

| 类别 | CUDA 写法 | 昇腾要改成 | 说明 |
| --- | --- | --- | --- |
| 引入后端 | (无) | `import torch_npu` | 必须，且尽量靠前 |
| 设备字符串 | `"cuda"` / `.cuda()` | `"npu"` / `.npu()` | 全局搜索替换 |
| 设备上下文 | `torch.cuda.set_device(i)` | `torch_npu.npu.set_device(i)` | 多卡指定卡号 |
| 混合精度 | `torch.cuda.amp` | `torch.npu.amp` / autocast(device_type="npu") | autocast 设备类型要对 |
| 通信后端 | `backend="nccl"` | `backend="hccl"` | 分布式 init 时 |
| 显存查询 | `nvidia-smi` | `npu-smi info` | 运维侧 |
| 量化 | bitsandbytes(4bit/8bit) | msModelSlim 产出的低比特权重 | QLoRA 关键差异(见坑③) |
| FlashAttention | `flash-attn`(CUDA kernel) | CANN 融合注意力算子 | 不能直接装 flash-attn |

> 心法：**「上层逻辑别动，底层设备名全换」**。Firefly 的数据处理、LoRA 注入、Trainer 配置都与硬件无关；真正要改的是「张量去哪台硅片」这层薄薄的胶水。改完后 `grep -ri cuda` 把残留的硬编码 `cuda` 找干净，是迁移收尾的必做动作。

## 6. 注意事项与常见坑

- **坑①「三件套版本必须互相匹配」**：Driver/Firmware ↔ CANN ↔ torch_npu 三者版本是强绑定的，版本错配是「装好了却 import 失败」的头号原因。**具体配套版本以华为昇腾官方文档(Ascend 社区)的版本配套表为准**，不要凭印象拼版本。

- **坑②「忘记 source 环境变量」**：新开 shell / 新进容器没 `source set_env.sh`，CANN 库路径没注入 → `import torch_npu` 报找不到 so。每个训练脚本开头都应有这一步(就像本文件顶部那几段那样)。

- **坑③「QLoRA 的 bitsandbytes 不能直接用」**：bitsandbytes 的 4bit/8bit kernel 是 CUDA 专属，昇腾上不能照搬。NPU 路线的低比特要走 **msModelSlim**(对标 GPTQ/AWQ)产出量化权重，再喂给 LoRA。直接 `pip install bitsandbytes` 跑 QLoRA 是典型踩坑点。

- **坑④「某些算子/融合在 NPU 上缺失或未走加速路径」**：FlashAttention 这类 CUDA 手写 kernel 不存在于昇腾——要用 **CANN 提供的融合注意力算子**替代。若遇到「算子未支持」报错，思路是：换等价标准算子、或升级 CANN、或确认是否退回 eager 模式。

- **坑⑤「精度与溢出」**：FP16 在大模型上易溢出，昇腾同样支持 **BF16**，长序列/大模型优先 BF16 + 适当 loss scaling。混合精度的 autocast 设备类型要写 `npu`，否则 amp 实际没生效。

- **坑⑥「多卡通信选错后端」**：分布式必须用 **HCCL** 不是 NCCL。同时注意卡间拓扑——HCCL 的环/树算法在不同 NPU 互联拓扑下性能差异明显，多机时尤其要关注 rank 编排是否贴合物理拓扑。

- **坑⑦「性能调优是机制层面的」**：吞吐不达预期，先看是不是 (a) batch/序列长度没喂满 Cube、(b) 频繁 host↔device 拷贝、(c) 该用图模式编译却在 eager、(d) 通信占比过高。这些是「机制层」问题，不是换个 flag 就能解决——**精确调优参数与 profiling 工具用法以昇腾官方文档为准**。

## 常见问题

| 问题 | 答案 |
| --- | --- |
| Firefly 跑昇腾要重写框架吗？ | 不用。换 `torch_npu` 后端 + 擦掉硬编码 `cuda` 即可，上层训练逻辑基本不动。 |
| 昇腾上能跑 QLoRA 吗？ | 能，但低比特权重要走 msModelSlim，不是 bitsandbytes。 |
| 多卡怎么同步梯度？ | HCCL 做 AllReduce(对标 NCCL)，分布式 backend 选 `hccl`。 |
| 为什么 import torch_npu 失败？ | 多半是没 source set_env.sh，或 Driver/CANN/torch_npu 版本不配套。 |
| FlashAttention 能装吗？ | 不能直接装 CUDA 版；用 CANN 的融合注意力算子替代。 |
| 用 FP16 还是 BF16？ | 大模型优先 BF16，更不易溢出；FP16 需配 loss scaling。 |
| 微调完怎么部署？ | 合并 LoRA 后用 MindIE 推理(对标 TensorRT-LLM/vLLM)。 |
| 显存怎么看？ | `npu-smi info`(对标 nvidia-smi)。 |
| 具体版本/命令去哪查？ | 一律以华为昇腾官方文档(Ascend 社区)为准，切勿凭记忆拼。 |

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
