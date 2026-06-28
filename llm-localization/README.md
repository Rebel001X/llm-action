# 大模型国产化适配(以昇腾 Ascend 为主线)

> 把"在英伟达 GPU 上跑通的大模型"迁到国产 NPU(昇腾/海光/寒武纪)上的全栈适配地图:硬件→驱动→CANN→框架→训练/推理套件,逐层对标 CUDA 世界,讲清"换了什么、坑在哪"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|--------------|--------|
| 0 | 一句话锚点:国产化适配到底在适配什么 | 软硬件栈 / 迁移 |
| 1 | 地基:昇腾软件栈分层 + 昇腾↔英伟达全生态对照表 | CANN↔CUDA / HCCL↔NCCL |
| 2 | 达芬奇架构原理:Cube/Vector/Scalar 三引擎 | DaVinci / 矩阵单元 |
| 3 | 软件栈分层机制:从驱动到套件每层干什么 | Driver/CANN/框架/套件 |
| 4 | 训练侧适配:MindFormers / ModelLink ↔ Megatron | 并行 / 套件 |
| 5 | 推理侧适配:MindIE ↔ TensorRT-LLM/vLLM | 引擎 / 服务化 |
| 6 | 量化压缩:msModelSlim ↔ GPTQ/AWQ | W8A8 / 校准 |
| 7 | 集合通信:HCCL ↔ NCCL 与环算法 | AllReduce / Ring |
| 8 | 其他国产生态:天数智芯 / 海光 / 寒武纪 / 魔搭 | 多元算力 |
| — | 迁移要点 / 注意事项与坑 | 必看 |
| — | 常见问题 + 跳转链接 | 速查 |

> 护栏说明:本文不写任何精确命令行、包名、版本号、镜像名或确切性能数字。凡涉及安装/docker/配置,只讲"这步在做什么、依赖谁、为什么要、常见坑";具体命令与版本一律**以华为昇腾官方文档(Ascend 社区)为准**。

---

## 0. 一句话锚点

**国产化适配 = 把模型代码里所有"默认绑定 CUDA/NCCL/cuDNN 的隐式假设"显式化,再替换成国产栈对应的实现。**

模型本身(Transformer 结构、权重)是与硬件无关的;真正"绑死英伟达"的是它脚下的那一摞软件:`torch.cuda` 设备语义、NCCL 通信、cuDNN/cuBLAS 算子、FlashAttention 这类手写 CUDA kernel、以及 vLLM/TensorRT-LLM 这类推理引擎。适配工作就是把这一摞**逐层替换**为昇腾对应物,并修掉替换后暴露出来的精度、性能、算子缺失问题。

---

## 1. 地基:昇腾软件栈的定位 + 对标英伟达生态

### 1.1 昇腾在哪一层

昇腾(Ascend)是华为的 AI 处理器品牌(训练/推理 NPU)。它和英伟达一样,卖的不只是芯片,而是**一整摞软件栈**。理解适配,先看清这摞栈的分层:

```
        ┌──────────────────────────────────────────────┐
 应用/套件 │  MindFormers · ModelLink · MindIE · msModelSlim │  ← 大模型训练/推理/量化套件
        ├──────────────────────────────────────────────┤
   框架   │   MindSpore   |   PyTorch(torch_npu 插件)     │  ← 训练框架(原生 or 适配)
        ├──────────────────────────────────────────────┤
        │           CANN(异构计算架构)                  │  ← 对标 CUDA 的核心
   软件栈 │   ├ 图引擎 GE   ├ 算子库 AOL   ├ 通信库 HCCL   │
        │   ├ 编译器     ├ Runtime      ├ ACL 接口     │
        ├──────────────────────────────────────────────┤
 驱动/固件 │        Driver & Firmware(NPU 驱动)            │
        ├──────────────────────────────────────────────┤
   硬件   │   昇腾 NPU(达芬奇架构,Cube/Vector/Scalar)    │
        └──────────────────────────────────────────────┘
```

**记忆要点:CANN 之于昇腾 = CUDA 之于英伟达。** 它是把上层框架的算子调用、翻译成 NPU 能执行的指令的那一层;你迁移时碰到的 90% 的"算子不支持""精度对不齐",根因都在这一层。

### 1.2 昇腾 ↔ 英伟达 生态对照表(最重要的迁移心智图)

| 维度 | 英伟达世界 | 昇腾世界 | 说明 / 对应关系 |
|------|-----------|---------|----------------|
| AI 加速芯片 | GPU(A100/H100…) | **NPU**(昇腾 910/310 系列) | 都是 AI 加速器,但 NPU 是 ASIC 化的达芬奇架构 |
| 底层计算平台 | **CUDA** | **CANN**(异构计算架构) | 核心对标。驱动+运行时+编译器+算子库的总集 |
| 设备运行时/底层 API | CUDA Runtime / Driver API | **ACL**(Ascend Computing Language) | 设备管理、内存、流、kernel 下发 |
| 深度学习算子库 | cuDNN / cuBLAS | **CANN 算子库 / AOL** | 卷积、矩阵乘、归一化等高性能算子 |
| 集合通信库 | **NCCL** | **HCCL**(Huawei Collective Comm Lib) | AllReduce/AllGather 等多卡多机通信 |
| 训练框架(原生) | (无直接原生绑定) | **MindSpore** | 华为自研全场景框架 |
| 训练框架(适配) | **PyTorch** | **PyTorch + torch_npu 插件** | 让 `torch` 把 `cuda` 语义映射到 NPU |
| 大模型训练套件 | **Megatron-LM** | **MindFormers / ModelLink** | 并行训练、主流模型库 |
| 大模型推理引擎 | **TensorRT-LLM / vLLM** | **MindIE**(MindIE-LLM / MindIE-Service) | 推理加速+服务化 |
| 量化压缩工具 | GPTQ / AWQ / TensorRT 量化 | **msModelSlim** | W8A8/W4A16 等低比特量化 |
| 设备选择语义 | `cuda` / `torch.cuda` | `npu` / `torch.npu`(经 torch_npu) | 代码里最直接的改动点 |
| 性能分析工具 | Nsight / nvprof | **msProf / Profiling 工具链** | 算子级耗时、流水分析 |
| 模型/数据中心 | HuggingFace Hub | **魔搭 ModelScope** | 国内模型与数据集托管社区 |

> 用法:迁移时把你代码里出现的左列名词逐个找到右列对应物,这张表就是你的"翻译词典"。

---

## 2. 机制:达芬奇(DaVinci)架构原理

达芬奇是昇腾 NPU 的计算核心架构。和 GPU "海量小核 SIMT" 的思路不同,达芬奇是**为张量运算专门设计的多引擎 AI Core**,核心是把矩阵乘做成硬件原语。

```
            ┌────────────── 一个 AI Core ──────────────┐
   数据流 →  │   ┌─────────┐   矩阵乘(GEMM)主力        │
            │   │  Cube   │   16x16x16 立方体单元       │  ← 算力大头
   片上缓存  │   │ (矩阵)  │   一拍完成一批 MAC          │
   (Buffer) │   └─────────┘                            │
            │   ┌─────────┐   逐元素 / 激活 / 归一化     │
            │   │ Vector  │   向量化 SIMD                │  ← 灵活算子
            │   └─────────┘                            │
            │   ┌─────────┐   标量控制 / 地址 / 分支     │
            │   │ Scalar  │   程序流控制                 │
            │   └─────────┘                            │
            └──────────────────────────────────────────┘
                Cube 啃 GEMM,Vector 啃 elementwise,Scalar 管调度
```

**三引擎分工:**
- **Cube(立方体)单元**:矩阵乘加专用,Transformer 里的 QKV 投影、FFN、Attention 打分这些 GEMM 全靠它。这是 NPU 算力密度高的来源。
- **Vector(向量)单元**:做 LayerNorm、Softmax、激活、逐元素加这些"非矩阵"运算。
- **Scalar(标量)单元**:做循环、分支、地址计算等控制流。

**对适配的含义:** 模型要跑得快,关键是让计算尽量落到 Cube 上,并让数据在片上 Buffer 里复用(减少与外部内存搬运)。形状不规整、过多碎小算子、频繁数据格式转换,都会让 Cube 空转、Vector/搬运成瓶颈——这是昇腾性能调优的核心矛盾。

---

## 3. 机制:软件栈分层,每层在适配中干什么

### 3.1 Driver & Firmware(驱动固件层)
让操作系统认识 NPU 卡。对标英伟达驱动。**坑:** 驱动、固件、CANN、框架插件之间有严格的版本配套关系——版本错配是新手第一大坑。具体配套矩阵以官方文档为准。

### 3.2 CANN(核心适配层)
对标 CUDA。内部关键子模块:
- **GE(Graph Engine,图引擎)**:把整张计算图编译、优化、下发。昇腾偏好"图模式"(整图编译执行)以发挥流水并行,这与 PyTorch 默认的"单算子即时执行(eager)"理念不同。
- **算子库 / AOL**:提供高性能算子实现,对标 cuDNN/cuBLAS。
- **Runtime + ACL**:设备/内存/流管理与 kernel 下发。
- **HCCL**:集合通信(见第 7 节)。

### 3.3 框架层:两条路线
- **MindSpore 路线**:华为原生框架,与 CANN 配合最紧,默认图模式,昇腾上性能与功能完整度最好。代价是要学新框架 API。
- **PyTorch + torch_npu 路线**:最受欢迎的迁移路径。`torch_npu` 是一个插件,让 PyTorch 多出一个 `npu` 后端设备。理想情况下,代码里把 `cuda` 改成 `npu` 即可大体跑通——但理想之外有大量算子覆盖与精度细节(见迁移要点)。

### 3.4 套件层
在框架之上提供"开箱即用的大模型能力":训练用 MindFormers/ModelLink,推理用 MindIE,量化用 msModelSlim。下面分别展开。

---

## 4. 训练侧适配:MindFormers / ModelLink ↔ Megatron

**定位:** 它们是昇腾上的"大模型训练套件",对标英伟达世界的 **Megatron-LM**:提供主流模型实现 + 张量并行/流水并行/数据并行/序列并行等分布式策略 + 训练脚本。

- **MindFormers**:基于 MindSpore 的大模型套件,模型库丰富,与 MindSpore 图模式深度配合。
- **ModelLink**(基于 PyTorch + torch_npu):面向习惯 Megatron/PyTorch 的用户,接口、并行划分思路与 Megatron 高度相似,迁移心智负担小。

```
   Megatron-LM(英伟达/PyTorch)          ModelLink / MindFormers(昇腾)
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │ TP / PP / DP / SP 并行     │  ≈≈≈→   │ 同样的 TP/PP/DP/SP 概念    │
   │ 通信走 NCCL               │         │ 通信走 HCCL               │
   │ 算子走 cuDNN/cuBLAS/手写   │         │ 算子走 CANN 算子库         │
   │ 设备 cuda                 │         │ 设备 npu                  │
   └──────────────────────────┘         └──────────────────────────┘
```

**迁移含义:** 并行切分的"概念"几乎一一对应,所以从 Megatron 迁来主要是换通信后端、换设备、对齐算子与超参,而不是重写并行逻辑。**坑:** 权重/切分格式在套件间不通用,常需做权重转换(checkpoint 格式转换),具体转换脚本以官方文档为准。

---

## 5. 推理侧适配:MindIE ↔ TensorRT-LLM / vLLM

**定位:** MindIE 是昇腾的大模型推理引擎与服务套件,对标英伟达的 **TensorRT-LLM(图编译加速)+ vLLM(服务化/调度)** 的组合。

它通常分为:
- **MindIE-LLM**:模型推理加速层,负责把模型在 NPU 上高效执行(融合算子、KV Cache 管理等)。对标 TensorRT-LLM。
- **MindIE-Service**:服务化层,提供推理服务、请求调度、批处理。其"连续批处理 / PagedAttention 式 KV 管理"的思想对标 vLLM。

```
   请求 ──→ ┌─────────────── MindIE ───────────────┐ ──→ 流式输出
            │  Service: 调度 / 续批 / KV 分页管理     │
            │  ─────────────────────────────────── │
            │  LLM:     融合算子 / Attention / 采样   │
            │  ─────────────────────────────────── │
            │  CANN + 达芬奇 Cube/Vector 执行         │
            └───────────────────────────────────────┘
            ≈ vLLM(调度)  +  TensorRT-LLM(算子加速)
```

**迁移含义:** 如果你原来用 vLLM 提供 OpenAI 兼容接口,迁到 MindIE 后服务接口形态相似,但模型需要转成 MindIE 支持的格式/配置,且并非所有结构都开箱支持——支持模型列表与转换流程以官方文档为准。

---

## 6. 量化压缩:msModelSlim ↔ GPTQ / AWQ

**定位:** msModelSlim 是昇腾的模型压缩工具,对标英伟达世界里 **GPTQ / AWQ / SmoothQuant** 这类后训练量化(PTQ)工具,目标是用 W8A8、W4A16 等低比特把显存与带宽压下来、把推理提上去。

量化的"原理"是跨硬件通用的(用少比特表示权重/激活,关键是选好缩放因子、保护离群值),msModelSlim 把这些算法工程化并对接 CANN/MindIE 的低比特算子。

**迁移含义与坑:**
- 量化后**必须在昇腾上重新做精度评估**,不能直接复用英伟达上的量化结果——底层低比特算子实现不同,精度表现会有差异。
- 校准数据(calibration set)的选择直接影响量化精度,这一点与 GPTQ/AWQ 完全一致。
- 量化产物要与下游推理引擎(MindIE)的算子支持对齐,否则"量化成功但跑不起来"。具体支持的量化方案与流程以官方文档为准。

---

## 7. 集合通信:HCCL ↔ NCCL 与环算法

**定位:** HCCL(Huawei Collective Communication Library)是昇腾的集合通信库,一一对标 **NCCL**:提供 AllReduce、AllGather、ReduceScatter、Broadcast 等原语,是多卡多机训练/推理的通信底座。

分布式训练里梯度同步靠 AllReduce,而 AllReduce 的经典高效实现是 **Ring(环)算法**——HCCL 与 NCCL 在这点上思想一致:

```
   Ring AllReduce(4 卡,简化示意)
   NPU0 ─→ NPU1 ─→ NPU2 ─→ NPU3 ─┐
     ↑                            │   每卡把自己的分块沿环传给下一卡,
     └────────────────────────────┘   边传边累加;两轮后所有卡持有全和。

   优势:每卡收发量 ≈ 2*(N-1)/N * 数据量,与卡数 N 几乎无关 → 带宽友好
```

**机制要点:** HCCL 会根据物理拓扑(卡间是 HCCS/片内总线还是跨节点 RoCE 网络)选择通信算法与路径。**坑:** 跨机通信对网络配置(RDMA/RoCE)敏感,集群组网与 rank 表配置错误是大规模训练起不来的常见原因;具体组网与环境变量以官方文档为准。

---

## 8. 其他国产生态(横向对照)

| 厂商/平台 | 定位 | 对标/类比 | 适配心智 |
|-----------|------|-----------|----------|
| **天数智芯(Iluvatar)** | 国产 GPGPU(通用 GPU 架构) | 更接近英伟达 GPGPU 路线 | 架构上更"像 GPU",CUDA 生态迁移路径相对短,但仍需其自有软件栈对接 |
| **海光(Hygon DCU)** | 国产 GPGPU(类 ROCm 生态) | 对标 AMD ROCm / HIP | 软件栈与 ROCm 同源思路,`hip` 化迁移 |
| **寒武纪(Cambricon MLU)** | 国产 AI 加速卡 | 自有 Neuware/CNNL 栈 | 类似昇腾,有自己的算子库与编程模型 |
| **魔搭 ModelScope** | 模型/数据集社区 | 对标 HuggingFace Hub | 国产模型权重、适配版本的主要来源 |

> 共性规律:GPGPU 路线(天数/海光)迁移时"换后端"成本相对低,因为编程模型贴近 CUDA/ROCm;ASIC 路线(昇腾/寒武纪)算力密度高但**算子覆盖**是适配主战场。

---

## 迁移要点 / 注意事项与坑(从 CUDA 迁到昇腾,必看)

1. **设备语义替换**:代码里 `cuda` 设备、`torch.cuda.*` 调用要换成 `npu`/`torch.npu.*`(经 torch_npu)。这是第一步也是最浅的一步。
2. **算子覆盖才是大头**:`torch_npu` 不可能 100% 覆盖所有 PyTorch 算子,尤其是自定义/冷门算子和手写 CUDA kernel(如某些 FlashAttention 变体)。缺失算子要么换等价实现、要么走昇腾提供的融合算子。**这是迁移工作量的真正来源。**
3. **精度对齐**:不同硬件的浮点累加顺序、低精度(fp16/bf16)实现细节不同,迁移后要做**端到端精度比对**(逐层输出对比 + 最终指标),不能"能跑"就当迁完了。
4. **图模式 vs eager**:昇腾偏好整图编译(图模式)以发挥流水并行。PyTorch 习惯的动态/eager 写法可能限制性能,某些情况下需要改造成图友好的写法。
5. **数据格式(layout)**:NPU 内部有自己偏好的数据排布(如 NZ 等格式),频繁的格式转换会拖慢性能。算子间格式不一致导致的隐式转换是隐蔽性能坑。
6. **通信后端切换**:分布式从 NCCL 换 HCCL,要确保 rank 表、拓扑、网络(RoCE/RDMA)配置正确。
7. **权重格式转换**:HF/Megatron 的 checkpoint 往往不能直接喂给 MindFormers/MindIE,需要转换;并行切分变了也要重切。
8. **版本配套**:驱动/固件/CANN/torch_npu/套件之间是强配套关系,版本矩阵以官方文档为准,错配会出现各种诡异报错。
9. **量化要重测精度**:见第 6 节,英伟达上的量化结论不能照搬。
10. **性能调优思路(机制层)**:让计算尽量落 Cube、增大片上数据复用、减少碎小算子和格式转换、用融合算子、合理设置并行与 batch——用 Profiling 工具找瓶颈,而非盲调。

> 再次提醒:以上**不含任何具体命令、包名、版本号、镜像名**。所有可执行细节、版本配套矩阵、支持模型列表、性能数据,**一律以华为昇腾官方文档(Ascend 社区)为准**。

---

## 常见问题

| 问题 | 解答 |
|------|------|
| CANN 等于 CUDA 吗? | 定位等价(都是芯片厂商的底层异构计算平台),但不是二进制兼容;CUDA 代码不能直接在 CANN 上跑,要走框架/插件适配。 |
| 把 `cuda` 改成 `npu` 就迁完了? | 远没有。设备替换是最浅一层,算子覆盖、精度对齐、格式与图模式才是真正工作量。 |
| MindSpore 和 torch_npu 选哪个? | 追求昇腾上最佳性能与功能完整度可选 MindSpore;追求迁移成本低、复用 PyTorch 生态走 torch_npu。 |
| MindIE 对应英伟达什么? | 思想上 ≈ TensorRT-LLM(算子加速)+ vLLM(服务化/续批)的合体。 |
| HCCL 和 NCCL 是同一套吗? | 不是同一实现,但原语与算法思想(如 Ring AllReduce)对应,迁移时换后端即可,概念不变。 |
| 国产卡里哪类迁移最省力? | GPGPU 路线(天数智芯、海光 ROCm 系)编程模型贴近 CUDA/ROCm,换后端成本相对低;ASIC 路线(昇腾/寒武纪)算力强但算子适配工作量大。 |
| 量化结果能从 A100 直接搬到昇腾吗? | 不能,低比特算子实现不同,必须在昇腾上重新评估精度。 |

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

---

## 参考来源(原始仓库链接)

- [大模型国产化适配-华为昇腾AI全栈软硬件平台总结](https://github.com/liguodongiot/llm-action/blob/main/docs/llm_localization/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%BD%E4%BA%A7%E5%8C%96%E9%80%82%E9%85%8D-%E5%8D%8E%E4%B8%BA%E6%98%87%E8%85%BEAI%E5%85%A8%E6%A0%88%E8%BD%AF%E7%A1%AC%E4%BB%B6%E5%B9%B3%E5%8F%B0%E6%80%BB%E7%BB%93.md)
- 昇腾训练套件 ModelLink:https://gitee.com/ascend/ModelLink
- 昇腾 Ascend 处理器介绍:https://huahuaboy.blog.csdn.net/article/details/127171363
- 华为 Ascend(昇腾)910 结构分析:https://blog.csdn.net/evolone/article/details/100061616
