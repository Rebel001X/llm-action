# ModelLink:昇腾大模型分布式训练套件

> ModelLink 是华为昇腾官方的「大模型训练加速库」,在 NPU 上把 Megatron-LM 那套 3D 并行能力跑起来——你可以把它理解成「昇腾世界的 Megatron-LM」。📍 导航:[[00-知识地图]]
> 🔗 相关:[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-framework/megatron-lm/README]] [[ai-framework/huggingface-transformers/README]] [[llm-train/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点 | ModelLink = NPU 版 Megatron |
| 1 | 它在昇腾软件栈的位置 + 昇腾↔英伟达对照表 | CANN / AscendSpeed / Megatron |
| 2 | 软件栈分层 ASCII 图 | 硬件→CANN→框架→套件 |
| 3 | 端到端训练流水线五步 | HF→权重转换→预处理→训练→回转 |
| 4 | 权重转换机制(本仓库的 convert_ckpt) | TP/PP 切分 / w-pack |
| 5 | 3D 并行在 NPU 上怎么落地 | TP / PP / DP + HCCL |
| 6 | AscendSpeed 与融合算子 | flash-attn / rms-norm 融合 |
| — | 从 GPU 迁移到 ModelLink 的要点与坑 | 迁移清单 |
| — | 常见问题表 + 跳转链接 | FAQ |

## 0. 一句话锚点

**ModelLink 让你用「几乎和 Megatron 一样的命令行参数」,在昇腾 NPU 集群上完成 LLM 的预训练 / 增量预训练 / 指令微调。**

它不是一个新框架,而是「Megatron-LM 的核心思想 + 昇腾后端适配 + 一批主流模型(LLaMA/Baichuan/Qwen/GPT 等)的开箱配方」的集合。本仓库给出的就是把一个 HuggingFace 的 Baichuan2-7B 权重,转成 ModelLink 内部的 tp/pp 切分格式,再喂给训练脚本的最小路径。

> 仓库地址、分支、依赖版本、`set_env.sh` 路径等,**一律以华为昇腾官方文档(Ascend 社区 / Gitee ModelLink 仓库 README)为准**,本文只讲「每一步在做什么、为什么要做、容易踩什么坑」。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

ModelLink 处于昇腾软件栈的**最上层「套件层」**:它向下依赖 PyTorch(经 torch_npu 适配到昇腾)、再向下依赖 CANN(昇腾的「CUDA」),最底层才是达芬奇架构的 NPU 硬件。它对标的是英伟达世界里「Megatron-LM」这一层——负责把单卡的训练逻辑,沿张量/流水线/数据三个维度切到成百上千张卡上。

**昇腾 ↔ 英伟达 生态对照表(迁移心智图,最重要的一张表):**

| 层 | 昇腾(Ascend) | 英伟达(NVIDIA) | 一句话说明 |
|----|----------------|-------------------|-----------|
| 加速硬件 | NPU(昇腾 910 等) | GPU(A100/H100 等) | 训练算力载体 |
| 计算核心 | 达芬奇 Cube + Vector 单元 | Tensor Core + CUDA Core | 矩阵乘 / 向量运算的硬件单元 |
| 底层软件栈 | CANN | CUDA | 编程框架 + 运行时 + 编译器 |
| 算子库 | CANN 算子 / AOL | cuDNN / cuBLAS | 卷积、GEMM 等高性能算子 |
| 集合通信 | HCCL | NCCL | AllReduce/AllGather 等多卡通信 |
| 深度学习框架 | PyTorch + **torch_npu** / MindSpore | PyTorch / TensorFlow | 上层训练框架 |
| 加速底座 | **AscendSpeed** | apex / Megatron-Core | 融合算子 + 并行底层支持 |
| **大模型训练套件** | **ModelLink** | **Megatron-LM** | 3D 并行 + 模型配方 |
| 推理引擎 | MindIE | TensorRT-LLM / vLLM | 部署阶段(不在本文范围) |
| 量化工具 | msModelSlim | GPTQ / AWQ 工具链 | 压缩(不在本文范围) |

**记住这条对应主线:`ModelLink ↔ Megatron-LM`,`AscendSpeed ↔ Megatron-Core/apex`,`CANN ↔ CUDA`,`HCCL ↔ NCCL`。** 你脑子里只要装下这一行,从 GPU 迁到 NPU 的 90% 概念就对上号了。

## 2. 软件栈分层:一张图看清 ModelLink 站在哪

```
   ┌───────────────────────────────────────────────────────┐
   │  ModelLink  (训练套件 / 配方层  ≈ Megatron-LM)        │
   │   - pretrain / 微调脚本、TP·PP·DP 参数、模型 args      │
   │   - tools/checkpoint/convert_ckpt.py(权重转换)       │
   ├───────────────────────────────────────────────────────┤
   │  AscendSpeed  (加速底座  ≈ Megatron-Core + apex)      │
   │   - 融合算子(FlashAttention / RMSNorm / RoPE 等)     │
   ├───────────────────────────────────────────────────────┤
   │  PyTorch + torch_npu  (框架层  ≈ PyTorch+CUDA)        │
   │   - 把 .cuda() 语义映射到 .npu()                       │
   ├───────────────────────────────────────────────────────┤
   │  CANN  (异构计算架构  ≈ CUDA Toolkit)                 │
   │   - 算子库 / 图编译 / 运行时 / HCCL 集合通信          │
   ├───────────────────────────────────────────────────────┤
   │  达芬奇架构 NPU 硬件 (Cube + Vector + 片上缓存)       │
   └───────────────────────────────────────────────────────┘
```

读法:**从下往上越来越「业务」,从上往下越来越「底层」。** 你写训练命令是在最顶层 ModelLink;它调用 AscendSpeed 的融合算子;算子最终编译成 CANN 的指令,跑在达芬奇 NPU 上;多卡之间的梯度同步走 HCCL。本仓库 `clone ModelLink` + `clone AscendSpeed && pip install -e .` 的两步,正是在搭顶上两层。

## 3. 端到端训练流水线:从 HuggingFace 到训完的五步

ModelLink 的工作流和 Megatron 几乎一模一样,核心是「**HF 格式和 Megatron 格式不通用,必须来回转换**」:

```
 ┌────────────┐  ①转入   ┌──────────────┐  ②预处理  ┌────────────┐
 │ HF 权重     │ ───────► │ ModelLink    │           │ 数据集      │
 │(.bin/.safe)│ convert  │ tp/pp 切分权重│           │ bin+idx    │
 └────────────┘          └──────┬───────┘           └─────┬──────┘
                                │  ③训练(pretrain / 微调)│
                                ▼◄───────────────────────┘
                         ┌──────────────┐  ④转出   ┌────────────┐
                         │ 训练后 ckpt   │ ───────► │ HF 权重     │
                         │(tp/pp 分片)  │ convert  │(便于推理)  │
                         └──────────────┘          └────────────┘
                                ⑤ 推理 / 评测(交给 MindIE 等)
```

五步的含义:

1. **环境搭建**:`clone ModelLink`、`clone AscendSpeed && pip install -e .`、装 `requirements.txt`,并 `source .../ascend-toolkit/set_env.sh` 把 CANN 环境变量注入当前 shell。这一步就是在搭第 2 节图里上面三层。**坑**:`set_env.sh` 的路径因安装方式而异,**具体路径以官方文档为准**;忘了 source 会报「找不到 CANN / torch_npu 算子」之类的错。
2. **权重转入**:把 HF 的连续大权重,按目标 TP/PP 切成分片(详见第 4 节)。
3. **数据预处理**:把原始语料(jsonl 等)转成 Megatron 的 `bin + idx` 二进制索引格式,训练时按 token 偏移直接 mmap 读取,避免训练时反复 tokenize。
4. **启动训练**:用 `pretrain_*.sh` 类脚本,指定 TP/PP/DP、global/micro batch、学习率、保存间隔等。多机时由 HCCL 负责跨卡梯度同步。
5. **权重转出**:训练得到的是 tp/pp 分片 ckpt,要转回 HF 单文件格式,才方便交给 MindIE / vLLM 类引擎做推理或上传分享。

> 具体脚本名、命令行、batch/lr 数值等 **以官方文档为准**;本文只保证「步骤顺序与依赖关系」是对的。

## 4. 权重转换机制:看懂本仓库的 convert_ckpt

本仓库给的就是「步骤②权重转入」,逐参数拆解(参数值仅为示例,**真实取值以官方文档为准**):

```
python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \                 # 模型大类(decoder-only 走 GPT)
    --loader llama2_hf \               # 用哪个「读取器」解析 HF 权重(此处按 llama2 结构读)
    --saver megatron \                 # 存成 Megatron(ModelLink 内部)格式
    --target-tensor-parallel-size 2 \  # 目标 TP:把每个权重矩阵横/纵切成 2 份
    --load-dir  .../Baichuan2-7B-Chat/ # 输入:HF 权重目录
    --save-dir  .../Baichuan2-7B-...-tp8-pp1/ # 输出:切分后的 Megatron ckpt
    --tokenizer-model .../tokenizer.model \   # 分词器
    --params-dtype bf16 \              # 以 bf16 保存权重
    --w-pack True                      # Baichuan 把 QKV 打包成一个权重,需开启拆包逻辑
```

**机制要点:**

- **loader / saver 是一对「适配器」**:loader 负责「按某个模型族的命名规则,把 HF 的 state_dict 读进来」;saver 负责「按 ModelLink 的张量切分约定写出去」。换模型主要是换 loader(如 `llama2_hf` / `qwen_hf`)。
- **TP 切分发生在转换期,不是训练期**:`--target-tensor-parallel-size` 决定了每个 nn.Linear 的权重在保存时就被沿行或列切成 N 份,每份放进对应 rank 的子目录。训练时各 rank 直接加载自己那份,**不需要再切**。所以「转换时设的 TP/PP」必须和「训练脚本里设的 TP/PP」一致,否则加载会维度对不上——这是头号高频坑。
- **`--w-pack True` 是 Baichuan 专属坑**:Baichuan 把 Q、K、V 三个投影合并成一个大权重矩阵存储,转换器要按这个布局把它正确拆成 Megatron 期望的形状;模型族不同,这类「结构特例开关」也不同。
- **`params-dtype` 决定显存与精度**:bf16 是大模型训练主流,动态范围比 fp16 大、更不易溢出,达芬奇架构对 bf16 有良好支持。

## 5. 3D 并行在 NPU 上怎么落地

ModelLink 继承 Megatron 的三维并行,只是底层通信从 NCCL 换成了 **HCCL**:

```
  并行维度        切什么            谁来通信        昇腾后端
  ───────────    ─────────────     ───────────    ──────────
  TP 张量并行     单层权重矩阵       层内频繁通信    HCCL AllReduce(机内 NVLink 级)
  PP 流水线并行   按层切成多段       段间点对点      HCCL Send/Recv
  DP 数据并行     同一份模型多副本   梯度 AllReduce  HCCL AllReduce
```

- **TP(Tensor Parallel)**:把每个注意力 / FFN 的大矩阵乘切到多张卡,**层内**就要做 AllReduce,通信极频繁,所以 TP 组通常放在「机内、互联带宽最高」的卡之间(对标 GPU 的 NVLink,昇腾用 HCCS / 机内高速互联)。
- **PP(Pipeline Parallel)**:把模型按层分成若干 stage,前一段算完把激活 Send 给下一段。靠 micro-batch 流水来填满气泡。
- **DP(Data Parallel)**:多副本各吃一份数据,反向后对梯度做 AllReduce 求平均。
- 三者组合关系:**总卡数 = TP × PP × DP**。这就是为什么转换时要先想清楚 TP/PP——它直接决定权重要切成几份。

HCCL 在这里的角色完全等同于 NCCL:提供 AllReduce / AllGather / ReduceScatter / Send / Recv 等原语,并根据拓扑自动选环(Ring)/ 树等算法。详见 [[ai-infra/网络/集合通信原语]]。

## 6. AscendSpeed 与融合算子:性能从哪来

光把 Megatron 逻辑搬过来还不够快,**AscendSpeed** 这一层提供了一批针对达芬奇架构手工优化的**融合算子**,这是 NPU 上训练能打的关键:

- **FlashAttention 融合**:把「QK^T → softmax → ×V」融成一个核,避免把巨大的注意力矩阵写回显存(HBM),在长序列下显存和速度收益巨大。
- **RMSNorm / LayerNorm 融合**、**RoPE 融合**、**融合优化器(如 fused Adam)**:把多个小的 elementwise 操作合并,减少 Vector 单元反复读写片上/片外内存的开销。
- 这些算子的本质是**减少访存、提升 Cube/Vector 单元利用率**——和 GPU 上 apex / Megatron-Core 的融合算子是同一思路,只是实现落到 CANN 算子上。

所以本仓库要 `pip install -e .` 安装 AscendSpeed:没有它,ModelLink 的脚本会因为找不到这些融合算子而退化甚至报错。

## 迁移要点 / 注意事项与坑

**从 GPU+Megatron 迁到 NPU+ModelLink 的迁移清单:**

| 你在 GPU 上做的 | 在 NPU+ModelLink 上对应改成 | 注意点 |
|----------------|---------------------------|--------|
| `tensor.cuda()` / `device='cuda'` | `tensor.npu()` / `device='npu'`(torch_npu) | 大部分由套件内部处理,自己写补丁时才需手改 |
| `import` apex / Megatron-Core | 安装并依赖 AscendSpeed | 不装 → 找不到融合算子 |
| NCCL 后端 | HCCL 后端 | 分布式 init 的 backend 改 hccl |
| `nvidia-smi` 看卡 | `npu-smi` 看卡 | 监控命令不同(**具体以官方为准**) |
| CUDA 版本对齐 | CANN 版本对齐 | torch_npu 与 CANN 版本**必须配套**,错配是头号环境坑 |

**高频坑清单:**

1. **没 source `set_env.sh`** → 报找不到 CANN/算子;每开一个新终端都要重新 source。
2. **转换 TP/PP ≠ 训练 TP/PP** → 加载时张量维度对不上,直接挂。两边数字必须一致。
3. **torch_npu 与 CANN 版本不配套** → 各种诡异算子错误。**版本对应表以官方文档为准**,不要凭感觉装。
4. **模型结构开关漏开**(如 Baichuan 的 `--w-pack`、是否合并 gate/up、RoPE base 等) → 转换出来的权重「能加载但训不收敛 / loss 异常」,极难排查。换模型族时务必核对官方该模型的转换配方。
5. **bf16 vs fp16 混用** → 精度/溢出问题;主流用 bf16。
6. **机内 / 机间拓扑没规划好**:把通信最频繁的 TP 组跨机摆放,会被机间带宽拖死。原则:**TP 优先机内,PP/DP 可跨机**。

**性能调优思路(机制层面):**

- 先保证融合算子(FlashAttention 等)真的生效——这是最大的免费午餐。
- TP 太大会让层内 AllReduce 通信占比上升;PP 太大气泡变多。需要按集群规模和模型大小折中,经验是「TP 不超过单机卡数」。
- 增大 micro-batch 数量摊薄 PP 气泡;用重计算(activation recomputation)换显存。

## 常见问题

| 问题 | 答案 |
|------|------|
| ModelLink 对标英伟达的什么? | Megatron-LM(大模型 3D 并行训练套件) |
| 它和 AscendSpeed 啥关系? | ModelLink 在上(配方/脚本),AscendSpeed 在下(融合算子/底座),后者类比 Megatron-Core+apex |
| 为什么要做权重转换? | HF 格式和 Megatron 的 tp/pp 切分格式不通用,训练前转入、训练后转出 |
| 转换时的 TP/PP 能和训练时不一样吗? | 不能,必须一致,否则加载维度对不上 |
| `--w-pack True` 是干嘛的? | 处理 Baichuan 把 QKV 打包存储的特例,换模型族开关不同 |
| 多卡通信走什么? | HCCL(对标 NCCL) |
| 用 PyTorch 还是 MindSpore? | ModelLink 走 PyTorch + torch_npu 路线 |
| 训完怎么推理? | 把 ckpt 转回 HF,再交给 MindIE / vLLM 等推理引擎 |
| 具体命令/版本去哪查? | 一律以华为昇腾官方文档(Ascend 社区 / Gitee ModelLink README)为准 |

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
