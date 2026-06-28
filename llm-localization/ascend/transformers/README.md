# 昇腾 Transformers(HuggingFace Transformers on Ascend)

> 让 HuggingFace `transformers` 这套"模型即代码"的事实标准生态，跑在昇腾 NPU 上的适配层。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-framework/huggingface-transformers/README]] [[ai-infra/ai-hardware/CUDA]]

仓库参考：https://gitee.com/ascend/transformers/ (昇腾官方维护的 Transformers 适配/示例仓库)

---

## 阅读地图

| 节 | 你将搞清楚 | 关键词 |
|----|-----------|--------|
| 0  | 一句话锚点 | transformers + torch_npu |
| 1  | 在昇腾软件栈的定位 + 昇腾↔英伟达生态对照 | CANN / torch_npu |
| 2  | 整体分层:从一行 `.to("npu")` 到硬件 | PrivateUse1 / 算子下沉 |
| 3  | 一段 GLUE 微调代码在昇腾上的执行流 | Trainer / 图模式 |
| 4  | torch_npu 适配机制:设备、算子、混精 | aten 算子映射 |
| 5  | 加速库:从原生算子到融合大算子 | FlashAttention / 融合算子 |
| 迁移 | 从 CUDA 代码迁到昇腾要改什么 | device 字符串 / 坑 |
| FAQ | 常见疑问速查 | — |
| 链接 | 枢纽跳转 | — |

---

## 0. 一句话锚点

**昇腾 Transformers = HuggingFace `transformers`(模型库与训练/推理高层 API)+ `torch_npu`(把 PyTorch 接到昇腾 NPU 的插件)+ CANN(昇腾的算子与运行时底座)。**

它的目标只有一个:让你**几乎不改 HuggingFace 的建模代码**(`AutoModel`、`Trainer`、`generate`),把 `cuda` 换成 `npu`,就能在昇腾卡上完成训练与推理。所以它不是一个"新框架",而是一层**适配 + 示例 + 最佳实践**。

---

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈的四层心智模型

```
          ┌──────────────────────────────────────────────┐
 应用/套件 │  MindFormers / MindIE / 本仓 transformers示例   │  ← 你写的训练/推理脚本在这
          ├──────────────────────────────────────────────┤
   框架层  │  PyTorch + torch_npu 插件  (或 MindSpore)        │  ← HuggingFace transformers 跑在这层
          ├──────────────────────────────────────────────┤
   CANN层  │  ACL运行时 / GE图引擎 / AOL算子库 / HCCL通信     │  ← 对标 CUDA+cuDNN+cuBLAS+NCCL
          ├──────────────────────────────────────────────┤
   硬件层  │  昇腾 NPU(达芬奇架构:Cube + Vector + Scalar)   │  ← 对标 GPU(SM/Tensor Core)
          └──────────────────────────────────────────────┘
```

HuggingFace `transformers` 本身是**框架无关的高层库**:它把模型结构、分词器、Trainer、`generate` 封装好,底下既可以接 PyTorch+CUDA,也可以接 PyTorch+`torch_npu`。**本仓做的事**,就是确保这条"transformers → PyTorch → torch_npu → CANN → NPU"的链路顺畅,并给出经过验证的示例(如 `run_glue.py` 文本分类微调)。

### 1.2 昇腾 ↔ 英伟达生态对照表(迁移心智图)

| 维度 | 英伟达世界 | 昇腾世界 | 说明 |
|------|-----------|---------|------|
| 加速芯片 | GPU | NPU(昇腾) | 计算硬件 |
| 微架构计算单元 | Tensor Core / CUDA Core | 达芬奇 Cube / Vector / Scalar 单元 | Cube 专攻矩阵乘 |
| 软件底座 | CUDA | CANN | 编程框架 + 运行时 |
| 算子库 | cuDNN / cuBLAS | AOL(昇腾算子库)/ TBE 自定义算子 | 高性能算子 |
| 运行时/驱动 | CUDA Runtime / Driver | ACL(AscendCL)/ Driver | 设备管理、流、内存 |
| 集合通信 | NCCL | HCCL | 多卡 AllReduce 等 |
| 深度学习框架 | PyTorch(原生 CUDA 后端) | PyTorch + **torch_npu** 插件 | torch_npu 是关键桥梁 |
| 国产原生框架 | —— | MindSpore | 华为自研框架 |
| 设备字符串 | `"cuda"` / `"cuda:0"` | `"npu"` / `"npu:0"` | 代码里最直观的差别 |
| 模型生态库 | HuggingFace transformers | **同一套 transformers**(本仓适配) | 复用上游,不另造轮子 |
| 大规模训练套件 | Megatron-LM | MindFormers / ModelLink | 见各自目录 |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 高吞吐推理 |
| 量化压缩工具 | GPTQ / AWQ 工具链 | msModelSlim | 权重量化 |
| 混合精度数据类型 | FP16 / BF16 | FP16 / BF16 | 概念一致 |

> 一句话记忆:**NPU↔GPU、CANN↔CUDA、torch_npu↔PyTorch原生CUDA后端、HCCL↔NCCL、AOL↔cuDNN/cuBLAS**。而 `transformers` 这一层**两边共用**——这正是昇腾"拥抱主流生态"的策略。

---

## 2. 整体分层:从一行 `.to("npu")` 到硬件

`transformers` 的建模代码本质上是一堆 PyTorch `nn.Module` 与张量运算。理解"它怎么落到 NPU 上",关键是理解 **PyTorch 的 PrivateUse1 设备扩展机制**:torch_npu 把 NPU 注册成一个 PyTorch 后端设备。

```
 你的脚本                  PyTorch 框架            torch_npu 插件          CANN / NPU
 ─────────                ──────────────          ──────────────          ──────────
 model = AutoModel...
 model.to("npu") ───────► 解析 device 字符串 ────► 注册的 NPU 后端 ──────► 设备初始化(ACL)
                                                                            │
 out = model(x.to("npu"))                                                   │
   └ 触发 matmul/softmax ─► dispatch 到 aten 算子 ─► 映射到昇腾算子 ────────► Cube/Vector 执行
                                                       (AOL 算子库)
 loss.backward() ───────► autograd 反向图 ─────────► 反向算子映射 ──────────► NPU 计算梯度
 hccl AllReduce(多卡) ──────────────────────────────► HCCL 通信 ──────────► 跨卡梯度同步
```

要点:
- **`transformers` 自己几乎不感知硬件**——它只调用 PyTorch 的张量算子。
- **torch_npu 做"翻译"**:把 PyTorch 的 `aten::matmul`、`aten::softmax` 等算子,dispatch 到昇腾对应的高性能算子。
- **CANN 做"执行"**:把算子下发到达芬奇架构的 Cube/Vector 单元真正算出来。

---

## 3. 一段 GLUE 微调代码在昇腾上的执行流

以本仓示例(BERT 在 MRPC 上做文本分类微调,`run_glue.py`)为例,理解端到端流程的**每一步含义**(具体命令与版本以华为昇腾官方文档 / Ascend 社区为准):

```
[1] 准备环境          安装 CANN(驱动+固件+toolkit)→ 安装匹配版本的 torch + torch_npu
                      └ 坑:torch 与 torch_npu 版本必须严格配套,否则算子找不到
[2] 安装 transformers  pip 安装上游 transformers + examples 的 requirements
[3] import torch_npu   关键:脚本里要 import torch_npu,设备才会被注册
[4] 数据加载           HuggingFace datasets 加载 GLUE/MRPC,与硬件无关
[5] 模型 .to("npu")    AutoModelForSequenceClassification → 搬到 NPU
[6] Trainer 训练循环   前向/反向/优化器 step 全部落到 NPU 算子
[7] 多卡(可选)        分布式后端用 hccl(对标 nccl),梯度用 HCCL AllReduce 同步
[8] 评估/保存          do_eval 计算指标,保存权重(与硬件无关)
```

示例命令(来自本仓,展示**接口与上游一致**,真正运行参数以官方为准):

```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

**注意**:这条命令和在 GPU 上跑 HuggingFace 官方示例**几乎一字不差**——这正是适配层的价值。差别只在于:运行前要装好 CANN + torch_npu,脚本里 `import torch_npu`,设备从 `cuda` 变成 `npu`。

---

## 4. torch_npu 适配机制:设备、算子、混精

torch_npu 是整条链路的"心脏",理解它的三件事就理解了昇腾上的 PyTorch:

### 4.1 设备注册(PrivateUse1)
PyTorch 预留了 `PrivateUse1` 这一"第三方设备"扩展点。torch_npu 把 NPU 注册到这里,于是 `torch.device("npu")`、`tensor.to("npu")`、`torch.npu.synchronize()` 等接口就能用了。**这就是为什么必须 `import torch_npu`**——不导入,设备根本没注册。

### 4.2 算子映射(aten → 昇腾算子)
PyTorch 的每个张量运算最终是一个 `aten` 算子。torch_npu 通过 dispatch 机制,把 `aten::matmul`、`aten::layer_norm`、`aten::softmax` 等映射到昇腾算子库(AOL)里的高性能实现。
- 大多数常用算子**已覆盖**;
- 个别冷门算子**可能未覆盖**,会报"算子不支持"——这是迁移时最常见的坑(见迁移章)。

### 4.3 混合精度与数据格式
- 支持 FP16 / BF16 自动混合精度(对标 CUDA AMP);
- 昇腾内部对张量有自己偏好的**数据排布格式(如 NZ 等私有格式)**,框架会在算子间自动转换,但**频繁的格式转换会拖慢性能**——这是性能调优的一个重要观察点。

---

## 5. 加速库:从原生算子到融合大算子

仅靠"逐个 aten 算子映射"能跑通,但要跑得**快**,需要**融合算子(fused kernel)**——把多个小算子合并成一个大算子,减少访存与 kernel 启动开销。这对标英伟达的 FlashAttention、fused LayerNorm 等。

```
 朴素执行(慢)                        融合执行(快)
 ────────────                        ────────────
 QK^T  ──► 写回显存                   ┌──────────────────────────┐
 softmax ─► 写回显存                  │  FlashAttention 融合算子   │
 PV    ──► 写回显存                   │  (QK^T+softmax+PV 一次完成)│
   每步都访存,带宽瓶颈                └──────────────────────────┘
                                       中间结果留在片上,省访存
```

在昇腾生态里,这类能力可能通过以下途径提供(具体 API 与可用性以官方文档为准):
- torch_npu 直接提供的**融合算子接口**(如融合的注意力、RMSNorm 等);
- 上层套件(MindFormers / MindIE)内置的优化;
- 图模式(见下)带来的**整图算子融合**。

**图模式 vs 单算子模式(eager)** 是一个核心概念:
- **单算子(eager)模式**:像 PyTorch 默认那样,一个算子一个算子地下发——灵活、好调试,但融合机会少。
- **图模式(graph)**:把整张计算图一次性交给 CANN 的 GE 图引擎,引擎可以做**整图优化、算子融合、内存复用**——吞吐更高,但灵活性差、首次编译有开销。这对标 PyTorch 的 `torch.compile` / CUDA Graph 的思路。

---

## 迁移要点 / 注意事项与坑

从一段在 GPU 上跑得好好的 HuggingFace 代码,迁到昇腾,**要改的通常很少,但坑很集中**:

| 类别 | 要做什么 / 容易踩什么坑 |
|------|----------------------|
| 导入 | 脚本顶部必须 `import torch_npu`,否则 `npu` 设备不存在 |
| 设备字符串 | 所有 `"cuda"` / `"cuda:0"` 改成 `"npu"` / `"npu:0"`;`torch.cuda.xxx` 改成 `torch.npu.xxx`(或用与设备无关的写法) |
| 版本配套 | **torch 与 torch_npu 版本必须严格匹配**,且与 CANN 版本匹配——版本错配是最高频故障源 |
| 算子缺失 | 个别算子未适配会报"not supported"——可换等价写法、升级版本、或反馈官方 |
| 数据格式 | 留意私有格式(如 NZ)的隐式转换开销,避免在算子链中反复来回转 |
| 分布式后端 | `init_process_group` 的 backend 从 `nccl` 改为 `hccl` |
| 混合精度 | AMP 思路一致,但 autocast 的设备类型要对应 NPU |
| 随机性/对齐 | 跨硬件数值不会逐位对齐,验收看**指标收敛**而非逐位一致 |
| 性能调优 | 优先开启融合算子;评估单算子模式 vs 图模式;关注 host 下发瓶颈(host-bound)与算子格式转换 |
| 安装类 | CANN 驱动/固件/toolkit、torch_npu 的**具体命令与版本一律以华为昇腾官方文档(Ascend 社区)为准**,本文不给死命令以免误导 |

**调优思路(机制层面)**:
1. 先用单算子模式跑通、验证精度;
2. 用 profiling 找瓶颈——是算子慢(compute-bound)、还是 host 下发慢(host-bound)、还是格式转换/访存慢;
3. 针对性地:启用融合算子、切到图模式、调整 batch/序列长度、减少不必要的 device-host 拷贝;
4. 多卡场景关注 HCCL 通信是否与计算重叠。

---

## 常见问题

| 问题 | 回答 |
|------|------|
| 这是华为自己 fork 的 transformers 吗? | 主体是**复用上游 HuggingFace transformers**,本仓提供适配、示例与最佳实践;真正的硬件桥梁是 `torch_npu` |
| 我必须改建模代码吗? | 大多数情况下**不用改模型结构**,只改设备字符串、加 `import torch_npu`、对齐版本 |
| 和 MindFormers 什么关系? | MindFormers 是昇腾的**大规模训练套件**(对标 Megatron);本仓的 transformers 路线更贴近 HuggingFace 习惯,适合快速复用上游模型 |
| 和 MindSpore 什么关系? | 两条路线:一是 **PyTorch + torch_npu**(本仓),二是 **MindSpore** 原生框架;前者迁移成本低,后者国产化更彻底 |
| 报"算子不支持"怎么办? | 多半是算子未适配:换等价实现、升级 torch_npu/CANN、或向官方反馈 |
| 推理高吞吐用什么? | 训练/小规模推理可用本路线;追求高吞吐生产推理用 **MindIE**(对标 TensorRT-LLM / vLLM) |
| 版本怎么选? | torch ↔ torch_npu ↔ CANN 三者必须配套,**以官方配套表为准**,切勿随意混搭 |

---

## 🔗 跳转链接

枢纽:
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

相关昇腾组件:
- MindFormers(训练套件,对标 Megatron)
- MindIE(推理引擎,对标 TensorRT-LLM / vLLM)
- msModelSlim(量化工具,对标 GPTQ / AWQ)
- ModelLink(训练,对标 Megatron)
- HCCL(集合通信,对标 NCCL)
