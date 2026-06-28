# FasterTransformer 部署 BLOOM（含推理性能统计）

> 用 NVIDIA FasterTransformer（FT）这套"手写 CUDA 算子 + C++ 推理引擎"把 BLOOM 系（这里是 firefly-2b6 / belle7b 微调模型）跑起来，并**逐 token 统计推理耗时**：单卡跑、双卡张量并行跑，对比平均每 token 生成时长。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/faster-transformer/README]] [[llm-algo/transformer/模型架构]] [[llm-optimizer/kv-cache]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 节 | 讲什么 | 你能带走什么 |
|----|--------|--------------|
| 0 | 一句话锚点 | 这篇 README 到底在做"哪台戏" |
| 1 | 地基：BLOOM 是什么 / FT 为什么能跑它 | BLOOM 结构特点 + FT 适配点 |
| 2 | 全链路架构 | 从 HF 权重→FT `.bin`→逐 token 推理统计 |
| 3 | 评测数据格式 | 原文那段 JSON 的字段含义与设计意图 |
| 4 | 单卡推理统计 | 原文单卡命令逐参数拆解 |
| 5 | 双卡张量并行（TP=2） | 为什么要 `mpirun -n 2`、TP 怎么切 |
| 6 | 性能指标怎么算 | 平均每 token 时长、prefill vs decode |
| — | 实操命令汇总 | 原文真料原样保留 |
| — | 常见坑 | 路径/精度/并行/统计口径高频问题 |

## 0. 一句话锚点

**这篇 README 干的是"用 FT 把一个 BLOOM 系中文模型跑起来，并量出它每生成一个 token 要多久"。** 模型是 `firefly-2b6`（约 2.6B 参数、BLOOM 架构的中文指令微调模型，权重来自 belle7b checkpoint），任务是**电销（dianxiao）对话续写**——给定客户画像 + 历史对话，让模型生成 Assistant 的下一句。

它不教你怎么编译 FT（那是上级 [[llm-inference/faster-transformer/README]] 的事），而聚焦两件事：
1. **喂什么数据**：一段电销对话的 JSON（第 3 节）。
2. **怎么跑 + 统计**：单卡 / 双卡 TP 两条命令，输出每条样本的逐 token 耗时（第 4、5 节）。

> ⚠️ 时代背景：FasterTransformer 已进入**维护/归档**状态，NVIDIA 主推 **TensorRT-LLM**。本文讲 FT 是为了理解"手写算子级推理引擎 + 性能统计"的底层机制，生产新项目应优先评估 TensorRT-LLM / vLLM / SGLang。具体版本与维护状态以官方仓库为准。

## 1. 地基：BLOOM 是什么，FT 凭什么能跑它

### 1.1 BLOOM 的结构特点

BLOOM 是 BigScience 训练的多语言自回归大模型，结构上是 **GPT 风格的 decoder-only Transformer**，但有两处和原版 GPT 不同，正是 FT 适配它要注意的点：

| 组件 | 标准 GPT | BLOOM | 含义 |
|------|----------|-------|------|
| 位置编码 | 学习式绝对位置 | **ALiBi**（Attention with Linear Biases） | 不加位置 embedding，而在 attention score 上按距离加线性偏置，外推长度友好 |
| 输入归一化 | 词 embedding 后无额外 norm | **embedding 后接一个 LayerNorm** | BLOOM 在 word embedding 之后多了一层 LayerNorm |
| 归一化类型 | LayerNorm | **LayerNorm**（带均值/方差/bias，非 RMSNorm） | 与 LLaMA 的 RMSNorm 不同 |
| FFN 激活 | GELU | **GELU** | 与 LLaMA 的 SwiGLU 不同 |

ALiBi 是和 LLaMA 的 RoPE 并列的另一类位置方案，对比见 [[llm-algo/旋转编码RoPE]]。

```
ALiBi 的直觉：注意力打分时，离得越远扣分越多
  score(q_i, k_j) = q_i·k_j  +  m · (j - i)     (j ≤ i, m 为每个头固定的斜率)
                    └─内容相关─┘  └─只看距离的线性惩罚─┘
  → 不需要任何"位置向量"，天然能外推到比训练更长的序列
```

### 1.2 FT 为什么比 HF transformers 快

直接用 HuggingFace `transformers` 推理，瓶颈在三处：

```
HF 推理一层（简化）：
  x ─LN─► ─Linear(Q)─► ─Linear(K)─► ─Linear(V)─► ─Attn─► ─Linear(O)─► ─FFN─►
       每个箭头 = 一次独立 kernel launch + 一次 HBM 读写
```

1. **kernel launch 太多**：每个小算子单独启动，launch 开销 + 中间结果反复进出显存，对**访存密集（memory-bound）**的 decode 阶段是致命的。
2. **缺融合**：LayerNorm+bias+残差、加 bias+激活等本可合并的没合并。
3. **缺推理专用并行**：HF 主要服务训练，对**推理时张量并行**不原生、不极致。

FT 的应对：**算子融合**（把一层能合的压成少数大 CUDA kernel，如融合 MHA、融合 add-bias-LayerNorm）+ **手写/调优 GEMM**（cuBLAS/CUTLASS，针对 FP16/BF16/INT8/FP8 特化）+ **内建 TP+PP**（NCCL+MPI 跨卡通信）。

一句话：**FT 把"一堆 Python 小算子"压成"少数大 CUDA kernel + 多卡并行"，换吞吐和延迟。** 这就是本文要对 firefly-2b6 量出来的东西。

## 2. 全链路架构

```
┌──────────────────────────────────────────────────────────────┐
│ 阶段一：离线（一次性，在上级 README 里做）                     │
│   HF/Megatron BLOOM 权重 (firefly-2b6, FP16)                  │
│        │  转换脚本（按 TP 切分）                               │
│        ▼                                                       │
│   FT 权重目录：                                               │
│     firefly-2b6-dx-1tp/belle7b/1/1-gpu   ← 单卡，不切          │
│     firefly-2b6-dx-2tp/belle7b/1/2-gpu   ← TP=2，切成 2 份     │
│        ├─ config.ini   (层数/头数/hidden/词表/ALiBi…)         │
│        └─ ...weight.0.bin / .1.bin  (按 rank 切分的二进制)     │
└──────────────────────────────────────────────────────────────┘
                         │ 加载 libth_transformer.so
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 阶段二：在线推理 + 逐 token 统计（本文）                       │
│                                                                │
│  读 dataset (lambada_test.jsonl) + 电销样本 JSON               │
│        ▼                                                       │
│  ┌─ Prefill ─┐  一次吃完 input（截到 input-token-len=64）      │
│  │ 算 K,V    │  → 写 KV Cache                                  │
│  └───────────┘                                                │
│        ▼                                                       │
│  ┌─ Decode ──┐  逐 token 生成（最多 output-token-len=256）     │
│  │ 每步 1 token + 记下耗时 │  ← 读历史 KV Cache                │
│  └───────────┘                                                │
│        ▼                                                       │
│  写出统计 JSON：firefly_random_sample_1w_256_stat_ft*.json     │
│  （每条样本的总耗时、平均每 token 时长等）                     │
└──────────────────────────────────────────────────────────────┘
```

要点：
- `1tp`/`2tp` 两套权重目录是**离线就按并行度切好的**——单卡加载 `1-gpu`，双卡各 rank 加载 `2-gpu` 里属于自己的 `.bin`。
- 统计脚本 `firefly_lambada_dianxiao_1w_stat_token.py` 的活：**跑推理 + 计时 + 落盘**，把"平均每 token 生成时长"这类指标量出来。

## 3. 评测数据格式（原文那段 JSON 的含义）

原文给出的数据是一个 **list，每个元素一条对话样本**，三个字段：

| 字段 | 含义 | 作用 |
|------|------|------|
| `id` | 样本序号 | 对齐统计结果 / 复现 |
| `input` | 模型输入 = 客户画像 + 历史对话，以 `Assistant: ` 结尾 | 模型从这里"接着说" |
| `answer` | 期望的 Assistant 回复（参考答案，以 `</s>` 收尾） | 评测质量时和模型输出对比 |

`input` 的结构（拆原子看）：

```
#姓名：何#性别：女士#剩余额度：一万七千#当前额度：…#活动：D#…#cust_type：授信T91-180
└──────────────── 客户画像：以 # 分隔的 key:value（数字也写成中文）─────────────┘

\n\nHuman: 你好。
Assistant: 好，请问一下是何女士对吧？</s>     ← 一来一回，</s> 是回合结束符
Human: …
Assistant:                                    ← input 以这个空的 Assistant: 收尾
                                               让模型从此处续写下一句
```

设计意图：
- **画像前缀**给模型业务上下文（额度、活动类型、客户分层 `cust_type`），让回复更贴电销场景。
- **`Human:`/`Assistant:` 多轮**是 BLOOM/firefly 这类指令模型常见的对话模板；`</s>` 是 BLOOM 的句末/回合结束 token。
- **input 以 `Assistant: ` 结尾且无内容**——这是续写式生成的标准做法：把"轮到模型说话"的位置留空，模型从这里往后 decode。

> 数字写成中文（"一万七千"而非 17000）是该数据集的脱敏/口语化习惯，对 token 化没有特殊要求，照原样喂即可。

## 4. 单卡推理统计：命令逐参数拆解

原文单卡命令（保留原样，见下方"实操命令汇总"）。逐参数解释：

```
CUDA_VISIBLE_DEVICES=1            # 只用 1 号 GPU（屏蔽其余卡）
python …/firefly_lambada_dianxiao_1w_stat_token.py   # 跑推理+统计的脚本
--checkpoint-path …/firefly-2b6-dx-1tp/belle7b/1/1-gpu  # FT 权重（单卡=1-gpu，不切）
--tokenizer-path  …/firefly-2b6-dx                   # 分词器目录（HF tokenizer）
--dataset-path    …/lambada_test.jsonl               # 数据集（lambada 格式驱动 + 电销样本）
--lib-path        …/libth_transformer.so             # FT 编译产物：PyTorch op 动态库
--inference-data-type fp16                           # 推理精度 FP16
--show-progress                                      # 打印进度条
--input-token-len  64                                # input 截/补到 64 token（统一 prefill 规模）
--output-token-len 256                               # 每条最多 decode 256 个新 token
--dianxiao-path-stat …/firefly_random_sample_1w_256_stat_ft.json  # 统计结果落盘路径
```

为什么固定 `input-token-len=64` / `output-token-len=256`？
- **统计要可比**：性能（每 token 时长）受输入长度（prefill 规模）和输出长度（decode 步数）影响很大。把两者**钉死成常数**，不同卡数/精度之间才公平对比。
- `1w` = 1 万条样本（`firefly_random_sample_1w`），随机抽样保证代表性。

`--lib-path` 指向的 `libth_transformer.so` 就是上级 README 里 `make -j` 编出来的产物——**没有它，Python 调不到 FT 的 C++/CUDA 引擎**。

## 5. 双卡张量并行（TP=2）

原文第二条命令把同一份任务切到 2 卡跑。关键差异：

```
CUDA_VISIBLE_DEVICES=2,3          # 暴露 2 张卡（2 号、3 号）
mpirun -n 2 python …             # 起 2 个进程，一进程绑一卡（一 rank 一 GPU）
--checkpoint-path …/firefly-2b6-dx-2tp/belle7b/1/2-gpu  # 用 TP=2 切好的权重（2-gpu）
--tensor-para-size 2             # 张量并行度 = 2
--pipeline-para-size 1           # 不做流水并行
--dianxiao-path-stat …_stat_ft_tp2.json   # 输出文件名带 tp2，便于和单卡对比
```

### 5.1 为什么必须 `mpirun -n 2` + 专门的 `2-gpu` 权重

```
单卡：1 进程，加载完整权重 (1-gpu 目录)
TP=2：2 进程，各加载半份权重 (2-gpu 目录里的 .0.bin / .1.bin)
       ┌── rank0 (GPU2) ──┐   ┌── rank1 (GPU3) ──┐
       │  W 的左半 / 上半  │   │  W 的右半 / 下半  │
       └──────────┬───────┘   └────────┬─────────┘
                  └──── NCCL All-Reduce ────┘   每层 Attn 出口 + FFN 出口各一次
```

两条硬约束：
1. **进程数 = TP × PP**：这里 `mpirun -n 2` 对应 `tensor-para-size 2 × pipeline-para-size 1 = 2`。对不上会报错或挂死。
2. **权重要用 TP=2 那一套（`2-gpu`）**，不能拿单卡的 `1-gpu` 去跑双卡——切分布局不匹配，结果错乱。

### 5.2 TP 切矩阵的本质（Megatron 风格）

```
注意力/FFN 第一层（列并行）：Y = X·W，W 按列切 [W0 | W1]
   rank0 算 Y0 = X·W0，rank1 算 Y1 = X·W1  → 各得部分列，暂不通信
第二层（行并行）：Z = Y·V，V 按行切
   rank0、rank1 各算部分和 → 一次 All-Reduce 相加 = 完整 Z
```

- 通信量与 hidden、序列长度成正比，**对带宽敏感** → TP 最好放**单机内 NVLink** 互联的卡之间（这里 GPU2、GPU3 同机）。原理见 [[ai-infra/网络/集合通信原语]]、[[ai-infra/网络/InfiniBand]]。
- firefly-2b6 单卡其实装得下，做 TP=2 主要是为了**量"切两卡后每 token 时长变化"**——对小模型，通信开销可能抵消甚至超过算力翻倍的收益（见第 6 节）。

## 6. 性能指标怎么算（这正是统计脚本的产物）

脚本名里的 `stat_token` 点题：**逐 token 计时**。核心指标：

```
一条样本的时间线（output-token-len = 256）：
  ├─ Prefill：一次吃完 64 个 input token，算首个输出  → 计入 "首 token 延迟(TTFT)"
  └─ Decode：第 2~256 个 token，逐个生成
       平均每 token 生成时长 ≈ (总生成耗时 − prefill 耗时) / (输出 token 数 − 1)
```

| 指标 | 直觉定义 | 受什么影响 |
|------|----------|------------|
| 总耗时 / 样本 | 从喂入到生成完 256 token | input/output 长度、精度、卡数 |
| 平均每 token 时长 | decode 阶段每生成 1 token 的均时 | **decode 是访存密集**，主要看显存带宽 + Cache 读写 |
| 首 token 延迟(TTFT) | prefill 算出第一个 token 的耗时 | **prefill 是计算密集**，看算力(FLOPS) |
| 吞吐 (token/s) | 单位时间生成的 token 数 | batch、并行度 |

更系统的指标定义见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

数值直觉（手算示意，非实测）：假设单卡每 token 约 20ms，256 token 的 decode ≈ `255 × 20ms ≈ 5.1s`。若 TP=2 让算力翻倍但每层多两次 All-Reduce，对 2.6B 这种小模型，每 token 可能只降到 ~15ms（通信占比上来了），**并非线性加速**——这正是要跑 `tp2` 对比文件的原因：**用数据说话，而不是假设 TP 一定更快。**

为什么 decode 慢且难加速？

```
Prefill：64 个 token 一起算 → 矩阵又大又满 → GPU 算力吃得满（compute-bound）
Decode：每步只算 1 个新 token → 矩阵退化成"向量×矩阵" → 算力闲、卡在搬权重/KV（memory-bound）
```

所以 decode 的瓶颈是**把权重和 KV Cache 从 HBM 搬进来**，TP 把权重切小能减轻每卡搬运量，但又引入 All-Reduce——两者权衡，量出来才知道。KV Cache 机制见 [[llm-optimizer/kv-cache]]。

## 实操命令汇总（原文真料，原样保留）

### 评测数据格式

```
[
    {
        "id":0,
        "input":"#姓名：何#性别：女士#剩余额度：一万七千#当前额度：一万七千#绑定银行：无#活动：D#初始额度：nan#手机尾号：七八八#注册时间：二零二三年一月三日#借款日期：nan#优惠券数：0.0#电销时间：二零二三年四月二十三日#登录类型：无#提额时间：二零二三年四月二十二日#offer类型：提额#cust_type：授信T91-180\n\nHuman: 你好。\nAssistant: ",
        "answer":"Assistant: 好，请问一下是何女士对吧？</s>"
    },
    {
        "id":1,
        "input":"#姓名：岳#性别：先生#剩余额度：五万#…(多轮对话，略)…\nAssistant: ",
        "answer":"Assistant: 就是后你提前还款了之后，后面是不会收取这利息的哈。</s>"
    },
    {
        "id":2,
        "input":"#姓名：李#性别：先生#…(多轮对话，略)…\nAssistant: ",
        "answer":"Assistant: 您现在呢只需要将您的剩余额度两万元按去暂时借出来。</s>"
    }
]
```

> 字段：`id`（序号）、`input`（客户画像 # 分隔 + 多轮 `Human:`/`Assistant:` 对话，以空 `Assistant: ` 结尾让模型续写）、`answer`（参考回复，`</s>` 收尾）。完整三条原始样本见本仓库历史版本，此处中段省略以便阅读，字段结构完全一致。

### 推理（统计推理耗时、平均每 token 生成时长）

单卡：

```
CUDA_VISIBLE_DEVICES=1 python examples/pytorch/gpt/firefly_lambada_dianxiao_1w_stat_token.py \
--checkpoint-path /workspace/model/firefly-2b6-dx-1tp/belle7b/1/1-gpu \
--tokenizer-path /workspace/model/firefly-2b6-dx \
--dataset-path /workspace/data/lambada_test.jsonl \
--lib-path  /workspace/lib/libth_transformer.so \
--inference-data-type fp16 --show-progress --input-token-len 64 --output-token-len 256 \
--dianxiao-path-stat /workspace/output/firefly_random_sample_1w_256_stat_ft.json
```

双卡张量并行：

```
CUDA_VISIBLE_DEVICES=2,3  mpirun -n 2 python examples/pytorch/gpt/firefly_lambada_dianxiao_1w_stat_token.py \
--checkpoint-path /workspace/model/firefly-2b6-dx-2tp/belle7b/1/2-gpu \
--tokenizer-path /workspace/model/firefly-2b6-dx \
--dataset-path /workspace/data/lambada_test.jsonl \
--lib-path  /workspace/lib/libth_transformer.so \
--inference-data-type fp16 \
--tensor-para-size 2 \
--pipeline-para-size 1 \
--show-progress \
--input-token-len 64 \
--output-token-len 256 \
--dianxiao-path-stat  /workspace/output/firefly_random_sample_1w_256_stat_ft_tp2.json
```

> 这两条命令是 BLOOM/firefly 推理统计的核心真料。`libth_transformer.so`、`1-gpu`/`2-gpu` 权重目录都来自上级 [[llm-inference/faster-transformer/README]] 的编译与权重转换步骤。

## 常见问题 / 坑

| 现象 | 根因 | 处理思路 |
|------|------|----------|
| 双卡报错 / 卡死 | `mpirun -n` 与 `tensor-para-size × pipeline-para-size` 不符 | 保证进程数 = TP×PP（本文 2=2×1），一卡一 rank |
| 双卡加载失败 / 维度对不上 | 拿单卡 `1-gpu` 权重去跑 TP=2 | 必须用 TP=2 切好的 `2-gpu` 目录 |
| 找不到 `libth_transformer.so` | 路径错 / 没编译多卡版本 | 确认上级 README 的 `make -j` 产物路径；多卡需 `-DBUILD_MULTI_GPU=ON` |
| 输出乱码 / 全 `</s>` | tokenizer 与模型不匹配 / 模板不对 | `--tokenizer-path` 用配套 `firefly-2b6-dx`；input 保持 `Human:`/`Assistant:` 模板 |
| 单卡双卡耗时几乎没差 / 双卡更慢 | 2.6B 小模型，TP 的 All-Reduce 通信抵消了算力收益（memory-bound） | 正常现象；小模型不一定 TP 更快，看实测 `tp2.json` |
| 结果与 HF 略有差异 | FP16 + 算子融合的数值误差 | 正常；要逐位对齐就拉高精度对比 |
| 统计不可比 | 不同跑次 input/output 长度不一致 | 钉死 `--input-token-len`/`--output-token-len`（本文 64/256） |
| OOM | `output-token-len` 大 + batch + KV Cache | 降输出长度/batch，或上 TP 分摊权重，参考 [[llm-optimizer/kv-cache]] |

> 护栏：本文的脚本名、参数名、路径是**机制层面**的说明。**确切的镜像 tag、commit、脚本参数、默认值请以你所用 FT 分支的源码 / README 为准**，不同 fork 差异很大，切勿照抄硬记。

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 上级（FT 总览 + 编译/权重转换）：[[llm-inference/faster-transformer/README]]
- 推理总览：[[llm-inference/README]]
- 模型结构（BLOOM 是 decoder-only GPT 风格）：[[llm-algo/transformer/模型架构]]
- 位置编码对照（BLOOM 用 ALiBi，对比 RoPE）：[[llm-algo/旋转编码RoPE]]
- 显存与解码（KV Cache / FlashAttention）：[[llm-optimizer/kv-cache]] [[llm-optimizer/FlashAttention]]
- 解码策略（top_k/top_p/温度/beam）：[[llm-inference/解码策略]]
- 性能指标定义：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] [[llm-eval/README]]
- 通信底座（TP 靠它）：[[ai-infra/网络/集合通信原语]] [[ai-infra/网络/InfiniBand]]
- 在线服务对比（FT 之后的选型）：[[llm-inference/vllm/README]] [[llm-inference/PD分离]]
- 低精度提速：[[llm-compression/README]] [[llm-compression/quantization/量化基础]] [[llm-compression/quantization/fp8]]
- 训练侧框架（权重来源）：[[ai-framework/megatron-lm/README]] [[ai-framework/deepspeed/README]]
- 内存估算：[[docs/transformer内存估算]]
