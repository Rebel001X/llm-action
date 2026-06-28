# PaddleNLP 微调 BLOOM 实战(SFT + 张量并行 + 重计算)

> 用 PaddlePaddle/PaddleNLP 对 BLOOM(bloomz-560m)做监督微调(SFT),核心招式是 `张量并行(TP=4) + 重计算(recompute) + FP16-O2`。这是"用国产框架把一个多语言因果 LM 调到下游任务"的最小可跑通范式。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]] · [[B07:llm-inference/大模型推理张量并行]] · [[docs/transformer内存估算]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|--------------|--------|
| 0 | 一句话锚点 | SFT / TP / recompute |
| 1 | BLOOM 是什么、为什么选它 | ALiBi / 多语言 / 因果LM |
| 2 | PaddleNLP 训练栈与环境 | venv / paddlenlp / launch |
| 3 | SFT 数据流(AdvertiseGen) | 指令模板 / src_length |
| 4 | 张量并行 TP=4 拆解 | 列切/行切 / all-reduce |
| 5 | 重计算 recompute | 显存换算力 / 激活值 |
| 6 | FP16-O2 与 batch 账 | 梯度累积 / 有效批量 |
| 7 | `sft_argument.json` 逐字段 | 配置即实验 |
| 公式区 | 显存/通信/有效批量手算 | 数值示例 |
| 评价 | 与 PyTorch/DeepSpeed 对照 | 局限 |

## 0. 一句话锚点

**这件事到底在做什么**:把一个 5.6 亿参数的多语言因果语言模型 `bloomz-560m`,在广告文案生成数据集 `AdvertiseGen` 上做**全参数监督微调(full SFT)**;为了在多卡上跑起来并省显存,用 **4 路张量并行** 把每个权重矩阵横/竖切成 4 份分到 4 张卡,再用 **重计算** 把前向激活值丢掉、反向时重算,最后用 **FP16-O2 混合精度** 把权重和计算压到半精度。

一条命令即全部:

```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    finetune_generation.py ./bloom/sft_argument.json
```

## 1. 地基:BLOOM 是什么、为什么这样训

### 1.1 BLOOM 模型本身

BLOOM(BigScience Large Open-science Open-access Multilingual)是 BigScience 协作开源的**仅解码器(decoder-only)因果语言模型**家族,核心特征:

- **多语言**:覆盖 46 种自然语言 + 13 种编程语言(以官方为准),中文也在其中,所以适合做中文下游任务。
- **`bloomz`** 是在指令数据上做过多任务微调(instruction-tuned)的 BLOOM 变体,zero-shot 跨语言泛化更好,作为 SFT 起点比裸 `bloom` 收敛更快。
- **位置编码用 ALiBi(Attention with Linear Biases)** 而非 RoPE/绝对位置嵌入。ALiBi 不加位置 embedding,而是在注意力分数上叠加一个与"query-key 距离"成正比的线性惩罚偏置,使模型外推到更长序列时更稳。

ALiBi 注意力打分(简化):

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} - m \cdot (i - j)$$

其中 $m$ 是每个注意力头固定的斜率(slope),$(i-j)$ 是相对距离。距离越远,惩罚越大——这就是 ALiBi"无参数、可外推"的精髓。

### 1.2 为什么需要 TP / recompute / FP16

560m 模型本身不大,单卡 fp16 也放得下,**本配置真正的教学价值是把分布式训练全套机制串起来**:这套 `TP + recompute + FP16-O2` 配方直接照搬到 7B/176B 就是真实生产训练。所以把它当"分布式训练的最小教学沙盘"来读。

```
   一句话因果关系
   ┌─────────────────────────────────────────────┐
   │  模型/激活太大放不下单卡显存                   │
   │        │                                      │
   │        ├──► 张量并行 TP  : 权重切到多卡(省静态)│
   │        ├──► 重计算 recompute: 丢激活反向重算(省动态)│
   │        └──► FP16-O2     : 半精度(省一半)      │
   └─────────────────────────────────────────────┘
```

## 2. PaddleNLP 训练栈与环境

### 2.1 建虚拟环境

```bash
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 paddle-venv-py310-cu117
source /home/guodong.li/virtual-venv/paddle-venv-py310-cu117/bin/activate
# 安装(版本以官方为准)
pip install --upgrade paddlenlp==2.6.1 -i https://pypi.org/simple
# paddlepaddle-gpu 需与 CUDA 版本匹配(此处 cu117),命令以官方 whl 页面为准
```

> 护栏:具体 paddlepaddle-gpu 安装命令/版本号随 CUDA、PaddleNLP 版本变化,**以 paddlepaddle.org.cn 官方 whl 页面为准**,不要硬背。

### 2.2 训练栈分层

```
┌──────────────────────────────────────────────┐
│ sft_argument.json   ← 实验配置(你改这里)       │
├──────────────────────────────────────────────┤
│ finetune_generation.py ← 训练入口脚本          │
├──────────────────────────────────────────────┤
│ paddlenlp.trainer.Trainer ← 训练循环/分布式封装 │
├──────────────────────────────────────────────┤
│ AutoModelForCausalLM / AutoTokenizer ← 模型层  │
├──────────────────────────────────────────────┤
│ paddle.distributed.launch ← 多进程拉起/通信组   │
├──────────────────────────────────────────────┤
│ PaddlePaddle 框架 + NCCL + CUDA                │
└──────────────────────────────────────────────┘
```

`paddle.distributed.launch --gpus "0,1,2,3"` 会在一台机器上拉起 4 个进程(每张卡一个 rank),并建立 NCCL 通信组;`--tensor_parallel_degree=4` 告诉框架"这 4 个 rank 组成一个张量并行组"。

### 2.3 先确认单机推理跑通(冒烟测试)

正式微调前,先用最小代码确认模型/分词器能加载、能生成,排除环境问题:

```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", dtype="float16")

# 也可指向本地已下载目录,避免重复下载
# AutoModelForCausalLM.from_pretrained("/home/guodong.li/.paddlenlp/models/bigscience/bloomz-560m", dtype="float16")

input_features = tokenizer("hi, my name is", return_tensors="pd")  # pd=PaddlePaddle tensor
outputs = model.generate(**input_features, max_length=128)
print(tokenizer.batch_decode(outputs[0]))
```

注意 `return_tensors="pd"`:这是 Paddle 的张量标记(对应 PyTorch 的 `"pt"`、TF 的 `"tf"`)。

## 3. SFT 数据流:AdvertiseGen 怎么进模型

`dataset_name_or_path` 指向 `AdvertiseGen`——一个"商品属性 → 广告文案"的中文生成数据集(典型样本:输入若干商品标签,输出一段卖点描述)。SFT 的本质是把"指令/输入"和"期望输出"拼成一条序列,只在输出部分算 loss。

```
原始样本: {"content": "类型#裙*风格#简约*图案#条纹", "summary": "这条..."}
                  │
                  ▼  套指令模板
┌───────────────────────────────────────────────┐
│  [src: content(≤1024 token)] [tgt: summary]     │
│  └─────── 不算 loss ──────┘ └── 算 loss(label)──┘│
└───────────────────────────────────────────────┘
                  │
                  ▼  分词 + padding/truncation
        input_ids / labels  (max_length=2048)
```

两个长度旋钮:

- `src_length=1024`:输入(prompt)最多保留 1024 token,超出截断。
- `max_length=2048`:整条序列(src+tgt)上限 2048 token。

`eval_with_do_generation=false` 表示评估时**不做自回归生成**,只算 teacher-forcing 的 token 级 loss/accuracy(快很多);`metric_for_best_model="accuracy"` + `load_best_model_at_end=true` 表示按验证集 accuracy 选最优 checkpoint。

## 4. 张量并行 TP=4 拆解(本配置的核心)

张量并行(Tensor Parallelism, TP)把**单个权重矩阵**切到多张卡,每张卡只存/算一部分,中间用集合通信拼回。BLOOM 的每个 Transformer 层有两块大矩阵:注意力的 QKV/输出投影、FFN 的两层。

### 4.1 列切 + 行切(Megatron 范式)

以 FFN 两层 $Y = \text{GeLU}(XA)B$ 为例,标准做法:

- 第一层 $A$ 按**列**切成 $[A_1, A_2, A_3, A_4]$,每卡算 $XA_i$,各卡得到一段中间结果,**无需通信**(GeLU 逐元素,可独立做)。
- 第二层 $B$ 按**行**切成 $[B_1;B_2;B_3;B_4]$,每卡算 $\text{GeLU}(XA_i)B_i$,得到部分和,最后 **all-reduce 求和** 得到完整 $Y$。

```
         X (每卡都有完整 X)
          │
   ┌──────┼──────┬──────┬──────┐
  GPU0   GPU1   GPU2   GPU3
  XA₁    XA₂    XA₃    XA₄     ← 列切, 不通信
  GeLU   GeLU   GeLU   GeLU
  ·B₁    ·B₂    ·B₃    ·B₄     ← 行切, 各得部分和
   └──────┴──── all-reduce ────┘
          │
          Y (完整, 每卡一致)
```

注意力同理:QKV 投影按头(head)列切,每卡算自己那批头的注意力,输出投影行切 + all-reduce。

### 4.2 通信代价

每个 Transformer 层在**前向**有 2 次 all-reduce(注意力块 1 次 + FFN 块 1 次),**反向**对称再来 2 次。所以 TP 的通信量随层数线性增长,且发生在前向/反向关键路径上——**这就是 TP 通常只在单机内 NVLink 高带宽下用、不跨机的根本原因**。

`pipeline_parallel_degree=1` 表示本配置**不开流水线并行**,只用 TP。4 卡全部归一个 TP 组,数据并行度 = 总卡数 / (TP × PP) = 4 / (4×1) = **1**(没有数据并行)。

## 5. 重计算 recompute(显存换算力)

`recompute=true`(又名 gradient/activation checkpointing):前向时**只保留每层的输入**,丢弃层内中间激活;反向需要这些激活时,**重新做一次该层前向**算出来。

```
普通训练:                    重计算:
fwd ─ 存所有激活 ─► bwd       fwd ─ 只存层输入 ─► bwd
   显存 ∝ 激活总量               反向时再 fwd 一次该层
   (大)                         显存 ∝ √层数 (小)
                                代价: 多 ~1 次前向算力 (慢 ~20-30%)
```

权衡:**激活显存** 从 $O(L)$ 降到约 $O(\sqrt{L})$(L 为层数),代价是反向时多一遍前向,约 +30% 计算量(具体随实现/序列长度变化,见原文/官方)。在 `max_length=2048` 这种长序列下,激活显存是主要瓶颈,recompute 收益明显。

## 6. FP16-O2 与 batch 账

- `fp16=true` + `fp16_opt_level="O2"`:O2 是更激进的纯半精度策略——权重、激活、大部分计算都用 FP16,只在数值敏感处(如 loss scaling、部分累加)保留 FP32。相对 O1(自动混合,部分算子白名单 FP32),O2 显存更省、更快,但需要 loss scaling 防下溢。
- `per_device_train_batch_size=100`:每卡每步 100 条。
- `gradient_accumulation_steps=100`:累积 100 个 micro-step 再更新一次参数。

**有效批量(本配置最容易踩坑的地方)**:由于 TP=4 把 4 张卡组成**一个**模型副本(它们处理的是同一份数据的不同切片,而非不同数据),数据并行度=1。所以:

$$\text{有效批量} = \text{per\_device\_bs} \times \text{grad\_accum} \times \text{DP数}$$
$$= 100 \times 100 \times 1 = 10000 \ \text{条/优化步}$$

(若把 TP 误当 DP,会错算成 ×4 = 40000——这是初学者常见误解。)`warmup_steps=30`、`num_train_epochs=3`、`learning_rate=3e-5` 都是按"优化步"而非"micro-step"计数的。

## 关键公式 / 数值示例

### 7.1 全参 SFT 显存粗算(Adam, FP16-O2)

单卡需驻留的"模型态"显存按经验公式(Adam + 混合精度):

$$M_{\text{model}} \approx P \times (2_{\text{fp16权重}} + 2_{\text{fp16梯度}} + 4_{\text{fp32权重副本}} + 4_{\text{m}} + 4_{\text{v}}) = 16P \ \text{字节}$$

560m 参数(P=5.6e8)单卡完整持有:

$$16 \times 5.6\times10^8 \approx 9.0\ \text{GB}$$

开 TP=4 后,权重/优化器态均匀切 4 份,每卡模型态 ≈ $9.0/4 \approx 2.2$ GB。剩余显存留给激活——而激活又被 recompute 进一步压低。**结论:本配置在消费级/单机 4 卡上很宽松,真正意义在于"配方可直接放大到 7B+"**。

### 7.2 TP 通信量手算

设隐藏维 $h$、批量 $b$、序列 $s$。每次 all-reduce 传输的张量大小约 $b \times s \times h$ 个元素。Ring all-reduce 单次通信量约 $2(N-1)/N \times$ 张量大小($N$=TP度)。每层前向 2 次、反向 2 次:

$$\text{每层每步通信} \approx 4 \times 2\cdot\frac{N-1}{N}\cdot (b\,s\,h)\ \text{元素}$$

以 N=4:$2(N-1)/N = 1.5$,即每次 all-reduce 实际搬运约 1.5 倍张量。这解释了"TP 度越高、单机带宽吃得越紧",也是 TP 一般 ≤ 单机卡数的原因。

### 7.3 recompute 省显存比例

设无 recompute 时激活显存为 $A$,分块 recompute(每层一段)后约降到 $\sqrt{L}\cdot A/L$ 量级。L=24 层(bloom-560m 量级)时,$\sqrt{24}/24 \approx 0.20$,即激活显存约降到 **1/5**,代价 +1 次前向。

## 评价 / 对照 / 局限

| 维度 | PaddleNLP(本配置) | PyTorch + DeepSpeed | 备注 |
|------|-------------------|---------------------|------|
| 并行原语 | TP/PP/DP 内建,配置即开 | ZeRO/TP/PP 需 Megatron-DS | Paddle 配置化更省心 |
| 混合精度 | O1/O2 内建 | amp/bf16 | O2≈纯半精 |
| 省显存 | recompute + TP | ZeRO-1/2/3 + ckpt | 思路不同:切权重 vs 切优化器态 |
| 生态 | 中文/国产链路友好 | 社区/资料更多 | 选型看团队 |
| 本例规模 | 560m 偏小 | — | 仅作教学沙盘 |

**局限与注意**:

1. 560m 体量下 TP=4 是"杀鸡用牛刀",真实价值是**配方可平移到大模型**;小模型上 TP 通信反而可能成为净开销。
2. `eval_with_do_generation=false`:验证 accuracy 是 token 级 teacher-forcing,**不等于真实生成质量**,最终要另跑生成评测。
3. 路径(`/home/guodong.li/...`)、版本号(paddlenlp==2.6.1)是作者环境快照,**复现以官方文档/你自己的路径为准**。
4. 数据集 AdvertiseGen 的指令模板、字段名以 PaddleNLP `llm` 目录当前版本为准,不同版本脚本参数可能变化。

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引枢纽
- [[llm-train/README]] — 训练总览,本文是其中"Paddle 分支"的一个落点
- [[llm-train/pytorch/distribution/README]] — 对照 PyTorch 分布式训练栈
- [[B07:llm-inference/大模型推理张量并行]] — 同一个 TP 思想在推理侧的应用
- [[docs/transformer内存估算]] — 第 7 节显存公式的完整推导
- [[ai-infra/网络/集合通信原语]] — all-reduce/all-gather 原理
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] — 有效批量/吞吐等指标定义
