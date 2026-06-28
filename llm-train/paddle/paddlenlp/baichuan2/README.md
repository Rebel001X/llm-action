# Baichuan2 在 PaddleNLP 上的训练与微调

> 用飞桨（PaddlePaddle）+ PaddleNLP 套件加载、全参微调、LoRA 微调并部署百川 Baichuan2 系列大模型（7B/13B），并讲清 Baichuan2 相对标准 LLaMA 的几处架构差异。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/paddle/README]] [[llm-train/paddle/paddlenlp/README]] [[llm-train/README]] [[llm-algo/transformer/模型架构]]

## 阅读地图

| 小节 | 你将搞清楚 | 关键词 |
|------|-----------|--------|
| 0 锚点 | 在 PaddleNLP 上跑 Baichuan2 到底分几步 | AutoModel / Trainer / 4D 并行 |
| 1 地基 | Baichuan2 是什么、和 LLaMA 差在哪 | NormHead / max_z_loss / Alibi |
| 2 加载 | 怎么把权重拉下来并加载成模型 | from_pretrained / dtype / 缓存目录 |
| 3 数据 | SFT 的数据长什么样、怎么对齐 | src/tgt / max_length / 模板 |
| 4 全参 SFT | 用 `finetune_generation.py` 全量微调 | launch / sft_argument.json |
| 5 并行与显存 | 7B/13B 要几张卡、怎么切 | TP/PP/Sharding / recompute |
| 6 LoRA | 显存不够时只训低秩增量 | rank / 合并参数 / 省显存 |
| 7 量化与推理 | 训完怎么压缩、怎么上线 | WINT4 / W8A8 / 动静态图 |
| 关键示例 | 配置项逐行解释 + 显存手算 | per_device_bs / grad_accum |
| 坑 | 高频报错与规避 | 词表大小 / dtype / shm |

## 0. 一句话锚点

**在 PaddleNLP 里跑 Baichuan2，本质就是一条标准流水线：`AutoModelForCausalLM.from_pretrained()` 把百川的权重加载成飞桨模型 → 准备 SFT 数据 → 用 `paddle.distributed.launch` 多卡拉起 `finetune_generation.py`（内部是 PaddleNLP `Trainer`）→ 训完导出/量化/部署。** 你写的代码和训 LLaMA、Bloom 几乎一模一样，真正的差异只在两处：(1) Baichuan2 模型本身有几个小改动（见 §1）；(2) 你要把并行度和显存账算对（见 §5）。

```
   下载权重           准备数据            多卡微调              部署
 ┌──────────┐      ┌──────────┐      ┌────────────────┐   ┌──────────┐
 │from_      │      │ SFT json │      │ launch +       │   │ 动/静态图 │
 │pretrained │ ───→ │ src/tgt  │ ───→ │ finetune_gen.py│ ─→│ 量化/服务 │
 │ (.pdparams)│      │ 对话模板 │      │ (Trainer+4D并行)│   │ Gradio   │
 └──────────┘      └──────────┘      └────────────────┘   └──────────┘
        百川权重托管在 bos/aistudio，PaddleNLP 自动下载并缓存
```

> 本目录原始内容只给了一段最小加载代码（`AutoTokenizer` + `AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Base", dtype="float16")`）。下面把它扩成一条能落地的完整训练链路。**所有具体参数名/默认值以 PaddleNLP 对应版本源码为准。**

## 1. 地基：Baichuan2 是什么、和 LLaMA 差在哪

Baichuan2 是百川智能开源的中英双语大模型，有 **7B / 13B** 两档，每档又分 **Base（基座）** 和 **Chat（对话对齐版）**。骨架是标准的 Decoder-only Transformer，**绝大部分和 LLaMA 同构**：RMSNorm、SwiGLU、旋转位置编码（7B 用 RoPE）。但有几处工程上必须知道的差异：

### 1.1 词表更大（中文友好）

Baichuan2 词表 **约 12.5 万**（`vocab_size ≈ 125696`），远大于 LLaMA 的约 3.2 万。这是为了对中文有更高的压缩率（同样一句中文用更少 token）。代价是 **Embedding 和输出 LM Head 这两层变大**：

$$\text{Embedding 参数} = V \times d_{model}$$

7B 模型 $d_{model}=4096$，则 Embedding ≈ $125696 \times 4096 \approx 5.15\text{亿}$ 参数——光这一层就比 LLaMA 多出一大块，做显存估算时不能忽略。

### 1.2 NormHead：对输出层权重做归一化

标准 LLaMA 的 LM Head 是普通线性层 $\text{logits} = h W^\top$。Baichuan2 在算 logits 前先对 Head 权重按行做 L2 归一化：

$$\hat{W}_i = \frac{W_i}{\|W_i\|_2}, \qquad \text{logits} = h\,\hat{W}^\top$$

直觉：让每个 token 的输出向量都落在单位球面上，**稳定训练、缓解少见 token 的范数漂移**，对超大词表尤其有用。工程上意味着加载/转换权重时要确认这一层的实现一致（PaddleNLP 已内置）。

### 1.3 max_z_loss：抑制 logits 爆炸

为防止 softmax 前的 logits 数值过大导致训练不稳，Baichuan2 在交叉熵之外额外加一项正则，惩罚归一化常数 $Z$（即 logsumexp）的平方：

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot z^2, \qquad z = \log\sum_j e^{\text{logit}_j}$$

其中 $\lambda$ 是很小的系数（原文用约 $2\times10^{-4}$，**具体见原文**）。它把 $z$ 往 0 拉，等价于约束所有 logits 不要整体偏大，从而让 fp16 下不易溢出。

### 1.4 13B 用 Alibi 而非 RoPE

这是 7B 和 13B 的关键分叉：

| | Baichuan2-7B | Baichuan2-13B |
|--|--------------|---------------|
| 位置编码 | RoPE（旋转） | **Alibi**（注意力线性偏置） |
| 层数 $L$ | 32 | 40 |
| 隐维 $d_{model}$ | 4096 | 5120 |
| 注意力头 | 32 | 40 |
| 上下文 | 约 4096 | 约 4096 |

Alibi 不修改 Q/K，而是在注意力打分上按相对距离加一个**线性递减的负偏置**：距离越远，得分被压得越低——这给了模型一定的长度外推能力。所以 7B/13B 在加载、并行切分上略有不同（如 head 数、层数影响 PP 切分）。

```
   Baichuan2 与标准 LLaMA 的差异（其余完全同构）
   ┌────────────────────────────────────────────────┐
   │ 输入 Embedding   ← 词表~12.5万（中文友好，层更大）│
   │   RMSNorm                                        │
   │   Self-Attn   ← 7B:RoPE / 13B:Alibi              │
   │   RMSNorm                                        │
   │   FFN (SwiGLU)                                   │
   │      ……×L 层                                     │
   │   Final RMSNorm                                  │
   │   LM Head     ← NormHead（权重按行L2归一化）      │
   │   Loss        ← CE + max_z_loss 正则             │
   └────────────────────────────────────────────────┘
```

> 想温习 Transformer 主干、注意力、FFN 的原理见 [[llm-algo/transformer/模型架构]]；RoPE/Alibi 都属于位置编码大类。

## 2. 加载模型：把权重变成飞桨对象

最小可用代码（原始文件给的就是这段）：

```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
model = AutoModelForCausalLM.from_pretrained(
    "baichuan-inc/Baichuan2-7B-Base", dtype="float16"
)
```

逐行说清楚每个名字在做什么：

| 元素 | 作用 | 注意点 |
|------|------|--------|
| `AutoModelForCausalLM` | 自回归（CausalLM）任务的"自动模型"，按权重名自动识别成 Baichuan2 结构 | 训练/推理统一入口 |
| `from_pretrained("baichuan-inc/...")` | 从百川官方仓库名下载并加载，权重缓存到 `~/.paddlenlp/models/` | 联网慢可换本地绝对路径 |
| `dtype="float16"` | 用 fp16 加载，显存减半 | 也可 `bfloat16`（Ampere+ 更稳）/`float32`（最准最占） |
| `AutoTokenizer` | 加载百川分词器（约 12.5 万词表的 SentencePiece） | 词表大，注意和模型版本配套 |

跑一次生成验证加载成功：

```python
input_features = tokenizer("登鹳雀楼->王之涣\n夜雨寄北->", return_tensors="pd")  # pd = paddle 张量
outputs = model.generate(**input_features, max_length=128)
print(tokenizer.batch_decode(outputs[0]))
```

下载慢/离线场景：先手动把权重放到缓存目录，再用**本地绝对路径**加载（和 bloom 的做法一致）：

```python
model = AutoModelForCausalLM.from_pretrained(
    "/home/guodong.li/.paddlenlp/models/baichuan-inc/Baichuan2-7B-Base",
    dtype="float16",
)
```

> Base 还是 Chat？**Base 适合做你自己的 SFT 起点**（它没被对齐过，可塑性强）；Chat 已经做过指令对齐，适合直接推理或在其上做小幅继续微调。

## 3. 准备 SFT 数据：模型到底吃什么

监督微调（SFT）= 给一堆"输入→期望输出"，让模型学会按指令回答。PaddleNLP 的对话/生成微调数据通常是 **每行一个 JSON**（`jsonl`），核心是源（src）与目标（tgt）：

```json
{"src": "请把下面这句话翻译成英文：今天天气真好。", "tgt": "The weather is really nice today."}
{"src": "用一句话解释什么是张量并行。", "tgt": "把单层大权重矩阵切到多张卡上分别计算再合并的并行方式。"}
```

训练时框架会把它拼成模型实际看到的序列（概念）：

```
  实际喂给模型的 token 序列（SFT 的 loss 只在 tgt 段计算）
  ┌──────────── 不算 loss ────────────┐┌──── 算 loss ────┐
  [BOS] <模板><src 用户提问> <分隔/角色标记>  <tgt 模型回答> [EOS]
        └──── prompt 部分(mask 掉) ────┘└── 监督信号 ──┘
```

关键点：

- **只对 tgt 段算损失**：prompt（用户问题）部分被 mask 掉，模型只学"该怎么答"，不学"复读问题"。
- **`max_length` / `src_length`**：`src_length` 限制输入最大长度、`max_length` 限制整条（src+tgt）最大长度；超了截断。设太长 → 显存暴涨；太短 → 长样本被切丢答案。
- **对话模板**：Chat 模型有固定角色标记（如用户/助手轮次），SFT 时要用和该模型一致的模板，否则推理时格式对不上。

## 4. 全参微调：用 `finetune_generation.py` + launch

PaddleNLP 的 `llm` 目录提供统一脚本 `finetune_generation.py`，配置全部写在一个 JSON 里（和 bloom 的 `sft_argument.json` 同款）。启动方式就是用 launch 拉多卡：

```bash
# 单机 4 卡全参 SFT（脚本接受一个 json 配置）
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
    finetune_generation.py ./baichuan2/sft_argument.json
```

一个可参考的 `sft_argument.json`（字段含义见下方"关键示例"小节逐行解释）：

```json
{
    "model_name_or_path": "baichuan-inc/Baichuan2-7B-Base",
    "dataset_name_or_path": "/home/guodong.li/workspace/data/your_sft_data",
    "output_dir": "./checkpoints/baichuan2_sft_ckpts",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 3,
    "learning_rate": 3e-05,
    "warmup_steps": 30,
    "src_length": 1024,
    "max_length": 2048,
    "bf16": true,
    "fp16_opt_level": "O2",
    "recompute": true,
    "tensor_parallel_degree": 4,
    "pipeline_parallel_degree": 1,
    "sharding": "stage2",
    "do_train": true,
    "do_eval": true,
    "save_total_limit": 1
}
```

```
   launch 干了什么（单机 4 卡示意）
   python -m paddle.distributed.launch --gpus "0,1,2,3"
            │  设置 rank/world_size/通信端点 等环境变量
            ▼
   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
   │rank0│ │rank1│ │rank2│ │rank3│  ← 4 个进程，每个绑一张 GPU
   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
      └───────┴── NCCL AllReduce/Send-Recv ──┴───────┘
        Trainer 内部按 TP/PP/Sharding 切分并同步
```

> launch 只负责"把 N 个进程拉起来并配好通信环境"；真正的并行切分由 JSON 里的 `tensor_parallel_degree` / `pipeline_parallel_degree` / `sharding` 决定。**参数名以你所用 PaddleNLP 版本为准。**

## 5. 并行策略与显存账：7B/13B 要几张卡

这是大模型训练真正的难点。先把"放不放得下"算清楚（详见 [[docs/transformer内存估算]]）。

### 5.1 全参微调显存的四大块

对参数量 $P$ 的模型，全参 + Adam 优化器的常驻显存约为：

$$M \approx \underbrace{2P}_{\text{fp16参数}} + \underbrace{4P}_{\text{fp32主参}} + \underbrace{8P}_{\text{Adam: m,v}} + \underbrace{2P}_{\text{梯度}} + M_{\text{激活}}$$

混合精度 Adam 训练里光"参数+优化器+梯度"就约 **$16P$ 字节**（这里把 fp32 主副本与一/二阶动量都算上）。

### 5.2 7B 手算（不含激活）

$$P = 7\times10^9 \Rightarrow 16P = 16 \times 7\times10^9 = 1.12\times10^{11}\,\text{B} \approx 112\,\text{GB}$$

一张 80GB 卡放不下 → **必须切**。再加激活（取决于 batch 和序列长度），实际更高。

### 5.3 用并行/重计算把它摊下去

| 手段 | 省什么 | 代价 |
|------|--------|------|
| `tensor_parallel_degree=4`（TP=4） | 把单层权重切 4 份，参数/优化器显存约降到 1/4 | 层内 AllReduce，最重，TP 必须放机内走 NVLink |
| `sharding="stage2/3"`（类 ZeRO） | 切优化器状态(2)/再切梯度参数(3) | 通信变多，stage3 最省也最慢 |
| `pipeline_parallel_degree`（PP） | 按层切 stage，省的是层数维度 | stage 间传激活、有气泡 |
| `recompute=true`（重计算） | 不存激活，反向时重算 → 砍激活显存 | 多约 30% 算力换显存 |

把 112GB 摊到 TP=4 上：参数侧约 $112/4 = 28$GB/卡，再叠加激活与通信缓冲，4×80GB 通常足够训 7B 全参；**13B 更大（约 $16\times13\times10^9 \approx 208$GB），一般需要 TP×Sharding 甚至加 PP**。

```
   7B 全参微调放卡示意（TP=4，机内 NVLink）
   单卡放不下 112GB ─────────────────┐
                                      ▼ 切成 4 份
   ┌──────┐NVLink┌──────┐NVLink┌──────┐NVLink┌──────┐
   │GPU0  │──────│GPU1  │──────│GPU2  │──────│GPU3  │
   │~28GB │      │~28GB │      │~28GB │      │~28GB │  + 激活/通信
   └──────┘      └──────┘      └──────┘      └──────┘
      层内 AllReduce 同步（最频繁，故必须同机）
```

> TP/PP/DP/Sharding 的原理、卡布局原则（重通信放机内）见 [[llm-train/paddle/README]] §6 与 [[llm-train/pytorch/distribution/README]]；集合通信原语见 [[ai-infra/网络/集合通信原语]]。

## 6. LoRA：单卡也能微调

当卡不够（比如就 1 张 24/48GB），**LoRA** 是首选：冻结原始权重 $W$，只训练一个低秩增量 $\Delta W = BA$（$A\in\mathbb{R}^{r\times d}, B\in\mathbb{R}^{d\times r}$，秩 $r$ 很小如 8）：

$$h = Wx + \Delta W x = Wx + B(Ax)$$

可训练参数从 $d^2$ 降到 $2rd$。对 7B、$d=4096$、单个投影：全参 $\approx 4096^2 = 1.68\times10^7$；LoRA（$r=8$）$\approx 2\times8\times4096 = 6.55\times10^4$——**少了约 256 倍**。优化器状态随之骤降，单卡即可训。

```bash
# LoRA 微调：在 json 里打开 lora 开关（字段名以版本为准）
python -u -m paddle.distributed.launch --gpus "0" \
    finetune_generation.py ./baichuan2/lora_argument.json
```

训完要**合并参数**才能像普通模型一样部署（把 $\Delta W$ 加回 $W$）：

```bash
# 合并 LoRA 增量到基座权重（导出/静态图推理前的必做步骤）
python merge_lora_params.py \
    --model_name_or_path baichuan-inc/Baichuan2-7B-Base \
    --lora_path ./checkpoints/baichuan2_lora_ckpts \
    --output_path ./checkpoints/baichuan2_lora_merged
```

> LoRA 只动很小一块增量，所以**学不动需要大改的知识**；它最适合"风格/任务对齐"这类微调。原理同 [[llm-alignment/DPO]] 之外的 PEFT 家族。**合并脚本名/参数以官方为准。**

## 7. 量化与推理部署

训练只是上半场，上线还要压缩 + 服务化（和 bloom 文档同一套流程）。

- **量化**：PaddleNLP 配 PaddleSlim 提供 **WINT4（权重 4bit）** 与 **W8A8（权重+激活 8bit）**，显著减少显存与带宽。注意 Baichuan2 的 NormHead/大词表层在量化时可能要特殊处理，跟随官方配置。量化基础见 [[llm-compression/quantization/量化基础]]、fp8 见 [[llm-compression/quantization/fp8]]。
- **动态图推理**：直接用 `predictor.py --mode dynamic`，灵活、易调试。
- **静态图推理**：先 `export_model.py` 把动态图导成静态图（LoRA 要先合并），再 `predictor.py --mode static`，**部署更快**（动静统一是飞桨卖点）。
- **服务化**：`flask_server.py` 起 HTTP / Gradio UI，多卡用 launch 拉起。

```
   训练 → 上线的下半场
   ckpt ──(合并LoRA)──→ 量化(WINT4/W8A8) ──→ 导出静态图 ──→ predictor/flask 服务
                         省显存/带宽          部署提速        对外提供 API
```

## 关键示例：`sft_argument.json` 逐行解释 + 显存手算

对照 §4 的配置（数值借鉴 bloom 的 `sft_argument.json`，**实际请按你的数据和卡数调**）：

| 字段 | 含义 | 调参直觉 |
|------|------|----------|
| `per_device_train_batch_size` | 每张卡每步的样本数 | 大→快但更吃显存；7B 全参常设 1~2 |
| `gradient_accumulation_steps` | 累积多步梯度再更新 | 用来"凑大 batch"而不爆显存 |
| `learning_rate` | 学习率 | SFT 常用 1e-5~3e-5，太大易崩 |
| `warmup_steps` | 预热步数 | 前期小步长稳住训练 |
| `num_train_epochs` | 训练轮数 | SFT 多为 2~3，太多易过拟合 |
| `src_length` / `max_length` | 输入/总长上限 | 决定激活显存与截断 |
| `bf16` / `fp16` + `fp16_opt_level:"O2"` | 混合精度 | O2 更激进省显存；A100+ 优先 bf16 更稳 |
| `recompute` | 重计算省激活 | true 省显存、约 +30% 算力 |
| `tensor_parallel_degree` | TP 并行度 | 7B 全参常用 4；务必机内 |
| `pipeline_parallel_degree` | PP 并行度 | 层多/卡多时叠加 |
| `sharding` | ZeRO 式切分 | stage2 切优化器、stage3 再切参数 |
| `save_total_limit` | 最多保留几个 ckpt | 省磁盘 |

**有效 batch 怎么算**（决定真正的优化步长）：

$$B_{\text{eff}} = \text{per\_device\_bs} \times \text{grad\_accum} \times \text{数据并行卡数}$$

例：`per_device_train_batch_size=1`、`gradient_accumulation_steps=16`、TP=4/PP=1 占满 4 卡（无额外 DP，DP=1）→ $B_{\text{eff}} = 1 \times 16 \times 1 = 16$。若想更大有效 batch，增大 `gradient_accumulation_steps` 比增大 per-device bs 更省显存。

**13B vs 7B 的显存差**（仅参数+优化器，$16P$ 估）：

$$\frac{M_{13B}}{M_{7B}} = \frac{16\times13\times10^9}{16\times7\times10^9} \approx 1.86\times$$

所以从 7B 升 13B，并行度通常也要相应加码（TP 翻倍或叠 Sharding/PP）。

## 评价/对照/局限

| 维度 | 说明 |
|------|------|
| 与 LLaMA 同构度 | 极高；差异仅 NormHead / max_z_loss / 大词表 /（13B）Alibi |
| 中文能力 | 大词表 + 中英语料，中文压缩率和效果好 |
| 全参 vs LoRA | 全参效果上限高但要多卡；LoRA 单卡可训、改动有限 |
| 7B vs 13B | 13B 效果更好但显存约 1.86×、位置编码换 Alibi |
| 量化部署 | WINT4/W8A8 可上线；NormHead/大词表层注意特殊处理 |
| 局限/护栏 | 具体并行参数名、默认值、量化精度损失、$\lambda$ 系数等**以官方/原文为准**，本文数字为估算 |

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-train/paddle/README]] — 飞桨训练环境与 4D 并行总览（本文的上游）
- [[llm-train/paddle/paddlenlp/README]] — PaddleNLP 套件（含 bloom/llama 对照）
- [[llm-train/README]] — 大模型训练总览
- [[llm-algo/transformer/模型架构]] — Transformer / 注意力 / FFN 主干
- [[llm-train/pytorch/distribution/README]] — PyTorch 分布式（对照理解 TP/PP/DP）
- [[ai-infra/网络/集合通信原语]] — AllReduce/Send-Recv 等通信原语
- [[llm-compression/quantization/量化基础]] — WINT4/W8A8 量化原理
- [[llm-compression/quantization/fp8]] — fp8 低比特
- [[docs/transformer内存估算]] — 参数/优化器/激活显存估算
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] — 吞吐/有效 batch 等指标
