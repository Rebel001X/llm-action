# ChatGLM 微调

> 把开源对话基座 ChatGLM-6B「掰」到你自己的任务上：用 **P-Tuning v2** 省显存、用 **LoRA** 折中、用 **全量微调** 上限最高；本目录给出了官方 `main.py` + DeepSpeed 的可跑脚本。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/chatglm/README]] [[llm-train/peft/Prefix-Tuning]]

## 阅读地图

| 节 | 你会得到什么 | 适合谁 |
|----|--------------|--------|
| 0 | 一句话锚点：三种微调到底改了什么 | 想 30 秒抓住本质 |
| 1 | 地基：ChatGLM 是什么、为什么不能直接全量训 | 刚接触的人 |
| 2 | 三条路线总览（P-Tuning v2 / LoRA / 全量）对比图 | 选型决策 |
| 3 | P-Tuning v2：机制 + `pre_seq_len` + ASCII 数据流 | 显存紧张的人 |
| 4 | LoRA：低秩旁路 + 与 v2 的取舍 | 想要可叠加适配器 |
| 5 | 全量微调 + DeepSpeed ZeRO 显存账 | 有多卡、追上限 |
| 6 | 数据格式：AdvertiseGen / content-summary / 多轮 | 准备数据集 |
| 7 | 显存逐项估算（6B 一张表看懂） | 估算资源 |
| 8 | 典型流程：训练循环 + 调用链 ASCII | 想跑通全程 |
| 9 | 部署与推理：prefix_encoder 怎么加载 | 上线落地 |
| 配 | 本目录脚本逐参数讲解 | 照着改 |
| 问 | 常见坑（loss 不一致 / OOM / 复读） | 排错 |

## 0. 一句话锚点

> **全量微调改「权重 $W$」；LoRA 改「权重的一个低秩增量 $\Delta W=BA$」；P-Tuning v2 一个权重都不改，只在每层注意力的 K/V 前面插一段学出来的「前缀」。**

三句话记住差别：
1. **P-Tuning v2**：冻结 6B 主干，只训前缀（约 0.1%~0.5% 参数）。最省显存，可单卡 / 消费级卡跑。
2. **LoRA**：冻结主干，给若干线性层挂低秩旁路（约 0.1%~1% 参数）。适配器可插拔、可合并回主干。
3. **全量**：所有 62 亿参数都更新。效果上限最高，但要多卡 + DeepSpeed ZeRO 才放得下。

本目录（`llm-train/chatglm/`）对应官方 `ChatGLM-6B/ptuning` 的工程：`main.py` 是入口，`train.sh` 跑 P-Tuning v2，`ds_train_finetune.sh` 跑全量微调，`deepspeed.json` 是 ZeRO 配置，`inference.py` 演示推理。

## 1. 地基：ChatGLM 是什么，为什么不能随手全量训

### 1.1 ChatGLM-6B 极简画像
- **62 亿参数**的双语（中英）对话模型，结构源自 **GLM**（General Language Model，自回归填空预训练），不是纯 GPT 也不是纯 BERT。
- 提供 FP16 / INT8 / INT4 量化权重；INT4 下推理仅需约 6GB 显存，因此「人人能玩」是它出圈的关键。
- 对话接口是 `model.chat(tokenizer, query, history=[...])`，`history` 是多轮上下文列表。

### 1.2 为什么「全量微调」对个人很重 —— 一笔显存账的直觉
训练一个参数，显存里通常要同时住下：**参数本身 + 梯度 + 优化器状态（Adam 的一阶/二阶矩）**。用混合精度 + Adam 粗算，每个参数大约要 **16 字节**（FP16 权重 2 + FP16 梯度 2 + FP32 权重副本 4 + Adam 两个矩各 4）：

$$6\times10^9 \times 16\ \text{B} \approx 96\ \text{GB}$$

这还没算激活值。一张 24GB 卡根本放不下 —— **这就是 PEFT（参数高效微调）存在的全部理由**：冻住这 96GB 的大头，只训练极小一部分新参数，把「要存优化器状态的参数」从 60 亿降到几百万。

## 2. 三条路线总览

```
                         ChatGLM-6B 主干（62 亿参数）
                         ┌───────────────────────────┐
                         │  28 层 GLM Transformer     │
                         └───────────────────────────┘
   ┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
   │      全量微调            │           LoRA           │      P-Tuning v2          │
   ├──────────────────────────┼──────────────────────────┼──────────────────────────┤
 改 │ 全部 W 都更新            │ 给 q/v 等线性层挂 BA      │ 每层 K/V 前插 prefix 向量 │
 冻 │ 无                       │ 冻结主干 W               │ 冻结整个主干              │
 训 │ 100%                     │ ~0.1%–1%                 │ ~0.1%–0.5%                │
存 │ 多卡 + ZeRO（80GB+）      │ 单卡可行（看秩）         │ 单卡 / 消费卡可行         │
 产 │ 新的完整模型             │ 小适配器（可合并）       │ 小 prefix_encoder 权重    │
 │ 效果上限最高            │ 折中、可叠加多任务       │ 最省、生成式任务友好      │
   └──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

选型口诀：**先 P-Tuning v2 验证数据/任务是否可行 → 不够再 LoRA → 还不够且有卡再上全量。**

## 3. P-Tuning v2（本目录默认路线）

### 3.1 它是什么
P-Tuning v2 ≈ 工程化的 **Prefix-Tuning**（机制细节见 [[llm-train/peft/Prefix-Tuning]]）：在**每一层** Transformer 的注意力里，把一段**可训练的虚拟 token** 拼到 Key / Value 序列最前面。主干完全冻结，只训这段前缀对应的 `prefix_encoder`。

「v1 只在输入层加 → 信号传到深层衰减；v2 每层都加 → 表达力强、能做 NLG（生成）任务」是 v1→v2 的核心进化。

### 3.2 数据流（一层注意力内部）

```
真实 token 序列 X ──► Q=XWq ─┐
                              ├──► Attn(Q, [Pk; K], [Pv; V]) ──► 输出
冻结的 Wk/Wv ── K=XWk, V=XWv ─┘            ▲      ▲
                                           │      │
        prefix_encoder（唯一可训练）── Pk ──┘      │
        长度 = pre_seq_len ──────────── Pv ────────┘
```

- `Pk, Pv` 是凭空学出来的、不对应任何真实词的向量；后面每个真实 token 在 softmax 时都会「注意到」它们，从而把行为往任务方向掰。
- **唯一更新的参数**就是生成 `Pk/Pv` 的 `prefix_encoder`（通常一个 embedding + MLP 重参数化，重参数化是为了训练稳定）。

### 3.3 关键开关 `pre_seq_len`
脚本里 `PRE_SEQ_LEN=128` 即前缀长度 $L$。

| pre_seq_len | 含义 | 权衡 |
|-------------|------|------|
| 小（如 32） | 前缀短，可训参数少 | 更省显存，表达力弱，难任务欠拟合 |
| 中（如 128）| 官方常用默认 | 平衡点 |
| 大（如 256+）| 前缀长，表达力强 | 显存上升，可能过拟合小数据集 |

> 学习率注意：P-Tuning v2 用 **`LR=2e-2`**（见 `train.sh`），比全量微调（`1e-4`）大两个数量级。因为只训一小撮新初始化的参数，需要更大步长才学得动。

## 4. LoRA（折中路线）

> 本目录脚本以 P-Tuning v2 / 全量为主；LoRA 在 ChatGLM 上同样常用，原理详见 [[llm-train/peft/LoRA-QLoRA]]，这里给定位。

核心思想：冻结原权重 $W_0$，旁挂一个低秩增量

$$h = W_0 x + \Delta W x = W_0 x + B A x,\qquad A\in\mathbb{R}^{r\times d},\ B\in\mathbb{R}^{d\times r},\ r\ll d$$

只训 $A,B$。秩 $r$（如 8/16/64）控制容量与显存。

```
          x
          │
   ┌──────┴───────┐
   ▼              ▼
W0(冻结)      A(r×d) ─► B(d×r)   ← 只训这条低秩旁路
   │              │
   └──────┬───────┘ + （缩放 α/r）
          ▼
          h
```

LoRA vs P-Tuning v2：
| 维度 | P-Tuning v2 | LoRA |
|------|-------------|------|
| 改什么 | K/V 前缀（不动权重） | 权重的低秩增量 |
| 占用序列长度 | 是（前缀挤占 KV 上下文） | 否 |
| 推理可合并 | 否，需保留 prefix_encoder | 可把 $BA$ 合并回 $W_0$，零额外开销 |
| 多任务 | 切前缀 | 切/叠适配器 |

## 5. 全量微调 + DeepSpeed ZeRO

### 5.1 为什么必须 DeepSpeed
第 1.2 节算过：6B 全量约 96GB 优化器+权重，单卡放不下。**ZeRO（Zero Redundancy Optimizer）** 把「优化器状态 / 梯度 / 参数」按 GPU 数切分，不再每卡存一份完整副本。

```
普通 DP：  GPU0[全参+全梯度+全优化器]  GPU1[全参+全梯度+全优化器] … 每卡都冗余
ZeRO-2 ：  优化器状态&梯度 分片到各卡，参数仍各持一份
           ┌GPU0: 参数 | 梯度分片0 | 优化器分片0┐
           ├GPU1: 参数 | 梯度分片1 | 优化器分片1┤  → 显存随卡数近似线性下降
           └GPU7: 参数 | 梯度分片7 | 优化器分片7┘
ZeRO-3 ：  连参数也分片，最省显存，通信更重
```

本目录 `deepspeed.json` 用的是 **`"stage": 2`**（ZeRO-2）+ FP16。`ds_train_finetune.sh` 里 `--num_gpus=8` 表示 8 卡数据并行 + ZeRO-2 分片。

### 5.2 deepspeed.json 关键项含义
| 配置项 | 含义 | 权衡 |
|--------|------|------|
| `zero_optimization.stage: 2` | 切分优化器状态+梯度 | 显存↓；3 更省但通信更多 |
| `fp16.enabled: "auto"` | 混合精度由 Trainer 决定 | 提速省显存；数值需 loss_scale 防溢出 |
| `train_micro_batch_size_per_gpu: "auto"` | 由命令行 `--per_device_train_batch_size` 决定 | 与脚本对齐，避免冲突 |
| `allgather/reduce_bucket_size: 5e8` | 通信分桶大小 | 大桶吞吐高但峰值显存高 |
| `overlap_comm: false` | 通信与计算是否重叠 | true 更快但更吃显存 |

## 6. 数据格式

本目录脚本用的是 **AdvertiseGen**（广告文案生成）数据集，结构是「商品标签 → 文案」。脚本里通过 `--prompt_column content --response_column summary` 指定两列：

```json
{"content": "类型#上衣*版型#宽松*颜色#黑色*风格#简约", "summary": "这件黑色简约宽松上衣，百搭又不挑身材……"}
```

每行一个 JSON 对象（JSON Lines）。训练时 `content` 是输入、`summary` 是要学的目标输出。

ASCII 看清一条样本怎么变成 loss：

```
{"content": X, "summary": Y}
        │
   tokenizer 拼成对话格式：[prompt(X)] [gMASK][sop] [target(Y)] [eos]
        │            截断：max_source_length=64 / max_target_length=64
        ▼
   只在 Y 部分算交叉熵（X 部分 label 置 -100 被忽略）
        ▼
   loss = -Σ log p(y_t | y_<t, X)
```

> 换成你自己的任务：把数据整理成 `{prompt列, response列}` 的 JSON Lines，再用 `--prompt_column / --response_column` 指到你的列名即可。**多轮对话**则需把历史拼进 prompt（ChatGLM 推理侧用 `history` 列表，训练侧需自行展开成单条输入-输出）。

## 7. 显存逐项估算（6B）

| 场景 | 谁在显存里 | 粗略量级（单卡） |
|------|-----------|------------------|
| INT4 推理 | 量化权重 | ~6 GB |
| FP16 推理 | 权重 | ~13 GB |
| P-Tuning v2 训练 | 冻结权重(FP16) + 小 prefix 的梯度/优化器 + 激活 | 单卡可行（视 batch/序列长） |
| LoRA 训练 | 冻结权重 + 低秩 $A,B$ 的梯度/优化器 + 激活 | 单卡可行（视 r） |
| 全量训练（无 ZeRO） | 权重+梯度+Adam 状态 | ~96 GB（放不下） |
| 全量训练（ZeRO-2 ×8） | 上者按卡分片 | 每卡显著下降，多卡可行 |

省显存的三个旋钮（任何路线都管用）：**降 `per_device_train_batch_size`、升 `gradient_accumulation_steps`（等效大 batch 又不涨峰值显存）、开 `--fp16` / 梯度检查点**。脚本里 v2 用 `batch=128, accum=16`，全量用 `batch=30, accum=2`，就是在峰值显存和等效 batch 之间各自找的平衡。

## 8. 典型流程（训练循环 + 调用链）

```
① 准备  下载 chatglm-6b 权重 + AdvertiseGen 数据
            │
② 选路线 ├─ P-Tuning v2 → bash train.sh            （单卡, pre_seq_len=128, LR=2e-2）
         ├─ 多卡 v2     → bash train_ptuningv2_dp.sh（deepspeed --include localhost:1,2,3）
         └─ 全量微调    → bash ds_train_finetune.sh （deepspeed --num_gpus=8 + ZeRO-2）
            │
③ 训练循环（main.py → Seq2SeqTrainer）
     load_dataset(JSONL) ─► preprocess(分词/截断/label屏蔽prompt)
              │
              ▼  for each step:
        前向 → loss(只在target上) → 反向 → (PEFT只更新prefix/LoRA) → optimizer.step
              │  每 logging_steps 打日志；每 save_steps 存 checkpoint
              ▼
④ 产物  output_dir/  ← v2 存的是 prefix_encoder 权重；全量存的是完整模型
            │
⑤ 评测  evaluate.sh / evaluate_finetune.sh（--do_predict + ROUGE/BLEU，main.py 里用 rouge_chinese/jieba）
            │
⑥ 部署  inference.py：加载权重 → model.chat() 多轮问答
```

> 训练循环的精髓：**前向只对 `summary`（target）部分计算交叉熵**，prompt 部分的 label 被置 `-100` 忽略，所以模型学的是「给定 content 该生成什么 summary」，而不是去复述 content。

## 9. 部署与推理

`inference.py` 展示了两种加载方式（对应两条训练路线）：

**A. 全量微调产物** —— 直接当普通模型加载：
```python
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model  = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
model  = model.half().cuda().eval()
```

**B. P-Tuning v2 产物** —— 主干 + 单独加载 prefix（脚本里被注释，按需启用）：
```python
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, pre_seq_len=128)
model  = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
# 只挑出 transformer.prefix_encoder.* 这部分参数灌进去
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
```

```
关键区别：
全量 → 一个完整模型文件，直接 from_pretrained
v2   → 原始 6B 主干（不变） + 一小份 prefix_encoder 权重（你训出来的）
        加载时必须传 pre_seq_len，否则结构对不上、前缀挂不进去
```

部署时的两个旋钮：`model.quantize(4)`（INT4 量化进一步压显存，约 6GB 可跑）、`model.half()`（FP16）。`model.chat(tokenizer, query, history)` 里 `history` 控制是否带多轮上下文。

## 配置示例：本目录脚本逐参数

`train.sh`（P-Tuning v2，单卡）核心参数：
| 参数 | 值 | 含义 |
|------|----|------|
| `--pre_seq_len` | 128 | 前缀长度，决定可训参数量与表达力 |
| `--learning_rate` | 2e-2 | 大 LR（只训新参数） |
| `--per_device_train_batch_size` | 128 | 单卡 batch |
| `--gradient_accumulation_steps` | 16 | 等效 batch=128×16 |
| `--max_source_length / target` | 64 / 64 | 输入/输出截断长度 |
| `--prompt_column / --response_column` | content / summary | 指定数据两列 |
| `--predict_with_generate` | — | 评测时真正 generate 算 ROUGE/BLEU |

`ds_train_finetune.sh`（全量）差异：去掉 `--pre_seq_len`、`LR=1e-4`（小 LR）、加 `--deepspeed deepspeed.json` 与 `--fp16`，`deepspeed --num_gpus=8` 多卡。

## 常见问题

| 问题 | 原因 / 排查 |
|------|-------------|
| P-Tuning v2 + DeepSpeed 数据并行 **loss 不一致** | 已知 issue（见下方链接）；DP 下 prefix 的梯度同步/loss 聚合需注意，跟进官方修复，**以官方源码为准** |
| OOM 显存爆 | 降 `per_device_train_batch_size`、升 `gradient_accumulation_steps`、开 `--fp16` / 梯度检查点、用 ZeRO-2/3 |
| 全量微调放不下 | 必须 DeepSpeed ZeRO（`deepspeed.json stage:2`）+ 多卡；单卡只能走 PEFT |
| v2 推理加载报结构不匹配 | `from_pretrained` 必须传 `pre_seq_len`，再单独 load `prefix_encoder` 权重 |
| 微调后「复读 / 不听指令」 | 检查 label 是否屏蔽了 prompt（应只在 target 算 loss）、LR 是否过大、数据格式列名是否对 |
| 该选哪条路线 | 先 P-Tuning v2 验证可行性 → LoRA 折中（可合并/多任务）→ 有多卡再全量上限 |
| 自定义数据怎么接 | 整理成 `{content, summary}` JSON Lines，用 `--prompt_column/--response_column` 指列名 |

> ⚠️ 精确的 CLI 默认值、prefix_encoder 内部结构、ZeRO 通信细节请**以官方 `ChatGLM-6B/ptuning` 源码与 DeepSpeed 文档为准**；本文讲机制与思路，不替代源码。

## 🔗 跳转链接
- [[00-知识地图]] — 全局索引
- [[llm-algo/chatglm/README]] — ChatGLM 模型本身（GLM 结构 / 架构）
- [[llm-train/peft/Prefix-Tuning]] — P-Tuning v2 的底层机制（每层 K/V 前缀、重参数化、手算）
- [[llm-train/peft/LoRA-QLoRA]] — LoRA / QLoRA 原理与显存
- [[llm-train/peft/README]] — PEFT 路线全景
- ChatGLM-6B 源码：`https://github.com/THUDM/ChatGLM-6B`（commit `8633db1503fc3b0edc1d035f64aa35dce5d97969`）
- DP + P-Tuning v2 loss 不一致 issue：`https://github.com/THUDM/ChatGLM-6B/issues/644`
