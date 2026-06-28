# DeepSpeed-Chat 实操：三阶段脚本、数据切分与索引缓存

> DeepSpeed-Chat 的「落地视角」笔记：用真实 `run_*.sh` 把 SFT → Reward Model → PPO 三阶段串起来，并讲清 `--data_split 2,4,4` 与 `/tmp/data_files/` 里那一堆 `.npy / .pt` 缓存文件到底是什么、为什么会生成、怎么读懂。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/deepspeedchat/llama/README]]（DS-Chat 原理总览 / Hybrid Engine） · [[ai-framework/deepspeed/README]]（ZeRO / offload） · [[llm-alignment/RLHF]]（三阶段与 PPO） · [[llm-train/peft/PEFT-API]]（LoRA）

## 阅读地图

| 节 | 你会学到 | 关键词 |
|----|---------|--------|
| 0 | 一句话锚点 | 三阶段脚本 + 数据缓存 |
| 1 | 地基：DS-Chat 在 llm-action 里的目录结构 | training/step1-2-3、utils/data |
| 2 | 数据流水线：从 HF 数据集到 token 张量 | PromptRawDataset、四个数据集 |
| 3 | `--data_split 2,4,4` 到底切了什么 | 阶段间不重叠切分 |
| 4 | 读懂 `/tmp/data_files/` 里的 `.npy` 与 `.pt` | 索引缓存 vs token 缓存 |
| 5 | Step1 SFT 脚本逐行 | OPT-2.7b、LoRA、ZeRO-3 |
| 6 | Step2 Reward Model 脚本逐行 | OPT-350m、pairwise、ZeRO-0 |
| 7 | Step3 PPO 脚本逐行 | actor/critic 双 LR、Hybrid Engine、TP |
| 8 | 三脚本超参对照表 | 一眼看全 |
| — | 常见问题 + 跳转链接 | FAQ |

## 0. 一句话锚点

**这份笔记是 DS-Chat 的「手上沾泥」版**：[[llm-train/deepspeedchat/llama/README]] 讲「为什么」（Hybrid Engine、四模型、ZeRO 原理），这里讲「这台机器上实际跑了什么命令、磁盘上落了哪些文件」。

一句话串起来：

```
四个 HF 偏好数据集  ──(--data_split 2,4,4)──▶  阶段间不重叠的索引
        │                                            │
        ▼                                            ▼
  下载/解析(PromptRawDataset)              落盘成 .npy 索引缓存 + .pt token 缓存
        │                                            │ (放在 /tmp/data_files/)
        └────────────▶ step1 SFT ─▶ step2 RM ─▶ step3 PPO ◀┘ (二次运行直接读缓存)
```

## 1. 地基：本目录的真实结构

`llm-train/deepspeedchat/` 在 llm-action 仓库里的实际布局（已核对）：

```
deepspeedchat/
├── README.md                  # 本文件（实操视角）
├── llama/README.md            # DS-Chat 原理深度笔记（Hybrid Engine 等）
└── training/
    ├── step1_supervised_finetuning/
    │   └── training_scripts/single_node/run_13b.sh   # 实为 OPT-2.7b SFT
    ├── step2_reward_model_finetuning/
    │   └── training_scripts/single_node/run_350m.sh  # OPT-350m 奖励模型
    ├── step3_rlhf_finetuning/
    │   └── training_scripts/single_node/run_13b.sh   # PPO + Hybrid Engine
    └── utils/
        └── data/raw_datasets.py   # 各 HF 数据集的统一封装类
```

> ⚠️ 命名坑：step1/step3 的脚本叫 `run_13b.sh`，但脚本内 `--model_name_or_path` 指向的是 `hf-opt-2.7b`。文件名是 DS-Chat 官方模板沿用下来的，**以脚本内的 `--model_name_or_path` 为准**，不要被文件名误导。

三个 step 互相喂数据，形成一条流水线：

```
 step1 SFT ──产出 actor 初始权重──┐
 step2 RM  ──产出 reward 权重 ────┤
                                  ▼
 step3 PPO  ◀── 同时吃 actor(step1) 和 critic/reward(step2)
```

## 2. 数据流水线：四个数据集与统一封装

三个脚本的 `--data_path` 用的是同一组 **四个 HuggingFace 偏好数据集**（step1/step2 全用，step3 仅用第一个）：

```
Dahoas/rm-static
Dahoas/full-hh-rlhf
Dahoas/synthetic-instruct-gptj-pairwise
yitingxie/rlhf-reward-datasets
```

每个数据集在 `utils/data/raw_datasets.py` 里被包成一个 `PromptRawDataset` 子类（如 `DahoasRmstaticDataset`、`DahoasFullhhrlhfDataset`、`YitingxieRlhfrewarddatasetsDataset`），对外暴露统一 API：

```python
class PromptRawDataset(object):
    def get_train_data(self): ...
    def get_eval_data(self):  ...
    # prompt 统一格式: " Human: " + 实际问题 + " Assistant:"
    def get_prompt(self, sample): ...
    # chosen/rejected 统一格式: " " + 实际回答
    def get_chosen(self, sample): ...
    def get_rejected(self, sample): ...   # 没有 rejected 的数据集返回 None
    def get_prompt_and_chosen(self, sample): ...
    def get_prompt_and_rejected(self, sample): ...
```

**为什么要这层封装**：不同 HF 数据集字段名五花八门（有的叫 `prompt/chosen/rejected`，有的字段拼在一起）。统一成「prompt / chosen / rejected」三元组后，上层的 SFT / RM / PPO 代码就不必关心数据来源——这是把「数据异构」隔离在最底层的经典做法。

**prompt 模板很关键**：注释里写死了 `" Human: " + 问题 + " Assistant:"`。这个对话模板必须在三阶段保持一致，否则推理时模型见到的格式和训练时不同，效果会塌。

## 3. `--data_split 2,4,4` 到底切了什么

三个脚本都带 `--data_split 2,4,4`。这是 DS-Chat 的核心数据切分参数，含义是把**每个数据集**按比例 `2 : 4 : 4` 切成三份，分别供三个阶段使用：

```
一个数据集的训练样本(100%)
┌────────┬────────────────┬────────────────┐
│  20%   │      40%       │      40%        │
│ step1  │     step2      │     step3       │
│  SFT   │  Reward Model  │   PPO(RLHF)     │
└────────┴────────────────┴────────────────┘
   2    :        4        :        4
```

**为什么要阶段间不重叠切分**：如果三个阶段都用同一批数据，会有「信息泄漏」——奖励模型见过的样本，PPO 又拿去采样优化，等于让模型「背答案」，评估虚高。`2,4,4` 让每个阶段看到**互不相交**的子集，更接近真实泛化。比例 `2:4:4` 反映了「SFT 需要的数据相对少，RM 和 PPO 需要更多」的经验。

数值示例：某数据集训练集有 76.3k 条样本，则
- step1 拿前 $76300 \times 0.2 \approx 15260$ 条；
- step2 拿中间 $76300 \times 0.4 \approx 30520$ 条；
- step3 拿最后 $30520$ 条。

切分由固定 `--seed 1234` 决定，所以可复现——这也是缓存文件名里带 `seed1234` 的原因。

## 4. 读懂 `/tmp/data_files/`：索引缓存 vs token 缓存（原文真料）

DS-Chat 第一次跑会把「切分结果」和「tokenize 结果」落盘到一个数据目录（这里是 `/tmp/data_files/`），**第二次运行直接读缓存，跳过下载、切分、tokenize**，省下大量预处理时间。下面是该目录的真实 `ls` 输出（保留原文）：

```
> ls -al --block-size=K /tmp/data_files/
total 6533712K
drwxrwxr-x    2 guodong.li guodong.li       4K Jul  6 16:02 .
drwxrwxrwt. 328 root       root            24K Jul  7 10:00 ..
-rw-rw-r--    1 guodong.li guodong.li      10K Jul  6 10:26 Dahoas_full_hh_rlhf_seed1234_eval_2,4,4_0.npy
-rw-rw-r--    1 guodong.li guodong.li      20K Jul  6 10:26 Dahoas_full_hh_rlhf_seed1234_eval_2,4,4_1.npy
-rw-rw-r--    1 guodong.li guodong.li      20K Jul  6 10:26 Dahoas_full_hh_rlhf_seed1234_eval_2,4,4_2.npy
-rw-rw-r--    1 guodong.li guodong.li      88K Jul  6 10:25 Dahoas_full_hh_rlhf_seed1234_train_2,4,4_0.npy
-rw-rw-r--    1 guodong.li guodong.li     176K Jul  6 10:25 Dahoas_full_hh_rlhf_seed1234_train_2,4,4_1.npy
-rw-rw-r--    1 guodong.li guodong.li     176K Jul  6 10:25 Dahoas_full_hh_rlhf_seed1234_train_2,4,4_2.npy
-rw-rw-r--    1 guodong.li guodong.li       5K Jul  6 10:24 Dahoas_rm_static_seed1234_eval_2,4,4_0.npy
-rw-rw-r--    1 guodong.li guodong.li       9K Jul  6 10:24 Dahoas_rm_static_seed1234_eval_2,4,4_1.npy
-rw-rw-r--    1 guodong.li guodong.li       9K Jul  6 10:24 Dahoas_rm_static_seed1234_eval_2,4,4_2.npy
-rw-rw-r--    1 guodong.li guodong.li      60K Jul  6 10:23 Dahoas_rm_static_seed1234_train_2,4,4_0.npy
-rw-rw-r--    1 guodong.li guodong.li     120K Jul  6 10:23 Dahoas_rm_static_seed1234_train_2,4,4_1.npy
-rw-rw-r--    1 guodong.li guodong.li     120K Jul  6 10:23 Dahoas_rm_static_seed1234_train_2,4,4_2.npy
-rw-rw-r--    1 guodong.li guodong.li       3K Jul  6 10:28 Dahoas_synthetic_instruct_gptj_pairwise_seed1234_eval_2,4,4_0.npy
-rw-rw-r--    1 guodong.li guodong.li       6K Jul  6 10:28 Dahoas_synthetic_instruct_gptj_pairwise_seed1234_eval_2,4,4_1.npy
-rw-rw-r--    1 guodong.li guodong.li       6K Jul  6 10:28 Dahoas_synthetic_instruct_gptj_pairwise_seed1234_eval_2,4,4_2.npy
-rw-rw-r--    1 guodong.li guodong.li      24K Jul  6 10:28 Dahoas_synthetic_instruct_gptj_pairwise_seed1234_train_2,4,4_0.npy
-rw-rw-r--    1 guodong.li guodong.li      47K Jul  6 10:28 Dahoas_synthetic_instruct_gptj_pairwise_seed1234_train_2,4,4_1.npy
-rw-rw-r--    1 guodong.li guodong.li      47K Jul  6 10:28 Dahoas_synthetic_instruct_gptj_pairwise_seed1234_train_2,4,4_2.npy
-rw-rw-r--    1 guodong.li guodong.li     117K Jul  6 10:28 Dahoas_synthetic_instruct_gptj_pairwise_seed1234_train_eval_9,1_0.npy
-rw-rw-r--    1 guodong.li guodong.li      14K Jul  6 10:28 Dahoas_synthetic_instruct_gptj_pairwise_seed1234_train_eval_9,1_1.npy
-rw-rw-r--    1 guodong.li guodong.li  508630K Jul  6 10:31 evaldata_1cddaf9eb98d6e8394f4c53fd5ab8c97c5db3cce6e2699bc8e607027a642af35.pt
-rw-rw-r--    1 guodong.li guodong.li   15778K Jul  6 16:02 evaldata_d32e7f6108f0e9ba9b6d2642029fa6f57d50b7bc812d3ce2436afab7714d181d.pt
-rw-rw-r--    1 guodong.li guodong.li 5775824K Jul  6 10:31 traindata_1cddaf9eb98d6e8394f4c53fd5ab8c97c5db3cce6e2699bc8e607027a642af35.pt
-rw-rw-r--    1 guodong.li guodong.li  232010K Jul  6 16:02 traindata_d32e7f6108f0e9ba9b6d2642029fa6f57d50b7bc812d3ce2436afab7714d181d.pt
-rw-rw-r--    1 guodong.li guodong.li       5K Jul  6 10:30 yitingxie_rlhf_reward_datasets_seed1234_eval_2,4,4_0.npy
-rw-rw-r--    1 guodong.li guodong.li       9K Jul  6 10:30 yitingxie_rlhf_reward_datasets_seed1234_eval_2,4,4_1.npy
-rw-rw-r--    1 guodong.li guodong.li       9K Jul  6 10:30 yitingxie_rlhf_reward_datasets_seed1234_eval_2,4,4_2.npy
-rw-rw-r--    1 guodong.li guodong.li      60K Jul  6 10:29 yitingxie_rlhf_reward_datasets_seed1234_train_2,4,4_0.npy
-rw-rw-r--    1 guodong.li guodong.li     120K Jul  6 10:29 yitingxie_rlhf_reward_datasets_seed1234_train_2,4,4_1.npy
-rw-rw-r--    1 guodong.li guodong.li     120K Jul  6 10:29 yitingxie_rlhf_reward_datasets_seed1234_train_2,4,4_2.npy
```

### 4.1 文件名解码（`.npy`：索引缓存）

`.npy` 是 NumPy 数组，存的不是文本，而是**样本下标（index）**——告诉训练代码「这个阶段该用原数据集的哪些行」。命名规则拆开看：

```
Dahoas_full_hh_rlhf _ seed1234 _ train _ 2,4,4 _ 1 . npy
└──── 数据集名 ────┘   └ 切分种子 ┘ └split┘ └切比┘ └阶段┘
```

| 字段 | 取值 | 含义 |
|------|------|------|
| 数据集名 | `Dahoas_full_hh_rlhf` 等 | 对应 `raw_datasets.py` 里的 `dataset_name_clean`（`/`→`_`） |
| `seed1234` | 来自 `--seed 1234` | 决定随机切分，保证可复现 |
| `train` / `eval` | 训练集 / 验证集 | 各自再切三段 |
| `2,4,4` | 来自 `--data_split` | 该缓存对应的切分比例 |
| 末尾 `_0/_1/_2` | 阶段索引 | `0`=step1(SFT)、`1`=step2(RM)、`2`=step3(PPO) |

对照第 3 节：每个数据集的每个 `train`/`eval` 都生成 `_0/_1/_2` 三个 `.npy`，恰好就是「2,4,4」三段的下标集合。size 也吻合——`train_..._0`(SFT,占 20%) 比 `train_..._1`/`_2`(各 40%) 小一半左右，例如 `full_hh_rlhf` 是 `88K : 176K : 176K`，正好 `2 : 4 : 4`。

> 多出来的 `..._train_eval_9,1_*.npy`（只在 synthetic 数据集出现）：是某个数据集本身**没有现成验证集**，于是按 `9:1` 从训练集再切出一小块当 eval（`_0`=训练 90%、`_1`=验证 10%）。

### 4.2 `.pt`：tokenize 后的 token 缓存

`traindata_<hash>.pt` / `evaldata_<hash>.pt` 是 PyTorch 序列化文件，存的是**已经 tokenize、padding 到 `max_seq_len` 的 token 张量**（真正喂进模型的东西）。

- 文件名里的 `<hash>` 是「数据集组合 + 切分 + tokenizer + max_seq_len」等参数的指纹。**任一参数变了，hash 变，就会重新生成**——这是缓存命中/失效的判据。
- 体积差异巨大：`traindata_1cdd...pt = 5775824K ≈ 5.5 GB`，而 `traindata_d32e...pt = 232010K ≈ 226 MB`。同为「train」缓存，前者多半对应「四数据集合并、长序列」的大组合，后者对应小组合（如 step3 只用 `Dahoas/rm-static`）。两组 hash 共存，说明这台机器跑过两种不同的数据配置。

**一句话区分两类文件**：

```
.npy  = 「用哪些样本」的索引（轻量，KB 级，人可按名解码）
.pt   = 「样本变成的 token 张量」（重量，MB~GB 级，按参数 hash）
```

> 💡 调试经验：改了数据组合/`max_seq_len` 却发现结果没变，多半是命中了旧 `.pt` 缓存。清掉数据目录（这里 `/tmp/data_files/`）即可强制重建。同理，磁盘紧张时这些 GB 级 `.pt` 是首要清理对象。

## 5. Step1：监督微调（SFT）脚本逐行

文件：`training/step1_supervised_finetuning/training_scripts/single_node/run_13b.sh`（实跑 OPT-2.7b）。原样保留：

```bash
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then OUTPUT=./output; fi
if [ "$ZERO_STAGE" == "" ]; then ZERO_STAGE=3; fi   # 默认 ZeRO-3
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path /home/guodong.li/model/hf-opt-2.7b \
   --per_device_train_batch_size 128 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 6  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
```

关键参数的「为什么」：

| 参数 | 值 | 解释 / 为什么 |
|------|----|--------------|
| `--data_split 2,4,4` | — | SFT 只吃第一段（见第 3 节），生成 `..._train_..._0.npy` |
| `--max_seq_len 512` | 512 | prompt+response 截断长度；越大显存越吃，且影响 `.pt` 的 hash |
| `--per_device_train_batch_size 128` × `--gradient_accumulation_steps 8` | 有效 batch=1024/卡 | 用梯度累积撑大有效 batch 而不爆显存 |
| `--learning_rate 1e-4` | — | 比预训练大、是因为配了 LoRA（只训低秩增量，可用更大 LR） |
| `--gradient_checkpointing` | 开 | 用算力换显存：反向时重算激活，省激活显存 |
| `--zero_stage 3` | 3 | 参数+梯度+优化器全切片，最省显存（见 [[ai-framework/deepspeed/README]]） |
| `--lora_dim 128` + `--lora_module_name decoder.layers.` | — | 在 OPT 的 `decoder.layers.*` 上挂秩 128 的 LoRA，大幅省显存（见 [[llm-train/peft/PEFT-API]]） |
| `&> $OUTPUT/training.log` | — | stdout+stderr 都重定向到日志，便于事后排查 |

> 用法：`bash run_13b.sh ./step1_output 3`（第一参数=输出目录，第二参数=ZeRO 级别）。

## 6. Step2：奖励模型脚本逐行

文件：`training/step2_reward_model_finetuning/training_scripts/single_node/run_350m.sh`（OPT-350m）。原样保留：

```bash
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then OUTPUT=./output; fi
if [ "$ZERO_STAGE" == "" ]; then ZERO_STAGE=0; fi   # 默认 ZeRO-0
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path /home/guodong.li/model/hf-opt-350m \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --disable_dropout \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
```

与 step1 的差异点（都有原因）：

| 差异 | step2 取值 | 为什么 |
|------|-----------|--------|
| 模型 | OPT-**350m**（比 SFT 小） | RM 只需「打分」能力，小模型够用，省资源（见 llama 笔记第 4 节） |
| 默认 `zero_stage` | **0**（不分片） | 350m 很小，单卡放得下，关掉分片省通信、更快 |
| `--num_padding_at_beginning 1` | 1 | OPT 这类模型句首有特殊 token；打分时要跳过开头 padding，避免把 pad 当有效 token |
| `--disable_dropout` | 开 | RM 要给出稳定一致的分数，dropout 引入随机性会污染 pairwise 比较 |
| `--num_train_epochs 1` | 1 | RM 极易过拟合偏好数据，跑 1 epoch 即止 |
| 无 LoRA | — | 模型本就小，直接全参微调 |

RM 用 **pairwise ranking loss**（chosen 比 rejected 得分高），数学细节见 [[llm-train/deepspeedchat/llama/README]] 第 4 节：

$$\mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\theta(x, y_c) - r_\theta(x, y_r)\big)$$

> 用法：`bash run_350m.sh ./step2_output 0`。

## 7. Step3：PPO（RLHF）脚本逐行 ★ Hybrid Engine 在此

文件：`training/step3_rlhf_finetuning/training_scripts/single_node/run_13b.sh`。原样保留：

```bash
ACTOR_MODEL_PATH=$1        # 来自 step1 的 SFT 产物
CRITIC_MODEL_PATH=$2       # 来自 step2 的 RM 产物
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
[ -z "$OUTPUT" ] && OUTPUT=./output
[ -z "$ACTOR_ZERO_STAGE" ]  && ACTOR_ZERO_STAGE=3
[ -z "$CRITIC_ZERO_STAGE" ] && CRITIC_ZERO_STAGE=3
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1   # this is model related
Actor_Lr=5e-4
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 32 \
   --per_device_mini_train_batch_size 16 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 2 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --enable_hybrid_engine \
   --inference_tp_size 2 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --actor_lora_dim 128 \
   --actor_lora_module_name decoder.layers. \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
```

step3 的「PPO 一次迭代」数据流（采集→学习），配合脚本参数看：

```
┌── (a) 经验采集 rollout：纯推理 ──────────────────────┐
│ prompt(≤256) ─▶ Actor.generate() ─▶ response(≤256)  │  ← max_prompt/answer_seq_len
│ 走 Hybrid Engine: --enable_hybrid_engine             │  ← 聚合参数+TP(--inference_tp_size 2)+KV-Cache
│ (prompt+resp) ─▶ Ref/Reward/Critic 各前向一遍         │
└──────────────────────────────────────────────────────┘
                  │ 攒成 batch
                  ▼
┌── (b) 学习：训练 ───────────────────────────────────┐
│ KL 罚 → 优势 GAE → Actor(PPO clip) + Critic(MSE)     │
│ 切回 ZeRO 训练态做反向 (actor/critic_zero_stage 3)    │
└──────────────────────────────────────────────────────┘
```

逐参数解读：

| 参数 | 值 | 为什么 |
|------|----|--------|
| `--data_path Dahoas/rm-static` | 只用 1 个数据集 | PPO 只需 prompt（自己生成 response），数据需求比 SFT/RM 小 |
| `--actor_model_name_or_path` | step1 产物 | Actor=被训策略，用 SFT 初始化；Ref 是它的冻结副本 |
| `--critic_model_name_or_path` | step2 产物 | Critic=价值网络，用 RM 初始化（结构相近）；Reward 也来自这里 |
| `Actor_Lr=5e-4` / `Critic_Lr=5e-6` | actor 比 critic **大 100 倍** | 注意：这里 actor LR 反而更大，因为 actor 挂了 LoRA（小增量需大步长）、critic 全参（需小步长稳收敛）。两者解耦设置是 PPO 稳定的关键 |
| `--per_device_train_batch_size 32` / `--per_device_mini_train_batch_size 16` | 32 / 16 | 采集 batch 32，PPO 更新切成 mini-batch 16 多次走，平衡显存与样本利用 |
| `--ppo_epochs 1` | 1 | 每批经验重复优化几次；1 较保守，防策略偏离过远 |
| `--max_answer_seq_len 256` | 256 | 生成长度——**直接决定 rollout 耗时**，是 PPO 慢的主因（见 llama 笔记第 5 节） |
| `--enable_hybrid_engine` ★ | 开 | DS-Chat 核心：生成态走推理优化，端到端提速（原理见 llama 笔记第 7 节） |
| `--inference_tp_size 2` | 2 | 推理态张量并行宽度=2，单卡省显存换通信（见 [[B07:llm-inference/大模型推理张量并行]]） |
| `--actor_gradient_checkpointing` + `--disable_actor_dropout` | 开 | 省激活显存 / 稳定策略输出 |
| `--actor_lora_dim 128` | LoRA | 只训 actor 的低秩增量，大幅省显存（呼应 Actor_Lr 偏大） |
| `--master_port 12346` | — | 指定分布式通信端口，避免多任务端口冲突 |

> 用法：`bash run_13b.sh <step1输出路径> <step2输出路径> 3 3 ./step3_output`。**必须先跑完 step1、step2 拿到两个产物路径**，否则 step3 无从初始化 actor/critic。

## 8. 三脚本超参对照表（一眼看全）

| 维度 | Step1 SFT | Step2 RM | Step3 PPO |
|------|-----------|----------|-----------|
| 模型 | OPT-2.7b | OPT-350m | actor=step1 / critic=step2 |
| 数据集数 | 4 个全用 | 4 个全用 | 仅 `Dahoas/rm-static` |
| `data_split` 取段 | 第 0 段(20%) | 第 1 段(40%) | 第 2 段(40%) |
| epochs | 6 | 1 | 1 |
| 学习率 | 1e-4 | 5e-5 | actor 5e-4 / critic 5e-6 |
| 默认 ZeRO | 3 | 0 | actor/critic 各 3 |
| LoRA | dim 128 | 无 | actor dim 128 |
| 损失 | causal LM | pairwise rank | PPO clip + critic MSE |
| Hybrid Engine | — | — | ✅ `--enable_hybrid_engine` |
| max_seq_len | 512 | 512 | prompt 256 + answer 256 |

> ⚠️ 数值口径声明：上表数字均来自仓库内 `run_*.sh` 脚本原文（如本机的 OPT-2.7b/350m 配置），是**这份 llm-action 教程作者的实验设置**，不是官方推荐默认值。换模型/换机器请以你的脚本为准，不要直接背。

## 常见问题

| 问题 | 解答 |
|------|------|
| `run_13b.sh` 文件名是 13b，怎么里面是 2.7b？ | 文件名沿用官方模板，**以脚本内 `--model_name_or_path` 为准**（实为 hf-opt-2.7b） |
| `/tmp/data_files/` 里 `.npy` 和 `.pt` 区别？ | `.npy`=样本下标索引(KB级,名字可解码)；`.pt`=tokenize 后的 token 张量(MB~GB级,按参数 hash) |
| `.pt` 文件名末尾那串 hash 是什么？ | 数据组合+切分+tokenizer+`max_seq_len` 的指纹；任一参数变则 hash 变、缓存重建 |
| 改了 `max_seq_len`/数据集却没生效？ | 多半命中旧 `.pt` 缓存；清空数据目录强制重建 |
| `2,4,4` 为什么要不重叠切？ | 防阶段间信息泄漏（RM 见过的样本不该再喂 PPO），更接近真实泛化 |
| 为什么 step3 actor 学习率(5e-4)反而比 critic(5e-6)大？ | actor 挂 LoRA(小增量需大步长)，critic 全参(需小步长稳收敛)；两者解耦是 PPO 稳定关键 |
| step3 报错找不到 actor/critic 路径？ | 必须先跑完 step1、step2，把两个 `output_dir` 作为 step3 前两个参数传入 |
| `--num_padding_at_beginning 1` 干嘛的？ | OPT 句首有特殊 token；打分/计算时跳过开头 padding，避免把 pad 当有效 token |
| 磁盘被 `/tmp/data_files/` 占满？ | GB 级的 `traindata_*.pt`/`evaldata_*.pt` 是首要清理对象，删后下次运行会重建 |
| Hybrid Engine 原理在哪看？ | 见 [[llm-train/deepspeedchat/llama/README]] 第 7 节（训练态↔推理态瞬切） |

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航枢纽
- [[llm-train/README]] — 大模型训练总览
- [[llm-train/deepspeedchat/llama/README]] — DS-Chat 原理深度笔记（Hybrid Engine / 四模型 / ZeRO）
- [[ai-framework/deepspeed/README]] — ZeRO 分级 / offload / DeepSpeed 引擎机制
- [[ai-framework/megatron-lm/README]] · [[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]] — 张量/流水并行训练栈
- [[llm-train/pytorch/distribution/README]] — PyTorch 分布式基础
- [[ai-framework/pytorch/README]] · [[ai-framework/huggingface-peft/README]] — 框架与 PEFT 生态
- [[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]] — LoRA 与高效微调
- [[llm-alignment/RLHF]] — RLHF 三阶段与 PPO 原理详解
- [[llm-algo/transformer/模型架构]] — Transformer / OPT 架构基础
- [[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]] — ZeRO/TP 背后的 all-gather/all-reduce 通信
- [[B07:llm-inference/大模型推理张量并行]] — Hybrid Engine 推理态的 TP 来由
- [[llm-compression/quantization/量化基础]] — 进一步省显存的正交手段
- 官方源码：`microsoft/DeepSpeed` → `applications/DeepSpeed-Chat/`（精确参数以官方源码为准）
