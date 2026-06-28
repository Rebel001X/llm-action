# LLM-QAT 复现代码精读（cfd70ff：data-free 蒸馏 + W-A-KV 量化感知训练）

> 一句话定位：这是 Meta「LLM-QAT」论文的官方复现代码片段（一个 git commit 快照目录），核心是**用预训练模型自己生成的数据（data-free）做知识蒸馏**，并对 **权重 W / 激活 A / KV 缓存** 同时做**量化感知训练（QAT）**，把 LLaMA 压到 4-bit 仍保持精度。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/SmoothQuant]] · [[llm-compression/quantization/kv-cache-quant]] · [[llm-alignment/RLHF]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|-------------|--------|
| 0 | 一句话锚点：QAT vs PTQ，这个目录在干嘛 | data-free / W-A-KV |
| 1 | 地基：为什么 PTQ 到 4-bit 会崩、QAT 凭什么救 | 异常值 / 直通估计器 |
| 2 | 三段式流水线总览（生成→合并→训练→推理） | 6 个文件如何串起来 |
| 3 | 第一步：data-free 数据生成 `generate_data.py` | 单 token 起手 / 混合采样 |
| 4 | 第二步：合并数据 `merge_gen_data.py` | 8 chunk → all_gen.jsonl |
| 5 | 第三步：QAT 训练 `train.py` + `run_train.sh` | w/a/kv_bits / 师生蒸馏 / FSDP |
| 6 | 第四步：量化推理 `inference.py` | 假量化模型如何跑 |
| 7 | 关键公式：MinMax 线性量化 + STE + logits 蒸馏 | 手算 4-bit 步长 |
| 8 | 显存/通信账：FSDP + KD 双模型的代价 | 数值估算 |
| — | 评价 / 对照 / 局限 | 表格 |

## 0. 一句话锚点

- **PTQ（训练后量化）**：模型训练完，拿少量校准数据「一次性」把权重/激活折叠成低比特。便宜、快，但 ≤4-bit 时精度崩塌。
- **QAT（量化感知训练）**：训练时就**模拟量化误差**（前向「假量化」、反向用直通估计器让梯度照常流），让权重主动适应低比特。贵，但能救 4-bit。
- **LLM-QAT 的三大卖点**：
  1. **Data-free 蒸馏**——不碰原始训练语料（往往拿不到/有版权），让**预训练模型自己生成**几万条文本当蒸馏数据；
  2. **同时量化 W / A / KV**——KV 缓存量化对长序列吞吐至关重要，是本文相对前作的关键补充；
  3. **logits 蒸馏**——用全精度教师的软标签（完整 logit 分布）监督量化学生，比硬标签好得多。
- 本目录（`cfd70ff`，一个提交哈希命名的快照）= **复现脚本集**，不含 `models/`、`utils/datautils.py` 等核心模块（它们在仓库其他位置），但**入口、流程、超参全在这里**。

## 1. 地基：为什么 4-bit PTQ 会崩，QAT 凭什么救

### 1.1 问题背景

PTQ（RTN / GPTQ / SmoothQuant）在 **8-bit** 上几乎无损，但论文实测：一旦压到 **4-bit 权重 + 4-bit 激活（W4A4）**，PTQ 困惑度（PPL）爆炸——因为 LLM 激活里有**少量但极大的异常值（outlier）**，它们撑大了量化范围 $\alpha$，把绝大多数「正常值」挤进极少数量化格子，信息被抹平。

```
全精度激活分布（一维示意）：
   |·············▮▮▮▮▮▮▮▮·············|        ← 绝大多数值挤在中间
   |                              ▮  |        ← 极少数 outlier 在远端
   ↑min                          max↑

4-bit PTQ（16 个格子）把 [min,max] 均匀切：
   [0][1][2]...[14][15]
   ▮▮▮▮▮▮▮▮ 全落进 [7][8] 两格   → 中间值彼此无法区分 → 崩
```

### 1.2 QAT 的直觉

PTQ 是「先训练好再量化」，量化误差是**事后强加**的、模型从没见过；QAT 是「训练时就带着量化误差一起学」，让权重在反向传播中**主动挪位**去抵消量化损失。

关键技术叫**直通估计器（Straight-Through Estimator, STE）**：取整函数 $\lfloor\cdot\rceil$ 的导数几乎处处为 0，没法反传梯度。STE 直接「假装」量化是恒等函数，让梯度**绕过取整**原样流回去：

```
前向：  x ──► 量化(假量化, 有误差) ──► x_q ──► 后续计算
反向：  grad ◄── (STE: 当作恒等, 梯度直通) ◄────── grad
```

> 一句话：**前向带量化误差让模型「感知」低比特，反向用 STE 保证还能学。**

## 2. 三段式流水线总览（6 个文件如何串起来）

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段 A：data-free 数据生成（不需要任何真实语料！）              │
│                                                                   │
│   全精度 LLaMA-7B ──► generate_data.py（每张卡跑 i_start 分片）   │
│        │  ① 单 token 起手 (input_ids=[[i]], i∈[0,500))            │
│        │  ② 前几步贪心 do_sample=False → 锁定话题                 │
│        │  ③ 之后随机采样 do_sample=True → 多样性, 到 1024 token   │
│        ▼                                                           │
│   gen_data/gen.chunk.00.jsonl ... gen.chunk.07.jsonl              │
└───────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 B：合并   merge_gen_data.py                                 │
│   8 个 chunk ─► 拼成 gen_data/all_gen.jsonl（QAT 的蒸馏语料）     │
└───────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 C：QAT 训练   run_train.sh ─► train.py（torchrun 8 卡）     │
│   ┌──────────────┐  KD(可选)  ┌──────────────────────────┐       │
│   │ 教师: 全精度  │ ─logits──► │ 学生: 量化版(w/a/kv_bits)│       │
│   │ LLaMA(冻结)  │            │ LlamaForCausalLMQuant     │       │
│   └──────────────┘            └────────────┬─────────────┘       │
│        交叉熵 logits 蒸馏 + FSDP full_shard │ STE 反传            │
│                                             ▼                     │
│                              7B-finetuned（量化感知权重）         │
└───────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 D：推理验证   inference.py（加载量化模型, 跑一句 prompt）   │
└─────────────────────────────────────────────────────────────────┘
```

| 文件 | 角色 | 关键点 |
|------|------|--------|
| `generate_data.py` | 阶段 A，data-free 造数据 | 单 token 种子 + 混合采样 |
| `merge_gen_data.py` | 阶段 B，合并 8 分片 | → `all_gen.jsonl` |
| `train.py` | 阶段 C，QAT 主程序 | 量化模型 / KD / FSDP / 评估 PPL |
| `run_train.sh` | 阶段 C 启动脚本 | `bash run_train.sh 8 8 8` 即 W8A8KV8 |
| `utils.py` | 工具：日志 + 安全存盘 | 存盘时跳过 `teacher` 权重 |
| `inference.py` | 阶段 D，量化模型推理 | 加载 `7B-finetuned` |
| `pip.conf` | 国内 pip 源配置 | （环境用，非算法） |

## 3. 第一步：data-free 数据生成 `generate_data.py`（逐行拆解）

**核心思想**：不用真实语料，让教师模型「自言自语」造数据。但有两个工程巧思——

```python
n_vocab = 500            # 每张卡用前 500 个 token id 作为「种子」
i_start = sys.argv[1]    # 卡编号/分片号；8 卡并行 → i_start ∈ {0..7}

for j in range(3, 6):                          # 外层：贪心步数 3~5
  for i in range(i_start*500, (i_start+1)*500):# 内层：500 个种子 token
      input_ids = torch.tensor([[i]]).cuda()   # ① 单个 token 起手
      outputs1 = model.generate(input_ids, do_sample=False, max_length=j)  # ② 前 j 步贪心
      outputs  = model.generate(outputs1,      do_sample=True,  max_length=1024) # ③ 之后采样
      gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      # 写入 gen.chunk.{i_start}.jsonl
```

**三个「为什么」：**

1. **为什么单 token 起手？** 不给任何 prompt，纯靠 token id 触发，覆盖词表里各种话题，数据无偏、无版权。
2. **为什么前几步贪心、后面采样？**
   - 前 `j`（3~5）步 `do_sample=False`（贪心）：先用**确定性**的最高概率 token 把话题「锚定」住，避免开头就发散成乱码；
   - 之后 `do_sample=True`（随机采样）：引入**多样性**，让生成文本覆盖分布的尾部，更像真实语料。
   - 论文消融特别强调：**要从分布采样、不能永远 top-1**——否则蒸馏数据太单调，量化学生学不到完整分布。
3. **为什么 `range(3,6)` 三种贪心步数？** 同一种子用 3/4/5 步贪心各生成一条，等于一个种子产出**多条不同开头**的样本，扩充数据量与多样性。

**断点续跑**：开头检查 `gen.chunk.XX.jsonl` 是否已存在，算出已写多少行，从中断处继续（`inner_loop`/`outer_loop`）——生成 1024-token 序列很慢，必须可恢复。

```
单 token i ──贪心3~5步──► "锚定开头" ──随机采样到1024──► 一条样本
   500 个 i × 3 种步数 × 8 卡 ≈ 1.2 万条/轮，多轮累积成几万条蒸馏语料
```

## 4. 第二步：合并数据 `merge_gen_data.py`

逻辑极简：把 8 张卡各自产出的 `gen.chunk.00.jsonl … gen.chunk.07.jsonl` 逐行读出、拼成一个大文件 `gen_data/all_gen.jsonl`。这个文件就是 `run_train.sh` 里 `--train_data_local_path` 指向的蒸馏训练集。

```
gen.chunk.00.jsonl ┐
gen.chunk.01.jsonl ├─► all_gen.jsonl ─► 喂给 train.py 的 --train_data_local_path
   ...             │
gen.chunk.07.jsonl ┘
```
> 注意：用 `open(..., "a")` 追加写，重复执行会**追加重复数据**，复现时需先清空旧文件（以官方仓库为准）。

## 5. 第三步：QAT 训练 `train.py` + `run_train.sh`

### 5.1 启动脚本：`bash run_train.sh 8 8 8` 到底设了什么

```
bash run_train.sh  <w_bits> <a_bits> <kv_bits>
                      $1        $2       $3
       例：    8 8 8  → W8A8KV8（接近无损，PTQ 也能做）
               4 8 4  → W4A8KV4（论文推荐的最佳性价比点）
               4 4 4  → W4A4KV4（最激进，需配 SmoothQuant）
```

脚本本质是一条 `torchrun --nproc_per_node=8 train.py ...`，关键参数：

| 参数 | 值 | 含义 / 为什么 |
|------|----|--------------|
| `--qat True` | 开 QAT | 走量化模型分支 `LlamaForCausalLMQuant` |
| `--w_bits/a_bits/kv_bits` | `$1/$2/$3` | 三路比特位，注入 `student_config` |
| `--use_kd False` | 默认关蒸馏 | 设 True 才加载教师做 logits 蒸馏 |
| `--bf16 True / --fp16 False` | bf16 | 主计算精度（量化是「假量化」叠在 bf16 上）|
| `--learning_rate 2e-5` | 小 lr | QAT 是微调，不是从头训 |
| `--lr_scheduler_type cosine` | 余弦 | 配 `warmup_ratio 0.` |
| `--num_train_epochs 1` | 1 轮 | 蒸馏数据走一遍即可 |
| `--model_max_length 512` | 序列长 | 训练时截断长度 |
| `--gradient_checkpointing False` | 关 | 省显存可开（换算力）|
| `--fsdp "full_shard offload auto_wrap"` | FSDP | 全分片 + CPU offload，省显存 |
| `--fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer` | 按层包 | FSDP 以 decoder 层为分片单元 |

### 5.2 `train.py` 主流程（量化模型 + 师生蒸馏）

```
process_args() ─► model_args / data_args / training_args
        │
  qat?  ├─ True ─► LlamaConfig 注入 w_bits/a_bits/kv_bits/smoothquant
        │          ─► LlamaForCausalLMQuant.from_pretrained(student_config)  ← 学生(量化)
        │
  use_kd?├─ True ─► AutoModelForCausalLM(全精度) .eval(), requires_grad=False ← 教师(冻结)
        │          model.teacher = teacher_model;  KDTrainer
        │          └ False ─► 普通 Trainer
        ▼
  数据：wikitext-2 取前 1000 条做 train，test 做 valid（block_size=512/1024）
        ▼
  trainer.train() ─► STE 反传更新学生权重
        ▼
  safe_save_model_for_hf_trainer() ─► 存盘时跳过含 "teacher" 的 key（见 utils.py）
        ▼
  do_eval ─► perplexity = exp(eval_loss)   ← 用 PPL 衡量量化后语言建模质量
```

**两个易错点**：
1. **教师权重不能存盘**：`utils.py` 的 `safe_save_model_for_hf_trainer` 显式 `if "teacher" in key: continue`，否则 checkpoint 里混进一份全精度教师，体积翻倍、加载报错。
2. **`use_cache=False`**：训练时必须关 KV 缓存（`model.config.use_cache=False`），因为 QAT 要对**整段** KV 张量做量化（图 2 的「按 token 量化」在生成时才逐 token 存 scale）。

## 6. 第四步：量化推理 `inference.py`

加载训练好的 `7B-finetuned`（用 `LlamaForCausalLMQuant`，即带假量化的模型类），喂一句 prompt 生成。它验证的是「**量化感知权重 + 假量化前向**」能正常出文本——注意这仍是**模拟量化（simulated/fake quant）**，权重以 bf16 存储、前向时即时量化反量化，**不是**真正的 INT4 kernel 加速（论文结论里明说 4-bit 当时无开箱即用硬件支持）。

```
7B-finetuned ─load─► LlamaForCausalLMQuant(bf16, 假量化) ─generate─► 文本
                     （精度 = 真 4-bit，速度 ≠ 真 4-bit）
```

## 7. 关键公式 / 算法 / 数值示例

### 7.1 MinMax 线性（均匀）量化

论文采用**保留全值域、不裁剪**的对称 MinMax 量化（消融发现：对 LLaMA，保留异常值比裁剪更好）。$N$-bit 对称量化：

$$
\mathbf{X}_\mathbf{Q}^i = \alpha\,\Big\lfloor \frac{\mathbf{X}_\mathbf{R}^i}{\alpha}\Big\rceil,
\qquad
\alpha = \frac{\max(|\mathbf{X}_\mathbf{R}|)}{2^{N-1}-1}
$$

其中 $\mathbf{X}_\mathbf{R}$ 是实数值，$\alpha$ 是步长（scale），$\lfloor\cdot\rceil$ 是就近取整。非对称（带零点 $\beta$）形式：

$$
\mathbf{X}_\mathbf{Q}^i = \alpha\Big\lfloor \frac{\mathbf{X}_\mathbf{R}^i-\beta}{\alpha}\Big\rceil + \beta,
\qquad
\alpha = \frac{\max(\mathbf{X}_\mathbf{R})-\min(\mathbf{X}_\mathbf{R})}{2^{N}-1},\ \ \beta=\min(\mathbf{X}_\mathbf{R})
$$

- **权重**：per-channel（逐输出通道）对称量化；
- **激活 / KV**：per-token（逐 token）量化——每个 token 一组 scale，对付 token 间幅度差异大的问题（图 3）。

### 7.2 直通估计器（STE）

$$
\frac{\partial\,\lfloor x\rceil}{\partial x}\approx 1 \quad(\text{反向当作恒等})
$$
让量化误差进入前向 loss、又不阻断梯度。

### 7.3 logits 知识蒸馏损失

学生 $\mathcal{S}$ 用教师 $\mathcal{T}$ 的**完整软标签分布**做交叉熵监督（$c$ 遍历词表，$n$ 为样本数）：

$$
\mathcal{L}_{CE} = -\frac{1}{n}\sum_c\sum^n_{i=1} p_c^{\mathcal{T}}(X_i)\,\log\big(p_c^{\mathcal{S}}(X_i)\big)
$$

消融结论：**只用 logits 蒸馏最好**；额外加注意力蒸馏 / 隐藏层蒸馏反而掉点；只用硬标签（采样出的 next-token）次优（采样自带噪声）。

### 7.4 数值手算：4-bit 对称量化一个权重通道

设某权重通道实数范围 $[-0.8, +0.6]$，取 $N=4$：

```
max(|X|) = 0.8
α = 0.8 / (2^(4-1) − 1) = 0.8 / 7 ≈ 0.1143      ← 步长
量化级别（对称, 含 0）：{−7,…,−1,0,1,…,7}，共 15 个可用整数级

量化 w = 0.45：
   round(0.45 / 0.1143) = round(3.94) = 4
   反量化 ŵ = 4 × 0.1143 = 0.4571
   误差 = |0.45 − 0.4571| ≈ 0.0071   （约 1.6% 相对步长）

若该通道有个 outlier = 0.8 撑大了 α：
   普通值 0.05 → round(0.05/0.1143)=round(0.44)=0 → 反量化 0.0  ← 被抹成 0!
   这正是 PTQ 在低比特崩的原因；QAT 让权重在训练中「躲开」这种塌缩。
```

> 直觉：**bit 每降 1 位，可用级别数减半**（$2^{N-1}-1$：4-bit=7 级，3-bit=3 级），步长 $\alpha$ 翻倍，误差翻倍——所以 3-bit 以下极难，4-bit 是 QAT 的甜点。

## 8. 显存 / 通信账：FSDP + KD 双模型代价

### 8.1 为什么必须 FSDP `full_shard offload`

QAT 微调 7B 模型，且开 KD 时**同时驻留教师（全精度）+ 学生（量化感知）两份**：

```
朴素 DDP（每卡全量）显存粗算（7B, bf16）：
  学生权重   7B × 2 B = 14 GB
  学生梯度   7B × 2 B = 14 GB
  Adam 状态  7B × 8 B = 56 GB   (fp32 m,v + fp32 master)
  教师权重   7B × 2 B = 14 GB   (冻结, 无梯度/优化器)
  ─────────────────────────────
  合计 ≈ 98 GB / 卡  →  单张 80GB A100 装不下！

FSDP full_shard（8 卡均摊参数/梯度/优化器）：
  (14+14+56)/8 + 教师14 + 激活 ≈ 10.5 + 14 + 激活
  再叠 offload（优化器状态搬 CPU）→ 显存进一步降到可装下
```

### 8.2 FSDP 通信账（直觉）

- **前向**：每层用前 all-gather 把分片权重拼回整层 → 算完即丢；
- **反向**：再 all-gather 权重算梯度 → reduce-scatter 把梯度分回各卡。
- 每步通信量 ≈ $O(\text{参数量})$ 级别的 all-gather + reduce-scatter，用**显存换通信**。`auto_wrap` + `LlamaDecoderLayer` 让每个 decoder 层成为独立分片/通信单元，把一次大通信拆成多次小通信，与计算重叠。

```
[卡0..7] 每层: all-gather(W分片→整W) ─► 前向 ─► 丢弃整W
反向:      all-gather(整W) ─► 计算梯度 ─► reduce-scatter(梯度→分片)
            └─ 通信与下一层计算重叠 ─┘
```

## 评价 / 对照 / 局限

| 维度 | LLM-QAT 的做法 | 评价 |
|------|---------------|------|
| 数据 | data-free，教师自生成 | 摆脱原始语料/版权，可量化任意生成模型；但生成慢、质量依赖教师 |
| 量化对象 | W + A + **KV** 三路 | KV 量化是亮点，利好长序列吞吐 |
| 蒸馏 | 仅 logits 蒸馏 | 简单且最优；加 attn/hidden 反而掉点 |
| 量化函数 | 对称 MinMax，**不裁剪** | 保留 outlier；对含 GeLU 的模型不一定成立 |
| 兼容性 | 可叠 SmoothQuant | W4A4 时有效；W4A8 时叠了反而可能掉点 |
| 推荐配置 | **W4A8KV4** | 论文推荐的精度/效率甜点 |

**局限（论文自陈，数字见原文）：**
1. **4-bit 激活（A4）仍不够好**——W4A4 是上限，A4 需进一步研究；
2. 当时**无开箱即用的 4-bit 硬件 kernel**，本工作只做精度验证（fake quant），不含真加速实现；
3. 数据生成开销大（逐 token 生成 1024 长度 × 上万条）；
4. 对称量化结论绑定 LLaMA 的近对称权重/激活分布，**含 GeLU 的模型需重新评估**。

**和邻居方法的关系：**
- 对照 [[llm-compression/quantization/SmoothQuant]]：SmoothQuant 是 PTQ，把激活的「难」迁移一部分到权重；LLM-QAT 可**叠加**它（W4A4 时增益明显）。
- 对照 [[llm-compression/quantization/GPTQ]]：GPTQ 是逐层重建的 PTQ，无需反传；LLM-QAT 需端到端反传（STE），更贵但 4-bit 更稳。
- 对照 [[llm-compression/quantization/kv-cache-quant]]：本文首次把 QAT 思路用到 KV 缓存量化。

## 🔗 跳转链接

- 总览：[[00-知识地图]]
- 量化地基：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/大模型量化概述]]
- 同族方法：[[llm-compression/quantization/SmoothQuant]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/kv-cache-quant]] · [[llm-compression/quantization/fp8]]
- 训练与并行：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]]
- 推理侧 KV：[[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/vllm/README]]
- 对齐（论文展望：可用于指令精调/RL 后的模型）：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 性能名词：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
