# Vicuna

> Vicuna 是把 LLaMA 基座用 ShareGPT 上的「真人 × ChatGPT 多轮对话」做指令微调而成的开源聊天模型，并首创了「用 GPT-4 当裁判」的自动化评测范式。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/llama/模型架构]] [[llm-train/chinese-llama-alpaca/README]]

---

## 阅读地图

| 节 | 你将搞清楚 | 关键词 |
|----|-----------|--------|
| 0 | 一句话锚点 | LLaMA + ShareGPT + GPT-4 judge |
| 1 | 它解决什么问题 | Alpaca 单轮的天花板、对话数据稀缺 |
| 2 | 谱系定位 | LLaMA→Alpaca→Vicuna 的演化 |
| 3 | 数据：ShareGPT 多轮对话 | 清洗、长度切分、HTML→markdown |
| 4 | 训练目标：只在「回答」上算 loss | loss mask、多轮拼接 |
| 5 | FastChat 是什么 | 训练 / 推理 / 服务 / 评测一体 |
| 6 | 分布式与显存 | FSDP、gradient checkpointing、flash-attn |
| 7 | 对话模板与角色标记 | system / USER / ASSISTANT |
| 8 | 推理服务架构 | controller + worker + OpenAI API |
| 9 | 评测：GPT-4 as judge 起源 | 80 题、pairwise、位置偏置 |
| 10 | 与同类对比 | Alpaca / Koala / Dolly / ChatGLM |
| 11 | 意义与局限 | 「90% ChatGPT 质量」的来龙去脉 |
| — | 流程示例 / 常见问题 / 跳转 | 实操 |

> 说明：本文讲清的是**机制与思路**这类稳定知识。精确的版本号、CLI 默认值、API 签名、源码行号请以 FastChat 官方仓库与论文为准，文中凡涉及处均会标注。

---

## 0. 一句话锚点

```
LLaMA(基座，已具备语言能力，但不会"听话聊天")
        │  用 7 万条真人与 ChatGPT 的多轮对话(ShareGPT)做监督微调
        ▼
Vicuna(会多轮对话、能跟随指令的助手模型)
        │  再用 GPT-4 给 80 道题打分做自动评测
        ▼
结论:"达到 ChatGPT 约 90% 质量"(作者博客口径,非严格基准)
```

- 出身：2023 年 3 月由 LMSYS Org（UC Berkeley / CMU / Stanford / UCSD / MBZUAI 联合）发布。
- 三个关键词：**基座 = LLaMA**、**数据 = ShareGPT 多轮对话**、**评测 = GPT-4 裁判**。
- 配套工程 = **FastChat**（训练、推理、服务、评测全套），这是 Vicuna 留给业界最持久的资产。

---

## 1. 地基：它解决什么问题

把背景拆到最原子，才能看懂 Vicuna 的每个设计。

**(a) 基座只会"续写"，不会"对话"。** LLaMA 是在海量网页/书籍上做下一个 token 预测的预训练模型。给它 `"中国的首都是"` 它能补 `"北京"`，但你直接问 `"帮我写封请假邮件"`，它倾向于继续"补全"而不是"应答"——因为预训练语料里没有「指令→响应」这种范式。需要**指令微调（SFT）**把它对齐到"助手"角色。

**(b) Alpaca 的天花板：单轮 + 蒸馏指令。** 在 Vicuna 之前，Stanford Alpaca 用 `text-davinci-003` 自动生成的 5.2 万条 **self-instruct** 数据微调 LLaMA-7B，证明了「小成本也能让基座听话」。但 Alpaca 的数据是**单轮**（一问一答）、且由模型"造"出来，话题分布窄、缺乏真实对话的来回澄清。结果：Alpaca 不擅长**多轮**、回答偏短、上下文跟随弱。

**(c) 真实多轮对话数据稀缺。** 想让模型像 ChatGPT 那样能多轮澄清、记住前文、写长内容，就需要**真实的人机多轮对话**。Vicuna 的核心 insight：**ShareGPT** 这个网站上，大量用户主动分享了自己与 ChatGPT 的完整对话，天然就是高质量、多领域、多轮的 SFT 语料。

```
                 数据来源对比
Alpaca:  人写种子 ──► text-davinci-003 自造指令 ──► 单轮 (问,答)
Vicuna:  真人提问 ◄──► ChatGPT 真实作答 ──► 多轮 (问,答,问,答,...)
                         (用户自愿分享到 ShareGPT)
```

一句话：**Alpaca 解决"听不听话"，Vicuna 解决"会不会聊"。**

---

## 2. 谱系定位：一张演化图

```
                    LLaMA (Meta, 预训练基座: 7B/13B/33B/65B)
                      │
        ┌─────────────┼─────────────────────────────┐
        ▼             ▼                             ▼
   Alpaca         Vicuna                      Chinese-LLaMA-Alpaca
 (self-instruct  (ShareGPT 多轮对话           (中文增量预训练
  5.2万单轮)       7万对话, FastChat)            +中文指令)  ──► [[llm-train/chinese-llama-alpaca/README]]
        │             │
        │             └──► Koala / FastChat-T5 / 各类衍生
        ▼
    后续被"GPT-4 as judge"评测范式启发的一票工作
```

要点：Vicuna **不改 LLaMA 的网络结构**（RMSNorm / RoPE / SwiGLU / 因果注意力等参见 [[llm-algo/llama/模型架构]]），它的全部创新在**数据**与**评测**两侧，外加 **FastChat** 这套工程。理解 Vicuna，本质是理解「数据工程 + 训练目标 + 评测方法」。

---

## 3. 数据：ShareGPT 多轮对话的清洗管线

数据决定上限。Vicuna 的数据处理是它"比 Alpaca 强"的物理来源，逐步拆解：

```
原始 ShareGPT 导出 (HTML, 含真人×ChatGPT 多轮)
   │ ① HTML → Markdown  (保留代码块/列表, 去掉样式噪声)
   ▼
结构化对话 (一条样本 = 一整段多轮 conversation)
   │ ② 语言过滤 / 去重 / 删掉不当与低质内容
   ▼
   │ ③ 按上下文长度切分: 超过最大序列长度的长对话
   │    被切成多个不超过 max_len 的片段(而非直接截断丢弃)
   ▼
训练集 (作者口径约 7 万条对话)
```

每步的"为什么"：

- **① HTML→Markdown**：ShareGPT 页面是富文本，代码、表格、列表若不规整化，模型会学到一堆 `<div>` 噪声。转成 markdown 既干净又保留了"代码块"这种对编程问答至关重要的结构。
- **② 过滤去重**：真实数据混入大量重复、机翻、空洞、违规内容；不清洗会污染对齐方向。
- **③ 长度切分**：这是和 Alpaca 最大的工程差异。多轮对话往往很长，超过模型上下文窗口。Vicuna **不简单截断**，而是把长对话**切成多段**，每段都是合法的多轮片段——这样既不浪费长样本，又能让模型见到"长上下文"。这也是 Vicuna 把 LLaMA 默认上下文（早期 2048）训练/使用到更大窗口（论文/博客提到扩到 2048 量级并优化内存）的动机之一。具体长度阈值以官方脚本为准。

> 经验数值（量级感，非精确）：约 7 万条多轮对话；相较 Alpaca 5.2 万条单轮，**轮次更深、单样本更长、领域更广**（编程、写作、推理、闲聊都有）。

---

## 4. 训练目标：只在「助手回答」上计算 loss

这是 SFT 对话微调的灵魂，必须拆到 token 级别。

普通语言模型对每个位置都算交叉熵：

$$\mathcal{L}=-\sum_{t} \log P_\theta(x_t \mid x_{<t})$$

但对话 SFT 我们**不想**模型去"学会生成用户的话"——用户输入是给定条件，不该贡献梯度。于是引入 **loss mask**：只在 ASSISTANT 的回答 token 上算 loss。

```
拼接后的一条多轮样本(token 级):

[SYSTEM ...] USER: 你好 ASSISTANT: 你好,有什么可以帮你 </s> USER: 写首诗 ASSISTANT: 床前明月光... </s>
└──── mask=0 ────┘└mask=0┘ └──── mask=1 ────┘     └mask=0┘ └──── mask=1 ───┘
        ▲不算loss          ▲只在助手回答上算loss(mask=1)

mask=0 的位置: 仍然参与"注意力上下文"(模型能看到), 但不回传 loss。
```

为什么这样设计：

- **不学用户文风**：若对用户 token 也算 loss，模型会去拟合"提问的写法"，跑偏。
- **多轮共享一条样本**：一段对话里的多轮回答**同时**贡献 loss，模型在一个前向里学会"基于前几轮上下文继续作答"——这正是多轮能力的来源，也是 Alpaca（每条只一问一答）学不到的。
- **拼接 + EOS**：每轮回答末尾放 `</s>`（EOS），教模型"该停"，避免推理时停不下来。

数值直觉：若一条样本 600 个 token，其中助手回答占 350 个，则只有这 350 个位置回传梯度，其余 250 个（system+user+模板符）mask 掉。

训练其余设置（量级，精确以官方为准）：全参数微调（非 LoRA，原版 Vicuna 是 full fine-tune）、AdamW、cosine 学习率衰减、warmup、bf16/fp16 混合精度，3 个 epoch 左右。

---

## 5. FastChat：Vicuna 的"操作系统"

Vicuna 不是一个孤立权重，而是随附了 **FastChat** 这套开源平台。它是 Vicuna 工程价值的核心。

> 源码以官方仓库为准：`https://github.com/lm-sys/FastChat`（具体 commit/版本号请查仓库）。

**FastChat 解决什么**：把"做一个聊天模型"涉及的四件事——**训练、单机推理、多机服务、自动评测**——统一到一个代码库，并提供**与 OpenAI 兼容的 API**，让生态工具可以无缝接入开源模型。

```
                       FastChat 总体模块
┌───────────────────────────────────────────────────────────┐
│  train/          多轮对话 SFT(loss mask + 模板),支持 FSDP   │
│  conversation    对话模板注册表(Vicuna/各模型的角色与分隔符)│
│  serve/                                                     │
│    ├─ controller        服务注册中心,路由到空闲 worker      │
│    ├─ model_worker      加载权重,执行生成,上报心跳          │
│    ├─ openai_api_server 暴露 /v1/chat/completions(OpenAI 兼容)│
│    └─ gradio_web_server 网页聊天 UI / 对战 Arena            │
│  llm_judge/      MT-Bench: 用 GPT-4 评测(见第 9 节)         │
└───────────────────────────────────────────────────────────┘
```

记住一句：**模型权重是"内容"，FastChat 是"播放器+录音棚+评分台"。**

---

## 6. 分布式训练与显存：怎么把 13B 喂下去

全参数微调 13B 对显存是硬挑战。Vicuna 训练用到的标准手段（机制层面）：

```
显存四大招(组合使用):
┌──────────────────────┬──────────────────────────────────────┐
│ FSDP(全分片数据并行)  │ 把参数/梯度/优化器状态切片分到各 GPU  │
│ Gradient Checkpoint   │ 前向只存部分激活,反向时重算,省激活显存│
│ Mixed Precision       │ bf16/fp16 算,关键处 fp32,省一半显存   │
│ Flash Attention       │ 注意力不落地 N×N 矩阵,省显存且更快    │
└──────────────────────┴──────────────────────────────────────┘
```

为什么需要它们一起上：

- **FSDP**：单卡放不下 13B 的"参数 + 梯度 + Adam 的一阶/二阶动量"（优化器状态约是参数量的 2 倍）。FSDP 把这三样**分片**到 N 张卡，每张只持有 1/N，临用时 all-gather。相比 ZeRO，理解为同一思想的 PyTorch 原生实现。
- **Gradient Checkpointing**：长序列（多轮对话很长）激活显存爆炸，用"重算换显存"。
- **Flash Attention**：序列越长，朴素注意力的 $O(L^2)$ 中间矩阵越致命；flash-attn 通过分块在 SRAM 里融合计算，**不显式存** $L\times L$ 注意力矩阵。对 Vicuna 的"长对话"场景尤其关键。

> 这些是工程惯例，具体 batch size / 学习率 / 节点数请以官方训练脚本为准，不要照搬本文示意值。

---

## 7. 对话模板：角色标记是怎么回事

模型本身只懂 token 序列，"谁在说话"靠**模板**编码。Vicuna 的对话模板大致形如：

```
{system 提示, 例如 "A chat between a curious user and an AI assistant..."}

USER: {第1轮用户输入}
ASSISTANT: {第1轮助手回答}</s>
USER: {第2轮用户输入}
ASSISTANT: {第2轮助手回答}</s>
...
USER: {当前用户输入}
ASSISTANT:            ← 推理时模型从这里开始续写
```

关键点（精确分隔符以 FastChat `conversation.py` 注册表为准，不同版本略有差异）：

- **角色串 `USER:` / `ASSISTANT:`**：纯文本标记，告诉模型当前是谁的回合。训练与推理**必须用同一套模板**，否则分布不匹配、效果崩。
- **分隔符 `</s>`**：标记一轮助手回答结束（EOS）。这是模型学会"停"的信号。
- **system 段**：设定助手人格与边界，放在最前，所有轮共享。
- **多轮拼接**：历史轮全部拼进 prompt 作为上下文——这就是"记得前文"的实现，本质是把记忆塞进了输入窗口（所以上下文长度是硬约束，呼应第 3 节的长度切分）。

---

## 8. 推理服务：一次请求的生命周期

FastChat 的服务是**分布式可扩展**的（controller + 多 worker），便于 Arena 这种高并发场景。请求生命周期：

```
   用户/客户端
      │ ① POST /v1/chat/completions  (OpenAI 兼容格式: messages[])
      ▼
 openai_api_server
      │ ② 问 controller: 哪个 worker 有 "vicuna-13b" 且空闲?
      ▼
   controller ──③ 返回某 worker 地址──┐
      ▲                               │
      │ (worker 定时上报心跳/负载)     ▼
      │                          model_worker
      │ ④ 按 conversation 模板把 messages 拼成 prompt
      │ ⑤ 模型自回归生成(可流式 token-by-token)
      │ ⑥ 解码 + 遇 </s> 或 max_tokens 停
      └──────── ⑦ 流式/整段返回 ─────► 用户
```

设计权衡：

- **controller 解耦**：模型与入口分离，可水平扩容 worker、灰度多模型、做负载均衡。代价是多一跳网络与一个需维护的注册中心。
- **OpenAI 兼容**：让所有为 OpenAI 写的客户端/框架（LangChain 等）零改动接开源模型——这是 FastChat 生态影响力的关键。
- **流式输出**：逐 token 推送，降低首字延迟体验。

---

## 9. 评测：GPT-4 as a Judge 的起源

这是 Vicuna 留给整个领域的方法论遗产，单独细讲。

**问题背景**：聊天质量是**开放式**的——没有标准答案，传统准确率/BLEU 评不出"哪个回答更有帮助"。人工评测又慢又贵且难复现。

**Vicuna 的破局思路（首创）**：**让更强的模型（GPT-4）当裁判**。

```
        GPT-4 as Judge 早期形态(Vicuna 博客)
┌────────────────────────────────────────────────┐
│ ① 准备 ~80 道覆盖多类别的题(写作/角色扮演/数学/  │
│    代码/常识/费米估算/反事实...)                  │
│ ② 同一道题让两个模型(如 Vicuna vs ChatGPT)各答   │
│ ③ 把"问题 + 两个回答"喂给 GPT-4                   │
│ ④ GPT-4 给两边各打分(1-10)并给出文字理由          │
│ ⑤ 汇总相对得分 → "Vicuna 达 ChatGPT 约 90% 质量"  │
└────────────────────────────────────────────────┘
```

为什么能成立、又有什么坑（作者也坦承的局限）：

- **可行性**：GPT-4 的判断与人类偏好相关性较高，且**可批量、可复现、近零边际成本**。这把"对齐质量"从主观感受变成了可量化数字。
- **位置偏置（position bias）**：把同一个模型放在"答案 A"和"答案 B"位置，GPT-4 打分会偏向某个位置。**缓解**：交换顺序各评一次取平均。
- **冗长偏置（verbosity bias）**：裁判倾向于给更长的回答更高分。
- **自我偏好（self-enhancement bias）**：裁判可能偏爱与自己风格相近的回答。
- **能力天花板**：裁判不会比自己更懂；数学/推理题它自己都可能判错。

**演化**：这个 80 题原型后来被 FastChat 团队系统化为 **MT-Bench**（多轮、含打分 prompt 与裁判校准）与 **Chatbot Arena**（真人盲评 + Elo 排名）。可以说：**今天满天飞的"LLM-as-a-judge"，源头就在 Vicuna。**

```
评测三件套谱系:
Vicuna 80题(2023.3) ──► MT-Bench(单轮+多轮,GPT-4打分) ──► Chatbot Arena(人类Elo)
        ▲ "GPT-4 as judge" 的起点
```

---

## 10. 与同类对比（同期 LLaMA 系指令模型）

| 模型 | 基座 | 训练数据 | 多轮 | 评测亮点 | 一句话 |
|------|------|---------|------|---------|--------|
| **Vicuna** | LLaMA | ShareGPT 真实多轮 ~7万 | 强 | 首创 GPT-4 judge | 数据真实+评测创新 |
| Alpaca | LLaMA | self-instruct 单轮 5.2万 | 弱 | 人评/简单对比 | 证明"小钱能对齐" |
| Koala | LLaMA | 多源对话(含蒸馏+真实) | 中 | 人评盲测 | 强调数据来源混合 |
| Dolly | 非LLaMA(Pythia) | 员工手写指令(可商用) | 弱 | — | 主打开源可商用 |
| ChatGLM | GLM 自研 | 中英双语对齐 | 强 | 中文为主 | 中文聊天强 |

辨析：

- **Vicuna vs Alpaca**：同基座，差别全在**数据（真实多轮 vs 自造单轮）**与**评测（GPT-4 vs 人评）**。
- **数据来源争议**：ShareGPT 内容由 ChatGPT 产出，存在 OpenAI 服务条款与版权的**合规风险**，且权重受 LLaMA 原始许可约束（早期非商用）。要商用语境请看 Dolly 这类自采数据方案，中文场景见 [[llm-train/chinese-llama-alpaca/README]]。

---

## 11. 意义与局限

**意义（为什么 Vicuna 是里程碑）：**

1. **数据范式**：证明"真实人机多轮对话"是性价比极高的 SFT 语料，把开源聊天模型的体验拉近闭源一个档次。
2. **评测范式**：**GPT-4 as judge / MT-Bench / Arena** 成为此后两年开源社区的事实标准评测方式。
3. **工程范式**：**FastChat** 提供了训练→服务→评测的一站式底座，OpenAI 兼容 API 极大降低了用开源模型替换闭源的门槛。

**局限（务必清醒）：**

- **"90% ChatGPT"是作者博客口径**，基于 80 题 + GPT-4 裁判，**不是严格学术基准**，且裁判本身有上文所述偏置，不要当硬指标引用。
- **合规**：ShareGPT 数据来自 ChatGPT 输出，叠加 LLaMA 许可，**原版 Vicuna 仅供研究**。
- **能力上限受基座限制**：SFT 只是"对齐表达"，知识与推理上限仍由 LLaMA 预训练决定；且未做 RLHF，安全性/拒答能力弱于经过完整对齐流水线的闭源模型。
- **幻觉与时效**：继承基座的事实性短板与知识截止。

---

## 典型流程示例（讲含义，非可直接运行命令）

下面是"复刻 Vicuna 思路"的端到端流程，命令名/参数以 FastChat 官方文档为准：

```
1) 准备数据
   ShareGPT 导出 ──HTML→markdown──► 清洗去重 ──按 max_len 切分──► train.json
       含义: 得到"多轮对话"格式样本, 每条是一整段 conversation

2) 训练 (多卡, 全参数微调)
   torchrun/deepspeed 启动 FastChat 的 train 脚本
   关键开关含义:
     --model_name_or_path   LLaMA 基座权重路径
     --data_path            上一步的多轮对话 json
     --fsdp                 开 FSDP 分片(省显存,见第6节)
     --gradient_checkpointing  重算换显存
     --bf16                 混合精度
     --model_max_length     上下文长度上限(对应数据切分阈值)
       ↑ 训练时对 USER 段做 loss mask(第4节), 只在 ASSISTANT 段回传梯度

3) 起服务 (controller + worker + api)
     controller        路由中心
     model_worker      加载 vicuna 权重, 注册到 controller
     openai_api_server 暴露 OpenAI 兼容 /v1/chat/completions

4) 评测 (GPT-4 as judge)
     用 MT-Bench: 让模型答多轮题 → 调 GPT-4 按 prompt 打分(1-10)
     注意: 交换 A/B 顺序两评取平均, 缓解位置偏置(第9节)
```

参数权衡速记：

| 配置 | 调大的好处 | 调大的代价 | 经验取向 |
|------|-----------|-----------|---------|
| `model_max_length` | 能学/用更长对话 | 显存与算力 $O(L^2)$ 上升 | 配 flash-attn 才划算 |
| epoch 数 | 拟合更充分 | 过拟合、复读、灾难性遗忘 | 对话 SFT 常 3 左右 |
| 学习率 | 收敛快 | 太大崩、太小学不动 | full-ft 取小(配 warmup+cosine) |
| FSDP 分片粒度 | 省显存 | 通信开销增大 | 按卡数与带宽权衡 |

---

## 常见问题

| 问题 | 解答 |
|------|------|
| Vicuna 改了 LLaMA 的网络结构吗？ | 没有。结构同 LLaMA（见 [[llm-algo/llama/模型架构]]），创新在数据与评测。 |
| 和 Alpaca 到底差在哪？ | 数据：真实**多轮** ShareGPT vs 自造**单轮** self-instruct；评测：GPT-4 judge vs 人评。 |
| 为什么只在 ASSISTANT 回答上算 loss？ | 用户输入是给定条件，不该回传梯度；否则模型会去学"提问写法"，且这样才能学到多轮跟随（第 4 节）。 |
| "多轮能力"是怎么实现的？ | 把历史轮全拼进 prompt 当上下文 + 一条样本里多个回答同时算 loss；记忆受上下文窗口限制。 |
| Vicuna 用 LoRA 还是全参数微调？ | 原版是**全参数微调**；社区有 LoRA 版（如 vicuna-lora）作为低成本替代。 |
| "达到 ChatGPT 90%"可信吗？ | 是博客口径，基于 80 题 + GPT-4 裁判，裁判有位置/冗长/自偏好等偏置，**不宜当硬基准**。 |
| FastChat 和 Vicuna 是一回事吗？ | 不是。Vicuna 是**权重**，FastChat 是承载训练/服务/评测的**平台**，二者由同一团队发布。 |
| GPT-4 as judge 有什么坑？ | 位置偏置、冗长偏置、自我偏好、裁判能力天花板；用顺序交换、控长度等缓解（第 9 节）。 |
| 能商用吗？ | 原版受 LLaMA 许可 + ShareGPT 数据合规限制，**仅供研究**；商用看自采数据方案。 |
| 中文场景怎么办？ | 看 [[llm-train/chinese-llama-alpaca/README]]，做中文增量预训练 + 中文指令对齐。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-algo/llama/模型架构]] — Vicuna 的基座结构（RMSNorm / RoPE / SwiGLU / 因果注意力）
- [[llm-train/chinese-llama-alpaca/README]] — 同谱系的中文化方案对照

> 参考：FastChat 官方仓库 `https://github.com/lm-sys/FastChat`、LMSYS Vicuna 博客与 MT-Bench/Chatbot Arena 论文。文中精确版本号/CLI 默认值/分隔符等，请以官方源码与文档为准。
