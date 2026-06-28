# 昇腾 MindFormers · ChatGLM3-6B 推理实战

> 一句话定位：本目录是「把 ChatGLM3-6B 在华为昇腾 NPU 上用 MindFormers 跑起来并精确测速」的最小可复现实战，三个脚本分别对应**功能验证 → 逐 Token 观测 → 标准性能基准**三层递进。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/vllm/README]] [[llm-optimizer/kv-cache]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] [[ai-infra/算力/GPU工作原理]]

## 阅读地图

| 章节 | 你将搞懂 | 关键词 |
| --- | --- | --- |
| 0 一句话锚点 | 这三个脚本到底各干什么 | 推理 / 测速 / 基准 |
| 1 地基 | ChatGLM3 是什么、为什么上昇腾、前置环境 | GLM / NPU / MindFormers |
| 2 模型实例化 | `AutoModel` 两种加载法的差别 | from_pretrained / from_config |
| 3 增量推理与 KV Cache | `use_past=True` 为何能加速 | Prefill / Decode / KV |
| 4 静态图模式 | `GRAPH_MODE` 在 NPU 上意味着什么 | 图编译 / 首次慢 |
| 5 对话输入构造 | `build_chat_input` 干了什么 | 角色 / 模板 / token |
| 6 脚本一：基础推理 | 跑通 + 多轮交互 | REPL / generate |
| 7 脚本二：逐 Token 计时 | 暴露 Decode 的线性增长 | new_tokens 扫描 |
| 8 脚本三：性能基准 | TTFT / TPOT / TP99 怎么算 | 首 Token / 增量 / 分位 |
| 9 性能指标与数值手算 | 把时延拆成可解释的账 | 端到端 = 首 + 增量×步数 |
| 10 评价 / 局限 | 这套测法的盲点 | batch=1 / 采样关闭 |

---

## 0. 一句话锚点

**这三个脚本 = 同一个 ChatGLM3-6B 模型，三种「用法/测法」。**

- `chatglm-inference.py`：**能不能用** —— 跑通一次推理 + 一个多轮对话 REPL。
- `chatglm-gen.py`：**慢在哪** —— 对同一输入扫描 `max_new_tokens=1..20`，肉眼看到「生成越长、耗时越长」的线性规律。
- `chatglm-stat.py`：**到底多快** —— 喂一批真实 prompt（alpaca 数据），算出**首 Token 时延 / 单步增量时延 / 端到端时延**及其 TP50/TP90/TP99 分位。

三者共用同一套加载与生成代码，理解一份即理解全部，差异只在「循环怎么测」。

---

## 1. 地基：背景与前置

### 1.1 ChatGLM3-6B 是什么

ChatGLM3 是智谱 AI 的 GLM 系列对话模型，6B 参数量、中英双语、支持多轮对话与工具调用。底层是 **GLM（General Language Model）** 架构——一种自回归填空预训练的 Transformer 变体；推理阶段对外行为与标准 Decoder-only 大模型一致：**给一段上下文 token，自回归地一个个吐出新 token**。本目录关心的是「推理这一步在昇腾上怎么跑、怎么测」，架构细节见 [[llm-algo/transformer/模型架构]]。

### 1.2 为什么是昇腾 + MindFormers

CUDA 世界里你会用 `transformers` + vLLM；昇腾世界里对应的是 **MindFormers + MindSpore + CANN**（完整软件栈与类比见上级目录 [[llm-inference/ascend/mindformers/README]]）。一句话对照：

```
   HuggingFace 世界                 昇腾世界（本目录）
  ┌────────────────┐              ┌──────────────────────┐
  │ transformers   │   对应 ───►  │ mindformers          │
  │ AutoModel      │              │ AutoModel (MindSpore) │
  │ model.generate │              │ model.generate (同名) │
  │ NVIDIA GPU     │              │ 昇腾 NPU (Ascend 910) │
  └────────────────┘              └──────────────────────┘
```

API 名字几乎一模一样（`AutoModel / AutoTokenizer / generate`），**迁移成本主要在底层算子与权重格式**，不在你写的这几十行脚本。

### 1.3 前置环境（以官方为准）

- 昇腾 NPU + 对应 CANN / 驱动；MindSpore + MindFormers（版本须配套，**具体版本以官方文档为准**）。
- 已转换为 MindSpore 格式的权重目录：脚本里写死为 `/root/workspace/model/chatglm3-6b_ms`，内含 `glm3_6b.ckpt`、`run_glm3_6b.yaml`、分词器文件。HF 原始权重 → `.ckpt` 的转换由 MindFormers 提供脚本（流程见上级目录「模型迁移」一节）。

```
chatglm3-6b_ms/                ← AutoXXX.from_pretrained 指向这里
├── glm3_6b.ckpt               ← MindSpore 权重（checkpoint）
├── run_glm3_6b.yaml           ← 模型 + 推理 + 并行 配置
├── tokenizer.model            ← 分词器
└── ...
```

---

## 2. 模型实例化：两条路

三个脚本头部都给了**两种加载方式**（第二种被注释掉，作模板）：

```python
# 方式 1：默认配置，一行加载（脚本实际使用）
model = AutoModel.from_pretrained('/root/workspace/model/chatglm3-6b_ms')

# 方式 2：先拿配置、改完再实例化（模板，演示如何打开优化）
config = AutoConfig.from_pretrained('.../run_glm3_6b.yaml')
config.use_past = True          # ★ 开增量推理，加速 Decode
config.seq_length = 2048        # 最大序列长度
config.checkpoint_name_or_path = '.../glm3_6b.ckpt'
model = AutoModel.from_config(config)
```

| 维度 | from_pretrained | from_config |
| --- | --- | --- |
| 写法 | 一行，吃目录里的默认 yaml | 显式拿 `config` 再改字段 |
| 何时用 | 默认配置就够（快速验证） | 要改 `use_past`/`seq_length`/并行等 |
| 关键开关 | 由 yaml 决定 | 代码里可临时覆盖 |

```
        from_pretrained(目录)
   目录 ─────────────► 读默认 yaml ──► 建图 ──► 加载 ckpt ──► model
        from_config(config)
   yaml ──► config 对象 ──► 改字段(use_past=True...) ──► 建图 ──► model
                          ▲ 这一步是「方式2」存在的唯一理由：可编程地改配置
```

> 实战建议：先用方式 1 跑通，再切方式 2 打开 `use_past=True` 对比测速——这正是脚本把方式 2 留作模板的用意。

---

## 3. 增量推理与 KV Cache（`use_past=True` 为什么快）

这是整个目录最值钱的一节。自回归生成天然分两个阶段：

```
阶段                输入                          算什么
─────────────────────────────────────────────────────────────
Prefill(首Token)   整段 prompt（n 个 token）      一次性算完 n 个位置的 K/V
                                                  → 产出第 1 个新 token
Decode(后续Token)  只喂「上一个新 token」(1 个)    只算这 1 个位置的 Q
                                                  复用之前缓存的所有 K/V
```

**没有 KV Cache**：每生成一个新 token，都把「prompt + 已生成」整段重算注意力——计算量随序列长度平方膨胀。
**有 KV Cache（`use_past=True`）**：历史 token 的 Key/Value 算一次就**缓存**起来，Decode 时新 token 只需算自己的 Q，再去和缓存里的 K/V 做注意力。

```
  无 Cache 的 Decode：               有 Cache 的 Decode：
  ┌──────────────────────┐         ┌──────────────────────┐
  │ 重算 t0..t_{n} 全部 K/V│        │ K/V 缓存:[t0..t_{n-1}] │ ← 复用
  │ 才能加 1 个新 token   │         │ 只算 t_n 的 Q/K/V     │ ← 增量
  └──────────────────────┘         └──────────────────────┘
   每步 O(n²)                        每步 O(n)，省下重复 GEMM
```

> 直觉：写作文时，前面写过的内容你不会每加一个字就从头重读一遍——你「记得」前文。KV Cache 就是模型的「记得」。详见 [[llm-optimizer/kv-cache]] 与 [[llm-inference/KV-Cache优化]]。

正因为有这个两阶段差异，**首 Token（Prefill）天然比后续每个 Token（Decode）慢得多**——脚本三专门把这两者分开测，就是为了把这笔账拆清楚。

---

## 4. 静态图模式：`GRAPH_MODE` 在 NPU 上的含义

三个脚本第一行实质都是：

```python
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
```

- `device_target="Ascend"`：算到昇腾 NPU 上（对标 `.cuda()`）。
- `device_id`：选第几张卡（脚本里分别写了 0 / 6 / 3，按你机器空闲卡改）。
- `mode=ms.GRAPH_MODE`：**静态图模式**——MindSpore 先把整个前向**编译成一张固定的算子图**，再交给 CANN 在 NPU 上执行。

```
  动态图(PyNative)            静态图(GRAPH_MODE，本脚本)
  ┌────────────┐             ┌────────────────────────┐
  │ 逐算子解释执行│           │ 整图先编译 → 再整图下发  │
  │ 易调试、较慢 │           │ 首次编译慢、之后很快     │
  └────────────┘             └────────────────────────┘
                              ▲ 适合推理：图固定、反复跑
```

> **重要副作用**：静态图「第一次跑要编译整张图」，所以**第一次 `generate` 明显比之后慢**。这就是为什么脚本一/三都先做一次「热身生成」（打印「第一次生成时间」）再进入正式循环——避免把编译耗时算进基准。详见 [[docs/transformer内存估算]] 旁注的图编译概念。

---

## 5. 对话输入构造：`build_chat_input`

ChatGLM3 是对话模型，不能直接把裸字符串塞进去，要按**对话模板**拼角色标记：

```python
role = "user"
inputs = tokenizer.build_chat_input(text, history=history, role=role)
inputs = inputs['input_ids']        # 取出 token id 张量
input_token_lens = len(inputs[0])   # 记录输入 token 数（测速要用）
```

`build_chat_input` 做的事（概念上）：

```
 "可以帮我做一份旅游攻略吗？" + history + role=user
        │ 套对话模板（加 <|user|> / <|assistant|> 等特殊 token）
        ▼
 "<|user|>\n可以帮我...\n<|assistant|>"
        │ 分词器编码
        ▼
 input_ids = [[ id, id, id, ... ]]   ← 喂给 model.generate
```

注意脚本里 `history=[]` 始终为空、且循环中没回填——**本目录是「单轮重复测速」，不是真多轮记忆对话**；要做真多轮需把每轮回复 append 进 `history`。

---

## 6. 脚本一 `chatglm-inference.py`：基础推理 + REPL

最朴素的「跑通」脚本：一次热身生成，然后进入交互循环。

```python
outputs = model.generate(
    inputs, do_sample=False, num_beams=1,
    top_k=1, top_p=1, temperature=1,
    repetition_penalty=1.0, max_new_tokens=128)
outputs = outputs[0][len(inputs[0]):]   # ★ 切掉输入部分，只留新生成的
response = tokenizer.decode(outputs)     # token → 文字
```

**采样参数全部关掉**（`do_sample=False, num_beams=1, top_k=1, top_p=1, temperature=1`）= **贪心解码**：每步取概率最高的 token，**输出确定、可复现**——测速就要这种确定性，否则每次生成长度不同没法比。

`outputs[0][len(inputs[0]):]` 这步切片是关键细节：`generate` 返回的是「输入 + 输出」拼一起，必须按输入长度切掉前缀，才得到「纯新生成」。

```
  generate 返回:  [ 输入 prompt 的 token ... | 新生成的 token ... ]
                   └────── len(inputs[0]) ─────┘
  切片后:                                       [ 新生成的 token ... ]  ← decode 它
```

REPL 循环：`line = input()` 读一行 → 生成 → 打印「生成时间」→ 回到等待。空行退出。

---

## 7. 脚本二 `chatglm-gen.py`：逐 Token 计时，看清 Decode 线性增长

与脚本一几乎相同，唯一差别在内层循环——对**同一个输入**，让 `max_new_tokens` 从 1 扫到 20：

```python
for i in range(20):
    max_new_tokens = i + 1
    start = time.perf_counter()
    outputs = model.generate(inputs, ..., max_new_tokens=max_new_tokens)
    gen_time = time.perf_counter() - start
    print("生成时间：", gen_time,
          "输入Token长度：", input_token_lens,
          "生成Token长度：", new_token_lens)
```

它暴露的规律——**生成 k 个 token 的总时间 ≈ 首 Token 时间 + (k−1) × 单步增量时间**：

```
 max_new_tokens →   1     2     3     ...    20
 总耗时(示意)    →  ▇     ▇▇    ▇▇▇          ▇▇▇...▇▇
                    │      └─ 每多 1 个 token，多 ~一段固定增量时间
                    └─ 这一段最重（含 Prefill），后面每段近似相等
```

> 这张「阶梯图」是直观理解 TTFT vs TPOT 的最好教具：第 1 阶台阶高（Prefill 贵），之后每阶高度接近（Decode 每步成本稳定）。注意这里是**逐次独立调用 generate**（每个 k 重头跑），与真实「一次生成 20 个」略有差异，但足以观测趋势。

---

## 8. 脚本三 `chatglm-stat.py`：标准性能基准

这是工程上真正会跑的「测试报告」脚本。输入是一份真实数据集 `alpaca_gpt4_data_input_2k.json`（约 2k 条 prompt），分两趟测：

```
第 1 趟：对每条 prompt 设 max_new_tokens=1
         → 只生成 1 个 token = 纯 Prefill 耗时 = 首 Token 时延(TTFT)
         记入 first_token_time_list

第 2 趟：对每条 prompt 设 max_new_tokens=100
         → Prefill + 99 步 Decode = 端到端耗时
         记入 total_token_time_list，同时记录实际生成长度 new_token_lens
```

核心推导——把「单步增量时延（TPOT）」从两趟测量里反解出来：

```python
# 端到端 = 首Token + 增量×(生成步数-1)  ⇒ 反解增量
token_time = (total_time[i] - first_time[i]) / (new_token_lens[i] - 1)
```

```
  total_token_time ─────────────────────────────────────►
  ├─ first_token_time ─┤├─ 增量 ─┤├─ 增量 ─┤ ... ├─ 增量 ─┤
        (TTFT)          └──────── (n-1) 段，平均即 TPOT ────┘
```

最后按分位数汇报（这是性能基准的标准姿势，避免被极值误导）：

```python
np.percentile(first_token_time_list, 50)  # TP50 中位数
np.percentile(first_token_time_list, 90)  # TP90
np.percentile(first_token_time_list, 99)  # TP99 长尾
```

对**首 Token 时延 / 端到端时延 / 生成 Token 长度**三组各报 min/max/TP50/TP90/TP99。指标定义详见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

---

## 关键公式 / 指标 / 数值手算

### 三个核心时延指标

| 指标 | 英文 | 脚本怎么测 | 直觉 |
| --- | --- | --- | --- |
| 首 Token 时延 | TTFT (Time To First Token) | `max_new_tokens=1` 的耗时 | 用户多久看到第一个字（Prefill 成本） |
| 单步增量时延 | TPOT (Time Per Output Token) | $(T_{end}-T_{first})/(n-1)$ | 之后每个字多久蹦一个（Decode 成本） |
| 端到端时延 | E2E Latency | `max_new_tokens=100` 的耗时 | 整句生成完总共多久 |

### 端到端时延模型（脚本的理论基础）

$$T_{e2e} \approx T_{TTFT} + (n-1)\times T_{TPOT}$$

其中 $n$ 为生成 token 数。该式正是脚本三反解 TPOT 的依据。

### 数值手算示例（假设值，仅演示算法，非实测）

设某条 prompt 测得：首 Token 时延 $T_{TTFT}=0.5\text{ s}$，端到端（生成 100 个）$T_{e2e}=3.0\text{ s}$，实际生成 $n=100$ 个 token。

- 单步增量时延：
$$T_{TPOT}=\frac{T_{e2e}-T_{TTFT}}{n-1}=\frac{3.0-0.5}{99}\approx 0.0253\ \text{s/token}\approx 25.3\ \text{ms/token}$$
- 解码吞吐：$1/T_{TPOT}\approx 39.5\ \text{token/s}$。
- 若把生成长度从 100 翻到 200，预估端到端：
$$T_{e2e}'\approx 0.5+199\times0.0253\approx 5.5\ \text{s}$$

> 数字均为**假设值用于演示公式**，真实数值以你机器实测/官方为准。

### 分位数为什么比平均值重要

```
  时延分布（长尾）           平均值被长尾拉高，骗人
  ┌──────────────────┐      TP50：一半用户体验好于此
  │ ▇▇▇▇▇▇▇▇         │      TP90：90% 用户在此之内
  │ ▇▇▇▇▇▇▇▇▇▇▇      │      TP99：保最差 1% 的 SLA
  │ ▇▇▇▇▇▇  ┄┄┄ 尾   │      → 线上承诺必须看 TP99，不能只看均值
  └──────────────────┘
```

---

## 评价 / 对照 / 局限

| 维度 | 本目录脚本现状 | 说明 / 改进方向 |
| --- | --- | --- |
| 并发 | **batch=1，逐条串行** | 无 Continuous Batching，吞吐被严重低估，参考 [[llm-inference/连续批处理?]] |
| 采样 | 贪心（全关采样） | 利于复现，但与线上带温度采样的真实分布有别 |
| 历史 | `history=[]` 不回填 | 是「单轮重复测速」，非真多轮记忆对话 |
| KV Cache | 默认走方式 1，未必显式开 `use_past` | 想测增量加速需用方式 2 打开对比 |
| 图编译 | 已用「热身生成」规避 | 静态图首次编译耗时不计入基准，做法正确 |
| 脚本二 | 每个 k 重头独立 generate | 观测趋势够用，但非「一次连续生成」的真实路径 |
| 显存/利用率 | 未采集 | 真基准应同时看 NPU 占用与显存峰值 |
| 数据依赖 | 路径写死 `/root/workspace/...` | 复现需改成你本地路径与权重目录 |

**适用场景**：昇腾 NPU 上对 ChatGLM3-6B 做「能跑通 + 出一份首/增量/端到端时延的分位报告」。**不适用**：高并发吞吐评估、长上下文压测、多轮对话质量评测——这些需引入批处理与服务化框架（昇腾侧对标 vLLM 的方案见 [[llm-inference/vllm/README]] 的生态对照）。

---

## 🔗 跳转链接

- 📍 [[00-知识地图]]
- 上级总览：[[llm-inference/ascend/mindformers/README]]
- KV Cache 机制：[[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]]
- 性能指标定义：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 批处理提升吞吐：[[llm-inference/连续批处理?]]
- 模型架构：[[llm-algo/transformer/模型架构]]
- CUDA 侧对照引擎：[[llm-inference/vllm/README]]
- 内存/图编译旁注：[[docs/transformer内存估算]]
- 硬件直觉：[[ai-infra/算力/GPU工作原理]]
