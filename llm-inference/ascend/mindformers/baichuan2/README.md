# 昇腾 MindFormers · Baichuan2 推理与性能压测

> 一句话定位：本目录是「百川 Baichuan2（7B / 13B）在华为昇腾 NPU 上用 MindFormers 跑推理 + 量化首/续 Token 时延」的最小可复现实战卡片。📍 导航：[[00-知识地图]]
> 🔗 相关：[[昇腾 MindFormers 推理]] [[llm-optimizer/kv-cache]] [[llm-inference/连续批处理]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 章节 | 你将搞懂 | 关键词 |
| --- | --- | --- |
| 0 一句话锚点 | 这俩脚本到底干了啥 | inference / stat |
| 1 地基 | Baichuan2 是什么、和 LLaMA 什么关系 | 架构 / 复用 LlamaConfig |
| 2 目录与脚本 | 两个 .py 各自职责 | 交互推理 / 批量压测 |
| 3 模型加载链路 | yaml → Config → 网络 → tokenizer | MindFormerConfig / .ckpt |
| 4 一条 prompt 的旅程 | generate 内部发生了什么 | Prefill / Decode / use_past |
| 5 采样参数逐字解释 | do_sample/top_k/top_p... | 贪心 vs 采样 |
| 6 性能压测方法学 | 怎么把"快慢"变成数字 | 首Token / 增量 / TP99 |
| 7 关键公式与数值手算 | 时延、吞吐、KV Cache、显存 | $TPOT$ / GB |
| 8 7B vs 13B 与坑 | 选型与排错 | OOM / 版本配套 |
| 评价/对照/局限 | 和 vLLM/HF 对比 | 表格 |

---

## 0. 一句话锚点

本目录有两个脚本，是同一件事的「演示版」和「测量版」：

- **`baichuan-inference.py`** —— **交互式问答**。加载 Baichuan2，先固定问一句"可以帮我做一份旅游攻略吗？"，再进入 `while line` 循环让你不停输入、不停生成。用来**验证模型能正确推理**。
- **`baichuan-stat.py`** —— **批量性能压测**。读一份 2000 条的指令数据集（`alpaca_gpt4_data_input_2k.json`），分别测「只生成 1 个 token」和「生成 100 个 token」两轮，算出**首 Token 时延、增量 Token 时延、端到端时延**以及 TP50/TP90/TP99 分位数。用来**量化模型在昇腾上跑多快**。

> 记忆锚点：`inference = 能不能跑通`，`stat = 跑得多快`。两者共用同一套「加载模型」代码，区别只在后半段——前者读键盘，后者读数据集 + 掐表统计。

---

## 1. 地基：Baichuan2 是什么，为什么代码里全是 `Llama`

### 1.1 Baichuan2 的身世

Baichuan2 是百川智能开源的中英双语大模型，有 **7B** 和 **13B** 两个尺寸，结构上是一个**类 LLaMA 的 Decoder-Only Transformer**：RoPE 旋转位置编码、RMSNorm、SwiGLU 激活的 FFN、因果自注意力。它与原版 LLaMA 的主要差异（以官方为准）：

- **更大词表**：约 12.5 万 token 的多语种词表（远超 LLaMA 的 3.2 万），中文压缩率更高；
- **位置编码差异**：7B 用 **RoPE**，13B 用 **ALiBi**（线性偏置），这是 7B/13B 在结构上最关键的不同；
- **NormHead**：对输出投影头做归一化，稳定训练（细节见原论文）。

### 1.2 为什么脚本里用 `LlamaConfig`

```python
from mindformers.models import LlamaConfig
from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_13b import Baichuan13BV2ForCausalLM
```

因为 Baichuan2 骨架就是 LLaMA，MindFormers 直接**复用 `LlamaConfig` 这套超参容器**（hidden_size / num_layers / num_heads / vocab_size ...），只把「真正不一样的算子层」单独实现成 `Baichuan7BV2ForCausalLM`、`Baichuan13BV2ForCausalLM` 两个类。这正体现了 MindFormers「模型库 = LLaMA 基类 + 各家差异补丁」的设计哲学（详见 [[昇腾 MindFormers 推理]] 第 3 节）。

```
        LlamaConfig (通用超参)
              │
     ┌────────┴─────────┐
     ▼                  ▼
Baichuan7BV2        Baichuan13BV2
 (RoPE)              (ALiBi, NormHead...)
     │                  │
     └──── model_dict 按 model_name 选 ────┘
```

> 启示：迁移一个「类 LLaMA」模型到昇腾，工作量主要在「找出与 LLaMA 的差异并打补丁」，而非从零重写。

---

## 2. 目录结构与脚本职责

```
baichuan2/
├── baichuan-inference.py   ← 交互式推理 demo（读键盘）
├── baichuan-stat.py        ← 批量性能压测（读数据集 + 掐表）
└── README.md               ← 本文

依赖（来自 MindFormers research/baichuan2/，不在本目录）：
├── baichuan2_7b.py          Baichuan7BV2ForCausalLM 定义
├── baichuan2_13b.py         Baichuan13BV2ForCausalLM 定义
├── baichuan2_tokenizer.py   Baichuan2Tokenizer (SentencePiece)
└── run_baichuan2_7b.yaml    所有配置的"总开关"
```

| 脚本 | 输入 | 核心动作 | 输出 |
| --- | --- | --- | --- |
| `baichuan-inference.py` | 键盘文本 | `generate(max_new_tokens=64)` 循环 | 屏幕打印回答 + 单次耗时 |
| `baichuan-stat.py` | 2k 条 JSON 指令 | 跑两轮（1 token / 100 token）+ 统计 | 首/增量/端到端时延 + TP 分位 |

> 注意：两个脚本顶部都 `from mindspore import context`，但只有 `baichuan-stat.py` 真正调了 `context.set_context(device_id=2, mode=0)` —— 显式指定**用第 2 号卡、跑静态图（GRAPH_MODE=0）**。`inference.py` 没设，会走默认上下文（设备/模式以环境为准）。

---

## 3. 模型加载链路：从一行 yaml 到能 `generate` 的网络

两个脚本前半段几乎逐字相同，这是 MindFormers 推理的「标准开机流程」，拆成 5 步：

```
[1] yaml 路径
    run_baichuan2_7b.yaml
        │  MindFormerConfig(path)
        ▼
[2] baichuan2_config  (一个可点属性的大字典)
        │  改 batch_size = 1
        ▼
[3] LlamaConfig(**config.model.model_config)
        │  把 yaml 里的 model_config 段解包成结构超参
        ▼
[4] model_dict[model_name](config=...)   ← 按名字选 7B/13B 类
        │  内部 from .ckpt 加载权重到 NPU
        ▼
[5] Baichuan2Tokenizer(vocab_file=...)    ← SentencePiece 分词器
        │
        ▼
    network.generate(...) 就绪
```

对应代码骨架：

```python
baichuan2_config_path = "/root/mindformers/research/baichuan2/run_baichuan2_7b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)   # [2]
baichuan2_config.model.model_config.batch_size = 1           # 在线改超参，不动文件

baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)  # [3]
model_name = baichuan2_config.trainer.model_name             # "baichuan2_7b" 或 "_13b"
baichuan2_network = model_dict[model_name](config=baichuan2_model_config)    # [4]

tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file)              # [5]
```

> 关键点 1：`batch_size=1` 是写死的。这套脚本是**单样本逐条**推理（pad 到 batch=1），不是动态批处理。要提吞吐需另做 batching（见 [[llm-inference/连续批处理]]）。
>
> 关键点 2：换 13B 不改代码，只要把 yaml 换成 `run_baichuan2_13b.yaml`、且 yaml 里 `trainer.model_name` 写成 `baichuan2_13b`，`model_dict` 自动选对类。**配置驱动，代码不动**——这就是 MindFormers 的 yaml 哲学。
>
> 关键点 3：`run_baichuan2_7b.yaml` 里必须正确填 `checkpoint_name_or_path` 指向转换好的 `.ckpt`，权重才会真正加载到 NPU（HF→MS 转换见父级 README 第 6 节）。

---

## 4. 一条 prompt 的旅程：`generate` 内部发生了什么

以 `baichuan-inference.py` 第一句为例，输入 `"可以帮我做一份旅游攻略吗？"`，要 `max_new_tokens=64`：

```
"可以帮我做一份旅游攻略吗？"
        │  tokenizer(text)["input_ids"]
        ▼
input_ids = [ ... ]  长度 L（约 10 个 token）
        │  generate() 开始计时 perf_counter()
        ▼
┌─────────── Prefill 全量阶段（compute-bound）───────────┐
│  一次性吃完 L 个 token：Embedding→N层Block→LMHead       │
│  每层算 Q,K,V，把 K,V 全部写进 KV Cache                  │
│  采样出第 1 个新 token                                   │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌─────────── Decode 增量阶段（memory-bound）×63 ──────────┐
│  只喂"上一个 token"(shape [1,1])                         │
│  复用 KV Cache，只算新 token 的 Q，与历史 K,V 做注意力   │
│  采样 → 追加进 KV Cache → 下一个                          │
└──────────────────────────────────────────────────────┘
        │  够 64 个新 token 或遇 EOS 停
        ▼  end_time = perf_counter()，打印"第一次生成时间"
outputs[0][len(input_ids):]  → tokenizer.decode → 中文回答
```

### 4.1 `use_past` 是性能命门

`generate` 能走上面那条「Prefill 一次 + Decode 增量」的高效路径，前提是 **yaml 里 `use_past: True`**（开 KV Cache）。否则每生成一个 token 都要把前面所有 token 重算一遍，复杂度从 $O(T^2)$ 退化到 $O(T^3)$ 量级，慢几十倍。这是昇腾推理「必开」的开关（见 [[llm-optimizer/kv-cache]]）。

### 4.2 为什么"第一次生成时间"单独打印

脚本里第一句之所以单独 `print("第一次生成时间：", first_gen_time)`，是因为**静态图首次执行要编译预热**（GRAPH_MODE 把整张图编译再下发），首 token 会异常慢（几秒～几十秒）。后续才进入稳态。压测时必须把这「第一次」与稳态分开看，否则平均值被污染——这也是 `baichuan-stat.py` 先单独跑一遍首句再开始统计的原因。

---

## 5. 采样参数逐字解释（两脚本通用）

```python
outputs = network.generate(inputs_ids,
    do_sample=False,        # 关采样 → 贪心(确定性), 每次输出一样, 便于压测复现
    num_beams=1,            # 不做 beam search, 只走单条路径
    top_k=1,                # 只看概率最高的 1 个候选(配 do_sample=False 即 argmax)
    top_p=1.0,              # 核采样阈值, 1.0 = 不裁剪
    repetition_penalty=1.0, # 重复惩罚, 1.0 = 不惩罚
    temperature=1.0,        # 温度, 1.0 = 不缩放 logits
    max_new_tokens=64)      # 最多新生成多少 token (压测里分别取 1 / 100)
```

| 参数 | 取值含义 | 调大/调小的效果 |
| --- | --- | --- |
| `do_sample` | `False`=贪心 / `True`=按分布采样 | 压测固定 `False` 保证**可复现**（同输入同输出） |
| `top_k` | 只在概率前 k 个里选 | =1 等价 argmax；越大越多样 |
| `top_p` | 累积概率达 p 的最小集合（核采样） | <1 砍掉长尾，更稳；=1 不裁剪 |
| `temperature` | 缩放 logits 再 softmax | <1 更保守、>1 更发散 |
| `repetition_penalty` | 对已出现 token 降权 | >1 抑制复读（交互脚本里用了 1.05） |
| `max_new_tokens` | 新 token 上限 | 压测的核心变量：=1 测首 Token，=100 测端到端 |

> 设计意图：压测全程 `do_sample=False`（贪心）是刻意的——只有**确定性输出**才能让"耗时"这个量可比、可复现；采样会引入随机长度，污染时延统计。

---

## 6. 性能压测方法学（`baichuan-stat.py` 的精髓）

把"快慢"变成数字，核心是把端到端时延拆成两段：**首 Token（Prefill）** 和 **后续每 Token（Decode）**。脚本用了一个巧妙的「两遍法」：

```
数据集 2000 条
   │
   ├── 第 1 遍：每条都 max_new_tokens = 1
   │       └─► 只触发 Prefill，量到的就是【首 Token 时延 TTFT】
   │           存入 first_token_time_list
   │
   └── 第 2 遍：每条都 max_new_tokens = 100
           └─► Prefill + 99 步 Decode，量到【端到端时延】
               存入 total_token_time_list, new_token_lens_list
```

然后用**减法**剥离出纯增量时延：

```python
# 单条的"每增量 token 时延" = (端到端 - 首token) / (实际新token数 - 1)
token_time = (total_token_time_list[i] - first_token_time_list[i]) / (new_token_lens_list[i] - 1)
```

直觉：端到端时间 = 首 Token 时间 + （新 token 数 − 1）× 每增量 token 时间。反解出每增量 token 时间，再对全体取平均，得到 `avg_token_time`（即 **TPOT，Time Per Output Token**）。脚本还对每条用 `if new_token_lens <= 1: continue` 跳过没法做除法的样本，避免除零。

### 6.1 三个核心指标 + 分位数

脚本对**首 Token 时延、端到端时延、生成 Token 长度**三组各打印 `min/max/TP50/TP90/TP99`：

```python
print("TP99：", np.percentile(np.array(first_token_time_list), 99))
```

```
   分位数为什么重要（不能只看平均）
   ┌────────────────────────────────────────┐
   │ 平均值 = 3.0s   听起来还行              │
   │ 但 TP99 = 12s  → 1% 的请求要等 12 秒    │
   │ → 在线服务体验由长尾 TP99 决定, 非均值  │
   └────────────────────────────────────────┘
```

| 指标 | 英文 | 怎么测 | 关注谁 |
| --- | --- | --- | --- |
| 首 Token 时延 | TTFT | `max_new_tokens=1` 的耗时 | 用户"等待开始"的感受 |
| 每增量 Token 时延 | TPOT | (端到端−首Token)/(新token−1) | 生成"流畅度"、吞吐基础 |
| 端到端时延 | E2E Latency | `max_new_tokens=100` 的耗时 | 整体响应快慢 |
| TP50/90/99 | 分位 | `np.percentile` | 长尾、SLA 达标率 |

> 术语对齐见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

---

## 关键公式与数值手算

### A. 端到端时延分解

设首 Token 时延 $T_\text{first}$，每增量 Token 时延 $T_\text{tpot}$，新生成 $n$ 个 token：

$$
T_\text{e2e} = T_\text{first} + (n-1)\cdot T_\text{tpot}
$$

反解（脚本所做）：

$$
T_\text{tpot} = \frac{T_\text{e2e} - T_\text{first}}{n-1}
$$

**手算示例**：若某条 $T_\text{first}=0.8\text{s}$，$T_\text{e2e}=3.76\text{s}$，$n=100$：

$$
T_\text{tpot} = \frac{3.76 - 0.8}{100-1} = \frac{2.96}{99} \approx 0.0299\text{s} \approx 29.9\text{ms/token}
$$

### B. 解码吞吐（生成阶段 token/s）

$$
\text{Throughput} = \frac{1}{T_\text{tpot}} \approx \frac{1}{0.0299} \approx 33.4\ \text{token/s}
$$

> 即每秒吐约 33 个 token（单 batch、本例假设值，真实数字以实测为准）。

### C. KV Cache 显存（Decode 是 memory-bound 的根因）

$$
\text{KV} = 2 \times L_\text{seq} \times n_\text{layer} \times n_\text{head} \times d_\text{head} \times \text{batch} \times \text{bytes}
$$

**手算 Baichuan2-7B**（$n_\text{layer}=32$, hidden $=4096 \Rightarrow n_\text{head}\times d_\text{head}=4096$, $L=2048$, batch $=1$, FP16=2B）：

$$
\text{KV} = 2 \times 2048 \times 32 \times 4096 \times 1 \times 2\,\text{B} \approx 1.07\,\text{GB}
$$

batch 提到 16 → 约 **17 GB**，这就是 `batch_size` 写死 1 的现实约束；长上下文 + 大 batch 极易 OOM。

### D. 权重显存（FP16）

$$
\text{Weights} \approx P \times 2\,\text{B}
$$

- 7B：$7\times10^9 \times 2 \approx 14\,\text{GB}$
- 13B：$13\times10^9 \times 2 \approx 26\,\text{GB}$

> 一张 32GB 的 910 跑 13B（26GB 权重）已经很紧，再叠 KV Cache 就要靠量化或张量并行（见 [[B07:llm-inference/大模型推理张量并行]]）。

### E. Prefill 的注意力计算量（compute-bound）

注意力打分 $QK^T$ 复杂度约 $O(L^2 d)$，所以 Prefill 时延随 prompt 长度 $L$ **平方级**上涨——这解释了为什么 `baichuan-stat.py` 还要统计「输入 Token 长度」分布：长输入会显著抬高 TTFT 的长尾。

---

## 7. 评价 / 对照 / 局限

| 维度 | 本目录脚本做法 | 和生产级推理引擎（vLLM/MindIE）对比 |
| --- | --- | --- |
| 批处理 | `batch_size=1` 写死，逐条 | 缺**连续批处理**，吞吐有上限（见 [[llm-inference/连续批处理]]） |
| KV Cache | 靠 yaml `use_past` 开启 | 无 **PagedAttention** 式分页管理，长序列显存碎片化（见 [[llm-inference/vllm/README]]） |
| 调度 | 同步阻塞 `generate` | 无请求级调度/抢占 |
| 量化 | 默认 FP16 | 未启 W8A8/W8A16，显存与带宽未压（见 [[llm-compression/quantization/fp8]]） |
| 用途定位 | **基准测试 / 功能验证** | 不是在线服务部署方案 |

> 一句话评价：这套脚本是**「把单卡 Baichuan2 在昇腾上跑通并量化时延」的教学/基准工具**，不是高并发服务。要上线得换 MindIE / 自研服务层并引入连续批处理 + 分页 KV + 量化。

### 7.1 7B vs 13B 选型与排错

| 项目 | Baichuan2-7B | Baichuan2-13B |
| --- | --- | --- |
| 位置编码 | RoPE | **ALiBi**（结构差异关键点） |
| FP16 权重 | 约 14 GB | 约 26 GB |
| 单卡可行性 | 32GB 卡轻松 | 32GB 卡偏紧，常需 TP=2 |
| 入口类 | `Baichuan7BV2ForCausalLM` | `Baichuan13BV2ForCausalLM` |

| 现象 | 可能原因 | 排查思路 |
| --- | --- | --- |
| 首句卡几十秒 | 静态图编译预热 | 正常，做 warmup；统计排除首次 |
| Device OOM | KV Cache/权重超显存 | 降 `seq_length`、量化、上 `model_parallel`（TP） |
| 输出乱码 | 权重转换转置/命名错 | 逐层比 logits（父 README 第 6 节） |
| `model_dict` KeyError | yaml `trainer.model_name` 拼错 | 必须是 `baichuan2_7b`/`baichuan2_13b` |
| 加载即报版本错 | CANN↔MindSpore↔MindFormers 不配套 | 严格按官方配套表对齐（以官方文档为准） |
| 13B 结果不对但 7B 对 | 误用 RoPE 处理 ALiBi | 确认用了 13B 专属类 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航总入口
- [[昇腾 MindFormers 推理]] — 上层套件全景（软件栈/加载/迁移），本文是它的 Baichuan2 实战分册
- [[llm-optimizer/kv-cache]] — KV Cache 原理与显存账（`use_past` 背后）
- [[llm-inference/连续批处理]] — 突破 `batch_size=1` 吞吐上限的关键
- [[llm-inference/vllm/README]] — PagedAttention 与生产级推理引擎对照
- [[B07:llm-inference/大模型推理张量并行]] — 13B 放不下时的 TP 切分
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] — TTFT / TPOT / TP99 术语对齐
- [[llm-compression/quantization/fp8]] — 进一步压显存与带宽

## 参考文档

- baichuan2 model card: https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan2/baichuan2.md
- 父级总览：`../README.md`（昇腾 MindFormers 推理）
