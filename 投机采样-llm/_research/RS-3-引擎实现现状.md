# RS-3　投机解码在主流推理引擎里的真实实现现状

> **性质**：调研笔记（`_research/`，不参与双链、不走十段式模板）。供 `20-主流引擎实现-vLLM与SGLang与TensorRTLLM`、`19-负收益全解`、`23-与其它优化的相互作用` 三篇取用。
> **核验日期**：2026-08-22。所有版本号、参数名、默认值均标注了取证来源。
> **纪律**：参数名与默认值一律抄自实际页面/源码；抄不到写「未查证」。加速比数字按 §写作规范 铁律二尽量补齐七项口径，缺项显式标「原文未给出」。

---

## 0. 一张总表：谁支持什么（截至 2026-08-22）

| 引擎 | 版本锚点 | ngram / lookup | 独立 draft model | **Medusa** | EAGLE / EAGLE3 | MTP | 其它自有方法 |
|---|---|---|---|---|---|---|---|
| **vLLM** | v0.27.1（2026-08-11） | ✅ `ngram`、`ngram_gpu` | ✅ `draft_model` | ⚠️ 代码在、**无文档** | ✅ `eagle` / `eagle3` | ✅ `mtp`（22 个模型族别名） | `suffix`、`dflash`、`dspark`、`mlp_speculator`、`custom_class`、`extract_hidden_states`、PARD（`parallel_drafting`） |
| **SGLang** | v0.5.18（2026-08-22） | ✅ `NGRAM` | ✅ `STANDALONE` | ❌ **全仓库零实现** | ✅ `EAGLE` / `EAGLE3` | ✅（`NEXTN`，复用 EAGLE 通路） | `DFLASH`、`DSPARK`、`FROZEN_KV_MTP`(内部)、multi-layer EAGLE、自定义注册 |
| **TensorRT-LLM** | `1.3.0rc25`（正式版 v1.2.1，2026-04-20） | ✅ `NGram`、`SA`(后缀自动机) | ✅ `DraftTarget` | ❌ **1.2 起随 TRT engine 后端一起被删** | ⚠️ 只有 `Eagle3`；**EAGLE-1/2 已删**，传 `Eagle` 自动转 Eagle3 | ✅ `MTP` | `PARD`、`DFlash`、`DSpark`、`User_Provided`、`AUTO`、`SaveState`；**ReDrafter / Lookahead 已随 TRT engine 一起删除** |
| **llama.cpp** | master `b10573`（2026-08-22） | ✅ **5 种**（`ngram-simple/map-k/map-k4v/mod/cache`） | ✅ `draft-simple` | ❌ 无 | ✅ `draft-eagle3`（无 EAGLE-2） | ✅ `draft-mtp` | `draft-dflash`、`draft-dspark` |
| **HF transformers** | 5.15.1 | ✅ `prompt_lookup_num_tokens` | ✅ `assistant_model` | ❌ 无 | ❌ **无** | ✅ `use_mtp` | UAG 跨 tokenizer、`assistant_early_exit` 早退自投机、`speculation_type="dflash"`、Gemma4 SinglePositionMultiToken |

**从这张表能直接读出的三条结论**：

1. **Medusa 已经出局。** 三家服务引擎里 SGLang 与 llama.cpp 从未实现、vLLM 保留代码但已撤掉文档页。写 `12-Medusa-多头草稿与树注意力的诞生` 的"它被推翻了什么"时，这是硬证据——**Medusa 的历史地位（把树注意力带进投机解码）保住了，但它作为一个可部署方法已经死了。**
2. **EAGLE-3 是唯一的跨引擎共识。** vLLM / SGLang / llama.cpp（+ §3 的 TensorRT-LLM）全部实现。**唯一的例外是 HF transformers——它至今没有 EAGLE**（`grep -i "eagle\|medusa"` 在三个 generation 源文件里命中 0）。
3. **2026 年冒出来的新家族是 DFlash / DSpark**（块并行草稿 + 置信度调度），**四家全部在同一年内跟进**。这是 `24-前沿进展-2025到2026` 必须写的一条主线。

### 0b. 2026 新家族速查（`24-前沿进展` 用）

| 方法 | 论文 | 机制一句话 | vLLM | SGLang | TRT-LLM | llama.cpp | HF |
|---|---|---|---|---|---|---|---|
| **PARD**（Parallel Draft） | arXiv:2504.18583（2025-04） | target-**无关**；用 mask token 一次前向出 K 个草稿，草稿模型本身要为并行草稿训练过 | ✅ `parallel_drafting: true` | ✅ P-EAGLE（SpecForge 可训，arXiv:2602.01469） | ✅ `PARD` | ❌ | ❌ |
| **DFlash** | arXiv:2602.06036 | target-**相关**；块扩散式一次出整块草稿，把 target 若干层的 hidden states 注入草稿模型的 attention | ✅ `dflash` | ✅ `DFLASH` | ✅ `DFlash` | ✅ `draft-dflash` | ✅ `speculation_type="dflash"` |
| **DSpark** | arXiv:2607.05147 / DeepSeek DeepSpec | DFlash + **Markov 头 + confidence 头**；半自回归块草稿，置信度用于调度验证预算 | ✅ `dspark`（配 `enable_adaptive_verification`） | ✅ `DSPARK` | ✅ `DSpark` | ✅ `draft-dspark` | ❌ |
| **Suffix / 后缀自动机** | Suffix Decoding arXiv:2411.04975（2024-11） | 无模型；跨请求全局后缀树/自动机，按频率提议，草稿长度自适应 | ✅ `suffix`（需 Arctic Inference） | ✅ `NGRAM` 的 trie 变体 | ✅ `SA`（GPU 原生后缀自动机） | ✅ `ngram-mod`（跨 slot 共享 hash pool） | ❌ |
| **Domino** | arXiv:2605.29707 | SpecForge 的训练法（D-PACE loss） | — | ✅ 训练侧 | — | — | — |

**三条观察**：
1. **DFlash 是唯一被五家全部实现的 2026 新方法**（HF 也有）。
2. **"块并行草稿"（一次出整块而非逐 token 自回归）是 2026 的共同方向**——PARD / DFlash / DSpark 都是。它解决的是 EAGLE 系"草稿本身也要跑 K 次前向"的开销。
3. **无模型草稿在 2026 反而更强了**（suffix / 后缀自动机 / 跨请求共享），因为 agentic / RL rollout 负载重复度极高。**vLLM 的 suffix decoding 直接依赖 Snowflake 的 Arctic Inference（`pip install arctic-inference==0.1.1`），这是少见的"官方引擎依赖第三方投机实现"。**

（§6 由并行调研补齐。）

---

## 1. vLLM（重点）

### 1.1 版本锚点与取证来源

- 最新 release：**v0.27.1，发布于 2026-08-11**（`v0.27.0` 2026-08-10、`v0.26.0` 2026-07-27、`v0.25.0` 2026-07-11、`v0.24.0` 2026-06-29、`v0.23.0` 2026-06-15）。
  来源：`https://api.github.com/repos/vllm-project/vllm/releases`（2026-08-22 拉取）
- 本节的参数名/默认值主要抄自两处**源码与文档原文**：
  - `vllm/config/speculative.py` @ tag `v0.27.1` — https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/config/speculative.py
  - `docs/features/speculative_decoding/*.md` @ `main` — https://github.com/vllm-project/vllm/tree/main/docs/features/speculative_decoding
  - 渲染版：https://docs.vllm.ai/en/latest/features/speculative_decoding/
- ⚠️ 文档目录在 2026 年已从旧的单文件 `docs/features/spec_decode.md`（URL `features/spec_decode.html`）拆成了目录 `docs/features/speculative_decoding/`（URL `features/speculative_decoding/`）。**旧 URL 已失效**，网上大量教程仍指向旧页。

### 1.2 V1 架构下投机解码是怎么组织的

vLLM V1 自 v0.8.0 起为默认引擎。V1 的调度器把 prompt token 与 output token 一视同仁，这是投机解码能和 chunked prefill / prefix caching 共存的结构性原因。官方原话（`docs/usage/v1_guide.md`）：

> vLLM V1's unified scheduler treats both prompt and output tokens the same way by using a simple dictionary (e.g., `{request_id: num_tokens}`) to dynamically allocate a fixed token budget per request, enabling features like chunked prefills, prefix caching, and speculative decoding **without a strict separation between prefill and decode phases**.

V1 功能状态表里，`**Spec Decode** | 🟢 Functional`（同文件）。

**统一入口**：所有投机相关配置都收拢进一个 JSON 对象 `--speculative-config`（别名 `-sc`），Python 侧是 `LLM(..., speculative_config={...})`。

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<draft-model>",
    "num_speculative_tokens": 5
  }'
```
（原文抄自 `docs/features/speculative_decoding/README.md`）

v0.27.1 的 `arg_utils.py` 里还新增了三个**快捷 flag**，与 `--speculative-config` 的同名 key **互斥**（同时给会报 `--spec-method and --speculative-config['method'] are mutually exclusive`）：

| flag | 映射到 |
|---|---|
| `--spec-method` | `speculative_config.method` |
| `--spec-model` | `speculative_config.model` |
| `--spec-tokens` | `speculative_config.num_speculative_tokens` |

来源：`vllm/engine/arg_utils.py` @ v0.27.1，`create_speculative_config()`。

**旧写法已废弃**（文档 warning 原文）：

> Note: Please use `--speculative-config` to set all configurations related to speculative decoding. The previous method of specifying the model through `--speculative-model` and adding related parameters such as `--num-speculative-tokens` separately has been deprecated.

### 1.3 支持的 method 全清单（抄自源码 Literal）

`vllm/config/speculative.py` @ v0.27.1 的类型定义，**原样**：

```python
MTPModelTypes = Literal[
    "deepseek_mtp", "mimo_mtp", "mimo_v2_mtp", "glm4_moe_mtp", "glm4_moe_lite_mtp",
    "glm_ocr_mtp", "ernie_mtp", "nemotron_h_mtp", "exaone_moe_mtp", "exaone4_5_mtp",
    "qwen3_next_mtp", "qwen3_5_mtp", "longcat_flash_mtp", "minimax_m3_mtp",
    "bailing_hybrid_mtp", "mtp", "kimi_k3_mtp", "pangu_ultra_moe_mtp",
    "step3p5_mtp", "hy_v3_mtp", "gemma4_mtp", "inkling_mtp",
]
NgramGPUTypes = Literal["ngram_gpu"]
DFlashModelTypes = Literal["dflash"]
DSparkModelTypes = Literal["dspark"]
EagleModelTypes = Literal[
    "eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes
]
SpeculativeMethod = Literal[
    "ngram", "medusa", "mlp_speculator", "draft_model", "suffix", "custom_class",
    EagleModelTypes, NgramGPUTypes, DSparkModelTypes,
]
RejectionSampleMethod = Literal["standard", "synthetic", "block"]
DraftSampleMethod = Literal["greedy", "probabilistic"]
```

要点：

1. **所有 `*_mtp` 别名已被折叠成 `mtp`**。源码里显式打日志：
   ```python
   if self.method in get_args(MTPModelTypes) and self.method != "mtp":
       logger.warning("method `%s` is deprecated and replaced with mtp.", self.method)
       self.method = "mtp"
   ```
   → 网上教程里的 `"method": "deepseek_mtp"` 仍能跑，但会打 deprecation 警告。
2. `medusa` **仍在** `SpeculativeMethod` 里（没被删），但**文档目录里已经没有 Medusa 专页**——`docs/features/speculative_decoding/` 下只有 EAGLE / MTP / draft_model / PARD / MLP / n-gram / suffix / hidden-state 八篇。源码里唯一的 Medusa 特判是给老格式 checkpoint 注入 `model_type`：
   ```python
   # Old-format Medusa checkpoints (e.g. FasterDecoding/medusa-*) lack a
   # model_type key in config.json, so AutoConfig cannot detect them.
   if self.method == "medusa":
       draft_hf_overrides = {"model_type": "medusa"}
   ```
   → **判断：Medusa 在 vLLM 里处于"能用但不推荐、无文档"的状态**（截至 v0.27.1）。
3. **method 推断规则**（`__post_init__` 原文逻辑）：不给 `method` 时，若 `model` 形如自定义类路径 → `custom_class`；若 `model in ("ngram", "[ngram]")` → `ngram`；**否则一律当 `draft_model`**。

### 1.4 完整配置字段与默认值（抄自 v0.27.1 源码）

> 下表的 Default 列**逐条抄自 `SpeculativeConfig` 的 dataclass 字段声明**，不是文档里的转述。

#### 通用

| 字段 | 类型 | 默认值 | 说明（源码 docstring 摘要） |
|---|---|---|---|
| `method` | `SpeculativeMethod \| None` | `None` | 不给则自动推断（见 1.3） |
| `model` | `str \| None` | `None` | draft model / eagle head / 附加权重路径 |
| `num_speculative_tokens` | `int`（`Field(gt=0)`） | `None` | 无默认；可从 draft config 的 `n_predict` 推出，否则必填 |
| `draft_tensor_parallel_size` | `int`（`ge=1`） | `None` | **只能是 1 或 target 的 TP size**，否则 raise |
| `tensor_parallel_size` | `int \| None` | `None` | **陷阱字段**：存在只为报错。传了就 raise `'tensor_parallel_size' is not a valid argument in the speculative_config. Please pass 'draft_tensor_parallel_size' instead.` |
| `max_model_len` | `int`（`ge=1`） | `None` | draft 的最大长度；不给则取 `min(draft_max_model_len, target_max_model_len)` |
| `quantization` | `QuantizationMethods \| str \| None` | `None` | draft 权重量化方式 |
| `moe_backend` | `MoEBackend \| None` | `None` | draft 的 MoE 后端；`None` 继承 target |
| `attention_backend` | `AttentionBackendEnum \| None` | `None` | draft 的 attention 后端。docstring 点名：「DFlash needs a non-causal-capable backend like FLASH_ATTN」 |
| `kv_cache_dtype` | `CacheDType \| None` | `None` | draft 的 KV cache dtype；`None` 继承 target |
| `revision` / `code_revision` | `str \| None` | `None` | draft 的 HF revision |
| `enforce_eager` | `bool \| None` | `None` | 覆盖 model_config 的 enforce_eager |
| `draft_load_config` | `LoadConfig \| None` | `None` | 不给则用 target 的 load config |

#### 草稿与采样策略

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `draft_sample_method` | `Literal["greedy","probabilistic"]` | **`"greedy"`** | `greedy` 取 argmax，拒绝采样时把 draft 概率当 one-hot；`probabilistic` 从 draft 分布随机采样并用完整 draft logits 做概率比检验，**代价是额外显存** |
| `rejection_sample_method` | `Literal["standard","synthetic","block"]` | **`"standard"`** | `block` = block verification（Sun et al.）整块联合验证；`synthetic` 是按标定的接受率**人为接受**（用于性能建模，非无损） |
| `synthetic_acceptance_rates` | `list[float] \| None` | `None` | 仅 `synthetic` 有效。长度须 == `num_speculative_tokens`，每项 ∈[0,1]，**必须单调不增** |
| `synthetic_acceptance_length` | `float \| None` | `None` | 仅 `synthetic` 有效，∈`[1, num_speculative_tokens+1]`，与上一项**互斥** |
| `parallel_drafting` | `bool` | **`False`** | PARD / P-EAGLE：一次并行出全部草稿而非逐个自回归。**只兼容 EAGLE 与 draft_model**；`dflash`/`dspark` 会被强制置 `True` |
| `disable_padded_drafter_batch` | `bool` | **`False`** | 允许 draft batch 内变长；**只影响 EAGLE** |
| `use_local_argmax_reduction` | `bool` | **`False`** | 用 vocab-parallel 局部 argmax 代替 all-gather 全 logits，通信从 O(vocab) 降到 O(2·tp_size)。**只对贪心、非树形草稿生效** |
| `use_heterogeneous_vocab` | `bool` | **`False`** | 允许 draft/target 词表不同（TLI 算法）。**只兼容 `method="draft_model"`**，且**只支持 `draft_sample_method="greedy"`** |
| `enable_adaptive_verification` | `bool` | **`False`** | 自适应验证预算，**目前仅 DSpark 且需 confidence head** |
| `dspark_draft_topk` | `int`（`ge=1`） | `None` | Qwen3 DSpark 的 Markov 投影 top-k |

#### N-gram 专属

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt_lookup_max` | `int`（`ge=1`） | `None` → **两者都不给时置 5** | 最大 n-gram 窗口 |
| `prompt_lookup_min` | `int`（`ge=1`） | `None` → **两者都不给时置 5** | 最小 n-gram 窗口 |

> ⚠️ **文档与源码打架，以源码为准**：`prompt_lookup_min` 的 docstring 写的是「Defaults to 1.」，但 `__post_init__` 的实际逻辑是：
> ```python
> if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
>     # TODO(woosuk): Tune these values. They are arbitrarily chosen.
>     self.prompt_lookup_min = 5
>     self.prompt_lookup_max = 5
> ```
> 只给其中一个时，另一个**镜像**成相同值。`min > max` 直接 raise。
> 源码里那句 `They are arbitrarily chosen`（"这两个值是随便选的"）值得原样引用——说明这个默认值**没有调优依据**。

#### Suffix decoding 专属

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `suffix_decoding_max_tree_depth` | `int` | **`24`** | 前缀匹配长度 + 投机长度之和的上限 |
| `suffix_decoding_max_cached_requests` | `int` | **`10000`** | 全局 suffix tree 缓存的请求数，超出 FIFO 淘汰；**设 `0` 关闭全局树**（只留 prompt 树） |
| `suffix_decoding_max_spec_factor` | `float` | **`1.0`** | `max_spec_tokens = max_spec_factor × prefix_match_length` |
| `suffix_decoding_min_token_prob` | `float` | **`0.1`** | 低于此频率估计概率的 token 不投机 |

依赖：**需要装 Arctic Inference**。源码 raise 原文：
> `Arctic Inference is required for suffix decoding. Install via 'pip install arctic-inference==0.1.1'.`

不给 `num_speculative_tokens` 时会 fallback 到 `suffix_decoding_max_tree_depth`（即 24）并打 warning。文档建议：

> Suffix Decoding will speculate a dynamic number of tokens for each request at each decoding step, so the `num_speculative_tokens` configuration specifies the *maximum* number of speculative tokens. It is suggested to use a high number such as `16` or `32` (default).

#### 动态投机（Dynamic SD）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `num_speculative_tokens_per_batch_size` | `list[tuple[int,int,int]] \| None` | `None` | 每项 `(range_start, range_end, num_speculative_tokens)`，闭区间 |

文档给的例子（原文）：
```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [[1, 64, 3], [65, 128, 1], [129, 512, 0]]
  }'
```
含义：并发 1–64 用 K=3，65–128 用 K=1，**129–512 用 K=0（即完全不产草稿）**。

#### 观测

| flag | 取值 | 默认 |
|---|---|---|
| `--per-request-spec-decode-metrics` | `none` / `summary` / `detailed` | **`none`** |

`summary` 会在响应的 `metrics.speculative_decoding` 里返回 `mean_acceptance_length`、`draft_acceptance_rate`、`acceptance_histogram`、`num_spec_steps`、`num_accepted_draft_tokens`、`num_draft_tokens`、`num_spec_tokens`；`detailed` 额外给 `per_step_accepted` / `per_step_drafted` 两个逐步数组。

服务端 Prometheus 计数器（与上面逐请求字段对应）：
`vllm:spec_decode_num_drafts_total`、`vllm:spec_decode_num_draft_tokens_total`、`vllm:spec_decode_num_accepted_tokens_total`；离线侧还有 `vllm:spec_decode_num_accepted_tokens_per_pos`（Vector 类型，逐位置接受计数）。

官方算平均接受长度的口径（抄自 `examples/features/speculative_decoding/spec_decode_offline.py`）：

```python
acceptance_length = 1 + (num_accepted_tokens / num_drafts)
```

即 **含 bonus token**，取值范围 `[1.0, num_spec_tokens + 1]`。这一口径务必写进 `05-接受率alpha-定义口径与怎么测`——它和很多论文里"平均接受的草稿 token 数"差 1。

### 1.5 草稿模型/草稿头从哪来

| method | 权重来源 | 需要训练吗 |
|---|---|---|
| `ngram` / `ngram_gpu` / `suffix` | 无权重，纯算法 | 否 |
| `mtp` | **target checkpoint 内自带**。源码：`self.model = self.target_model_config.model` | 否（模型预训练时已带 MTP 头） |
| `dspark` | 同上，权重在 target checkpoint 里 | 否 |
| `eagle` / `eagle3` | 独立 HF 仓库 | 是（但有大量现成权重） |
| `mlp_speculator` | `ibm-ai-platform/*-accelerator`、`ibm-granite/*-accelerator` | 是（IBM 已放出 9 个） |
| `draft_model` | 任何同族小模型（如 `Qwen/Qwen3-8B` + `Qwen/Qwen3-0.6B`） | 否 |
| PARD | `amd/PARD-Qwen3-0.6B` 等，`https://huggingface.co/collections/amd/pard` | 是（AMD 已放出） |

**EAGLE 官方权重来源**（文档点名两处）：
- `https://huggingface.co/collections/RedHatAI/speculator-models`
- `https://huggingface.co/yuhuili/models?search=eagle`

文档给的实际 model id 举例：`yuhuili/EAGLE-LLaMA3-Instruct-8B`、`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`、`RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3`、`nvidia/gpt-oss-120b-Eagle3-v2`。

**vLLM 官方训练框架 = `vllm-project/speculators`**（对标 SGLang 的 SpecForge）：
- 文档：https://docs.vllm.ai/projects/speculators/en/latest/ ；仓库：https://github.com/vllm-project/speculators
- 安装 `pip install speculators`。PyPI 最新版 **0.7.0.1（2026-08-13）**；前几版 0.7.0（2026-07-30）、0.6.0.1（2026-08-12）。PyPI summary 原文：**"A unified library for creating, representing, and storing speculative decoding algorithms for LLM serving such as in vLLM."** 维护方是 **Red Hat**
- 版本节奏对照 vLLM blog：v0.3.0（2025-12-13 blog）→ **v0.5.0 加入 DFlash 支持与 online training**（2026-05-28 blog）→ 0.7.x（2026-07/08）
- 能力（抄自 `docs/features/speculative_decoding/speculators.md`）：
  > **Offline training data generation using vLLM**: Enable the generation of hidden states using vLLM. Data samples are saved to disk and can be used for draft model training.
  > **Draft model training support**: E2E training support of single and multi-layer draft models. Training is supported for both non-MoE and MoE models.
  > **Standardized, extensible format**: Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories into a standard speculators format.
- 支持的草稿架构：文档页点名 **EAGLE-3 / DFlash / P-EAGLE**（**未见提及 Medusa 或 HASS**）
- checkpoint 里带一个 `speculators_config` 字段，vLLM 读它来同时装配 speculator 与 verifier
- **最省事的用法**：speculators 格式的模型可以**直接当 `--model` 传**，vLLM 会自动识别并把真正的 target 换进来：
  ```bash
  vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
  ```
  这条路径在源码里是 `maybe_override_with_speculators()`（`arg_utils.py`），它在 `ModelConfig` 创建**之前**跑，会同时改写 `model` / `tokenizer` / `speculative_config` 三者。注释里点名一个坑：
  > Skip speculator detection for cloud storage models (eg: S3, GCS) since HuggingFace cannot load configs directly from S3 URLs. S3 models can still use speculators with explicit `--speculative-config`.

**另有一条"自己造训练数据"的官方通道**：`method="extract_hidden_states"`。它不是投机方法，而是借投机的配置口子让 vLLM 在推理时把中间层激活 dump 成 `.safetensors`，用来训 EAGLE 系草稿头。配置示例（原文）：

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/dev/shm/hidden_states"}}'
```
该页最后一句 note 是**硬限制**：
> Chunked prefill is not compatible with this feature and must be disabled.

### 1.6 与连续批处理 / chunked prefill / prefix caching / CUDA graph / PD 分离的相互作用

#### 官方 Feature × Feature 兼容矩阵（抄自 `docs/features/README.md` @ v0.27.1）

SD 行/列的取值：

| 与 SD 组合 | 结论 | 备注 |
|---|---|---|
| Chunked Prefill (CP) | ✅ | |
| Automatic Prefix Caching (APC) | ✅ | |
| **LoRA** | **❌** | 无 tracking issue 链接 |
| CUDA graph | ✅ | |
| pooling 模型 | ❌ | |
| enc-dec 模型 | ❌ | issue #7366 |
| logprobs | ✅ | |
| prompt logprobs | ✅ | |
| **async output processing** | **❌** | |
| multi-step | ❌ | |
| multimodal (mm) | ❔ Unknown/TBD | |
| **best-of** | **❌** | issue #6137 |
| **beam-search** | **❌** | issue #6137 |
| **prompt-embeds** | **❌** | |

硬件侧：SD 在 Volta/Turing/Ampere/Ada/Hopper/AMD/Intel GPU 上 ✅，**CPU 上 ❌**（对应 issue #28384「Enabling draft model based speculative decoding for CPUs」，截至核验日仍是 Feature request）。

#### 源码里写死的相互作用（`vllm/config/vllm.py` @ v0.27.1）

这些是矩阵没覆盖、但**会在启动时真的触发**的规则，比矩阵更有工程价值：

1. **异步调度（async scheduling）只对部分 method 开放**。显式开 `--async-scheduling` 时：
   ```python
   raise ValueError(
       "Currently, async scheduling is only supported "
       "with EAGLE/MTP/Draft Model/NGram GPU/DSpark kind of speculative decoding")
   ```
   即：`eagle` / `eagle3` / `mtp`（含所有别名）/ `dflash` / `extract_hidden_states` / `ngram_gpu` / `draft_model` / `dspark` 可以；**普通 `ngram`、`suffix`、`medusa`、`mlp_speculator`、`custom_class` 不行**。
   不显式指定时（`async_scheduling is None`）则自动关闭并打 warning：
   > `Async scheduling not supported with %s-based speculative decoding and will be disabled.`

2. **async scheduling 与 `disable_padded_drafter_batch=True` 互斥**（显式开就 raise，否则自动关 async）。

3. **异步投机会强制关掉 cascade attention**：
   > `Disabling cascade attention (not yet compatible with async speculative decoding).`
   → 这条对长共享前缀场景是**实打实的性能代价**：开投机 + 异步调度会失去 cascade attention 的收益。

4. **KV sharing fast prefill 与 EAGLE 硬冲突**（raise）：
   > `Fast prefill optimization for KV sharing is not compatible with EAGLE as EAGLE requires correct logits for all tokens while fast prefill gives incorrect logits for prompt tokens.`
   这句话的机理值得写进正文：**EAGLE 需要 prompt 全部 token 的正确 logits/hidden states**，任何"prefill 时偷懒只算最后一个位置"的优化都会和它冲突。

5. **chunked local attention + EAGLE 会强制关掉 hybrid KV cache manager**（`need_disable_hybrid_kv_cache_manager = True`）。

6. **spec decode 存在时关掉 `fast_moe_cold_start`**：
   > `# this config is unsafe if any spec decoding draft model has a MOE. We'll conservatively turn it off if we see spec decoding.`

7. **DeepSeek-V3.2（`deepseek_v32`）的 MTP 强制 `enforce_eager = True`**：
   ```python
   # FIXME(luccafong): cudagraph with v32 MTP is not supported, remove this when the issue is fixed.
   self.enforce_eager = True
   ```
   → **这是"开了 MTP 就失去 CUDA graph"的具体实例**，对 DeepSeek 部署影响很大。

8. **Dynamic SD 与 CUDA graph**：非 V2 model runner 下会把 cudagraph_mode 降级：
   > `Dynamic speculative decoding changes the target verification length at runtime. Overriding cudagraph_mode from %s to PIECEWISE for reliability. Use VLLM_USE_V2_MODEL_RUNNER=1 if you want to use full CUDA graphs.`

9. **Dynamic SD 与 data parallel 不兼容**（自动降级，不报错）：
   > `Dynamic speculative decoding is not supported with data parallelism because data-parallel ranks can select different speculative-token counts, causing DP divergence and deadlocks. Disabling num_speculative_tokens_per_batch_size and falling back to static num_speculative_tokens=%d.`

10. **`dspark` 与 `dflash`（混合 sliding/full attention）强制走 V2 model runner**（`use_v2_model_runner` 返回 True）。

11. **MLA DSpark 不支持 decode context parallel**：
    > `MLA DSpark does not currently support decode context parallelism; set decode_context_parallel_size=1.`

#### ⭐ 容易漏掉的一层：V1 model runner vs **V2 model runner**

v0.27.1 里 V1 引擎内部**又分了两套 model runner**，由环境变量 `VLLM_USE_V2_MODEL_RUNNER` 控制（不设时按模型走默认表；`dspark`、混合 sliding/full 的 `dflash`、diffusion 模型会被**强制**切到 V2）。
`VllmConfig._get_v2_model_runner_unsupported_features()` 列出了 **V2 runner 目前不支持的投机相关组合**（源码原文逻辑）：

```python
if speculative_config is not None:
    # TODO: ngram / ngram_gpu are not supported by the v2 model runner yet
    if speculative_config.method in ("ngram", "ngram_gpu"):
        unsupported.append("ngram/ngram_gpu speculative decoding")
    elif speculative_config.method not in ("eagle", "eagle3", "mtp", "dflash", "dspark"):
        unsupported.append(f"speculative method '{speculative_config.method}'")

    # V2 EagleSpeculator does not support parallel_drafting (for P-Eagle).
    if (speculative_config.parallel_drafting
            and speculative_config.method not in ("dflash", "dspark")):
        unsupported.append("parallel drafting for EAGLE speculative decoding")

    if (speculative_config.method == "eagle3"
            and self.parallel_config.pipeline_parallel_size > 1):
        unsupported.append("EAGLE3 with pipeline parallelism")
```

翻译成可判定的话：

| 想用 | V2 runner 下 |
|---|---|
| `ngram` / `ngram_gpu` | **不支持** |
| `draft_model` / `medusa` / `mlp_speculator` / `suffix` / `custom_class` | **不支持**（只有 `eagle`/`eagle3`/`mtp`/`dflash`/`dspark` 在白名单里） |
| EAGLE + `parallel_drafting`（P-EAGLE） | **不支持**（`dflash`/`dspark` 例外，它们原生并行草稿） |
| **EAGLE3 + pipeline parallel (PP>1)** | **不支持** |

反过来，`dynamic_speculative_decoding.md` 的 Limitations 又说：
> Full Cudagraph only works with **Model Runner V2**. MRv1 only supports piece-wise cuda graph with this feature.

**于是出现一个真实的两难**：Dynamic SD 想要 full CUDA graph 就得上 V2，但 V2 又把 ngram/draft_model 等一半 method 排除在外；而官方 Dynamic SD 的 EAGLE 示例命令行里写的偏偏是 `VLLM_USE_V2_MODEL_RUNNER=0`。
另外源码注释点明：V2 不支持时**是 raise 而不是静默回退**——
> If V2 is unsupported for the rest of the config, `_validate_v2_model_runner` raises rather than silently falling back to V1 (which can't run dspark).

#### Pipeline parallel

文档 `Known Feature Incompatibility` 节原文，只有两条：

> 1. Pipeline parallelism is not composable with speculative decoding as of `vllm<=0.15.0`
> 2. Speculative decoding with draft models is not supported in `vllm<=0.10.0`

⚠️ 注意口径：这两句写的是 `vllm<=0.15.0` 和 `vllm<=0.10.0`，而当前 release 已是 **v0.27.1**。字面读法是「0.15.0 及以前不兼容 PP」。

我在 v0.27.1 的 `VllmConfig` 校验里**没有找到**「PP>1 + 任意投机方法」的硬性拦截（`grep pipeline_parallel_size` 只命中 `enable_return_routed_experts`、sequence parallelism、external_launcher、以及 V2 runner 的 EAGLE3 三处）。同时两处**局部**限制是确凿的：

- V2 model runner：`EAGLE3 with pipeline parallelism` → unsupported（见上一节源码）
- `adaptive_verification.md`：「Not supported with LoRA ..., **pipeline parallelism** (cost curves and confidences exist only on the last rank), or output logprobs (to be fixed).」

**保守结论**：PP × SD 在 v0.27.1 上**不再是全局禁止**，但 EAGLE3+PP（V2 runner）与 adaptive verification+PP 明确不可用。文档那句 `vllm<=0.15.0` 是历史表述，**具体到 EAGLE/MTP + PP 在 v0.27.1 上能否正常工作，未查证到官方明确声明，请勿在正文里断言**。

#### PD 分离（disaggregated prefill）

查了 `docs/features/disagg_prefill.md` 与 `docs/features/automatic_prefix_caching.md` 全文，**均未出现 "speculative" 字样**。官方兼容矩阵也不覆盖 PD。
→ **结论：vLLM 官方文档对 "PD 分离 × 投机解码" 没有任何明确表述（未查证）**。不要在正文里替它下结论。

### 1.7 采样兼容性

| 采样项 | 开投机时 | 证据 |
|---|---|---|
| `temperature` | ✅ 支持（含 T>0 随机采样） | 所有官方示例都用 `SamplingParams(temperature=0.8, top_p=0.95)` |
| `top_p` / `top_k` | ✅ **实际支持**，但有性能警告 | 见下 |
| `logprobs` / `prompt logprobs` | ✅（兼容矩阵为 ✅）；但**adaptive verification 不支持 output logprobs**（"to be fixed"） | 兼容矩阵 + `adaptive_verification.md` |
| `n > 1` | 引擎层支持，但**逐请求接受率指标为 `null`** | `acceptance_metrics.md`：「they are reported only for single-sequence requests and are `null` for `n > 1`」 |
| `best_of` | ❌ 与 SD 不兼容（且 V1 已整体移除 `best_of`，RFC #13361） | 兼容矩阵 + `v1_guide.md` |
| `beam search` | ❌ 与 SD 不兼容（issue #6137） | 兼容矩阵 |
| `min_p` / `logit_bias` | ⚠️ **大概率不作用于被验证的草稿位置**。v0.27.1 `rejection_sampler.py::apply_sampling_constraints()` 只做了 temperature 缩放 + `apply_top_k_top_p`，**全文件 grep 无 `min_p` / `logit_bias`**。二手说法称「min_p and logit_bias are not yet supported with speculative decoding」，但**未定位到官方文档里的这句原文，标未查证**；建议按"草稿位置不生效、bonus token 位置生效"理解，正文引用前实测 | 源码 grep |

**top-k/top-p 的一个"文档 vs 代码"矛盾（值得写进坑清单）**：
`vllm/v1/sample/rejection_sampler.py` 的类 docstring 说：

> For example, we can use top_p, top_k sampling for bonus tokens, while spec decode **does not support these sampling strategies**.

但同一文件里的 `apply_sampling_constraints()` 明确对 target logits 依次做了 temperature 缩放 + `apply_top_k_top_p(logits, top_k, top_p)`。
→ **以代码为准：top-k/top-p 是生效的**；docstring 是陈旧残留。同一函数还留了一条性能告警：

> NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask, which is slow for large vocab sizes. This may cause performance issues.

→ **推论（可写进 19 篇）**：大词表模型（如 Qwen3 的 15 万词表）开 top-p/top-k + 投机解码时，排序开销会被草稿长度 K 放大 K 倍，是一条隐藏的负收益来源。

另有硬上限：`MAX_SPEC_LEN = 128`（rejection_sampler.py 常量，单步单请求最多 128 个草稿 token）。

**关于"无损"的官方口径**（对应写作规范铁律一）。`README.md` 的三层表述：

> 1. **Theoretical Losslessness** — Speculative decoding sampling is theoretically lossless up to the precision limits of hardware numerics.
> 2. **Algorithmic Losslessness** — vLLM's implementation of speculative decoding is algorithmically validated to be lossless.（列了 Rejection Sampler Convergence 和 Greedy Sampling Equality 两类测试）
> 3. **vLLM Logprob Stability** — vLLM does not currently guarantee stable token log probabilities (logprobs). This can result in different outputs for the same request across runs.

→ vLLM 自己的分层正好对应本库的 **L1（分布无损）/ L2（贪心等价）**，可以直接引用作为口径的官方背书。同页还点明两个"结果不一样"的合法来源：**Floating-Point Precision** 与 **Batch Size and Numerical Stability**。

`speculators.md` 里那句宣传语则是**社区高频误解的源头之一**，需要在 `07-无损的三种口径` 里点评：
> **No quality loss**: Speculative decoding does not approximate the target model. Accepted tokens are exactly those the target model would have produced **under the same sampling configuration**.
这句在 T>0 下只有分布意义（L1），字面读容易被理解成 L2。

### 1.8 V0 → V1 迁移：哪些能力被删了、哪些加回来了

**证据链**：

- **v0.8.5（2025）的 V1 user guide** 原文：
  > **Spec Decode**: 🚧 WIP ([PR #13933](https://github.com/vllm-project/vllm/pull/13933))
  > **Spec Decode**: Currently, **only ngram-based spec decode is supported in V1**. There will be follow-up work to support other types of spec decode (e.g., see PR #13933). **We will prioritize the support for Eagle, MTP compared to draft model based spec decode.**
  来源：https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5/docs/source/getting_started/v1_user_guide.md
- **当前文档**：`Speculative decoding with draft models is not supported in vllm<=0.10.0` → 独立 draft model 是在 **v0.11.x 附近**才回来的，对应 tracking issue #28947 里记的 **PR #24322（merged）**。
- **v0.27.1 的 V1 guide**：`**Spec Decode** | 🟢 Functional`。

**V0 有、V1 没有的字段**（对比 v0.9.2 的 `vllm/config.py::SpeculativeConfig` 与 v0.27.1 的 `vllm/config/speculative.py`）：

| V0 字段（v0.9.2） | V0 默认值 | v0.27.1 状态 |
|---|---|---|
| `acceptance_method` | `"rejection_sampler"`（可选 `"typical_acceptance_sampler"`） | **已删**。Medusa 式 typical acceptance（L3 近似）在 V1 里**没有对应开关** |
| `posterior_threshold` | `None` → typical 时置 **`0.09`** | **已删** |
| `posterior_alpha` | `None` → typical 时置 **`0.3`** | **已删** |
| `disable_logprobs` | **`True`** | **已删**（V0 默认开投机就不返回 logprobs；V1 不再有此开关，兼容矩阵标 logP ✅） |
| `disable_mqa_scorer` | `False` | **已删**（V1 无 batch expansion 路径） |
| **`disable_by_batch_size`** | `None`，语义「Disable speculative decoding for new incoming requests when the number of enqueued requests is larger than this value」 | **已删**。取而代之的是 `num_speculative_tokens_per_batch_size`（Dynamic SD，可把 K 降到 0） |
| `speculative_token_tree` | `None`，「Specifies the tree structure for speculative token generation」 | **在 v0.27.1 的 SpeculativeConfig 中已不存在**（grep 无匹配）。树形草稿是否有等价入口，未查证 |
| `enable_chunked_prefill`（内部字段） | 注释：「Used for raising an error since it's **not yet compatible with speculative decode**」 | **已删**。V1 下 CP × SD = ✅ |
| `disable_log_stats` | — | 已删（移到全局） |

**这张表是本库 §20 篇最有价值的一段**：它把"V0 时代关于投机解码的常识"整段作废了——
- 「开投机就没有 logprobs」→ 曾经为真（V0 `disable_logprobs=True`），现在为假；
- 「投机和 chunked prefill 不能同时开」→ 曾经为真（V0 显式 raise），现在为假；
- 「用 `disable_by_batch_size` 在大 batch 下自动关投机」→ 曾经的官方做法，**现在这个参数不存在了**（issue #25112 抱怨它「Spec decoding is not disabled at/after configured batch size」，之后 V1 换成了 Dynamic SD）。

**V1 新增、V0 完全没有的**：`suffix`、`dflash`、`dspark`、`ngram_gpu`、`custom_class`、`extract_hidden_states`、`parallel_drafting`(PARD)、`use_heterogeneous_vocab`(TLI)、`rejection_sample_method`(block/synthetic)、`enable_adaptive_verification`、`num_speculative_tokens_per_batch_size`、`--per-request-spec-decode-metrics`。

### 1.9 vLLM 官方承认的负收益（重点，另见 §7 汇总）

按"从概念到可判定"的顺序排：

**(a) 定位句**——文档第一句就把适用区间圈死了：
> This document shows how to use Speculative Decoding with vLLM to reduce inter-token latency **under medium-to-low QPS (queries per second), memory-bound workloads**.

**(a+) ⭐ 全库最有价值的一条：vLLM 官方 blog 自己贴出了「变慢多少倍」的数字。**
《How Speculative Decoding Boosts vLLM Performance by up to 2.8x》（2024-10-17，https://vllm.ai/blog/2024-10-17-spec-decode ）原文：

> Speculative decoding offers significant performance benefits in **low-QPS (queries per second)** environments.
> However, in **high-QPS environments, speculative decoding may introduce performance trade-offs.** The extra compute required to propose and verify tokens **can sometimes slow down the system when it is already compute-bound**, as seen when the number of requests per second increases. In such cases, **the overhead of speculative decoding can outweigh its benefits, leading to reduced performance.**
> As high QPS, we see **1.4x slowdown Llama3-70B on ShareGPT with 4xH100, 1.8x slowdown Llama3-70B on CNN Dailymail with 4xH100**

口径（按铁律二逐项，缺项标注）：

| 口径项 | 值 |
|---|---|
| 硬件 | **4× H100**（全部测试） |
| target | **Llama3-70B** |
| draft | 测试一：`turboderp/Qwama-0.5B-Instruct`；测试二：**n-gram / prompt lookup（无独立模型）** |
| γ / K | **原文未给出** |
| 低 QPS 收益 | ShareGPT **1.5×**（draft-model 法，QPS=1）；CNN/DailyMail **2.8×**（prompt lookup 法，QPS=1） |
| **高 QPS 负收益** | ShareGPT **1.4× slowdown**；CNN/DailyMail **1.8× slowdown** |
| 高 QPS 的具体 QPS 数值 | **原文未给出**（只写 "high QPS"） |
| batch size | **原文未给出** |
| 平均接受长度 / α | **原文未给出** |
| 测的是什么 | token generation 速度（口径未进一步说明，**原文未给出**是 TPOT 还是端到端） |

> 这就是本库 `19-负收益全解` 最该开篇引用的材料：**不是第三方吐槽，是引擎作者自己在宣传"2.8× 加速"的同一篇 blog 里，贴出了同一模型同一硬件下 1.4×–1.8× 的减速。** 标题只写 2.8×，负收益写在正文中段——这个"标题与正文的落差"本身就值得点评。
> 该 blog 把 dynamic speculative decoding 列为未来解法；**未提及** `disable_by_batch_size`。
> ⚠️ 配套引用：issue [#10318](https://github.com/vllm-project/vllm/issues/10318) 的标题就是「Results from the vLLM Blog article "How Speculative Decoding Boosts vLLM Performance by up to 2.8x" are unreproducible」——**连那个 2.8× 也有人复现不出来**。两条要一起写。

**(b) 方法选择表**（README，原文表格）：

| Method | Low QPS (latency focused) | High QPS (throughput focused) |
|---|---|---|
| EAGLE | High gain | Medium to high gain |
| MTP | High gain | Medium to high gain |
| Draft model | High gain | **Medium gain** |
| Parallel Draft Model | High gain | Medium to high gain |
| MLP speculator | Medium to high gain | Medium gain |
| N-gram | Low to medium gain | Medium gain |
| Suffix decoding | Low to medium gain | Medium gain |

表前的免责声明也要抄：
> Use this qualitative table as a starting point for method selection. **Real gains depend on your model family, traffic pattern, hardware, and sampling settings.**
> 注意这是**定性**表，没有任何数字，不能当加速比引用。

**(c) 最锋利的一段**——`adaptive_verification.md` 开头，是我在所有引擎文档里找到的**对"投机是一笔用算力换延迟的交易"最直白的官方表述**（原文）：

> Speculative decoding buys fewer decode steps with more compute. **At batch size 1 that is a good trade**: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. **At batch size 256 it is a much more delicate one.** Draft tokens now compete with real tokens for the same compute, and **every rejected token is compute wasted; with enough of them, throughput drops.**
>
> That matters because **per-position acceptance decays fast.** While the GPU is memory-bound that slot is effectively free and worth the gamble; **once it saturates the gamble has a real throughput cost.** The crossover moves with load and with workload-dependent acceptance rates, so **no static `num_speculative_tokens` is right across concurrencies.**

**(d) 可判定形式**——`dynamic_speculative_decoding.md` 给了一个粗糙但可操作的判据（原文）：

> SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes **BS*K** which increases the compute requirement during verification. **When this BS*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT).**

→ 这就是本库 `18-batch与吞吐-收益衰减曲线` 要量化的那条曲线的官方版本：**有效 batch = BS × K，超过临界 batch 后 TPOT 恶化**。

**(e) V0 时代的自认**（历史材料，仍值得引用）——v0.9.2 及更早文档页顶部的两条 warning：
> Please note that **speculative decoding in vLLM is not yet optimized and does not usually yield inter-token latency reductions for all prompt datasets or sampling parameters.** The work to optimize it is ongoing and can be followed here: gh-issue:4630
> Currently, speculative decoding in vLLM is **not compatible with pipeline parallelism**.
来源：https://raw.githubusercontent.com/vllm-project/vllm/v0.9.2/docs/features/spec_decode.md
（这两条 warning 在当前 `main` 的文档里**已被删除**——说明 vLLM 认为 V1 下的投机解码已经"优化过了"。写正文时要注意这是**版本相关的表述变化**，不能拿老 warning 当今天的结论。）

**(f) RFC #4565（Automate Speculative Decoding，LiuXiaoxuanPKU，2024-05-02，Closed as not planned）** 的问题陈述：
> when deploying Speculative Decoding in real online LLM serving systems that use continuous batching, improvements are not always observed. Paradoxically, **under conditions of high request rates or low speculation accuracy, latency may actually increase.**
其 Milestone 1 就是「monitor `running_queue` size, 超过阈值就 suspend speculation」——这正是后来 V0 `disable_by_batch_size` 的来源，也是 V1 Dynamic SD 的前身。
URL：https://github.com/vllm-project/vllm/issues/4565

**(g) 反向证据（必须一起写，否则是选择性引用）**：Red Hat 2026-04-16 的 gpt-oss 实测明确写了"打脸传统认知"：
> Crucially, **these gains persist up to 200 concurrent requests, contradicting the conventional expectation that speculative decoding majorly helps in low-QPS scenarios.**

口径（按铁律二逐项列，缺项标注）：

| 口径项 | 值 |
|---|---|
| 硬件 | NVIDIA **H200-PCIe-141GB**，单卡为主（另测 TP=2） |
| target | `openai/gpt-oss-120b`（MoE，MXFP4 量化） |
| draft | `nvidia/gpt-oss-120b-Eagle3-v2`（EAGLE3） |
| 引擎版本 | **vLLM v0.13.0**（原文如此）。⚠️ v0.13.0 的发布日是 **12-19**（GitHub release 页显示 "released this 19 Dec"，即 2025-12-19），而文章发表于 2026-04-16——**benchmark 用的是发表时已落后约 4 个月、距今（2026-08）落后 14 个 minor 版本的引擎**。引用这组数字时必须带这句 |
| K (`num_speculative_tokens`) | 测了 2 / 3 / 4，**选定 3** |
| 并发 | **1, 5, 25, 50, 100, 200** |
| 数据集 | ShareGPT / MLPerf / SWE-bench，各 600 prompt 分 6 组 |
| 测量对象 | output throughput、TTFT P95、ITL P95（**延迟与吞吐都测了**） |
| 结果 | ShareGPT：吞吐 +20.7%，TTFT P95 +10.8%，ITL P95 +12.4%；MLPerf：TP=1 +9.5%，TP=2 +16%；SWE-bench：吞吐 +20.5%，每百万 token 成本 −19.4% |
| 接受率 / 接受长度（ShareGPT） | K=2 → 45.4% / 1.91；K=3 → 35.6% / **2.07**；K=4 → 28.3% / **2.13** |
| 基准工具 | GuideLLM v0.5.3 |

URL：https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss

**(h) 同一家（Red Hat）一年前的另一组实测，结论正好相反**——EAGLE3 在高请求率下延迟**上升**。原文两句：

> The latency for each request of the 70B model is reduced by up to 1.6X at low request rates, **but latency increases at higher request rates due to compute saturation.**
> **At high request rates, though, it is optimal to reduce the draft length** to avoid drafting tokens that have a lower probability of being accepted.

口径：

| 口径项 | 值 |
|---|---|
| 硬件 | **A100**：8B 用单卡，70B 用 4 卡 |
| target | `meta-llama/Llama-3.1-8B-Instruct`、`Llama-3.3-70B-Instruct` |
| draft | `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`（70B 侧） |
| 引擎版本 | **vLLM 0.9.1** |
| K | 3（主基准值） |
| 数据集 | MT-Bench（两轮对话），输出截到 1024 token |
| 测量对象 | **request latency（端到端）** |
| 结果 | 整体 "up to 2.5X"；8B "up to 1.8X"；70B 低请求率 "up to 1.6X"；RAG 与数学推理任务 "up to 2.1X better latency" |
| 平均接受长度数值 | **原文未给出**（只说"随任务与草稿长度变化"） |
| 具体 QPS 数值 | **原文未给出** |
| batch size | **原文未给出**（只有定性的 low / high request rate） |

URL：https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding

> ⚠️ **(g) 与 (h) 不能放进同一张表比较**（违反铁律二）：硬件不同（H200 vs A100）、模型不同（MoE-MXFP4-120B vs dense-70B）、引擎版本差 14 个 minor、测量对象不同（吞吐+P95 延迟 vs 端到端请求延迟）、负载刻画不同（并发数 vs 请求率）。它们能一起说明的只有一件事：**"投机在大 batch 下是否还有收益"没有普适答案，取决于该模型在该硬件上何时进入 compute-bound。**

> **怎么把 (c)(d) 和 (g)(h) 放在一起讲**（建议写进 19 篇）：两者不矛盾。(c)(d) 说的是**给定接受率下 BS×K 越界就亏**；(g) 的场景是 **MoE + MXFP4 的 120B 模型在单张 H200 上**——MoE 的每 token 激活参数少、算力空转严重，memory-bound 区间被拉得极宽，所以 200 并发时 GPU 仍未饱和。**判据不是"并发数"，是"GPU 有没有进入 compute-bound"**。这条正好呼应表里 K=4 时接受率跌到 28.3% 而平均接受长度只从 2.07 涨到 2.13——**边际收益已经几乎为零，但边际算力成本仍是线性的**。

### 1.10 vLLM 已知 bug / 反复出现的坑

（本节部分条目来自并行调研，见 §8。）

已在官方文档里被点名的：

1. **MLP speculator 的 70B 加速器崩溃**（文档 `!!! warning "Known issue"` 原文）：
   > `ibm-ai-platform/llama3-70b-accelerator` can fail with: `AttributeError: 'MLPSpeculatorConfig' object has no attribute 'num_attention_heads'`. Track status in [#34106](https://github.com/vllm-project/vllm/issues/34106) and [#34163](https://github.com/vllm-project/vllm/pull/34163).
2. **EAGLE 草稿的 rotary cache 越界（#48894）**。源码注释：
   > a draft checkpoint with a smaller `max_position_embeddings` than the target under-sizes its rotary cache (#48894)
   现已由 `_maybe_override_draft_max_position_embeddings()` 自动把 draft 的 `max_position_embeddings` 抬到 target 的 `max_model_len`。**老版本上这是个真实崩溃**（`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` 的值是 2048）。
3. **`vllm<0.7.0` 的 EAGLE 权重需要转换脚本**（文档 warning，附 gist 链接）。
4. **Gemma 4 assistant checkpoint 被误判成 draft_model**。文档给了自查方法：
   > If startup logs show `SpeculativeConfig(method='draft_model', ...)` for a Gemma 4 assistant checkpoint, the installed vLLM version does not include Gemma 4 MTP support for that path.
5. **MTP 的 `num_speculative_tokens > 1` 会掉接受率**（源码 warning）：
   > `Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate`
   （例外：`step3p5_mtp`、`inkling_mtp` 不打这个警告；`inkling_mtp` 更是**只允许 K=1**。）
6. **DSpark 的 K 小于 block size 会输出乱码而不是"只是变慢"**（源码注释，很典型的静默错误）：
   > A speculative length smaller than the checkpoint's block feeds the block / Markov-head machinery an unsupported layout and **yields incorrect (garbled) output rather than merely lower acceptance.**
7. **词表大小不一致直接 raise**（非 TLI 模式）：
   > `Target and draft model should have the same vocabulary size. ... Using models with different tokenizers can cause out-of-bounds errors during speculative decoding.`

**只有读源码才知道的两条（`vllm/v1/spec_decode/ngram_proposer.py` @ v0.27.1）**：

8. **ngram 提议器跑在 CPU 上，用 numba JIT，而且当前实际只用 1 个线程**：
   ```python
   # TODO(ekagra-ranjan): bump up the cap from 1 to 8
   # when TP parallelization for ngram is implemented.
   self.num_numba_thread_available = min(1, (cpu_count // 2))
   self.num_numba_thread_available //= tp_size
   ```
   `min(1, ...)` 恒等于 1（除非 cpu_count//2 == 0），再除以 `tp_size`——**TP>1 时甚至会变成 0**。另有阈值 `self.num_tokens_threshold = 8192`：batch 总 token 数低于它就强制单线程（注释：「If total tokens is small, using multiple threads may slow down due to overhead」）。
   → **这解释了 `ngram_gpu` 为什么被单独做出来（PR #29184）**：CPU 侧的 n-gram 匹配在大 batch 下是真实的 CPU 瓶颈，而且每个 TP rank 都要各跑一遍。
9. **「prompt lookup」这个名字是历史包袱**。`batch_propose` 吃的是 `token_ids_cpu`（形状 `(batch_size, max_model_len)`）配 `num_tokens_no_spec`，即**当前序列迄今为止的全部 token（prompt + 已生成）**，不只是 prompt。
   与 suffix decoding 的真正区别不在"看不看已生成的 token"，而在**suffix 额外维护一棵跨请求的全局 suffix tree**（`suffix_decoding_max_cached_requests` 默认 10000，设 0 关闭）。文档 `n_gram.md` 里那句 "matching n-grams in the prompt" 属于措辞不严谨。

社区侧高频（搜索命中，**逐条给了 issue 号，但正文引用前建议再点开确认状态**）：

| issue | 标题 | 要点 |
|---|---|---|
| [#8439](https://github.com/vllm-project/vllm/issues/8439) | why speculate decoding is slower than normal decoding？ | ngram 投机后 decode 61.79 tok/s vs 不开 69.79 tok/s（**硬件/模型/batch 原文未核**） |
| [#10318](https://github.com/vllm-project/vllm/issues/10318) | vLLM 博客「How Speculative Decoding Boosts vLLM Performance by up to 2.8x」的结果**不可复现** | 官方 blog 数字被用户质疑复现不出来 |
| [#25112](https://github.com/vllm-project/vllm/issues/25112) | `disable_by_batch_size` 在配置的 batch size 之后没有真的关掉投机 | 该参数在 V1 已被移除 |
| [#9565](https://github.com/vllm-project/vllm/issues/9565) | EAGLE 在 vLLM 上的加速比低于原论文实现 | |
| [#28384](https://github.com/vllm-project/vllm/issues/28384) | 为 CPU 启用 draft-model 投机 | CPU 后端目前 ❌ |
| [#28947](https://github.com/vllm-project/vllm/issues/28947) | **[Tracking Issue][Performance] Speculative decoding performance/QoL improvements**（2025-11-18，作者 xinli-sw，POC @benchislett） | 见下 |
| [#28135](https://github.com/vllm-project/vllm/issues/28135) | RFC: Robust End-to-End CI/CD Regression Testing for Speculative Decoding | 说明投机解码的回归测试覆盖此前是**不足**的 |

**#28947 的路线图**（很能说明"还有哪些没做好"）：
- 新草稿方式：Draft-model（#24322 已合）、Parallel Drafting/PARD/P-EAGLE（#32887 已合）、DFlash Parallel Drafting（WIP #32206）、NGram-GPU（在评审 #29184）、**Hybrid ngram-eagle drafting**（在评审 #24344）
- 模型支持：DeepSeek V3.2 支持 >1 个投机 token（#31845）
- 性能：**drafter 的 Full CudaGraph（#33341，说明 drafter 目前还没有完整 CUDA graph）**、EAGLE drafting kernel 优化三连（#28597 已合 / #33455 / #33456）、drafter 编译（#26179 评审中）、**多模态 EAGLE 支持（#33458，即 mm × SD 现在确实是 ❔）**
- 采样：**Probability-aware sampling（#20459，stalled）**
- 异步调度：基础支持 #24799 已合；penalties/bad_words 兼容 #30495；logprobs 支持 #29223 已合、#31336；**structured outputs 兼容 #29821**

→ 从这张表可以直接读出 v0.27.1 时点的**真实短板**：drafter 没有 full CUDA graph、多模态 EAGLE 未支持、概率感知采样停滞、异步调度与结构化输出的兼容还在做。

**结构化输出（structured output / guided decoding）× 投机**：官方没有单独的兼容性声明，但 `acceptance_metrics.md` 对 `num_draft_tokens` 的定义泄露了实现细节：
> Total proposed draft tokens, **after subtracting drafts invalidated by structured-output constraints.**
→ 即：**结构化输出会作废一部分草稿 token**，两者能一起开，但接受率会被约束打折。

---

### 1.11 vLLM 官方 blog 里的投机解码系列（2024–2026 全清单）

抓自 https://vllm.ai/blog （2026-08-22）。**这条清单本身就是 `24-前沿进展-2025到2026` 的时间线骨架**：

| 日期 | 标题 | slug |
|---|---|---|
| 2024-10-17 | How Speculative Decoding Boosts vLLM Performance by up to 2.8x | `/blog/2024-10-17-spec-decode` |
| 2025-12-13 | Diving into speculative decoding training support for vLLM with Speculators v0.3.0 | `/blog/2025-12-13-speculators-v030` |
| 2026-03-13 | **P-EAGLE**: Faster LLM inference with Parallel Speculative Decoding in vLLM | `/blog/2026-03-13-p-eagle` |
| 2026-03-30 | Extracting hidden states from vLLM | `/blog/2026-03-30-extract-hidden-states` |
| 2026-05-26 | **EAGLE 3.1**: Advancing Speculative Decoding Through Collaboration Between the EAGLE Team, vLLM, and **TorchSpec** | `/blog/2026-05-26-eagle-3-1` |
| 2026-05-28 | Speculators v0.5.0: **DFlash Support and Online Training** | `/blog/2026-05-28-speculators-v050` |
| 2026-05-28 | Accelerating Laguna XS.2 Inference with vLLM, Speculators, and LLM Compressor | `/blog/2026-05-28-laguna-xs2-dflash-llm-compressor` |
| 2026-07-13 | **EAGLE3 Speculative Decoding on AMD Instinct GPUs**: Training and Serving with vLLM and AMD Quark | `/blog/2026-07-13-eagle-3-amd-instinct` |
| 2026-07-28 | Parallel All the Way Down: Beyond Single-Token Generation with Speculative Decoding | `/blog/2026-07-28-speculators-parallel-drafting` |
| 2026-08-14 | **Adaptive Verification in vLLM: DSpark confidence-scheduled verification** | `/blog/2026-08-14-dspark-adaptive-verification` |

#### ⭐ EAGLE 3.1（2026-05-26）—— 顺带回答了"TorchSpec 是什么"

- **TorchSpec = https://github.com/lightseekorg/TorchSpec**，由 "the TorchSpec Team" 维护。定位（原文）：TorchSpec "provides efficient **training** support for EAGLE 3.1 and future speculative decoding algorithms"，目标是 "lower training overhead and simplify experimentation workflows"。
  → **它是训练框架，不是推理引擎**，与 vLLM 的 `speculators`、SGLang 的 `SpecForge`、NVIDIA 的 `ModelOpt` 同一层级。**这一层现在已经有四个竞品了。**
- **EAGLE 3.1 相对 EAGLE-3 改了什么**：针对所谓 **"attention drift"**——
  > "FC normalization after each target hidden state and before the FC layer"
  > "Feeding post-norm hidden states into the next decoding step"
  收益（原文）："better training-time to inference-time extrapolation"、"stronger long-context robustness"、"higher resilience to chat template and system prompt variation"，并称在长上下文负载上 **"EAGLE 3.1 achieves up to 2× longer acceptance length compared with EAGLE 3"**。
  > **"对 chat template 与 system prompt 变化更鲁棒"这一点值得注意**——它暗示 EAGLE-3 的接受率对提示词模板敏感，这是社区里很少被提及的一个脆弱点，可写进 `13-EAGLE三代` 的"什么被后来推翻"。
- 配置（原文）：
  ```bash
  --speculative-config '{"model":"lightseekorg/kimi-k2.6-eagle3.1-mla","method":"eagle3","num_speculative_tokens":3}'
  ```
  注意 **method 仍写 `eagle3`**——EAGLE 3.1 不是一个新的 `method` 值，是新的权重 + 训练法。
- 权重：`lightseekorg/kimi-k2.6-eagle3.1-mla`
- benchmark 口径：**Kimi-K2.6-NVFP4，GB200，TP=4，非 disagg，SPEED-Bench coding，指标 = per-user output throughput**

| concurrency | 1 | 4 | 16 |
|---|---|---|---|
| 相对无投机的加速 | **2.03×** | **1.71×** | **1.66×** |

  **acceptance length：原文未给出**（benchmark 段没给）。ISL/OSL：原文未给出。

#### Parallel All the Way Down（2026-07-28）—— vLLM 的 `dflash` 配置写法在这里

文档目录 `docs/features/speculative_decoding/` 下**没有 DFlash 专页**，`dflash` 的配置写法只出现在这篇 blog：
```json
"speculative-config": {
  "model": "RedHatAI/Qwen3-30B-A3B-speculator.dflash",
  "num_speculative_tokens": 7,
  "method": "dflash"
}
```
（对照 §1.4：`method="dflash"` 会被源码强制置 `parallel_drafting = True`，且 `attention_backend` docstring 点名 "DFlash needs a non-causal-capable backend like FLASH_ATTN"。）

blog 给的对比只有定性（**无 acceptance length、无具体加速比**）：

| 方法 | 模型 | 任务 | 硬件 | 结论 |
|---|---|---|---|---|
| P-EAGLE | Qwen3-8B | Math (GSM8k) | 1×A100 | 优于 EAGLE-3 |
| DFlash | Qwen3-30B-A3B | Coding (HumanEval) | 2×A100 | 优于 EAGLE-3 |
| DSpark | Gemma-4-31B-it | Coding (HumanEval) | 2×A100 | 优于 EAGLE-3 |

官方自己的免责声明（原文，值得引用）：
> **Performance will vary across models, tasks, and hardware configurations—we encourage the community to benchmark on their own workloads.**

> **这句免责声明和 vLLM README 方法选择表前那句 "Real gains depend on your model family, traffic pattern, hardware, and sampling settings."、以及 TensorRT-LLM blog06 的 "speculative decoding speedups depend upon the dataset!" 是同一件事**：**三家官方都在自己的 benchmark 旁边写了"别照抄我的数字"。** 这本身就是铁律二最好的背书，可以放进 `27-常见误解与判据`。

#### EAGLE3 on AMD Instinct（2026-07-13）—— 唯一一组非 NVIDIA 硬件的官方数据

口径：**AMD MI355X × 4（TP=4）**；vLLM **v0.19.0**（BF16）与 nightly（FP8）；**ISL=1024 / OSL=1024**；指标为**吞吐**。

| target | draft | 精度 | `num_speculative_tokens` | 吞吐增益 |
|---|---|---|---|---|
| `amd/Kimi-K2.5-MXFP4` | `lightseekorg/kimi-k2.5-eagle3` | BF16 | 3 | **1.69×–2.00×** |
| `amd/Kimi-K2.5-MXFP4` | `amd/kimi-k2.5-eagle3-fp8` | FP8 | 3 | **1.76×–2.00×** |
| `MiniMaxAI/MiniMax-M2.5` | `thoughtworks/MiniMax-M2.5-Eagle3` | BF16 | 3（+`draft_tensor_parallel_size=1`，TP=4 + EP） | **1.38×–1.79×** |

原文关于并发的表述：**"the gain is largest at low concurrency"**——增益随 batch 增大而缩小（**具体并发档位数值本次未提取到，标未查证**）。

**接受长度（MiniMax-M3）：11 个领域平均 2.80 token；从 1K 到 32K 上下文基本稳定（2.69 → 2.65）。**
> ⭐ **最后这句对 `22-长上下文下的投机采样` 很关键**：它是我找到的**唯一一组"接受长度 vs 上下文长度"的官方实测**，结论是**基本不衰减**（2.69 → 2.65，跨 32 倍上下文）。
> 这与 EAGLE 3.1 blog 声称"长上下文鲁棒性改善、AL 最多提升 2×"存在张力——**同一个问题上两篇官方材料给的图景不同**（一篇说 EAGLE-3 在长上下文基本稳定，一篇说 EAGLE-3 有 attention drift、3.1 修好后 AL 翻倍）。差别可能在模型（MiniMax-M3 vs Kimi-K2.6）与上下文量级（32K vs 更长）。**正文里必须并列呈现，不要只取一边。**

#### ⭐⭐ P-EAGLE / 并行草稿（2026-03-13）—— 少见的「同一实验里既给 AL 又给 c=1/c=64 两档」

机制（原文要点）：**顺序 EAGLE 需要 K 次前向才能出 K 个草稿；P-EAGLE 一次前向出全部 K 个**——做法是捕获 target 在每个 prompt 位置的 hidden states，把 token embedding 与 hidden state 配对成并行输入，对"还不存在的未来位置"用**可学习的 mask token embedding + 共享 hidden state 占位符**，然后一次性过 transformer 层。

配置（原文）：
```json
"speculative-config": {
  "method": "eagle3",
  "model": "amazon/GPT-OSS-20B-P-EAGLE",
  "num_speculative_tokens": 7,
  "parallel_drafting": true
}
```
> 注意：**`parallel_drafting` 是挂在 `method: "eagle3"` 上的**（vLLM 文档里 PARD 的例子挂在 `method: "draft_model"` 上）——两条路径都支持并行草稿。

口径：**1 × NVIDIA B200，GPT-OSS-20B，P-EAGLE drafter（4 层，训练到最多 10 token）**。

| Benchmark | 并发 | K | **P-EAGLE AL** | **EAGLE-3 AL** | 加速比 |
|---|---|---|---|---|---|
| MT-Bench | **c=1** | 7 | **3.70** | 3.27 | **1.55×** |
| MT-Bench | **c=64** | 5 | 原文未给出 | 原文未给出 | **1.05×** |
| HumanEval | **c=1** | 7 | **3.94** | 3.03 | **1.55×** |
| HumanEval | **c=64** | 7 | 原文未给出 | 原文未给出 | **1.23×** |
| SPEED-Bench | **c=1** | 7 | **3.38** | 2.59 | **1.69×** |
| SPEED-Bench | **c=64** | 7 | 原文未给出 | 原文未给出 | **1.25×** |

（加速比的口径——延迟还是吞吐——**原文未明确**；ISL/OSL 原文未给出。）

blog 自陈的高并发收益（原文）：
> Gains of **5–25% sustained at high concurrency (c=64)**

> ⭐ **这张表有三个高价值点**：
> ① **c=1 → c=64 时加速比从 1.55–1.69× 掉到 1.05–1.25×**，且**同一实验、同一硬件、同一模型、同一 K**——这是本次调研里**唯一一组只变并发、其余七项口径全部一致的官方数据**，按铁律二**可以直接横比**。建议作为 `18-batch与吞吐-收益衰减曲线` 的主图数据。
> ② MT-Bench 在 c=64 时把 K 从 7 降到 5——**官方自己在高并发档调小了 γ**，与 §7.2 成因 B 的判据一致。
> ③ **P-EAGLE 的 AL（3.38–3.94）显著高于 EAGLE-3（2.59–3.27）**，而这两组是同一 target、同一 K、同一硬件——**这是本调研里少数几组可以直接比较"草稿方法优劣"的干净数据**。注意 AL 高不等于加速比高（并行草稿本身每步更贵），blog 给的加速比是相对无投机的。

#### ⭐⭐ Adaptive Verification / DSpark（2026-08-14，本次调研拿到的最新一篇）

配置（原文）：
```json
"speculative-config": {
  "method": "dspark",
  "attention_backend": "FLASH_ATTN",
  "num_speculative_tokens": 7,
  "draft_sample_method": "probabilistic",
  "enable_adaptive_verification": true
}
```
口径：**8 × B300（SM100），DeepSeek-V4-Pro-0813，TP=8 + expert parallel，FP8 KV cache，`max_model_len 16384`；880 prompts，temperature 1.0，输出上限 2048 token/请求；并发扫 1 → 256。**

**最有价值的一个数字——逐位置接受率的衰减**：
> per-position acceptance on DeepSeek-V4-Pro-0813 showed the last drafted token survived **less than 10% of the time, against more than 70% for the first**.

> ⭐⭐ **这是本次调研里对 `06-期望接受长度与加速比模型` 最重要的一条实测**：K=7 时，**第 1 位接受率 >70%，第 7 位 <10%**。
> 它直接证伪了加速比模型里最常见的简化假设——**"每个位置的接受率都是同一个 α"**（即 $E[\tau] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 的推导前提）。
> 如果 $\alpha_1 = 0.7$ 而 $\alpha_7 < 0.1$，那么按几何分布算的话 $0.7^7 \approx 0.082$——**恰好落在"<10%"这个区间内**。也就是说 **DeepSeek-V4-Pro 的实测衰减与"独立同分布 + 链式相乘"的几何模型高度吻合**。
> **建议在 `_lab/accept.py` 里加一个测试：给定 $\alpha_1=0.7$，验证几何模型预测的第 7 位无条件接受率与官方观测的 "<10%" 相符。** 这既能验证模型，也能给写作规范铁律四提供一个真实锚点。
> 同时注意：vLLM `adaptive_verification.md` 里那句 "**per-position acceptance decays fast**" 现在有了具体数字支撑。

blog 正文关于大 batch 的表述与文档一致：
> at batch size 256 the trade is much more delicate. Draft tokens now compete with real tokens for the same compute, and **every rejected token wastes useful compute; with enough rejected tokens, throughput drops significantly.**

结果：adaptive verification 在整个并发扫描区间内都维持了吞吐收益（**具体加速比数值原文未给出/未提取到，标未查证**）。

---

## 2. SGLang

### 2.0 版本锚点与一条方法论警告

- **SGLang v0.5.18**（GitHub release 发布于 `2026-08-22T00:09:15Z`；PyPI `sglang 0.5.18` 上传于 `2026-08-21T20:58:46`）。代码核验基准为 `main` 分支 2026-08-22 快照。
- ⚠️ **文档域名已迁移**：`https://docs.sglang.ai/...` **301 跳转**到 `https://docs.sglang.io/...`。
- ⚠️ **踩坑记录（值得写进 §8 取证方法）**：用通用网页抓取工具读 `docs.sglang.io/advanced_features/server_arguments.html` **渲染页**时，返回内容里出现了 `--speculative-algorithm` 取值 `mlp/eagle/medusa`、以及参数 `--speculative-num-prefill-batches`、`--speculative-disable-by-batch-size`。**这些全部是幻觉**（其中几个实为 vLLM 的参数）。改读 GitHub 上的 `.mdx` 文档源文件 + `python/sglang/srt/server_args.py` 源码后全部推翻。
  **SGLang 不存在 `--speculative-disable-by-batch-size` 这个参数。** 这是本次调研里最重要的一次自我纠错。

### 2.1 支持哪些方法

权威来源：`python/sglang/srt/speculative/spec_info.py` 的 `SpeculativeAlgorithm` 枚举
（https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_info.py ）

```python
class SpeculativeAlgorithm(Enum):
    DFLASH = auto()
    DSPARK = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()
```

`server_args.py` help 原文：
> "Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK. Or any name registered via `SpeculativeAlgorithm.register`."

| 方法 | 状态 | 依据 |
|---|---|---|
| **EAGLE（EAGLE-2）** | ✅ | 枚举 + 文档整节 |
| **EAGLE3** | ✅，官方推荐首选 | 文档："**Best speed/quality (recommended)**: Use **EAGLE-3** with `--speculative-algorithm EAGLE3`" |
| **NEXTN** | ✅，但**是 EAGLE 的别名** | `_resolve_speculative_algorithm_alias()`：`NEXTN`/`EAGLE` → 都返回 `"EAGLE"` |
| **MTP** | ✅，**复用 EAGLE 通路，不是独立算法** | 文档 "Multi Token Prediction" 节用 `--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2` |
| **STANDALONE**（独立小草稿模型） | ✅ | 枚举 + 文档整节 |
| **NGRAM** | ✅（仅 CUDA 或 CPU） | 枚举 + 文档整节 |
| **DFLASH**（block-parallel drafting） | ✅ | 论文 arXiv:2602.06036 |
| **DSPARK**（Confidence-Scheduled Semi-AR） | ✅ | 论文 arXiv:2607.05147。⚠️ **文档 `server_arguments.mdx` 的 Options 列只写了 `EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM`，漏了 DFLASH/DSPARK——文档自身不一致** |
| **FROZEN_KV_MTP** | ⚠️ 内部算法，**不能 CLI 指定**；检测到 Gemma4 assistant draft 架构时由 `NEXTN`/`EAGLE` 自动提升 | |
| **Multi-layer EAGLE** | ✅ 独立开关 `--enable-multi-layer-eagle`；MiMoV2 与 Step3p5 自动开 | |
| 自定义算法 | ✅ `SpeculativeAlgorithm.register(name, supports_overlap=..., ...)` | |
| **Medusa** | ❌ **完全不支持**（全仓库 grep `medusa` 命中 **0**） | |
| **Lookahead Decoding** | ❌ 不支持（grep `lookahead` 只命中正则/mamba 等无关用法） | |

> **对比 vLLM 的一条重要差异**：vLLM 仍保留 `medusa` 实现（`vllm/v1/spec_decode/medusa.py`），SGLang 则**从未实现过 Medusa**。写 `12-Medusa` 篇的"它被淘汰了吗"时这是硬证据。

### 2.2 参数名与默认值（抄自 `python/sglang/srt/server_args.py`）

#### 核心

| 参数 | 类型 | 默认值 | help 原文 |
|---|---|---|---|
| `--speculative-algorithm` | `Optional[str]` | `None` | 见上 |
| `--speculative-draft-model-path`（别名 `--speculative-draft-model`） | `Optional[str]` | `None` | The path of the draft model weights. |
| `--speculative-draft-model-revision` | `Optional[str]` | `None`（给了 path 未给 revision 时自动置 `"main"`） | |
| `--speculative-draft-load-format` | `Optional[str]` | `None` | |
| `--speculative-num-steps` | `Optional[int]` | `None`（自动，见 2.3） | The number of steps sampled from draft model in Speculative Decoding. |
| `--speculative-eagle-topk` | `Optional[int]` | `None`（自动） | The number of tokens sampled from the draft model in eagle2 each step. |
| `--speculative-num-draft-tokens` | `Optional[int]` | `None`（自动） | |
| `--speculative-accept-threshold-single` | `float` | **`1.0`** | Accept a draft token if its probability in the target model is greater than this threshold. |
| `--speculative-accept-threshold-acc` | `float` | **`1.0`** | The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc). |
| `--speculative-use-rejection-sampling` | `bool` | **`False`** | Use rejection sampling for speculative decoding (**requires topk=1**). |
| `--speculative-token-map` | `Optional[str]` | `None` | draft 模型的小词表（FR-Spec）；**EAGLE-3 忽略此参数** |
| `--speculative-attention-mode` | `str` | **`"prefill"`**（`prefill`/`decode`） | |
| `--speculative-draft-attention-backend` | `Optional[str]` | `None` | |
| `--speculative-draft-kv-cache-dtype` | `Optional[str]` | `None`（跟随 `--kv-cache-dtype`） | |
| `--speculative-draft-window-size` | `Optional[int]` | `None`（全注意力） | 「Honored by Llama EAGLE-3 (`LlamaForCausalLMEagle3`) and DFLASH only; other EAGLE-3 backends (e.g. MLA-based drafters) **silently ignore it**.」 |
| `--speculative-moe-runner-backend` / `--speculative-moe-a2a-backend` | `Optional[str]` | `None`（同 target） | |
| `--speculative-draft-model-quantization` | `Optional[str]` | `None`→继承 target；传 `"unquant"` 表示不量化 | |
| `--speculative-skip-dp-mlp-sync` | `bool` | `False` | **仅 EAGLE 可用**，否则 assert 失败 |
| `--enable-multi-layer-eagle` | `bool` | `False` | |
| `--speculative-adaptive` | `bool` | `False` | 按接受率动态调 num_steps |
| `--speculative-adaptive-config` | `Optional[str]` | `None` | JSON 配置路径 |
| `--spec-trace-dir` | `Optional[str]` | `None` | |

#### NGRAM 专属

| 参数 | 默认值 |
|---|---|
| `--speculative-ngram-min-bfs-breadth` | **`1`** |
| `--speculative-ngram-max-bfs-breadth` | **`10`** |
| `--speculative-ngram-match-type` | **`"BFS"`**（`BFS`/`PROB`） |
| `--speculative-ngram-max-trie-depth` | **`18`** |
| `--speculative-ngram-capacity` | **`10_000_000`** |
| `--speculative-ngram-external-corpus-path` | `None` |
| `--speculative-ngram-external-sam-budget` | **`0`** |
| `--speculative-ngram-external-corpus-max-tokens` | **`10000000`** |

NGRAM 下 `--speculative-num-draft-tokens` 未设时代码里硬置 **12**（文档写 `min(max_trie_depth, 12)`）；`speculative_eagle_topk` 被**强制覆盖**为 `speculative_ngram_max_bfs_breadth`。

#### DFLASH / DSPARK / 解耦式

| 参数 | 默认值 |
|---|---|
| `--speculative-dflash-block-size` | `None`（DFLASH 下是 `--speculative-num-draft-tokens` 的别名；两者都设且不等→报错；都不设→从 draft config 推断，推断失败 fallback **16**） |
| `--speculative-dflash-draft-window-size` | `None`（源码 dest 为 `speculative_draft_window_size`；必须 `>= num_draft_tokens`） |
| `--speculative-dspark-block-size` | `None`（gamma，`num_draft_tokens = gamma+1`） |
| `--speculative-dspark-sps-table-path` / `--speculative-dspark-confidence-sts-path` | `None` |
| `--speculative-dspark-align-verify-tokens-to-graph-tier` | `False` |
| `--decoupled-spec-role` | `"null"`（`null`/`verifier`/`drafter`） |
| `--decoupled-spec-bind-endpoint` / `--decoupled-spec-connect-endpoints` / `--decoupled-spec-rank` | `None` |

#### 环境变量

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `SGLANG_NGRAM_FORCE_GREEDY_VERIFY` | `False` | |
| **`SGLANG_SIMULATE_ACC_LEN`** | 关闭（<0） | **伪造接受长度**，用于基准测试。见 2.8 的可信度警告 |
| `SGLANG_RAGGED_VERIFY_MODE` | `static` | DSPARK ragged verify |
| `SGLANG_ENABLE_SPEC_V2` | **已移除** | 源码原文：「SGLANG_ENABLE_SPEC_V2 has been removed: speculative decoding always runs the V2 worker. Use `--disable-overlap-schedule` to select the non-overlap (synchronous) path.」 |

### 2.3 三元组自动值与「会静默改你配置」的联动规则

`_auto_choose_speculative_params()` 返回 `(num_steps, eagle_topk, num_draft_tokens)`：

```python
if speculative_algorithm == "STANDALONE":            return (3, 1, 4)
if model_arch in ["LlamaForCausalLM"]:               return (5, 4, 8)
elif model_arch in [DeepseekV32/V3/V2, GptOss, Glm4Moe, Glm4MoeLite, GlmMoeDsa,
                    BailingMoe*, MistralLarge3, Pixtral, MiMoV2, MiMoV2Flash]:
                                                     return (3, 1, 4)
elif model_arch in ["Grok1ForCausalLM","Grok1VForCausalLM"]: return (5, 4, 8)
else:                                                return (3, 1, 4)
```

**四条会静默改配置的规则（本节最有工程价值的部分）**：

1. 三个参数**要么全不设（走自动），要么全设**。文档："leave all three unset to use auto-tuning, or set all three explicitly when tuning"；只设 `num_steps` 会 assert 另两个为 `None`。
2. `topk == 1` 时 `num_draft_tokens` 被**强制改写**为 `num_steps + 1`，并打 warning。
3. 开启投机后 `--max-running-requests` 未设时被**强制置为 48**：「Max running requests is reset to 48 for speculative decoding.」
4. `--enable-mixed-chunk`（mixed chunked prefill）被**强制关闭**。

> 第 3 条尤其阴险：**很多人测"开投机后吞吐变差"，真实原因是并发上限被悄悄压到了 48**，而不是投机本身。写 `19-负收益全解` 时这是必须点名的"假负收益"来源。

### 2.4 草稿模型来源与 SpecForge

**需不需要单独训练，分三类**：

1. **EAGLE / EAGLE3 / STANDALONE / DFLASH / DSPARK** — 必须给 `--speculative-draft-model-path`。EAGLE/EAGLE3 的头要专门训；STANDALONE 可直接用同族小模型（如 `Qwen/Qwen2.5-1.5B-Instruct` 配 `Qwen/Qwen2.5-7B-Instruct`）。
2. **自带 MTP 头的模型**（DeepSeek V3/V32/V4、GLM4Moe、BailingMoe、MistralLarge3、Pixtral、HYV3 等）：`_handle_eagle_family` 里未给 draft path 时**自动设为 `server_args.model_path` 本身**，日志提示「DeepSeek MTP does not require setting speculative_draft_model_path.」
3. **NGRAM** — 不需要模型。

**官方权重**：文档示例用 `lmsys/sglang-EAGLE-llama2-chat-7B`、`lmsys/sglang-EAGLE-LLaMA3-Instruct-8B`、`jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B`、`thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt`（FR-Spec 词表）、`z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`。

**SpecBundle 官方权重矩阵**（https://github.com/sgl-project/sglang/blob/main/docs/cookbook/specbundle/supported_models.mdx ；HF collection https://huggingface.co/collections/lmsys/specbundle ）——摘录：

| Target | EAGLE3 Draft |
|---|---|
| meta-llama/Llama-3.1-8B-Instruct | `lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge` |
| meta-llama/Llama-3.3-70B-Instruct | `lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge` |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | `lmsys/SGLang-EAGLE3-Llama-4-Scout-17B-16E-Instruct-SpecForge` |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | `lmsys/sglang-EAGLE3-Llama-4-Maverick-17B-128E-Instruct-v1` |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | `lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex` |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | `lmsys/SGLang-EAGLE3-Qwen3-235B-A22B-Instruct-2507-SpecForge-Meituan` |
| Qwen/Qwen3-Coder-480B-A35B-Instruct | `lmsys/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI` |
| moonshotai/Kimi-K2-Instruct | `AQ-MedAI/Kimi-K2-Instruct-eagle3` |
| openai/gpt-oss-120b | `lmsys/EAGLE3-gpt-oss-120b-bf16` |

DSpark 旗舰草稿：`RadixArk/Inkling-DSpark-Preview`、`RadixArk/Kimi-K3-DSpark`。

**SpecForge**（仓库 https://github.com/sgl-project/SpecForge ；文档 https://docs.sglang.io/SpecForge/ ；论文 **arXiv:2603.18567**《SpecForge: A flexible and efficient open-source training framework for speculative decoding》）

README 定义原文：
> "SpecForge is an ecosystem project developed by the SGLang team. It is a framework for training speculative decoding models so that you can smoothly port them over to the SGLang serving framework to speed up your inference."

时间线（README "News" 节）：
- `2025-07` 首发，随 Llama4-Eagle3 checkpoints 一起（博客 2025-07-25）
- `2025-08` 为 GPT-OSS 训 EAGLE3 草稿；列为 LMSYS 旗舰项目
- `2025-12` **SpecBundle phase 1 + SpecForge v0.2.0**
- `2026-01` DFlash block-parallel online training
- `2026-06` Domino online training / D-PACE loss
- `2026-07` DSpark online training；训练与推理完全解耦
- `2026-08` **SpecBundle phase 2 + SpecForge v0.3.0**（博客 2026-08-04）

支持训练的方法：**EAGLE3、P-EAGLE（arXiv:2602.01469）、EAGLE3.1、DFlash（arXiv:2602.06036）、Domino（arXiv:2605.29707）、DSpark（arXiv:2607.05147）**。
统一入口：`specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml`；README 强调「There are no method-specific Python training entry points.」
训练拓扑：online（disaggregated external / managed-local）与 offline（colocated / disaggregated），支持 data/tensor/sequence parallel。早期博客提到 offline 需预算 hidden states，UltraChat+ShareGPT 约 **12TB**；分布式用 FSDP + TP。

**产物怎么接进 SGLang**（`docs/cookbook/specbundle/specbundle_usage.mdx` 原文）：
```bash
python3 -m sglang.launch_server \
    --model <target-model-path> \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft-model-path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```
即：**训练产物就是普通 HF 目录，直接喂 `--speculative-draft-model-path`，无需格式转换**。
（对比 vLLM 的 speculators：vLLM 是把 speculator 当 `--model` 传，靠 `speculators_config` 字段反查 target；SGLang 是显式两个路径。）

### 2.5 与 RadixAttention / chunked prefill / CUDA graph / PD 分离 / overlap 的相互作用

**RadixAttention / prefix caching**
- **不冲突，radix cache 保持开启**：`server_args.py` 中没有任何一处因 `speculative_algorithm` 而设 `disable_radix_cache = True`；spec worker 正常引用 `batch.tree_cache`。
- 但 **PD 分离下 decode 侧 radix cache 明确不兼容**，`--disaggregation-decode-enable-radix-cache` help 原文：
  > "Enable radix cache on decode server (PD mode). Caches KV prefixes to avoid redundant transfers. **Incompatible with --enable-hisparse, speculative decoding**, and --disaggregation-transfer-backend fake."
- ⚠️ 社区实测的严重问题见 2.7 的 issue #32459（多轮流量 prefix 复用率 97% → 40~53%）。

**chunked prefill**
- 普通 chunked prefill 兼容（PD/cookbook 例子里普遍同时开）。
- **mixed chunked prefill（`--enable-mixed-chunk`）不兼容，被强制关闭**。日志原文：
  > "Mixed chunked prefill is disabled because of using eagle speculative decoding."
  > "Mixed chunked prefill is disabled because of using dflash speculative decoding."
  > "The mixed chunked prefill are disabled because of using ngram speculative decoding."
  另有一处断言的**文案本身写反了**（SGLang 自身的文案 bug，值得当趣闻记）：
  ```python
  if self.speculative_algorithm is not None:
      assert not self.enable_mixed_chunk, "enable_mixed_chunk is required for speculative decoding"
  ```

**CUDA graph**
- 兼容，且有专用 graph runner（`eagle_draft_cuda_graph_runner.py`、`eagle_draft_extend_cuda_graph_runner.py`、`frozen_kv_mtp_cuda_graph_runner.py`）。
- 代价是显存。文档 OOM 章原文：
  > "**Out of Memory (OOM)?** Speculative decoding may increase GPU memory usage because the draft tree, CUDA graphs, and verification-related buffers consume additional VRAM."
- 明确不兼容：`flex_attention` backend（"Speculative decoding is currently not supported with Flex Attention backend"）、`hpc_ops` backend、XPU graph、`CPUGraphRunner`（"CPUGraphRunner does not support speculative inference yet."）。
- 自适应模式为每个候选 step 层各捕获一套 CUDA graph（显存换切换速度）。

**overlap scheduler（Spec V2，默认开）—— 最容易踩的坑**
文档 "Speculative Decoding V2" 节原文：
> - The overlap scheduler currently only supports `--speculative-eagle-topk 1`; **set `--speculative-eagle-topk 1` explicitly**.
> - If you explicitly set `--speculative-eagle-topk > 1`, the server will error.
> - If you omit `--speculative-eagle-topk`, auto-tuning may pick `topk > 1` for some models (e.g. Llama). **This is incompatible with the overlap scheduler and may not always trigger an immediate config error**, so set `--speculative-eagle-topk 1` explicitly.

→ **Llama 的自动值恰好是 topk=4**。即"什么都不配、用默认"在 Llama 上会撞进这个文档亲口承认"可能不会立刻报错"的组合。
CPU 设备强制关 overlap：「Overlap schedule is not implemented for speculative decoding on CPU.」

**PD 分离（disaggregation）**
- **官方支持**：有 `eagle_disaggregation.py` / `dspark_disaggregation.py`；`pd_disaggregation.mdx` 给出 prefill 与 decode **两侧都带 `--speculative-*`** 的完整启动脚本（示例 prefill 用 `steps=1/topk=1/draft-tokens=2`，decode 用 `steps=3/topk=1/draft-tokens=4`，且**都带 `--disable-radix-cache`**）。
- `carries_draft_hidden_states()`：仅 EAGLE 家族在 prefill→decode 传输里携带 draft hidden states（"STANDALONE's vanilla draft ignores them"）。
- 不兼容：decode 侧 radix cache；`--enable-linear-replayssm-spec` 在 PD prefill 侧不支持；`mamba_radix_cache_strategy=extra_buffer_lazy`「Compatible with speculative decoding (EAGLE/NGRAM/DSPARK/DFLASH); **not supported under PD disaggregation**」。

> **这是全库唯一一处"PD 分离 × 投机"的官方明确文档**（vLLM 侧完全没有）。写 `23-与其它优化的相互作用-量化与KVcache与PD分离` 时要以 SGLang 为主要素材。

**其它「不兼容 XXX」断言原文汇总**

| 冲突项 | 原文 |
|---|---|
| Pipeline parallelism | `"Pipeline parallelism is not compatible with overlap schedule, speculative decoding"`（`pp_size > 1` 时断言） |
| Flex Attention | `"Speculative decoding is currently not supported with Flex Attention backend"` |
| hpc_ops backend | `"hpc_ops backend does not support speculative decoding for now."` |
| DWDP | `"DWDP does not support speculative decoding (MTP/draft workers)"` |
| `--weight-cache-mode` | `"--weight-cache-mode is not supported together with speculative decoding (--speculative-algorithm): the weight cache daemon does not export the draft model's weights."` |
| HiSparse prefetch | 「The prefetch is enabled automatically for eligible models (no pipeline parallelism, **no speculative decoding**)」 |
| 动态 KV 池（byte buffer） | 「...not yet compatible with PD disaggregation or speculative decoding.」 |
| DP attention | DFLASH / STANDALONE / NGRAM **均不支持** `--enable-dp-attention`（EAGLE/EAGLE3 支持） |
| `topk>1` + `page_size>1` | 仅 `flashinfer / fa3 / triton` 支持；`flashmla/trtllm_mla/cutlass_mla` 报错 |
| `trtllm_mha` backend | `"trtllm_mha backend only supports topk = 1 for speculative decoding."` |
| DFLASH + `return_hidden_states` | `"DFLASH speculative decoding does not support return_hidden_states yet."` |
| `return_sampling_mask` | `"return_sampling_mask is not supported with speculative decoding."` |
| **LoRA** | **兼容** NGRAM/EAGLE/NEXTN/EAGLE3/DFLASH/DSPARK，但源码注释：「Adapters apply to the target only; **a shared draft runs unadapted**.」；且 LoRA+spec 不支持 `--speculative-adaptive`、非 static 的 ragged verify、`experimental_sgl_trtllm` MoE runner |

> **LoRA 这条是 vLLM/SGLang 的关键分野**：vLLM 兼容矩阵直接标 SD × LoRA = ❌；SGLang 支持，但**草稿模型不带 adapter**——这会系统性压低接受率（草稿在预测"没被微调过的模型"会说什么）。这条应写进 `21-草稿模型怎么训-对齐与在线蒸馏`。

**自适应投机（`--speculative-adaptive`）自身限制**，文档 "Current support" 原文：
> - Only `--speculative-algorithm EAGLE` or `EAGLE3`
> - Only `--speculative-eagle-topk 1`
> - If either condition is not met, SGLang falls back to static speculative settings

Frozen-KV MTP 断言：「Frozen-KV MTP does not support adaptive speculative decoding yet.」

### 2.6 采样兼容性

**结论先行：SGLang 的 EAGLE 不是"只支持 greedy"，官方文档里没有任何 "EAGLE only supports greedy / temperature=0" 的说法。** 但有三个必须知道的陷阱。

核验文件：`python/sglang/srt/speculative/eagle_utils.py::eagle_sample`、`spec_utils.py`、`layers/logprob_processor.py::compute_spec_logprobs`

| 采样特性 | 状态 | 依据 |
|---|---|---|
| temperature > 0 | ✅ verify 路径做 `F.softmax(next_token_logits / expanded_temperature)` | `eagle_utils.py` L782-788 |
| `top_k` | ✅ `if sampling_info.need_top_k_sampling: target_probs = top_k_renorm_prob(...)` | L790-796 |
| `top_p` | ✅ `if sampling_info.need_top_p_sampling: target_probs = top_p_renorm_prob(...)` | L800-806 |
| repetition/presence penalties、logit_bias | ✅ 但源码注释明说是**放松版**：「This is a relaxed version of penalties for speculative decoding.」 | L692-714 |
| grammar / 结构化输出 | ✅（verify 前 apply grammar mask）；`supports_grammar_overlap()` 对 EAGLE/STANDALONE/DFLASH 家族为 True，**NGRAM 为 False**（「NGRAM drafts from a host corpus lookup, so it stays synchronous by design」） | `eagle_worker_common.py`, `spec_info.py` |
| logprobs / top_logprobs / token_ids_logprobs | ✅ 走 `compute_spec_logprobs()`（EAGLE/NGRAM/DFLASH/DSPARK 都调用） | `logprob_processor.py` L352+ |
| **n > 1** | ✅ 与投机正交——`parallel_sample_num` 在 `tokenizer_manager.py` 层 fan-out 成多请求。源码 FIXME：「When using batch and parallel_sample_num together, the perf is not optimal.」 | L1802-1868 |
| **beam search** | ❌ **SGLang 根本没实现 beam search**（只有 OpenAI 协议里一个未用的 `best_of` 字段）。所以"投机下 beam search 受限"这个问题在 SGLang 上不成立 | |
| 自定义 logit processor | ⚠️ 有 issue 报告被绕过（#26330） | |
| deterministic inference | ❌ 与 `--speculative-use-rejection-sampling` 互斥：「the sampling kernel draws coins from the global RNG and is not batch-invariant.」 | |

#### ⚠️ 三个采样陷阱（本节最有价值的部分）

**(1) 在 CPU / NPU / ROCm(HIP) / XPU 上，verify 被强制走 greedy argmax，无视你的 temperature。**
```python
if sampling_info.is_all_greedy or _is_cpu or _is_npu or _is_hip or _is_xpu:
    target_predict = torch.argmax(next_token_logits, dim=-1)
    ... verify_tree_greedy_func(...)
else:
    ...  # 才走 temperature/top_k/top_p 的采样 kernel
```
即：**非 CUDA 平台上开投机 + temperature>0，实际拿到的是贪心结果**（`eagle_utils.py` L730 附近）。遍历 `docs/` 未找到相关说明，**判定为文档未记载**。

**(2) 默认的验证算法不是经典 rejection sampling。**
默认 `--speculative-use-rejection-sampling=False` 时走 `tree_speculative_sampling_target_only`，且 `draft_probs = torch.zeros_like(target_probs)`（**草稿分布被丢弃**）。经典 Leviathan 链式 rejection sampling 需显式开 `--speculative-use-rejection-sampling`，且要求 topk=1、算法为 EAGLE/EAGLE3、无 accept threshold、无 deterministic inference。
→ **按本库铁律一的口径：SGLang 的默认配置不是 L1（分布无损），而是一种未标注的近似。** 这条必须写进 `07-无损的三种口径`——它是"引擎默认值和论文承诺不一致"的最好例子。

**(3) accept threshold 默认 1.0，是"有损开关"的入口。**
`--speculative-accept-threshold-single` / `--speculative-accept-threshold-acc` 默认都是 `1.0`（严格）。调低会更激进地接受，即**主动牺牲分布保真换速度**。文档只写 "Lower values accept more aggressively"，**未标注这会改变输出分布**——属于铁律一说的"没标口径的无损"的反面：没标口径的有损。

### 2.7 已知 bug / 反复出现的坑

搜索规模：`repo:sgl-project/sglang is:issue eagle in:title` 共 **132** 条；`speculative in:title` 共 **171** 条；`"accept length"` 共 **100** 条。

> ⚠️ **区分声明**：以下除标注 "maintainer/roadmap" 外，**均为用户提交的 issue，不等于官方确认**；多数页面未见维护者回复。

| 主题 | 代表 issue |
|---|---|
| **TP 下 rank 分歧 → 集合通信死锁**（同一根因至少 3 次） | #28815(closed)、#31071(open, "EAGLE greedy verify lacks TP broadcast → ranks diverge on accepted tokens → collective deadlock (tp>1)")、#35144(closed, XPU TP=2 warmup hang) |
| verify kernel 挂死 | #35822(open, 2026-08-21, AMD 上 hang 在 `tree_speculative_sampling_target_only`) |
| 采样正确性 | #35771(open, 2026-08-21) — 见下 |
| PD 分离 + DP attention 死锁 | #32527(open)、#33642(open) |
| radix/prefix 复用被打穿 | #32459(open) — 见下；关联 #20451（`cached_tokens` 在投机下误报）、#19796 |
| HiCache/L3 交互 | #28873(closed)、#32176(open, "HiCache loadback 后 GLM-5.2 EAGLE acceptance 静默坍塌")、#28011(closed) |
| CUDA graph 捕获失败 / 显存 | #22359(closed)、#31588(open, multi-layer EAGLE 自动 KV sizing 分配 draft pool 时 OOM)、#29857(open, EAGLE/MTP 开启后 KV pool profiler 白留 ~50GB)、#25512(closed, verify segfault during CUDA graph replay padding) |
| 结构化输出 / grammar | #29660(closed)、#31978(closed, "Multi-layer EAGLE verify ignores grammar vocab mask, aborting structured output")、#31711(open) |
| attention backend 特化崩溃 | #22679(closed, "EAGLE family + sliding-window attention broken on both flashinfer and triton")、#27367(closed)、#30209/#32377/#25563（GLM FP4 + EAGLE 在 Blackwell 上 illegal memory access） |
| **DSPARK 精度回退** | #32038(open)、#34959(open, "DSPARK silently corrupts identifiers on DeepSeek-V4-Flash, making speculative decoding unsafe")、#33800(open, "DSpark draft depth 5 (the checkpoint default) corrupts output on SM120 — depths 3, 4, 6, 7 are clean") |
| **接受长度远低于宣传值** | #27924(closed, "MiMo-V2.5-Pro-FP4-DFlash: DFlash drafter accept_length anomaly (**1.42 vs advertised 6.30**)") |
| 自定义 logit processor 被绕过 | #26330(closed) |

**三个最值得写进正文的**：

**① #35771（open，2026-08-21，@nanjiangwill，Contributor）— 采样正确性 bug，波及 EAGLE/NEXTN/MTP/DFlash**
标题：*[Bug] Target-only speculative sampler can accept a zero-probability draft at RNG boundary*
> "`tree_speculative_sampling_target_only` can accept a draft token whose target probability is exactly zero when its verification coin is also exactly zero."
> "A draft model may legitimately propose a token removed by the target model's top-k/top-p filtering. For that token, `target_prob_single == 0` and `prob_acc == 0`. **If the RNG returns zero, the inclusive comparison evaluates `0 <= 0` and accepts the impossible draft.**"
> "**This violates the target distribution: a token with zero target probability must never be accepted.**"
> "The bug is not confined to DFlash. **Non-greedy EAGLE verification calls the same target-only sampler when rejection sampling is disabled** ... **NEXTN/MTP reuses this EAGLE verification path, so it has the same boundary condition.**"
报告称在 upstream `main` commit `f825d729363136a2d4a4b330fa694d0b37a878fa` 上仍存在。
→ **这是"无损"在工程实现里被浮点/RNG 边界打破的教科书级案例**，直接写进 `07-无损的三种口径` 与 `27-常见误解与判据`。

**② #32459（open，2026-07-27，divyvasal，社区用户）— EAGLE 打穿 radix prefix 复用**
标题：*[Bug] EAGLE speculative decoding defeats radix prefix reuse for multi-turn traffic (GLM-DSA NVFP4, v0.5.16) — no crash, silent 97%→40-53% reuse collapse*
> "enabling EAGLE speculative decoding **collapses radix prefix reuse for multi-turn agentic traffic — at any draft length.**"
> "The engine keeps serving but behaves as if cached prefixes are not reused (TTFT and throughput match full re-prefill)..."
> "HiCache on/off does not change the result — **the regression tracks EAGLE alone.**"

| 配置 | Input tok/s | TTFT p50 | Cache Hit（≥20K-token prompts） |
|---|---|---|---|
| TP8，不开投机 | 43,981 | 0.54 s | **97%** |
| TP8 + EAGLE steps=5/draft=6 | 12,881 | 2.15 s | **53%** |
| TP8 + EAGLE steps=3/draft=4 | 16,811 | 2.68 s | **40%** |

（模型 GLM-DSA NVFP4，v0.5.16；**硬件型号原文未给出**，只知 TP=8。页面未见维护者回复。）
→ **这是"投机与 prefix caching 相互作用"最有冲击力的一组数字**：输入吞吐掉到 1/3。写 `23-与其它优化的相互作用` 必用，但要标注"用户报告、未获官方确认"。

**③ #32038（open，2026-07-22，fangtang1121，社区用户）— DSPARK 精度回退**
AIME25：baseline **97.08%** → DSPARK **93.96%**（−3.12pp）。硬件 4× NVIDIA B300 SXM6，TP=4，`--speculative-algorithm DSPARK --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`，`cuda-graph-max-bs 128`，`max-running-requests 256`。（未见维护者回复。）

**官方 roadmap（maintainer 出品）**：#23005 "Speculative Decoding Development Roadmap (2026 Q2)"（closed，作者 Qiaolin-Yu，assignees Qiaolin-Yu / hnyls2002 / kpham-sgl，labels: roadmap, speculative-decoding）。关键条目：Spec v2 → "Enable v2 by default"、"Fully overlap (remove wait_for_verify sync)"、"Constrained decoding support"、**"Top-k > 1 (optional feature)"**（说明 topk>1 在 overlap 下当时仍是待办）；"Make speculative decoding compatible with piecewise-cuda-graph"；DFlash 初始实现；Adaptive(#23705)；Parallel Spec Decoding(#27462)；N-gram(#21052)。
N-gram roadmap #21052（kpham-sgl）自陈两条短板：
> 1. **No external corpus support** — trie 只用当前会话 token（v0.5.18 已加 `--speculative-ngram-external-corpus-path`，部分落地）
> 2. **Poor scaling** — "Insert paths exhibit near-O(n²) memory growth for long inputs; capacity-based eviction struggles when full"

### 2.8 SGLang 官方 benchmark 数字（含缺项标注与可信度警告）

**(a) 文档首页**（https://docs.sglang.io/advanced_features/speculative_decoding ）

| 配置 | Throughput |
|---|---|
| SGLang（不开投机，1× H100） | 158.34 tokens/s |
| SGLang + EAGLE-2（1× H100） | 244.10 tokens/s |
| SGLang + EAGLE-3（1× H100） | 373.25 tokens/s |

模型 LLaMA-Instruct 3.1 8B，数据集 MT-bench，硬件 1× H100。
**batch size 原文未给出**（从量级看应为 bs=1 单流）；**未区分是单用户输出速度还是总吞吐**；**accept length 原文未给出**；**草稿模型 原文未给出**。→ 按铁律二，这组数字**口径不全，不可横向比较**。

**(b) SpecBundle 官方示例输出（有真实 accept length）**，`specbundle_usage.mdx`：
Qwen3-30B-A3B-Instruct-2507 + `lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex`，`--tp 4`，`steps=3/topk=1/draft-tokens=4`：

| 数据集 | batch_size | latency | output_throughput | **accept_length** | accuracy |
|---|---|---|---|---|---|
| mtbench (5 q) | 1 | 12.2328 s | 319.714 tok/s | **2.170366** | null |
| gsm8k (100 q) | 1 | 37.4208 s | 373.616 tok/s | **2.643411** | 0.96 |

**硬件 原文未给出。**
→ ⭐ **官方自己示例里的真实接受长度只有 2.17 / 2.64**（K=4，即上限 5），远低于宣传口径里动辄 4–6 的说法。这一条对 `05-接受率alpha` 与 `27-常见误解与判据` 极有价值。

**(c) Intern-S2-Mobius cookbook（唯一一组要素齐全的）**
硬件 **H200，TP=1，FP8**；`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`；8K-in / 1K-out：
> "We measured **accept-length ~3.9/4 draft tokens** at 8K-in / 1K-out, roughly tripling single-stream decode speed (**median TPOT 9.79 ms → 3.13 ms at conc=1, 14.26 ms → 6.84 ms at conc=16**) and roughly doubling mid-concurrency total throughput (**9358 → 18029 tokens/s at conc=16, 21395 → 26033 tokens/s at conc=64**). **The high-throughput recipe stays spec-off because once you can batch wide, its saturation point is higher (34786 tokens/s at conc=256 vs the spec recipe's peak at conc=64).**"

→ 即：**conc=256 不开投机（34786 tok/s） > conc=64 开投机（26033 tok/s）**。**这是官方文档里最直接的"投机在高并发下输给不投机"的数据**，而且七项口径基本齐全（缺 α 的定义口径与精确的 draft 权重名）。

**(d) SpecForge v0.3.0 博客**（https://www.lmsys.org/blog/2026-08-04-specforge-v0-3 ）

| 组合 | 硬件 | 并发 | 数据集 | speedup | accept length |
|---|---|---|---|---|---|
| Qwen3.6-27B + Domino-B16 | 2×A100 | **1**（greedy） | GSM8K/MATH500/HumanEval/MBPP/MT-Bench/Alpaca | 5.25× / 5.72× / 4.98× / 4.49× / 3.44× / 3.34× | 原文未给出 |
| Qwen3.6-27B + Domino | 2×A100 | **32** | 同上 | **1.48–2.11×** | 原文未给出 |
| Qwen3.5-397B-A17B + DFlash (block size 8) | 8×B200 | 1 | — | 最高 4.31× | 原文未给出 |
| Kimi-K3 + DSpark | 8×B300 | **1** | GSM8K | 3.14× | 原文未给出 |
| Kimi-K3 + DSpark | 8×B300 | **16** | — | **1.37–2.36×** | 原文未给出 |
| 训练吞吐（解耦 vs 同置，3 server + 5 trainer） | 8×H20 | — | — | 1.10× | n/a |

**speedup 的口径（延迟还是吞吐）原文未说明。**

**(e) SpecForge 首发博客**（https://www.lmsys.org/blog/2025-07-25-spec-forge/ ）：Llama 4 Maverick draft **2.18×**、Llama 4 Scout draft **2.0×**，MT-Bench；**硬件 / batch size / accept length 全部原文未给出**。固定 `speculative-eagle-topk=8`、`speculative-num-draft-tokens=10`（为启用 tree attention）；训练数据 320K（ShareGPT + UltraChat）。

**(f) SpecBundle phase 1 博客**（https://lmsys.org/blog/2025-12-23-spec-bundle-phase-1/ ）：只有笼统的 "up to **4× end-to-end inference speedup**"，**正文没有任何带硬件/batch/accept length 的数值表**。训练集 Perfect-Blend，1.4M 样本（对比原 EAGLE 论文的 320K）。

#### ⚠️⚠️ 关于官方 benchmark 可信度的重大发现

GLM-5.2 cookbook（`docs/cookbook/autoregressive/GLM/GLM-5.2.mdx` L60）原文：
> "Speed numbers are measured with `--random-range-ratio 1.0`, `--flush-cache`, on `main @ 09ca4fc` (H200 FP8 cells: `v0.5.14 @ 49e384ce`). **Spec cells pin the EAGLE acceptance length via the serve env `SGLANG_SIMULATE_ACC_LEN`** (low-latency 5-1-6 = 3.5; FP8 balanced 1-1-2 = 2; NVFP4 balanced 2-1-3 = 2); high-throughput has no spec."

→ **GLM-5.2 Deploy 面板里投机档位的速度数字，接受长度是用环境变量人为写死模拟的，不是真实草稿模型跑出来的。**
这一条应当在 `26-社区精彩解释精选` 或 `27-常见误解与判据` 里单列：**引用任何引擎官方 benchmark 之前，先查它有没有用"模拟接受率"开关**。vLLM 侧的对应物是 `rejection_sample_method: "synthetic"` + `synthetic_acceptance_length`——两家都提供了"假装接受率是 X"的基准测试模式，且都可能出现在官方数字里。

**SpecBundle 性能 dashboard**（https://sgl-project.github.io/SpecForge/SpecBundle/index.html ）是纯 JS 前端，HTML 里无嵌入数据，**逐行 benchmark 表未取到（未查证）**。

### 2.9 SGLang 侧「未查证」清单

1. SpecForge 各 release tag 的日期（GitHub API 匿名限流）；仅确认 README News 中的 v0.2.0（2025-12）与 v0.3.0（2026-08）。
2. SpecBundle dashboard 的逐行数字（JS 渲染）。
3. SpecForge 论文 arXiv:2603.18567 的正文表格（仅确认标题与 bibtex）。
4. 文档首页 158/244/373 tok/s 那组数字的 batch size / accept length / 草稿模型。
5. 是否有维护者在 issue 中以第一人称说"大 batch 请关投机"——**未找到**；但官方文档（cookbook + adaptive 文档）已多处明文写出该结论，证据强度更高（见 §7）。
6. `--speculative-use-rejection-sampling` 何时引入、是否计划改为默认。
7. 非 CUDA 平台强制 greedy verify 是否在任何官方文档中说明——遍历 `docs/` 未找到，判定为**文档未记载**。

---

## 3. TensorRT-LLM

### 3.0 版本口径 + ⚠️ 本次调研最大的单条发现

| 项 | 值 |
|---|---|
| `main` 分支 `__version__` | **`1.3.0rc25`** |
| 最新预发布 tag | `v1.3.0rc24`（2026-08-12） |
| 最新**正式版** | **`v1.2.1`**（2026-04-20） |
| **TensorRT engine 后端被删除** | **Release 1.2** |

#### ⭐⭐ 「老 TRT engine 路线」在 1.2 已经不存在了

`docs/source/legacy/tensorrt-backend-removal.md` 原文：
> **Breaking change.** The TensorRT engine backend has been removed. **PyTorch is now the sole execution backend for TensorRT LLM** (AutoDeploy, built on the PyTorch backend, remains available).

Release Notes 1.2 的 API Changes（BREAKING CHANGE）：
> **[BREAKING CHANGE] TensorRT backend removed.** PyTorch is now the sole execution backend. `LLM(backend="tensorrt")` now raises a `ValueError`; `TrtLlmArgs`, `tensorrt_llm._tensorrt_engine.LLM`, the **`trtllm-build` / `trtllm-refit` / `trtllm-prune` CLIs**, the `--backend tensorrt` CLI choice, and the **per-model `convert_checkpoint.py` scripts** have all been removed.

**直接后果——这对本库的历史篇至关重要**：
**只在 TRT engine 里实现的 Medusa / ReDrafter / EAGLE-1 / EAGLE-2 / Lookahead Decoding，在 TensorRT-LLM 1.2+ 已经实际无法运行。**
它们的 C++ kernel 还留在树里（`cpp/tensorrt_llm/layers/medusaDecodingLayer.cpp`、`lookaheadDecodingLayer.cpp`、`eagleDecodingLayer.cpp`、`thop/redrafterCurandOp.cpp`），但 Python 侧的模型定义与 `examples/` 目录已删除——`examples/` 下现在只剩 `ngram/` 一个投机相关目录（GitHub contents API 实测：**无 `medusa/`、`eagle/`、`redrafter/`、`lookahead/`、`draft_target_model/`**）。

⚠️⚠️ **同时 `docs/source/legacy/advanced/speculative-decoding.md`（渲染地址 `nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html`）仍在描述 Medusa/ReDrafter/EAGLE-1/2/Lookahead，并链向 `examples/medusa/README.md` 等已删除路径——这些链接现在全是 404。**
**这是当前最容易被二手资料误导的点**：网上讲"TensorRT-LLM 支持 Medusa/ReDrafter/Lookahead"的文章，引用的都是这个已经腐烂的页面。

> **写 `15-谱系图与被淘汰的分支` 时这是一条决定性证据**：ReDrafter（Apple，2024）与 Lookahead Decoding（2023）**在唯一实现过它们的生产级引擎里被整段删除**。加上 §0 表里 Medusa 的三家态度——**2026 年的实际部署面已经收敛到 EAGLE-3 / MTP / n-gram-suffix / DFlash-DSpark 四条线。**

### 3.1 支持哪些方法（分两条路线说）

#### 当前 PyTorch backend（`tensorrt_llm._torch`，1.3.0rc25）

依据：`docs/source/features/speculative-decoding.md` + `tensorrt_llm/llmapi/llm_args.py` 里每个 config 类的 `supports_backend()`。

| 方法 | PyTorch backend | config 类 / `decoding_type` | 代码判据 |
|---|---|---|---|
| **Draft-Target（独立草稿模型）** | ✅ | `DraftTargetDecodingConfig` / `"Draft_Target"`，YAML 写 `DraftTarget` | `backend == "pytorch" or "_autodeploy"` |
| **Medusa** | ❌ | `MedusaDecodingConfig` / `"Medusa"` | `backend not in ("pytorch","_autodeploy")` → PyTorch 下直接抛错 |
| **EAGLE-1 / EAGLE-2** | ❌ | `EagleDecodingConfig` / `"Eagle"` | 传 `Eagle` 会**被自动转成 Eagle3 并警告**；EAGLE v1/v2 权重不兼容 |
| **EAGLE-3** | ✅ | `Eagle3DecodingConfig` / `"Eagle3"` | 默认 `eagle3_one_model=True` |
| **ReDrafter (Apple)** | ❌ | **PyTorch 侧根本没有 config 类** | 仅存在于 TRT engine 的 `explicit_draft_tokens` 模式 |
| **Lookahead Decoding** | ❌ | `LookaheadDecodingConfig` / `"Lookahead"` | `backend not in ("pytorch","_autodeploy")` |
| **MTP** | ✅ | `MTPDecodingConfig` / `"MTP"` | `backend in ("pytorch","_autodeploy")` |
| **NGram / prompt lookup** | ✅ | `NGramDecodingConfig` / `"NGram"` | `backend == "pytorch"` |

PyTorch backend **额外**支持（老路线完全没有的新方法）：

| 方法 | config 类 / `decoding_type` | 说明 |
|---|---|---|
| **PARD**（PARallel Draft） | `PARDDecodingConfig` / `"PARD"` | target-independent，用 mask token 一次前向出 K 个草稿；arXiv:2504.18583 |
| **DFlash** | `DFlashDecodingConfig` / `"DFlash"` | target-dependent，用 target 层 hidden states 做 cross-attention；arXiv:2602.06036 |
| **DSpark** | `DSparkDecodingConfig` / `"DSpark"` | DeepSeek DeepSpec；半并行 + Markov 头 + confidence 头 |
| **SA（Suffix Automaton）** | `SADecodingConfig` / `"SA"`；或作为增强 `sa_config` 挂在 Eagle3/MTP/PARD 上 | GPU 原生后缀自动机，model-free |
| **User-provided** | `UserProvidedDecodingConfig` / `"User_Provided"` | 自己实现 `Drafter.prepare_draft_tokens()` |
| **AUTO** | `AutoDecodingConfig` / `"AUTO"` | 启发式自动选（实测就是 NGram，见 3.6） |
| **SaveState** | `SaveHiddenStatesDecodingConfig` / `"SaveState"` | 不做投机，只 dump hidden states 供 EAGLE3 训练 |

`SpeculativeConfig` 是一个 discriminated union（`llm_args.py` 末尾）：
```python
SpeculativeConfig: TypeAlias = Annotated[
    Union[DraftTargetDecodingConfig, Eagle3DecodingConfig,  # Must be before EagleDecodingConfig since it's a subclass
          EagleDecodingConfig, LookaheadDecodingConfig, MedusaDecodingConfig, MTPDecodingConfig,
          NGramDecodingConfig, SADecodingConfig, UserProvidedDecodingConfig, SaveHiddenStatesDecodingConfig,
          PARDDecodingConfig, DFlashDecodingConfig, DSparkDecodingConfig, AutoDecodingConfig],
    Field(discriminator="decoding_type"),
]
```

> ⚠️ **一个极容易读错的官方句子**。features 文档里有：
> > Note: The PyTorch backend supports only `Eagle3`. `decoding_type: Eagle` is accepted as a backward-compatible alias for `Eagle3`, but EAGLE (v1/v2) draft checkpoints are incompatible.
>
> 这句**只针对 EAGLE 家族**（紧跟在讨论 `Eagle` 别名的段落后），**不是**说 PyTorch backend 只支持 Eagle3 一种投机方法。同一页上方明确列出 CLI 可用的 `decoding_type`：`MTP` / `Eagle3` / `NGram` / `DraftTarget` / `PARD` / `DFlash` / `SA`。
> **多个二手来源（含搜索引擎摘要）把这句误读成「PyTorch 只支持 Eagle3」——写正文时务必避开。**

#### 老 TRT engine 路线（≤1.1.x，已删除，仅供考古）

| 方法 | `--speculative_decoding_mode` 取值 |
|---|---|
| Draft-Target-Model | `draft_tokens_external` |
| NGram（V1 workflow） | `draft_tokens_external`（与 DTM 共用） |
| Medusa | `medusa`（**只支持 Vicuna**） |
| ReDrafter | `explicit_draft_tokens`（logits 预测 + beam search + 接受判定全在 engine 内） |
| EAGLE-1 / EAGLE-2 | `eagle` |
| Lookahead | `lookahead_decoding` |

### 3.2 配置参数名与默认值（抄自 `llm_args.py`）

#### `DecodingBaseConfig`（所有投机 config 的公共基类）

| 字段 | 类型 | 默认值 | 官方 description（节选原文） |
|---|---|---|---|
| `max_draft_len` | `Optional[NonNegativeInt]` | `None` | "The maximum number of draft tokens." |
| `max_total_draft_tokens` | `Optional[int]` | `None` | linear tree 时 `== max_draft_len`；static/dynamic tree 时 `>=` |
| `speculative_model` | `Optional[Union[str,Path]]` | `None` | **`validation_alias=AliasChoices("speculative_model", "speculative_model_dir")`** — **两个名字等价**。可以是 HF Hub model ID（自动下载）或本地路径 |
| **`max_concurrency`** | `Optional[PositiveInt]` | `None` | "When specified (>0), speculation will be **disabled at batch sizes above this value**. Otherwise, speculation will always be on. **PyTorch backend only.**" |
| **`draft_len_schedule`** | `Optional[dict[int,int]]` | `None` | "Developer interface: dynamically adjust draft length based on active batch size in runtime." 例 `{4:4, 8:2, 32:1}` → BS 33+ 隐式 draft_len=0（投机关闭）。与 `max_concurrency` **互斥** |
| `load_format` | `Optional[str]` | `None` | 草稿模型的 load format |
| `acceptance_rate_window_size` | `Optional[NonNegativeInt]` | `None` | 滚动窗口 N；不设或 0 关闭。PyTorch only |
| `acceptance_rate_threshold` | `Optional[float]`（0.0~1.0） | `None` | 滚动平均真实接受率低于阈值 → **永久关闭投机** |
| `use_rejection_sampling` | `bool` | **`False`**（prototype） | 为 one-model 路径开 rejection sampling；全 greedy batch 永远走 argmax 快路径。非 dynamic-tree 的 one-model 路径**需要 FlashInfer** |
| `allow_advanced_sampling` | `bool` | `False`（**deprecated，no-op**） | "DEPRECATED: no-op kept for backward compatibility." |
| `advanced_sampling_mode` | `AdvancedSamplingMode` | **`FULL`** | `full` / `no_topk` / `no_topp` / `no_topk_no_topp`，跳过被禁用的过滤 kernel |
| `enable_penalty` | `bool` | **`False`**（prototype） | one-model 投机下的 repetition/presence/frequency penalty 总开关；关闭时带 penalty 的请求会**在准入时被拒绝** |

私有属性（非用户接口但决定行为）：`_allow_chain_drafter=True`、`_allow_greedy_draft_tokens=True`、`_allow_separate_draft_kv_cache=True`、`_use_shared_kv_cache=False`。

#### `EagleDecodingConfig` / `Eagle3DecodingConfig`

| 字段 | 默认值 | 说明（原文节选） |
|---|---|---|
| `decoding_type` | `"Eagle"` / `"Eagle3"` | |
| `eagle_choices` | `None` | 静态树结构，与 `use_dynamic_tree` 互斥。**已废弃**：`validate_eagle_choices` 打印 "The eagle_choices/static tree feature is deprecated and **will be removed in release 1.4**." |
| `greedy_sampling` | **`True`** | greedy（Top-1 + token equality acceptance） vs typical acceptance + multinomial |
| `posterior_threshold` | `None` | typical acceptance 的最小 token 概率阈值（Medusa 论文的 epsilon） |
| `use_dynamic_tree` | **`False`** | 打开 EAGLE-2 式动态树 |
| `dynamic_tree_max_topK` | `None` | `use_dynamic_tree=True` 时**必填** |
| `num_eagle_layers` | `None` | "**Deprecated TensorRT-only field**... Do not use on the PyTorch backend."（内部强制设为 `max_draft_len`） |
| `max_non_leaves_per_layer` | `None` | |
| **`eagle3_one_model`** | **`True`** | "Always uses the one-model implementation (draft as submodule). **Setting False is ignored and falls back to True; the two-model path is deprecated and will be removed in a future release.**" |
| `eagle3_layers_to_capture` | `None` | 默认 `{1, num_layers//2-1, num_layers-4}`，`num_capture_layers` 默认 **3** |
| `eagle3_model_arch` | **`"llama3"`** | 另一取值 `"mistral_large3"` |
| `sa_config`（仅 Eagle3） | `None`（beta） | `SAEnhancerConfig` |
| `_max_batch_size`（PrivateAttr） | `None` | dynamic tree buffer 用，必须等于全局 `max_batch_size`，由 `py_executor_creator` 自动填 |

dynamic tree 下 `max_total_draft_tokens` 的约束（文档原文）：
> Must satisfy `max_draft_len <= max_total_draft_tokens <= dynamic_tree_max_topK * max_draft_len`. Defaults to `dynamic_tree_max_topK * max_draft_len` if not set.

dynamic tree 的限制（文档 note 原文）：
> Dynamic tree mode is currently **not supported** for models that use sliding window attention or **MLA (Multi-Latent Attention), such as DeepSeek and gpt-oss models.**

#### `MTPDecodingConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `decoding_type` | `"MTP"` | |
| `use_relaxed_acceptance_for_thinking` | **`False`** | 推理模型 thinking 阶段放松接受（**L3 有损**） |
| `relaxed_topk` | **`1`**（PositiveInt） | |
| `relaxed_delta` | **`0.0`**（NonNegativeFloat） | 保留 `log(P(top1)) - log(P(t)) <= delta` 的候选 |
| `use_mtp_vanilla` | **`False`** | True 强制串行 MTP 层；False 时单层 checkpoint 走 EAGLE-style MTP |
| `mtp_eagle_one_model` | **`True`** | "Setting False is ignored and falls back to True; the two-model path is deprecated" |
| `use_dynamic_tree` / `dynamic_tree_max_topK` | `False` / `None` | |
| `sa_config` | `None`（beta） | |
| **`num_nextn_predict_layers`** | `None`，**`init=False`** | ⚠️ **已废弃且不可手工设置**："Number of MTP layers in the model checkpoint. **Auto-populated from the target pretrained config**... **Do not set manually.**" 传入告警 `'num_nextn_predict_layers' is deprecated ... Use 'max_draft_len' instead.`（传入值会被搬到 `max_draft_len`） |
| `begin_thinking_phase_token` | **`128798`** | |
| `end_thinking_phase_token` | **`128799`** | |

⚠️ **文档与代码不一致**：features 文档正文仍写 "`num_nextn_predict_layers`: Number of MTP modules to use. Currently must match `max_draft_len`."，而代码已把它标为 deprecated/`init=False`。官方 blog02 与部分 deployment guide 的 YAML 里还在用 `num_nextn_predict_layers: 3`（会触发 deprecation warning）。

#### `NGramDecodingConfig`

| 字段 | 默认值 |
|---|---|
| `decoding_type` | `"NGram"` |
| `max_matching_ngram_size` | **`2`**（PositiveInt） |
| `is_keep_all` | **`True`** |
| `is_use_oldest` | **`True`** |
| `is_public_pool` | **`True`** |

`max_draft_len` 必须 > 0，否则 `ValueError: max_draft_len must be > 0 for NGram`。（老 v1.1 的 `examples/ngram/README.md` 里 `max_draft_len` 默认写 4。）

#### `DraftTargetDecodingConfig`
`decoding_type="Draft_Target"`；`_draft_target_one_model`（PrivateAttr）默认 **`True`** → 走 `DRAFT_TARGET_ONE_MODEL` 模式。校验：`max_draft_len > 0`、`speculative_model` 必填。

#### `SADecodingConfig` / `SAEnhancerConfig`
- `SADecodingConfig`：`max_matching_ngram_size` 默认 **`-1`**（-1 = 后缀自动机最长匹配；0 非法）；`enable_global_pool` 默认 **`False`**（限制原文："at most 1024 concurrent slots; suffix matching is capped at 64 tokens per request"）；`global_pool_size` 默认 `None`（开全局池时 = `max(64, max_batch_size)`）。
- `SAEnhancerConfig`（挂在 Eagle3/MTP/PARD 的 `sa_config`）：`threshold` 默认 **`4`**、`enable_global_pool` 默认 `False`。用户级别名是 `use_sa_spec=True` / `sa_spec_threshold=4`。

#### `PARDDecodingConfig` / `DFlashDecodingConfig` / `DSparkDecodingConfig`
- **PARD**：`mask_token_id` 默认 `None`（从 draft config 读，通常 = `vocab_size`）；`sa_config` 默认 `None`。`tokens_per_gen_step = 2 * max_draft_len`。
- **DFlash**：`mask_token_id`=`None`、`target_layer_ids`=`None`（从 `dflash_config` 读）、**`attention_backend` 默认 `"VANILLA"`**（另一取值 `"TRTLLM"`，需 FlashInfer + Blackwell SM100/SM103）。`tokens_per_gen_step = max_draft_len + 1`。
- **DSpark**：`mask_token_id`/`target_layer_ids`/`block_size`/`markov_rank`/`markov_head_type`（`"vanilla"`/`"gated"`/`"rnn"`）默认全 `None`（从 checkpoint 的 `dspark_*` key 读）。`speculative_model` 必填。源码注释明说 `enable_confidence_head` / `confidence_threshold` **本版本故意未暴露**。

#### `MedusaDecodingConfig` / `LookaheadDecodingConfig`（PyTorch 下不可用，仅存类定义）
- Medusa：`medusa_choices`（`Optional[List[List[int]]]`，默认 `None`）、`num_medusa_heads`（默认 `None`，回落到 checkpoint `config.json` 的 `medusa_num_heads`）。
- Lookahead：默认取自 C++ 常量（`cpp/include/tensorrt_llm/executor/executor.h`）：
  ```cpp
  static constexpr SizeType32 kDefaultLookaheadDecodingWindow = 4;
  static constexpr SizeType32 kDefaultLookaheadDecodingNgram = 3;
  static constexpr SizeType32 kDefaultLookaheadDecodingVerificationSet = 4;
  ```

#### `trtllm-serve` / `trtllm-bench` 的 YAML

features 文档原文：
> Speculative decoding options must be specified via **`--config config.yaml`** for both `trtllm-bench` and `trtllm-serve`. All speculative decoding options can be specified in this YAML file. An additional `decoding_type` option is used to specify the type of speculation to use.

CLI flag 别名：
> **Non-breaking**: `--config <file.yaml>` is the preferred flag for passing a YAML configuration file. Existing workflows using `--extra_llm_api_options <file.yaml>` continue to work; it is an equivalent alias.

官方 YAML 示例（原文）：
```yaml
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 4
  speculative_model: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```
```yaml
# Dynamic tree mode
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 6
  speculative_model: /path/to/eagle3_model
  use_dynamic_tree: true
  dynamic_tree_max_topK: 10
  max_total_draft_tokens: 60
  max_batch_size: 4
```
```yaml
# SA combination
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 4
  speculative_model: /path/to/draft/model
  use_sa_spec: true
  sa_spec_threshold: 4
```

#### `examples/llm-api/quickstart_advanced.py` 的 CLI
```python
parser.add_argument('--spec_decode_algo', type=str, default=None)          # MTP/EAGLE3/DFLASH/DRAFT_TARGET/NGRAM/AUTO
parser.add_argument('--spec_decode_max_draft_len', type=int, default=1)
parser.add_argument('--draft_model_dir', type=str, default=None)
parser.add_argument('--max_matching_ngram_size', type=int, default=5)
parser.add_argument('--use_one_model', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--eagle_choices', type=str, default=None)
parser.add_argument('--use_dynamic_tree', default=False, action='store_true')
parser.add_argument('--dynamic_tree_max_topK', type=int, default=None)
```

#### 老 TRT engine 路线的 CLI（≤1.1.x，取自 v1.1.0rc5 tag 的 examples README，仅供考古）

**Medusa**：build `trtllm-build --speculative_decoding_mode medusa`；convert `--medusa_model_dir medusa-vicuna-7b-v1.3 --num_medusa_heads 4`；runtime `--medusa_choices="[[0], [0,0], [1], [0,1], ...]"`。限制原文：
> - TensorRT-LLM supports Medusa **only for Vicuna** (fine tuned LLaMA).
> - We match only tokens during the validation phase that is `medusa_temperature=0`.
> - **Beam search is not compatible with Medusa.**
> **Specifying paths-only instead of all choices is currently supported only in the Python runtime.**

**EAGLE**：build `--speculative_decoding_mode eagle`；convert `--eagle_model_dir EAGLE-Vicuna-7B-v1.3 --max_draft_len 63 --num_eagle_layers 4 --max_non_leaves_per_layer 10`；runtime `--eagle_choices=...`、`--eagle_posterior_threshold`、`--eagle_use_dynamic_tree`（默认 False）、`--eagle_dynamic_tree_max_top_k 10`。限制原文：
> - **EAGLE-2 is not supported.**
> - All EAGLE choices have to have exactly the same depth as `num_eagle_layers` of the engine.
> - **Pipeline parallelism is not supported.**
⚠️ 同一 README 后文又写 "EAGLE-2 can be enabled with 2 runtime flags (`--eagle_use_dynamic_tree` and `--eagle_dynamic_tree_max_top_k=N`)" —— **README 自相矛盾**，原样记录。

**ReDrafter**：build `--speculative_decoding_mode explicit_draft_tokens`；超参 `redrafter_num_beams`（**默认 5**）、`redrafter_draft_len_per_beam`（**默认 5**）、`redrafter_greedy_search`（**默认 True**）。
> **ReDrafter 内部用 beam search 生成草稿**："This method also allows the use of beam search to identify more prominent draft tokens"

**Draft-Target-Model** 限制原文：
> - `--use_paged_context_fmha=enable` **must be specified since we need KV-Cache reuse in this approach.**
> - `--kv_cache_enable_block_reuse` **must be specified** for this approach.
> - `--num_beams` can not be specified as larger than 1 since **beam search is not supported in this approach yet**.
> - **Streaming mode or batched-request mode are not supported in DTM yet.**
配置：`--draft_target_model_config="[4,[0],[1],False]"`（= `[max_draft_len, draft device list, target device list, use_logits]`）

**Lookahead**：build `--speculative_decoding_mode=lookahead_decoding --max_draft_len=... --max_beam_width=1`；runtime `--lookahead_config=[W,N,G]`（例 `[7,7,7]`）。`max_draft_len` 换算公式（原文代码）：
```python
def max_draft_len(windows_size, ngram_size, verification_set_size):
    return (0 if (ngram_size == 1) else ngram_size - 2) \
         + (windows_size - 1 + verification_set_size) * (ngram_size - 1)
```
原文注意点：`(1,1,0)` 强制退化为自回归；最小有意义配置 `(2,2,1)`；每请求需满足 `w<=W, n<=N, g<=G`。

### 3.3 草稿模型/草稿头从哪来

**官方文档指定的三个来源**（features 文档原文）：
> The following draft model checkpoints can be used for EAGLE 3:
> * Llama 3 variants: use the checkpoints from the authors of the original EAGLE 3 paper（https://huggingface.co/yuhuili ）.
> * Llama 4 Maverick: use the checkpoint from the NVIDIA HuggingFace repository（`nvidia/Llama-4-Maverick-17B-128E-Eagle3`）.
> * Other models, including `gpt-oss-120b` and `Qwen3`: check out the **Speculative Decoding Modules** collection from NVIDIA.

**NVIDIA 官方 HF 权重**（实测抓取 https://huggingface.co/collections/nvidia/speculative-decoding-modules ）：
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash`、`...-NVFP4-DSpark`、`nvidia/Kimi-K2.5-Thinking-Eagle3`、`nvidia/Kimi-K2-Thinking-Eagle3`、`nvidia/gpt-oss-120b-Eagle3-{short-context,long-context,throughput}`、`nvidia/Qwen3-235B-A22B-Eagle3`、`nvidia/Llama-4-Maverick-17B-128E-Eagle3`、`nvidia/Qwen3-235B-A22B-Thinking-2507-Eagle3`、`...-FP4-Eagle3`、`nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3`、**`nvidia/Llama-3.1-8B-Medusa-FP8`**、`nvidia/Llama-3.3-70B-Instruct-Eagle3`、`nvidia/Kimi-K2.6-Eagle3`、`nvidia/MiniMax-M3-DSpark`、`nvidia/DeepSeek-V4-Pro-nvfp4-DSpark`、`nvidia/DeepSeek-V4-Flash-nvfp4-DSpark`。另有 blog11 引用的 `nvidia/gpt-oss-120b-Eagle3`。

> ⚠️ 注意 `nvidia/Llama-3.1-8B-Medusa-FP8` 仍在，但 **TensorRT-LLM 1.2+ 已经跑不了 Medusa**——该权重现在只对 ModelOpt / 其他框架有意义。**"官方还在发权重"不等于"引擎还能跑"，这是很好的一个考据教训。**
> 各仓库的 license 与逐仓库 base model 对应关系：**未查证**。

**需要单独训练吗——需要，NVIDIA 提供 ModelOpt 脚手架**
（`NVIDIA/TensorRT-Model-Optimizer` 的 `examples/speculative_decoding/README.md`）

- 一行端到端：`bash train_eagle3_and_export.sh --base_model meta-llama/Llama-3.2-1B-Instruct`（初始化 draft → 微调 → **在 MT-Bench 上评估 acceptance rate** → 导出可部署 checkpoint）
- 灵活训练：`./launch_train.sh --config ../../modelopt_recipes/general/speculative_decoding/eagle3.yaml model.model_name_or_path=... data.data_path=... training.output_dir=...`
- 数据集：默认 **Daring-Anteater**（`nvidia/Daring-Anteater`），另支持 MTBench / ShareGPT / Magpie(1M,500k,300k) / Nemotron Post-Training Dataset V2（~3.3M 行生成 / ~1.9M 行训练）与 V3（~3.4M / ~3.9M）
- **三种训练模式**：
  - **Online**（base+draft 同驻显存，仅适合小模型）
  - **Offline**（先把 hidden states dump 到磁盘）——原文：**"requires several to tens of terabytes of disk storage depending on dataset size"**
  - **Streaming**（从活的 `vllm serve` 经 **NIXL RDMA** 流式取 hidden states，支持多节点）
- 数据合成（原文）：**"To achieve higher acceptance rates during speculative decoding, it is beneficial to use conversations generated by the base model as training data."**（`scripts/server_generate.py` + `vllm serve`）
- 草稿词表压缩：`scripts/calibrate_draft_vocab.py --draft_vocab_size 32000` → 产出 `d2t.pt`，推理时 `target_token = draft_token + d2t[draft_token]`
- 导出：`python scripts/export_hf_checkpoint.py --model_path $OUTPUT_DIR --export_path $EXPORT_PATH`
- DFlash 训练也已进 ModelOpt：`dflash.dflash_block_size`（默认 **8**）、`dflash_num_anchors`（**512**）、`dflash_loss_decay_factor`（**4.0**）、`dflash_self_logit_distillation`（**true**）、`dflash_architecture_config.num_hidden_layers`（**5**）

**⚠️ ModelOpt 的 Support Matrix（原文）与推理侧已经脱节**：

| Model | Medusa | EAGLE1/2 | EAGLE3 |
|---|---|---|---|
| LLAMA 2 / 3 / 3.1 | ✅ | ✅ | ✅ |
| Mistral / Phi 3 / QWen 1.5,2,2.5,3 | ✅ | ✅ | ✅ |
| Kimi-K2.5, K2.6 | | | ✅ |

> **ModelOpt 仍然能训 Medusa / EAGLE1/2，但 TRT-LLM 1.2+ 已经推不了它们**——训练侧与推理侧的支持面已经脱节。这条也很值得写进 `15-谱系图与被淘汰的分支`。

ModelOpt 给出的 TRT-LLM 部署 YAML（原文，注意它关了三样东西）：
```yaml
enable_attention_dp: false
disable_overlap_scheduler: true
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: <draft_model_checkpoint>
kv_cache_config:
    enable_block_reuse: false
```

**TRT-LLM 内置的 hidden-states 采集通路**：`SaveHiddenStatesDecodingConfig`（`decoding_type: "SaveState"`）——`output_directory`（必填）、`write_interval` 默认 **20**、`file_prefix` 默认 `"data"`、`eagle3_layers_to_capture` 默认 `None`（默认捕获 3 个中间层 + post-norm，`num_capture_layers`=**4**）。
> **三家引擎都做了同一件事**：vLLM 的 `method="extract_hidden_states"`、TRT-LLM 的 `decoding_type="SaveState"`、SpecForge 的 offline hidden-state dump、ModelOpt 的 `compute_hidden_states_trtllm.py`。**"用推理引擎给草稿模型造训练数据"已经成为标准工序**——这是 `21-草稿模型怎么训` 的骨架。

### 3.4 与 IFB / chunked prefill / KV cache reuse / CUDA graph / disagg / overlap / PP 的相互作用

#### 官方 Feature Combination Matrix（`docs/source/features/feature-combination-matrix.md`，main）

投机解码被拆成三行：**Speculative Decoding — Linear** / **— Dynamic Trees** / **— Legacy Path (NGram, user-provided)**。三行取值**完全一致**：

| 对手特性 | 取值 |
|---|---|
| Overlap Scheduler | **Yes** |
| CUDA Graph | **Yes** |
| Tensor Parallelism | **Yes** |
| **Pipeline Parallelism** | **No** ← |
| Expert Parallelism | Yes |
| **Helix Parallelism** | **No** ← |
| Attention Data Parallelism | Yes |
| **Disaggregated Serving** | **Yes** |
| **Chunked Prefill** | **Yes** |
| Torch Sampler | Yes（Dynamic Trees 行标注 "Yes (MTP dynamic tree: **greedy only**)"） |
| **TLLM C++ Sampler** | **No** ← 必须用 TorchSampler |
| **KV Cache Reuse** | **Yes** |
| Sliding Window Attention | Yes |
| **Logits Post Processor** | **No** ← |
| Guided Decoding | **Yes** |
| LoRA | **Untested** |
| 三种 spec dec 互相之间 | **No**（不能叠加） |

> **回答原问题**：**官方文档没有说 speculative decoding 与 chunked context 不兼容，也没有说与 KV cache reuse 不兼容——恰恰相反，矩阵里两项都是 Yes。**
> KV cache 文档（`docs/source/features/kvcache.md`）更有一句正面陈述：
> > ### Speculative Decoding
> > **Reuse across requests is supported by all speculative decoding models.**

#### 矩阵的历史演进（很能说明"兼容性是逐年补上来的"）

| 组合 | v1.0.0 | v1.2.0 / v1.2.1 | main (1.3.0rc25) |
|---|---|---|---|
| EAGLE-3 **Two Model** × Overlap Scheduler | **No** | Yes | 两模型路径已废弃 |
| MTP × Guided Decoding | **No** | Yes | Yes |
| EAGLE-3 One Model × Guided Decoding | **No** | Yes | Yes |
| MTP × Sliding Window Attention | **No** | Yes | Yes |
| MTP / EAGLE3 × Chunked Prefill | Yes | Yes | Yes |
| MTP / EAGLE3 × KV Cache Reuse | Yes | Yes | Yes |
| Spec dec × Pipeline Parallelism | （v1.0 无此列） | **No** | **No** |
| 矩阵行的命名 | 按方法（MTP / EAGLE-3 One / Two） | 同左 | **改为按拓扑（Linear / Dynamic Trees / Legacy）** |

> 最后一行的命名变化本身是个信号：**NVIDIA 已经不按"哪个方法"组织兼容性，而按"草稿的拓扑形状（链 / 树 / 检索）"组织**——因为方法在爆炸，但拓扑只有三种。**这个抽象值得本库 `16-树形草稿与树注意力` 借用。**

#### 代码里真实生效的门禁（比矩阵更细，且**有与矩阵冲突处**）

**(a) Overlap scheduler 会被自动关掉**（`py_executor_creator.py`）：
```python
if not llm_args.disable_overlap_scheduler and spec_config is not None:
    if not spec_config.spec_dec_mode.support_overlap_scheduler():
        logger.warning(f"Disable overlap scheduler for speculation mode {spec_config.spec_dec_mode.name}")
        llm_args.disable_overlap_scheduler = True
```
而 `interface.py`：
```python
def support_overlap_scheduler(self):
    return self.is_mtp_one_model() or self.is_eagle3_one_model() \
        or self.is_sa() or self.has_draft_model() or self.is_external_drafter()
```
`NGRAM` 与 `USER_PROVIDED` **都不在这个集合里** → **NGram / user-provided 会被静默关掉 overlap scheduler**。
⚠️ **这与矩阵里 "Legacy Path (NGram, user-provided) × Overlap Scheduler = Yes" 直接矛盾。** features 文档自己的 NGram 示例代码也是佐证：
```python
llm = LLM("/path/to/target_model", speculative_config=speculative_config, disable_overlap_scheduler=True)
```

**(b) Pipeline parallel**：矩阵 No；老 TRT EAGLE README 也写 "Pipeline parallelism is not supported."。代码侧未找到统一的 PP 拒绝断言（**未查证**具体是硬报错还是仅未验证）。

**(c) FlashInfer attention backend**：
> `FLASHINFER attention backend supports one-engine speculative decoding only when the draft model shares the target KV cache.`

**(d) Attention DP 与独立 draft KV cache**（`_util.py`）：`"Attention DP is enabled, separate draft KV cache is not supported."`

**(e) KV Cache Manager V2 与两模型投机**（`kvcache.md` 原文）：
> **Two-model speculative decoding (for example Eagle3 with `eagle3_one_model=False`) is not supported by V2**: the draft model runs in a separate engine with its own KV cache manager, and V2 sizes both managers from the full `max_gpu_total_bytes` budget instead of partitioning it between them. Under `auto`, a model default of V2 falls back to V1 for that combination; setting `use_kv_cache_manager_v2: true` explicitly raises an error.

**(f) KV cache 压缩**（`_util.py`）：`"KV-cache compression does not support speculative decoding ..."`，`KvCacheCompressionConfig.supports_speculative_decoding()` 默认返回 `False`。

**(g) Disaggregated serving**（`py_executor_creator.py`）：
```python
# WAR for https://nvbugs/5807902
# Disable separate draft KV cache in disaggregated mode
if cache_transceiver_config is not None:
    spec_config._allow_separate_draft_kv_cache = False
```

**(h) Guided decoding 三分支**：
```python
if spec_config is None or spec_config.spec_dec_mode.support_guided_decoder():
    guided_decoder = GuidedDecoder(**kwargs)      # 非投机 + 两模型投机
elif spec_config.spec_dec_mode.support_capturable_guided_decoder():
    ... CapturableGuidedDecoder(**kwargs)          # one-model 投机
else:
    raise ValueError(f"Guided decoding is not supported for speculative decoding mode: {...}")
```

**(i) torch.compile / piecewise CUDA graph**（`torch_compile_and_piecewise_cuda_graph.md`）：
> Torch compile cannot work with multi-ModelEngine config.
> 1. Speculative Decoding in Two-Model Style
> ```yaml
> speculative_config:
>   decoding_type: "MTP"
>   mtp_eagle_one_model: False # Not supported
> speculative_config:
>   decoding_type: "Eagle3"
>   eagle3_one_model: False # Not supported
> ```

**(j) `draft_len_schedule` 会强制打开 CUDA graph padding**（`llm_args.py` 注释）：
> If `speculative_config.draft_len_schedule` is provided, `cuda_graph_config.enable_padding` is automatically set to True.

**(k) Disagg 上下文服务器的 overlap**（`disagg-serving.md` 示例注释）：
> Overlap scheduler for context servers are disabled because it's not supported for disaggregated context servers yet

#### 官方实测过的 disagg + EAGLE3 配置模板

`tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_gptoss_eagle_trtllm.yaml`——注意 **ctx 和 gen 两侧都挂同一份 `speculative_config`**，`enable_chunked_prefill: true` 与投机共存，`enable_block_reuse: false`：
```yaml
context_servers:
  enable_chunked_prefill: true
  disable_overlap_scheduler: true
  speculative_config: &eagle_spec
    decoding_type: Eagle
    max_draft_len: 3
    eagle3_one_model: true
    speculative_model: gpt_oss/gpt-oss-120b-Eagle3
  kv_cache_config:
    enable_block_reuse: false
generation_servers:
  speculative_config: *eagle_spec
```
Release Notes 1.1 记 "Enabled guided decoding to work in conjunction with speculative decoding (including 2-model and **draft model chunked prefill**)"；1.0 记 "Support chunked prefill on spec decode 2 model"。

> **PD 分离 × 投机的官方素材至此有两家**：SGLang（`pd_disaggregation.mdx`，两侧都配 spec，都关 radix cache）与 TensorRT-LLM（上面这份测试配置，两侧都配 spec，关 block reuse）。**两家给出的做法高度一致：prefill 侧与 decode 侧各配一份投机配置，且都关掉前缀复用。** 这是 `23-与其它优化的相互作用` 的核心结论。

#### ⭐ 一个值得写进正文的观察：官方 EAGLE3 recipe **全部**主动关掉 KV cache reuse

尽管矩阵说 KV cache reuse 与投机兼容，**NVIDIA 自己的每一份 EAGLE3 部署配方都写 `enable_block_reuse: false`**：
- blog06（Llama4 Maverick + Eagle3）、blog11（GPT-OSS-120B + Eagle3）
- ModelOpt 部署 YAML
- disagg 测试配置
- `examples/configs/curated/deepseek-v4-pro-{latency,throughput}.yaml`、`qwen3.8-*-mtp3.yaml`

→ **「兼容」≠「官方推荐一起开」**。正文里必须区分这两件事。

### 3.5 官方承认的限制与负收益（TensorRT-LLM 部分）

#### (a) features 文档开篇——总纲

> Speculative decoding is a technique for accelerating LLM inference **at low batch sizes**.
> For all speculation algorithms, when speculation is enabled, a single sequence of draft tokens with length `max_draft_len` is created for every request. **There is currently no way to dynamically disable speculation, thus speed ups are only observable at low batch sizes.**

（此句已回读文档站页面二次核验，一字不差。）
⚠️ 这句与 `max_concurrency` / `draft_len_schedule` / `SpeculationGate` 三个机制**存在张力**——那三个是"按 batch size / 按接受率把 draft_len 降到 0"的粗粒度门控，不是 per-request 的动态关闭。**文档落后于代码，两者都要引并说明差异。**

#### (b) NGram 技术博客（blog07）——最直白的负收益机制说明

> Run the target model with **`verification_batch = original_batch × (v+1)`**; ... **The iteration latency grows as the verification batch becomes larger than the original batch. As we increase `max_draft_len (v)`, the overhead grows even more. Therefore, speculative decoding tends to work best with small batch sizes and low concurrency.**
> We can see that N-Gram can provide speed-ups **for batch sizes up to 32** and works best with a single batch. **The main overhead with larger batch sizes is the verification cost.** With batch size being 1 and 4, `k = 3, v = 5` is the best N-Gram configuration. **With batch size = 32, `k = 5, v = 3` is the best configuration since the verification batch size is smaller and the overhead is less.**
> This policy adds **less than 15% overhead** to iteration latency yet offers nets double-digit end-to-end speed-ups.

> **`verification_batch = original_batch × (v+1)` 与 vLLM 的 `effective BS = BS*K` 是同一个公式的两种写法**（差 1 是因为 TRT-LLM 把 bonus token 也算进去）。**两家独立给出同一个模型——这条可以当作本库 `18-batch与吞吐` 的核心公式，且有双引擎背书。**

#### (c) Llama4 Maverick + EAGLE3 部署指南（blog06）——高并发下每用户 TPS 暴跌

> **Note:** This configuration is optimized for minimum latency (`enable_min_latency: true`). **When increasing the concurrency of requests, the tokens per second (TPS) per user degrades rapidly.** This setup is designed to maximize single-user performance rather than high-concurrency throughput.
> (Note that your specific performance numbers may vary—**speculative decoding speedups depend upon the dataset!**)

#### (d) legacy 文档的前提假设（讲清了什么时候会亏）

> This can lead to a reduction in the average per-token latency **in situations where the GPU is underutilized due to small batch sizes.**
> The underlying assumptions are twofold:
> 1. processing multiple draft tokens concurrently will be as rapid as processing a single token
> 2. multiple draft tokens will be validated successfully over the course of the full generation
> **If the first assumption holds true**, the latency of speculative decoding will no worse than the standard approach.
> It's important to note that **the effectiveness of speculative decoding techniques is highly dependent on the specific task at hand.** For instance, forecasting subsequent tokens in a code-completion scenario may prove simpler than generating a summary for an article.
> It is important to make sure that the draft and target models were trained with the same tokenizer, **else the acceptance rate is extremely low and performance is regressed.**

> **这段"两条前提假设"的表述是我在所有引擎文档里见过最清楚的一段负收益理论**——它把加速比拆成了"验证是否真的免费"与"草稿是否真的被接受"两个可分别证伪的命题。**直接对应本库 `03-并行验证为什么几乎免费` 与 `05-接受率alpha` 两篇。**

#### (e) Release Notes 里承认的 Known Issue

1.0：
> **Enabling disaggregated serving, MTP, and the overlap scheduler at the same time can lead to accuracy problems.**

1.2：
> **Disaggregated Serving:** A hang may occur in disaggregated serving with context pipeline parallelism and generation tensor parallelism configurations.

#### (f) 模型级"不支持"声明

- GLM-5 部署指南：**"MTP is not currently supported with the NVFP4 checkpoint."**
- MiniMax-M3 部署指南：**"MTP is not supported on the sparse-attention path in this release."**、"The block-sparse attention path does not currently support KV cache reuse or Multi-Token Prediction (MTP) in this release."、"KV cache reuse must be disabled."
- Kimi K3 部署指南：**"Speculative decoding and disaggregated serving are not yet available for Kimi K3; support is under development."**

#### (g) ⭐⭐ 官方自己的「按并发调 draft_len」实测表——本节最硬的证据

`examples/configs/database/` 下是 NVIDIA 逐并发 audit 过的配置文件。**这就是官方在"什么并发下该把投机调小/关掉"上的实际答案：**

**DeepSeek-R1-0528 / B200 / TP8 / ISL-OSL 1k-1k**

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MTP `max_draft_len` | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | **1** | **1** | **1** | **1** |

**DeepSeek-R1-0528 / H200 / TP8** —— 拐点随 ISL/OSL 大幅移动：

| ISL-OSL | 拐点（draft_len 3→1） |
|---|---|
| 1k-1k | concurrency **128** |
| 1k-8k | concurrency **512** |
| **8k-1k** | concurrency **64** |

**nvidia/DeepSeek-R1-0528-FP4-v2 / B200 / TP8**

| ISL-OSL | 拐点 |
|---|---|
| 1k-1k | 128 之后降到 1（**conc 256 与 conc 2048 甚至完全没有 `speculative_config`**） |
| 1k-8k | concurrency **64** |
| 8k-1k | concurrency **64** |

**openai/gpt-oss-120b / B200 与 H200 / 全部 TP 与全部并发（含 concurrency=1）：整套 audit 配置里没有任何一个开了投机解码。**
→ 这与 blog11「用 Eagle3 跑 gpt-oss-120b」形成鲜明对照——**NVIDIA 自己的性能基线配置并不用它。**

curated 配置的一致规律：
- `deepseek-r1-latency.yaml`：`max_batch_size: 4`，MTP `max_draft_len: 3` + relaxed acceptance（topk 10 / delta 0.6）
- `deepseek-r1-throughput.yaml`：`max_batch_size: 1024`，MTP **`max_draft_len: 1`**
- `deepseek-v4-pro-latency.yaml`：MTP `max_draft_len: 3`；`-throughput.yaml`：MTP **`max_draft_len: 1`**
- `qwen3.8-low-latency-mtp3.yaml` / `qwen3.8-high-throughput-mtp3.yaml`：都用 3，但吞吐档 `max_batch_size` 只有 36

> **一句话概括官方立场：低并发用 draft_len=3，高并发降到 1，极端高并发直接不配。**
> ⭐ **更值得注意的是 ISL/OSL 对拐点的影响：同样 H200+TP8+DeepSeek-R1，长输出（1k-8k）的拐点是 512，短输出长输入（8k-1k）的拐点只有 64——差 8 倍。** 这正好印证用户任务里点名的"长输入短输出"场景是投机的重灾区。**这组数据是本库 `19-负收益全解` 里"长输入短输出"那一节唯一的官方量化证据。**

### 3.6 三层「自动降级 / 关闭」机制

| 机制 | 字段 | 语义 |
|---|---|---|
| 并发阈值 | `speculative_config.max_concurrency` | ">0 时，batch size 超过该值即关闭投机"。内部翻译成 `draft_len_schedule = {max_concurrency: max_draft_len}` |
| 按 batch 分档降 draft_len | `speculative_config.draft_len_schedule` | `{4:4, 8:2, 32:1}` → BS 1-4 用 4，5-8 用 2，9-32 用 1，**33+ 隐式 0（关闭）**。最小 key 必须映射到 `max_draft_len`；所有值 ≤ `max_draft_len` |
| 按接受率永久熔断 | `acceptance_rate_window_size` + `acceptance_rate_threshold` | 见下 |

`support_dynamic_draft_len()` 只对以下模式返回 True：MTP one-model、EAGLE3 one-model、MTP-EAGLE one-model、PARD、DFlash、DraftTarget one-model、SA。**NGram 不在其中。**

**`SpeculationGate`（接受率熔断器）** —— `tensorrt_llm/_torch/speculative/speculation_gate.py` 类 docstring 原文：
> Tracks a rolling average of true acceptance-rate samples over the last N speculation-enabled decoding iterations. **Permanently disables speculation when the rolling average falls below the configured threshold.**

日志：`[SpeculationGate] Speculative decoding disabled: rolling acceptance rate avg {x:.3f} < threshold {t} over last {N} iterations`。
⚠️ **注意是 permanently——一旦熔断不会再打开。**

**`AUTO` 启发式的真实内容**（`auto_heuristic.py` 全文核心）：
```python
def suggest_spec_config(max_batch_size: int) -> "DecodingBaseConfig":
    """
    Suggests a reasonable draft model free speculation scheme.
    Used when the user specifies spec_mode == AUTO.
    For now, we always use an ngram scheme that gets disabled at BS>=32.
    """
    return NGramDecodingConfig(
        max_draft_len=5 if max_batch_size <= 4 else 3,
        max_matching_ngram_size=3 if max_batch_size <= 4 else 5,
        max_concurrency=32,
        is_keep_all=True, is_use_oldest=True, is_public_pool=True,
    )
```
Release Notes 1.0 对应条目："**Auto-enable ngram with concurrency <= 32**"、"Support turning on/off spec decoding dynamically"。

> **⭐ 这段代码是本节最有价值的一处**：当用户说"你帮我自动选"时，NVIDIA 的答案是——**用 n-gram，并在 BS≥32 时关掉。** 这是官方对"默认该怎么做"最直接的表态。
> 三家对照：vLLM 的 Dynamic SD 示例把 K 在 BS≥129 降到 0；SGLang 的 adaptive 默认在 BS≥32 锁 K=1；**TensorRT-LLM 的 AUTO 在 BS≥32 直接关掉。三家的临界点都落在 32~128 这个区间。**

### 3.7 采样兼容性

#### (a) 官方最硬的一句「只支持 greedy」

blog11（GPT-OSS-120B + Eagle3）原文：
> Note: **This Eagle3 + TensorRT LLM endpoint currently supports only greedy sampling. The following Chat Completions parameters are ignored (no-ops): `temperature`, `top_p`, `top_k`, and `seed`.**

⚠️ 这是**该篇部署配方**（gpt-oss-120b + Eagle3 + `sampler_type: TorchSampler`）的声明，不是全局声明。1.3 代码已把非 greedy 做成 per-request 自动检测（见下），所以这句可能已过时——**但它是目前能找到的、NVIDIA 白纸黑字最强的"spec dec 只支持 greedy"表述**，写正文必须引，并注明它是配方级而非全局。

#### (b) 1.3 代码里的实际状态：非 greedy 已支持，但有一串条件

`allow_advanced_sampling` 的 description（原文）：
> DEPRECATED: no-op kept for backward compatibility. Will be removed in a future release. **Non-greedy sampling is now auto-detected per request; this flag no longer has any effect.**

`use_rejection_sampling` 的 description（原文）：
> If true, enables rejection sampling for one-model speculative decoding paths when the batch contains **any non-greedy request**. **All-greedy batches always take the argmax fast path** regardless of this flag. Set to false (default) to use **exact-match verification** on non-greedy batches. The non-dynamic-tree one-model path **requires FlashInfer**.

> ⭐⭐ **这是本次调研里对本库铁律一最重要的一条发现**：
> **TensorRT-LLM 的默认（`use_rejection_sampling=False`）在非贪心 batch 上用的是 "exact-match verification"，也就是"相等即接受"，而不是 Leviathan 的拒绝采样。**
> 至此四家全部对上了：
> | 引擎 | 默认验证方式 | 经典 Leviathan Alg.1 |
> |---|---|---|
> | vLLM | `rejection_sample_method="standard"` + `draft_sample_method="greedy"`（草稿概率当 one-hot） | 需设 `draft_sample_method="probabilistic"` |
> | SGLang | `tree_speculative_sampling_target_only`（`draft_probs` 清零） | 需 `--speculative-use-rejection-sampling` |
> | TensorRT-LLM | exact-match verification | 需 `use_rejection_sampling=True` |
> | llama.cpp | 相等即接受（`common_sampler_sample_and_accept_n`） | 无此选项 |
> | HF transformers | `assistant_model` + `do_sample=True` 时**默认就是 Alg.1** | 其余路径是相等即接受 |
> **结论：五家里只有 HF transformers 在标准配置下默认跑教科书上的拒绝采样；四个生产引擎的默认路径都是"相等即接受"的保守变体。** 这是本库能给出的最反直觉、也最有价值的一条工程结论，`04` / `07` / `27` 三篇都该写。

rejection sampling 的白名单/黑名单（`validate_speculative_config`）：
- **支持**：Eagle3 one-model、vanilla MTP、MTP-Eagle one-model、PARD、DFlash、DSpark、DraftTarget one-model
- **不支持**：**NGram**（"retrieval-based drafters that emit only token ids with **no q** are excluded"）、**SA**（`sa_config` 存在即拒）
- 额外拒绝条件：`use_relaxed_acceptance_for_thinking`、**context parallelism**（`context_parallel_size > 1`，"the draft path resolves only the global argmax, not full distributions"）、**guided decoding**、`dynamic_tree_max_topK > 32`（"exceeds the dynamic-tree rejection kernel limit of **32 tried siblings per level**"）
- 显式设 `True` 撞到上述条件会 **raise**；默认继承值则静默关掉

> **"NGram 因为没有 q 所以不能做拒绝采样"这句话本身就是一条极好的教学点**：无模型草稿（n-gram / prompt lookup / suffix）**在数学上根本无法参与 Leviathan 的 $\min(1,p/q)$ 判据**，因为它压根没有草稿分布 $q$。**它们只能"相等即接受"。** 这解释了为什么 `11-无模型草稿` 那一篇必须单独讲无损性口径。

#### (c) `docs/source/features/sampling.md` 里逐条的限制（原文）

> * Top-P decay is **not supported** in combination with beam search or with speculative decoding modes that route draft tokens through the Torch Sampler; **such requests are rejected**.
> * **Positive Min-P is not supported in combination with one-model speculative decoding. Such requests are rejected at admission.**
> * With one-model speculative decoding the penalties must be enabled at deploy time with `enable_penalty: true`..., because they need an occurrence workspace that is allocated up front. **While the flag is off, a request that sets any of the three is rejected at admission rather than decoded without them. Tree speculation (`eagle_choices` or a dynamic tree) is not supported and such requests are rejected even when the flag is on.** Only the target distribution is penalized; the draft model proposes from its unpenalized distribution, which leaves the sampled result unchanged but **can lower the acceptance rate as the penalty grows**.
> * This optimization does not compute the complete set of token sampling probabilities (after top-k / top-p masking etc.), which typically can be omitted **unless requested by the user or required for speculative decoding (rejection sampling)**.

> 最后一条"penalty 只施加于 target，草稿从未加惩罚的分布提议 → **结果不变但接受率会随惩罚增大而下降**"是极精确的表述，**和 SGLang 的 "Adapters apply to the target only; a shared draft runs unadapted"（LoRA）是同一类现象**：**任何只作用于 target 而不作用于 draft 的修改，都会以接受率为代价。** 这是一条可以推广的判据，值得写进 `21-草稿模型怎么训` 或 `27-常见误解与判据`。

Advanced sampling mode 表（原文）：

| Mode | `top_k` kernel | `top_p` kernel |
|---|---|---|
| `full`（默认） | applied | applied |
| `no_topk` | **skipped** | applied |
| `no_topp` | applied | **skipped** |
| `no_topk_no_topp` | **skipped** | **skipped** |

（`sampling.md` 顶部特性表的 "Forward Pass" 列只列了 No drafting / Draft target model / Eagle 3 / Ngram，未列 MTP/PARD/DFlash/SA，疑似陈旧，不宜当排他性证据。）

#### (d) Beam search
- **老 TRT 路线**：Medusa "Beam search is **not** compatible with Medusa."；Draft-Target "`--num_beams` can not be specified as larger than 1"；NGram V1 同；Lookahead build 强制 `--max_beam_width=1`。**ReDrafter 反而内部用 beam search 生成草稿。**
- **PyTorch 1.3**：`sampling.md` 的 Beam search 段列出的拒绝组合是 disaggregated serving / 递减 `beam_width_array` / `best_of != max_beam_width`，**没有列 speculative decoding**。且 `TorchLlmArgs` 新增了 `enable_speculative_beam_history_d2h`（prototype，默认 `False`），说明 beam search + 投机在 1.3 已至少部分打通。
  → **「PyTorch backend 下 beam search 与 spec dec 是否全面可用」官方文档未给出明确陈述，标未查证。**

#### (e) logprobs / n>1
- **logprobs**：features/sampling 文档均未出现"spec dec 下 logprobs 受限"的陈述。Release Notes 1.1 记 "Added support for `prompt_logprobs` in the PyTorch backend"、"Standardized `topk` logprob returns across TRT and PyTorch backends"。**投机开启时 logprobs 的精确语义：未查证。**
- **n>1 / best_of**：`sampling.md` 特性表把 "Best of / n (composable)" 与 "Rejection sampling (composable)" 并列。`sampling_params.py` 对 `best_of > 1` 的限制是**贪心解码**相关（需 `TLLM_ALLOW_N_GREEDY_DECODING=1`），与投机无关。**投机开启时 n>1 的行为：未查证。**
- **Logits Post Processor × spec dec = No**（矩阵）；**Guided Decoding × spec dec = Yes**（矩阵 + blog12）。

### 3.8 GitHub issue 里反复出现的坑

#### (a)「greedy 下结果不一致 / 精度掉」——最高频主题

| # | 标题 | 状态 | 日期 | 要点 |
|---|---|---|---|---|
| [#10309](https://github.com/NVIDIA/TensorRT-LLM/issues/10309) | Qwen3 + Eagle3 generates different result with greedy decoding compared to "no eagle version" | **Open** | 2025-12-26 | Qwen3-4B + EAGLE3-Qwen3-4B-DenseHead，`max_draft_len: 4`、`eagle3_one_model: true`，TRT-LLM 1.2.0rc6。报告者原话 "**Eagle3 should NOT change the output when we do greedy decoding.**" 输出开头一致、中途发散；怀疑 "eagle specific parallel verification kernels" |
| [#5791](https://github.com/NVIDIA/TensorRT-LLM/issues/5791) | [Eagle3] Accuracy Issue | — | — | 早期 EAGLE3 精度问题 |
| [#14825](https://github.com/NVIDIA/TensorRT-LLM/issues/14825) | kimi 2-6 nvfp4 + SpecD gives low accuracy on GPQA Diamond | Closed | 2026-08-21 | 投机 + NVFP4 下评测掉分 |
| [#15182](https://github.com/NVIDIA/TensorRT-LLM/issues/15182) | Rejection sampling in Kimi K2.6 | Closed | 2026-06-17 | |

> **#10309 + #14825 与 llama.cpp #25618（量化 target 下贪心发散）是同一族群**：**低比特量化 + 投机解码，贪心等价性容易破。** 这是 §7b 那条跨引擎发现的第四个例证，且 TRT-LLM 和 llama.cpp 都指向"量化"这个共同因素。

#### (b) 接受长度随 batch size 下降 —— 直接对应 §7 的负收益

| # | 标题 | 状态 | 日期 |
|---|---|---|---|
| [#9208](https://github.com/NVIDIA/TensorRT-LLM/issues/9208) | **[Bug]: Eagle3 acc len decreases with bigger batch size** | **Closed as not planned** | 2025-11-17 |

实测数字（Qwen3-235B）：**BS 8 → AL 2.16；BS 16 → AL 2.14；BS 64 → AL 1.93**。另一数据集：**BS 1 → 1.96 vs BS 32 → 1.55**。报告者期望 "Same acc len (at least avg) for each batch size"。
**NVIDIA 未在 issue 中给出根因解释，直接 closed as not planned**（assignee: mikeiovine）。

> ⭐ **这条极有价值**：它说明**接受长度本身就随 batch 下降**，不只是"验证变贵"。也就是说负收益是双重的——**分子（接受长度）在跌，分母（验证成本）在涨。**
> 但要注意：**"接受长度随 batch 下降"在数学上并不显然**（每个请求的草稿质量理论上与 batch 无关）。可能的解释是 batch 大时长请求占比高、上下文更长、或调度导致的 chunk 边界效应。**NVIDIA 把它 closed as not planned 而没解释，本库应当把这当作一个「开放问题」记录下来，而不是当成已知规律。** 建议在 `18-batch与吞吐` 里明确写"此现象有报告但机理未获官方解释"。

| # | 标题 | 状态 | 日期 |
|---|---|---|---|
| [#16767](https://github.com/NVIDIA/TensorRT-LLM/issues/16767) | [Bug] DSpark speculative decoding: **accept length collapses to ~1 at generation batch size > 1 in disaggregated serving** | **Open** | 2026-07-23 |

#### (c) 结构化输出 / tool calling 被投机静默破坏

| # | 标题 | 状态 | 日期 |
|---|---|---|---|
| [#16377](https://github.com/NVIDIA/TensorRT-LLM/issues/16377) | **[Bug]: Eagle3 speculative decoding silently drops tool_calls for gpt-oss on trtllm-serve** | **Open** | 2026-07-14 |
| [#8713](https://github.com/NVIDIA/TensorRT-LLM/issues/8713) | [Bug]: Eagle3 + Harmony Format Bug Report | — | — |
| [#17437](https://github.com/NVIDIA/TensorRT-LLM/issues/17437) | Interior control-token rejection can split a tool-call prelude and silently drop tool_calls | **Open** | 2026-08-08 |
| [#16967](https://github.com/NVIDIA/TensorRT-LLM/issues/16967) | [Bug]: Structured Output with DSpark Speculative Drafter | Closed | 2026-08-12 |
| [#15022](https://github.com/NVIDIA/TensorRT-LLM/issues/15022) | Guided decoding (xgrammar) + EAGLE-3 + `draft_len_schedule` reaching 0 crashes during CUDA graph capture | Closed | 2026-06-15 |

**#16377 的关键点**：返回 HTTP 200 + `finish_reason: "stop"` + **空 `tool_calls`**，但 `completion_tokens: 37` 且 `reasoning_content` 显示模型确实想调工具——"**The tool-call channel is lost specifically when Eagle3 is on**"；**关掉投机、其他不变，tool calling 就恢复正常**。报告者指出这条路径正是官方 DGX Spark playbook 推荐的配置。

> **这与 HF #47912/#48039（块内接受越过 EOS / stop_strings）是同一类 bug**：**一次接受多个 token 时，"边界"语义（EOS、stop string、control token、tool-call 分隔符）容易被跨过去。**
> 三家都有这类 bug（HF、TRT-LLM、SGLang #31978 grammar vocab mask），**说明这是投机解码的一个结构性风险，不是某家的实现疏忽。** 值得在 `19-负收益全解` 或 `27-常见误解与判据` 里单列成一类：**「投机解码的正确性风险不止在概率分布上，还在于所有『按 token 边界触发』的逻辑」。**

#### (d) 崩溃 / 内存 / CUDA graph

| # | 标题 | 状态 | 触发组合 |
|---|---|---|---|
| [#7310](https://github.com/NVIDIA/TensorRT-LLM/issues/7310) | segfault when using speculative decoding on 1.1.0rc1 | **Closed as not planned** | `decoding_type: AUTO` + **chunked prefill** + CUDA graph padding + B200，**并发 >10 才复现，关掉投机就不出**。先报 `The size of tensor a (56) must match the size of tensor b (77)` 再 segfault in KV-cache management |
| [#13909](https://github.com/NVIDIA/TensorRT-LLM/issues/13909) | Excessive memory allocation during CUDA graph capture in Eagle3 flow | Closed | |
| [#16826](https://github.com/NVIDIA/TensorRT-LLM/issues/16826) | SA speculative decoding crashes the executor event loop when a prompt is near `max_seq_len` | Closed | |
| [#16448](https://github.com/NVIDIA/TensorRT-LLM/issues/16448) | KV cache connector hangs with Eagle speculative decoding **when rewind crosses block boundary** | **Open** | |
| [#17095](https://github.com/NVIDIA/TensorRT-LLM/issues/17095) | DSpark: disaggregated gen-only benchmark **deadlocks** under attention-DP with asymmetric spec dec | **Open** | |
| [#17024](https://github.com/NVIDIA/TensorRT-LLM/issues/17024) | DeepSeek-V4 + MTP: `AttributeError: '_num_tables'` in `DeepseekV4CacheManager.copy_batch_block_offsets` during warmup | **Open** | |
| [#12257](https://github.com/NVIDIA/TensorRT-LLM/issues/12257) | CachedModelLoader issue when enabling speculative decoding with multiple MPI processes | Closed | |
| [#13135](https://github.com/NVIDIA/TensorRT-LLM/issues/13135) | [AutoDeploy][Bug]: Llama + Eagle3 + torch-simple + trtllm-attention crashes | Closed | |
| [#15120](https://github.com/NVIDIA/TensorRT-LLM/issues/15120) | [AutoDeploy] Speculative-only Mamba SSM/conv buffers allocated when spec dec is off (OOM) | Closed | |

> **#16448「rewind crosses block boundary」值得注意**：与 llama.cpp 的 `SEQ_RM_TYPE` 三档回滚机制指向同一件事——**投机解码的 KV 回滚跨 page/block 边界时是高危路径**。三家都在这里出过问题。

#### (e) 组合不兼容

| # | 标题 | 状态 |
|---|---|---|
| [#12617](https://github.com/NVIDIA/TensorRT-LLM/issues/12617) | **[Bug]: Ngram speculative decoding is incompatible with LoRA** | Closed |

报错两态：开 CUDA graph 时 `IndexError: index_copy_(): Number of indices (64) should be equal to source.size(dim) (576)`；关 CUDA graph 时 `Assertion failed: LoraParams and input dims don't match, lora tokens 1 input tokens 9`。**与 feature 矩阵 LoRA × spec dec = Untested 相互印证。**
> LoRA × 投机的三家现状：**vLLM ❌（矩阵直接标不兼容）、TensorRT-LLM ⚠️Untested（且有 issue 证明 NGram+LoRA 直接崩）、SGLang ✅但草稿不带 adapter。** 三家没有一家真正解决了这个问题。

#### (f) 老方法的历史坑
| # | 标题 | 状态 |
|---|---|---|
| [#3125](https://github.com/NVIDIA/TensorRT-LLM/issues/3125) | Model built with ReDrafter produces substantially lower quality outputs | Closed |
| [#2155](https://github.com/NVIDIA/TensorRT-LLM/issues/2155) | Recurrent Drafter not working | Closed |

#### (g) Release Notes 里自己修过的 spec dec bug（可作为"坑的密度"证据）

1.1 Fixed Issues：
> - Fixed **race conditions** in one-model speculative decoding.
> - Resolved **CUDA graph warmup issues** that caused failures when using speculative decoding.
> - Fixed **KV cache recompute logic** in `draft_target` speculative decoding.
> - Fixed **numerical stability issues for XQA kernels** when using speculative decoding.

0.21 Fixed Issues：
> - Fixed cuda graph padding for spec decoding (#4853)
> - Fixed index out of bounds error in spec decoding (#5954)
> - Fixed MTP illegal memory access in cuda graph warmup (#5947)
> - Fixed **no free slots error with spec decode + disagg** (#5975)
> - Fix **PD + MTP + overlap scheduler accuracy issue** (#6136)
> - Fix disagg + speculative decoding (#5558)
> - Fix mtp vanilla draft inputs (#5568)

### 3.9 TensorRT-LLM 官方 benchmark 数字

#### ⭐⭐ (a) Qwen3.8 MoE + MTP3 —— 全库最完整的「低延迟 vs 高吞吐」对照

来源：`docs/source/deployment-guide/deployment-guide-for-qwen3.8-qwen3.5-on-trtllm.md`

口径原文：
> The performance reference uses **GB300**, aggregated serving, and an exact **8192-input/1024-output-token** workload. Low-latency profiles use **TP16/EP1 on 16 GPUs at concurrency 1**. High-throughput profiles use attention DP with **TP32/EP32 on 32 GPUs** and a 544-slot static EPLB map. Low-latency throughput is output tokens per second per user (`1000 / median TPOT`); high-throughput is total input-plus-output tokens per second per GPU.

| Objective | Spec dec | Topology | Concurrency | Best audited value |
|---|---|---|---:|---:|
| Low latency | Disabled | TP16/EP1 | **1** | **133.502 output tok/s/user**（7.491 ms median TPOT） |
| Low latency | **MTP3** | TP16/EP1 | **1** | **383.051 output tok/s/user**（2.611 ms median TPOT） |
| High throughput | Disabled | Attention DP32/EP32, Static544 | **3264** | **4059.200 total tok/s/GPU** |
| High throughput | **MTP3** | Attention DP32/EP32, Static544 | **2304** | **4118.164 total tok/s/GPU** |

> MTP3 performance results use a **controlled accepted-draft count of 2.3**.

**换算：并发 1 时 MTP3 = 2.87× 延迟收益；并发 2304/3264 时 MTP3 只有 +1.46% 吞吐收益。**

口径完整性检查（铁律二七项）：
| 项 | 值 |
|---|---|
| batch size / 并发 | ✅ 1 与 2304/3264 |
| γ（K） | ✅ MTP3 = 3 |
| α / 平均接受长度 | ✅ **accepted-draft count 2.3**（注意这是 controlled，即人为控制的） |
| draft/target 组合 | ✅ Qwen3.8-2.4T-A95B **FP8** + 自带 MTP 头 |
| 硬件与精度 | ✅ GB300，FP8 权重，`kv_cache_config.dtype: fp8` |
| 测的是什么 | ✅ 低延迟档 = output tok/s/user（`1000/median TPOT`）；高吞吐档 = total tok/s/GPU |
| 任务分布 | ⚠️ 只给了 ISL/OSL = 8192/1024，未给数据集 |
| 引擎版本 | ❌ **原文未给出** |

> **这是本次调研中唯一一组七项口径基本齐全的官方数字，应当作为 `18-batch与吞吐` 的主表。**
> 但要注意 "**controlled** accepted-draft count of 2.3" 这个措辞——**它暗示这组数字也可能用了类似 `SGLANG_SIMULATE_ACC_LEN` 的接受率控制手段**。引用时必须带上这个词。

#### (b) DeepSeek-R1-FP4 + MTP（blog02）

> We tested the **min-latency (batch size = 1)** performance of the **DeepSeek-R1-FP4** model with different MTP next-n on a **B200 node**. The **MLA runs with TP=8, and the MoE runs with EP=2**. And there are **ten different requests with ISL/OSL=1K/2K**. ... **MTP=3** can help get the best min-latency performance on 8 B200 GPUs, which can bring **2.16x speedup** compared with the baseline nextn=0. And with the help of the relaxed acceptance, the min-latency performance can be further improved to achieve a **2.33x speedup**. ... **CUDA graph can achieve a 7.22x average speedup**, while the **overlap scheduler can achieve 1.03x** average latency.

| 项 | 值 |
|---|---|
| 模型 | DeepSeek-R1-**FP4** 671B |
| 硬件 | 1 node × 8 B200；MLA TP=8，MoE EP=2 |
| batch size | **1**（min-latency） |
| ISL/OSL | **1K / 2K**，10 requests |
| 指标 | **延迟**（相对加速比） |
| MTP=3 | **2.16×**；+ relaxed acceptance → **2.33×** |
| acceptance rate | 图 7 有 AR 曲线，**正文未给数值 → 原文未给出** |
| 引擎版本 | **原文未给出** |

> ⭐ **同一篇里 "CUDA graph 7.22×" 这个数字必须一起引**：**在 bs=1 的 min-latency 场景下，CUDA graph 的收益（7.22×）远大于 MTP3 的收益（2.16×）。** 这是一条很好的"先做对基础优化再谈投机"的证据，适合放进 `19-负收益全解` 的开头或 `03-并行验证为什么几乎免费`。

同篇精度结论：
> Compared with Strict Acceptance, the impact of Relaxed Acceptance on output quality is limited, resulting in only a slight accuracy drop.（ablation：MTP nextn=3, top-10, delta=0.6；**具体分数以图给出，未给数值**）
> Note that the Relaxed Acceptance will only be used during the thinking phase... And the **Relaxed Acceptance only supports the DeepSeek-R1 model now.**

> **Relaxed Acceptance 是 L3（有损）**，官方自己说 "slight accuracy drop"。按铁律一，凡引用 2.33× 这个数字必须同时标注"这是有损口径"。

且该 blog 写作时的状态（现已被 dynamic tree 取代）：
> TensorRT LLM PyTorch backend can only support **chain-based** speculative decoding now, both MTP Vanilla and MTP Eagle.

#### (c) Llama 4 Maverick + EAGLE-3（1000 TPS/user）

来源：https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/ （2025-05）+ blog06

| 项 | 值 |
|---|---|
| 成绩 | **> 1000 TPS/user**（Artificial Analysis 独立测量） |
| 硬件 | **单台 DGX B200，8 × Blackwell** |
| 模型 | Llama 4 Maverick 400B，**FP8**（GEMM/MoE/Attention 均 FP8） |
| 草稿 | `nvidia/Llama-4-Maverick-17B-128E-Eagle3`（**BF16**）。原文："NVIDIA uses a EAGLE3-based architecture as its speculative decoding method, **modifying only the FFN size of the speculative layer for better AL**" |
| draft 长度 | **"draft-length=3 provides the best speed-up"** |
| 该 draft 长度的加速 | **2.06×**，**Acceptance Length ≈ 2.0** |
| 整体 4× | "a 4x speed-up relative to the best prior Blackwell baseline"（**含全部软件优化，非投机单独贡献**） |
| 服务器峰值吞吐 | 72,000 TPS/server（**最高吞吐配置，与 1000 TPS/user 不是同一配置**） |
| batch size / ISL / OSL / 引擎版本 | **原文未给出** |
| 高并发行为 | blog06 明确警告 TPS/user "degrades rapidly"（见 3.5c） |

> ⚠️ **"4×" 与 "2.06×" 必须分清**：4× 是全部软件优化的合计，投机单独贡献是 2.06×。**媒体报道普遍把 4× 记在投机头上，这是典型的口径污染。**

#### (d) DFlash（2026 年最新的官方 blog）

来源：https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/

> DFlash increases inference performance for **gpt-oss-120b** on NVIDIA Blackwell by **up to 15x at the same interactivity level**.

| 场景 | 数字 | 硬件 | 框架 |
|---|---|---|---|
| gpt-oss-120b，500–600 tok/s/user 处 | **15× vs 自回归；1.5× vs EAGLE-3** | 8 × DGX B300 | TensorRT-LLM |
| gpt-oss-120b，**batch size 1** | interactivity > **2×** | Blackwell | TensorRT-LLM |
| SPEED-Bench 同并发平均加速 | gpt-oss-120b：**DFlash 2.3× vs EAGLE-3 1.7×**；Llama 3.1 8B：**DFlash 2.8× vs EAGLE-3 2.2×** | — | — |
| Gemma 4 31B（Math500/GSM8K/HumanEval/MBPP/MT-Bench） | 5.8× / 5.3× / 5.6× / 4.4× / 3.0× | 1 × B300 | **vLLM** |
| Qwen3-8B（Math500 / HumanEval） | 5.1× / 4.2× | 1 × B200 | **SGLang** |

**concurrency / batch size / ISL / OSL / acceptance rate / acceptance length / 引擎版本：全部原文未给出。**
> ⚠️ **"15×" 这个数字口径极不完整**（只知道"在 500–600 tok/s/user 这个 interactivity 点上"），按铁律二**属于不可横向比较的裸奔数字**，引用时必须显式标注。同时注意它是"vs 自回归"，而 vs EAGLE-3 只有 1.5×。

#### (e) NGram（blog07，Llama-4-Scout）

口径原文：**Hardware: 8 × B200；Model: Llama-4-Scout-17B-16E, FP8 weights；Tensor Parallel: 8**。指标为 E2E runtime 加速比。

| 数据集 | 配置 | batch size | 加速 / AL |
|---|---|---|---|
| Magpie 多轮，第 1 轮 | k=3, v=5 | 1 / 4 / 32 | 整体 **10–60% E2E**；AL **1.37** |
| Magpie，第 2 轮 | k=3, v=5 | **1** | **96.13%** |
| Magpie，第 2 轮 | k=3, v=5 | **4** | **63.99%** |
| Magpie，第 2 轮 | k=5, v=3 | **32** | **33.06%** |
| AL 表（3000 conversations） | k=3,v=5 / k=5,v=5 / k=5,v=3 | — | Turn1 **1.37 / 1.40 / 1.37**；Turn2 **1.66 / 1.77 / 1.66** |
| 翻译数据集（4000 requests） | k=5,v=7 → v=23 | — | AL 从 **3.44** 单调升到 **4.73**；延迟最多降 **70%** |

ISL/OSL 与引擎版本：**原文未给出**。
⚠️ blog07 的目录里有一节 `Feature Gaps`，但**正文中该节内容不存在**（文档缺失）。

> ⭐ **这组数据有两个极好的教学点**：
> ① **第 1 轮 AL=1.37，第 2 轮 AL=1.66**——n-gram 在多轮对话第二轮才真正起效（因为有了可匹配的历史）。**这正是 suffix decoding 的全局树、以及 llama.cpp `ngram-mod` 的"跨 slot 共享 hash pool"想解决的问题。**
> ② **翻译任务上 AL 能到 4.73（v=23）**——**无模型草稿在高度可预测任务上能超过很多模型式草稿**。这条挑战了"n-gram 只能有低收益"的直觉，应写进 `11-无模型草稿`。

#### (f) Guided decoding + spec dec（blog12）

口径原文：**LLaMA 3.1 8B (TP1) 与 LLaMA 3.3 70B (TP4)，GPU 为 H200，grammar backend 为 XGrammar，concurrency 从 1 到 128**，数据集 JSON Mode Eval / JSON Schema Bench。

> Speculative decoding achieves significant speedup for low-latency or throughput@latency scenarios. In particular, **the speedup can be up to ~2x for batch size 1**. The **one-model EAGLE3 implementation is more performant than the two-model EAGLE3**, and this performance gap is amplified for small models.

Acceptance length 表（原文）：

| Dataset | Model | EAGLE3 | EAGLE3 w/o draft | NGram |
|---|---|---|---|---|
| JSON Mode Eval | LLaMA 3.1 8B | **2.86** | 2.65 | 2.59 |
| JSON Mode Eval | LLaMA 3.3 70B | **2.72** | 2.60 | 2.44 |

> Note that although NGram is a two-model implementation, **it performs surprisingly well. This is because JSON Mode Eval is an information extraction task**... Note that the NGram becomes less performant since the task is no longer an information extraction task（JSON Schema Bench 上）。

> **再一次印证："任务分布"是接受率的第一决定因素**，且在信息抽取类任务上 n-gram 能逼近 EAGLE3（2.59 vs 2.86）。

#### (g) DeepSeek-V3.2（blog15）的 MTP 选型建议
> For latency-critical scenarios... **MTP-3 is recommended** to maximize GPU utilization and achieve optimal performance. **For other scenarios, MTP-1 typically offers performance gains as well.**

#### (h) trtllm-bench 会报的投机指标

`tensorrt_llm/bench/dataclasses/statistics.py`：`acceptance_length`、`num_draft_tokens_percentiles`、`num_accepted_draft_tokens_percentiles`、`draft_acceptance_rate_percentiles`、`acceptance_length_percentiles`。
且是**按解码迭代次数加权平均**（原文："This is used to weight per-request metrics (e.g. speculative decoding acceptance rate / acceptance length) by the number of decoding iterations"）。
> **这个加权口径很重要**：简单对请求取平均与按迭代数加权，在长短请求混合的负载下会给出不同的数字。**本库 `05-接受率alpha-定义口径与怎么测` 应当把这条列为口径分歧之一。**

Release Notes 1.0 记 "Add speculative metrics for trtllm-bench"、"Add Acceptance Rate calculation to benchmark_serving"。
DFlash/DSpark 另有**默认关闭**的每步接受直方图记录器：`TLLM_DFLASH_ACCEPT_STATS_DIR`、`TLLM_DFLASH_ACCEPT_STATS_FLUSH_EVERY`（默认 50）。原文注明 "**never perf-reference runs**"（会引入每步 D2H）。

### 3.10 TensorRT-LLM 侧「未查证」清单

1. NGC catalog 上是否有官方托管的 draft model / 草稿头。
2. HF `nvidia/*-Eagle3` 各仓库的 license 与逐仓库 base model 对应关系。
3. PyTorch backend 下 **beam search + speculative decoding** 是否全面可用。
4. 投机开启时 **logprobs / prompt_logprobs 的精确语义**。
5. 投机开启时 **n>1 / best_of>1** 的行为。
6. Pipeline parallel + spec dec 究竟是**硬报错**还是仅"未测试"（矩阵写 No，代码里未找到统一断言）。
7. DFlash blog 的 concurrency / ISL / OSL / acceptance length / 引擎版本（原文未给出）。
8. 1000 TPS/user blog 的 batch size / ISL / OSL / 引擎版本（原文未给出）。
9. features 文档中 `speculative-decoding.md#developer-guide` 锚点指向的 Developer Guide 段落——**该文件里不存在这一节**（死锚点）。

---

## 4. llama.cpp

### 4.0 版本锚点 + ⚠️ 一条必读的前置警告

| 项 | 值 |
|---|---|
| 最新 nightly build tag | **`b10573`**（2026-08-22T05:39:21Z） |
| 另有语义化 release | `v0.2.0`（2026-08-21T18:32:48Z） |
| 本节核验基准 | `master` 分支，2026-08-22（逐个文件 curl + grep 核验，非二手转述） |

> ⚠️⚠️ **网上（包括我原本的记忆）流传的 llama.cpp 投机参数已全部作废**。
> `--draft-max` / `--draft-min` / `--draft-p-min` / `-cd, --ctx-size-draft` / `--spec-replace` **在当前 master 上已全部被移除或改名**；旧默认值 `16 / 0 / 0.75` 也已变成 **`3 / 0 / 0.0`**。
> 2026 年上半年 llama.cpp 对投机解码做了一次彻底重构，引入统一的 `--spec-type`。**任何 2025 年及以前的 llama.cpp 投机教程现在都跑不通。**

核验过的文件：`common/arg.cpp`、`common/common.h`、`common/speculative.{h,cpp}`、`common/sampling.cpp`、`common/common.cpp`、`docs/speculative.md`、`tools/server/README.md`、`tools/server/server-context.cpp`、`server-task.{cpp,h}`、`server-schema.cpp`、`server-common.cpp`、`tools/server/tests/unit/test_speculative.py`、`tools/server/bench/speed-bench/README.md`。

### 4.1 支持哪些方法：**10 种，由 `--spec-type` 统一选择，且可逗号组合**

`common/common.h` L170–183 逐字：

```c
enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,          // no speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,  // standalone draft model speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,  // Eagle3 speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_MTP,     // Multi-token prediction
    COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,  // DFlash speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,  // DSpark speculative decoding (DFlash + Markov head)
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative decoding based on n-grams
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,   // self-speculative decoding with n-gram keys only
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, // self-speculative decoding with n-gram keys and 4 m-gram values
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_COUNT
};
```

| 方法 | 状态 |
|---|---|
| 独立小 draft model | ✅ `draft-simple` |
| lookup decoding | ✅ 演化成 `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` |
| ngram cache（老 `llama-lookup` 那套 3 级缓存） | ✅ `ngram-cache` |
| **Medusa** | ❌ **没有**（源码搜不到 medusa） |
| **EAGLE-3** | ✅ `draft-eagle3`，支持 SpecForge / vLLM-AngelSlim 两种 checkpoint 格式 |
| MTP | ✅ `draft-mtp`（DeepSeek / Qwen3.6 那类自带 nextn 头的模型） |
| DFlash | ✅ `draft-dflash` |
| DSpark | ✅ `draft-dspark` |

> **一个值得记的事实**：**llama.cpp 现在支持 EAGLE-3，vLLM 支持 Medusa，SGLang 两者都不支持 Medusa。** 三家引擎对 Medusa 的取舍高度一致——它已经出局。而 EAGLE-3 是唯一被三家（+TensorRT-LLM）全部实现的模型式草稿方法。

`docs/speculative.md` 对各方法的原文定义：
> **EAGLE-3** uses a small draft model that reads the target model's hidden states to predict the next tokens, so it reaches higher acceptance than a standalone draft model of the same size. The draft is a one-layer transformer trained for a specific target model; it shares the target model's tokenizer and, optionally, uses a reduced draft vocabulary with its own `lm_head`, which is mapped back using a `d2t` table.
> **DFlash** produces an entire block of draft tokens in a single forward pass (block diffusion) and injects the target model's hidden states into the draft model's attention, instead of drafting one token at a time.
> （ngram-mod）Currently, a **single hash pool is shared across all server slots, so different requests can benefit from each other.**

example 二进制现状：`examples/` 下**仍存在** `speculative`、`speculative-simple`、`lookup`、`lookahead`、`parallel`；`tools/` 下没有 speculative 独立二进制；主力入口是 `llama-server` / `llama-cli`。

### 4.2 参数名与默认值（逐字抄自源码与自动生成的 README）

#### ⚠️ 已被移除、传了会直接报错退出的旧参数

`common/arg.cpp` L4311–4345：
```c
{"--draft", "--draft-n", "--draft-max"}, "N",
"the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max",
{"--draft-min", "--draft-n-min"}, "N",
"the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min",
{"--spec-ngram-size-n"}, "N", "the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match",
{"--spec-ngram-size-m"}, "N",  "the argument has been removed. ..."
{"--spec-ngram-min-hits"}, "N", "the argument has been removed. ..."
```
抛错函数（L344）：`throw std::invalid_argument("the argument has been removed. " + msg);`

**连报错桩都没有、彻底消失的**（传了会被当未知参数）：
- `-cd, --ctx-size-draft` —— master 的 `arg.cpp` 与 `server/README.md` grep 命中 **0**。草稿上下文改由 `common_base_params_to_speculative()` 从目标模型参数推导（`speculative.cpp:2319`）。
- `--spec-replace` —— master grep 命中 **0**。

**`--spec-replace` 的版本二分核验**（对各 tag 的 `common/arg.cpp` 做 `grep -c`）：

| tag | `spec-replace` | `--spec-type` |
|---|---|---|
| b6500 / b6999 / b7100 | 1 | 0 |
| b8100 / b9000 | 1 | 1 |
| **b9500** | **0** | 4 |
| b10000 / master(b10573) | 0 | 多处 |

→ `--spec-type` 在 **b7100~b8100 之间**引入；`--spec-replace` 在 **b9000~b9500 之间**删除。**具体 PR 号未查证**（GitHub 搜索对连字符做模糊匹配，返回全是无关 PR）。
b6500 时 `--spec-replace` 的定义（供考古）：
```c
{"--spec-replace"}, "TARGET", "DRAFT",
"translate the string in TARGET into DRAFT if the draft model and main model are not compatible",
```

#### 当前 master 的默认值（`common/common.h` L324–367）

```c
struct common_params_speculative_draft {
    int32_t n_max = 3; // maximum number of tokens to draft during speculative decoding
    int32_t n_min = 0; // minimum number of draft tokens to use for speculative decoding
    float p_split = 0.1f;
    float p_min   = 0.0f; // minimum speculative decoding probability (greedy)
    bool backend_sampling = true; // offload draft sampling to the backend (default: on)
    int32_t n_gpu_layers = -1;
    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;
};
struct common_params_speculative_ngram_mod { int32_t n_match = 24; int32_t n_max = 64; int32_t n_min = 48; };
struct common_params_speculative_ngram_map { uint16_t size_n = 12; uint16_t size_m = 48; uint16_t min_hits = 1; };
struct common_params_speculative { std::vector<enum common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_NONE }; ... };
```

**新旧默认值对照**（b6500 的 `common/common.h:203`）：

| 字段 | 旧（b6500） | **新（master）** |
|---|---|---|
| `n_max` | **16** | **3** |
| `n_min` | 0 | 0 |
| `p_min` | **0.75f** | **0.0f** |
| `p_split` | 0.1f | 0.1f |
| `n_ctx`（draft 上下文） | 0（有该字段） | **字段已删** |
| `replacements` | 有 | **已删** |

> `n_max` 从 16 掉到 3 是个大变化：因为默认类型已改成"不开"，而 3 是 EAGLE3/MTP 这类高接受率草稿的常用深度。**这条对本库 `17-动态草稿长度` 有直接价值——引擎默认 γ 从 16 降到 3，是"草稿质量提高后最优 γ 反而下降"的实证。**

#### 完整参数表（逐字抄自 `tools/server/README.md`，该表由 `llama-gen-docs` 自动生成）

> README 原文开头：**IMPORTANT: The list below is auto-generated by llama-gen-docs; do NOT modify it manually**（所以这张表可信度高于手写文档）

**总开关**

| 参数 | 说明（原文） |
|---|---|
| `--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache` | comma-separated list of types of speculative decoding to use **(default: none)** (env: `LLAMA_ARG_SPEC_TYPE`) |
| `--spec-default` | enable default speculative decoding config |

`--spec-default` 实际做了什么（`arg.cpp:4647`）：
```c
params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
params.speculative.ngram_mod.n_match = 24;
params.speculative.ngram_mod.n_min = 48;
params.speculative.ngram_mod.n_max = 64;
// TODO: not sure if this is a good config - explore more settings and potentially enable it
//params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
```

**draft 模型参数**

| 参数 | 说明（原文含默认值） |
|---|---|
| `--spec-draft-model, -md, --model-draft FNAME` | draft model for speculative decoding **(default: unused)** |
| `--spec-draft-hf, -hfd, -hfrd, --hf-repo-draft <user>/<model>[:quant]` | Same as --hf-repo, but for the draft model |
| `--spec-draft-n-max N` | number of tokens to draft **(default: 3)** |
| `--spec-draft-n-min N` | minimum number of draft tokens **(default: 0)** |
| `--spec-draft-p-split, --draft-p-split P` | speculative decoding split probability **(default: 0.10)** |
| `--spec-draft-p-min, --draft-p-min P` | minimum speculative decoding probability (greedy) **(default: 0.00)** |
| `--spec-draft-backend-sampling` / `--no-spec-draft-backend-sampling` | offload draft sampling to the backend **(default: enabled)** |
| `--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft N` | max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' **(default: auto)** |
| `--spec-draft-device, -devd, --device-draft <dev1,dev2,..>` | 卸载草稿模型的设备（none = 不卸载） |
| `--spec-draft-type-k, -ctkd, --cache-type-k-draft TYPE` | draft 的 K cache dtype **(default: f16)**；允许 f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1 |
| `--spec-draft-type-v, -ctvd, --cache-type-v-draft TYPE` | 同上，V **(default: f16)** |
| `--spec-draft-override-tensor, -otd` / `--spec-draft-cpu-moe, -cmoed` / `--spec-draft-n-cpu-moe, -ncmoed N` | draft 侧的张量 buffer/MoE 放置 |
| `--spec-draft-threads, -td` / `--spec-draft-threads-batch, -tbd` | (default: same as --threads / --threads-draft) |
| `--spec-draft-cpu-mask, -Cd` / `-Crd` / `--cpu-strict-draft` / `--prio-draft` / `--poll-draft` (+各 `-batch` 变体) | CPU 亲和性系列 |

**n-gram 系列**

| 参数 | 默认值 |
|---|---|
| `--spec-ngram-mod-n-match N` | **24** |
| `--spec-ngram-mod-n-min N` | **48** |
| `--spec-ngram-mod-n-max N` | **64** |
| `--spec-ngram-simple-size-n / -size-m / -min-hits` | **12 / 48 / 1** |
| `--spec-ngram-map-k-size-n / -size-m / -min-hits` | **12 / 48 / 1** |
| `--spec-ngram-map-k4v-size-n / -size-m / -min-hits` | **12 / 48 / 1** |

**开箱预设**：`--fim-qwen-7b-spec`（Qwen 2.5 Coder 7B + 0.5B draft）、`--fim-qwen-14b-spec`（14B + 0.5B draft）。

### 4.3 ⚠️⚠️ 最大的坑：`-md` 单独用**不会**开启投机解码

从源码链路完整验证：

1. `common_speculative_init()`（`speculative.cpp:2455`）通过 `common_get_enabled_speculative_configs(params.types)` 做位掩码，**只有 `types` 里显式包含某类型才实例化对应实现**：
   ```c
   auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
       if (available && (enabled_configs & (1u << type))) { configs.emplace_back(type, params); }
   };
   add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
   ```
2. 自动推断**只覆盖 MTP / DFlash / DSpark / EAGLE3**（`arg.cpp:564`）：
   ```c
   // infer the speculative type from the draft GGUF metadata when none is requested
   // note: reads only the first split - sharded drafts need an explicit --spec-type
   if (spec_types_is_default(params) && !params.speculative.draft.mparams.path.empty()) {
       const auto types_gguf = common_speculative_types_from_gguf(params.speculative.draft.mparams.path);
   ```
   而 `common_speculative_types_from_gguf()`（`speculative.cpp:2238`）只认 `general.architecture == "dflash"` 或 `blk.N.nextn.eh_proj.weight` 张量。**普通 Qwen/Llama 小模型 → 返回空 → `types` 停留在 `{NONE}`。**
3. 在 `arg.cpp` / `common.cpp` / `server-context.cpp` / `server-task.cpp` 全量 grep `DRAFT_SIMPLE`，**命中 0**。

**结论：普通小 draft model 必须显式写 `--spec-type draft-simple`，否则草稿模型会被加载进显存但永远不会被调用——不报错、不提示、白占显存。**

上游自己也踩过这个坑，PR #26814 的 commit message 原文：
> When `-md` loads a local draft model without `--spec-type`, the sidecar inference in `common_models_handler_apply` only checks HF repo sidecars and misses local files. **The draft model loads into VRAM but speculative decoding never activates (types stays NONE).**
（该 PR 只修了 dflash/dspark，`draft-simple` 依然需要显式指定。）

**顺带发现的疑似上游 bug**：`--fim-qwen-7b-spec` / `--fim-qwen-14b-spec` 两个预设（`arg.cpp:4543`）只设置了 `hf_repo`/`hf_file`，**没有设置 `params.speculative.types`**。按上述链路推断这两个预设当前不会真正启用投机解码。**此为读源码的推断，未实机验证。**

### 4.4 ⚠️ 服务端 per-request 覆盖字段：**当前被 `#if 0` 关掉了**

`tools/server/server-schema.cpp` L193–225 逐字：
```c
//
// Speculative decoding params
//
// TODO: to keep things simple, we disable speculative parameter adjustments for now
#if 0
// TODO: for now, be able to adjust only the draft-model based speculative parameters
add((new field_num("speculative.n_max", params.speculative.draft.n_max)) ...
add((new field_num("speculative.n_min", params.speculative.draft.n_min)) ...
add((new field_num("speculative.p_min", params.speculative.draft.p_min)) ...
add((new field_str("speculative.type")) ...
add((new field_num("speculative.ngram_size_n", params.speculative.ngram_simple.size_n)) ...
```

**→ `/completion` 请求体里的 `speculative.n_max` / `n_min` / `p_min` / `type` 目前不生效。**
`server-task.cpp` 只在返回的 generation_settings 中报告 `{"speculative.types", common_speculative_type_name_str(speculative.types)}`。

**README 自身是矛盾且过时的**：`tools/server/README.md` 的 `/props` 与 `/slots` 响应示例里仍写着
```json
"speculative.n_max": 16, "speculative.n_min": 5,  "speculative.p_min": 0.8999999761581421   // L885-887
"speculative.n_max": 16, "speculative.n_min": 0,  "speculative.p_min": 0.75                 // L1032-1034, L1097-1099
```
**三组值互相矛盾且与代码默认值（3/0/0.0）都不一致**，是手写的陈旧示例，不要采信。

同样，`docs/speculative.md` L103 仍写着 `--spec-draft-conf-min P`，但 `arg.cpp` 里 `grep -c "conf-min"` = **0**。PR #25173 commit "dspark: fold conf_min into p_min" 原文：
> p_min and conf_min express the same thing ... so reuse it for the confidence threshold and **drop the separate `--spec-draft-conf-min` flag**. Both defaulted to 0 (disabled), so behavior is unchanged.

### 4.5 对 draft model 的硬性要求：词表必须匹配

`common/speculative.cpp` L29–30：
```c
#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5
```

`common_speculative_are_compatible()`（L69–130）四道检查：
1. **vocab type 必须相同**（硬失败）：`"draft model vocab type must match target model to use speculation but vocab_type_dft = %d while vocab_type_tgt = %d"`
2. **BOS 必须一致**：`"draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)"`
3. **EOS 必须一致**：同上句式
4. **词表大小差 ≤ 128，且逐 token 文本必须相同**（从 id=5 开始比）：
   > `"draft model vocab must closely match target model to use speculation but target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d"`
   > `"draft model vocab must match target model to use speculation but token %d content differs - target '%s', draft '%s'"`

失败时（L236–242）：
```c
if (!vocab_cmpt) {
    SPC_ERR("%s", "the target and draft vocabs are not compatible\n");
    throw std::runtime_error("draft model vocab type must match target model to use speculation");
}
```

**结论**：
- 词表**必须**基本一致（允许 ≤128 的尾部差异，通常是新增 special token）。**没有跨 tokenizer 的通用投机**（不像 HF 的 UAG，也不像 vLLM 的 `use_heterogeneous_vocab` TLI）。
- 唯一放宽手段 `--spec-replace`（字符串级替换）**已被移除**。
- 唯一"合法的不同词表"路径是 **EAGLE-3 的 reduced draft vocab + `d2t` 映射表**，但那是 EAGLE3 转换脚本内部处理的，不是通用放宽。EAGLE3/DFlash/DSpark 转换必须带 `--target-model-dir`：
  > Convert the EAGLE-3 checkpoint with `--target-model-dir` so it inherits the target's tokenizer and the layer indices to read.

> **三家横比（写进 `20-主流引擎实现` 的好素材）**：跨 tokenizer 草稿——HF transformers 有 UAG（无损版，arXiv:2502.05202（2025-02））、vLLM 有 TLI（`use_heterogeneous_vocab`，仅 greedy）、**llama.cpp 完全没有**、SGLang 有 FR-Spec 式的小词表（`--speculative-token-map`）但那是同 tokenizer 的子集，不是跨 tokenizer。

### 4.6 与 server 连续批处理 / slots / `-np` / prompt cache / KV shift 的交互

**并行（`-np`）：现在支持了，不再是 `n_parallel=1` 限制**
`server-context.cpp:1242`：`spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));`
每个 slot 拿同一个 `spec` 句柄，draft 状态按 `llama_seq_id` 分离。输出预算按并发算（`speculative.cpp:2442`）：
```c
common_speculative_output_limits common_speculative_get_output_limits(int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;
    return { std::min(n_batch, total), std::min(n_batch, per_seq) };
}
```
官方测试覆盖多 slot：`test_speculative.py::test_multi_requests_parallel` 参数化 `(n_slots, n_requests) = (1,2), (2,2)`。

> **这是 llama.cpp 与 HF transformers 最大的架构差异**：llama.cpp 的投机能与多 slot 连续批处理共存；transformers 的 assisted generation **硬性 batch_size=1** 且与其 continuous batching 互斥（见 §5.5）。

**context 能力门禁**（`server-context.cpp:1221`）：
```c
ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
    SRV_WRN("%s", "speculative decoding not supported by this context\n");
}
if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
    SRV_TRC("%s", "speculative decoding will use checkpoints\n");
}
```
三档行为：
- `SEQ_RM_TYPE_NO` → **完全禁用投机**（只打一条 WARN）。适用于无法按序列局部回滚 KV 的上下文，例如部分 recurrent / SSM 类模型。
- `SEQ_RM_TYPE_FULL` → 部分接受时**无法局部回滚，必须靠 checkpoint 全量恢复**（更贵）
- `SEQ_RM_TYPE_RS` → 回滚量超过 `llama_n_rs_seq(ctx_tgt)` 时也退化到 checkpoint

对应代码（`server-context.cpp:3851`）：
```c
const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));
if (n_rollback > 0) {
    if (use_ckpt_tgt) {
        SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", ...);
        slot.spec_is_replay = true;
```

> **这条揭示了一个本库正文该讲的机制**：投机解码要求 KV cache **能按序列局部回滚**（拒绝后裁掉多余的 K/V）。做不到局部回滚的架构（SSM/Mamba/RWKV 这类状态式模型）要么完全不能开投机，要么必须靠"全量 checkpoint + 重放"来模拟，代价大幅上升。HF transformers 那边的对应表述是 `raise ValueError("assisted generation is not supported with stateful models")`（见 §5.4）。**两家用不同方式撞上了同一堵墙。**

**sub-batch 冲突（已知硬限制，会直接抛异常）**，`server-context.cpp:3720` 逐字：
```c
// TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
// so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
iterate(slots, [&](server_slot & slot) {
    for (auto & i : slot.spec_i_batch) {
        if (!is_inside_view(i)) {
            throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
        }
    }
});
```
→ `-b` / `-ub` 太小、导致一次 draft 的 `n_max+1` 个 token 被拆进不同 sub-batch 时，**服务端直接抛 runtime_error**。实践含义：**`-b` / `-ub` 必须 ≥ `n_parallel × (n_max + 1)`**。追踪 issue：ggml-org/llama.cpp#24840。

**KV shift / context shift：兼容，有官方测试**（`test_with_ctx_shift()`：`n_ctx=256`、`enable_ctx_shift=True`、prompt 248 token、`n_predict=256`，断言 200 且 `tokens_predicted == 256`、`truncated == True`）。另有 `test_slot_ctx_not_exceeded()` 覆盖"draft 不得撑爆 slot 上下文"。
注意 ctx_shift 自身有前置禁用条件：`ctx_shift is not supported by multimodal, it will be disabled` / `ctx_shift is not supported by this context, it will be disabled`。

**prompt cache / checkpoint**：投机状态会跟着 context checkpoint 一起存（`server-context.cpp:2295` 注释 `// stash the draft's speculative state with the checkpoint`），配合 `--ctx-checkpoints` / `checkpoint_min_step`。

### 4.7 采样兼容性与无损性口径

#### ⭐ 接受规则：**不是 Leviathan 的拒绝采样，而是"目标采样 + 相等即接受"**

`common/sampling.cpp:678` 全文逐字：
```c
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    std::vector<llama_token> result;
    result.reserve(idxs.size());
    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
        if (draft[i] != id) { break; }
    }
    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);
    }
    return result;
}
```

**解读（对本库铁律一极其重要）**：
- 每个位置都从**目标模型自己的完整采样链**（temperature / top-k / top-p / min-p / DRY / XTC / mirostat / grammar）里采一个 `id`
- 只有 `draft[i] == id` 才继续；否则立刻停在 `id`
- 输出的每个 token 都是从目标分布采出来的 → **口径上是 L1 分布无损**
- 但**接受率低于**标准投机采样（Leviathan Alg.1 用 $\min(1, p/q)$ 接受，比"必须完全相等"宽松）
- → **同样的草稿质量下，llama.cpp 的理论加速比低于 HF 的 `do_sample=True` 路径**

> 这一条应当写进 `04-拒绝采样修正-无损性的完整证明` 的"工程变体"小节：**"相等即接受"是拒绝采样的一个特例/退化版——它无损但更保守。** 一个引擎选它不是因为不懂，而是因为它不需要草稿模型的概率 $q$，实现简单且与任意 sampler chain 天然兼容。SGLang 的 `tree_speculative_sampling_target_only`、HF 的 Case 2、llama.cpp 全都是这一路。**"三大本地/服务引擎的默认路径都不是教科书上的 Leviathan Alg.1"——这是本库能给出的一个反直觉结论。**

**官方测试直接验证无损性**（`test_speculative.py::test_with_and_without_draft`）：
```python
request = {"prompt": "I believe the meaning of life is", "temperature": 0.2, "top_k": 5,
           "seed": 4242, "n_predict": 16, "return_tokens": True}
# 无 draft 跑一遍，有 draft 跑一遍
assert tokens_no_draft == tokens_draft
```
→ **temperature=0.2 + top_k=5 的随机采样下，token 序列完全一致**（因为固定 seed + 相等即接受 ⇒ 连 L2 逐 token 相同都成立）。
`test_different_draft_min_draft_max()` 遍历 `(1,2)(1,4)(4,8)(4,12)(8,16)` 五组，断言 `last_content == res.body["content"]` —— **draft 深度不影响输出内容**。

近期 commit 里贡献者自报的等价性验证：
> Verified against gemma4-26b-a4b-dspark: **greedy outputs are byte-identical with and without the draft**; acceptance 0.46, mean draft len 3.7 (n_max 6).
> Output equivalence holds ... at temperature 0 the drafted response is **byte-identical** to the same request served with no draft attached (1213/1213 chars)

**破坏无损性的唯一已知因素是 backend sampling 的浮点差异**，`docs/speculative.md` "Backend Sampling" 节逐字：
> Unsupported samplers and device layouts fall back to CPU sampling. **Tensor split mode does not support backend sampling.** A fixed seed produces repeatable random draws, but **stochastic CPU and backend sampling can still select different tokens because floating-point operations can differ between implementations and devices. Use greedy sampling when exact output matching is required.**

#### 采样特性支持表

| 特性 | 状态 |
|---|---|
| temperature / top-p / top-k / min-p / DRY / XTC / mirostat | ✅ 全支持不受限（验证走目标模型完整 sampler chain；`--spec-draft-p-min` 只过滤草稿自己没把握的位置，不改目标分布） |
| **Grammar (GBNF) / JSON schema** | ✅ 支持。`common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first)` 在每个草稿位置都调完整采样器；`common_grammar_type` 覆盖 `USER`(`--grammar`)/`OUTPUT_FORMAT`(`--json-schema`)/`TOOL_CALLS`。**没有找到任何"开 draft 就禁 grammar"的代码或文档** |
| **logprobs / `n_probs`** | ❌ **投机路径下不产出** |
| beam search | 不适用（llama.cpp 无 beam search） |

**logprobs 的证据**（`server-context.cpp:3923`，草稿接受后发 token 的循环）：
```c
for (size_t i = 0; i < ids.size(); ++i) {
    completion_token_output result;
    result.tok          = ids[i];
    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
    result.prob         = 1.0f; // set later
    // TODO: set result.probs
```
该分支**没有调用 `populate_token_probs()`**（普通解码路径才调）。
→ **开启投机解码时 `n_probs` / `logprobs` 拿不到真实概率，`prob` 恒为 1.0，`probs` 数组为空。这是本次调研发现的、未见于任何文档的实际限制。**
（对照：vLLM 兼容矩阵标 SD × logP = ✅；SGLang 走 `compute_spec_logprobs()` 支持。**只有 llama.cpp 在这一点上是残缺的。**）

### 4.8 监控与观测字段

`tools/server/server-common.cpp:83`：
```c
base["draft_n"]          = n_draft_tokens;
base["draft_n_accepted"] = n_draft_accepted;
```
→ `/completion` 响应的 `timings` 对象里有 `draft_n` 与 `draft_n_accepted`（`test_speculative.py` 断言 `res.body["timings"]["draft_n"] > 0`）。

Prometheus 指标（README L1137–1140 逐字）：

| 指标 | 类型 | 说明 |
|---|---|---|
| `llamacpp:spec_decode_num_draft_tokens_total` | Counter | Total draft tokens generated (0 when spec-decode is off). |
| `llamacpp:spec_decode_num_accepted_tokens_total` | Counter | Total draft tokens accepted by the target model (0 when spec-decode is off). |
| `llamacpp:spec_decode_num_drafts_total` | Counter | Total speculative decoding verification steps (0 when spec-decode is off). |
| `llamacpp:spec_decode_num_accepted_tokens_per_pos_total` | Counter | Accepted tokens per draft position (labeled `position="N"`). |

> **有意思的一致性**：llama.cpp 这四个指标名与 vLLM 的 `vllm:spec_decode_num_draft_tokens_total` / `num_accepted_tokens_total` / `num_drafts_total` / `num_accepted_tokens_per_pos` **一一对应**。说明"drafts / draft_tokens / accepted_tokens / accepted_per_pos"这四元组已经成为事实标准的投机解码可观测性口径。**写 `05-接受率alpha-定义口径与怎么测` 时应当以这四元组为基准口径。**

CLI 结束时打印（`docs/speculative.md`）：
```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

### 4.9 llama.cpp 官方自认的限制与负收益

**① ggerganov 在首发 PR #10455（"server : add speculative decoding support"，2024-11-25 merged）的定调：**
> "The speedup can vary in both ways based on the inputs, but **enabling speculative should almost never result in slower than normal decoding.**"
https://github.com/ggml-org/llama.cpp/pull/10455

⚠️ **但同一个 PR 的实测数据反驳了这句话**（见 §4.10 表 A）：mostlygeek 的第一组表格里，开 `-md` 后 3090 从 34→31 tps、P40 从 11.22→10.52、3×P40 从 10.6→9.8，**三台机器全部变慢**。
→ **这是本库 `19-负收益全解` 的又一个好素材：维护者的定性判断"几乎不会更慢"，在同一页的实测表格里当场被推翻。**

**② 多实现叠加时的优先级**（`docs/speculative.md`）：
> If a draft model is combined with a draftless decoding **the draftless decoding has higher precedence.**

硬编码优先级（`speculative.cpp:2470`）：
```c
// this list here defines the priority of the speculators
// the one with highest priority are listed first
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
...
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
```

**③ ngram-mod 的适用/不适用，文档直说**：
> ```
> # notes:
> # - small `n` are not recommended
> # - MoEs require long drafts
> # - dense models: can reduce `--spec-ngram-mod-n-min` and `--spec-ngram-mod-n-max`
> ```
> Applications:
> - Iterating over a block of text/code (e.g. in llama.vim)
> - Reasoning models (when they have to repeat their thinking in the final answer)
> - Summarization

**④ 块草稿被强制截断**：
> `--spec-draft-n-max` is clamped to the draft model's trained block size.

对应 warning（`speculative.cpp:981`）：`"requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d"`

**⑤ DSpark 后端限制**：
> Currently only drafts with a Qwen3 backbone are supported; support for other backbones (e.g. Gemma4) is planned.

**⑥ ⚠️ 关于"大 batch 会变慢"**：**在 `docs/speculative.md` 和 `tools/server/README.md` 中都没有这句话**，ggerganov 说的反而是 "should almost never result in slower"。SPEED-Bench 文档还鼓励在 `--concurrency == --np` 下测吞吐：
> If you care about throughput numbers, set the client `--concurrency` to the server's slot count (`--np`)
→ **llama.cpp 官方文档里没有"大 batch 请关投机"的表述，标为「未查证」。** 这与 vLLM/SGLang 形成鲜明对比——很可能是因为 llama.cpp 的典型场景是本地单用户（bs=1），大 batch 不是它的主战场。

### 4.10 llama.cpp benchmark 数字

#### PR #10455（2024-11，服务端投机解码首发）—— mostlygeek 实测

**模型对**：Qwen2.5-Coder-32B-Instruct（Q8_0）target + Qwen2.5-Coder-0.5B-Instruct（Q4_0）draft
**命令**（PR 正文逐字，注意用的是已废弃的旧参数名）：
```
./bin/llama-server \
    -m ../models/qwen2.5-32b-coder-instruct/ggml-model-q8_0.gguf \
    -md ../models/qwen2.5-0.5b-coder-instruct/ggml-model-q4_0.gguf \
    -ngl 99 -ngld 99 -fa --port 8033 -c 32768 \
    --draft-max 16 --draft-min 5
```
**batch size = 1（单请求，延迟口径 tg tokens/s）**

**表 A —— 初版（有 draft vs 无 draft），全部变慢：**

| GPU | baseline pp | baseline eval | `-md` pp | `-md` eval |
|---|---|---|---|---|
| 3090 | 299 tps | **34 tps** | 300 tps | **31 tps** |
| P40 | 101 tps | **11.22 tps** | 101 tps | **10.52 tps** |
| 3×P40 | 91 tps | **10.6 tps** | 90 tps | **9.8 tps** |

**表 B —— 修正配置后（n_max:0 vs n_max:16），高可预测性 prompt：**

| GPU | n_max:0 | n_max:16 | 加速比 |
|---|---|---|---|
| P40 | 8.7 tps | 39.4 tps | **4.45×** |
| 3×P40 | 12.70 tps | 53 tps | **4.17×** |
| 3090 | 29 tps | 167 tps | **5.73×** |

**表 C —— 短 prompt（"write snake game in swift"），同硬件同模型：**

| GPU | n_max:0 | n_max:16 | 加速比 |
|---|---|---|---|
| P40 | 10.54 tps | 17.11 tps | **1.62×** |
| 3×P40 | 16.22 tps | 22.80 tps | **1.4×** |
| 3090 | 34.78 tps | 51.31 tps | **1.47×** |

**表 D —— 换模型对（Llama 3.1 70B + 1B draft）：**

| GPU | n_max:0 | n_max:16 | 加速比 |
|---|---|---|---|
| 3×P40 | 9.80 tps | 12.27 tps | **1.25×** |

来源：https://github.com/ggml-org/llama.cpp/pull/10455
⚠️ 数字经两次独立抓取交叉一致；但均为**社区贡献者实测，非维护者官方基准**。
> ⭐ **表 B 与表 C 的 4 倍差距（同硬件、同模型、同参数，只换 prompt）是本库最好的"任务分布决定接受率"证据**。按铁律二的七项口径，这两表只差"任务分布"一项，所以可以直接对比——这在本次调研的所有数据里是唯一一组真正可横比的。

#### Discussion #10466（社区大规模横评，steampunque，RTX 4070）
> "Coding shows a max speedup of **2.5x tg at 10 draft tokens** speculated using 0.5B model."
> "With **Gemma2 9B it drafted by Gemma2 2B it there is never any speculative decoding speedup.**"
> "0.5B size seems to work well...**returns fall off rapidly as draft gets bigger, already questionable at 1.5B and not really useful at 3B draft.**"
> "**Small draft model is needed (sine qua non).**"

极端可预测任务（"generate the integers from 0 to 100 separated by spaces"）：**176.53 tps（开）vs 61.83 tps（关）**。
随机数生成任务：**63.1 tps（开/关无差别）**。
来源：https://github.com/ggml-org/llama.cpp/discussions/10466
⚠️ 表格数字转述自讨论页摘要，**未逐字核验到原始 markdown**（引号内句子是逐字的）。

#### 近期 commit 里自报的接受率（逐字）

| 组合 | 接受率 | 出处 |
|---|---|---|
| gemma4-26b-a4b + DSpark draft（n_max 6） | **acceptance 0.46, mean draft len 3.7** | `model: support speculators-format checkpoints for DSpark (#26275)` |
| RedHat gemma-4-31b speculator draft（n_max 7） | **acceptance 0.26** | 同上 |
| muse-glimmer-30B + DFlash draft（图片请求） | **acceptance 0.34012 (167 accepted / 491 generated), mean len 3.04** | Muse Glimmer PR |

#### 官方基准工具（2026 新增）：`tools/server/bench/speed-bench/`

基于 NVIDIA 的 [SPEED-Bench 数据集](https://huggingface.co/datasets/nvidia/SPEED-Bench)：
> A lightweight SPEED-Bench client for benchmarking an already-running `llama-server` through its OpenAI-compatible API. It is primarily meant to evaluate speculative decoding (draft model, n-gram, MTP, EAGLE3, ...) by reporting per-category throughput, latency, and draft acceptance.

指标定义（逐字）：
- `avg_pred_t/s` — decode throughput from llama.cpp (`timings.predicted_per_second`)
- `accept_rate` — `accepted / draft_n` over the category，或 `n/a`（`draft_n == 0` 时）
- `decode_speedup = spec_avg_pred_t/s / base_avg_pred_t/s`
- `latency_speedup = base_avg_latency / spec_avg_latency`

分类：`coding`, `humanities`, `math`, `multilingual`, `qa`, `rag`, `reasoning`, `roleplay`, `stem`, `summarization`, `writing`；吞吐档 `throughput_1k` ~ `throughput_32k`。
→ **这是目前评估 llama.cpp 投机解码最可靠的方法**，因为它强制 baseline/spec 两次跑相同 prompt 集，且**按任务分类给接受率**——正好对应铁律二的第 7 项。

### 4.11 llama.cpp 已知 bug / 反复出现的坑

| # | 标题（逐字） | 状态 | 日期 |
|---|---|---|---|
| **#25618** | Eval bug: Speculative decoding (draft-mtp / draft-dspark): **greedy output diverges from vanilla on quantized targets** | Open | 更新于 2026-08-22 |
| #27306 | [CUDA] Eval bug: draft-mtp DeviceLost during *prompt* on AMD RADV | Open | 2026-08-22 |
| #27460 | Eval bug: draft-mtp (self-speculative) models crash on Vulkan/RADV after Linux kernel bump 7.1.3 → 7.1.7 — same class as #24492 | Open | 2026-08-20 |
| #27428 | eval bug: draft-mtp **roughly halves prompt processing on multi-GPU layer split**（单 GPU 正常） | Open | 2026-08-20 |
| #27469 | Feature Request: Expose Speculative Decoding (MTP / Draft) in public C API (llama.h) for downstream bindings | Open | 2026-08-21 |
| #24840 | sub-batch 与 `common_sampler_sample_and_accept_n` 不兼容（由源码注释直接引用） | 见 §4.6 | — |
| #27404 | `spec : avoid binding reference to null pointer` | Merged | 2026-08-17 |

**归纳的坑（按复现频率）**：
1. **量化目标模型 + MTP/DSpark 会破坏无损性**（#25618）——贪心输出与不开投机时不一致。**这是投机解码最严重的一类 bug，因为它违反"drop-in 替换"的核心承诺**，且与 HF #47932、SGLang #35771 属同一族群（详见 §7）。
2. **Vulkan/RADV 后端 + draft-mtp 崩溃**（#27306、#27460、#24492）——跨内核版本反复出现。
3. **多 GPU layer split + draft-mtp 让 prompt processing 慢一半**（#27428）。
4. **sub-batch 边界异常**（#24840）——`-b`/`-ub` 太小时直接 500。
5. **参数改名导致静默失效**（当前最普遍的用户侧坑）：老命令行 `-md ... --draft-max 16` 现在会因 `--draft-max` 被移除而**启动失败**；而只写 `-md` 又会**静默不启用**（§4.3）。第三方博客专门写过这个：`neoteric.no/blog/llama-cpp-mtp-flag-rename`（"Your Local Qwen3.6 Throughput Probably Just Halved (and How to Fix It)"，2026-05-13 起 MTP 标志改名为 `--spec-type draft-mtp`）。
6. **多模态 + DFlash 曾整个挂掉**（已修），commit 逐字记录症状：
   > Every image request with `--spec-type draft-dflash` failed with HTTP 500. Text-only was unaffected... before: HTTP 500, `failed to process speculative batch`; after: HTTP 200, draft acceptance 0.34012 (167 accepted / 491 generated), mean len 3.04

### 4.12 llama.cpp 侧「未查证」清单

1. `--spec-replace` 被移除的**具体 PR 号与日期** —— 只做到版本二分（b9000 有、b9500 无）。
2. `--spec-type` 引入的**具体 PR 号** —— 只做到 b7100~b8100 区间。相关线索：#18039 (EAGLE3)、#22673 (MTP)、#25173 (DSpark)、#22105 (DFlash)、#26814 (GGUF 自动探测)，但"重命名 `--draft-max` → `--spec-draft-n-max`"具体是哪个 PR 未确认。
3. llama.cpp 官方是否书面声明过"大 batch 下投机会变慢" —— **在两处主要文档中都没有这句话**。
4. `--fim-qwen-7b-spec` / `--fim-qwen-14b-spec` 预设不设 `--spec-type` 因而不生效 —— **源码推断，未实机验证**。
5. Discussion #10466 的表格数字未逐字核验原始 markdown。
6. issue 列表覆盖不完整（GitHub 搜索页只返回前若干条）。

---

## 5. HuggingFace transformers

### 5.0 版本锚点

- PyPI 最新：**transformers 5.15.1**
- 源码抓取自 `huggingface/transformers` 的 `main` 分支，2026-08-22：`src/transformers/generation/{configuration_utils.py, candidate_generator.py, utils.py}`
- 文档：https://huggingface.co/docs/transformers/main/en/main_classes/text_generation

### 5.1 Assisted generation 现在是 **8 条路线**，不是 3 条

`candidate_generator.py` 的类清单（逐字，含行号）：
```
39:   class CandidateGenerator
80:   class AssistedCandidateGenerator(CandidateGenerator)
341:  class AssistedCandidateGeneratorDifferentTokenizers(AssistedCandidateGenerator)
682:  class AssistantToTargetTranslator
845:  class AssistantVocabTranslatorCache
899:  class UniversalSpeculativeDecodingGenerator(AssistedCandidateGeneratorDifferentTokenizers)
1018: class PromptLookupCandidateGenerator(CandidateGenerator)
1174: class EarlyExitCandidateGenerator(AssistedCandidateGenerator)
1230: class SinglePositionMultiTokenCandidateGenerator(AssistedCandidateGenerator)
1423: class MTPCandidateGenerator(AssistedCandidateGenerator)
1564: class DFlashTokenCandidateGenerator(CandidateGenerator)
```

**路由逻辑**（`utils.py::_get_candidate_generator`，L1005–1120）——唯一真相来源，优先级从上到下：

| 优先级 | 触发条件 | 生成器 |
|---|---|---|
| 1 | `generation_config.assistant_early_exit is not None` | `EarlyExitCandidateGenerator` |
| 2 | `generation_config.prompt_lookup_num_tokens is not None` | `PromptLookupCandidateGenerator` |
| 3 | `generation_config.use_mtp` | `MTPCandidateGenerator` |
| 4 | `assistant_model` 类名以 `Gemma4Assistant`/`Gemma4UnifiedAssistant` 开头 | `SinglePositionMultiTokenCandidateGenerator` |
| 5 | `generation_config.speculation_type == "dflash"` | `DFlashTokenCandidateGenerator` |
| 6 | 不同 tokenizer 且 `do_sample=True` | `UniversalSpeculativeDecodingGenerator`（UAG，词表翻译 + **无损**投机采样） |
| 7 | 不同 tokenizer 且 `do_sample=False` | `AssistedCandidateGeneratorDifferentTokenizers`（UAG 贪心） |
| 8 | 其余 | `AssistedCandidateGenerator`（经典独立草稿模型） |

**逐条回答**：

- **`assistant_model=` 独立草稿模型** ✅（路径 8）。文档原文：
  > **assistant_model** (`PreTrainedModel`, *optional*) — An assistant model that can be used to accelerate generation. **The assistant model must have the exact same tokenizer.** The acceleration is achieved when forecasting candidate tokens with the assistant model is much faster than running generation with the model you're calling generate from. As such, the assistant model should be much smaller.
- **Prompt lookup（`prompt_lookup_num_tokens=`）** ✅（路径 2）。类 docstring 指向 https://github.com/apoorvumang/prompt-lookup-decoding
- **UAG（`tokenizer=` + `assistant_tokenizer=`）** ✅（路径 6/7）。校验逻辑逐字（`utils.py:1601`）：
  ```python
  doc_reference = ("(see https://huggingface.co/docs/transformers/en/generation_strategies#universal-assisted-decoding)")
  if self.config.get_text_config().vocab_size == assistant_model.config.get_text_config().vocab_size:
      if "assistant_tokenizer" in generation_mode_kwargs:
          raise ValueError(f"`assistant_tokenizer` is not required when the main and assistant models use the same tokenizer. Please omit `assistant_tokenizer` from `generate()` {doc_reference}.")
  else:
      if "tokenizer" not in generation_mode_kwargs or "assistant_tokenizer" not in generation_mode_kwargs:
          raise ValueError(f"The main and assistant models have different tokenizers. Please provide `tokenizer` and `assistant_tokenizer` to `generate()` {doc_reference}.")
  ```
  ⚠️ **`doc_reference` 指向的 `generation_strategies#universal-assisted-decoding` 锚点在当前 `main` 的文档页上已不存在**（该页只剩 Greedy search / Sampling / Beam search / Custom generation methods 四节，投机解码内容整体被移出）。**这是一个官方文档死链，且死链地址被硬编码进了报错信息里。**
- **Early-exit 自投机（`assistant_early_exit=`）** ✅（路径 1）。文档逐字：
  > **assistant_early_exit** (`int`, *optional*) — If set to a positive integer, early exit of the model will be used as an assistant. **Can only be used with models that support early exit** (i.e. models where logits from intermediate layers can be interpreted by the LM head).
  类 docstring 点名模型：`facebook/layerskip-llama3.2-1B`。实现很朴素——临时改层数：
  ```python
  base_model.config.num_hidden_layers = self.assistant_early_exit
  candidate_ids, candidate_logits = super().get_candidates(input_ids)
  base_model.config.num_hidden_layers = original_num_hidden_layers
  ```
  > **这条对本库 `10-自投机-跳层早退与Draft-and-Verify` 很关键**：HF 是**唯一**内置了早退自投机的主流框架（vLLM/SGLang/llama.cpp/TensorRT-LLM 都没有），而且实现只有三行。
- **EAGLE / Medusa** ❌ **都没有**。对 `candidate_generator.py` + `utils.py` + `configuration_utils.py` 做 `grep -rn -i "eagle\|medusa"`，**命中数 0**。
- **MTP（`use_mtp=True`）** ✅。文档：`Whether or not to use Multi-Token Prediction (MTP) if the model supports it.` 实现依赖 `MtpModel`/`MtpCache`，从主模型 config 读 `num_mtp_layers`，读不到就 raise `"Could not find num_mtp_layers in the model config. This model probably has no associated mtp weights."`
- **DFlash（`speculation_type="dflash"`）** ✅。文档：`The requested speculation type. Accepted values are ['dflash'].` 类 docstring：
  > Dflash reuses the main model's `hidden_states` at several `config.target_layer_ids` to populate the kv cache of the assistant, and then uses the last "bonus" decoded token from the main model, along with `config.block_size` as its main forward to draft `config.block_size` new tokens at every iteration, **similar to a diffusion window**.
- **Gemma 4 专属 SinglePositionMultiToken** ✅。路由处注释：
  > Currently, the only models that can provide this are **Gemma 3n and Gemma 4**, and the only model that can work from it is a Gemma 4 Assistant

### 5.2 `GenerationConfig` 字段名与默认值

> ⚠️ **文档页面上这些字段全部标注为 `*optional*` 且不写 default**，因为实际默认值是运行时由 `_get_default_generation_params()` 注入的，不在 `__init__` 里。**只看文档拿不到默认值，必须读源码。**

**实际默认值**（`configuration_utils.py::_get_default_generation_params()` L636–641）：
```python
"num_assistant_tokens": 20,
"num_assistant_tokens_schedule": "constant",
"assistant_confidence_threshold": 0.4,
"assistant_lookbehind": 10,
"target_lookbehind": 10,
```

| 字段 | **实际默认值** | 依据 |
|---|---|---|
| `num_assistant_tokens` | **20** | `configuration_utils.py:637` |
| `num_assistant_tokens_schedule` | **`"constant"`**（可选 `"heuristic"` / `"heuristic_transient"` / `"constant"`） | `:638` |
| `assistant_confidence_threshold` | **0.4** | `:639` |
| `assistant_lookbehind` | **10** | `:640` |
| `target_lookbehind` | **10** | `:641` |
| `prompt_lookup_num_tokens` | **`None`**（传入即触发 prompt lookup） | `:462` |
| `max_matching_ngram_size` | **`None`**；调用点 fallback 到 2：`max_matching_ngram_size=generation_config.max_matching_ngram_size or 2`（`utils.py:1030`）。`PromptLookupCandidateGenerator.__init__` 签名默认也是 `max_matching_ngram_size: int = 2`、`num_output_tokens: int = 10` | 文档写 "Default to 2 if not provided" |
| `is_assistant` | **`None`** | `:458` |
| `assistant_early_exit` | **`None`** | `:464` |
| `assistant_ensemble_weight` | **`None`**（= lossless） | |
| `use_mtp` | **`None`** | `:403` |
| `speculation_type` | **`None`**（唯一合法值 `"dflash"`） | `:468` |

**schedule 三种取值的文档原文**：
> - `"heuristic"`: When all speculative tokens are correct, **increase `num_assistant_tokens` by 2 else reduce by 1**. `num_assistant_tokens` value is persistent over multiple generation calls with the same assistant model.
> - `"heuristic_transient"`: Same as `"heuristic"` but `num_assistant_tokens` is reset to its initial value after each generation call.
> - `"constant"`: `num_assistant_tokens` stays unchanged during generation

**`assistant_confidence_threshold` 文档原文**（重要，这是"动态草稿长度"的官方实现）：
> The confidence threshold for the assistant model. **If the assistant model's confidence in its prediction for the current token is lower than this threshold, the assistant model stops the current token generation iteration**, even if the number of _speculative tokens_ (defined by `num_assistant_tokens`) is not yet reached. The assistant's confidence threshold is adjusted throughout the speculative iterations to reduce the number of unnecessary draft and target forward passes, **biased towards avoiding false negatives**. ... It is an **unsupervised version of the dynamic speculation lookahead** from *Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models*.

**⭐ `assistant_ensemble_weight` —— 官方提供的"主动放弃无损"开关**（文档原文）：
> Enables static ensemble verification in speculative decoding. If set to a value in `(0.0, 1.0)`, the verifier accepts tokens against the mixture **`w * p_target + (1 - w) * q_draft`** instead of `p_target`, **trading a controlled distributional bias for a higher acceptance rate**. Defaults to `None`, which **keeps decoding lossless**. Requires the assistant model to return logits, so it is not compatible with prompt lookup decoding.

校验（L671）：
```python
if self.assistant_ensemble_weight is not None and not (0.0 < self.assistant_ensemble_weight < 1.0):
    raise ValueError(f"`assistant_ensemble_weight` must be in the open interval `(0.0, 1.0)`, but is {self.assistant_ensemble_weight}. Use `None` for standard (lossless) speculative decoding.")
```
`_speculative_sampling()` docstring 逐字：
> When `assistant_ensemble_weight` is set to a value in (0, 1), applies **static ensemble verification from DIVERSE (https://arxiv.org/abs/2604.07622)**, which relaxes the verification distribution to `v(x) = w * p(x) + (1 - w) * q(x)`, increasing acceptance rate at the cost of controlled distributional bias.

> **这是本库铁律一的最佳正面教材**：HF 把"无损/有损"做成了一个**显式的、有默认值、有文档、有报错信息的开关**，并在文档里直说 `None` 才是 lossless。对比 SGLang 的 `--speculative-accept-threshold-*`（默认 1.0，文档只说 "accept more aggressively"，**不说这会改变分布**）——**同一件事，两家的文档诚实度差距很大。**

**默认值组合的含义**：`schedule="constant"` + `num_assistant_tokens=20` + `confidence_threshold=0.4` —— 这就是 dynamic speculation 博客里的"动态推测"默认模式：**固定上限 20，但草稿模型自信度低于 0.4 就提前收手**。要退回 Leviathan 原始行为，博客给的配方是：
```python
assistant_model.generation_config.num_assistant_tokens_schedule='heuristic'
assistant_model.generation_config.assistant_confidence_threshold=0
assistant_model.generation_config.num_assistant_tokens=5
```

### 5.3 采样与功能限制（全部是源码里的硬 raise）

> 这些约束**不在** `generation_strategies` 文档页（该页已被精简），而在源码的 raise 语句里。

**① batch size 必须为 1（硬性 ValueError）**，`utils.py` L3667–3670：
```python
batch_size, cur_len = input_ids.shape[:2]
if batch_size > 1:
    raise ValueError("assisted generate is only supported for batch_size = 1")
```
→ **截至 transformers 5.15.1，assisted generation 依然只支持 batch_size = 1。** 官方博客 2023 年的说法至今成立：
> At the time of the release of this blog post, **assisted generation is limited to a batch size of 1**.（https://huggingface.co/blog/assisted-generation ）

**② 只支持 greedy search 与 sampling，beam search 静默降级**，`configuration_utils.py::get_generation_mode()` L573–585：
```python
if (assistant_model is not None or self.use_mtp or self.prompt_lookup_num_tokens is not None
        or self.assistant_early_exit is not None):
    if generation_mode in ("greedy_search", "sample"):
        generation_mode = GenerationMode.ASSISTED_GENERATION
    else:
        logger.warning(
            "You've set `assistant_model` or `use_mtp`, which triggers assisted generate. Currently, assisted generate "
            "is only supported with Greedy Search and Sample. However, the base decoding mode (based on "
            f"current flags) is {generation_mode} -- some of the set flags will be ignored.")
```
→ **beam search 不是报错，是 `logger.warning` 后静默降级**——投机相关参数被忽略，退回普通 beam search。**比报错更坑：用户看不到加速也看不到错误。**

**③ `num_return_sequences` 必须为 1（硬性 ValueError）**，`utils.py` L1572–1577：
```python
if generation_mode == GenerationMode.ASSISTED_GENERATION:
    if generation_config.num_return_sequences > 1:
        raise ValueError("num_return_sequences has to be 1 when doing assisted generate, "
                         f"but is {generation_config.num_return_sequences}.")
```

**④ 不支持 stateful 模型（Mamba / RWKV 一类）**，同处 L1578–1583：
```python
if self._is_stateful:
    # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
    # which is not possible with stateful models (they can't reset to a previous subset of generated text)
    raise ValueError(f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}")
```
> 与 llama.cpp 的 `COMMON_CONTEXT_SEQ_RM_TYPE_NO` 是同一堵墙（见 §4.6）：**投机解码要求状态可回滚。**

**⑤ encoder-decoder 结构必须匹配**：
> "The main model and the assistant don't have compatible encoder-dependent input shapes. Ensure you load the assistant with the correct encoder-decoder class, e.g. `AutoModelForSpeechSeq2Seq` for Whisper."

**⑥ 采样正确性：两条不同的路径**，`utils.py::_assisted_decoding` L3740–3771 注释逐字：
```python
# 3. Select the accepted tokens. There are two possible cases:
# Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
# 👉 Apply algorithm 1 from the speculative decoding paper (https://huggingface.co/papers/2211.17192).
if do_sample and candidate_logits is not None:
    valid_tokens, n_matches = _speculative_sampling(...)
# Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
# original model logits with the candidate tokens. We can keep the candidate tokens until the first
# mismatch, or until the max length is reached.
else:
    if do_sample:
        probs = new_logits.softmax(dim=-1)
        selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
    else:
        selected_tokens = new_logits.argmax(dim=-1)
    candidate_new_tokens = candidate_input_ids[:, cur_len:]
    n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
```
- **Case 1**（`assistant_model` + `do_sample=True`）：真正的 Leviathan 拒绝采样，接受率高
- **Case 2**（prompt lookup、或贪心）：退化为"目标采/argmax + 相等即接受"——**与 llama.cpp 的做法完全一样**

> **这段代码是本库 `04-拒绝采样修正` 与 `07-无损的三种口径` 最值得贴的一段**：HF 在同一个函数里并排放了两种验证策略，还在注释里写明了各自的出处（"originally from speculative decoding" vs "originally from assisted generation"）——**它把"投机采样"与"辅助生成"两个历史支流的分野直接刻在了代码注释里。**

### 5.4 与 cache / StaticCache / torch.compile / continuous batching 的交互

三条硬约束，`utils.py::_assisted_decoding` 开头 L3619–3632 逐字：
```python
# The cache must be dynamic for assisted generation, and the check must happen AFTER preparing cache
if not model_kwargs["use_cache"]:
    raise ValueError("assisted generate requires `use_cache=True`")
if (generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]
        or type(model_kwargs.get("past_key_values")) is StaticCache):
    raise ValueError("assisted generate is not supported with Static cache classes`")
# Make sure we can record past on the cache
cache = model_kwargs.get("past_key_values")
if cache is None:
    raise RuntimeError("assisted decoding requires a cache")
cache.activate_past_recording()
```

含义链条：
1. `use_cache=True` 强制
2. `cache_implementation` 不能是 `static` / `hybrid` / `sliding_window`
3. → **`torch.compile` 的 full-graph / CUDA-graph 加速路径与投机解码互斥**（transformers 的自动编译只在可编译 cache（static 系）下触发）
4. cache 必须支持 `activate_past_recording()` / `crop()`，因为拒绝后要裁掉多余的 KV

**continuous batching 也明确不支持**（`utils.py:2420`）：
```python
if assistant_model is not None:
    raise NotImplementedError(f"assistant_model is not supported for continuous batching. Got {assistant_model = }")
```

> ⭐ **这一条决定了 transformers 投机解码的性质**：**它无法与自己的连续批处理后端组合，因此在 transformers 里投机解码本质上是一个"单请求低延迟"特性，不是服务化吞吐特性。** 这与 llama.cpp（投机 + 多 slot 连续批处理可共存）形成对比，更与 vLLM/SGLang（投机就是为服务化做的）形成根本分野。
> **写 `20-主流引擎实现` 时应该这样分层**：HF transformers = 研究/单流参考实现；llama.cpp = 本地单用户 + 有限并发；vLLM/SGLang/TensorRT-LLM = 服务化吞吐。**同一个"投机解码"在三层里是三种不同的工程问题。**

### 5.5 已知 issue 的坑

| # | 标题（逐字） | 状态 | 日期 |
|---|---|---|---|
| **#47932** | Speculative decoding: **candidate generators return a `q` they did not sample from, breaking losslessness with `do_sample=True`** | Closed | 开于 2026-08-12 |
| #47912 | **Assisted/speculative decoding generates past an EOS that is accepted mid-block** | Closed | 2026-08-18 |
| #48039 | Assisted/speculative decoding **ignores `stop_strings` completed mid-block**, and raises with `assistant_model` | Closed | 2026-08-18 |
| #47272 | Avoid repeated Cohere ASR encoder projection during cached generation | Closed | 2026-08-21 |

**#47932 是最值得写进正文的（无损性被打破）**，issue 正文关键点逐字：
> Three candidate generator implementations (**DFlash, MTP, SinglePositionMultiToken**) sample tokens from processor-applied logits but return **raw logits** to `_speculative_sampling`. This creates a distribution mismatch where the acceptance test runs against an incorrect `q`, violating the losslessness guarantee documented in `assistant_ensemble_weight`.
> With `top_k=50` on a 49,152-token vocabulary, the returned `q` has **acceptance-ratio inflation up to 3.02x at p95** and commits tokens with **total variation up to 0.168** from the target distribution.
> "**p/q is computed a factor `1/Z` too large and draft tokens are accepted more often than the algorithm allows.**"

以及一条对做 KM 极有价值的方法论教训：
> Existing tests call `_speculative_sampling` directly with hand-crafted logits, **bypassing the generator-verifier contract where the bug resides.**

> ⭐ **这句话应该直接进 `_lab/` 的设计说明**：本库铁律四要求"数学断言必须有可执行验证"。#47932 说明了**验证写在错误的层级上等于没写**——测 `_speculative_sampling(p, q)` 函数本身永远测不出"传进来的 q 根本不是采样时用的那个 q"。**正确的测试要跨越 generator 与 verifier 的契约边界。**

**归纳的坑**：
1. **新路线（DFlash/MTP/Gemma4-MTP）+ `do_sample=True` 的无损性 bug**（#47932）——与 llama.cpp #25618、SGLang #35771 同族。
2. **块内接受导致越过停止条件**（#47912 EOS、#48039 `stop_strings`）——一次接受多个 token 时，中间位置命中 EOS/stop string 会被漏检，多吐出后续 token。**这是投机解码特有的一类正确性 bug，三家都可能有，值得在 `19-负收益全解` 或 `27-常见误解与判据` 单列。**
3. **beam search 静默降级**（§5.3②）。
4. **`max_matching_ngram_size` 的默认值有两处**（config 里是 `None`，调用点与类签名 fallback 到 2）——读文档容易误以为 config 默认就是 2。

### 5.6 HF 官方博客的加速比数字（按铁律二逐项标注）

#### (a) `huggingface.co/blog/assisted-generation`（Joao Gante，2023-05-11）

**batch size = 1，延迟口径。** 结论段逐字：
> 🤏 Requires access to an assistant model that is **at least an order of magnitude smaller** than your model (the bigger the difference, the better);
> 🚀 **Gets up to 3x speedups in the presence of INT8 and up to 2x otherwise, when the model fits in the GPU memory;**
> 🤯 **If you're playing with models that do not fit in your GPU and are relying on memory offloading, you can see up to 10x speedups;**
> 📄 **Shines in input-grounded tasks, like automatic speech recognition or summarization.**

> Let's have a look at the latency numbers for the greedy decoding case ..., **considering a batch size of 1**.

⚠️ **具体 tps 数字在图片里，正文无文本表格 → 逐点数值未查证。**

温度的影响（逐字）：
> Drawing samples from a probability distribution for the next token will cause our greedy assistant to fail more often, reducing its latency benefits. However, we can control how sharp the probability distribution for the next tokens is, using the temperature coefficient... **Low temperatures are, therefore, more favorable to your assistant model, retaining most of the latency benefits from assisted generation.**

原始启发式（现已非默认）：
> The number of produced candidate tokens is initialized to **5** the first time assisted generation is called.
> ... our original heuristic **increases it by 2 if ALL tokens match and decreases it by 1 otherwise**.

对照的 batching 基准（RTX3090，distilgpt2）：
```
print_tokens_per_second(1)   # Tokens per second: 418.3
print_tokens_per_second(64)  # Tokens per second: 16266.2 (~39x more tokens per second)
```
> **这组对照数字很有教学价值**：batch=64 相对 batch=1 有 ~39× 的吞吐提升。**它是"为什么大 batch 下投机不划算"的另一面：batching 本身已经把算力空转填满了，投机没有剩余算力可用。** 可用于 `03-并行验证为什么几乎免费` 与 `18-batch与吞吐`。

#### (b) `huggingface.co/blog/whisper-speculative-decoding`（Sanchit Gandhi，2023-12-20）

**设置**：float16，`attn_implementation="sdpa"`，73 条样本，**逐条单独 generate（batch size = 1）**，延迟口径（总耗时秒）。**硬件型号原文未给出。**

英文（LibriSpeech validation-clean，73 samples）：

| 配置 | 耗时 | WER |
|---|---|---|
| `openai/whisper-large-v2` 单独 | **72.995 秒** | 0.03507271171941831 |
| + `distil-whisper/distil-large-v2` 助手 | **32.697 秒** | 0.03507271171941831 |
| 加速比 | **2.2×** | **WER 完全相同** |

多语言（VoxPopuli 荷兰语 nl，73 samples），助手换成 `openai/whisper-tiny`：

| 配置 | 耗时 | WER |
|---|---|---|
| large-v2 单独 | 116.510 | 0.127190136275146 |
| + whisper-tiny 助手 | 62.102 | 0.127190136275146 |
| 加速比 | **1.9×** | 相同 |

**⭐⭐ 最关键的 batch size 结论（逐字）——这是全部引擎文档里对"batch 阈值"给得最具体的一段：**
> **It is worth noting that the largest speed gains with speculative decoding come with a batch size of 1. For batched speculative decoding, all candidate tokens across the batch must match the validation tokens in order for the tokens to be accepted. If a token in the batch at a given position does not agree, all candidate tokens that proceed the position are discarded. Consequently, speculative decoding favours lower batch sizes. In practice, we find that speculative decoding provides a speed-up until a batch size of 4. Above batch size 4, speculative decoding returns slower inference than the main model alone.** For full results, refer to Section D.3 of the Distil-Whisper paper.

> ⭐ **这段值得逐字进 `19-负收益全解` 与 `18-batch与吞吐`**，理由有二：
> ① 它给出了一个**具体的可判定阈值（batch > 4 就变慢）**，是所有官方材料里唯一一个具体数字；
> ② 它给出的**机制解释是 HF 自己那套朴素实现特有的**——"batch 内所有序列的候选必须全部匹配才接受，否则整批丢弃"。这是**实现缺陷造成的阈值，不是投机解码的普遍规律**。vLLM/SGLang 用 per-request 的 ragged/padded 验证，不存在"一个序列失配拖累全批"的问题，所以它们的阈值远高于 4。
> **写正文时必须点破这个区别，否则读者会把"batch>4 就变慢"当成投机解码的普遍定律——那是错的。**

助手模型选型判据（逐字）：
> the assistant model should be **at least 3x faster** than the main model (the more the better), while predicting all the "easy" tokens... since **70-80% of all predicted tokens tend to be "easier" tokens**

显存开销：
> In practice, this results in only an **8% increase to VRAM** over using the main model alone.
（因为 Distil-Whisper 复用了 Whisper 的整个 encoder，只加载 32 层 decoder 中的 2 层。）

#### (c) `huggingface.co/blog/dynamic_speculation_lookahead`（Intel Labs + HF，2024-10-08）

**硬件：RTX 4090。贪心解码（temperature = 0）。batch size 未标注（按 assisted generation 的硬限制推断为 1）。**

| Target model | Draft (Assistant) model | Task | Speedup - heuristic | Speedup - dynamic |
|---|---|---|---|---|
| facebook/opt-6.7b | facebook/opt-125m | summarization | 1.82× | **2.71×** |
| facebook/opt-6.7b | facebook/opt-125m | open-ended generation | 1.23× | 1.59× |
| Salesforce/codegen-6B-mono | Salesforce/codegen-350M-mono | code generation (python) | **0.89×（变慢！）** | 1.09× |
| google/flan-t5-xl | google/flan-t5-small | summarization | 1.18× | 1.31× |
| meta-llama/Llama-3.1-8B | meta-llama/Llama-3.2-1B | summarization | **1.00×（无收益）** | 1.52× |
| meta-llama/Llama-3.1-8B | meta-llama/Llama-3.2-1B | open-ended generation | **1.00×** | 1.18× |
| meta-llama/Llama-3.1-8B | meta-llama/Llama-3.2-1B | code generation (python) | 1.09× | 1.15× |

> The results in the table reflect greedy decoding (temperature = 0). Similar trends were observed when using sampling (temperature > 0).

**⭐ heuristic 列里的负收益证据**（博客自己点名）：
> using the dynamic approach with Llama3.2-1B as the assistant for Llama3.1-8B, we observe speedups of up to 1.52x, whereas **the heuristic approach showed no significant speedups with the same setup**. Another observation is that **codegen-6B-mono yields slowdown using the heuristic approach**, whereas the dynamic approach shows speedup.

一个量化的开销对比（MBPP 单例）：
> The static speculation lookahead (blue bars), where the number of generated draft tokens is fixed to 5, performs **38 target forward passes and 192 draft forward passes**, whereas the oracle speculation lookahead performs only **27 target forward passes and 129 draft forward passes**.

基准代码：https://github.com/gante/huggingface-demos/tree/main/experiments/faster_generation

> **这张表是本库 `17-动态草稿长度与自适应停止` 的核心素材**：**同一对模型（Llama-3.1-8B + Llama-3.2-1B）在固定 γ 下是 1.00×（白干），换成动态 γ 后是 1.52×。** 也就是说——**"投机没用"的结论里，有相当一部分其实是"γ 没调"。** 这条必须写。

#### (d) `huggingface.co/blog/universal_assisted_generation`（Intel Labs + HF，2024-10-29）

**硬件（逐字）**：
> Each experiment was conducted on **100 randomly selected examples**. Experiments with **Llama and Mixtral target models use 2 and 4 A100 GPUs**, respectively. **All other experiments ran with a single A6000 GPU.**

**batch size 未标注**；延迟口径（表头写 "latency improvements"）。

| Target model | Assistant model | Dataset | Task | Speedup |
|---|---|---|---|---|
| codellama/CodeLlama-13b-Instruct-hf | bigcode/tiny_starcoder_py | openai/humaneval | code generation | **1.90×** |
| mistralai/Mixtral-8x22B-Instruct-v0.1 | double7/vicuna-68m | cnn_dailymail | summarization | 1.52× |
| google/gemma-2-9b | double7/vicuna-68m | cnn_dailymail | summarization | 1.76× |
| mistralai/Mixtral-8x22B-Instruct-v0.1 | Qwen/Qwen2-0.5B-Instruct | tau/scrolls | long-context summarization | 1.78× |
| meta-llama/Llama-3.1-70B | Qwen/Qwen2-0.5B-Instruct | tau/scrolls | long-context summarization | 1.78× |
| microsoft/Phi-3-medium-128k-instruct | Qwen/Qwen2-0.5B-Instruct | tau/scrolls | long-context summarization | **1.91×** |

> Note that the target models above **do not have small variants (under 1 billion parameters) which are suitable for acceleration using standard assisted generation.**

**草稿模型选型的经验判据（逐字）**：
> Based on our experience, **meaningful speedups are typically seen when the assistant model is at least 50-100 times smaller than the target one.**

**⚠️ 博客里有一条限制现在已经过时**：
> While passing `do_sample=True` with standard assisted generation uses the speculative sampling algorithm (Algorithm 1 from the paper), **UAG currently supports multinomial sampling only.** ... **UAG with `do_sample=True` will have a lower throughput** compared to the case where the assistant has the same tokenizer. In the future, we plan to add support for speculative sampling with UAG.

**这一条在当前 main 已实现**：`_get_candidate_generator` 在 `different_tokenizers and do_sample=True` 时走 `UniversalSpeculativeDecodingGenerator`（配 `AssistantToTargetTranslator` + `_PruneReindexingLMHead`），对应论文 **arXiv:2502.05202**《Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies》。
UAG 词表剪枝的副作用（源码注释逐字）：
> Since we prune the LM head, we cannot use the repetition penalty on the assistant model due to mismatches between token ids and logits index
```python
assistant_model.generation_config.repetition_penalty = None
```

### 5.7 HF 侧「未查证」清单

1. `assisted-generation` 博客的逐点 tps 数字（在图片中，正文无文本表格）。
2. `universal_assisted_generation` 表格的 batch size（博客未标注）。
3. issue 覆盖不完整（本次会话 WebSearch 配额耗尽）。

---

## 5b. llama.cpp × transformers 横向对比速查

| 维度 | llama.cpp (master, b10573) | transformers 5.15.1 |
|---|---|---|
| 独立草稿模型 | ✅ `--spec-type draft-simple` | ✅ `assistant_model=` |
| n-gram / prompt lookup | ✅ **5 种**（`ngram-simple/map-k/map-k4v/mod/cache`） | ✅ 1 种（`prompt_lookup_num_tokens=`） |
| EAGLE-3 | ✅ `draft-eagle3` | ❌ 无 |
| Medusa | ❌ 无 | ❌ 无 |
| MTP | ✅ `draft-mtp` | ✅ `use_mtp=True` |
| DFlash / DSpark | ✅ 两者都有 | ✅ DFlash（`speculation_type="dflash"`）；DSpark ❌ |
| Early-exit 自投机 | ❌ 无 | ✅ `assistant_early_exit=` |
| 跨 tokenizer 草稿 | ❌ 词表必须匹配（±128） | ✅ UAG（`tokenizer=` + `assistant_tokenizer=`），无损版 |
| batch / 并发 | ✅ 多 slot `-np`，与连续批处理共存 | ❌ **硬性 batch_size=1**，且与 continuous batching 互斥 |
| beam search | 不适用（无 beam search） | ❌ **静默降级 + warning** |
| `num_return_sequences>1` | 不适用 | ❌ ValueError |
| grammar / JSON schema | ✅ 完整支持 | ✅ 通过 `logits_processor` 支持 |
| **logprobs** | ❌ **投机路径下不产出**（`// TODO: set result.probs`） | ✅ 支持（`output_scores` / `output_logits`） |
| 无损性口径 | L1（相等即接受，分布精确）；backend sampling 有浮点风险 | Case 1 = Leviathan Alg.1；Case 2 = 相等即接受；`assistant_ensemble_weight` 可**显式**换成 L3 有损 |
| StaticCache / torch.compile | 不适用 | ❌ 明确 raise |
| 官方 benchmark 工具 | ✅ `tools/server/bench/speed-bench`（SPEED-Bench） | ⚠️ 第三方仓库 `gante/huggingface-demos` |

---

## 6. 其它引擎（Arctic Inference / LMDeploy / 昇腾 / TorchSpec / 国产栈 / 云厂商）

> 本节取证方法说明：调研后半程 WebSearch 配额耗尽（200/200），改用 `curl` 直抓 raw 文件 + sitemap 全量枚举。**对"某项目没有该特性"这类否定结论，全量 grep 反而是比搜索引擎召回更硬的证据**（枚举全部 URL / grep 全仓库）。

### 6.0 总览表

| 项目 | 有无投机 | 方法 | 核心参数（默认值） | 维护状态（实测最后 commit） |
|---|---|---|---|---|
| **Arctic Inference** | ✅ vLLM 插件 | `arctic`(MLP/LSTM speculator) + `suffix` | `method`、`num_speculative_tokens`(3)、`enable_suffix_decoding`(false)、**`disable_by_batch_size`(64)** | 2026-06-29；PyPI 0.2.0 **锁 `vllm==0.18.0`** |
| **LMDeploy** | ✅ **仅 PyTorch engine** | eagle / eagle3 / deepseek_mtp / hy3_mtp / qwen3_5_mtp | `--speculative-algorithm`(None)、`--speculative-num-draft-tokens`(**1**) | v0.16.0（2026-08-19），官方标 experimental |
| **MindIE（华为）** | ✅ 叫「并行解码」+ MTP | `la` / `memory_decoding` / `mtp` | `plugin_params`→`plugin_type`、`speculationGamma` | MindIE 2.3.0 / CANN 8.5.0 |
| **vllm-ascend** | ✅ **国产栈里最全** | 10 种（ngram/suffix/medusa/eagle/eagle3/mtp/dflash/dspark/draft_model/extract_hidden_states） | `speculative_config` 同上游 | **v0.23.0（2026-08-16）** |
| **TorchSpec** | ⚠️ **训练**框架 | 训 EAGLE3 / DFlash / DSpark / DFlash2 草稿模型 | YAML config | 2026-08-20，PyTorch 官方博客背书 |
| **gpt-fast** | ✅ demo | 经典 draft model | `--speculate_k`(**5**)、`--draft_checkpoint_path`(None) | **2025-08-22，停更一年** |
| **torchchat** | ✅ 但隐藏 | 同 gpt-fast | `--speculate-k`(**5**)，全部 `argparse.SUPPRESS` | **官方声明 no longer under active development** |
| **LightLLM** | ✅ 自研 MTP | `vanilla`/`eagle` × `with_att`/`no_att` | `--mtp_mode`(None)、`--mtp_step`(**0**) | 活跃 2026-08-21 |
| **KTransformers** | ⚠️ **转交 SGLang** | EAGLE（由 SGLang 实现） | 用 SGLang 的 `--speculative-algorithm` | 活跃；自研版已进 `archive/` |
| **RTP-LLM（阿里）** | ✅ **方法最全** | vanilla/deterministic/mtp/eagle/eagle3/**dspark** | `--sp_type`("")、`--gen_num_per_cycle`(**1**) | **未归档**，活跃 2026-08-20 |
| **NVIDIA Dynamo** | ✅ 纯转发 | vLLM+Eagle3 ✅ / SGLang 🚧 / TRT-LLM 🚧 | 透传 backend 参数 | v1.5.0，活跃 |
| **MLC-LLM** | ✅ 三模式 | `small_draft` / `eagle` / `medusa` | `speculative_mode`("disable")、`spec_draft_length`(**0=自适应**) | 活跃 2026-08-17 |
| **Ollama** | ✅ **有，且会自动开** | `draft-mtp` / `draft-dflash` | `draft_num_predict`(**4**) | v0.33.0（2026-08-22） |
| **DeepSpeed-MII / FastGen** | ❌ **全仓零命中** | — | — | **停更，2025-07-01** |
| **Medusa 官方 repo** | ✅ | Medusa heads | 推理 CLI 无投机专有参数 | **停更，2024-04-18** |
| **EAGLE 官方 repo** | ✅ | EAGLE-1/2/3 | `total_token`(60)、`depth`(7)、`top_k`(10) | 2026-02-19 |

> **三条与常见印象相反的纠正**：
> ① **TorchSpec 真实存在，但它是训练框架不是推理引擎**（详见 6.4，与我在 §1.11 从 vLLM blog 拿到的证据互相印证）；
> ② **llama.cpp 的 `--draft-max` / `--draft-min` 已被删除**（本节独立复核，与 §4.2 结论一致——**两条独立取证路径得到同一结论**）；
> ③ **Ollama 早就支持投机解码，而且权重自带 MTP 头时会自动开启，用户完全无感。**

---

### 6.1 Arctic Inference (Snowflake)

仓库 https://github.com/snowflakedb/ArcticInference · 文档 https://arcticinference.readthedocs.io/en/latest/ · 论文 **arXiv 2507.11830（2025-07）**

#### 安装与版本锁死（最大的落地障碍）
```console
$ pip install arctic-inference[vllm]
$ ARCTIC_INFERENCE_ENABLED=1 vllm serve ...
```
PyPI `arctic-inference` 最新 **0.2.0（2026-06-26）**，`requires_dist` 里 `[vllm]` extra 写死 **`vllm==0.18.0`**（2026-03-20 发布）。**而 vLLM 当前是 0.27.1——想用完整 Arctic 插件要把 vLLM 退回 5 个大版本 / 约 5 个月。**

**但**：只要 suffix decoding 的话，装 base 包即可（base `requires_dist` 为空、零依赖），因为**上游 vLLM 已经把它集成了**——`vllm/v1/spec_decode/suffix_decoding.py` 逐字：
```python
class SuffixDecodingProposer:
    """...This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference)."""
    from arctic_inference.suffix_decoding import SuffixDecodingCache
```
上游化的来龙去脉：Snowflake 2025-05-13 提了 [RFC #18037](https://github.com/vllm-project/vllm/issues/18037)，推荐方案 2.ii（把 arctic-inference 作为 vLLM 依赖），**已做成**。
→ **今天在 vLLM / vllm-ascend 里用 `method: "suffix"`，只需 `pip install arctic-inference`（base），不需要 `[vllm]` extra、不需要降级 vLLM、不需要环境变量。**

#### suffix decoding 与 ngram 的差别（上游 vLLM 官方定义，最权威）
> Like n-gram, Suffix Decoding can generate draft tokens by pattern-matching using the last `n` generated tokens. Unlike n-gram, Suffix Decoding **(1) can pattern-match against both the prompt and previous generations, (2) uses frequency counts to propose the most likely continuations, and (3) speculates an adaptive number of tokens for each request at each iteration** to get better acceptance rates.
> Suffix Decoding can achieve better performance for tasks with high repetition, such as **code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts**.

Arctic 自己的机制描述：维护**两棵后缀树**——Global Suffix Tree（跨请求，存已完成请求的历史输出）+ Per-Request Suffix Tree（当前请求的 prompt + 已生成 token）；按最近 pattern 匹配、按频次挑最可能延续、**按匹配长度动态调投机数**。**纯 CPU、不需要草稿模型，约 20 微秒 / 投机 token。**

#### ⚠️ 三套命名空间，别混

**(A) Arctic 插件 `method: "arctic"`**（`docs/arctic-speculator.rst`）

| Key | 默认 |
|---|---|
| `method` | 必填 = `"arctic"` |
| `model` | 必填（HF ID / 路径） |
| `num_speculative_tokens` | **3** |
| `enable_suffix_decoding` | **false** |

**(B) Arctic 插件的 suffix 参数**（`docs/suffix-decoding.rst`）

| Key | 默认 |
|---|---|
| `enable_suffix_decoding` | **false** |
| `suffix_cache_max_depth` | **64** |
| `suffix_max_spec_factor` | **1.0**（论文的 α） |
| `suffix_max_spec_offset` | **0.0**（负值可抑制过度投机） |
| `suffix_min_token_prob` | **0.1** |
| `suffix_cache_max_requests` | **100000**（超出 FIFO 淘汰） |

公式：`max_speculated_tokens = suffix_max_spec_factor * matched_length + suffix_max_spec_offset`

**(C) 上游 vLLM 原生 `method: "suffix"`**（参数名完全不同！见 §1.4）：`suffix_decoding_max_tree_depth`(**24**)、`suffix_decoding_max_cached_requests`(**10000**)、`suffix_decoding_max_spec_factor`(**1.0**)、`suffix_decoding_min_token_prob`(**0.1**)

> ⚠️ **(B) 与 (C) 是同一个功能的两套 key，默认深度 64 vs 24 也不同——这是最容易踩的坑。**

**(D) ⭐ `disable_by_batch_size` —— 官方唯一的「大 batch 自动关闭」开关，且只在源码里**

README / docs 示例里都带 `"disable_by_batch_size": 64`，但**两个参数参考表都没解释它**。源码 `arctic_inference/vllm/config.py`：
```python
class ArcticSpeculativeConfig(SpeculativeConfig):
    disable_by_batch_size: int | None = None
    hard_disable_by_batch_size: int | None = None   # 文档完全没提
...
if (use_suffix or is_arctic_method) and self.disable_by_batch_size is None:
    logger.info("Defaulting disable_by_batch_size to 64")
    self.disable_by_batch_size = 64
```
语义（源码 docstring 逐字）：
> * `hard_disable_by_batch_size` (HARD): once the running batch reaches this, **no** request drafts (returns 0). SD is effectively off...
> * `disable_by_batch_size` (SOFT): above this, **only the first N requests draft** (caps total draft tokens to N * width).

> ⭐ **即：Arctic 对 suffix / arctic 方法默认在 running batch > 64 时就开始软限流。这本身就是官方"大 batch 别全开"的态度，只是写在代码里没写在文档里。**
> 注意 `disable_by_batch_size` **不是上游 vLLM 的 key**（`grep -c` 在 `vllm/config/speculative.py` = 0）；上游对应物是 `num_speculative_tokens_per_batch_size`。
> **有意思的历史回声**：vLLM V0 也有过 `disable_by_batch_size`（§1.8），V1 删掉了；Arctic 在自己的插件里把它又实现了一遍，还加了 hard/soft 两档。

#### Benchmark

**(a) 2025-05-01 首篇**（https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/ ，vLLM v0.8.4）

| Workload | No Spec | N-gram | EAGLE | Arctic | Suffix | Arctic+Suffix |
|---|---|---|---|---|---|---|
| ShareGPT | 76.0 tok/s | 91.2 | 102 | 172 | 113 | **179** |
| HumanEval | 77.2 | 100 | 112 | 203 | 148 | **217** |
| SWE-Bench | 75.8 | 175 | **Error** | 294 | 286 | **302** |
| Mixed | 82.9 | 112 | **Error** | 184 | 155 | **209** |

硬件 **8×H100**；模型 **Llama-3.1-70B-Instruct**；指标 tok/s；**batch size / QPS 原文未给出**。
接受率：LSTM-Speculator 1.8B 在 ShareGPT **44.5%**；MLP-Speculator 2.1B Arctic 训练版 **42.7%**（原版仅 **13.7%**）。
SWE-Bench（CodeAct Agent，Llama-3.3-70B-Instruct，8×H100）：Suffix Decoding **decode 2.3×–6.3×**，端到端任务完成 **1.8×–4.5×**。
EAGLE 两栏 Error 的原因（原文）：
> **EAGLE could not be run on SWE-Bench and Mixed because its draft models only supported 2K sequence length**
> → 与 §1.10 里 vLLM 的 #48894（EAGLE3 draft `max_position_embeddings` 只有 2048）是同一个根因。**"EAGLE 草稿的上下文长度限制"是一个跨引擎的真实工程障碍。**

**(b) 2025-08-25 GPT-OSS**：配置逐字 `{"method":"arctic","model":"Snowflake/Arctic-LSTM-Speculator-gpt-oss-120B","num_speculative_tokens":3,"disable_by_batch_size":64}`

| 模型 | 数据集 | Baseline | With Spec | 加速 |
|---|---|---|---|---|
| gpt-oss-120B | ShareGPT | 220.2 tok/s | 377.3 | 1.7× |
| gpt-oss-120B | HumanEval | 220.7 | 400.0 | 1.8× |
| gpt-oss-20B | ShareGPT | 298.5 | 476.2 | 1.6× |
| gpt-oss-20B | HumanEval | 301.2 | 490.2 | 1.6× |

推理 TP=4；训练用 16×H200；接受率 **44%–50%**；**batch size 原文未给出**。

**(c) ⭐ 2025-12-02 生产规模博客——承认了负收益**（https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/ ）
> "At high concurrency levels (e.g., 64 requests), **speculation overhead began to dominate runtime**."
> "While the performance of SuffixDecoding was better than N-gram… at low concurrencies, **it actually became worse at a concurrency of 64**."
> "With the optimizations we discussed in this blog, we measured that the remaining overhead due to SuffixDecoding speculation and updates to be around **10% at a concurrency of 64**."
> "SuffixDecoding consistently outperforms both N-gram configurations across all concurrency levels with a **single, fixed configuration**"（N-gram 则需要换参数：并发 1 时 N-gram[5,5] 好，并发 64 时 N-gram[3,5] 好）

数据：SpecBench 并发 1 = 5.21 ms TPOT，并发 64 ≈ 11–12 ms TPOT；BlazeEdit 端到端 **1.96×–3.12×**。工程优化：自研 HashMap 3.4× 投机加速 / 1.5× 更新加速 / 2.3× 内存下降；两级链表 2.2× top-child 选择加速。**硬件规格该文未披露。**

> ⭐ **"无模型草稿也有 overhead" 是本条最有价值的信息**：suffix decoding 不跑草稿模型，但**维护后缀树本身在并发 64 时就吃掉 10% 的时间**（优化前更差，一度不如 n-gram）。
> **这推翻了一个常见直觉："n-gram/suffix 是免费的，不会有负收益"。** 应写进 `11-无模型草稿` 与 `19-负收益全解`。

#### 适用/不适用（Arctic 自己的表述）
> optimized to accelerate **repetitive and predictable inference workloads typical in agentic tasks, self-refinement loops, and multi-agent pipelines**

Arctic 没有一句"不要开"，但态度体现在三处：① `disable_by_batch_size` 默认 64 的软限流；② 并发 64 时曾经比 N-gram 差、优化后仍有 10% overhead；③ 混合模式的存在（suffix 对开放式对话弱，所以要配 LSTM speculator 兜底）。

#### 边界澄清：SwiftKV **不是**投机解码
SwiftKV 是 prefill 计算削减（SingleInputKV，后层复用前层 KV），Llama 3.1 70B 上 prefill 计算最多降 50%。**无配置开关**——加载 `Snowflake/*-SwiftKV-*` 微调权重即自动生效。

---

### 6.2 LMDeploy（InternLM / 上海 AI Lab）

**结论：TurboMind 引擎不支持，且是静默忽略；投机解码只在 PyTorch engine 上，v0.11.0（2025-12-04）首发，官方至今标 experimental。**

#### TurboMind 不支持的硬证据
`lmdeploy/serve/core/async_engine.py`：
```python
if speculative_config is not None and backend == 'turbomind':
    logger.warning('speculative decoding is not supported by turbomind ')
```
> ⚠️ **注意是 `warning` 不是 `raise`——传了不报错，只是被静默丢弃。这是最容易踩的坑。**
紧接着 `self.num_spec_token = 0 if backend == 'turbomind' ...`；`_build_turbomind()` 签名根本不传 `speculative_config`。全部投机代码在 `lmdeploy/pytorch/spec_decode/`，`src/turbomind/` 下**零个**投机文件。

官方文档：
> This feature is supported for spec methods that inherit from `DeepseekMTP`, including `deepseek_mtp`, `qwen3_5_mtp`, and `eagle3`. **Only the PyTorch backend is supported.**

#### 参数与默认值（`lmdeploy/cli/utils.py::add_spec_group()`）

| flag | 类型 | **默认值** | choices |
|---|---|---|---|
| `--speculative-algorithm` | str | **`None`**（= 关闭） | `eagle`, `eagle3`, `deepseek_mtp`, `hy3_mtp`, `qwen3_5_mtp` |
| `--speculative-draft-model` | str | **`None`** | 任意 HF id / 本地路径 |
| `--speculative-num-draft-tokens` | int | **`1`** | 原文未给出上限 |

Python API `SpeculativeConfig`（`lmdeploy/messages.py`）：`method: str`（必填）、`model: str = ''`、`num_speculative_tokens: int = 1`。

> ⚠️ **默认值陷阱：`num_speculative_tokens` 默认是 1，而官方文档所有示例都写 3。不显式传等于几乎没开。**

#### 草稿模型来源
- **MTP 系**（`deepseek_mtp` / `qwen3_5_mtp` / `hy3_mtp`）：权重内置在主模型 checkpoint，**不传** `--speculative-draft-model`
- **EAGLE 系**：外挂 `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`。**这是文档 + CI 里唯一点名的外部 draft model**
- HF 架构映射只有 `EagleLlamaForCausalLM` / `Eagle3LlamaForCausalLM` → **EAGLE 系只有 Llama 一个架构实现**

#### 限制与冲突（逐字）

| 冲突项 | 行为 |
|---|---|
| TurboMind backend | `logger.warning(...)` **静默忽略** |
| MemDecode | `raise ValueError('MemDecode and speculative decoding cannot be enabled together.')` |
| `async_get_ppl` | `raise ValueError('async_get_ppl is not supported when speculative decoding is on.')` |
| KV cache TurboQuant + FA3 | `raise NotImplementedError('quant_policy=QuantPolicy.TURBO_QUANT is not supported with FA3 speculative decoding (max_q_seqlen > 1)...')` |
| draft 模型并行度 | `# TODO support tp > 1, ep > 1 for other methods` —— 只有 `qwen3_5_mtp`/`hy3_mtp`/(`deepseek_mtp`+glm_moe_dsa) 跟随主模型 TP/EP，**其余强制 TP=1** |
| 编译门槛 | Eagle3 需自编译 **flash-attention 3 (hopper)**；Deepseek MTP 需自编译 **FlashMLA**。均 Hopper+ 专用，**无 pip wheel** |

**已知未修 Bug**：[Issue #4883](https://github.com/InternLM/lmdeploy/issues/4883)（2026-08-19 提，至今 open）——H200 上 Qwen3.5-VL MoE 35B-A3B + `qwen3_5_mtp` + `num_speculative_tokens=2`，agentic RL rollout 跑约 3.5 小时后 **8 个实例同时静默 hang**（进程存活、无日志、无报错）；**关掉投机的对照组连跑 5 小时以上正常**。

#### Benchmark 与"什么时候不该开"
**LMDeploy 官方从未发布任何投机解码性能数字。** `spec_decoding.md` 零个数字，全部 release notes 零个数字，PR #3945 无 benchmark 表。
**官方文档中也不存在任何"什么时候不该开"的表述**（grep `not recommend|should not|avoid|degrade|slower|overhead` 中英文档均空）。唯一态度是那句 `This is an experimental feature in lmdeploy.`

官方给的是尺子不是刻度——Prometheus 指标：`lmdeploy:spec_decode_mean_accept_rate`、`lmdeploy:spec_decode_mean_accept_length`（**含 bonus token**）、`lmdeploy:spec_decode_per_position_accept_rate{position=N}`。
> **第五家引擎的指标口径也是"含 bonus token"**——与 vLLM / llama.cpp / TensorRT-LLM 一致。**这条口径可以当成定论写进 `05-接受率alpha`。**

---

### 6.3 昇腾侧

#### 6.3.1 MindIE / MindIE-LLM（华为）

**结论：有，但不叫「投机推理」。官方术语是「并行解码」(Parallel Decoding) 和「MTP」，是两套互斥的插件。MindIE 从未实现 Medusa / EAGLE / 独立小草稿模型投机——这一点有一手源码证据。**

**MindIE-LLM 已于 2025-12 开源到 GitHub**：https://github.com/Ascend/MindIE-LLM （master = MindIE 2.3.0，配套 CANN 8.5.0）。
⚠️ Gitee 上的 `ascend/MindIE-LLM` 返回 **403 "You hasn't joined this enterprise!"**，走 GitHub。

**源码级白名单**（`mindie_llm/text_generator/plugins/plugin_utils.py`，逐字）：
```python
PLUGIN_WHITE_LIST = ["la", "memory_decoding", "mtp", "prefix_cache"]
SPECULATIVE_PLUGIN_LIST = ["la", "memory_decoding", "mtp"]
ASYNC_INFERENCE_UNSUPPORTED_OPTIONS = ["memory_decoding", "la"]
```
> **→ 没有 medusa、没有 eagle、没有 draft model。「MindIE 支持 Medusa/EAGLE」的说法无任何依据。**

**参数**（配置文件 `{MindIE安装目录}/latest/mindie-service/conf/config.json`）

**lookahead**（`ModelDeployConfig.ModelConfig.plugin_params`）：

| 配置项 | 范围 | 默认 |
|---|---|---|
| `plugin_type` | `la` | — |
| `level` (N) | [3, 16] | **4** |
| `window` (W) | [1, 16] | **5** |
| `guess_set_size` (G) | [1, 16] | **5** |
| `speculationGamma`（ModelDeployConfig 层） | ≥ (N−1)×(W+G)，建议取等 | **官方未给默认值** |

官方样例：`"speculationGamma": 30` + `"plugin_params":"{\"plugin_type\":\"la\",\"level\":4,\"window\":5,\"guess_set_size\":5}"`（30 = (4−1)×(5+5)，自洽）

**memory_decoding**：`decoding_length` [1,16] 默认 **16**；`dynamic_algo` 默认 **false**

**MTP**（`docs/zh/user_guide/feature/mtp.md` 逐字）：
> `plugin_params` | std::string | `plugin_type: mtp`、`num_speculative_tokens: [1]` | **num_speculative_tokens 表示 MTP 的层数，可设置为 1 或 2**…【注】**对于低时延场景，可配置使用 1 或 2，对于高吞吐场景，建议配置不超过 1**

⚠️ 文档表格"取值范围"列写 `[1]`、说明列写"可设置为 1 或 2"——**官方文档自相矛盾**。

**源码里有文档没写的硬约束**（`plugin_utils.py`）：
```python
def validation_func_mtp(data, speculation_gamma):
    num_speculative_tokens = data.get("num_speculative_tokens")
    max_reserved_len = max(num_speculative_tokens, num_speculative_tokens * 2 - 2)
    flag1 = max_reserved_len <= speculation_gamma
    flag2 = (num_speculative_tokens <= 5) and (num_speculative_tokens >= 0)
```
→ **代码上限是 5，文档只写 1~2**；`max(n, 2n−2) ≤ speculationGamma` 这条文档完全没写。三个投机参数在代码里都是 `required_fields`，**缺字段直接 `NotImplementedError`——文档说的"默认值"其实是推荐值，不能省略**。`plugin_type` 支持逗号分隔多插件（如 `"mtp,prefix_cache"`），`plugin_params` 字符串上限 1024 字符。

**支持范围**
- **并行解码**：Atlas 800I A2 / Atlas 300I Duo；LLaMA3 系列、Qwen2/Qwen2.5 系列、Qwen3-14B、Qwen3-32B；**仅 W8A8 与稀疏量化**
- **MTP**：Atlas 800I A2 / Atlas 800I A3；**仅 DeepSeek-R1 / DeepSeek-V3 的 W8A8 / KV Cache int8 量化模型**（另支持 W4A8）

**⭐「什么时候才有收益」——中文侧写得最系统的一段**（并行解码文档逐字）：
> 为了发挥并行解码的优势，需满足如下前提：
> 1. **当前的并发数不高，属于内存带宽受限、计算资源有冗余的情况。**
> 2. **有较长的输入作为猜测token的初步来源。**
> 3. 并行解码主要通过减少推理步数获取增益，因此**需要一定长度的输出才有性能提升效果**。

> 因为通过验证token的比率会直接影响到并行解码的收益，因此**贪婪场景更能充分发挥并行解码的效果，而采样或惩罚类操作会影响并行解码的收益空间**。

> 但是，由于开启并行解码会使用Prompt输入维护前缀树和草稿token map，所以**会对首token时延有一定影响**。

> ⭐ **这三条前提是本次调研里唯一一处把"输入要长、输出也要长"同时写出来的官方表述**，正好补上 Azure 那条"长输入短输出收益有限"的另一半。
> 第三段"维护前缀树会影响首 token 时延"也很重要——**无模型草稿的开销不只在 decode，还在 prefill**，与 Arctic 那条 10% overhead 互为佐证。

**限制与互斥**（并行解码，9 条中最致命的三条）：
> - 该特性**不能和 PD分离、Multi-LoRA、SplitFuse、长序列、MTP、异步调度以及多机推理**特性同时使用。
> - **并行解码场景暂不支持流式推理。**
> - **并行解码场景暂不支持开启健康检查HealthCheck。**
> - lookahead 和 memory_decoding 算法不可同时使能。

MTP 侧：可叠加 prefix cache / KV cache 池化 / 异步调度 / kv_cache_int8 / function call / 思考解析 / **PD 分离**；不可与并行解码、Multi-LoRA、SplitFuse 同时用；PD 混部叠加 CP/SP 时仅支持 `num_speculative_tokens=1`。

**Benchmark**：**官方文档零个数字**。以下**全部二手、未在官方文档核实**（量子位转述华为技术报告 https://www.qbitai.com/2025/05/284713.html ）：CloudMatrix 384 超节点 DeepSeek V3/R1 decode 单卡 1920 tok/s（TPOT 50ms 约束）；Atlas 800I A2 decode 808 tok/s（TPOT 100ms）。
⚠️ **关键：文中称接受率「默认按 70% 折算」——这是折算假设不是实测；decode batch size 原文未给出。** 华为侧的投机引擎在该报告里叫 **FusionSpec**。

> ⚠️ **口径差异警告**：MindIE 的 `num_speculative_tokens` 官方定义是**「MTP 的层数」**，vllm-ascend 的同名参数是「投机 token 数」——**不是一个东西，迁移时别抄数。**

#### 6.3.2 ⭐ vllm-ascend（国产栈里做得最全的）

**版本 v0.23.0（2026-08-16）**，对齐上游 vLLM v0.23.0。
文档 https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html

**支持矩阵定位**（`supported_features.md` 逐字）：`| Speculative decoding | 🟢 Functional | Basic support |`（🟢 Functional = "Fully operational, with ongoing optimizations."）

**支持 10 种方法**（逐字表格）：

| Method | Description |
|---|---|
| `ngram` | Match n-grams from the prompt |
| `suffix` | Suffix-based pattern matching (**requires Arctic Inference**) |
| `medusa` | Medusa heads embedded in the target model |
| `eagle` | EAGLE-based draft model |
| `eagle3` | EAGLE-3 based draft model |
| `mtp` | Multi-Token Prediction with shared embedding head |
| `dflash` | Block diffusion-based parallel draft model |
| `dspark` | Semi-autoregressive block drafting with a sequential Markov logit-bias head |
| `draft_model` | Generic external draft LLM |
| `extract_hidden_states` | Extract hidden states for EAGLE training |

**⭐ 昇腾专属硬约束**（逐字）：
> On Ascend NPUs, the `npu_fused_infer_attention_score` operator supports a maximum of **16 tokens per decode round**. Therefore, **`(num_speculative_tokens + 1)` must be ≤ 16**.

**PD 分离额外约束**（逐字）：
> 1. Hybrid Mamba models (e.g., Qwen-Next and Qwen3.5 series): `num_speculative_tokens` should be **equal on P nodes and D nodes**.
> 2. Other models: `num_speculative_tokens` on P nodes should be **1**, and on D nodes should be ≥ 1.

> ⭐ **这是全库第三份「PD 分离 × 投机」的官方素材**，而且给出了**最具体的规则**（P 节点 K=1、D 节点 K≥1；Mamba 类必须两侧相等）。加上 SGLang（两侧都配、都关 radix cache）与 TensorRT-LLM（两侧都配、关 block reuse），`23-与其它优化的相互作用` 的 PD 分离一节现在有三家可以互相对照。

**真实配置示例**（逐字）
MTP（DeepSeek-V3.2-Exp-W8A8，TP=16，EP）：
```bash
vllm serve /deepseek-ai/DeepSeek-V3.2-Exp-W8A8 \
  --tensor-parallel-size 16 --enable-expert-parallel \
  --max-model-len 36768 --max-num-seqs 10 --quantization ascend \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 2, "method":"mtp", "disable_padded_drafter_batch": false}'
```
> **NOTE**: Due to the fact that only a single layer of weights is exposed in DeepSeek's MTP, **accuracy and performance are not effectively guaranteed in scenarios where `num_speculative_tokens > 1` (especially ≥ 3)**.
> In the fullgraph mode with `num_speculative_tokens > 1`, the capture size of each ACLGraph must be **an integer multiple of `(num_speculative_tokens + 1)`**.

EAGLE（注意 `--enforce-eager`）：
```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 4 --enforce-eager \
  --speculative-config '{"method": "eagle", "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 2}'
```
> After enabling EAGLE, the main model needs to verify `(1 + K)` tokens... the fullgraph mode will fix the number of tokens during the verification stage, so **`cudagraph_capture_sizes` must be a list of capture sizes, where each size is calculated as `n * (K + 1)`**. For instance, to support batch sizes from 1 to 4 with `num_speculative_tokens = 4`, `cudagraph_capture_sizes` should be `[5, 10, 15, 20]`.

Suffix：`--speculative-config '{"method": "suffix", "num_speculative_tokens": 15}'`，需 `pip install arctic-inference`。

**⭐ Dynamic SD——昇腾侧也明写了「什么时候投机会变负收益」**（逐字）：
> SD methods need to verify K tokens for each sequence during decoding. As batch size (BS) increases, **the effective batch becomes `BS * K`**, which increases the compute requirement during verification. **When `BS * K` goes beyond a critical batch size, speculative decoding can hurt decode speed (TPOT).**

`num_speculative_tokens_per_batch_size` 用法同上游（`[[1,64,3],[65,128,1],[129,512,0]]`，最后一档 **K=0 即不产草稿**）。
支持 MTP / EAGLE-3 / DFlash / n-gram；**不支持 Suffix**，原因逐字：
> Suffix Decoding is already per-request dynamic, so **an outer dynamic K is redundant or conflicting**; this is an upstream constraint, not Ascend-specific. DSpark and `MTP + DCP + DSD` are currently not supported on this path.

**⭐ 昇腾独有的两个"精度换吞吐"开关**
```bash
--additional-config '{"rejection_sampler_config":{"enable_block_verify":true,"enable_entropy_verify":true,"posterior_threshold":0.95,"posterior_alpha":0.4}}'
```
- `posterior_threshold` **默认 0.95**，范围 (0,1]；`posterior_alpha` **默认 0.4**，范围 ≥0
> **WARNING**: Both Block Verify and Entropy Verify **modify the token acceptance criteria and may cause minor precision degradation**… Evaluate the quality impact on your specific workload before enabling them in production.

> ⭐ **这是 §7c 附表（L3 有损开关）该补的第六家**，而且**文档诚实度很高**——直接用 WARNING 写明"会改变接受判据、可能降精度"。
> 注意 `posterior_threshold` / `posterior_alpha` 这两个名字正是 **vLLM V0 已删的 typical acceptance 参数**（V0 默认 0.09 / 0.3）。**昇腾把它们以 `additional_config` 的形式又实现了一遍，且默认值完全不同（0.95 / 0.4）。**

Synthetic 采样（**仅供 benchmark**）：`rejection_sample_method`（**默认 `"standard"`**）、`synthetic_acceptance_rates`、`synthetic_acceptance_length`：
> **WARNING**: Synthetic mode accepts draft tokens regardless of whether they match the target distribution, so the generated output is **not semantically correct**. **Use it for benchmarking and validation only, never for production serving.**

Confidence-based 动态验证（DSpark/DFlash，`additional_config.dynamic_spec_config.method_params`）：`initial_verify_budget_per_req`（**5**）、`budget_update_interval`（**16**）、`budget_threshold`（**0.3**）、`min_verify_tokens`（**1**）。

**⭐⭐ 关于「V0 支持 / V1 不支持」与 torchair 冲突——两个问题的答案都是"已经过时了"**

**V0/V1**：限制存在过，但早已消失。release notes 逐字时间线：
- v0.8.4rc1：> "Spec decode feature works now. **Currently it only works on V0 engine.** V1 engine support will come soon. [#500]"
- v0.9.0rc1：> "**Spec decode and MTP features work with V1 Engine now.** [#874] [#890]"
- v0.9.2rc1：> "this release is the last version to support V0 engine"
- v0.10.0rc1：> "**V0 is completely removed from this version.**"

今天 vllm-ascend 内部的 "V1/V2" 指的是 **Model Runner v1/v2 (MRv1/MRv2)**，不是 vLLM 引擎 V0/V1——**别搞混**（与 §1.6 里 vLLM 上游的 V2 model runner 是同一件事）。MRv2 的投机支持仍在补（`Expanded Model Runner V2 with initial MoE and Eagle support #7885 #7922`、`Fix mrv2 runtime error with speculative decoding #8209`）。

**torchair**：**torchair graph mode 已被整体删除，"与投机冲突"这个问题不存在了。**
- v0.11.0rc1（2025-11-10）：> "Torchair is deprecated. We'll remove it once the performance of ACL Graph is good enough. **The deadline is Q1 2026.**"
- v0.12.0rc1（2025-12-13）：> "**Torchair graph mode is removed.** `--additional-config {"torchair_graph_config": {"enabled": true}}` doesn't work anymore. Please use aclgraph instead."
- v0.13.0（2026-02-06）：> "**Torchair** has been dropped. [#4814]"

历史上确实有过 MTP 适配 torchair 的一堆 PR（#2145「MTP support torchair graph mode now」、#1294、#1840、#2610），**但今天全部无意义**。现在的图模式是 **ACLGraph（+Npugraph_ex）**，v0.23.0 起默认 `FULL_AND_PIECEWISE` 且无需手配；投机与图模式的真实约束换成了上面那两条 capture size 必须是 `(K+1)` 整数倍。

**⭐⭐ Benchmark（Suffix Decoding）——国产栈里唯一一份口径完整的公开数据**

来源 `docs/source/tutorials/features/suffix_speculative_decoding.md`（v0.23.0）
配置：单台 **Atlas 800T A2，4 卡 TP=4**，模型 **Qwen3-32B**，`--speculative-config '{"method":"suffix","num_speculative_tokens":3}'`，`--max-model-len 5500`，工具 AISBench，**指标 TPOT(ms) + 吞吐(TPS)**，SLO 约束 TPOT < 50ms。

| 数据集 | 并发 | Avg In | Avg Out | Base TPOT | Base TPS | Suffix TPOT | Suffix TPS | **Accept Rate** | **TPS Gain** |
|---|---|---|---|---|---|---|---|---|---|
| HumanEval | 1 | 150 | 2700 | 55.1 | 18.1 | 37.9 | 26.3 | 27.0% | **45.1%** |
| HumanEval | 26 | 150 | 2700 | 64.7 | 403.8 | 50.9 | 519.2 | 27.0% | 28.6% |
| ARC | 1 | 76 | 960 | 52.8 | 18.9 | 39.5 | 25.4 | 23.9% | 34.6% |
| ARC | 15 | 76 | 960 | 59.8 | 245.8 | 48.9 | 311.7 | 23.9% | 26.8% |
| GSM8K | 1 | 67 | 1570 | 55.5 | 18.0 | 35.7 | 28.5 | 31.1% | **58.4%** |
| GSM8K | 26 | 67 | 1570 | 63.9 | 396.4 | 50.0 | 527.6 | 31.1% | 33.1% |
| ShareGPT | 1 | **666** | **231** | 54.1 | 18.3 | 39.2 | 24.1 | 23.9% | 31.5% |
| ShareGPT | 14 | **666** | **231** | 61.8 | 227.0 | 49.9 | 273.9 | 23.9% | **20.7%** |
| SuperGLUE_BoolQ | 32 | 207 | 314 | 62.7 | 396.4 | 47.8 | 507.5 | 33.4% | 28.0% |
| **AGIEval** | 1 | 735 | 1880 | 53.1 | 18.7 | 31.8 | **34.1** | **50.3%** | **81.9%** |
| AGIEval | 34 | 735 | 1880 | 70.0 | 494.6 | 50.2 | 768.4 | 50.3% | 55.3% |

官方归纳：`| High Gain | AGIEval, GSM8K | > 50% |` / `| Medium-Low Gain | ARC, ShareGPT | 20% ~ 30% |`

> ⭐⭐ **这张表是本次调研中口径最完整的一份**（数据集 / 并发 / 输入输出长度 / 基线与投机的 TPOT 和 TPS / 接受率 / 增益全有），三个读法：
> ① **收益随并发单调下降**（HumanEval 45.1%→28.6%，AGIEval 81.9%→55.3%，GSM8K 58.4%→33.1%）；
> ② **接受率是收益的第一解释变量**（AGIEval 50.3% 对应最高增益，ARC/ShareGPT 23.9% 对应最低）；
> ③ ⭐ **ShareGPT 那两行直接量化了"长输入短输出"的劣势**：Avg In 666 / Avg Out 231，是全表 TPS Gain 最低的一组（并发 14 时仅 20.7%）。对比 AGIEval（In 735 / Out 1880，输入同样长但输出长 8 倍）的 55.3%——**输入长度相近、只有输出长度差 8 倍，增益差 2.7 倍。这是"输出长度决定投机收益"的最干净证据。**
> **建议把这张表和 §7.2 成因 A 的 TRT-LLM ISL/OSL 拐点表放在一起，作为 `22-长上下文` 与 `19-负收益全解` 的双证据。**

**版本演进摘要**

| 版本 | 日期 | 投机相关 |
|---|---|---|
| v0.8.4rc1 | 2025 | spec decode 首次可用，**仅 V0** |
| v0.9.0rc1 | 2025 | spec decode + MTP 支持 V1 |
| v0.17.0rc1 | 2026-03-15 | 统一并行投机解码 #6766 |
| v0.18.0rc1 | 2026-04-01 | target/draft 可用不同 attention backend #7342 |
| v0.18.0 | 2026-04-30 | Suffix Speculative Decoding benchmark 教程 #6323；eagle3/mtp 合并同一 proposer #6349 |
| v0.19.1rc1 | 2026-04-30 | **Zero Bubble 异步调度 + 投机解码** #7640 |
| v0.22.1rc1 | 2026-06-30 | **P-Eagle 和 PARD 成为稳定的并行投机方法**（仅 release note 提及，`speculative_decoding.md` 方法表里没有） |
| **v0.23.0** | **2026-08-16** | `FULL_AND_PIECEWISE` 默认图模式；DFlash `FULL_DECODE_ONLY`；`--async-scheduling` + 投机；D2D NetLoader 加载 draft 权重 #9893；DeepSeek V4 + Ascend 950 全链路含 MTP |

**已知问题（v0.23.0 release notes 逐字）**：
> - Qwen3.6-35B-A3B **may shut down when MTP/speculative decoding is enabled**, with `numAcceptedTokens[0]=4 exceeds varlen segment length=3` reported… [#9956]
> - GLM5 W4A8 deployments can see a **significantly lower speculative decoding acceptance rate when MTP3 is used together with FlashComm**. [#9803]
> - MiniMax-M2.7 W8A8/QuaRot can show lower-than-expected GPQA accuracy in long-sequence deployments **when PCP/DCP is combined with Eagle3 speculative decoding**. [#9959]
> - Qwen3.x with PD disaggregation plus MTP **can still show precision issues because former KVCache blocks may remain dirty**. [#10961]

> **最后两条又是 §7b 那族"量化/并行组合下精度退化"的新例证**，且第四条点出了机理方向：**PD 分离下 KV cache block 可能是脏的**。

---

### 6.4 PyTorch 生态 / TorchSpec

#### ⭐ TorchSpec 真实存在，但它是**训练**框架不是推理框架

- **仓库**：https://github.com/lightseekorg/TorchSpec （229 stars，最后 commit **2026-08-20**）。原组织 `torchspec-project/TorchSpec` **301 重定向**到此。
- **PyTorch 官方博客背书**：https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/ （**2026-03-19**）
- ⚠️ **PyPI 上的 `torchspec` 是个占位包**：version 0.0.1，summary 逐字 `"TorchSpec (placeholder package name reservation)."`，homepage 还是模板占位 `https://github.com/your-org/torchspec`。**别 pip install 它。**

README 逐字定义：
> TorchSpec is a **torch-native speculative decoding training framework**. We introduce a **disaggregated** way of training speculative decoding draft models where inference and training are fully decoupled and **stream hidden states directly from inference engine groups to distributed training workers via Mooncake store**, allowing each side to scale independently.

**推理引擎支持**（作为 hidden state 生产者）：vLLM（First-class, Available）、TokenSpeed（First-class, In progress）、TensorRT-LLM（First-class, Available）、SGLang（社区尽力, Available）、HF Transformers（社区尽力, Available）。
**能训**：EAGLE3（主力）、DFlash、DSpark、DFlash2。仓库里有 `torchspec/config/dspark_draft_config_qwen36_35b.json`、`tools/convert_to_torchspec.py`、`tools/convert_to_hf.py`。

**生产采用**（README 逐字）：DigitalOcean（MiniMax-M2.5 的 EAGLE3）、**vLLM 官方**（Artificial Analysis 榜单用的定制 EAGLE3）、CoreWeave（Kimi K2.7 Code 的 DFlash，并把 D-PACE 贡献回上游）、fal（Ideogram V4 prompt expander 的 DSpark，报 16× 吞吐）、**腾讯混元（AngelSpec，arXiv 2607.25852（2026-07））**。

**⭐ Benchmark（PyTorch 博客，难得带 batch 分档）**：
> With TorchSpec, we successfully trained a Kimi K2.5 EAGLE-3 draft model with **1500 H200 GPU hours**, scaling to 600k training samples, 6 billion tokens.
> With the draft model trained, **output throughput improves by over +60% at batch size 1, +30% at batch size 8, and +26% at batch size 16 under a lookahead of 3 tokens.**

> ⭐ **官方自己的数据就展示了收益随 batch 衰减：+60% (bs=1) → +30% (bs=8) → +26% (bs=16)**，且**只变 batch、其余口径一致，可直接横比**——与 vLLM P-EAGLE 的 c=1 vs c=64 是同类证据。

长上下文能力：
> With a lookahead of 4 and disaggregated training, a single H100 GPU can train on input sequences up to **44K tokens**, and a single B200 GPU can scale to **200K tokens**.

存储动机数字：Kimi K2.5 单条 128K token 样本需 **~7.0 GB** hidden states（3 层 aux 5.25GB + last 1.75GB）→ **10k 样本 70TB / 100k 样本 700TB**。
> **这个数字解释了为什么四个训练框架都在做 online/streaming 训练**（SpecForge 的 disaggregated、TorchSpec 的 Mooncake 流式、ModelOpt 的 NIXL RDMA、Speculators 的 online training）——**offline dump hidden states 在长上下文下磁盘成本不可承受。** 应写进 `21-草稿模型怎么训`。

#### gpt-fast
仓库 https://github.com/meta-pytorch/gpt-fast （`pytorch-labs/gpt-fast` 301 重定向到此）。**⚠️ 最后 commit 2025-08-22——恰好停更整一年。**

| flag | **默认** |
|---|---|
| `--speculate_k` | **5** |
| `--draft_checkpoint_path` | **None**（不传即不开） |
| `--batch_size` | **1** |

（函数签名 `generate()` 内另有 `speculate_k: Optional[int] = 8`，CLI 覆盖为 5。）

**Benchmark（README 逐字）**：
> Benchmarks run on an **8xA100-80GB**, power limited to 330W with a hybrid cube mesh topology. Note that all benchmarks are run at **batch size=1**, making the reported tokens/s numbers equivalent to "tokens/s/user". In addition, they are run with a **very small prompt length (just 5 tokens)**.
> ### Speculative Sampling
> [Verifier: Llama-70B (int4), Draft: Llama-7B (int4)]: **48.4 tok/s**

对照 Llama-2-70B 4-bit(G=32) 无投机 = **25.25 tok/s** → 约 **1.92×**。脚本 `scripts/speculate_70B_int4.sh` 用 `--speculate_k 4 --temperature 0 --max_new_tokens 100 --num_samples 50`，首行注释写 `# 49.26`。

**PyTorch 官方博客**（https://pytorch.org/blog/accelerating-generative-ai-2/ ）：
> *Note: We will be focusing on **latency (i.e. batch size=1)** for all of these benchmarks. Unless otherwise specified, all benchmarks are run on an **A100-80GB**, power limited to 330W.*
> Although speculative decoding guarantees that we have mathematically identical results compared to regular generation, **it does have the property that the runtime performance varies depending on the generated text, as well as how aligned the draft and verifier model are**. For example, when running **CodeLlama-34B + CodeLlama-7B**, we're able to obtain a **2x** boost in tokens/s for generating code. On the other hand, when using **Llama-7B + TinyLlama-1B**, we're only able to obtain about a **1.3x** boost.

> **最后这组 2× vs 1.3× 是"draft/verifier 对齐度决定一切"最简洁的一组对照**：同一硬件同一 batch，只换模型对，加速比差 1.5 倍。

#### torchchat / torchtitan / PyTorch core
**torchchat**（https://github.com/pytorch/torchchat ，**不在** meta-pytorch 组织下；最后 commit 2025-09-10）README 顶部逐字：
> **IMPORTANT**: torchchat is **no longer under active development**.

但代码里确实有投机解码（从 gpt-fast 继承，`torchchat/generate.py::speculative_decode()`）。CLI 参数：`--speculate-k`（**5**）、`--draft-checkpoint-path`（**None**）、`--draft-quantize`（**`"{ }"`**）——**三个 flag 全部 `help=argparse.SUPPRESS`，即功能存在但从 `--help` 里隐藏，属于未文档化状态。**

**torchtitan**：README（11585 字节）grep `specul|draft model` **零命中**——它是预训练框架，不涉及推理投机，符合预期。

**PyTorch core**：未查证到把投机解码做进 `torch` 本体的证据。PyTorch 的官方路线是 ① gpt-fast 作为 ~1000 行参考实现（"we encourage users to copy-paste, fork, and modify"，**明确说不做成库**）；② 通过官方博客背书生态项目——TorchSpec 做训练、[The Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) 做 IBM MLP speculator 的科普（后者正对应上游 vLLM 的 `method: "mlp_speculator"` + `ibm-ai-platform/llama3-8b-accelerator` 系列权重，见 §1.5）。

---

### 6.5 其它值得一提的

#### LightLLM (ModelTC) —— 自研 MTP，全仓不叫 speculative

参数（`lightllm/server/api_cli.py`）：

| 参数 | **默认** | choices |
|---|---|---|
| `--mtp_mode` | **`None`** | `vanilla_with_att`, `eagle_with_att`, `vanilla_no_att`, `eagle_no_att`, `None` |
| `--mtp_draft_model_dir` | **`None`** | — |
| `--mtp_step` | **`0`** | — |

> "None: Disables MTP. `*_with_att`: Uses the MTP model with an attention mechanism… `*_no_att`: Uses the MTP model without an attention module…"
> 官方推荐：**"it is recommended to use `eagle_with_att` for better performance"**

语义：`vanilla_*` → 加载 `mtp_step` 个草稿模型；`eagle_*` → 只加载 1 个草稿头、递归 rollout `mtp_step` 次。**未给 `--mtp_draft_model_dir` 时自动填成 `[args.model_dir] * mtp_step`（用主模型自身当草稿）。**
草稿模型按 `config.json` 的 `model_type` 硬编码分发：`deepseek_v3` / `qwen3_moe` / `mistral` / `glm4_moe_lite` / `qwen3_5` / `qwen3_5_moe`，其余 `raise ValueError`。
限制（代码 assert）：FA3/MLA/FP8/GDN 后端要求 **`batch_size % (mtp_step+1) == 0`**；`diverse_backend` 只支持 `*_no_att`；**开 MTP 会改变 decode attention 后端优先级**——逐字 `(priority when mtp_step > 0: fa3 > flashinfer > triton; otherwise: flashinfer > fa3 > triton)`。

**文档与代码打架**：CLI help 与文档都写 "deepseekv3 model only support 1 step"，但官方回归脚本 `test/acc/test_deepseekr1_mtp.sh` 用 `--mtp_step 2`，GLM-4.7-Flash cookbook 用 `--mtp_step 4`。**"只支持 1 step"是过时文档。**
**Benchmark：仓库内零个实测加速比。** 只有模拟工具 `--mtp_accept_rate`（float，**默认 1.0**，help 逐字："per-draft-token MTP acceptance probability; **sampling is outside the timed decode section**"）——**又一个"接受率是模拟的不是实测的"开关**（第三家，前两家是 SGLang 的 `SGLANG_SIMULATE_ACC_LEN` 与 vLLM 的 `synthetic`）。

#### KTransformers —— 自研版已归档，MTP 转交 SGLang
`README.md` / `doc/en/DeepseekR1_V3_tutorial.md` / `doc/en/balance-serve.md` grep `MTP|specul` **全部零命中**。唯一活的路径在 `doc/en/DeepSeek-V4-Flash.md`（标题逐字 "Optional: Enable MTP (Multi-Token Prediction) Speculative Decoding"）：
> V4-Flash ships a NextN draft head that can be run as EAGLE-style speculative decoding for **~1.2× throughput on single-request decode (validated 26.5 → 32.74 tok/s on 8× RTX 5090, 90% accept rate at chain depth 1)**.

追加到 `python -m sglang.launch_server` 的 flags（**全是 SGLang 的参数**）：
```bash
--speculative-algorithm EAGLE --speculative-num-steps 3 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--speculative-moe-runner-backend auto
```
> ⭐ **这是本次调研中元数据最完整的一条 benchmark**：硬件 8× RTX 5090 ✅、模型 DeepSeek-V4-Flash ✅、吞吐 tok/s ✅、接受率 90% @ chain depth 1 ✅、batch = single-request（`--max-running-requests 2`, `--cuda-graph-bs 1`）✅。
> **但也是一条很好的"高接受率 ≠ 高加速比"的例子**：接受率 90%、chain depth 1 → 每步最多多出 1 个 token → 理论上限约 1.9×，实测 1.2×。**深度只有 1 时，即使接受率 90% 也拿不到多少收益。** 这条正好呼应 Fireworks 那句 "A high acceptance rate alone does not guarantee a speedup"。

老的自研实现（`--draft_model_path` / `--ngram_decoding` 等）全部在 `archive/` 目录下，**不要按它写配置**。

#### RTP-LLM（阿里）—— 方法覆盖最全，且**没有归档**
最后 commit **2026-08-20**，且 commit 标题本身就是投机相关（`feat(dspark): DSv4 merge leftovers — probabilistic draft sampling series…`）。
文档写 4 种，C++ 枚举其实有 7 种：
```cpp
enum SpeculativeType { SP_TYPE_NONE=0, SP_TYPE_VANILLA=1, SP_TYPE_MTP=2,
  SP_TYPE_EAGLE3=3, SP_TYPE_EAGLE=4, SP_TYPE_DETERMINISTIC=5, SP_TYPE_DSPARK=6 };
```
（`SP_TYPE_EAGLE` 与 `SP_TYPE_DSPARK` **文档没写**。）

参数（代码默认值，`speculative_decoding_group_args.py`）：`--sp_type`（**`""`**）、`--sp_model_type`（**`""`**）、`--sp_checkpoint_path`（**None**）、`--sp_quantization`（**None**）、`--gen_num_per_cycle`（**1**）、`--sp_min_token_match`（**2**）、`--sp_max_token_match`（**2**）、`--force_score_context_attention`（**True**）
⚠️ **CLI 是 `--gen_num_per_cycle`（cycle），环境变量却是 `GEN_NUM_PER_CIRCLE`（circle 拼写错误被保留）。**
文档推荐值（与代码默认值不同）：vanilla/mtp/eagle3 用 `gen_num_per_cycle=5`；**deterministic 用 128（batch=1）** + 请求 `extra_config` 里给 `sp_advice_prompt` 和 `sp_edit`(0/1)。

**⭐⭐ 唯一给出量化接受率门槛的框架**：
> **Need to ensure 1st token acceptance rate >80%, 2nd token acceptance rate >60%, 3rd token acceptance rate >40%**
> gen_num_per_cycle: Default is 5; **can be increased if acceptance rate >40%**.
> Execution time of MTP small model can be assumed to be about 1ms. Based on the main model's execution time and acceptance rate, **the optimal GEN_NUM_PER_CIRCLE can be calculated**
> On Hopper series, it is recommended to enable `sp_quantization=FP8_PER_BLOCK`

观测：请求加 `"aux_info": true` → `avg_tokens_per_iter = output_len / iter_count`。
> **The most important metric for speculative sampling is avg_tokens_per_iter, higher is better.**

> ⭐ **"第1位>80% / 第2位>60% / 第3位>40%"是全网唯一一条逐位置的可判定门槛**，而且与 vLLM DSpark blog 的实测（第1位>70%、第7位<10%）量级吻合。**直接写进 `05-接受率alpha` 与 `17-动态草稿长度` 作为工程判据。**
> **Benchmark：只给方法学不给绝对数字，加速比原文未给出。**
> `deterministic` 模式 + `sp_advice_prompt` 是 Fireworks "speculative edits" 的开源对应物（其文档直接引用了 Fireworks 那篇）。

#### NVIDIA Dynamo —— 纯转发，自己不实现
**v1.5.0**，活跃。官方 backend 支持表（逐字）：
> | vLLM | ✅ | Eagle3 draft model support |
> | SGLang | 🚧 | Not yet documented |
> | TensorRT-LLM | 🚧 | Not yet documented |

Limitations（逐字）：
> - **Currently only supports Eagle3 as the draft model**
> - Requires compatible model architectures between target and draft

交叉验证（https://docs.nvidia.com/dynamo/resources/feature-matrix ）：vLLM 那栏 Speculative Decoding = ✓（注 "Eagle3"）；另一 backend 注释逐字 **"Limited integration — Code hooks exist, but examples and documentation are not yet available."**；三家 backend 的「投机解码 × SLA Planner」「投机解码 × 多模态」全是 `na`。

参数全是 backend 的：`python -m dynamo.vllm ... --speculative_config '{"model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","draft_tensor_parallel_size":1,"num_speculative_tokens":2,"method":"eagle3"}'`
Dynamo 自己的投机参数都是**模拟/调度**用的：`--aic-nextn`（默认 None，"max 5"）、`--aic-nextn-accept-rates`（"Entry i is **P(draft i accepted | all earlier drafts were accepted)**"）、`--aic-mtp-seed`（默认 **42**）、mocker `--decode-speedup-ratio`（默认 **1.0**）、planner `speculative_nextn`（默认 **0**）。
⚠️ **易混淆**：`nvext.speculative_prefill`（bool，默认 **false**）**不是投机解码**，是多轮 agent 场景下提前预填下一轮 prompt 前缀暖 KV cache。

> ⭐ `--aic-nextn-accept-rates` 的定义 "P(draft i accepted | **all earlier drafts were accepted**)" 是**条件接受率**，与 vLLM `synthetic_acceptance_rates` 的"**unconditional** acceptance rates"正好是一对。**这两个口径差异必须写进 `05-接受率alpha`——同一个"逐位置接受率"在两家是两种定义。**

**⭐ Dynamo 是所有框架里「什么时候不该开」写得最直白的**（`agent-docs/guides/knob-tuning/`）：
> **MTP can be worse at high concurrency if draft state reduces batch**
> **Validate acceptance rate and memory pressure before treating MTP as a win.** … **If MTP becomes worse than non-MTP at a concurrency, do not extend that lane without evidence.**
> Caveat: **draft tokens and Mamba/SSM state consume capacity**
> **Speculative decode splits high-concurrency batches** … **A token budget that omits draft or lookahead tokens can split active requests across scheduler steps.**

**仓库内无任何投机解码实测数字。**

#### MLC-LLM —— 唯一同时支持 small_draft + eagle + medusa
`python/mlc_llm/serve/config.py`：
```python
speculative_mode: Literal["disable", "small_draft", "eagle", "medusa"] = "disable"
spec_draft_length: int = 0
spec_tree_width: int = 1
```
> `spec_draft_length`: "**Being 0 means to enable adaptive speculative mode**, where the draft length will be automatically adjusted based on engine state. The default values is 0."

草稿模型来源（`docs/deploy/rest.rst` 逐字）：
> When engine is enabled with speculative decoding, additional models are needed. **We only support one additional model for speculative decoding now.** The way of specifying the additional model is: `--additional-models model_path_1` or `--additional-models model_path_1,model_lib_1`.

限制：Medusa 复用 eagle 的 action（`cpp/serve/config.h` `kMedusa = 3`）；`eagle_new_request_prefill.cc` 注释逐字 `// Note: spec_draft_length in engine config has to be match the model config in Medusa.` → **Medusa 模式下 `spec_draft_length` 必须与 medusa head 数一致**。**仓库内无任何加速比数字。**

#### Ollama —— ⭐ 用 llama.cpp 的 draft，而且会自动开
`llm/llama_server.go`：
```go
const ( draftTypeMTP = "draft-mtp"; draftTypeDFlash = "draft-dflash" )
func appendDraftArgs(params []string, draftType, draftModelPath string, opts api.Options) []string {
	if draftType == "" { return params }
	if opts.DraftNumPredict <= 0 { return params }
	params = append(params, "--spec-type", draftType)
	params = append(params, "--spec-draft-n-max", strconv.Itoa(opts.DraftNumPredict))
	if draftType == draftTypeMTP { params = append(params, "--spec-draft-backend-sampling") }
	if draftModelPath != "" { params = append(params, "--spec-draft-model", draftModelPath) }
	return params
}
```
**自动触发**：主模型 GGUF 的 KV 里 `nextn_predict_layers > 0`（或 arch ∈ {qwen35, qwen35moe} 且有 `mtp.` 张量）→ `config.EnableMTP = true`。**权重自带 MTP 头就自动开，用户完全无感。**

参数：`options.draft_num_predict`（`Runner.DraftNumPredict`），`DefaultOptions()` **默认 4**。文档 `docs/modelfile.mdx` 逐字：
> Maximum number of speculative draft tokens to predict per step when a draft model is available. **Separate draft models default to 4; embedded MTP tensors require setting this parameter.** Set to 0 to disable speculative drafting.

create 侧：`draft_files`（map）、`draft_quantize`（str）；CLI `ollama create --draft-quantize <level>`。
Modelfile 有 `DRAFT` 指令（parser 支持、有测试），但 **`docs/modelfile.mdx` 的 Instruction 表里没列它**——能用但未文档化。报错文本：`"DRAFT cannot be used with remote models"`、`"--draft-quantize requires a DRAFT model"`。
**Ollama 只暴露 MTP / DFlash 两条路，不暴露经典 `draft-simple`，也不暴露任何 n-gram。** Apple 侧另有独立实现 `x/mlxrunner/{mtp.go, dflash.go}`。

> **这条对"本地推理场景"的图景很重要**：**普通用户其实已经在用投机解码了，只是不知道**——只要模型权重带 MTP 头，Ollama 就自动开，默认 K=4。

#### DeepSpeed-FastGen / MII —— ❌ 确认没有
HEAD `8abdd98742198`，最后 commit **2025-07-01**（停更 >13 个月），version 0.3.4。
全仓库（含 .md/.py，仅排除 .git）大小写不敏感 grep：`grep -rilE "specul|draft|lookahead" .` → **0 命中**。
README（20477 字节）`grep -c -i "specul\|draft\|medusa\|eagle"` = **0**。README 列的能力集逐字：
> blocked KV-caching, continuous batching, Dynamic SplitFuse, tensor parallelism, and high-performance CUDA kernels

**→ 未查证到任何官方支持证据，路线图上也没有。**

#### Medusa 官方 repo —— 已死，且自己写明只支持 batch=1
**最后 commit 2024-04-18（>2 年 4 个月），72 commits，GitHub 上未标 archived。**
用法 `python -m medusa.inference.cli --model [path]`；CLI **没有暴露任何投机专有参数**（`--temperature` 默认 0.7、`--max-steps` 默认 512），树结构与 `posterior_threshold`/`posterior_alpha` 只在 `medusa/model/utils.py` 内部。

**⭐ 官方限制（README 逐字，加粗是原文语气）**：
> We currently support **single-GPU inference with a batch size of 1**, which is the most common setup for local model hosting.
> In the initial release, our primary focus is on optimizing Medusa for **a batch size of 1**

`ROADMAP.md` 的 `- [ ] Batched inference` **至今未勾选**；集成清单里 vllm / lightllm / llama.cpp / mlc-llm **全部未勾选**。

加速比：
> The new results show a **2.2-3.6x speedup** over the original model on a range of LLMs.（arXiv 2401.10774）/ approximately a **2x** speed increase across a range of Vicuna models

**硬件、延迟还是吞吐、平均接受长度 τ —— 原文均未给出**；batch size = 1（原文明确）。
Together AI 的 Medusa 博客（2023-09-11）补充：训练 "**a single A100-80G GPU**，几小时到一天"；33B 用 8-bit 量化；MT-bench 评测；"a 33B parameter Vicuna model could operate as swiftly as a 13B model"。

> ⭐ **"Medusa 官方从未支持 batch>1、roadmap 至今未勾选、四个集成目标全部未勾选、停更两年多"——这是 `12-Medusa` 篇"它为什么出局"最完整的证据链。** 加上四大引擎的态度（§0），可以下一个明确结论：**Medusa 的贡献是概念（多头草稿 + 树注意力 + typical acceptance），不是实现。**

#### EAGLE 官方 repo —— 活着，EAGLE-3 已发布
**最后 commit 2026-02-19**（⚠️ README 的 Update 时间线最新只到 2025.9.18，**README 滞后于代码，别拿它判断活跃度**）。
> The default main branch is the implementation of **EAGLE-3 and EAGLE-2**. For using EAGLE-1, please switch to the **v1 branch**.
> The current code defaults to using EAGLE-3. If you want to use EAGLE weights, please specify **`use_eagle3=False`**

推理 API 默认值（`EaModel.from_pretrained`）：`use_eagle3=True`、`total_token=60`、`depth=7`、`top_k=10`、`threshold=1.0`。
> The *total-token* is the number of draft tokens. For smaller models and advanced GPUs, this value can be set larger. **If set to -1, EAGLE-2 will automatically configure this parameter.**
（自动调参：在 `[40,48,50,56,60]` 上跑 20 次前向计时取最快。）

**加速比（README 逐字，统一条件）**：
> - EAGLE：**2×** speedup on gpt-fast；**3×** faster than vanilla decoding (13B)；**2×** faster than Lookahead (13B)；**1.6×** faster than Medusa (13B)
> - EAGLE-2：**4×** faster than vanilla decoding (13B)；1.4× faster than EAGLE-1
> - EAGLE-3：**5.6** faster than vanilla decoding (13B)；1.8× faster than EAGLE-1
> - 统一条件：_Inference is conducted on **2x RTX 3090 GPUs at fp16 precision using the Vicuna 13B model**._

**batch size 原文未给出**（评测走 MT-bench 逐题生成，无 batch 参数）；**延迟还是吞吐原文未给出**（只写 "faster"）；**平均接受长度 τ README 未给出**。

权重来源警告（逐字）：
> *Note:* This repository **recognizes only official EAGLE-3 checkpoints**. Performance of unofficial checkpoints may vary.

官方 EAGLE-3 权重只有 4 个（Vicuna1.3-13B / LLaMA3.1-8B / LLaMA3.3-70B / DeepSeek-R1-Distill-LLaMA-8B）；Qwen3 / Llama-4 / GLM-4.7-Flash 等**全部标 Official=No**（第三方训的）。
官方还自己推荐别用这个 repo 训：
> **2025.7.23**: We strongly recommend using **SpecForge** for out-of-the-box training of EAGLE-3 with SGLang.

> ⭐ **"EAGLE 论文作者自己把训练推荐给了 SGLang 的 SpecForge"** 是一条很能说明生态分工的史料：**算法作者做算法，训练交给引擎方，推理交给引擎方。** 写进 `13-EAGLE三代` 与 `21-草稿模型怎么训`。
> ⚠️ 另外注意 README 的 "5.6 faster"（缺 ×）与 EAGLE-1 的 "3× vs vanilla / 1.6× vs Medusa" 都是 **2×RTX 3090 + Vicuna 13B + fp16 + 无 batch 说明**——按铁律二属于口径不全，**不可与本调研中任何引擎侧数字横比**。

---

### 6.6 云厂商 / 推理服务商公开博客

> ⚠️ **一条重要更正**：社区常说的「Anyscale 讲大 batch 下 spec dec 收益消失」那篇——**不在 anyscale.com 上**。枚举了 Anyscale sitemap 全部 **620 个 URL（含 319 个 /blog/ 页）**，`speculative` 命中 = **0**。实际那篇是 **vLLM 官方博客**（vLLM 的 spec decode 由当时在 Anyscale 的 Cade Daniel 主导，社区常归给 Anyscale）。

#### Baseten —— 内容最系统，且唯一给出数值阈值

（日期歧义：三篇 2024 系列文章页面显示 `May 16, 2025`，但 JSON-LD `datePublished` 为 `2024-12-19`。**无法判定，两个都如实标出。**）

**[A quick introduction to speculative decoding](https://www.baseten.co/blog/a-quick-introduction-to-speculative-decoding/)** —— ⭐ 吞吐 vs 延迟的显式取舍：
> **However, speculative decoding limits a model server's throughput.** Speculative decoding performance suffers with larger batch sizes, and batching is essential for high throughput. While there are strategies we used to address this limitation, it's important to understand that **the latency improvements offered by speculative decoding do reduce maximum system throughput, ultimately resulting in higher per-token inference costs as more hardware is required to serve the same traffic**.
> While this setup is compatible with in-flight batching, **speculative decoding isn't as useful with high batch sizes**. Instead, it's more useful when we have spare compute capacity, as is the case when running long input sequences through large models.
> Speculative decoding is a latency optimization that works best for **large models and small batch sizes**. … **When you must meet latency SLAs for large models. When your batch size is already limited by large models and long contexts taking up GPU memory.**
> Draft model selection is generally limited to the **same model family**… as the LLMs need to use the **same tokenizer**. / In our testing, we saw a **15% improvement in acceptance rate on a fine-tuned draft model**.

> ⭐ **"投机解码提升延迟但降低最大吞吐、因而抬高每 token 成本"是本次调研里对这笔交易的最完整表述**，正好是本库写作规范里"投机是一笔用算力换延迟的交易"的商业版。**建议直接引用进 `19-负收益全解` 的开篇或 `29-本质总结`。**

**[Engine Builder integration](https://www.baseten.co/blog/speculative-decoding-engine-builder-integration/)**：
> **That said, there are scenarios where you might not want to use speculative decoding.** For instance, **under high load (when GPU usage is at or near 100%), running the draft model in addition to the larger one can cause performance bottlenecks.** And if you're already using **relatively lightweight LLMs**, the overhead of running two models could outweigh any performance gains.

**[How we built production-ready speculative decoding with TensorRT-LLM](https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm/)** —— ⭐ 最诚实的一篇，**承认自己 benchmark 有一组是负收益**：
> While two benchmarks follow the expected pattern, **speculative decoding actually resulted in higher p50 latencies for one test. This is a risk when using speculative decoding.** To debug, we'd examine the draft tokens and target model output to figure out why the draft model wasn't making acceptable tokens.
> Inefficient batching: requests were either **failing to batch or performance was dropping massively at batch sizes higher than 1**.
> the draft and target models had a tendency to **fight for resources** when run on the same GPU… **the model would run at half speed when they were in contention**.
> if you fully utilize the KV cache's memory allocation, this leads to **repeated prefill computations, significantly degrading performance**.
> **speculative decoding performance degrades when the model is given more creative freedom (e.g. high temperature, high top_k, high top_p)** as the token distribution gets less predictable
> While speculative decoding makes the overall response generation faster, **it still does make the time to first token slightly worse**

配置（本次调研中最可复现的一组）：
- 组 A：target=`Qwen2.5-Coder-14B-Instruct`，draft=`Qwen2.5-Coder-0.5B-Instruct`，**共享 1×H100**，`speculative_decoding_mode: DRAFT_TOKENS_EXTERNAL`，`num_draft_tokens: 4`，`max_seq_len: 10000`，`kv_cache_free_gpu_mem_fraction: 0.62`
- 组 B：target=`Llama-3.1-70B-Instruct`（fp8_kv），draft=`Llama-3.1-8B-Instruct`，**4×H100**，`tensor_parallel_count: 4`，`num_draft_tokens: 4`，`max_batch_size: 256`，`batch_scheduler_policy: guaranteed_no_evict`

⚠️ **p50/TTFT/TPOT 的具体数值全在图片里，HTML 正文不含数字**，只有标题级的 "up to 90% faster in production"；**实测并发数原文未给出**。

**[Baseten 官方文档](https://docs.baseten.co/engines/bis-llm/advanced-features)** —— ⭐⭐ **全场唯一给出数值阈值**：
> The BIS-LLM dashboard exposes `speculation_rate`…
> * **Above 80%**: Draft is well-aligned with the main model. Speculation is effective.
> * **40-80%**: Some rejections. Consider tuning the draft model or switching decoding types.
> * **Below 40%: Speculation likely costs more than it saves. Disable it or reduce draft length.**

功能不兼容清单（[lookahead 文档](https://docs.baseten.co/engines/engine-builder-llm/lookahead-decoding)）：
> * During speculative decoding, **sampling is disabled and temperature is set to 0.0**.
> * **Structured output (xgrammar, outlines) with state-machine guarantees … isn't possible when lookahead decoding is enabled.**
> * **Chunked prefill isn't supported with lookahead decoding.**

**[Boosting MTP acceptance rates](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/)**（2026-05-05）—— 领域依赖性最强证据：
> **SA Decoding shines at code generation, where the accept length is 10+ with long context, but performs poorly on reasoning and other writing tasks, with accept rates near 0.** Meanwhile, MTP produces consistent speed-ups across all domains, though the accept rate is usually only **2-4 tokens per iteration**.
数据：`nvidia/DeepSeek-V3.1-NVFP4` + `glaiveai/code-edit-samples`，混合方法比纯 MTP **接受长度与吞吐高 30%–33%**；生产 agentic coding 上「等延迟下吞吐 +40% / 等吞吐下延迟 −40%」。**具体 batch、GPU 型号、绝对 tok/s 原文未给出。**

**[Live draft model training](https://www.baseten.co/blog/live-draft-model-training-for-speculative-decoding/)**（2026-06-25）：
> **Alignment drift: Fine-tuning or reinforcement learning (RL) on the base model often degrades the draft model's accept rate unless it is retrained alongside it.**
> median increase in accept rate of **20%**, with some constrained traffic patterns seeing **100%+** increases

> ⭐ **"alignment drift"这条对 `21-草稿模型怎么训` 很关键**：**草稿模型是会过期的资产**——base 模型每做一次 SFT/RL，草稿就要重训，否则接受率衰减。这与 Together ATLAS 的"输入分布漂移"是同一族问题的两个面（模型漂移 vs 流量漂移）。

#### ⭐⭐ Together AI —— 唯一系统性「反驳」大 batch 无用论的

**[Speculative decoding for high-throughput long-context inference](https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference)**（2024-09-05）——**这篇必须和 vLLM 那篇成对引用**：

> **Conventional wisdom (e.g., Chen et al., 2023; Li et al., 2024; Liu et al., 2024) is that in the high-throughput regime (i.e., large batch sizes), speculative decoding—which leverages underutilized GPU compute during memory-bound decoding—does not make sense, because decoding will be compute-bound and the GPUs will thus be fully utilized.** Surprisingly, we show analytically and empirically that **for large batch sizes, if the input sequences are long enough, decoding once again becomes memory-bound due to the large size of the KV cache.** Building on this key observation, we demonstrate that speculative decoding can improve throughput and latency by up to **2x on 8 A100s** in this large-batch, long-context setting.

机制（这段推导本身就值得进正文）：
> operations involving model parameters become compute-bound as the batch size increases (as their **arithmetic intensity equals the batch size b**), operations involving the KV cache are **always memory-bound** (as their arithmetic intensity is constant…).
> counterintuitively, in the long-context regime, **a larger batch size results in decoding being more memory bound**, instead of the other way around.
> This makes the draft-to-target cost ratio **decrease** with increasing batch size… surprisingly making **speculative decoding more effective for larger batch sizes**.

**Table 2（8×A100）**：

| Target | Draft | Prefill | **Batch** | Optimal spec len | Speedup |
|---|---|---|---|---|---|
| Llama2-7b-32k | TinyLlama-1.1B | 8000 | 32 | 3 | 1.29 |
| Llama2-7b-32k | TinyLlama-1.1B | 8000 | 64 | 3 | 1.57 |
| Llama2-7b-32k | TinyLlama-1.1B | 8000 | 128 | 4 | **1.66** |
| Llama2-7b-32k | TinyLlama-1.1B | 32000 | 32 | 4 | 1.91 |
| Llama2-7b-32k | Self-spec | 32000 | 32 | 4 | **2.00** |
| Llama3.1-8b | Self-spec | 32000 | 32→128 | 3→4 | 1.22→1.47 |
| Llama3.1-8b | Self-spec | 100000 | 32 | 5 | 1.84 |

> ⭐⭐ **注意方向性：固定 prefill=8000，batch 32→64→128 时加速比 1.29→1.57→1.66 单调上升——与 vLLM 的高 QPS 减速正好相反。** 接受率原文未给出；"Speedup" 未区分延迟/吞吐。

draft 选型随 regime 反转（很反直觉）：
> In the **low-latency regime**… **using a small draft model is generally the key**. However, in the **high throughput regime**… the bottleneck is loading the target model KV cache… we can afford to use a **larger and more powerful** target model as long as its KV cache is kept small. Thus, we propose using **self-speculation, where the target model is used as the draft model**, but with limited context size.

> ⭐ **"高吞吐 + 长上下文下，草稿模型反而该更大"** 直接推翻了社区常识"草稿越小越好"（写作规范 §4 已把"说草稿模型越小越好"列为高频错误）。**这条是本库 `10-自投机` 与 `22-长上下文` 的核心素材。**

其余：[Custom Speculators for DeepSeek-R1](https://www.together.ai/blog/customized-speculative-decoding)（2025-05-12，定制 speculator 相对 base speculator **1.23–1.45×** decode 加速 + ~25% 成本下降；相对无投机 **1.85–2.97×** + ~55% 成本下降；口径声明很少见：**"Inference measurements are obtained through hold-out sets from each user's traffic at the low latency regime." / "Costs are estimated… at the high throughput regime."**）；[ATLAS](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)（2025-10-10，DeepSeek-V3.1 最高 **500 TPS**、**2.65×**，⚠️ **全部 batch size = 1，4×B200**，105 TPS → 501 TPS；RL 场景 Qwen2.5-7B-Instruct-1M + DeepScaler + H100，**接受率从 <10% 升到 >80%**，训练总时长降 >60%）；[Sequoia](https://www.together.ai/blog/sequoia)、[SpecExec](https://www.together.ai/blog/specexec)、[Distribution-aware spec dec for RL rollouts](https://www.together.ai/blog/distribution-aware-speculative-decoding)（2026-04-24）。

ATLAS 对静态 speculator 的失效条件（逐字）：
> Once deployed, the speculator cannot adapt, leading to **degrading performance if the input distribution evolves**. This problem is particularly pronounced in **serverless, multi-tenant environments, where input diversity is sky-high**.

Together 的 Medusa 博客里对经典投机解码的三条批评（逐字，写 `09-2023奠基` 时可用）：
> **Finding the Ideal Draft Model**: Identifying a 'small yet mighty' draft model that aligns well with the original model is easier said than done. **System Complexity**: Hosting two distinct models in one system introduces layers of complexity… **Sampling Inefficiency**: When doing sampling with speculative decoding, an importance sampling scheme needs to be used. This introduces additional overhead on generation, **especially at higher sampling temperatures**.

#### Modal（2026 年最新视角）
**[Speculation Is All You Need](https://modal.com/blog/spec-is-all-u-need)**（2026-06-19）—— ⭐ 唯一提到 MoE vs dense 行为不同：
> it correctly predicts that **speculator speedup is non-monotonic ('u-shaped') as a function of batch size for mixture-of-experts models, like DeepSeek-V4 Pro, but monotonic for dense models, like Qwen 3.5 27B**.
> **for small batch sizes**, transformer decoding without speculation doesn't use all the arithmetic bandwidth of the GPU, and so **even when much work is wasted, this needn't slow down execution**.
> speculators must run faster than the target… **Smaller models struggle to saturate the memory and arithmetic bandwidth and more readily incur host overhead**… **Disaggregation of speculator and target is intriguing, but we expect communication latency to relegate it to a minority of niche use cases.**
> We have seen fine-tuned models **increase acceptance lengths from a baseline of 3 to over 9. That's the difference between a 25% speedup and a 3x speedup**

⚠️ **口径警告**：这篇多数数字来自 **roofline 模拟器 + SGLang `SGLANG_SIMULATE_ACC_LEN` 接受率 mocking，不是端到端实测**，原文自己声明 "This simulator is imperfect"。
> **"MoE 的 U 型曲线"是一条本库应当收录但标为"待验证"的假说**——它能同时解释 Red Hat gpt-oss（MoE，200 并发仍有收益）与 SGLang Intern-S2（dense-ish，conc 256 输给不开）两组看似矛盾的数据。

**[Achieve SOTA inference latencies with speculative decoding](https://modal.com/blog/achieve-sota-specdec)**（2026-06-24，Modal × Decagon 语音 agent，SGLang + Blackwell）：**~290ms p50 → 砍掉 100ms**，比最好的商业方案快 60ms+。**target/draft 模型名、batch、并发原文未给出**（客户负载被 `<redacted>`）。

#### Fireworks AI
[官方文档](https://docs.fireworks.ai/deployments/speculative-decoding) —— ⭐ 一句很重要的反直觉提醒：
> **A high acceptance rate alone does not guarantee a speedup because the drafter also consumes compute.**
> The benefit depends on both the cost of producing draft tokens and how often the target model accepts them. **A poorly matched drafter can make generation slower**, so benchmark with representative traffic before overriding Fireworks defaults.
> A model that is merely smaller is **not** necessarily a useful drafter; **its acceptance rate and execution cost both matter**.
> Acceptance normally falls at later positions. **Increase `--draft-token-count` only while the additional accepted tokens outweigh the extra drafting and verification work.**

默认 drafter：`llama-v3p2-1b-instruct`（给 >3B 的 Llama）、`qwen2p5-0p5b-instruct`（给 >3B 的 Qwen）；支持 EAGLE / DFlash / DSpark / Medusa addon，但 "they are **not drop-in replacements** for a small base-model drafter."
指标：`speculation-generated-tokens`、`speculation-acceptance`（**acceptance rate by proposed-token position**）。**文档无任何 benchmark 数字。**

**[How Cursor built Fast Apply](https://fireworks.ai/blog/cursor)**（2024-06-23）：
> Fireworks deployed Cursor's special fine-tune of Llama-3-70b… enabling them to **~13x speedup over vanilla inference using Llama-3-70b and a ~9x speedup over their previous GPT-4 speculative edits deployment** leading them to achieve **~1000 tokens/sec**.
> Cursor built a variant of speculative decoding called '**speculative edits**', an algorithm that uses **much longer speculations**… possible **in case of partial text rewriting when the caller has a strong guess of what the generation might look like**
> the server will find the **longest prefix** of the 'speculation' field that matches the model's generation with **temperature=0**.
> **Most LLM use cases have a wide variety of possible inputs which makes it hard to produce good speculations.**

⚠️ **13× 是「调用方已知大部分输出」的特殊场景，不可外推到通用 chat。硬件 / batch / QPS / 接受率原文均未给出。**
> **speculative edits 值得在 `11-无模型草稿` 里单列**：它是把"草稿来源"从模型/n-gram 换成了**调用方提供的先验**（用户要改的原文），是接受率能到 90%+ 的唯一现实场景。RTP-LLM 的 `deterministic` 模式 + `sp_advice_prompt` 是它的开源对应物。

#### Azure / AWS / Google Cloud
**Azure（Microsoft Foundry，ms.date 2026-06-01，preview）** —— ⭐ 唯一把 prefill 限制写进文档的云厂商：
> Speculative decoding pairs a target model with a same-family, architecture-compatible draft model to reduce token generation latency during decoding. **It does not improve context processing latency in the prefill phase, so workloads with long inputs and short outputs may see limited benefit.**
（底层是 Fireworks on Foundry。URL: https://learn.microsoft.com/en-us/azure/foundry/how-to/fireworks/import-custom-models ）

**AWS SageMaker AI**：
> Speculative decoding is a technique to speed up the decoding process of large LLMs. **It optimizes models for latency without compromising the quality of the generated text.**
> **You incur no additional costs for jobs that apply speculative decoding.**
用法 `ModelBuilder.optimize(speculative_decoding_config={"ModelProvider": "SAGEMAKER"})`；用 SageMaker 自带 draft model 时**必须 `enable_network_isolation=True`**。
**❌ 无任何「什么时候不该开」表述，无任何 benchmark 数字。**

**Google Vertex AI**：拉了 cloud.google.com 的 **180 个子 sitemap 全量 grep**，`speculative` 命中 260 条 URL，**全部是各语言 SDK 的 `SpeculativeDecodingSpec` 类型参考页，教程/指南/概念文档命中 = 0**。支持 `DraftModelSpeculation` 和 `NgramSpeculation`，参数 `speculative_token_count`。⚠️ 该 sitemap **不含 cloud.google.com/blog**，Google Cloud Blog 未查证。

#### 未查证到公开投机解码内容的厂商

| 厂商 | 核查方法 | 结果 |
|---|---|---|
| **Anyscale** | sitemap 620 URL（319 个 /blog/）全量 grep | `specul` = **0**（仅 docs 一页提及，无数字无负收益表述） |
| **Databricks / Mosaic** | sitemap 不含 /blog/；直抓旗舰文 *LLM Inference Performance Engineering*（741KB） | `specul` = **0** ⚠️ **本次覆盖度最弱的一项，建议单独复核** |
| **Cerebras** | sitemap 431 URL | **0** |
| **Groq** | sitemap 49 URL | **0** |
| **SambaNova** | sitemap 768 URL（含大量 /blog/） | **0** |
| **Replicate** | sitemap + /blog 页 | **0** |
| **Predibase** | sitemap 返回 Apache 默认页；/blog 返回 **HTTP 403** | **完全无法核查** |

---

### 6.7 §6 的「未查证」清单

1. LMDeploy 自身的加速比 / 接受率实测数字——**官方从未发布**；`num_speculative_tokens` 的推荐值与上限同样无官方说法。
2. LMDeploy 官方「什么时候不该开」的表述——**不存在**，只有 4 条代码级硬互斥 + 一句 experimental。
3. LMDeploy `method='eagle'`(v1) 的实际可用性——代码注册了、模型文件在（仅 Llama），但 CI 零覆盖、文档零示例。
4. TurboMind 是否有投机解码 roadmap——未找到任何 issue / roadmap 表明官方有此计划。
5. MindIE `speculationGamma` 的默认值——查了 3 个官方来源确认**官方未给**。
6. MindIE 并行解码 / MTP 的精确引入版本——并行解码至少在 1.0.RC3 已存在但更早未确证；MTP 最早查到 2.0.RC1，「2.0.RC1 首发」属推断。
7. MindIE lookahead / memory_decoding 的 benchmark——中英文均未查证到任何带口径数字。MTP 性能数字全部二手，且接受率是 70% 折算假设不是实测。
8. vllm-ascend 的 P-Eagle / PARD 参数名——只在 release note 里出现，`speculative_decoding.md` 与 `additional_config.md` 都没有对应 key（grep `parallel_drafting|p_eagle|peagle` 零命中）。
9. vllm-ascend MTP / EAGLE 的 benchmark——只有 Suffix Decoding 那一份教程有完整数字。
10. Arctic Inference 的 `hard_disable_by_batch_size`——仅存在于源码，文档零提及，默认 None。
11. EAGLE / Medusa 论文里的平均接受长度 τ——README 均未给出，需读 arXiv 2401.15077 / 2406.16858 / 2503.01840 / 2401.10774。
12. Baseten 三篇 2024 系列文章的真实发布日——页面 `May 16, 2025` vs JSON-LD `2024-12-19`，无法判定。
13. Baseten TRT-LLM 深度文的 p50/TTFT/TPOT 绝对值——全在图片里。
14. **Databricks / Mosaic 是否有投机解码博客**——sitemap 不含博客 URL + 搜索引擎被拦截，**无法排除存在**。本次覆盖度最弱的一项。
15. Google Cloud Blog——sitemap 不含 /blog，未覆盖。
16. Predibase——站点对 curl 返回 403，完全未能核查。
17. 本节结论全部基于一手源码与仓库内文档，**未做 issue 层面搜索**（WebSearch 配额耗尽 + GitHub API 限流 + 多个搜索引擎 captcha 拦截）。

---

## 7. 专节：官方承认的「什么时候不该开投机解码」

> 这是本次调研**最该被 `19-负收益全解-什么时候投机反而更慢` 直接取用**的一节。原则：只收**引擎官方**（文档 / 官方博客 / 维护者 PR）的表述，社区实测另起一栏并明确标注。

### 7.1 一句话总表

| 引擎 | 有没有官方"别开"的表述 | 最锋利的一句 |
|---|---|---|
| **vLLM** | ✅ 有，且**自己贴了减速倍数** | 「As high QPS, we see **1.4x slowdown** Llama3-70B on ShareGPT with 4xH100, **1.8x slowdown** ... on CNN Dailymail」 |
| **SGLang** | ✅ 有，且**在 4 个模型 cookbook 的 high-throughput 档位里直接把投机关掉** | 「**at saturation the verify step costs more than it saves**」 |
| **llama.cpp** | ❌ **没有**（反而说 "should almost never result in slower"） | — |
| **HF transformers** | ✅ 有，且给出了**唯一一个具体阈值** | 「**Above batch size 4, speculative decoding returns slower inference than the main model alone.**」 |
| **TensorRT-LLM** | ✅ 有，且**用逐并发 audit 配置把结论落成了文件** | 「**There is currently no way to dynamically disable speculation, thus speed ups are only observable at low batch sizes.**」 |
| **vllm-ascend** | ✅ 有，与上游同源 | 「**When `BS * K` goes beyond a critical batch size, speculative decoding can hurt decode speed (TPOT).**」 |
| **MindIE（华为）** | ✅ **中文侧写得最系统的三条前提** | 「**当前的并发数不高，属于内存带宽受限、计算资源有冗余的情况**」「**需要一定长度的输出才有性能提升效果**」 |
| **Arctic Inference** | ✅ 承认自己一度不如 n-gram | 「**it actually became worse at a concurrency of 64**」「remaining overhead… **around 10% at a concurrency of 64**」 |
| **Baseten** | ✅ **唯一给出数值阈值** | 「**Below 40%: Speculation likely costs more than it saves. Disable it** or reduce draft length.」 |
| **Fireworks** | ✅ 反直觉提醒 | 「**A high acceptance rate alone does not guarantee a speedup because the drafter also consumes compute.**」 |
| **NVIDIA Dynamo** | ✅ 运维口径最直白 | 「**If MTP becomes worse than non-MTP at a concurrency, do not extend that lane without evidence.**」 |
| **Azure Foundry** | ✅ 唯一把 prefill 限制写进文档 | 「**workloads with long inputs and short outputs may see limited benefit**」 |
| **LMDeploy** | ❌ 没有（只有一句 experimental） | — |
| **AWS SageMaker** | ❌ 没有 | — |

**⭐ 三条最该被引用的官方"临界点"数字**（全部有出处，口径见各节）：

| 来源 | 临界点 |
|---|---|
| TensorRT-LLM `AUTO` 启发式源码 | **BS ≥ 32 关闭投机**（`max_concurrency=32`） |
| SGLang adaptive 内置默认 | **BS ≥ 32 锁 `steps=1`** |
| TensorRT-LLM DeepSeek-R1 audit 配置 | **concurrency 64~512（随 ISL/OSL 变动 8 倍）时 `max_draft_len` 从 3 降到 1** |
| vLLM Dynamic SD 文档示例 | **concurrency ≥ 129 时 K=0** |
| HF Whisper 博客 | **batch size > 4**（⚠️ 是 HF 自身实现缺陷造成的，见成因 D） |

→ **三家生产引擎的临界点都落在 BS 32~128 这个区间**，且 TensorRT-LLM 的 audit 数据显示它随 ISL/OSL 可以浮动 8 倍。

### 7.2 按"负收益的成因"归类（这是写正文的骨架）

#### 成因 A：GPU 已进入 compute-bound，草稿 token 与真 token 抢算力

**vLLM `adaptive_verification.md`**（最完整的一段官方论述）：
> Speculative decoding buys fewer decode steps with more compute. **At batch size 1 that is a good trade**: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. **At batch size 256 it is a much more delicate one.** Draft tokens now compete with real tokens for the same compute, and **every rejected token is compute wasted; with enough of them, throughput drops.**
> That matters because **per-position acceptance decays fast.** ... **once it saturates the gamble has a real throughput cost.** The crossover moves with load and with workload-dependent acceptance rates, so **no static `num_speculative_tokens` is right across concurrencies.**

**vLLM 官方 blog**：
> The extra compute required to propose and verify tokens **can sometimes slow down the system when it is already compute-bound**, as seen when the number of requests per second increases. In such cases, **the overhead of speculative decoding can outweigh its benefits, leading to reduced performance.**

**SGLang cookbook（DeepSeek-V4）**：
> `high-throughput`: **MTP disabled — at saturation the verify step costs more than it saves.**

**SGLang cookbook（Qwen3.8）**：
> **MTP is off at saturation because draft-plus-verify overhead outweighs the speedup.**

**SGLang cookbook（Intern-S2-Mobius）**：
> **The high-throughput recipe stays spec-off because once you can batch wide, its saturation point is higher** (34786 tokens/s at conc=256 vs the spec recipe's peak at conc=64).

**TensorRT-LLM features 文档**：
> Speculative decoding is a technique for accelerating LLM inference **at low batch sizes**.
> ... **There is currently no way to dynamically disable speculation, thus speed ups are only observable at low batch sizes.**

**TensorRT-LLM legacy 文档**——把加速比拆成两条可分别证伪的前提，是所有引擎里讲得最清楚的一段：
> This can lead to a reduction in the average per-token latency **in situations where the GPU is underutilized due to small batch sizes.**
> The underlying assumptions are twofold:
> 1. processing multiple draft tokens concurrently will be as rapid as processing a single token
> 2. multiple draft tokens will be validated successfully over the course of the full generation
> **If the first assumption holds true**, the latency of speculative decoding will no worse than the standard approach.

**TensorRT-LLM blog06（Llama4 Maverick + EAGLE3）**：
> **When increasing the concurrency of requests, the tokens per second (TPS) per user degrades rapidly.**

> **判据（可判定形式）**：不是"并发数 > N"，而是**"该模型在该硬件上有没有进入 compute-bound"**。同一个并发数下，MoE + 低比特量化的模型（激活参数少）远未饱和，dense + FP16 的模型早已饱和。Red Hat 的 gpt-oss-120B（MoE/MXFP4/H200）在 200 并发下仍有 +20% 吞吐，而 SGLang 的 Intern-S2-Mobius（H200/FP8）在 conc=256 时开投机反而输给不开——**两者不矛盾，差别在模型的算术强度。**
> **TensorRT-LLM 的 audit 数据给了这条判据最强的支持**：同样 H200+TP8+DeepSeek-R1，长输出（ISL/OSL=1k-8k）的拐点在 concurrency 512，而长输入短输出（8k-1k）的拐点只有 64——**同一模型同一硬件，仅因输入输出长度比不同，临界并发就差 8 倍。**

#### 成因 B：有效 batch = BS × K，超过临界值 TPOT 恶化

**vLLM `dynamic_speculative_decoding.md`**（给了公式形态）：
> SD methods need to verify K tokens for each sequence during decoding. As BS increases, **the effective BS becomes BS*K** which increases the compute requirement during verification. **When this BS*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT).**

**TensorRT-LLM blog07（NGram）给出了同一个公式的另一种写法**：
> Run the target model with **`verification_batch = original_batch × (v+1)`**; ... **The iteration latency grows as the verification batch becomes larger than the original batch. As we increase `max_draft_len (v)`, the overhead grows even more. Therefore, speculative decoding tends to work best with small batch sizes and low concurrency.**
> We can see that N-Gram can provide speed-ups **for batch sizes up to 32** and works best with a single batch. **The main overhead with larger batch sizes is the verification cost.**

> ⭐ vLLM 的 `effective BS = BS×K` 与 TensorRT-LLM 的 `verification_batch = original_batch × (v+1)` 是**同一个公式**（差 1 是因为 TRT-LLM 把 bonus token 也算进去）。**两家独立给出同一个模型——这条可以当作 `18-batch与吞吐` 的核心公式，且有双引擎背书。**

对应的官方缓解手段（**四家都做了同一件事**，命名不同）：

| 引擎 | 机制 | 参数 |
|---|---|---|
| vLLM（V0，**已删**） | 队列长度超阈值就对新请求关投机 | `disable_by_batch_size` |
| vLLM（V1，现役） | 按并发区间给不同 K，**可降到 K=0** | `num_speculative_tokens_per_batch_size`，如 `[[1,64,3],[65,128,1],[129,512,0]]` |
| vLLM（前沿） | 逐 step 按 survival probability 竞价，预算由启动时 profiling 的成本模型给出 | `enable_adaptive_verification`（仅 DSpark + confidence head） |
| SGLang | 按接受率动态调 num_steps；**内置默认 BS≥32 锁 `steps=1`** | `--speculative-adaptive` / `--speculative-adaptive-config` |
| **TensorRT-LLM** | 并发超阈值即关投机 | `max_concurrency`（`AUTO` 启发式硬编码为 **32**） |
| **TensorRT-LLM** | 按 batch 分档降 draft_len，最后一档隐式为 0 | `draft_len_schedule`，如 `{4:4, 8:2, 32:1}` |
| **TensorRT-LLM** | 滚动接受率低于阈值 → **永久熔断** | `acceptance_rate_window_size` + `acceptance_rate_threshold`（`SpeculationGate`） |
| HF transformers | 草稿自信度低于阈值就提前收手 | `assistant_confidence_threshold`（默认 **0.4**）+ `num_assistant_tokens_schedule` |

SGLang adaptive 文档原文：
> **At high batch sizes, the cost of each wasted draft step is multiplied across all sequences in the batch, so the optimal step count is often lower than at low batch sizes.**
> `ceiling_coeff` — an optional EMA ceiling rule can cap `num_steps` proportionally to observed draft quality, **preventing over-speculation at high BS**
> **At high batch sizes, narrower ladders (e.g., `[1, 2]` or `[1]`) often outperform wide ones**
> This is the built-in default: BS 8–31 allows `[1, 3]`, and **BS≥32 locks to `step=1` to avoid wasted compute.**

> **这条对本库 `18-batch与吞吐-收益衰减曲线` 是直接可用的**：四家引擎独立收敛到了同一个结论（K 必须随 batch 下降），且 SGLang 给了一个具体的分档表（BS 8–31 → K∈[1,3]；BS≥32 → K=1），TensorRT-LLM 的 `AUTO` 直接硬编码 `max_concurrency=32`。**这可以当作经验曲线的锚点。**

**⭐ 衰减曲线的最干净一组实测**（vLLM P-EAGLE blog，1×B200，GPT-OSS-20B，**只变并发、其余口径全一致**）：

| Benchmark | c=1 加速 | c=64 加速 | 衰减 |
|---|---|---|---|
| MT-Bench | 1.55×（K=7） | 1.05×（K=5） | −32% |
| HumanEval | 1.55×（K=7） | 1.23×（K=7） | −21% |
| SPEED-Bench | 1.69×（K=7） | 1.25×（K=7） | −26% |

> 按铁律二，**这是本次调研里唯一一组七项口径全部一致、可以直接横比的官方数据**，应作为 `18` 篇的主表。
> 注意 MT-Bench 那行官方在 c=64 时把 K 从 7 降到了 5——**连官方自己都在高并发档调小 γ**。

#### 成因 C：接受率本身塌方（任务分布 / 草稿模型不匹配）

**HF `dynamic_speculation_lookahead` 博客的表**（RTX 4090，greedy，bs=1）：
- `Salesforce/codegen-6B-mono` + `codegen-350M-mono` 在固定 γ=5 下 **0.89×（真的变慢）**
- `Llama-3.1-8B` + `Llama-3.2-1B` 在固定 γ 下 **1.00×（完全没收益）**
- 同样两组换成动态 γ 后分别是 1.09× 和 1.52×
→ **"投机没用"的结论里有相当一部分其实是"γ 没调"。**

**llama.cpp Discussion #10466**（RTX 4070，社区实测）：
> **With Gemma2 9B it drafted by Gemma2 2B it there is never any speculative decoding speedup.**
> **returns fall off rapidly as draft gets bigger, already questionable at 1.5B and not really useful at 3B draft.**
> **Small draft model is needed (sine qua non).**
极端可预测任务 176.53 tps（开）vs 61.83 tps（关）；随机数生成任务 63.1 tps（开/关无差别）。

**llama.cpp PR #10455**（同硬件同模型，只换 prompt，加速比差 4 倍）：
- 高可预测 prompt：3090 上 **5.73×**
- 短 prompt "write snake game in swift"：3090 上 **1.47×**

**SpecForge v0.3.0 博客关于训练数据的自认**：
> When dataset responses were written by humans or by a different model, the draft learns to continue text the target itself would rarely produce, and **acceptance saturates well below what the same draft reaches on target-generated data.**

**SGLang 官方示例里的真实接受长度**（K=4，上限 5）：MT-Bench **2.17**、GSM8K **2.64**。
**Red Hat gpt-oss 实测接受率随 K 衰减**：K=2 → 45.4%，K=3 → 35.6%，K=4 → 28.3%；对应平均接受长度 1.91 → 2.07 → **2.13**（边际几乎为零，但算力成本线性增长）。
**llama.cpp commit 自报**：gemma4-26b-a4b + DSpark **acceptance 0.46**；RedHat gemma-4-31b speculator **acceptance 0.26**。

**TensorRT-LLM 侧的接受长度实测**：
- blog12（guided decoding）：JSON Mode Eval 上 LLaMA 3.1 8B 的 EAGLE3 **AL = 2.86**、NGram **2.59**；LLaMA 3.3 70B EAGLE3 **2.72**、NGram **2.44**
- 1000 TPS/user blog：Llama 4 Maverick + EAGLE3，**Acceptance Length ≈ 2.0**（draft-length=3）
- Qwen3.8 部署指南：MTP3 的 **controlled accepted-draft count 2.3**
- blog07（NGram，Llama-4-Scout）：Magpie 第 1 轮 **AL 1.37**、第 2 轮 **AL 1.66**；**翻译任务上 v=23 时 AL 能到 4.73**
- issue [#9208](https://github.com/NVIDIA/TensorRT-LLM/issues/9208)：Qwen3-235B + EAGLE3，**BS 8 → AL 2.16；BS 16 → 2.14；BS 64 → 1.93**；另一数据集 **BS 1 → 1.96 vs BS 32 → 1.55**。NVIDIA **closed as not planned，未给根因解释**

> ⚠️ **#9208 揭示的是一个尚未被解释的现象**：接受长度本身随 batch 下降。这在数学上并不显然（每个请求的草稿质量理论上与 batch 无关）。**负收益因此可能是双重的——分子（接受长度）在跌，分母（验证成本）在涨。**
> 但因为官方没有给根因，本库应当把它记成**开放问题**而非规律。可能的候选解释（均未验证）：大 batch 下长请求占比高、上下文更长、chunk 边界效应、调度导致的草稿被截断。**建议在 `18-batch与吞吐` 明确写"此现象有报告但机理未获官方解释"。**

> **这组数字是本库 `05-接受率alpha` 的现实校准**：**真实生产环境里的平均接受长度普遍在 1.4–2.9 之间**（含 bonus token），远低于论文和宣传里的 4–6。少数高度可预测任务（翻译、极端重复）能到 4+。写正文时必须把"论文口径 α"与"引擎实测 accept_length"分开列。
> **同时注意任务分布的决定性**：blog12 里 NGram 在信息抽取任务上能逼近 EAGLE3（2.59 vs 2.86），换成非抽取任务就明显落后——官方原话「**it performs surprisingly well. This is because JSON Mode Eval is an information extraction task**」。

#### 成因 D：实现层的批处理缺陷（**不是投机解码的普遍规律**）

**HF `whisper-speculative-decoding` 博客**：
> For batched speculative decoding, **all candidate tokens across the batch must match the validation tokens in order for the tokens to be accepted. If a token in the batch at a given position does not agree, all candidate tokens that proceed the position are discarded.** Consequently, speculative decoding favours lower batch sizes. In practice, we find that speculative decoding provides a speed-up until a batch size of 4. **Above batch size 4, speculative decoding returns slower inference than the main model alone.**

> ⚠️ **必须点破**：这个 batch>4 的阈值是 **HF 自己朴素实现的产物**——"一个序列失配拖累全批"。vLLM/SGLang 用 per-request 的 ragged/padded 验证（`disable_padded_drafter_batch`、ragged verify），不存在这个问题，阈值远高于 4。
> **把"batch>4 就变慢"当成投机解码的普遍定律是错的**，这是本库 `27-常见误解与判据` 该收录的一条。

#### 成因 E：伪负收益（配置被引擎悄悄改了）

- **SGLang**：开投机后 `--max-running-requests` 未设时被**强制置为 48**（日志：「Max running requests is reset to 48 for speculative decoding.」）。**很多"开投机后吞吐变差"的实测，真实原因是并发上限被压到了 48。**
- **SGLang**：开投机后 `--enable-mixed-chunk` 被强制关闭。
- **SGLang**：overlap scheduler 只支持 `topk=1`，而 Llama 的自动值是 `topk=4`，文档亲口承认这个组合「**may not always trigger an immediate config error**」。
- **llama.cpp**：只写 `-md` 不写 `--spec-type draft-simple` → 草稿模型进显存但**永远不被调用**（不报错）。
- **vLLM**：DeepSeek-V3.2 的 MTP 强制 `enforce_eager=True` → **失去 CUDA graph**，这部分损失会被算到"投机的账"上。
- **vLLM**：开投机 + 异步调度会强制关掉 cascade attention → 长共享前缀场景的收益丢失。
- **SGLang**：开 EAGLE 后社区实测 radix prefix 复用率 97% → 40~53%（#32459），输入吞吐掉到 1/3。
- **TensorRT-LLM**：NGram / user-provided 会被**静默关掉 overlap scheduler**（与官方兼容矩阵自相矛盾）；接受率熔断是**永久性**的（`SpeculationGate` 一旦触发不再打开）；官方每份 EAGLE3 recipe 都写 `enable_block_reuse: false`——**前缀复用被主动关掉的损失会被算到"投机的账"上**。
- **TensorRT-LLM（口径污染）**：Llama 4 Maverick 那条 "4×" 是**全部软件优化的合计**，投机单独贡献只有 **2.06×**；DFlash 那条 "15×" 是 **vs 自回归**，而 **vs EAGLE-3 只有 1.5×**。媒体转载普遍把大数字记在投机头上。

> **成因 E 是本库能提供的最高附加值**：市面上讲负收益的文章几乎都只讲 A/B/C，**没人系统整理过"引擎悄悄改了你的配置"这一类**。写 `19-负收益全解` 时应当把 E 单独列成一节，并给出**排查清单**：开投机前后必须对比引擎启动日志里的 `max_running_requests` / `cudagraph_mode` / `enable_mixed_chunk` / `cascade attention` / `prefix cache hit rate` 五项。

#### 成因 F：负收益之外——"看起来在跑，其实没有收益"

**SGLang DeepSeek-V4 cookbook 的 `<Warning>`**（本次调研里最阴险的一条）：
> **Do not use EAGLE on the checkpoints that bundle a DSpark head.** On 0813, `--speculative-algorithm EAGLE` starts and serves without any error, but the draft head it binds accepts nothing — every decode batch logs `accept len: 1.00, accept rate: 0.00`, so **you pay the draft cost for zero speedup. Output stays correct, which is what makes it easy to miss.**

> → **排查判据：开了投机就必须看 accept length。`accept len == 1.00` 意味着一个草稿都没被接受，纯亏。** 四家引擎都提供了这个指标（vLLM 的 `mean_acceptance_length`、SGLang 的 `accept len`、llama.cpp 的 `draft_n_accepted / draft_n`、TensorRT-LLM 的 `acceptance_length`），**没看这个数字就报"投机有/没有用"的结论一律不成立。**

#### 成因 G：**边界语义被块内接受跨过去**（跨引擎结构性风险，不是某家的疏忽）

一次接受多个 token 时，"按 token 边界触发"的逻辑容易被跳过。四家都有同类 bug：

| 引擎 | issue | 现象 |
|---|---|---|
| HF transformers | [#47912](https://github.com/huggingface/transformers/issues/47912) | **Assisted/speculative decoding generates past an EOS that is accepted mid-block** |
| HF transformers | [#48039](https://github.com/huggingface/transformers/issues/48039) | **ignores `stop_strings` completed mid-block** |
| TensorRT-LLM | [#16377](https://github.com/NVIDIA/TensorRT-LLM/issues/16377)（Open） | **Eagle3 silently drops `tool_calls` for gpt-oss on trtllm-serve**——HTTP 200 + `finish_reason:"stop"` + **空 `tool_calls`**，但 `completion_tokens: 37` 且 `reasoning_content` 显示模型确实想调工具。**关掉投机、其他不变，tool calling 就恢复正常** |
| TensorRT-LLM | [#17437](https://github.com/NVIDIA/TensorRT-LLM/issues/17437)（Open） | **Interior control-token rejection can split a tool-call prelude and silently drop tool_calls** |
| SGLang | [#31978](https://github.com/sgl-project/sglang/issues/31978)（Closed） | **Multi-layer EAGLE verify ignores grammar vocab mask, aborting structured output** |

> ⭐ **这是一条本库能提供的原创归纳**：**投机解码的正确性风险不止在概率分布上，还在于所有"按 token 边界触发"的逻辑**——EOS、stop string、control token、tool-call 分隔符、grammar 状态机转移。
> 机理：非投机解码是"生成一个 token → 检查一次停止条件"；投机解码是"一次提交 k 个 token → 再检查"。**任何在这 k 个 token 中间应当触发的边界事件，都要靠实现方额外写逻辑去回溯，写漏就是静默错误。**
> **判据（可写进 27 篇）**：凡是用了 stop string / 结构化输出 / tool calling / 自定义停止条件的生产系统，开投机前必须做一次**开关对拍**，不能只看 token 吞吐。

### 7.3 反向证据（必须一起写，否则是选择性引用）

**Red Hat 2026-04-16（vLLM v0.13.0 + gpt-oss-120b MoE/MXFP4 + EAGLE3，H200 单卡）**：
> Crucially, **these gains persist up to 200 concurrent requests, contradicting the conventional expectation that speculative decoding majorly helps in low-QPS scenarios.**

**llama.cpp 维护者 ggerganov（PR #10455）**：
> The speedup can vary in both ways based on the inputs, but **enabling speculative should almost never result in slower than normal decoding.**
（⚠️ 同一 PR 的实测表 A 里三台机器全部变慢，见 §4.10。）

> **正确的写法**：不要写成"投机在大 batch 下一定变慢"，而要写成——
> **"投机是一笔用算力换延迟的交易。这笔交易在 GPU 仍是 memory-bound 时接近免费，在 GPU 饱和后按 (1−α)·K 的比例亏损。判断它划不划算，看的不是并发数，而是三个可测量：① 该模型在该负载下 GPU 是否已 compute-bound；② 实测平均接受长度是多少；③ 引擎有没有因为开投机而悄悄改掉别的配置。"**

### 7.4 各引擎「自动降级 / 自动关闭」机制对照

这是"官方承认负收益"最硬的证据——**它们把结论写进了代码**：

| 引擎 | 触发条件 | 动作 |
|---|---|---|
| vLLM | `num_speculative_tokens_per_batch_size` 命中 K=0 区间 | 不产草稿 |
| vLLM | data parallel > 1 + Dynamic SD | 自动禁用 Dynamic SD，回退静态 K（防 DP 死锁） |
| vLLM | Dynamic SD + 非 V2 runner + full cudagraph | cudagraph_mode 降级为 PIECEWISE |
| vLLM | 非 EAGLE 系 method + 未显式开 async scheduling | 自动关闭 async scheduling |
| vLLM | 开投机 + async scheduling | 自动关闭 cascade attention |
| vLLM | 开投机 | 自动关闭 `fast_moe_cold_start` |
| SGLang | 开投机且未设 `--max-running-requests` | **强制置 48** |
| SGLang | 开投机 | **强制关闭** `--enable-mixed-chunk` |
| SGLang | `--speculative-adaptive` 且 BS≥32 | **锁 `steps=1`** |
| SGLang | adaptive 条件不满足（非 EAGLE/EAGLE3 或 topk≠1） | falls back to static speculative settings |
| SGLang | `topk=1` | `num_draft_tokens` 强制改写为 `num_steps+1` |
| **TensorRT-LLM** | `decoding_type: AUTO` | 选 NGram，并**在 BS≥32 时关闭**（`max_concurrency=32` 硬编码） |
| **TensorRT-LLM** | 滚动接受率 < `acceptance_rate_threshold` | **永久关闭投机**（`SpeculationGate`，不会再打开） |
| **TensorRT-LLM** | 非 one-model 系 method（NGram / user-provided） | **静默关闭 overlap scheduler**（与官方兼容矩阵冲突） |
| **TensorRT-LLM** | 给了 `draft_len_schedule` | 自动置 `cuda_graph_config.enable_padding = True` |
| **TensorRT-LLM** | disaggregated serving | 自动置 `_allow_separate_draft_kv_cache = False`（WAR nvbugs/5807902） |
| llama.cpp | 上下文不支持按序列回滚（`SEQ_RM_TYPE_NO`） | **完全禁用投机**（只打一条 WARN） |
| llama.cpp | `n_max` 超过草稿模型训练时的 block size | clamp 到 block size |
| llama.cpp | 同时开了 draftless 与 draft-model | **draftless 优先** |
| HF transformers | beam search + assisted 参数 | **静默降级**为普通 beam search（只 warning） |
| HF transformers | 草稿自信度 < `assistant_confidence_threshold`（默认 0.4） | 提前结束本轮草稿 |

---

## 7b. ⭐ 跨引擎发现之一：四家引擎近期都在"无损性"上翻过车

这是本次调研**最意外、也最有本库特色的一条**：`04-拒绝采样修正-无损性的完整证明`、`07-无损的三种口径`、`27-常见误解与判据` 三篇都该引用。

| 引擎 | issue / 现象 | 日期 | 根因 |
|---|---|---|---|
| **HF transformers** | [#47932](https://github.com/huggingface/transformers/issues/47932) *candidate generators return a `q` they did not sample from, breaking losslessness with `do_sample=True`* | 2026-08-12 | DFlash / MTP / SinglePositionMultiToken 三个生成器**从"已施加 logits processor 的分布"采样，却把 raw logits 当作 $q$ 返回**给 `_speculative_sampling`。`p/q` 大了一个 $1/Z$ 因子 → 接受过多。实测：`top_k=50` + 49,152 词表下，**接受比膨胀 p95 达 3.02×，与目标分布的总变差最大 0.168** |
| **SGLang** | [#35771](https://github.com/sgl-project/sglang/issues/35771) *Target-only speculative sampler can accept a zero-probability draft at RNG boundary* | 2026-08-21 | 草稿提议了一个被目标 top-k/top-p 滤掉的 token（$p=0$），此时 `prob_acc == 0`；若 RNG 恰好返回 0，`0 <= 0` 成立 → **接受了一个目标概率为零的 token**。波及 **EAGLE / NEXTN / MTP / DFlash**（都走同一个 target-only sampler） |
| **llama.cpp** | [#25618](https://github.com/ggml-org/llama.cpp/issues/25618) *Speculative decoding (draft-mtp / draft-dspark): greedy output diverges from vanilla on quantized targets* | 更新于 2026-08-22，Open | **量化目标模型**下贪心输出与不开投机时不一致——连 L2 贪心等价都破了 |
| **TensorRT-LLM** | [#10309](https://github.com/NVIDIA/TensorRT-LLM/issues/10309) *Qwen3 + Eagle3 generates different result with greedy decoding compared to "no eagle version"* | 2025-12-26，**Open** | 报告者原话 "**Eagle3 should NOT change the output when we do greedy decoding.**" 输出开头一致、中途发散；怀疑 EAGLE 的并行验证 kernel |
| **TensorRT-LLM** | [#14825](https://github.com/NVIDIA/TensorRT-LLM/issues/14825) *kimi 2-6 nvfp4 + SpecD gives low accuracy on GPQA Diamond* | 2026-08-21，Closed | **NVFP4 量化 + 投机**下评测掉分 |

**共同点**：

1. **绝大多数发生在 2025–2026 新加的方法上**（DFlash / DSpark / MTP / EAGLE3 / target-only 路径），不是经典 draft-model 路径。**新方法在"更快"上竞赛，"无损"这一条最先被牺牲。**
2. **全部是静默的**——输出看起来正常，不报错、不崩溃。SGLang 的 DeepSeek-V4 cookbook 那句话可以直接套用到全部四家：「**Output stays correct, which is what makes it easy to miss.**」
2b. **低比特量化是一个反复出现的共同因素**：llama.cpp #25618（量化 target）、TRT-LLM #14825（NVFP4）都指向"量化 + 投机"。合理猜测（**未获官方确认，勿当结论写**）：量化把 target 的 logits 分布压平/抖动，使 draft 与 target 在边界 token 上的排序更容易翻转；贪心路径尤其敏感，因为它靠 argmax 严格相等来判定接受。**这条应写成"开放问题"而非结论。**
3. **HF #47932 给出的方法论教训对本库 `_lab/` 有直接指导意义**（逐字）：
   > Existing tests call `_speculative_sampling` directly with hand-crafted logits, **bypassing the generator-verifier contract where the bug resides.**
   → **测 `_speculative_sampling(p, q)` 函数本身永远测不出"传进来的 q 根本不是采样时用的那个 q"。** 按本库铁律四写 `_lab/test_lossless.py` 时，必须有一个测试是**跨越 proposer 与 verifier 的契约边界**的端到端测试，而不只是验证拒绝采样公式本身。

**写正文时的措辞建议**：
> "投机采样在数学上是无损的（L1）"这句话是对的，但**它描述的是算法，不是你正在运行的那个引擎**。2026 年 8 月同一个月内，HF transformers、SGLang、llama.cpp 分别报出了各自的无损性破损 bug（TensorRT-LLM 的 #10309 自 2025-12 起仍 Open），全部出现在近两年新加的草稿方法上，全部是静默的。
> 判据：**声称"我们的投机解码是无损的"必须配一个可复跑的等价性测试**。llama.cpp 在这一点上做得最好——`test_speculative.py::test_with_and_without_draft` 用固定 seed 断言开关 draft 的 token 序列逐位相同，且 `test_different_draft_min_draft_max()` 遍历五组 (n_min, n_max) 断言内容不变。

**vLLM 的对应测试目录**（`tests/v1/spec_decode/` @ v0.27.1，实测列出）——可作为本库 `_lab/` 的测试清单参照：
```
test_acceptance_length.py          test_backup_token_async_spec.py    test_dflash_causality.py
test_dflash_lookahead.py           test_dynamic_sd.py                 test_dynamic_sd_cug.py
test_eagle.py                      test_eagle_draft_attn_metadata.py  test_eagle_step_kernel.py
test_extract_hidden_states.py      test_llm_base_proposer.py          test_llm_base_proposer_sampling.py
test_max_len.py                    test_mtp.py                        test_mtp_structured_output.py
test_ngram.py                      test_rejection_sampler_utils.py    test_speculators_correctness.py
test_speculators_eagle3.py         test_synthetic_rejection_sampler_utils.py  test_vocab_mapping.py
```
> 值得注意的三个文件名：`test_speculators_correctness.py`（正确性专测）、`test_mtp_structured_output.py`（**结构化输出 × MTP 专测**，说明 §7.2 成因 G 那类 bug 已被纳入回归）、`test_dflash_causality.py`（**因果性专测**——块并行草稿最容易出的错就是把未来信息泄漏进当前位置）。
> 对照 SGLang 的 issue #28135「RFC: Robust End-to-End CI/CD Regression Testing for Speculative Decoding」——**SGLang 自己承认投机解码的回归测试此前覆盖不足**。**两家在测试成熟度上有可见差距。**

---

## 7c. ⭐⭐ 跨引擎发现之二：**四个生产引擎的默认验证方式都不是教科书上的拒绝采样**

这是本次调研信息密度最高、也最反直觉的一条，`04-拒绝采样修正-无损性的完整证明`、`07-无损的三种口径`、`27-常见误解与判据` 三篇必写。

| 引擎 | 默认验证方式 | 怎样才能切到经典 Leviathan Alg.1 |
|---|---|---|
| **vLLM** | `rejection_sample_method="standard"` **+ `draft_sample_method="greedy"`（默认）**——草稿概率被当作 one-hot | 设 `draft_sample_method="probabilistic"`（"uses the full draft logits for the probability ratio test... **at the cost of additional GPU memory usage**"） |
| **SGLang** | `tree_speculative_sampling_target_only`，且 **`draft_probs = torch.zeros_like(target_probs)`（草稿分布直接丢弃）** | `--speculative-use-rejection-sampling`（默认 `False`），且要求 topk=1、算法为 EAGLE/EAGLE3、无 accept threshold、无 deterministic inference |
| **TensorRT-LLM** | `use_rejection_sampling=False`（默认）→ **"exact-match verification"** | `use_rejection_sampling=True`（prototype），且不能是 NGram/SA、不能开 guided decoding、不能 CP>1、`dynamic_tree_max_topK ≤ 32`；非 dynamic-tree 的 one-model 路径**需要 FlashInfer** |
| **llama.cpp** | `common_sampler_sample_and_accept_n`——每位从 target 完整 sampler chain 采一个，`draft[i] != id` 即停 | **没有这个选项**（从未实现拒绝采样） |
| **HF transformers** | ✅ **`assistant_model` + `do_sample=True` 时默认就是 Alg.1**（`_speculative_sampling`）；prompt lookup 与贪心路径退化为相等即接受 | 默认即是 |

**结论**：**五家里只有 HF transformers 在标准配置下默认跑教科书上的拒绝采样；四个生产引擎的默认路径都是"相等即接受"的保守变体。**

### 为什么生产引擎都选"相等即接受"

从源码与文档里能读出三条理由：

1. **不需要草稿模型的概率 $q$。** TensorRT-LLM 把这条说得最直白——它拒绝对 NGram 开 rejection sampling 的理由是：
   > retrieval-based drafters that emit only token ids with **no q** are excluded
   **无模型草稿（n-gram / prompt lookup / suffix / 后缀自动机）在数学上根本无法参与 $\min(1, p/q)$ 判据，因为它压根没有 $q$。它们只能"相等即接受"。**
2. **省显存与带宽。** 保留完整 draft logits 才能算概率比；vLLM 的 `draft_sample_method="probabilistic"` docstring 明说 "**This comes at the cost of additional GPU memory usage.**" 在 15 万词表 × K 个位置 × batch 的规模下，这不是小数目。
3. **与任意 sampler chain 天然兼容。** llama.cpp 的写法最典型：每个位置都调完整的 `common_sampler_sample()`（含 temperature / top-k / top-p / min-p / DRY / XTC / mirostat / grammar），然后只比 token id 是否相等。**这样不管用户开了什么采样器，无损性都自动成立**——而拒绝采样需要 $p$ 和 $q$ 定义在同一个（被过滤后的）分布上，与每种 sampler 的交互都要单独处理。

### 这条结论的三个推论

**推论一（口径）**：**"相等即接受"是 L1 无损的**（输出每个 token 都是从目标分布采出来的），但**接受率严格低于**拒绝采样——因为 Leviathan 的 $\min(1,p/q)$ 在 $p \ge q$ 时必接受，且 $p<q$ 时仍以概率 $p/q$ 接受，而"相等即接受"要求采样结果恰好落在草稿那一个 token 上。
→ **同样的草稿质量下，四个生产引擎的理论加速比低于论文上界。** 这解释了 §7.2 成因 C 里"真实 accept_length 普遍只有 1.4–2.9"的一部分——**不全是草稿不好，也有验证判据更严的因素。**

**推论二（对本库 `_lab/` 的要求）**：`_lab/spec.py` 的参考实现如果只写 Leviathan Alg.1，那它**不能代表任何一个生产引擎的默认行为**。建议在 `_lab/` 里同时实现两种接受判据并对比 $E[\tau]$：
- `accept_rejection(p, q, x)` —— $\min(1, p(x)/q(x))$
- `accept_exact_match(p, q)` —— 从 $p$ 采一个，与草稿比是否相等
两者在 $T=0$（贪心）下等价，在 $T>0$ 下前者的 $E[\tau]$ 严格更大。**这个对比正好可以做成一个测试，验证"引擎默认比论文保守"这条断言。**

**推论三（社区误解）**：社区讲解里常见的"接受判据写成 `p(x) > q(x) 就接受`"（写作规范 §4 已列为高频错误）之所以流传，可能部分因为**大家读的是引擎代码而不是论文**——而引擎代码里确实没有 $\min(1,p/q)$。**评点社区材料时可以指出：它错在把"某引擎的工程简化"当成了"算法本身"，而正确的说法是"两者都无损，但判据不同、接受率不同"。**

### 附：各家提供的"主动放弃无损"开关（L3 入口）

| 引擎 | 开关 | 默认 | 文档是否说明这会改变分布 |
|---|---|---|---|
| **HF transformers** | `assistant_ensemble_weight`（接受判据换成 `w·p + (1-w)·q`，DIVERSE arXiv:2604.07622（2026-04）） | `None`（= lossless） | ✅ **说得最清楚**："trading a controlled distributional bias for a higher acceptance rate"、"Use `None` for standard (lossless) speculative decoding" |
| **TensorRT-LLM** | MTP 的 `use_relaxed_acceptance_for_thinking` + `relaxed_topk` + `relaxed_delta` | `False` / 1 / 0.0 | ✅ 说了 "only a slight accuracy drop"，且限定"仅 thinking 阶段、仅 DeepSeek-R1" |
| **TensorRT-LLM** | EAGLE 的 `greedy_sampling=False` + `posterior_threshold`（Medusa 式 typical acceptance） | `greedy_sampling=True` | ⚠️ 只说是 typical acceptance，未标注有损 |
| **SGLang** | `--speculative-accept-threshold-single` / `--speculative-accept-threshold-acc` | 均为 `1.0`（严格） | ❌ **只说 "Lower values accept more aggressively"，不说这会改变输出分布** |
| **vLLM** | `rejection_sample_method="synthetic"` + `synthetic_acceptance_rates` / `synthetic_acceptance_length` | `"standard"` | ⚠️ 说是"按标定的接受率接受"，用途是性能建模；**未显式标注它非无损**（但从定义上显然是） |
| **vLLM（V0，已删）** | `acceptance_method="typical_acceptance_sampler"` + `posterior_threshold=0.09` / `posterior_alpha=0.3` | `"rejection_sampler"` | — |
| **llama.cpp** | 无 | — | — |

> **写 `07-无损的三种口径` 时，这张表就是 L3 的完整清单。**
> **诚实度排序：HF transformers > TensorRT-LLM > vLLM > SGLang。** HF 在字段 description 里直接写了 "lossless"/"distributional bias" 两个词并给了论文出处；SGLang 的 accept threshold 则完全没提分布会变。**这个对比本身值得在 `26-社区精彩解释精选-好在哪与错在哪` 里做一次"文档诚实度"评点。**

---

## 8. 本调研可以喂给哪些篇（取用索引）

| 本库篇目 | 从本调研拿什么 |
|---|---|
| `03-并行验证为什么几乎免费` | TensorRT-LLM legacy 文档的"两条前提假设"原文（假设 1 = 并行验证是否真的免费）；blog02 里 **bs=1 下 CUDA graph 收益 7.22× 远大于 MTP3 的 2.16×**；HF blog 的 batch1 vs batch64 = 418 vs 16266 tok/s |
| `04-拒绝采样修正` | **§7c**：四个生产引擎默认都不是 Leviathan Alg.1，而是"相等即接受"；HF `utils.py` 里 Case 1 / Case 2 并排的那段注释 |
| `05-接受率alpha-定义口径与怎么测` | vLLM 的官方公式 `acceptance_length = 1 + num_accepted/num_drafts`（**含 bonus token**）；四家统一的四元组指标口径（drafts / draft_tokens / accepted_tokens / accepted_per_pos）；TensorRT-LLM 的**按迭代数加权**口径；真实 accept_length 普遍 1.4–2.9 的现实校准表 |
| `06-期望接受长度与加速比模型` | **DSpark blog 的逐位置接受率：第 1 位 >70%，第 7 位 <10%**（与几何模型 $0.7^7≈0.082$ 高度吻合）；Red Hat gpt-oss 的 K=2/3/4 → AL 1.91/2.07/2.13 边际衰减 |
| `07-无损的三种口径` | vLLM 的官方三层表述（Theoretical / Algorithmic / Logprob Stability）；**§7b 四家的无损性 bug**；**§7c 附表：各家的 L3 开关与文档诚实度排序** |
| `11-无模型草稿` | TensorRT-LLM 拒绝对 NGram 开 rejection sampling 的理由（"**no q**"）；blog07 的 Magpie 第1轮 AL 1.37 → 第2轮 1.66、翻译任务 AL 4.73；blog12 里 NGram 在信息抽取任务上逼近 EAGLE3（2.59 vs 2.86）；llama.cpp 的 5 种 n-gram 变体与"跨 slot 共享 hash pool" |
| `12-Medusa` | **Medusa 已出局的三家证据**（SGLang 零实现、llama.cpp 无、vLLM 有代码无文档、TensorRT-LLM 随 TRT engine 一并删除）；ModelOpt 还能训但 TRT-LLM 已推不了的"训推脱节" |
| `13-EAGLE三代` | EAGLE 3.1 的 attention drift 修正与"对 chat template / system prompt 变化更鲁棒"（反推出 EAGLE-3 的脆弱点）；vLLM `_maybe_override_draft_max_position_embeddings`（#48894 rotary cache 越界） |
| `14-MTP` | vLLM 把 22 个 `*_mtp` 别名折叠成 `mtp`；SGLang 的 NEXTN=EAGLE 别名（**MTP 在 SGLang 里不是独立算法**）；TRT-LLM `num_nextn_predict_layers` 废弃改用 `max_draft_len`；`num_speculative_tokens > 1` 会降接受率的官方 warning |
| `15-谱系图与被淘汰的分支` | **ReDrafter / Lookahead 随 TensorRT-LLM 1.2 的 TRT engine 后端一起被删除**——唯一实现过它们的生产引擎；`§0b` 的 2026 新家族速查表 |
| `16-树形草稿与树注意力` | TensorRT-LLM 把兼容性矩阵**从"按方法"改成"按拓扑（Linear / Dynamic Trees / Legacy）"**；`eagle_choices` 静态树将于 1.4 移除、改用 dynamic tree；dynamic tree 不支持 MLA/sliding window；rejection kernel 的 "32 tried siblings per level" 上限 |
| `17-动态草稿长度与自适应停止` | HF `dynamic_speculation_lookahead` 那张表（**同一对模型固定 γ 是 1.00×，动态 γ 是 1.52×**）；`assistant_confidence_threshold=0.4` 默认；vLLM Dynamic SD / adaptive verification；SGLang adaptive；TRT-LLM 三层门控 + SpeculationGate；**llama.cpp 默认 n_max 从 16 降到 3** |
| `18-batch与吞吐-收益衰减曲线` | **主表 = vLLM P-EAGLE blog 的 c=1 vs c=64 三组（唯一一组七项口径全一致、可直接横比的官方数据）**；`effective BS = BS×K`（vLLM）= `verification_batch = original_batch×(v+1)`（TRT-LLM）双引擎背书；四家临界点都在 BS 32~128；TRT-LLM audit 表显示拐点随 ISL/OSL 浮动 8 倍；issue #9208 的"接受长度本身随 batch 下降"（**开放问题，官方未解释**） |
| `19-负收益全解` | **整个 §7**。开篇建议用 vLLM 官方 blog 那条 "1.4× / 1.8× slowdown"（标题写 2.8× 加速、正文写同模型同硬件的减速）；成因 A–G 七类；成因 E（伪负收益）与成因 G（边界语义）是本库相对市面文章的增量 |
| `20-主流引擎实现` | 全篇。建议按"三层"组织：HF transformers = 单流参考实现 / llama.cpp = 本地 + 有限并发 / vLLM·SGLang·TensorRT-LLM = 服务化吞吐。**同一个"投机解码"在三层里是三种不同的工程问题** |
| `21-草稿模型怎么训` | 四个训练框架并列（vLLM `speculators` / SGLang `SpecForge` / NVIDIA `ModelOpt` / `TorchSpec`）；三家都做了"用推理引擎 dump hidden states"（`extract_hidden_states` / `SaveState` / SpecForge offline / `compute_hidden_states_trtllm.py`）；**ModelOpt 原文"用 base model 自己生成的对话做训练数据接受率更高"**；SpecForge 自认"人写或别的模型写的回复会让接受率饱和在低位"；offline 模式要"several to tens of terabytes"磁盘 |
| `22-长上下文下的投机采样` | **唯一一组"AL vs 上下文长度"实测：MiniMax-M3 从 1K 到 32K，AL 2.69 → 2.65（基本不衰减）**，与 EAGLE 3.1 blog 声称的 attention drift 存在张力，需并列呈现；TRT-LLM audit 表里 ISL/OSL=8k-1k 的拐点只有 concurrency 64（vs 1k-8k 的 512）；vLLM 的 EAGLE rotary cache 越界（#48894，draft 的 `max_position_embeddings` 只有 2048） |
| `23-与其它优化的相互作用` | vLLM 兼容矩阵 + 源码里 11 条硬规则（含 V1/V2 model runner 分野）；SGLang 的"不兼容 XXX"断言表；TRT-LLM 兼容矩阵 + 代码门禁；**PD 分离 × 投机只有 SGLang 与 TRT-LLM 有官方素材，且两家做法一致（两侧都配 spec、都关前缀复用）**；LoRA × 投机三家都没真正解决；量化 × 投机的无损性风险 |
| `24-前沿进展-2025到2026` | §0b 新家族速查表；§1.11 vLLM blog 时间线；SpecForge News 时间线；EAGLE 3.1 / DFlash / DSpark / PARD 的论文号 |
| `26-社区精彩解释精选` | **"文档诚实度"评点**：HF 在 `assistant_ensemble_weight` 里直写 lossless/bias vs SGLang 的 accept threshold 完全不提分布会变；`SGLANG_SIMULATE_ACC_LEN` 与 vLLM `synthetic` 模式——**两家都提供了"假装接受率是 X"的基准模式且都出现在官方数字里** |
| `27-常见误解与判据` | "batch>4 就变慢"是 HF 实现缺陷不是普遍定律；"开投机就没 logprobs"曾真现假（V0→V1）；"投机=Leviathan Alg.1"在四个生产引擎上不成立；`accept len == 1.00` 的自查判据；开投机前后必查的五项配置漂移 |
| `28-面试题库` | 好题源：①为什么 n-gram 不能做拒绝采样？②为什么 EAGLE 与"prefill 只算最后一个位置"的优化冲突？③为什么 SSM/Mamba 类模型开不了投机？④`effective BS = BS×K` 的拐点怎么估？⑤开了投机吞吐反而降，列出五种可能原因 |
| `_lab/` | §7c 推论二：同时实现 `accept_rejection` 与 `accept_exact_match` 并对比 $E[\tau]$；用 DSpark 的 70%/&lt;10% 校准几何模型；HF #47932 的教训——**测试必须跨越 proposer/verifier 契约边界** |

---

## 9. 附：取证方法备忘

- GitHub 原始文件用 `raw.githubusercontent.com/<org>/<repo>/<tag>/<path>` 直取，**带 tag 而不是 `main`**，否则复核时内容已变。
- 目录清单用 `https://api.github.com/repos/<org>/<repo>/contents/<dir>?ref=<tag>`（匿名有速率限制，`/search/code` 需要认证会 401）。
- 文档渲染页经小模型摘要后**会丢失默认值和 admonition**，凡涉及默认值一律回源码 dataclass。本节 `prompt_lookup_min` 的「文档写 1、代码是 5」就是这么发现的。
