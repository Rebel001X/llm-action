# vLLM 投机解码

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：验证是精确重采样不改分布，"动态 K"是静态查表不是自适应。

## 0. 结论先行

- **proposer 不是一种，是一整个家族。**
  - `SpeculativeMethod`（`vllm/config/speculative.py:69-79`）枚举出 `ngram` / `medusa` / `mlp_speculator` / `draft_model` / `suffix` / `custom_class` 六个"顶层方法"，外加 `EagleModelTypes`（`eagle` / `eagle3` / `extract_hidden_states` / 24 种 MTP 模型类型 / `dflash`）和 `NgramGPUTypes`（`ngram_gpu`）、`DSparkModelTypes`（`dspark`）。
  - 会跑自己前向、需要管理自己 KV 缓存的一大类（EAGLE / EAGLE3 / MTP / DFlash / draft_model）共享同一个基类 `SpecDecodeBaseProposer`（`vllm/v1/spec_decode/llm_base_proposer.py:71`）。
  - 不需要模型前向的（`ngram`、`suffix`）是纯 CPU/GPU 核函数；`custom_class` 是用户自己注册的插件（`vllm/v1/spec_decode/custom_class_proposer.py:12-20`，从字符串路径动态 import）。

- **草稿生成的时机在 `sample_tokens()` 里，且排在"本步验证"之后，不在 `execute_model()` 里。**
  - `execute_model()` 只做目标模型前向并把结果暂存进 `self.execute_model_state`（`vllm/v1/worker/gpu_model_runner.py:4284` 起）。
  - `sample_tokens()`（`vllm/v1/worker/gpu_model_runner.py:4663`）先解包这份暂存态、套结构化输出 bitmask、跑 `_sample()`（内部按需调用拒绝采样器验证上一轮草稿），**再**调用 `propose_draft_token_ids()`（`vllm/v1/worker/gpu_model_runner.py:5116`）用刚采样出的真实 token 当锚点生成下一轮草稿。
  - 这与兄弟篇 [[06-vLLM-模型执行与CUDA-Graph]] 确认的"两次独立 RPC"结论完全对应，本篇把 spec decode 具体挂在哪一侧钉死。

- **草稿模型的 KV 缓存不是独立系统，是统一 KVCacheManager 里的一个独立分组（`kv_cache_gid`）。**
  - `SpecDecodeBaseProposer.initialize_attn_backend()` 在 `kv_cache_config.kv_cache_groups` 里找出草稿层所属的 group id 并记到 `self.kv_cache_gid`（`vllm/v1/spec_decode/llm_base_proposer.py:1744-1750`）。
  - 物理块池、前缀缓存、block table 的分配逻辑与目标模型共用同一套代码，只是逻辑上分组，不是另开一套内存管理系统。

- **验证算法是精确重采样，不是近似。**
  - `vllm/v1/sample/rejection_sampler.py` 的类文档直接写"implementation strictly follows the algorithm described in https://arxiv.org/abs/2211.17192"（`:40-41`，Leviathan et al. 2023）。
  - 非贪心路径用概率比检验 `target_prob/draft_prob >= uniform_prob`（`:829`）接受，拒绝时从残差分布 `max(target_prob - draft_prob, 0)` 重采样（`:926`）——这是原论文证明"输出分布严格等于只用目标模型采样"的那个精确构造，不是启发式近似。
  - 贪心路径（温度=0）更直接：`token_id = target_argmax_id` 永远赋值为目标模型的 argmax（`:751-756`），接受与否只影响"这一位是否提前截断"，输出内容与不做投机解码的贪心解码逐 token 位对位相同。
  - **例外**：`rejection_sample_method="synthetic"` 用配置的伪接受率替换真实概率比检验（专为压测/校准设计，故意打破分布保真），`"block"` 用 Sun et al. 2024（`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:596`，`arxiv 2403.10444`）的整块联合验证，两者都是显式 opt-in（默认 `"standard"`）。

- **"动态 K"是配置期算好的静态查表，不是运行时看接受率自适应。**
  - `num_speculative_tokens_per_batch_size`（`vllm/config/speculative.py:181-186`，默认 `None`）是用户手填的 `(batch_size_start, batch_size_end, K)` 区间表。
  - `Scheduler.__init__` 把它经 `build_dynamic_sd_schedule_lookup()`（`vllm/v1/spec_decode/dynamic/utils.py:77`）展开成一个按 batch size 直接下标的稠密数组 `self.dynamic_sd_lookup`（`vllm/v1/core/sched/scheduler.py:265-271`）；每步 `schedule()` 收尾处只做一次数组查表 `self.dynamic_sd_lookup[len(num_scheduled_tokens)]`（`vllm/v1/core/sched/scheduler.py:1264-1269`）。
  - **这张表里能不能出现 `K=0`（即高并发下自动关掉投机解码）取决于用户自己有没有在配置里填这一档**——vLLM 本身不提供默认表、不采集实测接受率来自动生成或调整它。
  - `SpecDecodingStats`（`vllm/v1/spec_decode/metrics.py:17-47`）收集的接受率纯粹用于 Prometheus 可观测性，没有反馈进任何控制回路。

- **打开投机解码后至少四处联动会变形，都有源码可指：**
  - 动态 K 会把 `cudagraph_mode` 从 FULL 强制降到 PIECEWISE（`vllm/config/vllm.py:967-984`）。
  - 动态 K 在数据并行（DP>1）下被直接禁用并回退成静态 K（`vllm/config/vllm.py:986-1002`，理由是"不同 DP rank 可能选出不同 K 导致 DP 分叉/死锁"）。
  - 结构化输出不是被禁用而是被"容纳"——xgrammar 的 `GrammarMatcher` 拿 `num_speculative_tokens` 当 `max_rollback_tokens`（`vllm/v1/structured_output/backend_xgrammar.py:73-76,127`），调度器在草稿产出后立刻用语法状态机 `validate_tokens()` 过滤草稿并把不合法位置填成 `-1` 占位符（`vllm/v1/core/sched/scheduler.py:2264,2286-2318`），这个 `-1` 正是两套拒绝采样核函数里"遇到 `-1` 直接判拒绝"分支的输入来源。
  - chunked prefill 与多模块 MTP drafter 冲突——调度器必须在 chunk 边界前保留至少 `num_prefill_lookahead` 个 token 不切断，否则 drafter 会读到还没算完的位置（`vllm/v1/core/sched/scheduler.py:463-481`）。

- **反直觉：高并发下投机解码常是负收益，但 vLLM 默认没有据此自动关闭的机制，SGLang 有。**
  - vLLM 唯一的杠杆是上面提到的、需要用户自己离线跑出来再手填的 `num_speculative_tokens_per_batch_size` 表；没配就没有这张表，K 恒定，不管 batch 多大都照样起草稿模型前向去抢已经吃满的 GPU。
  - SGLang 的 `speculative_adaptive`（同样默认 `False`）打开后是一套真正闭环的机制：按 batch size 分档（`1`/`8`/`32`/`64`），档位越高候选步数集合越小，最高档 `"64"` 的 `candidate_steps` 直接是 `[0]`（`sglang:python/sglang/srt/speculative/adaptive_spec_params.py:22-45`）。
  - 档内还用观测到的接受长度做 EMA + 滞回来微调（`sglang:python/sglang/srt/speculative/adaptive_spec_params.py:141-148,175-190`）——`sglang:python/sglang/srt/speculative/eagle_worker_v2.py:1206-1209` 的注释原话是 `"Drafting disabled (high batch size)"`。见 `## 6`、`## 7` 展开。

证据分级：本篇除标注"文档所述""本库推断"外均为「源码为证」，带 `文件:行`。不产出任何接受率/加速比实测数字（本机无 GPU，且接受率高度依赖模型对与数据分布）。

## 1. 它在系统里的位置

投机解码不是独立子系统，是嵌进已有主循环里的一段"多算一次、验一次"逻辑，触点分布在四个既有部件上：

1. **调度器**（详见 [[03-vLLM-调度器解剖]]）。`Scheduler.__init__` 在读到 `speculative_config` 时设 `self.use_eagle`、`self.num_spec_tokens`、`self.num_prefill_lookahead`，并按 `num_speculative_tokens_per_batch_size` 是否配置决定要不要建 `self.dynamic_sd_lookup`（`vllm/v1/core/sched/scheduler.py:250-276`）。`schedule()` 每步做两件与 spec decode 有关的事：把上一轮 drafter 产出的 `request.spec_token_ids` 截断进 `SchedulerOutput.scheduled_spec_decode_tokens`（`vllm/v1/core/sched/scheduler.py:711-727`，`SchedulerOutput` 定义见 `vllm/v1/core/sched/output.py:226`），以及在收尾处查表算出**这一步该让 drafter 提议几个草稿**、写进 `scheduler_output.num_spec_tokens_to_schedule`（`vllm/v1/core/sched/scheduler.py:1264-1269`，字段定义 `vllm/v1/core/sched/output.py:283`）。
2. **模型执行器 `GpuModelRunner`**（详见 [[06-vLLM-模型执行与CUDA-Graph]]）。它持有 `self.drafter`——一个按 `speculative_config.method` 实例化出来的具体 proposer 对象（`vllm/v1/worker/gpu_model_runner.py:632-695`）——并在 `execute_model()` / `sample_tokens()` 两次 RPC 里分别调用目标模型前向、拒绝采样验证、drafter 前向三件事，顺序细节见 `## 4`。
3. **采样器**。`RejectionSampler`（`vllm/v1/sample/rejection_sampler.py:38`）是 `Sampler`（详见 [[06-vLLM-模型执行与CUDA-Graph]] `## 4` 提及）之外的另一条采样路径，只在 `spec_decode_metadata is not None` 时启用（`vllm/v1/worker/gpu_model_runner.py:3766-3767`）。
4. **结构化输出**（详见 [[11-vLLM-结构化输出]]）与**KV 缓存管理**（详见 [[04-vLLM-KV缓存与前缀缓存]]）分别在 `## 5` 里各占一条：语法状态机要在草稿产出后立刻插一刀做合法性过滤，KV 缓存管理器要单开一个 lookahead 窗口容纳 drafter 提前读取的位置。
5. **可观测性**。`SpecDecodingStats`（`vllm/v1/spec_decode/metrics.py:17-47`）和转成 Prometheus 指标的 `SpecDecodingProm`（`:177`）是唯一对外暴露接受率的通道——它只朝外走（进日志、进 `/metrics`），不朝内走（不反馈进调度器或 proposer 的任何决策），这条"只出不进"的单向性质是 `## 7` 反直觉论证的关键前提之一。

一次典型的"多轮解码"里，投机解码把**串行的目标模型调用次数**换成了"目标模型验证一次多 token + 草稿模型（或非模型方法）提议若干 token"。这笔交换值不值得做、什么时候会亏本，是 `## 7` 的主题；它牵动的不只是模型执行器一处，`## 5` 会逐条展开与 CUDA Graph、结构化输出、chunked prefill、KV 缓存管理这四个既有部件的具体互动方式。

## 2. 代码地图（文件 → 职责，带行号）

`_lab/out/struct_map.json` 记录 spec decode 相关文件 64 个、共 18,360 行（`subsystems.speculative.n_files/lines`）。核心文件：

| 文件:行 | 职责 |
|---|---|
| `vllm/config/speculative.py:85` | `class SpeculativeConfig`，35 个字段（AST 统计），`method`/`num_speculative_tokens`/`rejection_sample_method`/`draft_sample_method`/`num_speculative_tokens_per_batch_size` 等旋钮全在这 |
| `vllm/config/speculative.py:69-79` | `SpeculativeMethod` / `EagleModelTypes` / `NgramGPUTypes` / `DSparkModelTypes` 枚举，proposer 全集的唯一权威出处 |
| `vllm/config/speculative.py:1504-1505` | `uses_dynamic_speculative_decoding()`——是否配了动态 K 表 |
| `vllm/v1/spec_decode/llm_base_proposer.py:71` | `class SpecDecodeBaseProposer`，EAGLE 系 proposer 的共享基类，35 方法 |
| `vllm/v1/spec_decode/eagle.py:10` | `class EagleProposer(SpecDecodeBaseProposer)` |
| `vllm/v1/spec_decode/draft_model.py:19` | `class DraftModelProposer(SpecDecodeBaseProposer)`——独立小模型式经典投机解码 |
| `vllm/v1/spec_decode/dflash.py:23` | `class DFlashProposer(SpecDecodeBaseProposer)`——半自回归并行起草 |
| `vllm/v1/spec_decode/gemma4.py:32` | `class Gemma4Proposer(SpecDecodeBaseProposer)`——与目标模型跨模型共享 KV |
| `vllm/v1/spec_decode/step3p5.py:24` | `class Step3p5MTPProposer(EagleProposer)`——按层选草稿步的 MTP 变体 |
| `vllm/v1/spec_decode/medusa.py:18` | `class MedusaProposer`——多头并行预测，不继承 `SpecDecodeBaseProposer`（不需要自己的 attention/KV） |
| `vllm/v1/spec_decode/suffix_decoding.py:9` | `class SuffixDecodingProposer`——后缀树匹配，纯 CPU，无神经网络 |
| `vllm/v1/spec_decode/extract_hidden_states.py:28` | `class ExtractHiddenStatesProposer`——固定 1 步 |
| `vllm/v1/spec_decode/ngram_proposer.py:12` | `class NgramProposer`——CPU/numba 版 prompt-lookup |
| `vllm/v1/spec_decode/ngram_proposer_gpu.py:217` | `class NgramProposerGPU`——同一思路的 GPU 核函数版 |
| `vllm/v1/spec_decode/custom_class_proposer.py:12-20` | `create_custom_proposer()`——从字符串路径动态 import 用户自定义 proposer |
| `vllm/v1/spec_decode/vocab_mapping.py:68` | `class VocabMapping`——`draft_model` 异构词表（`use_heterogeneous_vocab`）场景下做 draft/target 词表 token id 互译 |
| `vllm/v1/spec_decode/metadata.py:9-23` | `class SpecDecodeMetadata`——一步验证要用到的全部张量打包 |
| `vllm/v1/spec_decode/metrics.py:17-47` | `class SpecDecodingStats`——纯观测性接受率统计，不反馈进控制逻辑 |
| `vllm/v1/spec_decode/metrics.py:177` | `class SpecDecodingProm`——把 `SpecDecodingStats` 转成 Prometheus `Counter`，走 HTTP `/metrics` 端点暴露，仍然只是观测通道 |
| `vllm/v1/spec_decode/dynamic/utils.py:77` | `build_dynamic_sd_schedule_lookup()`——把用户配的区间表展开成稠密 batch_size→K 数组 |
| `vllm/v1/sample/rejection_sampler.py:38` | `class RejectionSampler(nn.Module)`——V1 主路径的拒绝采样器，类文档直接引用 Leviathan 2023 |
| `vllm/v1/sample/rejection_sampler.py:715` | `rejection_greedy_sample_kernel`——贪心路径，Triton kernel |
| `vllm/v1/sample/rejection_sampler.py:774` | `rejection_random_sample_kernel`——非贪心路径，概率比检验 |
| `vllm/v1/sample/rejection_sampler.py:873` | `sample_recovered_tokens_kernel`——拒绝后从残差分布重采样 |
| `vllm/v1/worker/gpu_model_runner.py:632-695` | `self.drafter = ...`——按 `method` 字符串分发到具体 proposer 类的大 `if/elif` |
| `vllm/v1/worker/gpu_model_runner.py:4284` | `execute_model()`——目标模型前向，产出暂存在 `execute_model_state` |
| `vllm/v1/worker/gpu_model_runner.py:4663` | `sample_tokens()`——验证 + 起草下一轮，两件事都在这 |
| `vllm/v1/worker/gpu_model_runner.py:5116` | `propose_draft_token_ids()`——按 `method` 分发到 `self.drafter.propose(...)` |
| `vllm/v1/core/sched/scheduler.py:1264-1269` | 调度收尾处计算 `num_spec_tokens_to_schedule`（动态 K 查表发生地） |
| `vllm/v1/core/sched/scheduler.py:2264` / `2286` | `update_draft_token_ids()` / `update_draft_token_ids_in_output()`——草稿产出后立刻过语法状态机 |
| `vllm/config/vllm.py:967-984` | `_maybe_override_dynamic_sd_cudagraph_mode()`——动态 K 强降 CUDA Graph 模式 |
| `vllm/config/vllm.py:986-1002` | `_maybe_disable_dynamic_sd_for_data_parallel()`——DP>1 下禁用动态 K |
| `vllm/v1/structured_output/backend_xgrammar.py:73-76,127` | `max_rollback_tokens=num_speculative_tokens`——xgrammar 对齐投机解码的机制 |
| `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:114` | `class AdaptiveVerificationManager`——DSpark 专用、按置信度动态分配草稿验证预算（不是本篇主线的"动态 K"） |
| `vllm/config/vllm.py:649` | `use_v2_model_runner` 属性——决定走 V1 (`vllm/v1/spec_decode/`) 还是 V2 (`vllm/v1/worker/gpu/spec_decode/`) proposer 实现 |
| `vllm/config/vllm.py:658-666` | `method=="dspark"` 强制走 V2 model runner 的判定，注释原话"V1 ... can't run dspark" |
| `vllm/config/vllm.py:702-711` | `_is_dflash2_draft()`——判定 DFlash2 checkpoint，未强制 V2 时会在 V1 静默降级成 DFlash1 |
| `vllm/v1/spec_decode/ngram_proposer.py:55-60` | `__init__` 里主动跑一次 dummy `propose()` 触发 Numba JIT 预热 |
| `vllm/v1/spec_decode/dynamic/utils.py:4` | `DynamicSDSchedule` 类型别名——`list[tuple[int,int,int]]`，动态 K 配置的规范形态 |
| `vllm/config/vllm.py:2555-2564` | `_validate_v2_model_runner()`——V1/V2 特性不匹配时直接抛异常，不静默回退 |

## 3. 核心数据结构

**`SpeculativeConfig`**（`vllm/config/speculative.py:85`，35 字段，AST 统计）是唯一入口，按用途分五组：

- **通用控制**：`num_speculative_tokens`、`method`、`model`（草稿模型/EAGLE head/额外权重路径）。
- **草稿模型相关**：`quantization`、`moe_backend`、`draft_tensor_parallel_size`、`draft_model_config`（`__post_init__` 里按 `method` 派生出来，见 `## 4`）。
- **验证相关**：`rejection_sample_method: Literal["standard","synthetic","block"]`（`vllm/config/speculative.py:216-224` 文档字符串）、`draft_sample_method: Literal["greedy","probabilistic"]`（`vllm/config/speculative.py:290-296`）。
- **动态 K**：`num_speculative_tokens_per_batch_size: list[tuple[int,int,int]] | None`（`vllm/config/speculative.py:181-186`）。
- **方法专属**：ngram 专属 `prompt_lookup_max` / `prompt_lookup_min`；suffix 专属 `suffix_decoding_max_tree_depth` 等四个字段；DSpark 专属 `enable_adaptive_verification`、`dspark_draft_topk`。

**`SpecDecodeMetadata`**（`vllm/v1/spec_decode/metadata.py:9-23`）是一步验证要喂给拒绝采样器的全部索引信息，字段全是"稀疏 token 序列 → 稠密张量位置"的映射：

- `draft_token_ids`（本步待验证的草稿 token，`[num_tokens]`）；
- `num_draft_tokens`/`cu_num_draft_tokens`（每请求草稿数与其前缀和）；
- `target_logits_indices`/`bonus_logits_indices`（草稿位置与 bonus 位置各自在 logits 里的下标）；
- `logits_indices`（两者拼起来）。

`__post_init__` 顺手算出 `self.max_spec_len = max(self.num_draft_tokens)`（`vllm/v1/spec_decode/metadata.py:26-27`），驱动输出缓冲区 `output_token_ids: [batch_size, max_spec_len+1]` 的形状。

**`self.drafter`**（`vllm/v1/worker/gpu_model_runner.py:632`）是一个"运行时多态"字段，声明为一个大 union 类型，实际类型由 `speculative_config.method` 决定，`if/elif` 链（`:632-695`）逐个方法分发到具体类。这意味着 `GpuModelRunner` 对所有 proposer 只依赖一份共同接口（至少 `propose()`），proposer 之间的实现差异被封在各自类内部——`## 0` 第一条列出的十几种方法从调用方视角看只是一个字段的不同取值。

**`SchedulerOutput.scheduled_spec_decode_tokens: dict[str, list[int]]`**（`vllm/v1/core/sched/output.py:226`）是调度器与 worker 之间关于"这一步要验证哪些草稿 token"的唯一契约；`num_spec_tokens_to_schedule: int`（`:283`）是反方向的契约——"这一步之后 drafter 应该提议几个新草稿"。两个字段的值都来自 `## 2` 提到的动态 K 查表，但作用的是**不同轮次**：前者用的是上一轮 drafter 的产出，后者决定的是这一轮 drafter 的产出量。这两个字段的关系是 `## 4` 走读的主线。

### 3.1 proposer 全家福：一句话原理 + 适用场景 + 额外权重

`SpeculativeMethod`（`vllm/config/speculative.py:69-79`）本身就是全部方法名的权威清单：

```python
# vllm/config/speculative.py:69-79
SpeculativeMethod = Literal[
    "ngram",
    "medusa",
    "mlp_speculator",
    "draft_model",
    "suffix",
    "custom_class",
    EagleModelTypes,
    NgramGPUTypes,
    DSparkModelTypes,
]
```

其中 `EagleModelTypes`（`vllm/config/speculative.py:66-68`）本身又是一个嵌套的 `Literal`，展开后包含 `"eagle"`、`"eagle3"`、`"extract_hidden_states"`、24 种 `MTPModelTypes`（`vllm/config/speculative.py:37-62`）、`"dflash"`——这也是为什么本篇反复用"proposer 全家福"而不是"proposer 类型"来描述这套体系：真正独立的方法名有十几个，但配置层面暴露出来的组合数远不止这十几个（每种 MTP 模型类型都是各自独立的字符串）。按 `method` 值逐个列全，`method` 列即配置里 `speculative_config.method` 的取值：

| `method` | 一句话原理 | 适用场景 | 需要什么额外权重 | proposer 类 |
|---|---|---|---|---|
| `ngram` | 在已生成的 prompt+输出 token 里做 n-gram 前缀匹配（prompt lookup），命中后把匹配串后面跟着的历史 token 直接当草稿 | 输出与输入高度重复的任务（代码编辑、摘要复述、检索增强生成里的抄录段） | 无，纯 CPU/numba，`vllm/v1/spec_decode/ngram_proposer.py:12` |
| `ngram_gpu` | 同 `ngram` 原理，改用 Triton/GPU 核函数实现 | 同上，但要求更高吞吐、不占用 CPU 线程 | 无，`vllm/v1/spec_decode/ngram_proposer_gpu.py:28,217` |
| `suffix` | 维护全局/单请求后缀树（复用 Arctic Inference 官方实现，`arxiv 2411.04975`），按频次估计概率，仅当估计概率 ≥ `suffix_decoding_min_token_prob` 才推测 | 多轮对话/批量请求间存在大量重复模式（客服、代码补全） | 无内置权重，但依赖 `arctic_inference` 第三方包，`vllm/v1/spec_decode/suffix_decoding.py:9` |
| `medusa` | 在目标模型最后一层 hidden state 上接若干并行预测头，每个头独立预测未来第 i 个位置 | 已有对应模型的 Medusa head 权重 | Medusa head 权重（远小于完整模型），`vllm/v1/spec_decode/medusa.py:18` |
| `mlp_speculator` | 按类型声明理应类似 Medusa，用 MLP 结构接预测头 | 未查证：本快照 `self.drafter` 分发链（`vllm/v1/worker/gpu_model_runner.py:632-703`）与 V2 speculator 目录均检索不到对应 proposer 类，只在 `vllm/transformers_utils/configs/mlp_speculator.py` 等配置解析文件里出现 | 未查证 |
| `draft_model` | 经典投机解码：独立的小号自回归模型逐 token 生成草稿 | 有一个更小、同分布的模型可用（如小模型配大模型），可选异构词表 | 完整独立小模型 checkpoint，`vllm/v1/spec_decode/draft_model.py:19` |
| `eagle` / `eagle3` | 用目标模型的 hidden state（EAGLE3 额外用 aux hidden states）接一个轻量自回归头逐步生成草稿 | 有对应模型的官方/社区 EAGLE head 权重 | EAGLE head（远小于完整模型），`vllm/v1/spec_decode/eagle.py:10` |
| `mtp`（24 种模型类型统一归并，见 `MTPModelTypes`） | 模型训练时自带 Multi-Token Prediction 模块，不是外挂投机头而是原生架构一部分，推理时复用该模块当 drafter | DeepSeek-V3、GLM-4-MoE 等原生带 MTP 的模型 | 通常已包含在模型 checkpoint 内，`vllm/config/speculative.py:37-62`（`MTPModelTypes` 枚举） |
| `dflash` | 半自回归并行起草：一次前向同时预测一个 block 内的多个位置，而非逐 token 自回归（"Only one forward pass due to parallel drafting"，`vllm/v1/spec_decode/dflash.py:233-235`） | 需要更高起草吞吐、能接受专门训练的 DFlash checkpoint | DFlash 专用 draft 权重，`vllm/v1/spec_decode/dflash.py:23` |
| `dspark` | 复用 DFlash 的并行 backbone，再叠加一个轻量 Markov 头修正块内顺序依赖（"reusing the DFlash machinery... injects intra-block dependency with a lightweight sequential Markov head"，`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:3-7`） | 同 DFlash 场景但要更高接受率 | DSpark 专用权重；**仅能在 V2 model runner 下运行**（`vllm/config/vllm.py:658-666`，见 `## 7`） |
| `extract_hidden_states` | 固定 1 步，直接从目标模型隐藏状态映射，不做多步自回归 | 只需要 1 个额外 token 的轻量场景 | 复用目标模型隐藏状态，无独立自回归权重，`vllm/v1/spec_decode/extract_hidden_states.py:28` |
| `gemma4_mtp` | Gemma4 assistant 模型每步跑全部 decoder 层生成 1 个 token，attention 层与目标模型跨模型共享 KV | Gemma4 系列模型专用 | Gemma4 assistant 模型权重，`vllm/v1/spec_decode/gemma4.py:32` |
| `step3p5_mtp` | 继承 `EagleProposer`，按层做草稿步选择（"per-layer draft-step selection"） | Step3.5 模型专用 | 模型自带的 MTP 权重，`vllm/v1/spec_decode/step3p5.py:24` |
| `custom_class` | 非内置算法，是插件机制：从字符串路径动态 import 用户自己实现的 proposer 类 | 研究/内部实验用的自定义投机策略 | 用户自行决定，`vllm/v1/spec_decode/custom_class_proposer.py:12-20` |

表里 `mlp_speculator` 一行是本篇诚实标准要求下必须标注的一处"查不到"：它在类型系统（`SpeculativeMethod` Literal）和配置解析（`SpeculativeConfig.__post_init__` 会从 `hf_config.model_type == "mlp_speculator"` 自动识别出这个方法名）两层都是完整的，但本库在 `self.drafter` 的分发链（`## 2` 已给出行号 `vllm/v1/worker/gpu_model_runner.py:632-703`）和 `vllm/v1/worker/gpu/spec_decode/`（V2 speculator 目录）里都没检索到任何一个类去处理它——按 `:699-703` 那个 `else: raise ValueError("Unknown speculative decoding method: ...")` 的兜底分支，配置 `method="mlp_speculator"` 在本快照下大概率会在模型初始化阶段直接报错。这是本库推断（基于"检索不到对应实现"这一负面证据），不代表未来版本或其他分支同样如此。

### 3.2 验证旋钮的组合：`rejection_sample_method` × `draft_sample_method`

proposer 决定"草稿怎么来"，这两个字段决定"草稿怎么被验证"，二者可以独立组合：

| `rejection_sample_method` | `draft_sample_method="greedy"`（默认） | `draft_sample_method="probabilistic"` |
|---|---|---|
| `"standard"`（默认） | 草稿概率按 one-hot 处理，概率比检验退化成"draft 是否等于 target 认为的高概率 token"（`## 7` 已展开的精确特例） | 用草稿模型真实的 `[num_tokens, vocab_size]` logits 做概率比检验，理论上更贴近原始 Leviathan 算法对任意 `q` 的一般表述，代价是多存一份草稿 logits |
| `"synthetic"` | 两列的差异被屏蔽：`SYNTHETIC_MODE` 分支直接读 `synthetic_conditional_rates_ptr` 判定接受与否，完全不看真实的 draft/target 概率（`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:583-585` 与 `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:656-658`），`draft_sample_method` 取哪个值对结果没有影响 | 同左 |
| `"block"` | 走块联合验证（Sun et al. 2024），仍然读取 draft/target 概率做残差质量估计，one-hot 与真实概率两种输入都支持 | 同左，但残差质量估计使用真实草稿 logits 会更准确 |

`ngram`/`suffix`/`custom_class` 这类没有神经网络输出分布的方法，"草稿概率"这个概念本身不存在，`draft_sample_method` 字段对它们没有意义——只有 EAGLE 系（`SpecDecodeBaseProposer` 子类）和 `draft_model` 这类会真正产出 logits 的方法才需要在这一列上做选择。

这张表格和 `## 5` 决策 1、决策 2 是同一件事的两个切面：决策 1、2 讲的是"为什么 `"standard"` 这一列的算法能保真"，这里补的是"另外两列（`"synthetic"`、`"block"`）和另一个维度（`draft_sample_method`）分别在什么条件下参与或不参与这个保真性论证"。

## 4. 主流程走读

以一个已开启 EAGLE 投机解码、非首步的 decode 迭代为例（batch 内至少一个请求处于 running 且 `request.spec_token_ids` 非空），整个循环跨两次调度、两次 RPC，拆成六步。

### 4.1 调度阶段一：把上一轮草稿装进本步的验证任务

`Scheduler.schedule()` 主循环遍历 running 请求时：

- 若 `request.spec_token_ids` 非空，按 `num_new_tokens` 反推 `num_scheduled_spec_tokens`，把草稿 token 截断进 `scheduled_spec_decode_tokens[request_id]`（`vllm/v1/core/sched/scheduler.py:711-727`）。
- 随后清空 `request.spec_token_ids = []`（`:727`），等待这一步产出的新草稿在下面 4.5 回填。
- `num_new_tokens` 本身可能被静态填充逻辑抬高过——见 4.2。

### 4.2 静态 K 的 CUDA Graph 填充分支

若配了**静态** K（`self.dynamic_sd_lookup is None`）且当前 batch 是"单请求单 token 的纯 decode 补丁"场景，调度器会主动把 `num_new_tokens` 抬高到 `1 + self.num_spec_tokens`：

```python
# vllm/v1/core/sched/scheduler.py:941-951（节选）
if (
    (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
    and self.num_sampled_tokens_per_step > 0
    and num_new_tokens == 1
    and (scheduled_running_reqs and not prefill_scheduled)
):
    num_new_tokens = 1 + self.num_spec_tokens
    ...
    pad_spec_decode = True
```

这是"把 decode 步填成统一形状，好让 CUDA Graph 能整图捕获"的 padding 逻辑，`## 5` 决策 3 展开代价——注意判定条件里显式写了 `self.dynamic_sd_lookup is None`，也就是说**这条填充逻辑只在静态 K 下生效**，这正是 `## 5` 决策 3 与 `vllm/config/vllm.py:967-984`（动态 K 强降 CUDA Graph 模式）互相呼应的地方。

### 4.3 调度收尾：动态 K 查表

调度收尾处，若配了**动态** K，查表算出 `num_spec_tokens_to_schedule` 并写进 `SchedulerOutput`：

```python
# vllm/v1/core/sched/scheduler.py:1264-1269（节选）
num_spec_tokens_to_schedule = self.num_spec_tokens
if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
    num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
        len(num_scheduled_tokens)
    ]
```

这个值决定的是**下一步**要提议几个草稿，与 4.1 已经定好的 `scheduled_spec_decode_tokens`（本步验证多少个）是两件独立的事——一个管"验证上一轮留下的草稿"，一个管"给下一轮生产多少新草稿"。

### 4.4 `execute_model()`：目标模型前向，不采样

`execute_model()`（`vllm/v1/worker/gpu_model_runner.py:4284`）目标模型对"已确认的历史 token + 本步待验证的草稿 token"做一次前向，产出的 `logits`（覆盖每个草稿位置和 bonus 位置）、`hidden_states` 等打包进 `ExecuteModelState`（详见 [[06-vLLM-模型执行与CUDA-Graph]] `## 3`），函数返回 `None`，不做采样。

紧接着是结构化输出 bitmask 计算的并行窗口：如兄弟篇所述，EngineCore 在 `execute_model()` 之后、`sample_tokens()` 之前可以并行算语法约束比特掩码，因为 `sample_tokens()` 的入参就是 `grammar_output`（`vllm/v1/worker/gpu_model_runner.py:4664`）。

### 4.5 `sample_tokens()`：先验证，再起草

`sample_tokens()`（`vllm/v1/worker/gpu_model_runner.py:4663`）内部顺序（骨架节选）：

```python
# vllm/v1/worker/gpu_model_runner.py:4663 起（骨架节选，行号见正文）
def sample_tokens(self, grammar_output):
    (scheduler_output, logits, spec_decode_metadata, ...) = self.execute_model_state  # L4676-4687
    self.execute_model_state = None
    if grammar_output is not None:
        apply_grammar_bitmask(scheduler_output, grammar_output, ...)      # L4693-4695
    sampler_output = self._sample(logits, spec_decode_metadata)           # L4696，定义于 L3755
    self._update_states_after_model_execute(sampler_output.sampled_token_ids, ...)
    def propose_draft_token_ids(sampled_token_ids):                       # L4721
        self._draft_token_ids = self.propose_draft_token_ids(...)         # L4724，定义于 L5116
    propose_draft_token_ids(sampled_token_ids)                            # L4768/4793/4848 多处调用点
```

验证部分：

- 解包 `execute_model_state`（`:4676-4687`），若 `grammar_output` 非空则 `apply_grammar_bitmask()`（`:4693-4695`）压低不合法 token 的 logits；
- 调 `self._sample(logits, spec_decode_metadata)`（`:4696`，定义于 `vllm/v1/worker/gpu_model_runner.py:3755`）：`spec_decode_metadata is None` 时走普通 `Sampler`，非空时取 `draft_probs`（草稿概率，`draft_sample_method="probabilistic"` 时才有意义）并调用 `self.rejection_sampler(spec_decode_metadata, draft_probs, logits, sampling_metadata)`（`:3778-3782`），拒绝采样内部逻辑见 `## 5` 决策 1、2；
- `_update_states_after_model_execute()` 把本步真正采出来的 token 写回请求状态。

起草部分：验证结束后立刻调用闭包 `propose_draft_token_ids(sampled_token_ids)`（`vllm/v1/worker/gpu_model_runner.py:4721-4735`，多处调用点见 `:4768/4793/4848`），内部走到 `self.propose_draft_token_ids()`（`vllm/v1/worker/gpu_model_runner.py:5116`），读出 4.3 算出的 `num_spec_tokens_to_schedule`（`:5131`），按 `method` 分发调用 `self.drafter.propose(..., num_speculative_tokens=num_spec_tokens_to_schedule, ...)`（`:5139-5241` 各分支）。**关键点：这次调用用的锚点是本步刚采样出的真实 token，不是上一步的草稿**——所以 drafter 的这次前向天然排在验证之后。产出的 `self._draft_token_ids` 之后经 `take_draft_token_ids()`（`:4995`）打包成 `DraftTokenIds` 交回调度器。

### 4.6 调度阶段二：草稿回填与语法过滤

回到调度器，`Scheduler.update_draft_token_ids()`（`:2264`）或异步路径下的 `update_draft_token_ids_in_output()`（`:2286`）把新草稿写回 `request.spec_token_ids`：

- 若该请求正在受结构化输出约束（`should_advance()` 为真），先过 `metadata.grammar.validate_tokens()`（`:2281,2312`）把不合语法的草稿位置截掉；
- 异步路径下还会用 `-1` 占位符补齐到原长度（`:2314-2316`）——这批 `-1` 正是两套拒绝采样核函数里"遇到负数 draft_token_id 直接判拒绝"分支的输入来源（`vllm/v1/sample/rejection_sampler.py:829` 附近、`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:559-565`）；
- 下一轮调度器再从这批 `spec_token_ids` 里截出 `scheduled_spec_decode_tokens`，回到 4.1，循环闭合。

整个循环里，"验证"永远比"起草下一轮"先发生，且两者共用同一次 `sample_tokens()` 调用——这也是为什么 `execute_model()` 和 `sample_tokens()` 必须拆成两次 RPC：如果不拆，语法 bitmask 就没有并行窗口，拒绝采样和起草下一轮草稿也没法插进 `execute_model()` 与"下一步 `execute_model()`"之间那个本该留给调度器做决策的间隙。

## 5. 设计决策与代价

**决策 1：验证用精确重采样（Leviathan et al. 2023），不用近似比对。**

- **为什么这么设计**：这是唯一能同时满足"用更小的草稿模型加速"和"不改变模型的输出分布"两个目标的构造。`vllm/v1/sample/rejection_sampler.py:40-58` 的类文档明确区分"accepted tokens / recovered tokens / bonus tokens"三类输出来源，`:829` 的概率比检验 `target_prob/draft_prob >= uniform_prob` 与 `:926` 的残差采样 `max(target_prob - draft_prob, 0)` 合在一起，正是原论文证明"输出边际分布严格等于直接从目标模型采样"的那个构造——不是启发式，是可以手推的等式。
- **不这样会怎样**：如果只是"草稿等于目标就接受，否则整批扔掉重新跑目标模型"（贪心比对，无重采样），温度 > 0 时会系统性地偏向草稿模型更容易生成的 token（草稿分布越偏，偏差越大），相当于把输出分布悄悄换成了草稿模型和目标模型的某种混合，而使用者往往毫无察觉——因为解码结果依然"看起来通顺"，只有跟无草稿基线做逐 token 分布对比才能发现。
- **什么时候可以不这样**：`rejection_sample_method="synthetic"`（`vllm/config/speculative.py:219-223`）就是故意不这样——它用配置好的 `synthetic_acceptance_rates` 直接决定接受与否，专门用于"没有真实草稿模型时压测调度器/CUDA Graph 在给定接受率下的吞吐"，代价是显式放弃分布保真，因此默认关闭且需要用户手动选择。`"block"` 模式（`:596`，Sun et al. 2024）是另一种精确算法而非近似，联合验证整块 token 而不是逐位扫描，仍然保真但计算路径不同，本篇不展开其推导。

**决策 2：贪心（温度=0）路径不做概率比检验，直接比对 argmax。**

- **为什么这么设计**：温度=0 时目标分布退化成一个 one-hot（只有 argmax 有非零概率），Leviathan 算法在这个极限下的解析解就是"draft 等于 target argmax 则接受，否则直接输出 target argmax"——`rejection_greedy_sample_kernel`（`vllm/v1/sample/rejection_sampler.py:715-766`）把这个化简后的特例直接写死：`token_id = target_argmax_id`（`:751`，`:756` 处 `not (draft_token_id != target_argmax_id)` 决定是否继续验证下一位），完全跳过重采样和它需要的残差分布计算，省掉一次全词表规模的额外运算。
- **不这样会怎样**：如果贪心也走概率比检验，数学上等价（one-hot 情形下比值退化成 0/1 判定），但要多算一次 `logsumexp`/softmax 且多引入一次随机数消耗，纯粹浪费。
- **什么时候可以不这样**：`rejection_sample_method="synthetic"` 时贪心路径依然要走随机接受（`:729-737`，`SYNTHETIC_MODE` 分支），因为此时"接受与否"由配置的伪接受率而非真实 argmax 比对决定，此时贪心与非贪心在代码结构上被迫统一处理。

**决策 3："动态 K"是配置期静态查表，不是运行时闭环。**

- **为什么这么设计**：闭环自适应（拿到接受率立刻改 K）需要在多进程/多 DP rank 之间同步这个决策，否则不同 rank 会选出不同验证长度，直接违反 vLLM 的"所有并行 rank 每步执行相同形状"假设（这也是下面"DP 下禁用动态 K"这条防御性代码存在的原因）。开环查表把决策提前到配置解析阶段，运行时只做一次数组下标查询（`vllm/v1/core/sched/scheduler.py:1264-1269`），代价小、行为在多进程间天然一致。
- **不这样会怎样**：使用者必须自己离线跑不同 batch size 下不同 K 的实际效果、把结果整理成 `(range_start, range_end, K)` 表喂进配置——vLLM 不提供任何默认表，也不采集实测数据自动生成它。`SpecDecodingStats`（`vllm/v1/spec_decode/metrics.py:17-47`）虽然逐位统计了接受 token 数，但这份数据只流向 Prometheus 日志，从未被读回来调整 `dynamic_sd_lookup`。
- **什么时候可以不这样**：负载模式高度可预测（比如固定并发的在线服务、batch size 波动范围小）时，静态表足够且更省心；负载抖动大、离线画不出稳定的"batch size → 最优 K"曲线时，开环查表容易在过渡区间选错档位——这正是 `## 6` 对照 SGLang 闭环方案时的落点。

**决策 4：草稿模型 KV 缓存与目标模型共用同一个 `KVCacheManager`，只是分组（`kv_cache_gid`）不同。**

- **为什么这么设计**：EAGLE/EAGLE3/MTP 类 drafter 通常只有 1 层或极少层，如果单独起一套块池管理、前缀缓存、驱逐策略，代码要重复维护两遍，且草稿层与目标层的 KV 生命周期高度相关（同一个请求同时终止/抢占）。`SpecDecodeBaseProposer.validate_same_kv_cache_group()`（`vllm/v1/spec_decode/llm_base_proposer.py:1708-1728`）先断言所有草稿层必须落在同一个 kv_cache_group（当前实现的前提假设），`initialize_attn_backend()` 再据此定位 `self.kv_cache_gid`（`:1744-1750`）。
- **不这样会怎样**：如果草稿层被允许散落在多个 kv_cache_group，drafter 就需要同时持有多套 `AttentionMetadata`，`SpecDecodeBaseProposer` 现在的实现直接假定单一 group，代码里用 `assert` 而不是分支处理多 group 情形——这是当前的一个真实限制（源码为证，见 `## 8`）。
- **什么时候可以不这样**：`draft_model` 方法（独立小模型）理论上可以有和目标模型完全不同的层数、不同的 attention backend，但只要它自己所有层落在同一 group 内，上述断言依然成立；真正"drafter 多套 KV group"的场景（比如 drafter 本身也做张量并行且切分方式与 target 不同）在这套实现里没有覆盖。

**决策 5：结构化输出不禁用投机解码，而是用"回滚 + 提前过滤"兼容它。**

- **为什么这么设计**：直接禁用会让"用户开了 JSON schema 就自动损失投机解码"，代价太大；xgrammar 的 `GrammarMatcher` 原生支持 rollback（`max_rollback_tokens=self.num_speculative_tokens`，`vllm/v1/structured_output/backend_xgrammar.py:127`），配合调度器在草稿产出后立即用 `validate_tokens()` 过滤（`vllm/v1/core/sched/scheduler.py:2281,2312`），可以做到"语法机随草稿走几步、验证结果如果比语法机预判的短就回滚几步"。
- **不这样会怎样**：若不做提前过滤，草稿里语法非法的 token 会被正常送进目标模型验证（目标模型的 logits 在这些位置已经被 bitmask 压过，非法 token 的概率理论上趋近于 0，大概率被拒绝），代码路径能跑通但会白白浪费一次验证 slot；用 `-1` 占位提前标记，等于提前把"注定会被拒绝"的位置标出来，拒绝采样核函数直接短路。
- **什么时候可以不这样**：不用结构化输出（无 grammar）时这整条路径完全不触发，`should_advance()` 为假直接跳过（`vllm/v1/core/sched/scheduler.py:2280`）。

**决策 6：chunked prefill 为 MTP 类 drafter 保留 lookahead 窗口，代价是牺牲一点 prefix cache 精度。**

- **为什么这么设计**：多模块 MTP drafter 在 chunk 边界处需要读到"接下来 `num_prefill_lookahead` 个已知的 prompt token"来预测下一模块，`_reserve_prefill_lookahead()`（`vllm/v1/core/sched/scheduler.py:463-481`）的文档字符串直接写明原因："boundary closer to the end than that would make it fall back to sampled drafts, permanently polluting the trailing modules' KV caches"——即如果不留够窗口，drafter 会被迫用采样出的假设 token 而不是真实 prompt token 去推进后续模块，污染那些模块的 KV，且这种污染不可逆。
- **不这样会怎样**：`KVCacheCoordinator` 里 `num_reprefillable_tokens = max(0, num_prefill_lookahead - 1)`（`vllm/v1/core/kv_cache_coordinator.py:96`）划出的这一小段"可能被重新 prefill"的尾部 token 不能被当作已经缓存完成的前缀直接复用给别的请求命中，因为它们的 KV 在 lookahead 逻辑下可能还没最终定型——这是为 spec decode 精度让出的一点前缀缓存收益。
- **什么时候可以不这样**：EAGLE 系列（非多模块 MTP）的 lookahead 只有 1（`vllm/v1/core/sched/scheduler.py:274-276` 的 `else` 分支），文档字符串里也写"No-op for eagle-family drafters (lookahead 1)"——单步 lookahead 不构成跨 chunk 边界的问题，这条保留逻辑事实上只对多模块 MTP 生效。

## 6. 同位对照：SGLang 的 EAGLE 怎么做

SGLang 的 `python/sglang/srt/speculative/` 同样是一个 proposer 家族目录（EAGLE / EAGLE3 / 多层 EAGLE / DFlash / Frozen-KV MTP / ngram / standalone worker）：

- 入口是 `EAGLEWorkerV2`（`sglang:python/sglang/srt/speculative/eagle_worker_v2.py:1050`）。
- 核心的 draft/verify/extend 三段式在 `EagleDraftWorker.draft()`（`sglang:python/sglang/srt/speculative/eagle_worker_v2.py:501`）、`EAGLEWorkerV2.verify()`（`:1539`）、`EagleDraftWorker.draft_extend()`（`:768`）里分开实现。
- 这与 vLLM 把"验证"和"起草下一轮"都塞进同一个 `sample_tokens()` 不同，SGLang 显式区分成三个方法调用，调用边界更清楚，但也意味着调度侧需要多一次跨方法的状态传递。

真正值得对照的是**动态 K 的实现方式**。SGLang 有一个专门的闭环模块 `adaptive_spec_params.py`（模块文档第一句就是"Adjusts speculative_num_steps at runtime based on observed acceptance lengths"）：

```python
# sglang:python/sglang/srt/speculative/adaptive_spec_params.py:22-45（节选）
DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1":  {"candidate_steps": [1, 3, 7], "down_hysteresis": -0.25, ...},
    "8":  {"candidate_steps": [0, 1, 3], ...},
    "32": {"candidate_steps": [0, 1], ...},
    "64": {"candidate_steps": [0], ...},
}
```

四个 batch size 档位（`1`/`8`/`32`/`64`），档位越高候选步数集合越小、越早出现 `0`。运行时通过 `AdaptiveController.activate_step_by_batch(batch_size)`（`sglang:python/sglang/srt/speculative/adaptive_runtime_state.py:106-108`）按当前 batch size 定档，`EAGLEWorkerV2` 在每步解码开头调用它（`sglang:python/sglang/srt/speculative/eagle_worker_v2.py:1187`），紧接着的分支处理是：

```python
# sglang:python/sglang/srt/speculative/eagle_worker_v2.py:1206-1209（节选）
if self.speculative_num_steps == 0:
    # Drafting disabled (high batch size). _draft_extend below still
    # runs, keeping draft KV warm for when the batch shrinks.
    verify_input = self._build_trivial_verify_input(batch)
```

档位内部还有第二层调节：

- `AdaptiveController.on_verify_complete()`（`sglang:python/sglang/srt/speculative/adaptive_runtime_state.py:113-121`）在每次验证后把本轮实际接受的 draft 数量喂给 `AdaptiveStepSlot.update()`（`sglang:python/sglang/srt/speculative/adaptive_spec_params.py:172-192`）。
- 用指数滑动平均（`ema_alpha` 默认 0.2）加滞回（`up_hysteresis`/`down_hysteresis`）决定要不要在候选步数集合里升档或降档，公式写在文档字符串里："`target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)`"（`:146`）。
- 即使 `speculative_num_steps` 因高 batch size 被压到 0，`_draft_extend` 仍会跑（`sglang:python/sglang/srt/speculative/eagle_worker_v2.py:1206-1209` 注释原话），让草稿模型的 KV 保持热更新——这样一旦 batch 缩小、重新升档，草稿模型不需要从冷启动重新追赶。

对照下来的结构性差异：**vLLM 的"动态"只有 batch-size 一个维度，且需要用户自己提供表；SGLang 的"动态"有 batch-size 分档 + 组内接受率 EMA 两个维度，且默认自带一份 `DEFAULT_ADAPTIVE_CONFIG`**（虽然 `speculative_adaptive` 开关本身默认关闭，`sglang:python/sglang/srt/server_args.py:2293-2297`）。两边都不是"默认打开的自动挡"，但 SGLang 打开后不需要用户自己离线画曲线，vLLM 打开后需要。

### 6.1 补一个次要对照：ngram 方法的底层加速手段不同

除了动态 K，`ngram` 这个不需要模型前向的方法在两边也走了不同的加速路线：

- vLLM 的 `NgramProposer`（`vllm/v1/spec_decode/ngram_proposer.py:12`）用 `numba` 的 `njit`/`jit`/`prange` 装饰器把 Python 函数即时编译成机器码，`__init__` 里甚至会主动跑一次 dummy `propose()` 触发 JIT 编译预热（`vllm/v1/spec_decode/ngram_proposer.py:53-60`），避免第一次真实调用时才付编译开销。
- SGLang 的 `ngram_worker.py`（`sglang:python/sglang/srt/speculative/ngram_worker.py`）依赖 `NgramCorpus`（`sglang:python/sglang/srt/speculative/cpp_ngram/ngram_corpus.py:14`），是一个独立编译的 C++ 扩展，而不是 JIT 编译的 Python。
- 两边思路相同（都是不训练任何模型、纯查表式的 prompt-lookup），差异只在"用 JIT 编译 Python 换开发便利"还是"用预编译 C++ 扩展换启动时不需要 JIT 预热"这个工程取舍上，不构成算法层面的分歧。

## 7. 踩坑与反直觉

**反直觉：投机解码在高并发下经常是负收益，本质原因是它把"用空闲算力换更少串行步数"这笔交易在算力不再空闲时继续做了一遍。**

- 单请求或小 batch 的 decode 步本质上是访存带宽瓶颈——GPU 的计算单元大部分时间在等 KV 缓存和权重从显存搬过来。此时让草稿模型多做一次前向，只要不把访存也挤爆，"顺便"验证 K 个 token 的边际算力代价很低，能省下的却是原本要串行跑 K 次的目标模型前向。
- 但当并发请求数涨起来、decode batch 已经把 GPU 的计算单元喂满（从访存瓶颈切换到算力瓶颈）之后，草稿模型的这次额外前向不再是"顺便"，它在直接和验证前向抢同一批计算单元——而验证前向本身已经因为要处理 `1+K` 倍的 token 数而变贵了。
- 这时"省下的串行步数"这笔收益是否能盖过"草稿前向 + 更贵的验证前向"这笔支出，不再是显然的。

vLLM 源码里能找到的、承认这一点的证据是**间接的**：

- `num_speculative_tokens_per_batch_size` 这个配置项存在本身（允许把大 batch 档位的 K 配成 0）就是承认"K 应该随 batch size 变化"。
- `_maybe_override_dynamic_sd_cudagraph_mode()`（`vllm/config/vllm.py:967-984`）和 `_maybe_disable_dynamic_sd_for_data_parallel()`（`:986-1002`）这两处防御性代码进一步说明这个特性被认真实现过、而不是一个摆设配置项。
- 但**vLLM 没有内置的默认表、没有运行时接受率反馈、也没有基于当前 batch size/GPU 利用率的自动降级**——搜索 `vllm/config/speculative.py` 与 `vllm/v1/spec_decode/*.py` 全文找不到任何"memory-bound"/"compute-bound"/"large batch"式的注释来解释这个权衡（已核实，未查证到解释性注释，只查到功能性代码），意味着这条决策的"为什么"停留在配置项存在这一层，没有写进代码注释或文档。

对照组是 SGLang：`## 6` 里展示的 `DEFAULT_ADAPTIVE_CONFIG["64"]["candidate_steps"] = [0]` 和 `sglang:python/sglang/srt/speculative/eagle_worker_v2.py:1206-1209` 那条 `"Drafting disabled (high batch size)"` 的注释，是本库目前读到的、**两个引擎里唯一一处用自然语言明确写出"高 batch size 应该关闭投机解码"这条设计理由的位置**——但要强调，这在 SGLang 里同样是 opt-in（`speculative_adaptive` 默认 `False`），不是任何一个引擎的默认行为。

**踩坑：贪心草稿（`draft_sample_method="greedy"`）把草稿概率当 one-hot，这不是简化，是数学上仍然精确的特例。**

`vllm/config/speculative.py:290-296` 的字段文档写："'greedy' always picks the argmax token, and the draft probabilities are treated as one-hot during rejection sampling."——第一次读容易以为这是一种为了省内存的近似（不存草稿完整 logits，当然会损失精度）。但重新看 `## 5` 决策 1 的算法：

- Leviathan 算法本身对任意草稿分布 q 都成立，包括 q 是退化到单点的 Dirac（one-hot）。
- `sample_recovered_tokens_kernel` 里 `NO_DRAFT_PROBS` 分支直接把目标概率在 `draft_token_id` 处置零当残差（`vllm/v1/sample/rejection_sampler.py:900-909`），这正是把通用公式 `max(target_prob - draft_prob, 0)` 代入 one-hot 的 `draft_prob` 后的解析特例，分布保真性没有丢失。
- 丢失的只是"drafter 可以按自己的概率而不是仅按 argmax 采样"这个自由度（这个自由度由 `draft_sample_method="probabilistic"` 换回来，代价是要多存一份 `[num_tokens, vocab_size]` 的草稿 logits，`:294-296` 原话是"comes at the cost of additional GPU memory usage"）。
- `ngram`/`suffix` 这类没有神经网络输出分布的方法则更彻底——它们的"草稿"本来就不是从某个概率分布采样出来的，而是确定性查找的结果，天然只能是 one-hot，谈不上"退化"。

**踩坑：spec decode 源码里其实有两套并行实现，一套默认关闭；`dspark` 强制切到那一套，`dflash2` 格式的 checkpoint 在默认关闭的那套上会静默降级。**

`## 2` 代码地图里出现的两组目录容易让人以为是同一套代码的两份拷贝，实际是两套独立实现：

- `vllm/v1/spec_decode/`（本篇 `## 2`~`## 6` 走读的主线）挂在 `GpuModelRunner`（"V1"）上，是否启用取决于 `VllmConfig.use_v2_model_runner` 这个属性（`vllm/config/vllm.py:649`）——默认读环境变量 `VLLM_USE_V2_MODEL_RUNNER`（`vllm/envs.py:300`，默认 `None`），未显式设置时按一串启发式规则决定。
- `vllm/v1/worker/gpu/spec_decode/`（"V2"，`## 2` 表里 `AdaptiveVerificationManager` 那一行已经点出这个目录）是另一套更晚近的实现，有自己独立的 `RejectionSampler`（`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:75`）和 proposer 类层级（`AutoRegressiveSpeculator`、`DFlashSpeculator`、`MultiModuleMTPSpeculator` 等）。

`use_v2_model_runner` 属性里的启发式规则，读起来就是一份"V1 覆盖不到的场景清单"：

```python
# vllm/config/vllm.py:658-677（节选）
# DSpark is implemented only by the V2 GPU model runner, and DeepSeek-V4
# is not otherwise a default-V2 architecture, so force V2 for it. If V2
# is unsupported for the rest of the config, _validate_v2_model_runner
# raises rather than silently falling back to V1 (which can't run dspark).
if (
    self.speculative_config is not None
    and self.speculative_config.method == "dspark"
):
    return True
...
# The DFlash2 candidate selector exists only in the V2 speculator. On V1
# the same checkpoint drafts through DFlashProposer, which never calls
# it, so the draft degrades to DFlash1 silently. Force V2 as for dspark.
if self._is_dflash2_draft():
    return True
```

两处注释各点出一个真实的坑：

1. `method="dspark"` 在 V1 的 `self.drafter` 分发链（`vllm/v1/worker/gpu_model_runner.py:632-703`）里**根本没有对应分支**——`dspark` 只在 `use_eagle()` 判定里被算作真（`vllm/config/speculative.py:1492-1496`），但真正的 dispatch 顺序是先查 `use_dflash()` 再查 `use_eagle()`（`vllm/v1/worker/gpu_model_runner.py:679-685`），`dspark` 两个分支都不匹配，理论上会一路落到 `else: raise ValueError(...)`（`:699-703`）。代码注释也直接承认"V1 ... can't run dspark"——这不是本库的推断，是上游注释原话。`use_v2_model_runner` 因此把 `method=="dspark"` 列为强制切 V2 的条件之一，避免用户撞上这个报错。
2. `dflash2` 格式的 checkpoint（`_is_dflash2_draft()` 判定，`vllm/config/vllm.py:702-711`）如果因为某种原因没有被强制切到 V2（比如显式设置了 `VLLM_USE_V2_MODEL_RUNNER=0` 覆盖掉自动判定），会落到 V1 的 `DFlashProposer`（`vllm/v1/spec_decode/dflash.py:23`）——但 V1 的 `DFlashProposer` 从不调用 DFlash2 的候选选择器，注释原话是"the draft degrades to DFlash1 silently"。也就是说**权重是按 DFlash2 训练的，跑起来的算法却是 DFlash1**，没有报错、没有警告，只是起草质量按 DFlash1 的水平打折——这正是本库反复强调的"静默降级"模式（同类问题见兄弟篇 [[06-vLLM-模型执行与CUDA-Graph]] `## 7` 的 cascade attention 案例）。
3. 同一份强制列表里还有第三条更小众的条件：`_dflash_needs_multi_kv_group()`（`vllm/config/vllm.py:713-723`）判定 DFlash 草稿模型是否混用了滑窗注意力与全量注意力层，混用意味着需要多个 KV 缓存组，这一能力目前只有 V2 实现——命中同样强制切 V2，不属于上面两条"静默降级"的范畴，因为它是在配置阶段就被拦下，不会真的跑到 V1 上。

判断当前到底跑的是 V1 还是 V2，`_get_v2_model_runner_unsupported_features()`（`vllm/config/vllm.py:2448`）还额外列出了一批 V2 尚不支持、会被强制拉回 V1 的组合——包括 `ngram`/`ngram_gpu`（`:2476-2478`）、EAGLE 的 `parallel_drafting`（`:2489-2496`）、EAGLE3 配流水线并行（`:2498-2502`）、`enable_adaptive_verification` 配 LoRA 或 `cudagraph_mode=NONE` 或流水线并行（`:2504-2530`）——两套实现各自覆盖不同的方法子集，没有哪一套是"全集"。

换句话说，`use_v2_model_runner` 与 `_get_v2_model_runner_unsupported_features()` 是同一枚硬币的两面：前者列的是"这些情况必须用 V2"，后者列的是"这些情况 V2 还不能用"。两份清单一旦出现交集（比如同时要求 `dspark` 强制 V2、又用了 V2 明确不支持的组合），`_validate_v2_model_runner()`（`vllm/config/vllm.py:2555-2564`，其存在在 `:660` 处的注释里被提前提及）的设计是直接抛 `ValueError`（`:2562-2564`）而不是静默回退到跑不了 `dspark` 的 V1——这一点本身是对"静默降级"教训的正确应对，值得和上面两条真正的静默降级坑对照着读。

## 8. 可改进点

以下均为本库阅读代码后的推断，未在上游 issue/PR 中核实是否已有讨论（诚实标准第 6 条：不编 issue 号）。

1. **动态 K 表没有配套的离线画表工具，门槛集中在用户身上。** `num_speculative_tokens_per_batch_size` 要求用户自己跑出不同 batch size 下不同 K 的实测效果再手填三元组，`vllm/v1/spec_decode/dynamic/utils.py` 里只有"把配好的表展开成稠密数组"的逻辑（`validate_and_normalize_dynamic_sd_schedule` / `build_dynamic_sd_schedule_lookup`），没有任何采集/画曲线/推荐配置的辅助脚本。对照 `## 6` 里 SGLang 自带 `DEFAULT_ADAPTIVE_CONFIG` 这个"至少能用的默认档位表"，vLLM 在这一点上把认知负担全部转嫁给了用户。补一个基于 `SpecDecodingStats`（已经在采集接受率了）离线聚合、按 batch size 分桶输出建议表的小工具，不需要改动核心执行路径，纯粹是"把已经在收集的数据用起来"。
2. **`SpecDecodingStats` 采集的数据完全没有反馈通道，是一个"写了但没人读"的度量。** `vllm/v1/spec_decode/metrics.py:17-47` 里的 `observe_draft()` 逐位记录接受 token 数，最终只流向 `SpecDecodingLogging`（Prometheus/日志），全仓库找不到任何地方把这份统计读回来调整 `num_speculative_tokens` 或 `dynamic_sd_lookup`。哪怕不做 SGLang 那种全自动 EMA 闭环，先把"当前接受率是否长期低于某阈值"做成一条 `logger.warning_once`（类比兄弟篇 [[06-vLLM-模型执行与CUDA-Graph]] `## 8` 里对 CUDA Graph 静默降级提的同类建议），也能帮用户更快发现"这个草稿模型/这批数据其实不划算"。
3. **`SpecDecodeBaseProposer.validate_same_kv_cache_group()`（`vllm/v1/spec_decode/llm_base_proposer.py:1708-1728`）用 `assert` 硬编码"草稿层必须全部落在同一个 kv_cache_group"这一假设，报错信息里没有指出具体是哪个 group 冲突。** 这类断言失败通常发生在用户手动拼接自定义 draft 模型结构（比如 `draft_model` 方法接一个和 target 层数结构差异较大的模型）时，把断言信息里的 `kv_cache_groups` 映射也打印出来，能省下一轮"翻源码猜断言原因"的调试时间。
4. **静态 K 的 CUDA Graph 填充分支（`## 4` 4.2，`pad_spec_decode`）和动态 K 是互斥的两条路径，没有中间档。** 一旦打开动态 K，`## 5` 决策 3 提到的 padding 逻辑整体失效（判定条件里的 `self.dynamic_sd_lookup is None` 直接把这条分支锁死），CUDA Graph 模式也被迫整体降到 PIECEWISE（`vllm/config/vllm.py:967-984`）——用户没有办法"只在某几档 batch size 上用动态 K，其余档位仍吃 FULL 图"。给 `_maybe_override_dynamic_sd_cudagraph_mode()` 补一个"按档位判定是否需要降级"的精细化路径（比如只在真正跨档切换 K 的那一步临时切到 PIECEWISE，其余步骤仍用 FULL 图 replay），理论上能拿回一部分被牺牲掉的 CUDA Graph 收益，但需要先确认 CUDA Graph 捕获集合能否支持"按 K 值分别捕获"（`## 6` 提到 SGLang 用 `SpecRuntimeState` 为每个候选步数各自维护一套 CUDA Graph runner，这条路径本库判断值得借鉴，但涉及较大改动，仅记录方向不展开设计）。
5. **`mlp_speculator` 是一个在类型系统里活着、在执行路径里已经死掉的方法名。** `## 3.1` 表格脚注已经指出：`SpeculativeMethod` Literal（`vllm/config/speculative.py:69-79`）和 `SpeculativeConfig.__post_init__` 的自动识别逻辑（`hf_config.model_type == "mlp_speculator"` 时把 `method` 设成它）都完整存在，但 `GpuModelRunner.__init__` 的 `self.drafter` 分发链（`vllm/v1/worker/gpu_model_runner.py:632-703`）和 V2 speculator 目录都没有任何分支处理这个取值——配置成它大概率在模型初始化阶段直接撞上 `:699-703` 那个 `else: raise ValueError(...)`。这类"配置层还认得、执行层已经不认得"的方法名，要么该在 `SpeculativeMethod` 里加一条弃用注释、要么该在 `__post_init__` 解析到它时提前给出比"Unknown speculative decoding method"更明确的报错（比如直接提示"mlp_speculator 在当前版本没有对应的 proposer 实现"），现状是把这个诊断成本留给了第一个踩到的用户。

## 9. 自测题与延伸阅读

**自测题**（闭卷回答，答案均可在 `## 2`~`## 7` 用到的行号里核实）：

1. `execute_model()` 和 `sample_tokens()` 里，`self.drafter.propose(...)` 是在哪一个方法里被调用的？它用来当锚点的 token 是本步真实采样出的 token，还是上一步的草稿 token？为什么必须是前者？
2. 温度=0（贪心）时，拒绝采样核函数最终写入输出的 `token_id` 永远等于什么？这意味着开了投机解码的贪心解码和不开投机解码的贪心解码，在同一个随机种子下逐 token 位对位比较，理论上应该完全一致还是允许不同？
3. `rejection_sample_method="synthetic"` 和 `"standard"` 相比，牺牲了什么来换取什么？什么场景下会主动选它？
4. `num_speculative_tokens_per_batch_size` 不配置（默认 `None`）时，`Scheduler.dynamic_sd_lookup` 是什么？这时候 `K` 会不会随 batch size 变化？
5. `_maybe_disable_dynamic_sd_for_data_parallel()` 为什么要在 DP>1 时禁用动态 K 而不是允许各 rank 独立选自己的 K？
6. `draft_sample_method="greedy"` 时，`sample_recovered_tokens_kernel` 里残差分布 `max(target_prob - draft_prob, 0)` 退化成了什么形式？这个退化是否损失了"输出分布等于目标模型"这条正确性保证？
7. `_reserve_prefill_lookahead()` 为什么"No-op for eagle-family drafters (lookahead 1)"，但对多模块 MTP drafter 不是 no-op？
8. SGLang 的 `DEFAULT_ADAPTIVE_CONFIG` 里，batch size 越大候选步数集合里为什么越早出现 `0`？这和 vLLM 里 `num_speculative_tokens_per_batch_size` 允许配 `K=0` 的动机是不是同一件事？两者默认行为（不配置/不打开）是否相同？
9. `method="dspark"` 在 V1 的 `self.drafter` 分发链里为什么没有对应分支？`use_v2_model_runner` 属性是怎么处理这种情况的，用户会不会真的撞上那个 `ValueError`？
10. 一个用 DFlash2 权重训练出来的 checkpoint，如果被强制跑在 V1 的 `DFlashProposer` 上，行为上会报错，还是会正常跑完但起草质量打折？为什么源码注释用"silently"这个词？

**延伸阅读**：

- [[06-vLLM-模型执行与CUDA-Graph]]——`execute_model()`/`sample_tokens()` 两次 RPC 拆分的完整背景，`ExecuteModelState` 的其余字段
- [[03-vLLM-调度器解剖]]——`SchedulerOutput` 的其余字段与 decode/prefill 调度全貌
- [[04-vLLM-KV缓存与前缀缓存]]——`kv_cache_group`、block table 的分配细节
- [[11-vLLM-结构化输出]]——`GrammarOutput`/`apply_grammar_bitmask` 的产出方与调用时机
- [[10-SGLang-投机解码EAGLE]]——SGLang 侧的完整解剖（本篇 `## 6` 只做了动态 K 一个维度的对照）
- [[00-总览与阅读地图]]
