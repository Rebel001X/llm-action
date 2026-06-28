# TensorRT-LLM 引擎构建参数

> `trtllm-build` 把一份模型权重（checkpoint）"编译"成一个针对特定 GPU、特定批量/序列长度组合优化好的推理引擎（engine），构建参数就是这份"编译指令"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/tensorrt-llm/README]] [[llm-inference/tensorrt-llm/Triton服务启动参数]] [[llm-optimizer/kv-cache]] [[llm-optimizer/FlashAttention]]

## 阅读地图

| 小节 | 解决什么问题 | 你会得到什么 |
| --- | --- | --- |
| 0 锚点 | 一句话记住引擎构建是干嘛的 | 心智模型 |
| 1 地基 | 为什么 TRT-LLM 要"提前构建引擎"而不是即时运行 | AOT 编译思想 |
| 2 工作流 | checkpoint → build → engine → serve 全链路 | 调用链 ASCII 图 |
| 3 参数分类 | 几十个 flag 怎么归类记忆 | 5 大类参数地图 |
| 4 形状/容量参数 | max_input_len / max_seq_len / max_num_tokens 怎么定 | 调参权衡 |
| 5 KV Cache 相关 | paged / reuse / chunked / 量化 KV | 显存与吞吐 |
| 6 注意力/Attention | context_fmha / paged_context_fmha / fp8 | 算子融合 |
| 7 内存与并行 | weight_streaming / TP / PP / 算子融合 | 大模型塞进小卡 |
| 8 参数速查 | 把原文参数整理成表 | 含义+权衡 |
| 9 坑 | 构建期与运行期不一致的典型错误 | 排错清单 |

## 0. 一句话锚点

**TensorRT-LLM 是"提前编译（AOT）"路线**：先用 `trtllm-build` 把模型 + 一组"形状假设"（最大批量、最大序列长度、并行切分方式、量化精度）固化成一个二进制 `.engine` 文件，运行时只负责喂数据、不再做图优化。**构建参数 = 你在编译期对运行期工况做的承诺**。承诺得太小，跑大请求会报错；承诺得太大，显存白白浪费。

## 1. 地基：它解决什么问题

普通 PyTorch 推理是"解释执行"——每一步都临时决定用哪个内核、怎么分配显存。TensorRT-LLM 走的是**编译执行**：把整张计算图提前做内核选择（kernel auto-tuning）、算子融合、内存复用规划，编进一个针对**这块具体 GPU**优化的引擎。

```
                    解释执行 (vLLM/HF)              编译执行 (TensorRT-LLM)
                    ──────────────────              ──────────────────────
  准备阶段            几乎没有                        trtllm-build（分钟级，一次性）
  每次推理            动态决策内核、动态分配           直接跑预编译好的固定内核
  灵活性              高（任意 shape）                低（受 build 期承诺约束）
  峰值性能            好                              通常更极致（融合+autotune）
```

**代价**：引擎一旦构建，它的"能力边界"（最大序列长度、并行度、精度、是否支持某高级特性）就被冻结了。想改这些，必须**重新构建**。所以 `trtllm-build` 的参数选择，本质是**用编译期信息换运行期性能**，参数选错只能重来。

> 具体子命令名、默认值随版本变化，一切以官方文档为准：
> https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html

## 2. 整体工作流（调用链）

构建不是孤立一步，它处在"转权重 → 建引擎 → 起服务"的中间环节：

```
  ┌─────────────┐   convert_checkpoint.py   ┌──────────────────┐
  │ HF / 原始权重 │ ───（转格式 + 量化标定）──▶ │ TRT-LLM checkpoint │
  │  (safetensors)│                          │  (统一中间表示)    │
  └─────────────┘                            └────────┬─────────┘
                                                       │  trtllm-build
                          形状/并行/精度/特性参数 ──────▶│  ＝本文主角
                                                       ▼
                                            ┌────────────────────┐
                                            │   .engine + config   │
                                            │ (绑定到某型号GPU)     │
                                            └────────┬───────────┘
                                                     │  加载部署
                         ┌───────────────────────────┼───────────────────────────┐
                         ▼                            ▼                            ▼
                  trtllm-serve              Triton + tensorrtllm_backend     C++/Python runtime
                  (OpenAI 兼容)             (生产级，配 Triton 参数)         (嵌入式集成)
```

关键认知：**构建参数**和**运行/服务参数**是两层。例如 `--use_paged_context_fmha`（构建期决定"引擎是否具备 KV 重用能力"）与 Triton 侧的 `enable_kv_cache_reuse`（运行期决定"是否真的启用"）必须**配套打开**，缺一不可 → 见 [[llm-inference/tensorrt-llm/Triton服务启动参数]]。

## 3. 参数分类地图

几十个 flag 看着乱，按"它影响引擎的哪个维度"分成 5 类就好记：

```
                       trtllm-build 参数
                            │
   ┌──────────┬────────────┼────────────┬──────────────┐
   ▼          ▼            ▼            ▼              ▼
[形状/容量]  [KV Cache]   [Attention]  [内存/并行]    [插件/特性开关]
max_input_len use_paged_   context_fmha  weight_       gpt_attention_plugin
max_seq_len   context_fmha use_paged_    streaming     gemm_plugin
max_num_tokens kv_cache_   context_fmha  tp/pp_size    reduce_fusion
max_batch_size  type       use_fp8_      remove_input_ streamingllm
tokens_per_block paged_kv   context_fmha  padding       ...
                 _cache
```

- **形状/容量类**：定义引擎的"尺寸上限"，直接决定显存预分配和能否处理大请求。
- **KV Cache 类**：控制键值缓存的组织方式（分页）、复用、分块、量化——吞吐与显存的核心旋钮。
- **Attention 类**：选择哪种融合注意力内核，与精度（FP8）联动。
- **内存/并行类**：多卡切分（张量并行 TP / 流水并行 PP）和权重流式加载，决定模型能否塞下。
- **插件/特性开关**：打开特定高级能力（StreamingLLM、AllReduce 融合等）。

## 4. 形状 / 容量参数（最容易踩坑的一类）

这组参数定义引擎处理请求的"尺寸合同"。理解三个概念的区别是关键：

```
   单条请求                       一个批次（batch）
   ┌───────────────────────┐      ┌──────────────────────────────┐
   │ prompt(输入) │ 输出     │      │ req1 │ req2 │ req3 │  ...      │
   │◀ max_input_len ▶        │      └──────────────────────────────┘
   │◀──────── max_seq_len ────────▶│   ◀ max_num_tokens（去填充后整批 token 上限）▶
                                       ◀ max_batch_size（最多并发几条）▶
```

| 参数 | 管的是什么 | 调大 / 调小的影响 |
| --- | --- | --- |
| `--max_input_len` | 单请求**输入（prompt）**最大 token 数 | 太小 → 长 prompt 报错；太大 → context 阶段计算/显存上界变高 |
| `--max_seq_len` / `--max_decoder_seq_len` | 单请求**输入+输出**总长度 | 直接决定单条最长能生成多长；未指定时从模型配置推导 |
| `--max_num_tokens` | **去除填充后**整批输入 token 总数上限 | 启用 in-flight batching 时的核心吞吐旋钮；过小限制并发，过大增显存 |
| `--max_batch_size` | 同时在飞（in-flight）的最大请求条数 | 与 max_num_tokens 共同决定并发规模 |

**为什么 `max_num_tokens` 取代了"batch × seq"的老算法？** 因为引擎默认开启 `--remove_input_padding`：把不同长度的请求**首尾相接打包成一条连续序列**，不再为短句补 PAD。这样真正占显存的是"实际 token 总数"，而不是"批量 × 最长序列"那个被填充撑大的矩形。

```
  开启 remove_input_padding 前（填充浪费）       开启后（紧凑打包）
  req1: [t t t t]                                [t t t t | t t | t t t t t t]
  req2: [t t P P]   ← P 是填充，白算            └─ 一条连续序列，按 max_num_tokens 计 ─┘
  req3: [t t t t t t]
```

经验上：`max_num_tokens` 决定吞吐天花板，`max_seq_len` 决定单请求长度天花板，二者独立设置。具体默认值以官方文档为准（不要背死数字，版本间会变）。

## 5. KV Cache 相关参数

KV Cache 是自回归推理的"记忆"，每生成一个 token 就要把它的 Key/Value 存下来供后续注意力复用。它通常是**长上下文场景下最大的显存开销**。原理见 [[llm-optimizer/kv-cache]]。

### 5.1 Paged KV Cache（分页）

```
  连续分配（旧）                       分页分配（PagedAttention 思路）
  ┌──────────────────────┐            ┌──┐┌──┐┌──┐┌──┐  按需取块
  │ 为最长序列预留一大块   │            │块│块│块│块│ ← 每块 tokens_per_block 个 token
  │ 短请求 → 大量空洞浪费  │            └──┘└──┘└──┘└──┘
  └──────────────────────┘            像操作系统的"虚拟内存分页"，碎片少
```

- `--kv_cache_type paged`：启用分页 KV Cache。**注意** `--paged_kv_cache enable` 是已过时（deprecated）写法，新版用 `--kv_cache_type`。
- `--tokens_per_block`：每个 KV 缓存块装多少 token。块越小越省（碎片少）但管理开销略增；块越大管理简单但浪费略多。是显存利用率与调度开销的权衡。

### 5.2 KV Cache 复用（Reuse）

多个请求若**共享相同前缀**（如同一段 system prompt、同一篇 RAG 文档），可以让它们复用已算好的 KV，而不是各算一遍：

```
  请求A: [系统提示 S][用户问1]   ┐
  请求B: [系统提示 S][用户问2]   ┘─ S 的 KV 只算一次，B 直接复用
```

启用复用需**两端配合**：
- 构建期：`--use_paged_context_fmha enable`（让引擎具备复用所需的分页 context FMHA 能力）。
- 运行期：Triton 配置里 `enable_kv_cache_reuse`（真正开启复用）。

### 5.3 Chunked Context（分块上下文）

超长 prompt 的 context（prefill）阶段一次性算完会产生巨大的瞬时显存峰值，并阻塞解码。分块上下文把长 prompt **切成若干 chunk 分批处理**，平滑显存与延迟，也利于和解码阶段交错调度。它依赖 `--use_paged_context_fmha`。

### 5.4 INT8 / FP8 KV Cache（量化缓存）

把 KV Cache 从 FP16 压成 INT8/FP8，直接砍掉一半甚至更多显存，从而装下更长上下文或更多并发。

```
  FP16 KV: ████████  16 bit/值
  FP8  KV: ████      8  bit/值  → 同样显存能存 2× 的上下文
```

注意点：目前 KV 量化通常仅支持 **per-tensor** 粒度（整张张量共用一个 scale），精度上比 per-channel/per-token 粗，对极敏感模型需评估质量。量化原理见 [[llm-compression/quantization/量化基础]]。

## 6. 注意力（Attention / FMHA）参数

FMHA = Fused Multi-Head Attention，把 QK^T、softmax、再乘 V 这一串操作**融合进单个 GPU 内核**，省去中间结果反复读写显存——这正是 FlashAttention 的思想（见 [[llm-optimizer/FlashAttention]]）。

注意力大致分两个阶段：
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- **Context（prefill）阶段**：一次性处理整段 prompt 的所有 token。
- **Generation（decode）阶段**：每步只处理 1 个新 token。

| 参数 | 作用 | 何时用 |
| --- | --- | --- |
| `--context_fmha` | 在 **context 阶段**启用融合 MHA/MQA/GQA 内核，单 kernel 完成注意力块 | 通常默认开，几乎总要开 |
| `--use_paged_context_fmha` | 分页版 context FMHA，**KV 复用、分块上下文等高级特性的前提** | 需要复用/chunked 时必开 |
| `--use_fp8_context_fmha` | 在 FP8 量化激活时，让 context 注意力也走 FP8，进一步提速 | 已用 FP8 量化时 |

理解链路：`context_fmha`（基础融合）→ `use_paged_context_fmha`（解锁高级特性）→ `use_fp8_context_fmha`（叠加 FP8 提速）。后者依赖前提条件逐层递进。

## 7. 内存与并行参数

### 7.1 权重流式加载 `--weight_streaming`

当模型权重 + KV Cache 超过单卡显存时，可把**部分权重卸载到 CPU 内存**，运行时按需流式加载回 GPU：

```
  GPU 显存:  [当前层权重][KV Cache][激活]
                 ▲ 用完即换下一层
  CPU 内存:  [............全部权重............]  ── PCIe 流入 ──▶
```

代价是 PCIe 搬运带宽成为瓶颈、延迟上升。属于"显存实在不够、宁慢勿挂"的兜底手段，默认关闭。

### 7.2 张量并行 / 流水并行（TP / PP）

单卡塞不下时把模型切到多卡。**TP（张量并行）**把每一层的权重矩阵横/纵切到多卡，每步都要 AllReduce 同步；**PP（流水并行）**把不同层分到不同卡，像流水线接力。

```
  TP=2（层内切分）              PP=2（层间切分）
  GPU0: [W 左半]  ┐ AllReduce   GPU0: [Layer 0~15]
  GPU1: [W 右半]  ┘ 每层同步    GPU1: [Layer 16~31]  ← 激活在卡间传递
```

并行度（如 `--tp_size` / `--pp_size`，名称以文档为准）必须在**构建期**确定并写进引擎——这也是引擎"绑定部署形态"的体现。底层通信靠 NCCL（见 [[ai-infra/网络/NCCL]]）。

### 7.3 算子融合 `--reduce_fusion`

`--reduce_fusion enable` 会把 TP 通信中的 **AllReduce 之后的 ResidualAdd + LayerNorm 融合进一个内核**，减少 kernel 启动开销与显存往返，提升端到端性能。默认 disable，多卡 TP 场景值得试开。

### 7.4 去填充 `--remove_input_padding`

如第 4 节所述，把变长请求紧凑打包，去掉 PAD 的无效计算与显存。默认 enable；要关用 `--remove_input_padding disable`（一般没理由关）。

## 8. 典型构建命令与参数说明

一个**示意性**命令（参数名/默认值以官方文档为准，不要把这里的值当权威）：

```bash
trtllm-build \
  --checkpoint_dir ./ckpt/llama-7b-fp16 \
  --output_dir     ./engine/llama-7b \
  --max_input_len  4096 \      # 单请求 prompt 上限
  --max_seq_len    8192 \      # 单请求 总长上限
  --max_num_tokens 16384 \     # 整批去填充后 token 上限（吞吐旋钮）
  --max_batch_size 64 \        # 最大并发请求数
  --kv_cache_type  paged \     # 分页 KV Cache
  --tokens_per_block 64 \      # 每块 token 数
  --use_paged_context_fmha enable \   # 解锁 KV 复用 / chunked context
  --context_fmha enable \      # 基础融合注意力
  --remove_input_padding enable \     # 去填充打包
  --reduce_fusion enable       # TP AllReduce 后算子融合
```

**心法**：先按业务确定形状（input/seq/batch），再据显存预算决定 KV 策略（分页+复用+量化），最后叠加注意力/融合优化。改任一项都要**重新 build**。

## 9. 常见问题 / 坑

| 现象 / 坑 | 根因 | 处理 |
| --- | --- | --- |
| 长请求运行时报错 "exceeds max ... len" | `max_input_len` / `max_seq_len` 在构建期设小了 | 重新 build，调大对应上限 |
| 开了 `enable_kv_cache_reuse` 却没复用 | 构建期没开 `--use_paged_context_fmha`，运行期单开无效 | 两端配套打开 |
| 用了 deprecated 的 `--paged_kv_cache enable` 报警 | 旧参数名 | 改用 `--kv_cache_type paged` |
| FP8 量化没提速 | 只量化了权重，没开 `--use_fp8_context_fmha` | 注意力侧也走 FP8 |
| 引擎换台 GPU 跑不动/性能掉 | 引擎绑定到构建时那块 GPU 型号 | 目标 GPU 上重新构建 |
| 显存不够但又不想多卡 | 没考虑 KV 量化/weight_streaming | 先试 INT8/FP8 KV，再考虑 streaming 兜底 |
| 改了并行度（TP/PP）报维度错 | 并行度是构建期承诺，运行期不可改 | 按目标并行度重新 build |
| 背了默认值结果对不上 | 默认值随版本变化 | 永远以 `trtllm-build --help` 与官方文档为准 |

> 核心提醒：**TensorRT-LLM 的所有"能力边界"都在构建期冻结**。遇到运行期限制，第一反应是回到 `trtllm-build` 检查对应参数，而不是改运行配置。

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-inference/tensorrt-llm/README]]
- [[llm-inference/tensorrt-llm/Triton服务启动参数]]
- [[llm-optimizer/kv-cache]]
- [[llm-optimizer/FlashAttention]]
- [[llm-compression/quantization/量化基础]]
- [[ai-infra/网络/NCCL]]
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
