# SGLang ServerArgs 旋钮全景

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：476 个字段靠反射生成 CLI，29 个手写只是历史包袱

## 0. 结论先行

- **`ServerArgs` 有 476 个字段（`python/sglang/srt/server_args.py:473`-`3637`），文件里却只有 29 处 `parser.add_argument(`（`python/sglang/srt/server_args.py:8761`-`9020`）。** 这不是"少数旋钮走了捷径"，而是反过来：476 个字段里的绝大多数根本不在 `add_cli_args` 里手写，靠一个通用循环 `add_cli_args_from_dataclass(parser, ServerArgs)`（`python/sglang/srt/server_args.py:8764`，实现在 `python/sglang/srt/arg_groups/arg_utils.py:218`-`337`）读取每个字段的 `Annotated[T, Arg(...)]` 元数据自动 `add_argument`。**29 个手写调用里，25 个是废弃标志的重定向壳，3 个是运行时才能确定 choices 的动态字段，1 个是不对应任何 dataclass 字段的 `--config` 元参数**——真正"自动生成机制处理不了"的情况只有 4 个，不是 29 个。
- 每个字段除了 CLI 元数据（`Arg`），还带一个 `NS("namespace.path")` 标记（`python/sglang/srt/arg_groups/arg_utils.py:85`-`99`）。**这不是注释性的分类标签，而是运行时真的会用来构建一棵嵌套配置树**（`RuntimeContext` 的 `_build_config_bags`，`python/sglang/srt/runtime_context.py:677`-`722`）——`ServerArgs` 本身始终是一个**扁平**的 dataclass，嵌套访问（`ctx.exec.moe.xxx`）是运行时用反射投影出来的，不是靠嵌套 dataclass 结构。
- **本篇在核对 `_lab/out/api_surface.json` 时抓到了取证工具自己的一个 bug**：该 JSON 里字段的 `type` 字符串被截断到 160 字符，恰好 196 个字段的注解文本超过这个长度，导致排在 `Annotated[...]` 最后一位的 `NS(...)` 标记被截断丢失，`_lab` 的正则统计误判这 196 个字段"没有命名空间"。用 `ast.get_source_segment` 重新对源码逐字段取**完整**注解文本后核实：**476 个字段全部带 `NS(...)`，一个不缺**，与仓库自己的覆盖率测试 `test/registered/unit/test_server_args_namespaces.py:44`-`50`（要求 `len(nsmap) >= 440`）一致。本篇 `## 2` 和 `## 7` 详细说这个坑。
- `__post_init__`（`python/sglang/srt/server_args.py:3643`）只有一行，调用 `_run_resolution_pipeline()`（`python/sglang/srt/server_args.py:3646`），后者是一个有序调度器，依次调用 **88 个 `_handle_*`/`_apply_*`/`_validate_*`/`_disable_*` 辅助方法**，横跨 `python/sglang/srt/server_args.py:3646`-`8759`，约 **5,113 行**——占全文件（10,142 行）过半篇幅。这条流水线做两件事：给 `None` 默认值做硬件相关的自动推断（如按 GPU 显存分级设置 `chunked_prefill_size`），以及在字段组合不合法时报错。
- **有一类旋钮会被无条件覆盖，不管用户是否显式设置过**——不是"默认值为 `None` 才推断"这种温和情形，而是"设了也不算"。最典型的例子：`attention_backend` 解析成 `torch_native` 或 `flex_attention` 后，`cuda_graph_config.prefill/decode.backend` 会被强制设为 `Backend.DISABLED`（`python/sglang/srt/server_args.py:6010`-`6022`），哪怕用户在命令行里明确传了 `--cuda-graph-backend-prefill=full`。`_disable_tc_piecewise_cudagraph_if_incompatible`（`python/sglang/srt/server_args.py:4649`-`4723`）更极端，列了 **18 条互斥规则**，任意一条命中就把 `cuda_graph_config.prefill.backend` 拍成 `DISABLED`。`## 3` 逐条列出这类"设了也白设"的旋钮。
- 挑出的 13 个关键旋钮（`## 4`）横跨并行（`tp_size`/`pp_size`/`ep_size`/`enable_dp_attention`）、内存（`mem_fraction_static`/`page_size`/`disable_radix_cache`）、调度（`chunked_prefill_size`/`schedule_policy`/`max_running_requests`/`disable_overlap_schedule`）、量化与内核（`kv_cache_dtype`/`attention_backend`）——覆盖面广不代表旋钮数量合理，这正是 `## 5` 要处理的工程判断。
- 全篇不给任何"调大 X 提升 Y%"的实测数字（本机无 GPU），只给方向性影响与副作用。

## 1. 它在系统里的位置

`ServerArgs` 是 SGLang 服务端进程的**单一真源配置对象**：CLI 解析出来的第一件事就是 `ServerArgs.from_cli_args(args)`（`python/sglang/srt/server_args.py:9022`-`9030`），产出的实例随后被传给 `TokenizerManager`、`Scheduler`、`ModelRunner`、HTTP 层等几乎所有子系统的构造函数——它是这些子系统之间共享的"配置真相来源"，而不是某个子系统私有的参数包。

它和另外两个配置类的边界很清楚（三者都在 `_lab/out/api_surface.json` 的 `sglang.config_classes` 里，都定义在 `server_args.py`，字段数分别是 476 / 10 / 30）：

- **`PortArgs`**（`python/sglang/srt/server_args.py:9970`，10 个字段）——纯粹是进程间通信端点（ZMQ IPC 文件名、NCCL 端口等），跟"要不要调大这个旋钮"无关，是 `ServerArgs` 解析完之后**派生**出来的运行时寻址信息，不走 CLI。
- **`SamplingParams`**（`python/sglang/srt/sampling/sampling_params.py:45`，30 个字段）——这是**每请求**级别的配置（temperature/top_p/max_new_tokens…），随 HTTP 请求体传入，生命周期是一次生成；`ServerArgs` 是**每进程**级别的配置，生命周期是整个服务进程。两者是正交的两个轴：`ServerArgs` 决定"这台服务器能干什么、性能边界在哪"，`SamplingParams` 决定"这一次生成要什么样的输出"。

`ServerArgs` 解析完成后并不是原地保持"一个大 dataclass"的样子被到处传递——`## 3` 会讲它如何通过 `NS(...)` 标记被投影成一棵嵌套的 `RuntimeContext` 配置树，供子系统按命名空间读取（例如 `get_context().schedule.chunked_prefill_size`），这一层间接是"扁平字段 + 元数据驱动的结构化视图"这个设计的核心收益点。

## 2. 代码地图（文件 → 职责，带行号）

| 文件:行 | 职责 |
|---|---|
| `python/sglang/srt/server_args.py:473` | `class ServerArgs` 定义起点，类体一路延伸到约 9021 行 |
| `python/sglang/srt/server_args.py:483`-`511` | 类文档字符串："加新参数的三条规则"——用 `A[T, Arg(...)]`，只有废弃/动态 choices/`--config` 才手写 |
| `python/sglang/srt/server_args.py:517` | 第一个字段 `model_path`（无默认值，必填） |
| `python/sglang/srt/server_args.py:3637` | 最后一个字段 `msprobe_dump_config`（字段区间共 3,121 行，476 个字段，平均每字段约 6.6 行——含大段多行 help 文本） |
| `python/sglang/srt/server_args.py:3643`-`3644` | `__post_init__`，只有一行：调用 `_run_resolution_pipeline()` |
| `python/sglang/srt/server_args.py:3646`-`3853` | `_run_resolution_pipeline`——**调度器本体**，本身不做校验/推断，只按依赖顺序调 88 个 `_handle_*` 方法，方法体分布在 `3855`-`8759` |
| `python/sglang/srt/server_args.py:4649`-`4723` | `_disable_tc_piecewise_cudagraph_if_incompatible`——18 条互斥规则，命中任一条即无条件覆盖 `cuda_graph_config.prefill.backend` |
| `python/sglang/srt/server_args.py:4860`-`4990` | `_handle_gpu_memory_settings`——按 GPU 显存分 6 档自动推断 `chunked_prefill_size`/`cuda_graph_config.decode.max_bs`/`mem_fraction_static`（仅在字段为 `None` 时介入） |
| `python/sglang/srt/server_args.py:5988`-`6058` | `_handle_attention_backend_compatibility`——解析注意力后端并联动关闭不兼容的 CUDA Graph |
| `python/sglang/srt/server_args.py:8761`-`8764` | `add_cli_args` 静态方法起点，第一行就是 `add_cli_args_from_dataclass(parser, ServerArgs)` |
| `python/sglang/srt/server_args.py:8766`-`9020` | 29 处手写 `parser.add_argument`：3 个动态 choices + 1 个 `--config` 元参数 + 25 个废弃标志重定向 |
| `python/sglang/srt/server_args.py:9022`-`9030` | `from_cli_args` 类方法——`argparse.Namespace` → `ServerArgs` 实例的唯一入口 |
| `python/sglang/srt/server_args.py:9970`-`9986` | `PortArgs` dataclass 定义起点（10 个字段，纯 IPC 寻址信息） |
| `python/sglang/srt/arg_groups/arg_utils.py:58` | `A = Annotated`——CLI 元数据的载体就是标准库 `typing.Annotated` 的别名 |
| `python/sglang/srt/arg_groups/arg_utils.py:61`-`83` | `Arg` frozen dataclass——CLI 生成用的全部元数据字段（`help`/`choices`/`aliases`/`type_parser`/`nargs`/`action`/`no_cli`/`resolvable`…） |
| `python/sglang/srt/arg_groups/arg_utils.py:85`-`99` | `NS` frozen dataclass——命名空间路径标记，与 `Arg` 分离存放 |
| `python/sglang/srt/arg_groups/arg_utils.py:101`-`119` | `namespace_of()`——从 `Annotated` 元数据里抠出 `{field_name: 命名空间路径}`，`lru_cache` 缓存 |
| `python/sglang/srt/arg_groups/arg_utils.py:218`-`337` | `add_cli_args_from_dataclass`——**"29 撑起 476"的答案本体**，通用反射循环 |
| `python/sglang/srt/arg_groups/overrides.py:179`-`208` | `run_post_process_pass`——声明式覆盖的执行入口，写入"声明暂存区"而不直接改字段 |
| `python/sglang/srt/arg_groups/overrides.py:257`-`266` | `materialize_declarations`——解析流水线收尾，把暂存区一次性物化到字段上 |
| `python/sglang/srt/runtime_context.py:677`-`722` | `_build_config_bags`——用 `NS(...)` 元数据把扁平的 `ServerArgs` 投影成嵌套配置树 |
| `test/registered/unit/test_server_args_namespaces.py:18`-`50` | 命名空间覆盖率 lint：锁定 21 个合法命名空间，断言每个字段都要有 `NS`，且 `len(nsmap) >= 440` |

（18 条，超过硬指标要求的 10 条；行号来自实读源码，非 `_lab` JSON 直接复制——原因见 `## 7`。）

## 3. 核心数据结构

### 3.1 `Arg`——CLI 元数据（`python/sglang/srt/arg_groups/arg_utils.py:61`-`83`）

```python
@dataclasses.dataclass(frozen=True)
class Arg:
    help: str = ""
    choices: Optional[list] = None
    aliases: Optional[List[str]] = None
    cli_name: Optional[str] = None
    type_parser: Optional[Callable] = None
    nargs: Optional[str] = None
    required: Optional[bool] = None
    action: Optional[Any] = None
    action_kwargs: Optional[dict] = None
    const: Optional[Any] = None
    no_cli: bool = False
    resolvable: bool = False
```

两个字段值得单独说：`resolvable=True` 是"这个字段允许被解析流水线的声明/物化机制覆盖"的白名单标记，`resolvable_fields()`（`python/sglang/srt/arg_groups/arg_utils.py:122`-`137`）读它来构建白名单——`## 3.3` 展开这个机制。`no_cli=True` 让 `add_cli_args_from_dataclass` 跳过该字段（`python/sglang/srt/arg_groups/arg_utils.py:244`-`245`），全文件只有 **8 处**（`grep -c "no_cli=True"` 实测）；`## 3.6` 会展开——它的主要用途不是"SDK-only 字段"这种直觉印象，而是给废弃标志腾位置。

一个字段甚至可以不写 `Arg(...)`，直接用裸字符串当 help 文本：`_unwrap_annotated`（`python/sglang/srt/arg_groups/arg_utils.py:147`-`163`）把 `Annotated[T, "帮助文本"]` 等价处理成 `Arg(help="帮助文本")`。这也是为什么 476 个字段里绝大多数看起来"没有显式声明太多东西"——最简单的写法就是一行注解。

### 3.2 `NS`——命名空间标记，与 `Arg` 分离存放

```python
@dataclasses.dataclass(frozen=True)
class NS:
    path: str
```

`python/sglang/srt/arg_groups/arg_utils.py:93`-`96` 的类文档字符串给出了这个设计的直接动机：**把命名空间标记从 `Arg` 里拆出来单独作为 `Annotated` 的第三个参数，是为了让"给已有的裸字符串/多行注解字段补命名空间"这件事只需要**追加**一个 `NS(...)` 元素，不用把已有的 `Arg(...)` 调用重写一遍**。这解释了为什么 476 个字段中有些是 `A[bool, "help text", NS("memory")]` 这种三元组，有些是 `A[str, Arg(help=..., choices=...), NS("schedule")]` 这种更复杂的形式——两种写法共存正是"迁移期"的产物。

`namespace_of(cls)`（`python/sglang/srt/arg_groups/arg_utils.py:101`-`119`）用 `get_type_hints(cls, include_extras=True)` 拿到每个字段的完整 `Annotated` 元组，扫描其中的 `NS` 实例取出 `path`，产出 `{field_name: "a.b.c"}` 的字典，`lru_cache` 装饰保证只算一次。**这个映射不是文档性质的——`python/sglang/srt/runtime_context.py:677`-`722` 的 `_build_config_bags` 直接用它把扁平的 `ServerArgs` 实例投影成一棵嵌套树**：每个 `NS("exec.moe")` 字段最终能通过 `ctx.exec.moe.<field_name>` 访问到。命名空间路径按 `.` 分层建 `_ConfigBag` 节点，`python/sglang/srt/runtime_context.py:705`-`709` 和 `:716`-`720` 各有一处显式的"叶子和子分组同名"冲突检测，宁可在启动时报错也不允许静默遮蔽。

`test/registered/unit/test_server_args_namespaces.py:18`-`39` 锁定了合法命名空间的完整清单（21 个）——本篇 `## 4` 的分族表就是照这个清单核对的，而不是凭直觉分类。

### 3.3 声明-物化模式：`resolvable_fields` / `run_post_process_pass` / `materialize_declarations`

这是 `__post_init__` 里 88 个 handler 方法能够互相依赖、却不至于产生"谁写在谁前面谁的值就被覆盖"这种脆弱顺序耦合的关键机制。核心矛盾是：**很多字段的最终值依赖模型架构（`hf_config.architectures[0]`）**，比如 `MistralLarge3ForCausalLM` 强制 `dtype="bfloat16"`（`python/sglang/srt/arg_groups/overrides.py:69`-`72` 的 `MODEL_OVERRIDES` 常量表），但模型架构相关的判断逻辑分散在几十个 handler 里，如果每个 handler 都直接 `self.dtype = "bfloat16"`，后写的 handler 会不知不觉覆盖前面某个 handler 的判断，产生"最后一个写的赢，但没人知道顺序为什么这样"的心智负担。

SGLang 的解法（`python/sglang/srt/arg_groups/overrides.py:14`-`27` 模块文档字符串原话："model code never mutates `ServerArgs` fields imperatively"）：

1. 任何 handler 需要覆盖字段时，不直接赋值，而是调用 `run_post_process_pass(self, fn)`（`python/sglang/srt/arg_groups/overrides.py:179`-`208`），`fn` 是一个只读函数，接收一个 `ResolvedView`（叠加了此前所有声明的只读视图），返回 `{field: value}` 字典。
2. 这个字典被追加进 `self._resolved_overrides` 这个"声明暂存区"列表（`python/sglang/srt/arg_groups/overrides.py:195`-`205`），**不写字段本身**。
3. 中途需要读"目前解析到哪一步"的 handler，用 `resolved_view(server_args)`（`python/sglang/srt/arg_groups/overrides.py:269`-`274`）拿到叠加了暂存区的只读视图，而不是直接读字段——直接读字段会读到"物化前的旧值"。
4. `_run_resolution_pipeline` 收尾处（`python/sglang/srt/server_args.py:3851`-`3853`）调用一次 `materialize_declarations(self)`（`python/sglang/srt/arg_groups/overrides.py:257`-`266`），把暂存区里的所有声明按追加顺序一次性写回字段（"gate order：后写的赢"），从这一刻起 `_declarations_materialized = True`，往后任何进程里的任何读者都可以直接读字段，不用再走 `resolved_view`。

`resolvable_fields(cls)`（`python/sglang/srt/arg_groups/arg_utils.py:122`-`137`）是这套机制的白名单：只有 `Arg(resolvable=True)` 的字段允许被声明覆盖，本篇 `## 4` 表格里的 `quantization`/`page_size`/`attention_backend`/`ep_size`/`enable_dp_attention`/`disable_overlap_schedule` 等旋钮都带这个标记——这不是巧合，恰恰是"最容易被模型架构/硬件条件覆盖"的那批旋钮。

### 3.4 旋钮分族：21 个命名空间怎么分布

`## 7` 会讲到 `_lab/out/api_surface.json` 里字段 `type` 字符串被截断导致的统计失真——这里给出**用 `ast.get_source_segment` 对源码逐字段取完整注解文本、重新统计出的真实分布**（476 个字段，21 个命名空间，与 `test/registered/unit/test_server_args_namespaces.py:18`-`39` 锁定的合法命名空间清单逐一核对无遗漏）：

| 命名空间 | 字段数 | 代表字段（行号） |
|---|---|---|
| `serving` | 55 | `host`（`1307`）、`port`（`1308`）、`api_key`（`1389`）、`tool_call_parser`（`1454`）——HTTP/协议层与外部集成 |
| `observability` | 43 | `enable_metrics`（`1577`）、`log_level`（`1527`）——日志、指标、追踪 |
| `model` | 42 | `quantization`（`683`）、`kv_cache_dtype`（`705`）、`modelopt_quant`（`735`）——模型来源、精度、量化（**量化不是独立命名空间，是 `model` 里的一个子集**，约 7-10 个字段名含 `quant`/`dtype`/`modelopt`） |
| `schedule` | 38 | `chunked_prefill_size`（`826`）、`page_size`（`921`）、`schedule_policy`（`858`）、`disable_overlap_schedule`（`993`） |
| `parallel` | 37 | `tp_size`（`1040`）、`pp_size`（`1056`）、`dp_size`（`1072`）、`ep_size`（`2380`） |
| `spec` | 37 | `speculative_algorithm`（`2116`）——投机解码全部旋钮单独成一个命名空间 |
| `disagg` | 26 | `disaggregation_mode`（`3159`）——PD 分离全部旋钮单独成一个命名空间 |
| `memory` | 22 | `disable_radix_cache`（`967`）——前缀缓存与内存池策略 |
| `mm` | 19 | 多模态处理相关 |
| `lora` | 14 | `enable_lora`（`2937`）、`lora_paths`（`2961`） |
| `device` | 12 | `watchdog_timeout`（`1276`）等硬件探测与运行时环境 |
| `exec.moe` | 36 | MoE 专家路由/负载均衡内核细节 |
| `exec.kernel` | 22 | `attention_backend`（`1728`）——内核选型 |
| `exec.graph` | 18 | `cuda_graph_config`（`1897`） |
| `exec.mamba` | 17 | Mamba/线性注意力状态空间模型专属 |
| `exec.comm` | 13 | 集合通信后端细节 |
| `exec.features` | 12 | 零散的执行期特性开关 |
| `exec.offload` | 5 | `cpu_offload_gb`（`3051`） |
| `exec.overlap` | 3 | 调度重叠相关的执行细节 |
| `exec.dllm` | 3 | 扩散语言模型（diffusion LLM）推理 |
| `exec.deterministic` | 2 | 确定性推理模式 |

两个观察值得单独指出：**`exec.*` 前缀的 10 个子命名空间合计 131 个字段（约占 27.5%）**，比任何单一的顶层命名空间都多——这说明 476 个字段里超过四分之一是"内核/执行路径选型"这类偏底层的旋钮，不是"模型多大、并行度多少"这类用户第一眼会调的参数；**`quantization`/`kv_cache_dtype` 等通常被单独讨论的"量化"话题在命名空间层面并不独立**，混在 `model` 的 42 个字段里——这是源码自己的分类方式，本篇按任务要求单列"量化"只是叙述上的切分，不代表 SGLang 团队把量化当成独立子系统看待。

### 3.5 类体的整体形状

`ServerArgs` 这一个类，物理上由三段组成：**476 个 `A[...]` 注解字段**（`python/sglang/srt/server_args.py:517`-`3637`，约 3,121 行）+ **88 个解析/校验辅助方法**（`3646`-`8759`，约 5,113 行）+ **CLI 生成与实例化的两个入口方法**（`add_cli_args` / `from_cli_args`，`8761`-`9030`，约 270 行）。字段声明只占整个类体不到三分之一的篇幅——**多数代码量花在了"字段之间怎么互相约束"上，不是"字段本身怎么声明"上**，这条观察是 `## 5` 讨论工程代价的直接证据。

### 3.6 废弃标志怎么和反射循环共存：`no_cli` + 四个 `Deprecated*Action`

`## 4.1` 会讲到 29 个手写 `add_argument` 里有 25 个是废弃标志重定向。这里先把机制讲清楚，因为它直接解释了 `no_cli=True` 真正的主要用途——**不是"SDK-only 便利字段"，而是给废弃标志腾出 CLI 名字空间**。

废弃标志的重定向壳分四种（全部定义在 `python/sglang/srt/arg_groups/argparse_actions.py`），行为依次递进：

| 类名（行号） | `__call__` 做什么 | 典型场景 |
|---|---|---|
| `DeprecatedAction`（`32`-`44`） | 打印警告；若带 `error_message` 直接 `parser.error()` 拒绝启动 | 功能已彻底移除，没有替代路径（如 `--enable-expert-distribution-metrics`） |
| `DeprecatedStoreTrueAction`（`47`-`70`） | 打印警告，把 `dest` 无条件设成 `True` | 旧的布尔开关重命名（如 `--enable-prefill-context-parallel` → `enable_prefill_context_parallel`） |
| `DeprecatedStoreConstAction`（`73`-`98`） | 打印警告，把 `dest` 设成一个**固定常量**（可以和字段名本身不一致的值） | 旧布尔开关被拆分进新的多值配置（如 `--disable-piecewise-cuda-graph` 把 `Backend.DISABLED` 写进 `cuda_graph_backend_prefill`） |
| `DeprecatedAliasStoreAction`（`101`-`113`） | 打印警告，把用户传的值原样转发到新 `dest` | 纯改名，值语义不变（如 `--nsa-prefill-backend` → `dsa_prefill_backend`） |

关键在 `dest=`：这四类 action 在 `parser.add_argument(...)` 里几乎都显式指定了 `dest=<字段名>`，让旧 CLI 名字和字段共享同一个存储位置。多数情况下这不需要 `no_cli`——比如 `--nsa-prefill-backend`（`python/sglang/srt/server_args.py:8839`-`8840`）把 `dest` 指向 `dsa_prefill_backend`，但这个字段自己的反射生成名字是**另一个字符串** `--dsa-prefill-backend`，两个 flag 文本不同，argparse 允许它们共享一个 dest，不需要摘掉反射注册。**真正需要 `no_cli=True` 的场景，是废弃 flag 的字面文本和字段自身的反射生成名字完全相同**——例如 `--enable-prefill-context-parallel`（`python/sglang/srt/server_args.py:8971`-`8977`）要把 `dest` 指向 `enable_prefill_context_parallel`，而这个字段名反射出来的 CLI 名字恰好就是 `--enable-prefill-context-parallel` 本身；`--disable-cuda-graph`/`--enable-flashinfer-allreduce-fusion` 是同样情形。如果不加 `no_cli=True`，反射循环会用同一个 flag 文本再注册一次，argparse 遇到重复的 `add_argument("--enable-prefill-context-parallel", ...)` 会直接抛 `ArgumentError`（重复选项字符串），服务器连命令行都解析不了。**`no_cli=True` 就是用来避免这种"字段名撞上自己废弃前身"的自冲突**：把字段从反射循环里摘出来，只留手写的 `add_argument` 一条注册路径。

实测 `no_cli=True` 全文件只有 8 处（`python/sglang/srt/server_args.py:1166`/`1169`/`1170`/`1171`/`1956`/`2069`/`2278`/`2634`），其中 **6 处**是废弃标志的直接背书对象——`enable_dsa_prefill_context_parallel`/`dsa_prefill_cp_mode`/`enable_prefill_context_parallel`/`prefill_cp_mode`（`1166`-`1171`，分别被 `--enable-dsa-prefill-context-parallel`/`--enable-nsa-prefill-context-parallel`/`--dsa-prefill-cp-mode`/`--nsa-prefill-cp-mode`/`--enable-prefill-context-parallel`/`--prefill-cp-mode` 六个废弃 flag 共同或分别指向）、`disable_cuda_graph`（`1956`，对应 `--disable-cuda-graph`）、`enable_flashinfer_allreduce_fusion`（`2069`，对应同名的 `store_true` 手写调用，`python/sglang/srt/server_args.py:9015`-`9019`）。**只有 2 处是真正意义上的"内部/派生字段，天生没有 CLI 语义"**：`_speculative_draft_quantization_explicitly_set`（`2277`-`2279`，下划线前缀的内部记账字段）和 `uses_mamba_radix_cache`（`2628`-`2638`，help 文本明确写着"(Derived) ... resolved from the model architecture, no CLI surface"）。

同一份反射循环里还藏着第三种escape hatch——`Arg.action` 可以整体接管某个字段的注册方式，不止用于废弃标志。`dllm_fdfo`（`python/sglang/srt/server_args.py:3147`-`3154`）是全部 476 个字段里**唯一一个默认值为 `True` 的裸 `bool` 字段**，如果按通用分支（`python/sglang/srt/arg_groups/arg_utils.py:314`-`319`，`bool` 类型统一转 `action="store_true"`）注册，用户将**没有任何 CLI 方式把它显式设回 `False`**——`store_true` 只能把值从默认改成 `True`，命令行不传就还是默认值，没有对称的"关闭"语义。`dllm_fdfo` 的字段定义手动指定了 `action=argparse.BooleanOptionalAction`（`3147`-`3154`，默认值 `True`），换来一对 `--dllm-fdfo`/`--no-dllm-fdfo`，这正是 `## 6` 会提到的、vLLM 对**所有** `bool` 字段的默认处理方式——SGLang 只在唯一需要的地方手动开了这个口子。

## 4. 主流程走读：一次 `ServerArgs()` 构造做了什么

### 4.1 从 CLI 到解析流水线收尾

1. **CLI 解析阶段**：`argparse.ArgumentParser` 由 `ServerArgs.add_cli_args(parser)`（`python/sglang/srt/server_args.py:8761`）填充。第一行 `add_cli_args_from_dataclass(parser, ServerArgs)`（`8764`）用 `get_type_hints` 拿到全部 476 个字段的注解，逐个走 `python/sglang/srt/arg_groups/arg_utils.py:233`-`337` 的分支树：跳过无 `Arg`/无裸字符串 help 的字段 → 处理自定义 `action` → 拆 `Optional[X]` → 识别 `Literal[...]` 自动转 `choices` → 识别 `List[X]` 自动配 `nargs="+"` → `bool` 类型转成 `action="store_true"` → 兜底走标量 `type=` 路径。之后追加 29 个手写 `add_argument`（`8766`-`9020`）：3 个动态 choices（`--reasoning-parser`/`--tool-call-parser`/`--kv-canary-real-data`，choices 依赖运行时注册表，写死在注解里做不到）、1 个 `--config` 元参数（不对应任何字段，只是"从 YAML 读配置"的开关）、25 个废弃标志的重定向壳（`DeprecatedAction`/`DeprecatedAliasStoreAction`/`DeprecatedStoreConstAction`/`DeprecatedStoreTrueAction`，定义在 `python/sglang/srt/arg_groups/argparse_actions.py:32`/`47`/`73`/`101`）。
2. **`from_cli_args`**（`python/sglang/srt/server_args.py:9022`-`9030`）把 `argparse.Namespace` 转成 `ServerArgs(**kwargs)` 调用——只挑 `hasattr(args, field.name)` 为真的字段传参，`no_cli=True` 的字段（在 `Namespace` 上压根不存在）自然落回 dataclass 默认值。
3. **`__post_init__` → `_run_resolution_pipeline`**（`python/sglang/srt/server_args.py:3643`-`3853`）：先设 `self._resolved_overrides = []`（声明暂存区），然后按依赖顺序跑 88 个 handler。顺序不是随意的——`_run_resolution_pipeline` 的文档字符串（`3651`-`3667`）明确写了"按依赖域排序，不按历史插入顺序"，并把顺序分成十个阶段：内部/引导 → API/网络/协议校验 → 模型路径解析 → 硬件/平台 → 模型专属调整 → 并行策略 → 内核/注意力后端 → CUDA Graph → 内存/缓存 → 高级/调试特性。中途在 `model_path.lower() in ["none", "dummy"]` 时会提前 `return`（`3687`-`3688`）——dummy 模型跳过几乎全部后续校验，这是刻意为之的"哑模型边界"（文档字符串规则 2）。
4. **收尾物化**（`3847`-`3853`）：调用 `materialize_declarations(self)`，把 88 个 handler 里通过 `run_post_process_pass` 声明式覆盖的字段一次性写回，`_declarations_materialized` 置真。

### 4.2 十三个真正影响吞吐/延迟/显存的旋钮

476 个字段里，多数是"要不要开某个专属特性"（LoRA 路径、扩散 LLM 算法、某个厂商特有的通信后端）这类只在特定场景下才碰的开关，日常调优真正反复要动的旋钮集中在并行、内存、调度、内核选型这四个命名空间。下表挑出 13 个，覆盖 `## 3.4` 分族表里权重最高的几类；"调大/调小"只给方向性影响和副作用，不给数字（本机无 GPU，不产出实测数据）。

| 参数名（命名空间） | 默认值 | 作用 | 调大会怎样 | 调小会怎样 |
|---|---|---|---|---|
| `tp_size`（`parallel`，`1040`） | `1` | 张量并行度，权重与矩阵计算按维度切分到多卡 | 单卡显存压力降低（权重被摊薄），但引入跨卡 all-reduce，通信可能成为新瓶颈 | 显存压力集中在少数卡上，甚至装不下模型，但省掉跨卡同步开销 |
| `pp_size`（`parallel`，`1056`） | `1` | 流水线并行度，按层切成多阶段接力执行 | 单阶段显存需求降低，但引入流水线气泡（bubble），需要足够并发请求填满流水线才划算 | 每张卡要装更多层，显存压力回升，但没有气泡开销 |
| `ep_size`（`parallel`，`2380`） | `1` | 专家并行度（MoE 模型），专家权重分散到多卡 | 单卡专家显存占用降低，但 all-to-all 专家路由通信量上升 | 专家权重集中在少数卡，显存压力大，通信压力小 |
| `enable_dp_attention`（`parallel`，`1178`） | `False` | 注意力层走数据并行、FFN 走张量并行的混合模式（`resolvable=True`，仅限 DeepSeek-V2/Qwen2-3 MoE 等架构） | 打开后按 DP 组切分注意力批大小，显著降低单卡 KV cache 占用，但要求 `dp_size == tp_size` | 关闭则注意力与 FFN 用同一套并行策略，实现简单，但注意力层 KV cache 无法从 DP 获益 |
| `mem_fraction_static`（`schedule`，`799`） | `None`（按 GPU 显存分 6 档自动推断，`4860`-`5041`） | 静态显存占比：模型权重 + KV cache 池 / 总显存 | 给 KV cache 池更多空间，支撑更多并发/更长上下文，但压缩激活值与 CUDA Graph buffer 的余量，OOM 风险上升 | 给激活值/CUDA Graph 更多安全余量，但 KV cache 池变小，并发数或最大上下文被压缩 |
| `chunked_prefill_size`（`schedule`，`826`） | `None`（按显存分档，`2048`~`16384`） | 分块 prefill 每步允许处理的最大 token 数；设为 `-1` 关闭分块 | 长 prompt 更快跑完 prefill（更少步数），但单步激活值显存占用上升，且挤占同批 decode 请求的 token 名额 | 每步给 prefill 的名额更少，长 prompt 需要更多步，但给 decode 请求让出更多每步预算 |
| `max_running_requests`（`schedule`，`804`） | `None`（不设上限，受内存池实际可用空间约束） | 同时处于 RUNNING 状态的请求数上限 | 允许更多请求并发，提升吞吐，但每请求分到的 KV cache/计算资源变少，单请求延迟可能上升 | 限制并发数，单请求资源更充足，但吞吐降低，更多请求排队等待 |
| `page_size`（`schedule`，`921`） | `None`（由 `_page_size_default` 按后端解析，`6212`-`6220`） | KV cache 按多少 token 为一页分配 | 页表项数量减少、内存管理开销降低，但内存碎片化（页内浪费）风险上升 | 内存利用更精细，但页表项更多；部分注意力后端（如 MLA）对取值有强约束 |
| `schedule_policy`（`schedule`，`858`） | `"fcfs"` | 等待队列的排队策略，6 选一 | 换成 `"lpm"` 等前缀感知策略能提升前缀缓存命中率，但排序本身有计算开销，且队列超过 128 个请求时自动降级回 fcfs（`sglang:python/sglang/srt/managers/schedule_policy.py:291`-`293`，工程上限） | 保持 `fcfs` 简单可预测，但完全不利用请求间的前缀共享关系 |
| `disable_radix_cache`（`memory`，`967`） | `False` | 是否关闭 RadixAttention 前缀缓存 | 打开（禁用缓存）后每个请求从零算 KV，失去前缀复用的计算节省，但省掉前缀树维护/淘汰开销，适合请求间几乎不共享前缀的场景 | 保持默认在共享前缀多的场景大幅减少重复计算，但要为前缀树本身付出内存与维护成本 |
| `kv_cache_dtype`（`model`，`705`） | `"auto"`（跟随模型 dtype） | KV cache 存储的数据类型 | 换成 `fp8_e4m3`/`fp8_e5m2` 等低精度显著压缩 KV cache 显存占用，支撑更长上下文/更大并发，但精度损失可能影响长序列生成质量 | 保持 `auto` 精度更高，但 KV cache 显存占用是低精度方案的数倍 |
| `attention_backend`（`exec.kernel`，`1728`） | `None`（按硬件/模型自动探测，`resolvable=True`） | 选择注意力层的计算内核实现 | 显式指定高性能后端（如 `trtllm_mha`/`trtllm_mla`）在满足硬件门槛（SM90+/SM100/SM120）时性能更好，但会触发 `## 5` 决策 4 提到的多条兼容性联动，可能连带关闭 CUDA Graph | 保持 `None` 让系统自动探测，兼容性最稳妥，但不保证选到当前硬件上的最优实现 |
| `disable_overlap_schedule`（`schedule`，`993`） | `False`（`resolvable=True`） | 是否关闭 CPU 调度与 GPU 模型执行的重叠 | 打开（禁用重叠）后调度器等 GPU 执行完再调度下一步，延迟增加但行为更容易复现调试（消除调度与执行的竞态） | 保持默认让 CPU 调度与 GPU 计算流水线化，提升吞吐，但增加调试复杂度 |

## 5. 设计决策与代价

### 决策 1：CLI 生成走"反射循环"而不是"手写映射表"

- **为什么这么设计**：476 个字段如果每个都手写一行 `parser.add_argument("--xxx", ...)`，光是 `add_cli_args` 这一个方法就要多出至少 3,000-4,000 行（参照 vLLM 每个字段一行的写法估算，见 `## 6`），而且每加一个新字段都要同时改两处（`dataclass` 声明 + `add_argument` 调用），两处不同步是最常见的"CLI 有这个参数但代码不认"或反过来的 bug 来源。用 `Annotated[T, Arg(...)]` 把 CLI 元数据和字段声明**焊在一起**，加字段只改一处，`add_cli_args_from_dataclass` 自动跟上。
- **不这样会怎样**：`add_cli_args` 会退化成一个几千行的巨型方法，且随着字段数增长线性变长；`server_args.py` 本身已经 10,142 行，如果字段和 CLI 注册各占一份，全文件规模会显著膨胀，且两份之间的漂移几乎不可避免（历史上 vLLM 就经历过手写 `add_argument` 与 dataclass 字段脱节的问题，虽然 vLLM 近期也转向了自动生成 kwargs，见 `## 6`）。
- **什么时候可以不这样**：字段数量很小（几十个以内）、且很少变动的项目，手写映射表的可读性反而更好——一眼就能看到某个 CLI 参数对应什么行为，不需要跳到 `arg_utils.py` 理解反射规则。476 个字段的规模已经越过了"手写更清晰"的门槛。

### 决策 2：`NS(...)` 命名空间标记独立于 `Arg` 存在

- **为什么这么设计**：`python/sglang/srt/arg_groups/arg_utils.py:93`-`96` 的文档字符串给出了直接理由——如果把命名空间信息塞进 `Arg` 本身，给已有的约 400 个裸字符串/`Arg(...)` 字段补命名空间，就要逐个重写这些调用；拆成独立的 `NS(...)` 元素，只需要在 `Annotated[...]` 末尾追加一项，是纯增量的迁移路径。这也解释了为什么全部 476 个字段最终都能补齐 `NS`（`## 2` 引用的覆盖率测试），而不是只有新字段才有——迁移成本被压到最低。
- **不这样会怎样**：命名空间信息如果不独立出来，要么揉进 `Arg.help` 文本（不可机器读取，`namespace_of` 这类工具就做不出来），要么另开一个手写映射表（回到"两处声明容易漂移"的老问题）。
- **什么时候可以不这样**：如果一开始设计时就用嵌套 dataclass（每个子系统一个 `XxxConfig`，像 vLLM 那样），命名空间天然就是 Python 的属性访问路径，根本不需要一个额外的 `NS` 标记层。SGLang 选了相反的路线——保持 `ServerArgs` 扁平，靠反射在运行时投影出嵌套视图（`## 3.2` 的 `_build_config_bags`）。两条路线各有代价：嵌套 dataclass 让 IDE 补全和类型检查更直接，但每加一层嵌套都要改多个文件（子配置类定义 + 组装点 + CLI 分组）；扁平字段 + `NS` 标记让新增字段只改一处，但命名空间树是运行时反射出来的，IDE 补全不到 `ctx.exec.moe.xxx` 这种路径，只能补全到 `ServerArgs` 的字段名本身。

### 决策 3：声明-物化模式取代直接字段赋值

- **为什么这么设计**：88 个 handler 方法之间存在真实的依赖关系（`## 4` 提到的十阶段顺序），如果允许任意 handler 直接 `self.xxx = yyy`，读者要靠通读全部 88 个方法的调用顺序才能确定"某个字段最终是谁写的"；把"声明"和"写入"拆成两步，中途读取一律走只读的 `resolved_view`，只有流水线收尾时才真正物化，把"谁最后写的赢"这件事收敛到一个单点（`materialize_declarations`），而不是散布在 88 个方法体内部。
- **不这样会怎样**：`overrides.py` 模块文档字符串（`14`-`19`）里的原话是"model code never mutates `ServerArgs` fields imperatively"——反过来说，如果允许直接赋值，模型专属覆盖（`MODEL_OVERRIDES` 表 + `register_model_override` 装饰器）和通用 handler 之间就会出现顺序竞争：一个 handler 先按硬件条件把 `attention_backend` 设成 A，另一个 handler 后来因为模型架构又设成 B，读者除了通读源码顺序，没有别的办法判断最终值是 A 还是 B。
- **什么时候可以不这样**：如果一个字段从始至终只被一个 handler 触碰（没有跨 handler 的竞争关系），直接赋值也不会有歧义——事实上大多数字段确实这样处理（`_handle_gpu_memory_settings` 里 `if self.chunked_prefill_size is None: self.chunked_prefill_size = ...` 这类"只在未设置时填充"的写法就是直接赋值，见 `python/sglang/srt/server_args.py:4891`-`4892`）。声明-物化模式只在"多个 handler 可能对同一字段有不同意见"的场景才必要，`resolvable_fields()` 白名单的存在本身就说明了这一点——不是所有字段都需要这层间接。

### 决策 4：不兼容组合选择"静默覆盖 + 日志"而不是一律报错

- **为什么这么设计**：`_disable_tc_piecewise_cudagraph_if_incompatible`（`python/sglang/srt/server_args.py:4649`-`4723`）列的 18 条规则（DP attention、LoRA、多模态、GGUF 量化、PD 分离…）里，绝大多数是"这个特性天生和 tc_piecewise CUDA Graph 冲突"，而不是"用户配置错了"——如果每条冲突都报错退出，用户光是想开个 LoRA 就要先猜到"哦原来还要显式关掉 CUDA Graph 后端"，体验会非常差。SGLang 选择"探测到冲突就自动降级 + `logger.warning`"（例如 `python/sglang/srt/server_args.py:6011`-`6013` 用 `torch_native` 后端时自动关 CUDA Graph 并打警告），让常见组合"开箱可用"。
- **不这样会怎样**：如果把这类组合冲突全部改成 `raise ValueError`，用户每次尝试一个新特性组合都可能被拦下来，需要反复试错才能拼出一个能跑的命令行——这对"探索性调参"极不友好，尤其是新手。
- **什么时候可以不这样**：涉及正确性而非性能的冲突，SGLang 确实选择了报错而不是静默处理——`_handle_dcp_validation`（`python/sglang/srt/server_args.py:4034`-`4064`）对 `--dcp-comm-backend` 与 `--dcp-size` 的非法组合直接 `raise ValueError`，因为这类组合不是"性能次优"而是"根本跑不通/语义不成立"。区分标准大致是：**降级只会变慢或退化到更保守的实现路径→静默覆盖；组合本身没有合法语义或会导致运行时崩溃→报错**。这条界线本身没有在源码里被显式写成一条规则，是本库读完 88 个 handler 后归纳出来的（本库推断，依据：`_disable_*` 系列方法全部走静默覆盖，`_handle_*_validation` 系列方法里出现 `raise` 的比例明显更高）。

### 决策 5（本库推断）：476 个旋钮本身是一种代价，不只是"功能多"

前四条决策讲的都是"某个具体机制为什么这样设计"，这一条要单独处理一个更上层的问题：**476 这个数字本身，不管背后的生成机制多精巧，都是要付代价的。这条整节标注为本库推断——源码和文档都没有正面讨论"旋钮数量是否应该被控制"这件事，以下是本库读完全部 88 个 handler 之后的判断。**

- **为什么会长到 476 个**：这不是一次性设计出来的，是几十个特性（LoRA、投机解码、PD 分离、专家并行、扩散 LLM、确定性推理…）各自贡献一批专属旋钮、逐个 PR 累积的结果——`## 3.4` 的命名空间分布本身就是证据：`exec.*` 前缀下 10 个子命名空间合计 131 个字段，对应的是 10 个不同的底层执行路径特性，每个特性平均贡献 13 个旋钮。**反射式 CLI 生成机制（`## 5` 决策 1）恰恰是让这种增长得以持续的原因之一**——正因为加一个字段不需要同步改 `add_cli_args`，新增旋钮的边际成本被压得很低，"要不要加这个开关"的决策门槛也跟着降低。工具让某件事变容易，某件事就会变多，这是一个常见的因果链条，不是 SGLang 独有的问题。
- **代价是什么**：
  1. **组合爆炸让"全量测试覆盖"在数学上不可能。** 476 个字段里哪怕只有 100 个是布尔开关，理论组合数就是 2^100 量级，而 CI 能跑的组合数是有限的几十到几百种（本库未去核实 SGLang CI 实际覆盖了多少种 `ServerArgs` 组合，未查证）。`## 5` 决策 4 提到的 18 条 CUDA Graph 互斥规则、`_handle_attention_backend_compatibility` 里散布的十几处兼容性判断，都是"社区在事后一条一条把踩过的坑写成规则"的产物，不是"设计阶段穷举过全部组合"的产物——规则数量本身就是"有多少种组合被人踩过"的下限估计，真实的不兼容组合数只会更多，只是还没人踩到。
  2. **默认值成为事实标准，比文档更有约束力。** 大多数用户不会去读 88 个 handler 的推断逻辑，只会用默认配置跑起来——`mem_fraction_static`/`chunked_prefill_size` 这类"按显存分档自动推断"的默认值，实际上定义了"SGLang 在每种硬件上的标准形态"，比任何文档描述都更能决定大多数用户实际体验到的性能特征。一旦默认值被大量用户依赖，未来想调整分档阈值（比如把 `<35GB` 那一档的 `chunked_prefill_size` 从 2048 改成别的值）就要考虑"这会不会让某个已经在生产环境跑的配置意外变化"——默认值的可变更性会随着使用规模增长而下降,这和显式配置的旋钮完全不是一回事。
  3. **调参从"读文档"退化成"读源码猜行为"。** `## 0`/`## 7` 已经指出至少两类"设置被覆盖"和一类"覆盖只在特定条件下生效"的行为——用户面对性能不及预期时，没有一份文档能穷举"你设的这个值在你的硬件/模型/并行度组合下最终会不会生效"，唯一可靠的答案在 88 个 handler 的源码里。这正是"调参变成玄学"的字面意思：观测到的行为是真实的，但因果链条只有读源码才能重建，社区里流传的"经验参数"很容易只是巧合地绕开了某条隐藏规则，换个模型或硬件就失效。
- **有没有替代设计**：
  - **profile 驱动**——把"硬件档位 → 一组旋钮组合"这件事从"88 个 if-else 散布在 `__post_init__`"改成一份可审查、可版本化的 profile 表（类似 `_handle_gpu_memory_settings` 里那 6 档 GPU 显存分类的思路,但独立成配置文件而不是内嵌逻辑），用户选一个 profile 名字而不是逐个设置几十个旋钮，能显著降低"猜行为"的成本，代价是 profile 之间的组合仍然需要人工设计和验证，只是把组合爆炸从"旋钮乘旋钮"降到了"profile 乘 profile"，问题规模变小但没有消失。
  - **自动调参**——运行一次轻量 profiling（类似部分推理引擎在启动时做的 warmup + 显存探测），根据实测结果而不是硬编码的显存分档表来设置 `chunked_prefill_size`/`cuda_graph_config.decode.max_bs` 等旋钮，理论上比"按显存区间分 6 档"更精确，但增加了启动时间，且需要 profiling 本身的正确性保证（profiling 结果错了，自动设置的旋钮也跟着错，且更难排查，因为用户连"这个值是从哪张分档表来的"这条线索都没有了）。
  - 这两条都不能消除 476 这个数字本身，只能降低"用户需要理解全部 476 个字段"这件事的必要性——**旋钮数量的治理和旋钮易用性的治理是两个独立的问题，SGLang 目前的反射式 CLI 生成机制只解决了后者（加旋钮的工程成本低），没有触碰前者（旋钮是否应该被合并/收敛）**。

## 6. 同位对照：vLLM `EngineArgs` 在同一位置怎么做

（对应 [[09-vLLM-Python-API与EngineArgs]]；vLLM `EngineArgs` 定义在 `vllm:vllm/engine/arg_utils.py:424`，233 个字段，`_lab/out/api_surface.json` 统计其文件内有 228 处 `add_argument(`。）

- **"反射生成"的深度不同**：vLLM 同样有一层反射——`_compute_kwargs(cls)`（`vllm:vllm/engine/arg_utils.py:295`-`354`）遍历 `dataclasses.fields(cls)`，用 `get_type_hints` 推断类型、从 docstring 提取 help 文本（`get_attr_docs`）、`Optional[bool]` 转 `nargs="?"` 语义、`bool` 转 `BooleanOptionalAction`（自动生成 `--no-xxx`）,`get_kwargs(cls)`（`vllm:vllm/engine/arg_utils.py:410`）把这套推断包成一个 `{field_name: kwargs_dict}` 的字典。**但反射到此为止**——真正的 `group.add_argument(name, **kwargs[name])` 调用仍然要对每个字段手写一行（例如 `vllm:vllm/engine/arg_utils.py:849`-`895` 那一串 `model_group.add_argument(...)`）。SGLang 的 `add_cli_args_from_dataclass`（`python/sglang/srt/server_args.py:8764`）则连这最后一行调用也吞并了——**vLLM 自动生成"怎么注册"，SGLang 连"要不要注册、注册到哪个 parser"都自动化了**。这正是"233 个字段对应 228 处 `add_argument`，476 个字段只对应 29 处"这个悬殊比例的直接成因。
- **配置对象的形状不同**：vLLM 的 233 个字段并非全部直接躺在一个扁平类里——`EngineArgs` 组装自约 30 个更小的类型化子配置（`ModelConfig` 74 字段 `vllm:vllm/config/model.py:124`、`ParallelConfig` 59 字段 `vllm:vllm/config/parallel.py:119`、`CacheConfig` 33 字段 `vllm:vllm/config/cache.py:76`、`CompilationConfig` 36 字段 `vllm:vllm/config/compilation.py:398`、`SchedulerConfig` 23 字段 `vllm:vllm/config/scheduler.py:26`、`SpeculativeConfig` 35 字段 `vllm:vllm/config/speculative.py:85` 等），`get_kwargs(ModelConfig)`/`get_kwargs(ParallelConfig)`/… 分别处理（`vllm:vllm/engine/arg_utils.py:843`/`1015`/`1216`/…）。SGLang 反过来：`ServerArgs` 476 个字段全部是**同一个类**的直接成员，靠 `NS(...)` 字符串路径在运行时投影出"看起来嵌套"的配置树（`## 3.2`），而不是在类型系统层面真的嵌套。两条路线在 `## 5` 决策 2 已经对比过代价。
- **"自动推断覆盖用户设置"两边都有，但触发方式不同**：vLLM `EngineArgs.__post_init__`（`vllm:vllm/engine/arg_utils.py:769`）里，用户传了 `--fault-tolerance-config` 但没显式开 `--enable-fault-tolerance` 时，会自动把 `enable_fault_tolerance` 拨成 `True` 并打 `logger.warning`（`vllm:vllm/engine/arg_utils.py:789`-`798`）——这是"设置了细节配置就顺带打开开关"的隐式联动，与 SGLang `_disable_tc_piecewise_cudagraph_if_incompatible` 的"检测到冲突就强制关闭"方向相反（一个是隐式开启，一个是强制关闭），但都属于"字段最终值不等于用户字面传入值"这一类需要警惕的行为。
- **内存旋钮的默认值哲学不同**：vLLM `gpu_memory_utilization` 默认是写死的常量 `0.92`（`vllm:vllm/config/cache.py:111`），跨所有硬件档位统一；SGLang `mem_fraction_static` 默认 `None`，由 `_handle_gpu_memory_settings`（`python/sglang/srt/server_args.py:4860`-`5041`）按实测 GPU 显存分 6 档（`<20GB`/`<35GB`/`<60GB`/`<90GB`/`<160GB`/其余）分别给出不同的 `chunked_prefill_size`/CUDA Graph `max_bs` 组合，再据此反推 `mem_fraction_static`。这是本篇不给实测数字前提下**能确认的方向性差异**：一个用固定比例走遍所有硬件，一个用查表法按硬件档位给不同起点——哪种更准确需要实测数据支撑，本篇不下结论。

| 对比维度 | SGLang `ServerArgs` | vLLM `EngineArgs` |
|---|---|---|
| 字段总数 | 476 | 233 |
| 手写 `add_argument` 数 | 29（其中 25 个是废弃重定向） | 228（每字段一行） |
| kwargs 是否自动推断 | 是（`add_cli_args_from_dataclass`） | 是（`_compute_kwargs`/`get_kwargs`） |
| `add_argument` 调用本身是否自动化 | 是 | 否，仍需手写调用 |
| 配置对象结构 | 单一扁平 dataclass + `NS` 运行时投影 | 约 30 个嵌套子配置类型组装 |
| 命名空间/分组的实现层 | 字符串元数据 + 运行时反射 | Python 类型系统（真嵌套） |
| 内存旋钮默认值策略 | 按 GPU 显存分档查表 | 固定常量（0.92） |

## 7. 踩坑与反直觉

1. **`_lab/out/api_surface.json` 的字段 `type` 字符串被截断到 160 字符，会让基于它做的命名空间统计系统性失真。** 本篇写作过程中先用这份 JSON 做了一次正则统计，得到"196 个字段没有 `NS` 标记"的结论；核对发现 160 字符的截断恰好落在很多字段冗长的 `help=` 文本中间，而 `NS(...)` 总是被追加在 `Annotated[...]` 的**最后一位**（`## 5` 决策 2 已解释这是有意为之的迁移设计），于是长 help 文本的字段全部被误判成"无命名空间"。用 `ast.get_source_segment` 对源码逐字段取完整注解重新统计后，476 个字段全部带 `NS`，与仓库自己的覆盖率测试（`test/registered/unit/test_server_args_namespaces.py:44`-`50`）吻合。**教训**：`_lab` 的 AST 抽取工具为了控制 JSON 体积做了截断，这类"抽取工具本身有损"的情况在长文本字段（尤其是 help 文本很长的配置项）上会系统性地影响任何基于该字段做统计的下游结论，直接引用前要留意字符串是否可能被截断。
2. **不是所有标着"废弃"的字段都会在运行时警告用户。** `elastic_ep_rejoin`（`python/sglang/srt/server_args.py:2557`-`2559`）的 help 文本写着 `"[Deprecated] Alias for --elastic-ep-join-mode recover."`，但它仍然是走通用反射循环的普通 `A[bool, ...]` 字段，没有像另外 25 个废弃标志那样接上 `DeprecatedAction` 家族——也就是说用户传 `--elastic-ep-rejoin` 不会收到任何运行时警告，废弃信息只存在于 `--help` 输出里，容易被忽略。这与类文档字符串（`python/sglang/srt/server_args.py:501`-`509`）"废弃标志必须手动注册到 `add_cli_args`"的规则不完全一致，是一处规则和实践之间的漂移（源码为证：截至取证 sha，`elastic_ep_rejoin` 是唯一一个 help 文本含 `[Deprecated]` 但未接 `DeprecatedAction` 的字段，`## 2` 已给出统计口径的对照，未查证是否已有 issue 跟踪清理）。
3. **`chunked_prefill_size`/`cuda_graph_config.decode.max_bs` 的自动推断只在字段为 `None` 时介入，不是"总会按显存重算"。** `_handle_gpu_memory_settings`（`python/sglang/srt/server_args.py:4891`-`4892` 等多处）全部包在 `if self.chunked_prefill_size is None:` 里——用户一旦显式传了 `--chunked-prefill-size`，无论传的值是否适合当前硬件，这段自动分档逻辑完全不生效。这一条和 `## 0`/`## 5` 提到的"设了也白设"（`cuda_graph_config.prefill.backend`）恰好相反：前者是"用户的显式设置永远优先，自动推断只补空白"，后者是"自动检测到冲突就无条件覆盖，不管用户设没设"。同一个文件里两种相反的覆盖策略并存，读代码时不能一概而论，要具体到每个字段查它走的是哪一条路径。
4. **`schedule_policy` 默认值是 `"fcfs"`（`python/sglang/srt/server_args.py:858`-`873`），不是很多资料里暗示的 `"lpm"`（最长前缀匹配）。** RadixAttention/前缀缓存是 SGLang 的招牌特性，容易让人推断"调度策略默认也是前缀感知的"，但源码字面默认值就是 FCFS——`lpm` 是可选项，需要用户显式传 `--schedule-policy lpm` 才会启用。这条本篇特意核实过源码字面量，不是凭印象写。

## 8. 可改进点

（以下均为本库基于源码走读的推断，标注证据依据；未核实是否已有官方 issue 在跟踪。）

- **`_disable_tc_piecewise_cudagraph_if_incompatible` 的 18 条规则（`python/sglang/srt/server_args.py:4654`-`4719`）全部是"运行时探测 + 静默覆盖"，缺一层"为什么被关掉"的用户可见汇总。** 每条规则触发时只是把 `cuda_graph_config.prefill.backend` 设成 `DISABLED`，没有 `logger.warning` 说明是哪一条规则命中的（对比 `_handle_attention_backend_compatibility` 在类似场景会打印警告，见 `python/sglang/srt/server_args.py:6011`-`6013`）。用户看到"CUDA Graph 没生效"时，除非去读这 18 条规则的源码，没有办法从日志反推是哪个特性导致的。补一条 `logger.info(f"tc_piecewise CUDA Graph disabled: {_name}")` 成本很低，收益是显著减少"为什么我的配置没生效"这类排障时间（本库推断，依据是同文件内其他覆盖路径普遍带日志，这一处是例外）。
- **`elastic_ep_rejoin` 这类"文档已废弃但未接废弃 action"的字段应该有一条自动化检查。** `## 7` 第 2 条已说明现状；给 `Arg` 加一条 CI 断言（"help 文本以 `[Deprecated]` 开头的字段，`action` 字段不能是 `None`"）就能防止未来再出现这种"废弃标记只写在文档里、运行时不生效"的漂移，成本是一条几行的测试，收益是保证类文档字符串里的规则真被贯彻。
- **命名空间 lint（`test/registered/unit/test_server_args_namespaces.py:44`-`50`）目前的下界断言是 `len(nsmap) >= 440`，而实际已经是 476/476 全覆盖。** 这个宽松下界大概率是历史遗留（当年还没完全覆盖时定的门槛，覆盖完之后没有跟着收紧成 `== len(fields)`），把它改成精确断言能防止未来新增字段"忘记补 `NS`"却因为总数仍然大于 440 而蒙混过关（本库推断，依据是当前源码已经全覆盖，`>= 440` 这个阈值不再是一个有效的守门条件）。

## 9. 自测题与延伸阅读

**闭卷自测题**（合上本文，尝试不看源码回答）：

1. `ServerArgs` 有 476 个字段，`add_cli_args` 里手写的 `add_argument` 调用是 29 个。这 29 个里，动态 choices、`--config` 元参数、废弃标志重定向各占几个？
2. `Arg` 元数据里的 `resolvable=True` 是干什么用的？它和"声明-物化"（`run_post_process_pass` / `materialize_declarations`）机制是什么关系？
3. `NS(...)` 标记除了给人看的分类作用之外，在运行时还被用来做什么？定义这个投影逻辑的函数叫什么，在哪个文件？
4. `chunked_prefill_size` 的自动分档推断在什么条件下**不会**生效？举一个"用户设置被无条件覆盖，不管有没有显式设置"的反例旋钮说明二者的区别。
5. `_lab/out/api_surface.json` 里为什么会把 196 个字段误判为"没有命名空间"？这个错误的根源是什么，怎么核实出真实数字的？
6. SGLang `schedule_policy` 的默认值是什么？和"SGLang 因为 RadixAttention 默认走前缀感知调度"这个直觉是否一致？
7. vLLM `EngineArgs` 的 228 处 `add_argument` 和 SGLang 的 29 处相比，两边是不是都做了"反射生成 kwargs"？差别具体出在反射链条的哪一步？

**延伸阅读（本库内）**：

- [[08-SGLang-HTTP-API表面全解]] —— `ServerArgs` 之外，运行时 HTTP 层暴露的另一套配置面（协议类、路由）
- [[09-vLLM-Python-API与EngineArgs]] —— `## 6` 同位对照的对手篇，`EngineArgs` 的完整解剖
- [[04-SGLang-内存池与KV布局]] —— `mem_fraction_static`/`page_size`/`disable_radix_cache` 这几个旋钮在内存子系统里的具体落地，本篇只讲了它们在 `ServerArgs` 层面的解析与覆盖行为
