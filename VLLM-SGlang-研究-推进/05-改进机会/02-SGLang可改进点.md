# SGLang 可改进点

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：一个死掉的招牌优化、一处卡住整批调度的忙等、和一个被本库工具高估的密度数字。

## 0. 结论先行

- **本篇不是又一次读代码找茬，是把三样东西对齐后再下结论**：`_lab/out/improve.json` 的机器扫描信号（静默异常、`TODO`、超大文件）、[[02-SGLang-Scheduler事件循环]] 等 11 篇兄弟篇 `## 8` 已经核实过的具体缺口、以及本篇对每一条重新打开源码做的复核。三者对不上的地方，以本篇复核为准——复核过程中确实推翻了材料清单里的一条（见 `## 7` 第 3 条）。
- **最重的一条：jump-forward decoding 是全仓死代码。** 四个语法后端（xgrammar / llguidance / outlines / reasoner 包装）都完整实现了 `try_jump_forward` / `jump_forward_str_state` / `jump_and_retokenize` 三件套，但除了 `python/sglang/srt/constrained/reasoner_grammar_backend.py:228` 那一处纯委托，全仓（`python/`、`test/`）搜不到第二处调用——scheduler 从未调用它们。outlines 的实现更进一步：构造对象时把 `jump_forward_map` 硬编码成 `None`（`python/sglang/srt/constrained/outlines_backend.py:157`），是双重死代码。见 `## 8` 第 1 条。
- **语法编译异步，但"编译完没完"这一步在调度主循环里跑一个最长 5ms 的忙等**——只要 `grammar_queue` 非空，`event_loop_normal`/`event_loop_overlap` 每一轮（不只是等语法的那个请求，是**整批**）都要先过这道忙等（`python/sglang/srt/constrained/grammar_manager.py:205-225`，默认间隔 `python/sglang/srt/environ.py:377` 的 `EnvFloat(0.005)`）。见 `## 8` 第 2 条。
- **`RadixCache`（863 行）在生产工厂链里从未被直接构造**，`default_radix_cache_factory` 的每条分支最终都落到 `UnifiedRadixCache`（2,894 行）或其它专用类；`RadixCache` 唯一的构造点是给调度策略做本地模拟用的 `create_simulated()`。它的子类 `HiRadixCache` 更进一步：全仓唯一的实例化点是一个单元测试，生产代码路径完全不认识这个类。见 `## 8` 第 3 条。
- **机器信号里最扎眼的"SGLang 热路径静默 except 密度是 vLLM 的 2.6 倍"这个数字，本篇抽样后认为大部分是分类口径造成的，不是真实的风险密度差**——原因和结论见 `## 7`，这是本库诚实标准要求的自我核验，不是走过场。
- 本篇给出 **14 条可改进点**（模板：现状/问题/证据等级/改进方向/代价与反驳/难度）和 **3 条"不建议改"**，两边都要求证据，不许只列缺点不列代价。

**速查：任务要求的问题在哪一节回答**

| 问题 | 对应章节 |
|---|---|
| jump-forward 死代码的完整证据链 | `## 4`、`## 8` 第 1 条 |
| 语法编译忙等对调度延迟的影响 | `## 8` 第 2 条 |
| RadixCache/UnifiedRadixCache 的默认路径 | `## 8` 第 3 条 |
| 热路径静默 except 密度数字的抽样核验 | `## 7` |
| 不建议改的三条 | `## 8` 末尾「不建议改」 |

## 1. 它在系统里的位置

`05-改进机会/` 这一组文件是本库的收束层：`01-vLLM可改进点.md`（⬜ 未写）和本篇分别对两个引擎单独下结论，`03-跨引擎共性缺口.md`、`04-可落地贡献清单.md` 再往上一层做跨引擎归纳。本篇的输入不是重新读一遍源码，而是三条已有材料线的交汇：

1. **`_lab/out/improve.json` 的 `engines.sglang`**——AST 扫描出的静默 `except`、`raise NotImplementedError`、`TODO` 系标记、超大文件清单，是"哪里可能有问题"的**线索**，不是结论（扫描器不理解语义，只数模式）。
2. **`02-SGLang/01`~`11` 十一篇正文的 `## 8. 可改进点`**——那十一篇已经带着完整上下文（数据结构、主流程、设计权衡）读过一遍代码，各自的 `## 8` 是"读完一个子系统之后覚得哪里能改"，比机器扫描更懂语义，但视野局限在单篇讨论的子系统内。
3. **本篇的复核**——把 1、2 的候选逐条重新打开源码验证，一部分被坐实并加深（jump-forward、忙等、RadixCache），一部分被本篇发现证据不足或直接是错的（`## 7` 第 3 条）。

三条材料线互相校验：机器信号发现"哪里密度异常"，兄弟篇发现"这个模式在语义上是不是问题"，本篇的复核决定"要不要写进最终清单"。这也是为什么本篇要单独占一整篇——如果只摘抄兄弟篇的 `## 8`，读者拿到的是十一份局部视角的拼贴，看不出跨子系统重复出现的模式：`## 8` 第 1、13 条是同一种"接口/数据结构/导出函数都写完了，但接进调用链的最后一步被漏掉"的模式在语法约束这一个子系统里独立出现了两次（jump-forward 三件套、`GrammarStats` 的 Prometheus 桥接函数）——单篇视角看不出这是模式，汇总之后才看得出。这也是本篇为什么要在 `03-跨引擎共性缺口.md`（⬜ 未写）动笔之前先把 SGLang 自己的清单理清楚——跨引擎归纳需要先有单引擎的干净输入。

## 2. 代码地图

以下是本篇论证依赖的核心文件与行号，按主题分组（完整列表见 `## 8` 各条的「现状」字段）：

| 文件:行 | 是什么 |
|---|---|
| `python/sglang/srt/constrained/base_grammar_backend.py:120-140` | jump-forward 三件套接口定义（`try_jump_forward`/`jump_forward_str_state`/`jump_and_retokenize`） |
| `python/sglang/srt/constrained/reasoner_grammar_backend.py:226-239` | 全仓唯一的 `try_jump_forward` 调用点，且只是纯委托 |
| `python/sglang/srt/constrained/outlines_backend.py:157-158` | `jump_forward_map = None` 硬编码，outlines 侧的双重死代码 |
| `python/sglang/srt/constrained/grammar_manager.py:184-240` | `get_ready_grammar_requests`，主循环忙等的实现 |
| `python/sglang/srt/environ.py:377-378` | `SGLANG_GRAMMAR_POLL_INTERVAL`（默认 5ms）与最大轮询次数 |
| `python/sglang/srt/managers/scheduler.py:1755-1766` / `1799-1810` | `event_loop_normal`/`event_loop_overlap` 里逐字重复的收请求+定批片段 |
| `python/sglang/srt/managers/scheduler.py:3257-3259` | `_get_new_batch_prefill_raw` 里触发忙等的调用点，每轮调度都过一次 |
| `python/sglang/srt/mem_cache/registry.py:80-143` | `default_radix_cache_factory`，所有分支都不返回裸 `RadixCache` |
| `python/sglang/srt/mem_cache/radix_cache.py:303`、`334-350` | `class RadixCache` 定义与它唯一的构造入口 `create_simulated` |
| `python/sglang/srt/mem_cache/hiradix_cache.py:77` | `class HiRadixCache(RadixCache)`，生产工厂里查无此类 |
| `python/sglang/srt/managers/schedule_policy.py:235` | `RadixCache.create_simulated()` 的调用方——纯调度模拟用途 |
| `python/sglang/srt/server_args.py:473`、`3646`、`4654-4719` | `ServerArgs` 类定义、`_run_resolution_pipeline` 分发器、18 条静默 CUDA Graph 关闭规则 |
| `python/sglang/srt/managers/scheduler.py:1124` | 一条自己承认"dead chain"的裸 `TODO`，佐证 `## 8` 第 5 条 |
| `python/sglang/srt/speculative/spec_registry.py:37-40` | 类文档明确写"spec V1 worker path has been removed"，见 `## 7` 第 3 条 |

## 3. 核心数据结构

本篇最重要的"数据结构"不是 SGLang 自己的类，而是 `_lab/out/improve.json` 里 `engines.sglang` 的产出结构，理解它的分类口径是看懂 `## 7` 的前提：

- **`summary`**：聚合计数。`silent_except=1034`（AST 找到的、既不 `raise` 也不打日志的 `except` 块），按 `role` 拆成 `probe`（导入期/能力探测）376、`hot`（在被认定为热路径的目录下）171、`other`（其余）487；`silent_except_hot_per_10k_hot_lines=4.9` 就是 `171 / 352420 行热路径代码 × 10000`。`todo=706`，`todo_owned=167`（认领率 24%）；`not_impl=775` 是 `raise NotImplementedError` 计数；`god_file=23` 是 ≥3000 行文件数。
- **`role` 的判定是按文件路径分类，不是按调用频率分类**（本库 `_lab/improve.py` 第 52~68 行的 `_classify`/`HOT_SUBSYSTEMS` 逻辑）：一个 `except` 块只要待在 `layers/attention/`、`managers/scheduler.py` 这类被登记为"热路径子系统"的文件里，就被记成 `hot`，哪怕它其实只在模块导入时执行一次。这条口径本身在 `improve.json` 的 `caveat` 字段里有一句明确的自我提醒——本篇 `## 7` 就是去验证这条提醒在 SGLang 身上兑现到什么程度。
- **`god_files`/`hotspots_silent_except_hot_path`/`samples_silent_except_hot_path`**：前者是行数最大的文件清单（截断到前 20），后两者各给 10/25 条抽样，是 `## 7` 做人工复核的原始素材，不是结论本身。
- **"热路径子系统"这个分类本身是一份写死的集合**：本库 `_lab/improve.py` 第 44~45 行定义 `HOT_SUBSYSTEMS = {"scheduler", "kv_cache", "attention", "model_exec", "disagg", "speculative", "structured_output"}`，一个文件只要被 `_classify()` 归到这七个子系统名字之一，其中的 `except` 块就会被标成 `hot`。这份集合本身是合理的（这确实是推理服务里请求密度最高的七块），但它是静态枚举，不随子系统边界的实际变化自动更新——如果未来 SGLang 新增一个新的高频子系统（不属于这七个名字），它的静默异常会被计入 `other` 而不是 `hot`，密度统计会漏掉它。
- **`todo` 的"认领"判据**是 `f"{tag}(" in comment`——即注释里出现 `TODO(` 这个子串。这个判据是照着 SGLang 自己代码规范里"`# TODO(<gh-handle>):` 必须带认领人"的写法定的（见 `## 8` 第 5 条），但它对格式很敏感：`# TODO (lianmin): ...`（`python/sglang/srt/managers/scheduler.py:3524`，`TODO` 和 `(` 之间多一个空格）会被判成"未认领"，尽管人眼一看就知道认领人是谁。

## 4. 主流程走读：以 jump-forward 为例的取证路径

用本篇分量最重的发现走一遍取证过程，作为其它 11 条改进点共享的方法论示范：

1. **起点是材料清单**：`## 6` 兄弟篇 [[06-SGLang-约束解码与语法后端]] 的 `## 8` 第 1 条已经指出"要么删掉跳跃解码的死接口，要么把调用点接回去"，本篇的任务是重新核实这条判断在当前 sha 下依然成立，并把它落进统一模板。
2. **接口定义**：`grep -rn "def try_jump_forward" python/sglang/srt/constrained/` 命中 5 处——基类 `python/sglang/srt/constrained/base_grammar_backend.py:120`、三个具体后端（`python/sglang/srt/constrained/xgrammar_backend.py:164`、`python/sglang/srt/constrained/llguidance_backend.py:191`、`python/sglang/srt/constrained/outlines_backend.py:80`）、以及 reasoner 包装 `python/sglang/srt/constrained/reasoner_grammar_backend.py:226`。三个具体后端的实现都是真代码，不是占位符：xgrammar 和 llguidance 分别包着各自库的 `find_jump_forward_string`/`compute_ff_tokens`。
3. **调用点排查**：`grep -rn "try_jump_forward(" python/` 只命中定义本身和 `python/sglang/srt/constrained/reasoner_grammar_backend.py:228` 的一次转发调用（`return self.grammar.try_jump_forward(tokenizer)`）——这是**把调用转给内层 grammar 对象**，不是消费结果的地方。继续搜 `jump_forward_str_state(` 和 `jump_and_retokenize(`，结果一样：定义比比皆是，调用只在这一处互相转发。
4. **交叉验证"scheduler 不调用"**：在 `python/sglang/srt/managers/` 整个目录搜 `jump_forward`/`jump_and_retokenize`，零命中。调度器接受/拒绝候选 token、推进请求状态的地方（`schedule_batch.py`、`scheduler.py` 里 accept 相关逻辑）完全不知道这套接口的存在。
5. **outlines 侧的双重死代码**：进一步读 `outlines_backend.py` 发现，即便有一天调用点被接回，outlines 后端自己也接不上——`_compile_regex` 构造 `OutlinesGrammar` 时直接写死 `jump_forward_map = None`（`python/sglang/srt/constrained/outlines_backend.py:157-158`），`try_jump_forward` 方法体第一行 `if not self.jump_forward_map: return None` 永远为真。
6. **结论**：这不是"没测到"的假死代码，是**证据充分的真死代码**——接口、实现、调用链三层都核实过，且证据等级标「源码为证」。

其余 13 条改进点用同样的三步（读兄弟篇线索 → 精确定位 → 重新验证调用链/触发条件）产出，为避免正文冗长，`## 8` 只保留每条的结论与证据，不重复展开取证过程。第 13 条更进一步：不是复核兄弟篇的既有判断，而是在核实过程中发现兄弟篇的原判断本身需要修正（详见该条正文）。

## 5. 设计决策与代价：本篇方法论自身的三个选择

作为一篇"改进机会"篇，这里的"设计决策"不是 SGLang 的设计决策，是本篇怎么从材料里筛出结论的三个选择，同样按「为什么这样 / 不这样会怎样 / 什么时候可以不这样」交代：

**决策一：用文件路径而非调用频率定义"热路径"，继承 `_lab/improve.py` 的口径而不是重新做动态 profiling。**
- 为什么这样：本库没有 GPU，静态 AST 扫描是唯一能在纯 CPU 环境下低成本、可复现地跑遍 2862 个文件的办法；用文件路径近似"这段代码在请求处理链路上"，比完全不分类要好。
- 不这样会怎样：改用调用频率需要真实 profiling（`py-spy`/`torch.profiler` 之类），本库无法产出，且 `_PLAN.md` §1.4 明确禁止编造性能数字——退路只有路径近似这一条。
- 什么时候可以不这样：如果有人在真实 GPU 环境跑过 profile，能直接拿采样得到的调用计数替换掉路径分类，`## 7` 的抽样发现就可以从"我猜测"升级成"我证明"。

**决策二：本篇只吸收"证据等级=源码为证"的候选，兄弟篇标了「本库推断」但本篇没时间/没条件复核的条目一律不搬进 `## 8`。**
- 为什么这样：十一篇兄弟篇的 `## 8` 加起来接近 50 条建议，直接全搬会让本篇变成"建议的建议"，读者分不清哪条是真核实过的、哪条只是设想；本篇的独特价值就是"复核过"这三个字。
- 不这样会怎样：全部搬运确实能把条目数堆到更高（比如兄弟篇提到的"自适应机制联动 topk"、"HiRadixCache/RadixCacheCpp 等价性测试"都值得做），但会稀释本篇的可信度——一条复核过的死代码发现，价值高于十条没时间验证的设想。
- 什么时候可以不这样：如果本库以后有专门的"待复核清单"文件（目前没有），可以把这些没来得及复核的条目单独归档，而不是丢弃——这本身也是 `03-跨引擎共性缺口.md`/`04-可落地贡献清单.md` 可以承接的分工。

**决策三：抽样验证密度数字时，选择"打开源码读上下文"而不是"统计 role 分布就下结论"。**
- 为什么这样：`role` 字段本身是启发式分类，光看 `role=hot` 的计数分布回答不了"这些 except 是不是真问题"——必须回到每一条的具体代码，看它到底在保护什么、吞掉的异常会不会真的丢信息。
- 不这样会怎样：如果只报"hot 角色占比 X%"就收尾，等于把 `_lab` 工具的分类误差原样继承进正文，重复本库在 `compare.py` 上已经犯过一次的错误（`_PLAN.md` 记录的 37 个假差异）。
- 什么时候可以不这样：样本量足够大、且已经用真实流量 A/B 验证过误报率时，统计口径本身可以直接采信——本篇的抽样只有 50 条，达不到这个门槛，所以必须逐条读。

## 6. 同位对照：三引擎的机器信号横向对比

`_lab/out/improve.json` 对 vLLM、TensorRT-LLM 用的是同一套 AST 扫描器，三者可以直接摆在一张表里——这不是"哪个引擎更差"的排名，是给 `## 7` 的密度讨论提供分母：

| 指标 | SGLang | vLLM | TensorRT-LLM |
|---|---|---|---|
| 扫描的 Python 文件数 | 2862 | 2096 | 1272 |
| Python 总行数 | 1,143,185 | 827,314 | 685,891 |
| 被分类为"热路径"的行数 | 352,420 | 471,206 | 201,014 |
| 热路径占全部代码的比例 | **30.8%** | **57.0%** | **29.3%** |
| 静默 `except`（全部角色） | 1034 | 449 | 577 |
| 热路径静默 `except` 每万行 | 4.9 | 1.9 | 5.7 |
| `raise NotImplementedError` | 775 | 823 | 443 |
| ≥3000 行的超大文件数 | 23 | 7 | 25 |
| `TODO` 总数 / 已认领 | 706 / 167（24%） | 603 / 205（34%） | 418 / 49（12%） |

两条从这张表直接能读出、但材料清单没明说的信息：

- **"SGLang 热路径密度是 vLLM 的 2.6 倍"这句话默认两边的"热路径"是同一把尺子，但分母比例差了近一倍**——vLLM 有 57.0% 的代码被算进"热路径"，SGLang 只有 30.8%，TensorRT-LLM 更低（29.3%）。SGLang 和 TensorRT-LLM 的热路径占比彼此接近，密度却相差近一倍（4.9 对 5.7），说明"占比"本身不能单独解释密度差；但 vLLM 的分母几乎是 SGLang 的两倍宽，分子却只有 SGLang 的一半（88 对 171），两个方向都在把 vLLM 的密度往下拉——这条在 `## 7` 继续展开。
- **超大文件数上，TensorRT-LLM（25 个）其实比 SGLang（23 个）还多**，只是材料清单选择性地只提了"SGLang 是 vLLM 的三倍多"这一条对比；把 TensorRT-LLM 摆进来看，SGLang 的超大文件问题在同类 C++/Python 混合、多后端支持的引擎里不算孤例，更像是"支持的硬件后端和模型架构越多，越容易在配置/路由类文件上失控"这一类工程的共性，而不是 SGLang 独有的代码质量问题——这条判断本篇标「本库推断」，未跨引擎逐文件核实每个超大文件的成因。
- **`TODO` 认领率上，SGLang（24%）介于 TensorRT-LLM（12%）和 vLLM（34%）之间**，不是三者里最差的；但 `## 7` 会指出 SGLang 这个数字本身还被工具的格式敏感度拉低了一截。

超大文件的具体名单也值得摆出来对照——两边最大的文件都不是配置类文件，而是模型执行/算子层：

| 引擎 | 最大的 3 个超大文件 | 行数 |
|---|---|---|
| SGLang | `python/sglang/srt/server_args.py` | 10,142 |
| SGLang | `python/sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm100.py` | 5,611 |
| SGLang | `python/sglang/srt/managers/scheduler.py` | 5,229 |
| vLLM | `vllm:vllm/v1/worker/gpu_model_runner.py` | 7,753 |
| vLLM | `vllm:vllm/_custom_ops.py` | 4,343 |
| vLLM | `vllm:vllm/_aiter_ops.py` | 3,566 |

两边最大的文件都超过 SGLang 自己 `server_args.py` 之外第二名的两倍以上，且都集中在"模型执行主循环"或"算子分发层"——vLLM 最大的文件 `gpu_model_runner.py`（7,753 行）本身就是模型执行的核心编排文件，和 SGLang 第二大的 `flash_fwd_sm100.py`（一个 vendored CUDA kernel 文件，本篇 `## 8` 未展开讨论 vendored kernel 是否该计入"代码质量"统计，`_PLAN.md` 也没有规定这一点）性质不同。真正可比的是两边的调度器文件：SGLang `managers/scheduler.py` 5,229 行 vs vLLM `v1/core/sched/scheduler.py` 3,038 行——SGLang 的调度器确实明显更大，这条差异比笼统的"god_file 数量"对比更有信息量，也是本篇 `## 8` 第 7 条讨论 `event_loop_normal`/`event_loop_overlap` 重复代码的背景。

## 7. 踩坑与反直觉

### 7.1 「热路径静默 except 密度是 vLLM 的 2.6 倍」——抽样之后，这个数字被本库自己的工具高估了

**方法**：`improve.json` 给了两组样本可用——`samples_silent_except_hot_path`（25 条，全部 `role=hot`）和 `samples.silent_except`（25 条，`role` 以 `probe`/`other` 为主）。本篇没有逐条打开全部 1034 处（做不到，也没必要），而是从这 50 条分层样本里按代码语境归类，并对每一类精读 2-4 个真实实例，直接打开源码看上下文，而不是只看 `role` 字段的分布。

**归类结果**：

1. **模块导入期 / 首次调用期的能力探测，共性是"只执行一次，不在请求处理的热路径调用链上"，但因为文件躺在 `layers/attention/`、`kernels/ops/attention/` 这类被登记为"热路径子系统"的目录里，被 `role` 分类器算成了 `hot`。** 例子：
   - `python/sglang/kernels/ops/attention/decode_attention.py:90-99` 的 `_keep_scheduler_splits()`——用 `global _KEEP_SCHEDULER_SPLITS` 把探测结果缓存成模块级单例，`try: exec_cfg = get_exec() except ValueError: return False`，注释直接写明"not published yet, ask again on the next call"，是配置尚未发布时的正常状态，不是异常。
   - `python/sglang/kernels/ops/attention/extend_attention.py:37-42`——探测 `triton.__version__` 解析失败时退化成 `(0, 0)`，是版本号解析的标准防御写法。
   - `python/sglang/kernels/ops/attention/flash_attention_v4.py:18-19`——可选依赖导入失败时把 `_flash_attn_varlen_func` 置 `None` 并记下 `_flash_attn_import_error`，连 `# pragma: no cover` 都标了，说明作者清楚这是一条边界分支。
   - `python/sglang/kernels/ops/attention/verify_mla.py:711-713`、`792-796`——`int(...)`/`float(...)` 强转失败时给默认值，注释明确解释了为什么不能在这里做设备同步（CUDA Graph 捕获期间同步会触发 `hipErrorStreamCaptureUnsupported`）。
   这一类在 25 条 `hot` 样本里占了半数以上，全部读下来没有一条会真的吞掉"应该让用户看到"的错误信息。

2. **非阻塞 socket 轮询的标准写法**，命中 `zmq.Again`/`zmq.error.Again` 直接 `return`/`return None`，是"这次没有消息，下一轮再看"的正常控制流，不是错误处理缺陷——`python/sglang/multimodal_gen/runtime/scheduler_client.py:214-215`、`python/sglang/multimodal_gen/runtime/disaggregation/orchestrator.py:279-281`、`:331-333` 都是这个模式。如果给这类分支加日志，等于给每一次轮询空转都打一行 log，反而是噪音。

3. **真正站得住脚的一类缺口，是同一个文件里对同一类异常的处理不一致**：`python/sglang/multimodal_gen/runtime/disaggregation/orchestrator.py:378-379` 对 `self._tracker.transition(...)` 抛出的 `ValueError` 会 `logger.warning("DiffusionServer: duplicate request_id %s", request_id)`，但同一个类里 `python/sglang/multimodal_gen/runtime/disaggregation/orchestrator.py:423-424`（`_handle_decoder_result_frames` 内）对**同一个 `transition()` 方法**抛出的同一种 `ValueError` 却是裸 `except ValueError: pass`，什么都不记。两处调用的是同一个状态机方法，一处诊断信息完整、一处完全静默，这不是"探测期正常分支"，是可以直接改的诊断缺口，本篇计入 `## 8` 相关条目的佐证材料（未单独开条，因为改动收益局限于排障体验，量级够不上单列一条改进点）。

**结论**：50 条分层样本里，能明确归为"存在真实诊断缺口"的只有第 3 类里的个位数实例；其余绝大多数是探测期/轮询期的标准防御写法，被"文件路径=热路径"这条分类规则误算成了高风险密度。再结合 `## 6` 指出的"vLLM 的热路径分母比 SGLang 宽近一倍"，`4.9 对 1.9` 这个差距里，**相当一部分来自分类口径本身，不是 SGLang 真的在运行时悄悄吞掉了两倍半的错误**。这不是本库第一次发现自己的工具高估差异——`compare.py` 此前也因为口径问题产出过 37 个假差异（`_PLAN.md` 有记录）；诚实地写"数字被自己的工具高估了"，比拿着一个好看但站不住的数字去写结论更符合本库红线。

### 7.2 材料清单里"speculative/ 下 v1/v2 worker 并存"的说法，本身是错的

写作前的材料清单把"多套并存实现（`speculative/` 下 v1/v2 worker）的迁移中间态成本"列为待查线索。复核后发现这条不成立：全仓找不到任何一个不带 `_v2` 后缀的 worker 类，`python/sglang/srt/speculative/spec_registry.py:37-40` 的类文档明确写着"the spec V1 worker path has been removed, so such algorithms run on the V2 scheduler schema"。`_v2` 留在文件名（`eagle_worker_v2.py`、`standalone_worker_v2.py` 等）里纯粹是命名遗留，没有 v1 分支需要兼容——[[10-SGLang-投机解码EAGLE]] 的 `## 0` 第 2 条已经指出过这一点，本篇独立复核后确认成立，因此不把这条写进 `## 8`；`speculative/` 目录里真正值得讨论的是"独立实现 vs 继承实现"的取舍，见 `## 8` 的「不建议改」第 3 条。

### 7.3 `TODO` 认领率的测量口径本身有一个空格造成的误差

`## 3` 已经说明"认领"的判据是子串 `TODO(` 是否出现在注释里。`python/sglang/srt/managers/scheduler.py:3524` 写的是 `# TODO (lianmin): support return_logprob + mixed chunked prefill`——`TODO` 和 `(` 之间多了一个空格，工具判定的子串匹配不上，这条会被计入"未认领"的 539 条之一，尽管认领人写得清清楚楚。这不影响 24% 这个总体比例的量级判断（706 条里格式差异只会影响个位数到十位数量级），但说明 `todo_owned` 这个数字本身有轻微的低估，`## 8` 第 5 条的改进方向里会一并给出修法。

## 8. 可改进点

以下每条按「现状 / 问题 / 证据等级 / 改进方向 / 代价与反驳 / 难度」统一模板给出，全部经过本篇独立复核（方法见 `## 4`）；证据等级标「本库推断」的会在字段里说明推断依据。

按子系统分组导航（同一条可能横跨多个子系统，只列主要归属）：

| 子系统 | 涉及条目 |
|---|---|
| 语法约束解码 | 第 1、2、9、10、13 条 |
| 前缀缓存 / RadixCache | 第 3、14 条 |
| `ServerArgs` 配置解析 | 第 4 条 |
| 调度器主循环 | 第 2、7 条（第 2 条同时属于语法约束和调度器） |
| Attention backend 注册 | 第 6、8 条 |
| HiCache 分层缓存 | 第 11、12 条 |
| 工程规范 / 可观测性 | 第 5、13 条（第 13 条同时属于语法约束和可观测性） |

证据等级上，14 条里 **9 条纯「源码为证」**（第 1、2、3、4、7、8、9、10、13 条），**5 条在「现状」用源码为证、但「问题」的严重程度评估或「改进方向」的具体方案掺了「本库推断」**（第 5、6、11、12、14 条——具体是哪句推断，在各条正文里用「本库推断」显式标出，不是笼统盖章）。没有一条是纯粹靠印象、查不到源码支撑就写的——这也是本篇和"随手列一份代码异味清单"的区别：每条都要求先能回到 `file:line`，再谈值不值得改。
### 1. jump-forward decoding 是全仓死代码

- **现状**：四个语法后端（`python/sglang/srt/constrained/base_grammar_backend.py:120-140` 定义接口，`python/sglang/srt/constrained/xgrammar_backend.py:164-174`、`python/sglang/srt/constrained/llguidance_backend.py:191-201`、`python/sglang/srt/constrained/outlines_backend.py:80-108` 分别实现）都完整实现了 `try_jump_forward`/`jump_forward_str_state`/`jump_and_retokenize`。全仓（`python/`、`test/`）对这三个方法名的调用，只有 `python/sglang/srt/constrained/reasoner_grammar_backend.py:228`、`:233`、`:238` 三处纯委托转发，`python/sglang/srt/managers/` 整个目录零命中。outlines 后端更进一步：`python/sglang/srt/constrained/outlines_backend.py:157-158` 构造 `OutlinesGrammar` 时把 `jump_forward_map` 硬编码成 `None`，自身实现里的 `try_jump_forward` 第一行判断永远为真、直接返回 `None`（`python/sglang/srt/constrained/outlines_backend.py:81-82`）。
- **问题**：跳跃解码曾是 SGLang 的招牌优化之一（跳过语法唯一确定的后续 token，减少前向步数），现在完整实现挂在代码里但从不被执行——新贡献者读到 `try_jump_forward()` 里 xgrammar 那段处理字节边界、重新分词的完整正确性逻辑（`python/sglang/srt/constrained/xgrammar_backend.py:164-174` 往后的实现），容易误判这是一条生效中的优化路径，进而在其上继续叠加功能，或者复制到别处当参考实现，越描越黑。更具体的场景：一个带大量固定字面量的 JSON Schema（比如每个对象都必须有 `"type": "function_call"` 这种键名和取值都固定的字段）在语法自动机上会有很长一段"唯一确定后续 token"的路径——这正是跳跃解码本该发挥作用、一次性把这几个固定 token 都跳过去而不必逐个采样的场景，现在这条路径完全走不到，退化成和没有跳跃解码时一样逐 token 采样；受益幅度取决于真实流量里带大段固定字面量 schema 的请求占比，本库没有测过这个分布，也不产出"能省多少步"这类性能数字（`_PLAN.md` §1.4）。
- **证据等级**：源码为证（`## 4` 给出完整取证路径）。
- **改进方向**：两条路都好过现状——(a) 接回调度器：在 `schedule_batch.py` 的 token 接受逻辑之后，对开启语法约束且 `grammar.try_jump_forward()` 返回非空的请求做一次 token 序列拼接和重新定位，这需要先解决变长 token 数与 CUDA Graph/`future_map` 定长假设的冲突（见 `## 8` 不建议改之外的设计讨论，本条不展开）；(b) 直接删除死代码：去掉三个方法在四个后端里的实现和基类接口，只保留将来想恢复时能在 git 历史里找回的记录。
- **代价与反驳**：接回调度器工作量不小，且收益依赖"语法唯一确定后续 token"在真实流量里出现的频率——如果这个场景本来就稀少，接回的复杂度可能不划算，这也可能是 SGLang 团队至今没做的真实原因（本库推断，未查证具体决策记录）。直接删除的风险是不知道有没有下游 fork 或外部项目在依赖这几个公开方法名做自己的实现（未查证）。
- **难度**：需要设计讨论（接回）；小改（删除，但需先确认无外部依赖）。

### 2. 语法编译"是否就绪"检查是卡住整批调度的忙等

- **现状**：语法编译本身跑在 `ThreadPoolExecutor`（`python/sglang/srt/constrained/base_grammar_backend.py:206`）里，不阻塞主循环；但检查编译状态的 `get_ready_grammar_requests()`（`python/sglang/srt/constrained/grammar_manager.py:184-240`）在 PP0 上会跑一个 `while time.perf_counter() - start_time < SGLANG_GRAMMAR_POLL_INTERVAL` 的循环（`python/sglang/srt/constrained/grammar_manager.py:205-206`），每次没等到就 `time.sleep(SGLANG_GRAMMAR_POLL_INTERVAL / 10)`（`python/sglang/srt/constrained/grammar_manager.py:225`）。默认 `SGLANG_GRAMMAR_POLL_INTERVAL=0.005`（`python/sglang/srt/environ.py:377`），即最长等 5ms、每 0.5ms 检查一次。这个函数由 `_get_new_batch_prefill_raw` 在 `self.grammar_manager.has_waiting_grammars()` 为真时调用（`python/sglang/srt/managers/scheduler.py:3257-3259`），而 `_get_new_batch_prefill_raw` 又是 `get_next_batch_to_run`（`python/sglang/srt/managers/scheduler.py:3080`）的一部分，后者在 `event_loop_normal`（`python/sglang/srt/managers/scheduler.py:1761`）和 `event_loop_overlap`（`python/sglang/srt/managers/scheduler.py:1805`）里**每一轮调度**都会被调用——不只是等语法的那个请求所在的轮次。
- **问题**：只要 `grammar_queue` 非空（有任意请求的 schema 还在编译），调度器接下来的每一轮 `get_next_batch_to_run` 都要先过这道最长 5ms 的忙等，才能继续给**所有**运行中的请求（包括跟语法编译毫无关系、纯解码的请求）组装本轮要跑的 batch。对小模型、小 batch、单步解码延迟本来就在个位数毫秒量级的部署，这道忙等可能和真实前向延迟一个数量级，等于把"一个请求的语法还没编译完"这件事，变成"全体请求这一轮都要多等最多 5ms"。
- **证据等级**：源码为证。
- **改进方向**：把内层 `while ... sleep` 循环改成只在 `running_batch` 本轮无事可做（没有可推进的运行中请求、也没有其它可调度的等待请求）时才允许等待到 `SGLANG_GRAMMAR_POLL_INTERVAL`；否则只做一次不等待的即时检查（`.done()` 查一遍就返回），把"还没编译完的请求"留到下一轮自然的调度节拍里再捡——反正 `get_next_batch_to_run` 本来就是每个前向步都会被调用一次，不需要在函数内部再自己造一个等待窗口。
- **代价与反驳**：现在的忙等本质是在拿"让新就绪的请求赶上这一轮"去换"其它请求多等一点"，如果编译经常在几毫秒内完成，这个设计能让语法请求少排一轮队；改成即时返回会让编译完成的请求systematically多等一个调度周期（通常 < 一次前向步耗时），对大 batch、长前向步耗时的部署几乎无感——所以这条问题的严重性和"batch 小、单步延迟低"这个具体场景强相关，不是所有部署都会踩到，这也可能是它至今没被当作优先级很高的 bug 修的原因。
- **难度**：小改（改一个条件判断，不涉及数据结构变更）。

### 3. `RadixCache` 已不是默认生产路径，`HiRadixCache` 是彻底的死代码

- **现状**：`default_radix_cache_factory`（`python/sglang/srt/mem_cache/registry.py:80-143`）的每条分支要么返回 `ChunkCache`/`PureSWARadixCache`/`RadixCacheCpp`/`LMCRadixCache`/flexkv 实现，要么落到 `_create_unified_radix_cache`（`python/sglang/srt/mem_cache/registry.py:143`、`146-196`）构造 `UnifiedRadixCache`（`python/sglang/srt/mem_cache/registry.py:187`）；`class RadixCache`（`python/sglang/srt/mem_cache/radix_cache.py:303`，863 行）在这条工厂链里完全没被提及。它唯一的构造入口是 `create_simulated()`（`python/sglang/srt/mem_cache/radix_cache.py:334-350`），只服务 `python/sglang/srt/managers/schedule_policy.py:235` 的等待队列前缀匹配模拟和它自己文件内的 `__main__` 测试块（`python/sglang/srt/mem_cache/radix_cache.py:850`）。它的子类 `HiRadixCache`（`python/sglang/srt/mem_cache/hiradix_cache.py:77`）情况更极端：全仓搜索 `HiRadixCache(` 只有一处命中——`test/registered/unit/mem_cache/test_hiradix_cache_unit.py:76`，生产工厂 `registry.py` 从未提到这个类名；HiCache 分层缓存在生产路径上是靠 `UnifiedRadixCache.init_hicache()`（`python/sglang/srt/mem_cache/registry.py:192`）直接实现的，根本不经过 `HiRadixCache`。
- **问题**：`RadixCache` 是 [[03-SGLang-RadixAttention与前缀缓存]] 一篇的主要走读对象，代码量小、逻辑干净，是理解"匹配-分裂-驱逐"算法的最佳教材；但读者如果误以为它是生产实现（类名和目录位置都没有任何标注说明它已经退居模拟器角色），会对"SGLang 的前缀缓存怎么工作"形成一个和真实运行路径不符的心智模型。`HiRadixCache` 则是更纯粹的风险：它存在、被继承、有自己的方法覆写，但没有任何生产代码路径能构造出它的实例——针对它写的任何 bug 修复都不会在真实部署里生效，而这个事实从类定义本身完全看不出来。
- **证据等级**：源码为证。
- **改进方向**：给 `radix_cache.py` 顶部加一句模块级说明："生产环境的前缀缓存实现见 `unified_radix_cache.py`；本文件的 `RadixCache` 目前只用于调度策略的本地模拟（见 `create_simulated`）和作为算法参照。" 对 `HiRadixCache`：要么把它接回 `registry.py` 的一条可达路径（如果分层缓存本该走独立子类而不是 `UnifiedRadixCache.init_hicache`，这需要先搞清楚两条路径的历史关系，本篇未查证），要么删除这个类和它专属的单元测试，避免死代码继续误导。
- **代价与反驳**：完全删除 `RadixCache` 本身会有损失——它是比 `UnifiedRadixCache` 短得多的参照实现，[[03-SGLang-RadixAttention与前缀缓存]] `## 8` 第 4 条已经指出过"两个实现的匹配/分裂/驱逐语义是否一致"目前只能靠人工读代码确认，删掉更短的那个参照版本会让这类核对失去比对基准；真正该动的只是 `HiRadixCache` 这个死掉的子类，且删除前需要先排查有没有下游/fork 直接 `import` 它（未查证）。加模块说明的成本接近零，唯一的"代价"是需要有人愿意去核实清楚 `HiRadixCache` 当年是不是被 `UnifiedRadixCache` 整体取代了，还是只是暂时被绕过——这条历史本篇没有找到文档记录。
- **难度**：小改（模块说明 + 删除 `HiRadixCache`）；若要恢复 `HiRadixCache` 的可达路径则需要设计讨论。

### 4. ServerArgs 的静默 CUDA Graph 关闭规则缺用户可见日志

- **现状**：`_disable_tc_piecewise_cudagraph_if_incompatible`（`python/sglang/srt/server_args.py:4649-4719`）用一张 18 条 `(名字, 判据函数)` 的 `rules` 列表（`python/sglang/srt/server_args.py:4654` 起）遍历检查，任意一条命中就把 `self.cuda_graph_config.prefill.backend` 静默设成 `Backend.DISABLED`（`python/sglang/srt/server_args.py:4719` 前后），**不打印任何日志**说明是哪条规则命中的。对照同一个文件里 `_handle_attention_backend_compatibility` 处理 `torch_native` 后端时的写法（`python/sglang/srt/server_args.py:6009-6013`）：同样是关掉 CUDA Graph，这里会先 `logger.warning("Cuda graph is disabled because of using torch native attention backend")` 再改配置。
- **问题**：用户看到 CUDA Graph 没生效时，如果命中的是 18 条规则里的任意一条（比如 `python/sglang/srt/server_args.py:4658-4661` 的"model-arch blacklist"或 DP attention 之类），日志里没有任何线索；如果命中的是 `torch_native` 分支，日志会直接告诉你原因。同一份代码里两种处理方式并存，排障体验完全取决于命中的是哪条规则，纯属运气。
- **证据等级**：源码为证。
- **改进方向**：在 `for _name, predicate in rules: if predicate(): ...` 的循环体里加一行 `logger.info(f"tc_piecewise CUDA Graph disabled: {_name}")`，把现有的 `_name` 字段（已经存在，只是没被用来打日志）利用起来即可，不需要新增数据结构。
- **代价与反驳**：18 条规则里有的判据本身开销很小（读一个已缓存的布尔属性），加日志的运行时成本可以忽略；唯一需要注意的是别让这行日志在每次请求都触发——但这段代码只在 `ServerArgs` 构造时跑一次，不在请求路径上，不存在日志刷屏的风险。真正的成本只是"决定日志文案措辞"这种琐碎的评审开销。
- **难度**：小改。

### 5. `TODO` 认领率 24%，且工具的认领判据比项目自己的规范更严格

- **现状**：`summary.todo=706`，`summary.todo_owned=167`（认领率 24%，`## 6` 表里横向看低于 vLLM 的 34%，高于 TensorRT-LLM 的 12%）。判据是注释里出现子串 `TODO(`。仓库里能同时找到三种状态的真实例子：`python/sglang/srt/managers/scheduler.py:1124` 是完全裸的 `# TODO: max_running_requests_under_SLO has no setter — dead chain.`（连认领人都没有，还自己承认是条死链）；`python/sglang/srt/managers/scheduler.py:4002` 是规范写法 `# TODO(lsyin): make the delayed sample a default behavior after...`；`python/sglang/srt/managers/scheduler.py:3524` 是 `# TODO (lianmin): support return_logprob + mixed chunked prefill`——认领人写了，但 `TODO` 和 `(` 之间多了一个空格，被工具判成"未认领"（`## 7.3` 已展开）。
- **问题**：24% 的认领率本身已经不高，且是被工具用一个比人类阅读更严格的字符串匹配算出来的偏低估计——真实的"完全裸 `TODO`，连认领人都没写"的比例应该比 24% 对应的"未认领 76%"更低一些（低多少未查证，因为无法在不重新跑一遍带模糊匹配的扫描的情况下知道有多少条属于"`TODO` (人名)"这种带空格的变体）。不管修不修测量口径，"没有认领人的 `TODO` 长期没人认领"这个真实问题依然存在——`python/sglang/srt/managers/scheduler.py:1124` 那条本身承认自己是死链，这类 `TODO` 更接近该被处理或者转成 issue，而不是继续留在注释里。
- **证据等级**：源码为证（三个例子）+ 本库推断（"真实认领率应该比 24% 略高"这一句是推断，没有重新跑宽松匹配的扫描去验证具体数字）。
- **改进方向**：两件事分开做——(a) 给本库 `_lab/improve.py` 第 75~89 行的 `_todo_comments` 把 `owned` 判据从 `f"{tag}(" in comment` 放宽成允许 `TODO` 和 `(` 之间有空格的正则，让统计数字更准；(b) 更实质的是让 `sglang:.claude/rules/comment-style.md:136` 里已经写明的"`# TODO(<gh-handle>):` 必须带认领人"这条规范有一个自动化的 CI 检查（grep 找出裸 `# TODO:` 且后面不带括号的行），而不是只停留在文档里靠 review 人工把关。
- **代价与反驳**：现有 706 条 `TODO` 里预计有相当比例（未精确统计）不满足新规则，一次性引入 CI 检查会导致大量存量违规——需要给存量开一个豁免清单（legacy grandfather）而不是一次性全部拦下来，这条 CI 规则本身的价值也有上限："有认领人"不保证"真的会被处理"，认领人几个月后转岗或离职是常态，这只是一道社会性弱约束，不是自动化能保证结果的机制。
- **难度**：小改（放宽正则）+ 中等（CI 检查 + 存量豁免清单）。

### 6. 5 张互相独立维护的 attention backend 白名单缺一致性检查

- **现状**：`server_args.py` 里独立维护着 `ATTENTION_BACKEND_CHOICES`（23 项，`python/sglang/srt/server_args.py:181-207`）、`DRAFT_ATTENTION_BACKEND_CHOICES`（6 项，`python/sglang/srt/server_args.py:213-219`）、`CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS`（8 项，`python/sglang/srt/server_args.py:226-235`）、`DETERMINISTIC_ATTENTION_BACKEND_CHOICES`（5 项，`python/sglang/srt/server_args.py:237-242`）四张字符串列表，加上 `python/sglang/srt/layers/attention/attention_registry.py:31` 的 `ATTENTION_BACKENDS = {}` 模块字典（靠 `register_attention_backend` 装饰器在各后端文件里分别注册，`python/sglang/srt/layers/attention/attention_registry.py:33-38`）——五处登记表，靠人工在新增/修改后端时同步保持一致。
- **问题**：新增一个 attention backend 时，只要漏掉其中一张表（比如忘了把它加进 `CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS`），不会有任何启动期报错，只会在真正用到分块前缀缓存 + 这个新后端的组合时才暴露——[[05-SGLang-注意力后端矩阵]] `## 7` 已经记录过一次真实的类似缺口（`cutedsl_mla` 未被某个白名单覆盖）。
- **证据等级**：源码为证（五张表的行号）+ 本库推断（"新后端接入时容易漏掉某张表"是根据现有代码结构做的合理推断，未查证具体历史事故）。
- **改进方向**：给 `register_attention_backend` 装饰器加可选的布尔关键字参数（比如 `supports_chunked_prefix=False`、`supports_draft=False`、`deterministic=False`），把四张字符串列表改成从 `ATTENTION_BACKENDS` 这张单一登记表按标记过滤出来的视图函数，新增后端时只需要在注册处声明一次能力，不需要记住"这个新特性还要同步改哪几张表"。
- **代价与反驳**：这是一次跨越 `server_args.py` 和所有具体后端注册点的改动，需要给已有的 20+ 个后端逐一补上能力标记（现有信息就散落在四张表里，迁移本身工作量不小但机械、可批量做）；迁移期间如果标记漏打，效果和现状一样差，所以这类重构需要配一次性的"从旧四张表反推标记，自动生成新注册代码"的脚本，而不是纯手工搬。
- **难度**：中等。

### 7. `event_loop_normal` 与 `event_loop_overlap` 的公共前半段是两份逐字重复的代码

- **现状**：`event_loop_normal`（`python/sglang/srt/managers/scheduler.py:1747-1781`）和 `event_loop_overlap`（`python/sglang/srt/managers/scheduler.py:1782-` 往后）开头的"收请求 → `process_input_requests` → 暂停检查 → `get_next_batch_to_run`"这几行逐字重复：`python/sglang/srt/managers/scheduler.py:1755-1766` 和 `python/sglang/srt/managers/scheduler.py:1799-1810` 是同一段逻辑的两份拷贝（`recv_requests()`/`process_input_requests()`/`_engine_paused` 检查/`get_next_batch_to_run()` 调用/结果拆包全部一致）。
- **问题**：任何一处要改（比如给"收请求"阶段加一个新的前置检查），都需要记得同步改两处；这类重复是最容易在后续维护里"改了一处忘了另一处"的地方，且没有测试能捕获这种遗漏（两个循环各自有集成测试覆盖端到端行为，但没有专门断言"这段前置逻辑在两个循环里完全一致"的测试）。
- **证据等级**：源码为证。
- **改进方向**：抽成一个私有辅助方法（例如 `_recv_and_build_next_batch(self)`，返回 `NextBatchPlan`），两个 `event_loop_*` 各自调用一次，把重复的 10 余行收敛成一处。
- **代价与反驳**：这是一处几乎零风险的重复代码消除——两段代码逻辑完全一致，抽取本身不改变任何行为；唯一需要注意的是 `event_loop_overlap` 在这段代码之后紧跟着 `disable_overlap_for_batch` 这类只属于 overlap 模式的后续处理（`python/sglang/srt/managers/scheduler.py:1811` 起），抽取时要确保共享方法的返回值能干净地喂给两条分叉后的不同后续逻辑，不要为了复用把 overlap 特有的分支也拖进共享方法里。
- **难度**：小改。

### 8. 草稿模型 attention backend 解析逻辑分散在两个文件的两个函数里

- **现状**：`resolve_draft_attention_backend()`（`python/sglang/srt/model_executor/model_runner.py:267-282`）决定 `ModelRunner.draft_attention_backend` 属性的初始值，逻辑是"草稿 worker 用调用方传入的 backend，否则用 `--speculative-draft-attention-backend`"；`_resolve_draft_attention_backend_fallback()`（`python/sglang/srt/speculative/draft_worker_common.py:31-49`）在真正构建草稿 worker 时做二次校验，如果解析出的 backend 不在 `DRAFT_ATTENTION_BACKEND_CHOICES` 白名单里，会静默 fallback 成 `triton`/`flashinfer` 并打一条 warning。
- **问题**：想搞清楚"草稿模型最终用的 attention backend 是怎么定下来的"，需要跨两个文件、两个函数才能拼出完整链路——前者决定候选值，后者决定候选值是否被接受、不被接受时怎么兜底。这个链路目前没有单一入口，容易在后续新增草稿算法时漏掉其中一环的校验。
- **证据等级**：源码为证。
- **改进方向**：让 `resolve_draft_attention_backend()` 直接调用 `_resolve_draft_attention_backend_fallback()` 做校验，或者反过来把两者合并成一个函数，明确"决定候选值"和"校验/兜底"是同一条调用链上的两步，而不是两条平行存在的独立路径。
- **代价与反驳**：两个函数目前分别属于"frozen 文件"（`model_runner.py` 按 `.claude/skills/large-class-style` 的约定是 orchestration-only 的冻结文件，域逻辑不该写在里面）和普通模块（`draft_worker_common.py`）——合并需要考虑这条冻结约定：校验逻辑本来就不该搬进 `model_runner.py`，所以更合理的方向是让 `resolve_draft_attention_backend()` 保持轻量（只做"选哪个候选值"的编排），显式调用 `draft_worker_common.py` 里的校验函数，而不是把两者的代码本体合并到一处。
- **难度**：小改。

### 9. 语法编译缓存 key 不做归一化，语义等价的 schema 会被重复编译

- **现状**：`BaseGrammarBackend.cache`（`python/sglang/srt/constrained/base_grammar_backend.py:207`）的类型是 `Dict[Tuple[str, str], BaseGrammarObject]`，`get_cached_or_future_value()`（`python/sglang/srt/constrained/base_grammar_backend.py:284-296`）直接用 `(key_type, key_string)` 这个原始字符串元组做字典 key（`python/sglang/srt/constrained/base_grammar_backend.py:287`、`296`），没有任何归一化步骤。
- **问题**：两个字段顺序不同但语义完全等价的 JSON Schema（比如 `{"a": ..., "b": ...}` 和 `{"b": ..., "a": ...}`）会被当成两个不同的缓存条目，各自触发一次完整的语法编译——编译本身跑在线程池里不阻塞主循环（`## 8` 第 2 条已经说明检查完成状态才是真正的成本），但重复编译本身仍然浪费 CPU、且让缓存命中率低于理论值。
- **证据等级**：源码为证。
- **改进方向**：在 `key = ("json", req.sampling_params.json_schema)` 这类构造缓存 key 的地方（`python/sglang/srt/constrained/grammar_manager.py:145`），对 JSON 类型的 schema 先做一次 `json.dumps(json.loads(schema), sort_keys=True)` 归一化再作为 key 的一部分,理论上其它可解析成结构化对象的类型（比如部分 `structural_tag`）也可以做类似处理。
- **代价与反驳**：归一化本身有 CPU 开销（一次 `json.loads`+`json.dumps`），需要确认这个开销显著小于一次完整语法编译才划算——大 schema 场景下应该是净赚，但对本来就很小的 schema，归一化的相对开销占比可能不小，需要用真实 schema 分布验证收益（本库不产出性能数字，这里只给方向）。
- **难度**：小改。

### 10. llguidance 后端用 `assert` 做输入校验，`-O` 模式下理论上会被跳过

- **现状**：`python/sglang/srt/constrained/llguidance_backend.py:286` 的 `dispatch_structural_tag` 里，`assert is_legacy_structural_tag(structural_tag)` 是校验用户传入的 `structural_tag` 是否为旧版格式的唯一关卡；Python 解释器以 `-O`/`-OO` 优化模式启动时，`assert` 语句会被整体跳过。
- **问题**：如果部署环境碰巧以优化模式启动 Python（罕见但不是不可能），这条校验会静默失效，新版 `structural_tag` 写法会被当成合法输入继续往下走，最终报错信息会来自更深层、更难定位的地方，而不是这条本该在入口就给出的清晰提示。
- **证据等级**：源码为证（是否真的有部署会用 `-O` 启动 SGLang 未查证，风险的现实发生概率本库无法评估）。
- **改进方向**：把 `assert is_legacy_structural_tag(structural_tag)` 换成 `if not is_legacy_structural_tag(structural_tag): raise ValueError("llguidance 后端不支持新版 structural_tag 写法，请改用 xgrammar 或转换成 legacy 格式")`——同时解决"不受 `-O` 影响"和"错误信息从裸 `AssertionError` 变成可操作提示"两个问题。
- **代价与反驳**：零运行时代价（`if`+`raise` 和 `assert` 性能几乎无差异），是这份清单里代价最低的一条；不改的理由只能是"反正没人拿 `-O` 启动 SGLang"，但这是一个无法用静态分析验证、只能靠约定成俗的假设。
- **难度**：小改。

### 11. HiCache 的 `write_through_threshold` 硬编码成 1/2，不看前缀长度

- **现状**：`python/sglang/srt/mem_cache/hiradix_cache.py:206-209`：`self.write_through_threshold = (1 if server_args.hicache_write_policy == "write_through" else 2)`，上面紧跟着一行源码自己留下的 `# todo: dynamically adjust the threshold`（`python/sglang/srt/mem_cache/hiradix_cache.py:206`）。这个阈值是"一段前缀被命中几次才值得往 host 层写一份备份"的判据，目前不管前缀多长、多短，命中次数门槛都是同一个数字。
- **问题**：一段很短的前缀即使命中一次就写入 host 层，重算这段前缀的成本可能比一次 PCIe 往返还低，提前写入纯粹是浪费带宽；反过来一段很长的前缀本该更积极地被写入。命中次数这个单一维度没有用到"这段前缀有多少 token"这个已经在代码别处存在的量。
- **证据等级**：源码为证（现状）+ 本库推断（"短前缀重算比 PCIe 往返更便宜"是基于常识的推断，本库没有做过实测对比，也没有 GPU 环境可以测）。
- **改进方向**：把 admission 判据从单纯的命中次数，改成命中次数和前缀 token 数的联合判断——比如"命中次数 × 前缀长度"超过某个阈值才触发 write-through，短前缀需要更多次命中才达标。
- **代价与反驳**：这个改动引入了一个新的复合公式和至少一个新的可调参数,调不好可能比现在的固定阈值更难预测；源码自己留的 `TODO` 只说"动态调整"，没有说明具体往哪个方向调，本条给出的"结合前缀长度"是本库推断的一种可能方向，不是源码作者原本设想的方案（未查证作者的真实意图）。
- **难度**：中等（需要先弄清楚 `cell_size`/带宽这类量在当前代码里以什么形式存在，再决定怎么接入判据）。

### 12. GPU 驱逐与 Host 驱逐各自独立触发，长期 host 紧张时会抖动

- **现状**：`python/sglang/srt/mem_cache/hiradix_cache.py:855-858` 的 `write_backup` 分配 host 内存失败时，现场调用 `self.evict_host(len(node.value))` 再重试一次写入；这条"失败 → 驱逐 → 重试"路径没有对"最近是否刚驱逐过"做任何节流或冷却。
- **问题**：在 host 内存长期紧张的场景下（比如 hierarchical cache 开得比较激进、host 层容量配置偏小），这条路径可能反复触发"分配失败 → 驱逐一批 → 立刻又被下一次写入填满 → 再次分配失败 → 再驱逐"的抖动循环，驱逐本身的开销（要遍历驱逐候选、可能涉及跨层同步）在高频抖动下会变成额外的固定开销。
- **证据等级**：源码为证（现状）+ 本库推断（"长期紧张会抖动"是根据代码结构做的合理推断，本库没有对应的压测场景可以复现实测数据）。
- **改进方向**：加一个简单的冷却窗口——刚触发过 `evict_host` 之后的一小段时间内，同一层级的新写入请求直接走更保守的降级路径（比如暂时跳过 write-through、只保留设备侧缓存），而不是立刻再触发一次驱逐。
- **代价与反驳**：冷却窗口本身会引入新的旋钮参数（冷却时长）和新的边界情况（冷却期内到达的、本该被写入的请求会被延后处理，如果冷却时长设得不合适可能反而降低命中率）；这属于"需要用真实负载先验证抖动确实存在、量级有多大"之后再决定值不值得做的长期方向，不是可以直接提 PR 的改动。
- **难度**：需要设计讨论。

### 13. 语法编译指标的 Prometheus 导出桥接函数本身是死代码——比材料清单原本描述的问题更深一层

- **现状**：`GrammarStats`（`python/sglang/srt/constrained/base_grammar_backend.py:39-47`）在编译期间被逐字段填充（`compilation_time`/`schema_count`/`ebnf_size`/`tree_traversal_time`/`is_cache_hit`/`is_grammar_aborted`/`num_timeout`，例如 `python/sglang/srt/constrained/base_grammar_backend.py:281` 写入 `compilation_time`）；`python/sglang/srt/observability/metrics_collector.py:744-868` 也确实注册了对应的 Prometheus `Histogram`（`sglang:grammar_compilation_time_seconds`、`sglang:grammar_tree_traversal_time_avg` 等），并且 `python/sglang/srt/observability/metrics_collector.py:1420-1441` 写了一个 `log_grammar_stats(self, grammar_stats)` 方法,把 `GrammarStats` 的字段一一喂给这些 histogram。但全仓（`python/`、`test/`）搜索 `log_grammar_stats`，唯一命中的就是这个方法自己的定义——没有任何调用点。
- **问题**：这比"没有导出路径"更容易误导人——指标本身已经在 `/metrics` 端点注册（会出现在 Prometheus 抓取结果里），运维看到指标名存在,会以为它在正常汇报数据,实际上这些 histogram 永远不会被写入任何数据点，因为唯一能把 `GrammarStats` 灌进它们的 `log_grammar_stats` 从未被调用。这是本篇继 jump-forward 之后第二次在约束解码这个子系统里发现"接口/基础设施建好了,粘合调用那一步却漏掉了"的模式，值得作为一类通用风险记下来：**新功能落地时,数据结构、消费端接口都写完之后,容易漏掉"谁在什么时候调用消费端"这最后一环，且这一环缺失不会有任何报错提示**。
- **证据等级**：源码为证。
- **改进方向**：在 `GrammarManager`（或 `Scheduler` 里语法请求编译完成、状态转移到 ready 的那个位置，`get_ready_grammar_requests` 之后）加一次 `metrics_collector.log_grammar_stats(req.grammar.grammar_stats)` 调用，把已经写好的数据结构和已经写好的导出函数用一行代码接上。
- **代价与反驳**：这条改动本身代价很低（一行调用），唯一需要确认的是调用时机——`GrammarStats` 的字段是在编译期间逐步填充的（比如 `tree_traversal_time` 可能在多次 `accept_token` 过程中持续追加），需要先搞清楚"编译完成的那一刻"是不是所有字段都已经是最终值，还是有些字段（比如运行期持续更新的树遍历耗时）本来就该在请求结束时才汇报一次——如果是后者，接入点应该放在请求生命周期结束的地方而不是编译完成的地方，这需要先读一遍 `tree_traversal_time` 具体在哪些时间点被追加（本篇未展开）。
- **难度**：小改（确认接入时机后，加一行调用）。

### 14. `RadixCacheCpp` 与 Python 版 `RadixCache`/`UnifiedRadixCache` 缺功能对等性文档

- **现状**：`RadixCacheCpp`（`python/sglang/srt/mem_cache/radix_cache_cpp.py:35`）直接继承 `BasePrefixCache`，不是任何 Python radix 实现的子类，是一条完全独立的 C++/Python 混合实现；启用开关 `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE`（`python/sglang/srt/environ.py:591`，默认 `False`）关闭时走不到这条路径（`python/sglang/srt/mem_cache/registry.py:104-109`）。它已知至少有一处主动拒绝的功能差异：`_reject_cache_salt`（`python/sglang/srt/mem_cache/radix_cache_cpp.py:36-40`）会在传入 `cache_salt` 时直接 `raise ValueError`，明确说明"实验性 C++ radix tree 不支持这个参数"。
- **问题**：这是一条 opt-in 但已经写进代码、随时可能被用户打开的路径，它和默认路径（`UnifiedRadixCache`）之间"哪些功能受支持、哪些参数组合会被拒绝"目前只能靠读两份完全独立的实现逐行对比才能确认——`_reject_cache_salt` 这种"显式拒绝"还算友好（至少会报错），但没有文档保证所有的功能差异都用这种显式拒绝的方式暴露出来，用户打开这个实验开关之前无法一眼看出自己会失去什么。
- **证据等级**：源码为证（`RadixCacheCpp` 的独立继承关系和 `_reject_cache_salt` 这一具体差异）+ 本库推断（"没有文档保证所有差异都显式拒绝"是基于本篇未逐行比对两个实现的全部方法得出的合理担忧，不是穷举验证过的结论）。
- **改进方向**：在 `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE` 的帮助文本或专门的一份对照文档里，列出已知的功能差异清单（目前至少已知 `cache_salt` 不支持），并且理想情况下用一组共享的行为测试（相同的插入/匹配/驱逐序列，同时跑两个实现，断言可比较的输出一致）来自动发现新增差异，而不是依赖开发者记得在改一个实现时想到去核对另一个。
- **代价与反驳**：两个实现分属 Python 和 C++（通过 `cpp_radix_tree` 绑定），写共享的行为等价性测试本身工作量不小,需要先定义清楚"什么叫等价"（比如驱逐顺序在实现细节不同的情况下是否允许不同,只要求最终缓存内容集合一致）；纯文档层面的差异清单成本低得多，可以先做,等价性测试作为更完整但更贵的后续步骤。
- **难度**：小改（差异清单文档）；中等（共享行为等价性测试）。

### 不建议改的三条

**A. `except ImportError: pass` 这类可选依赖探测，不该被无差别地"补上日志"或改成显式抛错。**
`python/sglang/kernels/ops/attention/flash_attention_v4.py:18-19`、`check_env.py` 系列（`## 7.1` 已经举了具体例子）大量使用"尝试导入可选后端，失败就把标志位设成不可用"的模式。这类代码的语义就是"这个能力在当前环境下可能存在也可能不存在，两种情况都是合法状态"——给每一处都加日志会在没装可选依赖（比如没装 AMD ROCm 相关包、没装某个厂商专用 kernel 库）的正常环境里刷出大量无意义的警告，把真正的问题淹没在噪音里。这类 `except` 本来就该"安静地失败"，改动它是在制造问题而不是解决问题。

**B. `server_args.py` 巨大（10,142 行），但作为唯一权威来源集中一处，未必比拆分更差——这条要两面说。**
支持集中的一面：`_run_resolution_pipeline`（`python/sglang/srt/server_args.py:3646-3667`）不是一坨杂乱代码,它有一份写在 docstring 里的五条排序原则（按依赖域分组、把厂商/特性细节隐藏在通用命名的 handler 名字后面等），是一个**有纪律的分发器模式**，不是"越写越乱"的产物；把 476 个字段集中在一个文件里,任何一次"这个旋钮的默认值是什么"的排查都只需要 `grep` 一个文件，不需要在几十个小文件之间跳转,也不需要担心跨文件的循环 import。反对集中的一面：这个仓库自己的工程规范文档（`sglang:.claude/rules/general-code-style.md:14`）明确写着"Files stay small. Keep each file under ~2k LOC"——10,142 行是这条自定规则上限的五倍多；即便有分发器纪律,单文件超过一万行本身还是会拖慢 IDE 索引、拉长 diff、增加合并冲突概率。**本条不建议做"大拆分"这种大动作**：`ServerArgs` 本质上是一份不可再分割的全局配置命名空间（几乎每个字段都可能被 `_run_resolution_pipeline` 里任意一个 handler 读到）,强行拆成多个文件大概率只是把"一处大文件"换成"十处互相 import 的小文件 + 一处更复杂的组装逻辑",复杂度未必真的下降；真正值得做的是本篇 `## 8` 第 4、6 条这种局部改进（补日志、收敛白名单）,而不是推倒重来。

**C. 四个独立实现的投机解码 worker（`MultiLayerEagleWorkerV2`/`DFlashWorkerV2`/`DSparkWorkerV2`/`NGRAMWorker`）不该被强行合并进共享基类。**
`## 7.2` 已经纠正过"这是 v1/v2 并存"的错误说法——真实情况是 SGLang 已经完成了 worker 层的版本统一（`python/sglang/srt/speculative/spec_registry.py:37-40`），当前的多套实现是**算法本身不同**导致的：这四个 worker 的验证/KV 裁剪语义有实质差异（非树验证、ragged 布局、无神经网络草稿这几种情况互相之间没有共同的抽象），继承共享基类 `EAGLEWorkerV2` 换不来真正的代码复用，只会强迫不同的算法削足适履去适配一套不适合它们的接口——[[10-SGLang-投机解码EAGLE]] `## 4` 已经用具体代码路径说明过这一点。反过来 `StandaloneWorkerV2`/`FrozenKVMTPWorkerV2` 这两个确实继承了 `EAGLEWorkerV2` 并复用了它的编排逻辑，说明团队并不是不知道"能复用就复用"，而是已经按"能不能共享语义"这条线做过一次取舍，四个独立实现是这次取舍之后的合理结果，不是没做完的重构。

**D. `raise NotImplementedError` 的 775 处，大多数是故意的后端能力占位符，不该被当成"未完工"的信号批量清理。** `python/sglang/kernels/fused_op.py:425-447` 是典型样本：`forward_triton`/`forward_jit`/`forward_aot`/`forward_cute_dsl`/`forward_flashinfer`/`forward_deepgemm`/`forward_aiter`/`forward_torch_npu` 八个方法，每一个都是 `raise NotImplementedError(f"{self._op_label()}: no {backend} backend")`——这是一个算子基类给"当前算子在这个后端没有实现"预留的标准占位符，每个具体算子子类只覆写自己真正支持的那几个后端方法，未覆写的自然继承基类的报错。`improve.json` 给出的 `not_impl` 抽样里，`python/sglang/kernels/ops/attention/flash_attention_v3.py:138`、`python/sglang/kernels/ops/attention/flash_attention_v4.py:264-268`、`python/sglang/kernels/ops/attention/fla/fused_recurrent.py:744`/`1192` 都是同一种"per-backend/per-硬件组合的能力矩阵占位符"模式。这类代码如果被当成"未完工"批量催办或者机械地加 `TODO`，只会制造噪音——它们大多数情况下**本来就不该有实现**（比如一个 AMD-only 的融合算子在"纯 NVIDIA CUDA 分支"里没有 `torch_npu` 版本是设计使然）。本条标「本库推断」：只对 `not_impl` 的一小撮样本做了精读，没有对全部 775 处做穷举分类，不能排除其中确实混有个别"本该实现但被遗忘"的真缺口——但仅凭"数字很大"就断言这是代码质量问题，本身就是本篇 `## 7` 反复强调要避免的那种误判。

## 9. 自测题与延伸阅读

**闭卷自测**（合上本文，能不能回答）：

1. jump-forward decoding 的三个方法（`try_jump_forward`/`jump_forward_str_state`/`jump_and_retokenize`）在四个语法后端里都有实现，为什么说它是"死代码"？全仓唯一的调用点在哪个文件、做了什么？outlines 后端为什么是"双重死代码"？
2. `get_ready_grammar_requests()` 里最长 5ms 的忙等，为什么会拖慢**跟语法编译完全无关**的纯解码请求？把 `_get_new_batch_prefill_raw` → `get_next_batch_to_run` → `event_loop_normal`/`event_loop_overlap` 这条调用链讲清楚。
3. `RadixCache` 类现在在生产环境里到底扮演什么角色？它和 `UnifiedRadixCache`、`HiRadixCache` 三者的可达性（能不能被生产工厂构造出来）分别是什么？
4. 材料清单里"speculative/ 下 v1/v2 worker 并存"这个说法为什么是错的？真正的证据在哪一行代码/文档？
5. "SGLang 热路径静默 except 密度是 vLLM 的 2.6 倍"——本篇认为这个数字被高估了，给出的两条独立理由分别是什么（跟"分类口径"和"分母定义"两个角度对应）？
6. 本篇给出的三条"不建议改"分别是什么？对 `server_args.py` 那一条,支持集中和反对集中的证据各是什么（各举一处源码/文档依据）？
7. `TODO` 认领率 24% 这个数字为什么会被空格问题拉低？举出仓库里的具体一行代码。

**延伸阅读**：

- [[06-SGLang-约束解码与语法后端]]——本篇 `## 8` 第 1、2、9、10 条（jump-forward、语法忙等、缓存 key 归一化、llguidance assert）全部建立在这一篇 `## 4`/`## 5`/`## 7` 已经做过的第一手走读之上，想看完整的语法后端架构、bitmask 生命周期、四个后端支持类型对照表,去那一篇。
- [[03-SGLang-RadixAttention与前缀缓存]]——本篇 `## 8` 第 3 条（`RadixCache`/`HiRadixCache` 死代码）依赖那一篇 `## 7` 第 1、2 条已经发现的"`RadixCache` 不是默认路径"这条线索,本篇在此基础上把 `HiRadixCache` 的死代码程度坐实到"全仓唯一构造点是一个单元测试"。
- [[10-SGLang-投机解码EAGLE]]——本篇 `## 7.2` 纠正的"v1/v2 并存"错误说法、以及「不建议改」第 3 条关于四个独立 worker 实现的论证,证据链完整版在那一篇的 `## 0` 第 2 条和 `## 4`。
