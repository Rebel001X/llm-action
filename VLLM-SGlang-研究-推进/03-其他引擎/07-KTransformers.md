# KTransformers：CPU/GPU 异构卸载与 YAML 注入式优化

> **本篇取证基准**：`ktransformers` @ `95009ea6`（2026-08-21）
> **一句话**：它不比谁快，它比谁能在一张消费卡上把 671B 跑起来。

## 0. 结论先行

- **这个仓库刚经历一次结构性瘦身，`archive/` 装着老仓库、`kt-kernel/` 才是现在真正发布的东西。** 按 `_lab/out/repo_stats.json` 的 `top_dirs`，`archive/kt-sft`（105,925 行）+ `archive/ktransformers`（81,039 行）+ `archive/csrc`（28,852 行）+ `archive/third_party`（26,873 行）合计约 242,689 行，占全仓 403,076 行的六成；而承载"现在还在维护、还能 `pip install` 的东西"的 `kt-kernel/` 全部子目录合计约 139,657 行。**读这个仓库最容易踩的坑就是拿 `archive/` 里的老代码当成当前架构在讲**——`README.md:159` 用一句话把这件事挑明了："The original integrated KTransformers framework has been archived to the `archive/` directory for reference."
- **新架构不是"KTransformers 自己的推理框架"，而是"kt-kernel 这个 CPU MoE 算子库 + kvcache-ai 自己维护的一份 SGLang fork"。** `pyproject.toml:24` 显示顶层 `ktransformers` 包现在只剩一个 `py-modules = ["ktransformers"]`（单文件 `ktransformers.py`），`setup.py:17`-`18` 把它的唯一依赖写成 `install_requires=["kt-kernel==<version>"]`——`pip install ktransformers` 现在做的事情，就是装一个转发用的壳子加 `kt-kernel`。真正的批处理、调度、HTTP 服务，全部外包给 `.gitmodules` 里声明的 `third_party/sglang`（`kvcache-ai/sglang.git`，`branch = main`）这份 fork，`install.sh:99`-`142` 的 `install_sglang()`/`install_kt_kernel()` 两个函数就是把这两块分别装起来。
- **老的 YAML 注入式优化机制（本库最值得看的设计）现在活在 `archive/ktransformers/optimize/`，不在新架构的默认路径上。** 匹配逻辑只有 163 行（`archive/ktransformers/optimize/optimize.py`），核心是 `gen_optimize_config`（`:67`）按前序遍历模块树，对每个模块名字符串跑 `re.search`（`:86`-`:87`），命中就把这个模块替换成 YAML 里指定的自定义算子类，替换的粒度精确到"专家层放 CPU、注意力层放 GPU"这种设备级配置。`## 2` `## 3` 会逐行展开。
- **新架构里的 CPU/GPU 异构卸载不再靠 YAML 正则匹配模块名，而是靠一个 Python 包装类 + 一个布尔掩码。** `kt-kernel/python/experts_base.py:21` 的 `generate_gpu_experts_masks` 按专家历史激活频率选 Top-K 放 GPU，`kt-kernel/python/experts.py:72` 的 `KTMoEWrapper` 拿着这个掩码在 SGLang 的 MoE 层里做工厂分发。这是同一个"冷热专家分居两块硬件"的想法，换了一套更贴近生产集成（而不是研究原型）的实现方式。
- **它没有自己的连续批处理调度器了。** 老架构确实有一个用 C++ 手写的调度器（`archive/csrc/balance_serve/sched/scheduler.h:161` 的 `class Scheduler`，`archive/csrc/balance_serve/sched/scheduler.cpp:826`/`:923` 的 `FCFS_single_prefill`/`FCFS`），但它现在整个躺在 `archive/` 里。新架构的 `kt-cli run` 子命令直接拼一条 `python -m sglang.launch_server` 命令（`kt-kernel/python/cli/commands/run.py:636`），把 `--max-running-requests`、`--max-total-tokens`、`--chunked-prefill-size` 这些调度参数原样转发给 SGLang（`:684`-`:698`），自己只在命令行里多插 `--kt-num-gpu-experts`、`--kt-cpuinfer` 这类专家放置参数（`:672`-`:679`）。**连续批处理这件事，KTransformers 现在完全不用自己操心。**
- **32 条抽出来的 API 路由全部来自 `archive/`，且 AST 抽取器在这上面交了一份"假阴性"报告：`/v1/*` 命中数是 0，不代表它不兼容 `/v1`。** 原因是 `archive/ktransformers/server/api/openai/__init__.py:7` 把前缀写在了父路由构造上（`APIRouter(prefix='/v1')`），子路由自己的装饰器只写 `/chat/completions` 这种相对路径，AST 扫描器如果只看单个 `@router.post(...)` 的字面路径就会漏掉这层拼接。`_lab/out/api_surface.json` 里 `unique_paths` 确实一条都不含 `/v1` 前缀，这是工具口径的问题，不是这个引擎真的没做 OpenAI 兼容。`## 7` 会把这条坑完整摊开。

## 1. 它在系统里的位置

### 1.1 它解的是哪道题

KTransformers 是清华 MADSys 实验室与 Approaching.AI 联合的研究项目，论文题目直接点出了它要解决的问题：《KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models》（SOSP 2025，见 `README.md` 引用块）。

它和 vLLM/SGLang 面对的不是同一道题——vLLM/SGLang 的默认假设是"模型能整个塞进（若干张）GPU 显存，问题是怎么伺候更多并发用户"；KTransformers 的默认假设是**"模型塞不进显存，问题是怎么让它至少能跑起来"**。

DeepSeek-671B 这类超大 MoE 模型，参数量远超单张消费级显卡（哪怕是 24GB 的 RTX 4090）能装下的量，但 MoE 的稀疏激活特性意味着**每个 token 真正用到的专家只是全部专家里的一小撮**。于是把不常用的"冷"专家放到便宜的 CPU 内存里，只在需要时临时算一下，把常用的"热"专家和注意力这类每 token 必算的部分留在 GPU 上，就成了在显存预算之外硬撑起超大模型的办法。

### 1.2 这次重构：从"一体化框架"到"算子库 + fork"

这条主线在这个快照里经历了一次工程路线的转向：**从"自己造一整套推理服务"转向"只造好 CPU 侧的算子库，把服务框架的活儿交给别人"**。

旧架构（`archive/ktransformers/`）里有一整套独立的东西——自己的 HTTP server（`archive/ktransformers/server/`）、自己的 YAML 驱动模块替换机制（`archive/ktransformers/optimize/`）、自己用 C++ 写的连续批处理调度器（`archive/csrc/balance_serve/`）。新架构（`kt-kernel/`）只保留"CPU 端怎么把 MoE 专家算快"这一件事，把它打包成一个可以 `pip install kt-kernel` 装、能被 SGLang（准确说是 kvcache-ai 自己维护的 SGLang fork，包名 `sglang-kt`）当作后端调用的 Python/C++ 库。

`README.md:157`-`163` 用"KT original Code"这个标题把这次转向写得很直白：两条能力入口（Inference 和 SFT）现在都从 `kt-kernel` 源码树里组织，老框架整体降级为"仅供参考"。

值得一提的是，`kt-kernel` 本身也不是无中生有——它的 CPU 算子实现（`amx.py`/`llamafile.py`/`moe_kernel.py`）和老架构 `archive/ktransformers/operators/experts.py` 里的 `KExpertsCPU`/`KExpertsMarlin` 系列在算法思路上是同一套东西的第二次实现：都是"按量化格式分发到不同 CPU kernel"，区别在于旧实现是被 YAML 注入进一个完整 `nn.Module` 树的自定义层，新实现是一个独立分发、被外部框架按需调用的包装类（`## 3` 详细对比这两种数据结构）。

从 `README.md` 的更新日志（`Updates` 一节）能拼出这次转向的时间线——注意这里只摘录"做了什么、什么时候做的"这类事实性条目，不引用条目里带的吞吐倍数说法（那些数字缺硬件/模型/口径，本库不采信，`00-总览与阅读地图.md` 已定的红线）：

| 时间 | 事件 | 出处 |
|---|---|---|
| 2025-02-10 | 支持 DeepSeek-R1/V3 单卡（24GB 显存）+ 多卡异构推理 | `README.md:49` |
| 2025-04-02 | 支持多并发（`balance_serve` 引擎首次发布） | `README.md:44` |
| 2025-10-10 | 宣布"接入 SGLang"（Integrating into SGLang），链接指向 SGLang 官方 issue #11425 | `README.md:35` |
| 2026-04-30 | v0.6.1 版本把文档拆成 Inference（`kt-kernel/`）与 SFT 两条独立入口 | `README.md:23` |

这条时间线印证了 `## 0` 的判断：**"接入 SGLang"不是这个快照突然出现的决定，是一条从 2025 年 10 月就公开在推进、到这个快照（2026-08-21）已经落地成默认路径的渐进转向**——`balance_serve`（2025-04）曾经是官方主推的"自建多并发引擎"，十八个月后它变成了 `archive/` 里的历史存档，接盘的正是当初被拿来"借鉴思路"的 SGLang。

两代架构的拓扑差异画成图更直观：

```
旧架构（archive/，已归档）                新架构（当前默认路径）
┌─────────────────────────┐            ┌──────────────────────────┐
│  archive/ktransformers/  │            │  kt-cli run <model>       │
│  server/  (自研 HTTP)     │            │  (kt-kernel/python/cli)   │
│    ├─ api/openai/         │            └────────────┬─────────────┘
│    ├─ api/ollama/         │                         │ subprocess.run(cmd)
│    └─ api/web/            │                         ▼
│  backend/interfaces/       │            ┌──────────────────────────┐
│    └─ balance_serve.py     │            │  python -m                │
│         │                  │            │  sglang.launch_server     │
│         ▼                  │            │  （third_party/sglang，   │
│  archive/csrc/              │            │   kvcache-ai fork）        │
│    balance_serve/sched/     │            │    ├─ 自己的连续批处理调度  │
│    scheduler.cpp (FCFS,     │            │    ├─ 自己的 KV cache 管理 │
│    C++ 手写调度器)           │            │    └─ 自己的 HTTP 协议层   │
│         │                   │            │         │                 │
│         ▼                   │            │         ▼ 调用            │
│  archive/ktransformers/      │           │  kt-kernel (pip 包)        │
│    optimize/ (YAML 注入)      │           │    KTMoEWrapper /          │
│    operators/ (KExpertsCPU)   │           │    AMXMoEWrapper 等         │
└─────────────────────────┘            └──────────────────────────┘
```

左边是一整套自己造的服务系统，右边只剩最下面一层是 KTransformers 自己的代码。

**这张图本身就是 `## 6` "KTransformers 已经不在调度/协议这两层竞争"这个判断的可视化**。

### 1.3 代码规模的重新分布

这次拆分在行数上留下了清晰的痕迹。按 `_lab/out/repo_stats.json` 的 `top_dirs`：

| 目录 | 文件数 | 行数 | 属于 |
|---|---:|---:|---|
| `archive/kt-sft` | 342 | 105,925 | 归档：老的独立 SFT 框架（`archive/ktransformers` 的又一份分叉） |
| `archive/ktransformers` | 244 | 81,039 | 归档：老的一体化推理框架 |
| `kt-kernel/operators` | 105 | 61,576 | 当前：CPU MoE 算子的 C++/头文件实现（AMX/AVX2/llamafile 等） |
| `kt-kernel/python` | 64 | 32,449 | 当前：`KTMoEWrapper` 等 Python 包装层 + CLI + SFT 适配 |
| `archive/csrc` | 158 | 28,852 | 归档：老框架的 C++/CUDA 算子与 `balance_serve` 调度器 |
| `archive/third_party` | 34 | 26,873 | 归档：老框架自带的第三方依赖快照 |
| `kt-kernel/examples` | 38 | 12,719 | 当前：接入示例 |
| `kt-kernel/test` | 48 | 11,715 | 当前：`kt-kernel` 自己的测试 |

四个 `archive/*` 目录合计约 242,689 行，占全仓 403,076 行的六成；`kt-kernel/` 全部子目录合计约 139,657 行，占约三成五。

**读这份代码地图最容易犯的错误，就是被"archive 占了六成"这个数字带偏，误以为老框架依然是主线**——行数占比反映的是"历史包袱有多重"，不是"当前维护重心在哪"，`README.md:159` 那句归档声明才是权威口径。

## 2. 代码地图（文件 → 职责，带行号）

按"读这个仓库该先看哪块"分成三组：顶层包怎么把两条线粘起来、新架构的异构卸载代码在哪、旧架构（归档）的三大件在哪。

### 2.1 顶层包与安装：两条线怎么粘起来

| 层 | 文件:行 | 干什么 |
|---|---|---|
| 顶层包（新） | `pyproject.toml:24` | `py-modules = ["ktransformers"]`：顶层包只剩一个转发壳文件 |
| 顶层包（新） | `setup.py:17`-`18` | `install_requires=[f"kt-kernel=={_v}"]`：`pip install ktransformers` 实质是装 `kt-kernel` |
| 顶层包（新） | `ktransformers.py:1`-`10` | 模块 docstring 直接写明"The runtime kernels live in kt-kernel" |
| 一键安装 | `install.sh:93` | `init_submodules()` 里的 `git submodule update --init --recursive` |
| 一键安装 | `install.sh:99`-`142` | `install_sglang()` / `install_kt_kernel()`：分别装 fork 版 SGLang 和 `kt-kernel` |
| 归档说明 | `README.md:159` | 明确声明老框架已整体归档到 `archive/` |

### 2.2 新架构：CPU/GPU 异构卸载与 CLI

| 层 | 文件:行 | 干什么 |
|---|---|---|
| CLI 入口 | `kt-kernel/python/cli/commands/run.py:636` | `_build_sglang_command` 拼出 `python -m sglang.launch_server ...` |
| CLI 入口 | `kt-kernel/python/cli/commands/run.py:567` | `subprocess.run(cmd, env=env)`：真正拉起 SGLang 进程 |
| CLI 入口 | `kt-kernel/python/cli/commands/run.py:649`-`657` | `use_kt_kernel` 判定逻辑：有量化权重或配置了 CPU 卸载才启用 |
| CLI 入口 | `kt-kernel/python/cli/commands/run.py:664`-`680` | 注入的 kt 专属参数：`--kt-num-gpu-experts`、`--kt-cpuinfer`、`--kt-method` 等 |
| CLI 入口 | `kt-kernel/python/cli/commands/run.py:685`-`702` | 原样转发给 SGLang 的调度参数：`--max-running-requests`、`--chunked-prefill-size` 等 |
| CLI 旋钮 | `kt-kernel/python/cli/commands/run.py:56` | `@click.option("--kt-method", ...)`：`kt-cli run` 自己的 kt 专属旋钮 |
| CLI 旋钮 | `kt-kernel/python/cli/commands/run.py:62` | `@click.option("--max-running-requests", ...)`：直接对应 SGLang 的旋钮名 |
| 异构卸载 | `kt-kernel/python/experts_base.py:21` | `generate_gpu_experts_masks`：按激活频率选 Top-K 专家放 GPU |
| 异构卸载 | `kt-kernel/python/experts_base.py:227` | `class BaseMoEWrapper`：CPU MoE 包装类的公共基类 |
| 异构卸载 | `kt-kernel/python/experts_base.py:377` | `submit_forward`：向 CPU 线程池异步提交专家计算任务 |
| 异构卸载 | `kt-kernel/python/experts_base.py:427` | `self.cpu_infer.submit_with_cuda_stream(...)`：任务与 CUDA 流绑定，做 CPU/GPU 计算重叠 |
| 异构卸载 | `kt-kernel/python/experts_base.py:485` | `forward`：`submit_forward` + `sync_forward` 的同步封装 |
| 异构卸载 | `kt-kernel/python/experts.py:72` | `class KTMoEWrapper`：外部代码用的工厂类，按 `method`/`mode` 分发到具体后端实现 |
| 异构卸载 | `kt-kernel/python/experts.py:122` | `KTMoEWrapper.__new__`：真正做分发决策的地方 |
| 异构卸载 | `kt-kernel/python/utils/amx.py:248` | `class AMXMoEWrapper`：Intel AMX 指令集专属实现 |

### 2.3 旧架构（归档）：YAML 注入、CPU 算子、调度器、HTTP

| 层 | 文件:行 | 干什么 |
|---|---|---|
| YAML 注入 | `archive/ktransformers/optimize/optimize.py:67` | `gen_optimize_config`：前序遍历模块树，逐条 YAML 规则做匹配 |
| YAML 注入 | `archive/ktransformers/optimize/optimize.py:28` | `inject`：真正把命中的模块替换成自定义算子类实例 |
| YAML 注入 | `archive/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml:46`-`55` | MoE 专家层规则：`generate_device: cpu` + `generate_op: KExpertsCPU` |
| 入口触发 | `archive/ktransformers/local_chat.py:139` | `optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)`：YAML 注入的调用点 |
| CPU 算子 | `archive/ktransformers/operators/experts.py:143` | `class KExpertsCPU`：老框架里 CPU 侧专家算子的具体实现 |
| CPU 算子 | `archive/ktransformers/operators/experts.py:167` | `self.backend = kwargs.get("backend", "llamafile")`：默认走 llamafile GEMM 后端 |
| 调度器 | `archive/csrc/balance_serve/sched/scheduler.h:161` | `class Scheduler`：纯虚接口，`add_query`/`update_last_batch` |
| 调度器 | `archive/csrc/balance_serve/sched/scheduler.h:143` | `struct QueryAdd`：单个请求的调度状态（token、SLO 目标） |
| 调度器 | `archive/csrc/balance_serve/sched/scheduler.cpp:923` | `struct FCFS`：先到先服务的连续批处理实现 |
| 调度器 | `archive/ktransformers/server/balance_serve/sched_rpc.py` | `sched_ext.create_scheduler`：Python 侧通过 ZMQ 调用编译好的 C++ 调度器 |
| HTTP 路由 | `archive/ktransformers/server/api/openai/__init__.py:7` | `router = APIRouter(prefix='/v1')`：`/v1` 前缀在这里拼接，不在各端点装饰器上 |
| HTTP 路由 | `archive/ktransformers/server/api/ollama/completions.py:18` | `router = APIRouter(prefix='/api')`：Ollama 协议端点挂在 `/api` 下 |
| HTTP 路由 | `archive/ktransformers/server/main.py:34` | `mount_app.include_router(router)`：三套协议路由在这里汇总挂载 |

## 3. 核心数据结构

**旧架构：一个从 YAML 生成的 `dict`，键是模块的点分路径。** `gen_optimize_config`（`archive/ktransformers/optimize/optimize.py:67`）遍历模型的 `nn.Module` 树时，会给每个命中规则的模块生成一条形如
```
{"model.layers.3.mlp.experts": {"class": "ktransformers.operators.experts.KTransformersExperts",
                                 "key": "model.layers.3.mlp.experts",
                                 "kwargs": {"prefill_device": "cuda", "prefill_op": "KExpertsTorch",
                                            "generate_device": "cpu", "generate_op": "KExpertsCPU",
                                            "out_device": "cuda"}}}
```
的记录（字段名照抄 `archive/ktransformers/optimize/optimize.py:96`-`98` 的构造逻辑），没命中任何规则的模块会兜底落一条 `"class": "default"`（同文件 `:105`-`110`）。

这个 `dict` 随后被 `inject`（`archive/ktransformers/optimize/optimize.py:28`）逐层消费：命中就 `getattr(__import__(...), ...)` 动态 import 出替换类，用 `module_cls(key=..., gguf_loader=..., config=..., orig_module=child, **kwargs)` 构造一个新模块，`set_module` 换掉原来的子模块。

**这一步发生在 `torch.device("meta")` 上下文里**（`archive/ktransformers/optimize/optimize.py:141`、`:150`），也就是说替换时权重还没真正加载，只是在搭"应该长成什么形状"的骨架，随后才用 `load_weights` 把真实权重按各自的 `generate_device`/`prefill_device` 搬到对应硬件上。

**新架构：一个布尔掩码 + 一个工厂类。** `generate_gpu_experts_masks`（`kt-kernel/python/experts_base.py:21`）输入形状 `(num_layers, num_experts)` 的激活频率表，输出同形状的布尔掩码，`True` 表示"这个专家放 GPU"。`KTMoEWrapper.__new__`（`kt-kernel/python/experts.py:122`）拿着一长串参数（下表摘录关键几个及其行号），按 `mode`（`"inference"`/`"sft"`）分发到 `AMXMoEWrapper`/`NativeMoEWrapper`/`LlamafileMoEWrapper`/`GeneralMoEWrapper` 四选一的具体实现（导入语句见 `kt-kernel/python/experts.py:26`-`29`）：

| 参数 | 行号 | 作用 |
|---|---|---|
| `gpu_experts_mask` | `kt-kernel/python/experts.py:129` | 哪些专家放 GPU，`None` 则全放 CPU |
| `cpuinfer_threads` | `kt-kernel/python/experts.py:130` | CPU 推理线程总数 |
| `weight_path` | `kt-kernel/python/experts.py:132` | CPU 侧权重路径（量化后格式） |
| `cpu_save` | `kt-kernel/python/experts.py:135` | 是否把权重常驻 CPU 内存 |
| `max_deferred_experts_per_token` | `kt-kernel/python/experts.py:136` | 每 token 推迟到下一层计算的专家数（`## 5` 决策二） |
| `method` | `kt-kernel/python/experts.py:138` | CPU 量化后端标识：`"AMXINT4"`/`"AMXINT8"`/`"LLAMAFILE"` 等 |
| `mode` | `kt-kernel/python/experts.py:140` | `"inference"` 或 `"sft"`，决定走哪条分发路径 |
| `num_gpu_experts` | `kt-kernel/python/experts.py:142` | SFT 模式下改用"放 GPU 的专家数量"而非掩码 |

没有 YAML，没有正则匹配模块名——这个决定权从"声明式规则文件"搬回了"调用者传参数"，`## 5` 会展开这个取舍。

**一条跨两代架构都成立的暗线：prefill 和 decode 对"专家放哪"的答案不一样。** 旧架构 `archive/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml:50`-`53` 那条专家层规则，`prefill_op` 填的是 `"KExpertsTorch"`（GPU 上跑）、`generate_op` 才是 `"KExpertsCPU"`（CPU 上跑）——同一个专家，prefill 阶段和 decode 阶段用的是两种不同硬件的实现。

新架构里这条逻辑没有消失，变成了一个可调参数：`kt-kernel/python/cli/commands/run.py:447` 的 `final_kt_gpu_prefill_threshold = resolve(kt_gpu_prefill_threshold, "inference.kt_gpu_prefill_token_threshold", 4096)`，对应 CLI 选项 `--kt-gpu-prefill-threshold`（`kt-kernel/python/cli/commands/run.py:58`，默认 4096 token）。

背后的道理是一致的：**prefill 阶段一次要处理一整段 prompt，tokens 多、计算密集，扔给 GPU 并行算比排队等 CPU 划算；decode 阶段每步只出一个 token，计算量小但要反复做，CPU 闲着不用才是浪费**——这就是为什么"专家放哪"不是一个一次性决定，而是要按 prefill/decode 两种负载模式分别回答的问题，新旧两代实现用不同的机制表达了同一个判断。

**支撑异步重叠的底层结构：双缓冲的 pinned memory 池。** `class KExpertsCPUBuffer`（`kt-kernel/python/experts_base.py:75`）管理一组固定内存（`pin_memory=True`，`kt-kernel/python/experts_base.py:101`）缓冲区，`buffer_depth: int = 2`（`kt-kernel/python/experts_base.py:86`）意味着每层的输入/输出张量都有两份槽位——`## 4`/`## 5` 提到的 `submit_forward` 用 `current_slot = self.layer_idx % KExpertsCPUBuffer.buffer_depth` 决定这一层写哪个槽位，下一层自动切到另一个槽位。这是典型的双缓冲手法：**GPU 还在读上一层的结果时，CPU 已经可以开始往另一个槽位写这一层的新结果，两者不用互相等对方"腾地方"**。

`get_buffer`（`kt-kernel/python/experts_base.py:89`）按 batch size 缓存这组张量，命中 `capture_bs` 里预设的批大小时直接复用（对应 `## 8` 提到的 `set_capture_batch_sizes` 预分配接口），避免每次推理都重新分配 pinned memory——这类内存的分配本身开销不小，运行时反复申请/释放会拖慢首 token 延迟。

**旧架构里"热专家"放 GPU 还有第二种实现，不止 `KExpertsTorch` 一种。** `archive/ktransformers/operators/experts.py:437` 的 `class KExpertsMarlin(KExpertsBase)` 是专门给 GPTQ 量化权重用的 GPU 侧算子——如果某个专家被 YAML 规则标记为 `prefill_op: "KExpertsMarlin"`，跑的就不是 `KExpertsTorch` 那条朴素 PyTorch 路径，而是 Marlin 这套针对 GPTQ INT4 权重优化的 GPU GEMM kernel。这条线索说明"专家放 GPU"这件事本身还分好几种具体实现，选哪种取决于权重的量化格式——**"CPU/GPU 异构"这个大判断之下，还叠着一层"用什么算子跑"的小判断**，两层判断互相独立、可以分别调整。

官方文档给出的标准用法印证了这张表（`kt-kernel/README.md:565`-`596`，"文档所述"）：

```python
from kt_kernel import KTMoEWrapper

wrapper = KTMoEWrapper(
    layer_idx=0, num_experts=8, num_experts_per_tok=2,
    hidden_size=4096, moe_intermediate_size=14336,
    num_gpu_experts=2, cpuinfer_threads=32, threadpool_count=2,
    weight_path="/path/to/weights", chunked_prefill_size=512,
    method="AMXINT4",
)
wrapper.load_weights(physical_to_logical_map)
output = wrapper.forward(hidden_states, topk_ids, topk_weights, cuda_stream)
```

**旧架构的调度器状态：`QueryAdd` / `FCFS`。** `struct QueryAdd`（`archive/csrc/balance_serve/sched/scheduler.h:143`，定义在 `class Scheduler` 之前）携带 `query_token`、`SLO_TTFT_ms`、`SLO_TBT_ms` 等字段；`class Scheduler`（`archive/csrc/balance_serve/sched/scheduler.h:161`）是一个纯虚接口，核心方法只有四个：

```cpp
class Scheduler {
 public:
  virtual void init(Settings settings) = 0;
  virtual void run() = 0;
  virtual void stop() = 0;
  virtual QueryID add_query(QueryAdd query) = 0;          // webserver 调这个
  virtual void cancel_query(QueryID id) = 0;
  virtual std::shared_ptr<BatchQueryTodo>
  update_last_batch(BatchQueryUpdate updates) = 0;         // 推理循环调这个
  virtual InferenceContext get_inference_context() = 0;
};
```

`struct FCFS`（`archive/csrc/balance_serve/sched/scheduler.cpp:923`）继承自 `FCFS_single_prefill`（`archive/csrc/balance_serve/sched/scheduler.cpp:826`），核心是把 in-flight 请求按到达顺序分成 `prefill_mini_batches` 和 `decode_mini_batches` 两条队列，每一步调度都用 `settings.recommended_chunk_prefill_token_count` 做 chunked prefill（该字段的赋值散布在 `archive/ktransformers/server/balance_serve/settings.py` 多处 `create_sched_settings_*` 函数里）。

Python 侧不直接跑这段 C++——`archive/ktransformers/server/balance_serve/sched_rpc.py` 里的 `SchedulerServer` 通过 `sched_ext.create_scheduler(settings)`（pybind11 绑定）拿到一个调度器实例，再用 ZeroMQ 的 `ROUTER`/`DEALER` 套接字对外提供 RPC，`add_query`/`update_last_batch` 这两个虚方法就是 RPC 调用的落点。

这一整套状态机现在没有调用者了——新架构里同等职责的状态（`running`/`waiting` 队列、chunked prefill 预算）全部长在 SGLang fork 内部，KTransformers 这边看不到。

## 4. 主流程走读

**旧架构（归档，仍是理解"注入"这个设计的最佳样本）：** 用户跑 `local_chat.py`，走读它的七步：

1. `archive/ktransformers/local_chat.py:34` `from ktransformers.optimize.optimize import optimize_and_load_gguf`。
2. `:58`-`:77` 的 `default_optimize_rules` 字典按模型架构名（如 `"DeepseekV3ForCausalLM"`）映射到一份默认 YAML 路径——这是"选哪份规则文件"这件事的默认值来源。
3. `:125`-`:128` 如果用户没在命令行传 `--optimize_config_path`，就按架构名查上面那张表。
4. `:139` `optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)` 是唯一的调用点，从这里往下进入 `optimize.py`。
5. `archive/ktransformers/optimize/optimize.py:131` 先 `yaml.load` 整份规则列表（`Loader=yaml.FullLoader`）。
6. `archive/ktransformers/optimize/optimize.py:134` 调 `gen_optimize_config` 生成替换计划（`## 3` 描述的那个 `dict`）。
7. `archive/ktransformers/optimize/optimize.py:141`、`:150` 调 `inject`，在 `torch.device("meta")` 上下文里重建模型骨架，随后 `load_weights` 按各自的 `generate_device`/`prefill_device` 把真实权重分头加载到 CPU/GPU。

整个过程发生在进程启动阶段（不是请求处理阶段），跑完之后模型对象里的每个子模块要么是原生 PyTorch 层，要么是被替换掉的 `K*` 自定义算子。

**"注入"这个词准确描述了它的效果：外部看起来还是同一个 `transformers` 模型类，内部实现已经被换了一遍**。

**新架构（当前默认路径）：** 用户跑 `kt-cli run <model>`。走读它拼命令的六步：

1. 命令落到 `kt-kernel/python/cli/commands/run.py:606` 的 `_build_sglang_command` 函数。
2. `:649`-`:657` 判断要不要用 `kt-kernel`：只要传了量化权重路径（`:652`-`:654`），或者配置了 CPU 线程数/GPU 专家数（`:655`-`:657`），就把 `use_kt_kernel` 置 `True`；否则这次启动根本不碰 CPU 卸载，纯 GPU 跑 SGLang。
3. `:633`-`:643` 拼出固定前缀：`[sys.executable, "-m", "sglang.launch_server", "--host", host, "--port", str(port), "--model", str(model_path)]`——第 636 行的字符串 `"sglang.launch_server"` 是全篇最直接的证据：**这不是它自己的服务进程，是在拉起 SGLang 的官方启动模块**。
4. 若 `use_kt_kernel` 为真，`:664`-`:680` 追加 kt 专属参数：`--kt-weight-path`（`:668`附近）、`--kt-cpuinfer`、`--kt-num-gpu-experts`（`:672`）、`--kt-method`（`:674`）、`--kt-gpu-prefill-token-threshold`（`:676`）。
5. `:685`-`:702` 追加 SGLang 原生调度参数：`--attention-backend`、`--mem-fraction-static`、`--chunked-prefill-size`、`--max-running-requests`、`--max-total-tokens`、`--tensor-parallel-size`——这些字符串和 SGLang 自己的 `ServerArgs` 字段名一一对应，KTransformers 这边只是原样转发，不做任何二次解释。
6. `kt-kernel/python/cli/commands/run.py:567` 的 `subprocess.run(cmd, env=env)` 把整条命令当子进程拉起来。

**这六步走完，KTransformers 这边的"主流程"就结束了**——剩下的请求调度、批处理、KV cache 管理全部是 SGLang fork 进程内部的事，`kt-cli` 从这一刻起只是个旁观的父进程。

SGLang 进程跑起来之后，遇到需要 MoE 专家计算的地方会调用 `kt-kernel` 提供的 Python API（`KTMoEWrapper` 及其 `submit_forward`/`sync_forward`），这条调用边界就是 `kt-kernel/python/experts_base.py:485` 的 `forward` 方法——**这是新架构里唯一还留在 KTransformers 手上的计算逻辑**。

## 5. 设计决策与代价（异构卸载三样）

三条决策合起来才是"异构卸载"这件事的完整代价账本，先给一张速览表，细节在下面逐条展开：

| 决策 | 解决什么 | 主要代价 |
|---|---|---|
| 冷专家放 CPU、热路径留 GPU | 显存装不下整份权重 | CPU 算力慢、PCIe 带宽、并发时线程池排队 |
| 异步提交 + 延后同步（可选延迟专家） | GPU 和 CPU 的计算串行等待 | 工程复杂度；延迟专家有未量化的准确度代价 |
| 调度/HTTP 外包给 SGLang fork | 独立维护调度器的工程成本 | 被锁定在一个可能滞后官方版本的私有 fork 上 |

**决策一：把"不常用的专家"放 CPU，"每 token 必算的部分"（注意力 + 常用专家）留 GPU。**
- **为什么这么设计**：MoE 模型的显存占用主要来自专家权重，但每个 token 只激活其中一小部分（DeepSeek-V3 是 256 选 8）。把全部专家常驻显存是给"可能被激活"的容量买单，而不是给"实际被激活"的计算买单——这笔账在专家数远超单卡容量时永远亏。

  CPU 侧内存便宜且量大（几百 GB 到几 TB 很常见），拿它装那些偶尔才被路由到的"冷"专家，是**在显存预算之外硬造出可用容量的唯一办法**。旧架构里 `archive/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml:52`-`53` 的 `generate_device: "cpu"` / `generate_op: "KExpertsCPU"` 和新架构里 `generate_gpu_experts_masks`（`kt-kernel/python/experts_base.py:21`）按激活频率选 Top-K 放 GPU，是同一个判断在两套实现里的两次重申。

- **不这样会怎样**：DeepSeek-671B 的全参数量在 FP8 精度下也有几百 GB，不做卸载意味着要么买够多张 H100/H200 拼出这个显存（不是"消费级硬件"能碰的价位），要么直接 OOM，模型加载阶段就失败——不是"变慢"，是"跑不起来"。

  这正是 KTransformers 论文标题里 "Unleashing the Full Potential" 这句话的字面所指：不是优化一个已经能跑的东西，是让一个原本跑不了的东西变得能跑。
- **什么时候这是错的**：三种情况。

  1. **显存本来就够**——如果模型能整个装进 GPU 集群，卸载到 CPU 纯属倒贴：CPU 算力比 GPU 慢一到两个数量级，PCIe/NVLink 搬运数据的延迟也比不搬动更高，这时候该用 vLLM/SGLang 的纯 GPU 路径，而不是 KTransformers。
  2. **PCIe 带宽在多请求下先撑不住**——CPU 专家算完的结果要搬回 GPU 参与后续注意力计算，请求数一多，PCIe 总线上的数据搬运量线性增长，很容易先于计算本身成为瓶颈，`kt-kernel/python/experts_base.py:427` 那个 `submit_with_cuda_stream` 试图靠异步流重叠这段延迟，但重叠得再好也遮不住带宽上限。
  3. **并发一高，CPU 侧算力直接被打爆**——`KTMoEWrapper` 的 CPU 线程池是全局单例（`kt-kernel/python/experts_base.py` 的 `_MoEBase._get_cpu_infer` 用 `cls._cpu_infer_instance` 做懒加载单例），所有并发请求的"冷"专家计算都排在同一批 NUMA 线程池上，请求数一高就要排队。

  这也是为什么 `README.md` 给出的官方吞吐数字（8×L20 GPU + Xeon Gold 6454S，227.85 tokens/s 总吞吐/87.58 tokens/s 输出吞吐，8-way 并发）明确标了硬件配置：**这是一个 CPU 强、GPU 也不弱的服务器级配置在跑，不是单卡消费级场景下的数字，两者不能互相套用**。

**决策二：用"提交异步任务 + 延后同步"而不是同步阻塞调用 CPU 专家计算。**
- **为什么这么设计**：`submit_forward`（`kt-kernel/python/experts_base.py:377`）把 CPU 计算任务提交给线程池后立刻返回，真正等结果发生在后续单独调用的 `sync_forward`（`:457`）——中间这段时间 GPU 可以去做别的事（比如算下一层的注意力）。

  如果每次都同步阻塞，GPU 就要干等 CPU 把专家算完，等于把两条硬件的延迟直接串联相加。

- **不这样会怎样**：GPU 利用率会打折扣，端到端延迟约等于"GPU 计算时间 + CPU 计算时间"的简单相加，而不是两者尽量重叠后取较大值。

- **什么时候这是错的**：如果 CPU 计算时间远大于 GPU 那一部分（比如专家数很多、CPU 算力本身偏弱），重叠能省下的时间占比很小，这时候投入工程复杂度换来的收益有限。

  `max_deferred_experts_per_token`（`kt-kernel/python/experts_base.py:248`）这个参数把部分低权重专家的计算推迟到下一层再算，进一步拉长重叠窗口，但代价是这些专家的贡献会**迟一层才计入残差流**——这是一个未公开量化过准确度损失的近似，官方文档没有给出这个参数对生成质量的影响幅度，本库对此标注为**未查证**，不代表它没有代价。

**决策三（附带的架构决策）：不再自己维护调度器/HTTP 服务，把这块整体交给一份 SGLang fork。**
- **为什么这么设计**：`archive/csrc/balance_serve/` 里那套连续批处理调度器是团队自己用 C++ 手写的（`doc/en/balance-serve.md:22` 原话说"Inspired by the scheduling framework of sglang...implemented high-performance asynchronous concurrent scheduling in C++"）。维护一整套独立的调度器、KV 管理、HTTP 协议层，工程成本和 SGLang/vLLM 这些专职做这件事的项目相比没有优势——把精力收回到"CPU MoE 算子这一件事能不能做到极致"，是更聚焦的打法。
- **不这样会怎样**：继续维护两套独立的服务框架（老 `ktransformers` 自己的 + 借鉴 SGLang 思路重写的 `balance_serve`），团队规模撑不住持续追赶 SGLang/vLLM 在调度、PD 分离、投机解码这些方向上的迭代速度。
- **什么时候这是错的**：这个决策把 KTransformers 的可用性焊死在"必须用 `kvcache-ai/sglang` 这份 fork，不能用官方 SGLang"这条约束上（`kt-kernel/README.md:280` 明确写了"Use `sglang-kt`...not the official `sglang` package"）。

  如果这份 fork 落后官方 SGLang 太多个版本，或者停止同步上游更新，用户就被锁在一个功能滞后的分支上，这是典型的"依赖一个私有 fork"的代价。

## 6. 同位对照（vLLM 在同一位置怎么做）

KTransformers 和 vLLM 不是同一道题的两种解法，是两道不同的题：**vLLM/SGLang 优化的是"给定固定显存，尽量多塞并发请求、尽量提高吞吐"；KTransformers 优化的是"给定固定显存，尽量把塞不下的模型跑起来"**。先用一张表把几个可以对齐的维度摆出来，再逐条展开：

| 维度 | vLLM | KTransformers（当前架构） |
|---|---|---|
| 默认硬件假设 | 权重/KV cache/激活值都在 GPU 显存里 | 先承认装不下，主动把一部分权重放 CPU |
| 连续批处理调度器 | 自己实现，`` `vllm:vllm/v1/core/sched/scheduler.py:73` `` | 不自己实现，委托给 `third_party/sglang`（kvcache-ai fork） |
| 进程角色 | 一整套独立服务系统（前端 + EngineCore + GPU worker） | 一个被上层服务系统（SGLang 进程）调用的算子库 |
| HTTP/协议层 | 自己维护，多协议兼容（OpenAI/Anthropic/Cohere） | 不自己维护，继承 SGLang fork 的协议层 |
| 优化的目标函数 | 每秒处理更多并发请求（吞吐） | 让原本装不下的模型先能跑起来（可行性） |

- **调度策略**：vLLM 的连续批处理调度器是 `` `vllm:vllm/v1/core/sched/scheduler.py:73` `` 的 `Scheduler` 类，FCFS 语义下抢占选的是最新请求（`self.running.pop()`，LIFO，见 `00-总览与阅读地图.md` 已查实的硬结论）。

  KTransformers 旧架构自己写的 `archive/csrc/balance_serve/sched/scheduler.cpp:923` 的 `struct FCFS` 也是先到先服务语义，但它现在**不是这个引擎的调度器**——新架构直接把这一层整体委托给 `third_party/sglang`（kvcache-ai fork），相当于放弃了"自己实现调度"，改成"把调度当黑盘依赖"。这条对比本身就是本篇 `## 0` 的核心结论：不是两种调度算法孰优孰劣,而是**KTransformers 已经不在调度这个问题上竞争了**。

- **显存/硬件的默认假设**：vLLM 的 `GPUModelRunner`（`` `vllm:vllm/v1/worker/gpu_model_runner.py:501` ``）默认假设权重、KV cache、激活值都在 GPU 显存里，唯一要精打细算的是"一张卡（或几张卡）里怎么切"。

  KTransformers 的默认假设反过来：**先承认装不下**，`generate_gpu_experts_masks`（`kt-kernel/python/experts_base.py:21`）要解的问题是"这一批专家权重，哪些子集配得上住进显存"。这也是为什么 vLLM 篇幅里从来不会出现"CPU 计算专家"这种设计——那道题在 vLLM 的默认工作点上不存在。

- **进程模型**：vLLM V1 的默认拓扑是"前端进程 + 独立 EngineCore 进程 + 若干 GPU worker 子进程"（见 `01-vLLM-全景与代码地图`）。

  KTransformers 新架构里，`kt-cli run` 本身只是一层薄壳，真正的进程是它 `subprocess.run` 拉起来的 `sglang.launch_server`（`kt-kernel/python/cli/commands/run.py:636`）——KTransformers 贡献的是这个进程内部调用的一个 Python/C++ 库，而不是一个独立的进程角色。这个差别决定了两者根本不是同一层级的比较对象：vLLM 是一整个服务系统，`kt-kernel`（新架构下）更接近 FlashAttention 之于 vLLM 的位置——一个被上层服务系统调用的算子库。

- **HTTP/协议层的"自研 vs 继承"**：vLLM 自己维护一整套多协议入口（`vllm/entrypoints/openai/`、`anthropic/`、`cohere/`，见 `01-vLLM-全景与代码地图`），这是它作为独立服务系统的必然成本。

  KTransformers 旧架构也自己维护过一套（`archive/ktransformers/server/api/`，`## 2.3` 已列出），但新架构里这层协议兼容性完全来自它所依赖的 SGLang fork——**KTransformers 团队现在没有自己的 HTTP 协议代码需要维护**，这也是"聚焦到 CPU 算子"这个决策在协议层面的直接体现。

这张对照表本身也说明了一件事：**把 KTransformers 和 vLLM 放在"谁的调度更好""谁的协议更全"这类维度上比较，从新架构开始就已经问错了问题**——调度和协议这两层它已经不参与竞争。

真正该比的是"CPU MoE 算子这一件事，KTransformers 做得比其他同类方案（比如 llama.cpp 的 CPU offload）好在哪"，这个问题需要另开一篇讲 CPU 算子内部实现的文章才能回答完整，本篇止步于"它现在长什么样、这个形状怎么来的"。

## 7. 踩坑与反直觉

**"32 条路由，`/v1/*` 命中数是 0"看起来像"这个引擎不兼容 OpenAI"，实际是前缀拼接方式骗过了 AST 扫描器。** `archive/ktransformers/server/api/openai/__init__.py:7` 把 `/v1` 写在 `APIRouter(prefix='/v1')` 的构造参数里，各端点自己的 `@router.post(...)` 装饰器（比如 `archive/ktransformers/server/api/openai/endpoints/chat.py` 里的 `/chat/completions`）只写相对路径。`_lab/api_surface.py` 的抽取逻辑如果只看单个路由函数上的字面路径字符串，永远拼不出完整 URL——这不是这个引擎的问题，是**静态分析工具本身有盲区**，越是"prefix 层层嵌套"的 FastAPI 项目越容易在这类脚本上产生假阴性。

同理，`archive/ktransformers/server/api/ollama/completions.py:18` 的 `prefix='/api'` 说明它的 Ollama 兼容端点实际路径是 `/api/generate`、`/api/chat`、`/api/tags`、`/api/show`，同时兼容 OpenAI 与 Ollama 两套协议，正对应它面向的是本地单机用户（Ollama 生态的典型使用场景）而不是需要标准化 API 网关的集群场景。

**`archive/` 目录不是"历史存档"意义上的死代码，是"上一代仍完整可读、但不再被任何新代码 import"的活体化石。** 它占了这个仓库 60% 的行数，`grep -rn "import ktransformers" kt-kernel doc --include="*.py"` 在 `kt-kernel/` 下确实**零命中**（唯一命中在 `doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md` 这个文档示例里，不是产品代码）——说明这次拆分是真的做了断链，不是挂了个新皮肤、内部还偷偷 import 老包。

但与此同时，`README.md:159` 说明它"for reference"被保留，且 `doc/en/*.md` 的多篇模型接入教程仍然写着"from ktransformers root"这种指向老框架用法的措辞（`doc/en/Kimi-K2.5.md:38` 等），**新旧两条文档路径目前混在一起，读者很容易读到一半从新架构的教程点进老架构的教程**。

**`third_party/llamafile` 不在 `.gitmodules` 里，是被直接 vendor 进仓库的普通目录（529KB，`sgemm.cpp`/`tinyblas_cpu_*.cpp` 等文件），这是"universal CPU backend"的来源。** 它是 Mozilla `llamafile` 项目里那批混合精度 GEMM kernel（`tinyblas`），不依赖 AMX/AVX512 这类新指令集，靠它撑住"没有 Sapphire Rapids 也能跑"这条兜底路径——`kt-kernel/README.md:32` 把它列为"Universal CPU (llamafile backend): Supported (using GGUF-format weights)"。

这条线索容易被忽略：CPU 侧加速不是单一技术栈，是 **AMX（最快，需要新 CPU）> AVX512（次快，覆盖面更广）> llamafile/AVX2（最慢，几乎任何 x86 都能跑）** 这样一个按硬件门槛分层的后端矩阵，`kt-kernel/README.md:110`-`116` 那张"CPU Variants Included"表把六档运行时自动探测的变体列全了。

**SFT（微调）能力经历了一次十倍量级的瘦身，而不是简单的"复制粘贴"。** `archive/kt-sft`（342 文件、105,925 行）是老的独立微调框架，本质上是 `archive/ktransformers` 那套注入机制的另一份分叉；新架构下 SFT 的落地是 `kt-kernel/python/sft/`（`__init__.py`/`amx.py`/`base.py`/`layer.py`/`lora.py` 等 18 个文件，合计约 11,675 行），只有老实现的十分之一体量。

`ktransformers.py:27` 的 `has_sft_support()` 通过 `import kt_kernel.sft` 探测这个新模块是否可用——**它不是"archive 版的精简移植"，而是围绕 `KTMoEWrapper` 同一套工厂类重新搭的**（`kt-kernel/python/experts.py` 里 `mode="sft"` 分支就是复用点）。这个数字对比本身是一条有用的信号：如果十万行能被十分之一的代码量替代还保住核心能力，说明老实现里有相当一部分复杂度不是"问题本身需要"，是"框架自身的历史包袱"。

**"YAML 注入"省掉的是"怎么替换模块"的样板代码，省不掉"每个模型家族要单独适配"的工作量。** `archive/ktransformers/operators/experts.py` 一个文件里就有 17 个以 `K` 开头的 MoE 相关类：`KQwen2MoeSparseMoeBlock`（`:770`）、`KDeepseekV2MoE`（`:874`）、`KDeepseekV3MoE`（`:972`）、`KMistralSparseMoEBlock`（`:1074`）、`KGlm4MoeMoE`（`:1868`）、`KQwen3NextSparseMoeBlockV2`（`:1974`）等——每一个都是针对某个具体模型架构（DeepSeek-V2/V3、Qwen2Moe、Qwen3Moe、Qwen3Next、Mixtral、GLM4-MoE、SmallThinker）手写的自定义 `forward`。

YAML 规则文件（`archive/ktransformers/optimize/optimize_rules/*.yaml` 下按模型分了几十份）决定"哪个模块该被替换成哪个类"，但**类本身要怎么实现，还是要为每个模型架构单独写一份**——"注入"这套机制解决的是"往模型里插入自定义实现"这个通用问题，不是"自动适配新模型架构"这个问题，两者经常被读者混为一谈。

**"85 个 CLI 开关"这个数字，和 `## 0` 那条 `/v1` 路由的坑是同一种病：AST 抽取器只认它认识的写法。** `_lab/out/api_surface.json` 里全部 85 条 `cli_flags` 无一例外来自 `archive/ktransformers/server/args.py`（用 `argparse` 的 `parser.add_argument(...)` 写法，比如该文件 `:16`-`:18` 的 `--host`/`--port`/`--api_key`）。

新架构 `kt-kernel/python/cli/` 下的旋钮全部用 `click.option(...)` 装饰器写（`kt-kernel/python/cli/commands/run.py:56`、`:62` 就是两例），`_lab/api_surface.py` 的抽取脚本目前不识别 `click` 的装饰器写法，于是**当前真正在用的 CLI 旋钮，一个都没被数进这 85 个里**。这不是"新架构旋钮更少"，是同一类工具盲区在两个不同维度上各发作了一次——遇到"某个数字看起来小得不合理"时，先怀疑抽取脚本认不认识这种语法，比先怀疑代码本身更省时间。

**浅克隆下 `third_party/` 子模块大多是空目录，容易被误读成"这些依赖不存在"。** `.gitmodules` 声明了 `third_party/llama.cpp`、`third_party/pybind11`、`third_party/custom_flashinfer`、`third_party/sglang` 四个子模块，但这份快照是 `--depth 1` 浅克隆、没有 `--recurse-submodules`，实测这四个目录大小均为 0（只有非子模块的 `third_party/llamafile` 有真实内容）。**这是取证方法论上的一个坑，不是仓库本身的问题**——脚本核对代码规模、判断"某功能是否存在"时，如果不知道这条约束，容易把"没拉取"误判成"没有依赖"。

## 8. 可改进点

1. **文档层面的新旧混淆没有清理干净。**
   - 现状：`doc/en/` 下多篇模型接入教程（如 `doc/en/DeepSeek-V4-Flash.md:114`、`doc/en/Kimi-K2.5.md:38`）仍写着"Option A: One-click install (from ktransformers root)"，读者按着走大概率会先碰到已经归档、行为可能与当前默认路径不一致的老安装脚本。
   - 建议：给这批教程统一加一条"新架构下请改用 `install.sh` + `kt-cli run`"的顶部提示，成本很低，收益是直接减少新用户在安装第一步就走错路的概率。

2. **`archive/` 与 `kt-kernel/` 之间没有任何自动化的一致性校验。**
   - 现状：两边各自维护了一份指令集探测逻辑（老代码在 `archive/ktransformers/operators/experts.py` 里手写 `elif self.backend == "AMXBF16"` 分支，新代码在 `kt-kernel/python/utils/amx.py` 里重新实现了一遍），如果未来老代码某个 bugfix 想同步到新代码（或者反过来），完全靠人工比对。
   - 建议：考虑到 `archive/` 已经声明只做参考不做维护，这个风险会随时间衰减，但短期内新旧实现分叉导致的行为不一致仍是潜在的用户困惑源，值得在 `archive/README.md` 顶部补一句"此后的 bugfix 不会回填到这里"的明确声明，断掉读者"这两边应该保持同步"的错误预期。

3. **`max_deferred_experts_per_token` 这个延迟隐藏参数缺一份量化过准确度影响的文档。**
   - 现状：`kt-kernel/python/experts_base.py:248` 的 docstring 只说"Number of experts per token to defer",没有给出"推迟一层计算"对生成质量的影响范围（哪怕是一个粗略的困惑度对比区间）。
   - 建议：用户想调这个参数换吞吐时，缺一份"这样做要付多少准确度代价"的参考数据——哪怕只是在几个基准模型上跑一次困惑度对比，也比现在的"纯靠猜"要好。

4. **对 `sglang-kt` fork 与官方 SGLang 的版本追赶策略没有公开说明。**
   - 现状：`kt-kernel/README.md:280` 只警告用户"别装错包",没有说明这份 fork 落后官方多少个版本、是否有定期 rebase 的计划。
   - 建议：这对准备长期依赖它的生产用户是一个不透明的风险点，补一份"fork 与上游的同步节奏"说明（哪怕只是"每月 rebase 一次"这种粗粒度承诺）能显著降低选型时的不确定性。

5. **浅克隆场景下的依赖可见性没有兜底提示。**
   - 现状：`install.sh:107`-`110` 在 `third_party/sglang` 缺失（子模块没拉）时会给出明确的 `log_error`，这一点做得对；但项目文档里没有一处提醒"如果你是从 zip 包或非 `--recursive` 方式拿到源码，`third_party/` 下四个目录会是空的"。
   - 建议：对刚接触这个仓库、习惯直接下载 zip 的用户，这条约束容易在实际踩坑之前完全没有预警，在 README 的安装章节最前面加一句"必须 `git clone --recursive`"能省掉不少 issue。

## 9. 自测题与延伸阅读

**自测题**

1. `pip install ktransformers` 现在实际装的是什么？它和 `kt-kernel`、`sglang-kt` 三者的依赖关系是怎样的？（提示：`pyproject.toml:24`、`setup.py:17`-`18`）
2. 旧架构的 YAML 注入规则里，一条 `match` 规则同时给了 `class` 和 `name` 两个匹配条件时，`gen_optimize_config` 是"两者都满足才算命中"还是"任一满足即可"？回到 `archive/ktransformers/optimize/optimize.py:79`-`87` 看代码逻辑再回答。
3. 为什么 `_lab/out/api_surface.json` 抽出的 32 条路由里 `/v1/*` 命中数是 0，而这不等于这个引擎不支持 `/v1` 前缀？完整解释这个假阴性是怎么产生的。
4. 新架构下，"哪些专家放 GPU"这个决定是在什么时候、依据什么数据做出的？和旧架构 YAML 规则里的静态 `generate_device: "cpu"` 相比，这种方式的优势与代价分别是什么？
5. `submit_forward` / `sync_forward` 这一对分离的方法解决的是什么问题？如果把它们合并成一个同步阻塞的 `forward` 调用，端到端延迟会怎样变化？
6. KTransformers 新架构里，"连续批处理"这件事发生在哪个代码库里？为什么说 `archive/csrc/balance_serve/sched/scheduler.cpp` 里的 `FCFS` 实现现在已经不是这个引擎调度行为的真实来源？
7. `archive/kt-sft`（约 10.6 万行）和 `kt-kernel/python/sft/`（约 1.2 万行）都实现了 SFT 能力，两者是什么关系？为什么行数差了近十倍还能覆盖大体相同的功能？
8. 如果你只是把这份仓库下载成 zip（而不是 `git clone --recursive`），`third_party/sglang` 目录会是什么状态？这对你直接读代码理解"调度到底在哪实现"会造成什么误导？

**延伸阅读**

- [[00-总览与阅读地图]] —— 全库的证据标准与已查实硬结论汇总
- [[01-vLLM-全景与代码地图]] —— 对照组：一个假设"显存够用、要伺候更多并发"的引擎长什么样
- [[11-开源推理引擎谱系图]] —— 把 KTransformers 放进全部引擎的谱系里看它的定位
