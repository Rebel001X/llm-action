# _PLAN.md —— 开源推理引擎深度研究库·施工蓝图

> 这份文件是本库的**契约**。写任何一篇正文前先读它；改结构也先改它。
> 续写入口：读本文件 → 挑 ⬜ → 写 → 跑 `python _verify.py` + `cd _lab && python -m pytest tests -q`。

---

## 0. 这个库要回答什么

一句话：**把 vLLM / SGLang 以及其他主流开源推理引擎，从"听说它很快"降落到"它第几行代码在做什么、这个设计的代价是什么、哪里还能改"。**

三条边界，划清楚免得和已有库打架：

| 库 | 讲什么 | 和本库的边界 |
|---|---|---|
| [[vllm-anatomy-kb]] 对应的 `vLLM解剖-高吞吐推理系统/` | 精读 Aleksa Gordić 一篇长文（2025-08 的 vLLM） | 那是**读一篇文章**；本库是**读源码**，且横跨多引擎 |
| `attention-optimization/` | FlashAttention / PagedAttention 的**数学与算子** | 那讲"算子为什么快"；本库讲"引擎怎么把算子编排成服务" |
| `llm-inference/` | 各推理技术的概念笔记 | 那是概念索引；本库是带行号的源码级解剖 |

本库的增量是三样东西：**真源码**（`_src/` 里 clone 的 12 个仓库）、**确定性抽取**（`_lab/` 把 API/结构/规模算成 JSON）、**可改进点**（读完之后能提得出 PR 的清单）。

---

## 1. 诚实标准（本库红线，违反即回炉）

1. **每条代码断言必须能回到一行真实源码。** 正文里引用源码一律写成行内代码 `` `vllm/v1/core/sched/scheduler.py:123` ``（或跨引擎 `` `sglang:python/sglang/srt/managers/scheduler.py:456` ``）。`_verify.py` 会逐条打开文件核行号，**越界即报错**。
2. **每篇开头必须声明取证基准**：
   `> **本篇取证基准**：`vllm` @ `<sha8>`（<commit 日期>）`
   sha 与 `_lab/out/*.json` 里记录的 clone sha 不一致，`_verify.py` 直接 FAIL。**这是为了防"拿 A 版本代码讲 B 版本行号"。**
3. **区分证据等级，三种标注不许混**：
   - 「源码为证」——读了代码，带 `文件:行`；
   - 「文档所述」——上游 README/docs 的说法，注明出处文件；
   - 「本库推断」——我的判断，必须显式标注，且给出推断依据。
   查不到就写「未查证」，**不许拿印象补**。
4. **性能数字必须带口径。** 本机无 GPU，本库**不产出任何实测吞吐/延迟数字**。凡引用上游 benchmark，必须同时给出：硬件、模型、输入输出长度分布、并发、版本。做不到就不写这个数字。
5. **每条设计结论配三样**（沿用 [[llm-training-pipeline-kb]] 的铁律）：**为什么这么设计 / 不这样会怎样 / 什么时候可以不这样**。只夸不说代价的段落一律回炉。
6. **不编 URL、不编 PR 号、不编 issue 号。** 要引用就引用 `_src/` 里真实存在的文件。

---

## 2. 取证基准（`_src/` 快照）

源码用 `--depth 1` clone 到 `_src/`（已 gitignore，不入库）。复现命令：

```bash
mkdir -p _src && cd _src
git clone --depth 1 --no-tags --single-branch -c core.longpaths=true https://github.com/vllm-project/vllm.git vllm
git clone --depth 1 --no-tags --single-branch -c core.longpaths=true https://github.com/sgl-project/sglang.git sglang
# 其余引擎见 _lab/common.py 的 ENGINES 表
```

> Windows 踩坑：不加 `-c core.longpaths=true` 会在 SGLang 的
> `python/sglang/srt/layers/moe/.../E=161,N=192,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json`
> 上以 `Filename too long` 失败 —— 这不是网络问题，别去重试网络。

各引擎的实际 sha 与日期以 `_lab/out/repo_stats.json` 的 `ref` 字段为准，**写正文时从那里抄**。

---

## 3. `_lab/` 确定性层

| 脚本 | 干什么 | 产物 |
|---|---|---|
| `common.py` | 引擎登记表、遍历、落盘 | — |
| `repo_stats.py` | 语言构成、目录规模、CUDA/Triton 核数、测试占比 | `out/repo_stats.json` |
| `api_surface.py` | **AST** 抽 HTTP 路由 / pydantic 协议类 / 配置对象 / argparse 开关 | `out/api_surface.json` |
| `struct_map.py` | 按 10 个子系统定位文件与关键类（带行号） | `out/struct_map.json` |
| `compare.py` | 跨引擎集合运算：路由差、chat 字段差、同名旋钮默认值差 | `out/compare.json` `out/compare.md` |

规矩：
- 每个脚本都有 `--selftest`，**用临时目录造假仓库自检解析逻辑**，不依赖 `_src` 也能验对错；
- `out/*.json` **入库**（可审计），`_src/` 不入库；
- 正文里任何统计数字，必须能在 `out/*.json` 里找到同一个数 —— 写的时候从 JSON 抄，别心算。

跑法：
```bash
cd _lab
python repo_stats.py && python api_surface.py && python struct_map.py && python compare.py
python -m pytest tests -q
```

---

## 4. 篇目表

图例：⬜ 未写 / 🟨 进行中 / ✅ 已完成

### 00 总览
- ✅ `00-总览与阅读地图.md` —— **亲笔**，不许 agent 代写

### 01-vLLM/（取证基准 `vllm`）
- ✅ `01-vLLM-全景与代码地图.md`
- ✅ `02-vLLM-V1架构与EngineCore循环.md`
- ✅ `03-vLLM-调度器解剖.md`
- ✅ `04-vLLM-KV缓存与前缀缓存.md`
- ✅ `05-vLLM-注意力后端与算子层.md`
- ✅ `06-vLLM-模型执行与CUDA-Graph.md`
- ✅ `07-vLLM-分布式与并行策略.md`
- ✅ `08-vLLM-HTTP-API表面全解.md`
- ✅ `09-vLLM-Python-API与EngineArgs.md`
- ✅ `10-vLLM-投机解码.md`
- ⬜ `11-vLLM-结构化输出.md`
- ⬜ `12-vLLM-PD分离与KV-Connector.md`

### 02-SGLang/（取证基准 `sglang`）
- ✅ `01-SGLang-全景与代码地图.md`
- ✅ `02-SGLang-Scheduler事件循环.md`
- ✅ `03-SGLang-RadixAttention与前缀缓存.md`
- ✅ `04-SGLang-内存池与KV布局.md`
- ✅ `05-SGLang-注意力后端矩阵.md`
- ✅ `06-SGLang-约束解码与语法后端.md`
- ✅ `07-SGLang-前端DSL与编程模型.md`
- ✅ `08-SGLang-HTTP-API表面全解.md`
- ✅ `09-SGLang-ServerArgs旋钮全景.md`
- ✅ `10-SGLang-投机解码EAGLE.md`
- ✅ `11-SGLang-PD分离与HiCache分层.md`

### 03-其他引擎/
- ✅ `01-TensorRT-LLM.md`
- ⬜ `02-LMDeploy与TurboMind.md`
- ⬜ `03-TGI.md`
- ⬜ `04-LightLLM.md`
- ⬜ `05-llama.cpp-server.md`
- ⬜ `06-MLC-LLM.md`
- ⬜ `07-KTransformers.md`
- ⬜ `08-Mooncake传输引擎.md`
- ⬜ `09-NVIDIA-Dynamo.md`
- ⬜ `10-Tokasaurus.md`
- ⬜ `11-开源推理引擎谱系图.md`

### 04-横向对比/
- ⬜ `01-API兼容性横向对比.md`
- ⬜ `02-调度策略横向对比.md`
- ⬜ `03-KV缓存与前缀复用横向对比.md`
- ⬜ `04-工程规模与代码结构对比.md`
- ⬜ `05-选型决策树.md`
- ⬜ `06-性能口径与基准陷阱.md`

### 05-改进机会/
- ⬜ `01-vLLM可改进点.md`
- ⬜ `02-SGLang可改进点.md`
- ⬜ `03-跨引擎共性缺口.md`
- ⬜ `04-可落地贡献清单.md`

### 99 收束
- ✅ `99-本质总结-亲笔.md` —— **亲笔**，不许 agent 代写

---

## 5. 每篇的固定骨架（十段式）

```markdown
# <标题>

> **本篇取证基准**：`<engine>` @ `<sha8>`（<日期>）
> **一句话**：<20 字以内的结论>

## 0. 结论先行
## 1. 它在系统里的位置
## 2. 代码地图（文件 → 职责，带行号）
## 3. 核心数据结构
## 4. 主流程走读
## 5. 设计决策与代价（为什么这样 / 不这样会怎样 / 什么时候可以不这样）
## 6. 同位对照（另一个引擎在同一位置怎么做）
## 7. 踩坑与反直觉
## 8. 可改进点
## 9. 自测题与延伸阅读
```

硬指标（沿用 [[km-best-practices-kb]] 第 18 篇）：
- `## 2` 至少 6 条 `文件:行` 引用；
- `## 5` 每条决策三样齐全；
- `## 9` 至少 5 道闭卷自测题 + 3 条本库双链；
- 一事一文，别把两个子系统塞一篇。

---

## 6. 双链名册（写 `[[...]]` 只许从这里挑）

`00-总览与阅读地图`、`99-本质总结-亲笔`、`_PLAN`
以及第 4 节篇目表里所有文件名（去掉 `.md` 与目录前缀，例如 `[[03-vLLM-调度器解剖]]`）。

**不在名册里的名字一律不许写成双链** —— `_verify.py` 会判死链。

---

## 7. 已知的施工踩坑

1. **`.ps1` 脚本必须存成带 BOM 的 UTF-8**：否则 `powershell.exe -File` 按 GBK 读，路径里的中文 `研究-推进` 会变成 `鐮旂┒-鎺ㄨ繘`，于是**在旁边悄悄建出一个乱码目录**，clone 全落到那里，主目录看起来"什么都没发生"。这一条已经踩过一次。
2. **`git clone` 走 sandbox 会 DNS 失败**（`Could not resolve host`），网络操作要脱 sandbox 跑。
3. **vLLM 的 entrypoints 已经拆包**：旧的 `vllm/entrypoints/openai/protocol.py` 单文件不存在了，改成 `vllm/entrypoints/<family>/<endpoint>/{api_router,protocol,serving}.py`。抽 API 的 glob 必须跟着改，否则会得到「0 条路由」这种**假的空结果**——空结果要当 bug 查，不许当事实写。
4. **多 agent 并发别一次起太多**：按 6~8 个一批，跑完按磁盘审计（`ls` 数文件）再起下一批，别信汇报。
5. **`git -C <dir> <cmd>` 在 `<dir>` 不是仓库时会向上找父仓库** —— 这条踩得最重。
   `git clone` 失败后目录是空的，紧跟着的
   `git -C _src/tensorrt-llm sparse-checkout set ...` **落到了 `llm-action` 主仓库上**，
   给整个主仓库开了 sparse-checkout，清掉了所有未被 pattern 覆盖且无本地改动的已跟踪文件。
   救回来的办法是 `git sparse-checkout disable` + 删掉 `.git/info/sparse-checkout`，
   已跟踪文件全部复原（`git ls-files --deleted` 为 0）；
   **但用户此前本地删除的十来个文件被一并"复活"了，这个副作用无法精确回滚。**
   以后规矩：对 `_src/` 下任何仓库跑 git 之前，先断言它自己是仓库根 ——
   `git -C <dir> rev-parse --show-toplevel` 的输出必须等于 `<dir>` 本身，不等就中止。
6. **大仓库（TensorRT-LLM ~300MB+ zipball）一次性 `r.read()` 会 IncompleteRead**，
   必须分块流式写盘 + 失败重试，别把整个 zip 读进内存。
