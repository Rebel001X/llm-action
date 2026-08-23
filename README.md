# 推理引擎：读懂它，然后造一个

这个分支放两个互补的库，**不含 llm-action 主库的历史** —— 刻意如此，见下文「为什么是孤立分支」。

| 库 | 回答什么 | 红线 |
|---|---|---|
| [`VLLM-SGlang-研究-推进/`](VLLM-SGlang-研究-推进/) | **读代码**：12 个开源推理引擎是怎么写的 | 每条断言回到**别人源码**的一行 |
| [`从0实现推理引擎/`](从0实现推理引擎/) | **写代码**：从空文件搭一个引擎要懂什么、用什么语言 | 每个原理有**本库自己能跑的代码** |

两个库各自的入口都是它目录下的 `_PLAN.md`。

---

## 一、开源推理引擎深度研究库

对 **vLLM / SGLang 及另外 10 个开源推理引擎**做源码级解剖。

**它跟别的"推理引擎介绍"有什么不同**：每条代码断言都能回到一行真实源码，且有脚本逐条核对。

- 正文里的源码引用一律写成 `` `vllm/v1/core/sched/scheduler.py:123` ``；
- `_verify.py` 会**打开那个文件、数到那一行**，行号越界直接 FAIL；
- 每篇开头声明「取证基准：`<engine>` @ `<sha>`」，与实际 clone 的 commit 不符也 FAIL。

所以这里不会出现"我记得 vLLM 大概是这样"的段落。查不到的就写「未查证」。

### 确定性取证层 `_lab/`

正文里的统计数字不是心算的，是脚本从真源码算出来再落进 `_lab/out/*.json` 的：

| 脚本 | 算什么 |
|---|---|
| `repo_stats.py` | 语言构成、目录规模、CUDA/Triton 核数、测试与产品码比 |
| `api_surface.py` | **AST** 抽 HTTP 路由 / pydantic 协议类 / 配置对象 / argparse 开关 |
| `struct_map.py` | 按 10 个子系统定位文件与关键类（带行号） |
| `compare.py` | 跨引擎集合运算：路由差、chat 字段差、同名旋钮默认值差 |

`out/*.json` 入库（可审计），`_src/` 下 clone 的上游源码不入库（体积 + 各家 license）。
复现方式见该库 `_PLAN.md` §2，clone 到同一个 sha 即可逐条复算。

```bash
cd VLLM-SGlang-研究-推进/_lab
python repo_stats.py && python api_surface.py && python struct_map.py && python compare.py
python -m pytest tests -q          # 含不依赖 _src 的解析逻辑自检
cd .. && python _verify.py         # 全库体检：取证基准 + 行号 + 双链 + 占位符
```

### 覆盖的引擎

vLLM、SGLang、TensorRT-LLM、LMDeploy、TGI、LightLLM、NVIDIA Dynamo、
KTransformers、Mooncake、llama.cpp、MLC-LLM、Tokasaurus。

---

## 二、从 0 实现推理引擎（教学库）

回答三个问题：**从零搭一个引擎要懂哪些原理、按什么顺序搭、用什么语言写。**

**它跟别的教程有什么不同**：每个原理都配一段**能跑的代码**，
而且有测试**证明这个优化没有改变输出**。只有文字说明、没有可执行验证的原理，一律不写。

### 能跑的最小引擎 `_lab/`

纯 numpy + 标准库，**不装 torch、不装 web 框架、不需要 GPU**：

| 文件 | 是什么 |
|---|---|
| `minigpt.py` | 最小 Transformer：朴素前向 / KV 缓存前向 / 生成 / **算术强度解析模型** |
| `paged.py` | 分页 KV：块表、分配器、引用计数、碎片统计 |
| `engine.py` | 调度器 + 连续批处理 + 前缀缓存 |
| `sampling.py` | 采样、停止条件、**流式输出的状态机** |
| `serve.py` | HTTP 服务：两道队列解耦并发 HTTP 与单线程引擎；SSE、背压、断连回收 |
| `flashattn.py` | 在线 softmax + 分块注意力 + decode 的 split-KV |
| `parallel.py` | TP / PP / EP —— 等价性与代价都能在单机验证 |
| `spec_decode.py` | 投机解码 + 20 万次采样的分布检验 |
| `structured.py` | 语法约束解码：状态×词表掩码索引 + jump-forward |
| `quantize.py` | 三档量化的误差、实际位宽、量权重还是量 KV 的访存账 |

```bash
cd 从0实现推理引擎/_lab
python minigpt.py --selftest && python minigpt.py    # 后者打印算术强度表
python -m pytest tests -q                            # 119 项
cd .. && python _verify.py                           # 全库体检
```

### 测试判据分四套 —— 这个区分本身就是内容

| 判据 | 用在哪 | 抓不到什么 |
|---|---|---|
| **输出恒等**（算两遍逐位比） | KV 缓存 / 分页 / 批处理 / 前缀缓存 / FlashAttention / TP | **两侧同时犯的错** |
| **分布恒等**（大量采样比全变差） | 投机解码 | 低于采样噪声的偏差 |
| **约束恒成立** | 结构化输出（它**故意**改变分布） | 分布层面的偏移 |
| **误差有界且单调** | 量化（它本来就是有损的） | 特定输入上的灾难性失效 |

「输出恒等」是主力，但它有一个**结构性盲区**：
把 4 处残差 `x = x + ...` 一起写成 `x = ...` 时，两条前向路径同时被改到，
**25 条差分断言全绿，而模型输出的 argmax 已从 36 变成 44**。
解药是 `tests/test_invariants.py` —— **不比"实现和实现"，比"实现和它必须满足的性质"**。
完整复现见 `05-工程与陷阱/01-正确性怎么保证.md`，那一篇还记了另外五次自我证伪。

### 关于性能数字

写这个库的机器**没有 GPU**。所以：

- **可以有**：本库玩具引擎在 CPU 上的实测（它是我们自己写的，测它天经地义），
  以及解析成本模型算出来的比值 —— 但每次出现都标注了适用条件；
- **绝不许有**：把玩具数字外推到 vLLM/SGLang，或凭空给真实引擎的吞吐/延迟。

---

## 为什么是孤立分支

上游 `liguodongiot/llm-action` 的 2024 年历史里含一个 **Hugging Face User Access Token**
（`llm-localization/ascend/baichuan2.md` 与 `llm下载.md`），GitHub 推送保护会拦截任何
携带那段历史的 push。本分支不带上游历史，因此不受影响，也不会把那个 token 再传播一次。
**如果那是真实有效的 token，应当去 Hugging Face 吊销它** —— 从 git 历史里删掉并不等于失效。
