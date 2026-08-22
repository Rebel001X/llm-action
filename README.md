# 开源推理引擎深度研究库

对 **vLLM / SGLang 及另外 10 个开源推理引擎**做源码级解剖。本分支只含这一个研究库，
**不含 llm-action 主库的历史**——刻意如此，见下文「为什么是孤立分支」。

内容在 [`VLLM-SGlang-研究-推进/`](VLLM-SGlang-研究-推进/)，先读那里的 `_PLAN.md`。

## 这个库跟别的"推理引擎介绍"有什么不同

一句话：**每条代码断言都能回到一行真实源码，且有脚本逐条核对。**

- 正文里的源码引用一律写成 `` `vllm/v1/core/sched/scheduler.py:123` ``；
- `_verify.py` 会**打开那个文件、数到那一行**，行号越界直接 FAIL；
- 每篇开头声明「取证基准：`<engine>` @ `<sha>`」，与实际 clone 的 commit 不符也 FAIL。

所以这里不会出现"我记得 vLLM 大概是这样"的段落。查不到的就写「未查证」。

## 确定性取证层 `_lab/`

正文里的统计数字不是心算的，是脚本从真源码算出来再落进 `_lab/out/*.json` 的：

| 脚本 | 算什么 |
|---|---|
| `repo_stats.py` | 语言构成、目录规模、CUDA/Triton 核数、测试与产品码比 |
| `api_surface.py` | **AST** 抽 HTTP 路由 / pydantic 协议类 / 配置对象 / argparse 开关 |
| `struct_map.py` | 按 10 个子系统定位文件与关键类（带行号） |
| `compare.py` | 跨引擎集合运算：路由差、chat 字段差、同名旋钮默认值差 |

`out/*.json` 入库（可审计），`_src/` 下 clone 的上游源码不入库（体积 + 各家 license）。
复现方式见 `_PLAN.md` §2，clone 到同一个 sha 即可逐条复算。

```bash
cd VLLM-SGlang-研究-推进/_lab
python repo_stats.py && python api_surface.py && python struct_map.py && python compare.py
python -m pytest tests -q          # 14 项，含不依赖 _src 的解析逻辑自检
cd .. && python _verify.py         # 全库体检：取证基准 + 行号 + 双链 + 占位符
```

## 为什么是孤立分支

上游 `liguodongiot/llm-action` 的 2024 年历史里含一个 **Hugging Face User Access Token**
（`llm-localization/ascend/baichuan2.md` 与 `llm下载.md`），GitHub 推送保护会拦截任何
携带那段历史的 push。本分支不带上游历史，因此不受影响，也不会把那个 token 再传播一次。
**如果那是真实有效的 token，应当去 Hugging Face 吊销它** —— 从 git 历史里删掉并不等于失效。

## 覆盖的引擎

vLLM、SGLang、TensorRT-LLM、LMDeploy、TGI、LightLLM、NVIDIA Dynamo、
KTransformers、Mooncake、llama.cpp、MLC-LLM、Tokasaurus。
