# Rust在推理栈里到底占了什么位置

> **一句话**：12 个引擎没有一个纯 Rust 写，但 4 个不约而同把它放进了同一段位置。
> **前置**：先读 `01-用什么语言写推理引擎-12个真实引擎的证据.md` 建立语言构成的事实基础；再读 `02-Python为什么还没被淘汰.md`，本篇会反复借用它的 `T_gpu`/`T_py` 判据框架，但用在一个不同的边界上。

## 0. 这一篇要解决什么问题

用户问"该用 Rust 吗"，背后往往藏着一个没说出口的预设：Rust 内存安全、没有 GC、比 Python 快、比 C++ 少踩坑，是不是应该整个引擎都用它写？

把 12 个真实开源推理引擎摆在一起看，答案是两句话：

> **12 个主流开源推理引擎里，没有一个是「用 Rust 写的」。**
> **但有 4 个在特定层用了 Rust，而且位置高度一致。**

第一句话在 `01-用什么语言写推理引擎-12个真实引擎的证据.md` 已经用统计事实立住了（`_PLAN.md:121-125`）：10 个引擎主语言是 Python，2 个是 C++，Rust 从没当过任何一个引擎的主语言。第二句话是本篇要单独展开的——Dynamo 27%、SGLang 6%、vLLM 5%、TGI 12%（`_PLAN.md:110,111,116` 及 §3.2 全表），乍看行数占比差异巨大，从 5% 到 27% 差了五倍多，但只要真的翻开这些 Rust 代码在做什么，会发现一件更有意思的事：**它们几乎都在同一段位置**——HTTP 服务、请求路由、批处理队列、跨机 KV 路由。没有一行是在算 QKV 投影、算 attention、算 FFN。

本篇要搞清楚三件事：

1. **Rust 到底被放在哪一层**——四个项目逐个开，带真实行号；
2. **为什么是那一层，不是别的层**——用 CPU 密集 / 高并发 / 不碰模型生态三条共性解释；
3. **那一层有什么共性**——尤其是 SGLang 内部两种完全不同的 Rust 集成形态，一个是进程内扩展，一个是独立进程，看懂这两者的区别，比记住"SGLang Rust 占 6%"这个数字有用得多。

最后落到读者真正关心的问题：**如果你从 0 写一个推理引擎，什么时候该用 Rust、什么时候不该。**

## 1. 先看现象（可跑的代码/可算的数）

### 1.1 四个数字，四个不同的行数占比

`_PLAN.md` §3.2 的统计（数据源 `../VLLM-SGlang-研究-推进/_lab/out/repo_stats.json`）：

| 引擎 | Rust 占比 | Rust 所在目录（本篇实测，非 `_PLAN.md` 原文） |
|---|---:|---|
| Dynamo | 27%（`_PLAN.md:110`） | `lib/kv-router/`、`lib/llm/src/http/service/` 等——集群路由与 HTTP 前端 |
| TGI | 12%（`_PLAN.md:116`） | `backends/v2/`、`backends/v3/`、`router/`——请求路由与连续批处理队列 |
| SGLang | 6%（`_PLAN.md:109`） | `python/sglang/srt/managers/rust_server.py` 挂进的 PyO3 扩展 **+** `sgl-model-gateway/`、`experimental/sgl-router/` 两个独立进程——**两种完全不同的东西加在一起的数字** |
| vLLM | 5%（`_PLAN.md:111`） | `rust/`——独立的 HTTP 前端可执行文件 |

这张表提出一个问题：占比从 5% 到 27%，差了五倍多，是不是意味着这四家对"Rust 该管多少"有五倍的分歧？第 4 节会给出答案：**不是**。差异主要来自"这个项目本身有多大"和"这个 Rust 组件覆盖了几种功能"，而不是"Rust 被赋予了多不同的职责"。Dynamo 占比最高，是因为 Dynamo **整个产品定位就是这一层**（详见 4.4）；vLLM/SGLang 占比最低，是因为 Rust 只是这两个巨型 Python 引擎里加装的一个可选组件。

### 1.2 一个更有信息量的现象：四处代码的共同边界

先把结论摆出来，第 4 节逐条给源码证据：

- **vLLM 的 `rust/`**：一个独立可执行文件（`vllm-rs`），自己起 HTTP 服务，但**不跑模型前向**——它通过 ZMQ 把请求转发给一个独立的 Python "headless" 引擎子进程。
- **SGLang 的 `rust_server.py`**：**不是独立进程**，是几条运行在 Python scheduler 进程内部的 Rust 线程（PyO3 扩展），替换的是 Python 的 api-server + TokenizerManager + DetokenizerManager 这一段。
- **SGLang 的 `sgl-model-gateway`**：又是一个**独立进程**，管的是"一批 SGLang worker 之间怎么分流量"，跟上面那条完全是两回事。
- **TGI 的 `backends/*/src/queue.rs`**：Rust 负责连续批处理的排队与批次组装决策，通过 gRPC 跟一个只管前向计算的 Python/PyTorch 分片进程通信。
- **Dynamo 的 `lib/kv-router/`**：一棵纯 CPU 的 radix tree，给每个新请求算"哪个 worker 的 KV 缓存跟这个请求重叠最多"，本身完全不碰模型计算。

五处代码，五种具体实现，但拆开看职责，共同点已经很明显：**全部发生在"请求从网络进来"到"交给某个跑模型的进程/GPU 之前"这一段**。没有一处越过这条线去做矩阵乘。

## 2. 原理

### 2.1 为什么是这一段，不是别的段

`02-Python为什么还没被淘汰.md` 建立过一条判据（该篇 §1.3）：

```
T_gpu：GPU 执行一步 decode 所需时间
T_py： Python 侧为下一步做准备所需时间（调度决策 + 张量组织 + kernel 提交 + 采样 + 簿记）

T_py < T_gpu → Python 被 GPU 的异步执行掩盖，几乎免费
T_py > T_gpu → Python 变成新瓶颈
```

本篇讨论的 Rust 所在层，跟这条判据里的 `T_py` 是**同一类问题的更上游版本**，但有一个关键区别必须先讲清楚：`02-Python为什么还没被淘汰.md` 里的 `T_py` 描述的是"已经知道要跑哪个模型、已经建立好一个引擎进程之后，每一步 decode 循环里的编排开销"——这部分开销天然跟 GPU 计算**并行**发生，只要发指令的速度跟得上 GPU 消费指令的速度，它就能被掩盖（该篇 §2.1）。

而 Rust 出现的这一层——HTTP 请求解析、跨实例路由、批处理队列——往往发生在**更早的一步**：这个请求还没有被切进任何一个 batch、还没有对应的 CUDA kernel 排进任何队列，甚至可能还不知道该发给哪台机器上的哪个引擎实例。这一步没有 GPU 在旁边替它打掩护——它面对的不是"已经组织好的张量"，而是原始的 HTTP 连接、JSON/msgpack 字节流、一大堆待排序的请求对象。

三条共性，决定了为什么恰好是这一层：

**(a) CPU 密集，且没有 GPU 可以掩盖它**

字符串解析（HTTP header、JSON/msgpack 解码）、哈希查找（前缀匹配、KV 索引）、堆/队列操作（排队、抢占排序）——这些全是纯 CPU 上的数据结构操作，跟"这次前向要不要用 GPU"无关。`02-Python为什么还没被淘汰.md` 的整套论证成立的前提是"CPU 侧发指令的时间能被 GPU 侧算数的时间盖住"；但网关/路由层没有对应的"GPU 算数时间"可以借——它服务的可能是多台机器上的多个引擎实例，每台机器各自忙各自的，路由层的 CPU 开销是**真实的串行瓶颈**，不是被谁掩盖的影子开销。

**(b) 高并发，是 Rust/Tokio 这类运行时的舒适区**

网关要同时扛住成千上万条 TCP 连接（尤其是流式 SSE 响应，连接要保持很久），这是异步运行时的经典场景。Python 的 `asyncio` 也能做类似的事，但仍然要受 GIL 和解释器对象开销的天花板限制（`02-Python为什么还没被淘汰.md` §2.3a）——单进程内所有并发连接的字节码执行仍然是串行的。Rust 的异步运行时（Tokio）没有这个限制：多个连接可以真正跑在不同 OS 线程上并行处理，语言本身不需要一把全局锁。

**(c) 不需要碰模型生态**

这是最容易被忽略的一条，但恰恰是"为什么不是 C++、也不是继续用 Python"的关键。模型执行层背后压着两块巨大的存量资产：权重格式/config/tokenizer 生态是 Python（HuggingFace `transformers`、safetensors），算子生态是 CUDA/C++（cuBLAS、FlashAttention、Triton）。这两块地盘各自已经被打磨了多年，谁都不会重新发明一遍。**但请求路由/编排这一层是空地**——它不需要 import `transformers`，不需要理解权重张量的 shape，只需要理解"这是一个 HTTP 请求，该转给谁"。Rust 在这块空地上落子，不用跟任何一方的存量资产竞争，这是它能进来的根本原因。

### 2.2 为什么从不出现在模型执行层

反过来问：为什么没有一个引擎把前向计算本体也用 Rust 重写？

答案就在 2.1(c) 反过来看：模型执行层的两块地盘都不是空的。Rust 要进去，只有两条路：

- **调用 CUDA**：需要写 FFI 胶水去绑定 cuBLAS/cuDNN 这些 C 接口。一旦通过 FFI 调用外部 C 代码，Rust 的内存安全保证在这段边界内**不再生效**——`unsafe` 块里发生的显存越界、生命周期错误，Rust 编译器管不到。到这一步，Rust 相对 C++ 的核心卖点（编译期内存安全）已经打了折扣，工作量却要多绕一层 FFI 胶水，性价比不明显。
- **另起一套算子库**：不依赖 CUDA 生态，自己写一遍矩阵乘、attention、量化算子。工程量要对齐 cuBLAS/FlashAttention 这些被数百人年打磨过的库，对任何团队都是不现实的投入。

这不是抽象论证——是 12 个真实项目共同做出的选择：0 个例外。llama.cpp 走的是纯 CPU/轻量 GPU 这条路，但它选的是 C++ 不是 Rust（`03-C++路线-llama.cpp的取舍.md`），提示"要不要碰模型执行层"这条边界的取舍，跟 Rust 的内存安全卖点关系不大，真正决定因素是**能不能接上现成的算子库和编译器后端**——这条路 C++ 和 CUDA 生态天然更近。

### 2.3 编译产物的形态，本身就是一条可读的边界信号

第 4 节要逐个打开四个项目的源码，这里先说一个贯穿全篇的读代码技巧：**不用等看懂业务逻辑，光看 Rust 代码编译成什么产物，就能大致猜出它在架构里扮演的角色。**

- 编译成 **`cdylib`**（动态链接库）、被 Python `import` 进同一个进程——这是"进程内扩展"，说明这段 Rust 代码是在**替换**某个 Python 组件本来在同一进程里干的事，通常伴随一个默认关闭的开关（4.2 节形态 A）。
- 编译成独立的 **`bin`**（可执行文件），自己起监听端口——这是"独立进程"，说明这段 Rust 代码是在给一批**已经存在、各自独立运行**的引擎实例做前面那道关口，进程边界清楚，可以单独部署、单独扩缩容（4.1 节 vLLM、4.2 节形态 B、4.4 节 Dynamo）。
- 只提供一个 **`rlib`**（供其他 Rust crate 静态链接的库），自己不直接运行——通常是被上面两种形态复用的公共逻辑层（比如 TGI 的 `router` crate 被 `backends/v2`、`backends/v3` 共享）。

这条读法在第 4 节会反复用到：**光凭 `Cargo.toml` 里 `lib` 段和 `bin` 段（TOML 数组表语法，两层方括号）分别怎么写，往往就能猜对这段 Rust 代码是不是"在同一个进程里替换 Python"，不需要先读完整个调度逻辑。**下一节给出能自己验证的具体命令。

## 3. 自己动手：数各项目 Rust 代码的分布

本篇所有行号引用都对着 `../VLLM-SGlang-研究-推进/_src/` 下各引擎的真实 clone 核对过。想自己验证，不需要额外装什么，用 shell 内建工具就够：

```bash
cd ../VLLM-SGlang-研究-推进/_src   # 研究库的引擎源码 clone 目录

# 数每个引擎的 Rust 代码总行数（近似——不排除生成代码/vendor代码，
# 想要精确口径请用 cloc 或研究库自己的 repo_stats.json 统计脚本）
for repo in vllm sglang tgi dynamo; do
  echo "== $repo =="
  find "$repo" -name '*.rs' -not -path '*/target/*' | xargs wc -l | tail -1
done

# 看 Rust 代码具体分布在哪些目录（比行数更有信息量）
find vllm -name '*.rs' -not -path '*/target/*' | sed 's#/[^/]*$##' | sort -u | head -20
find sglang -name '*.rs' -not -path '*/target/*' | sed 's#/[^/]*$##' | sort -u | head -20
find tgi -name '*.rs'   -not -path '*/target/*' | sed 's#/[^/]*$##' | sort -u | head -20
find dynamo -name '*.rs' -not -path '*/target/*' | sed 's#/[^/]*$##' | sort -u | head -20

# 验证本篇最关键的一条区分：cdylib（进程内扩展）vs bin（独立可执行文件）
grep -A2 '^\[lib\]' sglang/rust/sglang-server/Cargo.toml    # 应该看到 crate-type = ["cdylib", "rlib"]
grep -A2 '^\[\[bin\]\]' sglang/sgl-model-gateway/Cargo.toml # 应该看到 path = "src/main.rs"

# 验证两家的 Rust 前端默认是不是关闭的
grep -n 'VLLM_USE_RUST_FRONTEND' vllm/vllm/envs.py | head -1
grep -n 'SGLANG_RUST_SERVER = EnvBool' sglang/python/sglang/srt/environ.py
```

跑完这几条命令，读者能自己确认本篇接下来所有结论的两个物理事实：**Rust 代码集中在少数几个跟"服务/路由/队列"相关的目录里，且两家把 Rust 前端做成了默认关闭的可选项**。第 4 节把每一条都钉到具体行号。

有一点要提前说明：这条 `find | wc -l` 命令数出来的绝对行数，大概率跟 `_PLAN.md` §3.2 表格里的百分比对不上——原因不是哪边算错了，是口径不同。`_PLAN.md` 的数字来自研究库统一的仓库快照统计脚本（`repo_stats.json`），大概率排除了 `target/` 编译产物、可能对生成代码（比如 protobuf 生成的 `.rs`）和 vendor 依赖有自己的一套过滤规则；这里给的是最朴素的 `find`，没有做任何过滤。**两者的用途不同**：`_PLAN.md` 的数字用来做"跨引擎的占比对比"，本节的命令用来"确认 Rust 代码具体落在哪些目录、这些目录是不是跟本篇讲的位置对得上"——后者只需要相对准确，不需要跟前者的百分比小数点对齐。



如果研究库那份 clone 不在本机，`_verify.py` 会自动跳过对应的引擎引用校验并在报告里明说跳过条数——不影响本篇正文的可读性，但建议按研究库 `_PLAN.md` §2 的说明把 clone 拉下来，跑一遍上面的命令比读别人总结出来的表格更有说服力。

## 4. 真实引擎是怎么做的（对照 vLLM/SGLang 等，带 `引擎:文件:行`）

### 4.1 vLLM：独立可执行文件 + headless Python 引擎子进程，默认关闭

vLLM 的 `rust/` 是一个独立的 Cargo workspace，自称 "a Rust drop-in alternative frontend for vLLM"（`vllm:rust/README.md:3`），按分层组织成几个 crate（`vllm:rust/README.md:9-31`）：

```text
vllm-cmd / vllm-rs   CLI 入口：可以拉起 Python 前端子进程、可以拉起 Rust managed-engine 模式、也能纯渲染不接引擎
vllm-server          OpenAI 兼容 HTTP API（基于 axum）
vllm-chat            chat completions：模板渲染、结构化事件、reasoning/tool 解析
vllm-text            tokenizer 与增量 detokenizer
vllm-llm             对引擎客户端的一层 token-in/token-out 门面
vllm-engine-core-client  ZMQ 传输 + MessagePack 协议，对接 headless vLLM 引擎
```

关键的一句话在 README 里说得很直白：`vllm-rs` **不跑模型前向**，它是"integrates into Python vllm as a Rust frontend subprocess"（`vllm:rust/README.md:32-34`）——Python 仍然拥有进程启动权，把 Rust API 服务器当成一个被 Python 监督的子进程拉起来，继承好的监听 socket 和传输地址再交给它。启用方式是一个环境变量：

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B    # vllm:rust/README.md:43
```

**默认是关闭的**：`vllm:vllm/envs.py:165` 那一行写的是 `VLLM_USE_RUST_FRONTEND: bool = False`。不设这个环境变量，vLLM 用的还是原来的 Python 前端。

Rust 侧真正干的事，落在 `managed-engine` 这个 crate 里——它管理的是"托管一个 headless Python vLLM 引擎子进程"这件事本身。核心结构体的文档写得很清楚：

```
/// RAII-style handle for one managed Python headless engine subprocess.
pub struct ManagedEngineHandle { ... }        // vllm:rust/src/managed-engine/src/process.rs:78-79

impl ManagedEngineHandle {
    /// Spawn one managed Python headless engine and return a handle for monitoring it.
    pub async fn spawn(config: ManagedEngineConfig) -> Result<Self> {   // vllm:rust/src/managed-engine/src/process.rs:87
        ...
        let child = command.spawn().context("failed to spawn managed engine")?;
        ...
    }
}
```

`ManagedEngineHandle` 结构体定义在 `vllm:rust/src/managed-engine/src/process.rs:78-79`，`spawn` 方法在 `vllm:rust/src/managed-engine/src/process.rs:87`；调用点在 CLI 入口里：`vllm:rust/src/cmd/src/main.rs:145` 那一行就是 `ManagedEngineHandle::spawn(engine_config)`。整条链路合起来是：Rust 进程自己起一个完整的 HTTP 服务器接外部请求，但真正跑 Transformer 前向的，是它用标准 `Command`/`Child`（进程级别的子进程，不是线程）拉起来的另一个 Python 进程——`vllm-engine-core-client` 那层再通过 ZMQ + MessagePack 跟这个 Python 子进程通信。这跟 `02-Python为什么还没被淘汰.md` §4.1-4.2 讲的"vLLM 用 `msgspec.Struct` + 多进程绕开 GIL"是同一条设计思路的延伸——只是这次连"发指令的那一端"也换成了 Rust。

README 还提到一种"External Engine"部署形态（`vllm:rust/README.md:46-48`）：`vllm-rs serve` 可以单独跑、只当 Rust 前端，Python 引擎进程在别的地方（甚至别的机器）用 `--headless` 启动，两边通过 data-parallel 的握手地址对上：

```bash
# 第一步：只起 Python headless 引擎，不带任何前端
vllm serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1

# 第二步：另起一个只跑 Rust 前端的进程，通过同一个握手地址接上第一步的 Python 引擎
vllm-rs serve Qwen/Qwen3-0.6B \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 0        # vllm:rust/README.md:51-68
```

两条命令分别起两个独立进程，`--data-parallel-size-local 0` 明确告诉 Rust 前端"本地不跑任何引擎，去别处找"。这进一步说明 Rust 这层的职责边界很清楚：**它只管请求的接入、解析、路由，模型推理这件事从头到尾都在 Python/CUDA 那一侧**，Rust 甚至不需要跟运行推理的进程在同一台机器上。

### 4.2 SGLang：两种完全不同的 Rust 形态

这是本篇最值得核实清楚的一段——SGLang 的 "Rust 占 6%" 这个数字，背后其实盖住了两种设计目标、部署方式、职责边界都不一样的代码。

**形态 A：进程内 PyO3 扩展（`rust_server.py` 挂进来的东西）**

入口文件 `python/sglang/srt/managers/rust_server.py` 一开头就把自己的定位写清楚了：

> "The Rust server replaces the Python api-server + `TokenizerManager` + `DetokenizerManager` stack (hence this module sits beside them in `managers/`), running them as Rust threads inside the scheduler process." （`sglang:python/sglang/srt/managers/rust_server.py:1-8`）

**"running them as Rust threads inside the scheduler process"**——这一句是形态 A 和形态 B 最本质的区别。`RustServer` 类的文档字符串重复了同一件事：

```
class RustServer:
    """Owns the embedded multi-threaded Rust server (``sglang_server.Server``).

    The server owns the api-server, tokenizermanager, tokenizer, and detokenizer
    all implemented as Rust threads in scheduler process.
    """
```
（`sglang:python/sglang/srt/managers/rust_server.py:351-356`）

它是怎么"挂"进 Python 进程的：`RustServer.launch` 这个 classmethod 里，用的是 `load_rust_extension` 动态导入一个编译好的原生扩展模块（`sglang:python/sglang/srt/managers/rust_server.py:369,375,377`）：

```python
from sglang.srt.rust_extensions import load_rust_extension
Server = load_rust_extension("sglang.srt.rust_extensions._server").Server
```

这不是拉起一个新进程，是 **import 了一个 `.so`/`.pyd` 动态库**，这个库是用 PyO3 编译出来的 Python 扩展模块。证据在对应 Rust crate 的 `Cargo.toml` 里写得非常直白：

```toml
# Consumed by python/setup.py: registers this crate as a PyO3 extension module.
[package.metadata.sglang]
python-module = "sglang.srt.rust_extensions._server"     # sglang:rust/sglang-server/Cargo.toml:8-10

[lib]
name = "sglang_server"
# cdylib: the pyo3 module imported by the (embedded) Python scheduler process.
# rlib:   so optional standalone bins / integration tests can link the core.
crate-type = ["cdylib", "rlib"]                            # sglang:rust/sglang-server/Cargo.toml:14-17
```

这段 `python-module` 元数据在 `sglang:rust/sglang-server/Cargo.toml:8-10`，`crate-type` 声明在 `sglang:rust/sglang-server/Cargo.toml:14-17`。`crate-type = ["cdylib", ...]` 编译出来的是一个**动态链接库**，不是一个独立可执行文件——它没有自己的 `main` 函数，没有自己的进程，是被 Python 解释器 `import` 进同一个地址空间、以一组额外 OS 线程的形式运行的。

这一层默认是关闭的：`sglang:python/sglang/srt/environ.py:1519` 一行 `SGLANG_RUST_SERVER = EnvBool(False)`。打开条件也很具体——只有 rank 0 才会承担这份职责（避免多个 data-parallel/tensor-parallel rank 重复起一份服务）：

```python
def _hosts_rust_server(self) -> bool:
    return envs.SGLANG_RUST_SERVER.get() and (
        self.ps.pp_rank == 0
        and self.ps.attn_tp_rank == 0
        and self.ps.attn_cp_rank == 0
    )
```
（`sglang:python/sglang/srt/managers/scheduler.py:2012-2020`，调用点在 `sglang:python/sglang/srt/managers/scheduler.py:2022-2033` 的 `maybe_init_rust_server`）

为什么能在同一个 Python 进程里塞进 Rust 线程还不被 GIL 拖累？答案跟 `02-Python为什么还没被淘汰.md` §2.3(a) 讲的"IO 系统调用会主动释放 GIL"是同一个心智模型，只是走得更彻底——PyO3 允许 Rust 代码显式释放 GIL 后再执行一段不触碰 Python 对象的逻辑，这段时间里 Rust 线程可以真正并行跑，不需要等 Python 解释器的调度。**这是绕开 GIL 的第三条路**：`02-Python为什么还没被淘汰.md` §4.2 讲的是"开多个 Python 进程各自持有自己的 GIL"，这里是"把一段逻辑整体挪出 Python 字节码，用一种能主动释放 GIL 的语言实现，运行在同一个进程里"。

**形态 B：独立进程（`sgl-model-gateway` 与 `experimental/sgl-router`）**

`sgl-model-gateway` 的 `Cargo.toml` 跟形态 A 的 `Cargo.toml` 形成直接对照：

```toml
[[bin]]
name = "sgl-model-gateway"
path = "src/main.rs"          # sglang:sgl-model-gateway/Cargo.toml:24-26
```

Cargo.toml 里这个 `bin` 段（TOML 数组表语法，两层方括号包住 `bin`）配上 `path = "src/main.rs"` 编译出来的是一个**独立可执行文件**，有自己的 `main` 函数、自己的进程、自己的 PID。`src/main.rs` 是一个普通的 `clap` 命令行程序（`sglang:sgl-model-gateway/src/main.rs:3`），自己起 HTTP/gRPC 服务器，管理一批**已经在别处独立跑起来的** SGLang worker（可能是不同机器上的多个 SGLang 实例）——README 自己的定位是"High-performance model routing control and data plane for large-scale LLM deployments"，"orchestrates fleets of workers"、"routes requests across HTTP, PD (prefill/decode), gRPC ... backends"（`sglang:sgl-model-gateway/README.md:3,6-7`）。

`experimental/sgl-router` 是同类思路更轻的一个实现，README 一句话说清楚定位："Slim, KV-aware, OpenAI-compatible router for SGLang workers. Serves a single model and routes across its workers."（`sglang:experimental/sgl-router/README.md:3,6-7`）

**两种形态的区别，逐条对比**：

| 维度 | 形态 A（`rust_server.py` / PyO3 扩展） | 形态 B（`sgl-model-gateway` / `sgl-router`） |
|---|---|---|
| 进程边界 | **没有独立进程**——是 Python scheduler 进程内的几条 Rust 线程 | **完全独立的 OS 进程**，可以跑在别的机器上 |
| 构建产物 | `crate-type = ["cdylib", ...]`，编译成动态库被 `import` | Cargo.toml 里的 `bin` 段，编译成独立可执行文件，自己是入口 |
| 部署关系 | 一对一——随每个 SGLang scheduler 实例一起启停 | 一对多——一个 gateway/router 管理一批 worker 实例 |
| 默认开关 | 默认关闭（`SGLANG_RUST_SERVER=False`），需要显式打开才把 Python api-server 换成 Rust | 本身就是可选的独立组件，单实例部署完全用不上它 |
| 解决的问题 | "单实例内，把 Python 前端换成更快的前端"——跟 vLLM 的 `rust/` 是同一类思路 | "多实例之间怎么做负载均衡/KV 感知路由"——跟 4.4 节 Dynamo 的 `kv-router` 是同一类思路 |

这条区分是本篇最容易被"SGLang Rust 占 6%"这一个数字掩盖的事实：**这 6% 里装的是两种完全不同的设计哲学**，一种是"给单个引擎实例配一个可选的原生加速前端"，一种是"在多个引擎实例之上再加一层路由/负载均衡"。行数统计工具（`repo_stats.json` 背后用的 linguist 类工具）不区分这两种语义，只按文件后缀数行——这提醒我们：**语言构成表能告诉你"用了多少"，回答不了"用来干什么"，后者必须去读代码。**

### 4.3 TGI：router 的连续批处理调度在 Rust 侧

TGI 跟 vLLM/SGLang 不一样的地方在于：它的请求路由/调度层**从一开始就是 Rust 写的**，不是后加的可选组件。`_PLAN.md` 原计划提到的路径是 `router/src/queue.rs`，但实测这份 clone 里连续批处理的队列逻辑实际落在 `backends/v2/src/queue.rs` 与 `backends/v3/src/queue.rs`（TGI 从 v2 到 v3 做过一次后端拆分，`router/` 目录现在只保留通用的 HTTP/协议层）。

`backends/v3/src/queue.rs` 里三层结构很清楚：

```rust
pub(crate) struct Entry { ... }      // sglang风格的一条排队请求，含 response channel、span、block 分配
                                       // tgi:backends/v3/src/queue.rs:22

pub(crate) struct Queue {             // 对外暴露的队列句柄，内部只是个 mpsc channel
    queue_sender: mpsc::UnboundedSender<QueueCommand>,
}                                      // tgi:backends/v3/src/queue.rs:41

struct State {                        // 真正持有队列状态的地方：VecDeque<Entry>、block_allocator
    entries: VecDeque<(u64, Entry)>,
    ...
}                                      // tgi:backends/v3/src/queue.rs:166
```

三个结构体的定义分别在 `tgi:backends/v3/src/queue.rs:22`（`Entry`）、`tgi:backends/v3/src/queue.rs:41`（`Queue`）、`tgi:backends/v3/src/queue.rs:166`（`State`）。排队决策的核心逻辑在 `State` 的 `next_batch` 方法里（`tgi:backends/v3/src/queue.rs:237` 起），检查 `min_size`/`max_size`/`prefill_token_budget`/`token_budget` 这些参数——这正是连续批处理要回答的核心问题："这一步该往 batch 里塞多少条新请求、塞哪些"。这段决策逻辑完整地跑在 Rust 里，跟 GPU 计算之间隔着一层 gRPC：把组好的 batch 通过 `ShardedClient` 发给一个独立的 Python/PyTorch 分片进程，这个分片进程**只管前向计算**，本身不做任何调度决策（背景任务的入口是 `tgi:backends/v3/src/backend.rs:129` 的 `batching_task`，文档注释写的是"Batches requests and sends them to the inference server"）。

面向 HTTP handler 的门面是 `Infer` 结构体：

```rust
pub struct Infer {
    validation: Validation,
    backend: Arc<dyn Backend + Send + Sync>,   // v3 的 Backend 实现就是上面这套 Queue + batching_task
    ...
}   // tgi:router/src/infer/mod.rs:49
```

`Infer` 结构体定义在 `tgi:router/src/infer/mod.rs:49`，持有一个 `Backend` trait 对象——`backends/v3` 这个 crate 就是这个 trait 的一份具体实现，意味着 `router/` 那层通用的 HTTP/校验/chat 模板逻辑，跟具体的调度/批处理实现（v2、v3，未来可能还有别的后端）之间是解耦的。这跟 vLLM/SGLang 的模式反过来：vLLM/SGLang 是"默认 Python，可选换成 Rust"；TGI 是"从一开始就是 Rust 的路由/调度层 + 一个只管算数的 Python 分片进程"，没有"降级回 Python 路由"这个选项。

### 4.4 Dynamo：集群路由，KV-aware routing

Dynamo 跟前三家有一个根本区别，README 自己写得非常直接：

> "Dynamo is the orchestration layer above inference engines — it doesn't replace SGLang, TensorRT-LLM, or vLLM, it turns them into a coordinated multi-node inference system." （`dynamo:README.md:36`）
> "Built in Rust for performance, Python for extensibility." （`dynamo:README.md:38`）

**Dynamo 不是一个推理引擎，它是站在多个引擎实例之上的编排层**——这解释了它为什么是四家里 Rust 占比最高的（27%）：它不是"给某个引擎顺带配一个可选的原生前端"，Rust 就是它主要的实现语言，Python 负责的是可扩展性（自定义组件、脚本化编排）这一侧。

`lib/kv-router/` 是这个编排层里跟本篇主题最相关的部分——KV-aware 路由：给一个新来的请求，从一批 worker 里选出"KV 缓存跟这个请求的 prompt 重叠最多"的那一个，这样可以复用已经算过的 KV，省掉重复的 prefill。这件事的核心数据结构是一棵前缀基数树：

```rust
pub struct RadixTree {
    root: SharedRadixBlock,
    lookup: FxHashMap<WorkerWithDpRank, WorkerLookup>,
}   // dynamo:lib/kv-router/src/indexer/radix_tree.rs:49
```

`RadixTree` 结构体定义在 `dynamo:lib/kv-router/src/indexer/radix_tree.rs:49`；配套的路由配置结构体 `KvRouterConfig` 定义在 `dynamo:lib/kv-router/src/scheduling/config.rs:660`。crate 自己的 README 把定位说得很清楚："provides the core KV-aware routing data structures and scheduling primitives used by Dynamo to steer requests toward workers with the best cache overlap"（`dynamo:lib/kv-router/README.md:1-4`）。

注意这里发生的计算：radix tree 的前缀匹配、hash 查找、worker 打分——全部是纯 CPU 上的数据结构操作，**不需要 GPU，也不需要理解模型权重**，只需要知道"哪些 token 序列被哪个 worker 缓存过"。这跟 2.1(a) 的判断完全对应：这是路由决策，不是前向计算，天然没有 GPU 计算可以掩盖它的耗时，必须自己足够快。

Dynamo 的 HTTP 前端同样是 Rust 写的，用的是 `axum`：

```rust
pub struct HttpService {
    state: Arc<State>,
    router: axum::Router,
    port: u16,
    host: String,
    ...
}   // dynamo:lib/llm/src/http/service/service_v2.rs:620-635
```

`HttpService` 结构体定义在 `dynamo:lib/llm/src/http/service/service_v2.rs:620-635`，这跟 vLLM `rust/` 里的 `vllm-server`（同样基于 axum）、SGLang `sgl-model-gateway` 的 HTTP 层是完全同一类组件——"请求进来的第一站"用 Rust 写一个高并发的 HTTP 服务器，这几乎是四个项目里唯一没有分歧的一点。

### 4.5 五处 Rust，按 2.3 节的框架收拢成一张表

把前四节的证据按"编译产物"和"管的是什么"两条轴收拢：

| 位置 | 编译产物 | 进程边界 | 管的是什么 | 默认状态 |
|---|---|---|---|---|
| vLLM `rust/` | 独立可执行文件（`vllm-rs`） | 独立进程，通过 ZMQ 连 headless Python 引擎 | 单实例的 HTTP 前端：解析请求、chat 模板、增量 detokenize | 关闭（`VLLM_USE_RUST_FRONTEND=False`） |
| SGLang `rust_server.py` 挂的扩展 | `cdylib`，被 Python `import` | **无独立进程**，是 scheduler 进程内的线程 | 单实例内替换 api-server + TokenizerManager + DetokenizerManager | 关闭（`SGLANG_RUST_SERVER=False`） |
| SGLang `sgl-model-gateway`/`sgl-router` | 独立可执行文件 | 独立进程，可跨机器 | 多实例之间的负载均衡、KV 感知路由 | 可选的独立部署组件 |
| TGI `router` + `backends/*` | 独立可执行文件（router 是主程序） | 独立进程，通过 gRPC 连 Python 分片进程 | 单实例的连续批处理队列与调度决策 | **没有默认开关**——从一开始就是 Rust |
| Dynamo `lib/kv-router`、`lib/llm/.../service` | 独立可执行文件（编排层本身） | 独立进程，站在多个引擎实例之上 | 多实例/多节点的 KV 感知路由、HTTP 前端、编排 | Dynamo 本身就是这一层，无"开关"概念 |

这张表把 1.2 节的直觉钉实了：五处代码按"管一个实例内部"还是"管多个实例之间"，能整齐地分成两组——**vLLM 的 `rust/`、SGLang 的 `rust_server.py`、TGI 的 `router`+`backends`** 管的是单实例内"请求怎么被这一个引擎进程接住"；**SGLang 的 `sgl-model-gateway`、Dynamo 的 `kv-router`** 管的是"一个请求该被路由到一堆实例里的哪一个"。TGI 因为历史上就没有过 Python 前端，所以它是这张表里唯一没有"默认关闭"这一格的——不是它更激进，是它压根没有"退回 Python"这条路可选。

把这张表和第 0 节的两句话对照读：**"没有一个引擎是纯 Rust 写的"对应的是"管的是什么"这一列——五行里没有一行写着"模型前向"；"4 个在特定层用了 Rust、位置高度一致"对应的是"进程边界"这一列——除了 SGLang 的形态 A，其余全部是独立进程，而形态 A 本身也在 2.1(a)(b)(c) 的三条共性范围内（CPU 密集、高并发、不碰模型生态），只是被塞进了同一个 Python 地址空间而不是另起进程。** 接下来第 5 节把这张表翻译成读者能直接拿去用的判据。

## 5. 设计决策与代价（为什么这样 / 不这样会怎样 / 什么时候可以不这样）

### 决策一：Rust 放在"请求进来到交给 GPU 之前"这一段，不进模型执行层

- **为什么这样**：这一段 CPU 密集、高并发、不需要碰模型生态（2.1 节三条），恰好是 Rust 的舒适区；模型执行层的两块地盘（CUDA 算子生态、HF 权重/config 生态）都是巨大的存量资产，没有空子可钻（2.2 节）。
- **不这样会怎样**：如果把模型执行层也重写成 Rust，要么用 FFI 绑定 CUDA（内存安全的核心卖点在 `unsafe` 边界内失效，工作量堆在胶水代码上），要么放弃现有 GPU 生态从零建一套算子库（工程量对比 cuBLAS/FlashAttention 几乎不可能追上）——这不是理论推演，是 12 个真实项目共同做出的选择，0 个反例。
- **什么时候可以不这样**：如果目标从一开始就不是对齐现有 CUDA/PyTorch 生态，而是面向一个全新的、生态本来就小的执行环境（比如自研 ASIC，没有现成 CUDA 可依赖），Rust 和 C++ 在"从零搭算子库"这件事上处于同一起跑线，这时候 Rust 的内存安全会成为真实优势——但要清楚这已经不是"重写一个 vLLM"，而是一个规模大得多的项目，llama.cpp 选择 C++ 而不是 Rust 走这条路（`03-C++路线-llama.cpp的取舍.md`），提示这个选择更多取决于生态和团队积累，不是语言本身的必然结论。

### 决策二：两家的 Rust 前端都默认关闭——这说明它是可选加速，不是架构基石

- **为什么这样**：`02-Python为什么还没被淘汰.md` §5.2 的判据表已经说明，Python 侧调度开销追上 GPU 计算只在特定象限（小模型、高并发、短序列）才会发生；对大多数用户（大模型、常规并发、长序列），Python 前端和 Rust 前端的可感知差异很小，却要多背负一套需要保持功能对齐的代码——vLLM 的 `rust/README.md` 自己承认"experimental, and is not feature-complete. We are working to add more functionality from the python front-end"（`vllm:rust/README.md:5`），这说明新功能大概率先在 Python 前端出现，Rust 前端在追。默认关闭，等于不强迫所有用户绑定到功能滞后的路径上。
- **不这样会怎样**：如果默认打开，相当于让每个用户都承担一层实验性组件的风险——多一个编译产物要维护、功能覆盖率暂时落后于 Python 前端的用户会先撞上缺失的特性。
- **什么时候可以不这样**：确认自己的工作点确实落在"小模型 + 高并发 + 短序列"这个 Python 调度开销显性化的象限（`02-Python为什么还没被淘汰.md` §5.2 判据表第三行），并且愿意接受实验性组件当前的维护成本时，显式打开 `VLLM_USE_RUST_FRONTEND=1` 或 `SGLANG_RUST_SERVER=1` 才是划算的——这也是这两个开关目前唯一被设计出来要服务的场景。**如实说一句**：默认关闭不代表这条路线没用或者失败了，它只是说明收益是工作点相关的，不是普适的；渐进式把功能从 Python 搬到 Rust、观察反馈再决定要不要转正，是一条正常的项目演进路径，不是被放弃的信号。

### 决策三：如果你只写路由/网关层

- **为什么这样**：如果目标是给一批已经写好的引擎（自己的或者 vLLM/SGLang）做负载均衡/请求路由，这正是四个真实项目一致选中的位置——不需要碰模型，收益边界清楚（HTTP 吞吐、路由延迟、连接数），Rust 的类型系统和无 GC 在高 QPS、长连接（SSE 流式响应）场景下有实打实的工程收益：少一层解释器开销，内存占用可预测，不会有 GC 停顿打断正在流式返回的连接。
- **不这样会怎样**：用 Python 写这一层，在"小模型/高并发/短序列"这个象限里，Python 固定开销可能真的追上甚至超过后端处理时间——这不是假设，是 TGI、Dynamo 两家真实项目做出的同一个选择（TGI 甚至没有给 Python 路由留退路，一开始就是 Rust）。
- **什么时候可以不这样**：路由逻辑本身简单（单模型、单组 worker、round-robin 或者简单的健康检查就够），并且 QPS 远没到 Python 调度开销显性化的量级——这时候用 Python 写一个薄的反向代理完全够用，不需要为了"看起来更专业"引入 Rust 的编译/工具链成本。`experimental/sgl-router` 之所以定位为"slim"，正是因为它承认多数场景不需要 `sgl-model-gateway` 那么重的一整套 control plane。

### 决策四：如果你想写一个完整引擎（含模型执行层）

- **为什么不建议把执行层也用 Rust 写**：12 个真实项目 0 个先例。模型生态（权重格式、config、tokenizer）和算子生态（CUDA kernel、Triton）两头都是 Python/C++ 的地盘，在 Rust 里重新对齐这两套生态的工程量，大概率比"引擎本身要解决的问题"更大——这是一个机会成本问题，不是"Rust 做不到"的问题。
- **不这样会怎样（如果坚持整体用 Rust）**：要么用 FFI 包一层 CUDA，本质上退化成"用 Rust 语法写 C++"，最危险的显存操作恰好发生在 `unsafe` FFI 边界内部，内存安全的卖点在最需要它的地方反而不生效；要么放弃 GPU 转向纯 CPU 推理——但也没有任何证据表明 Rust 在这条路上比 C++（llama.cpp 已经证明可行）有本质优势，说这有优势会超出"本库推断"能负责的范围。
- **什么时候可以不这样**：目标本来就不是对齐现有 CUDA/PyTorch 生态，而是面向一个全新的、更小的执行环境——这时候决策四和决策一的"什么时候可以不这样"是同一个答案：生态是空地时，Rust 才有机会进模型执行层，但项目规模也会随之大很多。

### 决策五：如果你想学 Rust，顺便做个项目

- **为什么合理**：路由/网关层是一个边界清晰、外部依赖少、能独立验收的子项目——实现一个"给 N 个后端做健康检查 + 负载均衡 + SSE 流式转发"的 HTTP 服务，恰好覆盖 Rust 最擅长的领域（异步 IO、类型安全的状态机、无 GC 的可预测延迟），而且四个真实项目提供了可以对照学习的成熟设计——TGI 的 `Queue`/`batching_task`（4.3 节）、Dynamo 的 `kv-router`（4.4 节）都是可读的参考实现。
- **不这样会怎样（如果选了模型执行层练手）**：会把"学 Rust 本身"这条曲线，跟"对齐 PyTorch/CUDA 生态"这个巨大工程量捆在一起，大概率大部分时间耗在"怎么调 CUDA 的 FFI、怎么让 tensor 的生命周期在 Rust 类型系统里说得通"上，反而学不到 Rust 真正好在哪。
- **什么时候可以不这样**：如果学习目标根本不是"练 Rust"而是"看懂 GPU 怎么被喂饱"，那应该去写 `02-最小可跑引擎/` 里那类东西（`_lab/minigpt.py`），完全不需要 Rust——语言要服务于学习目标，不是反过来为了用某个语言而找一个理由。

### 5.3 一个常被漏掉的中间状态：先用 Python 写完整个引擎，再决定要不要抽出 Rust 层

前面五条决策容易给人一种错觉：仿佛"用不用 Rust"是写引擎第一天就要拍板的事。四个真实项目的实际路径并非如此——vLLM、SGLang 都是先有了成熟的 Python 引擎，Rust 前端是后来才作为可选层加上去的（`rust/README.md` 自称 "experimental" 就是这段历史的痕迹）；只有 TGI 是个例外，从项目早期就选择了 Rust 路由层。这提示一条更现实的路径：**自己写引擎时，不必在动笔前就纠结要不要引入 Rust**——先用 Python（或者任何你最快能迭代的语言）把路由/调度层写出来，跑起来、测出真实的请求量级和序列长度分布，再对照决策三的判据看是否真的落在"判据容易翻转"的象限。四个项目没有一个是"先决定用 Rust 再倒推需求"，都是先有真实的工作负载压力，再把那一层单独抽出来重写。**先能跑、再谈性能层的语言选型**，这条顺序本身也是一条设计决策，代价是"可能要重写一次路由层"，好处是"不会在需求还不清楚的时候，为一个可能根本不存在的瓶颈预先花掉 Rust 的学习/维护成本"。

## 6. 常见错误与踩坑

**误区一："Rust 更快，所以该把模型执行层也用 Rust 重写"**

12 个开源引擎、0 个先例反驳了这句话。快不快从来不是这里的决定因素——2.2 节已经拆开讲过：模型执行层的两块存量资产（CUDA 算子生态、HF 权重生态）谁都没有理由重新发明，Rust 想进去，要么付出 FFI 代价，要么付出从零建生态的代价，两条路的成本都远超收益。**"更快"只在没有存量资产要对齐的空地上才是决定性优势**，而模型执行层恰恰不是这样的空地。

**误区二："用了 Rust" = "用 Rust 重写了整个引擎"**

Dynamo 的 Rust 占比 27%，乍看是四家里最高的，容易让人以为它是"最激进的 Rust 引擎"。但 Dynamo 根本不是一个推理引擎——它自己说自己是"orchestration layer above inference engines"，"不替代 SGLang、TensorRT-LLM 或 vLLM"（`dynamo:README.md:36`）。27% 这个数字高，是因为 Dynamo **整个产品的核心职责就落在本篇讨论的这一层**，而不是因为它比其他三家更愿意用 Rust 写模型计算。反过来，SGLang 6%、vLLM 5%、TGI 12%，都只是"给一个巨大的 Python/CUDA 引擎加装的一层"，占比低恰恰因为分母（整个引擎）里绝大部分还是 Python 和 CUDA。**读行数占比时一定要问"分母是什么"，不问就容易把"占比高"直接等同于"更彻底"。**

**误区三：把默认关闭理解成"这条路线没用/是失败品"**

vLLM 的 Rust 前端、SGLang 的 `SGLANG_RUST_SERVER` 都默认关闭，这个事实本身如实写就好，不要过度解读。默认关闭说明的是**收益的工作点相关性**——只有在 Python 调度开销确实显性化的场景（小模型、高并发、短序列）才值得承担这层复杂度；不代表"上游团队试了发现不行"。`vllm:rust/README.md:5` 自己写的是"experimental, and is not feature-complete"，这是一个渐进式迁移项目正常应有的状态描述，不是免责声明式的自我否定。同样，SGLang 的 `rust_server.py` 模块文档把自己定位成对 Python 三件套（api-server + TokenizerManager + DetokenizerManager）的一个"替代"（`sglang:python/sglang/srt/managers/rust_server.py:3`），而不是"补丁"，说明设计上是认真的，只是还没有理由让所有人默认承担它的维护成本。

**误区四：把 SGLang 的两种 Rust 形态混为一谈**

`rust_server.py` 挂进来的 PyO3 扩展和 `sgl-model-gateway` 是解决完全不同问题的两套代码：前者是单实例内的前端加速（跟 vLLM 的 `rust/` 是同一类别），后者是多实例之间的负载均衡/路由（跟 Dynamo 的 `kv-router` 是同一类别）。语言构成统计工具不区分这两种语义，"SGLang Rust 占 6%"这一个数字把两者的行数直接相加——不拆开看代码，很容易得出"SGLang 只有一种用 Rust 的方式"这种错误印象，进而误判"SGLang 的 Rust 集成程度比 vLLM 深一点点"（6% vs 5%），实际上这两个 6% 和 5% 根本不是在比较同一类东西：vLLM 的 5% 只对应一种形态（独立前端进程），SGLang 的 6% 是两种形态加总，单看形态 A 甚至可能比 vLLM 的独立前端更小。

**误区五：以为 Dynamo 里出现的另一门语言（Go）也是在跟 Rust 抢同一层**

`_PLAN.md` §3.2 的表格里，Dynamo 前四语言是 "JSON 34%, Rust 27%, Python 18%, Go 6%"（`_PLAN.md:110`）——Rust 和 Go 都出现在同一个项目里，容易让人以为这是"两种系统语言在同一层竞争"。实测并不是：Dynamo 里的 Go 代码集中在 `deploy/operator/` 和 `deploy/inference-gateway/`，前者的 README 说得很直接——"A Kubernetes Operator to manage all Dynamo pipelines using custom resources"（`dynamo:deploy/operator/README.md:3`）。这是 **Kubernetes 生态的部署/生命周期管理工具**（Operator 模式、CRD、Kubebuilder），管的是"怎么把 Dynamo 的这些组件部署到一个 K8s 集群里、怎么声明式地描述这套拓扑"，跟本篇讨论的"请求进来之后怎么被路由/调度"是完全不同的两个问题维度——一个是**部署时**的问题，一个是**请求时**的问题。Go 出现在这里，是因为 Kubernetes 生态本身的 Operator/controller 开发工具链（Kubebuilder、client-go）是 Go 生态的地盘，这跟 2.1(c) 讲的"不需要碰模型生态、落在空地上"是同一条道理的另一个例子——只是这次的"空地"属于 Kubernetes 生态而不是请求路由，落子的语言也就自然换成了那个生态惯用的语言。**这条误区的教训可以推广**：看到一个项目里出现好几种"系统语言"，先问它们各自服务于哪个维度（请求时 / 部署时 / 构建时），再判断它们是不是真的在互相竞争同一个位置。

五条误区合起来指向同一条方法论：**行数占比是一个入口，不是一个结论。** 它能提示"这里可能有值得细看的东西"，但"这一层具体在做什么、跟别的语言/别的组件是什么关系"，永远要靠打开代码、找到真正的入口函数、读一遍调用链才能确认——这也是本篇从第 4 节开始每一条断言都配真实文件行号，而不是停在 `_PLAN.md` 那张统计表上的原因。

## 7. 自测题与延伸阅读

**自测题（闭卷回答，答案都在本篇里）**：

1. 本篇说 Rust "从不出现在模型执行层"，这条判断成立的两个前提条件分别是什么（对应 2.2 节的两块"存量资产"）？如果这两个前提都不成立（比如面向一个全新的专有硬件），这条判断还站得住吗？
2. vLLM 的 `rust/` 和 SGLang 的 `rust_server.py` 都用 Rust 实现了一层"前端"，但它们跟"跑模型的那一侧"之间的进程边界完全不同。分别描述这两种边界，并说出各自对应哪一种 Cargo `crate-type`。
3. SGLang 的 `sgl-model-gateway`/`sgl-router` 和 `rust_server.py` 挂进来的 PyO3 扩展，各自解决的是"单实例内"还是"多实例之间"的问题？把这两者跟 TGI、Dynamo 的 Rust 组件分别配对，哪两组解决的是同一类问题？
4. 为什么 TGI 的 Rust 路由层没有"默认关闭、可选打开 Python 路由"这个选项，而 vLLM/SGLang 的 Rust 前端有？这跟三家引擎的历史架构选择有什么关系？
5. Dynamo 的 Rust 占比（27%）远高于其他三家，本篇给出的解释是什么？这个解释能不能反过来支持"Dynamo 比 vLLM/SGLang 更彻底地信任 Rust 处理模型计算"这句话？为什么不能？
6. 第 2.1 节说 Rust 所在层的 CPU 开销"没有 GPU 可以掩盖它"，这跟 `02-Python为什么还没被淘汰.md` 里 `T_py < T_gpu` 判据讨论的 CPU 开销是同一类问题吗？两者的边界差在哪里？
7. 如果你要给自己在 `02-最小可跑引擎/07-把它变成一个HTTP服务.md` 里搭的玩具引擎加一层路由/网关，会不会现在就用 Rust？决策三给出的判据，代入你的玩具引擎的实际 QPS 和序列长度，落在哪一边？
8. Dynamo 里同时出现 Rust（27%）和 Go（6%）两种系统语言，误区五给出的判断是"它们不在竞争同一层"。用哪个具体证据支持这个判断？如果换一个假想场景——某个引擎的 Go 代码也出现在请求路由路径里——误区五的结论还成立吗？

**延伸阅读（本库内）**：

- `01-用什么语言写推理引擎-12个真实引擎的证据.md` —— 本篇引用的语言构成表的完整出处和方法论
- `02-Python为什么还没被淘汰.md` —— 本篇反复借用的 `T_gpu`/`T_py` 判据框架，以及 GIL/多进程/msgspec 这些细节的完整展开
- `03-C++路线-llama.cpp的取舍.md` —— 另一条"不碰 Python 生态"的语言选型路线，对照本篇决策一/决策四的取舍逻辑

双链：[[01-用什么语言写推理引擎-12个真实引擎的证据]] [[02-Python为什么还没被淘汰]] [[03-C++路线-llama.cpp的取舍]]
