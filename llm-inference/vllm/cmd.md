# vLLM 命令行与部署命令(cmd)

> 把 vLLM 从「装环境 → 起服务 → 发请求 → 压测」这条命令链一次讲透:每条命令在做什么、参数为何这么填、坑在哪。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/vllm/README]] [[llm-inference/vllm/服务启动参数]] [[llm-inference/README]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 你想干的事 | 看哪一节 | 关键命令/动作 |
|---|---|---|
| 搞清楚 cmd 这条链整体长啥样 | §0 §1 | docker → server → client |
| 用容器跑起来 | §2 | `docker run` 的每个 flag |
| 启动一个推理服务 | §3 | `vllm serve` / `api_server` |
| 让外部机器访问 | §4 | `--host` / `--port` / `--network=host` |
| 量化/并行/显存调优 | §5 | TP、`--gpu-memory-utilization`、`--max-model-len` |
| 验证服务通不通 | §6 | `curl` 探活 |
| 压测吞吐与时延 | §7 | `benchmark_serving.py` |
| 命令报错怎么办 | 常见坑 | OOM / 端口 / shm |

## 0. 一句话锚点

vLLM 的命令行使用 = **一台带 GPU 的机器上,用一条 `vllm serve`(或 `python -m vllm.entrypoints.openai.api_server`)命令把模型拉起成一个 HTTP 服务,再用 HTTP 客户端或自带的 benchmark 脚本去打它**。本文是这条命令链的「逐字说明书」,不是参数字典(完整参数见同目录 `服务启动参数.md`)。

> 注意:vLLM 迭代很快,**CLI 子命令名、参数名、默认值都可能随版本变化**。下文凡涉及确切名字/默认值的地方,均以「类别 + 作用 + 权衡」讲解,**具体以你所装版本的 `vllm serve --help` 和官方文档为准**。

## 1. 地基:这条命令链解决什么问题

朴素地用 HuggingFace `transformers` 写个 `model.generate()` 循环也能推理,但生产部署会遇到三个问题,vLLM 的命令行正是围绕它们设计的:

1. **环境难复现** → 用 **Docker 镜像**固化 CUDA / PyTorch / vLLM 版本(§2)。
2. **要服务化、要并发** → 用 **API Server** 把模型常驻成进程,对外暴露 OpenAI 兼容接口,内部用 PagedAttention + Continuous Batching 提吞吐(§3)。
3. **要量化效果** → 用 **benchmark 脚本**在固定数据集、固定请求速率下测吞吐/时延(§7)。

```
            一条命令链(本文主线)
 ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────────┐
 │ docker   │──▶│ vllm serve  │──▶│ curl 探活    │──▶│ benchmark压测 │
 │ run 起壳 │   │ 起 HTTP服务 │   │ 确认能回话   │   │ 看吞吐/时延   │
 └──────────┘   └─────────────┘   └──────────────┘   └───────────────┘
   §2             §3 §4 §5            §6                 §7
```

## 2. 第一步:用容器把环境立起来(`docker run`)

为什么先 Docker?因为 vLLM 对 CUDA / 驱动 / PyTorch 的版本组合很敏感,容器把这些钉死,避免「在我机器上能跑」。逐 flag 拆解一条典型命令:

```bash
docker run -dt --name vllm_env --restart=always --gpus all \
  --network=host \
  --shm-size 16g \
  -v /home/user/workspace:/workspace \
  -w /workspace \
  vllm/vllm-openai:latest \
  /bin/bash
```

| flag | 干什么 | 为什么这么填 / 权衡 |
|---|---|---|
| `-d` `-t` | 后台运行 + 分配伪终端 | 服务长期常驻用 `-d`;调试想交互用 `-it` |
| `--name` | 给容器命名 | 后续 `docker exec -it <name> bash` 进去操作 |
| `--restart=always` | 容器挂了自动拉起 | 生产常开;调试期可不加,免得反复重启掩盖错误 |
| `--gpus all` | 把宿主机所有 GPU 暴露进容器 | 也可 `--gpus '"device=0,1"'` 只给指定卡 |
| `--network=host` | 容器直接用宿主机网络栈 | **省去端口映射**,容器里监听的端口外部直接可达;代价是端口隔离性变差 |
| `--shm-size` | 共享内存大小 | **极易踩坑**:vLLM 多进程/多卡靠 `/dev/shm` 通信,默认 64MB 太小会报错,**调到几 GB ~ 16G** |
| `-v 宿主:容器` | 挂载目录 | 把模型权重、数据集挂进来,**别打进镜像**,否则镜像巨大 |
| `-w` | 容器内工作目录 | 进去就在 `/workspace` |
| 镜像名 | 用哪个镜像 | 官方 `vllm/vllm-openai` 直接带 vLLM;`nvcr.io/nvidia/pytorch` 是纯 PyTorch 底座,需自己再 `pip install vllm` |

> 坑:`--shm-size` 写小了,多卡张量并行时会出现 NCCL/共享内存相关报错;`--network=host` 与端口映射 `-p` 二选一,同时用会冲突。

## 3. 第二步:把模型拉起成服务(server 端)

vLLM 现在主推 **`vllm serve` 子命令**,等价于早期的 `python -m vllm.entrypoints.openai.api_server`。三种入口都见得到,理清它们的关系:

```
  vllm serve <model>                         ← 新版推荐,CLI 糖
        │  (内部就是调用)
        ▼
  python -m vllm.entrypoints.openai.api_server   ← OpenAI 兼容 /v1/* 接口
        ▲
        └── (老/简单接口) python -m vllm.entrypoints.api_server  ← 仅 /generate,非 OpenAI 风格
```

- **`openai.api_server`**:暴露 `/v1/completions`、`/v1/chat/completions`、`/v1/models` 等 **OpenAI 兼容**接口,能直接被 OpenAI SDK / LangChain 调用 → **生产首选**。
- **`api_server`(无 openai)**:只有简单的 `/generate`,本仓库旧示例(老 `cmd.md`)用的就是它 → 适合快速验证,新项目别用。

最小可用的服务启动:

```bash
# 新版写法(推荐)
vllm serve /workspace/model/your-model \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096

# 等价的模块写法
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/model/your-model \
  --host 0.0.0.0 --port 8000
```

> `--model` 可以是本地权重目录,也可以是 HuggingFace 仓库名(会自动下载,需联网/`HF_TOKEN`)。本地目录最稳,离线环境必须本地。

## 4. 让别人访问到:`--host` / `--port` / 网络

很多人「服务起来了但连不上」,根因在 host 绑定:

```
 外部客户端 ──HTTP──▶  容器内 vllm 进程监听 :8000
                          │
        --host 127.0.0.1  └─▶ 只接受本机回环,外部打不进来 ❌
        --host 0.0.0.0    └─▶ 接受所有网卡的连接,外部可达 ✅
```

| 参数 | 作用 | 实践建议 |
|---|---|---|
| `--host` | 监听哪个地址 | 要被别的机器访问就填 `0.0.0.0`;只本机自测可 `127.0.0.1` |
| `--port` | 监听端口 | 默认 8000;多实例同机部署要错开,如 8001/18001 |
| `--api-key` | 接口鉴权 token | 暴露到公网/共享集群务必加,客户端需带 `Authorization: Bearer <key>` |
| `--served-model-name` | 对外暴露的模型名 | 让客户端用一个干净别名,而非一长串本地路径 |

> 老 `cmd.md` 里 `--host 127.0.0.1 \n --port 8001`(端口行前少了反斜杠续行)是个真实坑:**多行命令每行末尾要 `\` 续行**,漏一个会让 `--port` 被当成新命令,导致服务起在默认端口而非你以为的 8001。

## 5. 调优类参数:并行、显存、长度(讲含义,不背默认值)

这些是「让大模型在你的卡上跑得起来、跑得快」的核心旋钮:

| 参数(类别) | 它控制什么 | 怎么权衡 |
|---|---|---|
| `--tensor-parallel-size`(TP) | 把单个模型**切到 N 张卡**上算 | 模型放不下单卡才需要 >1;一般取 = 单机 GPU 数;跨卡通信走 NVLink/IB,TP 越大通信开销越大 |
| `--pipeline-parallel-size`(PP) | 按层切分到多卡/多机 | 配合 TP 做更大规模;增加流水气泡,调度更复杂 |
| `--gpu-memory-utilization` | 允许 vLLM 用到的**显存比例**(如 0.9) | 越大留给 **KV Cache** 的空间越多 → 并发更高;太满易 OOM 或与其他进程争抢 |
| `--max-model-len` | 单条请求**最大上下文长度** | 决定 KV Cache 每条占多少;调小可省显存、提并发,但截断长请求 |
| `--max-num-seqs` | 同时在批里的**最大序列数** | 并发上限的硬约束,受显存与 `max-model-len` 共同制约 |
| `--dtype` | 计算精度(如 fp16/bf16) | bf16 数值更稳;A100/H100 用 bf16 常更好 |
| `--quantization` | 量化方式(如 awq/gptq/fp8 等) | 省显存、提吞吐,可能掉点;需配套量化权重 |
| `--enforce-eager` | 关掉 CUDA Graph,走 eager | 调试用;生产一般关掉它以享受图优化 |

显存怎么被吃掉的(理解 `--gpu-memory-utilization` 与 `--max-model-len` 为何联动):

```
  GPU 显存 = 模型权重(固定) + 激活(小) + KV Cache(大头,随并发&长度增长)
                                              ▲
                  --gpu-memory-utilization ───┘ 圈出能用的总盘子
                  剩给 KV 的 = 总盘子 - 权重
                  KV 容量 ≈ 剩余 / (单 token KV 大小 × max-model-len)
                                                         ▲
                                           --max-model-len 越大,能装的并发越少
```

数值直觉:粗略地,单 token 的 KV 缓存大小 $\approx 2 \times L \times H \times d_{head} \times \text{bytes}$($L$ 层数、$H$ 头数、2 为 K 和 V)。上下文越长、并发越多,KV 占用线性上涨——这就是 `--max-model-len` 调小能换并发的根本原因。

## 6. 第三步:探活——服务到底通没通(client 端最小验证)

起完服务先别急着压测,用 `curl` 确认能回话。OpenAI 兼容服务的探活:

```bash
# 列出已加载的模型(最轻量的探活)
curl http://localhost:8000/v1/models

# 发一条补全请求
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "your-model",
        "prompt": "你好,介绍一下你自己",
        "max_tokens": 64
      }'
```

排错决策:

```
 curl 报 Connection refused ─▶ 服务没起 / 端口不对 / host 绑了 127.0.0.1
 curl 卡很久才回         ─▶ 模型在加载,或并发被排队
 返回 404 /v1/...        ─▶ 用错了入口(api_server vs openai.api_server)
 返回 401                ─▶ 设了 --api-key,客户端没带 Authorization
```

## 7. 第四步:压测吞吐与时延(`benchmark_serving.py`)

vLLM 源码 `benchmarks/` 下自带压测脚本,用来回答「这套配置 QPS 多少、P99 时延多少」。典型用法:

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model /workspace/model/your-model \
  --tokenizer /workspace/model/your-model \
  --dataset-name sharegpt \
  --dataset-path /workspace/data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --host 0.0.0.0 --port 8000 \
  --request-rate 8
```

| 参数 | 含义 | 调它看什么 |
|---|---|---|
| `--backend` | 打哪个后端(vllm/tgi 等) | 跨框架对比时切换 |
| `--tokenizer` | 用哪个分词器统计 token 数 | 必须与模型一致,否则 token 统计偏差 |
| `--dataset-*` | 用什么数据集造请求 | ShareGPT 是常用的真实对话分布 |
| `--request-rate` | 每秒发多少请求(泊松到达) | **核心扫描变量**:从小到大扫,找吞吐拐点 |
| `--num-prompts` | 总共发多少条 | 太少不稳,太多耗时 |

关注的指标(与 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] 对齐):

```
   请求到达 ──▶ 排队 ──▶ 首 token ──▶ 逐 token 解码 ──▶ 完成
              │                │                       │
              └── 队列时延      └── TTFT(首 token 时延) └── TPOT(每 token 时延)
   整体: Throughput(tok/s, req/s)、E2E Latency(P50/P99)
```

- 提高 `--request-rate` 时,吞吐先上升后饱和,**饱和后再加压只会让排队时延暴涨**——这个拐点就是该配置的服务能力上限。
- 想提吞吐:调大 `--gpu-memory-utilization`、合理设 `--max-model-len`、上量化、加卡做 TP;想压时延:降并发、关无谓日志。

## 常见问题 / 坑

| 现象 | 根因 | 处理 |
|---|---|---|
| 启动即 CUDA OOM | `--gpu-memory-utilization` 过高 / `--max-model-len` 过大 / 模型放不下 | 调小利用率与长度;放不下就上 TP 多卡 |
| 多卡报共享内存/NCCL 错 | `--shm-size` 太小 | `docker run` 加 `--shm-size 16g` |
| 外部连不上服务 | `--host 127.0.0.1` 只绑回环 | 改 `--host 0.0.0.0`;确认 `--network=host` 或端口映射 |
| 多行命令只有部分生效 | 行尾漏了 `\` 续行符 | 每行(除最后一行)末尾补 `\`,且 `\` 后不能有空格 |
| 客户端 404 | 用了非 OpenAI 入口 | 生产用 `vllm serve` / `openai.api_server`,路径是 `/v1/...` |
| 401 未授权 | 服务设了 `--api-key` | 客户端加 `Authorization: Bearer <key>` |
| HF 模型下载失败 | 离线/无 token | 提前下到本地目录,`--model` 指向本地路径 |
| 改了参数没生效 | 改了旧 `api_server` 但实际跑的是 `vllm serve` | 确认你启动用的是哪个入口,参数名也可能随版本变 |

> 再次强调:**具体的子命令名、参数名、默认值请以你所装 vLLM 版本的 `vllm serve --help` 与官方文档为准**,本文聚焦「这条命令链每一步在做什么、为什么这么填」。

## 🔗 跳转链接

- 知识地图:[[00-知识地图]]
- vLLM 总览:[[llm-inference/vllm/README]]
- 启动参数字典:[[llm-inference/vllm/服务启动参数]]
- 推理总览:[[llm-inference/README]]
- 性能指标名词:[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
