# MindIE：昇腾推理引擎全流程总览

> MindIE（Mind Inference Engine）是华为昇腾 NPU 上的端到端大模型推理套件，对标 GPU 侧的 vLLM/TGI/Triton，提供"权重转换 → 拉起服务 → OpenAI/TGI 兼容 API"的一条龙能力。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|---------|--------|
| 0  | MindIE 一句话是什么、在昇腾栈里的位置 | MindIE-Service / ATB / CANN |
| 1  | 前置地基：达芬奇架构、CANN、Docker 直通 NPU | davinci / hccn / driver |
| 2  | 软件分层：MindIE-Service / MindIE-LLM / ATB-Models | 三层栈 |
| 3  | 镜像获取与离线迁移 | ascendhub / docker save / rsync |
| 4  | 容器拉起：8 张卡 `--device` 直通逐项解释 | davinci0-7 / davinci_manager |
| 5  | 权重转换：bin → safetensors | convert_weights.py |
| 6  | FA vs PA 两种启动脚本 + 调度原理 | run_fa.py / run_pa.py / PagedAttention |
| 7  | config.json 全字段对照表 | npuDeviceIds / worldSize / npuMemSize |
| 实操 | 原文真实命令一键复制区 | —— |
| 坑  | 直通/转换/显存/端口踩坑表 | —— |

## 0. 一句话锚点

**MindIE = 昇腾上的"vLLM + Triton Server"**。它把一个 HuggingFace 权重，经过格式转换、图编译（ATB 算子）、PagedAttention 显存管理、连续批调度，最终暴露成一个监听 `1025` 端口、兼容 OpenAI `/v1/chat/completions` 的推理服务。

官方入口与模型支持列表：

- 文档总入口：https://www.hiascend.com/document/detail/zh/mindie/20RC2/index/index.html
- 模型支持列表：https://www.hiascend.com/software/mindie/modellist
- "什么是 MindIE"：https://www.hiascend.com/document/detail/zh/mindie/10RC1/description/whatismindie/mindie_what_0000.html

## 1. 地基：昇腾推理栈从硬件到引擎

要让 MindIE 跑起来，下面四层必须就位，缺一层服务都拉不起来：

```
   ┌───────────────────────────────────────────────┐
   │  应用层   OpenAI / TGI / Triton 兼容 HTTP API   │  ← 业务调用 (port 1025)
   ├───────────────────────────────────────────────┤
   │  MindIE-Service  请求调度 / 连续批 / KV 管理     │  ← 本文主角
   │  MindIE-LLM      模型并行 / 采样 / 后处理        │
   │  ATB-Models      昇腾加速算子库 (Ascend Trans   │
   │                  former Boost) + 模型脚本        │
   ├───────────────────────────────────────────────┤
   │  CANN            算子编译 / HCCL 集合通信 / 内存  │  ← 类比 CUDA+NCCL
   │  Driver          /usr/local/Ascend/driver        │  ← 类比 nvidia driver
   ├───────────────────────────────────────────────┤
   │  达芬奇 NPU       /dev/davinci0 ... davinci7      │  ← 类比 GPU
   └───────────────────────────────────────────────┘
```

**为什么要分这么多层？** 与 GPU 栈一一对应即可记住：达芬奇 NPU ≈ GPU，CANN ≈ CUDA，HCCL ≈ NCCL，ATB ≈ cuBLAS/FlashAttention 算子，MindIE-Service ≈ vLLM/Triton。理解了对应关系，GPU 上的推理直觉几乎可以平移过来。

## 2. 软件三层栈与目录约定

容器内的关键路径（原文真实路径，务必记住）：

| 路径 | 含义 |
|------|------|
| `/home/HwHiAiUser/mindie-service_1.0.RC1_linux-aarch64/bin` | MindIE-Service 可执行与启动目录，`conf/config.json` 在此 |
| `/home/HwHiAiUser/atb-models/examples/convert` | 权重转换脚本目录（`convert_weights.py`） |
| `/home/HwHiAiUser/atb-models/examples` | `run_fa.py` / `run_pa.py` 推理启动脚本 |
| `/usr/local/Ascend/driver` | 宿主机驱动，容器需 `-v` 挂入 |
| `/workspace/aicc/model_from_hf/...` | HF 原始权重（由 `-v /home:/workspace` 映射进来） |

注意 `aarch64`：昇腾 800I A2 是 **ARM 架构**，镜像、scp 过来的 tar、自己编的轮子都必须是 arm64，x86 的二进制无法运行——这是跨机迁移最常踩的坑。

## 3. 镜像获取与离线迁移

镜像仓库：https://ascendhub.huawei.com/#/detail/mindie

**3.1 在线拉取**（原文命令保留）：

```bash
# 获取登录访问权限，输入已设置的"镜像下载凭证"
# 如果未设置或凭证超过 24 小时过期，请在登录用户名下拉处重新设置镜像下载凭证
docker login -u 157xxxx4031 ascendhub.huawei.com

# 下载镜像
docker pull ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64
```

> 镜像 tag `1.0.RC1-800I-A2-aarch64` 三段信息：版本 `1.0.RC1` / 硬件型号 `800I-A2` / 架构 `aarch64`。换硬件（如 300I Duo）或换架构，tag 要跟着换。

**3.2 离线迁移**（生产环境往往不通外网，先导出再拷贝）：

```bash
# 1. 把镜像打成 tar
docker save -o mindie-1.0.tar ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64

# 2. 直接 scp（小文件 / 网络稳定时）
scp root@192.xxx.16.211:/root/mindie-1.0.tar .

# 3. 断点续传（镜像动辄十几 GB，网络抖动时强烈推荐 rsync -P）
rsync -P --rsh=ssh -r root@192.xxx.16.211:/root/mindie-1.0.tar .
```

**为什么用 `rsync -P`？** MindIE 镜像通常 10~20GB，`scp` 中断要从头再来；`rsync -P`（`--partial --progress`）保留已传字节，断网后从断点继续，省时间也省带宽。导入用 `docker load -i mindie-1.0.tar`。

## 4. 容器拉起：NPU 直通逐项拆解

这是 MindIE 部署最关键的一条命令（原文完整保留），它的本质是**把宿主机的 NPU 设备节点和驱动透传进容器**：

```bash
docker run -it -u root --name=mindie_server_t35 --net=host --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v /var/log/npu/:/usr/slog \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /home:/workspace \
mindie_server:1.0.T35 \
/bin/bash

# 进入已运行的容器
docker exec -it mindie_server_t35 bash
```

逐项对照（**理解每一行"为什么"，出错才能定位**）：

```
--net=host / --ipc=host   共享宿主网络栈和 IPC，多卡张量并行靠共享内存通信，少了 ipc 会通信失败
/dev/davinci0..7          8 张 NPU 计算卡设备节点，张量并行 worldSize=8 时全要直通
/dev/davinci_manager      NPU 管理设备：枚举/复位/状态查询，少了 npu-smi 看不到卡
/dev/devmm_svm            SVM 共享虚拟内存设备，Host-Device 统一寻址，少了内存分配报错
/dev/hisi_hdc             Host-Device 通信通道（HDC），少了 Host 与 NPU 握手失败
-v .../driver             驱动用户态库 .so，容器内的 CANN 要 dlopen 它
-v /etc/hccn.conf         每张卡的 RoCE/IP 配置，多机/多卡集合通信(HCCL)按它建链
-v /var/log/npu/...       slog/profiling/dump 落到宿主，容器删了日志还在，便于排障
-v /home:/workspace       把宿主 /home 挂为 /workspace，权重就放在这里
```

直通设备数量必须 ≥ 你要用的并行度。**`worldSize=8` 却只 `--device` 直通了 4 张卡，启动时会卡在 HCCL 建链或直接报卡数不匹配**——这是 4 号坑。

## 5. 权重转换：bin → safetensors

MindIE 的 ATB-Models 推理脚本默认吃 `safetensors`，而很多 HF 权重是 `pytorch_model.bin`。需要先转：

```bash
# 进入容器，配置环境变量（加载 CANN/ATB 的 PATH 与 LD_LIBRARY_PATH）
cd /home/HwHiAiUser
source set_env.sh

# 进入转换脚本目录
cd /home/HwHiAiUser/atb-models   # ${llm_path}

# 示例：把 bin 转成 safetensor，输出保存在 bin 权重同目录下
python examples/convert/convert_weights.py --model_path ${weight_path}

# Baichuan2 实例（原文真实命令）
python examples/convert/convert_weights.py \
    --model_path /workspace/aicc/model_from_hf/Baichuan2-7B-Chat --from_pretrained False

python examples/convert/convert_weights.py \
    --model_path /workspace/aicc/model_from_hf/Baichuan2-7B-Chat
```

**`--from_pretrained` 是什么意思？**

| 取值 | 行为 | 适用 |
|------|------|------|
| `True`（默认） | 用 `AutoModel.from_pretrained` 加载后再存 safetensors，会跑一遍模型构图 | 权重需要重映射/补齐 buffer 的模型 |
| `False` | 直接读 `.bin` 张量字典，逐个 `save_file` 成 safetensors，不构图，更快更省内存 | 权重 key 已和 ATB 对齐、只想换容器格式 |

> **为什么要 safetensors 而不是 bin？** ① bin 是 pickle，加载时会执行任意代码，有安全风险；② safetensors 支持 mmap 零拷贝按需读，多卡切分时各 rank 只读自己那片，启动更快、峰值内存更低。转换产物与原 bin 同目录，转完原 bin 可保留也可删。

## 6. 两种启动脚本：Flash Attention vs Page Attention

ATB-Models 提供两条推理路径（原文保留）：

```
Flash Attention 启动脚本：${llm_path}/examples/run_fa.py
Page Attention  启动脚本：${llm_path}/examples/run_pa.py
```

**它们解决的是两个不同问题，别混淆：**

```
FlashAttention (run_fa.py)            PagedAttention (run_pa.py)
─────────────────────────            ──────────────────────────
解决"算 attention 时显存搬运慢"        解决"KV Cache 显存碎片浪费大"
把 softmax 分块在片上 SRAM 完成         把 KV Cache 切成定长 block 像分页内存
不实例化 N×N 注意力矩阵                 按需分配 block，请求间共享/复用
→ 省的是 attention 计算的中间显存       → 省的是 KV Cache 的总占用、提升并发
```

二者并不互斥：服务态 MindIE-Service 走的是 **PagedAttention 路线**（连续批 + 分页 KV），FlashAttention 作为底层算子被 PA 内部调用。`run_fa.py` 多用于单条/调试与算子验证，`run_pa.py` 才是高吞吐服务的基础。

PagedAttention 的显存账（直觉手算）：一个 7B 模型、`maxSeqLen=2560`、`maxBatchSize=200`，若按整段连续分配 KV，需要预留 `200 × 2560` 个槽位；分页后只为**实际生成到的长度**分配 block（`cacheBlockSize=128` 个 token 一块），平均序列若只用 512，则 KV 占用降到约 $\frac{512}{2560}\approx 20\%$，省下的显存直接换成更高 batch。详见 [[llm-optimizer/kv-cache]] 与 [[llm-optimizer/FlashAttention]]。

## 7. config.json 全字段对照（服务核心配置）

服务靠 `conf/config.json` 驱动（原文 `vim conf/config.json`）。下面是同目录下真实配置（`config-1.0.RC1.json` / `docker/qwen-72b.json`）的字段含义对照：

```json
{
  "OtherParam": {
    "ResourceParam": { "cacheBlockSize": 128, "preAllocBlocks": 8 },
    "LogParam":      { "logLevel": "Info", "logPath": "/logs/mindservice.log" },
    "ServeParam":    { "ipAddress": "0.0.0.0", "port": 1025, "maxLinkNum": 300,
                       "httpsEnabled": false }
  },
  "WorkFlowParam": {
    "TemplateParam": { "templateType": "Standard",
                       "templateName": "Standard_llama", "pipelineNumber": 1 }
  },
  "ModelDeployParam": {
    "maxSeqLen": 2560,
    "npuDeviceIds": [[0,1,2,3,4,5,6,7]],
    "ModelParam": [{
      "modelInstanceType": "Standard",
      "modelName": "qwen-72b",
      "modelWeightPath": "/home/aicc/model_from_hf/qwen-72b-chat-hf",
      "worldSize": 8, "cpuMemSize": 5, "npuMemSize": 8, "backendType": "atb"
    }]
  },
  "ScheduleParam": {
    "maxPrefillBatchSize": 50, "maxPrefillTokens": 8192,
    "maxBatchSize": 200, "maxIterTimes": 512,
    "maxPreemptCount": 200, "supportSelectBatch": false,
    "maxQueueDelayMicroseconds": 5000
  }
}
```

| 字段 | 作用 | 调参直觉 |
|------|------|---------|
| `port` | 服务监听端口（本仓库常见 1025/1125） | 多实例同机要错开端口 |
| `npuDeviceIds` | 用哪几张卡，二维数组每个子数组是一个实例 | 必须与 `--device` 直通一致 |
| `worldSize` | 张量并行卡数 | 72B 用 8，越大单卡显存压力越小但通信占比升高 |
| `npuMemSize` | 每卡留给 KV Cache 的显存(GB) | 调大→更多 block→更高并发，太大触发 OOM |
| `cpuMemSize` | CPU 侧 KV 换出空间(GB) | 抢占时把冷请求 KV 换到主存 |
| `maxSeqLen` | 单请求最大 (prompt+生成) 长度 | 超过会被截断/拒绝 |
| `cacheBlockSize` | PagedAttention 每块 token 数 | 128 为常见值，越小碎片越少但管理开销升高 |
| `preAllocBlocks` | 预分配块数 | 减少运行时首次分配抖动 |
| `maxPrefillTokens` | 一次 prefill 批的 token 上限 | 防止长 prompt 把 prefill 撑爆 |
| `maxBatchSize` | decode 阶段最大并发请求数 | 吞吐主旋钮，受 `npuMemSize` 约束 |
| `maxPreemptCount` | 最多可被抢占请求数 | 高并发下保护长请求不被饿死 |
| `supportSelectBatch` | 是否启用选择性组批 | 开启可提吞吐，关注公平性时关掉 |

**`npuMemSize` 与 `maxBatchSize` 是一对耦合旋钮**：可用 KV 显存 ≈ `npuMemSize`，能装下的并发数 ≈ 可用显存 ÷ 单请求 KV 占用。盲目调大 `maxBatchSize` 而不给够 `npuMemSize`，运行时会因 block 不足触发抢占（preempt）甚至失败。

## 实操：原文真实命令速查区

```bash
# 1) 拉起并进入容器（见第 4 节完整版）
docker run -it -u root --name=mindie_server_t35 --net=host --ipc=host \
  --device=/dev/davinci0 ... --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver ... \
  -v /home:/workspace  mindie_server:1.0.T35  /bin/bash
docker exec -it mindie_server_t35 bash

# 2) 进入服务 bin 目录、准备数据、改配置
cd /home/HwHiAiUser/mindie-service_1.0.RC1_linux-aarch64/bin
cp -r /workspace/token_input_gsm.csv .
vim conf/config.json

# 3) 模型权重位置（HF 原始权重）
cd /workspace/aicc/model_from_hf/chatglm3-6b-chat
#   /workspace/aicc/model_from_hf/Baichuan2-7B-Chat

# 4) 权重转换 bin -> safetensors
cd /home/HwHiAiUser && source set_env.sh
cd /home/HwHiAiUser/atb-models
python examples/convert/convert_weights.py \
  --model_path /workspace/aicc/model_from_hf/Baichuan2-7B-Chat --from_pretrained False

# 5) 推理脚本（二选一）
#   Flash Attention : examples/run_fa.py
#   Page  Attention : examples/run_pa.py

# 6) 性能压测（后台跑，日志重定向）
nohup python performance-stream-baichuan2.py > baichuan2.log 2>&1 &
```

服务起来后，调用方式见 [[mindie-api]]（OpenAI / vLLM / TGI / Triton / MindIE-Service 各版 curl）。最简健康检查：

```bash
curl "http://127.0.0.1:1025/v1/models"
```

## 常见问题 / 坑

| 现象 | 根因 | 解决 |
|------|------|------|
| `docker pull` 401/超时 | 镜像下载凭证超 24h 过期 | 重新 `docker login`，登录名下拉处重设凭证 |
| 跨机迁移后镜像跑不起 | x86 拉了 aarch64 镜像（或反之） | tag 末尾架构 `aarch64` 必须匹配目标机 CPU |
| scp 大镜像中断要重来 | 普通 scp 不续传 | 改用 `rsync -P --rsh=ssh`，断点续传 |
| 启动卡在 HCCL 建链 | `--device` 直通卡数 < `worldSize` | 直通的 davinci 数 ≥ 并行度；检查 `/etc/hccn.conf` 已挂 |
| 容器内 `npu-smi` 看不到卡 | 漏挂 `davinci_manager` / 驱动 `.so` | 补 `--device=/dev/davinci_manager` 与 `-v .../driver` |
| 多卡通信失败 | 未加 `--ipc=host` | 张量并行依赖共享内存，必须 `--ipc=host`（或调大 `--shm-size`） |
| 加载报权重格式错 | 脚本要 safetensors 但只有 bin | 先跑 `convert_weights.py` 转换 |
| 转换 OOM / 太慢 | `--from_pretrained True` 走了构图 | key 已对齐时用 `--from_pretrained False` 直转 |
| 启动 OOM | `npuMemSize`/`maxBatchSize` 过大 | 调小二者，按"可用显存 ÷ 单请求 KV"反推并发 |
| 端口被占 / 多实例冲突 | 同机多实例用同一 `port` | config.json 里错开 `port`（如 1025/1125） |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 同主题：[[mindie-api]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/硬件对比]]
- 推理引擎对照：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/解码策略]]
- 显存与算子：[[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[docs/transformer内存估算]]
- 模型与架构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 压缩量化（昇腾部署常配）：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 通信底座：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/InfiniBand]]
- 评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
