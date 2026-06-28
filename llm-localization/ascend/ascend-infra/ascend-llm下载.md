# 昇腾环境下大模型权重下载与搬运

> 在国内/内网昇腾（Ascend）服务器上，如何把 HuggingFace 上的大模型权重（Baichuan2 / ChatGLM3 / Qwen 系列）稳定、可断点续传地拉到本地，再在集群内部机器间高效拷贝。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[llm-inference/README]] · [[docs/transformer内存估算]]

## 阅读地图

| 小节 | 解决什么问题 | 关键词 |
|------|------------|--------|
| 0 锚点 | 一句话说清这篇在干嘛 | 镜像、续传、搬运 |
| 1 地基 | 为什么内网拉权重这么难 | HF 墙、分片、token |
| 2 镜像加速 | `HF_ENDPOINT` 改成 hf-mirror | 透明替换 |
| 3 huggingface-cli | 官方下载器 + 断点续传 + 后台 | `--resume-download` |
| 4 wget 逐文件 | 不用 CLI 也能拉的兜底方案 | `resolve/main` |
| 5 软链接陷阱 | 为什么要 `--local-dir-use-symlinks False` | 缓存 vs 真文件 |
| 6 集群内搬运 | 拉到一台后怎么分发到 NPU 机器 | rsync / scp |
| 实操 | 原始命令清单（可直接改用） | — |
| 坑 | token 泄露、断网、磁盘 | — |

## 0. 一句话锚点

**下载 = 把镜像源指好（`HF_ENDPOINT`）+ 用支持断点续传的工具（`huggingface-cli` 或 `wget`）+ 后台挂着别让 SSH 断开（`nohup … &`）+ 拉完在内网用 `rsync` 分发。**

模型动辄 14GB（7B）到 140GB（72B），任何一次网络抖动都可能前功尽弃，所以"续传"和"后台"是这篇的两个核心词。

## 1. 地基：为什么内网拉大模型权重很痛

三个叠加的困难，缺一个都不至于这么麻烦：

1. **网络可达性**：`huggingface.co` 在国内直连不稳定/被墙，昇腾服务器又常常在没有公网代理的内网机房。
2. **体积巨大且分片**：一个 7B fp16 模型 ≈ $7\times10^9 \times 2\text{ B} \approx 14\text{ GB}$；72B ≈ $72\times10^9\times2 \approx 144\text{ GB}$。HF 会把它切成 `pytorch_model-00001-of-00002.bin` 这样的分片，任何一片下崩都要能"接着下"。
3. **门控模型需要鉴权**：部分仓库（gated repo）要登录后的 `token` 才能拉。

```
   HuggingFace Hub (huggingface.co)  ← 国内直连：慢/断/墙
            │
            │  把域名透明换成镜像
            ▼
   hf-mirror.com  (HF_ENDPOINT)      ← 国内 CDN，稳定
            │
            │  huggingface-cli / wget （支持断点续传）
            ▼
   昇腾下载机 /home/aicc/model_from_hf/...
            │
            │  rsync / scp （内网千兆/万兆）
            ▼
   各 NPU 推理/训练机 192.x.16.x
```

> 关键认知：**下载机 ≠ 跑模型的机器**。通常一台有"相对好一点"外网的机器负责拉，再通过内网把权重分发到一堆昇腾卡机器。这就是为什么本篇最后一节是"拷贝模型"。

## 2. 镜像加速：HF_ENDPOINT 的透明替换

`huggingface_hub` 库读取环境变量 `HF_ENDPOINT`，把所有原本指向 `huggingface.co` 的请求改写到你指定的域名。**这是"换源"而不是"翻墙"**——下载逻辑、URL 路径全不变，只是主机名变了。

```bash
pip3 install -U huggingface_hub          # 装/升级官方库
export HF_ENDPOINT=https://hf-mirror.com # 关键：本次 shell 内所有 HF 请求走镜像
```

为什么每段下载命令前都重复 `export HF_ENDPOINT=…`？因为 `export` 只对**当前 shell 会话**有效。原文里每个 `nohup …` 块前面都重置一次，是为了防止换了终端/换了用户后忘记设置，导致又去直连 `huggingface.co`。想一劳永逸可写进 `~/.bashrc`。

数值直觉：直连 200 KB/s 拉一个 14GB 的 7B 模型要 $14\times1024\times1024 / 200 \approx 73400\text{ s} \approx 20$ 小时；镜像跑满 20 MB/s 则约 $720\text{ s} \approx 12$ 分钟。差两个数量级，这就是换源的意义。

## 3. huggingface-cli：官方下载器 + 断点续传 + 后台

最推荐的方式。`huggingface-cli download` 会自动处理分片、并发、校验。

```bash
huggingface-cli download \
  --token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \   # 门控模型才需要；非门控可省
  --resume-download baichuan-inc/Baichuan2-7B-Chat \ # repo_id（带组织名）
  --local-dir Baichuan2-7B-Chat \                    # 落地目录
  --local-dir-use-symlinks False                     # 见第5节
```

逐个参数拆原子：

| 参数 | 作用 | 为什么需要 |
|------|------|-----------|
| `--token hf_…` | 携带 HF 访问令牌 | gated repo 鉴权；公开模型可不带 |
| `--resume-download` | 断点续传 | 断网后重跑命令，已下好的分片**跳过**，不从头来 |
| `repo_id` | 仓库标识 | 注意写全 `baichuan-inc/Baichuan2-7B-Chat`，只写后半段会找不到 |
| `--local-dir DIR` | 下到指定真实目录 | 不写则进 `~/.cache/huggingface` |
| `--local-dir-use-symlinks False` | 落真文件而非软链 | 见第 5 节，搬运/隔离环境必须 False |

### 3.1 用 nohup + & 挂后台（核心技巧）

72B 模型要下几小时，SSH 一断进程就被 SIGHUP 杀掉、白下。标准做法：

```bash
nohup huggingface-cli download --token hf_xxx --resume-download \
  Qwen/Qwen-72B-Chat --local-dir Qwen-72B-Chat \
  --local-dir-use-symlinks False > qwen-72b.log 2>&1 &
```

```
nohup ……… > qwen-72b.log 2>&1 &
  │            │          │    └─ & 放后台，立刻交还终端
  │            │          └────── 2>&1 把 stderr 也并进同一日志
  │            └───────────────── 进度/错误写入 qwen-72b.log
  └────────────────────────────── 忽略 SIGHUP，SSH 断开也不死
```

之后 `tail -f qwen-72b.log` 看进度，`jobs`/`ps -ef | grep huggingface` 查进程。原文为每个模型单独建日志（`Baichuan2.log` / `chatglm3.log` / `Qwen1.5-14B-Chat.log` …），就是为了多个下载并行时互不串台、各查各的。

## 4. wget 逐文件下载：兜底方案

当 `huggingface-cli` 不可用，或只想补几个缺失文件时，可以直接对每个文件的 `resolve/main/<文件名>` URL 用 `wget` 拉。规律：

```
https://huggingface.co/<org>/<repo>/resolve/main/<filename>
                        └── Baichuan2-7B-Base ──┘ └ config.json 等
```

一个完整模型目录通常包含这几类文件（以 Baichuan2-7B-Base 为例）：

| 文件 | 作用 |
|------|------|
| `config.json` | 模型结构超参（层数/隐藏维/词表大小） |
| `configuration_baichuan.py` / `modeling_baichuan.py` | 自定义模型代码（`trust_remote_code`） |
| `generation_utils.py` / `generation_config.json` | 生成/解码默认参数 |
| `pytorch_model-0000X-of-0000N.bin` | **权重分片**（大头） |
| `pytorch_model.bin.index.json` | 分片索引，告诉框架每个权重张量在哪一片 |
| `tokenizer.model` / `tokenization_baichuan.py` / `tokenizer_config.json` | 分词器 |
| `special_tokens_map.json` | 特殊 token（bos/eos/pad）映射 |
| `quantizer.py` | 量化相关代码（Baichuan 自带 8bit/4bit 工具） |

> 注意 **Base vs Chat 的差异**：Base 版常是两个分片 `pytorch_model-00001/00002-of-00002.bin`，Chat 版（7B）可能是单文件 `pytorch_model.bin`，且多一个 `generation_config.json`。所以两份 wget 清单不能照抄——文件名是按真实仓库列的。

`wget` 默认就支持续传：加 `-c`（`wget -c <url>`）即可断点接着下。原文目录约定为 `/home/aicc/model_from_hf/<模型名>/`，与第 6 节搬运路径一致。

## 5. 软链接陷阱：为什么要 `--local-dir-use-symlinks False`

`huggingface_hub` 默认行为：真正的文件存在全局缓存 `~/.cache/huggingface/hub/`，`--local-dir` 里放的只是**指向缓存的软链接**。

```
默认 (symlinks True):
  ./Qwen-72B-Chat/pytorch_model-00001.bin ──► ~/.cache/huggingface/hub/.../blobs/abc123
                  (软链，几十字节)               (真文件，几十 GB)

False:
  ./Qwen-72B-Chat/pytorch_model-00001.bin   (就是真文件本体)
```

为什么搬运场景必须 `False`：
- 你 `scp`/`rsync` 软链目录时，**拷过去的是断掉的链接**，目标机没有那个缓存，模型直接用不了。
- 想删缓存省磁盘时，软链会全部失效。
- 设 `False` 后 `--local-dir` 是自包含的真实目录，可整包搬走。

代价：会占双份磁盘（缓存 + 本地各一份）。下载机磁盘紧张时，可下完后手动清 `~/.cache/huggingface`。

## 6. 集群内搬运：rsync / scp 把权重分发到 NPU 机器

下载机拉好后，用内网高速链路分发到真正跑昇腾卡的机器（原文 IP 形如 `192.x.16.210`）。

```bash
# rsync：推荐，支持断点续传与增量同步
rsync -P --rsh=ssh -r root@192.xxx.16.210:/home/model_from_hf/Qwen1.5-72B/ ./

# scp：简单直接，但中断要从头来
scp -r root@192.xxx.16.210:/home/model_from_hf/Qwen1.5-72B/ ./
```

为什么优先 `rsync` 而不是 `scp`：

| 维度 | `rsync -P` | `scp -r` |
|------|-----------|----------|
| 断点续传 | ✅ `-P`(=`--partial --progress`) 接着传 | ❌ 断了重头 |
| 增量同步 | ✅ 只传有差异的块 | ❌ 全量覆盖 |
| 进度显示 | ✅ 每文件进度条 | 基本没有 |
| 适合大模型 | ✅（72B 144GB 必选） | 仅小文件 |

`-r` 递归整个目录，`--rsh=ssh` 指定走 SSH 通道（与 scp 一样的鉴权/加密）。72B 在万兆内网（理论 1.25 GB/s）下传 144GB ≈ $144/1.25 \approx 115\text{ s}$，比从公网重下快几十倍——这就是"下一次、分发多份"策略的价值。

## 实操：原始命令清单（可直接套用）

### 环境准备

```bash
# 镜像站
# https://hf-mirror.com/

yum install python3-pip

virtualenv -p python3 venv-py3
source /home/aicc/venv-py3/bin/activate

pip3 install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```

### huggingface-cli 下载各模型（token 请替换为你自己的）

```bash
export HF_ENDPOINT=https://hf-mirror.com

# Baichuan2 系列
huggingface-cli download --token hf_xxx --resume-download Baichuan2-7B-Base --local-dir Baichuan2-7B-Base
huggingface-cli download --token hf_xxx --resume-download baichuan-inc/Baichuan2-7B-Chat --local-dir Baichuan2-7B-Chat --local-dir-use-symlinks False
nohup huggingface-cli download --token hf_xxx --resume-download baichuan-inc/Baichuan2-7B-Chat  --local-dir Baichuan2-7B-Chat  --local-dir-use-symlinks False > Baichuan2.log         2>&1 &
nohup huggingface-cli download --token hf_xxx --resume-download baichuan-inc/Baichuan2-13B-Chat --local-dir Baichuan2-13B-Chat --local-dir-use-symlinks False > Baichuan2-13B-Chat.log 2>&1 &

# ChatGLM3
nohup huggingface-cli download --token hf_xxx --resume-download THUDM/chatglm3-6b --local-dir chatglm3-6b-chat --local-dir-use-symlinks False > chatglm3.log 2>&1 &

# Qwen / Qwen1.5 系列
nohup huggingface-cli download --token hf_xxx --resume-download Qwen/Qwen-72B-Chat    --local-dir Qwen-72B-Chat    --local-dir-use-symlinks False > qwen-72b.log          2>&1 &
nohup huggingface-cli download --token hf_xxx --resume-download Qwen/Qwen1.5-7B-Chat  --local-dir Qwen1.5-7B-Chat  --local-dir-use-symlinks False > Qwen1.5-7B-Chat.log  2>&1 &
nohup huggingface-cli download --token hf_xxx --resume-download Qwen/Qwen1.5-14B-Chat --local-dir Qwen1.5-14B-Chat --local-dir-use-symlinks False > Qwen1.5-14B-Chat.log 2>&1 &
nohup huggingface-cli download --token hf_xxx --resume-download Qwen/Qwen1.5-72B-Chat --local-dir Qwen1.5-72B-Chat --local-dir-use-symlinks False > Qwen1.5-72B-Chat.log 2>&1 &
```

### wget 逐文件下载（Baichuan2-7B-Base）

```bash
cd /home/aicc
mkdir -p ./model_from_hf/Baichuan2-7B-Chat/
cd ./model_from_hf/Baichuan2-7B-Chat/
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/generation_utils.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model-00001-of-00002.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model-00002-of-00002.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/quantizer.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenizer.model
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/resolve/main/tokenizer_config.json
cd ../../
```

### wget 逐文件下载（Baichuan2-7B-Chat）

```bash
cd /home/aicc
mkdir -p ./model_from_hf/Baichuan2-7B-Chat/
cd ./model_from_hf/Baichuan2-7B-Chat/
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/configuration_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/generation_config.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/generation_utils.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/modeling_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/pytorch_model.bin
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/quantizer.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/special_tokens_map.json
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/tokenization_baichuan.py
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/tokenizer.model
wget https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/resolve/main/tokenizer_config.json
cd ../../
```

### 拷贝模型（内网搬运）

```bash
rsync -P --rsh=ssh -r root@192.xxx.16.210:/home/model_from_hf/Qwen1.5-72B/ ./
scp -r root@192.xxx.16.210:/home/model_from_hf/Qwen1.5-72B/ ./
```

## 常见问题 / 坑

| 现象 | 根因 | 解决 |
|------|------|------|
| 还是连 huggingface.co 很慢 | 新开 shell 没 `export HF_ENDPOINT` | 写进 `~/.bashrc`，或每段命令前重置 |
| SSH 一断下载就停 | 前台进程收到 SIGHUP | `nohup … &` 挂后台 + 日志 |
| 断网后重下从头来 | 没加续传 | CLI 用 `--resume-download`；wget 用 `-c` |
| 拷到 NPU 机模型加载报缺文件 | `--local-dir` 是软链，scp 拷了断链 | 下载时加 `--local-dir-use-symlinks False` |
| gated repo 403/401 | 缺 token 或没接受协议 | 带 `--token`，并先在网页 Accept 协议 |
| `trust_remote_code` 报错 | 漏下 `*_baichuan.py` 等自定义代码 | wget 清单里这些 `.py` 必须一并下 |
| 磁盘爆满 | 缓存 + local-dir 双份 | 下完清 `~/.cache/huggingface/hub` |
| **token 写进文档/日志** | 明文 `hf_…` 泄露 | 立即去 HF 后台吊销重置；用 `huggingface-cli login` 或 `HF_TOKEN` 环境变量，别写进脚本 |
| rsync 中途断 | 网络抖动 | `-P` 已开续传，重跑同命令即可 |

> 安全提醒：原始文档里把真实的 `hf_…` token 直接写进了命令。**HF token 等同账号密码**，一旦进版本库/日志就视为泄露——应当作废重申，并改用 `huggingface-cli login`（存到 `~/.cache/huggingface/token`）或临时 `export HF_TOKEN=…`。本文已用 `hf_xxx` 占位。

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 硬件底座：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/GPU工作原理]]
- 内网传输：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 下载后干嘛：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-compression/README]] · [[llm-compression/quantization/量化基础]]
- 显存/容量估算：[[docs/transformer内存估算]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 模型结构背景：[[llm-algo/transformer/模型架构]] · [[llm-algo/旋转编码RoPE]]
