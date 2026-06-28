# 云 GPU 服务器：从零搭建 LLM 开发/训练/推理环境

> 一句话定位：把一台「裸的」云 GPU 实例（以优云智算/UCloud 为例）改造成可跑大模型训练与推理的工作机——挂载公共模型、装好框架、起 Jupyter、配 git/数据传输，并理解每一步「为什么这么做」。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]] · [[llm-inference/vllm/README]] · [[ai-infra/算力/GPU工作原理]]

## 阅读地图

| 章节 | 你将学到 | 关键词 |
| --- | --- | --- |
| 0 | 一张图看懂一台 LLM 工作机长什么样 | 锚点 |
| 1 | 为什么云 GPU 机器需要「二次改造」 | 镜像/驱动/网络 |
| 2 | 公共模型目录 `/model/` 怎么挂、怎么用 | 只读软链/ModelScope |
| 3 | conda 环境与依赖：版本对齐的坑 | torch/CUDA/transformers |
| 4 | Jupyter Lab 远程起服务 + 内核注册 | nohup/token/端口 |
| 5 | 数据进出：scp / rsync / 对象存储 | 带宽账 |
| 6 | git 配置与多机协作 | user/SSH |
| 7 | 显存与磁盘账：实例选多大 | 数值手算 |
| — | 一键 setup 脚本骨架 | bash |
| — | 排错对照表 | 局限 |

## 0. 一句话锚点

**一台能跑 LLM 的服务器 = 「正确的 GPU 驱动 + 对齐版本的深度学习框架 + 已下载好的模型权重 + 可远程访问的工作界面 + 顺畅的数据通道」五件套。** 缺任何一件，要么跑不起来、要么慢得离谱、要么半夜断线丢实验。

```
                ┌──────────────────────── 云 GPU 实例 ────────────────────────┐
                │                                                              │
  你的笔记本 ──SSH/HTTPS──►  Jupyter Lab (:8888, token 鉴权)                   │
                │                    │                                         │
                │            ┌───────┴────────┐                               │
                │            │  conda env py310│  ← torch / transformers / vllm│
                │            └───────┬────────┘                               │
                │                    │                                         │
                │   ┌────────────────┼──────────────────┐                     │
                │   │                │                   │                     │
                │  GPU(A800/H800) ── NVLink/PCIe ── 系统内存 ── NVMe 本地盘    │
                │   ▲                                    ▲                     │
                │   │ CUDA Runtime + Driver              │ /model/ (只读公共模型)│
                │   │                                    │ workspace/ (你的代码) │
                └───┼────────────────────────────────────┼─────────────────────┘
                    │                                    │
              scp/rsync 进出数据 ◄────────────────► 对象存储 / 数据集
```

> 本文以优云智算（UCloud 系）实例为蓝本，里面出现的具体 IP、token、路径以你实例控制台为准；命令的「形」通用，「值」请替换。

## 1. 地基：为什么云 GPU 机器要「二次改造」

你买到的实例通常已经预装了某个镜像（如 Ubuntu + CUDA + 一个基础 conda）。但直接拿来跑大模型几乎一定会踩三类坑，理解它们才知道后面每步在补哪个洞：

1. **驱动 vs CUDA Toolkit vs 框架编译版本，三者必须兼容。** GPU 驱动决定「这台机器最高能用哪个 CUDA」；你 `pip install` 的 torch 是按某个 CUDA 版本预编译的（如 `torch==2.4.0` 默认带 cu121）。如果框架要求的 CUDA 高于驱动支持上限 → 运行时报 `CUDA error: no kernel image` 或干脆 import 就崩。**先 `nvidia-smi` 看右上角 CUDA Version（=驱动支持上限），再选框架。**

2. **公共模型在 `/model/`，但不是你的工作区。** 平台把热门权重放在一个共享只读目录（省去每个人重新下几十 GB）。你要做的是「指向它」而非「拷贝它」——拷贝既慢又吃光本地盘。

3. **远程界面与会话保活。** SSH 一断，前台进程就被杀。所以 Jupyter / 训练任务要用 `nohup &` 或 `tmux/screen` 托管，否则关掉笔记本盖子，跑了 3 小时的训练就没了。

```
   镜像自带                你要补的
 ┌──────────┐          ┌────────────────────┐
 │ OS+驱动   │          │ conda env(版本对齐) │
 │ 基础CUDA  │  ──►     │ 框架(torch/vllm)   │
 │ /model/   │          │ Jupyter+内核        │
 └──────────┘          │ git/数据通道        │
                        └────────────────────┘
   「能开机」     →           「能干活」
```

### 1.1 上机第一组体检命令（为什么先做）

```bash
nvidia-smi                 # 看 GPU 型号/数量/显存/驱动支持的 CUDA 上限/当前占用
nvcc --version             # 看 CUDA Toolkit 版本（编译用，可能和上面不同）
df -h                      # 看各盘剩余空间——本地 NVMe 往往才是真正能写的大盘
free -g                    # 看系统内存（数据加载/dataloader 吃这里）
lscpu | grep -E 'CPU\(s\)|Model name'   # CPU 核数（dataloader workers 上限参考）
```

「为什么」：这五条决定了你**能选多大的模型、开几路并行、batch 设多大、把代码和缓存放哪个盘**。跳过它们去装环境，等于不量尺寸就裁衣。

## 2. 公共模型目录 `/model/`：挂载与使用

优云智算把公共模型放在 `/model/`，典型路径形如：

```
/model/ModelScope/Qwen/Qwen3-0.6B
        └────┬───┘ └─┬─┘ └───┬───┘
          来源平台   组织    具体模型(含config/权重/tokenizer)
```

### 2.1 工作区初始化（沿用原始脚本，解释每行）

```bash
base_path=`pwd`                         # 当前目录作为根（建议在大盘下，如 /root 或 /data）

mkdir -p "$base_path/workspace"         # 你的总工作区
mkdir -p "$base_path/workspace/code"    # 代码区，和「数据/模型缓存」分开放便于备份
```

> 为什么分目录：`/model/` 只读且公共（不能改），`workspace/` 才是你能写的地方。把代码、checkpoint、下载缓存分开，迁移/打包/清理时一目了然，也避免误删公共权重（其实你也删不动）。

### 2.2 直接用公共权重（不要拷贝！）

transformers 既能吃模型名（联网下载），也能吃**本地绝对路径**。机器上有 `/model/` 就直接喂路径，省下几十 GB 下载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/model/ModelScope/Qwen/Qwen3-0.6B"   # 直接指向公共只读目录
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,    # A800/H 系支持 bf16；老卡(如 V100)只能 fp16
    device_map="cuda",
)
```

> 关键创新点（工程层面）：「公共只读权重 + 本地可写工作区」是云平台的标准范式。它把「重而不变的东西（权重）」和「轻而常变的东西（你的代码/输出）」物理隔离——前者共享省盘、后者隔离防误删。

### 2.3 想微调/改权重？用软链或 `local_files_only`

公共目录只读，但你可以：

```bash
# 方式A：软链到工作区，再在工作区做你自己的拷贝（仅拷你要改的）
ln -s /model/ModelScope/Qwen/Qwen3-0.6B ~/workspace/Qwen3-0.6B-ref

# 方式B：完全离线，禁止任何联网下载（受限网络/避免偷偷拉模型很有用）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 3. conda 环境与依赖：版本对齐是头号大坑

### 3.1 为什么单独建环境

系统自带的 base 环境往往被平台锁定/混装。**给每个项目建独立 conda env**，是为了：依赖互不污染、可复现、出问题直接删环境重来不影响系统。

```bash
conda create -n py310 python=3.10 -y    # 3.10/3.11 兼容性最稳；3.12 部分库还没跟上
conda activate py310
```

### 3.2 安装框架（沿用并解释原脚本）

```bash
# 基础推理/可视化栈
pip install transformers accelerate seaborn matplotlib

# 指定 torch 版本——这是最关键的一步
pip install torch==2.4.0 transformers accelerate matplotlib
```

> ⚠️ 版本对齐三角（务必理解）：
> - `nvidia-smi` 的 CUDA Version = 驱动支持上限（如 12.4）。
> - `torch==2.4.0` 默认带 cu121（CUDA 12.1 编译），只要**驱动上限 ≥ 12.1** 就能跑。
> - 若驱动较老（如只支持 11.8），要装对应轮子：`pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118`（具体 index-url 以 PyTorch 官网为准）。
>
> 一句话规则：**框架编译用的 CUDA ≤ 驱动支持的 CUDA**，否则跑挂。反之向下兼容没问题。

### 3.3 装完自检（为什么必须验一次）

```python
import torch
print(torch.__version__)              # 2.4.0
print(torch.cuda.is_available())      # True ← 这一条是底线，False 就别往下走了
print(torch.cuda.get_device_name(0))  # NVIDIA A800-SXM4-80GB 之类
print(torch.cuda.device_count())      # 看到几张卡（决定能不能多卡并行）
print(torch.version.cuda)             # 框架编译用的 CUDA（如 12.1）
```

> `is_available()` 为 False 的高频原因：装成了 CPU 版 torch（pip 默认在无 GPU index 时可能给 CPU 轮子）/ 驱动与 CUDA 不匹配 / 没 `conda activate`。

### 3.4 把环境注册成 Jupyter 内核（沿用原脚本）

```bash
conda deactivate
# 把 py310 这个环境装成 Jupyter 可选内核——否则 notebook 里用的还是 base，import 不到你装的包
python -m ipykernel install --user --name=py310

# 确认注册成功
jupyter kernelspec list
```

> 为什么这步常被忘：很多人「在终端 pip 装好了，notebook 里却 ModuleNotFoundError」——因为 notebook 默认连的是 base 内核，没连 py310。注册内核 + 在 notebook 右上角选 `py310` 才闭环。

```
   终端 (py310)  ──pip install──►  py310/site-packages
        │                               ▲
        │ ipykernel install             │ 选对内核才走这条线
        ▼                               │
   Jupyter 内核列表: [base, py310] ──────┘
        └─ notebook 右上角必须选 py310，否则连到 base 报 ModuleNotFound
```

## 4. Jupyter Lab 远程起服务

### 4.1 后台启动（沿用并逐项解释）

```bash
nohup jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888 \
      > jupyter.log 2>&1 &
sleep 2
tail -100f jupyter.log | grep http://
```

逐参数「为什么」：

| 参数 | 作用 | 不写会怎样 |
| --- | --- | --- |
| `nohup ... &` | 脱离终端后台运行 | SSH 一断，Jupyter 被杀，会话全丢 |
| `--allow-root` | 允许 root 启动 | 云实例常以 root 运行，不写直接拒绝启动 |
| `--no-browser` | 不在服务器开浏览器 | 服务器没图形界面，开了也没用 |
| `--ip=0.0.0.0` | 监听所有网卡 | 默认只听 127.0.0.1，外网连不进来 |
| `--port=8888` | 指定端口 | 端口被占就换 8889 等 |
| `> jupyter.log 2>&1` | 把含 token 的日志落盘 | 否则拿不到访问 URL/token |

启动后日志里会出现形如：

```
http://117.50.213.xxx:8888/lab?token=e6d93f34f936c4a485d06f6ca267614ae09e585497adbeae
```

把 `117.50.213.xxx` 换成你实例的公网 IP（控制台可查），在本地浏览器打开即可。`token=...` 是一次性鉴权串，**别贴到公开仓库**。

### 4.2 连不上的排查顺序（带「为什么」）

```
浏览器打不开 8888？按这条链路从近到远查：
  1) 进程在吗      → ps -ef | grep jupyter          (没在 → 看 jupyter.log 报错)
  2) 端口在听吗    → ss -ltnp | grep 8888           (没听 → ip/port 写错或启动失败)
  3) 安全组放行吗  → 控制台「安全组/防火墙」放行 8888入站  (最常见！云默认不开)
  4) IP 对吗       → 用公网 IP，不是内网 10.x/172.x
  5) token 对吗    → 直接复制日志里整条 URL，别手敲
```

> 经验：90% 的「Jupyter 连不上」是**第 3 步安全组没放行端口**，而不是命令写错。

## 5. 数据进出：scp / rsync / 对象存储

### 5.1 scp（沿用原命令，解释方向）

```bash
# 把本地 lambada 数据集递归上传到服务器工作区
scp -r lambada root@ucloud:/root/workspace/data
       │        │      └────────── 目标：用户@主机:远程路径
       │        └──────────────── ucloud 是你 ~/.ssh/config 里配的别名
       └───────────────────────── -r 递归整个目录
```

> `root@ucloud` 里的 `ucloud` 是 SSH 别名（在 `~/.ssh/config` 里写好 `Host ucloud / HostName <ip> / User root / IdentityFile ~/.ssh/xxx`），就不用每次记长 IP。

### 5.2 大数据集优先 rsync（为什么）

```bash
rsync -avP --partial lambada/ root@ucloud:/root/workspace/data/lambada/
```

- `-a` 保留权限/时间戳；`-v` 显示进度；`-P`=`--partial --progress` **断点续传**。
- 比 scp 强在：**中断后再跑只补缺的部分**，几十 GB 数据传一半断网不用从头来。这正是 scp 的痛点。

### 5.3 通信量/时间手算（建立数量感）

> 估算公式：$t \approx \dfrac{S}{B}$，其中 $S$ 为数据量，$B$ 为实测带宽（注意 Mbps 是「比特/秒」，要 ÷8 换成字节）。

例：上传一个 50 GB 数据集，实测带宽 100 Mbps：

$$B = \frac{100\ \text{Mbps}}{8} = 12.5\ \text{MB/s},\quad t = \frac{50\times1024\ \text{MB}}{12.5\ \text{MB/s}} \approx 4096\ \text{s} \approx 1.1\ \text{小时}$$

> 结论：**大数据走公网 scp 很慢**。能用「同区域对象存储 + 内网拉取」（内网带宽常达 GB/s 级）就别走公网；或先压缩（`tar -I zstd`）再传。这就是为什么生产里数据通常落在对象存储而非靠 scp 搬。

## 6. git 配置与协作

```bash
git config --global user.email "liguodongiot@foxmail.com"
git config --global user.name "wintfru"
```

> 为什么先配这两条：不配 `user.email/user.name`，`git commit` 直接报错或提交署名乱。`--global` 写进 `~/.gitconfig`，整机生效一次到位。

多机/私有仓库还需：

```bash
ssh-keygen -t ed25519 -C "server-ucloud"   # 生成密钥，把 .pub 加到 GitHub/GitLab
git config --global credential.helper store # 或用 SSH 免每次输密码
```

## 7. 选多大实例：显存与磁盘账（数值手算）

> 推理显存粗估：$\text{显存} \approx \underbrace{P \times b}_{\text{权重}} + \underbrace{\text{KV Cache}}_{\text{随并发/序列长}} + \text{激活/碎片}$
> 其中 $P$ 为参数量，$b$ 为每参数字节数（fp16/bf16=2，int8=1，int4≈0.5）。

例 1：Qw3-0.6B 用 bf16 推理 → $0.6\text{B}\times 2\text{B} = 1.2\ \text{GB}$ 权重，单卡随便跑。

例 2：7B 模型 fp16 推理 → $7\times2 = 14\ \text{GB}$ 权重；再算 KV Cache：

> KV Cache（单序列）$\approx 2 \times L \times n_{layer} \times d_{model} \times b$（2 = K 和 V）。
> 设 $L=2048,\ n_{layer}=32,\ d_{model}=4096,\ b=2$：
> $2\times2048\times32\times4096\times2 \approx 1.07\ \text{GB/序列}$。

并发 16 路就是 $\approx 17\ \text{GB}$ KV Cache，加 14 GB 权重 + 激活/碎片，**一张 40 GB 卡吃紧、80 GB 从容**。这解释了为什么推理服务要上 [[llm-inference/KV-Cache优化]] 与 PagedAttention（见 [[llm-inference/vllm/README]]）。

磁盘账：一个 7B 模型权重 ~14 GB，加上数据集、checkpoint（训练时每个 checkpoint ≈ 权重×3，含优化器状态），**本地盘建议 ≥ 200 GB**，且确认大盘在 `/data` 或 `/root` 这种 NVMe 上（用 `df -h` 核对）。

## 一键 setup 脚本骨架（把上面串起来）

```bash
#!/usr/bin/env bash
set -euo pipefail   # 出错即停 / 用未定义变量即停 / 管道任一失败即停——避免「装到一半默默错」

# 0. 体检
nvidia-smi; nvcc --version || true; df -h; free -g

# 1. 工作区
base_path=$(pwd)
mkdir -p "$base_path/workspace/code" "$base_path/workspace/data"

# 2. 环境
conda create -n py310 python=3.10 -y
source activate py310 || conda activate py310
pip install torch==2.4.0 transformers accelerate seaborn matplotlib ipykernel
python -m ipykernel install --user --name=py310

# 3. 自检（失败就让脚本红着退出）
python - <<'PY'
import torch; assert torch.cuda.is_available(), "CUDA 不可用，检查驱动/torch 版本"
print("OK:", torch.__version__, torch.cuda.get_device_name(0))
PY

# 4. git
git config --global user.email "you@example.com"
git config --global user.name  "you"

echo ">>> 就绪。起 Jupyter： nohup jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888 > jupyter.log 2>&1 &"
```

> 数字、邮箱、token、IP 均为占位，请按你实例替换；版本以 PyTorch / 平台官方为准。

## 评价 / 排错对照表 / 局限

| 现象 | 大概率原因 | 处置 |
| --- | --- | --- |
| `torch.cuda.is_available()=False` | 装成 CPU 版 / 没 activate / 驱动不匹配 | 重装 GPU 版 torch，核对 `nvidia-smi` CUDA 上限 |
| `CUDA error: no kernel image` | 框架 CUDA 版本 > 驱动上限 | 换更低 CUDA 编译的 torch 轮子 |
| notebook `ModuleNotFoundError` | 连到 base 内核没连 py310 | 注册内核 + 右上角切 py310 |
| Jupyter 外网连不上 | 安全组没放行 8888 | 控制台放行入站端口（最常见） |
| SSH 断后训练没了 | 前台跑没托管 | `nohup &` 或 `tmux/screen` |
| scp 巨慢 | 走公网带宽 | 改 rsync 续传 / 对象存储内网拉 / 先压缩 |
| 磁盘写满 | checkpoint/缓存堆在小系统盘 | 缓存与输出指到大 NVMe（`HF_HOME` 重定向） |

**局限与边界**：
- 本文以优云智算/UCloud 系实例为例，`/model/` 路径、安全组入口、内网带宽各平台不同，**具体以你控制台为准**。
- 命令中的版本号（`torch==2.4.0`、`python=3.10`）是示例，请按你拿到的驱动上限与项目要求选；**默认 index-url、CUDA 兼容矩阵以 PyTorch 官方为准**。
- 多机分布式训练（多节点、RDMA/IB 网络、NCCL 调参）超出本文范围，见 [[llm-train/pytorch/distribution/README]] 与 [[ai-infra/网络/集合通信原语]]。
- 本文聚焦「单机环境就绪」；真要上线推理服务，请进 [[llm-inference/vllm/README]] 与 [[llm-inference/PD分离]]。

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 训练侧：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]]
- 推理侧：[[llm-inference/vllm/README]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/PD分离]]
- 硬件/网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/集合通信原语]]
- 性能名词：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 内存估算：[[docs/transformer内存估算]]
