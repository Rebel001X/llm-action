# NVIDIA DCGM —— GPU 集群监控基石

> DCGM（Data Center GPU Manager）是 NVIDIA 官方的数据中心级 GPU 监控/管理引擎，负责采集利用率、显存、温度、功耗、ECC、NVLink、Profiling 等遥测数据，并通过 dcgm-exporter 对接 Prometheus + Grafana 形成可观测体系。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/GPU工作原理]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[ai-infra/网络/InfiniBand]] · [[llm-inference/vllm/README]]

## 阅读地图

| 节 | 内容 | 你将带走 |
|---|---|---|
| 0 | 一句话锚点 | DCGM 是什么、解决什么 |
| 1 | 地基/前置 | nv-hostengine、组件分层、为什么不用 nvidia-smi |
| 2 | 架构与数据流 | hostengine ↔ exporter ↔ Prometheus ↔ Grafana |
| 3 | 卸载旧版本 | 为什么先停 hostengine（原文真料） |
| 4 | 安装：Ubuntu/CentOS | 包仓库、CUDA keyring（原文真料） |
| 5 | 容器化部署 | dcgm 与 dcgm-exporter 镜像（原文真料） |
| 6 | 运行 dcgm 容器 | SYS_ADMIN、端口 5555、健康检查（原文真料） |
| 实操 | 命令/配置速查 | 一页跑起来 |
| 坑 | 常见问题 | 少踩雷 |

## 0. 一句话锚点

**DCGM = 给 GPU 集群装的"心电监护仪"。** 一台机器跑 `nvidia-smi` 够用；但成百上千张卡做训练/推理时，你需要一个**常驻服务**持续采集每张卡的健康与性能指标，并把它们汇入时序数据库做告警、画板、容量规划。DCGM 就是这个常驻服务，dcgm-exporter 是它接 Prometheus 的"翻译官"。

## 1. 地基/前置

### 1.1 为什么不直接用 nvidia-smi？

`nvidia-smi`（见 [[nvidia-smi]]）是**一次性快照命令**：执行一次、打印一次、退出。监控需要的是：

- **常驻采集**：秒级持续拉取，而非人工敲命令。
- **低开销**：`nvidia-smi` 每次调用都要初始化 NVML 上下文，频繁轮询代价高；DCGM 用守护进程 `nv-hostengine` 维持一个长连接的采集会话。
- **深度指标**：DCGM 能拿到 `nvidia-smi` 拿不全的 **Profiling 指标**（SM 占用率、Tensor Core 活跃度、显存带宽利用率、PCIe/NVLink 吞吐），这些是判断"GPU 是不是真的在干活"的关键。
- **标准化导出**：直接吐 Prometheus 格式，接告警/看板。

```
    人盯屏幕              监控系统
  ┌──────────┐        ┌──────────────────────────┐
  │nvidia-smi│  vs    │ nv-hostengine(常驻)        │
  │ 一次快照  │        │   └ 秒级采集 → exporter    │
  │ 高开销轮询│        │       └ Prometheus 抓取    │
  └──────────┘        │           └ Grafana 看板    │
                      └──────────────────────────┘
```

### 1.2 三个核心组件

| 组件 | 角色 | 类比 |
|---|---|---|
| **nv-hostengine** | DCGM 后台守护进程，真正对接 NVML/驱动采集数据 | 传感器+采集卡 |
| **dcgmi** | 命令行客户端，连 hostengine 查指标（如 `dcgmi dmon`） | 现场读数表 |
| **dcgm-exporter** | 把 DCGM 指标转成 Prometheus 文本格式的 HTTP 端点 | 数据上云的网关 |

> `nv-hostengine` 默认监听 TCP **5555**。客户端（dcgmi、exporter）通过这个端口连它。

### 1.3 前置条件

- 已安装 NVIDIA 驱动（DCGM 依赖 NVML）。
- 包名统一叫 **`datacenter-gpu-manager`**（这就是 DCGM 的安装包名）。
- 容器场景需 NVIDIA Container Toolkit（`--gpus all` 才生效）。

## 2. 架构与数据流

DCGM 在监控栈中处于"最底层采集"，向上一路接到看板：

```
 物理层        采集层              导出层            存储层        展示/告警
┌──────┐   ┌──────────────┐   ┌──────────────┐  ┌──────────┐  ┌──────────┐
│ GPU  │──▶│ nv-hostengine│──▶│ dcgm-exporter│─▶│Prometheus│─▶│ Grafana  │
│ (NVML)│  │  :5555        │   │  /metrics    │  │ (TSDB)   │  │  看板     │
└──────┘   │  (DCGM核心)   │   │  HTTP :9400  │  │ 拉取scrape│  │ Alertmgr │
           └──────────────┘   └──────────────┘  └──────────┘  └──────────┘
              ▲
              │ dcgmi dmon（人工现场读数，旁路）
              └────────────
```

**数据流向**（拉模型 pull）：Prometheus 周期性 GET exporter 的 `/metrics` → exporter 内部向本机 hostengine 查询一批 DCGM field → hostengine 调 NVML 读 GPU → 返回时序点 → 落 Prometheus → Grafana PromQL 查询渲染。

为什么是"拉"而不是"推"？Prometheus 主动 scrape，使得**目标存活性本身就是一个监控信号**（拉不到=节点挂了），且新增/下线节点只需改服务发现，无需重配每台机器。

## 3. 卸载旧版本（原文真料）

升级前要清掉旧安装，**关键是先停掉 `nv-hostengine`**——否则文件被占用、服务残留会导致新版本采集冲突或端口占用。

确认 `nv-hostengine` 没在跑，用下面命令停止它：

```bash
# 启动nv-hostengine
# nv-hostengine
# 停止 nv-hostengine
sudo nv-hostengine -t
```

> `-t` = terminate（停止守护进程）。直接敲 `nv-hostengine`（无参数）则是**启动**它。

移除旧安装（以 RPM 系为例）：

```bash
sudo yum remove datacenter-gpu-manager
```

**为什么先停再删？** hostengine 持有对 NVML 的会话与监听端口 5555；删包但进程仍在，会出现"新版二进制装好了，但内存里跑的还是旧逻辑"的脏状态，必须 `-t` 让它优雅退出。

## 4. 安装

包源都来自 NVIDIA 的 CUDA 仓库，核心是**先把仓库/GPG key 配好，再装 `datacenter-gpu-manager`**。

### 4.1 Ubuntu（原文真料）

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y datacenter-gpu-manager
```

逐行拆解：

| 行 | 作用 | 为什么 |
|---|---|---|
| `distribution=...` | 拼出形如 `ubuntu2004` 的发行版标识 | `sed 's/\.//g'` 把 `20.04` 去掉点变 `2004`，匹配 NVIDIA 仓库目录命名 |
| `wget ...cuda-keyring` | 下载 NVIDIA 仓库 GPG 公钥包 | apt 校验包签名，缺 key 会报 `NO_PUBKEY` |
| `dpkg -i cuda-keyring...` | 安装 key 与仓库源 | 注册 NVIDIA apt 源到本机 |
| `apt-get update` | 刷新索引 | 让 apt 看到新源里的包 |
| `apt-get install datacenter-gpu-manager` | 装 DCGM | 同时装上 hostengine、dcgmi |

> `. /etc/os-release` 中开头的点 `.` 是 `source` 的简写，把该文件里的 `$ID`、`$VERSION_ID` 变量加载进当前 shell。

### 4.2 CentOS（原文真料）

> centos8 以上使用 dnf，centos7 建议还是使用 yum。

**centos7：**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
#或者使用wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install datacenter-gpu-manager
systemctl restart nvidia-dcgm.service && systemctl enable nvidia-dcgm.service

systemctl status nvidia-dcgm.service
```

- `systemctl enable` 让 `nvidia-dcgm.service` **开机自启**（节点重启后监控不丢）。
- `systemctl restart` 立即拉起服务，`status` 确认 `active (running)`。
- 这里的 `nvidia-dcgm.service` 就是 hostengine 的 systemd 封装；交给 systemd 管理后，进程挂了能自动重启。

**centos8（先备依赖）：**

```bash
yum install epel-release
yum install dnf
```

然后用 dnf 走标准仓库流程：

```bash
# Determine the distribution name
distribution=$(. /etc/os-release;echo $ID`rpm -E "%{?rhel}%{?fedora}"`)
# Install the repository meta-data and the CUDA GPG key
sudo dnf config-manager \
    --add-repo http://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-rhel8.repo
# Update the repository metadata
sudo dnf clean expire-cache
# install DCGM
sudo dnf install -y datacenter-gpu-manager
```

> `rpm -E "%{?rhel}%{?fedora}"` 求值出 RHEL/Fedora 主版本号（如 `8`），拼成 `rhel8` 这类标识——比 Ubuntu 多这一步是因为 RHEL 系发行版标识更复杂。`dnf clean expire-cache` 强制下次访问重新拉元数据，避免用到过期索引。

## 5. 容器化部署（原文真料）

NVIDIA 提供两类官方镜像，分工不同：

| 镜像 | 仓库 | 内含 | 用途 |
|---|---|---|---|
| `nvidia/dcgm` | hub.docker.com/r/nvidia/dcgm | hostengine + dcgmi | 跑 DCGM 核心服务，供客户端连 5555 |
| `nvidia/dcgm-exporter` | hub.docker.com/r/nvidia/dcgm-exporter | exporter | 暴露 Prometheus `/metrics`，给 K8s 监控用 |

```bash
# DCGM 核心（含 hostengine）
docker pull nvidia/dcgm:3.1.7-1-ubuntu20.04

# DCGM Exporter（接 Prometheus）
docker pull nvidia/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04
```

> exporter 镜像 tag `3.1.8-3.1.5-ubuntu20.04` 中两个版本号分别是 **exporter 自身版本**与**内嵌 DCGM 版本**——保持与底层 DCGM 兼容很重要。

## 6. 运行 dcgm 容器（原文真料）

### 6.1 访问 GPU Telemetry（需要 Profiling 指标 → 要 SYS_ADMIN）

> 在此场景中，DCGM 独立容器已使用以下命令启动，其中端口 5555 映射到主机，以便其他客户端可以访问容器中运行的 nv-hostengine 服务。
> 请注意，要收集分析指标（Profiling），需要向容器提供 SYS_ADMIN 功能：

```bash
docker run --gpus all \
   --cap-add SYS_ADMIN \
   -p 5555:5555 \
   nvidia/dcgm:3.1.7-1-ubuntu20.04
```

> 现在，诸如 `dcgmi dmon` 之类的客户端可以在控制台上传输 GPU Telemetry/指标。

**为什么 Profiling 要 `SYS_ADMIN`？** SM 占用率、Tensor Core 活跃度等深度指标依赖访问 GPU 的硬件性能计数器（perf counters）。读这些计数器需要更高的内核权限，容器默认被剥夺，必须 `--cap-add SYS_ADMIN` 显式授予。

```
            含 Profiling             不含 Profiling
          ┌──────────────┐        ┌──────────────┐
 权限      │ SYS_ADMIN ✅  │        │ 无需额外cap   │
 拿得到    │ SM/TensorCore│        │ 利用率/温度   │
 的指标    │ /显存带宽利用 │        │ /功耗/显存量  │
          └──────────────┘        └──────────────┘
```

### 6.2 仅做 GPU 健康检查（非特权，更安全）

> 在这种情况下，DCGM 不需要任何额外的 caps，并且可以以非特权方式运行：

```bash
docker run --gpus all \
   -p 5555:5555 \
   nvidia/dcgm:3.1.7-1-ubuntu20.04
```

> 现在用于报告运行状况的 DCGM API 可以通过连接到 DCGM 容器的客户端访问。

**取舍**：健康检查（XID 报错、ECC 错误、温度/功耗越界、NVLink 状态）不碰性能计数器，所以**不需要 SYS_ADMIN**。生产上若只为告警"卡是否健康"，用非特权模式更符合最小权限原则；只有要做性能下钻分析时才开 SYS_ADMIN。

## 实操 / 命令速查

```bash
# —— 守护进程 ——
nv-hostengine            # 启动 hostengine（前台/默认配置）
sudo nv-hostengine -t    # 停止 hostengine

# —— systemd 管理（RHEL 系）——
systemctl restart nvidia-dcgm.service && systemctl enable nvidia-dcgm.service
systemctl status  nvidia-dcgm.service

# —— 安装（Ubuntu）——
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb && apt-get update && apt-get install -y datacenter-gpu-manager

# —— 卸载 ——
sudo nv-hostengine -t
sudo yum remove datacenter-gpu-manager

# —— 容器 ——
docker pull nvidia/dcgm:3.1.7-1-ubuntu20.04
docker pull nvidia/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04
docker run --gpus all --cap-add SYS_ADMIN -p 5555:5555 nvidia/dcgm:3.1.7-1-ubuntu20.04   # 含Profiling
docker run --gpus all                     -p 5555:5555 nvidia/dcgm:3.1.7-1-ubuntu20.04   # 仅健康检查

# —— 客户端查指标 ——
dcgmi dmon               # 实时流式打印 GPU 遥测（详见 dcgmi.md）
```

### 关键指标读法（数值示例）

某卡训练时 exporter 抓到：`DCGM_FI_DEV_GPU_UTIL = 98`、`DCGM_FI_PROF_SM_ACTIVE = 0.30`、`DCGM_FI_DEV_FB_USED = 70000`（MiB）。

- `GPU_UTIL=98%` 看似满载，但 `SM_ACTIVE=0.30` 说明**SM 平均只有 30% 在算**——典型的"利用率虚高"。GPU_UTIL 只要有 kernel 在跑就计数，而 SM_ACTIVE 才反映真实算力占用。
- 若 $\text{SM\_ACTIVE} \ll \text{GPU\_UTIL}$，多半是 **kernel 太小/启动开销大/数据供给不足（dataloader 瓶颈）**，是性能优化的信号。
- `FB_USED=70000 MiB ≈ 68.4 GiB`（$70000/1024$），接近 80GB 卡上限，警惕 OOM。

> 这就是为什么要 DCGM 而非只看 `nvidia-smi`：单看 GPU_UTIL 会被骗，SM_ACTIVE/Tensor Core 活跃度才是真相。详见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

## 常见问题 / 坑

| 现象 | 根因 | 解法 |
|---|---|---|
| 升级后采集异常/端口被占 | 旧 `nv-hostengine` 没停就删包 | 先 `sudo nv-hostengine -t` 再 `remove` |
| apt 报 `NO_PUBKEY` / 找不到包 | 没装 cuda-keyring 或没 `apt-get update` | 先 `dpkg -i cuda-keyring...` 再 update |
| `distribution` 变量拼错（如 `ubuntu20.04`） | 忘了 `sed 's/\.//g'` 去点 | 必须去掉点变 `ubuntu2004` |
| 容器里 Profiling 指标全为 0/N/A | 缺 `--cap-add SYS_ADMIN` | 加该 cap（仅性能分析需要） |
| 客户端连不上容器 5555 | 没做 `-p 5555:5555` 端口映射 | 补端口映射；确认 hostengine 在容器内已起 |
| 没装 `--gpus all` | 容器看不到 GPU | 装 NVIDIA Container Toolkit + `--gpus all` |
| 节点重启后监控丢失 | 服务没设开机自启 | `systemctl enable nvidia-dcgm.service` |
| GPU_UTIL 高但训练慢 | 只看 GPU_UTIL，未看 SM_ACTIVE | 用 DCGM Profiling 指标定位真实利用率 |
| exporter 与底层 DCGM 版本不配 | tag 里两段版本不兼容 | 选 `dcgm-exporter` 时对齐内嵌 DCGM 版本 |
| centos8 缺 dnf/config-manager | 没装 epel/dnf 插件 | 先 `yum install epel-release dnf` |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 同目录：[[nvidia-smi]] · [[dcgmi]] · [[nvidia-smi-dmon]] · [[gpu-env-var]] · [[NVIDIA-Nsight-Systems性能分析]]
- 硬件原理：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 网络/通信：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 性能/评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 推理/训练：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-train/README]]
