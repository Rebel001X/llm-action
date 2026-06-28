# 昇腾 npu-smi 与 hccn_tool：NPU 运维监控全景

> 一句话定位：`npu-smi` 是昇腾 NPU 的 "nvidia-smi"，`hccn_tool` 是昇腾 RDMA 网卡的 "ip + ping + ethtool"，二者合起来是你在昇腾集群上排障、监控、巡检的两把瑞士军刀。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 你将学到 | 关键命令 / 概念 |
| --- | --- | --- |
| 0 | 一句话锚点 | npu-smi / hccn_tool 各管什么 |
| 1 | 地基：NPU 软硬件栈 | Device / Chip / HBM / DDR / AICore |
| 2 | 显存与算力监控 | `npu-smi info -t common / usages` |
| 3 | 设备共享（虚拟化） | `-t device-share` |
| 4 | 闪存与内存详情 | `-t flash` / `-t memory` |
| 5 | RDMA 网卡：状态/IP/路由 | `hccn_tool -status/-ip/-route` |
| 6 | RDMA 连通性自检 | `hccn_tool -ping` + `/etc/hccn.conf` |
| 实操 | 命令速查 | 全部原始命令汇总 |
| 坑 | 常见问题 | 单位/编号/HBM 满载等 |

## 0. 一句话锚点

- **`npu-smi`**：查 NPU 的「身体指标」——算力利用率、显存（HBM）占用、温度、功耗、频率、闪存。对标 NVIDIA 的 `nvidia-smi`。
- **`hccn_tool`**：查 NPU 之间「神经网络」——每张卡自带的 RDMA 网卡（RoCE）状态、IP、路由、连通性。这是昇腾做多机多卡集合通信（HCCL）的物理底座。
- 一句话：**算力看 `npu-smi`，互联看 `hccn_tool`**。

> 官方文档参考：https://support.huawei.com/enterprise/zh/doc/EDOC1100288566/e39bbfe6

## 1. 地基：先搞懂昇腾的硬件层级

调命令前，必须分清几个名词，否则 `-i`、`-c` 参数会用错。

```
   一台服务器（昇腾 AI 服务器，如 Atlas 800）
   ┌─────────────────────────────────────────────┐
   │  NPU 0   NPU 1   NPU 2  ...  NPU 7           │  ← npu-smi -i <NPU ID>
   │   │        │       │          │              │
   │  Chip0   Chip0   Chip0       Chip0           │  ← npu-smi -c <Chip ID>
   │   ├ AICore × 20 （做矩阵/向量运算的核）       │
   │   ├ HBM 32GB   （高带宽显存，放权重/激活）    │
   │   ├ DDR        （片上控制内存）               │
   │   └ RDMA 网卡 eth0..eth7（RoCE，做跨卡通信）  │  ← hccn_tool -i <设备号>
   └─────────────────────────────────────────────┘
```

- **NPU ID（`-i`）**：一张昇腾加速卡的编号，0~7 是典型的 8 卡机。
- **Chip ID（`-c`）**：一张卡上可能有多个 Die；训练卡常见 `Chip Count : 1`。
- **AICore**：昇腾达芬奇架构里做 Cube（矩阵乘）+ Vector 运算的核，原文里每芯片 `Aicore Count : 20`。它是「算力利用率」的来源。
- **HBM**：High Bandwidth Memory，相当于 GPU 显存，原文 `HBM Capacity(MB) : 32768`（= 32GB）。**大模型放不下/放得下，看的就是它。**
- **DDR**：片上 DDR，给控制 CPU（Ctrlcpu/Aicpu）用，容量小（原文 15079MB），别和 HBM 混。

> 为什么要分这么细？因为推理/训练时「卡住了」往往是 **HBM 满（OOM）** 或 **AICore 利用率低（瓶颈在别处）**，而跨卡训练慢往往是 **RDMA 链路**。诊断入口不同，工具也不同。

## 2. 显存与算力监控：`npu-smi info -t common`

最常用的体检命令。`-t common` 取「常用指标」，`-i 1` 指定 NPU 1。

```
 npu-smi info -t common -i 1
        NPU ID                         : 1
        Chip Count                     : 1

        Chip ID                        : 0
        Memory Usage Rate(%)           : 0
        HBM Usage Rate(%)              : 91      ← 显存吃了 91%，逼近 OOM
        Aicore Usage Rate(%)          : 6        ← 算力只用了 6%
        Aicore Freq(MHZ)               : 1800
        Aicore curFreq(MHZ)            : 1800
        Aicore Count                   : 20
        Temperature(C)                 : 46
        NPU Real-time Power(W)         : 130.4

        Chip Name                      : mcu
        Temperature(C)                 : 40
```

再看一张「忙碌」的卡，对照着读：

```
npu-smi info -t common -i 7
        NPU ID                         : 7
        Chip ID                        : 0
        HBM Usage Rate(%)              : 82
        Aicore Usage Rate(%)          : 70       ← 算力 70%，真正在干活
        Temperature(C)                 : 66       ← 温度更高，符合
        NPU Real-time Power(W)         : 331.1    ← 功耗 331W，远高于 NPU1
```

**怎么读这两张卡（核心思维）：**

| 指标 | NPU 1 | NPU 7 | 含义 |
| --- | --- | --- | --- |
| HBM 使用率 | 91% | 82% | 显存占用，>90% 警惕 OOM |
| AICore 使用率 | 6% | 70% | 算力打满程度，越高越「在算」 |
| 功耗 | 130.4W | 331.1W | 算力越满功耗越高 |
| 温度 | 46°C | 66°C | 功耗高 → 温度高，物理自洽 |

> **关键洞察（面试常考）**：NPU 1 的状态是 **「显存高但算力低」**——这是经典的「显存占着、计算闲着」。常见原因：模型已加载但请求空闲、batch 太小、或被某算子（如频繁的 D2H 拷贝、通信等待）卡住。判断「卡是否真在干活」，要看 **AICore 使用率 + 功耗**，而不是 HBM。

ASCII 帮助记忆两类典型卡：

```
   HBM ████████████████████░ 91%      HBM ██████████████████░░ 82%
   AIC ██░░░░░░░░░░░░░░░░░░░  6%       AIC ██████████████░░░░░░ 70%
   →「显存满，算力闲」              →「显存+算力都在跑」
     疑似空转/被通信卡住               健康的计算负载
```

### 设备整体统计：`-t usages`

`-t common` 偏「芯片瞬时」，`-t usages` 给「设备级别的资源汇总」，多了 **带宽利用率** 和 **各类 CPU 利用率**：

```
npu-smi info -t usages -i 1

NPU ID                         : 1
Chip Count                     : 1

DDR Capacity(MB)               : 15079
DDR Usage Rate(%)              : 14
DDR Hugepages Total(page)      : 0
DDR Hugepages Usage Rate(%)    : 0
HBM Capacity(MB)               : 32768       ← 32GB 显存总量
HBM Usage Rate(%)              : 0
Aicore Usage Rate(%)          : 0
Aicpu Usage Rate(%)            : 0            ← AI CPU（跑非 Cube 算子）
Ctrlcpu Usage Rate(%)          : 0            ← 控制 CPU
DDR Bandwidth Usage Rate(%)    : 0
HBM Bandwidth Usage Rate(%)    : 0            ← 显存带宽利用率：memory-bound 看它
Chip ID                        : 0
```

> **为什么要看 HBM Bandwidth Usage Rate？** 大模型推理的 decode 阶段几乎都是 **memory-bound**（每生成一个 token 要把全部权重从 HBM 搬一遍）。如果 AICore 利用率不高但 HBM 带宽接近 100%，说明瓶颈是显存带宽而非算力——这时上量化/KV-Cache 优化才有用，加算力没用。对照 [[llm-optimizer/kv-cache]]、[[llm-compression/quantization/量化基础]]。

## 3. 设备共享（NPU 虚拟化）：`-t device-share`

```
npu-smi info -t device-share -i 1
```

- `device-share` 查询/管理一张 NPU 的「算力切分」状态（类似 NVIDIA MIG / vGPU 思路）。
- 适用场景：一张昇腾卡切给多个容器/租户共享，提升利用率。
- 排障价值：当某容器只看到「半张卡」的显存时，先用它确认是否开了共享切分。

> 共享是「空间复用」的省钱手段，但会牺牲隔离性与峰值性能；训练大模型一般 **不开** 共享、独占整卡。

## 4. 闪存与内存详情

### 闪存信息：`-t flash`

```
npu-smi info -t flash -i 0
        NPU ID                         : 0
        Flash Count                    : 1
        Flash ID                       : 1730504
        Manufacturer ID                : 0xC8
        Capacity(MB)                   : 64       ← 64MB 板载 Flash，存固件/启动配置
        Chip ID                        : 0
```

- 这块 Flash 是 **板卡固件存储**（类似主板 BIOS Flash），64MB，**不是给模型用的存储**，别和 HBM 混。
- 巡检价值：固件升级/Manufacturer ID 核对时用。

### 内存信息：`-t memory`

`-t memory` 比 `usages` 更细，给到 HBM 的时钟、温度、厂商：

```
npu-smi info -t memory -i 1
        NPU ID                         : 1
        DDR Capacity(MB)               : 0
        DDR Clock Speed(MHz)           : 0
        HBM Capacity(MB)               : 32768
        HBM Clock Speed(MHz)           : 1600     ← HBM 频率
        HBM Temperature(C)             : 35        ← 显存温度，过热会降频
        HBM Manufacturer ID            : 0x57
        Chip ID                        : 0
```

> **数值手算：HBM 带宽量级**。HBM2E 单 stack 位宽 1024-bit，等效数据速率取决于颗粒。以 1600 MHz 时钟、DDR（双倍）、位宽 W 估算单 stack 带宽 ≈ $W \times 2 \times f = 1024 \text{bit} \times 2 \times 1.6\text{e}9 / 8 \approx 410\,\text{GB/s}$。多 stack 叠加得到整卡上 TB/s 级别带宽——这正是大模型推理 memory-bound 时的「天花板」。（具体数字以官方 Spec 为准，这里只给量级直觉。）

## 5. RDMA 网卡：`hccn_tool` 看互联

昇腾每张卡自带一个 **RoCE（RDMA over Converged Ethernet）网口**，是多机多卡 HCCL 集合通信（AllReduce/AllGather）的物理通道。`hccn_tool -i <设备号>` 操作它。注意：**这里的 `-i 3` 是网卡设备号，不是 npu-smi 的 NPU ID 概念，别混。**

### 5.1 链路状态：`-status -g`（g = get）

```
hccn_tool -i 3 -status -g

Netdev status:Settings for eth3:
        Supported link modes:   40000baseCR4/Full ... 200000baseCR4/Full
        Supports auto-negotiation: No
        Supported FEC modes: None        RS
        Speed: 200000Mb/s          ← 200Gbps 链路，正常
        Duplex: Full
        Auto-negotiation: off      ← RoCE 直连一般关自协商
        Port: Direct Attach Copper ← DAC 铜缆直连
        Link detected: yes         ← 关键：链路 UP
```

排障三看：

| 字段 | 健康值 | 异常含义 |
| --- | --- | --- |
| `Link detected` | `yes` | `no` → 物理断链/光模块坏，集合通信会 hang |
| `Speed` | `200000Mb/s` | 掉到低速 → 协商/线材问题，训练吞吐暴跌 |
| `FEC modes` | `RS` | FEC 不匹配会导致误码/丢包 |

> **为什么 RDMA 这么重要？** 千卡训练的梯度同步走的就是这些网口。一根线 `Link: no` 或 speed 掉档，整组 AllReduce 被最慢的卡拖住（木桶效应），整机训练速度断崖。详见 [[ai-infra/网络/集合通信原语]]、[[ai-infra/网络/InfiniBand]]。

### 5.2 IP 与路由：`-ip -g` / `-route -g`

```
> hccn_tool -i 3 -ip -g
ipaddr:10.20.11.24
netmask:255.255.255.0

> hccn_tool -i 3 -route -g
Routing table:
Destination     Gateway   Genmask         Flags Metric Ref Use Iface
10.20.11.0      *         255.255.255.0   U     0      0   0   eth3
127.0.0.1       *         255.255.255.255 UH    0      0   0   lo
192.168.1.0     *         255.255.255.0   U     0      0   0   end3v0
192.168.2.0     *         255.255.255.0   U     0      0   0   end3v0
```

- 每张卡的 RDMA 口在 **独立子网**（这里 `10.20.11.0/24`），卡间直接二层/三层互通。
- 路由表确认：去 `10.20.11.0/24` 走 `eth3`（本卡 RDMA 口）。如果某卡 ping 不通对端，先用 `-route -g` 看有没有路由。

### 5.3 启动配置文件：`/etc/hccn.conf`

`hccn_tool -ip` 看的是「当前运行态」，**开机时的 IP 是从这个文件刷下去的**：

```
> cat /etc/hccn.conf   # RDMA 网卡 0-7 的配置
address_0=10.20.11.11
netmask_0=255.255.255.0
address_1=10.20.11.12
netmask_1=255.255.255.0
...
address_7=10.20.11.18
netmask_7=255.255.255.0
```

```
   /etc/hccn.conf  ──(开机/重启 hccn 服务)──▶  各 RDMA 网卡运行态 IP
      持久化配置                                   hccn_tool -i N -ip -g 看到的
```

> **坑**：临时用 `hccn_tool -i N -ip -s` 改的 IP 重启会丢；要持久化必须改 `/etc/hccn.conf`。8 张卡的 `address_0..7` 要规划在同一子网且不冲突，否则集合通信建链失败。

## 6. RDMA 连通性自检：`-ping -g`

配完 IP，验证两张卡能不能在 RDMA 平面互通：

```
> hccn_tool -i 3 -ping -g address 10.20.11.16
device 3 PING 10.20.11.16
recv seq=0,time=0.137000ms
recv seq=1,time=0.046000ms
recv seq=2,time=0.058000ms
3 packets transmitted, 3 received, 0.00% packet loss
```

- `-i 3` 从设备 3 出发，ping 对端 `10.20.11.16`（对照 `/etc/hccn.conf` 是 `address_5`）。
- **0.00% packet loss + 亚毫秒延迟（~0.05ms）** = RDMA 平面健康。
- 这是跑 HCCL / 训练前的标准自检：**先 ping 通，再起任务**，避免任务起来后才在 AllReduce 阶段 hang 住难定位。

```
   起训练前的网络自检流水线：
   hccn_tool -status -g   → 链路 UP & 200G？
        │ yes
   hccn_tool -ip/-route   → IP/路由对？
        │ yes
   hccn_tool -ping -g     → 0% 丢包？  →  放心起 HCCL 训练
```

## 实操：命令速查

```bash
# ── npu-smi：算力/显存监控 ──
npu-smi info -t common      -i 1   # 瞬时：HBM/AICore 使用率、温度、功耗、频率
npu-smi info -t usages      -i 1   # 设备汇总：含 HBM/DDR 带宽利用率、各 CPU 利用率
npu-smi info -t memory      -i 1   # 内存细节：HBM 容量/频率/温度/厂商
npu-smi info -t flash       -i 0   # 板载 Flash（固件存储）信息
npu-smi info -t device-share -i 1  # 设备共享（算力切分/虚拟化）状态

# ── hccn_tool：RDMA 网卡运维 ──
hccn_tool -i 3 -status -g                       # 链路状态/速率（ethtool 类）
hccn_tool -i 3 -ip    -g                        # 查 RDMA 口 IP/掩码
hccn_tool -i 3 -route -g                        # 查路由表
hccn_tool -i 3 -ping  -g address 10.20.11.16    # RDMA 平面连通性自检
cat /etc/hccn.conf                              # 8 张卡 RDMA IP 的持久化配置
```

记忆口诀：`npu-smi info -t <类型> -i <NPU号>`；`hccn_tool -i <设备号> -<动作> -g`（`-g`=get，`-s`=set）。

## 常见问题 / 坑

| 现象 / 问题 | 根因 | 排查 / 解法 |
| --- | --- | --- |
| HBM 91% 但 AICore 仅 6% | 显存占着、算力闲（空转或被通信/拷贝卡） | 看功耗+温度确认是否真算；查 batch、D2H 拷贝、通信等待 |
| 推理慢但 AICore 不满 | memory-bound（显存带宽是瓶颈） | 看 `-t usages` 的 HBM Bandwidth；上量化/KV-Cache 而非加算力 |
| 把 DDR 当显存看 OOM | DDR 是控制内存（~15GB），HBM 才是显存（32GB） | 显存只认 `HBM Capacity/Usage` |
| `-i` 用混了 | npu-smi 的 `-i`=NPU ID；hccn_tool 的 `-i`=网卡设备号 | 两套编号体系，别套用 |
| 改了 IP 重启就丢 | `hccn_tool -ip -s` 只改运行态 | 持久化必须写 `/etc/hccn.conf` |
| 训练在 AllReduce 处 hang | 某卡 RDMA 链路 down / 丢包 / 子网不通 | `-status -g` 看 Link、`-ping -g` 看丢包、`-route -g` 看路由 |
| `Link detected: no` | 物理断链/线材/光模块 | 换线、复位网口、查 FEC/Speed 是否协商一致 |
| 集合通信建链失败 | `/etc/hccn.conf` 8 卡 IP 不在同一子网或冲突 | 统一规划同子网、不重复 |
| 温度/功耗异常高 | 算力打满（正常）或散热故障 | 对照 AICore 使用率：算力低却高温 → 查散热 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 硬件与生态：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/GPU工作原理]]
- 网络与通信：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 显存与优化：[[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 推理与评测：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
