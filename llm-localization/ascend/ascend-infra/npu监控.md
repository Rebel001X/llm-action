# NPU 监控

> 把昇腾 NPU 的算力/显存/温度/功耗/网卡指标采集出来，喂给 Prometheus + Grafana，做集群级可观测——对标英伟达世界的 DCGM-Exporter。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 你想知道 | 看哪节 |
| --- | --- |
| NPU 监控到底监控什么、在昇腾栈哪一层 | 第 0、1 节 |
| 昇腾监控工具 ↔ 英伟达监控工具怎么对照 | 第 1 节对照表 |
| 单机手查（npu-smi / ascend-dmi）怎么用 | 第 2 节 |
| 指标是怎么从硬件一路冒到 Grafana 的（DCMI→Exporter→Prometheus） | 第 3 节 |
| 容器/K8s 集群里怎么部署监控 | 第 4 节 |
| 该盯哪些核心指标、阈值怎么定 | 第 5 节 |
| 从 GPU/DCGM 迁过来要改什么、有哪些坑 | 「迁移要点」节 |
| 常见报错速查 | 「常见问题」节 |

## 0. 一句话锚点

**NPU 监控 = 「采集昇腾芯片的运行态指标 + 暴露成 Prometheus 可抓取的格式 + 用 Grafana 可视化告警」的一整套可观测体系。** 单机临场排障用 `npu-smi`（对标 `nvidia-smi`），集群长期监控用 **NPU-Exporter**（对标 **DCGM-Exporter**）。所有指标最终都来自硬件之上的 **DCMI/DSMI** 设备管理接口（对标英伟达的 **NVML**）。

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

### 1.1 处于软件栈哪一层

监控不是一个独立框架，而是「横切」在整条栈上的能力——它靠最底层的设备管理接口（DCMI/DSMI）读硬件寄存器，再往上层层封装成命令行工具、Exporter 和大盘：

```
┌─────────────────────────────────────────────────────────┐
│  可视化/告警层   Grafana 大盘  +  Prometheus AlertManager  │  ← 看图、报警
├─────────────────────────────────────────────────────────┤
│  采集/暴露层     NPU-Exporter (HTTP /metrics)             │  ← 集群长期监控
│                 npu-smi / ascend-dmi (命令行)             │  ← 单机临场排障
├─────────────────────────────────────────────────────────┤
│  设备管理接口    DCMI / DSMI / ACL                        │  ← 监控数据的「源头」
├─────────────────────────────────────────────────────────┤
│  CANN 驱动+固件  Driver / Firmware                        │  ← 真正读硬件寄存器
├─────────────────────────────────────────────────────────┤
│  硬件层          昇腾 NPU（达芬奇 AI Core）+ HCCN 网卡     │
└─────────────────────────────────────────────────────────┘
```

要点：**监控数据的唯一真实来源是 DCMI/DSMI**。无论是 `npu-smi` 命令、NPU-Exporter，还是你自己写的 Python 脚本，最终都是调这套接口。接口之上的工具只是「换个包装」把数字呈现出来。

### 1.2 昇腾 ↔ 英伟达 监控生态对照表

这是从 GPU 世界迁过来最该先建立的心智图：

| 维度 | 英伟达（NVIDIA） | 昇腾（Ascend） | 说明 |
| --- | --- | --- | --- |
| 加速卡 | GPU | NPU | 算力硬件本体 |
| 底层管理接口 | NVML（NVIDIA Management Library） | DCMI / DSMI（设备控制/系统管理接口） | 监控数据的「源头库」 |
| 计算运行时 | CUDA | CANN | 监控工具随运行时一起装 |
| 单机命令行工具 | `nvidia-smi` | `npu-smi` | 临场看利用率/显存/温度 |
| 进阶诊断工具 | `dcgmi` / `nvidia-smi dmon` | `ascend-dmi` | 带宽/算力/功耗压测、故障诊断 |
| 集群指标 Exporter | **DCGM-Exporter** | **NPU-Exporter** | 暴露 Prometheus `/metrics` |
| 指标后端 | Prometheus | Prometheus | 两边通用，无需替换 |
| 可视化 | Grafana | Grafana | 两边通用，仅大盘/指标名不同 |
| K8s 设备插件 | `k8s-device-plugin` | **Ascend Device Plugin**（Volcano 生态） | 调度 + 配合监控 |
| 显存类型 | HBM / GDDR | HBM / DDR | 昇腾常分别上报 HBM 与 DDR |
| 高速互联网卡 | NVLink / IB（用 `nvidia-smi nvlink`、IB 工具） | HCCN 网卡（用 `hccn_tool`） | RoCE/参数面网络监控 |

**一句话记忆**：`nvidia-smi → npu-smi`、`NVML → DCMI`、`DCGM-Exporter → NPU-Exporter`，Prometheus + Grafana 两边照旧。

## 2. 单机手查：npu-smi 与 ascend-dmi

集群监控之前，先掌握单机怎么「肉眼看一眼」。

### 2.1 npu-smi —— 对标 nvidia-smi

随驱动一起安装，最常用。核心能力（具体命令与参数以华为昇腾官方文档（Ascend 社区）为准）：

- **看整体利用率/显存/温度/功耗**：查询某张卡的 AI Core 使用率、HBM 使用率、温度、实时功耗。这对应 `nvidia-smi` 主表里的 `GPU-Util / Memory-Usage / Temp / Power`。
- **看进程占用**：哪个进程占了多少显存（对标 `nvidia-smi` 下半部分的进程列表）。
- **看带宽使用率**：HBM 带宽使用率、DDR 带宽使用率——这是昇腾比 `nvidia-smi` 默认输出更细的地方。
- **看内存/闪存/网卡信息**：HBM 容量与时钟、闪存信息等。

> 关键认知：`npu-smi info` 里的 **AI Core Usage Rate** 才是「算力是否打满」的核心指标；**HBM Usage Rate** 是显存占用。两者一个看算、一个看存，别混淆——HBM 占满但 AI Core 很低，往往是「数据搬进来了但没在算」（典型的访存瓶颈或 host 侧卡住）。

### 2.2 ascend-dmi —— 对标 dcgmi 的诊断/压测

定位为「带宽测试、算力测试、功耗测试、故障诊断、拓扑检测」的进阶工具，底层同样调 DCMI/DSMI/ACL。它解决的是 `npu-smi` 解决不了的问题：**这张卡/这条互联链路「健不健康、性能够不够」**。新机器到货验收、怀疑某张卡掉速时用它。

## 3. 机制：指标怎么从硬件冒到 Grafana

集群监控的标准数据流，和 GPU 世界几乎一模一样，只是把 DCGM-Exporter 换成 NPU-Exporter：

```
   昇腾 NPU 硬件寄存器
        │ (驱动/固件读出)
        ▼
   ┌──────────────┐   每张卡的算力/显存/温度/功耗/带宽
   │ DCMI / DSMI  │   ← 监控数据「源头」(对标 NVML)
   └──────┬───────┘
          │ 调接口取值
          ▼
   ┌──────────────┐   把指标转成 Prometheus 文本格式
   │ NPU-Exporter │   暴露 HTTP  http://<ip>:<port>/metrics
   └──────┬───────┘   (对标 DCGM-Exporter)
          │ HTTP 周期性 scrape (pull 模型)
          ▼
   ┌──────────────┐   存时序数据 + 算告警规则
   │  Prometheus  │
   └──────┬───────┘
          │ PromQL 查询
          ▼
   ┌──────────────┐        ┌───────────────┐
   │   Grafana    │        │ AlertManager  │
   │   (大盘看图)  │        │  (温度/掉卡告警)│
   └──────────────┘        └───────────────┘
```

### 3.1 为什么是 Exporter + Pull 模型

Prometheus 是**拉（pull）**模型：它按配置的 `scrape_interval` 主动来 `/metrics` 抓数据。所以 NPU-Exporter 的职责很纯粹——**把 DCMI 读到的数字翻译成 Prometheus 认识的文本格式，挂在一个 HTTP 端口上等着被抓**。它本身不存数据、不画图、不告警，这些分别交给 Prometheus、Grafana、AlertManager。理解了这个分工，整套监控就不神秘了。

### 3.2 NPU-Exporter 暴露的指标长什么样

指标是「指标名 + 标签（labels）+ 数值」的形式。标签通常带 `id`（卡号）、`vdie_id`/`model_name` 等，便于在 PromQL 里按卡聚合。典型可观测维度（指标的确切名称以官方文档为准）：

- AI Core 利用率、AI CPU 利用率
- HBM/DDR 使用量、使用率、带宽使用率
- 芯片温度、HBM 温度
- 实时功耗、电压、频率
- 卡的健康状态（在位/掉卡/ECC 错误计数等）
- 网络（HCCN/RoCE）链路状态

> 这些维度和 DCGM-Exporter 暴露的 `DCGM_FI_DEV_GPU_UTIL / DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_GPU_TEMP / DCGM_FI_DEV_POWER_USAGE` 几乎一一对应，迁 Grafana 大盘时主要工作就是**把这些指标名替换掉**。

## 4. 容器与 K8s 集群里的监控部署

生产环境多在容器/K8s 里跑，监控也随之容器化。整体流程的「含义与依赖」如下（具体镜像名、yaml、版本以官方文档为准）：

```
宿主机: 装好 NPU 驱动+固件 (DCMI 才有数据源)
        │  必须先有 ascend-docker-runtime，容器才能透传 NPU 设备
        ▼
K8s 节点: 以 DaemonSet 部署 NPU-Exporter (每个节点一个)
        │  容器需挂载 /dev 下的 NPU 设备 + 驱动目录, 才能调 DCMI
        ▼
Prometheus: 通过 ServiceMonitor / 静态配置, 发现并抓取各节点 Exporter
        ▼
Grafana:    导入昇腾 NPU 大盘, 选 Prometheus 数据源
        ▼
告警:       AlertManager 配温度过高/掉卡/ECC 错误等规则
```

依赖链上的关键点：

1. **宿主机驱动是地基**：容器里的 Exporter 不可能凭空读到硬件，它依赖宿主机已正确安装 NPU 驱动/固件，且容器要通过 `ascend-docker-runtime` 把设备和驱动目录透传进去（参见 [[llm-localization/ascend/ascend-infra/ascend-docker-runtime]]）。这一步漏了，Exporter 起得来但**指标全空/全 0**。
2. **DaemonSet 一节点一个**：监控的是「本机的卡」，所以按节点部署，不是按 Pod。
3. **Prometheus 服务发现**：K8s 里通常用 ServiceMonitor（Prometheus Operator）或 Kubernetes SD 自动发现 Exporter，免去手填 IP。
4. **大盘可复用**：Grafana 本身两边通用，找官方/社区的昇腾大盘 JSON 导入即可。

## 5. 该盯哪些核心指标

| 指标 | 看什么 | 异常信号 |
| --- | --- | --- |
| AI Core 利用率 | 算力是否打满 | 训练时长期 < 50% → host/数据/通信瓶颈 |
| HBM 使用率 | 显存够不够 | 接近 100% → 有 OOM 风险 |
| HBM 带宽使用率 | 是否访存受限 | 高带宽 + 低 AI Core → 访存瓶颈 |
| 芯片/HBM 温度 | 散热是否正常 | 持续高温 → 降频掉速，需查机房散热 |
| 实时功耗 | 是否在正常区间 | 异常偏低常意味着没在干活 |
| 卡健康/在位状态 | 是否掉卡 | 数量变化 → 立刻告警 |
| ECC 错误计数 | 显存硬件健康 | 持续增长 → 硬件隐患 |
| HCCN 链路状态 | 集合通信网络 | Down → 多机训练直接挂 |

## 迁移要点 / 注意事项与坑

从 GPU + DCGM 体系迁到昇腾 + NPU-Exporter，**架构思路完全一样，换的是工具与指标名**：

1. **接口/工具名一一替换**：`nvidia-smi → npu-smi`、`dcgmi → ascend-dmi`、`DCGM-Exporter → NPU-Exporter`、`NVML → DCMI/DSMI`。脚本里凡是调 NVML 的地方，都要改成调 DCMI。
2. **指标名要重映射**：DCGM 的 `DCGM_FI_DEV_*` 系列指标名在昇腾这边是另一套命名。迁 Grafana 大盘 / Prometheus 告警规则时，**逐个把 PromQL 里的指标名换掉**是主要工作量，不是「装上就能用」。
3. **HBM 与 DDR 要分清**：昇腾常把 HBM 和 DDR 分开上报，做「显存占用」告警时别只盯一个；推理/训练真正吃的是 HBM。
4. **「AI Core 利用率」≠「卡忙」**：利用率高只代表 AI Core 在跑指令，不代表算得高效；要结合 HBM 带宽、功耗一起判断瓶颈在算还是在搬数据。
5. **容器透传是头号坑**：Exporter 在容器里跑却读不到指标，**九成是宿主机驱动没装好或设备/驱动目录没透传**（缺 `ascend-docker-runtime` 或挂载漏了 `/dev` 下设备）。先在宿主机上 `npu-smi` 能不能出数验证数据源，再排容器。
6. **网络面要单独监控**：多机训练靠 HCCN/RoCE 参数面网络，这部分要用 `hccn_tool` 看链路状态/ping（见 [[llm-localization/ascend/ascend-infra/ascend-npu-smi]]）。链路 Down 会让 HCCL 集合通信整体挂死，但 AI Core 指标可能看起来「正常」，要专门盯网络指标。
7. **版本对齐**：Exporter、驱动、CANN 之间有版本配套关系；版本不匹配会导致指标缺失或服务起不来。**具体配套版本与命令以华为昇腾官方文档（Ascend 社区）为准**，切勿凭印象拼版本号。

## 常见问题

| 问题 | 可能原因 | 排查方向 |
| --- | --- | --- |
| Exporter 起来了但 `/metrics` 指标全为 0/空 | 容器没透传设备、宿主机驱动异常 | 先在宿主机 `npu-smi info` 验证数据源；再查容器挂载与 runtime |
| Prometheus 抓不到 Exporter | 服务发现/端口/网络策略问题 | 检查 ServiceMonitor 选择器、端口、防火墙；手动 curl `/metrics` |
| Grafana 大盘空白 | 指标名不匹配 | 大盘是按 DCGM 指标名写的，需替换成昇腾指标名 |
| AI Core 利用率长期偏低 | host/数据/通信瓶颈，非卡的问题 | 看 HBM 带宽、功耗、CPU/IO；多机看 HCCN 链路 |
| 温度持续偏高、性能掉 | 散热不足导致降频 | 查机房散热/风扇；结合功耗、频率指标 |
| 显存看着没满却 OOM | 只看了 DDR 没看 HBM | 分别看 HBM 与 DDR 使用率 |
| 多机训练突然卡死 | HCCN/RoCE 链路 Down | `hccn_tool` 查链路状态/ping，查 HCCL 报错 |
| npu-smi 与 Exporter 数值对不上 | 采样时刻/聚合口径不同 | 属正常瞬时差异；关注趋势而非单点 |

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 硬件与生态：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/CUDA]]
- 网络与通信：[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]] · [[llm-localization/ascend/ascend-infra/HCCL]]
- 同目录工具：[[llm-localization/ascend/ascend-infra/ascend-npu-smi]] · [[llm-localization/ascend/ascend-infra/ascend-dmi]] · [[llm-localization/ascend/ascend-infra/ascend-docker-runtime]] · [[llm-localization/ascend/ascend-infra/达芬奇架构]]
- 框架与套件：[[ai-framework/megatron-lm/README]] · [[ai-framework/huggingface-transformers/README]]
- 上下游主题：[[llm-compression/quantization/量化基础]] · [[llm-inference/README]] · [[llm-train/README]] · [[llm-algo/transformer/模型架构]]
