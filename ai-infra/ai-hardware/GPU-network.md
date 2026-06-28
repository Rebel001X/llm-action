# GPU 互联 (NVLink/NVSwitch)

> 一句话定位：NVLink 是 NVIDIA 自家的 GPU↔GPU 高速直连总线，NVSwitch 是把它扩成"机内全互联交换机"的芯片；它们让单机 8 卡像一个大 GPU 一样协同，是张量并行 (TP) 能跑起来的物理前提。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/ai-hardware/硬件对比]] [[ai-infra/网络/InfiniBand]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 内容 | 你将能回答 |
|----|------|-----------|
| 0 | 一句话锚点 | NVLink/NVSwitch 到底是什么 |
| 1 | 地基：带宽/延迟/总线为何重要 | 为什么不能只靠 PCIe |
| 2 | NVLink 点对点：物理与代际带宽 | 一条链路多快、怎么算 |
| 3 | NVSwitch 全互联 | 8 卡如何"人人直连" |
| 4 | 单机 8 卡拓扑演进 | Cube-Mesh vs NVSwitch fabric |
| 5 | NVLink vs PCIe 量化对比 | 差几倍、差在哪 |
| 6 | 为什么 TP 必须高带宽互联 | All-Reduce 的通信量推导 |
| 7 | 跨机：NVLink 到此为止，IB 接力 | 机内 vs 机间两级网络 |
| 8 | NVLink Switch / NVL72 一瞥 | 机柜级"超级 GPU" |
| 9 | 数值例子/对照 | 手算通信开销 |
| — | 常见问题 + 跳转链接 | 速查 |

---

## 0. 一句话锚点

- **NVLink**：两块 GPU（或 GPU 与 CPU/Switch）之间的一条**点对点高速串行链路**，带宽远高于 PCIe，且支持 GPU 显存的**直接读写**（GPUDirect P2P），不经过 CPU/主存中转。
- **NVSwitch**：一颗**片上交叉开关 (crossbar) 芯片**，多个 GPU 的 NVLink 全部接到它上面，使任意两卡都能以**全速、无阻塞**互通——把"点对点"升级为"全互联 (all-to-all)"。
- 一句话：**NVLink 是线，NVSwitch 是把这些线汇成无阻塞交换的交换机。**

---

## 1. 地基：带宽、延迟、为什么互联是瓶颈

### 1.1 三个最原子的概念

- **带宽 (Bandwidth)**：单位时间能搬多少字节，单位 GB/s。决定"搬一坨数据要多久"。
- **延迟 (Latency)**：发出第一个字节到对端收到的耗时，单位 μs。决定"一来一回反应多快"。
- **总线/链路 (Link)**：两个芯片之间传数据的物理通道。GPU 之间天然不共享显存，任何跨卡数据都要走某条链路。

搬运一块数据的近似时间模型：

$$T = \alpha + \frac{N}{B}$$

其中 $\alpha$ 是固定延迟（启动开销），$N$ 是字节数，$B$ 是带宽。大消息时 $N/B$ 主导，所以**带宽决定大模型训练/推理的跨卡成本**。

### 1.2 为什么"互联"会成为整个系统的瓶颈

单卡算力（FLOPs）十年涨了上百倍，但 GPU 之间搬数据的带宽涨得慢得多。于是出现"算得快、传得慢"——这就是 AI-Infra 反复出现的 **memory wall / communication wall**。NVLink/NVSwitch 正是为了把"传得慢"这条短板补上而生。

```
单卡算力 ↑↑↑↑↑↑↑↑   (涨得猛)
跨卡带宽 ↑↑↑         (涨得慢)
            └── 差距 = 通信瓶颈 = NVLink 要解决的核心矛盾
```

---

## 2. NVLink 点对点：物理结构与代际带宽

### 2.1 物理结构：Link 由若干 Lane 组成

NVLink 的最小单位是 **Lane（差分信号对）**，若干 Lane 捆成一条 **Link（链路）**，每条 Link **双向**传输（各方向独立）。一块 GPU 上有多条 Link，可分别连向不同的对端，把它们"捆"起来就得到该 GPU 的**总 NVLink 带宽**。

```
       GPU A                         GPU B
   ┌───────────┐  Link0 (双向)   ┌───────────┐
   │  NVLink   │═════════════════│  NVLink   │
   │  控制器   │  Link1          │  控制器   │
   │           │═════════════════│           │
   │           │  ...            │           │
   └───────────┘  LinkN          └───────────┘
   每条 Link = 多条 Lane;  GPU 总带宽 = 单 Link 带宽 × Link 数
```

关键：NVLink 走的是 GPU 自己的显存地址空间——A 卡可以**像访问本地显存一样**对 B 卡显存发起 load/store（P2P）。PCIe 时代要"拷到主存再拷过去"，NVLink 把这一步省了。

### 2.2 代际带宽（典型公开规格，约 / 以官方为准）

下表为每**代际单 GPU 聚合双向带宽**的典型值，具体随产品 SKU 不同，**以 NVIDIA 官方为准**：

| 代际 | 典型代表 GPU | 每 Link 双向带宽(约) | 每 GPU Link 数(约) | 单 GPU 聚合双向带宽(约) |
|------|--------------|----------------------|--------------------|--------------------------|
| NVLink 1.0 | P100 (Pascal) | ~40 GB/s | 4 | ~160 GB/s |
| NVLink 2.0 | V100 (Volta) | ~50 GB/s | 6 | ~300 GB/s |
| NVLink 3.0 | A100 (Ampere) | ~50 GB/s | 12 | ~600 GB/s |
| NVLink 4.0 | H100 (Hopper) | ~50 GB/s | 18 | ~900 GB/s |
| NVLink 5.0 | B200 (Blackwell) | ~100 GB/s | 18 | ~1800 GB/s |

> 解读口诀：**单 GPU 聚合带宽 ≈ 单 Link 带宽 × Link 数**。代际进步主要靠"每 Link 更快 + Link 更多"两条腿。这里的"双向"指两个方向加总；单向约为一半。

### 2.3 怎么算一条具体链路

直连两卡时，**两卡之间**可用的带宽 = 它们之间分配到的 Link 数 × 单 Link 带宽。例如 A100 之间若用 4 条 Link 直连，双向约 $4 \times 50 = 200$ GB/s。8 卡两两全直连时每卡 12 条 Link 要"分给"7 个邻居，于是出现了拓扑问题（见第 4 节）——这正是 NVSwitch 登场的原因。

---

## 3. NVSwitch 全互联：把"点对点"升级为"无阻塞交换"

### 3.1 没有 NVSwitch 的痛点

8 块 GPU 想两两直连，需要 $\binom{8}{2}=28$ 对连接；每卡的 Link 数有限，必然不够"人人满速直连"。结果是**部分卡之间带宽高、部分卡之间要绕路（多跳）**，带宽不均、延迟不一。

### 3.2 NVSwitch 的解法：crossbar 交换芯片

NVSwitch 内部是一个**交叉开关 (crossbar)**：把每块 GPU 的**全部 Link 都接进交换机**，交换机内部任意输入端口都能无阻塞地连到任意输出端口。于是**任意两卡之间都能拿到全速带宽**，且是**对称**的（任意一对都一样快）。

```
                ┌──────── NVSwitch (crossbar) ────────┐
                │   任意输入端口 ↔ 任意输出端口 无阻塞   │
                └──┬───┬───┬───┬───┬───┬───┬───┬──────┘
                   │   │   │   │   │   │   │   │
                 GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
   每块 GPU 的全部 NVLink 都进 Switch ⇒ 任意两卡满速、对称、单跳
```

### 3.3 为什么常见"多颗 NVSwitch"

单颗 Switch 的端口数有限，撑不下 8 卡 × 每卡十几条 Link 的总端口。于是一台机器里放**多颗 NVSwitch**，每块 GPU 把自己的 Link **均匀分摊**到所有 Switch 上。这样既凑够端口，又让流量分散、避免单颗 Switch 拥塞。

```
   GPU0 ─┬─→ NVSwitch-A
         ├─→ NVSwitch-B      每块 GPU 的 18 条 Link
         ├─→ NVSwitch-C      被切成几份，分别接到
         └─→ NVSwitch-D      不同 Switch ⇒ 负载均衡 + 端口够用
   (其余 GPU 同理)
```

---

## 4. 单机 8 卡拓扑：从 Cube-Mesh 到 NVSwitch Fabric

### 4.1 拓扑一：NVLink 直连（Cube-Mesh / Hybrid Cube）

早期（如 DGX-1 / V100）没有把所有卡接进 Switch，而是用 NVLink **直接两两连**，形成一个"立方体网格"。问题：**不是全互联**——有的卡对是直连（满速），有的卡对要经过中间卡**转发（多跳）**，于是带宽和延迟**不对称**。

```
   DGX-1 风格 (示意, 非全互联):
        GPU0 ── GPU1
         │  ╲  ╱ │          实线 = 直连 NVLink
        GPU2 ── GPU3        斜线 = 跨面连接
         │        │         GPU0→GPU5 可能要"借道"中间卡
        GPU4 ── GPU5
         │  ╲  ╱ │
        GPU6 ── GPU7
   特点: 邻居快, 远处慢 ⇒ 通信库需感知拓扑 (topology-aware)
```

### 4.2 拓扑二：NVSwitch 全互联（DGX-2 / A100 / H100 之后）

把每块 GPU 的全部 Link 接进多颗 NVSwitch（见第 3 节），8 卡形成**任意两卡满速、单跳、对称**的全互联。这是现代训练机的标准形态。

```
   现代 DGX / HGX (NVSwitch fabric):

   GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
     ╲    │    │    │     │    │    │    ╱
      ╲   └────┴──┬─┴──┬──┴────┴───┘   ╱
       └─────────[ N 颗 NVSwitch crossbar ]──────┘
   任意 GPU_i ↔ GPU_j : 同样的带宽、同样的延迟、单跳
```

### 4.3 两种拓扑对比

| 维度 | NVLink 直连 (Cube-Mesh) | NVSwitch 全互联 |
|------|--------------------------|------------------|
| 任意两卡带宽 | 不均（直连快/远处慢） | 均匀满速 |
| 跳数 | 1~2 跳 | 单跳 |
| All-Reduce 效率 | 受最慢路径拖累 | 接近理论上限 |
| 软件复杂度 | 需拓扑感知 | 透明、简单 |
| 代表机型 | DGX-1 (V100) | DGX-2/A100/H100 |

---

## 5. NVLink vs PCIe：量化对比

### 5.1 PCIe 是什么、为什么不够

PCIe 是 CPU↔设备的**通用**总线，所有外设（网卡、SSD、GPU）共享同一套规范。它带宽有限，且早期 GPU↔GPU 通信常要**经 CPU/主存中转**。对动辄几十上百 GB/s 跨卡流量的训练来说，PCIe 是"村口小路"。

### 5.2 PCIe 各代单向带宽（x16，约 / 以官方为准）

| PCIe 代 | 每 Lane 速率(约) | x16 单向带宽(约) |
|---------|------------------|-------------------|
| Gen3 | ~8 GT/s | ~16 GB/s |
| Gen4 | ~16 GT/s | ~32 GB/s |
| Gen5 | ~32 GT/s | ~64 GB/s |

### 5.3 直接对比

```
   单 GPU 聚合双向带宽 (约):
   PCIe Gen4 x16 ▏~64 GB/s (双向)
   PCIe Gen5 x16 ▏▏~128 GB/s (双向)
   A100 NVLink3  ████████████ ~600 GB/s
   H100 NVLink4  ██████████████████ ~900 GB/s
   B200 NVLink5  ████████████████████████████████████ ~1800 GB/s
                 (横条长度 ∝ 带宽)
```

| 对比项 | PCIe | NVLink |
|--------|------|--------|
| 角色 | 通用外设总线 | GPU 专用直连 |
| 单 GPU 带宽量级 | 几十 GB/s | 数百~上千 GB/s（差 5~20×+） |
| GPU↔GPU 直读显存 | 受限（GPUDirect 部分支持） | 原生 P2P，地址空间直访 |
| 是否经 CPU/主存 | 传统路径常需中转 | 不需要 |
| 适用 | 接网卡/盘/入门多卡 | 张量并行等高频跨卡 |

> 一句话：**带宽相差一个量级以上**，且 NVLink 省掉了 CPU 中转的延迟——这就是为什么训练机不用 PCIe 充当主互联。

---

## 6. 为什么张量并行 (TP) 必须要高带宽互联

### 6.1 TP 在做什么（最原子）

一个权重矩阵 $W$ 太大，单卡放不下/算不动，就把它**按列（或按行）切成若干块**，分给不同 GPU 各算一部分。但一层的输出需要把各卡的**部分结果合并**——这就要在**每一层、每一次前向/反向**都做一次跨卡通信（典型是 **All-Reduce** 或 All-Gather）。

```
   输入 X  ──┬──→ GPU0 算 X·W0  ┐
            ├──→ GPU1 算 X·W1  ├─ All-Reduce(求和/拼接) ─→ 完整输出
            ├──→ GPU2 算 X·W2  │
            └──→ GPU3 算 X·W3  ┘
   每一层都来一次! 层数几十~上百 ⇒ 通信极其频繁
```

### 6.2 通信量推导（为什么必须快）

Ring All-Reduce 搬运的数据量约为：

$$V_{\text{comm}} \approx 2 \cdot \frac{p-1}{p} \cdot S$$

其中 $p$ 是参与卡数，$S$ 是被规约张量的字节数。当 $p$ 较大时 $\frac{p-1}{p}\to 1$，即每卡约搬运 $2S$ 字节。这一步**夹在两段计算之间，是串行依赖**：通信没完成，下一层算不了。于是：

$$T_{\text{layer}} = T_{\text{compute}} + \underbrace{\frac{2S}{B}}_{\text{通信, 与 } B \text{ 成反比}}$$

带宽 $B$ 越大，通信项越小，GPU 越不"等数据"。**TP 把通信塞进了热路径，所以对带宽极度敏感——这正是 NVLink/NVSwitch 的主战场。**

### 6.3 为什么 TP 一般不跨机

跨机要走 IB（见第 7 节），带宽比机内 NVLink 低一个量级、延迟更高。把每层都要做的 TP All-Reduce 放到慢链路上，GPU 会大量空转。所以工程上：**TP 留在机内（吃 NVLink），跨机交给通信更稀疏的并行（DP / PP）。**

```
   并行维度 → 通信频率 → 放哪
   TP (张量并行)  每层都通信(最密)  → 机内 NVLink/NVSwitch
   PP (流水并行)  仅级间边界        → 可跨机 IB
   DP (数据并行)  每步梯度同步一次   → 可跨机 IB
```

---

## 7. 跨机：NVLink 到此为止，InfiniBand 接力

### 7.1 NVLink 的边界

NVLink/NVSwitch 是**机内（机箱内）**互联，覆盖一台服务器里的 8 卡（或机柜内若干卡，见第 8 节）。**出了机箱，靠的是网络**——主流是 **InfiniBand (IB)**，少数用 RoCE（RDMA over Ethernet）。

### 7.2 两级网络结构

```
   ┌──────── Server A (8 GPU) ────────┐     ┌──────── Server B (8 GPU) ────────┐
   │  GPU×8 ── NVSwitch (机内全互联)  │     │  GPU×8 ── NVSwitch (机内全互联)  │
   │      │                          │     │      │                          │
   │   IB HCA(网卡)                  │     │   IB HCA(网卡)                  │
   └──────┼───────────────────────────┘     └──────┼──────────────────────────┘
          └──────────── IB 交换机 / Fat-Tree 网络 ──┘
   机内: NVLink (数百GB/s)   机间: IB (单网卡约几十GB/s级, 多网卡聚合)
```

### 7.3 关键技术：GPUDirect RDMA

跨机时，IB 网卡可借 **GPUDirect RDMA** 直接读写远端 GPU 显存，**绕过 CPU 和系统内存**，把跨机延迟和 CPU 占用压下来——理念和 NVLink 的 P2P 一致：**让数据走最短路径，别绕 CPU**。细节见 [[ai-infra/网络/InfiniBand]]。

### 7.4 带宽断崖与并行映射

机内 NVLink（数百~上千 GB/s）vs 机间 IB（单端口约几十 GB/s 级，需多网卡/多轨聚合），中间有一道**带宽断崖**。集群拓扑设计的核心，就是**让通信最密的并行维（TP）落在带宽最高的链路（NVLink）上**，通信稀疏的（DP/PP）才跨机。这与第 6 节结论一致，是机内/机间分工的根本逻辑。

---

## 8. 机柜级：NVLink Switch / NVL72 一瞥

新一代把 NVLink 从"机内"推到"机柜内"：用**独立的 NVLink Switch 设备**（不再只是主板上的 NVSwitch 芯片）将一个机柜内**几十块 GPU** 连成统一的 NVLink 域，对软件呈现为一个"超大 GPU"。NVIDIA 的 **GB200 NVL72**（约 72 个 Blackwell GPU 同域，规格以官方为准）即此思路：把"全互联域"从 8 卡扩到几十卡，让超大模型的 TP/EP 也能享受 NVLink 级带宽。

```
   传统:  [8 GPU + NVSwitch] = 一个 NVLink 域
   NVL72: [数十 GPU + NVLink Switch 机柜] = 一个超大 NVLink 域
          ⇒ 跨节点的"机内级"带宽, TP/专家并行可扩到几十卡
```

---

## 9. 数值例子 / 对照（手算）

### 例 1：H100 上一次 TP All-Reduce 要多久

设规约张量 $S = 64$ MB $= 64\times 10^6$ 字节，8 卡 TP，机内 NVLink 有效带宽取 $B \approx 400$ GB/s（实测打折后，<峰值 900）。每卡搬运约 $2S$：

$$T \approx \frac{2 \times 64\times10^6}{400\times10^9}\,\text{s} \approx 3.2\times10^{-4}\,\text{s} = 320\ \mu s$$

### 例 2：同样的 All-Reduce 走 PCIe Gen4 会怎样

取有效带宽 $B \approx 25$ GB/s：

$$T \approx \frac{2 \times 64\times10^6}{25\times10^9}\,\text{s} \approx 5.1\times10^{-3}\,\text{s} = 5120\ \mu s$$

**慢约 16 倍。** 每层都来一次、几十上百层、每个训练 step 来一遍——差距被放大到不可接受。这就是"为什么 TP 必须吃 NVLink"的最直观证据。

### 例 3：把这步放到跨机 IB（单端口约 25 GB/s 级，未做多轨聚合）

数量级与 PCIe 相近甚至更糟（叠加更高延迟与网络抖动），所以 **TP 不跨机**。结论与第 6、7 节闭环。

| 链路 | 有效带宽(约) | 例1 那步耗时(约) | 相对 NVLink |
|------|--------------|-------------------|--------------|
| H100 NVLink | ~400 GB/s | ~320 μs | 1× |
| PCIe Gen5 | ~50 GB/s | ~2560 μs | ~8× 慢 |
| PCIe Gen4 | ~25 GB/s | ~5120 μs | ~16× 慢 |
| 跨机 IB(单端口) | ~25 GB/s + 高延迟 | >5120 μs | 更慢 |

> 数字为教学量级估算，"有效带宽"已对峰值打折；精确值**以官方与实测为准**。

---

## 常见问题

| 问题 | 答 |
|------|----|
| NVLink 和 NVSwitch 区别？ | NVLink 是两点之间的链路；NVSwitch 是把多条 NVLink 汇成无阻塞交换的芯片，实现 8 卡全互联。 |
| 没有 NVSwitch 能多卡吗？ | 能，但只能 NVLink 直连或走 PCIe，会出现带宽不均/多跳，All-Reduce 受最慢路径拖累。 |
| 为什么 TP 不跨机？ | TP 每层都做 All-Reduce，通信在热路径上；跨机 IB 比机内 NVLink 慢一个量级，GPU 会空转。 |
| NVLink 比 PCIe 快多少？ | 量级差距，单 GPU 聚合带宽从几十 GB/s（PCIe）到数百~上千 GB/s（NVLink），约 5~20×+。 |
| "双向带宽"怎么理解？ | 两个方向独立各跑一半；单向约为双向数字的一半。 |
| 跨机靠什么？ | InfiniBand（或 RoCE）+ GPUDirect RDMA，详见 [[ai-infra/网络/InfiniBand]]。 |
| All-Reduce 搬多少数据？ | Ring 算法每卡约 $2\cdot\frac{p-1}{p}\cdot S \approx 2S$ 字节，见第 6.2 节。 |
| NVL72 是什么？ | 用 NVLink Switch 把一个机柜内几十块 GPU 连成单一 NVLink 域，等效一个超大 GPU。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航总入口
- [[ai-infra/ai-hardware/硬件对比]] — GPU/带宽/算力规格对照
- [[ai-infra/网络/InfiniBand]] — 跨机 IB 与 GPUDirect RDMA 细节
- [[ai-infra/网络/集合通信原语]] — All-Reduce / All-Gather / Ring 等原语推导

### 官方参考（以官方文档为准）

- NVIDIA Networking / MLNX OFED 文档：https://docs.nvidia.com/networking/display/mlnxofedv583070101/introduction
- NVMe-oF (NVM Express over Fabrics)：https://docs.nvidia.com/networking/display/mlnxofedv583070101/nvme-of+-+nvm+express+over+fabrics
- NVIDIA Ada Lovelace 架构：https://www.nvidia.cn/geforce/ada-lovelace-architecture/ ｜ https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/
