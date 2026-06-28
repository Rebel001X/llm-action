# LLM in a Flash：用闪存把放不下的大模型"流"进内存来推理

> 一句话定位：当 DRAM 装不下整个 LLM 时，把模型权重存在 **闪存（Flash/SSD）** 上，靠"稀疏感知 + 按需加载 + 数据局部性"，让显存/内存只有半个模型大小的设备也能跑完整模型推理。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/KV-Cache优化]] · [[llm-compression/sparsity/README]] · [[llm-optimizer/kv-cache]] · [[llm-inference/vllm/README]]

- 论文：*LLM in a Flash: Efficient Large Language Model Inference with Limited Memory*（Apple, 2023/2024）
- arXiv：https://arxiv.org/abs/2312.11514
- 解读：https://medium.com/@marketing_novita.ai/llm-in-a-flash-efficient-inference-techniques-with-limited-memory-5f0a404794b0

---

## 阅读地图

| 节 | 内容 | 你会带走什么 |
|----|------|--------------|
| 0 | 一句话锚点 | 这篇到底解决了什么 |
| 1 | 地基：内存墙 + 存储层级 | 为什么"放不下"是真问题 |
| 2 | 关键观察：FFN 激活高度稀疏 | 只需加载用得到的那一小撮神经元 |
| 3 | 支柱一：Selective Loading（按需选择性加载） | 用 predictor 预测要哪些神经元 |
| 4 | 支柱二：Windowing（滑动窗口复用） | 跨 token 复用已加载的权重 |
| 5 | 支柱三：Row-Column Bundling（行列捆绑） | 把 I/O 凑成大块，吃满闪存带宽 |
| 6 | 内存管理：原地更新不重分配 | 避免拷贝/碎片化 |
| * | 关键公式 / 算法 / 数值手算 | I/O 量、带宽账、加速比 |
| 末 | 评价 / 对照 / 局限 | 它的边界在哪 |

---

## 0. 一句话锚点

> **把"参数放哪"从一刀切（全进 DRAM）变成分层（热的进 DRAM、冷的留在 Flash 按需取），再用稀疏性把"要取的量"压到极小，并把取数据的方式改成大块连续读以吃满闪存带宽——于是 DRAM 只够装半个模型，也能推理整个模型。**

核心三件套（论文反复强调的三个 idea）：
1. **Selective Loading（选择性加载）**：只从闪存读"这一步真正会被激活"的 FFN 神经元。
2. **Windowing（窗口化复用）**：相邻 token 激活的神经元高度重叠，已经在 DRAM 里的别重复读。
3. **Row-Column Bundling（行列捆绑）**：把零散小读改成大块连续读，匹配闪存"块设备"的脾气。

---

## 1. 地基：内存墙 + 存储层级（问题背景）

### 1.1 为什么"放不下"

一个 7B 模型，FP16 权重 ≈ $7\times10^9 \times 2\,\text{B} = 14\,\text{GB}$。手机/边缘设备的可用 DRAM 常常只有 6–8 GB，还要留给系统和 KV-Cache。**整模型进不了 DRAM**，这就是 *Limited Memory*。

传统三条路都不够：
- **量化**（INT4）：把 14GB 压到 ~3.5GB，能缓解但有精度损失，且更大的模型仍超标。见 [[llm-compression/quantization/量化基础]]。
- **蒸馏/裁剪**：换一个更小的模型，但能力下降。
- **卸载到 CPU/Disk（offloading）**：经典做法（如 ZeRO-Infinity、FlexGen）按层 offload，但**每一步都要把整层搬来搬去**，I/O 成了瓶颈。

本文的不同：**不是搬整层，而是只搬"被激活的那几行权重"**，把 I/O 量从"层级"压到"神经元级"。

### 1.2 存储层级（理解全文的物理地基）

```
            速度快 / 容量小 / 贵
   ┌──────────────────────────────┐
   │  GPU/NPU 寄存器、片上 SRAM     │  ← FlashAttention 在这里做文章
   ├──────────────────────────────┤
   │  DRAM（主存/统一内存）         │  ← 传统假设"模型必须全在这"
   ├──────────────────────────────┤
   │  Flash / SSD（NAND）           │  ← 本文：把冷权重放这里，按需取
   └──────────────────────────────┘
            速度慢 / 容量大 / 便宜
```

闪存的关键脾气（决定了第 5 节的设计）：
- **带宽不低，但延迟高**：一次小读（几 KB）有固定开销，**读 1 个 4KB 块和读 1 个 32KB 块的时间差不多**。
- **顺序读 >> 随机小读**：吞吐对"单次读的块大小"极其敏感。
- 结论：**要么少读，要么每次都读一大块**。本文两手都抓。

> 类比 [[llm-optimizer/FlashAttention]]：FlashAttention 是在 SRAM↔DRAM 这一层"少搬数据"；本文是在 DRAM↔Flash 这一层"少搬数据 + 搬就搬大块"。同一种内存层级思想，下移了一级。

---

## 2. 关键观察：FFN 激活高度稀疏

Transformer 每层 = Attention + FFN（前馈网络）。FFN 通常是：

$$\text{FFN}(x) = W_{\text{down}}\,\big(\,\sigma(W_{\text{up}}\,x)\,\big)$$

其中 $\sigma$ 多为 **ReLU / ReLU 系**激活（论文用的 OPT、Falcon 等用 ReLU 族）。

**观察**：ReLU 把大量中间值压成 0。论文统计：FFN 中间层**约 90% 以上的神经元在单个 token 上输出为 0**（具体比例见原文，约 90%+）。

```
         W_up · x  →  σ(·)  →  只有少数 > 0
  [■■■■■■■■■■■■■■■■]   ReLU   [□□■□□□□□■□□□■□□□]   ← 仅 ■ 处非零
   完整中间维度(几千~上万)            ~10% 被激活
```

这意味着：**算 FFN 时，$W_{\text{up}}$ 中只有被激活那几行、$W_{\text{down}}$ 中只有对应那几列真正参与计算。** 其余权重这一步**根本不需要在内存里**。

→ 这就是"按需加载"的理论依据：**把"加载哪些权重"和"会激活哪些神经元"绑定起来。**

> 注意：稀疏发生在 **FFN**。Attention 权重 + embedding 等"稠密、每步都用"的部分，本文选择**常驻 DRAM**；只对占参数大头的 FFN 做闪存按需加载。延伸阅读 [[llm-compression/sparsity/README]]。

---

## 3. 支柱一：Selective Loading（选择性加载）

### 3.1 机制

每一步推理，先用一个**轻量预测器（low-rank predictor）**预测"这一步 FFN 会激活哪些神经元"，**只从闪存把这些行/列读进来**。

```
  当前 token 的 x
        │
        ▼
  ┌─────────────┐   预测激活掩码(稀疏)
  │  Predictor  │ ───────────────► [□□■□□■□□...]
  └─────────────┘                       │
                                        ▼
                          只 read 这些行：W_up[这些行], W_down[这些列]
                          其余权重仍躺在 Flash，不读
                                        │
                                        ▼
                              在 DRAM 里完成 FFN 计算
```

### 3.2 Predictor 长什么样

是个**很小的低秩 MLP**（附加在每层上，参数量极小，常驻 DRAM）：输入该层的 attention 输出，输出"每个 FFN 神经元是否会被激活"的概率，取 top-k / 阈值得到稀疏掩码。

- 训练目标：尽量**不漏报**真正会激活的神经元（漏报 → 计算错误/精度掉），允许少量**误报**（误报 → 多读一点点，浪费 I/O 但不影响正确性）。
- 代价/收益权衡：predictor 自身要算、要常驻内存，但它换来的是"只读 ~10% 权重"的巨大 I/O 节省，净赚。

> 思想同源：很多 MoE 推理也是"先路由再只算被选专家"。这里相当于把 **dense FFN 当成一个细粒度的"软 MoE"**，predictor 就是路由器。对照 [[llm-algo/moe/README]]。

---

## 4. 支柱二：Windowing（滑动窗口复用）

### 4.1 直觉

自回归生成是一个 token 接一个 token。**相邻 token 激活的 FFN 神经元集合高度重叠**（语义连续→激活模式连续）。如果每步都从闪存重新读，会重复读大量上一两步已经在 DRAM 里的权重。

### 4.2 机制：维护一个"最近 W 步激活神经元"的窗口

只在 DRAM 中保留**最近 $W$ 个 token 用到过的神经元权重**。生成新 token 时：

- **新激活但不在窗口** → 从闪存 **加载（incremental load）**。
- **窗口里有、但已掉出最近 $W$ 步** → **释放（delete）**，腾内存。
- **既在窗口又仍被需要** → 直接复用，**零 I/O**。

```
  时间轴 →   t-2     t-1      t (当前)
  激活集合: {A,B,C} {B,C,D}  {C,D,E}
                                │
        窗口 W=2 内并集 ≈ {B,C,D,E} 常驻 DRAM
                                │
   生成 t 时:  C,D 已在窗口(复用,0 I/O)   E 是新的(只加载 E)
               A 掉出窗口(释放)
```

净 I/O 量 ≈ **「新激活的神经元数」− 「被释放的神经元数」**，而不是「全部激活的神经元数」。窗口越大复用越多、I/O 越省，但占用 DRAM 越多 → **$W$ 是内存 vs I/O 的旋钮**。

---

## 5. 支柱三：Row-Column Bundling（行列捆绑）

### 5.1 问题：稀疏读 = 一堆随机小读，最伤闪存

第 3 节选出的激活神经元在权重矩阵里是**离散分布**的行/列。若逐个去闪存读 $W_{\text{up}}$ 的某一行（几 KB），就是大量**随机小 I/O**，闪存延迟开销吃满，带宽利用率极低。

### 5.2 解法：把"同一个神经元相关的数据"在闪存里**物理捆绑**成一块连续存储

对 FFN 第 $i$ 个中间神经元，它对应：
- $W_{\text{up}}$ 的**第 $i$ 行**（up 投影）
- $W_{\text{down}}$ 的**第 $i$ 列**（down 投影）

把这"第 $i$ 行 + 第 $i$ 列"**拼在一起、连续存放**。这样预测要神经元 $i$ 时，**一次连续读**就把它的 up-行和 down-列全拿到，读的块更大 → 单位时间搬的有用数据更多。

```
  Flash 上的物理布局（捆绑后）：

  [ neuron_0: up_row_0 | down_col_0 ][ neuron_1: up_row_1 | down_col_1 ] ...
   └──────── 一次连续读 ────────┘   每个 bundle = 一个神经元的全部权重

  对比未捆绑：up_row_i 在矩阵A某处, down_col_i 在矩阵B某处 → 两次随机小读
```

效果：把**两次随机小读**合并成**一次较大连续读**，**读块翻倍、随机次数减半**，直接吃闪存的"大块顺序读更快"特性。论文还讨论把多个相邻被选神经元进一步合并成更大读块。

### 5.3 配套：原地内存管理（避免重分配）

加载/释放神经元会让 DRAM 里的"激活矩阵"频繁变形。论文用**预分配一块固定大缓冲 + 原地增删**：
- 删神经元：把缓冲尾部那一行**搬过来覆盖**被删行，再缩短有效长度（不真正 free）。
- 加神经元：直接写到缓冲尾部、长度 +1。
- 好处：**零重新分配、零大块拷贝、不产生碎片**，配合 windowing 的频繁增删极其关键。

```
  缓冲区(预分配, 容量固定):
  [ r0 | r1 | r2 | r3 | r4 |  空  |  空 ]   len=5
  删除 r1 → 把末尾 r4 搬到 r1 位置:
  [ r0 | r4 | r2 | r3 |  空 |  空 |  空 ]   len=4   (O(1) 一行拷贝, 无 free)
```

---

## 关键公式 / 算法 / 数值示例

### 算法（一次 FFN 前向的伪流程）

```
输入: token 隐状态 x, 层 ℓ 的 FFN, DRAM 窗口 window[ℓ]
1. mask = Predictor_ℓ(x)               # 预测激活神经元集合 (稀疏)
2. need   = {i | mask[i]=1}             # 这步需要的神经元
3. load   = need \ window[ℓ]            # 不在窗口的 → 要从 Flash 读
4. evict  = (掉出最近 W 步的) ∩ window  # 释放
5. for i in load:  从 Flash 连续读 bundle_i (up_row_i + down_col_i)  # 行列捆绑
6. for i in evict: 原地删除 (末行覆盖)                              # 原地内存管理
7. 用 window 中 need 对应的行/列计算 FFN(x)                         # 只算被激活部分
8. 更新 window[ℓ] 的近用记录
```

### 公式：有效 I/O = 加载量 × (1 − 复用率)

设单层 FFN 中间维度 $d_{\text{ff}}$，平均激活率 $s$（≈10%），相邻步激活重叠率 $r$（复用率），单神经元 bundle 字节 $b$。则**每生成 1 个 token、每层从闪存读的字节**：

$$
\text{IO}_{\text{layer}} \approx d_{\text{ff}} \cdot s \cdot (1 - r)\cdot b
$$

对比朴素 offloading 的"每步搬整层 FFN"：

$$
\text{IO}_{\text{naive}} \approx d_{\text{ff}} \cdot b \quad(\text{up+down 全搬})
$$

节省倍数 $\approx \dfrac{1}{s(1-r)}$。

### 数值手算（直观感受量级，数字为示例非原文实测）

取 OPT-6.7B 量级：$d_{\text{ff}}=16384$，激活率 $s=0.1$，重叠复用率 $r=0.7$，一个神经元 bundle（up 行 + down 列，FP16）≈ $2\times d_{\text{model}}\times 2\text{B}$，$d_{\text{model}}=4096$ → $b\approx 16\,\text{KB}$。

- 朴素整层搬：$16384 \times 16\text{KB} \approx 256\ \text{MB/层/步}$。
- 本文按需读：$16384 \times 0.1 \times (1-0.7) \times 16\text{KB} \approx 7.7\ \text{MB/层/步}$。
- **节省倍数 $\approx \dfrac{1}{0.1\times0.3}\approx 33\times$**（与论文"约 2 倍模型大小可放入一半内存、I/O 大降"的定性结论同向；精确数字见原文）。

带宽账：若闪存有效带宽 ~1 GB/s，7.7MB/层 × 32 层 ≈ 246 MB/步 → 单看 I/O 约 0.25 s/token 量级。**行列捆绑把"有效带宽"从随机小读的几十 MB/s 拉到接近顺序读的 GB/s，这一步是把上面账单变现的关键**（精确加速见原文）。

---

## 评价 / 对照 / 局限

| 维度 | 结论（定性，数字见原文） |
|------|--------------------------|
| 主要收益 | 可在 **约一半模型大小**的 DRAM 上跑完整模型；相比朴素 offloading，推理速度有**数倍**提升（CPU 上更明显，约 4–5×；GPU 上约 20–25×，见原文） |
| 适用前提 | FFN 激活**高度稀疏**（ReLU 族）。这是硬约束 |
| 三支柱定位 | Selective=少读；Windowing=不重复读；Bundling=读得快。互补，缺一打折 |
| 与量化关系 | **正交可叠加**：先 INT4 量化再按需加载，I/O 量再降 → [[llm-compression/quantization/fp8]] / [[llm-compression/quantization/量化基础]] |
| 与 MoE 关系 | 思想同源（条件计算/稀疏激活）；本文把 dense FFN 当细粒度软 MoE → [[llm-algo/moe/README]] |

**局限 / 边界（务实地说）**：
- **强依赖 ReLU 稀疏**。对 **GeLU/SwiGLU**（LLaMA、Qwen 等主流）几乎不稀疏，方法直接失效，需先做"ReLU 化/稀疏化微调"（后续工作如 ReLU²、CATS、ProSparse 等正是补这个坑）。
- **Predictor 有开销且可能漏报**：漏报会引入数值误差，需要训练得足够准。
- **对闪存随机读延迟敏感**：在差的存储介质上收益缩水；行列捆绑只能缓解不能消灭延迟。
- **主要面向单请求/边缘端**：和云端 [[llm-inference/连续批处理]]、[[llm-inference/PD分离]] 的大 batch 吞吐场景目标不同——本文是"装得下"，那边是"跑得快、并发高"。
- KV-Cache 仍需占内存，长上下文下 KV 增长可能重新成为内存瓶颈 → [[llm-inference/KV-Cache优化]]。

**对工程的启示**：
1. **内存层级思维要下沉到 SSD**：不只是 SRAM↔HBM，DRAM↔Flash 也能做"少搬 + 搬大块"。
2. **稀疏不只是省算力，更是省 I/O**：在"算不是瓶颈、搬才是瓶颈"的场景，稀疏的价值被放大。
3. **把数据布局当成一等公民**：行列捆绑说明——**物理存储布局和访问模式匹配**，常常比算法本身带来更大加速。
4. **predictor + 按需调度**这套范式可迁移到：MoE 专家预取、KV-Cache 分层、参数 offload 调度。

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 稀疏/激活：[[llm-compression/sparsity/README]] · [[llm-algo/moe/README]]
- 内存层级近亲：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]]
- 量化叠加：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 推理系统对照：[[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/连续批处理]]
- 架构基础：[[llm-algo/transformer/模型架构]]
- 性能指标：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
