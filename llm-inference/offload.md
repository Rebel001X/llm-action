# 推理 Offload(显存卸载)

> 当 GPU 显存(HBM)装不下整个模型的权重 + KV Cache 时,把"暂时用不到"的张量搬到 CPU 内存乃至 NVMe SSD,用时再搬回 GPU 计算——以**带宽换容量**,让小卡也能跑大模型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-optimizer/kv-cache]] [[ai-framework/deepspeed/README]]

## 阅读地图

| 节 | 内容 | 你会得到什么 |
| --- | --- | --- |
| 0 | 一句话锚点 | Offload 的本质：用带宽换容量 |
| 1 | 地基:显存预算账 | 权重/KV/激活各占多少,为什么会爆 |
| 2 | 存储层级与带宽 | HBM/PCIe/CPU-DRAM/NVMe 的真实速度差 |
| 3 | Offload 的三类对象 | 权重 / KV Cache / 激活,各自的卸载策略 |
| 4 | ZeRO-Inference 思路 | 层流水 + 预取,把权重当"流" |
| 5 | FlexGen 思路 | 三维搜索空间 + zig-zag block 调度 |
| 6 | 带宽瓶颈与隐藏术 | 双缓冲/异步拷贝/计算-传输重叠 |
| 7 | 何时该用 / 不该用 | 决策树 |
| 数值示例 | 70B 在 24GB 卡上的逐数手算 | 显存账 + 搬运时间 + 吞吐估算 |
| 对照表 | ZeRO-Inf vs FlexGen vs vLLM-CPU | 选型 |
| 高频追问 | 面试踩点 | 可背诵要点 |

## 0. 一句话锚点

> **Offload = 把显存当"缓存"而非"仓库"。** 仓库(权重全集、KV 全集)放在便宜但慢的 CPU 内存 / NVMe;GPU 只保留"当前这一步要算的那一小块",算完即换。代价是每一步都要付**搬运时间**,所以 Offload 几乎总是**带宽受限(bandwidth-bound)**而非算力受限。

核心权衡一句话:

$$
\text{总时间} \approx \max(\underbrace{T_{\text{compute}}}_{\text{GPU算}},\ \underbrace{T_{\text{transfer}}}_{\text{搬权重/KV}})\quad(\text{理想重叠时})
$$

如果搬运慢于计算,GPU 就在"干等数据",利用率被拖到个位数百分比——这就是 Offload 的宿命:**能跑,但慢**。

## 1. 地基:先把显存预算账算清楚

不理解"为什么会爆",就不理解 Offload 救的是哪部分。推理时 GPU 显存被三块吃掉:

```
GPU 显存 (HBM) 占用
┌─────────────────────────────────────────────┐
│  ① 模型权重 Weights      (静态,最大头)        │
│  ② KV Cache              (随序列/batch 线性增长)│
│  ③ 激活 Activations      (单层瞬时,较小)       │
│  + 框架/CUDA context 开销 (固定 ~1-2GB)         │
└─────────────────────────────────────────────┘
```

**① 权重显存**:参数量 $N$,每参数 $b$ 字节:

$$
M_{\text{weight}} = N \times b
$$

- FP16/BF16:$b=2$。70B → $70\times10^9 \times 2 = 140\text{ GB}$。
- INT8:$b=1$ → 70 GB。INT4:$b=0.5$ → 35 GB。

**② KV Cache 显存**(详见 [[llm-optimizer/kv-cache]]):

$$
M_{\text{kv}} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times S \times B \times b_{\text{kv}}
$$

其中 $L$=层数,$n_{kv}$=KV 头数,$d_{head}$=每头维度,$S$=序列长度,$B$=batch,$b_{kv}$=每元素字节,前面的 2 是 K 和 V 两份。

**③ 激活**:推理时只需保留**当前层**的中间结果(不像训练要存全部反向用的激活),所以单 token 解码阶段激活极小,通常可忽略。

> 一台 24GB 的 RTX 4090 连 70B 的权重(140GB FP16)都装不下 1/5。要么多卡张量并行,要么——Offload。

## 2. 存储层级:速度差几个数量级是关键

Offload 的全部设计动机来自这张"带宽金字塔"。数字为典型量级(具体以硬件为准):

```
         容量小   速度快
            ▲          ┌──────────────┐  ~3 TB/s  (片上,免费)
            │          │ GPU HBM 24-80GB│
   带宽 ↓   │   ┌──────┴──────────────┴──────┐
            │   │   PCIe Gen4 x16  ~25-32 GB/s │ ← 搬权重/KV 必经此桥
            │   ├──────────────────────────────┤
            │   │   CPU DRAM  128GB-2TB  ~50-100 GB/s (CPU 本地访问)│
            │   ├──────────────────────────────┤
            ▼   │   NVMe SSD  1-8TB   ~3-7 GB/s (PCIe Gen4)│
         容量大   └──────────────────────────────┘
                                 速度慢
```

关键认知:
- **HBM 比 PCIe 快约 ~100×**。一旦数据要过 PCIe,带宽瞬间塌方。
- **PCIe 是咽喉**。GPU↔CPU、GPU↔NVMe(无 GPUDirect 时还要绕 CPU)都挤这座桥。
- **NVMe 比 DRAM 慢约 ~10×**,但容量便宜到可以放下整模型。

所以 Offload 的层级选择天然有序:**HBM 放不下 → 溢出到 DRAM → DRAM 也放不下 → 溢出到 NVMe**。越往下越慢,但越能装。

## 3. Offload 的三类对象

不是所有东西都按同样方式卸载。三类对象的访问模式不同,策略也不同。

```
┌──────────┬─────────────────┬──────────────────────────┐
│  对象     │ 访问模式         │ Offload 策略              │
├──────────┼─────────────────┼──────────────────────────┤
│ 权重     │ 顺序、可预测     │ 层流水 + 预取(prefetch)   │
│          │ (L0→L1→...→Ln)  │ 算第 i 层时搬第 i+1 层    │
├──────────┼─────────────────┼──────────────────────────┤
│ KV Cache │ 随机、随 step 增长│ 按需取(注意力时拉回)      │
│          │ 历史 token 都要读 │ 或保留近窗 + 卸远窗       │
├──────────┼─────────────────┼──────────────────────────┤
│ 激活     │ 瞬时、单层       │ 一般不卸(太小,搬不划算)  │
└──────────┴─────────────────┴──────────────────────────┘
```

**为什么权重最适合 Offload?** 因为它的访问是**确定性顺序**的:Transformer 永远从第 0 层算到第 L-1 层。既然能预知下一层是谁,就能在算当前层时**提前**把下一层从 CPU 搬过来(预取),把搬运时间藏在计算时间背后。

**为什么 KV Cache 难卸?** 自注意力要读**所有历史 token** 的 K、V。卸到 CPU 后每生成一个新 token 都得把历史 KV 拉回来算注意力——访问量大且随序列增长。"Attention Offloading"(arxiv 2405.01814)的思路正是把**注意力计算本身**也搬到能就近访问 KV 的设备上(如 CPU),而非反复搬数据。

## 4. ZeRO-Inference 思路:把权重当"流"

DeepSpeed 的 ZeRO-Inference(见 [[ai-framework/deepspeed/README]])专攻**权重 Offload**,目标是"单卡跑下超大模型,哪怕慢"。核心三招:

**(a) 层粒度托管**:全部权重常驻 CPU(或 NVMe),GPU 只在算到某层时才持有该层权重,算完立即释放。

**(b) 预取(prefetch)流水**:用 CUDA stream 把"搬下一层权重"和"算当前层"重叠:

```
时间轴 ──────────────────────────────────────────►
计算流:  [算L0]   [算L1]   [算L2]   [算L3] ...
拷贝流:[取L1]  [取L2]  [取L3]  [取L4] ...
           ↑ 算 L0 时,L1 已在路上
理想:计算把传输"盖住" → GPU 不空转
现实:若 T_取(Li) > T_算(L_{i-1}) → 计算流被迫等待(气泡)
```

**(c) 大 batch 摊薄**:权重搬一次,可以服务 batch 里所有样本。batch 越大,"每样本分摊的搬运成本"越低,吞吐越高。这是 Offload 提吞吐的命根子。

$$
\text{每样本搬运成本} = \frac{M_{\text{weight}} / \text{BW}_{\text{PCIe}}}{B} \xrightarrow{B\uparrow} \text{摊薄}
$$

> ZeRO-Inference 的定位:**延迟敏感场景别用**(单条请求慢得感人),**离线大批量吞吐场景**才是它的主场。

## 5. FlexGen 思路:把 Offload 当"搜索问题"

FlexGen(github FMInference/FlexGen)更激进:它认为"权重、KV、激活分别放哪一层、按什么顺序遍历"是一个**优化问题**,目标是单 GPU 上**最大化离线吞吐**。

**(a) 三维存储分配**:每类张量都可拆比例放到 GPU / CPU / NVMe 三层。比如权重 20% 在 GPU、50% 在 CPU、30% 在 NVMe。

**(b) 计算图遍历:zig-zag block schedule**。普通做法是"逐行"(一个 batch 走完所有层再下一个 batch),FlexGen 改成**按列块(block)**遍历——让一份权重加载后服务尽可能多的 batch,最大化权重复用:

```
        layer →   L0   L1   L2   L3
batch ↓
  row-by-row(差):  每行重新加载所有层权重 → 权重反复搬
        ┌──┬──┬──┬──┐
  b0    │1 │2 │3 │4 │   b0 走完 → b1 又把 L0..L3 权重搬一遍
  b1    │5 │6 │7 │8 │
        └──┴──┴──┴──┘

  block(好):一列(某层)权重加载后,把堆叠的多个 batch 都算掉
        ┌──┬──┬──┬──┐
  b0    │1 │3 │5 │7 │   L0 权重在 GPU 时,b0、b1 都用它算完
  b1    │2 │4 │6 │8 │   → 权重搬运次数减半
        └──┴──┴──┴──┘
        ↑列内复用权重
```

**(c) 量化压缩 + overlap**:FlexGen 还把权重/KV 压到 INT4 group-wise 量化,既省显存又省搬运字节数(搬得少=搬得快),并用多流重叠传输与计算。

> FlexGen 的标志性成果:在**单张 16GB T4** 上跑 OPT-175B 的离线推理,吞吐数量级超过简单 Offload baseline。代价依旧是**高延迟**——它明确为**吞吐导向的离线批处理**而生。

## 6. 带宽瓶颈与隐藏术(调度的核心)

Offload 工程 90% 的功夫花在"让 GPU 别等数据"。手段:

**① 异步拷贝 + 双缓冲(double buffering)**:开两块 GPU 缓冲区,一块给当前层计算用,另一块同时接收下一层。算完交换。

```
缓冲A: [L_i 权重] ← GPU 正在算
缓冲B: [L_{i+1} 权重] ← 拷贝流正在填
        算完 → swap(A,B),无需等拷贝从头开始
```

**② Pinned(页锁定)内存**:CPU 侧 buffer 用 `cudaHostAlloc` 锁页,DMA 才能跑满 PCIe 带宽。可分页内存会让拷贝带宽腰斩。

**③ 计算-传输重叠率决定一切**。定义重叠后有效时间:

$$
T_{\text{step}} = \max(T_{\text{compute}},\ T_{\text{transfer}}) + T_{\text{无法隐藏的尾部}}
$$

若 $T_{\text{transfer}} \gg T_{\text{compute}}$(典型 Offload 情形),则 $T_{\text{step}} \approx T_{\text{transfer}}$,GPU 利用率 $\approx T_{\text{compute}}/T_{\text{transfer}}$,常常只有 5%~20%。

**④ 减少搬运字节**:量化(INT8/INT4)直接把要过 PCIe 的字节数砍半/砍到 1/4,等价于带宽翻倍。这是为什么 Offload 几乎总和量化绑定。

## 7. 何时用 / 不该用(决策树)

```
显存够装下 [权重 + 峰值KV]?
   │
   ├─ 够 ──────────────────────────► 别 Offload,纯 GPU 最快
   │
   └─ 不够
        │
        ├─ 能加卡 / 多卡并行可行? ──► 优先张量并行/流水并行(快得多)
        │
        └─ 只有单卡 or 卡很小
             │
             ├─ 在意单条延迟(在线服务)? ──► Offload 很可能太慢,
             │                                考虑换小模型/更激进量化
             │
             └─ 离线、大批量、可接受慢? ──► ✅ Offload 主场
                                            (ZeRO-Inf / FlexGen)
```

一句话原则:**Offload 是"用得起 vs 用不起"的开关,不是"快 vs 更快"的旋钮。** 它把"跑不了"变成"能跑",但几乎一定比纯 GPU 慢一个量级。

## 数值示例:70B 模型在单张 24GB 4090 上跑

**场景**:Llama-70B 级别,FP16 权重,想在 24GB 卡上做离线批量生成。

**第 1 步:权重显存账**
$$
M_{\text{weight}} = 70\times10^9 \times 2\text{ B} = 140\text{ GB}
$$
24GB 显存连零头都不够,**权重必须 Offload 到 CPU DRAM**(假设主机有 256GB 内存,够放)。

**第 2 步:搬一遍全部权重要多久?**
PCIe Gen4 x16 实测有效带宽取 $25\text{ GB/s}$:
$$
T_{\text{搬权重一遍}} = \frac{140\text{ GB}}{25\text{ GB/s}} = 5.6\text{ s}
$$
**每生成一个 token 都要把 140GB 权重过一遍 PCIe**(每层算时取该层权重)。即单 token 至少 5.6 秒搬运 → 这就是延迟灾难的来源。

**第 3 步:计算时间对比(看谁是瓶颈)**
解码单 token 的算力约 $2N$ FLOPs(每参数一次乘加):
$$
\text{FLOPs}_{\text{token}} = 2 \times 70\times10^9 = 1.4\times10^{11}
$$
4090 FP16 算力约 $165\text{ TFLOP/s}$(实际利用打折,取理论上界):
$$
T_{\text{compute}} \approx \frac{1.4\times10^{11}}{1.65\times10^{14}} \approx 0.85\text{ ms}
$$
对比:$T_{\text{transfer}}=5600\text{ ms}$ vs $T_{\text{compute}}=0.85\text{ ms}$。

> **传输是计算的约 6600 倍!** 完美印证:Offload 彻底**带宽受限**。GPU 算力 99.98% 时间在闲置。

**第 4 步:大 batch 摊薄救场**
权重搬一遍能服务整个 batch。设 batch $B=64$,这 64 条共享同一次权重搬运:
$$
T_{\text{每条每token搬运}} = \frac{5.6\text{ s}}{64} \approx 0.0875\text{ s} = 87.5\text{ ms}
$$
吞吐(忽略计算,纯搬运受限):
$$
\text{Throughput} \approx \frac{B}{T_{\text{搬权重一遍}}} = \frac{64}{5.6\text{ s}} \approx 11.4\text{ tokens/s(聚合)}
$$
batch 提到 256:$\approx 256/5.6 \approx 45.7$ tokens/s 聚合。**batch 越大,吞吐越高**——这正是 Offload 一定要配大 batch 的根本原因。

**第 5 步:量化再省一半**
权重换 INT8($b=1$)→ 70GB → 搬运时间减半到 2.8s,吞吐翻倍。INT4 再翻倍。所以实战 Offload 几乎总开量化。

**结论数字感**:24GB 卡 + Offload 跑 70B,**离线大 batch 能到几十 tokens/s 聚合吞吐**(可用于批量数据生成),但**单条请求延迟在秒级/token**(在线服务不可接受)。

## 对照/复杂度表

| 方案 | 卸载对象 | 主存储层 | 调度核心 | 定位 | 单条延迟 | 大batch吞吐 |
| --- | --- | --- | --- | --- | --- | --- |
| HF Accelerate `device_map` | 权重 | CPU/Disk | 朴素层切分,无重叠优化 | 能跑就行 | 很高 | 低 |
| ZeRO-Inference | 权重(+NVMe) | CPU/NVMe | 层流水+预取+大batch | 超大模型单卡离线 | 高 | 中-高 |
| FlexGen | 权重+KV+激活 | GPU/CPU/NVMe 三层 | 搜索最优分配 + zigzag block | 单卡极限吞吐 | 很高 | 最高 |
| Attention Offloading(2405.01814)| KV+注意力计算 | CPU | 把 attn 算子搬到近 KV 端 | 长上下文省 KV 显存 | 中 | 高 |
| vLLM CPU/KV offload | KV Cache | CPU | 块表 + 按需换入换出 | 在线扩容上下文 | 中 | 中 |
| 纯多卡张量并行(对照) | 不卸载 | 全 HBM | NVLink 通信 | 在线低延迟 | 低 ✅ | 高 |

复杂度直觉:Offload 把空间复杂度从 $O(\text{全模型})_{\text{HBM}}$ 降到 $O(\text{单层})_{\text{HBM}} + O(\text{全模型})_{\text{DRAM/NVMe}}$,代价是每步多付 $O(\text{搬运字节}/\text{BW})$ 的时间。

## 常见问题 / 高频追问

| 问题 | 踩点答案 | 追问/陷阱 |
| --- | --- | --- |
| Offload 为什么慢? | 数据要过 PCIe(~25GB/s),比 HBM(~3TB/s)慢约百倍,几乎总是带宽受限,GPU 大量空转 | 追问:瓶颈是算力还是带宽?→ **带宽**,且常差几千倍 |
| 为什么 Offload 一定配大 batch? | 权重搬一次可服务整个 batch,大 batch 摊薄每样本搬运成本,提升吞吐 | 追问:batch 太大会怎样?→ KV 和激活显存又会涨,需平衡 |
| 权重和 KV 哪个更适合卸载? | 权重(顺序可预取),KV 难(随机+随序列增长+每步都要读历史) | 追问:KV 怎么办?→ 量化KV/近窗保留/Attention Offloading |
| ZeRO-Inference 和 FlexGen 区别? | ZeRO-Inf 主攻权重层流水(NVMe 也行);FlexGen 把权重/KV/激活三层分配做成搜索,zigzag 复用权重,极限离线吞吐 | 追问:都适合在线吗?→ 都不适合,延迟太高 |
| 怎么把搬运时间藏起来? | 双缓冲 + pinned 内存 + 异步 CUDA stream,让"搬下一层"和"算当前层"重叠 | 追问:能完全藏住吗?→ 仅当 T_算 ≥ T_搬,Offload 时通常藏不住 |
| 在线低延迟服务能用 Offload 吗? | 基本不行,单条延迟秒级;优先多卡张量并行 / 换小模型 / 更激进量化 | 陷阱:别拿 Offload 救在线 P99 |
| Offload 和量化什么关系? | 量化减少过 PCIe 的字节数 = 等效带宽翻倍,几乎必配 | 追问:INT4 比 FP16 搬运快几倍?→ 4× |
| NVMe Offload 什么时候才用? | CPU DRAM 也装不下时(如单机内存 < 模型大小),用 NVMe 兜底,更慢但能跑 | 追问:NVMe 带宽?→ ~3-7GB/s,比 DRAM 再慢约 10× |

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航,Offload 在推理优化版图中的位置
- [[llm-optimizer/kv-cache]] — KV Cache 原理与显存账(Offload 的主要卸载对象之一)
- [[ai-framework/deepspeed/README]] — DeepSpeed / ZeRO-Inference 工程实现

参考资料:
- HF Accelerate 大模型推理:https://huggingface.co/docs/accelerate/concept_guides/big_model_inference
- HF Big Models:https://huggingface.co/docs/transformers/big_models
- Attention Offloading:https://arxiv.org/pdf/2405.01814
- DeepSpeed Inference:https://arxiv.org/pdf/2207.00032
- FlexGen:https://github.com/FMInference/FlexGen
- FlexFlow:https://github.com/flexflow/FlexFlow
- TensorRT-LLM KV cache reuse:https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/kv_cache_reuse.md
