# AI 芯片

> 用一张算力账本看懂"显存装得下、算力跑得动、数值不溢出"这三件大事——AI 芯片就是把这三件事压进硅片里的产物。📍 导航：[[00-知识地图]]
> 🔗 相关：[[GPU工作原理]] · [[硬件对比]] · [[昇腾NPU]] · [[AI芯片软件生态]] · [[大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 章节 | 你将搞清楚的问题 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | AI 芯片到底在卖什么 | 算力·显存·带宽 |
| 1. 地基 | 看懂参数表前要懂的 5 个量纲 | FLOPS·HBM·CUDA/Tensor Core |
| 2. 数值精度 | FP32/TF32/FP16/BF16/FP8/INT8 谁管什么 | 指数位·尾数位·溢出 |
| 3. CUDA Core vs Tensor Core | 为什么 Tensor 算力大一个数量级 | MAC 阵列·脉动 |
| 4. 显存与带宽 | 为什么推理常常"带宽墙"而非"算力墙" | HBM·GDDR·Roofline |
| 5. 特供卡 | A800/H800/H20/L20 被砍了什么 | 互联带宽·算力阉割 |
| 6. 国产芯片 | 昇腾达芬奇、摩尔线程 MUSA | NPU·统一架构 |
| 实操 | 主流 AI 芯片完整配置对照 | 参数表 |
| 常见坑 | 标称算力 vs 实测、TOPS vs TFLOPS | 利用率·稀疏 |

## 0. 一句话锚点

**AI 芯片卖的是三个数字的乘积兑现率**：算力（每秒多少次乘加）、显存容量（一次能装多少参数/激活/KV Cache）、显存带宽（每秒能从显存搬多少字节进计算单元）。参数表里一长串 TFLOPS，本质都是在描述这三件事的某个切面。

```
                AI 芯片 = 把矩阵乘加压进硅片
   ┌───────────────────────────────────────────────────┐
   │  显存(HBM/GDDR)  ──带宽──►  计算单元(Tensor/CUDA)   │
   │   "装得下"          "搬得动"        "算得快"        │
   │   容量 GB          带宽 TB/s        算力 TFLOPS     │
   └───────────────────────────────────────────────────┘
   训练:三者都吃紧   推理:常被"搬得动"卡住(带宽墙)
```

## 1. 地基：看懂参数表前的 5 个量纲

读懂下方的大表，先把这 5 个词钉死：

- **FLOPS / TFLOPS**：Floating-point Operations Per Second，每秒浮点运算次数。1 TFLOPS = $10^{12}$ 次/秒。一次"乘加"(MAC, $a\times b+c$) 通常计为 2 个 FLOP。
- **TOPS**：Tera Operations Per Second，整数运算（如 INT8）用 OPS 计数，不叫 FLOPS。所以表里 INT8 一栏写的是 **TOPS** 不是 TFLOPS——这是个高频笔误点。
- **HBM / GDDR**：显存类型。HBM（High Bandwidth Memory，3D 堆叠，贴在 GPU 旁）带宽极高，用于数据中心卡；GDDR（GDDR6/6X，平铺在 PCB 上）成本低带宽次之，用于消费/推理卡。
- **CUDA Core**：英伟达的通用标量/向量计算单元，干"什么都能算但不专"的活（激活函数、逐元素、reduce）。
- **Tensor Core**：专做矩阵乘加（MatMul）的硬件阵列，是 Transformer 训练/推理算力的主力。表里凡标"(Tensor Core)"的算力，都是它贡献的。

```
一个 GPU 的算力来自两条腿:
   CUDA Core  ── 标量/向量, 通用, 算力 = 表中 FP32 那列(不带Tensor Core标注)
   Tensor Core── 矩阵 MAC 阵列, 专用, 算力 = 表中带"(Tensor Core)"那列, 大一个数量级
```

> 为什么要分两条腿？因为 Transformer 里 ~90% 的浮点运算是矩阵乘（QKV 投影、FFN、Attention），把这部分交给专用阵列，能用同样的晶体管做出十倍算力。参见 [[GPU工作原理]]。

## 2. 数值精度：一张表分清 6 种数据类型

参数表横向那一排 FP32 / TF32 / FP16 / BF16 / FP8 / INT8，本质是"用多少 bit、怎么分配指数位和尾数位"。**指数位决定能表示的数值范围（防溢出），尾数位决定精度（防舍入误差）。**

| 类型 | 总位宽 | 指数位 | 尾数位 | 动态范围 | 典型用途 |
| --- | --- | --- | --- | --- | --- |
| FP32 | 32 | 8 | 23 | 大 | 主权重副本、优化器状态、精度基准 |
| TF32 | 19(占32) | 8 | 10 | 同 FP32 | Ampere+ 自动加速 MatMul，范围同 FP32 精度降 |
| FP16 | 16 | 5 | 10 | 小($\pm6.5\times10^4$) | 推理/混合精度，但易上溢，需 loss scaling |
| BF16 | 16 | 8 | 7 | 大(同FP32) | 训练首选：范围同 FP32，几乎不溢出 |
| FP8(E4M3/E5M2) | 8 | 4或5 | 3或2 | 很小 | Hopper+ 训练/推理，需 per-tensor 缩放 |
| INT8 | 8 | — | — | 整数 | 量化推理，配 scale/zero-point |

数值直觉（手算）：

- FP16 最大正规数 $\approx (2-2^{-10})\times 2^{15} \approx 6.55\times10^4$。训练时梯度若超过这个值就变 `inf` → 必须 loss scaling。
- BF16 指数位和 FP32 一样是 8 位，最大约 $3.4\times10^{38}$，所以训练几乎不溢出，代价是尾数只有 7 位（精度低于 FP16）。**结论：训练用 BF16 图稳，推理可用 FP16/FP8 图省**。
- FP8 → BF16 → FP32，每降一档，**同样的晶体管算力翻倍、显存占用减半**。这就是为什么表里 H800 的 FP8 算力(3,958 TFLOPS)恰好是 BF16(1,979)的两倍。

```
位宽与算力/显存的反比关系(同一代硬件,粗略):
   FP32  ──算力×1   显存×1      (基准)
   TF32  ──算力×8   显存×1      (Ampere Tensor Core 加速)
   BF16  ──算力×16  显存×0.5
   FP8   ──算力×32  显存×0.25   (Hopper)
   位宽减半 ⇒ 一次能搬2倍数据 + 算力翻倍 ⇒ 大模型省钱核心杠杆
```

> 量化把这个杠杆用到极致：[[量化基础]] / [[GPTQ]] / [[fp8]]。

## 3. CUDA Core vs Tensor Core：为什么差一个数量级

观察表中 RTX 4090：FP16 那列写 **369.7 TFLOPS（Tensor Core）82.58 TFLOPS**——两个数。前者是 Tensor Core 跑矩阵乘，后者是 CUDA Core 跑通用 FP16。**差了约 4.5 倍**，这不是笔误，而是两种硬件结构的本质差异。

```
CUDA Core(标量MAC):  每周期 1 个乘加
   for i: c[i] = a[i]*b[i] + c[i]    ← 一次一个数

Tensor Core(脉动阵列): 每周期一整块小矩阵 MAC
   ┌───────────────┐
   │  D = A·B + C  │   A,B,C,D 是 4×4 / 8×8 小块
   │ (一拍出整块)  │   一拍完成 ~64 次乘加
   └───────────────┘
   把矩阵乘"硬件化"成一个二维 MAC 阵列, 数据像波一样流过(脉动)
```

要点：

- Tensor Core 只擅长一件事——**固定尺寸的矩阵块乘加**。Transformer 恰好全是大矩阵乘，所以契合度极高。
- 表里数据中心卡（A800/H800/H20）算力动辄上千 TFLOPS，全靠 Tensor Core；它们的 FP32（CUDA Core，如 H800 的 67 TFLOPS）反而很普通。
- 推论：**评估一张卡能不能跑大模型，看带"(Tensor Core)"的 BF16/FP16 算力，而不是 FP32。** 这是新手最容易看错的一栏。

数值示例（手算 Tensor Core 加速比）：以 RTX 4090 FP16 为例，CUDA Core 82.58 TFLOPS、Tensor Core 369.7 TFLOPS，加速比 $369.7 / 82.58 \approx 4.48$。也就是说，把一个 $[M,K]\times[K,N]$ 的矩阵乘交给 Tensor Core，理论上比让 CUDA Core 硬算快约 4.5 倍——这正是所有深度学习框架默认走 Tensor Core 路径的原因。

## 4. 显存容量 / 带宽：推理为什么常撞"带宽墙"

算力大不等于跑得快。Transformer **解码阶段**每生成一个 token，要把整个模型权重 + KV Cache 从显存读一遍到计算单元，运算量却很小（一个 token 的矩阵-向量乘）。此时瓶颈是**带宽**不是**算力**。

```
Roofline 思维:
   达到的性能 = min( 算力上限,  带宽 × 算术强度 )
                         ▲              ▲
                    算力墙(训练/大batch)  带宽墙(单条解码/小batch)

   算术强度 = 计算量(FLOP) / 访存量(Byte)
   解码单 token: 算术强度极低 ⇒ 几乎一定撞带宽墙
```

实践含义：

- **训练**：大 batch、大矩阵乘 → 算术强度高 → 吃算力（看 BF16 TFLOPS）。
- **推理解码**：算术强度低 → 吃带宽和显存容量。所以 H20（算力被砍到 148 TFLOPS，但显存高达 96GB HBM3、带宽保留）反而是"特供推理/训练混用"的设计取向。
- 显存容量直接决定能不能装下模型 + KV Cache。粗算：一个参数 BF16 占 2 字节，7B 模型权重 ≈ 14GB；KV Cache 另算，序列越长越大。容量不够就得多卡切分。延伸阅读 [[transformer内存估算]] / [[kv-cache]]。

> 缓解带宽墙的工程手段：FlashAttention（减少 HBM 读写次数，见 [[FlashAttention]]）、KV Cache 量化、PD 分离（Prefill 吃算力、Decode 吃带宽，拆到不同卡，见 [[PD分离]]）。

## 5. 特供卡：A800 / H800 / H20 / L20 被砍了什么

表里带"特供"字样的型号，是为合规市场设计的"减配版"。理解它们要看**砍在哪里**：

| 卡 | 对标原版 | 主要阉割点 | 保留 | 定位 |
| --- | --- | --- | --- | --- |
| A800 | A100 | NVLink 互联带宽降（600→400 GB/s） | 算力、显存基本不变 | 训练（多卡通信受限） |
| H800 | H100 | 互联带宽降 | 单卡算力几乎满血(BF16 1,979 TFLOPS) | 训练 |
| H20 | H100 | **算力大砍**(BF16 仅 148 TFLOPS) | 显存 96GB HBM3、带宽高 | 推理为主、可训练 |
| L20 | L40 系 | 算力受限 | 48GB GDDR6 | 推理（PCIe） |

```
两种"砍法":
   砍互联带宽(A800/H800):  单卡能打, 但多卡组训练时 AllReduce 慢
                          → 影响 [[集合通信原语]] / [[InfiniBand]] 组网效率
   砍算力(H20/L20):        单卡算力弱, 但显存/带宽够 → 偏推理友好
```

> 为什么砍互联会拖慢训练？因为数据/张量/流水并行都要跨卡同步梯度（AllReduce），互联带宽降 1/3，通信时间就拉长，整机吞吐下降。互联与集合通信见 [[InfiniBand]] / [[集合通信原语]]。

## 6. 国产 AI 芯片：昇腾达芬奇 与 摩尔线程 MUSA

表底两行是华为昇腾：

- **架构叫"达芬奇"**（Da Vinci），是华为自研的 NPU（Neural-network Processing Unit）架构，核心是 3D Cube 矩阵计算单元，对标 Tensor Core 的角色。详见 [[昇腾NPU]]。
- **Atlas 800T A2 训练（910B3-HCCS）**：64GB HBM2e，BF16 313 TFLOPS。"HCCS"是华为的卡间高速互联（对标 NVLink），用于多卡训练组网。
- **Atlas 800I 推理（910B4）**：32GB HBM2e，算力略低（BF16 280 TFLOPS），定位推理。
- 表中 TF32 一栏标"（HF）"——指华为自有的近似 TF32 高精度浮点格式，不是英伟达 TF32，互不等价。

摩尔线程（原文真料）：

> 2022 年，摩尔线程推出了 GPU 统一系统架构 **MUSA**（Meta-computing Unified System Architecture），发布并量产"苏堤"和"春晓"两颗全功能 GPU 芯片，这也是国内采用现代 GPU 架构的代表。

要点解读：

- **"全功能 GPU"** 指同时支持 AI 计算、图形渲染、视频编解码、科学计算，而非只做 AI 的专用加速卡——这是 MUSA 区别于纯 NPU 路线的卖点。
- **统一架构（MUSA）** 类比 CUDA：一套编程模型 + 软件栈覆盖多代芯片，目标是降低生态迁移成本。国产软件生态适配见 [[AI芯片软件生态]]。

```
两条国产路线:
   昇腾(达芬奇 NPU): 专精 AI, Cube 矩阵单元, CANN 软件栈, HCCS 互联
   摩尔线程(MUSA GPU): 全功能 GPU(AI+图形+视频), 对标 CUDA 生态
```

## 实操：主流 AI 芯片完整配置对照

> 下表为原始权威数据，原样保留。读法：训练看带"(Tensor Core)"的 BF16；推理重点看显存容量/带宽 + INT8 TOPS；"特供"卡注意阉割点（见第 5 节）。

| 厂商  | 型号                          | 图形处理器        | 架构           | 显存           | FP16 算力                               | BF16 算力                      | INT8 算力                   | FP32算力       | TF32 算力                     | FP8算力                        | CUDA Core | Tensor Core |
| --- | --------------------------- | ------------ | ------------ | ------------ | ------------------------------------- | ---------------------------- | ------------------------- | ------------ | --------------------------- | ---------------------------- | --------- | ----------- |
| 英伟达 | RTX 3090                    | GA102-300-A1 | Ampere       | 24GB（GDDR6X） | 35.58 TFLOPS                          | -                           | -                        | 35.58 TFLOPS | -                          | 不支持                          | 10496     | 328         |
| 英伟达 | RTX 3090 Ti                 | GA102-350-A1 | Ampere       | 24GB（GDDR6X） | 40.00 TFLOPS                          | -                           | -                        | 40.00 TFLOPS | -                          | 不支持                          | 10752     | 336         |
| 英伟达 | RTX 4090                    | AD102-300-A1 | Ada Lovelace | 24GB（GDDR6X） | 369.7 TFLOPS（Tensor Core）82.58 TFLOPS | 369.7 TFLOPS（Tensor Core）    | 739.4 TFLOPS（Tensor Core） | 82.58 TFLOPS | -                          | -                           | 16384     | 512         |
| 英伟达 | RTX 4090 Ti                 | AD102-400-A1 | Ada Lovelace | 24GB（GDDR6X） | 93.24 TFLOPS                          | -                           | -                        | 93.24 TFLOPS | -                          | -                           | 18176     | 568         |
| 英伟达 | RTX 4090D-特供-消费级            | AD102-250-A1 | Ada Lovelace | 24GB（GDDR6X） | 329.3 TFLOPS（Tensor Core）73.54 TFLOPS | 329.3 TFLOPS（Tensor Core）    | 658.6 TFLOPS（Tensor Core） | 73.54 TFLOPS | -                          | -                           | 14592     | 456         |
| 英伟达 | L20（PCIe）-特供-推理（PCIe）       | AD102        | Ada Lovelace | 48GB（GDDR6）  | 119.5 TFLOPS（Tensor Core）             | 119.5 TFLOPS（Tensor Core）    | 239 TOPS（Tensor Core）     | 59.8 TFLOPS  | 59.8 TFLOPS（Tensor Core）    | 239 TFOPS（Tensor Core）       | 11776     | 368         |
| 英伟达 | H20-特供-训练（PCIe、Nvlink）      | -           | Hopper       | 96GB（HBM3）   | 148 TFLOPS（Tensor Core）               | 148 TFLOPS（Tensor Core）      | 296 TOPS（Tensor Core）     | 44 TFLOPS    | 74 TFLOPS（Tensor Core）      | 296 TFOPS（Tensor Core）       | -        | -          |
| 英伟达 | A800（PCIe）                  | GA100        | Ampere       | 80GB（HBM2e）  | 312 TFLOPS（Tensor Core）77.97 TFLOPS   | 312 TFLOPS（Tensor Core）      | 624 TOPS（Tensor Core）     | 19.5 TFLOPS  | 156 TFLOPS（Tensor Core）     | 不支持                          | 6912      | 432         |
| 英伟达 | H800（ SXM）                  | GH100        | Hopper       | 80GB（HBM3）   | 1,979 TFLOPS（Tensor Core）             | 1,979 teraFLOPS（Tensor Core） | 3,958 TOPS（Tensor Core）   | 67 teraFLOPS | 989 teraFLOPS （Tensor Core） | 3,958 teraFLOPS（Tensor Core） | 18,432    | 640         |
| 昇腾  | Atlas 800T A2训练（910B3-HCCS） | -           | 达芬奇          | 64GB（HBM2e）  | 313 TFLOPS                            | 313 TFLOPS                   | 640 TOPS                  | 75 TFLOPS    | 141 TFLOPS（HF）              | 不支持                          | -        | -          |
| 昇腾  | Atlas 800I 推理（910B4）        | -           | 达芬奇          | 32GB（HBM2e）  | 280 TFLOPS                            | 280 TFLOPS                   | 550 TOPS                  | 75 TFLOPS    | 141 TFLOPS（HF）              | 不支持                          | -        | - |

**横向对照速读**：

- 找训练旗舰 → H800：BF16 1,979 TFLOPS、FP8 3,958 TFLOPS（FP8 恰为 BF16 两倍，印证第 2 节"位宽减半算力翻倍"）。
- 找大显存推理 → H20：算力虽砍到 148 TFLOPS，但 96GB HBM3 是表中最大，装大模型 + 长 KV Cache 友好。
- 消费级跑模型 → RTX 4090：BF16 369.7 TFLOPS（Tensor Core）性价比高，但显存仅 24GB GDDR6X，大模型需多卡或量化。
- 国产训练 → Atlas 800T A2：BF16 313 TFLOPS、64GB HBM2e，与 A800 同档位。

## 常见问题 / 坑

| 坑 | 现象 | 真相 / 处理 |
| --- | --- | --- |
| INT8 写成 TFLOPS | 把 INT8 那列当浮点算力比 | INT8 是整数运算，单位是 **TOPS** 不是 TFLOPS（表里 INT8 栏用 TOPS） |
| 只看 FP32 选卡 | 觉得 H800 FP32 才 67 TFLOPS 很弱 | 大模型吃的是带 **(Tensor Core)** 的 BF16/FP16，FP32 只是 CUDA Core 基准 |
| 标称算力当实测 | 实际吞吐远低于 TFLOPS | 标称是峰值；真实利用率(MFU)常仅 30%~50%，受带宽、通信、kernel 效率制约 |
| 稀疏算力误读 | 看到双倍算力以为通用 | 部分标称含 2:4 结构化稀疏，需模型满足稀疏模式才达到，稠密模型打对折 |
| 特供卡当满血 | 以为 H800=H100 | H800/A800 砍**互联带宽**→多卡训练通信慢；H20/L20 砍**算力**→偏推理（见第5节） |
| 推理只堆算力 | 买高算力卡解码却不快 | 解码撞**带宽墙**，应看显存带宽 + 容量，而非峰值 TFLOPS（见第4节 Roofline） |
| TF32(HF) 当 NV TF32 | 昇腾 TF32 与英伟达直接比 | 昇腾标"（HF）"是华为自有高精度格式，与 NV TF32 不等价，跨平台勿直接对标 |
| 忽略显存类型 | 拿 GDDR 卡跑大 batch 训练 | HBM 带宽 >> GDDR；GDDR 卡(如 L20)更适合推理，训练带宽吃紧 |

## 🔗 跳转链接

- 知识地图枢纽：[[00-知识地图]]
- 硬件原理：[[GPU工作原理]] · [[硬件对比]] · [[昇腾NPU]] · [[AI芯片软件生态]]
- 网络互联：[[InfiniBand]] · [[集合通信原语]]
- 精度与压缩：[[量化基础]] · [[GPTQ]] · [[fp8]] · [[README]]（llm-compression）
- 推理优化：[[FlashAttention]] · [[kv-cache]] · [[PD分离]] · [[解码策略]]
- 推理框架：[[README]]（llm-inference）· [[README]]（vllm）
- 训练框架：[[README]]（deepspeed）· [[README]]（megatron-lm）· [[README]]（pytorch）
- 模型与对齐：[[模型架构]] · [[README]]（moe）· [[旋转编码RoPE]] · [[RLHF]] · [[DPO]] · [[PEFT-API]]
- 评测与估算：[[大模型场景下训练和推理性能指标名词解释]] · [[transformer内存估算]] · [[README]]（llm-eval）
