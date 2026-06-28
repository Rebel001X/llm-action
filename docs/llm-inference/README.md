# 大模型推理（LLM Inference）总览

> 把一段 Prompt 喂给训练好的大模型、一个 token 一个 token 吐出回答的全过程，就是「推理」。本篇从最底层把它讲清：算力打在哪、显存被谁吃掉、为什么解码这么慢、以及业界用哪些招把它变快变省。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/KV-Cache优化]]　[[llm-inference/解码策略]]　[[llm-inference/大模型推理张量并行]]　[[llm-optimizer/FlashAttention]]　[[llm-algo/FLOPs]]　[[llm-algo/transformer/模型架构]]

## 阅读地图

| 节 | 你会学到 | 关键产出 |
|----|----------|----------|
| 0 | 推理一句话锚点 | Prefill + Decode 两阶段心智模型 |
| 1 | 地基：自回归 / Transformer 前向 | 为什么必须逐 token |
| 2 | Prefill vs Decode 两阶段 | 一个算力受限、一个带宽受限 |
| 3 | KV-Cache 是什么、省了什么 | $O(n^2)\to O(n)$ |
| 4 | 显存账本：权重/KV/激活 | 逐数手算 70B 推理显存 |
| 5 | 为什么 Decode 是带宽受限 | 算术强度 + Roofline |
| 6 | 吞吐 vs 延迟 + 评测指标 | TTFT / TPOT / Throughput |
| 7 | Continuous Batching | GPU 利用率从 30% 到 90% |
| 8 | 并行：TP / PP / EP | 单卡放不下怎么办 |
| 9 | 加速三板斧：量化/FlashAttn/投机采样 | 各自打哪个瓶颈 |
| 10 | 服务框架全景 | vLLM / TGI / TRT-LLM / SGLang |
| 手算 | 端到端数值示例 | 一道完整估算题 |

---

## 0. 一句话锚点

> **推理 = 给定模型权重 $W$ 与输入 token 序列，自回归地、一次一个地预测下一个 token，直到 EOS。**

它天然分成两段，记住这张图就抓住了一切：

```
 用户输入 "中国的首都是"
        │
        ▼
 ┌──────────────┐   一次性并行处理全部输入 token
 │  Prefill 预填 │   (计算密集 / compute-bound)
 └──────┬───────┘   产出：首个 token + 整段 KV-Cache
        │  "北"
        ▼
 ┌──────────────┐   每步只处理 1 个新 token，循环 N 次
 │ Decode 解码  │   (访存密集 / memory-bound)
 └──────┬───────┘   "京" → "，" → "是" → ... → <EOS>
        ▼
   完整回答输出
```

两段的优化目标、瓶颈、甚至适用的硬件特性都不同——这是理解推理优化的总开关。

---

## 1. 地基：自回归与 Transformer 前向

**1.1 自回归（Autoregressive）**。语言模型建模联合概率，用链式法则拆开：

$$P(x_1,\dots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_1,\dots,x_{t-1})$$

每个新 token 的分布**依赖前面所有 token**。所以生成第 $t$ 个 token 必须等第 $t-1$ 个算完——**串行**，这是 Decode 慢的根本原因，无法靠堆算力消除。

**1.2 一次前向算了什么**。一个 decoder-only Transformer 由 $L$ 层堆叠，每层做：

```
 输入 hidden  x  (维度 d_model = d)
   │
   ▼  ── 自注意力 (Self-Attention) ──
 [Q K V 投影]  x·W_q, x·W_k, x·W_v
   │
 [注意力分数]  softmax(QKᵀ/√d_h) · V   ← 这里要用到历史 K,V
   │
 [输出投影]   ·W_o    + 残差 + LayerNorm
   │
   ▼  ── 前馈网络 (FFN/MLP) ──
 [up]  x·W1 (d → 4d)  → 激活(GELU/SwiGLU)
 [down] ·W2 (4d → d)  + 残差 + LayerNorm
   │
   ▼ 输出 hidden 给下一层
```

最后一层输出过 `lm_head`（$d\to V$ 词表）得到 logits，再采样出下一个 token。
> 📎 架构细节见 [[llm-algo/transformer/模型架构]]；FFN/MLP 见 [[llm-algo/mlp]]；位置编码见 [[llm-algo/旋转编码RoPE]]。

---

## 2. 两阶段：Prefill 与 Decode

| | Prefill（预填充） | Decode（解码） |
|---|---|---|
| 处理 token 数 | 整段 prompt，$n$ 个 | 每步 1 个 |
| 并行度 | 高（矩阵×矩阵 GEMM） | 低（矩阵×向量 GEMV） |
| 瓶颈 | **算力 compute-bound** | **带宽 memory-bound** |
| 关键指标 | TTFT（首 token 延迟） | TPOT（每 token 间隔） |
| KV-Cache | 一次性写满 prompt 的 KV | 每步追加 1 列 KV |

```
Prefill:  [t1 t2 t3 t4 t5]  ──同时──►  矩阵×矩阵 (GEMM)，GPU 吃满
                              产出 5 列 KV + 第 6 个 token

Decode:        [t6]          ──单个──►  矩阵×向量 (GEMV)，GPU 闲置等访存
   step1                       读全部权重，只算 1 个 token
   step2   [t7]  ...           再读一遍全部权重 ...（权重反复搬运）
```

**核心直觉**：Decode 每步都要把**整个模型权重**从显存搬到计算单元，却只算 1 个 token 的乘加——计算单元大部分时间在「等数据」。这就是为什么 Decode 是访存密集，也是 KV-Cache、量化、投机采样发力的战场。

---

## 3. KV-Cache：用空间换时间

**问题**：Decode 第 $t$ 步算注意力需要 $Q_t$ 与**所有历史** $K_{1..t}, V_{1..t}$。若每步都重算历史 K/V，则总计算量是 $O(n^2)$。

**做法**：把每步算出的 $K_t,V_t$ **存起来**，下一步直接复用，只新算当前 token 的 1 列。计算量降到 $O(n)$。

```
无 Cache（重算）         有 KV-Cache（追加）
step t:                  step t:
 重算 K1..Kt  (t 列)      读缓存 K1..K(t-1) + 新算 Kt (1 列)
 重算 V1..Vt             读缓存 V1..V(t-1) + 新算 Vt
 → 每步 O(t) 重复劳动     → 每步 O(1) 新增 + O(t) 访存
```

**代价**：KV-Cache 占显存，且随**序列长度线性增长**、随 **batch 线性增长**。它是长上下文 / 高并发场景的头号显存杀手。
> 📎 PagedAttention、量化 KV、MQA/GQA 等优化见 [[llm-inference/KV-Cache优化]] 与 [[llm-optimizer/kv-cache]]。

**KV-Cache 显存公式**（单位 byte）：

$$\text{KV} = 2 \times L \times n \times b \times d_{kv} \times \text{batch} \times \text{precision}$$

其中 2 = K 和 V 两份；$L$=层数；$n$=序列长度；$d_{kv}$=KV 头维度总和（MHA 时 $=d_{model}$，GQA 时按 KV 头数缩小）；precision=每元素字节数（FP16=2）。

---

## 4. 显存账本：推理显存花在哪

推理显存 ≈ **模型权重 + KV-Cache + 激活/临时buffer + 框架开销**。

```
┌─────────────────────────────────────────┐
│  模型权重 (固定)        ← 最大头，与并发无关 │
├─────────────────────────────────────────┤
│  KV-Cache (∝ batch×seq) ← 高并发/长文爆点  │
├─────────────────────────────────────────┤
│  激活 / 临时 buffer     ← 推理比训练小很多  │
├─────────────────────────────────────────┤
│  框架/CUDA context 开销 ← 几 GB 起跳       │
└─────────────────────────────────────────┘
```

**权重显存** $= \text{参数量} \times \text{每参数字节}$。FP16 → 每参数 2 字节，INT8 → 1，INT4 → 0.5。
> 📎 训练侧的优化器/梯度账本见 [[transformer内存估算]]；量化压权重见 [[llm-compression/quantization/量化基础]]、[[llm-compression/quantization/fp8]]。

---

## 5. 为什么 Decode 是带宽受限：算术强度与 Roofline

**算术强度（Arithmetic Intensity）** $= \dfrac{\text{FLOPs}}{\text{字节访存}}$，单位 FLOP/byte。它决定一个 kernel 是算力受限还是带宽受限。

```
性能
 ▲           ┌──────────── 算力屋顶 (peak FLOPS)
 │          /│   compute-bound 区
 │  带宽   / │
 │  斜坡  /  │
 │      /    │
 │     / Decode 落在这里（强度低，带宽受限）
 │    /  ●   │            ● Prefill（强度高，算力受限）
 └───┴───────┴──────────►  算术强度 (FLOP/byte)
     转折点 = peak_FLOPS / 带宽
```

- **Prefill**：GEMM，一次权重读取服务多个 token，算术强度高 → 撞到算力屋顶。
- **Decode**：GEMV，batch=1 时一次权重读取只服务 1 个 token，算术强度极低 → 被显存带宽卡死。

**这给出关键优化方向**：Decode 阶段**把 batch 做大**（多请求共享同一次权重读取）能成倍提升算术强度——这正是 Continuous Batching 的理论依据。
> 📎 算力/带宽硬件原理见 [[ai-infra/算力/GPU工作原理]]、[[ai-infra/ai-hardware/CUDA]]；FLOPs 估算见 [[llm-algo/FLOPs]]。

---

## 6. 吞吐 vs 延迟：评测指标

二者常此消彼长，必须分清你优化的是哪个：

| 指标 | 含义 | 谁关心 |
|------|------|--------|
| **TTFT** Time To First Token | 从请求到吐出第一个 token | 交互体验（聊天） |
| **TPOT** Time Per Output Token | 后续每个 token 的间隔 | 流式顺滑度 |
| **Latency** 端到端 | TTFT + TPOT×输出长度 | 单请求快慢 |
| **Throughput** 吞吐 | 系统每秒总 token 数 | 服务成本/QPS |
| **Goodput** | 满足 SLO 前提下的有效吞吐 | 生产真实指标 |

```
延迟优先（小 batch）        吞吐优先（大 batch）
 单请求快，GPU 利用率低       单请求略慢，GPU 吃满
 ┌─┐                        ┌─┬─┬─┬─┬─┬─┬─┬─┐
 │█│   闲   闲   闲          │█│█│█│█│█│█│█│█│
 └─┘                        └─┴─┴─┴─┴─┴─┴─┴─┘
 适合：低 QPS 交互           适合：离线批处理/高并发
```

---

## 7. Continuous Batching（连续批处理）

**静态批处理**的痛点：一个 batch 里各请求生成长度不同，短的早早结束却得**等最长的**，GPU 在等待中空转。

```
静态批 (Static)                连续批 (Continuous)
req A ████░░░░░░  (早完，空等)   req A ████▶[换入 req D]
req B ██████████                 req B ██████████
req C ██░░░░░░░░  (早完，空等)   req C ██▶[换入 req E]
      ↑ 整批一起放走             ↑ 谁完谁立刻被新请求顶替
GPU 利用率 ~30-40%              GPU 利用率 ~80-90%
```

**做法**：以 **iteration（每生成一个 token）为粒度**调度，完成的请求立刻离场，新请求随时插入空位。配合 PagedAttention（KV 像虚拟内存分页管理，消除碎片）是 vLLM 高吞吐的核心。
> 📎 见 [[llm-inference/llm推理框架]] 与 [[llm-inference/vllm]]；KV 分页见 [[llm-inference/KV-Cache优化]]。

---

## 8. 并行：单卡放不下怎么办

当模型权重 + KV 超过单卡显存，或要降延迟时，沿不同维度切：

```
张量并行 TP        流水并行 PP         专家并行 EP
切「层内矩阵」      切「层间」          切「MoE 专家」
每层都要 AllReduce  层间传 activation  按 token 路由到专家
通信频繁→同机内     通信少→跨机        用于稀疏 MoE
(NVLink)           (慢链路可)
```

- **张量并行 TP**：把每层的权重矩阵按行/列切到多卡，每层前向后做 **AllReduce** 合并。通信量大、频次高，要 NVLink 高带宽，一般限同机 8 卡内。
- **流水并行 PP**：不同层放不同卡，像流水线传递激活；通信少但有「气泡」空闲。
- **专家并行 EP**：MoE 模型把不同专家放不同卡，按路由分发 token。
> 📎 TP 推理细节与通信手算见 [[llm-inference/大模型推理张量并行]]；集合通信原语见 [[ai-infra/网络/集合通信原语]]；框架实现见 [[ai-framework/megatron-lm/README]]、[[ai-framework/deepspeed/README]]；MoE 见 [[llm-algo/moe/README]]。

---

## 9. 加速三板斧（各打不同瓶颈）

| 技术 | 打哪个瓶颈 | 一句话原理 |
|------|-----------|-----------|
| **量化** INT8/INT4/FP8 | 显存 + 带宽 | 权重位宽减半→搬运字节减半，Decode 直接提速 |
| **FlashAttention** | 带宽 + 显存 | 注意力分块融合，不写出 $n^2$ 大矩阵，IO 降一个量级 |
| **投机采样** Speculative | 串行性 | 小模型先猜 $k$ 个，大模型一次并行验证 |

**9.1 量化**：FP16→INT4，权重显存 ÷4，每步搬运字节 ÷4 → Decode（带宽受限）几乎线性加速。
> 📎 [[llm-compression/quantization/量化基础]]、[[llm-compression/quantization/fp8]]。

**9.2 FlashAttention**：把 softmax 注意力做**分块 + online softmax**，避免把 $n\times n$ 注意力矩阵落回显存，HBM 访存大幅下降，长序列尤甚。
> 📎 [[llm-optimizer/FlashAttention]]。

**9.3 投机采样（Speculative Decoding）**：用一个便宜的 draft 模型连猜 $k$ 个 token，再让大模型**一次前向并行验证**，接受的全要、第一个被拒处截断。把「$k$ 次串行大模型前向」换成「1 次并行验证」。

```
普通解码:  大█→大█→大█→大█   (4 步串行，4 次大模型前向)
投机解码:  小·小·小·小 (draft 猜 4 个)
           大████ (1 次并行验证) → 接受 3 个 + 重采 1 个
           平均加速 ≈ 接受率 × draft 长度
```

Medusa 则给大模型加**多个解码头**一次预测多 token，免去独立 draft 模型。
> 📎 投机采样仓库 LLMSpeculativeSampling；Medusa（Multiple Decoding Heads）。

---

## 10. 服务框架全景

| 框架 | 出处 | 看点 |
|------|------|------|
| **vLLM** | UC Berkeley | PagedAttention + Continuous Batching，吞吐标杆 |
| **TGI** | HuggingFace | Text Generation Inference，生产成熟 |
| **TensorRT-LLM** | NVIDIA | 编译优化 + 自定义 kernel，N 卡极致延迟 |
| **SGLang** | | RadixAttention（前缀缓存复用），多轮/Agent 友好 |
| **DeepSpeed-Inference** | Microsoft | 推理张量并行 + kernel 注入 |
| **TRITON Inference Server** | NVIDIA | 通用推理服务编排（多框架后端） |
| **OpenLLM** | BentoML | 部署/打包封装 |

> 📎 详见 [[llm-inference/LLM服务框架对比]]、[[llm-inference/llm推理框架]]、[[llm-inference/llm推理优化技术]]、[[llm-inference/vllm]]、[[llm-inference/DeepSpeed-Inference]]。框架版本/默认参数请「以官方为准」，不同版本默认值差异较大。

---

## 数值手算：端到端估算一道题

**设定**：模型 70B（700 亿）参数，类 LLaMA 结构，$L=80$ 层，$d_{model}=8192$，FP16 推理，序列长度 $n=2048$，batch=16，MHA（KV 头维度 $=d_{model}$）。

**(1) 权重显存**：

$$70\times10^9 \times 2\,\text{B} = 140\,\text{GB}$$

→ 单卡 80GB H100 放不下，至少 **TP=2**（每卡 70GB 权重）。

**(2) KV-Cache 显存**（公式见第 3 节）：

$$2 \times L \times n \times d_{model} \times \text{batch} \times 2\,\text{B}$$
$$= 2 \times 80 \times 2048 \times 8192 \times 16 \times 2$$

逐步：$2\times80=160$；$160\times2048=327{,}680$；$\times8192\approx2.684\times10^9$；$\times16\approx4.295\times10^{10}$；$\times2\,\text{B}\approx8.59\times10^{10}\,\text{B}\approx\mathbf{80\,GB}$。

**结论**：仅 KV-Cache 就要 80GB，和权重一个量级！这就是为什么高并发长上下文必须上 **GQA / KV 量化 / PagedAttention**。若换 GQA 把 KV 头数从 64 降到 8（÷8），KV-Cache → **10GB**，立竿见影。

**(3) Decode 单步是否带宽受限**（算术强度估算）：

batch=16 时，一次权重读取（140GB / TP后每卡 70GB）服务 16 个 token。
- 访存 ≈ 每卡 70GB 权重 + 该卡 KV
- 计算 ≈ $2\times N_{params}\times \text{batch}$ FLOPs（每参数 2 FLOP）$=2\times70\times10^9\times16\approx2.24\times10^{12}$ FLOPs

算术强度 $\approx \dfrac{2.24\times10^{12}}{70\times10^9\,\text{B}}\approx 32$ FLOP/B。H100 转折点约 $\dfrac{1000\,\text{TFLOPS}}{3.35\,\text{TB/s}}\approx 300$ FLOP/B。
$32 \ll 300$ → **仍深度带宽受限**。把 batch 加到 ~150 才接近转折点——印证第 5、7 节：**Decode 提吞吐靠加大 batch**。

> 📎 FLOPs/带宽更细的推导见 [[llm-algo/FLOPs]] 与 [[ai-infra/算力/GPU工作原理]]。

---

## 常见问题

| 问题 | 答案 |
|------|------|
| 为什么 Decode 比 Prefill 慢这么多？ | Decode 每步只算 1 token 却要搬全部权重，算术强度极低、带宽受限；Prefill 并行处理整段，算力受限。 |
| KV-Cache 一定省时间吗？ | 省计算（$O(n^2)\to O(n)$），但吃显存（∝ batch×seq）。它是「时间换空间」的反向——空间换时间。 |
| 加大 batch 总是好的？ | 对吞吐好（提算术强度），但单请求延迟 TPOT 上升，且 KV-Cache 显存随 batch 线性涨，受显存上限约束。 |
| TP 和 PP 怎么选？ | TP 降延迟但通信频繁需 NVLink（同机内）；PP 通信少可跨机但有气泡。常 TP×PP 混合。 |
| 量化会掉精度吗？ | INT8/FP8 通常近乎无损；INT4 需 GPTQ/AWQ 等校准技术控制误差，权衡显存与质量。 |
| 投机采样什么时候不划算？ | draft 接受率低（领域偏差大）时，验证开销 > 收益；接受率高时加速明显。 |
| 长上下文显存爆怎么办？ | GQA/MQA 减 KV 头、KV 量化（INT8/FP8）、PagedAttention 消碎片、滑窗注意力。 |

---

## 🔗 跳转链接

- 顶层导航：[[00-知识地图]]
- 本节兄弟篇：[[llm-inference/KV-Cache优化]]　[[llm-inference/解码策略]]　[[llm-inference/大模型推理张量并行]]　[[llm-inference/LLM服务框架对比]]　[[llm-inference/llm推理框架]]　[[llm-inference/llm推理优化技术]]　[[llm-inference/vllm]]　[[llm-inference/DeepSpeed-Inference]]
- 算法地基：[[llm-algo/transformer/模型架构]]　[[llm-algo/mlp]]　[[llm-algo/moe/README]]　[[llm-algo/旋转编码RoPE]]　[[llm-algo/FLOPs]]
- 优化器件：[[llm-optimizer/FlashAttention]]　[[llm-optimizer/kv-cache]]　[[llm-optimizer/计算通信重叠]]
- 压缩量化：[[llm-compression/quantization/量化基础]]　[[llm-compression/quantization/fp8]]
- 内存估算：[[transformer内存估算]]
- 基础设施：[[ai-infra/网络/集合通信原语]]　[[ai-infra/算力/GPU工作原理]]　[[ai-infra/ai-hardware/CUDA]]
- 框架：[[ai-framework/megatron-lm/README]]　[[ai-framework/deepspeed/README]]
