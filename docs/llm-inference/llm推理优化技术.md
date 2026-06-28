# LLM 推理优化技术全景

> 一句话定位：LLM 推理优化就是在「显存墙 + 访存墙 + 调度浪费」三座大山下，用量化 / 并行 / 调度 / 投机 / 编译 / 显存管理六类手段，把首 Token 延迟、平均 Token 延迟压下去，把吞吐量顶上去。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/README]] · [[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[llm-inference/vllm/README]] · [[llm-compression/quantization/量化基础]]

## 阅读地图

| 节 | 内容 | 你会得到 |
|----|------|----------|
| 0 | 一句话锚点 | 推理优化在优化什么 |
| 1 | 地基：自回归推理的两阶段与瓶颈 | Prefill / Decode、为什么访存受限 |
| 2 | 指标：延迟 vs 吞吐的权衡 | TTFT、TPOT、吞吐量定义 |
| 3 | 模型量化 | W4A16/W8A8、KV Cache 量化 |
| 4 | 分布式并行推理 | TP 降延迟、PP 提吞吐 |
| 5 | 服务化调度优化 | 动态 batch vs continuous batching |
| 6 | 投机采样 | 小模型起草 + 大模型校验 |
| 7 | 模型编译优化 | 前端 vs 后端优化 |
| 8 | 显存优化 | PagedAttention、CPU Offloading |
| 9 | 低精度浮点 | FP8 / FP16 / BF16 |
| 10 | FlashAttention vs PagedAttention | 两者目标与可组合性 |
| 实操 | 框架对照表 | 每种技术落在哪个框架 |
| 坑 | 常见问题 | 踩坑速查 |

## 0. 一句话锚点

推理优化的全部动作，都是在做同一道权衡题：

$$\text{优化目标} = \underbrace{\text{低延迟}}_{\text{TTFT, TPOT}} \;\;\text{vs}\;\; \underbrace{\text{高吞吐量}}_{\text{tokens/s}}$$

二者天然冲突：增大 batch 提吞吐会拉高单请求延迟；张量并行降延迟却引入通信开销降吞吐。关键词只有四个：**首 Token 延迟、平均 Token 延迟**（原文锚点）。所有技术都在这条权衡曲线上挪动你的工作点。

## 1. 地基：自回归推理的两阶段与瓶颈

一次 LLM 推理被切成两个性质完全不同的阶段：

```
  请求 "解释一下注意力机制"
        │
        ▼
 ┌─────────────────┐   一次前向，所有输入 token 并行算
 │  Prefill 预填充  │   → 算力受限 (compute-bound)
 │  (处理 prompt)   │   → 产出第 1 个 token + 写满 KV Cache
 └────────┬────────┘   → 决定 TTFT(首 Token 延迟)
          ▼
 ┌─────────────────┐   每步只算 1 个新 token
 │  Decode 解码    │   → 访存受限 (memory-bound)
 │  (逐 token 生成) │   → 每步都要读全部 KV Cache + 权重
 └────────┬────────┘   → 决定 TPOT(平均 Token 延迟)
          ▼
   "注意力机制是……"(逐字吐出)
```

**为什么 Decode 是访存受限？** 每生成 1 个 token，要把整个模型权重 + 整个 KV Cache 从 HBM 搬到 SRM/寄存器，但只做 batch=1 的极少计算。算术强度（FLOPs/Byte）极低，GPU 算力大量空转，瓶颈在显存带宽。

→ 这条性质直接决定了优化方向：**降访存量**（量化、KV Cache 压缩、FlashAttention）和**提算术强度**（batching 把多个请求拼成大矩阵乘）是 Decode 阶段的两大杠杆。详见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]、[[docs/transformer内存估算]]。

## 2. 指标：延迟与吞吐的权衡（原文锚点）

| 指标 | 英文 | 含义 | 谁敏感 |
|------|------|------|--------|
| 首 Token 延迟 | TTFT (Time To First Token) | 请求到第 1 个 token 的时间，由 Prefill 决定 | 对话、流式 UI |
| 平均 Token 延迟 | TPOT (Time Per Output Token) | 相邻两 token 间隔，由 Decode 决定 | 长文本生成 |
| 吞吐量 | Throughput | 单位时间总 tokens/s | 批量离线任务 |

权衡可视化（batch size 是核心旋钮）：

```
吞吐量 ▲
       │            ╭──────  大 batch：吞吐高、单请求延迟高
       │         ╱
       │      ╱
       │   ╱     小 batch：延迟低、吞吐低
       │ ╱
       └──────────────────▶ 单请求延迟
       「低延迟、高吞吐量（权衡）」—— 原文核心命题
```

## 3. 模型量化

把权重/激活从 FP16 降到更低比特，**目标是降访存量 + 降显存**，直接缓解 Decode 的访存瓶颈。

| 方案 | 精度记号 | 含义 | 参考框架 |
|------|----------|------|----------|
| AutoAWQ | W4A16 | 权重 4-bit，激活 16-bit | TensorRT-LLM |
| SmoothQuant | W8A8 | 权重 8-bit，激活 8-bit | TensorRT-LLM |
| GPTQ | W4 | 误差补偿的逐层权重量化 | HF transformers |
| bitsandbytes | LLM.int8() | 离群值走 FP16，其余 INT8 | HF transformers |
| KV Cache 量化 | — | 把 KV Cache 也量化 | 降显存→增大 batch |

记号速读：`WxAy` = Weight x-bit / Activation y-bit。

```
权重大小:  FP16 ████████████████ 16 bit
           W8   ████████          8 bit  (省 1/2)
           W4   ████              4 bit  (省 3/4)
```

**为什么 KV Cache 量化能提吞吐？**（原文括注）KV Cache 显存占用 ≈ `2 × layers × heads × head_dim × seq_len × batch × dtype_bytes`。把 dtype 从 FP16(2B) 降到 INT8(1B)，省一半显存 → 同样显存能塞更大 batch → Decode 阶段算术强度上升 → 吞吐量上升。数值示例：13B 模型、2048 上下文，KV Cache 从 FP16 的约 1.6 GB/请求 砍到 INT8 的约 0.8 GB/请求，单卡可并发请求数近似翻倍。

→ 深入：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/README]]

## 4. 分布式并行推理

| 并行方式 | 主要收益 | 代价 |
|----------|----------|------|
| 张量并行 (TP) | **降低延迟**（一层切到多卡并行算） | 每层一次 AllReduce 通信 |
| 流水线并行 (PP) | **提高吞吐量**（不同层分到不同卡，流水填充） | 有 bubble，单请求延迟反升 |

```
张量并行 TP（同一层横切，降延迟）        流水线并行 PP（按层纵切，提吞吐）
 ┌──────── Layer i ────────┐           GPU0: Layer 0-7
 GPU0 ┃ GPU1 ┃ GPU2 ┃ GPU3              GPU1: Layer 8-15
   └──── AllReduce 合并 ────┘            GPU2: Layer 16-23   ← 请求像流水线
   一层被 4 卡同时算完                    多请求重叠填充 bubble
```

为什么 TP 降延迟而 PP 提吞吐？TP 把单层矩阵乘切给多卡并行，单请求 wall-clock 变短 → 延迟降；但每层都要 AllReduce，强依赖高速互联。PP 把模型按层摆到多卡，靠多请求像流水线一样重叠各 stage 来摊薄 bubble → 总吞吐升，但单请求要穿过所有 stage，延迟不降反升。

TP 的通信对网络极敏感 → 见 [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]。框架实现见 [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]。

## 5. 模型服务化调度优化

| 方案 | 参考 | 粒度 | 问题 / 改进 |
|------|------|------|-------------|
| 动态 batch (Dynamic Batching) | Triton Inference Server | 请求级 | 等齐一批再发，短请求被长请求拖死，GPU 空转 |
| 连续批处理 (Continuous Batching) | vLLM | token 级（迭代级） | 一个请求生成完立刻让位，新请求即时插入 |

```
动态 batch（请求级，等最长那个）        continuous batching（迭代级，谁完谁走）
 t: R1 ████████░  (早完，干等)            t: R1 ████ ✔ R5 插入 ████
    R2 █████████                            R2 █████████
    R3 ██░       (早完，干等)               R3 ██ ✔ R6 插入 ███████
    └ 整批等 R2 完才释放 → GPU 空转          └ 槽位即时复用 → 利用率拉满
```

为什么 continuous batching 是 vLLM 等现代引擎的标配：自回归生成长度天然参差不齐，请求级 batch 必然有人早完干等。把调度粒度降到「每生成一步」，空出的槽立刻被等待队列里的新请求填上，GPU 利用率从此不被最长序列绑架。

→ [[llm-inference/vllm/README]] · [[llm-inference/PD分离]]（把 Prefill 和 Decode 拆到不同实例，进一步隔离两阶段的资源争抢）。

## 6. 投机采样（Speculative Decoding）

原文要点（保真保留）：

> - 就是使用一个小模型来做草稿，然后使用大模型做纠正检查。参考：FlexFlow Server。
> - 小模型的参数量要远小于原模型参数量一个级别才效果明显。
> - 小模型和原模型的 tokenizer 最好一模一样，不然会增加额外的解码、编码时间。

```
草稿模型 (小, 快)  ──▶  连猜 k 个 token: t1 t2 t3 t4
                              │
目标模型 (大, 准)  ──▶  一次前向并行校验这 k 个
                              │
      命中 t1 t2 t3 ✔，t4 ✘  ──▶  接受 3 个 + 重采 1 个
      → 一次大模型前向产出多个 token，TPOT 下降
```

**为什么能加速？** Decode 是访存受限，大模型「跑一次前向」和「校验 k 个候选」的耗时几乎一样（都被权重搬运主导）。小模型连猜 k 个几乎免费，大模型一次性并行验证，命中部分白赚 → 等效一次前向吐多 token。命中率越高、草稿越准、tokenizer 越一致（省去转码），收益越大。

→ [[llm-inference/解码策略]] · [[llm-inference/flexflow/投机采样]]

## 7. 模型编译优化（前端 + 后端）

原文定义（保真保留）：

> **前端优化**：输入计算图，关注计算图整体拓扑结构，而不关心算子的具体实现。对算子节点进行融合、消除、化简等操作，使计算图的计算和存储开销最小。
> **后端优化**：关注算子节点的内部具体实现，针对具体实现使得性能达到最优。重点关心节点的输入、输出、内存循环方式和计算逻辑。

| 层次 | 优化手段（原文清单） | 关注点 |
|------|----------------------|--------|
| AI 编译前端 | 图算融合、内存分配、常量折叠、公共子表达式消除、死代码消除、代数化简 | 计算图拓扑，不管算子实现 |
| AI 编译后端 | 算子融合、循环优化 | 单算子内部实现，输入输出/内存循环/计算逻辑 |

```
原始图:  Conv → BN → ReLU → Add → ...
   │  前端: 常量折叠 / 公共子表达式消除 / 死代码消除（图级别瘦身）
   ▼
精简图:  ConvBNReLU(融合) → Add → ...
   │  后端: 算子融合 + 循环优化（kernel 级别压榨带宽与缓存）
   ▼
高效 kernel: 一次 launch 完成多算子，减少 HBM 往返
```

为什么有效：前端在「图」层面砍掉冗余节点、合并可融合算子，减少 kernel 启动与中间张量落盘；后端在「kernel」层面优化访存循环顺序、寄存器复用，把单算子跑到带宽/算力上限。FlashAttention 就是后端「算子融合 + 减少 HBM 访问」的典范。

## 8. 显存优化

| 技术 | 参考 | 原理 | 收益 |
|------|------|------|------|
| PagedAttention | vLLM | 用分块内存 + 共享内存管理 KV Cache，按页分配不连续 | 显存碎片≈0，可增大 batch |
| CPU Offloading | — | 张量常驻 CPU 内存，计算时才拷到 GPU | 显存换带宽，能跑超大模型 |

```
传统 KV Cache（连续预分配, 浪费严重）      PagedAttention（分页, 类虚拟内存）
 ┌────────── 按 max_len 预留 ──────────┐    页表 → [块3][块7][块1]...
 │ 实际用██░░░░░░░░░░░░ 大量预留浪费    │    只在写入时按块分配
 └────────────────────────────────────┘    不同请求可共享相同前缀块
        碎片 + 过度预留                      碎片≈0，吞吐显著提升
```

CPU Offloading（原文）：把张量保存在 CPU 内存中，仅在计算时复制到 GPU。这是「用 PCIe 带宽换显存容量」，让有限显存跑下更大的模型/batch，代价是搬运延迟。

→ [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache]]

## 9. 低精度浮点数优化

原文要点（保真保留并展开）：

> **FP8**：NVIDIA H 系列 GPU 开始支持 FP8，兼有 FP16 的稳定性和 INT8 的速度。NVIDIA Transformer Engine 兼容 FP8，主要利用该精度做 **GEMM（通用矩阵乘法）** 计算，同时以 FP16 或 FP32 高精度保持主权重和梯度。MS-AMP 使用 FP8 进行训练。
> **FP16 / BF16**：半精度，训练推理通用基线。

```
精度位宽对照（数值越窄越快、越省，但越易溢出/精度损失）
 FP32 ████████████████████████████████  32 bit  基线精度
 FP16 ████████████████                  16 bit  半精度(动态范围窄)
 BF16 ████████████████                  16 bit  半精度(指数宽,范围大,尾数少)
 FP8  ████████                           8 bit  H100+, GEMM 加速
 INT8 ████████                           8 bit  最快但需量化校准
```

为什么 FP8 是「FP16 稳定性 + INT8 速度」：FP8 仍是浮点（有指数位），动态范围远好于定点 INT8，训练/推理更稳；位宽与 INT8 同为 8-bit，在 H 系列 Tensor Core 上吞吐近似翻倍。Transformer Engine 的混合策略——GEMM 走 FP8、主权重/梯度留 FP16/FP32——是数值稳定的关键。

→ [[llm-compression/quantization/fp8]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/GPU工作原理]]

## 10. FlashAttention vs PagedAttention（原文对比，保真保留）

> **PagedAttention** 应用在推理时，用分块内存和共享内存优化了 KV Cache 的存储。减少了单个序列的显存，从而可以增大 batch size，获得更大吞吐量。总的说，目标是**推理时减少显存、增大吞吐量**。该团队之前有篇类似工作 **FlexGen**，推理时通过对 KV Cache 进行高效 CPU 卸载提升吞吐量。
>
> **FlashAttention** 训练和推理都可用，通过分块计算和 kernel 融合，减少了 HBM 访问次数，实现计算加速，同时减少显存占用。
>
> 理论上，FlashAttention 和 PagedAttention **可以组合使用**。

| 维度 | FlashAttention | PagedAttention |
|------|----------------|----------------|
| 阶段 | 训练 + 推理 | 推理（KV 存储） |
| 手段 | 分块计算 + kernel 融合 | 分块内存 + 共享内存管理 KV Cache |
| 直接目标 | 减少 HBM 访问 → 算得快、省显存 | 减少 KV Cache 显存 → batch 更大 |
| 受益指标 | 计算速度（前后端编译式优化） | 吞吐量（调度/显存式优化） |
| 关系 | **正交，可组合**：算子层加速 + 显存层管理 | 同上 |

```
        ┌─────────────── 一次推理 ───────────────┐
        │  FlashAttention(算 attention 时省 HBM)   │  ← 算子/kernel 层
        │  PagedAttention(存 KV Cache 时省显存)    │  ← 显存/调度 层
        └────── 两者各管一层，叠加生效 ──────────┘
```

→ [[llm-optimizer/FlashAttention]]

## 实操：技术 → 框架对照速查（原文真料汇总）

| 优化类别 | 具体技术 | 推荐框架 / 参考 |
|----------|----------|----------------|
| 模型量化 | AutoAWQ(W4A16)、SmoothQuant(W8A8) | TensorRT-LLM |
| 模型量化 | GPTQ、bitsandbytes(LLM.int8()) | HF transformers |
| 模型量化 | KV Cache 量化 | 降显存→增大 batch |
| 并行推理 | 张量并行（降延迟） | Megatron-LM / DeepSpeed |
| 并行推理 | 流水线并行（提吞吐） | Megatron-LM / DeepSpeed |
| 服务调度 | 动态 batch | Triton Inference Server |
| 服务调度 | continuous batching | vLLM |
| 投机采样 | 小模型起草 + 大模型校验 | FlexFlow Server |
| 编译优化 | 前端：图算融合/常量折叠/CSE/DCE/代数化简 | AI 编译器前端 |
| 编译优化 | 后端：算子融合/循环优化 | AI 编译器后端 |
| 显存优化 | PagedAttention 管理 KV Cache | vLLM |
| 显存优化 | CPU Offloading（张量驻 CPU，按需拷 GPU） | — |
| 低精度 | FP8（H 系列，GEMM 加速） | NVIDIA Transformer Engine / MS-AMP |
| 低精度 | FP16 / BF16 | 通用基线 |

**参考资料（原文链接，保真保留）：**
- Mastering LLM Techniques: Inference Optimization：https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- 7 Ways To Speed Up Inference of Your Hosted LLMs：https://betterprogramming.pub/speed-up-llm-inference-83653aa24c47
- How to Speed Up LLM Training with Distributed Systems?：https://www.appypie.com/blog/llm-training-with-distributed-systems
- 详谈大模型训练和推理优化技术：https://wjn1996.blog.csdn.net/article/details/130764843
- LLM 盛行，如何优雅地训练大模型？：https://cloud.tencent.com/developer/article/2321394?areaId=106001

## 常见问题 / 坑

| 现象 / 疑问 | 原因 | 对策 |
|-------------|------|------|
| 增大 batch 吞吐升但延迟暴涨 | 延迟与吞吐天然权衡 | 在线服务限 batch 上限；用 continuous batching 折中 |
| 投机采样反而变慢 | 草稿模型太大或 tokenizer 不一致 | 草稿参数量需小一个级别；tokenizer 尽量一模一样 |
| KV Cache 量化精度掉太多 | 激活离群值敏感 | 只量化 KV、保留权重精度；评估困惑度回退 |
| TP 多卡未提速反变慢 | AllReduce 通信被慢网络拖累 | 上 NVLink/InfiniBand；卡间互联是 TP 生命线 |
| PP 单请求延迟变高 | 流水线 bubble + 穿过所有 stage | PP 提吞吐不降延迟，在线低延迟场景慎用 |
| FP8 训练发散 | 主权重也用了 FP8 | 仅 GEMM 走 FP8，主权重/梯度留 FP16/FP32 |
| 动态 batch GPU 空转 | 请求级 batch 等齐最长序列 | 换 token 级 continuous batching |
| 显存碎片导致 OOM 但利用率低 | 连续 KV Cache 过度预留 | 用 PagedAttention 分页管理 |
| 显存不够跑大模型 | 模型/KV 超显存容量 | CPU Offloading（牺牲 PCIe 带宽换容量） |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]] · [[llm-inference/README]]
- 架构基础：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 算子/显存：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理引擎：[[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 压缩量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练/对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 硬件/网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
