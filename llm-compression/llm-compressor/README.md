# llm-compressor

> vLLM 官方生态里的**统一模型压缩库**：用一份 `recipe` 把 GPTQ / AWQ / SmoothQuant / FP8 量化与剪枝统一起来，产出 **compressed-tensors** 格式的权重，让 vLLM 能"开箱即载、原生加速"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-compression/llm-compressor/量化方案]] [[llm-compression/llm-compressor/剪枝]] [[llm-inference/vllm/README]]

---

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点：它在压缩流水线里的位置 | producer / consumer |
| 1 | 它解决什么问题（为什么不是 AutoGPTQ/AutoAWQ） | 碎片化、格式割裂 |
| 2 | 整体架构与三大基石（compressor / compressed-tensors / vLLM） | 三件套 |
| 3 | 核心抽象：Modifier + Recipe + Pipeline | 配方驱动 |
| 4 | 支持的算法谱系：W8A8 / W4A16 / FP8 / KV-Cache / 剪枝 | scheme |
| 5 | 一次量化的完整生命周期（数据流 ASCII） | calibrate → compress → save |
| 6 | compressed-tensors 格式：磁盘上到底存了什么 | config + scale/zp |
| 7 | 与 vLLM 对接：加载与推理路径 | quant_method |
| 8 | 关键算法机制（GPTQ/AWQ/SmoothQuant/FP8/剪枝）| Hessian/迁移/缩放 |
| 9 | 配置示例：Recipe 字段逐项讲含义 | targets/ignore/scheme |
| — | 常见问题 + 跳转链接 | 排错 |

---

## 0. 一句话锚点

`llm-compressor` 是 **vLLM 项目（vllm-project）旗下的训练侧/离线侧压缩工具**。它不负责推理，而是**生产压缩好的权重**；真正的推理加速由 vLLM + `compressed-tensors` 完成。记住这条流水线：

```
        [生产者 producer]                         [格式 format]                 [消费者 consumer]
   ┌────────────────────────┐            ┌───────────────────────┐        ┌──────────────────┐
   │     llm-compressor     │  导出 →    │   compressed-tensors  │  加载→ │       vLLM        │
   │  (recipe 驱动量化/剪枝) │            │  (磁盘权重 + 量化元数据)│        │  (原生 kernel 推理)│
   └────────────────────────┘            └───────────────────────┘        └──────────────────┘
        离线 / 一次性                          标准化中间产物                     线上 / 高并发
```

> 一句话：**llm-compressor 写，compressed-tensors 存，vLLM 读。** 三者各司其职，避免"一个库一种格式"的混乱。

---

## 1. 地基：它解决什么问题

在它出现之前，开源量化生态是**碎片化**的，每个算法一个独立项目、一种私有权重格式：

| 算法 | 典型旧工具 | 产出格式 | 痛点 |
|------|-----------|---------|------|
| GPTQ | AutoGPTQ | GPTQ-specific | 格式各异，vLLM 要逐个适配 |
| AWQ | AutoAWQ | AWQ-specific | scale/zero-point 布局不统一 |
| SmoothQuant | 论文脚本 | 无标准实现 | 难落地 |
| FP8 | 各家自研 | 各不相同 | 没有统一 schema |

**碎片化带来三个痛点**：①**格式割裂**——每种量化的 `scale`/`zero-point`/打包方式都不同，引擎要逐格式写加载逻辑；②**不可组合**——"先 SmoothQuant、再 GPTQ、再 2:4 剪枝"要手工拼脚本；③**与推理脱节**——量化出的模型不保证 vLLM 跑得快，常"能存不能快"。

`llm-compressor` 的设计目标就是**统一**：

- **统一入口**：一个 `oneshot()`（离线一次性）/ 训练循环接口，覆盖所有算法。
- **统一配方**：用声明式 `recipe` 描述"对哪些层、用什么 scheme、按什么顺序"做压缩。
- **统一格式**：所有结果都落到 `compressed-tensors`，vLLM 只需认这一种格式。

```
   碎片化世界                                   llm-compressor 世界
   AutoGPTQ  AutoAWQ  SmoothQuant脚本  ──统一──▶  llm-compressor（GPTQ/AWQ/SQ/FP8/剪枝一锅端）
      ▼         ▼          ▼                                  ▼
   格式A      格式B      格式C                       compressed-tensors（唯一格式）
```

---

## 2. 整体架构与三大基石

llm-compressor 不是单一仓库在打天下，而是**三件套协作**：

```
┌──────────────────────────────────────────────────────────────────────┐
│                          三件套（生态分工）                              │
│                                                                        │
│  ① llm-compressor      —— 算法与流程层                                  │
│     · Modifier（每个压缩算法是一个 Modifier）                            │
│     · Recipe（把多个 Modifier 编排成一条流水线）                         │
│     · Pipeline（校准数据如何流过模型：sequential / basic 等）            │
│                                                                        │
│  ② compressed-tensors  —— 格式与运行时表示层                            │
│     · QuantizationScheme / QuantizationArgs（量化方案数据结构）          │
│     · 权重压缩/解压（packed int4、float8 等）                            │
│     · 与 safetensors 兼容的磁盘布局 + config                            │
│                                                                        │
│  ③ vLLM                —— 推理消费层                                    │
│     · 识别 config 里的 quant_method=compressed-tensors                  │
│     · 挑选对应 kernel（Marlin / Machete / cutlass-fp8 等）              │
└──────────────────────────────────────────────────────────────────────┘
```

- 仓库地址（上游）：`vllm-project/llm-compressor`、`vllm-project/compressed-tensors`。
- **关键认知**：`compressed-tensors` 既是"磁盘格式"，也是"内存中量化方案的数据结构定义"。`llm-compressor` 在运行时**复用**它的 `QuantizationScheme` 等类型，所以二者的概念是天然对齐的。
- 量化类型的"真值表"在 compressed-tensors 的 `quant_scheme.py` 里定义（W8A8-int8、W8A8-fp8、W4A16 等预设方案皆出于此）。

> 设计哲学：**算法的多样性收敛到一种格式**。任何新算法只要最终能写成 compressed-tensors 的 scheme，vLLM 就无需改动即可消费。

---

## 3. 核心抽象：Modifier + Recipe + Pipeline

这是理解整个库的钥匙。三个概念层层嵌套：

```
Recipe（一张配方）
  └── 一个有序的 Modifier 列表
        ├── SmoothQuantModifier   （先平滑激活）
        ├── GPTQModifier          （再量化权重）
        └── （可选）剪枝 Modifier  （再稀疏化）
              │
              ▼  在 Pipeline 控制下，用校准数据逐步执行
        Pipeline（sequential / basic ...）决定"数据怎么流过模型、何时触发每个 modifier"
```

### 3.1 Modifier——一个压缩算法的封装

每个算法（GPTQ、AWQ、SmoothQuant、剪枝……）被封装成一个 **Modifier 对象**。它定义了：

- **作用对象**：`targets`（对哪些层生效，如 `Linear`）、`ignore`（哪些层跳过，如 `lm_head`）。
- **方案**：`scheme`（如 `"W4A16"`、`"FP8"`）或显式的位宽/对称性/分组等参数。
- **生命周期钩子**：初始化 → 校准时收集统计量 → 结束时把量化参数固化进权重。

直觉：Modifier 就像"对模型施加的一道工序"。

### 3.2 Recipe——把工序编排成流水线

`recipe` 是**声明式配方**，可以是 Python 对象列表，也可以是一段 YAML。它表达：

> "先做 A，再做 B，C 这些层不要动。"

```
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),   # 工序1：迁移激活异常值到权重
    GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]),  # 工序2：4bit 权重量化
]
```

顺序很重要：SmoothQuant 必须在 GPTQ **之前**，因为它改变了权重分布，让后续量化更易处理。

### 3.3 Pipeline——校准数据如何流过模型

量化（尤其 GPTQ）需要**校准数据**来估计统计量。Pipeline 决定数据流策略：

```
basic pipeline：整模型一次前向，简单但显存峰值高
        prompt → [整个模型一次 forward] → 收集所有层统计 → 量化

sequential pipeline：逐层/逐块前向，显存友好（大模型常用）
        prompt → [layer0 forward → 量化 layer0 → 释放]
                  → [layer1 forward(用已量化的layer0输出) → 量化 layer1] → ...
```

`sequential` 的精髓：**量化第 N 层时，用的是前面已量化层的真实输出**，从而让误差不会层层放大——这与 GPTQ 的逐层补偿思想一致。

---

## 4. 支持的算法谱系（scheme 速查）

量化方案常用 **`W{权重位宽}A{激活位宽}`** 命名法：

| Scheme | 权重 | 激活 | 算法搭配 | 典型场景 |
|--------|------|------|---------|---------|
| `W8A8-INT8` | int8 | int8（动/静态） | SmoothQuant + GPTQ/RTN | 吞吐优先、有 INT8 Tensor Core |
| `W8A8-FP8` | fp8 | fp8（动/静态） | FP8（RTN 风格） | Hopper/Ada，精度损失极小 |
| `W4A16` | int4 | fp16 | GPTQ / AWQ | 显存/权重带宽瓶颈、单卡塞大模型 |
| `W4A8` | int4 | int8 | 组合 | 极致压缩（较新、需验证精度） |
| `FP8 dynamic` | fp8 | fp8 动态 | 无需校准 | 最省事的高精度方案 |
| KV-Cache 量化 | — | — | 对 KV 做 fp8/int8 | 长上下文省显存 |
| 剪枝（2:4 / 非结构化） | — | — | SparseGPT / 幅度剪枝 | 配合稀疏 kernel 提速 |

> **A16 vs A8 的本质权衡**：
> - `W4A16`：只压权重，激活仍 fp16 → 精度好、解码阶段（memory-bound）受益大；但矩阵乘要先反量化，算力密集场景（prefill）收益有限。
> - `W8A8`：权重+激活都低精度 → 能直接用 INT8/FP8 Tensor Core 算，**prefill/大 batch 吞吐**显著提升，但激活量化更易掉点。

激活量化的**动态 vs 静态**：动态(dynamic) 推理时按当前 batch 实时算 scale → 精度高、无需校准、略增开销；静态(static) 离线用校准集预定 scale → 推理零开销，但分布漂移时易掉点。

---

## 5. 一次量化的完整生命周期（数据流）

以"用 GPTQ 做 W4A16"为例，端到端走一遍：

```
 STEP0 准备：加载 fp16 原模型 + 校准集（如 512 条样本，seq 2048）
   │
 STEP1 oneshot(model,recipe,dataset)：解析 recipe → 实例化 Modifier
   │
 STEP2 CALIBRATE 校准：Pipeline 喂样本，各 Modifier 注册 hook 收集统计
        · GPTQ 累积 Hessian H=Σxxᵀ ；SmoothQuant 统计激活幅值
   │
 STEP3 COMPRESS 量化：逐层求 scale/zero-point；GPTQ 误差补偿；权重打包 int4
   │
 STEP4 SAVE 导出：save_pretrained(..., save_compressed=True)
        · 权重写成 compressed-tensors(packed) + config 记录 scheme
   ▼
 产物：safetensors（压缩权重） + config.json（含 quantization_config）
```

调用链（伪代码骨架，**API 以官方文档/源码为准**）：

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
oneshot(model="meta-llama/...", dataset="open_platypus", recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512,
        output_dir="./my-model-W4A16")
```

> `oneshot` = **一次性离线压缩**（无梯度更新）。若要"量化感知训练 / 蒸馏恢复精度"，库还提供 `train` 风格接口走训练循环；大多数 PTQ 场景 `oneshot` 足矣。

---

## 6. compressed-tensors 格式：磁盘上存了什么

导出后产物目录里，**两类信息**缺一不可：

```
my-model-W4A16/
├── config.json                 ← 含 "quantization_config" 块
│      {
│        "quant_method": "compressed-tensors",
│        "format": "pack-quantized",      # int4 打包方式
│        "config_groups": { "group_0": {  # 每组的量化方案
│             "weights": {"num_bits":4,"type":"int","symmetric":true,
│                         "strategy":"group","group_size":128},
│             "targets": ["Linear"] } },
│        "ignore": ["lm_head"]
│      }
├── model-00001-of-0000X.safetensors  ← 压缩后的张量
│      对每个被量化层，磁盘上保存：
│        weight_packed   （int4 多个值打包进一个 int32）
│        weight_scale    （反量化用的缩放因子，每 group 一个）
│        weight_zero_point（非对称才有）
│        weight_shape    （还原形状用）
└── tokenizer 等其他文件
```

**反量化公式**（int 量化的核心）：对原始权重 $w$，存的是

$$ q = \mathrm{round}\!\left(\frac{w}{s}\right) + z, \qquad \hat{w} = s\,(q - z) $$

其中 $s$ 是 scale、$z$ 是 zero-point。**对称量化** $z=0$，存得更省、kernel 更简单。

**数值例子**（per-group, group_size=128, 4-bit 对称）：某 group 内权重最大绝对值 $0.92$，4-bit 对称范围 $[-7,7]$，则

$$ s = \frac{0.92}{7} \approx 0.131 $$

某权重 $w=0.40$ → $q=\mathrm{round}(0.40/0.131)=3$ → 还原 $\hat w = 0.131\times 3 = 0.393$，误差 $\approx 0.007$。group_size 越小，$s$ 越贴合局部 → 误差越小，但 scale 数量越多（元数据开销越大）。这就是 **group_size 的精度 vs 体积权衡**。

```
strategy 取值的粒度（精度↑ / 元数据↑ 从左到右）：
   tensor（整张一个 scale） < channel（每输出通道一个） < group（每 group_size 一个）
```

---

## 7. 与 vLLM 对接：加载与推理路径

vLLM 加载模型时**读 config**，看到 `quant_method: compressed-tensors` 就走压缩路径：

```
vLLM 启动加载
   │
   ▼
读 config.json → quantization_config.quant_method == "compressed-tensors"
   │
   ▼
按 format / config_groups 解析每层 scheme（W4A16 / W8A8 / FP8 …）
   │
   ▼
为每种 scheme 选择最优 kernel：
   ├── W4A16 int   → Marlin / Machete（解包 int4 + GEMM 融合）
   ├── W8A8 int8   → cutlass int8 GEMM
   ├── W8A8 fp8    → cutlass fp8 GEMM（Hopper/Ada）
   └── 稀疏 2:4    → 稀疏 GEMM kernel
   │
   ▼
推理时：权重以压缩态驻留显存（省带宽/省显存）→ kernel 内即时反量化/直接低精度算
```

使用上几乎透明：

```python
from vllm import LLM
llm = LLM(model="./my-model-W4A16")   # 无需额外指定量化类型，vLLM 自动识别
```

> **为什么要原生格式而非"加载时再转换"**：压缩态权重**直接驻留显存**，省的是宝贵的显存与带宽；kernel 在计算时才反量化（W4A16）或直接低精度算（W8A8），避免了"先全部还原成 fp16 再算"的浪费。

---

## 8. 关键算法机制（讲思路，不抠实现）

### 8.1 GPTQ —— 逐层最优误差补偿

目标：量化每层权重时**最小化该层输出误差**，而非简单四舍五入。它用校准数据的 Hessian $H = X X^\top$ 决定量化顺序与补偿量：量化一列产生误差 $e$ → 按 Hessian 把 $e$ 分摊修正到"尚未量化"的其他列 → 整体输出误差最小。直觉：先量化的列让后量化的列"主动纠偏"。比逐元素 RTN（round-to-nearest）精度高得多，代价是要算 Hessian、较慢。

### 8.2 AWQ —— 激活感知的权重保护

观察：**不是所有权重同等重要**，对应"大激活通道"的权重最敏感。AWQ 给每个通道找缩放 $s$：

$$ y = (W \cdot \mathrm{diag}(s))\,(\mathrm{diag}(s)^{-1} x) $$

把重要通道的权重**放大后再量化**（等于给它更高有效精度），两边乘逆缩放保持数学等价；$s$ 由激活幅值搜索得到。结果：W4 也能保住关键信息。

### 8.3 SmoothQuant —— 把激活的"刺"挪给权重

激活里常有**离群值（outlier）**，让激活量化掉点严重，而权重通常很平滑。SmoothQuant 用一个平滑因子把难度从激活"迁移"到权重：

```
   激活分布（尖刺多，难量化）          权重分布（平滑，好量化）
        │  ╿                                  │
        │  ║ ← outlier        ──迁移 s──▶      │ ╱╲  （吸收一点难度后仍可控）
        │__╨__________                         │╱  ╲___________
```

$$ \hat X = X\,\mathrm{diag}(s)^{-1}, \qquad \hat W = \mathrm{diag}(s)\,W,\qquad \hat X\hat W = XW $$

`smoothing_strength`（常记 $\alpha$）控制迁移多少：$\alpha$ 大 → 激活更易量化但权重更难，需折中。所以它**常作为 W8A8-int8 的前置工序**。

### 8.4 FP8 —— 浮点低精度，几乎白送

FP8（E4M3/E5M2）保留指数位 → **动态范围大**、对离群值天然友好、掉点极小，很多时候 **动态 FP8 无需校准** 即可用；代价是需 Hopper/Ada 等支持 FP8 的硬件才有算力红利。

### 8.5 剪枝 —— 稀疏化（结构化 2:4 / 非结构化）

```
2:4 结构化稀疏：每连续 4 个权重里保留 2 个、置零 2 个
   [ w0 w1 w2 w3 ]  →  [ w0  0  w2  0 ]
   恰好契合 Ampere+ 的稀疏 Tensor Core（理论 2x）
```

- **非结构化剪枝**（SparseGPT/幅度）：剪掉最小幅值权重，灵活但硬件难加速；**2:4 半结构化** 精度与加速兼顾，是实践主力。剪枝可与量化**叠加**（先剪后量），由 recipe 编排顺序。

> 关键认知：**剪枝省"计算/稀疏 kernel"，量化省"存储/带宽"**，二者正交、可组合。

---

## 9. 配置示例：Recipe 字段逐项讲含义

下面是一段"SmoothQuant + GPTQ 做 W8A8-int8"的概念性 recipe（**字段名以官方文档/源码为准**，此处讲含义）：

```yaml
# 工序1：先平滑激活离群值
- SmoothQuantModifier:
    smoothing_strength: 0.8        # α：迁移强度，0~1，越大激活越好量化、权重越难

# 工序2：再做 W8A8 int8 量化
- GPTQModifier:
    scheme: "W8A8"                 # 权重int8 + 激活int8
    targets: ["Linear"]            # 只量化线性层（注意力/MLP 投影）
    ignore: ["lm_head"]            # 输出头不量化，保护最终 logits 精度
    dampening_frac: 0.01           # Hessian 阻尼，提升数值稳定（避免奇异）
```

逐项含义与权衡：

| 字段 | 含义 | 调大 / 调小的影响 |
|------|------|------------------|
| `scheme` | 量化方案（W8A8/W4A16/FP8…） | 决定精度-体积-速度三角的落点 |
| `targets` | 作用层类型/名称 | 只动 Linear 最常见；动太多易掉点 |
| `ignore` | 排除层（如 `lm_head`、`embed`） | 不忽略输出头通常明显掉点 |
| `group_size` | 量化分组粒度（如 128） | 小→精度高但 scale 多；大→省体积但糙 |
| `symmetric` | 是否对称（z=0） | 对称省存储、kernel 简单；非对称更贴合偏置分布 |
| `dampening_frac` | GPTQ Hessian 阻尼 | 太小数值不稳；太大补偿失真 |
| `smoothing_strength` | SmoothQuant 迁移强度 | 见 8.3，需折中 |
| `num_calibration_samples` | 校准样本数 | 太少统计不稳；太多耗时，几百条常够 |
| `max_seq_length` | 校准序列长度 | 影响激活统计的代表性 |

**校准集要点**：用**贴近目标分布**的数据（指令模型用指令数据），样本几百即可；FP8 动态量化通常**不需要**校准。

---

## 常见问题

| 问题 | 诊断 / 解法 |
|------|------------|
| 量化后精度大幅下降 | 优先 `ignore: ["lm_head"]`；W4 改 group_size=128；INT8 加 SmoothQuant；考虑换 FP8 |
| OOM（量化大模型时显存爆） | 用 **sequential pipeline** 逐层量化；减少校准样本/序列长度 |
| vLLM 加载报"未知量化类型" | 确认 config 里 `quant_method: compressed-tensors`、vLLM 版本支持该 scheme/format |
| W4A16 在 prefill/大batch 不提速 | 正常：A16 只省权重带宽，算力密集段收益小 → 想吞吐用 W8A8/FP8 |
| FP8 跑不出加速 | 检查硬件是否支持 FP8（Hopper/Ada）；老卡无 FP8 算力 |
| 剪枝后没变快 | 非结构化稀疏多数 kernel 不加速；改 **2:4 半结构化** + 稀疏 kernel |
| GPTQ 报数值不稳/NaN | 适当增大 `dampening_frac`；检查校准数据是否含异常样本 |
| scale 文件太大 | group_size 调大或用 channel/tensor 粒度，权衡精度 |
| 该选哪种 scheme？ | 显存瓶颈→W4A16；吞吐瓶颈→W8A8-int8/FP8；要省心高精度→FP8 dynamic |
| oneshot 与 train 怎么选 | 纯 PTQ 用 `oneshot`；需恢复精度/QAT 走训练接口 |

---

## 🔗 跳转链接

- [[00-知识地图]] —— 压缩专题总图，回到全局
- [[llm-compression/llm-compressor/量化方案]] —— GPTQ / AWQ / SmoothQuant / FP8 各 scheme 细节
- [[llm-compression/llm-compressor/剪枝]] —— SparseGPT / 2:4 稀疏 / 幅度剪枝
- [[llm-inference/vllm/README]] —— 下游消费者：vLLM 如何加载 compressed-tensors 并选 kernel

**官方参考（以最新文档/源码为准）**：量化方案真值表 `compressed-tensors/.../quantization/quant_scheme.py`；示例 `llm-compressor/examples/quantization_w8a8_int8` 与 `.../quantization_w8a8_fp8`。
