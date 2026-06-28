# vLLM FP8 量化

> 在 vLLM 里用 8-bit 浮点（FP8）压缩权重/激活，把显存砍掉约一半、靠 Hopper/Ada 张量核拿到约 2 倍吞吐，且精度损失极小。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/vllm/README]] [[llm-compression/quantization/量化基础]] [[llm-compression/quantization/kv-cache-quant]] [[llm-optimizer/kv-cache]]

## 阅读地图

| 小节 | 你将搞懂 | 关键词 |
| --- | --- | --- |
| 0 | 一句话锚点 | FP8 = 8-bit 浮点 |
| 1 | FP8 解决什么问题 | 显存墙 / 带宽墙 / 算力 |
| 2 | FP8 的两种格式 E4M3 vs E5M2 | 指数位 vs 尾数位 |
| 3 | 量化的本质：缩放因子 scale | per-tensor / per-channel |
| 4 | 权重量化 vs 激活量化 | W8A8 / W8A16 / dynamic / static |
| 5 | vLLM 里 FP8 的整体数据流 | 反量化 → 张量核 |
| 6 | 三种使用路径（在线/AutoFP8/已量化权重） | quantization 参数 |
| 7 | KV Cache 也能 FP8 | kv_cache_dtype |
| 配置示例 | 三套可跑的代码 | LLM(...) / AutoFP8 |
| 坑 | 硬件门槛、精度、scale 选择 | 常见问题表 |

---

## 0. 一句话锚点

**FP8 = 用 8 个比特表示一个浮点数**。原来一个权重用 FP16/BF16（16 bit）存，现在用 FP8（8 bit）存，**存储和显存直接减半**；并且新一代 GPU（NVIDIA Hopper H100/H200、Ada L40S/L4，以及部分国产 NPU）的张量核**原生支持 FP8 矩阵乘**，所以不仅省显存，还能**算得更快**。

vLLM 把这件事封装成一个开关：推理时给模型加一个 `quantization="fp8"`，或者直接加载已经量化好的 FP8 权重，就能享受这套收益。

---

## 1. 地基：FP8 解决什么问题

大模型推理有三堵墙，FP8 同时缓解了前两堵：

```
       一次推理的瓶颈
   ┌──────────────────────────┐
   │ ① 显存墙：权重 + KV Cache  │  70B 模型 FP16 需 ~140GB，单卡放不下
   │ ② 带宽墙：解码阶段每生成   │  GPU 要把全部权重从 HBM 搬到 SM，
   │    1 个 token 都要重读权重 │  搬运量 = 瓶颈（memory-bound）
   │ ③ 算力墙：Prefill 阶段大   │  GEMM 计算密集（compute-bound）
   │    矩阵乘                   │
   └──────────────────────────┘
        │                │              │
   FP8 权重减半       FP8 搬运量减半    FP8 张量核 ~2x FLOPS
   (省显存)          (解码更快)       (Prefill 更快)
```

- **显存墙**：权重从 16 bit → 8 bit，**权重显存减半**。70B 模型权重从 ~140GB 降到 ~70GB，可能从「要 2 张卡」变成「1 张卡放得下」。
- **带宽墙（解码阶段最关键）**：自回归解码是 **memory-bound** 的——每生成一个 token，GPU 都要把整套权重从 HBM（显存）读进计算单元。权重字节数减半 → 搬运时间减半 → **解码吞吐近乎翻倍**。
- **算力墙（Prefill 阶段）**：处理长 prompt 时是大矩阵乘，**compute-bound**。Hopper/Ada 的 FP8 张量核峰值算力约为 FP16 的 2 倍，Prefill 更快。

> 一句话：FP8 是「**几乎不掉精度**」前提下，**同时省显存 + 提吞吐**的高性价比手段，这也是它在生产推理里迅速普及的原因。

---

## 2. FP8 的两种格式：E4M3 与 E5M2

浮点数 = 符号位(S) + 指数位(E) + 尾数位(M)。8 个比特怎么分配，有两套业界标准（OCP FP8）：

```
 格式      位分配               动态范围        精度(尾数)   典型用途
 ─────────────────────────────────────────────────────────────────
 E4M3   S EEEE MMM           小（~±448）       高(3 尾数位)  权重 / 激活
        1  4   3

 E5M2   S EEEEE MM           大（~±57344）     低(2 尾数位)  梯度 / 误差
        1  5    2
```

**怎么理解这两者的取舍？**

- **指数位越多 → 能表示的数值范围越大**（能容纳很大或很小的数），但代价是**尾数位变少 → 分辨率（精度）变粗**。
- **E4M3**：范围小但精度高。推理时权重和激活的数值分布相对集中，更在意精度，所以**推理量化几乎都用 E4M3**。
- **E5M2**：范围大但精度低，更像 FP16 的「指数布局」，常用于训练中梯度这类动态范围大的张量。

数值直觉：FP8 只有 256 个可表示的值（其中还有特殊值）。所以**不可能直接把任意 FP16 数字硬塞进 FP8**——必须先把数值「压缩到 FP8 能表示的范围」，这就引出了下一节的**缩放因子**。

> 注意：vLLM 推理路径默认使用 **E4M3**。不同硬件对 E4M3/E5M2 的支持和具体表示（如是否有 inf/nan）略有差异，**确切的位级语义以硬件文档与 vLLM/框架实现为准**。

---

## 3. 量化的本质：缩放因子 scale

量化的核心动作只有一步：**把一个分布范围较大的高精度张量，线性映射到 FP8 的小范围里**。

设原始权重张量为 $W$（FP16），FP8 可表示的最大绝对值为 $F_{max}$（E4M3 约 448）。我们求一个**缩放因子** $s$：

$$ s = \frac{\max(|W|)}{F_{max}} \qquad W_{fp8} = \text{round}\!\left(\frac{W}{s}\right) $$

计算时再**反量化（dequantize）**还原回近似值：

$$ \hat{W} = W_{fp8} \times s \approx W $$

**为什么需要 scale？** 因为不同层、不同通道的权重数值大小差异巨大。如果用同一个固定映射，数值小的张量量化后几乎全是 0，数值大的会溢出。scale 让每个张量「**各自缩放到 FP8 的甜区**」。

**scale 的粒度（granularity）决定精度 vs 开销的取舍：**

```
 per-tensor（整张量一个 scale）
   ┌───────────────┐
   │  W            │ → 一个 s ── 省存储、最快，但被极端值拉偏
   └───────────────┘

 per-channel / per-token（每行/每列一个 scale）
   ┌─┬─┬─┬─┬─┐
   │s│s│s│s│s│  → 每通道一个 s ── 更贴合分布，精度更好，开销略增
   └─┴─┴─┴─┴─┘
```

- **per-tensor**：整个张量共用一个 scale。最省、最快，但若张量内有离群值（outlier），会把所有正常值压扁。
- **per-channel（权重常用）/ per-token（激活常用）**：每个输出通道或每个 token 单独一个 scale，更能抵抗离群值，精度更高，是当前主流权重量化的默认粒度。

> 离群值是 LLM 量化的头号难题（少数维度数值特别大）。FP8 因为是「浮点」量化、本身带指数位、动态范围比 INT8 宽，对离群值的容忍度天然比 INT8 好——这是 FP8 推理精度普遍优于 INT8 的根本原因之一。对比 INT8/SmoothQuant 路线见 [[llm-compression/quantization/SmoothQuant]]。

---

## 4. 量化谁？权重 vs 激活，dynamic vs static

「W8A8」这类记号含义：**W = Weight 权重，A = Activation 激活**，数字是位宽。

```
 命名      权重    激活     说明
 ───────────────────────────────────────────────
 W8A16   FP8     FP16    只量化权重(weight-only)，激活仍高精度
 W8A8    FP8     FP8     权重+激活都量化，张量核全程跑 FP8，最快
```

- **W8A16（weight-only）**：只把权重压成 FP8，激活保持 FP16，计算前把权重反量化。**省显存 + 解码提速明显**（解决带宽墙），但 GEMM 仍是 FP16 算力。门槛低、精度稳，适合显存吃紧但算力够用的场景。
- **W8A8**：权重和激活都 FP8，矩阵乘**全程走 FP8 张量核**，Prefill/算力提升最大，但激活也要量化，**对精度要求更高、更依赖好的激活 scale**。

**激活 scale 怎么来？两种方案：**

| 方案 | scale 来源 | 是否需要校准数据 | 特点 |
| --- | --- | --- | --- |
| **dynamic（动态）** | 推理时**实时**按当前激活算 scale | 不需要 | 精度好、零准备成本；每步多一点点开销 |
| **static（静态）** | **离线**用一批校准样本统计出固定 scale，存进权重 | 需要校准集 | 推理时省去在线统计，**延迟更低**；scale 固定，分布偏移大时风险略高 |

对应到文件最初的例子：
- `activation_scheme="dynamic"` → 不需要 `examples`，所以校准样本是空 `[]`。
- `activation_scheme="static"` → 需要喂一批 `examples`（如 512 条 ultrachat 样本）来统计激活的固定 scale。

---

## 5. vLLM 里 FP8 的整体数据流

无论哪条路径，运行时的核心机制是一样的：**FP8 存、按需反量化、在张量核上算**。

```
                vLLM FP8 推理一层 Linear 的数据流
  ┌──────────────────────────────────────────────────────────┐
  │  显存(HBM) 里存的是：                                       │
  │     W_fp8 (8-bit)  +  weight_scale  [ +激活 scale(static)] │
  └──────────────────────────────────────────────────────────┘
                         │  搬运量减半（带宽友好）
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  激活 X (FP16)                                             │
  │   ├─ W8A16: 反量化 W_fp8×scale → FP16，再做 FP16 GEMM      │
  │   └─ W8A8 : X 也量化成 FP8(dynamic/static 取 scale)，      │
  │              两个 FP8 直接进【FP8 张量核】做矩阵乘           │
  └──────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  累加用高精度（FP32/FP16 累加器）→ 输出 FP16，进下一层      │
  └──────────────────────────────────────────────────────────┘
```

要点：
1. **存得小**（FP8）→ 搬运快、显存省；
2. **算之前才反量化 / 把激活也量化**，匹配硬件张量核的输入要求；
3. **累加器仍是高精度（FP32）**，避免长串乘加的误差累积——这是 FP8 精度能保持的关键工程细节。

---

## 6. 三种使用路径

vLLM 支持三种把模型跑成 FP8 的方式，按「谁来做量化、什么时候做」区分：

```
 路径 A  在线动态量化(on-the-fly)
   原始 FP16 权重 ──加载时──> 临时量化成 FP8 ──推理
   quantization="fp8"，零准备，但每次启动都要量化、首包稍慢

 路径 B  离线量化(AutoFP8 / 现已多用 llm-compressor)
   原始权重 ──AutoFP8.quantize()──> 存一份 FP8 checkpoint ──vLLM 加载
   一次量化、反复加载；可选 dynamic / static 激活 scale

 路径 C  直接加载社区已量化好的 FP8 权重
   Hugging Face 上 *-FP8 / *-FP8-Dynamic 仓库 ──vLLM 直接 serve
   vLLM 读权重里的量化配置自动识别，最省事
```

**怎么选？**
- 想快速试一下、不想准备数据 → **路径 A**（在线 `quantization="fp8"`，本质是 weight-only 动态量化）。
- 要部署、追求稳定可复现、想用 static 激活 scale 把延迟压到最低 → **路径 B**。
- 别人已经发布了高质量 FP8 权重 → **路径 C**，直接 serve 最省心。

> 关于工具：早期离线量化用 **AutoFP8**（即文件原例），目前社区主推 **llm-compressor**（更通用，支持 FP8/INT8/W4A16 等多种方案）。**AutoFP8 部分功能已并入/被 llm-compressor 取代，具体以官方仓库 README 为准**。相关见 [[llm-compression/quantization/量化基础]]。

---

## 7. KV Cache 也能 FP8

除了权重/激活，**KV Cache（注意力的 Key/Value 缓存）也可以量化成 FP8**，这是长上下文/高并发场景的关键优化。

- KV Cache 占用随 **序列长度 × batch** 线性增长，长上下文时它甚至比权重还吃显存。
- 把 KV Cache 从 FP16 → FP8，**KV 显存减半**，等价于能放更长的上下文、或更大的 batch（更高吞吐）。
- vLLM 通过类似 `kv_cache_dtype="fp8"` 的开关启用；不同实现下可能区分 `fp8_e4m3` / `fp8_e5m2`，且可能需要校准 KV 的 scale。**确切参数名与取值以 vLLM 官方文档为准**。

```
 长上下文显存占用对比（示意）
   FP16 KV: ███████████████████  100%
   FP8  KV: ██████████           ~50%   → 同显存可塞 ~2x 上下文/并发
```

详见 [[llm-compression/quantization/kv-cache-quant]] 与 [[llm-optimizer/kv-cache]]。

---

## 配置示例（讲含义，不背默认值）

### A. 在线动态量化（最简单，weight-only）

```python
from vllm import LLM

# quantization="fp8" 让 vLLM 在【加载权重时】把 FP16 权重临时量化成 FP8。
# 不需要任何校准数据，代价是每次启动多一步量化、首包略慢。
model = LLM("facebook/opt-125m", quantization="fp8")
# 加载后权重显存约减半：Loading model weights took ~0.155 GB
result = model.generate("Hello, my name is")
```

### B. 离线量化 —— 动态激活 scale（无需校准集）

```python
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir  = "Meta-Llama-3-8B-Instruct-FP8-Dynamic"

# activation_scheme="dynamic": 激活 scale 推理时实时算 → 不需要校准样本
quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")
examples = []  # dynamic 不需要校准数据

model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples)
model.save_quantized(quantized_model_dir)   # 产出一份可被 vLLM 直接加载的 FP8 权重
```

### C. 离线量化 —— 静态激活 scale（需校准集，推理延迟更低）

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir  = "Meta-Llama-3-8B-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 用一批真实样本「跑一遍前向」，统计激活的取值范围 → 算出固定 static scale
ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(512))
examples = [tokenizer.apply_chat_template(b["messages"], tokenize=False) for b in ds]
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to("cuda")

# activation_scheme="static": scale 离线固定，推理时省去在线统计 → 延迟更低
quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static")

model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
```

**三者关系一句话**：A 是「加载时临时量化、weight-only」；B/C 是「离线产出可复用的 FP8 权重」，区别只在激活 scale 是 **动态算** 还是 **静态固定**。

### 部署（vLLM serve 已量化权重）

加载路径 B/C 产出的（或 HF 上现成的）FP8 权重时，vLLM 会读取权重目录里的量化配置自动识别，无需再手动指定方案。命令大致形如 `vllm serve <FP8权重路径>`。**确切 CLI 参数见** [[llm-inference/vllm/服务启动参数]]。

---

## 常见问题 / 坑

| 现象 / 问题 | 原因 | 应对 |
| --- | --- | --- |
| 启用 FP8 报硬件不支持 / 不加速 | FP8 张量核需 **Hopper(H100/H200) 或 Ada(L40S/L4)** 等较新架构；老卡（A100/V100）无原生 FP8 | 老卡退回 INT8/AWQ/GPTQ，或仅用 weight-only 省显存 |
| 在线 `quantization="fp8"` 启动慢 | 每次启动都在线量化整套权重 | 改用离线量化（路径 B）产出 checkpoint 复用 |
| static 量化后某些任务精度掉 | 校准集分布与实际请求差异大，固定 scale 失配 | 换更贴近线上分布的校准数据，或改用 dynamic |
| 精度比预期差 / 个别层崩 | per-tensor 粒度被离群值拉偏；敏感层（如 lm_head）量化 | 用 per-channel/per-token 粒度；对敏感层保持高精度（混合精度） |
| KV Cache FP8 后长文回答质量下降 | KV scale 未校准好 / e5m2 精度过低 | 优先 e4m3，必要时校准 KV scale；权衡上下文长度 vs 质量 |
| 用 AutoFP8 发现不再维护 | 工具迭代，主线转向 llm-compressor | 新项目优先 **llm-compressor**，老 checkpoint 仍可加载 |
| 以为 FP8 一定比 INT8 快/准 | 取决于硬件张量核支持与离群值情况 | FP8 通常精度更稳；速度看是否有原生 FP8 单元，实测为准 |

> 通用提醒：本文中所有**确切 CLI 参数名、默认值、库的最新接口与版本支持**，请以 **vLLM 官方文档 / AutoFP8 / llm-compressor 仓库源码为准**；FP8 领域工具迭代很快。

---

## 🔗 跳转链接

- 总图：[[00-知识地图]]
- vLLM 总览与参数：[[llm-inference/vllm/README]] · [[llm-inference/vllm/服务启动参数]]
- 量化原理打底：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/大模型量化概述]]
- 对照路线（INT8/离群值）：[[llm-compression/quantization/SmoothQuant]] · [[llm-compression/quantization/LLM-int8]]
- KV Cache 相关：[[llm-compression/quantization/kv-cache-quant]] · [[llm-optimizer/kv-cache]]
