# 多模态大模型（Multi-Modal）

> 让模型同时"看图 + 读文 + 写文/画图"——把视觉编码器接到语言模型上，统一在一个表示空间里对齐。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] · [[llm-train/peft/PEFT-API]] · [[llm-inference/README]] · [[llm-compression/quantization/量化基础]]

## 阅读地图

| 你想知道 | 看哪节 |
|---|---|
| 多模态到底在解什么问题 | §0 一句话锚点 |
| 视觉和文本怎么进同一个模型 | §1 地基：模态对齐 |
| CLIP 为什么是地基中的地基 | §2 对比学习 CLIP |
| BLIP / BLIP2 / InstructBLIP 怎么演进 | §3 图生文路线 |
| LLaVA / miniGPT4 这类对话模型 | §4 视觉指令微调 |
| Stable Diffusion 文生图怎么炼 | §5 扩散模型 + LoRA |
| 一张图能干哪些活 | §6 八大视觉语言任务 |
| FLAVA 这种通用底座 | §7 多模态通用模型 |
| 真实命令/数据集/模型卡 | 实操一节 |
| 容易踩的坑 | 常见问题 |

## 0. 一句话锚点

多模态大模型 = **视觉编码器**（把像素变成向量）+ **跨模态对齐/桥接**（让图向量和文字向量住在同一个空间）+ **语言模型**（在这个空间里推理、生成文字）。

```
   图像 ──► [视觉编码器 ViT] ──► 图像 token ┐
                                          ├─► [跨模态桥接] ─► [LLM] ─► 文字
   文本 ──► [Tokenizer/Embed] ──► 文本 token┘
```

反向的"文生图"则是另一条路：文本 → 条件 → **扩散模型**逐步去噪 → 图像（§5）。

## 1. 地基：什么叫"模态对齐"

不同模态天生不在一个空间：一张猫的图片是 `224×224×3` 的像素张量，"a cat" 是 3 个 token 的 id。要让模型理解"这张图 = 这句话"，必须把它们**映射到同一个向量空间**，使语义相近的图和文向量靠得近。

- **相似度**用余弦：$\text{sim}(I, T) = \dfrac{\mathbf{v}_I \cdot \mathbf{v}_T}{\|\mathbf{v}_I\|\,\|\mathbf{v}_T\|}$，取值 $[-1, 1]$，越大越像。
- 训练目标就是：配对的 (图, 文) 相似度拉高，不配对的拉低。

```
对齐前(各说各话)            对齐后(同一空间)
  图空间    文空间             共享空间
  ●cat                          ●cat图 ●cat文  (近)
        ●dog文                  ●dog图 ●dog文  (近)
  ●dog图                        cat 与 dog 相互远
```

这就是 CLIP 干的事，也是后面所有"看图说话"模型的地基。

## 2. 对比学习 CLIP——多模态的"地基"

CLIP（Contrastive Language–Image Pre-training）用**对比学习**在 4 亿网络图文对上训练：一个 batch 里 $N$ 张图、$N$ 句文，组成 $N\times N$ 相似度矩阵，对角线（真实配对）是正样本，其余 $N^2-N$ 个是负样本。

```
        文T1   文T2   文T3  ... 文TN
  图I1 [ ★ ]   .      .         .      ← 对角线拉高
  图I2   .   [ ★ ]    .         .
  图I3   .     .    [ ★ ]       .
  ...
  图IN   .     .      .       [ ★ ]
        ↑ 列方向 softmax + 行方向 softmax，双向对比损失
```

损失是对称 InfoNCE：

$$\mathcal{L} = \tfrac12\big(\mathcal{L}_{I\to T} + \mathcal{L}_{T\to I}\big),\quad \mathcal{L}_{I\to T} = -\frac1N\sum_i \log\frac{e^{\text{sim}(I_i,T_i)/\tau}}{\sum_j e^{\text{sim}(I_i,T_j)/\tau}}$$

其中 $\tau$ 是可学习的温度系数。**数值直觉**：若某行真实配对 logit 经温度缩放后 softmax 概率是 0.8，该样本贡献 $-\log 0.8 \approx 0.22$；若降到 0.1，则 $-\log 0.1 \approx 2.30$，损失放大十倍，逼模型把对角线顶上去。

CLIP 训完后两大用途：① **零样本分类**——把类别名写成"a photo of a {class}"，看哪句和图最像；② 给下游模型当**视觉编码器**（LLaVA、BLIP 都用 CLIP/EVA-CLIP 系视觉塔）。

## 3. 图生文路线：BLIP → BLIP2 → InstructBLIP

**BLIP** 在 CLIP 的"理解"之外加了"生成"，用 caption 数据训练"看图写字"。

**BLIP2** 是工程上的关键跳板——它**冻结**了昂贵的视觉编码器和 LLM，中间只训一个轻量的 **Q-Former**（Querying Transformer）：

```
[冻结 图像编码器] ─► 图特征 ─┐
                            ├─► [Q-Former: 32个可学query] ─► [冻结 LLM(OPT/T5)] ─► 文字
        文本提示 ───────────┘
```

- 为什么这么设计：视觉塔和 LLM 都是几十亿参数、预训练昂贵，**只训中间桥**（Q-Former 仅约 1 亿参数）就能让二者协作，训练成本骤降。
- 32 个 query token 像"摘要槽"，把任意分辨率图像压成定长的视觉 token 喂给 LLM，解决了视觉特征过长的问题。

仓库给出的 BLIP2 真实资源：模型 `Salesforce/blip2-opt-2.7b`，做 image→text（零样本图生文）；并提供 **int8 量化微调**脚本（见实操一节），即把 2.7B 的 OPT 以 8-bit 加载后只训 LoRA/适配层，单卡可跑。

**InstructBLIP** 在 BLIP2 基础上引入**指令感知的 Q-Former**——把指令文本也喂给 Q-Former，让它"按问题"提取视觉特征，提升通用视觉问答能力。**MDETR** 则是另一支：调制式检测，做细粒度视觉定位（短语 → 框）。

## 4. 视觉指令微调：LLaVA / miniGPT4

这一类把"对话能力强的 LLM"直接接上视觉塔，目标是**多轮看图对话**。

```
图 ─►[CLIP/EVA 视觉塔]─► 视觉特征 ─►[投影层 MLP]─► 视觉token ┐
                                                          ├─►[LLM(Vicuna等)]─► 回答
            "图里有什么?" ─► 文本token ───────────────────┘
```

- **LLaVA**：视觉塔冻结，仅训一个**线性/MLP 投影层**把视觉特征对齐到 LLM 词嵌入空间，再用 GPT 生成的视觉指令数据做指令微调。轻量、效果好，成了开源多模态对话的事实基线。
- **miniGPT4**：思路类似，用一层投影把 BLIP2 的视觉表示接到冻结的 Vicuna 上。

关键对照：BLIP2 用 **Q-Former** 桥接（重），LLaVA 用 **一层投影** 桥接（轻），后者更易复现、更省训练量。

## 5. 文生图：Stable Diffusion + LoRA 微调

反方向——**文本 → 图像**，用**扩散模型**（Diffusion）。原理：前向不断给图像加高斯噪声直到变成纯噪声，训练一个网络学会**反向逐步去噪**；推理时从随机噪声出发，以文本为条件一步步还原成图。

```
前向(训练造数据): 清晰图 ─加噪→─加噪→ ... → 纯噪声
反向(推理生成):   纯噪声 ─去噪→─去噪→ ... → 清晰图
                          ↑ 每步以 "文本提示" 为条件 (CLIP文本编码)
```

Stable Diffusion 在**潜空间**做这件事（先用 VAE 把图压到低维 latent，再在 latent 上扩散），所以快。多模态任务：**文生图、图生图**。

**为什么用 LoRA 微调而不全量训**：SD 的 UNet 很大，全量微调显存高、易过拟、产物几个 GB。LoRA 只在注意力权重旁加低秩矩阵 $\Delta W = BA$（$B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}$，秩 $r$ 远小于 $d,k$），冻结原权重只训 $A,B$。**数值示例**：一层 $1024\times1024$ 全量是 $\approx 1.05\text{M}$ 参数，取 $r=8$ 的 LoRA 只有 $1024\times8 + 8\times1024 \approx 16\text{K}$ 参数，约为 **1.5%**，产物从 GB 级降到几十 MB。

仓库列出 PEFT 对扩散微调支持三种技术，本质都是"低秩/低维增量"：

| 技术 | 含义 | 直觉 |
|---|---|---|
| **LoRA** | 低秩矩阵 $\Delta W = BA$ | 经典低秩增量 |
| **LoHa** | Hadamard 积低秩分解 | 表达力更强、参数仍少 |
| **LoKr** | Kronecker 积分解 | 用克罗内克积构造大矩阵、参数极省 |

经典做法 **DreamBound/DreamBooth + LoRA**：给几张某主体（如特定宠物）的图，让 SD 学会"按提示画这个主体"。仓库给出的真实模型/数据见实操一节。

## 6. 八大视觉语言任务（给一张图能干什么）

| 任务 | 输入 | 输出 | 代表能力 |
|---|---|---|---|
| 一、VQA 视觉问答 | 图 + 自然语言问题 | 答案（单词/短语） | 通用/文本导向 VQA |
| 二、Image Caption 图像字幕 | 一张图 | 一句自然语言描述 | 零样本图像描述生成 |
| 三、Referring Expression 指代表达 | 图 + 一句描述 | 判断描述是否正确（对/错） | 细粒度视觉定位 |
| 四、Visual Dialogue 视觉对话 | 一张图 | 多轮交互对话 | 看图聊天 |
| 五、VCR 视觉常识推理 | 1 问题 + 4 答案 + 4 理由 | 正确答案及理由 | 常识推理 |
| 六、NLVR 自然语言视觉推理 | 2 张图 + 一句描述 | true / false | 跨图判断 |
| 七、Visual Entailment 视觉蕴含 | 图 + 文本 | 蕴含 / 中性 / 矛盾 三类概率 | 图文一致性 |
| 八、Image-Text Retrieval 图文检索 | 图或文 | 文或图 | 以图搜文/以文搜图/以图搜图 |

更高层把它们归为四种范式：**文生图**（文本→图像）、**视觉问答**（图+文→文）、**多模态分类**（图+文→标签）、**更好的理解/生成**（图+文→标签/文本）。

```
       图文检索的三种打开方式
  以图搜文: 图 ─►(找最相似的文)─► 文
  以文搜图: 文 ─►(找最相似的图)─► 图
  以图搜图: 图 ─►(找最相似的图)─► 图
   ↑ 全靠 §2 CLIP 那个共享空间里的余弦相似度
```

## 7. 多模态通用模型 FLAVA

FLAVA（Foundational Language And Vision Alignment）是 Meta 的**通用底座**，一套模型同时覆盖"单模态视觉 + 单模态文本 + 跨模态"三类任务，用多目标联合预训练（图文对比 + 掩码图像/文本建模 + 多模态匹配）。代码示例见 `facebookresearch/multimodal` 的 `examples/flava`。

## 实操：真实命令 / 模型 / 数据集

### Stable Diffusion + LoRA / DreamBooth 微调（PEFT）

```
# 文档：DreamBooth + LoRA 任务指南
https://huggingface.co/docs/peft/task_guides/dreambooth_lora
# 训练示例代码（PEFT v0.6.2）
https://github.com/huggingface/peft/tree/v0.6.2/examples/lora_dreambooth
# 推理 notebook
https://github.com/huggingface/peft/blob/v0.6.2/examples/lora_dreambooth/lora_dreambooth_inference.ipynb
# 训练用图像数据
https://huggingface.co/datasets/diffusers/docs-images
# 扩散模型库（diffusers）教程总览
https://huggingface.co/docs/diffusers/tutorials/tutorial_overview
```

- 支持的微调技术：**LoRA、LoHa、LoKr**（见 §5 对照表）。
- 基础模型：`CompVis/stable-diffusion-v1-4`
- 微调数据集：`lambdalabs/pokemon-blip-captions`
  → https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions

### BLIP2 图生文 / int8 量化微调

```
# int8 微调脚本（PEFT v0.6.2）：8-bit 加载 2.7B OPT，只训适配层
https://github.com/huggingface/peft/blob/v0.6.2/examples/int8_training/fine_tune_blip2_int8.py
# 模型卡与示例
https://huggingface.co/Salesforce/blip2-opt-2.7b
# 教程：使用 BLIP-2 零样本"图生文"
https://huggingface.co/blog/zh/blip-2
```

### 其他参考实现

```
# FLAVA 多模态通用模型示例
https://github.com/facebookresearch/multimodal/tree/main/examples/flava
# CogView 文生图
https://github.com/THUDM/CogView
```

涉及的代表算法：**CLIP、BLIP、BLIP2、LLaVA、miniGPT4、InstructBLIP、MDETR、FLAVA、Stable Diffusion**。

## 常见问题 / 坑

| 现象 / 坑 | 原因 | 对策 |
|---|---|---|
| 微调链接打不开/报错 | 链接钉死在 **PEFT v0.6.2** | 复现就用该 tag；换新版需对照 API 变更 |
| BLIP2 显存爆 | OPT-2.7B 全精度太大 | 用 `int8_training` 脚本 8-bit 加载 + LoRA |
| SD LoRA 产物巨大或过拟 | 误用全量微调 | 用 LoRA/LoHa/LoKr，秩 $r$ 取 4~16 |
| DreamBooth 学不像主体 / 语言漂移 | 样本太少或步数过多 | 配先验保留正则、控制步数与学习率 |
| CLIP 零样本分类效果差 | 提示词模板不对 | 用"a photo of a {class}"等模板，多模板集成 |
| Q-Former vs 投影层选型纠结 | 二者权衡不同 | 求轻量易复现选投影(LLaVA)；求强桥接选 Q-Former(BLIP2) |
| 把"文生图"和"图生文"混为一谈 | 方向相反、架构不同 | 图生文=视觉塔+LLM；文生图=扩散去噪 |
| 余弦相似度未归一化 | 漏了 $\|\mathbf{v}\|$ 归一 | 比相似度前先做 L2 normalize |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 底座架构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 推理优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理部署：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 压缩量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练与微调：[[llm-train/README]] · [[llm-train/peft/PEFT-API]]
- 对齐：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 硬件与网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 内存：[[docs/transformer内存估算]]
