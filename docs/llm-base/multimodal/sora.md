# Sora：视频生成大模型（Diffusion Transformer 视角）

> 一句话定位：Sora 是 OpenAI 的文生视频模型，本质是「**视频 VAE 把时空像素压成隐空间 token → Diffusion Transformer 在隐空间里去噪生成 → VAE 解码回像素**」的三段式管线。
> 📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] · [[llm-optimizer/FlashAttention]] · [[llm-algo/旋转编码RoPE]] · [[llm-inference/大模型推理张量并行]] · [[llm-alignment/RLHF]]

> 说明：本文讲**稳定的原理与机制**。Sora 未开源、技术报告只给思路不给超参，文中数字均为**量级估算/演示用途**（标注「估算」），不代表官方真实配置；架构细节以 OpenAI 官方技术报告与公开论文为准。

---

## 阅读地图

| 你想搞清楚的问题 | 看哪一节 |
| --- | --- |
| Sora 整条管线长什么样 | §0 一句话锚点、§1 总览 |
| 扩散模型（Diffusion）的最底层原理 | §2 前向加噪 / §3 反向去噪 |
| 为什么用 Transformer 而不是 U-Net（DiT） | §4 DiT |
| 视频怎么变成 token（时空 patch） | §5 视频 VAE / §6 Spacetime Patch |
| 怎么支持任意分辨率/时长（NaViT） | §7 变长 token |
| 文本条件怎么注入（DALL-E 3 caption） | §8 条件注入 |
| 一条视频要多少 token、多少算力 | §9 数值手算 |
| 推理时怎么一步步采样出视频 | §10 采样 |
| 和图像扩散/LLM 的异同 | 常见问题 |

---

## 0. 一句话锚点

```
文本 prompt ─┐
             ▼
   [文本编码器] ── 条件 c
             │
噪声张量 z_T ─┼──► [Diffusion Transformer] ──去噪 T 步──► 干净隐变量 z_0
  (隐空间)   │         (在压缩后的隐空间里反复去噪)
             │
   z_0 ──► [VAE Decoder] ──► 视频像素帧序列
```

记住三句话：(1) **不在像素上做扩散**，而在 VAE 压缩后的**隐空间（latent）**做——省几十倍算力；(2) **去噪网络是 Transformer（DiT）**，不是 U-Net——天然可扩展、可变长；(3) **视频被切成时空 patch（spacetime token）**，像 GPT 处理文字 token 一样处理视频 token。

谱系（沿用本文件已有结论）：

```
Meta DiT (2022.12)        → 用 Transformer 替换扩散里的 U-Net
Google MAGViT-v2 (2023)   → 视频/图像统一的离散/连续 tokenizer
DeepMind NaViT (2023.07)  → Patch n' Pack，支持任意分辨率/比例
OpenAI DALL-E 3 (2023.09) → 用模型重写高质量 caption，造文本-视频对
        └──────────────► Sora (2024.02)
```

---

## 1. 总览：三段式管线

Sora 不是一个网络，而是一条**压缩—生成—解压**的管线：

```
┌────────────────────────────────────────────────────────────┐
│                       训 练 阶 段                            │
│                                                              │
│  原始视频 x  ──Encoder──►  隐变量 z_0  ──加噪──►  z_t         │
│  (T,H,W,3)              (t,h,w,c)             (t,h,w,c)       │
│                                          │                   │
│   去噪网络 ε_θ(z_t, t, c)  ◄─── 学习预测加进去的噪声 ε        │
│                                                              │
│  损失：‖ε - ε_θ(z_t, t, c)‖²   (条件 c = 文本/时长/分辨率)    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                       推 理 阶 段                            │
│                                                              │
│  纯噪声 z_T ──去噪T步──► z_0 ──Decoder──► 视频 (T,H,W,3)      │
│   (在隐空间里反复跑 DiT，每步去掉一点噪声)                    │
└────────────────────────────────────────────────────────────┘
```

| 模块 | 作用 | 借鉴 | 类比 LLM |
| --- | --- | --- | --- |
| 视频 VAE Encoder | 时空下采样、压成隐变量 | MAGViT | tokenizer（编码） |
| Patchify | 隐变量切成 spacetime token | ViT/NaViT | 分词成 token 序列 |
| Diffusion Transformer | 在隐空间里去噪 | DiT | Transformer 主干 |
| Patchify⁻¹ + VAE Decoder | token→隐变量→像素 | MAGViT | de-tokenize（解码） |

---

## 2. 地基一：扩散模型的前向过程（加噪）

扩散模型的核心直觉：**任何复杂数据，反复加高斯噪声，最终变成纯高斯噪声**；那么只要学会「**把噪声一点点去掉**」，就能从纯噪声里生出数据。

前向（forward / diffusion）过程：给干净数据 $z_0$ 逐步加噪，共 $T$ 步：

$$q(z_t \mid z_{t-1}) = \mathcal{N}\big(z_t;\ \sqrt{1-\beta_t}\,z_{t-1},\ \beta_t \mathbf{I}\big)$$

其中 $\beta_t$ 是第 $t$ 步的噪声方差（noise schedule，随 $t$ 增大）。关键的「**任意一步闭式解**」（DDPM 推导）：令 $\alpha_t = 1-\beta_t$、$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$，则可一步跳到任意 $t$：

$$z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,\mathbf{I})$$

```
 t=0      t=T/4     t=T/2     t=3T/4     t=T
 ┌──┐     ┌──┐      ┌──┐      ┌──┐       ┌──┐
 │🐱│ ──► │🐱·│ ──► │·🐱·│──►  │░░·│ ──►  │▒▒│
 └──┘     └──┘      └──┘      └──┘       └──┘
清晰图    略带噪    一半噪      大量噪     纯噪声
 z_0                                      z_T ~ N(0,I)
       ──────────── 加噪（已知、无需学习）────────►
```

**为什么是闭式？** 因为高斯加高斯还是高斯，$t$ 步的累积效果可以合并成「一份 $z_0$ + 一份噪声 $\varepsilon$」，权重就是 $\sqrt{\bar\alpha_t}$ 和 $\sqrt{1-\bar\alpha_t}$。训练时这让我们能**直接采任意 $t$**，不用真的跑 $t$ 步——这是扩散能训练的关键工程点。

---

## 3. 地基二：反向过程（去噪 = 真正要学的网络）

反向（reverse / denoising）：从 $z_T \sim \mathcal{N}(0,\mathbf{I})$ 出发，一步步还原 $z_0$。我们训练一个网络 $\varepsilon_\theta$ 去**预测当初加进去的噪声 $\varepsilon$**：

$$\mathcal{L} = \mathbb{E}_{z_0,\varepsilon,t,c}\ \big\|\,\varepsilon - \varepsilon_\theta(\underbrace{\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\varepsilon}_{z_t},\ t,\ c)\,\big\|^2$$

训练一步的流程（**整条管线的核心循环**）：

```
1. 取一条视频 → Encoder → z_0
2. 随机采步数 t ~ U(1,T)，采噪声 ε ~ N(0,I)
3. 合成带噪样本  z_t = √ᾱ_t·z_0 + √(1-ᾱ_t)·ε
4. 喂给 DiT：     ε̂ = ε_θ(z_t, t, c)        ← c 是文本等条件
5. 回归损失：     L = ‖ε - ε̂‖²
6. 反向传播更新 θ
```

```
   z_t ──►┌─────────────┐──► ε̂ (预测的噪声)
 t   ───►│ Diffusion   │        │
 c(文本)─►│ Transformer │        ▼  比较
          └─────────────┘   L = ‖ε - ε̂‖²
                                 ▲
                            ε (真实噪声)
```

**为什么预测「噪声」而不是直接预测「干净图」？** 数学上等价（已知 $z_t,t$ 时 $z_0$ 与 $\varepsilon$ 一一对应），但预测 $\varepsilon$ 时目标方差被归一化得更稳定，是 DDPM 的经验最佳实践。Sora 这类大模型也常用 $v$-prediction 等变体，原理同源。

---

## 4. 为什么是 Transformer：从 U-Net 到 DiT

历史上扩散去噪网络 $\varepsilon_\theta$ 用 **U-Net**（卷积 + 下/上采样 + skip）。Meta 的 **DiT（Diffusion Transformer, 2022.12）** 把 U-Net 换成纯 Transformer，Sora 沿用之。

```
   传统 U-Net                       DiT (Sora 路线)
 ┌──────────┐                    ┌──────────────────┐
 │  Conv↓   │                    │ Patchify         │
 │   Conv↓  │ skip               │  → token 序列    │
 │    bottl │◄───┐               │ +位置编码        │
 │   Conv↑  │    │               ├──────────────────┤
 │  Conv↑   │────┘               │ N × Transformer  │
 └──────────┘                    │  Block (Attn+MLP)│
 卷积，局部感受野                 │  +AdaLN(t,c 注入)│
 难扩展、难变长                   ├──────────────────┤
                                 │ Unpatchify       │
                                 └──────────────────┘
                                 全局注意力，天然可扩展/变长
```

DiT Block（每块 = 自注意力 + MLP，时间步 $t$ 与条件 $c$ 经 **AdaLN-Zero** 调制）：

```
 token ─► LayerNorm ─► [缩放γ/偏移β 来自 (t,c)] ─► Self-Attn ─► +残差
       ─► LayerNorm ─► [缩放γ/偏移β 来自 (t,c)] ─► MLP      ─► +残差
                         ▲
            t,c ──► MLP ─┘  (AdaLN：把条件变成每层的 scale/shift)
```

**为什么 DiT 适合 Sora？**

1. **可扩展性**：Transformer 加深加宽、加数据，loss 平滑下降，符合 scaling law；Sora 报告里展示「算力越大、视频越连贯」正是这个特性。
2. **变长友好**：注意力对序列长度无结构假设，能处理不同分辨率/时长产生的**不同 token 数**（见 §7）。
3. **复用 LLM 基建**：FlashAttention、张量并行、KV 优化等大模型工程能力可直接迁移（见 [[llm-optimizer/FlashAttention]]、[[llm-inference/大模型推理张量并行]]）。

> 注意力与 MLP 的算力占比、$O(L^2)$ 复杂度等，与文本 Transformer 完全同源，参见 [[llm-algo/transformer/模型架构]]、[[llm-algo/FLOPs]]、[[llm-algo/mlp]]。

---

## 5. 视频 VAE：把时空压成隐空间

直接在像素上做扩散太贵：一段 $1080\text{p}\times 60\text{s}$ 的视频是天文数字的像素。所以先用 **视频 VAE（时空自编码器，借鉴 MAGViT-v2）** 同时在**空间和时间**两个维度压缩。

```
 原始视频           视频 VAE Encoder            隐变量 latent
 (T, H, W, 3)  ──── 空间↓f_s  时间↓f_t ────►  (t, h, w, c)
                    通常 f_s=8, f_t=4 (估算)

   T 帧 ──时间压缩──► t = T/f_t 个 latent 帧
   H×W ──空间压缩──► h×w = (H/f_s)×(W/f_s)
   3 通道 ────────► c 个 latent 通道 (如 4~16)
```

```
 时间轴 ────────────────►
 帧:  ▏▏▏▏ ▏▏▏▏ ▏▏▏▏ ▏▏▏▏        16 帧
       └┬─┘ └┬─┘ └┬─┘ └┬─┘
        ▼    ▼    ▼    ▼
 latent帧: ■    ■    ■    ■        4 个 latent 帧 (时间↓4)
 每个 latent 帧空间也被 ↓8×↓8
```

**为什么时间也要压？** 视频相邻帧高度冗余（背景几乎不动）。时间压缩既减 token 数，又让模型学到**运动/连续性**而非逐帧静态。VAE 用重建损失 + KL 正则（或 VQ）训练，Decoder 负责把隐变量还原成像素。

---

## 6. Spacetime Patch：视频如何变成 token 序列

DiT 要的是**一维 token 序列**。Patchify 把隐变量 $(t,h,w,c)$ 切成不重叠的**时空小块（spacetime patch）**，每块拉平 + 线性投影成一个 token：

```
 latent (t=4, h=8, w=8)        patch 大小 p_t×p_h×p_w = 1×2×2
 ┌─────────────────┐
 │ □□ □□ □□ □□      │  每个 2×2(空间) × 1(时间) 块 = 1 个 token
 │ □□ □□ □□ □□      │
 │ ...             │  token 数 = (t/p_t)·(h/p_h)·(w/p_w)
 └─────────────────┘            = 4 · 4 · 4 = 64 个 token
        │  flatten + Linear
        ▼
 token 序列: [tok₁, tok₂, ..., tok₆₄]  每个维度 d_model
        + 位置编码(时间/高/宽 三维)
```

**位置编码**要编码「这个 patch 在第几帧、第几行、第几列」——三维位置。可用可学习位置嵌入或 3D 版旋转位置编码（RoPE 的多维推广，见 [[llm-algo/旋转编码RoPE]]），保证模型知道时空相对关系。

到这一步，视频就和一句话一样，成了「带位置的 token 序列」，DiT 用自注意力让**任意两个时空块互相看见**（这正是 Sora 能保持长时空一致性的关键）。

---

## 7. 变长 token：任意分辨率 / 时长 / 比例（NaViT）

传统模型固定分辨率（如必须 256×256）。Sora 借鉴 **NaViT（Patch n' Pack, 2023.07）**：不 resize、不裁剪，**原生**喂入不同尺寸的视频，token 数随之变化。

```
 竖屏短视频  9:16, 5s          横屏长视频 16:9, 20s
  ┌──┐                         ┌────────────┐
  │  │ → 较少 token            │            │ → 很多 token
  │  │                         └────────────┘
  └──┘
        把不同长度的 token 序列「打包(pack)」进同一 batch：
 ┌──tokenA(120)──┬──tokenB(800)──┬─padding─┐
 └───────────────┴───────────────┴─────────┘
   注意力 mask 保证 A、B 互不串扰
```

**为什么重要？**

1. **构图与原生比例**：训练时见过竖屏，推理就能直接出竖屏，不用先方再裁。
2. **可变时长**：同一模型生成 5s 也能生成 1min，token 数线性增长。
3. **训练效率**：Patch n' Pack 把碎片化样本塞满 batch，减少 padding 浪费。

代价：序列越长，注意力 $O(L^2)$ 越贵——这正是 §9 要手算的地方，也是 Sora 长视频成本飙升的根因。

---

## 8. 条件注入：文本 prompt 怎么控制画面（DALL-E 3 caption）

Sora 是**条件**扩散：去噪时必须看着文本 $c$。两个关键点：

**(1) 高质量训练对（DALL-E 3 recaption 方案）**：网络上的视频自带描述往往很烂。Sora 先训一个强 captioner，给海量视频**自动重写详细 caption**，造出大量「精准文本-视频对」。caption 越细，模型学到的「文字→画面」对应越准。推理时，用户的短 prompt 也会被 GPT 先**扩写**成详细描述再喂模型。

```
 用户: "一只猫"
   │  GPT 扩写
   ▼
 "一只橘色短毛猫坐在窗台上，阳光照射，慢慢眨眼，背景是城市天际线..."
   │  文本编码器
   ▼
 条件 c ──► 注入每个 DiT Block (AdaLN / cross-attention)
```

**(2) 条件注入方式**：文本 embedding 通过 **cross-attention**（token 去查文本）或 **AdaLN** 调制注入；时间步 $t$ 同样注入。这样每一步去噪都「带着文字的意图」。

**Classifier-Free Guidance（CFG）**——让画面更贴文本的关键技巧：同一网络既学**有条件** $\varepsilon_\theta(z_t,c)$ 又学**无条件** $\varepsilon_\theta(z_t,\varnothing)$（训练时随机丢弃条件）。推理时外推：

$$\hat\varepsilon = \varepsilon_\theta(z_t,\varnothing) + w\cdot\big(\varepsilon_\theta(z_t,c) - \varepsilon_\theta(z_t,\varnothing)\big)$$

$w$（guidance scale）越大越贴文本但可能失真，通常 $w\in[3,15]$（估算）。

```
 无条件方向 ●─────────► 有条件方向
            \         ↗
             \   ×w 放大「文本-无文本」的差
              ▼
            最终去噪方向（更听话）
```

---

## 9. 数值手算：token 数、序列长度、算力量级

> 全部为**量级演示**，参数自取（标「估算」），目的是建立直觉。

**设定（估算）**：生成 5 秒、512×512、24fps 的视频。

**Step 1：原始像素量**
- 帧数 $T = 5\times24 = 120$ 帧；每帧 $512\times512\times3$。
- 总像素 $=120\times512\times512\times3 \approx 9.4\times10^7$（约 9400 万个数）。

**Step 2：VAE 压缩后隐变量**（取空间 $f_s=8$、时间 $f_t=4$）
- 时间：$t = 120/4 = 30$ 个 latent 帧。
- 空间：$h=w=512/8 = 64$。
- 隐变量形状 $(30, 64, 64, c)$，取 $c=4$，元素数 $=30\times64\times64\times4 \approx 4.9\times10^5$。
- **压缩比** $\approx 9.4\times10^7 / 4.9\times10^5 \approx 190\times$。这就是不在像素上做扩散的理由。

**Step 3：patch → token 数**（patch $p_t\times p_h\times p_w = 1\times2\times2$）

$$L = \frac{t}{p_t}\cdot\frac{h}{p_h}\cdot\frac{w}{p_w} = 30 \cdot 32 \cdot 32 = 30720 \text{ 个 token}$$

对比文本 LLM 的上下文：3 万 token 序列，已和长文本相当。

**Step 4：注意力代价（$O(L^2)$）**
- 注意力打分矩阵 $L\times L = 30720^2 \approx 9.4\times10^8$ 个分数 / 每头 / 每层。
- 若 FP16 存全部注意力矩阵：$9.4\times10^8\times2\text{B} \approx 1.9\text{ GB}$/层/头——所以**必须 FlashAttention**（不落地完整矩阵，见 [[llm-optimizer/FlashAttention]]）。

**Step 5：时长翻倍的代价**
- 时长 5s→10s，token 数 $L$ 翻倍（30720→61440）。
- 注意力计算量 $\propto L^2$ **翻 4 倍**。这解释了「长视频成本超线性暴涨」。

**Step 6：采样步数 × 模型前向**
- 设去噪 $T=50$ 步，每步跑一次完整 DiT 前向。
- 总前向 $=50$ 次（用 DDIM/蒸馏可降到个位数步），生成一条视频 ≈ 50 × 一次 DiT 推理。这也是文生视频「慢」的根源。

```
 时长   token L    注意力∝L²   相对成本
 5s     30720      9.4e8       1×
 10s    61440      3.8e9       4×
 20s    122880     1.5e10      16×
        └─ 超线性！长视频贵在这里
```

---

## 10. 推理：从纯噪声采样出视频（DDPM/DDIM）

推理就是把 §3 的反向过程跑出来，每一步用 DiT 预测噪声、去掉一点：

```
 z_T ~ N(0,I)   (隐空间纯噪声，形状 (t,h,w,c))
   │  for k = T → 1:
   │     ε̂ = CFG( ε_θ(z_k, k, c), ε_θ(z_k, k, ∅) )   # 预测噪声
   │     z_{k-1} = 一步去噪公式(z_k, ε̂, k)            # DDPM/DDIM
   ▼
 z_0           (干净隐变量)
   │  VAE Decoder
   ▼
 视频帧 (T,H,W,3)
```

DDIM 一步去噪（确定性、可少步数）示意：

$$z_{k-1} = \sqrt{\bar\alpha_{k-1}}\,\underbrace{\frac{z_k-\sqrt{1-\bar\alpha_k}\,\hat\varepsilon}{\sqrt{\bar\alpha_k}}}_{\text{预测的 } \hat z_0} + \sqrt{1-\bar\alpha_{k-1}}\,\hat\varepsilon$$

```
 第50步      第30步       第10步       第1步        Decode
 ▒▒▒▒  ──►  ░▒░▒  ──►   ·🐱·  ──►    🐱  ──►   🎬(像素帧)
 纯噪声      隐约成形     大致清楚     干净 z_0     真实视频
```

**工程加速**：步数蒸馏（Consistency/LCM 把 50 步降到 1~4 步）、张量并行切大模型、计算通信重叠等，与 LLM 推理优化一脉相承（见 [[llm-inference/README]]、[[llm-optimizer/计算通信重叠]]）。

---

## 11. Sora 作为「世界模拟器」的额外能力

技术报告强调 Sora 规模化后**涌现**出一些能力：3D 一致性（镜头移动透视正确）、长程一致性（遮挡后再现仍一致）、物体交互（咬痕等因果，不完美）、数字世界模拟（类 Minecraft 画面）、跟随复杂多事件 prompt。

这些被解释为 **scaling law 在视频域的体现**：数据+算力+参数足够大，模型隐式学到时空规律。但它没有显式物理引擎，仍会出现「玻璃没碎水却洒了」之类违反因果的错误——**统计学习而非真物理**。

---

## 常见问题

| 问题 | 回答 |
| --- | --- |
| Sora 和 Stable Diffusion 的根本区别？ | SD 是图像、去噪网络用 U-Net；Sora 是视频、去噪网络用 Transformer（DiT），且在**时空** patch 上做扩散。 |
| 为什么不直接在像素上扩散？ | 太贵。VAE 先压约 100~200×（§9），扩散在隐空间做，省几十倍算力与显存。 |
| token 化和 LLM 的 tokenize 一样吗？ | 思想一样（切块→序列→Transformer），但单位是**时空 patch** 不是子词；位置是三维（时/高/宽）。 |
| 怎么支持竖屏/横屏/任意时长？ | 借 NaViT 的 Patch n' Pack：原生尺寸喂入，token 数变长，注意力 mask 隔离不同样本（§7）。 |
| 为什么生成慢？ | 要跑几十步去噪，每步一次完整 DiT 前向；且长视频 token 数大、注意力 $O(L^2)$ 超线性（§9）。 |
| 文本怎么精确控制画面？ | 训练用 DALL-E 3 式重写的高质量 caption 造对；推理用 cross-attention/AdaLN 注入 + CFG 放大文本方向（§8）。 |
| 和 LLM 工程栈能复用吗？ | 能。FlashAttention、张量并行、KV/计算通信重叠、混合精度都同源可迁移。 |
| 它是真的懂物理吗？ | 不是。是大规模统计学习涌现的近似规律，仍会违反因果（§11），以官方说法「世界模拟器」为隐喻。 |

---

## 🔗 跳转链接

- 知识地图与导航：[[00-知识地图]]
- Transformer 主干（注意力/MLP/位置编码同源）：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/旋转编码RoPE]]
- 算力与 FLOPs 估算方法（迁移到 DiT）：[[llm-algo/FLOPs]]
- 长序列注意力的工程关键：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]]
- 大模型并行与推理（切 DiT、采样加速）：[[llm-inference/大模型推理张量并行]] · [[llm-inference/README]] · [[llm-inference/解码策略]] · [[llm-optimizer/计算通信重叠]]
- 底层通信与硬件：[[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
- 训练与对齐生态：[[llm-train/README]] · [[llm-alignment/RLHF]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- 精度与量化（推理省显存）：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
