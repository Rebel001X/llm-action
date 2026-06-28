# llm-compressor 源码精读

> 不讲"怎么用配方"，而是**钻进代码**：从 `oneshot()` 入口一路追到权重被打包写盘，看清 Modifier / Lifecycle / Pipeline / Hooks 这套状态机在源码层面是怎么咬合运转的。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/llm-compressor/README]] [[llm-compression/llm-compressor/量化方案]] [[llm-compression/quantization/量化基础]] [[llm-inference/vllm/README]]

---

## 阅读地图

| 节 | 你将搞懂 | 源码关键词 |
|----|----------|-----------|
| 0 | 一句话锚点：本文相对 README 的定位 | 内部机制 vs 用法 |
| 1 | 仓库目录结构：每个包负责什么 | `entrypoints` / `modifiers` / `pipelines` |
| 2 | 两库分工：llm-compressor vs compressed-tensors | 算法层 / 格式层 |
| 3 | 入口 `oneshot()` 调用链拆解 | Entrypoint → Session → Lifecycle |
| 4 | 核心状态机：CompressionLifecycle 五个回调 | initialize / on_event / finalize |
| 5 | Modifier 基类与生命周期钩子 | `on_initialize` / `on_finalize` |
| 6 | Hooks：forward hook 如何"截流"激活 | `register_forward_hook` / IntermediatesCache |
| 7 | Pipeline：sequential 如何逐层量化省显存 | trace / subgraph / 偏移执行 |
| 8 | GPTQ Modifier 内部：Hessian 累积与补偿落点 | `accumulate_hessian` / `quantize_weight` |
| 9 | 量化的"挂载"机制：observer + fake-quant + 压缩 | `initialize_module_for_quantization` |
| 10 | 写盘：ModelCompressor 如何打包 int4 | `pack_to_int32` / `compress()` |
| — | 调试入口 + 跳转链接 | 断点放哪 |

> 警示：llm-compressor 迭代极快，**类名/文件名/函数签名以你本地 clone 的源码为准**，本文给的是"机制地图"，不是精确 API。下文凡涉及精确名字处都标了"约/以源码为准"。

---

## 0. 一句话锚点

README 回答"**怎么写配方、产出什么**"；本文回答"**这份配方在代码里是怎么被一步步执行的**"。一句话主线：

```
oneshot(...) ──► Entrypoint 组装参数 ──► 创建 Session/Lifecycle ──►
  Lifecycle.initialize() 让每个 Modifier 在模型上"挂载"(注册 hook / observer) ──►
  Pipeline 喂校准数据触发前向 ──► hook 在前向中收集统计(Hessian/激活幅值) ──►
  Lifecycle.finalize() 让每个 Modifier 把统计→scale/zp→量化→打包 ──►
  ModelCompressor 写出 compressed-tensors
```

把这条链记牢，后面每节都是在给它的某一环放大镜。

---

## 1. 仓库目录结构（src/llmcompressor）

clone 后核心代码在 `src/llmcompressor/` 下，**约**长这样（以你本地为准）：

```
src/llmcompressor/
├── entrypoints/          # 用户入口：oneshot()/train()/Arguments 解析
│     ├── oneshot.py            ← 离线一次性压缩入口
│     ├── train.py              ← 走训练循环(QAT/蒸馏恢复)
│     └── utils.py / args ...   ← 数据集/模型/recipe 参数对象
├── core/                 # 框架骨架：会话与生命周期状态机
│     ├── session.py            ← CompressionSession：一次压缩的总控
│     ├── lifecycle.py          ← CompressionLifecycle：状态机本体
│     ├── state.py              ← State：贯穿全程的"上下文"(model/data/optim)
│     └── events/               ← Event/EventType：训练步事件
├── modifiers/            # 算法层：每个压缩算法一个 Modifier
│     ├── modifier.py           ← Modifier 抽象基类(定义钩子)
│     ├── quantization/
│     │     ├── gptq/           ← GPTQModifier
│     │     └── quantization/   ← QuantizationModifier(RTN/FP8 走这条)
│     ├── awq/                  ← AWQModifier
│     ├── smoothquant/          ← SmoothQuantModifier
│     ├── pruning/              ← SparseGPT / 幅度剪枝 / 2:4
│     └── obcq / ...            ← 其它
├── pipelines/            # 校准数据如何流过模型
│     ├── basic/                ← 整模型一次前向
│     ├── sequential/           ← 逐层(subgraph)前向，省显存
│     └── layer_sequential/     ← 更老的逐层实现
├── observers/            # 统计观测器：从张量算 scale/zero-point
├── utils/                # hook 管理、pytorch 工具、helpers
└── transformers/         # 与 HF transformers 的胶水(save_pretrained 等)
```

> 读源码的"四个钉子"：入口在 `entrypoints/`，骨架在 `core/`，算法在 `modifiers/`，数据流在 `pipelines/`。其余都是辅助。

---

## 2. 两库分工：算法层 vs 格式层

llm-compressor **不**自己定义"什么叫 int4 对称分组量化"，那套数据结构在**另一个库** `compressed-tensors` 里。二者关系：

```
┌────────────────────────────┐        import        ┌──────────────────────────────┐
│       llm-compressor       │ ───────────────────► │      compressed-tensors        │
│  (算法 / 流程 / 状态机)     │   复用其数据结构与     │  (格式 / 量化原语 / 压缩器)     │
│                            │   forward-quant 原语   │                              │
│  · Modifier 决定"怎么算"   │                       │  · QuantizationScheme/Args   │
│  · Pipeline 决定"怎么喂"   │                       │  · initialize_module_for_q   │
│  · Lifecycle 决定"何时做"  │                       │  · fake_quantize / observer  │
│                            │                       │  · ModelCompressor.compress  │
└────────────────────────────┘                       └──────────────────────────────┘
```

关键源码事实（**以源码为准**）：

- `QuantizationScheme` / `QuantizationArgs`：在 compressed-tensors 里定义，`num_bits/type/strategy/symmetric/dynamic/group_size` 这些字段都来自它。
- `initialize_module_for_quantization(module, scheme)`：compressed-tensors 提供，给一个 `nn.Linear` **挂上**量化所需的 buffer（`weight_scale`、`weight_zero_point`、observer）并把它的 forward 包成"伪量化"。
- `ModelCompressor` / 各 `Compressor`：compressed-tensors 提供，负责把训练侧的 fp16+scale 真正**打包**成磁盘格式（如 int4 packed）。

> 直觉：**llm-compressor = 导演**（决定流程与算法），**compressed-tensors = 道具与剧本格式**（量化的数据结构和最终落盘形态）。本文聚焦导演，但会在第 9、10 节点出道具的关键接口。

---

## 3. 入口 `oneshot()` 调用链

从用户那行 `oneshot(...)` 追下去，**约**经历这些层（函数名以源码为准）：

```
oneshot(model=..., dataset=..., recipe=..., ...)            # entrypoints/oneshot.py
   │  1) 把零散 kwargs 收进 Arguments 数据类(ModelArgs/DatasetArgs/RecipeArgs)
   │  2) 加载/校验 model(可 from_pretrained) 与 tokenizer
   │  3) 构建校准 DataLoader(分词、截断到 max_seq_length、采样 N 条)
   ▼
Oneshot 对象 / apply()                                      # 组织一次压缩
   │  4) create_session(): 拿到全局唯一的 CompressionSession
   ▼
session.initialize(model, recipe, calib_data, ...)         # core/session.py
   │  5) 解析 recipe → 一串 Modifier 实例，灌进 Lifecycle
   ▼
CompressionLifecycle.initialize()                          # core/lifecycle.py
   │  6) 逐个 modifier.initialize(state): 在模型上"挂载"
   ▼
Pipeline.run(...)                                          # pipelines/*/
   │  7) 用 calib_data 触发前向，hook 收集统计
   ▼
session.finalize()                                         # core/session.py
   │  8) Lifecycle.finalize() → 逐个 modifier.finalize(): 量化定型
   ▼
model.save_pretrained(save_compressed=True)               # transformers 胶水
      9) ModelCompressor 打包写盘 → compressed-tensors 目录
```

要点：

- **Arguments 数据类**把"用户 API 的随意性"收敛成结构化对象，后面所有模块都消费这些对象，而不是裸 kwargs。
- **Session 是单例式总控**：`create_session()` / `active_session()` 让 Modifier、Pipeline 都能拿到同一个 `state`，不必层层传参。
- 真正"动模型权重"的只有两处：第 6 步（挂载，但不改值）和第 8 步（定型，改值）。中间第 7 步只是**读**（收集统计）。

---

## 4. 核心状态机：CompressionLifecycle

`core/lifecycle.py` 的 `CompressionLifecycle` 是整个库的**心脏**。它把"一次压缩"建模成一个有明确阶段的状态机，对每个 Modifier 广播事件：

```
                     ┌──────────── State(贯穿全程) ────────────┐
                     │  model / optimizer / data / loggers ...  │
                     └──────────────────┬──────────────────────┘
                                        │ (所有 modifier 共享)
        ┌───────────────┐   ┌───────────────────┐   ┌──────────────┐
  ──►   │  initialize   │──►│   on_event(每步)  │──►│   finalize   │  ──►
        └───────┬───────┘   └─────────┬─────────┘   └──────┬───────┘
                │                     │                    │
   对每个 modifier:            训练时每个 batch/step       对每个 modifier:
   modifier.initialize(state)  广播 Event(BATCH_START..)  modifier.finalize(state)
   → 挂 hook / observer        oneshot 下这步很轻          → 算 scale/zp、量化、卸 hook
```

三个核心方法（**名以源码为准**）：

| 方法 | 何时调 | 干了什么 |
|------|--------|---------|
| `initialize()` | 流程开始 | 遍历 modifiers，调各自 `on_initialize(state)`；把 hook/observer 挂上模型；标记 `initialized=True` |
| `event()` / `on_event()` | 训练每步（oneshot 极少用） | 构造 `Event(type=...)` 广播给所有 modifier，让其响应 `BATCH_START/OPTIM_PRE_STEP` 等 |
| `finalize()` | 校准结束 | 遍历 modifiers，调 `on_finalize(state)`：用收集到的统计定型量化，并清理 hook、释放显存 |

> 为什么要状态机而不是一根函数从头跑到尾？因为压缩算法有**跨阶段依赖**：GPTQ 必须"先在前向里攒够 Hessian（initialize+前向），才能在 finalize 里补偿量化"。状态机把"挂载—收集—定型"三段解耦，多个 Modifier 能在同一条前向里**各自收各自的统计**，互不打架。

`State` 对象是另一半灵魂：它是被所有 Modifier、Pipeline 共享的**可变上下文**，装着 `model`、校准 `data`、（训练时的）`optimizer` 等。Modifier 不直接互相通信，而是都读写同一个 `state` —— 这就是 SmoothQuant 改了权重、GPTQ 立刻能看到的原因。

---

## 5. Modifier 基类与生命周期钩子

`modifiers/modifier.py` 的 `Modifier` 抽象基类定义了所有算法必须实现/可选实现的**钩子**（名以源码为准）：

```
class Modifier(...):            # 伪代码骨架
    targets / ignore / scheme ...        # 声明作用域(来自 recipe)

    def on_initialize(self, state) -> bool:   # 必填：挂载阶段
        # 拿到 state.model，找出 targets 命中的子模块，
        # 给它们注册 forward hook / 挂 observer / 初始化量化 buffer
        ...

    def on_start(self, state, event): ...     # 可选：开始收集
    def on_event(self, state, event): ...     # 可选：训练每步响应
    def on_end(self, state, event):   ...     # 可选：停止收集

    def on_finalize(self, state) -> bool:     # 必填：定型阶段
        # 用收集到的统计算 scale/zp，执行量化(改权重)，注销 hook
        ...
```

一个 Modifier 的**完整一生**：

```
 recipe 解析 → 实例化 Modifier
        │
   on_initialize: 在 targets 命中层上 register_forward_hook(收集器)
        │
   ┌────┴───────────── 校准前向(Pipeline 驱动) ──────────────┐
   │ 每次该层被前向调用，hook 触发 → 累积统计到 modifier 内部缓存 │
   └────┬──────────────────────────────────────────────────┘
        │
   on_finalize: 用缓存的统计 → 求 scale/zp → 量化权重 → remove_hook → free
```

> 设计精髓：**算法逻辑全装进 Modifier，框架只负责按状态机节奏喊它**。想加一个新量化算法？写一个新 Modifier，实现 `on_initialize/on_finalize` 即可，Pipeline / Lifecycle / 写盘**完全不用动**——这就是 README 里"算法收敛到一种格式"在代码上的兑现。

---

## 6. Hooks：forward hook 如何"截流"激活

量化要的统计（Hessian、激活幅值）都藏在**层的输入/输出张量**里。怎么不改模型代码就拿到它们？答案是 PyTorch 的 **forward hook**。

llm-compressor 在 `utils/`（约 `HooksMixin` / hook 管理器）里封装了"批量挂/卸 hook"的能力：

```
        正常前向                          挂了 hook 的前向
   x ─► Linear.forward(x) ─► y      x ─► Linear.forward(x) ─► y
                                              │  └──► forward_hook(module, x, y)
                                              │         · 读 x：累积 Hessian Hₗ += xxᵀ
                                              │         · 读 |x|：更新激活 max(SmoothQuant)
                                              ▼
                                         统计落进 modifier 缓存，y 原样放行
```

机制要点：

- `register_forward_hook(fn)` 让 `fn(module, inputs, outputs)` 在每次该层前向**结束后**被自动调用；GPTQ 关心 `inputs`（要 $X$ 算 $H=XX^\top$），SmoothQuant 关心激活幅值。
- **不污染模型**：hook 只读不改（量化算法在 hook 里只攒统计），真正改权重发生在 `on_finalize`，因此前向数值不被破坏，统计才准确。
- **IntermediatesCache（约）**：sequential pipeline 需要把"上一层的输出"喂给"下一层的子图"，库用一个中间结果缓存承接这些张量，并尽量**及时下放/释放**以控显存。
- `on_finalize` 里**务必 remove hook**，否则前向会越来越慢且显存泄漏——读源码时确认每个 Modifier 都成对地挂/卸。

> 一句话：**hook = 在不改模型源码的前提下，给指定层装上"窃听器"收集量化所需统计**。这是整个校准阶段的物理基础。

---

## 7. Pipeline：sequential 如何逐层量化省显存

`pipelines/` 决定"校准数据怎么流过模型"。两种主力（README 讲过用途，这里讲**代码机制**）：

### 7.1 basic —— 整模型一次前向

```
for batch in calib_loader:
    model(**batch)        # 整模型跑完，所有层的 hook 同时攒统计
# 前向全跑完后，finalize 时所有 modifier 一起定型
```
简单，但**所有层的统计同时驻留显存**（每层一个 Hessian，$d\times d$），大模型易 OOM。

### 7.2 sequential —— 按子图逐块，量化完即释放

核心思想：把模型**切成顺序子图**（通常一个 decoder layer 一块），一块跑完就**立刻量化并释放它的统计**，再跑下一块：

```
   ┌─ 用 torch.fx / trace 把模型切成 subgraph: [blk0][blk1]...[blkL]
   │
   │  cache ← 全部校准样本在 embedding 后的初始激活
   ▼
 for blk in [blk0, blk1, ... blkL]:
     for x in cache:                 # 把"上一块的输出"喂进本块
         y = blk.forward(x)          # 触发本块内各层 hook → 攒统计
         new_cache.append(y)         # 记录本块输出，给下一块用
     量化 blk 内权重(用刚攒的统计) ──► 改权重
     cache ← new_cache               # 滚动前进；本块统计可释放
```

ASCII 时间线对比显存峰值：

```
basic:      |■■■■■■■■■■■■|  L 个 Hessian 同时在显存
sequential: |■|→|■|→|■|     同一时刻只 1 个块的 Hessian 在显存
```

**显存账（直觉）**：一层 Hessian 是 $d\times d$ fp32。设 $d=8192$，单个 $H\approx 8192^2\times4\,\text{B}\approx256\,\text{MB}$。模型 80 层：basic 同时需 $80\times256\,\text{MB}\approx20\,\text{GB}$ 仅 Hessian；sequential 任意时刻只 $\approx256\,\text{MB}$。这就是大模型量化必须 sequential 的硬道理。

**误差更小的额外红利**：sequential 量化第 $N$ 块时，喂进去的是**前面已量化块的真实输出**（cache 是滚动更新的），所以后续层会"看到"前面的量化误差并据此校准——量化误差不会层层放大。这与 GPTQ 的逐层补偿哲学一脉相承。

> 代价：要能把模型 **trace/切图**（依赖模型可被 fx 追踪），且实现更复杂。所以库保留了 `basic`（小模型/调试）与 `layer_sequential`（旧式逐层）作退路。

---

## 8. GPTQ Modifier 内部：Hessian 累积与补偿

以 `modifiers/quantization/gptq/` 为例，把第 5 节的抽象钩子落到具体算法（**函数名以源码为准**）：

```
on_initialize(state):
   找到 targets 命中的 Linear → 给每个挂 hook，
   并在 modifier 里为每层开一块 H = 0 (d×d) 的累加器

forward hook(module, inp, out):           # 校准每次前向触发
   X = inp                                 # 该层输入激活, 形状 [tokens, d]
   accumulate_hessian: H += X.T @ X        # 累积二阶信息(约 2/n 归一)

on_finalize(state):                        # 校准结束, 逐层:
   for 每个被量化层:
       quantize_weight(W, H, args):
          1) 对 H 做阻尼: H += dampening_frac * mean(diag(H)) * I   # 防奇异
          2) Cholesky 求 H⁻¹ 的上三角(决定量化顺序与补偿系数)
          3) 逐列量化: 量化第 j 列产生误差 e
                       把 e 按 H⁻¹ 行分摊修正到"尚未量化"的右侧列
          4) 得到量化后的整数权重 + 每 group 的 scale/zp
       remove hook; free H
```

为什么这套能减误差——一行公式直觉：GPTQ 最小化的是该层**输出**误差
$$ \min_{\hat W}\; \lVert XW - X\hat W\rVert_F^2 $$
其解依赖 $H=X^\top X$。量化一个权重后，最优做法是把它造成的输出偏差，用 $H^{-1}$ 给出的方向**补偿**到尚未量化的权重上。

**手算微缩例**：某列量化把 $w_j=0.40$ 舍到 $0.393$，误差 $e=-0.007$。GPTQ 不丢 $e$，按 $H^{-1}$ 耦合系数 $c{=}0.5$ 把 $-c\cdot e{=}{+}0.0035$ 加到右侧未量化列，让它"预先变大"抵消 $e$。RTN 直接扔掉 $e$，GPTQ 把它"接力"传下去——这就是精度差距来源。

`dampening_frac`（阻尼）的代码意义：校准数据不足时 Hessian 可能**接近奇异**、Cholesky 爆 NaN；给对角线加 $\lambda\cdot\bar{H}_{ii}$ 让它正定。太小→数值不稳；太大→补偿失真。

---

## 9. 量化的"挂载"机制：observer + fake-quant

第 5 节说 `on_initialize` 会"挂量化 buffer 和 observer"，这步**实际由 compressed-tensors 的 `initialize_module_for_quantization` 完成**（llm-compressor 调它）。挂载后一个 `Linear` 变成这样：

```
  挂载前:  Linear{ weight }
  挂载后:  Linear{
              weight,
              weight_scale       (buffer, 由 observer 计算)
              weight_zero_point  (buffer, 非对称才用)
              quantization_scheme(记录 num_bits/type/strategy...)
              + forward 被包成"伪量化(fake quant)":
                    w̃ = fake_quantize(weight, scale, zp, args)
                    return F.linear(x, w̃)
           }
```

三个角色：

- **observer**（`observers/`）：吃张量、吐 `scale/zero_point`。例如 min-max observer 看权重的 $[\min,\max]$ 算对称 scale $s=\max(|w|)/q_{max}$。strategy=group 时**每 group 各算一组** scale。
- **fake_quantize**：前向里把权重"量化再反量化"（仍是 fp 张量，但数值已落到量化网格上），用于在校准/QAT 时**模拟量化误差**。注意：oneshot-PTQ 下最终落盘的是真打包 int4，fake-quant 主要服务训练/验证路径。
- **scheme 字段**：记在模块上，写盘时 `ModelCompressor` 据此决定怎么打包、config 里怎么写 `config_groups`。

> 关键区分：**校准阶段权重还是 fp16+scale**（便于继续前向/补偿）；**只有写盘那一刻才真打包成 int4**（第 10 节）。很多人误以为量化"当场就把权重变 int4"，源码告诉你不是——量化是"先定 scale，落盘再压"。

---

## 10. 写盘：ModelCompressor 如何打包 int4

最后一步在 HF 胶水 `model.save_pretrained(save_compressed=True)` 里触发 compressed-tensors 的 `ModelCompressor.compress()`：

```
ModelCompressor.compress(model):
   读每层 quantization_scheme:
      对 W4A16 (format=pack-quantized):
         q = round(weight / weight_scale) (+ zp)      # fp16 → int4 整数
         weight_packed = pack_to_int32(q)             # 8 个 int4 塞进 1 个 int32
         存: weight_packed, weight_scale, [weight_zero_point], weight_shape
      对 W8A8-fp8:
         直接存 fp8 张量 + scale, 无需打包
   写 config.json.quantization_config:
      { quant_method:"compressed-tensors", format:"pack-quantized",
        config_groups:{...每组 scheme...}, ignore:[...] }
```

**打包数值账（int4 → int32）**：4-bit 权重每个占 4 bit，一个 int32 容器 $=32$ bit，恰好塞 $32/4=8$ 个权重。所以一行 $d=4096$ 的权重，打包后 int32 数 $=4096/8=512$ 个。

**体积账（W4A16 省多少）**：原 fp16 权重每个 2 字节；4-bit 权重每个 0.5 字节 + 每 128 个共享 1 个 fp16 scale（$2/128\approx0.016$ 字节/权重）。单权重平均 $\approx0.5+0.016\approx0.516$ 字节 vs fp16 的 2 字节 → **压到约 1/3.9**，即省约 74% 权重显存。group_size 越小，scale 这项越大，压缩比略降——这正是 README 第 6 节"精度 vs 体积"权衡的字节级证据。

```
落盘目录(回扣 README 第6节):
  config.json                       ← quantization_config 块(scheme 真相)
  model-0000X.safetensors           ← weight_packed/weight_scale/...
  tokenizer 等
```

> 至此闭环：llm-compressor 的 Modifier 算出 scale → compressed-tensors 的 Compressor 打包落盘 → vLLM 读 config 选 kernel（Marlin/cutlass-fp8…）原生跑。本文负责把中间"算 scale、打包"这段代码讲透。

---

## 调试入口（断点放哪）

| 想看什么 | 在哪下断点（约，以源码为准） |
|----------|------------------------------|
| 参数怎么被解析 | `entrypoints/oneshot.py` 的 Arguments 组装处 |
| recipe→modifier 列表 | `core/session.py` 的 `initialize()` 解析段 |
| modifier 何时挂载 | 各 Modifier 的 `on_initialize` |
| 统计怎么攒 | 各 Modifier 的 forward hook / `accumulate_hessian` |
| 量化怎么定型 | `on_finalize` / `quantize_weight` |
| 显存为什么爆 | `pipelines/sequential/` 的 cache 释放点 |
| 落盘格式 | compressed-tensors `ModelCompressor.compress` / `pack_to_int32` |

调试技巧：先用**小模型 + basic pipeline + 少量样本**把 `oneshot` 跑通并逐帧 step 进 Lifecycle，看清"initialize→前向 hook→finalize"三段；再换 sequential 对照显存差异，机制就立体了。

---

## 评价 / 对照 / 局限

| 维度 | 评价 |
|------|------|
| 架构优点 | Modifier/Lifecycle/Pipeline 三层正交解耦；加算法只写 Modifier，写盘/数据流复用 |
| 与 compressed-tensors 解耦 | 算法层与格式层分库，格式被 vLLM/transformers 共享，避免重复造轮子 |
| sequential 设计 | 用滚动 cache 把显存峰值从 $O(L)$ 个 Hessian 压到 $O(1)$，是大模型量化的关键 |
| 学习曲线 | 状态机+hook+fx-trace 叠加，初读门槛偏高；需先建立"挂载/收集/定型"心智模型 |
| 局限①：可追踪性 | sequential 依赖模型可被 trace/切图，特殊结构可能 fallback 到 basic |
| 局限②：迭代快 | 类名/路径常变动，**精确 API 务必对本地源码核验**；本文只画机制地图，非逐行注释 |

---

## 🔗 跳转链接

- [[00-知识地图]] —— 回到压缩专题总图
- [[llm-compression/llm-compressor/README]] —— 用法/配方/scheme 全貌（本文的"姊妹篇"）
- [[llm-compression/llm-compressor/量化方案]] —— GPTQ/AWQ/SmoothQuant/FP8 各 scheme 细节与字段
- [[llm-compression/quantization/量化基础]] —— scale/zero-point/对称/分组的第一性原理
- [[llm-inference/vllm/README]] —— 下游消费者：vLLM 如何加载 compressed-tensors 并选 kernel

**官方参考（以最新源码为准）**：算法层 `vllm-project/llm-compressor`（`src/llmcompressor/{entrypoints,core,modifiers,pipelines}`）；格式层 `vllm-project/compressed-tensors`（`QuantizationScheme`、`initialize_module_for_quantization`、`ModelCompressor`）。
