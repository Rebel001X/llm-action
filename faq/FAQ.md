# 大模型训练/微调常见报错排查 FAQ

> 一句话定位：把训练大模型时最高频的「框架/硬件/数值精度」报错按根因归类成可查表，给出原始报错、根因原理与一行修复。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] · [[llm-train/peft/PEFT-API]] · [[llm-compression/quantization/量化基础]] · [[ai-infra/ai-hardware/硬件对比]]

## 阅读地图

| 你遇到的现象 | 根因层 | 跳到 |
|---|---|---|
| `BitsAndBytesConfig object is not subscriptable` / `BaichuanTokenizer has no sp_model` | 库版本不兼容 | [§2 版本兼容](#2-第一类报因库版本不兼容api漂移) |
| `Parameter object has no attribute ds_status` / `Cannot copy out of meta tensor` | ZeRO-3 参数分片/惰性初始化 | [§3 ZeRO-3 与 meta tensor](#3-第二类报因zero-3-参数分片与-meta-tensor) |
| `Unsupported gpu architecture 'compute_89'` | CUDA 编译器不识别新算力 | [§4 GPU 算力与 CUDA 编译](#4-第三类报因gpu-算力sm-与-cuda-编译不匹配) |
| `cublasLt ran into an error` / `named symbol not found` | int8/低比特内核不支持该 GPU | [§5 低比特内核与 GPU 代际](#5-第四类报因低比特内核与-gpu-代际) |
| `element 0 of tensors does not require grad` | LoRA 冻结主干后梯度链断了 | [§6 PEFT 梯度链](#6-第五类报因peft-梯度链断裂) |
| `DataLoader worker killed by signal: Killed` | 共享内存/子进程 OOM | [§7 DataLoader 被杀](#7-第六类报因dataloader-worker-被信号杀死) |

## 0. 一句话锚点

大模型微调报错 90% 不在「你的代码」，而在 **库版本 × CUDA 版本 × GPU 代际 × 分布式策略** 这四者的笛卡尔积里某个组合不兼容。会查根因层，比记住每条报错更重要。

```
        ┌──────────────────────────────────────────────┐
        │              一次微调调用栈                    │
        ├──────────────────────────────────────────────┤
   你写的│  train.py (Trainer / 自定义 loop)            │ ← §6 梯度链
        ├──────────────────────────────────────────────┤
   库    │  transformers / peft / bitsandbytes          │ ← §2 版本漂移 §5 低比特
        ├──────────────────────────────────────────────┤
   框架  │  DeepSpeed ZeRO / DataLoader                 │ ← §3 分片 §7 子进程
        ├──────────────────────────────────────────────┤
   底座  │  PyTorch → CUDA Runtime → cuBLAS/cuBLASLt    │ ← §4 编译 §5 内核
        ├──────────────────────────────────────────────┤
   硬件  │  GPU (sm_75 / sm_80 / sm_89 / sm_90)         │ ← Turing/Ampere/Ada/Hopper
        └──────────────────────────────────────────────┘
报错冒泡：底层根因，往往在上层抛出一条看不懂的异常。
```

## 1. 地基：四个必须先搞清的坐标

排查任何报错前，先把环境的四个坐标量出来——它们决定了哪些组合是合法的。

| 坐标 | 怎么查 | 为什么决定成败 |
|---|---|---|
| GPU 算力 SM | `nvidia-smi`；查表 4090=8.9 | CUDA 编译器须显式支持该 `compute_XX` |
| CUDA 版本 | `nvcc --version` | 旧 CUDA 不认识新 GPU 的 SM |
| PyTorch 版本 | `python -c "import torch;print(torch.__version__, torch.version.cuda)"` | torch 自带一套 CUDA，须与驱动匹配 |
| 库版本 | `pip show transformers peft bitsandbytes` | 模型代码常绑死某段 API |

GPU 代际 ↔ 算力 ↔ 典型卡，背下这张表能秒判一半报错：

| 架构 | 算力 SM / `compute_XX` | 典型卡 | 低比特格式支持 |
|---|---|---|---|
| Turing | 7.5 / `compute_75` | T4、2080Ti | int8 (Turing 格式) |
| Ampere | 8.0 / 8.6 | A100、3090、A800 | int8 (Ampere 格式) |
| Ada | 8.9 / `compute_89` | 4090、L40 | 需 CUDA ≥ 11.8 |
| Hopper | 9.0 / `compute_90` | H100、H800 | int8 **当时不支持**（§5） |

> 关键认知：cuBLASLt 的 8-bit 矩阵乘对 **Ampere / Turing / Hopper 各有一套专用数据格式**，互不通用。这就是为什么换张更新的卡，int8 反而跑不起来。

## 2. 第一类报因：库版本不兼容 / API 漂移

模型仓库（baichuan2）的建模代码常常贴着某个 `transformers` 版本写的，库一升级，私有属性或 API 形态变了就崩。

**报错 A：`'BitsAndBytesConfig' object is not subscriptable`**
- 出处：baichuan2 加载量化配置时。官方讨论：`https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/discussions/2`
- 原理：`subscriptable` 指支持 `obj[key]` 下标取值。新版 `BitsAndBytesConfig` 是个普通对象不是 dict，旧建模代码用 `config['xxx']` 取值就报这个。属于「库给的对象变了形态」。

**报错 B：`AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'`**
- 修复（原文真料）：降版本
  ```bash
  pip install transformers==4.34.0
  ```
- 原理：`sp_model` 是 SentencePiece 分词器对象，baichuan 的 tokenizer 在旧版 transformers 的初始化顺序里会先建 `sp_model` 再用它；新版 transformers 改了 `PreTrainedTokenizer.__init__` 的调用时序，导致父类初始化时 `sp_model` 还没建好就被访问。**钉死兼容版本**是最稳的解。

```
旧 transformers: __init__ → 先建 self.sp_model → 父类用它  ✅
新 transformers: __init__ → 父类先跑 → 访问 self.sp_model → 还没建 ✗ AttributeError
```

> 坑：遇到 `has no attribute` 类报错，先别改模型代码，**先查这条建模代码是哪个 transformers 版本写的**，对齐版本往往一行解决。

## 3. 第二类报因：ZeRO-3 参数分片与 meta tensor

DeepSpeed ZeRO-3 会把每个参数张量**切片分散到各 GPU**，本地只持有一小片，并给参数挂上 `ds_status` 等元数据；初始化阶段还用 PyTorch 的 `meta` 设备造「只有形状没有数据」的占位张量来省显存。两类报错都源于此。

```
ZeRO-3 下一个 Linear 权重 W (假设 4 卡)：
┌────────── 完整 W [out, in] ──────────┐
│  GPU0 持有  │ GPU1 │ GPU2 │ GPU3 持有 │   每卡只存 1/4
└─────────────┴──────┴──────┴──────────┘
DeepSpeed 给每片挂 ds_status / ds_shape 等属性；
forward 前 all-gather 拼回完整 W，用完再切碎释放。
```

**报错 C：`AttributeError: 'Parameter' object has no attribute 'ds_status'`**
- 场景（原文真料）：`deepspeed + transformers 全量微调`，ZeRO-3，**eval 时**报错。issue：`https://github.com/baichuan-inc/Baichuan2/issues/215`
- 根因：某段代码在参数**还没被 ZeRO-3 接管**（没挂上 `ds_status`）就当成已分片参数去访问其 `ds_status`。常见于 eval/generate 路径绕过了 DeepSpeed 包装的 forward，导致 gather 逻辑找不到分片元数据。

**报错 D：`NotImplementedError: Cannot copy out of meta tensor; no data!`（chatglm3）**
- issue：`https://github.com/THUDM/ChatGLM-6B/issues/530`
- 根因原理：`AutoModel.from_pretrained()` 默认走「先用 meta 设备建空壳（省显存）→ 再 load 权重」的惰性初始化。但 chatglm 的某些 `nn.Parameter` 在 meta 设备上**只有形状、没有真实数据**，DeepSpeed 想把权重复制进去时，源端是 meta tensor（无数据），`copy_` 无法从「没有数据的张量」拷出，于是抛 `Cannot copy out of meta tensor`。
- 修复：加载时传 `empty_init=False`，让 `nn.Parameter` 直接被初始化为**含真实权重的张量**，而不是空的 meta 占位，DeepSpeed 才能正常拷贝。

```
empty_init=True (默认): Param 在 meta 设备 → 只有 shape → DeepSpeed copy_ 失败 ✗
empty_init=False:       Param 直接落实数据 → DeepSpeed 能 copy_ → 正常分片 ✅
```

> 为什么别的模型不用这个参数：它们的初始化/加载路径不依赖 meta 占位，或上游代码已为 DeepSpeed 改造过。这是**模型代码 × 框架初始化策略**的耦合坑，不是普适 bug。

## 4. 第三类报因：GPU 算力 SM 与 CUDA 编译不匹配

DeepSpeed 的 CPU-Adam、融合算子等需要现场用 `nvcc` 编译 CUDA 扩展；`nvcc` 必须认识目标 GPU 的 `compute_XX`，否则编不出来。

**报错 E：`Unsupported gpu architecture 'compute_89'`**
- 场景（原文真料）：`deepspeed zero3 offload` 在 **RTX 4090**（算力 8.9 → `compute_89`）。issue：`https://github.com/microsoft/DeepSpeed/issues/3488`
- 根因：当前 CUDA 编译器版本里没有 `compute_89` 这张「目标架构」，老 nvcc 不认识 Ada。

两条修复路线（原文真料，来自 issue 与 `https://blog.csdn.net/rellvera/article/details/130337185`）：

```
路线①（推荐）升级 CUDA：4090 是 compute_89，把 CUDA 升到 11.8 以上即可识别
路线②（绕过）   降算力目标：编译命令里 -arch=compute_75 之类，按更老架构编
```

还有一条针对 offload 的「避坑」配置——干脆不编译那个 fused CPU-Adam，改用 PyTorch 原生 Adam：

```yaml
# ds_config 里
torch_adam: true
# 注释（作者原话）：The torch.optim.Adam works fine for cpu offloading.
```

| 现象 | 真正根因 | 最稳解 |
|---|---|---|
| `Unsupported gpu architecture 'compute_89'` | nvcc 太老不认 Ada | CUDA 升到 ≥ 11.8 |
| 不想动 CUDA、只要 offload 能跑 | fused CPU-Adam 编译失败 | `torch_adam: true` |
| 实在改不了环境 | 编译目标过新 | `-arch=compute_75` 降目标 |

## 5. 第四类报因：低比特内核与 GPU 代际

**报错 F：`Exception: cublasLt ran into an error!`**
- 场景（原文真料）：**H100 上用 `LLM.int8()` 加载模型微调，当时不支持**。issue：`https://github.com/TimDettmers/bitsandbytes/issues/538`
- 作者（TimDettmers, 2023-11-02）原话要点：8-bit 实现用 cuBLASLt，其 8-bit 矩阵乘对 **Ampere / Turing / Hopper 各有专用格式**；Hopper **不支持 Ampere/Turing 格式**，需要为 Hopper 单独实现 CUDA kernel 与 cuBLASLt 集成才能让 int8 工作。当时更现实的做法是直接抛错告知用户「暂不支持」。

```
cuBLASLt 8-bit 专用数据布局（互不通用）：
  Turing(sm75) ─┐
  Ampere(sm80) ─┼─→ 各一套 kernel
  Hopper(sm90) ─┘  ← 缺 kernel → cublasLt ran into an error
```

**报错 G：`Error named symbol not found at line 74 in file /bitsandbytes/csrc/ops.cu`**
- 场景（原文真料）：**H800** 上用 **int8 LoRA / QLoRA** 训练都报错。
- 根因与修复：`named symbol not found` 是 CUDA 运行时在 GPU 二进制里找不到对应架构编译出来的符号——bitsandbytes 预编译的 `.so` 没含 H800（Hopper）所需的内核符号。原文修复：**H800 支持 CUDA 11.8 以上，但这里需把 CUDA 升级到 12 以上**，让 bitsandbytes 用支持 Hopper 的工具链重编/匹配。

| 报错 | 卡 | 操作 | 处置 |
|---|---|---|---|
| `cublasLt ran into an error` | H100 | `LLM.int8()` 加载微调 | 当时不支持，换 fp16/bf16 或等内核支持 |
| `named symbol not found ... ops.cu` | H800 | int8 LoRA / QLoRA | CUDA 升级到 12+ |

> 一句话规律：**越新的卡（Hopper），低比特生态越可能滞后**——量化内核要逐架构适配，bitsandbytes 跟进有延迟。生产上在 H 系列卡跑量化前务必先小批验证。

## 6. 第五类报因：PEFT 梯度链断裂

**报错 H：`element 0 of tensors does not require grad and does not have a grad_fn`**
- 场景（原文真料）：用 **LoRA（FP16 加载，而非 `LLM.int8()` 加载）微调时**报错。issue：`https://github.com/huggingface/peft/issues/137`
- 根因：LoRA 把**主干权重全部冻结**（`requires_grad=False`），只训练插入的低秩适配器。但若输入嵌入这一端没有任何张量参与梯度，反向传播时计算图第一个张量就「不需要梯度、也没有 `grad_fn`」，autograd 无从回传 → 报这条。

```
正常：input_embed(可回传) → 冻结主干 → LoRA适配器(requires_grad) → loss
                                            ↑ 梯度能到这停下，OK
断链：input_embed(无grad) → 全程冻结 → ... → loss
       ↑ 计算图起点就没 grad_fn → element 0 ... does not require grad ✗
```

两种修复（原文真料，二选一）：

```python
# 方案① 启用输入嵌入的梯度，给计算图一个可回传的起点
model.enable_input_require_grads()
# 在保持主干权重冻结的同时，让适配器权重能被微调

# 方案② 加载已有 PEFT 时显式声明可训练
PeftModel.from_pretrained(model, peft_model_id, is_trainable=True).to(device)
```

- 源码参考（原文真料）：
  - `transformers/modeling_utils.py#L1559`：`https://github.com/huggingface/transformers/blob/c9e3c0b45419804e11885120e25a35803d1fcf44/src/transformers/modeling_utils.py#L1559`
  - `peft/peft_model.py#L284`：`https://github.com/huggingface/peft/blob/6008f272a565f56c146c5d9fd78d00cb24392d7b/src/peft/peft_model.py#L284`

> 对照：`LLM.int8()` 加载时一般不犯这条，因为量化加载路径里通常已默认开启了输入梯度；纯 FP16 加载没人帮你开，得手动 `enable_input_require_grads()`。

## 7. 第六类报因：DataLoader worker 被信号杀死

**报错 I：`RuntimeError: DataLoader worker (pid xxxxx) is killed by signal: Killed.`**
- 参考：`https://blog.csdn.net/wjinjie/article/details/129733252` · `https://cloud.tencent.com/developer/article/2066826`

根因有二，按概率排查：

**(1) 共享内存太小**——多 worker 通过 `/dev/shm` 传 batch，容器默认 `/dev/shm` 仅 64MB，大 batch 一塞就 OOM 被内核 `Killed`。修复（原文真料，docker 启动时）：
```bash
--shm-size 4G
```

**(2) `num_workers` 设太大**——子进程数超出机器内存/CPU 承载（原文真料）：
```
num_workers：DataLoader 用多少子进程加载数据，越大越快但越吃内存/CPU。
默认 0 = 在主进程内加载（不开子进程）。

最简单：把 num_workers 调小一点；
（最终解决）如果还有问题，直接设成默认值 0；
也可以加机器内存来缓解。
```

```
num_workers=4：  主进程 ┬─ worker0 ─┐
                        ├─ worker1 ─┤ 经 /dev/shm 把 batch 传回主进程
                        ├─ worker2 ─┤ shm 太小 → 内核 OOM Killer → signal: Killed
                        └─ worker3 ─┘
num_workers=0：  主进程自己加载，不走 shm，最稳但最慢
```

| 现象 | 根因 | 处置 |
|---|---|---|
| worker killed，且在容器里 | `/dev/shm` 默认仅 64MB | `--shm-size 4G` |
| worker killed，本机/内存吃紧 | `num_workers` 过大 | 调小，极端情况设 `0` |

## 常见问题 / 坑速查

| 报错关键字 | 根因层 | 一行处置 |
|---|---|---|
| `BitsAndBytesConfig not subscriptable` | 库 API 漂移 | 对齐 baichuan2 要求的 transformers 版本 |
| `BaichuanTokenizer ... no sp_model` | 版本不兼容 | `pip install transformers==4.34.0` |
| `Parameter ... no attribute ds_status` | ZeRO-3 未接管参数 | 检查 eval/generate 是否绕过 DS forward |
| `Cannot copy out of meta tensor` | meta 惰性初始化 | `from_pretrained(..., empty_init=False)` |
| `Unsupported gpu architecture 'compute_89'` | nvcc 太老 | CUDA 升 ≥11.8，或 `torch_adam: true` |
| `cublasLt ran into an error` (H100) | Hopper 缺 int8 内核 | 暂不支持，改 bf16/fp16 |
| `named symbol not found ops.cu` (H800) | 预编译缺 Hopper 符号 | CUDA 升级到 12+ |
| `element 0 ... not require grad` | LoRA 梯度链断 | `model.enable_input_require_grads()` |
| `DataLoader worker killed: Killed` | shm/子进程 OOM | `--shm-size 4G` 或 `num_workers=0` |

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 微调/对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 量化/压缩：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 硬件/算力：[[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/算力/昇腾NPU]]
- 推理/评测：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 内存估算：[[docs/transformer内存估算]]
