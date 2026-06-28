# PyTorch on Ascend(torch_npu / Ascend Extension for PyTorch)

> 让原生 PyTorch 代码"几乎不改"地跑在昇腾 NPU 上的适配层:把 `cuda` 换成 `npu`,底层走 CANN 算子。📍 导航:[[00-知识地图]]
> 🔗 相关:[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-framework/huggingface-transformers/README]] [[ai-framework/megatron-lm/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|-------------|--------|
| 0 | 一句话锚点 | torch_npu = PyTorch 的昇腾后端 |
| 1 | 在昇腾软件栈的定位 + 昇腾↔英伟达对照表 | PrivateUse1、device='npu' |
| 2 | 整体架构:PyTorch ↔ torch_npu ↔ CANN | 插件式后端 |
| 3 | 执行机制:单算子下发 vs 图模式(GE/TorchAir) | Eager / Graph |
| 4 | 分布式:HCCL 后端怎么接进 DDP/Megatron | ProcessGroupHCCL |
| 5 | AMP 混合精度与 dtype 注意点 | bf16/fp16 |
| 6 | 典型工作流(训练/推理)流程含义 | 装环境→改代码→跑 |
| 迁移 | 从 CUDA 迁到昇腾要改什么、常见坑 | `.cuda()`→`.npu()` |
| FAQ | 高频疑问速查 | — |
| 链接 | 枢纽双链 | — |

## 0. 一句话锚点

`torch_npu`(官方名 **Ascend Extension for PyTorch**,昇腾 PyTorch 适配插件)是一个 **PyTorch 的设备后端插件**。它不替换 PyTorch,而是通过 PyTorch 的 **PrivateUse1** 扩展机制注册一个新设备类型 `npu`,把张量运算转发到 **CANN**(昇腾计算架构)上的算子去执行。

对你来说,迁移心智就一句话:**把代码里的 `cuda` 改成 `npu`,把 `nccl` 改成 `hccl`,其余 PyTorch 写法基本不动**。

```python
import torch
import torch_npu   # 关键:import 这一行才会把 npu 设备注册进 PyTorch

x = torch.randn(4, 4).npu()      # 对标 .cuda()
y = torch.randn(4, 4).to('npu')  # 对标 .to('cuda')
z = x @ y                         # 矩阵乘在 NPU 上由 CANN 算子执行
```

> 注意:**必须显式 `import torch_npu`**。不像 CUDA 是 PyTorch 自带后端,昇腾后端是外挂插件,不 import 就没有 `npu` 设备。这是新手第一坑。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层(自底向上)

```
┌─────────────────────────────────────────────────────────────┐
│  套件层   MindFormers / ModelLink / MindIE / msmodelslim ... │
├─────────────────────────────────────────────────────────────┤
│  框架层   PyTorch + torch_npu  │  MindSpore  │  TensorFlow    │  ← 本文在这里
├─────────────────────────────────────────────────────────────┤
│  CANN     图引擎 GE / 算子库 (AOL) / HCCL / Runtime / Driver   │
├─────────────────────────────────────────────────────────────┤
│  硬件层   昇腾 NPU(达芬奇架构:Cube / Vector / Scalar 单元)   │
└─────────────────────────────────────────────────────────────┘
```

`torch_npu` 处在 **框架层**:它向上承接 PyTorch 的算子调用,向下调用 CANN 的 Runtime / 算子库 / HCCL。它本身**不写算子内核**——内核由 CANN 提供;它做的是"翻译与调度":把 PyTorch 的 ATen 算子映射到 CANN 算子、管理 NPU 内存与流(Stream)、桥接分布式通信。

### 1.2 昇腾 ↔ 英伟达生态对照表(迁移心智图)

| 层次/能力 | 英伟达(CUDA 世界) | 昇腾(Ascend 世界) | 说明 |
|-----------|-------------------|---------------------|------|
| 加速硬件 | GPU | NPU(昇腾,达芬奇架构) | 计算芯片 |
| 底层软件栈 | CUDA Toolkit | **CANN** | 编程/运行时基座 |
| 设备字符串 | `'cuda'` / `'cuda:0'` | `'npu'` / `'npu:0'` | PyTorch device |
| 张量搬运 API | `.cuda()` / `.to('cuda')` | `.npu()` / `.to('npu')` | 用法对称 |
| 框架后端来源 | PyTorch **内置** | **外挂插件** `torch_npu`(需 import) | 关键差异 |
| 算子库 | cuDNN / cuBLAS | CANN 算子库(AOL/AscendCL) | 高性能内核 |
| 集合通信 | **NCCL** | **HCCL** | AllReduce 等原语 |
| PyTorch 通信后端名 | `backend='nccl'` | `backend='hccl'` | dist.init |
| 图编译/优化 | TorchInductor / TensorRT | **GE 图引擎 + TorchAir** | 图模式加速 |
| 性能分析 | Nsight / `torch.profiler` | CANN Profiling + `torch_npu` profiler | 调优工具 |
| 显存/内存 | VRAM(GPU 显存) | NPU 内存(HBM) | 容量受限要省 |
| 设备可见性环境变量 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` | 卡选择 |
| 训练大模型套件 | Megatron-LM / DeepSpeed | ModelLink / MindFormers | 在上层套件 |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 在上层套件 |

> 一句话:**CUDA↔CANN、NCCL↔HCCL、cuDNN/cuBLAS↔CANN算子库、cuda↔npu**。记住这四组对应,80% 的迁移直觉就有了。

## 2. 整体架构:PyTorch ↔ torch_npu ↔ CANN

PyTorch 的可扩展点叫 **PrivateUse1**:框架预留了一个"匿名第三方设备槽",允许厂商把自家加速器注册进来,而不必改 PyTorch 主干。`torch_npu` 就是用这个槽把 NPU 注册成 `npu` 设备。

```
   用户 PyTorch 代码 (model.npu(), loss.backward(), optimizer.step())
              │  调用 ATen 算子(如 aten::matmul, aten::add)
              ▼
   ┌───────────────── torch_npu 适配层 ─────────────────┐
   │  • 算子分发:ATen 算子 → CANN 算子(Op Adapter)     │
   │  • 内存管理:NPU 内存池 / 缓存分配器                 │
   │  • 流与事件:NPU Stream / Event(对标 CUDA Stream)  │
   │  • 自定义算子:torch_npu.npu_xxx(如融合算子)        │
   │  • 分布式:ProcessGroupHCCL(把 HCCL 接入 c10d)     │
   └───────────────────────┬────────────────────────────┘
              │  AscendCL / Runtime 调用
              ▼
   ┌──────────────────── CANN ──────────────────────────┐
   │  GE 图引擎 │ 算子库(AOL) │ HCCL │ Runtime │ Driver  │
   └───────────────────────┬────────────────────────────┘
              ▼
        昇腾 NPU 硬件(Cube/Vector/Scalar 计算)
```

要点:
- **算子分发(Op Adapter)**:绝大多数常见算子已被适配。少数算子若未适配,会"回退"到 CPU 执行(性能塌方),或直接报"算子不支持"——这是大模型迁移的高频坑(见迁移节)。
- **缓存分配器**:类似 CUDA caching allocator,复用 NPU 内存块、减少频繁申请/释放的开销。OOM 时同样有"已保留但未使用"的碎片问题。
- **`torch_npu` 自带融合算子**:形如 `torch_npu.npu_rms_norm`、`torch_npu.npu_fusion_attention` 等(具体算子名与签名以官方文档为准),对标 FlashAttention 这类手写融合内核,是性能关键。

## 3. 执行机制:单算子下发(Eager)vs 图模式(Graph)

和 GPU 一样,NPU 也有两种执行范式:

```
单算子(Eager)模式            图(Graph)模式
─────────────────            ───────────────
逐个算子下发到 NPU            先把计算图整体编译(GE/TorchAir),
执行,Host 与 Device          再整图下发执行
来回交互多                    Host 下发开销摊薄、算子可被融合优化
↑ 调试方便、灵活              ↑ 吞吐高、适合稳定的大模型训练/推理
↓ Host 调度可能成瓶颈         ↓ 编译耗时、动态 shape 支持受限
```

- **单算子模式**:默认、最接近原生 PyTorch 体验,逐算子下发。开发调试期首选。
- **图模式**:通过 **GE(Graph Engine)** 把子图编译成可在 NPU 上整体执行的图,做算子融合、内存复用等优化。PyTorch 侧的接入路径常被称为 **TorchAir**(把 `torch.compile` 的图导出给 GE)。
- 经验法则:**先用单算子模式跑通正确性,再视性能需求切图模式**。图模式遇到动态 shape、控制流复杂的模型时,可能需要额外处理(具体开关与限制以官方文档为准)。

## 4. 分布式训练:HCCL 怎么接进 PyTorch

多卡/多机训练时,PyTorch 用 `torch.distributed`(c10d)做集合通信。GPU 走 NCCL,NPU 走 **HCCL**(华为集合通信库)。

```python
import torch, torch_npu
import torch.distributed as dist

dist.init_process_group(backend='hccl')   # 对标 backend='nccl'
torch.npu.set_device(local_rank)          # 对标 torch.cuda.set_device
# 之后 DDP / FSDP / Megatron 的 AllReduce、AllGather 等
# 都由 ProcessGroupHCCL 落到 HCCL 上执行
```

- `torch_npu` 提供 **ProcessGroupHCCL**,把 HCCL 的 AllReduce / AllGather / ReduceScatter / Broadcast 等原语接入 c10d,所以 **DDP 几乎零改动**。
- 多机时还需要正确的 **网络与 rank table / 通信配置**(HCCL 依赖底层 RoCE 网络与设备拓扑);这些**具体配置文件格式、环境变量与命令以华为昇腾官方文档(Ascend 社区)为准**。
- 上层的 **ModelLink / MindFormers** 把张量并行(TP)、流水并行(PP)、序列并行等封装好了,通信底座仍是 HCCL。

## 5. 混合精度(AMP)与 dtype

- NPU 上同样支持 **混合精度**。昇腾对 **bf16** 友好,大模型训练常用 bf16,数值范围大、稳定性好。
- PyTorch 的 `torch.npu.amp`(自动混合精度,对标 `torch.cuda.amp`)提供 autocast 与 GradScaler 的 NPU 版本。
- 坑:某些算子在特定 dtype 下未适配或精度退化;遇到 NaN/精度对不齐时,**优先排查 dtype 与 autocast 覆盖范围**,必要时对个别算子强制 fp32。

## 6. 典型工作流(只讲流程含义,不写具体命令/版本)

```
① 装驱动+固件          → 让操作系统认识 NPU 硬件
② 装 CANN(Toolkit)    → 提供 Runtime/算子库/HCCL 等基座(对标装 CUDA)
③ 装匹配版 PyTorch     → 主框架
④ 装匹配版 torch_npu   → 昇腾后端插件(版本必须与 PyTorch、CANN 三方对齐!)
⑤ source 环境变量      → 让 PyTorch 能找到 CANN 的库与算子
⑥ 改代码:cuda→npu、nccl→hccl、import torch_npu
⑦ 跑训练/推理 → 用 ASCEND_RT_VISIBLE_DEVICES 选卡
```

**版本三方对齐是头号大坑**:`PyTorch 版本 ↔ torch_npu 版本 ↔ CANN 版本` 必须按官方给出的配套关系表匹配,错配会出现"import 失败 / 算子缺失 / 段错误"。

> 所有**具体命令、包名、版本号、镜像、环境变量取值,以华为昇腾官方文档(Ascend 社区)与配套关系表为准**;本文只解释每步"为什么要做、做错会怎样"。

### 6.1 训练参考(本仓库已有线索)

- MindFormers LLaMA 模型卡:`gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md`
- ModelZoo-PyTorch(昇腾官方模型库,含 LLaMA-13B / Qwen-7B 等已适配脚本):
  - `gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/foundation/LLaMA-13B`
  - `gitee.com/ascend/ModelZoo-PyTorch/.../Qwen-7B/test/qwen_7B_64p.sh`(64 卡 Qwen-7B 训练脚本示例)

启动示例(选卡环境变量已是昇腾写法,概念对标 `CUDA_VISIBLE_DEVICES`):

```bash
# 概念示例,具体变量/脚本以官方为准
ASCEND_RT_VISIBLE_DEVICES=6 python train_bloom_lora.py
```

### 6.2 baichuan2 LoRA 适配目标模块(示例)

在昇腾上对 baichuan2 做 LoRA 微调时,注入低秩矩阵的线性层与 GPU 上一致(框架无关,取决于模型结构):

```python
target_modules = ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

> `W_pack` 是 baichuan 把 Q/K/V 打包成一个线性层的命名;`o_proj` 为注意力输出投影;`gate/up/down_proj` 为 MLP(SwiGLU)三件套。LoRA 注入位置在 NPU 与 GPU 上没有区别。

## 迁移要点 / 注意事项与坑

**最小改动清单(CUDA → 昇腾)**

| 原 CUDA 写法 | 改成昇腾 | 备注 |
|-------------|---------|------|
| (无) | `import torch_npu` | 必须加,且在 `import torch` 之后 |
| `.cuda()` / `.to('cuda')` | `.npu()` / `.to('npu')` | 张量/模型搬运 |
| `device='cuda'` | `device='npu'` | 包括 `torch.device('npu')` |
| `torch.cuda.set_device` | `torch.npu.set_device` | 选卡 |
| `backend='nccl'` | `backend='hccl'` | 分布式后端 |
| `torch.cuda.amp` | `torch.npu.amp` | 混合精度 |
| `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` | 选卡环境变量 |
| `torch.cuda.synchronize` | `torch.npu.synchronize` | 同步 |

**高频坑**

1. **忘记 `import torch_npu`** → `npu` 设备不存在/报错。第一坑。
2. **版本三方错配**(PyTorch↔torch_npu↔CANN)→ import 失败、算子缺失、段错误。务必查配套关系表。
3. **算子未适配 / 回退 CPU** → 大模型里某个新算子在 NPU 上没适配,要么报"算子不支持",要么静默回退 CPU 导致**性能断崖**。排查:看是否有 host-device 频繁拷贝、用 profiling 看算子耗时分布。
4. **环境变量没 source** → 找不到 CANN 库,典型 `libascendcl.so not found` 一类错误。
5. **未替换 FlashAttention 等手写 CUDA 内核** → 直接用 GPU 上的 `flash_attn` 包会失败;昇腾要换成 `torch_npu` 提供的融合注意力算子(如 `npu_fusion_attention`,名称以官方为准)或上层套件已封装的实现。
6. **第三方库里隐藏的 `.cuda()` / `'cuda'` 硬编码**(如 HF Transformers 某些路径、自定义 Trainer)→ 用关键字搜 `cuda`、`nccl`、`flash_attn`、`bitsandbytes` 逐一替换或打补丁。
7. **bitsandbytes/AWQ-CUDA 等依赖 CUDA 的量化库不可直接用** → 昇腾走 msmodelslim + MindIE 路线。
8. **图模式遇动态 shape** → 序列长度变化、padding 策略会触发反复编译;评估前先确认 shape 稳定性。

**性能调优思路(机制层)**

- 优先用 **图模式 + 融合算子** 降低 host 下发开销。
- 用 **CANN Profiling / `torch_npu` profiler** 定位:是 host 调度瓶颈、算子回退 CPU、还是通信(HCCL)瓶颈。
- 分布式优先用上层 **ModelLink / MindFormers** 的成熟并行策略,而非自己手搓 TP/PP。
- 关注 **NPU 内存碎片**,必要时调整缓存分配器行为(具体开关以官方文档为准)。

## 常见问题

| 问题 | 解答 |
|------|------|
| torch_npu 会替换我的 PyTorch 吗? | 不会。它是插件,通过 PrivateUse1 把 `npu` 设备注册进现有 PyTorch。 |
| 为什么 `.npu()` 报错说没有 npu 设备? | 漏了 `import torch_npu`,或 torch_npu 与 PyTorch/CANN 版本错配。 |
| GPU 上的 `flash_attn` 包能用吗? | 不能。换成 torch_npu 的融合注意力算子或上层套件封装的实现。 |
| 分布式要改后端吗? | 把 `nccl` 改成 `hccl` 即可,DDP/Megatron 上层逻辑基本不动。 |
| 选哪几张卡用什么变量? | `ASCEND_RT_VISIBLE_DEVICES`(对标 `CUDA_VISIBLE_DEVICES`)。 |
| 训练大模型推荐怎么起步? | 直接用 ModelLink / MindFormers,或参考 ModelZoo-PyTorch 已适配脚本,少踩算子坑。 |
| 单算子模式慢怎么办? | 切图模式(GE/TorchAir)+ 融合算子,先 profiling 定位瓶颈。 |
| bf16 还是 fp16? | 大模型优先 bf16,数值范围大、训练稳。 |
| 量化怎么做? | 用昇腾的 msmodelslim,推理用 MindIE;不要直接搬 GPU 的 bitsandbytes/AWQ-CUDA。 |
| 具体版本怎么配? | 查华为昇腾官方"配套关系表",三方版本严格对齐。 |

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-infra/算力/昇腾NPU]]
- [[ai-infra/ai-hardware/AI芯片软件生态]]
- [[ai-infra/ai-hardware/CUDA]]
- [[ai-infra/网络/NCCL]]
- [[ai-infra/网络/集合通信原语]]
- [[ai-framework/megatron-lm/README]]
- [[ai-framework/huggingface-transformers/README]]
- [[llm-compression/quantization/量化基础]]
- [[llm-inference/README]]
- [[llm-train/README]]
- [[llm-algo/transformer/模型架构]]
