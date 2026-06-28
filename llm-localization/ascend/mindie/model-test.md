# 昇腾 MindIE ModelTest：大模型性能与精度测试

> 一句话定位：ModelTest 是昇腾 MindIE 体系下统一跑「性能基准 + 下游精度」的测试工具，一条 `run.sh` 命令覆盖 NPU(PA) 与 GPU(FA) 两种后端。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] · [[llm-inference/README]] · [[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 小节 | 你将搞懂 | 关键词 |
|---|---|---|
| 0. 一句话锚点 | ModelTest 到底测什么 | 性能 / 精度 |
| 1. 地基 | CANN / ATB / PA / FA 是什么关系 | 软件栈 |
| 2. 两条测试链路 | 性能链路 vs 精度链路 | e2e / 首Token / CEval |
| 3. 环境变量 | 每个 `source` / `export` 为什么必须 | set_env / 可见卡 |
| 4. run.sh 参数逐位拆解 | 命令每个位置含义 | case_pair / chip_num |
| 5. model_name 映射 | 为什么 Baichuan1 复用 baichuan2 | 权重适配 |
| 6. 性能指标手算 | 吞吐 / TTFT / TPOT 怎么算 | 数值示例 |
| 实操 | 原文命令原样可复制 | run.sh |
| 常见坑 | 300I DUO 特殊环境变量等 | starcoder / baichuan2-13b |

## 0. 一句话锚点

ModelTest 为大模型的**性能**和**精度**提供测试功能。它把「拉起模型 → 喂固定数据 → 收集指标」这套流程脚本化，让你用一条命令就能回答两个问题：

- **快不快**？——性能测试：指定 batch、指定输入/输出长度，量 e2e 时延、吞吐、首 Token 与非首 Token 性能。
- **准不准**？——精度测试：在 CEval、MMLU、BoolQ、HumanEval 四个下游数据集上跑准确率。

目前支持两种后端组合：

1. **NPU，PA 场景**，性能 / 精度测试，float16
2. **GPU，FA 场景**，精度测试，float16

> PA = Paged Attention（昇腾 ATB 的分页 KV-Cache 注意力实现）；FA = Flash Attention（GPU 侧参考实现）。注意 GPU 只做精度，不做性能——GPU 在这里是「精度对齐的标尺」，而非被优化对象。

## 1. 地基：从硬件到 ModelTest 的软件栈

要看懂后面的 `source` 命令，先建立这张栈图。从下往上，每一层只依赖它下面那层：

```
┌─────────────────────────────────────────────┐
│  ModelTest (run.sh)  ← 本文主角，测试编排层    │
├─────────────────────────────────────────────┤
│  模型仓 (set_env.sh)  ← Llama/Qwen... 的图实现 │
├─────────────────────────────────────────────┤
│  ATB 加速库 (atb/set_env.sh)                   │
│   Ascend Transformer Boost：PA、融合算子       │
├─────────────────────────────────────────────┤
│  CANN Toolkit (ascend-toolkit/set_env.sh)     │
│   编译器+算子库+Runtime，对标 CUDA+cuDNN       │
├─────────────────────────────────────────────┤
│  昇腾 NPU 硬件 (910 / 310P / 300I DUO)         │
└─────────────────────────────────────────────┘
```

**为什么是三个独立的 `set_env.sh`？** 因为这三层是分别发布、可独立升级的产物。CANN 给你算子和 Runtime；ATB 在 CANN 之上提供 Transformer 专用的融合算子与 PA；模型仓再在 ATB 之上拼出具体网络的计算图。每一层的环境变量（`PATH` / `LD_LIBRARY_PATH` / `PYTHONPATH` / `ASCEND_HOME` 等）都要 source 进当前 shell，缺一层就会在拉起时报「找不到 so / 找不到算子」。

> 类比 GPU 栈：CANN ≈ CUDA Toolkit，ATB ≈ cuBLAS/cuDNN/FlashAttention 内核库，模型仓 ≈ 框架里的 modeling 文件。详见 [[ai-infra/ai-hardware/AI芯片软件生态]]、[[ai-infra/算力/昇腾NPU]]。

## 2. 两条测试链路：性能 vs 精度

ModelTest 内部其实是两条互不干扰的流水线，由 `run.sh` 的第二个参数选择走哪条：

```
                    ┌──────────── performance ───────────┐
                    │  造定长输入(seq_in) → 跑生成(seq_out)│
 run.sh ──类型──────┤  → 计 e2e / TTFT / TPOT / 吞吐       │→ 写 ATB_TESTDATA
                    │                                     │
                    └─ full_CEval/MMLU/BoolQ/HumanEval ───┐
                       读数据集 → batch 推理 → 对答案 → Acc │→ 写 ATB_TESTDATA
                                                          ┘
```

**性能链路**：不关心生成内容对不对，只关心耗时。输入长度由你指定的 `case_pair`（如 `[[512,512]]`）决定，模型从 prefill 到 decode 把 512 个 token 生成出来，过程中打点采集。

**精度链路**：不关心快慢，把数据集题目灌进去，比对模型输出与标准答案算准确率。四个数据集各有侧重：

| 数据集 | 测什么 | 形态 |
|---|---|---|
| CEval | 中文学科知识（52 学科） | 选择题 |
| MMLU | 英文多任务知识（57 学科） | 选择题 |
| BoolQ | 阅读理解判断 | 是/否 |
| HumanEval | 代码生成 | 函数补全 + 单测 |

> 这就是为什么 GPU 只跑精度链路：它作为黄金参考，用 FA 跑出一份「正确的」准确率，NPU 的 PA 实现要对齐到它。详见 [[llm-eval/README]]。

## 3. 环境变量：每一行为什么必须

下面是原文的环境变量配置，逐行解释它的作用：

```shell
# source cann环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source 加速库环境变量
source /usr/local/Ascend/atb/set_env.sh
# source 模型仓tar包解压出来后的环境变量
source set_env.sh
# 设置ATB_TESTDATA环境变量
export ATB_TESTDATA="[path]"          # 用于存放测试结果的路径
# 设置使用卡号
export ASCEND_RT_VISIBLE_DEVICES="[卡号]"   # NPU场景，如"0,1,2,3,4,5,6,7"
# 或
export CUDA_VISIBLE_DEVICES="[卡号]"        # GPU场景，如"0,1,2,3,4,5,6,7"
```

| 变量 / 命令 | 作用 | 漏了会怎样 |
|---|---|---|
| `ascend-toolkit/set_env.sh` | 注入 CANN 的 PATH/库路径/Runtime | 找不到 `acl*` / 算子库，拉起即崩 |
| `atb/set_env.sh` | 注入 ATB 加速库（PA、融合算子） | PA 算子缺失，性能链路无法跑 |
| `set_env.sh`（模型仓） | 注入模型图实现到 `PYTHONPATH` | import 不到具体模型的 modeling |
| `ATB_TESTDATA` | 测试结果落盘目录 | 结果无处写，或写到默认目录难找 |
| `ASCEND_RT_VISIBLE_DEVICES` | NPU 可见卡（逻辑↔物理映射） | 多卡并行拿不到卡 / 用错卡 |
| `CUDA_VISIBLE_DEVICES` | GPU 可见卡 | 同上（GPU 侧） |

**可见卡变量的本质**：`"0,1,2,3,4,5,6,7"` 是把 8 张物理卡映射成进程内逻辑 rank 0~7。张量并行（TP）会把权重按这些 rank 切分，所以这里的卡数必须与 `run.sh` 末尾的 `chip_num` 一致——否则切分维度对不上直接报错。NPU 用 `ASCEND_RT_VISIBLE_DEVICES`，GPU 用 `CUDA_VISIBLE_DEVICES`，二选一，跟你跑哪个后端对应。

## 4. run.sh 参数逐位拆解

这是工具的核心命令。先看原文两种形态：

```shell
# NPU
bash run.sh pa_fp16 [performance|full_CEval|full_MMLU|full_BoolQ|full_HumanEval] ([case_pair]) [batch_size] [model_name] ([use_refactor]) [weight_dir] [chip_num] ([max_position_embedding/max_sequence_length])

# GPU
bash run.sh fa [full_CEval|full_MMLU|full_BoolQ|full_HumanEval] [batch_size] [model_name] ([use_refactor]) [weight_dir] [chip_num]
```

把 NPU 形态拆成「位置 → 含义」对照表（`()` 表示可选位）：

| 位置 | 参数 | 取值 | 说明 |
|---|---|---|---|
| 1 | 后端模式 | `pa_fp16` / `fa` | NPU 用 PA，GPU 用 FA |
| 2 | 测试类型 | `performance` 或 `full_*` | 选性能链路或某个数据集 |
| 3 | `case_pair` | `[[256,256],[512,512]]` | **仅 performance 场景**接受；一组或多组 `[seq_in, seq_out]` |
| 4 | `batch_size` | 整数 | 并发样本数 |
| 5 | `model_name` | 见第 5 节 | 选哪套模型图实现 |
| 6 | `use_refactor` | `True`/`False` | **仅当 model_name=llama 时必填**，统一用 `True` |
| 7 | `weight_dir` | 路径 | 权重所在目录 |
| 8 | `chip_num` | 整数 | 使用的卡数，须与可见卡一致 |
| 9 | `max_position_embedding` | 整数 | 可选；不传则用 config 默认 |

原文对各参数的权威说明（原样保留）：

```
1. case_pair只在performance场景下接受输入，接收一组或多组输入，
   格式为[[seq_in_1,seq_out_1],...,[seq_in_n,seq_out_n]], 如[[256,256],[512,512]]
3. 当model_name为llama时，须指定use_refactor为True或者False（统一使用True）
4. weight_dir: 权重路径
5. chip_num: 使用的卡数
6. max_position_embedding: 可选参数，不传入则使用config中的默认配置
7. 运行完成后，会在控制台末尾呈现保存数据的文件夹
```

**`case_pair` 为什么是 `[seq_in, seq_out]` 对？** 因为 Transformer 推理分两段，时延特征完全不同：

```
seq_in (prefill 一次算完，并行)        seq_out (decode 逐 token，串行)
┌──────────────────────────┐   ┌──┬──┬──┬──┬──┬──┐
│  512 个输入 token 一把过   │ → │t1│t2│t3│...│   │  ← 每步只算 1 个 token
└──────────────────────────┘   └──┴──┴──┴──┴──┴──┘
   决定 TTFT(首Token时延)          每步间隔决定 TPOT(每Token时延)
```

`seq_in` 越大，prefill 计算量越大，**首 Token 越慢**；`seq_out` 越大，decode 步数越多，**总时延线性增长**且 KV-Cache 占用越大。所以同一个模型，`[[256,256]]` 和 `[[512,512]]` 会测出截然不同的曲线——这正是 case_pair 存在的意义。相关原理见 [[llm-optimizer/kv-cache]]、[[llm-inference/PD分离]]。

**`use_refactor` 为什么只对 llama 必填？** llama 系列在仓里有「重构版（refactor）」与老版两套图实现，重构版通常融合更充分、性能更好，所以强制你显式选择且统一用 `True`。其他模型只有单一实现，不需要这个开关。

## 5. model_name 映射表：权重名 ≠ 实现名

一个反直觉但很关键的点：你传的不是模型「商品名」，而是它对应的**图实现名**。多个权重可能复用同一套实现：

| 权重（你手上的模型） | model_name（传给 run.sh） |
|---|---|
| Llama-65B, Llama2-7B/13B/70B | `llama` |
| CodeLlama-13B, Chinese-Alpaca-13B, Yi-6B-200K, Yi-34B | `llama` |
| Starcoder-15.5B | `starcoder` |
| Chatglm2-6B | `chatglm2_6b` |
| CodegeeX2-6B | `codegeex2_6b` |
| Baichuan2-7B | `baichuan2_7b` |
| Baichuan2-13B | `baichuan2_13b` |
| Qwen-14B, Qwen-72B | `qwen` |
| Aquila-7B | `aquila_7b` |
| Deepseek16B | `deepseek` |
| Mixtral 8*7B | `mixtral` |
| Bloom-7B | `bloom_7b` |
| **Baichuan1-7B** | `baichuan2_7b`（复用 2 代实现） |
| **Baichuan1-13B** | `baichuan2_13b`（复用 2 代实现） |

**为什么 Yi / CodeLlama / Chinese-Alpaca 都映射到 `llama`？** 因为它们在架构上就是 Llama 家族（RMSNorm + RoPE + SwiGLU + GQA/MHA），只是权重与词表不同，图实现可直接复用。这也解释了为什么 Baichuan1 借用 baichuan2 的实现——两代架构差异小到一套图能覆盖。架构共性见 [[llm-algo/transformer/模型架构]]、[[llm-algo/旋转编码RoPE]]；Mixtral 的 MoE 结构见 [[llm-algo/moe/README]]。

> 完整支持清单：Llama 系、Starcoder-15.5B、Chatglm2-6B、CodegeeX2-6B、Baichuan2(7B/13B)、Qwen(14B/72B)、Aquila-7B、Deepseek16B、Mixtral 8*7B、Bloom-7B、Baichuan1(7B/13B)、CodeLlama-13B、Yi(6B-200K/34B)、Chinese-Alpaca-13B。

## 6. 性能指标怎么算（数值手算）

性能链路输出的核心是这几个量。用一个具体例子算一遍，帮你看懂报告里的数字。

设定：batch=16，case_pair=`[[512,512]]`，即每条样本输入 512、输出 512 token。假设实测：

- 首 Token 时延 TTFT = 0.40 s（prefill 512 token 一次算完）
- 非首 Token 平均时延 TPOT = 25 ms/token

**单条样本 e2e 时延**（生成 512 个 token，第 1 个是首 Token，其余 511 个是非首）：

$$T_{e2e} = \text{TTFT} + (\text{seq\_out}-1)\times\text{TPOT} = 0.40 + 511\times0.025 \approx 13.2\ \text{s}$$

**生成吞吐（output tokens/s）**：batch 内并行生成，总输出 = $16\times512 = 8192$ token：

$$\text{Throughput} = \frac{B\times\text{seq\_out}}{T_{e2e}} = \frac{8192}{13.2} \approx 620\ \text{tok/s}$$

**直觉检查**：把 seq_out 翻倍到 1024，e2e ≈ 0.40+1023×0.025 ≈ 26 s，吞吐几乎不变（decode 是串行瓶颈，batch 越大每秒 token 越多但单样本时延不降）。把 batch 翻倍到 32，若不撑爆显存，吞吐近似翻倍而 e2e 基本不变——这就是为什么性能测试要扫不同 batch 和 case_pair。指标术语详见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]，内存估算见 [[docs/transformer内存估算]]。

## 实操：安装依赖与运行（原文真料）

### 安装 python 依赖

```shell
pip install -r requirements.txt
```

### 运行示例（原文保留）

```shell
# 举例 1：测试 Llama-70B 在 8 卡 [512,512] 场景下，16 batch 的性能，使用归一(重构)代码
bash run.sh pa_fp16 performance [[512,512]] 16 llama True /path 8

# 举例 2：测试 Starcoder-15.5B 在 8 卡 1 batch 下游数据集 BoolQ 的精度
bash run.sh pa_fp16 full_BoolQ 1 starcoder /path 8
```

逐字对照举例 1 与第 4 节参数表：`pa_fp16`(后端) / `performance`(类型) / `[[512,512]]`(case_pair) / `16`(batch) / `llama`(model_name) / `True`(use_refactor，llama 必填) / `/path`(weight_dir) / `8`(chip_num)。举例 2 因为 `starcoder` 不是 llama，**没有 use_refactor 位**，且因为是精度链路**没有 case_pair 位**——这正是可选位随场景增减的体现。

> 运行完成后，控制台**末尾**会打印保存数据的文件夹（即 `ATB_TESTDATA` 下的结果目录），结果就在那里看。

## 常见问题 / 坑

| 现象 / 场景 | 原因 | 解法 |
|---|---|---|
| 拉起报缺 so / 缺算子 | 三个 `set_env.sh` 没全 source | 按顺序 source：toolkit → atb → 模型仓 set_env |
| TP 切分维度报错 | `chip_num` 与可见卡数不一致 | 让 `ASCEND_RT_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` 的卡数 == chip_num |
| llama 直接报参数错 | 漏传 `use_refactor` | llama 必须显式给 `True`/`False`，统一用 `True` |
| performance 报 case_pair 解析失败 | 格式不对或非 performance 场景传了它 | 仅 performance 接受，格式 `[[in,out],...]` |
| Baichuan1 不知道传什么名 | 没有独立实现 | 7B→`baichuan2_7b`，13B→`baichuan2_13b` |
| GPU 想跑性能测试 | GPU 仅支持 FA 精度 | 性能只能在 NPU(PA) 跑 |
| 结果找不到 | 没设 `ATB_TESTDATA` | 显式 export；看控制台末尾打印的文件夹 |

### Starcoder 特别运行操作说明

对于 **300I DUO**，需修改 `core/starcoder.py` 中的 `prepare_environ` 函数，设置如下环境变量：

```python
os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
os.environ['LCCL_ENABLE_FALLBACK'] = "0"
```

- `ATB_LAUNCH_KERNEL_WITH_TILING="1"`：启用带 tiling 的 kernel 下发，适配 300I DUO 的算子切分方式。
- `LCCL_ENABLE_FALLBACK="0"`：关闭 LCCL 集合通信的回退路径，强制走优化通信。集合通信原理见 [[ai-infra/网络/集合通信原语]]。

### Baichuan2-13B 特别运行操作说明

对于 **300I DUO**，需修改 `core/baichuan2_13b_test.py` 中的 `prepare_environ` 函数：

```python
os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "0"
os.environ['TASK_QUEUE_ENABLE'] = "0"
```

- `ATB_OPERATION_EXECUTE_ASYNC="0"`：关闭 ATB 算子异步执行，改为同步——在 300I DUO 上规避异步调度引发的问题。
- `TASK_QUEUE_ENABLE="0"`：关闭下发任务队列，配合同步执行避免乱序。

> 这两段都属于「特定芯片 + 特定模型」的兼容性补丁：300I DUO 与 910 系列在算子下发/通信路径上有差异，故针对 starcoder、baichuan2-13b 单独打开/关闭这些开关。换其他芯片或模型一般不需要改。

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 昇腾硬件与生态：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/硬件对比]]
- 推理引擎与并行：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/解码策略]]
- 注意力与缓存：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 模型架构：[[llm-algo/transformer/模型架构]] · [[llm-algo/旋转编码RoPE]] · [[llm-algo/moe/README]]
- 量化（fp16/fp8 精度相关）：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 评测与指标：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
- 通信：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/InfiniBand]]
