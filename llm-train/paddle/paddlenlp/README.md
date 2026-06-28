# PaddleNLP 大模型训练与推理全流程

> 一句话定位：PaddleNLP 是百度飞桨（PaddlePaddle）生态里的「自然语言处理 + 大模型」全家桶，覆盖从权重加载、PyTorch 权重互转、动/静态图推理、服务化部署到量化压缩的完整链路；本笔记把官方命令逐条拆解并补足原理。📍 导航：[[00-知识地图]]
>
> 🔗 相关：[[llm-train/README]] · [[ai-framework/pytorch/README]] · [[ai-framework/huggingface-peft/README]] · [[llm-compression/quantization/量化基础]]

---

## 阅读地图

| 节 | 内容 | 你将学到 |
|----|------|----------|
| 0 | 一句话锚点 | PaddleNLP 在国产 AI-Infra 里的坐标 |
| 1 | 地基/前置 | 动态图 vs 静态图、`.pdparams`、Paddle ↔ PyTorch 概念映射 |
| 2 | 安装与 Docker 环境 | 三种安装方式 + GPU 镜像，为什么要锁版本 |
| 3 | 权重加载与 PyTorch 转 Paddle | `from_hf_hub` / `convert_from_torch` 机制 |
| 4 | Taskflow 开箱即用推理 | UIE 信息抽取、模型自动下载缓存 |
| 5 | 生成式推理（CausalLM） | tokenizer/model 加载、`return_tensors="pd"` |
| 6 | 动态图 vs 静态图推理 | `export_model` 导出、为什么静态图更快 |
| 7 | 服务化部署 | Flask & Gradio、多卡 launch |
| 8 | 量化压缩 | PTQ(W8A8) vs GPTQ(WINT4)、显存账 |
| 实操 | 命令/配置汇总 | 原文真料一处可查 |
| 坑 | 常见问题 | 版本/镜像/缓存/转换踩坑 |

---

## 0. 一句话锚点

> **PaddleNLP = 飞桨版的 HuggingFace Transformers + 推理 + 部署 + 压缩。**

它和 PyTorch 生态的对应关系是：

```
   PyTorch 生态                         Paddle 生态
 ┌───────────────────┐             ┌───────────────────────┐
 │ transformers      │  ≈          │ paddlenlp.transformers│
 │ AutoModel...      │             │ AutoModelForCausalLM  │
 │ .pt / .bin /      │  ←转换→     │ .pdparams             │
 │  safetensors      │             │                       │
 │ accelerate/deepspeed│  ≈        │ paddle.distributed    │
 │ pipeline()        │  ≈          │ Taskflow()            │
 └───────────────────┘             └───────────────────────┘
```

记住一个核心差异：**PyTorch 默认动态图（define-by-run）；Paddle 同时是动态图和静态图双模框架**，训练用动态图调试方便，部署用静态图（计算图固定）跑得更快。这条主线贯穿后面的「动态图推理 / 静态图推理」两节。

官方资源（原文保留）：
- 镜像仓库：https://hub.docker.com/r/paddlecloud/paddlenlp
- 大模型目录：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm
- PyTorch 转 Paddle 文档：https://github.com/PaddlePaddle/PaddleNLP/blob/v2.6.1/docs/community/contribute_models/convert_pytorch_to_paddle.rst

---

## 1. 地基：四个必须先懂的概念

### 1.1 动态图 vs 静态图

| 维度 | 动态图 (dynamic / imperative) | 静态图 (static / declarative) |
|------|------|------|
| 执行方式 | 写一行算一行（Python 逐行解释） | 先「画好整张计算图」再统一执行 |
| 调试 | 容易，可 `print`、可断点 | 难，图构建后不能随便插 print |
| 速度 | 略慢（有 Python 开销） | 快（图优化、算子融合、无 Python 解释开销） |
| 部署 | 不利于跨语言/跨平台 | 利于导出、C++/服务端加载 |
| PaddleNLP 对应 | `--mode "dynamic"` | `--mode "static"`（需先 `export_model`） |

ASCII 对照：

```
动态图：  input → [op] →(立刻算)→ tensor → [op] →(立刻算)→ ...
          每个 op 都回到 Python 解释器，灵活但有开销

静态图：  先描述：  input → op → op → op → output   (只建图，不算)
          再执行：  把整张图喂给 C++ 引擎一次性跑完（可融合、可优化）
```

### 1.2 `.pdparams` 是什么

`.pdparams` 是 Paddle 的权重序列化格式，相当于 PyTorch 的 `.bin` / `.pt`：里面是一个「参数名 → 张量」的字典。例如原文出现的：

```
model_state.pdparams   ← 模型权重
tokenizer_config.json  ← 分词器配置
```

### 1.3 张量后端标记 `return_tensors="pd"`

HuggingFace 里 `return_tensors="pt"` 返回 PyTorch 张量、`"tf"` 返回 TensorFlow；PaddleNLP 里是 **`"pd"`（Paddle）**。这是后面推理代码里最容易忘的一处：

```python
input_features = tokenizer("你好！", return_tensors="pd")   # pd = paddle
```

### 1.4 模型下载来源开关

PaddleNLP 加载模型时有三个「从哪下」的开关，理解它们能避免一半的下载报错：

| 参数 | 含义 | 默认 |
|------|------|------|
| （默认） | 从飞桨 BOS（bj.bcebos.com）下载 Paddle 原生权重 | 是 |
| `from_hf_hub=True` | 从 HuggingFace Hub 下载 | 否 |
| `from_aistudio=True` | 从百度 AI Studio 下载 | 否 |
| `convert_from_torch=True` | 下载的是 PyTorch 权重，加载时即时转成 Paddle | 否 |

---

## 2. 安装与 Docker 环境

### 2.1 三种安装方式（原文保留 + 解释）

```bash
# 方式一：装指定稳定版（推荐，可复现）
pip install --upgrade paddlenlp==2.6.1 -i https://pypi.org/simple

# 方式二：装预发布最新版（追新功能，但不稳定）
sudo pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

**为什么强烈建议锁版本 `==2.6.1`？** PaddleNLP 大模型 API 迭代极快，`predictor.py`、`export_model.py`、`finetune_generation.py` 的参数在不同版本之间会变。原文给出的 PyTorch 转 Paddle 文档链接里就明确带了 `v2.6.1` 标签——**文档版本必须和你装的包版本对齐**，否则示例命令会找不到脚本或参数对不上。

```
版本不对齐的典型连锁反应：
  paddlenlp 版本 ≠ 文档版本
        │
        ├─ predictor.py 没有某个 --flag      →  argparse 报错
        ├─ argument.json 字段名变了          →  KeyError
        └─ 模型仓库结构调整                  →  下载 404
```

### 2.2 Docker GPU 镜像（原文保留）

```bash
docker run --name dev \
  --runtime=nvidia \
  -v $PWD:/mnt \
  -p 8888:8888 \
  -it \
  paddlecloud/paddlenlp:develop-gpu-cuda10.2-cudnn7-cdd682 \
  /bin/bash
```

逐参数解释：

| 片段 | 作用 | 为什么 |
|------|------|--------|
| `--name dev` | 容器命名 | 后续 `docker exec dev` 方便 |
| `--runtime=nvidia` | 启用 NVIDIA 容器运行时 | 让容器看见 GPU（旧写法，新版可用 `--gpus all`） |
| `-v $PWD:/mnt` | 把当前目录挂进容器 `/mnt` | 代码/数据在宿主机改，容器里立即可见 |
| `-p 8888:8888` | 端口映射 | 容器内 Jupyter/服务暴露到宿主机 8888 |
| `-it` | 交互式 + 伪终端 | 进容器后能用 shell |
| `...cuda10.2-cudnn7...` | 镜像 tag | **CUDA 版本被钉死在 10.2** |

> ⚠️ 坑：这个镜像 tag 写死 `cuda10.2`。如果你的物理机驱动太新只支持 CUDA 12，或太旧不支持 10.2，容器里 GPU 会用不起来。镜像 CUDA 版本必须 ≤ 宿主机驱动支持的最高 CUDA 版本。

---

## 3. 权重加载与 PyTorch → Paddle 转换

### 3.1 数据流总览

```
                       ┌─────────────────────────────┐
   想用一个模型权重 →  │ 它原生是 Paddle 还是 PyTorch？ │
                       └──────────────┬──────────────┘
            原生 Paddle               │              原生 PyTorch (HF)
                 │                                          │
   from_pretrained("bigscience/             from_pretrained(..., from_hf_hub=True,
     bloomz-560m")  (默认走 BOS)              convert_from_torch=True)
                 │                                          │
        下载 .pdparams                       下载 .bin/safetensors → 即时转 .pdparams
                 │                                          │
                 └──────────────► paddle 模型对象 ◄─────────┘
```

### 3.2 分词器加载的几种写法（原文保留）

```python
# (1) 直接用 Bloom 专用 tokenizer，从飞桨 BOS 下
from paddlenlp.transformers.bloom.tokenizer import BloomTokenizer
tokenizer = BloomTokenizer.from_pretrained("bigscience/bloomz-560m")

# (2) 用 AutoTokenizer 从飞桨 BOS 下
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

# (3) 从 HuggingFace Hub 下
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", from_hf_hub=True)
tokenizer = AutoTokenizer.from_pretrained("ziqingyang/chinese-llama-7b", from_hf_hub=True)
```

> 飞桨 BOS 上分词器文件长这样（原文）：
> `https://bj.bcebos.com/paddlenlp/models/community/bigscience/bloomz-560m/tokenizer_config.json`

### 3.3 模型加载（原文保留 + 标注）

```python
# 从飞桨 BOS 下 Paddle 原生权重，单精度 float32
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", dtype="float32")

# 从 HF Hub 下 PyTorch 权重并即时转成 Paddle
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m",
    dtype="float32",
    from_aistudio=False,
    from_hf_hub=True,
    convert_from_torch=True,   # ← 关键：torch 权重 → paddle
)
```

**`convert_from_torch` 在转什么？** 主要做两件事：① 参数命名/结构对齐（Paddle 与 PyTorch 同一模型层命名规则不同）；② **Linear 层权重转置**。这是最经典的坑：

```
PyTorch Linear:  y = x · Wᵀ + b   （权重存成 [out, in]）
Paddle  Linear:  y = x · W  + b   （权重存成 [in, out]）
            ──────────────────────────────────
转换时必须把每个 Linear 的权重矩阵转置一次，
否则数值不报错但输出全是乱码！
```

官方专门有一篇文档讲这个手工转换流程：
`https://github.com/PaddlePaddle/PaddleNLP/blob/v2.6.1/docs/community/contribute_models/convert_pytorch_to_paddle.rst`

### 3.4 支持的模型示例（原文保留）

```
bigscience/bloom-560m
bigscience/bloomz-560m
```

---

## 4. Taskflow：开箱即用推理（UIE 信息抽取）

`Taskflow` 是 PaddleNLP 的「一行调用一个 NLP 任务」高层 API，类比 HF 的 `pipeline()`。

```python
import paddlenlp
from pprint import pprint
from paddlenlp import Taskflow

schema = ['时间', '选手', '赛事名称']           # 定义要抽取的实体 schema
ie = Taskflow('information_extraction', schema=schema)   # UIE 统一信息抽取
pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))
```

**原理：UIE（Universal Information Extraction）** 把「实体抽取/关系抽取/事件抽取」统一成一个 prompt 式的 span 抽取问题——你给一个 `schema`，模型就去原文里框出对应片段。上面这句话期望抽出：时间=`2月8日`，选手=`谷爱凌`，赛事名称=`北京冬奥会自由式滑雪女子大跳台决赛`。

### 4.1 模型缓存机制（原文保留）

第一次调用会自动下载，缓存到本地 `~/.paddlenlp/`：

```
# 远端（飞桨 BOS）
https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.1/model_state.pdparams

# 本地缓存（macOS 示例，原文）
/Users/liguodong/.paddlenlp/taskflow/information_extraction/uie-base/model_state.pdparams
```

```
首次调用 Taskflow
      │
      ▼
检查 ~/.paddlenlp/taskflow/.../model_state.pdparams 是否存在？
      │ 否                                  │ 是
      ▼                                     ▼
从 bj.bcebos.com 下载 → 写入缓存        直接加载缓存（离线可用）
```

> 坑：缓存目录在用户 HOME 下，**换用户 / Docker 重建容器 / 清家目录** 都会触发重新下载。离线机器要先在有网环境跑一次，再把 `~/.paddlenlp/` 整体拷过去。

---

## 5. 生成式推理（CausalLM）

```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", dtype="float32")

input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")  # pd!
outputs = model.generate(**input_features, max_length=128)
tokenizer.batch_decode(outputs[0])
```

四步流水线：

```
"你好！请自我介绍一下。"
        │ tokenizer(..., return_tensors="pd")
        ▼
input_ids (paddle.Tensor)   ──►  model.generate(max_length=128)
                                        │ 自回归逐 token 解码
                                        ▼
                                  output_ids
                                        │ batch_decode
                                        ▼
                                  "我是..." (文本)
```

**`max_length=128` 数值含义**：限制「prompt + 生成」的总 token 上限为 128。若 prompt 已占 20 token，最多再生成 108 token。

**dtype 显存账（560M 模型为例）**：参数量 ≈ 5.6×10⁸。

$$\text{显存} \approx 参数量 \times 每参数字节数$$

| dtype | 字节/参 | 仅权重显存 | 说明 |
|-------|---------|-----------|------|
| float32 | 4 | $5.6\times10^8 \times 4 \approx 2.24\text{ GB}$ | 原文用的精度，最稳但最占显存 |
| float16 | 2 | $\approx 1.12\text{ GB}$ | 推理常用，省一半 |
| int8(量化) | 1 | $\approx 0.56\text{ GB}$ | 见第 8 节量化 |

（实际还要加激活值/KV-Cache，上表只算权重。）

---

## 6. 动态图推理 vs 静态图推理

这是本仓库 `llm/` 目录里 `predictor.py` / `export_model.py` 的核心用法。

### 6.1 动态图推理（原文保留）

```bash
# 预训练 & SFT 动态图模型推理
python predictor.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16" \
    --mode "dynamic"
```

### 6.2 静态图推理（原文保留）

静态图必须先把动态图「导出」成静态图：

```bash
# 第 1 步：动态图 → 静态图导出
# 注意：LoRA 要先合并参数（见 3.7 LoRA 参数合并）；Prefix Tuning 暂不支持导出
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --output_path ./inference \
    --dtype float16

# 第 2 步：用导出的静态图推理（注意 model_name_or_path 指向 inference 目录）
python predictor.py \
    --model_name_or_path inference \
    --batch_size 1 \
    --data_file ./data/dev.json \
    --dtype "float16" \
    --mode "static"
```

### 6.3 为什么静态图更快（原理图）

```
动态图推理：
  python predictor.py --mode dynamic
      每生成一个 token，都回 Python 逐 op 解释 → 慢

export_model.py（一次性）：
  动态图 → 追踪算子 → 固化成静态计算图 → 存到 ./inference/
                              │
                              ├─ 算子融合（多个小 op 合成一个 kernel）
                              ├─ 常量折叠
                              └─ 去掉 Python 解释开销

静态图推理：
  python predictor.py --mode static --model_name_or_path inference
      引擎直接跑优化后的图 → 快
```

### 6.4 三个易错点

| 易错点 | 说明 |
|--------|------|
| 静态图 `--model_name_or_path inference` | 静态图推理要指向 **导出目录**，不是原始模型名 |
| LoRA 必须先合并 | 静态图导出不认 LoRA 旁路，需先把 LoRA 权重合并回主干 |
| Prefix Tuning 不支持 | 原文明确「Prefix Tuning 暂不支持」静态图导出 |
| `dtype` 三处要一致 | 导出、推理、权重精度尽量统一为 `float16` |

> 🔗 关于 LoRA / Prefix Tuning / Prompt Tuning 的原理，见 [[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prefix-Tuning]] · [[llm-train/peft/Prompt-Tuning]]。

---

## 7. Flask & Gradio UI 服务化部署

```bash
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" flask_server.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --port 8010 \
    --flask_port 8011 \
    --src_length 1024 \
    --dtype "float16"
```

逐项拆解：

| 参数 | 含义 |
|------|------|
| `paddle.distributed.launch` | 飞桨分布式启动器（≈ `torchrun`），拉起多进程 |
| `--gpus "0,...,7"` | 用 8 张卡做张量并行/多卡推理 |
| `--port 8010` | 模型服务（grpc/内部）端口 |
| `--flask_port 8011` | 对外 HTTP（Flask/Gradio UI）端口 |
| `--src_length 1024` | 输入最大长度 1024 token |
| `--dtype float16` | 半精度，省显存 |

部署拓扑：

```
浏览器 ──HTTP──► :8011 (Flask/Gradio UI)
                      │
                      ▼
              :8010 模型服务进程组
        ┌────┬────┬────┬────┬────┬────┬────┬────┐
       GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7   ← 模型按张量并行切到 8 卡
        └────┴────┴── NCCL AllReduce 通信 ──┴────┘
```

> 🔗 多卡为什么要 AllReduce、通信原语细节见 [[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]；张量并行如何切见 [[B07:llm-inference/大模型推理张量并行]]。

---

## 8. 量化压缩

> 原文定位：**量化算法可以将模型权重和激活转为更低比特数值类型表示，能够有效减少显存占用和计算开销。** PaddleNLP 提供 GPTQ 与 PaddleSlim 自研 PTQ，分别实现 **WINT4** 和 **W8A8** 量化。

参考：https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md

### 8.1 两种量化的命名拆解

```
W8A8     →  Weight 8-bit, Activation 8-bit   （权重和激活都量化到 int8）
WINT4    →  Weight INT4                        （只把权重压到 int4，激活仍高精度）
```

| 方案 | 量化对象 | 位宽 | 算法 | 典型收益 | 适用 |
|------|---------|------|------|----------|------|
| **PTQ / W8A8** | 权重 + 激活 | 8/8 | PaddleSlim 自研后训练量化 | 显存≈1/4，计算用 int8 加速 | 追求吞吐、激活也想加速 |
| **GPTQ / WINT4** | 仅权重 | 4 | GPTQ（逐层最小化量化误差） | 显存≈1/8（权重部分） | 显存极度受限、保激活精度 |

「PTQ = Post-Training Quantization 训练后量化」：不重训，拿少量校准数据统计每层数值范围，直接定标度（scale）把 fp16 映射到 int8。

```
量化映射（对称量化简化版）：
   scale = max(|x|) / 127           # int8 范围 [-127, 127]
   x_int8 = round(x / scale)        # 量化
   x_fp   = x_int8 * scale          # 反量化（用于计算）
```

**数值手算示例**：某层权重 fp16 取值范围 max|w| = 0.5，则
$$\text{scale} = 0.5 / 127 \approx 0.003937$$
一个权重 $w = 0.123$ 量化后：$\text{round}(0.123 / 0.003937) = \text{round}(31.24) = 31$；反量化 $31 \times 0.003937 \approx 0.1220$，误差约 0.0003。**量化误差来自这一步 round 取整**，GPTQ 的本质就是用 Hessian 信息逐层补偿这个取整误差，让整体输出尽量不变。

### 8.2 环境安装（原文保留）

```bash
# 装 develop 版 paddlepaddle-gpu（CUDA 11.7）
# 查找页：https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
python -m pip install paddlepaddle-gpu==0.0.0.post117 \
    -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# 装 PaddleSlim —— 发布版
pip install paddleslim

# 装 PaddleSlim —— develop 版（追最新量化策略）
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install

# 验证：进 python，import paddleslim 不报错即成功
python -c "import paddleslim"
```

> ⚠️ `==0.0.0.post117` 不是占位符——这是飞桨 **develop（每日构建）** 包的真实版本号约定，`post117` 表示对应 **CUDA 11.7**。你的 CUDA 版本不同要换对应 tag（如 `post118`），否则装上也跑不了。
>
> ⚠️ 原文 `git clone ... & cd PaddleSlim` 用的是 `&`（后台执行），在 bash 里 clone 还没下完就 `cd` 会失败。**实操建议改成 `&&`**：`git clone ... && cd PaddleSlim`。

### 8.3 执行量化（原文保留）

```bash
# PTQ 量化（W8A8）—— 读 llama/ptq_argument.json 配置
python finetune_generation.py ./llama/ptq_argument.json

# GPTQ 量化（WINT4）—— 读 llama/gptq_argument.json 配置
python finetune_generation.py ./llama/gptq_argument.json
```

注意：量化也复用 `finetune_generation.py` 这个脚本，**靠传入的 `*.json` 配置文件区分** PTQ 还是 GPTQ。配置项（量化位宽、校准数据、算法）都在 json 里。

### 8.4 量化效果直观对比（7B 模型权重显存）

7B 模型 ≈ 7×10⁹ 参数，仅权重：

| 精度 | 字节/参 | 权重显存 | 相对 fp16 |
|------|---------|----------|-----------|
| fp16（基线） | 2 | $\approx 14\text{ GB}$ | 1× |
| W8A8 (int8) | 1 | $\approx 7\text{ GB}$ | 1/2 |
| WINT4 (int4) | 0.5 | $\approx 3.5\text{ GB}$ | 1/4 |

> 🔗 量化的对称/非对称、per-channel、GPTQ/AWQ 等系统讲解见 [[llm-compression/quantization/量化基础]]。

---

## 实操：命令 / 配置一处可查（原文真料汇总）

```bash
# ── 安装 ──────────────────────────────────────────────
pip install --upgrade paddlenlp==2.6.1 -i https://pypi.org/simple
sudo pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

# ── Docker ────────────────────────────────────────────
docker run --name dev --runtime=nvidia -v $PWD:/mnt -p 8888:8888 -it \
  paddlecloud/paddlenlp:develop-gpu-cuda10.2-cudnn7-cdd682 /bin/bash

# ── 动态图推理 ────────────────────────────────────────
python predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat \
  --batch_size 1 --data_file ./data/dev.json --dtype "float16" --mode "dynamic"

# ── 静态图：导出 + 推理 ───────────────────────────────
python export_model.py --model_name_or_path meta-llama/Llama-2-7b-chat \
  --output_path ./inference --dtype float16
python predictor.py --model_name_or_path inference \
  --batch_size 1 --data_file ./data/dev.json --dtype "float16" --mode "static"

# ── 服务化部署（8 卡） ────────────────────────────────
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" flask_server.py \
  --model_name_or_path meta-llama/Llama-2-7b-chat \
  --port 8010 --flask_port 8011 --src_length 1024 --dtype "float16"

# ── 量化环境 ──────────────────────────────────────────
python -m pip install paddlepaddle-gpu==0.0.0.post117 \
  -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
pip install paddleslim
git clone https://github.com/PaddlePaddle/PaddleSlim.git && cd PaddleSlim && python setup.py install

# ── 量化执行 ──────────────────────────────────────────
python finetune_generation.py ./llama/ptq_argument.json    # PTQ  W8A8
python finetune_generation.py ./llama/gptq_argument.json   # GPTQ WINT4
```

```python
# ── 加载 / 推理 Python 真料 ───────────────────────────
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", dtype="float32")
input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
outputs = model.generate(**input_features, max_length=128)
tokenizer.batch_decode(outputs[0])

# 从 HF Hub + torch 权重转换
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m", dtype="float32",
    from_aistudio=False, from_hf_hub=True, convert_from_torch=True)

# Taskflow 信息抽取
from paddlenlp import Taskflow
ie = Taskflow('information_extraction', schema=['时间', '选手', '赛事名称'])
ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
```

---

## 常见问题 / 坑

| 现象 | 根因 | 解决 |
|------|------|------|
| 示例命令 `--xxx` 报 argparse 错 | paddlenlp 包版本 ≠ 文档版本 | 锁版本 `==2.6.1`，文档也看对应 tag |
| 容器里看不到 GPU | 镜像 `cuda10.2` 与宿主机驱动不匹配 | 选与驱动匹配的镜像 tag / 用 `--gpus all` |
| `return_tensors="pt"` 报类型错 | 用了 PyTorch 的 `pt` | Paddle 用 `return_tensors="pd"` |
| 转换后模型输出乱码 | Linear 权重未转置 | 用 `convert_from_torch=True`，参考官方转换文档 |
| 模型反复重新下载 | `~/.paddlenlp/` 缓存丢失（换用户/重建容器） | 离线机预下后拷贝整个缓存目录 |
| 静态图推理找不到模型 | `--model_name_or_path` 仍指原始名 | 静态图要指向 `export_model` 的 `--output_path`（如 `inference`） |
| LoRA 导出静态图失败 | 静态图不认 LoRA 旁路 | 先合并 LoRA 参数再 `export_model` |
| Prefix Tuning 无法导出 | 官方明确「暂不支持」 | 改用动态图推理 |
| `paddleslim` 安装后 import 报错 | paddlepaddle-gpu 的 CUDA tag（post117）与机器 CUDA 不符 | 装匹配 CUDA 的 `postXXX` 版本 |
| `git clone ... & cd` 失败 | `&` 是后台执行，clone 未完成就 cd | 改用 `&&` 串行 |
| float32 加载 7B 直接 OOM | 4 字节/参，7B≈28GB 权重 | 改 `dtype="float16"` 或上量化 |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 训练总览：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]]
- 其他训练框架：[[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]]
- 底层框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]] · [[ai-framework/huggingface-peft/README]]
- 高效微调：[[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]]
- 分布式通信：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]
- 推理并行：[[B07:llm-inference/大模型推理张量并行]]
- 模型与压缩：[[llm-algo/transformer/模型架构]] · [[llm-compression/quantization/量化基础]]
- 对齐：[[llm-alignment/RLHF]]
