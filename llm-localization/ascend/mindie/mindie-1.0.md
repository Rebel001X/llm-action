# MindIE 1.0：昇腾大模型推理引擎实战（ChatGLM2-6B / Qwen1.5-14B）

> MindIE 是华为昇腾（Ascend）的端到端大模型推理引擎，本篇用「拉镜像 → 起容器 → 使能 CANN → 转权重 → 跑推理 / 起服务」一条主线，把昇腾 800I A2（NPU）上的推理流程跑通。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[llm-inference/README]] · [[llm-inference/vllm/README]]

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|------|------------|--------|
| 0. 一句话锚点 | MindIE 在昇腾栈里的位置 | 推理引擎 / NPU |
| 1. 地基 | NPU/CANN/MindIE/llm_model 四层栈 | 软件生态 |
| 2. 镜像与 AscendHub | 镜像命名规则、为什么用容器 | 800I-A2 / aarch64 |
| 3. docker 启动脚本逐行 | 7 个设备/挂载为什么必须给 | davinci_manager |
| 4. 使能 CANN | 三个 set_env.sh 各管什么 | toolkit/mindie/llm_model |
| 5. 推理 ChatGLM2-6B | 权重转 safetensor、run_pa | Paged Attention |
| 6. Qwen1.5-14B 起服务 | config.json、daemon、踩坑 | mindieservice_daemon |
| 实操速查 | 全部真实命令汇总 | 复制即用 |
| 常见坑 | dtype/transformers 版本等 | 表格 |

## 0. 一句话锚点

**MindIE = 昇腾上的「vLLM 类」推理引擎**：它吃模型权重，吐 token，对外既能用 Python 脚本（`run_pa.py`）单跑，也能起一个常驻的 HTTP 推理服务（`mindieservice_daemon`）。`pa` 就是 **P**aged **A**ttention——和 vLLM 同源的显存分页思想（见 [[llm-optimizer/kv-cache]]）。

记住一条主线：

```
拉镜像 → start-docker.sh 起容器 → install_and_enable_cann.sh 使能 CANN
   → convert_weights.py 转权重 → run_pa.py 推理 / mindieservice_daemon 起服务
```

## 1. 地基：昇腾推理软件栈四层

在 GPU 上你熟悉的是「CUDA → cuDNN/TensorRT → vLLM」。昇腾的对应栈是：

```
┌───────────────────────────────────────────────┐
│  应用层  run_pa.py / mindieservice_daemon       │  ← 你直接调的
├───────────────────────────────────────────────┤
│  llm_model  模型库(examples/、convert/)         │  /usr/local/Ascend/llm_model
├───────────────────────────────────────────────┤
│  MindIE   推理引擎(service/、PagedAttn)         │  /usr/local/Ascend/mindie
├───────────────────────────────────────────────┤
│  CANN     算子+运行时(类比 CUDA+cuDNN)          │  ascend-toolkit/set_env.sh
├───────────────────────────────────────────────┤
│  Driver   NPU 驱动 + npu-smi                     │  /usr/local/Ascend/driver
├───────────────────────────────────────────────┤
│  硬件     昇腾 910 NPU(达芬奇架构)               │  /dev/davinci*
└───────────────────────────────────────────────┘
```

**为什么分这么多层？** 因为每层各管一件事，缺一层就跑不起来：

| 层 | 类比 GPU | 缺了会怎样 |
|----|---------|-----------|
| Driver | NVIDIA Driver | `npu-smi` 不通，看不到卡 |
| CANN | CUDA + cuDNN | 算子无法下发到 NPU |
| MindIE | TensorRT-LLM / vLLM | 没有 PagedAttention、batching、服务化 |
| llm_model | HF transformers 适配层 | 没有 ChatGLM2/Qwen 的昇腾实现 |

对照 GPU 生态请看 [[ai-infra/ai-hardware/AI芯片软件生态]] 和 [[ai-infra/ai-hardware/硬件对比]]；NPU 达芬奇架构细节见 [[ai-infra/算力/昇腾NPU]]。

## 2. 镜像与 AscendHub

镜像不在 Docker Hub，而在华为 **AscendHub**：

- 详情页：`https://ascendhub.huawei.com/#/detail/mindie`
- 镜像：`ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64`

**镜像标签逐段拆解**（这一串信息量很大，对错卡型会直接跑不起来）：

```
mindie : 1.0.RC1 - 800I-A2 - aarch64
  │        │         │          └─ CPU 架构：ARM64(鲲鹏服务器)，不是 x86
  │        │         └─ 硬件形态：Atlas 800I A2 推理服务器
  │        └─ 版本：1.0 Release Candidate 1
  └─ 引擎名
```

> 坑：`aarch64` 说明宿主机是**鲲鹏 ARM** 服务器；在 x86 机器上拉这个镜像会架构不匹配。800I A2 是**推理**形态（对应训练形态是 800T）。

**为什么一定用容器？** 因为 CANN/MindIE 对 OS、Python、glibc 版本极度敏感，容器把整套依赖冻结，宿主机只需提供「驱动 + 设备文件」。这也是下一节大量 `--device`/`-v` 的由来。

## 3. docker 启动脚本逐行解释

原文 `start-docker.sh`（存放于 `/home/chatglm2_6b`，ChatGLM2 用法）：

```bash
IMAGES_ID=$1
NAME=$2
if [ $# -ne 2 ]; then
    echo "error: need one argument describing your container name."
    exit 1
fi
docker run --name ${NAME} -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /tmp:/tmp \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    ${IMAGES_ID}
```

参数说明（原文）：

- `IMAGES_ID` 为镜像版本号（`docker images` 回显中的 IMAGE ID）。
- `NAME` 为启动容器名，可自定义设置。

**每个 flag 为什么必须给**——这是昇腾容器最容易踩的地方：

```
  容器内进程                 宿主机内核/驱动
 ┌──────────┐   ① 设备文件   ┌──────────────┐
 │ run_pa.py│──/dev/davinci──│  NPU Driver   │
 │ MindIE   │──manager等─────│  达芬奇硬件    │
 └────┬─────┘               └──────────────┘
      │ ② 复用宿主驱动/工具(-v 挂载，不重装)
      └── /usr/local/Ascend/driver, npu-smi, dcmi
```

| flag | 作用 | 不给会怎样 |
|------|------|-----------|
| `--device=/dev/davinci_manager` | NPU 设备管理器 | 看不到/用不了 NPU |
| `--device=/dev/hisi_hdc` | 海思设备控制 | 设备通信失败 |
| `--device=/dev/devmm_svm` | 设备内存共享虚拟内存 | 大块显存映射失败 |
| `--privileged=true` | 放开设备权限 | 设备访问被拒 |
| `-v .../driver` | 把宿主机 NPU 驱动映射进容器 | 容器内无驱动，CANN 报错 |
| `-v .../npu-smi` `-v .../dcmi` | 监控/管理工具 | 容器内 `npu-smi info` 不可用 |
| `--net=host` | 共用宿主网络 | 服务端口/外网访问不便 |
| `--shm-size=500g` | 共享内存(进程间/张量) | 大 batch/多进程 OOM；SHM 太小直接崩 |
| `-v .../localtime` | 时区对齐到上海 | 日志时间错乱 |
| `-e http_proxy` | 容器内能访问外网 | 装包/拉权重失败 |

> 关键直觉：**驱动留在宿主机、容器只挂载**。所以你升级镜像不用动驱动，升级驱动不用重做镜像——这是容器化昇腾环境的核心设计。

**启动并进入容器**（原文）：

```bash
cd /home/chatglm2_6b
# 用户可以设置 docker images 命令回显中的 IMAGES ID
image_id=001b7368f6e0
# 用户可以自定义设置镜像名
custom_image_name=chatGLM2_6B
# 启动容器(确保启动容器前，本机可访问外网)
bash start-docker.sh ${image_id} ${custom_image_name}
# 进入容器
docker exec -itu root ${custom_image_name} bash
```

## 4. 使能昇腾 CANN 软件栈

进容器后第一件事是「使能 CANN」。原文：

```bash
cd /opt/package
# 安装CANN包
source install_and_enable_cann.sh
# 若退出后重新进入容器，则需要重新加载 CANN 环境变量，执行以下三行命令
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh
```

**为什么是 `source` 而不是 `bash`？** `set_env.sh` 要往**当前 shell** 注入 `PATH/LD_LIBRARY_PATH/PYTHONPATH/ASCEND_*` 环境变量。`bash xxx.sh` 起子进程，变量随子进程退出而丢失；`source` 在当前进程执行，变量才能留下。

**三个 set_env.sh 各管哪一层**（对应第 1 节四层栈）：

```
set_env.sh                层      注入的关键变量
─────────────────────────────────────────────────────
ascend-toolkit/set_env.sh CANN   ASCEND_TOOLKIT_HOME, 算子库/LD路径
mindie/set_env.sh         MindIE  推理引擎库路径
llm_model/set_env.sh      模型库  模型代码路径(部分PYTHONPATH)
```

> 坑：**容器重进必须重新 source 这三行**——环境变量不持久化。原文专门强调了这点。一个常见现象是「昨天好好的，今天 `import torch_npu` 就报找不到」，十有八九是忘了 source。

## 5. 推理 ChatGLM2-6B

模型说明文档在 `/usr/local/Ascend/llm_model/pytorch/examples/chatglm2/6b/README.md`。

**权重放置约定**（原文）：建议将权重存放于 `/home/chatglm2_6b/weight`，并设置 `CHECKPOINT=/home/chatglm2_6b/weight`。

推理三步（原文）：

```bash
cd /usr/local/Ascend/llm_model

# ① 权重转 safetensor
python examples/convert/convert_weights.py --model_path ${CHECKPOINT}

# ② 执行推理脚本
python examples/run_pa.py --model_path ${CHECKPOINT}

# ③ 自定义问题
python examples/run_pa.py --model_path ${CHECKPOINT} --input_texts "What is deep learning?"
```

启动后会执行推理，显示默认问题 Question 和推理结果 Answer。

**① 为什么要转 safetensor？**
HF 原始权重多是 `pytorch_model.bin`（pickle 格式）。`safetensors` 的优势：

```
.bin (pickle)              .safetensors
─────────────             ─────────────
反序列化执行任意代码(风险)  纯数据，零代码执行(安全)
整文件读完才能用           头部记录offset → 零拷贝 mmap 按需读
加载慢                     加载快，省内存峰值
```

所以转换不只是格式问题，更是**加载速度 + 安全**。

**② `run_pa.py` 里的 `pa` = Paged Attention。**
KV-Cache 是推理显存大头。传统做法给每条序列预留「最大长度」连续显存，浪费严重；PagedAttention 把 KV-Cache 切成固定大小的 block，像操作系统分页一样按需分配：

```
传统连续KV：  [████████░░░░░░░░]  预留max_len，实际用一半 → 浪费50%
分页KV：      [██][██][██]        按block分配，几乎零浪费
              block表 → 物理块
```

数值直觉：一条 13B 模型、seq_len=2048 的序列，FP16 KV-Cache 约
$2 \times L \times 2 \times h \times \text{seqlen} \times 2\text{B}$。以 40 层、hidden=5120 估算单序列约
$2 \times 40 \times 5120 \times 2048 \times 2\text{B} \approx 1.6\,\text{GB}$。并发上百条时，分页能省下的就是几十 GB 量级——这正是 MindIE 能高并发的原因。原理同 vLLM，详见 [[llm-optimizer/kv-cache]] 与 [[llm-inference/vllm/README]]。

## 6. Qwen1.5-14B：从单跑到起推理服务

ChatGLM2 用 `run_pa.py` 单跑；Qwen1.5-14B 这一段演示**起常驻服务** `mindieservice_daemon`（生产形态）。

**起容器**（注意与 ChatGLM2 的差异，原文）：

```bash
# docker rm -f mindie-dev
docker run --name mindie-dev2 -it -d --net=host --ipc=host \
--shm-size=50g \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--entrypoint=bash \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
-v /tmp:/tmp \
-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
ascendhub.huawei.com/public-ascendhub/mindie:1.0.RC1-800I-A2-aarch64

docker exec -itu root mindie-dev2 bash
```

与 ChatGLM2 脚本两处差异及原因：

| 差异 | ChatGLM2 | Qwen1.5-14B | 为什么 |
|------|----------|-------------|--------|
| `--ipc=host` | 无 | 有 | 共享宿主 IPC 命名空间，便于多进程共享内存 |
| `--shm-size` | 500g | 50g | 单模型服务并发规模不同，按需给 |

**使能 CANN**（同第 4 节）：

```bash
cd /opt/package
source ./install_and_enable_cann.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh
```

**配置服务 config.json**（原文）：

```bash
rm /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 模型权重路径填： /home/aicc/model_from_hf/Qwen1.5-14B-Chat
```

`config.json` 是 MindIE Service 的总配置（模型路径、权重格式、并发/batch、端口等）。删旧建新是为了避免镜像里默认配置残留导致路径/参数对不上。把模型权重指向 `/home/aicc/model_from_hf/Qwen1.5-14B-Chat`。

**启动服务**（原文）：

```bash
export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

服务化结构：

```
   HTTP 请求
      │
      ▼
┌──────────────────┐
│ mindieservice_   │  读 conf/config.json
│ daemon (常驻)    │  ├─ 模型: Qwen1.5-14B-Chat
└────────┬─────────┘  ├─ batching / PagedAttn
         │            └─ 端口 / 并发
         ▼
   MindIE 引擎 → CANN 算子 → NPU
```

> `export PYTHONPATH=.../llm_model` 是因为 daemon 要 import 模型库代码；不加会 `ModuleNotFoundError`。与 [[llm-inference/PD分离]] 思路一致，生产推理走「常驻服务 + 调度」而非脚本单跑。

## 实操：全流程真实命令速查

```bash
# === 1. 起容器(ChatGLM2) ===
image_id=001b7368f6e0
custom_image_name=chatGLM2_6B
bash start-docker.sh ${image_id} ${custom_image_name}
docker exec -itu root ${custom_image_name} bash

# === 2. 使能 CANN(每次进容器都要) ===
cd /opt/package && source install_and_enable_cann.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh

# === 3. ChatGLM2 推理 ===
CHECKPOINT=/home/chatglm2_6b/weight
cd /usr/local/Ascend/llm_model
python examples/convert/convert_weights.py --model_path ${CHECKPOINT}
python examples/run_pa.py --model_path ${CHECKPOINT}
python examples/run_pa.py --model_path ${CHECKPOINT} --input_texts "What is deep learning?"

# === 4. Qwen1.5-14B 起服务 ===
rm /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json   # 模型路径: /home/aicc/model_from_hf/Qwen1.5-14B-Chat
export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

依赖与精度（原文踩坑）：

```text
transformers==4.30.2                       # 旧基线版本
pip install transformers==4.37.2 -i https://pypi.tuna.tsinghua.edu.cn/simple   # Qwen1.5 需要的版本
config.json 中 "torch_dtype": "bfloat16"  改为  "float16"
```

## 常见问题 / 坑

| 现象 / 操作 | 原因 | 解法 |
|------------|------|------|
| 重进容器后 `import torch_npu`/CANN 报错 | 环境变量不持久 | 重新 `source` 那三个 `set_env.sh` |
| 用 `bash set_env.sh` 后变量没生效 | 子进程注入变量丢失 | 必须用 `source` |
| `npu-smi info` 在容器里不可用 | 没挂 `-v npu-smi`/`-v dcmi` 或缺 `--device` | 按第 3 节补齐设备与挂载 |
| 大 batch / 多进程 OOM | `--shm-size` 太小 | 调大 SHM（ChatGLM2 示例给到 500g） |
| 镜像拉下来跑不起来 | 架构/卡型不符 | 确认 `aarch64`(ARM 鲲鹏) + `800I-A2` 与机器一致 |
| Qwen1.5 加载/分词报错 | transformers 版本太旧 | `pip install transformers==4.37.2`（Qwen1.5 需要） |
| `bfloat16` 跑不动 / 精度异常 | 该硬件/路径上用 fp16 更稳 | config 里 `"bfloat16"` → `"float16"` |
| daemon `ModuleNotFoundError` | 模型库不在 PYTHONPATH | `export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH` |
| 装包/拉权重超时 | 容器无外网 | 起容器前确认外网，传 `http_proxy/https_proxy` |
| `convert_weights.py` 找不到权重 | `CHECKPOINT` 未设或路径错 | 权重放 `/home/chatglm2_6b/weight` 并设 `CHECKPOINT` |

> bf16 vs fp16 直觉：两者都是 16 位。bf16 指数位多（动态范围大、训练友好），fp16 尾数位多（同范围内精度高）。某些昇腾推理路径对 fp16 支持更成熟，故改 `float16` 更稳。量化与精度全景见 [[llm-compression/quantization/fp8]] 与 [[llm-compression/quantization/量化基础]]。

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 硬件底座：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/GPU工作原理]]
- 推理引擎对照：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/解码策略]]
- 显存与注意力：[[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[docs/transformer内存估算]]
- 模型架构：[[llm-algo/transformer/模型架构]] · [[llm-algo/旋转编码RoPE]] · [[llm-algo/moe/README]]
- 压缩量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]] · [[llm-compression/quantization/GPTQ]]
- 性能评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
