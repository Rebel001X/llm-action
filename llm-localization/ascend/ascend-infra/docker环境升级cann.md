# 在 Docker 容器中升级昇腾 CANN（含 torch-npu 环境重建）

> 在不动宿主机的前提下，把昇腾容器里的 CANN（含 toolkit + kernels）从旧版本原地升级到新版本，并重建可用的 PyTorch+NPU 训练环境。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[llm-localization/ascend/ascend-infra/达芬奇架构]] · [[ai-framework/pytorch/README]]

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|------|------------|--------|
| 0 锚点 | 一句话理解"升级 CANN"到底升的是什么 | toolkit / kernels / driver |
| 1 地基 | 昇腾软件栈分层、为什么能"只升容器不升宿主机" | driver↔firmware↔CANN↔torch-npu |
| 2 拉取基础镜像 | `docker login/pull` 私有 hub | ascendhub |
| 3 下载 CANN 安装包 | toolkit 与 kernels 两个 `.run` 的区别 | aarch64 / 910b |
| 4 起升级容器 | `docker run` 每个挂载/参数为什么必须有 | /usr/slog / hccn_tool |
| 5 卸旧装新 | uninstall→check→install→set_env 顺序 | 版本号路径 |
| 6 装 kernels | `--feature=aclnn_ops_train` 含义 | 算子包 |
| 7 重建 Python 环境 | conda + torch + torch-npu 版本对齐 | post3 |
| 8 冒烟验证 | `x.mm(y)` 跑通 NPU 才算成功 | .npu() |
| 实操 | 全部真实命令汇总，可直接照抄 | — |
| 坑 | 高频报错与定位 | 版本错配 |

## 0. 一句话锚点

**升级 CANN = 在容器内把 `/usr/local/Ascend/ascend-toolkit/` 下的旧版本卸掉、装上新版本的 toolkit 与 kernels，再让上层 PyTorch（torch-npu）匹配新 CANN。** 宿主机的 driver/firmware 不动，所以是"轻量原地升级"。

```
旧: ascend-toolkit/7.0.0  +  torch-npu 旧      ← 容器内
                    │  uninstall.sh → install
                    ▼
新: ascend-toolkit/8.0.RC2 + kernels + torch-npu 2.1.0.post3
```

## 1. 地基：昇腾软件栈分层，谁在容器里谁在宿主机

要理解"为什么只升容器就行"，先把昇腾这套软件分层看清。它和 NVIDIA 的 `驱动 / CUDA / 框架` 三层是同构的：

```
┌─────────────────────────────────────────────┐
│  应用层   PyTorch 模型 / 训练脚本             │ 容器内
├─────────────────────────────────────────────┤
│  适配层   torch-npu (torch_npu)               │ 容器内 ← 本次重建
├─────────────────────────────────────────────┤
│  CANN     ┌ toolkit（编译器/ACL/aclnn 接口） │ 容器内 ← 本次升级
│           └ kernels（昇腾算子二进制 .o）      │ 容器内 ← 本次升级
├─────────────────────────────────────────────┤
│  Driver/Firmware  NPU 驱动、hccn_tool         │ 宿主机 ← 不动
├─────────────────────────────────────────────┤
│  硬件     Ascend 910B（达芬奇架构 Cube/Vector）│ 物理卡
└─────────────────────────────────────────────┘
```

| 类比维度 | NVIDIA | 昇腾 Ascend |
|---------|--------|-------------|
| 硬件 | GPU (SM) | NPU（达芬奇 Cube+Vector） |
| 驱动 | NVIDIA Driver | Ascend Driver/Firmware |
| 计算库 | CUDA Toolkit + cuDNN | CANN（toolkit + kernels） |
| 框架适配 | 原生 PyTorch | torch + **torch-npu** |
| 设备拷贝 | `.cuda()` | `.npu()` |

**为什么只升容器就够？** driver 在宿主机内核态，向上提供稳定 ABI；CANN 是用户态库，跑在容器里。只要容器里的 CANN 版本 ≤ 宿主机 driver 兼容范围，就能"宿主机不变、容器内换 CANN"。这正是用 Docker 做 AI-Infra 的最大价值——**升级隔离、可回滚**（旧容器还在，新容器另起一个 `pytorch_ubuntu_upgrade`）。

**三个必须对齐的版本**（错一个就跑不起来）：
- CANN ↔ driver：driver 必须 ≥ CANN 要求的最低版本
- toolkit ↔ kernels：同一 CANN 版本号（本例都是 `8.0.RC2.alpha001`）
- torch ↔ torch-npu：主版本必须一致（`torch 2.1.0` ↔ `torch-npu 2.1.0.post3`）

## 2. 拉取昇腾基础镜像（私有 ascendhub）

```bash
docker login -u 157xxx4031 ascendhub.huawei.com
docker pull ascendhub.huawei.com/public-ascendhub/ascend-mindspore:23.0.0-A2-ubuntu18.04
```

- `ascendhub.huawei.com` 是华为官方镜像仓库，需账号登录（用户名是手机号脱敏 `157xxx4031`）。
- 镜像 tag 拆解：`23.0.0` 镜像版本 / `A2` 对应 **Atlas A2 训练系列**（910B 属于 A2）/ `ubuntu18.04` 底座 OS。
- 注意：镜像名是 `ascend-mindspore`，但里面同样能装 PyTorch——基础镜像主要提供匹配的 CANN/driver 适配层和系统依赖，框架可自行替换。

## 3. 下载 CANN 安装包（toolkit + kernels 两个 .run）

版本历史页（用来挑版本、对照兼容矩阵）：`https://www.hiascend.com/zh/software/cann/community-history`

```bash
# kernels：昇腾算子二进制包（针对 910b 芯片）
wget -c https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C18B800TP015/Ascend-cann-kernels-910b_8.0.RC2.alpha001_linux.run

# toolkit：开发套件（编译器 ccec、ACL/aclnn 运行时、算子开发工具）
wget -c https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C18B800TP015/Ascend-cann-toolkit_8.0.RC2.alpha001_linux-aarch64.run
```

为什么是**两个包**？职责不同：

| 包 | 内容 | 类比 |
|----|------|------|
| `toolkit` | 图编译、ACL/aclnn API、profiling、算子编译工具链 | CUDA Toolkit |
| `kernels-910b` | 针对 910B 预编译好的算子库（`.o`） | cuDNN/算子库 |

细节：
- `-c` 表示**断点续传**（`.run` 包动辄几 GB，网络抖动可续）。
- 包名里 `aarch64` 说明这是 **ARM64** 架构——昇腾服务器（如鲲鹏 920）多为 ARM，**绝不能下成 x86_64 的包**。
- `kernels` 文件名带 `910b`，必须与你的实际芯片型号一致（用 `npu-smi info` 确认）。

## 4. 起一个干净的"升级容器"

为什么要新起一个容器而不是直接在老容器升？**为了可回滚**：升级有风险，新容器 `pytorch_ubuntu_upgrade` 出问题就删掉重来，老容器 `pytorch_ubuntu_dev` 始终保底。

```bash
docker stop pytorch_ubuntu_dev          # 停掉旧的开发容器（避免设备/端口冲突）
docker rm -f pytorch_ubuntu_upgrade     # 清掉可能残留的同名升级容器

docker run -it -u root \
  --name pytorch_ubuntu_upgrade \
  --network host \
  --shm-size 4G \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /etc/localtime:/etc/localtime \
  -v /var/log/npu/:/usr/slog \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /data/containerd/workspace/:/workspace \
  ascendhub.huawei.com/public-ascendhub/ascend-mindspore:23.0.0-A2-ubuntu18.04 \
  /bin/bash

docker start pytorch_ubuntu_upgrade
docker exec -it pytorch_ubuntu_upgrade bash
```

逐个参数说"为什么"：

| 参数 | 作用 | 不加会怎样 |
|------|------|-----------|
| `-u root` | 以 root 进入 | CANN 安装需写 `/usr/local/Ascend`，非 root 权限不足 |
| `--network host` | 容器直用宿主机网络栈 | 多机 HCCL 通信、RoCE/IB 走不通 |
| `--shm-size 4G` | 扩大共享内存 | PyTorch DataLoader 多 worker 会 `Bus error`（默认 64M 太小） |
| `-e ASCEND_VISIBLE_DEVICES=0..7` | 把 8 张 NPU 透传进容器 | 容器内看不到卡，`.npu()` 失败 |
| `-v /var/log/npu/:/usr/slog` | 把宿主机 NPU 日志目录映射进来 | 算子/driver 报错时拿不到 slog 日志，无法定位 |
| `-v /usr/bin/hccn_tool:...` | 复用宿主机网卡配置工具 | 无法在容器内查/配 NPU 网卡（hccn） |
| `-v /etc/localtime` | 时区与宿主机一致 | 日志时间戳错乱 |
| `-v /data/.../workspace/:/workspace` | 持久化工作目录 | 容器删了数据就没了 |

> 关键认知：`ASCEND_VISIBLE_DEVICES` 之于昇腾，等价于 `NVIDIA_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` 之于 NVIDIA——**设备隔离**靠它。配合 `ascend-docker-runtime` 才能把字符设备 `/dev/davinci*` 注入容器。

## 5. 卸旧 CANN、装新 toolkit（顺序不能乱）

```bash
# 5.1 卸载旧版本（路径里的 7.0.0 就是旧 CANN 版本号）
cd /usr/local/Ascend/ascend-toolkit/7.0.0/aarch64-linux/script/
./uninstall.sh

# 5.2 校验安装包完整性 → 安装新 toolkit
chmod +x Ascend-cann-toolkit_8.0.RC2.alpha001_linux-aarch64.run
./Ascend-cann-toolkit_8.0.RC2.alpha001_linux-aarch64.run --check
./Ascend-cann-toolkit_8.0.RC2.alpha001_linux-aarch64.run --install

# 5.3 加载新环境变量（每次新开 shell 都要 source）
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```

为什么是这个顺序：

```
uninstall.sh  ──► 删 7.0.0，避免新旧 toolkit 路径/软链冲突
   │
--check       ──► 先验包完整性（校验和/依赖），install 前发现坏包，省时间
   │
--install     ──► 落到 /usr/local/Ascend/ascend-toolkit/8.0.RC2/...
   │
set_env.sh    ──► 把 PATH/LD_LIBRARY_PATH/ASCEND_HOME 指向新版本
```

- `set_env.sh` 是关键：它导出 `LD_LIBRARY_PATH`、`PYTHONPATH`、`ASCEND_TOOLKIT_HOME` 等。**不 source 它，torch-npu 在 `import` 时就会因为找不到 `libascendcl.so` 等动态库而报错。**
- `--check` vs `--install`：先 check 再 install 是稳妥做法，等价于"先验货再下单"。

## 6. 安装 kernels 算子包

```bash
./Ascend-cann-kernels-910b_8.0.RC2.alpha001_linux.run --install --feature=aclnn_ops_train
```

- `--feature=aclnn_ops_train`：只装**训练所需的 aclnn 算子**。aclnn 是 CANN 的单算子执行接口（Ascend Computing Language Neural Network），torch-npu 调的就是这套。
- 训练选 `aclnn_ops_train`，纯推理可选推理算子集——按需安装能省镜像体积。
- **顺序硬约束**：kernels 必须在 toolkit 之后装，且两者版本号必须完全一致（都是 `8.0.RC2.alpha001`）。

装完再次确认环境变量已生效：

```bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 7. 重建 Python 环境（conda + torch + torch-npu）

CANN 升级后，旧的 torch-npu 往往不再兼容，**最干净的做法是新建一个 conda 环境**。

```bash
# 7.1 装 Miniconda 到工作目录（aarch64 版！）
bash Miniconda3-py39_24.4.0-0-Linux-aarch64.sh -p /workspace/installs/conda-upgrade
conda init
source ~/.bashrc

export PATH=/root/miniconda3/bin:$PATH   # 让 conda 进 PATH
conda list

# 7.2 建独立环境
conda create -n llm-dev python=3.9
conda activate llm-dev
```

```bash
# 7.3 装框架（torch 与 torch-npu 主版本必须对齐：2.1.0）
pip3 install torch==2.1.0
pip3 install pyyaml setuptools
pip3 install torch-npu==2.1.0.post3
pip3 install numpy attrs decorator psutil absl-py cloudpickle psutil scipy synr tornado
```

版本对齐表（这是整篇最容易踩雷的地方）：

| 组件 | 版本 | 约束来源 |
|------|------|---------|
| Python | 3.9 | 与 torch-npu 发行版匹配 |
| torch | 2.1.0 | torch-npu 要求**主版本严格一致** |
| torch-npu | 2.1.0.post3 | `post3` 是针对该 CANN 的补丁号 |
| CANN | 8.0.RC2.alpha001 | torch-npu 适配的 CANN 版本 |

> 记忆口诀：**torch 与 torch-npu 同主版本，torch-npu 与 CANN 同代**。`2.1.0.post3` 里 `2.1.0` 对齐 torch，`post3` 对齐 CANN。

## 8. 冒烟验证：NPU 上跑一次矩阵乘

```python
import torch
import torch_npu          # 必须显式 import，触发对 NPU 后端的注册

x = torch.randn(2, 2).npu()   # .npu() = 把张量从 host 拷到 NPU HBM
y = torch.randn(2, 2).npu()
z = x.mm(y)                   # 矩阵乘在 NPU 的 Cube 单元上执行

print(z)
```

这段为什么是**最小可信验证**：它一次性串起了全栈——
```
import torch_npu  →  注册 NPU 设备/算子   （torch-npu 装对了）
.npu()            →  host→device 数据搬运   （driver+CANN 通了）
.mm()             →  达芬奇 Cube 做 GEMM     （kernels 算子可用）
print(z)          →  device→host 回读        （端到端打通）
```
只要这四步无报错并打印出 2×2 结果，就证明 CANN 升级 + torch-npu 重建全部成功。**跑通这个再去跑大模型，否则白搭。**

运行前务必：`. set_env.sh` 已 source、`conda activate llm-dev` 已激活（见下节复现命令）。

## 实操：可直接照抄的全套命令

**(A) 依赖批量装 + 清缓存（瘦身镜像）**
```bash
pip install --no-cache-dir -r requirements-npu.txt && rm -rf ~/.cache/pip/* && conda clean -all
```
`--no-cache-dir` + 删 pip cache + `conda clean` 三连，是把升级后的镜像 commit 前**减小体积**的标准动作。

**(B) 每次重新进入升级好的容器的固定三步**
```bash
docker start pytorch_ubuntu_upgrade
docker exec -it pytorch_ubuntu_upgrade bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh   # ① 加载 CANN 环境
conda activate llm-dev                          # ② 激活 Python 环境
```
> 这两条（`set_env.sh` + `conda activate`）是**每次开新终端必做**，少一条就 `import torch_npu` 失败。

**(C) 给升级后的镜像打 tag（便于推送/版本管理）**
```bash
docker tag harbor.llm.io/base/llm-train-unify:v1-20240603 harbor.llm.io/base/llm-train-unify:v1-20240603
```
实践中第二个 tag 应改成新版本号（如 `:v2-cann8.0rc2`），再 `docker commit` 当前容器 → `docker push` 到内网 harbor，固化升级成果。

## 常见问题 / 坑

| 现象 | 根因 | 解决 |
|------|------|------|
| `import torch_npu` 报找不到 `.so` | 没 source `set_env.sh` | 先 `. /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `.npu()` 报无可用设备 | 容器没透传卡 / runtime 没装 | `docker run` 加 `ASCEND_VISIBLE_DEVICES`，确认 ascend-docker-runtime |
| `torch_npu` import 段错误/符号错 | torch 与 torch-npu 主版本不一致 | 严格 `torch==2.1.0` ↔ `torch-npu==2.1.0.post3` |
| kernels 装不上 / 算子缺失 | 先装 kernels 后装 toolkit，或版本号不一致 | 顺序：toolkit→kernels，版本号都 `8.0.RC2.alpha001` |
| 下错包装不上 | 下成 x86_64 / 芯片型号不符 | 必须 `aarch64` + `910b`（`npu-smi info` 确认） |
| DataLoader `Bus error` | 共享内存太小 | `docker run` 加 `--shm-size 4G` 或更大 |
| CANN 装上但跑算子崩 | 容器 CANN 版本 > 宿主机 driver 支持范围 | 查兼容矩阵，必要时先升宿主机 driver |
| 看不到 slog 日志难定位 | 没挂载 `/var/log/npu` | `-v /var/log/npu/:/usr/slog` |
| `wget` 中断要重下 | 没断点续传 | 加 `-c` 续传 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 硬件与生态：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/GPU工作原理]]
- 同目录昇腾基建：[[llm-localization/ascend/ascend-infra/达芬奇架构]] · [[llm-localization/ascend/ascend-infra/ascend-docker-runtime]] · [[llm-localization/ascend/ascend-infra/HCCL]] · [[llm-localization/ascend/ascend-infra/ascend-npu-smi]]
- 框架与训练：[[ai-framework/pytorch/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[llm-train/README]]
- 通信/网络：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/InfiniBand]]
- 推理侧延伸：[[llm-inference/README]] · [[llm-inference/vllm/README]]
