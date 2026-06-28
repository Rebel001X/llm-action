# 昇腾(Ascend)国产化栈 FAQ:常见问题与排错手册

> 昇腾 NPU 从环境搭建、训练、推理到迁移落地的高频问题与"避坑"速查。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 章节 | 你会得到什么 | 适合谁 |
| --- | --- | --- |
| 0. 一句话锚点 | 这份 FAQ 覆盖的边界 | 所有人 |
| 1. 昇腾栈定位 + 对标英伟达 | 软件栈分层 + 迁移心智图 | 从 CUDA 来的人 |
| 2. 环境与驱动类问题 | 驱动/固件/CANN/Docker 关系 | 装环境的人 |
| 3. 容器与 Docker Runtime | Ascend-Docker-Runtime 排错(含 runc 案例) | 运维/部署 |
| 4. 训练类问题 | MindFormers/ModelLink 常见坑 | 训练工程师 |
| 5. 推理类问题 | MindIE/量化 常见坑 | 推理工程师 |
| 6. HCCL 通信类问题 | 多卡/多机集合通信报错 | 分布式工程师 |
| 7. 性能与精度排查 | profiling、算子下沉、精度对齐 | 调优的人 |
| 迁移要点 | 从 GPU 迁到 NPU 的检查清单 | 迁移负责人 |
| 常见问题表 | 一句话速查 | 救火时 |

## 0. 一句话锚点

> **昇腾的所有"玄学问题",九成出在"层与层版本不匹配"上**:驱动/固件、CANN(含 toolkit + kernels)、框架(MindSpore/PyTorch+torch_npu)、套件(MindFormers/MindIE)这四层必须配套。把"它们之间的依赖关系"想清楚,大部分报错就有方向了。

本文不堆砌精确命令与版本号——**凡涉及具体命令、包名、版本号、镜像 tag、安装路径,一律以华为昇腾官方文档(Ascend 社区)为准**。这里讲的是"为什么会出这个问题、该往哪个方向查"。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 昇腾软件栈分层(自底向上)

```
┌──────────────────────────────────────────────────────────┐
│  应用 / 大模型                                            │
├──────────────────────────────────────────────────────────┤
│  套件层  MindFormers(训练) │ MindIE(推理) │ ModelLink   │
│          msmodelslim(量化)                                │
├──────────────────────────────────────────────────────────┤
│  框架层  MindSpore   │   PyTorch + torch_npu(昇腾适配插件)│
├──────────────────────────────────────────────────────────┤
│  异构计算架构 CANN                                        │
│   ├─ GE 图引擎 / ACL 应用接口                             │
│   ├─ 算子库(AI Core 算子, 对标 cuDNN/cuBLAS)            │
│   ├─ HCCL 集合通信库(对标 NCCL)                         │
│   └─ Runtime / Driver 接口                               │
├──────────────────────────────────────────────────────────┤
│  驱动 + 固件(Driver / Firmware)                         │
├──────────────────────────────────────────────────────────┤
│  硬件  昇腾 NPU(达芬奇架构: Cube + Vector + Scalar)     │
└──────────────────────────────────────────────────────────┘
```

> 关键认知:CANN ≈ "CUDA + cuDNN + cuBLAS + NCCL 的合体",是整个生态的底座。框架(PyTorch/MindSpore)通过它把算子下沉到 NPU。

### 1.2 昇腾 ↔ 英伟达 生态对照表(迁移心智图)

| 维度 | 英伟达生态 | 昇腾生态 | 说明 |
| --- | --- | --- | --- |
| 加速硬件 | GPU(A100/H100) | NPU(昇腾 910/310 系列) | NPU 是达芬奇架构,Cube 单元做矩阵 |
| 底层计算平台 | CUDA | CANN | 异构计算架构,生态底座 |
| 深度学习算子库 | cuDNN / cuBLAS | CANN 算子库(AI Core 算子) | 卷积/矩阵乘等高性能 kernel |
| 集合通信 | NCCL | HCCL | AllReduce/AllGather 等原语 |
| 通信硬件互联 | NVLink / NVSwitch | HCCS / 私有高速互联 | 卡间高带宽 |
| 原生框架 | PyTorch | MindSpore(原生) | 昇腾一等公民 |
| 框架适配插件 | —— | torch_npu | 让 PyTorch 跑在 NPU 上 |
| 训练大模型套件 | Megatron-LM / HF | MindFormers / ModelLink | 并行训练 + 模型库 |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | 高吞吐推理服务 |
| 量化工具 | GPTQ / AWQ / TensorRT 量化 | msModelSlim | 权重/激活量化 |
| 容器运行时 | nvidia-container-runtime | Ascend-Docker-Runtime | 把 NPU 挂进容器 |
| 设备查询工具 | nvidia-smi | npu-smi | 看卡状态/显存/温度 |
| 编程语言扩展 | CUDA C++ | Ascend C | 自定义算子开发 |

> 迁移者最该记的一行:**遇到任何"GPU 上怎么做"的问题,先在这张表里找到昇腾对应物,再去查它的文档**。心智模型对齐了,迁移就只剩工程细节。

## 2. 环境与驱动类问题

### 2.1 四层版本必须配套(最高频根因)

```
 Driver/Firmware  ──须匹配──▶  CANN(toolkit + kernels)
        │                            │
        ▼                            ▼
   npu-smi 能看到卡          框架(torch_npu / MindSpore)
                                     │
                                     ▼
                          套件(MindFormers / MindIE)
```

- **现象**:`import torch_npu` 报错、算子不支持、`npu-smi` 看不到卡、CANN 初始化失败。
- **本质**:四层之间存在严格的配套关系。torch_npu 的版本要对应某个 CANN 版本,CANN 又要求最低驱动版本。任何一层"超前"或"落后"都可能炸。
- **方向**:先确认驱动/固件装好(`npu-smi` 能正常出卡信息),再装 CANN,再装框架插件,最后装套件。每一步都对照官方"版本配套表"。具体版本对应关系以华为昇腾官方文档(Ascend 社区)为准。

### 2.2 环境变量没 source

- **现象**:命令找不到、ACL/CANN 库加载不到、`libascend*.so` not found。
- **本质**:CANN 安装后需要把其环境脚本 source 进当前 shell(设置 `LD_LIBRARY_PATH`、`PATH`、`ASCEND_HOME` 等)。新开终端或容器里没 source 就会"装了等于没装"。
- **方向**:把 CANN 的 `set_env` 脚本写进 `~/.bashrc` 或容器启动脚本。具体脚本路径以官方文档为准。

### 2.3 普通用户 vs root 权限

- **现象**:`npu-smi` 权限不足、设备 `/dev/davinci*` 打不开。
- **本质**:NPU 设备节点有属主/属组限制,普通用户需被加入相应用户组才能访问设备。

## 3. 容器与 Docker Runtime 类问题

### 3.1 Ascend-Docker-Runtime 的作用

它对标 `nvidia-container-runtime`:在容器启动时,自动把宿主机的 NPU 设备节点(`/dev/davinci*`)、驱动库、`npu-smi` 等挂载进容器,让容器内进程能直接用 NPU。**不装它,容器里看不到卡**。

```
  docker run ...
      │
      ▼
  Ascend-Docker-Runtime(hook)
      │  注入: /dev/davinci* + 驱动 .so + npu-smi
      ▼
  容器内可见 NPU ──▶ CANN/框架正常初始化
```

### 3.2 经典案例:runc owner not right(本仓库原始问题)

- **报错**:
  ```
  docker: Error response from daemon: failed to create shim task:
  OCI runtime create failed: ... ascend-docker-runtime did not
  terminate successfully: exit status 1: owner not right
  /usr/bin/runc 1000
  ```
- **本质**:报错里 `owner not right /usr/bin/runc 1000` 表示 `runc` 这个二进制的属主不是预期的 `root`,而是 UID 为 1000 的普通用户。Ascend-Docker-Runtime 作为 OCI hook 在创建容器时会校验/调用 `runc`,属主不正确就拒绝执行,容器起不来。
- **排查方向**:
  1. 查看属主:`ls -lah /usr/bin/runc`(若 owner 是普通用户而非 root 即命中)。
  2. 改回 root:`sudo chown root:root /usr/bin/runc`。
  3. 该问题常见于手动拷贝/覆盖 runc、或用错误用户解压安装包导致属主丢失。
- **延伸坑**:同理,Ascend-Docker-Runtime、`docker-runtime` 相关二进制的属主/权限也要正确;`/etc/docker/daemon.json` 里 runtime 配置项需指向正确的 ascend runtime,改完要重启 docker 守护进程。具体配置项以官方文档为准。

### 3.3 容器里看不到 NPU

- **现象**:容器内 `npu-smi` 报错或无卡。
- **本质**:启动时没经过 Ascend-Docker-Runtime,或没显式声明可见设备。
- **方向**:确认 daemon.json 已注册 ascend runtime 且 `docker run` 指定了该 runtime;确认通过环境变量/参数声明了容器可见的 NPU 卡。具体参数以官方文档为准。

## 4. 训练类问题(MindFormers / ModelLink)

### 4.1 这两个套件的定位

- **MindFormers**:基于 MindSpore 的大模型训练/微调套件,对标"Megatron-LM + HuggingFace"的合体——既有并行训练能力,又有模型库与配置化训练。
- **ModelLink**:面向 PyTorch(+torch_npu)生态的大模型训练,更贴近 Megatron 的并行范式(TP/PP/DP),便于从 Megatron 迁移。

```
  HF/Megatron 世界          昇腾世界
  ─────────────            ─────────────
  Megatron-LM    ───▶      MindFormers(MindSpore) / ModelLink(PyTorch)
  HF Trainer     ───▶      MindFormers 配置化训练
  TP/PP/DP 并行  ───▶      同名并行,经 HCCL 通信
```

### 4.2 常见坑

| 现象 | 本质 / 方向 |
| --- | --- |
| 权重加载维度对不上 | HF 权重需转成套件要求的格式(权重切分/命名映射),用官方转换脚本 |
| 并行配置与卡数不匹配 | TP×PP×DP 之积要等于总卡数;PP 时层数要能被整除 |
| 数据集格式不识别 | 需先用套件的预处理工具做 tokenize/打包成专用格式 |
| 算子不支持/精度异常 | 个别算子在 NPU 上未覆盖或需开混合精度白名单,查算子支持列表 |
| 显存 OOM | NPU 显存与 GPU 不同,batch/序列长/重计算(recompute)策略要重调 |

> 配置文件是"第一现场":并行度、序列长、recompute、优化器分片等都在 yaml 里,报错先回到配置对照官方样例。

## 5. 推理类问题(MindIE / 量化)

### 5.1 MindIE 定位

MindIE 是昇腾的推理引擎/服务化框架,对标 **TensorRT-LLM + vLLM**:做图优化、KV Cache 管理、连续批处理(continuous batching)、并对外暴露推理服务接口。

### 5.2 msModelSlim 量化定位

对标 GPTQ/AWQ 工具链:把 FP16 权重(及激活)量化到 INT8/W8A8 等,降低显存与带宽压力。量化后的权重需被 MindIE 正确加载。

### 5.3 常见坑

| 现象 | 本质 / 方向 |
| --- | --- |
| 模型加载失败 | 模型结构未被引擎支持,或权重格式/版本不匹配,查支持模型列表 |
| 量化后精度掉点 | 校准集不具代表性 / 敏感层未回退 FP16;调校准数据与 per-channel 策略 |
| 服务吞吐上不去 | batch、max-seq、KV Cache 块大小、并行度未调优 |
| 多卡推理通信报错 | 张量并行经 HCCL,回到第 6 节排查通信 |

## 6. HCCL 通信类问题(多卡/多机)

### 6.1 HCCL 是什么、对标谁

HCCL(Huawei Collective Communication Library)对标 **NCCL**,提供 AllReduce、AllGather、ReduceScatter、Broadcast 等集合通信原语,是数据并行/张量并行/流水并行的通信底座。卡间走 HCCS/私有高速互联(对标 NVLink),跨机走 RoCE 网络。

### 6.2 ranktable 与拓扑(高频根因)

```
  多机多卡训练
        │
        ▼
  ranktable(描述每个 rank → device_id / server_ip 的映射)
        │  HCCL 据此建立通信域、选择 ring/层级算法
        ▼
  AllReduce 等原语在正确拓扑上跑
```

- **现象**:HCCL 初始化超时、建链失败、rank 卡住不动。
- **本质**:ranktable(集群拓扑描述文件)写错——IP 不通、device_id 与物理卡对不上、rank 数与实际卡数不符。
- **方向**:逐一核对 ranktable 中每个 rank 的 server_ip / device_id;确认跨机网络互通(RoCE/网卡 up、MTU、防火墙);确认环境变量里的 rank/world_size 一致。具体字段格式以官方文档为准。

### 6.3 环算法心智(与 NCCL 同源)

AllReduce 在 ring 算法下:N 卡分 N 块,经 ReduceScatter(N-1 步求和)+ AllGather(N-1 步搬运)完成。带宽利用率与 NCCL 同理,**通信量与卡数弱相关**,所以单纯加卡不一定按比例提速,要关注 HCCS/RoCE 带宽是否成为瓶颈。

### 6.4 常见坑

| 现象 | 本质 / 方向 |
| --- | --- |
| 建链 timeout | ranktable IP/端口不通,或超时阈值太小 |
| 单机能跑多机挂 | 跨机网络(RoCE)配置问题,先 ping + 测带宽 |
| 部分 rank 卡死 | 各 rank 进程未对齐(数据量/step 不一致导致 hang) |
| 通信慢 | 走了低速链路;确认 HCCS/卡间拓扑被正确识别 |

## 7. 性能与精度排查

- **算子下沉/图模式**:昇腾推荐图模式(整图编译下沉)以减少 host-device 交互。性能差时先确认是否落到了"动态图逐算子下发",频繁的小算子下发会让 host 成瓶颈。
- **Profiling**:用昇腾 profiling 工具看 AI Core 利用率、算子耗时、通信占比,定位是 compute-bound 还是 comm-bound。具体工具命令以官方文档为准。
- **精度对齐**:从 GPU 迁来精度有偏差时,逐层对比(loss/中间张量),重点查混合精度白名单、个别算子实现差异、随机种子与初始化。
- **达芬奇架构利用率**:矩阵计算交给 Cube 单元最高效,逐元素运算走 Vector 单元。算子若没用上 Cube,MFU 会很低。

## 迁移要点 / 注意事项与坑(GPU → NPU 检查清单)

1. **先建心智映射**:照第 1.2 节对照表,把你用的每个 GPU 组件找到昇腾对应物。
2. **设备 API 替换**:`cuda()` → `npu()`、`torch.cuda.*` → `torch_npu` 对应接口;`nccl` 后端 → `hccl` 后端。
3. **版本配套优先**:装环境时严格按"驱动→CANN→框架→套件"顺序与配套表,别跨版本混装。
4. **算子覆盖度核对**:迁移前查算子支持列表,自定义/冷门算子可能要用 Ascend C 重写或走 CPU 回退。
5. **并行配置重标定**:NPU 显存与 GPU 不同,batch/recompute/并行度都要重新调。
6. **通信改 HCCL**:分布式后端从 NCCL 切到 HCCL,补 ranktable;多机先验证网络。
7. **精度白名单**:混合精度下注意哪些算子需保 FP32,逐层对齐 loss。
8. **容器化**:用 Ascend-Docker-Runtime 而非 nvidia runtime;注意 runc/设备节点权限(见 3.2)。
9. **别造命令**:任何具体命令/版本/路径都回官方文档核对,这是省时间不是浪费时间。

## 常见问题(速查表)

| 问题 | 一句话定位 |
| --- | --- |
| `import torch_npu` 失败 | 八成版本不配套或没 source CANN 环境 |
| `npu-smi` 看不到卡 | 驱动/固件没装好或权限/用户组问题 |
| 容器里没有 NPU | 没用 Ascend-Docker-Runtime 或没声明可见设备 |
| runc owner not right | `chown root:root /usr/bin/runc`(见 3.2) |
| HCCL 初始化超时 | ranktable 写错或跨机网络不通 |
| 权重加载维度对不上 | HF 权重未转成套件格式 |
| 量化后掉点 | 校准集不行 / 敏感层未回退 FP16 |
| 性能上不去 | 没走图模式 / Cube 利用率低 / 通信瓶颈 |
| 精度对不上 | 混合精度白名单 + 个别算子实现差异 |
| 多机网络慢 | RoCE 配置 / 走了低速链路 |

> 终极口诀:**报错先定位"在哪一层"(驱动/CANN/框架/套件/通信),再查"是不是版本不配套",最后回官方文档核命令**。九成问题到这一步就有解了。

## 🔗 跳转链接

- 知识地图:[[00-知识地图]]
- 硬件与算力:[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/CUDA]]
- 网络与通信:[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- 训练框架:[[ai-framework/megatron-lm/README]] · [[ai-framework/huggingface-transformers/README]]
- 压缩与量化:[[llm-compression/quantization/量化基础]]
- 上层主题:[[llm-inference/README]] · [[llm-train/README]] · [[llm-algo/transformer/模型架构]]
