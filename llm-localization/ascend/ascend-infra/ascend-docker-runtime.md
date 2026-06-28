# Ascend Docker Runtime(昇腾容器运行时)

> 让容器"看见"并正确使用昇腾 NPU 的 OCI 运行时挂钩:自动注入设备节点与驱动依赖,免去逐个 `--device`/`-v` 手动挂载。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[llm-inference/README]] [[llm-train/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|--------------|--------|
| 0 | 一句话锚点 | NPU 容器化的"最后一公里" |
| 1 | 在昇腾栈的定位 + 昇腾↔英伟达对照表 | OCI runtime、nvidia-docker 对标 |
| 2 | 容器要跑 NPU 缺什么(问题拆解) | 设备节点、驱动 so、ACL 路径 |
| 3 | 工作机制:OCI prestart hook 全流程(ASCII 图) | runc、hook、注入 |
| 4 | 与 K8s / MindCluster / Device Plugin 的关系 | 设备发现、调度、隔离 |
| 5 | 安装与使用流程的"含义"(不背命令) | daemon.json、default-runtime |
| 6 | 从 nvidia-docker 迁移要点与常见坑 | 驱动版本、挂载、可见性 |
| 7 | 常见问题(表格) | 排错速查 |

## 0. 一句话锚点

**Ascend Docker Runtime 是昇腾对标 `nvidia-container-runtime` 的那一层。** 它本质上是一个符合 OCI 规范的容器运行时封装:在容器启动的极早期(prestart hook 阶段),把宿主机上的 NPU 设备节点、驱动动态库、CANN/驱动相关路径"按需注入"到容器里,使容器内的程序无需感知宿主机的具体设备布局,就能像在裸机上一样调用昇腾 NPU。

没有它,你也能用容器跑 NPU——但要手动把一大堆 `/dev/davinciX`、`/dev/davinci_manager`、`/usr/local/Ascend/driver/...` 通过 `--device` 和 `-v` 一个个挂进去,极其繁琐且易错。Ascend Docker Runtime 就是把这件"脏活"自动化、标准化。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 它处在软件栈的哪一层

昇腾软件栈自底向上大致是:**硬件(NPU)→ 驱动(driver/firmware)→ CANN(异构计算架构,含运行时/算子库)→ 训练/推理框架(MindSpore、PyTorch+torch_npu)→ 套件(MindFormers、MindIE 等)**。

Ascend Docker Runtime **不属于这条"计算"主链**,而是横在"驱动"与"容器化部署"之间的**基础设施胶水层**。它不参与算子计算,只负责"把驱动和设备正确地搬进容器"。可以理解为:它服务的是"如何在容器里复用宿主机已装好的驱动",而 CANN/框架才是真正干活的。

```
           容器内应用 (训练/推理: torch_npu, MindIE ...)
                          │  需要访问 NPU
                          ▼
        ┌───────────────────────────────────────┐
        │   容器边界(namespace 隔离)            │
        │   容器内:CANN + 框架(镜像里自带)     │
        └───────────────────────────────────────┘
                          ▲  注入设备节点 + 驱动 .so
        ┌─────────────────┴─────────────────────┐
        │   Ascend Docker Runtime (OCI hook)     │ ← 本文主角
        └─────────────────┬─────────────────────┘
                          ▼
   宿主机:NPU 驱动 / firmware / /dev/davinci* / 驱动库路径
                          ▼
                     昇腾 NPU 硬件
```

> 关键认知:**驱动装在宿主机,CANN 装在镜像。** Runtime 的职责是把"宿主机的驱动"映射给"镜像里的 CANN"用。镜像通常**不**打包驱动(驱动与内核/硬件强相关),这点和英伟达世界完全一致。

### 1.2 昇腾 ↔ 英伟达生态对照表(迁移心智图)

| 维度 | 英伟达(CUDA 世界) | 昇腾(Ascend 世界) | 说明 |
|------|---------------------|----------------------|------|
| 加速器 | GPU | NPU(达芬奇架构) | 计算硬件 |
| 设备节点 | `/dev/nvidiaX`、`/dev/nvidiactl` | `/dev/davinciX`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` | 容器需要的设备文件 |
| 容器运行时 | **nvidia-container-runtime** | **Ascend Docker Runtime** | 本文主角的对标物 |
| 底层注入工具 | libnvidia-container / nvidia-container-cli | 昇腾自带的注入逻辑(hook) | 真正执行"注入"的部分 |
| 设备可见性变量 | `NVIDIA_VISIBLE_DEVICES` | `ASCEND_VISIBLE_DEVICES`(MindCluster/插件场景) | 控制哪几张卡进容器 |
| K8s 设备插件 | NVIDIA k8s-device-plugin | **Ascend Device Plugin**(MindCluster/MindX DL) | K8s 里暴露/调度卡 |
| 异构计算架构 | CUDA | **CANN** | 运行时 + 算子库 |
| 算子/数学库 | cuDNN / cuBLAS | CANN 内置算子库(如 AOL/算子加速库) | 高性能算子 |
| 集合通信 | NCCL | **HCCL** | 多卡通信 |
| 训练框架 | PyTorch / Megatron-LM | MindSpore / PyTorch+torch_npu / MindFormers | 上层框架 |
| 推理引擎 | TensorRT-LLM / vLLM | **MindIE** | 推理服务 |

> 一句话迁移口诀:**`nvidia-docker` 之于 GPU,等于 `Ascend Docker Runtime` 之于 NPU。** 你在英伟达侧靠 `--gpus` / `--runtime=nvidia` 拿到 GPU,在昇腾侧靠这个 runtime 拿到 NPU。

## 2. 为什么需要它:容器要跑 NPU,到底缺什么

把一个普通容器跑 NPU,至少缺三类东西:

1. **设备节点(device nodes)**:容器默认是看不到宿主机 `/dev` 下设备的。NPU 对应一组字符设备:每张卡的 `davinciX`、全局的 `davinci_manager`、内存管理 `devmm_svm`、HDC 通道 `hisi_hdc` 等。少一个,运行时初始化就会失败。

2. **驱动动态库(driver libs)**:CANN 在容器内运行时要 dlopen 宿主机驱动提供的 `.so`(如驱动层的 runtime/dcmi 等库)。这些库**版本必须与宿主机驱动匹配**,因此不能打进镜像,只能从宿主机挂进来。

3. **相关路径与环境**:驱动安装目录、`ld.so` 搜索路径、必要的环境变量等,要让容器内的加载器能找到上面的库。

手动方案是写一长串 `--device=/dev/davinci0 --device=/dev/davinci_manager ... -v /usr/local/Ascend/driver:...`,**卡数越多越痛苦,且不同机型设备名不同**。Ascend Docker Runtime 把"该注入哪些设备、哪些库、哪些路径"做成自动化规则,你只需声明"我要哪几张卡"。

## 3. 工作机制:OCI prestart hook 全流程

容器启动遵循 OCI 规范:`docker`/`containerd` 生成容器配置 → 调用底层 `runc` → `runc` 在创建容器进程前后会执行配置里登记的 **hooks**。Ascend Docker Runtime 的核心,就是把自己挂在 `runc` 之前(作为 `runc` 的 shim/包装),并往容器配置里**注入一个 prestart hook**;这个 hook 在容器 rootfs 已就绪、用户进程尚未启动的窗口里,完成设备与库的注入。

```
 docker run --runtime=ascend ...        (或 default-runtime=ascend)
        │
        ▼
 ┌──────────────────────────┐
 │ Ascend Docker Runtime     │  ← 替代/包装 runc 的运行时
 │  1) 读取请求的可见设备     │     (环境变量 / 设备挂载请求)
 │  2) 改写 OCI spec:        │
 │       + prestart hook      │
 └───────────┬──────────────┘
             ▼
        调用真正的 runc
             │
   ┌─────────┴──────────┐
   │ 创建容器 namespace   │
   │ 准备 rootfs          │
   └─────────┬──────────┘
             ▼  (容器进程启动前)
 ┌──────────────────────────┐
 │  prestart hook 执行:      │
 │   • mknod / bind 设备节点  │  /dev/davinci*, davinci_manager ...
 │   • 挂载驱动 .so 与路径    │  宿主驱动库 → 容器内
 │   • 设置加载器搜索路径     │
 └───────────┬──────────────┘
             ▼
        启动容器内用户进程  → CANN 初始化成功 → 看见 NPU
```

要点:**注入发生在用户进程启动之前**,所以容器内程序"一睁眼"NPU 环境就已就绪;而且注入只针对本次请求的设备,实现了一定程度的卡级隔离(配合 cgroup/设备白名单)。

> 具体的设备节点清单、注入库列表、是按整卡还是按 vNPU(虚拟化切分)注入,会随机型(训练卡/推理卡)、驱动版本、是否启用算力切分而不同,**以华为昇腾官方文档(Ascend 社区)为准**。

## 4. 与 K8s / MindCluster / Device Plugin 的协作

单机 `docker run` 只是入门。生产更多是 K8s 集群:

- **Ascend Device Plugin**:向 kubelet 上报"本节点有几张可用 NPU",让调度器能像调度 `nvidia.com/gpu` 那样调度 `huawei.com/Ascend910` 之类资源。
- **Ascend Docker Runtime**:Device Plugin 决定"哪张卡给哪个 Pod"后,**真正把那张卡注入容器**的还是这个 runtime。两者分工:**插件管调度与可见性,runtime 管落地注入**。
- **MindCluster(原 MindX DL)**:华为的集群调度/训练管理方案,集成了上述组件,还提供故障重调度、断点续训等能力。

类比英伟达:Device Plugin ↔ k8s-device-plugin,Ascend Docker Runtime ↔ nvidia-container-runtime,MindCluster ↔ GPU Operator + 训练 operator 生态。

## 5. 安装与使用流程的"含义"(讲为什么,不背命令)

下面只讲**步骤的意图和依赖关系**;凡涉及确切命令、包名、版本号、路径,一律**以华为昇腾官方文档(Ascend 社区)为准**。

1. **前置:宿主机先装好 NPU 驱动与固件。** 这是地基——runtime 注入的就是这套驱动。驱动没装好,后面一切免谈。(对应英伟达侧:宿主机先装 GPU driver。)

2. **安装 Ascend Docker Runtime 组件。** 把这个 OCI runtime 二进制部署到节点,并在容器引擎里**注册**它为一个可用 runtime。

3. **在容器引擎配置中登记 runtime(daemon.json 类配置)。** 含义:告诉 Docker/containerd "有个叫 ascend 的 runtime 可用"。可选把它设为 **default-runtime**,这样**所有容器默认就能用 NPU**,省去每次显式指定——这是很多昇腾镜像教程默认的做法,但也意味着该节点所有容器都走这条路径,需评估影响。

4. **重启容器引擎使配置生效。** 配置类改动几乎都需要 reload/restart daemon 才被读取——这是最常见的"我配了怎么不生效"的根因。

5. **运行容器时声明所需设备。** 通过指定该 runtime,并(在插件/MindCluster 场景)用可见设备变量声明要哪几张卡。容器内即可 `npu-smi`/框架检测到 NPU。

6. **验证。** 进容器跑设备查询工具,确认能列出 NPU 且数量正确,再跑一个最小训练/推理 smoke test。

## 6. 从 nvidia-docker 迁移要点与常见坑

**迁移要点(心智对应):**

- `--runtime=nvidia` ⟶ `--runtime=ascend`(名称以官方为准);或设 default-runtime 免去每次指定。
- `NVIDIA_VISIBLE_DEVICES=0,1` ⟶ `ASCEND_VISIBLE_DEVICES=0,1`(在支持的插件/调度场景)。
- "镜像不装 driver、只装 CUDA toolkit" ⟶ "镜像不装 driver、只装 CANN"。心智完全一致。
- K8s 资源名 `nvidia.com/gpu` ⟶ `huawei.com/Ascend910`(以实际机型/插件为准)。

**高频坑:**

| 坑 | 现象 | 根因/对策 |
|----|------|-----------|
| 驱动版本与镜像 CANN 不匹配 | 容器内初始化报版本/符号错误 | 镜像里的 CANN 要与宿主驱动版本**配套**;查官方版本配套表 |
| 配了 daemon.json 没重启引擎 | runtime 不可见、容器报 unknown runtime | reload/restart 容器引擎 |
| 设备节点缺失 | 容器内看不到卡或看到一半 | 确认本次请求的可见设备正确;某些机型设备名不同 |
| 直接 `--privileged` 硬怼 | 能跑但失去隔离、不规范 | 优先用 runtime 的按需注入,而非特权容器 |
| 镜像里误打包了驱动 .so | 与宿主驱动冲突 | 驱动只来自宿主注入,镜像别带 |
| 多卡训练 HCCL 不通 | 通信初始化卡住 | 除设备注入外,需保证 HCCL 所需设备/网络(如 RDMA、hccn 配置)也在;参见 HCCL 文档 |
| vNPU/算力切分场景设备名变化 | 注入逻辑不同 | 切分场景按官方文档单独配置 |

**性能/调优心智(机制层面,非具体数字):** runtime 本身只在启动期注入,**不影响运行期算力**;真正的吞吐取决于 CANN 算子、图模式(下沉到 NPU 的图执行)、HCCL 通信拓扑与 batch/并行策略。容器化不应引入算子层开销;若发现性能异常,先排查"是否落到了正确机型的算子库""HCCL 是否走了高速链路",而非怀疑 runtime。

## 常见问题

| 问题 | 简答 |
|------|------|
| 它和 CANN 是一回事吗? | 不是。CANN 是计算架构(算子/运行时),装在镜像里;本 runtime 只负责把宿主驱动注入容器。 |
| 不用它能跑 NPU 吗? | 能,但要手动挂一长串设备和库,易错;它把这件事自动化。 |
| 驱动该装宿主机还是镜像? | **宿主机**。驱动与内核/硬件强相关,镜像只装 CANN。和英伟达侧一致。 |
| 对标英伟达的什么? | `nvidia-container-runtime` / nvidia-docker。 |
| K8s 里谁决定卡分给谁? | Ascend Device Plugin 调度,本 runtime 负责落地注入。 |
| 为什么配了不生效? | 多半是改了 daemon.json 没重启容器引擎。 |
| 具体命令/版本去哪查? | 华为昇腾官方文档(Ascend 社区)与 Gitee `ascend/ascend-docker-runtime` 仓库。 |

## 🔗 跳转链接

- 枢纽:[[00-知识地图]]
- 算力硬件:[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/CUDA]]
- 通信:[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- 框架与套件:[[ai-framework/megatron-lm/README]] · [[ai-framework/huggingface-transformers/README]]
- 上下游:[[llm-inference/README]] · [[llm-train/README]] · [[llm-compression/quantization/量化基础]] · [[llm-algo/transformer/模型架构]]

> 参考源(以官方为准):Gitee `ascend/ascend-docker-runtime`;华为昇腾社区 MindX DL / MindCluster 安装与组件参考文档。
