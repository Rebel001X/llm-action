export const meta = {
  name: 'llmaction-补齐-run6',
  description: 'llm-action补齐run6: 28篇(infra网络/训练分布式/应用数据eval/maas/llmops)填到最大细节+双链',
  phases: [
    { title: 'W11-Infra网络与训练分布式', detail: '通信软件/NCCL-test/FSDP/PyTorch分布式/Megatron 14篇' },
    { title: 'W12-应用数据eval与LLMOps', detail: 'Agent/RAG问题/语料/压测/MaaS/K8s 14篇' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'
const SCHEMA = { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'number' }, hasHandCalc: { type: 'boolean' }, hasAscii: { type: 'boolean' }, links: { type: 'number' } }, required: ['file', 'lines', 'hasHandCalc', 'hasAscii', 'links'] }

function buildPrompt(n) {
  const rel = n.related.map(r => `[[${r}]]`).join(' ')
  return `你是顶尖 AI-Infra 讲师。把 llm-action 仓库里一个【薄文件】补齐到【最大细节】，用【中文】，并用 Write 工具覆盖写回原路径。

【文件(相对仓库根)】${n.file}
【主题】${n.title}
【必须覆盖】${n.cover}
【相关链接】${rel}

【第一步】先 Read ${DIR}${n.file}，尊重已有方向;近空则从零写。
【第二步】用 Write 写到绝对路径：${DIR}${n.file}

【风格——"从最底层讲清 + 可视化"】
- 不假设记得公式;概念拆到最原子;每步说"为什么"。
- 【必含 ASCII 可视化图】(架构/数据流/拓扑/流程)。
- 适用就给【数值例子/手算】(带宽/通信量/显存/吞吐等);硬件/工具类给典型公开数字并标"约/以官方为准"。
- 公式行内 $...$、独立 $$...$$。
- 工具/框架/配置类: 讲清【是什么、解决什么、核心机制、关键参数的含义与权衡(讲含义不背具体默认值)、与同类对比、实践要点】;【不编造】精确版本号/CLI默认值, 不确定标"以官方文档为准"。

【固定结构】
# ${n.title}
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：${rel}
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置
## 2.~N. 逐步拆解(每节配ASCII图)
## (单独一节)数值例子/对照/实践
## 常见问题(表格)
## 🔗 跳转链接

【硬要求】230~420 行, 高密度; 顶部导航+底部双链必须有。返回 JSON {file,lines,hasHandCalc,hasAscii,links}。直接开始。`
}

const W11 = [
  { file: 'ai-infra/网络/通信软件.md', title: '分布式通信软件栈', cover: 'NCCL/MPI/Gloo/UCX分工;集合通信库vs点对点;PyTorch后端选择;何时用哪个;与硬件(IB/NVLink)关系', related: ['00-知识地图','ai-infra/网络/NCCL','ai-infra/网络/集合通信原语','ai-infra/网络/IB软件'] },
  { file: 'ai-infra/网络/HPC性能测试.md', title: 'HPC/网络性能测试', cover: '带宽/时延/消息率测试;perftest(ib_send_bw);osu-microbenchmark;all-reduce带宽;瓶颈定位;指标解读', related: ['00-知识地图','ai-infra/网络/nccl-test-集合通讯的性能测试','ai-infra/网络/InfiniBand'] },
  { file: 'ai-infra/网络/nccl-test-集合通讯的性能测试.md', title: 'NCCL-Tests', cover: 'all_reduce_perf等;busbw vs algbw区别与公式;如何判断NVLink/IB是否跑满;多机测试;调优诊断', related: ['00-知识地图','ai-infra/网络/NCCL','ai-infra/网络/集合通信原语','ai-infra/网络/HPC性能测试'] },
  { file: 'ai-infra/网络/IB软件.md', title: 'InfiniBand 软件栈', cover: 'OFED驱动;verbs API;perftest工具;subnet manager;GPUDirect RDMA;ibstat/ibstatus诊断;与NCCL关系', related: ['00-知识地图','ai-infra/网络/InfiniBand','ai-infra/网络/通信软件'] },
  { file: 'ai-infra/ai-hardware/NIXL.md', title: 'NIXL (推理传输库)', cover: 'NVIDIA推理数据传输库;为PD分离/KV传输优化;统一内存/网络/存储传输抽象;与Dynamo/分离架构关系', related: ['00-知识地图','llm-inference/PD分离','llm-inference/Mooncake','ai-infra/ai-hardware/GPU-network'] },
  { file: 'ai-infra/ai-hardware/GPU相关环节变量.md', title: 'GPU 相关环境变量', cover: 'CUDA_VISIBLE_DEVICES/NCCL_*/PYTORCH_CUDA_ALLOC_CONF等常见环境变量含义与用途;调试/性能/显存;讲含义不背全量', related: ['00-知识地图','ai-infra/ai-hardware/CUDA','ai-infra/网络/NCCL'] },
  { file: 'ai-infra/存储/README.md', title: 'AI 存储总览', cover: '训练数据/checkpoint/KV/日志的存储需求;并行文件系统/对象存储/本地NVMe三层;带宽与IOPS;数据加载流水线', related: ['00-知识地图','ai-infra/存储/存储','ai-infra/存储/nvme-ssd'] },
  { file: 'ai-infra/communication.md', title: 'AI 通信总览', cover: '为什么通信是分布式瓶颈;层级(机内NVLink/机间IB);集合通信原语;通信量分析;计算通信重叠;拓扑', related: ['00-知识地图','ai-infra/网络/集合通信原语','ai-infra/ai-hardware/GPU-network','llm-optimizer/计算通信重叠'] },
  { file: 'ai-framework/huggingface-transformers/FSDP.md', title: 'FSDP (完全分片数据并行)', cover: 'FSDP原理(参数/梯度/优化器状态分片,类似ZeRO-3);前向all-gather反向reduce-scatter;与DDP/ZeRO对比;wrap策略;显存账', related: ['00-知识地图','ai-framework/deepspeed/README','ai-framework/pytorch/README'] },
  { file: 'ai-framework/huggingface-transformers/API.md', title: 'Transformers 核心 API', cover: 'AutoModel/AutoTokenizer/AutoConfig;from_pretrained/save_pretrained;generate参数语义;pipeline;Trainer要点(讲语义不背签名)', related: ['00-知识地图','ai-framework/huggingface-transformers/README','llm-inference/解码策略'] },
  { file: 'ai-framework/cuda/README.md', title: 'CUDA 生态(框架视角)', cover: 'CUDA Toolkit/驱动/cuDNN/cuBLAS/NCCL层次;版本兼容(driver vs runtime);PyTorch如何调用;容器化', related: ['00-知识地图','ai-infra/ai-hardware/CUDA','ai-infra/算力/GPU工作原理'] },
  { file: 'llm-train/pytorch/distribution/README.md', title: 'PyTorch 分布式训练', cover: 'torch.distributed;进程组/rank/world_size;DDP原理(梯度bucket+AllReduce重叠);init_process_group;launch方式', related: ['00-知识地图','ai-framework/pytorch/README','llm-train/pytorch/distribution/多机多卡','ai-infra/网络/集合通信原语'] },
  { file: 'llm-train/pytorch/distribution/多机多卡.md', title: '多机多卡训练', cover: '多机组网;rank分配;NCCL环境变量;rendezvous;故障排查(连不上/慢);带宽对吞吐影响;实践', related: ['00-知识地图','llm-train/pytorch/distribution/README','ai-infra/网络/InfiniBand','ai-infra/ai-cluster/README'] },
  { file: 'llm-train/megatron/README.md', title: 'Megatron 训练实战', cover: 'Megatron-LM训练流程;TP/PP/DP配置;数据预处理(indexed dataset);checkpoint;与Megatron-DeepSpeed区别;典型坑', related: ['00-知识地图','ai-framework/megatron-lm/README','llm-train/megatron-deepspeed/README','llm-train/README'] },
]

const W12 = [
  { file: 'llm-application/README.md', title: 'LLM 应用总览', cover: '应用模式(对话/RAG/Agent/工具);三种知识注入(微调/RAG/prompt);工程链路;评估与迭代;落地考量', related: ['00-知识地图','llm-application/应用场景','llm-application/rag/README','llm-application/agent/OpenCode/README'] },
  { file: 'llm-application/agent/OpenCode/README.md', title: 'Coding Agent (OpenCode类)', cover: 'AI编程Agent架构(规划/工具调用/编辑/执行/反馈);ReAct循环;上下文管理;与Claude Code类对比;关键能力', related: ['00-知识地图','llm-application/应用场景','llm-application/README'] },
  { file: 'llm-application/rag/存在的一些问题.md', title: 'RAG 的问题与优化', cover: '召回不准/上下文过长/幻觉残留/多跳推理弱/时效;对应优化(重排/query改写/父子块/GraphRAG/Agentic RAG)', related: ['00-知识地图','llm-application/rag/README','llm-application/rag/方案'] },
  { file: 'llm-application/pre-post-handle/README.md', title: '推理前后处理', cover: 'prompt模板/敏感词过滤/格式化;输出解析/JSON修复/安全过滤;流式处理;为什么需要;工程要点', related: ['00-知识地图','llm-inference/GuidedGeneration','llm-application/README'] },
  { file: 'llm-data-engineering/dataset/chinese-corpus-all.md', title: '中文预训练语料', cover: '常见中文语料(WuDao/CLUECorpus/中文Common Crawl);清洗去重;繁简/编码;质量;配比;与英文混合', related: ['00-知识地图','llm-data-engineering/dataset/README','llm-data-engineering/README'] },
  { file: 'llm-data-engineering/sft-dataset/evol-instruct.md', title: 'Evol-Instruct (指令进化)', cover: 'WizardLM的指令进化;深度进化(加约束/推理)+广度进化(新主题);为什么提升指令复杂度;自动构造SFT数据;质量控制', related: ['00-知识地图','llm-data-engineering/sft-dataset/数据格式设计','llm-data-engineering/README'] },
  { file: 'llm-eval/llm-performance/vllm-benchmark.md', title: 'vLLM 压测', cover: 'benchmark_serving;请求率/并发;TTFT/TPOT/吞吐;ShareGPT数据;如何画吞吐-延迟曲线;调参(讲方法不背CLI)', related: ['00-知识地图','llm-eval/llm-performance/推理性能测试','llm-inference/vllm/README','llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释'] },
  { file: 'llm-eval/llm-performance/perfetto.md', title: 'Perfetto 性能分析', cover: 'trace可视化;timeline看kernel/通信/气泡;PyTorch profiler导出;如何定位瓶颈(GPU空闲/通信暴露);分析方法', related: ['00-知识地图','llm-eval/llm-performance/推理性能测试','ai-infra/算力/GPU工作原理'] },
  { file: 'llm-maas/OpenAI-ChatGPT.md', title: 'MaaS / OpenAI API', cover: 'Model-as-a-Service;OpenAI兼容API(chat/completions/embeddings);流式SSE;function calling;token计费;兼容生态(vLLM也实现)', related: ['00-知识地图','llm-maas/README','llm-inference/vllm/README'] },
  { file: 'llm-maas/README.md', title: 'MaaS 模型即服务总览', cover: 'MaaS概念;API网关/限流/计费/多租户;OpenAI兼容标准;私有化vs云;路由多模型;可观测', related: ['00-知识地图','llm-maas/OpenAI-ChatGPT','llmops/模型推理平台方案'] },
  { file: 'llmops/kubernetes.md', title: 'Kubernetes for AI', cover: 'K8s调度GPU(device plugin);Pod/Deployment/Service;GPU共享/MIG;训练用Volcano/Kubeflow;推理用KServe;为什么用K8s', related: ['00-知识地图','llmops/README','llmops/模型推理平台方案','ai-infra/ai-cluster/README'] },
  { file: 'llmops/模型推理平台方案.md', title: '模型推理平台方案', cover: '推理平台架构(网关/调度/引擎/监控);多模型多版本;弹性扩缩;灰度;与vLLM/TRT-LLM/K8s集成;SLO保障', related: ['00-知识地图','llmops/kubernetes','llm-inference/README','llm-maas/README'] },
  { file: 'llm-algo/llama/README.md', title: 'LLaMA 模型说明', cover: 'LLaMA各代规模/许可/特点;权重获取;tokenizer(SentencePiece BPE);与生态(Alpaca/Vicuna)关系;部署', related: ['00-知识地图','llm-algo/llama/模型架构','llm-algo/llama'] },
  { file: 'llm-algo/qwen/参数说明及函数说明.md', title: 'Qwen 配置与参数详解', cover: 'Qwen config关键字段(hidden/heads/layers/kv heads/rope theta/vocab)含义;如何从config算参数量/显存;generate参数;讲含义', related: ['00-知识地图','llm-algo/qwen/README','llm-algo/FLOPs'] },
]

const runWave = (notes, phaseTitle) =>
  parallel(notes.map(n => () => agent(buildPrompt(n), { label: `fill:${n.file}`, phase: phaseTitle, schema: SCHEMA })))

phase('W11-Infra网络与训练分布式')
const r11 = await runWave(W11, 'W11-Infra网络与训练分布式')
log(`W11 完成: ${r11.filter(Boolean).length}/${W11.length}`)
phase('W12-应用数据eval与LLMOps')
const r12 = await runWave(W12, 'W12-应用数据eval与LLMOps')
log(`W12 完成: ${r12.filter(Boolean).length}/${W12.length}`)

const all = [...r11, ...r12].filter(Boolean)
return { total: W11.length + W12.length, written: all.length, files: all.map(x => x.file) }
