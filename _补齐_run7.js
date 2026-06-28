export const meta = {
  name: 'llmaction-补齐-run7',
  description: 'llm-action补齐run7: 28篇(推理工具源码/训练工具/压缩工具)填到最大细节+双链',
  phases: [
    { title: 'W13-推理工具与源码', detail: 'vLLM源码/SGLang/LightLLM/Xinference/TensorRT/部署 14篇' },
    { title: 'W14-训练工具与压缩工具', detail: 'PyTorch训练/torchrun/各微调项目/DeepSpeed配置/压缩工具 14篇' },
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
- 概念拆到最原子;每步说"为什么";【必含 ASCII 图】(架构/数据流/调用链/流程)。
- 适用就给数值例子;公式行内 $...$、独立 $$...$$。
- 工具/源码/配置类: 讲清【它是什么、解决什么、整体架构与核心模块、关键流程(如请求生命周期/训练循环)、关键参数/配置项的含义与权衡、与同类对比、实践要点】这些稳定知识;【绝不编造】精确版本号/CLI默认值/未见过的API签名/具体源码行号, 讲机制与思路, 不确定标"以官方文档/源码为准"。

【固定结构】
# ${n.title}
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：${rel}
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置(它解决什么问题)
## 2.~N. 逐步拆解(架构/流程/机制, 每节配ASCII图)
## (单独一节)典型流程/配置示例(讲含义)
## 常见问题(表格)
## 🔗 跳转链接

【硬要求】220~400 行, 高密度; 顶部导航+底部双链必须有。返回 JSON {file,lines,hasHandCalc,hasAscii,links}。直接开始。`
}

const W13 = [
  { file: 'llm-inference/vllm/请求处理流程.md', title: 'vLLM 请求处理流程', cover: '一个请求从进入到返回:tokenize→调度器加入waiting→分配KV block→prefill→decode循环→采样→detokenize→流式返回;调度/抢占;架构图', related: ['00-知识地图','llm-inference/vllm/README','llm-inference/KV-Cache优化','llm-inference/解码策略'] },
  { file: 'llm-inference/vllm/长文本推理.md', title: 'vLLM 长文本推理', cover: 'chunked prefill;长上下文KV显存;block分页;并行策略;长文本下的调度;配置思路(讲含义)', related: ['00-知识地图','llm-inference/vllm/README','llm-optimizer/SplitFuse','llm-inference/KV-Cache优化'] },
  { file: 'llm-inference/vllm/源码.md', title: 'vLLM 源码结构', cover: '核心模块(LLMEngine/Scheduler/BlockManager/ModelExecutor/AttentionBackend);调用关系;扩展点;讲架构不背行号', related: ['00-知识地图','llm-inference/vllm/请求处理流程','llm-inference/vllm/README'] },
  { file: 'llm-inference/sglang/项目代码结构.md', title: 'SGLang 源码结构', cover: 'Runtime/Scheduler/RadixCache/TokenizerManager等模块;前端DSL与后端;RadixTree实现思路;架构图', related: ['00-知识地图','llm-inference/sglang/README','llm-inference/vllm/源码'] },
  { file: 'llm-inference/lightllm/README.md', title: 'LightLLM', cover: '纯Python轻量推理框架;三进程异步;token-level调度;TokenAttention;与vLLM对比;适用', related: ['00-知识地图','llm-inference/vllm/README','llm-inference/README'] },
  { file: 'llm-inference/xinference/README.md', title: 'Xinference', cover: '分布式推理部署框架;模型注册/分发;多后端(vLLM/llama.cpp/transformers);集群;OpenAI兼容;定位', related: ['00-知识地图','llm-inference/README','llm-maas/README'] },
  { file: 'llm-inference/tensorrt/README.md', title: 'TensorRT', cover: 'NVIDIA推理优化引擎;图优化/层融合/精度校准/kernel自动调优;build engine流程;与TRT-LLM关系;为什么快', related: ['00-知识地图','ai-framework/TensorRT-Model-Optimizer','llm-inference/README'] },
  { file: 'llm-inference/huggingface-transformer/README.md', title: 'Transformers 推理', cover: 'model.generate推理;KV cache;为什么慢(无PagedAttn/连续批);作为baseline;何时够用;与vLLM差距', related: ['00-知识地图','ai-framework/huggingface-transformers/README','llm-inference/vllm/README','llm-inference/解码策略'] },
  { file: 'llm-inference/deepspeed-mii/README.md', title: 'DeepSpeed-MII', cover: 'DeepSpeed推理服务化;持久化部署;Dynamic SplitFuse;与DeepSpeed-Inference关系;适用', related: ['00-知识地图','llm-inference/DeepSpeed-Inference','llm-optimizer/SplitFuse','ai-framework/deepspeed/README'] },
  { file: 'llm-inference/chatgpt.md', title: 'ChatGPT API 调用', cover: 'chat completions接口;messages角色;参数(temperature/top_p/max_tokens等)语义;流式;function/tool calling;成本', related: ['00-知识地图','llm-inference/openai','llm-inference/解码策略','llm-maas/OpenAI-ChatGPT'] },
  { file: 'llm-inference/openai.md', title: 'OpenAI 兼容接口', cover: 'OpenAI API规范成为事实标准;/v1/chat/completions等;为什么各推理引擎都兼容;SSE流式;客户端生态', related: ['00-知识地图','llm-inference/chatgpt','llm-maas/OpenAI-ChatGPT','llm-inference/vllm/README'] },
  { file: 'llm-inference/web/fastapi/README.md', title: 'FastAPI 部署 LLM', cover: '为什么FastAPI(异步/高并发);流式响应SSE;与推理引擎对接;并发与背压;部署(uvicorn/gunicorn)思路', related: ['00-知识地图','llm-inference/openai','llm-inference/README'] },
  { file: 'llm-inference/ascend/mindformers/README.md', title: '昇腾 MindFormers 推理', cover: '华为昇腾上的大模型推理;MindFormers套件;CANN;模型迁移;与CUDA生态差异;适用国产化', related: ['00-知识地图','ai-infra/算力/昇腾NPU','ai-infra/ai-hardware/AI芯片软件生态'] },
  { file: 'llm-inference/lmdeploy/功能.md', title: 'LMDeploy 功能特性', cover: 'TurboMind/PyTorch后端;KV INT8/INT4;W4A16量化;持久批;多模型;分布式;功能矩阵(讲机制)', related: ['00-知识地图','llm-inference/lmdeploy/README','llm-compression/quantization/量化基础'] },
]

const W14 = [
  { file: 'llm-train/pytorch/README.md', title: 'PyTorch 训练流程', cover: '标准训练循环(forward/loss/backward/step/zero_grad);DataLoader;amp混合精度;梯度累积;checkpoint;最小可跑示例思路', related: ['00-知识地图','ai-framework/pytorch/README','llm-train/README','llm-train/pytorch/distribution/README'] },
  { file: 'llm-train/pytorch/distribution/api.md', title: 'PyTorch 分布式 API', cover: 'init_process_group/all_reduce/broadcast/barrier;DistributedSampler;DDP包装;rank/local_rank;常用API语义', related: ['00-知识地图','llm-train/pytorch/distribution/README','ai-infra/网络/集合通信原语'] },
  { file: 'llm-train/pytorch/torchrun.md', title: 'torchrun 启动', cover: 'torchrun替代launch;rendezvous弹性;环境变量(RANK/WORLD_SIZE/MASTER_ADDR);单机多卡/多机;故障重启;用法思路', related: ['00-知识地图','llm-train/pytorch/distribution/多机多卡','llm-train/pytorch/distribution/README'] },
  { file: 'llm-train/chatglm/README.md', title: 'ChatGLM 微调', cover: 'ChatGLM微调(P-Tuning v2/LoRA/全量);数据格式;显存;部署;典型流程', related: ['00-知识地图','llm-algo/chatglm/README','llm-train/peft/Prefix-Tuning'] },
  { file: 'llm-train/vicuna/README.md', title: 'Vicuna', cover: 'LLaMA+ShareGPT对话微调;FastChat;多轮对话数据;评测(GPT-4 as judge起源);意义', related: ['00-知识地图','llm-algo/llama/模型架构','llm-train/chinese-llama-alpaca/README'] },
  { file: 'llm-train/firefly/README.md', title: 'Firefly 微调', cover: '中文指令微调项目;QLoRA;支持模型;数据;packing;实践', related: ['00-知识地图','llm-train/peft/PEFT-API','llm-train/chinese-llama-alpaca/README'] },
  { file: 'llm-train/chinese-llama-alpaca/README.md', title: '中文 LLaMA & Alpaca', cover: '中文词表扩充;二次预训练+指令微调;LoRA;为什么扩词表(中文token效率);流程', related: ['00-知识地图','llm-algo/llama/模型架构','llm-train/peft/PEFT-API','llm-data-engineering/README'] },
  { file: 'llm-train/deepspeedchat/llama/README.md', title: 'DeepSpeed-Chat', cover: 'DeepSpeed的RLHF三阶段一键流程;Hybrid Engine(训练推理切换);为什么省;PPO实现;实践', related: ['00-知识地图','llm-alignment/RLHF','ai-framework/deepspeed/README'] },
  { file: 'ai-framework/deepspeed/DeepSpeed配置JSON文件.md', title: 'DeepSpeed 配置详解', cover: 'ds_config关键字段:zero_optimization(stage/offload)/fp16-bf16/gradient_accumulation/优化器/调度;每项含义与权衡(讲语义)', related: ['00-知识地图','ai-framework/deepspeed/README','llm-train/megatron-deepspeed/README'] },
  { file: 'ai-framework/dlrover.md', title: 'DLRover (弹性容错训练)', cover: '蚂蚁弹性训练;节点故障自动恢复;弹性扩缩;flash checkpoint;为什么万卡需要容错;架构', related: ['00-知识地图','ai-infra/ai-cluster/README','llm-train/README'] },
  { file: 'llm-compression/llm-compressor/README.md', title: 'llm-compressor', cover: 'vLLM生态压缩库;统一量化(GPTQ/AWQ/SmoothQuant/FP8)+剪枝;recipe;导出compressed-tensors;与vLLM对接', related: ['00-知识地图','llm-compression/llm-compressor/量化方案','llm-compression/llm-compressor/剪枝','llm-inference/vllm/README'] },
  { file: 'llm-compression/llm-compressor/剪枝.md', title: 'llm-compressor 剪枝', cover: '稀疏化(2:4/非结构化);SparseGPT;与量化组合;recipe;精度恢复;在vLLM上加速', related: ['00-知识地图','llm-compression/sparsity/README','llm-compression/llm-compressor/README'] },
  { file: 'llm-compression/PaddleSlim/README.md', title: 'PaddleSlim', cover: '百度模型压缩库;量化/剪枝/蒸馏/NAS;PaddlePaddle生态;与PyTorch系工具对比;适用', related: ['00-知识地图','ai-framework/paddlepaddle/README','llm-compression/README'] },
  { file: 'llm-compression/quantization/tools.md', title: '量化工具对比', cover: 'AutoGPTQ/GPTQModel/AutoAWQ/llm-compressor/TRT-ModelOpt/bitsandbytes横向对比;支持算法/硬件/导出格式;选型', related: ['00-知识地图','llm-compression/quantization/GPTQ','llm-compression/llm-compressor/README','llm-compression/gptqmodel/README'] },
]

const runWave = (notes, phaseTitle) =>
  parallel(notes.map(n => () => agent(buildPrompt(n), { label: `fill:${n.file}`, phase: phaseTitle, schema: SCHEMA })))

phase('W13-推理工具与源码')
const r13 = await runWave(W13, 'W13-推理工具与源码')
log(`W13 完成: ${r13.filter(Boolean).length}/${W13.length}`)
phase('W14-训练工具与压缩工具')
const r14 = await runWave(W14, 'W14-训练工具与压缩工具')
log(`W14 完成: ${r14.filter(Boolean).length}/${W14.length}`)

const all = [...r13, ...r14].filter(Boolean)
return { total: W13.length + W14.length, written: all.length, files: all.map(x => x.file) }
