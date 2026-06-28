export const meta = {
  name: 'llmaction-补齐-run3',
  description: 'llm-action补齐run3: 28篇(推理引擎/训练框架/压缩补全/应用)填到最大细节+双链',
  phases: [
    { title: 'W5-推理引擎与框架', detail: 'vLLM/SGLang/TGI/DeepSpeed/Megatron/PyTorch 14篇' },
    { title: 'W6-训练框架与压缩应用', detail: 'HF全家桶/Triton/蒸馏/稀疏/RAG/LLMOps 14篇' },
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

【第一步】先用 Read 读 ${DIR}${n.file}，尊重已有方向;近空则从零写。
【第二步】用 Write 写到绝对路径：${DIR}${n.file}

【风格——"从最底层讲清 + 可视化"】
- 不假设记得公式;概念拆到最原子;每步说"为什么"。
- 【必含 ASCII 可视化图】(架构/数据流/组件关系/流程)。
- 适用就给【数值例子/手算】(显存/吞吐/通信量/批大小等);框架/引擎类至少给一个具体配置或场景化数字。
- 公式行内 $...$、独立 $$...$$。
- 框架/引擎类: 讲清【它解决什么问题、整体架构、核心机制(如调度/并行/内存管理/kernel)、与同类对比、何时选它、关键权衡】这些稳定知识;【不编造】具体版本号/精确CLI参数/API签名, 不确定就讲通用机制并标"以官方文档为准"。

【固定结构】
# ${n.title}
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：${rel}
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置(它解决什么问题)
## 2.~N. 逐步拆解(架构/机制, 每节配ASCII图)
## (单独一节)数值例子/典型场景
## 对照表(与同类对比)
## 常见问题(表格)
## 🔗 跳转链接

【硬要求】250~430 行, 高密度; 顶部导航+底部双链必须有。返回 JSON {file,lines,hasHandCalc,hasAscii,links}。直接开始。`
}

const W5 = [
  { file: 'llm-inference/README.md', title: 'LLM 推理总览', cover: '推理两阶段;吞吐vs延迟指标;优化全景(KV/批处理/并行/量化/解码);主流引擎地图;选型', related: ['00-知识地图','llm-inference/解码策略','llm-inference/KV-Cache优化','llm-inference/vllm/README','llm-optimizer/kv-cache'] },
  { file: 'llm-inference/vllm/README.md', title: 'vLLM 推理引擎', cover: 'PagedAttention(分页KV消碎片)原理;连续批处理;前缀缓存;调度器;张量并行;架构图;何时选vLLM', related: ['00-知识地图','llm-inference/KV-Cache优化','llm-inference/sglang/README','llm-inference/README'] },
  { file: 'llm-inference/sglang/README.md', title: 'SGLang 推理引擎', cover: 'RadixTree前缀缓存(自动KV复用)原理;结构化生成;连续批;与vLLM对比;何时选;架构', related: ['00-知识地图','llm-inference/vllm/README','llm-inference/GuidedGeneration','llm-inference/README'] },
  { file: 'llm-inference/lmdeploy/README.md', title: 'LMDeploy 推理引擎', cover: 'TurboMind引擎;持久化批处理;KV量化;与vLLM/TGI对比;适用场景;架构', related: ['00-知识地图','llm-inference/vllm/README','llm-inference/README'] },
  { file: 'llm-inference/huggingface-tgi/README.md', title: 'Text Generation Inference (TGI)', cover: 'HF官方推理服务;连续批;张量并行;量化集成;Rust+Python架构;何时选', related: ['00-知识地图','llm-inference/vllm/README','llm-inference/README'] },
  { file: 'llm-inference/DeepSpeed-Inference.md', title: 'DeepSpeed-Inference', cover: '推理张量并行+kernel注入;ZeRO-Inference显存卸载;与训练DeepSpeed关系;适用大模型单机多卡;权衡', related: ['00-知识地图','ai-framework/deepspeed/README','llm-inference/offload','llm-inference/大模型推理张量并行'] },
  { file: 'llm-inference/FlashInfer.md', title: 'FlashInfer (注意力kernel库)', cover: '专为LLM serving的注意力kernel;支持多种KV布局/PagedKV;被vLLM/SGLang采用;与FlashAttention关系;为什么需要专用kernel', related: ['00-知识地图','llm-optimizer/FlashAttention','llm-inference/Flash-Decoding','llm-inference/vllm/README'] },
  { file: 'llm-inference/GuidedGeneration.md', title: '受限/结构化生成', cover: '强制输出符合JSON/正则/语法;有限状态机/Outlines/grammar约束logits;为什么(可靠结构化输出);性能影响;实现机制', related: ['00-知识地图','llm-inference/解码策略','llm-inference/sglang/README'] },
  { file: 'llm-inference/NanoFlow.md', title: 'NanoFlow', cover: 'intra-device并行(计算/访存/通信资源重叠);nano-batch;为什么提升单卡利用率;与连续批的层次;核心思想', related: ['00-知识地图','llm-optimizer/计算通信重叠','llm-inference/README'] },
  { file: 'llm-inference/RTP-LLM.md', title: 'RTP-LLM', cover: '阿里推理引擎;高性能serving;与vLLM/TRT-LLM对比;适用场景;核心优化', related: ['00-知识地图','llm-inference/vllm/README','llm-inference/README'] },
  { file: 'ai-framework/README.md', title: 'AI 训练/推理框架总览', cover: '框架地图(训练:PyTorch/DeepSpeed/Megatron;推理:vLLM/TRT-LLM;微调:PEFT/TRL);各自定位;如何组合;选型', related: ['00-知识地图','ai-framework/pytorch/README','ai-framework/deepspeed/README','ai-framework/megatron-lm/README'] },
  { file: 'ai-framework/pytorch/README.md', title: 'PyTorch 核心', cover: '动态图/autograd;Tensor与device;nn.Module;DDP/FSDP分布式;混合精度amp;torch.compile;为什么主流', related: ['00-知识地图','ai-framework/deepspeed/README','llm-train/README'] },
  { file: 'ai-framework/deepspeed/README.md', title: 'DeepSpeed', cover: 'ZeRO 1/2/3切分(参数/梯度/优化器状态)显存账;offload;3D并行;DeepSpeed-Chat对齐;何时用;与FSDP对比', related: ['00-知识地图','ai-framework/megatron-lm/README','llm-train/README','ai-framework/pytorch/README'] },
  { file: 'ai-framework/megatron-lm/README.md', title: 'Megatron-LM', cover: '张量并行(切注意力/FFN)实现;流水并行;序列并行;与DeepSpeed组合(Megatron-DeepSpeed);大规模训练标准;通信', related: ['00-知识地图','ai-framework/deepspeed/README','llm-train/megatron/README','ai-infra/网络/集合通信原语'] },
]

const W6 = [
  { file: 'ai-framework/huggingface-transformers/README.md', title: 'HuggingFace Transformers', cover: '统一模型接口AutoModel/Tokenizer;from_pretrained;generate;Trainer;生态;为什么事实标准', related: ['00-知识地图','ai-framework/huggingface-peft/README','ai-framework/pytorch/README'] },
  { file: 'ai-framework/huggingface-peft/README.md', title: 'HuggingFace PEFT', cover: 'LoRA/QLoRA/Prefix/Prompt-Tuning统一库;adapter注入;低秩更新;显存优势;与Transformers/TRL集成', related: ['00-知识地图','llm-train/peft/Prompt-Tuning','llm-train/peft/Prefix-Tuning','ai-framework/huggingface-transformers/README'] },
  { file: 'ai-framework/huggingface-accelerate/README.md', title: 'HuggingFace Accelerate', cover: '一套代码跑单卡/多卡/多机;封装DDP/FSDP/DeepSpeed;device_map大模型分片;混合精度;定位', related: ['00-知识地图','ai-framework/pytorch/README','ai-framework/deepspeed/README'] },
  { file: 'ai-framework/openai-triton/README.md', title: 'OpenAI Triton (kernel编程)', cover: 'Python写GPU kernel;block级编程模型vs CUDA thread级;自动优化;FlashAttention用它写;何时用;示例思路', related: ['00-知识地图','ai-infra/ai-hardware/CUDA','llm-optimizer/FlashAttention','ai-infra/算力/GPU工作原理'] },
  { file: 'llm-train/README.md', title: 'LLM 训练总览', cover: '预训练→SFT→对齐流程;分布式并行选型(DP/TP/PP/ZeRO);显存优化(重计算/混合精度);框架地图;稳定性', related: ['00-知识地图','llm-algo/训练范式','ai-framework/deepspeed/README','llm-train/megatron-deepspeed/README'] },
  { file: 'llm-train/megatron-deepspeed/README.md', title: 'Megatron-DeepSpeed 实战', cover: 'Megatron(TP/PP)+DeepSpeed(ZeRO)组合;3D并行配置;大模型训练流程;数据/checkpoint;典型坑', related: ['00-知识地图','ai-framework/megatron-lm/README','ai-framework/deepspeed/README','llm-train/README'] },
  { file: 'llm-train/fp8.md', title: 'FP8 训练', cover: 'FP8训练动机;E4M3/E5M2;per-tensor缩放与amax;delayed scaling;DeepSeek-V3 FP8实践;稳定性;与BF16混用', related: ['00-知识地图','llm-compression/quantization/fp8','llm-algo/deepseek/DeepSeek-V3'] },
  { file: 'llm-compression/README.md', title: '模型压缩总览', cover: '四条路线:量化/剪枝/蒸馏/低秩;各自原理与适用;压缩率vs精度;推理加速来源;组合策略;选型', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-compression/distillation/MINILLM','llm-compression/经验'] },
  { file: 'llm-compression/distillation/MINILLM.md', title: '知识蒸馏(白盒/MiniLLM)', cover: '蒸馏原理(teacher软标签);白盒vs黑盒;MiniLLM用反向KL;为什么反向KL更适合生成;损失;与SFT区别', related: ['00-知识地图','llm-compression/README','llm-compression/distillation/GKD'] },
  { file: 'llm-compression/sparsity/README.md', title: '稀疏化与剪枝', cover: '结构化vs非结构化剪枝;2:4半结构化稀疏(硬件加速);幅度剪枝;SparseGPT;为什么能加速;精度恢复', related: ['00-知识地图','llm-compression/README','llm-compression/经验'] },
  { file: 'llm-compression/quantization/SpinQuant.md', title: 'SpinQuant (旋转量化)', cover: '用旋转矩阵把激活离群值打散再量化;可学习旋转;与SmoothQuant/QuaRot关系;为什么旋转有效;W4A4', related: ['00-知识地图','llm-compression/quantization/SmoothQuant','llm-compression/quantization/量化基础'] },
  { file: 'llm-application/rag/README.md', title: 'RAG 检索增强生成', cover: '为什么RAG(知识更新/减幻觉);流程:切块→embedding→向量检索→重排→拼上下文→生成;关键参数;评估;常见问题', related: ['00-知识地图','llm-application/rag/embedding','llm-application/vector-db/README','llm-application/应用场景'] },
  { file: 'llm-application/应用场景.md', title: 'LLM 应用场景全景', cover: '对话/RAG/Agent/代码/总结/翻译/分类;Function calling;Prompt工程;选型(微调vs RAG vs prompt);落地考量', related: ['00-知识地图','llm-application/rag/README','llm-application/README'] },
  { file: 'llmops/README.md', title: 'LLMOps 总览', cover: '训练-评测-部署-监控全生命周期;K8s编排;多机多卡调度;推理平台;模型版本/灰度;可观测', related: ['00-知识地图','llmops/kubernetes','llmops/模型推理平台方案','llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释'] },
]

const runWave = (notes, phaseTitle) =>
  parallel(notes.map(n => () => agent(buildPrompt(n), { label: `fill:${n.file}`, phase: phaseTitle, schema: SCHEMA })))

phase('W5-推理引擎与框架')
const r5 = await runWave(W5, 'W5-推理引擎与框架')
log(`W5 完成: ${r5.filter(Boolean).length}/${W5.length}`)
phase('W6-训练框架与压缩应用')
const r6 = await runWave(W6, 'W6-训练框架与压缩应用')
log(`W6 完成: ${r6.filter(Boolean).length}/${W6.length}`)

const all = [...r5, ...r6].filter(Boolean)
return { total: W5.length + W6.length, written: all.length, files: all.map(x => x.file) }
