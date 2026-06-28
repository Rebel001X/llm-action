export const meta = {
  name: 'llmaction-补齐-run1',
  description: 'llm-action补齐run1: 28篇核心概念(Transformer/推理/压缩/Infra)填到最大细节+双链',
  phases: [
    { title: 'W1-核心内核与Infra', detail: 'Transformer/RoPE/MoE/GPU/CUDA/通信/FlashAttn 14篇' },
    { title: 'W2-推理与压缩', detail: '解码/KV/PD分离/张量并行/量化 14篇' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'

const SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    lines: { type: 'number' },
    hasHandCalc: { type: 'boolean' },
    hasAscii: { type: 'boolean' },
    links: { type: 'number' },
  },
  required: ['file', 'lines', 'hasHandCalc', 'hasAscii', 'links'],
}

function buildPrompt(n, phaseTitle) {
  const rel = n.related.map(r => `[[${r}]]`).join(' ')
  return `你是顶尖 AI-Infra 讲师。把 llm-action 仓库里一个【薄文件】补齐到【最大细节】，用【中文】，并用 Write 工具覆盖写回原路径。

【要补齐的文件(相对仓库根)】${n.file}
【主题】${n.title}
【必须覆盖】${n.cover}
【相关链接(底部"相关"用这些 Obsidian 双链, 路径已给, 不带.md)】${rel}

【第一步】先用 Read 工具读现有文件 ${DIR}${n.file}，尊重其已有方向(若有);若近空则从零写。
【第二步】用 Write 工具把补齐后的完整内容写到绝对路径：${DIR}${n.file}

【风格——"从最底层讲清 + 逐数手算 + ASCII可视化"】
- 不假设读者记得公式;每个概念拆到最原子;每步都说"为什么"。
- 【必含 ASCII 可视化图】(数据流/结构/流程/内存布局)。
- 【适用就必含逐数手算】(用小数字把关键计算算出来, 如 FLOPs/显存/量化scale/注意力分数等);纯叙述型主题至少给一个具体数值例子。
- 数学公式行内 $...$、独立 $$...$$;该推导就推导。
- 工具/框架类主题: 讲清【原理、架构、关键机制、何时用、权衡】等稳定知识;【不要编造】具体版本号/CLI 参数/API 签名(不确定就讲通用机制并标注"以官方文档为准")。

【固定结构】
# ${n.title}
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：${rel}

## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置(从最基础讲起)
## 2.~N. 逐步拆解(每节: 原理 + 为什么 + ASCII图; 适用处给公式/手算)
## (单独一节)数值示例/手算
## 对照/复杂度表
## 常见问题(表格: 疑问→真相)
## 🔗 跳转链接(列出相关 [[双链]])

【硬要求】长度 250~450 行, 信息密度高不灌水; 顶部导航 + 底部双链必须有。写完返回 JSON：{file, lines, hasHandCalc, hasAscii, links}。直接开始, 不要解释。`
}

// ---- W1: 核心内核 + Infra (14) ----
const W1 = [
  { file: 'llm-algo/基本概念.md', title: 'LLM 基本概念全景', cover: '什么是LLM;token/参数/上下文;预训练vs微调vs对齐;自回归;涌现;Scaling Law;推理两阶段;一张全景图把术语串起来', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-algo/训练范式','llm-algo/FLOPs'] },
  { file: 'llm-algo/transformer/模型架构.md', title: 'Transformer 模型架构', cover: 'Embedding+位置编码+多头注意力+FFN+残差+归一化拼成Block;Decoder-only;数据流形状追踪;一个token走完全程;参数量构成', related: ['00-知识地图','llm-algo/旋转编码RoPE','llm-algo/mlp','llm-algo/moe/README','llm-algo/FLOPs'] },
  { file: 'llm-algo/旋转编码RoPE.md', title: '旋转位置编码 RoPE', cover: '为什么要位置编码;绝对vs相对;RoPE二维旋转矩阵推导(复数视角);点积里如何自然出现相对位置;长度外推NTK/YaRN;手算2维在几个位置的旋转与点积', related: ['00-知识地图','llm-algo/transformer/模型架构'] },
  { file: 'llm-algo/mlp.md', title: 'MLP / FFN 前馈网络', cover: '两层升降维(4x);激活ReLU/GELU/SiLU;GLU与SwiGLU门控为什么更好;参数量占比;手算一个小FFN含SwiGLU', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-algo/moe/README'] },
  { file: 'llm-algo/moe/README.md', title: 'MoE 混合专家', cover: '稠密FFN→稀疏MoE;门控Top-k路由;专家容量/负载均衡loss;省算力原理;共享专家+细粒度(DeepSeekMoE);手算小MoE路由加权', related: ['00-知识地图','llm-algo/mlp','llm-algo/deepseek/DeepSeek-V2','llm-compression/quantization/moe模型量化'] },
  { file: 'llm-algo/FLOPs.md', title: 'FLOPs / 参数量 / 计算量估算', cover: 'Transformer参数量公式(注意力+FFN+embedding);前向FLOPs≈2N;训练≈6N;KV Cache显存公式;手算一个配置的参数量/FLOPs/显存;MFU', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-optimizer/kv-cache','ai-infra/算力/GPU工作原理'] },
  { file: 'llm-algo/训练范式.md', title: 'LLM 训练范式', cover: '预训练(next-token)→SFT(指令)→RLHF/DPO(对齐)三段;每段目标/数据/损失;为什么分段;持续预训练/退火', related: ['00-知识地图','llm-algo/基本概念','llm-alignment/RLHF','llm-alignment/DPO'] },
  { file: 'ai-infra/算力/GPU工作原理.md', title: 'GPU 工作原理', cover: 'SM/CUDA核/Tensor Core;SIMT与warp;显存层级(寄存器/共享/L2/HBM)带宽容量;算力vs带宽;roofline与arithmetic intensity;为什么LLM decode受带宽限;手算一个op是compute还是memory bound', related: ['00-知识地图','ai-infra/ai-hardware/CUDA','llm-algo/FLOPs','llm-optimizer/FlashAttention'] },
  { file: 'ai-infra/ai-hardware/CUDA.md', title: 'CUDA 编程模型', cover: 'grid/block/thread/warp层级;线程到数据映射;共享内存与同步;一个向量加/简单kernel;occupancy;CUDA与cuDNN/cuBLAS生态', related: ['00-知识地图','ai-infra/算力/GPU工作原理'] },
  { file: 'ai-infra/网络/集合通信原语.md', title: '集合通信原语', cover: 'AllReduce/AllGather/ReduceScatter/Broadcast/All2All语义+图示;ring-AllReduce算法与通信量2(N-1)/N推导;在DP/TP里各用哪个;手算一次ring-AllReduce数据流', related: ['00-知识地图','ai-infra/网络/NCCL','ai-infra/网络/InfiniBand'] },
  { file: 'ai-infra/网络/NCCL.md', title: 'NCCL 通信库', cover: 'NCCL是什么/做什么;通信原语实现;拓扑感知(NVLink/PCIe/IB);ring vs tree算法;通信子comm/stream;与PyTorch DDP集成;调优要点', related: ['00-知识地图','ai-infra/网络/集合通信原语','ai-infra/网络/InfiniBand'] },
  { file: 'ai-infra/网络/InfiniBand.md', title: 'InfiniBand 与 RoCE', cover: 'IB是什么;RDMA原理(绕过CPU/内核零拷贝);IB vs RoCE vs 以太网;带宽/时延;胖树拓扑;GPUDirect RDMA;为什么大规模训练需要它', related: ['00-知识地图','ai-infra/网络/NCCL','ai-infra/网络/集合通信原语'] },
  { file: 'llm-optimizer/FlashAttention.md', title: 'FlashAttention', cover: '标准注意力物化n×n显存痛点;分块tiling;online softmax(滚动max和分母)推导+手算;O(n)显存不改时间;IO-aware;v1/v2/v3思路', related: ['00-知识地图','llm-optimizer/kv-cache','ai-infra/算力/GPU工作原理','llm-inference/Flash-Decoding'] },
  { file: 'llm-optimizer/kv-cache.md', title: 'KV Cache 原理与优化', cover: 'KV Cache是什么/为什么;大小公式2·L·h·dh·n;手算显存;增长正比序列;优化总览(MQA/GQA/MLA/量化/PagedAttention);decode带宽瓶颈', related: ['00-知识地图','llm-inference/KV-Cache优化','llm-optimizer/FlashAttention','llm-algo/FLOPs'] },
]

// ---- W2: 推理 + 压缩 (14) ----
const W2 = [
  { file: 'llm-inference/解码策略.md', title: '解码与采样策略', cover: 'logits→softmax→采样;greedy/温度T/top-k/top-p(nucleus)/beam;repetition penalty;手算一组logits在不同温度/topk/topp下的分布;贪心vs采样取舍', related: ['00-知识地图','llm-inference/KV-Cache优化','llm-inference/README'] },
  { file: 'llm-inference/KV-Cache优化.md', title: 'KV Cache 优化(推理)', cover: 'KV Cache显存账;PagedAttention分页消碎片;前缀复用;量化KV;窗口/驱逐;MQA/GQA/MLA架构降系数;组合拳', related: ['00-知识地图','llm-optimizer/kv-cache','llm-inference/PD分离','llm-compression/quantization/量化基础'] },
  { file: 'llm-inference/PD分离.md', title: 'Prefill-Decode 分离', cover: 'prefill(compute-bound)与decode(memory-bound)特性不同;为什么分离部署;KV如何从P传到D;调度与SLO(TTFT/TPOT);Mooncake/分离架构', related: ['00-知识地图','llm-inference/分离式推理架构','llm-inference/Mooncake','llm-optimizer/kv-cache'] },
  { file: 'llm-inference/分离式推理架构.md', title: '分离式推理架构', cover: 'PD分离整体架构;KV缓存池/传输;调度器;弹性扩缩;与连续批/PagedAttention配合;典型系统', related: ['00-知识地图','llm-inference/PD分离','llm-inference/Mooncake','llm-inference/大模型推理张量并行'] },
  { file: 'llm-inference/Flash-Decoding.md', title: 'Flash-Decoding', cover: 'decode阶段FlashAttention的并行不足(序列维);Flash-Decoding沿KV序列切分并行+二次归约;为什么提升长上下文decode;与FlashAttention区别', related: ['00-知识地图','llm-optimizer/FlashAttention','llm-inference/KV-Cache优化'] },
  { file: 'llm-inference/大模型推理张量并行.md', title: '大模型推理张量并行', cover: '单卡放不下时TP切注意力/FFN;按列/行切+AllReduce;通信量;与PP/EP组合;推理TP与训练TP差异;手算一个矩阵乘TP切分', related: ['00-知识地图','ai-infra/网络/集合通信原语','llm-inference/分离式推理架构'] },
  { file: 'llm-inference/Mooncake.md', title: 'Mooncake(KV 池化架构)', cover: 'Kimi的Mooncake;以KV Cache为中心的分离架构;KVCache池(内存/SSD分层);prefill/decode/调度分离;前缀复用;为什么省显存提吞吐', related: ['00-知识地图','llm-inference/PD分离','llm-inference/分离式推理架构','llm-optimizer/kv-cache'] },
  { file: 'llm-optimizer/计算通信重叠.md', title: '计算-通信重叠', cover: '分布式里通信暴露成本;用多stream/异步把通信藏到计算后面;梯度AllReduce与反向重叠;TP里的重叠;流水并行气泡;示意图', related: ['00-知识地图','ai-infra/网络/集合通信原语','llm-inference/大模型推理张量并行'] },
  { file: 'llm-optimizer/SplitFuse.md', title: 'SplitFuse / Chunked Prefill', cover: '长prompt的prefill会阻塞decode;把长prefill切块与decode混合调度(chunked prefill/SplitFuse);提升TPOT与吞吐均衡;DeepSpeed-FastGen', related: ['00-知识地图','llm-inference/PD分离','llm-inference/KV-Cache优化'] },
  { file: 'llm-compression/quantization/量化基础.md', title: '量化基础', cover: '为什么量化;定点表示scale/zero-point;对称vs非对称;per-tensor/per-channel/per-group;手算一个张量量化反量化+误差;PTQ vs QAT;INT8/INT4/FP8概览', related: ['00-知识地图','llm-compression/quantization/SmoothQuant','llm-compression/quantization/GPTQ','llm-compression/quantization/fp8'] },
  { file: 'llm-compression/quantization/SmoothQuant.md', title: 'SmoothQuant', cover: '激活的离群值outlier导致难量化;把量化难度从激活迁移到权重(per-channel缩放)的数学;平滑因子;W8A8;手算一个迁移例子', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-compression/quantization/LLM-int8'] },
  { file: 'llm-compression/quantization/GPTQ.md', title: 'GPTQ', cover: '后训练逐层量化;基于Hessian的逐列量化与误差补偿(OBQ/OBS思想)推导;为什么按重要性顺序;group-size;W4精度;与AWQ对比', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-compression/quantization/SmoothQuant'] },
  { file: 'llm-compression/quantization/LLM-int8.md', title: 'LLM.int8()', cover: '大模型激活离群特征;混合精度分解(离群通道走FP16,其余INT8);向量级量化;为什么不掉点;手算一个分解例子', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-compression/quantization/SmoothQuant'] },
  { file: 'llm-compression/quantization/fp8.md', title: 'FP8 量化与训练', cover: 'FP8格式E4M3/E5M2位分配与动态范围;为什么FP8兼顾范围与精度;scaling;FP8训练(DeepSeek-V3)与推理;vs INT8;手算一个数的FP8表示', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-train/fp8'] },
]

const runWave = (notes, phaseTitle) =>
  parallel(notes.map(n => () =>
    agent(buildPrompt(n, phaseTitle), { label: `fill:${n.file}`, phase: phaseTitle, schema: SCHEMA })
  ))

phase('W1-核心内核与Infra')
const r1 = await runWave(W1, 'W1-核心内核与Infra')
log(`W1 完成: ${r1.filter(Boolean).length}/${W1.length}`)

phase('W2-推理与压缩')
const r2 = await runWave(W2, 'W2-推理与压缩')
log(`W2 完成: ${r2.filter(Boolean).length}/${W2.length}`)

const all = [...r1, ...r2].filter(Boolean)
return {
  total: W1.length + W2.length,
  written: all.length,
  files: all.map(x => x.file),
  missingHandCalc: all.filter(x => !x.hasHandCalc).map(x => x.file),
  missingAscii: all.filter(x => !x.hasAscii).map(x => x.file),
}
