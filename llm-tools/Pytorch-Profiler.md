# PyTorch Profiler 性能分析器

> 用上下文管理器在训练/推理循环里"打点采样"，把每个算子(operator)的 CPU/GPU 耗时、调用次数、张量内存、输入形状、Python 调用栈全部抓出来，再用 TensorBoard / Chrome Trace 可视化，定位真正的瓶颈算子。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/pytorch/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[llm-optimizer/FlashAttention]] · [[docs/transformer内存估算]]

## 阅读地图

| 节 | 你将学到 | 关键词 |
|---|---|---|
| 0 | 一句话锚点：Profiler 是采样器不是改写器 | 上下文管理器 / 算子级 |
| 1 | 地基：CPU 算子 vs CUDA kernel、self 时间 vs total 时间、异步执行 | aten:: / self / total |
| 2 | 最小示例：`profile()` + `record_function` + `key_averages().table()` | ProfilerActivity / table |
| 3 | 参数详解：activities / record_shapes / profile_memory / with_stack | 五大开关 |
| 4 | GPU/XPU 分析：为什么 Self CUDA 多为 0、kernel 行怎么读 | CUDA total / kernel |
| 5 | 内存分析：self mem vs total mem，一个 double→float 实战省内存 | profile_memory |
| 6 | schedule 调度：wait/warmup/active/repeat/skip_first 长任务采样 | schedule / span |
| 7 | 导出：TensorBoard 插件 + Chrome Trace + 调用栈跳源码 | trace.json / logdir |
| 实操 | 三个可直接跑的脚本 + 安装命令 | recipe / tensorboard |
| 坑 | 高频踩坑表 | 偏差 / 旧版本 |

---

## 0. 一句话锚点

PyTorch Profiler 是一个**采样器(observer)**：它包住你的前向/反向代码，记录每个算子被调用的次数、各自占用的 CPU 时间、CUDA 时间、内存增减、输入形状和 Python 调用栈，**但它本身不会改写或优化你的模型**。它只回答一个问题——"**时间和内存到底花在哪个算子上了？**"找到瓶颈后，优化是你的事。

仓库内三个示例脚本：

- `base-profiler.py` —— 用 `torch.autograd.profiler`（旧式低层 API），演示按调用栈分组 + 内存归因 + 一次真实优化（double→float→nonzero）。
- `profiler-recipe.py` —— 用 `torch.profiler`（推荐的新式 API），CPU/CUDA/内存/调用栈/Chrome Trace 全覆盖。
- `tensorboard-profiler.py` —— 带 `schedule` + TensorBoard handler 的训练循环版。

---

## 1. 地基：四个必须先建立的概念

### 1.1 算子(operator) 是分析的最小单位

PyTorch 执行 `model(inputs)` 时，会被拆成一连串底层算子调用，名字以 `aten::` 开头（ATen 是 PyTorch 的 C++ 张量库），例如 `aten::conv2d`、`aten::addmm`（矩阵乘加）、`aten::batch_norm`、`aten::copy_`（拷贝）。Profiler 的报表就是**按算子聚合**的。

```
model(inputs)
   └── aten::conv2d ──> aten::convolution ──> aten::_convolution ──> aten::mkldnn_convolution(CPU)
   └── aten::batch_norm ──> aten::_batch_norm_impl_index ──> aten::native_batch_norm
   └── aten::mean / aten::select / ...
```

一个高层算子(`conv2d`)会**层层调用**子算子，这是后面 self vs total 的根源。

### 1.2 self 时间 vs total 时间（最容易看错的一栏）

| 列 | 含义 | 类比 |
|---|---|---|
| **CPU total / CUDA total** | 该算子**及其所有子算子**的总耗时 | 经理工时=自己+整个团队 |
| **Self CPU / Self CUDA** | **仅该算子自身**、不含子调用的耗时 | 经理亲手干活的工时 |

为什么需要两者：`aten::conv2d` 的 total 很大，但它的时间几乎全花在子算子 `aten::mkldnn_convolution` 上，所以 `conv2d` 的 self 很小、`mkldnn_convolution` 的 self 很大。**排查"哪个原子操作真的慢"要看 self；排查"哪个高层模块整体慢"要看 total。**

```
ASCII：一次 conv2d 的时间归属
 conv2d total = 31.9ms ┌──────────────────────────────┐
                       │ conv2d self 0.23ms             │
                       │ └ mkldnn_convolution self 30.8ms │ ← 真正干活的人
                       └──────────────────────────────┘
看 self → 优化 mkldnn_convolution；看 total → 知道 conv 这层整体 31.9ms
```

### 1.3 GPU 是异步的 → Self CUDA 常为 0

CPU 发射(launch) CUDA kernel 后**立即返回**，不等 GPU 算完。所以像 `aten::conv2d` 这种"调度型"算子的 **Self CUDA = 0us**——它只负责发射，真正的 GPU 时间记在底层 kernel 行上，例如 `void at::native::im2col_kernel<float>(...)`、`sgemm_32x32x32_NN`（cuBLAS 的矩阵乘 kernel）。

> 看 GPU 瓶颈，要往下翻到**带 `<float>`、`sgemm`、`kernel` 字样的真正 kernel 行**，而不是 `aten::` 调度行。

### 1.4 warm-up（预热）是数据可信的前提

第一次跑某算子时，CUDA 要做 JIT 编译、cuDNN 要做 kernel 选择(autotune)、缓存要建立——这些一次性开销会让首次迭代**虚高几倍**。所以所有示例都先空跑一次：

```python
# warm-up
model(input, mask)
```

然后才在 `with profiler.profile(...)` 里采真正的数据。schedule 里的 `warmup` 阶段（见第 6 节）就是把这个动作自动化。

---

## 2. 最小示例：抓 CPU 时间（profiler-recipe.py）

安装与最简调用：

```python
# pip install torch torchvision
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

三个零件：

- `profile(activities=[...])` —— 上下文管理器，进入即开始采样、退出即停止。
- `record_function("name")` —— 给一段代码**贴用户自定义标签**，让它在报表里成为一行（如 `model_inference`），方便把若干算子归到一个逻辑块。
- `prof.key_averages().table(sort_by=..., row_limit=N)` —— 把采样聚合成表，按某列排序取前 N 行。

真实输出（节选）：

```
---------------------------------  ------------  ------------  ------------  ------------
                             Name      Self CPU     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------
                  model_inference       5.509ms      57.503ms      57.503ms             1
                     aten::conv2d     231.000us      31.931ms       1.597ms            20
                aten::convolution     250.000us      31.700ms       1.585ms            20
               aten::_convolution     336.000us      31.450ms       1.573ms            20
         aten::mkldnn_convolution      30.838ms      31.114ms       1.556ms            20  ← Self 最大，真瓶颈
                 aten::batch_norm     211.000us      14.693ms     734.650us            20
          aten::native_batch_norm       9.229ms      14.109ms     705.450us            20
                       aten::mean     332.000us       2.631ms     125.286us            21
                     aten::select       1.668ms       2.292ms       8.988us           255
---------------------------------  ------------  ------------  ------------  ------------
Self CPU time total: 57.549ms
```

**怎么读**：ResNet18 的 20 次 conv（`# of Calls=20`）整体吃掉 31.9ms（total），但其中 30.8ms 的 self 集中在 `aten::mkldnn_convolution`——这就是 CPU 上卷积的真实耗时所在。`aten::select` 被调了 255 次，单次只有 8.988us，量大但不致命。

### 2.1 按输入形状细分

加 `group_by_input_shape=True`，同一算子会按不同输入形状拆开——这能看出"哪种 shape 的卷积最贵"：

```python
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
```

```
                             Name     CPU total                                 Input Shapes
                     aten::conv2d       8.008ms       [5,64,56,56], [64,64,3,3], ...
                     aten::conv2d       6.332ms      [5,512,7,7], [512,512,3,3], ...
                     aten::conv2d       4.751ms     [5,256,14,14], [256,256,3,3], ...
```

> 前提：`profile(..., record_shapes=True)`。否则没有 Input Shapes 列。

---

## 3. 五大参数详解（原文核心配置）

| 参数 | 取值 | 作用 | 代价 |
|---|---|---|---|
| `activities` | `ProfilerActivity.CPU` | PyTorch 算子、TorchScript 函数、用户标签 | 低 |
| | `ProfilerActivity.CUDA` | 设备上的 CUDA kernel | 中 |
| | `ProfilerActivity.XPU` | 设备上的 XPU kernel（Intel GPU） | 中 |
| `record_shapes` | True/False | 记录输入形状，开启 `group_by_input_shape` | 小幅开销 |
| `profile_memory` | True/False | 报告 Tensor 占用内存的分配/释放 | 旧版(<1.10)若太慢需关闭或升级 |
| `with_stack` | True/False | 记录源码文件+行号；VS Code 里可点栈帧跳到源码行 | 较大开销 |

> ⚠️ 原文明确警告：`profile_memory` 在 **PyTorch < 1.10** 上，若分析时间过长会很慢——**禁用它或升级版本**。

---

## 4. GPU / XPU 分析

设备自动选择 + 三设备活动全开。`sort_by` 用 `device + "_time_total"` 写成设备无关：

```python
if torch.cuda.is_available():
    device = 'cuda'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    print('Neither CUDA nor XPU devices are available ...')
    import sys; sys.exit(0)

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
sort_by_keyword = device + "_time_total"   # 'cuda_time_total'

model = models.resnet18().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

with profile(activities=activities, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
```

真实 CUDA 输出（节选）：

```
                                                   Name     Self CUDA    CUDA total
                                        model_inference       0.000us      11.666ms
                                           aten::conv2d       0.000us      10.484ms   ← Self CUDA=0（只发射）
                                      aten::convolution       0.000us      10.484ms
                              aten::thnn_conv2d_forward      10.484ms      10.484ms
void at::native::im2col_kernel<float>(long, float co...       3.844ms       3.844ms   ← 真正的 GPU kernel
                                      sgemm_32x32x32_NN       3.206ms       3.206ms   ← cuBLAS 矩阵乘
                                  sgemm_32x32x32_NN_vec       3.093ms       3.093ms
Self CPU time total: 23.015ms
Self CUDA time total: 11.666ms
```

**读法**：高层 `aten::conv2d` 的 `Self CUDA=0.000us`（印证 1.3 节的异步），GPU 的 11.666ms 真正分布在 `im2col_kernel`(3.8ms)、两个 `sgemm`(3.2+3.1ms) 等 kernel 上——卷积在 GPU 上被拆成 im2col + 矩阵乘。注意末行同时给出 **Self CPU total** 和 **Self CUDA total** 两个独立合计。

```
ASCII：sort_by 的选取
 想看 CPU 瓶颈   → sort_by="cpu_time_total" / "self_cpu_time_total"
 想看 GPU 瓶颈   → sort_by="cuda_time_total" / "self_cuda_time_total"
 想看 CPU 内存   → sort_by="self_cpu_memory_usage" / "cpu_memory_usage"
 设备无关写法    → sort_by = "self_" + device + "_time_total"
```

---

## 5. 内存分析：self mem vs total mem

```python
with profile(activities=[ProfilerActivity.CPU],
             profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))  # self mem：自身分配
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))        # total mem：含子调用
```

```
# sort_by="self_cpu_memory_usage"           Name       CPU Mem  Self CPU Mem  # of Calls
                                      aten::empty      94.79 Mb      94.79 Mb         121  ← 内存几乎全来自 empty
                     aten::max_pool2d_with_indices      11.48 Mb      11.48 Mb           1
                                      aten::addmm      19.53 Kb      19.53 Kb           1
# sort_by="cpu_memory_usage"
                                 aten::batch_norm      47.41 Mb           0 b          20  ← Self=0，由子算子 empty 分配
                                     aten::conv2d      47.37 Mb           0 b          20
```

**关键认知**：`batch_norm`/`conv2d` 的 `Self CPU Mem=0`，因为它们自己不分配内存，是底层的 `aten::empty` 替它们分配的（121 次共 94.79 Mb）。**self mem 告诉你"谁亲手 malloc"，total mem 告诉你"哪个逻辑模块吃内存"。**

### 5.1 实战：一次 double→float 的内存优化（base-profiler.py）

这是用旧式 `torch.autograd.profiler` + `group_by_stack_n` 按调用栈归因，再据此优化的完整链路。`MyModule.forward` 把一段逻辑包进 `record_function`：

```python
import torch.autograd.profiler as profiler

class MyModule(nn.Module):
    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)
        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)  # 拷到 CPU 走 numpy
            hi_idx = torch.from_numpy(hi_idx).cuda()              # 再拷回 CUDA
        return out, hi_idx

model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()  # ← double！

model(input, mask)  # warm-up
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
```

**第①版（double）报告**：`MASK INDICES` 占 87.88%，Self CPU **5.212s**，Self CPU Mem **-953.67 Mb**（负号=释放）。`aten::copy_` 12.07%（715ms），来自 `.cpu().numpy()` 和 `.cuda()` 的两次跨设备拷贝。`Self CPU time total: 5.931s`。

**第②版：mask 用 `torch.float` 取代 `torch.double`**（内存减半）：

```python
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()
```

报告：`MASK INDICES` 内存从 -953.67 Mb 降到 **-476.84 Mb**（正好减半，因 float 是 4B 而 double 是 8B），`aten::copy_` 从 715ms 降到 **338ms**。`Self CPU time total: 5.347s`。

**手算验证**：`500³ × 8B = 1,000,000,000 B ≈ 953.67 MiB`（÷1024² ≈ 953.67），float 则 `500³ × 4B ≈ 476.84 MiB`——和报表完全对上。

**第③版：用 GPU 原生 `nonzero()` 取代 "拷到 CPU 跑 numpy.argwhere 再拷回"**：

```python
with profiler.record_function("MASK INDICES"):
    threshold = out.sum(axis=1).mean()
    hi_idx = (mask > threshold).nonzero(as_tuple=True)  # 全程留在 GPU
```

报告：彻底消灭了 5s 级的 `MASK INDICES` 和 `aten::copy_`，瓶颈变成 GPU 上的 `aten::gt`(129ms)+`aten::nonzero`(84ms)。`Self CPU time total: 225.801ms`——从 **5.9s → 0.23s，约 26× 加速**。这就是 Profiler 的价值：定位到"跨设备拷贝 + 高精度 dtype"两个真凶。

---

## 6. schedule：长任务的分阶段采样

跟踪每一步会让 trace 文件巨大且拖慢训练。`schedule` 把采样切成可重复的"周期(cycle/span)"，每周期分四阶段：

```python
from torch.profiler import schedule

my_schedule = schedule(
    skip_first=10,   # 先无脑跳过 10 步
    wait=5,          # 空转(Profiler 关闭)
    warmup=1,        # 开始跟踪但丢弃结果（去掉启动偏差）
    active=3,        # 真正记录数据
    repeat=2)        # 周期上限=2；默认 0=一直循环到任务结束
```

四阶段语义（原文精确表述）：

| 阶段 | 参数 | Profiler 状态 | 为什么 |
|---|---|---|---|
| skip_first | `skip_first=10` | 完全忽略 | 跳过最初不稳定的步（默认 0） |
| wait（Idling） | `wait=5` | **禁用**，不采样 | 让训练自然推进，降低开销 |
| warmup（预热） | `warmup=1` | 开始跟踪但**丢弃结果** | 采样启动开销高，结果会偏差，故丢弃 |
| active | `active=3` | **记录事件** | 这才是你要的数据 |

```
ASCII：skip_first=10, wait=5, warmup=1, active=3, repeat=2 的步序
step: 0..9 | 10..14 | 15 | 16..18 | 19..23 | 24 | 25..27 | 之后停止
      skip  | wait   |warm| ACTIVE | wait   |warm| ACTIVE |
      (10)  | (5)    |(1) | (3)记录 | (5)    |(1) | (3)记录 | repeat=2 满
```

> 原文逐字结论：profiler 跳过前 15 步（skip_first 10 + wait 5），第 16 步热身，记录 17~19，再跳 5 步，热身，再记录 3 步；因 `repeat=2`，两周期后停止。

`schedule` 必须配 `prof.step()`——**每步调用一次**，告诉 Profiler 步边界，它据此推进状态机：

```python
def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

with profile(
    activities=activities,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
    on_trace_ready=trace_handler
) as p:
    for idx in range(8):
        model(inputs)
        p.step()   # ← 每步必调，否则 schedule 不推进
```

`on_trace_ready` 在每个周期结束时被调用（参数是 profiler 引用），用来导出/打印结果（如 `export_chrome_trace`）。

---

## 7. 导出与可视化

### 7.1 TensorBoard 插件（tensorboard-profiler.py）

```python
# pip install torch torchvision
# pip install torch_tb_profiler
# tensorboard --logdir=./log
# http://localhost:6006/#pytorch_profiler
# tensorboard --logdir=log_dir --host=127.0.0.1
```

完整训练循环 + Profiler（原文配置原样）：

```python
def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        prof.step()  # 在每步调用，通知 profiler 步边界
        if step >= 1 + 1 + 3:   # wait+warmup+active=5 步后退出
            break
        train(batch_data)
```

- `tensorboard_trace_handler('./log/resnet18')` 把结果写到 `./log/resnet18`，再用 `tensorboard --logdir=./log` 打开。
- `with_stack=True` 时，在 VS Code 里启动的 TensorBoard 中**点击堆栈帧可跳到具体源码行**。
- 数据集是 CIFAR10（`Resize(224)` 适配 ResNet18），`batch_size=32`，`resnet18(weights='IMAGENET1K_V1')`，SGD `lr=0.001, momentum=0.9`。

### 7.2 非上下文管理器写法（start/stop）

不想用 `with` 时可手动控制：

```python
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)

prof.start()
for step, batch_data in enumerate(train_loader):
    prof.step()
    if step >= 1 + 1 + 3:
        break
    train(batch_data)
prof.stop()
```

### 7.3 Chrome Trace

```python
with profile(activities=activities) as prof:
    model(inputs)
prof.export_chrome_trace("trace.json")
```

生成的 `trace.json` 用 Chrome 浏览器打开 `chrome://tracing`（或 `edge://tracing`）即可看到时间轴火焰图，CPU/CUDA/XPU 可切换。

### 7.4 调用栈聚合

```python
sort_by_keyword = "self_" + device + "_time_total"
with profile(activities=activities, with_stack=True) as prof:
    model(inputs)
print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=2))
```

```
                     Name  Source Location
aten::thnn_conv2d_forward  .../torch/nn/modules/conv.py(439): _conv_forward
                           .../torch/nn/modules/conv.py(443): forward
                           .../torchvision/models/resnet.py(63): forward
```

`group_by_stack_n=5` 表示按"最后 5 层 Python 调用栈"分组，让你看到**同一个算子是从源码哪一行调过来的**。

---

## 常见问题 / 坑

| 现象 | 原因 | 解法 |
|---|---|---|
| Self CUDA 全是 0.000us | CPU 异步发射 kernel，时间记在底层 kernel 行 | 往下翻找 `<float>`/`sgemm`/`kernel` 行；用 `cuda_time_total` 排序 |
| 首次迭代时间虚高 | CUDA JIT / cuDNN autotune / 缓存一次性开销 | 先 `model(input)` warm-up；schedule 里用 `warmup` 阶段丢弃 |
| schedule 不生效、报表为空 | 忘了在循环里调 `prof.step()` | 每步必调 `prof.step()`，Profiler 靠它推进状态机 |
| trace 文件巨大、训练变慢 | 跟踪了每一步 | 用 `schedule(skip_first/wait/warmup/active/repeat)` 只采样几步 |
| `profile_memory` 极慢 | PyTorch < 1.10 的已知问题 | 关闭 `profile_memory` 或升级 PyTorch |
| 没有 Input Shapes / Source Location 列 | 没开对应开关 | 形状需 `record_shapes=True`；源码需 `with_stack=True` |
| 内存大头算子 Self Mem=0 | 它没亲手 malloc，由子算子(`aten::empty`)分配 | 看 self mem 找"谁分配"，看 total mem 找"哪个模块吃" |
| 报表数字偏大且不稳 | active 阶段太短被启动开销污染 / 没预热 | 增大 `active`、确保有 `warmup` |
| 跨设备拷贝 `aten::copy_` 占大头 | `.cpu().numpy()`→`.cuda()` 往返、或 numpy 混用 | 改用 GPU 原生算子如 `nonzero()`，避免离开 GPU |
| 高精度 dtype 内存翻倍 | `torch.double`(8B) 而非 `torch.float`(4B) | 非必要不用 double，省一半内存（见 5.1 手算） |

---

## 🔗 跳转链接

枢纽：[[00-知识地图]]

- 框架基础：[[ai-framework/pytorch/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 性能指标：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
- 算子/硬件：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 通信/网络：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 计算优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 压缩量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练/对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 架构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
