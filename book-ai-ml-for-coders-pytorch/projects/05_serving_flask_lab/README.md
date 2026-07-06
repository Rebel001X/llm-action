# 05 · 推理服务实验：Flask + TorchServe 手把手

> 对应《AI and Machine Learning for Coders》（PyTorch 版）**第 12–13 章**：
> 训练一个模型只是第一步，真正的价值是把它**服务化**——让别的程序通过
> HTTP 调用它做推理。本项目从「训练 → 导出 → Flask 上线 → TorchServe 打包」
> 一条龙走通，全程**离线、CPU、秒级**。

---

## 这个项目在教什么

第 12–13 章的核心信息是：**模型再复杂，上线的套路是固定的**。所以本项目刻意
用一个极小的分类器（4 特征、3 类、一层隐藏层的 MLP），把注意力放在「服务化」而
不是「调模型」上。你会学到：

1. **推理管线的四步拆分**：`preprocess → predict → postprocess`（外加 `load_model`），
   写成纯函数后，无论套 Flask 还是 TorchServe，逻辑都不用改，还能被单元测试。
2. **用 Flask 提供 `/predict` 和 `/health`**：应用工厂模式，`import` 时不启动服务器，
   测试用 `test_client()` 直接打接口，不绑端口、不联网。
3. **TorchServe 自定义 Handler**：`initialize / preprocess / inference / postprocess / handle`，
   与官方 `BaseHandler` 接口一致，既能被 TorchServe 加载，也能被 pytest 单测。

---

## 目录结构 / 每个文件干嘛

| 文件 | 作用 |
| --- | --- |
| `inference.py` | **推理核心**。`TinyClassifier` 模型 + `load_model / preprocess / predict / postprocess` 四步纯函数 + 进程内单例模型。Flask 和 TorchServe 都复用它。 |
| `train_export.py` | 在**确定性合成数据**（三类高斯簇）上训练小分类器，评估后把 `state_dict` 存到 `artifacts/model.pt`。 |
| `app.py` | **Flask 服务**。`GET /health`、`POST /predict`。应用工厂 `create_app()`，`import` 时不 `app.run()`。 |
| `handler.py` | **TorchServe 风格 Handler** 教学范例（`TinyClassifierHandler` 类 + 模块级 `handle` 入口）。 |
| `client.py` | 示例客户端，用 `requests` 打 `/health` 和 `/predict`（仅演示，测试不联网）。 |
| `tests/test_serving.py` | pytest：四步纯函数 + 训练收敛 + Flask 两个端点 + Handler 各方法。 |
| `requirements.txt` | 依赖清单。 |

---

## 3 分钟跑通

```bash
cd projects/05_serving_flask_lab

# 0) （可选）装依赖：torch / flask / requests / pytest 通常已装
pip install -r requirements.txt

# 1) 跑测试，确认一切正常（离线、秒级）
python -m pytest -q

# 2) 训练并导出权重到 artifacts/model.pt
python train_export.py

# 3) 启动 Flask 推理服务（默认 http://127.0.0.1:8080）
python app.py
```

服务起来后，**另开一个终端**试着调用：

```bash
python client.py
```

或者直接用 curl / PowerShell：

```bash
# 健康检查
curl http://127.0.0.1:8080/health

# 单样本预测（明显属于 class_0）
curl -X POST http://127.0.0.1:8080/predict \
     -H "Content-Type: application/json" \
     -d "{\"features\": [2.0, 2.0, 0.0, 0.0]}"
```

返回示例：

```json
{
  "predicted_class": 0,
  "class_name": "class_0",
  "confidence": 0.98,
  "predictions": [
    {"predicted_class": 0, "class_name": "class_0", "confidence": 0.98,
     "probabilities": [0.98, 0.01, 0.01]}
  ]
}
```

> 小贴士：即使你**没先跑** `train_export.py`，服务启动时也会在内存里
> 现训一个回退模型（见 `inference.get_model` → `_train_fallback_model`），
> 所以 `/predict` 永远有意义、不会因为缺文件而崩。

---

## 延伸：真正用 TorchServe 托管（打包 .mar）

`handler.py` 就是给 TorchServe 用的。真实上线时的命令如下（本项目默认不跑，
仅作延伸阅读；需要额外 `pip install torchserve torch-model-archiver`）：

```bash
# 1) 先导出权重
python train_export.py            # 生成 artifacts/model.pt

# 2) 把「权重 + handler + 依赖」打包成一个 .mar 归档
torch-model-archiver \
    --model-name tiny_classifier \
    --version 1.0 \
    --serialized-file artifacts/model.pt \
    --handler handler.py \
    --extra-files inference.py \
    --export-path model_store \
    --force

# 3) 启动 TorchServe，注册这个模型
torchserve --start \
    --model-store model_store \
    --models tiny_classifier=tiny_classifier.mar \
    --ncs \
    --disable-token-auth

# 4) 调用（TorchServe 默认推理端口 8080）
curl -X POST http://127.0.0.1:8080/predictions/tiny_classifier \
     -H "Content-Type: application/json" \
     -d '{"features": [2.0, 2.0, 0.0, 0.0]}'

# 5) 关闭
torchserve --stop
```

打包时要注意：`--serialized-file` 是权重（`.pt`），`--handler` 指向我们的
`handler.py`，`--extra-files` 把 `inference.py` 一起塞进去（因为 handler
`import inference`）。`initialize(context)` 会从 `context.system_properties`
里拿 `model_dir` 和 `serializedFile` 来定位权重——这段逻辑在 `handler.py`
里已经写好了。

---

## 如何换成「真实数据 / 真实 LLM」

本项目为了**离线、秒级、可复现**用的是合成数据 + 小 MLP，但服务化的骨架
可以原样复用。要接真实场景，只需替换「数据」和「模型」两块：

- **换真实数据**：改 `train_export.make_synthetic_data`，从 CSV/Parquet/数据库
  读你的特征与标签即可（比如 `pandas.read_csv` 后转 `torch.tensor`）。同时把
  `inference.NUM_FEATURES / NUM_CLASSES / CLASS_NAMES` 改成你的真实维度与类名。
- **换更大的模型**：把 `inference.TinyClassifier` 换成你的网络（甚至是加载
  `torchvision` 预训练模型）。只要 `forward` 输出仍是「每类一个 logit」，
  `predict / postprocess` 一行都不用改。
- **换成真实 LLM**：把 `inference.load_model` 换成加载 HuggingFace 模型
  （`transformers.AutoModelForSequenceClassification.from_pretrained(...)`），
  把 `preprocess` 从「特征向量」改成「用 tokenizer 编码文本」，`predict` 里
  调用 `model(**inputs).logits`。`app.py` / `handler.py` 依旧不用动——这正是
  「四步纯函数」拆分的价值。建议把「加载真实模型」做成 `try import` 的可选路径，
  缺依赖/缺网络时回退到本项目的离线合成分支，保证 CI 始终绿。

---

## 测试说明

`tests/test_serving.py` 覆盖：

- `preprocess / predict / postprocess` 的形状、键、softmax 归一、异常分支；
- `train_export.train` 的**损失下降**与**训练集准确率 > 90%**；
- Flask `/health`（200 + 字段）、`/predict`（单样本 / 批量 / 坏 JSON→400 / 维度错→400），
  **全程用 `app.test_client()`，不绑端口、不联网**；
- TorchServe `TinyClassifierHandler` 的 `preprocess`（dict / bytes / 批量）、
  `inference`、`postprocess`（返回 list）、`handle`（端到端）与异常分支。

全部在几秒内跑完，CPU 即可。
