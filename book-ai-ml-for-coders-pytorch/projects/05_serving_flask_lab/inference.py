# -*- coding: utf-8 -*-
"""
inference.py —— 推理核心（纯函数，便于单元测试）

对应《AI and Machine Learning for Coders》PyTorch 版 第 12-13 章：
把训练好的模型「服务化」。这里把一次推理拆成四个可独立测试的步骤：

    load_model()              加载模型（有磁盘产物就 load，没有就现训/回退）
    preprocess(payload)       把请求 JSON -> torch.Tensor
    predict(tensor) -> dict   前向 + softmax，返回原始结果
    postprocess(result)       把原始结果整理成对用户友好的响应

这样 Flask（app.py）和 TorchServe（handler.py）都只是「壳」，
真正的推理逻辑集中在这里、可以脱离网络框架被 pytest 直接测。
"""

import os

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# 全局约定：特征维度、类别数、类别名、默认权重路径
# ----------------------------------------------------------------------------
NUM_FEATURES = 4          # 每个样本的特征个数（本例：仿鸢尾花的 4 个测量值）
NUM_CLASSES = 3           # 分类类别数
CLASS_NAMES = ["class_0", "class_1", "class_2"]

# 训练脚本会把 state_dict 存到这里；服务启动时优先尝试加载它
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.pt")


class TinyClassifier(nn.Module):
    """一个极小的 MLP 分类器：Linear -> ReLU -> Linear。

    书里第 12 章强调「模型再复杂，导出/服务的套路是一样的」，
    所以这里刻意用最小网络，把注意力放在「服务化」而不是「调模型」。
    """

    def __init__(self, in_features=NUM_FEATURES, hidden=16, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(seed=0):
    """确定性地构造一个（未训练的）模型，固定随机种子保证可复现。"""
    torch.manual_seed(seed)
    model = TinyClassifier()
    model.eval()
    return model


def load_model(path=DEFAULT_MODEL_PATH):
    """加载模型权重。

    - 若 ``path`` 指向的文件存在，则 load 该 state_dict（真实上线路径）；
    - 否则返回一个确定性初始化的模型（不报错，方便离线/测试）。
    """
    model = build_model()
    if path and os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# 进程内单例模型：Flask / TorchServe 在启动时加载一次，之后每次请求复用
# ----------------------------------------------------------------------------
_MODEL = None


def _train_fallback_model():
    """没有磁盘产物时，现场快速训练一个可用模型（保证 demo「开箱即用」）。

    延迟 import train_export，避免与本模块产生循环导入；
    任何异常都回退到未训练模型，绝不让服务启动失败。
    """
    try:
        from train_export import train
        model, _ = train()
        model.eval()
        return model
    except Exception:
        return build_model()


def get_model():
    """返回进程内共享的模型实例（懒加载）。

    优先加载磁盘上的 artifacts/model.pt；不存在则现训一个回退模型。
    """
    global _MODEL
    if _MODEL is None:
        if os.path.exists(DEFAULT_MODEL_PATH):
            _MODEL = load_model(DEFAULT_MODEL_PATH)
        else:
            _MODEL = _train_fallback_model()
    return _MODEL


def set_model(model):
    """允许外部（如测试）注入自己的模型实例，覆盖单例。"""
    global _MODEL
    _MODEL = model
    return _MODEL


# ----------------------------------------------------------------------------
# 四步推理管线（都是纯函数，方便单测）
# ----------------------------------------------------------------------------
def preprocess(payload):
    """把请求体 JSON 转成模型输入张量，形状 (N, NUM_FEATURES)。

    支持两种写法：
        单样本：{"features": [f0, f1, f2, f3]}
        批量：  {"features": [[...], [...], ...]}
    """
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError("payload 必须是含有键 'features' 的字典")

    features = payload["features"]
    try:
        tensor = torch.as_tensor(features, dtype=torch.float32)
    except Exception as exc:  # 非数值等
        raise ValueError(f"'features' 无法转成浮点张量: {exc}")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # 单样本补 batch 维
    if tensor.ndim != 2:
        raise ValueError(f"'features' 维度非法，期望 1D 或 2D，得到 {tensor.ndim}D")
    if tensor.shape[-1] != NUM_FEATURES:
        raise ValueError(
            f"每个样本需要 {NUM_FEATURES} 个特征，实际得到 {tensor.shape[-1]}"
        )
    return tensor


def predict(tensor, model=None):
    """前向推理 + softmax，返回原始结果字典（都用 Python 原生类型，便于 JSON 化）。"""
    if model is None:
        model = get_model()
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)
    return {
        "logits": logits.tolist(),
        "probabilities": probs.tolist(),
        "predicted_class": predicted.tolist(),
        "confidence": confidence.tolist(),
    }


def postprocess(result):
    """把 predict 的原始结果整理成对用户友好的响应。

    - 始终包含 ``predictions`` 列表（每个样本一项）；
    - 若只有一个样本，额外把关键字段提到顶层，方便前端直接取用。
    """
    preds = result["predicted_class"]
    probs = result["probabilities"]
    confs = result["confidence"]

    items = []
    for i, cls in enumerate(preds):
        items.append(
            {
                "predicted_class": int(cls),
                "class_name": CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls),
                "confidence": round(float(confs[i]), 6),
                "probabilities": [round(float(p), 6) for p in probs[i]],
            }
        )

    response = {"predictions": items}
    if len(items) == 1:
        response["predicted_class"] = items[0]["predicted_class"]
        response["class_name"] = items[0]["class_name"]
        response["confidence"] = items[0]["confidence"]
    return response


def run_pipeline(payload, model=None):
    """一站式：preprocess -> predict -> postprocess，供 app / handler 复用。"""
    tensor = preprocess(payload)
    raw = predict(tensor, model=model)
    return postprocess(raw)
