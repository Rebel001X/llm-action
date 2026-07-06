# -*- coding: utf-8 -*-
"""
test_serving.py —— 离线、CPU、秒级的服务化测试

覆盖：
- inference 的四步纯函数（preprocess / predict / postprocess）
- train_export 的训练（损失下降 + 高准确率）
- Flask 的 /health 与 /predict（全程用 test_client，不绑端口、不联网）
- TorchServe Handler 的 preprocess / inference / postprocess / handle
"""

import json

import torch
import pytest

import inference
from inference import (
    TinyClassifier,
    NUM_FEATURES,
    NUM_CLASSES,
    build_model,
    preprocess,
    predict,
    postprocess,
    run_pipeline,
)
import train_export
from handler import TinyClassifierHandler


# ---------------------------------------------------------------------------
# 一个「训练好的」模型，多个测试复用（在 class_0/1/2 三个簇上高准确率）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_model():
    model, losses = train_export.train(epochs=40)
    return model, losses


# ---------------------------------------------------------------------------
# 1) preprocess 纯函数
# ---------------------------------------------------------------------------
def test_preprocess_single_sample():
    tensor = preprocess({"features": [1.0, 2.0, 3.0, 4.0]})
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, NUM_FEATURES)
    assert tensor.dtype == torch.float32


def test_preprocess_batch():
    tensor = preprocess({"features": [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]})
    assert tensor.shape == (2, NUM_FEATURES)


def test_preprocess_missing_key_raises():
    with pytest.raises(ValueError):
        preprocess({"x": [1.0, 2.0, 3.0, 4.0]})


def test_preprocess_wrong_dim_raises():
    with pytest.raises(ValueError):
        preprocess({"features": [1.0, 2.0]})  # 特征数不对


# ---------------------------------------------------------------------------
# 2) predict 纯函数
# ---------------------------------------------------------------------------
def test_predict_shapes_and_keys():
    model = build_model()
    tensor = preprocess({"features": [1.0, 2.0, 3.0, 4.0]})
    out = predict(tensor, model=model)
    for key in ("logits", "probabilities", "predicted_class", "confidence"):
        assert key in out
    assert len(out["predicted_class"]) == 1
    assert out["predicted_class"][0] in range(NUM_CLASSES)
    # softmax 概率之和约等于 1
    assert abs(sum(out["probabilities"][0]) - 1.0) < 1e-5
    # 置信度在 [0, 1]
    assert 0.0 <= out["confidence"][0] <= 1.0


def test_predict_batch():
    model = build_model()
    tensor = preprocess({"features": [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]})
    out = predict(tensor, model=model)
    assert len(out["predicted_class"]) == 2
    assert len(out["probabilities"]) == 2


def test_trained_model_classifies_cluster_centers(trained_model):
    """训练后的模型应能把三个簇中心分到正确类别。"""
    model, _ = trained_model
    centers = {
        0: [2.0, 2.0, 0.0, 0.0],
        1: [-2.0, 2.0, 0.0, 0.0],
        2: [0.0, -2.0, 2.0, 0.0],
    }
    for cls, feats in centers.items():
        out = predict(preprocess({"features": feats}), model=model)
        assert out["predicted_class"][0] == cls


# ---------------------------------------------------------------------------
# 3) postprocess 纯函数
# ---------------------------------------------------------------------------
def test_postprocess_single_has_top_level_fields():
    model = build_model()
    raw = predict(preprocess({"features": [1.0, 2.0, 3.0, 4.0]}), model=model)
    resp = postprocess(raw)
    assert "predictions" in resp and len(resp["predictions"]) == 1
    # 单样本时，关键字段被提到顶层
    assert "predicted_class" in resp
    assert "class_name" in resp
    assert "confidence" in resp
    assert resp["predicted_class"] in range(NUM_CLASSES)


def test_postprocess_batch_no_top_level_scalar():
    model = build_model()
    raw = predict(
        preprocess({"features": [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]}),
        model=model,
    )
    resp = postprocess(raw)
    assert len(resp["predictions"]) == 2
    # 多样本时不提供顶层标量 predicted_class
    assert "predicted_class" not in resp


# ---------------------------------------------------------------------------
# 4) 训练：损失下降 + 高准确率
# ---------------------------------------------------------------------------
def test_training_loss_decreases(trained_model):
    _, losses = trained_model
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5  # 簇分得很开，最终损失应明显下降


def test_training_accuracy_high(trained_model):
    model, _ = trained_model
    X, y = train_export.make_synthetic_data()
    acc = train_export.accuracy(model, X, y)
    assert acc > 0.9


# ---------------------------------------------------------------------------
# 5) Flask：/health 与 /predict（全程 test_client，不联网、不绑端口）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    import app as app_module
    return app_module.app.test_client()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["num_features"] == NUM_FEATURES
    assert data["num_classes"] == NUM_CLASSES


def test_predict_endpoint_ok(client):
    resp = client.post("/predict", json={"features": [2.0, 2.0, 0.0, 0.0]})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predicted_class" in data
    assert data["predicted_class"] in range(NUM_CLASSES)
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["predictions"]) == 1


def test_predict_endpoint_batch(client):
    resp = client.post(
        "/predict",
        json={"features": [[2.0, 2.0, 0.0, 0.0], [-2.0, 2.0, 0.0, 0.0]]},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["predictions"]) == 2


def test_predict_endpoint_bad_json(client):
    # content_type 声明为 json 但内容不是合法 json -> 400
    resp = client.post("/predict", data="not-json", content_type="application/json")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_endpoint_bad_shape(client):
    resp = client.post("/predict", json={"features": [1.0, 2.0]})  # 特征数不对
    assert resp.status_code == 400
    assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# 6) TorchServe Handler：preprocess / inference / postprocess / handle
# ---------------------------------------------------------------------------
def test_handler_preprocess_dict_body():
    h = TinyClassifierHandler()
    h.initialize()
    tensor = h.preprocess([{"body": {"features": [2.0, 2.0, 0.0, 0.0]}}])
    assert tensor.shape == (1, NUM_FEATURES)


def test_handler_preprocess_bytes_body():
    h = TinyClassifierHandler()
    h.initialize()
    payload = json.dumps({"features": [2.0, 2.0, 0.0, 0.0]}).encode("utf-8")
    tensor = h.preprocess([{"body": payload}])
    assert tensor.shape == (1, NUM_FEATURES)


def test_handler_preprocess_batch():
    h = TinyClassifierHandler()
    h.initialize()
    data = [
        {"body": {"features": [2.0, 2.0, 0.0, 0.0]}},
        {"body": {"features": [-2.0, 2.0, 0.0, 0.0]}},
    ]
    tensor = h.preprocess(data)
    assert tensor.shape == (2, NUM_FEATURES)


def test_handler_postprocess_returns_list():
    h = TinyClassifierHandler()
    h.initialize()
    tensor = h.preprocess([{"body": {"features": [2.0, 2.0, 0.0, 0.0]}}])
    raw = h.inference(tensor)
    out = h.postprocess(raw)
    assert isinstance(out, list)
    assert len(out) == 1
    assert "predicted_class" in out[0]
    assert "confidence" in out[0]


def test_handler_handle_end_to_end():
    h = TinyClassifierHandler()
    data = [{"body": {"features": [2.0, 2.0, 0.0, 0.0]}}]
    result = h.handle(data)  # 内部会自动 initialize
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["predicted_class"] in range(NUM_CLASSES)


def test_handler_missing_features_raises():
    h = TinyClassifierHandler()
    h.initialize()
    with pytest.raises(ValueError):
        h.preprocess([{"body": {"x": [1.0, 2.0, 3.0, 4.0]}}])
