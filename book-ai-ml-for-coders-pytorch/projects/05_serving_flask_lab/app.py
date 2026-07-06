# -*- coding: utf-8 -*-
"""
app.py —— 用 Flask 把模型包成 HTTP 推理服务

对应第 13 章「用 Web 框架提供推理接口」。提供两个端点：

    GET  /health   健康检查（探活 / K8s liveness）
    POST /predict  接收 JSON {"features": [...]} -> 返回预测 JSON

要点：
- import 时**不会** app.run()，只有 `python app.py` 直接运行才起服务器；
  这样测试可以用 app.test_client() 而不用真绑端口、不用联网。
- 真正的推理逻辑全在 inference.py，这里只做「HTTP 壳」。

本地起服务：
    python app.py          # 默认 http://127.0.0.1:8080
"""

from flask import Flask, request, jsonify

import inference


def create_app():
    """应用工厂：便于测试与多配置部署。"""
    app = Flask(__name__)

    # 启动时预加载模型（加载一次、请求间复用），避免每次请求都加载
    inference.get_model()

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": "TinyClassifier",
                "num_features": inference.NUM_FEATURES,
                "num_classes": inference.NUM_CLASSES,
            }
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "请求体必须是合法 JSON"}), 400
        try:
            response = inference.run_pipeline(payload)
        except ValueError as exc:
            # 输入不合法（缺字段 / 维度不对）返回 400，附带原因
            return jsonify({"error": str(exc)}), 400
        return jsonify(response)

    return app


# 模块级 app，供 test_client() 与 gunicorn/uwsgi 直接引用（不会自动 run）
app = create_app()


if __name__ == "__main__":
    # 只有直接运行本文件时才真正启动 HTTP 服务器
    app.run(host="127.0.0.1", port=8080, debug=False)
