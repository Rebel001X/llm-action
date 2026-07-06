# -*- coding: utf-8 -*-
"""
client.py —— 示例客户端：向运行中的 Flask 服务发请求

仅作演示。**测试不会导入或运行它，也不会联网。**
先在一个终端起服务：
    python app.py
再在另一个终端运行：
    python client.py
"""

import requests

DEFAULT_URL = "http://127.0.0.1:8080"


def call_health(base_url=DEFAULT_URL):
    """调用 GET /health。"""
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def call_predict(features, base_url=DEFAULT_URL):
    """调用 POST /predict。

    features 可以是单样本 [f0, f1, f2, f3]，也可以是批量 [[...], [...]]。
    """
    resp = requests.post(f"{base_url}/predict", json={"features": features}, timeout=10)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    print("健康检查:", call_health())
    # 一个明显属于 class_0 的样本（中心在 [2, 2, 0, 0] 附近）
    print("单样本预测:", call_predict([2.0, 2.0, 0.0, 0.0]))
    # 批量预测
    print("批量预测:", call_predict([[2.0, 2.0, 0.0, 0.0], [-2.0, 2.0, 0.0, 0.0]]))
