# -*- coding: utf-8 -*-
"""
handler.py —— TorchServe 风格的自定义 Handler（教学范例）

对应第 13 章的延伸：把同一个模型交给 **TorchServe** 托管。
TorchServe 要求提供一个带 initialize/preprocess/inference/postprocess
（或统一入口 handle）的 Handler。这里给出与官方 BaseHandler 一致的接口，
既能被 TorchServe 加载，也能脱离 TorchServe 被 pytest 直接单测。

真实打包成 .mar 的命令见 README。TorchServe 调用顺序大致是：
    initialize(context)          # 启动时一次
    handle(data, context)        # 每个（批）请求
        -> preprocess(data)
        -> inference(tensor)
        -> postprocess(raw)
"""

import json

import inference as core  # 用别名，避免和下面的方法名 inference() 混淆


class TinyClassifierHandler:
    """TorchServe 自定义 Handler：把 inference.py 的四步管线包成 TS 约定的接口。"""

    def __init__(self):
        self.model = None
        self.initialized = False

    # ------------------------------------------------------------------
    # 1) 初始化：TorchServe 启动 worker 时调用一次
    # ------------------------------------------------------------------
    def initialize(self, context=None):
        """加载模型。

        真实 TorchServe 会传入 context，里面能拿到 model_dir 与序列化文件名：
            props = context.system_properties
            model_dir = props.get("model_dir")
            serialized_file = context.manifest["model"]["serializedFile"]
        这里做了兼容：context 为 None（本地单测）时走默认加载路径。
        """
        model_path = None
        if context is not None:
            try:
                import os

                props = context.system_properties
                model_dir = props.get("model_dir")
                serialized_file = context.manifest["model"]["serializedFile"]
                model_path = os.path.join(model_dir, serialized_file)
            except Exception:
                model_path = None

        self.model = core.load_model(model_path) if model_path else core.load_model()
        self.model.eval()
        self.initialized = True
        return self.model

    # ------------------------------------------------------------------
    # 2) 预处理：TorchServe 把「一批」请求作为 list 传进来
    # ------------------------------------------------------------------
    def preprocess(self, data):
        """把 TorchServe 的批量请求转成一个 (N, NUM_FEATURES) 张量。

        data 形如：[{"body": <payload>}, {"data": <payload>}, ...]
        其中 payload 可能是 dict、JSON 字符串、或 bytes。
        """
        batch_features = []
        for row in data:
            payload = row.get("body") if isinstance(row, dict) else row
            if payload is None and isinstance(row, dict):
                payload = row.get("data")
            if payload is None:
                payload = row

            if isinstance(payload, (bytes, bytearray)):
                payload = json.loads(payload.decode("utf-8"))
            elif isinstance(payload, str):
                payload = json.loads(payload)

            if not isinstance(payload, dict) or "features" not in payload:
                raise ValueError("每个请求都需含有键 'features'")
            batch_features.append(payload["features"])

        # 复用 inference.preprocess 做维度校验与张量化
        return core.preprocess({"features": batch_features})

    # ------------------------------------------------------------------
    # 3) 推理
    # ------------------------------------------------------------------
    def inference(self, tensor):
        if not self.initialized:
            self.initialize()
        return core.predict(tensor, model=self.model)

    # ------------------------------------------------------------------
    # 4) 后处理：TorchServe 期望返回「一个 list，每个输入对应一项」
    # ------------------------------------------------------------------
    def postprocess(self, raw):
        response = core.postprocess(raw)
        return response["predictions"]

    # ------------------------------------------------------------------
    # 统一入口：TorchServe 每个请求会调用 handle
    # ------------------------------------------------------------------
    def handle(self, data, context=None):
        if not self.initialized:
            self.initialize(context)
        tensor = self.preprocess(data)
        raw = self.inference(tensor)
        return self.postprocess(raw)


# TorchServe 若用「模块级入口函数」模式，可用下面这个 _service 单例：
_service = TinyClassifierHandler()


def handle(data, context):
    """TorchServe 的模块级入口函数（另一种注册方式）。"""
    if data is None:
        return None
    if not _service.initialized:
        _service.initialize(context)
    return _service.handle(data, context)
