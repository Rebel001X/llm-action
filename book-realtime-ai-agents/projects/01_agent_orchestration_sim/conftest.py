# -*- coding: utf-8 -*-
"""
conftest.py —— 让 pytest 能直接 import 项目根目录下的模块。

本项目用「扁平模块」布局（tools.py / agent.py 等直接在项目根），
把项目根目录加进 sys.path，tests/ 下就能 `from agent import ...`。
"""
import os
import sys

# 项目根 = 本文件所在目录
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
