# -*- coding: utf-8 -*-
"""让 pytest 能从项目根目录导入 data / models / engine 模块。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
