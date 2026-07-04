# -*- coding: utf-8 -*-
"""
pytest 配置:把项目根目录加入 sys.path,让 tests/ 下的用例能
直接 `import scarcity_model`(无需安装成包)。
"""
import os
import sys

# 把本文件所在目录(项目根)插到 sys.path 最前
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
