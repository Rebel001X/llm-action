# -*- coding: utf-8 -*-
"""
pytest 配置：把项目根目录（本文件所在目录）加入 sys.path，
这样 tests/ 里的 `from latency_budget import ...` 就能找到模块。

放在项目根的 conftest.py 会被 pytest 在收集用例前自动加载，是让
「测试文件在子目录、被测模块在父目录」正常工作的最简做法。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
