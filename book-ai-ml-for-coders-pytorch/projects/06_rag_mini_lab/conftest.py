# -*- coding: utf-8 -*-
"""让 tests/ 下的用例能直接 `import rag / embed / store / knowledge`。

pytest 收集 tests/ 里的用例时，默认只把 tests/ 加进 sys.path；
把项目根目录也插进去，模块就能被找到。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
