# -*- coding: utf-8 -*-
"""让 tests/ 里的用例能直接 import 项目根目录下的 app / inference / handler。"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
