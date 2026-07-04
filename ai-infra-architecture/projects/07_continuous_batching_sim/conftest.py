"""让 tests/ 子目录下的用例能 `import continuous_batching`(把项目根塞进 sys.path)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
