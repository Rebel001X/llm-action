# 让 tests/ 子目录下的用例能 `import quantize`:
# pytest 默认 import 模式会把每个 conftest.py 所在目录加入 sys.path。
# 放一个空 conftest 在项目根目录即可让根目录上 sys.path,从而导入 quantize.py。
