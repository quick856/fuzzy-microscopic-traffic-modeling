"""依次运行参数模糊、环境几何模糊及三类实验对照。"""

from separate_fuzzy_stability import main
from compare_with_state_fuzzy import main as compare


if __name__ == "__main__":
    code_parameter = main("parameter")
    code_geometry = main("geometry")
    compare()
    raise SystemExit(max(code_parameter, code_geometry))
