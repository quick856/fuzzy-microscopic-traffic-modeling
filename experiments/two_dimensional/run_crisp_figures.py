"""运行确定性二维场景，只生成中文结果图，不写报告或 NPZ。"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from two_dimensional_fuzzy.dynamics import parameters_from_config
from two_dimensional_fuzzy.plotting import plot_following, plot_recovery
from two_dimensional_fuzzy.scenarios import make_leader, simulate_crisp


def _values(config: dict, group: str) -> dict:
    return {name: item["value"] for name, item in config[group].items()}


def main() -> int:
    config = json.loads(
        (ROOT / "configs/two_dimensional_crisp_diagnostic.json").read_text(encoding="utf-8")
    )
    params = parameters_from_config(config)
    scenario = _values(config, "scenario")
    dt = float(config["numerics"]["dt"]["value"])
    offsets = scenario["offsets"]
    figure_dir = ROOT / "figures/two_dimensional/crisp_diagnostic"

    constant_leader = make_leader(config, params, accelerating=False)
    recovery = {
        name: simulate_crisp(
            config, params, constant_leader, float(offset),
            float(scenario["recovery_duration"]), dt,
        )
        for name, offset in zip(("center", "left", "right"), offsets)
    }

    following = simulate_crisp(
        config, params, make_leader(config, params, accelerating=True),
        float(offsets[2]), float(scenario["following_duration"]), dt,
    )

    figures = plot_recovery(recovery, figure_dir, config)
    figures += plot_following(following, figure_dir, config)
    arrays = [value for run in (*recovery.values(), following) for value in run.values()]
    finite = all(bool(np.all(np.isfinite(value))) for value in arrays)
    minimum_headway = float(np.min(following["headway"]))
    maximum_ay = float(np.max(np.abs(following["acceleration"][:, 1])))
    print(json.dumps({
        "status": "pass" if finite and minimum_headway > 0 else "fail",
        "figure_count": len(figures),
        "minimum_headway_m": minimum_headway,
        "maximum_lateral_acceleration_m_s2": maximum_ay,
        "figure_directory": str(figure_dir),
    }, ensure_ascii=False, indent=2))
    return 0 if finite and minimum_headway > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
