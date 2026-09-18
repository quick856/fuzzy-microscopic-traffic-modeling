"""运行 T̃、ζ̃ 的 α 截集二维模糊仿真，只输出中文 PNG 和终端验证指标。"""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from two_dimensional_fuzzy.dynamics import parameters_from_config
from two_dimensional_fuzzy.fuzzy import (
    apply_parameter_values,
    fuzzy_parameters_from_config,
    nesting_violations,
    parameter_grid,
    trajectory_envelope,
)
from two_dimensional_fuzzy.plotting import plot_fuzzy_results
from two_dimensional_fuzzy.scenarios import make_leader, simulate_crisp


def _values(config: dict, group: str) -> dict:
    return {name: item["value"] for name, item in config[group].items()}


def main() -> int:
    config_path = ROOT / "configs/two_dimensional_fuzzy.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fuzzy_config = config["fuzzy_next_stage"]
    fuzzy_parameters = fuzzy_parameters_from_config(config)
    alpha_levels = tuple(float(value) for value in fuzzy_config["alpha_levels"])
    grid_n = int(fuzzy_config["grid_n"])
    scenario = _values(config, "scenario")
    dt = float(config["numerics"]["dt"]["value"])
    offset = float(scenario["offsets"][2])
    duration = float(scenario["following_duration"])

    center_values = {parameter.name: parameter.number.mode for parameter in fuzzy_parameters}
    center_config = deepcopy(config)
    apply_parameter_values(center_config, fuzzy_parameters, center_values)
    center_params = parameters_from_config(center_config)

    # 前车轨迹只由中心参数构造一次，所有模糊参数样本共用同一确定性前车。
    leader = make_leader(center_config, center_params, accelerating=True)
    cache: dict[tuple[tuple[str, float], ...], dict[str, np.ndarray]] = {}

    def simulate(values: dict[str, float]) -> dict[str, np.ndarray]:
        key = tuple((name, round(float(values[name]), 14)) for name in sorted(values))
        if key not in cache:
            sample_config = deepcopy(config)
            apply_parameter_values(sample_config, fuzzy_parameters, values)
            params = parameters_from_config(sample_config)
            cache[key] = simulate_crisp(sample_config, params, leader, offset, duration, dt)
        return cache[key]

    crisp = simulate(center_values)
    envelopes: dict[float, dict[str, np.ndarray]] = {}
    samples_by_alpha: dict[float, list[dict[str, np.ndarray]]] = {}
    combination_counts: dict[str, int] = {}
    base_combinations: dict[float, list[dict[str, float]]] = {}
    for alpha in alpha_levels:
        combinations = parameter_grid(fuzzy_parameters, alpha, grid_n)
        base_combinations[alpha] = combinations
        combination_counts[str(alpha)] = len(combinations)

    # 独立 linspace 网格并不天然嵌套。对每个 Ωα，在自身 5×5 网格基础上
    # 加入所有更高 α 的内部网格点；这些点严格属于当前 Ωα，不是包络裁剪。
    envelope_sample_counts: dict[str, int] = {}
    for alpha in alpha_levels:
        union: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}
        for inner_alpha in alpha_levels:
            if inner_alpha + 1e-15 < alpha:
                continue
            for values in base_combinations[inner_alpha]:
                key = tuple((name, round(float(values[name]), 14)) for name in sorted(values))
                union[key] = values
        samples = [simulate(values) for values in union.values()]
        envelopes[alpha] = trajectory_envelope(samples)
        samples_by_alpha[alpha] = samples
        envelope_sample_counts[str(alpha)] = len(samples)

    one = envelopes[1.0]
    alpha_one_errors = {
        name: float(np.max(np.abs(one["state_lower"][:, index] - crisp["state"][:, index])))
        for index, name in enumerate(("x", "vx", "y", "vy"))
    }
    alpha_one_maximum = max(alpha_one_errors.values())
    nesting = nesting_violations(envelopes)

    geometry = _values(config, "geometry")
    ay_limit = float(config["diagnostic_gate"]["absolute_ay_limit"]["value"])
    physical_pass = True
    physical_worst = {
        "minimum_vx": float("inf"),
        "minimum_headway": float("inf"),
        "minimum_road_margin": float("inf"),
        "maximum_abs_ay": 0.0,
    }
    for run in cache.values():
        state = run["state"]
        finite = all(bool(np.all(np.isfinite(value))) for value in run.values())
        minimum_vx = float(np.min(state[:, 1]))
        minimum_headway = float(np.min(run["headway"]))
        road_margin = float(min(
            np.min(state[:, 2] - geometry["boundary_left"]),
            np.min(geometry["boundary_right"] - state[:, 2]),
        ))
        maximum_ay = float(np.max(np.abs(run["acceleration"][:, 1])))
        physical_worst["minimum_vx"] = min(physical_worst["minimum_vx"], minimum_vx)
        physical_worst["minimum_headway"] = min(physical_worst["minimum_headway"], minimum_headway)
        physical_worst["minimum_road_margin"] = min(physical_worst["minimum_road_margin"], road_margin)
        physical_worst["maximum_abs_ay"] = max(physical_worst["maximum_abs_ay"], maximum_ay)
        physical_pass &= bool(
            finite and minimum_vx >= 0 and minimum_headway > 0
            and road_margin > 0 and maximum_ay < ay_limit
        )

    convergence_envelopes: dict[int, dict[str, np.ndarray]] = {}
    for size in (int(value) for value in fuzzy_config["grid_convergence"]):
        combinations = parameter_grid(fuzzy_parameters, 0.0, size)
        convergence_envelopes[size] = trajectory_envelope([simulate(values) for values in combinations])
    reference_size = max(convergence_envelopes)
    reference = convergence_envelopes[reference_size]
    convergence: dict[str, dict[str, float]] = {}
    for size, envelope in convergence_envelopes.items():
        if size == reference_size:
            continue
        convergence[str(size)] = {}
        for index, name in ((1, "vx"), (2, "y")):
            convergence[str(size)][f"lower_{name}"] = float(np.max(np.abs(
                envelope["state_lower"][:, index] - reference["state_lower"][:, index]
            )))
            convergence[str(size)][f"upper_{name}"] = float(np.max(np.abs(
                envelope["state_upper"][:, index] - reference["state_upper"][:, index]
            )))

    center_y = float(geometry["lane_center"])
    final_center_tolerance = float(config["diagnostic_gate"]["final_center_tolerance"]["value"])
    final_vy_tolerance = float(config["diagnostic_gate"]["final_lateral_speed_tolerance"]["value"])
    maximum_final_center_error = max(
        float(abs(run["state"][-1, 2] - center_y)) for run in cache.values()
    )
    maximum_final_abs_vy = max(
        float(abs(run["state"][-1, 3])) for run in cache.values()
    )
    center_recovery_pass = bool(
        maximum_final_center_error <= final_center_tolerance
        and maximum_final_abs_vy <= final_vy_tolerance
    )

    figure_dir = ROOT / "figures/two_dimensional/fuzzy"
    figures = plot_fuzzy_results(
        crisp, envelopes, samples_by_alpha[0.0], figure_dir, center_config
    )

    zero = envelopes[0.0]
    width_vx = zero["state_upper"][:, 1] - zero["state_lower"][:, 1]
    width_y = zero["state_upper"][:, 2] - zero["state_lower"][:, 2]
    index_vx = int(np.argmax(width_vx))
    index_y = int(np.argmax(width_y))
    success = bool(
        alpha_one_maximum < 1e-8 and nesting["pass"]
        and physical_pass and center_recovery_pass
    )
    output = {
        "status": "pass" if success else "fail",
        "fuzzy_parameters": center_values,
        "alpha_levels": alpha_levels,
        "grid_n": grid_n,
        "deterministic_combinations_per_alpha": combination_counts,
        "envelope_samples_after_nested_grid_augmentation": envelope_sample_counts,
        "unique_deterministic_runs": len(cache),
        "alpha_one_state_errors": alpha_one_errors,
        "alpha_one_maximum_error": alpha_one_maximum,
        "nesting": nesting,
        "physical_pass": physical_pass,
        "physical_worst": physical_worst,
        "center_recovery_pass": center_recovery_pass,
        "maximum_final_center_error_m": maximum_final_center_error,
        "maximum_final_abs_vy_m_s": maximum_final_abs_vy,
        "final_center_tolerance_m": final_center_tolerance,
        "final_lateral_speed_tolerance_m_s": final_vy_tolerance,
        "grid_convergence_against_n7": convergence,
        "maximum_vx_width_m_s": float(width_vx[index_vx]),
        "time_of_maximum_vx_width_s": float(zero["time"][index_vx]),
        "maximum_y_width_m": float(width_y[index_y]),
        "time_of_maximum_y_width_s": float(zero["time"][index_y]),
        "figure_count": len(figures),
        "figure_directory": str(figure_dir),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
