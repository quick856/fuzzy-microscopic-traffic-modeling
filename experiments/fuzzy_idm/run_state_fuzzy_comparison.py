"""Run State-Fuzzy IDM on the three existing Parameter-Fuzzy scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fuzzy_idm.state_fuzzy.fuzzy import (
    TriangularFuzzyNumber,
    alpha_input_grid,
    nesting_violations,
    trajectory_envelope,
)
from fuzzy_idm.state_fuzzy.model import (
    IDMParameters,
    SCENARIOS,
    ScenarioParameters,
    equilibrium_gap,
    simulate_platoon,
)

CONFIG_PATH = ROOT / "configs" / "state_fuzzy_idm.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _values(group: Dict[str, object]) -> Dict[str, float]:
    """Extract numeric values from ``{value, unit, source}`` entries."""
    return {
        name: float(item["value"] if isinstance(item, dict) else item)
        for name, item in group.items()
        if name != "shared_correlated_initial_state"
    }


idm_values = _values(CONFIG["idm"])
IDM_PARAMETERS = IDMParameters(**idm_values)
scenario_config = CONFIG["scenarios"]
SCENARIO_PARAMETERS = ScenarioParameters(
    accelerating_initial_speed=float(scenario_config["accelerating"]["initial_speed"]["value"]),
    accelerating_target_speed=float(scenario_config["accelerating"]["target_speed"]["value"]),
    accelerating_acceleration=float(scenario_config["accelerating"]["acceleration"]["value"]),
    braking_initial_speed=float(scenario_config["braking"]["initial_speed"]["value"]),
    braking_start=float(scenario_config["braking"]["braking_start"]["value"]),
    braking_deceleration=float(scenario_config["braking"]["deceleration"]["value"]),
    periodic_mean_speed=float(scenario_config["periodic"]["mean_speed"]["value"]),
    periodic_amplitude=float(scenario_config["periodic"]["amplitude"]["value"]),
    periodic_omega=float(scenario_config["periodic"]["omega"]["value"]),
    periodic_warmup=float(scenario_config["periodic"]["warmup"]["value"]),
    periodic_transient_cycles=int(scenario_config["periodic"]["transient_cycles"]),
    periodic_analysis_cycles=int(scenario_config["periodic"]["analysis_cycles"]),
)
numerics = CONFIG["numerics"]
ALPHA_LEVELS = tuple(float(value) for value in numerics["alpha_levels"])
GRID_N = int(numerics["grid_n"])
DT = float(numerics["dt"]["value"])
FOLLOWER_COUNT = int(numerics["follower_count"])
NESTING_TOLERANCE = float(numerics["nesting_absolute_tolerance"])
DISPLAYED_FOLLOWERS = {"first": 0, "last": FOLLOWER_COUNT - 1}

# State-fuzziness widths around each scenario's crisp equilibrium state.
fuzzy_values = _values(CONFIG["state_fuzziness"])
INITIAL_GAP_HALF_WIDTH_M = fuzzy_values["initial_gap_half_width"]
INITIAL_SPEED_HALF_WIDTH_M_PER_S = fuzzy_values["initial_speed_half_width"]


def validate_configuration() -> None:
    """Fail early when the reproducibility configuration is inconsistent."""
    if str(numerics["solver"]).upper() != "RK4":
        raise ValueError("This experiment currently supports solver='RK4' only.")
    if DT <= 0.0 or GRID_N < 2 or FOLLOWER_COUNT < 1:
        raise ValueError("Require dt>0, grid_n>=2, and follower_count>=1.")
    if tuple(sorted(ALPHA_LEVELS)) != ALPHA_LEVELS:
        raise ValueError("alpha_levels must be in ascending order.")
    if not ALPHA_LEVELS or ALPHA_LEVELS[0] < 0.0 or ALPHA_LEVELS[-1] != 1.0:
        raise ValueError("alpha_levels must lie in [0,1] and end at 1.")
    if INITIAL_GAP_HALF_WIDTH_M < 0.0 or INITIAL_SPEED_HALF_WIDTH_M_PER_S < 0.0:
        raise ValueError("Fuzzy half widths cannot be negative.")
    if not bool(CONFIG["state_fuzziness"]["shared_correlated_initial_state"]):
        raise ValueError("Only the documented correlated platoon state is supported.")
    maximum_leader_speed = max(
        SCENARIO_PARAMETERS.accelerating_target_speed,
        SCENARIO_PARAMETERS.braking_initial_speed,
        SCENARIO_PARAMETERS.periodic_mean_speed
        + abs(SCENARIO_PARAMETERS.periodic_amplitude),
    )
    if IDM_PARAMETERS.desired_speed <= maximum_leader_speed:
        raise ValueError("IDM desired_speed must exceed every prescribed leader speed.")


validate_configuration()

SCENE_NAMES = {
    "accelerating": "\u5300\u52a0\u901f\u524d\u8f66\u573a\u666f",
    "braking": "\u524d\u8f66\u51cf\u901f\u505c\u8f66\u573a\u666f",
    "periodic": "\u524d\u8f66\u5468\u671f\u6270\u52a8\u573a\u666f",
}


def set_plot_style() -> None:
    """Use a headless backend and Chinese-capable fonts."""
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "DengXian",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def scenario_fuzzy_initial_state(
    scenario: str, parameters: IDMParameters
) -> Tuple[TriangularFuzzyNumber, TriangularFuzzyNumber]:
    """Return fuzzy gap/speed centered on the scenario's crisp initial state."""
    center_speed = (
        SCENARIO_PARAMETERS.accelerating_initial_speed
        if scenario == "accelerating"
        else SCENARIO_PARAMETERS.braking_initial_speed
        if scenario == "braking"
        else SCENARIO_PARAMETERS.periodic_mean_speed
    )
    center_gap = equilibrium_gap(center_speed, parameters)
    fuzzy_gap = TriangularFuzzyNumber(
        center_gap - INITIAL_GAP_HALF_WIDTH_M,
        center_gap,
        center_gap + INITIAL_GAP_HALF_WIDTH_M,
    )
    fuzzy_speed = TriangularFuzzyNumber(
        center_speed - INITIAL_SPEED_HALF_WIDTH_M_PER_S,
        center_speed,
        center_speed + INITIAL_SPEED_HALF_WIDTH_M_PER_S,
    )
    return fuzzy_gap, fuzzy_speed


def run_scenario_family(
    scenario: str,
) -> Tuple[
    np.ndarray,
    Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    Dict[str, np.ndarray],
    Dict[float, int],
    Dict[str, float],
]:
    """Simulate one scenario's alpha-cut initial-state family."""
    parameters = IDM_PARAMETERS
    fuzzy_gap, fuzzy_speed = scenario_fuzzy_initial_state(scenario, parameters)
    envelopes: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    counts: Dict[float, int] = {}
    times: np.ndarray | None = None
    physical_min_gap = np.inf
    physical_min_speed = np.inf
    all_finite = True

    base_inputs = {
        alpha: list(alpha_input_grid(fuzzy_gap, fuzzy_speed, alpha, GRID_N))
        for alpha in ALPHA_LEVELS
    }
    simulation_cache: Dict[
        Tuple[float, float],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    # A separate linspace at every alpha is not a nested sample set.  Each
    # outer alpha domain therefore includes the sampled points of every inner
    # domain.  This preserves sample-level alpha nesting without clipping.
    for alpha in ALPHA_LEVELS:
        union: Dict[Tuple[float, float], Tuple[float, float]] = {}
        for inner_alpha in ALPHA_LEVELS:
            if inner_alpha + 1e-15 < alpha:
                continue
            for initial_gap, initial_speed in base_inputs[inner_alpha]:
                key = (round(float(initial_gap), 14), round(float(initial_speed), 14))
                union[key] = (float(initial_gap), float(initial_speed))

        samples: Dict[str, List[np.ndarray]] = {
            f"{label}_{variable}": []
            for label in DISPLAYED_FOLLOWERS
            for variable in ("x", "speed", "gap", "acceleration")
        }
        for key, (initial_gap, initial_speed) in union.items():
            if key not in simulation_cache:
                simulation_cache[key] = simulate_platoon(
                    scenario,
                    initial_gap,
                    initial_speed,
                    parameters,
                    dt=DT,
                    follower_count=FOLLOWER_COUNT,
                    scenario_parameters=SCENARIO_PARAMETERS,
                )
            (
                current_times,
                positions,
                speeds,
                gaps,
                accelerations,
                _leaders,
            ) = simulation_cache[key]
            times = current_times
            for label, vehicle_index in DISPLAYED_FOLLOWERS.items():
                samples[f"{label}_x"].append(positions[:, vehicle_index])
                samples[f"{label}_speed"].append(speeds[:, vehicle_index])
                samples[f"{label}_gap"].append(gaps[:, vehicle_index])
                samples[f"{label}_acceleration"].append(
                    accelerations[:, vehicle_index]
                )
            physical_min_gap = min(physical_min_gap, float(np.min(gaps)))
            physical_min_speed = min(physical_min_speed, float(np.min(speeds)))
            all_finite = bool(
                all_finite
                and np.all(np.isfinite(positions))
                and np.all(np.isfinite(speeds))
                and np.all(np.isfinite(gaps))
                and np.all(np.isfinite(accelerations))
            )

        counts[alpha] = len(samples["last_speed"])
        envelopes[alpha] = {
            key: trajectory_envelope(value) for key, value in samples.items()
        }

    if times is None:
        raise RuntimeError("No trajectory was simulated.")

    (
        _crisp_times,
        crisp_positions,
        crisp_speeds,
        crisp_gaps,
        crisp_accelerations,
        crisp_leaders,
    ) = simulate_platoon(
        scenario,
        fuzzy_gap.mode,
        fuzzy_speed.mode,
        parameters,
        dt=DT,
        follower_count=FOLLOWER_COUNT,
        scenario_parameters=SCENARIO_PARAMETERS,
    )
    crisp: Dict[str, np.ndarray] = {
        "leader_speed": crisp_leaders[:, 1],
        "leader_acceleration": crisp_leaders[:, 2],
    }
    for label, vehicle_index in DISPLAYED_FOLLOWERS.items():
        crisp[f"{label}_x"] = crisp_positions[:, vehicle_index]
        crisp[f"{label}_speed"] = crisp_speeds[:, vehicle_index]
        crisp[f"{label}_gap"] = crisp_gaps[:, vehicle_index]
        crisp[f"{label}_acceleration"] = crisp_accelerations[:, vehicle_index]
    initial_state = {
        "gap_left_m": fuzzy_gap.left,
        "gap_mode_m": fuzzy_gap.mode,
        "gap_right_m": fuzzy_gap.right,
        "speed_left_m_per_s": fuzzy_speed.left,
        "speed_mode_m_per_s": fuzzy_speed.mode,
        "speed_right_m_per_s": fuzzy_speed.right,
        "minimum_speed_m_per_s": physical_min_speed,
        "minimum_net_gap_m": physical_min_gap,
        "all_finite": all_finite,
    }
    return times, envelopes, crisp, counts, initial_state


def validate_scenario(
    times: np.ndarray,
    envelopes: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    crisp: Dict[str, np.ndarray],
    counts: Dict[float, int],
    diagnostics: Dict[str, float],
) -> Dict[str, object]:
    """Validate crisp degeneration, nesting, finiteness, speed, and gaps."""
    alpha_one_errors = {}
    nesting = {}
    state_keys = [
        f"{label}_{variable}"
        for label in DISPLAYED_FOLLOWERS
        for variable in ("x", "speed", "gap", "acceleration")
    ]
    for key in state_keys:
        low, high = envelopes[1.0][key]
        alpha_one_errors[key] = max(
            float(np.max(np.abs(low - crisp[key]))),
            float(np.max(np.abs(high - crisp[key]))),
        )
        nesting[key] = nesting_violations(
            {alpha: values[key] for alpha, values in envelopes.items()},
            tolerance=NESTING_TOLERANCE,
        )

    nesting_passed = all(
        item["lower"] == 0.0 and item["upper"] == 0.0
        for item in nesting.values()
    )
    passed = bool(
        max(alpha_one_errors.values()) < 1e-8
        and nesting_passed
        and diagnostics["all_finite"]
        and diagnostics["minimum_speed_m_per_s"] >= 0.0
        and diagnostics["minimum_net_gap_m"] > 0.0
        and counts[1.0] == 1
    )
    response_metrics = {}
    for label in DISPLAYED_FOLLOWERS:
        speed_key = f"{label}_speed"
        acceleration_key = f"{label}_acceleration"
        speed_width = (
            envelopes[0.0][speed_key][1] - envelopes[0.0][speed_key][0]
        )
        acceleration_width = (
            envelopes[0.0][acceleration_key][1]
            - envelopes[0.0][acceleration_key][0]
        )
        speed_peak_index = int(np.argmax(speed_width))
        acceleration_peak_index = int(np.argmax(acceleration_width))
        response_metrics[f"{label}_follower"] = {
            "maximum_speed_width_km_per_h": float(
                speed_width[speed_peak_index] * 3.6
            ),
            "time_of_maximum_speed_width_s": float(times[speed_peak_index]),
            "terminal_speed_width_km_per_h": float(speed_width[-1] * 3.6),
            "maximum_acceleration_width_m_per_s2": float(
                acceleration_width[acceleration_peak_index]
            ),
            "time_of_maximum_acceleration_width_s": float(
                times[acceleration_peak_index]
            ),
            "terminal_acceleration_width_m_per_s2": float(
                acceleration_width[-1]
            ),
            "crisp_peak_absolute_acceleration_m_per_s2": float(
                np.max(np.abs(crisp[acceleration_key]))
            ),
        }
    return {
        "passed": passed,
        "trajectory_count_per_alpha": {str(k): v for k, v in counts.items()},
        "alpha_1_max_abs_error": alpha_one_errors,
        "nesting_violation": nesting,
        "initial_state_and_physical_checks": diagnostics,
        "response_metrics": response_metrics,
    }


def plot_variable(
    scenario: str,
    times: np.ndarray,
    envelopes: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    crisp: Dict[str, np.ndarray],
    variable: str,
    output_path: Path,
) -> None:
    """Plot the first and fifth followers using the parameter-fuzzy layout."""
    set_plot_style()
    figure, axis = plt.subplots(figsize=(12.2, 6.7))
    scale = 3.6 if variable == "speed" else 1.0
    styles = {
        "first": {
            "number": 1,
            "band0": "#9ecae1",
            "band05": "#4292c6",
            "line": "#08519c",
        },
        "last": {
            "number": FOLLOWER_COUNT,
            "band0": "#a1d99b",
            "band05": "#fd8d3c",
            "line": "#e6550d",
        },
    }
    for label, style in styles.items():
        key = f"{label}_{variable}"
        low0, high0 = envelopes[0.0][key]
        low05, high05 = envelopes[0.5][key]
        vehicle_number = style["number"]
        axis.fill_between(
            times,
            low0 * scale,
            high0 * scale,
            color=style["band0"],
            alpha=0.18,
            label=f"跟驰车{vehicle_number} 状态模糊：α=0",
        )
        axis.fill_between(
            times,
            low05 * scale,
            high05 * scale,
            color=style["band05"],
            alpha=0.24,
            label=f"跟驰车{vehicle_number} 状态模糊：α=0.5",
        )
        axis.plot(
            times,
            crisp[key] * scale,
            color=style["line"],
            linewidth=1.8,
            label=f"跟驰车{vehicle_number} 确定性 IDM",
        )
    if variable == "speed":
        axis.plot(
            times,
            crisp["leader_speed"] * 3.6,
            color="#c65f36",
            linestyle="--",
            linewidth=1.8,
            label="\u524d\u8f66\u901f\u5ea6",
        )
        axis.set_ylabel("\u901f\u5ea6 (km/h)")
        variable_name = "\u901f\u5ea6\u54cd\u5e94"
    else:
        axis.plot(
            times,
            crisp["leader_acceleration"],
            color="#c65f36",
            linestyle="--",
            linewidth=1.6,
            label="\u524d\u8f66\u52a0\u901f\u5ea6",
        )
        axis.axhline(0.0, color="#777777", linewidth=0.9)
        axis.set_ylabel("\u52a0\u901f\u5ea6 (m/s\u00b2)")
        variable_name = "\u52a0\u901f\u5ea6\u54cd\u5e94"
    if scenario == "periodic":
        axis.axvline(
            SCENARIO_PARAMETERS.periodic_warmup,
            color="#666666",
            linestyle=":",
            linewidth=1.1,
            label="周期扰动开始",
        )
        axis.axvline(
            SCENARIO_PARAMETERS.periodic_analysis_start,
            color="#999999",
            linestyle="--",
            linewidth=1.0,
            label="稳定段统计开始",
        )
    axis.set_xlabel("\u65f6\u95f4 t (s)")
    figure.suptitle(
        f"{SCENE_NAMES[scenario]}\uff1a\u72b6\u6001\u6a21\u7cca IDM {variable_name}",
        y=0.975,
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.77))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    figure_dir = ROOT / "figures" / "state_fuzzy_idm"
    figure_dir.mkdir(parents=True, exist_ok=True)

    validation = {
        "model": "State-Fuzzy IDM comparison with crisp a, b, and T",
        "fixed_parameters": {
            "a_m_per_s2": IDM_PARAMETERS.max_acceleration,
            "b_m_per_s2": IDM_PARAMETERS.comfortable_deceleration,
            "T_s": IDM_PARAMETERS.desired_time_headway,
            "desired_speed_m_per_s": IDM_PARAMETERS.desired_speed,
            "minimum_gap_m": IDM_PARAMETERS.minimum_gap,
            "acceleration_exponent": IDM_PARAMETERS.acceleration_exponent,
            "vehicle_length_m": IDM_PARAMETERS.vehicle_length,
            "config": str(CONFIG_PATH.relative_to(ROOT)),
        },
        "numerical_method": {
            "solver": str(numerics["solver"]),
            "dt_s": DT,
            "alpha_levels": list(ALPHA_LEVELS),
            "grid_n": GRID_N,
            "follower_count": FOLLOWER_COUNT,
            "displayed_followers": [1, FOLLOWER_COUNT],
            "nesting_absolute_tolerance": NESTING_TOLERANCE,
        },
        "state_fuzzification": {
            "shared_correlated_initial_state": True,
            "gap_half_width_m": INITIAL_GAP_HALF_WIDTH_M,
            "speed_half_width_m_per_s": INITIAL_SPEED_HALF_WIDTH_M_PER_S,
            "source": "assumed_for_demo",
        },
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        times, envelopes, crisp, counts, diagnostics = run_scenario_family(scenario)
        scenario_validation = validate_scenario(
            times, envelopes, crisp, counts, diagnostics
        )
        validation["scenarios"][scenario] = scenario_validation
        plot_variable(
            scenario,
            times,
            envelopes,
            crisp,
            "speed",
            figure_dir / f"{scenario}_velocity.png",
        )
        plot_variable(
            scenario,
            times,
            envelopes,
            crisp,
            "acceleration",
            figure_dir / f"{scenario}_acceleration.png",
        )

    validation["passed"] = all(
        item["passed"] for item in validation["scenarios"].values()
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["passed"]:
        raise SystemExit("Validation failed; inspect the scenario diagnostics.")


if __name__ == "__main__":
    main()
