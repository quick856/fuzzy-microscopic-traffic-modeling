"""二维六状态模糊化及局部 Lyapunov 稳定性诊断。

本实验只模糊初始状态 [x,u,z_lon,y,v,z_lat]。道路几何、IDM 参数、
横向参数、OU 参数、前车和邻车轨迹均保持确定。所有模糊样本共用相同
Wiener 增量，因此包络表示给定随机实现下的初始状态不确定性传播。
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "state_fuzzy_only.json"
OUTPUT = HERE / "results"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
REPO = Path(SPEC["base_repository"])
sys.path.insert(0, str(REPO / "src"))

from two_dimensional_fuzzy_neighbor_ou.dynamics import (  # noqa: E402
    lateral_acceleration_components,
    stochastic_parameters_from_config,
    two_dimensional_stochastic_diffusion,
    two_dimensional_stochastic_drift,
)
from two_dimensional_fuzzy_neighbor_ou.fuzzy import TriangularFuzzyNumber  # noqa: E402
from two_dimensional_fuzzy_neighbor_ou.longitudinal import equilibrium_headway  # noqa: E402
from two_dimensional_fuzzy_neighbor_ou.neighbor import (  # noqa: E402
    is_neighbor_in_influence_region,
    neighbor_interaction_force,
)
from two_dimensional_fuzzy_neighbor_ou.scenarios import (  # noqa: E402
    configure_neighbor_validation_scenario,
    make_neighbor_trajectory,
)
from two_dimensional_fuzzy_neighbor_ou.solver import solve_sde_euler_maruyama  # noqa: E402


COLORS = {
    "center": "#174F78",
    "outer": "#9BC4DF",
    "inner": "#4D8EB9",
    "leader": "#C06442",
    "neighbor": "#2C8C7B",
    "road": "#46515B",
    "mark": "#B4943C",
    "lyapunov": "#7B4EA3",
}
PRIMES = (2, 3, 5, 7, 11, 13)
STATE_NAMES = ("x", "u", "z_lon", "y", "v", "z_lat")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 14.0,
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.9,
        "savefig.dpi": 300,
    })


def finish(fig: plt.Figure, ax: plt.Axes, filename: str) -> Path:
    ax.grid(True, color="#CBD2D9", alpha=0.45, linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                   ncol=min(4, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = OUTPUT / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index > 0:
        result += factor * (index % base)
        index //= base
        factor /= base
    return result


def unit_design(dimension: int, halton_count: int) -> np.ndarray:
    center = np.full(dimension, 0.5)
    points = [center.copy(), np.zeros(dimension), np.ones(dimension)]
    for axis in range(dimension):
        lower = center.copy()
        upper = center.copy()
        lower[axis] = 0.0
        upper[axis] = 1.0
        points.extend((lower, upper))
    for index in range(1, halton_count + 1):
        points.append(np.array([
            radical_inverse(index, PRIMES[axis]) for axis in range(dimension)
        ], dtype=float))
    unique = {}
    for point in points:
        unique[tuple(np.round(point, 14))] = point
    return np.vstack(list(unique.values()))


def fuzzy_numbers() -> dict[str, TriangularFuzzyNumber]:
    return {
        name: TriangularFuzzyNumber(
            float(item["left"]), float(item["mode"]), float(item["right"])
        )
        for name, item in SPEC["triangles"].items()
    }


def sample_alpha_domain(
    numbers: dict[str, TriangularFuzzyNumber], alpha: float, count: int
) -> list[dict[str, float]]:
    points = unit_design(len(STATE_NAMES), count)
    cuts = [numbers[name].alpha_cut(alpha) for name in STATE_NAMES]
    samples = {}
    for point in points:
        values = {
            name: float(lower + coordinate * (upper - lower))
            for name, coordinate, (lower, upper) in zip(STATE_NAMES, point, cuts)
        }
        key = tuple((name, round(values[name], 14)) for name in STATE_NAMES)
        samples[key] = values
    return list(samples.values())


def sample_key(values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple((name, round(float(values[name]), 14)) for name in STATE_NAMES)


def make_leader(base_config: dict, params) -> Callable[[float], np.ndarray]:
    cfg = SPEC["uniformly_accelerating_leader"]
    speed0 = float(cfg["initial_speed"]["value"])
    acceleration = float(cfg["acceleration"]["value"])
    start = float(cfg["acceleration_start"]["value"])
    end = float(cfg["acceleration_end"]["value"])
    y = float(cfg["lateral_position"]["value"])
    x0 = float(SPEC["triangles"]["x"]["mode"]) + equilibrium_headway(speed0, params.idm)

    def trajectory(t: float) -> np.ndarray:
        active = min(max(t - start, 0.0), end - start)
        after = max(t - end, 0.0)
        return np.array([
            x0 + speed0 * t + 0.5 * acceleration * active**2
            + acceleration * (end - start) * after,
            speed0 + acceleration * active,
            y,
            0.0,
        ], dtype=float)

    return trajectory


def make_neighbor(config: dict) -> Callable[[float], np.ndarray]:
    lateral = make_neighbor_trajectory(config)
    scenario = {name: item["value"] for name, item in config["neighbor_scenario"].items()}
    x0 = float(config["scenario"]["initial_x"]["value"]) + float(
        scenario["initial_longitudinal_gap"]
    )
    speed0 = float(scenario["speed"])
    acceleration = float(SPEC["deterministic_neighbor"]["longitudinal_acceleration"]["value"])
    leader_cfg = SPEC["uniformly_accelerating_leader"]
    start = float(leader_cfg["acceleration_start"]["value"])
    end = float(leader_cfg["acceleration_end"]["value"])

    def trajectory(t: float) -> np.ndarray:
        lat = lateral(t)
        active = min(max(t - start, 0.0), end - start)
        after = max(t - end, 0.0)
        return np.array([
            x0 + speed0 * t + 0.5 * acceleration * active**2
            + acceleration * (end - start) * after,
            speed0 + acceleration * active,
            lat[2],
            lat[3],
        ], dtype=float)

    return trajectory


def simulate_case(
    config: dict,
    values: dict[str, float],
    leader: Callable[[float], np.ndarray],
    neighbor: Callable[[float], np.ndarray],
) -> dict[str, np.ndarray]:
    params = stochastic_parameters_from_config(config)
    initial = np.array([values[name] for name in STATE_NAMES], dtype=float)
    duration = float(SPEC["numerics"]["duration"]["value"])
    dt = float(SPEC["numerics"]["dt"]["value"])
    seed = int(SPEC["numerics"]["random_seed"]["value"])

    def drift(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_drift(
            t, state, leader(t), params, (neighbor(t),)
        )

    def diffusion(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_diffusion(t, state, params)

    time, state = solve_sde_euler_maruyama(
        drift, diffusion, initial, duration, dt, seed
    )
    leaders = np.array([leader(float(t)) for t in time])
    neighbors = np.array([neighbor(float(t)) for t in time])
    acceleration = np.array([
        drift(float(t), current)[[1, 4]] for t, current in zip(time, state)
    ])
    compact = state[:, [0, 1, 3, 4]]
    neighbor_force = np.array([
        neighbor_interaction_force(current, other, params.deterministic.lateral,
                                   params.deterministic.neighbor)
        for current, other in zip(compact, neighbors)
    ])
    active = np.array([
        is_neighbor_in_influence_region(current, other, params.deterministic.lateral,
                                        params.deterministic.neighbor)
        for current, other in zip(compact, neighbors)
    ], dtype=bool)
    return {
        "time": time,
        "state": state,
        "leader": leaders,
        "neighbor": neighbors,
        "acceleration": acceleration,
        "neighbor_force": neighbor_force,
        "active": active,
        "headway": leaders[:, 0] - state[:, 0],
    }


ARRAY_KEYS = ("state", "acceleration", "neighbor_force", "headway")


def envelope(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {"time": samples[0]["time"]}
    for key in ARRAY_KEYS:
        stack = np.stack([sample[key] for sample in samples])
        result[key + "_lower"] = np.min(stack, axis=0)
        result[key + "_upper"] = np.max(stack, axis=0)
    return result


def nesting_check(envelopes: dict[float, dict[str, np.ndarray]]) -> dict[str, object]:
    worst = 0.0
    alphas = sorted(envelopes)
    for key in ARRAY_KEYS:
        for outer_alpha, inner_alpha in zip(alphas[:-1], alphas[1:]):
            outer = envelopes[outer_alpha]
            inner = envelopes[inner_alpha]
            worst = max(
                worst,
                float(np.max(outer[key + "_lower"] - inner[key + "_lower"])),
                float(np.max(inner[key + "_upper"] - outer[key + "_upper"])),
            )
    return {"pass": bool(worst <= 1e-10), "maximum_violation": max(worst, 0.0)}


def numerical_jacobian(function: Callable[[np.ndarray], np.ndarray], dimension: int) -> np.ndarray:
    step = float(SPEC["lyapunov"]["finite_difference_step"]["value"])
    jacobian = np.empty((dimension, dimension), dtype=float)
    origin = np.zeros(dimension)
    for column in range(dimension):
        delta = np.zeros(dimension)
        delta[column] = step
        jacobian[:, column] = (function(origin + delta) - function(origin - delta)) / (2.0 * step)
    return jacobian


def solve_lyapunov_matrix(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """求解 A^T P + P A = -I；使用列优先 vec 恒等式。"""
    n = a.shape[0]
    system = np.kron(np.eye(n), a.T) + np.kron(a.T, np.eye(n))
    vector = np.linalg.solve(system, -np.eye(n).reshape(-1, order="F"))
    p = vector.reshape((n, n), order="F")
    p = 0.5 * (p + p.T)
    residual = a.T @ p + p @ a + np.eye(n)
    return p, residual


def lyapunov_analysis(config: dict, center: dict, family: list[dict]) -> dict[str, object]:
    params = stochastic_parameters_from_config(config)
    leader_cfg = SPEC["uniformly_accelerating_leader"]
    u_eq = float(leader_cfg["initial_speed"]["value"]) + float(
        leader_cfg["acceleration"]["value"]
    ) * (float(leader_cfg["acceleration_end"]["value"]) - float(
        leader_cfg["acceleration_start"]["value"]
    ))
    h_eq = equilibrium_headway(u_eq, params.deterministic.idm)
    y_eq = float(config["geometry"]["lane_center"]["value"])
    mu_lon = params.noise.mu_lon
    mu_lat = params.noise.mu_lat

    def autonomous_error_drift(error: np.ndarray) -> np.ndarray:
        h, du, dz_lon, dy, dv, dz_lat = error
        leader_state = np.array([0.0, u_eq, y_eq, 0.0])
        vehicle_state = np.array([
            -(h_eq + h), u_eq + du, mu_lon + dz_lon,
            y_eq + dy, dv, mu_lat + dz_lat,
        ])
        raw = two_dimensional_stochastic_drift(
            0.0, vehicle_state, leader_state, params, ()
        )
        return np.array([-du, raw[1], raw[2], raw[3], raw[4], raw[5]])

    equilibrium_residual = autonomous_error_drift(np.zeros(6))
    a = numerical_jacobian(autonomous_error_drift, 6)
    eigenvalues = np.linalg.eigvals(a)
    hurwitz = bool(np.max(np.real(eigenvalues)) < -1e-9)
    p, residual = solve_lyapunov_matrix(a)
    p_eigenvalues = np.linalg.eigvalsh(p)
    residual_norm = float(np.linalg.norm(residual, ord=2))
    p_positive = bool(np.min(p_eigenvalues) > 1e-10)

    time = center["time"]
    gate = float(SPEC["lyapunov"]["analysis_start"]["value"])
    gate_index = int(np.searchsorted(time, gate))
    all_v = []
    all_dv = []
    for sample in family:
        delta = sample["state"] - center["state"]
        error = delta.copy()
        error[:, 0] *= -1.0  # 同一前车下，delta h = -delta x。
        values = np.einsum("ti,ij,tj->t", error, p, error)
        derivative = np.gradient(values, time)
        all_v.append(values)
        all_dv.append(derivative)
    v_stack = np.stack(all_v)
    dv_stack = np.stack(all_dv)
    active_mask = v_stack[:, gate_index:] > 1e-12
    nonincrease_fraction = float(np.mean(dv_stack[:, gate_index:][active_mask] <= 1e-8))
    v_max = np.max(v_stack, axis=0)
    v_gate = float(v_max[gate_index])
    v_final = float(v_max[-1])
    contraction_ratio = v_final / v_gate if v_gate > 0 else 0.0
    contraction_credit = float(np.clip(1.0 - np.sqrt(max(contraction_ratio, 0.0)), 0.0, 1.0))

    return {
        "relative_state_order": ["delta_h", "delta_u", "delta_z_lon", "delta_y", "delta_v", "delta_z_lat"],
        "terminal_equilibrium_speed_mps": u_eq,
        "terminal_equilibrium_headway_m": float(h_eq),
        "equilibrium_drift_residual_max": float(np.max(np.abs(equilibrium_residual))),
        "jacobian": a,
        "jacobian_eigenvalues": eigenvalues,
        "hurwitz": hurwitz,
        "p_matrix": p,
        "p_eigenvalues": p_eigenvalues,
        "p_positive_definite": p_positive,
        "lyapunov_equation_residual_2norm": residual_norm,
        "analysis_start_s": gate,
        "nonincrease_fraction_after_gate": nonincrease_fraction,
        "v_max": v_max,
        "v_gate": v_gate,
        "v_final": v_final,
        "v_final_to_gate_ratio": contraction_ratio,
        "contraction_credit": contraction_credit,
    }


def plot_band(
    filename: str, title: str, ylabel: str, time: np.ndarray,
    outer_low: np.ndarray, outer_high: np.ndarray,
    inner_low: np.ndarray, inner_high: np.ndarray,
    center: np.ndarray, scale: float = 1.0,
    reference: tuple[np.ndarray, str] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.fill_between(time, outer_low * scale, outer_high * scale,
                    color=COLORS["outer"], alpha=0.30, label="α=0 状态模糊包络")
    ax.fill_between(time, inner_low * scale, inner_high * scale,
                    color=COLORS["inner"], alpha=0.38, label="α=0.5 状态模糊包络")
    ax.plot(time, center * scale, color=COLORS["center"], linewidth=1.9,
            label="中心初始状态轨迹")
    if reference is not None:
        ax.plot(time, reference[0] * scale, color=COLORS["leader"], linestyle="--",
                linewidth=1.4, label=reference[1])
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel(ylabel)
    return finish(fig, ax, filename)


def plot_results(
    config: dict, center: dict, envelopes: dict[float, dict], family0: list[dict],
    lyapunov: dict[str, object]
) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    time = center["time"]
    outer, inner = envelopes[0.0], envelopes[0.5]
    figures = []
    state_plots = (
        ("01_state_x.png", "状态模糊下的纵向位置", "纵向位置 x（m）", 0, 1.0,
         (center["leader"][:, 0], "前车")),
        ("02_state_u.png", "状态模糊下的纵向速度", "纵向速度 u（km/h）", 1, 3.6,
         (center["leader"][:, 1], "前车")),
        ("03_state_z_lon.png", "纵向扰动状态传播", "z_lon（m/s²）", 2, 1.0, None),
        ("04_state_y.png", "状态模糊下的横向位置", "横向位置 y（m）", 3, 1.0, None),
        ("05_state_v.png", "状态模糊下的横向速度", "横向速度 v（m/s）", 4, 1.0, None),
        ("06_state_z_lat.png", "横向扰动状态传播", "z_lat（m/s²）", 5, 1.0, None),
    )
    for filename, title, ylabel, column, scale, reference in state_plots:
        figures.append(plot_band(
            filename, title, ylabel, time,
            outer["state_lower"][:, column], outer["state_upper"][:, column],
            inner["state_lower"][:, column], inner["state_upper"][:, column],
            center["state"][:, column], scale, reference,
        ))
    for filename, title, ylabel, column in (
        ("07_state_ax.png", "状态模糊下的纵向加速度", "纵向加速度 a_x（m/s²）", 0),
        ("08_state_ay.png", "状态模糊下的横向加速度", "横向加速度 a_y（m/s²）", 1),
    ):
        figures.append(plot_band(
            filename, title, ylabel, time,
            outer["acceleration_lower"][:, column], outer["acceleration_upper"][:, column],
            inner["acceleration_lower"][:, column], inner["acceleration_upper"][:, column],
            center["acceleration"][:, column],
        ))
    figures.append(plot_band(
        "09_state_headway.png", "状态模糊下的前车位置差", "前车位置差 h（m）", time,
        outer["headway_lower"], outer["headway_upper"],
        inner["headway_lower"], inner["headway_upper"], center["headway"],
    ))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x_min = max(float(run["state"][0, 0]) for run in family0)
    x_max = min(float(run["state"][-1, 0]) for run in family0)
    x_grid = np.linspace(x_min, x_max, 1000)
    y_curves = np.vstack([
        np.interp(x_grid, run["state"][:, 0], run["state"][:, 3]) for run in family0
    ])
    ax.fill_between(x_grid, np.min(y_curves, axis=0), np.max(y_curves, axis=0),
                    color=COLORS["outer"], alpha=0.34, label="α=0 二维轨迹包络")
    geometry = {name: item["value"] for name, item in config["geometry"].items()}
    for index, boundary in enumerate((geometry["boundary_left"], geometry["boundary_right"])):
        ax.axhline(boundary, color=COLORS["road"], linewidth=1.2,
                   label="确定道路边界" if index == 0 else None)
    for index, mark in enumerate(geometry["markings"]):
        ax.axhline(mark, color=COLORS["mark"], linestyle="--", linewidth=0.9,
                   label="确定车道标线" if index == 0 else None)
    ax.axhline(geometry["lane_center"], color=COLORS["neighbor"], linestyle=":",
               linewidth=1.2, label="确定目标车道中心线")
    ax.plot(center["state"][:, 0], center["state"][:, 3], color=COLORS["center"],
            linewidth=2.0, label="中心初始状态轨迹")
    ax.plot(center["neighbor"][:, 0], center["neighbor"][:, 2], color=COLORS["neighbor"],
            linewidth=1.3, linestyle="-.", label="邻车")
    ax.set_title("仅由初始状态模糊产生的二维轨迹包络", fontweight="bold", pad=12)
    ax.set_xlabel("纵向位置 x（m）")
    ax.set_ylabel("横向位置 y（m，向右为正）")
    figures.append(finish(fig, ax, "10_state_xy_envelope.png"))

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    widths = outer["state_upper"] - outer["state_lower"]
    for column, scale, label, color in (
        (1, 3.6, "W_u（km/h）", COLORS["center"]),
        (3, 1.0, "W_y（m）", COLORS["leader"]),
        (2, 1.0, "W_z_lon（m/s²）", COLORS["neighbor"]),
        (5, 1.0, "W_z_lat（m/s²）", COLORS["lyapunov"]),
    ):
        ax.plot(time, widths[:, column] * scale, linewidth=1.6, color=color, label=label)
    ax.set_title("α=0 状态模糊包络宽度", fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("包络宽度（单位见图例）")
    figures.append(finish(fig, ax, "11_state_widths.png"))

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.semilogy(time, np.maximum(np.asarray(lyapunov["v_max"]), 1e-16),
                color=COLORS["lyapunov"], linewidth=1.8,
                label="α=0 样本族最大 V(t)")
    gate = float(lyapunov["analysis_start_s"])
    ax.axvline(gate, color=COLORS["leader"], linestyle="--", linewidth=1.2,
               label="局部稳定性数值检查起点")
    ax.set_title("Lyapunov 函数沿状态模糊轨迹族的演化", fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("V=e^T P e（对数坐标）")
    figures.append(finish(fig, ax, "12_lyapunov_function.png"))
    return figures


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items() if key != "v_max"}
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> int:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(
        (REPO / "configs" / "two_dimensional_fuzzy_neighbor_ou.json").read_text(encoding="utf-8")
    )
    config = configure_neighbor_validation_scenario(base_config)
    config["neighbor"]["enabled"]["value"] = True
    config["idm"]["time_headway"]["value"] = 1.5
    config["lateral"]["zeta"]["value"] = 1.0
    params = stochastic_parameters_from_config(config).deterministic
    leader = make_leader(config, params)
    neighbor = make_neighbor(config)
    numbers = fuzzy_numbers()
    center_values = {name: numbers[name].mode for name in STATE_NAMES}
    alphas = tuple(float(alpha) for alpha in SPEC["alpha_levels"])
    count = int(SPEC["samples_per_alpha"])
    cache = {}

    def simulate(values: dict[str, float]) -> dict[str, np.ndarray]:
        key = sample_key(values)
        if key not in cache:
            cache[key] = simulate_case(config, values, leader, neighbor)
        return cache[key]

    center = simulate(center_values)
    standalone = {
        alpha: sample_alpha_domain(numbers, alpha, count) for alpha in alphas
    }
    families = {}
    envelopes = {}
    sample_counts = {}
    for alpha in alphas:
        union = {}
        for inner_alpha in alphas:
            if inner_alpha + 1e-15 < alpha:
                continue
            for values in standalone[inner_alpha]:
                union[sample_key(values)] = values
        families[alpha] = [simulate(values) for values in union.values()]
        envelopes[alpha] = envelope(families[alpha])
        sample_counts[str(alpha)] = len(families[alpha])

    alpha_one_error = float(np.max(np.abs(
        envelopes[1.0]["state_lower"] - center["state"]
    )))
    nesting = nesting_check(envelopes)
    lyapunov = lyapunov_analysis(config, center, families[0.0])

    convergence = {}
    counts = [int(value) for value in SPEC["convergence_sample_counts"]]
    conv_envs = {
        current: envelope([simulate(values) for values in sample_alpha_domain(numbers, 0.0, current)])
        for current in counts
    }
    reference = conv_envs[max(counts)]
    for current in counts:
        if current == max(counts):
            continue
        convergence[str(current)] = {
            "u_lower_mps": float(np.max(np.abs(
                conv_envs[current]["state_lower"][:, 1] - reference["state_lower"][:, 1]
            ))),
            "u_upper_mps": float(np.max(np.abs(
                conv_envs[current]["state_upper"][:, 1] - reference["state_upper"][:, 1]
            ))),
            "y_lower_m": float(np.max(np.abs(
                conv_envs[current]["state_lower"][:, 3] - reference["state_lower"][:, 3]
            ))),
            "y_upper_m": float(np.max(np.abs(
                conv_envs[current]["state_upper"][:, 3] - reference["state_upper"][:, 3]
            ))),
        }

    all_runs = list(cache.values())
    finite = all(np.all(np.isfinite(run[key])) for run in all_runs for key in ARRAY_KEYS)
    minimum_speed = min(float(np.min(run["state"][:, 1])) for run in all_runs)
    minimum_headway = min(float(np.min(run["headway"])) for run in all_runs)
    left = float(config["geometry"]["boundary_left"]["value"])
    right = float(config["geometry"]["boundary_right"]["value"])
    minimum_margin = min(float(min(
        np.min(run["state"][:, 3] - left), np.min(right - run["state"][:, 3])
    )) for run in all_runs)
    physical_validity = finite and minimum_speed >= 0.0 and minimum_headway > 0.0 and minimum_margin > 0.0

    score_parts = {
        "jacobian_hurwitz_30": 30.0 if lyapunov["hurwitz"] else 0.0,
        "positive_P_and_small_residual_20": 20.0 if (
            lyapunov["p_positive_definite"]
            and lyapunov["lyapunov_equation_residual_2norm"] < 1e-8
        ) else 0.0,
        "post_gate_nonincrease_20": 20.0 * float(lyapunov["nonincrease_fraction_after_gate"]),
        "post_gate_contraction_20": 20.0 * float(lyapunov["contraction_credit"]),
        "fuzzy_and_physical_validation_10": 10.0 if (
            alpha_one_error < 1e-10 and nesting["pass"] and physical_validity
        ) else 0.0,
    }
    score = float(sum(score_parts.values()))
    figures = plot_results(config, center, envelopes, families[0.0], lyapunov)

    outer_width = envelopes[0.0]["state_upper"] - envelopes[0.0]["state_lower"]
    summary = {
        "model_scope": "six fuzzy initial states only; crisp parameters and crisp road geometry; common OU path; neighbor interaction retained",
        "fuzzy_inputs": list(STATE_NAMES),
        "crisp_geometry": {
            "lane_center_m": config["geometry"]["lane_center"]["value"],
            "target_lane_markings_m": config["geometry"]["target_lane_markings"]["value"],
            "road_boundaries_m": [left, right],
        },
        "crisp_parameters": {"T_s": 1.5, "zeta_per_m": 1.0},
        "leader": {"initial_speed_mps": 15.0, "acceleration_mps2": 0.2,
                   "acceleration_interval_s": [10.0, 20.0], "final_speed_mps": 17.0},
        "neighbor": {"initial_speed_mps": config["neighbor_scenario"]["speed"]["value"],
                     "acceleration_mps2": 0.2, "acceleration_interval_s": [10.0, 20.0]},
        "alpha_levels": list(alphas),
        "nested_family_counts": sample_counts,
        "unique_simulations": len(cache),
        "alpha_one_max_state_error": alpha_one_error,
        "nesting": nesting,
        "sampling_convergence_against_128": convergence,
        "all_finite": bool(finite),
        "minimum_speed_mps": minimum_speed,
        "minimum_headway_m": minimum_headway,
        "minimum_road_margin_m": minimum_margin,
        "alpha0_initial_state_widths": outer_width[0].tolist(),
        "alpha0_max_state_widths": np.max(outer_width, axis=0).tolist(),
        "lyapunov": json_ready(lyapunov),
        "stability_score": {
            "score_out_of_100": score,
            "parts": score_parts,
            "interpretation": "Custom reproducible numerical diagnostic. It does not replace the local Lyapunov theorem or constitute a literature-standard safety score."
        },
        "figure_files": [str(path) for path in figures],
    }
    (OUTPUT / "state_fuzzy_lyapunov_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    success = bool(
        alpha_one_error < 1e-10 and nesting["pass"] and physical_validity
        and lyapunov["hurwitz"] and lyapunov["p_positive_definite"]
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
