"""参数模糊与道路几何模糊的两组独立二维微观交通实验。

parameter：只模糊 IDM 的 a_idm、b_idm、T 和横向 epsilon、zeta、gamma。
geometry：只模糊目标车道左/右标线位置与目标车道中心线位置。

两组实验的六维初始状态均为确定值；OU、前车、邻车和非目标输入均为确定值。
所有模糊样本共用同一 Wiener 增量，实现条件化于同一随机实现的公平比较。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = Path(r"D:\Desktop\fuzzy-microscopic-traffic-modeling")
sys.path.insert(0, str(REPO / "src"))

from two_dimensional_fuzzy_neighbor_ou.dynamics import (  # noqa: E402
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
    "center": "#174F78", "outer": "#9BC4DF", "inner": "#4D8EB9",
    "leader": "#C06442", "neighbor": "#2C8C7B", "road": "#46515B",
    "mark": "#B4943C", "lyapunov": "#7B4EA3",
}
PRIMES = (2, 3, 5, 7, 11, 13)
STATE0 = np.array([0.0, 15.0, 0.0, 4.5, 0.0, 0.0], dtype=float)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DT = 0.05
DURATION = 30.0
SEED = 20260922
HALTON_COUNT = 64
FD_STEP = 1e-6
PROBE_TIME = 8.0
PROBE_DT = 0.01
PROBE_E0 = np.array([0.20, 0.08, 0.01, 0.05, 0.02, 0.005], dtype=float)


EXPERIMENTS = {
    "parameter": {
        "title": "参数模糊化",
        "scope": "only a_idm, b_idm, T, epsilon, zeta and gamma are fuzzy",
        "names": ("a_idm", "b_idm", "T", "epsilon", "zeta", "gamma"),
        "triangles": {
            "a_idm": (0.8, 1.0, 1.2, "m/s²"),
            "b_idm": (1.2, 1.5, 1.8, "m/s²"),
            "T": (1.2, 1.5, 1.8, "s"),
            "epsilon": (0.08, 0.10, 0.12, "1"),
            "zeta": (0.8, 1.0, 1.2, "1/m"),
            "gamma": (0.448, 0.56, 0.672, "1/m"),
        },
        "prefix": "parameter",
    },
    "geometry": {
        "title": "环境几何模糊化",
        "scope": "only target left/right lane-mark and lane-center positions are fuzzy",
        "names": ("left_mark", "right_mark", "lane_center"),
        "triangles": {
            "left_mark": (1.70, 1.75, 1.80, "m"),
            "right_mark": (5.22, 5.25, 5.28, "m"),
            "lane_center": (3.45, 3.50, 3.55, "m"),
        },
        "prefix": "geometry",
    },
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 14.0, "axes.labelsize": 11.5,
        "legend.fontsize": 9.5, "xtick.labelsize": 10.0, "ytick.labelsize": 10.0,
        "axes.unicode_minus": False, "axes.linewidth": 0.9, "savefig.dpi": 300,
    })


def finish(fig: plt.Figure, ax: plt.Axes, output: Path, filename: str) -> Path:
    ax.grid(True, color="#CBD2D9", alpha=0.45, linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
                   ncol=min(4, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = output / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def radical_inverse(index: int, base: int) -> float:
    value, factor = 0.0, 1.0 / base
    while index > 0:
        value += factor * (index % base)
        index //= base
        factor /= base
    return value


def unit_design(dimension: int, count: int) -> np.ndarray:
    center = np.full(dimension, 0.5)
    points = [center.copy(), np.zeros(dimension), np.ones(dimension)]
    for axis in range(dimension):
        lower, upper = center.copy(), center.copy()
        lower[axis], upper[axis] = 0.0, 1.0
        points.extend((lower, upper))
    for index in range(1, count + 1):
        points.append(np.array([radical_inverse(index, PRIMES[j])
                                for j in range(dimension)], dtype=float))
    unique = {tuple(np.round(point, 14)): point for point in points}
    return np.vstack(list(unique.values()))


def fuzzy_numbers(spec: dict) -> dict[str, TriangularFuzzyNumber]:
    return {name: TriangularFuzzyNumber(*spec["triangles"][name][:3])
            for name in spec["names"]}


def sample_alpha_domain(spec: dict, alpha: float, count: int) -> list[dict[str, float]]:
    numbers = fuzzy_numbers(spec)
    cuts = [numbers[name].alpha_cut(alpha) for name in spec["names"]]
    samples = {}
    for point in unit_design(len(spec["names"]), count):
        values = {name: float(lo + p * (hi - lo))
                  for name, p, (lo, hi) in zip(spec["names"], point, cuts)}
        samples[sample_key(spec, values)] = values
    return list(samples.values())


def sample_key(spec: dict, values: dict[str, float]) -> tuple:
    return tuple((name, round(float(values[name]), 14)) for name in spec["names"])


def center_values(spec: dict) -> dict[str, float]:
    return {name: float(spec["triangles"][name][1]) for name in spec["names"]}


def apply_values(base: dict, kind: str, values: dict[str, float]) -> dict:
    cfg = deepcopy(base)
    if kind == "parameter":
        cfg["idm"]["a_idm"]["value"] = values["a_idm"]
        cfg["idm"]["b_idm"]["value"] = values["b_idm"]
        cfg["idm"]["time_headway"]["value"] = values["T"]
        for name in ("epsilon", "zeta", "gamma"):
            cfg["lateral"][name]["value"] = values[name]
    else:
        left, right, center = values["left_mark"], values["right_mark"], values["lane_center"]
        cfg["geometry"]["target_lane_markings"]["value"] = [left, right]
        marks = list(cfg["geometry"]["markings"]["value"])
        marks[2], marks[3] = left, right
        cfg["geometry"]["markings"]["value"] = marks
        cfg["geometry"]["lane_center"]["value"] = center
    return cfg


def make_leader(base_config: dict) -> Callable[[float], np.ndarray]:
    params = stochastic_parameters_from_config(base_config).deterministic
    speed0, acceleration, start, end, y = 15.0, 0.2, 10.0, 20.0, 3.5
    x0 = STATE0[0] + equilibrium_headway(speed0, params.idm)

    def trajectory(t: float) -> np.ndarray:
        active = min(max(t - start, 0.0), end - start)
        after = max(t - end, 0.0)
        return np.array([x0 + speed0*t + 0.5*acceleration*active**2
                         + acceleration*(end-start)*after,
                         speed0 + acceleration*active, y, 0.0], dtype=float)
    return trajectory


def make_neighbor(config: dict) -> Callable[[float], np.ndarray]:
    lateral = make_neighbor_trajectory(config)
    values = {k: v["value"] for k, v in config["neighbor_scenario"].items()}
    x0 = STATE0[0] + float(values["initial_longitudinal_gap"])
    speed0, acceleration, start, end = float(values["speed"]), 0.2, 10.0, 20.0

    def trajectory(t: float) -> np.ndarray:
        lat = lateral(t)
        active = min(max(t-start, 0.0), end-start)
        after = max(t-end, 0.0)
        return np.array([x0 + speed0*t + 0.5*acceleration*active**2
                         + acceleration*(end-start)*after,
                         speed0 + acceleration*active, lat[2], lat[3]], dtype=float)
    return trajectory


def simulate_case(config: dict, leader, neighbor) -> dict[str, np.ndarray]:
    params = stochastic_parameters_from_config(config)

    def drift(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_drift(t, state, leader(t), params, (neighbor(t),))

    def diffusion(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_diffusion(t, state, params)

    time, state = solve_sde_euler_maruyama(drift, diffusion, STATE0, DURATION, DT, SEED)
    leaders = np.array([leader(float(t)) for t in time])
    neighbors = np.array([neighbor(float(t)) for t in time])
    acceleration = np.array([drift(float(t), s)[[1, 4]] for t, s in zip(time, state)])
    compact = state[:, [0, 1, 3, 4]]
    neighbor_force = np.array([neighbor_interaction_force(s, n, params.deterministic.lateral,
                                                           params.deterministic.neighbor)
                               for s, n in zip(compact, neighbors)])
    active = np.array([is_neighbor_in_influence_region(s, n, params.deterministic.lateral,
                                                        params.deterministic.neighbor)
                       for s, n in zip(compact, neighbors)], dtype=bool)
    return {"time": time, "state": state, "leader": leaders, "neighbor": neighbors,
            "acceleration": acceleration, "neighbor_force": neighbor_force,
            "active": active, "headway": leaders[:, 0] - state[:, 0]}


ARRAY_KEYS = ("state", "acceleration", "neighbor_force", "headway")


def envelope(runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {"time": runs[0]["time"]}
    for key in ARRAY_KEYS:
        stack = np.stack([run[key] for run in runs])
        result[key + "_lower"] = np.min(stack, axis=0)
        result[key + "_upper"] = np.max(stack, axis=0)
    return result


def nesting_check(envelopes: dict[float, dict]) -> dict[str, object]:
    worst = 0.0
    levels = sorted(envelopes)
    for key in ARRAY_KEYS:
        for outer_alpha, inner_alpha in zip(levels[:-1], levels[1:]):
            outer, inner = envelopes[outer_alpha], envelopes[inner_alpha]
            worst = max(worst,
                        float(np.max(outer[key+"_lower"] - inner[key+"_lower"])),
                        float(np.max(inner[key+"_upper"] - outer[key+"_upper"])))
    return {"pass": bool(worst <= 1e-10), "maximum_violation": max(0.0, worst)}


def lateral_static_force(y: float, params) -> float:
    state = np.array([0.0, 17.0, y, 0.0])
    leader = np.array([100.0, 17.0, params.lateral.lane_center, 0.0])
    # Only the lateral acceleration is relevant; no neighbor and z_lat=0.
    full = np.array([state[0], state[1], 0.0, state[2], state[3], 0.0])
    stochastic = type("Holder", (), {"deterministic": params,
                                      "noise": type("Noise", (), {"theta_lon": 1.0,
                                                                   "mu_lon": 0.0,
                                                                   "theta_lat": 1.0,
                                                                   "mu_lat": 0.0})()})()
    return float(two_dimensional_stochastic_drift(0.0, full, leader, stochastic, ())[4])


def find_lateral_equilibrium(params) -> float:
    left, right = params.lateral.target_markings
    xs = np.linspace(left + 1e-5, right - 1e-5, 801)
    values = np.array([lateral_static_force(float(x), params) for x in xs])
    roots = []
    for k in range(len(xs)-1):
        if values[k] == 0.0:
            roots.append(float(xs[k]))
        elif values[k] * values[k+1] < 0.0:
            lo, hi = float(xs[k]), float(xs[k+1])
            flo = float(values[k])
            for _ in range(60):
                mid = 0.5*(lo+hi)
                fm = lateral_static_force(mid, params)
                if flo*fm <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            roots.append(0.5*(lo+hi))
    if not roots:
        return float(xs[int(np.argmin(np.abs(values)))])
    return min(roots, key=lambda value: abs(value - params.lateral.lane_center))


def numerical_jacobian(function: Callable[[np.ndarray], np.ndarray], n: int) -> np.ndarray:
    matrix = np.empty((n, n), dtype=float)
    for column in range(n):
        delta = np.zeros(n)
        delta[column] = FD_STEP
        matrix[:, column] = (function(delta) - function(-delta)) / (2.0*FD_STEP)
    return matrix


def solve_lyapunov_matrix(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    system = np.kron(np.eye(n), a.T) + np.kron(a.T, np.eye(n))
    vector = np.linalg.solve(system, -np.eye(n).reshape(-1, order="F"))
    p = vector.reshape((n, n), order="F")
    p = 0.5*(p+p.T)
    return p, a.T@p + p@a + np.eye(n)


def rk4_linear(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(0.0, PROBE_TIME + 0.5*PROBE_DT, PROBE_DT)
    state = np.empty((len(time), 6), dtype=float)
    state[0] = PROBE_E0
    for k in range(len(time)-1):
        current = state[k]
        k1 = a@current
        k2 = a@(current + 0.5*PROBE_DT*k1)
        k3 = a@(current + 0.5*PROBE_DT*k2)
        k4 = a@(current + PROBE_DT*k3)
        state[k+1] = current + PROBE_DT*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return time, state


def member_stability(config: dict) -> dict[str, object]:
    stochastic = stochastic_parameters_from_config(config)
    params, noise = stochastic.deterministic, stochastic.noise
    u_eq = 17.0
    h_eq = equilibrium_headway(u_eq, params.idm)
    y_eq = find_lateral_equilibrium(params)

    def error_drift(error: np.ndarray) -> np.ndarray:
        h, du, dz_lon, dy, dv, dz_lat = error
        leader = np.array([0.0, u_eq, y_eq, 0.0])
        vehicle = np.array([-(h_eq+h), u_eq+du, noise.mu_lon+dz_lon,
                            y_eq+dy, dv, noise.mu_lat+dz_lat])
        raw = two_dimensional_stochastic_drift(0.0, vehicle, leader, stochastic, ())
        return np.array([-du, raw[1], raw[2], raw[3], raw[4], raw[5]])

    residual0 = error_drift(np.zeros(6))
    a = numerical_jacobian(error_drift, 6)
    eigenvalues = np.linalg.eigvals(a)
    spectral = float(np.max(np.real(eigenvalues)))
    p, residual = solve_lyapunov_matrix(a)
    p_eigs = np.linalg.eigvalsh(p)
    time, errors = rk4_linear(a)
    values = np.einsum("ti,ij,tj->t", errors, p, errors)
    derivative = np.einsum("ti,ij,tj->t", errors, a.T@p+p@a, errors)
    diffusion = np.zeros((6, 2), dtype=float)
    diffusion[2, 0] = noise.sigma_lon
    diffusion[5, 1] = noise.sigma_lat
    generator_constant = float(np.trace(diffusion.T@p@diffusion))
    return {"h_eq": float(h_eq), "y_eq": float(y_eq), "equilibrium_residual": float(np.max(np.abs(residual0))),
            "A": a, "eigenvalues": eigenvalues, "spectral_abscissa": spectral,
            "hurwitz": bool(spectral < -1e-9), "P": p, "p_min": float(np.min(p_eigs)),
            "p_positive": bool(np.min(p_eigs) > 1e-10),
            "lyapunov_residual": float(np.linalg.norm(residual, ord=2)),
            "probe_time": time, "probe_V": values, "probe_dV": derivative,
            "nonincrease": bool(np.max(derivative) <= 1e-9),
            "final_ratio": float(values[-1]/values[0]),
            "stochastic_generator_constant": generator_constant,
            "individual_mean_square_ultimate_radius": float(np.sqrt(generator_constant))}


def robust_stability(configs: list[dict]) -> dict[str, object]:
    members = [member_stability(cfg) for cfg in configs]
    # unit_design 的首点为模糊数中心，故 members[0] 对应中心系统。
    # 用同一个 P_center 检查整个抽样族，比“每个成员各用一个 P”更强。
    common_p = members[0]["P"]
    common_derivative_max = [
        float(np.max(np.linalg.eigvalsh(item["A"].T@common_p + common_p@item["A"])))
        for item in members
    ]
    common_decay_min = [
        float(np.min(np.linalg.eigvalsh(-(item["A"].T@common_p + common_p@item["A"]))))
        for item in members
    ]
    # 所有实验的 OU 扩散矩阵相同；中心 P 下的 trace(G^T P G)。
    sigma_lon = float(configs[0]["ou_noise"]["sigma_lon"]["value"])
    sigma_lat = float(configs[0]["ou_noise"]["sigma_lat"]["value"])
    common_generator_constant = float(sigma_lon**2*common_p[2,2] + sigma_lat**2*common_p[5,5])
    worst_common_decay = min(common_decay_min)
    return {
        "members": members,
        "all_hurwitz": all(item["hurwitz"] for item in members),
        "worst_spectral_abscissa": max(item["spectral_abscissa"] for item in members),
        "all_p_positive": all(item["p_positive"] for item in members),
        "minimum_p_eigenvalue": min(item["p_min"] for item in members),
        "maximum_lyapunov_residual": max(item["lyapunov_residual"] for item in members),
        "all_probe_nonincreasing": all(item["nonincrease"] for item in members),
        "worst_probe_final_ratio": max(item["final_ratio"] for item in members),
        "common_center_P_certificate": bool(max(common_derivative_max) < -1e-9),
        "worst_lambda_max_of_ATP_plus_PA_using_center_P": max(common_derivative_max),
        "minimum_common_decay_eigenvalue": worst_common_decay,
        "maximum_individual_mean_square_ultimate_radius": max(
            item["individual_mean_square_ultimate_radius"] for item in members
        ),
        "common_generator_constant": common_generator_constant,
        "common_mean_square_ultimate_radius": (
            float(np.sqrt(common_generator_constant/worst_common_decay))
            if worst_common_decay > 0.0 else None
        ),
        "maximum_equilibrium_residual": max(item["equilibrium_residual"] for item in members),
        "equilibrium_headway_range": [min(item["h_eq"] for item in members),
                                      max(item["h_eq"] for item in members)],
        "lateral_equilibrium_range": [min(item["y_eq"] for item in members),
                                      max(item["y_eq"] for item in members)],
    }


def plot_band(output: Path, filename: str, title: str, ylabel: str, time: np.ndarray,
              outer_low, outer_high, inner_low, inner_high, center,
              label: str, scale: float = 1.0, reference=None) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.fill_between(time, outer_low*scale, outer_high*scale, color=COLORS["outer"], alpha=.30,
                    label=f"α=0 {label}包络")
    ax.fill_between(time, inner_low*scale, inner_high*scale, color=COLORS["inner"], alpha=.38,
                    label=f"α=0.5 {label}包络")
    ax.plot(time, center*scale, color=COLORS["center"], linewidth=1.9, label="中心值轨迹")
    if reference is not None:
        ax.plot(time, reference[0]*scale, color=COLORS["leader"], linestyle="--",
                linewidth=1.4, label=reference[1])
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel(ylabel)
    return finish(fig, ax, output, filename)


def plot_results(kind: str, spec: dict, output: Path, base_config: dict, center: dict,
                 envelopes: dict, family0: list[dict], stability: dict) -> list[Path]:
    time, outer, inner = center["time"], envelopes[0.0], envelopes[0.5]
    label, title_root = spec["title"], spec["title"]
    figures = []
    state_plots = (
        ("01_state_x.png", "纵向位置 x", "纵向位置 x（m）", 0, 1.0, (center["leader"][:,0], "前车")),
        ("02_state_u.png", "纵向速度 u", "纵向速度 u（km/h）", 1, 3.6, (center["leader"][:,1], "前车")),
        ("03_state_z_lon.png", "纵向扰动状态 z_lon", "z_lon（m/s²）", 2, 1.0, None),
        ("04_state_y.png", "横向位置 y", "横向位置 y（m）", 3, 1.0, None),
        ("05_state_v.png", "横向速度 v", "横向速度 v（m/s）", 4, 1.0, None),
        ("06_state_z_lat.png", "横向扰动状态 z_lat", "z_lat（m/s²）", 5, 1.0, None),
    )
    for filename, subtitle, ylabel, column, scale, reference in state_plots:
        figures.append(plot_band(output, filename, f"{title_root}：{subtitle}", ylabel, time,
            outer["state_lower"][:,column], outer["state_upper"][:,column],
            inner["state_lower"][:,column], inner["state_upper"][:,column],
            center["state"][:,column], label, scale, reference))
    for filename, subtitle, ylabel, column in (
        ("07_acceleration_x.png", "纵向加速度响应", "纵向加速度 a_x（m/s²）", 0),
        ("08_acceleration_y.png", "横向加速度响应", "横向加速度 a_y（m/s²）", 1),
    ):
        figures.append(plot_band(output, filename, f"{title_root}：{subtitle}", ylabel, time,
            outer["acceleration_lower"][:,column], outer["acceleration_upper"][:,column],
            inner["acceleration_lower"][:,column], inner["acceleration_upper"][:,column],
            center["acceleration"][:,column], label))
    figures.append(plot_band(output, "09_headway.png", f"{title_root}：前车位置差",
        "前车位置差 h（m）", time, outer["headway_lower"], outer["headway_upper"],
        inner["headway_lower"], inner["headway_upper"], center["headway"], label))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x_min = max(float(run["state"][0,0]) for run in family0)
    x_max = min(float(run["state"][-1,0]) for run in family0)
    x_grid = np.linspace(x_min, x_max, 1000)
    y_curves = np.vstack([np.interp(x_grid, run["state"][:,0], run["state"][:,3])
                          for run in family0])
    ax.fill_between(x_grid, np.min(y_curves, axis=0), np.max(y_curves, axis=0),
                    color=COLORS["outer"], alpha=.34, label=f"α=0 {label}二维轨迹包络")
    geometry = {k:v["value"] for k,v in base_config["geometry"].items()}
    for k, boundary in enumerate((geometry["boundary_left"], geometry["boundary_right"])):
        ax.axhline(boundary, color=COLORS["road"], linewidth=1.2,
                   label="确定道路边界" if k == 0 else None)
    if kind == "geometry":
        for name, color, text in (("left_mark", COLORS["mark"], "目标车道标线 α=0 范围"),
                                  ("right_mark", COLORS["mark"], None),
                                  ("lane_center", COLORS["neighbor"], "目标中心线 α=0 范围")):
            lo, _, hi, _ = spec["triangles"][name]
            ax.axhspan(lo, hi, color=color, alpha=.14, label=text)
        for k, mark in enumerate(geometry["markings"]):
            ax.axhline(mark, color=COLORS["mark"], linestyle="--", linewidth=.9,
                       label="中心几何车道标线" if k == 0 else None)
    else:
        for k, mark in enumerate(geometry["markings"]):
            ax.axhline(mark, color=COLORS["mark"], linestyle="--", linewidth=.9,
                       label="确定车道标线" if k == 0 else None)
    ax.axhline(geometry["lane_center"], color=COLORS["neighbor"], linestyle=":", linewidth=1.2,
               label="中心几何目标车道中心线")
    ax.plot(center["state"][:,0], center["state"][:,3], color=COLORS["center"], linewidth=2,
            label="中心值轨迹")
    ax.plot(center["neighbor"][:,0], center["neighbor"][:,2], color=COLORS["neighbor"],
            linewidth=1.3, linestyle="-.", label="邻车")
    ax.set_title(f"{title_root}：二维轨迹包络", fontweight="bold", pad=12)
    ax.set_xlabel("纵向位置 x（m）")
    ax.set_ylabel("横向位置 y（m，向右为正）")
    figures.append(finish(fig, ax, output, "10_xy_envelope.png"))

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    widths = outer["state_upper"] - outer["state_lower"]
    for column, scale, legend, color in ((1,3.6,"W_u（km/h）",COLORS["center"]),
                                         (3,1,"W_y（m）",COLORS["leader"]),
                                         (2,1,"W_z_lon（m/s²）",COLORS["neighbor"]),
                                         (5,1,"W_z_lat（m/s²）",COLORS["lyapunov"])):
        ax.plot(time, widths[:,column]*scale, linewidth=1.6, color=color, label=legend)
    ax.set_title(f"α=0 {title_root}状态响应包络宽度", fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("包络宽度（单位见图例）")
    figures.append(finish(fig, ax, output, "11_envelope_widths.png"))

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    members = stability["members"]
    v_stack = np.vstack([item["probe_V"]/item["probe_V"][0] for item in members])
    probe_t = members[0]["probe_time"]
    ax.fill_between(probe_t, np.min(v_stack, axis=0), np.max(v_stack, axis=0),
                    color=COLORS["outer"], alpha=.35, label="α=0 系统族 V(t)/V(0) 范围")
    center_index = int(np.argmin([abs(item["spectral_abscissa"] - np.median(
        [other["spectral_abscissa"] for other in members])) for item in members]))
    ax.plot(probe_t, v_stack[center_index], color=COLORS["lyapunov"], linewidth=1.8,
            label="代表成员")
    ax.axhline(1.0, color=COLORS["road"], linestyle="--", linewidth=1.0, label="初始值")
    ax.set_yscale("log")
    ax.set_title(f"{title_root}：局部 Lyapunov 探针", fontweight="bold", pad=12)
    ax.set_xlabel("局部误差系统时间 τ（s）")
    ax.set_ylabel("归一化 Lyapunov 函数 V(τ)/V(0)")
    figures.append(finish(fig, ax, output, "12_lyapunov_probe.png"))
    return figures


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k,v in value.items() if k not in {"members"}}
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_config(kind: str, spec: dict, output: Path) -> Path:
    payload = {
        "experiment": kind,
        "strict_scope": spec["scope"],
        "triangular_fuzzy_inputs": {
            name: {"left": values[0], "mode": values[1], "right": values[2],
                   "unit": values[3], "source": "assumed_for_simulation"}
            for name, values in spec["triangles"].items()
        },
        "crisp_initial_state": {"order": ["x","u","z_lon","y","v","z_lat"],
                                "value": STATE0.tolist()},
        "common_crisp_inputs": {
            "leader": "15 to 17 m/s, a=0.2 m/s^2 during 10-20 s",
            "neighbor": "existing prescribed adjacent-lane trajectory; 14.5 to 16.5 m/s during 10-20 s",
            "ou": "theta_lon=theta_lat=1; sigma_lon=0.1; sigma_lat=0.05; common seed",
            "neighbor_force": "Qi (2025) Appendix A Eq. (29)-(32)",
        },
        "numerics": {"dt_s": DT, "duration_s": DURATION, "seed": SEED,
                     "alpha_levels": list(ALPHAS), "halton_count_per_alpha": HALTON_COUNT},
        "stability": {"method": "memberwise local linearization and continuous Lyapunov equation",
                      "Q": "I_6", "probe_error": PROBE_E0.tolist(),
                      "warning": "custom score is a reproducible diagnostic, not a standard literature score"},
    }
    path = output.parent / f"{kind}_fuzzy_only.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(kind: str) -> int:
    configure_style()
    spec = EXPERIMENTS[kind]
    output = HERE / "results" / kind
    output.mkdir(parents=True, exist_ok=True)
    raw = json.loads((REPO / "configs" / "two_dimensional_fuzzy_neighbor_ou.json").read_text(encoding="utf-8"))
    base = configure_neighbor_validation_scenario(raw)
    base["neighbor"]["enabled"]["value"] = True
    base["idm"]["time_headway"]["value"] = 1.5
    base["lateral"]["zeta"]["value"] = 1.0
    leader, neighbor = make_leader(base), make_neighbor(base)
    center_input = center_values(spec)
    cache = {}

    def run(values: dict[str,float]) -> dict[str,np.ndarray]:
        key = sample_key(spec, values)
        if key not in cache:
            cache[key] = simulate_case(apply_values(base, kind, values), leader, neighbor)
        return cache[key]

    center = run(center_input)
    standalone = {alpha: sample_alpha_domain(spec, alpha, HALTON_COUNT) for alpha in ALPHAS}
    families, envelopes, counts = {}, {}, {}
    for alpha in ALPHAS:
        union = {}
        for inner in ALPHAS:
            if inner + 1e-15 < alpha:
                continue
            for values in standalone[inner]:
                union[sample_key(spec, values)] = values
        families[alpha] = [run(values) for values in union.values()]
        envelopes[alpha] = envelope(families[alpha])
        counts[str(alpha)] = len(families[alpha])

    alpha_one_error = float(np.max(np.abs(envelopes[1.0]["state_lower"] - center["state"])))
    nesting = nesting_check(envelopes)
    stability_values = standalone[0.0]
    stability_configs = [apply_values(base, kind, values) for values in stability_values]
    stability = robust_stability(stability_configs)

    all_runs = list(cache.values())
    finite = all(np.all(np.isfinite(run_[key])) for run_ in all_runs for key in ARRAY_KEYS)
    min_speed = min(float(np.min(run_["state"][:,1])) for run_ in all_runs)
    min_headway = min(float(np.min(run_["headway"])) for run_ in all_runs)
    left, right = base["geometry"]["boundary_left"]["value"], base["geometry"]["boundary_right"]["value"]
    min_margin = min(float(min(np.min(run_["state"][:,3]-left), np.min(right-run_["state"][:,3])))
                     for run_ in all_runs)
    physical = finite and min_speed >= 0 and min_headway > 0 and min_margin > 0
    score_parts = {
        "all_members_hurwitz_30": 30.0 if stability["all_hurwitz"] else 0.0,
        "all_P_positive_and_residual_small_20": 20.0 if stability["all_p_positive"] and stability["maximum_lyapunov_residual"] < 1e-8 else 0.0,
        "common_P_and_all_local_probes_nonincreasing_20": 20.0 if (
            stability["common_center_P_certificate"] and stability["all_probe_nonincreasing"]
        ) else 0.0,
        "worst_probe_contraction_20": 20.0*float(np.clip(1.0-np.sqrt(max(stability["worst_probe_final_ratio"],0)),0,1)),
        "fuzzy_and_physical_validation_10": 10.0 if alpha_one_error < 1e-10 and nesting["pass"] and physical else 0.0,
    }
    score = float(sum(score_parts.values()))
    figures = plot_results(kind, spec, output, base, center, envelopes, families[0.0], stability)
    config_path = write_config(kind, spec, output)
    widths = envelopes[0.0]["state_upper"] - envelopes[0.0]["state_lower"]
    summary = {
        "experiment": kind, "title": spec["title"], "strict_scope": spec["scope"],
        "fuzzy_inputs": list(spec["names"]), "fuzzy_input_source": "assumed_for_simulation",
        "crisp_initial_state": STATE0.tolist(), "common_seed": SEED,
        "alpha_levels": list(ALPHAS), "nested_family_counts": counts,
        "unique_simulations": len(cache), "alpha_one_max_state_error": alpha_one_error,
        "nesting": nesting, "all_finite": finite, "minimum_speed_mps": min_speed,
        "minimum_headway_m": min_headway, "minimum_road_margin_m": min_margin,
        "alpha0_max_state_widths": np.max(widths, axis=0).tolist(),
        "stability": json_ready(stability),
        "stability_score": {"score_out_of_100": score, "parts": score_parts,
            "interpretation": "Custom reproducible diagnostic; not a literature-standard stability or safety score."},
        "config_file": str(config_path), "figure_files": [str(path) for path in figures],
    }
    (output / f"{kind}_fuzzy_stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "alpha0_stability_members.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([*spec["names"], "h_eq_m", "y_eq_m", "spectral_abscissa",
                         "lambda_min_P", "lyapunov_residual", "probe_final_ratio"])
        for values, item in zip(stability_values, stability["members"]):
            writer.writerow([*[values[name] for name in spec["names"]], item["h_eq"], item["y_eq"],
                             item["spectral_abscissa"], item["p_min"], item["lyapunov_residual"],
                             item["final_ratio"]])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = alpha_one_error < 1e-10 and nesting["pass"] and physical and stability["all_hurwitz"] and stability["all_p_positive"]
    return 0 if ok else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.experiment))
