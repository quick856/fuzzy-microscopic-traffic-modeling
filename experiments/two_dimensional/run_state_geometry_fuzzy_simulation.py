"""六维状态、车道标线与车道中心线模糊化的二维模糊—随机仿真。

车辆 i 为被求解的目标跟驰车辆，le(i) 为分段匀加速前车，j 为中间车道邻车。
所有模糊样本共用同一组 Wiener 增量，因此包络表示固定随机实现条件下的
状态与感知几何模糊传播，而不是概率置信区间。
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


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "figures" / "state_geometry_fuzzy"
sys.path.insert(0, str(ROOT / "src"))

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
    "green": "#2F8F78",
    "purple": "#8064A2",
}

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "font.size": 10.0,
        "axes.titlesize": 13.0,
        "axes.labelsize": 11.0,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.85,
        "savefig.dpi": 300,
        "figure.dpi": 130,
    })


def decorate(ax: plt.Axes) -> None:
    ax.grid(True, color="#CBD2D9", alpha=0.45, linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def finish(fig: plt.Figure, ax: plt.Axes, filename: str) -> Path:
    decorate(ax)
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
    """固定的低差异设计，并显式加入中心、轴端点和两条对角端点。"""
    if dimension > len(PRIMES) or halton_count < 1:
        raise ValueError("采样维数或样本数非法")
    center = np.full(dimension, 0.5)
    points: list[np.ndarray] = [center.copy(), np.zeros(dimension), np.ones(dimension)]
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
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for point in points:
        unique[tuple(np.round(point, 14))] = point
    return np.vstack(list(unique.values()))


def fuzzy_numbers(spec: dict) -> tuple[tuple[str, ...], dict[str, TriangularFuzzyNumber]]:
    names = tuple(spec["triangles"].keys())
    numbers = {
        name: TriangularFuzzyNumber(
            float(item["left"]), float(item["mode"]), float(item["right"])
        )
        for name, item in spec["triangles"].items()
    }
    return names, numbers


def sample_alpha_domain(
    names: tuple[str, ...],
    numbers: dict[str, TriangularFuzzyNumber],
    alpha: float,
    halton_count: int,
) -> list[dict[str, float]]:
    points = unit_design(len(names), halton_count)
    cuts = [numbers[name].alpha_cut(alpha) for name in names]
    samples: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}
    for point in points:
        values = {
            name: float(lower + coordinate * (upper - lower))
            for name, coordinate, (lower, upper) in zip(names, point, cuts)
        }
        key = tuple((name, round(values[name], 14)) for name in names)
        samples[key] = values
    return list(samples.values())


def sample_key(names: tuple[str, ...], values: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple((name, round(float(values[name]), 14)) for name in names)


def apply_geometry(config: dict, values: dict[str, float]) -> None:
    left = float(values["lane_mark_left"])
    right = float(values["lane_mark_right"])
    center = float(values["lane_center"])
    if not left < center < right:
        raise ValueError("模糊几何样本必须满足左标线 < 中心线 < 右标线")
    marks = list(config["geometry"]["markings"]["value"])
    marks[-2:] = [left, right]
    config["geometry"]["markings"]["value"] = marks
    config["geometry"]["target_lane_markings"]["value"] = [left, right]
    config["geometry"]["lane_center"]["value"] = center


def make_uniform_leader(
    base_config: dict, spec: dict, nominal_params
) -> Callable[[float], np.ndarray]:
    leader_cfg = spec["uniformly_accelerating_leader"]
    speed0 = float(leader_cfg["initial_speed"]["value"])
    acceleration = float(leader_cfg["acceleration"]["value"])
    start = float(leader_cfg["acceleration_start"]["value"])
    end = float(leader_cfg["acceleration_end"]["value"])
    y = float(leader_cfg["lateral_position"]["value"])
    x_mode = float(spec["triangles"]["x"]["mode"])
    x0 = x_mode + equilibrium_headway(speed0, nominal_params.idm)
    if not 0.0 <= start < end:
        raise ValueError("前车加速区间必须满足 0 <= start < end")

    def leader(t: float) -> np.ndarray:
        if not np.isfinite(t) or t < 0:
            raise ValueError("前车时间必须为有限非负数")
        active_time = min(max(t - start, 0.0), end - start)
        after_time = max(t - end, 0.0)
        return np.array([
            x0 + speed0 * t + 0.5 * acceleration * active_time**2
            + acceleration * (end - start) * after_time,
            speed0 + acceleration * active_time,
            y,
            0.0,
        ], dtype=float)

    return leader


def make_accelerating_neighbor(
    nominal_config: dict, spec: dict
) -> Callable[[float], np.ndarray]:
    """保留既有横向接近轨迹，为外生邻车补充连续匀加速纵向轨迹。"""
    lateral_trajectory = make_neighbor_trajectory(nominal_config)
    scenario = {
        name: item["value"] for name, item in nominal_config["neighbor_scenario"].items()
    }
    x0 = float(nominal_config["scenario"]["initial_x"]["value"]) + float(
        scenario["initial_longitudinal_gap"]
    )
    speed0 = float(scenario["speed"])
    acceleration = float(spec["deterministic_neighbor"]["longitudinal_acceleration"]["value"])
    leader_cfg = spec["uniformly_accelerating_leader"]
    start = float(leader_cfg["acceleration_start"]["value"])
    end = float(leader_cfg["acceleration_end"]["value"])

    def neighbor(t: float) -> np.ndarray:
        lateral = lateral_trajectory(t)
        active_time = min(max(t - start, 0.0), end - start)
        after_time = max(t - end, 0.0)
        return np.array([
            x0 + speed0 * t + 0.5 * acceleration * active_time**2
            + acceleration * (end - start) * after_time,
            speed0 + acceleration * active_time,
            lateral[2],
            lateral[3],
        ], dtype=float)

    return neighbor


def simulate_case(
    nominal_config: dict,
    spec: dict,
    values: dict[str, float],
    leader: Callable[[float], np.ndarray],
    neighbor_j: Callable[[float], np.ndarray],
) -> dict[str, np.ndarray]:
    config = deepcopy(nominal_config)
    apply_geometry(config, values)
    params = stochastic_parameters_from_config(config)
    initial = np.array([
        values["x"], values["u"], values["z_lon"],
        values["y"], values["v"], values["z_lat"],
    ], dtype=float)
    duration = float(spec["numerics"]["duration"]["value"])
    dt = float(spec["numerics"]["dt"]["value"])
    seed = int(spec["numerics"]["random_seed"]["value"])

    def drift(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_drift(
            t, state, leader(t), params, (neighbor_j(t),)
        )

    def diffusion(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_diffusion(t, state, params)

    time, state_six = solve_sde_euler_maruyama(
        drift, diffusion, initial, duration, dt, seed
    )
    state = state_six[:, [0, 1, 3, 4]]
    leaders = np.array([leader(float(t)) for t in time])
    neighbors = np.array([neighbor_j(float(t)) for t in time])
    base = params.deterministic
    acceleration = np.array([
        drift(float(t), current)[[1, 4]] for t, current in zip(time, state_six)
    ])
    lateral_components = np.array([
        lateral_acceleration_components(current[2], current[3], base)
        for current in state
    ])
    neighbor_force = np.array([
        neighbor_interaction_force(current, other, base.lateral, base.neighbor)
        for current, other in zip(state, neighbors)
    ])
    active = np.array([
        is_neighbor_in_influence_region(current, other, base.lateral, base.neighbor)
        for current, other in zip(state, neighbors)
    ], dtype=bool)
    left_mark, right_mark = base.lateral.target_markings
    deltas = np.column_stack([
        state[:, 2] - base.lateral.lane_center,
        state[:, 2] - left_mark,
        state[:, 2] - right_mark,
        state[:, 2] - base.lateral.boundary_left,
        base.lateral.boundary_right - state[:, 2],
        neighbors[:, 2] - state[:, 2],
    ])
    return {
        "time": time,
        "state_six": state_six,
        "state": state,
        "leader": leaders,
        "neighbor": neighbors,
        "acceleration": acceleration,
        "lateral_components": lateral_components,
        "neighbor_force": neighbor_force,
        "active": active,
        "headway": leaders[:, 0] - state[:, 0],
        "delta_y": deltas,
        "geometry": np.array([left_mark, right_mark, base.lateral.lane_center]),
    }


ARRAY_KEYS = (
    "state_six", "state", "acceleration", "lateral_components",
    "neighbor_force", "headway", "delta_y",
)


def envelope(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("包络至少需要一条轨迹")
    result: dict[str, np.ndarray] = {"time": samples[0]["time"]}
    for key in ARRAY_KEYS:
        stacked = np.stack([sample[key] for sample in samples])
        result[f"{key}_lower"] = np.min(stacked, axis=0)
        result[f"{key}_upper"] = np.max(stacked, axis=0)
    geometry = np.stack([sample["geometry"] for sample in samples])
    result["geometry_lower"] = np.min(geometry, axis=0)
    result["geometry_upper"] = np.max(geometry, axis=0)
    result["active_any"] = np.any(np.stack([sample["active"] for sample in samples]), axis=0)
    result["active_all"] = np.all(np.stack([sample["active"] for sample in samples]), axis=0)
    return result


def nesting_check(envelopes: dict[float, dict[str, np.ndarray]]) -> dict[str, object]:
    worst = 0.0
    details: dict[str, float] = {}
    alphas = sorted(envelopes)
    for key in ARRAY_KEYS:
        current = 0.0
        for outer_alpha, inner_alpha in zip(alphas[:-1], alphas[1:]):
            outer = envelopes[outer_alpha]
            inner = envelopes[inner_alpha]
            current = max(
                current,
                float(np.max(outer[f"{key}_lower"] - inner[f"{key}_lower"])),
                float(np.max(inner[f"{key}_upper"] - outer[f"{key}_upper"])),
            )
        details[key] = max(current, 0.0)
        worst = max(worst, current)
    return {"pass": bool(worst <= 1e-10), "maximum_violation": max(worst, 0.0), "details": details}


def spatial_envelope(samples: list[dict[str, np.ndarray]], points: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for sample in samples:
        if np.any(np.diff(sample["state"][:, 0]) <= 0):
            raise ValueError("二维空间包络要求所有 x(t) 严格递增")
    start = max(float(sample["state"][0, 0]) for sample in samples)
    end = min(float(sample["state"][-1, 0]) for sample in samples)
    x_grid = np.linspace(start, end, points)
    curves = np.vstack([
        np.interp(x_grid, sample["state"][:, 0], sample["state"][:, 2])
        for sample in samples
    ])
    return x_grid, np.min(curves, axis=0), np.max(curves, axis=0)


def band_plot(
    filename: str,
    title: str,
    ylabel: str,
    time: np.ndarray,
    outer_lower: np.ndarray,
    outer_upper: np.ndarray,
    inner_lower: np.ndarray,
    inner_upper: np.ndarray,
    center: np.ndarray,
    scale: float = 1.0,
    reference: tuple[np.ndarray, str] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    ax.fill_between(time, outer_lower * scale, outer_upper * scale,
                    color=COLORS["outer"], alpha=0.30, label="α=0 模糊包络")
    ax.fill_between(time, inner_lower * scale, inner_upper * scale,
                    color=COLORS["inner"], alpha=0.35, label="α=0.5 模糊包络")
    ax.plot(time, center * scale, color=COLORS["center"], linewidth=1.8,
            label="中心状态—中心几何轨迹")
    if reference is not None:
        ax.plot(time, reference[0] * scale, color=COLORS["leader"], linestyle="--",
                linewidth=1.3, label=reference[1])
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel(ylabel)
    return finish(fig, ax, filename)


def plot_results(
    nominal_config: dict,
    spec: dict,
    center: dict[str, np.ndarray],
    envelopes: dict[float, dict[str, np.ndarray]],
    families: dict[float, list[dict[str, np.ndarray]]],
) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    time = center["time"]
    outer = envelopes[0.0]
    inner = envelopes[0.5]
    figures: list[Path] = []
    state_specs = (
        ("01_fuzzy_x.png", "模糊纵向位置", "纵向位置 x（m）", 0, 1.0, (center["leader"][:, 0], "前车")),
        ("02_fuzzy_u.png", "模糊纵向速度", "纵向速度 u（km/h）", 1, 3.6, (center["leader"][:, 1], "前车")),
        ("03_fuzzy_z_lon.png", "模糊纵向随机扰动状态", "纵向扰动 z_lon（m/s²）", 2, 1.0, None),
        ("04_fuzzy_y.png", "模糊横向位置", "横向位置 y（m）", 3, 1.0, None),
        ("05_fuzzy_v.png", "模糊横向速度", "横向速度 v（m/s）", 4, 1.0, None),
        ("06_fuzzy_z_lat.png", "模糊横向随机扰动状态", "横向扰动 z_lat（m/s²）", 5, 1.0, None),
    )
    for filename, title, ylabel, column, scale, reference in state_specs:
        figures.append(band_plot(
            filename, title, ylabel, time,
            outer["state_six_lower"][:, column], outer["state_six_upper"][:, column],
            inner["state_six_lower"][:, column], inner["state_six_upper"][:, column],
            center["state_six"][:, column], scale, reference,
        ))
    for filename, title, ylabel, column in (
        ("07_fuzzy_ax.png", "模糊纵向加速度响应", "纵向加速度 ax（m/s²）", 0),
        ("08_fuzzy_ay.png", "模糊横向加速度响应", "横向加速度 ay（m/s²）", 1),
    ):
        figures.append(band_plot(
            filename, title, ylabel, time,
            outer["acceleration_lower"][:, column], outer["acceleration_upper"][:, column],
            inner["acceleration_lower"][:, column], inner["acceleration_upper"][:, column],
            center["acceleration"][:, column],
        ))
    figures.append(band_plot(
        "09_fuzzy_headway.png", "模糊前车位置差", "前车位置差 h（m）", time,
        outer["headway_lower"], outer["headway_upper"],
        inner["headway_lower"], inner["headway_upper"], center["headway"],
    ))

    x0, y0_low, y0_high = spatial_envelope(families[0.0])
    x5, y5_low, y5_high = spatial_envelope(families[0.5])
    geometry = {name: item["value"] for name, item in nominal_config["geometry"].items()}
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.fill_between(x0, y0_low, y0_high, color=COLORS["outer"], alpha=0.30,
                    label="α=0 二维轨迹包络")
    ax.fill_between(x5, y5_low, y5_high, color=COLORS["inner"], alpha=0.35,
                    label="α=0.5 二维轨迹包络")

    def geometry_alpha_cut(name: str, alpha: float) -> tuple[float, float]:
        """返回感知道路几何三角模糊数在给定 α 水平下的截集。"""
        fuzzy_spec = spec["triangles"][name]
        return TriangularFuzzyNumber(**{
            key: float(fuzzy_spec[key]) for key in ("left", "mode", "right")
        }).alpha_cut(alpha)

    # 物理车道标线位置保持确定；色带表示车辆对两条目标车道标线的感知区间。
    for index, mark_name in enumerate(("lane_mark_left", "lane_mark_right")):
        mark_low_0, mark_high_0 = geometry_alpha_cut(mark_name, 0.0)
        mark_low_5, mark_high_5 = geometry_alpha_cut(mark_name, 0.5)
        ax.axhspan(mark_low_0, mark_high_0, color=COLORS["mark"], alpha=0.10,
                   label="α=0 感知车道标线范围" if index == 0 else None)
        ax.axhspan(mark_low_5, mark_high_5, color=COLORS["mark"], alpha=0.22,
                   label="α=0.5 感知车道标线范围" if index == 0 else None)

    center_low_0, center_high_0 = geometry_alpha_cut("lane_center", 0.0)
    center_low_5, center_high_5 = geometry_alpha_cut("lane_center", 0.5)
    ax.axhspan(center_low_0, center_high_0, color=COLORS["green"], alpha=0.10,
               label="α=0 感知中心线范围")
    ax.axhspan(center_low_5, center_high_5, color=COLORS["green"], alpha=0.22,
               label="α=0.5 感知中心线范围")

    for index, boundary in enumerate((geometry["boundary_left"], geometry["boundary_right"])):
        ax.axhline(boundary, color=COLORS["road"], linewidth=1.2,
                   label="物理道路边界" if index == 0 else None)
    for index, mark in enumerate(geometry["markings"]):
        ax.axhline(mark, color=COLORS["mark"], linestyle="--", linewidth=0.9,
                   label="物理车道标线" if index == 0 else None)
    ax.plot(center["state"][:, 0], center["state"][:, 2], color=COLORS["center"],
            linewidth=2.0, label="确定性轨迹")
    ax.plot(center["leader"][:, 0], center["leader"][:, 2], color=COLORS["leader"],
            linestyle="--", linewidth=1.3, label="前车")
    ax.plot(center["neighbor"][:, 0], center["neighbor"][:, 2], color=COLORS["neighbor"],
            linewidth=1.3, label="中间车道邻车")
    ax.set_title("状态与环境模糊化的二维轨迹包络", fontweight="bold", pad=12)
    ax.set_xlabel("纵向位置 x（m）")
    ax.set_ylabel("横向位置 y（m）")
    figures.append(finish(fig, ax, "10_fuzzy_xy_envelope.png"))

    figures.append(band_plot(
        "11_delta_y_middle.png", "车道中心线有向距离的模糊传播",
        "中心线有向距离 Δy_ml（m）", time,
        outer["delta_y_lower"][:, 0], outer["delta_y_upper"][:, 0],
        inner["delta_y_lower"][:, 0], inner["delta_y_upper"][:, 0],
        center["delta_y"][:, 0],
    ))

    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    for column, color, label in ((1, COLORS["green"], "左侧标线距离 Δy_L"),
                                 (2, COLORS["purple"], "右侧标线距离 Δy_R")):
        ax.fill_between(time, outer["delta_y_lower"][:, column],
                        outer["delta_y_upper"][:, column], color=color, alpha=0.16)
        ax.plot(time, center["delta_y"][:, column], color=color, linewidth=1.6, label=label)
    ax.axhline(0.0, color="#7A858E", linewidth=0.8)
    ax.set_title("车辆到两条感知车道标线的有向距离", fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("标线有向距离 Δy_mk（m）")
    figures.append(finish(fig, ax, "12_delta_y_markings.png"))

    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    labels = ("道路边界作用", "两侧车道标线合力", "车道中心线作用")
    colors = (COLORS["road"], COLORS["mark"], COLORS["green"])
    for column, (label, color) in enumerate(zip(labels, colors)):
        ax.fill_between(time, outer["lateral_components_lower"][:, column],
                        outer["lateral_components_upper"][:, column], color=color, alpha=0.12)
        ax.plot(time, center["lateral_components"][:, column], color=color,
                linewidth=1.5, label=label)
    ax.axhline(0.0, color="#7A858E", linewidth=0.8)
    ax.set_title("横向道路作用分项及其 α=0 包络", fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("横向加速度贡献（m/s²）")
    figures.append(finish(fig, ax, "13_lateral_force_components.png"))

    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    for column, color, label in ((0, COLORS["leader"], "邻车纵向作用"),
                                 (1, COLORS["neighbor"], "邻车横向作用")):
        ax.fill_between(time, outer["neighbor_force_lower"][:, column],
                        outer["neighbor_force_upper"][:, column], color=color, alpha=0.18)
        ax.plot(time, center["neighbor_force"][:, column], color=color,
                linewidth=1.6, label=label)
    ax.axhline(0.0, color="#7A858E", linewidth=0.8)
    ax.set_title("邻车二维作用及其 α=0 包络", fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("加速度贡献（m/s²）")
    figures.append(finish(fig, ax, "14_neighbor_force_envelopes.png"))

    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    widths = outer["state_six_upper"] - outer["state_six_lower"]
    for column, scale, color, label in (
        (1, 3.6, COLORS["center"], "W_u（km/h）"),
        (3, 1.0, COLORS["leader"], "W_y（m）"),
        (2, 1.0, COLORS["green"], "W_z_lon（m/s²）"),
        (5, 1.0, COLORS["purple"], "W_z_lat（m/s²）"),
    ):
        ax.plot(time, widths[:, column] * scale, color=color, linewidth=1.5, label=label)
    ax.set_title("α=0 状态包络宽度", fontweight="bold", pad=12)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("包络宽度（各曲线单位见图例）")
    figures.append(finish(fig, ax, "15_state_widths.png"))
    return figures


def main() -> int:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(
        (ROOT / "configs" / "two_dimensional_fuzzy_neighbor_ou.json").read_text(encoding="utf-8")
    )
    spec = json.loads(
        (ROOT / "configs" / "state_geometry_fuzzy.json").read_text(encoding="utf-8")
    )
    nominal_config = configure_neighbor_validation_scenario(base_config)
    nominal_config["neighbor"]["enabled"]["value"] = True
    # 参数模糊在本实验中关闭，T 与 zeta 保持配置中心值。
    nominal_config["idm"]["time_headway"]["value"] = 1.5
    nominal_config["lateral"]["zeta"]["value"] = 1.0
    nominal_params = stochastic_parameters_from_config(nominal_config).deterministic
    leader = make_uniform_leader(nominal_config, spec, nominal_params)
    neighbor_j = make_accelerating_neighbor(nominal_config, spec)
    names, numbers = fuzzy_numbers(spec)
    alphas = tuple(float(value) for value in spec["alpha_levels"])
    sample_count = int(spec["samples_per_alpha"])
    center_values = {name: number.mode for name, number in numbers.items()}

    cache: dict[tuple[tuple[str, float], ...], dict[str, np.ndarray]] = {}

    def simulate(values: dict[str, float]) -> dict[str, np.ndarray]:
        key = sample_key(names, values)
        if key not in cache:
            try:
                cache[key] = simulate_case(nominal_config, spec, values, leader, neighbor_j)
            except Exception as exc:
                raise RuntimeError(
                    "状态—几何模糊样本积分失败："
                    + json.dumps(values, ensure_ascii=False, sort_keys=True)
                ) from exc
        return cache[key]

    center = simulate(center_values)
    standalone_values = {
        alpha: sample_alpha_domain(names, numbers, alpha, sample_count) for alpha in alphas
    }
    families: dict[float, list[dict[str, np.ndarray]]] = {}
    envelopes: dict[float, dict[str, np.ndarray]] = {}
    sample_counts: dict[str, int] = {}
    for alpha in alphas:
        union: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}
        for inner_alpha in alphas:
            if inner_alpha + 1e-15 < alpha:
                continue
            for values in standalone_values[inner_alpha]:
                union[sample_key(names, values)] = values
        family = [simulate(values) for values in union.values()]
        families[alpha] = family
        envelopes[alpha] = envelope(family)
        sample_counts[str(alpha)] = len(family)

    alpha_one_error = float(np.max(np.abs(
        envelopes[1.0]["state_six_lower"] - center["state_six"]
    )))
    nesting = nesting_check(envelopes)

    convergence_counts = tuple(int(value) for value in spec["convergence_sample_counts"])
    convergence_envelopes: dict[int, dict[str, np.ndarray]] = {}
    for count in convergence_counts:
        values = sample_alpha_domain(names, numbers, 0.0, count)
        convergence_envelopes[count] = envelope([simulate(item) for item in values])
    reference_count = max(convergence_counts)
    reference = convergence_envelopes[reference_count]
    convergence: dict[str, dict[str, float]] = {}
    for count in convergence_counts:
        if count == reference_count:
            continue
        current = convergence_envelopes[count]
        convergence[str(count)] = {
            "u_lower_mps": float(np.max(np.abs(
                current["state_six_lower"][:, 1] - reference["state_six_lower"][:, 1]
            ))),
            "u_upper_mps": float(np.max(np.abs(
                current["state_six_upper"][:, 1] - reference["state_six_upper"][:, 1]
            ))),
            "y_lower_m": float(np.max(np.abs(
                current["state_six_lower"][:, 3] - reference["state_six_lower"][:, 3]
            ))),
            "y_upper_m": float(np.max(np.abs(
                current["state_six_upper"][:, 3] - reference["state_six_upper"][:, 3]
            ))),
            "delta_y_ml_lower_m": float(np.max(np.abs(
                current["delta_y_lower"][:, 0] - reference["delta_y_lower"][:, 0]
            ))),
            "delta_y_ml_upper_m": float(np.max(np.abs(
                current["delta_y_upper"][:, 0] - reference["delta_y_upper"][:, 0]
            ))),
        }

    all_runs = list(cache.values())
    finite = all(all(np.all(np.isfinite(run[key])) for key in ARRAY_KEYS) for run in all_runs)
    minimum_speed = min(float(np.min(run["state"][:, 1])) for run in all_runs)
    minimum_headway = min(float(np.min(run["headway"])) for run in all_runs)
    boundary_left = float(nominal_config["geometry"]["boundary_left"]["value"])
    boundary_right = float(nominal_config["geometry"]["boundary_right"]["value"])
    minimum_margin = min(float(min(
        np.min(run["state"][:, 2] - boundary_left),
        np.min(boundary_right - run["state"][:, 2]),
    )) for run in all_runs)
    active_indices = np.flatnonzero(center["active"])
    time = center["time"]
    leader_cfg = spec["uniformly_accelerating_leader"]
    start = float(leader_cfg["acceleration_start"]["value"])
    end = float(leader_cfg["acceleration_end"]["value"])
    active_time = np.minimum(np.maximum(time - start, 0.0), end - start)
    expected_leader_speed = (
        float(leader_cfg["initial_speed"]["value"])
        + float(leader_cfg["acceleration"]["value"]) * active_time
    )
    leader_speed_error = float(np.max(np.abs(center["leader"][:, 1] - expected_leader_speed)))
    figures = plot_results(nominal_config, spec, center, envelopes, families)

    outer_width = envelopes[0.0]["state_six_upper"] - envelopes[0.0]["state_six_lower"]
    summary = {
        "model_scope": "six fuzzy initial states + fuzzy perceived lane marks/center + common OU path + neighbor interaction",
        "fuzzy_inputs": list(names),
        "crisp_parameters": {"T_s": 1.5, "zeta_per_m": 1.0},
        "stochastic_interpretation": "conditional fuzzy envelope under common Wiener increments",
        "alpha_levels": list(alphas),
        "halton_samples_per_alpha": sample_count,
        "nested_family_counts": sample_counts,
        "unique_simulations": len(cache),
        "alpha_one_max_six_state_error": alpha_one_error,
        "nesting": nesting,
        "sampling_convergence_against_128": convergence,
        "all_finite": bool(finite),
        "minimum_speed_mps": minimum_speed,
        "minimum_headway_m": minimum_headway,
        "minimum_physical_road_margin_m": minimum_margin,
        "uniform_leader_speed_max_error_mps": leader_speed_error,
        "center_neighbor_active_start_s": float(time[active_indices[0]]) if active_indices.size else None,
        "center_neighbor_active_end_s": float(time[active_indices[-1]]) if active_indices.size else None,
        "alpha0_initial_six_state_widths": outer_width[0].tolist(),
        "alpha0_max_six_state_widths": np.max(outer_width, axis=0).tolist(),
        "figure_files": [str(path) for path in figures],
    }
    (OUTPUT / "state_geometry_fuzzy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    success = bool(
        alpha_one_error < 1e-10
        and nesting["pass"]
        and finite
        and minimum_speed >= 0.0
        and minimum_headway > 0.0
        and minimum_margin > 0.0
        and leader_speed_error < 1e-12
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
