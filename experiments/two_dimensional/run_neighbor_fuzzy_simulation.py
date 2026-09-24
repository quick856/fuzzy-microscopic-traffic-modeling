"""含 OU 扰动和相邻车 j 的 T̃、ζ̃ 二维模糊—随机仿真。

名称固定如下：车辆 i 为被求解的目标跟驰车辆，le(i) 为其前车，j 为
相邻车道车辆。前车和相邻车轨迹保持确定，模糊性来自 T̃ 与 ζ̃；所有
参数组合共用同一固定种子的 Wiener 增量，以分离模糊宽度与随机样本差异。
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "figures" / "neighbor_fuzzy_ou_simulation"
sys.path.insert(0, str(ROOT / "src"))

from two_dimensional_fuzzy_neighbor_ou.dynamics import (
    lateral_acceleration_components,
    stochastic_parameters_from_config,
    two_dimensional_stochastic_diffusion,
    two_dimensional_stochastic_drift,
)
from two_dimensional_fuzzy_neighbor_ou.fuzzy import (
    apply_parameter_values,
    fuzzy_parameters_from_config,
    parameter_grid,
    trajectory_envelope,
)
from two_dimensional_fuzzy_neighbor_ou.neighbor import (
    is_neighbor_in_influence_region,
    neighbor_influence_axes,
    neighbor_interaction_force,
)
from two_dimensional_fuzzy_neighbor_ou.scenarios import (
    configure_neighbor_validation_scenario,
    make_leader,
    make_neighbor_trajectory,
)
from two_dimensional_fuzzy_neighbor_ou.solver import solve_sde_euler_maruyama


COLORS = {
    "center": "#174F78",
    "without": "#707C86",
    "leader": "#C06442",
    "neighbor": "#2C8C7B",
    "alpha0": "#9BC4DF",
    "alpha05": "#4D8EB9",
    "road": "#46515B",
    "mark": "#B4943C",
}


def _values(config: dict, group: str) -> dict:
    return {name: item["value"] for name, item in config[group].items()}


def simulate_case(
    config: dict,
    params,
    leader,
    neighbor_j,
) -> dict[str, np.ndarray]:
    """用共同随机数求解含 OU 扰动和邻车作用的六维状态。"""
    scenario = _values(config, "scenario")
    center = float(config["geometry"]["lane_center"]["value"])
    initial_offset = float(scenario["offsets"][2])
    duration = float(config["neighbor_scenario"]["duration"]["value"])
    dt = float(config["numerics"]["dt"]["value"])
    seed = int(config["numerics"]["random_seed"]["value"])
    initial = np.array(
        [scenario["initial_x"], scenario["initial_vx"], 0.0,
         center + initial_offset, scenario["initial_vy"], 0.0],
        dtype=float,
    )

    def drift(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_drift(
            t, state, leader(t), params, (neighbor_j(t),)
        )

    def diffusion(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_diffusion(t, state, params)

    times, states_six = solve_sde_euler_maruyama(
        drift, diffusion, initial, duration, dt, seed
    )
    states = states_six[:, [0, 1, 3, 4]]
    leaders = np.array([leader(float(t)) for t in times])
    neighbors = np.array([neighbor_j(float(t)) for t in times])
    base = params.deterministic
    forces = np.array(
        [
            neighbor_interaction_force(state, other, base.lateral, base.neighbor)
            for state, other in zip(states, neighbors)
        ],
        dtype=float,
    )
    active = np.array(
        [
            is_neighbor_in_influence_region(state, other, base.lateral, base.neighbor)
            for state, other in zip(states, neighbors)
        ],
        dtype=bool,
    )
    acceleration = np.array(
        [drift(float(t), state)[[1, 4]] for t, state in zip(times, states_six)],
        dtype=float,
    )
    lateral_components = np.array(
        [lateral_acceleration_components(state[2], state[3], base) for state in states],
        dtype=float,
    )
    x_ellip = np.array(
        [neighbor_influence_axes(state[1], base.lateral, base.neighbor)[0] for state in states]
    )
    dx = neighbors[:, 0] - states[:, 0]
    dy = np.abs(neighbors[:, 2] - states[:, 2])
    metric = (dx / x_ellip) ** 2 + (dy / base.lateral.lane_width) ** 2
    return {
        "time": times,
        "state": states,
        "leader": leaders,
        "neighbor": neighbors,
        "acceleration": acceleration,
        "lateral_components": lateral_components,
        "neighbor_force": forces,
        "noise_state": states_six[:, [2, 5]],
        "active": active,
        "ellipse_metric": metric,
        "headway": leaders[:, 0] - states[:, 0],
    }


def fuzzy_envelope(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """扩展通用包络，并保留二维轨迹边界点的 x-y 配对。"""
    result = trajectory_envelope(samples)
    forces = np.stack([sample["neighbor_force"] for sample in samples])
    metrics = np.stack([sample["ellipse_metric"] for sample in samples])
    active = np.stack([sample["active"] for sample in samples])
    states = np.stack([sample["state"] for sample in samples])
    time_index = np.arange(states.shape[1])
    lower_index = np.argmin(states[:, :, 2], axis=0)
    upper_index = np.argmax(states[:, :, 2], axis=0)
    result.update(
        {
            "neighbor_force_lower": np.min(forces, axis=0),
            "neighbor_force_upper": np.max(forces, axis=0),
            "ellipse_metric_lower": np.min(metrics, axis=0),
            "ellipse_metric_upper": np.max(metrics, axis=0),
            "active_any": np.any(active, axis=0),
            "active_all": np.all(active, axis=0),
            "xy_lower": states[lower_index, time_index][:, [0, 2]],
            "xy_upper": states[upper_index, time_index][:, [0, 2]],
        }
    )
    return result


def nesting_check(envelopes: dict[float, dict[str, np.ndarray]]) -> dict[str, object]:
    """对状态、加速度和邻车作用的所有上下包络检查 α 嵌套。"""
    keys = (
        ("state_lower", "state_upper"),
        ("acceleration_lower", "acceleration_upper"),
        ("neighbor_force_lower", "neighbor_force_upper"),
    )
    worst = 0.0
    details: dict[str, float] = {}
    alphas = sorted(envelopes)
    for lower_key, upper_key in keys:
        key_worst = 0.0
        for outer_alpha, inner_alpha in zip(alphas[:-1], alphas[1:]):
            outer = envelopes[outer_alpha]
            inner = envelopes[inner_alpha]
            key_worst = max(
                key_worst,
                float(np.max(outer[lower_key] - inner[lower_key])),
                float(np.max(inner[upper_key] - outer[upper_key])),
            )
        details[lower_key.removesuffix("_lower")] = max(key_worst, 0.0)
        worst = max(worst, key_worst)
    return {"pass": bool(worst <= 1e-10), "maximum_violation": max(worst, 0.0), "details": details}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
        }
    )


def _shade_active(ax, time: np.ndarray, active: np.ndarray) -> None:
    ax.fill_between(
        time, 0.0, 1.0, where=active, transform=ax.get_xaxis_transform(),
        color="#B9D3E5", alpha=0.20, label="中心参数轨迹位于影响域",
    )


def plot_crisp(config: dict, without: dict, center: dict) -> Path:
    """确定性邻车场景的完整状态演化。"""
    time = center["time"]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.2), sharex=True)
    panels = axes.flat
    panels[0].plot(time, center["leader"][:, 1] * 3.6, color=COLORS["leader"], linestyle="--", label="前车 le(i)")
    panels[0].plot(time, center["state"][:, 1] * 3.6, color=COLORS["center"], label="目标车辆 i（含相邻车作用）")
    panels[0].plot(time, without["state"][:, 1] * 3.6, color=COLORS["without"], linestyle=":", label="目标车辆 i（无相邻车作用）")
    panels[0].set_title("纵向速度")
    panels[0].set_ylabel("速度（km/h）")

    panels[1].plot(time, center["acceleration"][:, 0], color=COLORS["center"], label="车辆 i 纵向总加速度")
    panels[1].plot(time, center["neighbor_force"][:, 0], color=COLORS["leader"], linestyle="--", label=r"相邻车纵向作用 $F_{ij}^{ne-lon}$")
    panels[1].axhline(0, color="#8A939A", linewidth=0.7)
    panels[1].set_title("纵向加速度")
    panels[1].set_ylabel("加速度（m/s²）")

    panels[2].plot(time, center["headway"], color=COLORS["center"], label="含相邻车作用")
    panels[2].plot(time, without["headway"], color=COLORS["without"], linestyle=":", label="无相邻车作用")
    panels[2].set_title("车辆 i 与前车 le(i) 的位置差")
    panels[2].set_ylabel(r"$h_i$（m）")

    lane_center = float(config["geometry"]["lane_center"]["value"])
    panels[3].plot(time, center["state"][:, 2], color=COLORS["center"], label="目标车辆 i")
    panels[3].plot(time, without["state"][:, 2], color=COLORS["without"], linestyle=":", label="车辆 i（无相邻车作用）")
    panels[3].plot(time, center["neighbor"][:, 2], color=COLORS["neighbor"], linestyle="--", label="相邻车 j")
    panels[3].axhline(lane_center, color="#7B8F83", linewidth=0.8, linestyle="-.", label="目标车道中心")
    panels[3].set_title("横向位置")
    panels[3].set_ylabel("y（m）")

    panels[4].plot(time, center["state"][:, 3], color=COLORS["center"], label="目标车辆 i")
    panels[4].plot(time, without["state"][:, 3], color=COLORS["without"], linestyle=":", label="无相邻车作用")
    panels[4].axhline(0, color="#8A939A", linewidth=0.7)
    panels[4].set_title("横向速度")
    panels[4].set_ylabel(r"$v_i$（m/s）")

    panels[5].plot(time, center["acceleration"][:, 1], color=COLORS["center"], label="车辆 i 横向总加速度")
    panels[5].plot(time, center["neighbor_force"][:, 1], color=COLORS["neighbor"], linestyle="--", label=r"相邻车横向作用 $F_{ij}^{ne-lat}$")
    panels[5].axhline(0, color="#8A939A", linewidth=0.7)
    panels[5].set_title("横向加速度")
    panels[5].set_ylabel("加速度（m/s²）")

    for ax in panels:
        _shade_active(ax, time, center["active"])
        ax.legend(frameon=False, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("时间 t（s）")
    fig.suptitle("相邻车作用下目标车辆 i 的确定性二维状态演化", y=0.99, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = OUTPUT / "01_neighbor_crisp_state_evolution.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_fuzzy_states(
    center_without: dict,
    center: dict,
    zero: dict,
    half: dict,
) -> Path:
    """车辆 i 的速度、加速度、横向位置和横向加速度包络。"""
    time = center["time"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    specs = (
        (axes[0, 0], 1, "state", 3.6, "纵向速度", r"$u_i$（km/h）"),
        (axes[0, 1], 0, "acceleration", 1.0, "纵向加速度", r"$a_x$（m/s²）"),
        (axes[1, 0], 2, "state", 1.0, "横向位置", r"$y_i$（m）"),
        (axes[1, 1], 1, "acceleration", 1.0, "横向加速度", r"$a_y$（m/s²）"),
    )
    for index, (ax, column, kind, scale, title, ylabel) in enumerate(specs):
        lower_key = f"{kind}_lower"
        upper_key = f"{kind}_upper"
        center_values = center[kind][:, column] if kind == "state" else center["acceleration"][:, column]
        without_values = center_without[kind][:, column] if kind == "state" else center_without["acceleration"][:, column]
        ax.fill_between(time, zero[lower_key][:, column] * scale, zero[upper_key][:, column] * scale,
                        color=COLORS["alpha0"], alpha=0.30, label="α=0 模糊包络")
        ax.fill_between(time, half[lower_key][:, column] * scale, half[upper_key][:, column] * scale,
                        color=COLORS["alpha05"], alpha=0.36, label="α=0.5 模糊包络")
        ax.plot(time, center_values * scale, color=COLORS["center"], linewidth=1.8, label="中心参数轨迹（含相邻车）")
        ax.plot(time, without_values * scale, color=COLORS["without"], linestyle=":", linewidth=1.2, label="中心参数轨迹（无相邻车）")
        _shade_active(ax, time, center["active"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if index >= 2:
            ax.set_xlabel("时间 t（s）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
    fig.suptitle(r"相邻车场景下 $\tilde{T}$ 与 $\tilde{\zeta}$ 产生的二维状态模糊包络", y=0.99, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    path = OUTPUT / "02_neighbor_fuzzy_state_envelopes.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _fill_xy_band(ax, envelope: dict, color: str, alpha: float, label: str) -> None:
    lower = envelope["xy_lower"]
    upper = envelope["xy_upper"]
    polygon_x = np.concatenate([lower[:, 0], upper[::-1, 0]])
    polygon_y = np.concatenate([lower[:, 1], upper[::-1, 1]])
    ax.fill(polygon_x, polygon_y, color=color, alpha=alpha, linewidth=0, label=label)


def plot_xy(config: dict, center: dict, zero: dict, half: dict) -> Path:
    """绘制保持同一样本 x-y 配对的逐时刻轨迹包络。"""
    geometry = _values(config, "geometry")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    for ax in axes:
        ax.axhspan(geometry["boundary_left"], geometry["boundary_right"], color="#F3F5F7", zorder=0)
        for idx, boundary in enumerate((geometry["boundary_left"], geometry["boundary_right"])):
            ax.axhline(boundary, color=COLORS["road"], linewidth=1.0, label="道路边界" if idx == 0 else None)
        for idx, mark in enumerate(geometry["markings"]):
            ax.axhline(mark, color=COLORS["mark"], linestyle="--", linewidth=0.8, label="车道标线" if idx == 0 else None)
        _fill_xy_band(ax, zero, COLORS["alpha0"], 0.42, "α=0 二维轨迹包络")
        _fill_xy_band(ax, half, COLORS["alpha05"], 0.46, "α=0.5 二维轨迹包络")
        ax.plot(center["state"][:, 0], center["state"][:, 2], color=COLORS["center"], linewidth=2.0, label="目标车辆 i 中心参数轨迹")
        ax.plot(center["leader"][:, 0], center["leader"][:, 2], color=COLORS["leader"], linestyle="--", linewidth=1.2, label="前车 le(i)")
        ax.plot(center["neighbor"][:, 0], center["neighbor"][:, 2], color=COLORS["neighbor"], linewidth=1.4, label="相邻车 j")
        ax.set_xlabel("纵向位置 x（m）")
        ax.set_ylabel("横向位置 y（m）")
    axes[0].set_title("完整道路范围")
    active_index = np.flatnonzero(center["active"])
    if active_index.size:
        x_active = center["state"][active_index, 0]
        axes[1].set_xlim(float(np.min(x_active) - 25.0), float(np.max(x_active) + 25.0))
        y_stack = np.concatenate([zero["xy_lower"][:, 1], zero["xy_upper"][:, 1]])
        axes[1].set_ylim(float(np.min(y_stack) - 0.05), float(np.max(y_stack) + 0.05))
    axes[1].set_title("相邻车作用区间局部放大")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=False)
    fig.suptitle("相邻车场景的二维模糊轨迹包络", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path = OUTPUT / "03_neighbor_fuzzy_xy_envelope.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_neighbor_forces(center: dict, zero: dict, half: dict) -> Path:
    """绘制作用力包络诊断；该图须结合网格收敛结果解释。"""
    time = center["time"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, column, title, color in (
        (axes[0, 0], 0, "相邻车纵向作用", COLORS["leader"]),
        (axes[0, 1], 1, "相邻车横向作用", COLORS["neighbor"]),
    ):
        ax.fill_between(time, zero["neighbor_force_lower"][:, column], zero["neighbor_force_upper"][:, column],
                        color=COLORS["alpha0"], alpha=0.30, label="α=0 模糊包络")
        ax.fill_between(time, half["neighbor_force_lower"][:, column], half["neighbor_force_upper"][:, column],
                        color=COLORS["alpha05"], alpha=0.36, label="α=0.5 模糊包络")
        ax.plot(time, center["neighbor_force"][:, column], color=color, linewidth=1.8, label="中心参数曲线")
        ax.axhline(0, color="#8A939A", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylabel("加速度贡献（m/s²）")
        ax.legend(frameon=False, loc="best")

    axes[1, 0].fill_between(time, zero["ellipse_metric_lower"], zero["ellipse_metric_upper"],
                            color=COLORS["alpha0"], alpha=0.32, label="α=0 判别值范围")
    axes[1, 0].plot(time, center["ellipse_metric"], color=COLORS["center"], linewidth=1.7, label="中心参数判别值")
    axes[1, 0].axhline(1.0, color=COLORS["leader"], linestyle="--", linewidth=1.0, label="影响域边界")
    axes[1, 0].set_title("椭圆影响域判别")
    axes[1, 0].set_ylabel("无量纲判别值")
    axes[1, 0].legend(frameon=False, loc="best")

    width_u = zero["state_upper"][:, 1] - zero["state_lower"][:, 1]
    width_y = zero["state_upper"][:, 2] - zero["state_lower"][:, 2]
    ax_left = axes[1, 1]
    ax_right = ax_left.twinx()
    line_u = ax_left.plot(time, width_u, color=COLORS["center"], label=r"速度宽度 $W_u$")[0]
    line_y = ax_right.plot(time, width_y, color=COLORS["leader"], label=r"横向位置宽度 $W_y$")[0]
    ax_left.set_title("α=0 状态包络宽度")
    ax_left.set_ylabel(r"$W_u$（m/s）", color=COLORS["center"])
    ax_right.set_ylabel(r"$W_y$（m）", color=COLORS["leader"])
    ax_left.legend([line_u, line_y], [line_u.get_label(), line_y.get_label()], frameon=False, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("时间 t（s）")
    fig.text(
        0.5, 0.015,
        "说明：瞬时相邻车作用包含影响域开关，须与椭圆判别值和参数网格收敛结果共同解释。",
        ha="center", va="bottom", fontsize=9, color="#A54A34",
    )
    fig.suptitle("诊断：相邻车作用力包络与状态不确定性传播", y=0.99, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    path = OUTPUT / "04_neighbor_force_envelope_grid_diagnostic.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_complete_model_figures(
    config: dict,
    center: dict,
    zero: dict,
    half: dict,
) -> list[Path]:
    """逐图绘制含相邻车辆作用的二维模糊模型结果。

    每个文件只放一个坐标轴。所有中心轨迹和模糊包络都来自启用相邻车辆
    作用的确定性 ODE 族，不再混入“关闭相邻车辆作用”的对照轨迹。
    """
    time = center["time"]
    paths: list[Path] = []

    def finish(fig: plt.Figure, ax: plt.Axes, name: str, title: str, ylabel: str) -> None:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("时间 t（s）")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#CBD2D9", alpha=0.45, linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                       ncol=min(4, len(handles)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        path = OUTPUT / name
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)

    def state_band(
        name: str,
        title: str,
        column: int,
        ylabel: str,
        scale: float = 1.0,
        reference: tuple[np.ndarray, str] | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.fill_between(time, zero["state_lower"][:, column] * scale,
                        zero["state_upper"][:, column] * scale,
                        color=COLORS["alpha0"], alpha=0.30, label="α=0 模糊包络")
        ax.fill_between(time, half["state_lower"][:, column] * scale,
                        half["state_upper"][:, column] * scale,
                        color=COLORS["alpha05"], alpha=0.38, label="α=0.5 模糊包络")
        ax.plot(time, center["state"][:, column] * scale, color=COLORS["center"],
                linewidth=2.0, label="中心参数轨迹")
        if reference is not None:
            ax.plot(time, reference[0] * scale, color=COLORS["leader"], linestyle="--",
                    linewidth=1.4, label=reference[1])
        finish(fig, ax, name, title, ylabel)

    def acceleration_band(name: str, title: str, column: int, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.fill_between(time, zero["acceleration_lower"][:, column],
                        zero["acceleration_upper"][:, column],
                        color=COLORS["alpha0"], alpha=0.30, label="α=0 模糊包络")
        ax.fill_between(time, half["acceleration_lower"][:, column],
                        half["acceleration_upper"][:, column],
                        color=COLORS["alpha05"], alpha=0.38, label="α=0.5 模糊包络")
        ax.plot(time, center["acceleration"][:, column], color=COLORS["center"],
                linewidth=2.0, label="中心参数轨迹")
        ax.axhline(0.0, color="#8A939A", linewidth=0.7)
        finish(fig, ax, name, title, ylabel)

    # 单独的邻车实现验证图；不作为主要状态结果图。
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(time, center["ellipse_metric"], color=COLORS["center"], linewidth=1.9,
            label="中心参数椭圆判别值")
    ax.axhline(1.0, color=COLORS["leader"], linestyle="--", linewidth=1.2,
               label="影响域边界")
    ax.fill_between(time, 0.0, 1.0, where=center["active"], color=COLORS["alpha0"],
                    alpha=0.30, label="满足 Δx>0 且判别值≤1")
    finish(fig, ax, "01_neighbor_influence_validation.png",
           "相邻车辆 j 的前方椭圆影响域判定", "无量纲判别值")

    state_band("02_fuzzy_longitudinal_position.png", "含随机扰动及相邻车辆作用的模糊纵向位置",
               0, r"$x_i$（m）", reference=(center["leader"][:, 0], "前车 le(i)"))
    state_band("03_fuzzy_longitudinal_speed.png", "含随机扰动及相邻车辆作用的模糊纵向速度",
               1, r"$u_i$（km/h）", scale=3.6,
               reference=(center["leader"][:, 1], "前车 le(i)"))
    acceleration_band("04_fuzzy_longitudinal_acceleration.png",
                      "含随机扰动及相邻车辆作用的模糊纵向加速度", 0, r"$a_i^x$（m/s²）")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.fill_between(time, zero["headway_lower"], zero["headway_upper"],
                    color=COLORS["alpha0"], alpha=0.30, label="α=0 模糊包络")
    ax.fill_between(time, half["headway_lower"], half["headway_upper"],
                    color=COLORS["alpha05"], alpha=0.38, label="α=0.5 模糊包络")
    ax.plot(time, center["headway"], color=COLORS["center"], linewidth=2.0,
            label="中心参数轨迹")
    finish(fig, ax, "05_fuzzy_headway.png", "含随机扰动及相邻车辆作用的模糊前车位置差",
           r"$h_i=x_{le(i)}-x_i$（m）")

    state_band("06_fuzzy_lateral_position.png", "含随机扰动及相邻车辆作用的模糊横向位置",
               2, r"$y_i$（m）",
               reference=(np.full_like(time, float(config["geometry"]["lane_center"]["value"])),
                          "目标车道中心"))
    state_band("07_fuzzy_lateral_velocity.png", "含随机扰动及相邻车辆作用的模糊横向速度",
               3, r"$v_i$（m/s）")
    acceleration_band("08_fuzzy_lateral_acceleration.png",
                      "含随机扰动及相邻车辆作用的模糊横向加速度", 1, r"$a_i^y$（m/s²）")

    geometry = _values(config, "geometry")
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.axhspan(geometry["boundary_left"], geometry["boundary_right"],
               color="#F3F5F7", zorder=0)
    for index, boundary in enumerate((geometry["boundary_left"], geometry["boundary_right"])):
        ax.axhline(boundary, color=COLORS["road"], linewidth=1.0,
                   label="道路边界" if index == 0 else None)
    for index, mark in enumerate(geometry["markings"]):
        ax.axhline(mark, color=COLORS["mark"], linestyle="--", linewidth=0.8,
                   label="车道标线" if index == 0 else None)
    _fill_xy_band(ax, zero, COLORS["alpha0"], 0.38, "α=0 二维轨迹包络")
    _fill_xy_band(ax, half, COLORS["alpha05"], 0.44, "α=0.5 二维轨迹包络")
    ax.plot(center["state"][:, 0], center["state"][:, 2], color=COLORS["center"],
            linewidth=2.0, label="目标车辆 i 中心参数轨迹")
    ax.plot(center["leader"][:, 0], center["leader"][:, 2], color=COLORS["leader"],
            linestyle="--", linewidth=1.2, label="前车 le(i)")
    ax.plot(center["neighbor"][:, 0], center["neighbor"][:, 2], color=COLORS["neighbor"],
            linewidth=1.4, label="相邻车辆 j")
    ax.set_xlabel("纵向位置 x（m）")
    ax.set_ylabel("横向位置 y（m）")
    ax.set_title("含随机扰动及相邻车辆作用的二维模糊轨迹", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, color="#CBD2D9", alpha=0.45, linewidth=0.65)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = OUTPUT / "09_fuzzy_xy_trajectory.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(path)

    for name, title, column, color in (
        ("10_neighbor_longitudinal_force.png", "相邻车辆 j 对目标车辆 i 的纵向作用",
         0, COLORS["leader"]),
        ("11_neighbor_lateral_force.png", "相邻车辆 j 对目标车辆 i 的横向作用",
         1, COLORS["neighbor"]),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.fill_between(time, zero["neighbor_force_lower"][:, column],
                        zero["neighbor_force_upper"][:, column],
                        color=COLORS["alpha0"], alpha=0.30, label="α=0 模糊包络")
        ax.fill_between(time, half["neighbor_force_lower"][:, column],
                        half["neighbor_force_upper"][:, column],
                        color=COLORS["alpha05"], alpha=0.38, label="α=0.5 模糊包络")
        ax.plot(time, center["neighbor_force"][:, column], color=color, linewidth=2.0,
                label="中心参数曲线")
        ax.axhline(0.0, color="#8A939A", linewidth=0.7)
        symbol = r"$F_{ij}^{ne-lon}$（m/s²）" if column == 0 else r"$F_{ij}^{ne-lat}$（m/s²）"
        finish(fig, ax, name, title, symbol)

    for name, title, column, color, ylabel in (
        ("12_longitudinal_noise_state.png", "纵向加速度扰动状态", 0,
         COLORS["leader"], r"$z_i^{lon}$（m/s²）"),
        ("13_lateral_noise_state.png", "横向加速度扰动状态", 1,
         COLORS["neighbor"], r"$z_i^{lat}$（m/s²）"),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.plot(time, center["noise_state"][:, column], color=color, linewidth=1.5,
                label="固定种子中心参数轨迹")
        ax.axhline(0.0, color="#8A939A", linestyle="--", linewidth=0.8,
                   label="长期均值 μ=0")
        finish(fig, ax, name, title, ylabel)

    return paths


def main() -> int:
    _style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs" / "two_dimensional_fuzzy_neighbor_ou.json").read_text(encoding="utf-8"))
    config = configure_neighbor_validation_scenario(config)
    config["neighbor"]["enabled"]["value"] = True
    fuzzy_parameters = fuzzy_parameters_from_config(config)
    fuzzy_cfg = config["fuzzy_next_stage"]
    alphas = tuple(float(value) for value in fuzzy_cfg["alpha_levels"])
    grid_n = int(fuzzy_cfg["grid_n"])
    center_values = {parameter.name: parameter.number.mode for parameter in fuzzy_parameters}
    center_config = deepcopy(config)
    apply_parameter_values(center_config, fuzzy_parameters, center_values)
    center_params = stochastic_parameters_from_config(center_config)
    leader = make_leader(center_config, center_params.deterministic, accelerating=False)
    neighbor_j = make_neighbor_trajectory(center_config)

    cache: dict[tuple[tuple[str, float], ...], dict[str, np.ndarray]] = {}

    def simulate(values: dict[str, float]) -> dict[str, np.ndarray]:
        key = tuple((name, round(float(values[name]), 14)) for name in sorted(values))
        if key not in cache:
            sample_config = deepcopy(config)
            apply_parameter_values(sample_config, fuzzy_parameters, values)
            params = stochastic_parameters_from_config(sample_config)
            cache[key] = simulate_case(sample_config, params, leader, neighbor_j)
        return cache[key]

    center = simulate(center_values)
    base_grids = {alpha: parameter_grid(fuzzy_parameters, alpha, grid_n) for alpha in alphas}
    envelopes: dict[float, dict[str, np.ndarray]] = {}
    sample_counts: dict[str, int] = {}
    for alpha in alphas:
        union: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}
        for inner_alpha in alphas:
            if inner_alpha + 1e-15 < alpha:
                continue
            for values in base_grids[inner_alpha]:
                key = tuple((name, round(float(values[name]), 14)) for name in sorted(values))
                union[key] = values
        samples = [simulate(values) for values in union.values()]
        envelopes[alpha] = fuzzy_envelope(samples)
        sample_counts[str(alpha)] = len(samples)

    alpha_one = envelopes[1.0]
    alpha_one_error = float(
        np.max(
            np.abs(alpha_one["state_lower"] - center["state"])
        )
    )
    nesting = nesting_check(envelopes)

    convergence_envelopes: dict[int, dict[str, np.ndarray]] = {}
    for size in (5, 7, 9):
        convergence_samples = [
            simulate(values) for values in parameter_grid(fuzzy_parameters, 0.0, size)
        ]
        convergence_envelopes[size] = fuzzy_envelope(convergence_samples)
    reference = convergence_envelopes[9]
    convergence: dict[str, dict[str, float]] = {}
    for size in (5, 7):
        envelope = convergence_envelopes[size]
        convergence[str(size)] = {
            "speed_lower_mps": float(np.max(np.abs(
                envelope["state_lower"][:, 1] - reference["state_lower"][:, 1]
            ))),
            "speed_upper_mps": float(np.max(np.abs(
                envelope["state_upper"][:, 1] - reference["state_upper"][:, 1]
            ))),
            "lateral_lower_m": float(np.max(np.abs(
                envelope["state_lower"][:, 2] - reference["state_lower"][:, 2]
            ))),
            "lateral_upper_m": float(np.max(np.abs(
                envelope["state_upper"][:, 2] - reference["state_upper"][:, 2]
            ))),
            "neighbor_lon_lower_mps2": float(np.max(np.abs(
                envelope["neighbor_force_lower"][:, 0]
                - reference["neighbor_force_lower"][:, 0]
            ))),
            "neighbor_lon_upper_mps2": float(np.max(np.abs(
                envelope["neighbor_force_upper"][:, 0]
                - reference["neighbor_force_upper"][:, 0]
            ))),
            "neighbor_lat_lower_mps2": float(np.max(np.abs(
                envelope["neighbor_force_lower"][:, 1]
                - reference["neighbor_force_lower"][:, 1]
            ))),
            "neighbor_lat_upper_mps2": float(np.max(np.abs(
                envelope["neighbor_force_upper"][:, 1]
                - reference["neighbor_force_upper"][:, 1]
            ))),
        }
    force_convergence_error = max(
        convergence["7"]["neighbor_lon_lower_mps2"],
        convergence["7"]["neighbor_lon_upper_mps2"],
        convergence["7"]["neighbor_lat_lower_mps2"],
        convergence["7"]["neighbor_lat_upper_mps2"],
    )
    force_convergence_tolerance = 0.01
    zero = envelopes[0.0]
    half = envelopes[0.5]
    width_u = zero["state_upper"][:, 1] - zero["state_lower"][:, 1]
    width_y = zero["state_upper"][:, 2] - zero["state_lower"][:, 2]
    active_index = np.flatnonzero(center["active"])
    geometry = _values(config, "geometry")
    finite = all(np.all(np.isfinite(run["state"])) for run in cache.values())
    minimum_speed = min(float(np.min(run["state"][:, 1])) for run in cache.values())
    minimum_headway = min(float(np.min(run["headway"])) for run in cache.values())
    minimum_margin = min(
        float(
            min(
                np.min(run["state"][:, 2] - geometry["boundary_left"]),
                np.min(geometry["boundary_right"] - run["state"][:, 2]),
            )
        )
        for run in cache.values()
    )

    figures = plot_complete_model_figures(config, center, zero, half)
    summary = {
        "simulation_scope": (
            "hybrid fuzzy-stochastic six-state model with OU noise and neighbor interaction; "
            "all fuzzy samples use common Wiener increments"
        ),
        "random_seed": int(config["numerics"]["random_seed"]["value"]),
        "vehicle_names": {
            "vehicle_i": "被求解的目标跟驰车辆",
            "leader_le_i": "车辆 i 的确定性前车",
            "neighbor_j": "相邻车道的确定性车辆",
        },
        "fuzzy_parameters": center_values,
        "alpha_levels": alphas,
        "grid_n": grid_n,
        "sample_counts": sample_counts,
        "unique_parameter_runs_with_common_noise": len(cache),
        "alpha_one_max_state_error": alpha_one_error,
        "nesting": nesting,
        "grid_convergence_against_n9": convergence,
        "force_envelope_status": (
            "converged_against_n9_at_0.01_mps2_tolerance"
            if force_convergence_error <= force_convergence_tolerance
            else "not_converged_in_sup_norm_up_to_grid_n_9"
        ),
        "force_envelope_n7_vs_n9_max_error_mps2": force_convergence_error,
        "all_finite": bool(finite),
        "minimum_speed_mps": minimum_speed,
        "minimum_headway_m": minimum_headway,
        "minimum_road_margin_m": minimum_margin,
        "center_active_start_s": float(center["time"][active_index[0]]) if active_index.size else None,
        "center_active_end_s": float(center["time"][active_index[-1]]) if active_index.size else None,
        "center_max_abs_neighbor_lon_mps2": float(np.max(np.abs(center["neighbor_force"][:, 0]))),
        "center_max_abs_neighbor_lat_mps2": float(np.max(np.abs(center["neighbor_force"][:, 1]))),
        "center_max_abs_lateral_displacement_m": float(np.max(np.abs(
            center["state"][:, 2] - float(config["geometry"]["lane_center"]["value"])
        ))),
        "alpha0_max_speed_width_mps": float(np.max(width_u)),
        "alpha0_time_of_max_speed_width_s": float(center["time"][int(np.argmax(width_u))]),
        "alpha0_max_lateral_width_m": float(np.max(width_y)),
        "alpha0_time_of_max_lateral_width_s": float(center["time"][int(np.argmax(width_y))]),
        "figure_files": [str(path) for path in figures],
    }
    (OUTPUT / "neighbor_fuzzy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    success = bool(
        alpha_one_error < 1e-8
        and nesting["pass"]
        and finite
        and minimum_speed >= 0.0
        and minimum_headway > 0.0
        and minimum_margin > 0.0
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())


