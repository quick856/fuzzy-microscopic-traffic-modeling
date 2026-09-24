"""生成二维模糊微分模型的主体论文图；OU 仅保留一张附属验证图。"""

from __future__ import annotations

from copy import deepcopy
import json
from math import pi, sqrt
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "figures" / "two_dimensional" / "fde_main"
sys.path.insert(0, str(REPO / "src"))

from two_dimensional_fuzzy.dynamics import (  # noqa: E402
    parameters_from_config,
    stochastic_parameters_from_config,
    two_dimensional_stochastic_diffusion,
    two_dimensional_stochastic_drift,
)
from two_dimensional_fuzzy.fuzzy import (  # noqa: E402
    apply_parameter_values,
    fuzzy_parameters_from_config,
    nesting_violations,
    parameter_grid,
    trajectory_envelope,
)
from two_dimensional_fuzzy.scenarios import make_leader, simulate_crisp  # noqa: E402
from two_dimensional_fuzzy.solver import solve_sde_euler_maruyama  # noqa: E402


COLORS = {
    "outer": "#9EC5DF",
    "inner": "#4C91BF",
    "center": "#164E78",
    "leader": "#C4663F",
    "target": "#2A8C7B",
    "gray": "#66727C",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.titlesize": 10.4,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.1,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "savefig.dpi": 360,
            "figure.dpi": 130,
        }
    )


def decorate(ax: plt.Axes, panel: str) -> None:
    ax.grid(True, color="#CBD2D9", alpha=0.45, linewidth=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.12, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=10)


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_fuzzy_data() -> tuple[dict, dict, dict, dict, dict[float, list[dict]]]:
    config = json.loads((REPO / "configs/two_dimensional_fuzzy.json").read_text(encoding="utf-8"))
    fuzzy_parameters = fuzzy_parameters_from_config(config)
    alpha_levels = tuple(float(value) for value in config["fuzzy_next_stage"]["alpha_levels"])
    grid_n = int(config["fuzzy_next_stage"]["grid_n"])
    scenario = {key: item["value"] for key, item in config["scenario"].items()}
    dt = float(config["numerics"]["dt"]["value"])
    offset = float(scenario["offsets"][2])
    duration = float(scenario["following_duration"])
    center_values = {parameter.name: parameter.number.mode for parameter in fuzzy_parameters}
    center_config = deepcopy(config)
    apply_parameter_values(center_config, fuzzy_parameters, center_values)
    center_params = parameters_from_config(center_config)
    leader = make_leader(center_config, center_params, accelerating=True)
    cache: dict[tuple, dict] = {}

    def simulate(values: dict[str, float]) -> dict:
        key = tuple((name, round(float(values[name]), 14)) for name in sorted(values))
        if key not in cache:
            sample_config = deepcopy(config)
            apply_parameter_values(sample_config, fuzzy_parameters, values)
            params = parameters_from_config(sample_config)
            cache[key] = simulate_crisp(sample_config, params, leader, offset, duration, dt)
        return cache[key]

    crisp = simulate(center_values)
    base_grids = {alpha: parameter_grid(fuzzy_parameters, alpha, grid_n) for alpha in alpha_levels}
    envelopes: dict[float, dict] = {}
    samples: dict[float, list[dict]] = {}
    for alpha in alpha_levels:
        union: dict[tuple, dict[str, float]] = {}
        for inner_alpha in alpha_levels:
            if inner_alpha + 1e-15 < alpha:
                continue
            for values in base_grids[inner_alpha]:
                key = tuple((name, round(float(values[name]), 14)) for name in sorted(values))
                union[key] = values
        family = [simulate(values) for values in union.values()]
        envelopes[alpha] = trajectory_envelope(family)
        samples[alpha] = family
    return config, center_config, crisp, envelopes, samples


def spatial_trajectory_envelope(
    runs: list[dict], points: int = 1200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在共同纵向位置 x 上重构 y(x) 的数值模糊包络。

    每条轨迹的 x 必须严格递增，才能唯一地把参数轨迹从
    (x(t), y(t)) 转换为 y(x)。共同定义域取所有样本均已覆盖的
    x 区间，避免对较短轨迹进行外推。
    """
    if not runs:
        raise ValueError("Spatial trajectory envelope requires at least one run")
    for run in runs:
        x = np.asarray(run["state"][:, 0], dtype=float)
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("Spatial envelope requires strictly increasing x trajectories")
    x_start = max(float(run["state"][0, 0]) for run in runs)
    x_end = min(float(run["state"][-1, 0]) for run in runs)
    if x_end <= x_start:
        raise ValueError("Fuzzy trajectories have no common longitudinal domain")
    x_grid = np.linspace(x_start, x_end, points)
    y_on_grid = np.vstack(
        [np.interp(x_grid, run["state"][:, 0], run["state"][:, 2]) for run in runs]
    )
    return x_grid, np.min(y_on_grid, axis=0), np.max(y_on_grid, axis=0)


def add_state_band(ax: plt.Axes, time: np.ndarray, outer: dict, inner: dict,
                   lower_key: str, upper_key: str, column: int, center: np.ndarray,
                   factor: float = 1.0) -> None:
    ax.fill_between(
        time, outer[lower_key][:, column] * factor, outer[upper_key][:, column] * factor,
        color=COLORS["outer"], alpha=0.32, label="α=0 模糊包络",
    )
    ax.fill_between(
        time, inner[lower_key][:, column] * factor, inner[upper_key][:, column] * factor,
        color=COLORS["inner"], alpha=0.36, label="α=0.5 模糊包络",
    )
    ax.plot(time, center * factor, color=COLORS["center"], linewidth=1.55, label="中心参数轨迹")


def make_fde_figures(config: dict, center_config: dict, crisp: dict,
                     envelopes: dict[float, dict], sample_families: dict[float, list[dict]]) -> dict:
    time = crisp["time"]
    outer = envelopes[0.0]
    inner = envelopes[0.5]
    geometry = {key: item["value"] for key, item in config["geometry"].items()}
    lane_center = float(geometry["lane_center"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), sharex=True)
    state_panels = (
        (axes[0, 0], 0, crisp["state"][:, 0], 1.0, "纵向位置 x", "位置（m）", "(a)"),
        (axes[0, 1], 1, crisp["state"][:, 1], 3.6, "纵向速度 u", "速度（km/h）", "(b)"),
        (axes[1, 0], 2, crisp["state"][:, 2], 1.0, "横向位置 y", "横向位置（m）", "(c)"),
        (axes[1, 1], 3, crisp["state"][:, 3], 1.0, "横向速度 v", "横向速度（m/s）", "(d)"),
    )
    for ax, column, center, factor, title, ylabel, panel in state_panels:
        add_state_band(ax, time, outer, inner, "state_lower", "state_upper", column, center, factor)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        decorate(ax, panel)
    axes[0, 0].plot(time, crisp["leader"][:, 0], color=COLORS["leader"], linestyle="--", linewidth=1.1, label="前车")
    axes[0, 1].plot(time, crisp["leader"][:, 1] * 3.6, color=COLORS["leader"], linestyle="--", linewidth=1.1, label="前车")
    axes[1, 0].axhline(lane_center, color=COLORS["target"], linestyle=":", linewidth=1.0, label="目标车道中心")
    axes[1, 1].axhline(0.0, color=COLORS["target"], linestyle=":", linewidth=0.9)
    axes[1, 0].set_xlabel("时间 t（s）")
    axes[1, 1].set_xlabel("时间 t（s）")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.suptitle("二维模糊微观模型的四个运动状态演化", y=1.09, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "01_fde_state_evolution")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), sharex=True)
    response_panels = (
        (axes[0, 0], "acceleration_lower", "acceleration_upper", 0, crisp["acceleration"][:, 0], "纵向加速度 $a_x$", "加速度（m/s²）", "(a)"),
        (axes[0, 1], "acceleration_lower", "acceleration_upper", 1, crisp["acceleration"][:, 1], "横向加速度 $a_y$", "加速度（m/s²）", "(b)"),
    )
    for ax, lower_key, upper_key, column, center, title, ylabel, panel in response_panels:
        add_state_band(ax, time, outer, inner, lower_key, upper_key, column, center)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        decorate(ax, panel)
    ax = axes[1, 0]
    ax.fill_between(time, outer["headway_lower"], outer["headway_upper"], color=COLORS["outer"], alpha=0.32, label="α=0 模糊包络")
    ax.fill_between(time, inner["headway_lower"], inner["headway_upper"], color=COLORS["inner"], alpha=0.36, label="α=0.5 模糊包络")
    ax.plot(time, crisp["headway"], color=COLORS["center"], linewidth=1.55, label="中心参数轨迹")
    ax.set_title("车头间距 h")
    ax.set_ylabel("间距（m）")
    decorate(ax, "(c)")
    ax = axes[1, 1]
    relative_speed = crisp["leader"][:, 1] - crisp["state"][:, 1]
    rel_low = crisp["leader"][:, 1] - outer["state_upper"][:, 1]
    rel_high = crisp["leader"][:, 1] - outer["state_lower"][:, 1]
    rel_low_inner = crisp["leader"][:, 1] - inner["state_upper"][:, 1]
    rel_high_inner = crisp["leader"][:, 1] - inner["state_lower"][:, 1]
    ax.fill_between(time, rel_low, rel_high, color=COLORS["outer"], alpha=0.32)
    ax.fill_between(time, rel_low_inner, rel_high_inner, color=COLORS["inner"], alpha=0.36)
    ax.plot(time, relative_speed, color=COLORS["center"], linewidth=1.55)
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    ax.set_title("前后车相对速度 $u_{le}-u_i$")
    ax.set_ylabel("相对速度（m/s）")
    decorate(ax, "(d)")
    axes[1, 0].set_xlabel("时间 t（s）")
    axes[1, 1].set_xlabel("时间 t（s）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.suptitle("二维模糊模型的动力响应与跟驰指标", y=1.09, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "02_fde_dynamic_response")

    state_names = ("W_x", "W_u", "W_y", "W_v")
    state_titles = ("$W_x(t)$", "$W_u(t)$", "$W_y(t)$", "$W_v(t)$")
    units = ("m", "m/s", "m", "m/s")
    widths = outer["state_upper"] - outer["state_lower"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    maxima = {}
    for index, (ax, name, title, unit, panel) in enumerate(zip(axes.flat, state_names, state_titles, units, ("(a)", "(b)", "(c)", "(d)"))):
        width = widths[:, index]
        maximum_index = int(np.argmax(width))
        maxima[name] = {"value": float(width[maximum_index]), "time_s": float(time[maximum_index]), "unit": unit}
        ax.plot(time, width, color=COLORS["center"], linewidth=1.55)
        ax.fill_between(time, 0.0, width, color=COLORS["outer"], alpha=0.30)
        ax.scatter([time[maximum_index]], [width[maximum_index]], color=COLORS["leader"], s=20, zorder=4)
        ax.annotate(
            f"最大值 {width[maximum_index]:.4g} {unit}\n t={time[maximum_index]:.2f} s",
            xy=(time[maximum_index], width[maximum_index]), xytext=(8, 8), textcoords="offset points",
            fontsize=7.8, color="#303A43",
        )
        ax.set_title(title)
        ax.set_ylabel(f"包络宽度（{unit}）")
        decorate(ax, panel)
    axes[1, 0].set_xlabel("时间 t（s）")
    axes[1, 1].set_xlabel("时间 t（s）")
    fig.suptitle("α=0 截集下四个运动状态的不确定性宽度", y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_fde_uncertainty_width")

    x_outer, y_outer_lower, y_outer_upper = spatial_trajectory_envelope(sample_families[0.0])
    x_inner, y_inner_lower, y_inner_upper = spatial_trajectory_envelope(sample_families[0.5])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for ax in axes:
        for mark in geometry["markings"]:
            ax.axhline(mark, color="#BA8B2D", linestyle="--", linewidth=0.75, alpha=0.8)
        ax.axhline(geometry["boundary_left"], color="#48545E", linewidth=1.0)
        ax.axhline(geometry["boundary_right"], color="#48545E", linewidth=1.0)
        ax.axhline(lane_center, color=COLORS["target"], linestyle=":", linewidth=1.0)
        decorate(ax, "")
    for ax in axes:
        ax.fill_between(
            x_outer, y_outer_lower, y_outer_upper,
            color=COLORS["outer"], alpha=0.34, label="α=0 空间模糊包络",
        )
        ax.fill_between(
            x_inner, y_inner_lower, y_inner_upper,
            color=COLORS["inner"], alpha=0.42, label="α=0.5 空间模糊包络",
        )
        ax.plot(crisp["state"][:, 0], crisp["state"][:, 2], color=COLORS["center"], linewidth=1.75, label="中心参数轨迹")
        ax.plot(crisp["leader"][:, 0], crisp["leader"][:, 2], color=COLORS["leader"], linestyle="--", linewidth=1.15, label="前车")
        ax.set_xlabel("纵向位置 x（m）")
        ax.set_ylabel("横向位置 y（m）")
    axes[0].set_title("完整道路几何")
    axes[0].set_ylim(geometry["boundary_left"] - 0.25, geometry["boundary_right"] + 0.25)
    axes[0].text(-0.10, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold", fontsize=10)
    axes[1].set_title("目标车道局部放大")
    axes[1].set_ylim(float(y_outer_lower.min()) - 0.12, float(y_outer_upper.max()) + 0.12)
    axes[1].text(-0.10, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold", fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    fig.suptitle("二维模糊轨迹的空间包络", y=1.10, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "04_fde_xy_trajectory_envelope")

    nesting = nesting_violations(envelopes)
    return {"maxima": maxima, "nesting": nesting}


def make_ou_validation_figure() -> dict:
    config = json.loads((REPO / "configs/two_dimensional_crisp_diagnostic.json").read_text(encoding="utf-8"))
    stochastic = stochastic_parameters_from_config(config)
    base = stochastic.deterministic
    scenario = {key: item["value"] for key, item in config["scenario"].items()}
    numerics = {key: item["value"] for key, item in config["numerics"].items()}
    leader = make_leader(config, base, accelerating=True)
    center = config["geometry"]["lane_center"]["value"]
    initial = np.array([0.0, scenario["initial_vx"], 0.0, center + 0.3, 0.0, 0.0])

    def drift(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_drift(t, state, leader(t), stochastic)

    def diffusion(t: float, state: np.ndarray) -> np.ndarray:
        return two_dimensional_stochastic_diffusion(t, state, stochastic)

    paths = []
    for index in range(180):
        times, states = solve_sde_euler_maruyama(
            drift, diffusion, initial, scenario["following_duration"], numerics["dt"],
            int(numerics["random_seed"]) + index,
        )
        paths.append(states)
    ensemble = np.stack(paths)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    noise = stochastic.noise
    specs = (
        (2, noise.mu_lon, noise.sigma_lon / sqrt(2 * noise.theta_lon), "纵向扰动状态 $z^{lon}$", "(a)"),
        (5, noise.mu_lat, noise.sigma_lat / sqrt(2 * noise.theta_lat), "横向扰动状态 $z^{lat}$", "(b)"),
    )
    for ax, (column, mu, stationary_sd, title, panel) in zip(axes[0], specs):
        low, med, high = np.quantile(ensemble[:, :, column], (0.05, 0.50, 0.95), axis=0)
        ax.fill_between(times, low, high, color=COLORS["outer"], alpha=0.35, label="5%–95% 概率区间")
        ax.plot(times, med, color=COLORS["leader"], linewidth=1.35, label="样本中位数")
        ax.axhline(mu, color=COLORS["gray"], linestyle="--", linewidth=0.85, label="长期均值")
        ax.set_title(title)
        ax.set_xlabel("时间 t（s）")
        ax.set_ylabel("加速度扰动（m/s²）")
        decorate(ax, panel)
    for ax, (column, mu, stationary_sd, title, panel) in zip(axes[1], specs):
        values = ensemble[:, -1, column]
        grid = np.linspace(min(values.min(), mu - 3.5 * stationary_sd), max(values.max(), mu + 3.5 * stationary_sd), 300)
        density = np.exp(-0.5 * ((grid - mu) / stationary_sd) ** 2) / (stationary_sd * sqrt(2 * pi))
        ax.hist(values, bins=16, density=True, color=COLORS["outer"], alpha=0.58, edgecolor="white", label="数值样本")
        ax.plot(grid, density, color=COLORS["leader"], linewidth=1.45, label="理论稳态密度")
        ax.set_title(title.replace("状态", "末时刻分布"))
        ax.set_xlabel("加速度扰动（m/s²）")
        ax.set_ylabel("概率密度")
        decorate(ax, "(c)" if column == 2 else "(d)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.suptitle("随机扰动的均值回归与有限方差", y=1.08, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "05_ou_component_validation")
    return {
        "paths": len(paths),
        "all_finite": bool(np.all(np.isfinite(ensemble))),
        "z_lon_final_sample_mean": float(np.mean(ensemble[:, -1, 2])),
        "z_lon_theoretical_mean": noise.mu_lon,
        "z_lat_final_sample_mean": float(np.mean(ensemble[:, -1, 5])),
        "z_lat_theoretical_mean": noise.mu_lat,
    }


def main() -> None:
    configure_style()
    config, center_config, crisp, envelopes, sample_families = build_fuzzy_data()
    fuzzy_summary = make_fde_figures(config, center_config, crisp, envelopes, sample_families)
    ou_summary = make_ou_validation_figure()
    summary = {
        "primary_model": "two-dimensional fuzzy differential-equation model",
        "fuzzy_parameters": ["T", "zeta"],
        "ou_role": "auxiliary stochastic component validation only",
        "fuzzy_summary": fuzzy_summary,
        "ou_summary": ou_summary,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "simulation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
