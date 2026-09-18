"""Static plots for the literature-informed crisp diagnostic experiment.

Plotting consumes saved trajectory samples only: it does not integrate, smooth,
clip, or modify them. SI labels use explicit eta-weighted acceleration terms;
the representative parameter set is not a calibrated driver model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


_STYLE: dict[str, Any] = {
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.65,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.dpi": 180,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
_CASE_STYLE = {
    "center": ("#4C5968", "--", "中心初始位置"),
    "left": ("#2477AF", "-", "左偏初始位置"),
    "right": ("#CB6043", "-", "右偏初始位置"),
}
_FOLLOWER = "#2477AF"
_LEADER = "#BC613F"


def _value(config: dict[str, Any], group: str, name: str) -> Any:
    """Read a parameter from the unflattened, provenance-bearing JSON."""
    return config[group][name]["value"]


def _read_run(run: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Validate sample dimensions without replacing or mutating source data.

    Nonfinite state/force samples are deliberately not removed: a failed
    diagnostic can still be plotted, with missing segments visible as gaps.
    """
    arrays = {
        name: np.asarray(run[name], dtype=float)
        for name in (
            "time", "state", "leader", "acceleration",
            "lateral_components", "headway",
        )
    }
    time = arrays["time"]
    if time.ndim != 1 or time.size == 0:
        raise ValueError("run['time'] must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("run['time'] must contain finite, strictly increasing times")
    expected = {
        "state": (time.size, 4),
        "leader": (time.size, 4),
        "acceleration": (time.size, 2),
        "lateral_components": (time.size, 3),
        "headway": (time.size,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"run[{name!r}] must have shape {shape}")
    return arrays


def _decorate(fig: Figure, title: str, config: dict[str, Any]) -> None:
    """仅添加主标题；图例使用独立留白，不覆盖数据曲线。"""
    del config
    fig.suptitle(title, y=0.975, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0.015, 0.025, 0.99, 0.86))


def _figure_legend(fig: Figure, ax: Axes, ncol: int) -> None:
    """把图例放在坐标区上方的专用留白中，避免遮挡曲线。"""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
            ncol=ncol, fontsize=8, frameon=False,
        )


def _save(fig: Figure, output_dir: Path, filename: str) -> str:
    """Save exactly under the caller's output directory; always close the figure."""
    path = output_dir / filename
    try:
        fig.savefig(path, facecolor="white")
    finally:
        plt.close(fig)
    return str(path.resolve())


def _time_axis(ax: Axes, time: NDArray[np.float64]) -> None:
    """Apply time limits without modifying the supplied samples."""
    ax.set_xlabel("时间 t（s）")
    if time.size > 1:
        ax.set_xlim(float(time[0]), float(time[-1]))


def plot_recovery(
    runs: dict[str, dict[str, Any]], output_dir: Path, config: dict[str, Any]
) -> list[str]:
    """Plot center/left/right recovery and right-case lateral force components.

    Parameters follow the raw diagnostic run contract: ``state`` is
    ``[x, vx, y, vy]``; ``acceleration`` is ``[ax, ay]``; the three
    ``lateral_components`` are boundary, marking, and middle-line acceleration
    contributions are eta-weighted accelerations in physical units.

    Returns absolute paths to ``diagnostic_recovery.png`` and
    ``diagnostic_force_components.png``. No simulation is run here.
    """
    # 先验证所有数据，避免由于后续缺字段而只生成部分图。
    samples = {name: _read_run(runs[name]) for name in _CASE_STYLE}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        try:
            for name, data in samples.items():
                color, style, label = _CASE_STYLE[name]
                for ax, values in zip(
                    axes,
                    (data["state"][:, 2], data["state"][:, 3], data["acceleration"][:, 1]),
                ):
                    ax.plot(data["time"], values, color=color, linestyle=style, label=label)
            for ax, ylabel in zip(
                axes, ("横向位置 y（m）", "横向速度 vᵧ（m/s）", "横向加速度 aᵧ（m/s²）")
            ):
                ax.set_ylabel(ylabel)
                ax.axhline(0, color="#87909A", linewidth=0.7, zorder=0)
            _figure_legend(fig, axes[0], ncol=3)
            _time_axis(axes[-1], samples["right"]["time"])
            _decorate(fig, "确定性横向恢复：中心与左右对称偏移", config)
            paths.append(_save(fig, target, "diagnostic_recovery.png"))
        finally:
            plt.close(fig)

        data = samples["right"]
        time = data["time"]
        tail = float(_value(config, "diagnostic_gate", "tail_window"))
        # 尾段仅缩放坐标轴；不重采样、不平滑、不替换切换处加速度。
        tail_start = max(float(time[0]), float(time[-1]) - tail)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.3), height_ratios=(1.2, 1))
        try:
            component_styles = (
                ("道路边界作用", "#647480", "--"),
                ("左右标线有向合力", "#CB6043", "-"),
                ("车道中心线作用（保留 vᵧ 因子）", "#2C8C7B", "-."),
            )
            for ax in axes:
                for column, (label, color, style) in enumerate(component_styles):
                    ax.plot(time, data["lateral_components"][:, column], label=label, color=color, linestyle=style)
                ax.axhline(0, color="#87909A", linewidth=0.7, zorder=0)
                ax.set_ylabel("对 aᵧ 的贡献（m/s²）")
                _time_axis(ax, time)
            axes[0].set_title("完整仿真时段")
            _figure_legend(fig, axes[0], ncol=3)
            axes[1].set_title(f"尾段窗口：{tail_start:g}–{time[-1]:g} s")
            if tail_start < time[-1]:
                axes[1].set_xlim(tail_start, float(time[-1]))
            # 尾段 y 轴由对应原始样本决定，以免全时段峰值掩盖残余振荡。
            tail_values = data["lateral_components"][time >= tail_start]
            finite = tail_values[np.isfinite(tail_values)]
            if finite.size:
                lower, upper = float(finite.min()), float(finite.max())
                margin = 0.08 * (upper - lower) if upper > lower else max(abs(lower) * 0.1, 1e-3)
                axes[1].set_ylim(lower - margin, upper + margin)
            _decorate(fig, "右偏恢复场景：横向加速度分项", config)
            paths.append(_save(fig, target, "diagnostic_force_components.png"))
        finally:
            plt.close(fig)
    return paths


def plot_following(
    run: dict[str, Any], output_dir: Path, config: dict[str, Any]
) -> list[str]:
    """Create the seven prescribed figures for an already-computed following run.

    ``headway`` is the saved position difference, not a reconstructed bumper
    gap. Speed is displayed in km/h; the other dimensional quantities retain
    the supplied SI values under the explicit reference-scale assumption.
    Returns absolute PNG paths and never changes source arrays or config.
    """
    data = _read_run(run)
    time, state, leader = data["time"], data["state"], data["leader"]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    left = float(_value(config, "geometry", "boundary_left"))
    right = float(_value(config, "geometry", "boundary_right"))
    center = float(_value(config, "geometry", "lane_center"))
    marks = np.asarray(_value(config, "geometry", "markings"), dtype=float)
    lane_width = float(_value(config, "geometry", "lane_width"))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.4))
        try:
            ax.axhspan(left, right, color="#F3F5F7", zorder=0)
            for index, boundary in enumerate((left, right)):
                ax.axhline(boundary, color="#414B56", linewidth=1.3, label="道路边界" if index == 0 else None)
            for index, mark in enumerate(marks):
                ax.axhline(mark, color="#B4943C", linestyle="--", linewidth=1, label="车道标线" if index == 0 else None)
            ax.axhline(center, color="#5D998A", linestyle=":", linewidth=1, label="目标车道中心线")
            ax.plot(leader[:, 0], leader[:, 2], color=_LEADER, linestyle="--", label="前车", zorder=3)
            ax.plot(state[:, 0], state[:, 2], color=_FOLLOWER, label="跟驰车", zorder=4)
            ax.set_xlabel("纵向位置 x（m）")
            ax.set_ylabel("横向位置 y（m，向右为正）")
            ax.set_ylim(left - 0.35, right + 0.35)
            _figure_legend(fig, ax, ncol=3)
            _decorate(fig, "确定性二维跟驰轨迹", config)
            paths.append(_save(fig, target, "01_crisp_xy_trajectory.png"))
        finally:
            plt.close(fig)

        specifications = (
            ("02_crisp_longitudinal_speed.png", "纵向速度", "速度（km/h）", state[:, 1] * 3.6),
            ("03_crisp_headway.png", "前后车位置差", "车头间距 x_前车 − x_跟驰车（m）", data["headway"]),
            ("04_crisp_lateral_position.png", "目标车道内的横向位置", "横向位置 y（m）", state[:, 2]),
            ("05_crisp_lateral_velocity.png", "横向速度", "横向速度 vᵧ（m/s）", state[:, 3]),
            ("06_crisp_lateral_acceleration.png", "横向加速度", "横向加速度 aᵧ（m/s²）", data["acceleration"][:, 1]),
            ("11_crisp_longitudinal_acceleration.png", "纵向加速度", "纵向加速度 aₓ（m/s²）", data["acceleration"][:, 0]),
        )
        for filename, title, ylabel, values in specifications:
            fig, ax = plt.subplots(figsize=(10, 5.4))
            try:
                ax.plot(time, values, color=_FOLLOWER, label="跟驰车")
                if filename.startswith("02_"):
                    ax.plot(time, leader[:, 1] * 3.6, color=_LEADER, linestyle="--", label="前车")
                    _figure_legend(fig, ax, ncol=2)
                elif filename.startswith("04_"):
                    # 目标车道边界优先取中心两侧最近的实际标线。
                    lower_marks, upper_marks = marks[marks < center], marks[marks > center]
                    lane_left = float(lower_marks.max()) if lower_marks.size else center - lane_width / 2
                    lane_right = float(upper_marks.min()) if upper_marks.size else center + lane_width / 2
                    ax.axhline(center, color="#5D998A", linestyle=":", label="目标车道中心线")
                    ax.axhline(lane_left, color="#B4943C", linestyle="--", label="目标车道标线")
                    ax.axhline(lane_right, color="#B4943C", linestyle="--")
                    _figure_legend(fig, ax, ncol=3)
                elif filename.startswith("06_"):
                    ax.axhline(0, color="#87909A", linewidth=0.7, zorder=0)
                elif filename.startswith(("05_", "11_")):
                    ax.axhline(0, color="#87909A", linewidth=0.7, zorder=0)
                ax.set_ylabel(ylabel)
                _time_axis(ax, time)
                _decorate(fig, title, config)
                paths.append(_save(fig, target, filename))
            finally:
                plt.close(fig)
    return paths


def plot_fuzzy_results(
    crisp_run: dict[str, Any],
    envelopes: dict[float, dict[str, NDArray[np.float64]]],
    alpha0_samples: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    """绘制 T̃、ζ̃ 参数网格产生的二维模糊包络和样本轨迹。"""
    crisp = _read_run(crisp_run)
    if 0.0 not in envelopes or 0.5 not in envelopes:
        raise ValueError("绘图需要 alpha=0 和 alpha=0.5 包络")
    zero, half = envelopes[0.0], envelopes[0.5]
    time = crisp["time"]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def band_plot(
        filename: str, title: str, ylabel: str, center: np.ndarray,
        lower0: np.ndarray, upper0: np.ndarray,
        lower05: np.ndarray, upper05: np.ndarray,
        reference: tuple[np.ndarray, str] | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 5.4))
        try:
            ax.fill_between(time, lower0, upper0, color="#8BB7D9", alpha=0.28, label="α=0 模糊带")
            ax.fill_between(time, lower05, upper05, color="#377EAE", alpha=0.36, label="α=0.5 模糊带")
            ax.plot(time, center, color="#174F78", linewidth=1.8, label="中心参数轨迹")
            if reference is not None:
                ax.plot(time, reference[0], color="#C06442", linestyle="--", label=reference[1])
            ax.set_ylabel(ylabel)
            _time_axis(ax, time)
            _figure_legend(fig, ax, ncol=4)
            _decorate(fig, title, config)
            paths.append(_save(fig, target, filename))
        finally:
            plt.close(fig)

    with plt.rc_context(_STYLE):
        band_plot(
            "07_fuzzy_longitudinal_speed.png", "模糊纵向速度包络", "速度（km/h）",
            crisp["state"][:, 1] * 3.6,
            zero["state_lower"][:, 1] * 3.6, zero["state_upper"][:, 1] * 3.6,
            half["state_lower"][:, 1] * 3.6, half["state_upper"][:, 1] * 3.6,
            (crisp["leader"][:, 1] * 3.6, "前车"),
        )

        band_plot(
            "08_fuzzy_lateral_position.png", "模糊横向位置包络", "横向位置 y（m）",
            crisp["state"][:, 2],
            zero["state_lower"][:, 2], zero["state_upper"][:, 2],
            half["state_lower"][:, 2], half["state_upper"][:, 2],
            (np.full_like(time, float(_value(config, "geometry", "lane_center"))), "目标车道中心线"),
        )

        left = float(_value(config, "geometry", "boundary_left"))
        right = float(_value(config, "geometry", "boundary_right"))
        center = float(_value(config, "geometry", "lane_center"))
        marks = np.asarray(_value(config, "geometry", "markings"), dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5.4))
        try:
            ax.axhspan(left, right, color="#F3F5F7", zorder=0)
            for index, boundary in enumerate((left, right)):
                ax.axhline(boundary, color="#414B56", linewidth=1.2, label="道路边界" if index == 0 else None)
            for index, mark in enumerate(marks):
                ax.axhline(mark, color="#B4943C", linestyle="--", linewidth=1, label="车道标线" if index == 0 else None)
            ax.axhline(center, color="#5D998A", linestyle=":", linewidth=1, label="目标车道中心线")
            for index, sample in enumerate(alpha0_samples):
                state = np.asarray(sample["state"], dtype=float)
                ax.plot(state[:, 0], state[:, 2], color="#629BC1", alpha=0.22, linewidth=0.9,
                        label="α=0 参数样本轨迹" if index == 0 else None)
            ax.plot(crisp["state"][:, 0], crisp["state"][:, 2], color="#174F78", linewidth=2,
                    label="中心参数轨迹")
            ax.plot(crisp["leader"][:, 0], crisp["leader"][:, 2], color="#C06442", linestyle="--",
                    label="前车")
            ax.set_xlabel("纵向位置 x（m）")
            ax.set_ylabel("横向位置 y（m，向右为正）")
            ax.set_ylim(left - 0.35, right + 0.35)
            _figure_legend(fig, ax, ncol=4)
            _decorate(fig, "α=0 参数空间产生的二维轨迹族", config)
            paths.append(_save(fig, target, "09_fuzzy_xy_trajectories.png"))
        finally:
            plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7.3), sharex=True)
        try:
            width_vx = zero["state_upper"][:, 1] - zero["state_lower"][:, 1]
            width_y = zero["state_upper"][:, 2] - zero["state_lower"][:, 2]
            axes[0].plot(time, width_vx, color="#174F78")
            axes[1].plot(time, width_y, color="#C06442")
            axes[0].set_ylabel("纵向速度宽度 Wᵤ（m/s）")
            axes[1].set_ylabel("横向位置宽度 Wᵧ（m）")
            _time_axis(axes[1], time)
            _decorate(fig, "α=0 模糊不确定性宽度", config)
            paths.append(_save(fig, target, "10_fuzzy_width.png"))
        finally:
            plt.close(fig)

        band_plot(
            "12_fuzzy_longitudinal_acceleration.png", "模糊纵向加速度包络", "纵向加速度 aₓ（m/s²）",
            crisp["acceleration"][:, 0],
            zero["acceleration_lower"][:, 0], zero["acceleration_upper"][:, 0],
            half["acceleration_lower"][:, 0], half["acceleration_upper"][:, 0],
        )
        band_plot(
            "13_fuzzy_lateral_acceleration.png", "模糊横向加速度包络", "横向加速度 aᵧ（m/s²）",
            crisp["acceleration"][:, 1],
            zero["acceleration_lower"][:, 1], zero["acceleration_upper"][:, 1],
            half["acceleration_lower"][:, 1], half["acceleration_upper"][:, 1],
        )

        fig, ax = plt.subplots(figsize=(10, 5.4))
        try:
            components = crisp["lateral_components"]
            ax.plot(time, components[:, 0], color="#647480", linestyle="--", label="道路边界作用")
            ax.plot(time, components[:, 1], color="#C06442", label="左右车道标线合力")
            ax.plot(time, components[:, 2], color="#2C8C7B", linestyle="-.", label="车道中心线作用")
            ax.plot(time, np.sum(components, axis=1), color="#174F78", linewidth=1.8,
                    label="横向总加速度")
            ax.axhline(0, color="#87909A", linewidth=0.7, zorder=0)
            ax.set_ylabel("对横向加速度的贡献（m/s²）")
            _time_axis(ax, time)
            _figure_legend(fig, ax, ncol=4)
            _decorate(fig, "中心参数轨迹的横向作用分解", config)
            paths.append(_save(fig, target, "14_fuzzy_lateral_force_components.png"))
        finally:
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
        try:
            component_specs = (
                (0, "道路边界作用", "#647480"),
                (1, "左右车道标线合力", "#C06442"),
                (2, "车道中心线作用", "#2C8C7B"),
                (None, "横向总加速度", "#174F78"),
            )
            for flat_index, (column, subtitle, color) in enumerate(component_specs):
                ax = axes.flat[flat_index]
                if column is None:
                    center_values = np.sum(crisp["lateral_components"], axis=1)
                    lower0 = zero["acceleration_lower"][:, 1]
                    upper0 = zero["acceleration_upper"][:, 1]
                    lower05 = half["acceleration_lower"][:, 1]
                    upper05 = half["acceleration_upper"][:, 1]
                else:
                    center_values = crisp["lateral_components"][:, column]
                    lower0 = zero["lateral_components_lower"][:, column]
                    upper0 = zero["lateral_components_upper"][:, column]
                    lower05 = half["lateral_components_lower"][:, column]
                    upper05 = half["lateral_components_upper"][:, column]
                ax.fill_between(
                    time, lower0, upper0, color="#8BB7D9", alpha=0.28,
                    label="α=0 模糊包络" if flat_index == 0 else None,
                )
                ax.fill_between(
                    time, lower05, upper05, color="#377EAE", alpha=0.36,
                    label="α=0.5 模糊包络" if flat_index == 0 else None,
                )
                ax.plot(
                    time, center_values, color=color, linewidth=1.7,
                    label="中心参数曲线" if flat_index == 0 else None,
                )
                ax.axhline(0, color="#87909A", linewidth=0.7, zorder=0)
                ax.set_title(subtitle)
                ax.set_ylabel("加速度贡献（m/s²）")
                ax.set_xlim(float(time[0]), float(time[-1]))
                if flat_index >= 2:
                    ax.set_xlabel("时间 t（s）")
            _figure_legend(fig, axes.flat[0], ncol=3)
            _decorate(fig, "横向作用分项的模糊包络", config)
            paths.append(_save(fig, target, "15_fuzzy_lateral_force_component_envelopes.png"))
        finally:
            plt.close(fig)
    return paths
