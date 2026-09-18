"""三角模糊参数、α 截集网格和轨迹包络工具。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TriangularFuzzyNumber:
    """三角模糊数 (left, mode, right)。"""

    left: float
    mode: float
    right: float

    def __post_init__(self) -> None:
        values = (self.left, self.mode, self.right)
        if not all(isfinite(value) for value in values):
            raise ValueError("三角模糊数参数必须为有限实数")
        if not self.left <= self.mode <= self.right:
            raise ValueError("三角模糊数必须满足 left <= mode <= right")

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        """返回 α 截集 [lower, upper]。"""
        if not isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 必须位于 [0,1]")
        lower = self.left + alpha * (self.mode - self.left)
        upper = self.right - alpha * (self.right - self.mode)
        return float(lower), float(upper)


@dataclass(frozen=True)
class FuzzyParameter:
    """把一个模糊数映射到配置文件中的 group/key。"""

    name: str
    config_group: str
    config_key: str
    number: TriangularFuzzyNumber


def fuzzy_parameters_from_config(config: dict[str, Any]) -> tuple[FuzzyParameter, ...]:
    """读取当前 T̃ 和 ζ̃；循环算法本身支持任意参数列表。"""
    triangles = config["fuzzy_next_stage"]["triangles"]
    mapping = {
        "time_headway": ("idm", "time_headway"),
        "zeta": ("lateral", "zeta"),
    }
    parameters = []
    for name in config["fuzzy_next_stage"]["selected_parameters"]:
        if name not in mapping or name not in triangles:
            raise ValueError(f"缺少模糊参数映射或三角数：{name}")
        group, key = mapping[name]
        item = triangles[name]
        parameters.append(FuzzyParameter(
            name=name,
            config_group=group,
            config_key=key,
            number=TriangularFuzzyNumber(item["left"], item["mode"], item["right"]),
        ))
    return tuple(parameters)


def parameter_grid(
    parameters: Iterable[FuzzyParameter], alpha: float, grid_n: int
) -> list[dict[str, float]]:
    """生成 Ωα 的笛卡尔网格；退化区间只生成一个点。"""
    if isinstance(grid_n, bool) or not isinstance(grid_n, int) or grid_n < 2:
        raise ValueError("grid_n 必须是至少为 2 的整数")
    parameter_list = tuple(parameters)
    axes: list[np.ndarray] = []
    for parameter in parameter_list:
        lower, upper = parameter.number.alpha_cut(alpha)
        axes.append(
            np.array([lower], dtype=float)
            if lower == upper else np.linspace(lower, upper, grid_n, dtype=float)
        )
    if not axes:
        return [{}]
    return [
        {parameter.name: float(value) for parameter, value in zip(parameter_list, values)}
        for values in product(*axes)
    ]


def apply_parameter_values(
    config: dict[str, Any], parameters: Iterable[FuzzyParameter], values: dict[str, float]
) -> None:
    """把一组确定性参数写入配置副本。"""
    for parameter in parameters:
        config[parameter.config_group][parameter.config_key]["value"] = values[parameter.name]


def trajectory_envelope(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """逐时刻对一族确定性轨迹取上下包络。"""
    if not samples:
        raise ValueError("至少需要一条确定性轨迹")
    time = np.asarray(samples[0]["time"], dtype=float)
    for sample in samples[1:]:
        if not np.array_equal(time, np.asarray(sample["time"], dtype=float)):
            raise ValueError("所有确定性轨迹必须使用相同时间网格")
    states = np.stack([sample["state"] for sample in samples])
    accelerations = np.stack([sample["acceleration"] for sample in samples])
    lateral_components = np.stack([sample["lateral_components"] for sample in samples])
    headways = np.stack([sample["headway"] for sample in samples])
    return {
        "time": time,
        "state_lower": np.min(states, axis=0),
        "state_upper": np.max(states, axis=0),
        "acceleration_lower": np.min(accelerations, axis=0),
        "acceleration_upper": np.max(accelerations, axis=0),
        "lateral_components_lower": np.min(lateral_components, axis=0),
        "lateral_components_upper": np.max(lateral_components, axis=0),
        "headway_lower": np.min(headways, axis=0),
        "headway_upper": np.max(headways, axis=0),
    }


def nesting_violations(
    envelopes: dict[float, dict[str, np.ndarray]], tolerance: float = 1e-10
) -> dict[str, float | bool]:
    """检查高 α 包络是否逐时刻包含于相邻低 α 包络。"""
    alphas = sorted(envelopes)
    worst_lower = 0.0
    worst_upper = 0.0
    for outer_alpha, inner_alpha in zip(alphas[:-1], alphas[1:]):
        outer = envelopes[outer_alpha]
        inner = envelopes[inner_alpha]
        worst_lower = max(
            worst_lower,
            float(np.max(outer["state_lower"] - inner["state_lower"])),
        )
        worst_upper = max(
            worst_upper,
            float(np.max(inner["state_upper"] - outer["state_upper"])),
        )
    worst = max(worst_lower, worst_upper, 0.0)
    return {
        "pass": bool(worst <= tolerance),
        "maximum_lower_violation": max(worst_lower, 0.0),
        "maximum_upper_violation": max(worst_upper, 0.0),
        "tolerance": tolerance,
    }
