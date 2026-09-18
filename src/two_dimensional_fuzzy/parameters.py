"""确定性模型参数；所有值必须由调用方显式提供。"""

from dataclasses import dataclass
from math import isfinite


def _finite(name: str, value: float) -> None:
    """拒绝非有限数与布尔值，不用默认值替换非法输入。"""
    try:
        valid = not isinstance(value, bool) and isfinite(value)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} 必须是有限实数，收到 {value!r}")


@dataclass(frozen=True)
class IDMParameters:
    """SI 制 IDM 参数；a_max_x 与根号内的 a_idm 保持独立。"""

    a_max_x: float
    a_idm: float
    b_idm: float
    desired_speed: float
    minimum_spacing: float
    time_headway: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _finite(name, getattr(self, name))
        for name in ("a_max_x", "a_idm", "b_idm", "desired_speed"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于零")
        for name in ("minimum_spacing", "time_headway"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不得小于零")


@dataclass(frozen=True)
class LateralParameters:
    """横向确定性模型参数，位置和速度均采用 SI 单位。

    横向坐标向右增加，boundary_left < boundary_right。所有位置、
    lane_width 以 m 计，gamma、zeta 以 1/m 计。三个 eta 是论文
    完整横向方程中的幅值调节系数，用于把形状函数转换为加速度贡献。
    markings 是不可变、非空、无重复的内部标线坐标元组，顺序不限。
    """

    lane_width: float
    boundary_left: float
    boundary_right: float
    markings: tuple[float, ...]
    lane_center: float
    gamma: float
    zeta: float
    epsilon: float
    eta_middle: float
    eta_boundary: float
    eta_mark: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "markings":
                _finite(name, getattr(self, name))
        for name in ("lane_width", "gamma", "zeta"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于零")
        for name in ("eta_middle", "eta_boundary", "eta_mark"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不得小于零")
        if self.boundary_left >= self.boundary_right:
            raise ValueError("boundary_left 必须小于 boundary_right，坐标向右增加")
        if self.lane_width > self.boundary_right - self.boundary_left:
            raise ValueError("lane_width 不得超过道路总宽度")
        if not self.boundary_left < self.lane_center < self.boundary_right:
            raise ValueError("lane_center 必须严格位于道路内部")
        if not 0 <= self.epsilon <= 10:
            raise ValueError("epsilon 必须位于 [0, 10]")
        if not isinstance(self.markings, tuple) or not self.markings:
            raise ValueError("markings 必须是非空元组")
        for index, marking in enumerate(self.markings):
            _finite(f"markings[{index}]", marking)
            if not self.boundary_left < marking < self.boundary_right:
                raise ValueError("所有 markings 必须严格位于道路内部")
        if len(set(self.markings)) != len(self.markings):
            raise ValueError("markings 不得包含重复坐标")
