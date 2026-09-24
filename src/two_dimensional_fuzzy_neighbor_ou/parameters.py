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
    lane_width 以 m 计，gamma、zeta 以 1/m 计。三个 eta 来自
    Qi and Hu (2023) 的加权横向方程；全部取 1 时恢复 Qi (2025)
    Eq. (13) 使用的未加权横向作用和。
    markings 保存道路上全部标线，供道路几何与绘图使用；target_markings
    只保存目标车道左右两条边界标线。横向标线作用只由后者计算，避免
    把远处其他车道的标线错误加入目标车辆的横向加速度。
    """

    lane_width: float
    boundary_left: float
    boundary_right: float
    markings: tuple[float, ...]
    target_markings: tuple[float, float]
    lane_center: float
    gamma: float
    zeta: float
    epsilon: float
    eta_middle: float
    eta_boundary: float
    eta_mark: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name not in ("markings", "target_markings"):
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
        if not isinstance(self.target_markings, tuple) or len(self.target_markings) != 2:
            raise ValueError("target_markings 必须是目标车道左右两条标线组成的二元组")
        left_mark, right_mark = self.target_markings
        _finite("target_markings[0]", left_mark)
        _finite("target_markings[1]", right_mark)
        if left_mark >= right_mark:
            raise ValueError("target_markings 必须按左、右顺序给出")
        if left_mark not in self.markings or right_mark not in self.markings:
            raise ValueError("target_markings 必须同时包含在 markings 中")
        if not left_mark < self.lane_center < right_mark:
            raise ValueError("lane_center 必须位于目标车道左右标线之间")


@dataclass(frozen=True)
class NeighborParameters:
    """Qi (2025) Eq. (9)-(11) 与 Appendix A 的邻车参数。

    max_deceleration 对应论文 a_MAX^x，用于计算纵向椭圆半轴
    x_ellip=u_i^2/(2*a_MAX^x)。论文未给出本研究场景的标定值，
    因而必须由配置文件显式给出并记录来源。
    """

    enabled: bool
    max_deceleration: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled 必须为布尔值")
        _finite("max_deceleration", self.max_deceleration)
        if self.max_deceleration <= 0:
            raise ValueError("max_deceleration 必须大于零")


@dataclass(frozen=True)
class OUParameters:
    """论文 Eq. (12) 的纵、横向 Ornstein-Uhlenbeck 噪声参数。

    z_lon 与 z_lat 作为加速度扰动进入 Eq. (13)，因此 mu 与 z 的单位为
    m/s²；theta 的单位为 1/s；sigma 乘以 Wiener 增量 dW。参数类不提供
    隐式默认值，避免把仿真假设误写成论文标定参数。
    """

    theta_lon: float
    mu_lon: float
    sigma_lon: float
    theta_lat: float
    mu_lat: float
    sigma_lat: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _finite(name, getattr(self, name))
        for name in ("theta_lon", "theta_lat"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于零")
        for name in ("sigma_lon", "sigma_lat"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不得小于零")
