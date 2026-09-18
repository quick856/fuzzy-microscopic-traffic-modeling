"""Qi 横向作用函数的确定性实现。

标量形状沿用论文公式；正负方向由统一的“y 向右增加”坐标约定补全。
左右标线同时叠加，完整加速度贡献由论文中的 eta 系数调节。
"""

from math import exp

from .parameters import LateralParameters, _finite


def _interior_position(y: float, params: LateralParameters) -> None:
    """道路边界奇异域直接报错，不裁剪位置或边界距离。"""
    _finite("y", y)
    if not params.boundary_left < y < params.boundary_right:
        raise ValueError("y 必须严格位于道路内部，不能等于或越过道路边界")


def road_boundary_force(y: float, params: LateralParameters) -> float:
    """采用最近道路边界，作用方向向内；左右等距时约定取零。

    最近距离 d 小于半车道宽时幅值为 (2*d/lane_width)^(-epsilon)-1。
    d >= lane_width/2 时取零是明确的分段实现约定；不向奇异点外推。
    """
    _interior_position(y, params)
    left_distance = y - params.boundary_left
    right_distance = params.boundary_right - y
    if left_distance == right_distance:
        return 0.0
    distance = min(left_distance, right_distance)
    if distance >= params.lane_width / 2:
        return 0.0
    magnitude = (2 * distance / params.lane_width) ** (-params.epsilon) - 1
    direction = 1.0 if left_distance < right_distance else -1.0
    return float(direction * magnitude)


def lane_mark_force(y: float, params: LateralParameters) -> float:
    """叠加全部相邻标线的有向排斥形状 exp[-(zeta*d)^2]。

    每条标线都把车辆推离自身；位于目标车道中心时，左右对称项抵消。
    论文给出幅值并说明图示为左右标线之和，方向由仿真坐标补全。
    恰好位于某条无限薄标线上时，该条标线方向未定义，约定其贡献为零。
    """
    _interior_position(y, params)
    total = 0.0
    for marking in params.markings:
        delta = y - marking
        if delta == 0:
            continue
        direction = 1.0 if delta > 0 else -1.0
        total += direction * exp(-((params.zeta * abs(delta)) ** 2))
    return float(total)


def lane_middle_force(y: float, vy: float, params: LateralParameters) -> float:
    """保留 Eq. (8) 的横向速度前因子，以 e=y-lane_center 代入括号。

    返回 vy*[gamma²*(2e-lw)*exp(-gamma²*(e-lw/2)²)
             -gamma²*(2e+lw)*exp(-gamma²*(e+lw/2)²)]。
    不另乘方向、不加弹簧或阻尼；vy=0 时该项为零。
    """
    _interior_position(y, params)
    _finite("vy", vy)
    error = y - params.lane_center
    gamma_squared = params.gamma**2
    width = params.lane_width
    bracket = (
        gamma_squared * (2 * error - width)
        * exp(-gamma_squared * (error - width / 2) ** 2)
        - gamma_squared * (2 * error + width)
        * exp(-gamma_squared * (error + width / 2) ** 2)
    )
    return float(vy * bracket)


def lateral_components(
    y: float, vy: float, params: LateralParameters
) -> tuple[float, float, float]:
    """返回三个已乘 eta 的横向加速度贡献，单位为 m/s²。"""
    return (
        params.eta_boundary * road_boundary_force(y, params),
        params.eta_mark * lane_mark_force(y, params),
        params.eta_middle * lane_middle_force(y, vy, params),
    )
