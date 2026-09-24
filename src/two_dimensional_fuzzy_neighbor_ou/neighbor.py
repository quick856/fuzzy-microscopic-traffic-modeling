"""Qi (2025) 邻车椭圆影响域及 Appendix A 二维作用力。

论文公式按前方邻车推导。横向镜像方向由统一坐标约定补全：y 向右增加，
邻车在本车右侧时横向分量向左，邻车在左侧时横向分量向右。
"""

from math import hypot, sqrt

import numpy as np

from .parameters import LateralParameters, NeighborParameters, _finite


_DOMAIN_TOLERANCE = 1e-12


def neighbor_influence_axes(
    vehicle_i_vx: float,
    lateral: LateralParameters,
    neighbor: NeighborParameters,
) -> tuple[float, float]:
    """返回论文 Eq. (10) 的纵向半轴和 Eq. (9) 的横向半轴。"""
    _finite("vehicle_i_vx", vehicle_i_vx)
    if vehicle_i_vx < 0:
        raise ValueError("vehicle_i_vx 不得小于零")
    x_ellip = vehicle_i_vx**2 / (2.0 * neighbor.max_deceleration)
    y_ellip = lateral.lane_width
    return float(x_ellip), float(y_ellip)


def _relative_position(
    vehicle_i_state: np.ndarray, neighbor_j_state: np.ndarray
) -> tuple[float, float, float, float]:
    vehicle_i = np.asarray(vehicle_i_state, dtype=float)
    neighbor_j = np.asarray(neighbor_j_state, dtype=float)
    if vehicle_i.shape != (4,) or neighbor_j.shape != (4,):
        raise ValueError("vehicle_i_state 与 neighbor_j_state 必须为 [x,vx,y,vy]")
    if not np.all(np.isfinite(vehicle_i)) or not np.all(np.isfinite(neighbor_j)):
        raise ValueError("车辆状态必须全部为有限值")
    dx = float(neighbor_j[0] - vehicle_i[0])
    dy_signed = float(neighbor_j[2] - vehicle_i[2])
    return dx, dy_signed, abs(dy_signed), float(vehicle_i[1])


def is_neighbor_in_influence_region(
    vehicle_i_state: np.ndarray,
    neighbor_j_state: np.ndarray,
    lateral: LateralParameters,
    neighbor: NeighborParameters,
) -> bool:
    """检查前方邻车是否满足 Eq. (11) 的椭圆影响域。

    Appendix A 的力包含 1/Delta x，因此第一版只在 Delta x>0 的
    前方定义域内使用。该限制是对论文推导适用域的显式实现，不是
    对后方车辆作用的猜测性补充。
    """
    if not neighbor.enabled:
        return False
    dx, _, dy, vehicle_i_vx = _relative_position(vehicle_i_state, neighbor_j_state)
    x_ellip, y_ellip = neighbor_influence_axes(vehicle_i_vx, lateral, neighbor)
    if dx <= _DOMAIN_TOLERANCE or x_ellip <= _DOMAIN_TOLERANCE:
        return False
    normalized = (dx / x_ellip) ** 2 + (dy / y_ellip) ** 2
    return bool(normalized <= 1.0 + _DOMAIN_TOLERANCE)


def neighbor_total_force(
    vehicle_i_state: np.ndarray,
    neighbor_j_state: np.ndarray,
    lateral: LateralParameters,
    neighbor: NeighborParameters,
) -> float:
    """返回 Appendix A Eq. (30) 的有符号总作用；域外返回零。

    论文公式在 |Delta y|=l_w/2 以及 Delta x=0 处奇异。本实现不裁剪
    距离；若处于影响域且横向距离不大于半车道宽，直接报告模型定义域
    错误，避免用数值上限改变论文公式。
    """
    if not is_neighbor_in_influence_region(vehicle_i_state, neighbor_j_state, lateral, neighbor):
        return 0.0
    dx, _, dy, vehicle_i_vx = _relative_position(vehicle_i_state, neighbor_j_state)
    x_ellip, y_ellip = neighbor_influence_axes(vehicle_i_vx, lateral, neighbor)
    half_width = lateral.lane_width / 2.0
    if dy <= half_width + _DOMAIN_TOLERANCE:
        raise ValueError("邻车横向距离进入 Appendix A 在 l_w/2 处的奇异/碰撞域")
    radial_term = 1.0 - (dx / x_ellip) ** 2
    if radial_term < -_DOMAIN_TOLERANCE:
        return 0.0
    boundary_y = y_ellip * sqrt(max(radial_term, 0.0))
    if boundary_y <= half_width + _DOMAIN_TOLERANCE:
        raise ValueError("椭圆截面未留下 Appendix A 所需的有效横向间隔")
    bracket = 1.0 / (dy - half_width) - 1.0 / (boundary_y - half_width)
    # 边界上的理论值为零；仅消除浮点舍入产生的极小符号误差。
    if abs(bracket) <= _DOMAIN_TOLERANCE:
        return 0.0
    return float(-(vehicle_i_vx**2) * bracket / (2.0 * dx))


def neighbor_interaction_force(
    vehicle_i_state: np.ndarray,
    neighbor_j_state: np.ndarray,
    lateral: LateralParameters,
    neighbor: NeighborParameters,
) -> tuple[float, float]:
    """按 Appendix A Eq. (31)-(32) 返回 (F_ne_lon, F_ne_lat)。"""
    total = neighbor_total_force(vehicle_i_state, neighbor_j_state, lateral, neighbor)
    if total == 0.0:
        return 0.0, 0.0
    dx, dy_signed, _, _ = _relative_position(vehicle_i_state, neighbor_j_state)
    distance = hypot(dx, dy_signed)
    if distance <= _DOMAIN_TOLERANCE:
        raise ValueError("车辆相对距离为零，邻车作用方向未定义")
    return float(total * dx / distance), float(total * dy_signed / distance)


def neighbor_total_components(
    vehicle_i_state: np.ndarray,
    neighbor_states: tuple[np.ndarray, ...],
    lateral: LateralParameters,
    neighbor: NeighborParameters,
) -> tuple[float, float]:
    """对所有邻车求和，返回论文 Eq. (13) 中的两个邻车求和项。"""
    lon = 0.0
    lat = 0.0
    for other in neighbor_states:
        current_lon, current_lat = neighbor_interaction_force(
            vehicle_i_state, other, lateral, neighbor
        )
        lon += current_lon
        lat += current_lat
    return float(lon), float(lat)
