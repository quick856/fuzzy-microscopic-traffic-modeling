"""严格保留 Eq. (2) 形式的确定性 IDM，输入输出采用 SI 单位。"""

from math import sqrt

from .parameters import IDMParameters, _finite


def _speed(name: str, value: float) -> None:
    """速度按非负物理域校验；非法速度报错而非裁剪。"""
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} 不得小于零")


def idm_acceleration(
    vx: float, leader_vx: float, headway: float, params: IDMParameters
) -> float:
    """返回纵向加速度（m/s²），速度为 m/s，位置差 headway 为 m。

    headway 是 leader_x - follower_x，不减车长。严格采用指数 4，
    S* = S0 + vx*T + vx*(vx-leader_vx)/(2*sqrt(a_idm*b_idm))。
    不裁剪速度、S*、间距或加速度；负 S* 仍直接参与平方。
    """
    _speed("vx", vx)
    _speed("leader_vx", leader_vx)
    _finite("headway", headway)
    if headway <= 0:
        raise ValueError("headway 必须大于零，且不得通过距离裁剪规避该条件")
    desired_spacing = (
        params.minimum_spacing
        + vx * params.time_headway
        + vx * (vx - leader_vx) / (2 * sqrt(params.a_idm * params.b_idm))
    )
    return float(
        params.a_max_x
        * (1 - (vx / params.desired_speed) ** 4 - (desired_spacing / headway) ** 2)
    )


def equilibrium_headway(vx: float, params: IDMParameters) -> float:
    """返回同速跟驰时使 IDM 加速度为零的有限正位置差（m）。

    仅在 0 <= vx < desired_speed 且 S0 + vx*T > 0 时存在这里采用的
    正间距解；超过该域直接报错，不返回无穷或非物理间距。
    """
    _speed("vx", vx)
    if vx >= params.desired_speed:
        raise ValueError("有限正平衡间距要求 vx < desired_speed")
    numerator = params.minimum_spacing + vx * params.time_headway
    if numerator <= 0:
        raise ValueError("有限正平衡间距要求 minimum_spacing + vx*time_headway > 0")
    return float(numerator / sqrt(1 - (vx / params.desired_speed) ** 4))
