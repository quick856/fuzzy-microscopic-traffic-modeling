"""一个确定性外生 leader 和一个 follower；初始状态不随样本参数改变。"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np

from .dynamics import ModelParameters, lateral_acceleration_components, two_dimensional_rhs
from .longitudinal import equilibrium_headway
from .solver import solve_ode

Leader = Callable[[float], np.ndarray]
NeighborTrajectory = Callable[[float], np.ndarray]


def _smoothstep(value: float) -> float:
    """三次 smoothstep；用于保证相邻车横向位置和速度连续。"""
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def configure_neighbor_validation_scenario(config: dict[str, Any]) -> dict[str, Any]:
    """返回邻车验证场景专用配置，不改变传入配置。

    验证场景把目标车辆 i 放在中间车道，避免相邻车辆的排斥作用把车辆
    直接推向道路边界。目标车道左右标线随目标车道中心同步设置。
    """
    configured = deepcopy(config)
    center = float(configured["neighbor_scenario"]["target_lane_center"]["value"])
    lane_width = float(configured["geometry"]["lane_width"]["value"])
    configured["geometry"]["lane_center"]["value"] = center
    configured["geometry"]["target_lane_markings"]["value"] = [
        center - lane_width / 2.0,
        center + lane_width / 2.0,
    ]
    return configured


def make_neighbor_trajectory(config: dict[str, Any]) -> NeighborTrajectory:
    """生成专门验证场景中的相邻车 j 外生轨迹。

    j 车保持恒定纵向速度，先从相邻车道中心平滑靠近目标车道，保持一段
    时间后平滑返回。所有数值均从 ``neighbor_scenario`` 读取。
    """
    values = {key: entry["value"] for key, entry in config["neighbor_scenario"].items()}
    target_x0 = float(config["scenario"]["initial_x"]["value"])
    x0 = target_x0 + float(values["initial_longitudinal_gap"])
    speed = float(values["speed"])
    target_center = float(values["target_lane_center"])
    lane_width = float(config["geometry"]["lane_width"]["value"])
    side = str(values["side"]).lower()
    if side not in {"left", "right"}:
        raise ValueError("相邻车 side 必须为 left 或 right")
    side_sign = 1.0 if side == "right" else -1.0
    base_y = target_center + side_sign * lane_width
    approach_y = float(values["approach_toward_target"])
    if not 0.0 <= approach_y < lane_width / 2.0:
        raise ValueError("靠近距离必须非负且小于半车道宽，避免进入论文公式奇异域")
    approach_start = float(values["approach_start"])
    approach_end = float(values["approach_end"])
    depart_start = float(values["depart_start"])
    depart_end = float(values["depart_end"])
    if not 0 <= approach_start < approach_end <= depart_start < depart_end:
        raise ValueError("相邻车场景时间必须满足 0<=靠近开始<靠近结束<=返回开始<返回结束")

    def neighbor_j(t: float) -> np.ndarray:
        if not np.isfinite(t) or t < 0:
            raise ValueError("相邻车轨迹时间必须为非负有限值")
        if t < approach_start:
            y = base_y
            vy = 0.0
        elif t < approach_end:
            duration = approach_end - approach_start
            q = (t - approach_start) / duration
            y = base_y - side_sign * approach_y * _smoothstep(q)
            vy = -side_sign * approach_y * (6.0 * q * (1.0 - q)) / duration
        elif t < depart_start:
            y = base_y - side_sign * approach_y
            vy = 0.0
        elif t < depart_end:
            duration = depart_end - depart_start
            q = (t - depart_start) / duration
            y = base_y - side_sign * approach_y * (1.0 - _smoothstep(q))
            vy = side_sign * approach_y * (6.0 * q * (1.0 - q)) / duration
        else:
            y = base_y
            vy = 0.0
        return np.array([x0 + speed * t, speed, y, vy], dtype=float)

    return neighbor_j


def make_leader(config: dict[str, Any], params: ModelParameters, accelerating: bool) -> Leader:
    """生成位置/速度连续的恒速—匀加速—恒速轨迹；位置差平衡只初始化一次。"""
    c = {key: entry["value"] for key, entry in config["scenario"].items()}
    speed = c["initial_vx"]
    x0 = c["initial_x"] + equilibrium_headway(speed, params.idm)
    center = config["geometry"]["lane_center"]["value"]
    start, end = c["acceleration_start"], c["acceleration_end"]
    if not 0 <= start < end:
        raise ValueError("Leader acceleration times must satisfy 0 <= start < end")
    acceleration = c["leader_acceleration"] if accelerating else 0.0

    def leader(t: float) -> np.ndarray:
        if not np.isfinite(t) or t < 0:
            raise ValueError("Leader time must be finite and nonnegative")
        active_time = min(max(t - start, 0.0), end - start)
        after_time = max(t - end, 0.0)
        vx = speed + acceleration * active_time
        x = x0 + speed*t + 0.5*acceleration*active_time**2 + acceleration*(end-start)*after_time
        return np.array([x, vx, center, 0.0])

    return leader


def simulate_crisp(
    config: dict[str, Any], params: ModelParameters, leader: Leader,
    offset: float, duration: float, dt: float,
) -> dict[str, np.ndarray]:
    """求解并保存状态、外生前车、原 RHS 加速度与分项作用。"""
    c = {key: entry["value"] for key, entry in config["scenario"].items()}
    center = config["geometry"]["lane_center"]["value"]
    initial = np.array([c["initial_x"], c["initial_vx"], center+offset, c["initial_vy"]], dtype=float)

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        # 每次 RK4 子步都调用 leader(t)，不冻结前车。
        return two_dimensional_rhs(t, state, leader(t), params)

    times, states = solve_ode(rhs, initial, duration, dt, config["numerics"]["method"]["value"])
    leaders = np.array([leader(float(t)) for t in times])
    acceleration = np.array([rhs(float(t), state)[[1, 3]] for t, state in zip(times, states)])
    components = np.array([lateral_acceleration_components(state[2], state[3], params) for state in states])
    return {
        "time": times, "state": states, "leader": leaders,
        "acceleration": acceleration, "lateral_components": components,
        "headway": leaders[:, 0] - states[:, 0],
    }
