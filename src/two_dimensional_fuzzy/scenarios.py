"""一个确定性外生 leader 和一个 follower；初始状态不随样本参数改变。"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .dynamics import ModelParameters, lateral_acceleration_components, two_dimensional_rhs
from .longitudinal import equilibrium_headway
from .solver import solve_ode

Leader = Callable[[float], np.ndarray]


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
