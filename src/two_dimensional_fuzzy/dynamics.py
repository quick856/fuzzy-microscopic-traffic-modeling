"""四维确定性 RHS；纵横向状态均采用 SI 单位。"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lateral import lateral_components
from .longitudinal import idm_acceleration
from .parameters import IDMParameters, LateralParameters


@dataclass(frozen=True)
class ModelParameters:
    """IDM 与横向参数；横向幅值由论文方程中的 eta 系数给出。"""

    idm: IDMParameters
    lateral: LateralParameters


def parameters_from_config(config: dict[str, Any]) -> ModelParameters:
    """读取有来源记录的配置；不提供隐式数值默认值。"""
    idm = IDMParameters(**{k: entry["value"] for k, entry in config["idm"].items()})
    geometry = {k: entry["value"] for k, entry in config["geometry"].items()}
    lateral = {k: entry["value"] for k, entry in config["lateral"].items()}
    lp = LateralParameters(
        lane_width=geometry["lane_width"],
        boundary_left=geometry["boundary_left"],
        boundary_right=geometry["boundary_right"],
        markings=tuple(geometry["markings"]),
        lane_center=geometry["lane_center"],
        gamma=lateral["gamma"],
        zeta=lateral["zeta"],
        epsilon=lateral["epsilon"],
        eta_middle=lateral["eta_middle"],
        eta_boundary=lateral["eta_boundary"],
        eta_mark=lateral["eta_mark"],
    )
    return ModelParameters(idm, lp)


def lateral_acceleration_components(y: float, vy: float, params: ModelParameters) -> np.ndarray:
    """返回 [边界、左右标线合力、中心线] 三个 SI 加速度贡献。"""
    return np.asarray(lateral_components(y, vy, params.lateral), dtype=float)


def two_dimensional_rhs(
    t: float, state: np.ndarray, leader_state: np.ndarray, params: ModelParameters
) -> np.ndarray:
    """返回 [vx,ax,vy,ay]；调用者在每个积分子步传入该时刻的前车状态。"""
    state = np.asarray(state, dtype=float)
    leader_state = np.asarray(leader_state, dtype=float)
    if state.shape != (4,) or leader_state.shape != (4,):
        raise ValueError("Both states must have shape (4,) = [x,vx,y,vy]")
    if not np.isfinite(t) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(leader_state)):
        raise ValueError("Non-finite time or state")
    x, vx, y, vy = state
    # paper formulation：位置差，不减车长，不裁剪。
    ax = idm_acceleration(vx, leader_state[1], leader_state[0] - x, params.idm)
    ay = float(np.sum(lateral_acceleration_components(y, vy, params)))
    return np.array([vx, ax, vy, ay], dtype=float)
