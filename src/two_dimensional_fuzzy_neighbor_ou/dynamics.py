"""四维确定性 RHS；纵横向状态均采用 SI 单位。"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lateral import lateral_components
from .longitudinal import idm_acceleration
from .neighbor import neighbor_total_components
from .parameters import IDMParameters, LateralParameters, NeighborParameters, OUParameters


@dataclass(frozen=True)
class ModelParameters:
    """IDM、道路横向作用与邻车作用参数。"""

    idm: IDMParameters
    lateral: LateralParameters
    neighbor: NeighborParameters


@dataclass(frozen=True)
class StochasticModelParameters:
    """论文正常型二维随机模型参数；确定性漂移中已包含邻车作用。"""

    deterministic: ModelParameters
    noise: OUParameters


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
        target_markings=tuple(geometry.get("target_lane_markings", geometry["markings"])),
        lane_center=geometry["lane_center"],
        gamma=lateral["gamma"],
        zeta=lateral["zeta"],
        epsilon=lateral["epsilon"],
        eta_middle=lateral["eta_middle"],
        eta_boundary=lateral["eta_boundary"],
        eta_mark=lateral["eta_mark"],
    )
    neighbor_values = {name: entry["value"] for name, entry in config["neighbor"].items()}
    return ModelParameters(idm, lp, NeighborParameters(**neighbor_values))


def stochastic_parameters_from_config(config: dict[str, Any]) -> StochasticModelParameters:
    """读取 Eq. (12) OU 参数；每个数值仍须在配置中显式给出来源。"""
    deterministic = parameters_from_config(config)
    noise_values = {name: entry["value"] for name, entry in config["ou_noise"].items()}
    return StochasticModelParameters(deterministic, OUParameters(**noise_values))


def lateral_acceleration_components(y: float, vy: float, params: ModelParameters) -> np.ndarray:
    """返回 [边界、左右标线合力、中心线] 三个 SI 加速度贡献。"""
    return np.asarray(lateral_components(y, vy, params.lateral), dtype=float)


def two_dimensional_rhs(
    t: float,
    state: np.ndarray,
    leader_state: np.ndarray,
    params: ModelParameters,
    neighbor_states: tuple[np.ndarray, ...] = (),
) -> np.ndarray:
    """返回 [vx,ax,vy,ay]，包含 Eq. (13) 的邻车纵、横向求和项。"""
    state = np.asarray(state, dtype=float)
    leader_state = np.asarray(leader_state, dtype=float)
    if state.shape != (4,) or leader_state.shape != (4,):
        raise ValueError("Both states must have shape (4,) = [x,vx,y,vy]")
    if not np.isfinite(t) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(leader_state)):
        raise ValueError("Non-finite time or state")
    x, vx, y, vy = state
    neighbor_lon, neighbor_lat = neighbor_total_components(
        state, neighbor_states, params.lateral, params.neighbor
    )
    # paper formulation：位置差，不减车长，不裁剪。
    ax = idm_acceleration(vx, leader_state[1], leader_state[0] - x, params.idm) + neighbor_lon
    ay = float(np.sum(lateral_acceleration_components(y, vy, params))) + neighbor_lat
    return np.array([vx, ax, vy, ay], dtype=float)


def two_dimensional_stochastic_drift(
    t: float,
    state: np.ndarray,
    leader_state: np.ndarray,
    params: StochasticModelParameters,
    neighbor_states: tuple[np.ndarray, ...] = (),
) -> np.ndarray:
    """返回论文 Eq. (13) 六维状态的漂移项，包含邻车作用。

    状态顺序严格采用 [x, u, z_lon, y, v, z_lat]。该接口同时包含
    Eq. (13) 的两个邻车求和项，z_lon、z_lat 分别进入纵横向加速度。
    """
    state = np.asarray(state, dtype=float)
    leader_state = np.asarray(leader_state, dtype=float)
    if state.shape != (6,) or leader_state.shape != (4,):
        raise ValueError(
            "state must be [x,u,z_lon,y,v,z_lat] and leader_state must be [x,u,y,v]"
        )
    if not np.isfinite(t) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(leader_state)):
        raise ValueError("Non-finite time or state")

    x, vx, z_lon, y, vy, z_lat = state
    base = params.deterministic
    noise = params.noise
    deterministic_state = np.array([x, vx, y, vy], dtype=float)
    neighbor_lon, neighbor_lat = neighbor_total_components(
        deterministic_state, neighbor_states, base.lateral, base.neighbor
    )
    ax = (
        idm_acceleration(vx, leader_state[1], leader_state[0] - x, base.idm)
        + neighbor_lon + z_lon
    )
    ay = float(np.sum(lateral_acceleration_components(y, vy, base))) + neighbor_lat + z_lat
    dz_lon = noise.theta_lon * (noise.mu_lon - z_lon)
    dz_lat = noise.theta_lat * (noise.mu_lat - z_lat)
    return np.array([vx, ax, dz_lon, vy, ay, dz_lat], dtype=float)


def two_dimensional_stochastic_diffusion(
    t: float,
    state: np.ndarray,
    params: StochasticModelParameters,
) -> np.ndarray:
    """返回六维状态的扩散矩阵。

    论文对两个方程均记作 dW，但没有给出二者相关系数。这里使用两个
    独立 Wiener 增量，分别只驱动 z_lon 与 z_lat；这是明确的数值实现
    约定，不是论文新增公式。
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(t) or not np.all(np.isfinite(state)):
        raise ValueError("state must be a finite [x,u,z_lon,y,v,z_lat] vector")
    diffusion = np.zeros((6, 2), dtype=float)
    diffusion[2, 0] = params.noise.sigma_lon
    diffusion[5, 1] = params.noise.sigma_lat
    return diffusion
