"""确定性 ODE 与 Euler-Maruyama SDE 积分器，均不裁剪状态。"""

from collections.abc import Callable
import math

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]
Diffusion = Callable[[float, np.ndarray], np.ndarray]


class IntegrationError(RuntimeError):
    """保留失败子步及状态，避免数值保护掩盖模型定义域错误。"""

    def __init__(self, t: float, stage: str, state: np.ndarray, reason: str) -> None:
        self.time = float(t)
        self.stage = stage
        self.state = np.asarray(state).tolist()
        super().__init__(f"t={t:.10g}, stage={stage}, state={self.state}: {reason}")


def solve_ode(
    rhs: RHS, initial_state: np.ndarray, duration: float, dt: float, method: str = "RK4"
) -> tuple[np.ndarray, np.ndarray]:
    """从 t=0 积分到 duration；最后一步可缩短，所有子步重新调用 RHS。"""
    if not all(math.isfinite(v) and v > 0 for v in (dt, duration)):
        raise ValueError("duration and dt must be finite and positive")
    if method not in ("Euler", "RK4"):
        raise ValueError("Only Euler and RK4 are supported")
    initial = np.array(initial_state, dtype=float, copy=True)
    if initial.ndim != 1 or not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("Initial state must be a finite nonempty vector")
    full_steps = int(math.floor(duration / dt))
    times = np.arange(full_steps + 1, dtype=float) * dt
    if math.isclose(times[-1], duration, rel_tol=1e-12, abs_tol=1e-12):
        times[-1] = duration
    else:
        times = np.append(times, duration)
    states = np.empty((len(times), initial.size), dtype=float)
    states[0] = initial

    def evaluate(t: float, value: np.ndarray, stage: str) -> np.ndarray:
        try:
            if not np.all(np.isfinite(value)):
                raise ValueError("Non-finite substage state")
            result = np.asarray(rhs(t, value), dtype=float)
            if result.shape != initial.shape or not np.all(np.isfinite(result)):
                raise ValueError("Invalid RHS shape or non-finite derivative")
            return result
        except (ValueError, OverflowError, FloatingPointError) as exc:
            raise IntegrationError(t, stage, value, str(exc)) from exc

    for index in range(len(times) - 1):
        t, value = times[index], states[index]
        step = times[index + 1] - t
        k1 = evaluate(t, value, "k1")
        if method == "Euler":
            next_value = value + step * k1
        else:
            k2 = evaluate(t + step / 2, value + step * k1 / 2, "k2")
            k3 = evaluate(t + step / 2, value + step * k2 / 2, "k3")
            k4 = evaluate(t + step, value + step * k3, "k4")
            next_value = value + step * (k1 + 2*k2 + 2*k3 + k4) / 6
        # 只验证端点定义域，不改变积分状态。
        evaluate(times[index + 1], next_value, "endpoint")
        states[index + 1] = next_value
    return times, states


def solve_sde_euler_maruyama(
    drift: RHS,
    diffusion: Diffusion,
    initial_state: np.ndarray,
    duration: float,
    dt: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用 Euler-Maruyama 求解 Itô SDE，并通过显式 seed 保证可复现。

    diffusion 必须返回 shape=(state_dim, brownian_dim) 的矩阵。每一步用
    同一时刻的漂移和扩散计算；随机增量为 sqrt(step)*N(0,I)。函数不做
    速度、位置或加速度裁剪，定义域错误会以 IntegrationError 抛出。
    """
    if not all(math.isfinite(v) and v > 0 for v in (dt, duration)):
        raise ValueError("duration and dt must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an explicit integer")
    initial = np.array(initial_state, dtype=float, copy=True)
    if initial.ndim != 1 or not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("Initial state must be a finite nonempty vector")

    full_steps = int(math.floor(duration / dt))
    times = np.arange(full_steps + 1, dtype=float) * dt
    if math.isclose(times[-1], duration, rel_tol=1e-12, abs_tol=1e-12):
        times[-1] = duration
    else:
        times = np.append(times, duration)
    states = np.empty((len(times), initial.size), dtype=float)
    states[0] = initial
    rng = np.random.default_rng(int(seed))
    brownian_dim: int | None = None

    def evaluate_drift(t: float, value: np.ndarray, stage: str) -> np.ndarray:
        try:
            if not np.all(np.isfinite(value)):
                raise ValueError("Non-finite substage state")
            result = np.asarray(drift(t, value), dtype=float)
            if result.shape != initial.shape or not np.all(np.isfinite(result)):
                raise ValueError("Invalid drift shape or non-finite derivative")
            return result
        except (ValueError, OverflowError, FloatingPointError) as exc:
            raise IntegrationError(t, stage, value, str(exc)) from exc

    def evaluate_diffusion(t: float, value: np.ndarray, stage: str) -> np.ndarray:
        try:
            result = np.asarray(diffusion(t, value), dtype=float)
            if result.ndim != 2 or result.shape[0] != initial.size or result.shape[1] == 0:
                raise ValueError("Diffusion must have shape (state_dim, brownian_dim)")
            if not np.all(np.isfinite(result)):
                raise ValueError("Non-finite diffusion")
            return result
        except (ValueError, OverflowError, FloatingPointError) as exc:
            raise IntegrationError(t, stage, value, str(exc)) from exc

    for index in range(len(times) - 1):
        t, value = times[index], states[index]
        step = times[index + 1] - t
        drift_value = evaluate_drift(t, value, "drift")
        diffusion_value = evaluate_diffusion(t, value, "diffusion")
        if brownian_dim is None:
            brownian_dim = diffusion_value.shape[1]
        elif diffusion_value.shape[1] != brownian_dim:
            raise IntegrationError(t, "diffusion", value, "Brownian dimension changed during integration")
        d_w = np.sqrt(step) * rng.standard_normal(brownian_dim)
        next_value = value + step * drift_value + diffusion_value @ d_w
        evaluate_drift(times[index + 1], next_value, "endpoint")
        states[index + 1] = next_value
    return times, states
