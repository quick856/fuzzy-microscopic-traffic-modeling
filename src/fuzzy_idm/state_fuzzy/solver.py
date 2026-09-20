"""Small deterministic ODE solvers used by the alpha-cut family."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


def solve_ode(
    rhs: RHS,
    initial_state: np.ndarray,
    duration: float,
    dt: float,
    method: str = "RK4",
    nonnegative_indices: Sequence[int] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate an ODE using fixed-step Euler or classical RK4."""
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive.")
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration):
        raise ValueError("duration must be an integer multiple of dt.")

    times = np.linspace(0.0, duration, steps + 1)
    states = np.empty((steps + 1, len(initial_state)), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)

    for index in range(steps):
        t = times[index]
        y = states[index]
        if method.upper() == "EULER":
            states[index + 1] = y + dt * rhs(t, y)
        elif method.upper() == "RK4":
            k1 = rhs(t, y)
            k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = rhs(t + dt, y + dt * k3)
            states[index + 1] = y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        else:
            raise ValueError(f"Unknown method: {method}")

        # Numerical safeguard: selected state components (vehicle speeds) cannot
        # become negative. This is an implementation constraint, not an ODE term.
        for component in nonnegative_indices:
            states[index + 1, component] = max(0.0, states[index + 1, component])

        if not np.all(np.isfinite(states[index + 1])):
            raise FloatingPointError(f"Non-finite state at step {index + 1}.")

    return times, states
