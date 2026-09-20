"""Fuzzy-number utilities for alpha-cut trajectory reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TriangularFuzzyNumber:
    """Triangular fuzzy number ``(left, mode, right)``."""

    left: float
    mode: float
    right: float

    def __post_init__(self) -> None:
        if not self.left <= self.mode <= self.right:
            raise ValueError("Require left <= mode <= right.")

    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        """Return the closed alpha-cut interval."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        low = self.left + alpha * (self.mode - self.left)
        high = self.right - alpha * (self.right - self.mode)
        return float(low), float(high)


def interval_grid(bounds: Tuple[float, float], count: int) -> np.ndarray:
    """Return a grid over an interval, avoiding duplicate modal samples."""
    low, high = bounds
    if np.isclose(low, high, atol=0.0, rtol=0.0):
        return np.array([low], dtype=float)
    if count < 2:
        raise ValueError("count must be at least 2 for a non-degenerate interval.")
    return np.linspace(low, high, count, dtype=float)


def alpha_input_grid(
    gap: TriangularFuzzyNumber,
    speed: TriangularFuzzyNumber,
    alpha: float,
    grid_n: int,
) -> Iterable[Tuple[float, float]]:
    """Generate the Cartesian alpha-cut grid for initial gap and speed."""
    gap_values = interval_grid(gap.alpha_cut(alpha), grid_n)
    speed_values = interval_grid(speed.alpha_cut(alpha), grid_n)
    return product(gap_values, speed_values)


def trajectory_envelope(samples: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pointwise minimum and maximum across trajectories."""
    if not samples:
        raise ValueError("At least one trajectory is required.")
    stack = np.stack(samples, axis=0)
    if not np.all(np.isfinite(stack)):
        raise ValueError("Trajectory family contains NaN or infinity.")
    return np.min(stack, axis=0), np.max(stack, axis=0)


def nesting_violations(
    envelopes: Dict[float, Tuple[np.ndarray, np.ndarray]], tolerance: float = 1e-10
) -> Dict[str, float]:
    """Measure alpha-cut nesting violations without clipping the envelopes."""
    levels = sorted(envelopes)
    max_lower_violation = 0.0
    max_upper_violation = 0.0
    for outer_alpha, inner_alpha in zip(levels[:-1], levels[1:]):
        outer_low, outer_high = envelopes[outer_alpha]
        inner_low, inner_high = envelopes[inner_alpha]
        max_lower_violation = max(
            max_lower_violation,
            float(np.max(outer_low - inner_low - tolerance)),
        )
        max_upper_violation = max(
            max_upper_violation,
            float(np.max(inner_high - outer_high - tolerance)),
        )
    return {
        "lower": max(0.0, max_lower_violation),
        "upper": max(0.0, max_upper_violation),
    }
