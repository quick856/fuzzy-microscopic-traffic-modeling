"""Minimal state-fuzzy IDM demonstration package."""

from .fuzzy import TriangularFuzzyNumber, trajectory_envelope
from .model import (
    IDMParameters,
    ScenarioParameters,
    equilibrium_gap,
    leader_state,
    scenario_duration,
    simulate_follower,
    simulate_platoon,
)
from .solver import solve_ode

__all__ = [
    "TriangularFuzzyNumber",
    "trajectory_envelope",
    "IDMParameters",
    "ScenarioParameters",
    "leader_state",
    "equilibrium_gap",
    "scenario_duration",
    "simulate_follower",
    "simulate_platoon",
    "solve_ode",
]
