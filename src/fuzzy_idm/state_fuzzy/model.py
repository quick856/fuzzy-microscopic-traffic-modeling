"""Deterministic IDM used inside the state-fuzzy alpha-cut family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .solver import solve_ode


@dataclass(frozen=True)
class IDMParameters:
    """Crisp IDM parameters in SI units."""

    desired_speed: float = 30.0
    max_acceleration: float = 1.0
    comfortable_deceleration: float = 1.5
    desired_time_headway: float = 1.5
    minimum_gap: float = 2.0
    acceleration_exponent: float = 4.0
    vehicle_length: float = 5.0


@dataclass(frozen=True)
class ScenarioParameters:
    """Prescribed leader profiles in SI units."""

    accelerating_initial_speed: float = 15.0
    accelerating_target_speed: float = 20.0
    accelerating_acceleration: float = 0.5
    braking_initial_speed: float = 20.0
    braking_start: float = 10.0
    braking_deceleration: float = 1.5
    periodic_mean_speed: float = 20.0
    periodic_amplitude: float = 2.0
    periodic_omega: float = 0.2
    periodic_warmup: float = 60.0
    periodic_transient_cycles: int = 2
    periodic_analysis_cycles: int = 5

    @property
    def periodic_period(self) -> float:
        return 2.0 * np.pi / self.periodic_omega

    @property
    def periodic_analysis_start(self) -> float:
        return self.periodic_warmup + self.periodic_transient_cycles * self.periodic_period

    @property
    def periodic_end(self) -> float:
        return self.periodic_warmup + (
            self.periodic_transient_cycles + self.periodic_analysis_cycles
        ) * self.periodic_period


SCENARIOS = ("accelerating", "braking", "periodic")
DEFAULT_SCENARIO_PARAMETERS = ScenarioParameters()

# 与现有 Parameter-Fuzzy IDM 周期工况保持一致。
PERIODIC_MEAN_SPEED = DEFAULT_SCENARIO_PARAMETERS.periodic_mean_speed
PERIODIC_AMPLITUDE = DEFAULT_SCENARIO_PARAMETERS.periodic_amplitude
PERIODIC_OMEGA = DEFAULT_SCENARIO_PARAMETERS.periodic_omega
PERIODIC_WARMUP = DEFAULT_SCENARIO_PARAMETERS.periodic_warmup
PERIODIC_TRANSIENT_CYCLES = DEFAULT_SCENARIO_PARAMETERS.periodic_transient_cycles
PERIODIC_ANALYSIS_CYCLES = DEFAULT_SCENARIO_PARAMETERS.periodic_analysis_cycles
PERIODIC_PERIOD = DEFAULT_SCENARIO_PARAMETERS.periodic_period
PERIODIC_ANALYSIS_START = DEFAULT_SCENARIO_PARAMETERS.periodic_analysis_start


def leader_state(
    t: float,
    scenario: str,
    scenario_parameters: ScenarioParameters = DEFAULT_SCENARIO_PARAMETERS,
) -> Tuple[float, float, float]:
    """Return prescribed leader position, speed, and acceleration.

    The three profiles match the existing Parameter-Fuzzy IDM experiment:
    acceleration from 54 to 72 km/h at 0.5 m/s^2, braking from 72 km/h
    after 10 s at 1.5 m/s^2 until rest. The periodic scenario first keeps
    72 km/h for 60 s, then applies a 7.2 km/h sinusoid with angular frequency
    0.2 rad/s, matching the existing Parameter-Fuzzy IDM experiment.
    """
    if scenario == "accelerating":
        initial_speed = scenario_parameters.accelerating_initial_speed
        acceleration = scenario_parameters.accelerating_acceleration
        maximum_speed = scenario_parameters.accelerating_target_speed
        acceleration_duration = (maximum_speed - initial_speed) / acceleration
        active_time = min(max(t, 0.0), acceleration_duration)
        after_time = max(t - acceleration_duration, 0.0)
        speed = initial_speed + acceleration * active_time
        position = (
            initial_speed * active_time
            + 0.5 * acceleration * active_time**2
            + maximum_speed * after_time
        )
        current_acceleration = acceleration if t < acceleration_duration else 0.0
        return float(position), float(speed), float(current_acceleration)

    if scenario == "braking":
        initial_speed = scenario_parameters.braking_initial_speed
        braking_start = scenario_parameters.braking_start
        deceleration = scenario_parameters.braking_deceleration
        braking_duration = initial_speed / deceleration
        if t < braking_start:
            return float(initial_speed * t), initial_speed, 0.0
        braking_time = min(t - braking_start, braking_duration)
        position = (
            initial_speed * braking_start
            + initial_speed * braking_time
            - 0.5 * deceleration * braking_time**2
        )
        speed = max(0.0, initial_speed - deceleration * braking_time)
        current_acceleration = -deceleration if t < braking_start + braking_duration else 0.0
        return float(position), float(speed), float(current_acceleration)

    if scenario == "periodic":
        mean_speed = scenario_parameters.periodic_mean_speed
        amplitude = scenario_parameters.periodic_amplitude
        omega = scenario_parameters.periodic_omega
        warmup = scenario_parameters.periodic_warmup
        if t < warmup:
            return float(mean_speed * t), mean_speed, 0.0
        disturbance_time = t - warmup
        speed = mean_speed + amplitude * np.sin(
            omega * disturbance_time
        )
        position = (
            mean_speed * t
            + amplitude
            * (1.0 - np.cos(omega * disturbance_time))
            / omega
        )
        acceleration = (
            amplitude
            * omega
            * np.cos(omega * disturbance_time)
        )
        return float(position), float(speed), float(acceleration)

    raise ValueError(f"Unknown scenario: {scenario}")


def scenario_duration(
    scenario: str,
    scenario_parameters: ScenarioParameters = DEFAULT_SCENARIO_PARAMETERS,
) -> float:
    """Return the duration used by the existing comparison scenarios."""
    if scenario == "periodic":
        # 60 s 预热 + 2 个过渡周期 + 5 个分析周期。取 280 s 以满足
        # dt=0.05 s 的固定步长要求；与原参数模糊脚本的 279.95 s 一致到一个步长。
        # Round upward to the next 0.05 s grid point. The solver performs the
        # final exact multiple-of-dt check when a different dt is requested.
        return float(np.ceil(scenario_parameters.periodic_end / 0.05) * 0.05)
    if scenario in ("accelerating", "braking"):
        return 50.0
    raise ValueError(f"Unknown scenario: {scenario}")


def equilibrium_gap(speed: float, parameters: IDMParameters) -> float:
    """Return the IDM net gap producing zero acceleration at equal speeds."""
    denominator = 1.0 - (speed / parameters.desired_speed) ** parameters.acceleration_exponent
    if denominator <= 0.0:
        raise ValueError("Finite equilibrium gap requires speed below desired speed.")
    desired_gap = parameters.minimum_gap + speed * parameters.desired_time_headway
    return float(desired_gap / np.sqrt(denominator))


def idm_acceleration(
    follower_x: float,
    follower_speed: float,
    leader_x: float,
    leader_speed: float,
    parameters: IDMParameters,
) -> Tuple[float, float]:
    """Return IDM acceleration and net gap.

    This demo uses the standard net-gap IDM variant. The nonnegative desired-gap
    clamp is a documented numerical/physical safeguard for extreme closing terms.
    """
    gap = leader_x - follower_x - parameters.vehicle_length
    if gap <= 0.0:
        raise FloatingPointError(f"Non-positive net gap: {gap}")

    relative_speed = follower_speed - leader_speed
    interaction = follower_speed * relative_speed / (
        2.0
        * np.sqrt(
            parameters.max_acceleration * parameters.comfortable_deceleration
        )
    )
    desired_gap = parameters.minimum_gap + max(
        0.0,
        follower_speed * parameters.desired_time_headway + interaction,
    )
    acceleration = parameters.max_acceleration * (
        1.0
        - (follower_speed / parameters.desired_speed)
        ** parameters.acceleration_exponent
        - (desired_gap / gap) ** 2
    )
    return float(acceleration), float(gap)


def simulate_platoon(
    scenario: str,
    initial_gap: float,
    initial_speed: float,
    parameters: IDMParameters,
    dt: float = 0.05,
    follower_count: int = 5,
    scenario_parameters: ScenarioParameters = DEFAULT_SCENARIO_PARAMETERS,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Simulate a deterministic five-vehicle follower platoon.

    One initial-gap and initial-speed sample is shared by all followers. This
    represents a correlated uncertain initial platoon state and keeps the state
    dimension at two fuzzy inputs for a clean comparison with parameter fuzziness.
    """
    if scenario == "periodic":
        duration = float(np.ceil(scenario_parameters.periodic_end / dt) * dt)
    else:
        duration = scenario_duration(scenario, scenario_parameters)
    initial_state = np.empty(2 * follower_count, dtype=float)
    for follower_index in range(follower_count):
        initial_state[2 * follower_index] = -(
            follower_index + 1
        ) * (parameters.vehicle_length + initial_gap)
        initial_state[2 * follower_index + 1] = initial_speed

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        derivative = np.empty_like(state)
        external_x, external_speed, _ = leader_state(t, scenario, scenario_parameters)
        for follower_index in range(follower_count):
            x = state[2 * follower_index]
            speed = state[2 * follower_index + 1]
            if follower_index == 0:
                leader_x = external_x
                leader_speed = external_speed
            else:
                leader_x = state[2 * (follower_index - 1)]
                leader_speed = state[2 * (follower_index - 1) + 1]
            acceleration, _ = idm_acceleration(
                x, speed, leader_x, leader_speed, parameters
            )
            if speed <= 0.0 and acceleration < 0.0:
                acceleration = 0.0
            derivative[2 * follower_index] = speed
            derivative[2 * follower_index + 1] = acceleration
        return derivative

    times, states = solve_ode(
        rhs,
        initial_state,
        duration,
        dt,
        method="RK4",
        nonnegative_indices=tuple(range(1, 2 * follower_count, 2)),
    )

    leaders = np.array(
        [leader_state(float(t), scenario, scenario_parameters) for t in times],
        dtype=float,
    )
    speeds = states[:, 1::2]
    positions = states[:, 0::2]
    gaps = np.empty_like(speeds)
    accelerations = np.empty_like(speeds)
    for time_index, (time, state) in enumerate(zip(times, states)):
        external_x, external_speed, _ = leader_state(
            float(time), scenario, scenario_parameters
        )
        for follower_index in range(follower_count):
            x = state[2 * follower_index]
            speed = state[2 * follower_index + 1]
            if follower_index == 0:
                leader_x = external_x
                leader_speed = external_speed
            else:
                leader_x = state[2 * (follower_index - 1)]
                leader_speed = state[2 * (follower_index - 1) + 1]
            acceleration, gap = idm_acceleration(
                x, speed, leader_x, leader_speed, parameters
            )
            if speed <= 1e-6 and acceleration < 0.0:
                acceleration = 0.0
            gaps[time_index, follower_index] = gap
            accelerations[time_index, follower_index] = acceleration
    return times, positions, speeds, gaps, accelerations, leaders


def simulate_follower(
    initial_gap: float,
    initial_speed: float,
    parameters: IDMParameters,
    duration: float = 30.0,
    dt: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible constant-leader single-follower helper for tests."""
    initial_x = 30.0 - parameters.vehicle_length - initial_gap

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        x, speed = state
        leader_x = 30.0 + 15.0 * t
        acceleration, _ = idm_acceleration(x, speed, leader_x, 15.0, parameters)
        return np.array([speed, acceleration], dtype=float)

    times, states = solve_ode(
        rhs,
        np.array([initial_x, initial_speed], dtype=float),
        duration,
        dt,
        method="RK4",
        nonnegative_indices=(1,),
    )
    leader_positions = 30.0 + 15.0 * times
    gaps = leader_positions - states[:, 0] - parameters.vehicle_length
    accelerations = np.array(
        [idm_acceleration(x, speed, leader_x, 15.0, parameters)[0]
         for x, speed, leader_x in zip(states[:, 0], states[:, 1], leader_positions)],
        dtype=float,
    )
    return times, states, gaps, accelerations
