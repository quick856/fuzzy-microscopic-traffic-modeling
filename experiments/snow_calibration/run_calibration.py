"""100 km/h S040/S050 IDM trial with wider lower numerical search bounds.

The 52 candidate episodes and 50-to-16 mm reference scaling are the same as
in a9_50mm_joint_100kmh_trial_20261021.py. This file does not alter source data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import scale_reference as scale
import calibration_core as method


U_DES_KMH = 100.0
LOW = np.array([15.0, 0.05, 0.05, 0.10])
HIGH = np.array([65.0, 16.0, 24.0, 4.0])
OUT = scale.OUT / "wide_bounds_100kmh_calibration.csv"


def fit_case(name: str, train: list, test: list, seeds: tuple[int, int]) -> dict:
    trials = [method.pso(train, seed, particles=32, iterations=70,
                         fixed_u_des=U_DES_KMH / 3.6) for seed in seeds]
    theta, score, _ = min(trials, key=lambda trial: trial[1])
    row = {
        "case": name,
        "train_n": len(train),
        "test_n": len(test),
        "u_des_kmh_fixed": U_DES_KMH,
        "a_mps2": float(theta[1]),
        "b_mps2": float(theta[2]),
        "T_s": float(theta[3]),
        "train_objective": float(score),
        "cross_objective": float(method.objective(theta[None, :], test)[0])
        if test else np.nan,
        "S040_objective": float(method.objective(
            theta[None, :], [ep for ep in train
                             if ep.camera == scale.CAMERAS[0]])[0])
        if name == "pooled" else np.nan,
        "S050_objective": float(method.objective(
            theta[None, :], [ep for ep in train
                             if ep.camera == scale.CAMERAS[1]])[0])
        if name == "pooled" else np.nan,
        "a_at_lower": bool(theta[1] - LOW[1] < 0.001),
        "b_at_lower": bool(theta[2] - LOW[2] < 0.001),
        "T_at_lower": bool(theta[3] - LOW[3] < 0.001),
        "a_at_upper": bool(HIGH[1] - theta[1] < 0.001),
        "b_at_upper": bool(HIGH[2] - theta[2] < 0.001),
        "T_at_upper": bool(HIGH[3] - theta[3] < 0.001),
        "two_seed_objective_range": float(abs(trials[0][1] - trials[1][1])),
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main() -> None:
    method.CAMERAS = scale.CAMERAS
    method.MIN_DURATION_S = 0.0
    method.MIN_RECONSTRUCTED_FRAMES = 2
    method.MAX_MEDIAN_THW_S = float("inf")
    method.S0_M = 2.0
    method.STEP_S = 0.20
    method.WEIGHT_BY_DURATION = False
    method.LOW = LOW.copy()
    method.HIGH = HIGH.copy()

    source = pd.read_csv(scale.SOURCE, encoding="utf-8-sig", low_memory=False)
    lookup, _ = method.reconstruct_selected_tracks(source)
    episodes, _, _ = method.prepare(source, lookup)
    counts = [sum(ep.camera == camera for ep in episodes)
              for camera in scale.CAMERAS]
    if counts != [26, 26]:
        raise RuntimeError(f"Expected 26 episodes per station, got {counts}")
    factors, _ = scale.fit_cross_lens_scale()
    episodes = [scale.rescale_episode(ep, factors[ep.camera]) for ep in episodes]
    s040 = [ep for ep in episodes if ep.camera == scale.CAMERAS[0]]
    s050 = [ep for ep in episodes if ep.camera == scale.CAMERAS[1]]
    rows = [
        fit_case("pooled", episodes, [], (20261021, 20261022)),
        fit_case("train_S040_test_S050", s040, s050, (20261031, 20261032)),
        fit_case("train_S050_test_S040", s050, s040, (20261041, 20261042)),
    ]
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
