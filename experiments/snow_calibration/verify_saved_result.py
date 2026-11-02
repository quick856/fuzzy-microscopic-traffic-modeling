"""Recompute the saved 100 km/h snow-scene calibration scores from packaged inputs.

Run from any directory: python verify_saved_result.py
This is a consistency check of the candidate-data calculation, not a claim that
the automatically selected vehicle pairs are ground-truth direct following.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import scale_reference as scale
import calibration_core as method


def main() -> None:
    method.CAMERAS = scale.CAMERAS
    method.MIN_DURATION_S = 0.0
    method.MIN_RECONSTRUCTED_FRAMES = 2
    method.MAX_MEDIAN_THW_S = float("inf")
    method.S0_M = 2.0
    method.STEP_S = 0.20
    method.WEIGHT_BY_DURATION = False

    source = pd.read_csv(scale.SOURCE, encoding="utf-8-sig", low_memory=False)
    assert len(source) == 3937 and source.episode_id.nunique() == 57
    lookup, _ = method.reconstruct_selected_tracks(source)
    episodes, _, _ = method.prepare(source, lookup)
    counts = [sum(ep.camera == camera for ep in episodes) for camera in scale.CAMERAS]
    assert counts == [26, 26], counts
    factors, mapping = scale.fit_cross_lens_scale()
    episodes = [scale.rescale_episode(ep, factors[ep.camera]) for ep in episodes]
    by_camera = {
        camera: [ep for ep in episodes if ep.camera == camera]
        for camera in scale.CAMERAS
    }

    saved = pd.read_csv(
        scale.OUT / "wide_bounds_100kmh_calibration.csv", encoding="utf-8-sig"
    )
    for row in saved.itertuples(index=False):
        theta = np.array([row.u_des_kmh_fixed / 3.6, row.a_mps2, row.b_mps2, row.T_s])
        if row.case == "pooled":
            train, test = episodes, []
        elif row.case == "train_S040_test_S050":
            train, test = by_camera[scale.CAMERAS[0]], by_camera[scale.CAMERAS[1]]
        elif row.case == "train_S050_test_S040":
            train, test = by_camera[scale.CAMERAS[1]], by_camera[scale.CAMERAS[0]]
        else:
            raise ValueError(row.case)
        train_score = float(method.objective(theta[None, :], train)[0])
        assert np.isclose(train_score, row.train_objective, atol=1e-8), (row.case, train_score)
        if test:
            test_score = float(method.objective(theta[None, :], test)[0])
            assert np.isclose(test_score, row.cross_objective, atol=1e-8), (row.case, test_score)
        print(f"{row.case}: n={len(train)}, J={train_score:.8f}")
    print("scale slopes:", {key: round(value, 9) for key, value in factors.items()})
    print("52 episodes; saved results reproducible from packaged inputs")


if __name__ == "__main__":
    main()
