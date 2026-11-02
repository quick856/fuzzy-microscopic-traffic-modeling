"""Estimate a 50 mm to 16 mm position scale from automatic paired tracks."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import calibration_core as method

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "A9雪天_双焦距尺度校正试验_20261021"
MATCHES = ROOT / "A9雪天_位置米制尺度核验_20261020" / "cross_lens_stable_matches.csv"
SOURCE = ROOT / "A9雪天_来向_仅S040S050_50mm_候选跟驰车对逐帧数据_20261019.csv"
CAMERAS = ("s040_camera_basler_north_50mm", "s050_camera_basler_south_50mm")

def fit_cross_lens_scale() -> tuple[dict[str, float], list[dict]]:
    """Fit x16 = intercept + slope*x50; hold out entire vehicle-ID pairs."""
    matches = pd.read_csv(MATCHES)
    rng = np.random.default_rng(20261021)
    slopes: dict[str, float] = {}
    diagnostics: list[dict] = []
    for station, camera in zip(("s040", "s050"), CAMERAS):
        frame = matches[(matches.station == station) &
                        (matches.variant == "dev_opposite")].copy()
        frame["vehicle_pair"] = frame.id16 + "|" + frame.id50
        ids = np.array(sorted(frame.vehicle_pair.unique()))
        rng.shuffle(ids)
        n_train = max(1, int(len(ids) * 0.8))
        train = frame[frame.vehicle_pair.isin(ids[:n_train])]
        test = frame[frame.vehicle_pair.isin(ids[n_train:])]
        slope, intercept = np.polyfit(train.x50_m, train.x16_m, 1)
        if slope <= 0:
            raise ValueError(f"Nonpositive scale for {station}")
        predicted = intercept + slope * test.x50_m.to_numpy(float)
        errors = np.abs(test.x16_m.to_numpy(float) - predicted)
        slopes[camera] = float(slope)
        diagnostics.append({
            "station": station,
            "reference": "16mm automatic same-vehicle matches, dev_opposite",
            "n_train_vehicle_pairs": n_train,
            "n_test_vehicle_pairs": len(ids) - n_train,
            "n_test_frames": len(test),
            "x16_equals_intercept_plus_slope_times_x50_intercept_m": float(intercept),
            "slope": float(slope),
            "test_position_mae_m": float(errors.mean()),
            "test_position_median_abs_error_m": float(np.median(errors)),
            "test_position_p90_abs_error_m": float(np.percentile(errors, 90)),
        })
    return slopes, diagnostics



def rescale_episode(ep: method.Episode, factor: float) -> method.Episode:
    """Scale both vehicles consistently; affine intercept cancels in headway.

    Derivatives of x16 = intercept + factor*x50 are factor times the original
    speed and acceleration. A common intercept does not affect IDM dynamics.
    """
    return replace(ep, lead_x=ep.lead_x * factor, lead_u=ep.lead_u * factor,
                   follower_x=ep.follower_x * factor,
                   follower_u=ep.follower_u * factor,
                   follower_ax=ep.follower_ax * factor)

