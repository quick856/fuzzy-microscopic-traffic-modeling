"""Reconstruct A9 candidate tracks and evaluate the leader-driven IDM objective.

The candidate pairs were selected automatically; this module does not turn
them into manually confirmed direct-following observations.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
import camera_projection as base

ROOT = Path(__file__).resolve().parent
OBS = ROOT / "a9_calibration_tables_20261016" / "a9_vehicle_observations.csv"
CAMERAS = ("s040_camera_basler_north_50mm", "s050_camera_basler_south_50mm")
LOW = np.array([15.0, 0.2, 0.2, 0.5])
HIGH = np.array([65.0, 4.0, 6.0, 4.0])
S0_M = 2.0
MIN_DURATION_S = 5.0
MIN_RECONSTRUCTED_FRAMES = 20
MAX_MEDIAN_THW_S = 3.0
WEIGHT_BY_DURATION = False
STEP_S = 0.20

@dataclass
class Episode:
    camera: str
    episode_id: str
    pair_id: str
    time: np.ndarray
    lead_x: np.ndarray
    lead_u: np.ndarray
    follower_x: np.ndarray
    follower_u: np.ndarray
    follower_ax: np.ndarray
    raw_count: int
    trial_fraction: float



def reconstruct_selected_tracks(candidates: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Reproject raw box pixels and fit each full incoming vehicle track.

    Cubic smoothing spline solves sum(position residual^2)+lambda*curvature.
    Pure generalized cross-validation can dramatically undersmooth projected
    image annotations. We first estimate outlier weights from a regularized
    pilot fit, then choose the smallest lambda on a declared grid that keeps
    the full-track speed and acceleration within broad physical QA limits.
    These limits select a smoother; they are not measured acceleration data.
    """
    ids = set(zip(candidates.camera_id, candidates.follower_id)) | set(
        zip(candidates.camera_id, candidates.leader_id))
    observations = pd.read_csv(OBS, encoding="utf-8-sig", low_memory=False)
    observations = observations[observations.apply(
        lambda r: (r.camera_id, r.vehicle_id_camera_local) in ids, axis=1)].copy()
    matrices = {}
    for camera in CAMERAS:
        file, facing = base.CAMERAS[camera]
        incoming = "south" if facing == "north" else "north"
        matrices[camera] = base.calibration(camera, file, incoming)[0]
    projected = []
    for row in observations.itertuples(index=False):
        if not row.pixel_in_image:
            continue
        h = matrices[row.camera_id]
        x, y, valid = base.project_pixel(
            np.array([row.box_bottom_centroid_u_px, row.box_bottom_centroid_v_px]), h)
        if valid and base.ROAD_X_RANGE_M[0] <= x <= base.ROAD_X_RANGE_M[1] \
                and base.ROAD_Y_RANGE_M[0] <= y <= base.ROAD_Y_RANGE_M[1]:
            projected.append((row.camera_id, row.vehicle_id_camera_local,
                              int(row.timestamp_unix_ns), x, y))
    table = pd.DataFrame(projected, columns=["camera", "vehicle", "ns", "x", "y"])
    lookup: dict[tuple[str, str, int], dict] = {}
    qa = []
    for (camera, vehicle), group in table.groupby(["camera", "vehicle"], sort=True):
        group = group.sort_values("ns").drop_duplicates("ns")
        ns = group.ns.to_numpy(np.int64)
        split = np.r_[0, np.flatnonzero(np.diff(ns) > 200_000_000) + 1, len(ns)]
        for segment_index, (a, b) in enumerate(zip(split[:-1], split[1:])):
            seg = group.iloc[a:b]
            if len(seg) < 15:
                continue
            t = (seg.ns.to_numpy(np.int64) - int(seg.ns.iloc[0])) / 1e9
            raw_x = seg.x.to_numpy(float)
            raw_y = seg.y.to_numpy(float)
            direction = int(np.sign(np.polyfit(t, raw_x, 1)[0]))
            if not direction:
                continue
            selected_lam = None
            passed_qa = False
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pilot = make_smoothing_spline(t, raw_x, lam=1.0)
                    residual = raw_x - pilot(t)
                    sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
                    sigma = max(float(sigma), 0.20)
                    weights = np.minimum(1.0, 3.0 * sigma / np.maximum(np.abs(residual), 1e-12))
                    grid = np.logspace(-3, 5, 25)
                    for lam in grid:
                        candidate = make_smoothing_spline(t, raw_x, w=weights, lam=float(lam))
                        candidate_speed = direction * candidate.derivative(1)(t)
                        candidate_accel = direction * candidate.derivative(2)(t)
                        if (np.isfinite(candidate_speed).all()
                                and np.isfinite(candidate_accel).all()
                                and candidate_speed.min() >= 0
                                and candidate_speed.max() <= 55
                                and np.max(np.abs(candidate_accel)) <= 8):
                            spline, selected_lam, passed_qa = candidate, float(lam), True
                            break
                    if not passed_qa:
                        spline, selected_lam = candidate, float(grid[-1])
            except (ValueError, np.linalg.LinAlgError):
                continue
            sx = spline(t)
            speed = direction * spline.derivative(1)(t)
            accel = direction * spline.derivative(2)(t)
            residual_rmse = float(np.sqrt(np.mean((sx - raw_x) ** 2)))
            position_qa_pass = passed_qa and residual_rmse <= 5.0
            if position_qa_pass:
                for j, row in enumerate(seg.itertuples(index=False)):
                    lookup[(camera, vehicle, int(row.ns))] = {
                    "raw_x": direction * float(raw_x[j]),
                    "smooth_x": direction * float(sx[j]),
                    "u": float(speed[j]), "ax": float(accel[j]),
                    "raw_y": float(raw_y[j]), "direction": direction,
                    "segment": segment_index,
                    }
            qa.append({"camera_id": camera, "vehicle_id": vehicle,
                       "segment": segment_index, "frames": len(seg),
                       "duration_s": float(t[-1]), "direction": direction,
                       "smoothing_lambda": selected_lam,
                       "physical_kinematic_qa_pass": int(passed_qa),
                       "position_residual_qa_pass": int(residual_rmse <= 5.0),
                       "position_residual_rmse_m": residual_rmse,
                       "speed_min_mps": float(speed.min()),
                       "speed_max_mps": float(speed.max()),
                       "accel_abs_max_mps2": float(np.max(np.abs(accel)))})
    return lookup, qa



def prepare(candidates: pd.DataFrame, lookup: dict) -> tuple[list[Episode], pd.DataFrame, list[dict]]:
    """Join selected candidate pairs to full-track spline estimates.

    Selection is fixed before fitting: >=5 s, independent of the old derivative-
    based auto-screen flag. Remaining frames require both tracks, plausible kinematics, and a positive
    projected center headway. This does not replace manual lane/following review.
    """
    episodes = []
    rows = []
    audit = []
    for (camera, eid), group in candidates.groupby(["camera_id", "episode_id"], sort=True):
        group = group.sort_values("timestamp_unix_ns")
        duration = float(group.time_s.iloc[-1] - group.time_s.iloc[0])
        reason = ""
        if duration < MIN_DURATION_S:
            reason = "duration_below_5s"
        records = []
        if not reason:
            for row in group.itertuples(index=False):
                fk = (camera, row.follower_id, int(row.timestamp_unix_ns))
                lk = (camera, row.leader_id, int(row.timestamp_unix_ns))
                f, l = lookup.get(fk), lookup.get(lk)
                if f is None or l is None or f["direction"] != l["direction"]:
                    continue
                h = l["smooth_x"] - f["smooth_x"]
                valid = bool(0 < f["u"] <= 55 and 0 < l["u"] <= 55
                             and 0 < h <= 150 and np.isfinite([f["ax"], l["ax"]]).all())
                if not valid:
                    continue
                records.append((row, f, l, h))
            if len(records) < MIN_RECONSTRUCTED_FRAMES:
                reason = f"under_{MIN_RECONSTRUCTED_FRAMES}_reconstructed_frames"
            elif (records[-1][0].timestamp_unix_ns - records[0][0].timestamp_unix_ns) / 1e9 < MIN_DURATION_S:
                reason = "reconstructed_span_below_5s"
        audit.append({"episode_id": eid, "camera_id": camera,
                      "raw_frames": len(group), "duration_s": duration,
                      "reconstructed_frames": len(records),
                      "included": int(not reason), "exclusion_reason": reason,
                      "manual_following_review_status": "unreviewed"})
        if reason:
            continue
        # Retain a regular time grid; observed positions and analytic spline
        # derivatives are interpolated only across accepted source frames.
        raw_t = np.array([(r.timestamp_unix_ns - records[0][0].timestamp_unix_ns) / 1e9
                          for r, _, _, _ in records])
        t = np.r_[np.arange(0, raw_t[-1], STEP_S), raw_t[-1]]
        if np.any(np.diff(t) <= 1e-9):
            t = np.unique(t)
        fx = np.interp(t, raw_t, [f["smooth_x"] for _, f, _, _ in records])
        fu = np.interp(t, raw_t, [f["u"] for _, f, _, _ in records])
        fax = np.interp(t, raw_t, [f["ax"] for _, f, _, _ in records])
        lx = np.interp(t, raw_t, [l["smooth_x"] for _, _, l, _ in records])
        lu = np.interp(t, raw_t, [l["u"] for _, _, l, _ in records])
        if np.any(lx - fx <= 0) or np.any(fu <= 0) or np.any(lu <= 0):
            audit[-1]["included"] = 0
            audit[-1]["exclusion_reason"] = "interpolated_nonphysical_state"
            continue
        median_time_headway = float(np.median((lx - fx) / fu))
        audit[-1]["median_time_headway_s"] = median_time_headway
        if median_time_headway > MAX_MEDIAN_THW_S:
            audit[-1]["included"] = 0
            audit[-1]["exclusion_reason"] = f"likely_free_flow_time_headway_above_{MAX_MEDIAN_THW_S:g}s"
            continue
        episodes.append(Episode(camera, eid, str(records[0][0].pair_id), t, lx, lu,
                                fx, fu, fax, len(group), float(group.trial_fit_use.mean())))
        for row, f, l, h in records:
            rows.append({"camera_id": camera, "pair_id": row.pair_id,
                         "episode_id": eid, "timestamp_unix_ns": int(row.timestamp_unix_ns),
                         "time_s": float((row.timestamp_unix_ns - records[0][0].timestamp_unix_ns) / 1e9),
                         "follower_x_raw_reprojected_m": f["raw_x"],
                         "follower_x_spline_m": f["smooth_x"],
                         "follower_u_spline_mps": f["u"],
                         "follower_ax_spline_mps2": f["ax"],
                         "leader_x_raw_reprojected_m": l["raw_x"],
                         "leader_x_spline_m": l["smooth_x"],
                         "leader_u_spline_mps": l["u"],
                         "leader_ax_spline_mps2": l["ax"],
                         "h_center_spline_m": h,
                         "lateral_sep_from_raw_projection_m": abs(f["raw_y"] - l["raw_y"]),
                         "original_trial_fit_use": int(row.trial_fit_use),
                         "manual_following_review_status": "unreviewed",
                         "calibration_eligible": 0})
    return episodes, pd.DataFrame(rows), audit



def simulate_swarm(particles: np.ndarray, ep: Episode) -> np.ndarray:
    """RK4 on one leader-driven episode, vectorized over PSO particles."""
    n = len(particles)
    states = np.empty((len(ep.time), n, 2))
    states[0, :, 0] = ep.follower_x[0]
    states[0, :, 1] = ep.follower_u[0]
    vdes, a, b, headway_t = particles.T

    def rhs(s: np.ndarray, lx: float, lu: float) -> np.ndarray:
        x = s[:, 0]
        u = s[:, 1]
        h = np.maximum(lx - x, 0.2)
        wanted = S0_M + u * headway_t + u * (u - lu) / (2.0 * np.sqrt(a * b))
        ax = a * (1.0 - (u / vdes) ** 4 - (wanted / h) ** 2)
        return np.column_stack((u, ax))

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(len(ep.time) - 1):
            dt = ep.time[k + 1] - ep.time[k]
            state = states[k]
            mx = (ep.lead_x[k] + ep.lead_x[k + 1]) / 2
            mu = (ep.lead_u[k] + ep.lead_u[k + 1]) / 2
            k1 = rhs(state, ep.lead_x[k], ep.lead_u[k])
            k2 = rhs(state + dt * k1 / 2, mx, mu)
            k3 = rhs(state + dt * k2 / 2, mx, mu)
            k4 = rhs(state + dt * k3, ep.lead_x[k + 1], ep.lead_u[k + 1])
            states[k + 1] = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return states



def objective(particles: np.ndarray, episodes: list[Episode]) -> np.ndarray:
    """Equal-episode mean of 0.5 RMSRE(headway) + 0.5 RMSRE(speed)."""
    score = np.zeros(len(particles))
    total_weight = 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for ep in episodes:
            episode_weight = float(ep.time[-1]) if WEIGHT_BY_DURATION else 1.0
            total_weight += episode_weight
            pred = simulate_swarm(particles, ep)
            h_obs = ep.lead_x - ep.follower_x
            h_pred = ep.lead_x[:, None] - pred[:, :, 0]
            speed_pred = pred[:, :, 1]
            e_h = np.sqrt(np.mean(((h_pred - h_obs[:, None]) / h_obs[:, None]) ** 2, axis=0))
            e_u = np.sqrt(np.mean(((speed_pred - ep.follower_u[:, None]) / ep.follower_u[:, None]) ** 2, axis=0))
            valid = np.isfinite(pred).all(axis=(0, 2)) & (speed_pred >= 0).all(axis=0) \
                & (h_pred > 0.2).all(axis=0) & (speed_pred < 80).all(axis=0)
            score += episode_weight * np.where(valid, 0.5 * e_h + 0.5 * e_u, 1e4)
    return score / total_weight



def pso(episodes: list[Episode], seed: int, particles: int = 32,
        iterations: int = 70, fixed_u_des: float | None = None) -> tuple[np.ndarray, float, list[dict]]:
    rng = np.random.default_rng(seed)
    low, high = LOW.copy(), HIGH.copy()
    if fixed_u_des is not None:
        if not LOW[0] <= fixed_u_des <= HIGH[0]:
            raise ValueError("fixed_u_des outside numerical search domain")
        low[0] = high[0] = fixed_u_des
    width = high - low
    x = rng.uniform(low, high, size=(particles, 4))
    x[0] = np.clip(np.array([35., 1., 1.5, 1.5]), low, high)
    v = rng.uniform(-0.05, 0.05, size=x.shape) * width
    personal = x.copy()
    personal_score = objective(x, episodes)
    j = int(np.argmin(personal_score))
    global_x, global_score = personal[j].copy(), float(personal_score[j])
    trace = []
    for iteration in range(iterations):
        r1 = rng.random(x.shape)
        r2 = rng.random(x.shape)
        v = 0.72 * v + 1.49 * r1 * (personal - x) + 1.49 * r2 * (global_x - x)
        v = np.clip(v, -0.2 * width, 0.2 * width)
        x = np.clip(x + v, low, high)
        score = objective(x, episodes)
        better = score < personal_score
        personal[better] = x[better]
        personal_score[better] = score[better]
        j = int(np.argmin(personal_score))
        if personal_score[j] < global_score:
            global_x, global_score = personal[j].copy(), float(personal_score[j])
        trace.append({"seed": seed, "iteration": iteration + 1,
                      "best_objective": global_score})
    return global_x, global_score, trace

