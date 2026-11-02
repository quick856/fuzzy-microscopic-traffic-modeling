"""Road-plane projection from A9 camera calibration JSON."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CALIB = ROOT / "a9_idm_calibration_pilot"
CAMERAS = {
    "s040_camera_basler_north_50mm": ("official_s040_north_50mm_calibration.json", "north"),
    "s050_camera_basler_south_50mm": ("official_s050_south_50mm_calibration.json", "south"),
}
ROAD_X_RANGE_M = (-100.0, 600.0)
ROAD_Y_RANGE_M = (-30.0, 30.0)

def calibration(camera: str, filename: str, direction: str) -> tuple[np.ndarray, dict]:
    path = CALIB / filename
    raw = path.read_bytes()
    data = json.loads(raw)
    k_key = f"calibrated_intrinsic_camera_matrix_{direction}_driving_direction"
    t_key = f"transformation_common_road_to_sensor_{direction}_driving_direction"
    k = np.asarray(data[k_key], dtype=float)[:, :3]
    t = np.asarray(data[t_key], dtype=float)[:3, :]
    h = (k @ t)[:, [0, 1, 3]]  # road Z=0; image ~ H [X,Y,1]
    if abs(np.linalg.det(h)) < 1e-9:
        raise ValueError(f"Singular road homography for {camera}")
    meta = {"filename": filename, "sha256": hashlib.sha256(raw).hexdigest(),
            "width": int(data["image_width"]), "height": int(data["image_height"]),
            "intrinsic_key": k_key, "transform_key": t_key,
            "road_frame_id": f"{camera}_{direction}_road_plane"}
    return h, meta



def project_pixel(pixel: np.ndarray, h: np.ndarray) -> tuple[float, float, bool]:
    q = np.linalg.solve(h, np.array([pixel[0], pixel[1], 1.0]))
    if not np.all(np.isfinite(q)) or q[2] <= 0 or abs(q[2]) < 1e-12:
        return np.nan, np.nan, False
    road = q[:2] / q[2]
    return float(road[0]), float(road[1]), bool(np.all(np.isfinite(road)))

