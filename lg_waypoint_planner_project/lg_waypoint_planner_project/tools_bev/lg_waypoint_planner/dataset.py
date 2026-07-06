# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .io_utils import load_json_gz
from .geometry import ensure_route_starts_at_ego, global_to_ego_local


def get_meters_per_pixel(meta: Dict, default_pixels_per_meter: float) -> float:
    if "meters_per_pixel" in meta:
        return float(meta["meters_per_pixel"])
    if "pixels_per_meter" in meta:
        ppm = float(meta["pixels_per_meter"])
        if ppm <= 0:
            raise ValueError(f"Invalid pixels_per_meter={ppm}")
        return 1.0 / ppm
    return 1.0 / float(default_pixels_per_meter)


def get_ego_center(meta: Dict, cost_shape) -> List[float]:
    if "ego_center" in meta and isinstance(meta["ego_center"], (list, tuple)) and len(meta["ego_center"]) == 2:
        return [float(meta["ego_center"][0]), float(meta["ego_center"][1])]
    h, w = cost_shape[:2]
    return [float(w // 2), float(h // 2)]


def get_route(measurement: Dict, route_key: str) -> np.ndarray:
    if route_key in measurement:
        route = measurement[route_key]
    elif "route" in measurement:
        route = measurement["route"]
    elif "route_original" in measurement:
        route = measurement["route_original"]
    else:
        raise KeyError(f"No route key found. Tried {route_key}, route, route_original.")
    return ensure_route_starts_at_ego(np.asarray(route, dtype=np.float32), threshold_m=0.5)


def get_current_ego_state(measurement: Dict) -> Dict:
    speed = max(float(measurement.get("speed", 0.0)), 0.0)
    return {"x": 0.0, "y": 0.0, "yaw": 0.0, "v": speed}


def get_pose_global_xy_yaw(measurement: Dict) -> Optional[Tuple[np.ndarray, float]]:
    if "pos_global" not in measurement or "theta" not in measurement:
        return None
    pos = np.asarray(measurement["pos_global"][:2], dtype=np.float32)
    yaw = float(measurement["theta"])
    if pos.shape[0] != 2 or not np.isfinite(pos).all() or not np.isfinite(yaw):
        return None
    return pos, yaw


def offset_frame_name(frame_name: str, offset: int, stride: int = 1) -> Optional[str]:
    try:
        val = int(frame_name)
    except Exception:
        return None
    return str(val + int(offset) * int(stride)).zfill(len(frame_name))


def load_future_ego_waypoints(route_dir: Path, frame_name: str, current_measurement: Dict, cfg) -> Optional[np.ndarray]:
    pose = get_pose_global_xy_yaw(current_measurement)
    if pose is None:
        return None
    ego_global, ego_yaw = pose
    pts = []
    for k in range(1, int(cfg.horizon.num_future_waypoints) + 1):
        name = offset_frame_name(frame_name, k, int(cfg.horizon.future_frame_stride))
        if name is None:
            return None
        p = route_dir / cfg.paths.measurements_folder / f"{name}.json.gz"
        if not p.exists():
            break
        m = load_json_gz(p)
        if "pos_global" not in m:
            break
        pts.append(global_to_ego_local(np.asarray(m["pos_global"][:2], dtype=np.float32), ego_global, ego_yaw))
    if len(pts) == 0:
        return None
    return np.asarray(pts, dtype=np.float32)
