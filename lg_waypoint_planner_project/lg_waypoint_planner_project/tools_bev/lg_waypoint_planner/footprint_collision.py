# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple
import math
import numpy as np


def obb_corners(center_xy, yaw: float, half_length: float, half_width: float) -> np.ndarray:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    hl, hw = float(half_length), float(half_width)
    local = np.asarray([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    out = np.zeros_like(local, dtype=np.float32)
    out[:, 0] = cx + c * local[:, 0] - s * local[:, 1]
    out[:, 1] = cy + s * local[:, 0] + c * local[:, 1]
    return out


def _axes(poly: np.ndarray) -> List[np.ndarray]:
    axes = []
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        n = np.asarray([-e[1], e[0]], dtype=np.float32)
        norm = float(np.linalg.norm(n))
        if norm > 1e-6:
            axes.append(n / norm)
    return axes


def polygons_overlap_sat(a: np.ndarray, b: np.ndarray) -> bool:
    for ax in _axes(a) + _axes(b):
        pa = a @ ax
        pb = b @ ax
        if float(np.max(pa)) < float(np.min(pb)) or float(np.max(pb)) < float(np.min(pa)):
            return False
    return True


def check_rollout_collisions(rollout: Dict, actor_timelines: Dict[int, List[Dict]], cfg) -> Dict:
    waypoints = np.asarray(rollout.get("waypoints", []), dtype=np.float32)
    yaws = np.asarray(rollout.get("yaws", []), dtype=np.float32)
    events = []
    min_clearance = float("inf")

    for k in range(1, len(waypoints) + 1):
        ego_poly = obb_corners(
            waypoints[k - 1],
            float(yaws[k - 1]) if len(yaws) >= k else 0.0,
            float(cfg.vehicle.ego_half_length_m),
            float(cfg.vehicle.ego_half_width_m),
        )
        actors = actor_timelines.get(k, [])
        for actor in actors:
            ax = float(actor.get("x_m", 0.0)); ay = float(actor.get("y_m", 0.0))
            actor_poly = obb_corners(
                [ax, ay],
                float(actor.get("yaw_rad", 0.0)),
                float(actor.get("half_length_m", cfg.actors.default_vehicle_half_length_m)),
                float(actor.get("half_width_m", cfg.actors.default_vehicle_half_width_m)),
            )
            center_dist = float(np.linalg.norm(np.asarray([ax, ay], dtype=np.float32) - waypoints[k - 1]))
            approx_clearance = center_dist - float(cfg.vehicle.ego_half_length_m) - float(actor.get("half_length_m", 2.0))
            min_clearance = min(min_clearance, approx_clearance)
            if polygons_overlap_sat(ego_poly, actor_poly):
                events.append({
                    "time_index": int(k),
                    "time_s": float(k / float(cfg.horizon.future_fps)),
                    "actor_id": actor.get("id", None),
                    "actor_class": actor.get("class", "actor"),
                    "actor_relative_position": actor.get("relative_position", "unknown"),
                    "actor_x_m": float(ax),
                    "actor_y_m": float(ay),
                })
    return {
        "collision_free": len(events) == 0,
        "num_collision_events": int(len(events)),
        "first_collision": events[0] if events else None,
        "collision_events": events,
        "min_approx_clearance_m": float(min_clearance) if np.isfinite(min_clearance) else None,
    }
