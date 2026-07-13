# -*- coding: utf-8 -*-

"""Candidate evaluation for occupancy-BEV based waypoint selection.

The map saved under ``costmap/*.npy`` is treated as a binary/near-binary
occupancy BEV rather than a continuous risk cost map:

    low value  -> free/drivable
    high value -> blocked / occupied / non-drivable

The evaluator first checks whether the ego footprint overlaps occupied cells,
then ranks feasible candidates by route consistency, smoothness, geometric
collision, and behavior prior.  To avoid overly conservative labels, normal
route following is explicitly preferred when there is no direct actor or static
obstacle constraining the ego vehicle.
"""

from typing import Dict, List, Any
import math
import numpy as np

from .costmap import temporal_index_for_dense_step
from .geometry import local_points_to_pixels, min_distance_to_polyline
from .footprint_collision import check_rollout_collisions
from .candidate_generator import build_boundary_refined_candidate


def _cfg_get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    try:
        return getattr(obj, key)
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _cfg_float(obj: Any, key: str, default: float) -> float:
    try:
        return float(_cfg_get(obj, key, default))
    except Exception:
        return float(default)


def _cfg_bool(obj: Any, key: str, default: bool) -> bool:
    value = _cfg_get(obj, key, default)
    if isinstance(value, str):
        return value.lower() in ["1", "true", "yes", "y", "on"]
    return bool(value)


def _cfg_int(obj: Any, key: str, default: int) -> int:
    try:
        return int(_cfg_get(obj, key, default))
    except Exception:
        return int(default)


def _cfg_str_list(obj: Any, key: str, default) -> List[str]:
    value = _cfg_get(obj, key, default)
    try:
        return [str(v) for v in value]
    except Exception:
        return [str(v) for v in default]


def _has_direct_factor(factor: Dict) -> bool:
    """Whether the current scene has a direct reason to slow down or deviate."""
    if not isinstance(factor, dict):
        return False
    actor = factor.get("critical_actor") or {}
    if actor.get("exists", False):
        x = float(actor.get("x_m", 1e9))
        rel = str(actor.get("relative_position", ""))
        # Front/front-side actors or static obstacles can justify conservative
        # motion. Rear-side/background actors should not.
        return x >= -1.0 and rel in ["front", "front_left", "front_right"]

    ftype = str(factor.get("type", ""))
    return ftype in [
        "front_actor",
        "front_left_actor",
        "front_right_actor",
        "pedestrian_crossing_or_near_path",
        "front_static_obstacle",
        "front_left_static_obstacle",
        "front_right_static_obstacle",
        "static_obstacle_nearby",
        "limited_forward_space",
        "costmap_blocked_corridor",
        "red_light_stop_line",
        "stop_sign_control",
    ] or bool(factor.get("blocked_by_costmap", False))


def _is_clear_or_weak_factor(factor: Dict) -> bool:
    """True when no concrete object explains slowing or stopping."""
    if not isinstance(factor, dict):
        return True
    actor = factor.get("critical_actor") or {}
    if actor.get("exists", False):
        return False
    ftype = str(factor.get("type", ""))
    return ftype in [
        "clear_reference_corridor",
        "no_directly_influential_actor",
        "conservative_speed_profile_without_direct_actor",
        "route_shape_or_clearance_preference",
        "unknown",
        "",
    ] or not _has_direct_factor(factor)


def ego_sample_points_single(xy, yaw: float, cfg) -> np.ndarray:
    """Return points used to test the ego vehicle against the occupancy BEV."""
    if not _cfg_bool(_cfg_get(cfg, "scoring", {}), "score_footprint", True):
        return np.asarray(xy, dtype=np.float32).reshape(1, 2)

    ex = float(cfg.vehicle.ego_half_length_m)
    ey = float(cfg.vehicle.ego_half_width_m)
    local = np.asarray([
        [0.0, 0.0],
        [ex, 0.0], [-ex, 0.0], [0.0, ey], [0.0, -ey],
        [ex, ey], [ex, -ey], [-ex, ey], [-ex, -ey],
    ], dtype=np.float32)

    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    pts = np.zeros_like(local)
    pts[:, 0] = float(xy[0]) + c * local[:, 0] - s * local[:, 1]
    pts[:, 1] = float(xy[1]) + s * local[:, 0] + c * local[:, 1]
    return pts.astype(np.float32)


def _sample_nearest_occupancy(occupancy: np.ndarray, pixels: np.ndarray, out_of_bounds_value: float):
    """Nearest-neighbor sampling for occupancy maps."""
    occ = np.asarray(occupancy, dtype=np.float32)
    h, w = occ.shape[:2]
    pix = np.asarray(pixels, dtype=np.float32)
    col = np.rint(pix[:, 0]).astype(np.int32)
    row = np.rint(pix[:, 1]).astype(np.int32)
    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    values = np.full((len(pix),), float(out_of_bounds_value), dtype=np.float32)
    if np.any(valid):
        values[valid] = occ[row[valid], col[valid]].astype(np.float32)
    return values, valid


def occupancy_bev_score(rollout: Dict, temporal_costmaps: List[np.ndarray], ego_center, meters_per_pixel: float, cfg) -> Dict:
    """Check whether the ego rollout overlaps occupied BEV cells."""
    dense_xy = np.asarray(rollout.get("dense_xy", []), dtype=np.float32)
    dense_yaw = np.asarray(rollout.get("dense_yaw", []), dtype=np.float32)
    maps = list(temporal_costmaps or [])

    if len(maps) == 0 or len(dense_xy) == 0:
        return {
            "occupancy_mean_value": 0.0,
            "occupancy_max_value": 0.0,
            "occupancy_blocked_ratio": 0.0,
            "occupancy_any_blocked": False,
            "out_of_bounds_ratio": 0.0,
            "temporal_blocked_ratios": {},
            "temporal_max_occupancy_values": {},
            "mean_cost": 0.0,
            "max_cost": 0.0,
            "hard_ratio": 0.0,
            "temporal_mean_costs": {},
            "temporal_max_costs": {},
        }

    scoring_cfg = _cfg_get(cfg, "scoring", {})
    threshold = _cfg_float(scoring_cfg, "occupancy_blocked_threshold", _cfg_float(scoring_cfg, "hard_cost_threshold", 80.0))
    out_of_bounds_value = _cfg_float(scoring_cfg, "out_of_bounds_cost", threshold)

    all_values = []
    all_valid = []
    all_blocked = []
    by_tidx_values: Dict[int, List[float]] = {}
    by_tidx_blocked: Dict[int, List[bool]] = {}

    for i, (xy, yaw) in enumerate(zip(dense_xy, dense_yaw)):
        tidx = temporal_index_for_dense_step(i, len(maps), cfg)
        pts = ego_sample_points_single(xy, float(yaw), cfg)
        pix = local_points_to_pixels(pts, ego_center, meters_per_pixel)
        vals, valid = _sample_nearest_occupancy(maps[tidx], pix, out_of_bounds_value)
        blocked = vals >= threshold

        all_values.append(vals.astype(np.float32))
        all_valid.append(valid.astype(bool))
        all_blocked.append(blocked.astype(bool))
        by_tidx_values.setdefault(int(tidx), []).extend(vals.astype(float).tolist())
        by_tidx_blocked.setdefault(int(tidx), []).extend(blocked.astype(bool).tolist())

    values = np.concatenate(all_values).astype(np.float32) if all_values else np.zeros((0,), dtype=np.float32)
    valid = np.concatenate(all_valid).astype(bool) if all_valid else np.zeros((0,), dtype=bool)
    blocked = np.concatenate(all_blocked).astype(bool) if all_blocked else np.zeros((0,), dtype=bool)

    blocked_ratio = float(np.mean(blocked)) if len(blocked) else 0.0
    mean_value = float(np.mean(values)) if len(values) else 0.0
    max_value = float(np.max(values)) if len(values) else 0.0
    out_of_bounds_ratio = float(1.0 - np.mean(valid)) if len(valid) else 0.0

    temporal_blocked_ratios = {
        str(k): float(np.mean(v)) for k, v in by_tidx_blocked.items() if len(v)
    }
    temporal_max_values = {
        str(k): float(np.max(v)) for k, v in by_tidx_values.items() if len(v)
    }

    return {
        "occupancy_mean_value": mean_value,
        "occupancy_max_value": max_value,
        "occupancy_blocked_ratio": blocked_ratio,
        "occupancy_any_blocked": bool(np.any(blocked)) if len(blocked) else False,
        "out_of_bounds_ratio": out_of_bounds_ratio,
        "temporal_blocked_ratios": temporal_blocked_ratios,
        "temporal_max_occupancy_values": temporal_max_values,
        "mean_cost": mean_value,
        "max_cost": max_value,
        "hard_ratio": blocked_ratio,
        "temporal_mean_costs": {str(k): float(np.mean(v)) for k, v in by_tidx_values.items() if len(v)},
        "temporal_max_costs": temporal_max_values,
    }


costmap_score = occupancy_bev_score



def red_light_constraint_score(
    rollout: Dict,
    red_light_maps: List[np.ndarray],
    ego_center,
    meters_per_pixel: float,
    cfg,
) -> Dict:
    """Reject trajectories whose front edge crosses an active red stop line."""
    rule_cfg = _cfg_get(cfg, "traffic_rules", {})
    enabled = _cfg_bool(rule_cfg, "red_light_enabled", True)
    dense_xy = np.asarray(rollout.get("dense_xy", []), dtype=np.float32)
    dense_yaw = np.asarray(rollout.get("dense_yaw", []), dtype=np.float32)
    maps = list(red_light_maps or [])

    empty = {
        "red_light_rule_enabled": bool(enabled),
        "red_light_any_overlap": False,
        "red_light_overlap_ratio": 0.0,
        "first_red_light_violation": None,
        "red_light_active_temporal_indices": [],
    }
    if not enabled or len(maps) == 0 or len(dense_xy) == 0:
        return empty

    threshold = _cfg_float(rule_cfg, "red_light_blocked_threshold", 0.5)
    sample_count = max(int(_cfg_get(rule_cfg, "front_edge_sample_count", 3)), 1)
    ex = float(cfg.vehicle.ego_half_length_m)
    ey = float(cfg.vehicle.ego_half_width_m)
    lateral_samples = np.linspace(-ey, ey, sample_count, dtype=np.float32)

    overlaps = []
    first = None
    active_indices = [int(i) for i, m in enumerate(maps) if np.any(np.asarray(m) >= threshold)]

    for i, (xy, yaw) in enumerate(zip(dense_xy, dense_yaw)):
        tidx = temporal_index_for_dense_step(i, len(maps), cfg)
        c, ss = math.cos(float(yaw)), math.sin(float(yaw))
        local = np.stack([
            np.full((sample_count,), ex, dtype=np.float32),
            lateral_samples,
        ], axis=1)
        pts = np.zeros_like(local)
        pts[:, 0] = float(xy[0]) + c * local[:, 0] - ss * local[:, 1]
        pts[:, 1] = float(xy[1]) + ss * local[:, 0] + c * local[:, 1]
        pix = local_points_to_pixels(pts, ego_center, meters_per_pixel)
        vals, valid = _sample_nearest_occupancy(maps[tidx], pix, 0.0)
        overlap = bool(np.any(valid & (vals >= threshold)))
        overlaps.append(overlap)
        if overlap and first is None:
            first = {
                "dense_step": int(i),
                "temporal_index": int(tidx),
                "ego_x_m": float(xy[0]),
                "ego_y_m": float(xy[1]),
            }

    arr = np.asarray(overlaps, dtype=bool)
    return {
        "red_light_rule_enabled": True,
        "red_light_any_overlap": bool(np.any(arr)) if len(arr) else False,
        "red_light_overlap_ratio": float(np.mean(arr)) if len(arr) else 0.0,
        "first_red_light_violation": first,
        "red_light_active_temporal_indices": active_indices,
    }


def stop_sign_constraint_score(
    rollout: Dict,
    stop_sign_maps: List[np.ndarray],
    ego_center,
    meters_per_pixel: float,
    cfg,
) -> Dict:
    """Check whether the candidate completes a stop before entering the stop region.

    Unlike a red light, a stop sign is not a permanent no-crossing barrier.  A
    candidate may proceed after it has first remained below the configured
    complete-stop speed for the required duration.  This stateful check avoids
    letting a candidate bypass the rule merely because the expert's future
    ``stop`` mask disappears after the expert completes its own stop.
    """
    rule_cfg = _cfg_get(cfg, "traffic_rules", {})
    enabled = _cfg_bool(rule_cfg, "stop_sign_enabled", True)
    dense_xy = np.asarray(rollout.get("dense_xy", []), dtype=np.float32)
    dense_yaw = np.asarray(rollout.get("dense_yaw", []), dtype=np.float32)
    dense_speed = np.asarray(rollout.get("dense_speed", []), dtype=np.float32).reshape(-1)
    maps = list(stop_sign_maps or [])

    empty = {
        "stop_sign_rule_enabled": bool(enabled),
        "stop_sign_any_violation": False,
        "stop_sign_completed": False,
        "stop_sign_completion_dense_step": None,
        "stop_sign_first_region_entry": None,
        "first_stop_sign_violation": None,
        "stop_sign_active_temporal_indices": [],
        "current_stop_sign_active": False,
    }
    if not enabled or len(maps) == 0 or len(dense_xy) == 0:
        return empty

    threshold = _cfg_float(rule_cfg, "stop_sign_blocked_threshold", 0.5)
    active_indices = [int(i) for i, m in enumerate(maps) if np.any(np.asarray(m) >= threshold)]
    current_active = bool(0 in active_indices)
    if not active_indices:
        return {**empty, "stop_sign_rule_enabled": True}

    # All masks have already been warped into the current ego frame.  Their
    # union is therefore a stable world-space anchor for candidate-specific
    # stop completion, even if future masks disappear after the expert stops.
    anchor = np.maximum.reduce([np.asarray(m, dtype=np.float32) for m in maps])
    region_rc = np.argwhere(anchor >= threshold)
    region_xy = np.zeros((len(region_rc), 2), dtype=np.float32)
    if len(region_rc) > 0:
        region_xy[:, 0] = (float(ego_center[1]) - region_rc[:, 0].astype(np.float32)) * float(meters_per_pixel)
        region_xy[:, 1] = (region_rc[:, 1].astype(np.float32) - float(ego_center[0])) * float(meters_per_pixel)

    sample_count = max(int(_cfg_get(rule_cfg, "front_edge_sample_count", 3)), 1)
    ex = float(cfg.vehicle.ego_half_length_m)
    ey = float(cfg.vehicle.ego_half_width_m)
    lateral_samples = np.linspace(-ey, ey, sample_count, dtype=np.float32)

    stop_speed = max(_cfg_float(rule_cfg, "stop_sign_complete_speed_mps", 0.20), 0.0)
    min_duration = max(_cfg_float(rule_cfg, "stop_sign_complete_duration_s", 0.50), 0.0)
    completion_max_distance = max(_cfg_float(rule_cfg, "stop_sign_completion_max_distance_m", 2.0), 0.0)
    model_fps = max(float(cfg.horizon.model_fps), 1.0)
    required_steps = max(int(math.ceil(min_duration * model_fps)), 1)

    below_count = 0
    completed = False
    completion_step = None
    first_entry = None
    violation = None

    n = min(len(dense_xy), len(dense_yaw))
    for i in range(n):
        xy = dense_xy[i]
        yaw = float(dense_yaw[i])
        speed = float(dense_speed[i]) if i < len(dense_speed) else 0.0
        c, ss = math.cos(yaw), math.sin(yaw)
        local = np.stack([
            np.full((sample_count,), ex, dtype=np.float32),
            lateral_samples,
        ], axis=1)
        pts = np.zeros_like(local)
        pts[:, 0] = float(xy[0]) + c * local[:, 0] - ss * local[:, 1]
        pts[:, 1] = float(xy[1]) + ss * local[:, 0] + c * local[:, 1]
        if len(region_xy) > 0:
            delta = pts[:, None, :] - region_xy[None, :, :]
            distance_to_region = float(np.min(np.linalg.norm(delta, axis=2)))
        else:
            distance_to_region = float("inf")

        # A stop counts only when it happens near the controlled region. This
        # prevents an unrelated earlier traffic stop from satisfying the sign.
        if speed <= stop_speed and distance_to_region <= completion_max_distance:
            below_count += 1
        else:
            below_count = 0
        if not completed and below_count >= required_steps:
            completed = True
            completion_step = int(i)

        pix = local_points_to_pixels(pts, ego_center, meters_per_pixel)
        vals, valid = _sample_nearest_occupancy(anchor, pix, 0.0)
        inside = bool(np.any(valid & (vals >= threshold)))
        if inside and first_entry is None:
            first_entry = {
                "dense_step": int(i),
                "ego_x_m": float(xy[0]),
                "ego_y_m": float(xy[1]),
                "speed_mps": speed,
            }
        if inside and not completed and violation is None:
            violation = {
                "dense_step": int(i),
                "ego_x_m": float(xy[0]),
                "ego_y_m": float(xy[1]),
                "speed_mps": speed,
                "required_stop_steps": int(required_steps),
            }

    return {
        "stop_sign_rule_enabled": True,
        "stop_sign_any_violation": violation is not None,
        "stop_sign_completed": bool(completed),
        "stop_sign_completion_dense_step": completion_step,
        "stop_sign_first_region_entry": first_entry,
        "first_stop_sign_violation": violation,
        "stop_sign_active_temporal_indices": active_indices,
        "current_stop_sign_active": current_active,
    }

def solid_lane_constraint_score(
    rollout: Dict,
    solid_lane_map: np.ndarray,
    ego_center,
    meters_per_pixel: float,
    cfg,
) -> Dict:
    """Detect whether the planned ego-center path crosses a solid lane line.

    The solid-lane BEV is low resolution, so it must not be treated as a wide
    physical obstacle. The cost-map generator stores an approximately one-pixel
    detector line, and this function checks only the trajectory centerline
    against that detector. The ego footprint is deliberately NOT expanded here.

    Dense trajectory segments are sampled in pixel space so a fast motion step
    cannot jump over a one-pixel solid line between two rollout states.
    """
    lane_cfg = _cfg_get(cfg, "lane_constraints", {})
    enabled = _cfg_bool(lane_cfg, "enabled", True)

    empty = {
        "solid_lane_constraint_enabled": bool(enabled),
        "solid_lane_map_available": solid_lane_map is not None,
        "solid_lane_any_overlap": False,
        "solid_lane_crossing": False,
        "solid_lane_overlap_step_ratio": 0.0,
        "solid_lane_overlap_steps": 0,
        "solid_lane_first_overlap_step": -1,
        "solid_lane_max_overlapping_cells": 0,
        "solid_lane_hit_samples": 0,
        "solid_lane_hit_pixels": 0,
    }
    if not enabled or solid_lane_map is None:
        return empty

    lane = np.asarray(solid_lane_map, dtype=np.float32)
    dense_xy = np.asarray(rollout.get("dense_xy", []), dtype=np.float32)
    threshold = _cfg_float(lane_cfg, "blocked_threshold", 80.0)

    if lane.ndim != 2 or dense_xy.ndim != 2 or len(dense_xy) < 2:
        empty["solid_lane_map_available"] = lane.ndim == 2
        return empty

    lane_binary = lane >= threshold
    if not np.any(lane_binary):
        empty["solid_lane_map_available"] = True
        return empty

    path_pixels = local_points_to_pixels(dense_xy[:, :2], ego_center, meters_per_pixel)
    h, w = lane_binary.shape
    sample_step_px = max(_cfg_float(lane_cfg, "center_path_sample_step_px", 0.25), 0.05)
    ignore_initial_travel_m = max(_cfg_float(lane_cfg, "ignore_initial_travel_m", 0.0), 0.0)

    cumulative_m = 0.0
    hit_samples = 0
    hit_steps = set()
    hit_pixels = set()
    first_step = -1

    for step_idx in range(1, len(path_pixels)):
        p0 = np.asarray(path_pixels[step_idx - 1], dtype=np.float32)
        p1 = np.asarray(path_pixels[step_idx], dtype=np.float32)
        local_seg_m = float(np.linalg.norm(dense_xy[step_idx, :2] - dense_xy[step_idx - 1, :2]))
        cumulative_m += local_seg_m
        if cumulative_m < ignore_initial_travel_m:
            continue

        seg_len_px = float(np.linalg.norm(p1 - p0))
        n_samples = max(int(np.ceil(seg_len_px / sample_step_px)), 1)
        alpha = np.linspace(0.0, 1.0, n_samples + 1, dtype=np.float32)
        samples = p0[None, :] + alpha[:, None] * (p1 - p0)[None, :]

        cols = np.rint(samples[:, 0]).astype(np.int32)
        rows = np.rint(samples[:, 1]).astype(np.int32)
        valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        if not np.any(valid):
            continue

        cols_v = cols[valid]
        rows_v = rows[valid]
        hits = lane_binary[rows_v, cols_v]
        if not np.any(hits):
            continue

        hit_samples += int(np.count_nonzero(hits))
        hit_steps.add(int(step_idx))
        if first_step < 0:
            first_step = int(step_idx)
        for row, col in zip(rows_v[hits], cols_v[hits]):
            hit_pixels.add((int(row), int(col)))

    crossed = bool(hit_samples > 0)
    n_segments = max(len(path_pixels) - 1, 1)
    result = dict(empty)
    result.update({
        "solid_lane_map_available": True,
        # Keep the old key for output/API compatibility; its semantics are now
        # center-path crossing rather than ego-footprint overlap.
        "solid_lane_any_overlap": crossed,
        "solid_lane_crossing": crossed,
        "solid_lane_overlap_step_ratio": float(len(hit_steps) / n_segments),
        "solid_lane_overlap_steps": int(len(hit_steps)),
        "solid_lane_first_overlap_step": int(first_step),
        "solid_lane_max_overlapping_cells": 1 if crossed else 0,
        "solid_lane_hit_samples": int(hit_samples),
        "solid_lane_hit_pixels": int(len(hit_pixels)),
    })
    return result


def smoothness_score(rollout: Dict, cfg) -> Dict:
    controls = np.asarray(rollout.get("dense_controls", []), dtype=np.float32)
    speeds = np.asarray(rollout.get("dense_speed", []), dtype=np.float32)
    if len(controls) == 0:
        return {
            "mean_abs_acc": 0.0,
            "mean_abs_steer": 0.0,
            "mean_abs_steer_rate": 0.0,
            "max_abs_lateral_accel": 0.0,
            "max_abs_yaw_rate": 0.0,
        }
    steer = controls[:, 0]
    acc = controls[:, 1]
    dt = 1.0 / float(cfg.horizon.model_fps)
    steer_rate = np.diff(steer, prepend=steer[:1]) / dt
    lat_acc = (speeds ** 2) * np.tan(steer) / max(float(cfg.bicycle.wheelbase_m), 1e-3)
    yaw_rate = speeds * np.tan(steer) / max(float(cfg.bicycle.wheelbase_m), 1e-3)
    return {
        "mean_abs_acc": float(np.mean(np.abs(acc))),
        "mean_abs_steer": float(np.mean(np.abs(steer))),
        "mean_abs_steer_rate": float(np.mean(np.abs(steer_rate))),
        "max_abs_lateral_accel": float(np.max(np.abs(lat_acc))),
        "max_abs_yaw_rate": float(np.max(np.abs(yaw_rate))),
    }


def rollout_motion_summary(rollout: Dict) -> Dict:
    wp = np.asarray(rollout.get("waypoints", []), dtype=np.float32)
    speeds = np.asarray(rollout.get("speeds", []), dtype=np.float32)
    dense_speed = np.asarray(rollout.get("dense_speed", []), dtype=np.float32)
    if wp.ndim != 2 or len(wp) == 0:
        final_forward = 0.0
        final_lateral = 0.0
    else:
        final_forward = float(wp[-1, 0])
        final_lateral = float(wp[-1, 1])
    speed_arr = speeds if len(speeds) else dense_speed
    return {
        "final_forward_m": final_forward,
        "final_lateral_m": final_lateral,
        "mean_speed_mps": float(np.mean(speed_arr)) if len(speed_arr) else 0.0,
        "final_speed_mps": float(speed_arr[-1]) if len(speed_arr) else 0.0,
    }


def evaluate_candidate(
    candidate: Dict,
    base_route: np.ndarray,
    temporal_bundle: Dict,
    ego_center,
    meters_per_pixel: float,
    actor_timelines: Dict[int, List[Dict]],
    cfg,
    factor: Dict = None,) -> Dict:
    
    rollout = candidate["rollout"]
    # 占用检查
    occ = occupancy_bev_score(rollout, temporal_bundle.get("costmaps", []), ego_center, meters_per_pixel, cfg)
    # 红灯停止线检查：交通规则与物理占用保持独立。
    red_rule = red_light_constraint_score(
        rollout,
        temporal_bundle.get("red_light_maps", []),
        ego_center,
        meters_per_pixel,
        cfg,
    )
    # 停止标志检查：必须在进入停止控制区域前完成停车。
    stop_rule = stop_sign_constraint_score(
        rollout,
        temporal_bundle.get("stop_sign_maps", []),
        ego_center,
        meters_per_pixel,
        cfg,
    )
    # 实线检查，检查自车中心轨迹是否穿越实线
    lane = solid_lane_constraint_score(
        rollout,
        temporal_bundle.get("solid_lane_constraint", None),
        ego_center,
        meters_per_pixel,
        cfg,
    )
    # 动力学平滑性检查
    smooth = smoothness_score(rollout, cfg)
    motion = rollout_motion_summary(rollout)
    # 碰撞检查
    collision = (
        check_rollout_collisions(rollout, actor_timelines, cfg)
        if bool(cfg.collision.enabled)
        else {"collision_free": True, "num_collision_events": 0, "first_collision": None, "collision_events": []}
    )

    # A dedicated hold-current-stop candidate under an active red light is a
    # hard-rule response, not a motion proposal that should be rejected because
    # a future actor footprint later overlaps the already occupied ego pose.
    # Otherwise every candidate can become invalid and the planner may fall back
    # to an expert trajectory that starts moving while the current light is red.
    red_factor_for_hold = factor if isinstance(factor, dict) else {}
    red_stats_for_hold = red_factor_for_hold.get("red_light_rule") or {}
    active_red_hold = (
        str(red_factor_for_hold.get("type", "")) == "red_light_stop_line"
        and bool(red_stats_for_hold.get("current_red_active", False))
        and str(candidate.get("intent_name", "")) == "yield_stop"
        and str(candidate.get("variant_id", "")) == "hold_current_stop__hold"
    )

    # 计算候选轨迹与原始参考路线的
    wp = np.asarray(rollout["waypoints"], dtype=np.float32)
    route_dev = min_distance_to_polyline(wp, base_route)
    mean_dev = float(np.mean(route_dev)) if len(route_dev) else 0.0
    max_dev = float(np.max(route_dev)) if len(route_dev) else 0.0

    scoring_cfg = _cfg_get(cfg, "scoring", {})
    name = candidate["intent_name"]

    behavior_prior = 0.0
    if not candidate["intent"].get("active", False):
        behavior_prior += _cfg_float(scoring_cfg, "inactive_intent_penalty", 1000.0)
    if name in ["left_nudge", "right_nudge"]:
        behavior_prior += _cfg_float(scoring_cfg, "nudge_prior_cost", 1.0)
    if name in ["yield_stop", "emergency_brake"]:
        behavior_prior += _cfg_float(scoring_cfg, "stop_prior_cost", 5.0)

    clear_or_weak = _is_clear_or_weak_factor(factor or {})
    if clear_or_weak and name in ["cautious_follow", "creep", "yield_stop", "emergency_brake"]:
        behavior_prior += _cfg_float(scoring_cfg, "clear_scene_slow_intent_penalty", 80.0)
    if clear_or_weak and name in ["left_nudge", "right_nudge"]:
        behavior_prior += _cfg_float(scoring_cfg, "clear_scene_lateral_intent_penalty", 30.0)
    if clear_or_weak and name == "route_follow":
        behavior_prior -= _cfg_float(scoring_cfg, "clear_scene_route_follow_bonus", 10.0)

    occupancy_blocked_ratio_weight = _cfg_float(
        scoring_cfg,
        "occupancy_blocked_ratio_weight",
        _cfg_float(scoring_cfg, "hard_ratio_weight", 80.0),
    )
    occupancy_overlap_penalty = _cfg_float(scoring_cfg, "occupancy_overlap_penalty", 0.0)

    occupancy_score = occupancy_blocked_ratio_weight * occ["occupancy_blocked_ratio"]
    if occ["occupancy_any_blocked"]:
        occupancy_score += occupancy_overlap_penalty

    collision_penalty = 0.0 if collision["collision_free"] else _cfg_float(scoring_cfg, "collision_penalty", 5000.0)
    solid_lane_penalty = (
        _cfg_float(_cfg_get(cfg, "lane_constraints", {}), "crossing_penalty", 5000.0)
        if lane["solid_lane_any_overlap"]
        else 0.0
    )
    red_light_penalty = (
        _cfg_float(_cfg_get(cfg, "traffic_rules", {}), "crossing_penalty", 5000.0)
        if red_rule["red_light_any_overlap"]
        else 0.0
    )
    stop_sign_penalty = (
        _cfg_float(_cfg_get(cfg, "traffic_rules", {}), "crossing_penalty", 5000.0)
        if stop_rule["stop_sign_any_violation"]
        else 0.0
    )

    progress_reward_weight = _cfg_float(scoring_cfg, "progress_reward_weight", 0.15)
    progress_reward = -progress_reward_weight * max(0.0, motion["final_forward_m"])

    score = (
        occupancy_score
        + _cfg_float(scoring_cfg, "out_of_bounds_weight", 200.0) * occ["out_of_bounds_ratio"]
        + _cfg_float(scoring_cfg, "route_deviation_weight", 8.0) * mean_dev
        + _cfg_float(scoring_cfg, "route_deviation_max_weight", 0.0) * max_dev
        + _cfg_float(scoring_cfg, "acc_weight", 0.2) * smooth["mean_abs_acc"]
        + _cfg_float(scoring_cfg, "steer_weight", 1.0) * smooth["mean_abs_steer"]
        + _cfg_float(scoring_cfg, "steer_rate_weight", 0.2) * smooth["mean_abs_steer_rate"]
        + _cfg_float(scoring_cfg, "lat_acc_weight", 0.5) * smooth["max_abs_lateral_accel"]
        + _cfg_float(scoring_cfg, "yaw_rate_weight", 0.5) * smooth["max_abs_yaw_rate"]
        + behavior_prior
        + collision_penalty
        + solid_lane_penalty
        + red_light_penalty
        + stop_sign_penalty
        + progress_reward
    )

    allowed = True
    reasons = []
    if not candidate["intent"].get("active", False) and not bool(cfg.behaviors.allow_inactive_selection):
        allowed = False
        reasons.append("intent_inactive")

    # A current red light requires an explicit longitudinal response label.
    # The nominal reference candidate may be nearly stationary because the expert
    # was already stopped for another reason; selecting it as "route_follow" would
    # create a language-action mismatch even though it does not geometrically cross.
    red_factor = factor if isinstance(factor, dict) else {}
    red_stats = red_factor.get("red_light_rule") or {}
    if (
        str(red_factor.get("type", "")) == "red_light_stop_line"
        and bool(red_stats.get("current_red_active", False))
        and name == "route_follow"
    ):
        allowed = False
        reasons.append("current_red_light_requires_explicit_stop_response")

    stop_factor = factor if isinstance(factor, dict) else {}
    stop_stats = stop_factor.get("stop_sign_rule") or {}
    if (
        str(stop_factor.get("type", "")) == "stop_sign_control"
        and bool(stop_stats.get("current_stop_active", False))
        and name == "route_follow"
    ):
        allowed = False
        reasons.append("current_stop_sign_requires_explicit_stop_response")

    max_occ_ratio = _cfg_float(
        scoring_cfg,
        "max_occupancy_blocked_ratio",
        _cfg_float(scoring_cfg, "max_hard_ratio", 0.45),
    )
    reject_any_occ = _cfg_bool(scoring_cfg, "reject_any_occupancy_overlap", False)
    if not active_red_hold:
        if reject_any_occ and occ["occupancy_any_blocked"]:
            allowed = False
            reasons.append("occupancy_overlap")
        elif occ["occupancy_blocked_ratio"] > max_occ_ratio:
            allowed = False
            reasons.append(f"occupancy_blocked_ratio={occ['occupancy_blocked_ratio']:.3f}>{max_occ_ratio}")

    lane_cfg = _cfg_get(cfg, "lane_constraints", {})
    if _cfg_bool(lane_cfg, "enabled", True) and lane["solid_lane_any_overlap"]:
        allowed = False
        reasons.append("solid_lane_crossing")

    rule_cfg = _cfg_get(cfg, "traffic_rules", {})
    if _cfg_bool(rule_cfg, "red_light_enabled", True) and red_rule["red_light_any_overlap"]:
        allowed = False
        reasons.append("red_light_stop_line_crossing")
    if _cfg_bool(rule_cfg, "stop_sign_enabled", True) and stop_rule["stop_sign_any_violation"]:
        allowed = False
        reasons.append("stop_sign_region_entered_before_complete_stop")

    max_oob = _cfg_float(scoring_cfg, "max_out_of_bounds_ratio", 0.10)
    if occ["out_of_bounds_ratio"] > max_oob:
        allowed = False
        reasons.append(f"out_of_bounds_ratio={occ['out_of_bounds_ratio']:.3f}>{max_oob}")

    max_route_dev = _cfg_float(scoring_cfg, "max_route_deviation_m", 4.0)
    if max_dev > max_route_dev:
        allowed = False
        reasons.append(f"route_deviation={max_dev:.3f}>{max_route_dev}")

    max_lat_acc = _cfg_float(scoring_cfg, "max_lateral_accel_mps2", 5.0)
    if smooth["max_abs_lateral_accel"] > max_lat_acc:
        allowed = False
        reasons.append(f"lat_acc={smooth['max_abs_lateral_accel']:.3f}>{max_lat_acc}")

    max_yaw_rate = _cfg_float(scoring_cfg, "max_yaw_rate_radps", 1.0)
    if smooth["max_abs_yaw_rate"] > max_yaw_rate:
        allowed = False
        reasons.append(f"yaw_rate={smooth['max_abs_yaw_rate']:.3f}>{max_yaw_rate}")

    if bool(cfg.collision.reject_colliding_candidates) and not collision["collision_free"] and not active_red_hold:
        allowed = False
        reasons.append("footprint_collision")
    if name == "emergency_brake" and not bool(cfg.behaviors.allow_emergency_selection):
        allowed = False
        reasons.append("emergency_brake_not_selectable_by_default")
    if name == "yield_stop" and not bool(cfg.behaviors.allow_yield_selection):
        allowed = False
        reasons.append("yield_stop_not_selectable_by_default")

    info = {
        "intent_id": int(candidate["intent"]["intent_id"]),
        "hard_rule_hold_override": bool(active_red_hold),
        "intent_name": name,
        "intent_active": bool(candidate["intent"].get("active", False)),
        "intent_en": candidate["intent"].get("intent_en", ""),
        "activation_reason": candidate["intent"].get("activation_reason", ""),
        "reference_mode": candidate["reference_mode"],
        "variant_id": candidate.get("variant_id", "default"),
        "response_level": float(candidate.get("response_level", 0.0)),
        "response_object": candidate.get("response_object", {"exists": False}),
        "target_speed": float(candidate["target_speed"]),
        "target_speed_profile": np.asarray(candidate.get("target_speed_profile", rollout.get("target_speed_profile", [])), dtype=np.float32).tolist(),
        "score": float(score),
        "allowed": bool(allowed),
        "reasons": reasons,
        "mean_route_deviation": mean_dev,
        "max_route_deviation": max_dev,
        "behavior_prior_cost": float(behavior_prior),
        "progress_reward": float(progress_reward),
        "clear_or_weak_factor": bool(clear_or_weak),
        **motion,
        **occ,
        **red_rule,
        **stop_rule,
        **lane,
        **smooth,
        "collision": collision,
        "score_breakdown": {
            "occupancy": float(occupancy_score),
            "out_of_bounds": _cfg_float(scoring_cfg, "out_of_bounds_weight", 200.0) * occ["out_of_bounds_ratio"],
            "route_deviation_mean": _cfg_float(scoring_cfg, "route_deviation_weight", 8.0) * mean_dev,
            "route_deviation_max": _cfg_float(scoring_cfg, "route_deviation_max_weight", 0.0) * max_dev,
            "comfort": _cfg_float(scoring_cfg, "acc_weight", 0.2) * smooth["mean_abs_acc"] + _cfg_float(scoring_cfg, "steer_weight", 1.0) * smooth["mean_abs_steer"],
            "behavior_prior": float(behavior_prior),
            "collision_penalty": float(collision_penalty),
            "solid_lane_penalty": float(solid_lane_penalty),
            "red_light_penalty": float(red_light_penalty),
            "stop_sign_penalty": float(stop_sign_penalty),
            "progress_reward": float(progress_reward),
        },
    }
    return {**candidate, "info": info}


def select_best(scored: List[Dict], factor: Dict = None, cfg=None) -> int:
    """Select the final candidate.

    In a clear scene, a valid route-follow candidate should not lose simply
    because a slower cautious candidate has slightly smaller occupancy or
    smoothness terms.  Otherwise the generated labels become systematically
    slower than the expert trajectory.
    """
    allowed = [(i, c) for i, c in enumerate(scored) if c["info"].get("allowed", False)]
    if not allowed:
        return -1

    scoring_cfg = _cfg_get(cfg, "scoring", {}) if cfg is not None else {}
    prefer_route = _cfg_bool(scoring_cfg, "prefer_route_follow_when_clear", True)

    if prefer_route and _is_clear_or_weak_factor(factor or {}):
        max_occ = _cfg_float(
            scoring_cfg,
            "clear_scene_route_max_occupancy_blocked_ratio",
            _cfg_float(scoring_cfg, "max_occupancy_blocked_ratio", 0.01),
        )
        max_oob = _cfg_float(
            scoring_cfg,
            "clear_scene_route_max_out_of_bounds_ratio",
            _cfg_float(scoring_cfg, "max_out_of_bounds_ratio", 0.10),
        )
        for i, c in allowed:
            info = c.get("info", {})
            if info.get("intent_name") != "route_follow":
                continue
            occ_ratio = float(info.get("occupancy_blocked_ratio", 0.0))
            oob_ratio = float(info.get("out_of_bounds_ratio", 0.0))
            collision = info.get("collision", {}) or {}
            collision_free = bool(collision.get("collision_free", True))
            if occ_ratio <= max_occ and oob_ratio <= max_oob and collision_free:
                info["selection_override"] = "clear_scene_route_follow_priority"
                return int(i)

    best_idx = int(min(allowed, key=lambda x: x[1]["info"]["score"])[0])
    return best_idx



def trajectory_response_metrics(rollout: Dict, reference_rollout: Dict, cfg=None) -> Dict:
    """Measure the physical response relative to the counterfactual reference.

    Absolute changes are used for response magnitude, while signed changes are
    retained so language can distinguish true deceleration from merely limiting
    acceleration relative to the object-removed reference.
    """
    wp = np.asarray(rollout.get("waypoints", []), dtype=np.float32)
    ref = np.asarray(reference_rollout.get("waypoints", []), dtype=np.float32)
    n = min(len(wp), len(ref))
    if n <= 0:
        return {
            "mean_abs_longitudinal_change_m": 0.0,
            "mean_signed_longitudinal_change_m": 0.0,
            "mean_abs_lateral_change_m": 0.0,
            "mean_signed_lateral_change_m": 0.0,
            "max_abs_lateral_change_m": 0.0,
            "terminal_longitudinal_change_m": 0.0,
            "terminal_lateral_change_m": 0.0,
            "mean_abs_speed_change_mps": 0.0,
            "mean_signed_speed_change_mps": 0.0,
            "terminal_speed_change_mps": 0.0,
            "selected_start_speed_mps": 0.0,
            "selected_end_speed_mps": 0.0,
            "selected_speed_delta_mps": 0.0,
            "reference_start_speed_mps": 0.0,
            "reference_end_speed_mps": 0.0,
            "reference_speed_delta_mps": 0.0,
            "response_distance": 0.0,
        }

    delta = wp[:n, :2] - ref[:n, :2]
    speeds = np.asarray(rollout.get("speeds", []), dtype=np.float32).reshape(-1)
    ref_speeds = np.asarray(reference_rollout.get("speeds", []), dtype=np.float32).reshape(-1)
    ns = min(len(speeds), len(ref_speeds))
    if ns > 0:
        speed_delta = speeds[:ns] - ref_speeds[:ns]
        speed_change = float(np.mean(np.abs(speed_delta)))
        signed_speed_change = float(np.mean(speed_delta))
        terminal_speed_change = float(speed_delta[-1])
        selected_start_speed = float(speeds[0])
        selected_end_speed = float(speeds[ns - 1])
        reference_start_speed = float(ref_speeds[0])
        reference_end_speed = float(ref_speeds[ns - 1])
    else:
        speed_change = 0.0
        signed_speed_change = 0.0
        terminal_speed_change = 0.0
        selected_start_speed = float(speeds[0]) if len(speeds) else 0.0
        selected_end_speed = float(speeds[-1]) if len(speeds) else selected_start_speed
        reference_start_speed = float(ref_speeds[0]) if len(ref_speeds) else 0.0
        reference_end_speed = float(ref_speeds[-1]) if len(ref_speeds) else reference_start_speed

    long_mean = float(np.mean(np.abs(delta[:, 0])))
    signed_long_mean = float(np.mean(delta[:, 0]))
    lat_mean = float(np.mean(np.abs(delta[:, 1])))
    signed_lat_mean = float(np.mean(delta[:, 1]))
    lat_max = float(np.max(np.abs(delta[:, 1])))
    terminal_long = float(delta[-1, 0])
    terminal_lat = float(delta[-1, 1])

    mr = _cfg_get(cfg, "minimal_response", {}) if cfg is not None else {}
    response_distance = (
        _cfg_float(mr, "longitudinal_weight", 1.0) * long_mean
        + _cfg_float(mr, "lateral_weight", 2.0) * lat_mean
        + _cfg_float(mr, "max_lateral_weight", 0.5) * lat_max
        + _cfg_float(mr, "speed_weight", 0.2) * speed_change
    )
    return {
        "mean_abs_longitudinal_change_m": long_mean,
        "mean_signed_longitudinal_change_m": signed_long_mean,
        "mean_abs_lateral_change_m": lat_mean,
        "mean_signed_lateral_change_m": signed_lat_mean,
        "max_abs_lateral_change_m": lat_max,
        "terminal_longitudinal_change_m": terminal_long,
        "terminal_lateral_change_m": terminal_lat,
        "mean_abs_speed_change_mps": speed_change,
        "mean_signed_speed_change_mps": signed_speed_change,
        "terminal_speed_change_mps": terminal_speed_change,
        "selected_start_speed_mps": selected_start_speed,
        "selected_end_speed_mps": selected_end_speed,
        "selected_speed_delta_mps": float(selected_end_speed - selected_start_speed),
        "reference_start_speed_mps": reference_start_speed,
        "reference_end_speed_mps": reference_end_speed,
        "reference_speed_delta_mps": float(reference_end_speed - reference_start_speed),
        "response_distance": float(response_distance),
    }


def response_selection_objective(candidate: Dict, reference_rollout: Dict, cfg) -> float:
    metrics = trajectory_response_metrics(candidate.get("rollout", {}), reference_rollout, cfg=cfg)
    info = candidate.setdefault("info", {})
    info["response_metrics"] = metrics

    mr = _cfg_get(cfg, "minimal_response", {})
    comfort = (
        float(info.get("mean_abs_acc", 0.0))
        + _cfg_float(mr, "steer_comfort_scale", 1.0) * float(info.get("mean_abs_steer", 0.0))
        + _cfg_float(mr, "steer_rate_comfort_scale", 0.1) * float(info.get("mean_abs_steer_rate", 0.0))
    )
    reference_final = 0.0
    ref_wp = np.asarray(reference_rollout.get("waypoints", []), dtype=np.float32)
    if len(ref_wp):
        reference_final = float(ref_wp[-1, 0])
    progress_loss = max(0.0, reference_final - float(info.get("final_forward_m", 0.0)))

    objective = (
        float(metrics["response_distance"])
        + _cfg_float(mr, "comfort_weight", 0.05) * comfort
        + _cfg_float(mr, "progress_loss_weight", 0.05) * progress_loss
    )
    info["minimum_response_objective"] = float(objective)
    return float(objective)


def select_minimum_response(scored: List[Dict], reference_rollout: Dict, cfg, allowed_indices=None) -> int:
    """Choose the least intervention that already satisfies all hard constraints.

    This replaces the old idea of simply minimizing a global planner score.  A
    response is first required to be sufficient (``allowed=True``), then the
    selector chooses the valid trajectory closest to the object-removed
    reference behavior.
    """
    allowed_set = None if allowed_indices is None else set(int(i) for i in allowed_indices)
    valid = []
    for i, candidate in enumerate(scored):
        if allowed_set is not None and i not in allowed_set:
            continue
        if not candidate.get("info", {}).get("allowed", False):
            continue
        objective = response_selection_objective(candidate, reference_rollout, cfg)
        valid.append((i, objective, float(candidate.get("info", {}).get("score", 0.0))))

    if not valid:
        return -1

    # Existing planner score is only a deterministic tie breaker.  Safety has
    # already been enforced and response magnitude is the primary criterion.
    valid.sort(key=lambda x: (x[1], x[2], x[0]))
    idx = int(valid[0][0])
    scored[idx]["info"]["selection_override"] = "minimum_necessary_response"
    return idx

def refine_minimum_sufficient_response(
    selected: Dict,
    reference_rollout: Dict,
    base_route: np.ndarray,
    temporal_bundle: Dict,
    ego_center,
    meters_per_pixel: float,
    actor_timelines: Dict[int, List[Dict]],
    measurement: Dict,
    cfg,
    factor: Dict = None,
) -> Dict:
    """Refine a safe discrete response toward the first sufficient boundary.

    The discrete candidate bank is retained as a robust coarse search.  For a
    selected continuous response family, this function then scales that response
    from zero to the selected safe candidate, performs an ordered coarse scan to
    find the first unsafe->safe transition, and bisects only that local bracket.

    Every probe is rolled out again with the kinematic bicycle model and passed
    through exactly the same occupancy, actor-collision, traffic-rule, lane, and
    dynamics checks as ordinary candidates.  The returned response is therefore
    a safe upper approximation of the minimum sufficient boundary, not a direct
    interpolation of output waypoints.
    """
    mr = _cfg_get(cfg, "minimal_response", {})
    enabled = _cfg_bool(mr, "boundary_refinement_enabled", True)
    allowed_intents = set(_cfg_str_list(
        mr,
        "boundary_refinement_intents",
        ["cautious_follow", "left_nudge", "right_nudge"],
    ))
    name = str(selected.get("intent_name", ""))
    info = selected.setdefault("info", {})

    base_meta = {
        "enabled": bool(enabled),
        "applicable": bool(name in allowed_intents),
        "intent_name": name,
        "search_mode": "ordered_coarse_scan_then_local_bisection",
        "discrete_variant_id": str(info.get("variant_id", selected.get("variant_id", "default"))),
        "discrete_response_level": float(info.get("response_level", selected.get("response_level", 0.0))),
        "refined": False,
        "converged": False,
        "bracket_found": False,
        "num_evaluations": 0,
    }

    if not enabled:
        base_meta["reason"] = "disabled"
        info["minimum_response_boundary"] = base_meta
        return selected
    if name not in allowed_intents:
        base_meta["reason"] = "intent_has_no_continuous_response_strength"
        info["minimum_response_boundary"] = base_meta
        return selected
    if not bool(info.get("allowed", False)):
        base_meta["reason"] = "selected_candidate_not_safe"
        info["minimum_response_boundary"] = base_meta
        return selected

    coarse_steps = max(_cfg_int(mr, "boundary_refinement_coarse_steps", 8), 2)
    max_iterations = max(_cfg_int(mr, "boundary_refinement_max_iterations", 8), 0)
    tolerance = max(_cfg_float(mr, "boundary_refinement_scale_tolerance", 0.01), 1e-4)
    min_improvement = max(_cfg_float(mr, "boundary_refinement_min_scale_improvement", 0.01), 0.0)

    # A mathematically first-safe probe can sit directly on a discretization or
    # prediction boundary.  After the boundary is localized, optionally move a
    # small, width-aware distance farther toward the already-safe discrete
    # response and run the complete evaluator again.  The margin is expressed in
    # response-scale space so it remains comparable across longitudinal and
    # lateral response families.
    robust_margin_enabled = _cfg_bool(mr, "boundary_robust_margin_enabled", True)
    robust_margin_require_converged = _cfg_bool(mr, "boundary_robust_margin_require_converged", True)
    robust_margin_width_multiplier = max(
        _cfg_float(mr, "boundary_robust_margin_width_multiplier", 2.0),
        0.0,
    )
    robust_margin_min_scale = max(
        _cfg_float(mr, "boundary_robust_margin_min_scale", 0.01),
        0.0,
    )
    robust_margin_max_scale = max(
        _cfg_float(mr, "boundary_robust_margin_max_scale", 0.05),
        robust_margin_min_scale,
    )

    cache = {1.0: selected}
    evaluations = 0

    # Ensure the selected endpoint carries the same response objective metadata
    # as every generated probe.
    response_selection_objective(selected, reference_rollout, cfg)

    def evaluate_scale(scale: float):
        nonlocal evaluations
        key = round(float(np.clip(scale, 0.0, 1.0)), 10)
        if key in cache:
            return cache[key]
        candidate = build_boundary_refined_candidate(
            selected_candidate=selected,
            base_route=base_route,
            reference_rollout=reference_rollout,
            measurement=measurement,
            cfg=cfg,
            response_scale=key,
        )
        if candidate is None:
            cache[key] = None
            return None
        scored_candidate = evaluate_candidate(
            candidate,
            base_route,
            temporal_bundle,
            ego_center,
            meters_per_pixel,
            actor_timelines,
            cfg,
            factor=factor,
        )
        response_selection_objective(scored_candidate, reference_rollout, cfg)
        evaluations += 1
        cache[key] = scored_candidate
        return scored_candidate

    zero = evaluate_scale(0.0)
    if zero is None:
        base_meta.update({
            "reason": "failed_to_build_zero_response_probe",
            "num_evaluations": int(evaluations),
        })
        info["minimum_response_boundary"] = base_meta
        return selected

    if bool(zero.get("info", {}).get("allowed", False)):
        # Without an unsafe lower endpoint there is no positive safety boundary
        # to bisect.  Keep the discrete result but explicitly avoid claiming a
        # proven minimum; this case is useful for diagnosing semantic filters or
        # non-safety selection effects.
        base_meta.update({
            "reason": "zero_response_probe_already_safe",
            "zero_response_already_safe": True,
            "num_evaluations": int(evaluations),
            "zero_response_objective": float(zero.get("info", {}).get("minimum_response_objective", 0.0)),
        })
        info["minimum_response_boundary"] = base_meta
        return selected

    # Ordered scan is intentionally used before bisection.  It finds the first
    # sufficient interval and does not assume that safety is globally monotonic
    # over large lateral-response ranges.
    scan_scales = np.linspace(0.0, 1.0, coarse_steps + 1, dtype=np.float32)
    previous_scale = 0.0
    previous_candidate = zero
    low_scale = None
    low_candidate = None
    high_scale = None
    high_candidate = None

    for raw_scale in scan_scales[1:]:
        scale = float(raw_scale)
        candidate = selected if abs(scale - 1.0) <= 1e-9 else evaluate_scale(scale)
        if candidate is None:
            previous_scale = scale
            previous_candidate = None
            continue
        if bool(candidate.get("info", {}).get("allowed", False)):
            if previous_candidate is not None and not bool(previous_candidate.get("info", {}).get("allowed", False)):
                low_scale = float(previous_scale)
                low_candidate = previous_candidate
                high_scale = float(scale)
                high_candidate = candidate
                break
        previous_scale = scale
        previous_candidate = candidate

    if low_scale is None or high_scale is None or high_candidate is None:
        base_meta.update({
            "reason": "no_local_unsafe_to_safe_transition_found",
            "num_evaluations": int(evaluations),
        })
        info["minimum_response_boundary"] = base_meta
        return selected

    iterations = 0
    while iterations < max_iterations and (high_scale - low_scale) > tolerance:
        mid = 0.5 * (low_scale + high_scale)
        candidate = evaluate_scale(mid)
        if candidate is None:
            break
        if bool(candidate.get("info", {}).get("allowed", False)):
            high_scale = float(mid)
            high_candidate = candidate
        else:
            low_scale = float(mid)
            low_candidate = candidate
        iterations += 1

    converged = bool((high_scale - low_scale) <= tolerance)
    boundary_width = float(high_scale - low_scale)
    boundary_improvement = float(1.0 - high_scale)
    lower_info = (low_candidate or {}).get("info", {}) if isinstance(low_candidate, dict) else {}

    margin_requested = bool(
        robust_margin_enabled
        and (converged or not robust_margin_require_converged)
        and boundary_improvement >= min_improvement
    )
    margin_amount = 0.0
    margin_target_scale = float(high_scale)
    margin_candidate = None
    margin_validation_passed = False
    margin_fallback_to_boundary = False
    margin_reason = "disabled"

    if not robust_margin_enabled:
        margin_reason = "disabled"
    elif robust_margin_require_converged and not converged:
        margin_reason = "boundary_not_converged"
    elif boundary_improvement < min_improvement:
        margin_reason = "no_material_boundary_improvement"
    else:
        margin_amount = float(np.clip(
            boundary_width * robust_margin_width_multiplier,
            robust_margin_min_scale,
            robust_margin_max_scale,
        ))
        margin_target_scale = float(min(1.0, high_scale + margin_amount))
        margin_amount = float(max(0.0, margin_target_scale - high_scale))

        if margin_amount <= 1e-12:
            margin_reason = "no_remaining_scale_room"
            margin_candidate = high_candidate
            margin_validation_passed = True
        else:
            margin_candidate = evaluate_scale(margin_target_scale)
            margin_validation_passed = bool(
                isinstance(margin_candidate, dict)
                and margin_candidate.get("info", {}).get("allowed", False)
            )
            if margin_validation_passed:
                margin_reason = "validated"
            else:
                margin_reason = "validation_failed_fallback_to_boundary"
                margin_fallback_to_boundary = True

    final_scale = float(margin_target_scale if margin_validation_passed else high_scale)
    final_candidate = margin_candidate if margin_validation_passed and margin_candidate is not None else high_candidate
    final_improvement = float(1.0 - final_scale)

    metadata = {
        **base_meta,
        "reason": "boundary_refined_with_robust_margin" if final_improvement >= min_improvement and margin_validation_passed and margin_amount > 0.0
                  else "boundary_refined" if final_improvement >= min_improvement
                  else "boundary_checked_no_material_improvement",
        "bracket_found": True,
        "converged": converged,
        "refined": bool(final_improvement >= min_improvement),
        "coarse_steps": int(coarse_steps),
        "bisection_iterations": int(iterations),
        "num_evaluations": int(evaluations),
        "scale_tolerance": float(tolerance),
        "lower_unsafe_scale": float(low_scale),
        "upper_safe_scale": float(high_scale),
        "boundary_width": float(boundary_width),
        "boundary_scale_improvement_from_discrete": float(boundary_improvement),
        "final_selected_scale": float(final_scale),
        "scale_improvement_from_discrete": float(final_improvement),
        "lower_unsafe_reasons": list(lower_info.get("reasons", [])),
        "robust_margin_enabled": bool(robust_margin_enabled),
        "robust_margin_require_converged": bool(robust_margin_require_converged),
        "robust_margin_requested": bool(margin_requested),
        "robust_margin_width_multiplier": float(robust_margin_width_multiplier),
        "robust_margin_min_scale": float(robust_margin_min_scale),
        "robust_margin_max_scale": float(robust_margin_max_scale),
        "robust_margin_scale": float(margin_amount),
        "robust_margin_target_scale": float(margin_target_scale),
        "robust_margin_validation_passed": bool(margin_validation_passed),
        "robust_margin_fallback_to_boundary": bool(margin_fallback_to_boundary),
        "robust_margin_applied": bool(
            final_improvement >= min_improvement
            and margin_validation_passed
            and margin_amount > 0.0
        ),
        "robust_margin_reason": str(margin_reason),
        "refined_response_level": float(selected.get("response_level", 0.0)) * float(final_scale),
    }

    if final_improvement < min_improvement:
        # The robust-margin probe may be safe but still too close to the original
        # discrete response to justify replacing it.  In that case the actual
        # returned trajectory is the original selected candidate (scale=1.0), so
        # every final-selection field must describe that real trajectory rather
        # than the tested-but-unused margin probe.
        metadata.update({
            "reason": "boundary_checked_no_material_improvement",
            "refined": False,
            "final_selected_scale": 1.0,
            "scale_improvement_from_discrete": 0.0,
            "robust_margin_applied": False,
            "refined_response_level": float(
                info.get("response_level", selected.get("response_level", 0.0))
            ),
        })
        if margin_validation_passed and margin_amount > 0.0:
            metadata["robust_margin_reason"] = (
                "validated_but_not_applied_no_material_improvement"
            )
        info["minimum_response_boundary"] = metadata
        info["selection_override"] = "minimum_sufficient_response_boundary_checked"
        return selected

    refined = final_candidate
    refined_info = refined.setdefault("info", {})
    metadata["num_evaluations"] = int(evaluations)
    metadata["refined_response_level"] = float(refined_info.get("response_level", refined.get("response_level", 0.0)))
    refined_info["minimum_response_boundary"] = metadata
    refined_info["selection_override"] = (
        "minimum_sufficient_response_boundary_refined_with_robust_margin"
        if margin_validation_passed and margin_amount > 0.0
        else "minimum_sufficient_response_boundary_refined"
    )
    # Refresh objective metadata after the final robust candidate is chosen.
    response_selection_objective(refined, reference_rollout, cfg)
    return refined
