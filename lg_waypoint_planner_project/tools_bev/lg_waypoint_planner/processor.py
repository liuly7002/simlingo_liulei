# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from .logger import LOGGER
from .io_utils import array_to_list, find_route_dirs, list_frame_names, load_costmap, load_json_gz, save_json_gz
from .dataset import (
    get_ego_center,
    get_meters_per_pixel,
    get_pose_global_xy_yaw,
    get_route,
    load_future_ego_waypoints,
    offset_frame_name,
)
from .geometry import cumulative_distance, remove_duplicate_points, resample_polyline, sample_polyline_by_s
from .costmap import build_temporal_costmaps, build_temporal_red_light_constraints, build_temporal_stop_sign_constraints, build_traffic_light_state_context
from .actor_loader import load_current_actor_records, load_future_actor_timelines
from .footprint_collision import check_rollout_collisions
from .critical_factor import identify_critical_factor, align_factor_with_selected_waypoints
from .intent_policy import INTENT_NAMES, infer_intents
from .evaluator import select_minimum_response, refine_minimum_sufficient_response, solid_lane_constraint_score, red_light_constraint_score, stop_sign_constraint_score
from .causal_response import (
    build_causal_candidate_pool,
    evaluate_candidate_pool,
    analyze_causal_objects,
    revalidate_causal_analysis,
    causal_consistent_candidate_indices,
    build_causal_factor,
    build_response_supervision,
    compact_causal_analysis,
)
from .language import (
    build_language_annotation,
    describe_factor_en,
    describe_driving_intent_en,
    describe_waypoint_shape_en,
    describe_driving_intent_name,
)
from .visualizer import save_bev_debug, save_rgb_waypoints_debug_image


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    try:
        return getattr(obj, key)
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _cfg_float(obj, key: str, default: float) -> float:
    try:
        return float(_cfg_get(obj, key, default))
    except Exception:
        return float(default)


def _cfg_bool(obj, key: str, default: bool) -> bool:
    value = _cfg_get(obj, key, default)
    if isinstance(value, str):
        return value.lower() in ["1", "true", "yes", "y", "on"]
    return bool(value)


def _required_reference_horizon_m(measurement: Dict, cfg) -> float:
    """Return the route length needed by the current prediction horizon.

    The reference must cover both the farthest physically reachable ego-center
    position and the pure-pursuit lookahead needed at that position.  Otherwise a
    fast but valid rollout can run beyond the finite route and be rejected as a
    route-deviation error.
    """
    ref_cfg = _cfg_get(cfg, "reference", {})
    minimum_horizon = max(_cfg_float(ref_cfg, "horizon_m", 30.0), 0.1)
    if not _cfg_bool(ref_cfg, "dynamic_horizon_enabled", True):
        return float(minimum_horizon)

    num_wp = max(int(_cfg_get(_cfg_get(cfg, "horizon", {}), "num_future_waypoints", 1)), 1)
    future_fps = max(_cfg_float(_cfg_get(cfg, "horizon", {}), "future_fps", 1.0), 1e-6)
    horizon_s = float(num_wp) / future_fps

    bicycle_cfg = _cfg_get(cfg, "bicycle", {})
    max_speed = max(_cfg_float(bicycle_cfg, "max_speed_mps", 22.0), 0.0)
    max_accel = max(_cfg_float(bicycle_cfg, "max_accel_mps2", 2.5), 0.0)
    current_speed = float(np.clip(float(measurement.get("speed", 0.0)), 0.0, max_speed))

    if max_accel <= 1e-6 or current_speed >= max_speed - 1e-6:
        reachable_m = current_speed * horizon_s
    else:
        accel_time = min(horizon_s, (max_speed - current_speed) / max_accel)
        reachable_m = (
            current_speed * accel_time
            + 0.5 * max_accel * accel_time * accel_time
            + max_speed * max(0.0, horizon_s - accel_time)
        )

    lookahead_m = max(_cfg_float(bicycle_cfg, "lookahead_max_m", 8.0), 0.0)
    extra_buffer_m = max(_cfg_float(ref_cfg, "dynamic_extra_buffer_m", 2.0), 0.0)
    return float(max(minimum_horizon, reachable_m + lookahead_m + extra_buffer_m))


def _measurement_route_without_ego_insertion(measurement: Dict, route_key: str) -> np.ndarray:
    """Return saved route points exactly as stored in one measurement frame.

    ``get_route`` intentionally prepends the ego origin when the first saved route
    point is farther than 0.5 m.  That is correct for the current frame, but it must
    not be done for a future frame used only to extend the current reference: the
    future ego position is not itself a route checkpoint and inserting it would
    bend the stitched route toward the expert trajectory.
    """
    if route_key in measurement:
        route = measurement[route_key]
    elif "route" in measurement:
        route = measurement["route"]
    elif "route_original" in measurement:
        route = measurement["route_original"]
    else:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.asarray(route, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    pts = pts[:, :2]
    pts = pts[np.isfinite(pts).all(axis=1)]
    return remove_duplicate_points(pts) if len(pts) else np.zeros((0, 2), dtype=np.float32)


def _future_local_route_to_current_frame(
    future_route_local: np.ndarray,
    future_pose,
    current_pose,
) -> np.ndarray:
    """Transform a future frame's ego-local route into the current ego frame."""
    pts = np.asarray(future_route_local, dtype=np.float32)[:, :2]
    future_global, future_yaw = future_pose
    current_global, current_yaw = current_pose

    cf, sf = np.cos(float(future_yaw)), np.sin(float(future_yaw))
    global_x = float(future_global[0]) + cf * pts[:, 0] - sf * pts[:, 1]
    global_y = float(future_global[1]) + sf * pts[:, 0] + cf * pts[:, 1]

    dx = global_x - float(current_global[0])
    dy = global_y - float(current_global[1])
    cc, sc = np.cos(float(current_yaw)), np.sin(float(current_yaw))
    current_x = dx * cc + dy * sc
    current_y = -dx * sc + dy * cc
    return np.stack([current_x, current_y], axis=1).astype(np.float32)


def _build_reference_route_from_measurements(
    route_dir: Path,
    frame_name: str,
    current_measurement: Dict,
    required_horizon_m: float,
    cfg,
):
    """Build the current reference from real routes saved in measurement frames.

    The current frame's route is kept as the authoritative prefix.  Only when its
    true arc length is shorter than the dynamically required horizon do we inspect
    later measurement frames.  Their saved local routes are transformed through
    global coordinates into the current ego frame, aligned to the current stitched
    tail, and only the non-overlapping real tail is appended.  No geometric
    extrapolation is performed.

    Returns ``(reference_route, info)``.  ``reference_route`` is ``None`` when the
    available real route cannot cover the required horizon and the configured
    policy is to skip that frame.
    """
    ref_cfg = _cfg_get(cfg, "reference", {})
    route_key = str(_cfg_get(_cfg_get(cfg, "paths", {}), "route_key", "route"))
    spacing_m = max(_cfg_float(ref_cfg, "spacing_m", 0.5), 1e-3)
    required_horizon_m = max(float(required_horizon_m), spacing_m)

    merged = remove_duplicate_points(get_route(current_measurement, route_key))
    current_length = float(cumulative_distance(merged)[-1]) if len(merged) else 0.0
    info = {
        "required_horizon_m": float(required_horizon_m),
        "initial_route_length_m": float(current_length),
        "stitched_route_length_m": float(current_length),
        "future_frames_used": [],
        "used_future_measurements": False,
        "coverage_complete": bool(current_length + 1e-6 >= required_horizon_m),
    }

    if info["coverage_complete"]:
        return resample_polyline(merged, spacing_m=spacing_m, horizon_m=required_horizon_m), info

    if not _cfg_bool(ref_cfg, "stitch_future_measurements", True):
        info["failure_reason"] = "future_measurement_stitching_disabled"
        return None, info

    current_pose = get_pose_global_xy_yaw(current_measurement)
    if current_pose is None:
        info["failure_reason"] = "missing_current_global_pose"
        return None, info

    horizon_cfg = _cfg_get(cfg, "horizon", {})
    default_max_frames = max(int(_cfg_get(horizon_cfg, "num_future_waypoints", 1)), 1)
    max_future_frames = max(int(_cfg_get(ref_cfg, "future_measurement_max_frames", default_max_frames)), 1)
    frame_stride = max(int(_cfg_get(horizon_cfg, "future_frame_stride", 1)), 1)
    join_tolerance_m = max(_cfg_float(ref_cfg, "future_route_join_tolerance_m", 3.0), spacing_m)
    min_extension_gain_m = max(_cfg_float(ref_cfg, "future_route_min_extension_gain_m", 0.05), 1e-3)

    measurement_dir = route_dir / cfg.paths.measurements_folder
    stitched_length = current_length

    for k in range(1, max_future_frames + 1):
        future_name = offset_frame_name(frame_name, k, frame_stride)
        if future_name is None:
            break
        future_path = measurement_dir / f"{future_name}.json.gz"
        if not future_path.exists():
            break

        future_measurement = load_json_gz(future_path)
        future_pose = get_pose_global_xy_yaw(future_measurement)
        if future_pose is None:
            continue
        future_route_local = _measurement_route_without_ego_insertion(future_measurement, route_key)
        if len(future_route_local) < 2:
            continue

        future_route_current = _future_local_route_to_current_frame(
            future_route_local,
            future_pose,
            current_pose,
        )
        future_route_current = remove_duplicate_points(future_route_current)
        if len(future_route_current) < 2:
            continue

        tail = merged[-1]
        distances = np.linalg.norm(future_route_current - tail[None, :], axis=1)
        join_idx = int(np.argmin(distances))
        join_distance = float(distances[join_idx])
        if join_distance > join_tolerance_m:
            continue

        append_tail = future_route_current[join_idx + 1 :]
        if len(append_tail) == 0:
            continue

        candidate = remove_duplicate_points(np.concatenate([merged, append_tail], axis=0))
        candidate_length = float(cumulative_distance(candidate)[-1]) if len(candidate) else 0.0
        if candidate_length <= stitched_length + min_extension_gain_m:
            continue

        merged = candidate
        stitched_length = candidate_length
        info["future_frames_used"].append(str(future_name))
        info["used_future_measurements"] = True
        info["stitched_route_length_m"] = float(stitched_length)

        if stitched_length + 1e-6 >= required_horizon_m:
            info["coverage_complete"] = True
            break

    if not info["coverage_complete"]:
        info["failure_reason"] = "insufficient_real_route_coverage"
        info["shortfall_m"] = float(max(required_horizon_m - stitched_length, 0.0))
        if _cfg_bool(ref_cfg, "skip_if_insufficient_real_route", True):
            return None, info

        # Optional conservative fallback for debugging only: use exactly the real
        # route that exists.  The default configuration skips instead so a finite
        # endpoint cannot create false route-deviation penalties.
        effective_horizon = max(min(required_horizon_m, stitched_length), spacing_m)
        return resample_polyline(merged, spacing_m=spacing_m, horizon_m=effective_horizon), info

    reference_route = resample_polyline(
        merged,
        spacing_m=spacing_m,
        horizon_m=required_horizon_m,
    )
    return reference_route, info


def _pad_waypoints(points: np.ndarray, n: int, reference_route: np.ndarray, cfg) -> np.ndarray:
    if points is None or len(points) == 0:
        q = np.arange(1, n + 1, dtype=np.float32) * float(cfg.reference.spacing_m)
        points = sample_polyline_by_s(reference_route, q)
    points = np.asarray(points, dtype=np.float32)[:, :2]
    if len(points) >= n:
        return points[:n]
    pad = np.repeat(points[-1:], n - len(points), axis=0)
    return np.concatenate([points, pad], axis=0).astype(np.float32)


def build_fallback(reference_route: np.ndarray, expert_future, cfg, reason: str) -> Dict:
    n = int(cfg.horizon.num_future_waypoints)
    waypoints = _pad_waypoints(expert_future, n, reference_route, cfg)
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), waypoints[:-1]], axis=0)
    delta = waypoints - prev
    yaws = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)
    speeds = (np.linalg.norm(delta, axis=1) * float(cfg.horizon.future_fps)).astype(np.float32)
    controls = np.zeros((n, 2), dtype=np.float32)
    rollout = {
        "waypoints": waypoints,
        "yaws": yaws,
        "speeds": speeds,
        "controls": controls,
        "dense_xy": waypoints,
        "dense_yaw": yaws,
        "dense_speed": speeds,
        "dense_controls": controls,
        "target_speed": float(speeds[0]) if len(speeds) else 0.0,
    }
    info = {
        "intent_id": -1,
        "intent_name": "expert_fallback",
        "intent_active": False,
        "intent_en": "use expert fallback",
        "activation_reason": reason,
        "reference_mode": "expert_reference_or_route_sample",
        "target_speed": float(rollout["target_speed"]),
        "score": 1e9,
        "allowed": False,
        "reasons": [reason],
        "fallback_to_expert": True,
        "fallback_reason": reason,
        "collision": {"collision_free": False, "num_collision_events": 0, "first_collision": None, "collision_events": []},
        "mean_cost": 0.0, "max_cost": 0.0, "hard_ratio": 0.0,
        "out_of_bounds_ratio": 0.0, "mean_route_deviation": 0.0, "max_route_deviation": 0.0,
        "mean_abs_acc": 0.0, "mean_abs_steer": 0.0, "mean_abs_steer_rate": 0.0,
        "max_abs_lateral_accel": 0.0, "max_abs_yaw_rate": 0.0,
    }
    return {
        "intent_name": "expert_fallback",
        "intent": {"intent_id": -1, "intent_name": "expert_fallback", "active": False},
        "reference_route": reference_route,
        "reference_mode": "expert_reference_or_route_sample",
        "target_speed": rollout["target_speed"],
        "rollout": rollout,
        "info": info,
    }


def valid_intent_mask(scored: List[Dict]) -> List[int]:
    mask = [0 for _ in INTENT_NAMES]
    for c in scored:
        iid = int(c["info"].get("intent_id", -1))
        if 0 <= iid < len(mask) and c["info"].get("allowed", False):
            mask[iid] = 1
    return mask


def _clean_actor(actor: Dict) -> Dict:
    if not actor or not actor.get("exists", False):
        return {"exists": False}
    out = {
        "exists": True,
        "class": actor.get("class", "unknown"),
        "relative_position": actor.get("relative_position", "unknown"),
    }
    for key in [
        "distance_m",
        "x_m",
        "y_m",
        "speed_mps",
        "route_longitudinal_m",
        "route_lateral_offset_m",
        "heading_difference_deg",
    ]:
        if key in actor:
            try:
                out[key] = round(float(actor[key]), 3)
            except Exception:
                pass
    if actor.get("heading_relation", None) is not None:
        out["heading_relation"] = actor.get("heading_relation")
    if actor.get("id", None) is not None:
        out["id"] = actor.get("id")
    return out


def concise_factor_json(factor: Dict) -> Dict:
    actor = _clean_actor(factor.get("critical_actor") or {})
    secondary_actor = _clean_actor(factor.get("secondary_attention_actor") or {})
    return {
        "type": factor.get("type", "unknown"),
        "critical_actor": actor,
        "secondary_attention_actor": secondary_actor,
        "reason_en": describe_factor_en(factor),
    }


def _user_reasons(info: Dict) -> List[str]:
    # Public-facing reason list.  Do not expose internal generation strategies
    # such as expert fallback as a driving intent or user-visible explanation.
    if bool(info.get("red_light_any_overlap", False)) or "red_light_stop_line_crossing" in info.get("reasons", []):
        return ["This motion would cross an active red-light stop line."]
    if bool(info.get("stop_sign_any_violation", False)) or "stop_sign_region_entered_before_complete_stop" in info.get("reasons", []):
        return ["This motion would enter the stop-sign control region before completing a stop."]
    if bool(info.get("solid_lane_any_overlap", False)) or "solid_lane_crossing" in info.get("reasons", []):
        return ["This motion crosses a solid lane boundary."]
    if not bool(info.get("collision", {}).get("collision_free", True)) and not bool(info.get("hard_rule_hold_override", False)):
        return ["Too close to nearby objects."]
    if not bool(info.get("allowed", False)) and not bool(info.get("fallback_to_expert", False)):
        return ["This motion is not suitable for the current scene."]
    return []


def concise_candidate_json(
    c: Dict,
    selected: bool,
    include_waypoints: bool = True,
    factor: Dict = None,
) -> Dict:
    info = c["info"]
    internal_name = info.get("intent_name", "unknown")
    semantic_factor = factor if selected else None
    name = describe_driving_intent_name(c, semantic_factor)
    user_intent_en = describe_driving_intent_en(c, semantic_factor)
    waypoint_shape_en = describe_waypoint_shape_en(c, semantic_factor)
    out = {
        "intent_id": int(info.get("intent_id", -1)),
        "intent_name": name,
        "internal_intent_name": internal_name,
        "variant_id": info.get("variant_id", c.get("variant_id", "default")),
        "response_level": float(info.get("response_level", c.get("response_level", 0.0))),
        "response_object": info.get("response_object", c.get("response_object", {"exists": False})),
        # User-facing intent text.  For internal fallback candidates this is
        # inferred from the waypoint shape, so the public JSON will not say
        # "expert fallback" as if it were a driving intent.
        "intent_en": user_intent_en,
        "selected": bool(selected),
        "active": bool(info.get("intent_active", True)),
        "valid": bool(info.get("allowed", False)),
        "solid_lane_crossing": bool(info.get("solid_lane_any_overlap", False)),
        "waypoint_shape_en": waypoint_shape_en,
        "user_reasons": _user_reasons(info),
        "response_metrics": info.get("response_metrics", None),
        "minimum_response_boundary": info.get("minimum_response_boundary", None),
    }
    if include_waypoints:
        out["waypoints"] = array_to_list(c["rollout"]["waypoints"])
    return out


def build_public_output(
    frame_name: str,
    factor: Dict,
    intents: List[Dict],
    scored: List[Dict],
    selected_idx: int,
    selected: Dict,
    selected_info: Dict,
    selected_rollout: Dict,
    reference_route: np.ndarray,
    reference_route_info: Dict,
    expert_future,
    language: Dict,
    causal_analysis: Dict,
    response_supervision: Dict,
    reference_rollout: Dict,
    cfg,
    debug_paths: Dict,
) -> Dict:
    candidate_items = [
        concise_candidate_json(
            c,
            selected=(i == selected_idx),
            include_waypoints=bool(_cfg_get(_cfg_get(cfg, "output", {}), "include_candidate_waypoints", True)),
            factor=factor,
        )
        for i, c in enumerate(scored)
    ]

    out = {
        "frame": frame_name,
        "generator": "causal_minimum_response_waypoint_generator_v3",
        "coordinate": "ego_local_x_forward_y_right_yaw_positive_right",
        "core_questions": {
            "most_influential_factor": concise_factor_json(factor),
            "driving_intent": {
                "intent_id": int(selected_info.get("intent_id", -1)),
                "intent_name": describe_driving_intent_name(selected, factor),
                "internal_intent_name": selected_info.get("intent_name", "unknown"),
                "intent_en": describe_driving_intent_en(selected, factor),
            },
            "waypoint_shape": {
                "en": describe_waypoint_shape_en(selected, factor),
            },
            "trajectory_quality": response_supervision.get("quality", {}),
        },
        "language_annotation": language,
        "causal_analysis": compact_causal_analysis(causal_analysis),
        "supervision": {
            "selector_label": int(selected_info.get("intent_id", -1)) if selected_info.get("allowed", False) else -1,
            "risk_label_valid": bool(selected_info.get("allowed", False)),
            "valid_intent_mask": valid_intent_mask(scored),
            "selected_index": int(selected_idx),
            "selected_intent_id": int(selected_info.get("intent_id", -1)),
            "selected_intent_name": describe_driving_intent_name(selected, factor),
            "selected_internal_intent_name": selected_info.get("intent_name", "unknown"),
            "selected_intent_en": describe_driving_intent_en(selected, factor),
            "risk_planned_waypoints": array_to_list(selected_rollout["waypoints"]),
            "risk_planned_speeds": array_to_list(selected_rollout["speeds"]),
            "risk_planned_yaws": array_to_list(selected_rollout["yaws"]),
            "reference_response": response_supervision,
        },
        "candidate_waypoints": candidate_items,
        "reference": {
            "source": response_supervision.get("reference_source", "nominal_no_interference"),
            "object_removed_reference_waypoints": array_to_list(reference_rollout.get("waypoints", [])),
            "object_removed_reference_speeds": array_to_list(reference_rollout.get("speeds", [])),
            "expert_future_waypoints": array_to_list(expert_future) if expert_future is not None else None,
            "selected_reference_route": array_to_list(selected["reference_route"]),
            "expert_reference_route": array_to_list(reference_route) if bool(_cfg_get(_cfg_get(cfg, "output", {}), "include_reference_route", False)) else None,
        },
        "debug_files": debug_paths,
    }
    if not bool(_cfg_get(_cfg_get(cfg, "output", {}), "include_expert_reference_route", False)):
        out["reference"].pop("expert_reference_route", None)

    # Optional lightweight diagnostics for validating dynamic reference coverage
    # and real-route stitching.  The whole block is omitted when disabled so the
    # normal training JSON remains unchanged and compact.
    debug_cfg = _cfg_get(cfg, "debug", {})
    if _cfg_bool(debug_cfg, "save_reference_stitch_debug", False):
        final_reference_length_m = (
            float(cumulative_distance(reference_route)[-1]) if len(reference_route) else 0.0
        )
        future_frames_used = [str(x) for x in reference_route_info.get("future_frames_used", [])]
        out["reference_route_debug"] = {
            "reference_required_horizon_m": float(reference_route_info.get("required_horizon_m", 0.0)),
            "current_route_length_m": float(reference_route_info.get("initial_route_length_m", 0.0)),
            "stitched_route_length_m": float(reference_route_info.get("stitched_route_length_m", 0.0)),
            "final_reference_length_m": float(final_reference_length_m),
            "reference_point_count": int(len(reference_route)),
            "reference_stitch_used": bool(reference_route_info.get("used_future_measurements", False)),
            "stitched_future_frame_count": int(len(future_frames_used)),
            "future_frames_used": future_frames_used,
            "coverage_complete": bool(reference_route_info.get("coverage_complete", False)),
        }
    return out


def candidate_to_full_json(c: Dict, selected: bool) -> Dict:
    r = c["rollout"]
    info = dict(c["info"])
    return {
        **info,
        "selected": bool(selected),
        "reference_route": array_to_list(c["reference_route"]),
        "waypoints": array_to_list(r["waypoints"]),
        "yaws": array_to_list(r["yaws"]),
        "speeds": array_to_list(r["speeds"]),
        "controls_steer_acc": array_to_list(r["controls"]),
    }


def build_full_debug_output(frame_name, mpp, ego_center, temporal_bundle, factor, current_actors, scored, selected_idx, selected_info, selected_rollout, selected, reference_route, expert_future, language, causal_analysis, response_supervision, reference_rollout, cfg) -> Dict:
    candidate_json = [candidate_to_full_json(c, selected=(i == selected_idx)) for i, c in enumerate(scored)]
    return {
        "frame": frame_name,
        "generator": "causal_minimum_response_waypoint_generator_v3_full_debug",
        "coordinate": "ego_local_x_forward_y_right_yaw_positive_right",
        "meters_per_pixel": float(mpp),
        "ego_center": [float(ego_center[0]), float(ego_center[1])],
        "model_fps": float(cfg.horizon.model_fps),
        "future_fps": float(cfg.horizon.future_fps),
        "num_future_waypoints": int(cfg.horizon.num_future_waypoints),
        "future_frame_stride": int(cfg.horizon.future_frame_stride),
        "temporal_costmap_frames": temporal_bundle.get("frames", []),
        "temporal_costmap_valid": temporal_bundle.get("valid", []),
        "temporal_red_light_frames": temporal_bundle.get("red_light_frames", []),
        "temporal_red_light_valid": temporal_bundle.get("red_light_valid", []),
        "temporal_stop_sign_frames": temporal_bundle.get("stop_sign_frames", []),
        "temporal_stop_sign_valid": temporal_bundle.get("stop_sign_valid", []),
        "traffic_light_state": temporal_bundle.get("traffic_light_state", {}),
        "critical_factor_aligned": factor,
        "critical_factor_raw": factor.get("raw_candidate_generation_factor", factor) if isinstance(factor, dict) else factor,
        "current_dynamic_actors": current_actors,
        "intent_names": INTENT_NAMES,
        "valid_intent_mask": valid_intent_mask(scored),
        "selected_index": int(selected_idx),
        "selected_intent_id": int(selected_info.get("intent_id", -1)),
        "selected_intent_name": selected_info.get("intent_name", "unknown"),
        "selector_label": int(selected_info.get("intent_id", -1)) if selected_info.get("allowed", False) else -1,
        "risk_label_valid": bool(selected_info.get("allowed", False)),
        "fallback_to_expert": bool(selected_info.get("fallback_to_expert", False)),
        "fallback_reason": selected_info.get("fallback_reason", ""),
        "selected_score": float(selected_info.get("score", 0.0)),
        "selected_info": selected_info,
        "language_annotation": language,
        "causal_analysis": compact_causal_analysis(causal_analysis),
        "reference_response_supervision": response_supervision,
        "object_removed_reference_waypoints": array_to_list(reference_rollout.get("waypoints", [])),
        "object_removed_reference_speeds": array_to_list(reference_rollout.get("speeds", [])),
        "risk_planned_waypoints": array_to_list(selected_rollout["waypoints"]),
        "risk_planned_yaws": array_to_list(selected_rollout["yaws"]),
        "risk_planned_speeds": array_to_list(selected_rollout["speeds"]),
        "risk_planned_controls_steer_acc": array_to_list(selected_rollout["controls"]),
        "risk_planned_reference_route": array_to_list(selected["reference_route"]),
        "expert_reference_route": array_to_list(reference_route),
        "expert_future_waypoints_from_measurements": array_to_list(expert_future) if expert_future is not None else None,
        "candidate_rollouts": candidate_json,
    }


def process_one_frame(route_dir: Path, frame_name: str, cfg) -> bool:

    # 读取场景数据，包括测量数据、代价地图、车道约束地图和元数据
    measurement_path = route_dir / cfg.paths.measurements_folder / f"{frame_name}.json.gz"
    costmap_path = route_dir / cfg.paths.costmap_folder / f"{frame_name}.npy"
    lane_constraint_folder = str(_cfg_get(cfg.paths, "lane_constraint_folder", "lane_constraints"))
    lane_constraint_path = route_dir / lane_constraint_folder / f"{frame_name}.npy"
    meta_path = route_dir / cfg.paths.bev_meta_folder / f"{frame_name}.json.gz"
    if not measurement_path.exists() or not costmap_path.exists():
        LOGGER.info(f"[Skip] missing measurement/costmap for {route_dir.name}/{frame_name}")
        return False

    lane_cfg = _cfg_get(cfg, "lane_constraints", {})
    lane_enabled = bool(_cfg_get(lane_cfg, "enabled", True))
    require_lane_map = bool(_cfg_get(lane_cfg, "require_map", True))
    if lane_enabled and require_lane_map and not lane_constraint_path.exists():
        LOGGER.info(
            f"[Skip] missing solid-lane constraint map for {route_dir.name}/{frame_name}: "
            f"{lane_constraint_path}. Re-run tools_bev/generate_costmap_from_masks.py first."
        )
        return False

    measurement = load_json_gz(measurement_path)
    costmap = load_costmap(costmap_path)
    if lane_enabled and lane_constraint_path.exists():
        solid_lane_constraint = load_costmap(lane_constraint_path)
        if solid_lane_constraint.shape[:2] != costmap.shape[:2]:
            raise ValueError(
                f"Lane constraint shape {solid_lane_constraint.shape} does not match "
                f"costmap shape {costmap.shape} for {route_dir.name}/{frame_name}"
            )
    else:
        solid_lane_constraint = np.zeros_like(costmap, dtype=np.float32)
    meta = load_json_gz(meta_path) if meta_path.exists() else {}
    mpp = get_meters_per_pixel(meta, float(cfg.paths.default_pixels_per_meter))
    ego_center = get_ego_center(meta, costmap.shape)

    reference_horizon_m = _required_reference_horizon_m(measurement, cfg)
    reference_route, reference_route_info = _build_reference_route_from_measurements(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        required_horizon_m=reference_horizon_m,
        cfg=cfg,
    )
    if reference_route is None:
        LOGGER.info(
            f"[Skip] insufficient real reference route for {route_dir.name}/{frame_name}: "
            f"required={reference_route_info.get('required_horizon_m', 0.0):.2f} m, "
            f"available={reference_route_info.get('stitched_route_length_m', 0.0):.2f} m, "
            f"reason={reference_route_info.get('failure_reason', 'unknown')}"
        )
        return False

    temporal_bundle = build_temporal_costmaps(route_dir, frame_name, measurement, costmap, meta, cfg)
    red_light_bundle = build_temporal_red_light_constraints(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        current_shape=costmap.shape,
        current_meta=meta,
        cfg=cfg,
    )
    temporal_bundle["red_light_maps"] = red_light_bundle.get("maps", [])
    temporal_bundle["red_light_frames"] = red_light_bundle.get("frames", [])
    temporal_bundle["red_light_valid"] = red_light_bundle.get("valid", [])
    temporal_bundle["red_light_missing_reasons"] = red_light_bundle.get("missing_reasons", [])
    traffic_light_state = build_traffic_light_state_context(
        route_dir=route_dir,
        frame_name=frame_name,
        current_shape=costmap.shape,
        cfg=cfg,
    )
    temporal_bundle["traffic_light_state"] = traffic_light_state
    stop_sign_bundle = build_temporal_stop_sign_constraints(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        current_shape=costmap.shape,
        current_meta=meta,
        cfg=cfg,
    )
    temporal_bundle["stop_sign_maps"] = stop_sign_bundle.get("maps", [])
    temporal_bundle["stop_sign_frames"] = stop_sign_bundle.get("frames", [])
    temporal_bundle["stop_sign_valid"] = stop_sign_bundle.get("valid", [])
    temporal_bundle["stop_sign_missing_reasons"] = stop_sign_bundle.get("missing_reasons", [])
    # Keep the solid-lane map as an immutable static constraint.  Counterfactual
    # actor removal only edits occupancy costmaps, so it cannot erase road rules.
    temporal_bundle["solid_lane_constraint"] = solid_lane_constraint.astype(np.float32)
    temporal_bundle["solid_lane_constraint_path"] = str(lane_constraint_path)
    current_actors = load_current_actor_records(route_dir, frame_name, cfg)
    future_actor_timelines = load_future_actor_timelines(route_dir, frame_name, cfg)

    # Stage 1: a permissive factor is used only to seed the response pool.
    # The final critical object is not accepted from this heuristic; it must be
    # verified by an explicit object-removal intervention below.
    initial_factor = identify_critical_factor(reference_route, current_actors, temporal_bundle, ego_center, mpp, cfg)

    expert_future = load_future_ego_waypoints(route_dir, frame_name, measurement, cfg)
    candidates, nominal_reference, causal_actor_candidates = build_causal_candidate_pool(
        base_route=reference_route,
        initial_factor=initial_factor,
        current_actors=current_actors,
        measurement=measurement,
        cfg=cfg,
        expert_future=expert_future,
    )

    # Stage 2: select a preliminary full-scene response relative to the nominal
    # no-interference motion.  This preliminary result is used only to measure
    # which object removal changes the ego behavior.
    scored = evaluate_candidate_pool(
        candidates=candidates,
        base_route=reference_route,
        temporal_bundle=temporal_bundle,
        ego_center=ego_center,
        meters_per_pixel=mpp,
        actor_timelines=future_actor_timelines,
        cfg=cfg,
        factor=initial_factor,
    )
    preliminary_idx = select_minimum_response(scored, nominal_reference["rollout"], cfg)

    if preliminary_idx >= 0:
        # Causal discovery should observe the refined preliminary response rather
        # than only one coarse response strength.  Keep this candidate in a
        # temporary pool so it does not bias the later final discrete selection.
        preliminary_for_causal = refine_minimum_sufficient_response(
            selected=scored[preliminary_idx],
            reference_rollout=nominal_reference["rollout"],
            base_route=reference_route,
            temporal_bundle=temporal_bundle,
            ego_center=ego_center,
            meters_per_pixel=mpp,
            actor_timelines=future_actor_timelines,
            measurement=measurement,
            cfg=cfg,
            factor=initial_factor,
        )
        full_scored_for_causal = scored
        preliminary_for_causal_idx = preliminary_idx
        if preliminary_for_causal is not scored[preliminary_idx]:
            full_scored_for_causal = list(scored) + [preliminary_for_causal]
            preliminary_for_causal_idx = len(full_scored_for_causal) - 1

        causal_analysis = analyze_causal_objects(
            candidates=candidates,
            full_scored=full_scored_for_causal,
            full_selected_idx=preliminary_for_causal_idx,
            nominal_reference_rollout=nominal_reference["rollout"],
            actor_candidates=causal_actor_candidates,
            current_actors=current_actors,
            base_route=reference_route,
            measurement=measurement,
            temporal_bundle=temporal_bundle,
            ego_center=ego_center,
            meters_per_pixel=mpp,
            actor_timelines=future_actor_timelines,
            cfg=cfg,
            expert_future=expert_future,
        )
        reference_rollout = causal_analysis.get("reference_rollout", nominal_reference["rollout"])
        consistent_indices = causal_consistent_candidate_indices(scored, causal_analysis)
        selected_idx = select_minimum_response(scored, reference_rollout, cfg, allowed_indices=consistent_indices)
        if selected_idx < 0:
            selected_idx = preliminary_idx

        # Refine the final discrete response before causal revalidation so the
        # public causal metadata and the supervised trajectory refer to exactly
        # the same boundary-level motion.
        final_selected = refine_minimum_sufficient_response(
            selected=scored[selected_idx],
            reference_rollout=reference_rollout,
            base_route=reference_route,
            temporal_bundle=temporal_bundle,
            ego_center=ego_center,
            meters_per_pixel=mpp,
            actor_timelines=future_actor_timelines,
            measurement=measurement,
            cfg=cfg,
            factor=initial_factor,
        )

        if causal_analysis.get("has_causal_object", False):
            causal_analysis = revalidate_causal_analysis(
                final_selected=final_selected,
                full_scored=scored,
                causal_analysis=causal_analysis,
                nominal_reference_rollout=nominal_reference["rollout"],
                cfg=cfg,
            )
            reference_rollout = causal_analysis.get("reference_rollout", nominal_reference["rollout"])

            # If the discovered object no longer explains the refined final
            # response, discard both the object-removed reference and the
            # causal-reference refinement, then reselect and refine against the
            # nominal no-interference behavior.
            if not causal_analysis.get("has_causal_object", False):
                selected_idx = select_minimum_response(scored, nominal_reference["rollout"], cfg)
                if selected_idx < 0:
                    selected_idx = preliminary_idx
                reference_rollout = nominal_reference["rollout"]
                final_selected = refine_minimum_sufficient_response(
                    selected=scored[selected_idx],
                    reference_rollout=reference_rollout,
                    base_route=reference_route,
                    temporal_bundle=temporal_bundle,
                    ego_center=ego_center,
                    meters_per_pixel=mpp,
                    actor_timelines=future_actor_timelines,
                    measurement=measurement,
                    cfg=cfg,
                    factor=initial_factor,
                )

        # Keep the public candidate tensor/list length unchanged.  The refined
        # trajectory replaces its selected coarse slot; intermediate probes are
        # internal diagnostics and are never appended to candidate_waypoints.
        scored[selected_idx] = final_selected
    else:
        causal_analysis = {
            "enabled": True,
            "has_causal_object": False,
            "causal_object": {"exists": False},
            "causal_score": 0.0,
            "preliminary_causal_score": 0.0,
            "final_causal_score": None,
            "final_revalidation_passed": None,
            "reference_rollout": nominal_reference["rollout"],
            "reference_source": "nominal_no_interference",
            "counterfactual_selected": None,
            "counterfactual_candidate_count": 0,
            "object_tests": [],
        }
        reference_rollout = nominal_reference["rollout"]
        selected_idx = -1

    if selected_idx < 0:
        # At this point the final user-facing `factor` has not been built yet.
        # Use the already available scene-level `initial_factor` to decide
        # whether the current red-light hard rule requires a stationary fallback.
        red_stats = (initial_factor or {}).get("red_light_rule") or {}
        current_red_active = (
            str((initial_factor or {}).get("type", "")) == "red_light_stop_line"
            and bool(red_stats.get("current_red_active", False))
        )

        if current_red_active:
            # Current red is a hard rule.  Never replace an unavailable
            # hold-stop candidate with a moving expert fallback merely because
            # the expert trajectory still stays behind the line in this finite
            # horizon.  Preserve the stopped pose instead.
            hold_points = np.zeros((int(cfg.horizon.num_future_waypoints), 2), dtype=np.float32)
            fallback = build_fallback(
                reference_route,
                hold_points,
                cfg,
                reason="current_red_light_no_allowed_candidate_use_stationary_hold",
            )
            fallback["intent_name"] = "yield_stop"
            fallback["intent"] = {"intent_id": 2, "intent_name": "yield_stop", "active": True}
            fallback["reference_mode"] = "hard-rule stationary hold while current red light remains active"
            fallback["target_speed"] = 0.0
            fallback["info"]["intent_id"] = 2
            fallback["info"]["intent_name"] = "yield_stop"
            fallback["info"]["intent_active"] = True
            fallback["info"]["intent_en"] = "remain stopped"
            fallback["info"]["reference_mode"] = fallback["reference_mode"]
            fallback["info"]["target_speed"] = 0.0
            fallback["info"]["allowed"] = True
            fallback["info"]["reasons"] = []
            fallback["info"]["fallback_to_expert"] = False
            fallback["info"]["hard_rule_hold_override"] = True
            fallback["info"]["fallback_reason"] = "current_red_light_no_allowed_candidate_use_stationary_hold"
            # Keep the stationary hold as the legally selected hard-rule
            # response, but do not falsify its predicted collision state.  A
            # future actor may still overlap the stopped ego pose; that does not
            # make moving through a red light legal, but collision_free must
            # reflect the actual OBB check.
            if _cfg_bool(_cfg_get(cfg, "collision", {}), "enabled", True):
                hold_collision = check_rollout_collisions(
                    fallback["rollout"],
                    future_actor_timelines,
                    cfg,
                )
            else:
                hold_collision = {
                    "collision_free": True,
                    "num_collision_events": 0,
                    "first_collision": None,
                    "collision_events": [],
                    "min_approx_clearance_m": None,
                }
            fallback["info"]["collision"] = hold_collision
            fallback["info"].update(
                solid_lane_constraint_score(
                    fallback["rollout"],
                    solid_lane_constraint,
                    ego_center,
                    mpp,
                    cfg,
                )
            )
            fallback["info"].update(
                red_light_constraint_score(
                    fallback["rollout"],
                    temporal_bundle.get("red_light_maps", []),
                    ego_center,
                    mpp,
                    cfg,
                )
            )
            fallback["info"].update(
                stop_sign_constraint_score(
                    fallback["rollout"],
                    temporal_bundle.get("stop_sign_maps", []),
                    ego_center,
                    mpp,
                    cfg,
                )
            )
        else:
            fallback = build_fallback(reference_route, expert_future, cfg, reason="no_allowed_minimum_response_candidate")
            fallback_lane = solid_lane_constraint_score(
                fallback["rollout"],
                solid_lane_constraint,
                ego_center,
                mpp,
                cfg,
            )
            fallback["info"].update(fallback_lane)
            fallback_red = red_light_constraint_score(
                fallback["rollout"],
                temporal_bundle.get("red_light_maps", []),
                ego_center,
                mpp,
                cfg,
            )
            fallback["info"].update(fallback_red)
            fallback_stop = stop_sign_constraint_score(
                fallback["rollout"],
                temporal_bundle.get("stop_sign_maps", []),
                ego_center,
                mpp,
                cfg,
            )
            fallback["info"].update(fallback_stop)

            # The expert fallback may not bypass hard geometric rules.
            if (
                fallback_lane.get("solid_lane_any_overlap", False)
                or fallback_red.get("red_light_any_overlap", False)
                or fallback_stop.get("stop_sign_any_violation", False)
            ):
                hold_points = np.zeros((int(cfg.horizon.num_future_waypoints), 2), dtype=np.float32)
                fallback = build_fallback(
                    reference_route,
                    hold_points,
                    cfg,
                    reason="expert_fallback_violates_hard_rule_use_stationary_hold",
                )
                fallback["intent_name"] = "stationary_hold_fallback"
                fallback["intent"]["intent_name"] = "stationary_hold_fallback"
                fallback["info"]["intent_name"] = "stationary_hold_fallback"
                fallback["info"]["fallback_reason"] = "expert_fallback_violates_hard_rule_use_stationary_hold"
                fallback["info"].update(
                    solid_lane_constraint_score(
                        fallback["rollout"],
                        solid_lane_constraint,
                        ego_center,
                        mpp,
                        cfg,
                    )
                )
                fallback["info"].update(
                    red_light_constraint_score(
                        fallback["rollout"],
                        temporal_bundle.get("red_light_maps", []),
                        ego_center,
                        mpp,
                        cfg,
                    )
                )
                fallback["info"].update(
                    stop_sign_constraint_score(
                        fallback["rollout"],
                        temporal_bundle.get("stop_sign_maps", []),
                        ego_center,
                        mpp,
                        cfg,
                    )
                )

        scored.append(fallback)
        selected_idx = len(scored) - 1

    selected = scored[selected_idx]
    selected_info = selected["info"]
    selected_rollout = selected["rollout"]

    # Stage 3: the user-facing object is causal when object removal verified an
    # effect.  Otherwise retain the stricter post-selection geometric alignment
    # used by the previous implementation.
    if causal_analysis.get("has_causal_object", False):
        factor = build_causal_factor(
            causal_analysis,
            initial_factor,
            reference_route=selected.get("reference_route", reference_route),
            cfg=cfg,
        )
    else:
        factor = align_factor_with_selected_waypoints(
            initial_factor=initial_factor,
            current_actors=current_actors,
            future_actor_timelines=future_actor_timelines,
            selected=selected,
            cfg=cfg,
        )

    response_supervision = build_response_supervision(selected, reference_rollout, causal_analysis, cfg)
    selected_info["action_effect"] = dict(response_supervision.get("action_effect", {}) or {})
    intents = infer_intents(factor, cfg)
    language = build_language_annotation(frame_name, factor, selected, scored, response_supervision=response_supervision)

    debug_paths = {}
    if bool(cfg.debug.save_bev_debug):
        debug_path = route_dir / cfg.paths.debug_folder / f"{frame_name}_bev_waypoints.png"
        save_bev_debug(debug_path, costmap, reference_route, scored, selected_idx, future_actor_timelines, ego_center, mpp, cfg)
        debug_paths["bev_waypoints"] = str(debug_path.relative_to(route_dir))

    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    if bool(_cfg_get(rgb_cfg, "save_rgb_debug", False)):
        rgb_folder = str(_cfg_get(rgb_cfg, "rgb_debug_folder", "language_grounded_waypoints_rgb_debug"))
        rgb_path = route_dir / rgb_folder / f"{frame_name}_rgb_waypoints.jpg"
        ok = save_rgb_waypoints_debug_image(
            route_dir=route_dir,
            frame_name=frame_name,
            risk_planned_waypoints=selected_rollout["waypoints"],
            expert_future_waypoints=expert_future,
            expert_reference_route=reference_route,
            selected_reference_route=selected["reference_route"],
            scored_candidates=scored,
            selected_idx=selected_idx,
            selected_info=selected_info,
            save_path=rgb_path,
            cfg=cfg,
            risk_planned_speeds=selected_rollout.get("speeds", None),
            factor=factor,
            occupancy_map=costmap,
            ego_center=ego_center,
            meters_per_pixel=mpp,
            actor_timelines=future_actor_timelines,
            causal_test_actors=causal_actor_candidates,
            language_annotation=language,
        )
        if ok:
            debug_paths["rgb_waypoints"] = str(rgb_path.relative_to(route_dir))

    public_out = build_public_output(
        frame_name=frame_name,
        factor=factor,
        intents=intents,
        scored=scored,
        selected_idx=selected_idx,
        selected=selected,
        selected_info=selected_info,
        selected_rollout=selected_rollout,
        reference_route=reference_route,
        reference_route_info=reference_route_info,
        expert_future=expert_future,
        language=language,
        causal_analysis=causal_analysis,
        response_supervision=response_supervision,
        reference_rollout=reference_rollout,
        cfg=cfg,
        debug_paths=debug_paths,
    )

    output_path = route_dir / cfg.paths.output_folder / f"{frame_name}.json.gz"
    save_json_gz(output_path, public_out)

    output_cfg = _cfg_get(cfg, "output", {})
    if bool(_cfg_get(output_cfg, "save_full_debug_json", False)):
        full_folder = str(_cfg_get(output_cfg, "full_debug_output_folder", "language_grounded_waypoints_full_debug"))
        full_path = route_dir / full_folder / f"{frame_name}.json.gz"
        full_out = build_full_debug_output(frame_name, mpp, ego_center, temporal_bundle, factor, current_actors, scored, selected_idx, selected_info, selected_rollout, selected, reference_route, expert_future, language, causal_analysis, response_supervision, reference_rollout, cfg)
        save_json_gz(full_path, full_out)

    if bool(cfg.run.verbose):
        LOGGER.info(
            f"[OK] {route_dir.name}/{frame_name}: "
            f"factor={factor.get('type')} selected={selected_info.get('intent_name')} "
            f"valid={selected_info.get('allowed', False)} debug={debug_paths}"
        )
    return True


def process_route_dir(route_dir: Path, cfg) -> Tuple[int, int]:
    frame_cfg = _cfg_get(_cfg_get(cfg, "run", {}), "frame", None)
    if frame_cfg not in [None, "", "None"]:
        frames = [str(frame_cfg)]
    else:
        frames = list_frame_names(route_dir, cfg)

    raise_on_error = bool(_cfg_get(_cfg_get(cfg, "run", {}), "raise_on_error", False))

    ok, total = 0, 0
    for f in frames:
        f = f.replace(".json.gz", "").replace(".npy", "")
        total += 1
        try:
            if process_one_frame(route_dir, f, cfg):
                ok += 1
        except Exception as exc:
            LOGGER.exception(f"[Error] route={route_dir}, frame={f}, error={exc}")
            if raise_on_error:
                raise
    LOGGER.info(f"[Route Done] {route_dir}: {ok}/{total}")
    return ok, total


def process_dataset(cfg) -> Tuple[int, int]:
    input_path = _cfg_get(_cfg_get(cfg, "run", {}), "input", None)
    if input_path in [None, "", "None"]:
        raise ValueError("Missing config field: run.input")

    root = Path(str(input_path)).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    recursive = bool(_cfg_get(_cfg_get(cfg, "run", {}), "recursive", False))
    route_dirs = find_route_dirs(root, cfg) if recursive else [root]
    LOGGER.info(f"[Info] Found {len(route_dirs)} route dirs; recursive={recursive}")

    total_ok, total_frames = 0, 0
    for rd in route_dirs:
        ok, total = process_route_dir(rd, cfg)
        total_ok += ok
        total_frames += total
    LOGGER.info(f"[All Done] {total_ok}/{total_frames} frames processed.")
    return total_ok, total_frames
