# -*- coding: utf-8 -*-

"""Critical driving-factor analysis.

This module has two stages:

1. ``identify_critical_factor`` is used before candidate generation.  It is a
   permissive diagnostic signal that helps activate a diverse set of behaviors.
2. ``align_factor_with_selected_waypoints`` is used after candidate selection.
   It produces the user-facing factor that should explain the finally selected
   waypoints.  This second stage is intentionally stricter: an actor is treated
   as the dominant factor only when it is relevant to the selected ego path,
   future clearance, or the lateral maneuver that was actually chosen.
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .costmap import sample_bilinear
from .geometry import local_points_to_pixels, sample_polyline_by_s, min_distance_to_polyline, ensure_route_starts_at_ego
from .actor_loader import is_static_obstacle_class


def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


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


def _factor_cfg_float(cfg, key: str, default: float) -> float:
    return _safe_float(_cfg_get(_cfg_get(cfg, "factor", {}), key, default), default)


def _normalize_angle_rad(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def actor_route_relation(actor: Dict, reference_route: Optional[np.ndarray], cfg) -> Dict:
    """Describe one actor in the local coordinate frame of the reference route.

    ``x/y`` in the current ego frame are not sufficient to distinguish a true
    same-lane lead vehicle from a nearby vehicle on an opposing lane, especially
    on narrow roads and curves.  This helper projects the actor center onto the
    nearest valid route segment, measures signed cross-track offset using the
    local route tangent, and compares actor heading with that tangent.

    Coordinate convention in this project: x is forward and y is right.
    Therefore a negative signed route offset is left of the reference route.
    """
    out = {
        "route_relation_valid": False,
        "route_longitudinal_m": None,
        "route_lateral_offset_m": None,
        "route_distance_m": None,
        "route_heading_rad": None,
        "heading_difference_deg": None,
        "heading_relation": "unknown",
        "relative_position": str(actor.get("relative_position", "unknown")),
    }

    route = np.asarray(reference_route if reference_route is not None else [], dtype=np.float32)
    if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] < 2:
        return out
    route = route[:, :2]
    finite = np.all(np.isfinite(route), axis=1)
    route = route[finite]
    if len(route) < 2:
        return out

    p = np.asarray([
        _safe_float(actor.get("x_m"), 0.0),
        _safe_float(actor.get("y_m"), 0.0),
    ], dtype=np.float32)

    a = route[:-1]
    seg = route[1:] - route[:-1]
    seg_len2 = np.sum(seg * seg, axis=1)
    valid = seg_len2 > 1e-8
    if not np.any(valid):
        return out

    t = np.zeros((len(seg),), dtype=np.float32)
    t[valid] = np.clip(
        np.sum((p[None, :] - a[valid]) * seg[valid], axis=1) / seg_len2[valid],
        0.0,
        1.0,
    )
    proj = a + t[:, None] * seg
    dist2 = np.sum((p[None, :] - proj) ** 2, axis=1)
    dist2[~valid] = np.inf
    idx = int(np.argmin(dist2))
    if not math.isfinite(float(dist2[idx])):
        return out

    seg_len = np.sqrt(seg_len2)
    tangent = seg[idx] / max(float(seg_len[idx]), 1e-6)
    # Right-pointing normal for x-forward / y-right coordinates.
    right_normal = np.asarray([-float(tangent[1]), float(tangent[0])], dtype=np.float32)
    delta = p - proj[idx]
    lateral = float(np.dot(delta, right_normal))

    cum_s = np.concatenate([
        np.zeros((1,), dtype=np.float32),
        np.cumsum(seg_len.astype(np.float32)),
    ])
    longitudinal = float(cum_s[idx] + t[idx] * seg_len[idx])
    route_heading = math.atan2(float(tangent[1]), float(tangent[0]))

    cls = str(actor.get("class", "actor"))
    is_static = is_static_obstacle_class(cls, cfg)
    heading_relation = "not_applicable"
    heading_diff_deg = None
    if cls == "vehicle" and not is_static:
        actor_yaw = _safe_float(actor.get("yaw_rad"), float("nan"))
        if math.isfinite(actor_yaw):
            heading_diff_deg = abs(math.degrees(_normalize_angle_rad(actor_yaw - route_heading)))
            same_th = _factor_cfg_float(cfg, "same_direction_heading_threshold_deg", 60.0)
            oncoming_th = _factor_cfg_float(cfg, "oncoming_heading_threshold_deg", 120.0)
            if heading_diff_deg <= same_th:
                heading_relation = "same_direction"
            elif heading_diff_deg >= oncoming_th:
                heading_relation = "oncoming"
            else:
                heading_relation = "crossing"
        else:
            heading_relation = "unknown"

    lateral_band = _factor_cfg_float(cfg, "front_lateral_band_m", 2.2)
    x = _safe_float(actor.get("x_m"), 0.0)
    if x >= 0.0:
        if abs(lateral) <= lateral_band:
            relative_position = "front"
        else:
            relative_position = "front_left" if lateral < 0.0 else "front_right"
    else:
        if abs(lateral) <= lateral_band:
            relative_position = "rear"
        else:
            relative_position = "rear_left" if lateral < 0.0 else "rear_right"

    out.update({
        "route_relation_valid": True,
        "route_longitudinal_m": longitudinal,
        "route_lateral_offset_m": lateral,
        "route_distance_m": float(math.sqrt(max(float(dist2[idx]), 0.0))),
        "route_heading_rad": float(route_heading),
        "heading_difference_deg": float(heading_diff_deg) if heading_diff_deg is not None else None,
        "heading_relation": heading_relation,
        "relative_position": relative_position,
    })
    return out


def enrich_actor_route_relation(actor: Dict, reference_route: Optional[np.ndarray], cfg) -> Dict:
    if not isinstance(actor, dict):
        return {}
    out = dict(actor)
    out.update(actor_route_relation(out, reference_route, cfg))
    return out


def corridor_cost_stats(route: np.ndarray, temporal_costmaps: List[np.ndarray], ego_center, meters_per_pixel: float, cfg) -> Dict:
    horizon = float(cfg.factor.diagnosis_horizon_m)
    spacing = float(max(cfg.factor.diagnosis_spacing_m, 0.2))
    q = np.arange(0.0, horizon + 1e-6, spacing, dtype=np.float32)
    pts = sample_polyline_by_s(route, q)
    values = []
    first_hard = None
    for i, p in enumerate(pts):
        if len(temporal_costmaps) <= 1:
            tidx = 0
        else:
            tidx = int(np.clip(round(i / max(len(pts) - 1, 1) * (len(temporal_costmaps) - 1)), 0, len(temporal_costmaps) - 1))
        pix = local_points_to_pixels(p.reshape(1, 2), ego_center, meters_per_pixel)
        val, _ = sample_bilinear(temporal_costmaps[tidx], pix, float(cfg.scoring.out_of_bounds_cost))
        v = float(val[0])
        values.append(v)
        if first_hard is None and v >= float(cfg.factor.blocked_cost_threshold):
            first_hard = float(q[i])
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return {"mean_cost": 0.0, "max_cost": 0.0, "hard_ratio": 0.0, "first_hard_distance_m": None}
    return {
        "mean_cost": float(np.mean(arr)),
        "max_cost": float(np.max(arr)),
        "hard_ratio": float(np.mean(arr >= float(cfg.factor.blocked_cost_threshold))),
        "first_hard_distance_m": first_hard,
    }


def actor_relevance_score(actor: Dict, cfg) -> float:
    
    x = float(actor["x_m"]); y = float(actor["y_m"])  # x:纵向距离 y:横向距离
    d = max(float(actor.get("distance_m", np.hypot(x, y))), 0.1)  # 欧式距离
    
    score = 0.0
    if x < -1.0:  # actor 在自车后方超过 1.0m 就不考虑了
        return -1e6

    cls = actor.get("class", "")
    is_static = is_static_obstacle_class(cls, cfg)  # 判断actor是不是静态障碍物

    # 1. 行人
    if cls == "pedestrian":
        score += 6.0
    
    # 2. 静态障碍物
    if is_static:
        score += 7.0

    # 3. 自车正前方30m范围内的actor
    if 0.0 <= x <= float(cfg.factor.actor_critical_horizon_m):
        score += 5.0
    
    # 4. Route-relative lateral relation.  When a reference route relation is
    # available, do not use raw ego-frame |y| as a same-lane proxy.  In
    # particular, an oncoming vehicle on a nearby opposing lane must not receive
    # the same center-corridor bonus as a same-direction lead vehicle.
    route_relation_valid = bool(actor.get("route_relation_valid", False))
    lateral = _safe_float(actor.get("route_lateral_offset_m"), y) if route_relation_valid else y
    heading_relation = str(actor.get("heading_relation", "unknown"))
    dynamic_vehicle = cls == "vehicle" and not is_static
    direction_compatible = (not dynamic_vehicle) or heading_relation in ["same_direction", "unknown"]

    if abs(lateral) <= float(cfg.factor.front_lateral_band_m):
        score += 5.0 if direction_compatible else 2.0
        if is_static:
            score += 4.0
    elif abs(lateral) <= float(cfg.factor.side_lateral_band_m):
        score += 2.5

    # 5. 欧式距离
    score += max(0.0, 5.0 - 0.15 * d)
    
    # 6. 相对位置
    rel = actor.get("relative_position", "")
    if rel in ["front", "front_left", "front_right"]:
        score += 2.0

    return score


def choose_critical_actor(actors: List[Dict], cfg) -> Dict:
    """Permissive actor choice before waypoint generation.

    This is not necessarily the final user-facing explanation.  The final
    explanation is refined by ``align_factor_with_selected_waypoints`` after the
    selected trajectory is known.
    """
    best = None
    best_score = -1e9
    for a in actors:
        s = actor_relevance_score(a, cfg)
        if s > best_score:
            best = a; best_score = s
    if best is None or best_score < float(cfg.factor.min_actor_relevance_score):
        return {"exists": False}
    out = dict(best)
    out["exists"] = True
    out["relevance_score"] = float(best_score)
    return out


def _factor_type_from_actor(actor: Dict, fallback: str = "nearby_actor") -> str:
    rel = actor.get("relative_position", "unknown")  # 相对位置
    cls = actor.get("class", "actor")  # 类别
    
    # 1. 静态障碍物
    if is_static_obstacle_class(cls):
        if rel == "front":
            return "front_static_obstacle"
        if rel == "front_left":
            return "front_left_static_obstacle"
        if rel == "front_right":
            return "front_right_static_obstacle"
        return "static_obstacle_nearby"
    
    # 2. 行人
    if cls == "pedestrian" and actor.get("x_m", 0.0) >= 0:
        return "pedestrian_crossing_or_near_path"
    
    # 3. 普通actor 一般指的是车辆
    if rel == "front":
        return "front_actor"
    if rel == "front_left":
        return "front_left_actor"
    if rel == "front_right":
        return "front_right_actor"
    return fallback



def red_light_rule_stats(route: np.ndarray, temporal_bundle: Dict, ego_center, meters_per_pixel: float, cfg) -> Dict:
    rule_cfg = _cfg_get(cfg, "traffic_rules", {})
    enabled = bool(_cfg_get(rule_cfg, "red_light_enabled", True))
    maps = list(temporal_bundle.get("red_light_maps", []) or [])
    if not enabled or len(maps) == 0:
        return {
            "active": False,
            "stop_line_distance_m": None,
            "required_stop_distance_m": None,
            "active_temporal_indices": [],
            "current_red_active": False,
        }

    threshold = _safe_float(_cfg_get(rule_cfg, "red_light_blocked_threshold", 0.5), 0.5)
    active_indices = [int(i) for i, m in enumerate(maps) if np.any(np.asarray(m) >= threshold)]
    if not active_indices:
        return {
            "active": False,
            "stop_line_distance_m": None,
            "required_stop_distance_m": None,
            "active_temporal_indices": [],
            "current_red_active": False,
        }

    union = np.maximum.reduce([np.asarray(m, dtype=np.float32) for m in maps])
    horizon = _safe_float(_cfg_get(rule_cfg, "stop_line_search_horizon_m", 30.0), 30.0)
    spacing = max(_safe_float(_cfg_get(rule_cfg, "stop_line_search_spacing_m", 0.25), 0.25), 0.05)
    q = np.arange(0.0, horizon + 1e-6, spacing, dtype=np.float32)
    # ``route`` is saved from the first future checkpoint, which can already be
    # several metres ahead of the current ego origin.  Red-light distance must
    # be measured from the current ego pose [0, 0], not from route[0].
    route_from_ego = ensure_route_starts_at_ego(route)
    pts = sample_polyline_by_s(route_from_ego, q)
    pix = local_points_to_pixels(pts, ego_center, meters_per_pixel)
    vals, valid = sample_bilinear(union, pix, 0.0)
    hit = valid & (vals >= threshold)
    if not np.any(hit):
        return {
            "active": False,
            "stop_line_distance_m": None,
            "required_stop_distance_m": None,
            "active_temporal_indices": active_indices,
            "current_red_active": bool(0 in active_indices),
        }

    first_idx = int(np.flatnonzero(hit)[0])
    stop_line_distance = float(q[first_idx])
    margin = _safe_float(_cfg_get(rule_cfg, "stop_margin_m", 0.50), 0.50)
    min_stop = _safe_float(_cfg_get(rule_cfg, "minimum_rule_stop_distance_m", 0.50), 0.50)
    required = max(min_stop, stop_line_distance - float(cfg.vehicle.ego_half_length_m) - margin)
    return {
        "active": True,
        "stop_line_distance_m": stop_line_distance,
        "required_stop_distance_m": float(required),
        "active_temporal_indices": active_indices,
        "current_red_active": bool(0 in active_indices),
    }


def stop_sign_rule_stats(route: np.ndarray, temporal_bundle: Dict, ego_center, meters_per_pixel: float, cfg) -> Dict:
    """Locate the first active stop-sign control region along the ego route."""
    rule_cfg = _cfg_get(cfg, "traffic_rules", {})
    enabled = bool(_cfg_get(rule_cfg, "stop_sign_enabled", True))
    maps = list(temporal_bundle.get("stop_sign_maps", []) or [])
    empty = {
        "active": False,
        "stop_region_distance_m": None,
        "required_stop_distance_m": None,
        "active_temporal_indices": [],
        "current_stop_active": False,
    }
    if not enabled or len(maps) == 0:
        return empty

    threshold = _safe_float(_cfg_get(rule_cfg, "stop_sign_blocked_threshold", 0.5), 0.5)
    active_indices = [int(i) for i, m in enumerate(maps) if np.any(np.asarray(m) >= threshold)]
    if not active_indices:
        return empty

    union = np.maximum.reduce([np.asarray(m, dtype=np.float32) for m in maps])
    horizon = _safe_float(_cfg_get(rule_cfg, "stop_sign_search_horizon_m", 30.0), 30.0)
    spacing = max(_safe_float(_cfg_get(rule_cfg, "stop_sign_search_spacing_m", 0.25), 0.25), 0.05)
    q = np.arange(0.0, horizon + 1e-6, spacing, dtype=np.float32)
    route_from_ego = ensure_route_starts_at_ego(route)
    pts = sample_polyline_by_s(route_from_ego, q)
    pix = local_points_to_pixels(pts, ego_center, meters_per_pixel)
    vals, valid = sample_bilinear(union, pix, 0.0)
    hit = valid & (vals >= threshold)
    if not np.any(hit):
        return {
            "active": False,
            "stop_region_distance_m": None,
            "required_stop_distance_m": None,
            "active_temporal_indices": active_indices,
            "current_stop_active": bool(0 in active_indices),
        }

    first_idx = int(np.flatnonzero(hit)[0])
    stop_region_distance = float(q[first_idx])
    margin = _safe_float(_cfg_get(rule_cfg, "stop_sign_stop_margin_m", 0.50), 0.50)
    min_stop = _safe_float(_cfg_get(rule_cfg, "minimum_rule_stop_distance_m", 0.50), 0.50)
    required = max(min_stop, stop_region_distance - float(cfg.vehicle.ego_half_length_m) - margin)
    return {
        "active": True,
        "stop_region_distance_m": stop_region_distance,
        "required_stop_distance_m": float(required),
        "active_temporal_indices": active_indices,
        "current_stop_active": bool(0 in active_indices),
    }

def identify_critical_factor(route: np.ndarray, actors: List[Dict], temporal_bundle: Dict, ego_center, meters_per_pixel: float, cfg) -> Dict:
    stats = corridor_cost_stats(route, temporal_bundle.get("costmaps", []), ego_center, meters_per_pixel, cfg)
    red_stats = red_light_rule_stats(route, temporal_bundle, ego_center, meters_per_pixel, cfg)
    stop_stats = stop_sign_rule_stats(route, temporal_bundle, ego_center, meters_per_pixel, cfg)
    traffic_light_state = dict(temporal_bundle.get("traffic_light_state", {}) or {})
    route_aware_actors = [enrich_actor_route_relation(a, route, cfg) for a in actors]
    actor = choose_critical_actor(route_aware_actors, cfg)
    blocked = bool(stats["hard_ratio"] >= float(cfg.factor.blocked_hard_ratio) or stats["max_cost"] >= float(cfg.factor.blocked_cost_threshold))

    if red_stats.get("active", False):
        ftype = "red_light_stop_line"
        # Traffic rules and nearby actors are not mutually exclusive.  The red
        # light is the primary action constraint, while a relevant actor remains
        # available as an additional attention target.
        reason_zh = ""
        reason_en = "The traffic light ahead is red and requires the ego vehicle to remain behind the stop line."
    elif stop_stats.get("active", False):
        ftype = "stop_sign_control"
        # Keep a relevant actor alongside the stop-sign rule for multi-factor
        # supervision instead of erasing it when the rule becomes active.
        reason_zh = ""
        reason_en = "The stop sign ahead requires a complete stop before proceeding."
    elif actor.get("exists", False):
        ftype = _factor_type_from_actor(actor)
        cls = actor.get("class", "actor")
        reason_zh = ""
        reason_en = f"A {cls} at {actor.get('relative_position', 'nearby')}, about {actor.get('distance_m', 0.0):.1f} m away, is detected for candidate generation."
    elif blocked:
        ftype = "limited_forward_space"
        reason_zh = ""
        reason_en = "The forward driving space is limited, so conservative or lateral candidates are needed."
    else:
        ftype = "clear_reference_corridor"
        reason_zh = ""
        reason_en = "No dominant risk factor is detected on the reference corridor, so normal route following is preferred."

    return {
        "type": ftype,
        "blocked_by_costmap": blocked,
        "corridor_stats": stats,
        "red_light_rule": red_stats,
        "stop_sign_rule": stop_stats,
        "traffic_light_state": traffic_light_state,
        "required_stop_distance_m": (
            red_stats.get("required_stop_distance_m", None)
            if red_stats.get("active", False)
            else stop_stats.get("required_stop_distance_m", None)
        ),
        "critical_actor": actor,
        "reason_zh": reason_zh,
        "reason_en": reason_en,
        "stage": "candidate_generation",
    }


def factor_from_actor(
    actor: Dict,
    stage: str = "causal_candidate_generation",
    reference_route: Optional[np.ndarray] = None,
    cfg=None,
) -> Dict:
    """Build a factor record anchored to one explicit actor.

    The old pipeline selected one actor with a relevance heuristic before
    planning.  The causal pipeline instead evaluates several actor-anchored
    response sets and later verifies influence through object removal.
    """
    if not actor:
        return {
            "type": "clear_reference_corridor",  # 
            "critical_actor": {"exists": False},
            "blocked_by_costmap": False,
            "stage": stage,
        }
    out_actor = dict(actor)
    if reference_route is not None and cfg is not None:
        out_actor = enrich_actor_route_relation(out_actor, reference_route, cfg)
    out_actor["exists"] = True
    cls = out_actor.get("class", "actor")
    return {
        "type": _factor_type_from_actor(out_actor),
        "critical_actor": out_actor,  # 关键actor
        "blocked_by_costmap": False,
        "reason_en": f"The {cls} is evaluated as a candidate causal object.",
        "stage": stage,
    }


# ---------------------------------------------------------------------------
# Post-selection, language-facing alignment
# ---------------------------------------------------------------------------


def _copy_actor_with_exists(actor: Dict) -> Dict:
    out = dict(actor) if isinstance(actor, dict) else {}
    out["exists"] = True
    return out


def _actor_key(actor: Dict):
    aid = actor.get("id", None)
    if aid is not None:
        return ("id", str(aid))
    return None


def _same_actor(a: Dict, b: Dict) -> bool:
    ka = _actor_key(a); kb = _actor_key(b)
    if ka is not None and kb is not None:
        return ka == kb
    return False


def _find_actor_by_event(event: Dict, current_actors: List[Dict], future_actor_timelines: Dict[int, List[Dict]]) -> Dict:
    if not isinstance(event, dict):
        return {"exists": False}
    event_id = event.get("actor_id", None)
    if event_id is not None:
        for actor in current_actors:
            if str(actor.get("id", None)) == str(event_id):
                out = _copy_actor_with_exists(actor)
                out["influence_type"] = "future_conflict"
                return out
        for actors in future_actor_timelines.values():
            for actor in actors:
                if str(actor.get("id", None)) == str(event_id):
                    out = _copy_actor_with_exists(actor)
                    out["influence_type"] = "future_conflict"
                    return out
    out = {
        "exists": True,
        "id": event_id,
        "class": event.get("actor_class", "actor"),
        "relative_position": event.get("actor_relative_position", "unknown"),
        "x_m": event.get("actor_x_m", 0.0),
        "y_m": event.get("actor_y_m", 0.0),
        "distance_m": float(np.hypot(_safe_float(event.get("actor_x_m"), 0.0), _safe_float(event.get("actor_y_m"), 0.0))),
        "influence_type": "future_conflict",
    }
    return out


def _min_distance_to_points(xy: np.ndarray, points: Optional[np.ndarray]) -> float:
    if points is None:
        return float("inf")
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(pts[:, :2] - xy.reshape(1, 2), axis=1)))


def _route_distance(actor: Dict, route: Optional[np.ndarray]) -> float:
    if route is None:
        return float("inf")
    pts = np.asarray(route, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return float("inf")
    xy = np.asarray([[float(actor.get("x_m", 0.0)), float(actor.get("y_m", 0.0))]], dtype=np.float32)
    try:
        return float(min_distance_to_polyline(xy, pts)[0])
    except Exception:
        return _min_distance_to_points(xy[0], pts)


def _future_clearance_for_actor(actor: Dict, waypoints: np.ndarray, future_actor_timelines: Dict[int, List[Dict]], cfg) -> Tuple[float, Optional[int], float, float]:
    """Approximate actor-to-selected-waypoint clearance at matched future steps.

    The returned clearance is only a coarse geometric signal.  For language-facing
    actor selection, a small Euclidean clearance alone is not enough, because a
    vehicle in a neighboring lane can be close in radius-based distance while not
    actually constraining the selected ego motion.  Therefore this function also
    returns the smallest lateral and longitudinal separations at the same matched
    future step.
    """
    key = _actor_key(actor)
    if key is None:
        return float("inf"), None, float("inf"), float("inf")
    ego_half_l = _safe_float(_cfg_get(_cfg_get(cfg, "vehicle", {}), "ego_half_length_m", 2.25), 2.25)
    ego_half_w = _safe_float(_cfg_get(_cfg_get(cfg, "vehicle", {}), "ego_half_width_m", 1.0), 1.0)
    ego_radius = math.hypot(ego_half_l, ego_half_w)

    best_clearance = float("inf")
    best_k = None
    best_lat_sep = float("inf")
    best_long_sep = float("inf")
    for k in range(1, len(waypoints) + 1):
        actors = future_actor_timelines.get(k, []) if isinstance(future_actor_timelines, dict) else []
        for fa in actors:
            if not _same_actor(actor, fa):
                continue
            actor_half_l = _safe_float(
                fa.get("half_length_m"),
                _safe_float(_cfg_get(_cfg_get(cfg, "actors", {}), "default_vehicle_half_length_m", 2.25), 2.25),
            )
            actor_half_w = _safe_float(
                fa.get("half_width_m"),
                _safe_float(_cfg_get(_cfg_get(cfg, "actors", {}), "default_vehicle_half_width_m", 1.0), 1.0),
            )
            actor_radius = math.hypot(actor_half_l, actor_half_w)
            center = np.asarray([_safe_float(fa.get("x_m"), 0.0), _safe_float(fa.get("y_m"), 0.0)], dtype=np.float32)
            ego_xy = waypoints[k - 1, :2]
            dist = float(np.linalg.norm(center - ego_xy))
            clearance = dist - ego_radius - actor_radius
            if clearance < best_clearance:
                best_clearance = clearance
                best_k = k
                best_lat_sep = float(abs(center[1] - ego_xy[1]))
                best_long_sep = float(abs(center[0] - ego_xy[0]))
    return best_clearance, best_k, best_lat_sep, best_long_sep


def _actor_influence_metrics(actor: Dict, selected_rollout: Dict, selected_reference_route: Optional[np.ndarray], future_actor_timelines: Dict[int, List[Dict]], cfg) -> Dict:
    waypoints = np.asarray(selected_rollout.get("waypoints", []), dtype=np.float32)
    xy = np.asarray([_safe_float(actor.get("x_m"), 0.0), _safe_float(actor.get("y_m"), 0.0)], dtype=np.float32)
    route_relation = actor_route_relation(actor, selected_reference_route, cfg)
    route_dist = route_relation.get("route_distance_m", None)
    if route_dist is None:
        route_dist = _route_distance(actor, selected_reference_route)
    route_dist = float(route_dist) if route_dist is not None else float("inf")
    wp_dist = _min_distance_to_points(xy, waypoints)
    future_clearance, future_k, future_lat_sep, future_long_sep = _future_clearance_for_actor(actor, waypoints, future_actor_timelines, cfg)

    same_lane_band = _factor_cfg_float(cfg, "same_lane_influence_band_m", 3.2)
    route_band = _factor_cfg_float(cfg, "route_influence_band_m", 3.5)
    planned_band = _factor_cfg_float(cfg, "planned_path_influence_band_m", 4.0)
    side_band = _factor_cfg_float(cfg, "lateral_maneuver_influence_band_m", 5.0)
    direct_gap = _factor_cfg_float(cfg, "direct_front_gap_m", 22.0)
    future_clearance_th = _factor_cfg_float(cfg, "future_clearance_influence_m", 2.5)
    horizon = _factor_cfg_float(cfg, "actor_critical_horizon_m", 30.0)

    x = _safe_float(actor.get("x_m"), 0.0)
    y = _safe_float(actor.get("y_m"), 0.0)
    cls = str(actor.get("class", "actor"))
    is_static = is_static_obstacle_class(cls, cfg)

    route_relation_valid = bool(route_relation.get("route_relation_valid", False))
    route_longitudinal = route_relation.get("route_longitudinal_m", None)
    route_lateral = route_relation.get("route_lateral_offset_m", None)
    heading_relation = str(route_relation.get("heading_relation", "unknown"))
    lateral_for_relation = float(route_lateral) if route_lateral is not None else y
    longitudinal_for_relation = float(route_longitudinal) if route_longitudinal is not None else x

    ahead = (-1.0 <= x <= horizon)
    behind = x < -1.0
    dynamic_vehicle = cls == "vehicle" and not is_static
    heading_compatible = (not dynamic_vehicle) or heading_relation == "same_direction"
    if route_relation_valid:
        same_lane = (
            ahead
            and abs(lateral_for_relation) <= same_lane_band
            and longitudinal_for_relation <= direct_gap
            and heading_compatible
        )
    else:
        # Compatibility fallback for malformed / unavailable routes only.
        same_lane = ahead and abs(y) <= same_lane_band and x <= direct_gap
    route_relevant = ahead and route_dist <= route_band
    planned_relevant = ahead and wp_dist <= planned_band

    # A radius-based future-clearance test alone can falsely select vehicles in
    # adjacent lanes, especially rear-left/rear-right actors.  Treat future
    # clearance as language-relevant only when the actor is also laterally close
    # to the selected ego motion.  Rear actors are not used as explanatory
    # factors unless an actual collision event has already selected them.
    future_lateral_band = _factor_cfg_float(cfg, "future_lateral_influence_band_m", same_lane_band)
    future_laterally_close = future_lat_sep <= future_lateral_band
    future_relevant = ahead and future_laterally_close and future_clearance <= future_clearance_th

    lateral_relevant = ahead and abs(lateral_for_relation) <= side_band and (route_dist <= side_band or wp_dist <= side_band)
    oncoming_attention = (
        ahead
        and heading_relation == "oncoming"
        and route_dist <= route_band
        and x <= horizon
    )

    score = 0.0
    if cls == "pedestrian":
        score += 4.0
    if is_static:
        score += 4.0
    if same_lane:
        score += 8.0
        if is_static:
            score += 4.0
    if route_relevant:
        score += max(0.0, 5.0 - route_dist)
    if planned_relevant:
        score += max(0.0, 5.0 - wp_dist)
    if future_relevant:
        score += 8.0 + max(0.0, future_clearance_th - future_clearance)
    if lateral_relevant:
        score += max(0.0, 3.0 - 0.3 * abs(lateral_for_relation))
    if oncoming_attention:
        score += 2.0
    if ahead:
        score += max(0.0, 2.0 - 0.05 * max(x, 0.0))

    if not (same_lane or route_relevant or planned_relevant or future_relevant or lateral_relevant or oncoming_attention):
        score = -1e6

    if future_relevant:
        influence_type = "future_clearance"
    elif oncoming_attention and not same_lane:
        influence_type = "oncoming_clearance"
    elif same_lane or route_relevant or planned_relevant:
        influence_type = "direct_path"
    elif lateral_relevant:
        influence_type = "lateral_clearance"
    else:
        influence_type = "background_actor"

    return {
        "route_distance_m": float(route_dist) if math.isfinite(route_dist) else None,
        "route_relation_valid": bool(route_relation_valid),
        "route_longitudinal_m": float(route_longitudinal) if route_longitudinal is not None else None,
        "route_lateral_offset_m": float(route_lateral) if route_lateral is not None else None,
        "route_heading_rad": route_relation.get("route_heading_rad", None),
        "heading_difference_deg": route_relation.get("heading_difference_deg", None),
        "heading_relation": heading_relation,
        "relative_position": route_relation.get("relative_position", actor.get("relative_position", "unknown")),
        "planned_path_distance_m": float(wp_dist) if math.isfinite(wp_dist) else None,
        "future_min_clearance_m": float(future_clearance) if math.isfinite(future_clearance) else None,
        "future_min_clearance_index": future_k,
        "future_lateral_separation_m": float(future_lat_sep) if math.isfinite(future_lat_sep) else None,
        "future_longitudinal_separation_m": float(future_long_sep) if math.isfinite(future_long_sep) else None,
        "is_behind_ego": bool(behind),
        "same_lane": bool(same_lane),
        "route_relevant": bool(route_relevant),
        "planned_relevant": bool(planned_relevant),
        "future_relevant": bool(future_relevant),
        "lateral_relevant": bool(lateral_relevant),
        "oncoming_attention": bool(oncoming_attention),
        "influence_type": influence_type,
        "trajectory_influence_score": float(score),
    }


def _actor_matches_intent(actor: Dict, metrics: Dict, intent_name: str) -> bool:
    if metrics.get("trajectory_influence_score", -1e6) < 0.0:
        return False

    # Language-facing critical actors should explain the selected ego motion.
    # A rear-side actor in another lane usually does not explain a forward
    # cautious-follow trajectory, even if a loose distance test marks it as
    # close.  Actual collision events are handled before this function.
    if bool(metrics.get("is_behind_ego", False)):
        return False

    rel = str(metrics.get("relative_position", actor.get("relative_position", "")))
    y = _safe_float(metrics.get("route_lateral_offset_m"), _safe_float(actor.get("y_m"), 0.0))

    direct = bool(metrics.get("same_lane") or metrics.get("route_relevant") or metrics.get("planned_relevant") or metrics.get("future_relevant"))
    lateral = bool(metrics.get("lateral_relevant"))

    if intent_name in ["cautious_follow", "yield_stop", "emergency_brake", "creep"]:
        return direct
    if intent_name == "left_nudge":
        return direct or (lateral and (rel == "front_right" or y > 0.0))
    if intent_name == "right_nudge":
        return direct or (lateral and (rel == "front_left" or y < 0.0))
    if intent_name == "route_follow":
        # For normal route-following, mention an actor only if it is genuinely
        # close to the selected path; otherwise the correct explanation is that
        # no object directly changes the ego trajectory.
        return bool(
            metrics.get("same_lane")
            or metrics.get("future_relevant")
            or metrics.get("oncoming_attention")
        )
    return direct or lateral


def _future_motion_metrics(actor: Dict, future_actor_timelines: Dict[int, List[Dict]], cfg) -> Dict:
    """Summarize the actor's observed future motion in the current ego frame.

    A single-frame speed can be zero even when the actor starts moving in the
    following frames.  Language generation therefore uses the matched future
    positions as a second motion cue instead of labeling such actors as
    stationary from the instantaneous speed alone.
    """
    key = _actor_key(actor)
    if key is None:
        return {
            "future_motion_available": False,
            "future_motion_is_moving": False,
            "future_max_displacement_m": 0.0,
            "future_terminal_displacement_m": 0.0,
        }

    start = np.asarray([
        _safe_float(actor.get("x_m"), 0.0),
        _safe_float(actor.get("y_m"), 0.0),
    ], dtype=np.float32)
    matched = []
    for k in sorted(future_actor_timelines.keys() if isinstance(future_actor_timelines, dict) else []):
        for fa in future_actor_timelines.get(k, []):
            if _same_actor(actor, fa):
                xy = np.asarray([
                    _safe_float(fa.get("x_m"), start[0]),
                    _safe_float(fa.get("y_m"), start[1]),
                ], dtype=np.float32)
                matched.append((int(k), xy))
                break

    if not matched:
        return {
            "future_motion_available": False,
            "future_motion_is_moving": False,
            "future_max_displacement_m": 0.0,
            "future_terminal_displacement_m": 0.0,
        }

    displacements = [float(np.linalg.norm(xy - start)) for _, xy in matched]
    max_disp = max(displacements) if displacements else 0.0
    terminal_disp = displacements[-1] if displacements else 0.0
    future_fps = max(_safe_float(_cfg_get(_cfg_get(cfg, "horizon", {}), "future_fps", 4.0), 4.0), 1e-6)
    last_k = max(int(k) for k, _ in matched)
    duration_s = float(last_k) / future_fps
    motion_speed = max_disp / max(duration_s, 1.0 / future_fps)

    # Require a visible displacement, not numerical jitter.  The speed cue is
    # secondary and only prevents a false "stationary" label.
    moving_disp_th = _factor_cfg_float(cfg, "future_actor_moving_displacement_m", 0.75)
    moving_speed_th = _factor_cfg_float(cfg, "future_actor_moving_speed_mps", 0.5)
    is_moving = bool(max_disp >= moving_disp_th or motion_speed >= moving_speed_th)
    return {
        "future_motion_available": True,
        "future_motion_is_moving": is_moving,
        "future_max_displacement_m": float(max_disp),
        "future_terminal_displacement_m": float(terminal_disp),
        "future_motion_speed_mps": float(motion_speed),
    }


def _enrich_explanatory_actor(actor: Dict, metrics: Optional[Dict], future_actor_timelines: Dict[int, List[Dict]], cfg) -> Dict:
    if not isinstance(actor, dict) or not actor.get("exists", False):
        return {"exists": False}
    out = dict(actor)
    if isinstance(metrics, dict):
        out.update(metrics)
    out.update(_future_motion_metrics(out, future_actor_timelines, cfg))
    return out


def _choose_explanatory_actor(current_actors: List[Dict], selected_rollout: Dict, selected_reference_route: Optional[np.ndarray], future_actor_timelines: Dict[int, List[Dict]], selected_info: Dict, cfg) -> Dict:
    intent_name = str(selected_info.get("intent_name", "unknown"))

    first_collision = selected_info.get("collision", {}).get("first_collision", None)
    if first_collision:
        collision_actor = _find_actor_by_event(first_collision, current_actors, future_actor_timelines)
        if collision_actor.get("exists", False):
            collision_metrics = _actor_influence_metrics(
                collision_actor, selected_rollout, selected_reference_route, future_actor_timelines, cfg
            )
            # A rear actor must not bypass the normal explanatory-actor filter
            # merely because a stationary ego rollout is later reached by that
            # actor.  This is especially important for red-light hold states:
            # the rear vehicle is the initiator of the future conflict and does
            # not explain why the ego vehicle is stopping.
            if not bool(collision_metrics.get("is_behind_ego", False)):
                return _enrich_explanatory_actor(
                    collision_actor, collision_metrics, future_actor_timelines, cfg
                )

    best_actor = None
    best_metrics = None
    best_score = -1e9
    for actor in current_actors:
        metrics = _actor_influence_metrics(actor, selected_rollout, selected_reference_route, future_actor_timelines, cfg)
        if not _actor_matches_intent(actor, metrics, intent_name):
            continue
        score = float(metrics.get("trajectory_influence_score", -1e6))
        if score > best_score:
            best_actor = actor
            best_metrics = metrics
            best_score = score

    min_score = _factor_cfg_float(cfg, "min_explanatory_actor_score", 3.0)
    if best_actor is None or best_score < min_score:
        return {"exists": False}
    out = _copy_actor_with_exists(best_actor)
    return _enrich_explanatory_actor(out, best_metrics, future_actor_timelines, cfg)


def align_factor_with_selected_waypoints(
    initial_factor: Dict,
    current_actors: List[Dict],
    future_actor_timelines: Dict[int, List[Dict]],
    selected: Dict,
    cfg,
) -> Dict:
    """Return a user-facing factor that explains the selected waypoints.

    The pre-selection factor may be intentionally broad so that multiple
    candidates are generated.  For language supervision, however, the factor
    should be consistent with the selected waypoint shape.  A vehicle in an
    adjacent lane is therefore not described as the dominant factor unless it
    actually constrains the selected path, future clearance, or chosen lateral
    maneuver.
    """
    if isinstance(initial_factor, dict) and str(initial_factor.get("type", "")) == "red_light_stop_line":
        selected_name = str(selected.get("info", {}).get("intent_name", "unknown")) if isinstance(selected, dict) else "unknown"
        red_stats = initial_factor.get("red_light_rule") or {}
        preserve_red = bool(red_stats.get("current_red_active", False)) or selected_name in [
            "cautious_follow", "yield_stop", "emergency_brake"
        ]
        if preserve_red:
            out = dict(initial_factor)
            out["raw_candidate_generation_factor"] = initial_factor
            out["selected_intent_name"] = selected_name
            out["stage"] = "language_aligned_to_selected_waypoints"
            out["language_focus"] = "red_light_stop_line"
            out["has_direct_trajectory_influence"] = True
            # Keep a coexisting actor only when it is genuinely relevant to the
            # selected path / future clearance.  This allows pedestrians or
            # nearby vehicles to coexist with the red light without inventing
            # an irrelevant secondary actor in a clean traffic-light scene.
            rule_actor = _choose_explanatory_actor(
                current_actors=current_actors,
                selected_rollout=selected.get("rollout", {}) if isinstance(selected, dict) else {},
                selected_reference_route=selected.get("reference_route", None) if isinstance(selected, dict) else None,
                future_actor_timelines=future_actor_timelines,
                selected_info=selected.get("info", {}) if isinstance(selected, dict) else {},
                cfg=cfg,
            )
            # No causal actor was verified on this path. A nearby actor may
            # still be worth monitoring, but it must not replace the traffic
            # rule as the primary explanation.
            out["critical_actor"] = {"exists": False}
            out["secondary_attention_actor"] = (
                rule_actor if rule_actor.get("exists", False) else {"exists": False}
            )
            return out

    if isinstance(initial_factor, dict) and str(initial_factor.get("type", "")) == "stop_sign_control":
        selected_name = str(selected.get("info", {}).get("intent_name", "unknown")) if isinstance(selected, dict) else "unknown"
        stop_stats = initial_factor.get("stop_sign_rule") or {}
        current_stop_active = bool(stop_stats.get("current_stop_active", False))
        upcoming_route_follow = (
            bool(stop_stats.get("active", False))
            and not current_stop_active
            and selected_name == "route_follow"
        )
        preserve_stop = current_stop_active or upcoming_route_follow or selected_name in [
            "cautious_follow", "yield_stop", "emergency_brake"
        ]
        if preserve_stop:
            out = dict(initial_factor)
            out["raw_candidate_generation_factor"] = initial_factor
            out["selected_intent_name"] = selected_name
            out["stage"] = "language_aligned_to_selected_waypoints"
            out["language_focus"] = "stop_sign_upcoming" if upcoming_route_follow else "stop_sign_control"
            out["upcoming_traffic_rule"] = bool(upcoming_route_follow)
            out["has_direct_trajectory_influence"] = not bool(upcoming_route_follow)
            rule_actor = _choose_explanatory_actor(
                current_actors=current_actors,
                selected_rollout=selected.get("rollout", {}) if isinstance(selected, dict) else {},
                selected_reference_route=selected.get("reference_route", None) if isinstance(selected, dict) else None,
                future_actor_timelines=future_actor_timelines,
                selected_info=selected.get("info", {}) if isinstance(selected, dict) else {},
                cfg=cfg,
            )
            out["critical_actor"] = {"exists": False}
            out["secondary_attention_actor"] = (
                rule_actor if rule_actor.get("exists", False) else {"exists": False}
            )
            return out

    # Green and yellow lights are semantic context, not new hard planning
    # constraints.  A red->green (or yellow->green) transition explains why a
    # previously stopped vehicle resumes motion.  A current yellow light is
    # described according to the selected motion, while the existing candidate
    # generation and red-light hard constraint remain unchanged.
    if isinstance(initial_factor, dict):
        light_state = dict(initial_factor.get("traffic_light_state", {}) or {})
        current_light = str(light_state.get("current_state", "unknown"))
        previous_light = str(light_state.get("previous_state", "unknown"))
        transition = str(light_state.get("transition", ""))
        red_to_green = current_light == "green" and (
            previous_light == "red" or transition == "red_to_green"
        )
        yellow_to_green = current_light == "green" and (
            previous_light == "yellow" or transition == "yellow_to_green"
        )
        yellow_attention = current_light == "yellow"

        if red_to_green or yellow_to_green or yellow_attention:
            selected_info = selected.get("info", {}) if isinstance(selected, dict) else {}
            out = dict(initial_factor)
            out["raw_candidate_generation_factor"] = initial_factor
            out["selected_intent_name"] = str(selected_info.get("intent_name", "unknown"))
            out["stage"] = "language_aligned_to_selected_waypoints"
            out["traffic_light_state"] = light_state
            if red_to_green:
                out["type"] = "green_light_release"
                out["language_focus"] = "green_light_release"
                out["reason_en"] = (
                    "The traffic light ahead has changed from red to green, so the previous red-light "
                    "stopping requirement is no longer active."
                )
            elif yellow_to_green:
                out["type"] = "green_light_after_yellow"
                out["language_focus"] = "green_light_after_yellow"
                out["reason_en"] = (
                    "The traffic light ahead has changed from yellow to green, so the vehicle may continue "
                    "through the intersection while monitoring the surroundings."
                )
            else:
                out["type"] = "yellow_light_caution"
                out["language_focus"] = "yellow_light_caution"
                out["reason_en"] = "The traffic light ahead is yellow and requires a cautious response."

            light_actor = _choose_explanatory_actor(
                current_actors=current_actors,
                selected_rollout=selected.get("rollout", {}) if isinstance(selected, dict) else {},
                selected_reference_route=selected.get("reference_route", None) if isinstance(selected, dict) else None,
                future_actor_timelines=future_actor_timelines,
                selected_info=selected_info,
                cfg=cfg,
            )
            out["critical_actor"] = {"exists": False}
            out["secondary_attention_actor"] = (
                light_actor if light_actor.get("exists", False) else {"exists": False}
            )
            out["has_direct_trajectory_influence"] = True
            return out

    selected_info = selected.get("info", {}) if isinstance(selected, dict) else {}
    selected_rollout = selected.get("rollout", {}) if isinstance(selected, dict) else {}
    selected_reference_route = selected.get("reference_route", None) if isinstance(selected, dict) else None
    intent_name = str(selected_info.get("intent_name", "unknown"))

    actor = _choose_explanatory_actor(
        current_actors=current_actors,
        selected_rollout=selected_rollout,
        selected_reference_route=selected_reference_route,
        future_actor_timelines=future_actor_timelines,
        selected_info=selected_info,
        cfg=cfg,
    )

    out = dict(initial_factor) if isinstance(initial_factor, dict) else {}
    out["raw_candidate_generation_factor"] = initial_factor
    out["selected_intent_name"] = intent_name
    out["stage"] = "language_aligned_to_selected_waypoints"

    # This function is only used after causal analysis found no verified
    # causal object. A geometrically close actor can still be useful as a
    # secondary attention target, but it must not become the main factor.
    out["critical_actor"] = {"exists": False}
    out["secondary_attention_actor"] = (
        actor if actor.get("exists", False) else {"exists": False}
    )
    out["has_direct_trajectory_influence"] = False

    # Non-actor explanations should still be consistent with the selected
    # waypoints.  They should not claim that an adjacent-lane object caused a
    # slowdown when the selected path and future clearance are unaffected.
    if intent_name == "route_follow":
        out["type"] = "clear_reference_corridor"
        out["language_focus"] = "clear_path"
    elif intent_name in ["cautious_follow", "yield_stop", "creep"]:
        if bool(initial_factor.get("blocked_by_costmap", False)):
            out["type"] = "limited_forward_space"
            out["language_focus"] = "limited_free_space"
        else:
            out["type"] = "conservative_speed_profile_without_direct_actor"
            out["language_focus"] = "conservative_speed_profile"
    elif intent_name in ["left_nudge", "right_nudge"]:
        out["type"] = "route_shape_or_clearance_preference"
        out["language_focus"] = "trajectory_shape"
    elif intent_name == "emergency_brake":
        out["type"] = "conservative_stop_without_identified_actor"
        out["language_focus"] = "conservative_stop"
    else:
        out["type"] = "no_directly_influential_actor"
        out["language_focus"] = "selected_trajectory"
    return out
