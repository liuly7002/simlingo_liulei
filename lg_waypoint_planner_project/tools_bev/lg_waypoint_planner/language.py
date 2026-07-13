# -*- coding: utf-8 -*-

"""English-only human-facing language annotations.

This module converts internal planning signals into short natural driving
sentences. It intentionally avoids engineering terms such as BEV, costmap,
footprint, hard ratio, collision checker, waypoint, trajectory, candidate,
score, or expert fallback in user-facing annotations.

The generated language answers four questions:
1. What deserves attention right now?
2. What constrains the motion ahead?
3. What is the appropriate driving response now?
4. How will the ego vehicle move next?
"""

from typing import Dict, List
import math
import numpy as np


INTENT_SHAPE_EN = {
    "route_follow": "pass through smoothly",
    "cautious_follow": "slow down slightly",
    "yield_stop": "slow down early and wait if needed",
    "left_nudge": "adjust slightly to the left",
    "right_nudge": "adjust slightly to the right",
    "creep": "move through slowly while observing",
    "emergency_brake": "slow down clearly and be ready to stop",
}

INTENT_TEXT_EN = dict(INTENT_SHAPE_EN)


ROUTE_TURN_ANGLE_THRESHOLD_DEG = 18.0
RELATIVE_LATERAL_ADJUST_THRESHOLD_M = 0.65


def _polyline_cumulative_s(points: np.ndarray) -> np.ndarray:
    pts = _as_np(points)
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(pts) == 1:
        return np.zeros((1,), dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg).astype(np.float32)])


def _sample_polyline_at_s(points: np.ndarray, s_query: float) -> np.ndarray:
    pts = _as_np(points)
    if len(pts) == 0:
        return np.zeros((2,), dtype=np.float32)
    if len(pts) == 1:
        return pts[0].copy()
    s = _polyline_cumulative_s(pts)
    sq = float(np.clip(float(s_query), 0.0, float(s[-1])))
    idx = int(np.searchsorted(s, sq, side="right") - 1)
    idx = max(0, min(idx, len(pts) - 2))
    ds = float(s[idx + 1] - s[idx])
    if ds <= 1e-6:
        return pts[idx].copy()
    t = (sq - float(s[idx])) / ds
    return ((1.0 - t) * pts[idx] + t * pts[idx + 1]).astype(np.float32)


def _normalize_angle_rad(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _route_maneuver_features(selected: Dict, rollout_points: np.ndarray) -> Dict:
    """Classify the nominal route shape over the actual rollout horizon.

    Positive heading change means a right turn in this project because the
    vehicle-local convention is x forward, y right.
    """
    route = _as_np(selected.get("reference_route", None) if isinstance(selected, dict) else None)
    pts = _as_np(rollout_points)
    if len(route) < 3:
        return {
            "route_maneuver": "straight",
            "route_turn_angle_deg": 0.0,
            "route_horizon_m": 0.0,
        }

    route_s = _polyline_cumulative_s(route)
    if len(route_s) == 0 or float(route_s[-1]) < 1.0:
        return {
            "route_maneuver": "straight",
            "route_turn_angle_deg": 0.0,
            "route_horizon_m": float(route_s[-1]) if len(route_s) else 0.0,
        }

    # Use the route point nearest the rollout endpoint so a long reference
    # route does not make a short planning horizon look like a turn.
    if len(pts) > 0:
        endpoint = pts[-1]
        nearest_idx = int(np.argmin(np.linalg.norm(route - endpoint[None, :], axis=1)))
        horizon_s = float(route_s[nearest_idx])
    else:
        horizon_s = min(float(route_s[-1]), 12.0)
    horizon_s = max(0.0, min(horizon_s, float(route_s[-1])))

    if horizon_s < 6.0:
        return {
            "route_maneuver": "straight",
            "route_turn_angle_deg": 0.0,
            "route_horizon_m": horizon_s,
        }

    start_a = _sample_polyline_at_s(route, min(0.5, 0.1 * horizon_s))
    start_b = _sample_polyline_at_s(route, min(3.0, max(1.5, 0.25 * horizon_s)))
    end_b = _sample_polyline_at_s(route, horizon_s)
    end_a = _sample_polyline_at_s(route, max(0.0, horizon_s - 3.0))

    v0 = start_b - start_a
    v1 = end_b - end_a
    if float(np.linalg.norm(v0)) < 1e-4 or float(np.linalg.norm(v1)) < 1e-4:
        turn_deg = 0.0
    else:
        h0 = math.atan2(float(v0[1]), float(v0[0]))
        h1 = math.atan2(float(v1[1]), float(v1[0]))
        turn_deg = math.degrees(_normalize_angle_rad(h1 - h0))

    if turn_deg >= ROUTE_TURN_ANGLE_THRESHOLD_DEG:
        maneuver = "right_turn"
    elif turn_deg <= -ROUTE_TURN_ANGLE_THRESHOLD_DEG:
        maneuver = "left_turn"
    else:
        maneuver = "straight"

    return {
        "route_maneuver": maneuver,
        "route_turn_angle_deg": round(float(turn_deg), 1),
        "route_horizon_m": round(float(horizon_s), 2),
    }


def _signed_offsets_from_route(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Signed lateral offsets from the selected route; positive is right."""
    pts = _as_np(points)
    ref = _as_np(route)
    if len(pts) == 0 or len(ref) < 2:
        return np.zeros((0,), dtype=np.float32)

    offsets = []
    for p in pts:
        best_dist2 = float("inf")
        best_signed = 0.0
        for i in range(len(ref) - 1):
            a = ref[i]
            b = ref[i + 1]
            v = b - a
            vv = float(np.dot(v, v))
            if vv <= 1e-8:
                continue
            t = float(np.clip(np.dot(p - a, v) / vv, 0.0, 1.0))
            q = a + t * v
            d = p - q
            dist2 = float(np.dot(d, d))
            if dist2 < best_dist2:
                tangent = v / max(float(np.linalg.norm(v)), 1e-6)
                right_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
                best_signed = float(np.dot(d, right_normal))
                best_dist2 = dist2
        offsets.append(best_signed)
    return np.asarray(offsets, dtype=np.float32)


def _relative_lateral_features(selected: Dict, rollout_points: np.ndarray, absolute_direction: str) -> Dict:
    route = _as_np(selected.get("reference_route", None) if isinstance(selected, dict) else None)
    offsets = _signed_offsets_from_route(rollout_points, route)
    if offsets.size == 0:
        return {
            "relative_lateral_direction": absolute_direction,
            "relative_lateral_offset_m": 0.0,
            "max_abs_route_offset_m": 0.0,
        }

    tail_start = max(0, int(math.floor(0.4 * len(offsets))))
    tail = offsets[tail_start:]
    sustained = float(np.median(tail)) if tail.size > 0 else float(offsets[-1])
    terminal = float(offsets[-1])
    score = 0.7 * sustained + 0.3 * terminal

    if score >= RELATIVE_LATERAL_ADJUST_THRESHOLD_M:
        direction = "right"
    elif score <= -RELATIVE_LATERAL_ADJUST_THRESHOLD_M:
        direction = "left"
    else:
        direction = "center"

    return {
        "relative_lateral_direction": direction,
        "relative_lateral_offset_m": round(float(score), 2),
        "max_abs_route_offset_m": round(float(np.max(np.abs(offsets))), 2),
    }


def _round_or_none(value, ndigits=1):
    try:
        v = float(value)
        if math.isfinite(v):
            return round(v, ndigits)
    except Exception:
        pass
    return None


def _as_np(points) -> np.ndarray:
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2]


def _speeds_from_rollout(selected: Dict, future_fps: float = 4.0) -> np.ndarray:
    rollout = selected.get("rollout", {}) if isinstance(selected, dict) else {}
    speeds = rollout.get("speeds", None)
    if speeds is not None:
        try:
            arr = np.asarray(speeds, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return arr
        except Exception:
            pass

    pts = _as_np(rollout.get("waypoints", None))
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float32)
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), pts[:-1]], axis=0)
    return (np.linalg.norm(pts - prev, axis=1) * float(future_fps)).astype(np.float32)


def _trajectory_features(selected: Dict) -> Dict:
    rollout = selected.get("rollout", {}) if isinstance(selected, dict) else {}
    info = selected.get("info", {}) if isinstance(selected, dict) else {}
    pts = _as_np(rollout.get("waypoints", None))
    speeds = _speeds_from_rollout(selected)

    if len(pts) == 0:
        return {
            "valid": False,
            "forward_m": 0.0,
            "end_lateral_m": 0.0,
            "max_abs_lateral_m": 0.0,
            "mean_speed_mps": 0.0,
            "start_speed_mps": 0.0,
            "end_speed_mps": 0.0,
            "speed_delta_mps": 0.0,
            "lateral_direction": "center",
            "relative_lateral_direction": "center",
            "relative_lateral_offset_m": 0.0,
            "max_abs_route_offset_m": 0.0,
            "route_maneuver": "straight",
            "route_turn_angle_deg": 0.0,
            "route_horizon_m": 0.0,
            "speed_pattern": "unknown",
            "motion_pattern": "unknown",
            "route_deviation_m": 0.0,
            "lateral_accel_mps2": 0.0,
        }

    forward = float(pts[-1, 0])
    end_lat = float(pts[-1, 1])
    max_abs_lat = float(np.max(np.abs(pts[:, 1])))
    max_displacement = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))

    mean_speed = float(np.mean(speeds)) if speeds.size > 0 else 0.0
    start_speed = float(speeds[0]) if speeds.size > 0 else mean_speed
    end_speed = float(speeds[-1]) if speeds.size > 0 else mean_speed
    speed_delta = end_speed - start_speed

    # Project convention: y > 0 means right.
    if end_lat > 0.65 or (max_abs_lat > 0.9 and float(np.mean(pts[:, 1])) > 0.25):
        lateral_direction = "right"
    elif end_lat < -0.65 or (max_abs_lat > 0.9 and float(np.mean(pts[:, 1])) < -0.25):
        lateral_direction = "left"
    else:
        lateral_direction = "center"

    route_features = _route_maneuver_features(selected, pts)
    relative_features = _relative_lateral_features(selected, pts, lateral_direction)
    route_maneuver = route_features["route_maneuver"]
    relative_lateral_direction = relative_features["relative_lateral_direction"]

    # Distinguish an already stationary vehicle from one that is still
    # approaching a stop.  This prevents descriptions such as "move a little
    # forward and come to a stop" when the rollout is effectively motionless.
    if mean_speed < 0.15 and max_displacement < 0.15:
        speed_pattern = "stationary"
    elif mean_speed < 0.35 or forward < 0.8:
        speed_pattern = "near_stop"
    elif end_speed < max(0.4, start_speed - 0.8) or speed_delta < -0.8:
        speed_pattern = "decelerating"
    elif end_speed > start_speed + 0.8:
        speed_pattern = "accelerating"
    elif mean_speed < 2.0:
        speed_pattern = "slow_steady"
    else:
        speed_pattern = "steady"

    if relative_lateral_direction in ["left", "right"]:
        motion_pattern = "gentle_lateral_adjustment"
    elif route_maneuver in ["left_turn", "right_turn"]:
        motion_pattern = route_maneuver
    elif lateral_direction == "center" and max_abs_lat < 0.5:
        motion_pattern = "straight"
    else:
        motion_pattern = "small_adjustment"

    return {
        "valid": True,
        "forward_m": forward,
        "end_lateral_m": end_lat,
        "max_abs_lateral_m": max_abs_lat,
        "max_displacement_m": max_displacement,
        "mean_speed_mps": mean_speed,
        "start_speed_mps": start_speed,
        "end_speed_mps": end_speed,
        "speed_delta_mps": speed_delta,
        "lateral_direction": lateral_direction,
        "relative_lateral_direction": relative_lateral_direction,
        "relative_lateral_offset_m": relative_features["relative_lateral_offset_m"],
        "max_abs_route_offset_m": relative_features["max_abs_route_offset_m"],
        "route_maneuver": route_maneuver,
        "route_turn_angle_deg": route_features["route_turn_angle_deg"],
        "route_horizon_m": route_features["route_horizon_m"],
        "speed_pattern": speed_pattern,
        "motion_pattern": motion_pattern,
        "route_deviation_m": _round_or_none(info.get("max_route_deviation", 0.0), 2),
        "lateral_accel_mps2": _round_or_none(info.get("max_abs_lateral_accel", 0.0), 2),
    }


def _actor_name_en(actor: Dict) -> str:
    planning_cls = str(actor.get("class", actor.get("raw_class", "actor")))
    semantic_cls = str(actor.get("semantic_class", planning_cls))
    cls_map = {
        "vehicle": "vehicle",
        "cyclist": "cyclist",
        "motorcyclist": "motorcyclist",
        "pedestrian": "pedestrian",
        "traffic_cone": "traffic cone",
        "traffic_warning": "warning sign",
        "barrier": "barrier",
    }
    cls = cls_map.get(semantic_cls, cls_map.get(planning_cls, "road object"))

    pos = str(actor.get("relative_position", "front"))
    pos_map = {
        "front": "in front of the ego vehicle",
        "front_left": "to the front-left of the ego vehicle",
        "front_right": "to the front-right of the ego vehicle",
        "rear": "behind the ego vehicle",
        "rear_left": "to the rear-left of the ego vehicle",
        "rear_right": "to the rear-right of the ego vehicle",
    }
    heading_relation = str(actor.get("heading_relation", "unknown"))
    if cls == "vehicle" and heading_relation == "oncoming":
        cls = "oncoming vehicle"
    elif cls == "vehicle" and heading_relation == "crossing":
        cls = "crossing vehicle"
    elif cls == "cyclist" and heading_relation == "oncoming":
        cls = "oncoming cyclist"
    elif cls == "cyclist" and heading_relation == "crossing":
        cls = "crossing cyclist"
    elif cls == "motorcyclist" and heading_relation == "oncoming":
        cls = "oncoming motorcyclist"
    elif cls == "motorcyclist" and heading_relation == "crossing":
        cls = "crossing motorcyclist"

    return f"the {cls} {pos_map.get(pos, pos)}"


def _actor_influence_phrase_en(actor: Dict) -> str:
    cls = str(actor.get("class", "actor"))
    influence = str(actor.get("influence_type", "direct_path"))

    if cls in ["traffic_cone", "traffic_warning", "barrier"]:
        if influence == "lateral_clearance":
            return "It narrows the available space on that side."
        return "It occupies part of the road ahead."

    if influence == "future_conflict":
        return "It may affect the ego vehicle soon."
    if influence == "future_clearance":
        return "It reduces the space around the ego vehicle."
    if influence == "lateral_clearance":
        return "It is close on the side."
    if influence == "oncoming_clearance":
        return "It is traveling in the opposite direction on a nearby lane."
    return "It is close to the ego vehicle."


def _additional_actor_attention_en(actor: Dict) -> str:
    """Describe one coexisting actor without displacing the primary factor."""
    if not actor.get("exists", False):
        return ""
    d = _round_or_none(actor.get("distance_m"), 1)
    name = _actor_name_en(actor)
    influence = _actor_influence_phrase_en(actor)
    if d is not None:
        return f"Also pay attention to {name}, about {d:.1f} m away. {influence}"
    return f"Also pay attention to {name}. {influence}"


def _secondary_actor_attention_en(actor: Dict) -> str:
    """Describe a non-causal nearby actor as secondary attention.

    The actor is worth monitoring because of proximity or clearance, but the
    counterfactual removal test did not show that it changed the selected
    motion.  The wording must therefore avoid calling it the main factor.
    """
    if not actor.get("exists", False):
        return ""

    d = _round_or_none(actor.get("distance_m"), 1)
    name = _actor_name_en(actor)
    if name.startswith("the "):
        name = name[4:]

    cls = str(actor.get("class", actor.get("raw_class", "actor")))
    speed = _round_or_none(actor.get("speed_mps"), 1)
    future_moving = bool(actor.get("future_motion_is_moving", False))
    # Do not call an actor stationary from one instantaneous zero-speed sample
    # when its matched future positions show clear motion.
    stationary = cls == "vehicle" and speed is not None and speed <= 0.3 and not future_moving

    if stationary:
        if d is not None:
            return f"A stationary {name} is close, about {d:.1f} m away, so it should still be monitored while passing."
        return f"A stationary {name} is close, so it should still be monitored while passing."

    if d is not None:
        return f"A nearby {name}, about {d:.1f} m away, should still be monitored while passing."
    return f"A nearby {name} should still be monitored while passing."


def _coexisting_actor_attention_en(factor: Dict) -> str:
    """Return causal co-attention first, otherwise secondary attention."""
    critical = factor.get("critical_actor") or {}
    if critical.get("exists", False):
        return _additional_actor_attention_en(critical)
    secondary = factor.get("secondary_attention_actor") or {}
    if secondary.get("exists", False):
        return _secondary_actor_attention_en(secondary)
    return ""


def describe_attention_en(factor: Dict) -> str:
    """Describe the salient object or traffic-control element that deserves attention."""
    actor = factor.get("critical_actor") or {}
    secondary_actor = factor.get("secondary_attention_actor") or {}
    ftype = str(factor.get("type", ""))
    language_focus = str(factor.get("language_focus", ""))

    if actor.get("exists", False):
        d = _round_or_none(actor.get("distance_m"), 1)
        name = _actor_name_en(actor)
        influence = _actor_influence_phrase_en(actor)
        if d is not None:
            return f"The main thing to watch is {name}, about {d:.1f} m away. {influence}"
        return f"The main thing to watch is {name}. {influence}"

    if ftype == "red_light_stop_line" or language_focus == "red_light_stop_line":
        return "The traffic light ahead deserves attention because it is red."
    if ftype == "green_light_release" or language_focus == "green_light_release":
        return "The traffic light ahead deserves attention because it has just changed from red to green."
    if ftype == "green_light_after_yellow" or language_focus == "green_light_after_yellow":
        return "The traffic light ahead deserves attention because it has changed from yellow to green."
    if ftype == "yellow_light_caution" or language_focus == "yellow_light_caution":
        return "The traffic light ahead deserves attention because it is yellow."
    if ftype == "stop_sign_control" or language_focus in ["stop_sign_control", "stop_sign_upcoming"]:
        return "The stop sign ahead deserves attention."

    secondary = _secondary_actor_attention_en(secondary_actor)
    if secondary:
        return secondary
    return "No single road user requires dominant attention right now."


def describe_motion_constraint_en(factor: Dict) -> str:
    """Describe the scene-level condition that constrains or permits motion."""
    actor = factor.get("critical_actor") or {}
    ftype = str(factor.get("type", ""))
    language_focus = str(factor.get("language_focus", ""))

    if ftype == "red_light_stop_line" or language_focus == "red_light_stop_line":
        d = _round_or_none((factor.get("red_light_rule") or {}).get("stop_line_distance_m"), 1)
        if d is not None:
            return f"The red signal requires the ego vehicle to remain behind the stop line about {d:.1f} m ahead."
        return "The red signal requires the ego vehicle to remain behind the stop line."

    if ftype == "green_light_release" or language_focus == "green_light_release":
        return "The previous red-light stopping constraint is no longer active, provided the way ahead remains clear."

    if ftype == "green_light_after_yellow" or language_focus == "green_light_after_yellow":
        return "The signal no longer restricts forward motion, provided the way through the intersection remains clear."

    if ftype == "yellow_light_caution" or language_focus == "yellow_light_caution":
        return "The changing signal limits the available margin for continuing through the intersection."

    if ftype == "stop_sign_control" or language_focus in ["stop_sign_control", "stop_sign_upcoming"]:
        d = _round_or_none((factor.get("stop_sign_rule") or {}).get("stop_region_distance_m"), 1)
        upcoming_only = bool(factor.get("upcoming_traffic_rule", False))
        if upcoming_only:
            if d is not None:
                return f"A mandatory stopping region lies about {d:.1f} m ahead, so the remaining forward margin is limited."
            return "A mandatory stopping region lies ahead, so the remaining forward margin is limited."
        if d is not None:
            return f"The stop sign requires a complete stop before the stop region about {d:.1f} m ahead."
        return "The stop sign requires a complete stop before proceeding."

    if ftype == "limited_forward_space" or language_focus == "limited_free_space":
        return "The available space ahead is limited."

    if ftype == "conservative_speed_profile_without_direct_actor":
        return "The situation ahead calls for a more cautious speed even though no object clearly blocks the way."

    if ftype == "conservative_stop_without_identified_actor":
        return "The situation ahead leaves limited margin for continued motion."

    if actor.get("exists", False):
        name = _actor_name_en(actor)
        subject = name[:1].upper() + name[1:] if name else "The nearby actor"
        rel = str(actor.get("relative_position", actor.get("position", ""))).lower()
        if "front" in rel or "ahead" in rel:
            return f"{subject} reduces the available forward margin."
        if any(k in rel for k in ["left", "right", "side", "cross"]):
            return f"{subject} limits the available maneuvering margin."
        return f"{subject} reduces the available motion margin."

    if ftype == "route_shape_or_clearance_preference":
        return "The path ahead remains clear enough to continue."

    return "The path ahead remains clear enough to continue."


def describe_factor_en(factor: Dict) -> str:
    """Backward-compatible fused factor description used by summary generation."""
    attention = describe_attention_en(factor)
    constraint = describe_motion_constraint_en(factor)
    return f"{attention} {constraint}".strip()

def _intent_from_shape_en(features: Dict) -> str:
    lateral = features.get("relative_lateral_direction", features.get("lateral_direction", "center"))
    route_maneuver = features.get("route_maneuver", "straight")
    speed = features.get("speed_pattern", "steady")

    if speed == "near_stop":
        if lateral == "left":
            return "slow down and be ready to stop while adjusting slightly to the left"
        if lateral == "right":
            return "slow down and be ready to stop while adjusting slightly to the right"
        if route_maneuver == "left_turn":
            return "slow down through the left turn and be ready to stop"
        if route_maneuver == "right_turn":
            return "slow down through the right turn and be ready to stop"
        return "slow down and be ready to stop"

    if speed == "decelerating":
        if lateral == "left":
            return "slow down and adjust slightly to the left"
        if lateral == "right":
            return "slow down and adjust slightly to the right"
        if route_maneuver == "left_turn":
            return "slow down while following the left turn"
        if route_maneuver == "right_turn":
            return "slow down while following the right turn"
        return "slow down slightly"

    if lateral == "left":
        return "adjust slightly to the left"
    if lateral == "right":
        return "adjust slightly to the right"

    if route_maneuver == "left_turn":
        if speed == "slow_steady":
            return "follow the route through the left turn slowly while observing"
        return "follow the route through the left turn"
    if route_maneuver == "right_turn":
        if speed == "slow_steady":
            return "follow the route through the right turn slowly while observing"
        return "follow the route through the right turn"

    if speed == "slow_steady":
        return "move through slowly while observing"
    return "pass through smoothly"


def _action_effect(selected: Dict) -> Dict:
    info = selected.get("info", {}) if isinstance(selected, dict) else {}
    effect = info.get("action_effect", {})
    return effect if isinstance(effect, dict) else {}


def _cautious_follow_public_name(selected: Dict) -> str:
    action = str(_action_effect(selected).get("longitudinal_action", ""))
    if action == "limit_acceleration":
        return "limited_acceleration"
    if action in ["slower_than_reference", "maintain_speed", "accelerate"]:
        return "cautious_forward"
    return "cautious_follow"


def _cautious_follow_intent_en(selected: Dict, preparing_to_stop: bool = False) -> str:
    effect = _action_effect(selected)
    action = str(effect.get("longitudinal_action", ""))
    magnitude = str(effect.get("effect_magnitude", ""))

    if preparing_to_stop:
        if action in ["decelerate", "approach_stop"]:
            return "slow down in preparation for a complete stop"
        if action == "limit_acceleration":
            return "continue forward while limiting acceleration and preparing to stop"
        if action == "slower_than_reference":
            return "continue forward cautiously at a slightly lower speed while preparing to stop"
        if action == "stationary":
            return "remain stopped briefly before proceeding"
        return "continue forward cautiously while preparing to stop"

    if action in ["decelerate", "approach_stop"]:
        return "slow down and follow cautiously"
    if action == "limit_acceleration":
        return "continue forward while limiting acceleration and maintaining a safe gap"
    if action == "slower_than_reference":
        return "continue at a slightly lower speed while maintaining a safe gap"
    if action == "stationary":
        return "remain stopped while maintaining a safe gap"
    if magnitude == "negligible":
        return "continue forward cautiously while maintaining a safe gap"
    return "continue forward cautiously while maintaining a safe gap"


def describe_driving_intent_name(selected: Dict, factor: Dict = None) -> str:
    info = selected.get("info", {}) if isinstance(selected, dict) else {}
    name = str(info.get("intent_name", "unknown"))
    f = _trajectory_features(selected)

    # Explicit lateral response candidates remain adjustments.  The route-aware
    # logic below is for nominal motion and internal fallback trajectories.
    if name in ["left_nudge", "right_nudge"]:
        return name
    if name == "cautious_follow":
        return _cautious_follow_public_name(selected)
    if name in INTENT_TEXT_EN and name != "route_follow":
        return name

    lateral = f.get("relative_lateral_direction", "center")
    route_maneuver = f.get("route_maneuver", "straight")
    speed = f.get("speed_pattern", "steady")

    if speed == "near_stop":
        return "slow_or_stop"
    if speed in ["decelerating", "slow_steady"]:
        if lateral == "left":
            return "slow_left_adjust"
        if lateral == "right":
            return "slow_right_adjust"
        if route_maneuver == "left_turn":
            return "cautious_left_turn"
        if route_maneuver == "right_turn":
            return "cautious_right_turn"
        return "cautious_forward"
    if lateral == "left":
        return "left_adjust"
    if lateral == "right":
        return "right_adjust"
    if route_maneuver == "left_turn":
        return "left_turn"
    if route_maneuver == "right_turn":
        return "right_turn"
    if name == "route_follow":
        return "route_follow"
    return "smooth_forward"


def describe_driving_intent_en(selected: Dict, factor: Dict = None) -> str:
    """Return the user-facing driving intent in English.

    Internal generation strategies such as expert fallback are never exposed as
    driving intents. If the internal label is not a real driving action, infer
    the intent from the generated motion itself.
    """
    info = selected.get("info", {}) if isinstance(selected, dict) else {}
    name = str(info.get("intent_name", "unknown"))
    variant_id = str(info.get("variant_id", selected.get("variant_id", ""))) if isinstance(selected, dict) else ""
    ftype = str((factor or {}).get("type", "")) if isinstance(factor, dict) else ""
    features = _trajectory_features(selected)

    # A stop sign is a stateful stop-then-release rule.  Describe the explicit
    # candidate from its actual rollout phase rather than collapsing it into a
    # generic deceleration or stationary phrase.
    if variant_id == "stop_sign__stop_then_go":
        rollout = selected.get("rollout", {}) if isinstance(selected, dict) else {}
        completed = bool(rollout.get("stop_then_go_completed", info.get("stop_sign_completed", False)))
        released = bool(rollout.get("stop_then_go_release_started", False))
        if released:
            return "come to a complete stop, wait briefly, and then proceed"
        if completed:
            return "come to a complete stop and wait briefly before proceeding"
        return "slow down and come to a complete stop before proceeding"
    if features.get("speed_pattern") == "stationary":
        if ftype == "red_light_stop_line":
            return "remain stopped and wait for the traffic light to change"
        if ftype == "green_light_release":
            return "remain stopped until it is safe to proceed"
        if ftype == "green_light_after_yellow":
            return "remain stopped until the surrounding traffic allows the vehicle to proceed"
        if ftype == "yellow_light_caution":
            return "remain stopped and wait for the signal to change"
        if ftype == "stop_sign_control":
            return "remain stopped briefly before proceeding"
        return "remain stopped"
    if ftype == "green_light_release":
        if features.get("speed_pattern") == "accelerating":
            return "start moving and proceed through the intersection"
        if features.get("speed_pattern") in ["near_stop", "decelerating"]:
            return "proceed when it is safe while continuing to monitor the surroundings"
        return "proceed through the intersection"

    if ftype == "green_light_after_yellow":
        if features.get("speed_pattern") == "accelerating":
            return "continue through the intersection while monitoring the surroundings"
        if features.get("speed_pattern") in ["near_stop", "decelerating"]:
            return "proceed cautiously while responding to the surrounding traffic"
        return "continue through the intersection if the path remains clear"

    if ftype == "yellow_light_caution":
        if features.get("speed_pattern") in ["near_stop", "decelerating"]:
            return "slow down and prepare to stop"
        if features.get("speed_pattern") == "accelerating":
            return "avoid further acceleration and proceed cautiously"
        return "proceed cautiously and be ready to stop"

    if ftype == "stop_sign_control":
        if name == "route_follow" and bool((factor or {}).get("upcoming_traffic_rule", False)):
            return "continue forward for now while preparing to stop at the stop sign ahead"
        if name == "cautious_follow":
            return _cautious_follow_intent_en(selected, preparing_to_stop=True)
        if name == "yield_stop":
            return "slow down and come to a complete stop before proceeding"
        if name == "emergency_brake":
            return "brake firmly and come to a complete stop"

    # Explicit nudge candidates describe a real adjustment relative to the
    # route.  Route-following and internal fallback motion are described from
    # route-relative geometry so a normal turn is not mislabeled as a nudge.
    if name in ["left_nudge", "right_nudge"]:
        return INTENT_TEXT_EN[name]
    if name == "route_follow":
        return _intent_from_shape_en(features)
    if name == "cautious_follow":
        return _cautious_follow_intent_en(selected, preparing_to_stop=False)
    if name in INTENT_TEXT_EN:
        return INTENT_TEXT_EN[name]
    return _intent_from_shape_en(features)


def _speed_phrase_en(speed_pattern: str) -> str:
    if speed_pattern == "stationary":
        return "while remaining stopped"
    if speed_pattern == "near_stop":
        return "and gradually come to a stop"
    if speed_pattern == "decelerating":
        return "with speed gradually easing down"
    if speed_pattern == "slow_steady":
        return "at a low speed"
    if speed_pattern == "accelerating":
        return "with speed gradually increasing"
    return "with little speed change"


def describe_waypoint_shape_en(selected: Dict, factor: Dict = None) -> str:
    """Describe how the vehicle will move next, without engineering terms.

    A route turn is described as a turn.  Left/right "adjustment" is reserved
    for motion that departs laterally from the selected reference route.
    """
    f = _trajectory_features(selected)
    if not f.get("valid", False):
        return "The next movement is not clear enough."

    info = selected.get("info", {}) if isinstance(selected, dict) else {}
    variant_id = str(info.get("variant_id", selected.get("variant_id", ""))) if isinstance(selected, dict) else ""
    lateral = f.get("relative_lateral_direction", "center")
    route_maneuver = f.get("route_maneuver", "straight")
    speed = f["speed_pattern"]
    speed_phrase = _speed_phrase_en(speed)

    if variant_id == "stop_sign__stop_then_go":
        rollout = selected.get("rollout", {}) if isinstance(selected, dict) else {}
        completed = bool(rollout.get("stop_then_go_completed", info.get("stop_sign_completed", False)))
        released = bool(rollout.get("stop_then_go_release_started", False))
        if released:
            if route_maneuver == "left_turn":
                return "The ego vehicle will come to a complete stop, wait briefly, and then continue along the left-curving route."
            if route_maneuver == "right_turn":
                return "The ego vehicle will come to a complete stop, wait briefly, and then continue along the right-curving route."
            return "The ego vehicle will come to a complete stop, wait briefly, and then continue forward."
        if completed:
            return "The ego vehicle will come to a complete stop and wait briefly before proceeding."
        return "The ego vehicle will slow down toward a complete stop at the stop sign."

    if speed == "stationary":
        return "The ego vehicle will remain stopped."

    if speed == "near_stop":
        if lateral == "left":
            return "The ego vehicle will move slightly left relative to the route and gradually come to a stop."
        if lateral == "right":
            return "The ego vehicle will move slightly right relative to the route and gradually come to a stop."
        if route_maneuver == "left_turn":
            return "The ego vehicle will follow a left-curving path and gradually come to a stop."
        if route_maneuver == "right_turn":
            return "The ego vehicle will follow a right-curving path and gradually come to a stop."
        return "The ego vehicle will move a little forward and gradually come to a stop."

    if lateral == "left":
        return f"The ego vehicle will make a slight adjustment to the left relative to the route, {speed_phrase}."
    if lateral == "right":
        return f"The ego vehicle will make a slight adjustment to the right relative to the route, {speed_phrase}."
    if route_maneuver == "left_turn":
        return f"The ego vehicle will follow a left-curving path, {speed_phrase}."
    if route_maneuver == "right_turn":
        return f"The ego vehicle will follow a right-curving path, {speed_phrase}."
    return f"The ego vehicle will mostly go straight, {speed_phrase}."


def _strip_en_period(text: str) -> str:
    return str(text).strip().rstrip(".!?")


def build_language_annotation(frame_name: str, factor: Dict, selected: Dict, candidates: List[Dict], response_supervision: Dict = None) -> Dict:
    intent_en = describe_driving_intent_en(selected, factor)
    shape_en = describe_waypoint_shape_en(selected, factor)
    attention_en = describe_attention_en(factor)
    constraint_en = describe_motion_constraint_en(factor)
    features = _trajectory_features(selected)

    response_en = _strip_en_period(intent_en)
    if response_en:
        response_en = response_en[0].upper() + response_en[1:] + "."

    # Follow the full reasoning chain: salient attention -> motion constraint ->
    # driving response -> future motion.  Avoid repeating the stationary motion
    # sentence in the fused summary when the response already fully describes it.
    summary_parts = [attention_en, constraint_en, response_en]
    if features.get("speed_pattern") != "stationary":
        summary_parts.append(shape_en)
    summary_en = " ".join(part.strip() for part in summary_parts if str(part).strip())

    qa_en = [
        {"question": "What deserves attention right now?", "answer": attention_en},
        {"question": "What constrains the motion ahead?", "answer": constraint_en},
        {"question": "What is the appropriate driving response now?", "answer": response_en},
        {"question": "How will the ego vehicle move next?", "answer": shape_en},
    ]

    return {
        "summary_en": summary_en,
        "qa_pairs_en": qa_en,
    }
