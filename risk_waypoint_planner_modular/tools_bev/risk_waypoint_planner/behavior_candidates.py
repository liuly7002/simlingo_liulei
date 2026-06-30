# -*- coding: utf-8 -*-

from .common import *
from .geometry_utils import (parse_offsets, parse_strings_csv, resample_polyline, cumulative_distance, sample_polyline_by_s, smoothstep, compute_right_normals, make_offset_candidate, make_yield_reference_route, min_distance_to_polyline_points, make_return_to_route_offset_candidate, make_straight_reference)
from .costmap_utils import local_points_to_pixels, sample_cost_bilinear

BEHAVIOR_ID = {
    "route_follow": 0,
    "cautious_follow": 1,
    "yield_stop": 2,
    "left_nudge": 3,
    "right_nudge": 4,
    "creep": 5,
    "emergency_brake": 6,
}

BEHAVIOR_NAMES = [
    "route_follow",
    "cautious_follow",
    "yield_stop",
    "left_nudge",
    "right_nudge",
    "creep",
    "emergency_brake",
]

def build_reference_candidates(expert_route: np.ndarray, args) -> List[Dict]:
    base = resample_polyline(
        expert_route,
        spacing_m=args.reference_spacing_m,
        horizon_m=args.reference_horizon_m,
    )

    candidates = []
    offsets = parse_offsets(args.offsets_m)

    for offset in offsets:
        ref = make_offset_candidate(
            base,
            offset_m=offset,
            offset_start_m=args.offset_start_m,
            offset_transition_m=args.offset_transition_m,
        )

        if abs(offset) < 1e-6:
            mode = "expert_reference"
        elif offset > 0:
            mode = f"right_offset_{abs(offset):.2f}m"
        else:
            mode = f"left_offset_{abs(offset):.2f}m"

        candidates.append({
            "reference_mode": mode,
            "offset_m": float(offset),
            "reference_route": ref,
        })

    if args.include_yield:
        yield_ref = make_yield_reference_route(
            base_route=base,
            stop_distance_m=args.yield_stop_distance_m,
            spacing_m=args.reference_spacing_m,
            horizon_m=args.reference_horizon_m,
        )
        candidates.append({
            "reference_mode": "yield_reference",
            "offset_m": 0.0,
            "reference_route": yield_ref,
        })

    return candidates

def build_speed_profiles(measurement: Dict, args) -> List[Dict]:
    speed_modes = parse_strings_csv(args.speed_modes)

    current_speed = float(max(measurement.get("speed", 0.0), 0.0))
    target_speed = measurement.get(args.target_speed_key, current_speed)
    try:
        target_speed = float(target_speed)
    except Exception:
        target_speed = current_speed

    # Clamp all target speeds to sane range.
    current_speed = float(np.clip(current_speed, 0.0, args.max_speed_mps))
    target_speed = float(np.clip(target_speed, 0.0, args.max_speed_mps))

    profiles = []
    for mode in speed_modes:
        if mode == "keep":
            v = target_speed if target_speed > 0.1 else current_speed
            profiles.append({"speed_mode": "keep", "target_speed": float(v)})
        elif mode == "current":
            profiles.append({"speed_mode": "current", "target_speed": float(current_speed)})
        elif mode == "slow":
            v = max(args.min_rollout_speed_mps, 0.6 * max(target_speed, current_speed))
            profiles.append({"speed_mode": "slow", "target_speed": float(v)})
        elif mode == "cautious":
            v = max(args.min_rollout_speed_mps, min(target_speed, current_speed, args.cautious_speed_mps))
            profiles.append({"speed_mode": "cautious", "target_speed": float(v)})
        elif mode == "fast":
            v = min(args.max_speed_mps, max(target_speed, current_speed) * 1.25 + 0.5)
            profiles.append({"speed_mode": "fast", "target_speed": float(v)})
        elif mode == "stop":
            profiles.append({"speed_mode": "stop", "target_speed": 0.0})
        else:
            LOGGER.warning(f"[Warn] unknown speed mode ignored: {mode}")

    # Ensure at least one profile.
    if len(profiles) == 0:
        profiles.append({"speed_mode": "keep", "target_speed": float(target_speed)})

    # Remove duplicates.
    out = []
    seen = set()
    for p in profiles:
        key = (p["speed_mode"], round(float(p["target_speed"]), 3))
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out

def make_progress_query_distances(args) -> np.ndarray:
    """Distances used for fast scene diagnosis along a candidate corridor."""
    horizon = float(min(args.reference_horizon_m, args.behavior_diagnosis_horizon_m))
    spacing = float(max(args.behavior_diagnosis_spacing_m, 0.1))
    q = np.arange(0.0, horizon + 1e-6, spacing, dtype=np.float32)
    if len(q) == 0:
        q = np.asarray([0.0], dtype=np.float32)
    return q.astype(np.float32)

def temporal_index_for_query_index(query_i: int, num_queries: int, num_temporal_maps: int) -> int:
    """
    Map a route/corridor sample to a temporal map index for light-weight scene
    diagnosis. This is not used for final scoring; final scoring is done at
    dense bicycle-model steps.
    """
    if num_temporal_maps <= 1 or num_queries <= 1:
        return 0
    alpha = float(query_i) / float(max(num_queries - 1, 1))
    return int(np.clip(round(alpha * (num_temporal_maps - 1)), 0, num_temporal_maps - 1))

def sample_corridor_cost_statistics(
    reference_route: np.ndarray,
    temporal_costmaps: List[np.ndarray],
    ego_center: List[float],
    meters_per_pixel: float,
    args,
) -> Dict:
    """
    Quickly estimate whether a reference corridor is risky.

    This function samples only the centerline points of a reference route. The
    final candidate score still uses the dense bicycle rollout and optional ego
    footprint sampling.
    """
    if len(temporal_costmaps) == 0:
        return {
            "mean_cost": 0.0,
            "max_cost": 0.0,
            "hard_ratio": 0.0,
            "first_hard_distance_m": None,
        }

    q = make_progress_query_distances(args)
    pts = sample_polyline_by_s(reference_route, q)

    sampled = []
    first_hard_distance = None
    for i, pt in enumerate(pts):
        tidx = temporal_index_for_query_index(i, len(pts), len(temporal_costmaps))
        pix = local_points_to_pixels(pt.reshape(1, 2), ego_center, meters_per_pixel)
        val, _ = sample_cost_bilinear(
            temporal_costmaps[tidx],
            pix,
            out_of_bounds_cost=args.out_of_bounds_cost,
        )
        cost_v = float(val[0])
        sampled.append(cost_v)
        if first_hard_distance is None and cost_v >= float(args.behavior_blocked_cost_threshold):
            first_hard_distance = float(q[i])

    arr = np.asarray(sampled, dtype=np.float32)
    return {
        "mean_cost": float(np.mean(arr)) if len(arr) else 0.0,
        "max_cost": float(np.max(arr)) if len(arr) else 0.0,
        "hard_ratio": float(np.mean(arr >= float(args.behavior_blocked_cost_threshold))) if len(arr) else 0.0,
        "first_hard_distance_m": first_hard_distance,
    }

def is_corridor_free(stats: Dict, args) -> bool:
    return (
        float(stats.get("max_cost", 0.0)) <= float(args.behavior_free_cost_threshold)
        and float(stats.get("hard_ratio", 0.0)) <= float(args.behavior_free_hard_ratio)
    )

def get_target_speed_from_measurement(measurement: Dict, args) -> float:
    current_speed = float(max(measurement.get("speed", 0.0), 0.0))
    target_speed = measurement.get(args.target_speed_key, current_speed)
    try:
        target_speed = float(target_speed)
    except Exception:
        target_speed = current_speed
    if target_speed <= 0.1:
        target_speed = current_speed
    return float(np.clip(target_speed, 0.0, args.max_speed_mps))

def make_speed_profile_for_behavior(behavior_name: str, measurement: Dict, args) -> Dict:
    current_speed = float(np.clip(float(max(measurement.get("speed", 0.0), 0.0)), 0.0, args.max_speed_mps))
    target_speed = get_target_speed_from_measurement(measurement, args)

    if behavior_name == "route_follow":
        return {"speed_mode": "keep", "target_speed": float(target_speed)}

    if behavior_name == "cautious_follow":
        v = max(args.min_rollout_speed_mps, min(target_speed, current_speed, args.cautious_speed_mps))
        return {"speed_mode": "cautious", "target_speed": float(v)}

    if behavior_name == "yield_stop":
        return {"speed_mode": "yield_stop", "target_speed": 0.0}

    if behavior_name == "left_nudge" or behavior_name == "right_nudge":
        v = max(args.min_rollout_speed_mps, min(target_speed, args.nudge_speed_mps))
        return {"speed_mode": "nudge_slow", "target_speed": float(v)}

    if behavior_name == "creep":
        return {"speed_mode": "creep", "target_speed": float(args.creep_speed_mps)}

    if behavior_name == "emergency_brake":
        return {"speed_mode": "emergency_brake", "target_speed": 0.0}

    return {"speed_mode": "keep", "target_speed": float(target_speed)}

def diagnose_scene_risk(
    expert_reference_route: np.ndarray,
    temporal_bundle: Dict,
    ego_center: List[float],
    meters_per_pixel: float,
    measurement: Dict,
    args,
) -> Dict:
    """
    Diagnose current scene risk to activate behavior candidates.

    This is deliberately lightweight. The final decision is still made by the
    dense rollout score. The diagnosis only answers questions such as whether
    the route ahead is risky and whether left/right lateral corridors look free.
    
    先用一个轻量级的方式判断当前导航路线前方是否有风险，
    以及左右避让通道是否可用，
    然后把这些判断结果打包成 scene_context，
    供后面的多行为候选生成函数使用。
    """

    # 1. 诊断原始路线前方风险 沿参考路线中心线采样
    temporal_costmaps = temporal_bundle.get("costmaps", [])
    base_stats = sample_corridor_cost_statistics(
        reference_route=expert_reference_route,
        temporal_costmaps=temporal_costmaps,
        ego_center=ego_center,
        meters_per_pixel=meters_per_pixel,
        args=args,
    )

    # 2. 构造左右避让参考线
    left_ref = make_return_to_route_offset_candidate(
        expert_reference_route,
        offset_m=-abs(float(args.left_nudge_offset_m)),
        offset_start_m=args.offset_start_m,
        offset_transition_m=args.offset_transition_m,
        return_start_m=args.nudge_return_start_m,
        return_transition_m=args.nudge_return_transition_m,
    )
    right_ref = make_return_to_route_offset_candidate(
        expert_reference_route,
        offset_m=abs(float(args.right_nudge_offset_m)),
        offset_start_m=args.offset_start_m,
        offset_transition_m=args.offset_transition_m,
        return_start_m=args.nudge_return_start_m,
        return_transition_m=args.nudge_return_transition_m,
    )
    # 沿左右避让参考线中心线采样做风险统计
    left_stats = sample_corridor_cost_statistics(
        reference_route=left_ref,
        temporal_costmaps=temporal_costmaps,
        ego_center=ego_center,
        meters_per_pixel=meters_per_pixel,
        args=args,
    )
    right_stats = sample_corridor_cost_statistics(
        reference_route=right_ref,
        temporal_costmaps=temporal_costmaps,
        ego_center=ego_center,
        meters_per_pixel=meters_per_pixel,
        args=args,
    )

    # 3. 判断原始路线是否被阻塞/冲突
    route_blocked = (
        float(base_stats["max_cost"]) >= float(args.behavior_blocked_cost_threshold)
        or float(base_stats["hard_ratio"]) >= float(args.behavior_blocked_hard_ratio)
    )
    # 同时计算原始路线是否安全
    route_safe = (
        float(base_stats["max_cost"]) <= float(args.behavior_free_cost_threshold)
        and float(base_stats["hard_ratio"]) <= float(args.behavior_free_hard_ratio)
    )

    # 4. 判断左右避让通道是否可用
    left_free = is_corridor_free(left_stats, args)
    right_free = is_corridor_free(right_stats, args)

    # 5. 计算安全停车距离(根据原始路线前方第一次遇到hard risk的距离估算一个停车距离)
    first_hard_distance = base_stats.get("first_hard_distance_m")
    if first_hard_distance is None:
        safe_stop_distance = float(args.yield_stop_distance_m)
    else:
        safe_stop_distance = max(float(args.min_yield_stop_distance_m), float(first_hard_distance) - float(args.yield_stop_margin_m))

    # 6. 判断专家车是否已经停住(如果当前速度小于等于0.3m/s,就认为专家车当前基本是停止状态)
    speed = float(max(measurement.get("speed", 0.0), 0.0))
    expert_is_stopped = speed <= float(args.expert_stopped_speed_mps)

    context = {
        "route_blocked": bool(route_blocked),  # 原始 route 前方是否被高风险区域阻塞
        "route_safe": bool(route_safe),        # 原始 route 是否足够安全
        "front_blocked": bool(route_blocked),  # 表示前方被挡住
        "conflict_ahead": bool(route_blocked), # 
        "left_corridor_free": bool(left_free),    # 左侧避让通道是否可用
        "right_corridor_free": bool(right_free),  # 右侧避让通道是否可用
        "expert_is_stopped": bool(expert_is_stopped),  # 当前专家车是否处于停止/近似停止状态
        "current_speed_mps": float(speed),             # 当前自车速度
        "safe_stop_distance_m": float(safe_stop_distance),  # 建议停车距离
        "base_corridor_stats": base_stats,  #    原 route 的风险统计信息
        "left_corridor_stats": left_stats,  # 左避让 route 的风险统计信息
        "right_corridor_stats": right_stats,# 右避让 route 的风险统计信息
    }
    return context

def add_behavior_candidate(
    candidates: List[Dict],
    behavior_name: str,
    reference_route: np.ndarray,
    measurement: Dict,
    active: bool,
    activation_reason: str,
    args,
    reference_mode: Optional[str] = None,
    offset_m: float = 0.0,
) -> None:
    if reference_mode is None:
        reference_mode = behavior_name
    speed_profile = make_speed_profile_for_behavior(behavior_name, measurement, args)
    candidates.append({
        "behavior_id": int(BEHAVIOR_ID[behavior_name]),
        "behavior_name": behavior_name,
        "behavior_active": bool(active),
        "activation_reason": str(activation_reason),
        "reference_mode": str(reference_mode),
        "offset_m": float(offset_m),
        "reference_route": np.asarray(reference_route, dtype=np.float32),
        "speed_profile": speed_profile,
    })

def build_behavior_candidates(
    expert_reference_route: np.ndarray,
    scene_context: Dict,
    measurement: Dict,
    args,
) -> List[Dict]:
    """
    Build behavior-conditioned candidates.

    The candidate set is intentionally semantic. Each item has behavior_id and
    behavior_name so downstream training can use selector labels and valid masks.
    """
    candidates: List[Dict] = []
    include_inactive = (str(args.behavior_candidate_policy).lower() == "all")

    def should_add(active: bool) -> bool:
        return bool(active) or include_inactive

    route_active = True
    if should_add(route_active):
        add_behavior_candidate(
            candidates,
            behavior_name="route_follow",
            reference_route=expert_reference_route,
            measurement=measurement,
            active=route_active,
            activation_reason="always_include_route_follow",
            args=args,
            reference_mode="route_follow",
            offset_m=0.0,
        )

    cautious_active = bool(scene_context.get("front_blocked", False)) or bool(scene_context.get("conflict_ahead", False))
    if should_add(cautious_active):
        add_behavior_candidate(
            candidates,
            behavior_name="cautious_follow",
            reference_route=expert_reference_route,
            measurement=measurement,
            active=cautious_active,
            activation_reason="front_blocked_or_conflict" if cautious_active else "inactive_no_front_risk",
            args=args,
            reference_mode="cautious_follow",
            offset_m=0.0,
        )

    yield_active = bool(scene_context.get("front_blocked", False)) or bool(scene_context.get("conflict_ahead", False))
    if should_add(yield_active):
        yield_ref = make_yield_reference_route(
            base_route=expert_reference_route,
            stop_distance_m=float(scene_context.get("safe_stop_distance_m", args.yield_stop_distance_m)),
            spacing_m=args.reference_spacing_m,
            horizon_m=args.reference_horizon_m,
        )
        add_behavior_candidate(
            candidates,
            behavior_name="yield_stop",
            reference_route=yield_ref,
            measurement=measurement,
            active=yield_active,
            activation_reason="front_blocked_or_conflict" if yield_active else "inactive_no_yield_need",
            args=args,
            reference_mode="yield_stop",
            offset_m=0.0,
        )

    left_active = bool(scene_context.get("left_corridor_free", False)) and bool(scene_context.get("front_blocked", False))
    if should_add(left_active):
        left_offset = -abs(float(args.left_nudge_offset_m))
        left_ref = make_return_to_route_offset_candidate(
            expert_reference_route,
            offset_m=left_offset,
            offset_start_m=args.offset_start_m,
            offset_transition_m=args.offset_transition_m,
            return_start_m=args.nudge_return_start_m,
            return_transition_m=args.nudge_return_transition_m,
        )
        add_behavior_candidate(
            candidates,
            behavior_name="left_nudge",
            reference_route=left_ref,
            measurement=measurement,
            active=left_active,
            activation_reason="front_blocked_and_left_free" if left_active else "inactive_left_not_needed_or_not_free",
            args=args,
            reference_mode="left_nudge",
            offset_m=left_offset,
        )

    right_active = bool(scene_context.get("right_corridor_free", False)) and bool(scene_context.get("front_blocked", False))
    if should_add(right_active):
        right_offset = abs(float(args.right_nudge_offset_m))
        right_ref = make_return_to_route_offset_candidate(
            expert_reference_route,
            offset_m=right_offset,
            offset_start_m=args.offset_start_m,
            offset_transition_m=args.offset_transition_m,
            return_start_m=args.nudge_return_start_m,
            return_transition_m=args.nudge_return_transition_m,
        )
        add_behavior_candidate(
            candidates,
            behavior_name="right_nudge",
            reference_route=right_ref,
            measurement=measurement,
            active=right_active,
            activation_reason="front_blocked_and_right_free" if right_active else "inactive_right_not_needed_or_not_free",
            args=args,
            reference_mode="right_nudge",
            offset_m=right_offset,
        )

    creep_active = bool(scene_context.get("expert_is_stopped", False)) and not bool(scene_context.get("front_blocked", False))
    if should_add(creep_active):
        add_behavior_candidate(
            candidates,
            behavior_name="creep",
            reference_route=expert_reference_route,
            measurement=measurement,
            active=creep_active,
            activation_reason="expert_stopped_but_route_not_blocked" if creep_active else "inactive_not_creep_case",
            args=args,
            reference_mode="creep",
            offset_m=0.0,
        )

    emergency_active = bool(scene_context.get("front_blocked", False)) or bool(scene_context.get("conflict_ahead", False))
    if should_add(emergency_active):
        # Do not use a straight reference here. Emergency braking should still
        # respect the current navigation route direction, otherwise debug/selected
        # route may look like an incorrect straight-line plan in turning scenes.
        emergency_ref = make_yield_reference_route(
            base_route=expert_reference_route,
            stop_distance_m=float(scene_context.get("safe_stop_distance_m", args.yield_stop_distance_m)),
            spacing_m=args.reference_spacing_m,
            horizon_m=args.reference_horizon_m,
        )

        add_behavior_candidate(
            candidates,
            behavior_name="emergency_brake",
            reference_route=emergency_ref,
            measurement=measurement,
            active=emergency_active,
            activation_reason="front_blocked_or_conflict" if emergency_active else "inactive_no_emergency_need",
            args=args,
            reference_mode="emergency_brake",
            offset_m=0.0,
        )

    if len(candidates) == 0:
        add_behavior_candidate(
            candidates,
            behavior_name="route_follow",
            reference_route=expert_reference_route,
            measurement=measurement,
            active=True,
            activation_reason="fallback_no_candidate",
            args=args,
            reference_mode="route_follow",
            offset_m=0.0,
        )

    return candidates

def build_legacy_candidates(expert_route_raw: np.ndarray, measurement: Dict, args) -> List[Dict]:
    """Adapter for the old offset x speed candidate style."""
    reference_candidates = build_reference_candidates(expert_route_raw, args)
    speed_profiles = build_speed_profiles(measurement, args)
    candidates = []
    for ref_cand in reference_candidates:
        for speed_prof in speed_profiles:
            if ref_cand["reference_mode"] == "yield_reference" and speed_prof["speed_mode"] not in ["stop", "slow", "cautious"]:
                continue
            behavior_name = "route_follow"
            if ref_cand["reference_mode"].startswith("left_offset"):
                behavior_name = "left_nudge"
            elif ref_cand["reference_mode"].startswith("right_offset"):
                behavior_name = "right_nudge"
            elif ref_cand["reference_mode"] == "yield_reference":
                behavior_name = "yield_stop"
            elif speed_prof["speed_mode"] in ["slow", "cautious"]:
                behavior_name = "cautious_follow"
            elif speed_prof["speed_mode"] == "stop":
                behavior_name = "yield_stop"
            candidates.append({
                "behavior_id": int(BEHAVIOR_ID.get(behavior_name, 0)),
                "behavior_name": behavior_name,
                "behavior_active": True,
                "activation_reason": "legacy_offset_speed_candidate",
                "reference_mode": ref_cand["reference_mode"],
                "offset_m": float(ref_cand["offset_m"]),
                "reference_route": ref_cand["reference_route"],
                "speed_profile": speed_prof,
            })
    return candidates
