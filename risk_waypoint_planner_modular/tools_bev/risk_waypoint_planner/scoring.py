# -*- coding: utf-8 -*-

from .common import *
from .costmap_utils import local_points_to_pixels, sample_cost_bilinear
from .geometry_utils import min_distance_to_polyline_points, sample_polyline_by_s
from .temporal_costmap import temporal_index_for_dense_step
from .behavior_candidates import BEHAVIOR_NAMES

def build_vehicle_footprint_points(
    xy: np.ndarray,
    yaw: np.ndarray,
    args,
) -> np.ndarray:
    """
    Return sampled footprint points for every state.

    Shape: [N * K, 2]
    """
    xy = np.asarray(xy, dtype=np.float32)
    yaw = np.asarray(yaw, dtype=np.float32)

    if not args.score_footprint:
        return xy

    ex = float(args.ego_extent_x_m)
    ey = float(args.ego_extent_y_m)

    local_samples = np.asarray([
        [0.0, 0.0],
        [ex, 0.0],
        [-ex, 0.0],
        [0.0, ey],
        [0.0, -ey],
        [ex, ey],
        [ex, -ey],
        [-ex, ey],
        [-ex, -ey],
    ], dtype=np.float32)

    pts = []
    for p, th in zip(xy, yaw):
        c = math.cos(float(th))
        s = math.sin(float(th))
        for q in local_samples:
            # vehicle to world, x forward, y right
            x_w = p[0] + c * q[0] - s * q[1]
            y_w = p[1] + s * q[0] + c * q[1]
            pts.append([x_w, y_w])

    return np.asarray(pts, dtype=np.float32)

def build_vehicle_footprint_points_single(
    xy: np.ndarray,
    yaw: float,
    args,
) -> np.ndarray:
    """
    Return footprint sample points for one rollout state in current-frame
    ego-local coordinates.
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(2)

    if not args.score_footprint:
        return xy.reshape(1, 2).astype(np.float32)

    ex = float(args.ego_extent_x_m)
    ey = float(args.ego_extent_y_m)

    local_samples = np.asarray([
        [0.0, 0.0],
        [ex, 0.0],
        [-ex, 0.0],
        [0.0, ey],
        [0.0, -ey],
        [ex, ey],
        [ex, -ey],
        [-ex, ey],
        [-ex, -ey],
    ], dtype=np.float32)

    c = math.cos(float(yaw))
    s = math.sin(float(yaw))

    pts = np.zeros_like(local_samples, dtype=np.float32)
    pts[:, 0] = xy[0] + c * local_samples[:, 0] - s * local_samples[:, 1]
    pts[:, 1] = xy[1] + s * local_samples[:, 0] + c * local_samples[:, 1]
    return pts.astype(np.float32)

def score_rollout(
    rollout: Dict,
    reference_candidate: Dict,
    expert_reference_route: np.ndarray,
    costmap: np.ndarray,
    ego_center: List[float],
    meters_per_pixel: float,
    args,
    temporal_costmaps: Optional[List[np.ndarray]] = None,
    temporal_frames: Optional[List[str]] = None,
) -> Dict:
    """
    Score one candidate rollout.

    If temporal_costmaps is provided, dense rollout states are scored against
    the time-aligned costmap. temporal_costmaps[0] is the current frame;
    temporal_costmaps[k] is future frame k warped into the current ego-local
    coordinate system. This is how future true motion of surrounding actors is
    injected into planning.
    """
    if temporal_costmaps is None or len(temporal_costmaps) == 0:
        temporal_costmaps = [costmap]
    if temporal_frames is None or len(temporal_frames) != len(temporal_costmaps):
        temporal_frames = [str(i) for i in range(len(temporal_costmaps))]

    sampled_chunks = []
    valid_chunks = []
    temporal_indices_used = []
    temporal_cost_by_index: Dict[int, List[float]] = {}

    dense_xy = np.asarray(rollout["dense_xy"], dtype=np.float32)
    dense_yaw = np.asarray(rollout["dense_yaw"], dtype=np.float32)

    for step_i, (xy_i, yaw_i) in enumerate(zip(dense_xy, dense_yaw)):
        temporal_idx = temporal_index_for_dense_step(
            step_idx=step_i,
            num_temporal_maps=len(temporal_costmaps),
            args=args,
        )
        temporal_indices_used.append(int(temporal_idx))

        score_points = build_vehicle_footprint_points_single(
            xy=xy_i,
            yaw=float(yaw_i),
            args=args,
        )
        pixels = local_points_to_pixels(score_points, ego_center, meters_per_pixel)
        sampled_cost_i, valid_i = sample_cost_bilinear(
            temporal_costmaps[temporal_idx],
            pixels,
            out_of_bounds_cost=args.out_of_bounds_cost,
        )

        sampled_chunks.append(sampled_cost_i)
        valid_chunks.append(valid_i)
        temporal_cost_by_index.setdefault(int(temporal_idx), []).extend(
            sampled_cost_i.astype(float).tolist()
        )

    if len(sampled_chunks) == 0:
        sampled_cost = np.zeros((1,), dtype=np.float32)
        valid = np.zeros((1,), dtype=bool)
    else:
        sampled_cost = np.concatenate(sampled_chunks).astype(np.float32)
        valid = np.concatenate(valid_chunks).astype(bool)

    mean_cost = float(np.mean(sampled_cost))
    max_cost = float(np.max(sampled_cost))
    sum_cost = float(np.sum(sampled_cost))
    hard_ratio = float(np.mean(sampled_cost >= args.hard_cost_threshold))
    out_ratio = float(1.0 - np.mean(valid.astype(np.float32)))

    temporal_mean_costs = {
        str(k): float(np.mean(v)) if len(v) else 0.0
        for k, v in temporal_cost_by_index.items()
    }
    temporal_max_costs = {
        str(k): float(np.max(v)) if len(v) else 0.0
        for k, v in temporal_cost_by_index.items()
    }

    waypoints = rollout["waypoints"]
    dev = min_distance_to_polyline_points(waypoints, expert_reference_route)
    mean_route_deviation = float(np.mean(dev)) if len(dev) else 0.0
    max_route_deviation = float(np.max(dev)) if len(dev) else 0.0

    controls = rollout["dense_controls"]
    steer = controls[:, 0] if len(controls) else np.zeros(1, dtype=np.float32)
    acc = controls[:, 1] if len(controls) else np.zeros(1, dtype=np.float32)
    steer_rate = np.diff(steer) * float(args.model_fps) if len(steer) > 1 else np.zeros(1, dtype=np.float32)

    lat_accs = np.asarray([aux["lat_acc_mps2"] for aux in rollout["dense_aux"]], dtype=np.float32)
    yaw_rates = np.asarray([aux["yaw_rate_radps"] for aux in rollout["dense_aux"]], dtype=np.float32)

    max_abs_lat_acc = float(np.max(np.abs(lat_accs))) if len(lat_accs) else 0.0
    max_abs_yaw_rate = float(np.max(np.abs(yaw_rates))) if len(yaw_rates) else 0.0
    mean_abs_acc = float(np.mean(np.abs(acc)))
    mean_abs_steer = float(np.mean(np.abs(steer)))
    mean_abs_steer_rate = float(np.mean(np.abs(steer_rate)))

    progress_m = float(np.linalg.norm(waypoints[-1] - waypoints[0])) if len(waypoints) >= 2 else 0.0

    route_deviation_cost = args.route_deviation_weight * mean_route_deviation
    comfort_cost = (
        args.acc_weight * mean_abs_acc
        + args.steer_weight * mean_abs_steer
        + args.steer_rate_weight * mean_abs_steer_rate
        + args.lat_acc_weight * max_abs_lat_acc
        + args.yaw_rate_weight * max_abs_yaw_rate
    )

    behavior_name = str(reference_candidate.get("behavior_name", reference_candidate.get("reference_mode", "unknown")))
    behavior_id = int(reference_candidate.get("behavior_id", -1))
    behavior_active = bool(reference_candidate.get("behavior_active", True))
    behavior_prior_cost = 0.0

    if not behavior_active:
        behavior_prior_cost += float(args.inactive_behavior_penalty)

    if behavior_name in ["yield_stop", "emergency_brake"]:
        # Stop-like behaviors should not win purely because they minimize risk.
        # They are selected only when active or when all moving candidates are unsafe.
        behavior_prior_cost += float(args.stop_behavior_prior_cost)

    if behavior_name in ["left_nudge", "right_nudge"]:
        behavior_prior_cost += float(args.nudge_behavior_prior_cost)

    total_score = (
        args.mean_cost_weight * mean_cost
        + args.max_cost_weight * max_cost
        + args.hard_ratio_weight * hard_ratio
        + args.out_of_bounds_weight * out_ratio
        + route_deviation_cost
        + comfort_cost
        + behavior_prior_cost
    )

    allowed = True
    reasons = []

    if not behavior_active and not args.allow_inactive_behavior_selection:
        allowed = False
        reasons.append("behavior_inactive")

    if out_ratio > args.max_out_of_bounds_ratio:
        allowed = False
        reasons.append(f"out_of_bounds_ratio={out_ratio:.3f}>{args.max_out_of_bounds_ratio:.3f}")
    if hard_ratio > args.max_hard_ratio:
        allowed = False
        reasons.append(f"hard_ratio={hard_ratio:.3f}>{args.max_hard_ratio:.3f}")
    if max_abs_lat_acc > args.max_lateral_accel_mps2:
        allowed = False
        reasons.append(f"lat_acc={max_abs_lat_acc:.3f}>{args.max_lateral_accel_mps2:.3f}")
    if max_abs_yaw_rate > args.max_yaw_rate_radps:
        allowed = False
        reasons.append(f"yaw_rate={max_abs_yaw_rate:.3f}>{args.max_yaw_rate_radps:.3f}")
    if max_route_deviation > args.max_route_deviation_m:
        allowed = False
        reasons.append(f"route_deviation={max_route_deviation:.3f}>{args.max_route_deviation_m:.3f}")

    # Emergency brake is saved as a diagnostic/safety candidate, but by default it
    # should not become the selected risk-planned waypoint label. Otherwise many
    # turning or blocked frames may collapse to a straight braking trajectory.
    if behavior_name == "emergency_brake" and not args.allow_emergency_selection:
        allowed = False
        reasons.append("emergency_brake_not_selectable_by_default")

    # Optional: keep yield_stop unselectable unless requested.
    # if behavior_name == "yield_stop" and not args.allow_yield_selection and not bool(reference_candidate.get("behavior_active", True)):
    #     allowed = False
    #     reasons.append("yield_inactive_not_selectable")
    if behavior_name == "yield_stop" and not args.allow_yield_selection:
        allowed = False
        reasons.append("yield_stop_not_selectable_by_default")

    temporal_indices_unique = sorted(set(int(i) for i in temporal_indices_used))
    temporal_frames_used = [temporal_frames[i] for i in temporal_indices_unique if i < len(temporal_frames)]

    return {
        "behavior_id": behavior_id,
        "behavior_name": behavior_name,
        "behavior_active": behavior_active,
        "activation_reason": str(reference_candidate.get("activation_reason", "")),
        "reference_mode": reference_candidate["reference_mode"],
        "offset_m": float(reference_candidate["offset_m"]),
        "speed_mode": rollout["speed_mode"],
        "target_speed": float(rollout["target_speed"]),
        "score": float(total_score),
        "mean_cost": mean_cost,
        "max_cost": max_cost,
        "sum_cost": sum_cost,
        "hard_ratio": hard_ratio,
        "out_of_bounds_ratio": out_ratio,
        "mean_route_deviation": mean_route_deviation,
        "max_route_deviation": max_route_deviation,
        "comfort_cost": float(comfort_cost),
        "route_deviation_cost": float(route_deviation_cost),
        "behavior_prior_cost": float(behavior_prior_cost),
        "score_breakdown": {
            "mean_cost_term": float(args.mean_cost_weight * mean_cost),
            "max_cost_term": float(args.max_cost_weight * max_cost),
            "hard_ratio_term": float(args.hard_ratio_weight * hard_ratio),
            "out_of_bounds_term": float(args.out_of_bounds_weight * out_ratio),
            "route_deviation_cost": float(route_deviation_cost),
            "comfort_cost": float(comfort_cost),
            "behavior_prior_cost": float(behavior_prior_cost),
        },
        "mean_abs_acc": mean_abs_acc,
        "mean_abs_steer": mean_abs_steer,
        "mean_abs_steer_rate": mean_abs_steer_rate,
        "max_abs_lateral_accel": max_abs_lat_acc,
        "max_abs_yaw_rate": max_abs_yaw_rate,
        "progress_m": progress_m,
        "temporal_score_enabled": bool(len(temporal_costmaps) > 1),
        "temporal_indices_used": temporal_indices_unique,
        "temporal_frames_used": temporal_frames_used,
        "temporal_mean_costs": temporal_mean_costs,
        "temporal_max_costs": temporal_max_costs,
        "allowed": bool(allowed),
        "reasons": reasons,
    }

def select_best_rollout(scored_rollouts: List[Dict]) -> int:
    """
    Select the best dynamically feasible risk-planned rollout.

    Return:
        >=0: index of the best allowed candidate
        -1: no allowed candidate exists

    Important:
        If no candidate is allowed, do NOT select emergency_brake here.
        The caller should fall back to expert future waypoints and mark
        risk_label_valid=False.
    """
    allowed = [
        (i, r)
        for i, r in enumerate(scored_rollouts)
        if bool(r["info"].get("allowed", False))
    ]

    if len(allowed) > 0:
        best_i, _ = min(allowed, key=lambda ir: ir[1]["info"]["score"])
        return int(best_i)

    return -1

def build_valid_behavior_mask(scored_rollouts: List[Dict]) -> List[int]:
    mask = [0 for _ in BEHAVIOR_NAMES]
    for r in scored_rollouts:
        info = r.get("info", {})
        bid = int(info.get("behavior_id", -1))
        if 0 <= bid < len(mask) and bool(info.get("allowed", False)):
            mask[bid] = 1
    return mask

def estimate_yaws_and_speeds_from_waypoints(
    waypoints: np.ndarray,
    future_fps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate yaw and speed from expert future waypoints.

    This is only used for expert fallback debug/output.
    The fallback frame is not used as valid risk-planned supervision.
    """
    waypoints = np.asarray(waypoints, dtype=np.float32)

    if waypoints.ndim != 2 or waypoints.shape[1] < 2 or len(waypoints) == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    points = waypoints[:, :2]
    prev = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), points[:-1]],
        axis=0,
    )
    delta = points - prev

    yaws = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)
    speeds = (np.linalg.norm(delta, axis=1) * float(future_fps)).astype(np.float32)

    return yaws, speeds

def build_expert_fallback_rollout(
    expert_reference_route: np.ndarray,
    expert_future_waypoints: Optional[np.ndarray],
    args,
    fallback_reason: str = "no_allowed_behavior_candidate",
) -> Dict:
    """
    Build a selected fallback item from expert route + expert future waypoints.

    This fallback is only for output/debug stability. It must NOT be treated
    as valid risk-planned supervision, so:
        allowed=False
        risk_label_valid=False
        selector_label=-1
    """
    expert_reference_route = np.asarray(expert_reference_route, dtype=np.float32)

    if expert_future_waypoints is not None:
        waypoints = np.asarray(expert_future_waypoints, dtype=np.float32)
        if waypoints.ndim == 2 and waypoints.shape[1] >= 2 and len(waypoints) > 0:
            waypoints = waypoints[:, :2].astype(np.float32)
        else:
            waypoints = None
    else:
        waypoints = None

    # If future expert waypoints are unavailable near the end of a route,
    # fall back to route samples only for debug/output. This frame is still invalid
    # for risk-planned training because allowed=False.
    if waypoints is None or len(waypoints) == 0:
        query_s = np.arange(
            1,
            int(args.num_future_waypoints) + 1,
            dtype=np.float32,
        ) * float(args.reference_spacing_m)
        waypoints = sample_polyline_by_s(expert_reference_route, query_s)

    # Match expected number of future waypoints as much as possible.
    if len(waypoints) > int(args.num_future_waypoints):
        waypoints = waypoints[:int(args.num_future_waypoints)]
    elif len(waypoints) < int(args.num_future_waypoints):
        pad_num = int(args.num_future_waypoints) - len(waypoints)
        last = waypoints[-1:].copy()
        waypoints = np.concatenate(
            [waypoints, np.repeat(last, pad_num, axis=0)],
            axis=0,
        )

    yaws, speeds = estimate_yaws_and_speeds_from_waypoints(
        waypoints,
        future_fps=float(args.future_fps),
    )

    controls = np.zeros((len(waypoints), 2), dtype=np.float32)

    info = {
        "behavior_id": -1,
        "behavior_name": "expert_fallback",
        "behavior_active": False,
        "activation_reason": fallback_reason,
        "reference_mode": "expert_reference",
        "speed_mode": "expert_future",
        "offset_m": 0.0,

        # This fallback is intentionally invalid as risk-planned supervision.
        "allowed": False,
        "reasons": [fallback_reason],
        "fallback_to_expert": True,
        "fallback_reason": fallback_reason,

        # Keep numeric fields available for debug text and JSON consistency.
        "score": 1.0e9,
        "mean_cost": 0.0,
        "max_cost": 0.0,
        "hard_ratio": 0.0,
        "route_deviation_mean_m": 0.0,
        "route_deviation_max_m": 0.0,
        "mean_abs_acc": 0.0,
        "mean_abs_steer": 0.0,
        "mean_abs_steer_rate": 0.0,
        "max_abs_lateral_accel": 0.0,
        "max_abs_yaw_rate": 0.0,
        "progress_m": float(np.linalg.norm(waypoints[-1] - waypoints[0])) if len(waypoints) > 1 else 0.0,
        "temporal_score_enabled": False,
        "temporal_indices_used": [],
        "temporal_frames_used": [],
        "temporal_mean_costs": [],
        "temporal_max_costs": [],
    }

    rollout = {
        "waypoints": waypoints.astype(np.float32),
        "yaws": yaws.astype(np.float32),
        "speeds": speeds.astype(np.float32),
        "controls": controls,
        "dense_xy": waypoints.astype(np.float32),
        "dense_yaw": yaws.astype(np.float32),
        "dense_speed": speeds.astype(np.float32),
        "dense_controls": controls,
        "dense_aux": [],
        "target_speed": float(speeds[0]) if len(speeds) > 0 else 0.0,
        "speed_mode": "expert_future",
    }

    return {
        "info": info,
        "reference_route": expert_reference_route.astype(np.float32),
        "rollout": rollout,
    }
