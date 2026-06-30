# -*- coding: utf-8 -*-

from .common import *

def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def mps_to_kmh(v: float) -> float:
    return float(v) * 3.6

def format_speed_en(v_mps: float) -> str:
    return f"{v_mps:.2f} m/s ({mps_to_kmh(v_mps):.1f} km/h)"

def format_speed_zh(v_mps: float) -> str:
    return f"{v_mps:.2f} m/s（约 {mps_to_kmh(v_mps):.1f} km/h）"

def estimate_path_length(points: Optional[np.ndarray]) -> float:
    if points is None:
        return 0.0
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:, :2] - pts[:-1, :2], axis=1)))

def infer_expert_speed_stats(
    expert_future_waypoints: Optional[np.ndarray],
    future_fps: float,
) -> Dict:
    """
    Estimate expert future speed statistics from future ego waypoints.

    This is used only for language annotation. It does not affect planning.
    """
    if expert_future_waypoints is None:
        return {
            "available": False,
            "mean_speed_mps": 0.0,
            "max_speed_mps": 0.0,
            "path_length_m": 0.0,
        }

    pts = np.asarray(expert_future_waypoints, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return {
            "available": False,
            "mean_speed_mps": 0.0,
            "max_speed_mps": 0.0,
            "path_length_m": estimate_path_length(pts),
        }

    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), pts[:-1, :2]], axis=0)
    delta = pts[:, :2] - prev
    speeds = np.linalg.norm(delta, axis=1) * float(future_fps)

    return {
        "available": True,
        "mean_speed_mps": float(np.mean(speeds)),
        "max_speed_mps": float(np.max(speeds)),
        "path_length_m": estimate_path_length(pts),
    }

def behavior_name_to_chinese(behavior_name: str) -> str:
    table = {
        "route_follow": "沿导航路线正常行驶",
        "cautious_follow": "谨慎跟车/减速通过",
        "yield_stop": "停车让行",
        "left_nudge": "向左侧风险规避",
        "right_nudge": "向右侧风险规避",
        "creep": "低速爬行",
        "emergency_brake": "紧急制动",
        "expert_fallback": "沿当前路线保守通行",
    }
    return table.get(str(behavior_name), str(behavior_name))

def behavior_name_to_english(behavior_name: str) -> str:
    table = {
        "route_follow": "follow the navigation route",
        "cautious_follow": "slow down and follow cautiously",
        "yield_stop": "yield and stop",
        "left_nudge": "nudge left to avoid risk",
        "right_nudge": "nudge right to avoid risk",
        "creep": "creep forward slowly",
        "emergency_brake": "perform emergency braking",
        "expert_fallback": "proceed conservatively along the current route",
    }
    return table.get(str(behavior_name), str(behavior_name))

def describe_scene_context_zh(scene_context: Dict, lateral_info: Optional[Dict] = None) -> str:
    """
    Human-readable scene explanation for language/QA training.

    This function intentionally avoids technical planner metrics such as cost,
    score, and hard-ratio. Those values remain available in scene_context and
    selected_info for debugging.
    """
    route_blocked = bool(scene_context.get("route_blocked", False))
    front_blocked = bool(scene_context.get("front_blocked", False))
    conflict_ahead = bool(scene_context.get("conflict_ahead", False))
    left_free = bool(scene_context.get("left_corridor_free", False))
    right_free = bool(scene_context.get("right_corridor_free", False))
    speed = safe_float(scene_context.get("current_speed_mps", 0.0))

    parts = [f"自车当前速度为 {format_speed_zh(speed)}。"]

    if route_blocked or front_blocked or conflict_ahead:
        parts.append(
            "导航路线前方存在潜在占用或碰撞风险，继续按照原速度前进可能会与前方交通参与者距离过近。"
        )
    else:
        parts.append(
            "导航路线前方没有明显阻塞，车辆可以优先保持原导航方向行驶。"
        )

    lateral_info = lateral_info or {}
    has_lateral = bool(lateral_info.get("has_lateral_maneuver", False))
    direction_zh = str(lateral_info.get("direction_zh", "无明显横向避让"))
    lat_m = safe_float(lateral_info.get("max_abs_lateral_m", 0.0))

    if has_lateral:
        parts.append(
            f"最终规划轨迹包含明显的横向动作，车辆会{direction_zh}，"
            f"最大横向偏移约 {lat_m:.2f} m，用于绕开前方风险区域。"
        )
    else:
        if left_free and right_free:
            parts.append("左右两侧都有一定可通行空间，因此可以考虑横向避让。")
        elif left_free and not right_free:
            parts.append("左侧相对更安全，右侧不适合明显横向避让。")
        elif right_free and not left_free:
            parts.append("右侧相对更安全，左侧不适合明显横向避让。")
        else:
            parts.append("左右两侧都没有明显安全的横向避让空间，因此更适合沿当前路线减速跟车。")

    return "".join(parts)

def describe_scene_context_en(scene_context: Dict, lateral_info: Optional[Dict] = None) -> str:
    """
    Human-readable scene explanation for language/QA training.

    This function avoids internal planner metrics and describes the driving
    situation in natural language.
    """
    route_blocked = bool(scene_context.get("route_blocked", False))
    front_blocked = bool(scene_context.get("front_blocked", False))
    conflict_ahead = bool(scene_context.get("conflict_ahead", False))
    left_free = bool(scene_context.get("left_corridor_free", False))
    right_free = bool(scene_context.get("right_corridor_free", False))
    speed = safe_float(scene_context.get("current_speed_mps", 0.0))

    parts = [f"The ego vehicle is moving at {format_speed_en(speed)}. "]

    if route_blocked or front_blocked or conflict_ahead:
        parts.append(
            "There is a potential obstacle or occupied space along the route ahead, "
            "so continuing at the original speed may bring the ego vehicle too close to traffic in front. "
        )
    else:
        parts.append(
            "The route ahead is not clearly blocked, so the ego vehicle can mainly keep following the navigation route. "
        )

    lateral_info = lateral_info or {}
    has_lateral = bool(lateral_info.get("has_lateral_maneuver", False))
    direction_en = str(lateral_info.get("direction_en", "no clear lateral maneuver"))
    lat_m = safe_float(lateral_info.get("max_abs_lateral_m", 0.0))

    if has_lateral:
        parts.append(
            f"The selected plan also {direction_en}, with a maximum lateral shift of about "
            f"{lat_m:.2f} m, to move around the risky area ahead."
        )
    else:
        if left_free and right_free:
            parts.append("Both sides appear to have some usable space, so a lateral avoidance maneuver may be possible.")
        elif left_free and not right_free:
            parts.append("The left side looks more usable, while the right side is not suitable for a clear avoidance maneuver.")
        elif right_free and not left_free:
            parts.append("The right side looks more usable, while the left side is not suitable for a clear avoidance maneuver.")
        else:
            parts.append("Neither side provides a clearly safe lateral avoidance corridor, so slowing down behind the traffic ahead is preferred.")
    
    return "".join(parts)

def describe_selected_decision_zh(
    selected_info: Dict,
    scene_context: Optional[Dict] = None,
    lateral_info: Optional[Dict] = None,
) -> str:
    """
    Human-readable explanation of why the selected behavior is chosen.

    Avoid exposing score/cost/hard-ratio in the main QA text.
    """
    lateral_info = lateral_info or {}
    has_lateral = bool(lateral_info.get("has_lateral_maneuver", False))
    direction_zh = str(lateral_info.get("direction_zh", ""))
    lat_m = safe_float(lateral_info.get("max_abs_lateral_m", 0.0))

    behavior_name = str(selected_info.get("behavior_name", "unknown"))
    behavior_zh = behavior_name_to_chinese(behavior_name)
    allowed = bool(selected_info.get("allowed", False))

    scene_context = scene_context or {}
    front_blocked = bool(scene_context.get("front_blocked", False)) or bool(scene_context.get("conflict_ahead", False))
    left_free = bool(scene_context.get("left_corridor_free", False))
    right_free = bool(scene_context.get("right_corridor_free", False))

    if not allowed:
        reasons = selected_info.get("reasons", [])
        return (
            f"当前输出为“{behavior_zh}”，但它不是有效的风险规划监督标签。"
            f"该轨迹没有通过可行性检查，原因包括：{reasons}。"
        )

    if behavior_name == "route_follow":
        return (
            "系统选择继续沿导航路线行驶，因为当前路线可以保持通行，"
            "没有必要进行明显的减速、停车或横向避让。"
        )

    if behavior_name == "cautious_follow" and has_lateral:
        return (
            f"系统选择谨慎通过，同时执行{direction_zh}。"
            f"前方路线存在潜在风险，规划轨迹没有直接沿原方向硬闯，"
            f"而是在降低速度的同时产生约 {lat_m:.2f} m 的横向偏移，"
            f"用于绕开前方车辆或占用区域。"
        )

    if behavior_name == "cautious_follow":
        if front_blocked and not left_free and not right_free:
            return (
                "系统选择减速跟车，因为导航路线前方存在潜在占用或碰撞风险，"
                "同时左右两侧都没有明显安全的避让空间。相比强行绕行，沿当前路线降低速度更加稳妥。"
            )
        if front_blocked:
            return (
                "系统选择减速跟车，因为前方路线存在风险。该策略可以保留导航方向，"
                "同时降低与前方交通参与者接近过快的风险。"
            )
        return (
            "系统选择谨慎跟车，因为该轨迹在保持导航方向的同时更加平稳和保守。"
        )

    if behavior_name == "left_nudge":
        return (
            "系统选择向左侧进行轻微避让，因为当前路线前方存在风险，"
            "且左侧比原路线更适合作为临时避让空间。"
        )

    if behavior_name == "right_nudge":
        return (
            "系统选择向右侧进行轻微避让，因为当前路线前方存在风险，"
            "且右侧比原路线更适合作为临时避让空间。"
        )

    if behavior_name == "yield_stop":
        return (
            "系统选择停车让行，因为前方存在明显冲突风险，继续前进可能不安全。"
        )

    if behavior_name == "creep":
        return (
            "系统选择低速爬行，因为当前场景需要谨慎向前确认可通行空间。"
        )

    if behavior_name == "emergency_brake":
        return (
            "系统选择紧急制动，因为前方风险较高，需要优先降低速度以避免危险。"
        )

    if behavior_name == "expert_fallback":
        return (
            "由于没有找到可靠的风险规划候选，系统回退到专家未来轨迹作为输出。"
        )

    return f"系统选择“{behavior_zh}”，因为该候选轨迹更符合当前场景的安全需求。"

def describe_selected_decision_en(
    selected_info: Dict,
    scene_context: Optional[Dict] = None,
    lateral_info: Optional[Dict] = None,
) -> str:
    """
    Human-readable explanation of why the selected behavior is chosen.

    This is intended for QA/language supervision, not planner debugging.
    """
    lateral_info = lateral_info or {}
    has_lateral = bool(lateral_info.get("has_lateral_maneuver", False))
    direction_en = str(lateral_info.get("direction_en", ""))
    lat_m = safe_float(lateral_info.get("max_abs_lateral_m", 0.0))

    behavior_name = str(selected_info.get("behavior_name", "unknown"))
    behavior_en = behavior_name_to_english(behavior_name)
    allowed = bool(selected_info.get("allowed", False))

    scene_context = scene_context or {}
    front_blocked = bool(scene_context.get("front_blocked", False)) or bool(scene_context.get("conflict_ahead", False))
    left_free = bool(scene_context.get("left_corridor_free", False))
    right_free = bool(scene_context.get("right_corridor_free", False))

    if not allowed:
        reasons = selected_info.get("reasons", [])
        return (
            f"The output behavior is to {behavior_en}, but it is not a valid risk-planned label. "
            f"It failed the feasibility check for the following reasons: {reasons}."
        )

    if behavior_name == "route_follow":
        return (
            "The planner keeps following the navigation route because the route is usable "
            "and there is no need for a clear slowdown, stop, or lateral avoidance maneuver."
        )

    if behavior_name == "cautious_follow" and has_lateral:
        return (
            f"The planner chooses a cautious passing maneuver and {direction_en}. "
            f"There is potential risk along the route ahead, so the plan does not simply "
            f"continue straight at the original pace. Instead, it reduces speed while making "
            f"about {lat_m:.2f} m of lateral movement to move around the vehicle or occupied area ahead."
        )

    if behavior_name == "cautious_follow":
        if front_blocked and not left_free and not right_free:
            return (
                "The planner slows down and follows cautiously because there is potential risk ahead "
                "on the ego route, and neither side offers a clearly safe space for lateral avoidance. "
                "Slowing down while staying on the route is safer than forcing a side maneuver."
            )
        if front_blocked:
            return (
                "The planner slows down and follows cautiously because there is potential risk ahead. "
                "This keeps the vehicle aligned with the route while reducing the chance of getting too close "
                "to traffic in front."
            )
        return (
            "The planner chooses a cautious following behavior because it keeps the route direction "
            "while producing a smoother and more conservative plan."
        )

    if behavior_name == "left_nudge":
        return (
            "The planner nudges left because the original route ahead is risky and the left side "
            "offers a better temporary avoidance space."
        )

    if behavior_name == "right_nudge":
        return (
            "The planner nudges right because the original route ahead is risky and the right side "
            "offers a better temporary avoidance space."
        )

    if behavior_name == "yield_stop":
        return (
            "The planner chooses to yield and stop because there is a clear conflict risk ahead, "
            "and moving forward may be unsafe."
        )

    if behavior_name == "creep":
        return (
            "The planner creeps forward slowly because the scene requires cautious progress "
            "to check whether the space ahead is clear."
        )

    if behavior_name == "emergency_brake":
        return (
            "The planner chooses emergency braking because the risk ahead is high and reducing speed "
            "is the safest immediate response."
        )

    if behavior_name == "expert_fallback":
        return (
            "No reliable risk-planned candidate is available, so the system falls back to the expert future trajectory."
        )

    return f"The planner selects to {behavior_en} because this candidate better matches the safety needs of the scene."

def analyze_selected_lateral_motion(
    selected_rollout: Dict,
    selected_reference_route: Optional[np.ndarray],
    lateral_threshold_m: float = 0.8,
) -> Dict:
    """
    Analyze whether the selected plan contains a meaningful lateral maneuver.

    Coordinate convention:
        y > 0: right
        y < 0: left

    This is only used for language annotation. It does not affect planning.
    """
    waypoints = np.asarray(selected_rollout.get("waypoints", []), dtype=np.float32)

    if selected_reference_route is not None:
        ref = np.asarray(selected_reference_route, dtype=np.float32)
    else:
        ref = np.zeros((0, 2), dtype=np.float32)

    ys = []

    if waypoints.ndim == 2 and waypoints.shape[1] >= 2 and len(waypoints) > 0:
        ys.extend(waypoints[:, 1].astype(float).tolist())

    if ref.ndim == 2 and ref.shape[1] >= 2 and len(ref) > 0:
        ys.extend(ref[:, 1].astype(float).tolist())

    if len(ys) == 0:
        return {
            "has_lateral_maneuver": False,
            "direction": "none",
            "direction_zh": "无明显横向避让",
            "direction_en": "no clear lateral maneuver",
            "max_left_m": 0.0,
            "max_right_m": 0.0,
            "max_abs_lateral_m": 0.0,
        }

    min_y = float(np.min(ys))
    max_y = float(np.max(ys))

    max_left = abs(min_y) if min_y < 0.0 else 0.0
    max_right = max_y if max_y > 0.0 else 0.0
    max_abs = max(max_left, max_right)

    if max_abs < float(lateral_threshold_m):
        direction = "none"
        direction_zh = "无明显横向避让"
        direction_en = "no clear lateral maneuver"
        has_lateral = False
    elif max_left >= max_right:
        direction = "left"
        direction_zh = "向左侧避让"
        direction_en = "shifts left to avoid the risk"
        has_lateral = True
    else:
        direction = "right"
        direction_zh = "向右侧避让"
        direction_en = "shifts right to avoid the risk"
        has_lateral = True

    return {
        "has_lateral_maneuver": bool(has_lateral),
        "direction": direction,
        "direction_zh": direction_zh,
        "direction_en": direction_en,
        "max_left_m": float(max_left),
        "max_right_m": float(max_right),
        "max_abs_lateral_m": float(max_abs),
    }

def compare_risk_and_expert_motion_zh(
    selected_rollout: Dict,
    expert_future_waypoints: Optional[np.ndarray],
    future_fps: float,
    lateral_info: Optional[Dict] = None,
) -> str:
    """
    Compare selected planned waypoints with expert future waypoints in a
    human-readable way.

    Keep speed/path-length comparison because it is important evidence for why
    the selected waypoints differ from the expert trajectory.
    """
    risk_speeds = np.asarray(selected_rollout.get("speeds", []), dtype=np.float32)
    risk_waypoints = np.asarray(selected_rollout.get("waypoints", []), dtype=np.float32)

    risk_mean_speed = float(np.mean(risk_speeds)) if len(risk_speeds) > 0 else 0.0
    risk_path_len = estimate_path_length(risk_waypoints)

    expert_stats = infer_expert_speed_stats(expert_future_waypoints, future_fps)

    horizon_s = 0.0
    if future_fps > 1e-6 and len(risk_waypoints) > 0:
        horizon_s = float(len(risk_waypoints)) / float(future_fps)

    if not expert_stats["available"]:
        return (
            f"在未来约 {horizon_s:.1f} 秒内，规划轨迹平均速度为 {format_speed_zh(risk_mean_speed)}，"
            f"行驶距离约 {risk_path_len:.2f} m。当前帧没有可用的专家未来轨迹用于对比。"
        )

    expert_mean_speed = expert_stats["mean_speed_mps"]
    expert_path_len = expert_stats["path_length_m"]
    diff = risk_mean_speed - expert_mean_speed

    if diff < -0.5:
        trend = (
            "规划轨迹比专家轨迹更慢，说明系统主动降低速度，以便和前方交通参与者保持更安全的距离。"
        )
    elif diff > 0.5:
        if bool((lateral_info or {}).get("has_lateral_maneuver", False)):
            trend = (
                "规划轨迹比专家轨迹更主动。相比停在原车道附近等待，系统选择通过受控的横向避让绕开前方风险区域。"
            )
        else:
            trend = (
                "规划轨迹比专家轨迹更快，说明系统认为前方路径仍具备通行条件，可以更积极地通过。"
            )
    else:
        trend = (
            "规划轨迹与专家轨迹速度接近，说明系统基本保持了专家驾驶节奏。"
        )

    lateral_info = lateral_info or {}
    if bool(lateral_info.get("has_lateral_maneuver", False)):
        lateral_sentence = (
            f"同时，规划轨迹还会{lateral_info.get('direction_zh', '进行横向避让')}，"
            f"最大横向偏移约 {safe_float(lateral_info.get('max_abs_lateral_m', 0.0)):.2f} m。"
        )
    else:
        lateral_sentence = ""

    return (
        f"在未来约 {horizon_s:.1f} 秒内，规划轨迹预计行驶约 {risk_path_len:.2f} m，"
        f"平均速度为 {format_speed_zh(risk_mean_speed)}；"
        f"专家轨迹预计行驶约 {expert_path_len:.2f} m，"
        f"平均速度为 {format_speed_zh(expert_mean_speed)}。"
        f"{trend}{lateral_sentence}"
    )

def compare_risk_and_expert_motion_en(
    selected_rollout: Dict,
    expert_future_waypoints: Optional[np.ndarray],
    future_fps: float,
    lateral_info: Optional[Dict] = None,
) -> str:
    """
    Compare selected planned waypoints with expert future waypoints in a
    human-readable way.

    This keeps the important speed/path-length evidence while avoiding planner
    debug terminology.
    """
    risk_speeds = np.asarray(selected_rollout.get("speeds", []), dtype=np.float32)
    risk_waypoints = np.asarray(selected_rollout.get("waypoints", []), dtype=np.float32)

    risk_mean_speed = float(np.mean(risk_speeds)) if len(risk_speeds) > 0 else 0.0
    risk_path_len = estimate_path_length(risk_waypoints)

    expert_stats = infer_expert_speed_stats(expert_future_waypoints, future_fps)

    horizon_s = 0.0
    if future_fps > 1e-6 and len(risk_waypoints) > 0:
        horizon_s = float(len(risk_waypoints)) / float(future_fps)

    if not expert_stats["available"]:
        return (
            f"Over the next {horizon_s:.1f} seconds, the planned trajectory travels about "
            f"{risk_path_len:.2f} m at an average speed of {format_speed_en(risk_mean_speed)}. "
            f"No expert future trajectory is available for comparison."
        )

    expert_mean_speed = expert_stats["mean_speed_mps"]
    expert_path_len = expert_stats["path_length_m"]
    diff = risk_mean_speed - expert_mean_speed

    if diff < -0.5:
        trend = (
            "The planned trajectory is slower than the expert trajectory, which means the planner "
            "is intentionally leaving more space and reducing the closing speed to traffic ahead."
        )
    elif diff > 0.5:
        if bool((lateral_info or {}).get("has_lateral_maneuver", False)):
            trend = (
                "The planned trajectory is more active than the expert trajectory. "
                "Instead of waiting near the current lane, the planner moves around the risky area "
                "with a controlled lateral maneuver."
            )
        else:
            trend = (
                "The planned trajectory is faster than the expert trajectory, which means the planner "
                "is choosing a more assertive motion because the forward path is considered passable."
            )
    else:
        trend = (
            "The planned trajectory has a similar speed to the expert trajectory, so it mostly "
            "keeps the expert driving pace."
        )

    lateral_info = lateral_info or {}
    if bool(lateral_info.get("has_lateral_maneuver", False)):
        lateral_sentence = (
            f" The selected plan also {lateral_info.get('direction_en', 'makes a lateral maneuver')}, "
            f"with a maximum lateral shift of about {safe_float(lateral_info.get('max_abs_lateral_m', 0.0)):.2f} m."
        )
    else:
        lateral_sentence = ""

    return (
        f"Over the next {horizon_s:.1f} seconds, the selected plan travels about {risk_path_len:.2f} m "
        f"at an average speed of {format_speed_en(risk_mean_speed)}. "
        f"The expert trajectory travels about {expert_path_len:.2f} m "
        f"at an average speed of {format_speed_en(expert_mean_speed)}. "
        f"{trend}{lateral_sentence}"
    )

def build_language_annotation(
    frame_name: str,
    scene_context: Dict,
    selected: Dict,
    selected_rollout: Dict,
    expert_future_waypoints: Optional[np.ndarray],
    future_fps: float,
    risk_label_valid: bool,
    fallback_to_expert: bool,
    selected_reference_route: Optional[np.ndarray] = None,
) -> Dict:
    """
    Build deterministic language/QA annotations for each generated frame.

    This function only reads planner outputs and does not affect planning,
    scoring, or candidate selection.
    """
    selected_info = selected["info"]
    behavior_name = str(selected_info.get("behavior_name", "unknown"))
    behavior_zh = behavior_name_to_chinese(behavior_name)
    behavior_en = behavior_name_to_english(behavior_name)

    lateral_info = analyze_selected_lateral_motion(
    selected_rollout=selected_rollout,
    selected_reference_route=selected_reference_route,
    lateral_threshold_m=0.8,
    )

    scene_zh = describe_scene_context_zh(scene_context, lateral_info)
    scene_en = describe_scene_context_en(scene_context, lateral_info)

    decision_zh = describe_selected_decision_zh(selected_info, scene_context, lateral_info)
    decision_en = describe_selected_decision_en(selected_info, scene_context, lateral_info)

    motion_zh = compare_risk_and_expert_motion_zh(
        selected_rollout=selected_rollout,
        expert_future_waypoints=expert_future_waypoints,
        future_fps=future_fps,
        lateral_info=lateral_info,
    )
    motion_en = compare_risk_and_expert_motion_en(
        selected_rollout=selected_rollout,
        expert_future_waypoints=expert_future_waypoints,
        future_fps=future_fps,
        lateral_info=lateral_info,
    )

    if fallback_to_expert:
        validity_zh = (
            "不建议把这一帧作为风险规划轨迹监督。因为当前没有找到可靠的风险规划候选，"
            "系统只是回退到专家未来轨迹用于保持输出稳定。"
        )
        validity_en = (
            "This frame should not be used as a risk-planned trajectory label. "
            "No reliable risk-planned candidate was found, so the system falls back to the expert future trajectory."
        )
    elif risk_label_valid:
        validity_zh = (
            "可以。当前选择的规划轨迹通过了可行性检查，并且代表了系统在该风险场景下的规划选择。"
        )
        validity_en = (
            "Yes. The selected trajectory passed the feasibility check and represents the planner's preferred behavior in this risk-aware scene."
        )
    else:
        validity_zh = (
            "不建议。当前输出没有通过有效风险规划标签的条件，训练时应该忽略这一帧的风险规划轨迹监督。"
        )
        validity_en = (
            "No. The output does not satisfy the conditions for a valid risk-planned label, so this frame should be ignored for risk-planned trajectory supervision."
        )

    summary_zh = (
        f"第 {frame_name} 帧：{scene_zh}"
        f"系统选择“{behavior_zh}”。{motion_zh}"
    )
    
    if bool(lateral_info.get("has_lateral_maneuver", False)):
        direction = str(lateral_info.get("direction", "none"))

        if direction == "left":
            summary_behavior_en = "slow down, pass cautiously, and shift left to avoid the risk"
        elif direction == "right":
            summary_behavior_en = "slow down, pass cautiously, and shift right to avoid the risk"
        else:
            summary_behavior_en = "slow down, pass cautiously, and make a lateral avoidance maneuver"
    else:
        summary_behavior_en = behavior_en

    summary_en = (
        f"Frame {frame_name}: {scene_en} "
        f"The planner selects to {summary_behavior_en}. {motion_en}"
    )

    if bool(lateral_info.get("has_lateral_maneuver", False)):
        direction = str(lateral_info.get("direction", "none"))

        if direction == "left":
            lateral_action_en = "shift left to avoid the risk"
        elif direction == "right":
            lateral_action_en = "shift right to avoid the risk"
        else:
            lateral_action_en = "make a lateral avoidance maneuver"

        selected_behavior_answer_en = (
            f"The selected behavior is to slow down, pass cautiously, and "
            f"{lateral_action_en}."
        )
        selected_behavior_answer_zh = (
            f"当前选择的行为是谨慎通过，并{lateral_info.get('direction_zh', '进行横向避让')}。"
        )
    else:
        selected_behavior_answer_en = f"The selected behavior is to {behavior_en}."
        selected_behavior_answer_zh = f"当前选择的行为是“{behavior_zh}”。"


    qa_pairs_zh = [
        {
            "question": "自车前方发生了什么？",
            "answer": scene_zh,
        },
        {
            "question": "为什么规划器选择这个行为？",
            "answer": decision_zh,
        },
        {
            "question": "当前选择的驾驶行为是什么？",
            "answer": selected_behavior_answer_zh,
        },
        {
            "question": "规划轨迹和专家轨迹有什么不同？",
            "answer": motion_zh,
        },
        {
            "question": "这一帧是否适合作为风险规划轨迹标签？",
            "answer": validity_zh,
        },
    ]

    qa_pairs_en = [
        {
            "question": "What is happening in front of the ego vehicle?",
            "answer": scene_en,
        },
        {
            "question": "Why does the planner choose this behavior?",
            "answer": decision_en,
        },
        {
            "question": "What driving behavior is selected?",
            "answer": selected_behavior_answer_en,
        },
        {
            "question": "How is the planned motion different from the expert motion?",
            "answer": motion_en,
        },
        {
            "question": "Is this a reliable risk-planned trajectory label?",
            "answer": validity_en,
        },
    ]

    return {
        "summary_zh": summary_zh,
        "decision_zh": decision_zh,
        "risk_reason_zh": scene_zh,
        "motion_reason_zh": motion_zh,
        "validity_zh": validity_zh,

        "summary_en": summary_en,
        "decision_en": decision_en,
        "risk_reason_en": scene_en,
        "motion_reason_en": motion_en,
        "validity_en": validity_en,

        "selected_behavior_zh": behavior_zh,
        "selected_behavior_en": behavior_en,
        "selected_lateral_motion": lateral_info,

        "qa_pairs_zh": qa_pairs_zh,
        "qa_pairs_en": qa_pairs_en,
    }
