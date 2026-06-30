# -*- coding: utf-8 -*-

from .common import *
from .io_utils import points_to_list, array_to_list
from .qa_annotation import safe_float, behavior_name_to_chinese, behavior_name_to_english


def _reason_text_zh(reasons: List[str]) -> str:
    """
    Convert internal planner rejection reasons into human-readable driving language.
    Do not expose technical variable names in VLA language.
    """
    if not reasons:
        return "这条轨迹不是当前最合适的选择"

    readable = []
    for r in reasons:
        r = str(r)

        if "not_selectable_by_default" in r:
            readable.append("这种动作只适合在更紧急或更明确需要让行的情况下使用")
        elif "inactive" in r:
            readable.append("当前场景并不需要执行这种动作")
        elif "hard_ratio" in r or "risk" in r or "cost" in r:
            readable.append("这条路线会让车辆过于接近前方或侧方的交通参与者")
        elif "out_of_bounds" in r:
            readable.append("这条路线会偏离可行驶区域")
        elif "route_deviation" in r:
            readable.append("这条路线偏离导航方向过多")
        elif "collision" in r or "crash" in r:
            readable.append("这条路线可能与其他交通参与者发生冲突")
        elif "no_allowed" in r:
            readable.append("当前没有找到可靠的可执行候选")
        else:
            readable.append("这条轨迹不如当前选择更适合这个场景")

    # 去重，保持顺序
    out = []
    for x in readable:
        if x not in out:
            out.append(x)

    return "；".join(out)


def _reason_text_en(reasons: List[str]) -> str:
    """
    Convert internal planner rejection reasons into human-readable driving language.
    Do not expose technical variable names in VLA language.
    """
    if not reasons:
        return "this trajectory is not the most suitable choice for the current scene"

    readable = []
    for r in reasons:
        r = str(r)

        if "not_selectable_by_default" in r:
            readable.append("this maneuver is only appropriate in a more urgent or clearly yielding situation")
        elif "inactive" in r:
            readable.append("the current scene does not require this maneuver")
        elif "hard_ratio" in r or "risk" in r or "cost" in r:
            readable.append("this path would bring the ego vehicle too close to traffic participants ahead or nearby")
        elif "out_of_bounds" in r:
            readable.append("this path would leave the drivable area")
        elif "route_deviation" in r:
            readable.append("this path deviates too much from the navigation direction")
        elif "collision" in r or "crash" in r:
            readable.append("this path may conflict with another traffic participant")
        elif "no_allowed" in r:
            readable.append("no reliable executable candidate is available")
        else:
            readable.append("this trajectory is less suitable than the selected one in this scene")

    out = []
    for x in readable:
        if x not in out:
            out.append(x)

    return "; ".join(out)


def _normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def _dedup_sentences_zh(parts: List[str]) -> str:
    """
    Join Chinese sentence fragments while removing exact duplicates.
    """
    out = []
    seen = set()

    for p in parts:
        p = _normalize_space(p)
        if not p:
            continue

        # Ensure Chinese sentence ending.
        if not p.endswith(("。", "！", "？")):
            p += "。"

        key = p.replace(" ", "")
        if key not in seen:
            out.append(p)
            seen.add(key)

    text = "".join(out)

    # Light post-processing for common causal-connector repetition.
    text = text.replace("。因此，因此，", "。因此，")
    text = text.replace("。因此，当前", "。当前")
    text = text.replace("。。", "。")

    return text


def _dedup_sentences_en(parts: List[str]) -> str:
    """
    Join English sentence fragments while removing exact duplicates.
    """
    out = []
    seen = set()

    for p in parts:
        p = _normalize_space(p)
        if not p:
            continue

        if not p.endswith((".", "!", "?")):
            p += "."

        key = p.lower()
        if key not in seen:
            out.append(p)
            seen.add(key)

    text = " ".join(out)

    # Light post-processing for common causal-connector repetition.
    text = text.replace("Therefore, Therefore,", "Therefore,")
    text = text.replace(". Therefore, therefore,", ". Therefore,")
    text = text.replace("..", ".")

    return text


def _candidate_role(info: Dict, is_selected: bool) -> str:
    allowed = bool(info.get("allowed", False))
    fallback = bool(info.get("fallback_to_expert", False))

    if is_selected and fallback:
        return "selected_expert_fallback"
    if is_selected and allowed:
        return "selected_executable"
    if allowed:
        return "valid_alternative"
    return "invalid_counterfactual"


def _role_zh(role: str) -> str:
    mapping = {
        "selected_executable": "当前推荐执行的驾驶轨迹",
        "selected_expert_fallback": "当前场景下的保守通行轨迹",
        "valid_alternative": "可执行但未被最终推荐的备选轨迹",
        "invalid_counterfactual": "不建议执行的反事实候选轨迹",
    }
    return mapping.get(role, role)


def _role_en(role: str) -> str:
    mapping = {
        "selected_executable": "recommended driving trajectory for the current scene",
        "selected_expert_fallback": "conservative driving trajectory for the current scene",
        "valid_alternative": "valid but non-recommended alternative trajectory",
        "invalid_counterfactual": "invalid counterfactual trajectory that should not be executed",
    }
    return mapping.get(role, role)


def _behavior_display_name_zh(behavior_name: str, role: str = "") -> str:
    """
    Display name used in VLA language.
    Do not expose internal names such as expert_fallback.
    """
    if behavior_name == "expert_fallback":
        return "沿当前路线保守通行"

    return behavior_name_to_chinese(behavior_name)


def _behavior_display_name_en(behavior_name: str, role: str = "") -> str:
    """
    Display name used in VLA language.
    Do not expose internal names such as expert_fallback.
    """
    if behavior_name == "expert_fallback":
        return "proceed conservatively along the current route"

    return behavior_name_to_english(behavior_name)


def _action_phrase_zh(behavior_name: str) -> str:
    """
    Short action phrase for final VLA answer.
    Avoid technical labels and avoid mentioning planner/fallback/expert.
    """
    table = {
        "route_follow": "沿当前导航路线平稳前进",
        "cautious_follow": "降低速度并沿当前路线谨慎前进",
        "yield_stop": "提前减速，并在安全位置停车等待",
        "left_nudge": "向左轻微偏移，通过受影响区域后回到导航路线",
        "right_nudge": "向右轻微偏移，通过受影响区域后回到导航路线",
        "creep": "以低速缓慢向前确认通行空间",
        "emergency_brake": "立即明显减速，优先保持安全距离",
        "expert_fallback": "沿当前路线保守通行",
    }
    return table.get(str(behavior_name), f"执行 {behavior_name} 对应的驾驶动作")


def _action_phrase_en(behavior_name: str) -> str:
    """
    Short action phrase for final VLA answer.
    Avoid technical labels and avoid mentioning planner/fallback/expert.
    """
    table = {
        "route_follow": "keep following the current navigation route steadily",
        "cautious_follow": "slow down and proceed cautiously along the current route",
        "yield_stop": "slow down early and stop at a safe position",
        "left_nudge": "make a slight leftward nudge and return to the navigation route after passing the affected area",
        "right_nudge": "make a slight rightward nudge and return to the navigation route after passing the affected area",
        "creep": "creep forward slowly to check whether the space ahead is passable",
        "emergency_brake": "brake immediately and prioritize maintaining a safe distance",
        "expert_fallback": "proceed conservatively along the current route",
    }
    return table.get(str(behavior_name), f"execute the {behavior_name} driving action")


def _preferred_action_phrase_zh(selected_behavior_name: str) -> str:
    """
    Short preferred-action phrase used when explaining why a counterfactual
    action should not be executed.
    """
    table = {
        "route_follow": "沿当前导航路线平稳前进",
        "cautious_follow": "降低速度并沿当前路线谨慎前进",
        "yield_stop": "在安全位置减速停车并等待",
        "left_nudge": "向左侧轻微避让",
        "right_nudge": "向右侧轻微避让",
        "creep": "低速缓慢前进并确认通行空间",
        "emergency_brake": "立即明显减速并保持安全距离",
        "expert_fallback": "沿当前路线保守通行",
    }
    return table.get(str(selected_behavior_name), "执行当前推荐动作")


def _preferred_action_phrase_en(selected_behavior_name: str) -> str:
    """
    Short preferred-action phrase used when explaining why a counterfactual
    action should not be executed.
    """
    table = {
        "route_follow": "keep following the current navigation route steadily",
        "cautious_follow": "slow down and proceed cautiously along the current route",
        "yield_stop": "slow down and stop at a safe position",
        "left_nudge": "make a slight left-side avoidance maneuver",
        "right_nudge": "make a slight right-side avoidance maneuver",
        "creep": "creep forward slowly to check the space ahead",
        "emergency_brake": "brake immediately and maintain a safe distance",
        "expert_fallback": "proceed conservatively along the current route",
    }
    return table.get(str(selected_behavior_name), "take the recommended action")


def _side_space_status(
    side: str,
    scene_context: Dict,
    key_dynamic_context: Optional[Dict],) -> Dict:
    """
    Describe left/right side space using:
      1. current side actor,
      2. front-left/front-right actor,
      3. corridor availability along the candidate path.

    This avoids mixing 'current side' and 'future candidate corridor' into one vague phrase.
    """
    ctx = key_dynamic_context or {}

    if side == "left":
        free = bool(scene_context.get("left_corridor_free", False))
        side_actor = ctx.get("left_side_actor", {}) or ctx.get("left_corridor_actor", {}) or {}
        front_side_actor = ctx.get("front_left_actor", {}) or {}
        stats = scene_context.get("left_corridor_stats", {}) or {}
    else:
        free = bool(scene_context.get("right_corridor_free", False))
        side_actor = ctx.get("right_side_actor", {}) or ctx.get("right_corridor_actor", {}) or {}
        front_side_actor = ctx.get("front_right_actor", {}) or {}
        stats = scene_context.get("right_corridor_stats", {}) or {}

    side_actor_exists = _actor_relevant_for_lateral_maneuver(
        side_actor,
        side=side,
        is_front_side=False,
    )

    front_side_actor_exists = _actor_relevant_for_lateral_maneuver(
        front_side_actor,
        side=side,
        is_front_side=True,
    )

    if side_actor_exists:
        status = "limited_by_current_side_actor"
    elif front_side_actor_exists:
        status = "affected_by_front_side_actor"
    elif free:
        status = "available"
    else:
        status = "limited_by_static_or_boundary"

    return {
        "side": side,
        "free": bool(free),
        "status": status,
        "has_current_side_actor": bool(side_actor_exists),
        "has_front_side_actor": bool(front_side_actor_exists),
        "side_actor": side_actor,
        "front_side_actor": front_side_actor,
        "actor": side_actor if side_actor_exists else front_side_actor,
        "stats": stats,
    }


def _get_candidate_static_context(
    static_candidate_context: Optional[Dict],
    source_candidate_index: int,) -> Dict:
    ctx = static_candidate_context or {}
    candidates = ctx.get("candidates", {}) or {}
    return candidates.get(str(source_candidate_index), {}) or {}


def _static_type(static_ctx: Optional[Dict]) -> str:
    return str((static_ctx or {}).get("dominant_static_type", "unknown"))


def _static_desc_zh(static_ctx: Optional[Dict]) -> str:
    return str((static_ctx or {}).get("description_zh", ""))


def _static_desc_en(static_ctx: Optional[Dict]) -> str:
    return str((static_ctx or {}).get("description_en", ""))


def _static_is_limited(static_ctx: Optional[Dict]) -> bool:
    static_type = _static_type(static_ctx)
    return static_type in [
        "sidewalk",
        "non_drivable_or_boundary",
        "solid_lane_marking",
        "mixed_or_uncertain",
    ]


def _actor_exists(actor: Optional[Dict]) -> bool:
    return bool((actor or {}).get("exists", False))


def _actor_relative_position(actor: Optional[Dict]) -> str:
    actor = actor or {}
    return str(actor.get("relative_position", "")).lower()


def _actor_x_m(actor: Optional[Dict]) -> float:
    actor = actor or {}
    return safe_float(actor.get("x_m", 0.0), default=0.0)


def _actor_y_m(actor: Optional[Dict]) -> float:
    actor = actor or {}
    return safe_float(actor.get("y_m", 0.0), default=0.0)


def _actor_is_rear(actor: Optional[Dict]) -> bool:
    rel = _actor_relative_position(actor)
    return rel in ["rear", "rear_left", "rear_right"]


def _actor_relevant_for_lateral_maneuver(
    actor: Optional[Dict],
    side: str,
    is_front_side: bool = False,) -> bool:
    """
    Decide whether an actor should be used to explain a left/right avoidance maneuver.

    Key rule:
    - rear/rear-left/rear-right actors should not explain a forward lateral nudge.
    - front-left/front-right actors are relevant to the corresponding nudge direction.
    - current side actors are relevant only if they are not behind ego.
    """
    if not _actor_exists(actor):
        return False

    rel = _actor_relative_position(actor)
    x = _actor_x_m(actor)

    # A rear actor should not be used as the reason for rejecting a forward nudge.
    if _actor_is_rear(actor):
        return False

    if is_front_side:
        if side == "left":
            return rel == "front_left" and x > 0.0
        if side == "right":
            return rel == "front_right" and x > 0.0
        return False

    # Current side actor: keep only actors that are at ego side or ahead.
    # Since relative_position currently does not have pure "left/right",
    # this mainly filters out rear-side objects.
    return x >= -0.5


def _actor_motion_state(actor: Optional[Dict]) -> str:
    actor = actor or {}
    state = str(actor.get("motion_state", "")).lower()
    speed = safe_float(actor.get("speed_mps", 0.0), default=0.0)

    if state == "stopped" or speed < 0.3:
        return "stopped"
    if state == "slow" or speed < 2.0:
        return "slow"
    return "moving"


def _actor_is_stopped(actor: Optional[Dict]) -> bool:
    return _actor_motion_state(actor) == "stopped"


def _side_actor_effect_zh(actor: Optional[Dict], side_zh: str, is_front_side: bool = False) -> str:
    """
    Describe how an actor affects a lateral maneuver.
    Avoid calling stopped vehicles 'dynamic traffic participants'.
    """
    if not _actor_exists(actor):
        return ""

    desc = _actor_short_desc_zh(actor)
    motion = _actor_motion_state(actor)

    if motion == "stopped":
        if is_front_side:
            return f"{desc}，会占用或压缩{side_zh}前方绕行空间。"
        return f"{desc}，需要在判断{side_zh}侧向空间时留意。"

    if motion == "slow":
        if is_front_side:
            return f"{desc}，向{side_zh}绕行需要注意其低速运动状态。"
        return f"{desc}，{side_zh}绕行空间需要谨慎判断。"

    if is_front_side:
        return f"{desc}，向{side_zh}绕行需要额外谨慎。"

    return f"{desc}，{side_zh}绕行空间不够稳定。"


def _side_actor_effect_en(actor: Optional[Dict], side_en: str, is_front_side: bool = False) -> str:
    """
    Describe how an actor affects a lateral maneuver.
    Avoid calling stopped vehicles 'dynamic traffic participants'.
    """
    if not _actor_exists(actor):
        return ""

    desc = _actor_short_desc_en(actor)
    motion = _actor_motion_state(actor)

    if motion == "stopped":
        if is_front_side:
            return f"{desc}, which occupies or compresses the front-{side_en} detour space."
        return f"{desc}, which should be considered when checking the {side_en}-side space."

    if motion == "slow":
        if is_front_side:
            return f"{desc}, so a {side_en}-side detour should consider its slow motion."
        return f"{desc}, so the {side_en}-side detour space should be treated cautiously."

    if is_front_side:
        return f"{desc}, so a {side_en}-side detour should be considered carefully."

    return f"{desc}, so the {side_en}-side detour space is not stable."


def _actor_short_desc_zh(actor: Optional[Dict], role_hint: str = "") -> str:
    actor = actor or {}
    if not _actor_exists(actor):
        return ""

    cls = str(actor.get("class_zh", "交通参与者"))
    pos = str(actor.get("relative_position_zh", "周围"))
    motion = str(actor.get("motion_state_zh", "正在移动"))

    if cls == "行人":
        return f"{pos}有行人"

    return f"{pos}有一辆{motion}的{cls}"


def _actor_short_desc_en(actor: Optional[Dict], role_hint: str = "") -> str:
    actor = actor or {}
    if not _actor_exists(actor):
        return ""

    cls = str(actor.get("class_en", "traffic participant"))
    pos = str(actor.get("relative_position_en", "nearby"))
    motion = str(actor.get("motion_state_en", "moving"))

    if cls == "pedestrian":
        return f"a pedestrian is {pos}"

    if motion == "nearly stopped":
        return f"a nearly stopped {cls} is {pos}"

    if motion == "moving slowly":
        return f"a slow-moving {cls} is {pos}"

    if motion == "moving":
        return f"a moving {cls} is {pos}"

    return f"a {cls} is {pos}"


def _get_front_center_actor(ctx: Dict) -> Dict:
    front_center = ctx.get("front_center_actor", {}) or {}
    if _actor_exists(front_center):
        return front_center

    front = ctx.get("front_actor", {}) or {}
    if _actor_exists(front) and str(front.get("relative_position", "")) == "front":
        return front

    return {"exists": False}


def _build_scene_brief_zh(scene_context: Dict, key_dynamic_context: Optional[Dict]) -> str:
    """
    Scene-level description.
    Only describe facts that are useful for driving decisions.
    Do not mention candidate behaviors here.
    """
    ctx = key_dynamic_context or {}
    route_safe = bool(scene_context.get("route_safe", False))
    front_blocked = bool(scene_context.get("front_blocked", False) or scene_context.get("conflict_ahead", False))

    front = _get_front_center_actor(ctx)
    front_left = ctx.get("front_left_actor", {}) or {}
    front_right = ctx.get("front_right_actor", {}) or {}
    walker = ctx.get("nearby_walker", {}) or {}

    parts = []

    if route_safe and not front_blocked:
        parts.append("自车当前路线前方整体仍可通行")
    elif front_blocked:
        parts.append("自车前方通行空间受到影响")
    else:
        parts.append("自车前方空间需要谨慎判断")

    if _actor_exists(front):
        desc = _actor_short_desc_zh(front)
        if desc:
            parts.append(f"{desc}，需要控制接近速度")

    if _actor_exists(front_left):
        desc = _actor_short_desc_zh(front_left)
        if desc:
            if route_safe and not front_blocked:
                parts.append(f"{desc}，但它没有直接阻塞自车当前路线")
            else:
                parts.append(f"{desc}，向左侧绕行需要额外谨慎")

    if _actor_exists(front_right):
        desc = _actor_short_desc_zh(front_right)
        if desc:
            if route_safe and not front_blocked:
                parts.append(f"{desc}，但它没有直接阻塞自车当前路线")
            else:
                parts.append(f"{desc}，向右侧绕行需要额外谨慎")

    if _actor_exists(walker):
        desc = _actor_short_desc_zh(walker)
        if desc:
            parts.append(f"{desc}，车辆需要保持谨慎")

    if not parts:
        return "当前没有明显影响自车决策的交通参与者。"

    return "，".join(parts) + "。"


def _build_scene_brief_en(scene_context: Dict, key_dynamic_context: Optional[Dict]) -> str:
    """
    Scene-level description.
    Only describe facts that are useful for driving decisions.
    Do not mention candidate behaviors here.
    """
    ctx = key_dynamic_context or {}
    route_safe = bool(scene_context.get("route_safe", False))
    front_blocked = bool(scene_context.get("front_blocked", False) or scene_context.get("conflict_ahead", False))

    front = _get_front_center_actor(ctx)
    front_left = ctx.get("front_left_actor", {}) or {}
    front_right = ctx.get("front_right_actor", {}) or {}
    walker = ctx.get("nearby_walker", {}) or {}

    parts = []

    if route_safe and not front_blocked:
        parts.append("the ego vehicle's current route remains passable")
    elif front_blocked:
        parts.append("the space ahead of the ego vehicle is affected")
    else:
        parts.append("the space ahead should be handled cautiously")

    if _actor_exists(front):
        desc = _actor_short_desc_en(front)
        if desc:
            parts.append(f"{desc}, so the ego vehicle should control its closing speed")

    if _actor_exists(front_left):
        desc = _actor_short_desc_en(front_left)
        if desc:
            if route_safe and not front_blocked:
                parts.append(f"{desc}, but it does not directly block the ego vehicle's current route")
            else:
                parts.append(f"{desc}, so a left-side detour should be considered carefully")

    if _actor_exists(front_right):
        desc = _actor_short_desc_en(front_right)
        if desc:
            if route_safe and not front_blocked:
                parts.append(f"{desc}, but it does not directly block the ego vehicle's current route")
            else:
                parts.append(f"{desc}, so a right-side detour should be considered carefully")

    if _actor_exists(walker):
        desc = _actor_short_desc_en(walker)
        if desc:
            parts.append(f"{desc}, so the ego vehicle should remain cautious")

    if not parts:
        return "No traffic participant is clearly affecting the ego vehicle's decision."

    return "In the current scene, " + "; ".join(parts) + "."


def _build_candidate_scene_brief_zh(
    behavior_name: str,
    role: str,
    scene_context: Dict,
    key_dynamic_context: Optional[Dict],
    candidate_static_context: Optional[Dict],) -> str:
    """
    Candidate-focused scene brief.
    For lateral candidates, focus on the corresponding side instead of always
    repeating global front-left/front-right objects.
    """
    ctx = key_dynamic_context or {}

    if behavior_name not in ["left_nudge", "right_nudge"]:
        return _build_scene_brief_zh(scene_context, key_dynamic_context)

    side = "left" if behavior_name == "left_nudge" else "right"
    side_zh = "左侧" if side == "left" else "右侧"

    static_desc = _static_desc_zh(candidate_static_context)

    if side == "left":
        front_side_actor = ctx.get("front_left_actor", {}) or {}
        side_actor = ctx.get("left_side_actor", {}) or ctx.get("left_corridor_actor", {}) or {}
    else:
        front_side_actor = ctx.get("front_right_actor", {}) or {}
        side_actor = ctx.get("right_side_actor", {}) or ctx.get("right_corridor_actor", {}) or {}

    parts = []

    if _actor_relevant_for_lateral_maneuver(
        side_actor,
        side=side,
        is_front_side=False,
    ):
        parts.append(
            _side_actor_effect_zh(
                side_actor,
                side_zh=side_zh,
                is_front_side=False,
            ).rstrip("。")
        )

    if _actor_relevant_for_lateral_maneuver(
        front_side_actor,
        side=side,
        is_front_side=True,
    ):
        parts.append(
            _side_actor_effect_zh(
                front_side_actor,
                side_zh=side_zh,
                is_front_side=True,
            ).rstrip("。")
        )

    if static_desc:
        parts.append(static_desc.rstrip("。"))

    if not parts:
        parts.append(f"{side_zh}候选空间没有明确相关车辆或行人，但仍需要结合道路边界和可行驶区域判断")

    return "。".join(parts) + "。"


def _build_candidate_scene_brief_en(
    behavior_name: str,
    role: str,
    scene_context: Dict,
    key_dynamic_context: Optional[Dict],
    candidate_static_context: Optional[Dict],) -> str:
    """
    Candidate-focused scene brief.
    For lateral candidates, focus on the corresponding side instead of always
    repeating global front-left/front-right objects.
    """
    ctx = key_dynamic_context or {}

    if behavior_name not in ["left_nudge", "right_nudge"]:
        return _build_scene_brief_en(scene_context, key_dynamic_context)

    side = "left" if behavior_name == "left_nudge" else "right"
    side_en = "left" if side == "left" else "right"

    static_desc = _static_desc_en(candidate_static_context)

    if side == "left":
        front_side_actor = ctx.get("front_left_actor", {}) or {}
        side_actor = ctx.get("left_side_actor", {}) or ctx.get("left_corridor_actor", {}) or {}
    else:
        front_side_actor = ctx.get("front_right_actor", {}) or {}
        side_actor = ctx.get("right_side_actor", {}) or ctx.get("right_corridor_actor", {}) or {}

    parts = []

    if _actor_relevant_for_lateral_maneuver(
        side_actor,
        side=side,
        is_front_side=False,
    ):
        parts.append(
            _side_actor_effect_en(
                side_actor,
                side_en=side_en,
                is_front_side=False,
            ).rstrip(".")
        )

    if _actor_relevant_for_lateral_maneuver(
        front_side_actor,
        side=side,
        is_front_side=True,
    ):
        parts.append(
            _side_actor_effect_en(
                front_side_actor,
                side_en=side_en,
                is_front_side=True,
            ).rstrip(".")
        )

    if static_desc:
        parts.append(static_desc.rstrip("."))

    if not parts:
        parts.append(
            f"the {side_en}-side candidate space has no clearly relevant vehicle or pedestrian, "
            f"but it still needs to be checked against road boundaries and drivable areas"
        )

    body = "; ".join(parts).strip()
    if body:
        body = body[0].lower() + body[1:]

    return "In the current scene, " + body + "."


def _build_action_rationale_zh(
    behavior_name: str,
    role: str,
    info: Dict,
    scene_context: Dict,
    key_dynamic_context: Optional[Dict],
    candidate_static_context: Optional[Dict],
    selected_behavior_name: str,) -> str:

    route_safe = bool(scene_context.get("route_safe", False))
    front_blocked = bool(scene_context.get("front_blocked", False) or scene_context.get("conflict_ahead", False))
    ctx = key_dynamic_context or {}

    front_left = ctx.get("front_left_actor", {}) or {}
    front_right = ctx.get("front_right_actor", {}) or {}

    if role in ["selected_executable", "selected_expert_fallback"]:
        if behavior_name == "route_follow":
            return "由于当前路线没有被明显阻塞，保持导航路线比额外横向绕行更直接、更稳定。"

        if behavior_name == "cautious_follow":
            return "由于前方空间存在不确定性，降低速度可以减少对前方对象的接近速度。"

        if behavior_name == "yield_stop":
            return "由于前方空间暂时不适合继续压缩，停车等待可以为后续通行保留更安全的距离。"

        if behavior_name == "left_nudge":
            return "左侧候选走廊提供了临时通过空间，因此可以轻微向左避让。"

        if behavior_name == "right_nudge":
            return "右侧候选走廊提供了临时通过空间，因此可以轻微向右避让。"

        if behavior_name == "creep":
            return "低速前进可以在不激进接近前方区域的情况下确认通行空间。"

        if behavior_name == "emergency_brake":
            return "当前需要优先降低速度，以避免与前方对象距离过近。"

        if behavior_name == "expert_fallback":
            return "当前不适合进行激进横向调整，因此更适合沿当前路线保守通行。"

        return "该动作与当前交通空间和导航意图一致。"

    if role == "valid_alternative":
        if behavior_name == "left_nudge":
            return "向左轻微避让可以作为备选动作，但当前推荐动作更符合主要导航意图。"

        if behavior_name == "right_nudge":
            return "向右轻微避让可以作为备选动作，但当前推荐动作更符合主要导航意图。"

        if behavior_name == "cautious_follow":
            return "减速谨慎前进是一种更保守的备选动作，但当前不一定需要明显降低速度。"

        if behavior_name == "yield_stop":
            return "停车等待是一种更保守的备选动作，但当前还可以选择更连续的通行方式。"

        return "该动作可行，但不是当前最直接的驾驶选择。"

    # invalid_counterfactual
    if behavior_name == "left_nudge":
        if _static_is_limited(candidate_static_context):
            return "该道路属性决定了左侧不适合作为绕行选择。"

        if _actor_relevant_for_lateral_maneuver(
            front_left,
            side="left",
            is_front_side=True,
        ):
            desc = _actor_short_desc_zh(front_left)
            if _actor_is_stopped(front_left):
                return f"{desc}，会占用或压缩左侧前方绕行空间；同时当前路线仍可通行，因此左侧绕行不是更合适的选择。"
            return f"{desc}，主动向左偏移需要额外谨慎；同时当前路线仍可通行，因此左侧绕行不是更合适的选择。"

        if route_safe:
            return "当前路线仍可通行，主动向左绕行会引入不必要的横向动作。"

        return "左侧绕行与当前交通空间不够匹配。"

    if behavior_name == "right_nudge":
        if _static_is_limited(candidate_static_context):
            return "该道路属性决定了右侧不适合作为绕行选择。"

        if _actor_relevant_for_lateral_maneuver(
            front_right,
            side="right",
            is_front_side=True,
        ):
            desc = _actor_short_desc_zh(front_right)
            if _actor_is_stopped(front_right):
                return f"{desc}，会占用或压缩右侧前方绕行空间；因此右侧绕行不是更合适的选择。"
            return f"{desc}，主动向右偏移需要额外谨慎；因此右侧绕行不是更合适的选择。"

        if route_safe:
            return "当前路线仍可通行，主动向右绕行会引入不必要的横向动作。"

        return "右侧绕行与当前交通空间不够匹配。"

    if behavior_name == "cautious_follow":
        if route_safe and not front_blocked:
            return "当前路线没有明显阻塞，明显减速跟车会显得过于保守。"
        return "该减速策略不是当前最合适的执行动作。"

    if behavior_name == "yield_stop":
        return "当前场景还没有达到必须停车等待的程度。"

    if behavior_name == "emergency_brake":
        return "当前没有直接迫近的紧急冲突，因此不需要紧急制动。"

    if behavior_name == "creep":
        return "当前路线仍可通行，不需要以低速试探方式前进。"

    return f"该动作不适合作为当前主动作，原因是：{_reason_text_zh(info.get('reasons', []))}。"


def _build_action_rationale_en(
    behavior_name: str,
    role: str,
    info: Dict,
    scene_context: Dict,
    key_dynamic_context: Optional[Dict],
    candidate_static_context: Optional[Dict],
    selected_behavior_name: str,) -> str:

    route_safe = bool(scene_context.get("route_safe", False))
    front_blocked = bool(scene_context.get("front_blocked", False) or scene_context.get("conflict_ahead", False))
    ctx = key_dynamic_context or {}

    front_left = ctx.get("front_left_actor", {}) or {}
    front_right = ctx.get("front_right_actor", {}) or {}

    if role in ["selected_executable", "selected_expert_fallback"]:
        if behavior_name == "route_follow":
            return "Because the current route is not clearly blocked, staying on the navigation route is more direct and stable than making an extra lateral maneuver."

        if behavior_name == "cautious_follow":
            return "Because the space ahead is uncertain, slowing down helps reduce the closing speed toward traffic ahead."

        if behavior_name == "yield_stop":
            return "Because the space ahead should not be compressed further, stopping and waiting preserves a safer gap for later movement."

        if behavior_name == "left_nudge":
            return "The left-side candidate corridor provides temporary passing space, so a slight leftward avoidance maneuver is appropriate."

        if behavior_name == "right_nudge":
            return "The right-side candidate corridor provides temporary passing space, so a slight rightward avoidance maneuver is appropriate."

        if behavior_name == "creep":
            return "Creeping forward allows the ego vehicle to check the space ahead without approaching the area aggressively."

        if behavior_name == "emergency_brake":
            return "The ego vehicle should prioritize reducing speed to avoid getting too close to the object ahead."

        if behavior_name == "expert_fallback":
            return "An aggressive lateral adjustment is not suitable, so proceeding conservatively along the current route is more appropriate."

        return "This action is consistent with the current traffic space and navigation intent."

    if role == "valid_alternative":
        if behavior_name == "left_nudge":
            return "A slight leftward nudge can be used as an alternative, but the recommended action better matches the main navigation intent."

        if behavior_name == "right_nudge":
            return "A slight rightward nudge can be used as an alternative, but the recommended action better matches the main navigation intent."

        if behavior_name == "cautious_follow":
            return "Slowing down and following cautiously is a more conservative alternative, but a clear slow-down is not necessarily required."

        if behavior_name == "yield_stop":
            return "Stopping and waiting is a more conservative alternative, but the scene still allows a more continuous driving behavior."

        return "This action is feasible, but it is not the most direct driving choice in this scene."

    # invalid_counterfactual
    if behavior_name == "left_nudge":
        if _static_is_limited(candidate_static_context):
            return "This road context makes the left side unsuitable as a detour choice."

        if _actor_relevant_for_lateral_maneuver(
            front_left,
            side="left",
            is_front_side=True,
        ):
            desc = _actor_short_desc_en(front_left)
            if _actor_is_stopped(front_left):
                return f"{desc}, which occupies or compresses the front-left detour space; since the current route remains passable, a leftward detour is not the more suitable choice."
            return f"{desc}, so an active leftward shift requires extra caution; since the current route remains passable, a leftward detour is not the more suitable choice."

        if route_safe:
            return "Since the current route remains passable, an active leftward detour would introduce an unnecessary lateral maneuver."

        return "A leftward detour does not match the current traffic space well."

    if behavior_name == "right_nudge":
        if _static_is_limited(candidate_static_context):
            return "This road context makes the right side unsuitable as a detour choice."

        if _actor_relevant_for_lateral_maneuver(
            front_right,
            side="right",
            is_front_side=True,
        ):
            desc = _actor_short_desc_en(front_right)
            if _actor_is_stopped(front_right):
                return f"{desc}, which occupies or compresses the front-right detour space; therefore, a rightward detour is not the more suitable choice."
            return f"{desc}, so an active rightward shift requires extra caution; therefore, a rightward detour is not the more suitable choice."

        if route_safe:
            return "Since the current route remains passable, an active rightward detour would introduce an unnecessary lateral maneuver."

        return "A rightward detour does not match the current traffic space well."

    if behavior_name == "cautious_follow":
        if route_safe and not front_blocked:
            return "The current route is not clearly blocked, so a strong cautious slow-down would be overly conservative."
        return "This slow-down strategy is not the most suitable executable action."

    if behavior_name == "yield_stop":
        return "The scene does not yet require the ego vehicle to stop and wait."

    if behavior_name == "emergency_brake":
        return "There is no immediate critical conflict, so emergency braking is not required."

    if behavior_name == "creep":
        return "The current route remains passable, so creeping forward is not necessary."

    return f"This action is not suitable as the main action because {_reason_text_en(info.get('reasons', []))}."


def _build_final_response_zh(
    behavior_name: str,
    role: str,
    selected_behavior_name: str = "unknown",) -> str:

    action = _action_phrase_zh(behavior_name)
    preferred = _preferred_action_phrase_zh(selected_behavior_name)

    if role == "invalid_counterfactual":
        if behavior_name == "left_nudge":
            return f"因此，车辆不应向左侧避让，而应{preferred}。"
        if behavior_name == "right_nudge":
            return f"因此，车辆不应向右侧避让，而应{preferred}。"
        if behavior_name == "yield_stop":
            return f"因此，当前不需要停车等待，车辆应{preferred}。"
        if behavior_name == "emergency_brake":
            return f"因此，当前不需要紧急制动，车辆应{preferred}。"
        if behavior_name == "cautious_follow":
            return f"因此，当前不需要明显减速跟车，车辆应{preferred}。"
        if behavior_name == "creep":
            return f"因此，当前不需要低速试探通行，车辆应{preferred}。"
        return f"因此，不建议车辆{action}，而应{preferred}。"

    if role == "valid_alternative":
        if behavior_name == "left_nudge":
            return "如果需要备选方案，车辆可以向左侧轻微避让。"
        if behavior_name == "right_nudge":
            return "如果需要备选方案，车辆可以向右侧轻微避让。"
        return f"如果需要备选方案，车辆可以{action}。"

    return f"因此，车辆应{action}。"


def _build_final_response_en(
    behavior_name: str,
    role: str,
    selected_behavior_name: str = "unknown",) -> str:

    action = _action_phrase_en(behavior_name)
    preferred = _preferred_action_phrase_en(selected_behavior_name)

    if role == "invalid_counterfactual":
        if behavior_name == "left_nudge":
            return f"Therefore, the ego vehicle should not make a left-side avoidance maneuver; it should {preferred}."
        if behavior_name == "right_nudge":
            return f"Therefore, the ego vehicle should not make a right-side avoidance maneuver; it should {preferred}."
        if behavior_name == "yield_stop":
            return f"Therefore, stopping and waiting is not required here; the ego vehicle should {preferred}."
        if behavior_name == "emergency_brake":
            return f"Therefore, emergency braking is not required here; the ego vehicle should {preferred}."
        if behavior_name == "cautious_follow":
            return f"Therefore, a strong cautious slow-down is not required here; the ego vehicle should {preferred}."
        if behavior_name == "creep":
            return f"Therefore, creeping forward is not required here; the ego vehicle should {preferred}."
        return f"Therefore, the ego vehicle should not {action}; it should {preferred}."

    if role == "valid_alternative":
        if behavior_name == "left_nudge":
            return "If an alternative is needed, the ego vehicle can make a slight left-side avoidance maneuver."
        if behavior_name == "right_nudge":
            return "If an alternative is needed, the ego vehicle can make a slight right-side avoidance maneuver."
        return f"If an alternative is needed, the ego vehicle can {action}."

    return f"Therefore, the ego vehicle should {action}."


def _build_scene_wise_answer_zh(
    behavior_name: str,
    role: str,
    info: Dict,
    scene_context: Dict,
    key_dynamic_context: Optional[Dict],
    candidate_static_context: Optional[Dict],
    selected_behavior_name: str,) -> Dict:

    scene_brief = _build_candidate_scene_brief_zh(
        behavior_name=behavior_name,
        role=role,
        scene_context=scene_context,
        key_dynamic_context=key_dynamic_context,
        candidate_static_context=candidate_static_context,
    )

    rationale = _build_action_rationale_zh(
        behavior_name=behavior_name,
        role=role,
        info=info,
        scene_context=scene_context,
        key_dynamic_context=key_dynamic_context,
        candidate_static_context=candidate_static_context,
        selected_behavior_name=selected_behavior_name,
    )
    
    final_response = _build_final_response_zh(
    behavior_name=behavior_name,
    role=role,
    selected_behavior_name=selected_behavior_name,
    )

    answer = _dedup_sentences_zh([
        scene_brief,
        rationale,
        final_response,
    ])

    return {
        "scene_brief": scene_brief,
        "action_rationale": rationale,
        "final_response": final_response,
        "answer": answer,
    }


def _build_scene_wise_answer_en(
    behavior_name: str,
    role: str,
    info: Dict,
    scene_context: Dict,
    key_dynamic_context: Optional[Dict],
    candidate_static_context: Optional[Dict],
    selected_behavior_name: str,) -> Dict:

    scene_brief = _build_candidate_scene_brief_en(
        behavior_name=behavior_name,
        role=role,
        scene_context=scene_context,
        key_dynamic_context=key_dynamic_context,
        candidate_static_context=candidate_static_context,
    )

    rationale = _build_action_rationale_en(
        behavior_name=behavior_name,
        role=role,
        info=info,
        scene_context=scene_context,
        key_dynamic_context=key_dynamic_context,
        candidate_static_context=candidate_static_context,
        selected_behavior_name=selected_behavior_name,
    )

    final_response = _build_final_response_en(
        behavior_name=behavior_name,
        role=role,
        selected_behavior_name=selected_behavior_name,
    )

    answer = _dedup_sentences_en([
        scene_brief,
        rationale,
        final_response,
    ])

    return {
        "scene_brief": scene_brief,
        "action_rationale": rationale,
        "final_response": final_response,
        "answer": answer,
    }


def _invalid_priority(candidate: Dict) -> int:
    """
    Select more informative negative samples first.
    Lower is more important.
    """
    behavior = str(candidate.get("behavior_name", ""))
    if behavior in ["left_nudge", "right_nudge"]:
        return 0
    if behavior in ["emergency_brake", "yield_stop"]:
        return 1
    if behavior in ["cautious_follow", "creep"]:
        return 2
    return 3


def _build_training_qa_pairs(dreamer_candidates: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Build compact training QA pairs.
    Do not generate one QA for every candidate; this avoids repetitive templates.
    """
    qa_zh: List[Dict] = []
    qa_en: List[Dict] = []

    selected = [c for c in dreamer_candidates if c.get("is_selected", False)]
    valid_alternatives = [c for c in dreamer_candidates if c.get("dreamer_role") == "valid_alternative"]
    invalids = [c for c in dreamer_candidates if c.get("dreamer_role") == "invalid_counterfactual"]

    if selected:
        s = selected[0]
        qa_zh.append({
            "task": "action_prediction",
            "question": "当前场景下车辆应该如何行驶？",
            "answer": s["vla_answer_zh"],
        })
        qa_en.append({
            "task": "action_prediction",
            "question": "What should the ego vehicle do in the current scene?",
            "answer": s["vla_answer_en"],
        })

    if valid_alternatives:
        valid_sorted = sorted(valid_alternatives, key=lambda c: safe_float(c.get("score", 1e9), 1e9))
        a = valid_sorted[0]
        qa_zh.append({
            "task": "alternative_action",
            "question": f"除了推荐动作，是否还可以选择“{a['behavior_name_zh']}”？",
            "answer": a["vla_answer_zh"],
        })
        qa_en.append({
            "task": "alternative_action",
            "question": f"Besides the recommended action, is it feasible to {a['behavior_name_en']}?",
            "answer": a["vla_answer_en"],
        })

    if invalids:
        invalid_sorted = sorted(
            invalids,
            key=lambda c: (_invalid_priority(c), safe_float(c.get("score", 1e9), 1e9)),
        )

        for c in invalid_sorted[:2]:
            qa_zh.append({
                "task": "action_suitability",
                "question": f"为什么不建议“{c['behavior_name_zh']}”？",
                "answer": c["vla_answer_zh"],
            })
            qa_en.append({
                "task": "action_suitability",
                "question": f"Why is it not recommended to {c['behavior_name_en']}?",
                "answer": c["vla_answer_en"],
            })

    return qa_zh, qa_en


def _select_candidate_indices(scored_rollouts: List[Dict], selected_idx: int, args) -> List[int]:
    include_invalid = bool(getattr(args, "dreamer_include_invalid", False))
    max_candidates = int(getattr(args, "dreamer_max_candidates", -1))

    selected = []
    valid = []
    invalid = []

    for i, r in enumerate(scored_rollouts):
        info = r.get("info", {})
        if i == selected_idx:
            selected.append(i)
        elif bool(info.get("allowed", False)):
            valid.append(i)
        elif include_invalid:
            invalid.append(i)

    valid = sorted(valid, key=lambda i: safe_float(scored_rollouts[i].get("info", {}).get("score", 0.0)))
    invalid = sorted(invalid, key=lambda i: safe_float(scored_rollouts[i].get("info", {}).get("score", 0.0)))

    indices = selected + valid + invalid
    if max_candidates > 0:
        # Always keep the selected candidate, then truncate the rest.
        indices = indices[:max_candidates]
    return indices


def build_dreamer_annotation(
    frame_name: str,
    scored_rollouts: List[Dict],
    selected_idx: int,
    scene_context: Dict,
    key_dynamic_context: Optional[Dict],
    static_candidate_context: Optional[Dict],
    expert_future_waypoints: Optional[np.ndarray],
    future_fps: float,
    args,) -> Dict:
    """
    Build SimLingo-style dreamer annotations from all risk-planned candidates.

    This does not change planning. It only converts existing scored rollouts into
    language-conditioned candidate actions.
    """

    candidate_indices = _select_candidate_indices(scored_rollouts, selected_idx, args)

    selected_info = scored_rollouts[selected_idx].get("info", {}) if 0 <= selected_idx < len(scored_rollouts) else {}
    selected_behavior_name = str(selected_info.get("behavior_name", "unknown"))

    dreamer_candidates = []

    for out_i, cand_i in enumerate(candidate_indices):
        r = scored_rollouts[cand_i]
        info = r.get("info", {})
        rollout = r.get("rollout", {})
        behavior_name = str(info.get("behavior_name", "unknown"))
        is_selected = bool(cand_i == selected_idx)
        role = _candidate_role(info, is_selected=is_selected)

        candidate_static = _get_candidate_static_context(
            static_candidate_context,
            source_candidate_index=cand_i,
        )

        vla_zh = _build_scene_wise_answer_zh(
            behavior_name=behavior_name,
            role=role,
            info=info,
            scene_context=scene_context,
            key_dynamic_context=key_dynamic_context,
            candidate_static_context=candidate_static,
            selected_behavior_name=selected_behavior_name,
        )
        vla_en = _build_scene_wise_answer_en(
            behavior_name=behavior_name,
            role=role,
            info=info,
            scene_context=scene_context,
            key_dynamic_context=key_dynamic_context,
            candidate_static_context=candidate_static,
            selected_behavior_name=selected_behavior_name,
        )

        waypoints = np.asarray(rollout.get("waypoints", []), dtype=np.float32)
        yaws = np.asarray(rollout.get("yaws", []), dtype=np.float32)
        speeds = np.asarray(rollout.get("speeds", []), dtype=np.float32)
        controls = np.asarray(rollout.get("controls", []), dtype=np.float32)

        dreamer_candidates.append({
            "dreamer_candidate_id": int(out_i),
            "source_candidate_index": int(cand_i),
            "is_selected": bool(is_selected),
            "dreamer_role": role,
            "dreamer_role_zh": _role_zh(role),
            "dreamer_role_en": _role_en(role),

            "behavior_id": int(info.get("behavior_id", -1)),
            "behavior_name": behavior_name,
            "behavior_name_zh": _behavior_display_name_zh(behavior_name, role),
            "behavior_name_en": _behavior_display_name_en(behavior_name, role),
            "behavior_active": bool(info.get("behavior_active", True)),
            "activation_reason": str(info.get("activation_reason", "")),
            "reference_mode": str(info.get("reference_mode", "")),
            "speed_mode": str(info.get("speed_mode", "")),
            "target_speed": safe_float(info.get("target_speed", 0.0)),

            "scene_brief_zh": vla_zh["scene_brief"],
            "scene_brief_en": vla_en["scene_brief"],
            "action_rationale_zh": vla_zh["action_rationale"],
            "action_rationale_en": vla_en["action_rationale"],
            "final_response_zh": vla_zh["final_response"],
            "final_response_en": vla_en["final_response"],

            "vla_answer_zh": vla_zh["answer"],
            "vla_answer_en": vla_en["answer"],
            "reasoning_style": "scene_intent_action",

            "candidate_static_context": candidate_static,

            "allowed": bool(info.get("allowed", False)),
            "reasons": info.get("reasons", []),
            "score": safe_float(info.get("score", 0.0)),
            "mean_cost": safe_float(info.get("mean_cost", 0.0)),
            "max_cost": safe_float(info.get("max_cost", 0.0)),
            "hard_ratio": safe_float(info.get("hard_ratio", 0.0)),
            "out_of_bounds_ratio": safe_float(info.get("out_of_bounds_ratio", 0.0)),
            "mean_route_deviation": safe_float(info.get("mean_route_deviation", 0.0)),
            "max_route_deviation": safe_float(info.get("max_route_deviation", 0.0)),
            "score_breakdown": info.get("score_breakdown", {}),

            "reference_route": points_to_list(r.get("reference_route", np.zeros((0, 2), dtype=np.float32))),
            "waypoints": points_to_list(waypoints),
            "yaws": array_to_list(yaws),
            "speeds": array_to_list(speeds),
            "controls_steer_acc": np.round(controls.astype(float), 4).tolist() if controls.size > 0 else [],
        })

    qa_pairs_zh, qa_pairs_en = _build_training_qa_pairs(dreamer_candidates)

    return {
        "frame": frame_name,
        # "dreamer_generator": "risk_grounded_multibehavior_dreamer_v1",
        "dreamer_generator": "risk_grounded_scenewise_contrastive_dreamer_v2",
        "dreamer_num_candidates": int(len(dreamer_candidates)),
        "dreamer_include_invalid": bool(getattr(args, "dreamer_include_invalid", False)),
        "dreamer_max_candidates": int(getattr(args, "dreamer_max_candidates", -1)),
        "future_fps": float(future_fps),
        "has_expert_future_waypoints": bool(expert_future_waypoints is not None),
        "key_dynamic_context": key_dynamic_context or {},
        "static_candidate_context": static_candidate_context or {},
        "lateral_space_context": {
            "left": _side_space_status("left", scene_context, key_dynamic_context),
            "right": _side_space_status("right", scene_context, key_dynamic_context),
        },
        "dreamer_candidates": dreamer_candidates,
        "dreamer_qa_pairs_zh": qa_pairs_zh,
        "dreamer_qa_pairs_en": qa_pairs_en,
    }