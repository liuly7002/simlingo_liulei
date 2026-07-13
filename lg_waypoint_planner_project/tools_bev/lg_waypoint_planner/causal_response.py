# -*- coding: utf-8 -*-

"""Causal object discovery and minimum-response supervision.

The module deliberately differs from SimLingo-style instruction intervention.
It does not treat alternative language commands as training targets.  Instead,
it intervenes on the scene by removing one actor from the offline environment,
replans in that counterfactual environment, and measures whether the selected
ego response changes.

The resulting pair

    object-removed reference trajectory -> full-scene response trajectory

provides supervision for (1) the causally influential object, (2) the required
driving response, (3) the resulting waypoint effect, and (4) whether the final
response is necessary, sufficient, and minimal.
"""

from typing import Dict, List, Any, Optional, Tuple
import copy
import math
import numpy as np

from .candidate_generator import build_candidates, build_nominal_reference_candidate
from .critical_factor import (
    actor_relevance_score,
    enrich_actor_route_relation,
    factor_from_actor,
    identify_critical_factor,
)
from .intent_policy import infer_intents
from .evaluator import evaluate_candidate, select_minimum_response, refine_minimum_sufficient_response, trajectory_response_metrics


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


def _cfg_int(obj: Any, key: str, default: int) -> int:
    try:
        return int(_cfg_get(obj, key, default))
    except Exception:
        return int(default)


def _cfg_bool(obj: Any, key: str, default: bool) -> bool:
    value = _cfg_get(obj, key, default)
    if isinstance(value, str):
        return value.lower() in ["1", "true", "yes", "y", "on"]
    return bool(value)


def actor_key(actor: Dict):
    actor_id = actor.get("id", None)  # 获取actor的id
    if actor_id is not None:
        return ("id", str(actor_id))  # ("id","123")
    return (
        "fallback",
        str(actor.get("class", "unknown")),      # 获取actor的类别
        round(float(actor.get("x_m", 0.0)), 1),  # 获取actor的x坐标并四舍五入到小数点后一位
        round(float(actor.get("y_m", 0.0)), 1),  # 获取actor的y坐标并四舍五入到小数点后一位
    )


def same_actor(a: Dict, b: Dict) -> bool:
    aid = a.get("id", None)
    bid = b.get("id", None)
    # 情况1：如果两个actor都有id，则直接比较id是否相同(正常情况下主要走这里)
    if aid is not None and bid is not None:
        return str(aid) == str(bid)
    # 情况2：如果两个actor都没有id，则比较class和位置是否相近
    if str(a.get("class", "")) != str(b.get("class", "")):
        return False
    dx = float(a.get("x_m", 0.0)) - float(b.get("x_m", 0.0))
    dy = float(a.get("y_m", 0.0)) - float(b.get("y_m", 0.0))
    return math.hypot(dx, dy) <= 1.5  # 如果两个actor的欧式距离小于等于1.5米，则认为是同一个actor


def rank_causal_actor_candidates(
    current_actors: List[Dict],
    cfg,
    reference_route: Optional[np.ndarray] = None,
) -> List[Dict]:
    cr = _cfg_get(cfg, "causal_response", {})
    max_objects = max(_cfg_int(cr, "max_objects_to_test", 5), 1)  # 最多测试的对象数量
    max_distance = _cfg_float(cr, "max_object_distance_m", 35.0)  # 最远距离<=35m
    min_forward = _cfg_float(cr, "min_forward_x_m", -1.0)         # 自车后方1m外的不测试

    # 遍历所有对象
    ranked = []
    for actor in current_actors:
        item = (
            enrich_actor_route_relation(actor, reference_route, cfg)
            if reference_route is not None
            else dict(actor)
        )
        x = float(item.get("x_m", 0.0))  # actor在自车纵向方向上的距离
        d = float(item.get("distance_m", math.hypot(x, float(item.get("y_m", 0.0)))))  # actor与自车之间的欧式距离
        if x < min_forward or d > max_distance:
            continue
        score = float(actor_relevance_score(item, cfg))  # 计算actor的相关性分数
        if not np.isfinite(score):
            continue
        # 添加测试信息
        item["causal_test_relevance_score"] = score  # 将相关性分数添加到actor字典中
        item["exists"] = True  # 标记actor存在
        ranked.append(item)

    # 排序: 首先按相关性分数降序排序，相关性分数越高越靠前，如果相关性分数相同，则按距离排序，距离越近越靠前
    ranked.sort(key=lambda a: (-float(a.get("causal_test_relevance_score", 0.0)), float(a.get("distance_m", 1e9))))
    return ranked[:max_objects]


def _joint_rule_actor_constraint(
    base_route: np.ndarray,
    actor_candidates: List[Dict],
    initial_factor: Dict,
    cfg,
) -> Dict:
    """Return the most restrictive front-actor stop constraint under a rule.

    A current red light or stop sign remains the primary semantic cause, but a
    nearer actor on the intended route can impose an earlier longitudinal stop
    position.  The constraints are composed geometrically instead of competing
    as separate candidate intents.
    """
    try:
        rule_stop = float((initial_factor or {}).get("required_stop_distance_m", float("nan")))
    except Exception:
        rule_stop = float("nan")
    if not np.isfinite(rule_stop):
        return {"active": False}

    route = np.asarray(base_route, dtype=np.float32)
    if route.ndim != 2 or len(route) == 0 or route.shape[1] < 2:
        return {"active": False}
    route = route[:, :2]

    rg = _cfg_get(cfg, "response_generation", {})
    ego_half_l = float(cfg.vehicle.ego_half_length_m)
    ego_half_w = float(cfg.vehicle.ego_half_width_m)
    longitudinal_margin = _cfg_float(
        rg,
        "longitudinal_safety_margin_m",
        float(cfg.behaviors.stop_margin_m),
    )
    lateral_margin = _cfg_float(rg, "lateral_safety_margin_m", 0.45)
    min_stop = float(cfg.behaviors.min_stop_distance_m)

    best = None
    for actor in actor_candidates:
        if not actor.get("exists", False):
            continue
        x = float(actor.get("x_m", -1.0))
        if x <= 0.0:
            continue

        actor_xy = np.asarray([x, float(actor.get("y_m", 0.0))], dtype=np.float32)
        route_distance = float(np.min(np.linalg.norm(route - actor_xy[None, :], axis=1)))
        actor_half_w = max(float(actor.get("half_width_m", 1.0)), 0.1)
        path_band = ego_half_w + actor_half_w + lateral_margin
        if route_distance > path_band:
            continue

        actor_half_l = max(float(actor.get("half_length_m", 1.0)), 0.1)
        actor_stop = max(
            min_stop,
            x - actor_half_l - ego_half_l - longitudinal_margin,
        )
        # Only actors that require an earlier stop than the traffic rule are
        # part of the joint longitudinal constraint.
        if actor_stop >= rule_stop - 1e-3:
            continue

        item = {
            "active": True,
            "rule_type": str((initial_factor or {}).get("type", "unknown")),
            "rule_required_stop_distance_m": float(rule_stop),
            "actor_required_stop_distance_m": float(actor_stop),
            "effective_stop_distance_m": float(min(rule_stop, actor_stop)),
            "actor_route_distance_m": float(route_distance),
            "constraint_actor": dict(actor),
        }
        if best is None or float(item["effective_stop_distance_m"]) < float(best["effective_stop_distance_m"]):
            best = item

    return best if best is not None else {"active": False}


def _candidate_pool_key(candidate: Dict):
    obj = candidate.get("response_object", {}) or {}
    obj_key = actor_key(obj) if obj.get("exists", False) else ("none",)  # 判断两个候选是不是重复候选
    return (
        str(candidate.get("intent_name", "unknown")),
        str(candidate.get("variant_id", "default")),
        obj_key,
    )


# 构建因果候选池
def build_causal_candidate_pool(
    base_route: np.ndarray,
    initial_factor: Dict,
    current_actors: List[Dict],
    measurement: Dict,
    cfg,
    expert_future: Optional[np.ndarray] = None,) -> Tuple[List[Dict], Dict, List[Dict]]:
    """Build a compact pool of object-conditioned response candidates.

    The pool contains one global nominal route-follow candidate and active
    response candidates anchored to several plausible objects.  The later
    object-removal test, rather than the relevance heuristic, decides which
    object actually caused the selected response.
    """
    nominal = build_nominal_reference_candidate(base_route, measurement, cfg, expert_future=expert_future)

    # Active traffic rules are immutable causes, not actors that can be removed
    # by the causal intervention.  When the current frame already requires a
    # stop, actor-conditioned lateral responses must not compete with the rule
    # response and steal the semantic label merely because the ego is nearly
    # stationary. Future-only rules keep the ordinary actor candidate pool.
    red_rule = (initial_factor or {}).get("red_light_rule") or {}
    current_red_active = (
        str((initial_factor or {}).get("type", "")) == "red_light_stop_line"
        and bool(red_rule.get("current_red_active", False))
    )
    stop_rule = (initial_factor or {}).get("stop_sign_rule") or {}
    current_stop_active = (
        str((initial_factor or {}).get("type", "")) == "stop_sign_control"
        and bool(stop_rule.get("current_stop_active", False))
    )
    # Keep ranking nearby actors even when a traffic rule is active.  They remain
    # valid attention targets and can coexist with the traffic light / stop sign
    # in the supervision.  However, while a current rule already requires a
    # stop, actor-conditioned response candidates are not allowed to compete
    # with the rule response; this preserves the correct driving intent.
    actor_candidates = rank_causal_actor_candidates(
        current_actors,
        cfg,
        reference_route=base_route,
    )
    rule_currently_active = bool(current_red_active or current_stop_active)

    # The nominal rollout is still returned separately as the no-interference
    # reference.  Once a current traffic rule is active, however, it must not
    # re-enter the final candidate competition as an ordinary route-follow
    # action.  Otherwise a nearly stationary route-follow / cautious candidate
    # can beat the true rule response merely because its numerical intervention
    # is smaller.
    pool = [] if rule_currently_active else [nominal]

    # Rule responses must always be generated, even when a relevant actor is
    # also present.  The actor remains available for attention and language,
    # but it is not the response object of the traffic-rule candidate.
    if rule_currently_active or not (initial_factor.get("critical_actor") or {}).get("exists", False):
        rule_factor = dict(initial_factor)
        if str(rule_factor.get("type", "")) in ["red_light_stop_line", "stop_sign_control"]:
            # Compose the active traffic rule with any nearer actor on the
            # intended route.  The traffic rule stays the primary semantic
            # cause, while the actor can move the immediate stop position
            # closer to the ego vehicle.
            if rule_currently_active:
                joint_constraint = _joint_rule_actor_constraint(
                    base_route=base_route,
                    actor_candidates=actor_candidates,
                    initial_factor=initial_factor,
                    cfg=cfg,
                )
                if joint_constraint.get("active", False):
                    rule_factor["rule_required_stop_distance_m"] = float(
                        joint_constraint["rule_required_stop_distance_m"]
                    )
                    rule_factor["required_stop_distance_m"] = float(
                        joint_constraint["effective_stop_distance_m"]
                    )
                    rule_factor["joint_longitudinal_constraint"] = joint_constraint
                    rule_factor["secondary_attention_actor"] = dict(
                        joint_constraint.get("constraint_actor", {"exists": False})
                    )
            rule_factor["critical_actor"] = {"exists": False}
        intents = infer_intents(rule_factor, cfg)
        for c in build_candidates(base_route, intents, rule_factor, measurement, cfg, expert_future=expert_future):
            if c.get("intent", {}).get("active", False) and c.get("intent_name") != "route_follow":
                pool.append(c)

    # Under an active traffic rule, keep actor candidates for attention /
    # counterfactual analysis but do not let their lateral responses steal the
    # selected rule intent.
    if not rule_currently_active:
        for actor in actor_candidates:
            factor = factor_from_actor(
                actor,
                stage="causal_candidate_generation",
                reference_route=base_route,
                cfg=cfg,
            )
            intents = infer_intents(factor, cfg)
            for c in build_candidates(base_route, intents, factor, measurement, cfg, expert_future=expert_future):
                # The causal pool uses only responses that are active for this
                # object.  Diagnostic inactive candidates would otherwise multiply
                # the pool without contributing to the final selector.
                if c.get("intent_name") == "route_follow":
                    continue
                if not c.get("intent", {}).get("active", False):
                    continue
                pool.append(c)

    dedup = []
    seen = set()
    for c in pool:
        key = _candidate_pool_key(c)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    return dedup, nominal, actor_candidates


def _actor_at_time(actor: Dict, actor_timelines: Dict[int, List[Dict]], time_index: int) -> Optional[Dict]:
    if time_index <= 0:
        return actor
    for other in actor_timelines.get(int(time_index), []):
        if same_actor(actor, other):
            return other
    return None


def remove_actor_from_timelines(actor: Dict, actor_timelines: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
    out = {}
    for k, records in actor_timelines.items():
        out[int(k)] = [dict(r) for r in records if not same_actor(actor, r)]
    return out


def _clear_actor_obb_from_map(costmap: np.ndarray, actor: Dict, ego_center, meters_per_pixel: float, cfg) -> np.ndarray:
    """Erase one actor footprint from a binary occupancy BEV.

    The operation is intentionally local.  Only cells whose centers fall inside
    the actor OBB plus a small configurable margin are replaced by the free
    value.  This avoids globally modifying road geometry when constructing the
    counterfactual environment.
    """
    if actor is None:
        return costmap

    cr = _cfg_get(cfg, "causal_response", {})
    margin = _cfg_float(cr, "costmap_clear_margin_m", 0.35)
    free_value = _cfg_float(cr, "counterfactual_free_value", 0.0)
    x = float(actor.get("x_m", 0.0))
    y = float(actor.get("y_m", 0.0))
    yaw = float(actor.get("yaw_rad", 0.0))

    raw_hl = max(float(actor.get("half_length_m", 1.0)), 0.1)
    raw_hw = max(float(actor.get("half_width_m", 0.5)), 0.1)

    # The collected walker BEV mask is not drawn from the raw CARLA walker
    # extent.  ObsManager first scales the walker bbox by 2.0 and then clamps
    # each horizontal half-extent to at least 0.8 m.  Counterfactual deletion
    # must reproduce that footprint; otherwise most of the pedestrian mask
    # remains after the actor is supposedly removed.
    is_pedestrian = str(actor.get("class", "")).lower() == "pedestrian"
    if is_pedestrian:
        walker_scale = _cfg_float(cr, "pedestrian_mask_extent_scale", 2.0)
        walker_min_half_extent = _cfg_float(cr, "pedestrian_mask_min_half_extent_m", 0.8)
        hl = max(raw_hl * walker_scale, walker_min_half_extent)
        hw = max(raw_hw * walker_scale, walker_min_half_extent)
    else:
        hl = raw_hl + margin
        hw = raw_hw + margin

    h, w = costmap.shape[:2]
    mpp = max(float(meters_per_pixel), 1e-6)
    if is_pedestrian:
        # ``cv.fillConvexPoly`` rasterizes boundary pixels whose centers can lie
        # just outside the geometric 0.8 m half-extent.  Half of one pixel
        # diagonal is the minimum resolution-aware padding that covers those
        # boundary cells without reverting to the old raw-bbox + fixed-margin
        # approximation.  At 0.5 m/pixel this is about 0.354 m.
        raster_pad = 0.5 * math.sqrt(2.0) * mpp
        hl += raster_pad
        hw += raster_pad
    # Bounding radius gives a small pixel window; we then test cells exactly in
    # the actor frame so no OpenCV dependency is required.
    radius = math.hypot(hl, hw)
    col0 = int(math.floor(float(ego_center[0]) + (y - radius) / mpp))
    col1 = int(math.ceil(float(ego_center[0]) + (y + radius) / mpp))
    row0 = int(math.floor(float(ego_center[1]) - (x + radius) / mpp))
    row1 = int(math.ceil(float(ego_center[1]) - (x - radius) / mpp))
    if col1 < 0 or col0 >= w or row1 < 0 or row0 >= h:
        return costmap
    col0 = max(0, min(w - 1, col0)); col1 = max(0, min(w - 1, col1))
    row0 = max(0, min(h - 1, row0)); row1 = max(0, min(h - 1, row1))
    if col1 < col0 or row1 < row0:
        return costmap

    rows, cols = np.meshgrid(
        np.arange(row0, row1 + 1, dtype=np.float32),
        np.arange(col0, col1 + 1, dtype=np.float32),
        indexing="ij",
    )
    px = (float(ego_center[1]) - rows) * mpp
    py = (cols - float(ego_center[0])) * mpp
    dx = px - x
    dy = py - y
    c, s = math.cos(yaw), math.sin(yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    inside = (np.abs(local_x) <= hl) & (np.abs(local_y) <= hw)

    out = costmap.copy()
    view = out[row0:row1 + 1, col0:col1 + 1]
    view[inside] = free_value
    return out


def remove_actor_from_temporal_bundle(
    actor: Dict,
    temporal_bundle: Dict,
    actor_timelines: Dict[int, List[Dict]],
    ego_center,
    meters_per_pixel: float,
    cfg,) -> Dict:
    """Construct the scene intervention ``do(environment without actor)``.

    Dynamic collision records are removed separately by
    ``remove_actor_from_timelines``.  Here we also erase the corresponding OBB
    from occupancy maps; otherwise the costmap would still contain the removed
    object and the intervention would be ineffective.
    """
    out = dict(temporal_bundle)
    maps = []
    future_mode = str(_cfg_get(_cfg_get(cfg, "costmap", {}), "future_combine", "max"))
    future_contains_current = future_mode in {"max", "add"}
    for tidx, costmap in enumerate(temporal_bundle.get("costmaps", [])):
        cm = np.asarray(costmap, dtype=np.float32).copy()

        if tidx == 0:
            # The first temporal map is the current frame.
            cm = _clear_actor_obb_from_map(cm, actor, ego_center, meters_per_pixel, cfg)
        else:
            # ``future_only`` maps contain occupancy at that future time only.
            # Union/add modes also contain the current footprint and therefore
            # require clearing both copies for a valid intervention.
            if future_contains_current:
                cm = _clear_actor_obb_from_map(cm, actor, ego_center, meters_per_pixel, cfg)
            future_actor = _actor_at_time(actor, actor_timelines, tidx)
            if future_actor is not None:
                cm = _clear_actor_obb_from_map(cm, future_actor, ego_center, meters_per_pixel, cfg)
        maps.append(cm)
    out["costmaps"] = maps
    out["counterfactual_removed_actor"] = {
        "id": actor.get("id", None),
        "class": actor.get("class", "unknown"),
    }
    return out


def evaluate_candidate_pool(
    candidates: List[Dict],
    base_route: np.ndarray,
    temporal_bundle: Dict,
    ego_center,
    meters_per_pixel: float,
    actor_timelines: Dict[int, List[Dict]],
    cfg,
    factor: Optional[Dict] = None,) -> List[Dict]:
    return [
        evaluate_candidate(
            c,
            base_route,
            temporal_bundle,
            ego_center,
            meters_per_pixel,
            actor_timelines,
            cfg,
            factor=factor,
        )
        for c in candidates
    ]


def _intent_change(a: Dict, b: Dict) -> bool:
    return str(a.get("info", {}).get("intent_name", "")) != str(b.get("info", {}).get("intent_name", ""))


def _route_follow_release(full_scored: List[Dict], cf_scored: List[Dict]) -> bool:
    """Return whether actor removal newly makes route-follow feasible.

    Full-scene and counterfactual candidate pools are generated independently,
    so they may differ in length, order, variants, and response objects.  Route
    release must therefore be detected within each pool separately rather than
    by zipping candidates with matching indices.
    """
    full_valid = any(
        c.get("intent_name") == "route_follow"
        and bool(c.get("info", {}).get("allowed", False))
        for c in full_scored
    )
    cf_valid = any(
        c.get("intent_name") == "route_follow"
        and bool(c.get("info", {}).get("allowed", False))
        for c in cf_scored
    )
    return (not full_valid) and cf_valid


def _causal_score(full_selected: Dict, cf_selected: Dict, route_follow_release: bool, cfg) -> Tuple[float, Dict]:
    """Return a physical action-effect score, not a semantic-label bonus score.

    Intent changes and route-follow release remain explicit structural evidence,
    but they are no longer added as large constants.  Consequently a tiny motion
    change cannot be reported as a large causal magnitude merely because two
    discrete intent names differ.
    """
    cr = _cfg_get(cfg, "causal_response", {})
    metrics = trajectory_response_metrics(
        full_selected.get("rollout", {}),
        cf_selected.get("rollout", {}),
        cfg=cfg,
    )
    intent_changed = _intent_change(full_selected, cf_selected)

    score = (
        _cfg_float(cr, "causal_longitudinal_weight", 1.0)
        * float(metrics["mean_abs_longitudinal_change_m"])
        + _cfg_float(cr, "causal_lateral_weight", 2.0)
        * float(metrics["mean_abs_lateral_change_m"])
        + _cfg_float(cr, "causal_max_lateral_weight", 0.5)
        * float(metrics["max_abs_lateral_change_m"])
        + _cfg_float(cr, "causal_speed_weight", 0.2)
        * float(metrics["mean_abs_speed_change_mps"])
    )
    details = dict(metrics)
    details["physical_effect_score"] = float(score)
    details["intent_changed"] = bool(intent_changed)
    details["route_follow_released_after_removal"] = bool(route_follow_release)
    return float(score), details


def _causal_acceptance(score: float, details: Dict, cfg) -> Tuple[bool, str]:
    """Decide causal validity while keeping magnitude and necessity separate.

    A sufficiently large physical effect is causal directly.  A smaller effect
    may still be necessary when actor removal uniquely releases route-following,
    but a small nonzero floor prevents numerical noise from becoming causal.
    """
    cr = _cfg_get(cfg, "causal_response", {})
    min_score = _cfg_float(cr, "min_causal_score", 0.60)
    min_release_effect = _cfg_float(cr, "min_route_follow_release_effect_score", 0.03)
    physical_score = float(score)
    if physical_score >= min_score:
        return True, "physical_effect_above_threshold"
    if (
        bool(details.get("route_follow_released_after_removal", False))
        and physical_score >= min_release_effect
    ):
        return True, "route_follow_release_with_nonzero_physical_effect"
    return False, "physical_effect_below_threshold"


def analyze_causal_objects(
    candidates: List[Dict],
    full_scored: List[Dict],
    full_selected_idx: int,
    nominal_reference_rollout: Dict,
    actor_candidates: List[Dict],
    current_actors: List[Dict],
    base_route: np.ndarray,
    measurement: Dict,
    temporal_bundle: Dict,
    ego_center,
    meters_per_pixel: float,
    actor_timelines: Dict[int, List[Dict]],
    cfg,
    expert_future: Optional[np.ndarray] = None,
) -> Dict:
    """Find the object whose removal most changes the selected ego behavior.

    Every object-removal intervention is treated as a new planning problem.  The
    counterfactual scene is re-diagnosed and receives its own intent hypotheses,
    candidate pool, safety evaluation, and minimum-response selection.  This
    prevents a newly exposed actor or traffic rule from being evaluated only
    through actions that were generated for the original full scene.
    """
    cr = _cfg_get(cfg, "causal_response", {})
    enabled = _cfg_bool(cr, "enabled", True)
    if not enabled or full_selected_idx < 0 or not actor_candidates:
        return {
            "enabled": enabled,
            "has_causal_object": False,
            "causal_object": {"exists": False},
            "causal_score": 0.0,
            "preliminary_causal_score": 0.0,
            "final_causal_score": None,
            "final_revalidation_passed": None,
            "reference_rollout": nominal_reference_rollout,
            "reference_source": "nominal_no_interference",
            "counterfactual_selected": None,
            "counterfactual_candidate_count": 0,
            "object_tests": [],
        }

    full_selected = full_scored[full_selected_idx]
    tests = []
    best = None

    for actor in actor_candidates:
        cf_bundle = remove_actor_from_temporal_bundle(
            actor,
            temporal_bundle,
            actor_timelines,
            ego_center,
            meters_per_pixel,
            cfg,
        )
        cf_timelines = remove_actor_from_timelines(actor, actor_timelines)

        # Removing actor A does not imply that the remaining scene is clear.
        # Re-run the same scene diagnosis used by the full-scene planner on the
        # counterfactual occupancy maps and the remaining current actors.
        cf_current_actors = [dict(a) for a in current_actors if not same_actor(actor, a)]
        cf_factor = identify_critical_factor(
            base_route,
            cf_current_actors,
            cf_bundle,
            ego_center,
            meters_per_pixel,
            cfg,
        )
        cf_factor = dict(cf_factor)
        cf_factor["stage"] = "counterfactual_object_removed_rediagnosed"
        cf_factor["counterfactual_removed_actor"] = {
            "id": actor.get("id", None),
            "class": actor.get("class", "unknown"),
        }

        # True counterfactual replanning: regenerate action hypotheses from the
        # re-diagnosed scene instead of re-scoring the original full-scene pool.
        cf_candidates, cf_nominal, _ = build_causal_candidate_pool(
            base_route=base_route,
            initial_factor=cf_factor,
            current_actors=cf_current_actors,
            measurement=measurement,
            cfg=cfg,
            expert_future=expert_future,
        )
        cf_scored = evaluate_candidate_pool(
            cf_candidates,
            base_route,
            cf_bundle,
            ego_center,
            meters_per_pixel,
            cf_timelines,
            cfg,
            factor=cf_factor,
        )
        cf_reference = cf_nominal.get("rollout", nominal_reference_rollout)
        cf_idx = select_minimum_response(cf_scored, cf_reference, cfg)
        if cf_idx >= 0:
            # Counterfactual references are planning outputs as well.  Refine a
            # continuous selected response before it is used to define the
            # object-removed reference and the causal effect.
            cf_refined = refine_minimum_sufficient_response(
                selected=cf_scored[cf_idx],
                reference_rollout=cf_reference,
                base_route=base_route,
                temporal_bundle=cf_bundle,
                ego_center=ego_center,
                meters_per_pixel=meters_per_pixel,
                actor_timelines=cf_timelines,
                measurement=measurement,
                cfg=cfg,
                factor=cf_factor,
            )
            if cf_refined is not cf_scored[cf_idx]:
                cf_scored.append(cf_refined)
                cf_idx = len(cf_scored) - 1
            else:
                cf_scored[cf_idx] = cf_refined

        if cf_idx < 0:
            test = {
                "actor": dict(actor),
                "counterfactual_valid": False,
                "causal_score": 0.0,
                "counterfactual_selected_index": -1,
                "counterfactual_candidate_count": int(len(cf_scored)),
                "counterfactual_factor_type": str(cf_factor.get("type", "unknown")),
                "counterfactual_remaining_actor_count": int(len(cf_current_actors)),
            }
            tests.append(test)
            continue

        cf_selected = cf_scored[cf_idx]
        release = _route_follow_release(full_scored, cf_scored)
        score, details = _causal_score(full_selected, cf_selected, release, cfg)
        accepted, acceptance_reason = _causal_acceptance(score, details, cfg)
        test = {
            "actor": dict(actor),
            "counterfactual_valid": True,
            "causal_score": float(score),
            "preliminary_causal_score": float(score),
            "causal_accepted": bool(accepted),
            "causal_acceptance_reason": acceptance_reason,
            "counterfactual_selected_index": int(cf_idx),
            "counterfactual_candidate_count": int(len(cf_scored)),
            "counterfactual_intent_name": cf_selected.get("info", {}).get("intent_name", "unknown"),
            "counterfactual_selected_variant": cf_selected.get("info", {}).get(
                "variant_id", cf_selected.get("variant_id", "default")
            ),
            "full_scene_intent_name": full_selected.get("info", {}).get("intent_name", "unknown"),
            "counterfactual_factor_type": str(cf_factor.get("type", "unknown")),
            "counterfactual_remaining_actor_count": int(len(cf_current_actors)),
            "effect": details,
        }
        tests.append(test)
        if accepted and (best is None or float(score) > float(best["causal_score"])):
            best = {
                **test,
                "counterfactual_selected": cf_selected,
                "counterfactual_scored": cf_scored,
                "counterfactual_factor": cf_factor,
            }

    if best is None:
        best_observed = max(tests, key=lambda x: float(x.get("causal_score", 0.0)), default=None)
        return {
            "enabled": True,
            "has_causal_object": False,
            "causal_object": {"exists": False},
            "causal_score": float(best_observed.get("causal_score", 0.0)) if best_observed else 0.0,
            "preliminary_causal_score": float(best_observed.get("causal_score", 0.0)) if best_observed else 0.0,
            "final_causal_score": None,
            "final_revalidation_passed": None,
            "causal_acceptance_reason": (
                best_observed.get("causal_acceptance_reason", "no_valid_counterfactual")
                if best_observed else "no_valid_counterfactual"
            ),
            "reference_rollout": nominal_reference_rollout,
            "reference_source": "nominal_no_interference",
            "counterfactual_selected": None,
            "counterfactual_candidate_count": int(best_observed.get("counterfactual_candidate_count", 0)) if best_observed else 0,
            "object_tests": tests,
        }

    causal_object = dict(best["actor"])
    causal_object["exists"] = True
    causal_object["causal_score"] = float(best["causal_score"])
    causal_object["causal_effect"] = dict(best.get("effect", {}))
    return {
        "enabled": True,
        "has_causal_object": True,
        "causal_object": causal_object,
        "causal_score": float(best["causal_score"]),
        "preliminary_causal_score": float(best["causal_score"]),
        "final_causal_score": None,
        "causal_acceptance_reason": best.get("causal_acceptance_reason", "unknown"),
        "final_revalidation_passed": None,
        "reference_rollout": best["counterfactual_selected"].get("rollout", nominal_reference_rollout),
        "reference_source": "object_removed_counterfactual",
        "counterfactual_selected": best["counterfactual_selected"],
        "counterfactual_scored": best["counterfactual_scored"],
        "counterfactual_intent_name": best.get("counterfactual_intent_name", "unknown"),
        "counterfactual_selected_variant": best.get("counterfactual_selected_variant", "default"),
        "counterfactual_candidate_count": int(best.get("counterfactual_candidate_count", 0)),
        "object_tests": tests,
    }


def revalidate_causal_analysis(
    final_selected: Dict,
    full_scored: List[Dict],
    causal_analysis: Dict,
    nominal_reference_rollout: Dict,
    cfg,
) -> Dict:
    """Re-test the discovered actor against the final supervised trajectory.

    Causal discovery is performed with a preliminary full-scene trajectory, but
    the final selector may subsequently choose a different response after the
    causal object is known.  The public causal score and effect must therefore
    be recomputed against that final trajectory.  A failed second test cancels
    the actor and restores the nominal no-interference reference.
    """
    out = dict(causal_analysis)
    if not out.get("has_causal_object", False):
        out.setdefault("final_revalidation_passed", None)
        return out

    preliminary_score = float(out.get("causal_score", 0.0))
    cf_selected = out.get("counterfactual_selected")
    cf_scored = out.get("counterfactual_scored", []) or []
    if not isinstance(cf_selected, dict):
        out.update({
            "has_causal_object": False,
            "rejected_causal_object": dict(out.get("causal_object", {})),
            "causal_object": {"exists": False},
            "causal_score": 0.0,
            "preliminary_causal_score": preliminary_score,
            "final_causal_score": 0.0,
            "final_revalidation_passed": False,
            "reference_rollout": nominal_reference_rollout,
            "reference_source": "nominal_no_interference",
            "counterfactual_selected": None,
        })
        return out

    release = _route_follow_release(full_scored, cf_scored)
    final_score, final_effect = _causal_score(final_selected, cf_selected, release, cfg)
    passed, acceptance_reason = _causal_acceptance(final_score, final_effect, cfg)

    out["preliminary_causal_score"] = preliminary_score
    out["final_causal_score"] = float(final_score)
    out["final_revalidation_passed"] = passed
    out["causal_acceptance_reason"] = acceptance_reason
    out["final_full_scene_intent_name"] = final_selected.get("info", {}).get("intent_name", "unknown")

    causal = dict(out.get("causal_object", {}))
    # Preserve both stages in the matching object-level diagnostic.
    for test in out.get("object_tests", []) or []:
        actor = test.get("actor", {}) or {}
        if same_actor(causal, actor):
            test["preliminary_causal_score"] = preliminary_score
            test["final_causal_score"] = float(final_score)
            test["final_revalidation_passed"] = passed
            test["final_causal_acceptance_reason"] = acceptance_reason
            test["final_full_scene_intent_name"] = out["final_full_scene_intent_name"]
            test["final_effect"] = dict(final_effect)
            break

    if passed:
        out["causal_score"] = float(final_score)
        causal["causal_score"] = float(final_score)
        causal["causal_effect"] = dict(final_effect)
        causal["causal_acceptance_reason"] = acceptance_reason
        out["causal_object"] = causal
        return out

    # The actor influenced the preliminary response but no longer explains the
    # final supervised response.  Do not retain its object-removed trajectory as
    # a reference once the causal claim has failed.
    out.update({
        "has_causal_object": False,
        "rejected_causal_object": causal,
        "causal_object": {"exists": False},
        "causal_score": float(final_score),
        "reference_rollout": nominal_reference_rollout,
        "reference_source": "nominal_no_interference",
        "rejected_counterfactual_intent_name": out.get("counterfactual_intent_name", None),
        "counterfactual_intent_name": None,
        "counterfactual_selected_variant": None,
        "counterfactual_selected": None,
    })
    return out


def causal_consistent_candidate_indices(scored: List[Dict], causal_analysis: Dict) -> Optional[List[int]]:
    """Keep global candidates and responses anchored to the discovered object."""
    if not causal_analysis.get("has_causal_object", False):
        return None
    causal = causal_analysis.get("causal_object", {})
    indices = []
    for i, candidate in enumerate(scored):
        obj = candidate.get("response_object", {}) or {}
        if not obj.get("exists", False):
            indices.append(i)
            continue
        if same_actor(causal, obj):
            indices.append(i)
    return indices


def build_causal_factor(
    causal_analysis: Dict,
    fallback_factor: Dict,
    reference_route: Optional[np.ndarray] = None,
    cfg=None,
) -> Dict:
    """Build the user-facing factor after causal object verification.

    A verified actor is allowed to coexist with traffic-light semantics.  The
    previous implementation replaced the whole factor with ``factor_from_actor``
    and therefore discarded ``traffic_light_state``.  This caused a yellow
    light or a red->green release to disappear whenever an actor was also
    causally relevant.

    Hard traffic rules keep their rule semantics as the primary factor.  For
    non-hard light states, red->green, yellow->green, and current-yellow are
    preserved as semantic context while the verified actor remains in
    ``critical_actor``.
    """
    if not causal_analysis.get("has_causal_object", False):
        return dict(fallback_factor)

    fallback = dict(fallback_factor) if isinstance(fallback_factor, dict) else {}
    actor = causal_analysis.get("causal_object", {})
    factor = factor_from_actor(
        actor,
        stage="causal_object_verified_by_removal",
        reference_route=reference_route,
        cfg=cfg,
    )

    # Preserve scene / rule context that is independent of which actor was
    # verified by the counterfactual removal test.
    for key in [
        "blocked_by_costmap",
        "corridor_stats",
        "red_light_rule",
        "stop_sign_rule",
        "traffic_light_state",
        "required_stop_distance_m",
    ]:
        if key in fallback:
            factor[key] = copy.deepcopy(fallback[key])

    fallback_type = str(fallback.get("type", ""))
    light_state = dict(fallback.get("traffic_light_state", {}) or {})
    current_light = str(light_state.get("current_state", "unknown"))
    previous_light = str(light_state.get("previous_state", "unknown"))
    transition = str(light_state.get("transition", ""))

    # Hard rules remain the primary semantic cause while the causal actor is
    # retained as a coexisting object of attention.
    if fallback_type == "red_light_stop_line":
        factor["type"] = "red_light_stop_line"
        factor["language_focus"] = "red_light_stop_line"
        factor["reason_en"] = fallback.get(
            "reason_en",
            "The traffic light ahead is red and requires the vehicle to remain behind the stop line.",
        )
    elif fallback_type == "stop_sign_control":
        factor["type"] = "stop_sign_control"
        factor["language_focus"] = "stop_sign_control"
        factor["reason_en"] = fallback.get(
            "reason_en",
            "The stop sign ahead requires a complete stop before proceeding.",
        )
    else:
        red_to_green = current_light == "green" and (
            previous_light == "red" or transition == "red_to_green"
        )
        yellow_to_green = current_light == "green" and (
            previous_light == "yellow" or transition == "yellow_to_green"
        )
        yellow_attention = current_light == "yellow"

        if red_to_green:
            factor["type"] = "green_light_release"
            factor["language_focus"] = "green_light_release"
            factor["reason_en"] = (
                "The traffic light ahead has changed from red to green, so the previous red-light "
                "stopping requirement is no longer active."
            )
        elif yellow_to_green:
            factor["type"] = "green_light_after_yellow"
            factor["language_focus"] = "green_light_after_yellow"
            factor["reason_en"] = (
                "The traffic light ahead has changed from yellow to green, so the vehicle may continue "
                "through the intersection while monitoring the surroundings."
            )
        elif yellow_attention:
            factor["type"] = "yellow_light_caution"
            factor["language_focus"] = "yellow_light_caution"
            factor["reason_en"] = "The traffic light ahead is yellow and requires a cautious response."

    factor["causal_score"] = float(causal_analysis.get("causal_score", 0.0))
    factor["reference_source"] = causal_analysis.get("reference_source", "object_removed_counterfactual")
    factor["has_direct_trajectory_influence"] = True
    return factor


def _action_effect_semantics(selected: Dict, reference_rollout: Dict, metrics: Dict, cfg) -> Dict:
    """Classify what the selected vehicle actually does, and how it differs from reference.

    This prevents a refined cautious response that still accelerates from being
    described as physical deceleration merely because its internal candidate
    family is called ``cautious_follow``.
    """
    sem = _cfg_get(cfg, "action_semantics", {})
    speed_ref_threshold = _cfg_float(sem, "speed_reference_delta_threshold_mps", 0.03)
    actual_trend_threshold = _cfg_float(sem, "actual_speed_trend_threshold_mps", 0.40)
    stationary_mean_threshold = _cfg_float(sem, "stationary_mean_speed_mps", 0.15)
    near_stop_end_threshold = _cfg_float(sem, "near_stop_end_speed_mps", 0.35)
    negligible_distance = _cfg_float(sem, "negligible_effect_distance", 0.05)
    slight_distance = _cfg_float(sem, "slight_effect_distance", 0.25)
    moderate_distance = _cfg_float(sem, "moderate_effect_distance", 1.0)

    speeds = np.asarray(selected.get("rollout", {}).get("speeds", []), dtype=np.float32).reshape(-1)
    selected_mean_speed = float(np.mean(speeds)) if len(speeds) else 0.0
    selected_start = float(metrics.get("selected_start_speed_mps", speeds[0] if len(speeds) else 0.0))
    selected_end = float(metrics.get("selected_end_speed_mps", speeds[-1] if len(speeds) else selected_start))
    selected_delta = float(metrics.get("selected_speed_delta_mps", selected_end - selected_start))
    signed_speed_effect = float(metrics.get("mean_signed_speed_change_mps", 0.0))
    terminal_speed_effect = float(metrics.get("terminal_speed_change_mps", 0.0))
    physical_effect = float(metrics.get("response_distance", 0.0))

    if selected_mean_speed <= stationary_mean_threshold:
        longitudinal_action = "stationary"
    elif selected_end <= near_stop_end_threshold and selected_delta < -0.20:
        longitudinal_action = "approach_stop"
    elif selected_delta <= -actual_trend_threshold:
        longitudinal_action = "decelerate"
    elif selected_delta >= actual_trend_threshold and signed_speed_effect <= -speed_ref_threshold:
        longitudinal_action = "limit_acceleration"
    elif signed_speed_effect <= -speed_ref_threshold:
        longitudinal_action = "slower_than_reference"
    elif signed_speed_effect >= speed_ref_threshold:
        longitudinal_action = "faster_than_reference"
    elif selected_delta >= actual_trend_threshold:
        longitudinal_action = "accelerate"
    else:
        longitudinal_action = "maintain_speed"

    if physical_effect < negligible_distance:
        magnitude = "negligible"
    elif physical_effect < slight_distance:
        magnitude = "slight"
    elif physical_effect < moderate_distance:
        magnitude = "moderate"
    else:
        magnitude = "strong"

    return {
        "longitudinal_action": longitudinal_action,
        "effect_magnitude": magnitude,
        "physical_effect_score": physical_effect,
        "selected_mean_speed_mps": selected_mean_speed,
        "selected_start_speed_mps": selected_start,
        "selected_end_speed_mps": selected_end,
        "selected_speed_delta_mps": selected_delta,
        "mean_signed_speed_change_vs_reference_mps": signed_speed_effect,
        "terminal_speed_change_vs_reference_mps": terminal_speed_effect,
        "reference_speed_delta_mps": float(metrics.get("reference_speed_delta_mps", 0.0)),
    }


def build_response_supervision(selected: Dict, reference_rollout: Dict, causal_analysis: Dict, cfg) -> Dict:
    metrics = trajectory_response_metrics(selected.get("rollout", {}), reference_rollout, cfg=cfg)
    info = selected.get("info", {})
    action_effect = _action_effect_semantics(selected, reference_rollout, metrics, cfg)
    causal_object_present = bool(causal_analysis.get("has_causal_object", False))
    red_indices = info.get("red_light_active_temporal_indices", []) or []
    red_rule_response = bool(red_indices) and str(info.get("intent_name", "")) in [
        "cautious_follow", "yield_stop", "emergency_brake"
    ]
    stop_indices = info.get("stop_sign_active_temporal_indices", []) or []
    stop_rule_response = bool(stop_indices) and str(info.get("intent_name", "")) in [
        "cautious_follow", "yield_stop", "emergency_brake"
    ]
    traffic_rule_response = bool(red_rule_response or stop_rule_response)
    traffic_rule_type = None
    if red_rule_response:
        traffic_rule_type = "red_light_stop_line"
    elif stop_rule_response:
        traffic_rule_type = "stop_sign_control"

    necessary = bool(causal_object_present or traffic_rule_response)
    sufficient = bool(info.get("allowed", False))
    boundary = dict(info.get("minimum_response_boundary", {}) or {})
    if bool(boundary.get("applicable", False)):
        minimal = bool(
            sufficient
            and boundary.get("bracket_found", False)
            and boundary.get("converged", False)
        )
        minimality_mode = (
            "continuous_safety_boundary"
            if minimal
            else "continuous_boundary_not_proven"
        )
    else:
        minimal = sufficient and str(info.get("selection_override", "")) in [
            "minimum_necessary_response",
            "minimum_sufficient_response_boundary_checked",
            "minimum_sufficient_response_boundary_refined",
        ]
        minimality_mode = "discrete_or_structural_response"

    if traffic_rule_response and not causal_object_present:
        reference_source = "traffic_rule_constraint"
    else:
        reference_source = causal_analysis.get("reference_source", "nominal_no_interference")

    return {
        "reference_source": reference_source,
        "causal_object_present": causal_object_present,
        "traffic_rule_response": traffic_rule_response,
        "traffic_rule_type": traffic_rule_type,
        "causal_score": float(causal_analysis.get("causal_score", 0.0)),
        "quality": {
            "necessary": necessary,
            "sufficient": sufficient,
            "minimal": bool(minimal),
            "minimality_mode": minimality_mode,
        },
        "minimum_response_boundary": boundary if boundary else None,
        "trajectory_effect": metrics,
        "action_effect": action_effect,
        "reference_waypoints": np.asarray(reference_rollout.get("waypoints", []), dtype=np.float32).tolist(),
        "reference_speeds": np.asarray(reference_rollout.get("speeds", []), dtype=np.float32).tolist(),
    }


def compact_causal_analysis(causal_analysis: Dict) -> Dict:
    """Remove internal candidate objects before writing public JSON."""
    out = {
        "enabled": bool(causal_analysis.get("enabled", False)),
        "has_causal_object": bool(causal_analysis.get("has_causal_object", False)),
        "causal_object": causal_analysis.get("causal_object", {"exists": False}),
        "causal_score": float(causal_analysis.get("causal_score", 0.0)),
        "preliminary_causal_score": float(causal_analysis.get("preliminary_causal_score", 0.0)),
        "final_causal_score": causal_analysis.get("final_causal_score", None),
        "final_revalidation_passed": causal_analysis.get("final_revalidation_passed", None),
        "causal_acceptance_reason": causal_analysis.get("causal_acceptance_reason", None),
        "reference_source": causal_analysis.get("reference_source", "nominal_no_interference"),
        "counterfactual_intent_name": causal_analysis.get("counterfactual_intent_name", None),
        "counterfactual_selected_variant": causal_analysis.get("counterfactual_selected_variant", None),
        "counterfactual_candidate_count": int(causal_analysis.get("counterfactual_candidate_count", 0)),
        "object_tests": causal_analysis.get("object_tests", []),
    }
    if causal_analysis.get("rejected_causal_object", None):
        out["rejected_causal_object"] = causal_analysis.get("rejected_causal_object")
    return out
