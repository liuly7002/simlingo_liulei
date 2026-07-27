# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np


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


def _cfg_bool(obj: Any, key: str, default: bool) -> bool:
    value = _cfg_get(obj, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _cfg_float(obj: Any, key: str, default: float) -> float:
    try:
        return float(_cfg_get(obj, key, default))
    except Exception:
        return float(default)


def actor_id(actor: Optional[Dict]) -> Optional[str]:
    if not isinstance(actor, dict):
        return None
    value = actor.get("id", None)
    return None if value is None else str(value)


def same_actor(a: Optional[Dict], b: Optional[Dict]) -> bool:
    aid, bid = actor_id(a), actor_id(b)
    if aid is not None and bid is not None:
        return aid == bid
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if str(a.get("class", "")) != str(b.get("class", "")):
        return False
    try:
        dx = float(a.get("x_m", 0.0)) - float(b.get("x_m", 0.0))
        dy = float(a.get("y_m", 0.0)) - float(b.get("y_m", 0.0))
    except Exception:
        return False
    return math.hypot(dx, dy) <= 1.5


def expert_speed_profile(expert_future: np.ndarray, future_fps: float) -> np.ndarray:
    points = np.asarray(expert_future, dtype=np.float32)
    if points.ndim != 2 or len(points) == 0 or points.shape[1] < 2:
        return np.zeros((0,), dtype=np.float32)
    points = points[:, :2]
    origin = np.zeros((1, 2), dtype=np.float32)
    steps = np.linalg.norm(np.diff(np.vstack([origin, points]), axis=0), axis=1)
    return (steps * max(float(future_fps), 1e-6)).astype(np.float32)


def _candidate_match_metrics(
    candidate: Dict,
    expert_future: np.ndarray,
    future_fps: float,
    cfg,
) -> Optional[Dict[str, float]]:
    rollout = candidate.get("rollout", {}) or {}
    candidate_wp = np.asarray(rollout.get("waypoints", []), dtype=np.float32)
    expert_wp = np.asarray(expert_future, dtype=np.float32)
    if (
        candidate_wp.ndim != 2
        or expert_wp.ndim != 2
        or candidate_wp.shape[1] < 2
        or expert_wp.shape[1] < 2
    ):
        return None

    count = min(len(candidate_wp), len(expert_wp))
    if count <= 0:
        return None
    candidate_wp = candidate_wp[:count, :2]
    expert_wp = expert_wp[:count, :2]
    if not np.isfinite(candidate_wp).all() or not np.isfinite(expert_wp).all():
        return None

    displacement = np.linalg.norm(candidate_wp - expert_wp, axis=1)
    ade = float(np.mean(displacement))
    fde = float(displacement[-1])

    candidate_speed = np.asarray(rollout.get("speeds", []), dtype=np.float32).reshape(-1)
    expert_speed = expert_speed_profile(expert_wp, future_fps)
    speed_count = min(len(candidate_speed), len(expert_speed))
    if speed_count > 0 and np.isfinite(candidate_speed[:speed_count]).all():
        speed_mae = float(
            np.mean(np.abs(candidate_speed[:speed_count] - expert_speed[:speed_count]))
        )
    else:
        speed_mae = 0.0

    match_cfg = _cfg_get(_cfg_get(cfg, "structured_world", {}), "expert_match", {})
    score = (
        _cfg_float(match_cfg, "ade_weight", 1.0) * ade
        + _cfg_float(match_cfg, "fde_weight", 0.25) * fde
        + _cfg_float(match_cfg, "speed_weight", 0.10) * speed_mae
    )
    return {
        "ade_m": ade,
        "fde_m": fde,
        "speed_mae_mps": speed_mae,
        "match_score": float(score),
        "compared_waypoint_count": int(count),
    }


def select_expert_matched_response(
    scored_candidates: Sequence[Dict],
    expert_future: np.ndarray,
    cfg,
) -> Dict:
    match_cfg = _cfg_get(_cfg_get(cfg, "structured_world", {}), "expert_match", {})
    require_allowed = _cfg_bool(match_cfg, "require_allowed_candidate", True)
    future_fps = float(cfg.horizon.future_fps)

    ranked: List[Tuple[int, Dict[str, float], Dict]] = []
    for index, candidate in enumerate(scored_candidates):
        if require_allowed and not bool(candidate.get("info", {}).get("allowed", False)):
            continue
        metrics = _candidate_match_metrics(candidate, expert_future, future_fps, cfg)
        if metrics is None:
            continue
        ranked.append((index, metrics, candidate))

    if not ranked:
        return {
            "valid": False,
            "reason": "no_expert_match_candidate",
            "selected_index": -1,
            "candidate_count": int(len(scored_candidates)),
            "eligible_candidate_count": 0,
        }

    ranked.sort(
        key=lambda item: (
            float(item[1]["match_score"]),
            float(item[1]["ade_m"]),
            float(item[1]["fde_m"]),
            int(item[0]),
        )
    )
    index, metrics, candidate = ranked[0]

    max_ade = _cfg_float(match_cfg, "max_ade_m", float("inf"))
    max_fde = _cfg_float(match_cfg, "max_fde_m", float("inf"))
    max_speed = _cfg_float(match_cfg, "max_speed_mae_mps", float("inf"))
    threshold_passed = (
        metrics["ade_m"] <= max_ade
        and metrics["fde_m"] <= max_fde
        and metrics["speed_mae_mps"] <= max_speed
    )

    info = candidate.get("info", {}) or {}
    result = {
        "valid": bool(threshold_passed),
        "reason": "ok" if threshold_passed else "expert_match_threshold_failed",
        "selected_index": int(index),
        "candidate_count": int(len(scored_candidates)),
        "eligible_candidate_count": int(len(ranked)),
        "intent_name": str(info.get("intent_name", candidate.get("intent_name", "unknown"))),
        "variant_id": str(info.get("variant_id", candidate.get("variant_id", "default"))),
        "candidate_allowed": bool(info.get("allowed", False)),
        **metrics,
        "thresholds": {
            "max_ade_m": float(max_ade),
            "max_fde_m": float(max_fde),
            "max_speed_mae_mps": float(max_speed),
        },
    }
    return result


def _test_score(test: Dict) -> float:
    for key in ("final_causal_score", "causal_score", "preliminary_causal_score"):
        value = test.get(key, None)
        if value is None:
            continue
        try:
            score = float(value)
            if math.isfinite(score):
                return score
        except Exception:
            continue
    return 0.0


def _effect_dict(test: Optional[Dict]) -> Dict:
    if not isinstance(test, dict):
        return {}
    # Compare object tests at the same preliminary stage. The primary test may
    # additionally contain final_effect after revalidation, while non-primary
    # tests do not; preferring effect avoids a stage mismatch.
    effect = test.get("effect")
    if isinstance(effect, dict):
        return effect
    final_effect = test.get("final_effect")
    return final_effect if isinstance(final_effect, dict) else {}


def _effect_l1_distance(a: Dict, b: Dict) -> float:
    keys = (
        "mean_abs_longitudinal_change_m",
        "mean_signed_longitudinal_change_m",
        "mean_abs_lateral_change_m",
        "mean_signed_lateral_change_m",
        "max_abs_lateral_change_m",
        "terminal_longitudinal_change_m",
        "terminal_lateral_change_m",
        "mean_abs_speed_change_mps",
        "mean_signed_speed_change_mps",
        "terminal_speed_change_mps",
    )
    total = 0.0
    for key in keys:
        try:
            av = float(a.get(key, 0.0))
        except Exception:
            av = 0.0
        try:
            bv = float(b.get(key, 0.0))
        except Exception:
            bv = 0.0
        av = av if math.isfinite(av) else 0.0
        bv = bv if math.isfinite(bv) else 0.0
        total += abs(av - bv)
    return float(total)


def _find_actor(records: Sequence[Dict], target: Optional[Dict]) -> Optional[Dict]:
    if target is None:
        return None
    for actor in records or []:
        if same_actor(actor, target):
            return actor
    return None


def _future_support(
    actor: Dict,
    expert_future: np.ndarray,
    actor_timelines: Dict[int, List[Dict]],
) -> Dict[str, float]:
    expert = np.asarray(expert_future, dtype=np.float32)
    if expert.ndim != 2 or expert.shape[1] < 2:
        expert = np.zeros((0, 2), dtype=np.float32)
    else:
        expert = expert[:, :2]

    distances: List[float] = []
    for step in range(1, len(expert) + 1):
        future_actor = _find_actor(actor_timelines.get(step, []), actor)
        if future_actor is None:
            continue
        try:
            actor_xy = np.asarray(
                [float(future_actor.get("x_m", 0.0)),
                 float(future_actor.get("y_m", 0.0))],
                dtype=np.float32,
            )
        except Exception:
            continue
        if np.isfinite(actor_xy).all():
            distances.append(float(np.linalg.norm(actor_xy - expert[step - 1])))

    return {
        "future_frames_found": int(len(distances)),
        "min_time_aligned_center_distance_m": (
            float(min(distances)) if distances else float("inf")
        ),
        "mean_time_aligned_center_distance_m": (
            float(np.mean(distances)) if distances else float("inf")
        ),
    }


def _find_primary_test(
    tests: Sequence[Dict],
    primary_actor: Optional[Dict],
) -> Optional[Dict]:
    for test in tests or []:
        if same_actor(test.get("actor", {}) or {}, primary_actor):
            return test
    return None


def select_secondary_actor(
    causal_analysis: Dict,
    primary_actor: Optional[Dict],
    primary_score: float,
    expert_future: np.ndarray,
    actor_timelines: Dict[int, List[Dict]],
    cfg,
) -> Tuple[Optional[Dict], str, float, Dict]:
    """Strictly select C4 as an independent expert-conditioned actor.

    A candidate must have an accepted counterfactual effect, enough future
    observations, sufficient time-aligned proximity to the expert trajectory,
    and a response distinguishable from the primary actor's response.
    """
    cfg_obj = _cfg_get(
        _cfg_get(cfg, "structured_world", {}),
        "secondary_actor",
        {},
    )
    enabled = _cfg_bool(cfg_obj, "enabled", True)
    diagnostics: Dict[str, Any] = {
        "enabled": enabled,
        "selection_policy": "strict_independent_marginal_counterfactual_effect",
        "selected": False,
        "candidate_test_count": 0,
        "eligible_candidate_count": 0,
        "rejection_counts": {},
        "candidate_diagnostics": [],
    }
    if not enabled:
        diagnostics["reason"] = "disabled"
        return None, "disabled", 0.0, diagnostics
    if primary_actor is None:
        diagnostics["reason"] = "no_primary_actor"
        return None, "none", 0.0, diagnostics

    require_accepted = _cfg_bool(cfg_obj, "require_causal_accepted", True)
    fallback_valid = _cfg_bool(
        cfg_obj, "fallback_to_valid_counterfactual_test", False
    )
    min_score = max(_cfg_float(cfg_obj, "min_causal_score", 0.60), 0.0)
    min_ratio = max(
        _cfg_float(cfg_obj, "min_score_ratio_to_primary", 0.35), 0.0
    )
    min_future_frames = max(
        int(_cfg_get(cfg_obj, "min_future_frames_found", 5)), 1
    )
    max_distance = max(
        _cfg_float(cfg_obj, "max_time_aligned_center_distance_m", 10.0),
        0.0,
    )
    reject_duplicate = _cfg_bool(
        cfg_obj, "reject_duplicate_primary_response", True
    )
    reject_ambiguous_groups = _cfg_bool(
        cfg_obj, "reject_ambiguous_secondary_response_groups", True
    )
    duplicate_tolerance = max(
        _cfg_float(cfg_obj, "duplicate_effect_l1_tolerance", 1e-4),
        0.0,
    )

    tests = (
        causal_analysis.get("object_tests", [])
        if isinstance(causal_analysis, dict)
        else []
    )
    diagnostics["candidate_test_count"] = len(tests or [])
    primary_test = _find_primary_test(tests, primary_actor)
    primary_effect = _effect_dict(primary_test)
    primary_intent = str(
        (primary_test or {}).get("counterfactual_intent_name", "")
    )
    primary_variant = str(
        (primary_test or {}).get("counterfactual_selected_variant", "")
    )
    primary_score = max(float(primary_score), 0.0)

    def register_rejection(reason: str) -> None:
        counts = diagnostics["rejection_counts"]
        counts[reason] = int(counts.get(reason, 0)) + 1

    ranked = []
    for test in tests or []:
        actor = test.get("actor", {}) or {}
        score = _test_score(test)
        support = _future_support(actor, expert_future, actor_timelines)
        reason = ""

        if not bool(test.get("counterfactual_valid", False)):
            reason = "counterfactual_invalid"
        elif not actor or not actor.get("exists", True):
            reason = "missing_actor"
        elif primary_actor is not None and same_actor(actor, primary_actor):
            reason = "is_primary_actor"
        elif require_accepted and not bool(test.get("causal_accepted", False)):
            reason = "causal_not_accepted"
        elif score < min_score:
            reason = "causal_score_below_secondary_threshold"
        elif primary_score > 1e-8 and score < primary_score * min_ratio:
            reason = "causal_score_ratio_below_primary"
        elif support["future_frames_found"] < min_future_frames:
            reason = "insufficient_future_actor_frames"
        elif support["min_time_aligned_center_distance_m"] > max_distance:
            reason = "too_far_from_time_aligned_expert_trajectory"

        effect_distance = float("nan")
        duplicate = False
        if not reason and reject_duplicate and primary_test is not None:
            effect_distance = _effect_l1_distance(
                _effect_dict(test), primary_effect
            )
            duplicate = (
                str(test.get("counterfactual_intent_name", "")) == primary_intent
                and str(test.get("counterfactual_selected_variant", "")) == primary_variant
                and effect_distance <= duplicate_tolerance
            )
            if duplicate:
                reason = "duplicate_primary_counterfactual_response"

        item = {
            "actor_id": actor_id(actor) or "",
            "counterfactual_valid": bool(test.get("counterfactual_valid", False)),
            "causal_accepted": bool(test.get("causal_accepted", False)),
            "causal_score": float(score),
            "future_frames_found": int(support["future_frames_found"]),
            "min_time_aligned_center_distance_m": (
                float(support["min_time_aligned_center_distance_m"])
                if math.isfinite(support["min_time_aligned_center_distance_m"])
                else None
            ),
            "mean_time_aligned_center_distance_m": (
                float(support["mean_time_aligned_center_distance_m"])
                if math.isfinite(support["mean_time_aligned_center_distance_m"])
                else None
            ),
            "effect_l1_distance_to_primary": (
                float(effect_distance) if math.isfinite(effect_distance) else None
            ),
            "duplicate_primary_response": duplicate,
            "accepted": not bool(reason),
            "rejection_reason": reason,
        }
        diagnostics["candidate_diagnostics"].append(item)
        if reason:
            register_rejection(reason)
            continue

        ranked.append(
            (
                -score,
                support["min_time_aligned_center_distance_m"],
                -support["future_frames_found"],
                str(actor_id(actor) or ""),
                actor,
                item,
                test,
            )
        )

    # Compatibility fallback is disabled by default. Even when enabled, it still
    # obeys the strict score, future-support, and proximity thresholds.
    if not ranked and require_accepted and fallback_valid:
        diagnostics["fallback_attempted"] = True
        for test in tests or []:
            actor = test.get("actor", {}) or {}
            if not bool(test.get("counterfactual_valid", False)):
                continue
            if not actor or (
                primary_actor is not None and same_actor(actor, primary_actor)
            ):
                continue
            score = _test_score(test)
            support = _future_support(actor, expert_future, actor_timelines)
            if score < min_score:
                continue
            if primary_score > 1e-8 and score < primary_score * min_ratio:
                continue
            if support["future_frames_found"] < min_future_frames:
                continue
            if support["min_time_aligned_center_distance_m"] > max_distance:
                continue
            item = {
                "actor_id": actor_id(actor) or "",
                "causal_score": float(score),
                "future_frames_found": int(support["future_frames_found"]),
                "min_time_aligned_center_distance_m": float(
                    support["min_time_aligned_center_distance_m"]
                ),
                "mean_time_aligned_center_distance_m": float(
                    support["mean_time_aligned_center_distance_m"]
                ),
                "accepted": True,
                "fallback": True,
            }
            ranked.append(
                (
                    -score,
                    support["min_time_aligned_center_distance_m"],
                    -support["future_frames_found"],
                    str(actor_id(actor) or ""),
                    actor,
                    item,
                    test,
                )
            )

    #修改20260727：若多个非主要actor产生相同的反事实意图、候选变体和
    # 作用向量，则无法从当前反事实证据中唯一判断哪个actor应作为C4。
    # 严格模式下拒绝整个等价响应组，避免按遍历顺序任意选择一个对象。
    if reject_ambiguous_groups and len(ranked) > 1:
        parent = list(range(len(ranked)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(ranked)):
            left_test = ranked[left][6]
            for right in range(left + 1, len(ranked)):
                right_test = ranked[right][6]
                if (
                    str(left_test.get("counterfactual_intent_name", ""))
                    == str(right_test.get("counterfactual_intent_name", ""))
                    and str(left_test.get("counterfactual_selected_variant", ""))
                    == str(right_test.get("counterfactual_selected_variant", ""))
                    and _effect_l1_distance(
                        _effect_dict(left_test),
                        _effect_dict(right_test),
                    ) <= duplicate_tolerance
                ):
                    union(left, right)

        groups: Dict[int, List[int]] = {}
        for index in range(len(ranked)):
            groups.setdefault(find(index), []).append(index)

        ambiguous_indices = {
            index
            for indices in groups.values()
            if len(indices) > 1
            for index in indices
        }
        if ambiguous_indices:
            diagnostics["ambiguous_response_groups"] = [
                [str(ranked[index][3]) for index in indices]
                for indices in groups.values()
                if len(indices) > 1
            ]
            filtered_ranked = []
            for index, record in enumerate(ranked):
                if index not in ambiguous_indices:
                    filtered_ranked.append(record)
                    continue
                item = record[5]
                item["accepted"] = False
                item["rejection_reason"] = (
                    "ambiguous_duplicate_secondary_response_group"
                )
                register_rejection(
                    "ambiguous_duplicate_secondary_response_group"
                )
            ranked = filtered_ranked

    diagnostics["eligible_candidate_count"] = len(ranked)
    diagnostics["thresholds"] = {
        "require_causal_accepted": require_accepted,
        "min_causal_score": min_score,
        "min_score_ratio_to_primary": min_ratio,
        "min_future_frames_found": min_future_frames,
        "max_time_aligned_center_distance_m": max_distance,
        "reject_duplicate_primary_response": reject_duplicate,
        "reject_ambiguous_secondary_response_groups": reject_ambiguous_groups,
        "duplicate_effect_l1_tolerance": duplicate_tolerance,
    }

    if not ranked:
        diagnostics["reason"] = "no_strict_secondary_actor"
        return None, "none", 0.0, diagnostics

    ranked.sort(key=lambda item: item[:4])
    score = -float(ranked[0][0])
    actor = dict(ranked[0][4])
    actor["exists"] = True
    actor["causal_score"] = score
    diagnostics.update(
        {
            "selected": True,
            "reason": "ok",
            "selected_actor_id": actor_id(actor) or "",
            "selected_score": score,
            "selected_metrics": ranked[0][5],
        }
    )
    return (
        actor,
        "expert_conditioned_strict_independent_secondary_actor",
        score,
        diagnostics,
    )
