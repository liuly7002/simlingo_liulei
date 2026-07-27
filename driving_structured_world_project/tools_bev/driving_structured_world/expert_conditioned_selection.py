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
        try:
            score = float(value)
            if math.isfinite(score):
                return score
        except Exception:
            continue
    return 0.0


def select_secondary_actor(
    causal_analysis: Dict,
    primary_actor: Optional[Dict],
    cfg,
) -> Tuple[Optional[Dict], str, float]:
    secondary_cfg = _cfg_get(
        _cfg_get(_cfg_get(cfg, "structured_world", {}), "secondary_actor", {}),
        "enabled",
        True,
    )
    if isinstance(secondary_cfg, bool):
        enabled = secondary_cfg
        cfg_obj = _cfg_get(_cfg_get(cfg, "structured_world", {}), "secondary_actor", {})
    else:
        cfg_obj = _cfg_get(_cfg_get(cfg, "structured_world", {}), "secondary_actor", {})
        enabled = _cfg_bool(cfg_obj, "enabled", True)
    if not enabled:
        return None, "disabled", 0.0

    require_accepted = _cfg_bool(cfg_obj, "require_causal_accepted", True)
    fallback_valid = _cfg_bool(cfg_obj, "fallback_to_valid_counterfactual_test", False)
    tests = causal_analysis.get("object_tests", []) if isinstance(causal_analysis, dict) else []

    def collect(require_acceptance: bool) -> List[Tuple[float, Dict]]:
        result = []
        for test in tests or []:
            if not bool(test.get("counterfactual_valid", False)):
                continue
            if require_acceptance and not bool(test.get("causal_accepted", False)):
                continue
            actor = test.get("actor", {}) or {}
            if not actor or not actor.get("exists", True):
                continue
            if primary_actor is not None and same_actor(actor, primary_actor):
                continue
            result.append((_test_score(test), actor))
        result.sort(key=lambda item: (-float(item[0]), str(actor_id(item[1]) or "")))
        return result

    ranked = collect(require_accepted)
    source = "expert_conditioned_second_accepted_causal_actor"
    if not ranked and require_accepted and fallback_valid:
        ranked = collect(False)
        source = "expert_conditioned_highest_valid_non_primary_actor"
    if not ranked:
        return None, "none", 0.0

    score, actor = ranked[0]
    selected = dict(actor)
    selected["exists"] = True
    selected["causal_score"] = float(score)
    return selected, source, float(score)
