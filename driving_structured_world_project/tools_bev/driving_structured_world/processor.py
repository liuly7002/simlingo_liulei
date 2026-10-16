# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np

from .config_loader import bootstrap_lg_tools

bootstrap_lg_tools()

from lg_waypoint_planner.actor_loader import load_current_actor_records, load_future_actor_timelines
from lg_waypoint_planner.causal_response import compact_causal_analysis
#修改20260728：LG与普通Driving共享同一套主要关键actor六视角投影实现。
from lg_waypoint_planner.camera_attention_target import (
    build_camera_attention_supervision,
)
from lg_waypoint_planner.costmap import (
    build_temporal_costmaps,
    build_temporal_red_light_constraints,
    build_temporal_stop_sign_constraints,
    build_traffic_light_state_context,
)
from lg_waypoint_planner.critical_factor import align_factor_with_selected_waypoints
from lg_waypoint_planner.dataset import (
    get_ego_center,
    get_meters_per_pixel,
    get_route,
    load_future_ego_waypoints,
    offset_frame_name,
)
from lg_waypoint_planner.io_utils import (
    find_route_dirs,
    list_frame_names,
    load_costmap,
    load_json_gz,
    save_json_gz,
)
from lg_waypoint_planner.logger import LOGGER
from lg_waypoint_planner.language import build_language_annotation
from lg_waypoint_planner.processor import (
    _build_core_questions,
    _build_reference_route_from_measurements,
    _required_reference_horizon_m,
    build_fallback,
)

from .expert_conditioned_selection import actor_id
from .grid_builder import (
    build_grid,
    save_debug_image,
    save_npz,
    training_equal_spacing_route,
)


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


def _normalize_frame_name(frame_name: str) -> str:
    return str(frame_name).replace(".json.gz", "").replace(".npy", "").replace(".npz", "")


def _load_lane_constraint(route_dir: Path, frame_name: str, costmap: np.ndarray, cfg):
    lane_cfg = _cfg_get(cfg, "lane_constraints", {})
    enabled = _cfg_bool(lane_cfg, "enabled", True)
    require_map = _cfg_bool(lane_cfg, "require_map", True)
    folder = str(_cfg_get(cfg.paths, "lane_constraint_folder", "lane_constraints"))
    path = route_dir / folder / f"{frame_name}.npy"
    if enabled and require_map and not path.exists():
        raise FileNotFoundError(f"Missing solid-lane constraint map: {path}")
    if enabled and path.exists():
        lane = load_costmap(path)
        if lane.shape[:2] != costmap.shape[:2]:
            raise ValueError(
                f"Lane constraint shape {lane.shape} does not match costmap {costmap.shape}."
            )
        return lane.astype(np.float32), path
    return np.zeros_like(costmap, dtype=np.float32), path


def _build_temporal_bundle(route_dir, frame_name, measurement, costmap, meta, lane_map, lane_path, cfg):
    bundle = build_temporal_costmaps(route_dir, frame_name, measurement, costmap, meta, cfg)
    red = build_temporal_red_light_constraints(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        current_shape=costmap.shape,
        current_meta=meta,
        cfg=cfg,
    )
    bundle["red_light_maps"] = red.get("maps", [])
    bundle["red_light_frames"] = red.get("frames", [])
    bundle["red_light_valid"] = red.get("valid", [])
    bundle["red_light_missing_reasons"] = red.get("missing_reasons", [])
    bundle["traffic_light_state"] = build_traffic_light_state_context(
        route_dir=route_dir,
        frame_name=frame_name,
        current_shape=costmap.shape,
        cfg=cfg,
    )
    stop = build_temporal_stop_sign_constraints(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        current_shape=costmap.shape,
        current_meta=meta,
        cfg=cfg,
    )
    bundle["stop_sign_maps"] = stop.get("maps", [])
    bundle["stop_sign_frames"] = stop.get("frames", [])
    bundle["stop_sign_valid"] = stop.get("valid", [])
    bundle["stop_sign_missing_reasons"] = stop.get("missing_reasons", [])
    bundle["solid_lane_constraint"] = lane_map.astype(np.float32)
    bundle["solid_lane_constraint_path"] = str(lane_path)
    return bundle


def _empty_causal_analysis(nominal_rollout: Dict, enabled: bool = True, reason: str = "") -> Dict:
    return {
        "enabled": bool(enabled),
        "has_causal_object": False,
        "causal_object": {"exists": False},
        "causal_score": 0.0,
        "preliminary_causal_score": 0.0,
        "final_causal_score": None,
        "final_revalidation_passed": None,
        "reference_rollout": nominal_rollout,
        "reference_source": "nominal_no_interference",
        "counterfactual_selected": None,
        "counterfactual_candidate_count": 0,
        "object_tests": [],
        "driving_selection_reason": str(reason),
    }


def _finite_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _load_expert_future_relative_yaws(
    route_dir: Path,
    frame_name: str,
    current_measurement: Dict,
    cfg,
) -> np.ndarray:
    """Load real expert body yaw for the same future frames as C1 waypoints.

    Each future global ``theta`` is converted into the current ego frame.  Missing
    or invalid entries are stored as NaN so ``grid_builder`` can use its robust
    displacement-based fallback only for those individual frames.
    """
    horizon = int(cfg.horizon.num_future_waypoints)
    stride = int(cfg.horizon.future_frame_stride)
    result = np.full((horizon,), np.nan, dtype=np.float32)

    try:
        current_theta = float(current_measurement["theta"])
    except Exception:
        return result
    if not math.isfinite(current_theta):
        return result

    measurement_dir = route_dir / cfg.paths.measurements_folder
    for step in range(1, horizon + 1):
        future_name = offset_frame_name(frame_name, step, stride)
        if future_name is None:
            break
        future_path = measurement_dir / f"{future_name}.json.gz"
        if not future_path.exists():
            break

        future_measurement = load_json_gz(future_path)
        try:
            future_theta = float(future_measurement["theta"])
        except Exception:
            continue
        if not math.isfinite(future_theta):
            continue

        relative_yaw = future_theta - current_theta
        result[step - 1] = math.atan2(
            math.sin(relative_yaw),
            math.cos(relative_yaw),
        )

    return result


def process_one_frame(route_dir: Path, frame_name: str, cfg) -> bool:
    frame_name = _normalize_frame_name(frame_name)
    measurement_path = route_dir / cfg.paths.measurements_folder / f"{frame_name}.json.gz"
    costmap_path = route_dir / cfg.paths.costmap_folder / f"{frame_name}.npy"
    meta_path = route_dir / cfg.paths.bev_meta_folder / f"{frame_name}.json.gz"
    if not measurement_path.exists() or not costmap_path.exists():
        LOGGER.info(f"[Skip] missing measurement/costmap for {route_dir.name}/{frame_name}")
        return False

    measurement = load_json_gz(measurement_path)
    costmap = load_costmap(costmap_path)
    meta = load_json_gz(meta_path) if meta_path.exists() else {}
    meters_per_pixel = get_meters_per_pixel(meta, float(cfg.paths.default_pixels_per_meter))
    ego_center = get_ego_center(meta, costmap.shape)
    # LG身份直接复用，无需再加载专家条件重选所需的lane constraint。

    planning_horizon = _required_reference_horizon_m(measurement, cfg)
    planning_route, route_info = _build_reference_route_from_measurements(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        required_horizon_m=planning_horizon,
        cfg=cfg,
    )

    structured_cfg = _cfg_get(cfg, "structured_world", {})
    reference_count = int(_cfg_get(structured_cfg, "reference_route_points", 20))
    reference_spacing = float(_cfg_get(structured_cfg, "reference_route_spacing_m", 1.0))
    raw_expert_route = np.asarray(
        measurement.get(str(cfg.paths.route_key), measurement.get("route", measurement.get("route_original", []))),
        dtype=np.float32,
    )
    expert_label_route = training_equal_spacing_route(
        raw_expert_route,
        count=reference_count,
        spacing_m=reference_spacing,
    )
    expert_future = load_future_ego_waypoints(route_dir, frame_name, measurement, cfg)
    #修改20260727：读取未来专家车辆真实theta，并转换为当前自车坐标系下的相对yaw。
    expert_future_yaws = _load_expert_future_relative_yaws(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        cfg=cfg,
    )
    if expert_future is None:
        expert_future = np.zeros((0, 2), dtype=np.float32)
    expert_future = np.asarray(expert_future, dtype=np.float32)

    valid = True
    invalid_reason = ""
    if planning_route is None:
        valid = False
        invalid_reason = "insufficient_real_reference_route"
    if len(expert_label_route) != reference_count:
        valid = False
        invalid_reason = invalid_reason or "invalid_expert_reference_route"
    if (
        expert_future.ndim != 2
        or expert_future.shape[1] < 2
        or len(expert_future) != int(cfg.horizon.num_future_waypoints)
    ):
        valid = False
        invalid_reason = invalid_reason or "invalid_expert_future_waypoint_count"

    #修改20261016：LG标签决定关键Actor的身份，Driving不进行专家轨迹条件下的重新筛选。
    lg_path = route_dir / str(cfg.paths.lg_folder) / f"{frame_name}.json.gz"
    lg_payload = load_json_gz(lg_path) if lg_path.is_file() else None
    lg_analysis = (lg_payload or {}).get("causal_analysis", {}) or {}
    lg_supervision = (lg_payload or {}).get("supervision", {}) or {}
    lg_status = (lg_payload or {}).get("critical_actor_status", None)
    if lg_status not in (-1, 0, 1):
        has_causal = lg_analysis.get("has_causal_object", None)
        lg_status = (
            int(has_causal)
            if bool(lg_supervision.get("risk_label_valid", False)) and isinstance(has_causal, bool)
            else -1
        )
    # 与LG数据过滤准则一致：无效或回退的LG决策均标为-1，不作为负样本。
    if not bool(lg_supervision.get("risk_label_valid", False)) or str(
        lg_supervision.get("selected_internal_intent_name", "")
    ) in {"expert_fallback", "stationary_hold_fallback"}:
        lg_status = -1
    critical_actor_status = int(lg_status)
    lg_actor = lg_analysis.get("causal_object", {}) or {}
    primary_actor = dict(lg_actor) if critical_actor_status == 1 else None
    if primary_actor is not None and actor_id(primary_actor) is None:
        critical_actor_status = -1
        primary_actor = None
    primary_score = _finite_float(lg_analysis.get("final_causal_score", lg_analysis.get("causal_score", 0.0))) if primary_actor else 0.0
    actor_timelines = (
        load_future_actor_timelines(route_dir, frame_name, cfg)
        if primary_actor is not None else {}
    )
    if critical_actor_status == -1:
        valid = False
        invalid_reason = invalid_reason or "lg_critical_actor_label_invalid"

    # 为保持Q1/Q2与C2同源，直接复用LG的决策关键Actor语言答案。
    lg_questions = (lg_payload or {}).get("core_questions", {}) or {}
    language_annotation: Dict = {}
    core_questions: Dict = {}
    initial_factor: Dict = {}
    scored: List[Dict] = []
    causal_analysis = lg_analysis if isinstance(lg_analysis, dict) else {}

    # 普通Driving与LG共享同一套Q1-Q4自然语言生成逻辑。
    # Q1/Q2使用专家条件因果结果；Q3/Q4直接描述真实专家未来轨迹。
    if (
        expert_future.ndim == 2
        and expert_future.shape[1] >= 2
        and len(expert_future) == int(cfg.horizon.num_future_waypoints)
    ):
        language_route = (
            planning_route
            if planning_route is not None
            else expert_label_route
        )
        selected_for_language = build_fallback(
            language_route,
            expert_future,
            cfg,
            reason="driving_expert_language_supervision",
        )

        language_factor = {}
        language_annotation = build_language_annotation(
            frame_name,
            language_factor,
            selected_for_language,
            scored,
        )
        core_questions = _build_core_questions(language_annotation)
        if critical_actor_status != -1:
            for key, index in (("attention", 0), ("motion_constraint", 1)):
                qa = lg_questions.get(key)
                if not isinstance(qa, dict) or not qa.get("question") or not qa.get("answer"):
                    valid = False
                    invalid_reason = invalid_reason or "missing_lg_critical_actor_language"
                    break
                core_questions[key] = dict(qa)
                language_annotation["qa_pairs_en"][index] = dict(qa)
            if valid:
                answers = [item["answer"] for item in language_annotation["qa_pairs_en"]]
                language_annotation["summary_en"] = " ".join(answers)


    grid_result = build_grid(
        shape=tuple(costmap.shape[:2]),
        expert_route=expert_label_route,
        expert_future=expert_future,
        expert_future_yaws=expert_future_yaws,
        primary_actor=primary_actor,
        actor_timelines=actor_timelines,
        ego_center=ego_center,
        meters_per_pixel=meters_per_pixel,
        cfg=cfg,
    )

    if primary_actor is not None and grid_result["primary_actor_future_frames_found"] == 0:
        valid = False
        invalid_reason = invalid_reason or "causal_actor_missing_in_all_future_frames"

    grid = np.asarray(grid_result["grid"], dtype=np.float32)
    #修改20261016：Driving结构化世界仅包含C0、C1、C2；另存C1/C2逐帧占据。
    if grid.ndim != 3 or grid.shape[0] != 3:
        valid = False
        invalid_reason = invalid_reason or "invalid_grid_shape"
    if not np.isfinite(grid).all():
        valid = False
        invalid_reason = invalid_reason or "non_finite_grid"
    if np.any(grid < 0.0) or np.any(grid > 1.0):
        valid = False
        invalid_reason = invalid_reason or "grid_out_of_range"

    #修改20260728：将专家条件主要关键actor投影到统一六视角空间，
    # 与LG使用完全相同的相机顺序、投影、离轴补偿、平滑和归一化逻辑。
    if not valid:
        critical_actor_status = -1
    visual_grounding = build_camera_attention_supervision(
        route_dir=route_dir,
        frame_name=frame_name,
        cfg=cfg,
        primary_actor=(primary_actor if valid and critical_actor_status == 1 else None),
        target_source=(
            "driving_lg_critical_actor_projection"
        ),
    )
    if not valid:
        visual_grounding["invalid_reason"] = (
            "driving_label_invalid:"
            f"{invalid_reason or 'unknown'}"
        )

    output_cfg = _cfg_get(cfg, "output", {})
    save_npz_enabled = _cfg_bool(output_cfg, "save_npz", True)
    save_invalid = _cfg_bool(output_cfg, "save_invalid_labels", True)
    if save_npz_enabled and (valid or save_invalid):
        output_path = route_dir / str(cfg.paths.output_folder) / f"{frame_name}.npz"
        save_npz(
            path=output_path,
            grid_result=grid_result,
            valid=valid,
            invalid_reason=invalid_reason,
            frame_name=frame_name,
            primary_actor=primary_actor,
            primary_score=primary_score,
            critical_actor_status=critical_actor_status,
            reference_route_points_used=len(expert_label_route),
            meters_per_pixel=meters_per_pixel,
            ego_center=ego_center,
            cfg=cfg,
        )

    if _cfg_bool(output_cfg, "save_actor_selection_json", True):
        diagnostic_path = (
            route_dir
            / str(cfg.paths.actor_selection_output_folder)
            / f"{frame_name}.json.gz"
        )
        diagnostic = {
            "frame": frame_name,
            "generator": "driving_lg_critical_actor_structured_world_v3",
            "coordinate": "ego_local_x_forward_y_right_yaw_positive_right",
            "valid": bool(valid),
            "invalid_reason": str(invalid_reason),
            "trajectory_source": "expert",
            "actor_selection_source": "lg_decision_critical_actor",
            "critical_actor_status": critical_actor_status,
            # 普通Driving与LG使用相同的四问题键结构。
            "core_questions": core_questions,
            "language_annotation": language_annotation,
            #修改20260728：保存普通Driving主要关键actor的统一六视角显式监督。
            "visual_grounding": visual_grounding,
            "primary_actor": primary_actor if primary_actor is not None else {"exists": False},
            "primary_causal_score": float(primary_score),
            "structured_world_channels": [
                "C0_expert_route",
                "C1_expert_ego_future",
                "C2_lg_critical_actor_future",
            ],
            "removed_channel": "C3_time_aligned_primary_future_interaction",
            "causal_analysis": compact_causal_analysis(causal_analysis),
            "reference_route": {
                "label_point_count": int(len(expert_label_route)),
                "planning_required_horizon_m": float(route_info.get("required_horizon_m", 0.0)),
                "planning_route_coverage_complete": bool(route_info.get("coverage_complete", False)),
                "planning_stitched_route_length_m": float(route_info.get("stitched_route_length_m", 0.0)),
            },
            "future_frames": {
                "required": int(cfg.horizon.num_future_waypoints),
                "expert_found": int(len(expert_future)),
                "primary_actor_found": int(grid_result["primary_actor_future_frames_found"]),
            },
        }
        save_json_gz(diagnostic_path, diagnostic)

    debug_cfg = _cfg_get(cfg, "debug", {})
    if _cfg_bool(debug_cfg, "save_debug", False):
        debug_path = (
            route_dir
            / str(cfg.paths.debug_folder)
            / f"{frame_name}_future_interaction.png"
        )
        save_debug_image(debug_path, grid_result, ego_center, meters_per_pixel, cfg)

    if bool(cfg.run.verbose):
        LOGGER.info(
            f"[OK] {route_dir.name}/{frame_name}: valid={valid} "
            f"primary={actor_id(primary_actor) or 'none'} "
            f"primary_frames={grid_result['primary_actor_future_frames_found']}/"
            f"{int(cfg.horizon.num_future_waypoints)} "
            f"reason={invalid_reason or 'none'}"
        )
    return bool(valid or (save_npz_enabled and save_invalid))


def process_route_dir(route_dir: Path, cfg) -> Tuple[int, int]:
    frame_cfg = _cfg_get(_cfg_get(cfg, "run", {}), "frame", None)
    if frame_cfg not in [None, "", "None", "null"]:
        frames = [_normalize_frame_name(str(frame_cfg))]
    else:
        frames = list_frame_names(route_dir, cfg)

    ok = 0
    total = 0
    raise_on_error = _cfg_bool(_cfg_get(cfg, "run", {}), "raise_on_error", False)
    for frame_name in frames:
        total += 1
        try:
            if process_one_frame(route_dir, frame_name, cfg):
                ok += 1
        except Exception as exc:
            LOGGER.exception(f"[Error] route={route_dir}, frame={frame_name}, error={exc}")
            if raise_on_error:
                raise
    LOGGER.info(f"[Route Done] {route_dir}: {ok}/{total}")
    return ok, total


def process_dataset(cfg) -> Tuple[int, int]:
    input_path = _cfg_get(_cfg_get(cfg, "run", {}), "input", None)
    if input_path in [None, "", "None", "null"]:
        raise ValueError("Missing config field: run.input")
    root = Path(str(input_path)).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    recursive = _cfg_bool(_cfg_get(cfg, "run", {}), "recursive", False)
    route_dirs = find_route_dirs(root, cfg) if recursive else [root]
    LOGGER.info(f"[Info] Found {len(route_dirs)} route dirs; recursive={recursive}")
    total_ok, total_frames = 0, 0
    for route_dir in route_dirs:
        ok, total = process_route_dir(route_dir, cfg)
        total_ok += ok
        total_frames += total
    LOGGER.info(f"[All Done] {total_ok}/{total_frames} frames processed.")
    return total_ok, total_frames
