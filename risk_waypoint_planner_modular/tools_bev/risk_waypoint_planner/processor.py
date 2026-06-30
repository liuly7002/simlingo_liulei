# -*- coding: utf-8 -*-

from .common import *
from .io_utils import load_json_gz, save_json_gz, load_costmap, points_to_list, array_to_list
from .measurement_utils import get_meters_per_pixel, get_ego_center, route_from_measurement, load_future_ego_waypoints_from_measurements
from .geometry_utils import resample_polyline
from .temporal_costmap import build_temporal_costmap_bundle
from .behavior_candidates import (BEHAVIOR_NAMES, diagnose_scene_risk, build_legacy_candidates, build_behavior_candidates)
from .bicycle_rollout import rollout_candidate_with_bicycle_model
from .scoring import score_rollout, select_best_rollout, build_expert_fallback_rollout, build_valid_behavior_mask
from .qa_annotation import safe_float, behavior_name_to_chinese, behavior_name_to_english, build_language_annotation
from .dreamer_annotation import build_dreamer_annotation
from .dynamic_context import build_key_dynamic_context
from .static_context import build_static_candidate_context
from .visualization import save_debug_image, save_rgb_waypoints_debug_image

def process_one_frame(route_dir: Path, frame_name: str, args) -> bool:
    
    # 1. measurements 
    measurement_path = route_dir / args.measurements_folder / f"{frame_name}.json.gz"

    # 2. costmap
    costmap_folder = args.costmap_folder_augmented if args.augmented else args.costmap_folder
    costmap_path = route_dir / costmap_folder / f"{frame_name}.npy"

    # 3. meta
    meta_folder = args.bev_meta_folder_augmented if args.augmented else args.bev_meta_folder
    meta_path = route_dir / meta_folder / f"{frame_name}.json.gz"

    # 4. Check existence of measurement and costmap files. If either is missing, skip this frame.
    if not measurement_path.exists():
        LOGGER.info(f"[Skip] measurement not found: {measurement_path}")
        return False
    if not costmap_path.exists():
        LOGGER.info(f"[Skip] costmap not found: {costmap_path}")
        return False

    # 5. Load data for the current frame.
    measurement = load_json_gz(measurement_path)
    costmap = load_costmap(costmap_path)
    meta = load_json_gz(meta_path) if meta_path.exists() else {}

    meters_per_pixel = get_meters_per_pixel(meta, args.default_pixels_per_meter)
    ego_center = get_ego_center(meta, costmap.shape)

    # 6. 构造多帧时序costmap bundle
    temporal_bundle = build_temporal_costmap_bundle(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        current_costmap=costmap,
        current_meta=meta,
        current_ego_center=ego_center,
        current_meters_per_pixel=meters_per_pixel,
        args=args,
    )

    # 7. 读取measurements中的route 并进行重采样，得到expert_reference_route
    expert_route_raw = route_from_measurement(measurement, args.route_key)
    expert_reference_route = resample_polyline(
        expert_route_raw,
        spacing_m=args.reference_spacing_m,  # 采样间距
        horizon_m=args.reference_horizon_m,  # 采样范围
    )

    # 8. 场景风险诊断, 构建scene_context
    scene_context = diagnose_scene_risk(
        expert_reference_route=expert_reference_route,
        temporal_bundle=temporal_bundle,
        ego_center=ego_center,
        meters_per_pixel=meters_per_pixel,
        measurement=measurement,
        args=args,
    )

    key_dynamic_context = build_key_dynamic_context(
        route_dir=route_dir,
        frame_name=frame_name,
        measurement=measurement,
        args=args,
    )

    # 9. 生成候选行为
    if args.disable_multibehavior:
        behavior_candidates = build_legacy_candidates(expert_route_raw, measurement, args)
    else:
        behavior_candidates = build_behavior_candidates(
            expert_reference_route=expert_reference_route,
            scene_context=scene_context,
            measurement=measurement,
            args=args,
        )

    # 10. 对每个候选行为, 使用自行车模型rollout,然后基于costmap进行评估打分
    scored_rollouts = []

    for beh_cand in behavior_candidates:
        speed_prof = beh_cand["speed_profile"]

        rollout = rollout_candidate_with_bicycle_model(
            reference_route=beh_cand["reference_route"],
            speed_profile=speed_prof,
            measurement=measurement,
            args=args,
        )
        rollout["speed_mode"] = speed_prof["speed_mode"]

        info = score_rollout(
            rollout=rollout,
            reference_candidate=beh_cand,
            expert_reference_route=expert_reference_route,
            costmap=costmap,
            ego_center=ego_center,
            meters_per_pixel=meters_per_pixel,
            args=args,
            temporal_costmaps=temporal_bundle["costmaps"],
            temporal_frames=temporal_bundle["frames"],
        )

        scored_rollouts.append({
            "info": info,
            "reference_route": beh_cand["reference_route"],
            "rollout": rollout,
        })

    if len(scored_rollouts) == 0:
        raise RuntimeError(f"No rollouts generated for route={route_dir}, frame={frame_name}")

    expert_future_from_measurements = load_future_ego_waypoints_from_measurements(
        route_dir=route_dir,
        frame_name=frame_name,
        current_measurement=measurement,
        args=args,
    )

    # 11. 从评估打分的结果中选择一个最优的行为作为最终输出. 如果没有任何候选行为满足可行性约束,则构建一个专家经验回退方案
    selected_idx = select_best_rollout(scored_rollouts)

    if selected_idx < 0:
        selected = build_expert_fallback_rollout(
            expert_reference_route=expert_reference_route,
            expert_future_waypoints=expert_future_from_measurements,
            args=args,
            fallback_reason="no_allowed_behavior_candidate",
        )

        # Append fallback as a pseudo-candidate so that debug image and JSON can
        # consistently mark it as selected.
        scored_rollouts.append(selected)
        selected_idx = len(scored_rollouts) - 1
    else:
        selected = scored_rollouts[selected_idx]


    # 12. 构建输出JSON的语言描述字段, 包含中英文版本. 该字段主要用于后续VLA的语言监督, 以及人工检查时的辅助理解.
    candidate_rollouts_json = []
    for i, r in enumerate(scored_rollouts):
        roll = r["rollout"]
        info = r["info"]
        behavior_name = str(info.get("behavior_name", "unknown"))
        behavior_zh = behavior_name_to_chinese(behavior_name)
        allowed = bool(info.get("allowed", False))
        reasons = info.get("reasons", [])

        if allowed:
            candidate_desc_zh = (
                f"候选行为为“{behavior_zh}”，该候选满足当前可行性约束。"
                f"综合评分为 {safe_float(info.get('score', 0.0)):.2f}，"
                f"平均风险代价为 {safe_float(info.get('mean_cost', 0.0)):.1f}，"
                f"高风险比例为 {safe_float(info.get('hard_ratio', 0.0)):.3f}。"
            )
        else:
            candidate_desc_zh = (
                f"候选行为为“{behavior_zh}”，该候选未被视为可选规划标签。"
                f"原因包括：{reasons}。"
            )

        candidate_rollouts_json.append({
            **info,
            "selected": bool(i == selected_idx),
            "description_zh": candidate_desc_zh,
            "description_en": (
                f"The candidate behavior is to {behavior_name_to_english(behavior_name)}. "
                f"Allowed={allowed}. Score={safe_float(info.get('score', 0.0)):.2f}. "
                f"Mean cost={safe_float(info.get('mean_cost', 0.0)):.1f}. "
                f"Hard-risk ratio={safe_float(info.get('hard_ratio', 0.0)):.3f}. "
                f"Invalid reasons={reasons}."
            ),
            "reference_route": points_to_list(r["reference_route"]),
            "waypoints": points_to_list(roll["waypoints"]),
            "yaws": array_to_list(roll["yaws"]),
            "speeds": array_to_list(roll["speeds"]),
            "controls_steer_acc": np.round(roll["controls"].astype(float), 4).tolist(),
        })

    selected_rollout = selected["rollout"]

    risk_label_valid = bool(selected["info"].get("allowed", False))
    fallback_to_expert = bool(selected["info"].get("fallback_to_expert", False))

    language_annotation = build_language_annotation(
        frame_name=frame_name,
        scene_context=scene_context,
        selected=selected,
        selected_rollout=selected_rollout,
        expert_future_waypoints=expert_future_from_measurements,
        future_fps=float(args.future_fps),
        risk_label_valid=risk_label_valid,
        fallback_to_expert=fallback_to_expert,
        selected_reference_route=selected["reference_route"],
    )

    # dreamer_annotation = None
    # if args.save_dreamer_candidates:
    #     dreamer_annotation = build_dreamer_annotation(
    #         frame_name=frame_name,
    #         scored_rollouts=scored_rollouts,
    #         selected_idx=selected_idx,
    #         scene_context=scene_context,
    #         key_dynamic_context=key_dynamic_context,
    #         expert_future_waypoints=expert_future_from_measurements,
    #         future_fps=float(args.future_fps),
    #         args=args,
    #     )

    dreamer_annotation = None
    static_candidate_context = None

    if args.save_dreamer_candidates:
        static_candidate_context = build_static_candidate_context(
            route_dir=route_dir,
            frame_name=frame_name,
            scored_rollouts=scored_rollouts,
            ego_center=ego_center,
            meters_per_pixel=meters_per_pixel,
            args=args,
        )

        dreamer_annotation = build_dreamer_annotation(
            frame_name=frame_name,
            scored_rollouts=scored_rollouts,
            selected_idx=selected_idx,
            scene_context=scene_context,
            key_dynamic_context=key_dynamic_context,
            static_candidate_context=static_candidate_context,
            expert_future_waypoints=expert_future_from_measurements,
            future_fps=float(args.future_fps),
            args=args,
        )

    out = {
        "frame": frame_name,
        "generator": "risk_grounded_multibehavior_bicycle_planner_v2",
        "route_key": args.route_key,
        "coordinate": "ego_local_x_forward_y_right_yaw_positive_right",
        "meters_per_pixel": float(meters_per_pixel),
        "ego_center": [float(ego_center[0]), float(ego_center[1])],

        "model_fps": float(args.model_fps),
        "future_fps": float(args.future_fps),
        "num_future_waypoints": int(args.num_future_waypoints),
        "wheelbase_m": float(args.wheelbase_m),

        "temporal_score_enabled": bool(temporal_bundle.get("enabled", False)),
        "temporal_costmap_frames": temporal_bundle.get("frames", []),
        "temporal_costmap_valid": temporal_bundle.get("valid", []),
        "temporal_costmap_warped": temporal_bundle.get("warped", []),
        "temporal_missing_reasons": temporal_bundle.get("missing_reasons", []),
        "future_costmap_combine": str(args.future_costmap_combine),
        "future_frame_stride": int(args.future_frame_stride),

        "scene_context": scene_context,
        "key_dynamic_context": key_dynamic_context,
        "static_candidate_context": static_candidate_context if static_candidate_context is not None else {},
        "behavior_names": BEHAVIOR_NAMES,
        "valid_behavior_mask": build_valid_behavior_mask(scored_rollouts),

        "selected_index": int(selected_idx),
        "selected_behavior_id": int(selected["info"].get("behavior_id", -1)),
        "selected_behavior_name": str(selected["info"].get("behavior_name", "unknown")),

        # Only allowed risk-planned candidates provide selector supervision.
        # Expert fallback is used for debug/output, but not for selector loss.
        "selector_label": int(selected["info"].get("behavior_id", -1)) if bool(selected["info"].get("allowed", False)) else -1,
        "risk_label_valid": bool(selected["info"].get("allowed", False)),

        "fallback_to_expert": bool(selected["info"].get("fallback_to_expert", False)),
        "fallback_reason": str(selected["info"].get("fallback_reason", "")),

        "selected_mode": f"{selected['info']['behavior_name']}:{selected['info']['reference_mode']}+{selected['info']['speed_mode']}",
        "selected_score": float(selected["info"]["score"]),
        "selected_info": selected["info"],

        # Language / QA annotation for VLA-style supervision.
        "language_annotation": language_annotation,
        "qa_pairs_zh": language_annotation["qa_pairs_zh"],
        "qa_pairs_en": language_annotation["qa_pairs_en"],

        # Important fields.
        "risk_planned_waypoints": points_to_list(selected_rollout["waypoints"]),
        "risk_planned_yaws": array_to_list(selected_rollout["yaws"]),
        "risk_planned_speeds": array_to_list(selected_rollout["speeds"]),
        "risk_planned_controls_steer_acc": np.round(selected_rollout["controls"].astype(float), 4).tolist(),
        "risk_planned_reference_route": points_to_list(selected["reference_route"]),

        # For comparison/debug.
        "expert_reference_route": points_to_list(expert_reference_route),
        "expert_future_waypoints_from_measurements": (
            points_to_list(expert_future_from_measurements)
            if expert_future_from_measurements is not None
            else None
        ),

        "behavior_candidates": candidate_rollouts_json,
        "candidate_rollouts": candidate_rollouts_json,  # backward-compatible alias

        # Optional SimLingo-style dreamer labels built from all candidate rollouts.
        "dreamer_annotation_enabled": bool(args.save_dreamer_candidates),
    }

    if dreamer_annotation is not None:
        out.update({
            "dreamer_candidates": dreamer_annotation["dreamer_candidates"],
            "dreamer_qa_pairs_zh": dreamer_annotation["dreamer_qa_pairs_zh"],
            "dreamer_qa_pairs_en": dreamer_annotation["dreamer_qa_pairs_en"],
            "dreamer_num_candidates": dreamer_annotation["dreamer_num_candidates"],
        })

    output_folder = (
        args.output_folder_augmented
        if args.augmented
        else args.output_folder
    )

    # -------------------------------------------------------------------------
    # Save output JSON / dreamer JSON / debug visualizations
    # -------------------------------------------------------------------------

    output_folder = (
        args.output_folder_augmented
        if args.augmented
        else args.output_folder
    )

    debug_folder = (
        args.debug_folder_augmented
        if args.augmented
        else args.debug_folder
    )

    rgb_debug_folder = (
        args.rgb_debug_folder_augmented
        if args.augmented
        else args.rgb_debug_folder
    )

    # Save the main risk waypoint label.
    output_path = route_dir / output_folder / f"{frame_name}.json.gz"
    save_json_gz(output_path, out)

    # Save optional SimLingo-style dreamer label.
    if dreamer_annotation is not None:
        dreamer_output_folder = (
            args.dreamer_output_folder_augmented
            if args.augmented
            else args.dreamer_output_folder
        )

        dreamer_path = route_dir / dreamer_output_folder / f"{frame_name}.json.gz"
        dreamer_out = {
            "frame": frame_name,
            "source_output_folder": output_folder,
            "source_output_file": f"{frame_name}.json.gz",
            "selected_index": int(selected_idx),
            "selected_behavior_name": str(selected["info"].get("behavior_name", "unknown")),
            "risk_label_valid": bool(selected["info"].get("allowed", False)),
            "fallback_to_expert": bool(selected["info"].get("fallback_to_expert", False)),
            "scene_context": scene_context,
            "key_dynamic_context": key_dynamic_context,
            "static_candidate_context": static_candidate_context if static_candidate_context is not None else {},
            **dreamer_annotation,
        }
        save_json_gz(dreamer_path, dreamer_out)

    # Save BEV rollout debug image.
    if args.save_debug:
        debug_path = route_dir / debug_folder / f"{frame_name}_rollouts.png"
        save_debug_image(
            costmap=costmap,
            scored_rollouts=scored_rollouts,
            selected_idx=selected_idx,
            expert_reference_route=expert_reference_route,
            ego_center=ego_center,
            meters_per_pixel=meters_per_pixel,
            save_path=debug_path,
            args=args,
        )

    # Save RGB projection debug image.
    if args.save_rgb_debug:
        rgb_debug_path = route_dir / rgb_debug_folder / f"{frame_name}_rgb_waypoints.jpg"

        save_rgb_waypoints_debug_image(
            route_dir=route_dir,
            frame_name=frame_name,
            risk_planned_waypoints=selected_rollout["waypoints"],
            expert_future_waypoints=expert_future_from_measurements,
            expert_reference_route=expert_reference_route,
            selected_reference_route=selected["reference_route"],
            selected_info=selected["info"],
            save_path=rgb_debug_path,
            args=args,
        )

    if args.verbose:
        selected_info = selected.get("info", selected)

        msg = (
            f"[OK] {route_dir.name}/{frame_name}: "
            f"selected={out['selected_mode']} "
            f"score={out['selected_score']:.3f} "
            f"allowed={selected_info.get('allowed', None)} "
            f"risk_valid={out.get('risk_label_valid', False)} "
            f"fallback_to_expert={out.get('fallback_to_expert', False)} "
            f"speed0={float(measurement.get('speed', 0.0)):.2f} "
            f"mpp={meters_per_pixel:.3f} "
            f"temporal_maps={len(temporal_bundle.get('costmaps', []))} "
            f"future_valid={sum(bool(v) for v in temporal_bundle.get('valid', [])[1:])}"
        )

        if not selected_info.get("allowed", True):
            msg += f" reasons={selected_info.get('reasons', [])}"

        LOGGER.info(msg)

    return True

def get_frame_names(route_dir: Path, args) -> List[str]:
    if args.frame is not None:
        return [args.frame.replace(".json.gz", "").replace(".npy", "").replace(".npz", "")]

    measurement_dir = route_dir / args.measurements_folder
    costmap_folder = args.costmap_folder_augmented if args.augmented else args.costmap_folder
    costmap_dir = route_dir / costmap_folder

    if measurement_dir.exists():
        measurement_names = {p.name.replace(".json.gz", "") for p in measurement_dir.glob("*.json.gz")}
    else:
        measurement_names = set()

    if costmap_dir.exists():
        cost_names = {p.stem for p in costmap_dir.glob("*.npy")}
    else:
        cost_names = set()

    return sorted(measurement_names & cost_names)

def process_route_dir(route_dir: Path, args) -> Tuple[int, int]:
    frame_names = get_frame_names(route_dir, args)

    if len(frame_names) == 0:
        if args.verbose:
            LOGGER.info(f"[Skip] no matched measurement/costmap frames: {route_dir}")
        return 0, 0

    ok = 0
    total = 0
    for frame_name in frame_names:
        total += 1
        try:
            if process_one_frame(route_dir, frame_name, args):
                ok += 1
        except Exception as e:
            LOGGER.exception(f"[Error] route={route_dir}, frame={frame_name}, error={e}")
            if args.raise_on_error:
                raise

    LOGGER.info(f"[Route Done] {route_dir}: {ok}/{total} frames processed.")
    return ok, total

def find_route_dirs(root: Path, args) -> List[Path]:
    costmap_folder = args.costmap_folder_augmented if args.augmented else args.costmap_folder
    candidates = []

    for p in root.rglob(costmap_folder):
        if p.is_dir() and (p.parent / args.measurements_folder).exists():
            candidates.append(p.parent)

    return sorted(set(candidates))
