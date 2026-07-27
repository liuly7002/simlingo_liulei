# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import math

import cv2
import numpy as np

from . import CHANNEL_NAMES
from .expert_conditioned_selection import actor_id, same_actor


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


def _cfg_int(obj: Any, key: str, default: int) -> int:
    try:
        return int(_cfg_get(obj, key, default))
    except Exception:
        return int(default)


def as_xy_points(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    arr = arr[:, :2]
    return arr[np.isfinite(arr).all(axis=1)].astype(np.float32)


def training_equal_spacing_route(points, count: int = 20, spacing_m: float = 1.0) -> np.ndarray:
    """Match BaseDataset.equal_spacing_route for the unaugmented expert route."""
    points = as_xy_points(points)
    count = max(int(count), 1)
    spacing_m = max(float(spacing_m), 1e-3)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    route = np.concatenate([np.zeros_like(points[:1]), points], axis=0)
    shift = np.roll(route, 1, axis=0)
    shift[0] = shift[1]
    distances = np.cumsum(np.linalg.norm(route - shift, axis=1))
    distances += np.arange(len(distances), dtype=np.float32) * 1e-4
    query = np.arange(count, dtype=np.float32) * spacing_m
    x = np.interp(query, distances, route[:, 0])
    y = np.interp(query, distances, route[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def infer_waypoint_yaws(
    waypoints: np.ndarray,
    min_displacement_m: float = 0.05,
) -> np.ndarray:
    """Infer fallback yaw from waypoint displacement.

    This function is used only when a future measurement does not provide a
    valid real ``theta``.  Very small displacements are ignored because their
    direction is dominated by localization noise, especially while the expert
    vehicle is stopped or creeping.
    """
    waypoints = as_xy_points(waypoints)
    if len(waypoints) == 0:
        return np.zeros((0,), dtype=np.float32)
    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), waypoints], axis=0)
    delta = points[1:] - points[:-1]
    result = np.zeros((len(waypoints),), dtype=np.float32)
    previous = 0.0
    min_displacement_m = max(float(min_displacement_m), 1e-5)
    for index, vector in enumerate(delta):
        if float(np.linalg.norm(vector)) >= min_displacement_m:
            previous = math.atan2(float(vector[1]), float(vector[0]))
        result[index] = previous
    return result


def resolve_expert_yaws(
    waypoints: np.ndarray,
    measured_yaws: Optional[np.ndarray],
) -> np.ndarray:
    """Use real future ego yaw and fall back only for missing entries.

    ``measured_yaws`` must already be expressed relative to the current ego
    heading.  Finite entries therefore preserve the expert vehicle's true body
    orientation even when its displacement is very small.
    """
    waypoints = as_xy_points(waypoints)
    fallback = infer_waypoint_yaws(waypoints)
    if measured_yaws is None:
        return fallback

    measured = np.asarray(measured_yaws, dtype=np.float32).reshape(-1)
    result = fallback.copy()
    count = min(len(result), len(measured))
    for index in range(count):
        if np.isfinite(measured[index]):
            angle = float(measured[index])
            result[index] = math.atan2(math.sin(angle), math.cos(angle))
    return result.astype(np.float32)


def infer_polyline_yaws(points: np.ndarray) -> np.ndarray:
    points = as_xy_points(points)
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(points) == 1:
        return np.zeros((1,), dtype=np.float32)
    tangent = np.zeros_like(points)
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangent[1:-1] = points[2:] - points[:-2]
    result = np.zeros((len(points),), dtype=np.float32)
    previous = 0.0
    for index, vector in enumerate(tangent):
        if float(np.linalg.norm(vector)) > 1e-5:
            previous = math.atan2(float(vector[1]), float(vector[0]))
        result[index] = previous
    return result


def _local_points_to_pixels(points: np.ndarray, ego_center: Sequence[float], meters_per_pixel: float) -> np.ndarray:
    points = as_xy_points(points)
    cx, cy = float(ego_center[0]), float(ego_center[1])
    col = cx + points[:, 1] / float(meters_per_pixel)
    row = cy - points[:, 0] / float(meters_per_pixel)
    return np.stack([col, row], axis=1).astype(np.float32)


def _obb_corners(center_xy, yaw_rad, half_length_m, half_width_m) -> np.ndarray:
    local = np.asarray(
        [
            [half_length_m, half_width_m],
            [half_length_m, -half_width_m],
            [-half_length_m, -half_width_m],
            [-half_length_m, half_width_m],
        ],
        dtype=np.float32,
    )
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return local @ rotation.T + np.asarray(center_xy, dtype=np.float32).reshape(1, 2)


def _fill_obb(mask, center_xy, yaw_rad, half_length_m, half_width_m, ego_center, meters_per_pixel):
    corners = _obb_corners(
        center_xy,
        yaw_rad,
        max(float(half_length_m), 0.05),
        max(float(half_width_m), 0.05),
    )
    pixels = _local_points_to_pixels(corners, ego_center, meters_per_pixel)
    polygon = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [polygon], 1)


def _draw_route_occupancy(shape, route_points, half_length, half_width, margin, ego_center, mpp):
    mask = np.zeros(shape, dtype=np.float32)
    yaws = infer_polyline_yaws(route_points)
    count = max(len(route_points), 1)
    for point, yaw in zip(route_points, yaws):
        step = np.zeros(shape, dtype=np.uint8)
        _fill_obb(
            step,
            point,
            yaw,
            half_length + margin,
            half_width + margin,
            ego_center,
            mpp,
        )
        mask += step.astype(np.float32) / float(count)
    return np.clip(mask, 0.0, 1.0)


def _find_actor(records, target_actor: Optional[Dict]):
    if target_actor is None:
        return None
    for actor in records or []:
        if same_actor(actor, target_actor):
            return actor
    return None


def build_grid(
    shape: Tuple[int, int],
    expert_route: np.ndarray,
    expert_future: np.ndarray,
    expert_future_yaws: Optional[np.ndarray],
    primary_actor: Optional[Dict],
    secondary_actor: Optional[Dict],
    actor_timelines: Dict[int, list],
    ego_center: Sequence[float],
    meters_per_pixel: float,
    cfg,
) -> Dict:
    horizon = int(cfg.horizon.num_future_waypoints)
    route_points = as_xy_points(expert_route)
    expert_future = as_xy_points(expert_future)
    #修改20260727：优先使用未来measurement中的真实相对theta；只有缺失时才根据位移回退推算。
    expert_yaws = resolve_expert_yaws(expert_future, expert_future_yaws)

    structured_cfg = _cfg_get(cfg, "structured_world", {})
    grid_cfg = _cfg_get(structured_cfg, "grid", {})
    ego_half_l = float(cfg.vehicle.ego_half_length_m)
    ego_half_w = float(cfg.vehicle.ego_half_width_m)
    route_margin = max(_cfg_float(grid_cfg, "reference_route_footprint_margin_m", 0.0), 0.0)
    ego_margin = max(_cfg_float(grid_cfg, "ego_footprint_margin_m", 0.0), 0.0)
    interaction_margin = max(_cfg_float(grid_cfg, "interaction_margin_m", 0.35), 0.0)
    actor_margin = max(_cfg_float(grid_cfg, "actor_footprint_margin_m", 0.0), 0.0)

    route_mask = _draw_route_occupancy(
        shape,
        route_points,
        ego_half_l,
        ego_half_w,
        route_margin,
        ego_center,
        meters_per_pixel,
    ) if len(route_points) else np.zeros(shape, dtype=np.float32)

    ego_union = np.zeros(shape, dtype=np.float32)
    primary_union = np.zeros(shape, dtype=np.float32)
    interaction_union = np.zeros(shape, dtype=np.float32)
    secondary_union = np.zeros(shape, dtype=np.float32)
    primary_frames = 0
    secondary_frames = 0

    for step in range(1, horizon + 1):
        if step - 1 >= len(expert_future):
            continue
        waypoint = expert_future[step - 1]
        yaw = float(expert_yaws[step - 1])
        ego_mask = np.zeros(shape, dtype=np.uint8)
        _fill_obb(
            ego_mask,
            waypoint,
            yaw,
            ego_half_l + ego_margin,
            ego_half_w + ego_margin,
            ego_center,
            meters_per_pixel,
        )
        ego_union += ego_mask.astype(np.float32)

        primary_future = _find_actor(actor_timelines.get(step, []), primary_actor)
        if primary_future is not None:
            primary_frames += 1
            primary_mask = np.zeros(shape, dtype=np.uint8)
            _fill_obb(
                primary_mask,
                [float(primary_future.get("x_m", 0.0)), float(primary_future.get("y_m", 0.0))],
                float(primary_future.get("yaw_rad", 0.0)),
                float(primary_future.get("half_length_m", 1.0)) + actor_margin,
                float(primary_future.get("half_width_m", 0.5)) + actor_margin,
                ego_center,
                meters_per_pixel,
            )
            primary_union += primary_mask.astype(np.float32)

            ego_interaction = np.zeros(shape, dtype=np.uint8)
            _fill_obb(
                ego_interaction,
                waypoint,
                yaw,
                ego_half_l + ego_margin + interaction_margin,
                ego_half_w + ego_margin + interaction_margin,
                ego_center,
                meters_per_pixel,
            )
            interaction_union += (
                (ego_interaction > 0) & (primary_mask > 0)
            ).astype(np.float32)

        secondary_future = _find_actor(actor_timelines.get(step, []), secondary_actor)
        if secondary_future is not None:
            secondary_frames += 1
            secondary_mask = np.zeros(shape, dtype=np.uint8)
            _fill_obb(
                secondary_mask,
                [float(secondary_future.get("x_m", 0.0)), float(secondary_future.get("y_m", 0.0))],
                float(secondary_future.get("yaw_rad", 0.0)),
                float(secondary_future.get("half_length_m", 1.0)) + actor_margin,
                float(secondary_future.get("half_width_m", 0.5)) + actor_margin,
                ego_center,
                meters_per_pixel,
            )
            secondary_union += secondary_mask.astype(np.float32)

    if horizon > 0:
        ego_union = np.clip(ego_union / float(horizon), 0.0, 1.0)
        primary_union = np.clip(primary_union / float(horizon), 0.0, 1.0)
        interaction_union = np.clip(interaction_union / float(horizon), 0.0, 1.0)
        secondary_union = np.clip(secondary_union / float(horizon), 0.0, 1.0)

    grid = np.stack(
        [route_mask, ego_union, primary_union, interaction_union, secondary_union],
        axis=0,
    ).astype(np.float32)
    return {
        "grid": grid,
        "route_mask": route_mask,
        "ego_future_mask": ego_union,
        "primary_actor_mask": primary_union,
        "interaction_mask": interaction_union,
        "secondary_actor_mask": secondary_union,
        "primary_actor_future_frames_found": int(primary_frames),
        "secondary_actor_future_frames_found": int(secondary_frames),
    }


def save_npz(
    path: Path,
    grid_result: Dict,
    valid: bool,
    invalid_reason: str,
    frame_name: str,
    primary_actor: Optional[Dict],
    primary_score: float,
    secondary_actor: Optional[Dict],
    secondary_score: float,
    secondary_source: str,
    expert_match: Dict,
    reference_route_points_used: int,
    meters_per_pixel: float,
    ego_center: Sequence[float],
    cfg,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(grid_result["grid"], dtype=np.float32)
    np.savez_compressed(
        str(path),
        future_interaction_grid=grid,
        valid=np.asarray(bool(valid), dtype=np.uint8),
        invalid_reason=np.asarray(str(invalid_reason), dtype="<U96"),
        channel_names=np.asarray(CHANNEL_NAMES, dtype="<U64"),
        frame=np.asarray(str(frame_name), dtype="<U32"),
        coordinate=np.asarray("ego_local_x_forward_y_right_yaw_positive_right", dtype="<U64"),
        critical_actor_id=np.asarray(actor_id(primary_actor) or "", dtype="<U64"),
        has_causal_actor=np.asarray(primary_actor is not None, dtype=np.uint8),
        primary_actor_future_frames_found=np.asarray(
            int(grid_result["primary_actor_future_frames_found"]), dtype=np.int16
        ),
        secondary_actor_id=np.asarray(actor_id(secondary_actor) or "", dtype="<U64"),
        has_secondary_actor=np.asarray(secondary_actor is not None, dtype=np.uint8),
        secondary_actor_source=np.asarray(str(secondary_source), dtype="<U96"),
        secondary_actor_future_frames_found=np.asarray(
            int(grid_result["secondary_actor_future_frames_found"]), dtype=np.int16
        ),
        num_future_frames=np.asarray(int(cfg.horizon.num_future_waypoints), dtype=np.int16),
        future_frame_stride=np.asarray(int(cfg.horizon.future_frame_stride), dtype=np.int16),
        temporal_weighting=np.asarray("equal_mean_over_all_future_frames", dtype="<U48"),
        reference_route_weighting=np.asarray("equal_mean_over_all_route_points", dtype="<U48"),
        reference_route_points_used=np.asarray(int(reference_route_points_used), dtype=np.int16),
        meters_per_pixel=np.asarray(float(meters_per_pixel), dtype=np.float32),
        ego_center=np.asarray(ego_center, dtype=np.float32),
        trajectory_source=np.asarray("expert", dtype="<U32"),
        actor_selection_source=np.asarray(
            "expert_matched_lg_counterfactual_reselection", dtype="<U64"
        ),
        expert_match_valid=np.asarray(bool(expert_match.get("valid", False)), dtype=np.uint8),
        expert_match_reason=np.asarray(str(expert_match.get("reason", "")), dtype="<U96"),
        expert_match_candidate_index=np.asarray(int(expert_match.get("selected_index", -1)), dtype=np.int32),
        expert_match_intent_name=np.asarray(str(expert_match.get("intent_name", "")), dtype="<U64"),
        expert_match_variant_id=np.asarray(str(expert_match.get("variant_id", "")), dtype="<U96"),
        expert_match_ade_m=np.asarray(float(expert_match.get("ade_m", float("nan"))), dtype=np.float32),
        expert_match_fde_m=np.asarray(float(expert_match.get("fde_m", float("nan"))), dtype=np.float32),
        expert_match_speed_mae_mps=np.asarray(
            float(expert_match.get("speed_mae_mps", float("nan"))), dtype=np.float32
        ),
        primary_causal_score=np.asarray(float(primary_score), dtype=np.float32),
        secondary_causal_score=np.asarray(float(secondary_score), dtype=np.float32),
    )


def _normalize(mask):
    value = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    maximum = float(value.max()) if value.size else 0.0
    return value / maximum if maximum > 1e-8 else value


def _colorize(mask, color, background):
    values = _normalize(mask)[..., None]
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    fg = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.rint(np.clip(bg + values * (fg - bg), 0, 255)).astype(np.uint8)


def _crop(image, ego_center, mpp, cfg):
    h, w = image.shape[:2]
    cx, cy = float(ego_center[0]), float(ego_center[1])
    forward = max(_cfg_float(cfg, "forward_m", 32.0), 1.0)
    rear = max(_cfg_float(cfg, "rear_m", 6.0), 0.0)
    side = max(_cfg_float(cfg, "side_m", 14.0), 1.0)
    r0 = max(0, int(math.floor(cy - forward / mpp)))
    r1 = min(h, int(math.ceil(cy + rear / mpp)))
    c0 = max(0, int(math.floor(cx - side / mpp)))
    c1 = min(w, int(math.ceil(cx + side / mpp)))
    return image[r0:r1, c0:c1] if r1 > r0 and c1 > c0 else image


def _label(image, text, cfg):
    out = image.copy()
    scale = max(_cfg_float(cfg, "corner_label_font_scale", 0.45), 0.2)
    thickness = max(_cfg_int(cfg, "corner_label_thickness", 1), 1)
    outline = max(_cfg_int(cfg, "corner_label_outline_thickness", 3), thickness)
    x = max(_cfg_int(cfg, "corner_label_margin_x_px", 8), 0)
    y_margin = max(_cfg_int(cfg, "corner_label_margin_y_px", 8), 0)
    (_, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = (x, y_margin + text_h + baseline)
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), outline, cv2.LINE_AA)
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), thickness, cv2.LINE_AA)
    return out


def save_debug_image(path: Path, grid_result: Dict, ego_center, meters_per_pixel, cfg) -> None:
    debug_cfg = _cfg_get(cfg, "debug", {})
    background = tuple(int(v) for v in _cfg_get(debug_cfg, "background_bgr", [54, 60, 66]))
    masks = [
        grid_result["route_mask"],
        grid_result["ego_future_mask"],
        grid_result["primary_actor_mask"],
        grid_result["interaction_mask"],
        grid_result["secondary_actor_mask"],
    ]
    colors = [(220, 145, 70), (95, 190, 95), (75, 165, 235), (200, 95, 215), (85, 205, 205)]
    labels = ["C0 Expert Route", "C1 Expert Ego", "C2 Primary", "C3 Interaction", "C4 Secondary"]
    panels = [_colorize(mask, color, background) for mask, color in zip(masks, colors)]

    overlay = np.full((*masks[0].shape, 3), background, dtype=np.uint8)
    for mask, color, alpha in zip(masks, colors, [0.78, 0.72, 0.90, 1.0, 0.85]):
        strength = (_normalize(mask)[..., None] * alpha).clip(0.0, 1.0)
        overlay = np.rint(
            overlay.astype(np.float32) * (1.0 - strength)
            + np.asarray(color, dtype=np.float32).reshape(1, 1, 3) * strength
        ).clip(0, 255).astype(np.uint8)
    panels.append(overlay)
    labels.append("Full")

    width = max(_cfg_int(debug_cfg, "panel_width", 320), 160)
    gap = max(_cfg_int(debug_cfg, "panel_gap_px", 12), 0)
    border = max(_cfg_int(debug_cfg, "panel_border_px", 2), 0)
    border_color = tuple(int(v) for v in _cfg_get(debug_cfg, "panel_border_bgr", [160, 160, 160]))
    processed = []
    for panel, label in zip(panels, labels):
        panel = _crop(panel, ego_center, meters_per_pixel, debug_cfg)
        scale = width / float(max(panel.shape[1], 1))
        panel = cv2.resize(panel, (width, max(1, int(round(panel.shape[0] * scale)))), interpolation=cv2.INTER_NEAREST)
        panel = _label(panel, label, debug_cfg)
        processed.append(panel)

    max_h = max(panel.shape[0] for panel in processed)
    normalized = []
    for panel in processed:
        panel = cv2.copyMakeBorder(
            panel, 0, max_h - panel.shape[0], 0, 0,
            borderType=cv2.BORDER_CONSTANT, value=background,
        )
        if border > 0:
            panel = cv2.copyMakeBorder(panel, border, border, border, border, cv2.BORDER_CONSTANT, value=border_color)
        normalized.append(panel)

    rows = []
    spacer_h = np.full((normalized[0].shape[0], gap, 3), background, dtype=np.uint8) if gap else None
    for start in (0, 2, 4):
        row = np.concatenate(
            [normalized[start], spacer_h, normalized[start + 1]] if spacer_h is not None else normalized[start:start + 2],
            axis=1,
        )
        rows.append(row)
    spacer_v = np.full((gap, rows[0].shape[1], 3), background, dtype=np.uint8) if gap else None
    composite = np.concatenate(
        [rows[0], spacer_v, rows[1], spacer_v, rows[2]] if spacer_v is not None else rows,
        axis=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), composite)

    if _cfg_bool(debug_cfg, "save_individual_channels", False):
        for index, panel in enumerate(processed):
            cv2.imwrite(str(path.with_name(f"{path.stem}_c{index}.png")), panel)
