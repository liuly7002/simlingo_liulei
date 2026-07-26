#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate LG-derived five-channel future interaction grid labels.

The saved grid is aligned with the existing BEV and contains:

    channel 0: equal-weight ego-footprint occupancy on 20 LG selected-route points
    channel 1: equal-weight future ego-footprint occupancy over 10 frames
    channel 2: equal-weight primary causal-actor footprint occupancy over 10 frames
    channel 3: equal-weight time-aligned ego/primary-actor interaction over 10 frames
    channel 4: equal-weight secondary-actor footprint occupancy over 10 frames

All route points and future frames use equal weights. No temporal attenuation is
applied. Channel 4 first uses an explicitly saved ``secondary_attention_actor``
when the LG label contains one; otherwise it can fall back to the highest-scoring
non-primary accepted actor in ``causal_analysis.object_tests``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import math
import sys

import cv2
import numpy as np


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "future_interaction_grid.yaml"

# Optional top-of-file overrides. Leave as None to use YAML.
CONFIG_PATH_OVERRIDE = None
INPUT_OVERRIDE = None
FRAME_OVERRIDE = None

sys.path.insert(0, str(CURRENT_FILE.parent))

from lg_waypoint_planner.actor_loader import load_future_actor_timelines
from lg_waypoint_planner.config_utils import load_yaml
from lg_waypoint_planner.dataset import get_ego_center, get_meters_per_pixel
from lg_waypoint_planner.geometry import local_points_to_pixels, resample_polyline
from lg_waypoint_planner.io_utils import find_route_dirs, load_costmap, load_json_gz
from lg_waypoint_planner.logger import LOGGER, setup_logger


CHANNEL_NAMES = np.asarray(
    [
        "selected_route_ego_footprint_occupancy",
        "future_ego_footprint_occupancy",
        "primary_causal_actor_future_footprint_occupancy",
        "time_aligned_primary_future_interaction",
        "secondary_actor_future_footprint_occupancy",
    ],
    dtype="<U64",
)


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    try:
        return getattr(obj, key)
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default
    return default


def _cfg_bool(obj, key: str, default: bool) -> bool:
    value = _cfg_get(obj, key, default)
    if isinstance(value, str):
        return value.lower() in ["1", "true", "yes", "y", "on"]
    return bool(value)


def _cfg_float(obj, key: str, default: float) -> float:
    try:
        return float(_cfg_get(obj, key, default))
    except Exception:
        return float(default)


def _cfg_int(obj, key: str, default: int) -> int:
    try:
        return int(_cfg_get(obj, key, default))
    except Exception:
        return int(default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def resolve_config_path() -> Path:
    if CONFIG_PATH_OVERRIDE is not None:
        return Path(CONFIG_PATH_OVERRIDE).expanduser().resolve()
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).expanduser().resolve()
    return DEFAULT_CONFIG


def _normalize_frame_name(frame_name: str) -> str:
    return str(frame_name).replace(".json.gz", "").replace(".npy", "").replace(".npz", "")


def _list_frame_names(route_dir: Path, cfg) -> List[str]:
    frame_cfg = _cfg_get(_cfg_get(cfg, "run", {}), "frame", None)
    if frame_cfg not in [None, "", "None", "null"]:
        return [_normalize_frame_name(str(frame_cfg))]

    lg_dir = route_dir / str(cfg.paths.lg_folder)
    costmap_dir = route_dir / str(cfg.paths.costmap_folder)
    if not lg_dir.exists() or not costmap_dir.exists():
        return []

    lg_frames = {p.name.replace(".json.gz", "") for p in lg_dir.glob("*.json.gz")}
    costmap_frames = {p.stem for p in costmap_dir.glob("*.npy")}
    return sorted(lg_frames & costmap_frames)


def _as_xy_points(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    arr = arr[:, :2]
    return arr[np.isfinite(arr).all(axis=1)].astype(np.float32)


def _fit_point_count(points: np.ndarray, count: int) -> np.ndarray:
    points = _as_xy_points(points)
    count = max(int(count), 1)
    if len(points) == 0:
        return points
    if len(points) >= count:
        return points[:count].astype(np.float32)
    pad = np.repeat(points[-1:], count - len(points), axis=0)
    return np.concatenate([points, pad], axis=0).astype(np.float32)


def _resample_reference_route(points: np.ndarray, count: int, spacing_m: float) -> np.ndarray:
    points = _as_xy_points(points)
    count = max(int(count), 1)
    spacing_m = max(float(spacing_m), 1e-3)
    if len(points) == 0:
        return points
    if count == 1:
        return points[:1].astype(np.float32)
    horizon_m = float(count - 1) * spacing_m
    return resample_polyline(
        points,
        spacing_m=spacing_m,
        horizon_m=horizon_m,
    )[:count].astype(np.float32)


def _infer_waypoint_yaws(waypoints: np.ndarray) -> np.ndarray:
    waypoints = _as_xy_points(waypoints)
    if len(waypoints) == 0:
        return np.zeros((0,), dtype=np.float32)

    points = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), waypoints],
        axis=0,
    )
    delta = points[1:] - points[:-1]
    yaws = np.zeros((len(waypoints),), dtype=np.float32)
    previous_yaw = 0.0
    for index, vector in enumerate(delta):
        if float(np.linalg.norm(vector)) > 1e-5:
            previous_yaw = float(math.atan2(float(vector[1]), float(vector[0])))
        yaws[index] = previous_yaw
    return yaws


def _infer_polyline_yaws(points: np.ndarray) -> np.ndarray:
    """Infer an OBB heading at every route point from the local route tangent."""
    points = _as_xy_points(points)
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(points) == 1:
        return np.zeros((1,), dtype=np.float32)

    tangent = np.zeros_like(points, dtype=np.float32)
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangent[1:-1] = points[2:] - points[:-2]

    yaws = np.zeros((len(points),), dtype=np.float32)
    previous_yaw = 0.0
    for index, vector in enumerate(tangent):
        if float(np.linalg.norm(vector)) > 1e-5:
            previous_yaw = float(math.atan2(float(vector[1]), float(vector[0])))
        yaws[index] = previous_yaw
    return yaws


def _load_waypoint_yaws(lg_label: Dict, waypoints: np.ndarray, count: int) -> np.ndarray:
    raw = lg_label.get("supervision", {}).get("risk_planned_yaws", [])
    yaws = np.asarray(raw, dtype=np.float32).reshape(-1)
    yaws = yaws[np.isfinite(yaws)]
    if len(yaws) >= count:
        return yaws[:count].astype(np.float32)

    inferred = _infer_waypoint_yaws(waypoints)
    if len(yaws) == 0:
        return inferred

    output = inferred.copy()
    output[: len(yaws)] = yaws
    return output


def _obb_corners(
    center_xy: Sequence[float],
    yaw_rad: float,
    half_length_m: float,
    half_width_m: float,
) -> np.ndarray:
    local = np.asarray(
        [
            [half_length_m, half_width_m],
            [half_length_m, -half_width_m],
            [-half_length_m, -half_width_m],
            [-half_length_m, half_width_m],
        ],
        dtype=np.float32,
    )
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    center = np.asarray(center_xy, dtype=np.float32).reshape(1, 2)
    return local @ rotation.T + center


def _fill_obb(
    mask: np.ndarray,
    center_xy: Sequence[float],
    yaw_rad: float,
    half_length_m: float,
    half_width_m: float,
    ego_center: Sequence[float],
    meters_per_pixel: float,
    value: int = 1,
) -> None:
    corners = _obb_corners(
        center_xy=center_xy,
        yaw_rad=yaw_rad,
        half_length_m=max(float(half_length_m), 0.05),
        half_width_m=max(float(half_width_m), 0.05),
    )
    pixels = local_points_to_pixels(corners, list(ego_center), meters_per_pixel)
    polygon = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, polygon, int(value), lineType=cv2.LINE_8)


def _draw_reference_footprint_occupancy(
    shape: Tuple[int, int],
    route_points: np.ndarray,
    ego_half_length_m: float,
    ego_half_width_m: float,
    footprint_margin_m: float,
    ego_center: Sequence[float],
    meters_per_pixel: float,
) -> np.ndarray:
    """Rasterize the ego footprint at every selected-route point with equal weight."""
    route_points = _as_xy_points(route_points)
    if len(route_points) == 0:
        return np.zeros(shape, dtype=np.float32)

    route_yaws = _infer_polyline_yaws(route_points)
    accumulated = np.zeros(shape, dtype=np.float32)
    for point, yaw in zip(route_points, route_yaws):
        footprint = np.zeros(shape, dtype=np.uint8)
        _fill_obb(
            footprint,
            center_xy=point,
            yaw_rad=float(yaw),
            half_length_m=float(ego_half_length_m) + float(footprint_margin_m),
            half_width_m=float(ego_half_width_m) + float(footprint_margin_m),
            ego_center=ego_center,
            meters_per_pixel=meters_per_pixel,
        )
        accumulated += footprint.astype(np.float32)

    return np.clip(accumulated / float(len(route_points)), 0.0, 1.0)


def _actor_id_equal(left, right) -> bool:
    if left is None or right is None:
        return False
    return str(left) == str(right)


def _find_actor(records: Sequence[Dict], actor_id) -> Optional[Dict]:
    for record in records:
        if _actor_id_equal(record.get("id", None), actor_id):
            return record
    return None


def _actor_exists(actor: Optional[Dict]) -> bool:
    return isinstance(actor, dict) and bool(actor.get("exists", False))


def _secondary_from_saved_factor(lg_label: Dict, primary_actor_id) -> Optional[Dict]:
    """Read an explicitly saved secondary attention actor from compatible labels."""
    for key in ["factor", "critical_factor", "most_influential_factor"]:
        factor = lg_label.get(key, {}) or {}
        if not isinstance(factor, dict):
            continue
        actor = factor.get("secondary_attention_actor", {}) or {}
        if not _actor_exists(actor):
            continue
        actor_id = actor.get("id", None)
        if actor_id is None or _actor_id_equal(actor_id, primary_actor_id):
            continue
        return dict(actor)
    return None


def _secondary_from_object_tests(lg_label: Dict, primary_actor_id, cfg) -> Optional[Dict]:
    """Select the strongest accepted non-primary counterfactual actor test.

    This fallback uses information already stored in the public LG label. It is
    deliberately recorded as a secondary *candidate* source because only the
    primary object receives the final post-selection causal revalidation.
    """
    secondary_cfg = _cfg_get(cfg, "secondary_actor", {})
    require_accepted = _cfg_bool(secondary_cfg, "require_causal_accepted", True)
    tests = (lg_label.get("causal_analysis", {}) or {}).get("object_tests", []) or []

    ranked = []
    for test in tests:
        if not isinstance(test, dict) or not bool(test.get("counterfactual_valid", False)):
            continue
        if require_accepted and not bool(test.get("causal_accepted", False)):
            continue
        actor = test.get("actor", {}) or {}
        if not isinstance(actor, dict):
            continue
        actor_id = actor.get("id", None)
        if actor_id is None or _actor_id_equal(actor_id, primary_actor_id):
            continue
        final_score = test.get("final_causal_score", None)
        score = _safe_float(
            final_score,
            _safe_float(test.get("causal_score", 0.0), 0.0),
        )
        ranked.append((score, actor))

    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    actor = dict(ranked[0][1])
    actor["exists"] = True
    actor["secondary_causal_score"] = float(ranked[0][0])
    return actor


def _select_secondary_actor(lg_label: Dict, primary_actor_id, cfg) -> Tuple[Optional[Dict], str]:
    secondary_cfg = _cfg_get(cfg, "secondary_actor", {})
    if not _cfg_bool(secondary_cfg, "enabled", True):
        return None, "disabled"

    if _cfg_bool(secondary_cfg, "prefer_saved_secondary_attention_actor", True):
        actor = _secondary_from_saved_factor(lg_label, primary_actor_id)
        if actor is not None:
            return actor, "saved_secondary_attention_actor"

    if _cfg_bool(secondary_cfg, "fallback_to_causal_object_tests", True):
        actor = _secondary_from_object_tests(lg_label, primary_actor_id, cfg)
        if actor is not None:
            return actor, "accepted_non_primary_causal_test"

    return None, "none"


def _empty_result(shape: Tuple[int, int]):
    return (
        np.zeros(shape, dtype=np.float32),  # selected route ego footprints
        np.zeros(shape, dtype=np.float32),  # future ego footprints
        np.zeros(shape, dtype=np.float32),  # primary actor future footprints
        np.zeros(shape, dtype=np.float32),  # primary interaction
        np.zeros(shape, dtype=np.float32),  # secondary actor future footprints
    )


def _save_npz(
    output_path: Path,
    grid: np.ndarray,
    valid: bool,
    invalid_reason: str,
    frame_name: str,
    critical_actor_id,
    has_causal_actor: bool,
    primary_actor_future_frames_found: int,
    secondary_actor_id,
    has_secondary_actor: bool,
    secondary_actor_source: str,
    secondary_actor_future_frames_found: int,
    reference_route_points_used: int,
    cfg,
    meters_per_pixel: float,
    ego_center: Sequence[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        future_interaction_grid=grid.astype(np.float32),
        valid=np.asarray(bool(valid), dtype=np.uint8),
        invalid_reason=np.asarray(str(invalid_reason), dtype="<U96"),
        channel_names=CHANNEL_NAMES,
        frame=np.asarray(str(frame_name), dtype="<U32"),
        coordinate=np.asarray(
            "ego_local_x_forward_y_right_yaw_positive_right",
            dtype="<U64",
        ),
        critical_actor_id=np.asarray(
            "" if critical_actor_id is None else str(critical_actor_id),
            dtype="<U64",
        ),
        has_causal_actor=np.asarray(bool(has_causal_actor), dtype=np.uint8),
        primary_actor_future_frames_found=np.asarray(
            primary_actor_future_frames_found,
            dtype=np.int16,
        ),
        secondary_actor_id=np.asarray(
            "" if secondary_actor_id is None else str(secondary_actor_id),
            dtype="<U64",
        ),
        has_secondary_actor=np.asarray(bool(has_secondary_actor), dtype=np.uint8),
        secondary_actor_source=np.asarray(str(secondary_actor_source), dtype="<U64"),
        secondary_actor_future_frames_found=np.asarray(
            secondary_actor_future_frames_found,
            dtype=np.int16,
        ),
        num_future_frames=np.asarray(int(cfg.horizon.num_future_waypoints), dtype=np.int16),
        future_frame_stride=np.asarray(int(cfg.horizon.future_frame_stride), dtype=np.int16),
        temporal_weighting=np.asarray("equal_mean_over_all_future_frames", dtype="<U48"),
        reference_route_weighting=np.asarray("equal_mean_over_all_route_points", dtype="<U48"),
        reference_route_points_used=np.asarray(reference_route_points_used, dtype=np.int16),
        meters_per_pixel=np.asarray(float(meters_per_pixel), dtype=np.float32),
        ego_center=np.asarray(ego_center, dtype=np.float32),
    )


def _normalize_debug_values(mask: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum > 1e-8:
        values = values / maximum
    return values


def _debug_background(debug_cfg) -> Tuple[int, int, int]:
    value = _cfg_get(debug_cfg, "background_bgr", [54, 60, 66])
    try:
        items = [int(np.clip(int(v), 0, 255)) for v in list(value)[:3]]
        if len(items) == 3:
            return tuple(items)
    except Exception:
        pass
    return (54, 60, 66)


def _colorize_mask(
    mask: np.ndarray,
    color: Tuple[int, int, int],
    background: Tuple[int, int, int],
    normalize_for_debug: bool = False,
) -> np.ndarray:
    values = (
        _normalize_debug_values(mask)
        if normalize_for_debug
        else np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    )
    background_array = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    image = background_array + values[..., None] * (color_array - background_array)
    return np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.85,
) -> np.ndarray:
    values = _normalize_debug_values(mask)[..., None]
    strength = np.clip(values * float(alpha), 0.0, 1.0)
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    output = image.astype(np.float32) * (1.0 - strength) + color_array * strength
    return np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)


def _crop_debug_roi(
    image: np.ndarray,
    ego_center: Sequence[float],
    meters_per_pixel: float,
    debug_cfg,
) -> np.ndarray:
    """Crop a forward-looking BEV region so ego appears near the lower edge."""
    height, width = image.shape[:2]
    cx, cy = float(ego_center[0]), float(ego_center[1])
    mpp = max(float(meters_per_pixel), 1e-6)

    forward_m = max(_cfg_float(debug_cfg, "forward_m", 32.0), 1.0)
    rear_m = max(_cfg_float(debug_cfg, "rear_m", 6.0), 0.0)
    side_m = max(_cfg_float(debug_cfg, "side_m", 14.0), 1.0)

    row0 = max(0, int(math.floor(cy - forward_m / mpp)))
    row1 = min(height, int(math.ceil(cy + rear_m / mpp)))
    col0 = max(0, int(math.floor(cx - side_m / mpp)))
    col1 = min(width, int(math.ceil(cx + side_m / mpp)))

    if row1 <= row0 or col1 <= col0:
        return image
    return image[row0:row1, col0:col1]


def _add_panel_title(image: np.ndarray, title: str, debug_cfg) -> np.ndarray:
    # 修改20260726：debug 图取消标题，保持函数以兼容旧调用。
    del title, debug_cfg
    return image


def _resize_panel(image: np.ndarray, panel_width: int) -> np.ndarray:
    panel_width = max(int(panel_width), 160)
    if image.shape[1] == panel_width:
        return image
    scale = panel_width / float(max(image.shape[1], 1))
    return cv2.resize(
        image,
        (panel_width, max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )


def _add_panel_border(image: np.ndarray, debug_cfg) -> np.ndarray:
    # 修改20260726：为单行 debug 图中的每个子图添加边框，便于区分五个小块。
    border_px = max(_cfg_int(debug_cfg, "panel_border_px", 2), 0)
    if border_px <= 0:
        return image
    border_color = _cfg_get(debug_cfg, "panel_border_bgr", [160, 160, 160])
    try:
        border_color = tuple(int(np.clip(int(v), 0, 255)) for v in list(border_color)[:3])
    except Exception:
        border_color = (160, 160, 160)
    return cv2.copyMakeBorder(
        image,
        border_px,
        border_px,
        border_px,
        border_px,
        borderType=cv2.BORDER_CONSTANT,
        value=border_color,
    )


def _add_panel_corner_label(image: np.ndarray, label: str, debug_cfg) -> np.ndarray:
    # 修改20260726：标签必须在面板缩放完成后绘制，避免文字跟随图像一起被放大。
    if not _cfg_bool(debug_cfg, "show_corner_labels", True):
        return image

    out = image.copy()
    text = str(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(_cfg_float(debug_cfg, "corner_label_font_scale", 0.45), 0.2)
    thickness = max(_cfg_int(debug_cfg, "corner_label_thickness", 1), 1)
    outline_thickness = max(_cfg_int(debug_cfg, "corner_label_outline_thickness", 3), thickness)
    margin_x = max(_cfg_int(debug_cfg, "corner_label_margin_x_px", 8), 0)
    margin_y = max(_cfg_int(debug_cfg, "corner_label_margin_y_px", 8), 0)

    text_color = _cfg_get(debug_cfg, "corner_label_text_bgr", [240, 240, 240])
    outline_color = _cfg_get(debug_cfg, "corner_label_outline_bgr", [35, 35, 35])
    try:
        text_color = tuple(int(np.clip(int(v), 0, 255)) for v in list(text_color)[:3])
    except Exception:
        text_color = (240, 240, 240)
    try:
        outline_color = tuple(int(np.clip(int(v), 0, 255)) for v in list(outline_color)[:3])
    except Exception:
        outline_color = (35, 35, 35)

    (_, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    origin = (margin_x, margin_y + text_h + baseline)
    cv2.putText(
        out,
        text,
        origin,
        font,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        origin,
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return out


def _save_debug_image(
    save_path: Path,
    route_mask: np.ndarray,
    primary_actor_mask: np.ndarray,
    interaction_mask: np.ndarray,
    secondary_actor_mask: np.ndarray,
    ego_future_mask: np.ndarray,
    valid: bool,
    invalid_reason: str,
    frame_name: str,
    primary_frames: int,
    secondary_frames: int,
    secondary_source: str,
    ego_center: Sequence[float],
    meters_per_pixel: float,
    cfg,) -> None:
    del valid, invalid_reason, frame_name, primary_frames, secondary_frames, secondary_source
    debug_cfg = _cfg_get(cfg, "debug", {})
    background = _debug_background(debug_cfg)

    route_panel = _colorize_mask(route_mask, (220, 145, 70), background, normalize_for_debug=True)
    ego_future_panel = _colorize_mask(ego_future_mask, (95, 190, 95), background, normalize_for_debug=True)
    primary_panel = _colorize_mask(primary_actor_mask, (75, 165, 235), background, normalize_for_debug=True)
    interaction_panel = _colorize_mask(interaction_mask, (200, 95, 215), background, normalize_for_debug=True)
    secondary_panel = _colorize_mask(secondary_actor_mask, (85, 205, 205), background, normalize_for_debug=True)

    overlay = np.full((*route_mask.shape, 3), background, dtype=np.uint8)
    overlay = _blend_mask(overlay, route_mask, (205, 125, 55), alpha=0.78)
    overlay = _blend_mask(overlay, ego_future_mask, (80, 190, 95), alpha=0.72)
    overlay = _blend_mask(overlay, primary_actor_mask, (55, 145, 235), alpha=0.90)
    overlay = _blend_mask(overlay, secondary_actor_mask, (65, 205, 205), alpha=0.85)
    overlay = _blend_mask(overlay, interaction_mask, (235, 235, 235), alpha=1.00)

    raw_panels = [route_panel, ego_future_panel, primary_panel, interaction_panel, secondary_panel, overlay]
    panel_labels = [
        "C0 Route",
        "C1 Ego",
        "C2 Primary",
        "C3 Interaction",
        "C4 Secondary",
        "Full",
    ]
    panels = []
    for panel in raw_panels:
        panel = _crop_debug_roi(panel, ego_center, meters_per_pixel, debug_cfg)
        panels.append(panel)

    panel_width = max(_cfg_int(debug_cfg, "panel_width", 320), 160)
    panel_gap = max(_cfg_int(debug_cfg, "panel_gap_px", 12), 0)
    panels = [_resize_panel(panel, panel_width) for panel in panels]
    panels = [
        _add_panel_corner_label(panel, panel_label, debug_cfg)
        for panel, panel_label in zip(panels, panel_labels)
    ]
    panel_height = max(panel.shape[0] for panel in panels)

    normalized = []
    for panel in panels:
        pad_bottom = panel_height - panel.shape[0]
        panel = cv2.copyMakeBorder(
            panel,
            0,
            pad_bottom,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=background,
        )
        panel = _add_panel_border(panel, debug_cfg)
        normalized.append(panel)

    panel_total_height = max(panel.shape[0] for panel in normalized)
    panel_total_width = max(panel.shape[1] for panel in normalized)

    # 修改20260726：debug 图改为三行两列，最后一个面板作为 overlay。
    grid_rows = [normalized[0:2], normalized[2:4], normalized[4:6]]
    row_images = []
    h_spacer = np.full((panel_total_height, panel_gap, 3), background, dtype=np.uint8) if panel_gap > 0 else None
    v_spacer = np.full((panel_gap, 2 * panel_total_width + (panel_gap if panel_gap > 0 else 0), 3), background, dtype=np.uint8) if panel_gap > 0 else None
    for row in grid_rows:
        padded = []
        for panel in row:
            bottom = panel_total_height - panel.shape[0]
            right = panel_total_width - panel.shape[1]
            if bottom > 0 or right > 0:
                panel = cv2.copyMakeBorder(
                    panel,
                    0,
                    bottom,
                    0,
                    right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=background,
                )
            padded.append(panel)
        if h_spacer is not None:
            row_image = np.concatenate([padded[0], h_spacer, padded[1]], axis=1)
        else:
            row_image = np.concatenate(padded, axis=1)
        row_images.append(row_image)

    if v_spacer is not None:
        composite = np.concatenate([row_images[0], v_spacer, row_images[1], v_spacer, row_images[2]], axis=0)
    else:
        composite = np.concatenate(row_images, axis=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), composite)

    if _cfg_bool(debug_cfg, "save_individual_channels", False):
        stem = save_path.stem
        individual = [
            ("c0_route_ego_footprints", route_mask, (220, 145, 70), "C0 Route"),
            ("c1_future_ego", ego_future_mask, (95, 190, 95), "C1 Ego"),
            ("c2_primary_actor", primary_actor_mask, (75, 165, 235), "C2 Primary"),
            ("c3_primary_interaction", interaction_mask, (200, 95, 215), "C3 Interaction"),
            ("c4_secondary_actor", secondary_actor_mask, (85, 205, 205), "C4 Secondary"),
        ]
        for suffix, mask, color, panel_label in individual:
            image = _colorize_mask(mask, color, background, normalize_for_debug=True)
            image = _crop_debug_roi(image, ego_center, meters_per_pixel, debug_cfg)
            image = _resize_panel(image, panel_width)
            image = _add_panel_corner_label(image, panel_label, debug_cfg)
            image = _add_panel_border(image, debug_cfg)
            cv2.imwrite(str(save_path.parent / f"{stem}_{suffix}.png"), image)

def _extract_label_inputs(lg_label: Dict, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = int(cfg.horizon.num_future_waypoints)
    reference_count = int(cfg.horizon.reference_route_points)

    supervision = lg_label.get("supervision", {}) or {}
    reference = lg_label.get("reference", {}) or {}

    waypoints = _fit_point_count(
        supervision.get("risk_planned_waypoints", []),
        horizon,
    )
    yaws = _load_waypoint_yaws(lg_label, waypoints, horizon)
    route_points = _resample_reference_route(
        reference.get("selected_reference_route", []),
        reference_count,
        _cfg_float(_cfg_get(cfg, "horizon", {}), "reference_route_spacing_m", 1.0),
    )
    return route_points, waypoints, yaws


def process_one_frame(route_dir: Path, frame_name: str, cfg) -> bool:
    frame_name = _normalize_frame_name(frame_name)
    lg_path = route_dir / str(cfg.paths.lg_folder) / f"{frame_name}.json.gz"
    costmap_path = route_dir / str(cfg.paths.costmap_folder) / f"{frame_name}.npy"
    meta_path = route_dir / str(cfg.paths.bev_meta_folder) / f"{frame_name}.json.gz"

    if not lg_path.exists():
        LOGGER.warning(f"[Skip] Missing LG label: {lg_path}")
        return False
    if not costmap_path.exists():
        LOGGER.warning(f"[Skip] Missing costmap: {costmap_path}")
        return False

    lg_label = load_json_gz(lg_path)
    costmap = load_costmap(costmap_path)
    meta = load_json_gz(meta_path) if meta_path.exists() else {}
    meters_per_pixel = get_meters_per_pixel(
        meta,
        float(cfg.paths.default_pixels_per_meter),
    )
    ego_center = get_ego_center(meta, costmap.shape)
    shape = tuple(costmap.shape[:2])

    (
        route_mask,
        ego_future_union,
        primary_actor_union,
        interaction_union,
        secondary_actor_union,
    ) = _empty_result(shape)

    valid = True
    invalid_reason = ""
    primary_actor_frames_found = 0
    secondary_actor_frames_found = 0
    critical_actor_id = None
    secondary_actor_id = None
    secondary_actor_source = "none"

    supervision = lg_label.get("supervision", {}) or {}
    if not bool(supervision.get("risk_label_valid", False)):
        valid = False
        invalid_reason = "risk_label_invalid"

    route_points, planned_waypoints, planned_yaws = _extract_label_inputs(lg_label, cfg)
    horizon = int(cfg.horizon.num_future_waypoints)
    reference_count = int(cfg.horizon.reference_route_points)

    if len(route_points) == 0:
        valid = False
        invalid_reason = invalid_reason or "missing_selected_reference_route"
    if len(planned_waypoints) != horizon:
        valid = False
        invalid_reason = invalid_reason or "invalid_risk_planned_waypoint_count"
    if len(planned_yaws) != horizon:
        valid = False
        invalid_reason = invalid_reason or "invalid_risk_planned_yaw_count"

    ego_cfg = _cfg_get(cfg, "vehicle", {})
    grid_cfg = _cfg_get(cfg, "grid", {})
    ego_half_length = _cfg_float(ego_cfg, "ego_half_length_m", 2.44619083404541)
    ego_half_width = _cfg_float(ego_cfg, "ego_half_width_m", 0.9183566570281982)
    route_footprint_margin = max(
        _cfg_float(grid_cfg, "reference_route_footprint_margin_m", 0.0),
        0.0,
    )
    ego_margin = max(_cfg_float(grid_cfg, "ego_footprint_margin_m", 0.0), 0.0)
    interaction_margin = max(_cfg_float(grid_cfg, "interaction_margin_m", 0.35), 0.0)
    actor_margin = max(_cfg_float(grid_cfg, "actor_footprint_margin_m", 0.0), 0.0)

    if len(route_points) > 0:
        route_mask = _draw_reference_footprint_occupancy(
            shape=shape,
            route_points=route_points,
            ego_half_length_m=ego_half_length,
            ego_half_width_m=ego_half_width,
            footprint_margin_m=route_footprint_margin,
            ego_center=ego_center,
            meters_per_pixel=meters_per_pixel,
        )

    causal_analysis = lg_label.get("causal_analysis", {}) or {}
    has_causal_actor = bool(causal_analysis.get("has_causal_object", False))
    causal_actor = causal_analysis.get("causal_object", {}) or {}
    if has_causal_actor:
        critical_actor_id = causal_actor.get("id", None)
        if critical_actor_id is None:
            valid = False
            invalid_reason = invalid_reason or "causal_actor_without_id"

    secondary_actor, secondary_actor_source = _select_secondary_actor(
        lg_label=lg_label,
        primary_actor_id=critical_actor_id,
        cfg=cfg,
    )
    has_secondary_actor = _actor_exists(secondary_actor)
    if has_secondary_actor:
        secondary_actor_id = secondary_actor.get("id", None)
        if secondary_actor_id is None:
            has_secondary_actor = False
            secondary_actor_source = "secondary_actor_without_id"

    actor_timelines = {}
    if (
        (has_causal_actor and critical_actor_id is not None)
        or (has_secondary_actor and secondary_actor_id is not None)
    ):
        actor_timelines = load_future_actor_timelines(
            route_dir=route_dir,
            frame_name=frame_name,
            cfg=cfg,
        )

    if len(planned_waypoints) == horizon and len(planned_yaws) == horizon:
        for step in range(1, horizon + 1):
            waypoint = planned_waypoints[step - 1]
            yaw = float(planned_yaws[step - 1])

            ego_mask = np.zeros(shape, dtype=np.uint8)
            _fill_obb(
                ego_mask,
                center_xy=waypoint,
                yaw_rad=yaw,
                half_length_m=ego_half_length + ego_margin,
                half_width_m=ego_half_width + ego_margin,
                ego_center=ego_center,
                meters_per_pixel=meters_per_pixel,
            )
            ego_future_union += ego_mask.astype(np.float32)

            if has_causal_actor and critical_actor_id is not None:
                primary_actor = _find_actor(
                    actor_timelines.get(step, []),
                    critical_actor_id,
                )
                if primary_actor is not None:
                    primary_actor_frames_found += 1
                    primary_mask = np.zeros(shape, dtype=np.uint8)
                    _fill_obb(
                        primary_mask,
                        center_xy=[
                            float(primary_actor.get("x_m", 0.0)),
                            float(primary_actor.get("y_m", 0.0)),
                        ],
                        yaw_rad=float(primary_actor.get("yaw_rad", 0.0)),
                        half_length_m=float(primary_actor.get("half_length_m", 1.0)) + actor_margin,
                        half_width_m=float(primary_actor.get("half_width_m", 0.5)) + actor_margin,
                        ego_center=ego_center,
                        meters_per_pixel=meters_per_pixel,
                    )
                    primary_actor_union += primary_mask.astype(np.float32)

                    # Interaction is computed at the same future step before the
                    # ten per-step interaction masks are merged. This avoids false
                    # conflicts from intersecting independently aggregated paths.
                    ego_interaction_mask = np.zeros(shape, dtype=np.uint8)
                    _fill_obb(
                        ego_interaction_mask,
                        center_xy=waypoint,
                        yaw_rad=yaw,
                        half_length_m=ego_half_length + ego_margin + interaction_margin,
                        half_width_m=ego_half_width + ego_margin + interaction_margin,
                        ego_center=ego_center,
                        meters_per_pixel=meters_per_pixel,
                    )
                    interaction_step = (
                        (ego_interaction_mask > 0) & (primary_mask > 0)
                    ).astype(np.uint8)
                    interaction_union += interaction_step.astype(np.float32)

            if has_secondary_actor and secondary_actor_id is not None:
                secondary_future_actor = _find_actor(
                    actor_timelines.get(step, []),
                    secondary_actor_id,
                )
                if secondary_future_actor is not None:
                    secondary_actor_frames_found += 1
                    secondary_mask = np.zeros(shape, dtype=np.uint8)
                    _fill_obb(
                        secondary_mask,
                        center_xy=[
                            float(secondary_future_actor.get("x_m", 0.0)),
                            float(secondary_future_actor.get("y_m", 0.0)),
                        ],
                        yaw_rad=float(secondary_future_actor.get("yaw_rad", 0.0)),
                        half_length_m=float(secondary_future_actor.get("half_length_m", 1.0)) + actor_margin,
                        half_width_m=float(secondary_future_actor.get("half_width_m", 0.5)) + actor_margin,
                        ego_center=ego_center,
                        meters_per_pixel=meters_per_pixel,
                    )
                    secondary_actor_union += secondary_mask.astype(np.float32)

    # Every one of the ten future frames contributes exactly 1 / horizon.
    # Missing observations contribute zero rather than changing the weights of
    # the remaining frames.
    if horizon > 0:
        primary_actor_union = np.clip(primary_actor_union / float(horizon), 0.0, 1.0)
        interaction_union = np.clip(interaction_union / float(horizon), 0.0, 1.0)
        secondary_actor_union = np.clip(secondary_actor_union / float(horizon), 0.0, 1.0)
        ego_future_union = np.clip(ego_future_union / float(horizon), 0.0, 1.0)

    if (
        has_causal_actor
        and critical_actor_id is not None
        and primary_actor_frames_found == 0
    ):
        valid = False
        invalid_reason = invalid_reason or "causal_actor_missing_in_all_future_frames"

    # A missing secondary actor does not invalidate the four core supervision
    # channels. Channel 4 simply remains zero and its metadata records the cause.
    if has_secondary_actor and secondary_actor_frames_found == 0:
        secondary_actor_source = f"{secondary_actor_source}:missing_in_all_future_frames"

    grid = np.stack(
        [
            route_mask,           # selected_reference_route 上的自车 footprint 占用
            ego_future_union,     # 自车未来 10 帧的 footprint 占用
            primary_actor_union,  # 主要因果 actor 未来 10 帧的 footprint 占用
            interaction_union,    # 自车与主要因果 actor 逐帧对齐后的未来交互区域
            secondary_actor_union,# 次要 actor 未来 10 帧的 footprint 占用
        ],
        axis=0,
    ).astype(np.float32)

    output_cfg = _cfg_get(cfg, "output", {})
    save_invalid = _cfg_bool(output_cfg, "save_invalid_labels", True)
    save_npz = _cfg_bool(output_cfg, "save_npz", True)
    if save_npz and (valid or save_invalid):
        output_path = route_dir / str(cfg.paths.output_folder) / f"{frame_name}.npz"
        _save_npz(
            output_path=output_path,
            grid=grid,
            valid=valid,
            invalid_reason=invalid_reason,
            frame_name=frame_name,
            critical_actor_id=critical_actor_id,
            has_causal_actor=has_causal_actor,
            primary_actor_future_frames_found=primary_actor_frames_found,
            secondary_actor_id=secondary_actor_id,
            has_secondary_actor=has_secondary_actor,
            secondary_actor_source=secondary_actor_source,
            secondary_actor_future_frames_found=secondary_actor_frames_found,
            reference_route_points_used=min(len(route_points), reference_count),
            cfg=cfg,
            meters_per_pixel=meters_per_pixel,
            ego_center=ego_center,
        )

    debug_cfg = _cfg_get(cfg, "debug", {})
    if _cfg_bool(debug_cfg, "save_debug", False):
        debug_path = (
            route_dir
            / str(cfg.paths.debug_folder)
            / f"{frame_name}_future_interaction.png"
        )
        _save_debug_image(
            save_path=debug_path,
            route_mask=route_mask,
            primary_actor_mask=primary_actor_union,
            interaction_mask=interaction_union,
            secondary_actor_mask=secondary_actor_union,
            ego_future_mask=ego_future_union,
            valid=valid,
            invalid_reason=invalid_reason,
            frame_name=frame_name,
            primary_frames=primary_actor_frames_found,
            secondary_frames=secondary_actor_frames_found,
            secondary_source=secondary_actor_source,
            ego_center=ego_center,
            meters_per_pixel=meters_per_pixel,
            cfg=cfg,
        )

    if bool(cfg.run.verbose):
        LOGGER.info(
            f"[OK] {route_dir.name}/{frame_name}: valid={valid} "
            f"primary={has_causal_actor} primary_frames={primary_actor_frames_found}/{horizon} "
            f"secondary={has_secondary_actor} secondary_frames={secondary_actor_frames_found}/{horizon} "
            f"secondary_source={secondary_actor_source} "
            f"route_mass={float(route_mask.sum()):.1f} "
            f"primary_mass={float(primary_actor_union.sum()):.1f} "
            f"interaction_mass={float(interaction_union.sum()):.1f} "
            f"secondary_mass={float(secondary_actor_union.sum()):.1f} "
            f"reason={invalid_reason or 'none'}"
        )
    return bool(valid or (save_npz and save_invalid))


def process_route_dir(route_dir: Path, cfg) -> Tuple[int, int]:
    frames = _list_frame_names(route_dir, cfg)
    if not frames:
        LOGGER.warning(f"[Skip] No matched LG/costmap frames in route: {route_dir}")
        return 0, 0

    ok = 0
    raise_on_error = bool(_cfg_get(_cfg_get(cfg, "run", {}), "raise_on_error", False))
    for frame_name in frames:
        try:
            if process_one_frame(route_dir, frame_name, cfg):
                ok += 1
        except Exception as exc:
            LOGGER.exception(
                f"[Error] route={route_dir}, frame={frame_name}, error={exc}"
            )
            if raise_on_error:
                raise

    LOGGER.info(f"[Route Done] {route_dir}: {ok}/{len(frames)}")
    return ok, len(frames)


def process_dataset(cfg) -> Tuple[int, int]:
    input_path = _cfg_get(_cfg_get(cfg, "run", {}), "input", None)
    if input_path in [None, "", "None"]:
        raise ValueError("Missing config field: run.input")

    root = Path(str(input_path)).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    recursive = bool(_cfg_get(_cfg_get(cfg, "run", {}), "recursive", False))
    route_dirs = find_route_dirs(root, cfg) if recursive else [root]
    LOGGER.info(f"[Info] Found {len(route_dirs)} route dirs; recursive={recursive}")

    total_ok = 0
    total_frames = 0
    for route_dir in route_dirs:
        ok, total = process_route_dir(route_dir, cfg)
        total_ok += ok
        total_frames += total

    LOGGER.info(f"[All Done] {total_ok}/{total_frames} frames processed.")
    return total_ok, total_frames


def main() -> None:
    cfg_path = resolve_config_path()
    cfg = load_yaml(str(cfg_path))

    if INPUT_OVERRIDE is not None:
        cfg.run.input = INPUT_OVERRIDE
    if FRAME_OVERRIDE is not None:
        cfg.run.frame = FRAME_OVERRIDE

    log_file = None
    if bool(cfg.run.save_log):
        log_root = Path(str(cfg.run.input)).expanduser().resolve()
        log_file = log_root / str(cfg.paths.log_folder) / "future_interaction_grid.log"

    setup_logger(verbose=bool(cfg.run.verbose), log_file=log_file)
    LOGGER.info(f"[Config] {cfg_path}")
    LOGGER.info(f"[Input] {cfg.run.input}")
    process_dataset(cfg)


if __name__ == "__main__":
    main()
