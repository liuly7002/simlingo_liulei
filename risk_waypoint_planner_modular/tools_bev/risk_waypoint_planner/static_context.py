# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .costmap_utils import local_points_to_pixels
from .qa_annotation import safe_float


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}

    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def _as_bool(mask: Optional[np.ndarray], shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if mask is None:
        if shape is None:
            raise ValueError("shape must be provided when mask is None.")
        return np.zeros(shape, dtype=bool)

    return mask.astype(np.uint8) > 0


def _infer_shape(mask_dict: Dict[str, np.ndarray]) -> Optional[Tuple[int, int]]:
    for v in mask_dict.values():
        if v.ndim == 2:
            return v.shape
    return None


def _static_folder_name(args) -> str:
    suffix = "_augmented" if bool(getattr(args, "augmented", False)) else ""
    return f"bev_static_masks{suffix}"


def _load_static_masks(route_dir: Path, frame_name: str, args) -> Dict[str, np.ndarray]:
    static_path = route_dir / _static_folder_name(args) / f"{frame_name}.npz"
    return _load_npz(static_path)


def _get_candidate_path_local(scored_rollout: Dict) -> np.ndarray:
    """
    Prefer actual rollout dense trajectory; fallback to sparse waypoints/reference route.
    The points are expected to be ego-local xy coordinates.
    """
    rollout = scored_rollout.get("rollout", {}) or {}

    for key in ["dense_xy", "waypoints"]:
        if key in rollout:
            pts = np.asarray(rollout.get(key, []), dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] >= 1 and pts.shape[1] >= 2:
                return pts[:, :2]

    if "reference_route" in scored_rollout:
        pts = np.asarray(scored_rollout.get("reference_route", []), dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 1 and pts.shape[1] >= 2:
            return pts[:, :2]

    return np.zeros((0, 2), dtype=np.float32)


def _make_corridor_mask(
    points_local: np.ndarray,
    shape: Tuple[int, int],
    ego_center: Tuple[float, float],
    meters_per_pixel: float,
    radius_m: float,
) -> np.ndarray:
    """
    Rasterize a tube around candidate trajectory.
    """
    mask = np.zeros(shape, dtype=np.uint8)

    if points_local is None or len(points_local) == 0:
        return mask.astype(bool)

    pix = local_points_to_pixels(points_local, ego_center, meters_per_pixel)
    pix = np.asarray(pix, dtype=np.int32)

    radius_px = max(1, int(round(float(radius_m) / float(meters_per_pixel))))
    thickness = max(1, 2 * radius_px + 1)

    if len(pix) == 1:
        x, y = int(pix[0, 0]), int(pix[0, 1])
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            cv2.circle(mask, (x, y), radius_px, 1, thickness=-1)
        return mask.astype(bool)

    for i in range(len(pix) - 1):
        x1, y1 = int(pix[i, 0]), int(pix[i, 1])
        x2, y2 = int(pix[i + 1, 0]), int(pix[i + 1, 1])
        cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=thickness)

    return mask.astype(bool)


def _ratio(mask: np.ndarray, corridor: np.ndarray) -> float:
    denom = int(corridor.sum())
    if denom <= 0:
        return 0.0
    return float((mask & corridor).sum()) / float(denom)


def _dominant_static_type(stats: Dict) -> str:
    """
    Decide the dominant static limitation.
    Thresholds are intentionally conservative.
    """
    total = int(stats.get("sample_pixels", 0))
    if total <= 0:
        return "unknown"

    sidewalk_ratio = float(stats.get("sidewalk_ratio", 0.0))
    nonroad_ratio = float(stats.get("nonroad_ratio", 0.0))
    road_ratio = float(stats.get("road_ratio", 0.0))
    lane_solid_ratio = float(stats.get("lane_solid_ratio", 0.0))

    if sidewalk_ratio >= 0.15:
        return "sidewalk"

    if nonroad_ratio >= 0.35:
        return "non_drivable_or_boundary"

    if lane_solid_ratio >= 0.15:
        return "solid_lane_marking"

    if road_ratio >= 0.70:
        return "drivable_road"

    return "mixed_or_uncertain"


def _side_from_behavior(behavior_name: str) -> str:
    if behavior_name == "left_nudge":
        return "left"
    if behavior_name == "right_nudge":
        return "right"
    return "route"


def _side_zh(side: str) -> str:
    if side == "left":
        return "左侧"
    if side == "right":
        return "右侧"
    return "当前路线"


def _side_en(side: str) -> str:
    if side == "left":
        return "left-side"
    if side == "right":
        return "right-side"
    return "current-route"


def _static_description_zh(side: str, dominant: str, stats: Dict) -> str:
    side_text = _side_zh(side)

    if dominant == "sidewalk":
        return f"{side_text}候选轨迹会进入或贴近人行道区域，这不是车辆稳定通行空间。"

    if dominant == "non_drivable_or_boundary":
        return f"{side_text}候选轨迹会靠近非可行驶区域或道路边界，不适合作为稳定绕行空间。"

    if dominant == "solid_lane_marking":
        return f"{side_text}候选轨迹会穿过或贴近实线车道标记，作为绕行选择不够稳妥。"

    if dominant == "drivable_road":
        return f"{side_text}候选轨迹主要位于可行驶道路区域内。"

    if dominant == "mixed_or_uncertain":
        return f"{side_text}候选轨迹经过的道路属性不够稳定，需要谨慎判断是否适合通行。"

    return f"{side_text}候选轨迹的静态道路属性不明确。"


def _static_description_en(side: str, dominant: str, stats: Dict) -> str:
    side_text = _side_en(side)

    if dominant == "sidewalk":
        return f"The {side_text} candidate trajectory enters or stays close to the sidewalk area, which is not a stable driving space for the vehicle."

    if dominant == "non_drivable_or_boundary":
        return f"The {side_text} candidate trajectory is close to a non-drivable area or road boundary, so it is not suitable as a stable detour space."

    if dominant == "solid_lane_marking":
        return f"The {side_text} candidate trajectory crosses or stays close to a solid lane marking, making it less reliable as a detour."

    if dominant == "drivable_road":
        return f"The {side_text} candidate trajectory is mainly within the drivable road area."

    if dominant == "mixed_or_uncertain":
        return f"The static road context along the {side_text} candidate trajectory is mixed, so its drivability should be treated cautiously."

    return f"The static road context along the {side_text} candidate trajectory is unclear."


def _compute_candidate_static_context(
    scored_rollout: Dict,
    static_masks: Dict[str, np.ndarray],
    shape: Tuple[int, int],
    ego_center: Tuple[float, float],
    meters_per_pixel: float,
    corridor_radius_m: float,
) -> Dict:
    info = scored_rollout.get("info", {}) or {}
    behavior_name = str(info.get("behavior_name", "unknown"))
    side = _side_from_behavior(behavior_name)

    road = _as_bool(static_masks.get("road"), shape)
    sidewalk = _as_bool(static_masks.get("sidewalk"), shape)
    lane_all = _as_bool(static_masks.get("lane_all"), shape)
    lane_broken = _as_bool(static_masks.get("lane_broken"), shape)

    if "lane_solid" in static_masks:
        lane_solid = _as_bool(static_masks.get("lane_solid"), shape)
    else:
        lane_solid = lane_all & (~lane_broken)

    pts = _get_candidate_path_local(scored_rollout)
    corridor = _make_corridor_mask(
        points_local=pts,
        shape=shape,
        ego_center=ego_center,
        meters_per_pixel=meters_per_pixel,
        radius_m=corridor_radius_m,
    )

    sample_pixels = int(corridor.sum())

    stats = {
        "sample_pixels": sample_pixels,
        "road_ratio": round(_ratio(road, corridor), 4),
        "sidewalk_ratio": round(_ratio(sidewalk, corridor), 4),
        "nonroad_ratio": round(_ratio(~road, corridor), 4),
        "lane_all_ratio": round(_ratio(lane_all, corridor), 4),
        "lane_broken_ratio": round(_ratio(lane_broken, corridor), 4),
        "lane_solid_ratio": round(_ratio(lane_solid, corridor), 4),
    }

    dominant = _dominant_static_type(stats)

    return {
        "behavior_name": behavior_name,
        "side": side,
        "dominant_static_type": dominant,
        "stats": stats,
        "description_zh": _static_description_zh(side, dominant, stats),
        "description_en": _static_description_en(side, dominant, stats),
    }


def build_static_candidate_context(
    route_dir: Path,
    frame_name: str,
    scored_rollouts: List[Dict],
    ego_center: Tuple[float, float],
    meters_per_pixel: float,
    args,
) -> Dict:
    """
    Build static semantic context for every candidate trajectory.

    It reads bev_static_masks/*.npz generated/used by generate_costmap_from_masks.py.
    This module only affects language annotation, not planning or scoring.
    """
    static_masks = _load_static_masks(route_dir, frame_name, args)
    shape = _infer_shape(static_masks)

    if not static_masks or shape is None:
        return {
            "enabled": False,
            "source": "bev_static_masks",
            "frame": frame_name,
            "reason": "missing_static_masks",
            "candidates": {},
        }

    corridor_radius_m = float(getattr(args, "static_context_corridor_radius_m", 1.0))

    candidates = {}
    for i, scored_rollout in enumerate(scored_rollouts):
        candidates[str(i)] = _compute_candidate_static_context(
            scored_rollout=scored_rollout,
            static_masks=static_masks,
            shape=shape,
            ego_center=ego_center,
            meters_per_pixel=meters_per_pixel,
            corridor_radius_m=corridor_radius_m,
        )

    return {
        "enabled": True,
        "source": "bev_static_masks",
        "frame": frame_name,
        "corridor_radius_m": corridor_radius_m,
        "candidates": candidates,
    }