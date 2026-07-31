# -*- coding: utf-8 -*-

# 根据LG或普通Driving确认的主要关键对象在六视角中的真实投影，
# 保存参与者级空间监督所需的可见区域与投影质量。

from pathlib import Path
from typing import Dict, Optional, Tuple
import math

import numpy as np

from .visualizer import (
    _actor_cuboid_corners_ego,
    _load_surround_rgb_images,
    build_projection_matrix,
    rotation_matrix_roll_pitch_yaw_deg,
)


CAMERA_ORDER: Tuple[str, ...] = (
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
)

CENTER_SIGMA = 0.75
MIN_VISIBLE_AREA_PX = 16.0
OFF_AXIS_AREA_COMPENSATION_POWER = 2.0
_MIN_PROJECTION_DEPTH_M = 0.1
_EPS = 1e-8


def _actor_summary(actor: Optional[Dict]) -> Dict:
    actor = actor if isinstance(actor, dict) else {}
    return {
        "primary_actor_id": actor.get("id", None),
        "primary_actor_class": actor.get(
            "semantic_class",
            actor.get(
                "class",
                actor.get("raw_class", "unknown"),
            ),
        ),
    }


def _empty_camera_evidence() -> Dict[str, Dict]:
    return {
        camera_name: {
            "visible": False,
            "visible_ratio": 0.0,
            "area_ratio": 0.0,
            "off_axis_angle_deg": 0.0,
            "off_axis_compensation": 0.0,
            "corrected_area_ratio": 0.0,
            "center_score": 0.0,
            "raw_score": 0.0,
            "image_width": 0,
            "image_height": 0,
            "bbox_xyxy": None,
            "bbox_normalized_xyxy": None,
        }
        for camera_name in CAMERA_ORDER
    }


def _invalid_result(
    reason: str,
    actor: Optional[Dict] = None,
    target_source: str = (
        "lg_replanned_primary_actor_spatial_projection"
    ),
) -> Dict:
    result = {
        "camera_order": list(CAMERA_ORDER),
        "participant_spatial_attention_valid": False,
        "target_source": str(target_source),
        "invalid_reason": str(reason),
        "parameters": {
            "center_sigma": float(CENTER_SIGMA),
            "min_visible_area_px": float(MIN_VISIBLE_AREA_PX),
            "off_axis_area_compensation_power": float(
                OFF_AXIS_AREA_COMPENSATION_POWER
            ),
        },
        "per_camera_evidence": _empty_camera_evidence(),
    }
    result.update(_actor_summary(actor))
    return result


def _project_actor_to_camera(
    actor: Dict,
    image_shape,
    camera_spec: Dict,
) -> Dict:
    """返回主要关键参与者在单个相机中的空间投影证据。"""

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        return _empty_camera_evidence()[CAMERA_ORDER[0]]

    corners_ego = _actor_cuboid_corners_ego(actor)
    if (
        corners_ego.ndim != 2
        or corners_ego.shape[1] < 3
        or len(corners_ego) == 0
    ):
        return _empty_camera_evidence()[CAMERA_ORDER[0]]

    position = camera_spec.get("position", [-1.5, 0.0, 2.0])
    rotation = camera_spec.get("rotation", [0.0, 0.0, 0.0])
    fov_deg = float(camera_spec.get("fov", 110.0))

    camera_translation = np.asarray(
        position[:3],
        dtype=np.float32,
    )
    camera_to_ego = rotation_matrix_roll_pitch_yaw_deg(
        float(rotation[0]),
        float(rotation[1]),
        float(rotation[2]),
    )
    points_relative = (
        corners_ego[:, :3]
        - camera_translation[None, :]
    )
    points_camera = (
        camera_to_ego.T @ points_relative.T
    ).T

    depth = points_camera[:, 0]
    front_mask = (
        np.isfinite(depth)
        & (depth > _MIN_PROJECTION_DEPTH_M)
    )
    if int(np.sum(front_mask)) < 2:
        return _empty_camera_evidence()[CAMERA_ORDER[0]]

    intrinsic = build_projection_matrix(
        width,
        height,
        fov_deg,
    )
    right = points_camera[:, 1]
    up = points_camera[:, 2]
    pixels = np.zeros(
        (len(points_camera), 2),
        dtype=np.float32,
    )
    pixels[front_mask, 0] = (
        intrinsic[0, 0]
        * right[front_mask]
        / depth[front_mask]
        + intrinsic[0, 2]
    )
    pixels[front_mask, 1] = (
        intrinsic[1, 2]
        - intrinsic[1, 1]
        * up[front_mask]
        / depth[front_mask]
    )

    projected = pixels[front_mask]
    projected = projected[
        np.isfinite(projected).all(axis=1)
    ]
    if len(projected) < 2:
        return _empty_camera_evidence()[CAMERA_ORDER[0]]

    full_x0 = float(np.min(projected[:, 0]))
    full_y0 = float(np.min(projected[:, 1]))
    full_x1 = float(np.max(projected[:, 0]))
    full_y1 = float(np.max(projected[:, 1]))
    full_width = max(full_x1 - full_x0, 0.0)
    full_height = max(full_y1 - full_y0, 0.0)
    full_area = full_width * full_height
    if full_area <= _EPS:
        return _empty_camera_evidence()[CAMERA_ORDER[0]]

    clip_x0 = float(np.clip(full_x0, 0.0, float(width)))
    clip_y0 = float(np.clip(full_y0, 0.0, float(height)))
    clip_x1 = float(np.clip(full_x1, 0.0, float(width)))
    clip_y1 = float(np.clip(full_y1, 0.0, float(height)))
    clip_width = max(clip_x1 - clip_x0, 0.0)
    clip_height = max(clip_y1 - clip_y0, 0.0)
    clipped_area = clip_width * clip_height

    if clipped_area < MIN_VISIBLE_AREA_PX:
        return _empty_camera_evidence()[CAMERA_ORDER[0]]

    visible_ratio = float(
        np.clip(
            clipped_area / (full_area + _EPS),
            0.0,
            1.0,
        )
    )
    area_ratio = float(
        np.clip(
            clipped_area
            / max(float(width * height), 1.0),
            0.0,
            1.0,
        )
    )

    center_x = 0.5 * (clip_x0 + clip_x1)
    center_y = 0.5 * (clip_y0 + clip_y1)
    norm_dx = (
        center_x - 0.5 * float(width)
    ) / max(0.5 * float(width), 1.0)
    norm_dy = (
        center_y - 0.5 * float(height)
    ) / max(0.5 * float(height), 1.0)
    normalized_distance_sq = (
        norm_dx * norm_dx + norm_dy * norm_dy
    )
    center_score = math.exp(
        -normalized_distance_sq
        / (2.0 * CENTER_SIGMA * CENTER_SIGMA)
    )

    actor_center_camera = np.mean(points_camera, axis=0)
    center_depth = float(actor_center_camera[0])
    center_range = float(np.linalg.norm(actor_center_camera))
    if (
        not np.isfinite(center_depth)
        or not np.isfinite(center_range)
        or center_depth <= _MIN_PROJECTION_DEPTH_M
        or center_range <= _EPS
    ):
        return _empty_camera_evidence()[CAMERA_ORDER[0]]

    optical_axis_cosine = float(
        np.clip(
            center_depth / center_range,
            0.0,
            1.0,
        )
    )
    off_axis_angle_deg = math.degrees(
        math.acos(optical_axis_cosine)
    )
    off_axis_compensation = (
        optical_axis_cosine
        ** float(OFF_AXIS_AREA_COMPENSATION_POWER)
    )
    corrected_area_ratio = float(
        np.clip(
            area_ratio * off_axis_compensation,
            0.0,
            1.0,
        )
    )

    raw_score = (
        visible_ratio
        * math.sqrt(max(corrected_area_ratio, 0.0))
        * (0.5 + 0.5 * center_score)
    )

    bbox_xyxy = (
        clip_x0,
        clip_y0,
        clip_x1,
        clip_y1,
    )
    bbox_normalized_xyxy = (
        clip_x0 / float(width),
        clip_y0 / float(height),
        clip_x1 / float(width),
        clip_y1 / float(height),
    )

    return {
        "visible": bool(raw_score > 0.0),
        "visible_ratio": round(float(visible_ratio), 8),
        "area_ratio": round(float(area_ratio), 8),
        "off_axis_angle_deg": round(
            float(off_axis_angle_deg),
            8,
        ),
        "off_axis_compensation": round(
            float(off_axis_compensation),
            8,
        ),
        "corrected_area_ratio": round(
            float(corrected_area_ratio),
            8,
        ),
        "center_score": round(float(center_score), 8),
        "raw_score": round(float(raw_score), 8),
        "image_width": int(width),
        "image_height": int(height),
        "bbox_xyxy": [
            round(float(value), 4)
            for value in bbox_xyxy
        ],
        "bbox_normalized_xyxy": [
            round(float(value), 8)
            for value in bbox_normalized_xyxy
        ],
    }


def build_camera_attention_supervision(
    route_dir: Path,
    frame_name: str,
    cfg,
    causal_analysis: Optional[Dict] = None,
    *,
    primary_actor: Optional[Dict] = None,
    target_source: str = (
        "lg_replanned_primary_actor_spatial_projection"
    ),
) -> Dict:
    """
    构建统一的主要关键参与者六视角空间投影监督。

    函数名保留以兼容LG与普通Driving现有调用位置；输出不再是六维相机权重，
    而是用于生成视觉token级空间目标的投影框和质量证据。
    """

    actor = primary_actor
    if actor is None and isinstance(causal_analysis, dict):
        if not bool(
            causal_analysis.get(
                "has_causal_object",
                False,
            )
        ):
            return _invalid_result(
                "no_verified_causal_object",
                target_source=target_source,
            )
        actor = causal_analysis.get("causal_object") or {}

    if (
        not isinstance(actor, dict)
        or not bool(actor.get("exists", False))
    ):
        return _invalid_result(
            "missing_verified_primary_actor",
            actor,
            target_source,
        )
    if (
        actor.get("x_m", None) is None
        or actor.get("y_m", None) is None
    ):
        return _invalid_result(
            "primary_actor_missing_ego_local_position",
            actor,
            target_source,
        )

    try:
        surround_images, camera_specs = (
            _load_surround_rgb_images(
                route_dir=route_dir,
                frame_name=frame_name,
                cfg=cfg,
            )
        )
        if surround_images is None or not camera_specs:
            return _invalid_result(
                "incomplete_six_view_images_or_calibration",
                actor,
                target_source,
            )

        per_camera_evidence = {}
        raw_scores = []
        for camera_name in CAMERA_ORDER:
            image = surround_images.get(camera_name, None)
            camera_spec = camera_specs.get(camera_name, None)
            if (
                image is None
                or not isinstance(camera_spec, dict)
            ):
                evidence = _empty_camera_evidence()[
                    camera_name
                ]
            else:
                evidence = _project_actor_to_camera(
                    actor=actor,
                    image_shape=image.shape[:2],
                    camera_spec=camera_spec,
                )
            per_camera_evidence[camera_name] = evidence
            raw_scores.append(
                float(evidence.get("raw_score", 0.0))
            )

        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        raw_sum = float(np.sum(raw_scores))
        if not np.isfinite(raw_sum) or raw_sum <= _EPS:
            result = _invalid_result(
                "primary_actor_not_visible_in_any_camera",
                actor,
                target_source,
            )
            result["per_camera_evidence"] = (
                per_camera_evidence
            )
            return result

        result = {
            "camera_order": list(CAMERA_ORDER),
            "participant_spatial_attention_valid": True,
            "target_source": str(target_source),
            "invalid_reason": None,
            "parameters": {
                "center_sigma": float(CENTER_SIGMA),
                "min_visible_area_px": float(
                    MIN_VISIBLE_AREA_PX
                ),
                "off_axis_area_compensation_power": float(
                    OFF_AXIS_AREA_COMPENSATION_POWER
                ),
            },
            "per_camera_evidence": per_camera_evidence,
        }
        result.update(_actor_summary(actor))
        return result
    except Exception as exc:
        return _invalid_result(
            "participant_spatial_projection_error:"
            f"{type(exc).__name__}:{exc}",
            actor,
            target_source,
        )
