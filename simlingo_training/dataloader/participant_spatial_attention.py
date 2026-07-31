# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import numpy as np


CAMERA_ORDER: Tuple[str, ...] = (
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
)
NUM_PATCHES_PER_CAMERA = 2
TOKEN_GRID_HEIGHT = 4
TOKEN_GRID_WIDTH = 8
TOKENS_PER_CAMERA = (
    NUM_PATCHES_PER_CAMERA
    * TOKEN_GRID_HEIGHT
    * TOKEN_GRID_WIDTH
)
LABEL_SMOOTHING = 0.01


def _choose_two_patch_layout(
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """复现当前两patch动态切图的横向或纵向布局选择。"""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    aspect_ratio = float(image_width) / float(image_height)
    candidates = (
        (2, 1),
        (1, 2),
    )
    return min(
        candidates,
        key=lambda layout: abs(
            aspect_ratio
            - float(layout[0]) / float(layout[1])
        ),
    )


def _bbox_to_patch_token_weights(
    bbox_normalized_xyxy: np.ndarray,
    patch_layout: Tuple[int, int],
) -> np.ndarray:
    """将归一化图像bbox转换为与2×4×8视觉token严格对齐的软占用。"""

    columns, rows = patch_layout
    if columns * rows != NUM_PATCHES_PER_CAMERA:
        raise ValueError(
            "Participant attention requires exactly two visual patches "
            f"per camera, but received layout={patch_layout}."
        )

    x0, y0, x1, y1 = (
        float(value)
        for value in bbox_normalized_xyxy.tolist()
    )
    x0 = float(np.clip(x0, 0.0, 1.0))
    y0 = float(np.clip(y0, 0.0, 1.0))
    x1 = float(np.clip(x1, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((TOKENS_PER_CAMERA,), dtype=np.float32)

    canvas_bbox = np.asarray(
        (
            x0 * columns,
            y0 * rows,
            x1 * columns,
            y1 * rows,
        ),
        dtype=np.float64,
    )

    patch_maps = []
    cell_width = 1.0 / float(TOKEN_GRID_WIDTH)
    cell_height = 1.0 / float(TOKEN_GRID_HEIGHT)
    cell_area = cell_width * cell_height

    for patch_index in range(NUM_PATCHES_PER_CAMERA):
        patch_column = patch_index % columns
        patch_row = patch_index // columns

        local_x0 = max(canvas_bbox[0] - patch_column, 0.0)
        local_y0 = max(canvas_bbox[1] - patch_row, 0.0)
        local_x1 = min(canvas_bbox[2] - patch_column, 1.0)
        local_y1 = min(canvas_bbox[3] - patch_row, 1.0)

        token_map = np.zeros(
            (TOKEN_GRID_HEIGHT, TOKEN_GRID_WIDTH),
            dtype=np.float64,
        )
        if local_x1 > local_x0 and local_y1 > local_y0:
            for row_index in range(TOKEN_GRID_HEIGHT):
                cell_y0 = row_index * cell_height
                cell_y1 = (row_index + 1) * cell_height
                overlap_height = max(
                    min(local_y1, cell_y1)
                    - max(local_y0, cell_y0),
                    0.0,
                )
                if overlap_height <= 0.0:
                    continue

                for column_index in range(TOKEN_GRID_WIDTH):
                    cell_x0 = column_index * cell_width
                    cell_x1 = (column_index + 1) * cell_width
                    overlap_width = max(
                        min(local_x1, cell_x1)
                        - max(local_x0, cell_x0),
                        0.0,
                    )
                    if overlap_width <= 0.0:
                        continue
                    token_map[row_index, column_index] = (
                        overlap_width * overlap_height / cell_area
                    )

        patch_maps.append(token_map.reshape(-1))

    return np.concatenate(patch_maps).astype(np.float32)


def _apply_training_crop(
    bbox_normalized_xyxy: np.ndarray,
    image_width: int,
    image_height: int,
    cut_bottom_quarter: bool,
) -> Tuple[np.ndarray, int, int]:
    """复现SurroundDatasetMixin中的底部裁剪并重新归一化bbox。"""

    bbox = np.asarray(
        bbox_normalized_xyxy,
        dtype=np.float64,
    ).copy()
    if bbox.shape != (4,):
        raise ValueError(
            "Normalized participant bbox must contain four values."
        )

    if not cut_bottom_quarter:
        return bbox, int(image_width), int(image_height)

    crop_height = int(
        image_height - (image_height * 4.8) // 16
    )
    crop_height = max(min(crop_height, image_height), 1)
    crop_ratio = float(crop_height) / float(image_height)

    bbox[1] = np.clip(bbox[1], 0.0, crop_ratio)
    bbox[3] = np.clip(bbox[3], 0.0, crop_ratio)
    bbox[1] /= crop_ratio
    bbox[3] /= crop_ratio

    return bbox, int(image_width), int(crop_height)


def extract_participant_spatial_attention_supervision(
    payload: Dict,
    *,
    cut_bottom_quarter: bool,
    use_global_img: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    从标签中的六视角参与者投影框生成[B前]的6×64视觉token软标签。

    返回：
        target: [6,64]，全局概率和为1
        valid: 是否存在可用的主要关键参与者视觉投影
    """

    empty_target = np.zeros(
        (len(CAMERA_ORDER), TOKENS_PER_CAMERA),
        dtype=np.float32,
    )

    if use_global_img:
        raise ValueError(
            "Participant spatial attention currently assumes exactly two "
            "patches per camera; use_global_img must remain False."
        )

    visual_grounding = payload.get("visual_grounding", {})
    if not isinstance(visual_grounding, dict):
        return empty_target, False
    if not bool(
        visual_grounding.get(
            "participant_spatial_attention_valid",
            False,
        )
    ):
        return empty_target, False
    if tuple(visual_grounding.get("camera_order", [])) != CAMERA_ORDER:
        return empty_target, False

    per_camera_evidence = visual_grounding.get(
        "per_camera_evidence",
        {},
    )
    if not isinstance(per_camera_evidence, dict):
        return empty_target, False

    target = np.zeros_like(empty_target, dtype=np.float64)

    for camera_index, camera_name in enumerate(CAMERA_ORDER):
        evidence = per_camera_evidence.get(camera_name, {})
        if not isinstance(evidence, dict):
            continue
        if not bool(evidence.get("visible", False)):
            continue

        bbox = np.asarray(
            evidence.get("bbox_normalized_xyxy", []),
            dtype=np.float64,
        )
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            continue

        image_width = int(evidence.get("image_width", 0))
        image_height = int(evidence.get("image_height", 0))
        if image_width <= 0 or image_height <= 0:
            continue

        bbox, processed_width, processed_height = (
            _apply_training_crop(
                bbox,
                image_width=image_width,
                image_height=image_height,
                cut_bottom_quarter=cut_bottom_quarter,
            )
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        patch_layout = _choose_two_patch_layout(
            processed_width,
            processed_height,
        )
        token_weights = _bbox_to_patch_token_weights(
            bbox,
            patch_layout,
        ).astype(np.float64)

        evidence_weight = float(
            evidence.get("raw_score", 0.0)
        )
        if not np.isfinite(evidence_weight) or evidence_weight <= 0.0:
            continue

        target[camera_index] = token_weights * evidence_weight

    target_sum = float(target.sum())
    if not np.isfinite(target_sum) or target_sum <= 0.0:
        return empty_target, False

    target /= target_sum
    target = (
        (1.0 - LABEL_SMOOTHING) * target
        + LABEL_SMOOTHING / float(target.size)
    )
    target /= max(float(target.sum()), 1e-12)

    return target.astype(np.float32), True
