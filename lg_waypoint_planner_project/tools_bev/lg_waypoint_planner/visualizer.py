# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import json
import math
import os
import re
import subprocess

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

from .geometry import local_points_to_pixels
from .footprint_collision import obb_corners


# The displayed order is fixed and must match the collected six-view folders.
_SURROUND_VIEW_LAYOUT = (
    ("front_left", "front", "front_right"),
    ("rear_left", "rear", "rear_right"),
)

_SURROUND_VIEW_YAWS_DEG = {
    "front": 0.0,
    "front_left": -60.0,
    "front_right": 60.0,
    "rear": 180.0,
    "rear_left": -120.0,
    "rear_right": 120.0,
}

_SURROUND_VIEW_FOLDERS = {
    "front": ("rgb_front", "rgb", "CAM_FRONT", "cam_front"),
    "front_left": ("rgb_front_left",),
    "front_right": ("rgb_front_right",),
    "rear": ("rgb_rear",),
    "rear_left": ("rgb_rear_left",),
    "rear_right": ("rgb_rear_right",),
}


def normalize_costmap(costmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(costmap, dtype=np.float32)
    vmax = float(np.percentile(arr, 98.0)) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 1e-6)
    return np.clip(arr / vmax * 255.0, 0, 255).astype(np.uint8)


def draw_polyline(img, pts, ego_center, mpp, color, thickness=2):
    if cv2 is None or pts is None or len(pts) < 2:
        return
    pix = local_points_to_pixels(np.asarray(pts, dtype=np.float32), ego_center, mpp)
    pix = np.round(pix).astype(np.int32)
    for i in range(len(pix) - 1):
        p1 = tuple(int(v) for v in pix[i])
        p2 = tuple(int(v) for v in pix[i + 1])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_points(img, pts, ego_center, mpp, color, radius=3):
    if cv2 is None or pts is None:
        return
    pix = local_points_to_pixels(np.asarray(pts, dtype=np.float32), ego_center, mpp)
    for p in np.round(pix).astype(np.int32):
        cv2.circle(img, tuple(int(v) for v in p), radius, color, -1, cv2.LINE_AA)


def draw_obb(img, center_xy, yaw, half_l, half_w, ego_center, mpp, color, thickness=1):
    if cv2 is None:
        return
    corners = obb_corners(center_xy, yaw, half_l, half_w)
    pix = local_points_to_pixels(corners, ego_center, mpp)
    pix = np.round(pix).astype(np.int32)
    for i in range(4):
        cv2.line(
            img,
            tuple(pix[i]),
            tuple(pix[(i + 1) % 4]),
            color,
            thickness,
            cv2.LINE_AA,
        )


def save_bev_debug(
    path: Path,
    costmap,
    base_route,
    scored_candidates,
    selected_idx,
    actor_timelines,
    ego_center,
    meters_per_pixel,
    cfg,
):
    """Full BEV debug image used for internal candidate inspection."""
    if cv2 is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    gray = normalize_costmap(costmap)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    draw_polyline(img, base_route, ego_center, meters_per_pixel, (180, 180, 180), 1)

    colors = {
        "route_follow": (255, 255, 255),
        "cautious_follow": (255, 200, 0),
        "yield_stop": (0, 220, 255),
        "left_nudge": (255, 0, 255),
        "right_nudge": (0, 255, 255),
        "creep": (200, 120, 0),
        "emergency_brake": (0, 0, 255),
    }
    for i, candidate in enumerate(scored_candidates):
        name = candidate["info"].get("intent_name", "unknown")
        color = colors.get(name, (120, 120, 120))
        thickness = 3 if i == selected_idx else 1
        waypoints = candidate["rollout"]["waypoints"]
        draw_polyline(img, waypoints, ego_center, meters_per_pixel, color, thickness)
        draw_points(
            img,
            waypoints,
            ego_center,
            meters_per_pixel,
            color,
            2 if i != selected_idx else 4,
        )

    for actors in actor_timelines.values():
        for actor in actors:
            draw_obb(
                img,
                [actor.get("x_m", 0.0), actor.get("y_m", 0.0)],
                actor.get("yaw_rad", 0.0),
                actor.get("half_length_m", 2.0),
                actor.get("half_width_m", 1.0),
                ego_center,
                meters_per_pixel,
                (0, 0, 255),
                1,
            )

    cv2.circle(
        img,
        tuple(int(v) for v in ego_center),
        4,
        (0, 255, 0),
        -1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), img)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

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


def _actor_occupancy_half_extents(
    actor: Dict,
    cfg,
    meters_per_pixel: Optional[float],
) -> Tuple[float, float]:
    raw_hl = max(float(actor.get("half_length_m", 1.0)), 0.1)
    raw_hw = max(float(actor.get("half_width_m", 0.5)), 0.1)

    planning_cls = str(actor.get("class", "")).strip().lower()
    semantic_cls = str(actor.get("semantic_class", "")).strip().lower()
    raw_cls = str(actor.get("raw_class", "")).strip().lower()
    base_type = str(actor.get("base_type", "")).strip().lower()
    type_id = str(actor.get("type_id", "")).strip().lower()

    is_vulnerable_road_user = (
        planning_cls == "pedestrian"
        or semantic_cls in {"pedestrian", "cyclist"}
        or base_type == "bicycle"
        or "bicycle" in raw_cls
        or "bike" in raw_cls
        or "diamondback" in type_id
        or "crossbike" in type_id
        or "omafiets" in type_id
    )
    if not is_vulnerable_road_user:
        return raw_hl, raw_hw

    causal_cfg = _cfg_get(cfg, "causal_response", {})
    extent_scale = float(
        _cfg_get(causal_cfg, "pedestrian_mask_extent_scale", 2.0)
    )
    min_half_extent = float(
        _cfg_get(causal_cfg, "pedestrian_mask_min_half_extent_m", 0.8)
    )

    hl = max(raw_hl * extent_scale, min_half_extent)
    hw = max(raw_hw * extent_scale, min_half_extent)
    mpp = max(float(meters_per_pixel or 0.0), 1e-6)
    raster_pad = 0.5 * math.sqrt(2.0) * mpp
    return hl + raster_pad, hw + raster_pad


# -----------------------------------------------------------------------------
# Six-camera loading and calibration
# -----------------------------------------------------------------------------

def _find_image_in_folders(
    route_dir: Path,
    frame_name: str,
    folders,
) -> Optional[Path]:
    for folder_name in folders:
        rgb_dir = route_dir / str(folder_name)
        for suffix in (".jpg", ".png", ".jpeg"):
            path = rgb_dir / f"{frame_name}{suffix}"
            if path.is_file():
                return path
    return None


def find_rgb_image_path(route_dir: Path, frame_name: str, cfg) -> Optional[Path]:
    """Backward-compatible front-camera lookup."""
    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    folders = []
    configured_folder = _cfg_get(rgb_cfg, "rgb_folder", None)
    if configured_folder:
        folders.append(str(configured_folder))
    for folder in _SURROUND_VIEW_FOLDERS["front"]:
        if folder not in folders:
            folders.append(folder)
    return _find_image_in_folders(route_dir, frame_name, folders)


def _load_surround_metadata(route_dir: Path) -> Dict:
    metadata_path = route_dir / "surround_camera_config.json"
    if not metadata_path.is_file():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _default_camera_spec(view_name: str, cfg, image_shape=None) -> Dict:
    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    height = int(image_shape[0]) if image_shape is not None else 0
    width = int(image_shape[1]) if image_shape is not None else 0
    return {
        "position": [
            float(_cfg_get(rgb_cfg, "camera_x", -1.5)),
            float(_cfg_get(rgb_cfg, "camera_y", 0.0)),
            float(_cfg_get(rgb_cfg, "camera_z", 2.0)),
        ],
        "rotation": [
            float(_cfg_get(rgb_cfg, "camera_roll_deg", 0.0)),
            float(_cfg_get(rgb_cfg, "camera_pitch_deg", 0.0)),
            float(_SURROUND_VIEW_YAWS_DEG[view_name]),
        ],
        "width": width,
        "height": height,
        "fov": float(_cfg_get(rgb_cfg, "camera_fov", 110.0)),
        "sensor_id": "rgb" if view_name == "front" else f"rgb_{view_name}",
        "save_folder": (
            "rgb_front" if view_name == "front" else f"rgb_{view_name}"
        ),
    }


def _camera_spec_for_view(
    view_name: str,
    metadata: Dict,
    cfg,
    image_shape,
) -> Dict:
    cameras = metadata.get("cameras", {}) if isinstance(metadata, dict) else {}
    spec = cameras.get(view_name, {}) if isinstance(cameras, dict) else {}
    if not isinstance(spec, dict):
        spec = {}

    default = _default_camera_spec(view_name, cfg, image_shape)
    merged = dict(default)
    merged.update(spec)

    position = merged.get("position", default["position"])
    rotation = merged.get("rotation", default["rotation"])
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        position = default["position"]
    if not isinstance(rotation, (list, tuple)) or len(rotation) < 3:
        rotation = default["rotation"]

    merged["position"] = [float(position[0]), float(position[1]), float(position[2])]
    merged["rotation"] = [float(rotation[0]), float(rotation[1]), float(rotation[2])]
    merged["fov"] = float(merged.get("fov", default["fov"]))
    return merged


def _load_surround_rgb_images(
    route_dir: Path,
    frame_name: str,
    cfg,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Dict]]:
    """
    Load all six collected RGB views.

    Returning ``None`` keeps the visualizer compatible with older front-only
    routes. The six-view layout is used only when every required view exists.
    """
    metadata = _load_surround_metadata(route_dir)
    images: Dict[str, np.ndarray] = {}
    specs: Dict[str, Dict] = {}

    for row in _SURROUND_VIEW_LAYOUT:
        for view_name in row:
            folders = list(_SURROUND_VIEW_FOLDERS[view_name])
            if view_name == "front":
                rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
                configured_folder = _cfg_get(rgb_cfg, "rgb_folder", None)
                if configured_folder and str(configured_folder) not in folders:
                    folders.insert(0, str(configured_folder))

            path = _find_image_in_folders(route_dir, frame_name, folders)
            if path is None:
                return None, {}

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                return None, {}

            images[view_name] = image
            specs[view_name] = _camera_spec_for_view(
                view_name,
                metadata,
                cfg,
                image.shape[:2],
            )

    return images, specs


def _split_size(total: int, count: int) -> List[int]:
    base, remainder = divmod(int(total), int(count))
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _make_surround_mosaic(
    images: Dict[str, np.ndarray],
    cfg,
) -> np.ndarray:
    """
    Form the fixed 2 x 3 surround mosaic with substantially larger view tiles.

    The previous implementation compressed all six views into the original
    single-front-image canvas. That made each camera tile too small for checking
    projected trajectories and actor boxes. Each tile now defaults to 75% of the
    original camera resolution, so the complete mosaic becomes three tiles wide
    and two tiles high while preserving the required relative view positions.

    Optional debug.rgb fields:
      surround_tile_scale: default 0.75
      surround_tile_width: explicit tile width in pixels
      surround_tile_height: explicit tile height in pixels
    """
    front = images["front"]
    front_h, front_w = front.shape[:2]
    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})

    scale = max(
        0.1,
        float(_cfg_get(rgb_cfg, "surround_tile_scale", 0.75)),
    )
    tile_w = int(
        _cfg_get(
            rgb_cfg,
            "surround_tile_width",
            round(float(front_w) * scale),
        )
    )
    tile_h = int(
        _cfg_get(
            rgb_cfg,
            "surround_tile_height",
            round(float(front_h) * scale),
        )
    )
    tile_w = max(64, tile_w)
    tile_h = max(32, tile_h)

    rendered_rows = []
    for row_names in _SURROUND_VIEW_LAYOUT:
        tiles = []
        for view_name in row_names:
            source = images[view_name]
            interpolation = (
                cv2.INTER_AREA
                if tile_w <= source.shape[1] and tile_h <= source.shape[0]
                else cv2.INTER_LINEAR
            )
            tile = cv2.resize(
                source,
                (tile_w, tile_h),
                interpolation=interpolation,
            )
            tiles.append(tile)
        rendered_rows.append(np.concatenate(tiles, axis=1))

    return np.concatenate(rendered_rows, axis=0)


# -----------------------------------------------------------------------------
# Camera projection
# -----------------------------------------------------------------------------

def build_projection_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = float(width) / (
        2.0 * math.tan(math.radians(float(fov_deg)) / 2.0)
    )
    return np.asarray(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_matrix_roll_pitch_yaw_deg(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float32,
    )
    ry = np.asarray(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float32,
    )
    rz = np.asarray(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (rz @ ry @ rx).astype(np.float32)


def project_ego_local_xyz_to_rgb(
    points_ego_xyz: np.ndarray,
    image_shape: Tuple[int, int],
    cfg,
    camera_spec: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project ego-local 3D points into one RGB view.

    When ``camera_spec`` is omitted, the original front-camera configuration is
    used. For the other five views, the matching collected camera calibration is
    supplied from ``surround_camera_config.json``.
    """
    points_ego = np.asarray(points_ego_xyz, dtype=np.float32)
    if (
        points_ego.ndim != 2
        or points_ego.shape[1] < 3
        or len(points_ego) == 0
    ):
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    h, w = int(image_shape[0]), int(image_shape[1])

    if not isinstance(camera_spec, dict):
        camera_spec = _default_camera_spec("front", cfg, image_shape)

    position = camera_spec.get("position", [-1.5, 0.0, 2.0])
    rotation = camera_spec.get("rotation", [0.0, 0.0, 0.0])
    fov = float(
        camera_spec.get(
            "fov",
            _cfg_get(rgb_cfg, "camera_fov", 110.0),
        )
    )

    intrinsic = build_projection_matrix(w, h, fov)
    camera_translation = np.asarray(position[:3], dtype=np.float32)
    camera_to_ego = rotation_matrix_roll_pitch_yaw_deg(
        float(rotation[0]),
        float(rotation[1]),
        float(rotation[2]),
    )

    points_relative = points_ego[:, :3] - camera_translation[None, :]
    points_camera = (camera_to_ego.T @ points_relative.T).T

    depth = points_camera[:, 0]
    right = points_camera[:, 1]
    up = points_camera[:, 2]
    pixels = np.zeros((len(points_camera), 2), dtype=np.float32)

    min_depth = float(_cfg_get(rgb_cfg, "min_projection_depth_m", 0.1))
    valid = depth > min_depth
    if np.any(valid):
        pixels[valid, 0] = (
            intrinsic[0, 0] * right[valid] / depth[valid]
            + intrinsic[0, 2]
        )
        pixels[valid, 1] = (
            intrinsic[1, 2]
            - intrinsic[1, 1] * up[valid] / depth[valid]
        )

    valid = (
        valid
        & np.isfinite(pixels[:, 0])
        & np.isfinite(pixels[:, 1])
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(w))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(h))
    )
    return pixels, valid


def project_ego_local_points_to_rgb(
    points_local_xy: np.ndarray,
    image_shape: Tuple[int, int],
    cfg,
    camera_spec: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    points_local_xy = np.asarray(points_local_xy, dtype=np.float32)
    if (
        points_local_xy.ndim != 2
        or points_local_xy.shape[1] < 2
        or len(points_local_xy) == 0
    ):
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    z = np.full(
        (len(points_local_xy), 1),
        float(_cfg_get(rgb_cfg, "waypoint_ground_z_m", 0.0)),
        dtype=np.float32,
    )
    points_ego = np.concatenate([points_local_xy[:, :2], z], axis=1)
    return project_ego_local_xyz_to_rgb(
        points_ego,
        image_shape,
        cfg,
        camera_spec=camera_spec,
    )


def draw_projected_waypoints_on_rgb(
    image: np.ndarray,
    points_local_xy: Optional[np.ndarray],
    cfg,
    color: Tuple[int, int, int],
    radius: int,
    thickness: int,
    camera_spec: Optional[Dict] = None,
) -> int:
    if points_local_xy is None:
        return 0
    points = np.asarray(points_local_xy, dtype=np.float32)
    if len(points) == 0:
        return 0

    pixels, valid = project_ego_local_points_to_rgb(
        points,
        image.shape[:2],
        cfg,
        camera_spec=camera_spec,
    )
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return 0

    pts = np.round(pixels).astype(np.int32)
    for i in range(len(pts) - 1):
        if valid[i] and valid[i + 1]:
            cv2.line(
                image,
                (int(pts[i, 0]), int(pts[i, 1])),
                (int(pts[i + 1, 0]), int(pts[i + 1, 1])),
                color,
                thickness,
                cv2.LINE_AA,
            )

    for rank, index in enumerate(valid_indices):
        point = (int(pts[index, 0]), int(pts[index, 1]))
        point_radius = radius + 1 if rank == 0 else radius
        cv2.circle(
            image,
            point,
            point_radius,
            color,
            -1,
            cv2.LINE_AA,
        )
    return int(len(valid_indices))


def draw_projected_route_on_rgb(
    image: np.ndarray,
    route_local_xy: Optional[np.ndarray],
    cfg,
    color: Tuple[int, int, int],
    thickness: int,
    camera_spec: Optional[Dict] = None,
) -> int:
    if route_local_xy is None:
        return 0
    route = np.asarray(route_local_xy, dtype=np.float32)
    if len(route) < 2:
        return 0

    pixels, valid = project_ego_local_points_to_rgb(
        route,
        image.shape[:2],
        cfg,
        camera_spec=camera_spec,
    )
    pts = np.round(pixels).astype(np.int32)
    for i in range(len(pts) - 1):
        if valid[i] and valid[i + 1]:
            cv2.line(
                image,
                (int(pts[i, 0]), int(pts[i, 1])),
                (int(pts[i + 1, 0]), int(pts[i + 1, 1])),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return int(np.sum(valid))


# -----------------------------------------------------------------------------
# Actor-box rendering
# -----------------------------------------------------------------------------

_CAUSAL_VISUAL_PALETTE = [
    (178, 114, 0),
    (0, 94, 213),
    (167, 121, 204),
    (0, 159, 230),
    (233, 180, 86),
    (126, 86, 155),
    (65, 145, 225),
    (180, 105, 40),
]

_NON_ACTOR_CANDIDATE_PALETTE = [
    (215, 120, 40),
    (90, 170, 220),
    (190, 90, 170),
    (70, 150, 210),
    (200, 130, 80),
]

_SELECTED_TRAJECTORY_COLOR = (0, 175, 0)


def _same_visual_actor(a: Optional[Dict], b: Optional[Dict]) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False

    actor_id_a = a.get("id", None)
    actor_id_b = b.get("id", None)
    if actor_id_a is not None and actor_id_b is not None:
        return str(actor_id_a) == str(actor_id_b)

    if str(a.get("class", "")) != str(b.get("class", "")):
        return False

    dx = float(a.get("x_m", 0.0)) - float(b.get("x_m", 0.0))
    dy = float(a.get("y_m", 0.0)) - float(b.get("y_m", 0.0))
    return math.hypot(dx, dy) <= 1.5


def _candidate_response_object(candidate: Dict) -> Dict:
    info = candidate.get("info", {}) if isinstance(candidate, dict) else {}
    obj = info.get("response_object", None) if isinstance(info, dict) else None
    if not isinstance(obj, dict):
        obj = (
            candidate.get("response_object", {})
            if isinstance(candidate, dict)
            else {}
        )
    return obj if isinstance(obj, dict) else {}


def _build_candidate_visual_colors(
    scored_candidates: List[Dict],
    selected_idx: int,
    causal_test_actors: Optional[List[Dict]],
):
    actors = []
    for actor in causal_test_actors or []:
        if not isinstance(actor, dict) or not bool(actor.get("exists", False)):
            continue
        if any(_same_visual_actor(actor, existing) for existing in actors):
            continue
        actors.append(dict(actor))

    selected_object = {}
    if 0 <= int(selected_idx) < len(scored_candidates):
        selected_object = _candidate_response_object(
            scored_candidates[int(selected_idx)]
        )

    actor_entries = []
    palette_cursor = 0
    for actor_index, actor in enumerate(actors):
        if _same_visual_actor(actor, selected_object):
            color = _SELECTED_TRAJECTORY_COLOR
        else:
            color = _CAUSAL_VISUAL_PALETTE[
                palette_cursor % len(_CAUSAL_VISUAL_PALETTE)
            ]
            palette_cursor += 1

        actor_entries.append(
            {
                "actor": actor,
                "color": color,
                "label": f"A{actor_index + 1}",
            }
        )

    candidate_colors = []
    non_actor_color_by_key = {}
    for candidate in scored_candidates:
        response_object = _candidate_response_object(candidate)
        color = None
        for entry in actor_entries:
            if _same_visual_actor(response_object, entry["actor"]):
                color = entry["color"]
                break

        if color is None:
            info = candidate.get("info", {}) if isinstance(candidate, dict) else {}
            key = (
                str(
                    info.get(
                        "intent_name",
                        candidate.get("intent_name", "unknown"),
                    )
                ),
                str(
                    info.get(
                        "variant_id",
                        candidate.get("variant_id", "default"),
                    )
                ),
            )
            if key not in non_actor_color_by_key:
                index = len(non_actor_color_by_key)
                non_actor_color_by_key[key] = _NON_ACTOR_CANDIDATE_PALETTE[
                    index % len(_NON_ACTOR_CANDIDATE_PALETTE)
                ]
            color = non_actor_color_by_key[key]

        candidate_colors.append(color)

    return actor_entries, candidate_colors


def _actor_cuboid_corners_ego(actor: Dict) -> np.ndarray:
    half_l = max(float(actor.get("half_length_m", 2.25)), 0.1)
    half_w = max(float(actor.get("half_width_m", 1.00)), 0.1)
    half_h = max(float(actor.get("half_height_m", 0.80)), 0.1)

    center_x = float(actor.get("x_m", 0.0))
    center_y = float(actor.get("y_m", 0.0))
    raw_z = actor.get("z_m", None)
    try:
        center_z = float(raw_z)
        if not np.isfinite(center_z) or abs(center_z) < 0.05:
            center_z = half_h
    except Exception:
        center_z = half_h

    yaw = float(actor.get("yaw_rad", 0.0))
    cosine, sine = math.cos(yaw), math.sin(yaw)

    local_xy = np.asarray(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float32,
    )
    xy = local_xy @ rotation.T
    xy[:, 0] += center_x
    xy[:, 1] += center_y

    bottom = np.concatenate(
        [
            xy,
            np.full(
                (4, 1),
                center_z - half_h,
                dtype=np.float32,
            ),
        ],
        axis=1,
    )
    top = np.concatenate(
        [
            xy,
            np.full(
                (4, 1),
                center_z + half_h,
                dtype=np.float32,
            ),
        ],
        axis=1,
    )
    return np.concatenate([bottom, top], axis=0)


def draw_projected_actor_bbox_on_rgb(
    image: np.ndarray,
    actor: Dict,
    cfg,
    color: Tuple[int, int, int],
    thickness: int = 1,
    label: Optional[str] = None,
    camera_spec: Optional[Dict] = None,
) -> int:
    """Draw the same solid 3D actor box used by the original front view."""
    corners = _actor_cuboid_corners_ego(actor)
    pixels, valid = project_ego_local_xyz_to_rgb(
        corners,
        image.shape[:2],
        cfg,
        camera_spec=camera_spec,
    )
    pts = np.round(pixels).astype(np.int32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    drawn = 0
    for i, j in edges:
        if valid[i] and valid[j]:
            cv2.line(
                image,
                (int(pts[i, 0]), int(pts[i, 1])),
                (int(pts[j, 0]), int(pts[j, 1])),
                color,
                thickness,
                cv2.LINE_AA,
            )
            drawn += 1

    visible = pts[valid]
    if label and len(visible) > 0:
        x = int(
            np.clip(
                np.min(visible[:, 0]),
                6,
                max(6, image.shape[1] - 60),
            )
        )
        y = int(
            np.clip(
                np.min(visible[:, 1]) - 5,
                22,
                max(22, image.shape[0] - 6),
            )
        )
        _put_label(
            image,
            label,
            (x, y),
            0.52,
            color,
            1,
            bold=True,
        )
    return drawn


def _secondary_attention_actor(factor: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(factor, dict):
        return None
    actor = factor.get("secondary_attention_actor") or {}
    if isinstance(actor, dict) and bool(actor.get("exists", False)):
        return actor
    return None


def _draw_dashed_segment(
    image: np.ndarray,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_px: float = 9.0,
    gap_px: float = 6.0,
):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return

    ux, uy = dx / length, dy / length
    step = max(float(dash_px) + float(gap_px), 1.0)
    start = 0.0
    while start < length:
        end = min(start + float(dash_px), length)
        q0 = (
            int(round(x0 + ux * start)),
            int(round(y0 + uy * start)),
        )
        q1 = (
            int(round(x0 + ux * end)),
            int(round(y0 + uy * end)),
        )
        cv2.line(
            image,
            q0,
            q1,
            color,
            int(thickness),
            cv2.LINE_AA,
        )
        start += step


def draw_projected_secondary_attention_actor_on_rgb(
    image: np.ndarray,
    actor: Dict,
    cfg,
    color: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 1,
    camera_spec: Optional[Dict] = None,
) -> int:
    """Draw the original dashed magenta secondary-attention 3D box."""
    corners = _actor_cuboid_corners_ego(actor)
    pixels, valid = project_ego_local_xyz_to_rgb(
        corners,
        image.shape[:2],
        cfg,
        camera_spec=camera_spec,
    )
    pts = np.round(pixels).astype(np.int32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    drawn = 0
    for i, j in edges:
        if valid[i] and valid[j]:
            _draw_dashed_segment(
                image,
                (int(pts[i, 0]), int(pts[i, 1])),
                (int(pts[j, 0]), int(pts[j, 1])),
                color=color,
                thickness=thickness,
            )
            drawn += 1
    return drawn


def _draw_original_actor_boxes_on_view(
    image: np.ndarray,
    camera_spec: Dict,
    actor_visual_entries,
    attention_actor: Optional[Dict],
    cfg,
):
    """
    Apply the original actor-box rendering to one camera.

    The function does not choose additional actors. It only projects the exact
    causal-test actors and secondary-attention actor already selected by the
    original visualizer. A box is drawn in this view only when its projected
    cuboid edges are inside the image.
    """
    for entry in actor_visual_entries:
        draw_projected_actor_bbox_on_rgb(
            image,
            entry["actor"],
            cfg,
            color=entry["color"],
            thickness=1,
            label=entry["label"],
            camera_spec=camera_spec,
        )

    if attention_actor is not None:
        draw_projected_secondary_attention_actor_on_rgb(
            image,
            attention_actor,
            cfg,
            color=(255, 0, 255),
            thickness=1,
            camera_spec=camera_spec,
        )


# -----------------------------------------------------------------------------
# Font and language panel
# -----------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _resolve_times_new_roman_font_path(
    bold: bool = False,
) -> Optional[str]:
    style_names = (
        [
            "timesbd.ttf",
            "Times_New_Roman_Bold.ttf",
            "TimesNewRomanBold.ttf",
            "Times New Roman Bold.ttf",
        ]
        if bold
        else [
            "times.ttf",
            "Times_New_Roman.ttf",
            "TimesNewRoman.ttf",
            "Times New Roman.ttf",
        ]
    )
    search_dirs = [
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        "/usr/share/fonts/truetype/msttcorefonts",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman",
        "/usr/share/fonts/truetype/msttcorefonts/TimesNewRoman",
        "C:/Windows/Fonts",
    ]
    for directory in search_dirs:
        for name in style_names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate

    family = (
        "Times New Roman:style=Bold"
        if bold
        else "Times New Roman:style=Regular"
    )
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{family}|%{file}", family],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.stdout and "|" in result.stdout:
            matched_family, matched_path = result.stdout.split("|", 1)
            matched_path = matched_path.strip()
            if os.path.isfile(matched_path) and (
                "Times New Roman" in matched_family
                or "Tinos" in matched_family
                or "Liberation Serif" in matched_family
            ):
                return matched_path
    except Exception:
        pass
    return None


@lru_cache(maxsize=64)
def _times_font(pixel_size: int, bold: bool = False):
    if ImageFont is None:
        return None
    path = _resolve_times_new_roman_font_path(bool(bold))
    if path is None:
        return None
    try:
        return ImageFont.truetype(path, size=max(8, int(pixel_size)))
    except Exception:
        return None


def _put_label(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (30, 30, 30),
    thickness: int = 1,
    bold: bool = False,
    outline: bool = True,
):
    text = str(text)
    font_px = max(10, int(round(32.0 * float(scale))))
    font = _times_font(font_px, bold=bold)

    if (
        Image is not None
        and ImageDraw is not None
        and font is not None
    ):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        fill_rgb = (
            int(color[2]),
            int(color[1]),
            int(color[0]),
        )
        stroke_width = max(1, int(thickness) + 1) if outline else 0
        kwargs = {
            "fill": fill_rgb,
            "font": font,
            "stroke_width": stroke_width,
            "stroke_fill": (255, 255, 255),
        }
        try:
            draw.text(
                (int(org[0]), int(org[1])),
                text,
                anchor="ls",
                **kwargs,
            )
        except Exception:
            draw.text(
                (int(org[0]), int(org[1]) - font_px),
                text,
                **kwargs,
            )
        img[:] = cv2.cvtColor(
            np.asarray(pil_img),
            cv2.COLOR_RGB2BGR,
        )
        return

    if outline:
        cv2.putText(
            img,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness + 2,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _measure_text_width(
    text: str,
    font_px: int,
    bold: bool = False,
) -> float:
    font = _times_font(font_px, bold=bold)
    if (
        Image is not None
        and ImageDraw is not None
        and font is not None
    ):
        try:
            draw = ImageDraw.Draw(
                Image.new("RGB", (8, 8), (255, 255, 255))
            )
            box = draw.textbbox((0, 0), str(text), font=font)
            return float(box[2] - box[0])
        except Exception:
            pass
    return float(len(str(text))) * float(font_px) * 0.52


def _wrap_text_to_width(
    text: str,
    max_width: int,
    font_px: int,
    bold: bool = False,
) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if (
            _measure_text_width(candidate, font_px, bold=bold)
            <= float(max_width)
        ):
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def make_language_qa_panel(
    language_annotation: Optional[Dict],
    target_w: int,
    cfg,
) -> Optional[np.ndarray]:
    """Create the original four-column language Q&A canvas."""
    if not isinstance(language_annotation, dict):
        return None

    qa_pairs = language_annotation.get("qa_pairs_en", [])
    if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
        return None
    qa_pairs = qa_pairs[:4]

    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    vertical_pad = 20
    gap = 16
    card_count = len(qa_pairs)

    total_w = int(target_w)
    total_gap = max(0, card_count - 1) * gap
    usable_w = max(card_count, total_w - total_gap)
    base_w, remainder = divmod(usable_w, max(card_count, 1))
    card_widths = [
        base_w + (1 if index < remainder else 0)
        for index in range(card_count)
    ]

    question_px = 22
    answer_px = 21
    question_line_h = 27
    answer_line_h = 26

    prepared = []
    required_card_h = 0
    for index, pair in enumerate(qa_pairs, start=1):
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()

        answer_label_re = re.compile(r"(?i)\bA[1234]\b")
        question = answer_label_re.sub("", question)
        answer = answer_label_re.sub("", answer)
        question = re.sub(r"\s+", " ", question).strip()
        answer = re.sub(r"\s+", " ", answer).strip()

        card_w = card_widths[index - 1]
        text_w = max(80, card_w - 32)

        question_lines = [
            line
            for line in _wrap_text_to_width(
                question,
                max(60, text_w - 34),
                question_px,
                bold=True,
            )
            if not re.fullmatch(r"(?i)A[1234]", line.strip())
        ]
        answer_lines = [
            line
            for line in _wrap_text_to_width(
                answer,
                text_w,
                answer_px,
                bold=False,
            )
            if not re.fullmatch(r"(?i)A[1234]", line.strip())
        ]

        card_h = (
            18
            + max(1, len(question_lines)) * question_line_h
            + 12
            + max(1, len(answer_lines)) * answer_line_h
            + 18
        )
        required_card_h = max(required_card_h, card_h)
        prepared.append((index, question_lines, answer_lines))

    min_panel_h = int(
        _cfg_get(rgb_cfg, "language_panel_height", 230)
    )
    panel_h = max(
        min_panel_h,
        required_card_h + 2 * vertical_pad,
    )
    panel = np.full(
        (panel_h, int(target_w), 3),
        250,
        dtype=np.uint8,
    )
    cv2.line(
        panel,
        (0, 0),
        (int(target_w) - 1, 0),
        (195, 195, 195),
        1,
        cv2.LINE_AA,
    )

    x_cursor = 0
    for column, (qa_index, question_lines, answer_lines) in enumerate(prepared):
        card_w = card_widths[column]
        x0 = x_cursor
        x1 = x0 + card_w - 1
        y0 = vertical_pad
        y1 = panel_h - vertical_pad
        cv2.rectangle(
            panel,
            (x0, y0),
            (x1, y1),
            (214, 214, 214),
            1,
            cv2.LINE_AA,
        )
        x_cursor = x1 + 1 + gap

        text_x = x0 + 16
        y = y0 + 32
        _put_label(
            panel,
            f"Q{qa_index}",
            (text_x, y),
            0.70,
            (45, 45, 45),
            1,
            bold=True,
            outline=False,
        )

        question_x = text_x + 36
        for line_index, line in enumerate(question_lines):
            line_x = question_x if line_index == 0 else text_x
            _put_label(
                panel,
                line,
                (line_x, y),
                question_px / 32.0,
                (35, 35, 35),
                1,
                bold=True,
                outline=False,
            )
            y += question_line_h

        y += 12
        for line in answer_lines:
            _put_label(
                panel,
                line,
                (text_x, y),
                answer_px / 32.0,
                (55, 55, 55),
                1,
                bold=False,
                outline=False,
            )
            y += answer_line_h

    return panel


# -----------------------------------------------------------------------------
# Compact BEV panel
# -----------------------------------------------------------------------------

def _bev_to_px(
    x: float,
    y: float,
    width: int,
    height: int,
    scale: float,
    origin_u: float,
    origin_v: float,
) -> Tuple[int, int]:
    del width, height
    u = origin_u + float(y) * scale
    v = origin_v - float(x) * scale
    return int(round(u)), int(round(v))


def _draw_local_obb_on_panel(
    panel: np.ndarray,
    center_xy: Tuple[float, float],
    yaw: float,
    half_l: float,
    half_w: float,
    scale: float,
    origin_u: float,
    origin_v: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    fill_color: Optional[Tuple[int, int, int]] = None,
):
    corners = obb_corners(center_xy, yaw, half_l, half_w)
    pix = np.asarray(
        [
            _bev_to_px(
                float(point[0]),
                float(point[1]),
                panel.shape[1],
                panel.shape[0],
                scale,
                origin_u,
                origin_v,
            )
            for point in corners
        ],
        dtype=np.int32,
    )
    if fill_color is not None:
        cv2.fillPoly(panel, [pix], fill_color, cv2.LINE_AA)
    cv2.polylines(
        panel,
        [pix],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _get_occupancy_blocked_threshold(cfg) -> float:
    scoring = _cfg_get(cfg, "scoring", {})
    for key in ("occupancy_blocked_threshold", "hard_cost_threshold"):
        value = _cfg_get(scoring, key, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return 80.0


def _draw_occupancy_background_on_panel(
    panel: np.ndarray,
    occupancy_map: Optional[np.ndarray],
    ego_center,
    meters_per_pixel: Optional[float],
    scale: float,
    origin_u: float,
    origin_v: float,
    cfg,
):
    if (
        occupancy_map is None
        or ego_center is None
        or meters_per_pixel is None
    ):
        return

    occupancy = np.asarray(occupancy_map, dtype=np.float32)
    if occupancy.ndim != 2 or occupancy.size == 0:
        return

    h, w = panel.shape[:2]
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    local_y = (uu - float(origin_u)) / max(float(scale), 1e-6)
    local_x = (float(origin_v) - vv) / max(float(scale), 1e-6)

    center_x = float(ego_center[0])
    center_y = float(ego_center[1])
    col = np.round(
        center_x + local_y / float(meters_per_pixel)
    ).astype(np.int32)
    row = np.round(
        center_y - local_x / float(meters_per_pixel)
    ).astype(np.int32)

    inside = (
        (row >= 0)
        & (row < occupancy.shape[0])
        & (col >= 0)
        & (col < occupancy.shape[1])
    )
    threshold = _get_occupancy_blocked_threshold(cfg)

    values = np.zeros((h, w), dtype=np.float32)
    values[inside] = occupancy[row[inside], col[inside]]
    free = inside & (values < threshold)
    blocked = inside & (values >= threshold)

    panel[~inside] = (238, 238, 238)
    panel[free] = (224, 241, 229)
    panel[blocked] = (146, 134, 122)


def _yaw_sequence_from_waypoints(
    points: Optional[np.ndarray],
    yaws: Optional[np.ndarray] = None,
) -> np.ndarray:
    pts = (
        np.asarray(points, dtype=np.float32)
        if points is not None
        else np.zeros((0, 2), dtype=np.float32)
    )
    if (
        pts.ndim != 2
        or pts.shape[1] < 2
        or len(pts) == 0
    ):
        return np.zeros((0,), dtype=np.float32)

    if yaws is not None:
        yaw_values = np.asarray(yaws, dtype=np.float32).reshape(-1)
        if len(yaw_values) >= len(pts):
            return yaw_values[: len(pts)].astype(np.float32)

    previous = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            pts[:-1, :2],
        ],
        axis=0,
    )
    delta = pts[:, :2] - previous
    return np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)


def _display_actor(factor: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(factor, dict):
        return None

    critical = factor.get("critical_actor") or {}
    if isinstance(critical, dict) and bool(critical.get("exists", False)):
        return critical

    secondary = factor.get("secondary_attention_actor") or {}
    if isinstance(secondary, dict) and bool(secondary.get("exists", False)):
        return secondary

    return None


def _critical_actor_id(factor: Optional[Dict]):
    actor = _display_actor(factor)
    return None if actor is None else actor.get("id", None)


def _match_critical_actor_at_step(
    factor: Optional[Dict],
    actor_timelines: Optional[Dict[int, List[Dict]]],
    step_idx: int,
) -> Optional[Dict]:
    if not isinstance(actor_timelines, dict):
        return None

    actor_id = _critical_actor_id(factor)
    actors = actor_timelines.get(int(step_idx), []) or []
    if actor_id is not None:
        for actor in actors:
            if actor.get("id", None) == actor_id:
                return actor

    display_actor = _display_actor(factor)
    if (
        isinstance(display_actor, dict)
        and bool(display_actor.get("exists", False))
        and actors
    ):
        actor_x = float(display_actor.get("x_m", 0.0))
        actor_y = float(display_actor.get("y_m", 0.0))

        def distance(actor):
            return (
                (float(actor.get("x_m", 0.0)) - actor_x) ** 2
                + (float(actor.get("y_m", 0.0)) - actor_y) ** 2
            )

        return min(actors, key=distance)

    return None


def make_compact_critical_bev_panel(
    factor: Optional[Dict],
    cfg,
    target_h: int,
    target_w: Optional[int] = None,
    occupancy_map: Optional[np.ndarray] = None,
    ego_center=None,
    meters_per_pixel: Optional[float] = None,
    planned_waypoints: Optional[np.ndarray] = None,
    planned_yaws: Optional[np.ndarray] = None,
    actor_timelines: Optional[Dict[int, List[Dict]]] = None,
) -> np.ndarray:
    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    panel_w = (
        int(target_w)
        if target_w is not None
        else int(_cfg_get(rgb_cfg, "left_bev_panel_width", 420))
    )
    panel_h = int(target_h)
    panel = np.full(
        (panel_h, panel_w, 3),
        245,
        dtype=np.uint8,
    )

    forward_m = float(_cfg_get(rgb_cfg, "left_bev_forward_m", 35.0))
    rear_m = float(_cfg_get(rgb_cfg, "left_bev_rear_m", 8.0))
    side_m = float(_cfg_get(rgb_cfg, "left_bev_side_m", 12.0))
    margin = 34.0
    scale_x = (
        panel_h - 2.0 * margin
    ) / max(forward_m + rear_m, 1.0)
    scale_y = (
        panel_w - 2.0 * margin
    ) / max(2.0 * side_m, 1.0)
    scale = float(min(scale_x, scale_y))
    origin_u = panel_w * 0.5
    origin_v = margin + forward_m * scale

    _draw_occupancy_background_on_panel(
        panel,
        occupancy_map,
        ego_center,
        meters_per_pixel,
        scale,
        origin_u,
        origin_v,
        cfg,
    )
    cv2.rectangle(
        panel,
        (0, 0),
        (panel_w - 1, panel_h - 1),
        (190, 190, 190),
        1,
    )

    ego_half_l = float(cfg.vehicle.ego_half_length_m)
    ego_half_w = float(cfg.vehicle.ego_half_width_m)
    _draw_local_obb_on_panel(
        panel,
        (0.0, 0.0),
        0.0,
        ego_half_l,
        ego_half_w,
        scale,
        origin_u,
        origin_v,
        (0, 120, 0),
        2,
        (205, 235, 205),
    )

    points = (
        np.asarray(planned_waypoints, dtype=np.float32)
        if planned_waypoints is not None
        else np.zeros((0, 2), dtype=np.float32)
    )
    yaw_values = _yaw_sequence_from_waypoints(
        points,
        planned_yaws,
    )

    if (
        points.ndim == 2
        and points.shape[1] >= 2
        and len(points) > 0
    ):
        pixel_points = []
        for index, point in enumerate(points[:, :2], start=1):
            yaw = (
                float(yaw_values[index - 1])
                if len(yaw_values) >= index
                else 0.0
            )
            _draw_local_obb_on_panel(
                panel,
                (float(point[0]), float(point[1])),
                yaw,
                ego_half_l,
                ego_half_w,
                scale,
                origin_u,
                origin_v,
                (0, 145, 0),
                1,
            )
            pixel_points.append(
                _bev_to_px(
                    float(point[0]),
                    float(point[1]),
                    panel_w,
                    panel_h,
                    scale,
                    origin_u,
                    origin_v,
                )
            )

        for index in range(len(pixel_points) - 1):
            cv2.line(
                panel,
                pixel_points[index],
                pixel_points[index + 1],
                (0, 160, 0),
                2,
                cv2.LINE_AA,
            )
        for index, pixel_point in enumerate(pixel_points, start=1):
            cv2.circle(
                panel,
                pixel_point,
                3,
                (0, 120, 0),
                -1,
                cv2.LINE_AA,
            )
            _put_label(
                panel,
                str(index),
                (pixel_point[0] + 4, pixel_point[1] - 4),
                0.35,
                (0, 100, 0),
                1,
            )

    current_actor = _display_actor(factor)
    if (
        isinstance(current_actor, dict)
        and bool(current_actor.get("exists", False))
    ):
        actor_x = float(current_actor.get("x_m", 0.0))
        actor_y = float(current_actor.get("y_m", 0.0))
        actor_yaw = float(current_actor.get("yaw_rad", 0.0))
        actor_half_l, actor_half_w = _actor_occupancy_half_extents(
            current_actor,
            cfg,
            meters_per_pixel,
        )

        actor_class = str(
            current_actor.get(
                "semantic_class",
                current_actor.get("class", "actor"),
            )
        )
        relative_position = str(
            current_actor.get("relative_position", "nearby")
        )

        _draw_local_obb_on_panel(
            panel,
            (actor_x, actor_y),
            actor_yaw,
            actor_half_l,
            actor_half_w,
            scale,
            origin_u,
            origin_v,
            (30, 30, 210),
            2,
            (210, 214, 255),
        )

        label_u, label_v = _bev_to_px(
            actor_x,
            actor_y,
            panel_w,
            panel_h,
            scale,
            origin_u,
            origin_v,
        )
        label_u = int(np.clip(label_u + 8, 8, panel_w - 150))
        label_v = int(np.clip(label_v - 8, 50, panel_h - 28))
        _put_label(
            panel,
            f"{actor_class} / {relative_position}",
            (label_u, label_v),
            0.42,
            (30, 30, 170),
            1,
        )

        number_of_steps = (
            len(points)
            if points.ndim == 2
            else int(
                _cfg_get(
                    cfg.horizon,
                    "num_future_waypoints",
                    0,
                )
            )
        )
        for step in range(1, int(number_of_steps) + 1):
            actor = _match_critical_actor_at_step(
                factor,
                actor_timelines,
                step,
            )
            if not isinstance(actor, dict):
                continue

            future_x = float(actor.get("x_m", 0.0))
            future_y = float(actor.get("y_m", 0.0))
            future_yaw = float(actor.get("yaw_rad", 0.0))
            future_half_l, future_half_w = _actor_occupancy_half_extents(
                actor,
                cfg,
                meters_per_pixel,
            )
            _draw_local_obb_on_panel(
                panel,
                (future_x, future_y),
                future_yaw,
                future_half_l,
                future_half_w,
                scale,
                origin_u,
                origin_v,
                (40, 40, 220),
                1,
            )
            future_pixel = _bev_to_px(
                future_x,
                future_y,
                panel_w,
                panel_h,
                scale,
                origin_u,
                origin_v,
            )
            cv2.circle(
                panel,
                future_pixel,
                2,
                (40, 40, 220),
                -1,
                cv2.LINE_AA,
            )

    return panel


# -----------------------------------------------------------------------------
# Speed panel
# -----------------------------------------------------------------------------

def _speed_from_waypoints(
    points: Optional[np.ndarray],
    future_fps: float,
) -> Optional[np.ndarray]:
    if points is None:
        return None
    pts = np.asarray(points, dtype=np.float32)
    if (
        pts.ndim != 2
        or pts.shape[1] < 2
        or len(pts) == 0
    ):
        return None

    previous = np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            pts[:-1, :2],
        ],
        axis=0,
    )
    return (
        np.linalg.norm(pts[:, :2] - previous, axis=1)
        * float(future_fps)
    ).astype(np.float32)


def _draw_curve(
    panel: np.ndarray,
    values: np.ndarray,
    time_values: np.ndarray,
    plot_rect: Tuple[int, int, int, int],
    max_speed: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    radius: int = 4,
):
    if values is None or len(values) == 0:
        return

    x0, y0, x1, y1 = plot_rect
    denominator = (
        max(float(time_values[-1] - time_values[0]), 1e-6)
        if len(time_values) > 1
        else 1.0
    )

    points = []
    for index, value in enumerate(values):
        time_value = (
            float(time_values[index])
            if index < len(time_values)
            else float(index + 1)
        )
        px = (
            x0
            + (time_value - float(time_values[0]))
            / denominator
            * (x1 - x0)
        )
        py = (
            y1
            - np.clip(float(value), 0.0, max_speed)
            / max(max_speed, 1e-6)
            * (y1 - y0)
        )
        points.append((int(round(px)), int(round(py))))

    for index in range(len(points) - 1):
        cv2.line(
            panel,
            points[index],
            points[index + 1],
            color,
            thickness,
            cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(
            panel,
            point,
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )


def make_speed_panel(
    planned_speeds: Optional[np.ndarray],
    expert_speeds: Optional[np.ndarray],
    cfg,
    target_h: int,
    target_w: Optional[int] = None,
) -> np.ndarray:
    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    panel_w = (
        int(target_w)
        if target_w is not None
        else int(_cfg_get(rgb_cfg, "speed_panel_width", 420))
    )
    panel_h = int(target_h)
    panel = np.full(
        (panel_h, panel_w, 3),
        250,
        dtype=np.uint8,
    )
    cv2.rectangle(
        panel,
        (0, 0),
        (panel_w - 1, panel_h - 1),
        (210, 210, 210),
        1,
    )

    planned = (
        None
        if planned_speeds is None
        else np.asarray(planned_speeds, dtype=np.float32).reshape(-1)
    )
    expert = (
        None
        if expert_speeds is None
        else np.asarray(expert_speeds, dtype=np.float32).reshape(-1)
    )
    count = max(
        len(planned) if planned is not None else 0,
        len(expert) if expert is not None else 0,
    )
    if count == 0:
        _put_label(
            panel,
            "no speed data",
            (16, 70),
            0.6,
            (80, 80, 80),
            1,
        )
        return panel

    future_fps = float(cfg.horizon.future_fps)
    time_values = (
        np.arange(count, dtype=np.float32) + 1.0
    ) / max(future_fps, 1e-6)

    if planned is not None and len(planned) < count:
        planned = np.pad(
            planned,
            (0, count - len(planned)),
            mode="edge",
        )
    if expert is not None and len(expert) < count:
        expert = np.pad(
            expert,
            (0, count - len(expert)),
            mode="edge",
        )

    configured_max = float(
        _cfg_get(rgb_cfg, "speed_panel_max_mps", 0.0)
    )
    values = []
    if planned is not None:
        values.append(planned)
    if expert is not None:
        values.append(expert)

    automatic_max = max(
        [
            float(np.nanmax(value))
            for value in values
            if len(value) > 0 and np.isfinite(value).any()
        ]
        + [1.0]
    )
    max_speed = (
        max(configured_max, math.ceil(automatic_max + 1.0))
        if configured_max > 0
        else max(4.0, math.ceil(automatic_max + 1.0))
    )

    x0, y0, x1, y1 = 52, 30, panel_w - 24, panel_h - 44
    cv2.rectangle(
        panel,
        (x0, y0),
        (x1, y1),
        (230, 230, 230),
        1,
    )
    cv2.line(
        panel,
        (x0, y1),
        (x1, y1),
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        panel,
        (x0, y0),
        (x0, y1),
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    for fraction in (0.25, 0.50, 0.75, 1.0):
        yy = int(round(y1 - fraction * (y1 - y0)))
        cv2.line(
            panel,
            (x0, yy),
            (x1, yy),
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        _put_label(
            panel,
            f"{fraction * max_speed:.0f}",
            (8, yy + 5),
            0.42,
            (90, 90, 90),
            1,
        )

    _put_label(
        panel,
        "m/s",
        (12, y0 - 10),
        0.42,
        (90, 90, 90),
        1,
    )
    _put_label(
        panel,
        f"{time_values[0]:.1f}s",
        (x0 - 8, y1 + 28),
        0.42,
        (90, 90, 90),
        1,
    )
    _put_label(
        panel,
        f"{time_values[-1]:.1f}s",
        (x1 - 38, y1 + 28),
        0.42,
        (90, 90, 90),
        1,
    )

    if expert is not None:
        _draw_curve(
            panel,
            expert,
            time_values,
            (x0, y0, x1, y1),
            max_speed,
            (255, 0, 0),
            2,
            4,
        )
    if planned is not None:
        _draw_curve(
            panel,
            planned,
            time_values,
            (x0, y0, x1, y1),
            max_speed,
            (0, 160, 0),
            2,
            4,
        )

    return panel


def _resize_to_height(
    image: np.ndarray,
    target_h: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image

    new_w = max(
        1,
        int(
            round(
                float(w)
                * float(target_h)
                / max(float(h), 1.0)
            )
        ),
    )
    return cv2.resize(
        image,
        (new_w, int(target_h)),
        interpolation=cv2.INTER_AREA,
    )


# -----------------------------------------------------------------------------
# Main RGB debug output
# -----------------------------------------------------------------------------

def save_rgb_waypoints_debug_image(
    route_dir: Path,
    frame_name: str,
    risk_planned_waypoints: np.ndarray,
    expert_future_waypoints: Optional[np.ndarray],
    expert_reference_route: np.ndarray,
    selected_reference_route: np.ndarray,
    scored_candidates: List[Dict],
    selected_idx: int,
    selected_info: Dict,
    save_path: Path,
    cfg,
    risk_planned_speeds: Optional[np.ndarray] = None,
    factor: Optional[Dict] = None,
    occupancy_map: Optional[np.ndarray] = None,
    ego_center=None,
    meters_per_pixel: Optional[float] = None,
    actor_timelines: Optional[Dict[int, List[Dict]]] = None,
    causal_test_actors: Optional[List[Dict]] = None,
    language_annotation: Optional[Dict] = None,
) -> bool:
    if cv2 is None:
        return False

    surround_images, camera_specs = _load_surround_rgb_images(
        route_dir,
        frame_name,
        cfg,
    )

    # Backward-compatible fallback for old front-only routes.
    using_surround = surround_images is not None
    if using_surround:
        view_images = {
            name: image.copy()
            for name, image in surround_images.items()
        }
        front_image = view_images["front"]
        front_camera_spec = camera_specs["front"]
    else:
        front_path = find_rgb_image_path(route_dir, frame_name, cfg)
        if front_path is None:
            return False

        front_image = cv2.imread(str(front_path), cv2.IMREAD_COLOR)
        if front_image is None:
            return False

        front_camera_spec = _default_camera_spec(
            "front",
            cfg,
            front_image.shape[:2],
        )
        view_images = {"front": front_image}

    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    draw_route = bool(
        _cfg_get(rgb_cfg, "draw_reference_route", True)
    )
    draw_expert = bool(
        _cfg_get(rgb_cfg, "draw_expert_waypoints", True)
    )
    draw_all_candidates = bool(
        _cfg_get(rgb_cfg, "draw_all_candidates", False)
    )
    draw_text = bool(_cfg_get(rgb_cfg, "draw_text", True))
    make_composite = bool(
        _cfg_get(rgb_cfg, "make_composite", True)
    )

    actor_visual_entries, candidate_colors = _build_candidate_visual_colors(
        scored_candidates=scored_candidates,
        selected_idx=selected_idx,
        causal_test_actors=causal_test_actors,
    )
    attention_actor = _secondary_attention_actor(factor)

    # The route and every waypoint curve remain on the front view only.
    if draw_route:
        draw_projected_route_on_rgb(
            front_image,
            expert_reference_route,
            cfg,
            color=(255, 255, 255),
            thickness=2,
            camera_spec=front_camera_spec,
        )
        draw_projected_route_on_rgb(
            front_image,
            selected_reference_route,
            cfg,
            color=(0, 255, 255),
            thickness=2,
            camera_spec=front_camera_spec,
        )

    if draw_all_candidates:
        for index, candidate in enumerate(scored_candidates):
            if index == selected_idx:
                continue

            color = (
                candidate_colors[index]
                if index < len(candidate_colors)
                else _NON_ACTOR_CANDIDATE_PALETTE[
                    index % len(_NON_ACTOR_CANDIDATE_PALETTE)
                ]
            )
            draw_projected_waypoints_on_rgb(
                front_image,
                candidate["rollout"]["waypoints"],
                cfg,
                color=color,
                radius=2,
                thickness=1,
                camera_spec=front_camera_spec,
            )

    if draw_expert and expert_future_waypoints is not None:
        draw_projected_waypoints_on_rgb(
            front_image,
            expert_future_waypoints,
            cfg,
            color=(255, 0, 0),
            radius=4,
            thickness=2,
            camera_spec=front_camera_spec,
        )

    draw_projected_waypoints_on_rgb(
        front_image,
        risk_planned_waypoints,
        cfg,
        color=_SELECTED_TRAJECTORY_COLOR,
        radius=5,
        thickness=3,
        camera_spec=front_camera_spec,
    )

    if draw_text:
        intent_text = (
            f"intent: {selected_info.get('intent_name', 'unknown')}"
        )
        _put_label(
            front_image,
            intent_text,
            (10, 24),
            # 0.55,
            1.00,
            (255, 255, 255),
            1,
            outline=False,
        )

    # Draw exactly the original selected actor boxes. With six-view data, the
    # same boxes are additionally projected into every other camera in which
    # their cuboid is visible.
    if using_surround:
        for row in _SURROUND_VIEW_LAYOUT:
            for view_name in row:
                _draw_original_actor_boxes_on_view(
                    view_images[view_name],
                    camera_specs[view_name],
                    actor_visual_entries,
                    attention_actor,
                    cfg,
                )
        center_panel = _make_surround_mosaic(view_images, cfg)
    else:
        _draw_original_actor_boxes_on_view(
            front_image,
            front_camera_spec,
            actor_visual_entries,
            attention_actor,
            cfg,
        )
        center_panel = front_image

    output_image = center_panel

    if make_composite:
        planned_speeds = risk_planned_speeds
        if planned_speeds is None:
            planned_speeds = _speed_from_waypoints(
                risk_planned_waypoints,
                float(cfg.horizon.future_fps),
            )
        expert_speeds = _speed_from_waypoints(
            expert_future_waypoints,
            float(cfg.horizon.future_fps),
        )

        selected_yaws = None
        if 0 <= int(selected_idx) < len(scored_candidates):
            try:
                selected_yaws = (
                    scored_candidates[int(selected_idx)]
                    .get("rollout", {})
                    .get("yaws", None)
                )
            except Exception:
                selected_yaws = None

        if using_surround:
            # New overall layout:
            #   row 1-2: large 2 x 3 surround RGB mosaic
            #   row 3:   BEV diagnostic panel + speed panel
            #   row 4:   four language questions
            #
            # This keeps all six RGB views together at the largest visual scale
            # instead of placing narrow BEV/speed columns beside them.
            mosaic_w = int(center_panel.shape[1])
            rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
            diagnostic_h = max(
                260,
                int(
                    _cfg_get(
                        rgb_cfg,
                        "surround_diagnostic_panel_height",
                        520,
                    )
                ),
            )
            diagnostic_gap = max(
                0,
                int(
                    _cfg_get(
                        rgb_cfg,
                        "surround_diagnostic_gap",
                        12,
                    )
                ),
            )
            default_bev_w = min(520, max(360, mosaic_w // 3))
            bev_w = int(
                _cfg_get(
                    rgb_cfg,
                    "surround_bev_panel_width",
                    default_bev_w,
                )
            )
            bev_w = int(
                np.clip(
                    bev_w,
                    300,
                    max(300, mosaic_w - diagnostic_gap - 320),
                )
            )
            speed_w = max(
                320,
                mosaic_w - diagnostic_gap - bev_w,
            )

            left_panel = make_compact_critical_bev_panel(
                factor=factor,
                cfg=cfg,
                target_h=diagnostic_h,
                target_w=bev_w,
                occupancy_map=occupancy_map,
                ego_center=ego_center,
                meters_per_pixel=meters_per_pixel,
                planned_waypoints=risk_planned_waypoints,
                planned_yaws=selected_yaws,
                actor_timelines=actor_timelines,
            )
            speed_panel = make_speed_panel(
                planned_speeds,
                expert_speeds,
                cfg,
                diagnostic_h,
                target_w=speed_w,
            )

            if diagnostic_gap > 0:
                gap_panel = np.full(
                    (diagnostic_h, diagnostic_gap, 3),
                    250,
                    dtype=np.uint8,
                )
                diagnostic_row = np.concatenate(
                    [left_panel, gap_panel, speed_panel],
                    axis=1,
                )
            else:
                diagnostic_row = np.concatenate(
                    [left_panel, speed_panel],
                    axis=1,
                )

            # Integer clipping above normally makes the widths exact. Keep a
            # defensive resize so the diagnostic row always matches the mosaic.
            if diagnostic_row.shape[1] != mosaic_w:
                diagnostic_row = cv2.resize(
                    diagnostic_row,
                    (mosaic_w, diagnostic_h),
                    interpolation=cv2.INTER_AREA,
                )

            output_image = np.concatenate(
                [center_panel, diagnostic_row],
                axis=0,
            )
        else:
            # Preserve the original horizontal layout for legacy front-only
            # routes.
            target_h = center_panel.shape[0]
            left_panel = make_compact_critical_bev_panel(
                factor=factor,
                cfg=cfg,
                target_h=target_h,
                occupancy_map=occupancy_map,
                ego_center=ego_center,
                meters_per_pixel=meters_per_pixel,
                planned_waypoints=risk_planned_waypoints,
                planned_yaws=selected_yaws,
                actor_timelines=actor_timelines,
            )
            speed_panel = make_speed_panel(
                planned_speeds,
                expert_speeds,
                cfg,
                target_h,
            )

            left_panel = _resize_to_height(left_panel, target_h)
            speed_panel = _resize_to_height(speed_panel, target_h)
            output_image = np.concatenate(
                [left_panel, center_panel, speed_panel],
                axis=1,
            )

    language_panel = make_language_qa_panel(
        language_annotation,
        output_image.shape[1],
        cfg,
    )
    if language_panel is not None:
        output_image = np.concatenate(
            [output_image, language_panel],
            axis=0,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(save_path), output_image))
