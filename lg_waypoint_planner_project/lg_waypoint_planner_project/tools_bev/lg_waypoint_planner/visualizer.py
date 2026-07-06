# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .geometry import local_points_to_pixels
from .footprint_collision import obb_corners


def normalize_costmap(costmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(costmap, dtype=np.float32)
    vmax = float(np.percentile(arr, 98.0)) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 1e-6)
    img = np.clip(arr / vmax * 255.0, 0, 255).astype(np.uint8)
    return img


def draw_polyline(img, pts, ego_center, mpp, color, thickness=2):
    if cv2 is None or pts is None or len(pts) < 2:
        return
    pix = local_points_to_pixels(np.asarray(pts, dtype=np.float32), ego_center, mpp)
    pix = np.round(pix).astype(np.int32)
    for i in range(len(pix) - 1):
        p1 = tuple(int(v) for v in pix[i]); p2 = tuple(int(v) for v in pix[i + 1])
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
        cv2.line(img, tuple(pix[i]), tuple(pix[(i + 1) % 4]), color, thickness, cv2.LINE_AA)


def save_bev_debug(path: Path, costmap, base_route, scored_candidates, selected_idx, actor_timelines, ego_center, meters_per_pixel, cfg):
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
    for i, c in enumerate(scored_candidates):
        name = c["info"].get("intent_name", "unknown")
        color = colors.get(name, (120, 120, 120))
        th = 3 if i == selected_idx else 1
        draw_polyline(img, c["rollout"]["waypoints"], ego_center, meters_per_pixel, color, th)
        draw_points(img, c["rollout"]["waypoints"], ego_center, meters_per_pixel, color, 2 if i != selected_idx else 4)

    for _, actors in actor_timelines.items():
        for a in actors:
            draw_obb(img, [a.get("x_m", 0.0), a.get("y_m", 0.0)], a.get("yaw_rad", 0.0), a.get("half_length_m", 2.0), a.get("half_width_m", 1.0), ego_center, meters_per_pixel, (0, 0, 255), 1)

    cv2.circle(img, tuple(int(v) for v in ego_center), 4, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


# -----------------------------------------------------------------------------
# RGB waypoint projection debug
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


def build_projection_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = float(width) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    return np.asarray([
        [f, 0.0, float(width) / 2.0],
        [0.0, f, float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def rotation_matrix_roll_pitch_yaw_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def project_ego_local_points_to_rgb(points_local_xy: np.ndarray, image_shape: Tuple[int, int], cfg) -> Tuple[np.ndarray, np.ndarray]:
    points_local_xy = np.asarray(points_local_xy, dtype=np.float32)
    if points_local_xy.ndim != 2 or points_local_xy.shape[1] < 2 or len(points_local_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    h, w = int(image_shape[0]), int(image_shape[1])
    K = build_projection_matrix(w, h, float(_cfg_get(rgb_cfg, "camera_fov", 110.0)))

    z = np.full((len(points_local_xy), 1), float(_cfg_get(rgb_cfg, "waypoint_ground_z_m", 0.0)), dtype=np.float32)
    points_ego = np.concatenate([points_local_xy[:, :2], z], axis=1)

    cam_t_ego = np.asarray([
        float(_cfg_get(rgb_cfg, "camera_x", -1.5)),
        float(_cfg_get(rgb_cfg, "camera_y", 0.0)),
        float(_cfg_get(rgb_cfg, "camera_z", 2.0)),
    ], dtype=np.float32)

    R_cam_to_ego = rotation_matrix_roll_pitch_yaw_deg(
        roll_deg=float(_cfg_get(rgb_cfg, "camera_roll_deg", 0.0)),
        pitch_deg=float(_cfg_get(rgb_cfg, "camera_pitch_deg", 0.0)),
        yaw_deg=float(_cfg_get(rgb_cfg, "camera_yaw_deg", 0.0)),
    )

    points_rel = points_ego - cam_t_ego[None, :]
    points_cam = (R_cam_to_ego.T @ points_rel.T).T

    depth = points_cam[:, 0]
    right = points_cam[:, 1]
    up = points_cam[:, 2]
    pixels = np.zeros((len(points_cam), 2), dtype=np.float32)

    valid = depth > float(_cfg_get(rgb_cfg, "min_projection_depth_m", 0.1))
    if np.any(valid):
        pixels[valid, 0] = K[0, 0] * right[valid] / depth[valid] + K[0, 2]
        pixels[valid, 1] = K[1, 2] - K[1, 1] * up[valid] / depth[valid]

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


def draw_projected_waypoints_on_rgb(image: np.ndarray, points_local_xy: Optional[np.ndarray], cfg, color: Tuple[int, int, int], radius: int, thickness: int) -> int:
    if points_local_xy is None:
        return 0
    points = np.asarray(points_local_xy, dtype=np.float32)
    if len(points) == 0:
        return 0
    pixels, valid = project_ego_local_points_to_rgb(points, image.shape[:2], cfg)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return 0
    pts = np.round(pixels).astype(np.int32)
    for i in range(len(pts) - 1):
        if valid[i] and valid[i + 1]:
            cv2.line(image, (int(pts[i, 0]), int(pts[i, 1])), (int(pts[i + 1, 0]), int(pts[i + 1, 1])), color, thickness, cv2.LINE_AA)
    for rank, idx in enumerate(valid_indices):
        p = (int(pts[idx, 0]), int(pts[idx, 1]))
        r = radius + 1 if rank == 0 else radius
        cv2.circle(image, p, r, color, -1, cv2.LINE_AA)
    return int(len(valid_indices))


def draw_projected_route_on_rgb(image: np.ndarray, route_local_xy: Optional[np.ndarray], cfg, color: Tuple[int, int, int], thickness: int) -> int:
    if route_local_xy is None:
        return 0
    route = np.asarray(route_local_xy, dtype=np.float32)
    if len(route) < 2:
        return 0
    pixels, valid = project_ego_local_points_to_rgb(route, image.shape[:2], cfg)
    pts = np.round(pixels).astype(np.int32)
    for i in range(len(pts) - 1):
        if valid[i] and valid[i + 1]:
            cv2.line(image, (int(pts[i, 0]), int(pts[i, 1])), (int(pts[i + 1, 0]), int(pts[i + 1, 1])), color, thickness, cv2.LINE_AA)
    return int(np.sum(valid))


def find_rgb_image_path(route_dir: Path, frame_name: str, cfg) -> Optional[Path]:
    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    folders = []
    folder = _cfg_get(rgb_cfg, "rgb_folder", None)
    if folder:
        folders.append(str(folder))
    for extra in ["rgb", "rgb_front", "CAM_FRONT", "cam_front"]:
        if extra not in folders:
            folders.append(extra)
    for folder_name in folders:
        rgb_dir = route_dir / folder_name
        for suffix in [".jpg", ".png", ".jpeg"]:
            p = rgb_dir / f"{frame_name}{suffix}"
            if p.exists():
                return p
    return None


# -----------------------------------------------------------------------------
# Composite panel helpers: left compact BEV + center RGB + right speed profile
# -----------------------------------------------------------------------------

def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = max(1, int(round(float(w) * float(target_h) / max(float(h), 1.0))))
    return cv2.resize(img, (new_w, int(target_h)), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.55, color: Tuple[int, int, int] = (30, 30, 30), thickness: int = 1):
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _bev_to_px(x: float, y: float, width: int, height: int, scale: float, origin_u: float, origin_v: float) -> Tuple[int, int]:
    # x: forward, y: right. Image top means forward.
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
    pix = np.asarray([_bev_to_px(float(p[0]), float(p[1]), panel.shape[1], panel.shape[0], scale, origin_u, origin_v) for p in corners], dtype=np.int32)
    if fill_color is not None:
        cv2.fillPoly(panel, [pix], fill_color, cv2.LINE_AA)
    cv2.polylines(panel, [pix], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_arrow(panel: np.ndarray, start_xy: Tuple[float, float], end_xy: Tuple[float, float], scale: float, origin_u: float, origin_v: float, color: Tuple[int, int, int], thickness: int = 2):
    p0 = _bev_to_px(start_xy[0], start_xy[1], panel.shape[1], panel.shape[0], scale, origin_u, origin_v)
    p1 = _bev_to_px(end_xy[0], end_xy[1], panel.shape[1], panel.shape[0], scale, origin_u, origin_v)
    cv2.arrowedLine(panel, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.18)



def _get_occupancy_blocked_threshold(cfg) -> float:
    scoring = _cfg_get(cfg, "scoring", {})
    for key in ["occupancy_blocked_threshold", "hard_cost_threshold"]:
        val = _cfg_get(scoring, key, None)
        if val is not None:
            try:
                return float(val)
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
    """Draw the current free/blocked BEV occupancy map on the compact panel.

    The input costmap is now treated as a binary occupancy map: low values are
    drivable/free and high values are blocked.  The visualization is only for
    checking whether the selected ego footprints leave the free road area or
    overlap occupied regions.
    """
    if occupancy_map is None or ego_center is None or meters_per_pixel is None:
        return
    occ = np.asarray(occupancy_map, dtype=np.float32)
    if occ.ndim != 2 or occ.size == 0:
        return

    h, w = panel.shape[:2]
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    local_y = (uu - float(origin_u)) / max(float(scale), 1e-6)
    local_x = (float(origin_v) - vv) / max(float(scale), 1e-6)

    cx, cy = float(ego_center[0]), float(ego_center[1])
    col = np.round(cx + local_y / float(meters_per_pixel)).astype(np.int32)
    row = np.round(cy - local_x / float(meters_per_pixel)).astype(np.int32)
    inside = (row >= 0) & (row < occ.shape[0]) & (col >= 0) & (col < occ.shape[1])

    blocked_threshold = _get_occupancy_blocked_threshold(cfg)
    free = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    values = np.zeros((h, w), dtype=np.float32)
    values[inside] = occ[row[inside], col[inside]]
    free[inside] = values[inside] < blocked_threshold
    blocked[inside] = values[inside] >= blocked_threshold

    # Light, low-contrast colors so the ego and actor footprints remain readable.
    panel[~inside] = (232, 232, 232)
    panel[free] = (225, 242, 225)
    panel[blocked] = (224, 224, 238)

    # Sparse grid helps judge relative position without making the panel crowded.
    grid_m = 5.0
    forward_m = (float(origin_v) - np.arange(h, dtype=np.float32)) / max(float(scale), 1e-6)
    right_m = (np.arange(w, dtype=np.float32) - float(origin_u)) / max(float(scale), 1e-6)
    for xm in np.arange(math.floor(forward_m.min() / grid_m) * grid_m, forward_m.max() + grid_m, grid_m):
        _, v = _bev_to_px(float(xm), 0.0, w, h, scale, origin_u, origin_v)
        if 0 <= v < h:
            cv2.line(panel, (0, v), (w - 1, v), (214, 214, 214), 1, cv2.LINE_AA)
    for ym in np.arange(math.floor(right_m.min() / grid_m) * grid_m, right_m.max() + grid_m, grid_m):
        u, _ = _bev_to_px(0.0, float(ym), w, h, scale, origin_u, origin_v)
        if 0 <= u < w:
            cv2.line(panel, (u, 0), (u, h - 1), (214, 214, 214), 1, cv2.LINE_AA)


def _yaw_sequence_from_waypoints(points: Optional[np.ndarray], yaws: Optional[np.ndarray] = None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32) if points is not None else np.zeros((0, 2), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return np.zeros((0,), dtype=np.float32)
    if yaws is not None:
        yy = np.asarray(yaws, dtype=np.float32).reshape(-1)
        if len(yy) >= len(pts):
            return yy[:len(pts)].astype(np.float32)
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), pts[:-1, :2]], axis=0)
    delta = pts[:, :2] - prev
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


def _match_critical_actor_at_step(factor: Optional[Dict], actor_timelines: Optional[Dict[int, List[Dict]]], step_idx: int) -> Optional[Dict]:
    if not isinstance(actor_timelines, dict):
        return None
    actor_id = _critical_actor_id(factor)
    actors = actor_timelines.get(int(step_idx), []) or []
    if actor_id is not None:
        for a in actors:
            if a.get("id", None) == actor_id:
                return a
    # Fallback: choose the future actor closest to the highlighted actor.
    actor = _display_actor(factor)
    if isinstance(actor, dict) and bool(actor.get("exists", False)) and actors:
        ax = float(actor.get("x_m", 0.0)); ay = float(actor.get("y_m", 0.0))
        def dist(a):
            return (float(a.get("x_m", 0.0)) - ax) ** 2 + (float(a.get("y_m", 0.0)) - ay) ** 2
        return min(actors, key=dist)
    return None


def make_compact_critical_bev_panel(
    factor: Optional[Dict],
    cfg,
    target_h: int,
    occupancy_map: Optional[np.ndarray] = None,
    ego_center=None,
    meters_per_pixel: Optional[float] = None,
    planned_waypoints: Optional[np.ndarray] = None,
    planned_yaws: Optional[np.ndarray] = None,
    actor_timelines: Optional[Dict[int, List[Dict]]] = None,
) -> np.ndarray:
    """Left diagnostic panel for the RGB debug image.

    It renders the free road area from the occupancy BEV, the selected ego
    footprints at every future point, and the critical actor footprint at the
    corresponding future frame.  This panel is intended for checking whether the
    generated motion overlaps occupied areas or leaves the drivable area.
    """
    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    panel_w = int(_cfg_get(rgb_cfg, "left_bev_panel_width", 420))
    panel_h = int(target_h)
    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)

    forward_m = float(_cfg_get(rgb_cfg, "left_bev_forward_m", 35.0))
    rear_m = float(_cfg_get(rgb_cfg, "left_bev_rear_m", 8.0))
    side_m = float(_cfg_get(rgb_cfg, "left_bev_side_m", 12.0))
    margin = 34.0
    scale_x = (panel_h - 2.0 * margin) / max(forward_m + rear_m, 1.0)
    scale_y = (panel_w - 2.0 * margin) / max(2.0 * side_m, 1.0)
    scale = float(min(scale_x, scale_y))
    origin_u = panel_w * 0.5
    origin_v = margin + forward_m * scale

    _draw_occupancy_background_on_panel(
        panel=panel,
        occupancy_map=occupancy_map,
        ego_center=ego_center,
        meters_per_pixel=meters_per_pixel,
        scale=scale,
        origin_u=origin_u,
        origin_v=origin_v,
        cfg=cfg,
    )

    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (190, 190, 190), 1)
    _draw_arrow(panel, (0.0, 0.0), (6.0, 0.0), scale, origin_u, origin_v, (70, 70, 70), 2)
    # _put_label(panel, "free area + footprints", (12, 26), 0.55, (30, 30, 30), 1)
    # _put_label(panel, "front", _bev_to_px(7.2, 0.3, panel_w, panel_h, scale, origin_u, origin_v), 0.45, (70, 70, 70), 1)

    ego_half_l = float(cfg.vehicle.ego_half_length_m)
    ego_half_w = float(cfg.vehicle.ego_half_width_m)

    # Current ego footprint.
    _draw_local_obb_on_panel(panel, (0.0, 0.0), 0.0, ego_half_l, ego_half_w, scale, origin_u, origin_v, (0, 120, 0), 2, (205, 235, 205))
    # _put_label(panel, "ego now", _bev_to_px(-1.0, 1.4, panel_w, panel_h, scale, origin_u, origin_v), 0.42, (0, 100, 0), 1)

    # Selected ego footprints at every generated future point.
    pts = np.asarray(planned_waypoints, dtype=np.float32) if planned_waypoints is not None else np.zeros((0, 2), dtype=np.float32)
    yaws = _yaw_sequence_from_waypoints(pts, planned_yaws)
    if pts.ndim == 2 and pts.shape[1] >= 2 and len(pts) > 0:
        pix_pts = []
        for k, p in enumerate(pts[:, :2], start=1):
            yaw = float(yaws[k - 1]) if len(yaws) >= k else 0.0
            _draw_local_obb_on_panel(panel, (float(p[0]), float(p[1])), yaw, ego_half_l, ego_half_w, scale, origin_u, origin_v, (0, 145, 0), 1)
            pix_pts.append(_bev_to_px(float(p[0]), float(p[1]), panel_w, panel_h, scale, origin_u, origin_v))
        for i in range(len(pix_pts) - 1):
            cv2.line(panel, pix_pts[i], pix_pts[i + 1], (0, 160, 0), 2, cv2.LINE_AA)
        for i, pp in enumerate(pix_pts, start=1):
            cv2.circle(panel, pp, 3, (0, 120, 0), -1, cv2.LINE_AA)
            _put_label(panel, str(i), (pp[0] + 4, pp[1] - 4), 0.35, (0, 100, 0), 1)

    # Primary causal actor, or secondary attention actor when no causal actor
    # exists, at the current frame plus corresponding future frames.
    current_actor = _display_actor(factor)
    if isinstance(current_actor, dict) and bool(current_actor.get("exists", False)):
        ax = float(current_actor.get("x_m", 0.0))
        ay = float(current_actor.get("y_m", 0.0))
        yaw = float(current_actor.get("yaw_rad", 0.0))
        half_l = float(current_actor.get("half_length_m", 2.25))
        half_w = float(current_actor.get("half_width_m", 1.0))
        cls = str(current_actor.get("class", "actor"))
        rel = str(current_actor.get("relative_position", "nearby"))
        _draw_local_obb_on_panel(panel, (ax, ay), yaw, half_l, half_w, scale, origin_u, origin_v, (30, 30, 210), 2, (210, 214, 255))
        label_u, label_v = _bev_to_px(ax, ay, panel_w, panel_h, scale, origin_u, origin_v)
        label_u = int(np.clip(label_u + 8, 8, panel_w - 150))
        label_v = int(np.clip(label_v - 8, 50, panel_h - 28))
        _put_label(panel, f"{cls} / {rel}", (label_u, label_v), 0.42, (30, 30, 170), 1)

        n_steps = len(pts) if pts.ndim == 2 else int(_cfg_get(cfg.horizon, "num_future_waypoints", 0))
        for k in range(1, int(n_steps) + 1):
            a = _match_critical_actor_at_step(factor, actor_timelines, k)
            if not isinstance(a, dict):
                continue
            bx = float(a.get("x_m", 0.0)); by = float(a.get("y_m", 0.0))
            byaw = float(a.get("yaw_rad", 0.0))
            bhl = float(a.get("half_length_m", half_l)); bhw = float(a.get("half_width_m", half_w))
            _draw_local_obb_on_panel(panel, (bx, by), byaw, bhl, bhw, scale, origin_u, origin_v, (40, 40, 220), 1)
            bp = _bev_to_px(bx, by, panel_w, panel_h, scale, origin_u, origin_v)
            cv2.circle(panel, bp, 2, (40, 40, 220), -1, cv2.LINE_AA)
    else:
        _put_label(panel, "no dominant actor", (12, panel_h - 34), 0.50, (80, 80, 80), 1)

    # Compact legend.
    legend_y = panel_h - 14
    cv2.rectangle(panel, (12, legend_y - 8), (24, legend_y + 4), (225, 242, 225), -1)
    _put_label(panel, "free", (30, legend_y + 3), 0.35, (70, 90, 70), 1)
    cv2.rectangle(panel, (78, legend_y - 8), (90, legend_y + 4), (224, 224, 238), -1)
    _put_label(panel, "blocked", (96, legend_y + 3), 0.35, (90, 90, 110), 1)
    cv2.line(panel, (164, legend_y - 2), (194, legend_y - 2), (0, 150, 0), 2, cv2.LINE_AA)
    _put_label(panel, "ego", (200, legend_y + 3), 0.35, (0, 110, 0), 1)
    cv2.line(panel, (246, legend_y - 2), (276, legend_y - 2), (30, 30, 210), 2, cv2.LINE_AA)
    _put_label(panel, "actor", (282, legend_y + 3), 0.35, (30, 30, 150), 1)

    return panel

def _speed_from_waypoints(points: Optional[np.ndarray], future_fps: float) -> Optional[np.ndarray]:
    if points is None:
        return None
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return None
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), pts[:-1, :2]], axis=0)
    return (np.linalg.norm(pts[:, :2] - prev, axis=1) * float(future_fps)).astype(np.float32)


def _draw_curve(panel: np.ndarray, values: np.ndarray, t: np.ndarray, plot_rect: Tuple[int, int, int, int], max_speed: float, color: Tuple[int, int, int], thickness: int = 2, radius: int = 4):
    if values is None or len(values) == 0:
        return
    x0, y0, x1, y1 = plot_rect
    denom_t = max(float(t[-1] - t[0]), 1e-6) if len(t) > 1 else 1.0
    pts = []
    for i, v in enumerate(values):
        ti = float(t[i]) if i < len(t) else float(i + 1)
        px = x0 + (ti - float(t[0])) / denom_t * (x1 - x0)
        py = y1 - np.clip(float(v), 0.0, max_speed) / max(max_speed, 1e-6) * (y1 - y0)
        pts.append((int(round(px)), int(round(py))))
    for i in range(len(pts) - 1):
        cv2.line(panel, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(panel, p, radius, color, -1, cv2.LINE_AA)


def make_speed_panel(
    planned_speeds: Optional[np.ndarray],
    expert_speeds: Optional[np.ndarray],
    cfg,
    target_h: int,
) -> np.ndarray:
    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    panel_w = int(_cfg_get(rgb_cfg, "speed_panel_width", 420))
    panel_h = int(target_h)
    panel = np.full((panel_h, panel_w, 3), 250, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (210, 210, 210), 1)
    _put_label(panel, "speed profile", (16, 30), 0.65, (30, 30, 30), 2)

    planned = None if planned_speeds is None else np.asarray(planned_speeds, dtype=np.float32).reshape(-1)
    expert = None if expert_speeds is None else np.asarray(expert_speeds, dtype=np.float32).reshape(-1)
    n = max(len(planned) if planned is not None else 0, len(expert) if expert is not None else 0)
    if n == 0:
        _put_label(panel, "no speed data", (16, 70), 0.6, (80, 80, 80), 1)
        return panel

    fps = float(cfg.horizon.future_fps)
    t = (np.arange(n, dtype=np.float32) + 1.0) / max(fps, 1e-6)
    if planned is not None and len(planned) < n:
        planned = np.pad(planned, (0, n - len(planned)), mode="edge")
    if expert is not None and len(expert) < n:
        expert = np.pad(expert, (0, n - len(expert)), mode="edge")

    max_cfg = float(_cfg_get(rgb_cfg, "speed_panel_max_mps", 0.0))
    vals = []
    if planned is not None:
        vals.append(planned)
    if expert is not None:
        vals.append(expert)
    auto_max = max([float(np.nanmax(v)) for v in vals if len(v) > 0 and np.isfinite(v).any()] + [1.0])
    max_speed = max(max_cfg, math.ceil(auto_max + 1.0)) if max_cfg > 0 else max(4.0, math.ceil(auto_max + 1.0))

    x0, y0, x1, y1 = 52, 58, panel_w - 24, panel_h - 56
    cv2.rectangle(panel, (x0, y0), (x1, y1), (230, 230, 230), 1)
    cv2.line(panel, (x0, y1), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(panel, (x0, y0), (x0, y1), (80, 80, 80), 1, cv2.LINE_AA)

    for frac in [0.25, 0.50, 0.75, 1.0]:
        yy = int(round(y1 - frac * (y1 - y0)))
        cv2.line(panel, (x0, yy), (x1, yy), (225, 225, 225), 1, cv2.LINE_AA)
        _put_label(panel, f"{frac * max_speed:.0f}", (8, yy + 5), 0.42, (90, 90, 90), 1)
    _put_label(panel, "m/s", (12, y0 - 10), 0.42, (90, 90, 90), 1)
    _put_label(panel, f"{t[0]:.1f}s", (x0 - 8, y1 + 28), 0.42, (90, 90, 90), 1)
    _put_label(panel, f"{t[-1]:.1f}s", (x1 - 38, y1 + 28), 0.42, (90, 90, 90), 1)

    if expert is not None:
        _draw_curve(panel, expert, t, (x0, y0, x1, y1), max_speed, (255, 0, 0), 2, 4)
    if planned is not None:
        _draw_curve(panel, planned, t, (x0, y0, x1, y1), max_speed, (0, 160, 0), 2, 4)

    legend_y = panel_h - 18
    cv2.line(panel, (20, legend_y), (50, legend_y), (0, 160, 0), 3, cv2.LINE_AA)
    _put_label(panel, "planned", (58, legend_y + 5), 0.45, (30, 120, 30), 1)
    cv2.line(panel, (160, legend_y), (190, legend_y), (255, 0, 0), 3, cv2.LINE_AA)
    _put_label(panel, "expert", (198, legend_y + 5), 0.45, (180, 40, 40), 1)
    return panel


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
) -> bool:
    if cv2 is None:
        return False
    rgb_path = find_rgb_image_path(route_dir, frame_name, cfg)
    if rgb_path is None:
        return False
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    rgb_cfg = _cfg_get(cfg.debug, "rgb", {})
    draw_route = bool(_cfg_get(rgb_cfg, "draw_reference_route", True))
    draw_expert = bool(_cfg_get(rgb_cfg, "draw_expert_waypoints", True))
    draw_all_candidates = bool(_cfg_get(rgb_cfg, "draw_all_candidates", False))
    draw_text = bool(_cfg_get(rgb_cfg, "draw_text", True))
    make_composite = bool(_cfg_get(rgb_cfg, "make_composite", True))

    if draw_route:
        draw_projected_route_on_rgb(img, expert_reference_route, cfg, color=(255, 255, 255), thickness=2)
        draw_projected_route_on_rgb(img, selected_reference_route, cfg, color=(0, 255, 255), thickness=2)

    if draw_all_candidates:
        for i, cand in enumerate(scored_candidates):
            if i == selected_idx:
                continue
            draw_projected_waypoints_on_rgb(img, cand["rollout"]["waypoints"], cfg, color=(160, 160, 160), radius=2, thickness=1)

    if draw_expert and expert_future_waypoints is not None:
        draw_projected_waypoints_on_rgb(img, expert_future_waypoints, cfg, color=(255, 0, 0), radius=4, thickness=2)

    num_selected = draw_projected_waypoints_on_rgb(img, risk_planned_waypoints, cfg, color=(0, 255, 0), radius=5, thickness=3)

    if draw_text:
        lines = [
            "green: planned waypoints",
            "blue: expert waypoints",
            "yellow: selected route",
            f"intent: {selected_info.get('intent_name', 'unknown')}",
            f"projected planned points: {num_selected}",
        ]
        x0, y0, gap = 10, 24, 20
        for i, text in enumerate(lines):
            org = (x0, y0 + i * gap)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    if make_composite:
        target_h = img.shape[0]
        planned_speeds = risk_planned_speeds
        if planned_speeds is None:
            planned_speeds = _speed_from_waypoints(risk_planned_waypoints, float(cfg.horizon.future_fps))
        expert_speeds = _speed_from_waypoints(expert_future_waypoints, float(cfg.horizon.future_fps))
        selected_yaws = None
        if 0 <= int(selected_idx) < len(scored_candidates):
            try:
                selected_yaws = scored_candidates[int(selected_idx)].get("rollout", {}).get("yaws", None)
            except Exception:
                selected_yaws = None
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
        speed_panel = make_speed_panel(planned_speeds, expert_speeds, cfg, target_h)
        left_panel = _resize_to_height(left_panel, target_h)
        speed_panel = _resize_to_height(speed_panel, target_h)
        img = np.concatenate([left_panel, img, speed_panel], axis=1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)
    return True
