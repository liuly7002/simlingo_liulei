# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .dataset import get_pose_global_xy_yaw, get_meters_per_pixel, get_ego_center, offset_frame_name
from .io_utils import load_costmap, load_json_gz
from .geometry import local_points_to_pixels


def sample_bilinear(cost: np.ndarray, pixels: np.ndarray, out_of_bounds_cost: float) -> tuple:
    """Bilinear sampler kept for compatibility with diagnostic modules.

    The main candidate evaluator now uses nearest-neighbor sampling because the
    simplified costmap is an occupancy BEV.  This function is still used by the
    coarse critical-factor diagnosis and older debug utilities.
    """
    h, w = cost.shape[:2]
    pix = np.asarray(pixels, dtype=np.float32)
    col, row = pix[:, 0], pix[:, 1]
    valid = (col >= 0) & (col <= w - 1) & (row >= 0) & (row <= h - 1)
    out = np.full((len(pix),), float(out_of_bounds_cost), dtype=np.float32)
    if not np.any(valid):
        return out, valid
    c = col[valid]
    r = row[valid]
    c0 = np.floor(c).astype(np.int32)
    r0 = np.floor(r).astype(np.int32)
    c1 = np.clip(c0 + 1, 0, w - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    dc = c - c0
    dr = r - r0
    v00 = cost[r0, c0]
    v01 = cost[r0, c1]
    v10 = cost[r1, c0]
    v11 = cost[r1, c1]
    v0 = v00 * (1 - dc) + v01 * dc
    v1 = v10 * (1 - dc) + v11 * dc
    out[valid] = (v0 * (1 - dr) + v1 * dr).astype(np.float32)
    return out, valid


def current_local_grid(shape, ego_center, meters_per_pixel) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    rows, cols = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    x = (float(ego_center[1]) - rows) * float(meters_per_pixel)
    y = (cols - float(ego_center[0])) * float(meters_per_pixel)
    return np.stack([x, y], axis=-1).astype(np.float32)


def transform_grid_current_to_source(grid_xy: np.ndarray, current_pose, source_pose) -> np.ndarray:
    current_pos, current_yaw = current_pose
    source_pos, source_yaw = source_pose
    x = grid_xy[..., 0]
    y = grid_xy[..., 1]
    c0, s0 = np.cos(current_yaw), np.sin(current_yaw)
    world_x = current_pos[0] + c0 * x - s0 * y
    world_y = current_pos[1] + s0 * x + c0 * y
    dx = world_x - source_pos[0]
    dy = world_y - source_pos[1]
    cs, ss = np.cos(source_yaw), np.sin(source_yaw)
    src_x = dx * cs + dy * ss
    src_y = -dx * ss + dy * cs
    return np.stack([src_x, src_y], axis=-1).astype(np.float32)


def warp_costmap_to_current(source_costmap: np.ndarray, current_shape, current_ego_center, current_mpp, current_pose, source_pose, source_ego_center, source_mpp, border_cost: float) -> np.ndarray:
    if cv2 is None:
        return source_costmap.copy()
    grid = current_local_grid(current_shape, current_ego_center, current_mpp)
    src_local = transform_grid_current_to_source(grid, current_pose=current_pose, source_pose=source_pose)
    src_pixels = local_points_to_pixels(src_local.reshape(-1, 2), source_ego_center, source_mpp).reshape(current_shape[0], current_shape[1], 2)
    map_x = src_pixels[..., 0].astype(np.float32)
    map_y = src_pixels[..., 1].astype(np.float32)
    return cv2.remap(source_costmap.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_cost)).astype(np.float32)


def combine_current_future(current: np.ndarray, future: np.ndarray, cfg) -> np.ndarray:
    """Combine current and future occupancy BEV maps.

    The files under costmap/*.npy are treated as simple occupancy maps rather
    than continuous risk fields.  ``max`` therefore means union of blocked
    regions across the current and warped future map.
    """
    mode = str(cfg.costmap.future_combine)
    w = float(cfg.costmap.future_weight)
    if mode == "future_only":
        out = w * future
    elif mode == "add":
        out = current + w * future
    else:
        out = np.maximum(current, w * future)
    clip_max = float(cfg.costmap.clip_max)
    if clip_max > 0:
        out = np.clip(out, 0.0, clip_max)
    return out.astype(np.float32)


def build_temporal_costmaps(route_dir: Path, frame_name: str, current_measurement: Dict, current_costmap: np.ndarray, current_meta: Dict, cfg) -> Dict:
    frames = [frame_name]
    costmaps = [current_costmap.astype(np.float32)]
    valid = [True]
    reasons = [""]
    if bool(cfg.costmap.disable_future_costmaps):
        return {"enabled": False, "frames": frames, "costmaps": costmaps, "valid": valid, "missing_reasons": reasons}

    current_pose = get_pose_global_xy_yaw(current_measurement)
    if current_pose is None:
        return {"enabled": False, "frames": frames, "costmaps": costmaps, "valid": valid, "missing_reasons": ["current_pose_missing"]}

    current_mpp = get_meters_per_pixel(current_meta, float(cfg.paths.default_pixels_per_meter))
    current_center = get_ego_center(current_meta, current_costmap.shape)
    last = current_costmap.astype(np.float32)

    for k in range(1, int(cfg.horizon.num_future_waypoints) + 1):
        future_name = offset_frame_name(frame_name, k, int(cfg.horizon.future_frame_stride))
        frames.append(str(future_name))
        if future_name is None:
            costmaps.append(last if cfg.costmap.future_missing_policy == "repeat_last" else current_costmap)
            valid.append(False); reasons.append("non_numeric_frame")
            continue
        cm_path = route_dir / cfg.paths.costmap_folder / f"{future_name}.npy"
        ms_path = route_dir / cfg.paths.measurements_folder / f"{future_name}.json.gz"
        meta_path = route_dir / cfg.paths.bev_meta_folder / f"{future_name}.json.gz"
        if not cm_path.exists() or not ms_path.exists():
            if cfg.costmap.future_missing_policy == "zero":
                fill = np.zeros_like(current_costmap, dtype=np.float32)
            elif cfg.costmap.future_missing_policy == "current":
                fill = current_costmap.astype(np.float32)
            else:
                fill = last.astype(np.float32)
            costmaps.append(fill); valid.append(False); reasons.append("future_file_missing")
            continue
        fm = load_json_gz(ms_path)
        fpose = get_pose_global_xy_yaw(fm)
        if fpose is None:
            costmaps.append(last); valid.append(False); reasons.append("future_pose_missing")
            continue
        fcost = load_costmap(cm_path)
        fmeta = load_json_gz(meta_path) if meta_path.exists() else {}
        fmpp = get_meters_per_pixel(fmeta, float(cfg.paths.default_pixels_per_meter))
        fcenter = get_ego_center(fmeta, fcost.shape)
        warped = warp_costmap_to_current(fcost, current_costmap.shape, current_center, current_mpp, current_pose, fpose, fcenter, fmpp, float(cfg.costmap.warp_border_cost))
        combined = combine_current_future(current_costmap, warped, cfg)
        costmaps.append(combined)
        last = combined
        valid.append(True); reasons.append("")
    return {"enabled": len(costmaps) > 1, "frames": frames, "costmaps": costmaps, "valid": valid, "missing_reasons": reasons}


def temporal_index_for_dense_step(step_idx: int, num_temporal_maps: int, cfg) -> int:
    if num_temporal_maps <= 1:
        return 0
    ratio = float(cfg.horizon.model_fps) / float(cfg.horizon.future_fps)
    x = float(step_idx + 1) / max(ratio, 1e-6)
    mode = str(cfg.costmap.temporal_index_mode)
    if mode == "ceil":
        idx = int(np.ceil(x))
    elif mode == "round":
        idx = int(np.round(x))
    else:
        idx = int(np.floor(x))
    return int(np.clip(idx, 0, num_temporal_maps - 1))



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


def _load_npz_mask(path: Path, key: str, expected_shape) -> np.ndarray:
    shape = tuple(int(v) for v in expected_shape[:2])
    if not path.exists():
        return np.zeros(shape, dtype=np.float32)
    with np.load(str(path)) as data:
        if key not in data.files:
            return np.zeros(shape, dtype=np.float32)
        arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[:2] != shape:
        raise ValueError(f"Traffic-rule mask shape {arr.shape} does not match expected shape {shape}: {path}")
    return arr.astype(np.float32)


def build_traffic_light_state_context(
    route_dir: Path,
    frame_name: str,
    current_shape,
    cfg,
) -> Dict:
    """Read the relevant traffic-light state at the current and previous saved frames.

    This context is used only for semantic supervision.  It does not alter
    occupancy maps or the hard red-light constraint.  In particular, the
    existing red-light planning logic remains unchanged.
    """
    route_dir = Path(route_dir)
    rule_cfg = _cfg_get(cfg, "traffic_rules", {})
    enabled = bool(_cfg_get(rule_cfg, "traffic_light_state_enabled", True))
    traffic_folder = str(_cfg_get(_cfg_get(cfg, "paths", {}), "bev_traffic_masks_folder", "bev_traffic_masks"))
    threshold = float(_cfg_get(rule_cfg, "traffic_light_state_threshold", 0.5))
    key_green = str(_cfg_get(rule_cfg, "traffic_light_green_mask_key", "tl_green"))
    key_yellow = str(_cfg_get(rule_cfg, "traffic_light_yellow_mask_key", "tl_yellow"))
    key_red = str(_cfg_get(rule_cfg, "red_light_mask_key", "tl_red"))
    shape = tuple(int(v) for v in current_shape[:2])

    def read_state(name: Optional[str]) -> Dict:
        if name is None:
            return {
                "frame": None,
                "valid": False,
                "state": "unknown",
                "counts": {"green": 0, "yellow": 0, "red": 0},
            }
        path = route_dir / traffic_folder / f"{name}.npz"
        if not path.exists():
            return {
                "frame": str(name),
                "valid": False,
                "state": "unknown",
                "counts": {"green": 0, "yellow": 0, "red": 0},
            }

        masks = {
            "green": _load_npz_mask(path, key_green, shape),
            "yellow": _load_npz_mask(path, key_yellow, shape),
            "red": _load_npz_mask(path, key_red, shape),
        }
        counts = {k: int(np.count_nonzero(np.asarray(v) >= threshold)) for k, v in masks.items()}
        # In a well-formed frame only one relevant signal state is active.  The
        # priority below makes the result deterministic if masks overlap.
        state = "none"
        for key in ["red", "yellow", "green"]:
            if counts[key] > 0:
                state = key
                break
        return {
            "frame": str(name),
            "valid": True,
            "state": state,
            "counts": counts,
        }

    current = read_state(frame_name)
    previous_name = offset_frame_name(frame_name, -1, int(cfg.horizon.future_frame_stride))
    previous = read_state(previous_name)
    transition = None
    if current.get("valid", False) and previous.get("valid", False):
        prev_state = str(previous.get("state", "unknown"))
        cur_state = str(current.get("state", "unknown"))
        if prev_state != cur_state:
            transition = f"{prev_state}_to_{cur_state}"

    return {
        "enabled": bool(enabled),
        "current_state": current.get("state", "unknown") if enabled else "unknown",
        "previous_state": previous.get("state", "unknown") if enabled else "unknown",
        "transition": transition if enabled else None,
        "current_frame": current.get("frame"),
        "previous_frame": previous.get("frame"),
        "current_valid": bool(current.get("valid", False) and enabled),
        "previous_valid": bool(previous.get("valid", False) and enabled),
        "current_counts": current.get("counts", {}),
        "previous_counts": previous.get("counts", {}),
    }


def build_temporal_red_light_constraints(
    route_dir: Path,
    frame_name: str,
    current_measurement: Dict,
    current_shape,
    current_meta: Dict,
    cfg,
) -> Dict:
    """Load and align time-varying red-light stop-line masks.

    The masks remain independent from physical occupancy.  Each temporal entry
    represents only the red stop line active at that corresponding future frame.
    """
    rule_cfg = _cfg_get(cfg, "traffic_rules", {})
    enabled = bool(_cfg_get(rule_cfg, "red_light_enabled", True))
    mask_key = str(_cfg_get(rule_cfg, "red_light_mask_key", "tl_red"))
    traffic_folder = str(_cfg_get(_cfg_get(cfg, "paths", {}), "bev_traffic_masks_folder", "bev_traffic_masks"))
    missing_policy = str(_cfg_get(rule_cfg, "future_missing_policy", "repeat_last"))

    shape = tuple(int(v) for v in current_shape[:2])
    current_path = route_dir / traffic_folder / f"{frame_name}.npz"
    current = _load_npz_mask(current_path, mask_key, shape)
    frames = [frame_name]
    maps = [current]
    valid = [bool(current_path.exists())]
    reasons = ["" if current_path.exists() else "current_traffic_mask_missing"]

    if not enabled:
        return {"enabled": False, "frames": frames, "maps": maps, "valid": valid, "missing_reasons": reasons}

    current_pose = get_pose_global_xy_yaw(current_measurement)
    if current_pose is None:
        return {"enabled": False, "frames": frames, "maps": maps, "valid": valid, "missing_reasons": ["current_pose_missing"]}

    current_mpp = get_meters_per_pixel(current_meta, float(cfg.paths.default_pixels_per_meter))
    current_center = get_ego_center(current_meta, shape)
    last = current.copy()

    for k in range(1, int(cfg.horizon.num_future_waypoints) + 1):
        future_name = offset_frame_name(frame_name, k, int(cfg.horizon.future_frame_stride))
        frames.append(str(future_name))
        if future_name is None:
            fill = last if missing_policy == "repeat_last" else (current if missing_policy == "current" else np.zeros(shape, dtype=np.float32))
            maps.append(np.asarray(fill, dtype=np.float32).copy())
            valid.append(False)
            reasons.append("non_numeric_frame")
            continue

        mask_path = route_dir / traffic_folder / f"{future_name}.npz"
        ms_path = route_dir / cfg.paths.measurements_folder / f"{future_name}.json.gz"
        meta_path = route_dir / cfg.paths.bev_meta_folder / f"{future_name}.json.gz"
        if not mask_path.exists() or not ms_path.exists():
            fill = last if missing_policy == "repeat_last" else (current if missing_policy == "current" else np.zeros(shape, dtype=np.float32))
            maps.append(np.asarray(fill, dtype=np.float32).copy())
            valid.append(False)
            reasons.append("future_traffic_file_missing")
            continue

        fm = load_json_gz(ms_path)
        fpose = get_pose_global_xy_yaw(fm)
        if fpose is None:
            maps.append(last.copy())
            valid.append(False)
            reasons.append("future_pose_missing")
            continue

        fmask = _load_npz_mask(mask_path, mask_key, shape)
        fmeta = load_json_gz(meta_path) if meta_path.exists() else {}
        fmpp = get_meters_per_pixel(fmeta, float(cfg.paths.default_pixels_per_meter))
        fcenter = get_ego_center(fmeta, fmask.shape)
        warped = warp_costmap_to_current(
            fmask,
            shape,
            current_center,
            current_mpp,
            current_pose,
            fpose,
            fcenter,
            fmpp,
            0.0,
        )
        maps.append(warped.astype(np.float32))
        last = warped.astype(np.float32)
        valid.append(True)
        reasons.append("")

    return {
        "enabled": len(maps) > 0,
        "frames": frames,
        "maps": maps,
        "valid": valid,
        "missing_reasons": reasons,
    }


def build_temporal_stop_sign_constraints(
    route_dir: Path,
    frame_name: str,
    current_measurement: Dict,
    current_shape,
    current_meta: Dict,
    cfg,
) -> Dict:
    """Load and align time-varying stop-sign control-region masks.

    The raw ``stop`` mask comes from ``bev_traffic_masks`` and remains
    independent from physical occupancy.  It is generated only while the
    current target stop sign has not yet been completed, so the temporal
    sequence can represent the approach/stop/release phases without relying on
    ``measurement['stop_sign_hazard']``.
    """
    rule_cfg = _cfg_get(cfg, "traffic_rules", {})
    enabled = bool(_cfg_get(rule_cfg, "stop_sign_enabled", True))
    mask_key = str(_cfg_get(rule_cfg, "stop_sign_mask_key", "stop"))
    traffic_folder = str(_cfg_get(_cfg_get(cfg, "paths", {}), "bev_traffic_masks_folder", "bev_traffic_masks"))
    missing_policy = str(_cfg_get(rule_cfg, "future_missing_policy", "repeat_last"))

    shape = tuple(int(v) for v in current_shape[:2])
    current_path = route_dir / traffic_folder / f"{frame_name}.npz"
    current = _load_npz_mask(current_path, mask_key, shape)
    frames = [frame_name]
    maps = [current]
    valid = [bool(current_path.exists())]
    reasons = ["" if current_path.exists() else "current_traffic_mask_missing"]

    if not enabled:
        return {"enabled": False, "frames": frames, "maps": maps, "valid": valid, "missing_reasons": reasons}

    current_pose = get_pose_global_xy_yaw(current_measurement)
    if current_pose is None:
        return {"enabled": False, "frames": frames, "maps": maps, "valid": valid, "missing_reasons": ["current_pose_missing"]}

    current_mpp = get_meters_per_pixel(current_meta, float(cfg.paths.default_pixels_per_meter))
    current_center = get_ego_center(current_meta, shape)
    last = current.copy()

    for k in range(1, int(cfg.horizon.num_future_waypoints) + 1):
        future_name = offset_frame_name(frame_name, k, int(cfg.horizon.future_frame_stride))
        frames.append(str(future_name))
        if future_name is None:
            fill = last if missing_policy == "repeat_last" else (current if missing_policy == "current" else np.zeros(shape, dtype=np.float32))
            maps.append(np.asarray(fill, dtype=np.float32).copy())
            valid.append(False)
            reasons.append("non_numeric_frame")
            continue

        mask_path = route_dir / traffic_folder / f"{future_name}.npz"
        ms_path = route_dir / cfg.paths.measurements_folder / f"{future_name}.json.gz"
        meta_path = route_dir / cfg.paths.bev_meta_folder / f"{future_name}.json.gz"
        if not mask_path.exists() or not ms_path.exists():
            fill = last if missing_policy == "repeat_last" else (current if missing_policy == "current" else np.zeros(shape, dtype=np.float32))
            maps.append(np.asarray(fill, dtype=np.float32).copy())
            valid.append(False)
            reasons.append("future_traffic_file_missing")
            continue

        fm = load_json_gz(ms_path)
        fpose = get_pose_global_xy_yaw(fm)
        if fpose is None:
            maps.append(last.copy())
            valid.append(False)
            reasons.append("future_pose_missing")
            continue

        fmask = _load_npz_mask(mask_path, mask_key, shape)
        fmeta = load_json_gz(meta_path) if meta_path.exists() else {}
        fmpp = get_meters_per_pixel(fmeta, float(cfg.paths.default_pixels_per_meter))
        fcenter = get_ego_center(fmeta, fmask.shape)
        warped = warp_costmap_to_current(
            fmask,
            shape,
            current_center,
            current_mpp,
            current_pose,
            fpose,
            fcenter,
            fmpp,
            0.0,
        )
        maps.append(warped.astype(np.float32))
        last = warped.astype(np.float32)
        valid.append(True)
        reasons.append("")

    return {
        "enabled": len(maps) > 0,
        "frames": frames,
        "maps": maps,
        "valid": valid,
        "missing_reasons": reasons,
    }
