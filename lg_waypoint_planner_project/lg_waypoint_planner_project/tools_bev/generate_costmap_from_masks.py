#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_costmap_from_masks.py

Simplified BEV collision-map generator.

This version intentionally removes the previous hand-crafted continuous cost
calculation. It converts saved BEV masks into a simple binary-style map for
collision/occupancy checking:

    free area      -> free_value
    blocked area   -> blocked_value

Blocked area is defined by dynamic actors and non-road regions. Lane markings
are intentionally NOT written into the ordinary cost map because the saved BEV
lane mask is low resolution and should not behave like a wide obstacle.

Solid lane markings are saved separately as a thin centerline-style constraint.
The waypoint evaluator uses that map only to detect whether the planned ego
center trajectory crosses a solid boundary. Broken lane markings remain
traversable.

Usage:
    1. Edit configs/simple_bev_collision_map.yaml
    2. Run:
       python tools_bev/generate_costmap_from_masks.py

No argparse is used. The optional first positional argument may be a config path,
but the normal workflow is to edit the YAML file directly.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml


# -----------------------------------------------------------------------------
# Default config path. Edit YAML instead of passing long command-line arguments.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "simple_bev_collision_map.yaml"


# OpenCV BGR colors for debug visualization.
COLOR_FREE_ROAD = (55, 55, 55)
COLOR_NON_ROAD = (18, 18, 18)
COLOR_SIDEWALK = (105, 105, 105)
COLOR_ACTOR = (40, 40, 220)
COLOR_VEHICLE = (255, 80, 80)
COLOR_WALKER = (255, 220, 80)
COLOR_BLOCKED = (0, 0, 255)
COLOR_LANE_SOLID = (255, 0, 255)
COLOR_LANE_BROKEN = (255, 140, 255)


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def cfg_get(cfg: Dict, dotted_key: str, default=None):
    cur = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def load_json_gz(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[Warn] Failed to read meta: {path}, error={exc}")
        return {}


def as_bool(mask: Optional[np.ndarray], shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if mask is None:
        if shape is None:
            raise ValueError("shape is required when mask is None")
        return np.zeros(shape, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    return mask.astype(np.uint8) > 0


def infer_shape(*mask_dicts: Dict[str, np.ndarray]) -> Tuple[int, int]:
    for mask_dict in mask_dicts:
        for value in mask_dict.values():
            if value.ndim != 2:
                raise ValueError(f"Expected 2D mask, got shape={value.shape}")
            return tuple(value.shape)
    raise ValueError("Cannot infer BEV shape because no mask was loaded")


def get_meters_per_pixel(meta: Dict, default_pixels_per_meter: float) -> float:
    if "meters_per_pixel" in meta:
        return float(meta["meters_per_pixel"])
    if "pixels_per_meter" in meta:
        ppm = float(meta["pixels_per_meter"])
        if ppm <= 0:
            raise ValueError(f"Invalid pixels_per_meter in meta: {ppm}")
        return 1.0 / ppm
    if default_pixels_per_meter <= 0:
        raise ValueError(f"Invalid default_pixels_per_meter: {default_pixels_per_meter}")
    return 1.0 / float(default_pixels_per_meter)


def disk_kernel(radius_px: int) -> np.ndarray:
    radius_px = int(max(radius_px, 0))
    if radius_px == 0:
        return np.ones((1, 1), dtype=np.uint8)
    ksize = radius_px * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    dilated = cv2.dilate(mask.astype(np.uint8), disk_kernel(radius_px), iterations=1)
    return dilated.astype(bool)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce a thick binary raster line to an approximately one-pixel skeleton.

    Only basic OpenCV morphology is used, so this does not require opencv-contrib
    or scikit-image. The operation preserves the rough topology of the original
    lane mask while removing the artificial width caused by low BEV resolution.
    """
    img = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8) * 255
    if not np.any(img):
        return np.zeros_like(img, dtype=bool)

    skel = np.zeros_like(img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = img.copy()

    # The image shrinks every iteration, so this loop is bounded by the maximum
    # thickness of the input line mask rather than by image size in practice.
    while np.any(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, residue)
        work = cv2.erode(work, element)

    return skel > 0


def get_route_paths(route_dir: Path, frame_name: str, augmented: bool) -> Tuple[Path, Path, Path, Path]:
    suffix = "_augmented" if augmented else ""
    static_path = route_dir / f"bev_static_masks{suffix}" / f"{frame_name}.npz"
    dynamic_path = route_dir / f"bev_dynamic_masks{suffix}" / f"{frame_name}.npz"
    traffic_path = route_dir / f"bev_traffic_masks{suffix}" / f"{frame_name}.npz"
    meta_path = route_dir / f"bev_meta{suffix}" / f"{frame_name}.json.gz"
    return static_path, dynamic_path, traffic_path, meta_path


def build_simple_collision_map(
    static_masks: Dict[str, np.ndarray],
    dynamic_masks: Dict[str, np.ndarray],
    traffic_masks: Dict[str, np.ndarray],
    meters_per_pixel: float,
    cfg: Dict,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build a binary-style BEV map for collision/occupancy checking."""
    shape = infer_shape(static_masks, dynamic_masks, traffic_masks)

    free_value = float(cfg_get(cfg, "output.free_value", 0.0))
    blocked_value = float(cfg_get(cfg, "output.blocked_value", 100.0))
    block_sidewalk = bool(cfg_get(cfg, "bev.block_sidewalk", True))
    assume_free_when_road_missing = bool(cfg_get(cfg, "bev.assume_free_when_road_missing", True))
    actor_inflate_m = float(cfg_get(cfg, "bev.actor_inflate_m", 0.0))
    thin_solid_line = bool(cfg_get(cfg, "bev.thin_solid_lane_to_centerline", True))

    if "road" in static_masks:
        road = as_bool(static_masks.get("road"), shape)
    else:
        road = np.ones(shape, dtype=bool) if assume_free_when_road_missing else np.zeros(shape, dtype=bool)

    sidewalk = as_bool(static_masks.get("sidewalk"), shape)
    lane_solid = as_bool(static_masks.get("lane_solid"), shape)
    lane_broken = as_bool(static_masks.get("lane_broken"), shape)

    vehicle = as_bool(dynamic_masks.get("vehicle"), shape)
    walker = as_bool(dynamic_masks.get("walker"), shape)
    if "actor" in dynamic_masks:
        actor = as_bool(dynamic_masks.get("actor"), shape)
    else:
        actor = vehicle | walker

    inflate_px = int(round(actor_inflate_m / meters_per_pixel)) if actor_inflate_m > 0 else 0
    actor_occupied = dilate_mask(actor, inflate_px)

    # Keep lane rules separate from ordinary occupancy. The raw BEV lane mask is
    # low-resolution and often several pixels wide; treating it as an obstacle
    # removes too much drivable space. Instead, reduce it to a thin detector line
    # and save it only in the dedicated lane-constraint map.
    lane_solid_constraint = skeletonize_mask(lane_solid) if thin_solid_line else lane_solid.copy()

    static_blocked = ~road
    if block_sidewalk:
        static_blocked = static_blocked | sidewalk

    blocked = static_blocked | actor_occupied

    simple_map = np.full(shape, free_value, dtype=np.float32)
    simple_map[blocked] = blocked_value

    lane_constraint_map = np.full(shape, free_value, dtype=np.float32)
    lane_constraint_map[lane_solid_constraint] = blocked_value

    debug_masks = {
        "road": road,
        "sidewalk": sidewalk,
        "lane_solid": lane_solid,
        "lane_broken": lane_broken,
        "lane_solid_constraint": lane_solid_constraint,
        "vehicle": vehicle,
        "walker": walker,
        "actor": actor,
        "actor_occupied": actor_occupied,
        "static_blocked": static_blocked,
        "blocked": blocked,
        "lane_constraint_map": lane_constraint_map,
    }
    return simple_map, debug_masks


def make_overlay(debug_masks: Dict[str, np.ndarray]) -> np.ndarray:
    shape = debug_masks["blocked"].shape
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    road = debug_masks["road"]
    sidewalk = debug_masks["sidewalk"]
    blocked = debug_masks["blocked"]
    actor = debug_masks["actor"]
    actor_occupied = debug_masks["actor_occupied"]
    vehicle = debug_masks["vehicle"]
    walker = debug_masks["walker"]
    lane_solid = debug_masks["lane_solid"]
    lane_broken = debug_masks["lane_broken"]

    image[:] = COLOR_NON_ROAD
    image[road] = COLOR_FREE_ROAD
    image[sidewalk] = COLOR_SIDEWALK
    image[blocked] = COLOR_BLOCKED
    image[lane_broken] = COLOR_LANE_BROKEN
    image[lane_solid] = COLOR_LANE_SOLID
    image[actor_occupied] = COLOR_ACTOR
    image[actor] = COLOR_ACTOR
    image[vehicle] = COLOR_VEHICLE
    image[walker] = COLOR_WALKER
    return image


def save_debug_images(simple_map: np.ndarray, debug_masks: Dict[str, np.ndarray], debug_dir: Path, frame_name: str):
    """Save occupancy plus raw/thinned solid-line masks for inspection."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    blocked = debug_masks["blocked"].astype(np.uint8) * 255
    solid_raw = debug_masks["lane_solid"].astype(np.uint8) * 255
    solid_detector = debug_masks["lane_solid_constraint"].astype(np.uint8) * 255
    cv2.imwrite(str(debug_dir / f"{frame_name}_blocked.png"), blocked)
    cv2.imwrite(str(debug_dir / f"{frame_name}_lane_solid_raw.png"), solid_raw)
    cv2.imwrite(str(debug_dir / f"{frame_name}_lane_solid_detector.png"), solid_detector)


def find_frame_names(route_dir: Path, cfg: Dict) -> List[str]:
    augmented = bool(cfg_get(cfg, "run.augmented", False))
    frame = cfg_get(cfg, "run.frame", None)
    if frame not in [None, "", "null"]:
        return [str(frame).replace(".npz", "").replace(".npy", "")]

    suffix = "_augmented" if augmented else ""
    candidate_dirs = [
        route_dir / f"bev_static_masks{suffix}",
        route_dir / f"bev_dynamic_masks{suffix}",
        route_dir / f"bev_traffic_masks{suffix}",
    ]
    for folder in candidate_dirs:
        if folder.exists():
            names = sorted([p.stem for p in folder.glob("*.npz")])
            if names:
                return names
    return []


def process_one_frame(route_dir: Path, frame_name: str, cfg: Dict) -> bool:
    augmented = bool(cfg_get(cfg, "run.augmented", False))
    static_path, dynamic_path, traffic_path, meta_path = get_route_paths(route_dir, frame_name, augmented)

    static_masks = load_npz(static_path)
    dynamic_masks = load_npz(dynamic_path)
    traffic_masks = load_npz(traffic_path)
    meta = load_json_gz(meta_path)

    if not static_masks and not dynamic_masks and not traffic_masks:
        print(f"[Skip] No BEV masks found: route={route_dir}, frame={frame_name}")
        return False

    meters_per_pixel = get_meters_per_pixel(
        meta=meta,
        default_pixels_per_meter=float(cfg_get(cfg, "bev.default_pixels_per_meter", 2.0)),
    )

    simple_map, debug_masks = build_simple_collision_map(
        static_masks=static_masks,
        dynamic_masks=dynamic_masks,
        traffic_masks=traffic_masks,
        meters_per_pixel=meters_per_pixel,
        cfg=cfg,
    )

    costmap_folder = str(cfg_get(cfg, "output.costmap_folder", "costmap"))
    lane_constraint_folder = str(cfg_get(cfg, "output.lane_constraint_folder", "lane_constraints"))
    debug_folder = str(cfg_get(cfg, "output.debug_folder", "costmap_debug"))
    costmap_dir = route_dir / costmap_folder
    lane_constraint_dir = route_dir / lane_constraint_folder
    debug_dir = route_dir / debug_folder
    costmap_dir.mkdir(parents=True, exist_ok=True)
    lane_constraint_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg_get(cfg, "output.save_npy", True)):
        np.save(str(costmap_dir / f"{frame_name}.npy"), simple_map.astype(np.float32))
        np.save(
            str(lane_constraint_dir / f"{frame_name}.npy"),
            debug_masks["lane_constraint_map"].astype(np.float32),
        )

    if bool(cfg_get(cfg, "output.save_npz", False)):
        np.savez_compressed(
            str(costmap_dir / f"{frame_name}.npz"),
            collision_map=simple_map.astype(np.float32),
            blocked=debug_masks["blocked"].astype(np.uint8),
            road=debug_masks["road"].astype(np.uint8),
            actor=debug_masks["actor"].astype(np.uint8),
            actor_occupied=debug_masks["actor_occupied"].astype(np.uint8),
            lane_solid=debug_masks["lane_solid"].astype(np.uint8),
            lane_broken=debug_masks["lane_broken"].astype(np.uint8),
            lane_solid_constraint=debug_masks["lane_solid_constraint"].astype(np.uint8),
            meters_per_pixel=np.array(meters_per_pixel, dtype=np.float32),
        )

    if bool(cfg_get(cfg, "output.save_debug", True)):
        save_debug_images(simple_map, debug_masks, debug_dir, frame_name)

    if bool(cfg_get(cfg, "run.verbose", True)):
        blocked_ratio = float(debug_masks["blocked"].mean()) * 100.0
        actor_ratio = float(debug_masks["actor_occupied"].mean()) * 100.0
        solid_ratio = float(debug_masks["lane_solid_constraint"].mean()) * 100.0
        print(
            f"[OK] {route_dir.name}/{frame_name}: "
            f"blocked={blocked_ratio:.2f}%, actor={actor_ratio:.2f}%, solid_lane={solid_ratio:.2f}%, "
            f"values=({float(simple_map.min()):.1f}, {float(simple_map.max()):.1f}), "
            f"m/px={meters_per_pixel:.3f}"
        )

    return True


def process_route_dir(route_dir: Path, cfg: Dict) -> Tuple[int, int]:
    frame_names = find_frame_names(route_dir, cfg)
    if not frame_names:
        if bool(cfg_get(cfg, "run.verbose", True)):
            print(f"[Skip] No frames found in route: {route_dir}")
        return 0, 0

    ok = 0
    for frame_name in frame_names:
        if process_one_frame(route_dir, frame_name, cfg):
            ok += 1
    print(f"[Route Done] {route_dir}: {ok}/{len(frame_names)} frames processed.")
    return ok, len(frame_names)


def find_route_dirs(root: Path, cfg: Dict) -> List[Path]:
    augmented = bool(cfg_get(cfg, "run.augmented", False))
    suffix = "_augmented" if augmented else ""
    route_dirs = set()
    for folder_name in [
        f"bev_static_masks{suffix}",
        f"bev_dynamic_masks{suffix}",
        f"bev_traffic_masks{suffix}",
    ]:
        for folder in root.rglob(folder_name):
            if folder.is_dir():
                route_dirs.add(folder.parent)
    return sorted(route_dirs)


def main():
    config_path = DEFAULT_CONFIG_PATH
    if len(sys.argv) >= 2:
        config_path = Path(sys.argv[1]).expanduser().resolve()

    cfg = load_yaml(config_path)
    input_path = Path(str(cfg_get(cfg, "run.input", ""))).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"[Config] {config_path}")
    print(f"[Input] {input_path}")

    if bool(cfg_get(cfg, "run.recursive", False)):
        route_dirs = find_route_dirs(input_path, cfg)
        print(f"[Info] Found {len(route_dirs)} route directories")
        total_ok = 0
        total_frames = 0
        for route_dir in route_dirs:
            ok, total = process_route_dir(route_dir, cfg)
            total_ok += ok
            total_frames += total
        print(f"[All Done] {total_ok}/{total_frames} frames processed.")
    else:
        process_route_dir(input_path, cfg)


if __name__ == "__main__":
    main()
