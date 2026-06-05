#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_costmap_from_masks.py

Generate BEV cost maps from collected BEV masks.

Input route directory example:
    data/town01_original/routes_xxx/
        bev_static_masks/
            0000.npz
        bev_dynamic_masks/
            0000.npz
        bev_traffic_masks/
            0000.npz
        bev_meta/
            0000.json.gz

Output:
    data/town01_original/routes_xxx/
        costmap/
            0000.npy
        costmap_debug/
            0000_cost.png
            0000_overlay.png
            0000_blend.png

This first version uses current-frame masks only:
    static masks  : road, sidewalk, lane_all, lane_broken, lane_solid
    dynamic masks : vehicle, walker, actor
    traffic masks : stop, tl_green, tl_yellow, tl_red

Coordinate note:
    This script does not rotate or transform the masks. It keeps exactly the
    saved mask coordinate frame. It only converts masks to cost maps.


使用方法:

1. 处理单个 route:
python tools_bev/generate_costmap_from_masks.py \
  --input data/town01_original/routes_xxx \
  --save_debug \
  --verbose

2. 递归处理整个 data 目录：
python tools_bev/generate_costmap_from_masks.py \
  --input data \
  --recursive \
  --save_debug \
  --verbose

3. 只处理某一帧：
python tools_bev/generate_costmap_from_masks.py \
  --input data/town01_original/routes_xxx \
  --frame 0000 \
  --save_debug \
  --verbose
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# OpenCV BGR colors for debug visualization.
COLOR_ROAD = (46, 52, 54)
COLOR_SIDEWALK = (128, 128, 128)
COLOR_LANE_ALL = (255, 0, 255)
COLOR_LANE_BROKEN = (255, 140, 255)
COLOR_LANE_SOLID = (180, 0, 180)
COLOR_VEHICLE = (255, 0, 0)
COLOR_WALKER = (255, 255, 0)
COLOR_ACTOR_INFLATED = (120, 80, 255)
COLOR_STOP = (0, 180, 255)
COLOR_TL_GREEN = (0, 255, 0)
COLOR_TL_YELLOW = (0, 255, 255)
COLOR_TL_RED = (0, 0, 255)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load an npz file into a normal dict."""
    if not path.exists():
        return {}

    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def load_meta(meta_path: Path) -> Dict:
    """Load optional BEV meta json.gz."""
    if not meta_path.exists():
        return {}

    try:
        with gzip.open(meta_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warn] Failed to read meta file: {meta_path}, error: {e}")
        return {}


def as_bool(mask: Optional[np.ndarray], shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert mask to bool. If mask is None, return all-false mask."""
    if mask is None:
        if shape is None:
            raise ValueError("shape must be provided when mask is None.")
        return np.zeros(shape, dtype=bool)
    return mask.astype(np.uint8) > 0


def infer_shape(*mask_dicts: Dict[str, np.ndarray]) -> Tuple[int, int]:
    for d in mask_dicts:
        for v in d.values():
            if v.ndim != 2:
                raise ValueError(f"Expected 2D mask, got shape {v.shape}")
            return v.shape
    raise ValueError("Cannot infer mask shape. No mask file was loaded.")


def get_meters_per_pixel(meta: Dict, default_pixels_per_meter: float) -> float:
    """
    Infer meters_per_pixel from meta if available.
    Existing meta may only contain width/ego_center/rotated, so
    default_pixels_per_meter is used as fallback.
    """
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
    """Create an elliptical structuring element."""
    radius_px = int(max(radius_px, 0))
    if radius_px == 0:
        return np.ones((1, 1), dtype=np.uint8)

    ksize = 2 * radius_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Dilate a boolean mask by radius_px."""
    if radius_px <= 0:
        return mask.astype(bool)

    kernel = disk_kernel(radius_px)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def add_actor_proximity_cost(
    cost: np.ndarray,
    actor_mask: np.ndarray,
    meters_per_pixel: float,
    proximity_weight: float,
    proximity_sigma_m: float,
    max_proximity_cost: float,
) -> np.ndarray:
    """
    Add continuous cost around actors using distance transform.

    actor_mask: True where occupied by actor.
    For non-actor pixels, distance to nearest actor is computed.
    cost += proximity_weight * exp(-dist_m / sigma_m)
    """
    if not actor_mask.any():
        return cost

    if proximity_weight <= 0 or proximity_sigma_m <= 0:
        return cost

    # distanceTransform computes distance to nearest zero pixel.
    # We need distance to actor pixels, so actor pixels are zeros.
    free_or_non_actor = (~actor_mask).astype(np.uint8)
    dist_px = cv2.distanceTransform(free_or_non_actor, cv2.DIST_L2, 5)
    dist_m = dist_px * meters_per_pixel

    prox = proximity_weight * np.exp(-dist_m / proximity_sigma_m)
    prox = np.clip(prox, 0.0, max_proximity_cost)

    cost += prox.astype(np.float32)
    return cost


def normalize_cost_to_uint8(cost: np.ndarray, clip_max: float) -> np.ndarray:
    """Normalize cost map to uint8 for visualization."""
    if clip_max <= 0:
        clip_max = float(np.max(cost)) if np.max(cost) > 0 else 1.0

    vis = np.clip(cost, 0.0, clip_max) / clip_max
    vis = (vis * 255.0).astype(np.uint8)
    return vis


def colorize_base_masks(
    static_masks: Dict[str, np.ndarray],
    dynamic_masks: Dict[str, np.ndarray],
    traffic_masks: Dict[str, np.ndarray],
    actor_inflated: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Create semantic overlay for debug."""
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    road = as_bool(static_masks.get("road"), shape)
    sidewalk = as_bool(static_masks.get("sidewalk"), shape)
    lane_all = as_bool(static_masks.get("lane_all"), shape)
    lane_broken = as_bool(static_masks.get("lane_broken"), shape)

    if "lane_solid" in static_masks:
        lane_solid = as_bool(static_masks.get("lane_solid"), shape)
    else:
        lane_solid = lane_all & (~lane_broken)

    vehicle = as_bool(dynamic_masks.get("vehicle"), shape)
    walker = as_bool(dynamic_masks.get("walker"), shape)

    if "actor" in dynamic_masks:
        actor = as_bool(dynamic_masks.get("actor"), shape)
    else:
        actor = vehicle | walker

    stop = as_bool(traffic_masks.get("stop"), shape)
    tl_green = as_bool(traffic_masks.get("tl_green"), shape)
    tl_yellow = as_bool(traffic_masks.get("tl_yellow"), shape)
    tl_red = as_bool(traffic_masks.get("tl_red"), shape)

    image[road] = COLOR_ROAD
    image[sidewalk] = COLOR_SIDEWALK
    image[lane_all] = COLOR_LANE_ALL
    image[lane_solid] = COLOR_LANE_SOLID
    image[lane_broken] = COLOR_LANE_BROKEN

    # Inflated actor area first, real actors later overwrite it.
    image[actor_inflated] = COLOR_ACTOR_INFLATED
    image[actor] = COLOR_VEHICLE
    image[vehicle] = COLOR_VEHICLE
    image[walker] = COLOR_WALKER

    image[stop] = COLOR_STOP
    image[tl_green] = COLOR_TL_GREEN
    image[tl_yellow] = COLOR_TL_YELLOW
    image[tl_red] = COLOR_TL_RED

    return image


def generate_costmap(
    static_masks: Dict[str, np.ndarray],
    dynamic_masks: Dict[str, np.ndarray],
    traffic_masks: Dict[str, np.ndarray],
    meters_per_pixel: float,
    args,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate one cost map from loaded masks.

    Return:
        cost: float32 HxW
        debug_masks: dict including actor_inflated etc.
    """
    shape = infer_shape(static_masks, dynamic_masks, traffic_masks)

    cost = np.zeros(shape, dtype=np.float32)

    # ---------- Static masks ----------
    road = as_bool(static_masks.get("road"), shape)
    sidewalk = as_bool(static_masks.get("sidewalk"), shape)
    lane_all = as_bool(static_masks.get("lane_all"), shape)
    lane_broken = as_bool(static_masks.get("lane_broken"), shape)

    if "lane_solid" in static_masks:
        lane_solid = as_bool(static_masks.get("lane_solid"), shape)
    else:
        lane_solid = lane_all & (~lane_broken)

    # Non-road and sidewalk should be strongly discouraged.
    cost[~road] += args.background_cost
    cost[sidewalk] += args.sidewalk_cost

    # Lane markings are not hard obstacles, but solid lines should be more costly.
    cost[lane_all] += args.lane_all_cost
    cost[lane_broken] += args.lane_broken_cost
    cost[lane_solid] += args.lane_solid_cost

    # ---------- Dynamic masks ----------
    vehicle = as_bool(dynamic_masks.get("vehicle"), shape)
    walker = as_bool(dynamic_masks.get("walker"), shape)

    if "actor" in dynamic_masks:
        actor = as_bool(dynamic_masks.get("actor"), shape)
    else:
        actor = vehicle | walker

    dilation_radius_px = int(round(args.actor_dilation_m / meters_per_pixel))
    actor_inflated = dilate_mask(actor, dilation_radius_px)

    cost[actor] += args.actor_cost
    cost[actor_inflated] += args.actor_inflated_cost

    # Optional extra weight for walkers, because they are vulnerable traffic participants.
    if args.walker_extra_cost > 0:
        walker_inflated = dilate_mask(walker, dilation_radius_px)
        cost[walker_inflated] += args.walker_extra_cost

    cost = add_actor_proximity_cost(
        cost=cost,
        actor_mask=actor,
        meters_per_pixel=meters_per_pixel,
        proximity_weight=args.actor_proximity_weight,
        proximity_sigma_m=args.actor_proximity_sigma_m,
        max_proximity_cost=args.actor_proximity_max_cost,
    )

    # ---------- Traffic masks ----------
    stop = as_bool(traffic_masks.get("stop"), shape)
    tl_green = as_bool(traffic_masks.get("tl_green"), shape)
    tl_yellow = as_bool(traffic_masks.get("tl_yellow"), shape)
    tl_red = as_bool(traffic_masks.get("tl_red"), shape)

    cost[stop] += args.stop_cost
    cost[tl_green] += args.tl_green_cost
    cost[tl_yellow] += args.tl_yellow_cost
    cost[tl_red] += args.tl_red_cost

    # Optional smoothing for more planner-friendly cost map.
    if args.gaussian_blur_ksize > 0:
        k = int(args.gaussian_blur_ksize)
        if k % 2 == 0:
            k += 1
        cost = cv2.GaussianBlur(cost, (k, k), args.gaussian_blur_sigma).astype(np.float32)

    debug_masks = {
        "road": road,
        "sidewalk": sidewalk,
        "lane_all": lane_all,
        "lane_broken": lane_broken,
        "lane_solid": lane_solid,
        "vehicle": vehicle,
        "walker": walker,
        "actor": actor,
        "actor_inflated": actor_inflated,
        "stop": stop,
        "tl_green": tl_green,
        "tl_yellow": tl_yellow,
        "tl_red": tl_red,
    }

    return cost.astype(np.float32), debug_masks


def save_debug_images(
    cost: np.ndarray,
    static_masks: Dict[str, np.ndarray],
    dynamic_masks: Dict[str, np.ndarray],
    traffic_masks: Dict[str, np.ndarray],
    actor_inflated: np.ndarray,
    debug_dir: Path,
    frame_name: str,
    clip_max: float,
):
    """Save cost visualization and overlay visualization."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    cost_u8 = normalize_cost_to_uint8(cost, clip_max=clip_max)
    cost_color = cv2.applyColorMap(cost_u8, cv2.COLORMAP_JET)

    shape = cost.shape
    overlay = colorize_base_masks(
        static_masks=static_masks,
        dynamic_masks=dynamic_masks,
        traffic_masks=traffic_masks,
        actor_inflated=actor_inflated,
        shape=shape,
    )

    # Blend semantic overlay and cost heatmap for easier inspection.
    blended = cv2.addWeighted(overlay, 0.55, cost_color, 0.45, 0.0)

    cv2.imwrite(str(debug_dir / f"{frame_name}_cost.png"), cost_color)
    cv2.imwrite(str(debug_dir / f"{frame_name}_overlay.png"), overlay)
    cv2.imwrite(str(debug_dir / f"{frame_name}_blend.png"), blended)


def process_one_frame(route_dir: Path, frame_name: str, args) -> bool:
    """Process one frame in one route directory."""
    suffix = "_augmented" if args.augmented else ""

    static_path = route_dir / f"bev_static_masks{suffix}" / f"{frame_name}.npz"
    dynamic_path = route_dir / f"bev_dynamic_masks{suffix}" / f"{frame_name}.npz"
    traffic_path = route_dir / f"bev_traffic_masks{suffix}" / f"{frame_name}.npz"
    meta_path = route_dir / f"bev_meta{suffix}" / f"{frame_name}.json.gz"

    static_masks = load_npz(static_path)
    dynamic_masks = load_npz(dynamic_path)
    traffic_masks = load_npz(traffic_path)
    meta = load_meta(meta_path)

    if not static_masks and not dynamic_masks and not traffic_masks:
        print(f"[Skip] No masks found: route={route_dir}, frame={frame_name}")
        return False

    meters_per_pixel = get_meters_per_pixel(
        meta=meta,
        default_pixels_per_meter=args.default_pixels_per_meter,
    )

    cost, debug_masks = generate_costmap(
        static_masks=static_masks,
        dynamic_masks=dynamic_masks,
        traffic_masks=traffic_masks,
        meters_per_pixel=meters_per_pixel,
        args=args,
    )

    cost_dir_name = "costmap_augmented" if args.augmented else "costmap"
    debug_dir_name = "costmap_debug_augmented" if args.augmented else "costmap_debug"

    cost_dir = route_dir / cost_dir_name
    debug_dir = route_dir / debug_dir_name

    cost_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(cost_dir / f"{frame_name}.npy"), cost)

    if args.save_compressed_npz:
        np.savez_compressed(
            str(cost_dir / f"{frame_name}.npz"),
            cost=cost,
            meters_per_pixel=np.array(meters_per_pixel, dtype=np.float32),
            default_pixels_per_meter=np.array(args.default_pixels_per_meter, dtype=np.float32),
        )

    if args.save_debug:
        save_debug_images(
            cost=cost,
            static_masks=static_masks,
            dynamic_masks=dynamic_masks,
            traffic_masks=traffic_masks,
            actor_inflated=debug_masks["actor_inflated"],
            debug_dir=debug_dir,
            frame_name=frame_name,
            clip_max=args.debug_clip_max,
        )

    if args.verbose:
        print(
            f"[OK] {route_dir.name}/{frame_name}: "
            f"cost min={float(cost.min()):.3f}, "
            f"max={float(cost.max()):.3f}, "
            f"mean={float(cost.mean()):.3f}, "
            f"m/px={meters_per_pixel:.3f}"
        )

    return True


def get_frame_names(route_dir: Path, args) -> list:
    """Find frame names from static mask directory."""
    suffix = "_augmented" if args.augmented else ""
    static_dir = route_dir / f"bev_static_masks{suffix}"
    dynamic_dir = route_dir / f"bev_dynamic_masks{suffix}"
    traffic_dir = route_dir / f"bev_traffic_masks{suffix}"

    if args.frame is not None:
        return [args.frame.replace(".npz", "").replace(".npy", "")]

    # Prefer static masks. If absent, fall back to dynamic or traffic masks.
    for d in [static_dir, dynamic_dir, traffic_dir]:
        if d.exists():
            names = sorted([p.stem for p in d.glob("*.npz")])
            if names:
                return names

    return []


def process_route_dir(route_dir: Path, args) -> Tuple[int, int]:
    """Process all selected frames in one route directory."""
    frame_names = get_frame_names(route_dir, args)

    if len(frame_names) == 0:
        if args.verbose:
            print(f"[Skip] No frames found in route: {route_dir}")
        return 0, 0

    total = 0
    ok = 0
    for frame_name in frame_names:
        total += 1
        if process_one_frame(route_dir, frame_name, args):
            ok += 1

    print(f"[Route Done] {route_dir}: {ok}/{total} frames processed.")
    return ok, total


def find_route_dirs(root: Path, args) -> list:
    """Recursively find route dirs containing mask folders."""
    suffix = "_augmented" if args.augmented else ""
    candidates = []

    for folder_name in [
        f"bev_static_masks{suffix}",
        f"bev_dynamic_masks{suffix}",
        f"bev_traffic_masks{suffix}",
    ]:
        for p in root.rglob(folder_name):
            if p.is_dir():
                candidates.append(p.parent)

    return sorted(set(candidates))


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Generate BEV cost maps from saved BEV mask npz files."
    )

    # Input mode.
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Route directory or data root.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find route directories under --input.",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default=None,
        help="Only process one frame, e.g. 0000 or 0000.npz.",
    )
    parser.add_argument(
        "--augmented",
        action="store_true",
        help="Use *_augmented mask folders.",
    )

    # Resolution.
    parser.add_argument(
        "--default_pixels_per_meter",
        type=float,
        default=2.0,
        help="Fallback pixels per meter if bev_meta does not contain resolution. Current data usually uses 2.0.",
    )

    # Static costs.
    parser.add_argument("--background_cost", type=float, default=100.0)
    parser.add_argument("--sidewalk_cost", type=float, default=100.0)
    parser.add_argument("--lane_all_cost", type=float, default=1.0)
    parser.add_argument("--lane_broken_cost", type=float, default=2.0)
    parser.add_argument("--lane_solid_cost", type=float, default=20.0)

    # Dynamic costs.
    parser.add_argument("--actor_cost", type=float, default=120.0)
    parser.add_argument("--actor_inflated_cost", type=float, default=80.0)
    parser.add_argument(
        "--actor_dilation_m",
        type=float,
        default=1.5,
        help="Safety inflation radius around actors, in meters.",
    )
    parser.add_argument(
        "--walker_extra_cost",
        type=float,
        default=30.0,
        help="Extra cost around walkers after inflation.",
    )
    parser.add_argument("--actor_proximity_weight", type=float, default=30.0)
    parser.add_argument("--actor_proximity_sigma_m", type=float, default=3.0)
    parser.add_argument("--actor_proximity_max_cost", type=float, default=30.0)

    # Traffic costs.
    parser.add_argument("--stop_cost", type=float, default=50.0)
    parser.add_argument("--tl_green_cost", type=float, default=0.0)
    parser.add_argument("--tl_yellow_cost", type=float, default=20.0)
    parser.add_argument("--tl_red_cost", type=float, default=80.0)

    # Optional smoothing.
    parser.add_argument(
        "--gaussian_blur_ksize",
        type=int,
        default=0,
        help="Set to an odd value such as 5 or 7 to smooth cost map. 0 disables smoothing.",
    )
    parser.add_argument("--gaussian_blur_sigma", type=float, default=1.0)

    # Saving options.
    parser.add_argument(
        "--save_compressed_npz",
        action="store_true",
        help="Also save cost as compressed npz with metadata.",
    )
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="Save debug PNGs under costmap_debug/.",
    )
    parser.add_argument(
        "--debug_clip_max",
        type=float,
        default=200.0,
        help="Cost value used as max for debug PNG normalization.",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.recursive:
        route_dirs = find_route_dirs(input_path, args)
        print(f"[Info] Found {len(route_dirs)} route directories under {input_path}")

        total_ok = 0
        total_frames = 0
        for route_dir in route_dirs:
            ok, total = process_route_dir(route_dir, args)
            total_ok += ok
            total_frames += total

        print(f"[All Done] {total_ok}/{total_frames} frames processed.")
    else:
        process_route_dir(input_path, args)


if __name__ == "__main__":
    main()