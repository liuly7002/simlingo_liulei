# -*- coding: utf-8 -*-

from .common import *

def local_points_to_pixels(points: np.ndarray, ego_center: List[float], meters_per_pixel: float) -> np.ndarray:
    cx, cy = float(ego_center[0]), float(ego_center[1])
    points = np.asarray(points, dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]

    col = cx + y / meters_per_pixel
    row = cy - x / meters_per_pixel
    return np.stack([col, row], axis=1).astype(np.float32)

def sample_cost_bilinear(
    cost: np.ndarray,
    pixels: np.ndarray,
    out_of_bounds_cost: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = cost.shape
    col = pixels[:, 0]
    row = pixels[:, 1]

    valid = (col >= 0.0) & (col <= w - 1.0) & (row >= 0.0) & (row <= h - 1.0)
    sampled = np.full((len(pixels),), float(out_of_bounds_cost), dtype=np.float32)

    if not np.any(valid):
        return sampled, valid

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

    v0 = v00 * (1.0 - dc) + v01 * dc
    v1 = v10 * (1.0 - dc) + v11 * dc
    v = v0 * (1.0 - dr) + v1 * dr

    sampled[valid] = v.astype(np.float32)
    return sampled, valid
