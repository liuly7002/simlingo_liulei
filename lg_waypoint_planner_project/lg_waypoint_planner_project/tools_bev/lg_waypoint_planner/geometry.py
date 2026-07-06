# -*- coding: utf-8 -*-

from typing import Dict, Iterable, List, Optional, Tuple
import math
import numpy as np


def as_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Invalid point array shape: {arr.shape}")
    return arr[:, :2].astype(np.float32)


def remove_duplicate_points(points: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    pts = as_points(points)
    if len(pts) <= 1:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > eps:
            keep.append(i)
    return pts[keep]


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    pts = as_points(points)
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(pts) == 1:
        return np.zeros((1,), dtype=np.float32)
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)


def resample_polyline(points: np.ndarray, spacing_m: float, horizon_m: float) -> np.ndarray:
    pts = remove_duplicate_points(points)
    spacing_m = float(max(spacing_m, 1e-3))
    horizon_m = float(max(horizon_m, spacing_m))
    if len(pts) == 0:
        raise ValueError("Cannot resample an empty polyline.")
    if len(pts) == 1:
        n = max(int(math.ceil(horizon_m / spacing_m)) + 1, 2)
        return np.repeat(pts[:1], n, axis=0)
    s = cumulative_distance(pts)
    total = float(max(s[-1], 1e-6))
    q = np.arange(0.0, horizon_m + 1e-6, spacing_m, dtype=np.float32)
    q = np.minimum(q, total)
    x = np.interp(q, s, pts[:, 0])
    y = np.interp(q, s, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_polyline_by_s(points: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    pts = remove_duplicate_points(points)
    q = np.asarray(query_s, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError("Cannot sample an empty polyline.")
    if len(pts) == 1:
        return np.repeat(pts[:1], len(q), axis=0)
    s = cumulative_distance(pts)
    q = np.clip(q, 0.0, float(s[-1]))
    x = np.interp(q, s, pts[:, 0])
    y = np.interp(q, s, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def ensure_route_starts_at_ego(route: np.ndarray, threshold_m: float = 0.5) -> np.ndarray:
    pts = as_points(route)
    if len(pts) == 0:
        return np.zeros((1, 2), dtype=np.float32)
    if float(np.linalg.norm(pts[0])) > threshold_m:
        pts = np.concatenate([np.zeros((1, 2), dtype=np.float32), pts], axis=0)
    else:
        pts[0] = 0.0
    return pts.astype(np.float32)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def right_normals(route: np.ndarray) -> np.ndarray:
    pts = as_points(route)
    if len(pts) <= 1:
        return np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (len(pts), 1))
    tan = np.zeros_like(pts, dtype=np.float32)
    tan[0] = pts[1] - pts[0]
    tan[-1] = pts[-1] - pts[-2]
    if len(pts) > 2:
        tan[1:-1] = pts[2:] - pts[:-2]
    n = np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
    tan = tan / n
    return np.stack([-tan[:, 1], tan[:, 0]], axis=1).astype(np.float32)


def make_lateral_offset_route(route: np.ndarray, offset_m: float, start_m: float, transition_m: float, return_start_m: Optional[float] = None, return_transition_m: Optional[float] = None) -> np.ndarray:
    pts = as_points(route)
    if abs(float(offset_m)) < 1e-6:
        return pts.copy()
    s = cumulative_distance(pts)
    up = smoothstep((s - float(start_m)) / max(float(transition_m), 1e-3))
    if return_start_m is not None:
        down = 1.0 - smoothstep((s - float(return_start_m)) / max(float(return_transition_m or transition_m), 1e-3))
        alpha = np.clip(up * down, 0.0, 1.0)
    else:
        alpha = up
    out = pts + (float(offset_m) * alpha)[:, None] * right_normals(pts)
    out[0] = pts[0]
    return out.astype(np.float32)


def make_stop_route(route: np.ndarray, stop_distance_m: float) -> np.ndarray:
    pts = as_points(route).copy()
    s = cumulative_distance(pts)
    idx = int(np.argmin(np.abs(s - float(stop_distance_m))))
    pts[idx:] = pts[idx]
    return pts.astype(np.float32)


def min_distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    pts = as_points(points)
    line = as_points(polyline)
    if len(pts) == 0 or len(line) == 0:
        return np.zeros((len(pts),), dtype=np.float32)
    d = np.linalg.norm(pts[:, None, :] - line[None, :, :], axis=2)
    return np.min(d, axis=1).astype(np.float32)


def normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def transform_world_to_vehicle(point_xy: np.ndarray, state: Dict) -> np.ndarray:
    dx = float(point_xy[0] - state["x"])
    dy = float(point_xy[1] - state["y"])
    yaw = float(state["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)


def global_to_ego_local(point_global: np.ndarray, ego_global: np.ndarray, ego_yaw: float) -> np.ndarray:
    dx = float(point_global[0] - ego_global[0])
    dy = float(point_global[1] - ego_global[1])
    c, s = math.cos(float(ego_yaw)), math.sin(float(ego_yaw))
    return np.asarray([dx * c + dy * s, -dx * s + dy * c], dtype=np.float32)


def yaw_from_matrix_xy(mat: np.ndarray) -> float:
    # In ego/local convention, x axis is forward and y axis is right.
    return float(math.atan2(float(mat[1, 0]), float(mat[0, 0])))


def local_points_to_pixels(points: np.ndarray, ego_center: List[float], meters_per_pixel: float) -> np.ndarray:
    pts = as_points(points)
    cx, cy = float(ego_center[0]), float(ego_center[1])
    col = cx + pts[:, 1] / float(meters_per_pixel)
    row = cy - pts[:, 0] / float(meters_per_pixel)
    return np.stack([col, row], axis=1).astype(np.float32)


def points_in_obb(local_points: np.ndarray, center_xy: np.ndarray, yaw: float, half_length: float, half_width: float) -> np.ndarray:
    pts = as_points(local_points)
    dx = pts[:, 0] - float(center_xy[0])
    dy = pts[:, 1] - float(center_xy[1])
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    # inverse rotation from current frame to actor frame
    x = c * dx + s * dy
    y = -s * dx + c * dy
    return (np.abs(x) <= float(half_length)) & (np.abs(y) <= float(half_width))
