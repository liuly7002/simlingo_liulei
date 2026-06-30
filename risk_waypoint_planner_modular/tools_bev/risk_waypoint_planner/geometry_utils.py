# -*- coding: utf-8 -*-

from .common import *

def parse_floats_csv(csv: str) -> List[float]:
    vals = []
    for s in csv.split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals

def parse_strings_csv(csv: str) -> List[str]:
    vals = []
    for s in csv.split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(s)
    return vals

def parse_offsets(offset_str: str) -> List[float]:
    vals = parse_floats_csv(offset_str)
    if not any(abs(v) < 1e-6 for v in vals):
        vals = [0.0] + vals

    out = []
    for v in vals:
        if not any(abs(v - u) < 1e-6 for u in out):
            out.append(v)
    return out

def remove_duplicate_points(points: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 1:
        return points

    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > eps:
            keep.append(i)
    return points[keep]

def cumulative_distance(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(points) == 1:
        return np.zeros(1, dtype=np.float32)
    seg_len = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_len)]).astype(np.float32)

def resample_polyline(points: np.ndarray, spacing_m: float, horizon_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = remove_duplicate_points(points)

    if len(points) == 0:
        raise ValueError("Cannot resample an empty route.")

    if len(points) == 1:
        num = max(int(math.ceil(horizon_m / spacing_m)) + 1, 2)
        return np.repeat(points[:1], num, axis=0)

    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1)

    clean = [points[0]]
    for i in range(len(seg_len)):
        if seg_len[i] > 1e-6:
            clean.append(points[i + 1])
    points = np.asarray(clean, dtype=np.float32)

    if len(points) == 1:
        num = max(int(math.ceil(horizon_m / spacing_m)) + 1, 2)
        return np.repeat(points[:1], num, axis=0)

    s = cumulative_distance(points)
    total_len = float(s[-1])

    query_s = np.arange(0.0, horizon_m + 1e-6, spacing_m, dtype=np.float32)
    query_s = np.minimum(query_s, total_len)

    x = np.interp(query_s, s, points[:, 0])
    y = np.interp(query_s, s, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)

def sample_polyline_by_s(points: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = remove_duplicate_points(points)

    if len(points) == 0:
        raise ValueError("Cannot sample an empty polyline.")
    if len(points) == 1:
        return np.repeat(points[:1], len(query_s), axis=0)

    s = cumulative_distance(points)
    total_len = float(s[-1])
    query_s = np.asarray(query_s, dtype=np.float32)
    query_s = np.minimum(np.maximum(query_s, 0.0), total_len)

    x = np.interp(query_s, s, points[:, 0])
    y = np.interp(query_s, s, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)

def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def compute_right_normals(points: np.ndarray) -> np.ndarray:
    """
    For a straight route along +x, right normal is [0, 1].
    """
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 1:
        return np.array([[0.0, 1.0]], dtype=np.float32)

    tangents = np.zeros_like(points, dtype=np.float32)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if n > 2:
        tangents[1:-1] = points[2:] - points[:-2]

    norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-6)
    tangents = tangents / norm

    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return normals.astype(np.float32)

def make_offset_candidate(
    route: np.ndarray,
    offset_m: float,
    offset_start_m: float,
    offset_transition_m: float,
) -> np.ndarray:
    if abs(offset_m) < 1e-6:
        return route.copy()

    s = cumulative_distance(route)
    if offset_transition_m <= 1e-6:
        alpha = (s >= offset_start_m).astype(np.float32)
    else:
        alpha = smoothstep((s - offset_start_m) / offset_transition_m).astype(np.float32)

    normals = compute_right_normals(route)
    shifted = route + (offset_m * alpha)[:, None] * normals
    shifted[0] = route[0]
    return shifted.astype(np.float32)

def make_yield_reference_route(
    base_route: np.ndarray,
    stop_distance_m: float,
    spacing_m: float,
    horizon_m: float,
) -> np.ndarray:
    dense = resample_polyline(base_route, spacing_m=spacing_m, horizon_m=horizon_m)
    s = cumulative_distance(dense)
    idx = int(np.argmin(np.abs(s - stop_distance_m)))
    stop_pt = dense[idx:idx + 1]
    out = dense.copy()
    out[idx:] = stop_pt
    return out.astype(np.float32)

def min_distance_to_polyline_points(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    polyline = np.asarray(polyline, dtype=np.float32)
    if len(points) == 0 or len(polyline) == 0:
        return np.zeros((len(points),), dtype=np.float32)

    # Dense route is small, broadcasting is acceptable.
    d = np.linalg.norm(points[:, None, :] - polyline[None, :, :], axis=2)
    return np.min(d, axis=1).astype(np.float32)

def make_return_to_route_offset_candidate(
    route: np.ndarray,
    offset_m: float,
    offset_start_m: float,
    offset_transition_m: float,
    return_start_m: float,
    return_transition_m: float,
) -> np.ndarray:
    """
    Create a lateral nudge candidate that smoothly leaves the original route
    and then returns to it before the horizon ends.

    This is closer to a behavior candidate than a permanently shifted route.
    """
    route = np.asarray(route, dtype=np.float32)
    if abs(offset_m) < 1e-6:
        return route.copy()

    s = cumulative_distance(route)
    if offset_transition_m <= 1e-6:
        ramp_up = (s >= offset_start_m).astype(np.float32)
    else:
        ramp_up = smoothstep((s - offset_start_m) / offset_transition_m).astype(np.float32)

    if return_transition_m <= 1e-6:
        ramp_down = (s < return_start_m).astype(np.float32)
    else:
        ramp_down = 1.0 - smoothstep((s - return_start_m) / return_transition_m).astype(np.float32)

    alpha = np.clip(ramp_up * ramp_down, 0.0, 1.0)
    normals = compute_right_normals(route)
    shifted = route + (float(offset_m) * alpha)[:, None] * normals
    shifted[0] = route[0]
    return shifted.astype(np.float32)

def make_straight_reference(horizon_m: float, spacing_m: float) -> np.ndarray:
    q = np.arange(0.0, float(horizon_m) + 1e-6, float(max(spacing_m, 0.1)), dtype=np.float32)
    if len(q) < 2:
        q = np.asarray([0.0, float(horizon_m)], dtype=np.float32)
    return np.stack([q, np.zeros_like(q)], axis=1).astype(np.float32)

def normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

def transform_world_to_vehicle(point_xy: np.ndarray, state: Dict) -> np.ndarray:
    dx = float(point_xy[0] - state["x"])
    dy = float(point_xy[1] - state["y"])
    yaw = float(state["yaw"])
    c = math.cos(yaw)
    s = math.sin(yaw)

    # Vehicle coordinates: x forward, y right.
    x_v = c * dx + s * dy
    y_v = -s * dx + c * dy
    return np.asarray([x_v, y_v], dtype=np.float32)

def transform_vehicle_to_world(local_xy: np.ndarray, state: Dict) -> np.ndarray:
    x_v = float(local_xy[0])
    y_v = float(local_xy[1])
    yaw = float(state["yaw"])
    c = math.cos(yaw)
    s = math.sin(yaw)

    # Columns are vehicle forward and right axes in world coordinates.
    x_w = float(state["x"]) + c * x_v - s * y_v
    y_w = float(state["y"]) + s * x_v + c * y_v
    return np.asarray([x_w, y_w], dtype=np.float32)

def global_point_to_current_ego_local(
    point_global: np.ndarray,
    ego_global: np.ndarray,
    ego_yaw: float,
) -> np.ndarray:
    """
    Convert global XY point to current ego-local coordinates.

    local:
        x: forward
        y: right
    """
    dx = float(point_global[0] - ego_global[0])
    dy = float(point_global[1] - ego_global[1])

    c = math.cos(float(ego_yaw))
    s = math.sin(float(ego_yaw))

    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c

    return np.asarray([local_x, local_y], dtype=np.float32)

def ensure_route_starts_at_ego(
    route: np.ndarray,
    prepend_threshold_m: float = 0.5,
) -> np.ndarray:
    """
    Ensure local reference route starts from current ego position [0, 0].

    Some measurement['route'] points start several meters ahead of ego.
    If we directly use such a route, early rollout states near [0,0] will be
    wrongly judged as far from the reference route.
    """
    route = np.asarray(route, dtype=np.float32)

    if route.ndim != 2 or route.shape[1] < 2:
        raise ValueError(f"Invalid route shape: {route.shape}")

    route = route[:, :2]

    if len(route) == 0:
        return np.zeros((1, 2), dtype=np.float32)

    first_dist = float(np.linalg.norm(route[0]))

    if first_dist > prepend_threshold_m:
        route = np.concatenate(
            [np.zeros((1, 2), dtype=np.float32), route],
            axis=0,
        )
    else:
        route[0] = np.array([0.0, 0.0], dtype=np.float32)

    return route.astype(np.float32)
