# -*- coding: utf-8 -*-

from typing import Dict, Tuple
import math
import numpy as np

from .dataset import get_current_ego_state
from .geometry import cumulative_distance, normalize_angle, sample_polyline_by_s, transform_world_to_vehicle


def closest_s(reference_route: np.ndarray, state: Dict) -> float:
    xy = np.asarray([state["x"], state["y"]], dtype=np.float32)
    d = np.linalg.norm(reference_route - xy[None, :], axis=1)
    idx = int(np.argmin(d))
    return float(cumulative_distance(reference_route)[idx])


def pure_pursuit_steer(reference_route: np.ndarray, state: Dict, cfg) -> float:
    v = float(state["v"])
    lookahead = float(cfg.bicycle.lookahead_base_m) + float(cfg.bicycle.lookahead_gain) * v
    lookahead = float(np.clip(lookahead, float(cfg.bicycle.lookahead_min_m), float(cfg.bicycle.lookahead_max_m)))
    s0 = closest_s(reference_route, state)
    target = sample_polyline_by_s(reference_route, np.asarray([s0 + lookahead], dtype=np.float32))[0]
    tv = transform_world_to_vehicle(target, state)
    if tv[0] < 0.2:
        target = sample_polyline_by_s(reference_route, np.asarray([s0 + max(lookahead, 2.0)], dtype=np.float32))[0]
        tv = transform_world_to_vehicle(target, state)
    ld2 = max(float(tv[0] ** 2 + tv[1] ** 2), 1e-4)
    curvature = 2.0 * float(tv[1]) / ld2
    steer = math.atan(float(cfg.bicycle.wheelbase_m) * curvature)
    return float(np.clip(steer, -float(cfg.bicycle.max_steer_rad), float(cfg.bicycle.max_steer_rad)))


def longitudinal_accel(v: float, target_speed: float, cfg) -> float:
    acc = float(cfg.bicycle.speed_kp) * (float(target_speed) - float(v))
    return float(np.clip(acc, float(cfg.bicycle.min_accel_mps2), float(cfg.bicycle.max_accel_mps2)))


def bicycle_step(state: Dict, steer: float, acc: float, dt: float, cfg) -> Dict:
    x, y, yaw, v = float(state["x"]), float(state["y"]), float(state["yaw"]), float(state["v"])
    steer = float(np.clip(steer, -float(cfg.bicycle.max_steer_rad), float(cfg.bicycle.max_steer_rad)))
    acc = float(np.clip(acc, float(cfg.bicycle.min_accel_mps2), float(cfg.bicycle.max_accel_mps2)))
    v_next = float(np.clip(v + acc * dt, 0.0, float(cfg.bicycle.max_speed_mps)))
    v_mid = 0.5 * (v + v_next)
    x_next = x + v_mid * math.cos(yaw) * dt
    y_next = y + v_mid * math.sin(yaw) * dt
    yaw_next = normalize_angle(yaw + v_mid / max(float(cfg.bicycle.wheelbase_m), 1e-3) * math.tan(steer) * dt)
    return {"x": x_next, "y": y_next, "yaw": yaw_next, "v": v_next}


def _prepare_dense_target_profile(target_speed, measurement: Dict, cfg, steps: int, save_stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a scalar or waypoint-rate speed profile to the internal 20 Hz grid.

    The original implementation accepted only one scalar target speed.  The
    planner now also accepts a future speed profile so that response candidates
    can preserve the temporal structure of expert motion instead of collapsing
    it to one constant speed.
    """
    n_wp = int(cfg.horizon.num_future_waypoints)
    max_speed = float(cfg.bicycle.max_speed_mps)

    if np.isscalar(target_speed):
        value = float(np.clip(float(target_speed), 0.0, max_speed))
        dense = np.full((steps,), value, dtype=np.float32)
        saved = np.full((n_wp,), value, dtype=np.float32)
        return dense, saved

    arr = np.asarray(target_speed, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        value = max(float(measurement.get("speed", 0.0)), 0.0)
        dense = np.full((steps,), value, dtype=np.float32)
        saved = np.full((n_wp,), value, dtype=np.float32)
        return dense, saved

    arr = np.clip(arr, 0.0, max_speed).astype(np.float32)

    if len(arr) == steps:
        dense = arr.copy()
        saved = dense[save_stride - 1::save_stride][:n_wp].copy()
        if len(saved) < n_wp:
            saved = np.pad(saved, (0, n_wp - len(saved)), mode="edge")
        return dense.astype(np.float32), saved.astype(np.float32)

    # Treat every non-dense sequence as a future waypoint-rate profile.  This
    # keeps the public API simple and works for both shorter partial profiles
    # and the standard N-waypoint profile.
    if len(arr) != n_wp:
        src = np.linspace(1.0 / float(cfg.horizon.future_fps), n_wp / float(cfg.horizon.future_fps), len(arr), dtype=np.float32)
        dst = np.linspace(1.0 / float(cfg.horizon.future_fps), n_wp / float(cfg.horizon.future_fps), n_wp, dtype=np.float32)
        arr = np.interp(dst, src, arr).astype(np.float32)

    current_speed = max(float(measurement.get("speed", 0.0)), 0.0)
    t_saved = np.arange(1, n_wp + 1, dtype=np.float32) / float(cfg.horizon.future_fps)
    t_dense = np.arange(1, steps + 1, dtype=np.float32) / float(cfg.horizon.model_fps)
    dense = np.interp(
        t_dense,
        np.concatenate([np.asarray([0.0], dtype=np.float32), t_saved]),
        np.concatenate([np.asarray([current_speed], dtype=np.float32), arr]),
    ).astype(np.float32)
    return dense, arr.astype(np.float32)


def rollout(reference_route: np.ndarray, target_speed, measurement: Dict, cfg) -> Dict:
    """Roll out a kinematic bicycle model along ``reference_route``.

    ``target_speed`` may be either a scalar or a future speed profile sampled at
    ``future_fps``.  Supporting profiles is important for the new reference-
    response supervision because the expert's acceleration/deceleration timing
    should not be discarded when constructing response candidates.
    """
    ratio = float(cfg.horizon.model_fps) / float(cfg.horizon.future_fps)
    save_stride = int(round(ratio))
    if abs(ratio - save_stride) > 1e-6:
        raise ValueError(f"model_fps/future_fps must be integer. Got {cfg.horizon.model_fps}/{cfg.horizon.future_fps}")
    dt = 1.0 / float(cfg.horizon.model_fps)
    steps = int(cfg.horizon.num_future_waypoints) * save_stride

    dense_target, saved_target = _prepare_dense_target_profile(target_speed, measurement, cfg, steps, save_stride)

    state = get_current_ego_state(measurement)
    dense_states, dense_controls, saved_states, saved_controls = [], [], [], []
    prev_steer = 0.0
    for step in range(steps):
        raw = pure_pursuit_steer(reference_route, state, cfg)
        max_delta = float(cfg.bicycle.max_steer_rate_radps) * dt
        steer = float(np.clip(raw, prev_steer - max_delta, prev_steer + max_delta))
        prev_steer = steer
        acc = longitudinal_accel(float(state["v"]), float(dense_target[step]), cfg)
        state = bicycle_step(state, steer, acc, dt, cfg)
        dense_states.append([state["x"], state["y"], state["yaw"], state["v"]])
        dense_controls.append([steer, acc])
        if (step + 1) % save_stride == 0:
            saved_states.append([state["x"], state["y"], state["yaw"], state["v"]])
            saved_controls.append([steer, acc])

    dense = np.asarray(dense_states, dtype=np.float32)
    controls = np.asarray(dense_controls, dtype=np.float32)
    saved = np.asarray(saved_states, dtype=np.float32)
    saved_ctrl = np.asarray(saved_controls, dtype=np.float32)
    return {
        "waypoints": saved[:, :2].copy(),
        "yaws": saved[:, 2].copy(),
        "speeds": saved[:, 3].copy(),
        "controls": saved_ctrl.copy(),
        "dense_xy": dense[:, :2].copy(),
        "dense_yaw": dense[:, 2].copy(),
        "dense_speed": dense[:, 3].copy(),
        "dense_controls": controls.copy(),
        "target_speed": float(saved_target[-1]) if len(saved_target) else 0.0,
        "target_speed_profile": saved_target.copy(),
        "dense_target_speed_profile": dense_target.copy(),
    }
