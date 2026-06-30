# -*- coding: utf-8 -*-

from .common import *
from .geometry_utils import cumulative_distance, sample_polyline_by_s, normalize_angle, transform_world_to_vehicle
from .measurement_utils import get_current_ego_state

def find_closest_s_on_reference(reference_route: np.ndarray, state: Dict) -> float:
    xy = np.asarray([state["x"], state["y"]], dtype=np.float32)
    d = np.linalg.norm(reference_route - xy[None, :], axis=1)
    idx = int(np.argmin(d))
    s = cumulative_distance(reference_route)
    return float(s[idx])

def pure_pursuit_steer(reference_route: np.ndarray, state: Dict, args) -> Tuple[float, Dict]:
    v = float(state["v"])
    lookahead = args.lookahead_base_m + args.lookahead_gain * v
    lookahead = float(np.clip(lookahead, args.lookahead_min_m, args.lookahead_max_m))

    s_closest = find_closest_s_on_reference(reference_route, state)
    target_s = s_closest + lookahead
    target = sample_polyline_by_s(reference_route, np.asarray([target_s], dtype=np.float32))[0]
    target_v = transform_world_to_vehicle(target, state)

    # If target is behind due to a bad route/edge case, use a slightly further point.
    if target_v[0] < 0.2:
        target_s = s_closest + max(lookahead, 2.0)
        target = sample_polyline_by_s(reference_route, np.asarray([target_s], dtype=np.float32))[0]
        target_v = transform_world_to_vehicle(target, state)

    ld2 = max(float(target_v[0] ** 2 + target_v[1] ** 2), 1e-4)
    curvature = 2.0 * float(target_v[1]) / ld2
    steer = math.atan(args.wheelbase_m * curvature)
    steer = float(np.clip(steer, -args.max_steer_rad, args.max_steer_rad))

    return steer, {
        "lookahead_m": lookahead,
        "target_s": float(target_s),
        "target_point": target.tolist(),
        "target_vehicle": target_v.tolist(),
        "curvature": float(curvature),
    }

def longitudinal_accel(state: Dict, target_speed: float, args) -> float:
    v = float(state["v"])
    err = float(target_speed) - v
    acc = args.speed_kp * err
    acc = float(np.clip(acc, args.min_accel_mps2, args.max_accel_mps2))
    return acc

def bicycle_step(state: Dict, steer: float, acc: float, dt: float, args) -> Dict:
    x = float(state["x"])
    y = float(state["y"])
    yaw = float(state["yaw"])
    v = float(state["v"])

    steer = float(np.clip(steer, -args.max_steer_rad, args.max_steer_rad))
    acc = float(np.clip(acc, args.min_accel_mps2, args.max_accel_mps2))

    v_next = float(np.clip(v + acc * dt, 0.0, args.max_speed_mps))
    v_mid = 0.5 * (v + v_next)

    x_next = x + v_mid * math.cos(yaw) * dt
    y_next = y + v_mid * math.sin(yaw) * dt
    yaw_next = yaw + v_mid / max(args.wheelbase_m, 1e-3) * math.tan(steer) * dt
    yaw_next = normalize_angle(yaw_next)

    return {"x": x_next, "y": y_next, "yaw": yaw_next, "v": v_next}

def rollout_candidate_with_bicycle_model(
    reference_route: np.ndarray,
    speed_profile: Dict,
    measurement: Dict,
    args,
) -> Dict:
    """
    Roll out ego future trajectory using kinematic bicycle model.

    Output waypoints are saved at args.future_fps, while internal model runs at
    args.model_fps.
    """
    model_fps = float(args.model_fps)
    future_fps = float(args.future_fps)
    save_stride = int(round(model_fps / future_fps))
    if save_stride <= 0:
        raise ValueError("Invalid model_fps/future_fps configuration.")

    dt = 1.0 / model_fps
    num_steps = int(args.num_future_waypoints * save_stride)

    state = get_current_ego_state(measurement)
    target_speed = float(speed_profile["target_speed"])

    dense_states = []
    dense_controls = []
    dense_aux = []

    saved_states = []
    saved_controls = []

    prev_steer = 0.0

    for step in range(num_steps):
        raw_steer, aux = pure_pursuit_steer(reference_route, state, args)

        # Steering-rate limit.
        max_delta = args.max_steer_rate_radps * dt
        steer = float(np.clip(raw_steer, prev_steer - max_delta, prev_steer + max_delta))
        prev_steer = steer

        acc = longitudinal_accel(state, target_speed=target_speed, args=args)

        state = bicycle_step(state, steer=steer, acc=acc, dt=dt, args=args)

        lat_acc = (float(state["v"]) ** 2) * math.tan(steer) / max(args.wheelbase_m, 1e-3)
        yaw_rate = float(state["v"]) * math.tan(steer) / max(args.wheelbase_m, 1e-3)

        dense_states.append([state["x"], state["y"], state["yaw"], state["v"]])
        dense_controls.append([steer, acc])
        aux.update({
            "lat_acc_mps2": float(lat_acc),
            "yaw_rate_radps": float(yaw_rate),
        })
        dense_aux.append(aux)

        if (step + 1) % save_stride == 0:
            saved_states.append([state["x"], state["y"], state["yaw"], state["v"]])
            saved_controls.append([steer, acc])

    dense_states = np.asarray(dense_states, dtype=np.float32)
    dense_controls = np.asarray(dense_controls, dtype=np.float32)
    saved_states = np.asarray(saved_states, dtype=np.float32)
    saved_controls = np.asarray(saved_controls, dtype=np.float32)

    return {
        "waypoints": saved_states[:, :2].copy(),
        "yaws": saved_states[:, 2].copy(),
        "speeds": saved_states[:, 3].copy(),
        "controls": saved_controls.copy(),
        "dense_xy": dense_states[:, :2].copy(),
        "dense_yaw": dense_states[:, 2].copy(),
        "dense_speed": dense_states[:, 3].copy(),
        "dense_controls": dense_controls.copy(),
        "dense_aux": dense_aux,
        "target_speed": target_speed,
    }
