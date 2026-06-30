# -*- coding: utf-8 -*-

from .common import *
from .io_utils import load_json_gz
from .geometry_utils import transform_vehicle_to_world, parse_strings_csv
from .costmap_utils import local_points_to_pixels
from .qa_annotation import safe_float

def build_projection_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    """
    Build camera intrinsic matrix from image size and horizontal FOV.

    Coordinate convention used for projection:
        camera x: forward/depth
        camera y: right
        camera z: up

    Pixel projection:
        u = fx * y / x + cx
        v = cy - fy * z / x
    """
    f = float(width) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    K = np.array([
        [f, 0.0, float(width) / 2.0],
        [0.0, f, float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K

def rotation_matrix_roll_pitch_yaw_deg(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """
    Build rotation matrix from camera local frame to ego frame.

    Ego frame:
        x forward, y right, z up

    roll  : rotation around x
    pitch : rotation around y
    yaw   : rotation around z
    """
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float32)

    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float32)

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)

def project_ego_local_points_to_rgb(
    points_local_xy: np.ndarray,
    image_shape: Tuple[int, int],
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project ego-local ground waypoints to RGB image pixels.

    Input points:
        points_local_xy[:, 0] = x forward in ego frame
        points_local_xy[:, 1] = y right in ego frame

    Camera mounting:
        args.camera_x, args.camera_y, args.camera_z
        args.camera_roll_deg, args.camera_pitch_deg, args.camera_yaw_deg

    Output:
        pixels: [N, 2], each row is [u, v]
        valid:  [N], whether the point is in front of camera and inside image
    """
    points_local_xy = np.asarray(points_local_xy, dtype=np.float32)

    if points_local_xy is None or len(points_local_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    if points_local_xy.ndim != 2 or points_local_xy.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    h, w = int(image_shape[0]), int(image_shape[1])
    K = build_projection_matrix(w, h, args.camera_fov)

    z = np.full((len(points_local_xy), 1), float(args.waypoint_ground_z_m), dtype=np.float32)
    points_ego = np.concatenate([points_local_xy[:, :2], z], axis=1)

    cam_t_ego = np.array(
        [args.camera_x, args.camera_y, args.camera_z],
        dtype=np.float32,
    )

    R_cam_to_ego = rotation_matrix_roll_pitch_yaw_deg(
        roll_deg=args.camera_roll_deg,
        pitch_deg=args.camera_pitch_deg,
        yaw_deg=args.camera_yaw_deg,
    )

    # ego -> camera
    points_rel = points_ego - cam_t_ego[None, :]
    points_cam = (R_cam_to_ego.T @ points_rel.T).T

    depth = points_cam[:, 0]
    right = points_cam[:, 1]
    up = points_cam[:, 2]

    pixels = np.zeros((len(points_cam), 2), dtype=np.float32)

    valid = depth > float(args.min_projection_depth_m)
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

def draw_projected_waypoints_on_rgb(
    image: np.ndarray,
    points_local_xy: Optional[np.ndarray],
    args,
    color: Tuple[int, int, int],
    radius: int,
    thickness: int,
) -> int:
    """
    Draw projected waypoints on RGB image.

    Note:
        image is OpenCV BGR image.
        color is BGR.
    """
    if points_local_xy is None:
        return 0

    points_local_xy = np.asarray(points_local_xy, dtype=np.float32)
    if len(points_local_xy) == 0:
        return 0

    pixels, valid = project_ego_local_points_to_rgb(
        points_local_xy=points_local_xy,
        image_shape=image.shape[:2],
        args=args,
    )

    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return 0

    pts = np.round(pixels).astype(np.int32)

    # Draw line segments only when two consecutive points are both valid.
    for i in range(len(pts) - 1):
        if valid[i] and valid[i + 1]:
            p1 = (int(pts[i, 0]), int(pts[i, 1]))
            p2 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
            cv2.line(image, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Draw waypoint points.
    for idx in valid_indices:
        p = (int(pts[idx, 0]), int(pts[idx, 1]))
        cv2.circle(image, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return int(len(valid_indices))

def draw_projected_route_on_rgb(
    image: np.ndarray,
    route_local_xy: Optional[np.ndarray],
    args,
    color: Tuple[int, int, int],
    thickness: int,
) -> int:
    """
    Draw projected route polyline on RGB image.

    Note:
        image is OpenCV BGR image.
        color is BGR.
    """
    if route_local_xy is None:
        return 0

    route_local_xy = np.asarray(route_local_xy, dtype=np.float32)
    if len(route_local_xy) < 2:
        return 0

    pixels, valid = project_ego_local_points_to_rgb(
        points_local_xy=route_local_xy,
        image_shape=image.shape[:2],
        args=args,
    )

    if len(pixels) < 2:
        return 0

    pts = np.round(pixels).astype(np.int32)

    num_valid = int(np.sum(valid))

    # Draw only consecutive valid segments.
    for i in range(len(pts) - 1):
        if valid[i] and valid[i + 1]:
            p1 = (int(pts[i, 0]), int(pts[i, 1]))
            p2 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
            cv2.line(
                image,
                p1,
                p2,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    return num_valid

def find_rgb_image_path(route_dir: Path, frame_name: str, args) -> Optional[Path]:
    rgb_folder = args.rgb_folder_augmented if args.augmented else args.rgb_folder
    rgb_dir = route_dir / rgb_folder

    candidates = [
        rgb_dir / f"{frame_name}.jpg",
        rgb_dir / f"{frame_name}.png",
        rgb_dir / f"{frame_name}.jpeg",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None

def find_boxes_path(route_dir: Path, frame_name: str, args) -> Optional[Path]:
    """
    Locate the boxes json.gz file for one frame.

    In the current data_agent.py, bounding boxes are saved under route_dir/boxes.
    This helper still supports an augmented folder name, but its default is also
    "boxes" because augmented RGB shares the same actor boxes in the current
    data-collection script.
    """
    boxes_folder = args.boxes_folder_augmented if args.augmented else args.boxes_folder
    boxes_dir = route_dir / boxes_folder
    p = boxes_dir / f"{frame_name}.json.gz"
    if p.exists():
        return p
    return None

def load_boxes_for_frame(route_dir: Path, frame_name: str, args) -> List[Dict]:
    boxes_path = find_boxes_path(route_dir, frame_name, args)
    if boxes_path is None:
        return []

    try:
        obj = load_json_gz(boxes_path)
    except Exception as e:
        LOGGER.info(f"[RGB Actor Debug Skip] failed to read boxes: {boxes_path}, error={e}")
        return []

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "bounding_boxes" in obj and isinstance(obj["bounding_boxes"], list):
        return obj["bounding_boxes"]
    return []

def parse_actor_class_set(class_csv: str) -> set:
    return set(parse_strings_csv(class_csv))

def get_box_matrix(box: Dict) -> Optional[np.ndarray]:
    if "matrix" not in box:
        return None
    try:
        mat = np.asarray(box["matrix"], dtype=np.float64)
    except Exception:
        return None
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        return None
    return mat

def get_current_ego_matrix_from_boxes(boxes: List[Dict]) -> Optional[np.ndarray]:
    for box in boxes:
        if box.get("class") == "ego_car":
            return get_box_matrix(box)
    return None

def collect_future_actor_tracks_from_boxes(
    route_dir: Path,
    frame_name: str,
    args,
) -> Dict[int, Dict]:
    """
    Collect surrounding vehicles' future center points in the CURRENT frame's
    ego-local coordinate system.

    The important part is the coordinate transform:
        future actor world pose -> current ego frame

    We do NOT directly use box['position'] from each future frame because that
    position is relative to the ego vehicle of that same future frame. Directly
    connecting those positions would mix coordinate systems.

    Returned format:
        {
            actor_id: {
                "id": actor_id,
                "class": "car",
                "frames": ["0001", ...],
                "points": np.ndarray [N, 2],  # x forward, y right in frame_name ego frame
            },
            ...
        }
    """
    if frame_name is None:
        return {}

    current_boxes = load_boxes_for_frame(route_dir, frame_name, args)
    if len(current_boxes) == 0:
        LOGGER.info(f"[RGB Actor Debug Skip] current boxes not found/empty: route={route_dir}, frame={frame_name}")
        return {}

    ego0_matrix = get_current_ego_matrix_from_boxes(current_boxes)
    if ego0_matrix is None:
        LOGGER.info(f"[RGB Actor Debug Skip] ego_car matrix not found in current boxes: route={route_dir}, frame={frame_name}")
        return {}

    ego0_inv = np.linalg.inv(ego0_matrix)
    actor_classes = parse_actor_class_set(args.future_actor_classes)

    # By default, only draw actors that are already surrounding the ego in the
    # current frame. This avoids drawing newly appearing far-away actors that are
    # irrelevant for current-frame collision debugging.
    current_actor_ids = set()
    for box in current_boxes:
        if box.get("class") not in actor_classes:
            continue
        if "id" not in box:
            continue
        try:
            actor_id = int(box["id"])
        except Exception:
            continue
        dist = float(box.get("distance", 0.0))
        if dist <= float(args.future_actor_max_distance_m):
            current_actor_ids.add(actor_id)

    tracks: Dict[int, Dict] = {}

    try:
        frame_int = int(frame_name)
    except ValueError:
        LOGGER.info(f"[RGB Actor Debug Skip] non-numeric frame name: {frame_name}")
        return {}

    width = len(frame_name)

    for k in range(1, int(args.num_future_waypoints) + 1):
        future_int = frame_int + k * int(args.future_frame_stride)
        future_name = str(future_int).zfill(width)
        future_boxes = load_boxes_for_frame(route_dir, future_name, args)
        if len(future_boxes) == 0:
            continue

        for box in future_boxes:
            actor_class = box.get("class")
            if actor_class not in actor_classes:
                continue
            if "id" not in box:
                continue

            try:
                actor_id = int(box["id"])
            except Exception:
                continue

            if (not args.future_actor_include_new) and actor_id not in current_actor_ids:
                continue

            actor_matrix = get_box_matrix(box)
            if actor_matrix is None:
                continue

            actor_world = actor_matrix[:3, 3]
            actor_world_h = np.asarray(
                [actor_world[0], actor_world[1], actor_world[2], 1.0],
                dtype=np.float64,
            )

            # Convert future actor world center to current-frame ego-local.
            actor_in_ego0 = ego0_inv @ actor_world_h
            x = float(actor_in_ego0[0])
            y = float(actor_in_ego0[1])

            if not np.isfinite([x, y]).all():
                continue
            if math.hypot(x, y) > float(args.future_actor_max_distance_m):
                continue

            if actor_id not in tracks:
                tracks[actor_id] = {
                    "id": actor_id,
                    "class": actor_class,
                    "frames": [],
                    "points_list": [],
                }

            tracks[actor_id]["frames"].append(future_name)
            tracks[actor_id]["points_list"].append([x, y])

    # Convert point lists to arrays and remove too-short tracks.
    out: Dict[int, Dict] = {}
    for actor_id, item in tracks.items():
        pts = np.asarray(item.pop("points_list"), dtype=np.float32)
        if len(pts) == 0:
            continue
        item["points"] = pts
        out[actor_id] = item

    return out

def draw_future_actor_tracks_on_rgb(
    image: np.ndarray,
    actor_tracks: Dict[int, Dict],
    args,
) -> Tuple[int, int]:
    """
    Draw future surrounding-vehicle center trajectories on front RGB.

    Note:
        image is OpenCV BGR image.
        actor track color is BGR.

    Returns:
        (num_tracks_with_at_least_one_visible_point, num_visible_points)
    """
    if not actor_tracks:
        return 0, 0

    num_tracks_visible = 0
    num_points_visible = 0

    color = (
        int(args.future_actor_rgb_color_b),
        int(args.future_actor_rgb_color_g),
        int(args.future_actor_rgb_color_r),
    )

    for actor_id, item in sorted(actor_tracks.items(), key=lambda kv: kv[0]):
        points = np.asarray(item.get("points", []), dtype=np.float32)
        if len(points) == 0:
            continue

        pixels, valid = project_ego_local_points_to_rgb(
            points_local_xy=points,
            image_shape=image.shape[:2],
            args=args,
        )

        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            continue

        num_tracks_visible += 1
        num_points_visible += int(len(valid_indices))
        pts = np.round(pixels).astype(np.int32)

        # Draw line segments only when two consecutive future points are visible.
        for i in range(len(pts) - 1):
            if valid[i] and valid[i + 1]:
                p1 = (int(pts[i, 0]), int(pts[i, 1]))
                p2 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
                cv2.line(
                    image,
                    p1,
                    p2,
                    color=color,
                    thickness=int(args.future_actor_rgb_thickness),
                    lineType=cv2.LINE_AA,
                )

        # Draw future points. The first visible point uses a larger outline.
        for local_rank, idx in enumerate(valid_indices):
            p = (int(pts[idx, 0]), int(pts[idx, 1]))
            radius = int(args.future_actor_rgb_radius)
            if local_rank == 0:
                cv2.circle(image, p, radius=radius + 2, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(image, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

        if args.future_actor_draw_ids:
            first_idx = int(valid_indices[0])
            p0 = (int(pts[first_idx, 0]), int(pts[first_idx, 1]))
            label = f"car{actor_id}"
            cv2.putText(
                image,
                label,
                (p0[0] + 4, p0[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label,
                (p0[0] + 4, p0[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

    return int(num_tracks_visible), int(num_points_visible)

def save_rgb_waypoints_debug_image(
    route_dir: Path,
    frame_name: str,
    risk_planned_waypoints: np.ndarray,
    expert_future_waypoints: Optional[np.ndarray],
    expert_reference_route: np.ndarray,
    selected_reference_route: np.ndarray,
    selected_info: Dict,
    save_path: Path,
    args,
) -> None:
    rgb_path = find_rgb_image_path(route_dir, frame_name, args)

    if rgb_path is None:
        LOGGER.info(f"[RGB Debug Skip] rgb image not found: route={route_dir}, frame={frame_name}")
        return

    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        LOGGER.info(f"[RGB Debug Skip] failed to read rgb image: {rgb_path}")
        return

    # BGR colors for OpenCV.
    # original route: white
    # selected route: yellow
    # expert waypoints: blue
    # risk-planned waypoints: green

    num_expert_route = draw_projected_route_on_rgb(
        image=img,
        route_local_xy=expert_reference_route,
        args=args,
        color=(255, 255, 255),
        thickness=2,
    )

    num_selected_route = draw_projected_route_on_rgb(
        image=img,
        route_local_xy=selected_reference_route,
        args=args,
        color=(0, 255, 255),
        thickness=2,
    )

    num_expert = draw_projected_waypoints_on_rgb(
        image=img,
        points_local_xy=expert_future_waypoints,
        args=args,
        color=(255, 0, 0),
        radius=4,
        thickness=2,
    )

    num_risk = draw_projected_waypoints_on_rgb(
        image=img,
        points_local_xy=risk_planned_waypoints,
        args=args,
        color=(0, 255, 0),
        radius=4,
        thickness=2,
    )

    num_actor_tracks = 0
    num_actor_points = 0
    if not args.disable_future_actor_rgb:
        future_actor_tracks = collect_future_actor_tracks_from_boxes(
            route_dir=route_dir,
            frame_name=frame_name,
            args=args,
        )
        num_actor_tracks, num_actor_points = draw_future_actor_tracks_on_rgb(
            image=img,
            actor_tracks=future_actor_tracks,
            args=args,
        )

    text_lines = [
        "white: original route",
        "yellow: selected planned route",
        "blue: expert future waypoints",
        "green: risk-planned waypoints",
        "orange: future vehicle waypoints",
        f"selected: {selected_info.get('reference_mode', 'unknown')} + {selected_info.get('speed_mode', 'unknown')}",
        f"projected route ori/sel: {num_expert_route}/{num_selected_route}",
        f"projected wp expert/risk: {num_expert}/{num_risk}",
        f"projected future vehicles tracks/pts: {num_actor_tracks}/{num_actor_points}",
    ]

    x0, y0, gap = 10, 24, 20
    for i, text in enumerate(text_lines):
        cv2.putText(
            img,
            text,
            (x0, y0 + i * gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            (x0, y0 + i * gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)

def costmap_to_color(costmap: np.ndarray, clip_max: float) -> np.ndarray:
    if clip_max <= 0:
        clip_max = float(np.max(costmap)) if float(np.max(costmap)) > 0 else 1.0
    vis = np.clip(costmap, 0.0, clip_max) / clip_max
    vis = (vis * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

def draw_polyline(
    image: np.ndarray,
    points_local: np.ndarray,
    ego_center: List[float],
    meters_per_pixel: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    points_local = np.asarray(points_local, dtype=np.float32)
    if len(points_local) < 2:
        return

    pix = local_points_to_pixels(points_local, ego_center, meters_per_pixel)
    pts = np.round(pix).astype(np.int32)

    h, w = image.shape[:2]
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 2:
        return

    pts[:, 0] = np.clip(pts[:, 0], -10000, w + 10000)
    pts[:, 1] = np.clip(pts[:, 1], -10000, h + 10000)
    cv2.polylines(image, [pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness)

def draw_points(
    image: np.ndarray,
    points_local: np.ndarray,
    ego_center: List[float],
    meters_per_pixel: float,
    color: Tuple[int, int, int],
    radius: int,
) -> None:
    points_local = np.asarray(points_local, dtype=np.float32)
    if len(points_local) == 0:
        return

    pix = local_points_to_pixels(points_local, ego_center, meters_per_pixel)
    h, w = image.shape[:2]
    for p in pix:
        if not np.isfinite(p).all():
            continue
        c = int(round(float(p[0])))
        r = int(round(float(p[1])))
        if -100 <= c < w + 100 and -100 <= r < h + 100:
            cv2.circle(image, (c, r), radius=radius, color=color, thickness=-1)

def save_debug_image(
    costmap: np.ndarray,
    scored_rollouts: List[Dict],
    selected_idx: int,
    expert_reference_route: np.ndarray,
    ego_center: List[float],
    meters_per_pixel: float,
    save_path: Path,
    args,
) -> None:
    img = costmap_to_color(costmap, args.debug_clip_max)

    # Draw all candidate reference routes lightly.
    for r in scored_rollouts:
        draw_polyline(
            img,
            r["reference_route"],
            ego_center,
            meters_per_pixel,
            color=(120, 120, 120),
            thickness=1,
        )

    # Draw expert reference route.
    draw_polyline(
        img,
        expert_reference_route,
        ego_center,
        meters_per_pixel,
        color=(255, 255, 255),
        thickness=2,
    )

    # Draw candidate rollouts.
    for i, r in enumerate(scored_rollouts):
        if i == selected_idx:
            continue
        color = (160, 160, 160) if r["info"]["allowed"] else (80, 80, 220)
        draw_polyline(
            img,
            r["rollout"]["waypoints"],
            ego_center,
            meters_per_pixel,
            color=color,
            thickness=1,
        )
        draw_points(
            img,
            r["rollout"]["waypoints"],
            ego_center,
            meters_per_pixel,
            color=color,
            radius=1,
        )

    # Draw selected reference and rollout.
    selected = scored_rollouts[selected_idx]
    draw_polyline(
        img,
        selected["reference_route"],
        ego_center,
        meters_per_pixel,
        color=(255, 255, 0),
        thickness=2,
    )
    draw_polyline(
        img,
        selected["rollout"]["waypoints"],
        ego_center,
        meters_per_pixel,
        color=(0, 255, 0),
        thickness=2,
    )
    draw_points(
        img,
        selected["rollout"]["waypoints"],
        ego_center,
        meters_per_pixel,
        color=(0, 255, 0),
        radius=2,
    )

    # Draw ego point.
    cx, cy = int(round(ego_center[0])), int(round(ego_center[1]))
    cv2.circle(img, (cx, cy), radius=3, color=(255, 255, 255), thickness=-1)
    cv2.circle(img, (cx, cy), radius=5, color=(0, 0, 0), thickness=1)

    text_lines = [
        f"selected: {selected['info']['reference_mode']} + {selected['info']['speed_mode']}",
        f"score: {selected['info']['score']:.2f}",
        f"allowed: {selected['info']['allowed']}",
        f"mean/max cost: {selected['info']['mean_cost']:.1f}/{selected['info']['max_cost']:.1f}",
    ]

    x0, y0, gap = 8, 18, 16
    for i, text in enumerate(text_lines):
        cv2.putText(
            img,
            text,
            (x0, y0 + i * gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)
