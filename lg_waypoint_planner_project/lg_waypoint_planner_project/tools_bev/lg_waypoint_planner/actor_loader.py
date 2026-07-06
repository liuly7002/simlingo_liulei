# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional
import math
import numpy as np

from .io_utils import load_json_gz
from .dataset import offset_frame_name
from .geometry import yaw_from_matrix_xy


# CARLA 0.9.15 built-in road-obstacle props that can physically constrain the
# ego vehicle.  The normalizer below maps common exported class/blueprint names
# onto these three stable categories.  We intentionally do not include arbitrary
# custom categories; unmatched static props are ignored for language focus.
DEFAULT_DYNAMIC_CLASSES = {"vehicle", "pedestrian"}
DEFAULT_STATIC_OBSTACLE_CLASSES = {"traffic_cone", "traffic_warning", "barrier"}



def load_boxes(route_dir: Path, frame_name: str, cfg) -> List[Dict]:
    p = route_dir / cfg.paths.boxes_folder / f"{frame_name}.json.gz"
    if not p.exists():
        return []
    obj = load_json_gz(p)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["boxes", "bounding_boxes", "actors", "objects"]:
            if isinstance(obj.get(key), list):
                return obj[key]
    return []


def _raw_class_string(box: Dict) -> str:
    values = []
    for key in ["class", "type", "type_id", "role_name", "blueprint", "blueprint_id", "mesh_path", "id"]:
        v = box.get(key, None)
        if isinstance(v, str) and v:
            values.append(v)
    return " ".join(values).lower()


def normalize_actor_class(box: Dict) -> str:
    """Normalize exported CARLA classes into a small planning vocabulary.

    Dynamic road users are normalized to ``vehicle`` and ``pedestrian``.
    CARLA built-in static road-obstacle props are normalized to:
        ``traffic_cone``, ``traffic_warning``, and ``barrier``.

    The static categories are intentionally conservative.  They cover CARLA
    built-in construction cones / traffic-warning props / barriers, without
    treating arbitrary static props or regular traffic lights as critical
    objects for language supervision.
    """
    raw = _raw_class_string(box)
    compact = raw.replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")

    if any(k in compact for k in ["ego_car", "ego_info", "ego_vehicle"]):
        return "ego"
    if compact in ["ego"]:
        return "ego"

    if any(k in compact for k in [
        "vehicle", "car", "truck", "bus", "van", "motorcycle", "bicycle", "bike"
    ]):
        return "vehicle"

    if any(k in compact for k in ["walker", "pedestrian", "person"]):
        return "pedestrian"

    # CARLA road-work / traffic-control props that physically occupy road space.
    if any(k in compact for k in ["trafficcone", "constructioncone", "traffic_cone", "construction_cone"]):
        return "traffic_cone"
    if any(k in compact for k in [
        "trafficwarning", "traffic_warning", "warningconstruction", "warning_construction",
        "constructionwarning", "construction_warning", "warningaccident", "warning_accident"
    ]):
        return "traffic_warning"
    if "barrier" in compact or "barricade" in compact:
        return "barrier"

    return compact


def _cfg_class_set(cfg, name: str, default: set) -> set:
    try:
        values = getattr(cfg.actors, name)
    except Exception:
        values = default
    try:
        return {str(v).lower() for v in values}
    except Exception:
        return set(default)


def dynamic_actor_classes(cfg) -> set:
    return _cfg_class_set(cfg, "dynamic_classes", DEFAULT_DYNAMIC_CLASSES)


def static_obstacle_classes(cfg=None) -> set:
    if cfg is None:
        return set(DEFAULT_STATIC_OBSTACLE_CLASSES)
    return _cfg_class_set(cfg, "static_obstacle_classes", DEFAULT_STATIC_OBSTACLE_CLASSES)


def is_static_obstacle_class(cls: str, cfg=None) -> bool:
    return str(cls).lower() in static_obstacle_classes(cfg)


def is_dynamic_actor(box: Dict, cfg) -> bool:
    cls = normalize_actor_class(box)
    return cls in dynamic_actor_classes(cfg)


def is_relevant_actor(box: Dict, cfg) -> bool:
    cls = normalize_actor_class(box)
    return cls in dynamic_actor_classes(cfg) or cls in static_obstacle_classes(cfg)


def box_matrix(box: Dict) -> Optional[np.ndarray]:
    if "matrix" not in box:
        return None
    try:
        mat = np.asarray(box["matrix"], dtype=np.float64)
    except Exception:
        return None
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        return None
    return mat


def ego_matrix_from_boxes(boxes: List[Dict]) -> Optional[np.ndarray]:
    for b in boxes:
        if normalize_actor_class(b) == "ego":
            m = box_matrix(b)
            if m is not None:
                return m
    return None


def _read_position_xyz(box: Dict) -> Optional[np.ndarray]:
    for key in ["position", "location", "center"]:
        pos = box.get(key)
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            xyz = [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) >= 3 else 0.0]
            if np.isfinite(xyz).all():
                return np.asarray(xyz, dtype=np.float64)
    return None


def _read_xy_from_position(box: Dict) -> Optional[np.ndarray]:
    pos = _read_position_xyz(box)
    if pos is None:
        return None
    return pos[:2].astype(np.float32)


def _relative_matrix_from_local_box(
    box: Dict,
    source_ego_matrix: Optional[np.ndarray],
    target_ego_inv: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Reconstruct a box pose in the target ego frame from a source-frame local pose.

    Some collected static props contain only ``position`` and ``yaw`` in the
    ego frame of the frame where they were saved; they do not contain a world
    ``matrix``.  For future timelines, the source ego frame is different from
    the current ego frame, so the local pose must be lifted through the source
    ego world matrix before being expressed in the current ego frame.
    """
    if source_ego_matrix is None or target_ego_inv is None:
        return None
    pos = _read_position_xyz(box)
    if pos is None:
        return None

    yaw = _read_yaw(box)
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.eye(4, dtype=np.float64)
    local[0, 0] = c
    local[0, 1] = -s
    local[1, 0] = s
    local[1, 1] = c
    local[:3, 3] = pos
    return target_ego_inv @ source_ego_matrix @ local


def _read_yaw(box: Dict) -> float:
    for key in ["yaw", "theta", "rotation_yaw"]:
        if key in box:
            try:
                return float(box[key])
            except Exception:
                pass
    return 0.0


def _read_speed(box: Dict) -> float:
    for key in ["speed", "velocity", "speed_mps"]:
        if key in box:
            v = box[key]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return float(math.hypot(float(v[0]), float(v[1])))
            try:
                return max(0.0, float(v))
            except Exception:
                pass
    return 0.0


def _cfg_float(cfg, name: str, default: float) -> float:
    try:
        return float(getattr(cfg.actors, name))
    except Exception:
        return float(default)


def _normalize_exported_extents(box: Dict, cls: str, ex: float, ey: float) -> tuple:
    """Normalize exporter-specific footprint axis conventions.

    In the collected data, ``static_car`` props keep the vehicle heading in
    ``yaw`` but export the long side in ``extent.y``.  Treating extent.x as
    longitudinal therefore draws these parked/static cars sideways in BEV and
    corrupts their collision footprint.  Only this explicitly identified raw
    class is normalized; regular dynamic CARLA vehicles keep their native axes.
    """
    raw = _raw_class_string(box)
    compact = raw.replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")
    is_static_car = cls == "vehicle" and ("static_car" in compact or "staticcar" in compact)
    if is_static_car and ey > ex:
        return float(ey), float(ex)
    return float(ex), float(ey)


def _read_half_extents(box: Dict, cls: str, cfg) -> tuple:
    # CARLA exported box['extent'] is normally half-size. dimensions/size are commonly full size.
    if isinstance(box.get("extent"), (list, tuple)) and len(box["extent"]) >= 2:
        ex = max(abs(float(box["extent"][0])), 0.1)
        ey = max(abs(float(box["extent"][1])), 0.1)
        return _normalize_exported_extents(box, cls, ex, ey)
    for key in ["dimensions", "size"]:
        if isinstance(box.get(key), (list, tuple)) and len(box[key]) >= 2:
            ex = max(abs(float(box[key][0])) * 0.5, 0.1)
            ey = max(abs(float(box[key][1])) * 0.5, 0.1)
            return _normalize_exported_extents(box, cls, ex, ey)

    if cls == "pedestrian":
        return _cfg_float(cfg, "default_ped_half_length_m", 0.35), _cfg_float(cfg, "default_ped_half_width_m", 0.35)
    if cls == "traffic_cone":
        return _cfg_float(cfg, "default_cone_half_length_m", 0.25), _cfg_float(cfg, "default_cone_half_width_m", 0.25)
    if cls == "traffic_warning":
        return _cfg_float(cfg, "default_warning_half_length_m", 0.75), _cfg_float(cfg, "default_warning_half_width_m", 0.30)
    if cls == "barrier":
        return _cfg_float(cfg, "default_barrier_half_length_m", 1.00), _cfg_float(cfg, "default_barrier_half_width_m", 0.35)

    return _cfg_float(cfg, "default_vehicle_half_length_m", 2.25), _cfg_float(cfg, "default_vehicle_half_width_m", 1.00)


def relative_position_sector(x: float, y: float) -> str:
    if x >= 0.0:
        if abs(y) <= 2.2:
            return "front"
        return "front_left" if y < 0.0 else "front_right"
    if abs(y) <= 2.2:
        return "rear"
    return "rear_left" if y < 0.0 else "rear_right"


def actor_record_from_box(
    box: Dict,
    cfg,
    ego0_inv: Optional[np.ndarray] = None,
    source_ego_matrix: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    cls = normalize_actor_class(box)
    if cls == "ego" or (cls not in dynamic_actor_classes(cfg) and cls not in static_obstacle_classes(cfg)):
        return None

    mat = box_matrix(box)
    if mat is not None and ego0_inv is not None:
        rel = ego0_inv @ mat
        center = rel[:3, 3]
        xy = np.asarray([center[0], center[1]], dtype=np.float32)
        yaw = yaw_from_matrix_xy(rel)
    else:
        rel = _relative_matrix_from_local_box(box, source_ego_matrix, ego0_inv)
        if rel is not None:
            center = rel[:3, 3]
            xy = np.asarray([center[0], center[1]], dtype=np.float32)
            yaw = yaw_from_matrix_xy(rel)
        else:
            xy = _read_xy_from_position(box)
            if xy is None:
                return None
            yaw = _read_yaw(box)

    # ``distance`` in a saved future box is measured in that future ego frame.
    # Actor records returned here are expressed in the CURRENT ego frame, so the
    # distance must be recomputed from the transformed coordinates.
    dist = float(np.linalg.norm(xy))
    half_l, half_w = _read_half_extents(box, cls, cfg)
    actor_id = box.get("id", box.get("track_id", None))
    try:
        actor_id = int(actor_id) if actor_id is not None else None
    except Exception:
        actor_id = str(actor_id)

    return {
        "id": actor_id,
        "class": cls,
        "raw_class": str(box.get("class", "")),
        "x_m": float(xy[0]),
        "y_m": float(xy[1]),
        "yaw_rad": float(yaw),
        "distance_m": float(dist),
        "speed_mps": float(_read_speed(box)),
        "half_length_m": float(half_l),
        "half_width_m": float(half_w),
        "relative_position": relative_position_sector(float(xy[0]), float(xy[1])),
    }


def load_current_actor_records(route_dir: Path, frame_name: str, cfg) -> List[Dict]:
    boxes = load_boxes(route_dir, frame_name, cfg)
    ego0 = ego_matrix_from_boxes(boxes)
    ego0_inv = np.linalg.inv(ego0) if ego0 is not None else None
    records = []
    for b in boxes:
        r = actor_record_from_box(b, cfg, ego0_inv=ego0_inv, source_ego_matrix=ego0)
        if r is None:
            continue
        if float(r["distance_m"]) <= float(cfg.actors.max_actor_distance_m):
            records.append(r)
    records.sort(key=lambda r: float(r["distance_m"]))
    return records


def load_future_actor_timelines(route_dir: Path, frame_name: str, cfg) -> Dict[int, List[Dict]]:
    """Return actor records in the CURRENT ego frame for each future waypoint index k=1..N."""
    current_boxes = load_boxes(route_dir, frame_name, cfg)
    ego0 = ego_matrix_from_boxes(current_boxes)
    ego0_inv = np.linalg.inv(ego0) if ego0 is not None else None
    out: Dict[int, List[Dict]] = {}
    for k in range(1, int(cfg.horizon.num_future_waypoints) + 1):
        name = offset_frame_name(frame_name, k, int(cfg.horizon.future_frame_stride))
        if name is None:
            out[k] = []
            continue
        boxes = load_boxes(route_dir, name, cfg)
        source_ego = ego_matrix_from_boxes(boxes)
        records = []
        for b in boxes:
            r = actor_record_from_box(
                b,
                cfg,
                ego0_inv=ego0_inv,
                source_ego_matrix=source_ego,
            )
            if r is None:
                continue
            if float(r["distance_m"]) <= float(cfg.actors.max_actor_distance_m):
                records.append(r)
        out[k] = records
    return out
