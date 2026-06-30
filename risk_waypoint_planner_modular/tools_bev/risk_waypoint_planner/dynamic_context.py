# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional, Any
import math

import numpy as np

from .io_utils import load_json_gz
from .qa_annotation import safe_float


def _get_boxes_folder(args) -> str:
    """
    Use existing boxes_folder args if they exist. Otherwise fallback to 'boxes'.
    This avoids adding new argparse options and keeps compatibility.
    """
    if bool(getattr(args, "augmented", False)):
        folder = getattr(args, "boxes_folder_augmented", None)
        if folder:
            return str(folder)

    return str(getattr(args, "boxes_folder", "boxes"))


def _load_current_boxes(route_dir: Path, frame_name: str, args) -> List[Dict]:
    boxes_folder = _get_boxes_folder(args)
    boxes_path = route_dir / boxes_folder / f"{frame_name}.json.gz"

    if not boxes_path.exists():
        return []

    obj = load_json_gz(boxes_path)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "boxes" in obj and isinstance(obj["boxes"], list):
        return obj["boxes"]
    return []


def _actor_class(box: Dict) -> str:
    cls = str(box.get("class", "")).lower()
    if cls in ["car", "vehicle"]:
        return "vehicle"
    if cls in ["walker", "pedestrian"]:
        return "pedestrian"
    return cls


def _is_dynamic_actor(box: Dict) -> bool:
    cls = _actor_class(box)
    if cls not in ["vehicle", "pedestrian"]:
        return False

    # Exclude ego fields.
    raw_cls = str(box.get("class", "")).lower()
    if raw_cls in ["ego_car", "ego_info"]:
        return False

    return "position" in box


def _get_position_xy(box: Dict) -> Optional[np.ndarray]:
    pos = box.get("position", None)
    if pos is None or not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None

    x = safe_float(pos[0], default=float("nan"))
    y = safe_float(pos[1], default=float("nan"))
    if not np.isfinite(x) or not np.isfinite(y):
        return None

    return np.asarray([x, y], dtype=np.float32)


def _get_distance_m(box: Dict, pos_xy: np.ndarray) -> float:
    if "distance" in box:
        d = safe_float(box.get("distance"), default=float("nan"))
        if np.isfinite(d):
            return float(d)
    return float(np.linalg.norm(pos_xy[:2]))


def _get_speed_mps(box: Dict) -> float:
    for key in ["speed", "velocity", "speed_mps"]:
        if key in box:
            return max(0.0, safe_float(box.get(key), default=0.0))
    return 0.0


def _motion_state(speed_mps: float) -> str:
    if speed_mps < 0.3:
        return "stopped"
    if speed_mps < 2.0:
        return "slow"
    return "moving"


def _motion_state_zh(state: str) -> str:
    return {
        "stopped": "基本静止",
        "slow": "低速移动",
        "moving": "正在移动",
    }.get(state, "运动状态不明确")


def _motion_state_en(state: str) -> str:
    return {
        "stopped": "nearly stopped",
        "slow": "moving slowly",
        "moving": "moving",
    }.get(state, "with unclear motion")


def _relative_position(x: float, y: float) -> str:
    """
    Ego-local convention:
        x forward
        y right
    Therefore:
        y < 0 means left
        y > 0 means right
    """
    if x >= 0.0:
        if abs(y) <= 2.2:
            return "front"
        if y < -2.2:
            return "front_left"
        return "front_right"

    if abs(y) <= 2.2:
        return "rear"
    if y < -2.2:
        return "rear_left"
    return "rear_right"


def _relative_position_zh(pos: str) -> str:
    return {
        "front": "正前方",
        "front_left": "左前方",
        "front_right": "右前方",
        "rear": "后方",
        "rear_left": "左后方",
        "rear_right": "右后方",
    }.get(pos, "周围")


def _relative_position_en(pos: str) -> str:
    return {
        "front": "ahead",
        "front_left": "front-left",
        "front_right": "front-right",
        "rear": "behind",
        "rear_left": "rear-left",
        "rear_right": "rear-right",
    }.get(pos, "nearby")


def _class_zh(cls: str) -> str:
    if cls == "vehicle":
        return "车辆"
    if cls == "pedestrian":
        return "行人"
    return "交通参与者"


def _class_en(cls: str) -> str:
    if cls == "vehicle":
        return "vehicle"
    if cls == "pedestrian":
        return "pedestrian"
    return "traffic participant"


def _relative_motion(
    actor_cls: str,
    rel_pos: str,
    actor_speed: float,
    ego_speed: float,
) -> str:
    """
    Natural relative-motion state.
    This is approximate and only used for language, not for planning.
    """
    if actor_cls == "pedestrian":
        if rel_pos in ["front", "front_left", "front_right"]:
            return "pedestrian_near_path"
        return "pedestrian_nearby"

    if rel_pos == "front":
        if actor_speed < 0.3:
            return "stopped_ahead"
        if ego_speed - actor_speed > 1.0:
            return "ego_closing_in"
        if actor_speed - ego_speed > 1.0:
            return "moving_away"
        return "similar_speed_ahead"

    if rel_pos in ["front_left", "front_right"]:
        if actor_speed < 0.3:
            return "side_space_partly_occupied"
        return "nearby_parallel_or_crossing"

    return "nearby"


def _relative_motion_zh(motion: str) -> str:
    return {
        "pedestrian_near_path": "行人靠近车辆行驶路径",
        "pedestrian_nearby": "附近有行人",
        "stopped_ahead": "前方对象基本静止",
        "ego_closing_in": "ego 车辆正在接近它",
        "moving_away": "该对象正在逐渐远离",
        "similar_speed_ahead": "该对象与 ego 车辆速度接近",
        "side_space_partly_occupied": "侧方空间被部分占用",
        "nearby_parallel_or_crossing": "侧方存在动态交通参与者",
        "nearby": "附近存在交通参与者",
    }.get(motion, "附近存在交通参与者")


def _relative_motion_en(motion: str) -> str:
    return {
        "pedestrian_near_path": "a pedestrian is close to the ego path",
        "pedestrian_nearby": "there is a nearby pedestrian",
        "stopped_ahead": "the object ahead is nearly stopped",
        "ego_closing_in": "the ego vehicle is closing in",
        "moving_away": "the object is moving away",
        "similar_speed_ahead": "the object has a similar speed to the ego vehicle",
        "side_space_partly_occupied": "the side space is partly occupied",
        "nearby_parallel_or_crossing": "there is a moving traffic participant on the side",
        "nearby": "there is a nearby traffic participant",
    }.get(motion, "there is a nearby traffic participant")


def _actor_description_zh(record: Dict) -> str:
    cls_zh = _class_zh(record["class"])
    pos_zh = _relative_position_zh(record["relative_position"])
    motion_zh = _relative_motion_zh(record["relative_motion"])
    state_zh = _motion_state_zh(record["motion_state"])

    if record["class"] == "pedestrian":
        return f"{pos_zh}有行人，{motion_zh}。"

    if record["relative_position"] == "front":
        return f"{pos_zh}有{state_zh}的{cls_zh}，{motion_zh}。"

    return f"{pos_zh}有{state_zh}的{cls_zh}，这会影响对应方向的通行空间。"


def _actor_description_en(record: Dict) -> str:
    cls_en = _class_en(record["class"])
    pos_en = _relative_position_en(record["relative_position"])
    motion_en = _relative_motion_en(record["relative_motion"])
    state_en = _motion_state_en(record["motion_state"])

    if record["class"] == "pedestrian":
        return f"There is a pedestrian {pos_en}, and {motion_en}."

    if record["relative_position"] == "front":
        return f"There is a {state_en} {cls_en} {pos_en}, and {motion_en}."

    return f"There is a {state_en} {cls_en} at the {pos_en}, which affects the available space in that direction."


def _make_actor_record(box: Dict, ego_speed_mps: float) -> Optional[Dict]:
    pos_xy = _get_position_xy(box)
    if pos_xy is None:
        return None

    x = float(pos_xy[0])
    y = float(pos_xy[1])
    d = _get_distance_m(box, pos_xy)

    cls = _actor_class(box)
    speed = _get_speed_mps(box)
    rel_pos = _relative_position(x, y)
    state = _motion_state(speed)
    rel_motion = _relative_motion(cls, rel_pos, speed, ego_speed_mps)

    record = {
        "exists": True,
        "id": box.get("id", None),
        "class": cls,
        "class_zh": _class_zh(cls),
        "class_en": _class_en(cls),
        "x_m": round(float(x), 3),
        "y_m": round(float(y), 3),
        "distance_m": round(float(d), 3),
        "speed_mps": round(float(speed), 3),
        "ego_speed_mps": round(float(ego_speed_mps), 3),
        "relative_position": rel_pos,
        "relative_position_zh": _relative_position_zh(rel_pos),
        "relative_position_en": _relative_position_en(rel_pos),
        "motion_state": state,
        "motion_state_zh": _motion_state_zh(state),
        "motion_state_en": _motion_state_en(state),
        "relative_motion": rel_motion,
        "relative_motion_zh": _relative_motion_zh(rel_motion),
        "relative_motion_en": _relative_motion_en(rel_motion),
    }

    record["description_zh"] = _actor_description_zh(record)
    record["description_en"] = _actor_description_en(record)
    return record


def _choose_closest(records: List[Dict]) -> Dict:
    if not records:
        return {"exists": False}
    records = sorted(records, key=lambda r: safe_float(r.get("distance_m", 1e9), 1e9))
    return records[0]


def _summarize_dynamic_scene_zh(
    front: Dict,
    left: Dict,
    right: Dict,
    walker: Dict,
    front_left: Optional[Dict] = None,
    front_right: Optional[Dict] = None,) -> str:
    parts = []
    front_left = front_left or {"exists": False}
    front_right = front_right or {"exists": False}

    if front.get("exists", False):
        parts.append(front["description_zh"])

    if front_left.get("exists", False) and front_left.get("id", None) != front.get("id", None):
        parts.append("左前方存在动态交通参与者，向左绕行需要谨慎判断。")

    if front_right.get("exists", False) and front_right.get("id", None) != front.get("id", None):
        parts.append("右前方存在动态交通参与者，向右绕行需要谨慎判断。")

    if walker.get("exists", False):
        if walker.get("id", None) != front.get("id", None):
            parts.append(walker["description_zh"])

    if left.get("exists", False):
        parts.append("当前左侧存在动态交通参与者，左侧绕行空间需要谨慎判断。")
    else:
        parts.append("当前左侧没有识别到明显靠近的动态交通参与者。")

    if right.get("exists", False):
        parts.append("当前右侧存在动态交通参与者，右侧绕行空间需要谨慎判断。")
    else:
        parts.append("当前右侧没有识别到明显靠近的动态交通参与者。")

    if not parts:
        return "当前没有识别到明显影响车辆决策的动态交通参与者。"

    return " ".join(parts)


def _summarize_dynamic_scene_en(
    front: Dict,
    left: Dict,
    right: Dict,
    walker: Dict,
    front_left: Optional[Dict] = None,
    front_right: Optional[Dict] = None,) -> str:
    parts = []
    front_left = front_left or {"exists": False}
    front_right = front_right or {"exists": False}

    if front.get("exists", False):
        parts.append(front["description_en"])

    if front_left.get("exists", False) and front_left.get("id", None) != front.get("id", None):
        parts.append("There is a dynamic traffic participant at the front-left, so a left-side detour should be considered carefully.")

    if front_right.get("exists", False) and front_right.get("id", None) != front.get("id", None):
        parts.append("There is a dynamic traffic participant at the front-right, so a right-side detour should be considered carefully.")

    if walker.get("exists", False):
        if walker.get("id", None) != front.get("id", None):
            parts.append(walker["description_en"])

    if left.get("exists", False):
        parts.append("There is a dynamic traffic participant on the current left side, so a left-side detour should be considered carefully.")
    else:
        parts.append("No clearly close dynamic traffic participant is detected on the current left side.")

    if right.get("exists", False):
        parts.append("There is a dynamic traffic participant on the current right side, so a right-side detour should be considered carefully.")
    else:
        parts.append("No clearly close dynamic traffic participant is detected on the current right side.")

    if not parts:
        return "No dynamic traffic participant is clearly affecting the current decision."

    return " ".join(parts)


def build_key_dynamic_context(
    route_dir: Path,
    frame_name: str,
    measurement: Dict,
    args,) -> Dict:
    """
    Extract a small set of key dynamic actors for dreamer language.

    This module does NOT affect planning or scoring. It only provides
    human-readable scene context for VLA-style QA.
    """
    boxes = _load_current_boxes(route_dir, frame_name, args)
    ego_speed = max(0.0, safe_float(measurement.get("speed", 0.0), default=0.0))

    max_dist = float(getattr(args, "dynamic_context_max_distance_m", 35.0))
    front_width = float(getattr(args, "dynamic_context_front_width_m", 2.2))
    front_side_width = float(getattr(args, "dynamic_context_front_side_width_m", 6.0))
    front_side_inner = float(getattr(args, "dynamic_context_front_side_inner_m", 0.8))

    side_width = float(getattr(args, "dynamic_context_side_width_m", 6.0))
    side_forward = float(getattr(args, "dynamic_context_side_forward_m", 25.0))
    rear_margin = float(getattr(args, "dynamic_context_rear_margin_m", -3.0))

    # A true side actor should be clearly outside the front corridor.
    # A front-left/front-right actor is still ahead of ego, but laterally biased.
    side_inner = float(getattr(args, "dynamic_context_side_inner_m", front_width))

    actor_records = []
    for box in boxes:
        if not _is_dynamic_actor(box):
            continue

        rec = _make_actor_record(box, ego_speed_mps=ego_speed)
        if rec is None:
            continue

        x = safe_float(rec.get("x_m", 0.0))
        y = safe_float(rec.get("y_m", 0.0))
        d = safe_float(rec.get("distance_m", 1e9), 1e9)

        if d > max_dist:
            continue
        if x < rear_margin:
            continue

        actor_records.append(rec)

    front_center_candidates = [
        r for r in actor_records
        if r["x_m"] > 0.0 and abs(r["y_m"]) <= front_width
    ]

    front_left_candidates = [
        r for r in actor_records
        if (
            r["x_m"] > 0.0
            and -front_side_width <= r["y_m"] < -front_side_inner
        )
    ]

    front_right_candidates = [
        r for r in actor_records
        if (
            r["x_m"] > 0.0
            and front_side_inner < r["y_m"] <= front_side_width
        )
    ]

    # True side actors: clearly outside the front corridor.
    left_side_candidates = [
        r for r in actor_records
        if (
            rear_margin <= r["x_m"] <= side_forward
            and -side_width <= r["y_m"] < -side_inner
        )
    ]

    right_side_candidates = [
        r for r in actor_records
        if (
            rear_margin <= r["x_m"] <= side_forward
            and side_inner < r["y_m"] <= side_width
        )
    ]

    # Backward-compatible corridor names.
    left_candidates = left_side_candidates
    right_candidates = right_side_candidates

    # Any object ahead, including front-left/front-right.
    front_candidates = (
        front_center_candidates
        + front_left_candidates
        + front_right_candidates
    )

    walker_candidates = [
        r for r in actor_records
        if r["class"] == "pedestrian" and r["x_m"] > -1.0 and r["distance_m"] <= min(max_dist, 20.0)
    ]

    front = _choose_closest(front_candidates)
    front_center = _choose_closest(front_center_candidates)
    front_left = _choose_closest(front_left_candidates)
    front_right = _choose_closest(front_right_candidates)

    left_side = _choose_closest(left_side_candidates)
    right_side = _choose_closest(right_side_candidates)

    # Backward-compatible names used by older dreamer code.
    left = left_side
    right = right_side

    walker = _choose_closest(walker_candidates)

    has_key_actor = bool(
        front.get("exists", False)
        or front_center.get("exists", False)
        or front_left.get("exists", False)
        or front_right.get("exists", False)
        or left_side.get("exists", False)
        or right_side.get("exists", False)
        or walker.get("exists", False)
    )

    return {
        "enabled": True,
        "source": "boxes",
        "frame": frame_name,
        "ego_speed_mps": round(float(ego_speed), 3),
        "num_dynamic_actors_considered": int(len(actor_records)),
        "has_key_actor": bool(has_key_actor),

        "front_actor": front,
        "front_center_actor": front_center,
        "front_left_actor": front_left,
        "front_right_actor": front_right,

        "left_side_actor": left_side,
        "right_side_actor": right_side,

        # Backward-compatible names.
        "left_corridor_actor": left_side,
        "right_corridor_actor": right_side,

        "nearby_walker": walker,

        "summary_zh": _summarize_dynamic_scene_zh(
            front=front,
            left=left_side,
            right=right_side,
            walker=walker,
            front_left=front_left,
            front_right=front_right,
        ),
        "summary_en": _summarize_dynamic_scene_en(
            front=front,
            left=left_side,
            right=right_side,
            walker=walker,
            front_left=front_left,
            front_right=front_right,
        ),
    }