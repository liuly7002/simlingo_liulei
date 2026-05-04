import os
import gzip
import json
import glob
import random
from collections import Counter, defaultdict

"""
第四轮验证脚本
该脚本可以验证以下问题:
1. VQA 风险判断是否能在 expert measurements 中找到外部行为响应；
2. expert response 是否出现在当前帧前后时间窗口内；
3. expert response 是否可以分解为 control slowdown、constraint response、lateral avoidance;
4. expert response 是否提供 object-matched、vehicle-context、infrastructure-context、generic ego response 等不同支持等级；
5. high risk / risk_reduced / partially_reduced / future increasing 是否具有风险相关 expert support;
6. 外部行为不一致样本是否可以分 hard / soft / candidate 导出复查。

备注:
该脚本主要验证 3 类外部行为一致性规则:
1. high-risk QA 是否具有风险相关 expert support；
2. risk_reduced / partially_reduced 反事实 QA 是否具有风险相关 expert support；
3. future_increasing QA 是否在 expert 行为中获得对象级、车辆上下文或通用 ego response 支持。

其中 infrastructure-only support 会被单独标记为 soft issue，
因为 stop sign / traffic light 等基础设施响应不能直接作为车辆风险 QA 的对象级证据。

"""

VQA_ROOT = "/root/database/simlingo_v2_2026_02_28/drivelm"
vqa_files = glob.glob(os.path.join(VQA_ROOT, "**", "vqa", "*.json.gz"), recursive=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 检查当前帧前后各 WINDOW_RADIUS 帧，用于缓解 expert 控制延迟或提前响应问题。
WINDOW_RADIUS = 2

def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def flatten_qas(data):
    all_items = []
    for qa_type in ["perception", "prediction", "planning", "behavior"]:
        all_items.extend(data.get("QA", {}).get(qa_type, []))
    return all_items

def get_measurement_path_from_vqa(data):
    img_path = data["image_paths"]["CAM_FRONT"]
    meas_path = img_path.replace("/rgb/", "/measurements/").replace(".jpg", ".json.gz")

    if meas_path.startswith("database/"):
        meas_path = "/root/" + meas_path

    return meas_path

def safe_get(d, key, default=None):
    return d[key] if key in d else default

def to_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def ids_equal(a, b):
    if a is None or b is None:
        return False
    return str(a) == str(b)


def get_measurement_window_paths(center_meas_path, radius=2):
    """
    根据当前 measurement 文件路径，构造前后若干帧的 measurement 路径。
    例如 0032.json.gz -> 0030.json.gz, 0031.json.gz, 0032.json.gz, 0033.json.gz, 0034.json.gz
    """
    folder = os.path.dirname(center_meas_path)
    filename = os.path.basename(center_meas_path)

    # 兼容 0032.json.gz
    if filename.endswith(".json.gz"):
        stem = filename.replace(".json.gz", "")
        suffix = ".json.gz"
    else:
        return [center_meas_path]

    try:
        frame_id = int(stem)
    except Exception:
        return [center_meas_path]

    width = len(stem)
    paths = []
    for offset in range(-radius, radius + 1):
        new_id = frame_id + offset
        if new_id < 0:
            continue
        new_name = str(new_id).zfill(width) + suffix
        new_path = os.path.join(folder, new_name)
        if os.path.exists(new_path):
            paths.append(new_path)

    return paths


def load_measurement_window(center_meas_path, radius=2):
    """
    读取当前帧前后窗口内可用的 measurements。
    """
    meas_list = []
    for p in get_measurement_window_paths(center_meas_path, radius=radius):
        try:
            meas_list.append({
                "path": p,
                "data": load_json_gz(p)
            })
        except Exception:
            continue
    return meas_list


def get_support_level(info):
    """
    将 expert 行为响应划分为不同外部支持等级。

    object_matched:
        expert 明确因为当前 QA object_id 对应的对象产生响应，最强证据。

    vehicle_context:
        expert 存在车辆相关风险响应，但没有精确匹配当前 object_id。

    infrastructure_context:
        expert 响应主要来自 stop sign / traffic light 等基础设施，
        不能直接作为车辆 Risk QA 的对象级证据。

    generic_ego_response:
        ego 有一般控制响应，例如刹车、目标速度降低、横向避让，
        但没有明确对象来源。

    no_response:
        没有可检测到的 expert 行为响应。
    """
    if info["object_matched_response"]:
        return "object_matched"

    if info["vehicle_context_response"]:
        return "vehicle_context"

    if info["infrastructure_response"]:
        return "infrastructure_context"

    if info["control_slowdown_response"] or info["lateral_response"]:
        return "generic_ego_response"

    return "no_response"


def get_expert_response_info(meas, object_id=None):
    """
    从单帧 measurement 中提取 expert 行为响应信息。

    论文版外部行为验证中，不再只判断 expert_response=True/False，
    而是进一步区分:
    1. 控制层响应 control_slowdown_response
    2. 约束层响应 constraint_response
    3. 对象级匹配响应 object_matched_response
    4. 车辆上下文响应 vehicle_context_response
    5. 基础设施响应 infrastructure_response
    6. 横向避让响应 lateral_response
    7. 总体行为支持等级 support_level
    """
    speed = to_float(safe_get(meas, "speed", 0.0), 0.0)
    target_speed = to_float(safe_get(meas, "target_speed", speed), speed)
    brake = bool(safe_get(meas, "control_brake", safe_get(meas, "brake", False)))
    throttle = to_float(safe_get(meas, "throttle", 1.0), 1.0)

    steer = to_float(
        safe_get(meas, "steer", safe_get(meas, "control_steer", 0.0)),
        0.0
    )

    changed_route = bool(safe_get(meas, "changed_route", False))

    speed_reduced_obj = safe_get(meas, "speed_reduced_by_obj_id", None)
    speed_reduced_obj_type = safe_get(meas, "speed_reduced_by_obj_type", None)
    speed_reduced_obj_distance = safe_get(meas, "speed_reduced_by_obj_distance", None)

    vehicle_hazard = bool(safe_get(meas, "vehicle_hazard", False))
    vehicle_affecting_id = safe_get(meas, "vehicle_affecting_id", None)

    walker_hazard = bool(safe_get(meas, "walker_hazard", False))
    walker_affecting_id = safe_get(meas, "walker_affecting_id", None)

    stop_sign_hazard = bool(safe_get(meas, "stop_sign_hazard", False))
    stop_sign_close = bool(safe_get(meas, "stop_sign_close", False))

    light_hazard = bool(safe_get(meas, "light_hazard", False))

    # 1. 控制层响应：真正体现当前控制动作或速度目标变化
    target_speed_reduced = target_speed < speed - 0.5
    throttle_suppressed = throttle < 0.2

    control_slowdown_response = (
        brake
        or target_speed_reduced
        or throttle_suppressed
    )

    # 2. 约束层响应：planner/controller 记录了某个对象或交通元素导致速度受限
    has_speed_reduced_obj = speed_reduced_obj is not None
    constraint_response = has_speed_reduced_obj

    # 3. 对象级匹配响应：最强外部证据
    object_matched_response = (
        ids_equal(speed_reduced_obj, object_id)
        or ids_equal(vehicle_affecting_id, object_id)
        or ids_equal(walker_affecting_id, object_id)
    )

    # 4. 基础设施响应：stop sign / traffic light 导致的响应，不能直接支持车辆 Risk QA
    infrastructure_response = (
        speed_reduced_obj_type in {"traffic.stop", "traffic.traffic_light"}
        or stop_sign_hazard
        or stop_sign_close
        or light_hazard
    )

    # 5. 车辆上下文响应：存在车辆风险上下文，但没有精确匹配当前 object_id
    vehicle_type_response = (
        speed_reduced_obj_type is not None
        and not str(speed_reduced_obj_type).startswith("traffic.")
        and not infrastructure_response
    )

    vehicle_context_response = (
        vehicle_hazard
        or vehicle_affecting_id is not None
        or vehicle_type_response
    )

    # 6. 行人上下文响应
    walker_context_response = (
        walker_hazard
        or walker_affecting_id is not None
    )

    # 7. 横向避让响应
    lateral_response = (
        abs(steer) > 0.15
        or changed_route
    )

    # 8. 总体 frame-level 行为响应
    slowdown_response = control_slowdown_response or constraint_response
    expert_response = slowdown_response or lateral_response

    reasons = []
    if brake:
        reasons.append("brake")
    if target_speed_reduced:
        reasons.append("target_speed_reduced")
    if throttle_suppressed:
        reasons.append("throttle_suppressed")
    if constraint_response:
        reasons.append("speed_constraint")
    if object_matched_response:
        reasons.append("object_matched_response")
    if vehicle_context_response:
        reasons.append("vehicle_context_response")
    if walker_context_response:
        reasons.append("walker_context_response")
    if infrastructure_response:
        reasons.append("infrastructure_response")
    if abs(steer) > 0.15:
        reasons.append("steering_response")
    if changed_route:
        reasons.append("changed_route")

    info = {
        "speed": speed,
        "target_speed": target_speed,
        "brake": brake,
        "throttle": throttle,
        "steer": steer,
        "changed_route": changed_route,

        "speed_reduced_by_obj_id": speed_reduced_obj,
        "speed_reduced_by_obj_type": speed_reduced_obj_type,
        "speed_reduced_by_obj_distance": speed_reduced_obj_distance,

        "vehicle_hazard": vehicle_hazard,
        "vehicle_affecting_id": vehicle_affecting_id,
        "walker_hazard": walker_hazard,
        "walker_affecting_id": walker_affecting_id,
        "stop_sign_hazard": stop_sign_hazard,
        "stop_sign_close": stop_sign_close,
        "light_hazard": light_hazard,

        "control_slowdown_response": control_slowdown_response,
        "constraint_response": constraint_response,
        "slowdown_response": slowdown_response,
        "lateral_response": lateral_response,
        "expert_response": expert_response,

        "object_matched_response": object_matched_response,
        "vehicle_context_response": vehicle_context_response,
        "walker_context_response": walker_context_response,
        "infrastructure_response": infrastructure_response,

        "response_reasons": reasons,
    }

    info["support_level"] = get_support_level(info)

    # Expert Response Score：用于论文中的辅助连续指标
    response_score = 0.0
    if control_slowdown_response:
        response_score += 1.0
    if constraint_response:
        response_score += 0.7
    if lateral_response:
        response_score += 0.5
    if object_matched_response:
        response_score += 1.0
    elif vehicle_context_response:
        response_score += 0.6
    elif infrastructure_response:
        response_score += 0.2

    info["response_score"] = response_score

    return info


def aggregate_expert_response(meas_window, object_id=None):
    """
    聚合时间窗口内的 expert 行为响应，并保留最强支持等级。
    """
    infos = [
        get_expert_response_info(x["data"], object_id=object_id)
        for x in meas_window
    ]

    if not infos:
        return {
            "window_size": 0,
            "expert_response": False,
            "slowdown_response": False,
            "control_slowdown_response": False,
            "constraint_response": False,
            "lateral_response": False,
            "object_matched_response": False,
            "vehicle_context_response": False,
            "walker_context_response": False,
            "infrastructure_response": False,
            "max_response_score": 0.0,
            "support_level": "no_response",
            "response_reasons": [],
        }

    support_rank = {
        "object_matched": 5,
        "vehicle_context": 4,
        "generic_ego_response": 3,
        "infrastructure_context": 2,
        "no_response": 1,
    }

    best_info = max(
        infos,
        key=lambda x: support_rank.get(x["support_level"], 0)
    )

    expert_response = any(x["expert_response"] for x in infos)
    slowdown_response = any(x["slowdown_response"] for x in infos)
    control_slowdown_response = any(x["control_slowdown_response"] for x in infos)
    constraint_response = any(x["constraint_response"] for x in infos)
    lateral_response = any(x["lateral_response"] for x in infos)

    object_matched_response = any(x["object_matched_response"] for x in infos)
    vehicle_context_response = any(x["vehicle_context_response"] for x in infos)
    walker_context_response = any(x["walker_context_response"] for x in infos)
    infrastructure_response = any(x["infrastructure_response"] for x in infos)

    max_response_score = max(x["response_score"] for x in infos)

    reasons = sorted(set(r for x in infos for r in x["response_reasons"]))

    return {
        "window_size": len(infos),

        "expert_response": expert_response,
        "slowdown_response": slowdown_response,
        "control_slowdown_response": control_slowdown_response,
        "constraint_response": constraint_response,
        "lateral_response": lateral_response,

        "object_matched_response": object_matched_response,
        "vehicle_context_response": vehicle_context_response,
        "walker_context_response": walker_context_response,
        "infrastructure_response": infrastructure_response,

        "max_response_score": max_response_score,
        "support_level": best_info["support_level"],
        "response_reasons": reasons,
    }


def safe_rate(num, den):
    return None if den == 0 else num / den

def get_risk_items_by_object(all_qas):
    obj2layers = defaultdict(dict)
    for qa in all_qas:
        if qa.get("cluster") == 4 and qa.get("layer") in [4, 5, 6, 7]:
            obj_id = qa.get("object_id")
            obj_tags = tuple(qa.get("object_tags", []))
            key = (obj_id, obj_tags)
            obj2layers[key][qa["layer"]] = qa
    return obj2layers

counter = Counter()
issues = []

hard_issue_types = {
    "high_risk_strong_hint_without_risk_relevant_support",
    "risk_reduced_counterfactual_without_risk_relevant_support",
}

soft_issue_types = {
    "high_risk_without_risk_relevant_support",
    "high_risk_with_infrastructure_only_support",

    "risk_reduced_counterfactual_with_infrastructure_only_support",
    "partially_reduced_counterfactual_without_risk_relevant_support",

    "future_increasing_strong_hint_without_risk_relevant_support",
    "future_increasing_large_closing_without_risk_relevant_support",
}

candidate_issue_types = {
    "future_increasing_soft_hint_without_risk_relevant_support",
    "missing_measurement_file",
}

for path in vqa_files:
    data = load_json_gz(path)
    meas_path = get_measurement_path_from_vqa(data)

    if not os.path.exists(meas_path):
        issues.append({
            "type": "missing_measurement_file",
            "severity": "candidate",
            "vqa_path": path,
            "measurement_path": meas_path,
        })
        continue

    meas = load_json_gz(meas_path)
    meas_window = load_measurement_window(meas_path, radius=WINDOW_RADIUS)

    all_qas = flatten_qas(data)
    obj2layers = get_risk_items_by_object(all_qas)

    for obj_key, layers in obj2layers.items():
        if 4 not in layers or 6 not in layers:
            continue

        qa1 = layers[4]
        qa3 = layers[6]
        qa4 = layers.get(7, None)

        m1 = qa1.get("qa_meta", {})
        m3 = qa3.get("qa_meta", {})
        m4 = qa4.get("qa_meta", {}) if qa4 else {}

        risk_level = m1.get("risk_level")
        cf_effect = m3.get("counterfactual_effect")
        future_trend = m4.get("future_risk_trend", None)

        object_id = obj_key[0]

        current_response_info = get_expert_response_info(meas, object_id=object_id)
        window_response_info = aggregate_expert_response(meas_window, object_id=object_id)

        expert_response = window_response_info["expert_response"]
        expert_slowdown = window_response_info["slowdown_response"]
        control_slowdown = window_response_info["control_slowdown_response"]
        constraint_response = window_response_info["constraint_response"]
        expert_lateral = window_response_info["lateral_response"]

        object_matched_response = window_response_info["object_matched_response"]
        vehicle_context_response = window_response_info["vehicle_context_response"]
        walker_context_response = window_response_info["walker_context_response"]
        infrastructure_response = window_response_info["infrastructure_response"]

        support_level = window_response_info["support_level"]
        max_response_score = window_response_info["max_response_score"]

        counter[f"support_{support_level}"] += 1

        key_prefix = f"risk_{risk_level}"
        counter[key_prefix] += 1
        counter[f"{key_prefix}_support_{support_level}"] += 1

        if expert_response:
            counter[f"{key_prefix}_window_response"] += 1
        if expert_slowdown:
            counter[f"{key_prefix}_window_slowdown"] += 1
        if control_slowdown:
            counter[f"{key_prefix}_control_slowdown"] += 1
        if constraint_response:
            counter[f"{key_prefix}_constraint_response"] += 1
        if expert_lateral:
            counter[f"{key_prefix}_window_lateral"] += 1
        if object_matched_response:
            counter[f"{key_prefix}_object_matched_response"] += 1
        if vehicle_context_response:
            counter[f"{key_prefix}_vehicle_context_response"] += 1
        if infrastructure_response:
            counter[f"{key_prefix}_infrastructure_response"] += 1

        if cf_effect:
            counter[f"cf_{cf_effect}"] += 1
            counter[f"cf_{cf_effect}_support_{support_level}"] += 1

            if expert_response:
                counter[f"cf_{cf_effect}_window_response"] += 1
            if expert_slowdown:
                counter[f"cf_{cf_effect}_window_slowdown"] += 1
            if control_slowdown:
                counter[f"cf_{cf_effect}_control_slowdown"] += 1
            if constraint_response:
                counter[f"cf_{cf_effect}_constraint_response"] += 1
            if object_matched_response:
                counter[f"cf_{cf_effect}_object_matched_response"] += 1
            if vehicle_context_response:
                counter[f"cf_{cf_effect}_vehicle_context_response"] += 1
            if infrastructure_response:
                counter[f"cf_{cf_effect}_infrastructure_response"] += 1

        if future_trend:
            counter[f"future_{future_trend}"] += 1
            counter[f"future_{future_trend}_support_{support_level}"] += 1

            if expert_response:
                counter[f"future_{future_trend}_window_response"] += 1
            if expert_slowdown:
                counter[f"future_{future_trend}_window_slowdown"] += 1
            if control_slowdown:
                counter[f"future_{future_trend}_control_slowdown"] += 1
            if constraint_response:
                counter[f"future_{future_trend}_constraint_response"] += 1
            if expert_lateral:
                counter[f"future_{future_trend}_window_lateral"] += 1
            if object_matched_response:
                counter[f"future_{future_trend}_object_matched_response"] += 1
            if vehicle_context_response:
                counter[f"future_{future_trend}_vehicle_context_response"] += 1
            if infrastructure_response:
                counter[f"future_{future_trend}_infrastructure_response"] += 1
        
        qa4_answer = qa4.get("A", "") if qa4 else ""
        future_delta = m4.get("future_distance_delta", None)

        strong_action_hint = (
            "prepared to slow down" in qa4_answer
            or "should slow down" in qa4_answer
            or "need to slow down" in qa4_answer
            or "should brake" in qa4_answer
            or "should pay close attention" in qa4_answer
        )

        large_future_closing = (
            future_delta is not None
            and future_trend == "increasing"
            and future_delta < -8.0
        )

        risk_relevant_support = support_level in {
            "object_matched",
            "vehicle_context",
            "generic_ego_response",
        }

        strong_risk_support = support_level in {
            "object_matched",
            "vehicle_context",
        }

        base_issue_info = {
            "vqa_path": path,
            "measurement_path": meas_path,
            "object": obj_key,
            "risk_level": risk_level,
            "counterfactual_effect": cf_effect,
            "future_trend": future_trend,
            "future_delta": future_delta,

            "current_response_info": current_response_info,
            "window_response_info": window_response_info,

            "support_level": support_level,
            "risk_relevant_support": risk_relevant_support,
            "strong_risk_support": strong_risk_support,

            "QA1_Q": qa1.get("Q"),
            "QA1_A": qa1.get("A"),
            "QA3_Q": qa3.get("Q"),
            "QA3_A": qa3.get("A"),
            "QA4_Q": qa4.get("Q") if qa4 else None,
            "QA4_A": qa4_answer,
        }

        no_behavior_support = support_level == "no_response"
        infrastructure_only_support = support_level == "infrastructure_context"

        if risk_level == "high" and not risk_relevant_support:
            if strong_action_hint and no_behavior_support:
                issue_type = "high_risk_strong_hint_without_risk_relevant_support"
                severity = "hard"
            elif infrastructure_only_support:
                issue_type = "high_risk_with_infrastructure_only_support"
                severity = "soft"
            else:
                issue_type = "high_risk_without_risk_relevant_support"
                severity = "soft"
            item = {
                "type": issue_type,
                "severity": severity,
            }
            item.update(base_issue_info)
            issues.append(item)

        # 规则2A：反事实为 risk_reduced，但时间窗口内没有 expert response
        if cf_effect == "risk_reduced" and not risk_relevant_support:
            if no_behavior_support:
                issue_type = "risk_reduced_counterfactual_without_risk_relevant_support"
                severity = "hard"
            elif infrastructure_only_support:
                issue_type = "risk_reduced_counterfactual_with_infrastructure_only_support"
                severity = "soft"
            else:
                issue_type = "risk_reduced_counterfactual_without_risk_relevant_support"
                severity = "soft"

            item = {
                "type": issue_type,
                "severity": severity,
            }
            item.update(base_issue_info)
            issues.append(item)


        # 规则2B：反事实为 partially_reduced，但没有风险相关 expert support
        if cf_effect == "partially_reduced" and not risk_relevant_support:
            item = {
                "type": "partially_reduced_counterfactual_without_risk_relevant_support",
                "severity": "soft",
            }
            item.update(base_issue_info)
            issues.append(item)

        # 规则3：未来风险上升，但没有风险相关 expert support
        if future_trend == "increasing" and not risk_relevant_support:
            if strong_action_hint:
                issue_type = "future_increasing_strong_hint_without_risk_relevant_support"
                severity = "soft"
            elif large_future_closing:
                issue_type = "future_increasing_large_closing_without_risk_relevant_support"
                severity = "soft"
            else:
                issue_type = "future_increasing_soft_hint_without_risk_relevant_support"
                severity = "candidate"

            item = {
                "type": issue_type,
                "severity": severity,
            }
            item.update(base_issue_info)
            issues.append(item)

issue_counter = Counter(x["type"] for x in issues)
severity_counter = Counter(x.get("severity", "unknown") for x in issues)

summary_lines = []

summary_lines.append("========== Action Consistency Summary ==========")
summary_lines.append("")

summary_lines.append("========== Counter ==========")
summary_lines.append(str(counter))
summary_lines.append("")

summary_lines.append("========== Issue counter ==========")
summary_lines.append(str(issue_counter))
summary_lines.append("")

summary_lines.append("========== Severity counter ==========")
summary_lines.append(str(severity_counter))
summary_lines.append("")

summary_lines.append("========== Support level distribution ==========")
support_levels = [
    "object_matched",
    "vehicle_context",
    "generic_ego_response",
    "infrastructure_context",
    "no_response",
]

total_risk_chains = sum(counter.get(f"support_{x}", 0) for x in support_levels)

for level_name in support_levels:
    total = counter.get(f"support_{level_name}", 0)
    summary_lines.append(f"support_{level_name}_total: {total}")
    summary_lines.append(
        f"support_{level_name}_rate: {safe_rate(total, total_risk_chains)}"
    )

summary_lines.append(f"total_support_count: {total_risk_chains}")

summary_lines.append("========== Response rates by risk level ==========")
for level in ["low", "medium", "high"]:
    base = counter.get(f"risk_{level}", 0)

    summary_lines.append(
        f"risk_{level}_window_response_rate: "
        f"{safe_rate(counter.get(f'risk_{level}_window_response', 0), base)}"
    )
    summary_lines.append(
        f"risk_{level}_object_matched_response_rate: "
        f"{safe_rate(counter.get(f'risk_{level}_object_matched_response', 0), base)}"
    )
    summary_lines.append(
        f"risk_{level}_vehicle_context_response_rate: "
        f"{safe_rate(counter.get(f'risk_{level}_vehicle_context_response', 0), base)}"
    )
    summary_lines.append(
        f"risk_{level}_infrastructure_response_rate: "
        f"{safe_rate(counter.get(f'risk_{level}_infrastructure_response', 0), base)}"
    )

    for support in support_levels:
        summary_lines.append(
            f"risk_{level}_support_{support}_rate: "
            f"{safe_rate(counter.get(f'risk_{level}_support_{support}', 0), base)}"
        )
summary_lines.append("")

summary_lines.append("========== Response rates by counterfactual effect ==========")
for effect in ["not_necessary", "limited_effect", "partially_reduced", "risk_reduced", "uncertain"]:
    base = counter.get(f"cf_{effect}", 0)

    summary_lines.append(
        f"cf_{effect}_window_response_rate: "
        f"{safe_rate(counter.get(f'cf_{effect}_window_response', 0), base)}"
    )
    summary_lines.append(
        f"cf_{effect}_object_matched_response_rate: "
        f"{safe_rate(counter.get(f'cf_{effect}_object_matched_response', 0), base)}"
    )
    summary_lines.append(
        f"cf_{effect}_vehicle_context_response_rate: "
        f"{safe_rate(counter.get(f'cf_{effect}_vehicle_context_response', 0), base)}"
    )
    summary_lines.append(
        f"cf_{effect}_infrastructure_response_rate: "
        f"{safe_rate(counter.get(f'cf_{effect}_infrastructure_response', 0), base)}"
    )

    for support in support_levels:
        summary_lines.append(
            f"cf_{effect}_support_{support}_rate: "
            f"{safe_rate(counter.get(f'cf_{effect}_support_{support}', 0), base)}"
        )
summary_lines.append("")

summary_lines.append("========== Response rates by future trend ==========")
for trend in ["increasing", "stable", "decreasing"]:
    base = counter.get(f"future_{trend}", 0)

    summary_lines.append(
        f"future_{trend}_window_response_rate: "
        f"{safe_rate(counter.get(f'future_{trend}_window_response', 0), base)}"
    )
    summary_lines.append(
        f"future_{trend}_window_slowdown_rate: "
        f"{safe_rate(counter.get(f'future_{trend}_window_slowdown', 0), base)}"
    )
    summary_lines.append(
        f"future_{trend}_window_lateral_rate: "
        f"{safe_rate(counter.get(f'future_{trend}_window_lateral', 0), base)}"
    )
    summary_lines.append(
        f"future_{trend}_object_matched_response_rate: "
        f"{safe_rate(counter.get(f'future_{trend}_object_matched_response', 0), base)}"
    )
    summary_lines.append(
        f"future_{trend}_vehicle_context_response_rate: "
        f"{safe_rate(counter.get(f'future_{trend}_vehicle_context_response', 0), base)}"
    )
    summary_lines.append(
        f"future_{trend}_infrastructure_response_rate: "
        f"{safe_rate(counter.get(f'future_{trend}_infrastructure_response', 0), base)}"
    )

    for support in support_levels:
        summary_lines.append(
            f"future_{trend}_support_{support}_rate: "
            f"{safe_rate(counter.get(f'future_{trend}_support_{support}', 0), base)}"
        )
summary_lines.append("")

known_issue_types = hard_issue_types | soft_issue_types | candidate_issue_types
unknown_issue_types = set(issue_counter.keys()) - known_issue_types

summary_lines.append("========== Issue type sanity check ==========")
summary_lines.append(f"known_issue_types: {sorted(list(known_issue_types))}")
summary_lines.append(f"unknown_issue_types: {sorted(list(unknown_issue_types))}")
summary_lines.append("")

print("\n".join(summary_lines))

# ================== Save selected action-consistency issues to txt ==================

output_txt = f"{VQA_ROOT}/action_consistency_issues.txt"

target_types = hard_issue_types | soft_issue_types | candidate_issue_types

selected_issues = [x for x in issues if x["type"] in target_types]

max_per_type = 50

issues_by_type = {}
for item in selected_issues:
    issues_by_type.setdefault(item["type"], []).append(item)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
    f.write("\n\n")

    f.write("========== Selected Issue Counter ==========\n")
    f.write(f"Selected issues: {len(selected_issues)}\n")
    for k, v in Counter(x["type"] for x in selected_issues).items():
        f.write(f"{k}: {v}\n")

    f.write("\n\n")

    for issue_type, issue_list in issues_by_type.items():
        sampled_items = random.sample(issue_list, min(max_per_type, len(issue_list)))

        f.write(f"========== {issue_type} ==========\n")
        f.write(f"Total: {len(issue_list)}\n")
        f.write(f"Randomly saved {len(sampled_items)} samples\n\n")

        for idx, item in enumerate(sampled_items):
            f.write(f"\n---------- Sample {idx + 1} ----------\n")
            f.write(json.dumps(item, indent=2, ensure_ascii=False))
            f.write("\n")

print(f"Saved selected action-consistency issues to: {output_txt}")