import os
import gzip
import json
import glob
from collections import Counter, defaultdict

VQA_ROOT = "/root/simlingo/database/simlingo_v2_2026_02_28/drivelm"
vqa_files = glob.glob(os.path.join(VQA_ROOT, "**", "vqa", "*.json.gz"), recursive=True)

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
    return img_path.replace("/rgb/", "/measurements/").replace(".jpg", ".json.gz")

def safe_get(d, key, default=None):
    return d[key] if key in d else default

def has_expert_slowdown(meas):
    speed = float(safe_get(meas, "speed", 0.0))
    target_speed = float(safe_get(meas, "target_speed", speed))
    brake = bool(safe_get(meas, "control_brake", False))
    throttle = float(safe_get(meas, "throttle", 1.0))
    speed_reduced_obj = safe_get(meas, "speed_reduced_by_obj_id", None)

    return (
        brake
        or target_speed < speed - 0.5
        or speed_reduced_obj is not None
        or throttle < 0.2
    )

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

for path in vqa_files:
    data = load_json_gz(path)
    meas_path = get_measurement_path_from_vqa(data)

    if not os.path.exists(meas_path):
        issues.append({
            "type": "missing_measurement_file",
            "vqa_path": path,
            "measurement_path": meas_path,
        })
        continue

    meas = load_json_gz(meas_path)
    expert_slowdown = has_expert_slowdown(meas)

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

        key_prefix = f"risk_{risk_level}"
        counter[key_prefix] += 1
        if expert_slowdown:
            counter[f"{key_prefix}_expert_slowdown"] += 1

        if cf_effect:
            counter[f"cf_{cf_effect}"] += 1
            if expert_slowdown:
                counter[f"cf_{cf_effect}_expert_slowdown"] += 1

        if future_trend:
            counter[f"future_{future_trend}"] += 1
            if expert_slowdown:
                counter[f"future_{future_trend}_expert_slowdown"] += 1

        # 高风险但 expert 完全没有减速迹象
        if risk_level == "high" and not expert_slowdown:
            issues.append({
                "type": "high_risk_without_expert_slowdown",
                "vqa_path": path,
                "measurement_path": meas_path,
                "object": obj_key,
                "speed": safe_get(meas, "speed"),
                "target_speed": safe_get(meas, "target_speed"),
                "brake": safe_get(meas, "control_brake"),
                "throttle": safe_get(meas, "throttle"),
                "speed_reduced_by_obj_id": safe_get(meas, "speed_reduced_by_obj_id"),
                "Q": qa1.get("Q"),
                "A": qa1.get("A"),
            })

        # 反事实说减速有效，但 expert 没有减速迹象
        if cf_effect == "risk_reduced" and not expert_slowdown:
            issues.append({
                "type": "risk_reduced_counterfactual_without_expert_slowdown",
                "vqa_path": path,
                "measurement_path": meas_path,
                "object": obj_key,
                "speed": safe_get(meas, "speed"),
                "target_speed": safe_get(meas, "target_speed"),
                "brake": safe_get(meas, "control_brake"),
                "throttle": safe_get(meas, "throttle"),
                "speed_reduced_by_obj_id": safe_get(meas, "speed_reduced_by_obj_id"),
                "Q": qa3.get("Q"),
                "A": qa3.get("A"),
            })

        # 未来风险上升但 expert 没有减速迹象
        # if future_trend == "increasing" and not expert_slowdown:
        #     issues.append({
        #         "type": "future_increasing_without_expert_slowdown",
        #         "vqa_path": path,
        #         "measurement_path": meas_path,
        #         "object": obj_key,
        #         "future_delta": m4.get("future_distance_delta"),
        #         "speed": safe_get(meas, "speed"),
        #         "target_speed": safe_get(meas, "target_speed"),
        #         "brake": safe_get(meas, "control_brake"),
        #         "throttle": safe_get(meas, "throttle"),
        #         "speed_reduced_by_obj_id": safe_get(meas, "speed_reduced_by_obj_id"),
        #         "Q": qa4.get("Q") if qa4 else None,
        #         "A": qa4.get("A") if qa4 else None,
        #     })
        if future_trend == "increasing" and not expert_slowdown:
            qa4_answer = qa4.get("A", "") if qa4 else ""

            strong_action_hint = (
                "prepared to slow down" in qa4_answer
                or "should slow down" in qa4_answer
                or "need to slow down" in qa4_answer
                or "should brake" in qa4_answer
            )

            issue_type = (
                "future_increasing_strong_hint_without_expert_slowdown"
                if strong_action_hint
                else "future_increasing_soft_hint_without_expert_slowdown"
            )

            issues.append({
                "type": issue_type,
                "vqa_path": path,
                "measurement_path": meas_path,
                "object": obj_key,
                "future_delta": m4.get("future_distance_delta"),
                "risk_level": risk_level,
                "counterfactual_effect": cf_effect,
                "speed": safe_get(meas, "speed"),
                "target_speed": safe_get(meas, "target_speed"),
                "brake": safe_get(meas, "control_brake"),
                "throttle": safe_get(meas, "throttle"),
                "speed_reduced_by_obj_id": safe_get(meas, "speed_reduced_by_obj_id"),
                "QA4_Q": qa4.get("Q") if qa4 else None,
                "QA4_A": qa4_answer,
            })

print("========== Counter ==========")
print(counter)

issue_counter = Counter(x["type"] for x in issues)
print("\n========== Issue counter ==========")
print(issue_counter)

num_future_increasing = counter.get("future_increasing", 0)
num_future_strong_conflict = issue_counter.get(
    "future_increasing_strong_hint_without_expert_slowdown", 0
)
num_future_soft_conflict = issue_counter.get(
    "future_increasing_soft_hint_without_expert_slowdown", 0
)

if num_future_increasing > 0:
    print("\n========== Future Increasing Breakdown ==========")
    print(
        "strong_hint_conflict_ratio:",
        num_future_strong_conflict / num_future_increasing
    )
    print(
        "soft_hint_conflict_ratio:",
        num_future_soft_conflict / num_future_increasing
    )

# ================== Save selected action-consistency issues to txt ==================

output_txt = "action_consistency_issues.txt"

# target_types = {
#     "future_increasing_without_expert_slowdown",
#     "risk_reduced_counterfactual_without_expert_slowdown",
#     "high_risk_without_expert_slowdown",
# }
target_types = {
    "future_increasing_strong_hint_without_expert_slowdown",
    "risk_reduced_counterfactual_without_expert_slowdown",
    "high_risk_without_expert_slowdown",
}

selected_issues = [x for x in issues if x["type"] in target_types]

max_per_type = 50

issues_by_type = {}
for item in selected_issues:
    issues_by_type.setdefault(item["type"], []).append(item)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write("========== Action Consistency Issues ==========\n\n")
    f.write(f"Total issues: {len(issues)}\n")
    f.write(f"Selected issues: {len(selected_issues)}\n\n")

    issue_counter = Counter(x["type"] for x in issues)
    f.write("========== Issue Counter ==========\n")
    for k, v in issue_counter.items():
        f.write(f"{k}: {v}\n")

    f.write("\n\n")

    for issue_type, issue_list in issues_by_type.items():
        f.write(f"========== {issue_type} ==========\n")
        f.write(f"Total: {len(issue_list)}\n")
        f.write(f"Saved first {min(max_per_type, len(issue_list))} samples\n\n")

        for idx, item in enumerate(issue_list[:max_per_type]):
            f.write(f"\n---------- Sample {idx + 1} ----------\n")
            f.write(json.dumps(item, indent=2, ensure_ascii=False))
            f.write("\n")

print(f"Saved selected action-consistency issues to: {output_txt}")