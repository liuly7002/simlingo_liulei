import os
import gzip
import json
import glob
from collections import Counter, defaultdict

VQA_ROOT = "/root/simlingo/database/simlingo_v2_2026_02_28/drivelm"
vqa_files = glob.glob(os.path.join(VQA_ROOT, "**", "vqa", "*.json.gz"), recursive=True)

def load_vqa(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def flatten_qas(data):
    all_items = []
    for qa_type in ["perception", "prediction", "planning", "behavior"]:
        all_items.extend(data.get("QA", {}).get(qa_type, []))
    return all_items

def get_risk_items_by_object(all_qas):
    obj2layers = defaultdict(dict)
    for qa in all_qas:
        if qa.get("cluster") == 4 and qa.get("layer") in [4, 5, 6, 7]:
            obj_id = qa.get("object_id")
            obj_tags = tuple(qa.get("object_tags", []))
            key = (obj_id, obj_tags)
            obj2layers[key][qa["layer"]] = qa
    return obj2layers

issues = []
counter = Counter()

for path in vqa_files:
    data = load_vqa(path)
    all_qas = flatten_qas(data)
    obj2layers = get_risk_items_by_object(all_qas)

    for obj_key, layers in obj2layers.items():
        if 4 not in layers or 5 not in layers or 6 not in layers:
            continue

        qa1 = layers[4]
        qa2 = layers[5]
        qa3 = layers[6]
        qa4 = layers.get(7, None)

        m1 = qa1.get("qa_meta", {})
        m2 = qa2.get("qa_meta", {})
        m3 = qa3.get("qa_meta", {})
        m4 = qa4.get("qa_meta", {}) if qa4 else {}

        risk_level = m1.get("risk_level")
        risk_score = m1.get("risk_score")
        evidence_list = m2.get("evidence_text_list", [])
        num_evidence = m2.get("num_positive_evidence", len(evidence_list))
        counterfactual_effect = m3.get("counterfactual_effect")
        future_trend = m4.get("future_risk_trend", None)

        counter[f"risk_{risk_level}"] += 1
        counter[f"cf_{counterfactual_effect}"] += 1
        if future_trend:
            counter[f"future_{future_trend}"] += 1

        # 规则1：low risk 但证据很多
        if risk_level == "low" and num_evidence >= 2:
            issues.append({
                "type": "low_risk_with_many_evidence",
                "path": path,
                "object": obj_key,
                "risk_level": risk_level,
                "num_evidence": num_evidence,
                "evidence": evidence_list,
                "Q": qa1.get("Q"),
                "A": qa1.get("A"),
            })

        # 规则2：high risk 但没有证据
        if risk_level == "high" and num_evidence == 0:
            issues.append({
                "type": "high_risk_without_evidence",
                "path": path,
                "object": obj_key,
                "risk_level": risk_level,
                "num_evidence": num_evidence,
                "Q": qa1.get("Q"),
                "A": qa1.get("A"),
            })

        # 规则3：low risk 但反事实说 risk_reduced
        if risk_level == "low" and counterfactual_effect == "risk_reduced":
            issues.append({
                "type": "low_risk_but_slowing_down_reduces_risk",
                "path": path,
                "object": obj_key,
                "risk_level": risk_level,
                "counterfactual_effect": counterfactual_effect,
                "Q": qa3.get("Q"),
                "A": qa3.get("A"),
            })

        # 规则4：low risk 但未来风险 increasing
        if risk_level == "low" and future_trend == "increasing":
            issues.append({
                "type": "low_risk_but_future_increasing",
                "path": path,
                "object": obj_key,
                "risk_level": risk_level,
                "future_trend": future_trend,
                "future_delta": m4.get("future_distance_delta"),
                "Q": qa4.get("Q") if qa4 else None,
                "A": qa4.get("A") if qa4 else None,
            })

        # 规则5：future_trend 和 distance_delta 符号不一致
        if qa4 is not None:
            delta = m4.get("future_distance_delta", None)
            if delta is not None:
                if future_trend == "increasing" and delta >= 0:
                    issues.append({
                        "type": "future_increasing_but_delta_nonnegative",
                        "path": path,
                        "object": obj_key,
                        "future_trend": future_trend,
                        "future_delta": delta,
                        "Q": qa4.get("Q"),
                        "A": qa4.get("A"),
                    })

                if future_trend == "decreasing" and delta <= 0:
                    issues.append({
                        "type": "future_decreasing_but_delta_nonpositive",
                        "path": path,
                        "object": obj_key,
                        "future_trend": future_trend,
                        "future_delta": delta,
                        "Q": qa4.get("Q"),
                        "A": qa4.get("A"),
                    })

print("========== Distribution ==========")
print(counter)

issue_counter = Counter(x["type"] for x in issues)
print("\n========== Issue counter ==========")
print(issue_counter)

# print("\n========== First 30 issues ==========")
# for x in issues[:30]:
#     print(json.dumps(x, indent=2, ensure_ascii=False))

# ================== Save selected issues to txt ==================

output_txt = "risk_consistency_issues.txt"

target_types = {
    "low_risk_with_many_evidence",
    "low_risk_but_future_increasing",
}

selected_issues = [x for x in issues if x["type"] in target_types]

max_per_type = 50

issues_by_type = {}
for item in selected_issues:
    issues_by_type.setdefault(item["type"], []).append(item)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write("========== Risk Consistency Issues ==========\n\n")
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

print(f"Saved selected consistency issues to: {output_txt}")