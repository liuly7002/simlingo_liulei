import os
import gzip
import json
import glob
from collections import Counter, defaultdict
import random

"""
第三轮验证脚本
风险图 QA 语义一致性验证脚本
该脚本可以验证以下问题:
1. low / medium / high 风险等级是否与证据数量一致
2. high risk 是否有证据支撑
3. low risk 是否严格对应 not_necessary 的反事实结果
4. non-low risk 是否异常对应 not_necessary
5. 当前 low risk 与 future increasing 是否属于轻微、强提示或大幅接近等不同 soft warning
6. future_risk_trend 是否与 future_distance_delta 的符号一致
7. 问题样本是否可以输出到 txt 供人工抽查

备注:
该脚本主要验证 5 类语义一致性规则:
1. 低风险对象不应该有很多正向风险证据
2. 高风险对象不应该没有任何风险证据
3. 低风险对象的反事实结果应为 not_necessary
4. 非低风险对象不应出现 not_necessary
5. 低风险但未来风险上升属于 soft warning,并进一步按语言强度和距离变化细分
6. 未来风险趋势必须和距离变化方向一致
"""

VQA_ROOT = "/root/database/simlingo_v2_2026_02_28/drivelm"
vqa_files = glob.glob(os.path.join(VQA_ROOT, "**", "vqa", "*.json.gz"), recursive=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

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

hard_issue_types = {
    "high_risk_without_evidence",
    "future_increasing_but_delta_nonnegative",
    "future_decreasing_but_delta_nonpositive",

    # Based on the QA generation logic:
    # low risk should always map to not_necessary in counterfactual QA.
    "low_risk_with_unexpected_counterfactual_effect",

    # not_necessary should only appear for low-risk objects.
    "non_low_risk_but_counterfactual_not_necessary",
}

soft_issue_types = {
    "low_risk_with_many_evidence",

    # Low risk + future increasing is a soft warning in the generation script,
    # so we further separate it by language strength and distance closing.
    "low_risk_but_future_soft_increasing",
    "low_risk_but_future_strong_increasing",
    "low_risk_but_future_large_distance_closing",
    "low_risk_but_future_increasing",
}

chain_stats = Counter()

for path in vqa_files:
    data = load_vqa(path)
    all_qas = flatten_qas(data)
    obj2layers = get_risk_items_by_object(all_qas)

    for obj_key, layers in obj2layers.items():
        chain_stats["total_object_risk_chains"] += 1    # 所有对象级风险链数量

        if {4, 5, 6}.issubset(layers.keys()):
            chain_stats["complete_4_5_6_chains"] += 1   # 至少包含风险等级、证据、反事实的核心链数量

        if {4, 5, 6, 7}.issubset(layers.keys()):
            chain_stats["complete_4_5_6_7_chains"] += 1 # 包含未来验证的完整链数量

        if 4 not in layers or 5 not in layers or 6 not in layers:
            chain_stats["incomplete_core_chains"] += 1  # 缺少 layer 4/5/6 的链数量
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
                "severity": "soft",
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
                "severity": "hard",
                "path": path,
                "object": obj_key,
                "risk_level": risk_level,
                "num_evidence": num_evidence,
                "Q": qa1.get("Q"),
                "A": qa1.get("A"),
            })

        # 规则3A：根据生成脚本，low risk 应该对应 not_necessary
        if risk_level == "low" and counterfactual_effect != "not_necessary":
            issues.append({
                "type": "low_risk_with_unexpected_counterfactual_effect",
                "severity": "hard",
                "path": path,
                "object": obj_key,
                "risk_level": risk_level,
                "counterfactual_effect": counterfactual_effect,
                "Q": qa3.get("Q"),
                "A": qa3.get("A"),
            })
        # 规则3B：根据生成脚本，not_necessary 应该只对应 low risk
        if risk_level in {"medium", "high"} and counterfactual_effect == "not_necessary":
            issues.append({
                "type": "non_low_risk_but_counterfactual_not_necessary",
                "severity": "hard",
                "path": path,
                "object": obj_key,
                "risk_level": risk_level,
                "counterfactual_effect": counterfactual_effect,
                "Q": qa3.get("Q"),
                "A": qa3.get("A"),
            })

        # 规则4：low risk 但未来风险 increasing
        # 生成脚本中该情况本身被视为 warning，而不是 hard inconsistency。
        # 因此这里只做细粒度 soft issue 分类。
        if risk_level == "low" and future_trend == "increasing":
            qa4_answer = qa4.get("A", "") if qa4 else ""
            delta = m4.get("future_distance_delta", None)

            strong_future_hint = (
                "likely to increase" in qa4_answer
                or "prepared to slow down" in qa4_answer
                or "should pay close attention" in qa4_answer
            )

            slight_future_hint = (
                "may slightly increase" in qa4_answer
                or "not expected to directly cross" in qa4_answer
            )

            if strong_future_hint:
                issue_type = "low_risk_but_future_strong_increasing"
            elif delta is not None and delta < -8.0:
                issue_type = "low_risk_but_future_large_distance_closing"
            elif slight_future_hint:
                issue_type = "low_risk_but_future_soft_increasing"
            else:
                issue_type = "low_risk_but_future_increasing"

            issues.append({
                "type": issue_type,
                "severity": "soft",
                "path": path,
                "object": obj_key,
                "risk_level": risk_level,
                "future_trend": future_trend,
                "future_delta": delta,
                "Q": qa4.get("Q") if qa4 else None,
                "A": qa4_answer,
            })

        # 规则5：future_trend 和 distance_delta 符号不一致
        if qa4 is not None:
            delta = m4.get("future_distance_delta", None)
            if delta is not None:
                if future_trend == "increasing" and delta >= 0:
                    issues.append({
                        "type": "future_increasing_but_delta_nonnegative",
                        "severity": "hard",
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
                        "severity": "hard",
                        "path": path,
                        "object": obj_key,
                        "future_trend": future_trend,
                        "future_delta": delta,
                        "Q": qa4.get("Q"),
                        "A": qa4.get("A"),
                    })


issue_counter = Counter(x["type"] for x in issues)
severity_counter = Counter(x.get("severity", "unknown") for x in issues)

num_valid_chains = chain_stats.get("complete_4_5_6_chains", 0)
num_hard_issues = severity_counter.get("hard", 0)
num_soft_issues = severity_counter.get("soft", 0)

hard_consistency_score = None
overall_consistency_score = None
core_chain_coverage = None
full_chain_coverage = None

total_object_risk_chains = chain_stats.get("total_object_risk_chains", 0)

if num_valid_chains > 0:
    hard_consistency_score = 1.0 - num_hard_issues / num_valid_chains
    overall_consistency_score = 1.0 - (num_hard_issues + num_soft_issues) / num_valid_chains

if total_object_risk_chains > 0:
    core_chain_coverage = chain_stats.get("complete_4_5_6_chains", 0) / total_object_risk_chains
    full_chain_coverage = chain_stats.get("complete_4_5_6_7_chains", 0) / total_object_risk_chains


# ================== Build summary report ==================

summary_lines = []

summary_lines.append("========== Distribution ==========")
summary_lines.append(str(counter))
summary_lines.append("")

summary_lines.append("========== Chain stats ==========")
summary_lines.append(str(chain_stats))
summary_lines.append("")

summary_lines.append("========== Issue counter ==========")
summary_lines.append(str(issue_counter))
summary_lines.append("")

summary_lines.append("========== Severity counter ==========")
summary_lines.append(str(severity_counter))
summary_lines.append("")

summary_lines.append("========== Consistency scores ==========")
summary_lines.append(f"hard_consistency_score: {hard_consistency_score}")
summary_lines.append(f"overall_consistency_score: {overall_consistency_score}")
summary_lines.append("")

summary_lines.append("========== Chain coverage ==========")
summary_lines.append(f"core_chain_coverage: {core_chain_coverage}")
summary_lines.append(f"full_chain_coverage: {full_chain_coverage}")
summary_lines.append("")

summary_lines.append("========== Summary numbers ==========")
summary_lines.append(f"num_vqa_files: {len(vqa_files)}")
summary_lines.append(f"total_object_risk_chains: {chain_stats.get('total_object_risk_chains', 0)}")
summary_lines.append(f"complete_4_5_6_chains: {chain_stats.get('complete_4_5_6_chains', 0)}")
summary_lines.append(f"complete_4_5_6_7_chains: {chain_stats.get('complete_4_5_6_7_chains', 0)}")
summary_lines.append(f"incomplete_core_chains: {chain_stats.get('incomplete_core_chains', 0)}")
summary_lines.append(f"total_issues: {len(issues)}")
summary_lines.append(f"hard_issues: {num_hard_issues}")
summary_lines.append(f"soft_issues: {num_soft_issues}")
summary_lines.append("")

known_issue_types = hard_issue_types | soft_issue_types
unknown_issue_types = set(issue_counter.keys()) - known_issue_types
summary_lines.append("========== Issue type sanity check ==========")
summary_lines.append(f"known_issue_types: {sorted(list(known_issue_types))}")
summary_lines.append(f"unknown_issue_types: {sorted(list(unknown_issue_types))}")
summary_lines.append("")

# 仍然打印一次，方便运行时快速看到结果。
# 如果不想看终端输出，可以把下面两行注释掉。
print("\n".join(summary_lines))

# ================== Save selected issues to txt ==================

output_txt = f"{VQA_ROOT}/risk_consistency_issues.txt"

target_types = hard_issue_types | soft_issue_types

selected_issues = [x for x in issues if x["type"] in target_types]

max_per_type = 50

issues_by_type = {}
for item in selected_issues:
    issues_by_type.setdefault(item["type"], []).append(item)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write("========== Risk Consistency Summary ==========\n\n")
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

print(f"Saved selected consistency issues to: {output_txt}")