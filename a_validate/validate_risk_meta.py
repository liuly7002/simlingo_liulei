import os
import gzip
import json
import glob
from collections import Counter, defaultdict

"""
风险推理链 QA 元信息完整性验证脚本
该脚本可以验证以下问题:
1. 新增 Risk Graph QA 是否都带有 qa_meta
2. qa_meta 中是否有 risk_chain_stage
3. 不同阶段是否包含对应的必要字段
4. risk_score / evidence_text_list / future_distance_delta 的类型是否正确
5. 风险等级分布是否合理
6. 反事实结果分布是否合理
7. 未来趋势分布是否合理
"""


# 修改1：更新VQA_ROOT路径以匹配新的数据位置
VQA_ROOT = "/root/simlingo/database/simlingo_v2_2026_02_28/drivelm"
vqa_files = glob.glob(os.path.join(VQA_ROOT, "**", "vqa", "*.json.gz"), recursive=True)


required_common_keys = [
    "risk_level",     # 当前风险等级文本，例如 low / medium / high
    "risk_level_id",  # 当前风险等级对应的数值 ID
    "risk_score",     # 当前风险分数
    "distance",       # 当前目标对象与自车的距离
]


required_by_stage = {
    "risk_level_estimation": [
        "risk_level",
        "risk_level_id",
        "risk_score",
    ],
    "evidence_attribution": [
        "num_positive_evidence",
        "evidence_text_list",
    ],
    "counterfactual_intervention": [
        "counterfactual_action",
        "counterfactual_effect",
    ],
    "future_temporal_verification": [
        "future_risk_trend",
        "future_current_distance",
        "future_distance",
        "future_distance_delta",
    ],
}

def load_vqa(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def flatten_qas(data):
    all_items = []
    for qa_type in ["perception", "prediction", "planning", "behavior"]:
        all_items.extend(data.get("QA", {}).get(qa_type, []))
    return all_items

stage_counter = Counter()
risk_level_counter = Counter()
counterfactual_counter = Counter()
future_trend_counter = Counter()
bad_meta_missing_stage = []
bad_meta_missing_keys = []
bad_meta_type = []

for path in vqa_files:
    data = load_vqa(path)
    all_qas = flatten_qas(data)

    for qa in all_qas:
        if qa.get("cluster") == 4 and qa.get("layer") in [4, 5, 6, 7]:
            meta = qa.get("qa_meta", {})

            stage = meta.get("risk_chain_stage", None)
            if stage is None:
                bad_meta_missing_stage.append({
                    "path": path,
                    "layer": qa.get("layer"),
                    "Q": qa.get("Q"),
                })
                continue

            stage_counter[stage] += 1

            for k in required_common_keys:
                if k not in meta:
                    bad_meta_missing_keys.append({
                        "path": path,
                        "layer": qa.get("layer"),
                        "stage": stage,
                        "missing_key": k,
                        "Q": qa.get("Q"),
                    })

            for k in required_by_stage.get(stage, []):
                if k not in meta:
                    bad_meta_missing_keys.append({
                        "path": path,
                        "layer": qa.get("layer"),
                        "stage": stage,
                        "missing_key": k,
                        "Q": qa.get("Q"),
                    })

            # 统计分布
            if "risk_level" in meta:
                risk_level_counter[meta["risk_level"]] += 1

            if "counterfactual_effect" in meta:
                counterfactual_counter[meta["counterfactual_effect"]] += 1

            if "future_risk_trend" in meta:
                future_trend_counter[meta["future_risk_trend"]] += 1

            # 类型检查
            if "risk_score" in meta and not isinstance(meta["risk_score"], (int, float)):
                bad_meta_type.append((path, qa.get("layer"), "risk_score", meta["risk_score"]))

            if "evidence_text_list" in meta and not isinstance(meta["evidence_text_list"], list):
                bad_meta_type.append((path, qa.get("layer"), "evidence_text_list", meta["evidence_text_list"]))

            if "future_distance_delta" in meta and not isinstance(meta["future_distance_delta"], (int, float)):
                bad_meta_type.append((path, qa.get("layer"), "future_distance_delta", meta["future_distance_delta"]))

print("========== Stage counter ==========")
print(stage_counter)

print("\n========== Risk level counter ==========")
print(risk_level_counter)

print("\n========== Counterfactual effect counter ==========")
print(counterfactual_counter)

print("\n========== Future trend counter ==========")
print(future_trend_counter)

print("\n========== Missing stage ==========")
print("bad_meta_missing_stage:", len(bad_meta_missing_stage))
for x in bad_meta_missing_stage[:10]:
    print(x)

print("\n========== Missing keys ==========")
print("bad_meta_missing_keys:", len(bad_meta_missing_keys))
for x in bad_meta_missing_keys[:10]:
    print(x)

print("\n========== Bad meta type ==========")
print("bad_meta_type:", len(bad_meta_type))
for x in bad_meta_type[:10]:
    print(x)