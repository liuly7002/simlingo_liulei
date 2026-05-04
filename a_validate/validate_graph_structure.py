import os
import gzip
import json
import glob
from collections import Counter, defaultdict

"""
风险图 QA 连接关系与链路完整性验证脚本
该脚本可以验证以下问题:
1. layer 4~7 的 con_up / con_down 是否符合预期图结构
2. 每个新增风险 QA 是否携带 qa_meta
3. 每个新增风险 QA 是否绑定 object_tags
4. 同一 object_id 的 layer 4、5、6 是否形成连续风险推理链
"""

# 修改1：更新VQA_ROOT路径以匹配新的数据位置
VQA_ROOT = "/root/simlingo/database/simlingo_v2_2026_02_28/drivelm"
vqa_files = glob.glob(os.path.join(VQA_ROOT, "**", "vqa", "*.json.gz"), recursive=True)

expected_edges = {
    4: {
        "con_up": [(4, 3)],
        "con_down": [(4, 0), (4, 1), (4, 2), (4, 3)],
    },
    5: {
        "con_up": [(4, 3), (4, 4)],
        "con_down": [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
    },
    6: {
        "con_up": [(4, 3), (4, 4), (4, 5)],
        "con_down": [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)],
    },
    7: {
        "con_up": [(4, 3), (4, 4), (4, 5), (4, 6)],
        "con_down": [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6)],
    },
}

def norm_edges(x):
    if x is None or x == -1:
        return x
    return sorted([tuple(e) for e in x])

def load_vqa(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def flatten_qas(data):
    all_items = []
    for qa_type in ["perception", "prediction", "planning", "behavior"]:
        all_items.extend(data.get("QA", {}).get(qa_type, []))
    return all_items

stats = Counter()
bad_edges = []
bad_missing_meta = []
bad_object_tags = []
risk_chain_by_object = defaultdict(lambda: defaultdict(set))

for path in vqa_files:
    data = load_vqa(path)
    all_qas = flatten_qas(data)

    for qa in all_qas:
        if qa.get("cluster") == 4 and qa.get("layer") in [4, 5, 6, 7]:
            layer = qa["layer"]
            stats[f"layer_{layer}_count"] += 1

            object_id = qa.get("object_id", None)
            object_tags = qa.get("object_tags", [])
            risk_chain_by_object[path][object_id].add(layer)

            # 1. 检查 Graph 边
            exp = expected_edges[layer]
            if norm_edges(qa.get("con_up")) != norm_edges(exp["con_up"]):
                bad_edges.append({
                    "path": path,
                    "layer": layer,
                    "field": "con_up",
                    "actual": qa.get("con_up"),
                    "expected": exp["con_up"],
                    "Q": qa.get("Q")
                })

            if norm_edges(qa.get("con_down")) != norm_edges(exp["con_down"]):
                bad_edges.append({
                    "path": path,
                    "layer": layer,
                    "field": "con_down",
                    "actual": qa.get("con_down"),
                    "expected": exp["con_down"],
                    "Q": qa.get("Q")
                })

            # 2. 检查 qa_meta
            if "qa_meta" not in qa:
                bad_missing_meta.append({
                    "path": path,
                    "layer": layer,
                    "Q": qa.get("Q")
                })

            # 3. 检查 object_tags
            if not object_tags:
                bad_object_tags.append({
                    "path": path,
                    "layer": layer,
                    "object_id": object_id,
                    "Q": qa.get("Q")
                })

# 4. 检查每个 object_id 的风险链是否连续
bad_incomplete_chains = []
for path, obj_dict in risk_chain_by_object.items():
    for object_id, layers in obj_dict.items():
        if not layers:
            continue
        # QA4 可能因为没有 future_trend_info 而不存在，所以 layer=7 不强制要求
        required = {4, 5, 6}
        if not required.issubset(layers):
            bad_incomplete_chains.append({
                "path": path,
                "object_id": object_id,
                "layers": sorted(list(layers)),
                "missing": sorted(list(required - layers))
            })

print("========== Basic ==========")
print("num_vqa_files:", len(vqa_files))
print(stats)

print("\n========== Bad graph edges ==========")
print("bad_edges:", len(bad_edges))
for x in bad_edges[:10]:
    print(x)

print("\n========== Missing qa_meta ==========")
print("bad_missing_meta:", len(bad_missing_meta))
for x in bad_missing_meta[:10]:
    print(x)

print("\n========== Missing object_tags ==========")
print("bad_object_tags:", len(bad_object_tags))
for x in bad_object_tags[:10]:
    print(x)

print("\n========== Incomplete risk chains ==========")
print("bad_incomplete_chains:", len(bad_incomplete_chains))
for x in bad_incomplete_chains[:10]:
    print(x)