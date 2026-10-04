#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import gzip
import json
from collections import Counter, OrderedDict
from pathlib import Path


# =============================================================================
# 配置
# =============================================================================

DATA_ROOT = Path(
    "/root/autodl-tmp/database/"
    "simlingo_v2_2026_09_12_22_24_28/"
    "data/simlingo"
)

KEYFRAME_FILE = DATA_ROOT / "keyframes.txt"

LG_OUTPUT_FOLDER = "language_grounded_waypoints"

OUTPUT_ROOT = DATA_ROOT / "lg_cvaa_results"


# =============================================================================
# 基础 IO
# =============================================================================

def load_json_gz(path: Path):
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_jsonl_line(fp, obj):
    fp.write(
        json.dumps(
            obj,
            ensure_ascii=False,
        )
        + "\n"
    )


def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def safe_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def actor_id_string(actor):
    if not isinstance(actor, dict):
        return None

    actor_id = actor.get("id", None)

    if actor_id is None:
        actor_id = actor.get("track_id", None)

    if actor_id is None:
        return None

    return str(actor_id)


# =============================================================================
# keyframes.txt
# =============================================================================

def load_keyframe_manifest(
    data_root: Path,
    keyframe_file: Path,
):
    """
    keyframes.txt 每行：

        relative/route/path/000123

    最后一段是 frame，
    前面的部分是相对于 DATA_ROOT 的 route path。
    """

    data_root = data_root.expanduser().resolve()
    keyframe_file = keyframe_file.expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {data_root}"
        )

    if not keyframe_file.is_file():
        raise FileNotFoundError(
            f"KEYFRAME_FILE does not exist: {keyframe_file}"
        )

    items = []
    seen = set()

    with keyframe_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, raw in enumerate(f, 1):

            value = raw.strip()

            if not value or value.startswith("#"):
                continue

            value = (
                value
                .replace("\\", "/")
                .rstrip("/")
            )

            parts = [
                x
                for x in value.split("/")
                if x
            ]

            if len(parts) < 2:
                raise ValueError(
                    f"{keyframe_file}:{line_no}: "
                    f"invalid entry: {value!r}"
                )

            frame = str(parts[-1])

            route_rel = Path(
                *parts[:-1]
            )

            route_dir = (
                data_root
                / route_rel
            ).resolve()

            try:
                route_dir.relative_to(data_root)
            except Exception:
                raise ValueError(
                    f"{keyframe_file}:{line_no}: "
                    f"path escapes DATA_ROOT: {value!r}"
                )

            key = (
                str(route_dir),
                frame,
            )

            if key in seen:
                continue

            seen.add(key)

            items.append(
                {
                    "route_id": route_dir.name,
                    "route_rel": route_rel.as_posix(),
                    "route_dir": str(route_dir),
                    "frame": frame,
                }
            )

    return items


# =============================================================================
# frame-level 数据
# =============================================================================

def build_frame_summary(
    manifest_item,
    data,
):
    causal = data.get(
        "causal_analysis",
        {},
    ) or {}

    supervision = data.get(
        "supervision",
        {},
    ) or {}

    causal_object = causal.get(
        "causal_object",
        {},
    ) or {}

    rejected_object = causal.get(
        "rejected_causal_object",
        None,
    )

    object_tests = causal.get(
        "object_tests",
        [],
    ) or []

    valid_counterfactual_tests = sum(
        1
        for x in object_tests
        if bool(
            x.get(
                "counterfactual_valid",
                False,
            )
        )
    )

    accepted_tests = sum(
        1
        for x in object_tests
        if bool(
            x.get(
                "causal_accepted",
                False,
            )
        )
    )

    return {
        "route_id": manifest_item["route_id"],
        "route_rel": manifest_item["route_rel"],
        "route_dir": manifest_item["route_dir"],
        "frame": manifest_item["frame"],

        "generator": data.get(
            "generator",
            None,
        ),

        "risk_label_valid": bool(
            supervision.get(
                "risk_label_valid",
                False,
            )
        ),

        "selector_label": safe_int(
            supervision.get(
                "selector_label",
                None,
            )
        ),

        "selected_intent_id": safe_int(
            supervision.get(
                "selected_intent_id",
                None,
            )
        ),

        "selected_intent_name": supervision.get(
            "selected_intent_name",
            None,
        ),

        "selected_internal_intent_name":
            supervision.get(
                "selected_internal_intent_name",
                None,
            ),

        # -------------------------------------------------------------
        # LG 最终 causal actor
        # -------------------------------------------------------------

        "has_causal_object": bool(
            causal.get(
                "has_causal_object",
                False,
            )
        ),

        "causal_actor_id":
            actor_id_string(
                causal_object
            ),

        "causal_actor_class":
            causal_object.get(
                "class",
                None,
            )
            if isinstance(
                causal_object,
                dict,
            )
            else None,

        "causal_score": safe_float(
            causal.get(
                "causal_score",
                None,
            )
        ),

        "preliminary_causal_score":
            safe_float(
                causal.get(
                    "preliminary_causal_score",
                    None,
                )
            ),

        "final_causal_score":
            safe_float(
                causal.get(
                    "final_causal_score",
                    None,
                )
            ),

        "final_revalidation_passed":
            causal.get(
                "final_revalidation_passed",
                None,
            ),

        "causal_acceptance_reason":
            causal.get(
                "causal_acceptance_reason",
                None,
            ),

        "reference_source":
            causal.get(
                "reference_source",
                None,
            ),

        "counterfactual_intent_name":
            causal.get(
                "counterfactual_intent_name",
                None,
            ),

        # -------------------------------------------------------------
        # actor tests
        # -------------------------------------------------------------

        "num_object_tests":
            int(
                len(object_tests)
            ),

        "num_valid_counterfactual_tests":
            int(
                valid_counterfactual_tests
            ),

        "num_causal_accepted_tests":
            int(
                accepted_tests
            ),

        # -------------------------------------------------------------
        # 如果 final revalidation 把初选 causal object 否掉
        # -------------------------------------------------------------

        "has_rejected_causal_object":
            isinstance(
                rejected_object,
                dict,
            )
            and bool(
                rejected_object
            ),

        "rejected_causal_actor_id":
            actor_id_string(
                rejected_object
            )
            if isinstance(
                rejected_object,
                dict,
            )
            else None,

        "rejected_causal_actor_class":
            rejected_object.get(
                "class",
                None,
            )
            if isinstance(
                rejected_object,
                dict,
            )
            else None,
    }


# =============================================================================
# actor-level 数据
# =============================================================================

def build_actor_record(
    manifest_item,
    test,
    test_index,
):
    actor = test.get(
        "actor",
        {},
    ) or {}

    effect = test.get(
        "effect",
        {},
    ) or {}

    final_effect = test.get(
        "final_effect",
        {},
    ) or {}

    record = {
        # 与 CVAA all_actor_scores.jsonl 保持同类主键
        "route_id": manifest_item["route_id"],
        "route_rel": manifest_item["route_rel"],
        "route_dir": manifest_item["route_dir"],
        "frame": manifest_item["frame"],

        "actor_id": actor_id_string(actor),

        "actor_class": actor.get(
            "class",
            None,
        ),

        "semantic_class": actor.get(
            "semantic_class",
            None,
        ),

        "raw_class": actor.get(
            "raw_class",
            None,
        ),

        "distance_m": safe_float(
            actor.get(
                "distance_m",
                None,
            )
        ),

        "x_m": safe_float(
            actor.get(
                "x_m",
                None,
            )
        ),

        "y_m": safe_float(
            actor.get(
                "y_m",
                None,
            )
        ),

        "causal_test_relevance_score":
            safe_float(
                actor.get(
                    "causal_test_relevance_score",
                    None,
                )
            ),

        # LG 测试顺序。
        # rank 后面会根据 causal_score 重新计算，
        # 所以这里单独保留 test_index。
        "test_index": int(
            test_index
        ),

        "counterfactual_valid": bool(
            test.get(
                "counterfactual_valid",
                False,
            )
        ),

        "causal_score": safe_float(
            test.get(
                "causal_score",
                None,
            )
        ),

        "preliminary_causal_score":
            safe_float(
                test.get(
                    "preliminary_causal_score",
                    None,
                )
            ),

        "causal_accepted":
            test.get(
                "causal_accepted",
                None,
            ),

        "causal_acceptance_reason":
            test.get(
                "causal_acceptance_reason",
                None,
            ),

        "counterfactual_selected_index":
            safe_int(
                test.get(
                    "counterfactual_selected_index",
                    None,
                )
            ),

        "counterfactual_candidate_count":
            safe_int(
                test.get(
                    "counterfactual_candidate_count",
                    None,
                )
            ),

        "counterfactual_intent_name":
            test.get(
                "counterfactual_intent_name",
                None,
            ),

        "counterfactual_selected_variant":
            test.get(
                "counterfactual_selected_variant",
                None,
            ),

        "full_scene_intent_name":
            test.get(
                "full_scene_intent_name",
                None,
            ),

        "counterfactual_factor_type":
            test.get(
                "counterfactual_factor_type",
                None,
            ),

        "counterfactual_remaining_actor_count":
            safe_int(
                test.get(
                    "counterfactual_remaining_actor_count",
                    None,
                )
            ),

        # 只有最终被选中的 causal actor
        # 才会经过 final revalidation，
        # 所以其他 actor 这里通常为空。
        "final_causal_score":
            safe_float(
                test.get(
                    "final_causal_score",
                    None,
                )
            ),

        "final_revalidation_passed":
            test.get(
                "final_revalidation_passed",
                None,
            ),

        "final_causal_acceptance_reason":
            test.get(
                "final_causal_acceptance_reason",
                None,
            ),

        "final_full_scene_intent_name":
            test.get(
                "final_full_scene_intent_name",
                None,
            ),

        # 完整 effect 保留下来，方便后续论文分析
        "effect": effect,

        "final_effect": final_effect
        if final_effect
        else None,

        # 完整原始 test 也保留，避免汇总时丢字段
        "raw_object_test": test,
    }

    return record


# =============================================================================
# LG actor ranking
# =============================================================================

def assign_lg_ranks(actor_records):
    """
    对同一 frame 内 counterfactual_valid=True 的 actor
    按 causal_score 从大到小排序。

    注意：
    这个 rank 是后处理排名，
    不改变 LG 原方法，也不改变 has_causal_object / threshold / revalidation。
    """

    valid_records = [
        x
        for x in actor_records
        if bool(
            x.get(
                "counterfactual_valid",
                False,
            )
        )
        and x.get(
            "causal_score",
            None,
        )
        is not None
    ]

    valid_records.sort(
        key=lambda x: (
            -float(
                x.get(
                    "causal_score",
                    0.0,
                )
            ),
            str(
                x.get(
                    "actor_id",
                    "",
                )
            ),
        )
    )

    rank_map = {}

    for rank, item in enumerate(
        valid_records,
        1,
    ):
        key = (
            item.get(
                "actor_id",
                None,
            ),
            item.get(
                "test_index",
                None,
            ),
        )

        rank_map[key] = rank

    for item in actor_records:

        key = (
            item.get(
                "actor_id",
                None,
            ),
            item.get(
                "test_index",
                None,
            ),
        )

        item["lg_rank"] = rank_map.get(
            key,
            None,
        )

    return actor_records


# =============================================================================
# CSV
# =============================================================================

def write_actor_csv(
    path: Path,
    records,
):
    columns = [
        "route_id",
        "route_rel",
        "route_dir",
        "frame",

        "actor_id",
        "actor_class",
        "semantic_class",
        "raw_class",

        "distance_m",
        "x_m",
        "y_m",

        "causal_test_relevance_score",

        "test_index",
        "lg_rank",

        "counterfactual_valid",

        "causal_score",
        "preliminary_causal_score",

        "causal_accepted",
        "causal_acceptance_reason",

        "counterfactual_selected_index",
        "counterfactual_candidate_count",
        "counterfactual_intent_name",
        "counterfactual_selected_variant",

        "full_scene_intent_name",
        "counterfactual_factor_type",
        "counterfactual_remaining_actor_count",

        "final_causal_score",
        "final_revalidation_passed",
        "final_causal_acceptance_reason",
        "final_full_scene_intent_name",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()

        for item in records:
            writer.writerow(item)


# =============================================================================
# 主程序
# =============================================================================

def main():

    data_root = DATA_ROOT.expanduser().resolve()

    keyframe_file = (
        KEYFRAME_FILE
        .expanduser()
        .resolve()
    )

    output_root = (
        OUTPUT_ROOT
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_keyframe_manifest(
        data_root=data_root,
        keyframe_file=keyframe_file,
    )

    print(
        f"[Manifest] keyframes={len(manifest)}"
    )

    full_output_path = (
        output_root
        / "all_frame_results_full.jsonl"
    )

    frame_output_path = (
        output_root
        / "all_frame_results.jsonl"
    )

    actor_output_path = (
        output_root
        / "all_actor_scores.jsonl"
    )

    actor_csv_path = (
        output_root
        / "all_actor_scores.csv"
    )

    missing_output_path = (
        output_root
        / "missing_keyframes.jsonl"
    )

    summary_path = (
        output_root
        / "run_summary.json"
    )

    # -------------------------------------------------------------------------
    # 全局统计
    # -------------------------------------------------------------------------

    outputs_found = 0
    outputs_missing = 0

    risk_valid_true = 0
    risk_valid_false = 0

    causal_true = 0
    causal_false = 0

    frames_with_object_tests = 0

    total_object_tests = 0
    total_valid_counterfactual_tests = 0
    total_accepted_tests = 0

    actor_class_counter = Counter()
    causal_actor_class_counter = Counter()

    all_actor_records = []

    route_ids = set()

    # -------------------------------------------------------------------------
    # 汇总
    # -------------------------------------------------------------------------

    with full_output_path.open(
        "w",
        encoding="utf-8",
    ) as full_fp, \
        frame_output_path.open(
            "w",
            encoding="utf-8",
        ) as frame_fp, \
        actor_output_path.open(
            "w",
            encoding="utf-8",
        ) as actor_fp, \
        missing_output_path.open(
            "w",
            encoding="utf-8",
        ) as missing_fp:

        for index, item in enumerate(
            manifest,
            1,
        ):

            route_dir = Path(
                item["route_dir"]
            )

            frame = item["frame"]

            route_ids.add(
                item["route_id"]
            )

            lg_path = (
                route_dir
                / LG_OUTPUT_FOLDER
                / f"{frame}.json.gz"
            )

            # -----------------------------------------------------------------
            # missing
            # -----------------------------------------------------------------

            if not lg_path.is_file():

                outputs_missing += 1

                measurement_path = (
                    route_dir
                    / "measurements"
                    / f"{frame}.json.gz"
                )

                costmap_path = (
                    route_dir
                    / "costmap"
                    / f"{frame}.npy"
                )

                lane_path = (
                    route_dir
                    / "lane_constraints"
                    / f"{frame}.npy"
                )

                missing_record = {
                    **item,

                    "expected_lg_output":
                        str(lg_path),

                    "measurement_exists":
                        measurement_path.is_file(),

                    "costmap_exists":
                        costmap_path.is_file(),

                    "lane_constraint_exists":
                        lane_path.is_file(),
                }

                write_jsonl_line(
                    missing_fp,
                    missing_record,
                )

                continue

            # -----------------------------------------------------------------
            # load
            # -----------------------------------------------------------------

            try:
                data = load_json_gz(
                    lg_path
                )
            except Exception as exc:

                outputs_missing += 1

                write_jsonl_line(
                    missing_fp,
                    {
                        **item,
                        "expected_lg_output":
                            str(lg_path),
                        "read_error":
                            repr(exc),
                    },
                )

                continue

            outputs_found += 1

            # -----------------------------------------------------------------
            # 完整 frame 原始结果
            # -----------------------------------------------------------------

            full_record = {
                "route_id":
                    item["route_id"],

                "route_rel":
                    item["route_rel"],

                "route_dir":
                    item["route_dir"],

                "frame":
                    item["frame"],

                "lg_output_path":
                    str(lg_path),

                "lg_result":
                    data,
            }

            write_jsonl_line(
                full_fp,
                full_record,
            )

            # -----------------------------------------------------------------
            # frame summary
            # -----------------------------------------------------------------

            frame_record = build_frame_summary(
                manifest_item=item,
                data=data,
            )

            write_jsonl_line(
                frame_fp,
                frame_record,
            )

            if frame_record[
                "risk_label_valid"
            ]:
                risk_valid_true += 1
            else:
                risk_valid_false += 1

            if frame_record[
                "has_causal_object"
            ]:
                causal_true += 1

                cls = frame_record.get(
                    "causal_actor_class",
                    None,
                )

                if cls is not None:
                    causal_actor_class_counter[
                        str(cls)
                    ] += 1

            else:
                causal_false += 1

            # -----------------------------------------------------------------
            # actor tests
            # -----------------------------------------------------------------

            causal = data.get(
                "causal_analysis",
                {},
            ) or {}

            object_tests = causal.get(
                "object_tests",
                [],
            ) or []

            if object_tests:
                frames_with_object_tests += 1

            frame_actor_records = []

            for test_index, test in enumerate(
                object_tests,
                1,
            ):

                record = build_actor_record(
                    manifest_item=item,
                    test=test,
                    test_index=test_index,
                )

                frame_actor_records.append(
                    record
                )

            # causal_score 后处理 ranking
            frame_actor_records = assign_lg_ranks(
                frame_actor_records
            )

            for record in frame_actor_records:

                total_object_tests += 1

                if record.get(
                    "counterfactual_valid",
                    False,
                ):
                    total_valid_counterfactual_tests += 1

                if record.get(
                    "causal_accepted",
                    False,
                ):
                    total_accepted_tests += 1

                cls = record.get(
                    "actor_class",
                    None,
                )

                if cls is not None:
                    actor_class_counter[
                        str(cls)
                    ] += 1

                all_actor_records.append(
                    record
                )

                write_jsonl_line(
                    actor_fp,
                    record,
                )

            if index % 500 == 0:
                print(
                    f"[Progress] "
                    f"{index}/{len(manifest)} "
                    f"found={outputs_found} "
                    f"missing={outputs_missing}"
                )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    write_actor_csv(
        actor_csv_path,
        all_actor_records,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "status": "complete",

        "data_root":
            str(data_root),

        "keyframe_file":
            str(keyframe_file),

        "output_root":
            str(output_root),

        "manifest_routes":
            int(
                len(route_ids)
            ),

        "manifest_keyframes":
            int(
                len(manifest)
            ),

        "lg_outputs_found":
            int(
                outputs_found
            ),

        "lg_outputs_missing":
            int(
                outputs_missing
            ),

        "lg_output_coverage":
            (
                float(
                    outputs_found
                    / len(manifest)
                )
                if manifest
                else 0.0
            ),

        "risk_label_valid_true":
            int(
                risk_valid_true
            ),

        "risk_label_valid_false":
            int(
                risk_valid_false
            ),

        "has_causal_object_true":
            int(
                causal_true
            ),

        "has_causal_object_false":
            int(
                causal_false
            ),

        "frames_with_object_tests":
            int(
                frames_with_object_tests
            ),

        "total_object_tests":
            int(
                total_object_tests
            ),

        "valid_counterfactual_tests":
            int(
                total_valid_counterfactual_tests
            ),

        "causal_accepted_tests":
            int(
                total_accepted_tests
            ),

        "tested_actor_class_counts":
            dict(
                sorted(
                    actor_class_counter.items()
                )
            ),

        "final_causal_actor_class_counts":
            dict(
                sorted(
                    causal_actor_class_counter.items()
                )
            ),

        "files": {
            "all_frame_results_full":
                str(full_output_path),

            "all_frame_results":
                str(frame_output_path),

            "all_actor_scores":
                str(actor_output_path),

            "all_actor_scores_csv":
                str(actor_csv_path),

            "missing_keyframes":
                str(missing_output_path),
        },
    }

    write_json(
        summary_path,
        summary,
    )

    # -------------------------------------------------------------------------
    # console
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("LG CVAA aggregation complete")
    print("=" * 72)

    print(
        f"Manifest routes      : "
        f"{summary['manifest_routes']}"
    )

    print(
        f"Manifest keyframes   : "
        f"{summary['manifest_keyframes']}"
    )

    print(
        f"LG outputs found     : "
        f"{summary['lg_outputs_found']}"
    )

    print(
        f"LG outputs missing   : "
        f"{summary['lg_outputs_missing']}"
    )

    print(
        f"Coverage             : "
        f"{summary['lg_output_coverage'] * 100:.2f}%"
    )

    print(
        f"Risk label valid     : "
        f"{summary['risk_label_valid_true']}"
    )

    print(
        f"Risk label invalid   : "
        f"{summary['risk_label_valid_false']}"
    )

    print(
        f"Has causal object    : "
        f"{summary['has_causal_object_true']}"
    )

    print(
        f"No causal object     : "
        f"{summary['has_causal_object_false']}"
    )

    print(
        f"Frames w/object test : "
        f"{summary['frames_with_object_tests']}"
    )

    print(
        f"Total object tests   : "
        f"{summary['total_object_tests']}"
    )

    print(
        f"Valid CF tests       : "
        f"{summary['valid_counterfactual_tests']}"
    )

    print(
        f"Accepted tests       : "
        f"{summary['causal_accepted_tests']}"
    )

    print()
    print(
        f"Results saved to:"
    )
    print(
        f"  {output_root}"
    )


if __name__ == "__main__":
    main()