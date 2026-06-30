#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_dreamer_qa_with_gpt.py

Read generated risk_waypoints_bicycle_dreamer/*.json.gz files and evaluate
their VLA QA quality with:
  1) rule-based checks
  2) GPT-based structured evaluation

Usage example:

python tools_bev/evaluate_dreamer_qa_with_gpt.py \
  --input /home/liulei/ll/simlingo/database/simlingo_v2_2026_06_02/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_06_02_16_40_27/risk_waypoints_bicycle_dreamer \
  --output_jsonl qa_eval_0073.jsonl \
  --output_csv qa_eval_0073.csv \
  --model gpt-4o-mini \
  --max_files 20

For full dataset:
python tools_bev/evaluate_dreamer_qa_with_gpt.py \
  --input /path/to/random_weather_seed_3_balanced_100 \
  --recursive \
  --folder_name risk_waypoints_bicycle_dreamer \
  --output_jsonl qa_eval.jsonl \
  --output_csv qa_eval.csv \
  --model gpt-4o-mini
"""

import argparse
import csv
import gzip
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


TECHNICAL_TERMS_ZH = [
    "综合评分", "平均风险代价", "最大风险代价", "高风险比例",
    "hard_ratio", "score", "cost", "route_deviation", "candidate",
    "fallback", "专家轨迹", "回退", "标签稳定",
]

TECHNICAL_TERMS_EN = [
    "score", "mean cost", "maximum cost", "hard-risk ratio", "hard ratio",
    "route deviation", "candidate score", "fallback", "expert trajectory",
    "label stable", "cost map", "costmap",
]


EVAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "factual_consistency": {"type": "integer", "minimum": 1, "maximum": 5},
        "focus_alignment": {"type": "integer", "minimum": 1, "maximum": 5},
        "action_correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "environment_specificity": {"type": "integer", "minimum": 1, "maximum": 5},
        "language_naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
        "redundancy": {"type": "integer", "minimum": 1, "maximum": 5},
        "technical_leakage": {"type": "integer", "minimum": 1, "maximum": 5},
        "keep_for_training": {"type": "boolean"},
        "main_issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "suggested_revision": {"type": "string"},
        "brief_comment": {"type": "string"},
    },
    "required": [
        "overall_score",
        "factual_consistency",
        "focus_alignment",
        "action_correctness",
        "environment_specificity",
        "language_naturalness",
        "redundancy",
        "technical_leakage",
        "keep_for_training",
        "main_issues",
        "suggested_revision",
        "brief_comment",
    ],
}


def load_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def find_json_gz_files(input_path: Path, recursive: bool, folder_name: str) -> List[Path]:
    if input_path.is_file() and input_path.name.endswith(".json.gz"):
        return [input_path]

    if not recursive:
        return sorted(input_path.glob("*.json.gz"))

    files = []
    for p in input_path.rglob("*.json.gz"):
        if folder_name and p.parent.name != folder_name:
            continue
        files.append(p)

    return sorted(files)


def split_sentences(text: str, lang: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []

    if lang == "zh":
        parts = re.split(r"[。！？!?]+", text)
    else:
        parts = re.split(r"[.!?]+", text)

    return [p.strip() for p in parts if p.strip()]


def repetition_ratio(text: str, lang: str) -> float:
    sents = split_sentences(text, lang)
    if not sents:
        return 0.0

    normalized = [re.sub(r"\s+", "", s.lower()) for s in sents]
    unique = set(normalized)
    return 1.0 - len(unique) / max(len(normalized), 1)


def find_technical_leakage(text: str, lang: str) -> List[str]:
    terms = TECHNICAL_TERMS_ZH if lang == "zh" else TECHNICAL_TERMS_EN
    lower = str(text).lower()
    found = []
    for t in terms:
        if t.lower() in lower:
            found.append(t)
    return found


def infer_question_focus(question: str, lang: str) -> str:
    q = str(question)

    if lang == "zh":
        if "右" in q:
            return "right"
        if "左" in q:
            return "left"
        if "如何行驶" in q or "应该" in q:
            return "route_or_selected"
        return "general"

    ql = q.lower()
    if "right" in ql:
        return "right"
    if "left" in ql:
        return "left"
    if "what should" in ql:
        return "route_or_selected"
    return "general"


def rule_based_checks(qa: Dict[str, Any], lang: str) -> Dict[str, Any]:
    question = qa.get("question", "")
    answer = qa.get("answer", "")

    leakage = find_technical_leakage(answer, lang)
    rep_ratio = repetition_ratio(answer, lang)
    focus = infer_question_focus(question, lang)

    warnings = []

    if leakage:
        warnings.append(f"technical_terms={leakage}")

    if rep_ratio >= 0.25:
        warnings.append(f"high_sentence_repetition={rep_ratio:.2f}")

    if focus == "right":
        if lang == "zh":
            if "右" not in answer:
                warnings.append("right_question_answer_does_not_mention_right")
            if "左前方" in answer and "右" not in answer[:30]:
                warnings.append("right_question_may_over_focus_left_front")
        else:
            al = answer.lower()
            if "right" not in al:
                warnings.append("right_question_answer_does_not_mention_right")
            if "front-left" in al and "right" not in al[:80]:
                warnings.append("right_question_may_over_focus_front_left")

    if focus == "left":
        if lang == "zh":
            if "左" not in answer:
                warnings.append("left_question_answer_does_not_mention_left")
        else:
            if "left" not in answer.lower():
                warnings.append("left_question_answer_does_not_mention_left")

    return {
        "focus": focus,
        "technical_leakage_terms": leakage,
        "sentence_repetition_ratio": round(rep_ratio, 4),
        "warnings": warnings,
    }


def compact_actor(actor: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(actor, dict) or not actor.get("exists", False):
        return {"exists": False}

    keys = [
        "exists", "class", "class_zh", "class_en",
        "relative_position", "relative_position_zh", "relative_position_en",
        "motion_state", "motion_state_zh", "motion_state_en",
        "relative_motion", "relative_motion_zh", "relative_motion_en",
        "distance_m", "x_m", "y_m", "description_zh", "description_en",
    ]
    return {k: actor.get(k) for k in keys if k in actor}


def compact_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    static_ctx = c.get("candidate_static_context", {}) or {}
    return {
        "dreamer_candidate_id": c.get("dreamer_candidate_id"),
        "source_candidate_index": c.get("source_candidate_index"),
        "is_selected": c.get("is_selected"),
        "dreamer_role": c.get("dreamer_role"),
        "behavior_name": c.get("behavior_name"),
        "behavior_name_zh": c.get("behavior_name_zh"),
        "behavior_name_en": c.get("behavior_name_en"),
        "allowed": c.get("allowed"),
        "reasons": c.get("reasons", []),
        "score": c.get("score"),
        "candidate_static_context": {
            "side": static_ctx.get("side"),
            "dominant_static_type": static_ctx.get("dominant_static_type"),
            "description_zh": static_ctx.get("description_zh"),
            "description_en": static_ctx.get("description_en"),
            "stats": static_ctx.get("stats", {}),
        },
        "scene_brief_zh": c.get("scene_brief_zh"),
        "scene_brief_en": c.get("scene_brief_en"),
        "action_rationale_zh": c.get("action_rationale_zh"),
        "action_rationale_en": c.get("action_rationale_en"),
        "final_response_zh": c.get("final_response_zh"),
        "final_response_en": c.get("final_response_en"),
    }


def build_eval_payload(frame_data: Dict[str, Any], qa: Dict[str, Any], lang: str) -> Dict[str, Any]:
    key_dyn = frame_data.get("key_dynamic_context", {}) or {}

    context = {
        "frame": frame_data.get("frame"),
        "selected_behavior_name": frame_data.get("selected_behavior_name"),
        "risk_label_valid": frame_data.get("risk_label_valid"),
        "fallback_to_expert": frame_data.get("fallback_to_expert"),
        "scene_context": frame_data.get("scene_context", {}),
        "lateral_space_context": frame_data.get("lateral_space_context", {}),
        "key_dynamic_context": {
            "ego_speed_mps": key_dyn.get("ego_speed_mps"),
            "front_actor": compact_actor(key_dyn.get("front_actor", {})),
            "front_center_actor": compact_actor(key_dyn.get("front_center_actor", {})),
            "front_left_actor": compact_actor(key_dyn.get("front_left_actor", {})),
            "front_right_actor": compact_actor(key_dyn.get("front_right_actor", {})),
            "left_side_actor": compact_actor(key_dyn.get("left_side_actor", {})),
            "right_side_actor": compact_actor(key_dyn.get("right_side_actor", {})),
            "nearby_walker": compact_actor(key_dyn.get("nearby_walker", {})),
        },
        "dreamer_candidates": [
            compact_candidate(c) for c in frame_data.get("dreamer_candidates", [])
        ],
    }

    return {
        "language": lang,
        "task": qa.get("task", ""),
        "question": qa.get("question", ""),
        "answer": qa.get("answer", ""),
        "rule_based_checks": rule_based_checks(qa, lang),
        "context": context,
    }


def call_gpt_evaluator(client: OpenAI, model: str, payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    system_prompt = """
You are an evaluator for VLA autonomous-driving QA labels.
Evaluate whether the generated QA answer is suitable for training a language-conditioned driving model.

Important evaluation principles:
1. Judge the answer against the provided structured context, not against generic driving assumptions.
2. If the question is about right-side avoidance, the answer should focus on right-side dynamic/static space.
3. If the question is about left-side avoidance, the answer should focus on left-side dynamic/static space.
4. If the question is about the selected action, the answer should explain the current route and relevant actors.
5. Penalize repeated wording, generic statements, and engineering terms such as score, cost, hard_ratio, route_deviation.
6. The answer should be natural, concise, and useful as VLA training supervision.
7. Return scores from 1 to 5, where 5 is best.
8. redundancy score: 5 means no repetition, 1 means severe repetition.
9. technical_leakage score: 5 means no technical leakage, 1 means severe leakage.
""".strip()

    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "dreamer_qa_evaluation",
                        "strict": True,
                        "schema": EVAL_SCHEMA,
                    }
                },
            )
            return json.loads(resp.output_text)
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"OpenAI evaluation failed after {max_retries} retries: {last_err}")


def iter_qa_items(frame_data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    items = []
    for qa in frame_data.get("dreamer_qa_pairs_zh", []) or []:
        items.append(("zh", qa))
    for qa in frame_data.get("dreamer_qa_pairs_en", []) or []:
        items.append(("en", qa))
    return items


def write_csv_row(writer, result: Dict[str, Any]):
    eval_result = result.get("gpt_eval", {}) or {}
    rb = result.get("rule_based_checks", {}) or {}

    writer.writerow({
        "file": result.get("file"),
        "frame": result.get("frame"),
        "lang": result.get("lang"),
        "task": result.get("task"),
        "question": result.get("question"),
        "answer": result.get("answer"),
        "overall_score": eval_result.get("overall_score"),
        "factual_consistency": eval_result.get("factual_consistency"),
        "focus_alignment": eval_result.get("focus_alignment"),
        "action_correctness": eval_result.get("action_correctness"),
        "environment_specificity": eval_result.get("environment_specificity"),
        "language_naturalness": eval_result.get("language_naturalness"),
        "redundancy": eval_result.get("redundancy"),
        "technical_leakage": eval_result.get("technical_leakage"),
        "keep_for_training": eval_result.get("keep_for_training"),
        "rule_warnings": "; ".join(rb.get("warnings", [])),
        "main_issues": " | ".join(eval_result.get("main_issues", [])),
        "suggested_revision": eval_result.get("suggested_revision"),
        "brief_comment": eval_result.get("brief_comment"),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="A .json.gz file, a dreamer folder, or dataset root.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--folder_name", type=str, default="risk_waypoints_bicycle_dreamer")
    parser.add_argument("--output_jsonl", type=str, default="dreamer_qa_eval.jsonl")
    parser.add_argument("--output_csv", type=str, default="dreamer_qa_eval.csv")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_files", type=int, default=-1)
    parser.add_argument("--max_qa_per_file", type=int, default=-1)
    parser.add_argument("--rules_only", action="store_true", help="Only run rule-based checks, do not call GPT.")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    files = find_json_gz_files(input_path, args.recursive, args.folder_name)

    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"[Info] Found {len(files)} json.gz files.")

    client = None if args.rules_only else OpenAI()

    csv_fields = [
        "file", "frame", "lang", "task", "question", "answer",
        "overall_score", "factual_consistency", "focus_alignment",
        "action_correctness", "environment_specificity",
        "language_naturalness", "redundancy", "technical_leakage",
        "keep_for_training", "rule_warnings", "main_issues",
        "suggested_revision", "brief_comment",
    ]

    with open(args.output_jsonl, "w", encoding="utf-8") as jf, \
         open(args.output_csv, "w", encoding="utf-8", newline="") as cf:

        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()

        total = 0
        for file_idx, path in enumerate(files):
            frame_data = load_json_gz(path)
            qa_items = iter_qa_items(frame_data)

            if args.max_qa_per_file > 0:
                qa_items = qa_items[: args.max_qa_per_file]

            for lang, qa in qa_items:
                payload = build_eval_payload(frame_data, qa, lang)
                rb = payload["rule_based_checks"]

                result = {
                    "file": str(path),
                    "frame": frame_data.get("frame"),
                    "lang": lang,
                    "task": qa.get("task"),
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "rule_based_checks": rb,
                }

                if args.rules_only:
                    result["gpt_eval"] = {}
                else:
                    result["gpt_eval"] = call_gpt_evaluator(
                        client=client,
                        model=args.model,
                        payload=payload,
                    )

                jf.write(json.dumps(result, ensure_ascii=False) + "\n")
                write_csv_row(writer, result)
                total += 1

                print(
                    f"[{total}] frame={result['frame']} lang={lang} "
                    f"task={result['task']} "
                    f"score={result.get('gpt_eval', {}).get('overall_score', 'NA')} "
                    f"warnings={rb.get('warnings', [])}"
                )

                if args.sleep > 0:
                    time.sleep(args.sleep)

    print(f"[Done] Evaluated {total} QA pairs.")
    print(f"[Saved] {args.output_jsonl}")
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()