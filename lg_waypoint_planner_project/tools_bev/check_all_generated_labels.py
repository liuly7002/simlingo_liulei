#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only, full-dataset audit of SimLingo-LG and Driving generated labels.

Verified against liuly7002/simlingo_liulei main, 2026-11-03,
Git commit af88da02bb9fca15231a89e6cd2fe85b77ec79da.

Inputs per route (unmodified):
  costmap/<frame>.npy
  measurements/<frame>.json.gz
  language_grounded_waypoints/<frame>.json.gz
  future_interaction_grids/<frame>.npz
  driving_future_interaction_grids/<frame>.npz
  driving_expert_conditioned_actor_selection/<frame>.json.gz

Expected npz: future_interaction_grid [3,H,W],
              future_interaction_temporal [2,T,H,W],
              critical_actor_status in {-1,0,1}.

ERROR = objectively inconsistent/malformed artifact.
WARNING = missing output, invalid sample, potentially unusable supervision.
REVIEW = heuristic or intrinsic LG/expert action difference; NOT label corruption.

Outputs: summary.json, summary.md, route_summary.csv, frame_summary.csv,
         issues.csv, review_cases.csv; never edits any input label.

Only dependency is numpy (plus Python standard library).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Dict

import numpy as np

CODE_VERSION = "1.0.0"
SOURCE_COMMIT = "af88da02bb9fca15231a89e6cd2fe85b77ec79da"
CHANNEL_NAMES = (
    "selected_route_ego_footprint_occupancy",
    "future_ego_footprint_occupancy",
    "primary_causal_actor_future_footprint_occupancy",
)
TEMPORAL_NAMES = ("C1_ego_future", "C2_critical_actor")
QA_KEYS = ("attention", "motion_constraint", "driving_response", "future_motion")
CAMERAS = ("front", "front_left", "front_right", "rear", "rear_left", "rear_right")
FOLDERS = {
    "measurements": "measurements",
    "costmap": "costmap",
    "lg": "language_grounded_waypoints",
    "lg_grid": "future_interaction_grids",
    "driving_grid": "driving_future_interaction_grids",
    "driving_json": "driving_expert_conditioned_actor_selection",
}
ISSUE_COLUMNS = ("severity", "code", "route", "frame", "detail")
FRAME_COLUMNS = (
    "route", "frame", "eligible", "has_costmap", "has_measurement", "has_lg",
    "has_lg_grid", "has_driving_grid", "has_driving_json", "lg_risk_valid",
    "lg_grid_valid", "lg_status", "driving_grid_valid", "driving_status",
    "lg_actor_id", "driving_actor_id", "lg_positive_test_count",
    "lg_test_count", "lg_valid_test_count", "lg_expert_ade_m", "lg_expert_fde_m",
    "lg_c2_mean_mass", "driving_c2_mean_mass", "lg_spatial_valid",
    "driving_spatial_valid", "error_count", "warning_count", "review_count",
)
ROUTE_COLUMNS = (
    "route", "eligible_frames", "costmap", "measurement", "lg", "lg_grid",
    "driving_grid", "driving_json", "lg_grid_valid", "driving_grid_valid",
    "lg_status_1", "lg_status_0", "lg_status_-1", "driving_status_1",
    "driving_status_0", "driving_status_-1", "errors", "warnings", "reviews",
)
REVIEW_COLUMNS = (
    "route", "frame", "reason", "lg_status", "driving_status",
    "lg_actor_id", "driving_actor_id", "expert_distance_m", "lg_distance_m",
    "lg_expert_ade_m", "lg_expert_fde_m", "lg_q1", "lg_q2", "driving_q3", "driving_q4",
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Full, non-destructive audit of LG and Driving labels (3-channel + 10-frame temporal)."
    )
    ap.add_argument("--input", required=True, type=Path,
                    help="Dataset root containing route directories (or a single route directory).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Report directory. Existing report files with the same names are replaced.")
    ap.add_argument("--horizon", type=int, default=10,
                    help="Expected temporal horizon T (default: 10).")
    ap.add_argument("--progress-every", type=int, default=2000,
                    help="Display a progress line every N frames (0 disables).")
    ap.add_argument("--max-issues", type=int, default=200000,
                    help="Maximum issue rows written; ALL issues are counted (0 means no limit).")
    ap.add_argument("--max-review-cases", type=int, default=20000,
                    help="Maximum detailed heuristic review rows written (0 means no limit).")
    ap.add_argument("--review-ade-m", type=float, default=3.0,
                    help="Threshold for LG vs expert waypoint ADE review, meters.")
    ap.add_argument("--review-fde-m", type=float, default=5.0,
                    help="Threshold for LG vs expert endpoint distance review, meters.")
    ap.add_argument("--stationary-distance-m", type=float, default=0.50,
                    help="Expert or LG trajectory endpoint norm at/below which we treat it as stationary.")
    ap.add_argument("--q2-stop-review-distance-m", type=float, default=2.50,
                    help="Expert future progress threshold for Q2 strict-stop wording review.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Read at most N frames for a quick test; 0 means ALL frames.")
    ap.add_argument("--skip-costmap-shape", action="store_true",
                    help="Skip opening costmap npy headers for H,W consistency. Faster for large data.")
    ap.add_argument("--skip-costmap-values", action="store_true",
                    help="Skip scanning costmap array finite/range values (default: audit these).")
    ap.add_argument("--skip-pixelwise-c2", action="store_true",
                    help="Skip comparing LG and Driving C2 masks pixel by pixel.")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="Treat missing labels for eligible costmap+measurement frames as ERROR (default: WARNING).")
    ap.add_argument("--lg-folder", default=FOLDERS["lg"])
    ap.add_argument("--lg-grid-folder", default=FOLDERS["lg_grid"])
    ap.add_argument("--driving-grid-folder", default=FOLDERS["driving_grid"])
    ap.add_argument("--driving-json-folder", default=FOLDERS["driving_json"])
    ap.add_argument("--costmap-folder", default=FOLDERS["costmap"])
    ap.add_argument("--measurements-folder", default=FOLDERS["measurements"])
    return ap.parse_args(argv)


def as_dict(x):
    return x if isinstance(x, dict) else {}


def scalar(x, default=None):
    if x is None:
        return default
    try:
        a = np.asarray(x)
        if a.ndim == 0:
            return a.item()
        if a.size == 1:
            return a.reshape(-1)[0].item()
    except (TypeError, ValueError, AttributeError):
        pass
    return default


def number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError, OverflowError):
        return None


def actor_id(x):
    if isinstance(x, dict):
        x = x.get("id")
    return "" if x is None else str(x).strip()


def qa_answer(qa, key):
    return str(as_dict(as_dict(qa).get(key)).get("answer", "")).strip()


def load_gz_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        o = json.load(f)
    if not isinstance(o, dict):
        raise ValueError("Top-level JSON is not an object")
    return o


def discover_routes(root: Path, folders: Dict[str, str]):
    """Traverse directory names, not millions of file entries or their contents."""
    hits = set(folders.values())
    routes = []
    for parent, dirs, _ in os.walk(root):
        present = hits.intersection(dirs)
        if present:
            routes.append(Path(parent))
            # No need to descend into per-route artifact folders; image subdirs etc.
            dirs[:] = [d for d in dirs if d not in hits]
    return sorted(set(routes))


def frames_in(route: Path, folder: str, suffix: str):
    d = route / folder
    if not d.is_dir():
        return set()
    return {p.name[:-len(suffix)] for p in d.glob("*" + suffix) if p.is_file()}


def npy_shape(path: Path):
    # mmap avoids loading the HxW pixel array.
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = a.shape
    del a
    return tuple(shape[-2:]) if len(shape) >= 2 else tuple(shape)


def route_points_valid(x):
    try:
        a = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] >= 2 and bool(np.isfinite(a).all())


def shape_points(x, n):
    try:
        a = np.asarray(x, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if a.ndim != 2 or a.shape != (n, 2) or not np.isfinite(a).all():
        return None
    return a


def distance_last(points):
    if points is None or points.shape[0] == 0:
        return None
    return float(np.linalg.norm(points[-1]))


def reconstruct_qa_pairs(label, frame, route, emit, origin):
    q = as_dict(label.get("core_questions"))
    anns = as_dict(label.get("language_annotation"))
    pairs = anns.get("qa_pairs_en")
    if not q and not pairs:
        return False
    if not isinstance(pairs, list) or len(pairs) != 4:
        emit("ERROR", "qa_not_four_pairs", route, frame, f"{origin}: qa_pairs_en must contain 4 pairs")
        return False
    valid = True
    for idx, key in enumerate(QA_KEYS):
        entry = as_dict(q.get(key))
        pair = as_dict(pairs[idx])
        if not entry.get("question") or not entry.get("answer") or not pair.get("question") or not pair.get("answer"):
            emit("ERROR", "qa_incomplete", route, frame, f"{origin}.{key}: missing question/answer")
            valid = False
        if entry != {"question": pair.get("question"), "answer": pair.get("answer")}:
            emit("ERROR", "qa_pair_disagrees_core", route, frame, f"{origin}.{key}: core vs list mismatch")
            valid = False
    return valid


def spatial_check(o, source, frame, route, status, emit, stats):
    g = as_dict(o.get("visual_grounding"))
    if not g:
        emit("WARNING", "spatial_missing", route, frame, f"{source}: visual_grounding missing")
        return None
    raw = g.get("participant_spatial_attention_valid")
    if not isinstance(raw, bool):
        emit("ERROR", "spatial_valid_flag_missing", route, frame, f"{source}: spatial validity is not bool")
        return None
    valid = bool(raw)
    stats[f"{source}_spatial_valid_{int(valid)}"] += 1
    cam = g.get("camera_order")
    if tuple(cam or []) != CAMERAS:
        emit("ERROR", "spatial_camera_order", route, frame, f"{source}: camera order={cam}")
    ev = as_dict(g.get("per_camera_evidence"))
    if valid:
        if status != 1:
            emit("ERROR", "spatial_valid_without_critical_actor", route, frame,
                 f"{source}: spatial=True, critical_actor_status={status}")
        for c in CAMERAS:
            e = as_dict(ev.get(c))
            if not e:
                emit("ERROR", "spatial_missing_camera", route, frame, f"{source}: no camera {c}")
                continue
            if e.get("visible"):
                box = e.get("bbox_normalized_xyxy")
                try:
                    b = np.asarray(box, dtype=np.float64)
                except (TypeError, ValueError):
                    b = np.zeros((0,))
                if b.shape != (4,) or not np.isfinite(b).all() or np.min(b) < -1e-5 or np.max(b) > 1 + 1e-5 or b[0] >= b[2] or b[1] >= b[3]:
                    emit("ERROR", "spatial_invalid_bbox", route, frame, f"{source}: camera={c}, bbox={box}")
    if status == 0 and valid:
        emit("ERROR", "no_actor_has_spatial_positive", route, frame, f"{source}: status=0 yet spatial valid")
    return valid


class Reporter:
    def __init__(self, output, max_issues, max_reviews):
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.counters = Counter()
        self.by_level = Counter()
        self.examples = defaultdict(list)
        self.written = 0
        self.review_written = 0
        self.max_issues = max_issues
        self.max_reviews = max_reviews
        self.issues_f = (output / "issues.csv").open("w", newline="", encoding="utf-8-sig")
        self.issue_writer = csv.DictWriter(self.issues_f, fieldnames=ISSUE_COLUMNS)
        self.issue_writer.writeheader()
        self.review_f = (output / "review_cases.csv").open("w", newline="", encoding="utf-8-sig")
        self.review_writer = csv.DictWriter(self.review_f, fieldnames=REVIEW_COLUMNS)
        self.review_writer.writeheader()
        self.frame_f = (output / "frame_summary.csv").open("w", newline="", encoding="utf-8-sig")
        self.frame_writer = csv.DictWriter(self.frame_f, fieldnames=FRAME_COLUMNS)
        self.frame_writer.writeheader()
        self.route_f = (output / "route_summary.csv").open("w", newline="", encoding="utf-8-sig")
        self.route_writer = csv.DictWriter(self.route_f, fieldnames=ROUTE_COLUMNS)
        self.route_writer.writeheader()

    def issue(self, severity, code, route, frame, detail):
        key = f"{severity}:{code}"
        self.counters[key] += 1
        self.by_level[severity] += 1
        if len(self.examples[key]) < 3:
            self.examples[key].append({"route": route, "frame": frame, "detail": str(detail)[:500]})
        if self.max_issues == 0 or self.written < self.max_issues:
            self.issue_writer.writerow({"severity": severity, "code": code,
                                          "route": route, "frame": frame, "detail": str(detail)[:2000]})
            self.written += 1

    def add_review_case(self, data):
        if self.max_reviews == 0 or self.review_written < self.max_reviews:
            self.review_writer.writerow({k: data.get(k, "") for k in REVIEW_COLUMNS})
            self.review_written += 1

    def close(self):
        for f in (self.issues_f, self.review_f, self.frame_f, self.route_f):
            f.close()


class LabelAuditor:
    def __init__(self, args):
        self.args = args
        self.folders = {
            "measurements": args.measurements_folder,
            "costmap": args.costmap_folder,
            "lg": args.lg_folder,
            "lg_grid": args.lg_grid_folder,
            "driving_grid": args.driving_grid_folder,
            "driving_json": args.driving_json_folder,
        }
        self.reporter = Reporter(args.output, args.max_issues, args.max_review_cases)
        self.stats = Counter()
        self.distributions = defaultdict(Counter)
        self.numeric = defaultdict(lambda: {"n": 0, "sum": 0.0, "max": None})
        self.shape_dist = Counter()
        self.total_frames = 0
        self.routes_seen = 0
        self.routes_processed = 0
        self.stopped = False

    def metric(self, name, value):
        value = number(value)
        if value is None:
            return
        m = self.numeric[name]
        m["n"] += 1
        m["sum"] += value
        m["max"] = value if m["max"] is None else max(m["max"], value)

    def read_npz(self, path, source, route, frame, cost_shape):
        """Return compact metadata and C2 masks, never keep full data across frames."""
        a = self.args
        R = self.reporter.issue
        info = {"exists": True, "valid": None, "status": None, "id": "",
                "temporal_c2": None, "grid_shape": None, "space": None, "mass": None,
                "n_frames": None, "mpp": None, "center": None, "actor_frames": None,
                "causal_score": None}
        try:
            with np.load(path, allow_pickle=False) as npz:
                keys = set(npz.files)
                required = ("future_interaction_grid", "future_interaction_temporal", "valid",
                            "critical_actor_status", "channel_names", "has_causal_actor",
                            "critical_actor_id", "primary_actor_future_frames_found", "num_future_frames")
                for name in required:
                    if name not in keys:
                        R("ERROR", "npz_missing_key", route, frame, f"{source}: missing {name}")
                if "future_interaction_grid" not in keys or "future_interaction_temporal" not in keys:
                    return info
                grid = np.asarray(npz["future_interaction_grid"])
                temporal = np.asarray(npz["future_interaction_temporal"])
                info["grid_shape"] = tuple(grid.shape)
                self.shape_dist[f"{source}:grid={tuple(grid.shape)},time={tuple(temporal.shape)}"] += 1
                if grid.ndim != 3 or grid.shape[0] != 3 or temporal.ndim != 4 or temporal.shape[0] != 2 or temporal.shape[1] != a.horizon or temporal.shape[2:] != grid.shape[1:]:
                    R("ERROR", "npz_shape", route, frame, f"{source}: grid={grid.shape}, temporal={temporal.shape}; expected [3,H,W], [2,{a.horizon},H,W]")
                    return info
                if cost_shape is not None and tuple(grid.shape[-2:]) != tuple(cost_shape):
                    R("ERROR", "grid_costmap_shape_mismatch", route, frame, f"{source}: grid {grid.shape[-2:]} vs costmap {cost_shape}")
                for name, x in (("grid", grid), ("temporal", temporal)):
                    if not np.issubdtype(x.dtype, np.number):
                        R("ERROR", "npz_non_numeric", route, frame, f"{source}: {name} dtype={x.dtype}")
                        return info
                    if not np.isfinite(x).all():
                        R("ERROR", "npz_non_finite", route, frame, f"{source}: {name} has nan/inf")
                        return info
                    vmin, vmax = float(np.min(x)), float(np.max(x))
                    if vmin < -1e-5 or vmax > 1 + 1e-5:
                        R("ERROR", "npz_value_range", route, frame, f"{source}: {name} min={vmin},max={vmax}")
                if "channel_names" in keys:
                    found = tuple(map(str, np.asarray(npz["channel_names"]).tolist()))
                    if found != CHANNEL_NAMES:
                        R("ERROR", "npz_channel_names", route, frame, f"{source}: {found}")
                if "temporal_channel_names" in keys:
                    found = tuple(map(str, np.asarray(npz["temporal_channel_names"]).tolist()))
                    if found != TEMPORAL_NAMES:
                        R("ERROR", "npz_temporal_channel_names", route, frame, f"{source}: {found}")
                val_raw = scalar(npz["valid"]) if "valid" in keys else None
                status_raw = scalar(npz["critical_actor_status"]) if "critical_actor_status" in keys else None
                info["valid"] = bool(val_raw) if val_raw in (0, 1, False, True) else None
                try:
                    status = int(status_raw) if status_raw is not None and status_raw in (-1, 0, 1) else None
                except (TypeError, ValueError):
                    status = None
                info["status"] = status
                info["id"] = str(scalar(npz["critical_actor_id"], "") or "") if "critical_actor_id" in keys else ""
                info["actor_frames"] = scalar(npz["primary_actor_future_frames_found"], None) if "primary_actor_future_frames_found" in keys else None
                if "primary_causal_score" in keys:
                    info["causal_score"] = number(scalar(npz["primary_causal_score"]))
                info["mpp"] = scalar(npz["meters_per_pixel"], None) if "meters_per_pixel" in keys else None
                info["center"] = np.asarray(npz["ego_center"], dtype=float).tolist() if "ego_center" in keys else None
                info["n_frames"] = scalar(npz["num_future_frames"], None) if "num_future_frames" in keys else None
                info["space"] = tuple(grid.shape[1:])
                # Keep C2 only for this frame; used for pixelwise LG/Driving comparison.
                info["temporal_c2"] = temporal[1]
                if info["valid"] is None:
                    R("ERROR", "npz_bad_valid", route, frame, f"{source}: valid={val_raw!r}")
                if status is None:
                    R("ERROR", "npz_bad_status", route, frame, f"{source}: status={status_raw!r}")
                if info["valid"] is not None and status is not None and info["valid"] != (status != -1):
                    R("ERROR", "npz_status_valid_inconsistent", route, frame,
                      f"{source}: valid={info['valid']}, status={status}")
                if "has_causal_actor" in keys and status is not None:
                    declared = bool(scalar(npz["has_causal_actor"], False))
                    if declared != (status == 1):
                        R("ERROR", "npz_has_actor_inconsistent", route, frame, f"{source}: has_causal_actor={declared},status={status}")
                if info["n_frames"] != a.horizon:
                    R("ERROR", "npz_wrong_horizon", route, frame, f"{source}: num_future_frames={info['n_frames']}")
                if "trajectory_source" in keys:
                    actual = str(scalar(npz["trajectory_source"], ""))
                    wanted = "lg" if source == "lg" else "expert"
                    if actual != wanted:
                        R("ERROR", "npz_trajectory_source", route, frame, f"{source}: {actual} != {wanted}")
                if "actor_selection_source" in keys:
                    selected = str(scalar(npz["actor_selection_source"], ""))
                    if selected != "lg_decision_critical_actor":
                        R("ERROR", "npz_actor_selection_source", route, frame, f"{source}: {selected}")
                if "frame" in keys and str(scalar(npz["frame"], "")) != frame:
                    R("ERROR", "npz_frame_mismatch", route, frame, f"{source}: metadata frame mismatch")
                if info["center"] is not None:
                    c = np.asarray(info["center"], dtype=float)
                    if c.shape != (2,) or not np.isfinite(c).all():
                        R("ERROR", "npz_ego_center_malformed", route, frame, f"{source}: ego_center={info['center']}")
                if info["mpp"] is not None and (number(info["mpp"]) is None or float(info["mpp"]) <= 0):
                    R("ERROR", "npz_invalid_meters_per_pixel", route, frame, f"{source}: mpp={info['mpp']}")
                if status == 1 and not info["id"]:
                    R("ERROR", "positive_missing_actor_id", route, frame, f"{source}: confirmed actor without id")
                if status == 0 and info["id"]:
                    R("ERROR", "negative_with_actor_id", route, frame, f"{source}: no actor but id={info['id']}")
                if info["valid"]:
                    for c, name in ((0, "c0"), (1, "c1"), (2, "c2")):
                        self.metric(f"{source}_{name}_mean_pixel_occupancy", float(np.mean(grid[c])))
                        if c < 2 and not np.any(grid[c] > 0):
                            R("WARNING", "valid_empty_ego_channel", route, frame, f"{source}: {name} is all zero")
                    # The code explicitly uses an equal-weight temporal mean.
                    for idx in range(2):
                        if not np.allclose(grid[idx + 1], temporal[idx].mean(axis=0), atol=2e-6, rtol=2e-6):
                            R("ERROR", "temporal_mean_disagrees_channel", route, frame,
                              f"{source}: C{idx+1} != mean(temporal[{idx}])")
                    if status == 0 and (np.any(grid[2] != 0) or np.any(temporal[1] != 0)):
                        R("ERROR", "negative_actor_c2_nonzero", route, frame, f"{source}: status=0 but C2 != 0")
                    if status == 1 and info["actor_frames"] is not None:
                        self.distributions[f"{source}_positive_future_frame_count"][str(info["actor_frames"])] += 1
                        if int(info["actor_frames"]) < max(1, a.horizon // 2):
                            R("REVIEW", "sparse_actor_future_coverage", route, frame,
                              f"{source}: only {info['actor_frames']}/{a.horizon} actor future frames matched")
                    if status == 1 and info["actor_frames"] == 0:
                        R("ERROR", "positive_actor_future_not_found", route, frame, f"{source}: status=1 but actor future found=0")
                    if status == 1 and not np.any(temporal[1] > 0):
                        R("WARNING", "positive_actor_not_in_bev", route, frame,
                          f"{source}: actor frames found but C2 all zero (possibly outside BEV)")
                    if status == 0 and info["actor_frames"] not in (None, 0):
                        R("ERROR", "negative_actor_future_found", route, frame, f"{source}: status=0, actor_frames={info['actor_frames']}")
                    if info["actor_frames"] is not None and not (0 <= int(info["actor_frames"]) <= a.horizon):
                        R("ERROR", "actor_frame_count_out_of_range", route, frame,
                          f"{source}: actor_frames={info['actor_frames']}")
                    # Occupancy overlap is a diagnostic, not a proof of a collision.
                    if status == 1:
                        overlap = int(np.count_nonzero((temporal[0] > 0) & (temporal[1] > 0)))
                        self.metric(f"{source}_temporal_overlap_pixels", overlap)
                        union_mass = int(np.count_nonzero(temporal[1] > 0))
                        if union_mass:
                            self.metric(f"{source}_actor_overlap_fraction", overlap / union_mass)
                    if not np.all((temporal <= 1e-5) | (temporal >= 1 - 1e-5)):
                        R("ERROR", "temporal_not_binary", route, frame, f"{source}: temporal mask not 0/1")
                info["mass"] = float(grid[2].sum() / grid[2].size)
                if source == "lg":
                    self.metric("lg_c2_mean_mass", info["mass"])
                else:
                    self.metric("driving_c2_mean_mass", info["mass"])
                if info["valid"] and status is not None:
                    self.stats[f"{source}_status_{status}"] += 1
                if info["valid"] is False:
                    self.stats[f"{source}_invalid"] += 1
                    if "invalid_reason" in keys:
                        reason = str(scalar(npz["invalid_reason"], "") or "unknown")
                        self.distributions[f"{source}_invalid_reason"][reason] += 1
        except Exception as exc:
            R("ERROR", "npz_read_exception", route, frame,
              f"{source}: {type(exc).__name__}: {str(exc)[:350]}")
        return info

    def audit_frame(self, root, route_path, route, frame, available):
        R = self.reporter.issue
        before = self.reporter.by_level.copy()
        r = {"route": route, "frame": frame}
        elig = frame in available["costmap"] and frame in available["measurements"]
        r["eligible"] = int(elig)
        for kind in FOLDERS:
            has = frame in available[kind]
            r["has_" + kind] = int(has)
            self.stats[f"has_{kind}"] += int(has)
        if elig:
            self.stats["eligible_frames"] += 1
            for kind in ("lg", "lg_grid", "driving_grid", "driving_json"):
                self.stats[f"eligible_with_{kind}"] += int(frame in available[kind])
            for kind in ("lg", "lg_grid", "driving_grid", "driving_json"):
                if frame not in available[kind]:
                    R("ERROR" if self.args.strict_coverage else "WARNING", "missing_" + kind, route, frame,
                      f"eligible costmap+measurement frame has no {kind} output")
        else:
            self.stats["noneligible_frames"] += 1
        cost_shape = None
        if frame in available["costmap"] and (not self.args.skip_costmap_shape or not self.args.skip_costmap_values):
            try:
                cm = np.load(route_path / self.folders["costmap"] / f"{frame}.npy", mmap_mode="r", allow_pickle=False)
                if cm.ndim != 2:
                    R("ERROR", "costmap_shape", route, frame, f"costmap expected [H,W] but got {cm.shape}")
                if not self.args.skip_costmap_shape and cm.ndim >= 2:
                    cost_shape = tuple(cm.shape[-2:])
                if not self.args.skip_costmap_values:
                    if not np.issubdtype(cm.dtype, np.number):
                        R("ERROR", "costmap_non_numeric", route, frame, f"costmap dtype={cm.dtype}")
                    elif not np.isfinite(cm).all():
                        R("ERROR", "costmap_nonfinite", route, frame, "costmap has NaN/Inf")
                    else:
                        cm_min, cm_max = float(np.min(cm)), float(np.max(cm))
                        if cm_min < -1e-5:
                            R("ERROR", "costmap_negative_values", route, frame, f"costmap min={cm_min}")
                        self.metric("costmap_nonzero_fraction", np.count_nonzero(cm) / cm.size)
                        self.metric("costmap_maximum", cm_max)
                del cm
            except Exception as exc:
                R("ERROR", "costmap_read_exception", route, frame, f"{type(exc).__name__}: {exc}")

        lg = {}
        if r["has_lg"]:
            try:
                lg = load_gz_json(route_path / self.folders["lg"] / f"{frame}.json.gz")
                self.stats["lg_json_read_ok"] += 1
                if str(lg.get("frame", "")) != frame:
                    R("ERROR", "lg_frame_mismatch", route, frame, f"LG frame={lg.get('frame')!r}")
                su = as_dict(lg.get("supervision"))
                ca = as_dict(lg.get("causal_analysis"))
                risk = bool(su.get("risk_label_valid", False))
                r["lg_risk_valid"] = int(risk)
                self.stats[f"lg_risk_valid_{int(risk)}"] += 1
                if not risk:
                    self.distributions["lg_selected_intent_invalid"][str(su.get("selected_internal_intent_name", "unknown"))] += 1
                has_causal = ca.get("has_causal_object")
                if not isinstance(has_causal, bool):
                    R("ERROR", "lg_causal_flag_invalid", route, frame, "causal_analysis.has_causal_object is not boolean")
                causal_obj = as_dict(ca.get("causal_object"))
                r["lg_actor_id"] = actor_id(causal_obj) if has_causal is True else ""
                if has_causal is True and not r["lg_actor_id"]:
                    R("ERROR", "lg_causal_actor_missing_id", route, frame, "has_causal_object=True but causal_object.id missing")
                tests = ca.get("object_tests")
                if tests is None:
                    tests = []
                if not isinstance(tests, list):
                    R("ERROR", "lg_causal_tests_malformed", route, frame, "object_tests not list")
                    tests = []
                valid_tests = sum(bool(as_dict(t).get("counterfactual_valid", False)) for t in tests)
                positive_tests = sum(bool(as_dict(t).get("causal_accepted", False)) for t in tests)
                r["lg_test_count"] = len(tests)
                r["lg_valid_test_count"] = valid_tests
                r["lg_positive_test_count"] = positive_tests
                self.metric("lg_tests_per_frame", len(tests))
                self.metric("lg_valid_tests_per_frame", valid_tests)
                self.distributions["lg_test_counts"][str(len(tests))] += 1
                if risk and ca.get("enabled") is False:
                    self.stats["lg_causal_analysis_disabled_with_valid_risk"] += 1
                    R("REVIEW", "valid_label_without_causal_analysis", route, frame,
                      "risk_label_valid=True but causal_analysis.enabled=False; status 0 may lack verification")
                if risk and has_causal is False:
                    if len(tests) == 0:
                        self.stats["negative_lg_no_test"] += 1
                    elif valid_tests == 0:
                        self.stats["negative_lg_all_tests_invalid"] += 1
                        R("REVIEW", "negative_status_without_valid_counterfactual", route, frame,
                          f"{len(tests)} tested actors but no valid counterfactual; status 0 merits review")
                if risk and has_causal is True:
                    revalidation = ca.get("final_revalidation_passed")
                    if revalidation is not True:
                        R("ERROR", "confirmed_actor_without_final_revalidation", route, frame,
                          f"LG causal actor confirmed but final_revalidation_passed={revalidation!r}")
                    if valid_tests == 0:
                        R("ERROR", "positive_actor_without_valid_test", route, frame,
                          "Confirmed causal actor exists but object_tests has no valid counterfactual")
                    selected_id = r["lg_actor_id"]
                    matching = [t for t in tests if actor_id(as_dict(t).get("actor")) == selected_id
                                and bool(as_dict(t).get("counterfactual_valid", False))]
                    if not matching:
                        R("ERROR", "confirmed_actor_not_in_valid_tests", route, frame,
                          f"LG confirmed actor {selected_id} lacks a corresponding valid actor-removal test")
                    final_score = number(ca.get("final_causal_score"))
                    if final_score is None or final_score < 0:
                        R("ERROR", "confirmed_actor_invalid_final_causal_score", route, frame,
                          f"final_causal_score={ca.get('final_causal_score')!r}")
                if risk:
                    ref = as_dict(lg.get("reference"))
                    waypoints = shape_points(su.get("risk_planned_waypoints"), self.args.horizon)
                    expert = shape_points(ref.get("expert_future_waypoints"), self.args.horizon)
                    self.metrics_for_trajectories(r, waypoints, expert)
                    if waypoints is None:
                        R("ERROR", "lg_invalid_waypoint_shape", route, frame,
                          f"risk_planned_waypoints must be [{self.args.horizon},2]")
                    if expert is None:
                        R("WARNING", "lg_missing_expert_waypoints", route, frame,
                          f"expert_future_waypoints unavailable or not [{self.args.horizon},2]")
                    if not route_points_valid(ref.get("selected_reference_route")):
                        R("ERROR", "lg_bad_selected_reference_route", route, frame,
                          "selected_reference_route must contain at least two finite xy points; generator resamples to 20")
                    yaws = su.get("risk_planned_yaws")
                    try:
                        yy = np.asarray(yaws, dtype=float)
                        if yy.shape != (self.args.horizon,) or not np.isfinite(yy).all():
                            R("ERROR", "lg_invalid_yaws", route, frame,
                              f"risk_planned_yaws must be [{self.args.horizon}]")
                    except (ValueError, TypeError):
                        R("ERROR", "lg_invalid_yaws", route, frame, "risk_planned_yaws nonnumeric")
                reconstruct_qa_pairs(lg, frame, route, R, "LG")
                r["lg_spatial_valid"] = spatial_check(lg, "lg", frame, route,
                                                        1 if has_causal else (0 if isinstance(has_causal, bool) else -1),
                                                        R, self.stats)
            except Exception as exc:
                R("ERROR", "lg_json_exception", route, frame, f"{type(exc).__name__}: {str(exc)[:350]}")
                lg = {}

        driving = {}
        if r["has_driving_json"]:
            try:
                driving = load_gz_json(route_path / self.folders["driving_json"] / f"{frame}.json.gz")
                self.stats["driving_json_read_ok"] += 1
                if str(driving.get("frame", "")) != frame:
                    R("ERROR", "driving_json_frame_mismatch", route, frame,
                      f"driving diagnostic frame={driving.get('frame')!r}")
                if driving.get("trajectory_source") != "expert":
                    R("ERROR", "driving_json_trajectory_source", route, frame,
                      f"expected expert, got {driving.get('trajectory_source')!r}")
                if driving.get("actor_selection_source") != "lg_decision_critical_actor":
                    R("ERROR", "driving_json_actor_source", route, frame,
                      f"actor_selection_source={driving.get('actor_selection_source')!r}")
                st = driving.get("critical_actor_status")
                if type(st) is not int or st not in (-1, 0, 1):
                    R("ERROR", "driving_json_status", route, frame, f"critical_actor_status={st!r}")
                    st = None
                r["driving_status"] = st
                if isinstance(driving.get("valid"), bool):
                    if (st != -1) != bool(driving["valid"]):
                        R("ERROR", "driving_json_valid_status", route, frame,
                          f"valid={driving['valid']}, status={st}")
                if st == 1:
                    r["driving_actor_id"] = actor_id(driving.get("primary_actor"))
                    if not r["driving_actor_id"]:
                        R("ERROR", "driving_json_positive_missing_id", route, frame,
                          "status=1 but primary_actor.id missing")
                reconstruct_qa_pairs(driving, frame, route, R, "Driving")
                r["driving_spatial_valid"] = spatial_check(driving, "driving", frame, route, st, R, self.stats)
            except Exception as exc:
                R("ERROR", "driving_json_exception", route, frame,
                  f"{type(exc).__name__}: {str(exc)[:350]}")
                driving = {}

        lg_grid = {}
        if r["has_lg_grid"]:
            lg_grid = self.read_npz(route_path / self.folders["lg_grid"] / f"{frame}.npz",
                                    "lg", route, frame, cost_shape)
        driving_grid = {}
        if r["has_driving_grid"]:
            driving_grid = self.read_npz(route_path / self.folders["driving_grid"] / f"{frame}.npz",
                                         "driving", route, frame, cost_shape)
        r["lg_grid_valid"] = lg_grid.get("valid")
        r["lg_status"] = lg_grid.get("status")
        r["driving_grid_valid"] = driving_grid.get("valid")
        if driving_grid.get("status") is not None:
            r["driving_status"] = driving_grid["status"]
        r["lg_c2_mean_mass"] = lg_grid.get("mass")
        r["driving_c2_mean_mass"] = driving_grid.get("mass")
        r["lg_actor_id"] = r.get("lg_actor_id") or lg_grid.get("id", "")
        r["driving_actor_id"] = r.get("driving_actor_id") or driving_grid.get("id", "")
        self.cross_check(r, lg, driving, lg_grid, driving_grid, route, frame)

        counts_after = self.reporter.by_level
        r["error_count"] = counts_after["ERROR"] - before["ERROR"]
        r["warning_count"] = counts_after["WARNING"] - before["WARNING"]
        r["review_count"] = counts_after["REVIEW"] - before["REVIEW"]
        self.reporter.frame_writer.writerow({k: r.get(k, "") for k in FRAME_COLUMNS})
        return r

    def metrics_for_trajectories(self, row, lg_waypoints, expert_waypoints):
        if lg_waypoints is None or expert_waypoints is None:
            return
        differences = np.linalg.norm(lg_waypoints - expert_waypoints, axis=1)
        ade = float(differences.mean())
        fde = float(differences[-1])
        row["lg_expert_ade_m"] = round(ade, 4)
        row["lg_expert_fde_m"] = round(fde, 4)
        row["expert_distance_m"] = distance_last(expert_waypoints)
        row["lg_distance_m"] = distance_last(lg_waypoints)
        self.metric("lg_expert_ade_m", ade)
        self.metric("lg_expert_fde_m", fde)

    def cross_check(self, row, lg, driving, lg_grid, driving_grid, route, frame):
        R = self.reporter.issue
        lg_ca = as_dict(lg.get("causal_analysis"))
        has_causal = lg_ca.get("has_causal_object")
        lg_status = lg_grid.get("status")
        dstatus = driving_grid.get("status")
        lvalid = lg_grid.get("valid")
        dvalid = driving_grid.get("valid")
        lg_actor = actor_id(as_dict(lg_ca.get("causal_object"))) if has_causal else ""
        for src, record in (("lg", lg_grid), ("driving", driving_grid)):
            if record.get("valid") is True:
                if src == "lg" and not bool(as_dict(lg.get("supervision")).get("risk_label_valid", False)):
                    R("ERROR", "lg_grid_valid_with_invalid_original_label", route, frame,
                      "LG grid valid=True yet LG JSON supervision.risk_label_valid=False")
                should = 1 if has_causal is True else (0 if has_causal is False else None)
                if should is not None and record.get("status") != should:
                    R("ERROR", "npz_status_vs_lg_causal_mismatch", route, frame,
                      f"{src}: critical_actor_status={record.get('status')}, LG has_causal_object={has_causal}")
                if record.get("status") == 1 and record.get("id") != lg_actor:
                    R("ERROR", "actor_id_vs_lg_causal_mismatch", route, frame,
                      f"{src}: id={record.get('id')} LG causal id={lg_actor}")
        if lvalid and dvalid:
            if lg_status != dstatus:
                R("ERROR", "valid_source_status_disagreement", route, frame,
                  f"LG status={lg_status} vs Driving status={dstatus}")
            if lg_status == 1 and dstatus == 1:
                if lg_grid.get("id") != driving_grid.get("id"):
                    R("ERROR", "cross_actor_id_mismatch", route, frame,
                      f"LG ID {lg_grid.get('id')} Driving ID {driving_grid.get('id')}")
            if lg_grid.get("space") == driving_grid.get("space"):
                try:
                    l_mpp = number(lg_grid.get("mpp"))
                    d_mpp = number(driving_grid.get("mpp"))
                    l_center = lg_grid.get("center")
                    d_center = driving_grid.get("center")
                    same_geom = l_mpp is not None and d_mpp is not None and abs(l_mpp - d_mpp) <= 1e-6
                    same_geom = same_geom and l_center is not None and d_center is not None and np.allclose(l_center, d_center, atol=1e-6)
                    if not same_geom:
                        R("WARNING", "cross_bev_geometry_mismatch", route, frame,
                          f"LG mpp/center={l_mpp}/{l_center}, Driving={d_mpp}/{d_center}")
                    elif not self.args.skip_pixelwise_c2 and lg_status == 1 and dstatus == 1:
                        # Same actor, same future boxes, same rasterization configs -> same C2.
                        lc = lg_grid.get("temporal_c2")
                        dc = driving_grid.get("temporal_c2")
                        if lc is not None and dc is not None and not np.allclose(lc, dc, atol=2e-6, rtol=2e-6):
                            mismatched_pixels = int(np.count_nonzero(np.abs(lc - dc) > 2e-6))
                            R("ERROR", "cross_c2_temporal_mismatch", route, frame,
                              f"LG and Driving same actor, C2 temporal differs at {mismatched_pixels} pixels")
                except Exception as exc:
                    R("ERROR", "cross_c2_compare_exception", route, frame, f"{type(exc).__name__}: {exc}")
        elif lvalid != dvalid and lvalid is not None and dvalid is not None:
            self.stats["validity_disagreement_lg_driving"] += 1
            # Different motion-source requirements can genuinely affect validity.
        if dvalid and dstatus == 1 and has_causal is True and driving_grid.get("causal_score") is not None:
            lg_score = number(lg_ca.get("final_causal_score"))
            dscore = driving_grid["causal_score"]
            if lg_score is not None and abs(lg_score - dscore) > 1e-4:
                R("ERROR", "cross_causal_score_mismatch", route, frame,
                  f"LG final causal score {lg_score} differs from Driving {dscore}")
        if driving:
            dst = driving.get("critical_actor_status")
            if dst is not None and dstatus is not None and int(dst) != dstatus:
                R("ERROR", "driving_json_npz_status_mismatch", route, frame,
                  f"driving JSON status={dst}, npz status={dstatus}")
            if dvalid is not None and isinstance(driving.get("valid"), bool) and bool(driving["valid"]) != dvalid:
                R("ERROR", "driving_json_npz_valid_mismatch", route, frame,
                  f"Driving JSON valid={driving['valid']} npz valid={dvalid}")
            if dvalid and dstatus == 1 and actor_id(as_dict(driving.get("primary_actor"))) != driving_grid.get("id"):
                R("ERROR", "driving_json_npz_actor_id_mismatch", route, frame,
                  "Driving diagnostic actor ID does not match npz")
        if driving and lg:
            ql = as_dict(lg.get("core_questions"))
            qd = as_dict(driving.get("core_questions"))
            if dvalid:
                for key in ("attention", "motion_constraint"):
                    if as_dict(ql.get(key)) != as_dict(qd.get(key)):
                        R("ERROR", "driving_q1_q2_not_lg", route, frame,
                          f"Driving {key} is not an exact copy of LG {key}")
            self.language_review(row, lg, driving, route, frame)

    def language_review(self, row, lg, driving, route, frame):
        R = self.reporter.issue
        su = as_dict(lg.get("supervision"))
        ref = as_dict(lg.get("reference"))
        lg_w = shape_points(su.get("risk_planned_waypoints"), self.args.horizon)
        ex_w = shape_points(ref.get("expert_future_waypoints"), self.args.horizon)
        if lg_w is None or ex_w is None:
            return
        lg_dist = distance_last(lg_w)
        exp_dist = distance_last(ex_w)
        ade = row.get("lg_expert_ade_m", None)
        fde = row.get("lg_expert_fde_m", None)
        lg_q1 = qa_answer(lg.get("core_questions"), "attention")
        lg_q2 = qa_answer(lg.get("core_questions"), "motion_constraint")
        d_q3 = qa_answer(driving.get("core_questions"), "driving_response")
        d_q4 = qa_answer(driving.get("core_questions"), "future_motion")
        reasons = []
        if isinstance(ade, float) and isinstance(fde, float) and (ade >= self.args.review_ade_m or fde >= self.args.review_fde_m):
            reasons.append("lg_expert_trajectory_divergence")
        # Narrow trigger: only firm, immediate stop requirements. Not 'prepare to stop'.
        q2_low = lg_q2.lower()
        if exp_dist is not None and exp_dist >= self.args.q2_stop_review_distance_m:
            if "requires a complete stop before" in q2_low or "requires the ego vehicle to remain behind the stop line" in q2_low:
                reasons.append("q2_stop_requirement_vs_expert_progress")
        # Q3/Q4 must describe expert movement, not the selected LG motion.
        d_motion = (d_q3 + " " + d_q4).lower()
        if (exp_dist is not None and exp_dist >= self.args.q2_stop_review_distance_m
                and ("remain stopped" in d_motion or "while remaining stopped" in d_motion)):
            reasons.append("driving_text_stationary_vs_expert_progress")
        if (exp_dist is not None and exp_dist <= self.args.stationary_distance_m
                and re.search(r"\b(?:proceed through|continue through|pass through smoothly)\b", d_motion)):
            reasons.append("driving_text_move_vs_stationary_expert")
        if reasons:
            for reason in reasons:
                R("REVIEW", reason, route, frame,
                  f"LG/expert ADE={ade},FDE={fde},expert_end={exp_dist},LG_end={lg_dist}; manual review needed")
            row_review = {
                "route": route, "frame": frame, "reason": ";".join(reasons),
                "lg_status": row.get("lg_status"), "driving_status": row.get("driving_status"),
                "lg_actor_id": row.get("lg_actor_id"), "driving_actor_id": row.get("driving_actor_id"),
                "expert_distance_m": round(exp_dist, 4) if exp_dist is not None else "",
                "lg_distance_m": round(lg_dist, 4) if lg_dist is not None else "",
                "lg_expert_ade_m": ade, "lg_expert_fde_m": fde,
                "lg_q1": lg_q1, "lg_q2": lg_q2, "driving_q3": d_q3, "driving_q4": d_q4,
            }
            self.reporter.add_review_case(row_review)

    def run(self):
        a = self.args
        root = a.input.expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Input dataset root does not exist: {root}")
        output = a.output.expanduser().resolve()
        if output == root:
            raise ValueError("Report directory must not equal dataset root")
        output.mkdir(parents=True, exist_ok=True)
        print(f"[Audit] root={root}\n[Audit] output={output}")
        routes = discover_routes(root, self.folders)
        if not routes:
            raise RuntimeError(f"No route directories discovered under {root}; check input and folder names.")
        self.routes_seen = len(routes)
        print(f"[Audit] found {len(routes)} route directories", flush=True)
        for route_path in routes:
            self.routes_processed += 1
            route = str(route_path.relative_to(root)) if route_path != root else "."
            available = {
                "measurements": frames_in(route_path, self.folders["measurements"], ".json.gz"),
                "costmap": frames_in(route_path, self.folders["costmap"], ".npy"),
                "lg": frames_in(route_path, self.folders["lg"], ".json.gz"),
                "lg_grid": frames_in(route_path, self.folders["lg_grid"], ".npz"),
                "driving_grid": frames_in(route_path, self.folders["driving_grid"], ".npz"),
                "driving_json": frames_in(route_path, self.folders["driving_json"], ".json.gz"),
            }
            all_names = sorted(set().union(*available.values()))
            route_counts = Counter()
            for frame in all_names:
                if a.max_frames and self.total_frames >= a.max_frames:
                    self.stopped = True
                    break
                self.total_frames += 1
                row = self.audit_frame(root, route_path, route, frame, available)
                route_counts["eligible_frames"] += int(row["eligible"])
                for kind in available:
                    route_counts[kind] += int(row["has_" + kind])
                for sev, name in (("ERROR", "errors"), ("WARNING", "warnings"), ("REVIEW", "reviews")):
                    route_counts[name] += int(row.get(name[:-1] + "_count", 0))
                if row.get("lg_grid_valid") is True:
                    route_counts["lg_grid_valid"] += 1
                if row.get("driving_grid_valid") is True:
                    route_counts["driving_grid_valid"] += 1
                for source in ("lg", "driving"):
                    status = row.get("lg_status") if source == "lg" else row.get("driving_status")
                    if status in (-1, 0, 1):
                        route_counts[f"{source}_status_{status}"] += 1
                if a.progress_every and self.total_frames % a.progress_every == 0:
                    print(f"[Audit] {self.total_frames:,} frames | ERROR={self.reporter.by_level['ERROR']:,} "
                          f"WARNING={self.reporter.by_level['WARNING']:,} REVIEW={self.reporter.by_level['REVIEW']:,}", flush=True)
            self.reporter.route_writer.writerow({"route": route, **{k: route_counts.get(k, 0) for k in ROUTE_COLUMNS if k != "route"}})
            if self.stopped:
                break
        self.reporter.close()
        self.save_summary(root, output)
        sev = self.reporter.by_level
        print(f"[Done] routes={self.routes_seen}, checked_frames={self.total_frames:,}, "
              f"ERROR={sev['ERROR']:,}, WARNING={sev['WARNING']:,}, REVIEW={sev['REVIEW']:,}")
        print(f"[Done] reports: {output}/summary.json and summary.md (plus four CSV files)")
        if sev["ERROR"] > 0:
            return 2
        return 0

    def save_summary(self, root, output):
        distributions = {k: dict(v.most_common()) for k, v in self.distributions.items()}
        numeric = {}
        for k, v in self.numeric.items():
            numeric[k] = {"count": v["n"], "mean": (v["sum"] / v["n"] if v["n"] else None), "max": v["max"]}
        eligible = self.stats["eligible_frames"]
        coverage_rates = {
            k: (self.stats[f"eligible_with_{k}"] / eligible if eligible else None)
            for k in ("lg", "lg_grid", "driving_grid", "driving_json")
        }
        valid_rates = {
            k: (
                (self.stats[f"{k}_status_0"] + self.stats[f"{k}_status_1"])
                / (self.stats[f"{k}_status_0"] + self.stats[f"{k}_status_1"] + self.stats[f"{k}_invalid"])
                if (self.stats[f"{k}_status_0"] + self.stats[f"{k}_status_1"] + self.stats[f"{k}_invalid"]) else None
            ) for k in ("lg", "driving")
        }
        summary = {
            "audit_version": CODE_VERSION,
            "source_code_commit": SOURCE_COMMIT,
            "root": str(root),
            "discovered_routes": self.routes_seen,
            "checked_routes": self.routes_processed,
            "checked_frames": self.total_frames,
            "partial_scan": self.stopped,
            "expected_horizon": self.args.horizon,
            "counts": dict(self.stats),
            "eligible_coverage_rates": coverage_rates,
            "valid_grid_rates": valid_rates,
            "issue_counts": dict(self.reporter.by_level),
            "issue_codes": dict(self.reporter.counters.most_common()),
            "issue_examples": dict(self.reporter.examples),
            "issue_rows_written": self.reporter.written,
            "issue_rows_truncated": sum(self.reporter.by_level.values()) > self.reporter.written,
            "review_case_rows_written": self.reporter.review_written,
            "shape_distribution": dict(self.shape_dist.most_common()),
            "distributions": distributions,
            "numeric_metrics": numeric,
            "scope_notes": [
                "Coverage baseline is intersection(costmap/*.npy, measurements/*.json.gz).",
                "Positive status=1 requires identified critical actor and usable selected C2 future.",
                "Negative status=0 is an LG producer decision; no-tested-candidate cases are counted separately.",
                "LG and Driving C0/C1 use different trajectory sources by design and need not match.",
                "C2 should agree for same valid actor with identical geometry/configuration.",
                "Language/expert mismatch cases are REVIEW only; no language model is used.",
                "No input dataset artifacts were modified.",
            ],
            "options": {**vars(self.args), "input": str(self.args.input), "output": str(self.args.output)},
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        md = ["# SimLingo LG/Driving full label audit", "",
              f"Source code commit: `{SOURCE_COMMIT}`", "",
              f"Checked routes: **{self.routes_seen:,}**; frames: **{self.total_frames:,}**; partial scan: **{self.stopped}**", "",
              "## Verdict counts", "",
              "| Severity | Events |", "|---|---:|"]
        for s in ("ERROR", "WARNING", "REVIEW"):
            md.append(f"| {s} | {self.reporter.by_level[s]:,} |")
        md.extend(["", "## Coverage", "", "| Source | Files found |", "|---|---:|"])
        for src in ("costmap", "measurements", "lg", "lg_grid", "driving_grid", "driving_json"):
            md.append(f"| {src} | {self.stats['has_' + src]:,} |")
        md.extend(["", "| Output stage | Coverage among eligible frames |", "|---|---:|"])
        for src in ("lg", "lg_grid", "driving_grid", "driving_json"):
            rate = coverage_rates[src]
            md.append(f"| {src} | {rate:.2%} |" if rate is not None else f"| {src} | N/A |")
        md.extend(["", f"Eligible costmap+measurement frames: **{self.stats['eligible_frames']:,}**", "",
                   "## Critical-actor status (among valid grid files)", "",
                   "| Source | 1 | 0 | invalid grids |", "|---|---:|---:|---:|"])
        for src in ("lg", "driving"):
            md.append(f"| {src} | {self.stats[f'{src}_status_1']:,} | {self.stats[f'{src}_status_0']:,} | {self.stats[f'{src}_invalid']:,} |")
        md.extend(["", "| Grid source | Valid grid fraction |", "|---|---:|"])
        for src in ("lg", "driving"):
            rate = valid_rates[src]
            md.append(f"| {src} | {rate:.2%} |" if rate is not None else f"| {src} | N/A |")
        md.extend(["", "## Top issues", "", "| Level/code | Events |", "|---|---:|"])
        for code, n in self.reporter.counters.most_common(30):
            md.append(f"| `{code}` | {n:,} |")
        md.extend(["", "## LG counterfactual-evidence coverage", "",
                   f"- Valid, no-causal-actor frames without any recorded actor test: **{self.stats['negative_lg_no_test']:,}**.",
                   f"- Valid, no-causal-actor frames with tests but zero valid counterfactuals: **{self.stats['negative_lg_all_tests_invalid']:,}**.",
                   "- These counts quantify evidence coverage; they are not automatic proof that label 0 is wrong.",
                   "", "## Review guidance", "",
                   "- ERROR: inspect and fix structural or contradictory labels before training.",
                   "- WARNING: inspect coverage, invalidity reasons, and usable supervision proportion.",
                   "- REVIEW: manually sample Q1/Q2 versus expert trajectories, plus LG/expert action differences.",
                   "- The test does not independently establish causal correctness or language-semantic correctness.", ""])
        (output / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    args = parse_args()
    if args.horizon <= 0 or args.progress_every < 0 or args.max_issues < 0 or args.max_review_cases < 0 or args.max_frames < 0:
        print("Invalid argument: expected horizon>0 and non-negative counters", file=sys.stderr)
        return 2
    auditor = LabelAuditor(args)
    try:
        return auditor.run()
    except Exception as exc:
        try:
            auditor.reporter.close()
        except Exception:
            pass
        print(f"[Fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
