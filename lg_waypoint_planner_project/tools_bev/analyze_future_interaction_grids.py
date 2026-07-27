#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 LG 五通道 future_interaction_grid 标签。

支持检查：
1. language_grounded_waypoints 与 future_interaction_grids 的文件覆盖率；
2. .npz 是否可读取，shape、channel_names、数值范围是否合法；
3. valid=True / valid=False 数量；
4. 五个通道的非零样本比例、平均占用比例、均值和最大值；
5. actor 元数据与通道内容是否存在明显矛盾；
6. 下采样到指定分辨率后，各通道是否变成全零；
7. 保存 summary.json 和 per_file.csv，便于进一步分析。

当前五通道顺序必须与 generate_future_interaction_grids.py 一致：
C0: selected route ego-footprint occupancy
C1: future ego-footprint occupancy
C2: primary causal actor future footprint occupancy
C3: time-aligned primary future interaction
C4: secondary actor future footprint occupancy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EXPECTED_CHANNEL_NAMES: Tuple[str, ...] = (
    "selected_route_ego_footprint_occupancy",
    "future_ego_footprint_occupancy",
    "primary_causal_actor_future_footprint_occupancy",
    "time_aligned_primary_future_interaction",
    "secondary_actor_future_footprint_occupancy",
)

SHORT_CHANNEL_NAMES: Tuple[str, ...] = (
    "C0_route",
    "C1_ego_future",
    "C2_primary_actor",
    "C3_interaction",
    "C4_secondary_actor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 future_interaction_grids 五通道标签质量。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/kemove/ll/simlingo_liulei/database/simlingo_v2_2026_07_21_16_16_03/data/simlingo"),
        help=(
            "数据集根目录、单条 route 目录，或 future_interaction_grids 目录。"
        ),
    )
    parser.add_argument(
        "--grid-folder",
        type=str,
        default="future_interaction_grids",
        help="结构化未来世界标签文件夹名称。",
    )
    parser.add_argument(
        "--lg-folder",
        type=str,
        default="language_grounded_waypoints",
        help="LG JSON 标签文件夹名称，用于统计文件覆盖率。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./future_interaction_grid_statistics"),
        help="统计结果保存目录。",
    )
    parser.add_argument(
        "--downsample-sizes",
        type=int,
        nargs="*",
        default=[128, 64],
        help="检查下采样后标签是否消失的正方形尺寸，例如 128 64。",
    )
    parser.add_argument(
        "--nonzero-threshold",
        type=float,
        default=1e-6,
        help="判断像素是否非零的阈值。",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="仅分析前 N 个 .npz；默认分析全部。",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="每处理多少个文件打印一次进度；设置为0关闭。",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="发现读取错误、非法shape、非法数值或通道顺序错误时返回非零退出码。",
    )
    return parser.parse_args()


def scalar_to_python(
    payload: "np.lib.npyio.NpzFile",
    key: str,
    default: Any = None,
) -> Any:
    if key not in payload.files:
        return default
    array = np.asarray(payload[key]).reshape(-1)
    if array.size == 0:
        return default
    value = array[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def decode_string_array(array: np.ndarray) -> Tuple[str, ...]:
    result: List[str] = []
    for item in np.asarray(array).reshape(-1).tolist():
        if isinstance(item, bytes):
            result.append(item.decode("utf-8", errors="replace"))
        else:
            result.append(str(item))
    return tuple(result)


def frame_from_lg_path(path: Path) -> str:
    suffix = ".json.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected LG label file name: {path.name}")
    return path.name[: -len(suffix)]


def discover_route_dirs(
    input_path: Path,
    grid_folder: str,
    lg_folder: str,
) -> List[Path]:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        raise ValueError("--input 必须是目录，而不是文件。")

    route_dirs = set()

    if input_path.name in {grid_folder, lg_folder}:
        route_dirs.add(input_path.parent)

    if (input_path / grid_folder).is_dir() or (input_path / lg_folder).is_dir():
        route_dirs.add(input_path)

    for folder_name in (grid_folder, lg_folder):
        for folder in input_path.rglob(folder_name):
            if folder.is_dir():
                route_dirs.add(folder.parent)

    return sorted(route_dirs, key=lambda path: str(path))


def area_downsample(channel: np.ndarray, target_size: int) -> np.ndarray:
    """优先使用纯 NumPy 面积平均；非整数缩放时再尝试 OpenCV。"""
    if target_size <= 0:
        raise ValueError(f"Downsample size must be positive: {target_size}")

    height, width = channel.shape
    if height == target_size and width == target_size:
        return channel.astype(np.float32, copy=False)

    if (
        height % target_size == 0
        and width % target_size == 0
        and target_size <= height
        and target_size <= width
    ):
        block_h = height // target_size
        block_w = width // target_size
        return (
            channel.reshape(
                target_size,
                block_h,
                target_size,
                block_w,
            )
            .mean(axis=(1, 3), dtype=np.float64)
            .astype(np.float32)
        )

    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "当前shape无法使用纯NumPy整数面积下采样，且未安装OpenCV。"
        ) from exc

    return cv2.resize(
        channel.astype(np.float32),
        (target_size, target_size),
        interpolation=cv2.INTER_AREA,
    )


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def new_channel_accumulator(
    downsample_sizes: Sequence[int],
) -> Dict[str, Any]:
    return {
        "sample_count": 0,
        "nonzero_sample_count": 0,
        "sum_nonzero_pixel_ratio": 0.0,
        "sum_mean_value": 0.0,
        "positive_value_sum": 0.0,
        "positive_pixel_count": 0,
        "global_max": float("-inf"),
        "downsample": {
            str(size): {
                "nonzero_sample_count": 0,
                "original_nonzero_became_zero_count": 0,
                "sum_nonzero_pixel_ratio": 0.0,
            }
            for size in downsample_sizes
        },
    }


def update_channel_accumulator(
    accumulator: Dict[str, Any],
    channel: np.ndarray,
    threshold: float,
    downsample_sizes: Sequence[int],
) -> Dict[str, Any]:
    positive = channel > threshold
    nonzero = bool(positive.any())
    nonzero_ratio = float(positive.mean())
    mean_value = float(channel.mean())
    max_value = float(channel.max())

    accumulator["sample_count"] += 1
    accumulator["nonzero_sample_count"] += int(nonzero)
    accumulator["sum_nonzero_pixel_ratio"] += nonzero_ratio
    accumulator["sum_mean_value"] += mean_value
    accumulator["positive_value_sum"] += float(channel[positive].sum())
    accumulator["positive_pixel_count"] += int(positive.sum())
    accumulator["global_max"] = max(
        float(accumulator["global_max"]),
        max_value,
    )

    downsample_result: Dict[str, Any] = {}
    for size in downsample_sizes:
        resized = area_downsample(channel, size)
        resized_positive = resized > threshold
        resized_nonzero = bool(resized_positive.any())
        resized_nonzero_ratio = float(resized_positive.mean())

        size_accumulator = accumulator["downsample"][str(size)]
        size_accumulator["nonzero_sample_count"] += int(resized_nonzero)
        size_accumulator["original_nonzero_became_zero_count"] += int(
            nonzero and not resized_nonzero
        )
        size_accumulator[
            "sum_nonzero_pixel_ratio"
        ] += resized_nonzero_ratio

        downsample_result[str(size)] = {
            "nonzero": resized_nonzero,
            "nonzero_pixel_ratio": resized_nonzero_ratio,
            "max": float(resized.max()),
        }

    return {
        "nonzero": nonzero,
        "nonzero_pixel_count": int(positive.sum()),
        "nonzero_pixel_ratio": nonzero_ratio,
        "mean": mean_value,
        "positive_mean": (
            float(channel[positive].mean()) if nonzero else 0.0
        ),
        "max": max_value,
        "downsample": downsample_result,
    }


def finalize_channel_accumulator(
    accumulator: Dict[str, Any],
) -> Dict[str, Any]:
    sample_count = int(accumulator["sample_count"])
    positive_pixel_count = int(accumulator["positive_pixel_count"])
    global_max = float(accumulator["global_max"])
    if not math.isfinite(global_max):
        global_max = 0.0

    result = {
        "sample_count": sample_count,
        "nonzero_sample_count": int(
            accumulator["nonzero_sample_count"]
        ),
        "nonzero_sample_ratio": safe_ratio(
            int(accumulator["nonzero_sample_count"]),
            sample_count,
        ),
        "mean_nonzero_pixel_ratio": (
            float(accumulator["sum_nonzero_pixel_ratio"])
            / sample_count
            if sample_count > 0
            else 0.0
        ),
        "mean_value": (
            float(accumulator["sum_mean_value"]) / sample_count
            if sample_count > 0
            else 0.0
        ),
        "mean_positive_value": (
            float(accumulator["positive_value_sum"])
            / positive_pixel_count
            if positive_pixel_count > 0
            else 0.0
        ),
        "global_max": global_max,
        "downsample": {},
    }

    for size, size_accumulator in accumulator["downsample"].items():
        result["downsample"][size] = {
            "nonzero_sample_count": int(
                size_accumulator["nonzero_sample_count"]
            ),
            "nonzero_sample_ratio": safe_ratio(
                int(size_accumulator["nonzero_sample_count"]),
                sample_count,
            ),
            "original_nonzero_became_zero_count": int(
                size_accumulator[
                    "original_nonzero_became_zero_count"
                ]
            ),
            "mean_nonzero_pixel_ratio": (
                float(
                    size_accumulator["sum_nonzero_pixel_ratio"]
                )
                / sample_count
                if sample_count > 0
                else 0.0
            ),
        }

    return result


def print_channel_table(
    title: str,
    channel_summary: Dict[str, Dict[str, Any]],
    downsample_sizes: Sequence[int],
) -> None:
    print(f"\n{title}")
    header = (
        f"{'通道':<22}"
        f"{'样本':>8}"
        f"{'非零样本':>12}"
        f"{'非零比例':>12}"
        f"{'平均占用率':>14}"
        f"{'均值':>12}"
        f"{'正值均值':>12}"
        f"{'最大值':>12}"
    )
    print(header)
    print("-" * len(header))
    for channel_name in SHORT_CHANNEL_NAMES:
        item = channel_summary[channel_name]
        print(
            f"{channel_name:<22}"
            f"{item['sample_count']:>8d}"
            f"{item['nonzero_sample_count']:>12d}"
            f"{item['nonzero_sample_ratio']:>12.4%}"
            f"{item['mean_nonzero_pixel_ratio']:>14.6%}"
            f"{item['mean_value']:>12.6f}"
            f"{item['mean_positive_value']:>12.6f}"
            f"{item['global_max']:>12.6f}"
        )

    for size in downsample_sizes:
        print(f"\n下采样到 {size}×{size}")
        for channel_name in SHORT_CHANNEL_NAMES:
            item = channel_summary[channel_name]["downsample"][str(size)]
            print(
                f"  {channel_name:<20}"
                f"非零样本={item['nonzero_sample_count']:>7d} "
                f"比例={item['nonzero_sample_ratio']:>9.4%} "
                "原本非零但下采样后全零="
                f"{item['original_nonzero_became_zero_count']:>7d}"
            )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    if args.nonzero_threshold < 0.0:
        raise ValueError("--nonzero-threshold 不能小于0。")

    downsample_sizes = tuple(
        sorted(set(int(size) for size in args.downsample_sizes), reverse=True)
    )

    route_dirs = discover_route_dirs(
        args.input,
        args.grid_folder,
        args.lg_folder,
    )
    if not route_dirs:
        raise RuntimeError(
            "没有找到包含 LG 标签或 future_interaction_grids 的 route 目录。"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    coverage = {
        "route_count": len(route_dirs),
        "lg_label_count": 0,
        "grid_file_count": 0,
        "matched_lg_grid_count": 0,
        "missing_grid_for_lg_count": 0,
        "orphan_grid_without_lg_count": 0,
    }

    grid_jobs: List[Tuple[Path, Path, bool]] = []
    missing_rows: List[Dict[str, Any]] = []

    for route_dir in route_dirs:
        lg_dir = route_dir / args.lg_folder
        grid_dir = route_dir / args.grid_folder

        lg_paths = sorted(lg_dir.glob("*.json.gz")) if lg_dir.is_dir() else []
        grid_paths = sorted(grid_dir.glob("*.npz")) if grid_dir.is_dir() else []

        lg_frames = {frame_from_lg_path(path): path for path in lg_paths}
        grid_frames = {path.stem: path for path in grid_paths}

        matched_frames = sorted(set(lg_frames) & set(grid_frames))
        missing_frames = sorted(set(lg_frames) - set(grid_frames))
        orphan_frames = sorted(set(grid_frames) - set(lg_frames))

        coverage["lg_label_count"] += len(lg_paths)
        coverage["grid_file_count"] += len(grid_paths)
        coverage["matched_lg_grid_count"] += len(matched_frames)
        coverage["missing_grid_for_lg_count"] += len(missing_frames)
        coverage["orphan_grid_without_lg_count"] += len(orphan_frames)

        for frame in missing_frames:
            missing_rows.append(
                {
                    "route": str(route_dir),
                    "frame": frame,
                    "status": "missing_grid_for_lg",
                    "lg_path": str(lg_frames[frame]),
                    "grid_path": "",
                }
            )

        for frame, grid_path in grid_frames.items():
            grid_jobs.append(
                (
                    route_dir,
                    grid_path,
                    frame in lg_frames,
                )
            )

    grid_jobs.sort(key=lambda item: str(item[1]))
    if args.max_files is not None:
        grid_jobs = grid_jobs[: max(int(args.max_files), 0)]

    shape_counter: Counter = Counter()
    invalid_reason_counter: Counter = Counter()
    secondary_source_counter: Counter = Counter()

    problem_counts: Counter = Counter()
    load_error_messages: List[Dict[str, str]] = []

    all_channel_accumulators = {
        channel_name: new_channel_accumulator(downsample_sizes)
        for channel_name in SHORT_CHANNEL_NAMES
    }
    valid_channel_accumulators = {
        channel_name: new_channel_accumulator(downsample_sizes)
        for channel_name in SHORT_CHANNEL_NAMES
    }

    loaded_count = 0
    valid_count = 0
    invalid_count = 0
    per_file_rows: List[Dict[str, Any]] = list(missing_rows)

    for file_index, (route_dir, grid_path, matched_lg) in enumerate(
        grid_jobs,
        start=1,
    ):
        row: Dict[str, Any] = {
            "route": str(route_dir),
            "frame": grid_path.stem,
            "status": "grid",
            "lg_matched": matched_lg,
            "grid_path": str(grid_path),
        }

        try:
            with np.load(str(grid_path), allow_pickle=False) as payload:
                if "future_interaction_grid" not in payload.files:
                    raise KeyError("missing key: future_interaction_grid")

                grid = np.asarray(
                    payload["future_interaction_grid"],
                    dtype=np.float32,
                )
                valid = bool(
                    scalar_to_python(payload, "valid", False)
                )

                channel_names = (
                    decode_string_array(payload["channel_names"])
                    if "channel_names" in payload.files
                    else tuple()
                )

                has_causal_actor = bool(
                    scalar_to_python(
                        payload,
                        "has_causal_actor",
                        False,
                    )
                )
                has_secondary_actor = bool(
                    scalar_to_python(
                        payload,
                        "has_secondary_actor",
                        False,
                    )
                )
                primary_frames = int(
                    scalar_to_python(
                        payload,
                        "primary_actor_future_frames_found",
                        0,
                    )
                )
                secondary_frames = int(
                    scalar_to_python(
                        payload,
                        "secondary_actor_future_frames_found",
                        0,
                    )
                )
                invalid_reason = str(
                    scalar_to_python(
                        payload,
                        "invalid_reason",
                        "",
                    )
                    or ""
                )
                secondary_source = str(
                    scalar_to_python(
                        payload,
                        "secondary_actor_source",
                        "",
                    )
                    or ""
                )

            loaded_count += 1
            valid_count += int(valid)
            invalid_count += int(not valid)

            shape = tuple(int(value) for value in grid.shape)
            shape_counter[str(shape)] += 1
            invalid_reason_counter[invalid_reason or "<empty>"] += int(
                not valid
            )
            secondary_source_counter[
                secondary_source or "<empty>"
            ] += 1

            shape_ok = grid.ndim == 3 and grid.shape[0] == 5
            channel_order_ok = (
                channel_names == EXPECTED_CHANNEL_NAMES
            )
            finite_ok = bool(np.isfinite(grid).all())
            range_ok = bool(
                finite_ok
                and np.all(grid >= 0.0)
                and np.all(grid <= 1.0)
            )

            row.update(
                {
                    "valid": valid,
                    "shape": str(shape),
                    "shape_ok": shape_ok,
                    "channel_names": "|".join(channel_names),
                    "channel_order_ok": channel_order_ok,
                    "finite_ok": finite_ok,
                    "range_ok": range_ok,
                    "has_causal_actor": has_causal_actor,
                    "has_secondary_actor": has_secondary_actor,
                    "primary_actor_future_frames_found": primary_frames,
                    "secondary_actor_future_frames_found": secondary_frames,
                    "invalid_reason": invalid_reason,
                    "secondary_actor_source": secondary_source,
                }
            )

            if not shape_ok:
                problem_counts["invalid_shape"] += 1
                per_file_rows.append(row)
                continue

            if not channel_order_ok:
                problem_counts["channel_order_mismatch"] += 1
            if not finite_ok:
                problem_counts["non_finite_values"] += 1
                per_file_rows.append(row)
                continue
            if not range_ok:
                problem_counts["out_of_range_values"] += 1

            channel_file_stats: List[Dict[str, Any]] = []
            for channel_index, channel_name in enumerate(
                SHORT_CHANNEL_NAMES
            ):
                channel = grid[channel_index]
                all_stats = update_channel_accumulator(
                    all_channel_accumulators[channel_name],
                    channel,
                    args.nonzero_threshold,
                    downsample_sizes,
                )
                if valid:
                    update_channel_accumulator(
                        valid_channel_accumulators[channel_name],
                        channel,
                        args.nonzero_threshold,
                        downsample_sizes,
                    )

                channel_file_stats.append(all_stats)

                prefix = channel_name
                row[f"{prefix}_nonzero"] = all_stats["nonzero"]
                row[
                    f"{prefix}_nonzero_pixel_count"
                ] = all_stats["nonzero_pixel_count"]
                row[
                    f"{prefix}_nonzero_pixel_ratio"
                ] = all_stats["nonzero_pixel_ratio"]
                row[f"{prefix}_mean"] = all_stats["mean"]
                row[
                    f"{prefix}_positive_mean"
                ] = all_stats["positive_mean"]
                row[f"{prefix}_max"] = all_stats["max"]

                for size in downsample_sizes:
                    resized_stats = all_stats["downsample"][str(size)]
                    row[
                        f"{prefix}_{size}_nonzero"
                    ] = resized_stats["nonzero"]
                    row[
                        f"{prefix}_{size}_nonzero_pixel_ratio"
                    ] = resized_stats["nonzero_pixel_ratio"]
                    row[
                        f"{prefix}_{size}_max"
                    ] = resized_stats["max"]

            c2_nonzero = channel_file_stats[2]["nonzero"]
            c3_nonzero = channel_file_stats[3]["nonzero"]
            c4_nonzero = channel_file_stats[4]["nonzero"]

            if (not has_causal_actor) and (c2_nonzero or c3_nonzero):
                problem_counts[
                    "no_causal_actor_but_c2_or_c3_nonzero"
                ] += 1
                row[
                    "metadata_problem_no_causal_actor_but_c2_or_c3_nonzero"
                ] = True
            else:
                row[
                    "metadata_problem_no_causal_actor_but_c2_or_c3_nonzero"
                ] = False

            if (not has_secondary_actor) and c4_nonzero:
                problem_counts[
                    "no_secondary_actor_but_c4_nonzero"
                ] += 1
                row[
                    "metadata_problem_no_secondary_actor_but_c4_nonzero"
                ] = True
            else:
                row[
                    "metadata_problem_no_secondary_actor_but_c4_nonzero"
                ] = False

            if has_causal_actor and primary_frames == 0 and valid:
                problem_counts[
                    "causal_actor_zero_future_frames_but_valid"
                ] += 1
                row[
                    "metadata_problem_causal_actor_zero_frames_but_valid"
                ] = True
            else:
                row[
                    "metadata_problem_causal_actor_zero_frames_but_valid"
                ] = False

            if (not valid) and not invalid_reason:
                problem_counts[
                    "invalid_label_without_reason"
                ] += 1
                row[
                    "metadata_problem_invalid_without_reason"
                ] = True
            else:
                row[
                    "metadata_problem_invalid_without_reason"
                ] = False

            if valid and not channel_file_stats[0]["nonzero"]:
                problem_counts["valid_but_c0_zero"] += 1
                row["problem_valid_but_c0_zero"] = True
            else:
                row["problem_valid_but_c0_zero"] = False

            if valid and not channel_file_stats[1]["nonzero"]:
                problem_counts["valid_but_c1_zero"] += 1
                row["problem_valid_but_c1_zero"] = True
            else:
                row["problem_valid_but_c1_zero"] = False

        except Exception as exc:  # pylint: disable=broad-except
            problem_counts["load_error"] += 1
            error_text = f"{type(exc).__name__}: {exc}"
            row.update(
                {
                    "status": "load_error",
                    "error": error_text,
                }
            )
            load_error_messages.append(
                {
                    "path": str(grid_path),
                    "error": error_text,
                }
            )

        per_file_rows.append(row)

        if (
            args.progress_every > 0
            and file_index % args.progress_every == 0
        ):
            print(
                f"[进度] {file_index}/{len(grid_jobs)}，"
                f"已成功读取 {loaded_count}，"
                f"读取错误 {problem_counts['load_error']}"
            )

    all_channel_summary = {
        channel_name: finalize_channel_accumulator(
            all_channel_accumulators[channel_name]
        )
        for channel_name in SHORT_CHANNEL_NAMES
    }
    valid_channel_summary = {
        channel_name: finalize_channel_accumulator(
            valid_channel_accumulators[channel_name]
        )
        for channel_name in SHORT_CHANNEL_NAMES
    }

    coverage["lg_grid_coverage_ratio"] = safe_ratio(
        coverage["matched_lg_grid_count"],
        coverage["lg_label_count"],
    )

    summary = {
        "input": str(args.input.expanduser().resolve()),
        "grid_folder": args.grid_folder,
        "lg_folder": args.lg_folder,
        "expected_channel_names": list(EXPECTED_CHANNEL_NAMES),
        "short_channel_names": list(SHORT_CHANNEL_NAMES),
        "nonzero_threshold": args.nonzero_threshold,
        "downsample_sizes": list(downsample_sizes),
        "coverage": coverage,
        "analyzed_grid_file_count": len(grid_jobs),
        "loaded_grid_file_count": loaded_count,
        "load_error_count": int(problem_counts["load_error"]),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "valid_ratio_among_loaded": safe_ratio(
            valid_count,
            loaded_count,
        ),
        "shape_counts": dict(shape_counter),
        "invalid_reason_counts": dict(invalid_reason_counter),
        "secondary_actor_source_counts": dict(
            secondary_source_counter
        ),
        "problem_counts": dict(problem_counts),
        "all_loaded_channel_statistics": all_channel_summary,
        "valid_only_channel_statistics": valid_channel_summary,
        "load_errors": load_error_messages,
    }

    summary_path = args.output_dir / "summary.json"
    per_file_path = args.output_dir / "per_file.csv"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(per_file_path, per_file_rows)

    print("\n================ 文件覆盖率 ================")
    print(f"route数量：{coverage['route_count']}")
    print(f"LG JSON数量：{coverage['lg_label_count']}")
    print(f".npz数量：{coverage['grid_file_count']}")
    print(
        "LG与.npz匹配数量："
        f"{coverage['matched_lg_grid_count']}"
    )
    print(
        "LG缺少.npz数量："
        f"{coverage['missing_grid_for_lg_count']}"
    )
    print(
        "无对应LG的孤立.npz数量："
        f"{coverage['orphan_grid_without_lg_count']}"
    )
    print(
        "LG标签覆盖率："
        f"{coverage['lg_grid_coverage_ratio']:.4%}"
    )

    print("\n================ 基础合法性 ================")
    print(f"实际分析.npz数量：{len(grid_jobs)}")
    print(f"成功读取：{loaded_count}")
    print(f"读取错误：{problem_counts['load_error']}")
    print(f"valid=True：{valid_count}")
    print(f"valid=False：{invalid_count}")
    print(
        "有效比例："
        f"{safe_ratio(valid_count, loaded_count):.4%}"
    )
    print(f"shape分布：{dict(shape_counter)}")
    print(f"问题统计：{dict(problem_counts)}")

    print_channel_table(
        "================ 全部成功读取样本：通道统计 ================",
        all_channel_summary,
        downsample_sizes,
    )
    print_channel_table(
        "================ 仅 valid=True 样本：通道统计 ================",
        valid_channel_summary,
        downsample_sizes,
    )

    print("\n================ 输出文件 ================")
    print(f"汇总JSON：{summary_path.resolve()}")
    print(f"逐文件CSV：{per_file_path.resolve()}")

    hard_error_keys = (
        "load_error",
        "invalid_shape",
        "channel_order_mismatch",
        "non_finite_values",
        "out_of_range_values",
    )
    hard_error_count = sum(
        int(problem_counts[key]) for key in hard_error_keys
    )

    if args.fail_on_error and hard_error_count > 0:
        print(
            f"\n发现 {hard_error_count} 个基础合法性问题，"
            "--fail-on-error 已启用。",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
