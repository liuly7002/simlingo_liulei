#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import time
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "language_grounded_waypoint.yaml"
)

sys.path.insert(
    0,
    str(CURRENT_FILE.parent),
)

from lg_waypoint_planner.config_utils import load_yaml
from lg_waypoint_planner.logger import setup_logger, LOGGER
from lg_waypoint_planner.processor import process_one_frame

from run_language_grounded_waypoint_planner_keyframes import (
    load_keyframe_manifest,
)


def resolve_config_path():
    if len(sys.argv) >= 2:
        return (
            Path(sys.argv[1])
            .expanduser()
            .resolve()
        )

    return DEFAULT_CONFIG


def threshold_tag(value):
    """
    0.40 -> 040
    0.50 -> 050
    0.60 -> 060
    0.70 -> 070
    0.80 -> 080
    """
    return "{:03d}".format(
        int(round(float(value) * 100.0))
    )


def write_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def count_existing_outputs(
    manifest,
    output_folder,
):
    found = 0
    missing = 0

    for route_dir, frames in manifest.items():
        for frame_name in frames:

            path = (
                route_dir
                / output_folder
                / "{}.json.gz".format(
                    frame_name
                )
            )

            if path.is_file():
                found += 1
            else:
                missing += 1

    return found, missing


def main():

    cfg_path = resolve_config_path()

    base_cfg = load_yaml(
        str(cfg_path)
    )

    sensitivity_cfg = base_cfg.get(
        "threshold_sensitivity",
        {},
    )

    if not sensitivity_cfg:
        raise RuntimeError(
            "Missing threshold_sensitivity "
            "section in config."
        )

    if not bool(
        sensitivity_cfg.get(
            "enabled",
            False,
        )
    ):
        raise RuntimeError(
            "threshold_sensitivity.enabled "
            "is false."
        )

    thresholds = [
        float(x)
        for x in sensitivity_cfg.get(
            "thresholds",
            [],
        )
    ]

    if not thresholds:
        raise RuntimeError(
            "No sensitivity thresholds configured."
        )

    baseline_threshold = float(
        sensitivity_cfg.get(
            "baseline_threshold",
            0.60,
        )
    )

    reuse_baseline_output = bool(
        sensitivity_cfg.get(
            "reuse_baseline_output",
            True,
        )
    )

    output_prefix = str(
        sensitivity_cfg.get(
            "output_prefix",
            "language_grounded_waypoints_tau",
        )
    )

    resume = bool(
        sensitivity_cfg.get(
            "resume",
            True,
        )
    )

    summary_folder = str(
        sensitivity_cfg.get(
            "summary_folder",
            "lg_threshold_sensitivity",
        )
    )

    data_root = Path(
        str(base_cfg.run.input)
    ).expanduser().resolve()

    keyframe_file = Path(
        str(base_cfg.run.keyframe_file)
    ).expanduser().resolve()

    summary_root = (
        data_root
        / summary_folder
    )

    summary_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        summary_root
        / "threshold_sensitivity.log"
    )

    setup_logger(
        verbose=bool(
            base_cfg.run.verbose
        ),
        log_file=log_file,
    )

    LOGGER.info(
        "[Config] {}".format(
            cfg_path
        )
    )

    LOGGER.info(
        "[Input] {}".format(
            data_root
        )
    )

    LOGGER.info(
        "[Keyframes] {}".format(
            keyframe_file
        )
    )

    manifest = load_keyframe_manifest(
        data_root=data_root,
        keyframe_file=keyframe_file,
    )

    total_routes = len(manifest)

    total_frames = sum(
        len(frames)
        for frames in manifest.values()
    )

    LOGGER.info(
        "[Manifest] routes={} keyframes={}".format(
            total_routes,
            total_frames,
        )
    )

    base_output_folder = str(
        base_cfg.paths.output_folder
    )

    run_index = []

    for threshold in thresholds:

        tag = threshold_tag(
            threshold
        )

        is_baseline = (
            abs(
                threshold
                - baseline_threshold
            )
            < 1e-12
        )

        # -------------------------------------------------------------
        # 0.60 uses the already completed formal LG experiment.
        # -------------------------------------------------------------

        if (
            is_baseline
            and reuse_baseline_output
        ):
            output_folder = (
                base_output_folder
            )

            found, missing = (
                count_existing_outputs(
                    manifest,
                    output_folder,
                )
            )

            LOGGER.info(
                "=" * 72
            )

            LOGGER.info(
                "[Threshold {}] "
                "reuse baseline output".format(
                    threshold
                )
            )

            LOGGER.info(
                "[Threshold {}] "
                "folder={}".format(
                    threshold,
                    output_folder,
                )
            )

            LOGGER.info(
                "[Threshold {}] "
                "found={} missing={}".format(
                    threshold,
                    found,
                    missing,
                )
            )

            summary = {
                "threshold":
                    float(threshold),

                "threshold_tag":
                    tag,

                "baseline":
                    True,

                "reused_existing_baseline":
                    True,

                "output_folder":
                    output_folder,

                "manifest_routes":
                    int(total_routes),

                "manifest_keyframes":
                    int(total_frames),

                "outputs_found":
                    int(found),

                "outputs_missing":
                    int(missing),

                "coverage":
                    (
                        float(found)
                        / float(total_frames)
                        if total_frames
                        else 0.0
                    ),

                "newly_processed":
                    0,

                "elapsed_seconds":
                    0.0,
            }

            summary_path = (
                summary_root
                / "tau{}_run_summary.json".format(
                    tag
                )
            )

            write_json(
                summary_path,
                summary,
            )

            run_index.append(
                summary
            )

            continue

        # -------------------------------------------------------------
        # Load a clean config for every threshold.
        # Only min_causal_score and output paths are changed.
        # -------------------------------------------------------------

        cfg = load_yaml(
            str(cfg_path)
        )

        cfg.causal_response[
            "min_causal_score"
        ] = float(threshold)

        output_folder = (
            "{}{}".format(
                output_prefix,
                tag,
            )
        )

        cfg.paths[
            "output_folder"
        ] = output_folder

        # Keep any optional debug outputs isolated as well.
        cfg.paths[
            "debug_folder"
        ] = (
            "language_grounded_waypoints_debug_tau{}"
            .format(tag)
        )

        output_cfg = cfg.get(
            "output",
            {},
        )

        if isinstance(
            output_cfg,
            dict,
        ):
            output_cfg[
                "full_debug_output_folder"
            ] = (
                "language_grounded_waypoints_"
                "full_debug_tau{}"
                .format(tag)
            )

        debug_cfg = cfg.get(
            "debug",
            {},
        )

        if isinstance(
            debug_cfg,
            dict,
        ):
            rgb_cfg = debug_cfg.get(
                "rgb",
                {},
            )

            if isinstance(
                rgb_cfg,
                dict,
            ):
                rgb_cfg[
                    "rgb_debug_folder"
                ] = (
                    "language_grounded_waypoints_"
                    "rgb_debug_tau{}"
                    .format(tag)
                )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "[Threshold {}] start".format(
                threshold
            )
        )

        LOGGER.info(
            "[Threshold {}] "
            "min_causal_score={}".format(
                threshold,
                cfg.causal_response[
                    "min_causal_score"
                ],
            )
        )

        LOGGER.info(
            "[Threshold {}] "
            "output_folder={}".format(
                threshold,
                output_folder,
            )
        )

        LOGGER.info(
            "[Threshold {}] "
            "route_follow_release_floor={}".format(
                threshold,
                cfg.causal_response.get(
                    "min_route_follow_release_effect_score",
                    None,
                ),
            )
        )

        start_time = time.time()

        total_ok = 0
        total_new = 0
        total_reused = 0
        total_failed = 0

        raise_on_error = bool(
            cfg.run.get(
                "raise_on_error",
                False,
            )
        )

        for route_index, (
            route_dir,
            frames,
        ) in enumerate(
            manifest.items(),
            1,
        ):

            route_ok = 0

            LOGGER.info(
                "[Threshold {}] "
                "[Route {}/{}] {} "
                "keyframes={}".format(
                    threshold,
                    route_index,
                    total_routes,
                    route_dir.name,
                    len(frames),
                )
            )

            for frame_name in frames:

                expected_output = (
                    route_dir
                    / output_folder
                    / "{}.json.gz".format(
                        frame_name
                    )
                )

                if (
                    resume
                    and expected_output.is_file()
                ):
                    total_ok += 1
                    total_reused += 1
                    route_ok += 1
                    continue

                try:

                    success = process_one_frame(
                        route_dir=route_dir,
                        frame_name=frame_name,
                        cfg=cfg,
                    )

                    if success:
                        total_ok += 1
                        total_new += 1
                        route_ok += 1
                    else:
                        total_failed += 1

                except Exception as exc:

                    total_failed += 1

                    LOGGER.exception(
                        "[Threshold {}] "
                        "[Error] "
                        "route={} "
                        "frame={} "
                        "error={}".format(
                            threshold,
                            route_dir,
                            frame_name,
                            exc,
                        )
                    )

                    if raise_on_error:
                        raise

            LOGGER.info(
                "[Threshold {}] "
                "[Route Done] {}: {}/{}".format(
                    threshold,
                    route_dir,
                    route_ok,
                    len(frames),
                )
            )

        elapsed = (
            time.time()
            - start_time
        )

        found, missing = (
            count_existing_outputs(
                manifest,
                output_folder,
            )
        )

        summary = {
            "threshold":
                float(threshold),

            "threshold_tag":
                tag,

            "baseline":
                False,

            "reused_existing_baseline":
                False,

            "output_folder":
                output_folder,

            "manifest_routes":
                int(total_routes),

            "manifest_keyframes":
                int(total_frames),

            "outputs_found":
                int(found),

            "outputs_missing":
                int(missing),

            "coverage":
                (
                    float(found)
                    / float(total_frames)
                    if total_frames
                    else 0.0
                ),

            "total_ok":
                int(total_ok),

            "newly_processed":
                int(total_new),

            "reused_existing":
                int(total_reused),

            "failed":
                int(total_failed),

            "elapsed_seconds":
                float(elapsed),

            "min_route_follow_release_effect_score":
                float(
                    cfg.causal_response.get(
                        "min_route_follow_release_effect_score",
                        0.03,
                    )
                ),
        }

        summary_path = (
            summary_root
            / "tau{}_run_summary.json".format(
                tag
            )
        )

        write_json(
            summary_path,
            summary,
        )

        run_index.append(
            summary
        )

        LOGGER.info(
            "[Threshold {}] "
            "Done found={}/{} "
            "new={} reused={} "
            "failed={} elapsed={:.1f}s".format(
                threshold,
                found,
                total_frames,
                total_new,
                total_reused,
                total_failed,
                elapsed,
            )
        )

    index_path = (
        summary_root
        / "run_index.json"
    )

    write_json(
        index_path,
        {
            "config":
                str(cfg_path),

            "data_root":
                str(data_root),

            "keyframe_file":
                str(keyframe_file),

            "baseline_threshold":
                float(
                    baseline_threshold
                ),

            "thresholds":
                thresholds,

            "runs":
                run_index,
        },
    )

    LOGGER.info(
        "=" * 72
    )

    LOGGER.info(
        "[All Thresholds Done] "
        "summaries={}".format(
            summary_root
        )
    )


if __name__ == "__main__":
    main()