#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "language_grounded_waypoint.yaml"
)

CONFIG_PATH_OVERRIDE = None


sys.path.insert(0, str(CURRENT_FILE.parent))

from lg_waypoint_planner.config_utils import load_yaml
from lg_waypoint_planner.logger import setup_logger, LOGGER
from lg_waypoint_planner.processor import process_one_frame


def resolve_config_path() -> Path:
    if CONFIG_PATH_OVERRIDE is not None:
        return Path(CONFIG_PATH_OVERRIDE).expanduser().resolve()

    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).expanduser().resolve()

    return DEFAULT_CONFIG


def load_keyframe_manifest(
    data_root: Path,
    keyframe_file: Path,
):
    """
    keyframes.txt 每一行格式：

        relative/route/path/000123

    最后一部分是 frame，
    前面的部分全部是相对于 data_root 的 route 路径。
    """

    data_root = data_root.expanduser().resolve()
    keyframe_file = keyframe_file.expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {data_root}"
        )

    if not keyframe_file.is_file():
        raise FileNotFoundError(
            f"Keyframe file does not exist: {keyframe_file}"
        )

    grouped = OrderedDict()
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
                    f"{keyframe_file}:{line_no} "
                    f"invalid keyframe entry: {value!r}"
                )

            frame_name = parts[-1]

            route_rel = Path(
                *parts[:-1]
            )

            route_dir = (
                data_root
                / route_rel
            ).resolve()

            try:
                route_dir.relative_to(
                    data_root
                )
            except Exception:
                raise ValueError(
                    f"{keyframe_file}:{line_no} "
                    f"escapes run.input: {value!r}"
                )

            if not route_dir.is_dir():
                raise FileNotFoundError(
                    f"Route does not exist: {route_dir}"
                )

            key = (
                str(route_dir),
                str(frame_name),
            )

            if key in seen:
                continue

            seen.add(key)

            grouped.setdefault(
                route_dir,
                [],
            ).append(
                str(frame_name)
            )

    if not grouped:
        raise RuntimeError(
            f"No keyframes found in: {keyframe_file}"
        )

    return grouped


def main():

    cfg_path = resolve_config_path()

    cfg = load_yaml(
        str(cfg_path)
    )

    data_root = Path(
        str(cfg.run.input)
    ).expanduser().resolve()

    keyframe_file = Path(
        str(cfg.run.keyframe_file)
    ).expanduser().resolve()

    log_file = None

    if bool(cfg.run.save_log):
        log_file = (
            data_root
            / cfg.paths.log_folder
            / "language_grounded_waypoint_planner_keyframes.log"
        )

    setup_logger(
        verbose=bool(cfg.run.verbose),
        log_file=log_file,
    )

    LOGGER.info(
        f"[Config] {cfg_path}"
    )

    LOGGER.info(
        f"[Input] {data_root}"
    )

    LOGGER.info(
        f"[Keyframes] {keyframe_file}"
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
        f"[Manifest] "
        f"routes={total_routes} "
        f"keyframes={total_frames}"
    )

    total_ok = 0
    total_processed = 0

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
            f"[Route "
            f"{route_index}/{total_routes}] "
            f"{route_dir.name} "
            f"keyframes={len(frames)}"
        )

        for frame_index, frame_name in enumerate(
            frames,
            1,
        ):

            total_processed += 1

            try:

                success = process_one_frame(
                    route_dir=route_dir,
                    frame_name=frame_name,
                    cfg=cfg,
                )

                if success:
                    total_ok += 1
                    route_ok += 1

            except Exception as exc:

                LOGGER.exception(
                    f"[Error] "
                    f"route={route_dir} "
                    f"frame={frame_name} "
                    f"error={exc}"
                )

                if raise_on_error:
                    raise

        LOGGER.info(
            f"[Route Done] "
            f"{route_dir}: "
            f"{route_ok}/{len(frames)}"
        )

    LOGGER.info(
        f"[All Done] "
        f"{total_ok}/{total_processed} "
        f"keyframes processed."
    )


if __name__ == "__main__":
    main()