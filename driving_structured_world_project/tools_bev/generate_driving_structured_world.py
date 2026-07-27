#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from driving_structured_world.config_loader import load_project_config


CURRENT_FILE = Path(__file__).resolve()
DEFAULT_CONFIG = CURRENT_FILE.parents[1] / "configs" / "driving_structured_world.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate expert-conditioned five-channel structured-world labels for Driving data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Driving structured-world YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_project_config(args.config)

    from lg_waypoint_planner.logger import setup_logger
    from driving_structured_world.processor import process_dataset

    log_file = None
    if bool(getattr(cfg.run, "save_log", False)):
        log_file = (
            Path(str(cfg.run.input)).expanduser().resolve()
            / str(cfg.paths.log_folder)
            / "driving_structured_world.log"
        )
    setup_logger(verbose=bool(cfg.run.verbose), log_file=log_file)
    process_dataset(cfg)


if __name__ == "__main__":
    main()
