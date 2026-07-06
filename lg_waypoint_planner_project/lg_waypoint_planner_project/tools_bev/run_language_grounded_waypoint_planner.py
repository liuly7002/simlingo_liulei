#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "language_grounded_waypoint.yaml"

# Optional top-of-file overrides. Leave as None to use YAML.
CONFIG_PATH_OVERRIDE = None
INPUT_OVERRIDE = None
FRAME_OVERRIDE = None

sys.path.insert(0, str(CURRENT_FILE.parent))

from lg_waypoint_planner.config_utils import load_yaml
from lg_waypoint_planner.logger import setup_logger, LOGGER
from lg_waypoint_planner.processor import process_dataset


def resolve_config_path() -> Path:
    if CONFIG_PATH_OVERRIDE is not None:
        return Path(CONFIG_PATH_OVERRIDE).expanduser().resolve()
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).expanduser().resolve()
    return DEFAULT_CONFIG


def main():
    cfg_path = resolve_config_path()
    cfg = load_yaml(str(cfg_path))

    if INPUT_OVERRIDE is not None:
        cfg.run.input = INPUT_OVERRIDE
    if FRAME_OVERRIDE is not None:
        cfg.run.frame = FRAME_OVERRIDE

    log_file = None
    if bool(cfg.run.save_log):
        log_root = Path(str(cfg.run.input)).expanduser().resolve()
        log_file = log_root / cfg.paths.log_folder / "language_grounded_waypoint_planner.log"

    setup_logger(verbose=bool(cfg.run.verbose), log_file=log_file)
    LOGGER.info(f"[Config] {cfg_path}")
    LOGGER.info(f"[Input] {cfg.run.input}")
    process_dataset(cfg)


if __name__ == "__main__":
    main()
