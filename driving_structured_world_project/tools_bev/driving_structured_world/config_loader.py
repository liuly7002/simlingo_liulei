# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    candidates = [current] + list(current.parents)
    for candidate in candidates:
        if (
            (candidate / "lg_waypoint_planner_project").is_dir()
            and (candidate / "driving_structured_world_project").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Cannot locate repository root containing both "
        "lg_waypoint_planner_project and driving_structured_world_project."
    )


def bootstrap_lg_tools(repo_root: Optional[Path] = None) -> Path:
    root = (repo_root or find_repo_root()).resolve()
    lg_tools = root / "lg_waypoint_planner_project" / "tools_bev"
    if not lg_tools.is_dir():
        raise FileNotFoundError(f"Missing LG tools directory: {lg_tools}")
    text = str(lg_tools)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def load_project_config(config_path: Path):
    root = bootstrap_lg_tools()

    from lg_waypoint_planner.config_utils import Config, deep_update, load_yaml

    config_path = Path(config_path).expanduser().resolve()
    override = load_yaml(str(config_path))
    base_spec = override.get("base_lg_config", None)
    if base_spec in [None, "", "None", "null"]:
        raise ValueError("Missing base_lg_config in Driving structured-world YAML.")

    base_path = Path(str(base_spec)).expanduser()
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    if not base_path.exists():
        # A second repo-root-relative resolution gives a clearer fallback when the
        # project directory was moved without editing the YAML.
        fallback = (root / str(base_spec)).resolve()
        if fallback.exists():
            base_path = fallback
        else:
            raise FileNotFoundError(f"Missing base LG config: {base_path}")

    base = load_yaml(str(base_path))
    override_dict = dict(override)
    override_dict.pop("base_lg_config", None)
    merged = deep_update(dict(base), override_dict)
    merged["base_lg_config"] = str(base_path)
    merged["repo_root"] = str(root)
    return Config(merged)
