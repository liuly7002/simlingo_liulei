# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, List, Optional
import gzip
import json
import numpy as np


def load_json_gz(path: Path) -> Any:
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_costmap(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if isinstance(arr, np.lib.npyio.NpzFile):
        # use the first array by default
        key = arr.files[0]
        arr = arr[key]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        # keep the first channel if an exported BEV stack is provided
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D costmap, got shape={arr.shape} at {path}")
    return arr


def list_frame_names(route_dir: Path, cfg) -> List[str]:
    m_dir = route_dir / cfg.paths.measurements_folder
    c_dir = route_dir / cfg.paths.costmap_folder
    if not m_dir.exists() or not c_dir.exists():
        return []
    m = {p.name.replace(".json.gz", "") for p in m_dir.glob("*.json.gz")}
    c = {p.stem for p in c_dir.glob("*.npy")}
    return sorted(m & c)


def find_route_dirs(root: Path, cfg) -> List[Path]:
    if (root / cfg.paths.measurements_folder).exists() and (root / cfg.paths.costmap_folder).exists():
        return [root]
    out = []
    for p in root.rglob(cfg.paths.costmap_folder):
        if p.is_dir() and (p.parent / cfg.paths.measurements_folder).exists():
            out.append(p.parent)
    return sorted(set(out))


def array_to_list(arr, decimals: int = 4):
    arr = np.asarray(arr)
    return np.round(arr.astype(float), decimals).tolist()
