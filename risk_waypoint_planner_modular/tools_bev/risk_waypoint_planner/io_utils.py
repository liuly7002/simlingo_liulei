# -*- coding: utf-8 -*-

from .common import *

def load_json_gz(path: Path) -> Dict:
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def save_json_gz(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

def load_costmap(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Costmap does not exist: {path}")
    cost = np.load(str(path))
    if cost.ndim != 2:
        raise ValueError(f"Expected 2D costmap, got shape {cost.shape}: {path}")
    return cost.astype(np.float32)

def points_to_list(points: np.ndarray, decimals: int = 3) -> List[List[float]]:
    points = np.asarray(points, dtype=np.float32)
    points = np.round(points.astype(float), decimals=decimals)
    return points.tolist()

def array_to_list(arr: np.ndarray, decimals: int = 4) -> List[float]:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.round(arr.astype(float), decimals=decimals)
    return arr.tolist()
