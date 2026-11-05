# -*- coding: utf-8 -*-
"""Stage-2 LG/Driving world targets and batch collate.

The generated label files are read-only.  This module keeps the existing
Data_LG_Counterfactual and DataModule pipelines intact while adapting their
four-channel legacy interface to C0/C1/C2 and C1/C2 temporal occupancy.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from simlingo_training.dataloader.dataset_lg_counterfactual import Data_LG_Counterfactual
from simlingo_training.dataloader.datamodule import DataModule
from hydra.utils import get_original_cwd


CHANNEL_NAMES = (
    "selected_route_ego_footprint_occupancy",
    "future_ego_footprint_occupancy",
    "primary_causal_actor_future_footprint_occupancy",
)
TEMPORAL_NAMES = ("C1_ego_future", "C2_critical_actor")




###################################################################### 🤠 功能一: 形成最终lg样本 🤠 ######################################################################

def load_stage2_grid(path: Path) -> Tuple[Optional[np.ndarray], bool, Optional[np.ndarray], int]:

    # path = /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_25/Town12_Rep0_101_route0_09_13_21_09_57/driving_future_interaction_grids/0000.npz
    
    # 安全性检查
    if not path.is_file():
        return None, False, None, -1

    # 加载.npz文件,也就是读取三通道+十帧占据的标签数据
    with np.load(str(path), allow_pickle=False) as data:
        required = {
            "future_interaction_grid", "future_interaction_temporal", "valid",
            "critical_actor_status", "channel_names", "temporal_channel_names",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Stage-2 label missing {sorted(missing)}: {path}")
        grid = np.asarray(data["future_interaction_grid"], dtype=np.float32)
        temporal = np.asarray(data["future_interaction_temporal"], dtype=np.float32)
        valid = bool(np.asarray(data["valid"]).reshape(-1)[0])
        status = int(np.asarray(data["critical_actor_status"]).reshape(-1)[0])
        channels = tuple(str(x) for x in np.asarray(data["channel_names"]).reshape(-1))
        temporal_channels = tuple(str(x) for x in np.asarray(data["temporal_channel_names"]).reshape(-1))
        actor_id = str(np.asarray(data["critical_actor_id"]).reshape(-1)[0]) if "critical_actor_id" in data.files else ""
    if channels != CHANNEL_NAMES or temporal_channels != TEMPORAL_NAMES:
        raise ValueError(f"Incorrect stage-2 channel order in {path}: {channels}/{temporal_channels}")
    if grid.ndim != 3 or grid.shape[0] != 3 or grid.shape[-2:] != temporal.shape[-2:]:
        raise ValueError(f"Invalid summary world shape {grid.shape} in {path}")
    if temporal.ndim != 4 or temporal.shape[:2] != (2, 10):
        raise ValueError(f"Expected [2,10,H,W] future world, got {temporal.shape} in {path}")
    if status not in (-1, 0, 1) or (valid and status == -1):
        raise ValueError(f"Inconsistent critical_actor_status/valid in {path}")
    if valid and status == 1 and not actor_id:
        raise ValueError(f"Valid Actor status=1 lacks ID in {path}")
    if not np.isfinite(grid).all() or not np.isfinite(temporal).all():
        raise ValueError(f"Non-finite stage-2 occupancy target: {path}")
    if np.any(grid < 0) or np.any(grid > 1) or np.any(temporal < 0) or np.any(temporal > 1):
        raise ValueError(f"Out-of-range stage-2 occupancy target: {path}")
    if valid and status == 0 and (np.any(grid[2] > 1e-6) or np.any(temporal[1] > 1e-6)):
        raise ValueError(f"Reliable negative Actor label has occupied C2: {path}")
    # True summaries are equal-weight means at the original label resolution.
    if not np.allclose(grid[1], temporal[0].mean(axis=0), atol=3e-5, rtol=0):
        raise ValueError(f"C1 does not agree with its temporal average: {path}")
    if not np.allclose(grid[2], temporal[1].mean(axis=0), atol=3e-5, rtol=0):
        raise ValueError(f"C2 does not agree with its temporal average: {path}")

    """
    本函数直接看这里：
    grid: [3,H,W] 三通道占据网格,分别是 -> C0:参考路径对应的自车 Footprint 占据, C1:自车未来10帧 Footprint 的平均占据, C2:关键 Actor 未来10帧 Footprint 的平均占据
    valid: bool, 表示当前帧的结构化未来世界标签是否有效 True: 标签有效, False: 标签无效,不应直接作为有效的未来世界监督
    temporal: [2,10,H,W] 未来10帧的逐帧占据信息 两个通道分别是 -> temporal[0]:自车未来10帧的逐帧 Footprint 占据, temporal[1]:关键 Actor 未来10帧的逐帧 Footprint 占据
    status: int, 关键 Actor 的三状态标签 -1:无法可靠判断,关键 Actor 监督应屏蔽, 0:未确认存在决策关键Actor, 1:存在确认的决策关键 Actor
    """
    return grid, valid, temporal, status


def strict_negative_actor_status(status: int, lg_json_path: Path) -> int:
    """Mask unreliable status-0 negatives with attempted but all-invalid CF tests.

    The audit found 2,565 such cases. This changes a *training mask* only,
    not the source NPZ/JSON, and does not invalidate C0/C1 reconstruction.
    No candidate tests (e.g. clear scenes) are not rejected.
    """
    if int(status) != 0 or not lg_json_path.is_file():
        return int(status)
    try:
        with gzip.open(lg_json_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        analysis = payload.get("causal_analysis", {}) or {}
        tests = analysis.get("object_tests", []) or []
        if tests and not any(bool(t.get("counterfactual_valid", False)) for t in tests if isinstance(t, dict)):
            return -1
    except (OSError, ValueError, TypeError):
        return -1
    return 0


class Data_LG_Stage2(Data_LG_Counterfactual):
    """LG source with original language, counterfactuals and route untouched."""

    def _load_future_interaction_grid(self, path: Path):
        # Cache only the current sample, avoiding a second decompress of the
        # large temporal NPZ when the parent builds its legacy sample fields.
        result = load_stage2_grid(path)
        self._stage2_last_grid = (path, result)
        return result[0], result[1]  # Retain Data_LG's two-value call.

    def __getitem__(self, index):
        self._stage2_last_grid = None
        sample = super().__getitem__(index)
        if not bool(getattr(self, "lg_use_future_interaction_grid", False)):
            return sample._replace(future_interaction_temporal=None, critical_actor_status=-1)
        path = self._future_interaction_grid_path_for_index(index)
        cached = self._stage2_last_grid
        _grid, valid, temporal, status = (
            cached[1] if cached is not None and cached[0] == path
            else load_stage2_grid(path)
        )
        self._stage2_last_grid = None
        if bool(getattr(self, "stage2_strict_negative_mask", True)) and status == 0:
            status = strict_negative_actor_status(status, self._lg_path_for_index(index))
        return sample._replace(
            future_interaction_temporal=temporal,
            critical_actor_status=status,
            # Do not use visual targets when Actor identity is not reliable.
            camera_attention_valid=(bool(sample.camera_attention_valid) and valid and status == 1),
        )



###################################################################### 🤠 功能二:  🤠 ######################################################################

def _actor_label_paths(dataset, index: int) -> Tuple[Path, Path]:
    """Locate stage-1 metadata for the *same current frame* as __getitem__.

    No camera images, language, waypoints or occupancy grids are loaded here.
    Labels are used only to build the training sampler, never as model inputs.
    """
    if isinstance(dataset, Data_LG_Stage2):
        return (
            dataset._future_interaction_grid_path_for_index(index),
            dataset._lg_path_for_index(index),
        )
    if not bool(getattr(dataset, "driving_use_future_interaction_grid", False)):
        raise ValueError("Balanced Actor sampling requires Driving stage-2 world labels")

    measurement_entry = dataset.measurements[index]
    if isinstance(measurement_entry, np.ndarray):
        measurement_entry = measurement_entry.reshape(-1)[0]
    elif isinstance(measurement_entry, (list, tuple)):
        measurement_entry = measurement_entry[0]
    measurement_dir = Path(Data_LG_Counterfactual._decode_path(measurement_entry))
    route_dir = measurement_dir.parent
    frame_id = int(dataset.sample_start[index]) + int(dataset.hist_len) - 1
    frame_name = f"{frame_id:04d}"
    return (
        route_dir
        / str(getattr(dataset, "driving_future_interaction_grid_folder", "driving_future_interaction_grids"))
        / f"{frame_name}.npz",
        route_dir
        / str(getattr(dataset, "lg_label_folder", "language_grounded_waypoints"))
        / f"{frame_name}.json.gz",
    )


def _actor_sampling_status(grid_path: Path, lg_json_path: Path, strict_negative: bool) -> int:
    """Classify each available frame before building the exact training subset.

    Status=-1, missing NPZ, invalid world or unverified Actor identity are
    excluded.  A status-0 sample is treated exactly as in __getitem__:
    attempted-but-all-invalid counterfactual tests are not reliable negatives.
    """
    if not grid_path.is_file():
        return -1
    # Read three small NPZ members, not the large occupancy arrays.
    with np.load(str(grid_path), allow_pickle=False) as payload:
        for key in ("valid", "critical_actor_status"):
            if key not in payload.files:
                raise ValueError(f"Missing {key} in stage-2 label: {grid_path}")
        valid = bool(np.asarray(payload["valid"]).reshape(-1)[0])
        status = int(np.asarray(payload["critical_actor_status"]).reshape(-1)[0])
        actor_id = (
            str(np.asarray(payload["critical_actor_id"]).reshape(-1)[0])
            if "critical_actor_id" in payload.files else ""
        )
    if status not in (-1, 0, 1):
        raise ValueError(f"Invalid critical_actor_status={status} in {grid_path}")
    if status == 1 and not valid:
        raise ValueError(f"Positive Actor frame has invalid world label: {grid_path}")
    if not valid or status == -1:
        return -1
    if status == 1 and not actor_id:
        raise ValueError(f"Positive Actor label is missing its ID: {grid_path}")
    if status == 0 and strict_negative:
        return strict_negative_actor_status(status, lg_json_path)
    return status


def _cached_actor_statuses(dataset, cache_directory: Path, source: str) -> np.ndarray:
    """Cache sampled frame statuses without modifying source labels.

    The fingerprint includes the original per-frame label paths and strict mask
    setting. Stage-1 labels are immutable in this project; if they are edited
    later, delete the cache directory before another training run.
    """
    if source == "lg" and not bool(getattr(dataset, "lg_use_future_interaction_grid", False)):
        raise ValueError("Balanced Actor sampling requires LG stage-2 world labels")
    strict = bool(getattr(dataset, "stage2_strict_negative_mask", True))
    paths = []
    signature = hashlib.sha256(b"stage2_actor_sampler_v1\0")
    signature.update(source.encode("utf-8"))
    signature.update(bytes([int(strict)]))
    for index in range(len(dataset)):
        grid_path, lg_path = _actor_label_paths(dataset, index)
        paths.append((grid_path, lg_path))
        signature.update(os.fsencode(grid_path))
        signature.update(b"\0")
        if strict:
            signature.update(os.fsencode(lg_path))
            signature.update(b"\0")
    fingerprint = signature.hexdigest()
    cache_file = cache_directory / f"{source}_{fingerprint}.npy"
    if cache_file.is_file():
        try:
            cached = np.load(str(cache_file), allow_pickle=False)
            if cached.dtype == np.dtype("int8") and cached.shape == (len(dataset),):
                if np.isin(cached, (-1, 0, 1)).all():
                    print(f"[Stage2 Actor sampling] cached index: {source}, {len(dataset)} samples", flush=True)
                    return cached
        except (OSError, ValueError):
            pass  # Rebuild an incomplete or corrupted cache.

    statuses = np.empty(len(paths), dtype=np.int8)
    for index, (grid_path, lg_path) in enumerate(paths):
        statuses[index] = _actor_sampling_status(grid_path, lg_path, strict)
    cache_directory.mkdir(parents=True, exist_ok=True)
    # Different Lightning ranks may write the same index concurrently.
    # Atomic replacement prevents partially written files from being reused.
    temporary = cache_directory / f".{cache_file.stem}.{os.getpid()}.npy"
    try:
        with temporary.open("wb") as out:
            np.save(out, statuses, allow_pickle=False)
        os.replace(str(temporary), str(cache_file))
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[Stage2 Actor sampling] built index: {source}, {len(dataset)} samples", flush=True)
    return statuses


def _negative_quotas(source_pools, total_negative: int):
    """Allocate one unique negative pool per source, proportional to positives.

    Preserve the real LG/Driving mixture: source sizes are NOT forced to 50/50.
    The final total is exact (within the integer precision of the desired 3:1
    class ratio), provided there are enough reliable negatives.
    """
    total_positive = sum(len(pool["positive"]) for pool in source_pools)
    if total_positive < 1:
        raise ValueError("Actor-focused training has no confirmed positive samples")
    available_negative = sum(len(pool["negative"]) for pool in source_pools)
    if total_negative > available_negative:
        raise ValueError(
            f"Only {available_negative} reliable negatives available; "
            f"need {total_negative} for the requested class ratio"
        )

    ideal = [total_negative * len(pool["positive"]) / total_positive for pool in source_pools]
    quotas = [min(int(np.floor(want)), len(pool["negative"]))
              for want, pool in zip(ideal, source_pools)]
    remaining = total_negative - sum(quotas)
    while remaining > 0:
        eligible = [i for i, pool in enumerate(source_pools)
                    if quotas[i] < len(pool["negative"])]
        if not eligible:
            raise ValueError("Unable to allocate the requested reliable negative samples")
        # Prefer the source whose selected count is farthest below its ideal.
        selected = max(eligible, key=lambda i: (ideal[i] - quotas[i], -i))
        quotas[selected] += 1
        remaining -= 1
    return quotas


def _target_negative_count(total_positive: int, positive_ratio: float,
                           training_block: int, negative_capacity: int) -> int:
    """Choose the nearest achievable 3:1 pool that does not require DDP padding.

    With two devices and batch size one, the resulting dataset length is even.
    All selected examples can therefore be consumed once per epoch without
    duplicated DistributedSampler padding or drop_last losses.
    """
    if not (0.0 < positive_ratio < 1.0):
        raise ValueError("stage2_actor_positive_ratio must be strictly between zero and one")
    if total_positive < 1:
        raise ValueError("No positive Actor samples remain after source filtering")
    if negative_capacity < 1:
        raise ValueError("No reliable negative Actor samples are available")
    if training_block < 1:
        raise ValueError("training_block must be positive")

    desired = total_positive * (1.0 - positive_ratio) / positive_ratio
    # Require N_positive + N_negative to be divisible by world_size*batch_size.
    first = (-total_positive) % training_block
    if first == 0:
        first = training_block  # Both classes must be nonempty.
    options = [first + k * training_block
               for k in range((negative_capacity - first) // training_block + 1)]
    if not options:
        raise ValueError(
            "Insufficient reliable negative samples for the distributed batch size: "
            f"positives={total_positive}, available_negatives={negative_capacity}, "
            f"world_size*batch_size={training_block}"
        )
    return min(options, key=lambda x: (abs(x - desired), x))


class DataModuleStage2(DataModule):

    def setup(self, stage=None):
        # Base DataModule still constructs the original training and validation
        # datasets; only the training selection below changes.
        super().setup(stage)
        if self.predict or not bool(getattr(self, "stage2_actor_balanced_sampling", False)):
            return
        if self.train_dataset is None:
            raise ValueError("Actor-focused training needs a nonempty training dataset")
        if not isinstance(self.train_dataset, torch.utils.data.ConcatDataset):
            raise TypeError("Expected original ConcatDataset from DataModule.setup")

        positive_ratio = float(getattr(self, "stage2_actor_positive_ratio", 0.75))
        cache_setting = str(getattr(self, "stage2_actor_sampling_cache_dir",
                                    "outputs/stage2_actor_sampling_cache"))
        cache_directory = Path(cache_setting)
        if not cache_directory.is_absolute():
            cache_directory = Path(get_original_cwd()) / cache_directory

        # The original Bucket pipeline may contain extra overlapping subsets
        # (turn_left, leading_object_vehicle, etc.). Use ONLY bucket_name=all
        # from each source, otherwise some positive physical frames would be
        # counted or trained multiple times in one epoch.
        source_datasets = {}
        for source_dataset in self.train_dataset.datasets:
            if str(getattr(source_dataset, "bucket_name", "")) != "all":
                continue
            source = "lg" if isinstance(source_dataset, Data_LG_Stage2) else "driving"
            if source in source_datasets:
                raise ValueError(f"Duplicate all-bucket training dataset for {source}")
            source_datasets[source] = source_dataset
        if not source_datasets:
            raise ValueError("Cannot find full bucket_name='all' training datasets")
        if self.driving_dataset is not None and "driving" not in source_datasets:
            raise ValueError("The full Driving 'all' bucket is missing")
        if self.dreamer_dataset is not None and "lg" not in source_datasets:
            raise ValueError("The full LG 'all' bucket is missing")

        source_pools = []
        for source in ("driving", "lg"):
            if source not in source_datasets:
                continue
            source_dataset = source_datasets[source]
            statuses = _cached_actor_statuses(source_dataset, cache_directory, source)
            positive = np.flatnonzero(statuses == 1)
            negative = np.flatnonzero(statuses == 0)
            if len(positive) == 0 or len(negative) == 0:
                raise ValueError(
                    f"{source} full training bucket requires both positive and "
                    f"reliable negative Actor labels (found {len(positive)}/{len(negative)})"
                )
            source_pools.append({
                "source": source,
                "dataset": source_dataset,
                "positive": positive,
                "negative": negative,
                "unknown": int(np.count_nonzero(statuses == -1)),
            })

        total_positive = sum(len(pool["positive"]) for pool in source_pools)
        negative_capacity = sum(len(pool["negative"]) for pool in source_pools)
        # Lightning's DistributedSampler and train DataLoader(drop_last=True)
        # must not silently duplicate/drop a positive frame in a training epoch.
        trainer = getattr(self, "trainer", None)
        world_size = int(getattr(trainer, "world_size", 1))
        training_block = world_size * int(self.batch_size)
        selected_negative = _target_negative_count(
            total_positive, positive_ratio, training_block, negative_capacity,
        )
        quotas = _negative_quotas(source_pools, selected_negative)

        # Fixed, reproducible *without-replacement* negative subset. Positive
        # frames are NEVER sampled or subsampled: each appears exactly once in
        # this epoch's dataset (and once per subsequent epoch, newly shuffled).
        generator = np.random.default_rng(
            int(getattr(self, "stage2_actor_negative_seed", 9876))
        )
        training_subsets = []
        source_summary = {}
        for pool, count in zip(source_pools, quotas):
            sampled_negative = generator.choice(pool["negative"], size=count,
                                                replace=False)
            selected_indices = np.sort(np.concatenate((
                pool["positive"], sampled_negative,
            ))).tolist()
            training_subsets.append(
                torch.utils.data.Subset(pool["dataset"], selected_indices)
            )
            source_summary[pool["source"]] = {
                "positive_all": len(pool["positive"]),
                "reliable_negative_selected": count,
                "reliable_negative_available": len(pool["negative"]),
                "unknown_excluded": pool["unknown"],
            }

        self.train_dataset = torch.utils.data.ConcatDataset(training_subsets)
        self.sampler_train = torch.utils.data.RandomSampler(
            self.train_dataset, replacement=False,
        )
        actual_positive_ratio = total_positive / len(self.train_dataset)
        print(
            "[Stage2 Actor sampling] all positives retained ONCE per epoch; "
            f"positive={total_positive}, reliable_negative={selected_negative}, "
            f"actual_ratio={actual_positive_ratio:.4%}/"
            f"{1.0-actual_positive_ratio:.4%}; "
            f"sources={source_summary}; "
            f"training_block={training_block}; validation=original distribution",
            flush=True,
        )

    def dl_collate_fn(self, data):
        converted = []
        for sample in data:
            grid = getattr(sample, "future_interaction_grid", None)
            if grid is not None:
                grid = np.asarray(grid, dtype=np.float32)
                if grid.ndim != 3 or grid.shape[0] != 3:
                    raise ValueError(f"Expected [3,H,W] sample world, got {grid.shape}")
                temporary = np.concatenate((grid, np.zeros_like(grid[:1])), axis=0)
                converted.append(sample._replace(future_interaction_grid=temporary))
            else:
                converted.append(sample)
        example = super().dl_collate_fn(converted)
        old = example.driving_label
        grid3 = old.future_interaction_grid[:, :3] if old.future_interaction_grid is not None else None
        batch_size = len(data)
        status = torch.full((batch_size,), -1, dtype=torch.int8)
        reference_shape = None
        for sample in data:
            if getattr(sample, "future_interaction_temporal", None) is not None:
                reference_shape = tuple(np.asarray(sample.future_interaction_temporal).shape)
                break
        temporal_batch = None
        if reference_shape is not None:
            if len(reference_shape) != 4 or reference_shape[:2] != (2, 10):
                raise ValueError(f"Temporal reference shape must be [2,10,H,W]: {reference_shape}")
            temporal_batch = torch.zeros((batch_size, *reference_shape), dtype=torch.float32)
        for i, sample in enumerate(data):
            raw_status = getattr(sample, "critical_actor_status", -1)
            status[i] = int(-1 if raw_status is None else raw_status)
            if int(status[i]) not in (-1, 0, 1):
                raise ValueError("Stage-2 critical Actor status must be -1/0/1")
            value = getattr(sample, "future_interaction_temporal", None)
            if value is None:
                if bool(getattr(sample, "future_interaction_valid", False)):
                    raise ValueError("A valid world sample lacks temporal occupancy")
                continue
            arr = np.asarray(value, dtype=np.float32)
            if tuple(arr.shape) != reference_shape:
                raise ValueError(f"Inconsistent temporal world shapes: {arr.shape} vs {reference_shape}")
            if temporal_batch is not None:
                temporal_batch[i] = torch.from_numpy(arr)
        updated = old._replace(
            future_interaction_grid=grid3,
            future_interaction_temporal=temporal_batch,
            critical_actor_status=status,
        )
        return example._replace(driving_label=updated)
