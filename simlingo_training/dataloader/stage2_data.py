# -*- coding: utf-8 -*-
"""Stage-2 LG/Driving world targets and batch collate.

The generated label files are read-only.  This module keeps the existing
Data_LG_Counterfactual and DataModule pipelines intact while adapting their
four-channel legacy interface to C0/C1/C2 and C1/C2 temporal occupancy.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from simlingo_training.dataloader.dataset_lg_counterfactual import Data_LG_Counterfactual
from simlingo_training.dataloader.datamodule import DataModule


CHANNEL_NAMES = (
    "selected_route_ego_footprint_occupancy",
    "future_ego_footprint_occupancy",
    "primary_causal_actor_future_footprint_occupancy",
)
TEMPORAL_NAMES = ("C1_ego_future", "C2_critical_actor")


def load_stage2_grid(path: Path) -> Tuple[Optional[np.ndarray], bool, Optional[np.ndarray], int]:
    """Read one genuine stage-1 label without manufacturing missing positives."""
    if not path.is_file():
        return None, False, None, -1
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


class DataModuleStage2(DataModule):
    """Collate current [3,H,W] labels without rewriting legacy data logic.

    The parent collate still contains a four-channel shape check. Temporarily
    pad its local input to four channels, then discard the synthetic channel
    before returning training labels. The fourth channel is NEVER supervised.
    """

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
