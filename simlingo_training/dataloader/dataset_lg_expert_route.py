# -*- coding: utf-8 -*-
"""Stage-2 LG samples with expert Route supervision and original LG objectives.

Only the 20-point ``DatasetOutput.path`` training label changes:
  LG candidate reference route -> the expert ``measurements['route']``.

LG ego future waypoints, language, counterfactual labels, actor targets,
C0/C1/C2 occupancy, and Stage-2 sampling rules remain unchanged.
This module does not rewrite any saved measurement, JSON or NPZ labels.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import ujson

from simlingo_training.dataloader.stage2_data import Data_LG_Stage2


class Data_LG_Stage2_ExpertRoute(Data_LG_Stage2):
    """Use the expert-adjusted 20-point Route for LG samples only."""

    def __getitem__(self, index):
        # Preserve the complete existing LG/Stage-2 loading and supervision.
        sample = super().__getitem__(index)

        # Data_LG already determines which measurement frame is current.
        # Reuse that exact path instead of guessing it from sample_start.
        measurement_path = Path(sample.measurement_path)
        with gzip.open(measurement_path, "rt", encoding="utf-8") as stream:
            measurement = ujson.load(stream)

        # Match Data_LG.load_route(): its expert target uses the *adjusted*
        # measurement['route'], with zero augmentation and 1-m resampling.
        # Do not use route_original or LG selected_reference_route here.
        expert_route = np.asarray(measurement["route"], dtype=np.float32)
        if expert_route.ndim != 2 or expert_route.shape[1] != 2 or expert_route.shape[0] < 2:
            raise ValueError(
                f"Invalid expert measurement route in {measurement_path}: "
                f"{expert_route.shape}"
            )
        if not np.isfinite(expert_route).all():
            raise ValueError(f"Non-finite expert route in {measurement_path}")

        expert_path = np.asarray(self.equal_spacing_route(expert_route), dtype=np.float32)
        if expert_path.shape != (20, 2) or not np.isfinite(expert_path).all():
            raise ValueError(
                f"Invalid resampled expert path in {measurement_path}: "
                f"{expert_path.shape}"
            )

        return sample._replace(path=expert_path)
