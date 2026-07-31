# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import torch

from simlingo_training.dataloader.datamodule import DataModule
from simlingo_training.dataloader.participant_spatial_attention import (
    CAMERA_ORDER,
    TOKENS_PER_CAMERA,
)


class InteractionDataModule(DataModule):
    """
    复用原DataModule的全部图像、文本、轨迹和四通道批处理逻辑，
    仅将历史六维相机监督接口升级为6×64参与者空间监督。
    """

    def dl_collate_fn(self, data: List):
        batch_size = len(data)
        spatial_target = torch.zeros(
            (
                batch_size,
                len(CAMERA_ORDER),
                TOKENS_PER_CAMERA,
            ),
            dtype=torch.float32,
        )
        spatial_valid = torch.zeros(
            (batch_size,),
            dtype=torch.bool,
        )

        compatibility_data = []
        for sample_index, sample in enumerate(data):
            sample_valid = bool(
                getattr(
                    sample,
                    "camera_attention_valid",
                    False,
                )
            )
            sample_target = getattr(
                sample,
                "camera_attention_target",
                None,
            )

            if sample_target is None:
                sample_target_array = np.zeros(
                    (
                        len(CAMERA_ORDER),
                        TOKENS_PER_CAMERA,
                    ),
                    dtype=np.float32,
                )
            else:
                sample_target_array = np.asarray(
                    sample_target,
                    dtype=np.float32,
                )

            expected_shape = (
                len(CAMERA_ORDER),
                TOKENS_PER_CAMERA,
            )
            if sample_target_array.shape != expected_shape:
                raise ValueError(
                    "Participant spatial attention target must have "
                    f"shape {expected_shape}, but received "
                    f"{tuple(sample_target_array.shape)}."
                )
            if (
                not np.isfinite(sample_target_array).all()
                or np.any(sample_target_array < 0.0)
            ):
                raise ValueError(
                    "Participant spatial attention target contains "
                    "non-finite or negative values."
                )

            target_sum = float(sample_target_array.sum())
            if sample_valid:
                if target_sum <= 0.0:
                    raise ValueError(
                        "A valid participant spatial attention target "
                        "must have a positive probability sum."
                    )
                sample_target_array = (
                    sample_target_array / target_sum
                )
                spatial_target[sample_index] = torch.from_numpy(
                    sample_target_array
                )
                spatial_valid[sample_index] = True

            # 原DataModule先按六维相机边缘概率完成其余批处理；
            # super返回后再将该字段替换为完整6×64空间目标。
            camera_marginal = sample_target_array.sum(axis=-1)
            compatibility_data.append(
                sample._replace(
                    camera_attention_target=(
                        camera_marginal.astype(np.float32)
                    ),
                    camera_attention_valid=sample_valid,
                )
            )

        example = super().dl_collate_fn(compatibility_data)
        driving_label = example.driving_label._replace(
            camera_attention_target=spatial_target,
            camera_attention_valid=spatial_valid,
        )
        return example._replace(driving_label=driving_label)
