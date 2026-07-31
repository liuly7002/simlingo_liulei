# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import numpy as np

from simlingo_training.dataloader.dataset_lg import Data_LG
from simlingo_training.dataloader.participant_spatial_attention import (
    extract_participant_spatial_attention_supervision,
)


class Data_LG_Spatial(Data_LG):  # pylint: disable=invalid-name
    """
    在原LG语言与轨迹数据逻辑上，将主要关键参与者投影转换为6×64视觉token监督。

    原Data_LG其余过滤、语言组织、waypoint和四通道标签逻辑保持不变。
    """

    def _extract_camera_attention_supervision(
        self,
        payload: Dict,
    ) -> Tuple[np.ndarray, bool]:
        # DatasetOutput中的历史字段名camera_attention_target继续作为批处理接口，
        # 但本类返回的实际语义已升级为participant spatial attention [6,64]。
        if not bool(
            getattr(
                self,
                "lg_use_participant_spatial_attention_supervision",
                False,
            )
        ):
            return np.zeros((6, 64), dtype=np.float32), False

        return extract_participant_spatial_attention_supervision(
            payload,
            cut_bottom_quarter=bool(
                self.cut_bottom_quarter
                or self.img_shift_augmentation
            ),
            use_global_img=bool(self.use_global_img),
        )
