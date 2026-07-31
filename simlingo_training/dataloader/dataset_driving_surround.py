# -*- coding: utf-8 -*-
"""
Six-view wrapper for the ordinary SimLingo driving dataset.

The driving-language/task construction remains exactly in Data_Driving. This
wrapper adds the common six synchronized camera views and optionally attaches
the ordinary-Driving four-channel structured future-world label and the shared
participant-level spatial-attention supervision.
"""

from pathlib import Path

import numpy as np

from simlingo_training.dataloader.dataset_base_surround import (
    SurroundDatasetMixin,
)
from simlingo_training.dataloader.dataset_driving import (
    Data_Driving,
)
from simlingo_training.dataloader.dataset_lg import (
    Data_LG,
)
from simlingo_training.dataloader.participant_spatial_attention import (
    CAMERA_ORDER,
    TOKENS_PER_CAMERA,
    extract_participant_spatial_attention_supervision,
)


class Data_Driving_Surround(
    SurroundDatasetMixin,
    Data_Driving,
):
    """Ordinary driving supervision with the same six-view input as Data_LG."""

    def __init__(self, **cfg):
        # Six collected cameras have no corresponding geometric-shift sensor
        # set. Keep all views and geometric supervision in the original frame.
        common_cfg = dict(cfg)
        common_cfg["img_shift_augmentation"] = False  # 六视角训练不使用几何增强。

        Data_Driving.__init__(self, **common_cfg)
        self._initialize_surround_dataset()

    def __getitem__(self, index):
        # Preserve all original driving/commentary/QA language logic,
        # waypoints, route, target points and prompt sampling.
        sample = Data_Driving.__getitem__(self, index)

        image_data = {}
        self.load_surround_images(
            image_data,
            self.surround_images[index],
        )

        future_interaction_grid = None
        future_interaction_valid = False
        participant_spatial_target = np.zeros(
            (len(CAMERA_ORDER), TOKENS_PER_CAMERA),
            dtype=np.float32,
        )
        participant_spatial_valid = False

        use_future_interaction = bool(
            getattr(
                self,
                "driving_use_future_interaction_grid",
                False,
            )
        )
        use_participant_spatial_attention = bool(
            getattr(
                self,
                "driving_use_participant_spatial_attention_supervision",
                False,
            )
        )

        if (
            use_future_interaction
            or use_participant_spatial_attention
        ):
            measurement_path = Path(
                Data_LG._decode_path(
                    sample.measurement_path
                )
            )
            route_dir = measurement_path.parent.parent
            frame_name = measurement_path.name.split(".", 1)[0]

            if use_future_interaction:
                future_interaction_path = (
                    route_dir
                    / str(
                        getattr(
                            self,
                            "driving_future_interaction_grid_folder",
                            "driving_future_interaction_grids",
                        )
                    )
                    / f"{frame_name}.npz"
                )
                (
                    future_interaction_grid,
                    future_interaction_valid,
                ) = Data_LG._load_future_interaction_grid(
                    future_interaction_path
                )

            if use_participant_spatial_attention:
                participant_attention_path = (
                    route_dir
                    / str(
                        getattr(
                            self,
                            "driving_participant_attention_label_folder",
                            "driving_expert_conditioned_actor_selection",
                        )
                    )
                    / f"{frame_name}.json.gz"
                )

                if participant_attention_path.is_file():
                    participant_attention_payload = (
                        Data_LG._load_gzip_json(
                            participant_attention_path
                        )
                    )
                    (
                        participant_spatial_target,
                        participant_spatial_valid,
                    ) = extract_participant_spatial_attention_supervision(
                        participant_attention_payload,
                        cut_bottom_quarter=bool(
                            self.cut_bottom_quarter
                            or self.img_shift_augmentation
                        ),
                        use_global_img=bool(self.use_global_img),
                    )

        # DatasetOutput中的历史字段名继续作为公共批处理接口；
        # target实际形状和语义已经升级为[6,64]参与者空间监督。
        return sample._replace(
            image_ff=image_data["rgb"],
            image_ff_org_size=image_data["rgb_org_size"],
            image_surround=image_data["rgb_surround"],
            image_surround_org_size=(
                image_data["rgb_surround_org_size"]
            ),
            camera_order=image_data["camera_order"],
            future_interaction_grid=(
                future_interaction_grid
            ),
            future_interaction_valid=(
                future_interaction_valid
            ),
            camera_attention_target=(
                participant_spatial_target
            ),
            camera_attention_valid=(
                participant_spatial_valid
            ),
        )
