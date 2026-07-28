# -*- coding: utf-8 -*-
"""
Six-view wrapper for the ordinary SimLingo driving dataset.

The driving-language/task construction remains exactly in Data_Driving. This
wrapper adds the common six synchronized camera views and optionally attaches
the ordinary-Driving four-channel structured future-world label and the shared
six-view camera-attention supervision.
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

        #修改20260728：普通Driving的两个共享辅助监督均允许独立关闭，
        # 便于分别消融四通道结构化未来世界和六视角显式注意力监督。
        future_interaction_grid = None
        future_interaction_valid = False
        camera_attention_target = np.zeros(
            (6,),
            dtype=np.float32,
        )
        camera_attention_valid = False

        use_future_interaction = bool(
            getattr(
                self,
                "driving_use_future_interaction_grid",
                False,
            )
        )
        use_camera_attention = bool(
            getattr(
                self,
                "driving_use_camera_attention_supervision",
                False,
            )
        )

        if use_future_interaction or use_camera_attention:
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

            if use_camera_attention:
                camera_attention_path = (
                    route_dir
                    / str(
                        getattr(
                            self,
                            "driving_camera_attention_label_folder",
                            "driving_expert_conditioned_actor_selection",
                        )
                    )
                    / f"{frame_name}.json.gz"
                )

                if camera_attention_path.is_file():
                    camera_attention_payload = (
                        Data_LG._load_gzip_json(
                            camera_attention_path
                        )
                    )
                    (
                        camera_attention_target,
                        camera_attention_valid,
                    ) = Data_LG._extract_camera_attention_supervision(
                        camera_attention_payload
                    )

        # Replace the legacy front fields with the front view taken from the
        # same six-view tensor, then expose the complete surround tensor.
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
                camera_attention_target
            ),
            camera_attention_valid=(
                camera_attention_valid
            ),
        )
