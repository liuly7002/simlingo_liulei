# -*- coding: utf-8 -*-
"""
Six-view wrapper for the ordinary SimLingo driving dataset.

The driving-language/task construction remains exactly in Data_Driving. This
wrapper only makes the common visual input six synchronized camera views and
returns them through DatasetOutput.
"""

from simlingo_training.dataloader.dataset_base_surround import (
    SurroundDatasetMixin,
)
from simlingo_training.dataloader.dataset_driving import (
    Data_Driving,
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
        common_cfg["img_shift_augmentation"] = False

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
        )
