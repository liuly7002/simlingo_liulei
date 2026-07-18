# -*- coding: utf-8 -*-
"""
Shared six-view data-indexing and image-loading support.

Both the ordinary driving dataset and the LG dataset use this module, so the
six synchronized RGB views are a common visual input rather than an LG-only
input.

Fixed tensor order:
    front, front_left, front_right, rear, rear_left, rear_right

Indexed paths:
    [N, T, V]

Loaded images:
    [T, V, C, H, W]
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from simlingo_training.dataloader.dataset_base import BaseDataset  # 父类


SURROUND_CAMERA_ORDER: Tuple[str, ...] = (
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
)

SURROUND_CAMERA_FOLDERS: Dict[str, Tuple[str, ...]] = {
    # The collector stores the front image in both rgb_front/ and rgb/.
    "front": ("rgb_front", "rgb"),
    "front_left": ("rgb_front_left",),
    "front_right": ("rgb_front_right",),
    "rear": ("rgb_rear",),
    "rear_left": ("rgb_rear_left",),
    "rear_right": ("rgb_rear_right",),
}


class SurroundDatasetMixin:
    """Add six-view indexing/loading to a dataset already built by BaseDataset."""

    def _initialize_surround_dataset(self) -> None:
        self.surround_camera_order = tuple(SURROUND_CAMERA_ORDER)
        self.surround_camera_folders = dict(SURROUND_CAMERA_FOLDERS)
        self._build_surround_image_index()

    @staticmethod
    def _decode_dataset_path(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return value.tobytes().decode("utf-8")
        return str(value)

    @staticmethod
    def _filter_index_container(values, indices, valid_indices):
        if isinstance(values, np.ndarray):
            return values[indices]
        return [values[int(index)] for index in valid_indices]

    def _filter_base_samples(
        self,
        valid_indices: List[int],
        original_count: int,
    ) -> None:
        """
        Apply the same six-view-valid indices to every sample-level container.

        This keeps image paths, boxes, measurements, frame IDs and optional
        trajectory files strictly aligned.
        """
        indices = np.asarray(valid_indices, dtype=np.int64)

        attributes = [
            "images",
            "boxes",
            "measurements",
            "sample_start",
            "augment_exists",
        ]

        if hasattr(self, "alternative_trajectories"):
            alternatives = getattr(self, "alternative_trajectories")
            if len(alternatives) == original_count:
                attributes.append("alternative_trajectories")

        for attribute in attributes:
            values = getattr(self, attribute)
            if len(values) != original_count:
                raise RuntimeError(
                    f"Surround index container length mismatch: {attribute} "
                    f"has {len(values)} entries, expected {original_count}"
                )

            filtered = self._filter_index_container(
                values,
                indices,
                valid_indices,
            )
            if attribute == "augment_exists":
                filtered = np.asarray(filtered, dtype=np.bool_)
            setattr(self, attribute, filtered)

    def _resolve_camera_path(
        self,
        route_dir: Path,
        frame_filename: str,
        camera_name: str,
    ) -> Path:
        folders = self.surround_camera_folders[camera_name]
        for folder_name in folders:
            candidate = route_dir / folder_name / frame_filename
            if candidate.is_file():
                return candidate

        # Return the preferred path for readable diagnostics.
        return route_dir / folders[0] / frame_filename

    def _build_surround_image_index(self) -> None:
        """
        Derive the synchronized six-view paths from BaseDataset's front index.

        A sample is discarded when any required view is absent at any history
        timestep. Missing views are never silently replaced by zero images.
        """
        original_count = len(self.images)
        valid_indices: List[int] = []
        valid_surround_paths: List[List[List[str]]] = []
        reasons = Counter()

        for sample_index in range(original_count):
            temporal_front_paths = np.asarray(
                self.images[sample_index]
            ).reshape(-1)

            if len(temporal_front_paths) != int(self.hist_len):
                reasons["invalid_history_length"] += 1
                continue

            temporal_views: List[List[str]] = []
            sample_valid = True

            for front_entry in temporal_front_paths:
                front_path = Path(
                    self._decode_dataset_path(front_entry)
                )
                route_dir = front_path.parent.parent
                frame_filename = front_path.name

                frame_views: List[str] = []
                for camera_name in self.surround_camera_order:
                    camera_path = self._resolve_camera_path(
                        route_dir,
                        frame_filename,
                        camera_name,
                    )
                    if not camera_path.is_file():
                        reasons[f"missing_{camera_name}"] += 1
                        sample_valid = False
                        break
                    frame_views.append(str(camera_path))

                if not sample_valid:
                    break
                temporal_views.append(frame_views)

            if not sample_valid:
                continue

            valid_indices.append(sample_index)
            valid_surround_paths.append(temporal_views)

        self._filter_base_samples(valid_indices, original_count)

        if valid_surround_paths:
            self.surround_images = np.asarray(
                valid_surround_paths,
                dtype=np.string_,
            )
        else:
            self.surround_images = np.empty(
                (
                    0,
                    int(self.hist_len),
                    len(self.surround_camera_order),
                ),
                dtype=np.string_,
            )

        if bool(
            getattr(
                self,
                "surround_print_filter_summary",
                True,
            )
        ):
            print(
                f"[{self.split} surround samples]: kept "
                f"{len(valid_indices)}/{original_count}; "
                f"filtered={dict(reasons)}"
            )

    def load_surround_images(
        self,
        data: Dict,
        surround_images: Sequence[Sequence],
    ) -> Dict:
        """
        Load the six synchronized RGB views into [T, V, C, H, W].

        No geometric camera augmentation is performed. During training, the
        existing photometric augmenter may still be used; all six views from
        the same timestep share one deterministic augmentation instance.
        Validation images remain deterministic.
        """
        paths = np.asarray(surround_images)
        expected_shape = (
            int(self.hist_len),
            len(self.surround_camera_order),
        )
        if paths.shape != expected_shape:
            raise ValueError(
                f"Expected surround image paths with shape {expected_shape}, "
                f"received {tuple(paths.shape)}"
            )

        loaded_images = []
        loaded_images_org_size = []

        for time_index in range(int(self.hist_len)):
            deterministic_augmenter = None
            if (
                self.split == "train"
                and bool(self.img_augmentation)
            ):
                deterministic_augmenter = self.tfs.to_deterministic()

            time_images = []
            time_images_org_size = []

            for camera_index, camera_name in enumerate(
                self.surround_camera_order
            ):
                image_path = self._decode_dataset_path(
                    paths[time_index, camera_index]
                )
                if not Path(image_path).is_file():
                    raise FileNotFoundError(
                        f"Missing {camera_name} image: {image_path}"
                    )

                image = cv2.imread(
                    image_path,
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise RuntimeError(
                        f"OpenCV failed to read {camera_name} image: "
                        f"{image_path}"
                    )
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

                if deterministic_augmenter is not None:
                    image = deterministic_augmenter(image=image)

                image_org_size = image.copy()

                # Retain the existing SimLingo crop behavior. Geometric camera
                # shift augmentation itself is disabled by the six-view driving
                # wrapper.
                if (
                    self.cut_bottom_quarter
                    or self.img_shift_augmentation
                ):
                    crop_height = int(
                        image.shape[0]
                        - (image.shape[0] * 4.8) // 16
                    )
                    image = image[:crop_height, :, :]

                time_images.append(image)
                time_images_org_size.append(image_org_size)

            loaded_images.append(time_images)
            loaded_images_org_size.append(
                time_images_org_size
            )

        processed = np.asarray(loaded_images)
        processed_org_size = np.asarray(
            loaded_images_org_size
        )

        # [T, V, H, W, C] -> [T, V, C, H, W]
        processed = np.transpose(
            processed,
            (0, 1, 4, 2, 3),
        )
        processed_org_size = np.transpose(
            processed_org_size,
            (0, 1, 4, 2, 3),
        )

        data["rgb_surround"] = processed
        data["rgb_surround_org_size"] = (
            processed_org_size
        )
        data["camera_order"] = (
            self.surround_camera_order
        )

        # Keep the legacy front fields for the current datamodule. They are
        # taken from the same loaded six-view tensor, so no second read or
        # independent augmentation is introduced.
        front_index = self.surround_camera_order.index(
            "front"
        )
        data["rgb"] = processed[:, front_index]
        data["rgb_org_size"] = (
            processed_org_size[:, front_index]
        )

        return data


class SurroundBaseDataset(
    SurroundDatasetMixin,
    BaseDataset,
):
    """
    BaseDataset variant used by datasets that can directly inherit the common
    six-view implementation, including Data_LG.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_surround_dataset()
