"""
Six-view surround-camera data collection agent for SimLingo.

This agent extends the existing ``DataAgent`` and records six synchronized,
non-augmented RGB views:

    front
    front_left
    front_right
    rear
    rear_left
    rear_right

The original geometric camera augmentation is explicitly disabled. The legacy
``rgb`` folder is still saved as the front view for compatibility with current
SimLingo/LG scripts, while the six-view images are additionally saved under
``rgb_<view_name>`` folders.
"""

import gzip
import json
from typing import Dict, List

import cv2
import numpy as np

from data_agent import DataAgent


# Fixed order that must also be used by the later dataloader and closed-loop agent.
SURROUND_CAMERA_ORDER = (
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
)

# CARLA/Unreal vehicle coordinates: x-forward, y-right, z-up.
# Negative yaw looks toward the vehicle's left side.
SURROUND_CAMERA_YAWS = {
    "front": 0.0,
    "front_left": -60.0,
    "front_right": 60.0,
    "rear": 180.0,
    "rear_left": -120.0,
    "rear_right": 120.0,
}

# Sensors created by the original DataAgent solely for geometric augmentation.
AUGMENTED_SENSOR_IDS = {
    "rgb_augmented",
    "semantics_augmented",
    "depth_augmented",
}

# Empty directories created by the original DataAgent setup but no longer used.
AUGMENTED_OUTPUT_FOLDERS = (
    "rgb_augmented",
    "semantics_augmented",
    "depth_augmented",
    "bev_semantics_augmented",
    "bev_static_masks_augmented",
    "bev_static_debug_augmented",
)


def get_entry_point():
    return "SurroundDataAgent"


class SurroundDataAgent(DataAgent):
    """
    DataAgent variant that records a non-augmented six-camera surround rig.

    The original ``rgb`` sensor is reused as ``front``. Five additional RGB
    cameras provide the other views. No camera is translated or rotated by the
    original SimLingo geometric augmentation.
    """

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(
            path_to_conf_file,
            route_index=route_index,
            traffic_manager=traffic_manager,
        )

        # Disable the original shifted/rotated camera augmentation.
        self.config.augment = False
        self.augmentation_translation = 0.0
        self.augmentation_rotation = 0.0

        self.surround_camera_specs = self._build_surround_camera_specs()

        if self.save_path is not None and self.datagen:
            # Remove unused augmentation directories created by DataAgent.setup().
            # They are empty at this point, so rmdir is safe.
            for folder_name in AUGMENTED_OUTPUT_FOLDERS:
                folder_path = self.save_path / folder_name
                if folder_path.exists():
                    try:
                        folder_path.rmdir()
                    except OSError:
                        # Keep a non-empty legacy directory instead of deleting data.
                        pass

            for camera_name in SURROUND_CAMERA_ORDER:
                (self.save_path / f"rgb_{camera_name}").mkdir(
                    parents=True,
                    exist_ok=True,
                )

            self._save_surround_camera_metadata()

    def _build_surround_camera_specs(self) -> Dict[str, Dict]:
        """
        Build a shared-calibration six-view camera rig.

        All cameras use the original SimLingo camera position, image size and
        field of view. Only their yaw angles differ. Adjacent 110-degree views
        overlap, providing complete 360-degree visual coverage.
        """
        position = [float(value) for value in self.config.camera_pos]
        base_rotation = [float(value) for value in self.config.camera_rot_0]

        specs = {}
        for camera_name in SURROUND_CAMERA_ORDER:
            rotation = base_rotation.copy()
            rotation[2] = float(SURROUND_CAMERA_YAWS[camera_name])

            specs[camera_name] = {
                "position": position.copy(),
                "rotation": rotation,
                "width": int(self.config.camera_width),
                "height": int(self.config.camera_height),
                "fov": float(self.config.camera_fov),
                "sensor_id": (
                    "rgb" if camera_name == "front"
                    else f"rgb_{camera_name}"
                ),
                "save_folder": f"rgb_{camera_name}",
            }

        return specs

    def _save_surround_camera_metadata(self) -> None:
        """Save camera order and calibration in each collected route folder."""
        metadata = {
            "geometric_augmentation": False,
            "coordinate_system": {
                "description": "CARLA/Unreal vehicle coordinates",
                "x": "forward",
                "y": "right",
                "z": "up",
                "rotation_order": ["roll", "pitch", "yaw"],
                "angle_unit": "degree",
            },
            "camera_order": list(SURROUND_CAMERA_ORDER),
            "cameras": self.surround_camera_specs,
        }

        metadata_path = self.save_path / "surround_camera_config.json"
        with metadata_path.open("w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2, ensure_ascii=False)

    @staticmethod
    def _make_rgb_sensor(spec: Dict) -> Dict:
        """Convert one camera specification into a CARLA sensor dictionary."""
        return {
            "type": "sensor.camera.rgb",
            "x": spec["position"][0],
            "y": spec["position"][1],
            "z": spec["position"][2],
            "roll": spec["rotation"][0],
            "pitch": spec["rotation"][1],
            "yaw": spec["rotation"][2],
            "width": spec["width"],
            "height": spec["height"],
            "fov": spec["fov"],
            "id": spec["sensor_id"],
        }

    def sensors(self) -> List[Dict]:
        """
        Keep the original sensors, remove augmentation-only cameras, and add the
        five non-front surround RGB cameras.
        """
        sensors = super().sensors()

        # Remove shifted/rotated RGB, semantic and depth cameras.
        sensors = [
            sensor
            for sensor in sensors
            if not (
                isinstance(sensor, dict)
                and sensor.get("id") in AUGMENTED_SENSOR_IDS
            )
        ]

        if self.save_path is not None and (self.datagen or self.tmp_visu):
            existing_ids = {
                sensor.get("id")
                for sensor in sensors
                if isinstance(sensor, dict)
            }

            for camera_name in SURROUND_CAMERA_ORDER:
                if camera_name == "front":
                    continue

                spec = self.surround_camera_specs[camera_name]
                sensor_id = spec["sensor_id"]

                if sensor_id in existing_ids:
                    raise RuntimeError(
                        f"Duplicate CARLA sensor id detected: {sensor_id}"
                    )

                sensors.append(self._make_rgb_sensor(spec))
                existing_ids.add(sensor_id)

        return sensors

    def tick(self, input_data):
        """
        Run the original DataAgent processing with zero augmentation and append
        six non-augmented RGB images.

        DataAgent.tick() still expects its historical augmented input keys.
        They are aliased to the original, unmodified sensors here. This does not
        create any geometric augmentation and keeps the parent implementation
        compatible without duplicating its LiDAR/BEV/bounding-box logic.
        """
        if "rgb" in input_data and "rgb_augmented" not in input_data:
            input_data["rgb_augmented"] = input_data["rgb"]

        if self.SAVE_TF_LABELS:
            if (
                "semantics" in input_data
                and "semantics_augmented" not in input_data
            ):
                input_data["semantics_augmented"] = input_data["semantics"]

            if "depth" in input_data and "depth_augmented" not in input_data:
                input_data["depth_augmented"] = input_data["depth"]

        result = super().tick(input_data)

        # Do not expose duplicate augmentation outputs downstream.
        result.pop("rgb_augmented", None)
        result.pop("semantics_augmented", None)
        result.pop("depth_augmented", None)
        result.pop("bev_semantics_augmented", None)
        result.pop("bev_static_masks_augmented", None)

        surround_images = {}

        if self.save_path is not None and (self.datagen or self.tmp_visu):
            surround_images["front"] = result["rgb"]

            for camera_name in SURROUND_CAMERA_ORDER:
                if camera_name == "front":
                    continue

                sensor_id = self.surround_camera_specs[camera_name]["sensor_id"]
                if sensor_id not in input_data:
                    raise KeyError(
                        f"Missing surround-camera sensor input: {sensor_id}"
                    )

                surround_images[camera_name] = input_data[sensor_id][1][:, :, :3]
        else:
            surround_images = {
                camera_name: None
                for camera_name in SURROUND_CAMERA_ORDER
            }

        result["surround_rgb"] = surround_images
        for camera_name, image in surround_images.items():
            result[f"rgb_{camera_name}"] = image

        return result

    def save_sensors(self, tick_data):
        """
        Save the original non-augmented front image, six surround views, BEV
        static masks/debug images and bounding boxes.

        No ``*_augmented`` image, semantic, depth or BEV files are written.
        """
        frame = self.step // self.config.data_save_freq
        frame_name = f"{frame:04d}"

        # Legacy front-view path retained for current SimLingo/LG scripts.
        if not cv2.imwrite(
            str(self.save_path / "rgb" / f"{frame_name}.jpg"),
            tick_data["rgb"],
        ):
            raise IOError("Failed to save legacy front RGB image.")

        jpeg_parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
        for camera_name in SURROUND_CAMERA_ORDER:
            image = tick_data.get(f"rgb_{camera_name}")
            if image is None:
                raise RuntimeError(
                    f"Cannot save empty surround view: {camera_name}"
                )

            save_path = (
                self.save_path
                / self.surround_camera_specs[camera_name]["save_folder"]
                / f"{frame_name}.jpg"
            )
            saved = cv2.imwrite(
                str(save_path),
                image,
                jpeg_parameters,
            )
            if not saved:
                raise IOError(f"Failed to save surround image: {save_path}")

        if self.SAVE_TF_LABELS:
            masks = tick_data["bev_static_masks"]
            np.savez_compressed(
                str(
                    self.save_path
                    / "bev_static_masks"
                    / f"{frame_name}.npz"
                ),
                road=masks["road"],
                sidewalk=masks["sidewalk"],
                lane_all=masks["lane_all"],
                lane_broken=masks["lane_broken"],
                lane_solid=masks["lane_solid"],
            )

            self.save_bev_static_debug_png(
                self.save_path
                / "bev_static_debug"
                / f"{frame_name}.png",
                masks,
            )

        with gzip.open(
            self.save_path / "boxes" / f"{frame_name}.json.gz",
            "wt",
            encoding="utf-8",
        ) as file_obj:
            json.dump(tick_data["bounding_boxes"], file_obj, indent=4)
