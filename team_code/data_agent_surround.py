"""
Six-view surround-camera data collection agent for SimLingo.

This agent extends the existing DataAgent and records:

1. Six synchronized, non-augmented surround RGB views:
   - front
   - front_left
   - front_right
   - rear
   - rear_left
   - rear_right

2. The existing top_rgb bird's-eye RGB image.

3. All non-augmented data saved by the parent DataAgent:
   - rgb
   - top_rgb
   - bev_static_masks
   - bev_dynamic_masks
   - bev_traffic_masks
   - bev_meta
   - boxes

The original geometric camera augmentation sensor and all augmented output
files are disabled. The parent DataAgent processing logic is retained by
aliasing the original front RGB input to the historical rgb_augmented key.
Because augmentation translation and rotation are both zero, this alias does
not introduce geometric augmentation.
"""

import gzip
import json
from typing import Dict, List

import cv2
import numpy as np

from data_agent import DataAgent


# The order must remain consistent in later dataloaders and closed-loop agents.
SURROUND_CAMERA_ORDER = (
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
)

# CARLA/Unreal vehicle coordinates:
# x points forward, y points right and z points upward.
SURROUND_CAMERA_YAWS = {
    "front": 0.0,
    "front_left": -60.0,
    "front_right": 60.0,
    "rear": 180.0,
    "rear_left": -120.0,
    "rear_right": 120.0,
}

# Sensors used only for the original geometric augmentation.
# top_rgb is deliberately retained.
AUGMENTED_SENSOR_IDS = {
    "rgb_augmented",
    "semantics_augmented",
    "depth_augmented",
}

# Output directories created by DataAgent.setup() but not used by this agent.
AUGMENTED_OUTPUT_FOLDERS = (
    "rgb_augmented",
    "semantics_augmented",
    "depth_augmented",
    "bev_semantics_augmented",
    "bev_static_masks_augmented",
    "bev_static_debug_augmented",
    "bev_dynamic_masks_augmented",
    "bev_traffic_masks_augmented",
    "bev_meta_augmented",
)


def get_entry_point():
    return "SurroundDataAgent"


class SurroundDataAgent(DataAgent):
    """Collect six non-augmented surround views while retaining top_rgb."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(
            path_to_conf_file,
            route_index=route_index,
            traffic_manager=traffic_manager,
        )

        # Completely disable front-camera geometric augmentation.
        self.config.augment = False
        self.augmentation_translation = 0.0
        self.augmentation_rotation = 0.0

        # Keep the high-mounted top_rgb camera configured by the parent class.
        self.SAVE_TOP_RGB = 1

        self.surround_camera_specs = self._build_surround_camera_specs()

        if self.save_path is not None and self.datagen:
            # Remove only augmentation-related empty directories.
            for folder_name in AUGMENTED_OUTPUT_FOLDERS:
                folder_path = self.save_path / folder_name
                if folder_path.exists():
                    try:
                        folder_path.rmdir()
                    except OSError:
                        # Never remove a non-empty directory.
                        pass

            # Ensure all required non-augmented parent outputs exist.
            required_parent_folders = (
                "rgb",
                "top_rgb",
                "boxes",
                "bev_static_masks",
                "bev_dynamic_masks",
                "bev_traffic_masks",
                "bev_meta",
            )
            for folder_name in required_parent_folders:
                (self.save_path / folder_name).mkdir(
                    parents=True,
                    exist_ok=True,
                )

            # Create the six surround-view output directories.
            for camera_name in SURROUND_CAMERA_ORDER:
                (self.save_path / f"rgb_{camera_name}").mkdir(
                    parents=True,
                    exist_ok=True,
                )

            self._save_surround_camera_metadata()

    def _build_surround_camera_specs(self) -> Dict[str, Dict]:
        """
        Build the six-view surround camera configuration.

        All six cameras use the original SimLingo front-camera position,
        resolution and field of view. Only the yaw angle changes.
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
                    "rgb"
                    if camera_name == "front"
                    else f"rgb_{camera_name}"
                ),
                "save_folder": f"rgb_{camera_name}",
            }

        return specs

    def _save_surround_camera_metadata(self) -> None:
        """Save surround-view and top-view camera metadata."""
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
            "top_rgb": {
                "enabled": bool(self.SAVE_TOP_RGB),
                "position": [0.0, 0.0, float(self.top_camera_z)],
                "rotation": [0.0, -90.0, 0.0],
                "width": int(self.top_camera_width),
                "height": int(self.top_camera_height),
                "fov": float(self.top_camera_fov),
                "sensor_id": "top_rgb",
                "save_folder": "top_rgb",
            },
        }

        metadata_path = self.save_path / "surround_camera_config.json"
        with metadata_path.open("w", encoding="utf-8") as file_obj:
            json.dump(
                metadata,
                file_obj,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def _make_rgb_sensor(spec: Dict) -> Dict:
        """Convert one view specification to a CARLA RGB sensor dictionary."""
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
        Retain all original non-augmented sensors, including top_rgb, remove the
        geometric-augmentation sensors, and add five surround RGB cameras.
        """
        sensors = super().sensors()

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

            # The original rgb sensor is reused as the front camera.
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
        Reuse the complete parent DataAgent processing and add six-view images.

        The current parent tick() still reads rgb_augmented. Because that sensor
        is intentionally not registered, the original front input is aliased to
        the expected key. Augmentation translation and rotation are both zero.
        """
        if "rgb" in input_data and "rgb_augmented" not in input_data:
            input_data["rgb_augmented"] = input_data["rgb"]

        # Compatibility with parent variants that may read these keys.
        if self.SAVE_TF_LABELS:
            if (
                "semantics" in input_data
                and "semantics_augmented" not in input_data
            ):
                input_data["semantics_augmented"] = input_data["semantics"]

            if (
                "depth" in input_data
                and "depth_augmented" not in input_data
            ):
                input_data["depth_augmented"] = input_data["depth"]

        result = super().tick(input_data)

        # Remove all augmentation-only outputs before saving or returning data.
        result.pop("rgb_augmented", None)
        result.pop("semantics_augmented", None)
        result.pop("depth_augmented", None)
        result.pop("bev_semantics_augmented", None)
        result.pop("bev_static_masks_augmented", None)
        result.pop("bev_dynamic_masks_augmented", None)
        result.pop("bev_traffic_masks_augmented", None)
        result.pop("bev_meta_augmented", None)

        if self.save_path is not None and (self.datagen or self.tmp_visu):
            surround_images = {
                "front": result["rgb"],
            }

            for camera_name in SURROUND_CAMERA_ORDER:
                if camera_name == "front":
                    continue

                sensor_id = self.surround_camera_specs[camera_name]["sensor_id"]
                if sensor_id not in input_data:
                    raise KeyError(
                        f"Missing surround-camera sensor input: {sensor_id}"
                    )

                surround_images[camera_name] = (
                    input_data[sensor_id][1][:, :, :3]
                )
        else:
            surround_images = {
                camera_name: None
                for camera_name in SURROUND_CAMERA_ORDER
            }

        result["surround_rgb"] = surround_images
        for camera_name, image in surround_images.items():
            result[f"rgb_{camera_name}"] = image

        return result

    @staticmethod
    def _write_image(path, image, jpeg_parameters=None):
        """Write an image and raise an explicit error if OpenCV fails."""
        if image is None:
            raise RuntimeError(f"Cannot save an empty image: {path}")

        if jpeg_parameters is None:
            success = cv2.imwrite(str(path), image)
        else:
            success = cv2.imwrite(
                str(path),
                image,
                jpeg_parameters,
            )

        if not success:
            raise IOError(f"Failed to save image: {path}")

    def save_sensors(self, tick_data):
        """
        Save six surround views, top_rgb and every non-augmented output that the
        current parent DataAgent saves.
        """
        frame = self.step // self.config.data_save_freq
        frame_name = f"{frame:04d}"
        jpeg_parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]

        # Legacy front-view path retained for existing SimLingo/LG pipelines.
        self._write_image(
            self.save_path / "rgb" / f"{frame_name}.jpg",
            tick_data["rgb"],
            jpeg_parameters,
        )

        # Six synchronized surround views.
        for camera_name in SURROUND_CAMERA_ORDER:
            image = tick_data.get(f"rgb_{camera_name}")
            save_path = (
                self.save_path
                / self.surround_camera_specs[camera_name]["save_folder"]
                / f"{frame_name}.jpg"
            )
            self._write_image(
                save_path,
                image,
                jpeg_parameters,
            )

        # Existing high-mounted top-view RGB image.
        if self.SAVE_TOP_RGB:
            top_rgb = tick_data.get("top_rgb")
            self._write_image(
                self.save_path / "top_rgb" / f"{frame_name}.jpg",
                top_rgb,
                jpeg_parameters,
            )

        if self.SAVE_TF_LABELS:
            # 1. Static environment masks.
            static_masks = tick_data["bev_static_masks"]
            np.savez_compressed(
                str(
                    self.save_path
                    / "bev_static_masks"
                    / f"{frame_name}.npz"
                ),
                road=static_masks["road"],
                sidewalk=static_masks["sidewalk"],
                lane_all=static_masks["lane_all"],
                lane_broken=static_masks["lane_broken"],
                lane_solid=static_masks["lane_solid"],
            )

            # 2. Current dynamic actor masks.
            dynamic_masks = tick_data["bev_dynamic_masks"]
            np.savez_compressed(
                str(
                    self.save_path
                    / "bev_dynamic_masks"
                    / f"{frame_name}.npz"
                ),
                vehicle=dynamic_masks["vehicle"],
                walker=dynamic_masks["walker"],
                actor=dynamic_masks["actor"],
            )

            # 3. Traffic-light and stop-line masks.
            traffic_masks = tick_data["bev_traffic_masks"]
            np.savez_compressed(
                str(
                    self.save_path
                    / "bev_traffic_masks"
                    / f"{frame_name}.npz"
                ),
                stop=traffic_masks["stop"],
                tl_green=traffic_masks["tl_green"],
                tl_yellow=traffic_masks["tl_yellow"],
                tl_red=traffic_masks["tl_red"],
            )

            # 4. BEV metadata.
            with gzip.open(
                self.save_path
                / "bev_meta"
                / f"{frame_name}.json.gz",
                "wt",
                encoding="utf-8",
            ) as file_obj:
                json.dump(
                    tick_data["bev_meta"],
                    file_obj,
                    indent=4,
                )

        # 5. Bounding-box and ego-state information.
        with gzip.open(
            self.save_path
            / "boxes"
            / f"{frame_name}.json.gz",
            "wt",
            encoding="utf-8",
        ) as file_obj:
            json.dump(
                tick_data["bounding_boxes"],
                file_obj,
                indent=4,
            )
