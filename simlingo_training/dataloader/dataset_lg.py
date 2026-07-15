"""LG supervision adapter for SimLingo training.

This module intentionally reuses SimLingo's existing ``dreamer_dataset`` slot.
The normal driving dataset, Dreamer dataset, DataModule, model, and losses are
not changed. Select this dataset through Hydra only for LG experiments.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import gzip
import random

import cv2
import numpy as np
import torch
import ujson

from simlingo_training.dataloader.dataset_base import BaseDataset
from simlingo_training.utils.custom_types import DatasetOutput


VIZ_DATA = False


class Data_LG(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """Load four-question LG language and selected LG waypoint supervision."""

    def __init__(self, **cfg):
        if not bool(cfg.get("use_lg_supervision", False)):
            raise ValueError(
                "Data_LG was selected but use_lg_supervision is False. "
                "Set data_module.dreamer_dataset.use_lg_supervision=true, "
                "or switch back to the original Data_Dreamer target."
            )

        # Data_LG performs its own language construction, so DriveLM QA and
        # commentary initialization are unnecessary for this auxiliary source.
        # These overrides are local to Data_LG and do not affect Data_Driving.
        base_cfg = dict(cfg)
        base_cfg["use_qa"] = False
        base_cfg["use_commentary"] = False

        # Data_Dreamer always uses the official routes_training/routes_validation
        # split. Reproduce that split for a fair Dreamer-vs-LG comparison.
        if bool(base_cfg.get("lg_match_dreamer_split", True)):
            base_cfg["use_town13"] = False

        super().__init__(dreamer=False, **base_cfg)

        self.lg_label_folder = str(getattr(self, "lg_label_folder", "language_grounded_waypoints"))
        self.lg_question_keys = tuple(
            getattr(
                self,
                "lg_question_keys",
                ("attention", "motion_constraint", "driving_response", "future_motion"),
            )
        )
        self._filter_samples_with_valid_lg_labels()

    # ---------------------------------------------------------------------
    # Index construction and validation
    # ---------------------------------------------------------------------
    @staticmethod
    def _decode_path(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return value.tobytes().decode("utf-8")
        return str(value)

    def _measurement_dir_for_index(self, index: int) -> Path:
        measurement_entry = self.measurements[index]
        if isinstance(measurement_entry, np.ndarray):
            measurement_entry = measurement_entry.reshape(-1)[0]
        elif isinstance(measurement_entry, (list, tuple)):
            measurement_entry = measurement_entry[0]
        return Path(self._decode_path(measurement_entry))

    def _current_frame_for_index(self, index: int) -> int:
        return int(self.sample_start[index]) + int(self.hist_len) - 1

    def _lg_path_for_index(self, index: int) -> Path:
        measurement_dir = self._measurement_dir_for_index(index)
        route_dir = measurement_dir.parent
        frame_id = self._current_frame_for_index(index)
        return route_dir / self.lg_label_folder / f"{frame_id:04d}.json.gz"

    @staticmethod
    def _load_gzip_json(path: Path) -> Dict:
        with gzip.open(path, "rt", encoding="utf-8") as file_obj:
            payload = ujson.load(file_obj)
        if not isinstance(payload, dict):
            raise ValueError("LG label root must be a dictionary")
        return payload

    def _extract_questions(self, payload: Dict) -> List[Dict[str, str]]:
        questions: List[Dict[str, str]] = []
        core = payload.get("core_questions", {})

        if isinstance(core, dict):
            for key in self.lg_question_keys:
                item = core.get(key)
                if not isinstance(item, dict):
                    questions = []
                    break
                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                if not question or not answer:
                    questions = []
                    break
                questions.append({"key": key, "question": question, "answer": answer})

        # Backward-compatible fallback. New labels should normally use the
        # synchronized core_questions structure produced by processor.py.
        if len(questions) != len(self.lg_question_keys):
            qa_pairs = payload.get("language_annotation", {}).get("qa_pairs_en", [])
            if isinstance(qa_pairs, list) and len(qa_pairs) == len(self.lg_question_keys):
                fallback_questions = []
                for key, item in zip(self.lg_question_keys, qa_pairs):
                    if not isinstance(item, dict):
                        fallback_questions = []
                        break
                    question = str(item.get("question", "")).strip()
                    answer = str(item.get("answer", "")).strip()
                    if not question or not answer:
                        fallback_questions = []
                        break
                    fallback_questions.append(
                        {"key": key, "question": question, "answer": answer}
                    )
                questions = fallback_questions

        return questions

    def _extract_lg_waypoints(self, payload: Dict) -> np.ndarray:
        supervision = payload.get("supervision", {})
        waypoints = np.asarray(
            supervision.get("risk_planned_waypoints", []),
            dtype=np.float32,
        )
        return waypoints

    def _validate_payload(self, payload: Dict) -> Tuple[bool, str]:
        supervision = payload.get("supervision", {})
        if not isinstance(supervision, dict):
            return False, "missing_supervision"

        if bool(getattr(self, "lg_require_risk_label_valid", True)):
            if not bool(supervision.get("risk_label_valid", False)):
                return False, "risk_label_invalid"

        if bool(getattr(self, "lg_skip_expert_fallback", True)):
            internal_name = str(supervision.get("selected_internal_intent_name", ""))
            if internal_name in {"expert_fallback", "stationary_hold_fallback"}:
                return False, f"fallback:{internal_name}"

        if bool(getattr(self, "lg_use_waypoints", True)):
            waypoints = self._extract_lg_waypoints(payload)
            expected_count = int(self.pred_len) - 1
            if waypoints.shape != (expected_count, 2):
                return False, f"waypoint_shape:{tuple(waypoints.shape)}"
            if not np.isfinite(waypoints).all():
                return False, "waypoint_non_finite"
            max_abs = float(getattr(self, "lg_max_abs_waypoint_m", 100.0))
            if np.max(np.abs(waypoints), initial=0.0) > max_abs:
                return False, "waypoint_out_of_range"

        language_mode = str(getattr(self, "lg_language_mode", "four_questions")).lower()
        require_questions = bool(getattr(self, "lg_require_four_questions", True))
        use_language = bool(getattr(self, "lg_use_language", True))
        if use_language and language_mode != "none" and require_questions:
            questions = self._extract_questions(payload)
            if len(questions) != len(self.lg_question_keys):
                return False, "incomplete_four_questions"

        return True, "ok"

    def _filter_samples_with_valid_lg_labels(self) -> None:
        valid_indices: List[int] = []
        valid_paths: List[str] = []
        reasons = Counter()

        for index in range(len(self.images)):
            label_path = self._lg_path_for_index(index)
            if not label_path.is_file():
                reasons["missing_label"] += 1
                continue

            try:
                payload = self._load_gzip_json(label_path)
                valid, reason = self._validate_payload(payload)
            except (OSError, EOFError, ValueError, ujson.JSONDecodeError) as exc:
                reasons[f"read_error:{type(exc).__name__}"] += 1
                continue

            if not valid:
                reasons[reason] += 1
                continue

            valid_indices.append(index)
            valid_paths.append(str(label_path))

        indices = np.asarray(valid_indices, dtype=np.int64)
        original_sample_count = len(self.images)

        for attribute in (
            "images",
            "boxes",
            "measurements",
            "sample_start",
            "augment_exists",
        ):
            values = getattr(self, attribute)

            if len(values) != original_sample_count:
                raise RuntimeError(
                    f"LG index container length mismatch: {attribute} has "
                    f"{len(values)} entries, expected {original_sample_count}"
                )

            # BaseDataset converts images, boxes, measurements, and sample_start
            # to NumPy arrays, but currently leaves augment_exists as a Python
            # list. NumPy advanced indexing works only for the former, so handle
            # both container types explicitly.
            if isinstance(values, np.ndarray):
                filtered_values = values[indices]
            else:
                filtered_values = [values[int(index)] for index in valid_indices]

            # Keep a uniform indexable representation after LG filtering.
            if attribute == "augment_exists":
                filtered_values = np.asarray(filtered_values, dtype=np.bool_)

            setattr(self, attribute, filtered_values)

        self.lg_label_paths = np.asarray(valid_paths, dtype=np.string_)

        if bool(getattr(self, "lg_print_filter_summary", True)):
            print(
                f"[{self.split} LG samples]: kept {len(valid_indices)} samples; "
                f"filtered={dict(reasons)}"
            )

    # ---------------------------------------------------------------------
    # Language and trajectory construction
    # ---------------------------------------------------------------------
    @staticmethod
    def _compute_waypoints_1d(waypoints: np.ndarray) -> np.ndarray:
        points = np.asarray(waypoints, dtype=np.float32)
        points_with_origin = np.concatenate(
            [np.zeros((1, 2), dtype=np.float32), points],
            axis=0,
        )
        segment_lengths = np.linalg.norm(np.diff(points_with_origin, axis=0), axis=1)
        cumulative = np.cumsum(segment_lengths)
        return np.stack([cumulative, np.zeros_like(cumulative)], axis=1).astype(np.float32)

    def _build_language_text(
        self,
        payload: Dict,
        prefix: str,
    ) -> Tuple[str, str]:
        use_language = bool(getattr(self, "lg_use_language", True))
        mode = str(getattr(self, "lg_language_mode", "four_questions")).lower()

        if not use_language or mode == "none":
            return f"{prefix} Predict the waypoints.", "Waypoints:"

        questions = self._extract_questions(payload)
        if len(questions) != len(self.lg_question_keys):
            raise ValueError("LG label does not contain the required four questions")

        if mode == "random_question":
            # Training keeps random question sampling. Validation always uses
            # the first configured question so that the same sample receives
            # exactly the same language input at every validation epoch.
            selected = (
                random.choice(questions)
                if self.split == "train"
                else questions[0]
            )
            prompt = f"{prefix} Q: {selected['question']} Then predict the waypoints."
            answer = f"A: {selected['answer']} Waypoints:"
            return prompt, answer

        if mode != "four_questions":
            raise ValueError(
                f"Unsupported lg_language_mode={mode!r}; expected four_questions, "
                "random_question, or none"
            )

        question_text = " ".join(
            f"Q{idx}: {item['question']}" for idx, item in enumerate(questions, start=1)
        )
        answer_text = " ".join(
            f"A{idx}: {item['answer']}" for idx, item in enumerate(questions, start=1)
        )
        prompt = (
            f"{prefix} Answer the four driving questions in order and then "
            f"predict the waypoints. {question_text}"
        )
        answer = f"{answer_text} Waypoints:"
        return prompt, answer

    def __getitem__(self, index):
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        lg_path = Path(self._decode_path(self.lg_label_paths[index]))

        loaded_measurements, current_measurement, measurement_file_current = (
            self.load_current_and_future_measurements(measurements, sample_start)
        )
        data["measurement_path"] = measurement_file_current

        # LG labels are generated in the original ego frame. Therefore geometric
        # image-shift augmentation is deliberately disabled for LG samples.
        aug_rotation = 0.0
        aug_translation = 0.0
        augment_sample = False

        data = self.load_waypoints(
            data,
            loaded_measurements,
            aug_translation,
            aug_rotation,
        )
        data["speed"] = current_measurement["speed"]
        speed_rounded = round(current_measurement["speed"], 1)

        data = self.load_route(
            data,
            current_measurement,
            aug_translation,
            aug_rotation,
        )

        target_point = np.asarray(current_measurement["target_point"], dtype=np.float32)
        target_point = self.augment_target_point(
            target_point,
            y_augmentation=aug_translation,
            yaw_augmentation=aug_rotation,
        )
        next_target_point = np.asarray(
            current_measurement["target_point_next"],
            dtype=np.float32,
        )
        next_target_point = self.augment_target_point(
            next_target_point,
            y_augmentation=aug_translation,
            yaw_augmentation=aug_rotation,
        )

        target_options, placeholder_values = self.get_navigational_conditioning(
            data,
            current_measurement,
            target_point,
            next_target_point,
        )

        payload = self._load_gzip_json(lg_path)
        valid, reason = self._validate_payload(payload)
        if not valid:
            raise ValueError(f"Invalid LG label at {lg_path}: {reason}")

        if bool(getattr(self, "lg_use_waypoints", True)):
            waypoints = self._extract_lg_waypoints(payload)
        else:
            waypoints = np.asarray(data["waypoints_org"], dtype=np.float32)
        waypoints_1d = self._compute_waypoints_1d(waypoints)

        # Keep SimLingo's original route target for the main comparison. Only the
        # language and waypoint supervision source changes.
        path = np.asarray(data["route_adjusted_org"], dtype=np.float32)

        prefix = f"Current speed: {speed_rounded} m/s."
        if (
            bool(getattr(self, "lg_include_navigation_conditioning", True))
            and len(target_options) > 0
        ):
            # Preserve SimLingo's random navigation wording during training,
            # but keep validation prompts deterministic across epochs.
            navigation_text = (
                random.choice(target_options)
                if self.split == "train"
                else target_options[0]
            )
            prefix = f"{prefix} {navigation_text}"

        prompt, answer = self._build_language_text(payload, prefix)
        prompt = prompt.replace("..", ".").replace("  ", " ").strip()
        answer = answer.replace("..", ".").replace("  ", " ").strip()

        data = self.load_images(data, images, augment_sample=augment_sample)

        conversation_answer = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        ]
        conversation_all = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        data_new = DatasetOutput(
            conversation=conversation_all,
            answer=conversation_answer,
            image_ff=data["rgb"],
            image_ff_org_size=data["rgb_org_size"],
            waypoints=waypoints,
            waypoints_1d=waypoints_1d,
            path=path,
            target_points=data["target_points"],
            speed=data["speed"],
            placeholder_values=placeholder_values,
            measurement_path=data["measurement_path"],
            # Keep the original dataset identifier so downstream SimLingo code
            # follows exactly the same path as Dreamer auxiliary samples.
            dataset="driving",
        )

        if VIZ_DATA:
            self.visualise_cameras(
                data_new,
                None,
                path,
                waypoints,
                options=None,
                name="lg_",
                prompt=prompt,
                answer=answer,
            )

        return data_new


if __name__ == "__main__":
    from hydra import compose, initialize

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    initialize(config_path="../config")
    cfg = compose(config_name="config")

    cfg.data_module.dreamer_dataset._target_ = (
        "simlingo_training.dataloader.dataset_lg.Data_LG"
    )
    cfg.data_module.dreamer_dataset.use_lg_supervision = True

    import hydra

    dataset = hydra.utils.instantiate(
        cfg.data_module.dreamer_dataset,
        split="train",
        bucket_name="all",
        **cfg.data_module,
        **cfg.data_module.base_dataset,
        _recursive_=False,
    )

    print(f"LG dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(sample.measurement_path)
        print(sample.waypoints.shape)
        print(sample.conversation)
