# -*- coding: utf-8 -*-

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gzip
import random

import cv2
import numpy as np
import torch
import ujson

from simlingo_training.dataloader.dataset_base_surround import (
    SurroundBaseDataset,
)
from simlingo_training.utils.custom_types import DatasetOutput


VIZ_DATA = False

# 必须与generate_future_interaction_grids.py中的保存顺序一致。
FUTURE_INTERACTION_CHANNEL_NAMES = (
    "selected_route_ego_footprint_occupancy",           # 参考路径footprint占用
    "future_ego_footprint_occupancy",                   # 未来ego footprint占用
    "primary_causal_actor_future_footprint_occupancy",  # 主要actor未来footprint占用
    "secondary_actor_future_footprint_occupancy",       # 次要actor未来footprint占用
)


class Data_LG(SurroundBaseDataset):

    def __init__(self, **cfg):
        
        # 如果决定使用 lg 数据,但未启用 use_lg_supervision,则抛出异常,提示用户修改配置
        if not bool(cfg.get("use_lg_supervision", False)):
            raise ValueError(
                "Data_LG was selected but use_lg_supervision is False. "
                "Set data_module.dreamer_dataset.use_lg_supervision=true, "
                "or switch back to the original Data_Dreamer target."
            )

        # 加载配置文件并修改
        base_cfg = dict(cfg)
        base_cfg["use_qa"] = False                 # lg 不使用 QA
        base_cfg["use_commentary"] = False         # lg 不使用 commentary
        base_cfg["img_shift_augmentation"] = False # lg 与普通 Driving 统一使用无几何增强六视角图像。

        #修改20260728：启用统一官方route划分后，
        # LG不再单独修改use_town13，保证LG与普通Driving使用同一划分逻辑。
        #
        # 只有未启用新的统一划分开关时，才保留旧的
        # lg_match_dreamer_split兼容行为。
        if (bool(base_cfg.get("lg_match_dreamer_split",True,)) and not bool(base_cfg.get("use_official_route_split",False,))):
            base_cfg["use_town13"] = False

        # 调用父类构造函数，初始化数据集
        super().__init__(dreamer=False, **base_cfg)

        # lg标签存放的文件夹名称
        self.lg_label_folder = str(getattr(self,"lg_label_folder","language_grounded_waypoints",))

        # 是否使用结构化未来世界标签
        self.lg_use_future_interaction_grid = bool(getattr( self, "lg_use_future_interaction_grid",False,))

        # 结构化未来世界标签存放的文件夹名称
        self.lg_future_interaction_grid_folder = str(getattr(self,"lg_future_interaction_grid_folder","future_interaction_grids",))

        # LG语言问题的键值顺序,必须与生成顺序一致
        self.lg_question_keys = tuple(
            getattr(
                self,
                "lg_question_keys",
                (
                    "attention",
                    "motion_constraint",
                    "driving_response",
                    "future_motion",
                ),
            )
        )

        self._filter_samples_with_valid_lg_labels()

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
        """
        返回当前样本的LG标签路径:
        ....../Town04_Rep0_Town04_lr_0_route0_07_25_20_57_48/language_grounded_waypoints/0026.json.gz
        """
        measurement_dir = self._measurement_dir_for_index(index)
        route_dir = measurement_dir.parent
        frame_id = self._current_frame_for_index(index)
        return (
            route_dir                   # ....../Town04_Rep0_Town04_lr_0_route0_07_25_20_57_48
            / self.lg_label_folder      # language_grounded_waypoints
            / f"{frame_id:04d}.json.gz" # 0026.json.gz
        )

    #修改20260726：获得当前帧的结构化未来世界标签路径。
    def _future_interaction_grid_path_for_index(self, index: int,) -> Path:
        measurement_dir = self._measurement_dir_for_index(index)
        route_dir = measurement_dir.parent
        frame_id = self._current_frame_for_index(index)
        return (
            route_dir
            / self.lg_future_interaction_grid_folder
            / f"{frame_id:04d}.npz"
        )

    @staticmethod
    def _load_gzip_json(path: Path) -> Dict:
        """
        从gzip压缩的JSON文件中加载LG标签数据
        """
        with gzip.open(path, "rt", encoding="utf-8") as file_obj:
            payload = ujson.load(file_obj)
        if not isinstance(payload, dict):
            raise ValueError("LG label root must be a dictionary")
        return payload

    #修改20260726：读取并检查四通道结构化未来世界标签。
    @staticmethod
    def _load_future_interaction_grid(path: Path,) -> Tuple[Optional[np.ndarray], bool]:
        if not path.is_file():
            return None, False

        with np.load(str(path), allow_pickle=False) as payload:
            if "future_interaction_grid" not in payload.files:
                raise ValueError(
                    f"Missing future_interaction_grid in {path}"
                )

            grid = np.asarray(
                payload["future_interaction_grid"],
                dtype=np.float32,
            )

            valid = False
            if "valid" in payload.files:
                valid_array = np.asarray(
                    payload["valid"]
                ).reshape(-1)
                if valid_array.size > 0:
                    valid = bool(valid_array[0])

            if "channel_names" in payload.files:
                channel_names = tuple(
                    str(item)
                    for item in np.asarray(
                        payload["channel_names"]
                    ).reshape(-1).tolist()
                )
                if channel_names != FUTURE_INTERACTION_CHANNEL_NAMES:
                    raise ValueError(
                        "Unexpected future interaction channel order "
                        f"in {path}: {channel_names}"
                    )

        if grid.ndim != 3 or grid.shape[0] != 4:
            raise ValueError(
                "future_interaction_grid must have shape "
                f"[4, H, W], but received {tuple(grid.shape)} "
                f"from {path}"
            )

        if not np.isfinite(grid).all():
            raise ValueError(
                f"future_interaction_grid contains non-finite values: {path}"
            )

        if np.any(grid < 0.0) or np.any(grid > 1.0):
            raise ValueError(
                "future_interaction_grid values must be within "
                f"[0, 1]: {path}"
            )

        return grid.astype(np.float32), valid

    def _extract_questions(self, payload: Dict) -> List[Dict[str, str]]:
        """
        返回LG标签中的四个问题及答案,如果不存在则返回空列表。
        """
        questions: List[Dict[str, str]] = []
        core = payload.get("core_questions", {})  # 从.json.gz文件中的 core_questions 字段中提取问题和答案

        if isinstance(core, dict):
            for key in self.lg_question_keys:
                item = core.get(key)
                if not isinstance(item, dict):
                    questions = []
                    break
                # 问题
                question = str(item.get("question", "")).strip()
                
                # 答案
                answer = str(item.get("answer", "")).strip()
                
                if not question or not answer:
                    questions = []
                    break
                
                questions.append(
                    {
                        "key": key,           # attention/motion_constraint/driving_response/future_motion
                        "question": question, # 对应的问题
                        "answer": answer,     # 对应的答案
                    }
                )

        # 这是一个兼容性检查,如果 core_questions 中没有四个问题,则尝试从 language_annotation.qa_pairs_en 中提取问题和答案
        if len(questions) != len(self.lg_question_keys):

            # 从 language_annotation.qa_pairs_en 中提取问题和答案
            qa_pairs = (payload.get("language_annotation", {}).get("qa_pairs_en", []))

            if (isinstance(qa_pairs, list) and len(qa_pairs) == len(self.lg_question_keys)):
                fallback_questions = []
                for key, item in zip(self.lg_question_keys, qa_pairs):
                    if not isinstance(item, dict):
                        fallback_questions = []
                        break
                    # 问题
                    question = str(item.get("question", "")).strip()
                    
                    # 答案
                    answer = str(item.get("answer", "")).strip()
                    if not question or not answer:
                        fallback_questions = []
                        break
                    fallback_questions.append(
                        {
                            "key": key,           # attention/motion_constraint/driving_response/future_motion
                            "question": question, # 对应的问题
                            "answer": answer,     # 对应的答案
                        }
                    )
                questions = fallback_questions

        return questions

    @staticmethod
    def _extract_lg_waypoints(payload: Dict) -> np.ndarray:
        """
        返回LG标签中的risk_planned_waypoints字段(规划出来的waypoints),如果不存在则返回空数组
        """
        supervision = payload.get("supervision", {})
        return np.asarray(supervision.get("risk_planned_waypoints", []),dtype=np.float32,)

    def _extract_lg_path(self, payload: Dict) -> np.ndarray:
        """
        返回LG标签中的reference.selected_reference_route字段(参考路径),如果不存在则返回空数组
        """
        reference = payload.get("reference", {})
        if not isinstance(reference, dict):
            return np.empty((0, 2), dtype=np.float32)

        path = np.asarray(reference.get("selected_reference_route", []),dtype=np.float32,)

        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
            return path

        path = self.equal_spacing_route(path)
        return np.asarray(path, dtype=np.float32)

    #修改20260720：读取并检查LG生成的六视角相机注意力软标签。
    @staticmethod
    def _extract_camera_attention_supervision(payload: Dict,) -> Tuple[np.ndarray, bool]:
        """
        返回LG标签中的六视角相机注意力软标签,如果不存在则返回空标签和无效标志
        """

        # 创建一个空的六视角注意力标签,如果后续检查失败,则返回该空标签和无效标志
        empty_target = np.zeros((6,), dtype=np.float32)

        # 检查 visual_grounding 字段是否存在且为字典类型
        visual_grounding = payload.get("visual_grounding", {})
        if not isinstance(visual_grounding, dict):
            return empty_target, False

        # 
        if not bool(visual_grounding.get("camera_attention_valid",False,)):
            return empty_target, False

        # 预期相机的顺序
        expected_camera_order = ("front","front_left","front_right","rear","rear_left","rear_right",)
        # 从 visual_grounding 中获取 camera_order 字段
        camera_order = tuple(visual_grounding.get("camera_order", []))
        # 如果相机顺序不一致,则返回空标签和无效标志
        if camera_order != expected_camera_order:
            return empty_target, False

        target = np.asarray(visual_grounding.get("camera_attention_target",[],),dtype=np.float32,)

        # 如果 target 的形状不为 (6,), 或者包含非有限值,或者包含负值,则返回空标签和无效标志
        if target.shape != (6,):
            return empty_target, False
        if not np.isfinite(target).all():
            return empty_target, False
        if np.any(target < 0.0):
            return empty_target, False

        # 如果 target 的和小于等于 0,则返回空标签和无效标志
        target_sum = float(target.sum())
        if target_sum <= 0.0:
            return empty_target, False

        # 归一化 target,使其和为 1,并返回有效标志
        target = target / target_sum

        return target.astype(np.float32), True

    def _validate_payload(self, payload: Dict) -> Tuple[bool, str]:
        """
        检查LG标签数据的有效性,返回是否有效以及无效原因
        """

        # 1. 检查supervision字段是否存在且为字典类型
        supervision = payload.get("supervision", {})
        if not isinstance(supervision, dict):
            return False, "missing_supervision"

        # 2. 检查 risk_label_valid 字段是否存在且为 True (True 表示在标签生成的时候通过安全性和可行性检查)
        if bool(getattr(self, "lg_require_risk_label_valid", True)):
            if not bool(supervision.get("risk_label_valid", False)):
                return False, "risk_label_invalid"

        # 3. 检查是否跳过专家回退样本
        if bool(getattr(self, "lg_skip_expert_fallback", True)):
            internal_name = str(supervision.get("selected_internal_intent_name","",))
            if internal_name in {
                "expert_fallback",          # 专家回退
                "stationary_hold_fallback", # 静止保持回退
            }:
                return False, f"fallback:{internal_name}"

        # 检查 waypoints 和 path 的有效性
        if bool(getattr(self, "lg_use_waypoints", True)):

            # lg规划出来的 waypoints
            waypoints = self._extract_lg_waypoints(payload)
            
            # 数量应该为 10 个
            expected_count = int(self.pred_len) - 1
            
            # 检查 waypoints 数量对不对
            if waypoints.shape != (expected_count, 2):
                return False, f"waypoint_shape:{tuple(waypoints.shape)}"
            
            # 检查 waypoints 是否包含非有限值
            if not np.isfinite(waypoints).all():
                return False, "waypoint_non_finite"
            
            # 检查 waypoints 是否超出范围
            max_abs = float(getattr(self, "lg_max_abs_waypoint_m", 100.0))
            if np.max(np.abs(waypoints), initial=0.0) > max_abs:
                return False, "waypoint_out_of_range"

            # lg标签中选择出来的参考路径
            path = self._extract_lg_path(payload)

            # 检查参考路径的形状
            if path.shape != (20, 2):
                return False, f"path_shape:{tuple(path.shape)}"
            
            # 检查参考路径是否包含非有限值
            if not np.isfinite(path).all():
                return False, "path_non_finite"
            
            # 检查参考路径是否超出范围
            if np.max(np.abs(path), initial=0.0) > max_abs:
                return False, "path_out_of_range"

        # 语言模式(四问题)
        language_mode = str(getattr(self,"lg_language_mode","four_questions",)).lower()
        
        # 是否要求必须为四个问题
        require_questions = bool(getattr(self, "lg_require_four_questions", True))

        # 是否启用语言描述
        use_language = bool(getattr(self, "lg_use_language", True))

        # 检查语言描述的有效性
        if (use_language and language_mode != "none" and require_questions):
            questions = self._extract_questions(payload)
            if len(questions) != len(self.lg_question_keys):
                return False, "incomplete_four_questions"

        return True, "ok"

    @staticmethod
    def _filter_container(values, indices, valid_indices):
        if isinstance(values, np.ndarray):
            return values[indices]
        return [values[int(index)] for index in valid_indices]

    def _filter_samples_with_valid_lg_labels(self) -> None:

        valid_indices: List[int] = []
        valid_paths: List[str] = []
        reasons = Counter()

        # 统计真正进入当前LG数据集的样本中,有效六视角注意力监督的数量及无效原因。
        camera_attention_valid_count = 0  # 有效六视角注意力监督数量
        camera_attention_invalid_reasons = Counter()  # 无效六视角注意力监督的原因统计

        for index in range(len(self.images)):

            # ....../Town04_Rep0_Town04_lr_0_route0_07_25_20_57_48/language_grounded_waypoints/0026.json.gz
            label_path = self._lg_path_for_index(index)
            # 文件缺失
            if not label_path.is_file():
                reasons["missing_label"] += 1
                continue

            try:
                # 加载lg标签数据
                payload = self._load_gzip_json(label_path)
                # 检查lg标签数据的有效性,如果有效则valid=True,reason=ok,否则为valid=False,reason=无效原因
                valid, reason = self._validate_payload(payload)
            except (
                OSError,
                EOFError,
                ValueError,
                ujson.JSONDecodeError,
            ) as exc:
                reasons[
                    f"read_error:{type(exc).__name__}"
                ] += 1
                continue

            # 如果lg标签数据无效,则统计无效原因并跳过该样本
            if not valid:
                reasons[reason] += 1
                continue

            # 读取并检查六视角注意力监督标签
            # 如果六视角注意力监督标签有效,那么_camera_attention_target是一个长度为6的numpy数组,表示六个相机的注意力权重,并且权重和为1,camera_attention_valid=True
            # 如果六视角注意力监督标签无效,那么_camera_attention_target是一个长度为6的零数组,camera_attention_valid=False
            _camera_attention_target, camera_attention_valid = self._extract_camera_attention_supervision(payload)

            # 如果六视角注意力监督标签有效,则统计有效数量
            if camera_attention_valid:
                camera_attention_valid_count += 1
            # 如果六视角注意力监督标签无效,则统计无效原因
            else:
                visual_grounding = payload.get("visual_grounding",{},)

                # 无效原因
                if isinstance(visual_grounding, dict):
                    invalid_reason = str(visual_grounding.get("invalid_reason","unknown",))
                else:
                    invalid_reason = ("missing_visual_grounding")

                camera_attention_invalid_reasons[invalid_reason] += 1

            # 有效样本的索引
            valid_indices.append(index)
            # 有效样本的路径
            valid_paths.append(str(label_path))

        # 将有效样本的索引转换为numpy数组,并获取原始样本数量
        indices = np.asarray(valid_indices, dtype=np.int64)
        original_sample_count = len(self.images)

        for attribute in (
            "images",
            "surround_images",
            "boxes",
            "measurements",
            "sample_start",
            "augment_exists",
        ):
            values = getattr(self, attribute)
            if len(values) != original_sample_count:
                raise RuntimeError(
                    f"LG index container length mismatch: {attribute} has "
                    f"{len(values)} entries, expected "
                    f"{original_sample_count}"
                )

            filtered_values = self._filter_container(
                values,
                indices,
                valid_indices,
            )
            if attribute == "augment_exists":
                filtered_values = np.asarray(
                    filtered_values,
                    dtype=np.bool_,
                )
            setattr(self, attribute, filtered_values)

        self.lg_label_paths = np.asarray(
            valid_paths,
            dtype=np.string_,
        )

        if bool(
            getattr(
                self,
                "lg_print_filter_summary",
                True,
            )
        ):
            print(
                f"[{self.split} LG samples]: kept "
                f"{len(valid_indices)} samples; "
                f"filtered={dict(reasons)}"
            )

            #修改20260721：输出六视角注意力标签的实际覆盖率。
            kept_sample_count = len(valid_indices)
            camera_attention_valid_ratio = (
                camera_attention_valid_count
                / kept_sample_count
                if kept_sample_count > 0
                else 0.0
            )

            bucket_name = str(
                getattr(
                    self,
                    "bucket_name",
                    "unknown",
                )
            )

            print(
                f"[{self.split} LG camera attention]"
                f"[bucket={bucket_name}]: "
                f"valid={camera_attention_valid_count}, "
                f"invalid="
                f"{kept_sample_count - camera_attention_valid_count}, "
                f"total={kept_sample_count}, "
                f"valid_ratio="
                f"{camera_attention_valid_ratio:.4f} "
                f"({camera_attention_valid_ratio * 100.0:.2f}%), "
                f"invalid_reasons="
                f"{dict(camera_attention_invalid_reasons)}"
            )

    @staticmethod
    def _compute_waypoints_1d(waypoints: np.ndarray,) -> np.ndarray:
        points = np.asarray(waypoints, dtype=np.float32)
        points_with_origin = np.concatenate(
            [
                np.zeros((1, 2), dtype=np.float32),
                points,
            ],
            axis=0,
        )
        segment_lengths = np.linalg.norm(
            np.diff(points_with_origin, axis=0),
            axis=1,
        )
        cumulative = np.cumsum(segment_lengths)
        return np.stack(
            [
                cumulative,
                np.zeros_like(cumulative),
            ],
            axis=1,
        ).astype(np.float32)

    def _build_language_text(self, payload: Dict, prefix: str,) -> Tuple[str, str]:
        use_language = bool(
            getattr(self, "lg_use_language", True)
        )
        mode = str(
            getattr(
                self,
                "lg_language_mode",
                "four_questions",
            )
        ).lower()

        if not use_language or mode == "none":
            return (
                f"{prefix} Predict the waypoints.",
                "Waypoints:",
            )

        questions = self._extract_questions(payload)
        if len(questions) != len(self.lg_question_keys):
            raise ValueError(
                "LG label does not contain the required "
                "four questions"
            )

        if mode == "random_question":
            # selected = (
            #     random.choice(questions)
            #     if self.split == "train"
            #     else questions[0]
            # )
            selected = random.choice(questions)
            prompt = (
                f"{prefix} Q: {selected['question']} "
                "Then predict the waypoints."
            )
            answer = (
                f"A: {selected['answer']} Waypoints:"
            )
            return prompt, answer

        if mode != "four_questions":
            raise ValueError(
                f"Unsupported lg_language_mode={mode!r}; "
                "expected four_questions, random_question, "
                "or none"
            )

        question_text = " ".join(
            f"Q{index}: {item['question']}"
            for index, item in enumerate(
                questions,
                start=1,
            )
        )
        answer_text = " ".join(
            f"A{index}: {item['answer']}"
            for index, item in enumerate(
                questions,
                start=1,
            )
        )

        prompt = (
            f"{prefix} Answer the four driving questions "
            "in order and then predict the waypoints. "
            f"{question_text}"
        )
        answer = f"{answer_text} Waypoints:"
        return prompt, answer

    def __getitem__(self, index):
        cv2.setNumThreads(0)

        data = {}
        surround_images = self.surround_images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        lg_path = Path(self._decode_path(self.lg_label_paths[index]))

        (
            loaded_measurements,
            current_measurement,
            measurement_file_current,
        ) = self.load_current_and_future_measurements(
            measurements,
            sample_start,
        )
        data["measurement_path"] = measurement_file_current

        # LG data does not use geometric camera augmentation.
        aug_rotation = 0.0
        aug_translation = 0.0

        data = self.load_waypoints(
            data,
            loaded_measurements,
            aug_translation,
            aug_rotation,
        )

        data["speed"] = current_measurement["speed"]
        speed_rounded = round(
            current_measurement["speed"],
            1,
        )

        data = self.load_route(
            data,
            current_measurement,
            aug_translation,
            aug_rotation,
        )

        target_point = np.asarray(
            current_measurement["target_point"],
            dtype=np.float32,
        )
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

        target_options, placeholder_values = (
            self.get_navigational_conditioning(
                data,
                current_measurement,
                target_point,
                next_target_point,
            )
        )

        payload = self._load_gzip_json(lg_path)
        valid, reason = self._validate_payload(payload)
        if not valid:
            raise ValueError(
                f"Invalid LG label at {lg_path}: {reason}"
            )

        #修改20260726：读取当前帧的四通道结构化未来世界标签。
        future_interaction_grid = None
        future_interaction_valid = False

        if self.lg_use_future_interaction_grid:
            future_interaction_path = (
                self._future_interaction_grid_path_for_index(index)
            )
            (
                future_interaction_grid,
                future_interaction_valid,
            ) = self._load_future_interaction_grid(
                future_interaction_path
            )

        #修改20260720：无有效因果actor时返回全零目标和False掩码，
        # 但不丢弃该LG语言/轨迹样本。
        (
            camera_attention_target,
            camera_attention_valid,
        ) = self._extract_camera_attention_supervision(
            payload
        )

        if bool(
            getattr(
                self,
                "lg_use_waypoints",
                True,
            )
        ):
            waypoints = self._extract_lg_waypoints(payload)
            path = self._extract_lg_path(payload)
        else:
            waypoints = np.asarray(
                data["waypoints_org"],
                dtype=np.float32,
            )
            path = np.asarray(
                data["route_adjusted_org"],
                dtype=np.float32,
            )

        waypoints_1d = self._compute_waypoints_1d(
            waypoints
        )

        prefix = f"Current speed: {speed_rounded} m/s."
        if (
            bool(
                getattr(
                    self,
                    "lg_include_navigation_conditioning",
                    True,
                )
            )
            and len(target_options) > 0
        ):
            # navigation_text = (
            #     random.choice(target_options)
            #     if self.split == "train"
            #     else target_options[0]
            # )
            # navigation_text = random.choice(target_options)
            # prefix = f"{prefix} {navigation_text}"
            
            # 导航类型由route_as唯一确定，
            # LG训练和验证均不再随机切换导航表达。
            navigation_text = target_options[0]
            prefix = f"{prefix} {navigation_text}"

        prompt, answer = self._build_language_text(
            payload,
            prefix,
        )
        prompt = (
            prompt.replace("..", ".")
            .replace("  ", " ")
            .strip()
        )
        answer = (
            answer.replace("..", ".")
            .replace("  ", " ")
            .strip()
        )

        # Load all six views once. The legacy front fields are taken directly
        # from view index 0, keeping front and surround inputs identical.
        data = self.load_surround_images(
            data,
            surround_images,
        )

        conversation_answer = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ],
            }
        ]
        conversation_all = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    # Keep one image placeholder until datamodule.py is changed
                    # to expand six views into visual tokens.
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ],
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
            dataset="driving",
            image_surround=data["rgb_surround"],
            image_surround_org_size=(
                data["rgb_surround_org_size"]
            ),
            camera_order=data["camera_order"],
            #修改20260720：传递LG六视角注意力软标签。
            camera_attention_target=(
                camera_attention_target
            ),
            camera_attention_valid=(
                camera_attention_valid
            ),

            #修改20260726：传递四通道结构化未来世界标签。
            future_interaction_grid=(
                future_interaction_grid
            ),
            future_interaction_valid=(
                future_interaction_valid
            ),
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
    import hydra

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
        print(sample.image_surround.shape)
        print(sample.camera_order)
        print(sample.conversation)
