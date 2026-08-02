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

from simlingo_training.dataloader.dataset_base_surround import SurroundBaseDataset
from simlingo_training.dataloader.participant_spatial_attention import CAMERA_ORDER,TOKENS_PER_CAMERA,extract_participant_spatial_attention_supervision
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

    def _future_interaction_grid_path_for_index(self, index: int,) -> Path:
        """
        返回当前样本结构化世界标签路径:
        ....../Town04_Rep0_Town04_lr_0_route0_07_25_20_57_48/future_interaction_grids/0026.json.gz
        """
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


    @staticmethod
    def _load_future_interaction_grid(path: Path,) -> Tuple[Optional[np.ndarray], bool]:
        
        # 安全性检查,如果文件损坏
        if not path.is_file():
            return None, False

        # 加载结构化世界标签文件
        with np.load(str(path), allow_pickle=False) as payload:
            
            # 安全性检查
            if "future_interaction_grid" not in payload.files:
                raise ValueError(
                    f"Missing future_interaction_grid in {path}"
                )

            # 加载future_interaction_grid内容
            grid = np.asarray(payload["future_interaction_grid"],dtype=np.float32,)

            # 有效性检查
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

        # 安全性检查
        if grid.ndim != 3 or grid.shape[0] != 4:
            raise ValueError(
                "future_interaction_grid must have shape "
                f"[4, H, W], but received {tuple(grid.shape)} "
                f"from {path}"
            )

        # 安全性检查
        if not np.isfinite(grid).all():
            raise ValueError(
                f"future_interaction_grid contains non-finite values: {path}"
            )

        # 安全性检查
        if np.any(grid < 0.0) or np.any(grid > 1.0):
            raise ValueError(
                "future_interaction_grid values must be within "
                f"[0, 1]: {path}"
            )

        # 返回结构化世界标签中的"future_interaction_grid"内容
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

    def _extract_participant_spatial_attention_supervision(self,payload: Dict,) -> Tuple[np.ndarray, bool]:
        """
        将LG标签中的主要关键参与者六视角投影信息转换为
        与视觉token严格对齐的[6, 64]空间注意力监督。

        DatasetOutput中继续使用camera_attention_target和
        camera_attention_valid字段名，只是为了兼容现有批处理接口。
        """

        # 主要关键参与者六视角空间投影并生成6×64视觉token监督
        if not bool(getattr(self,"lg_use_participant_spatial_attention_supervision",False,)):
            return (np.zeros((len(CAMERA_ORDER),TOKENS_PER_CAMERA,),dtype=np.float32,),False,)

        # 返回的是[6,64]的软监督权重+True/False
        return extract_participant_spatial_attention_supervision(
            payload,
            cut_bottom_quarter=bool(
                self.cut_bottom_quarter  # 裁剪图像下方四分之一区域(图像的这部分是车头引擎盖)
                or self.img_shift_augmentation  # 进行图像增强
            ),
            use_global_img=bool(self.use_global_img),
        )

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

        # 统计真正进入当前LG数据集的样本中，
        # 有效参与者空间注意力监督的数量及无效原因。
        participant_spatial_valid_count = 0
        participant_spatial_invalid_reasons = Counter()

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
            # 如果六视角注意力监督标签有效,那么_participant_spatial_target是一个[6,64]的数组,表示六个相机的注意力权重,并且权重和为1,participant_spatial_valid=True
            # 如果六视角注意力监督标签无效,那么_participant_spatial_target是一个[6,64]的零数组,表示六个相机的注意力权重都为0(此时主要是没有主要actor),participant_spatial_valid=False
            _participant_spatial_target,participant_spatial_valid, = self._extract_participant_spatial_attention_supervision(payload)

            # 有效 -> 有效标签数量+1
            if participant_spatial_valid:
                participant_spatial_valid_count += 1
            # 无效 -> 无效标签数量+1并注明原因
            else:
                visual_grounding = payload.get("visual_grounding",{},)

                if isinstance(visual_grounding, dict):
                    invalid_reason = str(visual_grounding.get("invalid_reason", None,) or "unknown")
                else:
                    invalid_reason = ("missing_visual_grounding")

                participant_spatial_invalid_reasons[invalid_reason] += 1

            # 有效样本的索引
            valid_indices.append(index)
            # 有效样本的路径
            valid_paths.append(str(label_path))

        # 将有效样本的索引转换为numpy数组,并获取原始样本数量
        indices = np.asarray(valid_indices, dtype=np.int64)
        original_sample_count = len(self.images)

        for attribute in ("images","surround_images","boxes","measurements","sample_start","augment_exists",):

            values = getattr(self, attribute)

            # 安全性检查
            if len(values) != original_sample_count:
                raise RuntimeError(
                    f"LG index container length mismatch: {attribute} has "
                    f"{len(values)} entries, expected "
                    f"{original_sample_count}"
                )

            filtered_values = self._filter_container(values,indices,valid_indices,)

            if attribute == "augment_exists":
                filtered_values = np.asarray(filtered_values,dtype=np.bool_,)
            
            setattr(self, attribute, filtered_values)

        # 有效样本的路径
        self.lg_label_paths = np.asarray(valid_paths,dtype=np.string_,)

        # 打印debug信息
        if bool(getattr(self,"lg_print_filter_summary",True,)):

            print(
                f"[{self.split} LG samples]: kept "
                f"{len(valid_indices)} samples; "  # 有效样本的数量
                f"filtered={dict(reasons)}"        # 无效的原因
            )

            # 输出存在主要参与者空间注意力标签的实际覆盖率。
            kept_sample_count = len(valid_indices)
            participant_spatial_valid_ratio = (participant_spatial_valid_count / kept_sample_count if kept_sample_count > 0 else 0.0)

            # bucket 名
            bucket_name = str(getattr(self,"bucket_name","unknown",))

            # 打印内容
            print(
                f"[{self.split} LG participant spatial attention]"
                f"[bucket={bucket_name}]: "
                f"valid={participant_spatial_valid_count}, "
                f"invalid="
                f"{kept_sample_count - participant_spatial_valid_count}, "
                f"total={kept_sample_count}, "
                f"valid_ratio="
                f"{participant_spatial_valid_ratio:.4f} "
                f"({participant_spatial_valid_ratio * 100.0:.2f}%), "
                f"invalid_reasons="
                f"{dict(participant_spatial_invalid_reasons)}"
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
        
        # 是否使用 lg 语言
        use_language = bool(
            getattr(self, "lg_use_language", True)
        )

        # 获取配置中的lg语言模式,默认四问题模式不变
        mode = str(getattr(self, "lg_language_mode", "four_questions",)).lower()

        # 1. 不使用lg语言或者lg语言模式为none,这是一个保底问题
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

        # 随机模式
        if mode == "random_question":
            selected = random.choice(questions)
            prompt = (
                f"{prefix} Q: {selected['question']} "
                "Then predict the waypoints."
            )
            answer = (
                f"A: {selected['answer']} Waypoints:"
            )
            return prompt, answer

        # 异常
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

        cv2.setNumThreads(0)  # 禁止opencv多线程




        data = {}




        ########################################### 🥭 初始化(父类的父类初始化得到) 🥭 ###########################################

        surround_images = self.surround_images[index]  # 六视角图像路径.../rgb_front/0026.jpg  .../rgb_front_left/0026.jpg  .../rgb_front_right/0026.jpg  .../rgb_rear/0026.jpg  .../rgb_rear_left/0026.jpg  .../rgb_rear_right/0026.jpg
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        lg_path = Path(self._decode_path(self.lg_label_paths[index]))  # lg 标签文件路径
        
        # images: [b'/root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/rgb/0026.jpg'], 
        # measurements: [b'/root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/measurements'], 
        # sample_start: 26, 
        # lg_path: ....../Town12_Rep0_493_route0_02_28_11_00_43/language_grounded_waypoints/0026.json.gz




        ########################################### 🥭 measurements 🥭 ###########################################

        loaded_measurements,current_measurement,measurement_file_current = self.load_current_and_future_measurements(measurements,sample_start,)
        data["measurement_path"] = measurement_file_current
        
        # loaded_measurements: 当前帧及未来11帧的 .json 内容 总共12帧
        # current_measurement: 当前帧的 .json.gz 内容
        # measurement_file_current: 当前 measurement .json.gz 文件路径

        # current_measurement[0] = {'pos_global': [-1932.94677734375, 6059.3369140625], 'theta': 2.716931982836451, 'speed': 10.893928527832031, 'target_speed': 10.0, 'speed_limit': 13.88888888888889, 'target_point': [166.21953678063582, -1.4034610299303338], 'target_point_next': [292.23548401248064, -4.342632889987669], 'command': 4, 'next_command': 4, 'aim_wp': [3.9551499024618035, 0.0007044955744153203], 'route': [[2.4544338729054394, 0.001334758810199066], [3.45489382915608, 0.001178228841105744], [4.45530941054533, 0.00044431978983006104], [5.455462377914564, 0.00014897731342955467], [6.455837259630751, -0.0007319459886239166], [7.456450110816944, -0.001226608638614568], [8.456802089316398, -0.002246499088830234], [9.45680703690629, -0.003423308025632288], [10.457292488832373, -0.0043828452118344075], [11.457613323532335, -0.005416818090294484], [12.457733948091944, -0.006541320864964284], [13.45801853223297, -0.007720296218779232], [14.458442006321436, -0.00937234785163632], [15.45912646937736, -0.011056433010823596], [16.4552730284497, -0.012434575571900197], [17.453081922137564, -0.014250703843281087], [18.45291606253329, -0.015772686794809587], [19.45341345977995, -0.017798580123078445], [20.453905537485518, -0.01996084850029245], [21.454136713433837, -0.02250902934549437], [22.454383848005413, -0.02464808504385907], [23.454714403468685, -0.02674941995219271], [24.455156186787544, -0.028800460473815903], [25.45528145442136, -0.031262560656781346], [26.45560789960201, -0.034169572262111814], [27.455783461622538, -0.036742900300669845], [28.45644479962843, -0.040168330393921536], [29.45702150748416, -0.042962179629153496], [30.457035547262382, -0.04547457419882939], [31.457073283721886, -0.04878007186882449], [32.457305668928676, -0.05199755436207809], [33.457593668063254, -0.055189889661976466], [34.45806035732186, -0.05870333493197144], [35.45899565468412, -0.06280870575544562], [36.45900558417972, -0.06612677702211833], [37.45917171637654, -0.06964215680661745], [38.45955619400193, -0.07386262451469605], [39.45984008285389, -0.07786063651159125], [40.45992279415901, -0.0814137370861232], [41.4604409870628, -0.085707711859758]], 'route_original': [[2.4544338729054394, 0.001334758810199066], [3.45489382915608, 0.001178228841105744], [4.45530941054533, 0.00044431978983006104], [5.455462377914564, 0.00014897731342955467], [6.455837259630751, -0.0007319459886239166], [7.456450110816944, -0.001226608638614568], [8.456802089316398, -0.002246499088830234], [9.45680703690629, -0.003423308025632288], [10.457292488832373, -0.0043828452118344075], [11.457613323532335, -0.005416818090294484], [12.457733948091944, -0.006541320864964284], [13.45801853223297, -0.007720296218779232], [14.458442006321436, -0.00937234785163632], [15.45912646937736, -0.011056433010823596], [16.4552730284497, -0.012434575571900197], [17.453081922137564, -0.014250703843281087], [18.45291606253329, -0.015772686794809587], [19.45341345977995, -0.017798580123078445], [20.453905537485518, -0.01996084850029245], [21.454136713433837, -0.02250902934549437], [22.454383848005413, -0.02464808504385907], [23.454714403468685, -0.02674941995219271], [24.455156186787544, -0.028800460473815903], [25.45528145442136, -0.031262560656781346], [26.45560789960201, -0.034169572262111814], [27.455783461622538, -0.036742900300669845], [28.45644479962843, -0.040168330393921536], [29.45702150748416, -0.042962179629153496], [30.457035547262382, -0.04547457419882939], [31.457073283721886, -0.04878007186882449], [32.457305668928676, -0.05199755436207809], [33.457593668063254, -0.055189889661976466], [34.45806035732186, -0.05870333493197144], [35.45899565468412, -0.06280870575544562], [36.45900558417972, -0.06612677702211833], [37.45917171637654, -0.06964215680661745], [38.45955619400193, -0.07386262451469605], [39.45984008285389, -0.07786063651159125], [40.45992279415901, -0.0814137370861232], [41.4604409870628, -0.085707711859758]], 'changed_route': False, 'speed_reduced_by_obj_type': None, 'speed_reduced_by_obj_id': None, 'speed_reduced_by_obj_distance': None, 'steer': 0.0, 'throttle': 0.0, 'brake': False, 'control_brake': True, 'junction': False, 'vehicle_hazard': False, 'vehicle_affecting_id': None, 'light_hazard': False, 'walker_hazard': False, 'walker_affecting_id': None, 'stop_sign_hazard': False, 'stop_sign_close': False, 'walker_close': False, 'walker_close_id': None, 'angle': 0.00011339540056267717, 'augmentation_translation': 0.36125444482272595, 'augmentation_rotation': 5.24434331572315, 'ego_matrix': [[-0.9111607074737549, -0.4120118319988251, 0.005693943705409765, -1932.94677734375], [0.4120037257671356, -0.9111785292625427, -0.002586618298664689, 6059.3369140625], [0.0062539163045585155, -1.0898917935264762e-05, 0.9999804496765137, 377.0238952636719], [0.0, 0.0, 0.0, 1.0]]}
        # measurement_file_current: /root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_532_route0_02_28_11_05_28/measurements/0010.json.gz




        ########################################### 🥭 是否进行几何增强 🥭 ###########################################

        # lg 数据不使用几何增强,故将旋转、平移设置为0
        aug_rotation = 0.0
        aug_translation = 0.0




        ########################################### 🥭 waypoints(来自专家) 🥭 ###########################################

        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation,)

        # data['waypoints']            : 自车坐标系下自车未来10帧(不包括当前帧)自车的位置 [x,y](无增强)
        # dsta['waypoints_org']        : 自车坐标系下自车未来10帧(不包括当前帧)自车的位置 [x,y]（无增强）
        # dsta['waypoints_1d']         : 自车坐标系下 10 帧距离 [x,0] (x是自车当前帧与第1、2、、、11帧之间的欧式距离)(无增强)
        # dsta['ego_waypoints']        : 11 个 4×4 矩阵（无增强）包括当前帧及未来 10 帧
        # dsta['ego_waypoints_org']    : 11 个 4×4 矩阵（无增强）包括当前帧及未来 10 帧

        


        ########################################### 🥭 当前帧的车速 🥭 ###########################################

        # 速度
        speed_rounded = round(current_measurement["speed"],1,)  # 用于 prompt 文本里显示 小数后一位
        data["speed"] = current_measurement["speed"]            # 用于模型输入或监督保留原始数值




        ########################################### 🥭 route 🥭###########################################

        data = self.load_route(data,current_measurement,aug_translation,aug_rotation,)




        ########################################### 🥭 target point 🥭 ###########################################

        target_point = np.asarray(current_measurement["target_point"],dtype=np.float32,)
        target_point = self.augment_target_point(target_point,y_augmentation=aug_translation,yaw_augmentation=aug_rotation,)
        
        # "target_point": [19.075672365994425,-13.26871181961684]





        ########################################### 🥭 next target point 🥭 ###########################################
        next_target_point = np.asarray(current_measurement["target_point_next"],dtype=np.float32,)
        next_target_point = self.augment_target_point(next_target_point,y_augmentation=aug_translation,yaw_augmentation=aug_rotation,)
        # "next_target_point": [19.703241532357737,-43.26213909836339]





        ########################################### 🥭 target_options, placeholder_values 🥭 ###########################################
        target_options, placeholder_values = (self.get_navigational_conditioning(data,current_measurement,target_point,next_target_point,))

        """
        target_options = 
        [
        "Target waypoint: <TARGET_POINT><TARGET_POINT>.",                # 这就是网络框架中的输入 TP
        "Command: {command} in {dist_to_command} meter{next_command}.",  # 这就是网络框架中的输入 HLC
        "Command: {lmdrive_command}."                                    # lmdrive_command 来自"/data/augmented_templates/lmdrive.json"语言增强模板文件
        ]   
        
        placeholder_values = { '<TARGET_POINT>': [[x_0, y_0], [x_1, y_1]] }
        """




        
        ########################################### 🥭 lg 标签文件 🥭 ###########################################

        payload = self._load_gzip_json(lg_path)


        # 检查 lg 标签数据的有效性,返回是否有效以及无效原因
        valid, reason = self._validate_payload(payload)
        
        # 如果标签无效
        if not valid:
            raise ValueError(
                f"Invalid LG label at {lg_path}: {reason}"
            )




        ########################################### 🥭 当前帧的四通道结构化未来世界标签 🥭 ###########################################
        future_interaction_grid = None
        future_interaction_valid = False

        # 如果使用未来结构化世界标签
        if self.lg_use_future_interaction_grid:
            
            # 当前样本的结构化世界标签文件路径
            # ....../Town04_Rep0_Town04_lr_0_route0_07_25_20_57_48/future_interaction_grids/0026.json.gz
            future_interaction_path = self._future_interaction_grid_path_for_index(index)
            
            # future_interaction_grid 为4通道结构化世界标签内容
            # future_interaction_valid 为是都有效
            future_interaction_grid, future_interaction_valid = self._load_future_interaction_grid(future_interaction_path)

        # 读取并检查六视角注意力监督标签
        # 如果六视角注意力监督标签有效,那么_participant_spatial_target是一个[6,64]的数组,表示六个相机的注意力权重,并且权重和为1,participant_spatial_valid=True
        # 如果六视角注意力监督标签无效,那么_participant_spatial_target是一个[6,64]的零数组,表示六个相机的注意力权重都为0(此时主要是没有主要actor),participant_spatial_valid=False
        participant_spatial_target, participant_spatial_valid = self._extract_participant_spatial_attention_supervision(payload)

        
        

        ########################################### 🥭 waypoints && path && waypoints_1d 🥭 ###########################################

        # 如果使用lg自己生成的自车未来的waypoints的话
        if bool(getattr(self, "lg_use_waypoints", True,)):
            
            # 自车未来的 waypoints (这是lg生成的)
            waypoints = self._extract_lg_waypoints(payload)
            
            # 参考路径 (这是lg生成的)
            path = self._extract_lg_path(payload)
        
        # 如果不使用lg自己生成的自车未来的waypoints的话
        else:
            
            # 自车未来的 waypoints (这是专家的,无几何增强)
            waypoints = np.asarray(data["waypoints_org"], dtype=np.float32,)
            
            # 参考路径 (这是专家的,无几何增强)
            path = np.asarray(data["route_adjusted_org"],dtype=np.float32,)

        # 1d 形式的自车未来的 waypoints
        waypoints_1d = self._compute_waypoints_1d(waypoints)

        


        ########################################### 🥭 生成 prompt & answer 🥭 ###########################################
        
        # 速度提示
        prefix = f"Current speed: {speed_rounded} m/s."

        # 这里的意思是：如果"lg_include_navigation_conditioning = True",那么提示词中的导航信息采用两个导航点的形式
        if (bool(getattr(self,"lg_include_navigation_conditioning",True,)) and len(target_options) > 0):

            # 导航提示
            navigation_text = target_options[0]   # navigation_text = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
            
            # 更新
            prefix = f"{prefix} {navigation_text}"


        # 构建 prompt 和 answer
        prompt, answer = self._build_language_text(payload, prefix,)

        # 整理 prompt 和 answer
        prompt = (prompt.replace("..", ".").replace("  ", " ").strip())
        answer = (answer.replace("..", ".").replace("  ", " ").strip())





        ############################################# 🥭 六视角图像 🥭 #############################################

        # Load all six views once. The legacy front fields are taken directly
        # from view index 0, keeping front and surround inputs identical.

        data = self.load_surround_images(data, surround_images,)

        # data['rgb'] = 增强(高斯模糊、高斯噪声等)的并且裁减了原图(images)底部包含自车引擎盖部分的 新图像  [T,C,H,W]
        # data['rgb_org_size'] = 增强(高斯模糊、高斯噪声等)的但是并未进行裁减的原图(images)  [T,C,H,W]

        # data['rgb_surround'] = 增强(高斯模糊、高斯噪声等)的并且裁减了原图(images)底部包含自车引擎盖部分的 新六视角图像  [T,V,C,H,W]
        # data['rgb_org_size'] = 增强(高斯模糊、高斯噪声等)的但是并未进行裁减的六视角图像原图(images)  [T,V,C,H,W]
        # data['camera_order'] = 六视角图像的顺序,内容为["front","front_left","front_right","rear","rear_left","rear_right"]


        ############################################# 🥭 构造对话格式 🥭 #############################################

        # 1. 只包含答案的版本  这是仅包含 assistant 输出的部分，通常用于监督目标
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

        # 2. 完整对话版本
        # 这是标准的多模态对话格式：user 发出文字 prompt，并附一张图片  assistant 输出文字答案
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




        ############################################# 🥭 最终返回结果 🥭 #############################################
        data_new = DatasetOutput(
            conversation=conversation_all,                           # 完整多模态对话列表，包含user提示与assistant监督答案
            answer=conversation_answer,                              # 仅包含assistant监督答案的对话列表

            image_ff=data["rgb"],                                    # 裁剪底部后的前视图像数组[T,C,H,W]；训练阶段且开启时应用光度增强
            image_ff_org_size=data["rgb_org_size"],                  # 保留原始尺寸、未裁剪的前视图像数组[T,C,H,W]；与image_ff使用相同光度增强结果

            waypoints=waypoints,                                     # 最终用于监督的未来轨迹[F,2]；默认来自LG重规划轨迹，关闭lg_use_waypoints时使用专家轨迹
            waypoints_1d=waypoints_1d,                               # 累计轨迹长度表示[F,2]，每个点为[从原点累计行驶距离,0]
            path=path,                                               # 等距采样后的参考路径[N,2]；默认来自LG标签，关闭lg_use_waypoints时使用专家参考路径

            target_points=data["target_points"],                     # 当前与下一导航点组成的[[x0,y0],[x1,y1]]，位于自车坐标系且无几何增强
            speed=data["speed"],                                     # 当前帧自车速度，单位m/s
            placeholder_values=placeholder_values,                   # 语言占位符取值：{"<TARGET_POINT>":[[x0,y0],[x1,y1]]}
            measurement_path=data["measurement_path"],               # 当前帧measurement .json.gz文件路径

            dataset="driving",

            image_surround=data["rgb_surround"],                     # 裁剪底部后的六视角RGB图像数组[T,V,C,H,W]；训练阶段且开启时应用同步光度增强
            image_surround_org_size=data["rgb_surround_org_size"],   # 保留原始尺寸、未裁剪的六视角RGB图像数组[T,V,C,H,W]
            camera_order=data["camera_order"],                       # 六视角顺序：(front,front_left,front_right,rear,rear_left,rear_right)

            camera_attention_target=participant_spatial_target,      # 主要关键参与者的六视角视觉token空间软目标[6,64]；有效时全局权重和为1
            camera_attention_valid=participant_spatial_valid,        # 当前样本是否存在可用的主要关键参与者空间监督

            future_interaction_grid=future_interaction_grid,         # 四通道结构化未来世界监督[4,H,W]；未启用时为None
            future_interaction_valid=future_interaction_valid,       # 当前结构化未来世界监督是否有效
        )

        # 是否可视化
        if VIZ_DATA:
            self.visualise_cameras(data_new, None, path, waypoints, options=None, name="lg_", prompt=prompt, answer=answer,)

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
        print(
            "participant spatial target shape:",
            sample.camera_attention_target.shape,
        )
        print(
            "participant spatial valid:",
            sample.camera_attention_valid,
        )
