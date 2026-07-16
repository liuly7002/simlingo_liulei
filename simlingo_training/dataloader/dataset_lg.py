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
        
        # 检查是否启用了lg监督,如果没有启用LG监督,立即抛出 ValueError,停止创建数据集。
        if not bool(cfg.get("use_lg_supervision", False)):
            raise ValueError(
                "Data_LG was selected but use_lg_supervision is False. "
                "Set data_module.dreamer_dataset.use_lg_supervision=true, "
                "or switch back to the original Data_Dreamer target."
            )



        # 复制配置，这样可以避免对传入的cfg修改
        base_cfg = dict(cfg)



        # Data_LG只负责LG语言和LG轨迹监督，不在这个辅助数据源内部混入原始VQA与Commentary任务
        base_cfg["use_qa"] = False   # 主要是在调用父类初始化的时候传进父类
        base_cfg["use_commentary"] = False  # 主要是在调用父类初始化的时候传进父类

        
        
        # lg_match_dreamer_split=True表示默认与Dreamer保持相同的数据划分，不额外使用Town13
        if bool(base_cfg.get("lg_match_dreamer_split", True)):
            base_cfg["use_town13"] = False



        # 调用父类初始化,dreamer=False告诉父类当前不是原始Dreamer数据集
        super().__init__(dreamer=False, **base_cfg)



        # lg标签文件夹名称:language_grounded_waypoints
        self.lg_label_folder = str(getattr(self, "lg_label_folder", "language_grounded_waypoints"))
        
        
        
        # 四个lg问题的键名字
        self.lg_question_keys = tuple(
            getattr(
                self,
                "lg_question_keys",
                ("attention", "motion_constraint", "driving_response", "future_motion"),
            )
        )

        # 根据lg文件重新过滤样本
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
        
        # 当前样本对应的measurements目录 "/root/.../Town12_xxx/measurements"
        measurement_entry = self.measurements[index]  
        
        # 判断是不是数组
        if isinstance(measurement_entry, np.ndarray):
            measurement_entry = measurement_entry.reshape(-1)[0]
        # 判断是不是list或者tuple
        elif isinstance(measurement_entry, (list, tuple)):
            measurement_entry = measurement_entry[0]

        return Path(self._decode_path(measurement_entry))

    def _current_frame_for_index(self, index: int) -> int:
        return int(self.sample_start[index]) + int(self.hist_len) - 1

    def _lg_path_for_index(self, index: int) -> Path:

        # 当前帧对应的measurements目录: /root/.../Town12_xxx/measurements
        measurement_dir = self._measurement_dir_for_index(index)

        # 取上级目录:/root/.../Town12_xxx/
        route_dir = measurement_dir.parent  # Path.parent 表示取当前路径的上一级目录

        # 当前样本真正对应的帧id
        frame_id = self._current_frame_for_index(index)  

        # 拼接lg文件路径.json.gz: /root/.../Town12_xxx/language_grounded_waypoints/0026.json
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

    def _extract_lg_path(self, payload: Dict) -> np.ndarray:

        reference = payload.get("reference", {})

        # 健壮
        if not isinstance(reference, dict):
            return np.empty((0, 2), dtype=np.float32)

        path = np.asarray(
            reference.get("selected_reference_route", []),
            dtype=np.float32,
        )

        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
            return path

        # 与SimLingo原始path保持一致：
        # 从自车原点开始，按 1m 间隔重采样为20个几何路径点。
        path = self.equal_spacing_route(path)

        return np.asarray(path, dtype=np.float32)

    def _validate_payload(self, payload: Dict) -> Tuple[bool, str]:
        """
        检查机制
        """
        # 获取.json标签文件中的 "supervision" 键
        supervision = payload.get("supervision", {})
        if not isinstance(supervision, dict):
            return False, "missing_supervision"

        # 过滤机制(risk_label_valid是否为true)这是一个重点,因为我们认定标签不合法时不能使用
        if bool(getattr(self, "lg_require_risk_label_valid", True)):
            if not bool(supervision.get("risk_label_valid", False)):
                return False, "risk_label_invalid"

        # 过滤机制(是否属于不允许的fallback)
        if bool(getattr(self, "lg_skip_expert_fallback", True)):
            internal_name = str(supervision.get("selected_internal_intent_name", ""))
            if internal_name in {"expert_fallback", "stationary_hold_fallback"}:
                return False, f"fallback:{internal_name}"

        # 过滤机制(lg waypoints是否为[10,2]  lg path是否为[20,2] path和waypoints是否有NaN或Inf 坐标是否超过100 四个语言问题是否完整)
        if bool(getattr(self, "lg_use_waypoints", True)):
            
            # 规划出来的waypoints
            waypoints = self._extract_lg_waypoints(payload)
            expected_count = int(self.pred_len) - 1  # 10个waypoints

            # 过滤机制(lg waypoints形状是否为10,2)
            if waypoints.shape != (expected_count, 2):
                return False, f"waypoint_shape:{tuple(waypoints.shape)}"
            
            # 过滤机制(lg waypoints是否有NaN或Inf)
            if not np.isfinite(waypoints).all():
                return False, "waypoint_non_finite"

            # 过滤机制 (坐标是否超过100m)
            max_abs = float(getattr(self, "lg_max_abs_waypoint_m", 100.0))
            if np.max(np.abs(waypoints), initial=0.0) > max_abs:
                return False, "waypoint_out_of_range"

            # 选择的path
            path = self._extract_lg_path(payload)

            # 过滤机制(path waypoints是否为[10,2])
            if path.shape != (20, 2):
                return False, f"path_shape:{tuple(path.shape)}"
            
            # 过滤机制(path 是否有NaN或Inf)
            if not np.isfinite(path).all():
                return False, "path_non_finite"
            
            # 过滤机制(坐标是否超过100m)
            if np.max(np.abs(path), initial=0.0) > max_abs:
                return False, "path_out_of_range"

        language_mode = str(getattr(self, "lg_language_mode", "four_questions")).lower()
        require_questions = bool(getattr(self, "lg_require_four_questions", True))
        use_language = bool(getattr(self, "lg_use_language", True))
        if use_language and language_mode != "none" and require_questions:
            questions = self._extract_questions(payload)
            if len(questions) != len(self.lg_question_keys):
                return False, "incomplete_four_questions"

        return True, "ok"

    def _filter_samples_with_valid_lg_labels(self) -> None:
        """
        在 BaseDataset 已经建立的全部样本索引中，
        只保留“存在对应LG文件,
        并且LG文件通过合法性检查”的样本,
        同时保证图像、measurement、帧号等索引仍然严格一一对应
        """


        valid_indices: List[int] = []  # 保存通过检查的原始样本索引
        valid_paths: List[str] = []    # 保存每个有效样本对应的lg标签文件路径
        reasons = Counter()            # 计数器,用于统计各类样本被过滤掉的原因

        # 遍历父类建立的全部基础样本
        for index in range(len(self.images)):

            # 根据当前索引,生成该样本对应的lg的.json.gz路径
            # label_path=/root/.../Town12_xxx/language_grounded_waypoints/0026.json
            label_path = self._lg_path_for_index(index)


            # 检查标签文件是否存在
            if not label_path.is_file():
                reasons["missing_label"] += 1
                continue

            try:
                payload = self._load_gzip_json(label_path)  # 解析,payload是完整的lg标签
                valid, reason = self._validate_payload(payload)  # 这里有一个重点,就是不合法的标签我们不使用
            except (OSError, EOFError, ValueError, ujson.JSONDecodeError) as exc:
                reasons[f"read_error:{type(exc).__name__}"] += 1
                continue

            if not valid:
                reasons[reason] += 1
                continue

            # 记录合法样本 保存当前基础样本的原始索引
            valid_indices.append(index)
            # 记录合法样本 保存当前lg标签文件路径
            valid_paths.append(str(label_path))

        indices = np.asarray(valid_indices, dtype=np.int64)  # 转换为numpy数组
        original_sample_count = len(self.images)  # 过滤前的样本总数


        # 依次过滤五个基础索引容器
        for attribute in (
            "images",
            "boxes",
            "measurements",
            "sample_start",
            "augment_exists",
        ):

            # 动态取得当前属性容器
            values = getattr(self, attribute)

            # 检查所有容器的长度是否一致
            if len(values) != original_sample_count:
                raise RuntimeError(
                    f"LG index container length mismatch: {attribute} has "
                    f"{len(values)} entries, expected {original_sample_count}"
                )


            # 取有效样本
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

            # 根据属性名将过滤后的容器覆盖回当前数据集对象
            setattr(self, attribute, filtered_values)

        # 保存过滤后的lg文件路径
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

    def _build_language_text(self, payload: Dict, prefix: str,) -> Tuple[str, str]:
        """
        根据LG语言配置，从当前LG标签中读取四个问题和答案，并组合成最终送给模型的 prompt 和监督模型输出的 answer
        """

        # 是否使用lg语言
        use_language = bool(getattr(self, "lg_use_language", True))
        
        
        # 语言组织模式
        mode = str(getattr(self, "lg_language_mode", "four_questions")).lower()

        
        # 不使用语言的情况
        if not use_language or mode == "none":
            return f"{prefix} Predict the waypoints.", "Waypoints:"

        
        # 提取4个问题和4个答案
        questions = self._extract_questions(payload)
        if len(questions) != len(self.lg_question_keys):
            raise ValueError("LG label does not contain the required four questions")
        # questions = [
        #     {
        #         "key": "attention",
        #         "question": "What is the main factor to focus on now?",
        #         "answer": "A pedestrian ahead is the main factor to focus on."
        #     },
        #     {
        #         "key": "motion_constraint",
        #         "question": "What constrains the motion ahead?",
        #         "answer": "The pedestrian constrains the forward space."
        #     },
        #     {
        #         "key": "driving_response",
        #         "question": "What motion should be planned next?",
        #         "answer": "Slow down early and wait if needed."
        #     },
        #     {
        #         "key": "future_motion",
        #         "question": "...",
        #         "answer": "..."
        #     }
        # ]



        # 如果配置是lg_language_mode: random_question,那么每个训练样本只使用四问中的一个问题，而不是全部四个问题
        if mode == "random_question":
            # Training keeps random question sampling. Validation always uses
            # the first configured question so that the same sample receives
            # exactly the same language input at every validation epoch.
            selected = (
                random.choice(questions)
                if self.split == "train"  # 训练阶段随机选择问题
                else questions[0]  # 验证阶段不能随机选择，否则同一个模型在不同epoch验证时，输入问题可能不同，使验证损失和结果产生随机波动
            )

            # 构造单问题的prompt和answer
            prompt = f"{prefix} Q: {selected['question']} Then predict the waypoints."
            answer = f"A: {selected['answer']} Waypoints:"
            # Current speed: 5.3 m/s. Command: follow the road.
            # Q: What constrains the motion ahead?
            # Then predict the waypoints.

            # A: The pedestrian ahead constrains the forward motion. Waypoints:

            return prompt, answer

        if mode != "four_questions":
            raise ValueError(
                f"Unsupported lg_language_mode={mode!r}; expected four_questions, "
                "random_question, or none"
            )


        # 构造4问题文本
        # 步骤一：遍历四个问题，并从1开始编号
        question_text = " ".join(
            f"Q{idx}: {item['question']}" for idx, item in enumerate(questions, start=1)
        )
        # question_text = 
        # Q1: What is the main factor to focus on now? Q2: What constrains the motion ahead? Q3: What motion should be planned next? Q4: What will the future motion look like?


        # 构造4答案文本
        # 步骤二：遍历四个答案,并从1开始编号
        answer_text = " ".join(
            f"A{idx}: {item['answer']}" for idx, item in enumerate(questions, start=1)
        )
        # answer_text = A1: A pedestrian ahead is the main factor. A2: The forward space is constrained by the pedestrian. A3: Slow down early and wait if needed. A4: The trajectory should remain centered and decelerate toward a stop.


        
        prompt = (
            f"{prefix} Answer the four driving questions in order and then "
            f"predict the waypoints. {question_text}"
        )
        """
        prompt = 

        Current speed: 5.3 m/s.
        Command: follow the road.
        Answer the four driving questions in order and then predict the waypoints.
        Q1: What is the main factor to focus on now?
        Q2: What constrains the motion ahead?
        Q3: What motion should be planned next?
        Q4: What will the future motion look like?

        这里明确要求是按顺序,也就是必须按照Q1、Q2、Q3、Q4顺序回答,避免模型随意打乱答案
        """

        answer = f"{answer_text} Waypoints:"
        """
        answer = 
        A1: A pedestrian ahead is the main factor.
        A2: The pedestrian constrains the forward space.
        A3: Slow down early and wait if needed.
        A4: The future trajectory decelerates toward a stop.
        Waypoints:

        这形成了一个固定的输出顺序:
            语言推理答案
            → Waypoints标记
            → 轨迹token监督
        这也是LG语言与动作监督建立联系的主要位置。
        """



        return prompt, answer




















    def __getitem__(self, index):
        cv2.setNumThreads(0)  # 禁用OpenCV多线程





        ########################################### 🥭 初始化(父类初始化得到) 🥭 ###########################################
        data = {}   # 字典
        images = self.images[index]              # 当前帧的图像路径.jpg
        measurements = self.measurements[index]  # measurements路径
        sample_start = self.sample_start[index]  # 当前样本的帧id
        lg_path = Path(self._decode_path(self.lg_label_paths[index]))  # lg标签路径
        # images: [b'/root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/rgb/0026.jpg'], 
        # measurements: [b'/root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/measurements'], 
        # sample_start: 26, 





        ########################################### 🥭 measurements 🥭 ###########################################

        loaded_measurements, current_measurement, measurement_file_current = (self.load_current_and_future_measurements(measurements, sample_start))
        data["measurement_path"] = measurement_file_current







        ########################################### 🥭 是否进行数据增强 🥭 ###########################################

        # LG labels are generated in the original ego frame. Therefore geometric
        # image-shift augmentation is deliberately disabled for LG samples.
        aug_rotation = 0.0
        aug_translation = 0.0
        augment_sample = False







        ########################################### 🥭 waypoints 🥭 ###########################################

        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation,)
        # data['waypoints']            : 自车坐标系下自车未来10帧(不包括当前帧)自车的位置 [x,y](无增强)
        # data['waypoints_org']        : 自车坐标系下自车未来10帧(不包括当前帧)自车的位置 [x,y]（无增强）
        # data['waypoints_1d']         : 自车坐标系下 10 帧距离 [x,0] (x是自车当前帧与第1、2、、、11帧之间的欧式距离)(没有增强)
        # data['ego_waypoints']        : 11 个 4×4 矩阵（无增强）包括当前帧及未来 10 帧
        # data['ego_waypoints_org']    : 11 个 4×4 矩阵（无增强）包括当前帧及未来 10 帧







        ########################################### 🥭 当前帧的车速 🥭 ###########################################

        data["speed"] = current_measurement["speed"]
        speed_rounded = round(current_measurement["speed"], 1)









        ########################################### 🥭 route 🥭###########################################

        data = self.load_route(data, current_measurement, aug_translation, aug_rotation,)








        ########################################### 🥭 target point 🥭 ###########################################

        target_point = np.asarray(current_measurement["target_point"], dtype=np.float32)
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation,)
        
        
        
        ########################################### 🥭 next target point 🥭 ###########################################

        next_target_point = np.asarray(current_measurement["target_point_next"], dtype=np.float32,)
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation,)







        ########################################### 🥭 target_options, placeholder_values 🥭 ###########################################

        target_options, placeholder_values = self.get_navigational_conditioning(data, current_measurement, target_point, next_target_point,)
        """
        target_options = 
        [
        "Target waypoint: <TARGET_POINT><TARGET_POINT>.",    # 这就是网络框架中的输入 TP
        "Command: {command} in {dist_to_command} meter{next_command}.",  # 这就是网络框架中的输入 HLC
        "Command: {lmdrive_command}."
        ]   # lmdrive_command 来自"/data/augmented_templates/lmdrive.json"语言增强模板文件
        
        placeholder_values = 
        {
        '<TARGET_POINT>': [[x_0, y_0], [x_1, y_1]]
        }
        """






        ########################################### 🥭 生成 waypoints & waypoints_1d & path 🥭 ###########################################

        # 加载.json.gz文件
        payload = self._load_gzip_json(lg_path)  # 解析,payload是当前帧的lg标签
        # 合法性检查
        valid, reason = self._validate_payload(payload)
        if not valid:
            raise ValueError(f"Invalid LG label at {lg_path}: {reason}")

        if bool(getattr(self, "lg_use_waypoints", True)):
            # LG时间轨迹与LG选中候选的几何路径配套使用。
            waypoints = self._extract_lg_waypoints(payload)
            path = self._extract_lg_path(payload)
        else:
            # 不使用LG轨迹时，waypoints和path同时回退到专家监督。
            waypoints = np.asarray(data["waypoints_org"], dtype=np.float32)
            path = np.asarray(data["route_adjusted_org"], dtype=np.float32)

        waypoints_1d = self._compute_waypoints_1d(waypoints)






        ########################################### 🥭 生成 prompt & answer 🥭 ###########################################

        prefix = f"Current speed: {speed_rounded} m/s."
        if (
            bool(getattr(self, "lg_include_navigation_conditioning", True))
            and len(target_options) > 0
        ):
            # Preserve SimLingo's random navigation wording during training,
            # but keep validation prompts deterministic across epochs.
            navigation_text = (
                random.choice(target_options)
                if self.split == "train"  # 训练期间随机选择
                else target_options[0]    # 验证期间选择固定的导航提示内容
            )
            prefix = f"{prefix} {navigation_text}"

        prompt, answer = self._build_language_text(payload, prefix)
        
        prompt = prompt.replace("..", ".").replace("  ", " ").strip()
        answer = answer.replace("..", ".").replace("  ", " ").strip()






        ############################################# 🥭 前视图像 🥭 #############################################

        data = self.load_images(data, images, augment_sample=augment_sample)






        ############################################# 🥭 构造对话格式 🥭 #############################################

        # 1. 只包含答案的版本  这是仅包含 assistant 输出的部分，通常用于监督目标
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






        # 最终返回结果
        data_new = DatasetOutput(
            conversation=conversation_all,          # 完整 user-assistant 对话，给模型做输入格式组织用。
            answer=conversation_answer,             # 只包含监督答案，通常给 loss 计算用。
            image_ff=data["rgb"],                   # front-forward image，当前样本前视图。
            image_ff_org_size=data["rgb_org_size"], # 原图尺寸。后处理或可视化时可能要用。
            waypoints=waypoints,
            waypoints_1d=waypoints_1d,
            path=path,
            target_points=data["target_points"],    # 导航目标点
            speed=data["speed"],                    # 当前速度
            placeholder_values=placeholder_values,  # 导航模板相关占位值
            measurement_path=data["measurement_path"],  # 当前样本来源，方便 debug
            # Keep the original dataset identifier so downstream SimLingo code
            # follows exactly the same path as Dreamer auxiliary samples.
            dataset="driving",
        )

        if VIZ_DATA:
            self.visualise_cameras(data_new, None, path, waypoints, options=None, name="lg_", prompt=prompt, answer=answer,)

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
