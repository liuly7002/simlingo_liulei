# -*- coding: utf-8 -*-


from collections import Counter
from pathlib import Path

import numpy as np

from simlingo_training.dataloader.dataset_base_surround import SurroundDatasetMixin
from simlingo_training.dataloader.dataset_driving import Data_Driving
from simlingo_training.dataloader.dataset_lg import Data_LG
from simlingo_training.dataloader.participant_spatial_attention import CAMERA_ORDER, TOKENS_PER_CAMERA, extract_participant_spatial_attention_supervision


class Data_Driving_Surround(SurroundDatasetMixin,Data_Driving,):
    """Ordinary driving supervision with the same six-view input as Data_LG."""

    def __init__(self, **cfg):

        common_cfg = dict(cfg)
        common_cfg["img_shift_augmentation"] = False  # 六视角训练不使用几何增强

        # 普通Driving语言模式：
        # four_questions为当前完整模型；commentary/qa用于语言类型消融；
        # mixed保留原始SimLingo随机语言任务逻辑。
        language_mode = str(common_cfg.get("driving_language_mode", "four_questions")).lower()
        if language_mode not in {
            "four_questions",
            "commentary",
            "qa",
            "mixed",
        }:
            raise ValueError(
                f"Unsupported driving_language_mode={language_mode!r}"
            )
        common_cfg["driving_language_mode"] = language_mode

        if language_mode == "four_questions":
            common_cfg["use_commentary"] = False
            common_cfg["use_qa"] = False
        elif language_mode == "commentary":
            common_cfg["use_commentary"] = True
            common_cfg["use_qa"] = False
        elif language_mode == "qa":
            common_cfg["use_commentary"] = False
            common_cfg["use_qa"] = True

        # 运行父类__init__函数
        Data_Driving.__init__(self, **common_cfg)

        # 固定语言消融时将对应任务概率设为1；mixed保持原有概率。
        if language_mode == "commentary":
            self.prompt_probabilities = {
                "commentary": 1.0,
                "qa": 0.0,
                "driving": 0.0,
            }
        elif language_mode == "qa":
            self.prompt_probabilities = {
                "commentary": 0.0,
                "qa": 1.0,
                "driving": 0.0,
            }

        self._initialize_surround_dataset()

        # 当前完整模型要求普通Driving必须具有完整Q1-Q4语言监督
        # 因此在Dataset初始化阶段直接过滤缺少完整四问的样本,
        # 避免训练过程中在__getitem__阶段才因缺失四问而报错.
        if language_mode == "four_questions":
            self._filter_samples_with_valid_four_questions()


    def _driving_four_question_label_path_for_index(self,index: int,) -> Path:
        """
        返回当前普通Driving样本对应的专家条件标签路径：
        .../driving_expert_conditioned_actor_selection/0026.json.gz
        """

        measurement_entry = self.measurements[index]

        if isinstance(measurement_entry, np.ndarray):
            measurement_entry = measurement_entry.reshape(-1)[0]
        elif isinstance(measurement_entry, (list, tuple)):
            measurement_entry = measurement_entry[0]

        measurement_dir = Path(
            Data_LG._decode_path(measurement_entry)
        )

        route_dir = measurement_dir.parent

        # 与当前Dataset中“当前帧”的定义保持一致：
        # 当前帧 = sample_start + hist_len - 1
        frame_id = (
            int(self.sample_start[index])
            + int(self.hist_len)
            - 1
        )

        return (
            route_dir
            / str(
                getattr(
                    self,
                    "driving_participant_attention_label_folder",
                    "driving_expert_conditioned_actor_selection",
                )
            )
            / f"{frame_id:04d}.json.gz"
        )


    @staticmethod
    def _extract_four_questions(payload):
        """
        从普通Driving专家条件标签中读取完整Q1-Q4。
        如果任意问题或答案缺失，则返回空列表。
        """

        question_keys = (
            "attention",
            "motion_constraint",
            "driving_response",
            "future_motion",
        )

        core_questions = payload.get(
            "core_questions",
            {},
        )

        questions = []

        if not isinstance(core_questions, dict):
            return questions

        for key in question_keys:

            item = core_questions.get(key)

            if not isinstance(item, dict):
                return []

            question = str(
                item.get("question", "")
            ).strip()

            answer = str(
                item.get("answer", "")
            ).strip()

            if not question or not answer:
                return []

            questions.append(
                {
                    "question": question,
                    "answer": answer,
                }
            )

        return questions


    def _filter_samples_with_valid_four_questions(self):
        """
        four_questions模式下,只保留具有完整Q1-Q4的普通Driving样本.

        注意:
        这里只检查Q1-Q4本身是否完整,不额外要求整个structured-world标签valid=True.

        future_interaction_valid和camera_attention_valid
        仍然继续由原有mask机制控制对应loss是否参与训练.
        """

        original_sample_count = len(self.images)

        valid_indices = []
        reasons = Counter()

        for index in range(original_sample_count):

            label_path = (
                self._driving_four_question_label_path_for_index(
                    index
                )
            )

            # 专家条件标签不存在
            if not label_path.is_file():
                reasons["missing_label"] += 1
                continue

            try:
                payload = Data_LG._load_gzip_json(
                    label_path
                )

            except (
                OSError,
                EOFError,
                ValueError,
            ) as exc:
                reasons[
                    f"read_error:{type(exc).__name__}"
                ] += 1
                continue

            # 必须具有完整Q1-Q4
            questions = self._extract_four_questions(
                payload
            )

            if len(questions) != 4:
                reasons[
                    "incomplete_four_questions"
                ] += 1
                continue

            valid_indices.append(index)

        indices = np.asarray(
            valid_indices,
            dtype=np.int64,
        )

        # surround_images必须与当前base samples严格一一对应
        if len(self.surround_images) != original_sample_count:
            raise RuntimeError(
                "Driving surround index length mismatch before "
                "four-question filtering: "
                f"{len(self.surround_images)} vs "
                f"{original_sample_count}"
            )

        # SurroundDatasetMixin已经提供了统一的基础样本过滤函数：
        # images / boxes / measurements /
        # sample_start / augment_exists
        self._filter_base_samples(
            valid_indices,
            original_sample_count,
        )

        # surround_images也必须使用完全相同的index同步过滤
        self.surround_images = (
            self._filter_index_container(
                self.surround_images,
                indices,
                valid_indices,
            )
        )

        if bool(
            getattr(
                self,
                "surround_print_filter_summary",
                True,
            )
        ):
            print(
                f"[{self.split} Driving four-question samples]: "
                f"kept {len(valid_indices)}/"
                f"{original_sample_count}; "
                f"filtered={dict(reasons)}"
            )

    @staticmethod
    def _replace_with_four_question_language(sample,payload,):
        questions = Data_Driving_Surround._extract_four_questions(payload)

        if len(questions) != 4:
            raise ValueError(
                "Driving expert-conditioned label does not contain "
                "the required four questions"
            )

        base_prompt = str(
            sample.conversation[0]["content"][0]["text"]
        ).strip()
        driving_suffix = "Predict the waypoints."
        if not base_prompt.endswith(driving_suffix):
            raise ValueError(
                "Driving four-question mode expects the base prompt "
                "to end with 'Predict the waypoints.'"
            )
        prefix = base_prompt[:-len(driving_suffix)].strip()

        question_text = " ".join(
            f"Q{index}: {item['question']}"
            for index, item in enumerate(questions, start=1)
        )
        answer_text = " ".join(
            f"A{index}: {item['answer']}"
            for index, item in enumerate(questions, start=1)
        )

        prompt = (
            f"{prefix} Answer the four driving questions "
            "in order and then predict the waypoints. "
            f"{question_text}"
        )
        answer = f"{answer_text} Waypoints:"

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
                    {
                        "type": "image",
                    },
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

        return sample._replace(
            conversation=conversation_all,
            answer=conversation_answer,
        )

    def __getitem__(self, index):
        # 先复用Data_Driving生成waypoints、route、target points及基础prompt;
        # four_questions模式下,后续再将基础driving语言替换为统一Q1-Q4语言
        sample = Data_Driving.__getitem__(self, index)

        image_data = {}

        # 六视角图像的
        self.load_surround_images(image_data,self.surround_images[index],)

        future_interaction_grid = None
        future_interaction_valid = False
        participant_spatial_target = np.zeros((len(CAMERA_ORDER), TOKENS_PER_CAMERA),dtype=np.float32,)
        participant_spatial_valid = False

        # 是否使用未来结构化世界标签监督
        use_future_interaction = bool(getattr(self,"driving_use_future_interaction_grid",False,))

        # 是否使用[6,64]软标签监督
        use_participant_spatial_attention = bool(getattr(self,"driving_use_participant_spatial_attention_supervision",False,))

        # 是否使用与LG相同的固定Q1-Q4语言监督
        use_four_question_language = (str(getattr(self, "driving_language_mode", "four_questions")).lower() == "four_questions")

        if (
            use_future_interaction                # 使用未来结构化世界标签监督
            or use_participant_spatial_attention  # 使用[6,64]软标签监督
            or use_four_question_language         # 使用与LG相同的固定Q1-Q4语言监督
        ):
            measurement_path = Path(Data_LG._decode_path(sample.measurement_path))
            route_dir = measurement_path.parent.parent
            frame_name = measurement_path.name.split(".", 1)[0]

            # 使用未来结构化世界标签监督
            if use_future_interaction:
                
                # 标签所在目录
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

                # future_interaction_grid 为4通道结构化世界标签内容
                # future_interaction_valid 为是都有效
                future_interaction_grid, future_interaction_valid = Data_LG._load_future_interaction_grid(future_interaction_path)

            # 专家条件actor标签同时提供[6,64]软标签与Q1-Q4语言
            if (use_participant_spatial_attention or use_four_question_language):

                # 标签所在目录
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

                    # 加载并解析0026.json.gz的内容
                    participant_attention_payload = Data_LG._load_gzip_json(participant_attention_path)

                    if use_participant_spatial_attention:
                        # 读取并检查六视角注意力监督标签
                        # 如果六视角注意力监督标签有效,那么participant_spatial_target是一个[6,64]的数组,表示六个相机的注意力权重,并且权重和为1,participant_spatial_valid=True
                        # 如果六视角注意力监督标签无效,那么participant_spatial_target是一个[6,64]的零数组,表示六个相机的注意力权重都为0(此时主要是没有主要actor),participant_spatial_valid=False

                        participant_spatial_target, participant_spatial_valid = extract_participant_spatial_attention_supervision(
                            participant_attention_payload,
                            cut_bottom_quarter=bool(
                                self.cut_bottom_quarter
                                or self.img_shift_augmentation
                            ),
                            use_global_img=bool(self.use_global_img),
                        )

                    # 替换语言为Q1-Q4语言监督
                    if use_four_question_language:
                        sample = self._replace_with_four_question_language(
                            sample,
                            participant_attention_payload,
                        )

                elif use_four_question_language:
                    raise FileNotFoundError(
                        "Driving four-question mode requires "
                        f"expert-conditioned label: {participant_attention_path}"
                    )

        # DatasetOutput中的历史字段名继续作为公共批处理接口；
        # target实际形状和语义已经升级为[6,64]参与者空间监督。
        return sample._replace(
            image_ff=image_data["rgb"],
            image_ff_org_size=image_data["rgb_org_size"],
            image_surround=image_data["rgb_surround"],
            image_surround_org_size=(image_data["rgb_surround_org_size"]),
            camera_order=image_data["camera_order"],
            future_interaction_grid=(future_interaction_grid),
            future_interaction_valid=(future_interaction_valid),
            camera_attention_target=(participant_spatial_target),
            camera_attention_valid=(participant_spatial_valid),
        )
