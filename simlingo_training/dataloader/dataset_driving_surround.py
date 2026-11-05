# -*- coding: utf-8 -*-


from collections import Counter
from pathlib import Path

import numpy as np

from simlingo_training.dataloader.dataset_base_surround import SurroundDatasetMixin
from simlingo_training.dataloader.dataset_driving import Data_Driving
from simlingo_training.dataloader.dataset_lg import Data_LG
from simlingo_training.dataloader.stage2_data import load_stage2_grid, strict_negative_actor_status
from simlingo_training.dataloader.participant_spatial_attention import CAMERA_ORDER, TOKENS_PER_CAMERA, extract_participant_spatial_attention_supervision


class Data_Driving_Surround(SurroundDatasetMixin,Data_Driving,):

    def __init__(self, **cfg):


        ################################## 🥭 设置配置文件 🥭 ##################################
        common_cfg = dict(cfg)
        common_cfg["img_shift_augmentation"] = False  # 六视角训练不使用几何增强！！！

        # 普通Driving的语言模式：
        # four_questions为当前完整模型；
        # commentary/qa用于语言类型消融
        # mixed保留原始SimLingo随机语言任务逻辑
        language_mode = str(common_cfg.get("driving_language_mode", "four_questions")).lower()
        # 安全性检查
        if language_mode not in {"four_questions", "commentary", "qa", "mixed",}:
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
        # 其他情况就是simlingo_training/config/experiment/simlingo_lg_seed1.yaml中的 common_cfg["use_commentary"] = True common_cfg["use_qa"] = True


        ################################## 🥭 调用Data_Driving类初始化函数 🥭 ##################################

        # 继承了Data_Driving,这里运行父类__init__函数,这里主要是数据的形式了
        Data_Driving.__init__(self, **common_cfg)

        # 固定语言消融时将对应任务概率设为1；mixed保持原有概率。
        if language_mode == "commentary":
            self.prompt_probabilities = { "commentary": 1.0, "qa": 0.0, "driving": 0.0,}
        elif language_mode == "qa":
            self.prompt_probabilities = { "commentary": 0.0, "qa": 1.0, "driving": 0.0,}


        ################################## 🥭 调用SurroundDatasetMixin类初始化函数 🥭 ##################################

        # 继承了 SurroundDatasetMixin,形成完整的数据池(这里主要是为了将六视角图像加进来),这里还是文件的形式
        self._initialize_surround_dataset()

        # 当前完整模型要求普通Driving必须具有完整Q1-Q4语言监督
        # 因此在Dataset初始化阶段直接过滤缺少完整四问的样本,
        # 避免训练过程中在__getitem__阶段才因缺失四问而报错.
        if language_mode == "four_questions":
            self._filter_samples_with_valid_four_questions()


    def _driving_four_question_label_path_for_index(self,index: int,) -> Path:
        """
        返回当前普通Driving样本对应的专家标签路径:
        /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_25/Town12_Rep0_101_route0_09_13_21_09_57/driving_expert_conditioned_actor_selection/0026.json.gz
        """

        # /root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/measurements'
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
        frame_id = (int(self.sample_start[index]) + int(self.hist_len) - 1)

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
        从driving标签中读取完整Q1-Q4。
        如果任意问题或答案缺失，则返回空列表。
        """

        question_keys = ("attention","motion_constraint","driving_response","future_motion",)

        core_questions = payload.get("core_questions",{},)

        questions = []

        # 安全性检查: 如果core_questions不是字典,则说明标签不完整,返回空列表
        if not isinstance(core_questions, dict):
            return questions

        # 遍历Q1-Q4的键,依次读取问题和答案
        for key in question_keys:

            item = core_questions.get(key)

            # 安全性检查: 如果item不是字典,则说明标签不完整,返回空列表
            if not isinstance(item, dict):
                return []

            question = str(item.get("question", "")).strip()

            answer = str(item.get("answer", "")).strip()

            # 安全性检查: 如果问题或答案为空,则说明标签不完整,返回空列表
            if not question or not answer:
                return []

            questions.append({"question": question,"answer": answer,})

        return questions


    def _filter_samples_with_valid_four_questions(self):

        # 只是全部样本的数量,这里以self.images为基准计算的
        original_sample_count = len(self.images)

        valid_indices = []
        reasons = Counter()

        # 遍历所有样本
        for index in range(original_sample_count):

            #保存 Driving 数据关键 Actor 标签的文件夹
            # /root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/driving_expert_conditioned_actor_selection/0026.json.gz
            label_path = (self._driving_four_question_label_path_for_index(index))  

            # 专家标签不存在
            if not label_path.is_file():
                reasons["missing_label"] += 1
                continue

            # 专家标签存在,尝试读取
            try:
                # 解析文件
                payload = Data_LG._load_gzip_json(label_path)
            # 读取文件失败,可能是文件损坏或格式错误
            except (OSError,EOFError,ValueError,) as exc:
                reasons[f"read_error:{type(exc).__name__}"] += 1
                continue

            # [{问题1,答案1},{问题2,答案2},{问题3,答案3},{问题4,答案4}]
            questions = self._extract_four_questions(payload)

            # 安全性检查: 如果问题数量不等于4,则说明标签不完整,跳过该样本
            if len(questions) != 4:
                reasons["incomplete_four_questions"] += 1
                continue

            # 当前样本通过了所有检查,保留该样本的索引
            valid_indices.append(index)

        indices = np.asarray(valid_indices,dtype=np.int64,)

        # 安全性检查
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
        self._filter_base_samples(valid_indices,original_sample_count,)

        # surround_images也必须使用完全相同的index同步过滤
        self.surround_images = (self._filter_index_container(self.surround_images,indices,valid_indices,))

        if bool(getattr(self,"surround_print_filter_summary",True,)):
            print(
                f"[{self.split} Driving four-question samples]: "
                f"kept {len(valid_indices)}/"
                f"{original_sample_count}; "
                f"filtered={dict(reasons)}"
            )

    @staticmethod
    def _replace_with_four_question_language(sample,payload,):

        # [{问题1,答案1},{问题2,答案2},{问题3,答案3},{问题4,答案4}]
        questions = Data_Driving_Surround._extract_four_questions(payload)

        # 安全性检查: 如果问题数量不等于4,则说明标签不完整,抛出异常
        if len(questions) != 4:
            raise ValueError(
                "Driving expert-conditioned label does not contain "
                "the required four questions"
            )

        # 这里是将原始的Driving语言替换为统一的Q1-Q4语言,并且在prompt中加入了"Answer the four driving questions in order and then predict the waypoints."
        base_prompt = str(sample.conversation[0]["content"][0]["text"]).strip()  # 这就是prompt
        
        # 在four_questions模式下,driving语言的内容为"Current speed: {speed_rounded} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>). Predict the waypoints."
        driving_suffix = "Predict the waypoints." 
        
        # 这里是安全性检查,确保原始的Driving语言以"Predict the waypoints."结尾,否则抛出异常
        if not base_prompt.endswith(driving_suffix):
            raise ValueError(
                "Driving four-question mode expects the base prompt "
                "to end with 'Predict the waypoints.'"
            )

        # 将base_prompt去掉"Predict the waypoints."后,得到的前缀部分"Current speed: {speed_rounded} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>).",用于构建新的prompt
        prefix = base_prompt[:-len(driving_suffix)].strip()

        # 依次为四个问题 Q1: 问题 Q2: 问题 Q3: 问题 Q4: 问题
        question_text = " ".join(f"Q{index}: {item['question']}" for index, item in enumerate(questions, start=1))
        
        # 依次为四个答案 A1: 答案 A2: 答案 A3: 答案 A4: 答案
        answer_text = " ".join(f"A{index}: {item['answer']}" for index, item in enumerate(questions, start=1))

        # "Current speed: {speed_rounded} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>). Answer the four driving questions in order and then predict the waypoints. Q1: 问题1 Q2: 问题2 Q3: 问题3 Q4: 问题4"
        prompt = (
            f"{prefix} Answer the four driving questions "
            "in order and then predict the waypoints. "
            f"{question_text}"
        )

        # "A1: 答案1 A2: 答案2 A3: 答案3 A4: 答案4 Waypoints:"
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

        """
        prompt =
        
        Current speed: ... m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>).
        Answer the four driving questions in order and then predict the waypoints.
        Q1: 相应问题?
        Q2: 相应问题？
        Q3: 相应问题？
        Q4: 相应问题?

        answer =
        A1: 相应答案。
        A2: 相应答案。
        A3: 相应答案。
        A4: 相应答案。
        Waypoints:
        """

        return sample._replace(
            conversation=conversation_all,
            answer=conversation_answer,
        )

    def __getitem__(self, index):

        # 这里调用父类Data_Driving的__getitem__方法,获取当前样本的基本信息
        sample = Data_Driving.__getitem__(self, index)

        image_data = {}

        # [B, T, V, C, H, W] 六视角图像
        self.load_surround_images(image_data,self.surround_images[index],)

        future_interaction_grid = None
        future_interaction_temporal = None
        future_interaction_valid = False
        critical_actor_status = -1
        participant_spatial_target = np.zeros((len(CAMERA_ORDER), TOKENS_PER_CAMERA),dtype=np.float32,)
        participant_spatial_valid = False

        # 是否使用未来结构化世界标签监督
        use_future_interaction = bool(getattr(self,"driving_use_future_interaction_grid",False,))  # True

        # 是否使用[6,64]软标签监督
        use_participant_spatial_attention = bool(getattr(self,"driving_use_participant_spatial_attention_supervision",False,))

        # 是否使用与LG相同的固定Q1-Q4语言监督  
        use_four_question_language = (str(getattr(self, "driving_language_mode", "four_questions")).lower() == "four_questions")

        if (
            use_future_interaction                # 使用未来结构化世界标签监督
            or use_participant_spatial_attention  # 使用[6,64]软标签监督
            or use_four_question_language         # 使用Q1-Q4语言监督
        ):
            
            # 当前帧.json.gz的路径
            # /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_25/Town12_Rep0_101_route0_09_13_21_09_57/measurements/0000.json.gz
            measurement_path = Path(Data_LG._decode_path(sample.measurement_path))

            # /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_25/Town12_Rep0_101_route0_09_13_21_09_57
            route_dir = measurement_path.parent.parent

            # 当前帧的文件名0000.json.gz,去掉后缀.json.gz,得到0000
            frame_name = measurement_path.name.split(".", 1)[0]

            # 如果使用交通交互世界标签监督
            if use_future_interaction:
                
                # 标签所在目录 
                # /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_25/Town12_Rep0_101_route0_09_13_21_09_57/driving_future_interaction_grids/0000.npz
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

                # 三通道结构化世界 + C1/C2逐帧占据 + 关键Actor三状态。
                (
                    future_interaction_grid,           # [3,H,W] 三通道占据网格,分别是 -> C0:参考路径对应的自车 Footprint 占据, C1:自车未来10帧 Footprint 的平均占据, C2:关键 Actor 未来10帧 Footprint 的平均占据
                    future_interaction_valid,          # bool, 表示当前帧的结构化未来世界标签是否有效 True: 标签有效, False: 标签无效,不应直接作为有效的未来世界监督
                    future_interaction_temporal,       # [2,10,H,W] 未来10帧的逐帧占据信息 两个通道分别是 -> temporal[0]:自车未来10帧的逐帧 Footprint 占据, temporal[1]:关键 Actor 未来10帧的逐帧 Footprint 占据
                    critical_actor_status,             # int, 关键 Actor 的三状态标签 -1:无法可靠判断,关键 Actor 监督应屏蔽, 0:未确认存在决策关键Actor, 1:存在确认的决策关键 Actor
                ) = load_stage2_grid(future_interaction_path)

                # 
                if bool(getattr(self, "stage2_strict_negative_mask", True)) and critical_actor_status == 0:
                    critical_actor_status = strict_negative_actor_status(
                        critical_actor_status,
                        route_dir / str(getattr(self, "lg_label_folder", "language_grounded_waypoints")) / f"{frame_name}.json.gz",
                    )

            # 专家条件actor标签同时提供[6,64]软标签与Q1-Q4语言
            if (
                use_participant_spatial_attention   # 使用[6,64]软标签监督
                or use_four_question_language       # 使用Q1-Q4语言监督
            ):

                # 标签所在目录
                # /root/autodl-tmp/database/simlingo_v2_2026_09_12_22_24_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_25/Town12_Rep0_101_route0_09_13_21_09_57/driving_expert_conditioned_actor_selection/0000.json.gz
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

                # 如果专家条件actor标签存在,则加载并解析其内容
                if participant_attention_path.is_file():

                    # 加载并解析participant_attention_path的内容
                    participant_attention_payload = Data_LG._load_gzip_json(participant_attention_path)

                    # 如果使用[6,64]软标签监督,则读取并检查六视角注意力监督标签
                    if use_participant_spatial_attention:
                        """
                        读取并检查六视角注意力监督标签
                        如果六视角注意力监督标签有效,那么participant_spatial_target是一个[6,64]的数组,表示六个相机的注意力权重,并且权重和为1,participant_spatial_valid=True
                        如果六视角注意力监督标签无效,那么participant_spatial_target是一个[6,64]的零数组,表示六个相机的注意力权重都为0(此时主要是没有主要actor),participant_spatial_valid=False
                        """
                        participant_spatial_target, participant_spatial_valid = extract_participant_spatial_attention_supervision(
                            participant_attention_payload,
                            cut_bottom_quarter=bool(
                                self.cut_bottom_quarter
                                or self.img_shift_augmentation
                            ),
                            use_global_img=bool(self.use_global_img),
                        )
                        # Status -1 (including unconfirmed negative) must not supervise actor attention.
                        if critical_actor_status != 1 or not future_interaction_valid:
                            participant_spatial_valid = False

                    # 替换语言为Q1-Q4语言监督
                    if use_four_question_language:
                        sample = self._replace_with_four_question_language(sample,participant_attention_payload,)

                elif use_four_question_language:
                    raise FileNotFoundError(
                        "Driving four-question mode requires "
                        f"expert-conditioned label: {participant_attention_path}"
                    )


        return sample._replace(
            image_ff=image_data["rgb"],
            image_ff_org_size=image_data["rgb_org_size"],

            ###################### 实际上下面这部分是在原来的基础上新增的内容 ######################
            """
            主要是新增了:
            1. 六视角图像数据
            2. 交通交互世界标签相关信息
            3. 六相机注意力权重标签相关信息
            """
            image_surround=image_data["rgb_surround"],                        # 
            image_surround_org_size=(image_data["rgb_surround_org_size"]),
            camera_order=image_data["camera_order"],                          # ['front', 'front_left', 'front_right', 'rear', 'rear_left', 'rear_right']
            future_interaction_grid=(future_interaction_grid),                # [3,H,W] 三通道占据网格,分别是 -> C0:参考路径对应的自车 Footprint 占据, C1:自车未来10帧 Footprint 的平均占据, C2:关键 Actor 未来10帧 Footprint 的平均占据
            future_interaction_valid=(future_interaction_valid),              # bool, 表示当前帧的结构化未来世界标签是否有效 True: 标签有效, False: 标签无效,不应直接作为有效的未来世界监督
            future_interaction_temporal=(future_interaction_temporal),        # [2,10,H,W] 未来10帧的逐帧占据信息 两个通道分别是 -> temporal[0]:自车未来10帧的逐帧 Footprint 占据, temporal[1]:关键 Actor 未来10帧的逐帧 Footprint 占据
            critical_actor_status=(critical_actor_status),                    # int, 关键 Actor 的三状态标签 -1:无法可靠判断,关键 Actor 监督应屏蔽, 0:未确认存在决策关键Actor, 1:存在确认的决策关键 Actor
            camera_attention_target=(participant_spatial_target),             # [6,64]的数组,表示六个相机的注意力权重,并且权重和为1
            camera_attention_valid=(participant_spatial_valid),               # participant_spatial_valid=True/False
        )
