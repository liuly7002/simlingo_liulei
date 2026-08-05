# -*- coding: utf-8 -*-


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

        # 运行父类__init__函数
        Data_Driving.__init__(self, **common_cfg)
        self._initialize_surround_dataset()

    def __getitem__(self, index):
        # Preserve all original driving/commentary/QA language logic,
        # waypoints, route, target points and prompt sampling.
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

        if (use_future_interaction or use_participant_spatial_attention):
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

            # 使用[6,64]软标签监督
            if use_participant_spatial_attention:

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
