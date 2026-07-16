"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson
import numpy as np
import random
import cv2
import gzip

import torch
from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.dataloader.dataset_base import BaseDataset


VIZ_DATA = False

class Data_Dreamer(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        super().__init__(dreamer=True, **cfg)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)   # 禁用OpenCV多线程，避免与PyTorch的多线程冲突








        ########################################### 🥭 初始化(父类初始化得到) 🥭 ###########################################
        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]
        alternative_trajectories = self.alternative_trajectories[index]

        ########################################### 🥭 measurements 🥭 ###########################################

        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(measurements, sample_start)
        data['measurement_path'] = measurement_file_current



        if self.use_safety_flag:
            if random.random() < 0.5:
                activate_safety = True
            else:
                activate_safety = False
        else:  # 执行
            activate_safety = None


        ########################################### 🥭 是否进行数据增强 🥭 ###########################################
        # if we want to use the alternative trajectories, we cant take the augmented images, since alternatives are calculated for the original view only
        # if activate_safety is not None and activate_safety == False or activate_safety is None:
        augment_sample = False
        aug_rotation = 0.0
        aug_translation = 0.0





        ########################################### 🥭 waypoints 🥭 ###########################################
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
       





        ########################################### 🥭 当前帧的车速 🥭 ###########################################
        speed_rounded = round(current_measurement['speed'], 1)
        data['speed'] = current_measurement['speed']







        ########################################### 🥭 route 🥭###########################################
        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)







        ########################################### 🥭 target point 🥭 ###########################################
        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        
        ########################################### 🥭 next target point 🥭 ###########################################
        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)







        ######################################################
        ################## get alternatives ##################
        ######################################################
        alternative_file = str(alternative_trajectories, encoding='utf-8')
        with gzip.open(alternative_file, 'rt') as f1:
            alternative_trajectories = ujson.load(f1)

        options = []
        for key, option in alternative_trajectories.items():
            if 'factor' in key:
                continue
            
            options.extend(option)

        chosen_option = random.choice(options)

        # replace 'org' with the original route
        if chosen_option['route'] == 'org': 
            chosen_option['route'] = data['route_adjusted_org']   # 这表示使用数据集原始的规划路径
        else:
            chosen_option['route'] = np.array(chosen_option['route'])  # 表示使用规划出来的路径
        
        if chosen_option['waypoints'] == 'org':
            chosen_option['waypoints'] = data['waypoints_org']    # 表示使用数据集原始的waypoints
        else:
            chosen_option['waypoints'] = np.array(chosen_option['waypoints'])  # 表示使用规划出来的waypoints
        
        chosen_option['dreamer_instruction'] = random.choice(chosen_option['dreamer_instruction'])

        dreamer_answer = f"Following the given instruction. Waypoints:"
        if activate_safety is not None:
            if activate_safety:
                if chosen_option['safe_to_execute']:
                    augment_sample = False
                else:
                    dreamer_answer = chosen_option['dreamer_answer_safety']
            else:
                augment_sample = False
        








        ########################################### 🥭 target_options, placeholder_values 🥭 ###########################################

        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
        """
        target_options = 
        [
        "Target waypoint: <TARGET_POINT><TARGET_POINT>.",
        "Command: {command} in {dist_to_command} meter {next_command}.",
        "Command: {lmdrive_command}."
        ]   # lmdrive_command 来自"/data/augmented_templates/lmdrive.json"语言增强模板文件
        
        placeholder_values = 
        {
        '<TARGET_POINT>': [[x_0, y_0], [x_1, y_1]]
        }
        """








        answer = ''

        if random.random() < 0.8:
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {chosen_option['dreamer_instruction']}"
        else:
            prompt = f"Current speed: {speed_rounded} m/s. {chosen_option['dreamer_instruction']}"
            
        waypoints = chosen_option['waypoints']
        waypoints = np.array(waypoints)
        
        waypoints_zero = np.concatenate((np.zeros((1, 2)), waypoints), axis=0)
        waypoints_1d = [np.linalg.norm(waypoints_zero[i+1] - waypoints_zero[i]) for i in range(len(waypoints_zero)-1)]
        waypoints_1d = np.cumsum(waypoints_1d)
        waypoints_1d = [[x, 0] for x in waypoints_1d]
        waypoints_1d = np.array(waypoints_1d).reshape(-1, 2)
        
        path = chosen_option['route']
        answer = dreamer_answer

        prompt = prompt.replace('..', '.').replace('  ', ' ').replace('!.', '!').replace('?.', '?')
                










        ############################################# 🥭 前视图像 🥭 #############################################
        data = self.load_images(data, images, augment_sample=augment_sample)
        
        # overwrite action when safety flag is active and action is not allowed
        if activate_safety is not None:
            if activate_safety:
                prompt = f"<SAFETY> {prompt}"
                if chosen_option['safe_to_execute'] == False:
                    waypoints = data['waypoints_org']
                    waypoints_1d = data["waypoints_1d"]
                    path = data['route_adjusted_org']
            else:
                prompt = f"<INSTRUCTION_FOLLOWING> {prompt}"








        ############################################# 🥭 构造对话格式 🥭 #############################################

        # 1. 只包含答案的版本  这是仅包含 assistant 输出的部分，通常用于监督目标
        conversation_answer = [
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            },
        ]
        # 2. 完整对话版本
        # 这是标准的多模态对话格式：user 发出文字 prompt，并附一张图片  assistant 输出文字答案
        conversation_all = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            }
        ]
        



        ############################################# 🥭 前视图像 🥭 #############################################

        # 这里虽然只用了一张前视图，但仍然包成 list，说明整个系统接口可能支持多图输入。
        images = [data['rgb']]






        # 最终的返回结果
        data_new = DatasetOutput(
            conversation = conversation_all,             # 完整 user-assistant 对话，给模型做输入格式组织用。
            answer = conversation_answer,                # 只包含监督答案，通常给 loss 计算用。
            image_ff = data['rgb'],                      # front-forward image，当前样本前视图。
            image_ff_org_size=data['rgb_org_size'],      # 原图尺寸。后处理或可视化时可能要用。
            waypoints = waypoints,
            waypoints_1d = waypoints_1d,
            path = path,
            target_points = data['target_points'],       # 导航目标点
            speed = data['speed'],                       # 当前速度
            placeholder_values = placeholder_values,     # 导航模板相关占位值
            measurement_path = data['measurement_path'], # 当前样本来源，方便 debug
            dataset = 'driving',                         # 显式标记这个样本来自 driving dataset
        )
        
        if VIZ_DATA:
            # front image with path and waypoints and commentary
            self.visualise_cameras(data_new, None, path, waypoints, options, name="dreamer_", prompt=prompt, answer=answer)
        return data_new


if __name__ == "__main__":
    from hydra import compose, initialize
    from simlingo_training.config import TrainConfig
    
    # seed all
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

    initialize(config_path="../config")
    cfg = compose(config_name="config")
    
    cfg.data_module.base_dataset.use_commentary = False
    cfg.data_module.base_dataset.img_shift_augmentation = True
    
    cfg.data_module.base_dataset.use_safety_flag = True

    print('Test Dataset')
    dataset = Data_Dreamer(                        
                        split="train",
                        bucket_name='all',
                        **cfg.data_module,
                        **cfg.data_module.base_dataset,
    )

    for i in range(len(dataset)):
        # shuffle
        # i = np.random.randint(0, len(dataset))
        data = dataset[i]
        # print(data)
        # if i == 100:
        #     break