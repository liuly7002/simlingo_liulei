"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson   # 更快的读jso
import json
import numpy as np
import random
import cv2
import re
import gzip

import torch
from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.dataloader.dataset_base import BaseDataset


VIZ_DATA = False   # True: 把前视图、route、waypoints、prompt、answer 可视化出来，方便检查样本构造是否正确。

class Data_Driving(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        super().__init__(dreamer=False, **cfg)  # dreamer=False的含义:告诉父类,现在处理的是driving dataset,不是dreamer dataset

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0) # 禁用opencv多线程

        # 从索引表里取出该sample的元信息(这里说明在父类初始化时,已经提前把整个数据集扫描了一遍,并把每个sample的"索引信息"存到若干列表里,所以这里不是现查目录,而是直接O(1)取出)
        data = {}
        images = self.images[index]                 # 图像路径
        measurements = self.measurements[index]     # measurement路径
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index] # 是否有图像增强

        ######################################################
        ######## load current and future measurements ########
        ######################################################
        # 读取当前帧和未来帧的measurements
        # loaded_measurements: 当前帧及未来若干帧的 measurement 集合
        # current_measurement: 当前帧的 measurement 字典
        # measurement_file_current: 当前 measurement 文件路径
        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        data['measurement_path'] = measurement_file_current

        # Determine whether the augmented camera or the normal camera is used.
        # 决定是否使用图像增强
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']
            aug_translation = current_measurement['augmentation_translation']
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0


        ######################################################
        ################## load waypoints ####################
        ######################################################
        # 加载waypoints(根据当前及未来 measurement，生成监督用的 future waypoints)
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)

        # 速度
        speed_rounded = round(current_measurement['speed'], 1)  # 用于 prompt 文本里显示
        data['speed'] = current_measurement['speed']            # 用于模型输入或监督保留原始数值

        # route导航信息
        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        ######################################################
        ################## get commentary & qa ##################
        ######################################################
        commentary_exists = False
        commentary = ''
        if self.use_commentary:
            commentary_file_path = measurement_file_current.replace('measurements', 'commentary').replace('data/', 'commentary/') # TODO: move to config
            # do not use evaluation routes!!!  验证集不使用 commentary 文件
            if 'validation_' in commentary_file_path:
                commentary_exists = False
            else:
                try:
                    with gzip.open(commentary_file_path, 'rt') as f:
                        commentary_file = ujson.load(f)
                        commentary_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    commentary_exists = False
                    commentary_file = None

                if commentary_file is not None:

                    commentary = commentary_file['commentary']
                    # we only augment in 60% of the cases and use the default commentary in 40% of the cases
                    # augmentation is used to increase generalization to a broader set of sentences
                    # but we do not want to overfit to the augmented sentences
                    # 开启 commentary augmentation 时,只在 60% 的情况下使用增强模板,另外 40% 直接用原始 commentary
                    # 意图: 增强是为了提升句式泛化，但不想让模型过拟合改写模板。
                    if self.commentary_augmentation and random.random() < 0.6:
                        # 说明 commentary augmentation 不是 LLM 生成，而是 模板库替换
                        # 找到这个 commentary 属于哪个模板类型 -> 从对应模板库里随机选一个表达 -> 用 placeholder 字典把 <OBJECT>、<LOCATION> 之类占位符替换掉
                        if commentary_file['commentary_template'] in self.templates_commentary:
                            commentary = random.choice(self.templates_commentary[commentary_file['commentary_template']])
                            for key, value in commentary_file['placeholder'].items():
                                if key in commentary:
                                    commentary = commentary.replace(key, value)
                            # regex check if <OBJECT> or <LOCATION> or any other <> is still in commentary, if so use default commentary_file['commentary']
                            # 健壮性保护: 如果最终替换完之后，还有 <...> 没被替掉，就说明模板有 bug 或 placeholder 不完整，此时回退到原始 commentary。
                            if re.search(r'<.*?>', commentary):
                                print(f"WARNING: {commentary} contains placeholders that are not replaced. Using default commentary.")
                                commentary = commentary_file['commentary']

                    # 文本清洗: 说明模板替换后有时会产生文本瑕疵,所以做简单清洗
                    commentary = commentary.replace('..', '.')
                    commentary = commentary.replace('in in', 'in')

        # 第二类语言监督
        qa_exists = False
        if self.use_qa:
            qa_path = measurement_file_current.replace('measurements', 'vqa').replace('data/', 'drivelm/')
            if 'validation_' in qa_path:
                qa_exists = False
            else:
                try:
                    with gzip.open(qa_path, 'rt') as f:
                        qa = ujson.load(f)
                    qa_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    qa_exists = False
                    qa = None

            if qa_exists:
                qas = qa['QA']
                qas = [values for values in qas.values()] # list of lists
                qas = [item for sublist in qas for item in sublist] # flatten list
                while True:
                    qa_chosen = random.choice(qas)  # 随机选一个QA
                    qa_question = qa_chosen['Q']
                    qa_answer = qa_chosen['A']

                    # 对“无信息/否定型答案”降采样
                    # 含义：
                    #如果这个 QA 的答案属于“无目标 / 无交通灯 / 不受影响 / 无需刹车 / 无法判断”这类低信息答案，
                    # 只保留 20% 概率；
                    # 其余 80% 丢掉重采。
                    # 本质上是在做 informative QA 的重采样增强
                    # 目的很明确 -> 避免训练数据里充斥大量“没有、无、不是、不受影响”这类低价值问答。
                    # TODO: make this nicer!!!
                    if 'There are no pedestrians.' in qa_answer or \
                            'There is no traffic light' in qa_answer or \
                            'There are no pedestrians.' in qa_answer or \
                            'No, the ego vehicle is not affected by a stop sign.' in qa_answer or \
                            'No, the ego vehicle is not affected by a junction.' in qa_answer or \
                            'There is no traffic light affecting the ego vehicle.' in qa_answer or \
                            'There is no stop sign affecting the ego vehicle.' in qa_answer or \
                            'There is no junction affecting the ego vehicle.' in qa_answer or \
                            'It is not possible to tell' in qa_answer or \
                            'There is no reason for the ego vehicle to brake.' in qa_answer:
                        # only keep in 20% of the cases
                        if random.random() < 0.2:
                            break
                    else:
                        break
                
                # we only augment in 60% of the cases and use the default QA in 40% of the cases
                # augmentation is used to increase generalization to a broader set of sentences
                # but we do not want to overfit to the augmented sentences
                # 开启增强时, 60% 用增强模板, 40% 保持原始 QA
                if self.qa_augmentation and random.random() < 0.6:
                    # 做占位符抽象 先把原始question/answer保存
                    qa_question_org = qa_question
                    qa_answer_org = qa_answer
                    locations = [
                        'nearby to the front of the ego vehicle',
                        'nearby to the front right of the ego vehicle',
                        'nearby to the front left of the ego vehicle',
                        'nearby on the left side of the ego vehicle',
                        'far to the front left of the ego vehicle',
                        'far to the front right of the ego vehicle',
                        'far to the front of the ego vehicle',
                        'far to the left side of the ego vehicle',
                        'far to the right side of the ego vehicle',
                        'to the front of the ego vehicle',
                        'to the front right of the ego vehicle',
                        'to the front left of the ego vehicle',
                        'on the left side of the ego vehicle',
                        'on the right side of the ego vehicle',
                    ]
                    objects = [value['Visual_description'] for key, value in qa['key_object_infos'].items()]
                    q_objects = []
                    a_objects = []
                    for object_type in objects:
                        if object_type in qa_question:
                            qa_question = qa_question.replace(object_type, '<OBJECT>')
                            q_objects.append(object_type)
                        if object_type in qa_answer:
                            qa_answer = qa_answer.replace(object_type, '<OBJECT>')
                            a_objects.append(object_type)
                    
                    q_location = ''
                    a_location = ''
                    for location in locations:
                        if location in qa_question:
                            qa_question = qa_question.replace(location, '<LOCATION>')
                            q_location = location
                        if location in qa_answer:
                            qa_answer = qa_answer.replace(location, '<LOCATION>')
                            a_location = location
                        
                    q_distance = re.search(r'in (\d+) m', qa_question)
                    qa_question = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_question)
                    a_distance = re.search(r'in (\d+) m', qa_answer)
                    qa_answer = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_answer)
                    if len(q_objects)==0:
                        q_objects = ['']
                    if len(a_objects)==0:
                        a_objects = ['']
                    
                    # in 40% of the cases we do not augment the question
                    if len(q_objects) > 1 or len(a_objects) > 1 or random.random() < 0.4: 
                        qa_question = qa_question_org
                        qa_answer = qa_answer_org
                    else:
                        if qa_question in self.q_augment:
                            qa_question = random.choice(self.q_augment[qa_question]).replace('<OBJECT>', q_objects[0]).replace('<LOCATION>', q_location)
                            if q_distance:
                                qa_question = qa_question.replace('<DISTANCE>', q_distance.group(1))
                        else:
                            print(f"WARNING: {qa_question} not in q_augment. Using default question.")
                            qa_question = qa_question_org
                        if qa_answer in self.a_augment:
                            qa_answer = random.choice(self.a_augment[qa_answer]).replace('<OBJECT>', a_objects[0]).replace('<LOCATION>', a_location)
                            if a_distance:
                                qa_answer = qa_answer.replace('<DISTANCE>', a_distance.group(1))
                        else:
                            print(f"WARNING: {qa_answer} not in a_augment. Using default answer.")
                            qa_answer = qa_answer_org

        ######################################################
        ######## load navigational_conditioning ########
        ######################################################
        # 构造导航条件
        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)

        # 决定本样本训练成哪一类任务
        answer = ''
        prompt_random = random.random()
        # 任务一: commentary 任务 (触发条件:1.配置打开 2.当前样本确实有commentary 3.随机数落在commentary概率区间)
        if self.use_commentary and commentary_exists and prompt_random < self.prompt_probabilities['commentary']:
            # 子模式A: 20% 让模型“根据 commentary 预测轨迹”
            # 意思是把 commentary 当作语言条件输入,模型要生成轨迹
            # 这其实是 language-conditioned driving
            if random.random() < 0.2: # 20% of the time we give commentary as prompt
                if random.random() < 0.5:
                    # 提示词
                    prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {commentary} Predict the waypoints."
                else:
                    prompt = f"Current speed: {speed_rounded} m/s. Command: {commentary} Predict the waypoints."
                answer = f"Waypoints:"
            # 子模式B：80% 让模型“根据场景生成 commentary，再接轨迹”
            # 用户问：接下来该怎么做？
            # 模型答：先输出 commentary，再输出轨迹
            # 这其实是 driving reasoning + trajectory generation
            # 这也是 SimLingo 这类框架的一个核心特点：
            # 不是只学控制，还学语言解释。
            else:
                # 80% of the time we want to predict commentary
                prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} What should the ego do next?"
                answer = f"{commentary} Waypoints:"
            self.num_sampled_per_type['commentary'] += 1
        # 任务二: QA任务
        elif self.use_qa and qa_exists and prompt_random < (self.prompt_probabilities['qa'] + self.prompt_probabilities['commentary']):
            # 标准 VQA 风格
            # 也就是说此时样本不是预测轨迹，而是回答场景问答
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Q: {qa_question}"
            answer = f"A: {qa_answer}"
            self.num_sampled_per_type['qa'] += 1
        # 任务三: driving任务
        else:
            # 标准的驾驶轨迹预测样本
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Predict the waypoints."
            answer = f"Waypoints:"
            self.num_sampled_per_type['driving'] += 1

        # recalculate the probabilties after warmup (when more than 1000 samples have been sampled)
        # we do this in case we dont have qa or commentary for every sample otherwise it would lead to undersampling one of those
        # 动态重估 prompt 概率
        # 为什么要这么做：因为不是每个 sample 都一定有 commentary / QA。
        # 如果你一开始固定：
        # commentary 30%
        # qa 30%
        # driving 40%
        # 但实际上很多 sample 没 commentary 或没 QA，那么最终真实采样比例可能会严重偏掉。
        # 所以这里每 10000 个样本动态重估一次。
        # 重估公式是什么意思
        # 1 / value 表示: 哪一类已经采得多，就给它更小权重. 哪一类采得少，就给它更大权重.
        # 注意: 这里平衡的不是数据集本身的静态分布，而是 训练过程中实际被采样成不同任务类型的次数。所以它属于 online task balancing。
        if sum(self.num_sampled_per_type.values()) > 10000 and sum(self.num_sampled_per_type.values()) % 10000 == 0:
            self.prompt_probabilities = {key: 1/value for key, value in self.num_sampled_per_type.items()}
            self.prompt_probabilities = {key: value/sum(self.prompt_probabilities.values()) for key, value in self.prompt_probabilities.items()}
            print(f"Prompt probabilities: {self.prompt_probabilities}")
            print(f"Number of samples per type: {self.num_sampled_per_type}")

        # 清洗 prompt / answer 文本,防止模板拼接后出现双句点
        answer = answer.replace('..', '.')
        prompt = prompt.replace('..', '.')

        ######################################################
        ######## load current and past images ########
        ######################################################
        # 读取图像
        data = self.load_images(data, images, augment_sample=augment_sample)
        

        # 构造对话格式   这里开始把 driving sample 变成 LLM/VLM 训练输入格式
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

        # 这里虽然只用了一张前视图，但仍然包成 list，说明整个系统接口可能支持多图输入。
        images = [data['rgb']]

        # 最终返回结果
        data_new = DatasetOutput(
            conversation = conversation_all,         # 完整 user-assistant 对话，给模型做输入格式组织用。
            answer = conversation_answer,            # 只包含监督答案，通常给 loss 计算用。
            image_ff = data['rgb'],                  # front-forward image，当前样本前视图。
            image_ff_org_size=data['rgb_org_size'],  # 原图尺寸。后处理或可视化时可能要用。
            waypoints = data["waypoints"],           # 未来轨迹监督。
            waypoints_1d = data["waypoints_1d"],     # waypoint 的另一种表示形式，可能用于某些特定训练头或损失。
            path = data['route_adjusted'],           # route 监督，这里是 route_adjusted。
            target_points = data['target_points'],   # 导航目标点
            speed = data['speed'],                   # 当前速度
            placeholder_values = placeholder_values, # 导航模板相关占位值
            measurement_path = data['measurement_path'],  # 当前样本来源，方便 debug
            dataset = 'driving',                     # 显式标记这个样本来自 driving dataset
        )

        # 可视化调试 如果打开,会把图像、commentary、route、waypoints、prompt、answer一起画出来，方便人工检查这个样本是否构造合理
        if VIZ_DATA:
            # front image with path and waypoints and commentary
            self.visualise_cameras(data_new, commentary, data['route_adjusted'], data['waypoints'], options=None, prompt=prompt, answer=answer, name="img")

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
    
    cfg.data_module.base_dataset.use_commentary = True            # 使用注释
    cfg.data_module.base_dataset.use_qa = True                    # 使用QA
    cfg.data_module.base_dataset.img_shift_augmentation = False   # 不进行图像增强

    print('Test Dataset')
    dataset = Data_Driving(                        
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