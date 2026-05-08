"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson
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
    
    
    
    def __init__(self, **cfg,):
        super().__init__(dreamer=False, **cfg)  # dreamer=False的含义:告诉父类,现在处理的是driving dataset,不是dreamer dataset

    
    
    
    
    
    
    def __getitem__(self, index):
        cv2.setNumThreads(0) # 禁用opencv多线程

        
        
        
        
        
        ########################################### 🥭 初始化(父类初始化得到) 🥭 ###########################################

        # 从索引表里取出该sample的元信息(这里说明在父类初始化时,已经提前把整个数据集扫描了一遍,并把每个sample的"索引信息"存到若干列表里,所以这里不是现查目录,而是直接O(1)取出)
        data = {}
        images = self.images[index]                 # 图像路径
        measurements = self.measurements[index]     # measurement路径
        sample_start = self.sample_start[index]     # 当前样本的帧id
        augment_exists = self.augment_exists[index] # 当前样本是否增强
        # images: [b'/root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/rgb/0026.jpg'], 
        # measurements: [b'/root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/measurements'], 
        # sample_start: 26, 
        # augment_exists: True
        
        
        
        
        
        
        
        ########################################### 🥭 measurements 🥭 ###########################################
        
        
        # loaded_measurements: 当前帧及未来11帧的 .json 内容 总共12帧
        # current_measurement: 当前帧的 .json.gz 内容
        # measurement_file_current: 当前 measurement .json.gz 文件路径
        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(measurements, sample_start)
        data['measurement_path'] = measurement_file_current  # 当前帧 .json.gz 文件路径
        # current_measurement[0] = {'pos_global': [-1932.94677734375, 6059.3369140625], 'theta': 2.716931982836451, 'speed': 10.893928527832031, 'target_speed': 10.0, 'speed_limit': 13.88888888888889, 'target_point': [166.21953678063582, -1.4034610299303338], 'target_point_next': [292.23548401248064, -4.342632889987669], 'command': 4, 'next_command': 4, 'aim_wp': [3.9551499024618035, 0.0007044955744153203], 'route': [[2.4544338729054394, 0.001334758810199066], [3.45489382915608, 0.001178228841105744], [4.45530941054533, 0.00044431978983006104], [5.455462377914564, 0.00014897731342955467], [6.455837259630751, -0.0007319459886239166], [7.456450110816944, -0.001226608638614568], [8.456802089316398, -0.002246499088830234], [9.45680703690629, -0.003423308025632288], [10.457292488832373, -0.0043828452118344075], [11.457613323532335, -0.005416818090294484], [12.457733948091944, -0.006541320864964284], [13.45801853223297, -0.007720296218779232], [14.458442006321436, -0.00937234785163632], [15.45912646937736, -0.011056433010823596], [16.4552730284497, -0.012434575571900197], [17.453081922137564, -0.014250703843281087], [18.45291606253329, -0.015772686794809587], [19.45341345977995, -0.017798580123078445], [20.453905537485518, -0.01996084850029245], [21.454136713433837, -0.02250902934549437], [22.454383848005413, -0.02464808504385907], [23.454714403468685, -0.02674941995219271], [24.455156186787544, -0.028800460473815903], [25.45528145442136, -0.031262560656781346], [26.45560789960201, -0.034169572262111814], [27.455783461622538, -0.036742900300669845], [28.45644479962843, -0.040168330393921536], [29.45702150748416, -0.042962179629153496], [30.457035547262382, -0.04547457419882939], [31.457073283721886, -0.04878007186882449], [32.457305668928676, -0.05199755436207809], [33.457593668063254, -0.055189889661976466], [34.45806035732186, -0.05870333493197144], [35.45899565468412, -0.06280870575544562], [36.45900558417972, -0.06612677702211833], [37.45917171637654, -0.06964215680661745], [38.45955619400193, -0.07386262451469605], [39.45984008285389, -0.07786063651159125], [40.45992279415901, -0.0814137370861232], [41.4604409870628, -0.085707711859758]], 'route_original': [[2.4544338729054394, 0.001334758810199066], [3.45489382915608, 0.001178228841105744], [4.45530941054533, 0.00044431978983006104], [5.455462377914564, 0.00014897731342955467], [6.455837259630751, -0.0007319459886239166], [7.456450110816944, -0.001226608638614568], [8.456802089316398, -0.002246499088830234], [9.45680703690629, -0.003423308025632288], [10.457292488832373, -0.0043828452118344075], [11.457613323532335, -0.005416818090294484], [12.457733948091944, -0.006541320864964284], [13.45801853223297, -0.007720296218779232], [14.458442006321436, -0.00937234785163632], [15.45912646937736, -0.011056433010823596], [16.4552730284497, -0.012434575571900197], [17.453081922137564, -0.014250703843281087], [18.45291606253329, -0.015772686794809587], [19.45341345977995, -0.017798580123078445], [20.453905537485518, -0.01996084850029245], [21.454136713433837, -0.02250902934549437], [22.454383848005413, -0.02464808504385907], [23.454714403468685, -0.02674941995219271], [24.455156186787544, -0.028800460473815903], [25.45528145442136, -0.031262560656781346], [26.45560789960201, -0.034169572262111814], [27.455783461622538, -0.036742900300669845], [28.45644479962843, -0.040168330393921536], [29.45702150748416, -0.042962179629153496], [30.457035547262382, -0.04547457419882939], [31.457073283721886, -0.04878007186882449], [32.457305668928676, -0.05199755436207809], [33.457593668063254, -0.055189889661976466], [34.45806035732186, -0.05870333493197144], [35.45899565468412, -0.06280870575544562], [36.45900558417972, -0.06612677702211833], [37.45917171637654, -0.06964215680661745], [38.45955619400193, -0.07386262451469605], [39.45984008285389, -0.07786063651159125], [40.45992279415901, -0.0814137370861232], [41.4604409870628, -0.085707711859758]], 'changed_route': False, 'speed_reduced_by_obj_type': None, 'speed_reduced_by_obj_id': None, 'speed_reduced_by_obj_distance': None, 'steer': 0.0, 'throttle': 0.0, 'brake': False, 'control_brake': True, 'junction': False, 'vehicle_hazard': False, 'vehicle_affecting_id': None, 'light_hazard': False, 'walker_hazard': False, 'walker_affecting_id': None, 'stop_sign_hazard': False, 'stop_sign_close': False, 'walker_close': False, 'walker_close_id': None, 'angle': 0.00011339540056267717, 'augmentation_translation': 0.36125444482272595, 'augmentation_rotation': 5.24434331572315, 'ego_matrix': [[-0.9111607074737549, -0.4120118319988251, 0.005693943705409765, -1932.94677734375], [0.4120037257671356, -0.9111785292625427, -0.002586618298664689, 6059.3369140625], [0.0062539163045585155, -1.0898917935264762e-05, 0.9999804496765137, 377.0238952636719], [0.0, 0.0, 0.0, 1.0]]}
        # measurement_file_current: /root/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_532_route0_02_28_11_05_28/measurements/0010.json.gz











        ########################################### 🥭 图像增强 🥭 ###########################################

        # 决定当前帧是否使用图像增强
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']        # R矩阵
            aug_translation = current_measurement['augmentation_translation']  # T矩阵
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0


        
        
        
        
        
        ########################################### 🥭 waypoints 🥭 ###########################################

        # 加载waypoints(根据当前帧及未来11帧 measurement，生成监督用的 future waypoints)
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)

        # data['waypoints']            : 自车坐标系下自车未来10帧(不包括当前帧)自车的位置 [x,y](若增强则增强)
        # dsta['waypoints_org']        : 自车坐标系下自车未来10帧(不包括当前帧)自车的位置 [x,y]（无增强）
        # dsta['waypoints_1d']         : 自车坐标系下 10 帧距离 [x,0] (x是自车当前帧与第1、2、、、11帧之间的欧式距离)(没有增强)
        # dsta['ego_waypoints']        : 11 个 4×4 矩阵（增强）包括当前帧及未来 10 帧
        # dsta['ego_waypoints_org']    : 11 个 4×4 矩阵（无增强）包括当前帧及未来 10 帧

        
        
        
        
        
        ########################################### 🥭 当前帧的车速 🥭 ###########################################

        # 速度
        speed_rounded = round(current_measurement['speed'], 1)  # 用于 prompt 文本里显示 小数后一位
        data['speed'] = current_measurement['speed']            # 用于模型输入或监督保留原始数值







        ########################################### 🥭 route 🥭###########################################
        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)






        
        
        ########################################### 🥭 target point 🥭 ###########################################

        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        # "target_point": [19.075672365994425,-13.26871181961684]

        ########################################### 🥭 next target point 🥭 ###########################################

        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        # "target_point_next": [19.703241532357737,-43.26213909836339]

        
        
        
        
        
        
        
        
        
        ########################################### 🥭 第一类语言监督 commentary 文件 🥭 ###########################################

        commentary_exists = False
        commentary = ''
        if self.use_commentary:  # 执行
            commentary_file_path = measurement_file_current.replace('measurements', 'commentary').replace('data/', 'commentary/') # TODO: move to config
            # commentary_file_path = /root/simlingo/database/simlingo_v2_2026_02_28/commentary/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/commentary/0026.json.gz...
            
            
            # do not use evaluation routes!!!  验证集不使用 commentary 文件
            if 'validation_' in commentary_file_path:
                commentary_exists = False
            else:
                try:
                    with gzip.open(commentary_file_path, 'rt') as f:
                        commentary_file = ujson.load(f)
                        commentary_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):  # 执行 因为目前commentary_file_path不存在
                    commentary_exists = False
                    commentary_file = None

                if commentary_file is not None:

                    commentary = commentary_file['commentary']

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

        
        
        
        
        
        ########################################### 🥭 第二类语言监督 本地生成的语言 (这是我们生成的drivelm语言QA) 🥭 ###########################################

        
        qa_exists = False   # 先初始化语言文件不存在
        
        if self.use_qa:  # 执行

            ################################# 🍟 一、获取语言文件路径(.json.gz文件) 🍟 #################################
            qa_path = measurement_file_current.replace('measurements', 'vqa').replace('data/', 'drivelm/')
            # qa_path = /root/simlingo/database/simlingo_v2_2026_02_28/drivelm/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_532_route0_02_28_11_05_28/vqa/0010.json.gz
            
            
            if 'validation_' in qa_path:
                qa_exists = False   # 验证数据集默认语言文件不存在


            else:
            
            ################################# 🍟 二、下载当前帧对应的QA语言文件(.json.gz文件) 🍟 #################################

                try:
                    with gzip.open(qa_path, 'rt') as f:
                        qa = ujson.load(f)
                    qa_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    qa_exists = False
                    qa = None

            
            ################################# 🍟 三、读取当前帧对应的QA语言文件内容(.json.gz文件) 🍟 #################################

            if qa_exists:
                qas = qa['QA']  # 选择QA键
                qas = [values for values in qas.values()]           # [ perception的值, prediction的值, planning的值, behavior的值 ]
                qas = [item for sublist in qas for item in sublist] # flatten list  这就是将一个样本中所有的QA语言罗列出来了
                while True:

                    ################################# 四、随机选择一条问答 #################################
                    qa_chosen = random.choice(qas)  # 随机选一个QA
                    qa_question = qa_chosen['Q']    # 问题！！！
                    qa_answer = qa_chosen['A']      # 回答！！！

                    # 对“无信息/否定型答案”降采样
                    # 含义：
                    # 如果这个 QA 的答案属于“无目标 / 无交通灯 / 不受影响 / 无需刹车 / 无法判断”这类低信息答案，
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
                
                
                
                
                
                
                ################################# 五、若进行语言增强 #################################

                
                # 我们仅在 60% 的情况下进行数据增强，而在 40% 的情况下使用默认的问答机制。
                # 数据增强旨在提升模型对更广泛句集（sentences）的泛化能力，但我们不希望模型对增强后的句子产生过拟合。
                
                
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


                    
                    # 视觉描述,主要描述的是前视图像中的一些危险的物体,比如某辆车、行人、红绿灯(如果数据收集的时候有这些东西那么就会在"key_object_infos"键里写)
                    objects = [value['Visual_description'] for key, value in qa['key_object_infos'].items()]
                    # objects = ['black police car', 'silver SUV'] 内容可以是一个可以是多个，甚至没有（如果前视图里没有什么特别的东西的话）
                    
                    
                    ############################################# 1.替换"视觉描述"为<OBJECT> #############################################
                    q_objects = []   # 存放问题中出现的描述
                    a_objects = []   # 存放回答中出现的描述
                    for object_type in objects:
                        # 如果这些描述出现在问题里,就将问题中的描述替换为"<OBJECT>",并将描述放在q_objects里
                        if object_type in qa_question:
                            qa_question = qa_question.replace(object_type, '<OBJECT>')
                            q_objects.append(object_type)
                        # 如果这些描述出现在回答里,就将回答中的描述替换为"<OBJECT>",并将描述放在a_objects里
                        if object_type in qa_answer:
                            qa_answer = qa_answer.replace(object_type, '<OBJECT>')
                            a_objects.append(object_type)
                    
                    
                    ############################################# 2.进一步替换"位置描述"为<LOCATION> #############################################
                    q_location = ''
                    a_location = ''
                    for location in locations:
                        # 如果上述的位置描述出现在qa_question里,就将qa_question中的描述替换为"<LOCATION>,并将位置描述放在q_location里
                        if location in qa_question:
                            qa_question = qa_question.replace(location, '<LOCATION>')
                            q_location = location
                        # 如果上述的位置描述出现在qa_answer里,就将qa_answer中的描述替换为"<LOCATION>",并将位置描述放在a_location里
                        if location in qa_answer:
                            qa_answer = qa_answer.replace(location, '<LOCATION>')
                            a_location = location

                    
                    ############################################# 3.最后替换"距离描述"为<DISTANCE> #############################################    
                    q_distance = re.search(r'in (\d+) m', qa_question)                # 提取数值
                    qa_question = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_question)   # 去掉具体数值,使用<DISTANCE>代替  qa_question
                    a_distance = re.search(r'in (\d+) m', qa_answer)                  # 提取数值
                    qa_answer = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_answer)       # 去掉具体数值,使用<DISTANCE>代替  qa_answer
                    if len(q_objects)==0:
                        q_objects = ['']
                    if len(a_objects)==0:
                        a_objects = ['']
                    
                    
                    


                    ############################################# 4.是否增强问题 40%情况下不增强 60%情况下增强 #############################################
                    
                    # 40%情况下使用原QA
                    if len(q_objects) > 1 or len(a_objects) > 1 or random.random() < 0.4: 
                        qa_question = qa_question_org
                        qa_answer = qa_answer_org
                    
                    # 60%情况下使用增强版本的QA(模板中不存在就打印警告，并使用原QA)
                    else:
                        # 如果模板中有qa_question,那么就在模板中随机选择一个,并将"<OBJECT>"、"<LOCATION>"、<DISTANCE>"替换回去
                        if qa_question in self.q_augment:
                            qa_question = random.choice(self.q_augment[qa_question]).replace('<OBJECT>', q_objects[0]).replace('<LOCATION>', q_location)
                            if q_distance:
                                qa_question = qa_question.replace('<DISTANCE>', q_distance.group(1))
                        # 如果模板中不存在的话,就打印警告,并使用qa_question_org
                        else:
                            print(f"WARNING: {qa_question} not in q_augment. Using default question.")
                            qa_question = qa_question_org
                        # 如果模板中有如果模板中有qa_question,那么就在模板中随机选择一个,并将"<OBJECT>"、"<LOCATION>"、<DISTANCE>"替换回去
                        if qa_answer in self.a_augment:
                            qa_answer = random.choice(self.a_augment[qa_answer]).replace('<OBJECT>', a_objects[0]).replace('<LOCATION>', a_location)
                            if a_distance:
                                qa_answer = qa_answer.replace('<DISTANCE>', a_distance.group(1))
                        # 如果模板中不存在的话,就打印警告,并使用qa_answer_org
                        else:
                            print(f"WARNING: {qa_answer} not in a_augment. Using default answer.")
                            qa_answer = qa_answer_org

        
        
        
        
        ########################################### 🥭 target point 🥭 ###########################################


        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
        """
        target_options = 
        [
        "Target waypoint: <TARGET_POINT><TARGET_POINT>.",
        "Command: {command} in {dist_to_command} meter{next_command}.",
        "Command: {lmdrive_command}."
        ]
        
        placeholder_values = 
        {
        '<TARGET_POINT>': [[x_0, y_0], [x_1, y_1]]
        }
        """
        
        
        
        
        
        
        
        
        ########################################### 🥭 生成 prompt & answer 🥭 ###########################################

        # 决定当前帧训练成哪一类任务
        answer = ''
        prompt_random = random.random()

        # 任务一(对应第一类语言监督): commentary 任务 (触发条件:1.配置打开 2.当前样本确实有commentary 3.随机数落在commentary概率区间)
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
        
        # 任务二(对应第二类语言监督): QA任务 DriveLM 风格
        elif self.use_qa and qa_exists and prompt_random < (self.prompt_probabilities['qa'] + self.prompt_probabilities['commentary']):
            # 标准 VQA 风格
            # 也就是说此时样本不是预测轨迹，而是回答场景问答
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Q: {qa_question}"
            answer = f"A: {qa_answer}"
            self.num_sampled_per_type['qa'] += 1
        
        # 任务三: driving任务
        else:  # 执行
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

        
        
        
        
        
        
        
        
        
        
        
        
        ############################################# 🥭 前视图像 🥭 #############################################

        data = self.load_images(data, images, augment_sample=augment_sample)
        # data['rgb'] = 增强(高斯模糊、高斯噪声等)的并且裁减了原图(images)底部包含自车引擎盖部分的 新图像  [T,C,H,W]
        # data['rgb_org_size'] = 增强(高斯模糊、高斯噪声等)的但是并未进行裁减的原图(images)  [T,C,H,W]
        

        
        
        
        
        
        
        
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