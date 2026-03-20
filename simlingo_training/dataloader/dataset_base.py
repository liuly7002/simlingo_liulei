"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""
import datetime
import glob
import gzip
import os
import pickle as pkl
import random
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import ujson
from hydra.utils import get_original_cwd
from imgaug import augmenters as ia
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from tqdm import tqdm

import simlingo_training.utils.transfuser_utils as t_u
from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.utils.projection import get_camera_intrinsics, project_points

VIZ_DATA = False

class BaseDataset(Dataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Base class for the dataset.
    """

    def __init__(self, dreamer = False, evaluation = False, **cfg,):
        """
        __init__：根据配置，扫描磁盘上的数据集目录，筛选合法样本，把后续 __getitem__() 要用到的所有"索引信息"先建立好。
        """

        # 把传进来的配置项cfg,全部直接挂到self上,变成成员变量.
        # 最终就变为了self.key=value的形式
        for key, value in cfg.items():
            setattr(self, key, value)

        # 一套随机增强规则，使用它可以对图像进行增强
        # 当prob=0.5的时候,常见会启用增强规则里的3-4个增强(以脚本中的7个规则来讲的话)
        # 图像完全不增强的概率是0.78%,所以几乎每张图像都会被增强
        self.tfs = image_augmenter(prob=self.img_augmentation_prob)

        # True: 表示会对carla收集的每条 route 做质量筛选：
        #   没有 results.json.gz 的 route 不要
        #   路线评分不达标的 route 不要
        #   有严重违规的 route 不要
        # 这个是数据清洗逻辑。
        filter_infractions_per_route = True

        self.rgb_folder = 'rgb'          # 默认图像从 rgb/ 目录读取
        self.dreamer_folder = 'dreamer'  # dreamer 相关文件默认在 dreamer/ 目录里

        # 初始化样本索引容器
        self.images = []                 # 保存每个样本对应的图像路径列表,注意一个样本不一定只有一张图,因为有时间历史长度 hist_len,所以可能保存多个时刻的图像路径.
        self.boxes = []                  # 和 self.images 一一对应，保存每个时刻的 box 标注路径
        self.measurements = []           # 保存 measurement 文件夹路径,注意这里存的是目录路径,不是单个 json 文件路径
        self.sample_start = []           # 表示这个样本从哪个时间步开始,比如一个样本对应 seq=20,那么后面读取 future waypoints、future measurements 都从这个起点展开
        self.augment_exists = []         # 标记这个样本是否存在增强版本,当前代码里几乎是直接设成 True ,这个变量的设计初衷是为了支持 原图 rgb/ 增强图 rgb_augmented/ 两套图像版本
        self.alternative_trajectories = []  # 仅在 dreamer 数据中使用,用来保存 dreamer 备选轨迹文件路径

        self.temporal_measurements = []

        # 统计变量 为了最后打印数据加载情况
        total_routes = 0      # 一共看了多少route
        perfect_routes = 0    # 有多少route合格
        crashed_routes = 0    # 有多少route被过滤

        fail_reasons = {}     # 过滤原因是什么

        # 获取程序启动前的原始项目根目录,这里是/home/liulei/ll/simlingo
        repo_path = get_original_cwd()
        
        # load templates  加载模板文件,这一块是为了给后续语言任务做准备
        # commentary 模板 无论dreamer与否,这个都会加载,说明commentary模板是通用的
        template_file = f"{repo_path}/data/augmented_templates/commentary_augmented.json"
        with open(template_file, 'r') as f:
            self.templates_commentary = ujson.load(f)
    
        # load templates
        # 如果当前是 dreamer 数据集,则额外加载 dreamer 的模板
        if dreamer:
            template_file = f"{repo_path}/data/augmented_templates/dreamer.json"
            with open(template_file, 'r') as f:
                self.templates_neg = ujson.load(f)

        # 如果配置中开启了 use_lmdrive_commands(目前是开启的),就加载语言命令模板
        if self.use_lmdrive_commands:
            command_templates_file = f"{repo_path}/data/augmented_templates/lmdrive.json"
            with open(command_templates_file, 'r') as f:
                self.command_templates = ujson.load(f)


        # during eval we only want to load predefines paths
        # evaluation 模式下的特殊处理：评测时不要随便从整个数据集中抽样，而是只评测预定义好的固定样本
        # 这样做的好处是：每次评测一致、可以对比不同模型、可以针对 QA 或 commentary 任务固定 benchmark
        if evaluation:
            # 根据任务类型选择评测集文件
            if self.use_qa: # QA 任务用 evalset_vqa.json
                chosen_eval_samples_path = f'{repo_path}/data/evalset_vqa.json'
            elif self.use_commentary:  # commentary 任务用 evalset_commentary.json
                chosen_eval_samples_path = f'{repo_path}/data/evalset_commentary.json'

            # 读入并整理评测样本
            # 这部分会把 json 里的路径，统一转换成真实的 measurement 文件路径，最终得到：
            # self.all_eval_samples         一个列表，里面是所有允许进入评测的 measurement 文件路径
            # self.all_eval_samples_dict    一个字典，把样本映射到它对应的问题/答案信息
            with open(chosen_eval_samples_path, 'r') as f:
                self.chosen_eval_samples = ujson.load(f)
                self.all_eval_samples = []
                self.all_eval_samples_dict = {}
                for key, value in self.chosen_eval_samples.items():
                    if self.use_qa:
                        if 'important objects' in key:
                            continue

                        for answer in value.keys():
                            for sample in value[answer]:
                                sample = repo_path + '/' + sample.replace('vqa', 'measurements').replace('drivelm', 'data')
                                self.all_eval_samples.append(sample)
                                
                                if sample not in self.all_eval_samples_dict:
                                    self.all_eval_samples_dict[sample] = [(key, answer)]
                                else:
                                    self.all_eval_samples_dict[sample].append((key, answer))
                    else:
                        for sample in value:
                            sample = repo_path + '/' + sample.replace('commentary/simlingo', 'data/simlingo').replace('commentary', 'measurements')
                            self.all_eval_samples.append(sample)


        augment_exist = False

        # 非 dreamer 情况下，构建 prompt 类型采样配置,这里是在为多任务 prompt 采样做准备
        if not dreamer:
            if self.use_qa:  # 如果开启 QA,则加载 QA 增强模板,对 DriveLM 风格的问答做模板扩充
                as_augment_file = f'{repo_path}/data/augmented_templates/drivelm_train_augmented_v2/all_as_augmented.json'
                with open(as_augment_file, 'r') as f:
                    self.a_augment = ujson.load(f)
                qs_augment_file = f'{repo_path}/data/augmented_templates/drivelm_train_augmented_v2/all_qs_augmented.json'
                with open(qs_augment_file, 'r') as f:
                    self.q_augment = ujson.load(f)

            # 构造 prompt_probabilities
            # 初始化的权重都为1.0 prompt_probabilities = {'driving': 1.0, 'qa': 1.0, 'commentary': 1.0}
            prompt_probabilities = {
                'driving': 1.0
            }
            if self.use_qa:  # 开启
                prompt_probabilities['qa'] = 1.0
            if self.use_commentary:  # 开启
                prompt_probabilities['commentary'] = 1.0
            
            # divide by the sum to get the probabilities
            # 然后归一化
            # 情况1：只用 driving prompt_probabilities = {'driving': 1.0}
            # 情况2：driving + qa prompt_probabilities = {'driving': 0.5, 'qa': 0.5}
            # 情况3：driving + qa + commentary prompt_probabilities = {'driving': 1/3, 'qa': 1/3, 'commentary': 1/3}
            prompt_probabilities = {k: v / sum(prompt_probabilities.values()) for k, v in prompt_probabilities.items()}
            self.prompt_probabilities = prompt_probabilities
            # 记录计数器, 用于统计后面每种 prompt 类型被采样了多少次
            self.num_sampled_per_type = {k: 0 for k in prompt_probabilities.keys()}


        # bucket 过滤逻辑   作用: 如果当前 dataset 不是 bucket_name="all"，那就只保留某个 bucket 里指定的样本
        if not self.bucket_name == "all":
            with open(f"{repo_path}/" + self.bucket_path + '/buckets_paths.pkl', 'rb') as f:
                bucket_dict = pkl.load(f)   # 这个 bucket_dict 里存的是不同 bucket 对应的 run_id 列表

            bucket_run_ids = None

            # TODO: this is stupid that its manual, should change bucket names to match the saved dict with pathes
            if self.bucket_name == "all":
                pass
            # 把逻辑 bucket_name 映射到真实 bucket key
            # 这里有个“名字不统一”的问题，所以手动映射了很多名字：
            # acceleration_negative_5 → acceleration_-5
            # acceleration_positive_1 → acceleration_5
            # lateral_control_higher_5 → 若干 bucket 合并
            # recovery → 两个 recovery bucket 合并
            # 所以这里本质上是在做：
            # 配置名 → 实际 bucket 文件中的键名
            elif self.bucket_name == 'acceleration_negative_5':
                bucket_run_ids = bucket_dict['acceleration_-5']# + bucket_dict['acceleration_-20'] + bucket_dict['acceleration_-40']
            elif self.bucket_name == "acceleration_negative_1":
                bucket_run_ids = bucket_dict['acceleration_-1']
            elif self.bucket_name == "acceleration_positive_1":
                bucket_run_ids = bucket_dict['acceleration_5']
            elif self.bucket_name == "acceleration_positive_5":
                bucket_run_ids = bucket_dict['acceleration_20']# + bucket_dict['acceleration_40'] + bucket_dict['acceleration_1000000']
            elif self.bucket_name == "lateral_control_1":
                bucket_run_ids = bucket_dict['lateral_control_1']
            elif self.bucket_name == "lateral_control_1_2":
                bucket_run_ids = bucket_dict['lateral_control_1'] + bucket_dict['lateral_control_2']
            elif self.bucket_name == "lateral_control_high":
                bucket_run_ids = bucket_dict['lateral_control_2'] + bucket_dict['lateral_control_5'] + bucket_dict['lateral_control_1000000']
            elif self.bucket_name == "lateral_control_higher_5":
                bucket_run_ids = bucket_dict['lateral_control_5'] + bucket_dict['lateral_control_1000000']
            elif self.bucket_name == "recovery":
                bucket_run_ids = bucket_dict['recovery_data_small'] + bucket_dict['recovery_data_large']
            else:
                if self.bucket_name not in bucket_dict:
                    raise ValueError(f"Bucket name {self.bucket_name} not found.")
                bucket_run_ids = bucket_dict[self.bucket_name]


            # 把 bucket 中允许的 measurement 文件整理成一个查找字典
            # 最终 run_id_dict 的结构大概像：
            # {
            #     "/abs/path/to/.../measurements": ["0001.json.gz", "0002.json.gz", ...],
            #     "/abs/path/to/.../measurements": ["0010.json.gz", "0011.json.gz", ...],
            # }
            run_id_dict = {}
            if bucket_run_ids is not None:
                for run_id in bucket_run_ids:
                    run_id = run_id.replace('database/simlingo_v2_2025_01_10', self.bucket_path)
                    run_id_path = Path(run_id)
                    run_id_parent = run_id_path.parent
                    run_id_name = run_id_path.name
                    run_id_absolut = str(run_id_parent)
                    # run_id_absolut = f"{repo_path}/{str(run_id_parent)}"
                    if run_id_absolut not in run_id_dict:
                        run_id_dict[run_id_absolut] = [run_id_name]
                    else:
                        run_id_dict[run_id_absolut].append(run_id_name)

        # 在数据根目录下，把所有 route 目录都找出来
        route_dirs = glob.glob(f"{repo_path}/" + self.data_path + '/data/simlingo/*/*/*/Town*')
        print(f'[liulei]Found {len(route_dirs)} routes in {repo_path + self.data_path} 这里的{len(route_dirs)}代表的是我们在carla上收集了{len(route_dirs)}条路线的数据')

        # lb1_split 代表 old towns 数据
        # use_old_towns=False：排除 old towns
        # use_only_old_towns=True：只用 old towns
        # 或者当前 bucket_name 就是 "old_towns"，也只保留 old towns
        if not self.use_old_towns:
            route_dirs = [route_dir for route_dir in route_dirs if 'lb1_split' not in route_dir]
            print(f'Found {len(route_dirs)} routes in {repo_path + self.data_path} after filtering out old towns')
        elif self.use_only_old_towns or self.bucket_name == "old_towns":
            route_dirs = [route_dir for route_dir in route_dirs if 'lb1_split' in route_dir]
            print(f'Found {len(route_dirs)} routes in {repo_path + self.data_path} after filtering out non old towns')
        

        # 打乱路线顺序,这个打乱只是在 route 级别随机化，避免总是按固定目录顺序加载
        random.shuffle(route_dirs)
        split_percentage = 0.99
        # 情况A：此时采用官方 town 划分,train 用 routes_training,val 用 routes_validation
        if dreamer or not self.use_town13:  # use_town13=True表示把 Town13 也纳入整体训练池/验证池中，而不再保留官方独立验证 town 的严格划分
            # split the data into official training(Town12 and old Towns) and validation set (Town13)
            if self.split == "train":
                print("Using Town12 for training")
                route_dirs = [route_dir for route_dir in route_dirs if 'routes_training' in route_dir]
            elif self.split == "val":
                print("Using Town13 for validation")
                route_dirs = [route_dir for route_dir in route_dirs if 'routes_validation' in route_dir]
                route_dirs = route_dirs[:int(0.02 * len(route_dirs))]  # 注意验证集这里还额外只取了前 2%
        else:
            # use all towns
            # 这里不是按 town 名字切，而是对所有 route 随机打乱后按比例切分
            # 前 99% 做 train  后 1% 做 val
            if self.split == "train":
                # print(f"[Debug]变化前 route_dirs = {route_dirs}")
                # route_dirs = ['/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_02_28_10_56_43', '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43', '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/Town12_Rep0_532_route0_02_28_11_05_28']
                print(f"一共有 {len(route_dirs)} 条路线的数据,但是【训练】的时候我们值选择了所有路线的前99%作为训练集.", end='')
                route_dirs = route_dirs[:int(split_percentage * len(route_dirs))]  # 取出所有路线的前99%个路线作为训练集
                # print(f"[Debug]变化后 route_dirs = {route_dirs}")
                # route_dirs = ['/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_02_28_10_56_43', '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43']
                print(f'所以【训练】使用了 {len(route_dirs)} 条路线的数据作为训练. ')

            elif self.split == "val":
                print(f"一共有 {len(route_dirs)} 条路线的数据,但是【验证】的时候我们值选择了所有路线的后1%作为验证集. ", end='')
                route_dirs = route_dirs[int(split_percentage * len(route_dirs)):]  # 取出所有路线的最后1%作为验证集
                print(f'所以【验证】使用了 {len(route_dirs)} 条路线的数据作为验证. ')

        # print(f"[Debug]这里 self.split = {self.split}")



        total_routes += len(route_dirs)  # 一共看了多少route
        
        # route_dirs = route_dirs[:100]
        # print(f'Use {len(route_dirs)} routes.')

        # 这里开始真正的处理每条route
        start_route = 0
        for sub_root in tqdm(route_dirs, file=sys.stdout):
            start_route+=1
            print(f"开始处理 ({start_route} / {len(route_dirs)}) 路线. 该路线来自 {sub_root}.", end="")
            # route_dirs = [
            # '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_02_28_10_56_43',
            # '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43'
            # ]
            route_dir = sub_root # + '/' + route

            # 如果是 dreamer 数据集，就要求 route 对应的 dreamer 目录必须存在，否则整条 route 不要
            if dreamer:
                dreamer_dir = route_dir.replace('data/', f'{self.dreamer_folder}/')
                if not os.path.exists(dreamer_dir):
                    continue

            # 按 route 做结果文件质量过滤
            if filter_infractions_per_route:
                # 没有 results.json.gz  -> 整条 route 不要
                if not os.path.isfile(route_dir + '/results.json.gz'):
                    print(f"!!! 没有 results.json.gz  -> 整条 route 不要")
                    total_routes += 1
                    crashed_routes += 1
                    if "no_results.json" not in fail_reasons:
                        fail_reasons["no_results.json"] = 1
                    else:
                        fail_reasons["no_results.json"] += 1
                    continue

                # 读取结果文件results.json.gz
                with gzip.open(route_dir + '/results.json.gz', 'rt') as f:
                    total_routes += 1
                    try:
                        results_route = ujson.load(f)
                    except Exception as e:
                        print(f"Error in {route_dir}")
                        print(e)
                        if "results.json_load_error" not in fail_reasons:
                            fail_reasons["results.json_load_error"] = 1
                        else:
                            fail_reasons["results.json_load_error"] += 1
                        continue

                # 评分过滤
                # 如果综合评分不是满分，则进入额外判定
                if results_route['scores']['score_composed'] < 100.0:  # we also count imperfect runs as failed (except minspeedinfractions)
                    # 允许保留的条件 cond1-路线完成度还不错（大于 94%） cond2-违规只来自"低速违规"\"出路线车道"
                    cond1 = results_route['scores']['score_route'] > 94.0  # we allow 6% of the route score to be missing
                    cond2 = results_route['num_infractions'] == (len(results_route['infractions']['min_speed_infractions']) + len(results_route['infractions']['outside_route_lanes']))
                    # 如果满足这两个条件，就放过；否则整条 route 过滤掉
                    # 这说明清洗数据的原则是:允许一些不太严重的问题，但不允许真正撞车、重大违规、严重失败的路线进入训练
                    if not (cond1 and cond2):  # if the only problem is minspeedinfractions, keep it
                        crashed_routes += 1
                        if "route_crashed" not in fail_reasons:
                            fail_reasons["route_crashed"] = 1
                        else:
                            fail_reasons["route_crashed"] += 1
                        continue

            perfect_routes += 1

            # if not os.path.exists(route_dir + f'/{self.rgb_folder}'):
            #     if "no_rgb_folder" not in fail_reasons:
            #         fail_reasons["no_rgb_folder"] = 1
            #     else:
            #         fail_reasons["no_rgb_folder"] += 1
            #     continue

            # route 合格后，就开始在 route 内逐帧切样本
            num_seq = len(os.listdir(route_dir + f'/{self.rgb_folder}'))
            print(f"该路线总计有 {num_seq} 个数据, 实际上这里的 {num_seq} 是根据 /rgb 文件夹下的图像的数量计算的."
                  f"去除最开始共计 {self.skip_first_n_frames} 帧数据(因为刚开始收集数据不稳定,所以我们去除了), 我们使用的是下标为 [{self.skip_first_n_frames},{num_seq - self.pred_len - self.hist_len - 1}] 之间的数据. ")

            # 从 skip_first_n_frames(10) 开始，到 num_seq - pred_len - hist_len - 1 结束
            # 1）为什么从 skip_first_n_frames 开始？
            # 因为前面几帧可能：车辆还没稳定、传感器/控制初始化阶段不可靠、刚开局信息不完整，所以前几帧跳过。
            # 2）为什么结束到 num_seq - pred_len - hist_len - 1？
            # 因为一个样本不仅需要当前帧，还需要：历史帧 hist_len、未来帧 pred_len，所以末尾不够长的序列不能作为样本，否则未来 measurement / future waypoints 不够。
            for seq in range(self.skip_first_n_frames, num_seq - self.pred_len - self.hist_len - 1):
                # 当前样本先构造一些临时变量 每个 seq 都会构造一个样本
                image = []
                box = []
                measurement = []
                augment_exist = False

                # 当前 measurement 文件
                # 注意这个索引不是 seq，而是"seq + hist_len - 1",因为这个样本的“当前时刻”定义为历史序列的最后一帧
                # 举例：hist_len = 3,seq = 10,那么图像历史帧会是：0010,0011,0012,此时当前 measurement 应该对应 0012，也就是：10 + 3 - 1 = 12
                measurement_file = route_dir + '/measurements' + f'/{(seq + self.hist_len-1):04}.json.gz'

                # evaluation 样本过滤  只保留评测集规定的 measurement
                if evaluation and measurement_file not in self.all_eval_samples:
                    continue
                
                # 如果当前 measurement 对应的 dreamer 文件不存在，就跳过这个样本
                if dreamer:
                    dreamer_file_path = measurement_file.replace('measurements', f'{self.dreamer_folder}').replace('data/', f'{self.dreamer_folder}/')
                    if not os.path.exists(dreamer_file_path):
                        continue
                 
                # 如果当前 bucket 不是 all 则会检查当前 measurement 是否属于 bucket
                if self.bucket_name is not None and self.bucket_name != "all":
                    # 判断逻辑
                    measurement_file_path = Path(measurement_file)
                    # print("\n===== DEBUG START =====")
                    # print("measurement_file:", measurement_file)
                    # print("parent:", str(measurement_file_path.parent))
                    # print("bucket_name:", self.bucket_name)
                    # print("run_id_dict keys sample:", list(run_id_dict.keys())[:5])
                    # print("parent in run_id_dict:", str(measurement_file_path.parent) in run_id_dict)
                    # measurement_file: /home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_02_28_10_56_43/measurements/0154.json.gz
                    # parent:           /home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_02_28_10_56_43/measurements
                    # bucket_name:      leading_object_vehicle
                    # run_id_dict keys sample: ['/home/liulei/ll/simlingo//home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_493_route0_02_28_11_00_43/measurements', '/home/liulei/ll/simlingo//home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/Town12_Rep0_1553_route0_02_28_10_56_43/measurements']
                    # parent in run_id_dict: False





                    # 第一层：所在文件夹在不在 bucket 中，不在就跳过
                    # 第二层：该文件名在不在 bucket 中，不在就跳过
                    # 这就能保证，当前 dataset 真正只由 bucket 指定的 measurement 样本构成
                    if str(measurement_file_path.parent) in run_id_dict:
                        if measurement_file_path.name not in run_id_dict[str(measurement_file_path.parent)]:
                            if "measurement_file_not_in_bucket" not in fail_reasons:
                                fail_reasons["measurement_file_not_in_bucket"] = 1
                            else:
                                fail_reasons["measurement_file_not_in_bucket"] += 1
                            continue
                    else:
                        if "measurement_folder_not_in_bucket" not in fail_reasons:
                            fail_reasons["measurement_folder_not_in_bucket"] = 1
                        else:
                            fail_reasons["measurement_folder_not_in_bucket"] += 1
                        continue

                # Loads the current (and past) frames (if seq_len > 1)
                # 构造历史图像路径和 box 路径
                skip = False
                augment_exist = True
                for idx in range(self.hist_len):
                    image.append(route_dir +  f'/{self.rgb_folder}' + (f'/{(seq + idx):04}.jpg'))
                    box.append(route_dir + '/boxes' + (f'/{(seq + idx):04}.json.gz'))

                if skip:
                    if "file_not_found" not in fail_reasons:
                        fail_reasons["file_not_found"] = 1
                    else:
                        fail_reasons["file_not_found"] += 1
                    continue

                # 不是保存当前具体文件，而是保存 measurement 文件夹路径
                measurement.append(route_dir + '/measurements')

                # 把样本注册进整个 dataset，到这里，一个样本就正式加入数据集了
                self.images.append(image)
                self.boxes.append(box)
                self.measurements.append(measurement)
                self.sample_start.append(seq)
                self.augment_exists.append(augment_exist)
                if dreamer:
                    self.alternative_trajectories.append(dreamer_file_path)

        # There is a complex "memory leak"/performance issue when using Python
        # objects like lists in a Dataloader that is loaded with
        # multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects
        # because they only have 1 refcount.
        self.images = np.array(self.images).astype(np.string_)
        self.boxes = np.array(self.boxes).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        if dreamer:
            self.alternative_trajectories = np.array(self.alternative_trajectories).astype(np.string_)

        self.sample_start = np.array(self.sample_start)
        # if rank == 0:
        print(f'[{self.split} samples]: Loading {len(self.images)} images from {self.data_path} for bucket {self.bucket_name}')
        print('Total amount of routes:', total_routes)
        print('Crashed routes:', crashed_routes)
        print('Perfect routes:', perfect_routes)
        print('Fail reasons:', fail_reasons)

    def __len__(self):
        """Returns the length of the dataset. """
        return self.images.shape[0]
    

    def load_current_and_future_measurements(self, measurements, sample_start):
        loaded_measurements = []

        ######################################################
        ######## load current and future measurements ########
        ######################################################

        # Since we load measurements for future time steps, we load and store them separately
        for i in range(self.hist_len):
            measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

            with gzip.open(measurement_file, 'rt') as f1:
                measurements_i = ujson.load(f1)
            loaded_measurements.append(measurements_i)

        end = self.pred_len + self.hist_len
        start = self.hist_len

        for i in range(start, end):
            try:
                measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')

                with gzip.open(measurement_file, 'rt') as f1:
                    measurements_i = ujson.load(f1)
                loaded_measurements.append(measurements_i)
            except FileNotFoundError:
                # If the file is not found, we just use the last available measurement
                print(f"File not found: {measurement_file}")
                loaded_measurements.append(loaded_measurements[-1])
        current_measurement = loaded_measurements[self.hist_len - 1]
        measurement_file_current = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + start-1):04}.json.gz')
        return loaded_measurements, current_measurement, measurement_file_current

    def load_waypoints(self, data, loaded_measurements, aug_translation=0.0, aug_rotation=0.0):

        waypoints = self.get_waypoints(loaded_measurements[self.hist_len - 1:],
                                                                        y_augmentation=aug_translation,
                                                                        yaw_augmentation=aug_rotation)
        data['waypoints'] = np.array(waypoints[1:-1])

        waypoints_org = self.get_waypoints(loaded_measurements[self.hist_len - 1:],
                                                                        y_augmentation=0,
                                                                        yaw_augmentation=0)
        data['waypoints_org'] = np.array(waypoints_org[1:-1])

        # 1D waypoints: only consider distance between waypoints
        waypoints_1d = [np.linalg.norm(waypoints_org[i+1] - waypoints_org[i]) for i in range(len(waypoints_org)-1)]
        # cumsum to get the distance from the start
        waypoints_1d = np.cumsum(waypoints_1d)
        waypoints_1d = [[x, 0] for x in waypoints_1d]
        data['waypoints_1d'] = np.array(waypoints_1d[:-1]).reshape(-1, 2)

        waypoints = [np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]) for x, y in waypoints]
        data['ego_waypoints'] = np.array(waypoints[:-1])
        
        waypoints_org = [np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]) for x, y in waypoints_org]
        data['ego_waypoints_org'] = np.array(waypoints_org[:-1])

        return data
    
    def load_route(self, data, current_measurement, aug_translation=0.0, aug_rotation=0.0):
        route = current_measurement['route_original']
        route = self.augment_route(route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        route_adjusted = np.array(current_measurement['route'])
        route_adjusted_org = self.augment_route(route_adjusted, y_augmentation=0, yaw_augmentation=0)
        route_adjusted = self.augment_route(route_adjusted, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        if len(route) < self.num_route_points:
            num_missing = self.num_route_points - len(route)
            route = np.array(route)
            # Fill the empty spots by repeating the last point.
            route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
        else:
            route = np.array(route[:self.num_route_points])
            
        route_adjusted = self.equal_spacing_route(route_adjusted)
        route_adjusted_org = self.equal_spacing_route(route_adjusted_org)
        route = self.equal_spacing_route(route)
        
        data['route'] = route
        data['route_adjusted_org'] = route_adjusted_org
        data['route_adjusted'] = route_adjusted

        return data
    
    def load_images(self, data, images, augment_sample=False):
        loaded_images = []
        loaded_images_org_size = []
        for i in range(self.hist_len):
            images_i = None
            images_path = str(images[i], encoding='utf-8')
            if augment_sample:
                images_path = images_path.replace('rgb', 'rgb_augmented')

            if not os.path.isfile(images_path):
                print(f"File not found: {images_path}")
                raise FileNotFoundError

            images_i = cv2.imread(images_path, cv2.IMREAD_COLOR)
            images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)

            if self.img_augmentation: # and random.random() <= self.img_augmentation_prob:
                images_i = self.tfs(image=images_i)
            
            image_org = images_i.copy()
            if self.cut_bottom_quarter or self.img_shift_augmentation:
                # to remove the bonnet whih is important for the shifted camera augmentation
                # we need to remove 4.8/16 of the bottomf of the image (empirical value)
                images_i = images_i[:int(images_i.shape[0] - (images_i.shape[0] * 4.8) // 16), :, :]

            loaded_images.append(images_i)
            loaded_images_org_size.append(image_org)
        
        processed_image = np.asarray(loaded_images)
        processed_image_org_size = np.asarray(loaded_images_org_size)

        # we want [T, N, C, H, W], T is the number of temporal frames, N is the number of cam views, C is the number of channels, H is the height and W is the width
        processed_image = np.transpose(processed_image, (0, 3, 1, 2)) # (T, C, H, W)
        processed_image_org_size = np.transpose(processed_image_org_size, (0, 3, 1, 2)) # (T, C, H, W)

        data['rgb'] = processed_image
        data['rgb_org_size'] = processed_image_org_size

        return data

    def get_navigational_conditioning(self, data, current_measurement, target_point, next_target_point):
        placeholder_values = {}
        target_options = []
                
        tp = [target_point, next_target_point]
        tp = np.array(tp)
        data['map_route'] = tp
        data['target_points'] = tp
        target_point1_round = np.round(data['target_points'][0], 2).tolist()
        target_point2_round = np.round(data['target_points'][1], 2).tolist()

        if 'target_point' in self.route_as:
            if 'target_point_language' in self.route_as:
                target_options.append(f"Target waypoint: 1:{target_point1_round} 2:{target_point2_round}")
            else:
                target_options.append(f"Target waypoint: <TARGET_POINT><TARGET_POINT>.")
                placeholder_values = {'<TARGET_POINT>': data['target_points']}
        if 'command' in self.route_as:
            # get distance from target_point
            dist_to_command = np.linalg.norm(target_point)
            dist_to_command = int(dist_to_command)
            map_command = {
                1: 'go left at the next intersection',
                2: 'go right at the next intersection',
                3: 'go straight at the next intersection',
                4: 'follow the road',
                5: 'do a lane change to the left',
                6: 'do a lane change to the right',        
            }
            command_template_mappings = {
                1: [0, 2, 4, 7],
                2: [1, 3, 5, 8],
                3: [6, 9],
                4: [38, 40, 42, 43, 44, 45],
                5: [34, 36],
                6: [35, 37],
            }
            command = map_command[current_measurement["command"]]
            next_command = map_command[current_measurement["next_command"]]
            if command != next_command:
                next_command = f' then {next_command}'
            else:
                next_command = ''
            if current_measurement["command"] == 4:
                command_str = f'Command: {command}{next_command}.'
            else:
                command_str = f'Command: {command} in {dist_to_command} meter{next_command}.'
            target_options.append(command_str)
            
            if self.use_lmdrive_commands:
                lmdrive_index = random.choice(command_template_mappings[current_measurement["command"]])
                lmdrive_command = random.choice(self.command_templates[str(lmdrive_index)])
                lmdrive_command = lmdrive_command.replace('[x]', str(dist_to_command))
                lm_command = f'Command: {lmdrive_command}.'
                target_options.append(lm_command)
        
        return target_options, placeholder_values

    def equal_spacing_route(self, points):
        route = np.concatenate((np.zeros_like(points[:1]),  points)) # Add 0 to front
        shift = np.roll(route, 1, axis=0) # Shift by 1
        shift[0] = shift[1] # Set wraparound value to 0

        dists = np.linalg.norm(route-shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists))*1e-4 # Prevents dists not being strictly increasing

        x = np.arange(0, 20, 1)
        interp_points = np.array([np.interp(x, dists, route[:, 0]), np.interp(x, dists, route[:, 1])]).T

        return interp_points
    
    def visualise_cameras(
        self,
        batch: DatasetOutput,
        language, route, waypoints,
        options,
        name: str = "img",
        prompt=None,
        answer=None,
    ) -> np.ndarray:
        
        fov = 110

        img_front_np = batch.image_ff_org_size #[0, ...]
        img_front_np = img_front_np.transpose(0, 2, 3, 1)
        # two patches..dim 1 of img_front is 2 (left and right patches)
        # concatenate them to get a single image
        # img_front_1 = img_front[:, 0, ...]
        # img_front_2 = img_front[:, 1, ...]
        # img_front_torch = torch.cat((img_front_1, img_front_2), dim=3)

        all_images = [Image.fromarray((img_front_np[i])) for i in range(1)]

        # all_images = [Image.fromarray((img_front_torch[i].cpu().permute(1, 2, 0).numpy())) for i in range(1)]
        all_draws = [ImageDraw.Draw(image) for image in all_images]
        
        # black image to be concatenated to the bottom of the image
        img_width = all_images[0].size[0]
        text_box = [Image.new("RGB", (img_width, 200), "black") for _ in range(1)]
        text_draw = [ImageDraw.Draw(image) for image in text_box]
        
        W=all_images[0].size[0]
        H=all_images[0].size[1]
        camera_intrinsics = np.asarray(get_camera_intrinsics(W,H,fov))

        for i in range(1):
            gt_waypoints_img_coords = project_points(batch.waypoints, camera_intrinsics)
            for points_2d in gt_waypoints_img_coords:
                all_draws[i].ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(0, 255, 0, 255))

            if route is not None:
                pred_route_img_coords = project_points(route, camera_intrinsics)
                for points_2d in pred_route_img_coords:
                    all_draws[i].ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(255, 0, 0, 255))

            if language is not None:
                y_curr = 10
                
                # write the language to the bottom of the image
                # all_draws[i].rectangle([0, H-60, W, H], fill=(0, 0, 0, 255))
                # all_draws[i].text((10, H-40), f"Pred: {language[i]}", fill=(255, 255, 255, 255))
                text_draw[i].text((10, y_curr), f"Commentary: {language}", fill=(255, 255, 255, 255))
            if prompt is not None:
                text_draw[i].text((10, 30), f"Prompt: {prompt}", fill=(255, 255, 255, 255))
            if answer is not None:
                text_draw[i].text((10, 50), f"Answer: {answer}", fill=(255, 255, 255, 255))
                
        # concat text box to the bottom of the image
        
        # duplicate all_images len(all_negatives) times, deepcopy!
        if options is not None:
            all_all_images = [None for _ in range(len(options))]
            all_blacks = [None for _ in range(len(options))]
            for i, option in enumerate(options):
                wp_altern = option['waypoints']
                route_altern = option['route']
                if isinstance(route_altern, str) and route_altern == 'org':
                    route_altern = route
                if 'dreamer_instruction' in option:
                    language = option['dreamer_instruction'][0] if isinstance(option['dreamer_instruction'], list) else option['dreamer_instruction']
                    answer = option['dreamer_answer_safety'][0] if isinstance(option['dreamer_answer_safety'], list) else option['dreamer_answer_safety']
                else:
                    language = None
                    answer = None
                img = all_images[0].copy()
                draw = ImageDraw.Draw(img)
                img_black = text_box[0].copy()
                draw_black = ImageDraw.Draw(img_black)
                gt_waypoints_img_coords = project_points(wp_altern, camera_intrinsics)
                for points_2d in gt_waypoints_img_coords:
                    draw.ellipse((points_2d[0]-3, points_2d[1]-3, points_2d[0]+3, points_2d[1]+3), fill=(0, 55, 0, 255))
                if route_altern is not None:
                    pred_route_img_coords = project_points(route_altern, camera_intrinsics)
                    for points_2d in pred_route_img_coords:
                        draw.ellipse((points_2d[0]-2, points_2d[1]-2, points_2d[0]+2, points_2d[1]+2), fill=(55, 0, 0, 255))
                if language is not None:
                    draw_black.text((10, 80), f"Alternative Traj: {language}", fill=(255, 255, 255, 255))
                    draw_black.text((10, 100), f"Alternative Traj: {answer}", fill=(255, 255, 255, 255))
                    
                all_all_images[i] = img
                all_blacks[i] = img_black
            
        all_images = [Image.fromarray(np.concatenate([np.array(image), np.array(text)], axis=0)) for image, text in zip(all_images, text_box)]
        if options is not None:
            all_images.extend([Image.fromarray(np.concatenate([np.array(image), np.array(text)], axis=0)) for image, text in zip(all_all_images, all_blacks)])
        
        # concat all images
        viz_image_np = np.concatenate([np.array(image) for image in all_images], axis=0)
        viz_image = Image.fromarray(viz_image_np)
        
        # get ucrrent work dir
        current_dir = os.getcwd()
        
        Path("viz_images").mkdir(parents=True, exist_ok=True)
        # save the image
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_image.save(f"viz_images/{name}_{time}.png")

        return viz_image_np
    

    def get_indices_speed_angle(self, target_speed, brake):
        target_speeds = [0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20, 24]  # v4 target speeds (0.72*speed limits) plus extra classes for obstacle scenarios and intersecions

        target_speed_bins = [x+0.001 for x in target_speeds[1:]]  
        target_speed_index = np.digitize(x=target_speed, bins=target_speed_bins)

        # Define the first index to be the brake action
        if brake:
            target_speed_index = 0
        else:
            target_speed_index += 1

        return target_speed_index

    def augment_route(self, route, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                    np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0, y_augmentation]])
        route_aug = (rotation_matrix.T @ (route - translation).T).T
        return route_aug

    def augment_target_point(self, target_point, y_augmentation=0.0, yaw_augmentation=0.0):
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0], [y_augmentation]])
        pos = np.expand_dims(target_point, axis=1)
        target_point_aug = rotation_matrix.T @ (pos - translation)
        return np.squeeze(target_point_aug)

    def parse_bounding_boxes(self, boxes, future_boxes=None, y_augmentation=0.0, yaw_augmentation=0):

        bboxes = []
        for current_box in boxes:
            # Ego car is always at the origin. We don't predict it.
            if current_box['class'] == 'ego_car':
                continue

            if 'position' not in current_box or 'extent' not in current_box:
                continue

            bbox, height = self.get_bbox_label(current_box, y_augmentation, yaw_augmentation)

            if current_box['class'] == 'traffic_light':
                # Only use/detect boxes that are red and affect the ego vehicle
                if not current_box['affects_ego']:
                    continue

            if current_box['class'] == 'stop_sign':
                # Don't detect cleared stop signs.
                if not current_box['affects_ego']:
                    continue

            bboxes.append(bbox)
        return bboxes

    def get_bbox_label(self, bbox_dict, y_augmentation=0.0, yaw_augmentation=0):
        # augmentation
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                np.cos(aug_yaw_rad)]])

        position = np.array([[bbox_dict['position'][0]], [bbox_dict['position'][1]]])
        translation = np.array([[0.0], [y_augmentation]])

        position_aug = rotation_matrix.T @ (position - translation)

        x, y = position_aug[:2, 0]
        # center_x, center_y, w, h, yaw
        bbox = np.array([x, y, bbox_dict['extent'][0], bbox_dict['extent'][1], 0, 0, 0, 0, 0])
        bbox[4] = t_u.normalize_angle(bbox_dict['yaw'] - aug_yaw_rad)

        if bbox_dict['class'] == 'car':
            bbox[5] = bbox_dict['speed']
            bbox[6] = bbox_dict['brake']
            bbox[7] = 0
        elif bbox_dict['class'] == 'walker':
            bbox[5] = bbox_dict['speed']
            bbox[7] = 1
        elif bbox_dict['class'] == 'traffic_light':
            bbox[7] = 2
            if bbox_dict['state'] == 'Green':
                bbox[8] = 0
            elif bbox_dict['state'] == 'Red' or bbox_dict['state'] == 'Yellow':
                bbox[8] = 1
            else:
                bbox[8] = 2
        elif bbox_dict['class'] == 'stop_sign':
            bbox[7] = 3

        else:
            bbox = np.zeros(9)
        return bbox, bbox_dict['position'][2]

    def get_route_image(self, route, target_point):
        route_img = np.zeros((64, 64, 3), dtype=np.uint8)
        route_new = np.array(route, dtype=np.float32)
        route_new[:, 0] = -route_new[:, 0]*2 + 63
        route_new[:, 1] = route_new[:, 1]*2 + 32
        route_new = route_new.clip(0, 63)
        route_new = route_new.astype(np.int32)
        route_img[route_new[:, 0], route_new[:, 1], :] = 255

        # # target point as red
        # target_point = np.array(target_point, dtype=np.float32)
        # target_point[0] = -target_point[0]*2 + 63
        # target_point[1] = target_point[1]*2 + 32
        # target_point = target_point.astype(np.int32)
        # # target_point = target_point.clip(0, 63)
        # route_img[target_point[0], target_point[1], 0] = 255

        # save route_img
        # cv2.imwrite('/home/wayve/katrinrenz/coding/WayveCode/route_img.png', route_img)

        return route_img

    def get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
        """transform waypoints to be origin at ego_matrix"""
        origin = measurements[0]
        origin_matrix = np.array(origin['ego_matrix'])[:3]
        origin_translation = origin_matrix[:, 3:4]
        origin_rotation = origin_matrix[:, :3]

        waypoints = []
        for index in range(len(measurements)):
            waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
            waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
            # Drop the height dimension because we predict waypoints in BEV
            waypoints.append(waypoint_ego_frame[:2, 0])

        # Data augmentation
        waypoints_aug = []
        aug_yaw_rad = np.deg2rad(yaw_augmentation)
        rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                                                                                                            np.cos(aug_yaw_rad)]])

        translation = np.array([[0.0], [y_augmentation]])
        for waypoint in waypoints:
            pos = np.expand_dims(waypoint, axis=1)
            waypoint_aug = rotation_matrix.T @ (pos - translation)
            waypoints_aug.append(np.squeeze(waypoint_aug))

        return waypoints_aug

def image_augmenter(prob=0.2, cutout=False):
    """
    prob = 0.5 的含义：
        下面列出的每一种增强操作，都有 50% 的概率 被启用。
        注意，是 每一种增强各自独立地以 0.5 概率决定是否执行，不是“整条流水线总共 50% 概率增强一次”。
    ia.Sometimes(0.5, 某个增强操作)
        意思是：
        以 50% 概率执行这个增强，否则跳过。
        所以最终一张图可能出现的情况是：
        什么都不增强
        只做模糊
        只做噪声
        同时做模糊+噪声+对比度变化
        做 4~5 种增强叠加
        甚至几乎全做
        因为每个增强是独立随机采样的。
    """

    augmentations = [
        ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),  # 高斯模糊
        ia.Sometimes(prob, ia.AdditiveGaussianNoise(loc=0, scale=(0., 0.05 * 255), per_channel=0.5)),  # 加性高斯噪声
        ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),  # Strong 随机像素丢弃
        ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),  # 亮度缩放
        ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),  # 线性对比度变化
        ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),                            # 灰度化
        ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),  # 弹性形变
    ]

    if cutout:  # 不生效，这是一种局部遮挡的图像增强方式
        augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False)))

    # 关键语句：把上面这些增强操作组成一个序列执行器，但每次执行时，增强操作的顺序是随机打乱的。
    # 如果固定顺序，比如永远：Blur → Noise → Dropout → Brightness → Contrast → Grayscale → Elastic
    # 那么模型总是看到同一种增强链路，可能会对这个顺序形成偏置。
    augmenter = ia.Sequential(augmentations, random_order=True)

    return augmenter  # 返回的是一个增强器对象,后续可以将其作为一套随机增强规则来使用


def get_camera_intrinsics(w, h, fov):
    """
    Get camera intrinsics matrix from width, height and fov.
    Returns:
        K: A float32 tensor of shape ``[3, 3]`` containing the intrinsic calibration matrices for
            the carla camera.
    """

    # print(f"[CAMERA MATRIX] Load camera intrinsics for TF++ default camera with w: {w}, h: {h}, fov: {fov}")

    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    K = torch.tensor(K, dtype=torch.float32)
    return K

def get_camera_extrinsics():
    """
    Get camera extrinsics matrix for the carla camera.
    extrinsics: A float32 tensor of shape ``[4, 4]`` containing the extrinic calibration matrix for
            the carla camera. The extriniscs are specified as homogeneous matrices of the form ``[R t; 0 1]``
    """

    # camera_pos = [-1.5, 0.0, 2.0]  # x, y, z mounting position of the camera
    # camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree

    # print("[CAMERA MATRIX] Load camera extrinsics for TF++ default camera with x: -1.5, y: 0.0, z: 2.0, roll: 0.0, pitch: 0.0, yaw: 0.0")
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    extrinsics[3, 3] = 1.0
    extrinsics[:3, :3] = np.eye(3)
    extrinsics[:3, 3] = [-1.5, 0.0, 2.0]

    extrinsics = torch.tensor(extrinsics, dtype=torch.float32)

    return extrinsics

def get_camera_distortion():
    """
    Get camera distortion matrix for the carla camera.
    distortion: A float32 tensor of shape ``[14 + 1]`` containing the camera distortion co-efficients
            ``[k0, k1, ..., k13, d]`` where ``k0`` to ``k13`` are distortion co-efficients and d specifies the
            distortion model as defined by the DistortionType enum in camera_info.hpp
    """

    print("[CAMERA MATRIX] Load camera distortion for TF++ default camera. No distortion.")
    distortion = np.zeros(14 + 1, dtype=np.float32)
    distortion[-1] = 0.0
    distortion = torch.tensor(distortion, dtype=torch.float32)

    return distortion




if __name__ == "__main__":
    from hydra import compose, initialize
    from simlingo_training.config import TrainConfig
    
    # set all seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    

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
        data = dataset[i]
        print(data)
        if i == 100:
            break