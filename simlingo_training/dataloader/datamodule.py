# Standard library imports
import itertools
from typing import List

# Third-party imports
import hydra
import line_profiler
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# Local/project specific imports
# from simlingo_training.dataloader.dataset_driving import Data_Driving # is called directly by hydra.utils.instantiate, keeping here to make it easier to find
# from simlingo_training.dataloader.dataset_dreamer import Data_Dreamer # is called directly by hydra.utils.instantiate, keeping here to make it easier to find
from simlingo_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel, LanguageLabel
from simlingo_training.utils.internvl2_utils import preprocess_image_batch, get_custom_chat_template, get_num_image_tokens_per_patch
from simlingo_training.utils.projection import get_camera_intrinsics, get_camera_extrinsics

def encode_uint8(strings: List[str], common_length: int) -> torch.Tensor:
    max_len = max(len(s) for s in strings)
    assert max_len <= common_length, f"String is too long: {max_len} > {common_length}"
    padded_strings = [s.ljust(common_length, '\0') for s in strings]
    return torch.tensor([bytearray(s, 'utf-8') for s in padded_strings], dtype=torch.uint8)


class DataModule(LightningDataModule):
    def __init__(
        self,
        base_dataset,
        processor,
        predict=False,
        **cfg,
    ):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)
            
        for key, value in base_dataset.items():
            setattr(self, key, value)
            
        self.cfg = cfg
        self.base_dataset = base_dataset
        self.processor = processor
        self.predict = predict   # 是否进行推理
        
        self.printed = False

        self.NUM_IMAGE_PATCHES = 2             # 一张原始输入图像会被拆成2个patch
        self.IMAGES_TO_CONSIDER = ['image_ff'] # front-forward image, other images are not supported
        # taken from:
        # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/train/constants.py#L7
        # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L294
        self.IMG_START_TOKEN='<img>'
        self.IMG_END_TOKEN='</img>'
        self.IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'  # 图像token占位符
        # <img><IMG_CONTEXT><IMG_CONTEXT>...</img>

        self.num_image_tokens_per_patch = get_num_image_tokens_per_patch(self.encoder_variant)  # self.encoder_variant=OpenGVLab/InternVL2-1B
        self.num_image_tokens_total = self.num_image_tokens_per_patch * self.NUM_IMAGE_PATCHES
            
        #
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor
        
        # TODO: not needed anymore?
        # 给 tokenizer 增加任务专用标记，让语言模型能识别自动驾驶结构化信息。
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>','<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tokenizer.padding_side = "left"  # padding 加在左边，而不是右边。

    def setup(self, stage=None):  # stage表示当前运行阶段 Lightning 会传入不同的值：fit-训练 validate-验证 test-测试 predict-推理

        if not self.predict:           # 如果不是预测模式: 构造train/val/test数据集
            # 1. 初始化容器
            self.val_datasets = []     # 用来存多个验证集 因为后面每个"数据源"都会生成一个 val dataset,最后再拼起来.
            sum_sample_weights = 1.0   # 临时变量,用来做权重归一化时的分母
            bucket_list = []           # 存所有bucket的名字 例如['all', 'turn_left', 'all_dreamer', 'curve_dreamer']
            sample_weights = []        # 和 bucket_list 一一对应，保存每个 bucket 的采样权重
            num_datasets = 0           # 统计当前用了几个"数据源"
            datasets = {}              # 字典 保存真正实例化出来的 bucket dataset
            # datasets['all'] = 某个dataset对象
            # datasets['turn_left'] = 某个dataset对象
            # datasets['all_dreamer'] = 某个dataset对象

            # 2. 检查有哪些训练源. 支持两类数据集来源: (1)driving dataset (2)dreamer dataset
            # self.driving_dataset={"_target_":"simlingo_training.dataloader.dataset_driving.Data_Driving"}
            # self.dreamer_dataset={"_target_":"simlingo_training.dataloader.dataset_dreamer.Data_Dreamer"}
            if self.driving_dataset is not None or self.dreamer_dataset is not None:
                # Create lists of datasets and their corresponding training partitions
                # 先把"存在的数据源"筛出来 这里准备两个并行列表
                used_driving_datasets = []  # 一个存真正的数据集配置对象
                used_train_partitions = []  # 一个存改数据集对应的bucket划分配置
                
                # Pair datasets with their training partitions and filter out None datasets
                dataset_pairs = zip(
                    [self.driving_dataset, self.dreamer_dataset],
                    [self.train_partitions, self.train_partitions_dreamer]
                )
                for dataset, partition in dataset_pairs:
                    if dataset is not None:
                        used_driving_datasets.append(dataset)
                        used_train_partitions.append(partition)
                # 意思是：
                # 如果某个 dataset 存在
                # 就把它加入训练使用列表
                # 它对应的 partition 也一起加入
                # 举个例子
                # 情况1：两个都存在 那么最后变为
                # used_driving_datasets = [driving_dataset, dreamer_dataset]
                # used_train_partitions = [train_partitions, train_partitions_dreamer]
                # 情况2：只有 driving_dataset 存在  那么最后变为
                # used_driving_datasets = [driving_dataset]
                # used_train_partitions = [train_partitions]

                # used_driving_datasets
                # = [
                #     {"_target_": "simlingo_training.dataloader.dataset_driving.Data_Driving"},
                #     {"_target_": "simlingo_training.dataloader.dataset_dreamer.Data_Dreamer"}
                #   ]
                # used_train_partitions
                # =[  {
                #     "all": 0.082,
                #     "acceleration_negative_5": 0.03,
                #     "acceleration_negative_1": 0.03,
                #     "acceleration_positive_1": 0.03,
                #     "acceleration_positive_5": 0.03,
                #     "lateral_control_1_2": 0.12,
                #     "lateral_control_higher_5": 0.12,
                #     "start_from_stop": 0.07,
                #     "vehicle_front": 0.04,
                #     "vehicle_side": 0.08,
                #     "leading_object_vehicle": 0.09,
                #     "leading_object_traffic.stop": 0.07,
                #     "leading_object_traffic.traffic_light": 0.07,
                #     "leading_object_walker": 0.05,
                #     "changed_route": 0.08,
                #     "parkinglane": 0.008
                #     },
                #     {"all":"1.0"}
                #     ]




                # 6. 设定 driving_data(VQA & Driving & Commentary) 和 dreamer_data(Dreamer) 两大类数据源的总权重
                weights_driving = 0.5
                weights_dreamer = 1 - weights_driving
                for udd_i, (used_driving_dataset, used_train_partitions) in enumerate(zip(used_driving_datasets, used_train_partitions)):
                    num_datasets += 1
                    # 如果当前数据源提供了 bucket 划分, 就按 bucket 拆开训练
                    if used_train_partitions is not None:
                        bucket_list_tmp = list(used_train_partitions.keys())
                        sample_weights_tmp = list(used_train_partitions.values())
                        sum_sample_weights = sum(sample_weights_tmp)
                        # bucket_list_tmp = ["all", "turn_left", "turn_right"]
                        # sample_weights_tmp = [0.6, 0.2, 0.2]
                        # sum_sample_weights = 1.0

                        # 对当前数据源内部的 bucket 权重做归一化
                        sample_weights_tmp = [w/sum_sample_weights for w in sample_weights_tmp]
                        # 如果 driving 和 dreamer 同时存在，还要乘以"数据源级总权重"
                        if self.driving_dataset is not None and self.dreamer_dataset is not None:
                            if udd_i == 0:
                                sample_weights_tmp = [w * weights_driving for w in sample_weights_tmp]
                            else:
                                sample_weights_tmp = [w * weights_dreamer for w in sample_weights_tmp]

                        # 给 dreamer 的 bucket 重命名,避免和 driving 冲突
                        if udd_i == 1:
                            bucket_list_tmp = [f"{b}_dreamer" for b in bucket_list_tmp]
                            # bucket_list_tmp = ["all_dreamer", "turn_left_dreamer", "turn_right_dreamer"]
                        # 把当前数据源的 bucket 名和权重追加到全局列表
                        bucket_list.extend(bucket_list_tmp)
                        sample_weights.extend(sample_weights_tmp)
                        # 比如处理完 driving 后可能变成：
                        # bucket_list = ['all', 'turn_left']
                        # sample_weights = [0.4, 0.1]
                        # 处理完 dreamer 后：
                        # bucket_list = ['all', 'turn_left', 'all_dreamer', 'rare_case_dreamer']
                        # sample_weights = [0.4, 0.1, 0.25, 0.25]
                    else:  # 如果当前数据源没有 partition，就默认整个数据源作为一个 bucket
                        if udd_i == 1:
                            bucket_list_tmp = ['all_dreamer']
                            sample_weights_tmp = [weights_dreamer]
                        else:
                            bucket_list_tmp = ['all']
                            sample_weights_tmp = [weights_driving]
                        bucket_list.extend(bucket_list_tmp)
                        sample_weights.extend(sample_weights_tmp)

                    
                    # 真正实例化每个 bucket 的训练集 setup()函数的核心
                    for bucket in bucket_list_tmp:
                        bucket_name = bucket.replace('_dreamer','')
                        datasets[bucket] = hydra.utils.instantiate(
                            used_driving_dataset,
                            split="train",
                            bucket_name=bucket_name,
                            **self.cfg,
                            **self.base_dataset,
                            _recursive_=False
                        )
                    # datasets = {
                    #    "all": Dataset1,
                    #    "vehicle_front": Dataset2,
                    #    "junction": Dataset3
                    #    "all_dreamer": Dataset4,
                    #    "vehicle_front_dreamer": Dataset5,
                    #    "junction_dreamer": Dataset6
                    # }
                    # 统一只构造一个 bucket_name="all" 的验证集
                    self.val_datasets.append(hydra.utils.instantiate(
                            used_driving_dataset,
                            split="val",
                            bucket_name="all",
                            **self.cfg,
                            **self.base_dataset,
                            _recursive_=False
                        ))
                    
                    sum_sample_weights = sum(sample_weights_tmp)
            

            # 开始构造最终训练集
            self.train_dataset = None
            if len(datasets) > 0:  # 如果确实实例化出了一些 bucket dataset, 就继续.
                
                # remove datasets with 0 samples  把那些空 bucket 对应的权重也一起删掉
                sample_weights = [sample_weights[i] for i, bucket in enumerate(bucket_list) if datasets[bucket].__len__() > 0]
                bucket_list = [bucket for bucket in bucket_list if datasets[bucket].__len__() > 0]
                if len(bucket_list) != len(datasets):
                    # print in red
                    print(f"\033[91mDatasets with 0 samples: {set(datasets.keys()) - set(bucket_list)}\033[00m")
                    print(f"\033[91mContinue without this bucket.\033[00m")
                # 真正把空 dataset 从字典里删掉
                datasets = {key: value for key, value in datasets.items() if value.__len__() > 0}
                # 把所有样本数为 0 的 bucket 全部删掉
                # 并同步更新：
                # - bucket_list
                # - sample_weights
                # - datasets

                # self.train_dataset = torch.utils.data.ConcatDataset([datasets[bucket] for bucket in bucket_list])
                # weights_train = [[sample_weights[i]] * datasets[bucket].__len__() for i, bucket in enumerate(bucket_list)]
                # weights_train = list(itertools.chain.from_iterable(weights_train))
                self.train_dataset = torch.utils.data.ConcatDataset(
                    [datasets[bucket] for bucket in bucket_list]
                )
                # sample_weights[i] 表示整个 bucket 期望占据的采样概率。
                # WeightedRandomSampler 接收的是逐样本权重，因此需要将
                # bucket 的总权重平均分配给该 bucket 内的所有样本。
                weights_train = [
                    [sample_weights[i] / datasets[bucket].__len__()]
                    * datasets[bucket].__len__()
                    for i, bucket in enumerate(bucket_list)
                ]
                weights_train = list(itertools.chain.from_iterable(weights_train))

                num_samples_all = [datasets[bucket].__len__() // sample_weights[i] for i, bucket in enumerate(bucket_list)]
                num_samples = int(min(num_samples_all))# * num_datasets
                print(f"Num samples: {num_samples}")
                if self.driving_dataset is not None:
                    print(f"Num samples all: {datasets['all'].__len__()}")
                # 把所有 bucket 拼成一个总训练集
                self.sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train, num_samples=num_samples, replacement=True)

            self.val_dataset = torch.utils.data.ConcatDataset(self.val_datasets)
            self.predict_dataset = None


            #### 训练部分 ####
            # train_dataset
            # 所有非空 bucket dataset 拼接后的总样本池
            # weights_train
            # 与 train_dataset 一一对应的 sample-level 权重向量
            # sampler_train
            # 基于 weights_train 的 WeightedRandomSampler
            # 决定每个 epoch 实际抽哪些样本、抽多少次
            # bucket1_dataset (5000)
            # bucket2_dataset (200)
            # bucket3_dataset (20)
            #         ↓
            # ConcatDataset
            #         ↓
            # train_dataset (5220)
            #         ↓
            # weights_train (5220个权重)
            #         ↓
            # WeightedRandomSampler
            #         ↓
            # 训练时按权重采样
            #### 验证部分 ####
            # val_dataset
            # 由 driving 和 dreamer 的 all 验证集拼接而成
            # 通常不加权、不重采样

        else:
            if self.qa_dataset is not None:
                predict_dataset = self.qa_dataset
                
            elif self.insteval_dataset is not None:
                predict_dataset = self.insteval_dataset

            self.predict_dataset = hydra.utils.instantiate(
                    predict_dataset,
                    split="val",
                    bucket_name="all",
                    **self.cfg,
                    **self.base_dataset,
                    _recursive_=False
                )


    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # shuffle=True, # we use custom sampler instead
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            sampler=self.sampler_train,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )


    @line_profiler.profile
    def dl_collate_fn(self, data):

        """
        这是 PyTorch DataLoader 的 collate_fn,
        DataLoader 每次从 dataset 里取出 batch_size 个样本后，会把这些样本组成一个列表 data, 然后交给这个函数处理.
        data: 长度为 batch size 的列表;
        data[i]: 第 i 个样本,一般是单样本版 DrivingExample;
        """

        
        
        
        
        BS = len(data)  # 这里就是取 batch size, 比如说此时 batch size=4 那么 BS=4, data=[sample1, sample2, sample3, sample4]
        grid_nums = [self.NUM_IMAGE_PATCHES] # we split the front forward into two patches (1x2)
        # 为什么写成列表？
            # 因为代码是按 self.IMAGES_TO_CONSIDER 循环处理图像的.
            # 虽然当前只支持 ['image_ff'] 一种图像，但simlingo作者把接口写成了多图像输入可扩展的形式。
            # 也就是说：
            # 第 1 种图像对应一个 patch 数
            # 第 2 种图像也可以对应另一个 patch 数
            # 只是现在只有前视图一种。
        
        
        
        
        
        
        
        # image_ff_pixel：预处理后的前视图图像张量      [BS, T, 2, 3, 448, 448]
        # image_ff_sizes：图像预处理后保留的一些尺寸信息
        image_ff_pixel, image_ff_sizes = None, None




        # 遍历这个batch中的每一个样本,取出它的img_ff_org_size(这是未经裁减的前视图像)
        image_ff_org = torch.tensor(np.asarray([data[i].image_ff_org_size for i in range(BS)]))

        
        
        
        
        
        ################################################### 🦺 1.图像预处理 🦺 ###################################################

        # 实际上当前只循环一次,indx=0,img_to_consider= 'image_ff',因为 self.IMAGES_TO_CONSIDER 里只有 'image_ff' 这一种图像. self.IMAGES_TO_CONSIDER=['image_ff']
        # 写成循环的原因仍然是:为了以后支持多路相机
        for idx, img_to_consider in enumerate(self.IMAGES_TO_CONSIDER):
            
            
            # img_tmp是经过裁减之后的前视图像,就是将image_ff_org的图像的底部包含自车引擎盖的那部分裁减掉了
            img_tmp = getattr(data[0], img_to_consider) # 等价于img_tmp = data[0].image_ff  也就是从 batch 第一个样本里取出前视图图像
            T, C, H, W = img_tmp.shape                  # img_tmp 的 shape 是 [T, C, H, W] 也就是说每个样本的这一种图像是一个有 T 帧的图像序列. 例如 T=4 就是4帧图像(但是当前只支持一帧图像). C=3 是通道数,H和W是图像尺寸.
            assert T == 1, "Only one timestep as input supported"
            
            # 对于当前 batch 中每个样本：
            # 如果这个样本的 image_ff 不为空，就取它
            # 如果为空，就用一个和 img_tmp 形状相同的全零图像代替
            # 作用是避免某些样本图像缺失时程序崩掉
            # 这是一种兜底策略
            # 把上面的列表变成 numpy 数组, 如果每个单样本图像是 [T,C,H,W]，那么 batch 后就变成[BS, T, C, H, W]
            # 转成 PyTorch float tensor  注意这里直接转成 float,说明后续视觉预处理函数是按浮点张量来处理的,而不是保留 uint8
            images_batch_tensor = torch.tensor(np.asarray([getattr(data[i], img_to_consider) if getattr(data[i], img_to_consider) is not None else np.zeros_like(img_tmp) for i in range(len(data))])).float()
            images_batch_tensor = images_batch_tensor.view(BS*T, C, H, W)  # 把 [BS,T,C,H,W] 变成 [BS*T,C,H,W]
            
            images_batch_list = list(images_batch_tensor)  # list 中每个元素是一张图像，每一个元素的形状是 [C,H,W], 一共有BS*T个元素.
            # print(f"images batch list 0: {images_batch_list[0]}")
            # images batch list 0: tensor([[[ 5., 20.,  7.,  ...,  0., 16.,  4.],
                                        #  [ 7.,  6.,  0.,  ...,  4., 15., 19.],
                                        #  [ 4.,  8.,  9.,  ...,  3., 10., 25.],
                                        #  ...,
                                        #  [72., 60., 54.,  ..., 22., 12.,  0.],
                                        #  [52., 63., 50.,  ...,  7., 19.,  8.],
                                        #  [42., 33., 15.,  ...,  6.,  0.,  5.]],

                                        # [[ 2., 17., 19.,  ...,  0., 15.,  0.],
                                        #  [ 0.,  5.,  0.,  ...,  0.,  8., 13.],
                                        #  [ 8.,  8.,  2.,  ...,  3.,  0., 23.],
                                        #  ...,
                                        #  [40., 50., 43.,  ..., 19.,  6.,  0.],
                                        #  [40., 42., 26.,  ...,  0.,  5.,  0.],
                                        #  [29., 19., 17.,  ...,  3.,  9., 10.]],

                                        # [[ 3., 10.,  9.,  ...,  5.,  6.,  6.],
                                        #  [ 3.,  3.,  5.,  ...,  0.,  0.,  0.],
                                        #  [ 2.,  0.,  0.,  ...,  0.,  0.,  7.],
                                        #  ...,
                                        #  [25., 27., 17.,  ...,  7.,  1.,  0.],
                                        #  [35., 29.,  6.,  ...,  0.,  0.,  0.],
                                        #  [21., 17.,  8.,  ...,  4.,  3.,  0.]]])


            # 根据视觉编码器类型做图像预处理
            if 'internvl2' in self.encoder_variant.lower():  # OpenGVLab/InternVL2-1B 转小写 包含 internvl2
                # get image patches
                images_processed = preprocess_image_batch(images_batch_list, input_size=448, use_global_img=self.use_global_img, max_num_grid=grid_nums[idx])    
            else:
                raise ValueError(f"Image preprocessing for {self.encoder_variant} not implemented")
                
            images_pixel = images_processed['pixel_values']  # 预处理后的像素张量 形状为 [BS*T, 2, 3, 448, 448]  也就是说，一张图像可能被拆成多个 patch，每个 patch 都变成适合视觉模型输入的张量
            image_sizes = images_processed['image_sizes']    # 当前帧图像的尺寸信息 形状为 [BS*T, 2]，每行是一个 [原图高度 H, 原图宽度 W]
            
            assert images_pixel.shape[0] == BS * T   # 检查预处理后第一维是否仍然和输入图像数一致,也就是确保没有多图少图
            num_patches = images_pixel.shape[1]      # 取出每张图像被切成多少个 patch
            assert images_pixel.shape[2] == C        # 检查通道数有没有变
            new_height = images_pixel.shape[3]       # 取patch高度
            new_width = images_pixel.shape[4]        # 取patch宽度
            images_pixel = images_pixel.view(BS, T, num_patches, C, new_height, new_width)  # 把图像张量重新 reshape 回 [BS, T, num_patches, C, new_H, new_W] 的形状
            # images_pixel 就是 DrivingInput.camera_images 的标准格式

            if img_to_consider == 'image_ff':
                image_ff_pixel = images_pixel  # [BS, T, 2, 3, 448, 448] 这就是前视图图像的预处理结果
                image_ff_sizes = image_sizes   # [BS*T, 2] 
            else:
                raise ValueError(f"Image type {img_to_consider} not supported")

        
        
        
        
        
        
        
        
        
        ################################################### 🦺 2.文本预处理 🦺 ###################################################
        
        
        # 把 batch 中每个样本的对话文本取出来
        conversations = [data[i].conversation for i in range(BS)]
        # conversations = [[{'role': 'user', 'content': [{'type': 'text', 'text': 'Current speed: 10.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.'}, {'type': 'image'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Waypoints:'}]}]]

        conversation_dict, question_dict = get_custom_chat_template(conversations, self.tokenizer, self.encoder_variant, self.num_image_tokens_total)
        
        """
            conversation_dict = {
            'phrase_ids': prompt_tokenized_ids,       # token id 形状为[B,L]
            'phrase_valid': prompt_tokenized_valid,   # 标记哪些token不是padding 形状为[B,L]
            'phrase_mask': prompt_tokenized_mask,     # 直接把有效token的位置作为mask 形状为[B,L]
            'language_string': prompts,               # 原始文本 形状为[B]  包括问题和答案！！！！！！！！！！！！
            'loss_masking': loss_mask                 # 哪些位置计算loss

            question_dict = {
            'phrase_ids': prompt_tokenized_ids,       # token id 形状为[B,L]
            'phrase_valid': prompt_tokenized_valid,   # 标记哪些token不是padding 形状为[B,L]
            'phrase_mask': prompt_tokenized_mask,     # 直接把有效token的位置作为mask 形状为[B,L]
            'language_string': prompts,               # 原始文本 形状为[B]  仅包括问题！！！！！！！！！！！！
            'loss_masking': loss_mask                 # 哪些位置计算loss
        }
        """        
        
        
        
        
        
        
        
        
        
        
        ################################################### 🦺 3.占位符预处理 🦺 ###################################################


        placeholder_batch_list = []
        for i in range(BS):  # 遍历 batch 中每个样本
            tmp = {}   # 给当前样本准备一个临时字典
            # 遍历这个样本里的 placeholder 映射 这里的 key 通常是字符串形式 special token，是 '<TARGET_POINT>' , value 则是它对应的数值内容
            for key, value in data[i].placeholder_values.items():           # placeholder_values = {'<TARGET_POINT>': data['target_points']}
                token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                tmp[token_nr_key] = value
                # print(f"Sample {i} placeholder key: {key}, token id: {token_nr_key}, value: {value}")
                # Sample 0 placeholder key: <TARGET_POINT>, token id: 151662, value: [[156.29730464  -1.23297884],[357.25860725  -6.47138093]]

            placeholder_batch_list.append(tmp)
            # print(f"Sample {i} placeholder batch: {tmp}")
            # Sample 0 placeholder batch: {151662: array([[156.29730464,  -1.23297884],[357.25860725,  -6.47138093]])}
            # print(f"placeholder batch list sample {i}: {placeholder_batch_list[i]}")
            # placeholder batch list sample 0: {151662: array([[16.18833904, -1.86136625],[72.12040229, -4.63741635]])}


                
        
        
        
        
        
        
        
        
        
        ################################################### 🦺 4.将 placeholder_values 融合进来 🦺 ###################################################


        prompt_languagelabel = LanguageLabel(
            phrase_ids=conversation_dict['phrase_ids'],           # token ids，也就是 tokenizer 编码后的整数序列
            phrase_valid=conversation_dict['phrase_valid'],       # 标记哪些 token 不是padding的
            phrase_mask=conversation_dict['phrase_mask'],         # 标记哪些 token 不是padding的(同phrase_valid)
            placeholder_values=placeholder_batch_list,            # {151662:[[16.18833904, -1.86136625],[72.12040229, -4.63741635]]}
            language_string=conversation_dict['language_string'], # 原始文本字符串(包括问题和答案)
            loss_masking=conversation_dict['loss_masking'],       # 当前language_string里哪些位置需要计算loss
        )

        prompt_question_languagelabel = LanguageLabel(
            phrase_ids=question_dict['phrase_ids'],
            phrase_valid=question_dict['phrase_valid'],
            phrase_mask=question_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,            # {151662: [[16.18833904, -1.86136625],[72.12040229, -4.63741635]]}
            language_string=question_dict['language_string'],     #
            loss_masking=question_dict['loss_masking'],
        )

        # 从当前batch中的每个样本中抽出文本答案
        answer_string_list = [data[i].answer[0]['content'][0]['text'] for i in range(BS)]
        answer_label =  LanguageLabel(
            phrase_ids=None,
            phrase_valid=None,
            phrase_mask=None,
            placeholder_values=None,
            language_string=answer_string_list,  # 这说明此处的 answer_label 更像是一个“容器对象”，用来统一接口；当前这里只需要保存 ground-truth 文本答案本身
            loss_masking=None,
        )
        
        
        
        







        ################################################### 🦺 5.拉取 waypoints 信息 🦺 ###################################################
        

        # 构造 waypoints 标签 这里的 waypoints 是未来轨迹标签，通常是一个点序列，形状是 [B, F, 2]，其中 F 是未来轨迹点的数量，每个点有 x,y 两个坐标
        if self.base_dataset.use_1d_wps:
            waypoints = torch.tensor(np.asarray([data[i].waypoints_1d for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        else:
            waypoints = torch.tensor(np.asarray([data[i].waypoints for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        
        
        
        
        
        
        ################################################### 🦺 6.额外信息(预测/评估模式才有) 🦺 ###################################################
        
        
        if self.predict:  # 如果当前是预测/评估模式，那么保留一些额外信息
            qa_templates = [data[i].qa_templates[0] if data[i].qa_templates is not None else None for i in range(BS) ]
            eval_infos = [data[i].eval_infos if data[i].eval_infos is not None else None for i in range(BS) ]
        else:
            qa_templates = None
            eval_infos = None
        
        
        
        
        
        
        
        ################################################### 🦺 7.最终形态 🦺 ###################################################


        # 这是整个函数最重要的结构之一: 模型输入对象
        driving_input=DrivingInput(
                camera_images=image_ff_pixel,  # [B, T, N, C, H, W] uint8 [0, 255]  这是视觉主输入
                image_sizes=image_ff_sizes,    # 每个 patch 的尺寸信息
                camera_intrinsics = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), BS, dim=0).view(BS, 3, 3).float(),  # 相机内参
                camera_extrinsics = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), BS, dim=0).view(BS, 4, 4).float(),           # 相机外参
                vehicle_speed=torch.tensor(np.asarray([data[i].speed for i in range(len(data))])).float(),  # [B, S] float32                     速度
                target_point=torch.tensor(np.asarray([data[i].target_points for i in range(len(data))])).float(),  # [B, 2] float32              target point
                prompt=prompt_languagelabel,                            # 训练用 prompt 包含问题和答案
                prompt_inference=prompt_question_languagelabel,         # 推理用 prompt 包含问题但不包含答案
            )

        # 整个 batch 的监督信号对象
        driving_label=DrivingLabel(
                waypoints=waypoints,        # waypoints 未来轨迹监督 [B, F, 2] float32
                path=torch.tensor(np.asarray([data[i].path for i in range(len(data))])).float(), # [B, 3, RH, RW] uint8 [0, 255]
                answer=answer_label,        # 文本答案监督
                image_ff_org=image_ff_org,  # 没有经过裁剪的原始图像
                eval_infos=eval_infos,      # 预测模式下的额外评估信息
            )
            
        return DrivingExample(
            driving_input=driving_input,  # 把刚才整理好的"模型输入"塞进去
            driving_label=driving_label,  # 把刚才整理好的"监督标签"塞进去
            run_id=encode_uint8([data[i].measurement_path for i in range(BS)], 1000),  # [B] str  把每个样本的 measurement_path 编码成固定长度 uint8 张量
            qa_templates=qa_templates,    # 保留预测模式下的问题模板信息
        )

    def dl_collate_fn_val(self, data):
        pass

    def dl_collate_fn_test(self, data):
        pass







@hydra.main(config_path=f"../config", config_name="config", version_base="1.1")
def test(cfg):
    
    get_waypoint_stats = True
    
        
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    dm = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        # tokenizer=llm_tokenizer,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant="llava-hf/llava-v1.6-mistral-7b-hf",
        _recursive_=False
    )


    dm.setup()
    dl = dm.val_dataloader()
    print(dl.dataset.__len__())

    iterations = 0
    
    all_waypoints = []
    all_waypoints_diff = []
    for batch in dl:
        
        iterations += 1
        
        if iterations % 100 == 0:
            print(f"Iteration: {iterations}")
        
        if iterations > 20000:
            break
        
        if get_waypoint_stats:
            # get stats about range of waypoints
            waypoints = batch.driving_label.waypoints
            all_waypoints.append(waypoints)
            
            # get residuals
            residuals = waypoints[:,1:] - waypoints[:,:-1]
            all_waypoints_diff.append(residuals)
            
    # get histogram of waypoints
    if get_waypoint_stats:
        all_waypoints = torch.cat(all_waypoints, dim=0)
        all_waypoints_diff = torch.cat(all_waypoints_diff, dim=0)
        
        all_waypoints = all_waypoints.view(-1, 2)
        all_waypoints_diff = all_waypoints_diff.view(-1, 2)
        
        
        
        import matplotlib.pyplot as plt
        plt.hist(all_waypoints[:,0].numpy(), bins=100)
        plt.savefig('waypoints_x.png')
        max_x = all_waypoints[:,0].max().item()
        min_x = all_waypoints[:,0].min().item()
        print(f"Max x: {max_x}, Min x: {min_x}")
        plt.clf()
        plt.hist(all_waypoints[:,1].numpy(), bins=100)
        plt.savefig('waypoints_y.png')
        max_y = all_waypoints[:,1].max().item()
        min_y = all_waypoints[:,1].min().item()
        print(f"Max y: {max_y}, Min y: {min_y}")
        plt.clf()
        
        plt.hist(all_waypoints_diff[:,0].numpy(), bins=100)
        plt.savefig('waypoints_diff_x.png')
        max_x_diff = all_waypoints_diff[:,0].max().item()
        min_x_diff = all_waypoints_diff[:,0].min().item()
        print(f"Max x diff: {max_x_diff}, Min x diff: {min_x_diff}")
        plt.clf()
        plt.hist(all_waypoints_diff[:,1].numpy(), bins=100)
        plt.savefig('waypoints_diff_y.png')
        max_y_diff = all_waypoints_diff[:,1].max().item()
        min_y_diff = all_waypoints_diff[:,1].min().item()
        print(f"Max y diff: {max_y_diff}, Min y diff: {min_y_diff}")
        plt.clf()
            
if __name__ == "__main__":
    test()