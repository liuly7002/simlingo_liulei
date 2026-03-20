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

        self.NUM_IMAGE_PATCHES = 2             # 盲猜:一张原始输入图像会被拆成几个“图像块”送入视觉模型
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
            
        # add <WAYPOINT> token
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




                # 6. 设定driving 和 dreamer 两大类数据源的总权重
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

                self.train_dataset = torch.utils.data.ConcatDataset([datasets[bucket] for bucket in bucket_list])
                weights_train = [[sample_weights[i]] * datasets[bucket].__len__() for i, bucket in enumerate(bucket_list)]
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
        BS = len(data)
        grid_nums = [self.NUM_IMAGE_PATCHES] # we split the front forward into two patches (1x2)

        image_ff_pixel, image_ff_sizes = None, None
        image_ff_org = torch.tensor(np.asarray([data[i].image_ff_org_size for i in range(BS)]))
            
        for idx, img_to_consider in enumerate(self.IMAGES_TO_CONSIDER):
            img_tmp = getattr(data[0], img_to_consider)
            T, C, H, W = img_tmp.shape
            assert T == 1, "Only one timestep as input supported"
            
            images_batch_tensor = torch.tensor(np.asarray([getattr(data[i], img_to_consider) if getattr(data[i], img_to_consider) is not None else np.zeros_like(img_tmp) for i in range(len(data))])).float()
            images_batch_tensor = images_batch_tensor.view(BS*T, C, H, W)
            images_batch_list = list(images_batch_tensor)

            if 'internvl2' in self.encoder_variant.lower():
                # get image patches
                images_processed = preprocess_image_batch(images_batch_list, input_size=448, use_global_img=self.use_global_img, max_num_grid=grid_nums[idx])    
            else:
                raise ValueError(f"Image preprocessing for {self.encoder_variant} not implemented")
                
            images_pixel = images_processed['pixel_values']
            image_sizes = images_processed['image_sizes']
            
            assert images_pixel.shape[0] == BS * T
            num_patches = images_pixel.shape[1]
            assert images_pixel.shape[2] == C
            new_height = images_pixel.shape[3]
            new_width = images_pixel.shape[4]
            images_pixel = images_pixel.view(BS, T, num_patches, C, new_height, new_width)
            
            if img_to_consider == 'image_ff':
                image_ff_pixel = images_pixel
                image_ff_sizes = image_sizes
            else:
                raise ValueError(f"Image type {img_to_consider} not supported")

        conversations = [data[i].conversation for i in range(BS)]
        conversation_dict, question_dict = get_custom_chat_template(conversations, self.tokenizer, self.encoder_variant, self.num_image_tokens_total)

        placeholder_batch_list = []
        for i in range(BS):
            tmp = {}
            for key, value in data[i].placeholder_values.items():
                token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)
                
        prompt_languagelabel = LanguageLabel(
            phrase_ids=conversation_dict['phrase_ids'],
            phrase_valid=conversation_dict['phrase_valid'],
            phrase_mask=conversation_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,
            language_string=conversation_dict['language_string'],
            loss_masking=conversation_dict['loss_masking'],
        )

        prompt_question_languagelabel = LanguageLabel(
            phrase_ids=question_dict['phrase_ids'],
            phrase_valid=question_dict['phrase_valid'],
            phrase_mask=question_dict['phrase_mask'],
            placeholder_values=placeholder_batch_list,
            language_string=question_dict['language_string'],
            loss_masking=question_dict['loss_masking'],
        )
        answer_string_list = [data[i].answer[0]['content'][0]['text'] for i in range(BS)]
        answer_label =  LanguageLabel(
            phrase_ids=None,
            phrase_valid=None,
            phrase_mask=None,
            placeholder_values=None,
            language_string=answer_string_list,
            loss_masking=None,
        )
        
        if self.base_dataset.use_1d_wps:
            waypoints = torch.tensor(np.asarray([data[i].waypoints_1d for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        else:
            waypoints = torch.tensor(np.asarray([data[i].waypoints for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        
        if self.predict:
            qa_templates = [data[i].qa_templates[0] if data[i].qa_templates is not None else None for i in range(BS) ]
            eval_infos = [data[i].eval_infos if data[i].eval_infos is not None else None for i in range(BS) ]
        else:
            qa_templates = None
            eval_infos = None
        
        driving_input=DrivingInput(
                camera_images=image_ff_pixel,  # [B, T, N, C, H, W] uint8 [0, 255]
                image_sizes=image_ff_sizes,
                camera_intrinsics = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), BS, dim=0).view(BS, 3, 3).float(),
                camera_extrinsics = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), BS, dim=0).view(BS, 4, 4).float(),
                vehicle_speed=torch.tensor(np.asarray([data[i].speed for i in range(len(data))])).float(),  # [B, S] float32
                target_point=torch.tensor(np.asarray([data[i].target_points for i in range(len(data))])).float(),  # [B, 2] float32
                prompt=prompt_languagelabel,
                prompt_inference=prompt_question_languagelabel,
            )

        driving_label=DrivingLabel(
                waypoints=waypoints,
                path=torch.tensor(np.asarray([data[i].path for i in range(len(data))])).float(), # [B, 3, RH, RW] uint8 [0, 255]
                answer=answer_label,
                image_ff_org=image_ff_org,
                eval_infos=eval_infos,
            )
            
        return DrivingExample(
            driving_input=driving_input,
            driving_label=driving_label,
            run_id=encode_uint8([data[i].measurement_path for i in range(BS)], 1000),  # [B] str
            qa_templates=qa_templates,
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