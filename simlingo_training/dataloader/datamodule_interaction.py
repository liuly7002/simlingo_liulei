# -*- coding: utf-8 -*-

import copy
from typing import List

import numpy as np
import torch

from simlingo_training.dataloader.datamodule import DataModule
from simlingo_training.dataloader.participant_spatial_attention import (
    CAMERA_ORDER,
    TOKENS_PER_CAMERA,
)
from simlingo_training.utils.custom_types import LanguageLabel
from simlingo_training.utils.internvl2_utils import (
    get_custom_chat_template,
)


class InteractionDataModule(DataModule):
    """
    复用原DataModule的全部图像、文本、轨迹和四通道批处理逻辑，
    仅将历史六维相机监督接口升级为6×64参与者空间监督。
    """

    def dl_collate_fn(self, data: List):

        batch_size = len(data)

        # 六视角注意力 token 级别  [B, 6, 64]
        spatial_target = torch.zeros((batch_size,len(CAMERA_ORDER),TOKENS_PER_CAMERA,),dtype=torch.float32,)
        
        # 六视角有效性 True False
        spatial_valid = torch.zeros((batch_size,),dtype=torch.bool,)

        compatibility_data = []
        for sample_index, sample in enumerate(data):

            # 当前样本是否有有效的注意力
            sample_valid = bool(getattr(sample,"camera_attention_valid",False,))

            # [6] 当前样本六视角相机级别注意力
            sample_target = getattr(sample,"camera_attention_target",None,)

            if sample_target is None:
                sample_target_array = np.zeros((len(CAMERA_ORDER),TOKENS_PER_CAMERA,),dtype=np.float32,)
            else:
                sample_target_array = np.asarray(sample_target,dtype=np.float32,)

            # 安全性检查
            expected_shape = (len(CAMERA_ORDER),TOKENS_PER_CAMERA,)
            if sample_target_array.shape != expected_shape:
                raise ValueError(
                    "Participant spatial attention target must have "
                    f"shape {expected_shape}, but received "
                    f"{tuple(sample_target_array.shape)}."
                )
            if (
                not np.isfinite(sample_target_array).all()
                or np.any(sample_target_array < 0.0)
            ):
                raise ValueError(
                    "Participant spatial attention target contains "
                    "non-finite or negative values."
                )

            # 六视角整体求和
            target_sum = float(sample_target_array.sum())
            
            
            if sample_valid:
                
                # 安全性检查
                if target_sum <= 0.0:
                    raise ValueError(
                        "A valid participant spatial attention target "
                        "must have a positive probability sum."
                    )

                # 归一化
                sample_target_array = (sample_target_array / target_sum)
                
                # 保存 六视角注意力 token 级别  [B, 6, 64]
                spatial_target[sample_index] = torch.from_numpy(sample_target_array)

                # 保存当前样本注意力有效性为True
                spatial_valid[sample_index] = True


            # 特殊处理,由于datamodule.py是使用的是六视角相机级别的注意力而不是token级别的,所以这里将sample_target_array由[B,6,64]转换到[B,6]
            camera_marginal = sample_target_array.sum(axis=-1)
            compatibility_data.append(
                sample._replace(
                    camera_attention_target=(
                        camera_marginal.astype(np.float32)
                    ),
                    camera_attention_valid=sample_valid,
                )
            )

        # 调用 datamodule.py的dl_collate_fn函数生成基本数据
        example = super().dl_collate_fn(compatibility_data)

        waypoint_shape = tuple(example.driving_label.waypoints.shape[1:])  # [10,2]

        # 初始化 反事实 waypoints [B,10,2]
        counterfactual_waypoints = torch.zeros((batch_size, *waypoint_shape),dtype=torch.float32,)
        # 初始化反事实 waypoints 的有效性 [B]
        counterfactual_waypoints_valid = torch.zeros((batch_size,),dtype=torch.bool,)
        # 初始化反事实 waypoints 的得分 [B]
        counterfactual_causal_score = torch.zeros((batch_size,),dtype=torch.float32,)

        for sample_index, sample in enumerate(data):

            sample_valid = bool(getattr(sample,"counterfactual_waypoints_valid",False,))
            if not sample_valid:
                continue

            # 1. 反事实 waypoints
            sample_waypoints = np.asarray(getattr(sample,"counterfactual_waypoints",None,),dtype=np.float32,)
            
            # 安全性检查
            if sample_waypoints.shape != waypoint_shape:
                raise ValueError(
                    "Counterfactual waypoints must have shape "
                    f"{waypoint_shape}, but received "
                    f"{tuple(sample_waypoints.shape)}."
                )
            if not np.isfinite(sample_waypoints).all():
                raise ValueError(
                    "Counterfactual waypoints contain non-finite values."
                )

            # 写入
            counterfactual_waypoints[sample_index] = (torch.from_numpy(sample_waypoints))
            counterfactual_waypoints_valid[sample_index] = True

            # 2. 反事实 waypoints 的得分
            causal_score = float(getattr(sample,"counterfactual_causal_score",0.0,))
            if not np.isfinite(causal_score) or causal_score < 0.0:
                raise ValueError(
                    "counterfactual_causal_score must be finite and "
                    "non-negative."
                )

            # 写入
            counterfactual_causal_score[sample_index] = causal_score

        # 反事实分支使用独立文本序列：
        # 有真实对象移除标签的LG样本使用反事实四问答案；
        # 其余样本只保留中性的Waypoints前缀，避免完整场景答案泄漏。
        counterfactual_conversations = []
        for sample in data:

            counterfactual_conversation = getattr(sample,"counterfactual_conversation",None,)
            if counterfactual_conversation is not None:
                counterfactual_conversations.append(
                    copy.deepcopy(counterfactual_conversation)
                )
                continue

            user_message = next(
                (
                    copy.deepcopy(message)
                    for message in sample.conversation
                    if str(message.get("role", "")) == "user"
                ),
                None,
            )
            if user_message is None:
                raise ValueError(
                    "A counterfactual prompt requires a user message."
                )
            counterfactual_conversations.append(
                [
                    user_message,
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Waypoints:",
                            }
                        ],
                    },
                ]
            )

        
        # 文本预处理
        counterfactual_dict, _ = get_custom_chat_template(
            counterfactual_conversations,
            self.tokenizer,
            self.encoder_variant,
            self.num_image_tokens_total,
        )



        # 反事实 prompt
        # 对于存在真实反事实语言标签的 LG 样本，使用反事实四问答案；其他样本则只保留中性的 Waypoints:
        counterfactual_prompt = LanguageLabel(
            phrase_ids=counterfactual_dict["phrase_ids"],
            phrase_valid=counterfactual_dict["phrase_valid"],
            phrase_mask=counterfactual_dict["phrase_mask"],
            placeholder_values=(example.driving_input.prompt.placeholder_values),
            language_string=counterfactual_dict["language_string"],
            loss_masking=counterfactual_dict["loss_masking"],
        )

        # 为输入添加反事实 prompt
        driving_input = example.driving_input._replace(
            counterfactual_prompt=counterfactual_prompt
        )

        # 为标签更新六视角相机token级注意力[B,6,64] 
        # 为标签增加反事实waypoints
        driving_label = example.driving_label._replace(
            camera_attention_target=spatial_target,
            camera_attention_valid=spatial_valid,
            counterfactual_waypoints=(counterfactual_waypoints),
            counterfactual_waypoints_valid=(counterfactual_waypoints_valid),
            counterfactual_causal_score=(counterfactual_causal_score),
        )
        return example._replace(
            driving_input=driving_input,
            driving_label=driving_label,
        )
