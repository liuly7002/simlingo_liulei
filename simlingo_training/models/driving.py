import datetime
import json
import os
import random
from pathlib import Path
from pprint import PrettyPrinter
from typing import Dict, Optional, Tuple, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from hydra.utils import get_original_cwd


from simlingo_training.models.adaptors.adaptors import DrivingAdaptor, LanguageAdaptor, WaypointInputAdaptor, AdaptorList
# 结构化未来世界预测分支
from simlingo_training.models.future_interaction import (
    FUTURE_INTERACTION_CHANNEL_KEYS,
    FutureInteractionDecoder,
    compute_future_interaction_losses,
)
from simlingo_training.models.utils import summarise_losses
from simlingo_training.utils.custom_types import (DrivingExample, DrivingInput,
                                                DrivingLabel, DrivingOutput,
                                                TrainingOutput)


pprint = PrettyPrinter().pprint

def decode_uint8(encoded: torch.Tensor) -> List[str]:
    return [row.tobytes().decode("utf-8").rstrip("\0") for row in encoded.cpu().numpy()]

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])


class DrivingModel(pl.LightningModule):
    
    def __init__(self,cfg_data_module,processor,cache_dir,**cfg,):

        # 调用父类__init__()函数
        super().__init__()
        self.save_hyperparameters()
        
        # 将配置文件内的变量变为类成员变量
        for key, value in cfg.items():
            setattr(self, key, value)
            
        self.processor = processor
        
        self.prediction = {}
        
        self.predict_language = True
        
        self.cfg_data_module = cfg_data_module
        
        # 视觉模型
        self.vision_model = hydra.utils.instantiate(
            self.vision_model,
            cfg_data_module=cfg_data_module,
            processor=self.processor,
            cache_dir=cache_dir,
            _recursive_=False
        )
            
        # 语言模型  
        self.language_model = hydra.utils.instantiate(
            self.language_model,
            cache_dir=cache_dir,
            _recursive_=False
        )

        self.all_predictions = {}
        self.all_losses = {}



        # 预测头 + 计算损失
        driving = None
        driving = DrivingAdaptor(
            self.language_model.hidden_size,   # 11
            speed_wps_mode=self.speed_wps_mode,# 2d
            predict_route_as_wps=self.predict_route_as_wps,
        )
        self.adaptors = AdaptorList(
            language=LanguageAdaptor(self.language_model),
            driving=driving,
        )



        # 四通道结构化未来世界辅助预测头
        self.future_interaction_decoder = None

        # 如果使用四通道结构化未来世界辅助预测,那么构造解码器
        if bool(getattr(self,"use_future_interaction_prediction",False,)):
            
            self.future_interaction_decoder = (
                FutureInteractionDecoder(
                    hidden_size=(self.language_model.hidden_size),  # 1 表示只使用当前1帧
                    output_channels=4,                              # 表示是四通道
                    output_size=int(getattr(self,"future_interaction_output_size",128,)  # 结构化世界大小是 128x128
                    ),
                )
            )





        self.wp_encoder = WaypointInputAdaptor(
            token_size=self.language_model.hidden_size,  # 语言模型的隐状态尺寸
            hidden_size=256,
            hidden_size2=512,
        )

        
        
        
        # 加载模型的 tokenizer
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor










    """
    1. 训练时最外层流程:

        Trainer.fit(...)
        ↓
        Lightning 自动调用 training_step(batch)
        ↓
        training_step 里调用 forward_loss(batch)
        ↓
        forward_loss 里调用 self.adaptors(example)
        ↓
        forward_loss 里调用 forward_model(...)
        ↓
        forward_model 里调用 language_model.model(...)
        ↓
        forward_loss 再调用 self.adaptors.compute_loss(...)
        ↓
        返回 loss


    2. 验证时最外层流程:

        Trainer.validate(...) 或 fit 中的 val loop
        ↓
        Lightning 自动调用 validation_step(batch)
        ↓
        validation_step 调 forward_loss(batch)
        ↓
        forward_loss 调 forward_model(...)
        ↓
        forward_loss 调 compute_loss(...)


    3. 推理时最外层流程:

        Trainer.predict(...)
        ↓
        Lightning 自动调用 predict_step(batch)
        ↓
        predict_step 调 self.forward(batch, return_language=True)
        ↓
        forward 内部做推理
        ↓
        得到 speed_wps, route, language
        ↓
        predict_step 再把预测结果和GT整理保存

    """



    ########################################### 推理接口 ###########################################
    def forward(self,example: DrivingExample,return_language: Optional[bool] = None,prompt_ids: Optional[Tensor] = None,) -> DrivingOutput:

        self.speed_wps, self.route, self.language = None, None, []
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
        
        if driving_input is not None:
            adaptor_dict = self.adaptors(example, inference=True)  # 推理(inference=True)
            adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
                    adaptor_dict = adaptor_dict,
                    pixel_values = driving_input.camera_images,                               # [BS*T,12,3,488,488] 六视角图像
                    placeholder_values = driving_input.prompt_inference.placeholder_values,   # {151662: [[x_0,  y_0],[x_1,  y_1]]}
                    wp_encoder = self.wp_encoder,                                             # 导航点编码器
                )
            
            input_embeds_all = adaptor_dict["language_inputs"]
            attention_masks = adaptor_dict['language_inputs_mask']


        if self.predict_language:

            # per batch item because of padding
            for b_idx, (input_embed, attention_mask) in enumerate(zip(input_embeds_all, attention_masks)):

                
                
                ########################## 1. 预处理 ##########################
                input_embed = input_embed.unsqueeze(0)       # 👉 从 [L, D] → [1, L, D]  因为模型需要 batch 维度
                attention_mask = attention_mask.unsqueeze(0) # 👉 从 [L, D] → [1, L, D]  因为模型需要 batch 维度


                ########################## 2. 设置 EOS token(不同模型不同), 用来控制生成停止 ##########################
                if self.language_model.variant == 'OpenGVLab/InternVL2-4B':
                    eos = self.tokenizer.added_tokens_encoder['<|end|>']
                elif self.language_model.variant == 'OpenGVLab/InternVL2-2B':
                    eos = self.tokenizer.added_tokens_encoder['<|im_end|>']
                else:
                    eos = self.tokenizer.eos_token_id

                
                
                
                ########################## ⭐ 3. 核心：语言生成（greedy decoding） ##########################
                # BUG: input_embeds, cot
                sampled_tokens, input_embeds = self.language_model.greedy_sample(
                    input_embed,   # 当前 prompt embedding（已经包含图像 + target point）
                    eos_token_id=eos,
                    max_new_tokens=100,
                    input_embed_matrix=self.adaptors.language.embed_tokens.weight,  # token → embedding
                    logit_matrix=self.adaptors.language.lm_head.weight,             # embedding → vocab logits
                    attention_mask=attention_mask,  # mask
                    # position_ids=position_ids,
                )
                # sampled_tokens: 生成的 token id
                # input_embeds: 生成后的 embedding（关键！）不是原始输入！而是原始输入 + 生成的 token embedding 拼接后的结果
                
                
                
                
                
                
                ########################## 4. 获取 driving 输入 ##########################
                inputs_driving = self.adaptors.driving(driving_input)
                
                
                
                
                ########################## 5. 拼接语言 + driving ##########################
                # 🔥 这一步是整个设计的核心
                # 拼接后变成: [语言token（含生成） | driving token] 也就是说 让 driving prediction "看到"语言生成结果
                input_embed_concat = torch.cat((input_embeds, inputs_driving["inputs"][b_idx].unsqueeze(0)), dim=1)




                ########################## 6. forward ##########################
                features, logits = self.language_model.forward(input_embed_concat)

                
                
                
                ########################## 7. 取 driving 部分 进行预测 ##########################
                len_driving = inputs_driving["inputs"].size(1)

                driving_features = features[:, -len_driving:]
                driving_logits = logits[:, -len_driving:]
                predictions = self.adaptors.driving.get_predictions(driving_features, driving_logits)
                    
                
                
                
                
                ########################## 8. 累加 batch 结果 ##########################
                for k, v in predictions.items():
                    if v is not None:
                        if hasattr(self, k) and getattr(self, k) is not None:
                            if isinstance(v, torch.Tensor):
                                setattr(self, k, torch.cat((getattr(self, k), v), dim=0))
                            elif isinstance(v, list):
                                getattr(self, k).append(v)
                            else:
                                raise NotImplementedError(f"Type of {k} not supported")
                        else:
                            setattr(self, k, v)
                                
                
                
                
                
                ########################## 9. 保存生成的语言(把 token 转成字符串) ##########################
                self.language.append(self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)[0])
        else:
            # single forward pass same as during training so we can use the same function
            features = self.forward_model(driving_input, adaptor_dict)
            outputs_by_adaptor = self.adaptors.split_outputs_by_adaptor(adaptor_dict, features)
            predictions = self.adaptors.driving.get_predictions(outputs_by_adaptor['driving'])

            for k, v in predictions.items():
                if v is not None:
                    setattr(self, k, v)

        return self.speed_wps, self.route, self.language


    
    
    
    
    ########################################### 💡 一、负责“把输入送进模型，得到特征和 logits” 💡 ###########################################
    def forward_model(self,driving_input: DrivingInput,adaptor_dict: Dict,driving_labels: DrivingLabel = None,) -> Tensor:
        
        # 在送入语言Transformer前需要将<IMG_CONTEXT>和<TARGET_POINTS>占位的embedding替换成真正的embedding
        adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict = adaptor_dict,
            pixel_values = driving_input.camera_images,                    # [BS*T,12,3,488,488] 六视角图像
            placeholder_values = driving_input.prompt.placeholder_values,  # {151662: [[x_0,  y_0],[x_1,  y_1]]}
            wp_encoder = self.wp_encoder,                                  # 导航点编码器
        )

        position_ids = None
        adaptor_embeds = adaptor_dict["inputs"]    # 这是最终的完整的embedding，图像的也替换了,target point的也替换了,同时包含了 language 和 driving 的 embedding
        adaptor_mask = adaptor_dict['inputs_mask'] # 对应的mask

        input_embeds = adaptor_embeds
        input_embeds = input_embeds.to(dtype=self.language_model.model.dtype)
        
        attention_mask = adaptor_mask


        # 训练阶段只运行语言模型的Transformer主干。
        # 不再为视觉token、问题token和Driving query生成整词表logits，
        # 从而避免保存巨大的[B, L, vocab_size]张量。
        features = self.language_model.forward_features(
            embeddings=input_embeds,         # 输入语言Transformer的embedding [B,L+20+10,D]
            attention_mask=attention_mask,   # 对应的mask [B,L]
            position_ids=position_ids,
            return_dict=True,
        )

        adaptor_features = features

        # LanguageAdaptor将在真正计算语言loss的位置局部生成logits。
        # DrivingAdaptor本身不使用logits。
        adaptor_logits = None



        return adaptor_features, adaptor_logits
    

    ########################################### 💡 二、训练/验证阶段内部使用的注意力日志函数 💡 ###########################################
    def log_target_point_camera_attention(self, mode: str,) -> None:
        """
        将目标点引导的六视角相机注意力记录到W&B。

        mode:
            train 或 val
        """

        image_encoder = self.vision_model.image_encoder

        target_point_camera_weights = getattr(
            image_encoder,
            "latest_target_point_camera_weights",
            None,
        )

        if not isinstance(
            target_point_camera_weights,
            torch.Tensor,
        ):
            return

        camera_names = (
            "front",
            "front_left",
            "front_right",
            "rear",
            "rear_left",
            "rear_right",
        )

        assert (
            target_point_camera_weights.ndim == 2
            and target_point_camera_weights.shape[-1]
            == len(camera_names)
        ), (
            "Target-point camera attention must have shape "
            f"[B,6], but received "
            f"{tuple(target_point_camera_weights.shape)}."
        )

        mean_camera_weights = (
            target_point_camera_weights.mean(dim=0)
        )

        batch_size = int(
            target_point_camera_weights.shape[0]
        )
        on_step = mode == "train"

        for camera_name, camera_weight in zip(
            camera_names,
            mean_camera_weights,
        ):
            self.log(
                f"{mode}_target_point_attention/"
                f"{camera_name}",
                camera_weight,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        target_point_attention_entropy = getattr(
            image_encoder,
            "latest_target_point_attention_entropy",
            None,
        )

        if isinstance(
            target_point_attention_entropy,
            torch.Tensor,
        ):
            self.log(
                f"{mode}_target_point_attention/"
                "normalized_entropy",
                target_point_attention_entropy.mean(),
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        if hasattr(
            image_encoder,
            "target_point_camera_attention_gate",
        ):
            attention_gate = torch.tanh(
                image_encoder
                .target_point_camera_attention_gate
                .detach()
            ).float()

            self.log(
                f"{mode}_target_point_attention/"
                "camera_scaling_gate",
                attention_gate,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    
    ########################################### 💡 三、训练/验证阶段内部使用的“前向 + 算损失”函数 💡 ###########################################
    def forward_loss(self, example: DrivingExample, per_sample=False) -> TrainingOutput:

        # example 就是一个batch内的所有数据

        adaptor_dict = self.adaptors(example)

        # 这是送入语言Transformer前的输入
        adaptor_embeds = adaptor_dict["inputs"]    # prompt、route和speed_wps的 embedding[B, L+30, D]
        
        # 这是送入语言Transformer前的输入对应的mask,他决定了哪些embedding需要送入,哪些不需要送入
        adaptor_mask = adaptor_dict['inputs_mask'] # prompt、route和speed_wps的有效性[B, L+30] 30全为True

        # 送入网络 获得输出  [B, L+30, D]
        adaptor_features, adaptor_logits = self.forward_model(example.driving_input, adaptor_dict, driving_labels=example.driving_label)
        
        # 计算损失
        loss_dict = self.adaptors.compute_loss(adaptor_features, adaptor_logits, adaptor_dict, example)





        # 是否使用四通道结构化未来世界辅助预测
        if bool(getattr(self,"use_future_interaction_prediction",False,)):
            
            # 安全性检查 检查解码器是否已经创建
            if self.future_interaction_decoder is None:
                raise RuntimeError(
                    "Future interaction prediction is enabled, "
                    "but the decoder was not initialized."
                )

            # 恢复language和driving各自对应的Transformer输出[B, L+30, D] 然后拆解成[B, L, D] [B, 30, D]
            features_by_adaptor = (
                self.adaptors.split_outputs_by_adaptor(
                    adaptor_dict,
                    adaptor_features,
                )
            )

            # route 和 speed_wps 的特征[B, 30, D]
            driving_features = features_by_adaptor.get("driving",None,)

            # 安全性检查
            if not isinstance(driving_features,torch.Tensor,):
                raise RuntimeError(
                    "Driving query features are required for "
                    "future interaction prediction."
                )

            # 解码四通道结构化世界
            future_interaction_logits = (
                self.future_interaction_decoder(
                    driving_features
                )
            )

            future_interaction_target = (
                example
                .driving_label
                .future_interaction_grid
            )

            future_interaction_valid = (
                example
                .driving_label
                .future_interaction_valid
            )

            batch_size = int(
                future_interaction_logits.shape[0]
            )

            # 普通Driving独立batch没有future_interaction_grid
            # 为保持所有batch的loss键一致，构造零目标和零valid掩码。
            if future_interaction_target is None:
                if (
                    isinstance(
                        future_interaction_valid,
                        torch.Tensor,
                    )
                    and bool(
                        future_interaction_valid.any().item()
                    )
                ):
                    raise RuntimeError(
                        "future_interaction_valid contains True, "
                        "but future_interaction_grid is None."
                    )

                future_interaction_target = torch.zeros(
                    (
                        batch_size,
                        4,
                        future_interaction_logits.shape[-2],
                        future_interaction_logits.shape[-1],
                    ),
                    device=(
                        future_interaction_logits.device
                    ),
                    dtype=torch.float32,
                )

                future_interaction_valid = torch.zeros(
                    (batch_size,),
                    device=(
                        future_interaction_logits.device
                    ),
                    dtype=torch.bool,
                )

            elif not isinstance(
                future_interaction_valid,
                torch.Tensor,
            ):
                raise RuntimeError(
                    "A future interaction target exists, but "
                    "future_interaction_valid is missing."
                )

            future_interaction_loss_dict = (
                compute_future_interaction_losses(
                    prediction_logits=(
                        future_interaction_logits
                    ),
                    target_grid=(
                        future_interaction_target
                    ),
                    valid_mask=(
                        future_interaction_valid
                    ),
                    positive_weights=tuple(
                        float(value)
                        for value in getattr(
                            self,
                            "future_interaction_positive_weights",
                            (
                                20.0,
                                20.0,
                                80.0,
                                100.0,
                            ),
                        )
                    ),
                )
            )

            loss_dict.update(
                future_interaction_loss_dict
            )





        #使用LG因果actor投影得到的六维软标签,显式监督目标点引导的六视角相机注意力。
        if bool(
            getattr(
                self,
                "use_lg_camera_attention_supervision",
                False,
            )
        ):
            predicted_camera_weights = adaptor_dict.get(
                "target_point_camera_weights_for_loss",
                None,
            )
            camera_attention_target = (
                example
                .driving_label
                .camera_attention_target
            )
            camera_attention_valid = (
                example
                .driving_label
                .camera_attention_valid
            )

            if not isinstance(
                predicted_camera_weights,
                torch.Tensor,
            ):
                raise RuntimeError(
                    "LG camera attention supervision is "
                    "enabled, but target-point camera "
                    "attention weights were not produced. "
                    "Set model.vision_model."
                    "use_target_point_camera_attention=true."
                )

            if (
                not isinstance(
                    camera_attention_target,
                    torch.Tensor,
                )
                or not isinstance(
                    camera_attention_valid,
                    torch.Tensor,
                )
            ):
                raise RuntimeError(
                    "LG camera attention supervision is "
                    "enabled, but the batch does not contain "
                    "camera attention labels."
                )

            predicted_camera_weights = (
                predicted_camera_weights.float()
            )
            camera_attention_target = (
                camera_attention_target.to(
                    device=(
                        predicted_camera_weights.device
                    ),
                    dtype=torch.float32,
                )
            )
            camera_attention_valid = (
                camera_attention_valid.to(
                    device=(
                        predicted_camera_weights.device
                    ),
                    dtype=torch.bool,
                )
            )

            if (
                predicted_camera_weights.ndim != 2
                or predicted_camera_weights.shape[-1] != 6
            ):
                raise RuntimeError(
                    "Predicted camera attention must have "
                    "shape [B,6], but received "
                    f"{tuple(predicted_camera_weights.shape)}."
                )

            if (
                camera_attention_target.shape
                != predicted_camera_weights.shape
            ):
                raise RuntimeError(
                    "LG camera attention target shape does "
                    "not match prediction shape: "
                    f"{tuple(camera_attention_target.shape)} "
                    "vs "
                    f"{tuple(predicted_camera_weights.shape)}."
                )

            if camera_attention_valid.shape != (
                predicted_camera_weights.shape[0],
            ):
                raise RuntimeError(
                    "LG camera attention valid mask must "
                    "have shape [B], but received "
                    f"{tuple(camera_attention_valid.shape)}."
                )

            target_sum = (
                camera_attention_target.sum(dim=-1)
            )

            invalid_valid_target = (
                camera_attention_valid
                & (target_sum <= 0.0)
            )
            if bool(
                invalid_valid_target.any().item()
            ):
                raise RuntimeError(
                    "A valid LG camera attention target has "
                    "a non-positive probability sum."
                )

            # 无效样本保持全零，不参与该损失。
            normalized_target = torch.where(
                camera_attention_valid.unsqueeze(-1),
                camera_attention_target
                / target_sum.clamp_min(1e-8).unsqueeze(-1),
                torch.zeros_like(
                    camera_attention_target
                ),
            )

            # 软标签交叉熵，每个样本得到一个标量。
            per_sample_camera_attention_loss = -(
                normalized_target
                * predicted_camera_weights
                .clamp_min(1e-8)
                .log()
            ).sum(dim=-1)

            camera_attention_loss_count = (
                camera_attention_valid.float()
            )

            #修改20260721：记录当前batch中真正参与
            # LG相机注意力监督的样本数量和比例。
            valid_attention_count = (
                camera_attention_loss_count.sum()
            )
            valid_attention_ratio = (
                valid_attention_count
                / max(
                    int(
                        camera_attention_loss_count.numel()
                    ),
                    1,
                )
            )

            attention_log_mode = (
                "train"
                if self.training
                else "val"
            )

            self.log(
                f"{attention_log_mode}_lg_camera_attention/"
                "valid_sample_count",
                valid_attention_count,
                on_step=self.training,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=int(
                    camera_attention_loss_count.numel()
                ),
                sync_dist=True,
            )

            self.log(
                f"{attention_log_mode}_lg_camera_attention/"
                "valid_sample_ratio",
                valid_attention_ratio,
                on_step=self.training,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=int(
                    camera_attention_loss_count.numel()
                ),
                sync_dist=True,
            )

            loss_dict[
                "lg_camera_attention_loss"
            ] = (
                per_sample_camera_attention_loss
                * camera_attention_loss_count,
                camera_attention_loss_count,
            )






        loss_dict_only_losses = {k:v for k, v in loss_dict.items() if k.endswith("loss")}
        loss_logs = {k:v for k, v in loss_dict.items() if k.endswith("log")}
        
        # pred_labels = {k:v for k, v in loss_dict.items() if not k.endswith("loss") and not k.endswith("log")}
        # if per_sample:
        #     return loss_dict_only_losses, pred_labels

        pred_labels = {
            k: v
            for k, v in loss_dict.items()
            if not k.endswith("loss") and not k.endswith("log")
        }

        image_encoder = self.vision_model.image_encoder

        target_point_camera_weights = getattr(
            image_encoder,
            "latest_target_point_camera_weights",
            None,
        )

        if isinstance(
            target_point_camera_weights,
            torch.Tensor,
        ):
            pred_labels[
                "target_point_camera_weights"
            ] = target_point_camera_weights

        target_point_attention_entropy = getattr(
            image_encoder,
            "latest_target_point_attention_entropy",
            None,
        )

        if isinstance(
            target_point_attention_entropy,
            torch.Tensor,
        ):
            pred_labels[
                "target_point_attention_entropy"
            ] = target_point_attention_entropy

        # 训练和验证都记录六视角注意力统计。
        self.log_target_point_camera_attention(
            "train" if self.training else "val"
        )

        if per_sample:
            # LocalValidationMetricsCallback需要保存每个样本真正参与
            # 当前前向传播的<TARGET_POINT>坐标。
            #
            # 这里根据language token序列判断当前prompt是否实际包含
            # <TARGET_POINT>，而不是仅根据placeholder_values中是否存在该值判断。
            # 因此，无论placeholder_values是否预先保存了目标点，
            # 只有prompt真正使用<TARGET_POINT>时has_target_point才为True。
            language_ids = adaptor_dict.get(
                "language__ids"
            )

            if not isinstance(language_ids, torch.Tensor):
                raise RuntimeError(
                    "language__ids is required to save "
                    "<TARGET_POINT> coordinates."
                )

            target_point_token_id = getattr(
                image_encoder,
                "target_point_token_id",
                None,
            )

            if target_point_token_id is None:
                raise RuntimeError(
                    "target_point_token_id was not initialized by "
                    "the InternVL2 image encoder."
                )

            target_point_token_id = int(
                target_point_token_id
            )

            # [B]，表示每个样本的当前prompt中是否真正使用了
            # <TARGET_POINT>。
            has_target_point = (
                language_ids == target_point_token_id
            ).any(dim=1)

            batch_size = int(language_ids.shape[0])

            # 当前<TARGET_POINT>实际包含两个二维导航点：
            # 第一个是target_point，第二个是target_point_next。
            # 因此batch内的保存形状为[B, 2, 2]。
            target_point_coordinates = torch.full(
                (
                    batch_size,
                    2,
                    2,
                ),
                float("nan"),
                device=language_ids.device,
                dtype=torch.float32,
            )

            placeholder_values = (
                example.driving_input.prompt.placeholder_values
            )

            if len(placeholder_values) != batch_size:
                raise RuntimeError(
                    "The number of placeholder value dictionaries "
                    "does not match the batch size: "
                    f"{len(placeholder_values)} vs {batch_size}"
                )

            # 从原始placeholder_values中读取未经embedding编码的
            # <TARGET_POINT>自车坐标，最终保存形状为[B, 2]。
            for sample_index in range(batch_size):
                if not bool(
                    has_target_point[sample_index].item()
                ):
                    continue

                sample_placeholder_values = (
                    placeholder_values[sample_index]
                )

                if (
                    target_point_token_id
                    not in sample_placeholder_values
                ):
                    raise KeyError(
                        "The prompt contains <TARGET_POINT>, but "
                        "its coordinate is missing from "
                        "placeholder_values."
                    )

                sample_target_points = torch.as_tensor(
                    sample_placeholder_values[
                        target_point_token_id
                    ],
                    device=language_ids.device,
                    dtype=torch.float32,
                )

                # 一个导航条件包含两个二维点：
                # [
                #     [target_point_x, target_point_y],
                #     [next_target_point_x, next_target_point_y],
                # ]
                if sample_target_points.numel() != 4:
                    raise ValueError(
                        "<TARGET_POINT> must contain two 2D "
                        "coordinates, but received shape "
                        f"{tuple(sample_target_points.shape)} with "
                        f"{sample_target_points.numel()} values."
                    )

                sample_target_points = (
                    sample_target_points.reshape(2, 2)
                )

                target_point_coordinates[
                    sample_index
                ] = sample_target_points

            pred_labels[
                "target_point_coordinates"
            ] = target_point_coordinates

            pred_labels[
                "has_target_point"
            ] = has_target_point

            return loss_dict_only_losses, pred_labels

        # return summarise_losses(loss_dict_only_losses), loss_logs

        #修改20260726：统一整理所有辅助任务的损失权重。
        # 原有语言、waypoint和route损失仍保持默认权重1.0。
        loss_weights = {}

        if bool(
            getattr(
                self,
                "use_lg_camera_attention_supervision",
                False,
            )
        ):
            loss_weights[
                "lg_camera_attention_loss"
            ] = float(
                getattr(
                    self,
                    "lg_camera_attention_loss_weight",
                    0.05,
                )
            )

        if bool(
            getattr(
                self,
                "use_future_interaction_prediction",
                False,
            )
        ):
            channel_weights = tuple(
                float(value)
                for value in getattr(
                    self,
                    "future_interaction_channel_weights",
                    (
                        1.0,
                        1.0,
                        2.0,
                        4.0,
                    ),
                )
            )

            if len(channel_weights) != 4:
                raise ValueError(
                    "future_interaction_channel_weights "
                    "must contain exactly 4 values."
                )

            future_interaction_loss_weight = float(
                getattr(
                    self,
                    "future_interaction_loss_weight",
                    0.05,
                )
            )

            future_interaction_dice_loss_weight = float(
                getattr(
                    self,
                    "future_interaction_dice_loss_weight",
                    1.0,
                )
            )

            for channel_key, channel_weight in zip(
                FUTURE_INTERACTION_CHANNEL_KEYS,
                channel_weights,
            ):
                loss_weights[
                    f"future_interaction_"
                    f"{channel_key}_bce_loss"
                ] = (
                    future_interaction_loss_weight
                    * channel_weight
                )

                loss_weights[
                    f"future_interaction_"
                    f"{channel_key}_dice_loss"
                ] = (
                    future_interaction_loss_weight
                    * channel_weight
                    * future_interaction_dice_loss_weight
                )

        if len(loss_weights) == 0:
            loss_weights = None

        return summarise_losses(
            loss_dict_only_losses,
            weights=loss_weights,
        ), loss_logs

    
    
    
    
    
    
    
    ########################################### 1. 训练时每个batch的总入口 ###########################################
    def training_step(self, batch: DrivingExample, _batch_idx: int = 0):
        output, loss_logs = self.forward_loss(batch)
        logs = output
        self.log_training_output(logs, "train")

        # log the loss
        self.log("train/loss", output.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": output.loss, "outputs": output}

    
    
    
    
    
    
    ########################################### 2. 验证时每个batch的总入口 ###########################################
    def validation_step(self, batch: DrivingExample, _batch_idx: int = 0):
        
        output, loss_logs = self.forward_loss(batch)
        logs = output #.update(loss_logs)
        self.log_training_output(logs, "val")

        # log the loss
        self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": output.loss, "outputs": output}

    
    
    
    
    

    
    ########################################### 3. 推理时每个batch的总入口 ###########################################
    def predict_step(self, batch: DrivingExample, _batch_idx: int = 0):
        run_ids = decode_uint8(batch.run_id)
        
        speed_wps, route, language = self.forward(batch, return_language=True)

        self.num_route_points = 20
        route_equal = []
        for i in range(len(route)):
            route_equal.append(self.equal_spacing_route(route[i].cpu()))
        route_equal = torch.tensor(route_equal)
        route = route_equal.to(route.device)
        
        
        route_gt = batch.driving_label.path
        speed_wps_gt = batch.driving_label.waypoints
        language_gt = batch.driving_label.answer.language_string
        
        if len(self.prediction) == 0:
            self.prediction = {
                "waypoints": [speed_wps],
                "route": [route],
                "language": language,
                "waypoints_gt": [speed_wps_gt],
                "route_gt": [route_gt],
                "language_gt": language_gt,
                "prompt": batch.driving_input.prompt.language_string,
                "path": run_ids,
                "qa_templates": batch.qa_templates,
                "eval_infos": batch.driving_label.eval_infos,
            }
        else:
            self.prediction["waypoints"].append(speed_wps)
            self.prediction["route"].append(route)
            self.prediction["language"].extend(language)
            self.prediction["waypoints_gt"].append(speed_wps_gt)
            self.prediction["route_gt"].append(route_gt)
            self.prediction["language_gt"].extend(language_gt)
            self.prediction["prompt"].extend(batch.driving_input.prompt.language_string)
            self.prediction["path"].extend(run_ids)
            self.prediction["qa_templates"].extend(batch.qa_templates)
            self.prediction["eval_infos"].extend(batch.driving_label.eval_infos)
            
        
        return speed_wps, route, language, speed_wps_gt, route_gt, language_gt


















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

    def on_predict_epoch_end(self) -> None:    

        repo_path = get_original_cwd()

        if self.trainer.ckpt_path is not None:
            ckpt_path = Path(self.trainer.ckpt_path).parent.parent
        else:
            ckpt_path = Path(f'{repo_path}/outputs/{self.language_model.variant}')
        save_prediction_path = ckpt_path / "predictions"
        save_prediction_path.mkdir(exist_ok=True, parents=True)
        
        samples_cot = [i for i, l in enumerate(self.prediction["prompt"]) if "What should the ego do next?" in l]
        samples_qa = [i for i, l in enumerate(self.prediction["prompt"]) if "Q:" in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]
        language = [(l, l_gt, p) for l, l_gt, p in zip(self.prediction["language"], self.prediction["language_gt"], self.prediction["path"])]
        
        if len(samples_qa) > 0:
            # sort by templates
            sorted_samples = {} # question: {answer: [language, language_gt]}
            for qa_template, language_sample in zip(self.prediction["qa_templates"], language):
                question = qa_template[0]
                answer = qa_template[1]
                if question not in sorted_samples:
                    sorted_samples[question] = {}
                if answer not in sorted_samples[question]:
                    sorted_samples[question][answer] = []
                sorted_samples[question][answer].append(language_sample)
            
            if os.path.exists(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json"):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}_{time}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
            else:
                with open(f"{str(save_prediction_path)}/sorted_qa_templates_rank_{self.local_rank}.json", "w") as f:
                    json.dump(sorted_samples, f, indent=4)
        
        for samples, name in zip([samples_cot, samples_qa, samples_all], ["cot", "qa", "all"]):
            language_samples = [l for i, l in enumerate(language) if i in samples]
        
            # save language predictions
            save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/language_preds_{name}_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(language_samples, f, indent=4)
            
        route_preds = self.prediction["route"]
        route_preds = torch.cat(route_preds, dim=0)
        route_gt = self.prediction["route_gt"]
        route_gt = torch.cat(route_gt, dim=0)
        
        waypoints_preds = self.prediction["waypoints"]
        waypoints_preds = torch.cat(waypoints_preds, dim=0)
        waypoints_gt = self.prediction["waypoints_gt"]
        waypoints_gt = torch.cat(waypoints_gt, dim=0)
        
        # calc distance between wps for 1d wps
        waypoints_preds_1d = []
        for i in range(len(waypoints_preds)):
            waypoint_pred = waypoints_preds[i]
            waypoints_preds_1d_tmp = torch.tensor([torch.linalg.norm(waypoint_pred[i+1] - waypoint_pred[i]) for i in range(len(waypoint_pred)-1)])
            # cumsum to get the distance from the start
            waypoints_preds_1d_tmp = torch.cumsum(waypoints_preds_1d_tmp, dim=0)
            waypoints_preds_1d_tmp = [[x, 0] for x in waypoints_preds_1d_tmp]
            waypoints_preds_1d.append(waypoints_preds_1d_tmp)
        waypoints_preds_1d = torch.tensor(waypoints_preds_1d)
        
        waypoints_gt_1d = []
        for i in range(len(waypoints_gt)):
            waypoint_gt = waypoints_gt[i]
            waypoints_gt_1d_tmp = torch.tensor([torch.linalg.norm(waypoint_gt[i+1] - waypoint_gt[i]) for i in range(len(waypoint_gt)-1)])
            # cumsum to get the distance from the start
            waypoints_gt_1d_tmp = torch.cumsum(waypoints_gt_1d_tmp, dim=0)
            waypoints_gt_1d_tmp = [[x, 0] for x in waypoints_gt_1d_tmp]
            waypoints_gt_1d.append(waypoints_gt_1d_tmp)
        waypoints_gt_1d = torch.tensor(waypoints_gt_1d)
        
        # calculate ade and fde for samples seperatly whihc have <SAFETY> in prompt and for <INSTRUCTION_FOLLOWING>
        samples_safety = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" in l]
        samples_instruction = [i for i, l in enumerate(self.prediction["prompt"]) if "<INSTRUCTION_FOLLOWING>" in l]
        samples_neither = [i for i, l in enumerate(self.prediction["prompt"]) if "<SAFETY>" not in l and "<INSTRUCTION_FOLLOWING>" not in l]
        samples_all = [i for i in range(len(self.prediction["prompt"]))]
        
        ade_fde = {}
        
        def get_desired_end_speed(wps):
            wp_freq = 5
            carla_fps = 20
            # one WP every 0.25 seconds
            # we want to get speed at the last WP
            last_wp = wps[-1] #.cpu().numpy()
            # we want the WP half second earlier than the last WP
            one_second = int(carla_fps // (wp_freq))
            half_second = one_second // 2
            prev_wp = wps[-1 - half_second] #.cpu().numpy()
            desired_speed = np.linalg.norm(prev_wp - last_wp) * 2.0
            return desired_speed
        def get_desired_speed(wps):
            wp_freq = 5
            carla_fps = 20
            # one WP every 0.25 seconds
            # we want to get speed at the last WP
            # last_wp = wps[-1] #.cpu().numpy()
            # we want the WP half second earlier than the last WP
            one_second = int(carla_fps // (wp_freq))
            half_second = one_second // 2
            wp_half_second = wps[half_second] #.cpu().numpy()
            wp_one_second = wps[one_second] #.cpu().numpy()
            desired_speed = np.linalg.norm(wp_half_second - wp_one_second) * 2.0
            return desired_speed
        
        def get_desired_avg_speed(wps):
            wp_freq = 5
            carla_fps = 20
            # one WP every 0.25 seconds
            # we want to get speed at the last WP
            # last_wp = wps[-1] #.cpu().numpy()
            # # we want the WP half second earlier than the last WP
            # one_second = int(carla_fps // (wp_freq))
            # half_second = one_second // 2
            first_wp = wps[0] #.cpu().numpy()
            last_wp = wps[-1] #.cpu().numpy()
            desired_speed = np.linalg.norm(first_wp - last_wp) / (len(wps) * 0.25)
            return desired_speed
        
        def get_1d_wps(wps):
            waypoints_1d = [np.linalg.norm(wps[i+1] - wps[i]) for i in range(len(wps)-1)]
            # cumsum to get the distance from the start
            waypoints_1d = np.cumsum(waypoints_1d)
            waypoints_1d = [[x, 0] for x in waypoints_1d]
            
            # prepend 0,0
            waypoints_1d = [[0, 0]] + waypoints_1d
            
            return np.array(waypoints_1d).reshape(-1, 2)
        
            
        wp_freq = 5
        carla_fps = 20
            
        
        for samples, name in zip([samples_safety, samples_instruction, samples_neither, samples_all], ["instruction"]):
            if len(samples) == 0:
                continue
            route_preds_sample = route_preds[samples].cpu().numpy()
            route_gt_sample = route_gt[samples].cpu().numpy()
            waypoints_preds_sample = waypoints_preds[samples].cpu().numpy()
            waypoints_gt_sample = waypoints_gt[samples].cpu().numpy()
            eval_infos_sample = [self.prediction["eval_infos"][i] for i in samples]
            waypoints_org_sample = [eval_infos_sample[i]["org_wps"] for i in range(len(samples))]
            route_org_sample = [eval_infos_sample[i]["org_path"] for i in range(len(samples))]
            waypoints_instruction_sample = [np.array(eval_infos_sample[i]["new_wps"]) for i in range(len(samples))]
            route_instruction_sample = [eval_infos_sample[i]["new_path"] for i in range(len(samples))]
            prompts = [self.prediction["prompt"][i].replace("<IMG_CONTEXT>", "") for i in samples]
            pred_language = [self.prediction["language"][i] for i in samples]
            paths = [self.prediction["path"][i] for i in samples]
            
            success_rate_all = []
            success_rate_by_mode = {}
            success_rate_by_allowed = {}
            
            paths_by_mode = {}
            
            for i in range(len(samples)):
                mode = eval_infos_sample[i]["mode"]
                allowed = eval_infos_sample[i]['allowed']
                sample_path = paths[i]
                
                # Initialize mode in dictionary if not present
                if mode not in success_rate_by_mode:
                    success_rate_by_mode[mode] = []
                if mode not in paths_by_mode:
                    paths_by_mode[mode] = []
                
                # Initialize allowed in dictionary if not present
                if allowed not in success_rate_by_allowed:
                    success_rate_by_allowed[allowed] = []
                
                # get desired speed form WPs
                desired_end_speed_pred = get_desired_end_speed(waypoints_preds_sample[i])
                desired_end_speed_gt = get_desired_end_speed(waypoints_gt_sample[i])
                desired_end_speed_org = get_desired_end_speed(waypoints_org_sample[i])
                desired_end_speed_instruction = get_desired_end_speed(waypoints_instruction_sample[i])
                
                desired_speed_pred = get_desired_speed(waypoints_preds_sample[i])
                desired_speed_gt = get_desired_speed(waypoints_gt_sample[i])
                desired_speed_org = get_desired_speed(waypoints_org_sample[i])
                desired_speed_instruction = get_desired_speed(waypoints_instruction_sample[i])
                
                desired_avg_speed_pred = get_desired_avg_speed(waypoints_preds_sample[i])
                desired_avg_speed_gt = get_desired_avg_speed(waypoints_gt_sample[i])
                desired_avg_speed_org = get_desired_avg_speed(waypoints_org_sample[i])
                desired_avg_speed_instruction = get_desired_avg_speed(waypoints_instruction_sample[i])

                
                pred_wps_1d = get_1d_wps(waypoints_preds_sample[i])
                pred_wps_1d_diffs = np.diff(pred_wps_1d[:, 0])
                pred_speeds = pred_wps_1d_diffs / (wp_freq/carla_fps)
                
                org_wps_1d = get_1d_wps(waypoints_org_sample[i])
                org_wps_1d_diffs = np.diff(org_wps_1d[:, 0])
                org_speeds = org_wps_1d_diffs / (wp_freq/carla_fps)
                
                instruction_wps_1d = get_1d_wps(waypoints_instruction_sample[i])
                instruction_wps_1d_diffs = np.diff(instruction_wps_1d[:, 0])
                instruction_speeds = instruction_wps_1d_diffs / (wp_freq/carla_fps)
                
                x = np.arange(len(pred_speeds))*0.25
                
                # linear regression np
                slope_pred, intercept_pred = np.polyfit(x, pred_speeds, 1)
                slope_org, intercept_org = np.polyfit(x, org_speeds, 1)
                slope_instruction, intercept_instruction = np.polyfit(x, instruction_speeds, 1)
                
                current_speed = float(prompts[i].split("Current speed: ")[-1].split(" ")[0])
                
                if mode == 'stop':
                    # route doesnt matter
                    paths_by_mode[mode].append(sample_path)
                    if name == 'instruction' or name == 'neither':
                        if np.min(pred_speeds) < 0.1:
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                            
                elif mode == 'slower':
                    paths_by_mode[mode].append(sample_path)

                    if name == 'instruction' or name == 'neither':
                        # forced instruction following
                        if slope_pred < (-0.05 * current_speed):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                        
                elif mode == 'faster':
                    paths_by_mode[mode].append(sample_path)
                    
                    if name == 'instruction' or name == 'neither':
                        # forced instruction following
                        if slope_pred > (0.05 * current_speed):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                elif mode == 'target_speed':
                    paths_by_mode[mode].append(sample_path)
                    
                    try:
                        target_speed = float(prompts[i].split("Target waypoint: ")[-1].split("Command")[-1].split(".<|im_end|>")[0].split(" ")[-2])
                    except:
                        target_speed = float(prompts[i].split("Target waypoint: ")[-1].split("Command")[-1].split(".<|im_end|>")[0].split(" ")[-3])
                    # ade from pred to instruction WP should be closer than to the GT WP
                    # ade_pred_org = np.mean(np.linalg.norm(waypoints_preds_sample[i] - waypoints_org_sample[i], axis=-1))
                    # ade_pred_instruction = np.mean(np.linalg.norm(waypoints_preds_sample[i] - waypoints_instruction_sample[i], axis=-1))
                    if name == 'instruction' or name == 'neither':
                        if ((desired_end_speed_pred > 0.8 * desired_end_speed_instruction and desired_end_speed_pred < 1.2 * desired_end_speed_instruction) or (desired_end_speed_pred > 0.8 * target_speed and desired_end_speed_pred < 1.2 * target_speed)):
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                    
                elif mode == 'lane_change':
                    paths_by_mode[mode].append(sample_path)
                    
                    # on path
                    fde_pred_org = np.linalg.norm(route_preds_sample[i][-1] - route_org_sample[i][-1], axis=-1)
                    fde_pred_instruction = np.linalg.norm(route_preds_sample[i][-1] - route_instruction_sample[i][-1], axis=-1)
                    if name == 'instruction' or name == 'neither':
                        if fde_pred_instruction < fde_pred_org:
                            success_rate_all.append(1)
                            success_rate_by_mode[mode].append(1)
                            success_rate_by_allowed[allowed].append(1)
                        else:
                            success_rate_all.append(0)
                            success_rate_by_mode[mode].append(0)
                            success_rate_by_allowed[allowed].append(0)
                elif mode == 'crash':
                    paths_by_mode[mode].append(sample_path)
                    
                    ade_path_org_instruction = np.mean(np.linalg.norm(route_org_sample[i] - route_instruction_sample[i], axis=-1))
                    ade_path_pred_org = np.mean(np.linalg.norm(route_preds_sample[i] - route_org_sample[i], axis=-1))
                    ade_path_pred_instruction = np.mean(np.linalg.norm(route_preds_sample[i] - route_instruction_sample[i], axis=-1))
                    if ade_path_org_instruction > 1.0:
                        if name == 'instruction' or name == 'neither':
                            if ade_path_pred_instruction < ade_path_pred_org:
                                success_rate_all.append(1)
                                success_rate_by_mode[mode].append(1)
                                success_rate_by_allowed[allowed].append(1)
                            else:
                                success_rate_all.append(0)
                                success_rate_by_mode[mode].append(0)
                                success_rate_by_allowed[allowed].append(0)
                    else:
                        if name == 'instruction' or name == 'neither':
                            if ade_path_pred_instruction < 1.0 and (np.mean(pred_speeds) < 1.3 * np.mean(instruction_speeds) or np.mean(pred_speeds) > 0.7 * np.mean(instruction_speeds)):
                                success_rate_all.append(1)
                                success_rate_by_mode[mode].append(1)
                                success_rate_by_allowed[allowed].append(1)
                            else:
                                success_rate_all.append(0)
                                success_rate_by_mode[mode].append(0)
                                success_rate_by_allowed[allowed].append(0)

                else:
                    print(f"Unknown mode: {mode} in sample {i} with path {sample_path}")
                                
            # save result per sample
            per_sample_results = {
                'paths_by_mode': paths_by_mode,
                'success_rate_by_mode': success_rate_by_mode
            }
            save_path_tmp = f"{str(save_prediction_path)}/results_per_sample_{name}_rank_{self.local_rank}.json"
            if os.path.exists(save_path_tmp):
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path_tmp = f"{str(save_prediction_path)}/results_per_sample_{name}_rank_{self.local_rank}_{time}.json"
            with open(save_path_tmp, "w") as f:
                json.dump(per_sample_results, f, indent=4)
            
            if len(success_rate_all) > 0:
                total_success_rate = sum(success_rate_all) / len(success_rate_all)
                ade_fde.update({f"success_rate_total_{name}": total_success_rate})
            else:
                ade_fde.update({f"success_rate_total_{name}": 0})
                
            min_samples_per_mode = min([len(success_rate_by_mode[mode]) for mode in success_rate_by_mode])
            balanced_total_success_rate = 0
            # Calculate success rate for each mode
            for mode in success_rate_by_mode:
                if len(success_rate_by_mode[mode]) > 0:
                    success_rate = sum(success_rate_by_mode[mode]) / len(success_rate_by_mode[mode])
                    ade_fde.update({f"success_rate_{name}_{mode}": success_rate})
                else:
                    ade_fde.update({f"success_rate_{name}_{mode}": 0})
                
            ade_route = np.mean(np.linalg.norm(route_preds_sample - route_gt_sample, axis=-1), axis=-1)
            
            ade_fde.update({
                f"num_samples_{name}": len(ade_route),
            })

        save_path_tmp = f"{str(save_prediction_path)}/dreamer_results_rank_{self.local_rank}.json"
        if os.path.exists(save_path_tmp):
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path_tmp = f"{str(save_prediction_path)}/dreamer_results_rank_{self.local_rank}_{time}.json"
        with open(save_path_tmp, "w") as f:
            json.dump(ade_fde, f, indent=4)
        
    def log_training_output(self, training_output: TrainingOutput, mode: str, dataset: Optional[str] = None):
        losses = {k: n.detach() for k, n in training_output.loss_averages.items()}
        counts = {k: n.detach().sum() for k, n in training_output.loss_counts.items()}
        losses["loss"] = training_output.loss.detach()
        counts["loss"] = 1  # loss is already averaged
        for k, v in sorted(losses.items()):
            log_key = f"{mode}_losses/{k}"
            self.log(log_key, v, batch_size=counts[k], sync_dist=True, add_dataloader_idx=False)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=max_steps,
            pct_start=self.pct_start,
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }