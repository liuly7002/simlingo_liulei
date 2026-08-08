# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from simlingo_training.models.driving import DrivingModel
from simlingo_training.models.future_interaction import (
    FUTURE_INTERACTION_CHANNEL_KEYS,
    compute_future_interaction_losses,
)
from simlingo_training.models.interaction_reasoning import (
    INTERACTION_TOKEN_KEYS,
    PreLanguageInteractionReasoner,
    compute_actor_disentanglement_losses,
    compute_counterfactual_attention_suppression_loss,
    compute_counterfactual_future_world_losses,
    compute_participant_spatial_attention_losses,
)
from simlingo_training.models.utils import summarise_losses
from simlingo_training.utils.custom_types import (
    DrivingExample,
    DrivingOutput,
    TrainingOutput,
)


class InteractionGroundedDrivingModel(DrivingModel):
    """
    统一语言—交互—动作模型，并执行参与者删除反事实训练。

    序列固定为：
        [4个统一交互token | 语言token | 30个Driving query]

    因此语言生成、轨迹预测和四通道未来世界预测不再通过相互独立的
    后处理分支读取交互信息，而是在同一次因果Transformer前向中共享证据。
    """

    def __init__(self,cfg_data_module,processor,cache_dir,**cfg,):
        
        super().__init__(cfg_data_module=cfg_data_module,processor=processor,cache_dir=cache_dir,**cfg,)

        driving_adaptor = self.adaptors.driving
        if driving_adaptor is None:
            raise RuntimeError(
                "Stage-2 interaction reasoning requires DrivingAdaptor."
            )

        # 构造四个交互token [B,4,D]
        self.interaction_reasoner = PreLanguageInteractionReasoner(
            hidden_size=self.language_model.hidden_size,
            num_route_queries=int(
                getattr(driving_adaptor, "future_waypoints", 20)
            ),
            num_ego_queries=int(
                getattr(
                    driving_adaptor,
                    "future_speed_waypoints",
                    10,
                )
            ),
            num_cameras=6,
            tokens_per_camera=64,
            attention_dim=int(
                getattr(self, "interaction_attention_dim", 128)
            ),
            num_heads=int(
                getattr(self, "interaction_num_heads", 8)
            ),
            dropout=float(
                getattr(self, "interaction_dropout", 0.1)
            ),
        )

        self.future_interaction_logits = None
        self.primary_spatial_attention = None
        self.secondary_spatial_attention = None
        self.contextual_interaction_tokens = None

        self._question_marker_patterns = tuple(
            self._tokenize_marker_variants(f"A{index}:")
            for index in range(1, 5)
        )
        self._waypoint_marker_patterns = (
            self._tokenize_marker_variants("Waypoints:")
        )

    def _tokenize_marker_variants(
        self,
        marker: str,) -> Tuple[Tuple[int, ...], ...]:
        variants: List[Tuple[int, ...]] = []
        for text in (marker, f" {marker}", f"\n{marker}"):
            token_ids = tuple(
                int(value)
                for value in self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )
            if token_ids and token_ids not in variants:
                variants.append(token_ids)
        if not variants:
            raise RuntimeError(
                f"Tokenizer produced no ids for marker {marker!r}."
            )
        return tuple(variants)

    @staticmethod
    def _append_tensor(
        current: Optional[Tensor],
        value: Tensor,) -> Tensor:
        if current is None:
            return value
        return torch.cat((current, value), dim=0)

    def _model_dtype(self) -> torch.dtype:
        return self.adaptors.language.embed_tokens.weight.dtype

    def _replace_multimodal_placeholders(self,example: DrivingExample,*,inference: bool,) -> Dict:
        """
        
        """

        # 创建language和driving adaptor
        adaptor_dict = self.adaptors(example,inference=inference,)
        
        # 获取batch输入数据中的"问题+答案"的prompt
        prompt = (example.driving_input.prompt_inference if inference else example.driving_input.prompt)
        
        # 将<IMG_CONTEXT>和<TARGET_POINT>占位embedding替换为真正的embedding  [B,L,D]
        return self.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=adaptor_dict,
            pixel_values=example.driving_input.camera_images,
            placeholder_values=prompt.placeholder_values,
            wp_encoder=self.wp_encoder,
        )

    def _build_counterfactual_adaptor_dict(
        self,
        example: DrivingExample,
        source_adaptor_dict: Dict,) -> Dict:
        """
        构造独立反事实文本序列，并复用完整场景已经编码的视觉与导航
        placeholder特征，避免第二次运行视觉编码器。
        """
        counterfactual_prompt = (
            example.driving_input.counterfactual_prompt
        )
        if counterfactual_prompt is None:
            raise RuntimeError(
                "Counterfactual intervention training requires "
                "driving_input.counterfactual_prompt."
            )

        counterfactual_input = example.driving_input._replace(
            prompt=counterfactual_prompt
        )
        counterfactual_example = example._replace(
            driving_input=counterfactual_input
        )
        counterfactual_dict = self.adaptors(
            counterfactual_example,
            inference=False,
        )

        image_encoder = self.vision_model.image_encoder
        placeholder_token_ids = (
            getattr(image_encoder, "img_context_token_id", None),
            getattr(image_encoder, "target_point_token_id", None),
        )
        if any(value is None for value in placeholder_token_ids):
            raise RuntimeError(
                "Image and target-point token ids must be initialized "
                "before counterfactual text construction."
            )

        source_ids = source_adaptor_dict["language__ids"]
        target_ids = counterfactual_dict["language__ids"]
        source_features = source_adaptor_dict["language_inputs"]
        target_features = counterfactual_dict["language_inputs"].clone()
        batch_size = int(source_ids.shape[0])

        for token_id in placeholder_token_ids:
            source_mask = source_ids == int(token_id)
            target_mask = target_ids == int(token_id)
            source_count = source_mask.sum(dim=1)
            target_count = target_mask.sum(dim=1)
            if not torch.equal(source_count, target_count):
                raise RuntimeError(
                    "Original and counterfactual prompts contain different "
                    f"numbers of placeholder token id {int(token_id)}: "
                    f"{source_count.tolist()} vs {target_count.tolist()}."
                )
            count = int(source_count[0].item())
            if not torch.all(source_count == count):
                raise RuntimeError(
                    "Placeholder counts must be constant within a batch."
                )
            if count == 0:
                continue
            replacement = source_features[source_mask].reshape(
                batch_size,
                count,
                source_features.shape[-1],
            )
            target_features[target_mask] = replacement.reshape(
                -1,
                replacement.shape[-1],
            ).to(dtype=target_features.dtype)

        counterfactual_dict["language_inputs"] = target_features
        return counterfactual_dict

    def _build_pre_language_interaction(self,adaptor_dict: Dict,*,batch_slice=slice(None),visual_intervention_target: Optional[Tensor] = None,visual_intervention_valid: Optional[Tensor] = None,) -> Dict[str, Tensor]:

        """
        核心任务，就是从已经完成多模态占位符替换的 Language 序列中,
        把视觉证据、导航证据和允许读取的语言上下文重新拆出来,
        再结合 30 个 Driving queries,
        构造进入 Language Transformer 之前的 4 个统一 interaction tokens;
        在反事实情况下，它还负责先对视觉证据实施主要 actor 删除干预.
        """

        # [B,L,D] 这是已经完成<IMG_CONTEXT>和<TARGET_POINT>占位embedding替换的语言embedding序列,这里我们可以称之为多模态Language embeddign序列
        raw_language_features = adaptor_dict["language_inputs"][batch_slice]
        
        # [B,30,D] 这是driving的30个可学习query
        raw_driving_queries = adaptor_dict["driving_inputs"][batch_slice]
        
        # [B,L] 语言token id
        language_ids = adaptor_dict["language__ids"][batch_slice]
        
        # [B,L] 语言有效位置,解决的是padding的问题
        language_valid = adaptor_dict["language_inputs_mask"][batch_slice].bool()

        # [B,L] 哪些位置的语言token是答案
        answer_mask = adaptor_dict.get("language__ids_mask",None,)
        if isinstance(answer_mask, torch.Tensor):
            answer_mask = answer_mask[batch_slice].bool()
        else:
            answer_mask = torch.zeros_like(language_valid)

        # 视觉编码器
        image_encoder = self.vision_model.image_encoder

        # 获取<IMG_CONTEXT>对应的token id
        image_context_token_id = getattr(image_encoder,"img_context_token_id",None,)
        
        # 获取<TARGET_POINT>对应的token id
        target_point_token_id = getattr(image_encoder,"target_point_token_id",None,)
        
        # 安全性检查
        if image_context_token_id is None:
            raise RuntimeError(
                "The image encoder did not initialize IMG_CONTEXT token id."
            )

        # 安全性检查
        if target_point_token_id is None:
            raise RuntimeError(
                "The image encoder did not initialize TARGET_POINT token id."
            )

        # 384个视觉token [B,L]
        visual_mask = (language_ids == int(image_context_token_id))
        
        # 384
        expected_visual_tokens = int(self.interaction_reasoner.num_visual_tokens)
        
        # 384
        visual_token_count = visual_mask.sum(dim=1)

        # 安全性检查
        if not torch.all(visual_token_count == expected_visual_tokens):
            raise RuntimeError(
                "Stage-2 interaction reasoning requires exactly "
                f"{expected_visual_tokens} visual tokens per sample, "
                f"but received counts={visual_token_count.tolist()}."
            )

        # 真正把视觉embedding从多模态Language embedding中抽出来 [B,384,D] 
        visual_features = raw_language_features[
            visual_mask
        ].reshape(
            raw_language_features.shape[0],
            expected_visual_tokens,
            raw_language_features.shape[-1],
        )




        ########################################################## 反事实视觉删除 ##########################################################
        intervention_mask = None
        if visual_intervention_target is not None:  # 正常场景训练时,visual_intervention_target is None,所以只有反事实训练的时候才会执行以下内容
            
            # 反事实时检查valid mask,因为如果告诉函数"我要删除主要actor",就必须同时告诉函数"哪些样本的actor空间标签有效"
            if visual_intervention_valid is None:
                raise ValueError(
                    "visual_intervention_valid is required when a "
                    "counterfactual target is provided."
                )
            
            # 构造删除主要actor之后的视觉特征
            (
                visual_features,   # [B,384,D] 新的视觉特征
                intervention_mask, # [B,6,64]  记录了到底哪些视觉区域被干预了
            ) = self.interaction_reasoner.build_counterfactual_visual_features(
                visual_features=visual_features,                # [B,384,D]
                participant_target=visual_intervention_target,  # [B,6,64] 这个参数是告诉网络"主要actor在六视角的哪些patch里"
                valid_mask=visual_intervention_valid,
                strength=float(
                    getattr(
                        self,
                        "counterfactual_intervention_strength",
                        1.0,
                    )
                ),
                mask_gamma=float(
                    getattr(
                        self,
                        "counterfactual_intervention_mask_gamma",
                        0.5,
                    )
                ),
            )

            # 将新的视觉特征重新放回raw_language_features对应的<IMG_CONTEXT>的位置
            raw_language_features = raw_language_features.clone()
            raw_language_features[visual_mask] = (
                visual_features.to(
                    dtype=raw_language_features.dtype
                ).reshape(-1, raw_language_features.shape[-1])
            )





        ########################################################## Target Point 上下文 ##########################################################

        # 找到两个target point
        navigation_mask = (language_ids == int(target_point_token_id))
        navigation_count = navigation_mask.sum(dim=1)
        if not torch.all(navigation_count == 2):
            raise RuntimeError(
                "Stage-2 interaction reasoning requires exactly two "
                "TARGET_POINT embeddings per sample, but received "
                f"counts={navigation_count.tolist()}."
            )
        
        # 把两个 Target Point embedding 汇聚成一个导航上下文(实际上就是求均值) [B,D]
        navigation_context = raw_language_features[
            navigation_mask
        ].reshape(
            raw_language_features.shape[0],
            2,
            raw_language_features.shape[-1],
        ).mean(dim=1)





        ########################################################## 哪些语言内容允许4个交互token看 ##########################################################

        # 交互token只读取用户问题、导航，不能读取视觉特征和训练答案
        language_context_mask = (
            language_valid
            & ~visual_mask   # 不能读取视觉特征,因为视觉特征已经通过visual_features进入Reasoner了,没有必要再把这些视觉embedding当成语言上下文平均一次了
            & ~answer_mask   # 不能读取答案
        )






        ########################################################## 调用 Interaction Reasoner 构造输出 ##########################################################

        reasoner_dtype = next(self.interaction_reasoner.parameters()).dtype
        outputs = self.interaction_reasoner(
            interaction_query_features=raw_driving_queries.to(dtype=reasoner_dtype),  # [B,30,D]  driving query
            visual_features=visual_features.to(dtype=reasoner_dtype),                 # [B,384,D] 视觉特征
            language_features=raw_language_features.to(dtype=reasoner_dtype),         # [B,L,D]   语言embedding
            language_context_mask=language_context_mask,                              # [B,L]     交互token读取信息的位置
            navigation_context=navigation_context.to(dtype=reasoner_dtype),           # [B,D]     导航上下文
        )
        outputs["language_ids"] = language_ids                       # [B,L] 语言token id
        outputs["language_valid"] = language_valid                   # [B,L] 语言有效位置,解决的是padding的问题
        outputs["answer_mask"] = answer_mask                         # [B,L] 哪些位置的语言token是答案
        outputs["raw_language_features"] = raw_language_features     # [B,L,D]   语言embedding
        outputs["raw_driving_queries"] = raw_driving_queries         # [B,30,D]  driving query
        # 如果是反事实场景则额外保存干预mask,
        # 后面可以利用它计算counterfactual_visual_suppression_loss,
        # 也就是,我明确删除了主要 actor 所在区域之后，网络对这个区域的主要 actor attention 是否真的降低了？
        if intervention_mask is not None:
            outputs["visual_intervention_mask"] = intervention_mask
        return outputs

    @staticmethod
    def _assemble_joint_sequence(interaction_tokens: Tensor,language_features: Tensor,language_valid: Tensor,driving_queries: Tensor,) -> Tuple[Tensor, Tensor, Tensor, Tuple[int, int, int]]:
        
        """
        把 4 个 interaction token、Language embedding、30 个 Driving query 拼成一个完整序列,
        并处理 padding,
        让这个序列可以直接送入 Language Transformer
        """

        # batch size
        batch_size = int(language_features.shape[0])

        # interaction token 的 mask 全部设为 True  [B,4]  因为这4个token每个样本都有,没有padding
        interaction_valid = torch.ones(
            interaction_tokens.shape[:2],
            device=interaction_tokens.device,
            dtype=torch.bool,
        )

        # driving token 的 mask 全部设为 True  [B,30]  因为这30个token每个样本都有,没有padding
        driving_valid = torch.ones(
            driving_queries.shape[:2],
            device=driving_queries.device,
            dtype=torch.bool,
        )

        # 按固定顺序拼接 [B,4+L+30,D]
        original_embeddings = torch.cat(
            (
                interaction_tokens,  # 1. 四种交互token    [B,4,D]
                language_features,   # 2. language token  [B.L,D]
                driving_queries,     # 3. driving token   [B,20+10,D]
            ),
            dim=1,
        )
        original_mask = torch.cat(
            (
                interaction_valid,
                language_valid,
                driving_valid,
            ),
            dim=1,
        )

        original_indices = torch.arange(
            original_embeddings.shape[1],
            device=original_embeddings.device,
        ).expand(batch_size, -1)
        valid_permutation = original_mask.byte().argsort(
            dim=-1,
            descending=True,
            stable=True,
        )
        permutation = original_indices.gather(
            1,
            valid_permutation,
        )
        batch_indices = torch.arange(
            batch_size,
            device=original_embeddings.device,
        )[:, None]

        return (
            original_embeddings[batch_indices, permutation],
            original_mask[batch_indices, permutation],
            permutation,
            (
                int(interaction_tokens.shape[1]),
                int(language_features.shape[1]),
                int(driving_queries.shape[1]),
            ),
        )

    @staticmethod
    def _split_joint_outputs(outputs: Tensor,permutation: Tensor,split_sizes: Tuple[int, int, int],) -> Tuple[Tensor, Tensor, Tensor]:
        inverse_permutation = permutation.argsort(dim=-1)
        batch_indices = torch.arange(
            outputs.shape[0],
            device=outputs.device,
        )[:, None]
        original_order = outputs[
            batch_indices,
            inverse_permutation,
        ]
        return tuple(
            original_order.split(split_sizes, dim=1)
        )

    def _forward_joint_transformer(self,interaction_outputs: Dict[str, Tensor],) -> Tuple[Tensor, Tensor, Tensor]:
        
        interaction_tokens = interaction_outputs["interaction_tokens"].to(dtype=self._model_dtype())
        language_features = interaction_outputs["raw_language_features"].to(dtype=self._model_dtype())
        driving_queries = interaction_outputs["raw_driving_queries"].to(dtype=self._model_dtype())



        ################################################# 拼接序列 #################################################
        (
            joint_embeddings,  # [B,4+L+20+10,D]   |4个交互token|多模态语言序列|route query|ego future query|
            joint_mask,
            permutation,
            split_sizes,
        ) = self._assemble_joint_sequence(
            interaction_tokens=interaction_tokens,                # [B,4,D]
            language_features=language_features,                  # [B,L,D] 这是已经完成<IMG_CONTEXT>和<TARGET_POINT>替换的多模态语言序列
            language_valid=interaction_outputs["language_valid"], # [B,L]   表示多模态语言序列中哪些是padding的哪些不是
            driving_queries=driving_queries,                      # [B,30,D]route query 和 ego future query
        )



        ################################################# 送入语言Tranformer进行前向推理 #################################################

        # 进行一次Transformer前向
        joint_features = self.language_model.forward_features(
            embeddings=joint_embeddings,
            attention_mask=joint_mask,
            position_ids=None,
            return_dict=True,
        )


        ################################################# |四通道结构化世界|语言|route query|ego future query| #################################################

        # 再拆分成三部分 [B,4,D] [B,L,D] [B,30,D]
        return self._split_joint_outputs(
            joint_features,
            permutation,
            split_sizes,
        )

    @staticmethod
    def _find_marker(
        token_ids: Sequence[int],
        allowed_mask: Sequence[bool],
        patterns: Sequence[Sequence[int]],
        start_index: int,) -> Optional[Tuple[int, int]]:
        for position in range(
            max(int(start_index), 0),
            len(token_ids),
        ):
            for pattern in patterns:
                pattern_length = len(pattern)
                end = position + pattern_length
                if end > len(token_ids):
                    continue
                if not all(allowed_mask[position:end]):
                    continue
                if tuple(token_ids[position:end]) == tuple(pattern):
                    return position, end
        return None

    def _build_question_span_masks(
        self,
        language_ids: Tensor,
        answer_mask: Tensor,) -> Tuple[Tensor, Tensor]:
        batch_size, sequence_length = language_ids.shape
        question_masks = torch.zeros(
            (batch_size, 4, sequence_length),
            device=language_ids.device,
            dtype=torch.bool,
        )
        valid_samples = torch.zeros(
            (batch_size,),
            device=language_ids.device,
            dtype=torch.bool,
        )

        for sample_index in range(batch_size):
            ids = [
                int(value)
                for value in language_ids[sample_index].detach().cpu().tolist()
            ]
            allowed = [
                bool(value)
                for value in answer_mask[sample_index].detach().cpu().tolist()
            ]

            markers: List[Tuple[int, int]] = []
            cursor = 0
            for patterns in self._question_marker_patterns:
                marker = self._find_marker(
                    ids,
                    allowed,
                    patterns,
                    cursor,
                )
                if marker is None:
                    markers = []
                    break
                markers.append(marker)
                cursor = marker[1]

            if len(markers) != 4:
                continue

            waypoint_marker = self._find_marker(
                ids,
                allowed,
                self._waypoint_marker_patterns,
                markers[-1][1],
            )
            if waypoint_marker is None:
                continue

            span_valid = True
            for question_index in range(4):
                span_start = markers[question_index][1]
                span_end = (
                    markers[question_index + 1][0]
                    if question_index < 3
                    else waypoint_marker[0]
                )
                if span_end <= span_start:
                    span_valid = False
                    break

                span_mask = answer_mask[
                    sample_index,
                    span_start:span_end,
                ]
                if not bool(span_mask.any().item()):
                    span_valid = False
                    break
                question_masks[
                    sample_index,
                    question_index,
                    span_start:span_end,
                ] = span_mask

            if span_valid:
                valid_samples[sample_index] = True
            else:
                question_masks[sample_index].zero_()

        return question_masks, valid_samples

    @staticmethod
    def _prepare_future_supervision(
        example: DrivingExample,
        prediction_logits: Tensor,) -> Tuple[Tensor, Tensor]:
        target = example.driving_label.future_interaction_grid
        valid = example.driving_label.future_interaction_valid
        batch_size = int(prediction_logits.shape[0])

        if target is None:
            if (
                isinstance(valid, torch.Tensor)
                and bool(valid.any().item())
            ):
                raise RuntimeError(
                    "future_interaction_valid contains True, but "
                    "future_interaction_grid is None."
                )
            target = torch.zeros(
                (
                    batch_size,
                    4,
                    prediction_logits.shape[-2],
                    prediction_logits.shape[-1],
                ),
                device=prediction_logits.device,
                dtype=torch.float32,
            )
            valid = torch.zeros(
                (batch_size,),
                device=prediction_logits.device,
                dtype=torch.bool,
            )
        elif not isinstance(valid, torch.Tensor):
            raise RuntimeError(
                "A future interaction target exists, but its valid mask "
                "is missing."
            )
        else:
            target = target.to(
                device=prediction_logits.device,
                dtype=torch.float32,
            )
            valid = valid.to(
                device=prediction_logits.device,
                dtype=torch.bool,
            )

        return target, valid

    @staticmethod
    def _counterfactual_loss_keys() -> Tuple[str, ...]:
        return (
            "counterfactual_visual_suppression_loss",
            "counterfactual_language_supervision_loss",
            "counterfactual_language_consistency_loss",
            "counterfactual_action_target_loss",
            "counterfactual_action_effect_loss",
            "counterfactual_route_invariance_loss",
            "counterfactual_world_primary_removal_loss",
            "counterfactual_world_invariance_loss",
        )

    def _empty_counterfactual_losses(
        self,
        reference: Tensor,
        batch_size: int,) -> Dict[str, Tuple[Tensor, Tensor]]:
        parameter_zero = (
            self.interaction_reasoner.counterfactual_parameter_zero()
        ).to(device=reference.device, dtype=reference.dtype)
        zero_value = (
            reference.reshape(reference.shape[0], -1).sum(dim=-1)
            * 0.0
        )
        if zero_value.shape[0] != int(batch_size):
            zero_value = torch.zeros(
                (int(batch_size),),
                device=reference.device,
                dtype=reference.dtype,
            )
        zero_value = zero_value + parameter_zero
        zero_count = torch.zeros_like(zero_value)
        return {
            key: (zero_value, zero_count)
            for key in self._counterfactual_loss_keys()
        }

    def _select_counterfactual_indices(
        self,
        participant_valid: Tensor,) -> Tensor:
        selected = participant_valid.bool().clone()
        probability = float(
            getattr(
                self,
                "counterfactual_intervention_probability",
                1.0,
            )
        )
        if self.training and probability < 1.0:
            selected = selected & (
                torch.rand(
                    selected.shape,
                    device=selected.device,
                ) < max(probability, 0.0)
            )
        return selected.nonzero(as_tuple=False).squeeze(1)

    def _compute_counterfactual_losses(
        self,
        *,
        example: DrivingExample,
        adaptor_dict: Dict,
        contextual_interaction_tokens: Tensor,
        language_features: Tensor,
        driving_features: Tensor,
        question_span_mask: Tensor,
        language_alignment_valid: Tensor,
        future_target: Optional[Tensor],
        future_valid: Tensor,) -> Tuple[
        Dict[str, Tuple[Tensor, Tensor]],
        Dict[str, Tensor],]:


        batch_size = int(driving_features.shape[0])
        empty_losses = self._empty_counterfactual_losses(
            driving_features,
            batch_size,
        )
        diagnostics = {
            "counterfactual_selected": torch.zeros(
                (batch_size,),
                device=driving_features.device,
                dtype=torch.bool,
            )
        }

        if not bool(
            getattr(
                self,
                "use_counterfactual_intervention_training",
                False,
            )
        ):
            return empty_losses, diagnostics

        participant_target = (
            example.driving_label.camera_attention_target
        )
        participant_valid = (
            example.driving_label.camera_attention_valid
        )
        counterfactual_waypoints = (
            example.driving_label.counterfactual_waypoints
        )
        counterfactual_waypoints_valid = (
            example.driving_label.counterfactual_waypoints_valid
        )
        counterfactual_causal_score = (
            example.driving_label.counterfactual_causal_score
        )
        required_tensors = {
            "participant spatial target": participant_target,
            "participant valid mask": participant_valid,
            "counterfactual waypoints": counterfactual_waypoints,
            "counterfactual waypoint valid mask": (
                counterfactual_waypoints_valid
            ),
            "counterfactual causal score": (
                counterfactual_causal_score
            ),
        }
        for name, value in required_tensors.items():
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"Counterfactual intervention requires {name}."
                )

        participant_target = participant_target.to(
            device=driving_features.device,
            dtype=torch.float32,
        )
        participant_valid = participant_valid.to(
            device=driving_features.device,
            dtype=torch.bool,
        )
        counterfactual_waypoints = counterfactual_waypoints.to(
            device=driving_features.device,
            dtype=torch.float32,
        )
        counterfactual_waypoints_valid = (
            counterfactual_waypoints_valid.to(
                device=driving_features.device,
                dtype=torch.bool,
            )
        )
        counterfactual_causal_score = (
            counterfactual_causal_score.to(
                device=driving_features.device,
                dtype=torch.float32,
            )
        )

        expected_participant_shape = (
            batch_size,
            self.interaction_reasoner.num_cameras,
            self.interaction_reasoner.tokens_per_camera,
        )
        if participant_target.shape != expected_participant_shape:
            raise RuntimeError(
                "Counterfactual participant target must have shape "
                f"{expected_participant_shape}, but received "
                f"{tuple(participant_target.shape)}."
            )
        if participant_valid.shape != (batch_size,):
            raise RuntimeError(
                "Counterfactual participant valid mask must have shape [B]."
            )
        if counterfactual_waypoints_valid.shape != (batch_size,):
            raise RuntimeError(
                "Counterfactual waypoint valid mask must have shape [B]."
            )
        if counterfactual_causal_score.shape != (batch_size,):
            raise RuntimeError(
                "Counterfactual causal score must have shape [B]."
            )

        selected_indices = self._select_counterfactual_indices(
            participant_valid
        )
        if selected_indices.numel() == 0:
            return empty_losses, diagnostics

        counterfactual_adaptor_dict = (
            self._build_counterfactual_adaptor_dict(
                example,
                adaptor_dict,
            )
        )
        diagnostics["counterfactual_selected"][selected_indices] = True
        selected_valid = torch.ones(
            (selected_indices.numel(),),
            device=driving_features.device,
            dtype=torch.bool,
        )
        counterfactual_outputs = (
            self._build_pre_language_interaction(
                counterfactual_adaptor_dict,
                batch_slice=selected_indices,
                visual_intervention_target=(
                    participant_target[selected_indices]
                ),
                visual_intervention_valid=selected_valid,
            )
        )
        (
            counterfactual_tokens,            
            counterfactual_language_features,
            counterfactual_driving_features,
        ) = self._forward_joint_transformer(
            counterfactual_outputs
        )

        losses = compute_counterfactual_attention_suppression_loss(
            counterfactual_primary_attention=(
                counterfactual_outputs[
                    "primary_spatial_attention"
                ]
            ),
            intervention_mask=(
                counterfactual_outputs[
                    "visual_intervention_mask"
                ]
            ),
            valid_mask=selected_valid,
        )

        original_predictions = (
            self.adaptors.driving.get_predictions(
                driving_features[selected_indices]
            )
        )
        counterfactual_predictions = (
            self.adaptors.driving.get_predictions(
                counterfactual_driving_features
            )
        )
        if (
            "speed_wps" not in original_predictions
            or "speed_wps" not in counterfactual_predictions
        ):
            raise RuntimeError(
                "Counterfactual action training requires speed_wps predictions."
            )

        action_valid = counterfactual_waypoints_valid[
            selected_indices
        ]
        causal_confidence = torch.where(
            action_valid,
            counterfactual_causal_score[selected_indices]
            .clamp(min=0.25, max=2.0),
            torch.zeros_like(
                counterfactual_causal_score[selected_indices]
            ),
        )
        full_scene_target = example.driving_label.waypoints.to(
            device=driving_features.device,
            dtype=torch.float32,
        )[selected_indices]
        counterfactual_target = counterfactual_waypoints[
            selected_indices
        ]
        original_speed_prediction = original_predictions[
            "speed_wps"
        ].float()
        counterfactual_speed_prediction = (
            counterfactual_predictions["speed_wps"].float()
        )
        expected_action_shape = tuple(
            counterfactual_speed_prediction.shape
        )
        for name, value in (
            ("original speed prediction", original_speed_prediction),
            ("full-scene waypoint target", full_scene_target),
            ("counterfactual waypoint target", counterfactual_target),
        ):
            if tuple(value.shape) != expected_action_shape:
                raise RuntimeError(
                    f"{name} must have shape {expected_action_shape}, "
                    f"but received {tuple(value.shape)}."
                )

        action_target_loss = torch.nn.functional.smooth_l1_loss(
            counterfactual_speed_prediction,
            counterfactual_target,
            reduction="none",
        ).sum(dim=-1).mean(dim=-1)
        predicted_effect = (
            original_speed_prediction
            - counterfactual_speed_prediction
        )
        target_effect = full_scene_target - counterfactual_target
        action_effect_loss = torch.nn.functional.smooth_l1_loss(
            predicted_effect,
            target_effect,
            reduction="none",
        ).sum(dim=-1).mean(dim=-1)
        losses["counterfactual_action_target_loss"] = (
            action_target_loss * causal_confidence,
            causal_confidence,
        )
        losses["counterfactual_action_effect_loss"] = (
            action_effect_loss * causal_confidence,
            causal_confidence,
        )

        if "route" not in counterfactual_predictions:
            raise RuntimeError(
                "Counterfactual route invariance requires route predictions."
            )
        route_target = example.driving_label.path.to(
            device=driving_features.device,
            dtype=torch.float32,
        )[selected_indices]
        route_prediction = counterfactual_predictions["route"].float()
        if route_prediction.shape != route_target.shape:
            raise RuntimeError(
                "Counterfactual route prediction and target shapes differ: "
                f"{tuple(route_prediction.shape)} vs "
                f"{tuple(route_target.shape)}."
            )
        route_loss = torch.nn.functional.smooth_l1_loss(
            route_prediction,
            route_target,
            reduction="none",
        ).sum(dim=-1).mean(dim=-1)
        route_count = torch.ones_like(route_loss)
        losses["counterfactual_route_invariance_loss"] = (
            route_loss,
            route_count,
        )

        counterfactual_language_input = {
            "_ids": counterfactual_outputs["language_ids"],
            "_ids_mask": counterfactual_outputs["answer_mask"],
        }
        counterfactual_language_loss = (
            self.adaptors.language.compute_loss(
                counterfactual_language_features,
                None,
                counterfactual_language_input,
                example,
            )["language_loss"]
        )
        language_loss_value, language_loss_count = (
            counterfactual_language_loss
        )
        language_weight = causal_confidence.view(
            -1,
            *([1] * (language_loss_value.ndim - 1)),
        )
        losses["counterfactual_language_supervision_loss"] = (
            language_loss_value * language_weight,
            language_loss_count.to(
                dtype=language_loss_value.dtype
            ) * language_weight,
        )

        (
            counterfactual_question_span_mask,
            counterfactual_language_alignment_valid,
        ) = self._build_question_span_masks(
            counterfactual_outputs["language_ids"],
            counterfactual_outputs["answer_mask"],
        )
        language_confidence = (
            language_alignment_valid[selected_indices]
            & counterfactual_language_alignment_valid
            & action_valid
        ).to(dtype=causal_confidence.dtype) * causal_confidence
        losses["counterfactual_language_consistency_loss"] = (
            self.interaction_reasoner
            .compute_counterfactual_language_consistency_loss(
                original_language_features=(
                    language_features[selected_indices]
                ),
                counterfactual_language_features=(
                    counterfactual_language_features
                ),
                original_interaction_tokens=(
                    contextual_interaction_tokens[selected_indices]
                ),
                counterfactual_interaction_tokens=(
                    counterfactual_tokens
                ),
                original_question_span_mask=(
                    question_span_mask[selected_indices]
                ),
                counterfactual_question_span_mask=(
                    counterfactual_question_span_mask
                ),
                full_scene_waypoints=full_scene_target,
                counterfactual_waypoints=(
                    counterfactual_target
                ),
                valid_mask=language_confidence,
                minimum_change=float(
                    getattr(
                        self,
                        "counterfactual_language_min_change",
                        0.02,
                    )
                ),
            )
        )

        if (
            self.future_interaction_decoder is not None
            and future_target is not None
        ):
            counterfactual_future_logits = (
                self.future_interaction_decoder(
                    counterfactual_tokens
                )
            )
            losses.update(
                compute_counterfactual_future_world_losses(
                    counterfactual_logits=(
                        counterfactual_future_logits
                    ),
                    full_scene_target=(
                        future_target[selected_indices]
                    ),
                    valid_mask=(
                        future_valid[selected_indices]
                    ),
                )
            )
        else:
            zero = counterfactual_driving_features.sum(
                dim=(-2, -1)
            ) * 0.0
            zero_count = torch.zeros_like(zero)
            losses["counterfactual_world_primary_removal_loss"] = (
                zero,
                zero_count,
            )
            losses["counterfactual_world_invariance_loss"] = (
                zero,
                zero_count,
            )

        full_action_valid = torch.zeros(
            (batch_size,),
            device=driving_features.device,
            dtype=torch.bool,
        )
        full_action_valid[selected_indices] = action_valid
        full_confidence = torch.zeros(
            (batch_size,),
            device=driving_features.device,
            dtype=torch.float32,
        )
        full_confidence[selected_indices] = causal_confidence
        full_intervention_mask = torch.zeros(
            (
                batch_size,
                self.interaction_reasoner.num_cameras,
                self.interaction_reasoner.tokens_per_camera,
            ),
            device=driving_features.device,
            dtype=torch.float32,
        )
        full_intervention_mask[selected_indices] = (
            counterfactual_outputs[
                "visual_intervention_mask"
            ].detach().float()
        )
        full_predicted_effect = torch.zeros_like(
            example.driving_label.waypoints,
            device=driving_features.device,
            dtype=torch.float32,
        )
        full_target_effect = torch.zeros_like(
            full_predicted_effect
        )
        full_predicted_effect[selected_indices] = (
            predicted_effect.detach()
        )
        full_target_effect[selected_indices] = target_effect.detach()
        diagnostics.update(
            {
                "counterfactual_action_valid": full_action_valid,
                "counterfactual_causal_confidence": full_confidence,
                "counterfactual_intervention_mask": (
                    full_intervention_mask
                ),
                "counterfactual_predicted_action_effect": (
                    full_predicted_effect
                ),
                "counterfactual_target_action_effect": (
                    full_target_effect
                ),
            }
        )
        for key in self._counterfactual_loss_keys():
            if key not in losses:
                losses[key] = empty_losses[key]
        return losses, diagnostics

    def _add_target_point_metadata(
        self,
        *,
        pred_labels: Dict,
        adaptor_dict: Dict,
        example: DrivingExample,) -> None:
        image_encoder = self.vision_model.image_encoder
        language_ids = adaptor_dict.get("language__ids")
        if not isinstance(language_ids, torch.Tensor):
            raise RuntimeError(
                "language__ids is required to save TARGET_POINT coordinates."
            )

        target_point_token_id = getattr(
            image_encoder,
            "target_point_token_id",
            None,
        )
        if target_point_token_id is None:
            raise RuntimeError(
                "target_point_token_id was not initialized by the image encoder."
            )
        target_point_token_id = int(target_point_token_id)

        has_target_point = (
            language_ids == target_point_token_id
        ).any(dim=1)
        batch_size = int(language_ids.shape[0])
        target_point_coordinates = torch.full(
            (batch_size, 2, 2),
            float("nan"),
            device=language_ids.device,
            dtype=torch.float32,
        )
        placeholder_values = (
            example.driving_input.prompt.placeholder_values
        )
        if len(placeholder_values) != batch_size:
            raise RuntimeError(
                "The number of placeholder dictionaries does not match "
                f"batch size: {len(placeholder_values)} vs {batch_size}."
            )

        for sample_index in range(batch_size):
            if not bool(has_target_point[sample_index].item()):
                continue
            sample_values = placeholder_values[sample_index]
            if target_point_token_id not in sample_values:
                raise KeyError(
                    "The prompt contains TARGET_POINT, but its coordinates "
                    "are missing from placeholder_values."
                )
            points = torch.as_tensor(
                sample_values[target_point_token_id],
                device=language_ids.device,
                dtype=torch.float32,
            )
            if points.numel() != 4:
                raise ValueError(
                    "TARGET_POINT must contain two 2D coordinates."
                )
            target_point_coordinates[sample_index] = points.reshape(2, 2)

        pred_labels["target_point_coordinates"] = target_point_coordinates
        pred_labels["has_target_point"] = has_target_point

    def _log_stage2_statistics(
        self,
        mode: str,
        interaction_outputs: Dict[str, Tensor],
        language_alignment_valid: Tensor,
        secondary_exists: Tensor,) -> None:
        primary_attention = interaction_outputs[
            "primary_spatial_attention"
        ].detach().float()
        secondary_attention = interaction_outputs[
            "secondary_spatial_attention"
        ].detach().float()
        batch_size = int(primary_attention.shape[0])
        on_step = mode == "train"
        camera_names = (
            "front",
            "front_left",
            "front_right",
            "rear",
            "rear_left",
            "rear_right",
        )

        for prefix, attention in (
            ("primary", primary_attention),
            ("secondary", secondary_attention),
        ):
            camera_marginal = attention.sum(dim=-1).mean(dim=0)
            for camera_name, camera_weight in zip(
                camera_names,
                camera_marginal,
            ):
                self.log(
                    f"{mode}_stage2_attention/{prefix}_{camera_name}",
                    camera_weight,
                    on_step=on_step,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

        self.log(
            f"{mode}_stage2_attention/primary_normalized_entropy",
            interaction_outputs[
                "primary_spatial_attention_entropy"
            ].detach().float().mean(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}_stage2_alignment/four_question_valid_ratio",
            language_alignment_valid.float().mean(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}_stage2_alignment/secondary_actor_ratio",
            secondary_exists.float().mean(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    # 训练/验证总入口
    def forward_loss(self,example: DrivingExample,per_sample: bool = False,) -> TrainingOutput:
        

        ############################################## 语言Transformer的输入 ##############################################

        # 语言embedding内容
        adaptor_dict = self._replace_multimodal_placeholders(example,inference=False,)

        # 构造进入 Language Transformer 之前的 4 个统一 interaction tokens
        interaction_outputs = self._build_pre_language_interaction(adaptor_dict)






        ############################################## 语言Transformer的输出(hidden features) ##############################################
        
        # contextual_interaction_tokens  [B,4,D]
        # language_features              [B,L,D]
        # driving_features               [B,30,D]

        # 知识点: 语言 Transformer 的输入和输出 hidden size 不会改变
        
        contextual_interaction_tokens,language_features,driving_features = self._forward_joint_transformer(interaction_outputs)




        ############################################## language 损失计算 ##############################################

        """
        Transformer后的language_features
                ↓
        lm_head
                ↓
        vocabulary logits
                ↓
        与真实答案token做Cross Entropy
        """

        language_input_dict = {
            "_ids": interaction_outputs["language_ids"],
            "_ids_mask": interaction_outputs["answer_mask"],
        }
        loss_dict = self.adaptors.language.compute_loss(
            language_features,
            None,
            language_input_dict,
            example,
        )




        ############################################## driving 损失计算 ##############################################

        """
        Transformer后的driving_features
                ↓
        预测头
                ↓
        预测的点
                ↓
        与真实点做Smooth L1
        """

        loss_dict.update(
            self.adaptors.driving.compute_loss(
                driving_features,
                None,
                {},
                example,
            )
        )




        ############################################## 基础的VLA训练 ##############################################
        ############################################## 基础的VLA训练 ##############################################





        


        ############################################## 找出 Q1-Q4 各自对应的答案区域 ##############################################

        # question_span_mask:[B,4,L] question_span_mask[:,0]表示A1答案对应哪些lannguage token
        # language_alignment_valid: [B] 表示这个样本是否成功识别出了完整的 A1、A2、A3、A4 四段答案

        question_span_mask, language_alignment_valid = (
            self._build_question_span_masks(
                interaction_outputs["language_ids"],  # 语言序列对应的token id
                interaction_outputs["answer_mask"],
            )
        )






        ############################################## language 与 interaction token 对齐 ##############################################

        # 主要是想要"interaction token 不只是预测未来世界,还被要求同时和语言语义、Driving动作表示保持一致"

        language_alignment_loss = (
            self.interaction_reasoner.compute_language_alignment_loss(
                language_features=language_features,    # [B,L,D] 语言Transformer的输出
                contextual_interaction_tokens=(contextual_interaction_tokens),  # [B,L,D] 语言Transformer的输出
                question_span_mask=question_span_mask,  # [B,4,L]
                valid_mask=language_alignment_valid,    # [B] 表示这个样本是否成功识别出了完整的 A1、A2、A3、A4 四段答案
                temperature=float(
                    getattr(
                        self,
                        "interaction_alignment_temperature",
                        0.1,
                    )
                ),
            )
        )

        # 作用:不能只是语言答案预测对了,还希望语言内部形成的表示和对应的 interaction token 真的表达同一件事情
        loss_dict["language_interaction_alignment_loss"] = (language_alignment_loss)






        ############################################## driving 与 interaction token 对齐 ##############################################

        # 主要是想要"interaction token 不只是预测未来世界,还被要求同时和语言语义、Driving动作表示保持一致"
        
        loss_dict["action_interaction_alignment_loss"] = (
            self.interaction_reasoner.compute_action_alignment_loss(
                driving_features=driving_features,
                contextual_interaction_tokens=(
                    contextual_interaction_tokens
                ),
            )
        )





        ############################################## 四通道结构化未来世界预测 ##############################################

        # 初始化
        future_target = None
        future_valid = torch.zeros((driving_features.shape[0],),device=driving_features.device,dtype=torch.bool,)
        
        # 如果配置文件use_future_interaction_prediction=True
        if bool(getattr(self,"use_future_interaction_prediction",False,)):
            
            # 安全性检查
            if self.future_interaction_decoder is None:
                raise RuntimeError(
                    "Future interaction prediction is enabled, but the "
                    "decoder was not initialized."
                )

            # 将语言Transformer的输出[B,4,D]放入四通道结构化未来世界解码器 future_logits:[B,4,128,128]这就是四通道结构化未来世界
            future_logits = self.future_interaction_decoder(contextual_interaction_tokens)
            
            # 读取四通道 GT,然后计算 BCE + Dice, 每个通道独立计算
            future_target, future_valid = self._prepare_future_supervision(example,future_logits,)
            loss_dict.update(
                compute_future_interaction_losses(
                    prediction_logits=future_logits,
                    target_grid=future_target,
                    valid_mask=future_valid,
                    positive_weights=tuple(
                        float(value)
                        for value in getattr(
                            self,
                            "future_interaction_positive_weights",
                            (20.0, 20.0, 80.0, 100.0),
                        )
                    ),
                )
            )






        ############################################## 主要actor的视觉空间监督 ##############################################

        # 如果配置use_participant_spatial_attention_supervision=True
        if bool(getattr(self,"use_participant_spatial_attention_supervision",False,)):
            
            # [B,6,64]  六视角相机 token 级注意力标签
            participant_target = (example.driving_label.camera_attention_target)
            
            # [B],样本有无可靠的相机注意力监督
            participant_valid = (example.driving_label.camera_attention_valid)
            
            # 安全性检查
            if not isinstance(participant_target, torch.Tensor):
                raise RuntimeError(
                    "Participant spatial supervision is enabled, but the "
                    "target tensor is missing."
                )

            # 安全性检查    
            if not isinstance(participant_valid, torch.Tensor):
                raise RuntimeError(
                    "Participant spatial supervision is enabled, but the "
                    "valid mask is missing."
                )

            # 计算损失
            loss_dict.update(
                compute_participant_spatial_attention_losses(
                    primary_attention=interaction_outputs["primary_spatial_attention"],  # [B.6,64] Interaction Reasoner 产生的 primary_spatial_attention
                    target_attention=participant_target,                                 # [B,6,64]  六视角相机 token 级注意力标签
                    valid_mask=participant_valid,
                )
            )

        # 判断有没有次要actor
        if future_target is None:
            secondary_exists = torch.zeros_like(future_valid)
        else:
            secondary_exists = (
                future_valid
                & (
                    future_target[:, 3]
                    .amax(dim=(-2, -1))
                    > 1e-6
                )
            )
        loss_dict.update(
            compute_actor_disentanglement_losses(
                primary_attention=interaction_outputs["primary_spatial_attention"],
                secondary_attention=interaction_outputs["secondary_spatial_attention"],
                contextual_interaction_tokens=(contextual_interaction_tokens),
                secondary_exists=secondary_exists,
                attention_overlap_margin=float(
                    getattr(
                        self,
                        "actor_attention_overlap_margin",
                        0.35,
                    )
                ),
                token_cosine_margin=float(
                    getattr(
                        self,
                        "actor_token_cosine_margin",
                        0.30,
                    )
                ),
            )
        )








        ############################################## 反事实训练总入口 ##############################################

        counterfactual_loss_dict, counterfactual_diagnostics = (
            self._compute_counterfactual_losses(
                example=example,
                adaptor_dict=adaptor_dict,
                contextual_interaction_tokens=(
                    contextual_interaction_tokens
                ),
                language_features=language_features,
                driving_features=driving_features,
                question_span_mask=question_span_mask,
                language_alignment_valid=(
                    language_alignment_valid
                ),
                future_target=future_target,
                future_valid=future_valid,
            )
        )
        loss_dict.update(counterfactual_loss_dict)







        ############################################## 所有loss ##############################################

        loss_dict_only_losses = {
            key: value
            for key, value in loss_dict.items()
            if key.endswith("loss")
        }




        ############################################## 完整训练 ##############################################
        ############################################## 完整训练 ##############################################




        loss_logs = {
            key: value
            for key, value in loss_dict.items()
            if key.endswith("log")
        }
        pred_labels = {
            key: value
            for key, value in loss_dict.items()
            if not key.endswith("loss")
            and not key.endswith("log")
        }
        pred_labels["primary_spatial_attention"] = (
            interaction_outputs[
                "primary_spatial_attention"
            ].detach()
        )
        pred_labels["secondary_spatial_attention"] = (
            interaction_outputs[
                "secondary_spatial_attention"
            ].detach()
        )
        pred_labels["interaction_token_norms"] = (
            contextual_interaction_tokens.detach().float().norm(dim=-1)
        )
        pred_labels["language_alignment_valid"] = (
            language_alignment_valid.detach()
        )
        pred_labels["secondary_actor_exists"] = (
            secondary_exists.detach()
        )
        pred_labels.update(counterfactual_diagnostics)



        ############################################## 记录 W&B 统计 ##############################################

        self._log_stage2_statistics(
            "train" if self.training else "val",
            interaction_outputs,
            language_alignment_valid,
            secondary_exists,
        )

        if per_sample:
            self._add_target_point_metadata(
                pred_labels=pred_labels,
                adaptor_dict=adaptor_dict,
                example=example,
            )
            return loss_dict_only_losses, pred_labels






        ############################################## 各种 loss 各占多大比例 ##############################################

        loss_weights: Dict[str, float] = {
            "language_interaction_alignment_loss": float(
                getattr(
                    self,
                    "language_interaction_alignment_loss_weight",
                    0.05,
                )
            ),
            "action_interaction_alignment_loss": float(
                getattr(
                    self,
                    "action_interaction_alignment_loss_weight",
                    0.05,
                )
            ),
            "actor_attention_separation_loss": float(
                getattr(
                    self,
                    "actor_attention_separation_loss_weight",
                    0.02,
                )
            ),
            "actor_token_separation_loss": float(
                getattr(
                    self,
                    "actor_token_separation_loss_weight",
                    0.02,
                )
            ),
            "counterfactual_visual_suppression_loss": float(
                getattr(
                    self,
                    "counterfactual_visual_suppression_loss_weight",
                    0.05,
                )
            ),
            "counterfactual_language_supervision_loss": float(
                getattr(
                    self,
                    "counterfactual_language_supervision_loss_weight",
                    0.10,
                )
            ),
            "counterfactual_language_consistency_loss": float(
                getattr(
                    self,
                    "counterfactual_language_consistency_loss_weight",
                    0.05,
                )
            ),
            "counterfactual_action_target_loss": float(
                getattr(
                    self,
                    "counterfactual_action_target_loss_weight",
                    0.10,
                )
            ),
            "counterfactual_action_effect_loss": float(
                getattr(
                    self,
                    "counterfactual_action_effect_loss_weight",
                    0.10,
                )
            ),
            "counterfactual_route_invariance_loss": float(
                getattr(
                    self,
                    "counterfactual_route_invariance_loss_weight",
                    0.05,
                )
            ),
            "counterfactual_world_primary_removal_loss": float(
                getattr(
                    self,
                    "counterfactual_world_primary_removal_loss_weight",
                    0.05,
                )
            ),
            "counterfactual_world_invariance_loss": float(
                getattr(
                    self,
                    "counterfactual_world_invariance_loss_weight",
                    0.02,
                )
            ),
        }

        if bool(getattr(self,"use_participant_spatial_attention_supervision",False,)):
            loss_weights[
                "participant_spatial_attention_loss"
            ] = float(
                getattr(
                    self,
                    "participant_spatial_attention_loss_weight",
                    0.05,
                )
            )

        if bool(getattr(self,"use_future_interaction_prediction",False,)):
            channel_weights = tuple(
                float(value)
                for value in getattr(
                    self,
                    "future_interaction_channel_weights",
                    (1.0, 1.0, 2.0, 4.0),
                )
            )
            if len(channel_weights) != len(
                FUTURE_INTERACTION_CHANNEL_KEYS
            ):
                raise ValueError(
                    "future_interaction_channel_weights must contain "
                    "exactly four values."
                )

            future_loss_weight = float(
                getattr(
                    self,
                    "future_interaction_loss_weight",
                    0.05,
                )
            )
            dice_loss_weight = float(
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
                    f"future_interaction_{channel_key}_bce_loss"
                ] = future_loss_weight * channel_weight
                loss_weights[
                    f"future_interaction_{channel_key}_dice_loss"
                ] = (
                    future_loss_weight
                    * channel_weight
                    * dice_loss_weight
                )







        ############################################## 汇总成总 loss 并返回 ##############################################

        return summarise_losses(loss_dict_only_losses,weights=loss_weights,), loss_logs



        

    def _eos_token_id(self) -> int:
        if self.language_model.variant == "OpenGVLab/InternVL2-4B":
            return self.tokenizer.added_tokens_encoder["<|end|>"]
        if self.language_model.variant == "OpenGVLab/InternVL2-2B":
            return self.tokenizer.added_tokens_encoder["<|im_end|>"]
        return self.tokenizer.eos_token_id

    def _store_predictions(self, predictions: Dict) -> None:
        for key, value in predictions.items():
            if value is None:
                continue
            current_value = getattr(self, key, None)
            if isinstance(value, torch.Tensor):
                setattr(
                    self,
                    key,
                    self._append_tensor(current_value, value),
                )
            elif isinstance(value, list):
                if current_value is None:
                    setattr(self, key, list(value))
                else:
                    current_value.extend(value)
            else:
                raise NotImplementedError(
                    f"Type of {key} is not supported."
                )

    # 推理接口
    def forward(self,example: DrivingExample,return_language: Optional[bool] = None,prompt_ids: Optional[Tensor] = None,) -> DrivingOutput:

        del return_language, prompt_ids

        self.speed_wps = None
        self.route = None
        self.language = []
        self.future_interaction_logits = None
        self.primary_spatial_attention = None
        self.secondary_spatial_attention = None
        self.contextual_interaction_tokens = None

        # 获取landuage embedding和driving embedding
        adaptor_dict = self._replace_multimodal_placeholders(example,inference=True,)
        batch_size = int(adaptor_dict["language_inputs"].shape[0])

        for batch_index in range(batch_size):
            batch_slice = slice(batch_index, batch_index + 1)
            interaction_outputs = (
                self._build_pre_language_interaction(
                    adaptor_dict,
                    batch_slice=batch_slice,
                )
            )
            interaction_tokens = interaction_outputs["interaction_tokens"].to(dtype=self._model_dtype())
            language_valid = interaction_outputs["language_valid"][0]
            prompt_embeddings = interaction_outputs["raw_language_features"][0, language_valid].unsqueeze(0).to(dtype=self._model_dtype())

            prefix_embeddings = torch.cat((interaction_tokens, prompt_embeddings),dim=1,)
            prefix_mask = torch.ones(
                prefix_embeddings.shape[:2],
                device=prefix_embeddings.device,
                dtype=torch.bool,
            )

            if self.predict_language:
                sampled_tokens, generated_embeddings = (
                    self.language_model.greedy_sample(
                        prefix_embeddings,
                        eos_token_id=self._eos_token_id(),
                        max_new_tokens=100,
                        input_embed_matrix=(
                            self.adaptors.language.embed_tokens.weight
                        ),
                        logit_matrix=(
                            self.adaptors.language.lm_head.weight
                        ),
                        attention_mask=prefix_mask,
                    )
                )
                self.language.append(
                    self.tokenizer.batch_decode(
                        sampled_tokens,
                        skip_special_tokens=True,
                    )[0]
                )
            else:
                generated_embeddings = prefix_embeddings

            driving_queries = interaction_outputs["raw_driving_queries"].to(dtype=self._model_dtype())
            full_embeddings = torch.cat((generated_embeddings, driving_queries),dim=1,)
            full_mask = torch.ones(
                full_embeddings.shape[:2],
                device=full_embeddings.device,
                dtype=torch.bool,
            )
            full_features = self.language_model.forward_features(
                embeddings=full_embeddings,
                attention_mask=full_mask,
                position_ids=None,
                return_dict=True,
            )

            contextual_interaction_tokens = full_features[:,: len(INTERACTION_TOKEN_KEYS),]
            driving_features = full_features[:,-self.interaction_reasoner.num_driving_queries :,]
            predictions = self.adaptors.driving.get_predictions(driving_features)
            self._store_predictions(predictions)

            if self.future_interaction_decoder is not None:
                future_logits = self.future_interaction_decoder(
                    contextual_interaction_tokens
                )
                self.future_interaction_logits = self._append_tensor(
                    self.future_interaction_logits,
                    future_logits,
                )

            self.primary_spatial_attention = self._append_tensor(
                self.primary_spatial_attention,
                interaction_outputs[
                    "primary_spatial_attention"
                ],
            )
            self.secondary_spatial_attention = self._append_tensor(
                self.secondary_spatial_attention,
                interaction_outputs[
                    "secondary_spatial_attention"
                ],
            )
            self.contextual_interaction_tokens = self._append_tensor(
                self.contextual_interaction_tokens,
                contextual_interaction_tokens,
            )

        return self.speed_wps, self.route, self.language
