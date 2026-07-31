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
    第二阶段统一语言—交互—动作模型。

    序列固定为：
        [4个统一交互token | 语言token | 30个Driving query]

    因此语言生成、轨迹预测和四通道未来世界预测不再通过相互独立的
    后处理分支读取交互信息，而是在同一次因果Transformer前向中共享证据。
    """

    def __init__(
        self,
        cfg_data_module,
        processor,
        cache_dir,
        **cfg,
    ):
        super().__init__(
            cfg_data_module=cfg_data_module,
            processor=processor,
            cache_dir=cache_dir,
            **cfg,
        )

        driving_adaptor = self.adaptors.driving
        if driving_adaptor is None:
            raise RuntimeError(
                "Stage-2 interaction reasoning requires DrivingAdaptor."
            )

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
        marker: str,
    ) -> Tuple[Tuple[int, ...], ...]:
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
        value: Tensor,
    ) -> Tensor:
        if current is None:
            return value
        return torch.cat((current, value), dim=0)

    def _model_dtype(self) -> torch.dtype:
        return self.adaptors.language.embed_tokens.weight.dtype

    def _replace_multimodal_placeholders(
        self,
        example: DrivingExample,
        *,
        inference: bool,
    ) -> Dict:
        adaptor_dict = self.adaptors(
            example,
            inference=inference,
        )
        prompt = (
            example.driving_input.prompt_inference
            if inference
            else example.driving_input.prompt
        )
        return self.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=adaptor_dict,
            pixel_values=example.driving_input.camera_images,
            placeholder_values=prompt.placeholder_values,
            wp_encoder=self.wp_encoder,
        )

    def _build_pre_language_interaction(
        self,
        adaptor_dict: Dict,
        *,
        batch_slice=slice(None),
    ) -> Dict[str, Tensor]:
        raw_language_features = adaptor_dict[
            "language_inputs"
        ][batch_slice]
        raw_driving_queries = adaptor_dict[
            "driving_inputs"
        ][batch_slice]
        language_ids = adaptor_dict[
            "language__ids"
        ][batch_slice]
        language_valid = adaptor_dict[
            "language_inputs_mask"
        ][batch_slice].bool()

        answer_mask = adaptor_dict.get(
            "language__ids_mask",
            None,
        )
        if isinstance(answer_mask, torch.Tensor):
            answer_mask = answer_mask[batch_slice].bool()
        else:
            answer_mask = torch.zeros_like(language_valid)

        image_encoder = self.vision_model.image_encoder
        image_context_token_id = getattr(
            image_encoder,
            "img_context_token_id",
            None,
        )
        target_point_token_id = getattr(
            image_encoder,
            "target_point_token_id",
            None,
        )
        if image_context_token_id is None:
            raise RuntimeError(
                "The image encoder did not initialize IMG_CONTEXT token id."
            )
        if target_point_token_id is None:
            raise RuntimeError(
                "The image encoder did not initialize TARGET_POINT token id."
            )

        visual_mask = (
            language_ids == int(image_context_token_id)
        )
        expected_visual_tokens = int(
            self.interaction_reasoner.num_visual_tokens
        )
        visual_token_count = visual_mask.sum(dim=1)
        if not torch.all(
            visual_token_count == expected_visual_tokens
        ):
            raise RuntimeError(
                "Stage-2 interaction reasoning requires exactly "
                f"{expected_visual_tokens} visual tokens per sample, "
                f"but received counts={visual_token_count.tolist()}."
            )
        visual_features = raw_language_features[
            visual_mask
        ].reshape(
            raw_language_features.shape[0],
            expected_visual_tokens,
            raw_language_features.shape[-1],
        )

        navigation_mask = (
            language_ids == int(target_point_token_id)
        )
        navigation_count = navigation_mask.sum(dim=1)
        if not torch.all(navigation_count == 2):
            raise RuntimeError(
                "Stage-2 interaction reasoning requires exactly two "
                "TARGET_POINT embeddings per sample, but received "
                f"counts={navigation_count.tolist()}."
            )
        navigation_context = raw_language_features[
            navigation_mask
        ].reshape(
            raw_language_features.shape[0],
            2,
            raw_language_features.shape[-1],
        ).mean(dim=1)

        # 交互token只读取用户问题、导航和视觉证据，不能读取训练答案。
        language_context_mask = (
            language_valid
            & ~visual_mask
            & ~answer_mask
        )

        reasoner_dtype = next(
            self.interaction_reasoner.parameters()
        ).dtype
        outputs = self.interaction_reasoner(
            interaction_query_features=raw_driving_queries.to(
                dtype=reasoner_dtype
            ),
            visual_features=visual_features.to(
                dtype=reasoner_dtype
            ),
            language_features=raw_language_features.to(
                dtype=reasoner_dtype
            ),
            language_context_mask=language_context_mask,
            navigation_context=navigation_context.to(
                dtype=reasoner_dtype
            ),
        )
        outputs["language_ids"] = language_ids
        outputs["language_valid"] = language_valid
        outputs["answer_mask"] = answer_mask
        outputs["raw_language_features"] = raw_language_features
        outputs["raw_driving_queries"] = raw_driving_queries
        return outputs

    @staticmethod
    def _assemble_joint_sequence(
        interaction_tokens: Tensor,
        language_features: Tensor,
        language_valid: Tensor,
        driving_queries: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[int, int, int]]:
        batch_size = int(language_features.shape[0])
        interaction_valid = torch.ones(
            interaction_tokens.shape[:2],
            device=interaction_tokens.device,
            dtype=torch.bool,
        )
        driving_valid = torch.ones(
            driving_queries.shape[:2],
            device=driving_queries.device,
            dtype=torch.bool,
        )

        original_embeddings = torch.cat(
            (
                interaction_tokens,
                language_features,
                driving_queries,
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
    def _split_joint_outputs(
        outputs: Tensor,
        permutation: Tensor,
        split_sizes: Tuple[int, int, int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
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

    def _forward_joint_transformer(
        self,
        interaction_outputs: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        interaction_tokens = interaction_outputs[
            "interaction_tokens"
        ].to(dtype=self._model_dtype())
        language_features = interaction_outputs[
            "raw_language_features"
        ].to(dtype=self._model_dtype())
        driving_queries = interaction_outputs[
            "raw_driving_queries"
        ].to(dtype=self._model_dtype())

        (
            joint_embeddings,
            joint_mask,
            permutation,
            split_sizes,
        ) = self._assemble_joint_sequence(
            interaction_tokens=interaction_tokens,
            language_features=language_features,
            language_valid=interaction_outputs["language_valid"],
            driving_queries=driving_queries,
        )
        joint_features = self.language_model.forward_features(
            embeddings=joint_embeddings,
            attention_mask=joint_mask,
            position_ids=None,
            return_dict=True,
        )
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
        start_index: int,
    ) -> Optional[Tuple[int, int]]:
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
        answer_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
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
        prediction_logits: Tensor,
    ) -> Tuple[Tensor, Tensor]:
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

    def _add_target_point_metadata(
        self,
        *,
        pred_labels: Dict,
        adaptor_dict: Dict,
        example: DrivingExample,
    ) -> None:
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
        secondary_exists: Tensor,
    ) -> None:
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

    def forward_loss(
        self,
        example: DrivingExample,
        per_sample: bool = False,
    ) -> TrainingOutput:
        adaptor_dict = self._replace_multimodal_placeholders(
            example,
            inference=False,
        )
        interaction_outputs = self._build_pre_language_interaction(
            adaptor_dict
        )
        (
            contextual_interaction_tokens,
            language_features,
            driving_features,
        ) = self._forward_joint_transformer(interaction_outputs)

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
        loss_dict.update(
            self.adaptors.driving.compute_loss(
                driving_features,
                None,
                {},
                example,
            )
        )

        question_span_mask, language_alignment_valid = (
            self._build_question_span_masks(
                interaction_outputs["language_ids"],
                interaction_outputs["answer_mask"],
            )
        )
        language_alignment_loss = (
            self.interaction_reasoner.compute_language_alignment_loss(
                language_features=language_features,
                contextual_interaction_tokens=(
                    contextual_interaction_tokens
                ),
                question_span_mask=question_span_mask,
                valid_mask=language_alignment_valid,
                temperature=float(
                    getattr(
                        self,
                        "interaction_alignment_temperature",
                        0.1,
                    )
                ),
            )
        )
        loss_dict["language_interaction_alignment_loss"] = (
            language_alignment_loss
        )

        loss_dict["action_interaction_alignment_loss"] = (
            self.interaction_reasoner.compute_action_alignment_loss(
                driving_features=driving_features,
                contextual_interaction_tokens=(
                    contextual_interaction_tokens
                ),
            )
        )

        future_target = None
        future_valid = torch.zeros(
            (driving_features.shape[0],),
            device=driving_features.device,
            dtype=torch.bool,
        )
        if bool(
            getattr(
                self,
                "use_future_interaction_prediction",
                False,
            )
        ):
            if self.future_interaction_decoder is None:
                raise RuntimeError(
                    "Future interaction prediction is enabled, but the "
                    "decoder was not initialized."
                )
            future_logits = self.future_interaction_decoder(
                contextual_interaction_tokens
            )
            future_target, future_valid = (
                self._prepare_future_supervision(
                    example,
                    future_logits,
                )
            )
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

        if bool(
            getattr(
                self,
                "use_participant_spatial_attention_supervision",
                False,
            )
        ):
            participant_target = (
                example.driving_label.camera_attention_target
            )
            participant_valid = (
                example.driving_label.camera_attention_valid
            )
            if not isinstance(participant_target, torch.Tensor):
                raise RuntimeError(
                    "Participant spatial supervision is enabled, but the "
                    "target tensor is missing."
                )
            if not isinstance(participant_valid, torch.Tensor):
                raise RuntimeError(
                    "Participant spatial supervision is enabled, but the "
                    "valid mask is missing."
                )
            loss_dict.update(
                compute_participant_spatial_attention_losses(
                    primary_attention=interaction_outputs[
                        "primary_spatial_attention"
                    ],
                    target_attention=participant_target,
                    valid_mask=participant_valid,
                )
            )

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
                primary_attention=interaction_outputs[
                    "primary_spatial_attention"
                ],
                secondary_attention=interaction_outputs[
                    "secondary_spatial_attention"
                ],
                contextual_interaction_tokens=(
                    contextual_interaction_tokens
                ),
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

        loss_dict_only_losses = {
            key: value
            for key, value in loss_dict.items()
            if key.endswith("loss")
        }
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
        }

        if bool(
            getattr(
                self,
                "use_participant_spatial_attention_supervision",
                False,
            )
        ):
            loss_weights[
                "participant_spatial_attention_loss"
            ] = float(
                getattr(
                    self,
                    "participant_spatial_attention_loss_weight",
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

        return summarise_losses(
            loss_dict_only_losses,
            weights=loss_weights,
        ), loss_logs

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

    def forward(
        self,
        example: DrivingExample,
        return_language: Optional[bool] = None,
        prompt_ids: Optional[Tensor] = None,
    ) -> DrivingOutput:
        del return_language, prompt_ids

        self.speed_wps = None
        self.route = None
        self.language = []
        self.future_interaction_logits = None
        self.primary_spatial_attention = None
        self.secondary_spatial_attention = None
        self.contextual_interaction_tokens = None

        adaptor_dict = self._replace_multimodal_placeholders(
            example,
            inference=True,
        )
        batch_size = int(
            adaptor_dict["language_inputs"].shape[0]
        )

        for batch_index in range(batch_size):
            batch_slice = slice(batch_index, batch_index + 1)
            interaction_outputs = (
                self._build_pre_language_interaction(
                    adaptor_dict,
                    batch_slice=batch_slice,
                )
            )
            interaction_tokens = interaction_outputs[
                "interaction_tokens"
            ].to(dtype=self._model_dtype())
            language_valid = interaction_outputs[
                "language_valid"
            ][0]
            prompt_embeddings = interaction_outputs[
                "raw_language_features"
            ][0, language_valid].unsqueeze(0).to(
                dtype=self._model_dtype()
            )

            prefix_embeddings = torch.cat(
                (interaction_tokens, prompt_embeddings),
                dim=1,
            )
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

            driving_queries = interaction_outputs[
                "raw_driving_queries"
            ].to(dtype=self._model_dtype())
            full_embeddings = torch.cat(
                (generated_embeddings, driving_queries),
                dim=1,
            )
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

            contextual_interaction_tokens = full_features[
                :,
                : len(INTERACTION_TOKEN_KEYS),
            ]
            driving_features = full_features[
                :,
                -self.interaction_reasoner.num_driving_queries :,
            ]
            predictions = self.adaptors.driving.get_predictions(
                driving_features
            )
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
