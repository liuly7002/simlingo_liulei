# -*- coding: utf-8 -*-

from typing import Dict, Optional

import torch
from torch import Tensor

from simlingo_training.models.driving import DrivingModel
from simlingo_training.models.future_interaction import (
    FUTURE_INTERACTION_CHANNEL_KEYS,
    compute_future_interaction_losses,
)
from simlingo_training.models.interaction_reasoning import (
    UnifiedInteractionReasoner,
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
    第一阶段统一决策交互模型。

    在原SimLingo语言和Driving Transformer输出之上构建四个共享交互token，
    并使以下任务共同读取同一表示：
        1. 主要/次要参与者六视角空间注意力；
        2. route与waypoint动作预测；
        3. C0/C1/C2/C4四通道结构化未来世界预测。
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
                "Unified interaction reasoning requires DrivingAdaptor."
            )

        self.interaction_reasoner = UnifiedInteractionReasoner(
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

    def _reason_from_raw_inputs(
        self,
        *,
        adaptor_dict: Dict,
        action_features: Tensor,
        batch_index: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if batch_index is None:
            batch_slice = slice(None)
        else:
            batch_slice = slice(batch_index, batch_index + 1)

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
                "Unified interaction reasoning requires exactly "
                f"{expected_visual_tokens} raw visual tokens per sample, "
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
                "Unified interaction reasoning requires exactly two "
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

        # 空间定位只能读取用户prompt、导航和视觉证据，不能读取训练阶段的
        # ground-truth assistant答案，避免通过语言标签直接泄漏关键actor。
        language_context_mask = (
            language_valid
            & ~visual_mask
            & ~answer_mask
        )

        return self.interaction_reasoner(
            interaction_query_features=raw_driving_queries,
            action_features=action_features,
            visual_features=visual_features,
            language_features=raw_language_features,
            language_context_mask=language_context_mask,
            navigation_context=navigation_context,
        )

    def _extract_interaction_inputs(
        self,
        adaptor_dict: Dict,
        adaptor_features: Tensor,
    ) -> Dict[str, Tensor]:
        features_by_adaptor = (
            self.adaptors.split_outputs_by_adaptor(
                adaptor_dict,
                adaptor_features,
            )
        )
        driving_features = features_by_adaptor.get(
            "driving",
            None,
        )
        if not isinstance(driving_features, torch.Tensor):
            raise RuntimeError(
                "Driving features are required for unified interaction "
                "reasoning."
            )

        return self._reason_from_raw_inputs(
            adaptor_dict=adaptor_dict,
            action_features=driving_features,
        )

    def _replace_driving_features(
        self,
        adaptor_dict: Dict,
        adaptor_features: Tensor,
        enhanced_driving_features: Tensor,
    ) -> Tensor:
        permutation = adaptor_dict["perm"]
        inverse_permutation = permutation.argsort(dim=-1)
        batch_index = torch.arange(
            permutation.shape[0],
            device=permutation.device,
        )[:, None]
        original_order_features = adaptor_features[
            batch_index,
            inverse_permutation,
        ]

        adaptor_names = list(self.adaptors.adaptors.keys())
        if "driving" not in adaptor_names:
            raise RuntimeError(
                "Driving adaptor is missing from the adaptor sequence."
            )
        split_sizes = [
            int(value)
            for value in adaptor_dict["split_sizes"]
        ]
        driving_index = adaptor_names.index("driving")
        driving_start = sum(split_sizes[:driving_index])
        driving_size = split_sizes[driving_index]

        if enhanced_driving_features.shape[1] != driving_size:
            raise RuntimeError(
                "Enhanced Driving feature count does not match adaptor "
                f"split size: {enhanced_driving_features.shape[1]} vs "
                f"{driving_size}."
            )

        updated_original_features = torch.cat(
            (
                original_order_features[:, :driving_start],
                enhanced_driving_features,
                original_order_features[
                    :,
                    driving_start + driving_size :,
                ],
            ),
            dim=1,
        )
        return updated_original_features[
            batch_index,
            permutation,
        ]

    def _build_interaction_for_inference(
        self,
        *,
        adaptor_dict: Dict,
        driving_features: Tensor,
        batch_index: int,
    ) -> Dict[str, Tensor]:
        return self._reason_from_raw_inputs(
            adaptor_dict=adaptor_dict,
            action_features=driving_features,
            batch_index=batch_index,
        )

    @staticmethod
    def _append_tensor(
        current: Optional[Tensor],
        value: Tensor,
    ) -> Tensor:
        if current is None:
            return value
        return torch.cat((current, value), dim=0)

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

        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example

        adaptor_dict = self.adaptors(
            example,
            inference=True,
        )
        adaptor_dict = (
            self.vision_model.image_encoder.replace_placeholder_tokens(
                adaptor_dict=adaptor_dict,
                pixel_values=driving_input.camera_images,
                placeholder_values=(
                    driving_input
                    .prompt_inference
                    .placeholder_values
                ),
                wp_encoder=self.wp_encoder,
            )
        )
        input_embeds_all = adaptor_dict["language_inputs"]
        attention_masks = adaptor_dict[
            "language_inputs_mask"
        ]

        if self.predict_language:
            inputs_driving = self.adaptors.driving(
                driving_input
            )
            len_driving = int(
                inputs_driving["inputs"].size(1)
            )

            for batch_index, (
                input_embed,
                attention_mask,
            ) in enumerate(
                zip(input_embeds_all, attention_masks)
            ):
                input_embed = input_embed.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

                if (
                    self.language_model.variant
                    == "OpenGVLab/InternVL2-4B"
                ):
                    eos = self.tokenizer.added_tokens_encoder[
                        "<|end|>"
                    ]
                elif (
                    self.language_model.variant
                    == "OpenGVLab/InternVL2-2B"
                ):
                    eos = self.tokenizer.added_tokens_encoder[
                        "<|im_end|>"
                    ]
                else:
                    eos = self.tokenizer.eos_token_id

                sampled_tokens, generated_embeds = (
                    self.language_model.greedy_sample(
                        input_embed,
                        eos_token_id=eos,
                        max_new_tokens=100,
                        input_embed_matrix=(
                            self.adaptors
                            .language
                            .embed_tokens
                            .weight
                        ),
                        logit_matrix=(
                            self.adaptors
                            .language
                            .lm_head
                            .weight
                        ),
                        attention_mask=attention_mask,
                    )
                )

                sample_driving_inputs = inputs_driving[
                    "inputs"
                ][batch_index].unsqueeze(0)
                input_embed_concat = torch.cat(
                    (
                        generated_embeds,
                        sample_driving_inputs,
                    ),
                    dim=1,
                )
                features, logits = self.language_model.forward(
                    input_embed_concat
                )
                driving_features = features[:, -len_driving:]
                driving_logits = logits[:, -len_driving:]

                interaction_outputs = (
                    self._build_interaction_for_inference(
                        adaptor_dict=adaptor_dict,
                        driving_features=driving_features,
                        batch_index=batch_index,
                    )
                )
                predictions = (
                    self.adaptors.driving.get_predictions(
                        interaction_outputs[
                            "enhanced_driving_features"
                        ],
                        driving_logits,
                    )
                )

                for key, value in predictions.items():
                    if value is None:
                        continue
                    current_value = getattr(self, key, None)
                    if isinstance(value, torch.Tensor):
                        setattr(
                            self,
                            key,
                            self._append_tensor(
                                current_value,
                                value,
                            ),
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

                if self.future_interaction_decoder is not None:
                    future_logits = self.future_interaction_decoder(
                        interaction_outputs[
                            "interaction_tokens"
                        ]
                    )
                    self.future_interaction_logits = (
                        self._append_tensor(
                            self.future_interaction_logits,
                            future_logits,
                        )
                    )

                self.primary_spatial_attention = (
                    self._append_tensor(
                        self.primary_spatial_attention,
                        interaction_outputs[
                            "primary_spatial_attention"
                        ],
                    )
                )
                self.secondary_spatial_attention = (
                    self._append_tensor(
                        self.secondary_spatial_attention,
                        interaction_outputs[
                            "secondary_spatial_attention"
                        ],
                    )
                )
                self.language.append(
                    self.tokenizer.batch_decode(
                        sampled_tokens,
                        skip_special_tokens=True,
                    )[0]
                )
        else:
            adaptor_features, _ = self.forward_model(
                driving_input,
                adaptor_dict,
            )
            interaction_outputs = self._extract_interaction_inputs(
                adaptor_dict,
                adaptor_features,
            )
            predictions = self.adaptors.driving.get_predictions(
                interaction_outputs[
                    "enhanced_driving_features"
                ]
            )
            for key, value in predictions.items():
                if value is not None:
                    setattr(self, key, value)

            if self.future_interaction_decoder is not None:
                self.future_interaction_logits = (
                    self.future_interaction_decoder(
                        interaction_outputs[
                            "interaction_tokens"
                        ]
                    )
                )
            self.primary_spatial_attention = (
                interaction_outputs[
                    "primary_spatial_attention"
                ]
            )
            self.secondary_spatial_attention = (
                interaction_outputs[
                    "secondary_spatial_attention"
                ]
            )

        return self.speed_wps, self.route, self.language

    def _log_participant_spatial_attention(
        self,
        mode: str,
        interaction_outputs: Dict[str, Tensor],
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
                    f"{mode}_participant_attention/"
                    f"{prefix}_{camera_name}",
                    camera_weight,
                    on_step=on_step,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

        entropy = interaction_outputs[
            "primary_spatial_attention_entropy"
        ].detach().float().mean()
        self.log(
            f"{mode}_participant_attention/"
            "primary_normalized_spatial_entropy",
            entropy,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        action_gate = torch.tanh(
            self.interaction_reasoner.action_gate.detach()
        ).float()
        self.log(
            f"{mode}_participant_attention/action_gate",
            action_gate,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

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
                "language__ids is required to save TARGET_POINT "
                "coordinates."
            )

        target_point_token_id = getattr(
            image_encoder,
            "target_point_token_id",
            None,
        )
        if target_point_token_id is None:
            raise RuntimeError(
                "target_point_token_id was not initialized by the "
                "InternVL2 image encoder."
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
                "The number of placeholder value dictionaries does not "
                f"match batch size: {len(placeholder_values)} vs "
                f"{batch_size}."
            )

        for sample_index in range(batch_size):
            if not bool(has_target_point[sample_index].item()):
                continue
            sample_placeholder_values = placeholder_values[
                sample_index
            ]
            if target_point_token_id not in sample_placeholder_values:
                raise KeyError(
                    "The prompt contains TARGET_POINT, but its value is "
                    "missing from placeholder_values."
                )
            sample_target_points = torch.as_tensor(
                sample_placeholder_values[
                    target_point_token_id
                ],
                device=language_ids.device,
                dtype=torch.float32,
            )
            if sample_target_points.numel() != 4:
                raise ValueError(
                    "TARGET_POINT must contain two 2D coordinates, "
                    f"but received {sample_target_points.numel()} values."
                )
            target_point_coordinates[sample_index] = (
                sample_target_points.reshape(2, 2)
            )

        pred_labels["target_point_coordinates"] = (
            target_point_coordinates
        )
        pred_labels["has_target_point"] = has_target_point

    def forward_loss(
        self,
        example: DrivingExample,
        per_sample: bool = False,
    ) -> TrainingOutput:
        adaptor_dict = self.adaptors(example)
        adaptor_features, adaptor_logits = self.forward_model(
            example.driving_input,
            adaptor_dict,
            driving_labels=example.driving_label,
        )

        interaction_outputs = self._extract_interaction_inputs(
            adaptor_dict,
            adaptor_features,
        )
        enhanced_adaptor_features = self._replace_driving_features(
            adaptor_dict,
            adaptor_features,
            interaction_outputs[
                "enhanced_driving_features"
            ],
        )
        loss_dict = self.adaptors.compute_loss(
            enhanced_adaptor_features,
            adaptor_logits,
            adaptor_dict,
            example,
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

            future_interaction_logits = (
                self.future_interaction_decoder(
                    interaction_outputs[
                        "interaction_tokens"
                    ]
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
                        "future_interaction_valid contains True, but "
                        "future_interaction_grid is None."
                    )

                future_interaction_target = torch.zeros(
                    (
                        batch_size,
                        4,
                        future_interaction_logits.shape[-2],
                        future_interaction_logits.shape[-1],
                    ),
                    device=future_interaction_logits.device,
                    dtype=torch.float32,
                )
                future_interaction_valid = torch.zeros(
                    (batch_size,),
                    device=future_interaction_logits.device,
                    dtype=torch.bool,
                )
            elif not isinstance(
                future_interaction_valid,
                torch.Tensor,
            ):
                raise RuntimeError(
                    "A future interaction target exists, but its valid "
                    "mask is missing."
                )

            loss_dict.update(
                compute_future_interaction_losses(
                    prediction_logits=future_interaction_logits,
                    target_grid=future_interaction_target,
                    valid_mask=future_interaction_valid,
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
                example
                .driving_label
                .camera_attention_target
            )
            participant_valid = (
                example
                .driving_label
                .camera_attention_valid
            )
            if not isinstance(participant_target, torch.Tensor):
                raise RuntimeError(
                    "Participant spatial attention supervision is enabled, "
                    "but the target tensor is missing."
                )
            if not isinstance(participant_valid, torch.Tensor):
                raise RuntimeError(
                    "Participant spatial attention supervision is enabled, "
                    "but the valid mask is missing."
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

            valid_count = participant_valid.float().sum()
            valid_ratio = valid_count / max(
                int(participant_valid.numel()),
                1,
            )
            mode = "train" if self.training else "val"
            self.log(
                f"{mode}_participant_attention/valid_sample_count",
                valid_count,
                on_step=self.training,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=int(participant_valid.numel()),
                sync_dist=True,
            )
            self.log(
                f"{mode}_participant_attention/valid_sample_ratio",
                valid_ratio,
                on_step=self.training,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=int(participant_valid.numel()),
                sync_dist=True,
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
            interaction_outputs[
                "interaction_tokens"
            ].detach().float().norm(dim=-1)
        )

        self._log_participant_spatial_attention(
            "train" if self.training else "val",
            interaction_outputs,
        )

        if per_sample:
            self._add_target_point_metadata(
                pred_labels=pred_labels,
                adaptor_dict=adaptor_dict,
                example=example,
            )
            return loss_dict_only_losses, pred_labels

        loss_weights = {}
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

        if len(loss_weights) == 0:
            loss_weights = None

        return summarise_losses(
            loss_dict_only_losses,
            weights=loss_weights,
        ), loss_logs
