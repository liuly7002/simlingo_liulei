# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

from simlingo_training.models.driving_interaction import InteractionGroundedDrivingModel
from simlingo_training.models.decision_future_interaction import (
    FutureInteractionDecoder,
    FUTURE_INTERACTION_CHANNEL_KEYS,
    compute_future_interaction_losses,
)
from simlingo_training.models.decision_interaction_reasoning import (
    INTERACTION_TOKEN_KEYS,
    PreLanguageInteractionReasoner,
    DecisionConditionedTrajectoryRefinement,
    compute_counterfactual_attention_suppression_loss,
    compute_counterfactual_future_world_losses,
    compute_decision_refinement_losses,
    compute_participant_spatial_attention_losses,
)
from simlingo_training.models.utils import summarise_losses
from simlingo_training.utils.custom_types import DrivingExample


class DecisionInteractionDrivingModel(InteractionGroundedDrivingModel):

    def __init__(self, cfg_data_module, processor, cache_dir, **cfg):
        super().__init__(cfg_data_module=cfg_data_module, processor=processor,
                         cache_dir=cache_dir, **cfg)
        if not bool(getattr(self, "use_future_interaction_prediction", False)):
            raise ValueError("Decision refinement requires predicted future interaction supervision.")
        driving = self.adaptors.driving
        # Keep the original 4-channel model untouched for controlled ablations.
        # The stage-2 model installs its own 3-semantic-token reasoner/decoder.
        self.interaction_reasoner = PreLanguageInteractionReasoner(
            hidden_size=int(self.language_model.hidden_size),
            num_route_queries=int(driving.future_waypoints),
            num_ego_queries=int(driving.future_speed_waypoints),
            num_cameras=6,
            tokens_per_camera=64,
            attention_dim=int(getattr(self, "interaction_attention_dim", 128)),
            num_heads=int(getattr(self, "interaction_num_heads", 8)),
            dropout=float(getattr(self, "interaction_dropout", 0.1)),
        )
        self.future_interaction_decoder = FutureInteractionDecoder(
            hidden_size=int(self.language_model.hidden_size),
            output_channels=3,
            output_size=int(getattr(self, "future_interaction_output_size", 128)),
            future_steps=int(driving.future_speed_waypoints),
        )
        self.trajectory_refinement = DecisionConditionedTrajectoryRefinement(
            hidden_size=int(self.language_model.hidden_size),
            future_steps=int(driving.future_speed_waypoints),
            num_heads=int(getattr(self, "interaction_num_heads", 8)),
            dropout=float(getattr(self, "interaction_dropout", 0.1)),
        )
        self.future_interaction_temporal_logits = None
        self.initial_speed_wps = None

    @staticmethod
    def _world_targets(example: DrivingExample, prediction: Dict[str, Tensor]):
        logits = prediction["grid_logits"]
        temporal = prediction["temporal_logits"]
        b = logits.shape[0]
        label = example.driving_label
        grid = label.future_interaction_grid
        time = label.future_interaction_temporal
        valid = label.future_interaction_valid
        status = label.critical_actor_status
        if grid is None:
            grid = logits.new_zeros((b, 3, *logits.shape[-2:]))
            valid = torch.zeros(b, device=logits.device, dtype=torch.bool)
        else:
            grid = grid.to(device=logits.device, dtype=torch.float32)
            if grid.ndim != 4 or grid.shape[:2] != (b, 3):
                raise ValueError(f"Expected [B,3,H,W] summary targets, got {grid.shape}")
            if not isinstance(valid, Tensor):
                raise ValueError("Summary targets require a valid mask")
            valid = valid.to(device=logits.device, dtype=torch.bool)
        if time is None:
            if valid.any():
                raise ValueError("Valid summary targets require C1/C2 temporal occupancy")
            time = temporal.new_zeros(temporal.shape)
        else:
            time = time.to(device=logits.device, dtype=torch.float32)
        if time.ndim != 5 or time.shape[:3] != temporal.shape[:3]:
            raise ValueError(f"Expected [B,2,T,H,W] temporal targets, got {time.shape}")
        if status is None:
            status = torch.full((b,), -1, device=logits.device, dtype=torch.int8)
        else:
            status = status.to(device=logits.device, dtype=torch.int8)
        if status.shape != (b,):
            raise ValueError("Actor statuses must be [B]")
        return grid, time, valid, status

    def _counterfactual_losses(
        self, example: DrivingExample, adaptor_dict: Dict,
        contextual_tokens: Tensor, language_features: Tensor,
        interaction_outputs: Dict, question_mask: Tensor,
        question_valid: Tensor, full_prediction: Dict,
        full_world: Dict, world_gt: Tensor,
        world_valid: Tensor, actor_status: Tensor,) -> Tuple[Dict, Dict]:

        b = contextual_tokens.shape[0]
        zero = contextual_tokens.float().sum(dim=(1, 2)) * 0.0
        zero = zero + self.interaction_reasoner.counterfactual_parameter_zero().float()
        zeros = torch.zeros_like(zero)
        names = (
            "counterfactual_visual_suppression_loss",
            "counterfactual_language_consistency_loss",
            "counterfactual_action_target_loss",
            "counterfactual_action_effect_loss",
            "counterfactual_route_invariance_loss",
            "counterfactual_world_primary_removal_loss",
            "counterfactual_world_invariance_loss",
        )
        empty = {name: (zero, zeros) for name in names}
        diagnostics = {"counterfactual_selected": torch.zeros(b, dtype=torch.bool, device=zero.device)}
        if not bool(getattr(self, "use_counterfactual_intervention_training", False)):
            return empty, diagnostics

        label = example.driving_label
        targets = (label.camera_attention_target, label.camera_attention_valid,
                   label.counterfactual_waypoints, label.counterfactual_waypoints_valid,
                   label.counterfactual_causal_score)
        if not all(isinstance(x, Tensor) for x in targets):
            raise RuntimeError("Counterfactual training requires spatial and replanned waypoint labels")
        spatial, spatial_valid, cf_waypoints, cf_valid, causal_score = targets
        spatial = spatial.to(device=zero.device, dtype=torch.float32)
        spatial_valid = spatial_valid.to(device=zero.device, dtype=torch.bool)
        cf_waypoints = cf_waypoints.to(device=zero.device, dtype=torch.float32)
        cf_valid = cf_valid.to(device=zero.device, dtype=torch.bool)
        causal_score = causal_score.to(device=zero.device, dtype=torch.float32)
        eligible = spatial_valid & cf_valid & (actor_status == 1) & world_valid
        selected = self._select_counterfactual_indices(eligible)
        if selected.numel() == 0:
            return empty, diagnostics
        diagnostics["counterfactual_selected"][selected] = True
        cf_inputs = self._build_counterfactual_adaptor_dict(example, adaptor_dict)
        cf_out = self._build_pre_language_interaction(
            cf_inputs,
            batch_slice=selected,
            visual_intervention_target=spatial[selected],
            visual_intervention_valid=torch.ones(selected.numel(), dtype=torch.bool, device=zero.device),
        )
        cf_tokens, cf_language, cf_driving = self._forward_joint_transformer(cf_out)
        cf_predictions = self.adaptors.driving.get_predictions(cf_driving)
        cf_world = self.future_interaction_decoder(cf_tokens)
        cf_refined = self.trajectory_refinement(
            cf_predictions["speed_wps"], cf_tokens, cf_world["temporal_logits"]
        )
        full_target = label.waypoints.to(device=zero.device, dtype=torch.float32)[selected]
        target_removed = cf_waypoints[selected]
        full_waypoints = full_prediction["final_waypoints"][selected]
        removed_waypoints = cf_refined["final_waypoints"]
        if not (full_waypoints.shape == removed_waypoints.shape == full_target.shape == target_removed.shape):
            raise ValueError("Counterfactual and full-scene trajectory shapes differ")
        confidence = causal_score[selected].clamp(min=0.25, max=2.0)

        def smooth(pred, target):
            return F.smooth_l1_loss(pred.float(), target.float(), reduction="none").sum(-1).mean(-1)

        out = {}
        cf_target_loss = smooth(removed_waypoints, target_removed)
        cf_effect_loss = smooth(full_waypoints - removed_waypoints, full_target - target_removed)
        out["counterfactual_action_target_loss"] = (cf_target_loss * confidence, confidence)
        out["counterfactual_action_effect_loss"] = (cf_effect_loss * confidence, confidence)
        route_target = label.path.to(device=zero.device, dtype=torch.float32)[selected]
        route_pred = cf_predictions["route"].float()
        if route_pred.shape != route_target.shape:
            raise ValueError("Counterfactual route and supervised route differ in shape")
        route_error = smooth(route_pred, route_target)
        out["counterfactual_route_invariance_loss"] = (route_error, torch.ones_like(route_error))
        out.update(compute_counterfactual_attention_suppression_loss(
            counterfactual_primary_attention=cf_out["primary_spatial_attention"],
            intervention_mask=cf_out["visual_intervention_mask"],
            valid_mask=torch.ones(selected.numel(), dtype=torch.bool, device=zero.device),
        ))
        cf_question_mask, cf_question_valid = self._build_question_span_masks(
            cf_out["language_ids"], cf_out["answer_mask"]
        )
        consistency_valid = (
            question_valid[selected] & cf_question_valid
        ).float() * confidence
        out["counterfactual_language_consistency_loss"] = (
            self.interaction_reasoner.compute_counterfactual_language_consistency_loss(
                original_language_features=language_features[selected],
                counterfactual_language_features=cf_language,
                original_interaction_tokens=contextual_tokens[selected],
                counterfactual_interaction_tokens=cf_tokens,
                original_question_span_mask=question_mask[selected],
                counterfactual_question_span_mask=cf_question_mask,
                full_scene_waypoints=full_target,
                counterfactual_waypoints=target_removed,
                valid_mask=consistency_valid,
                minimum_change=float(getattr(self, "counterfactual_language_min_change", 0.02)),
            )
        )
        out.update(compute_counterfactual_future_world_losses(
            counterfactual_logits=cf_world["grid_logits"],
            full_scene_target=world_gt[selected],
            valid_mask=world_valid[selected],
        ))
        # Return B-sized loss vectors, not selected-only tensors. The original
        # summarise_losses and per-sample callbacks index losses by batch item.
        for key in names:
            if key not in out:
                continue
            v, n = out[key]
            scatter_v = torch.zeros_like(zero).scatter(0, selected, v.float())
            scatter_n = torch.zeros_like(zero).scatter(0, selected, n.float())
            empty[key] = (scatter_v, scatter_n)
        return empty, diagnostics

    def _loss_weights(self) -> Dict[str, float]:
        get = lambda key, default: float(getattr(self, key, default))
        weights = {
            "route_loss": 1.0,
            "speed_wps_loss": get("initial_ego_waypoints_loss_weight", 0.10),
            "initial_ego_waypoints_aux_loss": 0.0,  # Same initial objective as speed_wps_loss.
            "final_ego_waypoints_loss": get("final_ego_waypoints_loss_weight", 1.0),
            "critical_actor_existence_loss": get("critical_actor_existence_loss_weight", 0.05),
            "language_interaction_alignment_loss": get("language_interaction_alignment_loss_weight", 0.05),
            "action_interaction_alignment_loss": get("action_interaction_alignment_loss_weight", 0.05),
            "participant_spatial_attention_loss": get("participant_spatial_attention_loss_weight", 0.05),
            "counterfactual_visual_suppression_loss": get("counterfactual_visual_suppression_loss_weight", 0.05),
            "counterfactual_language_consistency_loss": get("counterfactual_language_consistency_loss_weight", 0.05),
            "counterfactual_action_target_loss": get("counterfactual_action_target_loss_weight", 0.10),
            "counterfactual_action_effect_loss": get("counterfactual_action_effect_loss_weight", 0.10),
            "counterfactual_route_invariance_loss": get("counterfactual_route_invariance_loss_weight", 0.05),
            "counterfactual_world_primary_removal_loss": get("counterfactual_world_primary_removal_loss_weight", 0.05),
            "counterfactual_world_invariance_loss": get("counterfactual_world_invariance_loss_weight", 0.02),
        }
        channel_weights = tuple(float(x) for x in getattr(
            self, "future_interaction_channel_weights", (1.0, 1.0, 2.0),
        ))
        if len(channel_weights) != 3:
            raise ValueError("Future interaction channel weights must have three entries")
        main = get("future_interaction_loss_weight", 0.05)
        temporal = get("future_interaction_temporal_loss_weight", 0.03)
        dice = get("future_interaction_dice_loss_weight", 1.0)
        for i, name in enumerate(FUTURE_INTERACTION_CHANNEL_KEYS):
            weights[f"future_interaction_{name}_bce_loss"] = main * channel_weights[i]
            weights[f"future_interaction_{name}_dice_loss"] = main * channel_weights[i] * dice
            if i > 0:
                weights[f"future_temporal_{name}_bce_loss"] = temporal * channel_weights[i]
                weights[f"future_temporal_{name}_dice_loss"] = temporal * channel_weights[i] * dice
        return weights

    def forward_loss(self, example: DrivingExample, per_sample: bool = False):
        adaptor_dict = self._replace_multimodal_placeholders(example, inference=False)
        interaction_outputs = self._build_pre_language_interaction(adaptor_dict)
        contextual, language, driving = self._forward_joint_transformer(interaction_outputs)
        language_input = {"_ids": interaction_outputs["language_ids"],
                          "_ids_mask": interaction_outputs["answer_mask"]}
        loss_dict = self.adaptors.language.compute_loss(language, None, language_input, example)
        loss_dict.update(self.adaptors.driving.compute_loss(driving, None, {}, example))
        initial_predictions = self.adaptors.driving.get_predictions(driving)
        if "speed_wps" not in initial_predictions or "route" not in initial_predictions:
            raise RuntimeError("DrivingAdaptor must predict both route and speed_wps")
        world_prediction = self.future_interaction_decoder(contextual)
        world_target, time_target, valid, status = self._world_targets(example, world_prediction)
        refinement = self.trajectory_refinement(
            initial_predictions["speed_wps"], contextual, world_prediction["temporal_logits"]
        )
        loss_dict.update(compute_decision_refinement_losses(
            initial_waypoints=refinement["initial_waypoints"],
            final_waypoints=refinement["final_waypoints"],
            target_waypoints=example.driving_label.waypoints,
            critical_actor_logits=refinement["critical_actor_logits"],
            critical_actor_status=status,
            actor_positive_weight=float(getattr(self, "critical_actor_positive_weight", 18.0)),
        ))
        loss_dict.update(compute_future_interaction_losses(
            prediction_logits=world_prediction["grid_logits"],
            target_grid=world_target,
            valid_mask=valid,
            positive_weights=tuple(float(x) for x in getattr(
                self, "future_interaction_positive_weights", (20.0, 20.0, 80.0),
            )),
            temporal_logits=world_prediction["temporal_logits"],
            temporal_target=time_target,
            critical_actor_status=status,
        ))
        question_mask, question_valid = self._build_question_span_masks(
            interaction_outputs["language_ids"], interaction_outputs["answer_mask"]
        )
        loss_dict["language_interaction_alignment_loss"] = (
            self.interaction_reasoner.compute_language_alignment_loss(
                language_features=language,
                contextual_interaction_tokens=contextual,
                question_span_mask=question_mask,
                valid_mask=question_valid,
                temperature=float(getattr(self, "interaction_alignment_temperature", 0.1)),
            )
        )
        loss_dict["action_interaction_alignment_loss"] = (
            self.interaction_reasoner.compute_action_alignment_loss(
                driving_features=driving, contextual_interaction_tokens=contextual,
            )
        )
        if bool(getattr(self, "use_participant_spatial_attention_supervision", False)):
            spatial_gt = example.driving_label.camera_attention_target
            spatial_valid = example.driving_label.camera_attention_valid
            if spatial_gt is None or spatial_valid is None:
                raise RuntimeError("Spatial supervision enabled without spatial targets")
            spatial_valid = spatial_valid.to(contextual.device).bool() & valid & (status == 1)
            loss_dict.update(compute_participant_spatial_attention_losses(
                primary_attention=interaction_outputs["primary_spatial_attention"],
                target_attention=spatial_gt,
                valid_mask=spatial_valid,
            ))
        cf_losses, cf_diagnostics = self._counterfactual_losses(
            example, adaptor_dict, contextual, language, interaction_outputs,
            question_mask, question_valid, refinement, world_prediction,
            world_target, valid, status,
        )
        loss_dict.update(cf_losses)
        only_losses = {k: v for k, v in loss_dict.items() if k.endswith("loss")}
        if per_sample:
            predictions = {k: v for k, v in loss_dict.items()
                           if not k.endswith("loss") and not k.endswith("log")}
            predictions["speed_wps_initial_prediction"] = initial_predictions["speed_wps"].detach()
            predictions["speed_wps_prediction"] = refinement["final_waypoints"].detach()
            predictions["final_ego_waypoints_prediction"] = refinement["final_waypoints"].detach()
            predictions["critical_actor_probability"] = torch.sigmoid(refinement["critical_actor_logits"]).detach()
            predictions["critical_actor_status"] = status.detach()
            predictions["primary_spatial_attention"] = interaction_outputs["primary_spatial_attention"].detach()
            predictions["language_alignment_valid"] = question_valid.detach()
            predictions["interaction_token_norms"] = contextual.detach().float().norm(dim=-1)
            predictions.update(cf_diagnostics)
            self._add_target_point_metadata(
                pred_labels=predictions, adaptor_dict=adaptor_dict, example=example,
            )
            return only_losses, predictions
        losses = summarise_losses(only_losses, weights=self._loss_weights())
        return losses, {}

    def forward(self, example: DrivingExample, return_language: Optional[bool] = None, prompt_ids: Optional[Tensor] = None):
        
        del return_language, prompt_ids  # 表示当前实现并不使用传入的这两个参数。是否生成语言由 self.predict_language 控制
        
        self.speed_wps = None
        self.route = None
        self.language = []
        self.future_interaction_logits = None
        self.future_interaction_temporal_logits = None
        self.primary_spatial_attention = None
        self.contextual_interaction_tokens = None
        self.initial_speed_wps = None
        
        adaptor_dict = self._replace_multimodal_placeholders(example, inference=True)  # [B,L+30,D]
        batch_size = int(adaptor_dict["language_inputs"].shape[0])
        
        for i in range(batch_size):
            sl = slice(i, i + 1)
            interaction = self._build_pre_language_interaction(adaptor_dict, batch_slice=sl)
            interaction_tokens = interaction["interaction_tokens"].to(dtype=self._model_dtype())
            valid = interaction["language_valid"][0]
            prompt = interaction["raw_language_features"][0, valid].unsqueeze(0).to(dtype=self._model_dtype())
            prefix = torch.cat((interaction_tokens, prompt), dim=1)
            prefix_mask = torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device)
            if self.predict_language:
                sampled, generated = self.language_model.greedy_sample(
                    prefix, eos_token_id=self._eos_token_id(), max_new_tokens=100,
                    input_embed_matrix=self.adaptors.language.embed_tokens.weight,
                    logit_matrix=self.adaptors.language.lm_head.weight,
                    attention_mask=prefix_mask,
                )
                self.language.append(self.tokenizer.batch_decode(sampled, skip_special_tokens=True)[0])
            else:
                generated = prefix
            queries = interaction["raw_driving_queries"].to(dtype=self._model_dtype())
            sequence = torch.cat((generated, queries), dim=1)
            features = self.language_model.forward_features(
                embeddings=sequence,
                attention_mask=torch.ones(sequence.shape[:2], device=sequence.device, dtype=torch.bool),
                position_ids=None, return_dict=True,
            )
            contextual = features[:, :len(INTERACTION_TOKEN_KEYS)]
            driving_features = features[:, -self.interaction_reasoner.num_driving_queries:]
            prediction = self.adaptors.driving.get_predictions(driving_features)
            world = self.future_interaction_decoder(contextual)
            refined = self.trajectory_refinement(
                prediction["speed_wps"], contextual, world["temporal_logits"]
            )
            self.initial_speed_wps = self._append_tensor(self.initial_speed_wps, prediction["speed_wps"])
            prediction["speed_wps"] = refined["final_waypoints"]
            self._store_predictions(prediction)
            self.future_interaction_logits = self._append_tensor(
                self.future_interaction_logits, world["grid_logits"]
            )
            self.future_interaction_temporal_logits = self._append_tensor(
                self.future_interaction_temporal_logits, world["temporal_logits"]
            )
            self.primary_spatial_attention = self._append_tensor(
                self.primary_spatial_attention, interaction["primary_spatial_attention"]
            )
            self.contextual_interaction_tokens = self._append_tensor(
                self.contextual_interaction_tokens, contextual
            )
        return self.speed_wps, self.route, self.language
