from typing import Dict, List, NamedTuple, Optional, Tuple, TypedDict
import torch
from torch import Tensor


class DatasetOutput(NamedTuple):  # 这是dataset_driving.py的返回值
    conversation: Optional[list]
    answer: Optional[str]
    image_ff: Optional[Tensor]
    image_ff_org_size: Optional[Tensor]
    waypoints: Optional[List[Tuple[float, float]]]
    waypoints_1d: Optional[List[Tuple[float, float]]]
    path: Optional[str]
    target_points: Optional[List[Tuple[float, float]]]
    speed: Optional[float]
    placeholder_values: Optional[Dict]
    measurement_path: Optional[str]
    dataset: Optional[str]
    qa_templates: Optional[Tuple[str, str]] = None
    eval_infos: Optional[Dict] = None

    # Common six-view visual input used by ordinary driving and LG data.
    # Defaults preserve compatibility with unrelated auxiliary datasets.
    image_surround: Optional[Tensor] = None
    image_surround_org_size: Optional[Tensor] = None
    camera_order: Optional[Tuple[str, ...]] = None

    # 第一阶段兼容字段：实际保存主要关键参与者的[6,64]视觉token空间目标。
    camera_attention_target: Optional[Tensor] = None
    camera_attention_valid: Optional[bool] = None

    #修改20260726：LG单样本的结构化未来世界监督。
    future_interaction_grid: Optional[Tensor] = None
    future_interaction_valid: Optional[bool] = None

    # 反事实对象移除后的真实重规划轨迹监督。
    counterfactual_waypoints: Optional[List[Tuple[float, float]]] = None
    counterfactual_waypoints_valid: Optional[bool] = None
    counterfactual_causal_score: Optional[float] = None
    counterfactual_conversation: Optional[list] = None


class LanguageLabel(NamedTuple):
    phrase_ids: Tensor  # [B, max(len(tokens))] int64
    phrase_valid: Tensor  # [B, max(len(tokens))] bool, valid, true => is fed into model
    phrase_mask: Tensor  # [B, max(len(tokens))] bool, mask, true => takes part in loss
    placeholder_values: list
    language_string: list
    loss_masking: Tensor


class DrivingOutput(NamedTuple):
    waypoints: Tensor  # [B, F, 2] float32
    # Auxiliary outputs (MUST be at the end):
    language_tokens: Tensor  # [B, max(len(tokens))]
    trajectory_tokens: Tensor  # [B, F, max(len(tokens))]


class TrainingOutput(NamedTuple):
    loss: Tensor  # [] floating
    loss_averages: Dict[str, Tensor]  # [] floating
    loss_values: Dict[str, Tensor]  # [B] floating
    loss_counts: Dict[str, Tensor]  # [B] int64

    driving_output: Optional[DrivingOutput] = None


class DrivingInput(NamedTuple):
    camera_images: torch.Tensor  # [B, T, N, C, H, W] uint8 [0, 255]
    image_sizes: torch.Tensor
    camera_intrinsics: torch.Tensor  # [B, N, 3, 3] float32
    camera_extrinsics: torch.Tensor  # [B, N, 4, 4] float32
    vehicle_speed: torch.Tensor  # [B, S] float32 ms
    target_point: torch.Tensor  # [B, 2] float32
    prompt: LanguageLabel
    prompt_inference: LanguageLabel
    counterfactual_prompt: Optional[LanguageLabel] = None


class DrivingLabel(NamedTuple):
    waypoints: Tensor  # [B, F, 2] 11 future waypoints 0.2s apart
    path: Tensor
    answer: LanguageLabel
    image_ff_org: Tensor
    eval_infos: Optional[Dict] = None

    # 第一阶段兼容字段：batch级主要关键参与者空间监督，形状[B,6,64]。
    camera_attention_target: Optional[Tensor] = None
    camera_attention_valid: Optional[Tensor] = None

    #修改20260726：batch级LG结构化未来世界监督。
    future_interaction_grid: Optional[Tensor] = None
    future_interaction_valid: Optional[Tensor] = None

    # batch级反事实对象移除重规划轨迹，形状[B,F,2]。
    counterfactual_waypoints: Optional[Tensor] = None
    counterfactual_waypoints_valid: Optional[Tensor] = None
    counterfactual_causal_score: Optional[Tensor] = None


class DrivingExample(NamedTuple):
    driving_input: DrivingInput
    driving_label: DrivingLabel
    run_id: List[str]
    qa_templates: Optional[Tuple[str, str]] = None
