# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# 四通道名称、顺序与统一交互token严格一一对应。
FUTURE_INTERACTION_CHANNEL_KEYS = (
    "c0_route",
    "c1_ego_future",
    "c2_primary_actor",
    "c4_secondary_actor",
)


class _SingleInteractionChannelDecoder(nn.Module):
    """由一个具有明确语义的交互token解码一个结构化未来通道。"""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.output_size = int(output_size)
        self.seed_channels = 64
        self.seed_size = 8

        self.seed_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                self.seed_channels
                * self.seed_size
                * self.seed_size,
            ),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.ConvTranspose2d(
                16,
                8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
            nn.Conv2d(
                8,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, interaction_token: Tensor) -> Tensor:
        if interaction_token.ndim != 2:
            raise ValueError(
                "One interaction token must have shape [B,D], "
                f"but received {tuple(interaction_token.shape)}."
            )

        seed = self.seed_projection(interaction_token)
        seed = seed.reshape(
            interaction_token.shape[0],
            self.seed_channels,
            self.seed_size,
            self.seed_size,
        )
        logits = self.decoder(seed)

        expected_shape = (
            interaction_token.shape[0],
            1,
            self.output_size,
            self.output_size,
        )
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected single-channel interaction prediction shape: "
                f"expected {expected_shape}, received "
                f"{tuple(logits.shape)}."
            )
        return logits


class FutureInteractionDecoder(nn.Module):
    """
    使用四个统一决策交互token分别解码四通道结构化未来世界。

    输入：
        interaction_tokens: [B,4,D]

    token与输出通道的固定对应关系：
        route token           -> C0 route
        ego-future token      -> C1 ego future
        primary-actor token   -> C2 primary actor
        secondary-actor token -> C4 secondary actor

    输出：
        prediction_logits: [B,4,128,128]
    """

    def __init__(
        self,
        hidden_size: int,
        output_channels: int = 4,
        output_size: int = 128,
    ):
        super().__init__()

        if output_channels != len(FUTURE_INTERACTION_CHANNEL_KEYS):
            raise ValueError(
                "FutureInteractionDecoder requires exactly "
                f"{len(FUTURE_INTERACTION_CHANNEL_KEYS)} channels, "
                f"but received {output_channels}."
            )
        if output_size != 128:
            raise ValueError(
                "The current future-interaction decoder supports only "
                f"output_size=128, but received {output_size}."
            )

        self.output_channels = int(output_channels)
        self.output_size = int(output_size)
        self.channel_decoders = nn.ModuleList(
            [
                _SingleInteractionChannelDecoder(
                    hidden_size=hidden_size,
                    output_size=output_size,
                )
                for _ in FUTURE_INTERACTION_CHANNEL_KEYS
            ]
        )

    def forward(
        self,
        interaction_tokens: Tensor,
    ) -> Tensor:
        if interaction_tokens.ndim != 3:
            raise ValueError(
                "FutureInteractionDecoder expects interaction tokens "
                f"with shape [B,4,D], but received "
                f"{tuple(interaction_tokens.shape)}."
            )
        if interaction_tokens.shape[1] != self.output_channels:
            raise ValueError(
                "The number of interaction tokens must match the four "
                f"semantic channels: expected {self.output_channels}, "
                f"received {interaction_tokens.shape[1]}."
            )

        channel_logits = [
            decoder(interaction_tokens[:, channel_index])
            for channel_index, decoder in enumerate(
                self.channel_decoders
            )
        ]
        prediction_logits = torch.cat(channel_logits, dim=1)

        expected_shape = (
            interaction_tokens.shape[0],
            self.output_channels,
            self.output_size,
            self.output_size,
        )
        if tuple(prediction_logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected future interaction prediction shape: "
                f"expected {expected_shape}, received "
                f"{tuple(prediction_logits.shape)}."
            )

        return prediction_logits


def _resize_future_interaction_target(
    target_grid: Tensor,
    output_size: Tuple[int, int],
) -> Tensor:
    """
    C0/C1采用面积平均；C2/C4采用占用保持式最大池化。

    参与者通道非常稀疏，统一面积平均会把小目标footprint稀释成接近零的值。
    """

    if target_grid.shape[-2:] == output_size:
        return target_grid

    dense_channels = F.interpolate(
        target_grid[:, :2],
        size=output_size,
        mode="area",
    )

    actor_channels = target_grid[:, 2:]
    input_height, input_width = actor_channels.shape[-2:]
    output_height, output_width = output_size

    if (
        input_height >= output_height
        and input_width >= output_width
    ):
        actor_channels = F.adaptive_max_pool2d(
            actor_channels,
            output_size=output_size,
        )
    else:
        actor_channels = F.interpolate(
            actor_channels,
            size=output_size,
            mode="nearest",
        )

    return torch.cat(
        (dense_channels, actor_channels),
        dim=1,
    )


def compute_future_interaction_losses(
    prediction_logits: Tensor,
    target_grid: Tensor,
    valid_mask: Tensor,
    positive_weights: Sequence[float],
    dice_smooth: float = 1e-6,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """
    计算分通道的加权BCE和Dice损失。

    BCE：所有valid=True样本均参与，包括C2/C4为全零的负样本。
    Dice：仅在该通道真实标签非零时计算。
    """

    if prediction_logits.ndim != 4:
        raise ValueError(
            "Future interaction prediction must have shape "
            f"[B,4,H,W], but received "
            f"{tuple(prediction_logits.shape)}."
        )
    if target_grid.ndim != 4:
        raise ValueError(
            "Future interaction target must have shape "
            f"[B,4,H,W], but received "
            f"{tuple(target_grid.shape)}."
        )
    if prediction_logits.shape[0] != target_grid.shape[0]:
        raise ValueError(
            "Future interaction prediction and target batch sizes "
            "do not match."
        )
    if (
        prediction_logits.shape[1]
        != len(FUTURE_INTERACTION_CHANNEL_KEYS)
        or target_grid.shape[1]
        != len(FUTURE_INTERACTION_CHANNEL_KEYS)
    ):
        raise ValueError(
            "Future interaction prediction and target must both "
            "contain exactly four channels."
        )
    if valid_mask.shape != (prediction_logits.shape[0],):
        raise ValueError(
            "future_interaction_valid must have shape [B], "
            f"but received {tuple(valid_mask.shape)}."
        )
    if len(positive_weights) != len(FUTURE_INTERACTION_CHANNEL_KEYS):
        raise ValueError(
            "future_interaction_positive_weights must contain four "
            f"values, but received {len(positive_weights)}."
        )

    logits_float = prediction_logits.float()
    target_grid = target_grid.to(
        device=logits_float.device,
        dtype=torch.float32,
    )
    valid_mask = valid_mask.to(
        device=logits_float.device,
        dtype=torch.bool,
    )

    if not bool(torch.isfinite(target_grid).all().item()):
        raise ValueError(
            "Future interaction target contains non-finite values."
        )
    if bool(
        ((target_grid < 0.0) | (target_grid > 1.0)).any().item()
    ):
        raise ValueError(
            "Future interaction target values must be within [0, 1]."
        )

    target_grid = _resize_future_interaction_target(
        target_grid,
        output_size=tuple(prediction_logits.shape[-2:]),
    )

    positive_weight_tensor = torch.as_tensor(
        positive_weights,
        device=logits_float.device,
        dtype=torch.float32,
    ).reshape(1, len(FUTURE_INTERACTION_CHANNEL_KEYS), 1, 1)

    bce_map = F.binary_cross_entropy_with_logits(
        logits_float,
        target_grid,
        reduction="none",
        pos_weight=positive_weight_tensor,
    )
    bce_per_sample_channel = bce_map.mean(dim=(-2, -1))

    prediction_probability = torch.sigmoid(logits_float)
    intersection = (
        prediction_probability * target_grid
    ).sum(dim=(-2, -1))
    denominator = (
        prediction_probability.sum(dim=(-2, -1))
        + target_grid.sum(dim=(-2, -1))
    )
    dice_per_sample_channel = 1.0 - (
        2.0 * intersection + dice_smooth
    ) / (
        denominator + dice_smooth
    )

    channel_has_positive = (
        target_grid.amax(dim=(-2, -1)) > 1e-6
    )
    valid_count = valid_mask.float()

    loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}
    for channel_index, channel_key in enumerate(
        FUTURE_INTERACTION_CHANNEL_KEYS
    ):
        loss_dict[
            f"future_interaction_{channel_key}_bce_loss"
        ] = (
            bce_per_sample_channel[:, channel_index]
            * valid_count,
            valid_count,
        )

        dice_count = (
            valid_mask
            & channel_has_positive[:, channel_index]
        ).float()
        loss_dict[
            f"future_interaction_{channel_key}_dice_loss"
        ] = (
            dice_per_sample_channel[:, channel_index]
            * dice_count,
            dice_count,
        )

    return loss_dict
