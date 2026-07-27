# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


#修改20260726：五通道名称及顺序必须与标签生成脚本一致。
FUTURE_INTERACTION_CHANNEL_KEYS = (
    "c0_route",
    "c1_ego_future",
    "c2_primary_actor",
    "c3_interaction",
    "c4_secondary_actor",
)


class FutureInteractionDecoder(nn.Module):
    """
    使用已经完成多模态融合的Driving query特征，
    解码五通道结构化未来世界预测。

    输入：
        query_features: [B, N, D]
        当前N=30，包括20个route query和10个speed waypoint query。

    输出：
        prediction_logits: [B, 5, 128, 128]
    """

    def __init__(
        self,
        hidden_size: int,
        output_channels: int = 5,
        output_size: int = 128,
    ):
        super().__init__()

        if output_channels != 5:
            raise ValueError(
                "FutureInteractionDecoder currently requires "
                f"5 output channels, but received {output_channels}."
            )

        if output_size != 128:
            raise ValueError(
                "The first future-interaction decoder version "
                "supports only output_size=128, but received "
                f"{output_size}."
            )

        self.output_channels = output_channels
        self.output_size = output_size
        self.seed_channels = 128
        self.seed_size = 8

        # 对30个Driving query进行可学习加权池化。
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_score = nn.Linear(
            hidden_size,
            1,
            bias=False,
        )

        # 将全局多模态特征展开为8×8二维种子特征。
        self.seed_projection = nn.Sequential(
            nn.Linear(
                hidden_size,
                self.seed_channels
                * self.seed_size
                * self.seed_size,
            ),
            nn.SiLU(),
        )

        # 8→16→32→64→128。
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            nn.ConvTranspose2d(
                128,
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

            # 直接输出logits，训练前不要在这里做sigmoid。
            nn.Conv2d(
                16,
                output_channels,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        query_features: Tensor,
    ) -> Tensor:
        if query_features.ndim != 3:
            raise ValueError(
                "FutureInteractionDecoder expects query features "
                f"with shape [B,N,D], but received "
                f"{tuple(query_features.shape)}."
            )

        if query_features.shape[1] <= 0:
            raise ValueError(
                "FutureInteractionDecoder received no query tokens."
            )

        normalized_features = self.query_norm(
            query_features
        )

        query_weights = torch.softmax(
            self.query_score(
                normalized_features
            ).squeeze(-1),
            dim=1,
        )

        pooled_feature = (
            normalized_features
            * query_weights.unsqueeze(-1)
        ).sum(dim=1)

        seed_feature = self.seed_projection(
            pooled_feature
        )

        seed_feature = seed_feature.reshape(
            query_features.shape[0],
            self.seed_channels,
            self.seed_size,
            self.seed_size,
        )

        prediction_logits = self.decoder(
            seed_feature
        )

        expected_shape = (
            query_features.shape[0],
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


def compute_future_interaction_losses(
    prediction_logits: Tensor,
    target_grid: Tensor,
    valid_mask: Tensor,
    positive_weights: Sequence[float],
    dice_smooth: float = 1e-6,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """
    计算分通道的加权BCE和Dice损失。

    BCE：
        所有valid=True样本均参与，包括C2/C3/C4为全零的负样本。

    Dice：
        仅在该通道真实标签非零时计算，避免大量全零样本
        将稀疏通道的Dice监督完全淹没。
    """

    if prediction_logits.ndim != 4:
        raise ValueError(
            "Future interaction prediction must have shape "
            f"[B,5,H,W], but received "
            f"{tuple(prediction_logits.shape)}."
        )

    if target_grid.ndim != 4:
        raise ValueError(
            "Future interaction target must have shape "
            f"[B,5,H,W], but received "
            f"{tuple(target_grid.shape)}."
        )

    if (
        prediction_logits.shape[0]
        != target_grid.shape[0]
    ):
        raise ValueError(
            "Future interaction prediction and target batch "
            "sizes do not match."
        )

    if (
        prediction_logits.shape[1] != 5
        or target_grid.shape[1] != 5
    ):
        raise ValueError(
            "Future interaction prediction and target must "
            "both contain exactly 5 channels."
        )

    if valid_mask.shape != (
        prediction_logits.shape[0],
    ):
        raise ValueError(
            "future_interaction_valid must have shape [B], "
            f"but received {tuple(valid_mask.shape)}."
        )

    if len(positive_weights) != 5:
        raise ValueError(
            "future_interaction_positive_weights must contain "
            f"5 values, but received {len(positive_weights)}."
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

    if not bool(
        torch.isfinite(target_grid).all().item()
    ):
        raise ValueError(
            "Future interaction target contains non-finite values."
        )

    if bool(
        (
            (target_grid < 0.0)
            | (target_grid > 1.0)
        ).any().item()
    ):
        raise ValueError(
            "Future interaction target values must be within "
            "[0, 1]."
        )

    # 原始标签为256×256，在损失端采用面积平均降到128×128。
    # 不修改磁盘标签，也不修改dataloader中的原始标签。
    if target_grid.shape[-2:] != (
        prediction_logits.shape[-2:]
    ):
        target_grid = F.interpolate(
            target_grid,
            size=prediction_logits.shape[-2:],
            mode="area",
        )

    positive_weight_tensor = torch.as_tensor(
        positive_weights,
        device=logits_float.device,
        dtype=torch.float32,
    ).reshape(1, 5, 1, 1)

    bce_map = F.binary_cross_entropy_with_logits(
        logits_float,
        target_grid,
        reduction="none",
        pos_weight=positive_weight_tensor,
    )

    # [B,5]
    bce_per_sample_channel = bce_map.mean(
        dim=(-2, -1)
    )

    prediction_probability = torch.sigmoid(
        logits_float
    )

    intersection = (
        prediction_probability
        * target_grid
    ).sum(dim=(-2, -1))

    denominator = (
        prediction_probability.sum(
            dim=(-2, -1)
        )
        + target_grid.sum(
            dim=(-2, -1)
        )
    )

    # [B,5]
    dice_per_sample_channel = 1.0 - (
        2.0 * intersection + dice_smooth
    ) / (
        denominator + dice_smooth
    )

    # 该样本的该通道是否含正标签。
    channel_has_positive = (
        target_grid.amax(
            dim=(-2, -1)
        ) > 1e-6
    )

    valid_count = valid_mask.float()

    loss_dict: Dict[
        str,
        Tuple[Tensor, Tensor],
    ] = {}

    for channel_index, channel_key in enumerate(
        FUTURE_INTERACTION_CHANNEL_KEYS
    ):
        loss_dict[
            f"future_interaction_"
            f"{channel_key}_bce_loss"
        ] = (
            bce_per_sample_channel[
                :,
                channel_index,
            ] * valid_count,
            valid_count,
        )

        dice_count = (
            valid_mask
            & channel_has_positive[
                :,
                channel_index,
            ]
        ).float()

        loss_dict[
            f"future_interaction_"
            f"{channel_key}_dice_loss"
        ] = (
            dice_per_sample_channel[
                :,
                channel_index,
            ] * dice_count,
            dice_count,
        )

    return loss_dict