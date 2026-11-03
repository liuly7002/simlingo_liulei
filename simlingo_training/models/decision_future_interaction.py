# -*- coding: utf-8 -*-


"""
这是三语义通道与未来逐帧占据的结构化世界解码器
"""


from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# 三语义通道名称、顺序与统一交互token严格一一对应。
FUTURE_INTERACTION_CHANNEL_KEYS = (
    "c0_route",            # 参考路径(自车footprint占用)
    "c1_ego_future",       # 自车未来的 waypoints (自车footprint占用)
    "c2_primary_actor",    # 主要actor未来的waypoints (主要actor footprint占用)
)


class _SingleInteractionChannelDecoder(nn.Module):
    """
    由一个具有明确语义的交互token解码一个结构化未来通道
    """

    def __init__(self, hidden_size: int, output_size: int, output_channels: int = 1,): 
        # hidden_size [BS,D] D对应InternVL2-1B语言模型的hidden size

        # 调用父类__init__()函数
        super().__init__()

        # 最终输出空间图的尺寸 128
        self.output_size = int(output_size)
        self.output_channels = int(output_channels)
        
        # 定义一个初始空间特征图 
        self.seed_channels = 64  # 通道数
        self.seed_size = 8       # 空间大小8x8

        
        # 一维交互token [B,D]  -> 一维空间种子向量 [BS,64x8x8]
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

        # 转置卷积解码器  通过四层转置卷积逐步放大空间尺寸 由8x8->128x128  同时逐步降低通道数 64->1
        self.decoder = nn.Sequential(
            # 第一层转置卷积
            nn.ConvTranspose2d(64,64,kernel_size=4,stride=2,padding=1,),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 第二层转置卷积
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1,),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # 第三层转置卷积
            nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1,),
            nn.GroupNorm(4, 16),
            nn.SiLU(),

            # 第四层转置卷积
            nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1,),
            nn.GroupNorm(4, 8),
            nn.SiLU(),

            # 最后1x1 将每个像素位置上的 8 维特征压缩为 1 个数值
            nn.Conv2d(8, self.output_channels, kernel_size=1,),
        )

    def forward(self, interaction_token: Tensor) -> Tensor:
        
        # 安全性检查
        if interaction_token.ndim != 2:
            raise ValueError(
                "One interaction token must have shape [B,D], "
                f"but received {tuple(interaction_token.shape)}."
            )

        # 生成一维空间种子 形状[B,D] → [B,64x8x8]
        seed = self.seed_projection(interaction_token)
        
        # 恢复为[BS,64,8,8]
        seed = seed.reshape(
            interaction_token.shape[0],  # BS
            self.seed_channels,          # 64
            self.seed_size,              # 8
            self.seed_size,              # 8
        )

        # 解码  [BS,1,128,128] 1表示只预测单个通道的
        logits = self.decoder(seed)

        # 输出形状安全性检查
        expected_shape = (
            interaction_token.shape[0],
            self.output_channels,
            self.output_size,
            self.output_size,
        )
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected single-channel interaction prediction shape: "
                f"expected {expected_shape}, received "
                f"{tuple(logits.shape)}."
            )

        return logits  # [BS,1,128,128]


class FutureInteractionDecoder(nn.Module):
    """预测三种语义占据与 C1/C2 的逐帧未来占据。

    输出字典：
      grid_logits: [B,3,H,W]，C1/C2 是逐帧概率的时间均值
      temporal_logits: [B,2,T,H,W]，依次为未来 Ego/关键 Actor
    """

    def __init__(
        self,
        hidden_size: int,
        output_channels: int = 3,
        output_size: int = 128,
        future_steps: int = 10,
    ):
        super().__init__()
        # 原DrivingModel构造函数仍传入legacy参数4；该参数只是兼容入口，
        # 实际解码器始终只构造C0/C1/C2三个语义通道，不生成任何第四通道。
        if output_channels not in (3, 4):
            raise ValueError("Future interaction requires three semantic channels.")
        if output_size != 128:
            raise ValueError("The current decoder supports output_size=128.")
        if future_steps <= 0:
            raise ValueError("future_steps must be positive.")
        self.output_channels = len(FUTURE_INTERACTION_CHANNEL_KEYS)
        self.output_size = int(output_size)
        self.future_steps = int(future_steps)
        # C0 预测静态路径占据，C1/C2 分别同时预测 T 个时间片。
        self.channel_decoders = nn.ModuleList(
            [
                _SingleInteractionChannelDecoder(
                    hidden_size, output_size,
                    output_channels=(1 if i == 0 else self.future_steps),
                )
                for i in range(self.output_channels)
            ]
        )

    def forward(self, interaction_tokens: Tensor) -> Dict[str, Tensor]:
        if interaction_tokens.ndim != 3 or interaction_tokens.shape[1] != 3:
            raise ValueError(
                "Expected three interaction tokens [B,3,D], got "
                f"{tuple(interaction_tokens.shape)}."
            )
        route = self.channel_decoders[0](interaction_tokens[:, 0])
        ego_steps = self.channel_decoders[1](interaction_tokens[:, 1])
        actor_steps = self.channel_decoders[2](interaction_tokens[:, 2])

        temporal_logits = torch.stack((ego_steps, actor_steps), dim=1)
        # 与标签生成器完全同义：十个 future footprint 的等权时间均值。
        temporal_mean = torch.sigmoid(temporal_logits.float()).mean(dim=2)
        # 交互预测的汇总值保持在 logit 域，供 BCEWithLogitsLoss 使用。
        summary_logits = torch.logit(temporal_mean.clamp(1e-5, 1.0 - 1e-5))
        grid_logits = torch.cat((route, summary_logits), dim=1)
        return {
            "grid_logits": grid_logits,
            "temporal_logits": temporal_logits,
        }


def _resize_future_interaction_target(target_grid: Tensor,output_size: Tuple[int, int],) -> Tensor:
    """
    C0/C1采用面积平均；C2采用占用保持式最大池化。

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


def _resize_temporal_targets(
    target: Tensor,
    output_size: Tuple[int, int],
) -> Tensor:
    """对 C1 时间序列使用面积平均，对稀疏 C2 时间序列保持占据。"""
    b, channels, steps, h, w = target.shape
    if channels != 2:
        raise ValueError("Temporal target must have 2 semantic channels.")
    if (h, w) == tuple(output_size):
        return target
    dense = target[:, 0].reshape(b * steps, 1, h, w)
    sparse = target[:, 1].reshape(b * steps, 1, h, w)
    dense = F.interpolate(dense, size=output_size, mode="area")
    if h >= output_size[0] and w >= output_size[1]:
        sparse = F.adaptive_max_pool2d(sparse, output_size)
    else:
        sparse = F.interpolate(sparse, size=output_size, mode="nearest")
    return torch.stack(
        (dense.reshape(b, steps, *output_size),
         sparse.reshape(b, steps, *output_size)),
        dim=1,
    )


def compute_future_interaction_losses(
    prediction_logits: Tensor,
    target_grid: Tensor,
    valid_mask: Tensor,
    positive_weights: Sequence[float],
    dice_smooth: float = 1e-6,
    temporal_logits: Tensor = None,
    temporal_target: Tensor = None,
    critical_actor_status: Tensor = None,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """三语义通道及 C1/C2 逐帧损失；三状态仅控制 C2 的有效监督。

    状态 1：关键 Actor 正样本；状态 0：可靠的全零负样本；
    状态 -1：不监督 C2（C0/C1 仍可使用原有 world valid）。
    """
    if prediction_logits.ndim != 4 or prediction_logits.shape[1] != 3:
        raise ValueError("Summary logits must have shape [B,3,H,W].")
    if target_grid.ndim != 4 or target_grid.shape[1] != 3:
        raise ValueError("Summary target must have shape [B,3,H,W].")
    if prediction_logits.shape[0] != target_grid.shape[0]:
        raise ValueError("Summary batch sizes differ.")
    if valid_mask.shape != (prediction_logits.shape[0],):
        raise ValueError("valid_mask must have shape [B].")
    if len(positive_weights) != 3:
        raise ValueError("positive_weights must contain exactly 3 values.")

    device = prediction_logits.device
    gt = target_grid.to(device=device, dtype=torch.float32)
    if not torch.isfinite(gt).all() or ((gt < 0) | (gt > 1)).any():
        raise ValueError("Invalid summary occupancy target.")
    gt = _resize_future_interaction_target(gt, prediction_logits.shape[-2:])
    # C1/C2 summary targets must use the *same* resized per-frame masks as
    # their temporal losses. Pooling sparse C2 and averaging time do not commute.
    # This avoids contradictory summary/temporal targets at 128x128.
    temporal_gt = None
    if temporal_logits is not None and temporal_target is not None:
        if temporal_logits.ndim != 5 or temporal_logits.shape[1] != 2:
            raise ValueError("Temporal logits must be [B,2,T,H,W].")
        if temporal_target.ndim != 5 or temporal_target.shape[:3] != temporal_logits.shape[:3]:
            raise ValueError("Temporal target shape must match [B,2,T,H,W].")
        temporal_gt = temporal_target.to(device=device, dtype=torch.float32)
        if not torch.isfinite(temporal_gt).all() or ((temporal_gt < 0) | (temporal_gt > 1)).any():
            raise ValueError("Invalid temporal occupancy target.")
        temporal_gt = _resize_temporal_targets(temporal_gt, temporal_logits.shape[-2:])
        gt = torch.cat((gt[:, :1], temporal_gt.mean(dim=2)), dim=1)
    base_valid = valid_mask.to(device=device, dtype=torch.bool)
    if critical_actor_status is None:
        actor_valid = base_valid
    else:
        critical_actor_status = critical_actor_status.to(device=device)
        if critical_actor_status.shape != base_valid.shape:
            raise ValueError("critical_actor_status must have shape [B].")
        if not ((critical_actor_status == 1) | (critical_actor_status == 0) |
                (critical_actor_status == -1)).all():
            raise ValueError("critical_actor_status must be -1, 0, or 1.")
        actor_valid = base_valid & (critical_actor_status != -1)
        known_absent = base_valid & (critical_actor_status == 0)
        if known_absent.any() and (gt[known_absent, 2].abs().amax() > 1e-5):
            raise ValueError("Status 0 requires empty actor summary occupancy.")

    logits = prediction_logits.float()
    weights = torch.tensor(positive_weights, device=device, dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(
        logits, gt, reduction="none", pos_weight=weights.view(1, 3, 1, 1)
    ).mean(dim=(-2, -1))
    prob = torch.sigmoid(logits)
    dice = 1.0 - (2 * (prob * gt).sum(dim=(-2, -1)) + dice_smooth) / (
        prob.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) + dice_smooth
    )

    result: Dict[str, Tuple[Tensor, Tensor]] = {}
    for index, key in enumerate(FUTURE_INTERACTION_CHANNEL_KEYS):
        valid = actor_valid if index == 2 else base_valid
        weight = valid.float()
        result[f"future_interaction_{key}_bce_loss"] = (bce[:, index] * weight, weight)
        pos = (gt[:, index].amax(dim=(-2, -1)) > 1e-6) & valid
        pos_weight = pos.float()
        result[f"future_interaction_{key}_dice_loss"] = (
            dice[:, index] * pos_weight, pos_weight,
        )

    if temporal_logits is None and temporal_target is None:
        return result
    if temporal_logits is None or temporal_target is None:
        raise ValueError("Temporal logits and targets must be provided together.")
    assert temporal_gt is not None  # Validated and resized above.
    if critical_actor_status is not None and (base_valid & (critical_actor_status == 0)).any():
        if temporal_gt[base_valid & (critical_actor_status == 0), 1].abs().amax() > 1e-5:
            raise ValueError("Status 0 requires empty actor temporal occupancy.")

    temporal_float = temporal_logits.float()
    temporal_bce = F.binary_cross_entropy_with_logits(
        temporal_float, temporal_gt, reduction="none",
        pos_weight=weights[1:].view(1, 2, 1, 1, 1),
    ).mean(dim=(-3, -2, -1))
    temporal_prob = torch.sigmoid(temporal_float)
    temporal_dice = 1.0 - (
        2 * (temporal_prob * temporal_gt).sum(dim=(-3, -2, -1)) + dice_smooth
    ) / (
        temporal_prob.sum(dim=(-3, -2, -1)) +
        temporal_gt.sum(dim=(-3, -2, -1)) + dice_smooth
    )
    for index, key in enumerate(FUTURE_INTERACTION_CHANNEL_KEYS[1:]):
        valid = base_valid if index == 0 else actor_valid
        weight = valid.float()
        result[f"future_temporal_{key}_bce_loss"] = (
            temporal_bce[:, index] * weight, weight,
        )
        positive = (temporal_gt[:, index].amax(dim=(-3, -2, -1)) > 1e-6) & valid
        pos_weight = positive.float()
        result[f"future_temporal_{key}_dice_loss"] = (
            temporal_dice[:, index] * pos_weight, pos_weight,
        )
    return result
