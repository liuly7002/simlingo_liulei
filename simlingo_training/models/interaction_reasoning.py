# -*- coding: utf-8 -*-

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


INTERACTION_TOKEN_KEYS: Tuple[str, ...] = (
    "route",
    "ego_future",
    "primary_actor",
    "secondary_actor",
)


class _SemanticTokenPool(nn.Module):
    """将一组同语义Driving query汇聚为一个语义token。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] <= 0:
            raise ValueError(
                "Semantic token pooling expects [B,N,D] with N>0, "
                f"but received {tuple(features.shape)}."
            )

        normalized = self.norm(features)
        weights = torch.softmax(
            self.score(normalized).squeeze(-1).float(),
            dim=1,
        ).to(dtype=normalized.dtype)
        return (normalized * weights.unsqueeze(-1)).sum(dim=1)


class UnifiedInteractionReasoner(nn.Module):
    """
    从语言、六视角视觉token和Driving query中构造统一决策交互表示。

    四个token固定对应：
        0: route
        1: ego future
        2: primary actor
        3: secondary actor

    主要输出：
        interaction_tokens: [B,4,D]
        enhanced_driving_features: [B,30,D]
        spatial_attention: [B,4,6,64]
    """

    def __init__(
        self,
        hidden_size: int,
        num_route_queries: int = 20,
        num_ego_queries: int = 10,
        num_cameras: int = 6,
        tokens_per_camera: int = 64,
        attention_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}."
            )
        if num_route_queries <= 0 or num_ego_queries <= 0:
            raise ValueError("Driving query counts must be positive.")
        if num_cameras <= 0 or tokens_per_camera <= 0:
            raise ValueError("Visual token layout must be positive.")

        self.hidden_size = int(hidden_size)
        self.num_route_queries = int(num_route_queries)
        self.num_ego_queries = int(num_ego_queries)
        self.num_driving_queries = (
            self.num_route_queries + self.num_ego_queries
        )
        self.num_cameras = int(num_cameras)
        self.tokens_per_camera = int(tokens_per_camera)
        self.num_visual_tokens = (
            self.num_cameras * self.tokens_per_camera
        )
        self.attention_dim = int(attention_dim)

        self.route_pool = _SemanticTokenPool(hidden_size)
        self.ego_pool = _SemanticTokenPool(hidden_size)

        self.token_type_embeddings = nn.Parameter(
            0.02 * torch.randn(1, len(INTERACTION_TOKEN_KEYS), hidden_size)
        )
        self.primary_actor_seed = nn.Parameter(
            0.02 * torch.randn(1, hidden_size)
        )
        self.secondary_actor_seed = nn.Parameter(
            0.02 * torch.randn(1, hidden_size)
        )

        self.language_norm = nn.LayerNorm(hidden_size)
        self.language_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.navigation_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

        self.spatial_query = nn.Linear(
            hidden_size,
            self.attention_dim,
            bias=False,
        )
        self.spatial_key = nn.Linear(
            hidden_size,
            self.attention_dim,
            bias=False,
        )
        self.spatial_value = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        self.spatial_output = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        self.visual_context_norm = nn.LayerNorm(hidden_size)

        self.interaction_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.interaction_self_norm = nn.LayerNorm(hidden_size)
        self.interaction_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.interaction_ffn_norm = nn.LayerNorm(hidden_size)

        self.action_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.action_gate = nn.Parameter(torch.tensor(0.1))
        self.action_norm = nn.LayerNorm(hidden_size)

        nn.init.xavier_uniform_(self.spatial_query.weight)
        nn.init.xavier_uniform_(self.spatial_key.weight)
        nn.init.xavier_uniform_(self.spatial_value.weight)
        nn.init.xavier_uniform_(self.spatial_output.weight)

    @staticmethod
    def _masked_mean(
        features: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if features.ndim != 3:
            raise ValueError(
                "Language features must have shape [B,L,D], "
                f"but received {tuple(features.shape)}."
            )
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                "Language context mask must match [B,L]: "
                f"{tuple(valid_mask.shape)} vs "
                f"{tuple(features.shape[:2])}."
            )

        valid = valid_mask.to(
            device=features.device,
            dtype=features.dtype,
        )
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (
            features * valid.unsqueeze(-1)
        ).sum(dim=1) / denominator

    def forward(
        self,
        interaction_query_features: Tensor,
        action_features: Tensor,
        visual_features: Tensor,
        language_features: Tensor,
        language_context_mask: Tensor,
        navigation_context: Tensor,
    ) -> Dict[str, Tensor]:
        if interaction_query_features.ndim != 3:
            raise ValueError(
                "Interaction query features must have shape [B,N,D], "
                f"but received {tuple(interaction_query_features.shape)}."
            )
        if action_features.shape != interaction_query_features.shape:
            raise ValueError(
                "Raw interaction queries and contextualized action features "
                "must have the same [B,N,D] shape."
            )
        if interaction_query_features.shape[1] != self.num_driving_queries:
            raise ValueError(
                "Unexpected number of Driving queries: expected "
                f"{self.num_driving_queries}, received "
                f"{interaction_query_features.shape[1]}."
            )
        if visual_features.ndim != 3:
            raise ValueError(
                "Visual features must have shape [B,V,D], "
                f"but received {tuple(visual_features.shape)}."
            )
        if visual_features.shape[1] != self.num_visual_tokens:
            raise ValueError(
                "Unexpected number of visual tokens: expected "
                f"{self.num_visual_tokens}, received "
                f"{visual_features.shape[1]}."
            )
        if navigation_context.shape != (
            interaction_query_features.shape[0],
            self.hidden_size,
        ):
            raise ValueError(
                "Navigation context must have shape [B,D], but received "
                f"{tuple(navigation_context.shape)}."
            )
        if (
            interaction_query_features.shape[0] != visual_features.shape[0]
            or interaction_query_features.shape[0] != language_features.shape[0]
        ):
            raise ValueError(
                "Language, visual and Driving batch sizes must match."
            )
        if (
            interaction_query_features.shape[-1] != self.hidden_size
            or action_features.shape[-1] != self.hidden_size
            or visual_features.shape[-1] != self.hidden_size
            or language_features.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                "All interaction inputs must use the configured hidden size."
            )

        route_features = interaction_query_features[
            :,
            : self.num_route_queries,
        ]
        ego_features = interaction_query_features[
            :,
            self.num_route_queries :,
        ]

        route_token = self.route_pool(route_features)
        ego_token = self.ego_pool(ego_features)

        language_summary = self._masked_mean(
            self.language_norm(language_features),
            language_context_mask,
        )
        language_context = self.language_projection(language_summary)
        navigation_context = self.navigation_projection(
            navigation_context
        )

        route_token = (
            route_token + language_context + navigation_context
        )
        ego_token = (
            ego_token + language_context + navigation_context
        )

        batch_size = int(interaction_query_features.shape[0])
        primary_seed = self.primary_actor_seed.expand(
            batch_size,
            -1,
        )
        secondary_seed = self.secondary_actor_seed.expand(
            batch_size,
            -1,
        )

        initial_tokens = torch.stack(
            (
                route_token,
                ego_token,
                primary_seed + route_token + ego_token,
                secondary_seed + route_token + ego_token,
            ),
            dim=1,
        )
        initial_tokens = (
            initial_tokens
            + self.token_type_embeddings.to(
                dtype=initial_tokens.dtype
            )
        )

        spatial_query = self.spatial_query(initial_tokens)
        spatial_key = self.spatial_key(visual_features)
        spatial_scores = torch.einsum(
            "btd,bvd->btv",
            spatial_query,
            spatial_key,
        ) * (self.attention_dim ** -0.5)
        spatial_attention = torch.softmax(
            spatial_scores.float(),
            dim=-1,
        ).to(dtype=visual_features.dtype)

        visual_value = self.spatial_value(visual_features)
        visual_context = torch.einsum(
            "btv,bvd->btd",
            spatial_attention,
            visual_value,
        )
        interaction_tokens = self.visual_context_norm(
            initial_tokens + self.spatial_output(visual_context)
        )

        self_context, _ = self.interaction_self_attention(
            interaction_tokens,
            interaction_tokens,
            interaction_tokens,
            need_weights=False,
        )
        interaction_tokens = self.interaction_self_norm(
            interaction_tokens + self_context
        )
        interaction_tokens = self.interaction_ffn_norm(
            interaction_tokens
            + self.interaction_ffn(interaction_tokens)
        )

        action_context, action_token_weights = (
            self.action_cross_attention(
                action_features,
                interaction_tokens,
                interaction_tokens,
                need_weights=True,
                average_attn_weights=True,
            )
        )
        action_gate = torch.tanh(self.action_gate).to(
            dtype=action_features.dtype
        )
        enhanced_driving_features = self.action_norm(
            action_features + action_gate * action_context
        )

        spatial_attention = spatial_attention.reshape(
            batch_size,
            len(INTERACTION_TOKEN_KEYS),
            self.num_cameras,
            self.tokens_per_camera,
        )

        primary_attention = spatial_attention[:, 2]
        secondary_attention = spatial_attention[:, 3]

        primary_probability = primary_attention.float().clamp_min(1e-8)
        primary_entropy = -(
            primary_probability * primary_probability.log()
        ).sum(dim=(-2, -1)) / math.log(float(self.num_visual_tokens))

        return {
            "interaction_tokens": interaction_tokens,
            "enhanced_driving_features": enhanced_driving_features,
            "spatial_attention": spatial_attention,
            "primary_spatial_attention": primary_attention,
            "secondary_spatial_attention": secondary_attention,
            "primary_camera_attention": primary_attention.sum(dim=-1),
            "secondary_camera_attention": secondary_attention.sum(dim=-1),
            "primary_spatial_attention_entropy": primary_entropy,
            "driving_to_interaction_attention": action_token_weights,
        }


def compute_participant_spatial_attention_losses(
    primary_attention: Tensor,
    target_attention: Tensor,
    valid_mask: Tensor,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """计算主要关键参与者的六视角视觉token级软标签交叉熵。"""

    if primary_attention.ndim != 3:
        raise ValueError(
            "Primary participant attention must have shape [B,6,64], "
            f"but received {tuple(primary_attention.shape)}."
        )
    if target_attention.shape != primary_attention.shape:
        raise ValueError(
            "Participant spatial target shape does not match prediction: "
            f"{tuple(target_attention.shape)} vs "
            f"{tuple(primary_attention.shape)}."
        )
    if valid_mask.shape != (primary_attention.shape[0],):
        raise ValueError(
            "Participant spatial valid mask must have shape [B], "
            f"but received {tuple(valid_mask.shape)}."
        )

    prediction = primary_attention.float().clamp_min(1e-8)
    target = target_attention.to(
        device=prediction.device,
        dtype=torch.float32,
    )
    valid = valid_mask.to(
        device=prediction.device,
        dtype=torch.bool,
    )

    if not bool(torch.isfinite(target).all().item()):
        raise ValueError(
            "Participant spatial target contains non-finite values."
        )
    if bool((target < 0.0).any().item()):
        raise ValueError(
            "Participant spatial target contains negative values."
        )

    target_sum = target.sum(dim=(-2, -1))
    invalid_valid_target = valid & (target_sum <= 0.0)
    if bool(invalid_valid_target.any().item()):
        raise ValueError(
            "A valid participant spatial target has a non-positive sum."
        )

    normalized_target = torch.where(
        valid.view(-1, 1, 1),
        target / target_sum.clamp_min(1e-8).view(-1, 1, 1),
        torch.zeros_like(target),
    )

    per_sample_loss = -(
        normalized_target * prediction.log()
    ).sum(dim=(-2, -1))
    loss_count = valid.float()

    return {
        "participant_spatial_attention_loss": (
            per_sample_loss * loss_count,
            loss_count,
        )
    }
