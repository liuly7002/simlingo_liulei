# -*- coding: utf-8 -*-

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


INTERACTION_TOKEN_KEYS: Tuple[str, ...] = (
    "route",
    "ego_future",
    "primary_actor",
    "secondary_actor",
)

# Q1: primary actor; Q2: route constraint; Q3/Q4: ego future response.
LANGUAGE_QUESTION_TO_TOKEN_INDEX: Tuple[int, ...] = (2, 0, 1, 1)


class _SemanticTokenPool(nn.Module):
    """将同一语义分支的一组Driving query汇聚为一个交互token。"""

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


class PreLanguageInteractionReasoner(nn.Module):
    """
    在语言模型前构造四个统一决策交互token。

    这四个token被放置在语言序列最前方，因此：
      1. Q1-Q4语言生成能够直接读取交互证据；
      2. 后置Driving query同时读取交互token和生成/监督语言；
      3. 四通道未来世界预测继续读取同一组token。
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
            0.02 * torch.randn(
                1,
                len(INTERACTION_TOKEN_KEYS),
                hidden_size,
            )
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

        # 语言、动作与交互token的一致性投影仅服务第二阶段损失。
        self.language_alignment_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.action_alignment_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.token_alignment_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

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
                "Features must have shape [B,L,D], but received "
                f"{tuple(features.shape)}."
            )
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                "Mask must match [B,L]: "
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
        if interaction_query_features.shape[1] != self.num_driving_queries:
            raise ValueError(
                "Unexpected number of Driving queries: expected "
                f"{self.num_driving_queries}, received "
                f"{interaction_query_features.shape[1]}."
            )
        if visual_features.ndim != 3:
            raise ValueError(
                "Visual features must have shape [B,V,D], but received "
                f"{tuple(visual_features.shape)}."
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
            interaction_query_features.shape[0]
            != visual_features.shape[0]
            or interaction_query_features.shape[0]
            != language_features.shape[0]
        ):
            raise ValueError(
                "Language, visual and Driving batch sizes must match."
            )
        if (
            interaction_query_features.shape[-1] != self.hidden_size
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
            "spatial_attention": spatial_attention,
            "primary_spatial_attention": primary_attention,
            "secondary_spatial_attention": secondary_attention,
            "primary_camera_attention": primary_attention.sum(dim=-1),
            "secondary_camera_attention": secondary_attention.sum(dim=-1),
            "primary_spatial_attention_entropy": primary_entropy,
        }

    def compute_language_alignment_loss(
        self,
        language_features: Tensor,
        contextual_interaction_tokens: Tensor,
        question_span_mask: Tensor,
        valid_mask: Tensor,
        temperature: float,
    ) -> Tuple[Tensor, Tensor]:
        """将Q1-Q4答案表示分别对齐到固定语义交互token。"""

        if question_span_mask.shape[:2] != (
            language_features.shape[0],
            len(LANGUAGE_QUESTION_TO_TOKEN_INDEX),
        ):
            raise ValueError(
                "Question span mask must have shape [B,4,L]."
            )
        if question_span_mask.shape[-1] != language_features.shape[1]:
            raise ValueError(
                "Question span mask length must match language features."
            )

        span = question_span_mask.to(
            device=language_features.device,
            dtype=language_features.dtype,
        )
        denominator = span.sum(dim=-1, keepdim=True).clamp_min(1.0)
        question_features = torch.einsum(
            "bql,bld->bqd",
            span,
            language_features,
        ) / denominator

        question_features = F.normalize(
            self.language_alignment_projection(question_features).float(),
            dim=-1,
        )
        token_features = F.normalize(
            self.token_alignment_projection(
                contextual_interaction_tokens
            ).float(),
            dim=-1,
        )

        logits = torch.einsum(
            "bqd,btd->bqt",
            question_features,
            token_features,
        ) / max(float(temperature), 1e-4)
        targets = torch.as_tensor(
            LANGUAGE_QUESTION_TO_TOKEN_INDEX,
            device=logits.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(logits.shape[0], -1)
        per_question_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(logits.shape[0], logits.shape[1])
        per_sample_loss = per_question_loss.mean(dim=1)
        count = valid_mask.to(
            device=per_sample_loss.device,
            dtype=per_sample_loss.dtype,
        )
        return per_sample_loss * count, count

    def compute_action_alignment_loss(
        self,
        driving_features: Tensor,
        contextual_interaction_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """对齐route/ego动作query与对应的上下文化交互token。"""

        if driving_features.shape[1] != self.num_driving_queries:
            raise ValueError(
                "Driving feature count does not match configured queries."
            )

        route_action = driving_features[
            :,
            : self.num_route_queries,
        ].mean(dim=1)
        ego_action = driving_features[
            :,
            self.num_route_queries :,
        ].mean(dim=1)
        action_features = torch.stack(
            (route_action, ego_action),
            dim=1,
        )

        action_features = F.normalize(
            self.action_alignment_projection(action_features).float(),
            dim=-1,
        )
        target_tokens = F.normalize(
            self.token_alignment_projection(
                contextual_interaction_tokens[:, :2]
            ).float(),
            dim=-1,
        )
        cosine = (action_features * target_tokens).sum(dim=-1)
        per_sample_loss = (1.0 - cosine).mean(dim=1)
        count = torch.ones_like(per_sample_loss)
        return per_sample_loss, count


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


def compute_actor_disentanglement_losses(
    primary_attention: Tensor,
    secondary_attention: Tensor,
    contextual_interaction_tokens: Tensor,
    secondary_exists: Tensor,
    attention_overlap_margin: float,
    token_cosine_margin: float,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """
    当C4中确实存在次要参与者时，约束主要/次要参与者证据不塌缩。

    该约束直接使用现有C4标签判断次要参与者是否存在，
    不要求新增或重新生成次要参与者空间标签。
    """

    if primary_attention.shape != secondary_attention.shape:
        raise ValueError(
            "Primary and secondary spatial attention shapes must match."
        )
    if secondary_exists.shape != (primary_attention.shape[0],):
        raise ValueError("secondary_exists must have shape [B].")

    primary_probability = primary_attention.float().clamp_min(1e-8)
    secondary_probability = secondary_attention.float().clamp_min(1e-8)

    # Bhattacharyya overlap: 0表示完全分离，1表示完全相同。
    attention_overlap = torch.sqrt(
        primary_probability * secondary_probability
    ).sum(dim=(-2, -1))
    attention_loss = F.relu(
        attention_overlap - float(attention_overlap_margin)
    )

    primary_token = F.normalize(
        contextual_interaction_tokens[:, 2].float(),
        dim=-1,
    )
    secondary_token = F.normalize(
        contextual_interaction_tokens[:, 3].float(),
        dim=-1,
    )
    token_cosine = (primary_token * secondary_token).sum(dim=-1)
    token_loss = F.relu(
        token_cosine - float(token_cosine_margin)
    )

    count = secondary_exists.to(
        device=attention_loss.device,
        dtype=attention_loss.dtype,
    )
    return {
        "actor_attention_separation_loss": (
            attention_loss * count,
            count,
        ),
        "actor_token_separation_loss": (
            token_loss * count,
            count,
        ),
    }


def validate_question_token_mapping(mapping: Sequence[int]) -> None:
    if tuple(int(value) for value in mapping) != (
        LANGUAGE_QUESTION_TO_TOKEN_INDEX
    ):
        raise ValueError(
            "The Q1-Q4 interaction-token mapping must remain "
            f"{LANGUAGE_QUESTION_TO_TOKEN_INDEX}."
        )
