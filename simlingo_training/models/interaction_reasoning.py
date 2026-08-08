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
    """
    让网络自主学习哪些路径 query 更重要,哪些未来轨迹 query 更重要
    因此,不同场景下被赋予较高权重的 query 可以不同
    """

    def __init__(self, hidden_size: int):
        
        # 调用父类__init__()函数
        super().__init__()

        # 对每个query做LayerNorm 形状不变 [B,20,D] & [B,10,D]
        self.norm = nn.LayerNorm(hidden_size)

        # [B,20,D]->[B,20,1] & [B,10,D]->[B,10,1]
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:

        # 安全性检查
        if features.ndim != 3 or features.shape[1] <= 0:
            raise ValueError(
                "Semantic token pooling expects [B,N,D] with N>0, "
                f"but received {tuple(features.shape)}."
            )

        # [B,20,D] & [B,10,D]
        normalized = self.norm(features)
        
        # [B,20] & [B,10]
        weights = torch.softmax(self.score(normalized).squeeze(-1).float(),dim=1,).to(dtype=normalized.dtype)

        # [B,D] & [B,D]
        return (normalized * weights.unsqueeze(-1)).sum(dim=1)


class PreLanguageInteractionReasoner(nn.Module):
    """
    在正式进入语言 Transformer 之前,
    根据 30 个 Driving query、384 个视觉 token、语言上下文和导航信息,构造 4 个统一交互 token

    [4个交互token | 语言token | 20个route query | 10个speed_wps query]

    这四个token被放置在语言序列最前方,因此:
      1. Q1-Q4语言生成能够直接读取交互证据;
      2. 后置Driving query同时读取交互token和生成/监督语言;
      3. 四通道未来世界预测继续读取同一组token.
    """

    def __init__(
        self,
        hidden_size: int,              # D 语言模型的隐状态size
        num_route_queries: int = 20,
        num_ego_queries: int = 10,
        num_cameras: int = 6,
        tokens_per_camera: int = 64,
        attention_dim: int = 128,      # 四个交互 token 对视觉 token 执行空间注意力时,Query 和 Key 使用的维度
        num_heads: int = 8,
        dropout: float = 0.1,):

        super().__init__()

        # 参数合法性检查 
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
        self.num_route_queries = int(num_route_queries)  # 20, route点的数量
        self.num_ego_queries = int(num_ego_queries)      # 10, ego点的数量
        self.num_driving_queries = (self.num_route_queries + self.num_ego_queries)  # 30,driving query的数量
        self.num_cameras = int(num_cameras)              # 6,相机的数量
        self.tokens_per_camera = int(tokens_per_camera)  # 4x8x2=64,每个相机的token的数量
        self.num_visual_tokens = (self.num_cameras * self.tokens_per_camera) # 6x64=384,六个相机总的token数
        self.attention_dim = int(attention_dim)

        self.route_pool = _SemanticTokenPool(hidden_size)
        self.ego_pool = _SemanticTokenPool(hidden_size)

        # # [1,4,D] 可学习参数
        self.token_type_embeddings = nn.Parameter(0.02 * torch.randn(1,len(INTERACTION_TOKEN_KEYS),hidden_size,))
        # [1,D] 可学习参数
        self.primary_actor_seed = nn.Parameter(0.02 * torch.randn(1, hidden_size))
        # [1,D] 可学习参数
        self.secondary_actor_seed = nn.Parameter(0.02 * torch.randn(1, hidden_size))

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

        # 反事实语言变化必须与对象移除后的真实动作变化保持一致
        self.counterfactual_language_delta_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.counterfactual_action_effect_projection = nn.Sequential(
            nn.LayerNorm(self.num_ego_queries * 2),
            nn.Linear(
                self.num_ego_queries * 2,
                hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

        nn.init.xavier_uniform_(self.spatial_query.weight)
        nn.init.xavier_uniform_(self.spatial_key.weight)
        nn.init.xavier_uniform_(self.spatial_value.weight)
        nn.init.xavier_uniform_(self.spatial_output.weight)

    @staticmethod
    def _masked_mean(features: Tensor, valid_mask: Tensor,) -> Tensor:
        """
        language_summary = 有效语言位置特征之和 ÷ 有效位置数量
        """
        
        # 安全性检查
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

        valid = valid_mask.to(device=features.device,dtype=features.dtype,)

        # 有效位置数量
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (features * valid.unsqueeze(-1)).sum(dim=1) / denominator

    def forward(
        self,
        interaction_query_features: Tensor,                # [B,30,D]   这是语言Transformer之前的30个driving query
        visual_features: Tensor,                           # [B,384,D]  来自语言序列中已经替换完成的<IMG_CONTEXT>也就是图像embedding
        language_features: Tensor,                         # [B,L,D]    这是当前完整语言序列在进入正式语言Transformer前的embedding,其中<IMG_CONTEXT>和<TARGET_POINT>已经替换为特征embedding
        language_context_mask: Tensor,                     # [B,L]      它决定哪些语言位置可以用于构造交互token的语言上下文
        navigation_context: Tensor,) -> Dict[str, Tensor]: # [B,D]      这是两个导航点embedding的平均表示

        # 安全性检查
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
        if navigation_context.shape != (interaction_query_features.shape[0],self.hidden_size,):
            raise ValueError(
                "Navigation context must have shape [B,D], but received "
                f"{tuple(navigation_context.shape)}."
            )
        if (interaction_query_features.shape[0] != visual_features.shape[0] or interaction_query_features.shape[0] != language_features.shape[0]):
            raise ValueError(
                "Language, visual and Driving batch sizes must match."
            )
        if (interaction_query_features.shape[-1] != self.hidden_size or visual_features.shape[-1] != self.hidden_size or language_features.shape[-1] != self.hidden_size):
            raise ValueError(
                "All interaction inputs must use the configured hidden size."
            )


        ################################################### route_token & ego_token ###################################################

        # 拆分 route query 和 ego query
        route_features = interaction_query_features[:, :self.num_route_queries,]   # [B,20,D]
        ego_features = interaction_query_features[:, self.num_route_queries:,]     # [B,10,D]

        # 将20个route query 汇聚成一个 route token
        route_token = self.route_pool(route_features)  # [B,D]
        
        # 将10个 ego query 汇聚成一个 ego token
        ego_token = self.ego_pool(ego_features)        # [B,D]

        # 语言有效位置特征平均值 [B,D]
        language_summary = self._masked_mean(
            self.language_norm(language_features),  # 语言特征归一化 [B,L,D]
            language_context_mask,                  # [B,L]根据mask选择位置,只有=True的位置才能参与平均 不能参与的位置包括padding、图像token、答案token
        )
        # 投影语言上下文 [B,D] MLP
        language_context = self.language_projection(language_summary)

        # 投影导航上下文 [B，D] MLP
        navigation_context = self.navigation_projection(navigation_context)

        # route token 加入语言、导航条件 [B,D]  
        route_token = (route_token + language_context + navigation_context)
        
        # ego token 加入语言、导航条件  [B,D]
        ego_token = (ego_token + language_context + navigation_context)
        
        """
        到这里主要是想要实现,route和ego并不是仅仅由原来各自的 driving query 初始化,他们从一开始就能够受到语言问题和导航意图的共同调节
        """


        ################################################### 扩展 primary actor seed 和 secndary actor seed ###################################################

        # batch size
        batch_size = int(interaction_query_features.shape[0])

        # 扩展primary actor seed, 由[1,D]变为[B,D], 可以理解为查询角色:我要寻找主要因果参与者
        primary_seed = self.primary_actor_seed.expand(batch_size,-1,)

        # 扩展secondary actor seed, 由[1,D]变为[B,D] 可以理解为查询角色:我要寻找次要影响参与者
        secondary_seed = self.secondary_actor_seed.expand(batch_size,-1,)

        # 构造4个初始交互token [B,4,D]
        initial_tokens = torch.stack(
            (
                route_token,                                 # [B,D]
                ego_token,                                   # [B,D]
                primary_seed + route_token + ego_token,      # [B,D]  他的初始化含义是在当前参考路径和自车未来动作的共同上下文下,寻找最主要的参与者,主要 actor 不是仅靠一个随机 query 查找，而是已经知道"自车准备走哪条路径、自车预计怎样运动、语言和导航要求什么"
                secondary_seed + route_token + ego_token,    # [B,D]  同上
            ),
            dim=1,
        )
        # 这里需要说明一下为什么"primary_seed"/"secondary_seed"需要加上"route_token"和"ego_token"
        # 因为"主要参与者"不是纯粹由外观决定的,而是由它对路线和自车未来运动的影响决定的

        # [B,4,D] 四个类型embedding分别告诉网络:位置0是route、位置1是ego、位置2是primary actor、位置3是secondary actor
        # 虽然四个 token 已经通过构造方式有所不同，但加入独立类型 embedding 后，身份区分更明确
        initial_tokens = (initial_tokens + self.token_type_embeddings.to(dtype=initial_tokens.dtype))

        # 将四个交互token转为视觉查询 [B,4,D]->[B,4,128] 四个quary分别生成一个128维视觉查询
        spatial_query = self.spatial_query(initial_tokens)
        
        # 将视觉特征映射成空间key [B,384,D]-> [B,384,128]
        spatial_key = self.spatial_key(visual_features)
        
        # 计算空间注意力分数 [B,4,384]
        spatial_scores = torch.einsum(
            "btd,bvd->btv",
            spatial_query,
            spatial_key,
        ) * (self.attention_dim ** -0.5)
        
        # 把分数转换为空间注意力概率 [B,4,384] 384维度上的和为1,目的是表征六个相机之间是相互竞争的
        spatial_attention = torch.softmax(spatial_scores.float(),dim=-1,).to(dtype=visual_features.dtype)

        # 生成视觉value [B,4,384]  Key 用于决定“看哪里”，Value 用于读取“那个位置有什么内容”
        visual_value = self.spatial_value(visual_features)
        
        # 根据注意力读取视觉上下文 [B,4,D]
        visual_context = torch.einsum(
            "btv,bvd->btd",
            spatial_attention,
            visual_value,
        )

        # 残差融合初始token与视觉上下文 [B,4,D]
        interaction_tokens = self.visual_context_norm(
            initial_tokens + self.spatial_output(visual_context)
        )

        # 四个交互token之间互相交流 以下形式是标准多头自注意力 [B,4,D]
        # 它允许四种语义互相影响,表明它们不是四个完全独立的预测任务,而是同一驾驶决策的不同组成成分
        self_context, _ = self.interaction_self_attention(
            interaction_tokens,
            interaction_tokens,
            interaction_tokens,
            need_weights=False,
        )

        # 自注意力残差连接 这样可以保留原有语义,同时加入其他 token 提供的信息
        interaction_tokens = self.interaction_self_norm(interaction_tokens + self_context)

        # 输出 [B,4,D]
        interaction_tokens = self.interaction_ffn_norm(interaction_tokens + self.interaction_ffn(interaction_tokens))

        # 恢复六相机空间结构 [B,4,6,64]
        spatial_attention = spatial_attention.reshape(
            batch_size,
            len(INTERACTION_TOKEN_KEYS),
            self.num_cameras,
            self.tokens_per_camera,
        )

        # 单独提取主要参与者注意力 [B,6,64]
        primary_attention = spatial_attention[:, 2]

        # 单独提取次要参与者注意力 [B,6,64]
        secondary_attention = spatial_attention[:, 3]

        # 主要参与者注意力熵 primary_entropy:[B] 接近0表示注意力非常集中 接近1表示注意力在384个视觉token上非常分散 
        # 这个指标可以用来监控主要参与者注意力是否发生塌缩或过度扩散
        primary_probability = primary_attention.float().clamp_min(1e-8)
        primary_entropy = -(primary_probability * primary_probability.log()).sum(dim=(-2, -1)) / math.log(float(self.num_visual_tokens))

        return {
            "interaction_tokens": interaction_tokens,                      # [B,4,D] 四个最终统一交互 token 这是最核心的输出
            "spatial_attention": spatial_attention,                        # [B,4,6,64] 四类交互 token 完整的六视角空间注意力
            "primary_spatial_attention": primary_attention,                # [B,6,64] 主要参与者 token 对六视角 patch 的注意力
            "secondary_spatial_attention": secondary_attention,            # [B,6,64] 次要参与者 token 对六视角 patch 的注意力
            "primary_camera_attention": primary_attention.sum(dim=-1),     # [B,6] 六相机每个相机的注意力 和为1
            "secondary_camera_attention": secondary_attention.sum(dim=-1), # [B,6] 六相机每个相机的注意力 和为1
            "primary_spatial_attention_entropy": primary_entropy,          # [B] 主要参与者空间注意力的归一化熵
        }

    @staticmethod
    def _prediction_span_mask(question_span_mask: Tensor) -> Tensor:
        """将答案token位置转换为真正预测这些token的前一位置。"""
        prediction_mask = torch.zeros_like(question_span_mask)
        prediction_mask[..., :-1] = question_span_mask[..., 1:]
        return prediction_mask

    @classmethod
    def _pool_question_prediction_features(
        cls,
        language_features: Tensor,
        question_span_mask: Tensor,) -> Tensor:
        prediction_mask = cls._prediction_span_mask(
            question_span_mask
        ).to(
            device=language_features.device,
            dtype=language_features.dtype,
        )
        denominator = prediction_mask.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        return torch.einsum(
            "bql,bld->bqd",
            prediction_mask,
            language_features,
        ) / denominator

    def build_counterfactual_visual_features(
        self,
        visual_features: Tensor,
        participant_target: Tensor,
        valid_mask: Tensor,
        strength: float,
        mask_gamma: float,) -> Tuple[Tensor, Tensor]:
        """
        用主要参与者空间标签替换其视觉token，构造对象移除反事实场景。

        替换值使用同一相机内未被参与者覆盖token的加权均值，避免简单置零
        产生训练时不存在的异常视觉分布。也就是说不是简单的actor区域=0,而是用对影响及的背景特征去替代,从而减少非常不自然的视觉分布.
        """
        expected_shape = (
            visual_features.shape[0],
            self.num_cameras,
            self.tokens_per_camera,
        )
        if participant_target.shape != expected_shape:
            raise ValueError(
                "Counterfactual participant target must have shape "
                f"{expected_shape}, but received "
                f"{tuple(participant_target.shape)}."
            )
        if valid_mask.shape != (visual_features.shape[0],):
            raise ValueError(
                "Counterfactual valid mask must have shape [B]."
            )

        target = participant_target.to(
            device=visual_features.device,
            dtype=torch.float32,
        )
        valid = valid_mask.to(
            device=visual_features.device,
            dtype=torch.bool,
        )
        target_max = target.amax(
            dim=(-2, -1),
            keepdim=True,
        )
        invalid = valid & (
            target_max.reshape(-1) <= 0.0
        )
        if bool(invalid.any().item()):
            raise ValueError(
                "A valid counterfactual participant target has zero mass."
            )

        intervention_mask = torch.where(
            valid.view(-1, 1, 1),
            target / target_max.clamp_min(1e-8),
            torch.zeros_like(target),
        )
        intervention_mask = intervention_mask.clamp(0.0, 1.0).pow(
            max(float(mask_gamma), 1e-4)
        )
        intervention_mask = (
            intervention_mask * float(strength)
        ).clamp(0.0, 1.0)

        visual = visual_features.reshape(
            visual_features.shape[0],
            self.num_cameras,
            self.tokens_per_camera,
            visual_features.shape[-1],
        )
        background_weight = (1.0 - intervention_mask).to(
            dtype=visual.dtype
        )
        camera_background = (
            visual * background_weight.unsqueeze(-1)
        ).sum(dim=2) / background_weight.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1e-4)
        camera_background = camera_background.detach().unsqueeze(2)

        mask = intervention_mask.to(
            dtype=visual.dtype
        ).unsqueeze(-1)
        counterfactual = (
            visual * (1.0 - mask)
            + camera_background * mask
        )
        return (
            counterfactual.reshape_as(visual_features),
            intervention_mask,
        )

    def compute_counterfactual_language_consistency_loss(
        self,
        original_language_features: Tensor,
        counterfactual_language_features: Tensor,
        original_interaction_tokens: Tensor,
        counterfactual_interaction_tokens: Tensor,
        original_question_span_mask: Tensor,
        counterfactual_question_span_mask: Tensor,
        full_scene_waypoints: Tensor,
        counterfactual_waypoints: Tensor,
        valid_mask: Tensor,
        minimum_change: float,) -> Tuple[Tensor, Tensor]:
        """
        约束对象移除前后Q1-Q4预测状态的变化，与交互证据和真实动作
        变化保持一致。原始与反事实答案允许具有不同token长度。

        Q1对应主要参与者token变化，Q2对应route token变化，Q3/Q4对应
        标签生成阶段对象移除后重规划轨迹与完整场景轨迹之间的真实差异。
        """
        if (
            original_language_features.shape[0]
            != counterfactual_language_features.shape[0]
            or original_language_features.shape[-1]
            != counterfactual_language_features.shape[-1]
        ):
            raise ValueError(
                "Original and counterfactual language features must share "
                "batch and hidden dimensions."
            )
        if full_scene_waypoints.shape != counterfactual_waypoints.shape:
            raise ValueError(
                "Full-scene and counterfactual waypoint shapes must match."
            )
        if original_interaction_tokens.shape != (
            counterfactual_interaction_tokens.shape
        ):
            raise ValueError(
                "Original and counterfactual interaction token shapes must match."
            )

        original_question = self._pool_question_prediction_features(
            original_language_features,
            original_question_span_mask,
        )
        counterfactual_question = (
            self._pool_question_prediction_features(
                counterfactual_language_features,
                counterfactual_question_span_mask,
            )
        )
        language_delta = self.counterfactual_language_delta_projection(
            counterfactual_question - original_question
        ).float()

        token_delta = self.token_alignment_projection(
            counterfactual_interaction_tokens
            - original_interaction_tokens
        ).float()
        action_effect = (
            full_scene_waypoints.float()
            - counterfactual_waypoints.float()
        ).reshape(full_scene_waypoints.shape[0], -1)
        expected_action_dim = self.num_ego_queries * 2
        if action_effect.shape[-1] != expected_action_dim:
            raise ValueError(
                "Counterfactual waypoint effect must contain "
                f"{expected_action_dim} values, but received "
                f"{action_effect.shape[-1]}."
            )
        action_reference = self.counterfactual_action_effect_projection(
            action_effect
        ).float()

        references = torch.stack(
            (
                token_delta[:, 2],
                token_delta[:, 0],
                action_reference,
                action_reference,
            ),
            dim=1,
        )
        cosine = (
            F.normalize(language_delta, dim=-1)
            * F.normalize(references, dim=-1)
        ).sum(dim=-1)
        direction_loss = 1.0 - cosine

        scale = math.sqrt(float(self.hidden_size))
        language_change = language_delta.norm(dim=-1) / scale
        reference_change = references.detach().norm(dim=-1) / scale
        magnitude_target = reference_change.clamp_min(
            float(minimum_change)
        )
        magnitude_loss = F.smooth_l1_loss(
            language_change,
            magnitude_target,
            reduction="none",
        )

        per_sample_loss = (
            direction_loss + magnitude_loss
        ).mean(dim=1)
        count = valid_mask.to(
            device=per_sample_loss.device,
            dtype=per_sample_loss.dtype,
        )
        return per_sample_loss * count, count

    def counterfactual_parameter_zero(self) -> Tensor:
        return (
            sum(
                parameter.sum()
                for parameter in (
                    list(
                        self.counterfactual_language_delta_projection.parameters()
                    )
                    + list(
                        self.counterfactual_action_effect_projection.parameters()
                    )
                )
            )
            * 0.0
        )

    def compute_language_alignment_loss(self,
        language_features: Tensor,             # [B,L,D]
        contextual_interaction_tokens: Tensor, # [B,4,D]
        question_span_mask: Tensor,            # [B,4,L] 四个问题对应的四个答案的位置
        valid_mask: Tensor,                    # [B] 表示这个样本是否成功识别出了完整的 A1、A2、A3、A4 四段答案
        temperature: float,) -> Tuple[Tensor, Tensor]:
        
        """
        将Q1-Q4答案表示分别对齐到固定语义交互token
        """

        # 安全性检查
        if question_span_mask.shape[:2] != (language_features.shape[0],len(LANGUAGE_QUESTION_TO_TOKEN_INDEX),):
            raise ValueError(
                "Question span mask must have shape [B,4,L]."
            )

        # 安全性检查    
        if question_span_mask.shape[-1] != language_features.shape[1]:
            raise ValueError(
                "Question span mask length must match language features."
            )

        question_features = self._pool_question_prediction_features(
            language_features,
            question_span_mask,
        )

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
        contextual_interaction_tokens: Tensor,) -> Tuple[Tensor, Tensor]:
        """
        对齐route/ego动作query与对应的上下文化交互token
        """

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
    valid_mask: Tensor,) -> Dict[str, Tuple[Tensor, Tensor]]:
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


def compute_counterfactual_attention_suppression_loss(
    counterfactual_primary_attention: Tensor,
    intervention_mask: Tensor,
    valid_mask: Tensor,) -> Dict[str, Tuple[Tensor, Tensor]]:
    """对象移除后，主要参与者注意力不得继续停留在被删除区域。"""
    if counterfactual_primary_attention.shape != intervention_mask.shape:
        raise ValueError(
            "Counterfactual attention and intervention mask shapes must match."
        )
    overlap = (
        counterfactual_primary_attention.float()
        * intervention_mask.to(
            device=counterfactual_primary_attention.device,
            dtype=torch.float32,
        )
    ).sum(dim=(-2, -1))
    count = valid_mask.to(
        device=overlap.device,
        dtype=overlap.dtype,
    )
    return {
        "counterfactual_visual_suppression_loss": (
            overlap * count,
            count,
        )
    }


def compute_counterfactual_future_world_losses(
    counterfactual_logits: Tensor,
    full_scene_target: Tensor,
    valid_mask: Tensor,) -> Dict[str, Tuple[Tensor, Tensor]]:
    """
    对象移除后C2必须消失；导航路线C0和未删除次要参与者C4保持不变。
    C1不在此处监督，因为对象移除会真实改变自车未来轨迹。
    """
    if counterfactual_logits.ndim != 4 or counterfactual_logits.shape[1] != 4:
        raise ValueError(
            "Counterfactual future logits must have shape [B,4,H,W]."
        )
    if (
        full_scene_target.ndim != 4
        or full_scene_target.shape[:2]
        != counterfactual_logits.shape[:2]
    ):
        raise ValueError(
            "Counterfactual future target must have shape [B,4,H,W]."
        )
    full_scene_target = full_scene_target.to(
        device=counterfactual_logits.device,
        dtype=torch.float32,
    )
    if full_scene_target.shape[-2:] != counterfactual_logits.shape[-2:]:
        full_scene_target = F.interpolate(
            full_scene_target,
            size=counterfactual_logits.shape[-2:],
            mode="nearest",
        )

    primary_removed = F.binary_cross_entropy_with_logits(
        counterfactual_logits[:, 2].float(),
        torch.zeros_like(counterfactual_logits[:, 2].float()),
        reduction="none",
    ).mean(dim=(-2, -1))

    route_loss = F.binary_cross_entropy_with_logits(
        counterfactual_logits[:, 0].float(),
        full_scene_target[:, 0].float(),
        reduction="none",
    ).mean(dim=(-2, -1))
    secondary_loss = F.binary_cross_entropy_with_logits(
        counterfactual_logits[:, 3].float(),
        full_scene_target[:, 3].float(),
        reduction="none",
    ).mean(dim=(-2, -1))
    invariant = 0.5 * (route_loss + secondary_loss)

    count = valid_mask.to(
        device=counterfactual_logits.device,
        dtype=torch.float32,
    )
    return {
        "counterfactual_world_primary_removal_loss": (
            primary_removed * count,
            count,
        ),
        "counterfactual_world_invariance_loss": (
            invariant * count,
            count,
        ),
    }


def compute_actor_disentanglement_losses(
    primary_attention: Tensor,
    secondary_attention: Tensor,
    contextual_interaction_tokens: Tensor,
    secondary_exists: Tensor,
    attention_overlap_margin: float,
    token_cosine_margin: float,) -> Dict[str, Tuple[Tensor, Tensor]]:
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
