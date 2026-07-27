# -*- coding: utf-8 -*-

"""Expert-conditioned structured future-world label generation for Driving data."""

#修改20260727：Driving结构化世界标签删除原C3交互通道。
# 语义编号继续保留C0/C1/C2/C4，但张量中按以下4通道顺序连续存储。
CHANNEL_NAMES = (
    "selected_route_ego_footprint_occupancy",
    "future_ego_footprint_occupancy",
    "primary_causal_actor_future_footprint_occupancy",
    "secondary_actor_future_footprint_occupancy",
)
