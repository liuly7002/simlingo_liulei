# -*- coding: utf-8 -*-

"""LG-keyed structured future-world label generation for Driving data."""

#修改20261016：统一C0/C1/C2三个结构化语义通道。
CHANNEL_NAMES = (
    "selected_route_ego_footprint_occupancy",
    "future_ego_footprint_occupancy",
    "primary_causal_actor_future_footprint_occupancy",
)
