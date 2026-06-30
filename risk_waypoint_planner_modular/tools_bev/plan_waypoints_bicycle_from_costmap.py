#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modular entry point for risk-planned bicycle waypoint generation.

The original monolithic script has been split into tools_bev/risk_waypoint_planner/.
This file keeps the command-line interface compatible with the original script.
"""

from pathlib import Path

from risk_waypoint_planner.common import LOGGER
from risk_waypoint_planner.config import build_argparser
from risk_waypoint_planner.logging_utils import setup_logging
from risk_waypoint_planner.processor import find_route_dirs, process_route_dir


def main():
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    setup_logging(args, input_path)
    LOGGER.info(f"[Config] input={input_path}")
    LOGGER.info(f"[Config] recursive={args.recursive}")
    LOGGER.info(f"[Config] frame={args.frame}")
    LOGGER.info(f"[Config] save_debug={args.save_debug}")
    LOGGER.info(f"[Config] score_footprint={args.score_footprint}")
    LOGGER.info(f"[Config] speed_modes={args.speed_modes}")
    LOGGER.info(f"[Config] offsets_m={args.offsets_m}")
    LOGGER.info(f"[Config] multibehavior={not args.disable_multibehavior}")
    LOGGER.info(f"[Config] behavior_candidate_policy={args.behavior_candidate_policy}")
    LOGGER.info(f"[Config] left/right_nudge_offset_m={args.left_nudge_offset_m}/{args.right_nudge_offset_m}")
    LOGGER.info(f"[Config] temporal_future_costmaps={not args.disable_future_costmaps}")
    LOGGER.info(f"[Config] future_frame_stride={args.future_frame_stride}")
    LOGGER.info(f"[Config] future_costmap_combine={args.future_costmap_combine}")
    LOGGER.info(f"[Config] future_missing_policy={args.future_missing_policy}")

    LOGGER.info(f"[Config] save_rgb_debug={args.save_rgb_debug}")
    LOGGER.info(f"[Config] rgb_folder={args.rgb_folder}")
    LOGGER.info(f"[Config] boxes_folder={args.boxes_folder}")
    LOGGER.info(f"[Config] draw_future_actor_rgb={not args.disable_future_actor_rgb}")
    LOGGER.info(f"[Config] future_actor_classes={args.future_actor_classes}")
    LOGGER.info(f"[Config] future_actor_max_distance_m={args.future_actor_max_distance_m}")
    LOGGER.info(f"[Config] camera_fov={args.camera_fov}")
    LOGGER.info(f"[Config] camera_pos=({args.camera_x}, {args.camera_y}, {args.camera_z})")
    LOGGER.info(f"[Config] camera_rot_rpy_deg=({args.camera_roll_deg}, {args.camera_pitch_deg}, {args.camera_yaw_deg})")

    if abs(args.model_fps / args.future_fps - round(args.model_fps / args.future_fps)) > 1e-6:
        raise ValueError(
            f"model_fps/future_fps must be an integer ratio. Got {args.model_fps}/{args.future_fps}."
        )

    if args.recursive:
        route_dirs = find_route_dirs(input_path, args)
        LOGGER.info(f"[Info] Found {len(route_dirs)} route dirs under {input_path}")

        total_ok = 0
        total_frames = 0
        for route_dir in route_dirs:
            ok, total = process_route_dir(route_dir, args)
            total_ok += ok
            total_frames += total

        LOGGER.info(f"[All Done] {total_ok}/{total_frames} frames processed.")
    else:
        process_route_dir(input_path, args)


if __name__ == "__main__":
    main()
