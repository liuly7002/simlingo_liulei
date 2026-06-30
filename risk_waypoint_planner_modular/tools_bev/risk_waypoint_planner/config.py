# -*- coding: utf-8 -*-

from .common import *

def build_argparser():
    parser = argparse.ArgumentParser(
        description="Generate risk-planned ego future waypoints using kinematic bicycle rollout."
    )

    # Input/output.
    parser.add_argument("--input", type=str, required=True, help="Route directory or data root.")
    parser.add_argument("--recursive", action="store_true", help="Recursively process all route dirs.")
    parser.add_argument("--frame", type=str, default=None, help="Only process one frame, e.g. 0000.")
    parser.add_argument("--augmented", action="store_true", help="Use augmented folders.")

    parser.add_argument("--measurements_folder", type=str, default="measurements")
    parser.add_argument("--costmap_folder", type=str, default="costmap")
    parser.add_argument("--costmap_folder_augmented", type=str, default="costmap_augmented")
    parser.add_argument("--bev_meta_folder", type=str, default="bev_meta")
    parser.add_argument("--bev_meta_folder_augmented", type=str, default="bev_meta_augmented")

    parser.add_argument("--output_folder", type=str, default="risk_waypoints_bicycle")
    parser.add_argument("--output_folder_augmented", type=str, default="risk_waypoints_bicycle_augmented")
    parser.add_argument("--debug_folder", type=str, default="risk_waypoints_bicycle_debug")
    parser.add_argument("--debug_folder_augmented", type=str, default="risk_waypoints_bicycle_debug_augmented")

    # Route config.
    parser.add_argument("--route_key", type=str, default="route", help="Measurement route key: route or route_original.")
    parser.add_argument("--default_pixels_per_meter", type=float, default=2.0)

    # Reference candidate generation.
    parser.add_argument(
        "--offsets_m",
        type=str,
        default="0,-1.0,1.0,-1.5,1.5,-2.0,2.0",
        help="Comma-separated lateral offsets. Positive means right; negative means left.",
    )
    parser.add_argument("--offset_start_m", type=float, default=2.0)
    parser.add_argument("--offset_transition_m", type=float, default=8.0)
    parser.add_argument("--reference_horizon_m", type=float, default=30.0)
    parser.add_argument("--reference_spacing_m", type=float, default=0.5)

    # Multi-behavior generation.
    parser.add_argument(
        "--disable_multibehavior",
        action="store_true",
        help="Use the legacy offset x speed candidate generator instead of behavior-conditioned candidates.",
    )
    parser.add_argument(
        "--behavior_candidate_policy",
        type=str,
        default="all",
        choices=["all", "gated"],
        help="all: save all behavior types and mark inactive ones invalid; gated: only create active behavior candidates.",
    )
    parser.add_argument(
        "--allow_inactive_behavior_selection",
        action="store_true",
        help="Allow inactive behavior candidates to be selected if they score best. Usually keep this disabled for training-label generation.",
    )
    parser.add_argument("--left_nudge_offset_m", type=float, default=1.0)
    parser.add_argument("--right_nudge_offset_m", type=float, default=1.0)
    parser.add_argument("--nudge_return_start_m", type=float, default=16.0)
    parser.add_argument("--nudge_return_transition_m", type=float, default=8.0)
    parser.add_argument("--nudge_speed_mps", type=float, default=4.0)
    parser.add_argument("--creep_speed_mps", type=float, default=1.0)
    parser.add_argument("--expert_stopped_speed_mps", type=float, default=0.3)
    parser.add_argument("--behavior_diagnosis_horizon_m", type=float, default=24.0)
    parser.add_argument("--behavior_diagnosis_spacing_m", type=float, default=1.0)
    parser.add_argument("--behavior_blocked_cost_threshold", type=float, default=80.0)
    parser.add_argument("--behavior_blocked_hard_ratio", type=float, default=0.08)
    parser.add_argument("--behavior_free_cost_threshold", type=float, default=40.0)
    parser.add_argument("--behavior_free_hard_ratio", type=float, default=0.03)
    parser.add_argument("--yield_stop_margin_m", type=float, default=3.0)
    parser.add_argument("--min_yield_stop_distance_m", type=float, default=3.0)
    parser.add_argument("--inactive_behavior_penalty", type=float, default=1000.0)
    parser.add_argument("--stop_behavior_prior_cost", type=float, default=5.0)
    parser.add_argument("--nudge_behavior_prior_cost", type=float, default=1.0)

    # Yield candidate.
    parser.add_argument("--include_yield", action="store_true")
    parser.add_argument("--allow_yield_selection", action="store_true")
    parser.add_argument(
        "--allow_emergency_selection",
        action="store_true",
        help="Allow emergency_brake to be selected as the final risk-planned label. "
            "By default, emergency_brake is saved only as a diagnostic candidate.",
    )
    parser.add_argument("--yield_stop_distance_m", type=float, default=6.0)

    # Speed profiles.
    parser.add_argument(
        "--speed_modes",
        type=str,
        default="keep,slow,stop",
        help="Comma-separated speed modes: keep,current,slow,cautious,fast,stop.",
    )
    parser.add_argument("--target_speed_key", type=str, default="target_speed")
    parser.add_argument("--cautious_speed_mps", type=float, default=3.0)
    parser.add_argument("--min_rollout_speed_mps", type=float, default=0.5)

    # Future waypoint/output timing.
    parser.add_argument("--num_future_waypoints", type=int, default=10)
    parser.add_argument("--future_fps", type=float, default=4.0)
    parser.add_argument("--model_fps", type=float, default=20.0)

    # Temporal costmap scoring.
    parser.add_argument(
        "--disable_future_costmaps",
        action="store_true",
        help="Disable temporal future-costmap scoring and fall back to current-frame scoring only.",
    )
    parser.add_argument(
        "--future_frame_stride",
        type=int,
        default=1,
        help="Frame-id stride between adjacent future costmap frames. Default: frame+1, frame+2, ...",
    )
    parser.add_argument(
        "--future_costmap_combine",
        type=str,
        default="max",
        choices=["max", "add", "future_only"],
        help="How to combine current costmap with warped future costmap for each time step.",
    )
    parser.add_argument(
        "--future_costmap_weight",
        type=float,
        default=1.0,
        help="Weight applied to warped future costmaps before combination.",
    )
    parser.add_argument(
        "--future_warp_border_cost",
        type=float,
        default=0.0,
        help="Cost value used outside a source future costmap during warping.",
    )
    parser.add_argument(
        "--future_missing_policy",
        type=str,
        default="repeat_last",
        choices=["repeat_last", "current", "zero"],
        help="How to fill missing future costmaps near the end of a route.",
    )
    parser.add_argument(
        "--temporal_index_mode",
        type=str,
        default="round",
        choices=["round", "floor", "ceil"],
        help="How to map dense rollout integration steps to future costmap frame indices.",
    )
    parser.add_argument(
        "--temporal_cost_clip_max",
        type=float,
        default=0.0,
        help="Clip combined temporal costmaps to this max value. <=0 means no clipping.",
    )

    parser.add_argument(
        "--save_temporal_debug",
        action="store_true",
        help="Save cumulative temporal costmap debug image for each current frame.",
    )
    parser.add_argument(
        "--temporal_debug_folder",
        type=str,
        default="risk_waypoints_bicycle_temporal_debug",
        help="Folder name for cumulative temporal costmap debug images.",
    )
    parser.add_argument(
        "--temporal_debug_vmax",
        type=float,
        default=0.0,
        help="Visualization max cost for temporal debug images. <=0 means auto percentile normalization.",
    )
    parser.add_argument(
        "--temporal_debug_save_npy",
        action="store_true",
        help="Also save cumulative temporal costmap as .npy.",
    )


    # Kinematic bicycle model.
    parser.add_argument("--wheelbase_m", type=float, default=2.9)
    parser.add_argument("--max_speed_mps", type=float, default=12.0)
    parser.add_argument("--max_steer_rad", type=float, default=0.60)
    parser.add_argument("--max_steer_rate_radps", type=float, default=0.80)
    parser.add_argument("--max_accel_mps2", type=float, default=2.5)
    parser.add_argument("--min_accel_mps2", type=float, default=-5.0)
    parser.add_argument("--speed_kp", type=float, default=1.2)

    # Pure pursuit controller.
    parser.add_argument("--lookahead_base_m", type=float, default=2.0)
    parser.add_argument("--lookahead_gain", type=float, default=0.35)
    parser.add_argument("--lookahead_min_m", type=float, default=2.0)
    parser.add_argument("--lookahead_max_m", type=float, default=8.0)

    # Vehicle footprint for cost sampling.
    parser.add_argument("--score_footprint", action="store_true", help="Sample costmap using ego footprint points, not only centerline.")
    parser.add_argument("--ego_extent_x_m", type=float, default=2.25)
    parser.add_argument("--ego_extent_y_m", type=float, default=1.00)

    # Scoring weights.
    parser.add_argument("--mean_cost_weight", type=float, default=1.0)
    parser.add_argument("--max_cost_weight", type=float, default=0.05)
    parser.add_argument("--hard_ratio_weight", type=float, default=80.0)
    parser.add_argument("--out_of_bounds_weight", type=float, default=200.0)
    parser.add_argument("--route_deviation_weight", type=float, default=2.0)
    parser.add_argument("--acc_weight", type=float, default=0.2)
    parser.add_argument("--steer_weight", type=float, default=1.0)
    parser.add_argument("--steer_rate_weight", type=float, default=0.2)
    parser.add_argument("--lat_acc_weight", type=float, default=0.5)
    parser.add_argument("--yaw_rate_weight", type=float, default=0.5)

    # Feasibility thresholds.
    parser.add_argument("--hard_cost_threshold", type=float, default=100.0)
    parser.add_argument("--max_hard_ratio", type=float, default=0.35)
    parser.add_argument("--max_out_of_bounds_ratio", type=float, default=0.05)
    parser.add_argument("--out_of_bounds_cost", type=float, default=200.0)
    parser.add_argument("--max_lateral_accel_mps2", type=float, default=4.0)
    parser.add_argument("--max_yaw_rate_radps", type=float, default=1.2)
    parser.add_argument("--max_route_deviation_m", type=float, default=4.0)

    # Debug / logging.
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--debug_clip_max", type=float, default=200.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--raise_on_error", action="store_true")

    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to save runtime log. If not set, a timestamped log is saved under input/log_folder.",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="risk_waypoints_bicycle_logs",
        help="Default log folder name under input path when --log_file is not specified.",
    )
    parser.add_argument(
        "--quiet_console",
        action="store_true",
        help="Only write logs to file, do not print logs to terminal.",
    )

    # RGB waypoint projection debug.
    parser.add_argument(
        "--save_rgb_debug",
        action="store_true",
        help="Project expert and risk-planned waypoints onto front RGB images.",
    )
    parser.add_argument("--rgb_folder", type=str, default="rgb")
    parser.add_argument("--rgb_folder_augmented", type=str, default="rgb_augmented")
    parser.add_argument("--rgb_debug_folder", type=str, default="risk_waypoints_bicycle_rgb_debug")
    parser.add_argument("--rgb_debug_folder_augmented", type=str, default="risk_waypoints_bicycle_rgb_debug_augmented")

    # Future surrounding-vehicle trajectory projection debug.
    parser.add_argument("--boxes_folder", type=str, default="boxes")
    parser.add_argument("--boxes_folder_augmented", type=str, default="boxes")
    parser.add_argument(
        "--disable_future_actor_rgb",
        action="store_true",
        help="Do not draw surrounding vehicles' future tracks on RGB debug images.",
    )
    parser.add_argument(
        "--future_actor_classes",
        type=str,
        default="car",
        help="Comma-separated box classes to draw as future actors, e.g. car or car,static_car.",
    )
    parser.add_argument(
        "--future_actor_include_new",
        action="store_true",
        help="Also draw actors that are absent in the current frame but appear in future boxes.",
    )
    parser.add_argument(
        "--future_actor_max_distance_m",
        type=float,
        default=60.0,
        help="Maximum actor distance in current ego frame for RGB future-actor debug drawing.",
    )
    parser.add_argument("--future_actor_rgb_radius", type=int, default=3)
    parser.add_argument("--future_actor_rgb_thickness", type=int, default=2)
    parser.add_argument("--future_actor_rgb_color_b", type=int, default=0)
    parser.add_argument("--future_actor_rgb_color_g", type=int, default=165)
    parser.add_argument("--future_actor_rgb_color_r", type=int, default=255)
    parser.add_argument(
        "--no_future_actor_draw_ids",
        dest="future_actor_draw_ids",
        action="store_false",
        help="Do not draw actor ids beside future vehicle tracks.",
    )
    parser.set_defaults(future_actor_draw_ids=True)

    # Camera intrinsics/extrinsics for RGB projection.
    parser.add_argument("--camera_fov", type=float, default=110.0)

    parser.add_argument("--camera_x", type=float, default=-1.5)
    parser.add_argument("--camera_y", type=float, default=0.0)
    parser.add_argument("--camera_z", type=float, default=2.0)

    parser.add_argument("--camera_roll_deg", type=float, default=0.0)
    parser.add_argument("--camera_pitch_deg", type=float, default=0.0)
    parser.add_argument("--camera_yaw_deg", type=float, default=0.0)

    parser.add_argument(
        "--waypoint_ground_z_m",
        type=float,
        default=0.0,
        help="Ground height of waypoints in ego-local frame.",
    )
    parser.add_argument(
        "--min_projection_depth_m",
        type=float,
        default=0.1,
        help="Minimum positive camera depth for RGB projection.",
    )

    # Dreamer-style candidate language-action labels.
    parser.add_argument(
        "--save_dreamer_candidates",
        action="store_true",
        help="Save SimLingo-style dreamer labels for all risk-planned candidate rollouts.",
    )  # 开启 dreamer 数据生成
    parser.add_argument(
        "--dreamer_output_folder",
        type=str,
        default="risk_waypoints_bicycle_dreamer",
        help="Folder for dreamer candidate labels.",
    )  # 单独保存 dreamer 标签的文件夹
    parser.add_argument(
        "--dreamer_output_folder_augmented",
        type=str,
        default="risk_waypoints_bicycle_dreamer_augmented",
        help="Folder for augmented dreamer candidate labels.",
    )
    parser.add_argument(
        "--dreamer_include_invalid",
        action="store_true",
        help="Also save invalid/inactive counterfactual candidates as negative dreamer samples.",
    )  # 是否保存不可执行/未激活/高风险候选，作为负样本或反事实样本
    parser.add_argument(
        "--dreamer_max_candidates",
        type=int,
        default=-1,
        help="Maximum number of dreamer candidates to save per frame. <=0 means save all selected/allowed candidates and optionally invalid candidates.",
    )  # 每帧最多保存多少条候选。默认 -1 表示不限制

    return parser


def parse_config():
    parser = build_argparser()
    return parser.parse_args()
