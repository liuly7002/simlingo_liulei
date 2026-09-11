from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore


@dataclass
class VLMEncoderConfig:
    variant: str = 'OpenGVLab/InternVL2-1B'
    embed_dim: int = 512
    freeze: bool = False

    _target_: str = "simlingo_training.models.encoder.vlm.VLMEncoderModel"


@dataclass
class LanguageModelConfig:
    variant: str = 'OpenGVLab/InternVL2-1B'
    lora: bool = True
    lora_alpha: int = 64
    lora_r: int = 32
    lora_dropout: float = 0.1

    _target_: str = "simlingo_training.models.language_model.llm.LLM"


@dataclass
class DrivingModelConfig:
    vision_model: Any
    language_model: Any

    lr: float = 5e-2

    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = '2d'
    predict_route_as_wps: bool = True

    # 统一决策交互表示配置。
    interaction_attention_dim: int = 128
    interaction_num_heads: int = 8
    interaction_dropout: float = 0.1

    # 第二阶段语言—交互—动作一致性配置。
    interaction_alignment_temperature: float = 0.1
    language_interaction_alignment_loss_weight: float = 0.05
    action_interaction_alignment_loss_weight: float = 0.05
    actor_attention_separation_loss_weight: float = 0.02
    actor_token_separation_loss_weight: float = 0.02
    actor_attention_overlap_margin: float = 0.35
    actor_token_cosine_margin: float = 0.30

    # 反事实对象移除训练。
    use_counterfactual_intervention_training: bool = False
    counterfactual_intervention_probability: float = 1.0
    counterfactual_intervention_strength: float = 1.0
    counterfactual_intervention_mask_gamma: float = 0.5
    counterfactual_language_min_change: float = 0.02

    counterfactual_visual_suppression_loss_weight: float = 0.05
    counterfactual_language_supervision_loss_weight: float = 0.10
    counterfactual_language_consistency_loss_weight: float = 0.05
    counterfactual_action_target_loss_weight: float = 0.10
    counterfactual_action_effect_loss_weight: float = 0.10
    counterfactual_route_invariance_loss_weight: float = 0.05
    counterfactual_world_primary_removal_loss_weight: float = 0.05
    counterfactual_world_invariance_loss_weight: float = 0.02

    # 主要关键参与者六视角视觉token级空间监督。
    use_participant_spatial_attention_supervision: bool = False
    participant_spatial_attention_loss_weight: float = 0.05

    # 是否启用四通道结构化未来世界辅助预测。
    use_future_interaction_prediction: bool = False

    # 当前解码器固定输出128×128。
    future_interaction_output_size: int = 128

    # 完整未来世界辅助任务在总损失中的基础权重。
    future_interaction_loss_weight: float = 0.05

    # Dice损失相对于BCE损失的权重。
    future_interaction_dice_loss_weight: float = 1.0

    # 四个通道的正像素权重，顺序为C0、C1、C2、C4。
    future_interaction_positive_weights: Tuple[
        float,
        float,
        float,
        float,
    ] = (
        20.0,
        20.0,
        80.0,
        100.0,
    )

    # 四个通道在未来世界任务内部的权重，顺序为C0、C1、C2、C4。
    future_interaction_channel_weights: Tuple[
        float,
        float,
        float,
        float,
    ] = (
        1.0,
        1.0,
        2.0,
        4.0,
    )

    _target_: str = "simlingo_training.models.driving.DrivingModel"


@dataclass
class DatasetBaseConfig:
    data_path: str = "/home/katrinrenz/coding/wayve_carla/database/expertv3_2*"

    bucket_path: str = "data/buckets"

    cut_bottom_quarter: bool = False
    use_1d_wps: bool = False

    use_commentary: bool = False
    use_qa: bool = False
    qa_augmentation: bool = True
    commentary_augmentation: bool = True
    use_old_towns: bool = False
    use_only_old_towns: bool = False
    use_town13: bool = False

    #修改20260728：普通Driving和LG统一按route目录划分训练集和验证集。
    # true：
    #   train只使用routes_training；
    #   val只使用routes_validation；
    #   普通Driving与LG共享完全相同的路线集合。
    # false：保留原SimLingo的use_town13划分逻辑。
    use_official_route_split: bool = False

    skip_first_n_frames: int = 10
    pred_len: int = 11  # including the current time step
    hist_len: int = 1  # including the current time step
    hist_len_commentary: int = 5  # including the current time step

    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5

    use_safety_flag: bool = False

    num_route_points: int = 20

    route_as: str = 'target_point_command'  # none, target_point, command, target_point_command
    use_lmdrive_commands: bool = True


@dataclass
class DrivingDatasetConfig:
    _target_: str = "simlingo_training.dataloader.dataset_driving.Data_Driving"

    # 普通Driving语言监督模式。
    # four_questions：与LG统一使用Q1-Q4语言
    # commentary：原SimLingo commentary
    # qa：原DriveLM风格QA
    # mixed：原SimLingo随机混合语言任务
    driving_language_mode: str = "four_questions"

    # 普通Driving四通道结构化未来世界标签配置。
    driving_use_future_interaction_grid: bool = False
    driving_future_interaction_grid_folder: str = "driving_future_interaction_grids"

    # 普通Driving主要关键参与者空间投影监督配置。
    driving_use_participant_spatial_attention_supervision: bool = False
    driving_participant_attention_label_folder: str = (
        "driving_expert_conditioned_actor_selection"
    )


@dataclass
class DreamerDatasetConfig:
    _target_: str = "simlingo_training.dataloader.dataset_dreamer.Data_Dreamer"

    # ------------------------------------------------------------------
    # Optional LG supervision adapter.
    # These switches are ignored by the original Data_Dreamer class. They
    # become active only when _target_ is changed to an LG dataset class.
    # ------------------------------------------------------------------
    use_lg_supervision: bool = False
    lg_label_folder: str = "language_grounded_waypoints"

    # LG四通道结构化未来世界标签配置。
    lg_use_future_interaction_grid: bool = False
    lg_future_interaction_grid_folder: str = "future_interaction_grids"

    # LG主要关键参与者视觉token级空间监督配置。
    lg_use_participant_spatial_attention_supervision: bool = False

    # LG对象移除后的真实反事实重规划轨迹监督。
    lg_use_counterfactual_supervision: bool = False
    lg_counterfactual_min_effect_m: float = 0.03

    # Main ablation switches.
    # True/True  : complete LG method (four-question language + LG waypoints)
    # False/True : LG-WP Only
    # True/False : LG language + original expert waypoints
    lg_use_language: bool = True
    lg_use_waypoints: bool = True

    # Language format: four_questions, random_question, or none.
    lg_language_mode: str = "four_questions"
    lg_question_keys: Tuple[str, str, str, str] = (
        "attention",
        "motion_constraint",
        "driving_response",
        "future_motion",
    )
    lg_include_navigation_conditioning: bool = True

    # Label validation and filtering.
    lg_require_risk_label_valid: bool = True
    lg_require_four_questions: bool = True
    lg_skip_expert_fallback: bool = True
    lg_max_abs_waypoint_m: float = 100.0
    lg_print_filter_summary: bool = True

    # Match Data_Dreamer's official training/validation town split. This is
    # applied only inside Data_LG and does not alter the normal driving source.
    lg_match_dreamer_split: bool = True


@dataclass
class QADatasetConfig:
    _target_: str = "simlingo_training.dataloader.dataset_eval_qa_comm.Data_Eval"


@dataclass
class InstEvalDatasetConfig:
    _target_: str = "simlingo_training.dataloader.dataset_eval_dreamer.Eval_Dreamer"


@dataclass
class DrivingDataModuleConfig:
    base_dataset: DatasetBaseConfig

    driving_dataset: Optional[DrivingDatasetConfig] = field(default_factory=DrivingDatasetConfig)
    dreamer_dataset: Optional[DreamerDatasetConfig] = field(default_factory=DreamerDatasetConfig)
    qa_dataset: Optional[QADatasetConfig] = field(default_factory=QADatasetConfig)
    insteval_dataset: Optional[InstEvalDatasetConfig] = field(default_factory=InstEvalDatasetConfig)

    batch_size: int = 16
    num_workers: int = 10

    train_partitions: Optional[Dict[str, float]] = None
    train_partitions_dreamer: Optional[Dict[str, float]] = None
    use_global_img: bool = False

    _target_: str = "simlingo_training.dataloader.datamodule.DataModule"


@dataclass
class ValidationLoggingConfig:
    # Master switch. False preserves SimLingo's original validation behavior.
    enabled: bool = False

    # Saved relative to the Hydra run directory, e.g. outputs/<run>/validation_logs.
    output_dir: str = "./validation_logs"

    # Metric and file switches.
    separate_by_source: bool = True
    save_csv: bool = True
    save_epoch_json: bool = True
    save_per_sample_jsonl: bool = False
    save_visualizations: bool = True
    visualization_samples_per_source: int = 4

    # Geometric validation settings. LG and SimLingo use one waypoint every 0.25 s.
    log_per_horizon_error: bool = True
    waypoint_interval_s: float = 0.25
    stop_speed_threshold_mps: float = 0.5

    # Avoid writing the short Lightning sanity-validation pass.
    skip_sanity_check: bool = True
    print_epoch_summary: bool = True


@dataclass
class TrainConfig:
    model: DrivingModelConfig
    data_module: Any
    validation_logging: ValidationLoggingConfig = field(default_factory=ValidationLoggingConfig)

    seed: int = 42
    gpus: int = 8

    resume: bool = False
    resume_path: Optional[str] = None

    debug: bool = False
    overfit: int = 0
    fp16_loss_scale: float = 32.0  # 0.0 means dynamic loss scaling, only used with deepspeed

    enable_wandb: bool = True
    wandb_project: Optional[str] = "simlingo"
    if debug:
        wandb_name: Optional[str] = "debug"
        gpus: int = 1
    else:
        name: Optional[str] = 'test'
        wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"

    max_epochs: int = 20
    precision: str = "16-mixed"
    strategy: str = "deepspeed_stage_2"  # deepspeed_stage_2, ddp
    val_every_n_epochs: int = 1

    checkpoint: Optional[str] = None

    # Checkpointing remains disabled by default, matching the current repository.
    # Enable it from an experiment YAML when a model is needed for evaluation.
    enable_checkpointing: bool = False
    checkpoint_dir: str = "./checkpoints"
    checkpoint_filename: str = "{epoch:03d}"
    checkpoint_monitor: Optional[str] = "val/loss"
    checkpoint_mode: str = "min"
    checkpoint_save_top_k: int = 1
    checkpoint_save_last: bool = True


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="train_base", node=TrainConfig)
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="data_module/base_dataset", name="dataset", node=DatasetBaseConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model/vision_model", name="vlm", node=VLMEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)


register_configs()
