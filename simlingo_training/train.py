import os
import hydra

from omegaconf import OmegaConf
import torch
import wandb

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from transformers import AutoProcessor

from simlingo_training.utils.logging_project import setup_logging, sync_wandb
from simlingo_training.config import TrainConfig
from simlingo_training.callbacks.visualise import VisualiseCallback
from simlingo_training.callbacks.local_validation import (
    LocalValidationMetricsCallback,
    install_validation_output_capture,
)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"

    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"

    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant,
        trust_remote_code=True,
    )
    model_type_name = cfg.model.vision_model.variant.split('/')[-1]
    cache_dir = None

    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )

    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False,
    )

    # Optional local validation capture. The wrapper reuses the original
    # teacher-forced validation forward pass and does not run the model twice.
    if bool(cfg.validation_logging.enabled):
        install_validation_output_capture(model)

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

    os.environ["WANDB_DISABLE_CODE"] = "True"

    if cfg.overfit > 0:
        overfit = cfg.overfit

    setup_logging(cfg)

    resume_path = cfg.resume_path
    resume_wandb = False

    if resume_path is not None and not os.path.exists(resume_path):
        resume_wandb = True
    elif resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    if not (resume_path is not None and os.path.exists(resume_path) and cfg.resume):
        resume_path = None

    loggers = []
    wandblogger = WandbLogger(
        project=cfg.wandb_project,
        id=cfg.wandb_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=resume_wandb,
    )
    wandblogger.watch(model)
    loggers.append(wandblogger)

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2,
            loss_scale=cfg.fp16_loss_scale,
            logging_batch_size_per_gpu=cfg.data_module.batch_size,
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)
    callbacks = [
        model_summary,
        VisualiseCallback(interval=1000, val_interval=1000),
    ]

    if bool(cfg.validation_logging.enabled):
        validation_logging_cfg = OmegaConf.to_container(
            cfg.validation_logging,
            resolve=True,
        )
        callbacks.append(LocalValidationMetricsCallback(**validation_logging_cfg))

    # Optional checkpointing. With the switch disabled, this reproduces the
    # current repository behavior exactly. Enable it for closed-loop evaluation.
    if bool(cfg.enable_checkpointing):
        monitor = cfg.checkpoint_monitor
        if monitor is not None and str(monitor).strip().lower() in {"", "none", "null"}:
            monitor = None

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            filename=cfg.checkpoint_filename,
            monitor=monitor,
            mode=cfg.checkpoint_mode,
            save_top_k=int(cfg.checkpoint_save_top_k),
            save_last=bool(cfg.checkpoint_save_last),
            every_n_epochs=int(cfg.val_every_n_epochs),
            auto_insert_metric_name=False,
        )
        callbacks.insert(0, checkpoint_callback)

    if not cfg.debug:
        callbacks.append(lr_monitor)

    overfit = 0

    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            devices=cfg.gpus,
            enable_checkpointing=bool(cfg.enable_checkpointing),
            gradient_clip_val=0.3,
            logger=loggers,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
        )
    else:
        raise ValueError("cfg.gpus must be at least 1 for the current training entry point")

    trainer.fit(model, data_module, ckpt_path=resume_path)
    wandb.finish()


if __name__ == "__main__":
    main()
