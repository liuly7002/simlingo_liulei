"""Local validation metric recording for SimLingo.

The callback is deliberately optional. When disabled, SimLingo keeps its
original validation step and output behavior. When enabled, the training entry
point installs a validation-step wrapper that exposes the already-computed
teacher-forced predictions to this callback, so validation is not run twice.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig

from simlingo_training.models.utils import summarise_losses


LG_PROMPT_MARKER = "Answer the four driving questions in order"


def install_validation_output_capture(model: pl.LightningModule) -> None:
    """Expose validation predictions without adding a second model forward pass.

    The original ``DrivingModel.validation_step`` summarizes losses and discards
    the prediction tensors returned by ``DrivingAdaptor.compute_loss``. This
    wrapper reproduces the original logging behavior, while also returning
    detached per-sample losses and predictions for local metric calculation.
    """

    if getattr(model, "_local_validation_capture_installed", False):
        return

    def validation_step_with_outputs(
        self: pl.LightningModule,
        batch: Any,
        _batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        del dataloader_idx

        loss_dict, pred_labels = self.forward_loss(batch, per_sample=True)
        output = summarise_losses(loss_dict)

        # Preserve the repository's original validation logging.
        self.log_training_output(output, "val")
        self.log(
            "val/loss",
            output.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        detached_loss_dict = {
            key: (values.detach(), counts.detach())
            for key, (values, counts) in loss_dict.items()
        }
        detached_predictions = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in pred_labels.items()
        }

        return {
            "loss": output.loss,
            "outputs": output,
            "validation_loss_dict": detached_loss_dict,
            "validation_predictions": detached_predictions,
        }

    model.validation_step = MethodType(validation_step_with_outputs, model)
    model._local_validation_capture_installed = True


def _decode_run_ids(encoded: Any) -> List[str]:
    if isinstance(encoded, torch.Tensor):
        array = encoded.detach().cpu().numpy()
        return [row.tobytes().decode("utf-8").rstrip("\0") for row in array]
    return [str(item) for item in encoded]


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return _to_float(values.float().mean())


def _per_sample_loss(values: torch.Tensor, counts: torch.Tensor, index: int) -> float:
    sample_values = values[index].float()
    sample_counts = counts[index].float()
    denominator = sample_counts.sum()
    if _to_float(denominator) <= 0.0:
        return float("nan")
    return _to_float(sample_values.sum() / denominator)


def _trajectory_speeds(waypoints: torch.Tensor, interval_s: float) -> torch.Tensor:
    origin = torch.zeros_like(waypoints[:1])
    points = torch.cat((origin, waypoints), dim=0)
    segment_lengths = torch.linalg.norm(points[1:] - points[:-1], dim=-1)
    return segment_lengths / max(float(interval_s), 1e-6)


def _detect_source(prompt: str, answer: str) -> str:
    prompt = str(prompt)
    answer = str(answer)

    if LG_PROMPT_MARKER in prompt or answer.lstrip().startswith("A1:"):
        return "lg"
    if (
        answer.startswith("Following the given instruction. Waypoints:")
        or "<SAFETY>" in prompt
        or "<INSTRUCTION_FOLLOWING>" in prompt
    ):
        return "dreamer"
    return "driving"


def _merge_stat_dicts(stat_dicts: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for stats in stat_dicts:
        for source, source_stats in stats.items():
            target = merged.setdefault(source, {"num_samples": 0, "metrics": {}})
            target["num_samples"] += int(source_stats.get("num_samples", 0))
            for metric, pair in source_stats.get("metrics", {}).items():
                metric_target = target["metrics"].setdefault(metric, [0.0, 0])
                metric_target[0] += float(pair[0])
                metric_target[1] += int(pair[1])
    return merged


def _gather_objects(value: Any) -> List[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered: List[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


class LocalValidationMetricsCallback(pl.Callback):
    """Save validation metrics, summaries, and fixed examples to local files."""

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str = "./validation_logs",
        separate_by_source: bool = True,
        save_csv: bool = True,
        save_epoch_json: bool = True,
        save_per_sample_jsonl: bool = False,
        save_visualizations: bool = True,
        visualization_samples_per_source: int = 4,
        log_per_horizon_error: bool = True,
        waypoint_interval_s: float = 0.25,
        stop_speed_threshold_mps: float = 0.5,
        skip_sanity_check: bool = True,
        print_epoch_summary: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.separate_by_source = bool(separate_by_source)
        self.save_csv = bool(save_csv)
        self.save_epoch_json = bool(save_epoch_json)
        self.save_per_sample_jsonl = bool(save_per_sample_jsonl)
        self.save_visualizations = bool(save_visualizations)
        self.visualization_samples_per_source = max(int(visualization_samples_per_source), 0)
        self.log_per_horizon_error = bool(log_per_horizon_error)
        self.waypoint_interval_s = float(waypoint_interval_s)
        self.stop_speed_threshold_mps = float(stop_speed_threshold_mps)
        self.skip_sanity_check = bool(skip_sanity_check)
        self.print_epoch_summary = bool(print_epoch_summary)

        self._stats: Dict[str, Any] = {}
        self._per_sample_rows: List[Dict[str, Any]] = []
        self._visual_records: Dict[str, Dict[str, Any]] = {}
        self._fixed_paths: Dict[str, List[str]] = {}

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        del trainer, pl_module, stage
        if not self.enabled:
            return
        if not self.output_dir.is_absolute():
            try:
                run_dir = Path(HydraConfig.get().runtime.output_dir)
            except Exception:  # Hydra is not initialized in standalone tests.
                run_dir = Path.cwd()
            self.output_dir = run_dir / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        settings = {
            "teacher_forced_validation": True,
            "separate_by_source": self.separate_by_source,
            "save_csv": self.save_csv,
            "save_epoch_json": self.save_epoch_json,
            "save_per_sample_jsonl": self.save_per_sample_jsonl,
            "save_visualizations": self.save_visualizations,
            "visualization_samples_per_source": self.visualization_samples_per_source,
            "log_per_horizon_error": self.log_per_horizon_error,
            "waypoint_interval_s": self.waypoint_interval_s,
            "stop_speed_threshold_mps": self.stop_speed_threshold_mps,
        }
        if not (self.output_dir / "settings.json").exists():
            (self.output_dir / "settings.json").write_text(
                json.dumps(settings, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._stats = {}
        self._per_sample_rows = []
        self._visual_records = {}

    def _update_metric(self, source: str, metric: str, value: float) -> None:
        if not math.isfinite(value):
            return
        source_stats = self._stats.setdefault(source, {"num_samples": 0, "metrics": {}})
        pair = source_stats["metrics"].setdefault(metric, [0.0, 0])
        pair[0] += float(value)
        pair[1] += 1

    def _register_sample(self, source: str) -> None:
        self._stats.setdefault(source, {"num_samples": 0, "metrics": {}})["num_samples"] += 1

    def _metric_targets(self, source: str) -> List[str]:
        if self.separate_by_source:
            return ["all", source]
        return ["all"]

    def _build_sample_row(
        self,
        index: int,
        source: str,
        run_id: str,
        prompt: str,
        answer: str,
        loss_dict: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
        predictions: Mapping[str, Any],
        batch: Any,
        epoch: int,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        row: Dict[str, Any] = {
            "epoch": int(epoch),
            "source": source,
            "measurement_path": run_id,
        }

        total_loss = 0.0
        total_loss_valid = False
        for loss_name, (values, counts) in loss_dict.items():
            value = _per_sample_loss(values, counts, index)
            row[loss_name] = value
            if math.isfinite(value):
                total_loss += value
                total_loss_valid = True
        row["total_loss"] = total_loss if total_loss_valid else float("nan")

        speed_prediction = predictions.get("speed_wps_prediction")
        route_prediction = predictions.get("route_prediction")
        visual_record: Optional[Dict[str, Any]] = None

        if isinstance(speed_prediction, torch.Tensor):
            pred_wps = speed_prediction[index].detach().float().cpu()
            gt_wps = batch.driving_label.waypoints[index].detach().float().cpu()
            common = min(pred_wps.shape[0], gt_wps.shape[0])
            pred_wps = pred_wps[:common]
            gt_wps = gt_wps[:common]

            if common > 0:
                displacement = torch.linalg.norm(pred_wps - gt_wps, dim=-1)
                row["waypoint_ade_m"] = _safe_mean(displacement)
                row["waypoint_fde_m"] = _to_float(displacement[-1])
                row["waypoint_longitudinal_mae_m"] = _safe_mean(
                    torch.abs(pred_wps[:, 0] - gt_wps[:, 0])
                )
                row["waypoint_lateral_mae_m"] = _safe_mean(
                    torch.abs(pred_wps[:, 1] - gt_wps[:, 1])
                )

                if self.log_per_horizon_error:
                    for horizon_index, error in enumerate(displacement, start=1):
                        horizon = horizon_index * self.waypoint_interval_s
                        row[f"waypoint_de_{horizon:.2f}s_m"] = _to_float(error)

                pred_speeds = _trajectory_speeds(pred_wps, self.waypoint_interval_s)
                gt_speeds = _trajectory_speeds(gt_wps, self.waypoint_interval_s)
                row["pred_mean_speed_mps"] = _safe_mean(pred_speeds)
                row["gt_mean_speed_mps"] = _safe_mean(gt_speeds)
                row["mean_speed_mae_mps"] = abs(
                    row["pred_mean_speed_mps"] - row["gt_mean_speed_mps"]
                )
                row["pred_final_speed_mps"] = _to_float(pred_speeds[-1])
                row["gt_final_speed_mps"] = _to_float(gt_speeds[-1])
                row["final_speed_mae_mps"] = abs(
                    row["pred_final_speed_mps"] - row["gt_final_speed_mps"]
                )

                pred_stop = row["pred_final_speed_mps"] < self.stop_speed_threshold_mps
                gt_stop = row["gt_final_speed_mps"] < self.stop_speed_threshold_mps
                row["pred_stop"] = float(pred_stop)
                row["gt_stop"] = float(gt_stop)
                row["false_stop"] = float(pred_stop and not gt_stop)
                row["missed_stop"] = float(gt_stop and not pred_stop)

                visual_record = {
                    "source": source,
                    "measurement_path": run_id,
                    "prompt": prompt,
                    "answer": answer,
                    "pred_waypoints": pred_wps.numpy(),
                    "gt_waypoints": gt_wps.numpy(),
                    "waypoint_ade_m": row["waypoint_ade_m"],
                    "waypoint_fde_m": row["waypoint_fde_m"],
                }

        if isinstance(route_prediction, torch.Tensor):
            pred_route = route_prediction[index].detach().float().cpu()
            gt_route = batch.driving_label.path[index].detach().float().cpu()
            common = min(pred_route.shape[0], gt_route.shape[0])
            pred_route = pred_route[:common]
            gt_route = gt_route[:common]

            if common > 0:
                route_displacement = torch.linalg.norm(pred_route - gt_route, dim=-1)
                row["route_ade_m"] = _safe_mean(route_displacement)
                row["route_fde_m"] = _to_float(route_displacement[-1])

                if visual_record is not None:
                    pred_wps_tensor = torch.from_numpy(visual_record["pred_waypoints"])
                    point_to_route = torch.cdist(
                        pred_wps_tensor.unsqueeze(0),
                        gt_route.unsqueeze(0),
                    ).squeeze(0)
                    row["waypoint_to_route_error_m"] = _safe_mean(
                        point_to_route.min(dim=-1).values
                    )
                    visual_record["pred_route"] = pred_route.numpy()
                    visual_record["gt_route"] = gt_route.numpy()
                    visual_record["waypoint_to_route_error_m"] = row[
                        "waypoint_to_route_error_m"
                    ]

        return row, visual_record

    def _should_keep_visual(self, source: str, run_id: str) -> bool:
        if not self.save_visualizations or self.visualization_samples_per_source <= 0:
            return False

        fixed = self._fixed_paths.setdefault(source, [])
        if run_id in fixed:
            return True
        if len(fixed) < self.visualization_samples_per_source:
            fixed.append(run_id)
            return True
        return False

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Mapping[str, Any]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module, batch_idx, dataloader_idx
        if not self.enabled:
            return
        if self.skip_sanity_check and trainer.sanity_checking:
            return
        if not outputs:
            return

        loss_dict = outputs.get("validation_loss_dict")
        predictions = outputs.get("validation_predictions")
        if not isinstance(loss_dict, Mapping) or not isinstance(predictions, Mapping):
            raise RuntimeError(
                "Local validation logging is enabled, but validation outputs were "
                "not captured. Ensure install_validation_output_capture(model) is called."
            )

        prompts = list(batch.driving_input.prompt.language_string)
        answers = list(batch.driving_label.answer.language_string)
        run_ids = _decode_run_ids(batch.run_id)
        batch_size = len(run_ids)

        for index in range(batch_size):
            prompt = str(prompts[index])
            answer = str(answers[index])
            source = _detect_source(prompt, answer)
            row, visual_record = self._build_sample_row(
                index=index,
                source=source,
                run_id=run_ids[index],
                prompt=prompt,
                answer=answer,
                loss_dict=loss_dict,
                predictions=predictions,
                batch=batch,
                epoch=int(trainer.current_epoch),
            )

            for target in self._metric_targets(source):
                self._register_sample(target)
                for metric, value in row.items():
                    if metric in {"epoch", "source", "measurement_path"}:
                        continue
                    if isinstance(value, (int, float, np.number)):
                        self._update_metric(target, metric, float(value))

            if self.save_per_sample_jsonl:
                self._per_sample_rows.append(row)

            if (
                trainer.is_global_zero
                and visual_record is not None
                and self._should_keep_visual(source, run_ids[index])
            ):
                self._visual_records[run_ids[index]] = visual_record

    @staticmethod
    def _summarize(stats: Mapping[str, Any], epoch: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for source in sorted(stats.keys()):
            source_stats = stats[source]
            row: Dict[str, Any] = {
                "epoch": int(epoch),
                "source": source,
                "num_samples": int(source_stats.get("num_samples", 0)),
            }
            for metric, (value_sum, value_count) in sorted(
                source_stats.get("metrics", {}).items()
            ):
                if int(value_count) > 0:
                    row[metric] = float(value_sum) / int(value_count)
            rows.append(row)
        return rows

    def _write_metrics_csv(self, summary_rows: List[Dict[str, Any]]) -> None:
        path = self.output_dir / "metrics.csv"
        existing_rows: List[Dict[str, Any]] = []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as file_obj:
                existing_rows = list(csv.DictReader(file_obj))

        all_rows: List[Dict[str, Any]] = existing_rows + summary_rows
        fieldnames = sorted(
            {key for row in all_rows for key in row.keys()},
            key=lambda key: (key not in {"epoch", "source", "num_samples"}, key),
        )
        with path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    def _write_visualizations(self, epoch: int) -> None:
        if not self.save_visualizations or not self._visual_records:
            return

        epoch_dir = self.output_dir / "visualizations" / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        fixed_path_file = self.output_dir / "fixed_visualization_samples.json"
        fixed_path_file.write_text(
            json.dumps(self._fixed_paths, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for record_index, record in enumerate(self._visual_records.values()):
            source = record["source"]
            pred_wps = np.asarray(record["pred_waypoints"])
            gt_wps = np.asarray(record["gt_waypoints"])

            fig = plt.figure(figsize=(7.0, 7.0))
            axis = fig.add_subplot(1, 1, 1)
            axis.plot(pred_wps[:, 1], pred_wps[:, 0], marker="o", label="Pred waypoint")
            axis.plot(gt_wps[:, 1], gt_wps[:, 0], marker="x", label="GT waypoint")

            if "pred_route" in record:
                pred_route = np.asarray(record["pred_route"])
                gt_route = np.asarray(record["gt_route"])
                axis.plot(pred_route[:, 1], pred_route[:, 0], linestyle="--", label="Pred route")
                axis.plot(gt_route[:, 1], gt_route[:, 0], linestyle=":", label="GT route")

            axis.set_xlabel("Lateral position (m)")
            axis.set_ylabel("Longitudinal position (m)")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True)
            axis.legend()
            axis.set_title(
                f"{source} | ADE={record['waypoint_ade_m']:.3f} m | "
                f"FDE={record['waypoint_fde_m']:.3f} m"
            )
            fig.tight_layout()

            stem = f"{source}_{record_index:03d}"
            fig.savefig(epoch_dir / f"{stem}.png", dpi=160)
            plt.close(fig)

            metadata = {
                key: value
                for key, value in record.items()
                if not isinstance(value, np.ndarray)
            }
            (epoch_dir / f"{stem}.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        if not self.enabled:
            return
        if self.skip_sanity_check and trainer.sanity_checking:
            return

        gathered_stats = _gather_objects(self._stats)
        merged_stats = _merge_stat_dicts(gathered_stats)

        gathered_rows: List[Dict[str, Any]] = []
        if self.save_per_sample_jsonl:
            for rows in _gather_objects(self._per_sample_rows):
                gathered_rows.extend(rows)

        if not trainer.is_global_zero:
            return

        epoch = int(trainer.current_epoch)
        summary_rows = self._summarize(merged_stats, epoch)

        if self.save_csv:
            self._write_metrics_csv(summary_rows)

        epoch_payload = {
            "epoch": epoch,
            "teacher_forced_validation": True,
            "summary": summary_rows,
        }
        if self.save_epoch_json:
            epoch_dir = self.output_dir / "epochs"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / f"epoch_{epoch:04d}.json").write_text(
                json.dumps(epoch_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (self.output_dir / "latest.json").write_text(
                json.dumps(epoch_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if self.save_per_sample_jsonl:
            sample_dir = self.output_dir / "per_sample"
            sample_dir.mkdir(parents=True, exist_ok=True)
            with (sample_dir / f"epoch_{epoch:04d}.jsonl").open(
                "w", encoding="utf-8"
            ) as file_obj:
                for row in gathered_rows:
                    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._write_visualizations(epoch)

        if self.print_epoch_summary:
            print(f"[Local validation] saved epoch {epoch} to {self.output_dir}")
            for row in summary_rows:
                print(
                    "[Local validation] "
                    f"source={row['source']} samples={row['num_samples']} "
                    f"ADE={row.get('waypoint_ade_m', float('nan')):.4f} "
                    f"FDE={row.get('waypoint_fde_m', float('nan')):.4f} "
                    f"speed_MAE={row.get('mean_speed_mae_mps', float('nan')):.4f}"
                )
