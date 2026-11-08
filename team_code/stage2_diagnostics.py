# -*- coding: utf-8 -*-
"""Read-only diagnostics for SimLingo-LG Stage-2 Bench2Drive inference.

This module never changes predicted tensors, model parameters, or CARLA controls.
The saved C1/C2 arrays are *model predictions*, not ground-truth actor tracks.
No geometric collision or TTC estimate is made without a verified BEV transform.
"""
import json
from pathlib import Path

import numpy as np
import torch


class Stage2Diagnostics:
    def __init__(self, output_dir, enabled=False, every=1, grid_every=5):
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self.grid_every = max(1, int(grid_every))
        self.root = Path(output_dir)
        self.handle = None
        self.refinement = None
        self.hook = None
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / 'grids').mkdir(exist_ok=True)
            self.handle = (self.root / 'frames.jsonl').open('a', encoding='utf-8', buffering=1)
            self._write_manifest()
            print('[Stage2Diag] Saving to {}'.format(self.root), flush=True)

    def _write_manifest(self):
        manifest = {
            'format': 'simlingo_stage2_diagnostics_v2',
            'scalar_file': 'frames.jsonl',
            'grid_folder': 'grids',
            'scalar_every': self.every,
            'grid_every': self.grid_every,
            'coordinates': 'model ego-relative XY (metres), no map projection verified',
            'world_channels': ['C1 predicted ego temporal occupancy',
                               'C2 predicted primary-actor temporal occupancy'],
            'world_shape': '[T,H,W] (typically T=10, H=W=128)',
            'world_grid_warning': 'Predicted world maps do NOT by themselves establish actual collision or the identity of a visible actor. Metric calibration and GT tracks are not available here.',
            'c1_c2_overlap_warning': 'Decoder C1/C2 overlap is only a probability-map proxy; it is NOT final ego-waypoints/actor physical overlap.',
            'controls': 'returned_control is applied by leaderboard; ukf_stored_control may differ during warmup',
            'checkpoint_note': 'No weights or predictions are modified by this recorder.',
            'pid_debug': 'desired_speed_mps is the exact ORIGINAL control_pid speed target; no recomputation or changes.',
            'trajectory_delta_raw': 'un-gated refinement_head output; final ego = initial + gate*delta, before finite-precision rounding.',
            'actor_ground_truth': 'Not available from a SENSORS agent; use separate CARLA observer to record ground-truth actor tracks.',
        }
        (self.root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    def hook_refinement(self, _module, _args, output):
        # Model forward returns a fresh dict each frame. Keep only the last result.
        if isinstance(output, dict):
            self.refinement = output

    def attach(self, model):
        if self.enabled:
            self.hook = model.trajectory_refinement.register_forward_hook(self.hook_refinement)

    @staticmethod
    def _numpy(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().float().cpu().numpy()
        return np.asarray(tensor, dtype=np.float32)

    @classmethod
    def _waypoints(cls, tensor):
        array = cls._numpy(tensor)
        if array is None:
            return None
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2 or array.shape[1] < 2:
            return None
        return array[:, :2]

    @staticmethod
    def _scalar(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return float(tensor.detach().float().reshape(-1)[0].cpu().item())
        return float(np.asarray(tensor).reshape(-1)[0])

    @staticmethod
    def _control(control):
        if control is None:
            return None
        return {'steer': float(control.steer), 'throttle': float(control.throttle),
                'brake': float(control.brake)}

    @staticmethod
    def _description(points):
        if points is None or len(points) == 0:
            return None
        return {'first_xy': points[0].astype(float).tolist(),
                'last_xy': points[-1].astype(float).tolist(),
                'max_abs_lateral_m': float(np.abs(points[:, 1]).max()),
                'first_step_m': float(np.linalg.norm(points[0])),
                'total_step_displacement_m': float(np.linalg.norm(np.diff(
                    np.vstack((np.zeros((1, 2), np.float32), points)), axis=0), axis=1).sum())}

    def record(self, frame, timestamp, speed_mps, model, route, initial, final,
               returned_control, ukf_stored_control=None, pid_debug=None, language=None):
        if not self.enabled or frame % self.every:
            return
        init_xy = self._waypoints(initial)
        final_xy = self._waypoints(final)
        route_xy = self._waypoints(route)
        out = self.refinement or {}
        record = {
            'frame': int(frame), 'timestamp': float(timestamp),
            'ego_speed_mps': float(speed_mps),
            'returned_control': self._control(returned_control),
            'ukf_stored_control': self._control(ukf_stored_control),
            'actor_gate': self._scalar(out.get('interaction_gate')),
            'actor_logit': self._scalar(out.get('critical_actor_logits')),
            'route': self._description(route_xy),
            'initial_ego': self._description(init_xy),
            'final_ego': self._description(final_xy),
            'language': str(language) if language is not None else None,
            'pid_debug': dict(pid_debug) if isinstance(pid_debug, dict) else None,
        }
        if init_xy is not None and final_xy is not None and init_xy.shape == final_xy.shape:
            difference = np.linalg.norm(final_xy - init_xy, axis=1)
            record['refinement_mean_displacement_m'] = float(difference.mean())
            record['refinement_max_displacement_m'] = float(difference.max())
        else:
            record['refinement_mean_displacement_m'] = None
            record['refinement_max_displacement_m'] = None

        # The raw correction is computed *before* sigmoid actor gating. This
        # distinguishes a weak refinement head from an overly restrictive gate.
        raw_delta = self._waypoints(out.get('trajectory_delta'))
        record['trajectory_delta_raw'] = self._description(raw_delta)
        if raw_delta is not None and len(raw_delta):
            raw_norm = np.linalg.norm(raw_delta, axis=1)
            record['raw_delta_mean_norm_m'] = float(raw_norm.mean())
            record['raw_delta_max_norm_m'] = float(raw_norm.max())
        else:
            record['raw_delta_mean_norm_m'] = None
            record['raw_delta_max_norm_m'] = None
        gate_value = record['actor_gate']
        if gate_value is not None and raw_delta is not None:
            gated_norm = np.linalg.norm(raw_delta * gate_value, axis=1)
            record['gate_times_delta_mean_norm_m'] = float(gated_norm.mean())
            record['gate_times_delta_max_norm_m'] = float(gated_norm.max())
        else:
            record['gate_times_delta_mean_norm_m'] = None
            record['gate_times_delta_max_norm_m'] = None

        logits = getattr(model, 'future_interaction_temporal_logits', None)
        if isinstance(logits, torch.Tensor) and logits.ndim == 5 and logits.shape[1] == 2:
            # [B,2,T,H,W] -> [2,T,H,W]. These are DECODER predictions.
            probabilities = torch.sigmoid(logits.detach().float()[0])
            c1, c2 = probabilities[0], probabilities[1]
            record['predicted_c2_max_by_t'] = [float(x) for x in c2.amax(dim=(-1,-2)).cpu().tolist()]
            record['predicted_c2_mean_by_t'] = [float(x) for x in c2.mean(dim=(-1,-2)).cpu().tolist()]
            record['predicted_c2_fraction_gt05_by_t'] = [float(x) for x in (c2 > 0.5).float().mean(dim=(-1,-2)).cpu().tolist()]
            record['predicted_c1_c2_overlap_proxy_by_t'] = [float(x) for x in (c1*c2).mean(dim=(-1,-2)).cpu().tolist()]
            if frame % self.grid_every == 0:
                arr = probabilities.cpu().numpy().astype(np.float16)
                np.savez_compressed(self.root / 'grids' / ('frame_%06d.npz' % frame),
                                    c1_probability=arr[0], c2_probability=arr[1],
                                    route=route_xy if route_xy is not None else np.zeros((0,2), np.float32),
                                    initial_ego=init_xy if init_xy is not None else np.zeros((0,2), np.float32),
                                    final_ego=final_xy if final_xy is not None else np.zeros((0,2), np.float32),
                                    actor_gate=np.array(record['actor_gate'] if record['actor_gate'] is not None else np.nan),
                                    frame=np.array(frame, dtype=np.int64))
                record['grid_saved'] = 'grids/frame_%06d.npz' % frame
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.refinement = None

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None
