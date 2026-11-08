# -*- coding: utf-8 -*-


"""
本地操作指令：

codna activate simlingo


python start_eval_simlingo.py \
  --route bench2drive_04 \
  --workers 1 \
  --gpu-ids 0 \
  --graphics-ids 0 \
  --show-carla-window \
  --window-width 1280 \
  --window-height 720 \
  --checkpoint "/home/kemove/ll/A论文/VLA相关资料/完整网络训练结果/outputs/2026_11_04_09_48_33_simlingo_lg_stage2_seed1/checkpoints/009.ckpt/pytorch_model.pt/pytorch_model.bin" \
  --model-dir "/home/kemove/ll/InternVL2-1B" \
  --carla-root "/home/kemove/ll/simlingo/carla0915"


# 开启 KV Cache
SIMLINGO_USE_KV_CACHE=1 python start_eval_simlingo.py \
  --route bench2drive_04 \
  --workers 1 \
  --gpu-ids 0 \
  --graphics-ids 0 \
  --show-carla-window \
  --checkpoint "/home/kemove/ll/A论文/VLA相关资料/完整网络训练结果/outputs/2026_11_04_09_48_33_simlingo_lg_stage2_seed1/checkpoints/009.ckpt/pytorch_model.pt/pytorch_model.bin" \
  --model-dir "/home/kemove/ll/InternVL2-1B" \
  --carla-root "/home/kemove/ll/simlingo/carla0915"


# 不开启 KV Cache
  SIMLINGO_USE_KV_CACHE=0 python start_eval_simlingo.py \
  --route bench2drive_04 \
  --workers 1 \
  --gpu-ids 0 \
  --graphics-ids 0 \
  --show-carla-window \
  --checkpoint "/home/kemove/ll/A论文/VLA相关资料/完整网络训练结果/outputs/2026_11_04_09_48_33_simlingo_lg_stage2_seed1/checkpoints/009.ckpt/pytorch_model.pt/pytorch_model.bin" \
  --model-dir "/home/kemove/ll/InternVL2-1B" \
  --carla-root "/home/kemove/ll/simlingo/carla0915"


python start_eval_simlingo.py \
  --route bench2drive_06 \
  --workers 1 \
  --gpu-ids 0 \
  --graphics-ids 0 \
  --show-carla-window \
  --trajectory-viz \
  --trajectory-every 3 \
  --trajectory-save \
  --checkpoint "/home/kemove/ll/A论文/VLA相关资料/完整网络训练结果/outputs/2026_11_04_09_48_33_simlingo_lg_stage2_seed1/checkpoints/009.ckpt/pytorch_model.pt/pytorch_model.bin" \
  --model-dir "/home/kemove/ll/InternVL2-1B" \
  --carla-root "/home/kemove/ll/simlingo/carla0915"


python start_eval_simlingo.py \
  --route bench2drive_06 \
  --workers 1 \
  --gpu-ids 0 \
  --graphics-ids 0 \
  --show-carla-window \
  --trajectory-viz \
  --trajectory-every 5 \
  --diagnostics \
  --diagnostic-every 1 \
  --diagnostic-grid-every 5 \
  --checkpoint "/home/kemove/ll/A论文/VLA相关资料/完整网络训练结果/outputs/2026_11_04_09_48_33_simlingo_lg_stage2_seed1/checkpoints/009.ckpt/pytorch_model.pt/pytorch_model.bin" \
  --model-dir "/home/kemove/ll/InternVL2-1B" \
  --carla-root "/home/kemove/ll/simlingo/carla0915" \
  --results-root "/home/kemove/ll/simlingo_liulei/eval_results/Bench2Drive_diagnostics"


列出所有的bench2drive的路线以便查看
python start_eval_simlingo.py --list-routes --routes-dir /home/kemove/ll/simlingo_liulei/leaderboard/data/bench2drive_split

"""



import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
RUN_NAME = '2026_11_04_09_48_33_simlingo_lg_stage2_seed1'

configs = [
    {
        'agent': 'simlingo_lg_stage2',
        'checkpoint': str(REPO_ROOT / 'outputs' / RUN_NAME /
                          'checkpoints/009.ckpt/pytorch_model.pt/pytorch_model.bin'),
        'benchmark': 'bench2drive',
        'route_path': str(REPO_ROOT / 'leaderboard/data/bench2drive_split'),
        'seeds': [1],
        'tries': 2,
        'out_root': str(REPO_ROOT / 'eval_results/Bench2Drive'),
        'carla_root': os.environ.get('CARLA_ROOT', str(REPO_ROOT.parent / 'carla0915')),
        'repo_root': str(REPO_ROOT),
        'agent_file': str(REPO_ROOT / 'team_code/agent_simlingo.py'),
        'team_code': 'team_code',
        'agent_config': 'not_used',
    }
]


def result_status(path):
    """Inspect actual Bench2Drive result JSON rather than only process exit code."""
    if not path.is_file():
        return 'missing'
    try:
        with path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        ckpt = data['_checkpoint']
        progress = ckpt['progress']
        records = ckpt['records']
        if len(progress) < 2 or int(progress[1]) <= 0 or int(progress[0]) < int(progress[1]):
            return 'partial'
        if not records:
            return 'empty'
        if len(records) != int(progress[1]):
            return 'partial'
        # A finished route may legitimately have status 'Failed' due to driving
        # behavior (e.g. route deviation, blocked ego, timeout). That is a VALID
        # evaluated result, not an infrastructure failure to be retried.
        if data.get('entry_status') != 'Finished' or data.get('eligible') is not True:
            return 'incomplete'
        infrastructure_failures = (
            'Simulation crashed',
            'Agent crashed',
            "Agent couldn't be set up",
            "Agent's sensors were invalid",
        )
        for rec in records:
            status = str(rec.get('status', ''))
            if not status or status == 'Started':
                return 'invalid'
            if any(reason in status for reason in infrastructure_failures):
                return 'failed'
        return 'complete'
    except (OSError, ValueError, KeyError, TypeError):
        return 'invalid'


def path_preflight(cfg):
    root = Path(cfg['repo_root'])
    paths = (
        Path(cfg['checkpoint']),
        Path(cfg['carla_root']) / 'CarlaUE4.sh',
        Path(cfg['agent_file']),
        root / 'team_code/config_simlingo.py',
        root / 'Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py',
        root / 'leaderboard/data/weather.xml',
        Path(cfg['route_path']),
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError('Required evaluation input missing: {}'.format(path))
    if cfg['benchmark'].lower() != 'bench2drive':
        raise NotImplementedError('Only bench2drive is supported')
    # The original Agent discovers the Hydra config by walking upward from
    # the checkpoint. A plain .bin file alone is not enough on another PC.
    if not any((ancestor / '.hydra' / 'config.yaml').is_file()
               for ancestor in Path(cfg['checkpoint']).resolve().parents):
        raise FileNotFoundError(
            'Missing .hydra/config.yaml above the checkpoint. Copy the original '
            'outputs/<run>/.hydra/config.yaml beside the copied checkpoint tree.'
        )
    model_dir = cfg.get('model_dir')
    if model_dir:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError('Local InternVL2-1B directory missing: {}'.format(model_dir))
        if not (model_dir / 'config.json').is_file():
            raise FileNotFoundError('InternVL2-1B config.json missing: {}'.format(model_dir))


def port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            return False
        return True


def build_environment(cfg, route, viz_root, model_gpu):
    root = Path(cfg['repo_root'])
    carla = Path(cfg['carla_root'])
    leaderboard = root / 'Bench2Drive/leaderboard'
    scenario = root / 'Bench2Drive/scenario_runner'
    env = dict(os.environ)
    env.update({
        'CARLA_ROOT': str(carla),
        'WORK_DIR': str(root),
        'LEADERBOARD_ROOT': str(leaderboard),
        'SCENARIO_RUNNER_ROOT': str(scenario),
        'ROUTES': str(route),
        'SAVE_PATH': str(viz_root),
        'CUDA_VISIBLE_DEVICES': str(model_gpu),
        'SIMLINGO_MODEL_DIR': cfg.get('model_dir', ''),
        'SIMLINGO_TRAJ_VIZ': '1' if cfg.get('trajectory_viz', False) else '0',
        'SIMLINGO_TRAJ_SAVE': '1' if cfg.get('trajectory_save', False) else '0',
        'SIMLINGO_TRAJ_EVERY': str(cfg.get('trajectory_every', 5)),
        'SIMLINGO_STAGE2_DIAG': '1' if cfg.get('diagnostics', False) else '0',
        'SIMLINGO_STAGE2_DIAG_EVERY': str(cfg.get('diagnostic_every', 1)),
        'SIMLINGO_STAGE2_DIAG_GRID_EVERY': str(cfg.get('diagnostic_grid_every', 5)),
        'OMP_NUM_THREADS': env.get('OMP_NUM_THREADS', '8'),
        'OPENBLAS_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    })
    # Prepend the benchmark's own leaderboard/scenario_runner, never the
    # unrelated root-level competition packages.
    old = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = os.pathsep.join([str(root), str(root / 'team_code'),
                                      str(leaderboard), str(scenario),
                                      str(carla / 'PythonAPI/carla'), old])
    return env


def prepare_carla_window_launcher(cfg, launcher_dir):
    """Create a local CARLA GUI launcher without changing Bench2Drive code.

    The original evaluator always adds -RenderOffScreen to CarlaUE4.sh.
    For local windowed mode, point only that evaluator's CARLA_ROOT at this
    small launcher, which drops the flag and forwards all other arguments to
    the REAL, unmodified CARLA installation. The benchmark and its scoring
    logic remain identical to the default evaluation path.
    """
    launcher_dir = Path(launcher_dir).resolve()
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher = launcher_dir / 'CarlaUE4.sh'
    true_root = Path(cfg['carla_root']).resolve()
    true_launcher = true_root / 'CarlaUE4.sh'
    if not true_launcher.is_file():
        raise FileNotFoundError('Original CARLA launch script missing: {}'.format(true_launcher))

    width = int(cfg['window_width'])
    height = int(cfg['window_height'])
    content = '''#!/usr/bin/env bash
set -e
# The Bench2Drive evaluator sends -RenderOffScreen unconditionally.
# Remove ONLY that rendering flag; preserve RPC port, GPU index and other flags.
args=()
for arg in "$@"; do
    if [ "$arg" != "-RenderOffScreen" ]; then
        args+=("$arg")
    fi
done
export CARLA_ROOT={real_root}
exec {real_script} "${{args[@]}}" -windowed -ResX={width} -ResY={height}
'''.format(
        real_root=shlex.quote(str(true_root)),
        real_script=shlex.quote(str(true_launcher)),
        width=width,
        height=height,
    )
    launcher.write_text(content, encoding='utf-8')
    launcher.chmod(0o755)
    return launcher_dir


def available_routes(cfg):
    route_dir = Path(cfg['route_path'])
    if not route_dir.is_dir():
        raise FileNotFoundError('Route XML directory missing: {}'.format(route_dir))
    return sorted(route_dir.glob('*.xml'))


def assemble_jobs(cfg, limit, named_route, route_index=None, route_range=''):
    route_dir = Path(cfg['route_path'])
    routes = available_routes(cfg)
    if not routes:
        raise ValueError('No route XMLs found under {}. Copy the Bench2Drive route XML files.'.format(route_dir))
    if named_route:
        requested = Path(named_route).name
        routes = [p for p in routes if p.name == requested or p.stem == requested]
        if not routes:
            raise ValueError('Selected route {!r} not found in {}'.format(named_route, route_dir))
    if route_index is not None:
        all_routes = available_routes(cfg)
        if not (0 <= route_index < len(all_routes)):
            raise ValueError('--route-index must be between 0 and {}'.format(len(all_routes) - 1))
        routes = [all_routes[route_index]]
    if route_range:
        try:
            start, end = (int(x.strip()) for x in route_range.split(':'))
        except (ValueError, TypeError):
            raise ValueError('--route-range must be START:END, zero-based, END exclusive')
        all_routes = available_routes(cfg)
        if not (0 <= start < end <= len(all_routes)):
            raise ValueError('Invalid --route-range: {} for {} routes'.format(route_range, len(all_routes)))
        routes = all_routes[start:end]
    if limit:
        routes = routes[:limit]
    jobs = []
    for seed in cfg['seeds']:
        base = Path(cfg['out_root']) / cfg['agent'] / cfg['benchmark'] / str(seed)
        for folder in ('run', 'res', 'out', 'err', 'viz'):
            (base / folder).mkdir(parents=True, exist_ok=True)
        for route in routes:
            name = route.stem
            jobs.append({
                'cfg': cfg,
                'route': route,
                'name': name,
                'seed': seed,
                'base': base,
                'result_file': base / 'res' / (name + '_res.json'),
                'out_file': base / 'out' / (name + '_out.log'),
                'err_file': base / 'err' / (name + '_err.log'),
                'viz_path': base / 'viz' / name,
                'remaining': int(cfg['tries']),
            })
    return jobs


def spawn(job, slot, gpu_ids, graphics_ids, retry_index):
    cfg = job['cfg']
    gpu = gpu_ids[slot]
    graphics = graphics_ids[slot]
    # Separate local port pools per worker to prevent CARLA and TM collisions.
    rpc = cfg['rpc_port'] + slot * 100
    tm = cfg['tm_port'] + slot * 100
    if not port_available(rpc) or not port_available(tm):
        raise RuntimeError('Required CARLA ports busy: {}, {}'.format(rpc, tm))
    env = build_environment(cfg, job['route'], job['viz_path'], gpu)
    if cfg.get('show_carla_window', False):
        # Only alter how CARLA itself is launched: evaluator/score/sync unchanged.
        window_root = prepare_carla_window_launcher(
            cfg, job['base'] / 'run' / 'carla_window_launcher'
        )
        env['CARLA_ROOT'] = str(window_root)
    cmd = [
        sys.executable, '-u',
        str(Path(cfg['repo_root']) / 'Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py'),
        '--routes={}'.format(job['route']),
        '--repetitions=1',
        '--track=SENSORS',
        '--checkpoint={}'.format(job['result_file']),
        '--timeout=600',
        '--agent={}'.format(cfg['agent_file']),
        '--agent-config={}'.format(cfg['checkpoint']),
        '--traffic-manager-seed={}'.format(job['seed']),
        '--port={}'.format(rpc),
        '--traffic-manager-port={}'.format(tm),
        '--gpu-rank={}'.format(graphics),
    ]
    job['viz_path'].mkdir(parents=True, exist_ok=True)
    out = job['out_file'].open('a', encoding='utf-8')
    err = job['err_file'].open('a', encoding='utf-8')
    out.write('\n===== attempt {} | slot {} | model GPU {} | graphics {} =====\n'.format(
        retry_index, slot, gpu, graphics))
    out.write(' '.join(cmd) + '\n')
    if cfg.get('show_carla_window', False):
        out.write('[CARLA window] Enabled; launcher {} filters -RenderOffScreen '
                  'and starts the original CARLA in windowed mode ({}x{}).\n'.format(
                      env['CARLA_ROOT'], cfg['window_width'], cfg['window_height']))
    out.flush()
    try:
        proc = subprocess.Popen(cmd, cwd=cfg['repo_root'], env=env,
                                stdout=out, stderr=err, start_new_session=True)
    except Exception:
        out.close()
        err.close()
        raise
    return {'proc': proc, 'job': job, 'out': out, 'err': err,
            'slot': slot, 'start': time.monotonic()}


def terminate(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=0,
                        help='0: all selected routes; N: first N of selection')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--route', default='',
                           help='One route XML filename or stem, e.g. bench2drive_00')
    selection.add_argument('--route-index', type=int, default=None,
                           help='One route by zero-based index from --list-routes')
    selection.add_argument('--route-range', default='',
                           help='Run a partition START:END (0-based, end exclusive)')
    parser.add_argument('--list-routes', action='store_true',
                        help='Print available XML routes and exit, without CARLA/model')
    parser.add_argument('--routes-dir', default=str(REPO_ROOT / 'leaderboard/data/bench2drive_split'),
                        help='Directory containing bench2drive_XX.xml route files')
    parser.add_argument('--checkpoint', default=os.environ.get('SIMLINGO_CHECKPOINT', configs[0]['checkpoint']),
                        help='Local Stage-2 pytorch_model.bin (keep .hydra/config.yaml in ancestor run dir)')
    parser.add_argument('--model-dir', default=os.environ.get('SIMLINGO_MODEL_DIR', ''),
                        help='Local complete InternVL2-1B Hugging Face model directory')
    parser.add_argument('--carla-root', default=os.environ.get('CARLA_ROOT', configs[0]['carla_root']),
                        help='Local CARLA 0.9.15 installation directory')
    parser.add_argument('--results-root', default=str(REPO_ROOT / 'eval_results/Bench2Drive'),
                        help='Root for the original Bench2Drive per-route res/out/err output folders')
    parser.add_argument('--seed', type=int, default=1,
                        help='Traffic-manager random seed; keep 1 to match AutoDL')
    parser.add_argument('--rpc-port', type=int, default=10000)
    parser.add_argument('--tm-port', type=int, default=30000)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--gpu-ids', default='0',
                        help='CUDA device indices per worker; single RTX 4080: 0')
    parser.add_argument('--graphics-ids', default='0',
                        help='Unreal Vulkan -graphicsadapter indices (may differ from CUDA)')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate even routes with complete JSON results')
    parser.add_argument('--max-job-seconds', type=int, default=0,
                        help='Optional process timeout; disabled by default')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print selected routes and configured paths without starting CARLA')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Save read-only Stage-2 gate/trajectory/control and C1/C2 occupancy diagnostics')
    parser.add_argument('--diagnostic-every', type=int, default=1,
                        help='Record scalar diagnostics every N ticks (default 1)')
    parser.add_argument('--diagnostic-grid-every', type=int, default=5,
                        help='Save C1/C2 temporal probability grids every N ticks (default 5)')
    parser.add_argument('--trajectory-viz', action='store_true',
                        help='Open live front-camera + BEV prediction window (read-only)')
    parser.add_argument('--trajectory-save', action='store_true',
                        help='Save visualized frames and prediction .npz data')
    parser.add_argument('--trajectory-every', type=int, default=5,
                        help='Draw/update every N simulation ticks (default 5)')
    parser.add_argument('--show-carla-window', '--carla-window',
                        action='store_true', dest='show_carla_window',
                        help='Display the real CARLA Unreal window instead of -RenderOffScreen')
    parser.add_argument('--window-width', type=int, default=1280,
                        help='CARLA window width when --show-carla-window is set (default 1280)')
    parser.add_argument('--window-height', type=int, default=720,
                        help='CARLA window height when --show-carla-window is set (default 720)')
    args = parser.parse_args()
    cfg = configs[0]
    cfg['route_path'] = str(Path(args.routes_dir).expanduser().resolve())
    cfg['checkpoint'] = str(Path(args.checkpoint).expanduser().resolve())
    cfg['carla_root'] = str(Path(args.carla_root).expanduser().resolve())
    cfg['model_dir'] = str(Path(args.model_dir).expanduser().resolve()) if args.model_dir else ''
    cfg['out_root'] = str(Path(args.results_root).expanduser().resolve())
    cfg['seeds'] = [args.seed]
    cfg['rpc_port'] = args.rpc_port
    cfg['tm_port'] = args.tm_port
    cfg['show_carla_window'] = bool(args.show_carla_window)
    cfg['trajectory_viz'] = bool(args.trajectory_viz)
    cfg['trajectory_save'] = bool(args.trajectory_save)
    cfg['trajectory_every'] = args.trajectory_every
    cfg['diagnostics'] = bool(args.diagnostics)
    cfg['diagnostic_every'] = args.diagnostic_every
    cfg['diagnostic_grid_every'] = args.diagnostic_grid_every
    if args.diagnostic_every < 1 or args.diagnostic_grid_every < 1:
        parser.error('--diagnostic-every and --diagnostic-grid-every must be >= 1')
    if args.trajectory_every < 1:
        parser.error('--trajectory-every must be >= 1')
    if args.trajectory_viz and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print('[TrajectoryViz] Warning: desktop display unavailable, GUI will be disabled.', flush=True)
    cfg['window_width'] = args.window_width
    cfg['window_height'] = args.window_height
    if args.window_width < 320 or args.window_height < 240:
        parser.error('--window-width must be >= 320 and --window-height >= 240')
    if args.show_carla_window and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print('[CARLA window] Warning: neither DISPLAY nor WAYLAND_DISPLAY is set. '
              'A local Linux desktop/X11 session is needed to show the Unreal window.',
              flush=True)
    if args.list_routes:
        routes = available_routes(cfg)
        print('Route XML files ({}): {}'.format(len(routes), cfg['route_path']))
        for index, route in enumerate(routes):
            print('{:03d} {}'.format(index, route.stem))
        return
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
    graphics_ids = [int(x.strip()) for x in args.graphics_ids.split(',') if x.strip()]
    if len(gpu_ids) < args.workers or len(graphics_ids) < args.workers:
        parser.error('--gpu-ids and --graphics-ids need at least --workers entries')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.rpc_port < 1024 or args.tm_port < 1024 or args.rpc_port == args.tm_port:
        parser.error('--rpc-port/--tm-port need distinct ports >= 1024')
    if any(x < 0 for x in gpu_ids + graphics_ids):
        parser.error('GPU/graphics adapter indices must be nonnegative')
    if args.workers == 2:
        print('[Notice] Two workers require two independent GPU configurations and '
              'may exceed a single 16GB card; use --workers 1 on RTX 4080.', flush=True)
    else:
        print('[Notice] Model and CARLA share the selected GPU on a single-card PC. '
              'Check peak VRAM before lengthy evaluations.', flush=True)
    if not cfg['model_dir']:
        print('[Notice] --model-dir not specified: original Hydra paths must exist locally.', flush=True)
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        print('[Notice] Running as root: CARLA Unreal may refuse root. Use a normal '
              'Linux account, or a previously configured non-root CARLA launcher.', flush=True)

    all_jobs = []
    for cfg in configs:
        path_preflight(cfg)
        all_jobs.extend(assemble_jobs(cfg, args.limit, args.route, args.route_index, args.route_range))
    if args.dry_run:
        print('Selected routes: {}'.format(len(all_jobs)))
        print('Repository: {}'.format(REPO_ROOT))
        print('Checkpoint: {}'.format(cfg['checkpoint']))
        print('InternVL2-1B: {}'.format(cfg['model_dir'] or '<original Hydra config path>'))
        print('CARLA root: {}'.format(cfg['carla_root']))
        print('Model GPU(s): {} | graphics adapter(s): {}'.format(gpu_ids, graphics_ids))
        print('Trajectory window={} save={} every={}'.format(
            cfg['trajectory_viz'], cfg['trajectory_save'], cfg['trajectory_every']))
        print('CARLA window: {} ({}x{})'.format(
            'enabled' if cfg['show_carla_window'] else 'headless',
            cfg['window_width'], cfg['window_height']))
        for job in all_jobs:
            print('{}  JSON={}  current={}'.format(
                job['name'], job['result_file'], result_status(job['result_file'])))
        return
    pending = []
    complete = 0
    for job in all_jobs:
        if not args.force and result_status(job['result_file']) == 'complete':
            complete += 1
        else:
            pending.append(job)
    print('Routes: {}, already complete: {}, queued: {}'.format(
        len(all_jobs), complete, len(pending)), flush=True)

    active = {}
    failures = []
    try:
        while pending or active:
            for slot, task in list(active.items()):
                proc, job = task['proc'], task['job']
                if args.max_job_seconds and (time.monotonic() - task['start']) > args.max_job_seconds:
                    print('[Timeout] {}'.format(job['route'].name), flush=True)
                    terminate(proc)
                code = proc.poll()
                if code is None:
                    continue
                task['out'].close()
                task['err'].close()
                del active[slot]
                status = result_status(job['result_file'])
                if code == 0 and status == 'complete':
                    complete += 1
                    print('[Complete] {} seed={} ({}/{})'.format(
                        job['name'], job['seed'], complete, len(all_jobs)), flush=True)
                elif job['remaining'] > 0:
                    print('[Retry] {} rc={} JSON={} remaining={}'.format(
                        job['name'], code, status, job['remaining']), flush=True)
                    pending.insert(0, job)
                else:
                    failures.append((job, code, status))
                    print('[Failed] {} rc={} JSON={}'.format(
                        job['name'], code, status), flush=True)
            for slot in range(args.workers):
                if not pending or slot in active:
                    continue
                job = pending.pop(0)
                job['remaining'] -= 1
                attempt = int(job['cfg']['tries']) - job['remaining']
                print('[Start] {} seed={} attempt={} modelGPU={} graphics={}'.format(
                    job['name'], job['seed'], attempt, gpu_ids[slot], graphics_ids[slot]), flush=True)
                try:
                    active[slot] = spawn(job, slot, gpu_ids, graphics_ids, attempt)
                except Exception as exc:
                    print('[Launch error] {}: {}'.format(job['name'], exc), flush=True)
                    if job['remaining'] > 0:
                        pending.insert(0, job)
                    else:
                        failures.append((job, -1, str(exc)))
                    time.sleep(5)
            if active:
                time.sleep(5)
    except KeyboardInterrupt:
        print('[Interrupted] Stopping local evaluator jobs...', flush=True)
        for task in active.values():
            terminate(task['proc'])
            task['out'].close()
            task['err'].close()
        raise SystemExit(130)

    print('Completed: {} / {}; failed: {}'.format(
        complete, len(all_jobs), len(failures)), flush=True)
    if failures:
        print('Failures (inspect logs under eval_results):', flush=True)
        for job, code, status in failures:
            print('  {} seed={} rc={} status={}'.format(job['name'], job['seed'], code, status))
        raise SystemExit(1)


if __name__ == '__main__':
    main()