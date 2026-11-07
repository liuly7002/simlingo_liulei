# -*- coding: utf-8 -*-
"""Local AutoDL evaluator for SimLingo-LG Stage-2 (Bench2Drive).

Replaces the official SLURM-only launch mechanism. Run ``--limit 1`` first;
then run without --limit for the 220-route evaluation. Uses the ORIGINAL
Bench2Drive evaluator, agent, controllers, evaluation JSON and route files.
No model parameters or benchmark rules are changed here.

Examples:
    python start_eval_simlingo.py --limit 1 --workers 1
    python start_eval_simlingo.py --workers 1
    python start_eval_simlingo.py --workers 2 --gpu-ids 0,1 --graphics-ids 1,0
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path('/root/autodl-tmp/simlingo_liulei')
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
        'carla_root': '/root/autodl-tmp/carla0915',
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


def assemble_jobs(cfg, limit, named_route):
    route_dir = Path(cfg['route_path'])
    routes = sorted(route_dir.glob('*.xml'))
    if named_route:
        routes = [p for p in routes if p.name == named_route or p.stem == named_route]
    if limit:
        routes = routes[:limit]
    if not routes:
        raise ValueError('No route XMLs found under {}'.format(route_dir))
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
    rpc = 10000 + slot * 100
    tm = 30000 + slot * 100
    if not port_available(rpc) or not port_available(tm):
        raise RuntimeError('Required CARLA ports busy: {}, {}'.format(rpc, tm))
    env = build_environment(cfg, job['route'], job['viz_path'], gpu)
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
                        help='Use 1 for the first smoke test; 0 for all routes')
    parser.add_argument('--route', default='', help='Optional one route filename or stem')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--gpu-ids', default='0,1',
                        help='PyTorch physical CUDA devices, indexed per worker')
    parser.add_argument('--graphics-ids', default='2,0',
                        help='CARLA -graphicsadapter indices, indexed per worker')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate even routes with complete JSON results')
    parser.add_argument('--max-job-seconds', type=int, default=0,
                        help='Optional process timeout; disabled by default')
    args = parser.parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
    graphics_ids = [int(x.strip()) for x in args.graphics_ids.split(',') if x.strip()]
    if len(gpu_ids) < args.workers or len(graphics_ids) < args.workers:
        parser.error('--gpu-ids and --graphics-ids need at least --workers entries')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    print('[Notice] Using workers=2 puts a CARLA render and a model on each 4090; '
          'check GPU memory before enabling parallelism.', flush=True)
    if args.workers == 2 and not args.force:
        print('[Notice] Start with --workers 1 --limit 1 to verify memory usage.', flush=True)

    all_jobs = []
    for cfg in configs:
        path_preflight(cfg)
        all_jobs.extend(assemble_jobs(cfg, args.limit, args.route))
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
