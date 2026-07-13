"""
Generates a dataset on a single local machine (single GPU).
- Keeps the core data collection logic: generate start_files -> launch CARLA -> run leaderboard evaluator -> save data/results.
- Removes all SLURM-specific logic (sbatch/squeue/scancel/partition/max_num_jobs).
- Runs routes sequentially, and retries crashed/failed routes up to N times.
Best run inside tmux.
"""

from datetime import datetime
import os
import subprocess
import time
import glob
import json
from pathlib import Path
import random
import re

DEBUG = False   # For debug

def make_bash(code_dir, route_file_number, agent_name, route_file, ckeckpoint_endpoint, save_pth, seed, carla_root, town, repetition):
    print(f"[Debug] Writing ...")
    save_slurm = save_pth.replace("data/", "slurm/")
    jobfile = f"{save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh"
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)
    print(f"[Debug] Bash file = {jobfile}")

    run_command = "python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${FREE_WORLD_PORT} \
        --traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} --routes=${ROUTES} --repetitions=${REPETITIONS} \
            --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} \
                --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=600"

    qsub_template = f"""#!/bin/bash
set -e

export SCENARIO_RUNNER_ROOT={code_dir}/scenario_runner_autopilot
export LEADERBOARD_ROOT={code_dir}/leaderboard_autopilot

# carla
export CARLA_ROOT={carla_root}
export CARLA_SERVER={carla_root}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:leaderboard_autopilot
export PYTHONPATH=$PYTHONPATH:scenario_runner_autopilot

export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT={agent_name}
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES={route_file}
export TOWN={town}
export REPETITION={repetition}
export TM_SEED={seed}

export CHECKPOINT_ENDPOINT={ckeckpoint_endpoint}
export TEAM_CONFIG={route_file}
export RESUME=1
export DATAGEN=1
export SAVE_PATH={save_pth}

echo "Start python"

export FREE_STREAMING_PORT=$1
export FREE_WORLD_PORT=$2
export TM_PORT=$3

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"

# --- Ensure CARLA is killed on exit (Ctrl+C / error / normal exit) ---
cleanup() {{
  echo "[CLEANUP] killing CARLA pid=$CARLA_PID ..."
  kill -INT $CARLA_PID 2>/dev/null || true
  sleep 2
  kill -KILL $CARLA_PID 2>/dev/null || true
}}
trap cleanup EXIT
# -------------------------------------------------------------------------------

# Without GUI
# bash {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &
# CARLA_PID=$!

# With GUI
bash {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -quality-level=Low -carla-streaming-port=$FREE_STREAMING_PORT &
CARLA_PID=$!

# Give CARLA time to boot, No data is recorded during this period.
sleep 20

{run_command}
"""
    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)

    os.chmod(jobfile, 0o755)
    print(f"[Debug] Success.")
    return jobfile


def pick_free_port(start, end):
    """
    Pick a free TCP port in [start, end]. Uses 'ss' (Linux).
    """
    cmd = (
        f"comm -23 <(seq {start} {end} | sort) "
        f"<(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) "
        f"| shuf | head -n 1"
    )
    out = subprocess.check_output(["bash", "-lc", cmd]).decode("utf-8").strip()
    if not out:
        raise RuntimeError(f"No free port found in range [{start},{end}]")
    return int(out)


def evaluate_result_ok(result_file):
    """
    Mimic original 'finished or need_resubmit' logic.
    Returns True if result indicates a successful run; False otherwise.
    """
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            evaluation_data = json.load(f)
        progress = evaluation_data["_checkpoint"]["progress"]
        if len(progress) < 2 or progress[0] < progress[1]:
            return False

        for record in evaluation_data["_checkpoint"]["records"]:
            if record["scores"]["score_route"] <= 1e-11:
                return False
            status = record.get("status", "")
            if status in [
                "Failed - Agent couldn't be set up",
                "Failed",
                "Failed - Simulation crashed",
                "Failed - Agent crashed",
            ]:
                return False
        return True
    except Exception:
        return False


def run_one_route(start_sh, log_dir, route_id, repetition, timeout_hours=4):
    """
    Run start_sh locally: bash start_sh streaming_port world_port tm_port
    Capture stdout/stderr to log files.

    Always cleanup CARLA by killing world_port in a finally block,
    even when KeyboardInterrupt happens (Ctrl+C).
    """
    os.makedirs(log_dir, exist_ok=True)
    out_log = os.path.join(log_dir, f"local_out_{route_id}_Rep{repetition}.log")
    err_log = os.path.join(log_dir, f"local_err_{route_id}_Rep{repetition}.log")

    streaming_port = pick_free_port(10000, 10400)
    world_port = pick_free_port(20000, 20400)
    tm_port = pick_free_port(30000, 30400)

    cmd = ["bash", start_sh, str(streaming_port), str(world_port), str(tm_port)]
    print(f"[RUN] {' '.join(cmd)}")
    print(f"[LOG] out={out_log}")
    print(f"[LOG] err={err_log}")

    try:
        with open(out_log, "w", encoding="utf-8") as fo, open(err_log, "w", encoding="utf-8") as fe:
            try:
                subprocess.run(
                    cmd,
                    stdout=fo,
                    stderr=fe,
                    check=False,
                    timeout=timeout_hours * 3600,
                )
            except subprocess.TimeoutExpired:
                fe.write(f"\n[TIMEOUT] exceeded {timeout_hours} hours\n")
            except KeyboardInterrupt:
                fe.write("\n[INTERRUPT] KeyboardInterrupt (Ctrl+C)\n")
                # re-raise so main can stop immediately
                raise
    finally:
        # best-effort cleanup: kill any process still holding the CARLA world port
        subprocess.run(
            ["bash", "-lc", f"fuser -k {world_port}/tcp >/dev/null 2>&1 || true"],
            check=False,
        )

    return out_log, err_log


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n[Debug] Initializing...")
    repetitions = 1               # 重复收集的结束
    repetition_start = 0          # 重复收集的开始

    code_root   = r"/home/kemove/ll/simlingo"                         # 项目根目录
    carla_root  = "/home/kemove/ll/simlingo/carla0915"                # Carla根位置
    root_folder = r"database/"                                        # 这是数据集存放的根目录

    date = datetime.today().strftime("%Y_%m_%d")                      # 时间格式:年_月_日
    dataset_name = "simlingo_v2_" + date                              # 文件夹命名格式(示例): simlingo_v2_2026_02_27
    data_save_directory = root_folder + dataset_name                  # 收集数据集存放的位置: database/simlingo_v2_2026_02_27/

    route_folder = f"{code_root}/data/simlingo"                       # 收集路线文件(.xml)存放的位置
    if not DEBUG:
        routes = glob.glob(f"{route_folder}/**/*balanced*/*.xml", recursive=True)     # 只对含balanced的目录当前层下的 .xml 文件
        routes_lb1 = glob.glob(f"{route_folder}/**/*lb1*/**/*.xml", recursive=True)   # 对lb1_split文件夹下的所有路线进行收集
        routes = routes + routes_lb1                                                  # 用于收集数据的全部路线
        # Set a random seed of 42 to shuffle the order of routes in a reproducible manner.
        random.seed(42)
        random.shuffle(routes)
    else:  # true for debug
        print("[Debug] There's only one route, which makes debugging easier ......")
        routes = ["/home/liulei/ll/simlingo/data/simlingo/training_1_scenario/routes_training/random_weather_seed_1_balanced_150/1.xml"]

    if len(routes) == 0:
        raise RuntimeError(f"No route xml found under: {route_folder}")

    seed_counter = 1000000 * repetition_start - 1
    num_routes = len(routes)
    print(f"[Debug] Found {num_routes} routes.")

    # retry settings
    max_retries = 3
    timeout_hours = 4

    for repetition in range(repetition_start, repetitions):
        for idx, route in enumerate(routes, start=1):
            seed_counter += 1

            # 从 route 字符串路径中解析出属于哪个 Town。
            try:
                town = re.search(r"Town(\d+)", route).group(0)
            except Exception:
                if "validation" in route:
                    town = "Town13"
                elif "training" in route:
                    town = "Town12"
                else:
                    print(f"[SKIP] Town not found in route: {route}")
                    continue

            scenario_type = route.split("/")[-5:-1]
            scenario_type = "/".join(scenario_type)
            routefile_number = Path(route).stem  # e.g., 22_0

            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"
            save_path = f"{code_root}/{data_save_directory}/data/{scenario_type}"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            print(f"[Debug] Ckpt save path = {ckpt_endpoint}")
            print(f"[Debug] Data save path = {save_path}")

            agent = f"{code_root}/team_code/data_agent.py"
            print(f"[Debug] Agent = {agent}")

            # logs folder (keep the same slurm-like structure for compatibility)
            save_slurm = save_path.replace("data/", "slurm/")
            logs_dir = f"{save_slurm}/run_files/logs"
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            print(f"[Debug] Slurm logs dir = {logs_dir}")
            print(f"[Debug] Success.")

            print("\n[Debug] Bash file ...")
            print(f"[Debug] Make bash for {route}\n"
                   "        # 项目地址\n"
                  f"        code_root = {code_root}\n"
                   "        # 数据采集路线的数量\n"
                  f"        routefile_number = {routefile_number}\n"
                   "        # 代理\n"
                  f"        agent = {agent}\n"
                   "        # 当前数据采集路线\n"
                  f"        route = {route}\n"
                   "        # 路线运行结果文件\n"
                  f"        ckpt_endpoint = {ckpt_endpoint}\n"
                   "        # 采集数据的存放目录！！！\n"
                  f"        save_path = {save_path}\n"
                   "        # 计数器\n"
                  f"        seed_counter = {seed_counter}\n"
                   "        # carla目录位置\n"
                  f"        carla_root = {carla_root}\n"
                   "        # 当前路线来自哪个城镇\n"
                  f"        town = {town}\n"
                   "        # 当前重复采集次数/需要重复采集总次数\n"
                  f"        repetition = {repetition+1} / {repetitions}"
                  )
            start_sh = make_bash(
                code_root, routefile_number, agent, route,
                ckpt_endpoint, save_path, seed_counter, carla_root, town, repetition
            )

            print(f"\n[{idx}/{num_routes}] Route={routefile_number} Town={town}")
            ok = False

            for attempt in range(max_retries):
                if attempt > 0:
                    stamp = int(time.time())
                    arch = os.path.join(logs_dir, f"retry_{routefile_number}_t{stamp}")
                    os.makedirs(arch, exist_ok=True)
                    subprocess.run(
                        ["bash", "-lc", f"mv {logs_dir}/local_*_{routefile_number}_Rep{repetition}.log {arch}/ 2>/dev/null || true"],
                        check=False
                    )

                try:
                    out_log, err_log = run_one_route(
                        start_sh, logs_dir, routefile_number, repetition, timeout_hours=timeout_hours
                    )
                except KeyboardInterrupt:
                    print("\n[STOP] KeyboardInterrupt received. Exiting now.")
                    raise

                ok = evaluate_result_ok(ckpt_endpoint)
                print(f"[RESULT] ok={ok} result_file={ckpt_endpoint}")
                if ok:
                    break
                else:
                    print(f"[RETRY] {routefile_number} attempt {attempt+1}/{max_retries} failed. Logs: {out_log}, {err_log}")
                    time.sleep(5)

            if not ok:
                print(f"[GIVE UP] {routefile_number} failed after {max_retries} attempts. Continue next route.")