export CARLA_ROOT=/home/kemove/ll/simlingo/carla0915
export WORK_DIR=/home/kemove/ll/simlingo_liulei
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}


export PYTHONPATH=$PYTHONPATH:${WORK_DIR}