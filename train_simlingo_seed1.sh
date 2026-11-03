# #!/bin/bash
# #SBATCH --job-name=slv2
# #SBATCH --nodes=1
# #SBATCH --time=3-00:00
# #SBATCH --gres=gpu:8
# #SBATCH --cpus-per-task=64
# #SBATCH --output=/YOUR_PATH/results/logs/slv2_%a_%A.out  # File to which STDOUT will be written
# #SBATCH --error=/YOUR_PATH/results/logs/slv2_%a_%A.err   # File to which STDERR will be written
# #SBATCH --partition=YOUR_PARTITION

# # print info about current job
# #scontrol show job $SLURM_JOB_ID
# if command -v scontrol >/dev/null 2>&1; then
#   scontrol show hostnames "$SLURM_JOB_NODELIST"
# fi

# #source ~/.bashrc
# #conda activate ~/miniconda3/envs/simlingo
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate simlingo

# pwd
# export CARLA_ROOT=/root/autodl-tmp/carla0915
# export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
# export WORK_DIR=/root/autodl-tmp/simlingo_liulei/
# export PYTHONPATH=$PYTHONPATH:${WORK_DIR}

# export MASTER_ADDR=localhost
# export NCCL_DEBUG=INFO

# export OMP_NUM_THREADS=64 # Limits pytorch to spawn at most num cpus cores threads
# export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # simlingo本身训练
# # WANDB__SERVICE_WAIT=300 python simlingo_training/train.py experiment=simlingo_seed1 data_module.batch_size=1 gpus=1 name=simlingo_seed1

# # SimLingo官方预训练模型
# SIMLINGO_PRETRAINED=/root/autodl-tmp/models/simlingo/checkpoint/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

# # lg替换dreamer之后的训练
# WANDB__SERVICE_WAIT=300 python simlingo_training/train.py experiment=simlingo_lg_seed1 data_module.batch_size=1 gpus=1 name=simlingo_lg_seed1 "checkpoint='${SIMLINGO_PRETRAINED}'"




#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate simlingo

export CARLA_ROOT=/root/autodl-tmp/carla0915
export WORK_DIR=/root/autodl-tmp/simlingo_liulei

export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${WORK_DIR}:${PYTHONPATH}"

export MASTER_ADDR=localhost
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

export WANDB_MODE=offline

cd "${WORK_DIR}"

# 官方 SimLingo 预训练权重
SIMLINGO_PRETRAINED=/root/autodl-tmp/models/simlingo/checkpoint/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

# 检查预训练模型是否存在
if [ ! -f "${SIMLINGO_PRETRAINED}" ]; then
    echo "ERROR: SimLingo pretrained checkpoint not found."
    echo "${SIMLINGO_PRETRAINED}"
    exit 1
fi

# 正式训练
WANDB__SERVICE_WAIT=300 python simlingo_training/train.py \
    experiment=simlingo_lg_seed1 \
    data_module.batch_size=1 \
    gpus=2 \
    name=simlingo_lg_stage2_seed1 \
    "checkpoint='${SIMLINGO_PRETRAINED}'" \
    enable_checkpointing=true