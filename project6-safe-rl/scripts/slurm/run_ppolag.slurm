#!/bin/bash
#SBATCH --job-name=ppolag-pointgoal2
#SBATCH --account=jfnielse0
#SBATCH --partition=spgpu
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yonglihe@umich.edu

set -euo pipefail

WORKDIR=/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl
VENV_DIR=/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl/venv

module purge
module load python/3.10
module load cuda/12.6.3
source "$VENV_DIR/bin/activate"

mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

nvidia-smi || true
srun -u python -u scripts/train_ppolag.py \
    --config configs/ppo_lag/config_pointgoal2_1e7.yaml \
    --env_id SafetyPointGoal2-v0 \
    --seed 0
