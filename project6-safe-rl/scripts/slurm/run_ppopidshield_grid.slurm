#!/bin/bash
#SBATCH --job-name=ppopidshield-grid
#SBATCH --account=jfnielse0
#SBATCH --partition=spgpu
#SBATCH --qos=normal
#SBATCH --time=16:00:00
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
unset PYTHONPATH   # prevent ~/.local/python3.9 paths from leaking into the venv

SITE_PKGS="$VENV_DIR/lib/python3.10/site-packages"
export LD_PRELOAD="$SITE_PKGS/nvidia/cuda_cupti/lib/libcupti.so.12"

export NO_COLOR=1

mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

nvidia-smi || true
srun -u python -u scripts/run_ppopidshield_grid.py \
    --envs SafetyPointGoal2-v0 \
    --seeds 0 \
    --config configs/ppopidshield/config.yaml
