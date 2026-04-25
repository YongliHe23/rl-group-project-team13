#!/bin/bash
#SBATCH --job-name=eval-comparison
#SBATCH --account=jfnielse0
#SBATCH --partition=spgpu
#SBATCH --qos=normal
#SBATCH --time=01:00:00
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
unset PYTHONPATH

SITE_PKGS="$VENV_DIR/lib/python3.10/site-packages"
export LD_PRELOAD="$SITE_PKGS/nvidia/cuda_cupti/lib/libcupti.so.12"

export NO_COLOR=1

mkdir -p "$WORKDIR/logs" "$WORKDIR/results" "$WORKDIR/plots"
cd "$WORKDIR"

nvidia-smi || true

echo "=== Evaluating PPOLag ==="
srun -u python -u scripts/eval_actor.py \
    --checkpoint "runs/fromJingda1M/ppolag1M/torch_save/epoch-33.pt" \
    --env_id SafetyPointGoal2-v0 \
    --seeds 1 2 3 4 5 --episodes 20 \
    --label PPOLag \
    --out results/ppolag/fromJingda/eval_epoch34_100ep.csv

echo "=== Evaluating PPOLagAda ==="
srun -u python -u scripts/eval_actor.py \
    --checkpoint "runs/fromJingda1M/ada1M/torch_save/epoch-34.pt" \
    --env_id SafetyPointGoal2-v0 \
    --seeds 1 2 3 4 5 --episodes 20 \
    --label PPOLagAda \
    --out results/ppolagada/fromJingda/eval_hybrid_sigmoid_epoch34_100ep.csv

echo "=== Plotting comparison ==="
srun -u python -u scripts/plot_eval_comparison.py \
    --csvs results/ppolag/fromJingda/eval_epoch34_100ep.csv results/ppolagada/fromJingda/eval_hybrid_sigmoid_epoch34_100ep.csv \
    --labels "PPO-Lag" "PPO-Lag-Ada" \
    --out plots/actor_rollout/fromJingda/ppolag1M_vs_ppolagadaHybridSigmoid1M_100ep.png
