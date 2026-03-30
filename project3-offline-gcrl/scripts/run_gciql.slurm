#!/bin/bash
#SBATCH --job-name=gciql
#SBATCH --account=jfnielse0
#SBATCH --partition=spgpu
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/yonglihe/ece567/project/runs/%j/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yonglihe@umich.edu

set -euo pipefail

WORKDIR=/home/yonglihe/ece567/project
SCRIPT=$WORKDIR/rl-group-project-team13/project3-offline-gcrl/baselines/gciql/main_gciql.py
VENV_DIR=$WORKDIR/venv

module purge
module load python/3.10
module load cuda/12.6.3
source "$VENV_DIR/bin/activate"

RUN_DIR=$WORKDIR/runs/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

nvidia-smi || true
# ENV, TASK, DSIZE can be passed via: sbatch --export=ALL,ENV=antmaze,TASK=navigate,DSIZE=large run_gciql.slurm
ENV=${ENV:-pointmaze}
TASK=${TASK:-navigate}
DSIZE=${DSIZE:-medium}
TRAIN_STEPS=${TRAIN_STEPS:-}   # empty = use script default (500K)
SEED_ID=${SEED_ID:-}           # empty = run all 4 seeds sequentially
# disable tqdm for evaluation
RESULTS_DIR=$WORKDIR/rl-group-project-team13/project3-offline-gcrl/results/gciql
EXTRA_ARGS=""
[ -n "$TRAIN_STEPS" ] && EXTRA_ARGS="$EXTRA_ARGS --train-steps $TRAIN_STEPS"
[ -n "$SEED_ID" ]     && EXTRA_ARGS="$EXTRA_ARGS --seed-id $SEED_ID"
TQDM_DISABLE=1 srun -u python -u "$SCRIPT" --env "$ENV" --task "$TASK" --dsize "$DSIZE" --output-dir "$RESULTS_DIR" $EXTRA_ARGS
