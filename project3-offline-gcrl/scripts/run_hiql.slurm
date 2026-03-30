#!/bin/bash
#SBATCH --job-name=hiql
#SBATCH --account=jfnielse0
#SBATCH --partition=spgpu
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/yonglihe/ece567/project/runs/hiql/%j/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yonglihe@umich.edu

set -euo pipefail

WORKDIR=/home/yonglihe/ece567/project
SCRIPT=$WORKDIR/rl-group-project-team13/project3-offline-gcrl/baselines/hiql/main_hiql.py
VENV_DIR=$WORKDIR/venv

module purge
module load python/3.10
module load cuda/12.6.3
source "$VENV_DIR/bin/activate"

RUN_DIR=$WORKDIR/runs/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

nvidia-smi || true
# ENV, TASK, DSIZE can be passed via: sbatch --export=ALL,ENV=antmaze,TASK=navigate,DSIZE=large run_hiql.slurm
ENV=${ENV:-pointmaze}
TASK=${TASK:-navigate}
DSIZE=${DSIZE:-medium}
TRAIN_STEPS=${TRAIN_STEPS:-}     # empty = use script default (500K)
SEED_ID=${SEED_ID:-}             # single seed via --single-seed; empty = no override
SEEDS=${SEEDS:-}                 # explicit seed list via --seeds, e.g. "0 1 2 3"
PARALLEL_SEEDS=${PARALLEL_SEEDS:-}  # non-empty = add --parallel-seeds flag
VISUAL_ENABLED=${VISUAL_ENABLED:-}  # non-empty = add --visual-enabled flag (powderworld)

RESULTS_DIR=$WORKDIR/rl-group-project-team13/project3-offline-gcrl/results/hiql

EXTRA_ARGS=""
[ -n "$TRAIN_STEPS" ]    && EXTRA_ARGS="$EXTRA_ARGS --train-step $TRAIN_STEPS"
[ -n "$SEED_ID" ]        && EXTRA_ARGS="$EXTRA_ARGS --single-seed $SEED_ID"
[ -n "$SEEDS" ]          && EXTRA_ARGS="$EXTRA_ARGS --seeds $SEEDS"
[ -n "$PARALLEL_SEEDS" ] && EXTRA_ARGS="$EXTRA_ARGS --parallel-seeds"
[ -n "$VISUAL_ENABLED" ] && EXTRA_ARGS="$EXTRA_ARGS --visual-enabled"

TQDM_DISABLE=1 srun -u python -u "$SCRIPT" --env "$ENV" --task "$TASK" --dsize "$DSIZE" --output-dir "$RESULTS_DIR" $EXTRA_ARGS
