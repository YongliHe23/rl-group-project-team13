#!/bin/bash
#SBATCH --job-name=hiql-cse576
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=gpu
#SBATCH --qos=class
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail

WORKDIR=/home/stevengu/scratch/myproject0/stevengu/ece567
SCRIPT=$WORKDIR/main_hiql.py
VENV_DIR=/home/stevengu/scratch/myproject0/stevengu/ece567/venv

module purge
module load python/3.11
module load cuda
source "$VENV_DIR/bin/activate"

RUN_DIR=$WORKDIR/runs/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

nvidia-smi || true
srun -u python -u "$SCRIPT" --env antsoccer --task stitch --dsize medium

