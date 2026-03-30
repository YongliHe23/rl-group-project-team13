#!/bin/bash
# Submit 4 parallel jobs for powderworld-easy-play, one per seed.
# Usage: bash submit_gciql_powderworld.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/run_gciql.slurm"

for seed in 0 1 2 3; do
  echo "Submitting: powderworld-easy-play-v0 seed=$seed"
  sbatch --job-name="gciql-powderworld-easy-s${seed}" \
         --mem=32G \
         --export=ALL,ENV="powderworld",TASK="play",DSIZE="easy",TRAIN_STEPS=1000000,SEED_ID=$seed \
         "$SLURM_SCRIPT"
done
