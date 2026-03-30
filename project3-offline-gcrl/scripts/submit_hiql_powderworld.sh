#!/bin/bash
# Submit 4 individual single-seed jobs for powderworld-easy-play concurrently.
# Each job runs one seed via --single-seed; --visual-enabled activates the
# ImpalaSmall encoder and discrete AWR actor required for powderworld environments.
# Usage: bash submit_hiql_powderworld.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/run_hiql.slurm"

for seed in 0 1 2 3; do
  echo "Submitting: powderworld-easy-play-v0 seed=$seed"
  sbatch --job-name="hiql-powderworld-easy-s${seed}" \
         --mem=32G \
         --export=ALL,ENV="powderworld",TASK="play",DSIZE="easy",TRAIN_STEPS=1000000,SEED_ID=$seed,VISUAL_ENABLED=1 \
         "$SLURM_SCRIPT"
done
