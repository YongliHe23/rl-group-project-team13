#!/bin/bash
# Submit GCIVL jobs for cube-double and puzzle-3x3 with 1M training steps.
# Final score = avg of last 3 evals (800K, 900K, 1M).
# Usage: bash submit_gcivl_1M.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/run_gcivl.slurm"

jobs=(
  "cube   play double"
  "puzzle play 3x3"
)

for job in "${jobs[@]}"; do
  read -r env task dsize <<< "$job"
  dataset="${env}-${dsize}-${task}-v0"
  echo "Submitting: $dataset (1M steps)"
  sbatch --job-name="gcivl-${env}-${dsize}-${task}-1M" \
         --time=10:00:00 \
         --export=ALL,ENV="$env",TASK="$task",DSIZE="$dsize",TRAIN_STEPS=1000000 \
         "$SLURM_SCRIPT"
done
