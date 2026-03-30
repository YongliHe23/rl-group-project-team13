#!/bin/bash
# Submit one SLURM job per CRL dataset.
# Usage: bash submit_crl_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/run_crl.slurm"

jobs=(
  "antsoccer navigate medium"
  "cube      play     single"
)

for job in "${jobs[@]}"; do
  read -r env task dsize <<< "$job"
  dataset="${env}-${dsize}-${task}-v0"
  echo "Submitting: $dataset"
  sbatch --job-name="crl-${env}-${dsize}-${task}" \
         --export=ALL,ENV="$env",TASK="$task",DSIZE="$dsize" \
         "$SLURM_SCRIPT"
done
