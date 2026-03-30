#!/bin/bash
# Submit one SLURM job per HIQL dataset.
# Usage: bash submit_hiql_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/run_hiql.slurm"

jobs=(
  "pointmaze navigate large"
  #"antmaze   navigate medium"
  #"antmaze   navigate large"
  #"antmaze   stitch   medium"
  #"antmaze   stitch   large"
  #"antmaze   explore  medium"
  #"antmaze   explore  large"
  #"antsoccer navigate arena"
  #"antsoccer navigate medium"
  #"cube      play     single"
)

for job in "${jobs[@]}"; do
  read -r env task dsize <<< "$job"
  dataset="${env}-${dsize}-${task}-v0"
  echo "Submitting: $dataset"
  sbatch --job-name="hiql-${env}-${dsize}-${task}" \
         --export=ALL,ENV="$env",TASK="$task",DSIZE="$dsize" \
         "$SLURM_SCRIPT"
done
