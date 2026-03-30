#!/bin/bash
# Submit a single job for powderworld-easy-play running all 4 seeds in parallel.
# Seeds 0 1 2 3 are passed via --seeds; --parallel-seeds runs them concurrently
# (CPU workers, spawn context).  --visual-enabled activates the ImpalaSmall encoder
# and discrete AWR actor required for powderworld environments.
# Usage: bash submit_qrl_powderworld.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/run_qrl.slurm"

echo "Submitting: powderworld-easy-play-v0 seeds 0 1 2 3 (parallel, visual)"
sbatch --job-name="qrl-powderworld-easy-parallel" \
       --mem=32G \
       --export=ALL,ENV="powderworld",TASK="play",DSIZE="easy",TRAIN_STEPS=1000000,SEEDS="0 1 2 3",PARALLEL_SEEDS=1,VISUAL_ENABLED=1 \
       "$SLURM_SCRIPT"
