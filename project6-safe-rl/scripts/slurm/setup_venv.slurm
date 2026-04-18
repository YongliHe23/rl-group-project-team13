#!/bin/bash
#SBATCH --job-name=setup-venv-saferl
#SBATCH --account=jfnielse0
#SBATCH --partition=spgpu
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yonglihe@umich.edu

set -euo pipefail

WORKDIR=/home/yonglihe/ece567/project/rl-group-project-team13/project6-safe-rl
SCRATCH_VENV=/scratch/jfnielse_root/jfnielse0/yonglihe/ece_567/project6/venv
VENV_LINK=$WORKDIR/venv
REQUIREMENTS=$WORKDIR/requirements.txt

module purge
module load python/3.10
module load cuda/12.6.3

if [ -d "$SCRATCH_VENV" ]; then
  echo "Removing existing venv at $SCRATCH_VENV"
  rm -rf "$SCRATCH_VENV"
fi

mkdir -p "$(dirname "$SCRATCH_VENV")"
python -m venv "$SCRATCH_VENV"

# Symlink $WORKDIR/venv -> scratch venv
ln -sfn "$SCRATCH_VENV" "$VENV_LINK"
source "$SCRATCH_VENV/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Install CUDA-enabled PyTorch first (requirements.txt pins torch==2.10.0)
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

# Install remaining project requirements
python -m pip install -r "$REQUIREMENTS"

# Sanity check
python - <<'PY'
import omnisafe
print("omnisafe:", omnisafe.__version__)
import safety_gymnasium
print("safety_gymnasium: OK")
PY

echo "VENV READY at $SCRATCH_VENV (linked from $VENV_LINK)"
