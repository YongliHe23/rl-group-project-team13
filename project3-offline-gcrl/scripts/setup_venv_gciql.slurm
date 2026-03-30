#!/bin/bash
#SBATCH --job-name=setup-venv-gciql
#SBATCH --account=jfnielse0
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yonglihe@umich.edu

set -euo pipefail

WORKDIR=/home/yonglihe/ece567/project
VENV_DIR=$WORKDIR/venv
REQUIREMENTS=$WORKDIR/rl-group-project-team13/project3-offline-gcrl/requirements.txt
OGBENCH_IMPLS_REQ=$WORKDIR/ogbench/impls/requirements.txt

module purge
module load python/3.10
module load cuda/12.6.3

# Remove any broken venv (e.g. one created on a different machine)
if [ -d "$VENV_DIR" ]; then
  echo "Removing existing venv at $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Use python -m pip to ensure we're using the venv's Python, not any system pip
python -m pip install --upgrade pip wheel setuptools

# Install CUDA-enabled PyTorch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install project requirements
python -m pip install -r "$REQUIREMENTS"

# Install ogbench impls requirements (agents, utils dependencies)
python -m pip install -r "$OGBENCH_IMPLS_REQ"

# Sanity check
python - <<'PY'
import jax
print("JAX:", jax.__version__)
import ml_collections
print("ml_collections: OK")
import ogbench
print("ogbench: OK")
PY

echo "VENV READY at $VENV_DIR"
