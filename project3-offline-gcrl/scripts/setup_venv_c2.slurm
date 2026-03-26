#!/bin/bash
#SBATCH --job-name=setup-venv
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=gpu           # short GPU job to ensure CUDA libs are visible
#SBATCH --qos=class
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail

# ---- Adjust these paths if you prefer a different location ----
VENV_DIR=/home/stevengu/scratch/myproject0/stevengu/ece567/venv
WORKDIR=/home/stevengu/scratch/myproject0/stevengu/ece567

module purge
module load python/3.11
module load cuda           # ensure CUDA runtime is in env

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- Install CUDA-enabled PyTorch ---
# The index URL below targets CUDA wheels hosted by PyTorch.
# If you need a specific CUDA version, adjust 'cu126' to match the site default.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# --- Your project deps (edit as needed) ---
# If you have a requirements.txt in your repo, uncomment:
pip install -r "$WORKDIR/requirements.txt"

# Common scientific stack (optional)
#pip install numpy scipy pandas tqdm matplotlib 

# Quick sanity check
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("GPU:", torch.cuda.get_device_name(i))
PY

echo "VENV READY at $VENV_DIR"
