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

# Symlink $WORKDIR/venv -> scratch venv.
# Must remove first: if $VENV_LINK is an existing directory, ln -sfn would
# create the symlink *inside* it instead of replacing it.
rm -rf "$VENV_LINK"
ln -sfn "$SCRATCH_VENV" "$VENV_LINK"
source "$SCRATCH_VENV/bin/activate"

# Bootstrap pip into the venv using the PYTHONPATH-provided pip from ~/.local.
# ensurepip is unavailable on this system's python/3.10 module, so we rely on
# the user-site pip for this one step, then immediately upgrade to the latest.
python -m pip install --upgrade pip wheel setuptools

# Clear PYTHONPATH now that pip is bootstrapped.  Without this, Python 3.9
# packages from ~/.local leak into the venv and cause import errors at runtime
# (e.g. wrong typing_extensions version picked up by pydantic_core).
unset PYTHONPATH

# Install CUDA-enabled PyTorch first (requirements.txt pins torch==2.10.0)
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

# Install remaining project requirements
python -m pip install -r "$REQUIREMENTS"

# The system ldconfig cache resolves libcupti.so.12 to the system CUDA version,
# which is missing the versioned symbol `cuptiActivityEnableDriverApi@@libcupti.so.12`
# that torch 2.10+cu126 requires.  LD_PRELOAD forces the correct bundled version
# to be loaded first, before ldconfig or LD_LIBRARY_PATH are consulted.
SITE_PKGS="$SCRATCH_VENV/lib/python3.10/site-packages"
export LD_PRELOAD="$SITE_PKGS/nvidia/cuda_cupti/lib/libcupti.so.12"

# Sanity check
python - <<'PY'
import omnisafe
print("omnisafe:", omnisafe.__version__)
import safety_gymnasium
print("safety_gymnasium: OK")
PY

echo "VENV READY at $SCRATCH_VENV (linked from $VENV_LINK)"
