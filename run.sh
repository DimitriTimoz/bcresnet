#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=256G
#SBATCH --account=m25206

set -euo pipefail

# BCResNet training script for Juliet cluster
# Based on README: python main.py --tau 8 --gpu 0 --ver 2 --download

# Always work from the submit directory (where you ran sbatch)
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# Unbuffered logs and local log directory for Slurm outputs
export PYTHONUNBUFFERED=1
LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/slurm_logs"
mkdir -p "$LOG_DIR"

# If you know your python, pass it: PYTHON_BIN=/path/to/python sbatch run.sh
PYTHON_CANDIDATES=(
  "${PYTHON_BIN:-}"
  "$HOME/miniconda3/envs/h2ogpt/bin/python"
  "$HOME/miniconda3/envs/bcresnet/bin/python"
  "$HOME/anaconda3/envs/h2ogpt/bin/python"
  "$HOME/anaconda3/envs/bcresnet/bin/python"
)

# Ensure module command is available, then try loading miniconda module
if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load miniconda3/24.3.0 2>/dev/null || module load miniconda3 2>/dev/null || true
fi

# Pick the first existing python among candidates
PYTHON_BIN=""
for cand in "${PYTHON_CANDIDATES[@]}"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    PYTHON_BIN="$cand"
    break
  fi
done

# If still empty, try to source conda then activate envs
if [ -z "$PYTHON_BIN" ]; then
  CONDA_FOUND=false
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
  elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.conda/etc/profile.d/conda.sh"
    CONDA_FOUND=true
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    CONDA_FOUND=true
  fi

  if [ "$CONDA_FOUND" = true ]; then
    if conda activate bcresnet 2>/dev/null || conda activate h2ogpt 2>/dev/null; then
      PYTHON_BIN="$(command -v python)"
    fi
  fi
fi

# Last-resort: build a venv with system python if still empty
if [ -z "$PYTHON_BIN" ]; then
  if command -v module >/dev/null 2>&1; then
    module load python/3.11.9 2>/dev/null || module load python/3.10.10 2>/dev/null || true
  fi
  BASE_PY="$(command -v python3 || true)"
  if [ -z "$BASE_PY" ]; then
    echo "[ERROR] Aucun python3 trouvé. Fixez PYTHON_BIN (ex: ~/miniconda3/envs/h2ogpt/bin/python)." >&2
    echo "[DEBUG] Tried candidates: ${PYTHON_CANDIDATES[*]}" >&2
    exit 1
  fi
  VENV_DIR="${SLURM_TMPDIR:-$HOME/.cache}/bcresnet_venv"
  "${BASE_PY}" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install torch torchvision torchaudio tqdm requests >/dev/null
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

echo "[INFO] Using python at $PYTHON_BIN"

# Run BCResNet training with GPU 0, tau=8 (BCResNet-8), and Google Speech Commands v2
"$PYTHON_BIN" -u main.py --tau 8 --gpu 0 --ver 2 --download 2>&1 | tee -a "$LOG_DIR/job-${SLURM_JOB_ID:-local}.log"

