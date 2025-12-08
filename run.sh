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

# If you know your python, pass it: PYTHON_BIN=/path/to/python sbatch run.sh
PYTHON_CANDIDATES=(
  "${PYTHON_BIN:-}"
  "$HOME/miniconda3/envs/h2ogpt/bin/python"
  "$HOME/miniconda3/envs/bcresnet/bin/python"
  "$HOME/anaconda3/envs/h2ogpt/bin/python"
  "$HOME/anaconda3/envs/bcresnet/bin/python"
  "/usr/bin/python3"
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
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.conda/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "[ERROR] conda not found. Fixez PYTHON_BIN (ex: ~/miniconda3/envs/h2ogpt/bin/python) ou chargez le module miniconda3." >&2
    echo "[DEBUG] Tried candidates: ${PYTHON_CANDIDATES[*]}" >&2
    exit 1
  fi

  conda activate bcresnet || conda activate h2ogpt || {
    echo "[ERROR] conda env 'bcresnet' ou 'h2ogpt' introuvable. Fixez PYTHON_BIN ou créez l'env." >&2
    exit 1
  }

  PYTHON_BIN="$(command -v python)"
fi

echo "[INFO] Using python at $PYTHON_BIN"

# Run BCResNet training with GPU 0, tau=8 (BCResNet-8), and Google Speech Commands v2
"$PYTHON_BIN" main.py --tau 8 --gpu 0 --ver 2 --download

