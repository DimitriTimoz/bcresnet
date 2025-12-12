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

# Setup workspace paths according to Juliet best practices
# Data -> /scratch_l (fast local temporary storage per node)
# Logs -> submit directory (persistent)
# Venv -> /scratch_l (fast and cleaned between jobs)
SCRATCH_DIR="/scratch_l/${USER}/bcresnet_${SLURM_JOB_ID:-local}"
mkdir -p "$SCRATCH_DIR"
export DATA_DIR="$SCRATCH_DIR/data"
mkdir -p "$DATA_DIR"

# Copy project files to scratch (code + existing data)
echo "[INFO] Copying project to scratch..."
rsync -a --exclude='slurm_logs' --exclude='slurm-*.out' --exclude='.git' "${SLURM_SUBMIT_DIR}/" "$SCRATCH_DIR/"

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
  # Use scratch for venv (faster and auto-cleaned)
  VENV_DIR="$SCRATCH_DIR/bcresnet_venv"
  echo "[INFO] Creating temporary venv in $VENV_DIR..."
  "${BASE_PY}" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install torch torchvision torchaudio tqdm requests soundfile scikit-learn matplotlib seaborn scipy numpy >/dev/null
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

echo "[INFO] Using python at $PYTHON_BIN"

# Work in scratch directory for fast I/O
cd "$SCRATCH_DIR" || exit 1

# Clean corrupted or incomplete dataset directories
if [ -d "./data/speech_commands_v0.02" ]; then
  TRAIN_12_DIR="./data/speech_commands_v0.02/train_12class"
  # Check if train_12class exists and has at least 12 subdirectories (one per class)
  if [ ! -d "$TRAIN_12_DIR" ] || [ "$(find "$TRAIN_12_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)" -lt 12 ]; then
    echo "[INFO] Cleaning incomplete dataset (train_12class missing or incomplete)..."
    rm -rf ./data/speech_commands_v0.02 ./data/speech_commands_v0.02_split
  else
    # Check if there are actually audio files in train_12class subdirs
    NUM_WAV=$(find "$TRAIN_12_DIR" -name "*.wav" | wc -l)
    if [ "$NUM_WAV" -lt 100 ]; then
      echo "[INFO] Cleaning incomplete dataset (too few audio files: $NUM_WAV)..."
      rm -rf ./data/speech_commands_v0.02 ./data/speech_commands_v0.02_split
    fi
  fi
fi

# Run BCResNet training with GPU 0, tau=8 (BCResNet-8), and Google Speech Commands v2
echo "[INFO] Starting training in $(pwd)..."
"$PYTHON_BIN" -u main.py --tau 8 --gpu 0 --ver 2 --download 2>&1 | tee -a "$LOG_DIR/job-${SLURM_JOB_ID:-local}.log"
TRAIN_EXIT_CODE=$?

# Copy all results back to submit directory
echo "[INFO] Copying results back to ${SLURM_SUBMIT_DIR}..."
# Copy logs
if [ -d "$LOG_DIR" ]; then
  mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"
  cp -r "$LOG_DIR"/* "${SLURM_SUBMIT_DIR}/slurm_logs/" 2>/dev/null || true
fi
# Copy data and any model/result files in scratch
find "$SCRATCH_DIR" -maxdepth 1 \( -name '*.pth' -o -name '*.pt' -o -name '*.pkl' -o -name '*.png' \) -exec cp {} "${SLURM_SUBMIT_DIR}/" \; 2>/dev/null || true
if [ -d "$SCRATCH_DIR/data" ]; then
  mkdir -p "${SLURM_SUBMIT_DIR}/data"
  rsync -a --exclude='*.tar.gz' "$SCRATCH_DIR/data/" "${SLURM_SUBMIT_DIR}/data/" 2>/dev/null || true
fi

echo "[INFO] Results saved to:"
echo "  Logs: ${SLURM_SUBMIT_DIR}/slurm_logs/job-${SLURM_JOB_ID:-local}.log"
echo "  Data: ${SLURM_SUBMIT_DIR}/data/"

# Cleanup scratch
echo "[INFO] Cleaning up scratch directory..."
rm -rf "$SCRATCH_DIR"

exit $TRAIN_EXIT_CODE

