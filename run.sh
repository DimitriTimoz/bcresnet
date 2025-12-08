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

# Locate and activate conda env
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
	source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
	source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
else
	echo "[ERROR] conda not found. Load your conda module or adjust CONDA path." >&2
	exit 1
fi

conda activate bcresnet

# Run BCResNet training with GPU 0, tau=8 (BCResNet-8), and Google Speech Commands v2
python main.py --tau 8 --gpu 0 --ver 2 --download

