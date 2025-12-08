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

# Try direct python path first (faster/robust on Juliet)
PYTHON_CANDIDATES=(
	"${PYTHON_BIN:-}"
	"$HOME/miniconda3/envs/bcresnet/bin/python"
	"$HOME/miniconda3/envs/h2ogpt/bin/python"
	"$HOME/anaconda3/envs/bcresnet/bin/python"
	"$HOME/anaconda3/envs/h2ogpt/bin/python"
)

for cand in "${PYTHON_CANDIDATES[@]}"; do
	if [ -n "$cand" ] && [ -x "$cand" ]; then
		PYTHON_BIN="$cand"
		echo "[INFO] Using python at $PYTHON_BIN"
		break
	fi
done

if [ -z "${PYTHON_BIN:-}" ]; then
	# Fallback to conda activation
	if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
		source "$HOME/miniconda3/etc/profile.d/conda.sh"
	elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
		source "$HOME/anaconda3/etc/profile.d/conda.sh"
	elif command -v conda >/dev/null 2>&1; then
		eval "$(conda shell.bash hook)"
	else
		echo "[ERROR] conda not found. Set PYTHON_BIN to your env (e.g. ~/miniconda3/envs/h2ogpt/bin/python) or load the conda module." >&2
		exit 1
	fi
	conda activate bcresnet || conda activate h2ogpt || {
		echo "[ERROR] conda env 'bcresnet' or 'h2ogpt' not found. Set PYTHON_BIN or create/activate the env." >&2
		exit 1
	}
	PYTHON_BIN="$(command -v python)"
	echo "[INFO] Using python from conda env: $PYTHON_BIN"
fi

# Run BCResNet training with GPU 0, tau=8 (BCResNet-8), and Google Speech Commands v2
"$PYTHON_BIN" main.py --tau 8 --gpu 0 --ver 2 --download

