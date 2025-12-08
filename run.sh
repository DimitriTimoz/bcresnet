#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=256G
#SBATCH --account=m25206

# BCResNet training script for Juliet cluster
# Based on README: python main.py --tau 8 --gpu 0 --ver 2 --download

cd "$(dirname "$0")" || exit 1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bcresnet

# Run BCResNet training with GPU 0, tau=8 (BCResNet-8), and Google Speech Commands v2
python main.py --tau 8 --gpu 0 --ver 2 --download

