#!/bin/bash
#SBATCH --job-name=LightEst
#SBATCH --output=logs/light_est_%j.out
#SBATCH --error=logs/light_est_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
mkdir -p logs

echo "🔦 Light Estimation Experiment"
echo "   Node: $(hostname)"
echo "   GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

source ~/.bashrc
conda activate vlmdraw

python experiments/light_estimation.py \
    --real-dir data/aigenbench/real \
    --fake-dir data/aigenbench/fake \
    --output-dir results/light_estimation \
    --max-images 30 \
    --block-size 64 \
    --device cuda

echo "Done."
