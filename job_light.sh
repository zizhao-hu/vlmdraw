#!/bin/bash
#SBATCH --job-name=LightEst
#SBATCH --output=logs/light_est_%j.out
#SBATCH --error=logs/light_est_%j.err
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00


mkdir -p logs

echo "🔦 Light Estimation Experiment"
echo "   Node: $(hostname)"
echo "   GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

source ~/.bashrc || true
conda activate DREAM

cd /project2/jessetho_1732/zizhaoh/vlmdraw

# Download data if not present
if [ ! -d "data/aigenbench/real" ] || [ -z "$(ls -A data/aigenbench/real 2>/dev/null)" ]; then
    echo "📥 Downloading AIGenBench samples..."
    python experiments/download_real.py
    # Fake images should already exist; if not, download them
    if [ ! -d "data/aigenbench/fake" ] || [ -z "$(ls -A data/aigenbench/fake 2>/dev/null)" ]; then
        python experiments/download_samples.py --output-dir data/aigenbench --n-images 30
    fi
fi

python experiments/light_estimation.py \
    --real-dir data/aigenbench/real \
    --fake-dir data/aigenbench/fake \
    --output-dir results/light_estimation \
    --max-images 30 \
    --block-size 64 \
    --device cuda

echo "Done."
