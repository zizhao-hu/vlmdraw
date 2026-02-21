#!/bin/bash
#SBATCH --job-name=DepthAnalysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=jessetho_1732
#SBATCH --output=logs/depth_%j.out
#SBATCH --error=logs/depth_%j.err

PYTHON=/home1/zizhaoh/.conda/envs/DREAM/bin/python
PIP=/home1/zizhaoh/.conda/envs/DREAM/bin/pip

echo "============================================"
echo "Depth Analysis: Real vs. AI-Generated"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

cd /project2/jessetho_1732/zizhaoh/vlmdraw

# Redirect HF cache to scratch
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
mkdir -p $HF_HOME

# Install deps
$PIP install -q datasets matplotlib 2>/dev/null

# Step 1: Download sample images (if not already present)
if [ ! -d "data/samples/real" ] || [ ! -d "data/samples/fake" ]; then
    echo "Downloading sample images..."
    $PYTHON experiments/download_samples.py \
        --output-dir data/samples \
        --n-images 30
fi

echo ""
echo "Real images: $(ls data/samples/real/ 2>/dev/null | wc -l)"
echo "Fake images: $(ls data/samples/fake/ 2>/dev/null | wc -l)"
echo ""

# Step 2: Run depth analysis
$PYTHON experiments/depth_analysis.py \
    --real-dir data/samples/real \
    --fake-dir data/samples/fake \
    --output-dir results/depth_analysis \
    --model depth-anything/Depth-Anything-V2-Small-hf \
    --max-images 30

echo ""
echo "Done. Results in results/depth_analysis/"
