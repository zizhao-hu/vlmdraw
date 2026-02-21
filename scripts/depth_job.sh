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
echo "Using AI-GenBench data"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

cd /project2/jessetho_1732/zizhaoh/vlmdraw

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface
mkdir -p $HF_HOME

$PIP install -q datasets matplotlib 2>/dev/null

# Step 1: Download AI-GenBench samples
echo ""
echo "=== Downloading AI-GenBench samples ==="
$PYTHON experiments/download_samples.py \
    --output-dir data/aigenbench \
    --n-images 30

echo ""
echo "Real images: $(ls data/aigenbench/real/ 2>/dev/null | wc -l)"
echo "Fake images: $(ls data/aigenbench/fake/ 2>/dev/null | wc -l)"

# Step 2: Run depth analysis
echo ""
echo "=== Running Depth Analysis ==="
$PYTHON experiments/depth_analysis.py \
    --real-dir data/aigenbench/real \
    --fake-dir data/aigenbench/fake \
    --output-dir results/depth_aigenbench \
    --model depth-anything/Depth-Anything-V2-Small-hf \
    --max-images 30

echo ""
echo "Done. Results in results/depth_aigenbench/"
