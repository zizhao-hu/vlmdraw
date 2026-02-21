#!/bin/bash
#SBATCH --job-name=BrightDepth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=jessetho_1732
#SBATCH --output=logs/bd_%j.out
#SBATCH --error=logs/bd_%j.err

PYTHON=/home1/zizhaoh/.conda/envs/DREAM/bin/python

echo "============================================"
echo "Brightness-Depth Consistency Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

cd /project2/jessetho_1732/zizhaoh/vlmdraw

export HF_HOME=/scratch1/zizhaoh/.cache/huggingface

# Use existing AI-GenBench samples
echo "Real images: $(ls data/aigenbench/real/ 2>/dev/null | wc -l)"
echo "Fake images: $(ls data/aigenbench/fake/ 2>/dev/null | wc -l)"

# Run brightness-depth analysis
$PYTHON experiments/brightness_depth.py \
    --real-dir data/aigenbench/real \
    --fake-dir data/aigenbench/fake \
    --output-dir results/brightness_depth \
    --model depth-anything/Depth-Anything-V2-Small-hf \
    --max-images 30

echo ""
echo "Done."
