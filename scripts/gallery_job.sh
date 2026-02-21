#!/bin/bash
#SBATCH --job-name=Gallery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=jessetho_1732
#SBATCH --output=logs/gallery_%j.out
#SBATCH --error=logs/gallery_%j.err

PYTHON=/home1/zizhaoh/.conda/envs/DREAM/bin/python

echo "============================================"
echo "Gallery Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

cd /project2/jessetho_1732/zizhaoh/vlmdraw
export HF_HOME=/scratch1/zizhaoh/.cache/huggingface

$PYTHON experiments/make_gallery.py \
    --real-dir data/aigenbench/real \
    --fake-dir data/aigenbench/fake \
    --output-dir results/gallery \
    --n-images 15

echo "Done."
