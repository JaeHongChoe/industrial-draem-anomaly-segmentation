#!/bin/bash
#SBATCH --job-name=draem_tile_v4
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --mem=356G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.log

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vision

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1

# Run training with resume
python3 -u train.py \
  --train_dirs data/train \
  --real_defect_path ../test_orig \
  --checkpoint_path ./checkpoints \
  --log_path ./logs \
  --dino_model vit_large_patch16_dinov3.lvd1689m \
  --gpus 0,1 \
  --bs 256 \
  --num_workers 64 \
  --epochs 300 \
  --visualize \
  --resume latest \
  --lr 0.0001

echo "End time: $(date)"
