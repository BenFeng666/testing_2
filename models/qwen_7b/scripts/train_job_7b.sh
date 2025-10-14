#!/bin/bash
#SBATCH --job-name=qwen7b_train
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=../../../logs/train_7b_%j.out
#SBATCH --error=../../../logs/train_7b_%j.err

# Create logs directory
mkdir -p ../../../logs

# Load modules
module load cuda/12.1.1

# Print GPU info
echo "========================================"
echo "Qwen2-7B Training Job Started"
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
nvidia-smi
echo ""

# Navigate to script directory
cd /users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug/models/qwen_7b/scripts

# Run training
echo "Starting Qwen2-7B training..."
python3 train_7b.py

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"

