#!/bin/bash
#SBATCH --job-name=qwen_finetune
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory
mkdir -p logs

# Load modules
module load cuda/12.1.1

# Print GPU info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

# Run training
cd /users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug
python3 lora_finetuning.py

echo "Job completed at: $(date)"

