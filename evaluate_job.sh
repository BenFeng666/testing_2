#!/bin/bash
#SBATCH --job-name=qwen_eval
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Create logs directory
mkdir -p logs

# Load modules
module load cuda/12.1.1

# Print GPU info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

# Run evaluation
cd /users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug
python3 evaluate_model.py

echo "Job completed at: $(date)"

