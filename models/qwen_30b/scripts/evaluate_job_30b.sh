#!/bin/bash
#SBATCH --job-name=qwen30b_eval
#SBATCH --partition=a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=../../../logs/eval_30b_%j.out
#SBATCH --error=../../../logs/eval_30b_%j.err

# Create logs directory
mkdir -p ../../../logs

# Load modules
module load cuda/12.1.1

# Print GPU info
echo "========================================"
echo "Qwen2-32B Evaluation Job Started"
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
nvidia-smi
echo ""

# Navigate to script directory
cd /users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug/models/qwen_30b/scripts

# Run evaluation
echo "Starting Qwen2-32B evaluation on 1000 test samples..."
python3 evaluate_30b.py

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"

