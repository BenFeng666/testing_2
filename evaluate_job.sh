#!/bin/bash
#SBATCH --job-name=chemllm_eval
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=../../../logs/eval_chemllm_%j.out
#SBATCH --error=../../../logs/eval_chemllm_%j.err

# =====================================================
# Setup
# =====================================================

# Create logs directory
mkdir -p ../../../logs

# Load CUDA module (skip this on Colab)
module load cuda/12.1.1 || echo "⚠️ CUDA module not found (Colab likely). Continuing..."

# Print job info
echo "========================================"
echo "ChemLLM Evaluation Job Started"
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
nvidia-smi || echo "⚠️ nvidia-smi not available (Colab likely)"
echo ""

# =====================================================
# Run Evaluation
# =====================================================

# ✅ Navigate to ChemLLM model directory
cd /content/testing2/chemllm || exit 1

# ✅ Run the evaluation script
echo "🚀 Starting ChemLLM evaluation..."
python3 evaluate_chemllm.py

# =====================================================
# End
# =====================================================
echo ""
echo "========================================"
echo "ChemLLM Evaluation Completed"
echo "Job finished at: $(date)"
echo "========================================"


