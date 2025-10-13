# Training and Evaluation Guide

## Overview

Train Qwen 7B on `data_An/train_data.jsonl` (1100 samples) and evaluate on `data_An/test_data.csv` (100 samples) with 3 rounds of human feedback.

## Step-by-Step Process

### Step 1: Train the Model

```bash
# Option A: Use the automated script
./train_finetuned_model.sh

# Option B: Manual training
python3 lora_finetuning.py
# (Make sure to update paths in the script to use data_An/)
```

**Training Details:**
- Training data: `data_An/train_data.jsonl` (1100 samples)
- Output model: `qwen_lora_finetuned_An/`
- Scores: Integer range 1-10
- Expected time: ~15-25 minutes on RTX 3090

### Step 2: Evaluate with Human Feedback

```bash
python3 evaluate_with_feedback.py
```

**Evaluation Process:**
1. **Round 0 (Initial)**: Evaluate model on 100 test samples
2. **Round 1**: Collect human feedback → Re-evaluate
3. **Round 2**: Collect human feedback → Re-evaluate  
4. **Round 3**: Collect human feedback → Re-evaluate

**Metrics Reported:**
- **Exact Match**: Predicted score == Ground truth
- **Within ±1**: |Predicted - Truth| <= 1
- **Within ±2**: |Predicted - Truth| <= 2
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

### Expected Output Format

```
==================================================
ACCURACY RESULTS - Round 0
==================================================
Exact Match (score == ground truth):   12.00%
Within ±1 (|score - truth| <= 1):      35.00%
Within ±2 (|score - truth| <= 2):      58.00%
Mean Absolute Error (MAE):              2.45
Root Mean Squared Error (RMSE):         3.12
==================================================

[... Human Feedback Round 1 ...]

==================================================
ACCURACY RESULTS - Round 1
==================================================
Exact Match (score == ground truth):   15.00%
Within ±1 (|score - truth| <= 1):      40.00%
Within ±2 (|score - truth| <= 2):      63.00%
Mean Absolute Error (MAE):              2.20
Root Mean Squared Error (RMSE):         2.85
==================================================

[... continues for 3 rounds ...]
```

### Output Files

After evaluation, results are saved in `evaluation_results/`:

- `evaluation_results.json` - Detailed predictions and metrics
- `accuracy_summary.csv` - Summary table of accuracy across rounds

## Quick Test (Mock Predictions)

If you want to test the evaluation pipeline without training:

```bash
python3 evaluate_with_feedback.py
```

This will use mock predictions to demonstrate the workflow and output format.

## Data Format

**Test Data** (`data_An/test_data.csv`):
```csv
Structure,Score
CCCCCCCCCCCCCCCCCCNC(=O)...,4
CCCCCCCCCCCCCCCCNC(=O)...,8
```

**Scores**: Integer values from 1 to 10

## Hardware Requirements

- GPU: 16GB+ memory (for Qwen 7B)
- RAM: 32GB+ recommended
- Storage: ~15GB for model weights

## Troubleshooting

**Issue**: Model loading fails
- Check if model exists at `qwen_lora_finetuned_An/`
- Run training first: `./train_finetuned_model.sh`

**Issue**: GPU out of memory
- Reduce batch size in training script
- Use gradient accumulation

**Issue**: Predictions seem random
- Ensure model is properly trained
- Check test data format matches training data

## Notes

- Human feedback is currently simulated in the evaluation script
- For real human feedback, integrate with `human_feedback.py`
- Accuracy typically improves with each feedback round
- Final results are saved for analysis

