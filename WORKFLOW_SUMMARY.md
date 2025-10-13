# Training and Evaluation Workflow Summary

## 🎯 Goal

1. **Fine-tune** Qwen 7B model on `data_An/train_data.jsonl` (1100 samples)
2. **Evaluate** on `data_An/test_data.csv` (100 samples)
3. **Run** 3 rounds of human feedback loop
4. **Report** accuracy after each round

## 📊 Data Summary

### Training Data
- File: `data_An/train_data.jsonl`
- Samples: 1100
- Format: JSONL (conversation format for Qwen)
- Scores: Integer 1-10
- Score distribution:
  - Most common: Score 5 (23.9%), Score 6 (23.5%)
  - Least common: Score 1 (0.6%), Score 10 (0.4%)

### Test Data
- File: `data_An/test_data.csv` / `data_An/test_data.jsonl`
- Samples: 100
- Scores: Integer 2-9 (no samples with score 1, 8, or 10)
- Mean score: 4.32

## 🚀 How to Run

### Option 1: Full Training + Evaluation (REQUIRES GPU)

```bash
# Step 1: Update lora_finetuning.py to use data_An
# Edit lora_finetuning.py:
#   - Change train_data_path to "data_An/train_data.jsonl"
#   - Change output_dir to "./qwen_lora_finetuned_An"

# Step 2: Train model (requires GPU with 16GB+ memory)
python3 lora_finetuning.py

# Step 3: Evaluate with 3 feedback rounds
python3 evaluate_with_feedback.py
```

### Option 2: Quick Demo (No GPU needed)

```bash
# This will use mock predictions to demonstrate the workflow
python3 evaluate_with_feedback.py
```

## 📈 Expected Output

After each round, you'll see:

```
======================================================================
ACCURACY RESULTS - Round 0
======================================================================
Exact Match (score == ground truth):   12.00%
Within ±1 (|score - truth| <= 1):      35.00%
Within ±2 (|score - truth| <= 2):      58.00%
Mean Absolute Error (MAE):              2.45
Root Mean Squared Error (RMSE):         3.12
======================================================================
```

Then after 3 feedback rounds, final summary:

```
======================================================================
ACCURACY SUMMARY ACROSS ALL ROUNDS
======================================================================
Round           Exact        ±1           ±2           MAE       
----------------------------------------------------------------------
Initial          12.00%      35.00%      58.00%       2.45
Round 1          15.00%      40.00%      63.00%       2.20
Round 2          18.00%      45.00%      68.00%       2.00
Round 3          22.00%      50.00%      73.00%       1.85

----------------------------------------------------------------------
Total Improvement: +10.00%
  Initial Accuracy: 12.00%
  Final Accuracy:   22.00%
======================================================================
```

## 📝 Metrics Explanation

### Exact Match
- Predicted score **exactly** equals ground truth
- Example: Predict 5, Truth 5 ✓

### Within ±1
- |Predicted - Truth| ≤ 1
- Examples: 
  - Predict 5, Truth 5 ✓
  - Predict 5, Truth 6 ✓
  - Predict 5, Truth 4 ✓
  - Predict 5, Truth 7 ✗

### Within ±2
- |Predicted - Truth| ≤ 2
- Includes all Within ±1 plus one more level

### MAE (Mean Absolute Error)
- Average of |Predicted - Truth| across all samples
- Lower is better
- Range: 0 (perfect) to 9 (worst for 1-10 scale)

### RMSE (Root Mean Squared Error)
- Square root of mean squared errors
- Penalizes large errors more than MAE
- Lower is better

## 📂 Output Files

After running, check `evaluation_results/`:

1. **evaluation_results.json** - Complete details:
   - All predictions for each round
   - Ground truth values
   - Timestamps
   - Model configuration

2. **accuracy_summary.csv** - Quick summary table:
   ```csv
   round,exact_match_%,within_1_%,within_2_%,mae,rmse
   0,12.00,35.00,58.00,2.45,3.12
   1,15.00,40.00,63.00,2.20,2.85
   2,18.00,45.00,68.00,2.00,2.60
   3,22.00,50.00,73.00,1.85,2.40
   ```

## 🔄 Human Feedback Loop

In each round:

1. **Identify** uncertain predictions (low confidence)
2. **Select** pairs of molecules for comparison
3. **Present** to human expert: "Which has higher score?"
4. **Collect** expert judgment (A > B, B > A, or Equal)
5. **Update** model with feedback
6. **Re-evaluate** on test set

Currently the human feedback is **simulated** in the demo. For real feedback, integrate with `human_feedback.py`.

## ⚙️ Configuration

Edit `config.yaml` to adjust:

```yaml
human_feedback:
  num_pairs_per_round: 5         # Change number of pairs
  max_rounds: 3                  # Change number of rounds
  selection_strategy: "entropy"  # How to select pairs

prediction:
  num_samples: 10                # Affects confidence calculation
```

## 🎓 Training Details

When running real training (`lora_finetuning.py`):

- **Model**: Qwen-7B-Chat
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA rank**: 8
- **Learning rate**: 2e-4
- **Epochs**: 3
- **Batch size**: 4
- **Training time**: ~15-25 minutes on RTX 3090

## 📊 Baseline Expectations

For a well-trained model on this task:

- **Initial Exact Match**: 10-20%
- **Initial Within ±1**: 30-50%
- **After 3 rounds**: +5-15% improvement

These numbers depend on:
- Quality of training data
- Model capacity
- Quality of human feedback
- Similarity between train and test distributions

## 🛠️ Troubleshooting

**No improvement across rounds?**
- This is expected in demo mode (mock predictions)
- Real model will improve with actual feedback

**Low accuracy?**
- Check if model is properly trained
- Verify test/train data alignment
- Consider more training epochs or data

**GPU memory error?**
- Reduce batch_size in training
- Use gradient accumulation
- Try smaller model or quantization

## ✅ Quick Checklist

- [✓] Data prepared (1100 train, 100 test)
- [✓] Scores rescaled to integer 1-10
- [✓] Evaluation script ready
- [ ] Train model (requires GPU)
- [ ] Run evaluation with 3 feedback rounds
- [ ] Check results in `evaluation_results/`

## 📚 Next Steps

1. Train model on actual GPU
2. Implement real human feedback interface
3. Analyze prediction errors
4. Consider data augmentation
5. Experiment with different hyperparameters

Good luck! 🚀

