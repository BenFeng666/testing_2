# Qwen Model Fine-tuning for Molecular Score Prediction

Fine-tuning Qwen models (7B and 30B) on SMILES molecular data using LoRA.

**Repository**: https://github.com/Nemo0412/Sai_nemo_AI4drug.git  
**Location**: `/users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug`

---

## 📁 Project Structure

```
Sai_nemo_AI4drug/
├── data/                          # Shared training data
│   ├── train_data.jsonl (200)    # Training set
│   └── test_data.jsonl (1000)    # Test set
│
├── models/
│   ├── qwen_7b/                   # Qwen2-7B ✅ Completed
│   │   ├── scripts/               # Training/eval scripts
│   │   ├── checkpoints/           # Model weights (78MB)
│   │   └── output/                # Results
│   │
│   └── qwen_30b/                  # Qwen3-30B 🆕 Ready
│       ├── scripts/               # Training/eval scripts
│       ├── checkpoints/           # Will save model
│       └── output/                # Will save results
│
└── logs/                          # All job logs
```

---

## 🚀 Quick Start

### Train Qwen3-30B Model
```bash
cd models/qwen_30b/scripts
sbatch train_job_30b.sh

# Monitor progress
squeue -u $USER
tail -f ../../../logs/train_30b_*.out
```

### After training, run evaluation
```bash
sbatch evaluate_job_30b.sh

# View results
cat ../output/evaluation_results_30b.json
```

---

## 📊 Model Comparison

| Feature | Qwen2-7B | Qwen3-30B |
|---------|----------|-----------|
| **Status** | ✅ Completed | 🆕 Ready |
| **Model** | Qwen/Qwen2-7B-Instruct | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| **Parameters** | 7B | 30B |
| **Quantization** | 8-bit | 4-bit NF4 |
| **LoRA Rank** | 8 | 16 |
| **GPU** | 1x H100 | 2x A100 |
| **Memory** | ~40GB | ~60-80GB |
| **Training Time** | ~5 min | ~20-30 min |
| **Accuracy (±1)** | 53.5% | Expected higher |
| **Exact Match** | 23.9% | Expected higher |

---

## 📈 Qwen2-7B Results (Completed)

### Performance Metrics
- **Accuracy (±1 error allowed)**: **53.5%** ← Main metric
- **Exact match rate**: 23.9%
- **MAE**: 1.59
- **RMSE**: 2.04

### Key Findings
✅ **Strengths**:
- Good for scores 5-6 (64% and 37% exact match)
- Fast training (4.5 minutes)
- All predictions successful

⚠️ **Limitations**:
- Only predicts 3 values (4, 5, 6)
- Cannot predict extreme values (1-3, 8-10)
- Prediction diversity lacking

**Files**: `models/qwen_7b/output/`

---

## 🎯 Evaluation Criteria

**NEW**: Predictions within ±1 error are considered **correct**.

Example:
- True score: 5
- Predicted: 4, 5, or 6 → ✅ Correct
- Predicted: 3 or 7 → ❌ Incorrect

This reflects real-world tolerance for molecular score prediction.

---

## 💻 Training Configuration

### Qwen2-7B
```python
model_name: "Qwen/Qwen2-7B-Instruct"
quantization: 8-bit
lora_rank: 8
lora_alpha: 32
epochs: 3
batch_size: 1
gradient_accumulation: 16
learning_rate: 2e-4
```

### Qwen3-30B
```python
model_name: "Qwen/Qwen3-30B-A3B-Instruct-2507"
quantization: 4-bit NF4 + double quant
lora_rank: 16
lora_alpha: 32
epochs: 3
batch_size: 1
gradient_accumulation: 32
learning_rate: 1e-4
```

---

## 📝 Common Commands

### Check job status
```bash
squeue -u $USER
```

### View training logs
```bash
# 7B logs
tail -f logs/train_7b_*.out

# 30B logs  
tail -f logs/train_30b_*.out
```

### View results
```bash
# 7B results
cat models/qwen_7b/output/evaluation_results_7b.json

# 30B results
cat models/qwen_30b/output/evaluation_results_30b.json
```

### Cancel jobs
```bash
scancel <job_id>
```

---

## 🔧 GPU Resource Requirements

### For 7B Model
- Partition: `msigpu` or `interactive-gpu`
- GPU: 1x H100 or 1x A100
- Memory: 128GB RAM
- Time: ~4 hours limit

### For 30B Model
- Partition: `a100-8` (recommended)
- GPU: 2x A100 (40GB each)
- Memory: 256GB RAM
- Time: ~8 hours limit

---

## 📊 Data Information

- **Original**: 1200 samples from `data_new.xlsx`
- **Normalization**: Scores normalized from [-2.35, 15.96] to [1, 10] integers
- **Split**: 200 training + 1000 test
- **Format**: JSONL with conversational structure

---

## 🎯 Expected Improvements with 30B

1. **Better Accuracy**: 65-75% (vs 53.5%)
2. **Prediction Diversity**: Cover more score ranges
3. **Extreme Values**: Better handling of scores 1-3 and 8-10
4. **Lower Errors**: Reduced MAE and RMSE

---

## 📂 Output Files

### 7B Model (Completed)
- `models/qwen_7b/checkpoints/qwen_lora_finetuned/` - Model weights
- `models/qwen_7b/output/evaluation_results_7b.json` - Evaluation results
- `models/qwen_7b/output/detailed_analysis.txt` - Detailed analysis

### 30B Model (To be generated)
- `models/qwen_30b/checkpoints/qwen_30b_lora_finetuned/` - Model weights
- `models/qwen_30b/output/evaluation_results_30b.json` - Evaluation results

---

## ⚠️ Important Notes

1. **Accuracy Metric**: ±1 error is considered correct (53.5% for 7B)
2. **Quantization**: 
   - 7B uses 8-bit for speed
   - 30B uses 4-bit for memory efficiency
3. **Training Data**: Both models share the same 200 training samples
4. **GPU**: 30B requires 2x A100 GPUs

---

## 🔗 Quick Links

```bash
# Main directory
cd /users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug

# Train 30B
cd models/qwen_30b/scripts && sbatch train_job_30b.sh

# Evaluate 30B
cd models/qwen_30b/scripts && sbatch evaluate_job_30b.sh

# Compare results
cat models/qwen_7b/output/evaluation_results_7b.json
cat models/qwen_30b/output/evaluation_results_30b.json
```

---

**Status**: 7B ✅ | 30B 🔄  
**Next**: Train and evaluate Qwen3-30B model

