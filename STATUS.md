# Project Status Summary

**Date**: 2025-10-14  
**Location**: `/users/7/li003385/workspace/Ai4drug/Sai_nemo_AI4drug`

---

## ✅ Completed

### 1. Project Setup
- ✅ Cloned repository
- ✅ Created clean directory structure
- ✅ Data normalized to [1-10] integers
- ✅ Split: 200 training + 1000 test samples

### 2. Qwen2-7B Model  
- ✅ Trained with 8-bit quantization on H100
- ✅ Training time: 4.5 minutes
- ✅ Evaluated on 1000 test samples
- ✅ **Accuracy (±1)**: **53.5%** 
- ✅ Exact match: 23.9%
- ✅ Model saved: `models/qwen_7b/checkpoints/`

### 3. File Structure Reorganization
- ✅ Separated 7B and 30B directories
- ✅ Unified documentation (README_MODELS.md)
- ✅ Updated evaluation standard (±1 counts as correct)

---

## 🔄 In Progress

### Qwen3-30B Model
- **Job ID**: 42158227
- **GPU**: 2x A100 (agb02)
- **Status**: Running (21+ minutes)
- **Issue**: Training stuck at 0/21 steps (possible kernel version issue)

**Note**: System kernel 4.18.0 is below recommended 5.5.0, which may cause process hang.

---

## 📁 Final Structure

```
models/
├── qwen_7b/              # Qwen2-7B ✅ DONE
│   ├── scripts/          # train_7b.py, evaluate_7b.py, job scripts
│   ├── checkpoints/      # qwen_lora_finetuned/ (78MB)
│   └── output/           # evaluation_results_7b.json
│
└── qwen_30b/             # Qwen3-30B 🔄 TRAINING
    ├── scripts/          # train_30b.py, evaluate_30b.py, job scripts
    ├── checkpoints/      # (will be created after training)
    └── output/           # (will be created after evaluation)

data/                     # Shared data
├── train_data.jsonl (200)
└── test_data.jsonl (1000)
```

---

## 📊 Results with New Evaluation Standard

**Evaluation Criteria**: Predictions within ±1 error are considered correct.

### Qwen2-7B Results:
- **Accuracy (±1 allowed)**: **53.5%** ✅ Main metric
- **Exact match rate**: 23.9%
- **MAE**: 1.59
- **RMSE**: 2.04

### Qwen3-30B Results:
- Pending training completion

---

## 🎯 Next Options

### Option 1: Wait for 30B training
```bash
# Monitor (training may take hours if stuck)
tail -f logs/train_30b_42158227.err | grep "%"
```

### Option 2: Cancel and retry with different configuration
```bash
scancel 42158227

# Try with single GPU or different partition
# Edit: models/qwen_30b/scripts/train_job_30b.sh
```

### Option 3: Use 7B results
- 7B model is working and provides 53.5% accuracy
- Results available in `models/qwen_7b/output/`

---

## 📚 Documentation

**Main Guide**: `README_MODELS.md`  
**Status**: This file

---

## 🔗 Key Commands

```bash
# Check jobs
squeue -u li003385

# View 7B results
cat models/qwen_7b/output/evaluation_results_7b.json

# Monitor 30B
tail -f logs/train_30b_*.err

# Cancel 30B if needed
scancel 42158227
```

---

**Status**: 7B ✅ Complete | 30B 🔄 Training (may be stuck)

