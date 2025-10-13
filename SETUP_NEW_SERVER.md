# Setup on New Server

## Quick Setup Guide

### 1. Clone Repository

```bash
git clone https://github.com/Nemo0412/Sai_nemo_AI4drug.git
cd Sai_nemo_AI4drug
```

### 2. Transfer Data

**Option A: Copy from old server**
```bash
# On old server, data is already compressed
# File: data_An.tar.gz (contains 1100 train + 100 test samples)

# Transfer to new server
scp /home/li003385/workspace/AI4Drug/data_An.tar.gz user@newserver:/path/to/Sai_nemo_AI4drug/

# On new server, extract
cd /path/to/Sai_nemo_AI4drug
tar -xzf data_An.tar.gz
```

**Option B: If data files are already on new server**
```bash
# Just make sure data_An/ directory exists with:
# - train_data.jsonl (1100 samples)
# - test_data.jsonl (100 samples)
# - train_data.csv
# - test_data.csv
```

### 3. Install Dependencies

```bash
pip3 install torch transformers==4.35.0 peft==0.6.0 datasets accelerate pandas openpyxl pyyaml scipy numpy
```

**Important**: Use specific versions to avoid compatibility issues:
- transformers==4.35.0
- peft==0.6.0

### 4. Verify Setup

```bash
python3 test_pipeline.py
```

Expected output: All tests pass ✓

### 5. Run Training

```bash
# This will train Qwen-7B with LoRA on 1100 samples
python3 train_on_data_An.py
```

**Requirements**:
- GPU with 16GB+ memory
- CUDA installed
- ~20-30 minutes on RTX 3090/A6000

**Output**: Model saved to `qwen_lora_finetuned_An/`

### 6. Run Evaluation with 3 Human Feedback Rounds

```bash
python3 evaluate_with_feedback.py
```

**This will**:
1. Evaluate on 100 test samples
2. Show accuracy (Exact Match, Within ±1, etc.)
3. Run 3 rounds of human feedback (simulated for now)
4. Report accuracy after each round
5. Save results to `evaluation_results/`

## Data Info

**Training Set** (`data_An/train_data.jsonl`):
- 1100 samples
- Scores: Integer 1-10
- Distribution: Most samples have scores 5-6

**Test Set** (`data_An/test_data.csv`):
- 100 samples
- Scores: Integer 2-9
- Used for accuracy evaluation

## Expected Results

After training and 3 feedback rounds:

```
Round      Exact Match    Within ±1    MAE
----------------------------------------------
Initial    10-20%         30-50%       2.0-3.0
Round 1    15-25%         40-60%       1.5-2.5
Round 2    20-30%         50-70%       1.2-2.0
Round 3    25-35%         60-80%       1.0-1.5
```

## Troubleshooting on New Server

**Package conflicts?**
```bash
pip3 install --upgrade transformers==4.35.0 peft==0.6.0 --force-reinstall
```

**GPU not detected?**
```bash
nvidia-smi  # Check GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Out of memory?**
- Edit `train_on_data_An.py`: reduce `batch_size` from 4 to 2
- Increase `gradient_accumulation_steps` from 4 to 8

## Files in Repository

All code files are in the repository. Only need to transfer:
- `data_An.tar.gz` (training/test data)

Everything else will be cloned from GitHub.

Good luck on the new server! 🚀

