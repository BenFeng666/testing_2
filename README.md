# AI4Drug - Molecular Score Prediction System

AI-powered molecular scoring system with LoRA finetuning and human feedback active learning.

## 🚀 Quick Start

### Installation

```bash
pip3 install -r requirements.txt
python3 test_pipeline.py  # Test environment
```

### Run Training

```bash
# 1. Prepare training data (convert Excel to JSONL)
python3 prepare_training_data.py

# 2. Train model with LoRA
python3 lora_finetuning.py

# 3. Test trained model
python3 inference.py
```

### Run Active Learning with Human Feedback

```bash
# 1. Edit config.yaml to adjust thresholds (optional)
nano config.yaml

# 2. Run pipeline
python3 active_learning_pipeline.py

# 3. Provide feedback when prompted (A/B/E/S)
# 4. Check results in output/ directory
```

## 📁 File Structure

### Qwen Model Training (NEW - Organized Structure)

**Location**: `models/` directory contains separate configurations for 7B and 30B models.

#### Qwen2-7B Model ✅ (Completed)
- **Location**: `models/qwen_7b/`
- **Scripts**: 
  - `train_7b.py` - Training script (8-bit quantization)
  - `evaluate_7b.py` - Evaluation script
  - `train_job_7b.sh` - SLURM job script for training (1x H100 GPU)
  - `evaluate_job_7b.sh` - SLURM job script for evaluation
- **Results**: Accuracy 53.5% (±1 error allowed), exact match 23.9% in round 1.
- **Model**: Saved in `checkpoints/qwen_lora_finetuned/` (78MB)

#### Qwen3-30B Model 🔄 (In Progress)
- **Location**: `models/qwen_30b/`
- **Scripts**:
  - `train_30b.py` - Training script (4-bit NF4 quantization)
  - `evaluate_30b.py` - Evaluation script
  - `train_job_30b.sh` - SLURM job script for training (2x A100 GPUs)
  - `evaluate_job_30b.sh` - SLURM job script for evaluation
- **Expected**: Better accuracy than 7B model

**Quick Usage**:
```bash
# Train 7B model
cd models/qwen_7b/scripts && sbatch train_job_7b.sh

# Train 30B model
cd models/qwen_30b/scripts && sbatch train_job_30b.sh

# Evaluate models (after training)
sbatch evaluate_job_7b.sh  # or evaluate_job_30b.sh
```

**Shared Data**: Both models use `data/train_data.jsonl` (200) and `data/test_data.jsonl` (1000)

**Documentation**: See `README_MODELS.md` for detailed model comparison and usage.

---

### Core Scripts

| File | Description |
|------|-------------|
| `prepare_training_data.py` | Convert `dataset/smiles-data.xlsx` to training format (JSONL) |
| `lora_finetuning.py` | Train Qwen 7B model with LoRA on SMILES data |
| `inference.py` | Test the finetuned model on sample molecules |
| `active_learning_pipeline.py` | Main pipeline: predict → classify → human feedback loop |
| `model_predictor.py` | Load model and predict molecular scores with uncertainty |
| `confidence_calculator.py` | Calculate prediction confidence using entropy |
| `human_feedback.py` | Interactive interface for expert feedback |
| `test_pipeline.py` | Test suite to verify environment setup |

### Configuration

| File | Description |
|------|-------------|
| `config.yaml` | All configurable parameters (thresholds, rounds, sampling) |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules (excludes dataset, outputs, models) |

### Data & Output

| Directory | Description |
|-----------|-------------|
| `dataset/` | Training data (smiles-data.xlsx) and test set (1600set.xlsx) |
| `output/` | Results: target_smiles.json, unsure_smiles.json, predictions.json, feedback_history.json |

## ⚙️ Key Configuration (config.yaml)

```yaml
# Adjust these parameters as needed
thresholds:
  confidence_threshold: 8.0      # Confidence cutoff (0-10)
  target_score_threshold: 7.0    # Score cutoff for target molecules

human_feedback:
  num_pairs_per_round: 5         # Pairs to compare per round
  max_rounds: 10                 # Maximum feedback rounds

prediction:
  num_samples: 10                # Samples per molecule (affects confidence)
```

## 🎯 Human Feedback Interface

When prompted, choose:
- **A**: Molecule A has higher score
- **B**: Molecule B has higher score  
- **E**: Equal/Similar scores
- **S**: Skip this pair

## 📈 Output Files

After running, check `output/` directory:

| File | Content |
|------|---------|
| `target_smiles.json` | High-confidence, high-scoring molecules (confidence ≥8, score ≥7) |
| `unsure_smiles.json` | Low-confidence molecules that need more feedback |
| `predictions.json` | All predictions with scores, confidence, entropy, std |
| `feedback_history.json` | All human feedback records |

## 🔧 Common Adjustments

**Get more target molecules:**
```yaml
confidence_threshold: 7.0  # Lower from 8.0
target_score_threshold: 6.0  # Lower from 7.0
```

**Speed up prediction:**
```yaml
num_samples: 5  # Reduce from 10
num_pairs_per_round: 3  # Reduce from 5
```

**More accurate confidence:**
```yaml
num_samples: 20  # Increase from 10
```

## 📦 Requirements

- Python 3.10+
- GPU with 16GB+ memory (for Qwen 7B)
- See `requirements.txt` for all dependencies

## 🎉 Simple Workflow

```bash
# 1. Test environment
python3 test_pipeline.py

# 2. Train model (optional if model exists)
python3 lora_finetuning.py

# 3. Run active learning pipeline
python3 active_learning_pipeline.py
```

Results saved in `output/` directory. Enjoy! 🚀

