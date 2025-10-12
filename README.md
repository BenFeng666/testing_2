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

## 📊 Workflow

### Training Module
```
smiles-data.xlsx → prepare_training_data.py → train_data.jsonl 
→ lora_finetuning.py → qwen_lora_finetuned/ → inference.py
```

### Active Learning Module
```
1600set.xlsx → Predict all molecules → Calculate confidence
↓
Classify:
  - High confidence + High score → target_smiles.json
  - Low confidence → unsure_smiles.json
↓
Human Feedback Loop (max 10 rounds):
  - Show 5 pairs of uncertain molecules
  - Expert judges which has higher score
  - Re-predict uncertain molecules
  - Update classification
↓
Final results in output/
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

## 🐛 Troubleshooting

**Test failed?**
```bash
python3 test_pipeline.py  # Shows missing dependencies
```

**GPU memory error?**
- Reduce `batch_size` in `lora_finetuning.py`
- Reduce `num_samples` in `config.yaml`

**All confidence scores low?**
- Increase `num_samples` to 15-20
- Check model training quality

## 📦 Requirements

- Python 3.10+
- GPU with 16GB+ memory (for Qwen 7B)
- See `requirements.txt` for packages

## 📊 Technical Details

**Confidence Calculation**: Based on entropy of prediction distribution
- 10 predictions same → entropy=0 → confidence=10 (very certain)
- 10 predictions spread → entropy=max → confidence=0 (very uncertain)

**Model**: Qwen 7B with LoRA finetuning
**Training Data**: 301 SMILES structures with scores
**Test Set**: 12,276 molecules

## 🎉 That's It!

```bash
# Simple workflow:
python3 test_pipeline.py           # Verify setup
python3 lora_finetuning.py         # Train (optional if model exists)
python3 active_learning_pipeline.py # Run pipeline
```

Results saved in `output/` directory. Enjoy! 🚀
