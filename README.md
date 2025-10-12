# AI4Drug - Molecular Score Prediction with Human Feedback

A complete AI-powered molecular scoring system with LoRA finetuning and human-in-the-loop active learning for drug discovery.

## 🎯 Overview

This project provides two integrated systems:

1. **LoRA Finetuning Module**: Finetune Qwen 7B model on molecular SMILES data
2. **Human Feedback Active Learning Module**: Predict molecular scores with confidence estimation and human expert feedback

### Key Features

- ✅ Finetune Qwen 7B using LoRA on SMILES molecular structures
- ✅ Batch prediction of molecular scores (1-10 scale)
- ✅ Entropy-based confidence calculation (0-10 scale)
- ✅ Automatic classification of high-quality molecules
- ✅ Interactive human feedback loop for uncertain predictions
- ✅ Fully configurable via YAML
- ✅ Complete test suite

## 📁 Project Structure

```
AI4Drug/
├── README.md                        # This file - complete documentation
├── config.yaml                      # Configuration file (modify all parameters here)
├── requirements.txt                 # Python dependencies
│
├── dataset/
│   ├── smiles-data.xlsx            # Training data (301 samples)
│   └── 1600set.xlsx                # Test set (12,276 molecules)
│
├── Training Module/
│   ├── prepare_training_data.py    # Convert Excel to training format
│   ├── lora_finetuning.py          # Main LoRA training script
│   └── inference.py                # Test finetuned model
│
├── Active Learning Module/
│   ├── active_learning_pipeline.py # Main pipeline
│   ├── model_predictor.py          # Model predictions with uncertainty
│   ├── confidence_calculator.py    # Entropy-based confidence scoring
│   ├── human_feedback.py           # Human feedback interface
│   └── test_pipeline.py            # Test script
│
├── Documentation/
│   ├── USER_GUIDE.md               # Detailed user guide
│   ├── SYSTEM_SUMMARY.md           # Complete system summary
│   ├── README_TRAINING.md          # Training module details
│   ├── README_HUMAN_FEEDBACK.md    # Active learning details
│   └── QUICKSTART.md               # Quick start for training
│
└── output/                          # Results directory (auto-created)
    ├── target_smiles.json          # High-confidence target molecules
    ├── unsure_smiles.json          # Uncertain molecules
    ├── predictions.json            # All predictions
    └── feedback_history.json       # Human feedback records
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- GPU with 16GB+ memory (for Qwen 7B)
- CUDA-capable environment

### Installation

```bash
# Clone or navigate to project directory
cd /home/li003385/workspace/AI4Drug

# Install dependencies
pip3 install -r requirements.txt

# Test environment
python3 test_pipeline.py
```

### Option 1: Train Your Own Model

```bash
# Step 1: Prepare training data (already done - 301 samples)
python3 prepare_training_data.py

# Step 2: Train model with LoRA
python3 lora_finetuning.py

# Step 3: Test the model
python3 inference.py
```

### Option 2: Use Active Learning Pipeline

```bash
# Step 1: Modify config if needed
nano config.yaml

# Step 2: Run the pipeline
python3 active_learning_pipeline.py

# Step 3: Provide human feedback when prompted
# Step 4: Check results in output/ directory
ls -lh output/
```

---

## 📚 Part 1: LoRA Finetuning Module

### Training Data

- **Source**: `dataset/smiles-data.xlsx`
- **Samples**: 301 molecular structures with scores
- **Format**: Conversational JSONL for Qwen
- **Output**: `dataset/train_data.jsonl`

### Training Configuration

Default settings in `lora_finetuning.py`:

```python
config = {
    "model_name": "Qwen/Qwen-7B-Chat",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1
}
```

### Run Training

```bash
python3 lora_finetuning.py
```

**Expected Time**: ~10-20 minutes on RTX 3090 for 301 samples

**Output**: LoRA weights saved to `./qwen_lora_finetuned/`

### Test Trained Model

```bash
python3 inference.py
```

### Training Tips

**For Small Datasets (<500 samples)**:
- Increase epochs to 5-10
- Use lower learning rate (1e-4)
- Monitor for overfitting

**For GPU Memory Issues**:
- Reduce `batch_size` to 1-2
- Increase `gradient_accumulation_steps`
- Reduce `max_length` to 256

---

## 📚 Part 2: Human Feedback Active Learning Module

### System Workflow

```
1. Load Test Set (1600set.xlsx)
   ↓
2. Initial Prediction (all molecules)
   - Predict score 10 times per molecule
   - Calculate mean score and confidence
   ↓
3. Automatic Classification
   - High confidence (≥8) + High score (≥7) → Target SMILES
   - Low confidence (<8) → Unsure SMILES
   ↓
4. Human Feedback Loop (up to 10 rounds)
   - Select 5 most uncertain pairs
   - Expert compares and judges
   - Re-predict unsure molecules
   - Update classification
   ↓
5. Save Results
   - target_smiles.json
   - unsure_smiles.json
   - predictions.json
   - feedback_history.json
```

### Configuration (config.yaml)

```yaml
# Thresholds - MODIFY THESE AS NEEDED
thresholds:
  confidence_threshold: 8.0      # Confidence cutoff (0-10)
  target_score_threshold: 7.0    # Score cutoff (1-10)

# Human Feedback Settings
human_feedback:
  num_pairs_per_round: 5         # Pairs to show per round
  max_rounds: 10                 # Maximum feedback rounds
  selection_strategy: "entropy"  # How to select pairs

# Prediction Settings
prediction:
  num_samples: 10                # Samples per molecule
  temperature: 0.7               # Sampling temperature
  top_p: 0.8                     # Top-p sampling
```

### Run Active Learning

```bash
python3 active_learning_pipeline.py
```

### Human Feedback Interface

You'll see prompts like this:

```
================================================================================
HUMAN FEEDBACK ROUND 1
================================================================================

PAIR 1/5
--------------------------------------------------------------------------------
Molecule A:
  SMILES: CCCCCCCCCC(=O)OCCCC...
  Predicted Score: 6.50
  Confidence: 7.20

Molecule B:
  SMILES: CCCCCCCCCCCCCCCCCNC...
  Predicted Score: 7.10
  Confidence: 6.80

Which molecule should have a HIGHER score?
  Enter 'A' for Molecule A
  Enter 'B' for Molecule B
  Enter 'E' for Equal/Similar
  Enter 'S' to Skip this pair
Your choice: _
```

**How to Judge**:
- Base on your domain expertise (drug properties, efficacy, safety, etc.)
- 'A' if Molecule A should score higher
- 'B' if Molecule B should score higher
- 'E' if both are similar
- 'S' to skip if uncertain

### Output Files

After completion, check `output/` directory:

```bash
# View target molecules
cat output/target_smiles.json

# View uncertain molecules
cat output/unsure_smiles.json

# View all predictions
cat output/predictions.json

# View feedback history
cat output/feedback_history.json
```

---

## 🎓 Understanding Confidence Scores

### Entropy-Based Calculation

The system uses entropy to measure prediction uncertainty:

1. **Multiple Sampling**: Predict each molecule 10 times (configurable)
2. **Distribution**: Calculate score distribution histogram
3. **Entropy**: H = -Σ p(x) * log₂(p(x))
4. **Confidence**: (1 - H/H_max) * 10

### Examples

| Prediction Samples | Entropy | Confidence | Interpretation |
|-------------------|---------|------------|----------------|
| [8,8,8,8,8,8,8,8,8,8] | 0.0 | 10.0 | Very certain |
| [7,7,8,8,8,8,8,9,9,7] | 1.48 | 5.5 | Moderately uncertain |
| [1,2,3,4,5,6,7,8,9,10] | 3.32 | 0.0 | Very uncertain |

**Key Insight**: Low entropy = concentrated predictions = high confidence

---

## ⚙️ Configuration Guide

### Adjusting Thresholds

**Get More Target Molecules**:
```yaml
thresholds:
  confidence_threshold: 7.0  # Lower from 8.0
  target_score_threshold: 6.0  # Lower from 7.0
```

**Get Fewer but Higher Quality Targets**:
```yaml
thresholds:
  confidence_threshold: 9.0  # Higher from 8.0
  target_score_threshold: 8.0  # Higher from 7.0
```

### Adjusting Prediction Quality

**More Accurate (but slower)**:
```yaml
prediction:
  num_samples: 20  # More samples
  temperature: 0.5  # Less randomness
```

**Faster (but less accurate)**:
```yaml
prediction:
  num_samples: 5  # Fewer samples
  temperature: 0.9  # More randomness
```

### Adjusting Feedback

**More Feedback per Round**:
```yaml
human_feedback:
  num_pairs_per_round: 10  # Show more pairs
  max_rounds: 20  # Allow more rounds
```

**Less Time Commitment**:
```yaml
human_feedback:
  num_pairs_per_round: 3  # Show fewer pairs
  max_rounds: 5  # Fewer rounds
```

---

## 📊 Expected Results

### Training Module
- **Input**: 301 SMILES structures with scores
- **Output**: Finetuned Qwen 7B model with LoRA weights
- **Time**: ~10-20 minutes on GPU

### Active Learning Module

Based on 12,276 molecule test set:

**Initial Prediction**:
- Target SMILES: 20-40% (depends on model quality)
- Unsure SMILES: 60-80%

**After 10 Feedback Rounds**:
- Target SMILES: 40-60%
- Unsure SMILES: 40-60%
- Human annotations: ~50 comparisons

---

## 🔧 Troubleshooting

### Test Failures

Run diagnostics:
```bash
python3 test_pipeline.py
```

Common issues:
- Missing dependencies → Install with pip
- Wrong Python version → Use Python 3.10+
- GPU not detected → Check CUDA installation

### Model Loading Errors

Ensure model exists:
```bash
ls -lh qwen_lora_finetuned/
```

If not found, train the model first:
```bash
python3 lora_finetuning.py
```

### Data Loading Errors

Check data files exist:
```bash
ls -lh dataset/smiles-data.xlsx
ls -lh dataset/1600set.xlsx
```

### GPU Memory Issues

For training:
- Reduce `batch_size` in `lora_finetuning.py`
- Increase `gradient_accumulation_steps`

For prediction:
- Reduce `num_samples` in `config.yaml`
- Process fewer molecules at once

### All Confidence Scores Low

Possible causes:
1. Model not well-trained → Retrain with more data/epochs
2. Test set very different from training set → Check data distribution
3. `num_samples` too low → Increase to 15-20

---

## 💡 Best Practices

### For Training

1. **Data Quality**: Ensure training data is clean and representative
2. **Validation**: Hold out test set for validation
3. **Hyperparameters**: Start with defaults, then tune
4. **Monitoring**: Watch training loss and validation metrics

### For Active Learning

1. **First Run**: Use default settings to understand system behavior
2. **Threshold Tuning**: Adjust based on initial results
3. **Feedback Strategy**: Skip (S) when truly uncertain, use (E) for similar molecules
4. **Batch Processing**: Process large test sets in chunks
5. **Regular Saves**: Results auto-save after each round

### For Configuration

1. **Start Conservative**: High thresholds → high quality
2. **Iterate**: Lower thresholds if too few targets
3. **Balance**: Trade off between quantity and quality
4. **Document**: Keep notes on what settings work best

---

## 📈 Performance Optimization

### Speed Up Training
- Use smaller `max_length` (512 → 256)
- Reduce `num_epochs` for quick tests
- Use gradient accumulation instead of large batch sizes

### Speed Up Prediction
- Reduce `num_samples` (10 → 5)
- Lower `num_pairs_per_round` (5 → 3)
- Process subset of test set first

### Improve Accuracy
- Increase `num_samples` (10 → 20)
- Lower `temperature` (0.7 → 0.5)
- Use more training data
- Train for more epochs

---

## 🎓 Advanced Usage

### Custom Model Prompts

Edit system prompt in `model_predictor.py`:

```python
"content": "You are a helpful assistant specialized in [YOUR DOMAIN]..."
```

### Custom Selection Strategy

Add new strategy in `human_feedback.py`:

```python
def select_pairs(self, unsure_molecules, strategy='custom'):
    if strategy == 'custom':
        # Your custom logic here
        pass
```

### Custom Confidence Calculation

Modify `confidence_calculator.py`:

```python
def entropy_to_confidence(self, ent):
    # Your custom formula here
    pass
```

### Integration with Other Models

Adapt `model_predictor.py` to work with other models:
- Change model loading code
- Adjust prompt format
- Modify response parsing

---

## 📖 Additional Documentation

For more detailed information, see:

- **USER_GUIDE.md**: Complete user guide with examples
- **SYSTEM_SUMMARY.md**: Technical system overview
- **README_TRAINING.md**: Detailed training documentation
- **README_HUMAN_FEEDBACK.md**: Active learning technical details
- **QUICKSTART.md**: Quick start for training module

---

## 🧪 Testing

### Run Complete Test Suite

```bash
python3 test_pipeline.py
```

Expected output:
```
✓ PASS - Imports
✓ PASS - Configuration
✓ PASS - Confidence Calculator
✓ PASS - Data Loading
✓ PASS - Human Feedback
✓ PASS - Output Directory

Total: 6/6 tests passed
```

### Test Individual Components

```bash
# Test confidence calculator
python3 confidence_calculator.py

# Test human feedback (simulation)
python3 human_feedback.py

# Test model predictor (requires model)
python3 model_predictor.py
```

---

## 📦 Dependencies

Core requirements:
- pandas ≥ 2.0.0
- openpyxl ≥ 3.1.0
- numpy ≥ 1.23.0
- scipy ≥ 1.10.0
- pyyaml ≥ 6.0

Deep learning (for training and prediction):
- torch ≥ 2.0.0
- transformers ≥ 4.35.0
- peft ≥ 0.7.0
- datasets ≥ 2.14.0
- accelerate ≥ 0.24.0

Install all:
```bash
pip3 install -r requirements.txt
```

---

## 🔬 Use Cases

### Drug Discovery
- Screen large compound libraries
- Prioritize candidates for synthesis
- Identify molecules with desired properties

### Active Learning
- Efficiently label uncertain predictions
- Reduce human annotation effort
- Improve model iteratively

### Quality Control
- Identify high-confidence predictions
- Flag uncertain molecules for review
- Track prediction reliability

---

## 🎯 Workflow Examples

### Example 1: Quick Screening

```bash
# Use high thresholds for quick high-quality hits
# Edit config.yaml:
#   confidence_threshold: 9.0
#   target_score_threshold: 8.0

python3 active_learning_pipeline.py
```

### Example 2: Comprehensive Analysis

```bash
# Use lower thresholds, more feedback rounds
# Edit config.yaml:
#   confidence_threshold: 7.0
#   target_score_threshold: 6.0
#   max_rounds: 20

python3 active_learning_pipeline.py
```

### Example 3: Automated (No Feedback)

```bash
# Skip human feedback
# Edit config.yaml:
#   num_pairs_per_round: 0

python3 active_learning_pipeline.py
```

---

## 📊 Results Analysis

After running, analyze results:

```bash
# Count target molecules
jq length output/target_smiles.json

# Get average confidence
jq '[.[].confidence] | add/length' output/target_smiles.json

# Get score distribution
jq '[.[].predicted_score] | group_by(.) | map({score: .[0], count: length})' output/predictions.json

# View feedback summary
jq 'group_by(.human_choice) | map({choice: .[0].human_choice, count: length})' output/feedback_history.json
```

---

## 🤝 Contributing

To extend this system:

1. **Add new features**: Modify respective modules
2. **Test thoroughly**: Update `test_pipeline.py`
3. **Document**: Update relevant documentation
4. **Configuration**: Add parameters to `config.yaml`

---

## 📄 License

This project is for research and educational purposes.

---

## 🙏 Acknowledgments

- Based on Qwen 7B model
- Uses LoRA for efficient finetuning
- Implements entropy-based uncertainty estimation

---

## 📞 Support

For issues or questions:

1. Check documentation in `Documentation/` folder
2. Run test suite: `python3 test_pipeline.py`
3. Review example outputs in `output/` directory
4. Check configuration: `cat config.yaml`

---

## 🎉 Summary

This project provides a complete pipeline for molecular score prediction:

✅ **Train** custom models on your SMILES data
✅ **Predict** molecular scores with confidence estimates
✅ **Classify** molecules automatically by quality
✅ **Collect** human expert feedback efficiently
✅ **Iterate** to improve predictions
✅ **Export** results for downstream analysis

**Get Started Now:**

```bash
# Test your environment
python3 test_pipeline.py

# Run the pipeline
python3 active_learning_pipeline.py
```

**Happy Drug Discovery! 🚀**

---

*Last Updated: October 2025*

