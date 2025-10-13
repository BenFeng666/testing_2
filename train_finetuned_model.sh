#!/bin/bash
# Script to finetune Qwen 7B on data_An training set

echo "======================================================================"
echo "FINETUNING QWEN 7B WITH LORA ON data_An/train_data.jsonl"
echo "======================================================================"

# Update lora_finetuning.py to use data_An data
python3 << 'EOF'
import re

# Read the lora_finetuning.py file
with open('lora_finetuning.py', 'r') as f:
    content = f.read()

# Update the training data path
content = re.sub(
    r'"train_data_path": "dataset/train_data.jsonl"',
    '"train_data_path": "data_An/train_data.jsonl"',
    content
)

# Update output directory
content = re.sub(
    r'"output_dir": "./qwen_lora_finetuned"',
    '"output_dir": "./qwen_lora_finetuned_An"',
    content
)

# Save updated file
with open('lora_finetuning_An.py', 'w') as f:
    f.write(content)

print("Created lora_finetuning_An.py with data_An configuration")
EOF

echo ""
echo "Starting LoRA finetuning..."
echo "Training data: data_An/train_data.jsonl (1100 samples)"
echo "Output: qwen_lora_finetuned_An/"
echo ""

# Run finetuning
python3 lora_finetuning_An.py

echo ""
echo "======================================================================"
echo "FINETUNING COMPLETED"
echo "======================================================================"
echo "Model saved to: qwen_lora_finetuned_An/"

