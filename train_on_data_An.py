"""
LoRA Finetuning Script for ChemLLM-7B model using SMILES molecular dataset
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# =====================================================
# Dataset Class
# =====================================================

class SMILESDataset:
    """Dataset class for SMILES molecular data"""
    
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(jsonl_file)
    
    def load_data(self, jsonl_file):
        """Load data from JSONL file"""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def format_conversation(self, messages):
        """Format messages into a single string for training"""
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return formatted
    
    def preprocess_data(self):
        """Preprocess data for training"""
        processed = []
        for item in self.data:
            text = self.format_conversation(item['messages'])
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            encoded['labels'] = encoded['input_ids'].copy()
            processed.append(encoded)
        return Dataset.from_list(processed)

# =====================================================
# LoRA Config
# =====================================================

def setup_lora_config():
    """Configure LoRA parameters"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "w1", "w2", "w3"],  # ChemLLM / Qwen1.5 compatible
    )

# =====================================================
# Training Function
# =====================================================

def train_model(
    model_name="AI4Chem/ChemLLM-7B-Chat",
    train_data_path="/content/Sai_AI4Drug/models/models/chemllm/dataset/train_data.jsonl",
    output_dir="/content/Sai_AI4Drug/models/models/chemllm/chemllm_lora_output",
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4,
    max_length=512
):
    """
    Finetune ChemLLM-7B with LoRA on SMILES dataset
    """

    print("="*80)
    print("🚀 FINETUNING CHEMLLM-7B WITH LORA")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Training data: {train_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*80)

    # Load tokenizer
    print("\n🔡 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Load model
    print("\n🧠 Loading ChemLLM model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("✅ Model loaded successfully!")

    # Apply LoRA configuration
    print("\n⚙️ Applying LoRA configuration...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\n📂 Loading training data from: {train_data_path}")
    dataset_loader = SMILESDataset(train_data_path, tokenizer, max_length)
    train_dataset = dataset_loader.preprocess_data()
    print(f"✅ Training samples: {len(train_dataset)}")

    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        fp16=True,
        save_strategy="epoch",
        logging_steps=10,
        warmup_steps=50,
        optim="adamw_torch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("\n🔥 Starting training...")
    trainer.train()

    # Save model
    print("\n💾 Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Training completed and model saved successfully!")

# =====================================================
# Main Entry
# =====================================================

if __name__ == "__main__":
    config = {
        "model_name": "AI4Chem/ChemLLM-7B-Chat",
        "train_data_path": "/content/Sai_AI4Drug/models/models/chemllm/dataset/train_data.jsonl",
        "output_dir": "/content/Sai_AI4Drug/models/models/chemllm/chemllm_lora_output",
        "num_epochs": 3,
        "batch_size": 2,
        "learning_rate": 2e-4,
        "max_length": 512
    }
    train_model(**config)


