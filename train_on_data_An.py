"""
LoRA Finetuning script for Qwen 7B model - Using data_An dataset
Train on 1100 samples, evaluate on 100 test samples
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

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

def setup_lora_config():
    """Configure LoRA parameters"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # LoRA rank
        lora_alpha=32,  # LoRA alpha parameter
        lora_dropout=0.1,  # Dropout probability
        target_modules=["c_attn", "c_proj", "w1", "w2"],  # Qwen specific modules
    )
    return lora_config

def train_model(
    model_name="Qwen/Qwen-7B-Chat",
    train_data_path="data_An/train_data.jsonl",
    output_dir="./qwen_lora_finetuned_An",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_length=512
):
    """
    Finetune Qwen 7B model with LoRA on SMILES data
    """
    
    print("="*80)
    print("FINETUNING QWEN-7B WITH LORA ON data_An DATASET")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Training data: {train_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*80)
    
    print(f"\nLoading tokenizer and model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # Load model
    print(f"Loading base model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    print(f"Base model loaded successfully!")
    
    # Apply LoRA configuration
    print("\nApplying LoRA configuration...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print(f"\nLoading training data from: {train_data_path}")
    dataset_loader = SMILESDataset(train_data_path, tokenizer, max_length)
    train_dataset = dataset_loader.preprocess_data()
    print(f"Training samples: {len(train_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        save_strategy="epoch",
        logging_steps=10,
        warmup_steps=100,
        optim="adamw_torch",
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    trainer.train()
    
    # Save the final model
    print(f"\n" + "="*80)
    print(f"TRAINING COMPLETED")
    print("="*80)
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel and tokenizer saved successfully!")
    print(f"You can now use this model for inference or evaluation.")
    print("="*80)

if __name__ == "__main__":
    # Configuration for data_An dataset
    config = {
        "model_name": "Qwen/Qwen-7B-Chat",
        "train_data_path": "data_An/train_data.jsonl",  # 1100 samples
        "output_dir": "./qwen_lora_finetuned_An",
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "max_length": 512
    }
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*80)
    
    # Train the model
    train_model(**config)

