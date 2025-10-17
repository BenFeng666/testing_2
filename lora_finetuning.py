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
# 1. Dataset Class
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
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"### System:\n{content}\n"
            elif role == "user":
                formatted += f"### User:\n{content}\n"
            elif role == "assistant":
                formatted += f"### Assistant:\n{content}\n"
        return formatted
    
    def preprocess_data(self):
        """Preprocess data for training"""
        processed = []
        for item in self.data:
            text = self.format_conversation(item["messages"])
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # ✅ Mask out padding tokens for loss computation
            labels = [
                token if token != self.tokenizer.pad_token_id else -100
                for token in input_ids
            ]

            processed.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
        return Dataset.from_list(processed)


# =====================================================
# 2. LoRA Config
# =====================================================
def setup_lora_config():
    """Configure LoRA parameters"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "w1", "w2", "w3"],  # Qwen/ChemLLM-style layers
    )


# =====================================================
# 3. Training Function
# =====================================================
def train_model(
    model_name="AI4Chem/ChemLLM-7B-Chat",
    train_data_path="dataset/train_data.jsonl",
    output_dir="./chemllm_lora_output",
    num_epochs=50,
    batch_size=2,
    learning_rate=2e-4,
    max_length=512
):
    """
    Finetune ChemLLM-7B model with LoRA on SMILES dataset
    """

    print(f"\n🔹 Loading tokenizer and model: {model_name}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # A100-friendly
    )

    # Apply LoRA configuration
    print("⚙️ Applying LoRA configuration...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()   # ✅ Required for LoRA backprop
    model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()

    # Load dataset
    print(f"📘 Loading training data from: {train_data_path}")
    dataset_loader = SMILESDataset(train_data_path, tokenizer, max_length)
    train_dataset = dataset_loader.preprocess_data()
    print(f"✅ Training samples: {len(train_dataset)}")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        bf16=True,                                 # Use bf16 for better A100/H100 efficiency
        save_strategy="epoch",                     # Save at each epoch
        logging_steps=10,                          # Log every 10 steps
        warmup_steps=100,                          # Gradual LR warm-up
        lr_scheduler_type="cosine",                # Cosine learning rate schedule
        optim="adamw_torch",                       # Standard AdamW optimizer
        save_total_limit=2,                        # Keep only 2 most recent checkpoints
        remove_unused_columns=False,               # Keep all columns (important for custom datasets)
        gradient_checkpointing=True,               # Save memory during training
        max_grad_norm=0.5,                         # Clip gradient norm
        report_to="none"                           # Disable wandb/tensorboard logging
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("\n🔥 Starting training...\n")
    trainer.train()

    # Save the final adapter
    print(f"\n💾 Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("\n✅ Training completed successfully!\n")


# =====================================================
# 4. Run Training
# =====================================================
if __name__ == "__main__":
    config = {
        "model_name": "AI4Chem/ChemLLM-7B-Chat",
        "train_data_path": "dataset/train_data.jsonl",
        "output_dir": "./chemllm_lora_output",
        "num_epochs": 70,
        "batch_size": 2,
        "learning_rate": 2e-4,
        "max_length": 512
    }

    train_model(**config)

