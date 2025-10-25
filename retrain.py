"""
LoRA Incremental Finetuning Script for ChemLLM-7B using SMILES dataset
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# =====================================================
# 1) Dataset
# =====================================================
class SMILESDataset:
    """Dataset class for SMILES molecular data"""

    def __init__(self, json_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(json_file)

    def load_data(self, json_file):
        """Load data from either .json (array) or .jsonl (one JSON per line)"""
        with open(json_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if text.startswith("["):
            data = json.loads(text)  # JSON array
        else:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]  # JSONL
        return data

    def format_conversation(self, messages):
        """Format messages into a single string for training"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"### System:\n{content}\n")
            elif role == "user":
                parts.append(f"### User:\n{content}\n")
            elif role == "assistant":
                parts.append(f"### Assistant:\n{content}\n")
        return "".join(parts)

    def preprocess_data(self):
        """Tokenize & prepare labels"""
        rows = []
        for item in self.data:
            txt = self.format_conversation(item["messages"])
            enc = self.tokenizer(
                txt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            labels = [tid if tid != self.tokenizer.pad_token_id else -100 for tid in input_ids]
            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        return Dataset.from_list(rows)


# =====================================================
# (Optional) LoRA config helper (kept for reference)
# =====================================================
def setup_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.15,
        target_modules=["c_attn", "c_proj", "w1", "w2", "w3"],  # Qwen/ChemLLM-style
    )


# =====================================================
# 2) Continue training from existing LoRA adapter
# =====================================================
if __name__ == "__main__":
    config = {
        "base_model_name": "AI4Chem/ChemLLM-7B-Chat",
        "prev_lora_path": "/content/testing_2/chemllm_lora_output",  # existing adapter dir
        "train_data_path": "/content/testing_2/dataset/training_2_sample.json",
        "output_dir": "/content/testing_2/chemllm_lora_retrain_output",
        "num_epochs": 20,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "max_length": 512,
    }

    print(f"\n🔹 Loading previous LoRA adapter from {config['prev_lora_path']} ...\n")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model_name"], trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model_name"],
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Attach existing LoRA adapter
    model = PeftModel.from_pretrained(base_model, config["prev_lora_path"])

    # Ensure ONLY LoRA params are trainable
    for name, param in model.named_parameters():
        if ("lora_" in name) or ("lora_A" in name) or ("lora_B" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Activate adapter if needed
    try:
        if hasattr(model, "peft_config") and len(model.peft_config) > 0:
            ad_name = next(iter(model.peft_config.keys()))
            if hasattr(model, "set_adapter"):
                model.set_adapter(ad_name)
        if hasattr(model, "enable_adapters"):
            model.enable_adapters()
    except Exception:
        pass

    # Gradient checkpointing compatibility
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.train()
    model.print_trainable_parameters()  # should not be 0

    # Dataset
    print(f"📘 Loading training data from: {config['train_data_path']}")
    dataset_loader = SMILESDataset(config["train_data_path"], tokenizer, config["max_length"])
    train_dataset = dataset_loader.preprocess_data()
    print(f"✅ Training samples: {len(train_dataset)}")

    # Training args
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=8,
        learning_rate=config["learning_rate"],
        bf16=True,
        save_strategy="epoch",
        logging_steps=10,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        save_total_limit=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("\n🔥 Continuing LoRA fine-tuning on focused dataset…\n")
    trainer.train()

    print(f"\n💾 Saving updated adapter to: {config['output_dir']}")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print("\n✅ Incremental LoRA retraining completed!\n")



