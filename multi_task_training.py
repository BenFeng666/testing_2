"""
Multi-Task Training with Conditional Loss
Trains model to predict both toxicity and efficiency simultaneously
"""

import json
import torch
import torch.nn.functional as F
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import re
import numpy as np


class MultiTaskSMILESDataset:
    """Dataset class for multi-task SMILES data (toxicity + efficiency)"""

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

    def extract_labels(self, assistant_message):
        """
        Extract toxicity and efficiency labels from assistant message

        Expected format: "This molecular structure {smiles} is {toxic/non-toxic}. Toxicity value: {0/1}."
        Efficiency is an integer score 1–10 if present.
        """
        toxicity = 0
        efficiency = None  # None if not found

        # Extract toxicity (0 or 1)
        tox_match = re.search(r'Toxicity value:\s*(\d+)', assistant_message)
        if tox_match:
            toxicity = int(tox_match.group(1))

        # Keyword check as fallback
        msg_lower = assistant_message.lower()
        if 'non-toxic' in msg_lower:
            toxicity = 0
        elif 'toxic' in msg_lower and 'non-toxic' not in msg_lower:
            toxicity = 1

        # Extract efficiency if available
        eff_patterns = [
            r'Efficiency[:\s]+(\d+)',
            r'efficiency[:\s]+(\d+)',
            r'Score[:\s]+(\d+)',
            r'score[:\s]+(\d+)',
        ]
        for pattern in eff_patterns:
            eff_match = re.search(pattern, assistant_message)
            if eff_match:
                efficiency = int(eff_match.group(1))
                if 1 <= efficiency <= 10:
                    break
                else:
                    efficiency = None

        return toxicity, efficiency

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
            # Format conversation
            text = self.format_conversation(item['messages'])

            # Extract labels
            assistant_msg = item['messages'][-1]['content']  # Last message is assistant
            toxicity, efficiency = self.extract_labels(assistant_msg)

            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )

            # Set labels for language modeling (standard causal LM)
            encoded['labels'] = encoded['input_ids'].copy()

            # Add multi-task labels
            # toxicity: 0/1
            encoded['toxicity_label'] = int(toxicity)

            # efficiency: 1–10, or -1 if not available (sentinel)
            if efficiency is None:
                encoded['efficiency_label'] = -1
            else:
                encoded['efficiency_label'] = int(efficiency)

            processed.append(encoded)

        return Dataset.from_list(processed)


class MultiTaskTrainer(Trainer):
    """Custom Trainer with Conditional Multi-Task Loss"""

    def __init__(self, alpha=1.0, efficiency_num_classes=10, efficiency_eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.efficiency_num_classes = efficiency_num_classes
        self.efficiency_eps = efficiency_eps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute Conditional Multi-Task Loss:
        1. Toxicity loss: binary_cross_entropy_with_logits
        2. Efficiency loss: cross_entropy (only for non-toxic samples with valid efficiency > 0)
        3. Total loss: lm_loss + loss_tox + alpha * loss_eff
        """
        # Extract labels
        labels = inputs.pop("labels")
        toxicity_labels = inputs.pop("toxicity_label")
        efficiency_labels = inputs.pop("efficiency_label")

        # Make sure these are tensors on the right device
        if not isinstance(toxicity_labels, torch.Tensor):
            toxicity_labels_tensor = torch.tensor(toxicity_labels, device=model.device, dtype=torch.float32)
        else:
            toxicity_labels_tensor = toxicity_labels.to(model.device).float()

        if not isinstance(efficiency_labels, torch.Tensor):
            efficiency_labels_tensor = torch.tensor(efficiency_labels, device=model.device, dtype=torch.long)
        else:
            efficiency_labels_tensor = efficiency_labels.to(model.device).long()

        # Forward pass with hidden states
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

        # Standard causal LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Get hidden states for the last non-padding token
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1  # index of last non-pad token
            batch_size = hidden_states.size(0)
            last_token_hidden = hidden_states[torch.arange(batch_size), seq_lengths]
        else:
            last_token_hidden = hidden_states[:, -1, :]

        hidden_dim = last_token_hidden.size(-1)

        # Simple "heads" from hidden state
        # Toxicity logits: scalar per sample
        toxicity_logits = last_token_hidden.mean(dim=-1, keepdim=True)

        # Efficiency logits: first num_classes dims + bias from the rest
        efficiency_projection = last_token_hidden[:, :self.efficiency_num_classes]
        if hidden_dim > self.efficiency_num_classes:
            efficiency_bias = last_token_hidden[:, self.efficiency_num_classes:].mean(dim=-1, keepdim=True)
            efficiency_logits = efficiency_projection + efficiency_bias
        else:
            efficiency_logits = efficiency_projection

        # 1. Toxicity loss
        loss_tox = F.binary_cross_entropy_with_logits(
            toxicity_logits.squeeze(-1),
            toxicity_labels_tensor
        )

        # 2. Efficiency loss — only for samples with valid efficiency ( > 0 ) and non-toxic (toxicity == 0)
        # valid efficiency labels: 1–10; sentinel is -1
        valid_eff_mask = (efficiency_labels_tensor > 0).float()
        non_toxic_mask = (toxicity_labels_tensor == 0).float()
        mask = (valid_eff_mask * non_toxic_mask)  # shape [batch]

        has_efficiency = mask.sum() > 0

        if has_efficiency:
            # map 1–10 -> 0–9
            efficiency_labels_0_indexed = (efficiency_labels_tensor - 1).clamp(0, self.efficiency_num_classes - 1)

            ce_all = F.cross_entropy(
                efficiency_logits,
                efficiency_labels_0_indexed,
                reduction='none'
            )

            # apply mask and normalize
            loss_eff = (ce_all * mask).sum() / (mask.sum() + self.efficiency_eps)
        else:
            loss_eff = torch.tensor(0.0, device=model.device)

        total_loss = lm_loss + loss_tox + self.alpha * loss_eff

        # Optional logging
        if hasattr(self.state, "global_step") and (self.state.global_step % max(1, self.args.logging_steps) == 0):
            self.log({
                "loss": total_loss.item(),
                "lm_loss": lm_loss.item(),
                "toxicity_loss": loss_tox.item(),
                "efficiency_loss": loss_eff.item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss


def setup_lora_config(config):
    """Configure LoRA parameters from config"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
    )
    return lora_config


def train_multi_task_model(config_path="training_config.yaml"):
    """
    Train model with Conditional Multi-Task Loss
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("MULTI-TASK TRAINING WITH CONDITIONAL LOSS")
    print("=" * 80)
    print(f"Configuration loaded from: {config_path}")
    print(f"Alpha (efficiency loss weight): {config['loss']['alpha']}")
    print()

    # Load tokenizer
    print(f"Loading tokenizer: {config['model']['base_model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model_path'],
        trust_remote_code=True,
        padding_side='right',
        revision="main"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {config['model']['base_model_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model_path'],
        trust_remote_code=True,
        device_map="auto",
        load_in_4bit=config['optimization']['load_in_4bit'],
        torch_dtype=torch.bfloat16 if config['optimization']['bf16'] else torch.float16,
        low_cpu_mem_usage=True,
    )

    # Prepare for k-bit training
    if config['optimization']['load_in_4bit']:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print(f"\nLoading training data: {config['data']['train_data_path']}")
    dataset_loader = MultiTaskSMILESDataset(
        config['data']['train_data_path'],
        tokenizer,
        config['data']['max_length']
    )
    train_dataset = dataset_loader.preprocess_data()
    print(f"Training samples: {len(train_dataset)}")

    # Training arguments
    lr = float(config["training"]["learning_rate"])
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=lr,
        fp16=config['optimization']['fp16'],
        bf16=config['optimization']['bf16'],
        save_strategy=config['training']['save_strategy'],
        logging_steps=config['training']['logging_steps'],
        warmup_steps=config['training']['warmup_steps'],
        optim=config['optimization']['optim'],
        save_total_limit=config['training']['save_total_limit'],
        report_to="none",
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        max_grad_norm=config['training']['max_grad_norm'],
        remove_unused_columns=False,
    )

    # Initialize custom trainer
    trainer = MultiTaskTrainer(
        alpha=config['loss']['alpha'],
        efficiency_num_classes=config['loss']['efficiency_num_classes'],
        efficiency_eps=config['loss']['efficiency_eps'],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    print("\nStarting training...")
    print("Loss configuration:")
    print(f"  - Toxicity loss: {config['loss']['toxicity_loss_type']}")
    print(f"  - Efficiency loss: {config['loss']['efficiency_loss_type']} (alpha={config['loss']['alpha']})")
    print(f"  - Efficiency classes: {config['loss']['efficiency_num_classes']}")
    print()

    trainer.train()

    # Save model
    print(f"\nSaving model to: {config['model']['output_dir']}")
    model.save_pretrained(config['model']['output_dir'])
    tokenizer.save_pretrained(config['model']['output_dir'])

    print("\nTraining completed!")
    print("=" * 80)


if __name__ == "__main__":
    train_multi_task_model("training_config.yaml")
