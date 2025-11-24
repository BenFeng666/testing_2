"""
Multi-Task Training with Conditional Loss
Trains model to predict both toxicity and efficiency simultaneously
- Toxic data: for toxicity prediction (0 or 1)
- Efficiency data: for efficiency prediction (1-10 discrete values)
"""

import json
import torch
import torch.nn.functional as F
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import os
import re
import numpy as np


class MultiTaskSMILESDataset:
    """Dataset class for multi-task SMILES data (toxicity + efficiency)"""

    def __init__(self, toxic_data_path, efficiency_data_path, tokenizer, max_length=512):
        """
        Initialize dataset with both toxic and efficiency data

        Args:
            toxic_data_path: Path to toxic data JSONL file (contains toxicity labels 0/1)
            efficiency_data_path: Path to efficiency data JSONL file (contains efficiency scores 1-10)
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.toxic_data = self.load_data(toxic_data_path) if toxic_data_path else []
        self.efficiency_data = self.load_data(efficiency_data_path) if efficiency_data_path else []

    def load_data(self, jsonl_file):
        """Load data from JSONL file"""
        if not os.path.exists(jsonl_file):
            print(f"Warning: {jsonl_file} not found, skipping...")
            return []
        data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def extract_toxicity_label(self, assistant_message):
        """Extract toxicity label (0 or 1) from assistant message - MUST be 0 or 1"""
        toxicity = 0

        # Extract toxicity (0 or 1)
        tox_match = re.search(r"Toxicity value:\s*(\d+)", assistant_message)
        if tox_match:
            toxicity = int(tox_match.group(1))
            # Ensure it's 0 or 1 (discrete binary value)
            toxicity = 1 if toxicity >= 1 else 0

        # Check for toxic/non-toxic keywords
        lower = assistant_message.lower()
        if "non-toxic" in lower:
            toxicity = 0
        elif "toxic" in lower and "non-toxic" not in lower:
            toxicity = 1

        # Final check: ensure output is strictly 0 or 1
        return 1 if toxicity >= 1 else 0

    def extract_efficiency_label(self, assistant_message):
        """Extract efficiency label (1–10) from any numeric output, including floats."""
        match = re.search(r"[-+]?\d*\.?\d+", assistant_message)
        if not match:
            return None

        value = float(match.group())

        # Force discrete integer 1–10
        value = int(round(value))
        value = max(1, min(10, value))

        return value

    def format_conversation(self, messages):
        """Format messages into a single string for training"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return formatted

    def preprocess_data(self):
        """Preprocess data for training - combine toxic and efficiency data"""
        processed = []

        # Process toxic data (for toxicity prediction)
        print(f"Processing {len(self.toxic_data)} toxic data samples...")
        for item in self.toxic_data:
            text = self.format_conversation(item["messages"])
            assistant_msg = item["messages"][-1]["content"]
            toxicity = self.extract_toxicity_label(assistant_msg)

            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            encoded["labels"] = encoded["input_ids"].copy()
            encoded["toxicity_label"] = toxicity
            encoded["efficiency_label"] = None  # No efficiency label for toxic-only data
            encoded["data_type"] = "toxic"
            processed.append(encoded)

        # Process efficiency data (for efficiency prediction)
        print(f"Processing {len(self.efficiency_data)} efficiency data samples...")
        for item in self.efficiency_data:
            text = self.format_conversation(item["messages"])
            assistant_msg = item["messages"][-1]["content"]
            efficiency = self.extract_efficiency_label(assistant_msg)

            if efficiency is None:
                continue  # Skip if efficiency label not found

            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            encoded["labels"] = encoded["input_ids"].copy()
            encoded["toxicity_label"] = 0  # Assume non-toxic for efficiency data
            encoded["efficiency_label"] = efficiency
            encoded["data_type"] = "efficiency"
            processed.append(encoded)

        print(f"Total processed samples: {len(processed)}")
        print(f"  - Toxic samples: {len(self.toxic_data)}")
        print(
            f"  - Efficiency samples: {len([p for p in processed if p['data_type'] == 'efficiency'])}"
        )

        return Dataset.from_list(processed)


class MultiTaskDataCollator:
    """
    Custom data collator that:
    - stacks input_ids, attention_mask, labels
    - converts toxicity_label and efficiency_label into tensors
    - uses -1 to represent "no efficiency label"
    """

    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features):
        """
        features: list of dicts from Dataset
        Each has keys:
          input_ids, attention_mask, labels, toxicity_label, efficiency_label, data_type
        Some samples *might* miss toxicity_label/efficiency_label → we default them.
        """
        batch = {}

        # Stack sequence fields: already same length due to padding='max_length'
        batch["input_ids"] = torch.tensor(
            [f["input_ids"] for f in features], dtype=torch.long
        )
        batch["attention_mask"] = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )
        batch["labels"] = torch.tensor(
            [f["labels"] for f in features], dtype=torch.long
        )

        # Toxicity labels → float tensor [B], default 0.0 (non-toxic) if missing
        tox_list = []
        for f in features:
            tox_list.append(float(f.get("toxicity_label", 0.0)))
        batch["toxicity_label"] = torch.tensor(tox_list, dtype=torch.float32)

        # Efficiency labels → long tensor [B], -1 means "no label"
        effs = []
        for f in features:
            eff = f.get("efficiency_label", None)
            effs.append(-1 if eff is None else int(eff))
        batch["efficiency_label"] = torch.tensor(effs, dtype=torch.long)

        # DO NOT include data_type in batch (it is a list of strings, accelerator can't .to() it)
        return batch


class CheckpointEvaluationCallback(TrainerCallback):
    """Callback to evaluate model after each checkpoint save"""

    def __init__(self, eval_dataset, tokenizer, output_dir):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.results = []

    def on_save(self, args, state, control, model=None, **kwargs):
        """Called after each checkpoint save"""
        if state.global_step % args.save_steps == 0:
            print(f"\n{'=' * 80}")
            print(f"Evaluating checkpoint at step {state.global_step}")
            print(f"{'=' * 80}")

            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            print(f"Checkpoint saved at: {checkpoint_path}")
            print("Run 'python evaluate_checkpoints.py' after training to evaluate all checkpoints")


class MultiTaskTrainer(Trainer):
    """Custom Trainer with Conditional Multi-Task Loss"""

    def __init__(self, alpha=1.0, efficiency_num_classes=10, efficiency_eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.efficiency_num_classes = efficiency_num_classes
        self.efficiency_eps = efficiency_eps
        self.accuracy_history = []  # Store accuracy for each checkpoint

    def compute_metrics(self, eval_pred):
        # Metrics will be done by a separate script, so keep this empty for now
        return {}

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        metrics = output.metrics
        metrics[f"{metric_key_prefix}_toxicity_accuracy"] = 0.0
        metrics[f"{metric_key_prefix}_efficiency_accuracy"] = 0.0
        output.metrics = metrics
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Conditional Multi-Task Loss:
        - LM loss: causal LM token loss
        - Toxicity loss: BCEWithLogits over all samples
        - Efficiency loss: CE over classes 1–10, only for:
            * non-toxic samples (toxicity_label == 0)
            * efficiency_label in [1..10]
        Total: lm_loss + loss_tox + alpha * loss_eff
        """
        # Labels from batch
        labels = inputs.pop("labels")  # [B, T]
        toxicity_labels = inputs.pop("toxicity_label", None)  # [B]
        efficiency_labels = inputs.pop("efficiency_label", None)  # [B]

        # Forward pass with hidden states
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits  # [B, T, V]
        hidden_states = outputs.hidden_states[-1]  # [B, T, H]
        device = logits.device

        # 1. LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        batch_size, _, hidden_dim = hidden_states.size()

        # Last non-pad token hidden state per sample
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
            last_token_hidden = hidden_states[torch.arange(batch_size, device=device), seq_lengths]
        else:
            last_token_hidden = hidden_states[:, -1, :]  # [B, H]

        # 2. Toxicity head & loss
        if toxicity_labels is None:
            tox_labels_tensor = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            if not isinstance(toxicity_labels, torch.Tensor):
                tox_labels_tensor = torch.tensor(
                    toxicity_labels, dtype=torch.float32, device=device
                )
            else:
                tox_labels_tensor = toxicity_labels.to(device=device, dtype=torch.float32)

        tox_labels_tensor = tox_labels_tensor.clamp(0, 1)  # [0,1]
        toxicity_logits = last_token_hidden.mean(dim=-1, keepdim=True)  # [B, 1]

        loss_tox = F.binary_cross_entropy_with_logits(
            toxicity_logits.squeeze(-1),  # [B]
            tox_labels_tensor,  # [B]
        )

        # 3. Efficiency head & loss
        loss_eff = torch.tensor(0.0, device=device)

        if efficiency_labels is not None:
            if not isinstance(efficiency_labels, torch.Tensor):
                eff_labels_tensor = torch.tensor(efficiency_labels, dtype=torch.long, device=device)
            else:
                eff_labels_tensor = efficiency_labels.to(device=device, dtype=torch.long)

            # sentinel: -1 means "no label"
            has_eff = (eff_labels_tensor >= 1) & (eff_labels_tensor <= 10)
            non_toxic = tox_labels_tensor == 0
            mask = has_eff & non_toxic  # [B]

            if mask.any():
                # Project to 10 logits
                if hidden_dim >= self.efficiency_num_classes:
                    efficiency_logits = last_token_hidden[:, : self.efficiency_num_classes]  # [B, 10]
                    if hidden_dim > self.efficiency_num_classes:
                        bias = last_token_hidden[:, self.efficiency_num_classes :].mean(
                            dim=-1, keepdim=True
                        )
                        efficiency_logits = efficiency_logits + bias
                else:
                    padding = torch.zeros(
                        batch_size,
                        self.efficiency_num_classes - hidden_dim,
                        device=device,
                        dtype=last_token_hidden.dtype,
                    )
                    efficiency_logits = torch.cat([last_token_hidden, padding], dim=-1)  # [B, 10]

                eff_labels_0idx = (eff_labels_tensor - 1).clamp(
                    0, self.efficiency_num_classes - 1
                )

                eff_logits_masked = efficiency_logits[mask]  # [B_mask, 10]
                eff_labels_masked = eff_labels_0idx[mask]  # [B_mask]

                if eff_logits_masked.size(0) > 0:
                    loss_eff = F.cross_entropy(
                        eff_logits_masked,
                        eff_labels_masked,
                        reduction="mean",
                    )

        total_loss = lm_loss + loss_tox*100 + self.alpha * loss_eff

        # Logging
        if (
            hasattr(self.state, "global_step")
            and self.state.global_step % self.args.logging_steps == 0
        ):
            self.log(
                {
                    "loss": float(total_loss.detach().cpu()),
                    "lm_loss": float(lm_loss.detach().cpu()),
                    "toxicity_loss": float(loss_tox.detach().cpu()),
                    "efficiency_loss": float(loss_eff.detach().cpu()),
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss


def setup_lora_config(config):
    """Configure LoRA parameters from config"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
    )
    return lora_config


def train_multi_task_model(config_path="training_config.yaml"):
    """
    Train model with Conditional Multi-Task Loss using both toxic and efficiency data
    """
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("MULTI-TASK TRAINING WITH CONDITIONAL LOSS")
    print("=" * 80)
    print(f"Configuration loaded from: {config_path}")
    print(f"Alpha (efficiency loss weight): {config["loss"]["alpha"]}")
    print()

    # Load tokenizer
    print(f"Loading tokenizer: {config['model']['base_model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model_path"],
        trust_remote_code=True,
        padding_side="right",
        revision="main",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {config['model']['base_model_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_path"],
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=config["optimization"]["load_in_4bit"],
        torch_dtype=torch.bfloat16 if config["optimization"]["bf16"] else torch.float16,
        low_cpu_mem_usage=True,
    )

    # Prepare for kbit training
    if config["optimization"]["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset - load both toxic and efficiency data
    toxic_train_path = config["data"].get("toxic_train_data_path")
    efficiency_train_path = config["data"].get("efficiency_train_data_path")
    toxic_test_path = config["data"].get("toxic_test_data_path")
    efficiency_test_path = config["data"].get("efficiency_test_data_path")

    print("\nLoading training data:")
    print(f"  Toxic data: {toxic_train_path}")
    print(f"  Efficiency data: {efficiency_train_path}")

    train_dataset_loader = MultiTaskSMILESDataset(
        toxic_data_path=toxic_train_path,
        efficiency_data_path=efficiency_train_path,
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"],
    )
    train_dataset = train_dataset_loader.preprocess_data()
    print(f"Total training samples: {len(train_dataset)}")

    print("\nLoading evaluation data:")
    print(f"  Toxic test data: {toxic_test_path}")
    print(f"  Efficiency test data: {efficiency_test_path}")

    eval_dataset_loader = MultiTaskSMILESDataset(
        toxic_data_path=toxic_test_path,
        efficiency_data_path=efficiency_test_path,
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"],
    )
    eval_dataset = eval_dataset_loader.preprocess_data()
    print(f"Total evaluation samples: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=config["model"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        fp16=config["optimization"]["fp16"],
        bf16=config["optimization"]["bf16"],
        save_strategy=config["training"].get("save_strategy", "epoch"),
        save_steps=config["training"].get("save_steps", None),
        logging_steps=config["training"]["logging_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        optim=config["optimization"]["optim"],
        save_total_limit=config["training"].get("save_total_limit", 2),
        evaluation_strategy=config["training"].get("evaluation_strategy", "no"),
        eval_steps=config["training"].get("eval_steps", None),
        per_device_eval_batch_size=config["training"].get("per_device_eval_batch_size", 1),
        report_to="none",
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        max_grad_norm=config["training"]["max_grad_norm"],
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    eval_callback = CheckpointEvaluationCallback(
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        tokenizer=tokenizer,
        output_dir=config["model"]["output_dir"],
    )

    data_collator = MultiTaskDataCollator(tokenizer=tokenizer)

    trainer = MultiTaskTrainer(
        alpha=config["loss"]["alpha"],
        efficiency_num_classes=config["loss"]["efficiency_num_classes"],
        efficiency_eps=config["loss"]["efficiency_eps"],
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[eval_callback],
    )

    print("\nStarting training...")
    print("Loss configuration:")
    print(f"  - Toxicity loss: {config['loss']['toxicity_loss_type']} (all samples)")
    print(
        f"  - Efficiency loss: {config['loss']['efficiency_loss_type']} (only non-toxic samples, alpha={config['loss']['alpha']})"
    )
    print(f"  - Efficiency classes: {config['loss']['efficiency_num_classes']}")
    print("\nTraining settings:")
    print(f"  - Save every {config['training'].get('save_steps', 'N/A')} steps")
    print(f"  - Evaluate every {config['training'].get('eval_steps', 'N/A')} steps")
    print()

    trainer.train()

    print(f"\nSaving final model to: {config['model']['output_dir']}")
    model.save_pretrained(config["model"]["output_dir"])
    tokenizer.save_pretrained(config["model"]["output_dir"])

    print("\nTraining completed!")
    print("=" * 80)
    print("\nTo evaluate all checkpoints, run:")
    print(f"  python evaluate_checkpoints.py --config {config_path}")
    print("=" * 80)


if __name__ == "__main__":
    train_multi_task_model("training_config.yaml")
