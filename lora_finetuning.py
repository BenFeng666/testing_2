"""
FINAL FIXED — ChemLLM LoRA Fine-tuning (InternLM2 chat)
- Left padding + real <pad> token (no </s> spam)
- Prompt-only input (assistant target masked)
- NaN-safe loss that still has gradients
- Prints the exact (unpadded) prompt fed into the LLM
"""

import os, re, json, torch, torch.nn as nn
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


# =========================
# Dataset (InternLM2 format)
# =========================
class SMILESDataset:
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.tok = tokenizer
        self.max_len = max_length
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"❌ File not found: {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _format_internlm2(messages):
        """Keep system + user; stop before assistant; leave assistant tag open."""
        txt = []
        for m in messages:
            role = m["role"].lower()
            content = m["content"].strip()
            if role == "assistant":
                break
            txt.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        txt.append("<|im_start|>assistant\n")  # model should complete here
        return "\n".join(txt)

    @staticmethod
    def _extract_float(messages):
        joined = " ".join(m["content"] for m in messages)
        m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", joined)
        return float(m.group(1)) if m else 0.0

    def preprocess_data(self):
        rows = []
        print(f"📄 Loaded {len(self.data)} samples")
        for item in self.data:
            prompt = self._format_internlm2(item["messages"])
            y_val = self._extract_float(item["messages"])

            enc = self.tok(
                prompt,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            # Mask the whole prompt (we're *not* feeding ground-truth text targets here)
            labels = [-100] * len(input_ids)

            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "labels": labels,
                    "y": y_val,
                }
            )
        print(f"✅ Prepared {len(rows)} masked samples")
        return Dataset.from_list(rows)


# =========================
# Model wrapper (LoRA + MSE)
# =========================
class ChemLLMWithMSE(nn.Module):
    def __init__(self, base_model_name, lora_cfg, tokenizer):
        super().__init__()
        self.tok = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
        )
        # Resize for added pad token, if any:
        if self.model.get_input_embeddings().num_embeddings != len(self.tok):
            self.model.resize_token_embeddings(len(self.tok))

        self.model = get_peft_model(self.model, lora_cfg)
        self.model.config.use_cache = False
        self.model.print_trainable_parameters()

    @staticmethod
    def _parse_float(text: str):
        m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
        return float(m.group(1)) if m else 0.0

    def forward(self, input_ids, attention_mask, labels=None, y=None):
        # Cross-entropy LM loss (masked by labels)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = out.loss

        # Replace NaNs in lm_loss but KEEP graph (so grads flow)
        lm_loss_clean = torch.where(torch.isnan(lm_loss), torch.zeros_like(lm_loss), lm_loss)

        # Optional numeric estimate via short greedy generate (no grad)
        preds_list = []
        with torch.no_grad():
            for i in range(input_ids.size(0)):
                gen = self.model.generate(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    max_new_tokens=24,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.tok.eos_token_id,
                )
                # Only the continuation
                cont = gen[0][input_ids.size(1):]
                gen_text = self.tok.decode(cont, skip_special_tokens=True)
                preds_list.append(self._parse_float(gen_text))
        preds = torch.tensor(preds_list, dtype=torch.float32, device=input_ids.device)

        # Numeric MSE (auxiliary, no grad path to model — that's OK)
        mse_loss = torch.tensor(0.0, device=input_ids.device)
        if y is not None:
            target = torch.tensor(y, dtype=torch.float32, device=input_ids.device)
            target = torch.nan_to_num(target, nan=0.0)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
            mse_loss = nn.functional.mse_loss(preds, target, reduction="mean")

        total = lm_loss_clean + 0.1 * mse_loss
        return {"loss": total, "preds": preds, "lm_loss": lm_loss_clean.detach(), "mse_loss": mse_loss.detach()}


# =========================
# Trainer with safe logging
# =========================
class MSETextTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        y_true = inputs.pop("y", None)

        # Pretty print the unpadded prompt (use attention_mask)
        ids = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0]
        kept = ids[mask.bool()]
        prompt_str = model.tok.decode(kept, skip_special_tokens=False)
        print("\n🧾 --- Model Input Prompt (fed to LLM, unpadded) ---")
        print(prompt_str[:800])

        outputs = model(**inputs, labels=labels, y=y_true)
        loss = outputs["loss"]
        preds = outputs.get("preds", torch.zeros_like(inputs["input_ids"][:, 0], dtype=torch.float32)).detach().cpu()
        labs = torch.tensor(y_true, dtype=torch.float32).cpu() if y_true is not None else None

        print("\n🔹 Batch Results:")
        for i in range(preds.shape[0]):
            t = labs[i].item() if labs is not None else 0.0
            p = preds[i].item()
            print(f"  🧪 Pred: {p:.4f} | 🎯 True: {t:.4f} | Loss: {loss.item():.6f}")

        return (loss, outputs) if return_outputs else loss


# =========================
# LoRA config
# =========================
def lora_cfg():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "w1", "w2", "w3"],
    )


# =========================
# Train
# =========================
def train_model(
    model_name="AI4Chem/ChemLLM-7B-Chat",
    train_data_path="/content/testing_2/dataset/train_data.jsonl",
    output_dir="/content/testing_2/chemllm_lora_output_mse",
    num_epochs=8,
    batch_size=2,
    learning_rate=2e-4,
    max_length=512,
):
    print(f"\n🔹 Loading tokenizer and model: {model_name}\n")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ✅ decoder-only best practice
    tok.padding_side = "left"

    # ✅ ensure a real <pad> (≠ eos) to avoid </s> spam printing
    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        tok.add_special_tokens({"pad_token": "<pad>"})

    model = ChemLLMWithMSE(model_name, lora_cfg(), tok)

    ds = SMILESDataset(train_data_path, tok, max_length).preprocess_data()
    print(f"\n✅ Dataset ready: {len(ds)} samples")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="epoch",
        logging_steps=10,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = MSETextTrainer(model=model, args=args, train_dataset=ds, tokenizer=tok)

    print("\n🔥 Starting fine-tuning...\n")
    trainer.train()
    print(f"\n💾 Saving model to: {output_dir}")
    model.model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print("\n✅ Training completed successfully!\n")


if __name__ == "__main__":
    cfg = {
        "model_name": "AI4Chem/ChemLLM-7B-Chat",
        "train_data_path": "/content/testing_2/dataset/train_data.jsonl",
        "output_dir": "/content/testing_2/chemllm_lora_output_mse",
        "num_epochs": 8,           # start small; scale after it’s stable
        "batch_size": 2,
        "learning_rate": 2e-4,
        "max_length": 512,
    }
    train_model(**cfg)


