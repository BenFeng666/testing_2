

import os
import json
import torch
import yaml
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Model Loader
# ============================================================
def load_checkpoint_model(base_model_path, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    return model, tokenizer


# ============================================================
# Extraction Helpers (GT + Prediction)
# ============================================================
def extract_toxicity(text):
    m = re.search(r'Toxicity value:\s*(\d+)', text)
    if m:
        v = int(m.group(1))
        return 1 if v >= 1 else 0
    low = text.lower()
    if "non-toxic" in low:
        return 0
    if "toxic" in low:
        return 1
    nums = re.findall(r'\d+', text)
    if not nums:
        return 0
    return 1 if int(nums[-1]) >= 1 else 0


def extract_efficiency(text):
    patterns = [
        r'[Ee]fficiency [Ss]core[:\s]+(\d+)',
        r'[Ee]fficiency[:\s]+(\d+)',
        r'[Ss]core[:\s]+(\d+)',
        r'[Pp]redicted score[:\s]+(\d+)',
        r'is (\d+)',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            val = int(m.group(1))
            return max(1, min(10, val))
    return None


# ============================================================
# Prediction Function
# ============================================================
def predict_tox_eff(model, tokenizer, smiles):

    prompt = f"""Analyze this lipid molecule: {smiles}

Provide:
1. Predicted molecular score (integer 1 to 10) 
2. Toxicity (0 or 1, 0 means not toxic and 1 means toxic)

Format EXACTLY as:
Predicted score: Y
Toxicity: X
"""

    messages = [
        {"role": "system",
         "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular scores based on their SMILES structures. Focus on Predicting molecular score"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=40,
            do_sample=False,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    gen = output[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(gen, skip_special_tokens=True)

    # Parse model output
    tox_match = re.search(r"Toxicity:\s*(\d+)", response)
    eff_match = re.search(r"score:\s*(\d+)", response)

    tox = int(tox_match.group(1)) if tox_match else None
    eff = int(eff_match.group(1)) if eff_match else None

    return tox, eff, response


# ============================================================
# Dataset Loaders
# ============================================================
def load_efficiency_dataset(efficiency_test_path, limit=200):
    data = []
    with open(efficiency_test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(data) >= limit:
                break
            j = json.loads(line)
            msgs = j["messages"]
            user_msg = msgs[1]["content"]
            asst_msg = msgs[-1]["content"]

            m = re.search(r':\s*(.*)', user_msg)
            if not m:
                continue

            smiles = m.group(1).strip()
            eff = extract_efficiency(asst_msg)
            if eff is None:
                continue

            tox = extract_toxicity(asst_msg)

            data.append({
                "smiles": smiles,
                "true_toxicity": 0,
                "true_efficiency": eff
            })
    return data


def load_toxic_dataset(toxic_test_path):
    data = []
    with open(toxic_test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            msgs = item["messages"]
            user_msg = msgs[1]["content"]
            asst_msg = msgs[-1]["content"]

            parts = user_msg.strip().split()
            smiles = parts[-1].rstrip("?:;,.")  # SMILES last token

            true_tox = extract_toxicity(asst_msg)

            data.append({
                "smiles": smiles,
                "true_toxicity": true_tox,
                "true_efficiency": None
            })
    return data


def extract_step_from_checkpoint(checkpoint_path):
    m = re.search(r"checkpoint-(\d+)", str(checkpoint_path))
    return int(m.group(1)) if m else None


# ============================================================
# Evaluate a Single Checkpoint
# ============================================================
def evaluate_checkpoint(
        base_model_path,
        checkpoint_path,
        toxic_data,
        eff_data,
        output_file
    ):

    print(f"\nEvaluating checkpoint: {checkpoint_path}")

    model, tokenizer = load_checkpoint_model(base_model_path, checkpoint_path)

    toxic_subset = toxic_data[:200]
    eff_subset = eff_data[:200]

    print(f"Using {len(toxic_subset)} toxicity samples")
    print(f"Using {len(eff_subset)} efficiency samples\n")

    tox_true_all, tox_pred_all = [], []
    eff_true_list, eff_pred_list = [], []

    # =======================================================
    # Efficiency dataset evaluation
    # =======================================================
    print("\n" + "=" * 80)
    print(" EFFICIENCY DATASET (200 samples) ")
    print("=" * 80)

    for i, item in enumerate(eff_subset):
        smiles = item["smiles"]
        true_tox = item["true_toxicity"]
        true_eff = item["true_efficiency"]

        pred_tox, pred_eff, response = predict_tox_eff(model, tokenizer, smiles)

        tox_true_all.append(true_tox)
        tox_pred_all.append(pred_tox)

        if pred_eff is not None:
            eff_true_list.append(true_eff)
            eff_pred_list.append(pred_eff)

        print(f"[EFF] {i+1}/200 | SMILES={smiles} | True_Tox={true_tox}, Pred_Tox={pred_tox} "
              f"| True_Eff={true_eff}, Pred_Eff={pred_eff}")

    # =======================================================
    # Toxic dataset evaluation
    # =======================================================
    print("=" * 80)
    print(" TOXICITY DATASET (200 samples) ")
    print("=" * 80)

    for i, item in enumerate(toxic_subset):
        smiles = item["smiles"]
        true_tox = item["true_toxicity"]

        pred_tox, pred_eff, response = predict_tox_eff(model, tokenizer, smiles)

        tox_true_all.append(true_tox)
        tox_pred_all.append(pred_tox)

        print(f"[TOX] {i+1}/200 | SMILES={smiles} | True_Tox={true_tox} | Pred_Tox={pred_tox}")

    # =======================================================
    # METRICS
    # =======================================================
    tox_accuracy = np.mean(np.array(tox_true_all) == np.array(tox_pred_all))

    eff_true = np.array(eff_true_list)
    eff_pred = np.array(eff_pred_list)

    # NEW: Within ±1 accuracy
    eff_within1_accuracy = float(np.mean(np.abs(eff_true - eff_pred) <= 1))

    eff_mae = float(mean_absolute_error(eff_true, eff_pred))
    eff_rmse = float(np.sqrt(mean_squared_error(eff_true, eff_pred)))

    results = {
        "checkpoint": str(checkpoint_path),
        "step": extract_step_from_checkpoint(checkpoint_path),
        "toxicity_accuracy": float(tox_accuracy),
        "toxicity_samples": len(tox_true_all),

        "efficiency_within1_accuracy": eff_within1_accuracy,
        "efficiency_mae": eff_mae,
        "efficiency_rmse": eff_rmse,
        "efficiency_samples": len(eff_true_list)
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(f"Toxicity samples: {len(tox_true_all)}")
    print(f"Efficiency samples: {len(eff_true_list)}")
    print(f"Toxicity Accuracy: {tox_accuracy:.4f}")
    print(f"Efficiency Accuracy (±1): {eff_within1_accuracy:.4f}")
    print(f"MAE: {eff_mae:.4f}")
    print(f"RMSE: {eff_rmse:.4f}")

    return results


# ============================================================
# Main
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training_config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    base_model_path = config["model"]["base_model_path"]
    output_dir = config["model"]["output_dir"]
    toxic_test_path = config["data"]["toxic_test_data_path"]
    eff_test_path = config["data"]["efficiency_test_data_path"]

    print("Loading datasets…")
    toxic_data = load_toxic_dataset(toxic_test_path)
    eff_data = load_efficiency_dataset(eff_test_path, 200)

    print(f"Loaded toxic samples: {len(toxic_data)}")
    print(f"Loaded efficiency samples: {len(eff_data)}")

    checkpoints = sorted(
        [p for p in Path(output_dir).iterdir()
         if p.is_dir() and "checkpoint-" in p.name],
        key=lambda x: int(re.search(r"checkpoint-(\d+)", x.name).group(1)),
        reverse=True
    )

    Path("checkpoint_results").mkdir(exist_ok=True)

    all_results = []

    for ckpt in checkpoints:
        step = extract_step_from_checkpoint(ckpt)
        out_file = Path("checkpoint_results") / f"checkpoint-{step}.json"

        res = evaluate_checkpoint(
            base_model_path,
            ckpt,
            toxic_data,
            eff_data,
            out_file
        )
        all_results.append(res)

    json.dump(all_results, open("checkpoint_results/all_checkpoints_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
