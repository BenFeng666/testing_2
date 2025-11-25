"""
Evaluate Multi-Task Model on both Toxicity and Efficiency prediction
FAST VERSION — no chat template + prints GT vs Pred
"""

import json
import torch
import yaml
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# Load model
# ============================================================
def load_model(base_model_path, lora_path):
    print(f"Loading model: {base_model_path} with LoRA: {lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right"
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


# ============================================================
# Extraction utils
# ============================================================
def extract_toxicity(response):
    """Extract toxicity from model output."""
    nums = re.findall(r"\d+", response)
    if not nums:
        return 0
    val = int(nums[0])
    return 1 if val >= 1 else 0


def extract_efficiency(text):
    """
    Extract efficiency score (1–10) using the SAME logic as your working script.
    - Find ALL integers in the text.
    - Use the LAST one.
    - Clamp to [1, 10].
    """
    nums = re.findall(r'\b\d+\b', text)
    if not nums:
        return None

    score = int(nums[-1])  # LAST number = always correct ground truth
    score = max(1, min(10, score))
    return score



# ============================================================
# FAST generation (no chat template)
# ============================================================
def fast_generate(model, tokenizer, prompt, max_new=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
              max_new_tokens=16,
              do_sample=False,
              temperature=0.1,
              top_p=0.9,
              pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


# ============================================================
# Toxicity prediction (fast)
# ============================================================
# ============================================================
# Toxicity prediction (fast, but uses ORIGINAL PROMPT)
# ============================================================
def predict_toxicity(model, tokenizer, smiles, max_length=512):

    system_prompt = (
        "You are a helpful assistant specialized in drug discovery and "
        "molecular analysis. You can predict molecular toxicity based "
        "on their SMILES structures."
    )

    user_prompt = f"Is this molecular structure toxic? {smiles}"

    # Combine manually — NO chat template
    prompt = (
        f"{system_prompt}\n"
        f"{user_prompt}\n"
        "Answer with only 0 or 1.\n"
        "Toxicity:"
    )

    response = fast_generate(model, tokenizer, prompt)
    return extract_toxicity(response), response


# ============================================================
# Efficiency prediction (fast, but uses ORIGINAL PROMPT)
# ============================================================
def predict_efficiency(model, tokenizer, smiles, max_length=512):

    system_prompt = (
        "You are a helpful assistant specialized in drug discovery and "
        "molecular analysis. You can predict molecular scores based on "
        "their SMILES structures."
    )

    user_prompt = f"What is the predicted score for this molecular structure: {smiles}?"

    prompt = (
        f"{system_prompt}\n"
        f"{user_prompt}\n"
        "Respond only with an integer score 1–10.\n"
        "Score:"
    )

    response = fast_generate(model, tokenizer, prompt)
    return extract_efficiency(response), response

# ============================================================
# Toxicity evaluation
# ============================================================
def evaluate_toxicity(model, tokenizer, path):
    print("\n" + "="*80)
    print("EVALUATING TOXICITY")
    print("="*80)

    data = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    print(f"Loaded toxicity samples: {len(data)}")

    preds = []
    gts = []
    
    print("\n--- TOXICITY DETAILS ---\n")

    for item in data:
        smiles = item["messages"][1]["content"].split(":")[-1].strip()
        gt = extract_toxicity(item["messages"][-1]["content"])

        pred, raw = predict_toxicity(model, tokenizer, smiles)

        preds.append(pred)
        gts.append(gt)

        print(f"SMILES: {smiles}")
        print(f"GT Toxicity: {gt} | Predicted: {pred}")
        print(f"Raw model output: {raw}")
        print("-"*60)

    acc = accuracy_score(gts, preds)
    print("\nToxicity Accuracy:", acc)

    return {
        "accuracy": float(acc),
        "preds": preds,
        "gts": gts
    }


# ============================================================
# Efficiency evaluation
# ============================================================
def evaluate_efficiency(model, tokenizer, path):
    print("\n" + "="*80)
    print("EVALUATING EFFICIENCY")
    print("="*80)

    data = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    print(f"Loaded efficiency samples: {len(data)}")

    preds = []
    gts = []

    print("\n--- EFFICIENCY DETAILS ---\n")

    for item in data:
        smiles = item["messages"][1]["content"].split(":")[-1].strip()
        gt = extract_efficiency(item["messages"][-1]["content"])
        if gt is None:
            continue

        pred, raw = predict_efficiency(model, tokenizer, smiles)

        preds.append(pred)
        gts.append(gt)

        print(f"SMILES: {smiles}")
        print(f"GT Efficiency: {gt} | Predicted: {pred}")
        print(f"Raw model output: {raw}")
        print("-"*60)

    preds = np.array(preds)
    gts = np.array(gts)

    exact = np.mean(preds == gts)
    within1 = np.mean(np.abs(preds - gts) <= 1)
    mae = np.mean(np.abs(preds - gts))

    print("\nEfficiency Metrics:")
    print(f"Exact match: {exact}")
    print(f"Within ±1: {within1}")
    print(f"MAE: {mae}")

    return {
        "exact": float(exact),
        "within1": float(within1),
        "mae": float(mae),
        "preds": preds.tolist(),
        "gts": gts.tolist()
    }


# ============================================================
# Main
# ============================================================
def main():
    config = yaml.safe_load(open("training_config.yaml"))

    model, tokenizer = load_model(
        config["model"]["base_model_path"],
        config["model"]["output_dir"]
    )

    tox = evaluate_toxicity(model, tokenizer, config["data"]["toxic_test_data_path"])
    eff = evaluate_efficiency(model, tokenizer, config["data"]["efficiency_test_data_path"])

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"Toxicity Accuracy: {tox['accuracy']:.4f}")
    print(f"Efficiency Exact Match: {eff['exact']:.4f}")
    print(f"Efficiency Within ±1: {eff['within1']:.4f}")
    print(f"Efficiency MAE: {eff['mae']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
