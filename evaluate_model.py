"""
Evaluate the fine-tuned ChemLLM-7B LoRA model on test data
with entropy-based confidence scoring.

Outputs:
  - output/scores.json
  - output/confidence.json
  - output/evaluation_summary.json
"""

import os
import re
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from confidence_calculator import ConfidenceCalculator


# =====================================================
# CONFIGURATION
# =====================================================
BASE_MODEL = "AI4Chem/ChemLLM-7B-Chat"
LORA_MODEL_PATH = "/content/testing_2/chemllm_lora_output"
TEST_DATA_PATH = "/content/testing_2/dataset/testing.xlsx"
OUTPUT_DIR = "output"

NUM_SAMPLES = 10
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_LENGTH = 256
SCORE_MIN, SCORE_MAX = 1, 12


# =====================================================
# UTILITIES
# =====================================================
def extract_score(response: str):
    """Extract a numerical score (0–10 or decimal) from model output."""
    if not response:
        return None
    text = str(response)

    # Find the first floating-point or integer number
    match = re.search(r"(?<![\d.])-?\d+(?:\.\d+)?(?!\d)", text)
    if match:
        try:
            val = float(match.group(0))
            return val
        except ValueError:
            return None
    return None


def load_test_data(file_path):
    """Load test data from .xlsx or .jsonl, return a list of message dicts."""
    if file_path.endswith(".jsonl"):
        print("📘 Detected JSONL file format")
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    elif file_path.endswith(".xlsx"):
        print("📗 Detected Excel file format — converting to message format...")
        df = pd.read_excel(file_path)
        if "Structure" not in df.columns:
            raise ValueError(f"❌ Missing 'Structure' column. Found: {df.columns.tolist()}")
        df = df.dropna(subset=["Structure"])
        data = []
        for _, row in df.iterrows():
            structure = str(row["Structure"])
            data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant specialized in drug discovery and "
                            "molecular analysis. Respond only with a number. "
                            "Do not include text, units, or explanations."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"What is the predicted transfection efficiency score for this molecular structure: {structure}?"
                    }
                ]
            })
        return data
    else:
        raise ValueError("❌ Unsupported test data format — use .jsonl or .xlsx")


# =====================================================
# MAIN EVALUATION FUNCTION
# =====================================================
def evaluate_model():
    print("🚀 Starting ChemLLM-7B LoRA evaluation (with confidence)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load base model and LoRA adapter
    print(f"🧠 Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="right")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )

    print(f"🔌 Attaching LoRA adapter: {LORA_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()

    # Confidence calculator
    conf_calc = ConfidenceCalculator(SCORE_MIN, SCORE_MAX)

    # Load test data
    print(f"📄 Loading test data from: {TEST_DATA_PATH}")
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"✅ Loaded {len(test_data)} molecules")

    all_scores = []
    all_confidences = []
    all_errors = []

    # =====================================================
    # Generate predictions
    # =====================================================
    for idx, item in enumerate(tqdm(test_data, desc="Predicting")):
        messages = item["messages"]

        # ---- Plain text prompt (no ChatML) ----
        if len(messages) > 1 and "content" in messages[1]:
            user_prompt = messages[1]["content"]
        else:
            user_prompt = "Predict the molecular transfection efficiency score for this structure."

        system_prompt = (
            "You are a helpful assistant specialized in drug discovery and molecular analysis. "
            "Respond only with a number. Do not include any text or units.\n"
        )

        text = f"{system_prompt}\n{user_prompt}\nPredicted score:"

        # ---- Generate multiple samples ----
        sample_scores = []
        for _ in range(NUM_SAMPLES):
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(response)
            score = extract_score(response)
            if score is not None:
                sample_scores.append(score)

        # ---- Compute mean score and confidence ----
        if sample_scores:
            conf_data = conf_calc.calculate_confidence_from_samples(sample_scores)
            mean_score = conf_data["mean_score"]
            confidence = conf_data["confidence"]
        else:
            mean_score, confidence = 0.0, 0.0
            all_errors.append(idx)

        all_scores.append(round(mean_score, 4))
        all_confidences.append(round(confidence, 4))

    # =====================================================
    # Save results
    # =====================================================
    scores_file = os.path.join(OUTPUT_DIR, "scores.json")
    conf_file = os.path.join(OUTPUT_DIR, "confidence.json")
    summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.json")

    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False)
    with open(conf_file, "w", encoding="utf-8") as f:
        json.dump(all_confidences, f, ensure_ascii=False)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_molecules": len(test_data),
            "failed_predictions": len(all_errors),
            "average_confidence": float(np.mean(all_confidences)) if all_confidences else 0.0
        }, f, indent=2)

    print(f"\n💾 Saved {len(all_scores)} scores to: {scores_file}")
    print(f"💾 Saved {len(all_confidences)} confidences to: {conf_file}")
    print(f"💾 Summary saved to: {summary_file}")
    print("✅ Evaluation completed successfully.")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    evaluate_model()

