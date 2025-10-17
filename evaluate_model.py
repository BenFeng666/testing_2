"""
Evaluate the fine-tuned ChemLLM-7B model on test data
"""

import os
import json
import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================================================
# Utility Functions
# =====================================================

def load_test_data(file_path):
    """Load test data from either .jsonl or .xlsx"""
    if file_path.endswith(".jsonl"):
        print("📘 Detected JSONL file format")
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    elif file_path.endswith(".xlsx"):
        print("📗 Detected Excel file format — converting to message format...")
        df = pd.read_excel(file_path)
        if "Structure" not in df.columns or "Score" not in df.columns:
            raise ValueError(f"❌ Missing required columns. Found: {df.columns.tolist()}")
        df = df.dropna(subset=["Structure", "Score"])
        data = []
        for _, row in df.iterrows():
            structure = str(row["Structure"])
            score = row["Score"]
            data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant specialized in drug discovery. "
                            "When asked to predict a molecular score, you must respond "
                            "with a single number between 0 and 10 — no words, no units."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"What is the predicted score for this molecular structure: {structure}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"{score}"
                    }
                ]
            })
        return data
    else:
        raise ValueError("❌ Unsupported test data format — use .jsonl or .xlsx")


def extract_score(response):
    """Extract numerical score (0–10, supports decimals)"""
    matches = re.findall(r"(\d+(?:\.\d+)?)", response)
    for m in matches:
        try:
            val = float(m)
            if 0 <= val <= 10:
                return val
        except ValueError:
            continue
    return None

# =====================================================
# Evaluation Function
# =====================================================

def evaluate_model(
    model_path="/content/Sai_AI4Drug/chemllm_lora_output",
    base_model_name="AI4Chem/ChemLLM-7B-Chat",
    test_data_path="/content/Sai_AI4Drug/dataset/testing.xlsx",
    max_length=512
):
    print(f"🧠 Loading ChemLLM model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, padding_side="right"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Load test data
    print(f"📄 Loading test data from: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"✅ Test samples: {len(test_data)}")

    predictions, ground_truths, errors = [], [], []
    total, exact_matches = 0, 0

    print("\n🚀 Running evaluation...")
    for i, item in enumerate(tqdm(test_data)):
        messages = item["messages"]

        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if not assistant_msg:
            continue

        true_score = extract_score(assistant_msg["content"])
        if true_score is None:
            continue

        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                break
        prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,  # ✅ enable sampling for more flexible responses
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_score = extract_score(response)

        # Debug print every few samples
        if i < 5:
            print(f"\n🧪 Sample {i+1} response:\n{response}")

        if pred_score is not None:
            predictions.append(pred_score)
            ground_truths.append(true_score)
            if int(round(pred_score)) == int(round(true_score)):
                exact_matches += 1
            total += 1
        else:
            errors.append({"true_score": true_score, "response": response})

    # =====================================================
    # Compute Metrics
    # =====================================================
    if total > 0:
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        mae = np.mean(np.abs(predictions - ground_truths))
        rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
        within_1 = np.sum(np.abs(predictions - ground_truths) <= 1) / total * 100
        accuracy = exact_matches / total * 100
    else:
        mae = rmse = within_1 = accuracy = float('nan')

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS - ChemLLM-7B")
    print(f"{'='*60}")
    print(f"Total samples evaluated: {total}")
    print(f"Exact match accuracy: {accuracy:.2f}%")
    print(f"Accuracy within ±1: {within_1:.2f}%")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Failed predictions: {len(errors)}")
    print(f"{'='*60}")

    os.makedirs("output", exist_ok=True)
    results_file = "output/evaluation_results_chemllm.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": total,
            "accuracy_exact": accuracy,
            "accuracy_within_1": within_1,
            "mae": float(mae),
            "rmse": float(rmse),
            "failed_predictions": len(errors),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {results_file}")
    return {
        "total": total,
        "accuracy": accuracy,
        "within_1": within_1,
        "mae": mae,
        "rmse": rmse,
    }

# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    evaluate_model()

