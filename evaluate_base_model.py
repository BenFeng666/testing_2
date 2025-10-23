"""
Evaluate baseline or fine-tuned ChemLLM-7B model on test data (with preview + correct/wrong output JSONs)
"""

import os
import json
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================================
# Utility Functions
# =====================================================

def load_test_data(file_path):
    """Load test data from either .jsonl or .xlsx"""
    if file_path.endswith(".jsonl"):
        print("📘 Detected JSONL file format")
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    elif file_path.endswith(".xlsx"):
        print("📗 Detected Excel file format — converting to message format...")

        # --- Read Excel safely & verify ---
        df = pd.read_excel(file_path, usecols=["Structure", "Score"])
        df = df.dropna(subset=["Structure", "Score"])

        print("\n🔍 Preview of first few rows from Excel:")
        print(df.head(10))  # sanity check: make sure scores align with structures

        data = []
        for _, row in df.iterrows():
            structure = str(row["Structure"])
            score = float(row["Score"])
            data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular scores based on their SMILES structures."
                    },
                    {
                        "role": "user",
                        "content": f"What is the predicted score for this molecular structure: {structure}?"
                    },
                    {
                        "role": "assistant",
                        "content": f"The predicted score for the molecular structure {structure} is {score}."
                    }
                ]
            })
        return data

    else:
        raise ValueError("❌ Unsupported test data format — use .jsonl or .xlsx")


def extract_score(response: str):
    """
    Extract a numerical score in [0,10] from model output.
    Fixes:
      - Allows text between 'score' and 'is/=/:' (e.g., 'score for the molecule is 0.99')
      - Allows trailing punctuation after the number (e.g., '0.9.')
      - Avoids partial matches inside longer tokens
    """
    if not response:
        return None
    text = str(response)

    # 1) Prefer labeled patterns, allow up to ~100 non-digit chars between 'score' and the verb
    labeled = re.search(
        r'(?is)\b(?:predicted\s*)?(?:score|rating)[^\d-]{0,100}?(?:is|=|:)\s*(-?(?:10(?:\.0+)?|[0-9](?:\.\d+)?))',
        text
    )
    if labeled:
        try:
            val = float(labeled.group(1))
            return val if 0.0 <= val <= 10.0 else None
        except ValueError:
            pass

    # 2) Fallback: first standalone number token (0–10, decimals OK)
    #    Disallow preceding digit/dot; allow trailing punctuation, but not trailing digit
    m = re.search(
        r'(?<![\d.])(10(?:\.0+)?|[0-9](?:\.\d+)?)(?!\d)',
        text
    )
    if m:
        try:
            val = float(m.group(1))
            return val if 0.0 <= val <= 10.0 else None
        except ValueError:
            return None

    return None


# =====================================================
# Evaluation Function
# =====================================================

def evaluate_model(
    model_path=None,
    base_model_name="AI4Chem/ChemLLM-7B-Chat",
    test_data_path="/content/testing_2/dataset/testing.xlsx",
    max_length=512,
    preview_samples=5
):
    print("🚀 Starting ChemLLM evaluation...")
    print(f"🧠 Loading ChemLLM model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, padding_side="right"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )

    if model_path:
        from peft import PeftModel
        print(f"🧩 Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print("⚙️ Running baseline (no LoRA adapter)")
        model = base_model

    model.eval()

    # Load test data
    print(f"📄 Loading test data from: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"✅ Test samples: {len(test_data)}")

    predictions, ground_truths = [], []
    errors, previews = [], []
    within_1_data, wrong_data = [], []
    total, exact_matches = 0, 0

    print("\n🚀 Running evaluation...")
    for idx, item in enumerate(tqdm(test_data)):
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
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_score = extract_score(response)

        # store a few previews
        if idx < preview_samples:
            previews.append({
                "structure": messages[1]["content"],
                "true_score": true_score,
                "model_output": response,
                "pred_score": pred_score
            })

        if pred_score is not None:
            diff = abs(pred_score - true_score)
            result_item = {
                "structure": messages[1]["content"],
                "true_score": true_score,
                "pred_score": pred_score,
                "diff": diff,
                "model_output": response
            }

            if diff <= 1:
                within_1_data.append(result_item)
            else:
                wrong_data.append(result_item)

            predictions.append(pred_score)
            ground_truths.append(true_score)
            if round(pred_score) == round(true_score):
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
        mae = rmse = float("nan")
        within_1 = accuracy = 0

    # =====================================================
    # Output & Save
    # =====================================================
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
            "preview_outputs": previews,
        }, f, indent=2, ensure_ascii=False)

    # Save separated results
    with open("output/within_1.json", "w", encoding="utf-8") as f:
        json.dump(within_1_data, f, indent=2, ensure_ascii=False)

    with open("output/wrong.json", "w", encoding="utf-8") as f:
        json.dump(wrong_data, f, indent=2, ensure_ascii=False)

    print("\n🧾 Example Outputs (first few):")
    for p in previews:
        print(f"\n🧪 Input: {p['structure']}")
        print(f"💡 True Score: {p['true_score']}")
        print(f"🤖 Model Output: {p['model_output']}")
        print(f"🎯 Extracted Score: {p['pred_score']}")

    print("\n💾 Results saved to:")
    print(f" - Summary: {results_file}")
    print(f" - Within ±1: output/within_1.json")
    print(f" - Wrong (>±1): output/wrong.json")

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
    evaluate_model(
        model_path=None,  # baseline mode
        base_model_name="AI4Chem/ChemLLM-7B-Chat",
        test_data_path="/content/testing_2/dataset/testing.xlsx")
