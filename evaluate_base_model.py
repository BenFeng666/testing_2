"""
Evaluate ChemLLM-7B baseline model with confidence estimation.
No LoRA — uses multiple generations to compute confidence per molecule.
Outputs:
  - output/scores.json        (mean score per molecule)
  - output/confidence.json    (confidence per molecule)
"""

import os
import json
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from confidence_calculator import ConfidenceCalculator


# =====================================================
# CONFIGURATION
# =====================================================
BASE_MODEL = "AI4Chem/ChemLLM-7B-Chat"
TEST_DATA_PATH = "dataset/testing.xlsx"
OUTPUT_DIR = "output"

NUM_SAMPLES = 10          # number of generations per molecule
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_LENGTH = 256
SCORE_MIN, SCORE_MAX = 1, 12


# =====================================================
# SCORE EXTRACTION
# =====================================================
def extract_score(text):
    """Extract a number between 1–10 from text"""
    if not text:
        return None
    match = re.search(r'(?<![\d.])(10(?:\.0+)?|[0-9](?:\.\d+)?)(?!\d)', str(text))
    if match:
        val = float(match.group(1))
        return val if SCORE_MIN <= val <= SCORE_MAX else None
    return None


# =====================================================
# MAIN EVALUATION FUNCTION
# =====================================================
"""
ChemLLM-7B Baseline Evaluation with Confidence (no LoRA)
Supports both .xlsx and .jsonl datasets
Outputs:
  - output/scores.json
  - output/confidence.json
"""

import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from confidence_calculator import ConfidenceCalculator


# =====================================================
# CONFIGURATION
# =====================================================
BASE_MODEL = "AI4Chem/ChemLLM-7B-Chat"
TEST_DATA_PATH = "dataset/testing.xlsx"   # Can be .xlsx or .jsonl
OUTPUT_DIR = "output"

NUM_SAMPLES = 10
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_LENGTH = 256
SCORE_MIN, SCORE_MAX = 1, 10


# =====================================================
# UTILITIES
# =====================================================
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


def load_test_data(file_path):
    """Load test data from either .jsonl or .xlsx and return a list of message dicts"""
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
        df = df.dropna(subset=["Structure"])
        data = []
        for _, row in df.iterrows():
            structure = str(row["Structure"])
            data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular transfection efficiency based on their SMILES structures."
                    },
                    {
                        "role": "user",
                        "content": f"What is the predicted score for this molecular transfection efficiency: {structure}?"
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
    print("🚀 Starting ChemLLM-7B baseline evaluation (with confidence)...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🧠 Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()

    conf_calc = ConfidenceCalculator(SCORE_MIN, SCORE_MAX)

    # Load test data
    print(f"📄 Loading test data from: {TEST_DATA_PATH}")
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"✅ Loaded {len(test_data)} molecules")

    all_scores = []
    all_confidences = []

    for idx, item in enumerate(tqdm(test_data, desc="Predicting")):
        messages = item["messages"]
        # Build chat prompt
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        # Compute mean & confidence
        if sample_scores:
            conf_data = conf_calc.calculate_confidence_from_samples(sample_scores)
            mean_score = conf_data["mean_score"]
            confidence = conf_data["confidence"]
        else:
            mean_score, confidence = 0.0, 0.0

        all_scores.append(round(mean_score, 4))
        all_confidences.append(round(confidence, 4))

    # Save results
    scores_file = os.path.join(OUTPUT_DIR, "scores.json")
    conf_file = os.path.join(OUTPUT_DIR, "confidence.json")

    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False)
    with open(conf_file, "w", encoding="utf-8") as f:
        json.dump(all_confidences, f, ensure_ascii=False)
    
    print(f"\n💾 Saved {len(all_scores)} scores to: {scores_file}")
    print(f"💾 Saved {len(all_confidences)} confidences to: {conf_file}")
    print("✅ Evaluation completed successfully.")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    evaluate_model()
