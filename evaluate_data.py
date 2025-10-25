import json
import pandas as pd
import os
import numpy as np

# ======== PATH CONFIGURATION ========
BASE_DIR = "/content/testing_2"
DATASET_PATH = f"{BASE_DIR}/dataset/testing.xlsx"       # must have columns: "Structure" and "Score"
SCORES_PATH  = f"{BASE_DIR}/output/scores.json"
CONF_PATH    = f"{BASE_DIR}/output/confidence.json"
OUT_CORRECT  = f"{BASE_DIR}/output/correct.json"
OUT_WRONG    = f"{BASE_DIR}/output/wrong.json"
THRESHOLD    = 1.0  # error tolerance

# ======== LOAD DATA ========
print("📄 Loading data...")
with open(SCORES_PATH, "r") as f:
    scores = json.load(f)
with open(CONF_PATH, "r") as f:
    confidences = json.load(f)

df = pd.read_excel(DATASET_PATH)

# Check required columns
required_cols = ["Structure", "Score"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column '{col}'. Found: {df.columns.tolist()}")

smiles_list = df["Structure"].tolist()
truths = df["Score"].tolist()

if len(scores) != len(truths) or len(scores) != len(smiles_list):
    raise ValueError(f"❌ Length mismatch: {len(scores)} predictions, {len(truths)} truths, {len(smiles_list)} SMILES")

# ======== COMPARE AND CLASSIFY ========
correct, wrong = [], []

for i, (pred, conf, true, smiles) in enumerate(zip(scores, confidences, truths, smiles_list)):
    diff = abs(pred - true)
    record = {
        "index": i,
        "smiles": str(smiles),
        "predicted_score": float(pred),
        "confidence": float(conf),
        "ground_truth": float(true),
        "error_diff": float(diff)
    }
    if diff < THRESHOLD:
        correct.append(record)
    else:
        wrong.append(record)

# ======== SAVE ========
os.makedirs(f"{BASE_DIR}/output", exist_ok=True)
with open(OUT_CORRECT, "w") as f:
    json.dump(correct, f, indent=2)
with open(OUT_WRONG, "w") as f:
    json.dump(wrong, f, indent=2)

# ======== SUMMARY ========
print(f"✅ Total samples: {len(scores)}")
print(f"✅ Correct (<{THRESHOLD}): {len(correct)}")
print(f"❌ Wrong (≥{THRESHOLD}): {len(wrong)}")
print(f"📂 Saved results to:\n  {OUT_CORRECT}\n  {OUT_WRONG}")

# ======== OPTIONAL STATS ========
errors = [abs(p - t) for p, t in zip(scores, truths)]
print(f"📊 Mean abs error: {np.mean(errors):.3f}")
print(f"📊 Median abs error: {np.median(errors):.3f}")
print(f"📈 Accuracy (<{THRESHOLD}): {len(correct)/len(scores)*100:.2f}%")

