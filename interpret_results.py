import os, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== CONFIG =====
BASE_DIR = "/content/testing_2"
CORRECT_PATH = f"{BASE_DIR}/output/correct.json"
WRONG_PATH   = f"{BASE_DIR}/output/wrong.json"

OUT_MD_INTERP   = f"{BASE_DIR}/output/chemllm_json_analysis_subset.md"
OUT_SELECTION   = f"{BASE_DIR}/dataset/training_2_selection.json"   # stores selected ids + notes
OUT_TRAIN_JSON  = f"{BASE_DIR}/dataset/training_2_sample.json"      # final training data

BASE_MODEL      = "AI4Chem/ChemLLM-7B-Chat"
LORA_MODEL_PATH = f"{BASE_DIR}/chemllm_lora_output"

MAX_NEW_TOKENS  = 1800
TEMPERATURE     = 0.3
TOP_P           = 0.9

# limit how many wrong samples we show to the model (for context length)
MAX_CANDIDATES_TO_SHOW = 120
NUM_TO_SELECT = 20  # ask ChemLLM to pick ~20

# ===== MODEL LOADING =====
def load_model():
    print(f"🧠 Loading {BASE_MODEL} + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="right")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, LORA_MODEL_PATH)
    model.eval()
    return tokenizer, model

def run_llm(tokenizer, model, system_prompt, user_prompt, tokens=MAX_NEW_TOKENS):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    gen_ids = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

# ===== PROMPTS =====
def build_analysis_prompt(correct, wrong):
    # Take subset (25 correct + 25 wrong, or less if dataset smaller)
    subset_correct = correct[:25]
    subset_wrong = wrong[:25]

    system_prompt = (
        "You are ChemLLM, a model analysis expert for molecular regression tasks. "
        "You will be given JSON data for correct and wrong predictions. "
        "Each item includes SMILES, predicted score, confidence, and ground truth. "
        "Your goal is to analyze the overall pattern between them — focusing on the ground truth ranges "
        "(1–3 = low range, 3–6 = middle range, >6 = high range), confidence trends, and any common chemical features "
        "you notice among wrong predictions. Be concise but specific."
    )

    user_prompt = (
        "Here are the evaluation results (subset of 25 correct and 25 wrong):\n\n"
        f"### CORRECT PREDICTIONS JSON:\n{json.dumps(subset_correct, indent=2)}\n\n"
        f"### WRONG PREDICTIONS JSON:\n{json.dumps(subset_wrong, indent=2)}\n\n"
        "Now analyze them carefully and describe:\n"
        "- Which ground truth score range (low, middle, or high) tends to have more wrong predictions.\n"
        "- What confidence patterns you notice between correct vs. wrong.\n"
        "- Any recurring structural or chemical patterns in wrong SMILES.\n"
        "- What fine-tuning or calibration adjustments could help fix these issues."
    )

    return system_prompt, user_prompt

def build_numbered_candidates(wrong, lo=3.0, hi=6.0):
    """
    Build a compact, NUMBERED list restricted to mid-range ground truth: lo <= truth < hi.
    Only items in this range will be shown to ChemLLM and eligible for selection.
    """
    filtered = [d for d in wrong if float(d.get("ground_truth", 0.0)) >= lo and float(d.get("ground_truth", 0.0)) < hi]
    cand = []
    for i, d in enumerate(filtered[:MAX_CANDIDATES_TO_SHOW]):
        cand.append({
            "id": i,  # local id for this mid-range list (0..N-1)
            "smiles": str(d.get("smiles", ""))[:180],
            "predicted": float(d.get("predicted_score", 0.0)),
            "truth": float(d.get("ground_truth", 0.0)),
            "confidence": float(d.get("confidence", 0.0)),
            "abs_error": float(abs(d.get("predicted_score", 0.0) - d.get("ground_truth", 0.0))),
            "orig_index": int(d.get("index", i))
        })
    return cand

def build_training_selection_prompt(numbered_wrong, analysis_text):
    """
    Use ChemLLM's interpretation to pick IDs, but the list you see is already restricted
    to mid-range truth values (3–6). Select ONLY from these IDs.
    """
    available = len(numbered_wrong)
    target = min(NUM_TO_SELECT, available)

    sys_p = (
        "You are ChemLLM, a medicinal chemistry assistant for molecular score prediction. "
        "You will receive (A) your prior interpretation and (B) a NUMBERED list of wrong predictions "
        "already filtered to MID-RANGE ground truths (3 ≤ truth < 6). "
        "Fields: id, smiles, predicted, truth, confidence, abs_error. "
        f"Select about {target} diverse and informative mistakes that best address weaknesses you identified. "
        "Prefer cases with larger absolute error, lower confidence, and diverse chemotypes. "
        "Return STRICT JSON ONLY in this schema:\n"
        "{\n"
        "  \"selected_ids\": [id1, id2, ...],\n"
        "  \"notes\": {\n"
        "     \"id1\": \"(optional) short reason tied to your interpretation\",\n"
        "     \"id2\": \"(optional) short reason tied to your interpretation\"\n"
        "  }\n"
        "}\n"
        "Rules: Do NOT include any extra text. IDs must be integers from the list. No duplicates."
    )

    usr_p = (
        "=== YOUR PRIOR INTERPRETATION ===\n"
        f"{analysis_text.strip()}\n\n"
        "=== NUMBERED WRONG LIST (MID-RANGE ONLY: 3 ≤ truth < 6) ===\n"
        f"{json.dumps(numbered_wrong, indent=2)}\n\n"
        "Now output ONLY the JSON described above."
    )
    return sys_p, usr_p



# ===== PARSING & BUILDING TRAIN SET =====
def safe_parse_selection(text):
    """
    Try to extract and repair a possibly incomplete or too-long JSON output from ChemLLM.
    This function:
      - Finds the first JSON-like block {...}
      - Closes it safely if truncated
      - Filters invalid/non-integer ids
    """
    import re, json

    try:
        # find the first JSON-looking section
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        raw_json = m.group(0)

        # try to fix if it’s truncated (missing ']' or '}')
        if raw_json.count("{") > raw_json.count("}"):
            raw_json += "}"
        if raw_json.count("[") > raw_json.count("]"):
            raw_json += "]"

        # remove trailing commas before closing brackets
        raw_json = re.sub(r",(\s*[\}\]])", r"\1", raw_json)

        # parse safely
        obj = json.loads(raw_json)

        # validate structure
        if "selected_ids" not in obj or not isinstance(obj["selected_ids"], list):
            return None

        # ensure note dict is correct
        notes = obj.get("notes", {})
        if not isinstance(notes, dict):
            notes = {}

        # sanitize: only integers, unique
        selected = []
        for x in obj["selected_ids"]:
            try:
                i = int(x)
                if i not in selected:
                    selected.append(i)
            except:
                continue

        return {"selected_ids": selected[:20], "notes": notes}

    except Exception as e:
        print(f"⚠️ JSON parse error: {e}")
        return None


def build_training_samples_from_ids(numbered_wrong, selected_ids, notes_dict):
    """
    Build LoRA training samples from chosen ids.
    If ChemLLM gave a note for an id, use it; otherwise auto-generate a short correction.
    """
    samples = []
    used = set()
    for sid in selected_ids:
        if not isinstance(sid, int): 
            continue
        if sid < 0 or sid >= len(numbered_wrong): 
            continue
        if sid in used:
            continue
        used.add(sid)
        d = numbered_wrong[sid]
        pred = d["predicted"]
        truth = d["truth"]
        diff = truth - pred
        direction = "underestimates" if diff > 0 else "overestimates"
        auto_fb = f"Model {direction} by {abs(diff):.2f}; adjust toward {truth:.2f}. Consider features implied by the SMILES."
        fb = notes_dict.get(str(sid)) or notes_dict.get(sid) or auto_fb

        samples.append({
            "messages": [
                {"role": "system", "content": "You are ChemLLM, predicting molecular transfection efficiency (1–10)."},
                {"role": "user", "content": f"SMILES: {d['smiles']}"},
                {"role": "assistant", "content": f"Predicted: {pred:.2f}, True: {truth:.2f}. Feedback: {fb}"}
            ]
        })
        if len(samples) >= NUM_TO_SELECT:
            break
    return samples

# ===== MAIN =====
def main():
    print("🚀 ChemLLM: selecting wrong samples for fine-tuning (JSON-only mode)…")

    # Load BOTH files (we need 'correct' to produce analysis_text)
    with open(CORRECT_PATH, "r") as f:
        correct = json.load(f)
    with open(WRONG_PATH, "r") as f:
        wrong = json.load(f)

    tokenizer, model = load_model()

    # 1) Produce interpretation text to guide the selection
    sys_a, usr_a = build_analysis_prompt(correct, wrong)
    analysis_text = run_llm(tokenizer, model, sys_a, usr_a, tokens=700)

    # 2) Build mid-range candidates only (3 ≤ truth < 6)
    numbered_wrong = build_numbered_candidates(wrong, lo=3.0, hi=6.0)

    # If nothing is in range, exit gracefully
    if len(numbered_wrong) == 0:
        print(json.dumps({
            "selected_ids": [],
            "notes": {},
            "info": "No mid-range (3<=truth<6) wrong samples available."
        }, indent=2))
        return

    # 3) Build selection prompt using interpretation + mid-range data
    sys_s, usr_s = build_training_selection_prompt(numbered_wrong, analysis_text)
    selection_raw = run_llm(tokenizer, model, sys_s, usr_s, tokens=600)

    # 4) Parse and sanitize output
    selection = safe_parse_selection(selection_raw)
    if not selection:
        print("⚠️ Could not parse valid selection JSON. Here’s the raw output:")
        print(selection_raw)
        return

    clean_selection = {
        "selected_ids": selection["selected_ids"][:min(NUM_TO_SELECT, len(numbered_wrong))],
        "notes": selection.get("notes", {})
    }

        # === Create new fine-tuning dataset from selected samples ===
    OUT_TRAIN_JSON = f"{BASE_DIR}/dataset/training_2_sample.json"
    os.makedirs(os.path.dirname(OUT_TRAIN_JSON), exist_ok=True)

    training_samples = []
    notes_dict = clean_selection.get("notes", {})

    for sid in clean_selection["selected_ids"]:
        if not isinstance(sid, int):
            continue
        if sid < 0 or sid >= len(numbered_wrong):
            continue

        d = numbered_wrong[sid]
        pred = d["predicted"]
        truth = d["truth"]
        diff = truth - pred
        direction = "underestimates" if diff > 0 else "overestimates"
        fb = notes_dict.get(str(sid), f"Model {direction} by {abs(diff):.2f}; adjust toward {truth:.2f}.")

        training_samples.append({
            "messages": [
                {"role": "system", "content": "You are ChemLLM, predicting molecular transfection efficiency (1–10)."},
                {"role": "user", "content": f"SMILES: {d['smiles']}"},
                {"role": "assistant", "content": f"Predicted: {pred:.2f}, True: {truth:.2f}. Feedback: {fb}"}
            ]
        })

    # Save new training dataset
    with open(OUT_TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(training_samples, f, indent=2)

    print(f"✅ Saved {len(training_samples)} new fine-tuning samples to {OUT_TRAIN_JSON}")


    # 5) Print ONLY the final JSON (as requested)
    print(json.dumps(clean_selection, indent=2))


if __name__ == "__main__":
    main()






