"""
Inference script for ChemLLM 7B model fine-tuned with LoRA on SMILES data.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def load_model(base_model_path="AI4Chem/ChemLLM-7B-Chat", lora_path="./chemllm_lora_output"):
    """
    Load the ChemLLM-7B base model and merge with LoRA weights.
    """
    print(f"🧪 Loading base model: {base_model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA weights
    print(f"🔬 Loading LoRA weights from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    print("✅ Model loaded and ready for inference.")
    return model, tokenizer


def predict_score(model, tokenizer, smiles_structure, max_length=512):
    """
    Predict molecular score given a SMILES string.
    """
    prompt = f"What is the predicted score for this molecular structure: {smiles_structure}?"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specialized in drug discovery and molecular analysis. "
                "You can predict molecular scores based on their SMILES structures."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Use chat template if supported by ChemLLM tokenizer
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to plain text
        text = f"<|system|>{messages[0]['content']}<|user|>{prompt}<|assistant|>"

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


def batch_predict(model, tokenizer, smiles_list):
    """
    Run batch predictions for a list of SMILES.
    """
    predictions = []
    for i, smiles in enumerate(smiles_list):
        print(f"\n[{i+1}/{len(smiles_list)}] Predicting for SMILES:\n{smiles[:100]}...")
        prediction = predict_score(model, tokenizer, smiles)
        predictions.append(prediction)
        print(f"🧾 Prediction: {prediction}")
    return predictions


if __name__ == "__main__":
    # Load model
    model, tokenizer = load_model()

    # Example test SMILES
    test_smiles = [
        "[H][C@]1(OC[C@H]2CCCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)[C@]2([H])OC[C@@H]1OCCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC",
        "[H][C@]1(OC[C@H]2CCCCN(CCCCCCCC)CCCCCCCC)[C@]2([H])OC[C@@H]1OCCCN(CCCCCCCC)CCCCCCCC",
    ]

    print("\n🚀 Running batch predictions...")
    predictions = batch_predict(model, tokenizer, test_smiles)

    print("\n" + "=" * 80)
    print("✅ FINAL RESULTS")
    print("=" * 80)
    for smiles, pred in zip(test_smiles, predictions):
        print(f"\nStructure: {smiles[:60]}...")
        print(f"Prediction: {pred}")
