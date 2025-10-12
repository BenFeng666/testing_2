"""
Inference script for the finetuned Qwen 7B model with LoRA
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model(base_model_path="Qwen/Qwen-7B-Chat", lora_path="./qwen_lora_finetuned"):
    """
    Load the finetuned model with LoRA weights
    
    Args:
        base_model_path: Path to base Qwen model
        lora_path: Path to LoRA weights
    
    Returns:
        model, tokenizer
    """
    print(f"Loading base model: {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    return model, tokenizer

def predict_score(model, tokenizer, smiles_structure, max_length=512):
    """
    Predict score for a given SMILES structure
    
    Args:
        model: The finetuned model
        tokenizer: The tokenizer
        smiles_structure: SMILES string of the molecular structure
        max_length: Maximum generation length
    
    Returns:
        str: The model's prediction
    """
    # Create the prompt
    prompt = f"What is the predicted score for this molecular structure: {smiles_structure}?"
    
    # Format as chat message
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular scores based on their SMILES structures."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Format the conversation
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0.7,
            top_p=0.8,
        )
    
    # Decode
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def batch_predict(model, tokenizer, smiles_list):
    """
    Predict scores for multiple SMILES structures
    
    Args:
        model: The finetuned model
        tokenizer: The tokenizer
        smiles_list: List of SMILES strings
    
    Returns:
        list: List of predictions
    """
    predictions = []
    for i, smiles in enumerate(smiles_list):
        print(f"\n[{i+1}/{len(smiles_list)}] Predicting for: {smiles[:50]}...")
        prediction = predict_score(model, tokenizer, smiles)
        predictions.append(prediction)
        print(f"Prediction: {prediction}")
    
    return predictions

if __name__ == "__main__":
    # Example usage
    
    # Load the model
    model, tokenizer = load_model()
    
    # Example SMILES structures
    test_smiles = [
        "[H][C@]1(OC[C@H]2CCCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)[C@]2([H])OC[C@@H]1OCCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC",
        "[H][C@]1(OC[C@H]2CCCCN(CCCCCCCC)CCCCCCCC)[C@]2([H])OC[C@@H]1OCCCN(CCCCCCCC)CCCCCCCC",
    ]
    
    # Run predictions
    print("Running batch predictions...")
    predictions = batch_predict(model, tokenizer, test_smiles)
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print("="*80)
    for smiles, pred in zip(test_smiles, predictions):
        print(f"\nStructure: {smiles[:60]}...")
        print(f"Prediction: {pred}")

