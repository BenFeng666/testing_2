"""
Evaluate the finetuned model on test data
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from tqdm import tqdm
import numpy as np

def load_test_data(jsonl_file):
    """Load test data from JSONL file"""
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_score(response):
    """Extract numerical score from model response"""
    # Try to find integer between 1-10
    match = re.search(r'\b([1-9]|10)\b', response)
    if match:
        return int(match.group(1))
    return None

def evaluate_model(
    model_path="./qwen_lora_finetuned",
    base_model_name="Qwen/Qwen2-7B-Instruct",
    test_data_path="data/test_data.jsonl",
    max_length=512
):
    """
    Evaluate the finetuned model on test data
    
    Args:
        model_path: Path to the finetuned LoRA model
        base_model_name: Base model name
        test_data_path: Path to test data JSONL file
        max_length: Maximum sequence length
    """
    
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Load base model with 8-bit quantization (same as training)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"Test samples: {len(test_data)}")
    
    # Evaluate
    predictions = []
    ground_truths = []
    exact_matches = 0
    total = 0
    errors = []
    
    print("\nEvaluating...")
    for item in tqdm(test_data):
        messages = item['messages']
        
        # Extract ground truth score from assistant's response
        assistant_msg = next((msg for msg in messages if msg['role'] == 'assistant'), None)
        if not assistant_msg:
            continue
            
        true_score = extract_score(assistant_msg['content'])
        if true_score is None:
            continue
        
        # Construct prompt (system + user message)
        prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'user':
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                break
        prompt += "<|im_start|>assistant\n"
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred_score = extract_score(response)
        
        if pred_score is not None:
            predictions.append(pred_score)
            ground_truths.append(true_score)
            
            if pred_score == true_score:
                exact_matches += 1
            
            total += 1
        else:
            errors.append({
                'true_score': true_score,
                'response': response
            })
    
    # Calculate metrics
    accuracy = exact_matches / total * 100 if total > 0 else 0
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    
    # Calculate accuracy within ±1
    within_1 = np.sum(np.abs(predictions - ground_truths) <= 1) / total * 100 if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples evaluated: {total}")
    print(f"Exact match accuracy: {accuracy:.2f}%")
    print(f"Accuracy within ±1: {within_1:.2f}%")
    print(f"MAE (Mean Absolute Error): {mae:.3f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f}")
    print(f"Failed predictions: {len(errors)}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'total_samples': total,
        'exact_matches': exact_matches,
        'accuracy': accuracy,
        'accuracy_within_1': within_1,
        'mae': float(mae),
        'rmse': float(rmse),
        'failed_predictions': len(errors),
        'predictions': predictions.tolist(),
        'ground_truths': ground_truths.tolist()
    }
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    results_file = 'output/evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_file}")
    
    # Show some examples
    print(f"\nSample predictions:")
    for i in range(min(10, total)):
        print(f"  Sample {i+1}: True={ground_truths[i]}, Predicted={predictions[i]}")
    
    if errors:
        print(f"\nSample errors (first 3):")
        for i, err in enumerate(errors[:3]):
            print(f"  Error {i+1}: True={err['true_score']}, Response='{err['response']}'")
    
    return results

if __name__ == "__main__":
    # Evaluate the model
    results = evaluate_model(
        model_path="./qwen_lora_finetuned",
        base_model_name="Qwen/Qwen2-7B-Instruct",
        test_data_path="data/test_data.jsonl"
    )

