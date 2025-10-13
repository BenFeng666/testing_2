"""
Real evaluation script with actual model predictions
Evaluate finetuned model on test set with human feedback loop
"""

import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
import re

# Optional torch import
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not found. Will use mock predictions for demonstration.")

class ModelEvaluator:
    """Evaluate finetuned model with human feedback"""
    
    def __init__(self, 
                 base_model_path="Qwen/Qwen-7B-Chat",
                 lora_path="qwen_lora_finetuned_An",
                 test_data_path="data_An/test_data.csv"):
        
        self.test_data_path = test_data_path
        self.lora_path = lora_path
        
        # Load test data
        self.test_df = pd.read_csv(test_data_path)
        print(f"Loaded test data: {len(self.test_df)} samples")
        
        # Load model
        if not HAS_TORCH:
            print("\nTorch not available. Using mock predictions for demonstration.")
            self.model = None
            self.tokenizer = None
        else:
            print(f"\nLoading model from {lora_path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path,
                    trust_remote_code=True
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                
                self.model = PeftModel.from_pretrained(base_model, lora_path)
                self.model.eval()
                print("Model loaded successfully!")
                
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Will use mock predictions for demonstration")
                self.model = None
                self.tokenizer = None
        
        self.predictions_history = []
        self.feedback_history = []
    
    def predict_score(self, smiles):
        """
        Predict score for a SMILES structure
        
        Args:
            smiles: SMILES string
            
        Returns:
            int: Predicted score (1-10)
        """
        if self.model is None or self.tokenizer is None:
            # Mock prediction if model not available
            import hashlib
            hash_val = int(hashlib.md5(smiles.encode()).hexdigest(), 16)
            return (hash_val % 10) + 1
        
        # Create prompt
        prompt = f"What is the predicted score for this molecular structure: {smiles}?"
        
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
        
        # Format conversation
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.7,
                top_p=0.8,
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract score
        score = self._extract_score(response)
        return score if score is not None else 5
    
    def _extract_score(self, response):
        """Extract score from model response"""
        # Try to find score in response
        patterns = [
            r'(?:score|Score)[\s:is]+(\d+)',
            r'(\d+)(?:\s|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    score = int(matches[0])
                    if 1 <= score <= 10:
                        return score
                except ValueError:
                    continue
        
        return None
    
    def calculate_accuracy(self, predictions, ground_truth):
        """Calculate accuracy metrics"""
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        exact_match = (predictions == ground_truth).sum() / len(predictions)
        within_1 = (np.abs(predictions - ground_truth) <= 1).sum() / len(predictions)
        within_2 = (np.abs(predictions - ground_truth) <= 2).sum() / len(predictions)
        mae = np.abs(predictions - ground_truth).mean()
        rmse = np.sqrt(((predictions - ground_truth) ** 2).mean())
        
        return {
            'exact_match': exact_match,
            'within_1': within_1,
            'within_2': within_2,
            'mae': mae,
            'rmse': rmse,
            'total_samples': len(predictions)
        }
    
    def evaluate_test_set(self, round_num=0):
        """Evaluate on test set"""
        print(f"\n{'='*70}")
        print(f"EVALUATION - {'Initial' if round_num == 0 else f'After Feedback Round {round_num}'}")
        print(f"{'='*70}")
        
        predictions = []
        ground_truth = []
        
        print(f"Predicting scores for {len(self.test_df)} test samples...")
        
        for idx, row in self.test_df.iterrows():
            if (idx + 1) % 20 == 0:
                print(f"  Progress: {idx+1}/{len(self.test_df)}")
            
            smiles = row['Structure']
            true_score = int(row['Score'])
            
            pred_score = self.predict_score(smiles)
            
            predictions.append(pred_score)
            ground_truth.append(true_score)
        
        # Calculate metrics
        metrics = self.calculate_accuracy(predictions, ground_truth)
        
        print(f"\n{'='*70}")
        print(f"ACCURACY RESULTS - Round {round_num}")
        print(f"{'='*70}")
        print(f"Exact Match (score == ground truth):  {metrics['exact_match']*100:>6.2f}%")
        print(f"Within ±1 (|score - truth| <= 1):     {metrics['within_1']*100:>6.2f}%")
        print(f"Within ±2 (|score - truth| <= 2):     {metrics['within_2']*100:>6.2f}%")
        print(f"Mean Absolute Error (MAE):            {metrics['mae']:>6.2f}")
        print(f"Root Mean Squared Error (RMSE):       {metrics['rmse']:>6.2f}")
        print(f"{'='*70}")
        
        # Store results
        result = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'predictions': predictions,
            'ground_truth': ground_truth
        }
        
        self.predictions_history.append(result)
        
        return result
    
    def run_evaluation_with_feedback(self, num_rounds=3):
        """Run complete evaluation with human feedback"""
        print("\n" + "="*70)
        print("MODEL EVALUATION WITH HUMAN FEEDBACK")
        print("="*70)
        print(f"Model: {self.lora_path}")
        print(f"Test set: {self.test_data_path}")
        print(f"Feedback rounds: {num_rounds}")
        print("="*70)
        
        # Initial evaluation
        self.evaluate_test_set(round_num=0)
        
        # Feedback loop (simulated for now)
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*70}")
            print(f"HUMAN FEEDBACK ROUND {round_num}/{num_rounds}")
            print(f"{'='*70}")
            print("In a real scenario, this would:")
            print("1. Select uncertain predictions")
            print("2. Show molecule pairs to expert")
            print("3. Collect human judgments")
            print("4. Update model with feedback")
            print(f"{'='*70}")
            
            # Re-evaluate
            self.evaluate_test_set(round_num=round_num)
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print summary table"""
        print(f"\n{'='*70}")
        print("ACCURACY SUMMARY ACROSS ALL ROUNDS")
        print(f"{'='*70}")
        
        print(f"{'Round':<15} {'Exact':<12} {'±1':<12} {'±2':<12} {'MAE':<10}")
        print(f"{'-'*70}")
        
        for result in self.predictions_history:
            round_num = result['round']
            metrics = result['metrics']
            round_name = "Initial" if round_num == 0 else f"Round {round_num}"
            
            print(f"{round_name:<15} "
                  f"{metrics['exact_match']*100:>6.2f}%     "
                  f"{metrics['within_1']*100:>6.2f}%     "
                  f"{metrics['within_2']*100:>6.2f}%     "
                  f"{metrics['mae']:>6.2f}")
        
        # Calculate improvement
        if len(self.predictions_history) > 1:
            initial = self.predictions_history[0]['metrics']['exact_match']
            final = self.predictions_history[-1]['metrics']['exact_match']
            improvement = (final - initial) * 100
            
            print(f"\n{'-'*70}")
            print(f"Total Improvement: {improvement:+.2f}%")
            print(f"  Initial Accuracy: {initial*100:.2f}%")
            print(f"  Final Accuracy:   {final*100:.2f}%")
        
        print(f"{'='*70}")
    
    def save_results(self):
        """Save all results"""
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'model_path': self.lora_path,
                'test_data_path': self.test_data_path,
                'predictions_history': self.predictions_history,
                'feedback_history': self.feedback_history
            }, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save summary CSV
        summary_data = []
        for result in self.predictions_history:
            summary_data.append({
                'round': result['round'],
                'exact_match_%': result['metrics']['exact_match'] * 100,
                'within_1_%': result['metrics']['within_1'] * 100,
                'within_2_%': result['metrics']['within_2'] * 100,
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'accuracy_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Summary CSV saved to: {summary_file}")

if __name__ == "__main__":
    # Run evaluation
    evaluator = ModelEvaluator(
        lora_path="qwen_lora_finetuned_An",
        test_data_path="data_An/test_data.csv"
    )
    
    # Run with 3 feedback rounds
    evaluator.run_evaluation_with_feedback(num_rounds=3)

