"""
Model Predictor with Confidence Estimation
Predict molecular scores with uncertainty estimation
"""

import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from confidence_calculator import ConfidenceCalculator

class MolecularScorePredictor:
    """Predict molecular scores with confidence estimation"""
    
    def __init__(self, base_model_path, lora_path, score_min=1, score_max=10, device="auto"):
        """
        Initialize predictor
        
        Args:
            base_model_path: Path to base Qwen model
            lora_path: Path to LoRA weights
            score_min: Minimum score value
            score_max: Maximum score value
            device: Device to run model on
        """
        self.score_min = score_min
        self.score_max = score_max
        self.device = device
        
        print(f"Loading model from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side='right'
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.float16,
        )
        
        print(f"Loading LoRA weights from {lora_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.model.eval()
        
        self.confidence_calculator = ConfidenceCalculator(score_min, score_max)
        
    def predict_single(self, smiles_structure, num_samples=10, temperature=0.7, top_p=0.8, max_length=512):
        """
        Predict score for a single molecule with confidence
        
        Args:
            smiles_structure: SMILES string
            num_samples: Number of samples for uncertainty estimation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_length: Maximum generation length
            
        Returns:
            dict: Prediction results with confidence
        """
        # Create prompt
        prompt = f"What is the predicted score for this molecular structure: {smiles_structure}?"
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular scores based on their SMILES structures. Provide only the numerical score value between 1 and 10."
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
        
        # Generate multiple samples
        score_samples = []
        for _ in range(num_samples):
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract score from response
            score = self._extract_score(response)
            if score is not None:
                score_samples.append(score)
        
        # Calculate confidence
        if len(score_samples) >= num_samples // 2:  # At least half successful
            confidence_result = self.confidence_calculator.calculate_confidence_from_samples(score_samples)
            
            return {
                'smiles': smiles_structure,
                'predicted_score': confidence_result['mean_score'],
                'confidence': confidence_result['confidence'],
                'std': confidence_result['std_score'],
                'entropy': confidence_result['entropy'],
                'score_distribution': confidence_result['score_distribution'],
                'num_samples': len(score_samples),
                'raw_samples': score_samples
            }
        else:
            # Not enough valid predictions
            return {
                'smiles': smiles_structure,
                'predicted_score': None,
                'confidence': 0.0,
                'std': None,
                'entropy': None,
                'score_distribution': None,
                'num_samples': len(score_samples),
                'raw_samples': score_samples,
                'error': 'Insufficient valid predictions'
            }
    
    def _extract_score(self, response):
        """
        Extract numerical score from model response
        
        Args:
            response: Model response text
            
        Returns:
            float or None: Extracted score
        """
        # Try to find score patterns
        patterns = [
            r'(?:score|Score|SCORE)[\s:is]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:out of 10|/10)',
            r'(?:predicted|prediction)[\s:is]+(\d+(?:\.\d+)?)',
            r'\b(\d+(?:\.\d+)?)\b'  # Fallback: any number
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    score = float(matches[0])
                    # Validate score range
                    if self.score_min <= score <= self.score_max:
                        return score
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def predict_batch(self, smiles_list, num_samples=10, temperature=0.7, top_p=0.8, max_length=512, verbose=True):
        """
        Predict scores for multiple molecules
        
        Args:
            smiles_list: List of SMILES strings
            num_samples: Number of samples per molecule
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_length: Max generation length
            verbose: Print progress
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        
        for i, smiles in enumerate(smiles_list):
            if verbose:
                print(f"Predicting [{i+1}/{len(smiles_list)}]: {smiles[:60]}...")
            
            result = self.predict_single(
                smiles, 
                num_samples=num_samples,
                temperature=temperature,
                top_p=top_p,
                max_length=max_length
            )
            results.append(result)
            
            if verbose and result['predicted_score'] is not None:
                print(f"  Score: {result['predicted_score']:.2f}, Confidence: {result['confidence']:.2f}")
        
        return results

if __name__ == "__main__":
    # Test the predictor
    print("Testing Molecular Score Predictor...")
    
    # Example usage (would need actual model)
    # predictor = MolecularScorePredictor(
    #     base_model_path="Qwen/Qwen-7B-Chat",
    #     lora_path="./qwen_lora_finetuned"
    # )
    
    # test_smiles = "[H][C@]1(OC[C@H]2CCCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)[C@]2([H])OC[C@@H]1OCCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC"
    # result = predictor.predict_single(test_smiles)
    # print(result)
    
    print("Predictor module loaded successfully.")

