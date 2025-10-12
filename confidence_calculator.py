"""
Confidence Score Calculator using Entropy
Calculate confidence scores for molecular predictions based on output distribution entropy
"""

import numpy as np
from scipy.stats import entropy

# Optional torch import - only needed if working with torch tensors
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class ConfidenceCalculator:
    """Calculate confidence scores based on prediction uncertainty"""
    
    def __init__(self, score_min=1, score_max=10):
        """
        Initialize confidence calculator
        
        Args:
            score_min: Minimum score value
            score_max: Maximum score value
        """
        self.score_min = score_min
        self.score_max = score_max
        self.num_classes = score_max - score_min + 1
        
    def calculate_entropy(self, probabilities):
        """
        Calculate entropy of a probability distribution
        
        Args:
            probabilities: Probability distribution over scores (numpy array or list)
            
        Returns:
            float: Entropy value
        """
        if HAS_TORCH and 'torch' in str(type(probabilities)):
            probabilities = probabilities.cpu().numpy()
        
        probabilities = np.array(probabilities)
        
        # Normalize to ensure it's a valid probability distribution
        probabilities = probabilities / (probabilities.sum() + 1e-10)
        
        # Calculate entropy
        ent = entropy(probabilities, base=2)
        
        return ent
    
    def entropy_to_confidence(self, ent):
        """
        Convert entropy to confidence score (0-10 scale)
        Lower entropy = higher confidence
        
        Args:
            ent: Entropy value
            
        Returns:
            float: Confidence score (0-10)
        """
        # Maximum possible entropy for uniform distribution
        max_entropy = np.log2(self.num_classes)
        
        # Normalize entropy to 0-1 range (inverted, so low entropy = high confidence)
        normalized = 1.0 - (ent / max_entropy)
        
        # Scale to 0-10
        confidence = normalized * 10.0
        
        return float(confidence)
    
    def calculate_confidence_from_samples(self, score_samples):
        """
        Calculate confidence from multiple prediction samples
        
        Args:
            score_samples: List of predicted scores from multiple sampling
            
        Returns:
            dict: {
                'mean_score': mean predicted score,
                'std_score': standard deviation,
                'entropy': entropy value,
                'confidence': confidence score (0-10)
            }
        """
        scores = np.array(score_samples)
        
        # Calculate statistics
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        
        # Create histogram (probability distribution)
        hist, _ = np.histogram(scores, bins=self.num_classes, 
                              range=(self.score_min, self.score_max + 1))
        probabilities = hist / (len(scores) + 1e-10)
        
        # Calculate entropy
        ent = self.calculate_entropy(probabilities)
        
        # Convert to confidence
        confidence = self.entropy_to_confidence(ent)
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'entropy': ent,
            'confidence': confidence,
            'score_distribution': probabilities.tolist()
        }
    
    def calculate_confidence_from_logits(self, logits):
        """
        Calculate confidence directly from model logits
        
        Args:
            logits: Model output logits for each score class
            
        Returns:
            dict: Confidence metrics
        """
        if HAS_TORCH and 'torch' in str(type(logits)):
            probabilities = F.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy()
        else:
            probabilities = np.array(logits)
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        
        # Calculate entropy
        ent = self.calculate_entropy(probabilities)
        
        # Convert to confidence
        confidence = self.entropy_to_confidence(ent)
        
        # Get predicted score
        score_values = np.arange(self.score_min, self.score_max + 1)
        mean_score = float(np.sum(probabilities * score_values))
        
        return {
            'mean_score': mean_score,
            'entropy': ent,
            'confidence': confidence,
            'score_distribution': probabilities.tolist()
        }

def test_confidence_calculator():
    """Test the confidence calculator"""
    calc = ConfidenceCalculator(score_min=1, score_max=10)
    
    # Test 1: High confidence (low entropy) - all predictions same
    high_conf_samples = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    result1 = calc.calculate_confidence_from_samples(high_conf_samples)
    print("Test 1 - High Confidence:")
    print(f"  Mean Score: {result1['mean_score']:.2f}")
    print(f"  Std: {result1['std_score']:.2f}")
    print(f"  Entropy: {result1['entropy']:.2f}")
    print(f"  Confidence: {result1['confidence']:.2f}")
    
    # Test 2: Low confidence (high entropy) - predictions spread out
    low_conf_samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result2 = calc.calculate_confidence_from_samples(low_conf_samples)
    print("\nTest 2 - Low Confidence:")
    print(f"  Mean Score: {result2['mean_score']:.2f}")
    print(f"  Std: {result2['std_score']:.2f}")
    print(f"  Entropy: {result2['entropy']:.2f}")
    print(f"  Confidence: {result2['confidence']:.2f}")
    
    # Test 3: Medium confidence
    med_conf_samples = [7, 7, 8, 8, 8, 8, 8, 9, 9, 7]
    result3 = calc.calculate_confidence_from_samples(med_conf_samples)
    print("\nTest 3 - Medium Confidence:")
    print(f"  Mean Score: {result3['mean_score']:.2f}")
    print(f"  Std: {result3['std_score']:.2f}")
    print(f"  Entropy: {result3['entropy']:.2f}")
    print(f"  Confidence: {result3['confidence']:.2f}")

if __name__ == "__main__":
    test_confidence_calculator()

