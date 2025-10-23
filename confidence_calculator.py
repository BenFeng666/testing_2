"""
Confidence Score Calculator using Entropy
Automatically adapts to sample range (works for decimal or integer scores)
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
            score_min: Default minimum score (used if range cannot be detected)
            score_max: Default maximum score (used if range cannot be detected)
        """
        self.default_min = score_min
        self.default_max = score_max

    # =====================================================
    # Core Entropy Computation
    # =====================================================
    def calculate_entropy(self, probabilities):
        """Compute Shannon entropy of a probability distribution"""
        if HAS_TORCH and 'torch' in str(type(probabilities)):
            probabilities = probabilities.cpu().numpy()

        probabilities = np.array(probabilities)
        probabilities = probabilities / (probabilities.sum() + 1e-10)
        return entropy(probabilities, base=2)

    def entropy_to_confidence(self, ent, num_bins):
        """Convert entropy to confidence score (0–10)"""
        max_entropy = np.log2(num_bins)
        normalized = 1.0 - (ent / (max_entropy + 1e-10))
        return float(normalized * 10.0)

    # =====================================================
    # From Multiple Samples
    # =====================================================
    def calculate_confidence_from_samples(self, score_samples):
        """
        Calculate confidence from multiple prediction samples

        Args:
            score_samples: list of numeric predictions (int or float)

        Returns:
            dict with mean_score, std_score, entropy, confidence, and score_distribution
        """
        scores = np.array(score_samples, dtype=float)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # Dynamically detect range
        min_s, max_s = float(np.min(scores)), float(np.max(scores))
        if np.isclose(min_s, max_s):
            # All predictions identical → full confidence
            return {
                "mean_score": mean_score,
                "std_score": std_score,
                "entropy": 0.0,
                "confidence": 10.0,
                "score_distribution": [1.0],
            }

        # Define bins dynamically
        bins = 10
        hist, _ = np.histogram(scores, bins=bins, range=(min_s, max_s))
        probabilities = hist / (len(scores) + 1e-10)

        ent = self.calculate_entropy(probabilities)
        confidence = self.entropy_to_confidence(ent, num_bins=bins)

        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "entropy": ent,
            "confidence": confidence,
            "score_distribution": probabilities.tolist(),
            "detected_range": [min_s, max_s],
        }

    # =====================================================
    # Optional: From Logits
    # =====================================================
    def calculate_confidence_from_logits(self, logits):
        """Compute confidence directly from logits (for classification tasks)"""
        if HAS_TORCH and 'torch' in str(type(logits)):
            probabilities = F.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy()
        else:
            probabilities = np.array(logits)
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

        ent = self.calculate_entropy(probabilities)
        confidence = self.entropy_to_confidence(ent, num_bins=len(probabilities))

        score_values = np.linspace(self.default_min, self.default_max, len(probabilities))
        mean_score = float(np.sum(probabilities * score_values))

        return {
            "mean_score": mean_score,
            "entropy": ent,
            "confidence": confidence,
            "score_distribution": probabilities.tolist(),
        }


# =====================================================
# Quick Self-Test
# =====================================================
if __name__ == "__main__":
    calc = ConfidenceCalculator()

    # Case A: Continuous decimals (0–1)
    low_range = [0.12, 0.25, 0.55, 0.61, 0.49, 0.7, 0.9, 0.3, 0.8, 0.4]
    print("Decimals 0–1 →", calc.calculate_confidence_from_samples(low_range))

    # Case B: Integer scores (1–10)
    int_scores = [7, 8, 8, 9, 7, 8, 8, 9, 8, 8]
    print("Integers 1–10 →", calc.calculate_confidence_from_samples(int_scores))

    # Case C: Identical predictions
    identical = [0.5] * 10
    print("Identical predictions →", calc.calculate_confidence_from_samples(identical))

