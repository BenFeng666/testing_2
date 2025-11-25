"""
Verifier Agent for LipoAgent Multi-Agent Framework
Validates logical consistency between predicted scores and explanations
"""

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


class VerifierAgent:
    """
    Verifier Agent that inspects low-confidence predictions and validates
    logical consistency between predicted scores and explanations
    """

    def __init__(self, base_model_path, device="auto"):
        self.device = device

        print(f"[Verifier Agent] Loading model from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch.float16
        )
        self.model.eval()

    def verify(self, smiles_structure, predicted_toxicity, predicted_efficiency, reasoning):
        """
        Verify logical consistency between predicted scores and reasoning.
        Uses a plain text prompt (no chat template).
        """

        # ------------------------------------------------------------
        # PLAIN PROMPT MODE (no chat, consistent with PredictorAgent)
        # ------------------------------------------------------------
        prompt = f"""
You are an expert verification analyst for lipid nanoparticle (LNP) delivery systems.

Evaluate the following prediction:

Molecule: {smiles_structure}

Predicted Toxicity: {predicted_toxicity}/10
Predicted Efficiency: {predicted_efficiency}/10
Reasoning: {reasoning}

Please verify:
1. Is the reasoning logically consistent with the predicted scores?
2. Are the scores reasonable given the molecular structure?
3. If there are inconsistencies, what would be more appropriate scores?

Respond in the following exact format:
Consistency: [YES/NO/PARTIAL]
Verification: [detailed analysis]
Suggested Toxicity: [score or N/A]
Suggested Efficiency: [score or N/A]
Confidence: [HIGH/MEDIUM/LOW]

Your response:
"""

        # Create model inputs
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        # Generate verification
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=450,
                do_sample=False,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract only generated text
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Parse structured verification
        verification = self._parse_verification(response)

        return {
            "smiles": smiles_structure,
            "is_consistent": verification["is_consistent"],
            "consistency_level": verification["consistency_level"],
            "verification_text": verification["verification_text"],
            "suggested_toxicity": verification["suggested_toxicity"],
            "suggested_efficiency": verification["suggested_efficiency"],
            "verification_confidence": verification["confidence"],
            "raw_response": response
        }

    def _parse_verification(self, response):
        """
        Parse verification response.
        """
        result = {
            "is_consistent": False,
            "consistency_level": "UNKNOWN",
            "verification_text": "",
            "suggested_toxicity": None,
            "suggested_efficiency": None,
            "confidence": "MEDIUM"
        }

        # Consistency
        patterns = [
            r"[Cc]onsistency[:\s]+(YES|NO|PARTIAL)"
        ]
        for p in patterns:
            m = re.search(p, response)
            if m:
                level = m.group(1).upper()
                result["consistency_level"] = level
                result["is_consistent"] = (level == "YES")
                break

        # Verification text
        text_patterns = [
            r"[Vv]erification[:\s]+(.*?)(?:Suggested|Confidence|\Z)",
            r"[Aa]nalysis[:\s]+(.*?)(?:Suggested|Confidence|\Z)"
        ]
        for p in text_patterns:
            m = re.search(p, response, re.DOTALL)
            if m:
                result["verification_text"] = m.group(1).strip()
                break

        # Suggested Toxicity
        m = re.search(r"[Ss]uggested\s+[Tt]oxicity[:\s]+(\d+(?:\.\d+)?|N/A)", response)
        if m:
            val = m.group(1)
            if val.upper() != "N/A":
                try:
                    result["suggested_toxicity"] = float(val)
                except:
                    pass

        # Suggested Efficiency
        m = re.search(r"[Ss]uggested\s+[Ee]fficiency[:\s]+(\d+(?:\.\d+)?|N/A)", response)
        if m:
            val = m.group(1)
            if val.upper() != "N/A":
                try:
                    result["suggested_efficiency"] = float(val)
                except:
                    pass

        # Confidence
        m = re.search(r"[Cc]onfidence[:\s]+(HIGH|MEDIUM|LOW)", response)
        if m:
            result["confidence"] = m.group(1).upper()

        # fallback if missing
        if not result["verification_text"]:
            result["verification_text"] = response.strip()

        return result

    def agrees_with_prediction(self, verification_result, tolerance=1.0):
        """
        Determine whether verifier agrees with original scores.
        """
        if verification_result["is_consistent"]:
            return True

        # If suggested new scores exist → disagreement
        if verification_result["suggested_toxicity"] is not None:
            return False

        if verification_result["consistency_level"] == "PARTIAL":
            return False

        return False
