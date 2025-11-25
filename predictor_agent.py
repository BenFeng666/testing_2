"""
Predictor Agent for LipoAgent Multi-Agent Framework
Fast simultaneous prediction of lipid toxicity and delivery efficiency
with uncertainty estimation + detailed reasoning.
"""

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from confidence_calculator import ConfidenceCalculator


class PredictorAgent:
    """
    Predictor Agent that simultaneously predicts:
    - Lipid toxicity (1-10)
    - Delivery efficiency (1-10)
    - Detailed mechanistic reasoning
    - Confidence / uncertainty estimation
    """

    def __init__(
        self,
        base_model_path=None,
        lora_path=None,
        base_model_obj=None,
        tokenizer_obj=None,
        score_min=1,
        score_max=10,
        device="auto"
    ):
        """
        Initialize Predictor Agent.
        This version is compatible with the pipeline:
        - Accepts base_model_obj + tokenizer_obj (preloaded)
        - Or loads its own from paths.
        """

        self.score_min = score_min
        self.score_max = score_max
        self.device = device

        # 1. Use shared pre-loaded model if provided
        if base_model_obj is not None and tokenizer_obj is not None:
            print("[Predictor Agent] Using shared model from pipeline.")
            self.base_model = base_model_obj
            self.tokenizer = tokenizer_obj
        else:
            # 2. Otherwise load from disk
            print(f"[Predictor Agent] Loading model from {base_model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                padding_side='right'
            )

            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                device_map=device,
                torch_dtype=torch.float16
            )

        # 3. Load LoRA adapter
        print(f"[Predictor Agent] Loading LoRA weights from {lora_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.model.eval()

        self.confidence_calculator = ConfidenceCalculator(score_min, score_max)

    # =====================================================================
    # FAST PREDICT FUNCTION
    # =====================================================================
    def predict(self, smiles_structure, num_samples=5, temperature=0.6, top_p=0.8, max_length=128):
        """
        Fast prediction method:
        - Generates full reasoning ONCE
        - Score samples use short "score-only" generation

        Toxicity is binary:
        - 0 = not toxic
        - 1 = toxic
        """

        # ===== 1. Build prompts =====
        full_prompt = f"""Analyze this lipid molecule for LNP delivery systems: {smiles_structure}

Please provide:
1. Toxicity score (0 for not toxic and 1 for toxic)
2. Delivery efficiency score (1-10)
3. Detailed reasoning.

Format:
Toxicity: X
Efficiency: Y
Reasoning: ..."""

        score_only_prompt = f"""Analyze this lipid: {smiles_structure}
Return only the numerical toxicity and efficiency scores.

Toxicity must be 0 (not toxic) or 1 (toxic).
Efficiency must be from 1 to 10.

Format:
Toxicity: X
Efficiency: Y
"""

        # ===== 2. Build messages =====
        full_msg = [
            {
                "role": "system",
                "content": "You are an expert in lipid nanoparticles. Provide binary toxicity (0 or 1), efficiency (1–10), and reasoning."
            },
            {"role": "user", "content": full_prompt}
        ]

        score_only_msg = [
            {
                "role": "system",
                "content": "Return ONLY toxicity (0 or 1) and efficiency (1–10) numbers, no explanation."
            },
            {"role": "user", "content": score_only_prompt}
        ]

        # ===== Apply chat template once =====
        full_text = self.tokenizer.apply_chat_template(
            full_msg,
            tokenize=False,
            add_generation_prompt=True
        )
        score_text = self.tokenizer.apply_chat_template(
            score_only_msg,
            tokenize=False,
            add_generation_prompt=True
        )

        full_inputs = self.tokenizer([full_text], return_tensors="pt").to(self.model.device)
        score_inputs = self.tokenizer([score_text], return_tensors="pt").to(self.model.device)

        # ==================================================================
        # STEP 3: Generate full reasoning (ONCE)
        # ==================================================================
        with torch.no_grad():
            full_out = self.model.generate(
                full_inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )

        trim = full_out[0][len(full_inputs.input_ids[0]):]
        full_response = self.tokenizer.decode(trim, skip_special_tokens=True)
        parsed_full = self._parse_response(full_response)

        # ==================================================================
        # STEP 4: Fast score-only sampling (seeded with full pass)
        # ==================================================================
        toxicity_samples = []
        efficiency_samples = []

        full_tox = parsed_full.get("toxicity")
        full_eff = parsed_full.get("efficiency")
        if full_tox is not None:
            toxicity_samples.append(full_tox)
        if full_eff is not None:
            efficiency_samples.append(full_eff)

        for _ in range(num_samples):
            with torch.no_grad():
                out = self.model.generate(
                    score_inputs.input_ids,
                    max_new_tokens=16,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9
                )

            trim2 = out[0][len(score_inputs.input_ids[0]):]
            resp = self.tokenizer.decode(trim2, skip_special_tokens=True)
            parsed = self._parse_response(resp)

            if parsed["toxicity"] is not None:
                toxicity_samples.append(parsed["toxicity"])

            if parsed["efficiency"] is not None:
                efficiency_samples.append(parsed["efficiency"])

        # ==================================================================
        # STEP 5: Confidence estimation
        # ==================================================================
        tox_result = (
            self.confidence_calculator.calculate_confidence_from_samples(toxicity_samples)
            if len(toxicity_samples) > 0 else None
        )

        eff_result = (
            self.confidence_calculator.calculate_confidence_from_samples(efficiency_samples)
            if len(efficiency_samples) > 0 else None
        )

        return {
            "smiles": smiles_structure,
            "toxicity": {
                "score": tox_result["mean_score"] if tox_result else None,
                "confidence": tox_result["confidence"] if tox_result else 0.0,
                "std": tox_result["std_score"] if tox_result else None,
                "samples": toxicity_samples,
            },
            "efficiency": {
                "score": eff_result["mean_score"] if eff_result else None,
                "confidence": eff_result["confidence"] if eff_result else 0.0,
                "std": eff_result["std_score"] if eff_result else None,
                "samples": efficiency_samples,
            },
            "reasoning": parsed_full.get("reasoning", "No reasoning found."),
            "overall_confidence": min(
                tox_result["confidence"] if tox_result else 0.0,
                eff_result["confidence"] if eff_result else 0.0
            )
        }


    # =====================================================================
    # RESPONSE PARSER
    # =====================================================================
    def _parse_response(self, response):
        """
        Extract toxicity, efficiency, reasoning from generated text
        """
        result = {"toxicity": None, "efficiency": None, "reasoning": ""}

        # Toxicity
        tox_patterns = [
            r"[Tt]oxicity[:\s]+(\d+(?:\.\d+)?)",
            r"[Tt]oxic[:\s]+(\d+(?:\.\d+)?)"
        ]
        for p in tox_patterns:
            m = re.search(p, response)
            if m:
                try:
                    val = float(m.group(1))
                    if self.score_min <= val <= self.score_max:
                        result["toxicity"] = val
                        break
                except Exception:
                    pass

        # Efficiency
        eff_patterns = [
            r"[Ee]fficiency[:\s]+(\d+(?:\.\d+)?)",
            r"[Ee]fficient[:\s]+(\d+(?:\.\d+)?)",
            r"[Dd]elivery[:\s]+(\d+(?:\.\d+)?)"
        ]
        for p in eff_patterns:
            m = re.search(p, response)
            if m:
                try:
                    val = float(m.group(1))
                    if self.score_min <= val <= self.score_max:
                        result["efficiency"] = val
                        break
                except Exception:
                    pass

        # Reasoning
        reason_patterns = [
            r"[Rr]easoning[:\s]+(.*)",
            r"[Ee]xplanation[:\s]+(.*)"
        ]
        for p in reason_patterns:
            m = re.search(p, response, re.DOTALL)
            if m:
                result["reasoning"] = m.group(1).strip()
                break

        return result

    def is_low_confidence(self, prediction, confidence_threshold=6.0):
        """Check if prediction confidence is low."""
        return prediction["overall_confidence"] < confidence_threshold
