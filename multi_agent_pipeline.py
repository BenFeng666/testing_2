"""
Multi-Agent Pipeline for LipoAgent Framework
Implements Predictor-Verifier negotiation with human-in-the-loop fallback
"""

import os
import json
import yaml
import pandas as pd
from datetime import datetime
from predictor_agent import PredictorAgent
from verifier_agent import VerifierAgent
from human_feedback import HumanFeedbackInterface


class MultiAgentPipeline:
    """
    Multi-agent pipeline with Predictor and Verifier agents
    If agents cannot agree within 2 loops, human feedback is requested
    """
    
    def __init__(self, config_path="config.yaml"):

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("="*80)
        print("LIPOAGENT MULTI-AGENT PIPELINE INITIALIZED")
        print("="*80)
        print(f"Configuration loaded from: {config_path}")
        
        os.makedirs(self.config['data']['output_dir'], exist_ok=True)
        
        self.predictor_agent = None
        self.verifier_agent = None
        self.feedback_interface = HumanFeedbackInterface(
            self.config['data']['feedback_history_file']
        )
        
        self.all_molecules = []
        self.agreed_molecules = []
        self.disagreed_molecules = []
        self.human_feedback_molecules = []
        
        self.max_negotiation_loops = self.config.get('multi_agent', {}).get('max_negotiation_loops', 2)
        self.consensus_threshold = self.config.get('multi_agent', {}).get('consensus_threshold', 0.8)
        self.low_confidence_threshold = self.config.get('multi_agent', {}).get('low_confidence_threshold', 6.0)
    

    def load_agents(self):
        print("\nLoading agents...")

        self.predictor_agent = PredictorAgent(
            base_model_path=self.config['model']['base_model_path'],
            lora_path=self.config['model']['lora_path'],
            score_min=self.config['prediction']['score_min'],
            score_max=self.config['prediction']['score_max']
        )
        print("[Predictor Agent] Loaded successfully!")
        
        self.verifier_agent = VerifierAgent(
            base_model_path=self.config['model']['base_model_path']
        )
        print("[Verifier Agent] Loaded successfully!")
    


    # ============================================================
    # 🔧 FIXED: Proper SMILES extraction from JSONL or Excel
    # ============================================================
    def load_test_set(self):
        path = self.config["data"]["test_set"]
        print(f"\nLoading test set from: {path}")

        # Load JSONL / Excel
        if path.endswith(".jsonl"):
            df = pd.read_json(path, lines=True)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported test set format: {path}")

        print(f"Dataset shape: {df.shape}")
        print(f"Available columns: {df.columns.tolist()}")

        # Look for SMILES or STRUCTURE fields
        structure_col = None
        for col in df.columns:
            if "smiles" in col.lower() or "structure" in col.lower():
                structure_col = col
                break
        
        if structure_col is None:
            structure_col = df.columns[0]

        print(f"Using column '{structure_col}' as SMILES column.")

        self.all_molecules = []

        for entry in df[structure_col]:

            # 🔧 FIX: JSONL contains chat messages → extract SMILES from user's message
            if isinstance(entry, list):
                smiles = None
                for msg in entry:
                    if msg.get("role") == "user":
                        smiles = msg.get("content")
                if smiles is None:
                    continue
            else:
                smiles = str(entry).strip()

            # clean
            if smiles and len(smiles) > 4:
                self.all_molecules.append({
                    "smiles": smiles,
                    "status": "pending",
                    "negotiation_history": []
                })

        print(f"Loaded {len(self.all_molecules)} molecules.")
        return self.all_molecules
    


    # ============================================================
    # 🔧 FIXED: No-crash formatting even when model outputs None
    # ============================================================
    def _safe_print_scores(self, prediction):
        tox = prediction['toxicity']['score']
        eff = prediction['efficiency']['score']

        tox_c = prediction['toxicity']['confidence']
        eff_c = prediction['efficiency']['confidence']

        tox_str = f"{tox:.2f}" if tox is not None else "N/A"
        eff_str = f"{eff:.2f}" if eff is not None else "N/A"

        print(f"  Toxicity: {tox_str} (confidence: {tox_c:.2f})")
        print(f"  Efficiency: {eff_str} (confidence: {eff_c:.2f})")
        print(f"  Overall Confidence: {prediction['overall_confidence']:.2f}")


    def negotiate_prediction(self, molecule):

        smiles = molecule['smiles']
        negotiation_history = []

        print("\n" + "="*60)
        print(f"Negotiating prediction for: {smiles[:50]}...")
        print("="*60)

        for loop_num in range(1, self.max_negotiation_loops + 1):

            print(f"\n[Loop {loop_num}] Starting negotiation...")
            prediction = self.predictor_agent.predict(
                smiles,
                num_samples=self.config['prediction']['num_samples'],
                temperature=self.config['prediction']['temperature'],
                top_p=self.config['prediction']['top_p'],
                max_length=self.config['prediction']['max_length']
            )

            # 🔧 use safe printer
            self._safe_print_scores(prediction)

            # Skip verification if prediction is None
            if prediction['toxicity']['score'] is None or prediction['efficiency']['score'] is None:
                print(f"[Loop {loop_num}] ✗ Invalid prediction, forcing human feedback.")
                return {
                    "smiles": smiles,
                    "agreed": False,
                    "needs_human_feedback": True,
                    "toxicity": prediction['toxicity'],
                    "efficiency": prediction['efficiency'],
                    "reasoning": prediction['reasoning'],
                    "overall_confidence": prediction['overall_confidence'],
                    "negotiation_history": negotiation_history,
                    "final_loop": loop_num
                }

            # High confidence → accept
            if not self.predictor_agent.is_low_confidence(prediction, self.low_confidence_threshold):
                print(f"[Loop {loop_num}] ✓ High confidence prediction. No verification needed.")
                return {
                    "smiles": smiles,
                    "agreed": True,
                    "toxicity": prediction['toxicity'],
                    "efficiency": prediction['efficiency'],
                    "reasoning": prediction['reasoning'],
                    "overall_confidence": prediction['overall_confidence'],
                    "negotiation_history": negotiation_history,
                    "final_loop": loop_num,
                    "verified": False
                }

            # Low confidence → call verifier
            print(f"[Loop {loop_num}] Low confidence detected. Verifier checking...")

            verification = self.verifier_agent.verify(
                smiles,
                prediction['toxicity']['score'],
                prediction['efficiency']['score'],
                prediction['reasoning']
            )

            print(f"  Consistency: {verification['consistency_level']}")
            print(f"  Verification Confidence: {verification['verification_confidence']}")

            if self.verifier_agent.agrees_with_prediction(verification):
                print(f"[Loop {loop_num}] ✓ Agents reached agreement!")
                return {
                    "smiles": smiles,
                    "agreed": True,
                    "toxicity": prediction['toxicity'],
                    "efficiency": prediction['efficiency'],
                    "reasoning": prediction['reasoning'],
                    "overall_confidence": prediction['overall_confidence'],
                    "negotiation_history": negotiation_history,
                    "final_loop": loop_num
                }

            # disagreement
            negotiation_history.append({
                "loop": loop_num,
                "agreed": False,
                "prediction": prediction,
                "verification": verification
            })

        print(f"\n✗ No agreement after {self.max_negotiation_loops} loops. Requesting human feedback...")

        return {
            "smiles": smiles,
            "agreed": False,
            "needs_human_feedback": True,
            "toxicity": prediction['toxicity'],
            "efficiency": prediction['efficiency'],
            "reasoning": prediction['reasoning'],
            "overall_confidence": prediction['overall_confidence'],
            "negotiation_history": negotiation_history,
            "final_loop": self.max_negotiation_loops
        }


    # ============================================================
    # Keeping other functions the same (no changes needed)
    # ============================================================

    def process_molecules(self):
        print(f"\n{'='*80}")
        print("PROCESSING MOLECULES WITH MULTI-AGENT FRAMEWORK")
        print(f"{'='*80}")
        
        for i, molecule in enumerate(self.all_molecules):
            print(f"\n[{i+1}/{len(self.all_molecules)}] Processing molecule...")
            
            result = self.negotiate_prediction(molecule)

            molecule.update(result)
            molecule['last_updated'] = datetime.now().isoformat()
            
            if result['agreed']:
                molecule['status'] = 'agreed'
                self.agreed_molecules.append(molecule)
            elif result.get('needs_human_feedback'):
                molecule['status'] = 'needs_human_feedback'
                self.disagreed_molecules.append(molecule)
            else:
                molecule['status'] = 'disagreed'
                self.disagreed_molecules.append(molecule)


    def collect_human_feedback(self):
        if not self.disagreed_molecules:
            print("\nNo molecules need human feedback.")
            return
        
        print(f"\n{'='*80}")
        print(f"COLLECTING HUMAN FEEDBACK")
        print(f"{'='*80}")

        for m in list(self.disagreed_molecules):
            if not m.get("needs_human_feedback"):
                continue
            feedback = self.feedback_interface.collect_efficiency_feedback(m, round_num=1)
            m["true_efficiency"] = feedback["true_efficiency"]
            m["human_feedback"] = feedback
            m["status"] = "human_feedback_collected"
            self.human_feedback_molecules.append(m)
            self.disagreed_molecules.remove(m)


    def save_results(self):
        out_dir = self.config['data']['output_dir']
        
        with open(os.path.join(out_dir, 'agreed_molecules.json'), 'w') as f:
            json.dump(self.agreed_molecules, f, indent=2)

        with open(os.path.join(out_dir, 'disagreed_molecules.json'), 'w') as f:
            json.dump(self.disagreed_molecules, f, indent=2)

        with open(os.path.join(out_dir, 'human_feedback_molecules.json'), 'w') as f:
            json.dump(self.human_feedback_molecules, f, indent=2)

        with open(self.config['data']['predictions_file'], 'w') as f:
            json.dump(self.all_molecules, f, indent=2)

        print("\nResults saved.")


    def run(self):
        print("\n" + "="*80)
        print("STARTING LIPOAGENT MULTI-AGENT PIPELINE")
        print("="*80)
        
        self.load_agents()
        self.load_test_set()
        self.process_molecules()
        
        if self.disagreed_molecules:
            self.collect_human_feedback()
        
        self.save_results()


if __name__ == "__main__":
    pipeline = MultiAgentPipeline("config.yaml")
    pipeline.run()
