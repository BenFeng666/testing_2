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
        """
        Initialize multi-agent pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("="*80)
        print("LIPOAGENT MULTI-AGENT PIPELINE INITIALIZED")
        print("="*80)
        print(f"Configuration loaded from: {config_path}")
        
        # Create output directory
        os.makedirs(self.config['data']['output_dir'], exist_ok=True)
        
        # Initialize agents
        self.predictor_agent = None
        self.verifier_agent = None
        self.feedback_interface = HumanFeedbackInterface(
            self.config['data']['feedback_history_file']
        )
        
        # Data storage
        self.all_molecules = []
        self.agreed_molecules = []
        self.disagreed_molecules = []
        self.human_feedback_molecules = []
        
        # Multi-agent settings
        self.max_negotiation_loops = self.config.get('multi_agent', {}).get('max_negotiation_loops', 2)
        self.consensus_threshold = self.config.get('multi_agent', {}).get('consensus_threshold', 0.8)
        self.low_confidence_threshold = self.config.get('multi_agent', {}).get('low_confidence_threshold', 6.0)

        # Optional: limit how many test samples to load from data file
        self.max_test_samples = self.config.get('data', {}).get('max_test_samples', None)
        if self.max_test_samples is not None:
            self.max_test_samples = int(self.max_test_samples)
    
    def load_agents(self):
        """Load Predictor and Verifier agents"""
        print("\nLoading agents...")
        
        # Load Predictor Agent (with finetuned model)
        self.predictor_agent = PredictorAgent(
            base_model_path=self.config['model']['base_model_path'],
            lora_path=self.config['model']['lora_path'],
            score_min=self.config['prediction']['score_min'],
            score_max=self.config['prediction']['score_max']
        )
        print("[Predictor Agent] Loaded successfully!")
        
        # Load Verifier Agent (base model, no finetuning)
        self.verifier_agent = VerifierAgent(
            base_model_path=self.config['model']['base_model_path']
        )
        print("[Verifier Agent] Loaded successfully!")
    
    def load_test_set(self):
        """Load test set from JSONL or Excel (auto-detect)."""

        test_path = self.config['data']['test_set']
        print(f"\nLoading test set from: {test_path}")

        if test_path.endswith(".jsonl"):
            print("Detected JSONL efficiency dataset — loading...")
            molecules = []
            import json, re
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    messages = item.get("messages", [])
                    if len(messages) < 2:
                        continue
                    user_msg = messages[1]["content"]
                    m = re.search(r"molecular structure:\s*(.+?)\??$", user_msg)
                    if not m:
                        continue
                    smiles = m.group(1).strip()
                    molecules.append({
                        "smiles": smiles,
                        "status": "pending",
                        "negotiation_history": []
                    })
            self.all_molecules = molecules
            print(f"Loaded {len(self.all_molecules)} molecules from JSONL test set")

            # Optional: limit number of samples
            if self.max_test_samples is not None:
                self.all_molecules = self.all_molecules[:self.max_test_samples]
                print(f"Using first {len(self.all_molecules)} molecules (max_test_samples={self.max_test_samples})")

            return self.all_molecules

        print("Detected Excel dataset — loading with pandas")
        df = pd.read_excel(test_path)
        structure_col = None
        for col in df.columns:
            if 'structure' in str(col).lower() or 'smiles' in str(col).lower():
                structure_col = col
                break
        if structure_col is None:
            structure_col = df.columns[0]
        
        self.all_molecules = []
        for smiles in df[structure_col].dropna().astype(str):
            if len(smiles.strip()) > 5:
                self.all_molecules.append({
                    "smiles": smiles.strip(),
                    "status": "pending",
                    "negotiation_history": []
                })
        print(f"Loaded {len(self.all_molecules)} molecules from Excel test set")

        # Optional: limit number of samples for Excel too
        if self.max_test_samples is not None:
            self.all_molecules = self.all_molecules[:self.max_test_samples]
            print(f"Using first {len(self.all_molecules)} molecules (max_test_samples={self.max_test_samples})")

        return self.all_molecules
    
    def negotiate_prediction(self, molecule):
        """
        Negotiate prediction between Predictor and Verifier agents
        
        Args:
            molecule: Molecule dictionary
            
        Returns:
            dict: Final prediction result with agreement status
        """
        smiles = molecule['smiles']
        negotiation_history = []
        
        print(f"\n{'='*60}")
        print(f"Negotiating prediction for: {smiles[:50]}...")
        print(f"{'='*60}")
        
        # Loop 1: Initial prediction and verification
        for loop_num in range(1, self.max_negotiation_loops + 1):
            print(f"\n[Loop {loop_num}] Starting negotiation...")
            
            # Step 1: Predictor makes prediction
            print(f"[Loop {loop_num}] Predictor Agent making prediction...")
            prediction = self.predictor_agent.predict(
                smiles,
                num_samples=self.config['prediction']['num_samples'],
                temperature=self.config['prediction']['temperature'],
                top_p=self.config['prediction']['top_p'],
                max_length=self.config['prediction']['max_length']
            )
            
            tox_score = prediction['toxicity']['score']
            tox_conf = prediction['toxicity']['confidence']
            eff_score = prediction['efficiency']['score']
            eff_conf = prediction['efficiency']['confidence']
            overall_conf = prediction['overall_confidence']

            tox_score_str = f"{tox_score:.2f}" if tox_score is not None else "N/A"
            tox_conf_str = f"{tox_conf:.2f}" if tox_conf is not None else "N/A"
            eff_score_str = f"{eff_score:.2f}" if eff_score is not None else "N/A"
            eff_conf_str = f"{eff_conf:.2f}" if eff_conf is not None else "N/A"
            overall_conf_str = f"{overall_conf:.2f}" if overall_conf is not None else "N/A"

            print(f"  Toxicity: {tox_score_str} (confidence: {tox_conf_str})")
            print(f"  Efficiency: {eff_score_str} (confidence: {eff_conf_str})")
            print(f"  Overall Confidence: {overall_conf_str}")

            # Step 2: Check if low confidence - if so, verify
            if self.predictor_agent.is_low_confidence(prediction, self.low_confidence_threshold):
                print(f"[Loop {loop_num}] Low confidence detected. Verifier Agent checking...")
                
                # Step 3: Verifier verifies
                verification = self.verifier_agent.verify(
                    smiles,
                    prediction['toxicity']['score'],
                    prediction['efficiency']['score'],
                    prediction['reasoning']
                )
                
                print(f"  Consistency: {verification['consistency_level']}")
                print(f"  Verification Confidence: {verification['verification_confidence']}")
                
                # Step 4: Check if verifier agrees
                if self.verifier_agent.agrees_with_prediction(verification):
                    print(f"[Loop {loop_num}] ✓ Agents reached agreement!")
                    agreement_result = {
                        'loop': loop_num,
                        'agreed': True,
                        'final_toxicity': prediction['toxicity']['score'],
                        'final_efficiency': prediction['efficiency']['score'],
                        'reasoning': prediction['reasoning'],
                        'confidence': prediction['overall_confidence'],
                        'verification': verification
                    }
                    negotiation_history.append(agreement_result)
                    return {
                        'smiles': smiles,
                        'agreed': True,
                        'toxicity': prediction['toxicity'],
                        'efficiency': prediction['efficiency'],
                        'reasoning': prediction['reasoning'],
                        'overall_confidence': prediction['overall_confidence'],
                        'negotiation_history': negotiation_history,
                        'final_loop': loop_num
                    }
                else:
                    # Verifier disagrees - use suggested scores if available
                    print(f"[Loop {loop_num}] ✗ Verifier disagrees. Suggested scores:")
                    if verification['suggested_toxicity'] is not None:
                        print(f"  Suggested Toxicity: {verification['suggested_toxicity']}")
                    if verification['suggested_efficiency'] is not None:
                        print(f"  Suggested Efficiency: {verification['suggested_efficiency']}")
                    
                    # Record disagreement
                    negotiation_history.append({
                        'loop': loop_num,
                        'agreed': False,
                        'prediction': prediction,
                        'verification': verification
                    })
                    
                    # If we have suggestions and not last loop, update prediction for next iteration
                    if loop_num < self.max_negotiation_loops:
                        if verification['suggested_efficiency'] is not None:
                            # Update prediction with verifier's suggestion for next loop
                            print(f"[Loop {loop_num}] Updating prediction based on verifier feedback...")
                            # Note: In a real implementation, we might refine the prompt with verifier feedback
                            continue
            else:
                # High confidence - no need for verification
                print(f"[Loop {loop_num}] ✓ High confidence prediction. No verification needed.")
                return {
                    'smiles': smiles,
                    'agreed': True,
                    'toxicity': prediction['toxicity'],
                    'efficiency': prediction['efficiency'],
                    'reasoning': prediction['reasoning'],
                    'overall_confidence': prediction['overall_confidence'],
                    'negotiation_history': negotiation_history,
                    'final_loop': loop_num,
                    'verified': False
                }
        
        # No agreement reached after max loops
        print(f"\n✗ No agreement reached after {self.max_negotiation_loops} loops.")
        print("Requesting human feedback...")
        
        return {
            'smiles': smiles,
            'agreed': False,
            'toxicity': prediction['toxicity'],
            'efficiency': prediction['efficiency'],
            'reasoning': prediction['reasoning'],
            'overall_confidence': prediction['overall_confidence'],
            'negotiation_history': negotiation_history,
            'final_loop': self.max_negotiation_loops,
            'needs_human_feedback': True
        }
    
    def process_molecules(self):
        """Process all molecules through multi-agent negotiation"""
        print(f"\n{'='*80}")
        print("PROCESSING MOLECULES WITH MULTI-AGENT FRAMEWORK")
        print(f"{'='*80}")
        
        for i, molecule in enumerate(self.all_molecules):
            print(f"\n[{i+1}/{len(self.all_molecules)}] Processing molecule...")
            
            # Negotiate prediction
            result = self.negotiate_prediction(molecule)
            
            # Update molecule with result
            molecule.update(result)
            molecule['last_updated'] = datetime.now().isoformat()
            
            # Categorize based on agreement
            if result['agreed']:
                molecule['status'] = 'agreed'
                self.agreed_molecules.append(molecule)
                print(f"✓ Added to agreed molecules")
            elif result.get('needs_human_feedback'):
                molecule['status'] = 'needs_human_feedback'
                self.disagreed_molecules.append(molecule)
                print(f"✗ Added to disagreed molecules (needs human feedback)")
            else:
                molecule['status'] = 'disagreed'
                self.disagreed_molecules.append(molecule)
                print(f"✗ Added to disagreed molecules")
    
    def collect_human_feedback(self):
        """Collect human feedback for molecules where agents disagreed"""
        if not self.disagreed_molecules:
            print("\nNo molecules need human feedback.")
            return
        
        print(f"\n{'='*80}")
        print(f"COLLECTING HUMAN FEEDBACK")
        print(f"{'='*80}")
        print(f"{len(self.disagreed_molecules)} molecules need human feedback.\n")
        
        for i, molecule in enumerate(self.disagreed_molecules):
            if not molecule.get('needs_human_feedback'):
                continue
            
            print(f"\n[{i+1}/{len(self.disagreed_molecules)}]")
            
            # Collect feedback
            feedback = self.feedback_interface.collect_efficiency_feedback(
                molecule,
                round_num=1
            )
            
            # Update molecule with true efficiency
            molecule['true_efficiency'] = feedback['true_efficiency']
            molecule['status'] = 'human_feedback_collected'
            molecule['human_feedback'] = feedback
            self.human_feedback_molecules.append(molecule)
            
            # Remove from disagreed list
            self.disagreed_molecules.remove(molecule)
    
    def save_results(self):
        """Save all results to files"""
        # Save agreed molecules
        agreed_file = os.path.join(self.config['data']['output_dir'], 'agreed_molecules.json')
        with open(agreed_file, 'w', encoding='utf-8') as f:
            json.dump(self.agreed_molecules, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(self.agreed_molecules)} agreed molecules to {agreed_file}")
        
        # Save disagreed molecules
        disagreed_file = os.path.join(self.config['data']['output_dir'], 'disagreed_molecules.json')
        with open(disagreed_file, 'w', encoding='utf-8') as f:
            json.dump(self.disagreed_molecules, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.disagreed_molecules)} disagreed molecules to {disagreed_file}")
        
        # Save human feedback molecules
        human_feedback_file = os.path.join(self.config['data']['output_dir'], 'human_feedback_molecules.json')
        with open(human_feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.human_feedback_molecules, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.human_feedback_molecules)} human feedback molecules to {human_feedback_file}")
        
        # Save all molecules
        all_file = self.config['data']['predictions_file']
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_molecules, f, indent=2, ensure_ascii=False)
        print(f"Saved all {len(self.all_molecules)} molecules to {all_file}")
    
    def run(self):
        """Run the complete multi-agent pipeline"""
        print("\n" + "="*80)
        print("STARTING LIPOAGENT MULTI-AGENT PIPELINE")
        print("="*80)
        
        # Step 1: Load agents
        self.load_agents()
        
        # Step 2: Load test set
        self.load_test_set()
        
        # Step 3: Process molecules through negotiation
        self.process_molecules()
        
        # Step 4: Collect human feedback for disagreed molecules
        if self.disagreed_molecules:
            self.collect_human_feedback()
        
        # Step 5: Save results
        self.save_results()
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        print(f"\nFinal Results:")
        print(f"  Total molecules processed: {len(self.all_molecules)}")
        print(f"  Agreed molecules: {len(self.agreed_molecules)}")
        print(f"  Disagreed molecules: {len(self.disagreed_molecules)}")
        print(f"  Human feedback collected: {len(self.human_feedback_molecules)}")
        print(f"\nResults saved to: {self.config['data']['output_dir']}/")
        print("="*80)


if __name__ == "__main__":
    # Run the multi-agent pipeline
    pipeline = MultiAgentPipeline("config.yaml")
    pipeline.run()
