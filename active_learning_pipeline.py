"""
Active Learning Pipeline with Human Feedback
Main pipeline for molecular score prediction with human-in-the-loop
"""

import os
import json
import yaml
import pandas as pd
from datetime import datetime
from model_predictor import MolecularScorePredictor
from human_feedback import HumanFeedbackInterface
from confidence_calculator import ConfidenceCalculator

class ActiveLearningPipeline:
    """Main pipeline for active learning with human feedback"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("="*80)
        print("ACTIVE LEARNING PIPELINE INITIALIZED")
        print("="*80)
        print(f"Configuration loaded from: {config_path}")
        
        # Create output directory
        os.makedirs(self.config['data']['output_dir'], exist_ok=True)
        
        # Initialize components
        self.predictor = None
        self.feedback_interface = HumanFeedbackInterface(
            self.config['data']['feedback_history_file']
        )
        
        # Data storage
        self.all_molecules = []
        self.target_smiles = []
        self.unsure_smiles = []
        self.predictions = []
        
    def load_model(self):
        """Load the finetuned model"""
        print("\nLoading model...")
        self.predictor = MolecularScorePredictor(
            base_model_path=self.config['model']['base_model_path'],
            lora_path=self.config['model']['lora_path'],
            score_min=self.config['prediction']['score_min'],
            score_max=self.config['prediction']['score_max']
        )
        print("Model loaded successfully!")
    
    def load_test_set(self):
        """Load test set from Excel file"""
        print(f"\nLoading test set from: {self.config['data']['test_set']}")
        
        # Read the Excel file
        df = pd.read_excel(self.config['data']['test_set'])
        
        print(f"Dataset shape: {df.shape}")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Find the Structure/SMILES column
        structure_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'structure' in col_lower or 'smiles' in col_lower or 'smile' in col_lower:
                structure_col = col
                break
        
        if structure_col is None:
            # If only one column, use it
            if len(df.columns) == 1:
                structure_col = df.columns[0]
            else:
                # Try to find a column with long strings (likely SMILES)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        avg_len = df[col].astype(str).str.len().mean()
                        if avg_len > 20:  # SMILES are usually long
                            structure_col = col
                            break
                
                if structure_col is None:
                    structure_col = df.columns[0]
        
        print(f"Using column '{structure_col}' as SMILES structure")
        
        # Extract SMILES
        smiles_list = df[structure_col].dropna().tolist()
        
        # Convert to string and filter valid SMILES
        self.all_molecules = []
        for smiles in smiles_list:
            smiles_str = str(smiles).strip()
            if smiles_str and smiles_str != 'nan' and len(smiles_str) > 5:
                self.all_molecules.append({
                    'smiles': smiles_str,
                    'predicted_score': None,
                    'confidence': None,
                    'status': 'pending'
                })
        
        print(f"Loaded {len(self.all_molecules)} molecules from test set")
        
        return self.all_molecules
    
    def predict_molecules(self, molecules, update_existing=False):
        """
        Predict scores for molecules
        
        Args:
            molecules: List of molecule dictionaries
            update_existing: Whether to update existing predictions
            
        Returns:
            list: Updated molecule list with predictions
        """
        print(f"\nPredicting scores for {len(molecules)} molecules...")
        
        smiles_to_predict = []
        indices_to_update = []
        
        for i, mol in enumerate(molecules):
            if update_existing or mol.get('predicted_score') is None:
                smiles_to_predict.append(mol['smiles'])
                indices_to_update.append(i)
        
        if not smiles_to_predict:
            print("No molecules need prediction.")
            return molecules
        
        print(f"Predicting {len(smiles_to_predict)} molecules...")
        
        # Batch prediction
        predictions = self.predictor.predict_batch(
            smiles_to_predict,
            num_samples=self.config['prediction']['num_samples'],
            temperature=self.config['prediction']['temperature'],
            top_p=self.config['prediction']['top_p'],
            max_length=self.config['prediction']['max_length'],
            verbose=True
        )
        
        # Update molecules with predictions
        for idx, pred in zip(indices_to_update, predictions):
            molecules[idx].update({
                'predicted_score': pred.get('predicted_score'),
                'confidence': pred.get('confidence'),
                'std': pred.get('std'),
                'entropy': pred.get('entropy'),
                'score_distribution': pred.get('score_distribution'),
                'last_updated': datetime.now().isoformat()
            })
        
        return molecules
    
    def categorize_molecules(self, molecules):
        """
        Categorize molecules into target and unsure based on thresholds
        
        Args:
            molecules: List of molecules with predictions
            
        Returns:
            tuple: (target_smiles, unsure_smiles)
        """
        target = []
        unsure = []
        
        confidence_threshold = self.config['thresholds']['confidence_threshold']
        score_threshold = self.config['thresholds']['target_score_threshold']
        
        for mol in molecules:
            score = mol.get('predicted_score')
            confidence = mol.get('confidence')
            
            if score is None or confidence is None:
                continue
            
            if confidence >= confidence_threshold and score >= score_threshold:
                mol['status'] = 'target'
                target.append(mol)
            else:
                mol['status'] = 'unsure'
                unsure.append(mol)
        
        return target, unsure
    
    def save_results(self):
        """Save all results to files"""
        # Save target smiles
        target_file = self.config['data']['target_smiles_file']
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(self.target_smiles, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(self.target_smiles)} target molecules to {target_file}")
        
        # Save unsure smiles
        unsure_file = self.config['data']['unsure_smiles_file']
        with open(unsure_file, 'w', encoding='utf-8') as f:
            json.dump(self.unsure_smiles, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.unsure_smiles)} unsure molecules to {unsure_file}")
        
        # Save all predictions
        predictions_file = self.config['data']['predictions_file']
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_molecules, f, indent=2, ensure_ascii=False)
        print(f"Saved all {len(self.all_molecules)} predictions to {predictions_file}")
    
    def run_feedback_round(self, round_num):
        """
        Run one round of human feedback
        
        Args:
            round_num: Current round number
            
        Returns:
            list: Feedback records
        """
        print(f"\n{'='*80}")
        print(f"STARTING FEEDBACK ROUND {round_num}")
        print(f"{'='*80}")
        
        # Select pairs
        num_pairs = self.config['human_feedback']['num_pairs_per_round']
        strategy = self.config['human_feedback']['selection_strategy']
        
        pairs = self.feedback_interface.select_pairs(
            self.unsure_smiles,
            num_pairs=num_pairs,
            strategy=strategy
        )
        
        if not pairs:
            print("No pairs available for feedback.")
            return []
        
        print(f"Selected {len(pairs)} pairs using '{strategy}' strategy")
        
        # Collect feedback
        feedback_records = self.feedback_interface.collect_feedback(pairs, round_num)
        
        return feedback_records
    
    def run(self):
        """Run the complete active learning pipeline"""
        print("\n" + "="*80)
        print("STARTING ACTIVE LEARNING PIPELINE")
        print("="*80)
        
        # Step 1: Load model
        self.load_model()
        
        # Step 2: Load test set
        self.load_test_set()
        
        # Step 3: Initial prediction
        if self.config['active_learning']['initial_prediction']:
            print("\n" + "="*80)
            print("INITIAL PREDICTION PHASE")
            print("="*80)
            
            self.all_molecules = self.predict_molecules(self.all_molecules)
            
            # Categorize
            self.target_smiles, self.unsure_smiles = self.categorize_molecules(self.all_molecules)
            
            print(f"\nInitial categorization:")
            print(f"  Target molecules: {len(self.target_smiles)}")
            print(f"  Unsure molecules: {len(self.unsure_smiles)}")
            
            # Save initial results
            self.save_results()
        
        # Step 4: Human feedback loop
        max_rounds = self.config['human_feedback']['max_rounds']
        
        for round_num in range(1, max_rounds + 1):
            if not self.unsure_smiles:
                print(f"\nNo unsure molecules remaining. Stopping feedback loop.")
                break
            
            # Run feedback round
            feedback = self.run_feedback_round(round_num)
            
            if not feedback:
                print(f"\nNo feedback collected in round {round_num}.")
                continue
            
            # Update predictions for unsure molecules
            if self.config['active_learning']['update_after_feedback']:
                print(f"\nRe-predicting unsure molecules after feedback...")
                self.unsure_smiles = self.predict_molecules(self.unsure_smiles, update_existing=True)
                
                # Re-categorize
                newly_confident = []
                still_unsure = []
                
                confidence_threshold = self.config['thresholds']['confidence_threshold']
                score_threshold = self.config['thresholds']['target_score_threshold']
                
                for mol in self.unsure_smiles:
                    score = mol.get('predicted_score')
                    confidence = mol.get('confidence')
                    
                    if confidence is not None and confidence >= confidence_threshold:
                        if score is not None and score >= score_threshold:
                            mol['status'] = 'target'
                            newly_confident.append(mol)
                            self.target_smiles.append(mol)
                        else:
                            still_unsure.append(mol)
                    else:
                        still_unsure.append(mol)
                
                self.unsure_smiles = still_unsure
                
                print(f"\nAfter round {round_num}:")
                print(f"  Newly confident target molecules: {len(newly_confident)}")
                print(f"  Total target molecules: {len(self.target_smiles)}")
                print(f"  Remaining unsure molecules: {len(self.unsure_smiles)}")
                
                # Save updated results
                self.save_results()
            
            # Ask if user wants to continue
            print(f"\n{'='*80}")
            if round_num < max_rounds and self.unsure_smiles:
                response = input(f"Continue to round {round_num + 1}? (y/n): ").strip().lower()
                if response != 'y':
                    print("Stopping feedback loop as requested.")
                    break
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        print(f"\nFinal Results:")
        print(f"  Total molecules processed: {len(self.all_molecules)}")
        print(f"  Target molecules (high confidence & score): {len(self.target_smiles)}")
        print(f"  Unsure molecules: {len(self.unsure_smiles)}")
        print(f"  Feedback rounds completed: {round_num}")
        print(f"\nFeedback Summary:")
        print(self.feedback_interface.get_feedback_summary())
        print(f"\nResults saved to: {self.config['data']['output_dir']}/")
        print("="*80)

if __name__ == "__main__":
    # Run the pipeline
    pipeline = ActiveLearningPipeline("config.yaml")
    pipeline.run()

