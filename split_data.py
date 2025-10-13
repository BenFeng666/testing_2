"""
Split data_new.xlsx into training and test sets
"""

import pandas as pd
import numpy as np
import os
import json

def split_data(input_file, output_dir, test_size=100):
    """
    Load data, shuffle, and split into train/test sets
    
    Args:
        input_file: Path to input Excel file
        output_dir: Directory to save split data
        test_size: Number of samples for test set
    """
    print(f"Loading data from: {input_file}")
    
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show first few rows
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Use 'smiles' and 'mTP' columns
    structure_col = 'smiles'
    score_col = 'mTP'
    
    print(f"\nUsing columns:")
    print(f"  Structure: {structure_col}")
    print(f"  Score: {score_col}")
    
    # Extract and clean data
    df_clean = df[[structure_col, score_col]].dropna()
    df_clean.columns = ['Structure', 'Score']
    
    print(f"\nCleaned data: {len(df_clean)} samples")
    
    # Shuffle the data
    print(f"Shuffling data with random seed 42...")
    df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and test
    total_samples = len(df_shuffled)
    
    if test_size >= total_samples:
        print(f"\nWarning: Test size ({test_size}) >= total samples ({total_samples})")
        print(f"Setting test size to {total_samples // 2}")
        test_size = total_samples // 2
    
    train_size = total_samples - test_size
    
    print(f"\nSplitting data:")
    print(f"  Training set: {train_size} samples")
    print(f"  Test set: {test_size} samples")
    
    df_train = df_shuffled[:train_size]
    df_test = df_shuffled[train_size:]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    train_csv = os.path.join(output_dir, 'train_data.csv')
    test_csv = os.path.join(output_dir, 'test_data.csv')
    
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    print(f"\nSaved CSV files:")
    print(f"  {train_csv}")
    print(f"  {test_csv}")
    
    # Save as JSONL (for training)
    train_jsonl = os.path.join(output_dir, 'train_data.jsonl')
    test_jsonl = os.path.join(output_dir, 'test_data.jsonl')
    
    def create_jsonl(df, output_file):
        """Convert dataframe to JSONL format for Qwen training"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                structure = str(row['Structure'])
                score = row['Score']
                
                # Format as conversation
                conversation = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant specialized in drug discovery and molecular analysis. You can predict molecular scores based on their SMILES structures."
                        },
                        {
                            "role": "user",
                            "content": f"What is the predicted score for this molecular structure: {structure}?"
                        },
                        {
                            "role": "assistant",
                            "content": f"The predicted score for the molecular structure {structure} is {score}."
                        }
                    ]
                }
                
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    
    create_jsonl(df_train, train_jsonl)
    create_jsonl(df_test, test_jsonl)
    
    print(f"\nSaved JSONL files:")
    print(f"  {train_jsonl}")
    print(f"  {test_jsonl}")
    
    # Save statistics
    stats = {
        'total_samples': total_samples,
        'train_samples': train_size,
        'test_samples': test_size,
        'train_percentage': train_size / total_samples * 100,
        'test_percentage': test_size / total_samples * 100,
        'structure_column': structure_col,
        'score_column': score_col
    }
    
    stats_file = os.path.join(output_dir, 'data_split_info.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved statistics: {stats_file}")
    
    print(f"\n{'='*60}")
    print(f"DATA SPLIT COMPLETED")
    print(f"{'='*60}")
    print(f"Training set: {train_size} samples ({train_size/total_samples*100:.1f}%)")
    print(f"Test set: {test_size} samples ({test_size/total_samples*100:.1f}%)")
    print(f"All files saved to: {output_dir}/")
    print(f"{'='*60}")
    
    return df_train, df_test

if __name__ == "__main__":
    # Configuration
    input_file = "dataset/data_new.xlsx"
    output_dir = "data_An"
    test_size = 100  # Number of test samples
    
    # Run split
    df_train, df_test = split_data(input_file, output_dir, test_size)
    
    # Show sample from each set
    print(f"\nSample from training set:")
    print(df_train.head(3))
    
    print(f"\nSample from test set:")
    print(df_test.head(3))

