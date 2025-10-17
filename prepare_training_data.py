import pandas as pd
import json
import os

def convert_smiles_to_training_format(input_file, output_file):
    """
    Convert SMILES data from Excel to JSONL format for Qwen 7B finetuning.
    
    Args:
        input_file: Path to the input Excel file
        output_file: Path to the output JSONL file
    """
    # Read the Excel file - skip first empty row, use row 1 as header
    df = pd.read_excel(input_file, header=0)
    
    # Check if required columns exist
    if 'Structure' not in df.columns or 'Score' not in df.columns:
        print("Error: 'Structure' and 'Score' columns must exist in the Excel file")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Filter out rows with missing values
    df_clean = df[['Structure', 'Score']].dropna()
    
    print(f"Total samples: {len(df_clean)}")
    print(f"Sample data:\n{df_clean.head()}")
    
    # Convert to training format (JSONL)
    training_data = []
    for idx, row in df_clean.iterrows():
        structure = str(row['Structure'])
        score = row['Score']
        
        # Format as instruction-following conversation for Qwen
        # You can customize the prompt template based on your needs
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
        
        training_data.append(conversation)
    
    # Save to JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nTraining data saved to: {output_file}")
    print(f"Total training examples: {len(training_data)}")
    
    # Also create a simple CSV format as backup
    csv_output = output_file.replace('.jsonl', '.csv')
    df_clean.to_csv(csv_output, index=False)
    print(f"CSV backup saved to: {csv_output}")
    
    return training_data

if __name__ == "__main__":
    input_file = "dataset/smiles-data.xlsx"
    output_file = "dataset/train_data.jsonl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert the data
    convert_smiles_to_training_format(input_file, output_file)

