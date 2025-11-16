import pandas as pd
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "preprocessed"
OUTPUT_DIR = ROOT / "data" / "preprocessed"

# Remove the stance acronyms from the comment text
def remove_acronyms(text):
    if pd.isna(text):
        return text
    # Remove the specific acronyms while keeping the rest of the text
    cleaned = re.sub(r"\b(YTA|NTA|ESH|INFO|NAH)\b", "", str(text))
    # Clean up any extra whitespace created by removal
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# Process both train and test files
for filename in ["redditAITA_train.csv", "redditAITA_test.csv"]:
    print(f"\nProcessing {filename}...")
    
    # Load the data
    df = pd.read_csv(INPUT_DIR / filename)
    print(f"Original shape: {df.shape}")
    
    # Show sample before cleaning
    print(f"\nSample comment before cleaning:")
    if 'comment_text' in df.columns and len(df) > 0:
        print(df['comment_text'].iloc[0][:200])
    

    if 'comment_text' in df.columns:
        df['comment_text'] = df['comment_text'].apply(remove_acronyms)
        

        print(f"\nSample comment after cleaning:")
        print(df['comment_text'].iloc[0][:200])
    else:
        print("Warning: 'comment_text' column not found in the dataframe")
        print(f"Available columns: {df.columns.tolist()}")
    
    # Save as new files
    output_filename = filename.replace(".csv", "_cleaned.csv")
    output_path = OUTPUT_DIR / output_filename
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")
    print(f"Final shape: {df.shape}")

print("\n✓ All files processed successfully!")
