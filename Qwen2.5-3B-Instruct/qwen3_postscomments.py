# Stance Detection on Reddit AITA Posts using Qwen3 Model with Robust CSV Handling
import argparse
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

print("=" * 80)
print("Qwen3 CSV Processing with Error Handling")
print("=" * 80)

# ---------------------------
# Options for quick runs
# ---------------------------
parser = argparse.ArgumentParser(description="Run Qwen3 post/comment processing (with optional dry-run)")
parser.add_argument("--csv", type=str, default=r"D:\Quang Huy\Documents\EE6405\Project\EE6405_Final_Project\data\preprocessed\reddit_posts_and_comments.csv", help="Path to CSV file to process (default: combined preprocessed data)")
parser.add_argument("--nrows", type=int, default=5, help="Number of rows to read and process (default: 5). Use 0 to read all rows.")
parser.add_argument("--no-model", action="store_true", help="Do not load the Qwen model; useful for fast CSV parsing/dry-runs.")
args = parser.parse_args()

# Convert 0 to None for pandas
nrows = args.nrows if args.nrows > 0 else None

# ---------------------------
# Load your CSV with error handling
# ---------------------------
# Use CSV path from command-line argument or default to combined preprocessed data
csv_file_path = args.csv


print(f"\nLoading CSV from: {csv_file_path}")
print(f"Reading up to {nrows or 'ALL'} rows")
print("Attempting multiple parsing strategies...")

# Strategy 1: Try with different quoting and error handling
try:
    print("\nStrategy 1: Standard parsing with error_bad_lines=False")
    df = pd.read_csv(csv_file_path, on_bad_lines='skip', engine='python', encoding='utf-8', nrows=nrows)

    # Check that the expected columns exist in this dataset
    required_cols = ['post_text', 'comment_text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    # Optional: Keep ground truth if available
    ground_truth_available = 'label' in df.columns

    print(f"✓ Successfully loaded {len(df)} rows")
except Exception as e:
    print(f"✗ Strategy 1 failed: {e}")
    
    # Strategy 2: Try with different parameters
    try:
        print("\nStrategy 2: Parsing with engine='python'")
        df = pd.read_csv(
            csv_file_path,
            engine='python',
            on_bad_lines='skip',
            encoding='utf-8',
            quoting=1,
            nrows=nrows
        )
        print(f"✓ Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Strategy 2 failed: {e}")
        
        # Strategy 3: Try with explicit delimiter and escapechar
        try:
            print("\nStrategy 3: Parsing with explicit parameters")
            df = pd.read_csv(
                csv_file_path,
                sep=',',
                engine='python',
                on_bad_lines='skip',
                encoding='utf-8',
                escapechar='\\',
                quotechar='"',
                nrows=nrows
            )
            print(f"✓ Successfully loaded {len(df)} rows")
        except Exception as e:
            print(f"✗ Strategy 3 failed: {e}")
            print("\nError: Could not parse CSV file. Please check the file format.")
            exit(1)

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Display first few rows
print("\nFirst 3 rows:")
print(df.head(3))

# Check for expected columns
expected_columns = ['post_text', 'comment_text', 'stance']
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    print(f"\n⚠ Warning: Missing expected columns: {missing_columns}")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nPlease verify column names and update the script accordingly.")
    
    # Try to infer column names
    if len(df.columns) >= 3:
        print(f"\nAttempting to use first 3 columns as [post_text, comment_text, stance]")
        df.columns = ['post_text', 'comment_text', 'stance'] + list(df.columns[3:])

# Clean data
print("\nCleaning data...")
print(f"Rows before cleaning: {len(df)}")
df = df.dropna(subset=['post_text', 'comment_text', 'stance']).reset_index(drop=True)
print(f"Rows after cleaning: {len(df)}")

# Show stance distribution
print("\nStance distribution:")
print(df['stance'].value_counts())

# ---------------------------
# Load Qwen Model
# ---------------------------
print(f"\n{'='*80}")
print("Loading Qwen Model")
print(f"{'='*80}")

# Use Qwen/Qwen2.5-7B-Instruct or another Qwen variant
model_name = "Qwen/Qwen2.5-3B-Instruct"  # Adjust based on your model

if args.no_model:
    print("\n--no-model specified: skipping model download and load (dry-run mode)")
    llm_pipe = None
else:
    try:
        print(f"\nLoading model: {model_name}")
        print("This may take several minutes...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",        # Automatically assigns GPU/CPU
            torch_dtype=torch.float16, # Save memory
            trust_remote_code=True
        )

        print("✓ Model loaded successfully!")

        llm_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nContinuing in dry-run mode (predictions will be skipped).")
        llm_pipe = None

# ---------------------------
# Define prompt for label prediction & explanation
# ---------------------------
def generate_label_and_explanation(post, comment, llm_pipe=None):
    """Generate stance label and explanation for a post-comment pair.

    If `llm_pipe` is None, return a dry-run stub so script can run quickly.
    """
    # Truncate long texts
    post_truncated = post[:500] if len(str(post)) > 500 else post
    comment_truncated = comment[:300] if len(str(comment)) > 300 else comment

    prompt = f"""You are an NLP assistant analyzing Reddit AITA (Am I The Asshole) posts.

Given a Reddit post and a comment, predict the stance of the comment and explain why.

Post: "{post_truncated}"
Comment: "{comment_truncated}"

Output format:
Label: <concur/oppose/neutral>
Explanation: <brief explanation>
"""

    if llm_pipe is None:
        # Dry-run mode: return placeholder predictions without calling the model
        return "skipped", "dry-run: model not loaded"

    try:
        output = llm_pipe(prompt, max_new_tokens=150)[0]['generated_text']

        # Extract Label and Explanation using regex
        label_match = re.search(r"Label:\s*(concur|oppose|neutral)", output, re.IGNORECASE)
        explanation_match = re.search(r"Explanation:\s*(.*?)(?:\n|$)", output, re.IGNORECASE)

        label = label_match.group(1).lower() if label_match else None
        explanation = explanation_match.group(1).strip() if explanation_match else None

        return label, explanation
    except Exception as e:
        print(f"Error generating prediction: {e}")
        return None, None
    
# ---------------------------
# Run predictions on sample data
# ---------------------------
print(f"\n{'='*80}")
print("Running Predictions on Sample Data")
print(f"{'='*80}")

# Process all rows loaded (usually limited by --nrows). The default is 5 rows.
sample_size = len(df)
print(f"\nProcessing {sample_size} samples...")

results = []
for i, row in df.head(sample_size).iterrows():
    print(f"\nProcessing row {i+1}/{sample_size}...")
    label, explanation = generate_label_and_explanation(
        row['post_text'],
        row['comment_text'],
        llm_pipe=llm_pipe
    )
    results.append({
        "post": row['post_text'][:100] + "...",
        "comment": row['comment_text'][:100] + "...",
        "true_label": row['stance'],
        "predicted_label": label,
        "explanation": explanation
    })

results_df = pd.DataFrame(results)

# Display results
print(f"\n{'='*80}")
print("Results:")
print(f"{'='*80}\n")
print(results_df)

# Save results with name adapted from input CSV
import os
csv_basename = os.path.basename(csv_file_path)
csv_name_no_ext = os.path.splitext(csv_basename)[0]
output_path = f"{csv_name_no_ext}_predictions.csv"
results_df.to_csv(output_path, index=False)
print(f"\n✓ Results saved to: {output_path}")
print(f"  (Based on input: {csv_basename})")

print(f"\n{'='*80}")
print("Processing Complete!")
print(f"{'='*80}")

