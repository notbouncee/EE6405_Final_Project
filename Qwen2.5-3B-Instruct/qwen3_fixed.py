# Stance Detection on Reddit AITA Posts using Qwen3 Model with Robust CSV Handling
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

print("=" * 80)
print("Qwen3 CSV Processing with Error Handling")
print("=" * 80)

# ---------------------------
# Load your CSV with error handling
# ---------------------------
# csv_file_path = "reddit_posts_and_comments.csv"
csv_file_path = "/opt/tiger/MLLM_AUTO_EVALUATE_PIPELINE/EE6405_Final_Project/data/preprocessed/redditAITA_test.csv"


print(f"\nLoading CSV from: {csv_file_path}")
print("Attempting multiple parsing strategies...")

# Strategy 1: Try with different quoting and error handling
try:
    print("\nStrategy 1: Standard parsing with error_bad_lines=False")
    df = pd.read_csv(
        csv_file_path,
        on_bad_lines='skip',  # Skip bad lines (pandas >= 1.3)
        encoding='utf-8',
        quoting=1  # QUOTE_ALL
    )
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
            quoting=1
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
                quotechar='"'
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
    
    # ---------------------------
    # Define prompt for label prediction & explanation
    # ---------------------------
    def generate_label_and_explanation(post, comment):
        """Generate stance label and explanation for a post-comment pair."""
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
    
    sample_size = min(5, len(df))
    print(f"\nProcessing {sample_size} samples...")
    
    results = []
    for i, row in df.head(sample_size).iterrows():
        print(f"\nProcessing row {i+1}/{sample_size}...")
        label, explanation = generate_label_and_explanation(
            row['post_text'], 
            row['comment_text']
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
    
    # Save results
    output_path = "reddit_qwen3_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
except Exception as e:
    print(f"\n✗ Error loading model: {e}")
    print("\nPossible solutions:")
    print("  1. Verify model name is correct")
    print("  2. Ensure sufficient GPU memory")
    print("  3. Check Hugging Face authentication")
    print("  4. Install required packages: pip install transformers torch accelerate")

print(f"\n{'='*80}")
print("Processing Complete!")
print(f"{'='*80}")

