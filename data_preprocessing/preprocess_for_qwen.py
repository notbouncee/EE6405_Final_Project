"""
Unified preprocessing script for Reddit and Stance datasets.
Loads raw data from two sources and outputs a unified format for Qwen model.
"""

import pandas as pd
import re
import os
from pathlib import Path

print("=" * 80)
print("Preprocessing Raw Datasets for Qwen Model")
print("=" * 80)

# Paths to raw datasets
reddit_raw_path = r"D:\Quang Huy\Documents\EE6405\Project\EE6405_Final_Project\data\raw\reddit_posts_and_comments.csv"
stance_raw_path = r"D:\Quang Huy\Documents\EE6405\Project\EE6405_Final_Project\data\raw\stance_dataset.csv"

# Output paths
output_dir = r"D:\Quang Huy\Documents\EE6405\Project\EE6405_Final_Project\data\preprocessed"
reddit_output_path = os.path.join(output_dir, "reddit_preprocessed_for_qwen.csv")
reddit_json_path = os.path.join(output_dir, "reddit_preprocessed_for_qwen.json")
stance_output_path = os.path.join(output_dir, "stance_preprocessed_for_qwen.csv")
stance_json_path = os.path.join(output_dir, "stance_preprocessed_for_qwen.json")
combined_output_path = os.path.join(output_dir, "combined_preprocessed_for_qwen.csv")
combined_json_path = os.path.join(output_dir, "combined_preprocessed_for_qwen.json")

# Ensure output directory exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. PREPROCESS REDDIT DATASET
# ============================================================================
print("\n" + "=" * 80)
print("Processing Reddit Posts & Comments Dataset")
print("=" * 80)

try:
    print(f"\nLoading: {reddit_raw_path}")
    reddit_df = pd.read_csv(reddit_raw_path, on_bad_lines='skip', engine='python', encoding='utf-8')
    print(f"✓ Loaded {len(reddit_df)} rows")
    print(f"  Columns: {reddit_df.columns.tolist()}")
    
    # Extract and clean the text fields
    print("\nCleaning text fields...")
    
    # Remove extra whitespace and newlines
    reddit_df['post_text_clean'] = reddit_df['post_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    reddit_df['comment_text_clean'] = reddit_df['comment_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Remove URLs
    reddit_df['post_text_clean'] = reddit_df['post_text_clean'].str.replace(r'http\S+', '', regex=True)
    reddit_df['comment_text_clean'] = reddit_df['comment_text_clean'].str.replace(r'http\S+', '', regex=True)
    
    # Remove special characters but keep basic punctuation
    reddit_df['post_text_clean'] = reddit_df['post_text_clean'].str.replace(r'[^a-zA-Z0-9\s.,!?]', '', regex=True)
    reddit_df['comment_text_clean'] = reddit_df['comment_text_clean'].str.replace(r'[^a-zA-Z0-9\s.,!?]', '', regex=True)
    
    # Map stance labels to consistent format (neutral, concur, oppose)
    def map_stance(label):
        if pd.isna(label):
            return 'neutral'
        label = str(label).lower().strip()
        if label in ['concur', 'support', 'agree']:
            return 'concur'
        elif label in ['oppose', 'deny', 'disagree']:
            return 'oppose'
        else:
            return 'neutral'
    
    reddit_df['stance_clean'] = reddit_df['stance'].apply(map_stance)
    
    # Create unified format for Qwen: post_text, comment_text, stance
    reddit_unified = pd.DataFrame({
        'post_text': reddit_df['post_text_clean'],
        'comment_text': reddit_df['comment_text_clean'],
        'stance': reddit_df['stance_clean'],
        'source': 'reddit'
    })
    
    # Remove duplicates and empty rows
    reddit_unified = reddit_unified[
        (reddit_unified['post_text'].str.len() > 0) & 
        (reddit_unified['comment_text'].str.len() > 0)
    ].drop_duplicates(subset=['post_text', 'comment_text'])
    
    print(f"\nAfter cleaning:")
    print(f"  ✓ Rows: {len(reddit_unified)}")
    print(f"  Stance distribution:\n{reddit_unified['stance'].value_counts()}")
    
    # Save Reddit preprocessed
    reddit_unified.to_csv(reddit_output_path, index=False)
    reddit_unified.to_json(reddit_json_path, orient='records', indent=2)
    print(f"\n✓ Saved CSV: {reddit_output_path}")
    print(f"✓ Saved JSON: {reddit_json_path}")

except Exception as e:
    print(f"✗ Error processing Reddit data: {e}")
    reddit_unified = pd.DataFrame()


# ============================================================================
# 2. PREPROCESS STANCE DATASET
# ============================================================================
print("\n" + "=" * 80)
print("Processing Stance Dataset")
print("=" * 80)

try:
    print(f"\nLoading: {stance_raw_path}")
    stance_df = pd.read_csv(stance_raw_path, on_bad_lines='skip', engine='python', encoding='utf-8')
    print(f"✓ Loaded {len(stance_df)} rows")
    print(f"  Columns: {stance_df.columns.tolist()}")
    
    # Extract and clean the text fields
    print("\nCleaning text fields...")
    
    # Remove extra whitespace and newlines
    stance_df['target_text_clean'] = stance_df['target_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    stance_df['response_text_clean'] = stance_df['response_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Remove URLs
    stance_df['target_text_clean'] = stance_df['target_text_clean'].str.replace(r'http\S+', '', regex=True)
    stance_df['response_text_clean'] = stance_df['response_text_clean'].str.replace(r'http\S+', '', regex=True)
    
    # Remove special characters but keep basic punctuation
    stance_df['target_text_clean'] = stance_df['target_text_clean'].str.replace(r'[^a-zA-Z0-9\s.,!?]', '', regex=True)
    stance_df['response_text_clean'] = stance_df['response_text_clean'].str.replace(r'[^a-zA-Z0-9\s.,!?]', '', regex=True)
    
    # Map stance labels from this dataset to consistent format
    def map_stance_dataset(label):
        if pd.isna(label):
            return 'neutral'
        label = str(label).lower().strip()
        if label in ['implicit_support', 'explicit_support', 'support', 'agree']:
            return 'concur'
        elif label in ['implicit_denial', 'explicit_denial', 'deny', 'disagree', 'oppose']:
            return 'oppose'
        else:
            return 'neutral'
    
    stance_df['stance_clean'] = stance_df['label'].apply(map_stance_dataset)
    
    # Create unified format: post_text (target), comment_text (response), stance
    stance_unified = pd.DataFrame({
        'post_text': stance_df['target_text_clean'],
        'comment_text': stance_df['response_text_clean'],
        'stance': stance_df['stance_clean'],
        'source': 'stance_dataset'
    })
    
    # Remove duplicates and empty rows
    stance_unified = stance_unified[
        (stance_unified['post_text'].str.len() > 0) & 
        (stance_unified['comment_text'].str.len() > 0)
    ].drop_duplicates(subset=['post_text', 'comment_text'])
    
    print(f"\nAfter cleaning:")
    print(f"  ✓ Rows: {len(stance_unified)}")
    print(f"  Stance distribution:\n{stance_unified['stance'].value_counts()}")
    
    # Save Stance preprocessed
    stance_unified.to_csv(stance_output_path, index=False)
    stance_unified.to_json(stance_json_path, orient='records', indent=2)
    print(f"\n✓ Saved CSV: {stance_output_path}")
    print(f"✓ Saved JSON: {stance_json_path}")

except Exception as e:
    print(f"✗ Error processing Stance data: {e}")
    stance_unified = pd.DataFrame()


# ============================================================================
# 3. COMBINE BOTH DATASETS
# ============================================================================
print("\n" + "=" * 80)
print("Combining Both Datasets")
print("=" * 80)

try:
    if len(reddit_unified) > 0 and len(stance_unified) > 0:
        combined = pd.concat([reddit_unified, stance_unified], ignore_index=True)
        
        # Remove any duplicates across datasets
        combined = combined.drop_duplicates(subset=['post_text', 'comment_text'])
        
        print(f"\n✓ Combined dataset:")
        print(f"  Total rows: {len(combined)}")
        print(f"  Stance distribution:\n{combined['stance'].value_counts()}")
        print(f"  Source distribution:\n{combined['source'].value_counts()}")
        
        # Save combined
        combined.to_csv(combined_output_path, index=False)
        combined.to_json(combined_json_path, orient='records', indent=2)
        print(f"\n✓ Saved CSV: {combined_output_path}")
        print(f"✓ Saved JSON: {combined_json_path}")
    else:
        print("✗ Cannot combine: one or both datasets are empty")

except Exception as e:
    print(f"✗ Error combining datasets: {e}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("Preprocessing Complete!")
print("=" * 80)
print("\nOutput files created:")
print(f"  1. Reddit preprocessed CSV:    {reddit_output_path}")
print(f"  2. Reddit preprocessed JSON:   {reddit_json_path}")
print(f"  3. Stance preprocessed CSV:    {stance_output_path}")
print(f"  4. Stance preprocessed JSON:   {stance_json_path}")
print(f"  5. Combined CSV:               {combined_output_path}")
print(f"  6. Combined JSON:              {combined_json_path}")
print("\nAll files are ready for the Qwen model with columns:")
print("  - post_text:  Main post/target text")
print("  - comment_text: Response/comment text")
print("  - stance:     Label (concur, oppose, neutral)")
print("  - source:     Data source (reddit or stance_dataset)")
print("=" * 80)
