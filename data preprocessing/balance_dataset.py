"""
Balance redditAITA Dataset for Stance Detection

This script balances the imbalanced redditAITA dataset by:
- Sampling 100,000 concur samples (from 2.1M)
- Sampling 100,000 oppose samples (from 941k)
- Keeping all 45,276 neutral samples
"""

import pandas as pd
import argparse
from pathlib import Path

def balance_redditaita(data_dir="./data/preprocessed",
                       concur_samples=100000,
                       oppose_samples=100000,
                       random_seed=42):
    """
    Balance the redditAITA training dataset and overwrite original
    """
    
    print("\n" + "="*60)
    print("BALANCING redditAITA DATASET")
    print("="*60)
    
    # File paths
    data_path = Path(data_dir)
    train_file = data_path / "redditAITA_train.csv"
    
    # Check if file exists
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    # Load training data
    print(f"\nLoading: {train_file}")
    df_train = pd.read_csv(train_file)
    
    print(f"Original size: {len(df_train):,} samples")
    
    # Show original distribution
    print("\nOriginal distribution:")
    for stance in ['concur', 'oppose', 'neutral']:
        count = len(df_train[df_train['stance'] == stance])
        pct = (count / len(df_train)) * 100
        print(f"  {stance}: {count:,} ({pct:.1f}%)")
    
    # Separate by stance
    df_concur = df_train[df_train['stance'] == 'concur']
    df_oppose = df_train[df_train['stance'] == 'oppose']
    df_neutral = df_train[df_train['stance'] == 'neutral']
    
    print(f"\nBalancing with seed={random_seed}...")
    
    # Sample concur and oppose, keep all neutral
    df_concur_sampled = df_concur.sample(n=concur_samples, random_state=random_seed)
    df_oppose_sampled = df_oppose.sample(n=oppose_samples, random_state=random_seed)
    df_neutral_sampled = df_neutral 
    
    print(f"  Sampled {len(df_concur_sampled):,} concur")
    print(f"  Sampled {len(df_oppose_sampled):,} oppose")
    print(f"  Kept all {len(df_neutral_sampled):,} neutral")
    
    # Combine and shuffle
    df_balanced = pd.concat([df_concur_sampled, df_oppose_sampled, df_neutral_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Show balanced distribution
    print(f"\nBalanced size: {len(df_balanced):,} samples")
    print("\nBalanced distribution:")
    for stance in ['concur', 'oppose', 'neutral']:
        count = len(df_balanced[df_balanced['stance'] == stance])
        pct = (count / len(df_balanced)) * 100
        print(f"  {stance}: {count:,} ({pct:.1f}%)")
    
    # Overwrite original training file
    print(f"\nOverwriting original file: {train_file}")
    df_balanced.to_csv(train_file, index=False)
    print(f"✓ Balanced training data saved (original backed up)")

    
    print("\n" + "="*60)
    print("BALANCING COMPLETE!")
    print("="*60)
    print(f"✓ Training file updated: {train_file}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Balance redditAITA dataset and overwrite original training file'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./data/preprocessed',
        help='Directory containing original data (default: ./data/preprocessed)'
    )
    parser.add_argument(
        '--concur_samples', 
        type=int, 
        default=100000,
        help='Number of concur samples to keep (default: 100000)'
    )
    parser.add_argument(
        '--oppose_samples', 
        type=int, 
        default=100000,
        help='Number of oppose samples to keep (default: 100000)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    balance_redditaita(
        data_dir=args.data_dir,
        concur_samples=args.concur_samples,
        oppose_samples=args.oppose_samples,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()