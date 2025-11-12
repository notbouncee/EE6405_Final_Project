"""
Balance redditAITA Dataset for Stance Detection

This script balances the imbalanced redditAITA train dataset by:
- Sampling 46,000 concur samples (from 2.1M)
- Sampling 46,000 oppose samples (from 941k)
- Keeping all 45,276 neutral samples

It also reduces the test set size by a factor of 10 while maintaining the original distribution.
"""

import pandas as pd
import argparse
from pathlib import Path

def balance_train_set(data_dir="./data/preprocessed",
                       concur_samples=46000,
                       oppose_samples=46000,
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

    print("\n" + "="*60)
    print("BALANCING COMPLETE!")
    print("="*60)
    print(f"✓ Training file updated: {train_file}")
    print("="*60 + "\n")


def reduce_test_size(data_dir="./data/preprocessed",
                     reduction_factor=10,
                     random_seed=42):
    """
    Reduce test set size while maintaining original distribution
    """
    
    print("\n" + "="*60)
    print("REDUCING TEST SET SIZE")
    print("="*60)
    
    # File paths
    data_path = Path(data_dir)
    test_file = data_path / "redditAITA_test.csv"
    
    # Check if file exists
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Load test data
    print(f"\nLoading: {test_file}")
    df_test = pd.read_csv(test_file)
    
    print(f"Original size: {len(df_test):,} samples")
    
    # Show original distribution
    print("\nOriginal distribution:")
    for stance in ['concur', 'oppose', 'neutral']:
        count = len(df_test[df_test['stance'] == stance])
        pct = (count / len(df_test)) * 100
        print(f"  {stance}: {count:,} ({pct:.1f}%)")
    
    # Reduce each stance by the same factor
    print(f"\nReducing size by {reduction_factor}x with seed={random_seed}...")
    
    dfs_reduced = []
    for stance in ['concur', 'oppose', 'neutral']:
        df_stance = df_test[df_test['stance'] == stance]
        n_samples = len(df_stance) // reduction_factor
        df_sampled = df_stance.sample(n=n_samples, random_state=random_seed)
        dfs_reduced.append(df_sampled)
        print(f"  {stance}: {len(df_stance):,} → {n_samples:,}")
    
    # Combine and shuffle
    df_reduced = pd.concat(dfs_reduced)
    df_reduced = df_reduced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Show reduced distribution
    print(f"\nReduced size: {len(df_reduced):,} samples")
    print("\nReduced distribution:")
    for stance in ['concur', 'oppose', 'neutral']:
        count = len(df_reduced[df_reduced['stance'] == stance])
        pct = (count / len(df_reduced)) * 100
        print(f"  {stance}: {count:,} ({pct:.1f}%)")
    
    # Overwrite original test file
    print(f"\nOverwriting original file: {test_file}")
    df_reduced.to_csv(test_file, index=False)
    
    print("\n" + "="*60)
    print("TEST SET REDUCTION COMPLETE!")
    print("="*60)
    print(f"Test file updated: {test_file}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Balance redditAITA training set and reduce test set size'
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
        default=46000,
        help='Number of concur samples to keep (default: 46000)'
    )
    parser.add_argument(
        '--oppose_samples', 
        type=int, 
        default=46000,
        help='Number of oppose samples to keep (default: 46000)'
    )
    parser.add_argument(
        '--reduction_factor',
        type=int,
        default=10,
        help='Factor to reduce test set size by (default: 10)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--skip_train',
        action='store_true',
        help='Skip training set balancing (only reduce test set)'
    )
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='Skip test set reduction (only balance train set)'
    )
    
    args = parser.parse_args()
    
    # Balance training set
    if not args.skip_train:
        balance_train_set(
            data_dir=args.data_dir,
            concur_samples=args.concur_samples,
            oppose_samples=args.oppose_samples,
            random_seed=args.seed
        )
    
    # Reduce test set size
    if not args.skip_test:
        reduce_test_size(
            data_dir=args.data_dir,
            reduction_factor=args.reduction_factor,
            random_seed=args.seed
        )


if __name__ == "__main__":
    main()