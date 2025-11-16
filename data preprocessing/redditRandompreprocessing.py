import numpy as np
import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split


file_path = Path(r"c:\Users\Vince\OneDrive\Desktop\Y4S1\EE6405_Final_Project\data\raw\reddit_posts_and_comments_labeled.csv")

preprocessed_dir = Path(r"c:\Users\Vince\OneDrive\Desktop\Y4S1\EE6405_Final_Project\data\preprocessed")

preprocessed_dir.mkdir(parents=True, exist_ok=True)


# 1) Read the CSV file into DataFrame
data_df = pd.read_csv(file_path)

# 2) Inspect data
print(data_df.shape)

print(data_df.columns.tolist())

print(data_df.head())


# Normalise column names
data_df.columns = data_df.columns.str.strip()


data_df = data_df.drop(columns=["post_title","subreddit","post_author","post_url","post_upvotes","post_downvotes","comment_upvotes","comment_downvotes","comment_author","model_confidence"])


# Remove rows where 'comment_text' is missing or only whitespace
col = "comment_text"

if col not in data_df.columns:

    raise KeyError(f"Column '{col}' not found. Available columns: {data_df.columns.tolist()}")

data_df = data_df[data_df[col].notna() & data_df[col].astype(str).str.strip().ne("")].copy()


# Split train and test sets
train_df, test_df = train_test_split(
    data_df,
    test_size=0.2,
    random_state=42,
    stratify=data_df["stance"]
    )


# Save to CSV
train_df.to_csv(preprocessed_dir / "reddit_posts_and_comments_train.csv", index=False)

test_df.to_csv(preprocessed_dir / "reddit_posts_and_comments_test.csv", index=False)

data_df.to_csv(preprocessed_dir / "reddit_posts_and_comments.csv", index=False)