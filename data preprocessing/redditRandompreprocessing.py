import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split


# Change according to your paths
file_path = "/Users/hehvince/Desktop/EE6405/EE6405_Final_Project/data/raw/reddit_posts_and_comments_labeled.csv"

# Change according to your paths
preprocessed_file_path = "/Users/hehvince/Desktop/EE6405/EE6405_Final_Project/data/preprocessed/"

# 1) Read the CSV file -> DataFrame
data_df = pd.read_csv(file_path)

# 2) Inspect
print(data_df.shape)
print(data_df.columns.tolist())
print(data_df.head())

# Normalize column names (handles accidental leading/trailing spaces)
data_df.columns = data_df.columns.str.strip()

data_df = data_df.drop(columns=["post_title","subreddit","post_author","post_url","post_upvotes","post_downvotes","comment_upvotes","comment_downvotes","comment_author","model_confidence"])

# Remove rows where 'comment_text' is missing or only whitespace
col = "comment_text"
if col not in data_df.columns:
    raise KeyError(f"Column '{col}' not found. Available columns: {data_df.columns.tolist()}")

data_df = data_df[data_df[col].notna() & data_df[col].astype(str).str.strip().ne("")].copy()


# Split train and test sets (stratified)
train_df, test_df = train_test_split(
    data_df,
    test_size=0.2,
    random_state=42,
    stratify=data_df["stance"]
    )


# Save to CSV
train_df.to_csv(f"{preprocessed_file_path}reddit_posts_and_comments_train.csv", index=False)
test_df.to_csv(f"{preprocessed_file_path}reddit_posts_and_comments_test.csv", index=False)
data_df.to_csv(f"{preprocessed_file_path}reddit_posts_and_comments.csv", index=False)
