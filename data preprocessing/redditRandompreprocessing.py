import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split


# Change according to your paths
file_path = "/Users/hehvince/Desktop/EE6405/EE6405_Final_Project/data/raw/reddit_posts_and_comments.csv"

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

data_df = data_df.drop(columns=["Times_Labeled","response_created_at","target_created_at","response_id","target_id","interaction_type"])

# Remove rows where 'comment_text' is missing or only whitespace
col = "comment_text"
if col not in data_df.columns:
    raise KeyError(f"Column '{col}' not found. Available columns: {data_df.columns.tolist()}")

# Option A (clear, two-step):
data_df = data_df.dropna(subset=[col]).copy()
data_df[col] = data_df[col].astype(str).str.strip()
data_df = data_df[data_df[col] != ""].copy()

# Option B (single expression):
# data_df = data_df[data_df[col].notna() & data_df[col].astype(str).str.strip().ne("")].copy()


