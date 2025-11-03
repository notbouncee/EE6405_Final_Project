import numpy as np 
import pandas as pd
import sqlite3
import re
from sklearn.model_selection import train_test_split

# Extract dataset
conn = sqlite3.connect("/Users/hehvince/Desktop/EE6405/EE6405_Final_Project/data/raw/AmItheAsshole.sqlite")

# Change according to your paths
preprocessed_file_path = "/Users/hehvince/Desktop/EE6405/EE6405_Final_Project/data/preprocessed/"

# Show tables
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(query, conn)
print("Tablas:")
print(tables)

# Load some comments
comments = pd.read_sql("SELECT * FROM comment LIMIT 10", conn)
comments.head()

# View actual column names in 'submission'
submission_sample = pd.read_sql("SELECT * FROM submission LIMIT 1", conn)
print(submission_sample.columns)

# Search for all comments that contain judgment
judging_comments = pd.read_sql("""
SELECT submission_id, message, score
FROM comment
WHERE message LIKE '%YTA%' OR message LIKE '%NTA%' OR message LIKE '%ESH%' OR message LIKE '%INFO%' OR message LIKE '%NAH%'
""", conn)

# Choose the highest-scoring comment for each post (corrected with .copy())
judging_comments = judging_comments.sort_values("score", ascending=False)
top_comments = judging_comments.drop_duplicates(subset="submission_id", keep="first").copy()
top_comments.head()

#  Extract the comment tag
def extract_label(text):
    match = re.search(r"\b(YTA|NTA|ESH|INFO|NAH)\b", text)
    return match.group(1) if match else None

top_comments["label"] = top_comments["message"].apply(extract_label)

# Upload the original posts, including submission_id
submissions = pd.read_sql("SELECT submission_id, title, selftext FROM submission", conn)

# Join using submission_id (both tables have the same one)
merged = top_comments.merge(submissions, on="submission_id")

# Combine title + post body
merged["text"] = merged["title"].fillna('') + " " + merged["selftext"].fillna('')

# Filter necessary columns
data_df = merged[["text", "label"]].rename(columns={"label": "stance"})

data_df.head()

# Delete rows with no label
data_df = data_df[data_df["stance"].notna()].copy()

# Relabel 'label' column
mapping = {'YTA': 'oppose', 'ESH': 'oppose', 'INFO': 'neutral', 'NAH': 'concur', 'NTA': 'concur'}
if not set(data_df['stance'].astype(str).unique()) <= set(mapping.keys()):
    raise ValueError("label column has values outside {'YTA','ESH','INFO','NAH','NTA'}")
data_df['stance'] = data_df['stance'].astype(str).replace(mapping)

# Split train and test sets (stratified)
train_df, test_df = train_test_split(
    data_df,
    test_size=0.2,
    random_state=42,
    stratify=data_df["stance"]
    )



# Save to CSV
train_df.to_csv(f"{preprocessed_file_path}redditAITA_train.csv", index=False)
test_df.to_csv(f"{preprocessed_file_path}redditAITA_test.csv", index=False)
data_df.to_csv(f"{preprocessed_file_path}redditAITA.csv", index=False)

