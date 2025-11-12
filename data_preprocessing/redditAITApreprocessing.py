import numpy as np 
import pandas as pd
import sqlite3
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project-root relative paths (place the DB at repo-root/data/raw/AmItheAsshole.sqlite)
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "raw" / "AmItheAsshole.sqlite"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# open sqlite connection
conn = sqlite3.connect(str(DB_PATH))

# Show tables
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(query, conn)
print("Tables:")
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

#  Extract the comment tag
def extract_label(text):
    match = re.search(r"\b(YTA|NTA|ESH|INFO|NAH)\b", text)
    return match.group(1) if match else None

judging_comments["label"] = judging_comments["message"].apply(extract_label)

# Upload the original posts, including submission_id
submissions = pd.read_sql("SELECT submission_id, title, selftext FROM submission", conn)

# keep all submissions, attach any matching comment rows (may produce multiple rows per submission)
merged = pd.merge(submissions,
                  judging_comments,
                  on="submission_id",
                  how="left")


print(merged.shape)
merged.head()

# Drop unnecessary columns
merged.drop(columns=["submission_id", "title", "score"], inplace=True)

# Rename columns
data_df = merged.rename(columns={"message": "comment", "label": "stance", "selftext": "post"})

# Drop rows where 'stance' is null or only whitespace
data_df = data_df[data_df['stance'].notna() & data_df['stance'].astype(str).str.strip().ne('')].copy()

data_df.head()

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


# Save to CSV (write into project-relative data/preprocessed directory)
train_df.to_csv(PREPROCESSED_DIR / "redditAITA_train.csv", index=False)
test_df.to_csv(PREPROCESSED_DIR / "redditAITA_test.csv", index=False)
data_df.to_csv(PREPROCESSED_DIR / "redditAITA.csv", index=False)

