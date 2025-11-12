import numpy as np
import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "raw" / "stance_dataset.json"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# 1) Read the JSONL file -> DataFrame
data_df = pd.read_json(DB_PATH, lines=True)

# 2) Parse any *_created_at columns as datetimes
date_cols = [c for c in data_df.columns if c.lower().endswith("created_at")]
for col in date_cols:
    data_df[col] = pd.to_datetime(data_df[col], errors="coerce", utc=True)

# 3) Inspect
print(data_df.shape)
print(data_df.columns.tolist())
print(data_df.head())

# 4) Relabel 'label' column
labels = {"Implicit_Support":"concur", "Explicit_Support":"concur","Implicit_Denial":"oppose","Explicit_Denial":"oppose","Comment":"neutral"}


def relabel_values(df: pd.DataFrame,
                   col: str,
                   mapping,
                   *,
                   inplace: bool = True,
                   case_sensitive: bool = True) -> pd.Series | None:
    """
    Relabel values in df[col].

    Parameters
    ----------
    df : DataFrame
    col : str
        Column name to relabel.
    mapping : dict or callable
        - dict: {old_value: new_value}. Unspecified values stay unchanged.
        - callable: function(old_value) -> new_value_or_None; if None, keep original.
    inplace : bool, default True
        If True, write changes into df[col] and return None; otherwise return a Series.
    case_sensitive : bool, default True
        Relevant only when mapping is a dict of string keys. If False, match ignoring case.

    Returns
    -------
    Series | None
        New column if inplace=False, else None.
    """
    s = df[col]

    # Case 1: dict mapping (exact replacements)
    if isinstance(mapping, dict):
        if case_sensitive:
            new_s = s.replace(mapping)
        else:
            # case-insensitive: normalize keys and the series to lower for matching
            s_as_str = s.astype(str)
            lower_map = {str(k).lower(): v for k, v in mapping.items()}
            # Map lowercased values, keep originals when not matched
            mapped = s_as_str.str.lower().map(lower_map)
            # If a value didn't map, fall back to original
            # Preserve original dtype if possible
            new_s = pd.Series(pd.NA, index=s.index, dtype="object")
            new_s[:] = s_as_str
            new_s = mapped.where(mapped.notna(), new_s)
    else:
        # Case 2: callable mapping
        # Apply; keep original on None
        mapped = s.map(mapping)
        new_s = mapped.where(mapped.notna(), s)

    if inplace:
        df[col] = new_s
        return None
    else:
        return new_s
    

relabel_values(data_df, "label", labels)    

data_df = data_df.drop(columns=["Times_Labeled","response_created_at","target_created_at","response_id","target_id","interaction_type"])

train_df, test_df = train_test_split(
    data_df,
    test_size=0.2,
    random_state=42,
    stratify=data_df["label"]
    )

#4 Save to CSV

train_df.to_csv(PREPROCESSED_DIR / "stance_dataset_train.csv", index=False)
test_df.to_csv(PREPROCESSED_DIR / "stance_dataset_test.csv", index=False)
data_df.to_csv(PREPROCESSED_DIR / "stance_dataset.csv", index=False)
