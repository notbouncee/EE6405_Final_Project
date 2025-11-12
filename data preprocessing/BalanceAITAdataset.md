# Balance redditAITA Dataset

## Step 1: Run Preprocessing

```bash
python "data preprocessing/redditAITApreprocessing.py"
```

Generates files in `./data/preprocessed`:
- `redditAITA_train.csv`
- `redditAITA_test.csv`
- `redditAITA.csv`

## Step 2: Balance Train & Reduce Test Set

```bash
python "data preprocessing/balance_dataset.py"
```

**Training set** - Balanced:
- Concur: 10,000 samples
- Oppose: 10,000 samples
- Neutral: 10,000 samples

**Test set** - Reduced 20x (maintains original distribution):
- Concur: ~26,455 (68.2%)
- Oppose: ~11,774 (30.3%)
- Neutral: ~566 (1.5%)

## Result

A new `redditAITA_train.csv` and  `redditAITA_test.csv`  will be created under `./data/preprocessed`, overwriting the original dataset.

