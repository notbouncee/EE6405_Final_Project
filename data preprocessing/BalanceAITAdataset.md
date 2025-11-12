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
- Concur: 50,000 samples
- Oppose: 50,000 samples
- Neutral: 45,276 samples (all available)

**Test set** - Reduced 10x (maintains original distribution):
- Concur: ~52,910 (68.2%)
- Oppose: ~23,549 (30.3%)
- Neutral: ~1,132 (1.5%)

## Result

A new `redditAITA_train.csv` and  `redditAITA_test.csv`  will be created under `./data/preprocessed`, overwriting the original dataset.

