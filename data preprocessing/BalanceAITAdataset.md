# Balance redditAITA Training Set

## Step 1: Run Preprocessing Script

```bash
python "data preprocessing/redditAITApreprocessing.py"
```

This will generate the following files under `./data/preprocessed`:
- `redditAITA_train.csv`
- `redditAITA_test.csv`
- `redditAITA.csv`

## Step 2: Balance the Training Set

```bash
python "data preprocessing/balance_dataset.py"
```

This will balance the training set with the following distribution:
- Concur: 100,000 samples
- Oppose: 100,000 samples
- Neutral: 45,276 samples (all available)

Test set unchanged (keeps original distribution for realistic evaluation).

## Result

A new `redditAITA_train.csv` will be created under `./data/preprocessed`, overwriting the original unbalanced training set.