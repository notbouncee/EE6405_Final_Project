import numpy as np, pandas as pd, torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# Repro
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Model: keep BERT
MODEL_NAME = "answerdotai/ModernBERT-base"

# File paths (put your CSVs in the same folder)
TRAIN_PATH = "data/stance_data_cleaned_train.csv"
TEST_PATH  = "data/stance_data_cleaned_test.csv"

# Labels in your preprocessed data  (edit if required)
LABELS   = ["concur","oppose","neutral"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

# Set hyperparameters
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 4      
MAX_LEN = 128


def load_split(path: str):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    need = {"post_text","comment_text"}  # stance optional for pure inference
    assert need.issubset(df.columns), f"{path} must have columns: {need}"

    # hygiene
    df["post_text"]    = df["post_text"].astype(str).str.strip()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()

    if "stance" in df.columns:
        df["stance"] = df["stance"].astype(str).str.lower().str.strip()
        df = df[df["stance"].isin(LABEL2ID)]               # keep only known labels
        df["label"] = df["stance"].map(LABEL2ID).astype(int)

    # no duplicate triples
    keep_cols = ["post_text","comment_text"] + (["label"] if "label" in df.columns else [])
    df = df.drop_duplicates(subset=keep_cols).reset_index(drop=True)
    return df


train_df = load_split(TRAIN_PATH)
test = load_split(TEST_PATH)


train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['stance'])
print("Train shape:", train.shape)
print("Val shape:", val.shape)
print("Label distribution in train:\n", train["label"].value_counts())
print("Label distribution in val:\n", val["label"].value_counts())

## Tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def make_ds(df: pd.DataFrame, has_labels: bool):
    cols = ["comment_text","post_text"] + (["label"] if has_labels else [])
    ds = Dataset.from_pandas(df[cols])

    def _tok(batch):
        return tokenizer(
            batch["comment_text"],   
            batch["post_text"],     
            padding =True,
            truncation=True,
            max_length=MAX_LEN
        )

    ds = ds.map(_tok, batched=True, remove_columns=["comment_text","post_text"])
    if has_labels:
        ds = ds.with_format("torch", columns=["input_ids","attention_mask","token_type_ids","label"])
    else:
        ds = ds.with_format("torch", columns=["input_ids","attention_mask","token_type_ids"])
    return ds

train_ds = make_ds(train, has_labels=True)
val_ds   = make_ds(val,   has_labels=True)
test_ds  = make_ds(test,  has_labels=("label" in test.columns))

## Computing Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # Accuracy
    acc = accuracy_score(labels, preds)

    # Precision / Recall / F1 (macro and micro)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    # Specificity (average true negative rate)
    cm = confusion_matrix(labels, preds)
    num_classes = cm.shape[0]
    specificity_scores = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # remove row/col i
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_scores.append(specificity)
    specificity_macro = np.mean(specificity_scores)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "specificity_macro": specificity_macro
    }

## Model and Trainer


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

args = TrainingArguments(
    output_dir="./bert_stance_baseline",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    seed=SEED,
    eval_strategy="epoch",        # use correct arg name
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to=[]                         # avoid wandb nagging
)
    # keep it minimal so it works across transformers versions


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

## Evaluating on the validation set

trainer.train()

# Baseline evaluation on val 
metrics = trainer.evaluate(val_ds)
print(metrics)

# If you want a confusion matrix and per-class report:
pred = trainer.predict(val_ds)
y_pred = pred.predictions.argmax(axis=1)
y_true = np.array(val_ds["label"])

print(classification_report(y_true, y_pred, target_names=LABELS))
print(confusion_matrix(y_true, y_pred))


## Hyperparameter tuning with Optuna and Cross-Validation

import optuna
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, Trainer

def make_trainer_for_fold(args, train_ds, val_ds):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,   ##
        eval_dataset=val_ds,        ## 
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

def objective(trial, full_df, n_splits=5):
    # Search space for hyperparameters
    lr = trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True)
    batch = trial.suggest_categorical("per_device_train_batch_size", [16, 32])
    epochs = trial.suggest_int("num_train_epochs", 2, 5)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    f1s = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(full_df, full_df["label"]), 1):
        tr_df = full_df.iloc[tr_idx].reset_index(drop=True)
        va_df = full_df.iloc[va_idx].reset_index(drop=True)
        tr_ds = make_ds(tr_df, has_labels=True)
        va_ds = make_ds(va_df, has_labels=True)

        args = TrainingArguments(
            output_dir=f"./cv_trial{trial.number}_fold{fold}",
            learning_rate=lr,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            num_train_epochs=epochs,
            weight_decay=wd,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            dataloader_num_workers=0,
            logging_steps=50,
            report_to=[],
            seed=SEED
        )

        trainer = make_trainer_for_fold(args, tr_ds, va_ds)
        trainer.train()
        metrics = trainer.evaluate(va_ds)
        f1s.append(metrics["eval_f1_macro"])

        # pruning support
        trial.report(metrics["eval_f1_macro"], step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_f1 = float(np.mean(f1s))
    trial.set_user_attr("fold_f1s", f1s)
    return mean_f1


## Running the Optuna study
# Ensure train_df_reddit has labels
assert "label" in train_df.columns

sampler = optuna.samplers.TPESampler(seed=SEED)
pruner  = optuna.pruners.MedianPruner(n_warmup_steps=1)

study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="bert_stance_cv")
study.optimize(lambda t: objective(t, train_df, n_splits=2), n_trials=10, show_progress_bar=True)

print("Best CV macro-F1:", study.best_value)
print("Best params:", study.best_params)

# Keep all trials for later visuals
trials_df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs","system_attrs"))
trials_df.to_csv("optuna_trials.csv", index=False)
trials_df.head()


## Final model training with best hyperparameters
best = study.best_params

full_train_ds = make_ds(train_df, has_labels=True)
full_test_ds  = make_ds(test,  has_labels=("label" in test.columns))

final_args = TrainingArguments(
    output_dir="./final_cv_best",
    learning_rate=best["learning_rate"],
    per_device_train_batch_size=best["per_device_train_batch_size"],
    per_device_eval_batch_size=best["per_device_train_batch_size"],
    num_train_epochs=best.get("num_train_epochs", EPOCHS),
    weight_decay=best["weight_decay"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    dataloader_num_workers=0,
    logging_steps=50,
    report_to=[],
    seed=SEED
)

final_trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ),
    args=final_args,
    train_dataset=full_train_ds,
    eval_dataset=full_test_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

final_trainer.train()
final_metrics = final_trainer.evaluate(full_test_ds)
print(final_metrics)


## Optuna visualization of results:
from optuna.importance import FanovaImportanceEvaluator
from optuna.visualization.matplotlib import (
    plot_param_importances, plot_optimization_history, plot_slice,
    plot_parallel_coordinate, plot_contour
)
import matplotlib.pyplot as plt

# 1) Importance (fANOVA)
imp = optuna.importance.get_param_importances(study, evaluator=FanovaImportanceEvaluator())
print("Param importances:")
for k, v in imp.items():
    print(f"{k:28s} {v:.3f}")

fig = plot_param_importances(study); fig.suptitle("Hyperparameter Importance"); fig.savefig("param_importances.png", dpi=200, bbox_inches="tight")
fig = plot_optimization_history(study); fig.suptitle("Optimization History"); fig.savefig("opt_history.png", dpi=200, bbox_inches="tight")

# 2) Per-parameter response (effect on score across all trials)
fig = plot_slice(study); fig.suptitle("Per-Parameter Performance Slices"); fig.savefig("param_slices.png", dpi=200, bbox_inches="tight")

# 3) Interactions
fig = plot_parallel_coordinate(study); fig.suptitle("Parameter Interactions"); fig.savefig("parallel_coords.png", dpi=200, bbox_inches="tight")
fig = plot_contour(study); fig.suptitle("Pairwise Contours"); fig.savefig("contours.png", dpi=200, bbox_inches="tight")


## Custom plots per parameter
# trials_df already saved above
import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = trials_df[trials_df["state"]=="COMPLETE"].copy()
df["value"] = df["value"].astype(float)

def plot_numeric_response(df, param, metric_col="value", bins=8):
    x = pd.to_numeric(df[f"params_{param}"], errors="coerce")
    y = df[metric_col]
    m = ~x.isna() & ~y.isna()
    x, y = x[m].values, y[m].values
    if len(x) < 2: return
    plt.figure()
    plt.scatter(x, y, alpha=0.65)
    plt.xlabel(param); plt.ylabel(metric_col); plt.title(f"{param} vs {metric_col}")
    edges = np.linspace(x.min(), x.max(), bins+1)
    idx = np.digitize(x, edges) - 1
    means = [y[idx==i].mean() for i in range(bins)]
    mids  = (edges[:-1] + edges[1:]) / 2
    plt.plot(mids, means, linewidth=2)
    plt.tight_layout(); plt.savefig(f"resp_{param}.png", dpi=200); plt.close()

def plot_categorical_response(df, param, metric_col="value"):
    sub = df[[f"params_{param}", metric_col]].dropna()
    if sub.empty: return
    g = sub.groupby(f"params_{param}")[metric_col]
    cats, means, stds = list(g.mean().index), g.mean().values, g.std().fillna(0).values
    plt.figure()
    pos = np.arange(len(cats))
    plt.bar(pos, means, yerr=stds)
    plt.xticks(pos, cats, rotation=20, ha="right")
    plt.ylabel(metric_col); plt.title(f"{param} (mean ± std)")
    plt.tight_layout(); plt.savefig(f"resp_{param}.png", dpi=200); plt.close()

for c in df.columns:
    if c.startswith("params_"):
        p = c.replace("params_","")
        series = df[c]
        # heuristic: if most values parse to numeric, treat as numeric
        if pd.to_numeric(series, errors="coerce").notna().mean() > 0.7:
            plot_numeric_response(df, p)
        else:
            plot_categorical_response(df, p)
