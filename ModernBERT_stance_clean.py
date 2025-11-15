import numpy as np, pandas as pd, torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import optuna
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, Trainer, TrainerCallback
from optuna.importance import FanovaImportanceEvaluator
from optuna.visualization.matplotlib import (
    plot_param_importances, plot_optimization_history, plot_intermediate_values
)
import matplotlib.pyplot as plt


SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

MODEL_NAME = "answerdotai/ModernBERT-base"

TRAIN_PATH = "data/stance_data_cleaned_train.csv"
TEST_PATH  = "data/stance_data_cleaned_test.csv"

LABELS   = ["concur","oppose","neutral"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

# Set hyperparameters for baseline model
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
MAX_LEN = 128


def load_data(path: str):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    need = {"post_text","comment_text"}
    assert need.issubset(df.columns), f"{path} must have columns: {need}"

    df["post_text"]    = df["post_text"].astype(str).str.strip()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()

    if "stance" in df.columns:
        df["stance"] = df["stance"].astype(str).str.lower().str.strip()
        df = df[df["stance"].isin(LABEL2ID)]
        df["label"] = df["stance"].map(LABEL2ID).astype(int)

    keep_cols = ["post_text","comment_text"] + (["label"] if "label" in df.columns else [])
    df = df.drop_duplicates(subset=keep_cols).reset_index(drop=True)
    return df


train_df = load_data(TRAIN_PATH)
test = load_data(TEST_PATH)


train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['stance'])
print("Train shape:", train.shape)
print("Val shape:", val.shape)
print("Label distribution in train:\n", train["label"].value_counts())
print("Label distribution in val:\n", val["label"].value_counts())

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
        ds = ds.with_format("torch", columns=["input_ids","attention_mask","label"])
    else:
        ds = ds.with_format("torch", columns=["input_ids","attention_mask"])
    return ds

train_ds = make_ds(train, has_labels=True)
val_ds   = make_ds(val,   has_labels=True)
test_ds  = make_ds(test,  has_labels=("label" in test.columns))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

args = TrainingArguments(
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    seed=SEED,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.early_stopping_patience = num_steps


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(num_steps=3)]
)

trainer.train()

metrics = trainer.evaluate(test_ds)
print(metrics)

# Show classification report
pred = trainer.predict(test_ds)
y_pred = pred.predictions.argmax(axis=1)
y_true = np.array(test_ds["label"])

print(classification_report(y_true, y_pred, target_names=LABELS))
print(confusion_matrix(y_true, y_pred))

def create_trainer(args, train_ds, val_ds):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(num_steps=3)]
    )

def objective(trial, full_df, n_splits=5):
    lr = trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True)
    batch = trial.suggest_categorical("per_device_train_batch_size", [16, 32])
    epochs = trial.suggest_int("num_train_epochs", 2, 5)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    f1s = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(full_df, full_df["label"]), 1):
        train_set = full_df.iloc[tr_idx].reset_index(drop=True)
        eval_set = full_df.iloc[va_idx].reset_index(drop=True)
        train_ds = make_ds(train_set, has_labels=True)
        eval_ds = make_ds(eval_set, has_labels=True)

        args = TrainingArguments(
            learning_rate=lr,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            num_train_epochs=epochs,
            weight_decay=wd,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            seed=SEED,
            report_to="none"
        )

        trainer = create_trainer(args, train_ds, eval_ds)
        trainer.train()
        metrics = trainer.evaluate(eval_ds)
        f1s.append(metrics["eval_f1_macro"])

        # pruning support
        trial.report(metrics["eval_f1_macro"], step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_f1 = float(np.mean(f1s))
    return mean_f1


# Run optuna study
sampler = optuna.samplers.TPESampler(seed=SEED)
pruner  = optuna.pruners.MedianPruner(n_warmup_steps=1)

study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="bert_stance_cv")
study.optimize(lambda t: objective(t, train_df, n_splits=2), n_trials=10, show_progress_bar=True)

print("Best CV macro-F1:", study.best_value)
print("Best params:", study.best_params)

# Keep all trials for visualisation
trials_df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs","system_attrs"))
trials_df.to_csv("optuna_trials.csv", index=False)
trials_df.head()

## Train final model with best hyperparameters
best_params = study.best_params

final_args = TrainingArguments(
    output_dir="./final_cv_best",
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=best_params["per_device_train_batch_size"],
    per_device_eval_batch_size=best_params["per_device_train_batch_size"],
    num_train_epochs=best_params.get("num_train_epochs", EPOCHS),
    weight_decay=best_params["weight_decay"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    seed=SEED,
    report_to="none"
)

final_trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ),
    args=final_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

final_trainer.train()
final_metrics = final_trainer.evaluate(test_ds)
print(final_metrics)


# Visualise results
imp = optuna.importance.get_param_importances(study, evaluator=FanovaImportanceEvaluator())
print("Param importances:")
for k, v in imp.items():
    print(f"{k:28s} {v:.3f}")

ax1 = plot_optimization_history(study)
ax1.figure.savefig("optimization_history.png")

ax2 = plot_intermediate_values(study)
ax2.figure.savefig("intermediate_values.png")

ax3 = plot_param_importances(study)
ax3.figure.savefig("param_importances.png")