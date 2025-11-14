"""
BERTweet Stance Detection with Hyperparameter Tuning
"""

import numpy as np
import pandas as pd
import torch
import optuna
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            precision_recall_fscore_support)
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                         TrainingArguments, Trainer, DataCollatorWithPadding)
from optuna.importance import FanovaImportanceEvaluator
from optuna.visualization.matplotlib import (plot_param_importances, plot_optimization_history,
                                            plot_slice, plot_parallel_coordinate, plot_contour)


# ==================== CONFIGURATION ====================

# Model
MODEL_NAME = "vinai/bertweet-large"
LABELS = ["concur", "oppose", "neutral"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Data paths
TRAIN_PATH = "data/preprocessed/reddit_posts_and_comments_train.csv"
TEST_PATH = "data/preprocessed/reddit_posts_and_comments_test.csv"

# Training hyperparameters
SEED = 42
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 4
MAX_LEN = 512

# Optuna settings
N_TRIALS = 10
N_CV_SPLITS = 2

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==================== DATA UTILITIES ====================

def load_split(path: str):
    """Load and clean CSV data"""
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    # Validate columns
    need = {"post_text", "comment_text"}
    assert need.issubset(df.columns), f"{path} must have columns: {need}"

    # Clean text
    df["post_text"] = df["post_text"].astype(str).str.strip()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()

    # Process labels if present
    if "stance" in df.columns:
        df["stance"] = df["stance"].astype(str).str.lower().str.strip()
        df = df[df["stance"].isin(LABEL2ID)]
        df["label"] = df["stance"].map(LABEL2ID).astype(int)

    # Remove duplicates
    keep_cols = ["post_text", "comment_text"] + (["label"] if "label" in df.columns else [])
    df = df.drop_duplicates(subset=keep_cols).reset_index(drop=True)
    
    return df


def make_dataset(df: pd.DataFrame, tokenizer, has_labels: bool):
    """Convert DataFrame to HuggingFace Dataset with tokenization"""
    cols = ["comment_text", "post_text"] + (["label"] if has_labels else [])
    ds = Dataset.from_pandas(df[cols])

    def tokenize_batch(batch):
        return tokenizer(
            batch["comment_text"],
            batch["post_text"],
            truncation=True,
            max_length=MAX_LEN
        )

    ds = ds.map(tokenize_batch, batched=True, remove_columns=["comment_text", "post_text"])
    
    # BERTweet uses RoBERTa tokenizer - no token_type_ids
    format_cols = ["input_ids", "attention_mask"] + (["label"] if has_labels else [])
    ds = ds.with_format("torch", columns=format_cols)
    
    return ds


# ==================== METRICS ====================

def compute_metrics(eval_pred):
    """Compute classification metrics"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    # Specificity (average true negative rate)
    cm = confusion_matrix(labels, preds)
    specificity_scores = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
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


# ==================== MODEL UTILITIES ====================

def create_model():
    """Create fresh model instance"""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )


def create_trainer(model, args, train_ds, eval_ds, tokenizer):
    """Create Trainer instance"""
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )


# ==================== HYPERPARAMETER OPTIMIZATION ====================

def optuna_objective(trial, full_df, tokenizer):
    """Optuna objective function with cross-validation"""
    # Hyperparameter search space
    lr = trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True)
    batch = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    epochs = trial.suggest_int("num_train_epochs", 2, 5)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)

    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=SEED)
    f1_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(full_df, full_df["label"]), 1):
        tr_df = full_df.iloc[tr_idx].reset_index(drop=True)
        va_df = full_df.iloc[va_idx].reset_index(drop=True)
        tr_ds = make_dataset(tr_df, tokenizer, has_labels=True)
        va_ds = make_dataset(va_df, tokenizer, has_labels=True)

        args = TrainingArguments(
            output_dir=f"./cv_trial{trial.number}_fold{fold}",
            learning_rate=lr,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            num_train_epochs=epochs,
            weight_decay=wd,
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            dataloader_num_workers=0,
            logging_steps=50,
            report_to=[],
            seed=SEED
        )

        trainer = create_trainer(create_model(), args, tr_ds, va_ds, tokenizer)
        trainer.train()
        metrics = trainer.evaluate(va_ds)
        f1_scores.append(metrics["eval_f1_macro"])

        # Pruning support
        trial.report(metrics["eval_f1_macro"], step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_f1 = float(np.mean(f1_scores))
    trial.set_user_attr("fold_f1s", f1_scores)
    return mean_f1


# ==================== VISUALIZATION ====================

def plot_param_response(trials_df, param, metric_col="value"):
    """Plot parameter vs metric relationship"""
    df = trials_df[trials_df["state"] == "COMPLETE"].copy()
    df["value"] = df["value"].astype(float)
    
    x = pd.to_numeric(df[f"params_{param}"], errors="coerce")
    y = df[metric_col]
    mask = ~x.isna() & ~y.isna()
    x, y = x[mask].values, y[mask].values
    
    if len(x) < 2:
        return

    # Numeric plot with binned means
    if pd.to_numeric(df[f"params_{param}"], errors="coerce").notna().mean() > 0.7:
        plt.figure()
        plt.scatter(x, y, alpha=0.65)
        plt.xlabel(param)
        plt.ylabel(metric_col)
        plt.title(f"{param} vs {metric_col}")
        
        bins = 8
        edges = np.linspace(x.min(), x.max(), bins + 1)
        idx = np.digitize(x, edges) - 1
        means = [y[idx == i].mean() for i in range(bins)]
        mids = (edges[:-1] + edges[1:]) / 2
        plt.plot(mids, means, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f"resp_{param}.png", dpi=200)
        plt.close()
    
    # Categorical plot
    else:
        sub = df[[f"params_{param}", metric_col]].dropna()
        if sub.empty:
            return
        g = sub.groupby(f"params_{param}")[metric_col]
        cats = list(g.mean().index)
        means = g.mean().values
        stds = g.std().fillna(0).values
        
        plt.figure()
        pos = np.arange(len(cats))
        plt.bar(pos, means, yerr=stds)
        plt.xticks(pos, cats, rotation=20, ha="right")
        plt.ylabel(metric_col)
        plt.title(f"{param} (mean ± std)")
        plt.tight_layout()
        plt.savefig(f"resp_{param}.png", dpi=200)
        plt.close()


def generate_visualizations(study, trials_df):
    """Generate all Optuna visualization plots"""
    # Importance
    imp = optuna.importance.get_param_importances(study, evaluator=FanovaImportanceEvaluator())
    print("\nHyperparameter Importances:")
    for k, v in imp.items():
        print(f"  {k:28s} {v:.3f}")

    fig = plot_param_importances(study)
    fig.suptitle("Hyperparameter Importance")
    fig.savefig("param_importances.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig = plot_optimization_history(study)
    fig.suptitle("Optimization History")
    fig.savefig("opt_history.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig = plot_slice(study)
    fig.suptitle("Per-Parameter Performance Slices")
    fig.savefig("param_slices.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig = plot_parallel_coordinate(study)
    fig.suptitle("Parameter Interactions")
    fig.savefig("parallel_coords.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig = plot_contour(study)
    fig.suptitle("Pairwise Contours")
    fig.savefig("contours.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Custom per-parameter plots
    for col in trials_df.columns:
        if col.startswith("params_"):
            param = col.replace("params_", "")
            plot_param_response(trials_df, param)


# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 80)
    print("BERTweet Stance Detection - Training Pipeline")
    print("=" * 80)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df = load_split(TRAIN_PATH)
    test_df = load_split(TEST_PATH)
    train, val = train_test_split(train_df, test_size=0.2, random_state=SEED, 
                                  stratify=train_df['stance'])
    
    print(f"  Train: {train.shape[0]} samples")
    print(f"  Val:   {val.shape[0]} samples")
    print(f"  Test:  {test_df.shape[0]} samples")
    
    # Initialize tokenizer
    print("\n[2/6] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, normalization=True)
    
    # Prepare datasets
    train_ds = make_dataset(train, tokenizer, has_labels=True)
    val_ds = make_dataset(val, tokenizer, has_labels=True)
    test_ds = make_dataset(test_df, tokenizer, has_labels=("label" in test_df.columns))
    
    # Baseline training
    print("\n[3/6] Training baseline model...")
    baseline_args = TrainingArguments(
        output_dir="./BERTweet_reddit_baseline",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        seed=SEED,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[]
    )
    
    baseline_trainer = create_trainer(create_model(), baseline_args, train_ds, val_ds, tokenizer)
    baseline_trainer.train()
    
    # Evaluate baseline
    print("\n  Baseline validation results:")
    metrics = baseline_trainer.evaluate(val_ds)
    for k, v in metrics.items():
        if not k.startswith("eval_"):
            continue
        print(f"    {k}: {v:.4f}")
    
    pred = baseline_trainer.predict(val_ds)
    y_pred = pred.predictions.argmax(axis=1)
    y_true = np.array(val_ds["label"])
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS))
    
    # Hyperparameter optimization
    print(f"\n[4/6] Starting Optuna optimization ({N_TRIALS} trials, {N_CV_SPLITS}-fold CV)...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name="bertweet_stance_cv")
    
    study.optimize(lambda t: optuna_objective(t, train_df, tokenizer), 
                   n_trials=N_TRIALS, show_progress_bar=True)
    
    print(f"\n  Best CV macro-F1: {study.best_value:.4f}")
    print("  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    
    # Save trials
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params",
                                              "user_attrs", "system_attrs"))
    trials_df.to_csv("optuna_trials.csv", index=False)
    
    # Final model with best hyperparameters
    print("\n[5/6] Training final model with best hyperparameters...")
    best = study.best_params
    full_train_ds = make_dataset(train_df, tokenizer, has_labels=True)
    full_test_ds = make_dataset(test_df, tokenizer, has_labels=("label" in test_df.columns))
    
    final_args = TrainingArguments(
        output_dir="./final_cv_reddit_best",
        learning_rate=best["learning_rate"],
        per_device_train_batch_size=best["per_device_train_batch_size"],
        per_device_eval_batch_size=best["per_device_train_batch_size"],
        num_train_epochs=best.get("num_train_epochs", EPOCHS),
        weight_decay=best["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        dataloader_num_workers=0,
        logging_steps=50,
        report_to=[],
        seed=SEED
    )
    
    final_trainer = create_trainer(create_model(), final_args, full_train_ds, 
                                   full_test_ds, tokenizer)
    final_trainer.train()
    
    print("\n  Final test results:")
    final_metrics = final_trainer.evaluate(full_test_ds)
    for k, v in final_metrics.items():
        if not k.startswith("eval_"):
            continue
        print(f"    {k}: {v:.4f}")
    
    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    generate_visualizations(study, trials_df)
    print("  Saved: param_importances.png, opt_history.png, param_slices.png,")
    print("         parallel_coords.png, contours.png, resp_*.png")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()