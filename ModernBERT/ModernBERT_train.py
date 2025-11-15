#!/usr/bin/env python3
"""
ModernBERT Stance Classification with Hyperparameter Tuning
Optimized for HPC cluster execution with configurable parameters
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import optuna
from optuna.importance import FanovaImportanceEvaluator
from optuna.visualization.matplotlib import (
    plot_param_importances, plot_optimization_history, 
    plot_slice, plot_parallel_coordinate, plot_contour
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt

# Setup logging
def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"modernbert_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data loading and preprocessing
def load_split(path: str, logger):
    """Load and preprocess data split"""
    logger.info(f"Loading data from {path}")
    
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded {len(df)} rows from {path}")
    except UnicodeDecodeError:
        logger.warning(f"Unicode decode error, trying with latin1 encoding...")
        df = pd.read_csv(path, encoding="latin1")
        logger.info(f"Successfully loaded {len(df)} rows with latin1 encoding")
    except Exception as e:
        logger.error(f"Error loading file {path}: {e}")
        sys.exit(1)
    
    # Check required columns
    required_cols = {"post_text", "comment_text"}
    if not required_cols.issubset(df.columns):
        logger.error(f"{path} must have columns: {required_cols}")
        logger.error(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Clean text columns
    df["post_text"] = df["post_text"].astype(str).str.strip()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    
    # Process labels if present
    label2id = {"concur": 0, "oppose": 1, "neutral": 2}
    if "stance" in df.columns:
        df["stance"] = df["stance"].astype(str).str.lower().str.strip()
        logger.info(f"Original label distribution:\n{df['stance'].value_counts().to_dict()}")
        
        # Filter for known labels
        df = df[df["stance"].isin(label2id.keys())]
        df["label"] = df["stance"].map(label2id).astype(int)
        logger.info(f"Filtered label distribution:\n{df['label'].value_counts().to_dict()}")
    
    # Remove duplicates
    keep_cols = ["post_text", "comment_text"] + (["label"] if "label" in df.columns else [])
    initial_len = len(df)
    df = df.drop_duplicates(subset=keep_cols).reset_index(drop=True)
    
    if initial_len != len(df):
        logger.info(f"Removed {initial_len - len(df)} duplicate rows")
    
    logger.info(f"Final dataset size: {len(df)} samples")
    return df

# Dataset creation
def create_dataset(df: pd.DataFrame, tokenizer, max_len: int, has_labels: bool):
    """Create HuggingFace dataset from dataframe"""
    cols = ["comment_text", "post_text"] + (["label"] if has_labels else [])
    ds = Dataset.from_pandas(df[cols])
    
    def tokenize_function(batch):
        return tokenizer(
            batch["comment_text"],
            batch["post_text"],
            padding=True,
            truncation=True,
            max_length=max_len
        )
    
    ds = ds.map(tokenize_function, batched=True, remove_columns=["comment_text", "post_text"])
    
    # ModernBERT may not have token_type_ids
    format_cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in ds.column_names:
        format_cols.append("token_type_ids")
    if has_labels:
        format_cols.append("label")
    
    ds = ds.with_format("torch", columns=format_cols)
    return ds

# Metrics computation
def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    # Core metrics
    acc = accuracy_score(labels, preds)
    
    # Precision, Recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    
    # Specificity
    cm = confusion_matrix(labels, preds)
    num_classes = cm.shape[0]
    specificity_scores = []
    
    for i in range(num_classes):
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

# Training functions
def train_baseline(train_df, test_df, args, logger):
    """Train baseline model"""
    logger.info("Starting baseline training with ModernBERT...")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Max length: {args.max_len}")
    
    # Split train into train/val
    train, val = train_test_split(
        train_df, test_size=0.2, random_state=args.seed, stratify=train_df['stance']
    )
    logger.info(f"Train shape: {train.shape}, Val shape: {val.shape}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # Create datasets
    train_ds = create_dataset(train, tokenizer, args.max_len, has_labels=True)
    val_ds = create_dataset(val, tokenizer, args.max_len, has_labels=True)
    test_ds = create_dataset(test_df, tokenizer, args.max_len, has_labels=("label" in test_df.columns))
    
    # Label mappings
    labels = ["concur", "oppose", "neutral"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # For ModernBERT compatibility
    )
    
    # Training arguments
    output_dir = Path(args.output_dir) / "baseline"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        seed=args.seed,
        report_to=["tensorboard"] if not args.no_tensorboard else [],
        logging_dir=str(output_dir / "logs"),
        remove_unused_columns=False,  # Important for ModernBERT
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding=True),
        compute_metrics=compute_metrics
    )
    
    # Train
    trainer.train()
    
    # Evaluate on test set
    metrics = trainer.evaluate(test_ds)
    logger.info(f"Baseline test metrics: {metrics}")
    
    # Save results
    output_dir = Path(args.output_dir)
    with open(output_dir / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate classification report if test has labels
    if "label" in test_df.columns:
        pred = trainer.predict(test_ds)
        y_pred = pred.predictions.argmax(axis=1)
        y_true = np.array(test_ds["label"])
        
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        with open(output_dir / "baseline_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=labels)}")
        
        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.save(output_dir / "baseline_confusion_matrix.npy", cm)
        logger.info(f"Confusion Matrix:\n{cm}")
    
    return trainer, tokenizer

# Hyperparameter tuning
def objective(trial, train_df, args, tokenizer, logger):
    """Optuna objective function for hyperparameter tuning"""
    # Hyperparameter search space
    lr = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("num_epochs", 2, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
    cv_scores = []
    
    labels = ["concur", "oppose", "neutral"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"]), 1):
        logger.info(f"Trial {trial.number}, Fold {fold}/{args.cv_splits}")
        
        # Split data
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_ds = create_dataset(train_fold_df, tokenizer, args.max_len, has_labels=True)
        val_ds = create_dataset(val_fold_df, tokenizer, args.max_len, has_labels=True)
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        
        # Training arguments
        output_dir = Path(args.output_dir) / f"trial_{trial.number}_fold_{fold}"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            fp16=torch.cuda.is_available() and not args.no_fp16,
            dataloader_num_workers=4 if torch.cuda.is_available() else 0,
            seed=args.seed,
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
        )
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer, padding=True),
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate(val_ds)
        cv_scores.append(metrics["eval_f1_macro"])
        
        # Report for pruning
        trial.report(metrics["eval_f1_macro"], step=fold)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at fold {fold}")
            raise optuna.TrialPruned()
        
        # Clean up
        del model, trainer
        torch.cuda.empty_cache()
    
    mean_f1 = float(np.mean(cv_scores))
    trial.set_user_attr("cv_scores", cv_scores)
    trial.set_user_attr("cv_std", float(np.std(cv_scores)))
    
    return mean_f1

def run_hyperparameter_tuning(train_df, args, tokenizer, logger):
    """Run Optuna hyperparameter tuning"""
    logger.info("Starting hyperparameter tuning...")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"CV splits: {args.cv_splits}")
    
    # Create study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="modernbert_stance_optimization"
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_df, args, tokenizer, logger),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Save results
    logger.info(f"Best CV F1-Macro: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save trials dataframe
    output_dir = Path(args.output_dir)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    
    # Save best parameters
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    return study

def train_final_model(train_df, test_df, best_params, args, tokenizer, logger):
    """Train final model with best hyperparameters"""
    logger.info("Training final model with best parameters...")
    logger.info(f"Best parameters: {best_params}")
    
    # Split for validation during final training
    train, val = train_test_split(
        train_df, test_size=0.2, random_state=args.seed, stratify=train_df['stance']
    )
    
    # Create datasets  
    train_ds = create_dataset(train, tokenizer, args.max_len, has_labels=True)
    val_ds = create_dataset(val, tokenizer, args.max_len, has_labels=True)
    test_ds = create_dataset(test_df, tokenizer, args.max_len, has_labels=("label" in test_df.columns))
    
    labels = ["concur", "oppose", "neutral"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Training arguments with best parameters
    output_dir = Path(args.output_dir) / "final_model"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=best_params.get("learning_rate", args.learning_rate),
        per_device_train_batch_size=best_params.get("batch_size", args.batch_size),
        per_device_eval_batch_size=best_params.get("batch_size", args.batch_size),
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=best_params.get("num_epochs", args.epochs),
        weight_decay=best_params.get("weight_decay", 0.01),
        warmup_ratio=best_params.get("warmup_ratio", 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        logging_steps=50,
        seed=args.seed,
        report_to=["tensorboard"] if not args.no_tensorboard else [],
        logging_dir=str(output_dir / "logs"),
        remove_unused_columns=False,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding=True),
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Evaluate on test set
    if "label" in test_df.columns:
        metrics = trainer.evaluate(test_ds)
        logger.info(f"Final test metrics: {metrics}")
        
        output_dir = Path(args.output_dir)
        with open(output_dir / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Generate predictions
        pred = trainer.predict(test_ds)
        y_pred = pred.predictions.argmax(axis=1)
        y_true = np.array(test_ds["label"])
        
        # Save classification report
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        with open(output_dir / "final_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.save(output_dir / "final_confusion_matrix.npy", cm)
        
        logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=labels)}")
        logger.info(f"Confusion Matrix:\n{cm}")
    
    # Save model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    return trainer

def visualize_results(study, output_dir, logger):
    """Generate visualization plots for Optuna study"""
    logger.info("Generating visualization plots...")
    
    try:
        # Parameter importance
        imp = optuna.importance.get_param_importances(
            study, evaluator=FanovaImportanceEvaluator()
        )
        logger.info("Parameter importances:")
        for k, v in imp.items():
            logger.info(f"{k:30s} {v:.3f}")
        
        # Generate plots
        ax = plot_param_importances(study)
        ax.figure.savefig(output_dir / "param_importances.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        ax = plot_optimization_history(study)
        ax.figure.savefig(output_dir / "optimization_history.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        # Parameter slices
        axes = plot_slice(study)
        if axes:
            fig = axes[0].figure
            fig.suptitle("Parameter Performance Slices")
            fig.savefig(output_dir / "param_slices.png", dpi=200, bbox_inches="tight")
            plt.close()
        
        # Parallel coordinates
        ax = plot_parallel_coordinate(study)
        fig = ax.figure
        fig.suptitle("Parameter Interactions")
        fig.savefig(output_dir / "parallel_coordinates.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        logger.info("Visualizations saved successfully")
        
    except Exception as e:
        logger.warning(f"Error generating visualizations: {e}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='ModernBERT Stance Classification Training')
    
    # Data paths
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test CSV file')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./output_modernbert',
                        help='Directory for saving outputs')
    
    # Model parameters
    parser.add_argument('--model-name', type=str, default='answerdotai/ModernBERT-base',
                        help='Pretrained model name')
    parser.add_argument('--max-len', type=int, default=128,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Initial learning rate')
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='Gradient accumulation steps')
    
    # Optimization parameters
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Number of Optuna trials')
    parser.add_argument('--cv-splits', type=int, default=2,
                        help='Number of CV splits')
    
    # Execution control
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline training')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip hyperparameter tuning')
    parser.add_argument('--skip-final', action='store_true',
                        help='Skip final model training')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable tensorboard logging')
    
    return parser.parse_args()

def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting ModernBERT stance classification training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set seeds
    set_seeds(args.seed)
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        logger.warning("No GPU available, training will be slow")
    
    try:
        # Load data
        train_df = load_split(args.train_file, logger)
        test_df = load_split(args.test_file, logger)
        
        # Verify labels exist in training data
        if "label" not in train_df.columns:
            logger.error("Training data must have stance labels!")
            sys.exit(1)
        
        tokenizer = None
        study = None
        
        # Train baseline model
        if not args.skip_baseline:
            trainer, tokenizer = train_baseline(train_df, test_df, args, logger)
        else:
            logger.info("Skipping baseline training as requested")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        
        # Run hyperparameter tuning
        if not args.skip_tuning:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
            study = run_hyperparameter_tuning(train_df, args, tokenizer, logger)
            best_params = study.best_params
        else:
            logger.info("Skipping hyperparameter tuning as requested")
            best_params = {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1
            }
        
        # Train final model
        if not args.skip_final:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
            final_trainer = train_final_model(
                train_df, test_df, best_params, args, tokenizer, logger
            )
        else:
            logger.info("Skipping final model training as requested")
        
        # Generate visualizations
        if study is not None:
            visualize_results(study, output_dir, logger)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()