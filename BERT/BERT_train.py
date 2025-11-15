#!/usr/bin/env python3
"""
BERT-based Stance Classification with Hyperparameter Tuning
Optimized for HPC cluster execution with configurable data paths
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
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Configuration class
class Config:
    """Configuration parameters"""
    def __init__(self, train_path=None, test_path=None, output_dir=None, 
                 batch_size=16, epochs=3, n_trials=20, cv_splits=3,
                 model_name="bert-base-uncased", max_len=128):
        # Reproducibility
        self.SEED = 42
        
        # Model
        self.MODEL_NAME = model_name
        
        # File paths
        self.TRAIN_PATH = train_path or "data/reddit_posts_and_comments_train.csv"
        self.TEST_PATH = test_path or "data/reddit_posts_and_comments_test.csv"
        self.OUTPUT_DIR = Path(output_dir or "./output")
        
        # Labels
        self.LABELS = ["concur", "oppose", "neutral"]
        
        # Training parameters
        self.EPOCHS = epochs
        self.LR = 2e-5
        self.BATCH_SIZE = batch_size
        self.MAX_LEN = max_len
        self.GRADIENT_ACCUMULATION_STEPS = 2  # For effective batch size
        
        # Optimization parameters
        self.N_TRIALS = n_trials
        self.CV_SPLITS = cv_splits
        
        # Derived attributes
        self.LABEL2ID = {l: i for i, l in enumerate(self.LABELS)}
        self.ID2LABEL = {i: l for l, i in self.LABEL2ID.items()}
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds for reproducibility
def set_seeds(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data loading and preprocessing
def load_split(path: str, config: Config, logger):
    """Load and preprocess data split"""
    logger.info(f"Loading data from {path}")
    
    # Check if file exists
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Files in current directory: {os.listdir('.')}")
        if os.path.exists('data'):
            logger.error(f"Files in data directory: {os.listdir('data')}")
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
    
    # Log column names
    logger.info(f"Columns found: {df.columns.tolist()}")
    
    required_cols = {"post_text", "comment_text"}
    if not required_cols.issubset(df.columns):
        logger.error(f"{path} must have columns: {required_cols}")
        logger.error(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Clean text columns
    df["post_text"] = df["post_text"].astype(str).str.strip()
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    
    # Process labels if present
    if "stance" in df.columns:
        df["stance"] = df["stance"].astype(str).str.lower().str.strip()
        
        # Check label distribution before filtering
        logger.info(f"Original label distribution:\n{df['stance'].value_counts().to_dict()}")
        
        # Filter for known labels
        df = df[df["stance"].isin(config.LABEL2ID)]
        df["label"] = df["stance"].map(config.LABEL2ID).astype(int)
        
        logger.info(f"Filtered label distribution:\n{df['label'].value_counts().to_dict()}")
    else:
        logger.warning(f"No 'stance' column found in {path}. This file will be used for inference only.")
    
    # Remove duplicates
    keep_cols = ["post_text", "comment_text"] + (["label"] if "label" in df.columns else [])
    initial_len = len(df)
    df = df.drop_duplicates(subset=keep_cols).reset_index(drop=True)
    
    if initial_len != len(df):
        logger.info(f"Removed {initial_len - len(df)} duplicate rows")
    
    logger.info(f"Final dataset size: {len(df)} samples")
    return df

# Dataset creation
def create_dataset(df: pd.DataFrame, tokenizer, config: Config, has_labels: bool):
    """Create HuggingFace dataset from dataframe"""
    cols = ["comment_text", "post_text"] + (["label"] if has_labels else [])
    ds = Dataset.from_pandas(df[cols])
    
    def tokenize_function(batch):
        return tokenizer(
            batch["comment_text"],
            batch["post_text"],
            padding=True,
            truncation=True,
            max_length=config.MAX_LEN
        )
    
    ds = ds.map(tokenize_function, batched=True, remove_columns=["comment_text", "post_text"])
    
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
def train_baseline(train_df, test_df, config, logger):
    """Train baseline model"""
    logger.info("Starting baseline training...")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Max length: {config.MAX_LEN}")
    
    # Split train into train/val
    train, val = train_test_split(
        train_df, test_size=0.2, random_state=config.SEED, stratify=train_df['stance']
    )
    
    logger.info(f"Train shape: {train.shape}, Val shape: {val.shape}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
    
    # Create datasets
    train_ds = create_dataset(train, tokenizer, config, has_labels=True)
    val_ds = create_dataset(val, tokenizer, config, has_labels=True)
    test_ds = create_dataset(test_df, tokenizer, config, has_labels=("label" in test_df.columns))
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.LABELS),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR / "baseline"),
        learning_rate=config.LR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.EPOCHS,
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
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        seed=config.SEED,
        report_to=["tensorboard"] if torch.cuda.is_available() else [],
        logging_dir=str(config.OUTPUT_DIR / "logs"),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate(test_ds)
    logger.info(f"Baseline test metrics: {metrics}")
    
    # Save results
    with open(config.OUTPUT_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate classification report
    if "label" in test_df.columns:
        pred = trainer.predict(test_ds)
        y_pred = pred.predictions.argmax(axis=1)
        y_true = np.array(test_ds["label"])
        
        report = classification_report(y_true, y_pred, target_names=config.LABELS, output_dict=True)
        with open(config.OUTPUT_DIR / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=config.LABELS)}")
    
    return trainer, tokenizer

# Hyperparameter tuning
def objective(trial, full_df, config, tokenizer, logger):
    """Optuna objective function for hyperparameter tuning"""
    # Hyperparameter search space
    lr = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("num_epochs", 2, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.SEED)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"]), 1):
        logger.info(f"Trial {trial.number}, Fold {fold}/{config.CV_SPLITS}")
        
        # Split data
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_ds = create_dataset(train_df, tokenizer, config, has_labels=True)
        val_ds = create_dataset(val_df, tokenizer, config, has_labels=True)
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=len(config.LABELS),
            id2label=config.ID2LABEL,
            label2id=config.LABEL2ID
        )
        
        # Training arguments
        args = TrainingArguments(
            output_dir=str(config.OUTPUT_DIR / f"trial_{trial.number}_fold_{fold}"),
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4 if torch.cuda.is_available() else 0,
            seed=config.SEED,
            report_to=[],
            disable_tqdm=True,  # Less verbose for trials
        )
        
        # Train
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate(val_ds)
        cv_scores.append(metrics["eval_f1_macro"])
        
        # Report intermediate value for pruning
        trial.report(metrics["eval_f1_macro"], step=fold)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at fold {fold}")
            raise optuna.TrialPruned()
        
        # Clean up to save memory
        del model, trainer
        torch.cuda.empty_cache()
    
    mean_f1 = float(np.mean(cv_scores))
    trial.set_user_attr("cv_scores", cv_scores)
    trial.set_user_attr("cv_std", float(np.std(cv_scores)))
    
    return mean_f1

def run_hyperparameter_tuning(train_df, config, tokenizer, logger):
    """Run Optuna hyperparameter tuning"""
    logger.info("Starting hyperparameter tuning...")
    logger.info(f"Number of trials: {config.N_TRIALS}")
    logger.info(f"CV splits: {config.CV_SPLITS}")
    
    # Create study
    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="bert_stance_optimization"
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_df, config, tokenizer, logger),
        n_trials=config.N_TRIALS,
        show_progress_bar=True
    )
    
    # Save results
    logger.info(f"Best CV F1-Macro: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_df.to_csv(config.OUTPUT_DIR / "optuna_trials.csv", index=False)
    
    # Save best parameters
    with open(config.OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    return study

def train_final_model(train_df, test_df, best_params, config, tokenizer, logger):
    """Train final model with best hyperparameters"""
    logger.info("Training final model with best parameters...")
    logger.info(f"Best parameters: {best_params}")
    
    # Create datasets
    train_ds = create_dataset(train_df, tokenizer, config, has_labels=True)
    test_ds = create_dataset(test_df, tokenizer, config, has_labels=("label" in test_df.columns))
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.LABELS),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID
    )
    
    # Training arguments with best parameters
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR / "final_model"),
        learning_rate=best_params.get("learning_rate", config.LR),
        per_device_train_batch_size=best_params.get("batch_size", config.BATCH_SIZE),
        per_device_eval_batch_size=best_params.get("batch_size", config.BATCH_SIZE),
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=best_params.get("num_epochs", config.EPOCHS),
        weight_decay=best_params.get("weight_decay", 0.01),
        warmup_ratio=best_params.get("warmup_ratio", 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        logging_steps=50,
        seed=config.SEED,
        report_to=["tensorboard"] if torch.cuda.is_available() else [],
        logging_dir=str(config.OUTPUT_DIR / "logs" / "final"),
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Evaluate and save
    if "label" in test_df.columns:
        metrics = trainer.evaluate(test_ds)
        logger.info(f"Final test metrics: {metrics}")
        
        with open(config.OUTPUT_DIR / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Generate predictions
        pred = trainer.predict(test_ds)
        y_pred = pred.predictions.argmax(axis=1)
        y_true = np.array(test_ds["label"])
        
        # Save classification report
        report = classification_report(y_true, y_pred, target_names=config.LABELS, output_dict=True)
        with open(config.OUTPUT_DIR / "final_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.save(config.OUTPUT_DIR / "confusion_matrix.npy", cm)
        
        logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=config.LABELS)}")
    
    # Save model
    trainer.save_model(str(config.OUTPUT_DIR / "final_model"))
    tokenizer.save_pretrained(str(config.OUTPUT_DIR / "final_model"))
    
    return trainer

def visualize_results(study, config, logger):
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
        fig = plot_param_importances(study)
        fig.suptitle("Hyperparameter Importance")
        fig.savefig(config.OUTPUT_DIR / "param_importances.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        fig = plot_optimization_history(study)
        fig.suptitle("Optimization History")
        fig.savefig(config.OUTPUT_DIR / "optimization_history.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        fig = plot_slice(study)
        fig.suptitle("Parameter Performance Slices")
        fig.savefig(config.OUTPUT_DIR / "param_slices.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        fig = plot_parallel_coordinate(study)
        fig.suptitle("Parameter Interactions")
        fig.savefig(config.OUTPUT_DIR / "parallel_coordinates.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        logger.info("Visualizations saved successfully")
        
    except Exception as e:
        logger.warning(f"Error generating visualizations: {e}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='BERT Stance Classification Training')
    
    # Data paths
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test CSV file')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory for saving outputs (default: ./output)')
    
    # Model parameters
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                        help='Pretrained model name (default: bert-base-uncased)')
    parser.add_argument('--max-len', type=int, default=128,
                        help='Maximum sequence length (default: 128)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    
    # Optimization parameters
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of Optuna trials (default: 20)')
    parser.add_argument('--cv-splits', type=int, default=3,
                        help='Number of CV splits (default: 3)')
    
    # Execution control
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline training')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='Skip hyperparameter tuning')
    parser.add_argument('--skip-final', action='store_true',
                        help='Skip final model training')
    
    return parser.parse_args()

def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize configuration
    config = Config(
        train_path=args.train_file,
        test_path=args.test_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_trials=args.n_trials,
        cv_splits=args.cv_splits,
        model_name=args.model_name,
        max_len=args.max_len
    )
    
    # Setup logging
    logger = setup_logging(config.OUTPUT_DIR)
    logger.info("Starting stance classification training pipeline")
    logger.info(f"Command-line arguments: {vars(args)}")
    logger.info(f"Configuration: {vars(config)}")
    
    # Set seeds
    set_seeds(config.SEED)
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        logger.warning("No GPU available, training will be slow")
    
    try:
        # Load data
        train_df = load_split(config.TRAIN_PATH, config, logger)
        test_df = load_split(config.TEST_PATH, config, logger)
        
        # Verify labels exist in training data
        if "label" not in train_df.columns:
            logger.error("Training data must have stance labels!")
            sys.exit(1)
        
        tokenizer = None
        study = None
        
        # Train baseline model
        if not args.skip_baseline:
            trainer, tokenizer = train_baseline(train_df, test_df, config, logger)
        else:
            logger.info("Skipping baseline training as requested")
            # Still need to initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
        
        # Run hyperparameter tuning
        if not args.skip_tuning:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
            study = run_hyperparameter_tuning(train_df, config, tokenizer, logger)
            best_params = study.best_params
        else:
            logger.info("Skipping hyperparameter tuning as requested")
            # Use default parameters
            best_params = {
                "learning_rate": config.LR,
                "batch_size": config.BATCH_SIZE,
                "num_epochs": config.EPOCHS,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1
            }
        
        # Train final model with best parameters
        if not args.skip_final:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
            final_trainer = train_final_model(
                train_df, test_df, best_params, config, tokenizer, logger
            )
        else:
            logger.info("Skipping final model training as requested")
        
        # Generate visualizations
        if study is not None:
            visualize_results(study, config, logger)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()