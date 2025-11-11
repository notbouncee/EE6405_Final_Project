"""
Stance Detection Training Script (Stage 2) with Optuna Hyperparameter Tuning
"""

import argparse
import os
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback
)
import optuna


# ==================== DEVICE SETUP ====================

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: CUDA (GPU) - {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = "mps"
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = "cpu"
    print(f"Using device: CPU")


# ==================== CONFIGURATION ====================

STANCE_LABELS = ["concur", "oppose", "neutral"]
LABEL_MAP = {label: idx for idx, label in enumerate(STANCE_LABELS)}


# ==================== DATA LOADING ====================

def load_dataset(dataset_name, split="train", data_dir="./data/preprocessed"):
    """Load and standardize dataset to common format"""
    
    file_path = f"{data_dir}/{dataset_name}_{split}.csv"
    df = pd.read_csv(file_path)
    
    # Standardize column names based on dataset
    if dataset_name == "reddit_posts_and_comments":
        df['post_text'] = df['post_text'].fillna('').astype(str)
        df['comment_text'] = df['comment_text'].fillna('').astype(str)
        df['text'] = df['post_text'] + " [SEP] " + df['comment_text']
        df['label'] = df['stance']
        
    elif dataset_name == "redditAITA":
        df['post'] = df['post'].fillna('').astype(str)
        df['comment'] = df['comment'].fillna('').astype(str)
        df['text'] = df['post'] + " [SEP] " + df['comment']
        df['label'] = df['stance']
        
    elif dataset_name == "stance_dataset":
        df['target_text'] = df['target_text'].fillna('').astype(str)
        df['response_text'] = df['response_text'].fillna('').astype(str)
        df['text'] = df['target_text'] + " [SEP] " + df['response_text']
        df['label'] = df['label']
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert labels to indices
    # First, filter out any invalid labels (like "Queries")
    valid_labels = list(LABEL_MAP.keys())
    initial_count = len(df)
    df = df[df['label'].isin(valid_labels)]
    filtered_count = initial_count - len(df)
    
    if filtered_count > 0:
        print(f"  ℹ Filtered out {filtered_count} rows with invalid labels (e.g., 'Queries')")
    
    if len(df) == 0:
        raise ValueError(f"No valid labels found in {dataset_name} {split}. Expected: {valid_labels}")
    
    df['label'] = df['label'].map(LABEL_MAP)
    
    # Drop any rows with NaN labels (invalid data)
    df = df.dropna(subset=['label'])
    
    # Ensure text is string type and not empty
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']
    
    return df[['text', 'label']].reset_index(drop=True)


def prepare_datasets(dataset_name, tokenizer, max_length=128):
    """Load and tokenize train/test datasets"""
    
    print(f"\nLoading {dataset_name} dataset...")
    train_df = load_dataset(dataset_name, split="train")
    test_df = load_dataset(dataset_name, split="test")
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Validate data before tokenization
    print(f"  Validating data...")
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty for {dataset_name}")
    if len(test_dataset) == 0:
        raise ValueError(f"Test dataset is empty for {dataset_name}")
    
    # Check for any remaining NaN or invalid values
    sample_text = train_dataset[0]['text']
    if not isinstance(sample_text, str):
        raise TypeError(f"Text data is not string type: {type(sample_text)}")
    
    print(f"  Tokenizing...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    print(f"  ✓ Tokenization complete")
    
    return train_dataset, test_dataset, train_df, test_df


# ==================== TRAINING ====================

class MetricsCallback(TrainerCallback):
    """Callback to track metrics during training"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1s = []
        self.epoch_times = []
        self.start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        self.epoch_times.append(elapsed)
        print(f"Epoch {state.epoch} completed in {elapsed/60:.2f} minutes")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.val_losses.append(logs['eval_loss'])
            if 'eval_accuracy' in logs:
                self.val_accuracies.append(logs['eval_accuracy'])
            if 'eval_f1_macro' in logs:
                self.val_f1s.append(logs['eval_f1_macro'])


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, preds)
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }
    
    # Add per-class metrics
    for idx, label in enumerate(STANCE_LABELS):
        metrics[f'precision_{label}'] = precision_per_class[idx]
        metrics[f'recall_{label}'] = recall_per_class[idx]
        metrics[f'f1_{label}'] = f1_per_class[idx]
    
    return metrics


def train_final_model(model, tokenizer, train_dataset, test_dataset, output_dir, 
                      learning_rate, batch_size, num_epochs, weight_decay=0.01):
    """Train the final model with best hyperparameters from Optuna"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        save_total_limit=2,
    )
    
    metrics_callback = MetricsCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )
    
    print(f"\nTraining final model with best hyperparameters...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Epochs: {num_epochs}, Weight decay: {weight_decay}")
    
    trainer.train()
    
    return trainer, metrics_callback


# ==================== EVALUATION & RESULTS ====================

def evaluate_and_save_results(trainer, test_dataset, model_name, dataset_name, 
                               output_dir, metrics_callback):
    """Evaluate model and save comprehensive results"""
    
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    
    # Compute final metrics
    final_metrics = compute_metrics((predictions.predictions, labels))
    
    # Add training information
    final_metrics['model_name'] = model_name
    final_metrics['dataset'] = dataset_name
    final_metrics['total_training_time_minutes'] = sum(metrics_callback.epoch_times) / 60
    final_metrics['num_parameters'] = trainer.model.num_parameters()
    
    # Save metrics as JSON
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save detailed classification report
    report = classification_report(
        labels, preds, 
        target_names=STANCE_LABELS,
        digits=4
    )
    with open(results_dir / "classification_report.txt", 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(report)
    print(f"\nMacro F1: {final_metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {final_metrics['f1_weighted']:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    
    return final_metrics, preds, labels


def save_training_history(metrics_callback, output_dir):
    """Save training history as CSV"""
    
    # Prepare history data
    num_epochs = len(metrics_callback.val_losses)
    history_data = {
        'epoch': list(range(1, num_epochs + 1)),
        'val_loss': metrics_callback.val_losses,
        'val_accuracy': metrics_callback.val_accuracies,
        'val_f1_macro': metrics_callback.val_f1s,
        'epoch_time_minutes': [t/60 for t in metrics_callback.epoch_times]
    }
    
    df = pd.DataFrame(history_data)
    df.to_csv(f"{output_dir}/results/training_history.csv", index=False)
    
    return df


# ==================== VISUALIZATION ====================

def plot_training_curves(metrics_callback, model_name, dataset_name, output_dir):
    """Plot training and validation curves"""
    
    epochs = range(1, len(metrics_callback.val_losses) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(epochs, metrics_callback.val_losses, marker='o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Validation Loss', fontsize=11)
    axes[0].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, metrics_callback.val_accuracies, marker='o', 
                 linewidth=2, markersize=6, color='green')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation Accuracy', fontsize=11)
    axes[1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, metrics_callback.val_f1s, marker='o', 
                 linewidth=2, markersize=6, color='orange')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Validation F1 (Macro)', fontsize=11)
    axes[2].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - {dataset_name} - Training Curves', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: training_curves.png")


def plot_confusion_matrix(labels, preds, model_name, dataset_name, output_dir):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=STANCE_LABELS, yticklabels=STANCE_LABELS,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'{model_name} - {dataset_name}\nConfusion Matrix', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV
    cm_df = pd.DataFrame(cm, index=STANCE_LABELS, columns=STANCE_LABELS)
    cm_df.to_csv(f"{output_dir}/results/confusion_matrix.csv")
    
    print(f"✓ Saved: confusion_matrix.png and confusion_matrix.csv")


def plot_per_class_metrics(final_metrics, model_name, dataset_name, output_dir):
    """Plot per-class performance metrics"""
    
    metrics_data = {
        'Class': STANCE_LABELS,
        'Precision': [final_metrics[f'precision_{label}'] for label in STANCE_LABELS],
        'Recall': [final_metrics[f'recall_{label}'] for label in STANCE_LABELS],
        'F1-Score': [final_metrics[f'f1_{label}'] for label in STANCE_LABELS],
    }
    
    x = np.arange(len(STANCE_LABELS))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, metrics_data['Precision'], width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, metrics_data['Recall'], width, label='Recall', color='lightcoral')
    bars3 = ax.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Stance Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name} - {dataset_name}\nPer-Class Performance', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(STANCE_LABELS)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: per_class_metrics.png")


def plot_class_distribution(train_df, test_df, dataset_name, output_dir):
    """Plot class distribution in train and test sets"""
    
    # Ensure results directory exists
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (df, split) in enumerate([(train_df, 'Train'), (test_df, 'Test')]):
        counts = df['label'].value_counts().sort_index()
        labels = [STANCE_LABELS[i] for i in counts.index]
        
        bars = axes[idx].bar(labels, counts.values, color=['skyblue', 'salmon', 'lightgreen'])
        axes[idx].set_xlabel('Stance Class', fontsize=11)
        axes[idx].set_ylabel('Count', fontsize=11)
        axes[idx].set_title(f'{split} Set Distribution', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add count and percentage labels
        total = counts.sum()
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            percentage = (count / total) * 100
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{count}\n({percentage:.1f}%)',
                          ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'{dataset_name} - Class Distribution', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/results/class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: class_distribution.png")


def create_summary_table(final_metrics, output_dir):
    """Create a summary table for easy comparison"""
    
    summary = {
        'Model': [final_metrics['model_name']],
        'Dataset': [final_metrics['dataset']],
        'Accuracy': [f"{final_metrics['accuracy']:.4f}"],
        'Macro F1': [f"{final_metrics['f1_macro']:.4f}"],
        'Weighted F1': [f"{final_metrics['f1_weighted']:.4f}"],
        'Concur F1': [f"{final_metrics['f1_concur']:.4f}"],
        'Oppose F1': [f"{final_metrics['f1_oppose']:.4f}"],
        'Neutral F1': [f"{final_metrics['f1_neutral']:.4f}"],
        'Training Time (min)': [f"{final_metrics['total_training_time_minutes']:.2f}"],
        'Parameters': [final_metrics['num_parameters']],
    }
    
    df = pd.DataFrame(summary)
    df.to_csv(f"{output_dir}/results/summary_table.csv", index=False)
    
    print(f"✓ Saved: summary_table.csv")
    return df


# ==================== HYPERPARAMETER TUNING ====================

def objective(trial, model_path, tokenizer, train_dataset, test_dataset, output_dir):
    """Optuna objective function for hyperparameter tuning"""
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_epochs = trial.suggest_int('num_epochs', 3, 5)
    weight_decay = trial.suggest_float('weight_decay', 0.001, 0.05) 
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3,
        ignore_mismatched_sizes=True  # Ignore 2-class → 3-class mismatch
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/optuna_trial_{trial.number}",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_steps=100,
        load_best_model_at_end=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results['eval_f1_macro']


def hyperparameter_tuning(model_path, tokenizer, train_dataset, test_dataset, 
                          output_dir, n_trials=20):
    """Run Optuna hyperparameter tuning"""
    
    print("\n" + "="*60)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*60)
    
    study = optuna.create_study(direction='maximize', study_name='stance_detection')
    study.optimize(
        lambda trial: objective(trial, model_path, tokenizer, train_dataset, 
                               test_dataset, output_dir),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best hyperparameters
    with open(results_dir / "best_hyperparameters.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    # Save all trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(results_dir / "optuna_trials.csv", index=False)
    
    # Plot optimization history
    plot_optuna_results(study, output_dir)
    
    # Cleanup trial directories
    import shutil
    for trial_dir in Path(output_dir).glob("optuna_trial_*"):
        if trial_dir.is_dir():
            shutil.rmtree(trial_dir, ignore_errors=True)
    
    print(f"✓ Saved: hyperparameter_tuning.png")
    
    return study.best_params


def plot_optuna_results(study, output_dir):
    """Plot Optuna optimization results"""
    results_dir = Path(output_dir) / "results"
    trials_df = study.trials_dataframe()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Optimization history
    axes[0, 0].plot(trials_df['number'], trials_df['value'], marker='o')
    axes[0, 0].axhline(y=study.best_value, color='r', linestyle='--', label='Best')
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('F1 Score (Macro)')
    axes[0, 0].set_title('Optimization History')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate impact
    axes[0, 1].scatter(trials_df['params_learning_rate'], trials_df['value'], alpha=0.6)
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('F1 Score (Macro)')
    axes[0, 1].set_title('Learning Rate Impact')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weight decay impact
    axes[0, 2].scatter(trials_df['params_weight_decay'], trials_df['value'], alpha=0.6, c=trials_df['value'], cmap='viridis')
    axes[0, 2].set_xlabel('Weight Decay')
    axes[0, 2].set_ylabel('F1 Score (Macro)')
    axes[0, 2].set_title('Weight Decay Impact')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Batch size impact
    batch_sizes = trials_df['params_batch_size'].unique()
    for bs in sorted(batch_sizes):
        subset = trials_df[trials_df['params_batch_size'] == bs]
        axes[1, 0].scatter([bs] * len(subset), subset['value'], alpha=0.6, label=f'BS={bs}')
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('F1 Score (Macro)')
    axes[1, 0].set_title('Batch Size Impact')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epochs impact
    axes[1, 1].scatter(trials_df['params_num_epochs'], trials_df['value'], alpha=0.6)
    axes[1, 1].set_xlabel('Number of Epochs')
    axes[1, 1].set_ylabel('F1 Score (Macro)')
    axes[1, 1].set_title('Epochs Impact')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Hyperparameter correlation heatmap
    param_cols = ['params_learning_rate', 'params_batch_size', 'params_num_epochs', 'params_weight_decay']
    corr_data = trials_df[param_cols + ['value']].copy()
    corr_data.columns = ['LR', 'Batch', 'Epochs', 'Weight Decay', 'F1']
    correlation = corr_data.corr()
    
    im = axes[1, 2].imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_xticks(range(len(correlation.columns)))
    axes[1, 2].set_yticks(range(len(correlation.columns)))
    axes[1, 2].set_xticklabels(correlation.columns, rotation=45, ha='right')
    axes[1, 2].set_yticklabels(correlation.columns)
    axes[1, 2].set_title('Parameter Correlation')
    
    # Add correlation values as text
    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            text = axes[1, 2].text(j, i, f'{correlation.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.suptitle('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "hyperparameter_tuning.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Train stance detection model with Optuna')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['reddit_posts_and_comments', 'redditAITA', 'stance_dataset'],
                       help='Dataset to use')
    parser.add_argument('--model_path', type=str, 
                       default='./sentiment-bert-model-final',
                       help='Path to pretrained sentiment model')
    parser.add_argument('--model_name', type=str, default='Two-Stage-BERT',
                       help='Model name for identification')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-generated if not provided)')
    parser.add_argument('--n_trials', type=int, default=10,
                       help='Number of Optuna trials for hyperparameter tuning')
    parser.add_argument('--max_length', type=int, default=128)
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"./results/{args.model_name}/{args.dataset}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create results directory early for plotting
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"STANCE DETECTION TRAINING - OPTUNA AUTO TRAINING")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Optuna Trials: {args.n_trials}")
    print("="*60 + "\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Prepare datasets
    train_dataset, test_dataset, train_df, test_df = prepare_datasets(
        args.dataset, tokenizer, args.max_length
    )
    
    # Plot class distribution
    plot_class_distribution(train_df, test_df, args.dataset, args.output_dir)
    
    # Run hyperparameter tuning 
    best_params = hyperparameter_tuning(
        args.model_path, tokenizer, train_dataset, test_dataset,
        args.output_dir, args.n_trials
    )
    
    # Load model with best hyperparameters
    print(f"\nLoading model from {args.model_path} for final training...")
    print(f"Note: Ignoring classification head from Stage 1 (2 classes) and initializing for Stage 2 (3 classes)")
    
    # Load the model but ignore the classification head mismatch
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, 
        num_labels=3,
        ignore_mismatched_sizes=True  # This ignores the classifier layer mismatch
    )
    
    print(f"✓ Model loaded successfully with 3-class classification head")
    
    # Train final model with best hyperparameters
    trainer, metrics_callback = train_final_model(
        model, tokenizer, train_dataset, test_dataset, args.output_dir,
        best_params['learning_rate'], best_params['batch_size'], 
        best_params['num_epochs'], best_params['weight_decay']
    )
    
    # Evaluate and save results
    final_metrics, preds, labels = evaluate_and_save_results(
        trainer, test_dataset, args.model_name, args.dataset,
        args.output_dir, metrics_callback
    )
    
    # Save training history
    save_training_history(metrics_callback, args.output_dir)
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(metrics_callback, args.model_name, args.dataset, args.output_dir)
    plot_confusion_matrix(labels, preds, args.model_name, args.dataset, args.output_dir)
    plot_per_class_metrics(final_metrics, args.model_name, args.dataset, args.output_dir)
    
    # Create summary table
    summary_df = create_summary_table(final_metrics, args.output_dir)
    
    # Save best hyperparameters used
    hyperparams = {
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'num_epochs': best_params['num_epochs'],
        'weight_decay': best_params['weight_decay'],
        'max_length': args.max_length,
    }
    with open(f"{args.output_dir}/results/hyperparameters_used.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Save final model
    final_model_dir = f"{args.output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}/results/")
    print(f"Model saved to: {final_model_dir}")
    print(f"\nSummary:")
    print(summary_df.to_string(index=False))
    print("="*60 + "\n")


if __name__ == "__main__":
    main()