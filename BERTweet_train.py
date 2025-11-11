"""
BERTweet Stance Detection Training Script with Optuna Hyperparameter Tuning
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
    classification_report, confusion_matrix
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
    """Load and prepare dataset"""
    
    file_path = f"{data_dir}/{dataset_name}_{split}.csv"
    df = pd.read_csv(file_path)
    
    # Standardize columns based on dataset
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
    
    # Filter valid labels
    initial_count = len(df)
    df = df[df['label'].isin(STANCE_LABELS)]
    filtered = initial_count - len(df)
    if filtered > 0:
        print(f" Filtered {filtered} rows with invalid labels")
    
    # Map to indices
    df['label'] = df['label'].map(LABEL_MAP)
    
    # Clean data
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != '']
    df = df.dropna(subset=['label'])
    
    return df[['text', 'label']].reset_index(drop=True)


def prepare_datasets(dataset_name, tokenizer, max_length=128):
    """Load and tokenize datasets"""
    
    print(f"\nLoading {dataset_name}...")
    train_df = load_dataset(dataset_name, split="train")
    test_df = load_dataset(dataset_name, split="test")
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
    
    print(f"Tokenization complete")
    
    return train_dataset, test_dataset, train_df, test_df


# ==================== TRAINING ====================

class MetricsCallback(TrainerCallback):
    """Track metrics during training"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = [] 
        self.val_f1s = []
        self.epoch_times = []
        self.start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\nEpoch {int(state.epoch)}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        self.epoch_times.append(elapsed)
        print(f"  Completed in {elapsed/60:.2f} minutes")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.val_losses.append(logs['eval_loss'])
            if 'eval_accuracy' in logs:  # Added
                self.val_accuracies.append(logs['eval_accuracy'])
            if 'eval_f1_macro' in logs:
                self.val_f1s.append(logs['eval_f1_macro'])


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, preds)
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
    
    for idx, label in enumerate(STANCE_LABELS):
        metrics[f'precision_{label}'] = precision_per_class[idx]
        metrics[f'recall_{label}'] = recall_per_class[idx]
        metrics[f'f1_{label}'] = f1_per_class[idx]
    
    return metrics


def train_final_model(model, tokenizer, train_dataset, test_dataset, output_dir,
                      learning_rate, batch_size, num_epochs):
    """Train the final model with best hyperparameters from Optuna"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        save_total_limit=2,
        use_mps_device=(device == "mps"),
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
    print(f"  LR: {learning_rate}, Batch: {batch_size}, Epochs: {num_epochs}")
    
    trainer.train()
    
    return trainer, metrics_callback


# ==================== EVALUATION & RESULTS ====================

def save_results(trainer, test_dataset, model_name, dataset_name, output_dir, 
                 metrics_callback):
    """Evaluate and save results"""
    
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    
    # Compute metrics
    final_metrics = compute_metrics((predictions.predictions, labels))
    final_metrics['model_name'] = model_name
    final_metrics['dataset'] = dataset_name
    final_metrics['training_time_minutes'] = sum(metrics_callback.epoch_times) / 60
    final_metrics['num_parameters'] = trainer.model.num_parameters()
    
    # Save metrics
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save classification report
    report = classification_report(labels, preds, target_names=STANCE_LABELS, digits=4)
    with open(results_dir / "classification_report.txt", 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(report)
    print(f"Macro F1: {final_metrics['f1_macro']:.4f}")
    print(f"Training time: {final_metrics['training_time_minutes']:.2f} min")
    
    # Save summary table
    summary = {
        'Model': [model_name],
        'Dataset': [dataset_name],
        'Accuracy': [f"{final_metrics['accuracy']:.4f}"],
        'Macro F1': [f"{final_metrics['f1_macro']:.4f}"],
        'Weighted F1': [f"{final_metrics['f1_weighted']:.4f}"],
        'Concur F1': [f"{final_metrics['f1_concur']:.4f}"],
        'Oppose F1': [f"{final_metrics['f1_oppose']:.4f}"],
        'Neutral F1': [f"{final_metrics['f1_neutral']:.4f}"],
        'Training Time (min)': [f"{final_metrics['training_time_minutes']:.2f}"],
    }
    pd.DataFrame(summary).to_csv(results_dir / "summary_table.csv", index=False)
    
    return final_metrics, preds, labels


# ==================== VISUALIZATION ====================

def plot_training_curves(metrics_callback, model_name, dataset_name, output_dir):
    """Plot training curves"""
    
    results_dir = Path(output_dir) / "results"
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
    
    # F1
    axes[2].plot(epochs, metrics_callback.val_f1s, marker='o', 
                 linewidth=2, markersize=6, color='orange')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Validation F1 (Macro)', fontsize=11)
    axes[2].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - {dataset_name} - Training Curves', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: training_curves.png")


def plot_confusion_matrix(labels, preds, model_name, dataset_name, output_dir):
    """Plot confusion matrix"""
    
    results_dir = Path(output_dir) / "results"
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=STANCE_LABELS, yticklabels=STANCE_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - {dataset_name}\nConfusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    pd.DataFrame(cm, index=STANCE_LABELS, columns=STANCE_LABELS).to_csv(
        results_dir / "confusion_matrix.csv"
    )
    
    print(f"✓ Saved: confusion_matrix.png")


def plot_per_class_metrics(final_metrics, model_name, dataset_name, output_dir):
    """Plot per-class metrics"""
    
    results_dir = Path(output_dir) / "results"
    
    metrics_data = {
        'Precision': [final_metrics[f'precision_{label}'] for label in STANCE_LABELS],
        'Recall': [final_metrics[f'recall_{label}'] for label in STANCE_LABELS],
        'F1': [final_metrics[f'f1_{label}'] for label in STANCE_LABELS],
    }
    
    x = np.arange(len(STANCE_LABELS))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, metrics_data['Precision'], width, label='Precision', color='skyblue')
    ax.bar(x, metrics_data['Recall'], width, label='Recall', color='lightcoral')
    ax.bar(x + width, metrics_data['F1'], width, label='F1', color='lightgreen')
    
    ax.set_xlabel('Stance')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} - {dataset_name}\nPer-Class Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(STANCE_LABELS)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: per_class_metrics.png")


def plot_class_distribution(train_df, test_df, dataset_name, output_dir):
    """Plot class distribution in train and test sets"""
    
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
    plt.savefig(results_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: class_distribution.png")


def save_training_history(metrics_callback, output_dir):
    """Save training history as CSV"""
    
    results_dir = Path(output_dir) / "results"
    
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
    df.to_csv(results_dir / "training_history.csv", index=False)
    
    print(f"✓ Saved: training_history.csv")
    
    return df


# ==================== HYPERPARAMETER TUNING ====================

def objective(trial, model_name, tokenizer, train_dataset, test_dataset, output_dir):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    epochs = trial.suggest_int('num_epochs', 3, 5)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/optuna_trial_{trial.number}",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=100,
        use_mps_device=(device == "mps"),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results['eval_f1_macro']


def hyperparameter_tuning(model_name, tokenizer, train_dataset, test_dataset, 
                          output_dir, n_trials=10):
    """Run Optuna hyperparameter tuning"""
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, model_name, tokenizer, train_dataset, 
                               test_dataset, output_dir),
        n_trials=n_trials
    )
    
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "best_hyperparameters.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    study.trials_dataframe().to_csv(results_dir / "optuna_trials.csv", index=False)
    
    # Plot optimization results (detailed view)
    plot_optuna_results(study, output_dir)
    
    # Cleanup
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
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
    
    plt.suptitle('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / "hyperparameter_tuning.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Train BERTweet for stance detection with Optuna')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['reddit_posts_and_comments', 'redditAITA', 'stance_dataset'])
    parser.add_argument('--model_name', type=str, default='BERTweet',
                       help='Model identifier')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_trials', type=int, default=10,
                       help='Number of Optuna trials for hyperparameter tuning')
    parser.add_argument('--max_length', type=int, default=128)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"./results/{args.model_name}/{args.dataset}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    Path(args.output_dir).joinpath("results").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"BERTWEET STANCE DETECTION - OPTUNA AUTO TRAINING")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Optuna Trials: {args.n_trials}")
    print("="*60)
    
    # Load BERTweet tokenizer
    print("\nLoading BERTweet tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/bertweet-base",
        use_fast=False,
        normalization=True
    )
    
    # Prepare datasets
    train_dataset, test_dataset, train_df, test_df = prepare_datasets(
        args.dataset, tokenizer, args.max_length
    )
    
    # Plot class distribution
    plot_class_distribution(train_df, test_df, args.dataset, args.output_dir)
    
    # Run hyperparameter tuning
    best_params = hyperparameter_tuning(
        "vinai/bertweet-base", tokenizer, train_dataset, test_dataset,
        args.output_dir, args.n_trials
    )
    
    # Load model with best hyperparameters
    print(f"\nLoading BERTweet model for final training...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/bertweet-base",
        num_labels=3
    )
    print(f"Model loaded ({model.num_parameters():,} parameters)")
    
    # Train final model with best hyperparameters
    trainer, metrics_callback = train_final_model(
        model, tokenizer, train_dataset, test_dataset, args.output_dir,
        best_params['learning_rate'], best_params['batch_size'], 
        best_params['num_epochs']
    )
    
    # Evaluate and save
    final_metrics, preds, labels = save_results(
        trainer, test_dataset, args.model_name, args.dataset,
        args.output_dir, metrics_callback
    )
    
    # Save training history
    save_training_history(metrics_callback, args.output_dir)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(metrics_callback, args.model_name, args.dataset, args.output_dir)
    plot_confusion_matrix(labels, preds, args.model_name, args.dataset, args.output_dir)
    plot_per_class_metrics(final_metrics, args.model_name, args.dataset, args.output_dir)
    
    # Save hyperparameters used
    hyperparams = {
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'num_epochs': best_params['num_epochs'],
        'max_length': args.max_length,
        'weight_decay': 0.01,
    }
    results_dir = Path(args.output_dir) / "results"
    with open(results_dir / "hyperparameters_used.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print(f"✓ Saved: hyperparameters_used.json")
    
    # Save model
    final_model_dir = Path(args.output_dir) / "final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Results: {args.output_dir}/results/")
    print(f"Model: {final_model_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()