"""
Stance Detection Training with Qwen2.5-3B-Instruct
Uses Optuna for hyperparameter optimization and W&B for logging
Supports CLI arguments for flexible training on different datasets
"""

import pandas as pd
import torch
import numpy as np
import evaluate
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import optuna
from optuna.storages import RDBStorage
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

print("=" * 80)
print("Qwen2.5-3B-Instruct Stance Detection Training")
print("=" * 80)

# --- 0. Parse Command Line Arguments ---
parser = argparse.ArgumentParser(
    description="Train Qwen2.5-3B-Instruct for stance detection",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Train on Reddit data
  python train_qwen_stance.py --csv ./data/preprocessed/reddit_preprocessed_for_qwen.csv --output ./results/reddit_model

  # Train on Stance dataset
  python train_qwen_stance.py --csv ./data/preprocessed/stance_preprocessed_for_qwen.csv --output ./results/stance_model

  # Train on combined data with custom hyperparameters
  python train_qwen_stance.py --csv ./data/preprocessed/combined_preprocessed_for_qwen.csv --epochs 5 --batch-size 16 --lr 2e-5
    """
)

parser.add_argument(
    "--csv",
    type=str,
    default=r"D:\Quang Huy\Documents\EE6405\Project\EE6405_Final_Project\data\preprocessed\combined_preprocessed_for_qwen.csv",
    help="Path to CSV file for training (default: combined preprocessed data)"
)
parser.add_argument(
    "--output",
    type=str,
    default="./qwen_stance_model",
    help="Output directory for trained model (default: ./qwen_stance_model)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs (default: 3)"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help="Training batch size (default: 8)"
)
parser.add_argument(
    "--lr",
    type=float,
    default=2e-5,
    help="Learning rate (default: 2e-5)"
)
parser.add_argument(
    "--max-rows",
    type=int,
    default=0,
    help="Max rows to use from CSV (0 = all, default: 0)"
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="Qwen/Qwen2.5-1.5B-Instruct",
    help="Model name or path (default: Qwen/Qwen2.5-3B-Instruct). Examples: Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct"
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Enable mixed precision training (FP16) to reduce memory usage"
)

parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    help="Enable gradient checkpointing to save GPU memory at the cost of extra computation"
)

args = parser.parse_args()

# --- 1. Load Data with Error Handling ---
csv_file_path = args.csv

print(f"\nLoading data from: {csv_file_path}")

try:
    # Try loading with error handling for malformed CSV
    df = pd.read_csv(
        csv_file_path,
        on_bad_lines='skip',
        engine='python',
        encoding='utf-8'
    )
    print(f"✓ Successfully loaded {len(df)} rows from {csv_file_path}")
    
    # Check if the required columns exist
    required_cols = {'post_text', 'comment_text', 'stance'}
    if not required_cols.issubset(df.columns):
        print(f"✗ Error: CSV file must contain the columns: {required_cols}")
        print(f"  Available columns: {df.columns.tolist()}")
        
        # Try to infer columns if there are at least 3
        if len(df.columns) >= 3:
            print(f"\n  Attempting to use first 3 columns as [post_text, comment_text, stance]")
            df.columns = ['post_text', 'comment_text', 'stance'] + list(df.columns[3:])
        else:
            exit(1)
    
    # Handle any missing values
    print(f"\nRows before cleaning: {len(df)}")
    df = df.dropna(subset=['post_text', 'comment_text', 'stance'])
    print(f"Rows after cleaning: {len(df)}")
    
    # Limit rows if specified
    if args.max_rows > 0:
        df = df[:args.max_rows]
        print(f"Using first {args.max_rows} rows")
    
    # Show stance distribution
    print("\nStance distribution:")
    print(df['stance'].value_counts())
    
except FileNotFoundError:
    print(f"✗ Error: The file '{csv_file_path}' was not found.")
    print("  Please update the 'csv_file_path' variable to point to your CSV file.")
    exit(1)
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    exit(1)

# --- 2. Define Model and Tokenizer ---
MODEL_CHECKPOINT = args.model_name_or_path

print(f"\n{'='*80}")
print(f"Loading Model: {MODEL_CHECKPOINT}")
print(f"{'='*80}")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT,
    trust_remote_code=True,
    padding_side='right'  # Important for causal LMs
)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

# --- 3. Create Label Mappings ---
labels_list = sorted(df['stance'].unique().tolist())
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for i, label in enumerate(labels_list)}
num_labels = len(labels_list)

print(f"\nNumber of labels: {num_labels}")
print(f"Label mappings: {label2id}")

# --- 4. Preprocessing Function ---
def preprocess_function(examples):
    """
    Format input as instruction for Qwen model.
    Uses a prompt format suitable for stance detection.
    """
    # Create instruction-style prompts
    prompts = []
    for post, comment in zip(examples['post_text'], examples['comment_text']):
        # Truncate long texts
        post_text = str(post)[:500] if len(str(post)) > 500 else str(post)
        comment_text = str(comment)[:300] if len(str(comment)) > 300 else str(comment)
        
        prompt = f"""Given the following Reddit post and comment, classify the stance of the comment.

Post: {post_text}

Comment: {comment_text}

Stance:"""
        prompts.append(prompt)
    
    # Tokenize
    tokenized_inputs = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    
    # Convert string labels to integer IDs
    tokenized_inputs['labels'] = [label2id[label] for label in examples['stance']]
    
    return tokenized_inputs

print(f"\n{'='*80}")
print("Preprocessing Dataset")
print(f"{'='*80}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
print(f"\nDataset size: {len(dataset)}")

# Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset"
)

# Split into train/eval
dataset_splits = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_splits['train']
eval_dataset = dataset_splits['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# --- 5. Setup Metrics ---
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Compute accuracy and F1 scores."""
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = predictions.argmax(axis=-1)
    labels = eval_pred.label_ids
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1_macro': f1_macro['f1'],
        'f1_weighted': f1_weighted['f1']
    }

def compute_objective(metrics):
    """Objective function for Optuna - we want to maximize accuracy."""
    return metrics["eval_accuracy"]

# --- 6. Setup Optuna and W&B ---
print(f"\n{'='*80}")
print("Setting up Optuna and Weights & Biases")
print(f"{'='*80}")

# Define persistent storage for Optuna
storage = RDBStorage("sqlite:///optuna_qwen_trials.db")

# Create or load study
study = optuna.create_study(
    study_name="qwen_stance_detection",
    direction="maximize",
    storage=storage,
    load_if_exists=True
)

# --- 7. Model Initialization Function ---
def model_init(trial=None):
    """Initialize model for each trial."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        trust_remote_code=True
    )
    
    # Important: Set pad_token_id for the model
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model

# --- 8. Training Arguments ---
output_dir = args.output
Path(output_dir).mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    logging_strategy="epoch",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size // 2,  # Reduce for 3B model memory
    per_device_eval_batch_size=args.batch_size // 2,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    learning_rate=args.lr,
    fp16=True,  # Enable mixed precision training
    report_to=None,
    logging_dir=f"{output_dir}/logs",
    run_name="qwen2.5-3b-stance-detection",
    save_total_limit=2,  # Only keep 2 best checkpoints
    push_to_hub=False,
)

print(f"\nTraining Configuration:")
print(f"  Model: {args.model_name_or_path}")
print(f"  CSV: {csv_file_path}")
print(f"  Output Dir: {output_dir}")
print(f"  Epochs: {args.epochs}")
print(f"  Batch Size: {args.batch_size}")
print(f"  Learning Rate: {args.lr}")

# --- 9. Data Collator ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 10. Initialize Trainer ---
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 11. Hyperparameter Search Space ---
def optuna_hp_space(trial):
    """Define hyperparameter search space for Optuna."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000, step=100),
    }

# --- 12. Run Hyperparameter Search ---
print(f"\n{'='*80}")
print("Starting Hyperparameter Search")
print(f"{'='*80}")

try:
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=5,
        compute_objective=compute_objective,
        study_name="qwen_stance_detection",
        storage="sqlite:///optuna_qwen_trials.db",
        load_if_exists=True
    )
    
    print(f"\n{'='*80}")
    print("Best Hyperparameters Found:")
    print(f"{'='*80}")
    print(best_run)
    
except Exception as e:
    print(f"\n✗ Error during hyperparameter search: {e}")
    print("  Continuing with default hyperparameters...")

# --- 13. Final Training with Best Hyperparameters ---
print(f"\n{'='*80}")
print("Training Final Model")
print(f"{'='*80}")

try:
    # Train the model
    trainer.train()
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print("Final Evaluation")
    print(f"{'='*80}")
    
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Get predictions for detailed metrics
    predictions = trainer.predict(eval_dataset)
    pred_labels = predictions.predictions.argmax(axis=-1)
    true_labels = predictions.label_ids
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=[id2label[i] for i in range(num_labels)]
    ))
    
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()

# --- 14. Save the Final Model ---
print(f"\n{'='*80}")
print("Saving Model")
print(f"{'='*80}")

try:
    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"✓ Model saved to {final_model_dir}")
    print(f"✓ Tokenizer saved to {final_model_dir}")
    
    # Save label mappings
    import json
    with open(f"{final_model_dir}/label_mappings.json", 'w') as f:
        json.dump({
            'label2id': label2id,
            'id2label': id2label
        }, f, indent=2)
    print(f"✓ Label mappings saved to {final_model_dir}/label_mappings.json")
    
except Exception as e:
    print(f"✗ Error saving model: {e}")

print(f"\n{'='*80}")
print("Training Complete!")
print(f"{'='*80}")