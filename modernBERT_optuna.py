import pandas as pd
import torch
import numpy as np
import evaluate  # The new Hugging Face evaluation library
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import optuna
from optuna.storages import RDBStorage
import wandb

# Define the path to your CSV file
csv_file_path = "reddit_posts_and_comments.csv" 

# Load your data from the CSV file
# This assumes your CSV has the columns: 'post_text', 'comment_text', 'stance'
try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded {len(df)} rows from {csv_file_path}")
    
    # Optional: Check if the required columns exist
    required_cols = {'post_text', 'comment_text', 'stance'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV file must contain the columns: {required_cols}")
        # Or raise an error
        exit()
        
    # Handle any missing values by dropping rows with empty text or stance
    df = df.dropna(subset=['post_text', 'comment_text', 'stance'])
    print(f"Using {len(df)} rows after dropping any missing values.")
    
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please update the 'csv_file_path' variable to point to your CSV file.")
    exit()

# --- 2. Define Model and Tokenizer ---

# Use the "ModernBERT" model
MODEL_CHECKPOINT = "answerdotai/ModernBERT-base"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# --- 3. Create Label Mappings ---

# The model needs integer IDs for your string labels
labels_list = df['stance'].unique().tolist()
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for i, label in enumerate(labels_list)}
num_labels = len(labels_list)

print(f"Label Mappings: {label2id}")

# --- 4. Preprocessing Function (The MOST Important Step) ---

def preprocess_function(examples):
    # This is the key: Tokenize the post and comment as a pair.
    # The tokenizer will automatically format this as:
    # [CLS] post_text [SEP] comment_text [SEP]
    tokenized_inputs = tokenizer(
        examples['post_text'],
        examples['comment_text'],
        truncation=True,
        padding="max_length",
        max_length=512  # You can adjust this
    )
    
    # Convert the string labels (e.g., "concurs") to integer IDs (e.g., 0)
    tokenized_inputs['labels'] = [label2id[label] for label in examples['stance']]
    
    return tokenized_inputs

dataset = Dataset.from_pandas(df)
# Apply the preprocessing to the entire dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

dataset_splits = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = dataset_splits['train']
eval_dataset = dataset_splits['test']

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# Define persistent storage
storage = RDBStorage("sqlite:///optuna_trials.db")

study = optuna.create_study(
    study_name="transformers_optuna_study",
    direction="maximize",
    storage=storage,
    load_if_exists=True
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    labels = eval_pred.label_ids
    return metric.compute(predictions=predictions, references=labels)


def compute_objective(metrics):
    return metrics["eval_accuracy"]

wandb.init(project="hf-optuna", name="transformers_optuna_study")

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
        )
    
training_args = TrainingArguments(
    output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8, 
        report_to="wandb",  # Logs to W&B
        logging_dir="./logs",
        run_name="transformers_optuna_study",
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "gradient_accumulation_steps": trial.suggest_categorical(
             "gradient_accumulation_steps", [1, 2, 4, 8]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }


best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=5,
    compute_objective=compute_objective,
    study_name="transformers_optuna_study",
    storage="sqlite:///optuna_trials.db",
    load_if_exists=True
)

print(best_run)

# Save the final model
trainer.save_model("./dialogz_stance_model/final_model")
print("Model saved to ./dialogz_stance_model/final_model")