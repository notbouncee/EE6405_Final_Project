from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import time

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: CUDA (GPU) - {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = "mps"
    print(f"Using device: MPS (Apple Silicon GPU)")
else:
    device = "cpu"
    print(f"Using device: CPU")
    
class SpeedCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n Starting Epoch {state.epoch}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        print(f" Epoch completed in {elapsed/60:.2f} minutes")

# Load the SST-2 dataset
print("Loading dataset...")
dataset = load_dataset("glue", "sst2")  

# Preprocess the data
print("Tokenizing dataset...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"], 
        padding="max_length", 
        truncation=True,
        max_length=128  
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./sentiment-bert-checkpoints", 
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,  
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs',
    logging_steps=500,
    save_total_limit=2,
    use_mps_device=(device == "mps"),
)

# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    callbacks=[SpeedCallback()],
)

# Train the model
print("Starting training...")
print(f"Training on {len(tokenized_datasets['train'])} samples")
print(f"Validating on {len(tokenized_datasets['validation'])} samples")
trainer.train()

# Evaluate the model
print("Evaluating the model...")
eval_results = trainer.evaluate(tokenized_datasets["validation"])
print(f"\nEvaluation results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# Save the model
print("Saving the model...")
model.save_pretrained("./sentiment-bert-model-final")
tokenizer.save_pretrained("./sentiment-bert-model-final")

print("\nTraining complete. Model saved to './sentiment-bert-model-final'")