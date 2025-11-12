"""
Inference script for trained Qwen stance detection model
"""

import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class StancePredictor:
    def __init__(self, model_path="/opt/tiger/MLLM_AUTO_EVALUATE_PIPELINE/EE6405_Final_Project/results_qwen/run-4/checkpoint-1800"):
        """
        Initialize the stance predictor.
        
        Args:
            model_path: Path to the trained model directory
        """
        print("=" * 80)
        print("Qwen Stance Detection - Inference")
        print("=" * 80)
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        print(f"\nLoading model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Load label mappings
        label_mappings_path = self.model_path / "label_mappings.json"
        
        if label_mappings_path.exists():
            # Load from label_mappings.json if it exists
            with open(label_mappings_path) as f:
                mappings = json.load(f)
                self.label2id = mappings['label2id']
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
        else:
            # Fallback: Extract from model config
            print("\n⚠ Warning: label_mappings.json not found. Extracting from config...")
            if hasattr(self.model.config, 'label2id') and hasattr(self.model.config, 'id2label'):
                self.label2id = self.model.config.label2id
                self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
                print(f"✓ Extracted label mappings from config: {list(self.label2id.keys())}")
                
                # Optionally save it for next time
                try:
                    with open(label_mappings_path, 'w') as f:
                        json.dump({
                            'label2id': self.label2id,
                            'id2label': self.id2label
                        }, f, indent=2)
                    print(f"✓ Saved label mappings to: {label_mappings_path}")
                except Exception as e:
                    print(f"  (Could not save label mappings: {e})")
            else:
                raise ValueError(
                    "No label mappings found! Please either:\n"
                    "  1. Run: python extract_label_mappings.py <checkpoint_path>\n"
                    "  2. Use the final model instead of a checkpoint\n"
                    "  3. Wait for training to complete"
                )
        
        print(f"✓ Model loaded successfully")
        print(f"  Labels: {list(self.label2id.keys())}")
        print(f"  Device: {next(self.model.parameters()).device}")
    
    def create_prompt(self, post, comment):
        """Create instruction prompt for the model."""
        # Truncate long texts
        post_text = str(post)[:500] if len(str(post)) > 500 else str(post)
        comment_text = str(comment)[:300] if len(str(comment)) > 300 else str(comment)
        
        prompt = f"""Given the following Reddit post and comment, classify the stance of the comment.

Post: {post_text}

Comment: {comment_text}

Stance:"""
        return prompt
    
    def predict(self, post, comment, return_probabilities=False):
        """
        Predict stance for a single post-comment pair.
        
        Args:
            post: Post text
            comment: Comment text
            return_probabilities: If True, return probabilities for all classes
            
        Returns:
            If return_probabilities=False: predicted label (str)
            If return_probabilities=True: dict with label and probabilities
        """
        # Create prompt
        prompt = self.create_prompt(post, comment)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction_id = logits.argmax(dim=-1).item()
        
        predicted_label = self.id2label[prediction_id]
        
        if return_probabilities:
            probs_dict = {
                self.id2label[i]: probabilities[0, i].item()
                for i in range(len(self.id2label))
            }
            return {
                'label': predicted_label,
                'probabilities': probs_dict,
                'confidence': probabilities[0, prediction_id].item()
            }
        else:
            return predicted_label
    
    def predict_batch(self, posts, comments, batch_size=8, show_progress=True):
        """
        Predict stance for multiple post-comment pairs.
        
        Args:
            posts: List of post texts
            comments: List of comment texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of predicted labels
        """
        predictions = []
        
        iterator = range(0, len(posts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting", total=len(posts)//batch_size + 1)
        
        for i in iterator:
            batch_posts = posts[i:i+batch_size]
            batch_comments = comments[i:i+batch_size]
            
            # Create prompts
            prompts = [
                self.create_prompt(post, comment)
                for post, comment in zip(batch_posts, batch_comments)
            ]
            
            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_predictions = logits.argmax(dim=-1).cpu().tolist()
            
            predictions.extend([self.id2label[pred] for pred in batch_predictions])
        
        return predictions
    
    def predict_from_csv(self, csv_path, output_path=None, batch_size=8):
        """
        Predict stance for all rows in a CSV file.
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path to save results (optional)
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with predictions
        """
        print(f"\n{'='*80}")
        print(f"Processing CSV: {csv_path}")
        print(f"{'='*80}")
        
        # Load CSV
        df = pd.read_csv(
            csv_path,
            on_bad_lines='skip',
            engine='python',
            encoding='utf-8'
        )
        print(f"Loaded {len(df)} rows")
        
        # Check columns
        if 'post_text' not in df.columns or 'comment_text' not in df.columns:
            raise ValueError("CSV must contain 'post_text' and 'comment_text' columns")
        
        # Make predictions
        print("Making predictions...")
        predictions = self.predict_batch(
            df['post_text'].tolist(),
            df['comment_text'].tolist(),
            batch_size=batch_size
        )
        
        # Add predictions to dataframe
        df['predicted_stance'] = predictions
        
        # If ground truth exists, compute accuracy
        if 'stance' in df.columns:
            accuracy = (df['stance'] == df['predicted_stance']).mean()
            print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Show confusion info
            print("\nPrediction distribution:")
            print(df['predicted_stance'].value_counts())
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to: {output_path}")
        
        return df
    
    def evaluate(self, csv_path, batch_size=8, save_results=True, output_dir="./evaluation_results"):
        """
        Comprehensive evaluation on test data with detailed metrics.
        
        Args:
            csv_path: Path to test CSV file
            batch_size: Batch size for processing
            save_results: Whether to save results and plots
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*80}")
        print(f"Comprehensive Evaluation")
        print(f"{'='*80}")
        print(f"\nTest data: {csv_path}")
        
        # Create output directory
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load test data with flexible column names
        df = pd.read_csv(
            csv_path,
            on_bad_lines='skip',
            engine='python',
            encoding='utf-8'
        )
        print(f"Loaded {len(df)} test samples")
        
        # Check for column names (handle both formats)
        if 'post_text' in df.columns and 'comment_text' in df.columns:
            post_col, comment_col = 'post_text', 'comment_text'
        elif 'post' in df.columns and 'comment' in df.columns:
            post_col, comment_col = 'post', 'comment'
        else:
            raise ValueError(f"CSV must contain either (post_text, comment_text) or (post, comment) columns. Found: {df.columns.tolist()}")
        
        if 'stance' not in df.columns:
            raise ValueError("CSV must contain 'stance' column for evaluation")
        
        # Clean data
        df = df.dropna(subset=[post_col, comment_col, 'stance'])
        print(f"Using {len(df)} samples after dropping NaN values")
        
        # Show label distribution
        print("\nTrue label distribution:")
        print(df['stance'].value_counts())
        
        # Make predictions
        print(f"\n{'='*80}")
        print("Making predictions...")
        print(f"{'='*80}")
        
        predictions = self.predict_batch(
            df[post_col].tolist(),
            df[comment_col].tolist(),
            batch_size=batch_size,
            show_progress=True
        )
        
        # Get true labels
        true_labels = df['stance'].tolist()
        
        # Compute metrics
        print(f"\n{'='*80}")
        print("Computing Metrics")
        print(f"{'='*80}")
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, predictions)
        print(f"\n✓ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predictions,
            average=None,
            labels=list(self.label2id.keys())
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Print per-class metrics
        print("\nPer-Class Metrics:")
        print(f"{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 65)
        for i, label in enumerate(self.label2id.keys()):
            print(f"{label:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
        
        print("\nAveraged Metrics:")
        print(f"{'Metric':<20} {'Macro Avg':<15} {'Weighted Avg':<15}")
        print("-" * 50)
        print(f"{'Precision':<20} {precision_macro:<15.4f} {precision_weighted:<15.4f}")
        print(f"{'Recall':<20} {recall_macro:<15.4f} {recall_weighted:<15.4f}")
        print(f"{'F1-Score':<20} {f1_macro:<15.4f} {f1_weighted:<15.4f}")
        
        # Classification report
        print(f"\n{'='*80}")
        print("Detailed Classification Report")
        print(f"{'='*80}\n")
        print(classification_report(
            true_labels,
            predictions,
            labels=list(self.label2id.keys()),
            target_names=list(self.label2id.keys()),
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(
            true_labels,
            predictions,
            labels=list(self.label2id.keys())
        )
        
        print("\nConfusion Matrix:")
        print(f"{'':>15}", end='')
        for label in self.label2id.keys():
            print(f"{label:<15}", end='')
        print()
        for i, label in enumerate(self.label2id.keys()):
            print(f"{label:>15}", end='')
            for j in range(len(self.label2id)):
                print(f"{cm[i, j]:<15}", end='')
            print()
        
        # Save confusion matrix plot
        if save_results:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=list(self.label2id.keys()),
                yticklabels=list(self.label2id.keys())
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_path = Path(output_dir) / 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Confusion matrix saved to: {cm_path}")
        
        # Save detailed results
        if save_results:
            df['predicted_stance'] = predictions
            df['correct'] = df['stance'] == df['predicted_stance']
            
            results_path = Path(output_dir) / 'predictions.csv'
            df.to_csv(results_path, index=False)
            print(f"✓ Predictions saved to: {results_path}")
            
            # Save metrics to JSON
            metrics_dict = {
                'accuracy': float(accuracy),
                'macro_avg': {
                    'precision': float(precision_macro),
                    'recall': float(recall_macro),
                    'f1': float(f1_macro)
                },
                'weighted_avg': {
                    'precision': float(precision_weighted),
                    'recall': float(recall_weighted),
                    'f1': float(f1_weighted)
                },
                'per_class': {
                    label: {
                        'precision': float(precision[i]),
                        'recall': float(recall[i]),
                        'f1': float(f1[i]),
                        'support': int(support[i])
                    }
                    for i, label in enumerate(self.label2id.keys())
                },
                'confusion_matrix': cm.tolist()
            }
            
            metrics_path = Path(output_dir) / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"✓ Metrics saved to: {metrics_path}")
        
        print(f"\n{'='*80}")
        print("Evaluation Complete!")
        print(f"{'='*80}")
        
        return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Qwen Stance Detection Inference & Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./qwen_stance_model/final_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation on test data"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="/Users/bytedance/Downloads/qhuy/EE6405_Final_Project/data/preprocessed/redditAITA_test.csv",
        help="Path to test CSV file (for --test mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file for batch prediction"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (for CSV mode)"
    )
    parser.add_argument(
        "--post",
        type=str,
        help="Post text for single prediction"
    )
    parser.add_argument(
        "--comment",
        type=str,
        help="Comment text for single prediction"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for CSV processing"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = StancePredictor(args.model_path)
    
    # Test/Evaluation mode
    if args.test:
        predictor.evaluate(
            csv_path=args.test_data,
            batch_size=args.batch_size,
            save_results=True,
            output_dir=args.output_dir
        )
    
    # CSV mode
    elif args.csv:
        output_path = args.output or args.csv.replace('.csv', '_predictions.csv')
        results_df = predictor.predict_from_csv(
            args.csv,
            output_path,
            batch_size=args.batch_size
        )
        
        print("\nSample predictions:")
        print(results_df[['post_text', 'comment_text', 'predicted_stance']].head())
    
    # Single prediction mode
    elif args.post and args.comment:
        print(f"\n{'='*80}")
        print("Single Prediction")
        print(f"{'='*80}")
        
        result = predictor.predict(
            args.post,
            args.comment,
            return_probabilities=True
        )
        
        print(f"\nPost: {args.post[:100]}...")
        print(f"Comment: {args.comment[:100]}...")
        print(f"\nPredicted Stance: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
    
    # Interactive mode
    else:
        print(f"\n{'='*80}")
        print("Interactive Mode")
        print(f"{'='*80}")
        print("Enter post and comment to get stance predictions.")
        print("Type 'quit' to exit.\n")
        
        while True:
            post = input("Post: ").strip()
            if post.lower() == 'quit':
                break
            
            comment = input("Comment: ").strip()
            if comment.lower() == 'quit':
                break
            
            if post and comment:
                result = predictor.predict(
                    post,
                    comment,
                    return_probabilities=True
                )
                
                print(f"\n→ Predicted Stance: {result['label']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print("  Probabilities:")
                for label, prob in result['probabilities'].items():
                    print(f"    {label}: {prob:.4f}")
                print()
            else:
                print("Please provide both post and comment.\n")
        
        print("Goodbye!")


if __name__ == "__main__":
    main()

