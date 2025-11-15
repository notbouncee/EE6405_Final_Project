"""
Inference script for trained Qwen stance detection model
Updated version with flexible arguments for model path and CSV input
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
    def __init__(self, model_path):
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
                'confidence': probabilities[0, prediction_id].item(),
                'probabilities': probs_dict
            }
        else:
            return predicted_label
    
    def predict_from_csv(self, csv_path, output_path=None, batch_size=8):
        """
        Predict stance for all rows in a CSV file.
        
        Args:
            csv_path: Path to CSV file with 'post_text' and 'comment_text' columns
            output_path: Where to save predictions (optional)
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with predictions
        """
        print(f"\n{'='*80}")
        print("Batch Prediction from CSV")
        print(f"{'='*80}")
        
        # Load CSV
        print(f"\nLoading CSV from: {csv_path}")
        df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
        print(f"✓ Loaded {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Check for required columns
        if 'post_text' not in df.columns or 'comment_text' not in df.columns:
            raise ValueError("CSV must contain 'post_text' and 'comment_text' columns")
        
        # Predict
        predictions = []
        print("\nGenerating predictions...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                pred = self.predict(
                    row['post_text'],
                    row['comment_text'],
                    return_probabilities=True
                )
                predictions.append({
                    'post_text': row['post_text'],
                    'comment_text': row['comment_text'],
                    'true_stance': row.get('stance', 'N/A'),
                    'predicted_stance': pred['label'],
                    'confidence': pred['confidence']
                })
            except Exception as e:
                print(f"\nError on row {idx}: {e}")
                predictions.append({
                    'post_text': row['post_text'],
                    'comment_text': row['comment_text'],
                    'true_stance': row.get('stance', 'N/A'),
                    'predicted_stance': 'ERROR',
                    'confidence': 0.0
                })
        
        results_df = pd.DataFrame(predictions)
        
        # Save if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\n✓ Predictions saved to: {output_path}")
        
        print(f"\n{'='*80}")
        print("Sample Predictions:")
        print(f"{'='*80}")
        print(results_df[['post_text', 'comment_text', 'true_stance', 'predicted_stance']].head(10))
        
        return results_df
    
    def evaluate(self, csv_path, batch_size=8, save_results=True, output_dir="./evaluation_results"):
        """
        Evaluate model on a test dataset with metrics.
        
        Args:
            csv_path: Path to test CSV
            batch_size: Batch size
            save_results: Whether to save results
            output_dir: Directory to save results
        """
        print(f"\n{'='*80}")
        print("Model Evaluation")
        print(f"{'='*80}")
        
        # Load and predict
        results_df = self.predict_from_csv(csv_path)
        
        # Ensure ground truth labels exist
        if 'stance' not in results_df.columns or results_df['true_stance'].eq('N/A').all():
            print("\n⚠ Warning: No ground truth labels found. Skipping metrics calculation.")
            return
        
        # Calculate metrics
        print(f"\n{'='*80}")
        print("Evaluation Metrics")
        print(f"{'='*80}")
        
        y_true = results_df['true_stance'].values
        y_pred = results_df['predicted_stance'].values
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(self.label2id.keys())
        )
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        print(f"\nMacro Average:")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall: {recall_macro:.4f}")
        print(f"  F1: {f1_macro:.4f}")
        
        print(f"\nWeighted Average:")
        print(f"  Precision: {precision_weighted:.4f}")
        print(f"  Recall: {recall_weighted:.4f}")
        print(f"  F1: {f1_weighted:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, label in enumerate(self.label2id.keys()):
            print(f"\n  {label}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1: {f1[i]:.4f}")
            print(f"    Support: {support[i]}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(self.label2id.keys()))
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Save confusion matrix plot
        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
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
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            results_df['correct'] = results_df['true_stance'] == results_df['predicted_stance']
            
            results_path = Path(output_dir) / 'predictions.csv'
            results_df.to_csv(results_path, index=False)
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
    parser = argparse.ArgumentParser(
        description="Qwen Stance Detection Inference with flexible model and CSV support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on CSV with trained model
  python qwen_inference_v2.py --model-path ./results/checkpoint-1800 --csv ./data/preprocessed/reddit_preprocessed_for_qwen.csv

  # Batch prediction with custom output
  python qwen_inference_v2.py --model-path ./results/checkpoint-1800 --csv ./data/preprocessed/stance_preprocessed_for_qwen.csv --output results_predictions.csv

  # Evaluate on test data
  python qwen_inference_v2.py --model-path ./results/checkpoint-1800 --csv ./data/test.csv --eval --output-dir ./eval_results

  # Single prediction
  python qwen_inference_v2.py --model-path ./results/checkpoint-1800 --post "Your post here" --comment "Your comment here"
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (required)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with 'post_text' and 'comment_text' columns for batch prediction"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV path for predictions (default: <input_csv>_predictions.csv)"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation with metrics (requires 'stance' column in CSV)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results (default: ./evaluation_results)"
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
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for CSV processing (default: 8)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = StancePredictor(args.model_path)
        
        # CSV mode with evaluation
        if args.csv and args.eval:
            predictor.evaluate(
                csv_path=args.csv,
                batch_size=args.batch_size,
                save_results=True,
                output_dir=args.output_dir
            )
        
        # CSV batch prediction mode
        elif args.csv:
            output_path = args.output
            if not output_path:
                csv_name = Path(args.csv).stem
                output_path = f"{csv_name}_predictions.csv"
            
            results_df = predictor.predict_from_csv(
                args.csv,
                output_path,
                batch_size=args.batch_size
            )
        
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
            print(f"\n→ Predicted Stance: {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print("\n  Probabilities:")
            for label, prob in result['probabilities'].items():
                print(f"    {label}: {prob:.4f}")
        
        else:
            parser.print_help()
            print("\n⚠ Please provide either:")
            print("  --csv <path>              (for batch prediction)")
            print("  --post <text> --comment <text>  (for single prediction)")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    print(precision_recall_fscore_support)