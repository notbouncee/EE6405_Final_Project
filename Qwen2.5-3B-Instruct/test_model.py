#!/usr/bin/env python3
"""
Quick test script for evaluating trained Qwen model on test data.
This script uses the inference_qwen module to run comprehensive evaluation.
"""

from qwen_inference import StancePredictor

def main():
    print("=" * 80)
    print("Qwen Stance Detection - Quick Test Script")
    print("=" * 80)
    
    # Configuration - UPDATE THESE PATHS!
    MODEL_PATH = "/opt/tiger/MLLM_AUTO_EVALUATE_PIPELINE/EE6405_Final_Project/results_qwen/run-4/checkpoint-1800"  # or checkpoint path
    TEST_DATA_PATH = "./data/preprocessed/redditAITA_test.csv"  # TEST data only!
    OUTPUT_DIR = "./evaluation_results"
    BATCH_SIZE = 8  # Adjust based on your GPU memory
    
    # Validate test data path
    if 'train' in TEST_DATA_PATH.lower() and 'test' not in TEST_DATA_PATH.lower():
        print("\n" + "=" * 80)
        print("❌ ERROR: You're trying to use TRAINING data!")
        print("=" * 80)
        print(f"File: {TEST_DATA_PATH}")
        print("\nPlease update TEST_DATA_PATH to point to TEST data:")
        print("  - redditAITA_test.csv")
        print("  - reddit_posts_and_comments_test.csv")
        print("  - stance_dataset_test.csv")
        print("\nEvaluating on training data will give misleading results!")
        print("=" * 80)
        return
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Test Data: {TEST_DATA_PATH}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    # Initialize predictor
    print(f"\n{'='*80}")
    print("Step 1: Loading Model")
    print(f"{'='*80}")
    
    try:
        predictor = StancePredictor(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("  1. You have trained a model (run train_qwen_stance.py)")
        print("  2. The model path is correct")
        print(f"  3. Model files exist at: {MODEL_PATH}")
        return
    
    # Run evaluation
    print(f"\n{'='*80}")
    print("Step 2: Running Comprehensive Evaluation")
    print(f"{'='*80}")
    
    try:
        metrics = predictor.evaluate(
            csv_path=TEST_DATA_PATH,
            batch_size=BATCH_SIZE,
            save_results=True,
            output_dir=OUTPUT_DIR
        )
        
        # Summary
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"✓ Macro F1: {metrics['macro_avg']['f1']:.4f}")
        print(f"✓ Weighted F1: {metrics['weighted_avg']['f1']:.4f}")
        
        print(f"\n{'='*80}")
        print("Per-Class Performance:")
        print(f"{'='*80}")
        for label, scores in metrics['per_class'].items():
            print(f"\n{label.upper()}:")
            print(f"  Precision: {scores['precision']:.4f}")
            print(f"  Recall:    {scores['recall']:.4f}")
            print(f"  F1-Score:  {scores['f1']:.4f}")
            print(f"  Support:   {scores['support']}")
        
        print(f"\n{'='*80}")
        print("Results saved to:")
        print(f"{'='*80}")
        print(f"  • Predictions: {OUTPUT_DIR}/predictions.csv")
        print(f"  • Metrics: {OUTPUT_DIR}/metrics.json")
        print(f"  • Confusion Matrix: {OUTPUT_DIR}/confusion_matrix.png")
        
        print(f"\n{'='*80}")
        print("✓ Evaluation Complete!")
        print(f"{'='*80}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Test data not found at {TEST_DATA_PATH}")
        print("\nPlease ensure:")
        print("  1. You have preprocessed data (run redditAITApreprocessing.py or sqlite_prep.py)")
        print("  2. The test data path is correct")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

