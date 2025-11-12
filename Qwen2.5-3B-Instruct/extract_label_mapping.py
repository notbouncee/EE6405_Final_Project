#!/usr/bin/env python3
"""
Extract label mappings from a trained checkpoint
Run this if your checkpoint is missing label_mappings.json
"""

import json
from pathlib import Path
from transformers import AutoConfig
import argparse

def extract_label_mappings(checkpoint_path, output_path=None):
    """
    Extract label mappings from a checkpoint's config.json
    
    Args:
        checkpoint_path: Path to checkpoint directory
        output_path: Where to save label_mappings.json (default: same directory)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return False
    
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        print(f"❌ Error: config.json not found in {checkpoint_path}")
        return False
    
    print(f"Reading config from: {config_path}")
    
    # Load the config
    config = AutoConfig.from_pretrained(checkpoint_path)
    
    # Extract label mappings
    if not hasattr(config, 'label2id') or not hasattr(config, 'id2label'):
        print("❌ Error: No label mappings found in config")
        return False
    
    label2id = config.label2id
    id2label = config.id2label
    
    print(f"\n✓ Found label mappings:")
    print(f"  Labels: {list(label2id.keys())}")
    print(f"  Number of labels: {len(label2id)}")
    
    # Create label mappings dictionary
    label_mappings = {
        'label2id': label2id,
        'id2label': id2label
    }
    
    # Determine output path
    if output_path is None:
        output_path = checkpoint_path / "label_mappings.json"
    else:
        output_path = Path(output_path)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(label_mappings, f, indent=2)
    
    print(f"\n✓ Saved label mappings to: {output_path}")
    print("\nYou can now use this checkpoint with inference_qwen.py!")
    
    return True


def main():
    # parser = argparse.ArgumentParser(
    #     description="Extract label mappings from a checkpoint"
    # )
    # parser.add_argument(
    #     "checkpoint_path",
    #     type=str,
    #     default="/opt/tiger/MLLM_AUTO_EVALUATE_PIPELINE/EE6405_Final_Project/results_qwen/run-4/checkpoint-1800",
    #     help="Path to checkpoint directory (e.g., ./results_qwen/checkpoint-1800)"
    # )
    # parser.add_argument(
    #     "--output",
    #     type=str,
    #     default=None,
    #     help="Output path for label_mappings.json (default: same as checkpoint)"
    # )
    
    # args = parser.parse_args()
    checkpoint_path = "/opt/tiger/MLLM_AUTO_EVALUATE_PIPELINE/EE6405_Final_Project/results_qwen/run-4/checkpoint-1800"
    # output = "/opt/tiger/MLLM_AUTO_EVALUATE_PIPELINE/EE6405_Final_Project/results_qwen/run-4/checkpoint-1800"
    success = extract_label_mappings(checkpoint_path, None)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()

