#!/usr/bin/env python
"""
Merge checkpoint with config to create an all-in-one inference checkpoint.
This allows inference without needing opt.json file.

Usage:
    python tools/merge_checkpoint_with_config.py \
        --ckpt_path results/qvhighlights/.../model_best.ckpt \
        --config_path results/qvhighlights/.../.../opt.json \
        --output_path results/qvhighlights/model_best_merged.ckpt
"""

import json
import torch
import argparse
from pathlib import Path


def merge_checkpoint_with_config(ckpt_path, config_path, output_path):
    """Merge checkpoint and config into single file for standalone inference."""
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create merged checkpoint
    merged_checkpoint = {
        'model': checkpoint['model'],
        'config': config,
        'epoch': checkpoint.get('epoch', -1),
    }
    
    print(f"Saving merged checkpoint to {output_path}")
    torch.save(merged_checkpoint, output_path)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ Merged checkpoint saved ({file_size_mb:.2f} MB)")
    print(f"  - Model weights: {sum(p.numel() for p in checkpoint['model'].values() if isinstance(p, torch.Tensor)) / 1e6:.2f}M parameters")
    print(f"  - Config keys: {list(config.keys())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge checkpoint with config for standalone inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config (opt.json)")
    parser.add_argument("--output_path", type=str, required=True, help="Output merged checkpoint path")
    
    args = parser.parse_args()
    merge_checkpoint_with_config(args.ckpt_path, args.config_path, args.output_path)
