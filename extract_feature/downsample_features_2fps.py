"""
Downsample 1fps video features (150 frames) to 2fps-equivalent (75 frames)
by averaging consecutive pairs of frames.

This creates features compatible with the original paper's 2fps setup
(clip_length=2, max_v_l=75).

Usage:
    cd /data1/zhangshihang/GIRL_DETR
    python extract_feature/downsample_features_2fps.py
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm

FEAT_ROOT = "/data1/zhangshihang/Datasets/qvhl/features"

# Source (1fps) -> Target (2fps) directory mapping
# Target names match the original paper's expected directory names
MAPPING = {
    "sf_features": "slowfast_features",
    "clip_video_features": "clip_features",
    "blip_video_features": "blip_video_features_2fps",
}


def downsample_to_2fps(features):
    """Average consecutive pairs of frames: (N, D) -> (N//2, D)"""
    n_frames = features.shape[0]
    if n_frames % 2 == 1:
        # Pad by repeating last frame
        features = np.concatenate([features, features[-1:]], axis=0)
        n_frames += 1
    # Reshape to (N//2, 2, D) and average
    return features.reshape(n_frames // 2, 2, -1).mean(axis=1)


def process_directory(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob(os.path.join(src_dir, "*.npz")))
    
    skipped = 0
    processed = 0
    
    for fpath in tqdm(files, desc=f"  {os.path.basename(src_dir)} -> {os.path.basename(dst_dir)}"):
        fname = os.path.basename(fpath)
        dst_path = os.path.join(dst_dir, fname)
        
        if os.path.exists(dst_path):
            skipped += 1
            continue
        
        data = np.load(fpath)
        features = data["features"]  # (150, D)
        
        # Downsample
        features_2fps = downsample_to_2fps(features)  # (75, D)
        
        # Save with same dtype
        np.savez_compressed(dst_path, features=features_2fps.astype(features.dtype))
        processed += 1
    
    print(f"  Done: {processed} processed, {skipped} skipped (already exist)")
    
    # Verify a sample
    sample = glob(os.path.join(dst_dir, "*.npz"))[0]
    d = np.load(sample)
    print(f"  Sample shape: {d['features'].shape}, dtype: {d['features'].dtype}")


def main():
    print("Downsampling 1fps features to 2fps-equivalent")
    print(f"Feature root: {FEAT_ROOT}\n")
    
    for src_name, dst_name in MAPPING.items():
        src_dir = os.path.join(FEAT_ROOT, src_name)
        dst_dir = os.path.join(FEAT_ROOT, dst_name)
        
        if not os.path.exists(src_dir):
            print(f"SKIP: {src_dir} not found")
            continue
        
        n_files = len(glob(os.path.join(src_dir, "*.npz")))
        print(f"\n{src_name} ({n_files} files) -> {dst_name}")
        process_directory(src_dir, dst_dir)
    
    print("\nAll done! New feature directories:")
    for dst_name in MAPPING.values():
        dst_dir = os.path.join(FEAT_ROOT, dst_name)
        if os.path.exists(dst_dir):
            n = len(glob(os.path.join(dst_dir, "*.npz")))
            print(f"  {dst_dir}: {n} files")


if __name__ == "__main__":
    main()
