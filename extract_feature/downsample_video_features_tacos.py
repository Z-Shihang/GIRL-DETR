"""
Downsample 1fps TACoS video features by stride N (e.g., stride=4 → 0.25fps).

With stride=4 and max_v_l=256:
  coverage = 256 × 4 = 1024 seconds → covers ALL TACoS videos (max=928s).

  Previously, max_v_l=256 at 1fps only covered 256s, leaving 18.8% of val
  queries completely outside the visible window (GT start > 256s).

Usage:
    cd {project_root}
    python extract_feature/downsample_video_features_tacos.py --stride 4
"""

import argparse
import os
from glob import glob

import numpy as np
from tqdm import tqdm


FEAT_ROOT = os.environ.get("FEAT_ROOT", "data/tacos")

# Video feature directories to downsample (text features don't need this)
VIDEO_FEAT_DIRS = [
    "internvideo2_video_features",
    "clip_video_features",
    "blip_video_features",
]


def downsample_stride(features, stride):
    """Take every `stride`-th frame: (T, D) -> (T//stride, D)."""
    return features[::stride]


def process_directory(src_dir, dst_dir, stride):
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
        # Support both "features" and any other key
        keys = list(data.keys())
        feat_key = "features" if "features" in keys else keys[0]
        features = data[feat_key]  # (T, D)

        features_ds = downsample_stride(features, stride)
        np.savez_compressed(dst_path, features=features_ds.astype(features.dtype))
        processed += 1

    print(f"  Done: {processed} processed, {skipped} skipped")
    if processed + skipped > 0:
        sample = glob(os.path.join(dst_dir, "*.npz"))[0]
        d = np.load(sample)
        print(f"  Sample shape: {d['features'].shape}, dtype: {d['features'].dtype}")


def main(args):
    stride = args.stride
    print(f"Downsampling TACoS video features with stride={stride}")
    print(f"Coverage: {256 * stride}s at max_v_l=256, {512 * stride}s at max_v_l=512")
    print(f"Feature root: {FEAT_ROOT}\n")

    for src_name in VIDEO_FEAT_DIRS:
        src_dir = os.path.join(FEAT_ROOT, src_name)
        dst_dir = os.path.join(FEAT_ROOT, f"{src_name}_s{stride}")

        if not os.path.exists(src_dir):
            print(f"SKIP: {src_dir} not found")
            continue

        n_files = len(glob(os.path.join(src_dir, "*.npz")))
        print(f"{src_name} ({n_files} files) -> {os.path.basename(dst_dir)}")
        process_directory(src_dir, dst_dir, stride)
        print()

    print("Done! New feature directories:")
    for src_name in VIDEO_FEAT_DIRS:
        dst_dir = os.path.join(FEAT_ROOT, f"{src_name}_s{stride}")
        if os.path.exists(dst_dir):
            n = len(glob(os.path.join(dst_dir, "*.npz")))
            print(f"  {dst_dir}: {n} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=4,
                        help="Temporal stride (4 = keep every 4th frame, 0.25fps from 1fps)")
    main(parser.parse_args())
