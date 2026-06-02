"""
Extract EVA02-CLIP-B/16 video features for QVHighlights.
Adapted from extract_eva02clip_video_features_charades.py.

Output: .npz files with key "features" of shape (T, 512) float32
        where T = number of frames at 1 fps (matching Charades-STA framerate).

Requires: pip install open_clip_torch

Usage:
    python extract_feature/extract_eva02clip_video_features_qvhl.py \
        --video_dir "/data1/zhangshihang/data/QV highlight/videos/" \
        --output_dir /data1/zhangshihang/Datasets/qvhl/features/eva02clip_video_features/ \
        --batch_size 512 --gpu 0
"""

import numpy as np
import os
from os.path import join
import torch
from tqdm import tqdm
from PIL import Image
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_on_video.data_utils import VideoProcessor


@torch.no_grad()
def encode_frames(model, preprocess, frames, device, batch_size=512):
    """
    Encode video frames using EVA02-CLIP vision encoder.
    Returns numpy array of shape (T, 512) float32, or None if no frames.
    """
    all_feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        tensors = []
        for img in batch:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            tensors.append(preprocess(img))
        batch_input = torch.stack(tensors).to(device)
        feats = model.encode_image(batch_input)  # (B, 512) projected
        all_feats.append(feats.float().cpu())
    if all_feats:
        return torch.cat(all_feats, dim=0).numpy()
    return None


def load_eva02clip(device, weights_path=None):
    """Load EVA02-CLIP-B/16 model, trying local cache first then download."""
    import open_clip

    default_cache = os.path.expanduser(
        '~/.cache/modelscope/hub/models/timm/'
        'eva02_base_patch16_clip_224___merged2b_s8b_b131k/open_clip_pytorch_model.bin'
    )
    if weights_path is None:
        weights_path = default_cache

    if os.path.isfile(weights_path):
        print(f"Loading EVA02-CLIP-B/16 from local cache: {weights_path}")
        model, _, preprocess = open_clip.create_model_and_transforms('EVA02-B-16', pretrained='')
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print(f"Local cache not found at {weights_path}, downloading...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'EVA02-B-16', pretrained='merged2b_s8b_b131k'
        )

    model = model.to(device).eval()
    return model, preprocess


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, preprocess = load_eva02clip(device, args.weights_path)

    # Verify output dimension
    dummy = torch.randn(1, 3, 224, 224).to(device)
    dummy_out = model.encode_image(dummy)
    feat_dim = dummy_out.shape[-1]
    print(f"Visual feature dim: {feat_dim}")
    assert feat_dim == 512, f"Expected 512-dim but got {feat_dim}!"

    # 1 fps, 224x224 (matching Charades-STA extraction)
    video_loader = VideoProcessor(framerate=args.framerate, size=224, centercrop=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # QVHighlights videos are all .mp4 in a flat directory
    all_videos = {}
    for f in os.listdir(args.video_dir):
        if f.endswith(".mp4"):
            vid_name = os.path.splitext(f)[0]
            all_videos[vid_name] = join(args.video_dir, f)
    all_vids = sorted(all_videos.keys())

    existing = set(
        os.path.splitext(f)[0] for f in os.listdir(args.output_dir)
        if f.endswith(".npz")
    )
    pending = [v for v in all_vids if v not in existing]

    print(f"Total: {len(all_vids)}, Already done: {len(existing)}, Pending: {len(pending)}")

    for vid in tqdm(pending, desc="EVA02-CLIP Video Features (QVH)"):
        video_path = all_videos[vid]
        try:
            frames, _ = video_loader.read_raw_image_from_video_file(video_path)
            if not frames:
                continue
            feats = encode_frames(model, preprocess, frames, device, args.batch_size)
            if feats is not None:
                np.savez_compressed(
                    join(args.output_dir, f"{vid}.npz"),
                    features=feats
                )
        except Exception as e:
            print(f"\nFailed {vid}: {e}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract EVA02-CLIP-B/16 video features for QVHighlights"
    )
    parser.add_argument("--video_dir", type=str,
                        default="/data1/zhangshihang/data/QV highlight/videos/",
                        help="Directory containing QVHighlights .mp4 video files")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/qvhl/features/eva02clip_video_features/",
                        help="Output directory for .npz feature files")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for frame encoding")
    parser.add_argument("--framerate", type=float, default=1,
                        help="Frame extraction rate (fps). Use 1 for 1fps (clip_length=1), 0.5 for 2s clips")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Local path to open_clip_pytorch_model.bin")
    args = parser.parse_args()
    main(args)
