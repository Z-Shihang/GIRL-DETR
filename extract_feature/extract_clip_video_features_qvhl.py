"""
Extract CLIP ViT-B/32 video features for QVHighlights.
Adapted from extract_query_clip_video.py (Charades-STA version).

Output: .npz files with key "features" of shape (T, 512) float32
        where T = number of frames at 1 fps.

Usage:
    cd /data1/zhangshihang/GIRL_DETR
    python extract_feature/extract_clip_video_features_qvhl.py --gpu 0
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

# Import the custom clip module (jit=False is needed)
from extract_feature import clip
from run_on_video.data_utils import VideoProcessor


def is_file_present(directory, vid):
    return os.path.exists(join(directory, f"{vid}.npz"))


@torch.no_grad()
def encode_video_frames(model, preprocess, frames, device, batch_size=128):
    """Encode video frames in batches using CLIP. Returns (T, 512) float32."""
    video_features = []

    for i in range(0, len(frames), batch_size):
        batch_frames_raw = frames[i:i + batch_size]
        batch_tensors = []
        for img in batch_frames_raw:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            batch_tensors.append(preprocess(img))
        batch_input = torch.stack(batch_tensors).to(device)
        batch_feat = model.encode_image(batch_input)
        video_features.append(batch_feat)

    if video_features:
        return torch.cat(video_features, dim=0)
    return None


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Loading CLIP model on {device} (jit=False)...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    # 1 fps matching Charades-STA extraction
    video_loader = VideoProcessor(framerate=args.framerate, size=224, centercrop=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all QVHighlights videos from multiple directories
    all_videos = {}
    for vdir in args.video_dirs:
        if os.path.exists(vdir):
            for f in os.listdir(vdir):
                if f.endswith(".mp4"):
                    vid_name = os.path.splitext(f)[0]
                    if vid_name not in all_videos:
                        all_videos[vid_name] = join(vdir, f)
    pending = [vid for vid in sorted(all_videos) if not is_file_present(args.output_dir, vid)]

    print(f"Total: {len(all_videos)}, Pending: {len(pending)}")

    for vid in tqdm(pending, desc="CLIP Video Features (QVH)"):
        video_path = all_videos[vid]
        try:
            frames, _ = video_loader.read_raw_image_from_video_file(video_path)
            if not frames or len(frames) == 0:
                continue
            feats = encode_video_frames(model, preprocess, frames, device, args.batch_size)
            if feats is not None:
                np.savez_compressed(
                    join(args.output_dir, f"{vid}.npz"),
                    features=feats.cpu().numpy()
                )
        except Exception as e:
            print(f"\nFailed {vid}: {e}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP ViT-B/32 video features for QVHighlights"
    )
    parser.add_argument("--video_dirs", type=str, nargs="+",
                        default=[
                            "/data1/zhangshihang/data/QV highlight/train/",
                            "/data1/zhangshihang/data/QV highlight/val/",
                            "/data1/zhangshihang/data/QV highlight/test/",
                            "/data1/zhangshihang/data/QV highlight/videos/",
                        ],
                        help="QVHighlights video directories (videos split across train/val/test)")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/qvhl/features/clip_video_features/",
                        help="Output directory for .npz feature files")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for frame encoding")
    parser.add_argument("--framerate", type=float, default=1,
                        help="Frame extraction rate (fps). 1=1fps, 0.5=2s clips")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    args = parser.parse_args()
    main(args)
