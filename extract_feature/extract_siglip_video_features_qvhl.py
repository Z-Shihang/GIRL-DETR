"""
Extract SigLIP video features for QVHighlights dataset.
Adapted from extract_siglip_video_features_charades.py.

Output: .npz files with key "features" of shape (T, 768) float16
        where T = number of frames at 1 fps.

Requirements: transformers>=4.37.0, torch, Pillow, ffmpeg-python, numpy, tqdm

Usage:
    python extract_feature/extract_siglip_video_features_qvhl.py \
        --video_dir "/data1/zhangshihang/data/QV highlight/videos/" \
        --output_dir /data1/zhangshihang/Datasets/qvhl/features/siglip_video_features/ \
        --gpu 0
"""

import numpy as np
import os
from os.path import join
import torch
from tqdm import tqdm
from PIL import Image
import argparse
import math
import ffmpeg


# ----------------  Inline VideoProcessor ----------------

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


class VideoProcessor:
    def __init__(self, framerate=1, size=224, centercrop=True):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration'])
        except Exception:
            frames_length, duration = -1, -1
        return {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def read_video_from_file(self, video_path):
        try:
            info = self._get_video_info(video_path)
            h, w = info["height"], info["width"]
        except Exception:
            print(f'ffprobe failed at: {video_path}')
            return None
        height, width = self._get_output_dim(h, w)
        try:
            duration = info["duration"]
            fps = self.framerate
            if duration > 0 and duration < 1 / fps + 0.1:
                fps = 2 / max(int(duration), 1)
        except Exception:
            fps = self.framerate
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=fps)
            .filter('scale', width, height)
        )
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = torch.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)  # (T, 3, H, W)
        return video


# ----------------  SigLIP Model Loading ----------------

def load_siglip_model(model_name, device):
    from transformers import AutoModel, AutoProcessor
    print(f"Loading SigLIP model: {model_name} on {device}...")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


# ----------------  Feature Extraction ----------------

@torch.no_grad()
def encode_video_frames_siglip(frames_tensor, model, processor, device, batch_size=64):
    """
    Encode video frames using SigLIP vision encoder.
    Returns numpy array of shape (T, 768) float16.
    """
    n_frames = frames_tensor.shape[0]
    all_features = []

    for i in range(0, n_frames, batch_size):
        batch = frames_tensor[i:i + batch_size]
        pil_images = []
        for j in range(batch.shape[0]):
            img_np = batch[j].permute(1, 2, 0).numpy().astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        inputs = processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        feats = model.get_image_features(pixel_values=pixel_values)
        all_features.append(feats.cpu().to(torch.float16).numpy())

    if len(all_features) > 0:
        return np.concatenate(all_features, axis=0)
    return None


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, processor = load_siglip_model(args.model_name, device)

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

    for vid in tqdm(pending, desc="SigLIP Video Features (QVH)"):
        video_path = all_videos[vid]
        try:
            video_tensor = video_loader.read_video_from_file(video_path)
            if video_tensor is None or video_tensor.shape[0] == 0:
                continue
            feats = encode_video_frames_siglip(
                video_tensor, model, processor, device, args.batch_size
            )
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
        description="Extract SigLIP video features for QVHighlights"
    )
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--video_dir", type=str,
                        default="/data1/zhangshihang/data/QV highlight/videos/",
                        help="Directory containing QVHighlights .mp4 video files")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/qvhl/features/siglip_video_features/",
                        help="Output directory for .npz feature files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--framerate", type=float, default=1,
                        help="Frame extraction rate (fps). Use 1 for 1fps, 0.5 for 2s clips")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    args = parser.parse_args()

    import transformers
    from packaging import version
    if version.parse(transformers.__version__) < version.parse("4.37.0"):
        print(f"ERROR: transformers {transformers.__version__} too old for SigLIP. Need >=4.37.0")
        exit(1)

    main(args)
