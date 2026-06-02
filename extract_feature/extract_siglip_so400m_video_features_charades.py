"""
Extract SigLIP-SO400M (384px) video features for Charades-STA dataset.
Outputs .npz files with key "features" of shape (T, 1152) float16.

Model: google/siglip-so400m-patch14-384 loaded from local path.
  - Vision: 1152d, 428M params, 384px input, patch14
  - Much stronger than siglip-base-patch16-224 (768d, 86M params, 224px)

Usage:
    conda run -n siglip python extract_feature/extract_siglip_so400m_video_features_charades.py \
        --gpu 1 --batch_size 32
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
    def __init__(self, framerate=1, size=384, centercrop=True):
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


# ----------------  Helper Functions ----------------

def is_file_present(directory, vid):
    return os.path.exists(join(directory, f"{vid}.npz"))


def read_pending_files(input_dirs, output_dir):
    all_vids = set()
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            continue
        vids = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".mp4")]
        all_vids.update(vids)
    pending = [vid for vid in sorted(all_vids) if not is_file_present(output_dir, vid)]
    return pending, len(all_vids)


def find_video_path(vid, input_dirs):
    for d in input_dirs:
        path = join(d, f"{vid}.mp4")
        if os.path.exists(path):
            return path
    return None


# ----------------  Feature Extraction ----------------

@torch.no_grad()
def encode_video_frames(frames_tensor, model, processor, device, batch_size=32):
    """
    Encode video frames using SigLIP-SO400M vision encoder.
    Returns (T, 1152) float16 numpy array.
    """
    n_frames = frames_tensor.shape[0]
    all_features = []

    for i in range(0, n_frames, batch_size):
        batch = frames_tensor[i:i + batch_size]  # (B, 3, H, W)

        pil_images = []
        for j in range(batch.shape[0]):
            img_np = batch[j].permute(1, 2, 0).numpy().astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        inputs = processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        feats = model.get_image_features(pixel_values=pixel_values)  # (B, 1152)
        all_features.append(feats.cpu().to(torch.float16).numpy())

    if len(all_features) > 0:
        return np.concatenate(all_features, axis=0)
    return None


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from transformers import SiglipModel, SiglipProcessor
    print(f"Loading SigLIP-SO400M from {args.model_path} on {device}...")
    model = SiglipModel.from_pretrained(args.model_path).to(device)
    model.eval()
    processor = SiglipProcessor.from_pretrained(args.model_path)

    # Verify model
    print(f"  Vision hidden_size: {model.config.vision_config.hidden_size}")
    print(f"  Image size: {model.config.vision_config.image_size}")

    # 1fps, 384px center crop to match SO400M input
    video_loader = VideoProcessor(framerate=args.framerate, size=384, centercrop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    pending, total = read_pending_files(args.input_dirs, args.output_dir)
    print(f"Total videos: {total}, Pending: {len(pending)}")

    if len(pending) == 0:
        print("All videos already processed.")
        return

    failed = []
    for vid in tqdm(pending, desc="SO400M Video Features"):
        video_path = find_video_path(vid, args.input_dirs)
        if video_path is None:
            failed.append(vid)
            continue

        try:
            frames = video_loader.read_video_from_file(video_path)
            if frames is None or len(frames) == 0:
                failed.append(vid)
                continue

            feats = encode_video_frames(frames, model, processor, device, args.batch_size)
            if feats is not None:
                np.savez_compressed(
                    join(args.output_dir, f"{vid}.npz"),
                    features=feats
                )
        except Exception as e:
            print(f"\nFailed {vid}: {e}")
            failed.append(vid)

    if failed:
        print(f"\nFailed: {len(failed)} videos")
        with open(join(args.output_dir, "failed_videos.txt"), "w") as f:
            for v in failed:
                f.write(f"{v}\n")

    print(f"Done! {len(pending) - len(failed)}/{len(pending)} processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SigLIP-SO400M video features for Charades-STA")
    parser.add_argument("--model_path", type=str,
                        default="/data1/zhangshihang/DINO/siglip_model/",
                        help="Local path to SigLIP-SO400M model")
    parser.add_argument("--input_dirs", type=str, nargs="+",
                        default=[
                            "/data1/zhangshihang/data/Charades/Charades_Videos/train/",
                            "/data1/zhangshihang/data/Charades/Charades_Videos/test/",
                            "/data1/zhangshihang/data/Charades/Charades_v1_480/"
                        ])
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/charades/siglip_so400m_video_features/")
    parser.add_argument("--framerate", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (SO400M is larger, use smaller batch)")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    main(args)
