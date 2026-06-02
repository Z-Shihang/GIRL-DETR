"""
Extract SigLIP video features for Charades-STA dataset.
Outputs .npz files with key "features" of shape (T, 768) matching CLIP format.

Requirements: transformers>=4.37.0, torch, Pillow, ffmpeg-python, numpy, tqdm

Usage:
    conda run -n siglip_extract python extract_feature/extract_siglip_video_features_charades.py
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


# ----------------  Inline VideoProcessor (no external imports) ----------------

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
    """Pytorch video loader. Extracted from run_on_video/data_utils.py to avoid CLIP imports."""

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
    """Load SigLIP model and processor from HuggingFace."""
    from transformers import AutoModel, AutoProcessor
    print(f"Loading SigLIP model: {model_name} on {device}...")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


# ----------------  Helper Functions ----------------

def is_file_present(directory, vid):
    return os.path.exists(join(directory, f"{vid}.npz"))


def read_pending_files(input_dirs, output_dir):
    """Collect all mp4 files from multiple input directories, skip already processed."""
    all_vids = set()
    for input_dir in input_dirs:
        vids = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".mp4")]
        all_vids.update(vids)
    pending = [vid for vid in sorted(all_vids) if not is_file_present(output_dir, vid)]
    return pending


def find_video_path(vid, input_dirs):
    """Find the actual path of a video across multiple directories."""
    for d in input_dirs:
        path = join(d, f"{vid}.mp4")
        if os.path.exists(path):
            return path
    return None


def save_video_feature(vid, features, out_dir):
    v_feat_path = join(out_dir, f"{vid}.npz")
    np.savez_compressed(v_feat_path, features=features)


# ----------------  Feature Extraction ----------------

@torch.no_grad()
def encode_video_frames_siglip(frames_tensor, model, processor, device, batch_size=64):
    """
    Encode video frames using SigLIP vision encoder.
    Uses model.get_image_features() which returns the pooled visual representation.

    Note: Unlike CLIP (which has a separate visual.proj projection), SigLIP's
    vision and text towers both output 768-dim features without explicit projection.
    The alignment is learned implicitly via the sigmoid contrastive loss.
    
    Args:
        frames_tensor: torch.Tensor of shape (T, 3, H, W) float32 [0, 255]
        model: SigLIP model
        processor: SigLIP processor (for image normalization)
        device: torch device
        batch_size: batch size for encoding
    
    Returns:
        numpy array of shape (T, hidden_dim) float16
    """
    n_frames = frames_tensor.shape[0]
    all_features = []
    
    for i in range(0, n_frames, batch_size):
        batch = frames_tensor[i:i + batch_size]  # (B, 3, H, W)
        
        # Convert to PIL images for the processor
        pil_images = []
        for j in range(batch.shape[0]):
            # frames_tensor is (C, H, W) float32, values in [0, 255]
            img_np = batch[j].permute(1, 2, 0).numpy().astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        # Use the SigLIP processor (handles resize + normalize with mean=0.5, std=0.5)
        inputs = processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # get_image_features returns pooler_output (no additional projection in SigLIP)
        feats = model.get_image_features(pixel_values=pixel_values)  # (B, hidden_dim)
        
        all_features.append(feats.cpu().to(torch.float16).numpy())
    
    if len(all_features) > 0:
        return np.concatenate(all_features, axis=0)  # (T, hidden_dim)
    return None


def process_videos(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, processor = load_siglip_model(args.model_name, device)
    
    # Frame extraction with same settings as CLIP
    video_loader = VideoProcessor(framerate=args.framerate, size=224, centercrop=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_dirs = args.input_dirs
    pending_vids = read_pending_files(input_dirs, args.output_dir)
    print(f"Found {len(pending_vids)} videos to process (already done: skipped)")
    
    if len(pending_vids) == 0:
        print("All videos already processed.")
        return
    
    failed = []
    for vid in tqdm(pending_vids, desc="Extracting SigLIP Video Features"):
        video_path = find_video_path(vid, input_dirs)
        if video_path is None:
            failed.append(vid)
            continue
        
        try:
            # 1. Read frames using the same VideoProcessor as CLIP extraction
            frames = video_loader.read_video_from_file(video_path)  # (T, 3, H, W) float32
            
            if isinstance(frames, dict) or frames is None or len(frames) == 0:
                failed.append(vid)
                continue
            
            # 2. Extract features
            feats = encode_video_frames_siglip(frames, model, processor, device, 
                                               batch_size=args.batch_size)
            
            if feats is not None:
                # 3. Save in same format as CLIP features
                save_video_feature(vid, feats, args.output_dir)
                
        except Exception as e:
            print(f"Failed to process {vid}: {e}")
            failed.append(vid)
    
    if failed:
        print(f"\nFailed videos ({len(failed)}):")
        failed_path = join(args.output_dir, "failed_videos.txt")
        with open(failed_path, "w") as f:
            for vid in failed:
                f.write(f"{vid}\n")
        print(f"Saved to {failed_path}")
    
    print(f"Done! Processed {len(pending_vids) - len(failed)}/{len(pending_vids)} videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SigLIP video features for Charades-STA")
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224",
                        help="HuggingFace SigLIP model name")
    parser.add_argument("--input_dirs", type=str, nargs="+",
                        default=[
                            "/data1/zhangshihang/data/Charades/Charades_Videos/train/",
                            "/data1/zhangshihang/data/Charades/Charades_Videos/test/",
                            "/data1/zhangshihang/data/Charades/Charades_v1_480/"
                        ],
                        help="Directories containing .mp4 video files")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/charades/siglip_video_features/",
                        help="Output directory for .npz feature files")
    parser.add_argument("--framerate", type=float, default=1.0,
                        help="Frame extraction rate (default: 1.0 = 1 frame per second, same as CLIP for Charades)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for SigLIP encoding")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    
    args = parser.parse_args()
    
    # Check transformers version
    import transformers
    from packaging import version
    if version.parse(transformers.__version__) < version.parse("4.37.0"):
        print(f"ERROR: transformers {transformers.__version__} is too old for SigLIP.")
        print("Please upgrade: pip install transformers>=4.37.0")
        exit(1)
    
    process_videos(args)
