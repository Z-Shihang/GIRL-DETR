import numpy as np
import os
from os.path import join
import torch
from tqdm import tqdm
from PIL import Image

# Import the provided clip module
# jit=False is crucial here to avoid the RuntimeError
import clip

# Import VideoProcessor from your codebase (same as Code 2)
from run_on_video.data_utils import VideoProcessor

# ---------------- Configuration ----------------
# Path to your videos
v_input_dir = "/data1/zhangshihang/data/Charades/Charades_Videos/test/"
# Path to save features
v_feat_dir = "../Datasets/charades/clip_video_features/"

if not os.path.exists(v_feat_dir):
    os.makedirs(v_feat_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Model Initialization ----------------
print(f"Loading CLIP model on {device} (jit=False)...")

# CRITICAL FIX: jit=False prevents the 'forward method not defined' error
# This loads the standard PyTorch model from the model.py file you uploaded
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

# Initialize VideoProcessor (matches your BLIP code settings)
video_loader = VideoProcessor(framerate=1, size=224, centercrop=True)

# ---------------- Helper Functions ----------------

def is_file_present(directory, vid):
    file_path = join(directory, f"{vid}.npz")
    return os.path.exists(file_path)

def read_pending_files(input_dir, output_dir):
    all_files = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".mp4")]
    pending = [vid for vid in all_files if not is_file_present(output_dir, vid)]
    return pending

def save_video_feature(vid, features, out_dir):
    v_feat_path = join(out_dir, f"{vid}.npz")
    # Save features as compressed numpy array
    np.savez_compressed(v_feat_path, features=features.cpu().numpy())

# ---------------- Feature Extraction Logic ----------------

@torch.no_grad()
def encode_video_frames(frames, batch_size=128):
    """
    Encode frames in batches to avoid OOM.
    """
    video_features = []
    n_frames = len(frames)
    
    for i in range(0, n_frames, batch_size):
        # Get batch of raw PIL images
        batch_frames_raw = frames[i:i + batch_size]
        
        # Preprocess images
        batch_tensors = []
        for img in batch_frames_raw:
            # Safety check: convert numpy to PIL if necessary
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            # Apply CLIP preprocessing (Resize, Normalize, etc.)
            tensor = preprocess(img)
            batch_tensors.append(tensor)
        
        # Stack to (Batch, 3, 224, 224) and move to GPU
        batch_input = torch.stack(batch_tensors).to(device)
        
        # Encode using the standard model.encode_image method
        batch_feat = model.encode_image(batch_input)
        
        video_features.append(batch_feat)

    if len(video_features) > 0:
        # Concatenate all batches -> (T, D)
        video_features = torch.cat(video_features, dim=0)
        return video_features
    return None

def process_videos():
    pending_vids = read_pending_files(v_input_dir, v_feat_dir)
    print(f"Found {len(pending_vids)} videos to process.")
    
    if len(pending_vids) == 0:
        print("All videos processed.")
        return

    for vid in tqdm(pending_vids, desc="Extracting CLIP Features"):
        video_path = join(v_input_dir, f"{vid}.mp4")
        
        try:
            # 1. Read frames
            frames, _ = video_loader.read_raw_image_from_video_file(video_path)
            
            if not frames or len(frames) == 0:
                continue

            # 2. Extract features
            feats = encode_video_frames(frames, batch_size=128)
            
            if feats is not None:
                # 3. Save
                save_video_feature(vid, feats, v_feat_dir)
                
        except Exception as e:
            print(f"Failed to process {vid}: {e}")

if __name__ == "__main__":
    process_videos()