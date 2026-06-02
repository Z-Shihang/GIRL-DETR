"""
Extract BLIP (base) video + text features for QVHighlights.
Directly adapted from extract_query_blip_features_chards.py.
Uses BLIP base (blip_feature_extractor), NOT BLIP2!

Usage:
    cd {project_root}
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python extract_feature/extract_blip_features_qvhl.py
"""

import os
# Force offline mode — BLIP model(model_base.pth) and bert-base-uncased tokenizer
# are already cached locally, no need to access HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import io
import math
import json
import sys
import numpy as np
import torch
from os.path import join
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

# Add project root to path so run_on_video can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from run_on_video.data_utils import VideoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BLIP base (NOT BLIP2!) — matching Charades-STA extraction
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip_feature_extractor",
    model_type="base",
    is_eval=True,
    device=device
)

# 1 fps, 224x224 — matching Charades-STA extraction
video_loader = VideoProcessor(framerate=1, size=224, centercrop=True)

v_input_dirs = [
    os.environ.get("VIDEO_TRAIN_DIR", os.path.join(project_root, "data/QV_highlight/train/")),
    os.environ.get("VIDEO_VAL_DIR", os.path.join(project_root, "data/QV_highlight/val/")),
    os.environ.get("VIDEO_TEST_DIR", os.path.join(project_root, "data/QV_highlight/test/")),
    os.environ.get("VIDEO_FULL_DIR", os.path.join(project_root, "data/QV_highlight/videos/")),
]
v_feat_dir = os.environ.get("V_FEAT_DIR", os.path.join(project_root, "data/qvhl/features/blip_video_features/"))
q_feat_dir = os.environ.get("Q_FEAT_DIR", os.path.join(project_root, "data/qvhl/features/blip_text_features/"))


def encode_text_query(batch):
    batch_output = []
    with torch.no_grad():
        for text in batch:
            text_input = txt_processors["eval"](text)
            sample = {"text_input": [text_input]}
            features_text = model.extract_features(sample, mode="text")
            batch_output.append(features_text)
        return batch_output


@torch.no_grad()
def encode_video(video_path: str, batch_size=16):
    video_frames, _ = video_loader.read_raw_image_from_video_file(video_path)
    n_frames = len(video_frames)
    video_features = []
    for i in range(0, n_frames, batch_size):
        batch_frames = video_frames[i:i + batch_size]
        batch_tensors = torch.stack(
            [vis_processors["eval"](f) for f in batch_frames]
        ).to(device)
        sample = {"image": batch_tensors}
        features_image = model.extract_features(sample, mode="image")
        video_features.append(features_image.image_embeds[:, 0, :])
    video_features = torch.cat(video_features, dim=0)
    return video_features


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class QVHighlightsDataset(Dataset):
    def __init__(self, input_file):
        self.datalist = load_jsonl(input_file)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        new_dict = dict.fromkeys(
            ['qid', 'query', 'duration', 'vid', 'relevant_clip_ids',
             'saliency_scores', 'relevant_windows'])
        new_dict.update(self.datalist[i])
        return new_dict


def generate_batched_query(batch):
    return batch['query']


def generate_batched_vid(v_feat_dir, batch):
    return [vid for vid in batch['vid'] if not is_file_present(v_feat_dir, vid)]


def read_pending_files_from_directory(directory_path):
    return [os.path.splitext(f)[0] for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
            and not is_file_present(v_feat_dir, os.path.splitext(f)[0])]


def save_query_features(batch, batch_result, q_feat_dir, training=True):
    for i, result in enumerate(batch_result):
        qid = batch["qid"][i]
        aug_id = batch.get("aug_id", [0] * len(batch["qid"]))[i] if training else 0
        aug = f"_{aug_id}" if aug_id > 0 else ""
        q_feat_path = join(q_feat_dir, f"qid{qid}{aug}.npz")

        # Save both last_hidden_state and pooler_output
        # Matching Charades-STA blip_text_features format
        text_embeds = result.text_embeds.squeeze()  # (L, D)
        pooler_output = text_embeds[0, :]  # First token as pooler (D,)

        np.savez_compressed(
            q_feat_path,
            last_hidden_state=text_embeds.cpu().numpy(),
            pooler_output=pooler_output.cpu().numpy(),
        )


def save_video_feature(vid, result, v_feat_dir):
    v_feat_path = join(v_feat_dir, f"{vid}.npz")
    print("\nSaving: ", v_feat_path)
    np.savez_compressed(v_feat_path, features=result.cpu().numpy())


def is_file_present(v_feat_dir, vid):
    file_path = join(v_feat_dir, f"{vid}.npz")
    return os.path.exists(file_path)


def collate_fn(batch):
    collated_dict = {}
    for key in batch[0]:
        collated_dict[key] = [data[key] for data in batch]
    return collated_dict


def find_video_path(vid, input_dirs):
    """Find video path across multiple directories."""
    for d in input_dirs:
        path = join(d, f"{vid}.mp4")
        if os.path.exists(path):
            return path
    return None


def encode_video_query(input_dirs, batch, out_dir=v_feat_dir):
    with torch.no_grad():
        for vid in tqdm(batch):
            video_path = find_video_path(vid, input_dirs)
            if video_path is None:
                print(f"\nVideo not found: {vid}")
                continue
            video_feature = encode_video(video_path)
            save_video_feature(vid, video_feature, out_dir)


def extract_video_features(input_file):
    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_vid = generate_batched_vid(v_feat_dir, batch)
        if len(batch_vid) == 0:
            continue
        print("Processing:", batch_vid)
        encode_video_query(v_input_dirs, batch_vid)


def extract_pending_video_features():
    os.makedirs(v_feat_dir, exist_ok=True)
    # Collect all videos from all input directories
    all_vids = set()
    for input_dir in v_input_dirs:
        if os.path.exists(input_dir):
            vids = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".mp4")]
            all_vids.update(vids)
    batch_vid = [v for v in sorted(all_vids) if not is_file_present(v_feat_dir, v)]
    if len(batch_vid) == 0:
        print("All video files present")
        return
    print(f"Processing {len(batch_vid)} pending videos (total found: {len(all_vids)})...")
    encode_video_query(v_input_dirs, batch_vid)


def extract_query_features(input_file, feat_dir=q_feat_dir, training=True):
    os.makedirs(feat_dir, exist_ok=True)
    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        batch_result = encode_text_query(batch_query)
        save_query_features(batch, batch_result, feat_dir, training=training)


def extract_train_query_features():
    input_file = "data/highlight_train_release_paraphrased_openai.jsonl"
    extract_query_features(input_file, q_feat_dir, True)


def extract_val_query_features():
    input_file = "data/highlight_val_release.jsonl"
    extract_query_features(input_file, q_feat_dir, False)


def extract_test_query_features():
    input_file = "data/highlight_test_release.jsonl"
    extract_query_features(input_file, q_feat_dir, False)


def extract_all():
    # Text features
    extract_train_query_features()
    extract_val_query_features()
    extract_test_query_features()
    # Video features
    extract_pending_video_features()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--video-only":
        extract_pending_video_features()
    else:
        extract_all()
