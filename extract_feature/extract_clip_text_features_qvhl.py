"""
Extract CLIP ViT-B/32 text features for QVHighlights.
Adapted from extract_query_clip_features_chards.py.

Output: .npz files with keys:
    - last_hidden_state: (77, 512) - token-level features (with padding)
    - pooler_output: (512,) - pooled output

Handles train (paraphrased with aug_id), val, and test splits.

Usage:
    cd /data1/zhangshihang/GIRL_DETR
    python extract_feature/extract_clip_text_features_qvhl.py
"""

import numpy as np
import os
from os.path import join
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_on_video.data_utils import ClipFeatureExtractor


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class QVHQueryDataset(Dataset):
    def __init__(self, input_file):
        self.datalist = load_jsonl(input_file)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        item = self.datalist[i]
        result = {"qid": item["qid"], "query": item["query"]}
        if "aug_id" in item:
            result["aug_id"] = item["aug_id"]
        return result


def collate_fn(batch):
    collated_dict = {}
    for key in batch[0]:
        collated_dict[key] = [data[key] for data in batch]
    return collated_dict


def save_query_features(batch, batch_result, q_feat_dir, training=True):
    batch_size = len(batch["qid"])
    aug_ids = batch.get("aug_id", [0] * batch_size)

    for i, result in enumerate(batch_result):
        qid = batch["qid"][i]
        aug_id = aug_ids[i] if training else 0
        aug = f"_{aug_id}" if aug_id > 0 else ""
        q_feat_path = join(q_feat_dir, f"qid{qid}{aug}.npz")

        os.makedirs(os.path.dirname(q_feat_path), exist_ok=True)

        if isinstance(result, dict):
            lhs = result["last_hidden_state"]
            pooler = result["pooler_output"]
        else:
            lhs = result.last_hidden_state
            pooler = result.pooler_output

        np.savez_compressed(
            q_feat_path,
            last_hidden_state=lhs.detach().cpu().numpy(),
            pooler_output=pooler.detach().cpu().numpy()
        )


def extract_split(feature_extractor, input_file, q_feat_dir, training=True, batch_size=64):
    if not os.path.exists(input_file):
        print(f"Skipping: {input_file} not found")
        return

    print(f"\nExtracting: {input_file} -> {q_feat_dir}")
    os.makedirs(q_feat_dir, exist_ok=True)

    dataset = QVHQueryDataset(input_file)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    for batch in tqdm(dataloader, desc="CLIP Text Features"):
        batch_query = batch['query']
        batch_result = feature_extractor.encode_text_query(batch_query)
        save_query_features(batch, batch_result, q_feat_dir, training)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract CLIP ViT-B/32 text features for QVHighlights"
    )
    parser.add_argument("--train_file", type=str,
                        default="data/highlight_train_release_paraphrased_openai.jsonl")
    parser.add_argument("--val_file", type=str,
                        default="data/highlight_val_release.jsonl")
    parser.add_argument("--test_file", type=str,
                        default="data/highlight_test_release.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/qvhl/features/clip_text_features/",
                        help="Output directory for .npz text feature files")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    feature_extractor = ClipFeatureExtractor(
        framerate=1 / 2, size=224, centercrop=True,
        model_name_or_path="ViT-B/32", device=args.device
    )

    # Train split (paraphrased, with aug_id)
    extract_split(feature_extractor, args.train_file, args.output_dir,
                  training=True, batch_size=args.batch_size)

    # Val split
    extract_split(feature_extractor, args.val_file, args.output_dir,
                  training=False, batch_size=args.batch_size)

    # Test split
    extract_split(feature_extractor, args.test_file, args.output_dir,
                  training=False, batch_size=args.batch_size)

    # Verify output
    files = [f for f in os.listdir(args.output_dir) if f.endswith(".npz")]
    print(f"\nDone! Saved {len(files)} text feature files.")
    if files:
        sample = np.load(join(args.output_dir, files[0]))
        print(f"Sample: last_hidden_state={sample['last_hidden_state'].shape}, "
              f"pooler_output={sample['pooler_output'].shape}")


if __name__ == "__main__":
    main()
