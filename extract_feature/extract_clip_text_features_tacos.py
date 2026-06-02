"""
Extract CLIP ViT-B/32 text features for TACoS.

Output files follow TACoS annotation qid exactly:
    {qid}.npz

Each file contains:
    - last_hidden_state: (77, 512)
    - pooler_output: (512,)

Usage:
    cd /data1/zhangshihang/GIRL_DETR
    python extract_feature/extract_clip_text_features_tacos.py --gpu 5
"""

import argparse
import json
import os
import sys
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_on_video.data_utils import ClipFeatureExtractor


def load_jsonl(filename):
    with open(filename, "r") as file_obj:
        return [json.loads(line.strip("\n")) for line in file_obj.readlines()]


class TACoSQueryDataset(Dataset):
    def __init__(self, input_file):
        self.datalist = load_jsonl(input_file)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        item = self.datalist[index]
        return {"qid": item["qid"], "query": item["query"]}


def collate_fn(batch):
    collated_dict = {}
    for key in batch[0]:
        collated_dict[key] = [data[key] for data in batch]
    return collated_dict


def save_query_features(batch, batch_result, output_dir):
    for index, result in enumerate(batch_result):
        qid = batch["qid"][index]
        output_path = join(output_dir, f"{qid}.npz")

        if isinstance(result, dict):
            last_hidden_state = result["last_hidden_state"]
            pooler_output = result["pooler_output"]
        else:
            last_hidden_state = result.last_hidden_state
            pooler_output = result.pooler_output

        np.savez_compressed(
            output_path,
            last_hidden_state=last_hidden_state.detach().cpu().numpy(),
            pooler_output=pooler_output.detach().cpu().numpy(),
        )


def extract_split(feature_extractor, input_file, output_dir, batch_size):
    if not os.path.exists(input_file):
        print(f"Skipping missing split file: {input_file}")
        return

    dataset = TACoSQueryDataset(input_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    print(f"Extracting TACoS text features from {input_file}")
    for batch in tqdm(dataloader, desc="CLIP Text Features (TACoS)"):
        batch_result = feature_extractor.encode_text_query(batch["query"])
        save_query_features(batch, batch_result, output_dir)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    feature_extractor = ClipFeatureExtractor(
        framerate=1,
        size=224,
        centercrop=True,
        model_name_or_path="ViT-B/32",
        device=device,
    )

    extract_split(feature_extractor, args.train_file, args.output_dir, args.batch_size)
    extract_split(feature_extractor, args.val_file, args.output_dir, args.batch_size)
    extract_split(feature_extractor, args.test_file, args.output_dir, args.batch_size)

    saved_files = [file_name for file_name in os.listdir(args.output_dir) if file_name.endswith(".npz")]
    print(f"Done! Saved {len(saved_files)} TACoS CLIP text feature files to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP ViT-B/32 text features for TACoS")
    parser.add_argument("--train_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/train.jsonl")
    parser.add_argument("--val_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/val.jsonl")
    parser.add_argument("--test_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/test.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/clip_text_features/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    main(parser.parse_args())