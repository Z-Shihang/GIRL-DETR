"""
Extract SigLIP text features for Charades-STA dataset.
Outputs .npz files with keys "last_hidden_state" (L, 768) and "pooler_output" (768,)
matching the CLIP text feature format.

Requirements: transformers>=4.37.0, torch, numpy, tqdm

Usage:
    conda run -n video_lights python extract_feature/extract_siglip_text_features_charades.py
"""

import numpy as np
import os
from os.path import join
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class CharadesQueryDataset(Dataset):
    def __init__(self, input_file):
        self.datalist = load_jsonl(input_file)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        item = self.datalist[i]
        return {
            "qid": item["qid"],
            "query": item["query"],
        }


def collate_fn(batch):
    collated = {}
    for key in batch[0]:
        collated[key] = [item[key] for item in batch]
    return collated


@torch.no_grad()
def extract_text_features(model, tokenizer, text_list, device, max_length=64):
    """
    Extract SigLIP text features with padding removal.
    
    Changes vs original:
      - last_hidden_state is trimmed to remove padding tokens based on attention_mask,
        so the downstream model sees only actual text token representations.
    
    Note: Unlike CLIP (which applies text_projection to the EOS token), SigLIP has no
    separate projection layers. The 768-dim pooler_output is used directly.
    
    Returns list of dicts, each with:
        - last_hidden_state: (actual_len, hidden_dim) float16 numpy  (padding removed)
        - pooler_output: (hidden_dim,) float16 numpy
    """
    # Tokenize with padding to max_length (SigLIP was trained this way)
    inputs = tokenizer(
        text_list,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    attention_mask = inputs["attention_mask"]  # (B, L)
    
    # Get text model outputs
    text_outputs = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=attention_mask,
    )
    
    last_hidden_states = text_outputs.last_hidden_state  # (B, L, hidden_dim)
    pooler_outputs = text_outputs.pooler_output  # (B, hidden_dim)
    
    results = []
    for i in range(len(text_list)):
        # Strip padding tokens: keep only positions where attention_mask == 1
        mask_i = attention_mask[i]  # (L,)
        actual_len = mask_i.sum().item()
        lhs = last_hidden_states[i, :actual_len].cpu().to(torch.float16).numpy()  # (actual_len, dim)
        po = pooler_outputs[i].cpu().to(torch.float16).numpy()  # (hidden_dim,)
        results.append({
            "last_hidden_state": lhs,
            "pooler_output": po,
        })
    
    return results


def save_query_features(batch, batch_result, q_feat_dir, training=True):
    batch_size = len(batch["qid"])
    aug_ids = batch.get("aug_id", [0] * batch_size)
    
    for i, result in enumerate(batch_result):
        qid = batch["qid"][i]
        aug_id = aug_ids[i] if training else 0
        aug = f"_{aug_id}" if aug_id > 0 else ""
        
        q_feat_path = join(q_feat_dir, f"qid{qid}{aug}.npz")
        os.makedirs(os.path.dirname(q_feat_path), exist_ok=True)
        
        np.savez_compressed(
            q_feat_path,
            last_hidden_state=result["last_hidden_state"],
            pooler_output=result["pooler_output"],
        )


def extract_features_for_split(model, tokenizer, input_file, q_feat_dir, device, 
                                batch_size=32, training=True):
    print(f"Extracting features from: {input_file}")
    print(f"Saving to: {q_feat_dir}")
    os.makedirs(q_feat_dir, exist_ok=True)
    
    dataset = CharadesQueryDataset(input_file)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, collate_fn=collate_fn
    )
    
    for batch in tqdm(dataloader, desc="Extracting SigLIP Text Features"):
        batch_result = extract_text_features(model, tokenizer, batch["query"], device)
        save_query_features(batch, batch_result, q_feat_dir, training=training)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    from transformers import AutoModel, AutoTokenizer
    print(f"Loading SigLIP model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Extract train features
    extract_features_for_split(
        model, tokenizer,
        input_file=args.train_file,
        q_feat_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        training=True,
    )
    
    # Extract test features
    extract_features_for_split(
        model, tokenizer,
        input_file=args.test_file,
        q_feat_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        training=False,
    )
    
    # Verify output
    files = [f for f in os.listdir(args.output_dir) if f.endswith(".npz")]
    print(f"\nDone! Saved {len(files)} text feature files.")
    if files:
        sample = np.load(join(args.output_dir, files[0]))
        print(f"Sample shape: last_hidden_state={sample['last_hidden_state'].shape}, "
              f"pooler_output={sample['pooler_output'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SigLIP text features for Charades-STA")
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224",
                        help="HuggingFace SigLIP model name")
    parser.add_argument("--train_file", type=str,
                        default="data/charades_sta/charades_sta_train_tvr_format.jsonl",
                        help="Training data JSONL file")
    parser.add_argument("--test_file", type=str,
                        default="data/charades_sta/charades_sta_test_tvr_format.jsonl",
                        help="Test data JSONL file")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/charades/siglip_text_features/",
                        help="Output directory for .npz text feature files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    
    args = parser.parse_args()
    
    import transformers
    from packaging import version
    if version.parse(transformers.__version__) < version.parse("4.37.0"):
        print(f"ERROR: transformers {transformers.__version__} is too old for SigLIP.")
        print("Please upgrade: pip install transformers>=4.37.0")
        exit(1)
    
    main(args)
