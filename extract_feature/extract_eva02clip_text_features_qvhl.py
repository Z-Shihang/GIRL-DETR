"""
Extract EVA02-CLIP-B/16 text features for QVHighlights.
Adapted from extract_eva02clip_text_features_charades.py.

Output: .npz files with keys:
    - last_hidden_state: (context_length, 512) - token-level features
    - pooler_output: (512,) - EOS token feature after text_projection

Handles train (paraphrased with aug_id), val, and test splits.

Requires: pip install open_clip_torch

Usage:
    python extract_feature/extract_eva02clip_text_features_qvhl.py --gpu 0
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
    collated = {}
    for key in batch[0]:
        collated[key] = [item[key] for item in batch]
    return collated


@torch.no_grad()
def encode_text_with_hidden_states(model, text_tokens, device):
    """
    Forward through EVA02-CLIP text encoder, returning both
    token-level hidden states and projected pooler output.
    """
    text_tokens = text_tokens.to(device)

    text_enc = model.text if hasattr(model, 'text') and hasattr(model.text, 'transformer') else model

    x, attn_mask = text_enc._embeds(text_tokens)
    x = text_enc.transformer(x, attn_mask=attn_mask)
    x = text_enc.ln_final(x)

    last_hidden_state = x

    from open_clip.transformer import text_global_pool
    pooler_output = text_global_pool(
        x, text_tokens, pool_type=text_enc.pool_type,
        eos_token_id=getattr(text_enc, "eos_id", None)
    )

    text_proj = text_enc.text_projection
    if text_proj is not None:
        if isinstance(text_proj, torch.nn.Linear):
            pooler_output = text_proj(pooler_output)
        else:
            pooler_output = pooler_output @ text_proj

    return last_hidden_state.float(), pooler_output.float()


def save_features(batch, last_hidden_states, pooler_outputs, output_dir, training=True):
    batch_size = len(batch["qid"])
    aug_ids = batch.get("aug_id", [0] * batch_size)

    for i in range(batch_size):
        qid = batch["qid"][i]
        aug_id = aug_ids[i] if training else 0
        aug = f"_{aug_id}" if aug_id > 0 else ""

        path = join(output_dir, f"qid{qid}{aug}.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez_compressed(
            path,
            last_hidden_state=last_hidden_states[i].cpu().numpy(),
            pooler_output=pooler_outputs[i].cpu().numpy(),
        )


def extract_split(model, tokenizer, input_file, output_dir, device,
                   batch_size=256, training=True):
    print(f"Extracting: {input_file} -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    dataset = QVHQueryDataset(input_file)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    for batch in tqdm(loader, desc="EVA02-CLIP Text Features"):
        text_tokens = tokenizer(batch["query"])
        last_hidden_states, pooler_outputs = encode_text_with_hidden_states(
            model, text_tokens, device
        )
        save_features(batch, last_hidden_states, pooler_outputs, output_dir, training)


def load_eva02clip(device, weights_path=None):
    import open_clip

    default_cache = os.path.expanduser(
        '~/.cache/modelscope/hub/models/timm/'
        'eva02_base_patch16_clip_224___merged2b_s8b_b131k/open_clip_pytorch_model.bin'
    )
    if weights_path is None:
        weights_path = default_cache

    if os.path.isfile(weights_path):
        print(f"Loading EVA02-CLIP-B/16 from local cache: {weights_path}")
        model, _, _ = open_clip.create_model_and_transforms('EVA02-B-16', pretrained='')
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print(f"Local cache not found at {weights_path}, downloading...")
        model, _, _ = open_clip.create_model_and_transforms(
            'EVA02-B-16', pretrained='merged2b_s8b_b131k'
        )

    model = model.to(device).eval()
    return model


def main(args):
    import open_clip

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = load_eva02clip(device, args.weights_path)
    tokenizer = open_clip.get_tokenizer('EVA02-B-16')

    # Verify dimensions
    test_tokens = tokenizer(["test sentence"]).to(device)
    lhs, po = encode_text_with_hidden_states(model, test_tokens, device)
    print(f"Text hidden dim: {lhs.shape[-1]}, Pooler dim: {po.shape[-1]}, "
          f"Context length: {lhs.shape[1]}")
    assert po.shape[-1] == 512, f"Expected 512-dim pooler but got {po.shape[-1]}"

    # Train split (paraphrased, with aug_id)
    if os.path.exists(args.train_file):
        extract_split(
            model, tokenizer, args.train_file, args.output_dir,
            device, args.batch_size, training=True
        )

    # Val split
    if os.path.exists(args.val_file):
        extract_split(
            model, tokenizer, args.val_file, args.output_dir,
            device, args.batch_size, training=False
        )

    # Test split
    if os.path.exists(args.test_file):
        extract_split(
            model, tokenizer, args.test_file, args.output_dir,
            device, args.batch_size, training=False
        )

    # Verify output
    files = [f for f in os.listdir(args.output_dir) if f.endswith(".npz")]
    print(f"\nDone! Saved {len(files)} text feature files.")
    if files:
        sample = np.load(join(args.output_dir, files[0]))
        print(f"Sample: last_hidden_state={sample['last_hidden_state'].shape}, "
              f"pooler_output={sample['pooler_output'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract EVA02-CLIP-B/16 text features for QVHighlights"
    )
    parser.add_argument("--train_file", type=str,
                        default="data/highlight_train_release_paraphrased_openai.jsonl")
    parser.add_argument("--val_file", type=str,
                        default="data/highlight_val_release.jsonl")
    parser.add_argument("--test_file", type=str,
                        default="data/highlight_test_release.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/qvhl/features/eva02clip_text_features/",
                        help="Output directory for .npz text feature files")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--weights_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
