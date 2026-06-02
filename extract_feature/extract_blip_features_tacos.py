"""
Extract BLIP base text and video features for TACoS at 1 fps.

Output files:
    - Text:  {qid}.npz with last_hidden_state and pooler_output
    - Video: {vid}.npz with features

Video lookup is annotation-driven and supports TACoS source files named either
{vid}.mp4 or {vid}-cam-002.mp4.

Usage:
    cd /data1/zhangshihang/VideoLights
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python extract_feature/extract_blip_features_tacos.py --gpu 5
"""

import argparse
import json
import os
import sys
from os.path import join

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_on_video.data_utils import VideoProcessor


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
        return {"qid": item["qid"], "query": item["query"], "vid": item["vid"]}


def collate_fn(batch):
    collated_dict = {}
    for key in batch[0]:
        collated_dict[key] = [data[key] for data in batch]
    return collated_dict


def collect_annotated_vids(annotation_files):
    vid_set = set()
    for path in annotation_files:
        if not os.path.exists(path):
            continue
        for item in load_jsonl(path):
            vid_set.add(item["vid"])
    return sorted(vid_set)


def build_video_candidates(vid):
    base_vid = vid[:-8] if vid.endswith("-cam-002") else vid
    return [
        f"{base_vid}.mp4",
        f"{base_vid}-cam-002.mp4",
        f"{base_vid}.avi",
        f"{base_vid}-cam-002.avi",
    ]


def find_video_path(vid, input_dirs):
    for input_dir in input_dirs:
        for candidate in build_video_candidates(vid):
            path = join(input_dir, candidate)
            if os.path.exists(path):
                return path
    return None


def is_file_present(directory, name):
    return os.path.exists(join(directory, f"{name}.npz"))


def encode_text_query(model, txt_processors, batch):
    batch_output = []
    with torch.no_grad():
        for text in batch:
            text_input = txt_processors["eval"](text)
            sample = {"text_input": [text_input]}
            batch_output.append(model.extract_features(sample, mode="text"))
    return batch_output


@torch.no_grad()
def encode_video(model, vis_processors, video_loader, video_path, device, batch_size=16):
    video_frames, _ = video_loader.read_raw_image_from_video_file(video_path)
    n_frames = len(video_frames)
    video_features = []
    for index in range(0, n_frames, batch_size):
        batch_frames = video_frames[index:index + batch_size]
        batch_tensors = torch.stack([vis_processors["eval"](frame) for frame in batch_frames]).to(device)
        sample = {"image": batch_tensors}
        features_image = model.extract_features(sample, mode="image")
        video_features.append(features_image.image_embeds[:, 0, :])
    return torch.cat(video_features, dim=0)


def save_query_features(batch, batch_result, output_dir):
    for index, result in enumerate(batch_result):
        qid = batch["qid"][index]
        output_path = join(output_dir, f"{qid}.npz")

        text_embeds = result.text_embeds.squeeze()
        pooler_output = text_embeds[0, :]
        np.savez_compressed(
            output_path,
            last_hidden_state=text_embeds.cpu().numpy(),
            pooler_output=pooler_output.cpu().numpy(),
        )


def extract_text_split(model, txt_processors, input_file, output_dir, batch_size):
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

    print(f"Extracting TACoS BLIP text features from {input_file}")
    for batch in tqdm(dataloader, desc="BLIP Text Features (TACoS)"):
        batch_result = encode_text_query(model, txt_processors, batch["query"])
        save_query_features(batch, batch_result, output_dir)


def slice_pending_list(pending_vids, start_idx, end_idx):
    if start_idx < 0:
        start_idx = 0
    if end_idx < 0 or end_idx > len(pending_vids):
        end_idx = len(pending_vids)
    return pending_vids[start_idx:end_idx]


def extract_video_split(model, vis_processors, video_loader, device, args):
    annotated_vids = collect_annotated_vids([args.train_file, args.val_file, args.test_file])
    pending_vids = [vid for vid in annotated_vids if not is_file_present(args.video_output_dir, vid)]
    pending_vids = slice_pending_list(pending_vids, args.start_idx, args.end_idx)

    print(f"Annotated videos: {len(annotated_vids)}, Pending in this shard: {len(pending_vids)}")
    failed = []

    for vid in tqdm(pending_vids, desc="BLIP Video Features (TACoS)"):
        video_path = find_video_path(vid, args.input_dirs)
        if video_path is None:
            failed.append(vid)
            continue
        try:
            video_feature = encode_video(
                model,
                vis_processors,
                video_loader,
                video_path,
                device,
                batch_size=args.video_batch_size,
            )
            np.savez_compressed(join(args.video_output_dir, f"{vid}.npz"), features=video_feature.cpu().numpy())
        except Exception as exc:
            print(f"Failed {vid}: {exc}")
            failed.append(vid)

    if failed:
        failed_path = join(args.video_output_dir, "failed_videos.txt")
        with open(failed_path, "w") as file_obj:
            for vid in failed:
                file_obj.write(f"{vid}\n")
        print(f"Saved failed video list to {failed_path}")

    print(f"Done! Processed {len(pending_vids) - len(failed)}/{len(pending_vids)} BLIP videos.")


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.text_output_dir, exist_ok=True)
    os.makedirs(args.video_output_dir, exist_ok=True)

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_feature_extractor",
        model_type="base",
        is_eval=True,
        device=device,
    )
    video_loader = VideoProcessor(framerate=args.framerate, size=224, centercrop=True)

    if args.mode in ["all", "text"]:
        extract_text_split(model, txt_processors, args.train_file, args.text_output_dir, args.text_batch_size)
        extract_text_split(model, txt_processors, args.val_file, args.text_output_dir, args.text_batch_size)
        extract_text_split(model, txt_processors, args.test_file, args.text_output_dir, args.text_batch_size)

    if args.mode in ["all", "video"]:
        extract_video_split(model, vis_processors, video_loader, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract BLIP base text and video features for TACoS")
    parser.add_argument("--train_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/train.jsonl")
    parser.add_argument("--val_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/val.jsonl")
    parser.add_argument("--test_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/test.jsonl")
    parser.add_argument("--input_dirs", type=str, nargs="+",
                        default=["/data1/zhangshihang/Datasets/tacos"],
                        help="Directories containing TACoS video files")
    parser.add_argument("--text_output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/blip_text_features/")
    parser.add_argument("--video_output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/blip_video_features/")
    parser.add_argument("--mode", type=str, choices=["all", "text", "video"], default="all")
    parser.add_argument("--text_batch_size", type=int, default=32)
    parser.add_argument("--video_batch_size", type=int, default=16)
    parser.add_argument("--framerate", type=float, default=1.0,
                        help="Frame extraction rate (1.0 = 1 fps)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--start_idx", type=int, default=-1,
                        help="Start index into pending video list (inclusive). -1 = beginning")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index into pending video list (exclusive). -1 = end")
    main(parser.parse_args())