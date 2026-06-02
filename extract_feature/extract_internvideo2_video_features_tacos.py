"""
Extract InternVideo2-L/14 video features for TACoS at 1 fps.

Output files follow TACoS annotation vid exactly:
    {vid}.npz

Each file contains:
    - features: (T, 768)

Video lookup is annotation-driven and supports TACoS source files named either
{vid}.mp4 or {vid}-cam-002.mp4.

Usage:
    cd /data1/zhangshihang/VideoLights
    python extract_feature/extract_internvideo2_video_features_tacos.py --gpu 7
"""

import argparse
import json
import os
import sys
from os.path import join

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_feature.extract_internvideo2_video_features_charades import (  # noqa: E402
    VideoProcessor,
    build_model,
    extract_video_features,
)


def load_jsonl(filename):
    with open(filename, "r") as file_obj:
        return [json.loads(line.strip("\n")) for line in file_obj.readlines()]


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


def is_file_present(directory, vid):
    return os.path.exists(join(directory, f"{vid}.npz"))


def slice_pending_list(pending_vids, start_idx, end_idx):
    if start_idx < 0:
        start_idx = 0
    if end_idx < 0 or end_idx > len(pending_vids):
        end_idx = len(pending_vids)
    return pending_vids[start_idx:end_idx]


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = build_model(args.checkpoint, num_frames=args.num_frames, device=device)
    video_loader = VideoProcessor(framerate=args.framerate, size=224, centercrop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    annotated_vids = collect_annotated_vids([args.train_file, args.val_file, args.test_file])
    pending_vids = [vid for vid in annotated_vids if not is_file_present(args.output_dir, vid)]
    pending_vids = slice_pending_list(pending_vids, args.start_idx, args.end_idx)

    print(f"Annotated videos: {len(annotated_vids)}, Pending in this shard: {len(pending_vids)}")

    failed = []
    for vid in tqdm(pending_vids, desc="InternVideo2 Video Features (TACoS)"):
        video_path = find_video_path(vid, args.input_dirs)
        if video_path is None:
            failed.append(vid)
            continue

        try:
            frames = video_loader.read_video_from_file(video_path)
            if frames is None or len(frames) == 0:
                failed.append(vid)
                continue

            features = extract_video_features(
                model,
                frames,
                num_frames=args.num_frames,
                device=device,
                batch_size=args.batch_size,
            )

            if features is not None:
                np.savez_compressed(join(args.output_dir, f"{vid}.npz"), features=features)
        except Exception as exc:
            print(f"Failed {vid}: {exc}")
            failed.append(vid)

    if failed:
        failed_path = join(args.output_dir, "failed_videos.txt")
        with open(failed_path, "w") as file_obj:
            for vid in failed:
                file_obj.write(f"{vid}\n")
        print(f"Saved failed video list to {failed_path}")

    print(f"Done! Processed {len(pending_vids) - len(failed)}/{len(pending_vids)} videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract InternVideo2-L/14 video features for TACoS")
    parser.add_argument("--checkpoint", type=str,
                        default="../InternVideo-main/InternVideo2/model/pytorch_model.bin",
                        help="Path to InternVideo2 checkpoint")
    parser.add_argument("--train_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/train.jsonl")
    parser.add_argument("--val_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/val.jsonl")
    parser.add_argument("--test_file", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/test.jsonl")
    parser.add_argument("--input_dirs", type=str, nargs="+",
                        default=["/data1/zhangshihang/Datasets/tacos"],
                        help="Directories containing TACoS video files")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/TACoS/internvideo2_video_features/",
                        help="Output directory for .npz feature files")
    parser.add_argument("--framerate", type=float, default=1.0,
                        help="Frame extraction rate (1.0 = 1 fps)")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames per model input")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for feature extraction")
    parser.add_argument("--start_idx", type=int, default=-1,
                        help="Start index into pending video list (inclusive). -1 = beginning")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index into pending video list (exclusive). -1 = end")
    main(parser.parse_args())