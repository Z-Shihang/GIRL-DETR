import torch as th
import numpy as np
from extract_feature.slowfast.extract_feature.video_loader import (
    VideoLoader, clip_iterator, pack_pathway_output)
from torch.utils.data import DataLoader
import argparse
from extract_feature.slowfast.extract_feature.model import build_model
from extract_feature.slowfast.extract_feature.preprocessing import Preprocessing, Normalize
from extract_feature.slowfast.extract_feature.random_sequence_shuffler import RandomSequenceSampler
from extract_feature.slowfast.slowfast.config.defaults import get_cfg
import extract_feature.slowfast.slowfast.utils.checkpoint as cu
from tqdm import tqdm
from extract_feature.slowfast.extract_feature.prefetch_loader import PrefetchLoader
import sys
import os
import time
from extract_feature.slowfast.extract_feature.yuv_reader import YuvRgbConverter

FEATURE_LENGTH = 2304
YUV2RGB = YuvRgbConverter()

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Easy video feature extractor')

    parser.add_argument(
        '--csv',
        type=str,
        help='input csv with video input path')
    parser.add_argument(
        '--folder_path',
        type=str,
        help='Input folder path containing videos')
    parser.add_argument(
        '--output_folder',
        type=str,
        help='Output folder path for features')

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="extract_feature/slowfast/configs/Kinetics/c2/extract_SLOWFAST_8x8_R50.yaml",
        type=str,
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument(
        '--half_precision', type=int, default=1,
        help='output half precision float')
    parser.add_argument(
        '--dataflow', action='store_true',
        help='enable dataflow')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='allow overwrite output files')
    parser.add_argument(
        '--num_decoding_thread', type=int, default=0,
        help='Num parallel thread for video decoding')
    parser.add_argument(
        '--target_framerate', type=int, default=30,
        help='decoding frame per second')
    parser.add_argument(
        '--clip_len', type=str, default='1',
        help='decoding length of clip (in seconds)')
    parser.add_argument(
        '--min_num_features', type=int, default=1,
        help='minimum number of features')
    parser.add_argument(
        '--pix_fmt', type=str, default="rgb24", choices=["rgb24", "yuv420p"],
        help='decode video into RGB format')
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


@th.no_grad()
def extract_features(video_loader, sf_model, cfg, args, failed_log, n_dataset):
    sf_model.eval()
    norm = Normalize(
        mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    totatl_num_frames = 0
    total_time = 0
    pbar = tqdm(total=n_dataset)

    for _, data in enumerate(video_loader):
        input_file = data['input']
        
        if 'sf_output' in data:
            sf_output_file = data['sf_output']
        elif 'sf_feature_path' in data:
            sf_output_file = data['sf_feature_path']
        else:
            sf_output_file = list(data.values())[1] 
            
        if isinstance(input_file, (list,)):
            input_file = input_file[0]
            sf_output_file = sf_output_file[0]
            
        video = data['video']
        video_shape_len = len(video.shape)
        if video_shape_len == 6:
            video = video.squeeze(0)
        video_shape_len = len(video.shape)
        
        if video_shape_len == 5:
            n_chunk = len(video)
            print(f"\nProcessing {input_file} n_chunk {n_chunk}")

            sf_features = th.cuda.HalfTensor(
                n_chunk, FEATURE_LENGTH).fill_(0)
            clip_loader = PrefetchLoader(clip_iterator(video, args.batch_size))

            for _, (min_ind, max_ind, fast_clip) in enumerate(clip_loader):
                fast_clip = fast_clip.float()
                if args.pix_fmt == "yuv420p":
                    fast_clip = YUV2RGB(fast_clip)
                fast_clip = fast_clip.permute(0, 4, 1, 2, 3)
                fast_clip = fast_clip / 255.
                fast_clip = norm(fast_clip)
                inputs = pack_pathway_output(cfg, fast_clip)
                
                th.cuda.synchronize()
                start_time = time.time()
                batch_features = sf_model(inputs)
                th.cuda.synchronize()
                end_time = time.time()
                total_time += end_time - start_time
                sf_features[min_ind:max_ind] = batch_features.half()

            sf_features = sf_features.cpu().numpy().astype('float16')
            totatl_num_frames += sf_features.shape[0]

            save_video_feature(sf_features, sf_output_file)

        elif os.path.isfile(sf_output_file):
            print(f'\nVideo {input_file} already processed.')
        elif not os.path.isfile(input_file):
            failed_log.write(f'\n{input_file}, does not exist.\n')
        else:
            failed_log.write(f'\n{input_file}, failed at ffprobe.\n')
        pbar.update(1)

    print(f"Total number of frames: {totatl_num_frames}")
    print(f"Model inference time: {total_time}")


def save_video_feature(result, result_save_path):
    dirname = os.path.dirname(result_save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    try:
        np.savez_compressed(result_save_path, features=result)
    except Exception as e:
        print(e)
        print(result_save_path)


def main():
    opts = parse_args()
    cfg = load_config(opts)
    np.random.seed(cfg.RNG_SEED)
    th.manual_seed(cfg.RNG_SEED)

    temp_csv_created = False
    
    if opts.folder_path:
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        temp_csv_name = "temp_charades_list.csv"
        
        valid_files = []
        for root, dirs, files in os.walk(opts.folder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    full_input_path = os.path.join(root, file)
                    
                    rel_path = os.path.relpath(full_input_path, opts.folder_path)
                    filename_no_ext = os.path.splitext(os.path.basename(file))[0]
                    
                    output_filename = filename_no_ext + ".npz"
                    
                    rel_dir = os.path.dirname(rel_path)
                    full_output_path = os.path.join(opts.output_folder, rel_dir, output_filename)
                    
                    valid_files.append((full_input_path, full_output_path))
        
        if not valid_files:
            print(f"No video files found in {opts.folder_path}")
            return

        with open(temp_csv_name, 'w') as f:
            f.write("video_path,sf_feature_path,clip_feature_path,blip_feature_path\n")
            for v_path, o_path in valid_files:
                f.write(f"{v_path},{o_path},{o_path},{o_path}\n")
        
        print(f"Found {len(valid_files)} videos. Processing...")
        opts.csv = temp_csv_name
        temp_csv_created = True
        failed_log_path = "charades_failed.txt"
    else:
        if not opts.csv:
            print("Please provide --csv or --folder_path")
            return
        failed_log_path = opts.csv.split(".csv")[0] + "_failed.txt"

    failed_log = open(failed_log_path, "w")
    assert opts.target_framerate % opts.min_num_features == 0

    preprocess = Preprocessing(
        "3d", cfg, target_fps=opts.target_framerate,
        size=224, clip_len=opts.clip_len, padding_mode='tile',
        min_num_clips=opts.min_num_features)
    
    dataset = VideoLoader(
        opts.csv,
        preprocess,
        framerate=opts.target_framerate,
        size=224,
        centercrop=True,
        pix_fmt=opts.pix_fmt,
        overwrite=opts.overwrite
    )
    n_dataset = len(dataset)
    sampler = RandomSequenceSampler(n_dataset, 10)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opts.num_decoding_thread,
        sampler=sampler if n_dataset > 10 else None,
    )

    sf_model = build_model(cfg)

    try:
        extract_features(loader, sf_model, cfg, opts, failed_log, n_dataset)
    finally:
        if temp_csv_created and os.path.exists(opts.csv):
            os.remove(opts.csv)
        failed_log.close()

if __name__ == "__main__":
    main()