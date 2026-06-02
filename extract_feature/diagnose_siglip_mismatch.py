"""
Diagnose SigLIP feature re-extraction mismatches.

Checks:
1. Whether failing videos exist in multiple source directories
2. Extracts from each available source and compares with saved features
3. Reports which source directory (if any) matches the saved features

Usage:
    python extract_feature/diagnose_siglip_mismatch.py --gpu 0 --num_test 10
"""
import os, sys, random, argparse
import numpy as np
import torch
from PIL import Image
import ffmpeg, math

SIGLIP_DIR = "/data1/zhangshihang/Datasets/charades/siglip_video_features/"
VIDEO_DIRS = [
    "/data1/zhangshihang/data/Charades/Charades_Videos/train/",
    "/data1/zhangshihang/data/Charades/Charades_Videos/test/",
    "/data1/zhangshihang/data/Charades/Charades_v1_480/",
]


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        return float(num) / float(denom)


def extract_frames(video_path, framerate=1.0, size=224):
    """Same frame extraction as the original extraction script's VideoProcessor."""
    probe = ffmpeg.probe(video_path)
    vs = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    h, w = int(vs['height']), int(vs['width'])

    if h >= w:
        H, W = int(h * size / w), size
    else:
        H, W = size, int(w * size / h)

    fps = framerate
    try:
        dur = float(vs['duration'])
        if dur > 0 and dur < 1.0 / fps + 0.1:
            fps = 2 / max(int(dur), 1)
    except Exception:
        fps = framerate

    cmd = (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=fps)
        .filter('scale', W, H)
    )
    x = int((W - size) / 2)
    y = int((H - size) / 2)
    cmd = cmd.crop(x, y, size, size)
    out, _ = cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(
        capture_stdout=True, quiet=True
    )
    frames = np.frombuffer(out, np.uint8).reshape([-1, size, size, 3])
    return frames, fps, (h, w)


@torch.no_grad()
def extract_features(frames_np, model, processor, device):
    """Extract SigLIP features from numpy frames (T, H, W, 3) uint8."""
    pil_imgs = [Image.fromarray(fr) for fr in frames_np]
    inputs = processor(images=pil_imgs, return_tensors="pt")
    out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
    return out.pooler_output.cpu().to(torch.float16).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_test", type=int, default=10,
                        help="Number of random videos to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--specific_vids", type=str, nargs="*", default=None,
                        help="Specific video IDs to check (e.g., 8YZ8X ABOBW)")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModel, AutoProcessor
    model_name = "google/siglip-base-patch16-224"
    print(f"Loading {model_name} on {device}...")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    # Step 1: Gather all feature files
    all_files = sorted([f.replace('.npz', '') for f in os.listdir(SIGLIP_DIR) if f.endswith('.npz')])
    print(f"Total saved features: {len(all_files)}")

    # Step 2: For each video, find all directories it exists in
    print("\n" + "=" * 70)
    print("STEP 1: Checking multi-directory presence for all videos")
    print("=" * 70)
    multi_dir_count = 0
    multi_dir_vids = []
    for vid in all_files:
        dirs_found = []
        for d in VIDEO_DIRS:
            p = os.path.join(d, f"{vid}.mp4")
            if os.path.exists(p):
                dirs_found.append(d)
        if len(dirs_found) > 1:
            multi_dir_count += 1
            multi_dir_vids.append(vid)
    print(f"Videos in multiple directories: {multi_dir_count}/{len(all_files)}")
    if multi_dir_count > 0 and multi_dir_count <= 20:
        for vid in multi_dir_vids:
            dirs = [d for d in VIDEO_DIRS if os.path.exists(os.path.join(d, f"{vid}.mp4"))]
            print(f"  {vid}: {dirs}")

    # Step 3: Test specific or random videos
    print("\n" + "=" * 70)
    print("STEP 2: Re-extraction comparison")
    print("=" * 70)

    if args.specific_vids:
        test_vids = args.specific_vids
    else:
        # Pick random videos, preferring multi-dir ones if they exist
        test_vids = []
        if multi_dir_vids:
            test_vids = random.sample(multi_dir_vids, min(5, len(multi_dir_vids)))
        single_dir = [v for v in all_files if v not in multi_dir_vids]
        remaining = args.num_test - len(test_vids)
        if remaining > 0:
            test_vids += random.sample(single_dir, min(remaining, len(single_dir)))

    match_count = 0
    mismatch_count = 0
    for vid in test_vids:
        saved = np.load(os.path.join(SIGLIP_DIR, f"{vid}.npz"))['features']
        dirs_found = [d for d in VIDEO_DIRS if os.path.exists(os.path.join(d, f"{vid}.mp4"))]

        if not dirs_found:
            print(f"\n  {vid}: ❌ no video found in any directory!")
            continue

        print(f"\n  {vid}: saved shape={saved.shape}, found in {len(dirs_found)} dir(s)")

        for d in dirs_found:
            video_path = os.path.join(d, f"{vid}.mp4")
            try:
                frames, fps, orig_res = extract_frames(video_path)
                fresh = extract_features(frames, model, processor, device)
                max_diff = np.abs(fresh.astype(np.float32) - saved.astype(np.float32)).max()
                mean_diff = np.abs(fresh.astype(np.float32) - saved.astype(np.float32)).mean()
                dir_short = os.path.basename(os.path.normpath(d))

                if np.array_equal(fresh, saved):
                    print(f"    [{dir_short}] ✅ bit-exact match "
                          f"(frames={fresh.shape[0]}, orig_res={orig_res})")
                    match_count += 1
                elif max_diff < 1e-3:
                    print(f"    [{dir_short}] ⚠️ nearly identical max_diff={max_diff:.6f} "
                          f"(frames={fresh.shape[0]}, orig_res={orig_res})")
                    match_count += 1
                else:
                    print(f"    [{dir_short}] ❌ MISMATCH max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} "
                          f"(frames: fresh={fresh.shape[0]} saved={saved.shape[0]}, orig_res={orig_res})")
                    mismatch_count += 1

                    # Debug: compare first frame pixels
                    if fresh.shape == saved.shape:
                        frame_diffs = np.abs(fresh.astype(np.float32) - saved.astype(np.float32)).max(axis=1)
                        worst_frame = np.argmax(frame_diffs)
                        print(f"      Worst frame: {worst_frame} (max_diff={frame_diffs[worst_frame]:.4f})")
                        print(f"      Per-frame max_diffs (first 10): {frame_diffs[:10].round(4)}")

            except Exception as e:
                print(f"    [{os.path.basename(os.path.normpath(d))}] Error: {e}")

    print(f"\n{'='*70}")
    print(f"Summary: {match_count} match, {mismatch_count} mismatch")
    if mismatch_count > 0:
        print("\nPossible causes:")
        print("  1. Video was sourced from different directory during original extraction")
        print("  2. Different transformers/torch version between extraction and verification")
        print("  3. Features corrupted during extraction (partial run / interrupted)")
        print("\nRecommendation: re-extract all features with the same script+environment")
    print("=" * 70)


if __name__ == "__main__":
    main()
