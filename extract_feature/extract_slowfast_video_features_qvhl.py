"""
Extract SlowFast R50 video features for QVHighlights.
Collects videos from multiple directories (train/val/test/videos),
skips already-extracted, then calls the existing SlowFast extraction pipeline.

Usage:
    cd {project_root}
    python extract_feature/extract_slowfast_video_features_qvhl.py
"""

import os
import sys
import subprocess
from os.path import join

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

v_input_dirs = [
    os.environ.get("VIDEO_TRAIN_DIR", os.path.join(project_root, "data/QV_highlight/train/")),
    os.environ.get("VIDEO_VAL_DIR", os.path.join(project_root, "data/QV_highlight/val/")),
    os.environ.get("VIDEO_TEST_DIR", os.path.join(project_root, "data/QV_highlight/test/")),
    os.environ.get("VIDEO_FULL_DIR", os.path.join(project_root, "data/QV_highlight/videos/")),
]
output_dir = os.environ.get("OUTPUT_DIR", os.path.join(project_root, "data/qvhl/features/slowfast_video_features/"))

BATCH_SIZE = 16
CLIP_LEN = "1"  # 1 second per clip = 1 fps


def main():
    os.makedirs(output_dir, exist_ok=True)

    # Collect all videos from multiple directories, deduplicate by name
    all_videos = {}
    for vdir in v_input_dirs:
        if os.path.exists(vdir):
            for f in os.listdir(vdir):
                if f.endswith(".mp4"):
                    vid_name = os.path.splitext(f)[0]
                    if vid_name not in all_videos:
                        all_videos[vid_name] = join(vdir, f)

    # Filter out already extracted
    pending = {vid: path for vid, path in all_videos.items()
               if not os.path.exists(join(output_dir, f"{vid}.npz"))}

    if not pending:
        print("All SlowFast video features already extracted!")
        return

    print(f"Total videos: {len(all_videos)}, Pending: {len(pending)}")

    # Generate temporary CSV for VideoLoader
    # Format: video_path,sf_feature_path,clip_feature_path,blip_feature_path
    csv_path = join(project_root, "temp_qvhl_slowfast.csv")
    with open(csv_path, "w") as f:
        f.write("video_path,sf_feature_path,clip_feature_path,blip_feature_path\n")
        for vid in sorted(pending):
            vpath = pending[vid]
            out_path = join(output_dir, f"{vid}.npz")
            # clip/blip columns are unused by SlowFast extraction, use dummy paths
            f.write(f"{vpath},{out_path},{out_path},{out_path}\n")

    # Call existing SlowFast extraction
    cmd = [
        sys.executable, "-m",
        "extract_feature.slowfast.extract_feature.extract",
        "--csv", csv_path,
        "--batch_size", str(BATCH_SIZE),
        "--clip_len", CLIP_LEN,
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    print("Done!")


if __name__ == "__main__":
    main()
