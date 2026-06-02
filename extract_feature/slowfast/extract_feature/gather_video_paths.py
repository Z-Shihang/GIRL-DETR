import os
import argparse
import json
COMMON_VIDEO_ETX = set([
    ".webm", ".mpg", ".mpeg", ".mpv", ".ogg",
    ".mp4", ".m4p", ".mpv", ".avi", ".wmv", ".qt",
    ".mov", ".flv", ".swf"])


def main(opts):
    videopath = opts.video_path
    feature_path = opts.feature_path
    csv_folder = opts.csv_folder
    if not os.path.exists(csv_folder):
        os.mkdir(csv_folder)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    sf_feature_path = os.path.join(feature_path, "sf_features")
    clip_feature_path = os.path.join(feature_path, "clip_features")
    blip_feature_path = os.path.join(feature_path, "blip_features")
    if not os.path.exists(sf_feature_path):
        os.mkdir(sf_feature_path)
    if not os.path.exists(clip_feature_path):
        os.mkdir(clip_feature_path)
    if not os.path.exists(blip_feature_path):
        os.mkdir(blip_feature_path)
    if os.path.exists(opts.corrupted_id_file):
        corrupted_ids = set(json.load(
            open(opts.corrupted_id_file, 'r')))
    else:
        corrupted_ids = None

    outputFile = f"{csv_folder}/slowfast_info.csv"
    with open(outputFile, "w") as fw:
        fw.write("video_path,sf_feature_path,clip_feature_path,blip_feature_path\n")
        fileList = []
        for dirpath, _, files in os.walk(videopath):
            for fname in files:
                input_file = os.path.join(dirpath, fname)
                if os.path.isfile(input_file):
                    _, ext = os.path.splitext(fname)
                    if ext.lower() in COMMON_VIDEO_ETX:
                        fileList.append(input_file)

        for input_filename in fileList:
            filename = os.path.basename(input_filename)
            fileId, _ = os.path.splitext(filename)

            sf_output_filename = os.path.join(sf_feature_path, fileId+".npz")
            clip_output_filename = os.path.join(clip_feature_path, fileId+".npz")
            blip_output_filename = os.path.join(blip_feature_path, fileId+".npz")
            if (not os.path.exists(sf_output_filename)) \
                or (not os.path.exists(clip_output_filename)) \
                or (not os.path.exists(blip_output_filename)):
                fw.write(input_filename+","+sf_output_filename+","+clip_output_filename+","+blip_output_filename+"\n")
            if corrupted_ids is not None and fileId in corrupted_ids:
                fw.write(input_filename+","+sf_output_filename+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="/video/", type=str,
                        help="The input video path.")
    parser.add_argument("--feature_path", default="/output/slowfast_features",
                        type=str, help="output feature path.")
    parser.add_argument(
        '--csv_folder', type=str, default="/output/csv",
        help='output csv folder')
    parser.add_argument(
        '--corrupted_id_file', type=str, default="",
        help='corrupted id file')
    args = parser.parse_args()
    main(args)
