import json
import math
import os
from os.path import join
from threading import Thread

import numpy as np
import torch
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

from run_on_video.data_utils import VideoProcessor, ClipFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
#                                                                   model_type="pretrain", is_eval=True,
#                                                                   device=device)  # Blip2 featrures
# {
#   "qid": "x68guk71VFo_360.0_510.0_subs56",
#   "query": "yeah",
#   "vid": "x68guk71VFo_360.0_510.0",
#   "duration": 150,
#   "split": "train",
#   "relevant_windows": [[97.919, 98.39]]
# }

# model_extract, vis_processors_extract, txt_processors_extract = load_model_and_preprocess(name="blip2_feature_extractor",
#                                                                   model_type="pretrain", is_eval=True,
#                                                                   device=device)  # Blip2 featrures

model_sim, vis_processors_sim, txt_processors_sim = load_model_and_preprocess("blip_image_text_matching", "base",
                                                                              device=device, is_eval=True)
# model_extract, vis_processors_extract, txt_processors_extract = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)

# we associate a model with its preprocessors to make it easier for inference.
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device
)
# uncomment to use base model
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="base_coco", is_eval=True, device=device
# )
# vis_processors.keys()

# we associate a model with its preprocessors to make it easier for inference.
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
# )

# Other available models:
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
# )

video_loader = VideoProcessor(framerate=0.1, size=224, centercrop=True)
clip_loader = VideoProcessor(framerate=0.5, size=224, centercrop=True)
# clip_extractor = ClipFeatureExtractor()

v_input_dir = "../videos/pretrain_videos/"
pre_train_text_dir = "../Datasets/processed/charades/blip_pre_train_q_feat_dir/"

pretrain_jsonl_path = "data/pretrain/pre_train_blip_charades.jsonl"


# pretrain_jsonl_path_charades = "data/charades_sta/pre_train_blip_qvhl_pld.jsonl"
# pretrain_jsonl_path_new_charades = "data/charades_sta/pre_train_blip_new.jsonl"

# val_data_path_qvhl = "data/highlight_val_release.jsonl"
# test_data_path_qvhl = "data/highlight_test_release.jsonl"
# test_data_path_charades = "data/charades_sta/charades_sta_test_tvr_format.jsonl"


@torch.no_grad()
def encode_video(input_dir: str, vid: str, ext='.mp4'):
    video_path = join(input_dir, f"{vid}{ext}")
    video_frames, info = video_loader.read_raw_image_from_video_file(video_path)  # (T, H, W, 3)
    # clip_video_features = clip_extractor.encode_video(video_path)
    clips, _ = clip_loader.read_raw_image_from_video_file(video_path)  # (T, H, W, 3)
    split = "train"
    duration = math.floor(info["duration"])
    n_frames = len(video_frames)
    frame_duration = duration / n_frames
    start_duration = 0
    train_data = []
    if duration > 500:
        return train_data
    for i in range(n_frames):
        qid = f"{vid}_{start_duration}_{start_duration + frame_duration}"
        image = vis_processors["eval"](video_frames[i]).unsqueeze(0).to(device)
        query = model.generate({"image": image})[0]
        relevant_windows = [[start_duration, start_duration + frame_duration]]
        relevant_clip_ids, saliency_scores = get_relevant_window(clips, query, start_duration,
                                                                 start_duration + frame_duration)
        start_duration += frame_duration
        dict_data = {
            "qid": qid,
            "query": query,
            "vid": vid.replace("-cam-002", ""),
            "duration": duration,
            "split": split,
            "relevant_windows": relevant_windows,
            "relevant_clip_ids": relevant_clip_ids.tolist(),
            "saliency_scores": saliency_scores.tolist()
        }
        train_data.append(dict_data)
    return train_data


@torch.no_grad()
def get_relevant_window(clips, query, start_time, end_time):
    start_index = math.floor(start_time / 2)
    end_index = math.ceil(end_time / 2)
    if end_index == len(clips):
        end_index = len(clips) - 1
    relevant_clips = clips[start_index:end_index]
    similarity = torch.zeros(len(relevant_clips), dtype=torch.float)
    for index, clip in enumerate(relevant_clips):
        img = vis_processors_sim["eval"](clip).unsqueeze(0).to(device)
        txt = txt_processors_sim["eval"](query)
        itm_output = model_sim({"image": img, "text_input": txt}, match_head="itm")
        similarity[index] = F.softmax(itm_output, dim=1)[:, 1].item()
        # similarity = get_similarity(relevant_clips, query)
    similarity = torch.clamp(similarity * 2.5,
                             min=1)  # torch.clamp(similarity, min=0) * 2.5 #similarity[similarity > 0.1]
    scaled_similarity = torch.clamp(torch.exp(similarity), min=2)

    return torch.arange(start_index, end_index, 1), scaled_similarity.unsqueeze(1)


# @torch.no_grad()
# def get_similarity(video_frames, query):
#     n_frames = len(video_frames)
#     video_features = []
#     for i in range(n_frames):
#         image = vis_processors_extract["eval"](video_frames[i]).unsqueeze(0).to(device)
#         sample = {"image": image}
#         features_image = model_extract.extract_features(sample, mode="image")
#         video_features.append(features_image.image_embeds[:, 0, :])
#     video_features = torch.cat(video_features, dim=0)
#
#     text_input = txt_processors_extract["eval"](query)
#     sample = {"text_input": [text_input]}
#     query_features = model_extract.extract_features(sample, mode="text")
#
#     return F.cosine_similarity(video_features, query_features.unsqueeze(0), dim=1)

def read_all_files_from_directory(directory_path, exlude_vids=None):
    if exlude_vids is None:
        exlude_vids = []
    vid_to_process = [f for f in os.listdir(directory_path)
                      if os.path.isfile(os.path.join(directory_path, f))
                      and os.path.splitext(f)[0].replace("-cam-002", "") not in exlude_vids]
    return vid_to_process


def generate_pretrain_data(input_dir, pretrain_jsonl_path, t_feat_dir, exclude_vids=None, train_data=None):
    if exclude_vids is None:
        exclude_vids = {}
    if train_data is None:
        train_data = []
    video_files = read_all_files_from_directory(input_dir, exclude_vids)
    with torch.no_grad():
        for video in tqdm(video_files):
            batch = encode_video(input_dir, os.path.splitext(video)[0],
                                 ext=os.path.splitext(video)[1])
            train_data.extend(batch)

            # batch_query = generate_batched_query(batch)
            # # print(batch_prompt)
            # batch_result = encode_text_query(batch_query)
            # # print(batch_result)
            # save_query_features(batch, batch_result, t_feat_dir)

            thread = Thread(target=save_in_worker, args=(train_data, pretrain_jsonl_path))
            thread.start()
            # save_jsonl_file(train_data, pretrain_jsonl_path)
    return train_data


def generate_batched_query(batch):
    # print(batch)
    return batch['query']


# def encode_text_query(batch):
#     batch_output = []
#     with torch.no_grad():
#         for text in batch:
#             text_input = txt_processors["eval"](text)
#             sample = {"text_input": [text_input]}
#             features_text = model_extract.extract_features(sample, mode="text")
#             batch_output.append(features_text)
#         return batch_output


def save_query_features(batch, batch_result, q_feat_dir, training=False):
    for i, result in enumerate(batch_result):
        qid = batch["qid"][i]
        if training:
            aug_id = batch["aug_id"][i]
        else:
            aug_id = 0
        aug = f"_{aug_id}" if aug_id > 0 else ""
        q_feat_path = join(q_feat_dir, f"qid{qid}{aug}.npz")
        pooler_output = result.text_embeds[:, 0, :].squeeze()
        np.savez_compressed(q_feat_path, last_hidden_state=result.text_embeds.squeeze().cpu(),
                            pooler_output=pooler_output.cpu())


def save_jsonl_file(data, file_path):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    print("updated jsonl file: {} entries: {}".format(file_path, len(data)))


def save_in_worker(data, file_path):
    save_jsonl_file(data, file_path)


# def remove_val_test_videos_from_pretrain(pretrain_jsonl_path, pretrain_jsonl_path_new):
#     with open(pretrain_jsonl_path, "r") as f:
#         data = [json.loads(l.strip("\n")) for l in f.readlines()]
#
#     val_test_vids = []
#     with open(val_data_path_qvhl, "r") as f:
#         val_data = [json.loads(l.strip("\n")) for l in f.readlines()]
#     for val in val_data:
#         val_test_vids.append(val["vid"])
#
#     with open(test_data_path_qvhl, "r") as f:
#         test_data = [json.loads(l.strip("\n")) for l in f.readlines()]
#     for test in test_data:
#         val_test_vids.append(test["vid"])
#
#     with open(test_data_path_charades, "r") as f:
#         test_data = [json.loads(l.strip("\n")) for l in f.readlines()]
#     for test in test_data:
#         val_test_vids.append(test["vid"])
#
#     new_data = []
#     for d in tqdm(data):
#         if d["vid"] not in val_test_vids:
#             new_data.append(d)
#
#     with open(pretrain_jsonl_path_new, "w") as f:
#         for d in new_data:
#             f.write(json.dumps(d) + "\n")
#     return new_data


def extract_and_load_pretrain_data():
    videos_to_exclude = set()

    try:
        with open(pretrain_jsonl_path, "r") as f:
            existing_data = [json.loads(l.strip("\n")) for l in f.readlines()]
    except:
        existing_data = []

    for val in existing_data:
        videos_to_exclude.add(val["vid"])

    train_data = []
    seen_qids = set()
    for data in existing_data:
        if data["qid"] not in seen_qids:
            train_data.append(data)
            seen_qids.add(data["qid"])

    train_data = generate_pretrain_data(v_input_dir,
                                        pretrain_jsonl_path,
                                        pre_train_text_dir,
                                        videos_to_exclude, existing_data)
    save_jsonl_file(train_data, pretrain_jsonl_path)


if __name__ == "__main__":
    extract_and_load_pretrain_data()
