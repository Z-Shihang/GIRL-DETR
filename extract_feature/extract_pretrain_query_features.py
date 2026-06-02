import io
import math
import os

import numpy as np
import torch
from os.path import join

import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

from run_on_video.data_utils import VideoProcessor, ClipFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                  model_type="pretrain", is_eval=True,
                                                                  device=device)  # Blip2 featrures

# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor",
#                                                                   model_type="base", is_eval=True,
#                                                                   device=device) # Blip featrures
# text_input = txt_processors["eval"](caption)
# sample = {"image": image, "text_input": [text_input]}
video_loader = VideoProcessor(framerate=0.5, size=224, centercrop=True)  # clip len 2 sec

input_file = "data/pretrain/pre_train_blip.jsonl"

feature_extractor = ClipFeatureExtractor(
    framerate=1 / 2, size=224, centercrop=True,
    model_name_or_path="ViT-B/32", device="cuda"
)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class QVHighlightsDataset(Dataset):
    def __init__(self, input_file):
        self.datalist = load_jsonl(input_file)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        # query = self.datalist[i]["query"]
        # prompt = f'[INST]Paraphrase the text in quatation mark. "{query}"[/INST]\n'
        # return prompt
        new_dict = dict.fromkeys(
            ['qid', 'query', 'duration', 'vid', 'relevant_clip_ids', 'saliency_scores', 'relevant_windows'])
        new_dict.update(self.datalist[i])
        return new_dict


def generate_batched_query(batch):
    # print(batch)
    return batch['query']


def save_query_features_clip(batch, batch_result, q_feat_dir, training=False):
    for i, result in enumerate(batch_result):
        qid = batch["qid"][i]
        if training:
            aug_id = batch["aug_id"][i]
        else:
            aug_id = 0
        aug = f"_{aug_id}" if aug_id > 0 else ""
        q_feat_path = join(q_feat_dir, f"qid{qid}{aug}.npz")
        np.savez_compressed(q_feat_path, last_hidden_state=result["last_hidden_state"].cpu(),
                            pooler_output=result["pooler_output"].cpu())


def save_query_features_blip(batch, batch_result, q_feat_dir, training=False):
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


def collate_fn(batch):
    """Collates a batch of dictionaries into a single dictionary.

    Args:
      batch: A list of dictionaries.

    Returns:
      A single dictionary.
    """

    collated_dict = {}
    for key in batch[0]:
        collated_dict[key] = [data[key] for data in batch]
    return collated_dict


def encode_text_query(batch):
    batch_output = []
    with torch.no_grad():
        for text in batch:
            text_input = txt_processors["eval"](text)
            sample = {"text_input": [text_input]}
            features_text = model.extract_features(sample, mode="text")
            batch_output.append(features_text)
        return batch_output


def generate_batched_vid(v_feat_dir, batch):
    return [vid for vid in batch['vid'] if not is_file_present(v_feat_dir, vid)]


# write a code to check if a file is present in the directory
# if not present, then only extract the features
def is_file_present(v_feat_dir, vid):
    file_path = join(v_feat_dir, f"{vid}.npz")
    return os.path.exists(file_path)


def collate_fn(batch):
    """Collates a batch of dictionaries into a single dictionary.

    Args:
      batch: A list of dictionaries.

    Returns:
      A single dictionary.
    """

    collated_dict = {}
    for key in batch[0]:
        collated_dict[key] = [data[key] for data in batch]
    return collated_dict


def extract_clip_pretrain_query_features():
    q_feat_dir = "../pretrain/pretrain_features/clip_query_features"
    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        # print(batch_prompt)
        batch_result = feature_extractor.encode_text_query(batch_query)
        # print(batch_result)
        save_query_features_clip(batch, batch_result, q_feat_dir, False)


def extract_blip_pretrain_query_features():
    q_feat_dir = "../pretrain/pretrain_features/blip_query_features"
    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        # print(batch_prompt)
        batch_result = encode_text_query(batch_query)
        # print(batch_result)
        save_query_features_blip(batch, batch_result, q_feat_dir, training=False)


def extract_all_query_features():
    extract_clip_pretrain_query_features()
    extract_blip_pretrain_query_features()


if __name__ == "__main__":
    extract_all_query_features()
    # x = feature_extractor.encode_text_query(["Chef makes pizza and cuts it up.", "Chef makes pizza and cuts"])
    # print(x)
