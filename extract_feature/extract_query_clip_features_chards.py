import numpy as np
from os.path import join
import os
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from run_on_video.data_utils import ClipFeatureExtractor

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
        new_dict = dict.fromkeys(
            ['qid', 'query', 'duration', 'vid', 'relevant_clip_ids', 'saliency_scores', 'relevant_windows'])
        new_dict.update(self.datalist[i])
        return new_dict

def generate_batched_query(batch):
    return batch['query']

def save_query_features(batch, batch_result, q_feat_dir, training=True):
    # Determine the actual batch size from one of the keys (e.g., 'qid')
    batch_size = len(batch["qid"])
    
    # Get aug_ids if they exist, otherwise create a list of zeros matching the BATCH SIZE
    aug_ids = batch.get("aug_id", [0] * batch_size)

    for i, result in enumerate(batch_result):
        qid = batch["qid"][i]
        
        if training:
            aug_id = aug_ids[i]
        else:
            aug_id = 0
            
        aug = f"_{aug_id}" if aug_id > 0 else ""
        q_feat_path = join(q_feat_dir, f"qid{qid}{aug}.npz")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(q_feat_path), exist_ok=True)
        
        # Handle result types (dictionary vs object)
        if isinstance(result, dict):
            lhs = result["last_hidden_state"]
            pooler = result["pooler_output"]
        else:
            lhs = result.last_hidden_state
            pooler = result.pooler_output

        # Save as compressed numpy array
        np.savez_compressed(
            q_feat_path, 
            last_hidden_state=lhs.detach().cpu().numpy(),
            pooler_output=pooler.detach().cpu().numpy()
        )
def collate_fn(batch):
    collated_dict = {}
    for key in batch[0]:
        collated_dict[key] = [data[key] for data in batch]
    return collated_dict

def extract_train_query_features():
    input_file = "data/charades_sta/charades_sta_train_tvr_format.jsonl"
    q_feat_dir = "../Datasets/charades/clip_text_features"

    if not os.path.exists(q_feat_dir):
        os.makedirs(q_feat_dir, exist_ok=True)

    print(f"Extracting Train features to: {q_feat_dir}")
    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        batch_result = feature_extractor.encode_text_query(batch_query)
        save_query_features(batch, batch_result, q_feat_dir)

def extract_test_query_features():
    input_file = "data/charades_sta/charades_sta_test_tvr_format.jsonl"
    q_feat_dir = "../Datasets/charades/clip_text_features"

    if not os.path.exists(q_feat_dir):
        os.makedirs(q_feat_dir, exist_ok=True)

    print(f"Extracting Test features to: {q_feat_dir}")
    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        batch_result = feature_extractor.encode_text_query(batch_query)
        save_query_features(batch, batch_result, q_feat_dir, False)

def extract_all_query_features():
    extract_train_query_features()
    extract_test_query_features()

if __name__ == "__main__":
    extract_all_query_features()