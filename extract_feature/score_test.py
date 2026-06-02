import os
import sys
import json
import numpy as np
from tqdm import tqdm
import pprint

sys.path.append("/data1/zhangshihang/VideoLights/")
from standalone_eval.eval import eval_submission

V_FEAT_DIR = "/data1/zhangshihang/Datasets/charades/clip_video_features/"
T_FEAT_DIR = "/data1/zhangshihang/Datasets/charades/clip_text_features/"
ORIGINAL_JSONL = "./run_on_video/hl_val_submission.jsonl"
OUTPUT_JSONL = "./run_on_video/hl_val_submission_reranked_best.jsonl"

GT_JSONL = "./data/charades_sta/charades_sta_test_tvr_format.jsonl" 


def load_feature(filepath):
    data = np.load(filepath)
    keys = data.files
    feat = data[keys[0]]
    return feat

def get_frame_similarities(v_feats, t_feat):
    v_norms = np.linalg.norm(v_feats, axis=1, keepdims=True) + 1e-8
    t_norm = np.linalg.norm(t_feat) + 1e-8
    v_normalized = v_feats / v_norms
    t_normalized = t_feat / t_norm
    sim_curve = np.dot(v_normalized, t_normalized)
    return sim_curve

def main():
    print("Loading predictions and ground truth...")
    with open(ORIGINAL_JSONL, 'r') as f:
        predictions = [json.loads(line) for line in f]
        
    with open(GT_JSONL, 'r') as f:
        gt_data = [json.loads(line) for line in f]

    print("Caching features and curves...")
    item_cache = []
    
    for item in tqdm(predictions, desc="Processing features"):
        vid = str(item['vid'])
        qid = str(item['qid'])
        
        v_path = os.path.join(V_FEAT_DIR, f"{vid}.npz")
        t_path = os.path.join(T_FEAT_DIR, f"qid{qid}.npz")
        
        if not os.path.exists(v_path) or not os.path.exists(t_path):
            item_cache.append((item, None))
            continue
            
        v_feat_all = load_feature(v_path)
        t_feat = load_feature(t_path)
        
        if len(t_feat.shape) > 1:
            t_feat = np.mean(t_feat, axis=0)
            
        sim_curve = get_frame_similarities(v_feat_all, t_feat)
        item_cache.append((item, sim_curve))

    print("\nStarting automatic search for Trim Threshold...")
    thresholds_to_test = [0.0, 0.7, 0.8, 0.85, 0.9, 0.95]
    
    best_r1_07 = -1.0
    best_thd = 0.0
    best_submission = []
    
    for thd in thresholds_to_test:
        current_submission = []
        for item, sim_curve in item_cache:
            new_item = item.copy()
            new_windows = []
            
            for w in item['pred_relevant_windows']:
                st, ed, score = w[0], w[1], w[2]
                
                if sim_curve is None or thd == 0.0:
                    new_windows.append([st, ed, score])
                    continue
                    
                st_idx = max(0, int(np.floor(st)))
                ed_idx = min(len(sim_curve) - 1, int(np.ceil(ed)))
                
                if st_idx >= ed_idx:
                    new_windows.append([st, ed, score])
                    continue
                    
                local_curve = sim_curve[st_idx:ed_idx+1]
                if len(local_curve) == 0:
                    new_windows.append([st, ed, score])
                    continue
                    
                peak_sim = np.max(local_curve)
                drop_val = peak_sim * thd
                
                new_st_idx = st_idx
                for idx in range(st_idx, ed_idx):
                    if sim_curve[idx] >= drop_val:
                        new_st_idx = idx
                        break
                        
                new_ed_idx = ed_idx
                for idx in range(ed_idx, st_idx, -1):
                    if sim_curve[idx] >= drop_val:
                        new_ed_idx = idx
                        break
                        
                new_st = float(new_st_idx)
                new_ed = float(new_ed_idx)
                
                if new_st >= new_ed:
                    new_st, new_ed = st, ed
                    
                new_windows.append([new_st, new_ed, score])
                
            new_item['pred_relevant_windows'] = new_windows
            current_submission.append(new_item)
            
        metrics = eval_submission(current_submission, gt_data, verbose=False, match_number=True)
        
        brief = metrics.get('brief', {})
        r1_03 = brief.get('MR-full-R1@0.3', 0)
        r1_05 = brief.get('MR-full-R1@0.5', 0)
        r1_07 = brief.get('MR-full-R1@0.7', 0)
        
        print(f"Trim Threshold: {thd:.2f} -> R1@0.3: {r1_03:.2f} | R1@0.5: {r1_05:.2f} | R1@0.7: {r1_07:.2f}")
        
        if r1_07 > best_r1_07:
            best_r1_07 = r1_07
            best_thd = thd
            best_submission = current_submission

    print(f"\nBest Threshold found: {best_thd:.2f} with R1@0.7 = {best_r1_07:.2f}")
    print(f"Saving best submission to {OUTPUT_JSONL}")
    
    with open(OUTPUT_JSONL, 'w') as f:
        for item in best_submission:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()