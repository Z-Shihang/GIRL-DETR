"""Strip iou_embed keys from v7b_3phase checkpoint to create clean video_lights-compatible weights."""
import torch
import os
import sys

src = "../GIRL_DETR/results/charadesSTA/SCST_finetune_v7b_3phase/charadesSTA-video_tef-exp-scst-ft-v7b-3phase-2026_04_18_15_12_05/model_best.ckpt"
dst_dir = "../GIRL_DETR/results/charadesSTA/merged"
dst = os.path.join(dst_dir, "v7b_3phase_best_merged.ckpt")

os.makedirs(dst_dir, exist_ok=True)

print(f"Loading {src}")
ckpt = torch.load(src, map_location="cpu")

for key in ["model", "model_raw"]:
    if key not in ckpt:
        continue
    orig_keys = list(ckpt[key].keys())
    drop = [k for k in orig_keys if k.startswith("iou_embed")]
    for k in drop:
        del ckpt[key][k]
    print(f"  {key}: {len(orig_keys)} -> {len(ckpt[key])} keys (dropped {drop})")

# Remove optimizer/lr_scheduler to save space (will be re-created)
for k in ["optimizer", "lr_scheduler"]:
    if k in ckpt:
        del ckpt[k]
        print(f"  Removed {k}")

torch.save(ckpt, dst)
sz = os.path.getsize(dst) / 1024 / 1024
print(f"Saved to {dst} ({sz:.1f} MB)")
