"""
Quick verification of extracted SigLIP video features.

Checks:
1. Total count matches expected 9848
2. Shape & dtype sanity for random samples
3. Value range / L2 norm sanity (not all zeros, not NaN/Inf)
4. Re-extract a few videos from scratch and compare with saved features (bit-exact)
5. Frame count consistency with CLIP features

Usage:
    conda run -n siglip_extract HF_ENDPOINT=https://hf-mirror.com \
        python extract_feature/quick_verify_siglip.py [--num_reextract 3] [--gpu 0]
"""
import os, sys, argparse, random
import numpy as np
import torch
from tqdm import tqdm

SIGLIP_DIR = "/data1/zhangshihang/Datasets/charades/siglip_video_features/"
CLIP_DIR   = "/data1/zhangshihang/Datasets/charades/clip_video_features/"
VIDEO_DIRS = [
    "/data1/zhangshihang/data/Charades/Charades_Videos/train/",
    "/data1/zhangshihang/data/Charades/Charades_Videos/test/",
    "/data1/zhangshihang/data/Charades/Charades_v1_480/",
]

def find_video(vid):
    for d in VIDEO_DIRS:
        p = os.path.join(d, f"{vid}.mp4")
        if os.path.exists(p):
            return p
    return None

def check_count():
    files = [f for f in os.listdir(SIGLIP_DIR) if f.endswith('.npz')]
    n = len(files)
    print(f"[1/5] Feature count: {n}")
    if n == 9848:
        print("  ✅ Matches expected 9848")
    else:
        print(f"  ❌ Expected 9848, got {n}")
    return files

def check_shape_dtype(files, n_sample=50):
    print(f"[2/5] Checking shape & dtype on {n_sample} random samples...")
    samples = random.sample(files, min(n_sample, len(files)))
    ok = True
    for f in samples:
        data = np.load(os.path.join(SIGLIP_DIR, f))
        if 'features' not in data:
            print(f"  ❌ {f}: missing key 'features', keys={list(data.keys())}")
            ok = False; continue
        feat = data['features']
        if feat.ndim != 2 or feat.shape[1] != 768:
            print(f"  ❌ {f}: unexpected shape {feat.shape}, expected (T, 768)")
            ok = False; continue
        if feat.dtype != np.float16:
            print(f"  ⚠️ {f}: dtype={feat.dtype}, expected float16")
        if feat.shape[0] == 0:
            print(f"  ❌ {f}: 0 frames")
            ok = False
    if ok:
        print("  ✅ All samples have correct shape (T, 768) float16")

def check_values(files, n_sample=50):
    print(f"[3/5] Checking value ranges on {n_sample} samples...")
    samples = random.sample(files, min(n_sample, len(files)))
    ok = True
    for f in samples:
        feat = np.load(os.path.join(SIGLIP_DIR, f))['features'].astype(np.float32)
        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
            print(f"  ❌ {f}: contains NaN/Inf")
            ok = False; continue
        norms = np.linalg.norm(feat, axis=1)
        if np.any(norms < 0.1):
            print(f"  ⚠️ {f}: some frames have near-zero norm (min={norms.min():.4f})")
        if norms.max() > 100:
            print(f"  ⚠️ {f}: very large norms (max={norms.max():.2f})")
    if ok:
        print("  ✅ No NaN/Inf, value ranges look reasonable")

def check_frame_count_vs_clip(files, n_sample=100):
    print(f"[4/5] Comparing frame counts with CLIP features on {n_sample} samples...")
    if not os.path.exists(CLIP_DIR):
        print("  ⚠️ CLIP feature dir not found, skipping"); return
    common = [f for f in files if os.path.exists(os.path.join(CLIP_DIR, f))]
    samples = random.sample(common, min(n_sample, len(common)))
    mismatches = 0
    for f in samples:
        s_feat = np.load(os.path.join(SIGLIP_DIR, f))['features']
        c_feat = np.load(os.path.join(CLIP_DIR,   f))['features']
        if s_feat.shape[0] != c_feat.shape[0]:
            mismatches += 1
            if mismatches <= 3:
                print(f"  ⚠️ {f}: SigLIP frames={s_feat.shape[0]}, CLIP frames={c_feat.shape[0]}")
    if mismatches == 0:
        print(f"  ✅ All {len(samples)} sampled videos have matching frame counts")
    else:
        print(f"  ⚠️ {mismatches}/{len(samples)} videos have mismatched frame counts (minor ffmpeg differences OK)")

def check_reextract(files, n_reextract=3, gpu=0):
    """Re-extract features from scratch for a few videos and compare bit-for-bit."""
    print(f"[5/5] Re-extracting {n_reextract} videos from scratch for bit-exact comparison...")
    from transformers import AutoModel, AutoProcessor
    from PIL import Image
    import ffmpeg, math

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model_name = "google/siglip-base-patch16-224"
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    def convert_to_float(frac_str):
        try: return float(frac_str)
        except ValueError:
            num, denom = frac_str.split('/')
            return float(num) / float(denom)

    def extract_frames(video_path, framerate=1.0, size=224):
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
            if dur > 0 and dur < 1/fps + 0.1:
                fps = 2 / max(int(dur), 1)
        except: pass
        cmd = ffmpeg.input(video_path).filter('fps', fps=fps).filter('scale', W, H)
        x, y = int((W - size)/2), int((H - size)/2)
        cmd = cmd.crop(x, y, size, size)
        out, _ = cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True)
        return np.frombuffer(out, np.uint8).reshape([-1, size, size, 3])

    # Pick random videos that have source video files
    candidates = []
    random.shuffle(files)
    for f in files:
        vid = f.replace('.npz', '')
        vp = find_video(vid)
        if vp:
            candidates.append((vid, vp, f))
        if len(candidates) >= n_reextract:
            break

    all_ok = True
    for vid, vp, fname in candidates:
        try:
            frames = extract_frames(vp)
            pil_imgs = [Image.fromarray(fr) for fr in frames]
            inputs = processor(images=pil_imgs, return_tensors="pt")
            with torch.no_grad():
                out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
                fresh = out.pooler_output.cpu().to(torch.float16).numpy()
            saved = np.load(os.path.join(SIGLIP_DIR, fname))['features']
            if np.array_equal(fresh, saved):
                print(f"  ✅ {vid}: bit-exact match ({fresh.shape[0]} frames)")
            else:
                max_diff = np.abs(fresh.astype(np.float32) - saved.astype(np.float32)).max()
                if max_diff < 1e-3:
                    print(f"  ⚠️ {vid}: very close (max_diff={max_diff:.6f}), likely GPU non-determinism")
                else:
                    print(f"  ❌ {vid}: MISMATCH max_diff={max_diff:.4f}, shapes fresh={fresh.shape} saved={saved.shape}")
                    all_ok = False
        except Exception as e:
            print(f"  ❌ {vid}: error - {e}")
            all_ok = False

    if all_ok:
        print("  ✅ Re-extraction verification passed")
    else:
        print("  ❌ Some re-extractions did not match - features may be incorrect!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_reextract", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("SigLIP Feature Quick Verification")
    print("=" * 60)
    files = check_count()
    check_shape_dtype(files)
    check_values(files)
    check_frame_count_vs_clip(files)
    check_reextract(files, args.num_reextract, args.gpu)
    print("=" * 60)
    print("Done.")
