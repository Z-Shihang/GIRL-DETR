<div align="center">

# GIRL-DETR: Gradient-Isolated Reinforcement Learning for DETR-based Video Moment Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![PyTorch 12.4](https://img.shields.io/badge/PyTorch-CUDA_12.4-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)](https://arxiv.org/)

> Official PyTorch implementation of **"GIRL-DETR: Gradient-Isolated Reinforcement Learning for DETR-based Video Moment Retrieval"** (Under Review).

[**[Paper (Coming Soon)]**]() | [**[Checkpoints (Coming Soon)]**]() | [**[Dataset]**](#-dataset-preparation)

</div>

---

## 📢 News
* **[May 2026]** The paper is currently being submitted for review. The full source code, pre-trained models, and dataset preparation scripts are progressively being released here. Stay tuned for the arXiv preprint!

## 📖 Abstract

*(Please replace this placeholder with your paper's abstract once the arXiv link is available.)*

> The abstract of the paper will be updated here soon. It will briefly introduce the core motivation, the proposed GIRL-DETR framework, and the overall contributions to temporal action localization and video moment retrieval.

## 🖼️ Architecture

*(Please replace this placeholder with your actual architecture diagram)*
<div align="center">
  <img src="docs/architecture_placeholder.png" alt="GIRL-DETR Architecture" width="80%">
</div>

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/Z-Shihang/GIRL-DETR.git](https://github.com/Z-Shihang/GIRL-DETR.git)
cd GIRL-DETR
```

### 2. Setup Environment
Requirements:
- CUDA 12.4 compatible GPU
- Python $\ge$ 3.10
- Conda package manager

```bash
# Create and activate conda environment
conda create --name girl python=3.10 -y
conda activate girl

# Install PyTorch with CUDA 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install all dependencies
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "from girl.config import BaseOptions; print('✓ Installation successful')"
```

## 📂 Dataset Preparation

Ensure the following data and feature directories are available before training or evaluation. We utilize features such as **InternVideo2, CLIP, and BLIP**.

```text
GIRL-DETR/
├── data/
│   ├── highlight_train_release.jsonl        # QVHighlights train
│   ├── highlight_val_release.jsonl          # QVHighlights val
│   ├── tacos/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── charades_sta/
│       ├── charades_sta_train_tvr_format.jsonl
│       └── charades_sta_test_tvr_format.jsonl
```
*(Place feature files in your designated `FEAT_ROOT` directory: `../Datasets/{dataset_name}/features/`)*

## 🚀 Training & Testing

All training scripts are located in `girl/scripts/{dataset}/` with standardized names: `train.sh` for supervised training and `train_rl.sh` for RL fine-tuning.

### 1. Supervised Pre-training
Train the standard DETR-based architecture first:
```bash
# For QVHighlights
bash girl/scripts/qvhl/train.sh

# For TACoS
bash girl/scripts/tacos/train.sh

# For Charades-STA
bash girl/scripts/charades_sta/train.sh
```

### 2. RL Fine-tuning
Once supervised training completes, use the best checkpoint (`model_best.ckpt`) to initialize the Gradient-Isolated Reinforcement Learning phase:
```bash
# The script automatically loads the supervised weights for progressive RL optimization
bash girl/scripts/qvhl/train_rl.sh
```

### 3. Inference / Evaluation
To evaluate a trained checkpoint:
```bash
python girl/inference.py \
    --resume results/qvhighlights/model_best.ckpt \
    --eval_split_name val
```

## 🛠️ Advanced Usage

Modify hyperparameters directly in the shell scripts, or override variables via the command line:

```bash
# Use a specific GPU device and override feature directory
export CUDA_VISIBLE_DEVICES=0
export FEAT_ROOT=/path/to/custom/features
bash girl/scripts/qvhl/train.sh
```

## 📁 Project Structure

<details>
<summary>Click to expand</summary>

```text
GIRL-DETR/
├── girl/
│   ├── config.py              # Configuration and argument parsing
│   ├── train.py               # Main training loop
│   ├── inference.py           # Inference and evaluation
│   ├── transformer.py         # Transformer architecture with CMI
│   ├── model.py               # Model definition
│   └── scripts/               # Training scripts per dataset
├── data/                      # Dataset annotations
├── extract_feature/           # Feature extraction utilities
├── utils/                     # Utility functions
├── detectron2/                # Detectron2 submodule
└── requirements.txt           # Python dependencies
```
</details>

## 🙏 Acknowledgement

This code is based on [moment-detr](https://github.com/jayleicn/moment_detr), [QD-DETR](https://github.com/wjun0830/QD-DETR), [CG-DETR](https://github.com/wjun0830/CGDETR), [detr](https://github.com/facebookresearch/detr), [VideoLights](https://github.com/dpaul06/VideoLights) and [TVRetrieval XML](https://github.com/jayleicn/TVRetrieval). 

We also utilized resources from [LAVIS](https://github.com/salesforce/LAVIS), [CLIP](https://github.com/openai/CLIP) and [SlowFast](https://github.com/facebookresearch/SlowFast). We sincerely thank the authors for their awesome open-source contributions!

## ✒️ Citation

If you find our code or paper useful, please consider citing:

```bibtex
@article{girl2026,
  title={GIRL-DETR: Grounded Video-Language Understanding for Temporal Action Localization},
  author={...},
  journal={...},
  year={2026}
}
```

## ✉️ Contact
For any questions regarding the code or the paper, please feel free to reach out via email: **shihang_zhang@stu.scu.edu.cn** or open an issue in this repository.

## 📄 License
This project is licensed under the [MIT License](LICENSE).
