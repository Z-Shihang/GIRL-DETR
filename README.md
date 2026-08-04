<div align="center">

# GIRL-DETR: Gradient-Isolated Reinforcement Learning for Video Moment Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![PyTorch 12.4](https://img.shields.io/badge/PyTorch-CUDA_12.4-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/arXiv-2606.00775-b31b1b.svg)](https://arxiv.org/abs/2606.00775)

> Official PyTorch implementation of **"GIRL-DETR: Gradient-Isolated Reinforcement Learning for Video Moment Retrieval"**.

[**[Paper]**](https://arxiv.org/abs/2606.00775) | [**[Checkpoints]**]() | [**[Dataset]**](#-dataset-preparation)

</div>

---

## 📢 News
* **[June 2026]** Our paper is now available on [arXiv](https://arxiv.org/abs/2606.00775)! The full source code, pre-trained models, and dataset preparation scripts are progressively being released here. Stay tuned!

## 📖 Abstract

Video Moment Retrieval (VMR) task requires accurately localizing temporal boundaries aligned with natural language queries, but many models suffer from a misalignment between continuous surrogate losses and non-differentiable metrics, leading to optimization stagnation during the late stages of training and trapping boundary predictions in suboptimal solutions. Although Reinforcement Learning (RL) post-training successfully optimizes localization results for large models, applying it directly to lightweight networks easily disrupts the fragile feature representations established during the supervised phase. To overcome this optimization bottleneck, we propose Gradient-Isolated Reinforcement Learning for DETR (GIRL-DETR), introducing RL post-training into a lightweight temporal localization framework for the first time. The input video and text features first establish early alignment through Cross-Modal Interaction (CMI) before entering the transformer encoder. Subsequently, a Text-Guided Gating (TGG) mechanism dynamically injects semantic priors into the queries before the transformer decoder generates candidate proposals, providing high signal-to-noise ratio inputs for temporal prediction. After the supervised training reaches convergence, the backbone network is frozen to protect the feature manifold, while the detection head directly optimizes the non-differentiable evaluation metric tIoU to enhance localization accuracy through a Three-stage Progressive Reinforcement Learning (TPRL) strategy. This approach achieves an orthogonal decoupling of state representation and metric optimization. Experiments on Charades-STA, QVHighlights, and TACoS demonstrate that GIRL-DETR effectively resolves surrogate loss degradation and achieves substantial accuracy improvements with minimal parameter updates, providing a robust new pathway for RL applications in lightweight VMR models.

## 🖼️ Architecture

<div align="center">
  <img src="assets/architecture.png" alt="GIRL-DETR Architecture" width="85%">
</div>
<br/>
<p align="center">
  <em>Figure 1: Overall architecture of GIRL-DETR. The TGG mechanism injects semantic priors, followed by the TPRL strategy for direct tIoU optimization.</em>
</p>


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

We also utilized resources and features from [LAVIS](https://github.com/salesforce/LAVIS), [BLIP](https://github.com/salesforce/BLIP), [InternVideo2](https://github.com/OpenGVLab/InternVideo), [CLIP](https://github.com/openai/CLIP) and [SlowFast](https://github.com/facebookresearch/SlowFast). We sincerely thank the authors for their awesome open-source contributions!

## ✒️ Citation

If you find our code or paper useful, please consider citing:

```bibtex
@misc{zhang2026girldetrgradientisolatedreinforcementlearning,
      title={GIRL-DETR: Gradient-Isolated Reinforcement Learning for Video Moment Retrieval}, 
      author={Shihang Zhang and Mingjin Kuai and Ye Wei and Zhen Zhang and Wei Ji},
      year={2026},
      eprint={2606.00775},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2606.00775](https://arxiv.org/abs/2606.00775)}, 
}
```

## ✉️ Contact
For any questions regarding the code or the paper, please feel free to reach out via email: **shihang_zhang@stu.scu.edu.cn** or open an issue in this repository.

## 📄 License
This project is licensed under the [MIT License](LICENSE).

```
