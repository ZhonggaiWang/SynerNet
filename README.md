# SynerNet

Official implementation of:

**SynerNet: Broad-to-Precise CAM Synergy for Weakly Supervised Semantic Segmentation**

Zhonggai Wang, Guangyu Gao, Zhuoshu Li, and A. K. Qin

[![Paper](https://img.shields.io/badge/Paper-Neural%20Networks-blue)](#citation)
[![Task](https://img.shields.io/badge/Task-WSSS-green)](#overview)
[![Code](https://img.shields.io/badge/Code-PyTorch-red)](#getting-started)

## Overview

SynerNet is a one-stage weakly supervised semantic segmentation (WSSS) framework. It targets the coupled error amplification problem in one-stage WSSS, where CAM generation and segmentation prediction are optimized through tightly coupled features and can reinforce each other's mistakes.

The key idea is a dual-branch co-training design with explicit complementary objectives:

- **B-CAM branch:** learns broader CAMs by encouraging uncertain regions toward foreground, improving object coverage and recall.
- **P-CAM branch:** learns more precise CAMs by encouraging unreliable regions toward background, sharpening localization and precision.
- **Confidence-guided fusion:** uses multi-scale ViT features to identify reliable and uncertain pixels, then fuses the two branches so pseudo-labels are both complete and precise.

The main training entry for VOC is:

```text
scripts/test_du_instance.py
```

The COCO counterpart is:

```text
scripts/test_du_instance_coco.py
```

## Results

### Pseudo-label Quality on PASCAL VOC 2012

| Method | Backbone | Train | Val |
| --- | --- | ---: | ---: |
| ToCo | ViT-B | 72.2 | 70.5 |
| DuPL | ViT-B | 75.1 | 73.5 |
| SeCo | ViT-B | 76.5 | 74.7 |
| **SynerNet** | **ViT-B** | **77.5** | **76.3** |

### Segmentation Performance

| Method | Sup. | Net. | VOC val | VOC test | COCO val |
| --- | --- | --- | ---: | ---: | ---: |
| ToCo | I | ViT-B | 69.8 | 70.5 | 41.3 |
| DuPL | I | ViT-B | 72.2 | 71.6 | 43.5 |
| SeCo | I | ViT-B | 74.0 | 73.8 | 46.7 |
| FFR | I | ViT-B | 74.8 | 74.6 | 46.8 |
| **SynerNet** | **I** | **ViT-B** | **75.8** | **75.2** | **47.1** |

`I` denotes image-level supervision.

## Getting Started

### 1. Clone

```bash
git clone https://github.com/ZhonggaiWang/SynerNet.git
cd SynerNet
```

### 2. Environment

The repository was developed with PyTorch. Install the listed dependencies:

```bash
conda create -n synernet python=3.9 -y
conda activate synernet
pip install -r requirements.txt
```

For CRF post-processing, install `pydensecrf` if it is not already available in your environment.

### 3. Data

#### PASCAL VOC 2012

Prepare VOC 2012 with SBD augmented masks. The default script paths assume you run training from the `scripts/` directory and keep the dataset one level above it:

```text
SynerNet/
|-- datasets/
|   `-- voc/
|       |-- train.txt
|       |-- train_aug.txt
|       |-- val.txt
|       |-- test.txt
|       `-- cls_labels_onehot.npy
|-- scripts/
`-- VOC2012/ or ../VOC2012 depending on your launch directory
```

Expected VOC layout:

```text
VOC2012/
|-- JPEGImages/
|-- SegmentationClass/
|-- SegmentationClassAug/
`-- ImageSets/
```

If your dataset is elsewhere, pass `--data_folder` and `--list_folder` explicitly.

#### MS COCO 2014

For COCO, prepare images and VOC-style segmentation labels, then set:

```bash
--img_folder /path/to/MSCOCO
--label_folder /path/to/MSCOCO
--list_folder /path/to/SynerNet/datasets/coco
```

## Training

The primary VOC training entry is `scripts/test_du_instance.py`.

Run from the `scripts/` directory:

```bash
cd scripts

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
  --nproc_per_node=3 \
  --master_port=29501 \
  test_du_instance.py \
  --data_folder ../VOC2012 \
  --list_folder ../datasets/voc \
  --train_set train_aug \
  --val_set val \
  --spg 4 \
  --max_iters 8000 \
  --lr 6e-5 \
  --save_ckpt
```

This creates timestamped outputs under:

```text
scripts/work_dir_voc_wseg/<timestamp>/
|-- checkpoints/
|-- predictions/
`-- train.log
```

For COCO:

```bash
cd scripts

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
  --nproc_per_node=6 \
  --master_port=29502 \
  test_du_instance_coco.py \
  --img_folder ../MSCOCO \
  --label_folder ../MSCOCO \
  --list_folder ../datasets/coco \
  --train_set train \
  --val_set val_part \
  --spg 3 \
  --max_iters 40000 \
  --lr 1.5e-5 \
  --save_ckpt
```

## Inference

### CAM evaluation

```bash
cd tools

python infer_cam_du_head.py \
  --model_path ../scripts/work_dir_voc_wseg/<timestamp>/checkpoints/default_model_iter_8000.pth \
  --data_folder ../VOC2012 \
  --list_folder ../datasets/voc \
  --infer_set val
```

### Segmentation with CRF

```bash
cd tools

python Dcrf-voc-infer-du_head.py \
  --model_path ../scripts/work_dir_voc_wseg/<timestamp>/checkpoints/default_model_iter_8000.pth \
  --data_folder ../VOC2012 \
  --list_folder ../datasets/voc \
  --infer_set val
```

For COCO segmentation inference, use:

```bash
python Dcrf-coco-infe-du-head.py \
  --model_path ../scripts/work_dir_coco_wseg/<timestamp>/checkpoints/default_model_iter_40000.pth \
  --img_folder ../MSCOCO \
  --label_folder ../MSCOCO \
  --list_folder ../datasets/coco \
  --infer_set val_part
```

## Repository Structure

```text
SynerNet/
|-- datasets/                 # VOC/COCO lists and dataset loaders
|-- model/                    # ViT backbone, dual-head model, losses
|-- scripts/
|   |-- test_du_instance.py   # Main VOC training entry
|   `-- test_du_instance_coco.py
|-- tools/                    # CAM and segmentation inference utilities
|-- utils/                    # CAM, CRF, optimizer, metric helpers
|-- requirements.txt
`-- README.md
```

## Notes

- The code uses `torch.distributed.launch`; newer PyTorch versions may also support `torchrun` with small argument-name adjustments.
- `scripts/test_du_instance.py` sets `CUDA_VISIBLE_DEVICES` internally. Edit that line or set the visible devices consistently for your machine.
- The paper reports multi-scale testing and CRF post-processing for final segmentation numbers.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{wang2026synernet,
  title   = {SynerNet: Broad-to-Precise CAM Synergy for Weakly Supervised Semantic Segmentation},
  author  = {Wang, Zhonggai and Gao, Guangyu and Li, Zhuoshu and Qin, A. K.},
  journal = {Neural Networks},
  year    = {2026},
  note    = {Preprint submitted to Neural Networks}
}
```

## Acknowledgement

This project builds on common WSSS components including ViT-based CAM generation, patch-token contrastive learning, multi-scale testing, and dense CRF post-processing. We thank the authors of the related open-source WSSS projects.
