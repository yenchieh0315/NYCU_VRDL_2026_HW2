# NYCU Computer Vision 2026 HW2

- Student ID: 314554036
- Name: 郭彥頡, Yenchieh Kuo

## Introduction
In this assignment, we tackle the Digit Detection problem using the SVHN dataset. We utilize the DETR architecture with a ResNet-50 backbone. To address the challenge of small object detection and hardware constraints (8GB VRAM), we modified the backbone to extract C4 features (higher resolution), applied Bfloat16 Automatic Mixed Precision (AMP) with Gradient Accumulation, and used Focal Loss to overcome extreme foreground-background class imbalance.

## Environment Setup
It is recommended to use Miniconda to set up the environment. You can easily recreate the environment using the provided "VRDL_HW2_env.yml" file:

```bash
# Create the environment from the VRDL_HW2_env.yml file
conda env create -f VRDL_HW2_env.yml

# Activate the new environment
conda activate VRDL_HW2_env
```

## Usage
### Training
How to train your model.
```bash
python VRDL_HW2_v6.py
```
### Inference
How to run inference.
```bash
#Make sure the dataset is in ./dataset/test, and "cond_detr_epoch_40.pth" is prepared.
python predict_v6.py
```

## Performance Snapshot
![Leaderboard]<img width="1183" height="55" alt="螢幕擷取畫面 2026-04-21 234902" src="https://github.com/user-attachments/assets/0a6f6d95-0bf7-4862-9120-a76d5748ecfe" />

```
