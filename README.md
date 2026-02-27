# SCALE: Structured Calibration-Aware Loss Embedding for Reference-Based Membership Inference

This repository contains the source code for the paper **"SCALE: Structured Calibration-Aware Loss Embedding for Reference-Based Membership Inference"**.

SCALE is a feature-enhanced calibration framework for reference-based membership inference attacks (MIAs). It constructs a multi-dimensional embedding from the target loss and the reference loss distribution—encompassing statistical dispersion, normalized deviation, ratio-based, and interaction features—and integrates them via a lightweight MLP scoring model. SCALE consistently improves low-FPR true positive rates over the [RAPID](https://github.com/T0hsakar1n/RAPID) baseline while maintaining comparable target utility.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Step 1: Train the Target and Shadow Models](#step-1-train-the-target-and-shadow-models)
  - [Step 2: Train the Reference Models](#step-2-train-the-reference-models)
  - [Step 3: Run Membership Inference Attack](#step-3-run-membership-inference-attack)
  - [Step 4: Visualize Results](#step-4-visualize-results)
- [Feature Groups (G0–G4)](#feature-groups-g0g4)
- [Advanced Experiments](#advanced-experiments)
  - [Cross-Architecture Attack](#cross-architecture-attack)
  - [Ablation Study: Impact of Reference Models](#ablation-study-impact-of-reference-models)
  - [Ablation Study: Impact of Query Budget](#ablation-study-impact-of-query-budget)
  - [Ablation Study: ROC Comparison Across Feature Groups](#ablation-study-roc-comparison-across-feature-groups)
- [Supported Datasets and Architectures](#supported-datasets-and-architectures)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Overview

Reference-based MIAs calibrate the target model's per-sample loss using independently trained reference models to factor out intrinsic sample difficulty. However, calibration is often imperfect: relying on a small set of scalar scores fails to capture **structured calibration errors** such as biased difficulty estimates, high variance across reference models, and sample-dependent relative deviations—errors that are especially harmful in the low-FPR regime.

SCALE addresses this by constructing a **multi-dimensional loss embedding** from four complementary feature families:
1. **Statistical dispersion features** — capture the reliability of difficulty estimation
2. **Normalized deviation features** — characterize relative positioning via Z-score normalization
3. **Ratio-based features** — emphasize proportional discrepancies
4. **Interaction features** — model higher-order dependencies among signals

## Getting Started

### Requirements

- Python 3.9
- CUDA 12.1
- PyTorch 2.0.1
- NVIDIA GPU with ≥8 GB memory

### Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/SCALE.git
cd SCALE
```

2. Create a conda environment and install dependencies:
```bash
conda create -n scale python=3.9
conda activate scale
pip install numpy==1.23.0 torch==2.0.1 torchvision==0.15.2
pip install -r requirements.txt
```

3. Verify that CUDA is available:
```python
import torch
print(torch.__version__)          # 2.0.1
print(torch.cuda.is_available())  # True
```

## Quick Start

The following commands reproduce the main results on CIFAR-10 with VGG16 using SCALE (G4 features):

```bash
# 1. Train the target model and shadow model
python pretrain.py 0 config/cifar10/cifar10_vgg16.json

# 2. Train reference models (4 models by default)
python refer_model.py 0 config/cifar10/cifar10_vgg16.json --model_num 4

# 3. Run the SCALE attack (G4 = full feature set)
python mia_attack.py 0 config/cifar10/cifar10_vgg16.json --model_num 4 --query_num 8 --feature_group G4

# 4. Plot the ROC curve
python plot.py 0 config/cifar10/cifar10_vgg16.json --feature_group all
```

## Detailed Usage

### Step 1: Train the Target and Shadow Models

```bash
python pretrain.py <gpu_id> <config_path>
```

**Example:**
```bash
python pretrain.py 0 config/cifar10/cifar10_vgg16.json
```

This trains both the **target model** (victim) and the **shadow model** on the specified dataset and architecture. All training hyperparameters (learning rate, optimizer, scheduler, etc.) are defined in the JSON config file.

### Step 2: Train the Reference Models

```bash
python refer_model.py <gpu_id> <config_path> --model_num <num_reference_models>
```

**Example:**
```bash
# Train 4 reference models (low-budget, used in main experiments)
python refer_model.py 0 config/cifar10/cifar10_vgg16.json --model_num 4

# Train 64 reference models (for ablation studies)
python refer_model.py 0 config/cifar10/cifar10_vgg16.json --model_num 64
```

For distributed training (optional):
```bash
python refer_model.py 0 config/cifar10/cifar10_vgg16.json --distributed True --world_size 4 --model_num 4
```

### Step 3: Run Membership Inference Attack

```bash
python mia_attack.py <gpu_id> <config_path> --model_num <M> --query_num <Q> --feature_group <G>
```

| Argument | Description | Default |
|---|---|---|
| `gpu_id` | GPU device ID | `0` |
| `config_path` | Path to the config JSON file | — |
| `--model_num` | Number of reference models to use | `4` |
| `--query_num` | Number of queries per reference model | `8` |
| `--feature_group` | Feature group: `G0`, `G1`, `G2`, `G3`, or `G4` | `G0` |
| `--attack_epochs` | Training epochs for the attack model | `150` |

**Examples:**
```bash
# Run the RAPID baseline (G0)
python mia_attack.py 0 config/cifar10/cifar10_vgg16.json --model_num 4 --query_num 8 --feature_group G0

# Run SCALE (G4 = full feature set)
python mia_attack.py 0 config/cifar10/cifar10_vgg16.json --model_num 4 --query_num 8 --feature_group G4

# Run all feature groups for ablation
for G in G0 G1 G2 G3 G4; do
    python mia_attack.py 0 config/cifar10/cifar10_vgg16.json --model_num 4 --query_num 8 --feature_group $G
done
```

### Step 4: Visualize Results

**ROC Curves:**
```bash
# Plot ROC for a single feature group
python plot.py 0 config/cifar10/cifar10_vgg16.json --feature_group G4

# Plot ROC for all feature groups (G0–G4)
python plot.py 0 config/cifar10/cifar10_vgg16.json --feature_group all

# Compare specific attack variants
python plot.py 0 config/cifar10/cifar10_vgg16.json --attacks rapid_attack_G0,rapid_attack_G4
```

**TPR vs. Architecture Bar Chart:**
```bash
# Default: CIFAR-10, FPR = 0.1%
python plot_tpr_vs_architecture.py

# Custom dataset and FPR threshold
python plot_tpr_vs_architecture.py --dataset cifar10 --fpr 0.001 --groups "0,4"
```

## Feature Groups (G0–G4)

The feature groups form a progressive inclusion hierarchy:

| Group | Features | Description |
|---|---|---|
| **G0** | *S(x)*, *S'(x)* | RAPID baseline: target loss + calibrated loss |
| **G1** | G0 + *μ_ref*, *σ²_ref* | + reference distribution statistics |
| **G2** | G1 + *z(x)* | + Z-score normalized deviation |
| **G3** | G2 + *ρ(x)*, *log(1+ρ)* | + ratio-based features |
| **G4** | G3 + *f_int*, *h(x)* | + interaction features (**= SCALE**) |

**G4 is the complete SCALE method.** G0 is equivalent to the original RAPID baseline.

## Advanced Experiments

### Cross-Architecture Attack

Evaluate robustness when the shadow/reference models have a different architecture from the target:

```bash
# Same-architecture attack (baseline)
python mia_attack_cross_arch.py 0 config/cifar10/cifar10_vgg16.json \
    --victim_model_name vgg16 \
    --shadow_model_name vgg16 \
    --reference_model_name vgg16 \
    --feature_group G4

# Cross-architecture attack (shadow: VGG16 → target: ResNet50)
python mia_attack_cross_arch.py 0 config/cifar10/cifar10_resnet50.json \
    --victim_model_name resnet50 \
    --shadow_model_name vgg16 \
    --reference_model_name vgg16 \
    --feature_group G4
```

**Visualize cross-architecture results:**
```bash
python plot_cross_arch.py --dataset cifar10 --group G4
```

### Ablation Study: Impact of Reference Models

Analyze how the number of reference models affects TPR at a fixed FPR:

```bash
python plot_reference_models_impact.py \
    --dataset cifar10 --model vgg16 \
    --model_numbers "1,2,4,8,16,32,64" \
    --query_num 8
```

### Ablation Study: Impact of Query Budget

Analyze how the number of queries affects performance with a fixed reference-model budget:

```bash
python plot_query_nums_impact.py \
    --dataset cifar10 --model vgg16 \
    --ref_model 32 \
    --query_nums "1,2,4,8,16,32,64"
```

### Ablation Study: ROC Comparison Across Feature Groups

Compare ROC curves of different feature groups under a specific budget:

```bash
python plot_ablation_roc.py 0 config/cifar10/cifar10_vgg16.json \
    --groups "G0,G1,G2,G3,G4" \
    --ref_models "4" \
    --query_nums "8"
```

## Supported Datasets and Architectures

**Datasets:** CIFAR-10, CIFAR-100, SVHN, GTSRB, Flowers102, CINIC-10, and more.

All datasets are **automatically downloaded** when first used. See `datasets.py` for details.

**Target Architectures:** VGG16, ResNet50, DenseNet121, MobileNetV2, GoogLeNet, EfficientNet-B0, ShuffleNet-V2, WideResNet34.

Configuration files for each dataset–architecture combination are located in the `config/` directory.

## Project Structure

```
SCALE/
├── config/                     # JSON config files for each dataset-architecture pair
│   ├── cifar10/
│   │   ├── cifar10_vgg16.json
│   │   ├── cifar10_resnet50.json
│   │   └── ...
│   ├── flowers102/
│   ├── gtsrb/
│   ├── svhn/
│   └── ...
├── models/                     # Model architecture definitions
├── utils/                      # Utility functions (ROC plotting, etc.)
├── pretrain.py                 # Train target and shadow models
├── refer_model.py              # Train reference models
├── mia_attack.py               # Run membership inference attack (G0–G4)
├── mia_attack_cross_arch.py    # Cross-architecture attack evaluation
├── plot.py                     # Plot ROC curves
├── plot_tpr_vs_architecture.py # TPR vs. architecture bar chart
├── plot_reference_models_impact.py  # Ablation: reference model count
├── plot_query_nums_impact.py   # Ablation: query budget
├── plot_ablation_roc.py        # Ablation: ROC comparison across feature groups
├── plot_cross_arch.py          # Cross-architecture result visualization
├── datasets.py                 # Dataset loading and preprocessing
├── requirements.txt            # Python dependencies
└── README.md
```

## Acknowledgements

Our code is built upon the official repository of [RAPID](https://github.com/T0hsakar1n/RAPID) (He et al., ACM CCS 2024). We sincerely appreciate their valuable contributions to the community.

## Citation

If you find our work helpful in your research, please cite it using the following:

```bibtex
@article{scale2025,
  title   = {SCALE: Structured Calibration-Aware Loss Embedding for Reference-Based Membership Inference},
  author  = {},
  year    = {2025},
}
```
