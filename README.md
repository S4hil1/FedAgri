# FedAgri: Federated Learning for Non-IID Maize Disease Detection
Official implementation and experimental artifacts for **FedAgri** — a privacy-preserving federated learning framework for maize disease classification under heterogeneous (Non-IID) agricultural data distributions.
> **Paper:** *FedAgri: A Federated Learning Framework for Non-IID Maize Disease Detection*  
> Sahil Pathak, Sujata Pal, Deepti R. Bathula — IIT Ropar
---
## Overview
Agricultural disease data is distributed across farms, cooperatives, and research institutions and is rarely Independent and Identically Distributed (IID). **FedAgri** addresses this with two main components:
| Component | Description |
|-----------|-------------|
| **FedCADR** | Federated Class-aware Adaptive Drift-controlled Residual aggregation — separates feature-level averaging from class-wise classifier residual updates with magnitude normalization and adaptive scaling (β) |
| **FedCAMNet** | Lightweight context-aware model built on MobileNetV2 with channel attention and Kolmogorov–Arnold Network (KAN) layers for resource-constrained edge clients |
| **MaizeDNIID** | Non-IID maize disease benchmark with explicit random partitioning and quantified heterogeneity metrics |
### Key results
- **96.5%** global accuracy with 10 clients (FedCADR + FedCAMNet), ~4–5% above FedAvg
- **93.5%** accuracy at **64 clients** with negligible aggregation overhead
- Worst-case client accuracy gap reduced from **32% → 8%** (32-client Non-IID setting)
- ~**0.80 Wh** estimated energy per training round for FedCAMNet
---
## Repository structure
```

README.md
├── code/                                      # Experiment notebooks (PyTorch FL + TF baselines)
│   ├── sample_data.ipynb                      # MaizeDNIID partition creation & distribution plots
│   ├── FD_Maize_HETEROGENEOUS_multi_processing.ipynb   # Main FL training pipeline
│   ├── FD_Maize_HETEROGENEOUS.ipynb           # Earlier heterogeneous FL variant
│   ├── FD_Maize.ipynb                         # Initial federated maize experiments
│   ├── test_fed_agg_sync.ipynb                # Aggregation method comparison
│   ├── FL_finetune_global_on_all_clients_traditional.ipynb
│   ├── FL_Distillation.ipynb                  # Federated distillation experiments
│   ├── FL_Distillation_Subnets.ipynb
│   ├── Maize_Hetero.ipynb
│   ├── plots.ipynb                            # Paper figures and result plots
│   ├── Maize_experiments_on_raw.ipynb         # Centralized TensorFlow baselines
│   ├── Maize_experiments_on_raw_proposed_model.ipynb
│   ├── Maize_experiments_on_AUGMENTED_TENSORFLOW.ipynb
│   ├── Maize_disease_classification_pytorch.ipynb
│   └── ...
└── Dataset_Maize_DD/                          # Dataset
    └── 1_CREATED_NEW_SAMPLE_split/
        ├── train/
        ├── validation/
        └── test/
```
> **Note:** The dataset is not bundled in this repository due to size and licensing. See [Dataset setup](#dataset-setup) for expected layout and sources.
---
## MaizeDNIID benchmark
### Classes (7)
| Class | Folder name (typical) |
|-------|----------------------|
| Blight | `Blight` |
| Gray Leaf Spot | `Gray_Leaf_Spot` / `GLS` |
| Common Rust | `Common_Rust` / `Rust` |
| Healthy | `Healthy` |
| Maize Lethal Necrosis | `MLN` |
| Fall Armyworm | `faw` / `FAW` |
| Maize Streak Virus | `maize_streak` / `MSV` |
### Global dataset splits
| Split | Images | Role |
|-------|-------:|------|
| Train | 12,880 | Source for federated client partitions |
| Validation | 3,220 | Global server evaluation during FL (460/class) |
| Test | 3,220 | Held-out final evaluation (460/class) |
| **Total** | **19,320** | 80% / 10% / 10% train / val / test |
Data sources include [PlantVillage](https://plantvillage.psu.edu/), Corn/Maize, Maize in Field, Makerere University, Nelson Mandela, and KaraAgro field collections (see paper for citations).
### Non-IID partitioning protocol
Partitions are **not** Dirichlet-based. Client data is built with a **random class-and-sample assignment** procedure (`create_federated_dataset` in `code/sample_data.ipynb`):
1. Read class folders from the centralized **train** split.
2. For each client `k ∈ {0, …, K−1}`:
   - Randomly assign **1 to 7 disease classes**.
   - For each assigned class, randomly sample **5 to all** images from that class.
3. Write client folders to `Federated_data_maize/client_{K}/client_{k}/{class}/`.
This induces **label skew**, **quantity skew**, and **partial class availability**. Images may appear at multiple clients (overlap is allowed).
Within each client, data is split **80% / 20%** for local training and local validation.
### Client configurations
| Clients (K) | Avg JS divergence | Gini | Avg classes/client | FHI |
|------------:|------------------:|-----:|-------------------:|----:|
| 10 | 0.4116 | 0.2776 | 4.40 | 0.3593 |
| 32 | 0.5504 | 0.4196 | 2.78 | 0.5268 |
| 64 | 0.4430 | 0.2560 | 4.02 | 0.3819 |
Pre-built partitions used in experiments: `Federated_data_maize/client_10`, `client_32`, `client_64`.
---
## Federated training setup
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate (η) | 0.001 |
| Loss | Cross-entropy |
| Local batch size | 4, 8, 16 (main results use **16**) |
| Communication rounds (T) | **50** |
| Local epochs (E) | **2, 5, 10** |
| Clients (K) | **10, 32, 64** |
| Client participation | **Full** — all K clients every round |
| Local train/val split | 80% / 20% |
| FedCADR scaling (β) | **0.83** |
| Input resolution | 128×128 (FL notebooks), ImageNet normalization |
| Random seeds | Single-run experiments (no multi-seed averaging reported) |
### Evaluated aggregators
FedAvg, FedAvgM, FedAdam, FedYogi, FedNova, FedProx, FedMedian, Krum, Trimmed Mean, FedLPA, FedRAM, FedAdaptive, and **FedCADR** (proposed).
### Evaluated models
EfficientNet, MobileNet, MobileViT, SqueezeNet, and **FedCAMNet** (proposed).
---
## Getting started
### Requirements
- Python 3.8+
- PyTorch + Torchvision
- NumPy, Pandas, scikit-learn, Matplotlib, tqdm
- Jupyter Notebook or VS Code
- TensorFlow/Keras (optional — centralized baseline notebooks only)
- CUDA-capable GPU recommended (experiments used NVIDIA Tesla P100)
Install dependencies manually (no `requirements.txt` is provided yet):
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm jupyter
```
### Dataset setup
Place the maize disease dataset under:
```
Dataset_Maize_DD/
└── 1_CREATED_NEW_SAMPLE_split/
    ├── train/          # 12,880 images in 7 class subfolders
    ├── validation/     # 3,220 images
    └── test/           # 3,220 images
```
### Step 1 — Create Non-IID client partitions
Open `code/sample_data.ipynb` and run:
```python
from sample_data import create_federated_dataset  # defined in notebook
