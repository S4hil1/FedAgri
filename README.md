# Federated Learning for Non-IID Maize Disease Detection

A privacy-preserving federated learning framework for maize disease classification under heterogeneous (Non-IID) agricultural data distributions.

> **Paper:** *A Federated Learning Framework for Non-IID Maize Disease Detection*  
> Sahil Pathak, Sujata Pal, Deepti R. Bathula — IIT Ropar

---

## Overview

Agricultural disease data is distributed across farms, cooperatives, and research institutions and is rarely Independent and Identically Distributed (IID). 
Two main components:

| Component | Description |
|-----------|-------------|
| **FedCADR** | Federated Class-aware Adaptive Drift-controlled Residual aggregation — separates feature-level averaging from class-wise classifier residual updates with magnitude normalization and adaptive scaling (β) |
| **MaizeDNIID** | Non-IID maize disease benchmark with explicit random partitioning and quantified heterogeneity metrics |



## Repository structure

```
FL_P2/
├── README.md
├── new code/                                  # Clean, commented FL code (recommended)
│   ├── run/
│   │   ├── code/                           # Shared Python package
│   │   ├── utils/                             # Experiment CLI helpers
│   │   ├── baselines/ | proposed/ | experiments/
│   │   └── create_partitions.py
│   ├── FL_dataset_creation.ipynb              # MaizeDNIID partition creation
│   ├── FD_Maize_HETEROGENEOUS_multi_processing.ipynb
│   ├── FD_Maize_HETEROGENEOUS.ipynb
│   ├── FD_Maize.ipynb
│   ├── test_fed_agg_sync.ipynb
│   ├── Maize_Hetero.ipynb
│   └── requirements.txt
├── saved/code/                                # Original experiment notebooks (archive)
├── FL__1_Minimized/
│   └── main.tex                               # Paper source (IEEE journal format)
└── Dataset_Maize_DD/                          # Dataset (upload separately)
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

Partitions are **not** Dirichlet-based. Client data is built with a **random class-and-sample assignment** procedure (`create_federated_dataset` in `new code/FL_dataset_creation.ipynb`):

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
| Local epochs (E) | **2, 5, 10** (swept) |
| Clients (K) | **10, 32, 64** |
| Client participation | **Full** — all K clients every round |
| Local train/val split | 80% / 20% |
| FedCADR scaling (β) | **0.3** |
| Input resolution | 128×128 (FL notebooks), ImageNet normalization |
| Random seeds | Single-run experiments (no multi-seed averaging reported) |

### Evaluated aggregators

FedAvg, FedAvgM, FedAdam, FedYogi, FedNova, FedProx, FedMedian, Krum, Trimmed Mean, FedLPA, FedRAM, FedAdaptive, and **FedCADR** (proposed).

### Evaluated models

EfficientNet, MobileNet, MobileViT, SqueezeNet

---

## Getting started

### Requirements

- Python 3.8+
- PyTorch + Torchvision
- NumPy, Pandas, scikit-learn, Matplotlib, tqdm
- Jupyter Notebook or VS Code
- TensorFlow/Keras (optional — centralized baseline notebooks only)
- CUDA-capable GPU recommended (experiments used NVIDIA Tesla P100)

Install dependencies:

```bash
pip install -r requirements.txt
# or from new code/:
# pip install -r "new code/requirements.txt"
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

num_clients = 32
dataset_dir = "Dataset_Maize_DD/1_CREATED_NEW_SAMPLE_split/train"
output_dir = "Federated_data_maize/client_32"

create_federated_dataset(dataset_dir, num_clients, output_dir)
```

Use `count_images_per_class()` and `visualize_data_distribution()` in the same notebook to inspect skew across clients.

### Step 2 — Run federated training

Open `code/FD_Maize_HETEROGENEOUS_multi_processing.ipynb` and set the client folder:

```python
base_client_folder = "Federated_data_maize/client_32"

global_model, client_losses, client_accuracies, global_accuracies = StartFedLearnining(
    base_client_folder,
    num_clients=32,
    rounds=50,
    aggregation_method=federated_avg,  # or , krum, median_aggregation, etc.
)
```

The pipeline:

1. Loads per-client datasets from `base_client_folder`
2. Initializes a shared global model (7-class head)
3. For each of **T** rounds: all clients train locally for **E** epochs, then the server aggregates
4. Evaluates the global model on the held-out **validation** set each round

### Step 3 — Compare aggregation methods

Use `code/test_fed_agg_sync.ipynb` for a lighter-weight aggregator comparison loop.

### Step 4 — Generate plots

Use `code/plots.ipynb` to reproduce convergence curves, client accuracy boxplots, and ablation figures.

---

## Notebook guide

| Notebook | Purpose |
|----------|---------|
| `sample_data.ipynb` | Federated partition creation, augmentation pipelines, distribution visualization |
| `FD_Maize_HETEROGENEOUS_multi_processing.ipynb` | **Main FL pipeline** — multi-threaded client training, 10/32/64 clients, 50 rounds |
| `FD_Maize_HETEROGENEOUS.ipynb` | Earlier heterogeneous FL implementation |
| `test_fed_agg_sync.ipynb` | Sync FL with multiple aggregation methods |
| `plots.ipynb` | Result visualization for the paper |
| `Maize_experiments_on_raw*.ipynb` | Centralized (non-FL) baselines and proposed architecture search |
| `FL_Distillation*.ipynb` | Federated knowledge distillation experiments |

---

## FedCADR (summary)

FedCADR splits the global model into a **feature extractor** and a **classifier head**:

1. **Feature layers** — sample-weighted FedAvg across all clients.
2. **Classifier rows (per class c)** — for each class:
   - Compute residual updates Δ<sub>k,c</sub> = local − global classifier weights.
   - Discard near-zero residuals (|Δ| &lt; ε) to exclude absent-class clients.
   - Keep directionally consistent updates (positive cosine similarity with class consensus).
   - Magnitude-normalize, weight by client data contribution, scale by **β = 0.3**, and apply.

See Algorithm 1 in `FL__1_Minimized/main.tex` for full pseudocode.

---


## Authors

- **Sahil Pathak** — sahil.23csz0005@iitrpr.ac.in
- **Sujata Pal** — sujata@iitrpr.ac.in
- **Deepti R. Bathula** — bathula@iitrpr.ac.in

Department of Computer Science & Engineering, IIT Ropar

---

## License

This repository is released for academic research. Dataset images retain the licenses of their original sources (PlantVillage, field collections, etc.). Contact the authors for commercial use.

---

## Roadmap

- [ ] Release MaizeDNIID partition index files for reproducibility
- [ ] Add `requirements.txt` and seed-controlled partitioning script
- [ ] Refactor notebooks into importable Python modules (`run/code/`)
- [ ] Multi-seed experiment reporting
- [ ] Partial client participation and edge-device evaluation
