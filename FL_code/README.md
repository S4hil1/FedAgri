# FedAgri — Clean Federated Learning Package

All models, custom architectures, and aggregation algorithms used in the
FedAgri / MaizeDNIID experiments live in **`run/code/`**.

Original messy notebooks are archived under `saved/code/` and the paper-run
notebooks (`FED_MDD_V*.ipynb`) remain here for full experiment traces.

---

## Layout

```
new code/
├── run/
│   ├── code/                     # ★ importable implementation package
│   │   ├── config.py                # Paths + FLConfig (η, E, T, β, …)
│   │   ├── dataset.py               # MaizeDNIID Non-IID partitioning
│   │   ├── models.py                # ALL models (FedCAMNet, MD2, SE, KAN, …)
│   │   ├── aggregation.py           # ALL aggregators (FedCADR, FedAvg, …)
│   │   ├── training.py              # Local train / eval / plots
│   │   ├── federated_loop.py        # FL orchestration loops
│   │   ├── metrics.py               # evaluate / F1 / variance helpers
│   │   └── run_fed.py               # Original Run_FED CSV workflow
│   ├── utils/                       # CLI + experiment runners
│   ├── baselines/ | proposed/ | experiments/
│   └── create_partitions.py
├── FL_dataset_creation.ipynb        # Build client partitions
├── FD_Maize_HETEROGENEOUS_multi_processing.ipynb   # Main: FedCAMNet+FedCADR
├── FD_Maize_HETEROGENEOUS.ipynb
├── FD_Maize.ipynb
├── test_fed_agg_sync.ipynb          # Aggregator bake-off
├── Maize_Hetero.ipynb
├── FED_MDD_V*.ipynb                 # Full paper experiment traces
└── requirements.txt
```

Import setup (from ``new code/`` as cwd)::

```python
import sys
sys.path.insert(0, "run")
from code import get_model, FedCADR
```

---

## Install

```bash
cd "new code"
pip install -r requirements.txt
```

Place dataset at project root:

```
Dataset_Maize_DD/1_CREATED_NEW_SAMPLE_split/{train,validation,test}/
```

---

## Models (`code.models`)

### Proposed — FedCAMNet

| Class | Description |
|-------|-------------|
| `FedCAMNet` / `FederatedMobileNetKAN` | MobileNetV2 + **SE** channel attention + **KAN** head |
| `FederatedMobileNetKAN_Ablation` | Toggle `use_se` / `use_kan` |
| `ContextAwareMobileNet` | MobileNetV2 + ContextAwareModule |
| `FinalFLModel` | Backbone + `ContextKANHead` (RBF-KAN) |

Building blocks: `SEBlock`, `ChannelAttention`, `FeatureContext`,
`ContextAwareModule`, `KANLayer`, `KANLayerRBF`, `ContextKANHead`, `DualPooling`.

### Standard backbones

`get_resnet18_model`, `get_mobilenet_model`, `get_efficientnet_model`,
`get_squeezenet_model`, `get_shufflenet_model`, `get_densenet_model`

### Custom CNNs

| Class | Source |
|-------|--------|
| `MD2` | Lightweight DW-conv + SE (disease classification notebooks) |
| `MiniNetPP` | MiniNet++ PyTorch port |
| `SimpleCNN` | Early Maize_Hetero prototype |

### Factory

```python
import sys
sys.path.insert(0, "run")  # from new code/ cwd

from code import get_model

model = get_model("fedcamnet", num_classes=7)
model = get_model("resnet18", num_classes=7)
model = get_model("md2", num_classes=7)
model = get_model("fedcamnet_ablation", num_classes=7, use_se=True, use_kan=False)
```

---

## Aggregators (`code.aggregation`)

### Proposed — FedCADR

| Function | Description |
|----------|-------------|
| `fed_car` | Feature FedAvg + class-wise residual classifier (β / E) |
| `fed_car_plus` / `FedCADR` | Full FedCADR: near-zero skip + cosine filter + norm + β |
| `fed_dca` | Experimental cosine-weighted partial update |

### Baselines

| Function | Description |
|----------|-------------|
| `federated_avg` | Classic FedAvg |
| `weighted_federated_avg` | Sample-size weighted |
| `fedprox` | Proximal mix toward global |
| `federated_sgd` | Gradient average |
| `krum` | Byzantine-robust selection |
| `median_aggregation` | Coordinate-wise median |
| `trimmed_mean` | Trimmed mean |
| `fednova` | Normalized updates |
| `differential_privacy_aggregation` | FedAvg + Gaussian noise |
| `federated_momentum` | Momentum on deltas |
| `divergence_aware_agg` | Distance-based client weights |
| `fedadam` / `fedadagrad` | Stateful server optimizers |
| `FedAgri` | FedAvg + alignment loss (notebook experiment) |

```python
from code import get_aggregator, FedCADR, federated_avg

agg = get_aggregator("fedcadr")   # or "fedavg", "krum", …
```

---

## Runnable scripts (`run/`)

Separate entry points for **10 / 32 / 64** clients:

| Folder | Purpose |
|--------|---------|
| `run/baselines/` | Standard models × baseline aggregators |
| `run/proposed/` | **FedCAMNet + FedCADR** (paper main) |
| `run/experiments/` | Custom models + experimental aggregators + ablations |

```bash
# once: create Non-IID partitions
python run/create_partitions.py --clients 10 32 64 --seed 42

# baselines
python run/baselines/run_baselines_client10.py
python run/baselines/run_baselines_client32.py
python run/baselines/run_baselines_client64.py

# proposed FedAgri
python run/proposed/run_fedagri_client10.py
python run/proposed/run_fedagri_client32.py
python run/proposed/run_fedagri_client64.py

# custom / experimental
python run/experiments/run_experiments_client32.py
python run/experiments/run_ablation_client32.py
```

See [`run/README.md`](run/README.md) for all flags.

---

## Quick start (paper setup)

```python
import torch
from code import (
    FLConfig, VAL_DIR, federated_output_path,
    FedCAMNet, FedCADR,
)
from code.federated_loop import run_federated_learning_threaded

cfg = FLConfig(
    rounds=50, local_epochs=10, batch_size=16,
    learning_rate=0.001, optimizer="adam", fedcadr_beta=0.3,
)

model, losses, accs, global_accs = run_federated_learning_threaded(
    base_client_folder=federated_output_path(10),
    num_clients=10,
    val_dir=VAL_DIR,
    cfg=cfg,
    model_factory=FedCAMNet,
    aggregation_method=FedCADR,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

### Create Non-IID partitions first

```python
from code import create_federated_dataset, TRAIN_DIR, federated_output_path

create_federated_dataset(TRAIN_DIR, num_clients=32,
                         output_dir=federated_output_path(32), seed=42)
```

---

## Hyperparameters (`FLConfig`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| Optimizer | Adam | Paper table |
| Learning rate η | 0.001 | |
| Batch size | 16 | |
| Rounds T | 50 | |
| Local epochs E | 2 / 5 / 10 | Swept |
| FedCADR β | 0.3 | Scaling |
| Client participation | Full | All K every round |
| Local train/val | 80 / 20 | Per client |

---

## What was ported vs original notebooks

| Source | Now in |
|--------|--------|
| `FED_MDD_V2/V3` FedCAMNet, SE, KAN, Context | `code.models` |
| `FED_MDD_V2/V3` fed_car / fed_car_plus / fed_dca | `code.aggregation` |
| multi_processing aggregators + FedAgri | `code.aggregation` |
| MD2 / MiniNetPP / SEBlock (pytorch) | `code.models` |
| sample_data partitioning | `code.dataset` |
| FL training loops | `code.federated_loop` |

Notebooks under `new code/` are thin drivers that import from `code`.
