# Runnable FedAgri Experiments

Separate entry-point scripts for **10 / 32 / 64** clients, split by experiment type.

```
run/
├── code/                            # ★ FL package (models, agg, Run_FED, …)
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   ├── aggregation.py
│   ├── training.py
│   ├── federated_loop.py
│   ├── metrics.py
│   └── run_fed.py
├── _common.py                          # thin shim → utils (compat)
├── create_partitions.py                # build Non-IID client folders
├── utils/                              # shared helpers (import from here)
│   ├── __init__.py                     # public API re-exports
│   ├── paths.py                        # sys.path + results_dir
│   ├── presets.py                      # model / aggregator lists
│   ├── cli.py                          # parse_common_args
│   ├── config.py                       # build_config → FLConfig
│   ├── models.py                       # model_factory
│   ├── io.py                           # save_json
│   └── experiment.py                   # run_single / run_experiment
├── baselines/                          # standard models × baseline aggregators
│   ├── run_baselines_client10.py
│   ├── run_baselines_client32.py
│   └── run_baselines_client64.py
├── proposed/                           # FedCAMNet + FedCADR (paper main)
│   ├── run_fedagri_client10.py
│   ├── run_fedagri_client32.py
│   └── run_fedagri_client64.py
└── experiments/                        # custom models + experimental aggregators
    ├── run_experiments_client10.py
    ├── run_experiments_client32.py
    ├── run_experiments_client64.py
    └── run_ablation_client32.py        # SE/KAN / β / E sweeps
```

Results are written to `new code/results/<mode>/client_<K>/<timestamp>/`.

Each model run produces a **CSV with original notebook parity** — one file per model
covering all aggregators, with columns: `round`, `aggregation`, `global_acc`,
`global_loss`, `macro_f1`, `client_accs`, `client_losses`, `mean_client_acc`,
`worst_client_acc`, `std_client_acc`, timing fields, `Global_variance`, and
`client_compare_global`. Per-aggregator `.pth` weights and a compact JSON summary
are saved alongside.

---

## 0. Create partitions (once)

From the `new code/` directory:

```bash
python run/create_partitions.py --clients 10 32 64 --seed 42 --plot
```

Requires `Dataset_Maize_DD/1_CREATED_NEW_SAMPLE_split/train/` at the project root.

---

## 1. Baselines

Standard backbones: ResNet18, MobileNet, EfficientNet, SqueezeNet  
Aggregators: FedAvg, weighted FedAvg, FedProx, Median, Trimmed Mean, Krum

```bash
python run/baselines/run_baselines_client10.py
python run/baselines/run_baselines_client32.py
python run/baselines/run_baselines_client64.py

# subset
python run/baselines/run_baselines_client10.py --models resnet18 mobilenet --aggregators fedavg median --rounds 20
```

---

## 2. Proposed (FedAgri)

**FedCAMNet + FedCADR** only — main paper configuration.

```bash
python run/proposed/run_fedagri_client10.py
python run/proposed/run_fedagri_client32.py
python run/proposed/run_fedagri_client64.py

python run/proposed/run_fedagri_client32.py --local-epochs 5 --beta 0.3 --rounds 50
```

---

## 3. Custom / experimental

Custom models: FedCAMNet, FedCAMNet-ablation, MD2, MiniNetPP, ContextAware, MobileNet  
Aggregators: FedCADR, FedCAR, FedDCA, FedAvg, Median, Trimmed Mean, Krum, FedProx

```bash
python run/experiments/run_experiments_client10.py
python run/experiments/run_experiments_client32.py
python run/experiments/run_experiments_client64.py

python run/experiments/run_experiments_client32.py --models fedcamnet md2 --aggregators fedcadr fedavg
```

### Ablations (32 clients)

```bash
python run/experiments/run_ablation_client32.py                # all
python run/experiments/run_ablation_client32.py --only components
python run/experiments/run_ablation_client32.py --only beta
python run/experiments/run_ablation_client32.py --only epochs
```

---

## Common flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--rounds` | 50 | Communication rounds |
| `--local-epochs` | 10 | Local epochs E |
| `--batch-size` | 16 | Local batch size |
| `--lr` | 0.001 | Learning rate |
| `--beta` | 0.3 | FedCADR scaling β |
| `--seed` | 42 | RNG seed |
| `--train-ratio` | 0.8 | Client train/test split (notebook used 0.9) |
| `--mu` | 0.01 | FedProx proximal coefficient |
| `--max-workers` | 10 | Parallel local-train threads |
| `--csv-prefix` | (empty) | Prefix for CSV filename |
| `--device` | auto | e.g. `cuda:0` / `cpu` |
| `--sequential` | off | Disable threaded clients |
| `--models` | preset | Override model list |
| `--aggregators` | preset | Override aggregator list |

---

## What each mode runs

| Mode | Models | Aggregators |
|------|--------|-------------|
| **baseline** | resnet18, mobilenet, efficientnet, squeezenet | fedavg, weighted_fedavg, fedprox, median, trimmed_mean, krum |
| **proposed** | fedcamnet | fedcadr |
| **experiment** | fedcamnet, fedcamnet_ablation, md2, mininetpp, context_aware, mobilenet | fedavg, fedcadr, fed_car, fed_dca, median, trimmed_mean, krum, fedprox |
