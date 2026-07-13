"""
Original ``Run_FED`` workflow from FED_MDD_V3_CONTEXT_EXPs.ipynb.

One call per model runs all aggregators, records full round metrics, and writes
a single CSV matching the notebook column schema.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from .aggregation import aggregate, get_aggregator
from .config import FLConfig, DEFAULT_FL_CONFIG
from .dataset import FederatedDataset, build_global_class_mapping, get_fl_transform
from .metrics import compare_to_average, evaluate, histories_to_df


def _split_client_dataset(dataset, ratio: float, seed: int | None):
    """Notebook-style train/test split with fixed seed per client load."""
    rng = np.random.default_rng(seed if seed is not None else 42)
    idx = rng.permutation(len(dataset))
    split = int(len(idx) * ratio)
    return Subset(dataset, idx[:split].tolist()), Subset(dataset, idx[split:].tolist())


def load_run_fed_clients(
    base_client_folder: str | Path,
    val_dir: str | Path,
    cfg: FLConfig = DEFAULT_FL_CONFIG,
    device: torch.device | None = None,
) -> tuple[dict, DataLoader]:
    """
    Load client train/test loaders and a global validation loader.

    Each client folder is split with ``cfg.local_train_ratio`` (default 0.8;
    original notebook used 0.9). ``size`` is the train-set length (FedAvg weight).
    """
    base = Path(base_client_folder)
    transform = get_fl_transform(cfg.image_size)
    class_to_idx = build_global_class_mapping(val_dir)

    global_test_ds = FederatedDataset(val_dir, transform=transform, class_to_idx=class_to_idx)
    global_test_loader = DataLoader(
        global_test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        pin_memory=device is not None and device.type == "cuda",
    )

    clients: dict = {}
    for name in sorted(os.listdir(base)):
        if not name.startswith("client"):
            continue
        client_path = base / name
        if not client_path.is_dir():
            continue

        dataset = FederatedDataset(client_path, transform=transform, class_to_idx=class_to_idx)
        train_ds, test_ds = _split_client_dataset(dataset, cfg.local_train_ratio, cfg.seed)
        clients[name] = {
            "train": DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                pin_memory=device is not None and device.type == "cuda",
            ),
            "test": DataLoader(
                test_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                pin_memory=device is not None and device.type == "cuda",
            ),
            "size": len(train_ds),
        }

    return clients, global_test_loader


def _build_optimizer(model: nn.Module, cfg: FLConfig) -> optim.Optimizer:
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.learning_rate)
    return optim.SGD(model.parameters(), lr=cfg.learning_rate)


def local_train_run_fed(
    model: nn.Module,
    global_state: dict,
    loader: DataLoader,
    cfg: FLConfig,
    device: torch.device,
    method: str = "fedavg",
) -> tuple[dict, float]:
    """
    Local client training mirroring the notebook ``local_train``.

    Applies FedProx proximal term when *method* contains ``fedprox``.
    Returns CPU-detached state_dict and training duration (seconds).
    """
    model.load_state_dict(global_state)
    model.to(device)
    model.train()

    optimizer = _build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    use_fedprox = "fedprox" in method.lower()
    global_params = None
    if use_fedprox:
        global_params = {
            k: v.to(device) for k, v in global_state.items() if torch.is_floating_point(v)
        }

    start = time.time()
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)

            if use_fedprox and global_params is not None:
                prox = sum(
                    torch.norm(p - global_params[n]) ** 2
                    for n, p in model.named_parameters()
                    if n in global_params
                )
                loss = loss + (cfg.mu / 2) * prox

            loss.backward()
            optimizer.step()

    duration = time.time() - start

    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    state_dict_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return state_dict_cpu, duration


def _local_method(agg_name: str) -> str:
    """Map aggregator name to local training method (notebook convention)."""
    return agg_name.lower().split("_")[0]


def run_fed(
    model_name: str,
    aggregator_names: list[str],
    num_clients: int,
    cfg: FLConfig,
    model_factory: Callable,
    device: torch.device,
    out_dir: str | Path,
    base_client_folder: str | Path | None = None,
    val_dir: str | Path | None = None,
    csv_prefix: str = "",
    threaded: bool = True,
) -> tuple[dict[str, list[dict]], Path]:
    """
    Run the full ``Run_FED`` loop for one model and multiple aggregators.

    Saves:
      - One CSV: ``Client{K}_{csv_prefix}{model}_NUM_ROUNDS_{R}_LOCAL_EPOCHS_{E}.csv``
      - Per-aggregator weights: ``{model}__{agg}.pth``
      - Compact JSON summary: ``{model}__run_fed_summary.json``

    Returns:
        (histories, csv_path)
    """
    from .config import VAL_DIR
    from .dataset import federated_output_path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if base_client_folder is None:
        base_client_folder = federated_output_path(num_clients)
    if val_dir is None:
        val_dir = VAL_DIR

    clients, global_test_loader = load_run_fed_clients(
        base_client_folder, val_dir, cfg, device
    )

    global_models = {
        name: model_factory(cfg.num_classes) for name in aggregator_names
    }
    histories: dict[str, list[dict]] = {name: [] for name in aggregator_names}

    for agg_name in aggregator_names:
        agg_fn = get_aggregator(agg_name)
        local_method = _local_method(agg_name)
        model = global_models[agg_name]

        print(f"\n--- Aggregator: {agg_name.upper()} ---")
        print(f"local_method={local_method}")

        for rnd in range(cfg.rounds):
            print(f"\n========== ROUND {rnd} ==========")
            round_start = time.time()
            states, sizes, train_times = [], [], []

            global_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if threaded:
                with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
                    futures = []
                    for cid, data in clients.items():
                        local_model = copy.deepcopy(model)
                        future = executor.submit(
                            local_train_run_fed,
                            local_model,
                            global_state,
                            data["train"],
                            cfg,
                            device,
                            local_method,
                        )
                        futures.append((future, data["size"], data["test"]))

                    for future, size, _loader in futures:
                        state, t = future.result()
                        states.append(state)
                        sizes.append(size)
                        train_times.append(t)
            else:
                for _cid, data in clients.items():
                    local_model = copy.deepcopy(model)
                    state, t = local_train_run_fed(
                        local_model,
                        global_state,
                        data["train"],
                        cfg,
                        device,
                        local_method,
                    )
                    states.append(state)
                    sizes.append(data["size"])
                    train_times.append(t)

            agg_start = time.time()
            new_state = aggregate(
                states,
                agg_fn,
                client_weights=sizes,
                global_model=model,
                beta=cfg.fedcadr_beta,
                mu=cfg.mu,
                local_epochs=cfg.local_epochs,
            )
            model.load_state_dict(new_state)
            agg_time = time.time() - agg_start

            client_compare, variance = compare_to_average(model.state_dict(), states)
            print(f"Global variance: {variance}")

            eval_start = time.time()
            gl_loss, gl_acc, preds, labels = evaluate(model, global_test_loader, device)
            macro_f1 = f1_score(labels, preds, average="macro") if labels else 0.0

            client_metrics = [
                evaluate(model, data["test"], device)[:2] for data in clients.values()
            ]
            client_losses = [m[0] for m in client_metrics]
            client_accs = [m[1] for m in client_metrics]
            eval_time = time.time() - eval_start
            round_time = time.time() - round_start

            histories[agg_name].append({
                "round": rnd,
                "aggregation": agg_name,
                "global_acc": gl_acc,
                "global_loss": gl_loss,
                "macro_f1": macro_f1,
                "client_accs": client_accs,
                "client_losses": client_losses,
                "mean_client_acc": float(np.mean(client_accs)),
                "worst_client_acc": float(np.min(client_accs)),
                "std_client_acc": float(np.std(client_accs)),
                "avg_local_train_time": float(np.mean(train_times)),
                "aggregation_time": agg_time,
                "evaluation_time": eval_time,
                "round_time": round_time,
                "Global_variance": variance,
                "client_compare_global": client_compare,
            })

            print(
                f"[{agg_name.upper():>10}] "
                f"Global Acc: {gl_acc:.4f} | "
                f"Best: {max(client_accs):.4f} | "
                f"Mean: {np.mean(client_accs):.4f} | "
                f"Worst: {min(client_accs):.4f} | "
                f"aggregation_time: {agg_time:.2f}s"
            )

    csv_name = (
        f"Client{num_clients}_{csv_prefix}{model_name}"
        f"_NUM_ROUNDS_{cfg.rounds}_LOCAL_EPOCHS_{cfg.local_epochs}.csv"
    )
    csv_path = out_dir / csv_name
    df = histories_to_df(histories)
    df.to_csv(csv_path, index=False)
    print(f"Saved federated results to {csv_path}")

    summary = {
        "model": model_name,
        "num_clients": num_clients,
        "rounds": cfg.rounds,
        "local_epochs": cfg.local_epochs,
        "aggregators": {},
    }
    for agg_name, records in histories.items():
        final = records[-1] if records else {}
        ckpt = out_dir / f"{model_name}__{agg_name}.pth"
        torch.save(global_models[agg_name].state_dict(), ckpt)
        summary["aggregators"][agg_name] = {
            "final_global_acc": final.get("global_acc"),
            "final_global_loss": final.get("global_loss"),
            "final_macro_f1": final.get("macro_f1"),
            "final_mean_client_acc": final.get("mean_client_acc"),
            "weights": str(ckpt),
        }
        print(f"Saved weights → {ckpt}")

    summary_path = out_dir / f"{model_name}__run_fed_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {summary_path}")

    return histories, csv_path
