"""
Federated learning orchestration loops.

Three experiment tracks from the original notebooks:
  1. Heterogeneous (folder-based Non-IID) — sequential & threaded
  2. Homogeneous (random_split of centralized data) — FD_Maize.ipynb
  3. Synchronous aggregation benchmark — test_fed_agg_sync.ipynb
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from .aggregation import aggregate, federated_avg
from .config import FLConfig, DEFAULT_FL_CONFIG, VAL_DIR
from .dataset import (
    FederatedDataset,
    build_global_class_mapping,
    get_fl_transform,
    load_all_client_datasets,
)
from .models import FedCAMNet, get_mobilenet_model, get_resnet18_model
from .training import (
    evaluate_model,
    plot_client_performance,
    train_client_sync,
    train_client_threaded,
)

# Default for paper experiments: FedCAMNet (MobileNetV2 + SE + KAN)
DEFAULT_MODEL_FACTORY = FedCAMNet


def get_free_memory_mb() -> float:
    """Return free GPU memory in MB (requires pynvml + NVIDIA GPU)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free / (1024 ** 2)
    except Exception:
        return float("inf")


def closest_divisor(x: int, y: int) -> int:
    """Largest divisor of x that does not exceed y (for equal client shards)."""
    for i in range(y, 0, -1):
        if x % i == 0:
            return i
    return 1


def get_homogeneous_client_data(
    data_dir: str | Path,
    num_clients: int,
    val_fraction: float = 0.1,
    transform: transforms.Compose | None = None,
) -> tuple[list[Dataset], Dataset, int]:
    """
    Split a centralized ImageFolder-style dataset into equal IID client shards.

    Used by FD_Maize.ipynb (homogeneous FL baseline).

    Returns:
        (client_datasets, global_test_dataset, actual_num_clients)
    """
    from torchvision.datasets import ImageFolder

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    full_dataset = ImageFolder(str(data_dir), transform=transform)
    train_size = int((1 - val_fraction) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    n = closest_divisor(len(train_dataset), num_clients)
    shard_size = len(train_dataset) // n
    client_datasets = random_split(
        train_dataset,
        [shard_size] * n,
    )
    return list(client_datasets), test_dataset, n


def run_federated_learning_sequential(
    base_client_folder,
    num_clients: int,
    val_dir=VAL_DIR,
    cfg: FLConfig = DEFAULT_FL_CONFIG,
    model_factory: Callable = DEFAULT_MODEL_FACTORY,
    aggregation_method: Callable = federated_avg,
    device=None,
) -> tuple[nn.Module, dict, dict, list[float]]:
    """
    Sequential heterogeneous FL (FD_Maize_HETEROGENEOUS.ipynb).

    Defaults to FedCAMNet + pluggable aggregation (FedAvg / FedCADR / …).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_fl_transform(cfg.image_size)
    global_model = model_factory(cfg.num_classes).to(device)

    class_to_idx = build_global_class_mapping(val_dir)
    test_dataset = FederatedDataset(val_dir, transform=transform, class_to_idx=class_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

    client_datasets = load_all_client_datasets(base_client_folder, class_to_idx, transform)
    print(f"num_clients: {num_clients} | model: {model_factory.__name__} | agg: {aggregation_method.__name__}")

    client_losses = {i: [] for i in range(num_clients)}
    client_accuracies = {i: [] for i in range(num_clients)}
    global_accuracies: list[float] = []

    for round_i in range(cfg.rounds):
        print(f"Round {round_i + 1}/{cfg.rounds}")
        client_states = []

        for i in range(num_clients):
            client_model = model_factory(cfg.num_classes)
            client_model.load_state_dict(global_model.state_dict())
            state_dict, losses, accs = train_client_sync(
                client_model, client_datasets[i], device, cfg
            )
            client_states.append(state_dict)
            client_losses[i].extend(losses)
            client_accuracies[i].extend(accs)

        client_weights = [len(d) for d in client_datasets]
        aggregated = aggregate(
            client_states,
            aggregation_method,
            client_weights=client_weights,
            global_model=global_model,
            beta=cfg.fedcadr_beta,
            local_epochs=cfg.local_epochs,
        )
        global_model.load_state_dict(aggregated)

        acc = evaluate_model(global_model, test_loader, device)
        global_accuracies.append(acc)
        print(f"Global validation accuracy: {acc * 100:.2f}%")

    print("Federated learning complete.")
    return global_model, client_losses, client_accuracies, global_accuracies


def run_federated_learning_threaded(
    base_client_folder,
    num_clients: int,
    val_dir=VAL_DIR,
    cfg: FLConfig = DEFAULT_FL_CONFIG,
    aggregation_method: Callable = federated_avg,
    model_factory: Callable = DEFAULT_MODEL_FACTORY,
    device=None,
    save_dir: str = "saved_client_models",
    required_memory_mb: float = 2000,
) -> tuple[nn.Module, dict, dict, list[float]]:
    """
    Threaded heterogeneous FL (FD_Maize_HETEROGENEOUS_multi_processing.ipynb).

    Clients train in parallel threads with GPU memory gating.
    Supports pluggable aggregation methods including FedCADR.
    Default model: FedCAMNet.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    transform = get_fl_transform(cfg.image_size)
    global_model = model_factory(cfg.num_classes)

    class_to_idx = build_global_class_mapping(val_dir)
    test_dataset = FederatedDataset(val_dir, transform=transform, class_to_idx=class_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)
    client_datasets = load_all_client_datasets(base_client_folder, class_to_idx, transform)

    print(
        f"num_clients: {num_clients} | model: {model_factory.__name__} | "
        f"agg: {aggregation_method.__name__}"
    )
    client_losses = {i: [] for i in range(num_clients)}
    client_accuracies = {i: [] for i in range(num_clients)}
    global_accuracies: list[float] = []

    q_model: queue.Queue = queue.Queue()
    q_losses: queue.Queue = queue.Queue()
    q_accs: queue.Queue = queue.Queue()

    for round_i in range(cfg.rounds):
        print(f"Round {round_i + 1}/{cfg.rounds}")
        threads: list[threading.Thread] = []

        for i in range(num_clients):
            client_model = model_factory(cfg.num_classes)
            client_model.load_state_dict(global_model.state_dict())
            client_name = f"client_{i}"
            model_path = os.path.join(save_dir, f"{client_name}.pth")

            while True:
                if get_free_memory_mb() > required_memory_mb:
                    t = threading.Thread(
                        target=train_client_threaded,
                        args=(
                            client_model,
                            client_datasets[i],
                            device,
                            cfg,
                            q_model,
                            q_losses,
                            q_accs,
                            client_name,
                            model_path,
                        ),
                        daemon=True,
                    )
                    t.start()
                    threads.append(t)
                    break
                print(f"Waiting for GPU memory to train {client_name}...")
                time.sleep(60)

        for t in threads:
            t.join()

        client_states = []
        while not q_model.empty():
            name, path = q_model.get()
            checkpoint = torch.load(path, map_location="cpu")
            client_states.append(checkpoint)

        while not q_losses.empty():
            name, losses = q_losses.get()
            client_losses[int(name.split("_")[1])].extend(losses)

        while not q_accs.empty():
            name, accs = q_accs.get()
            client_accuracies[int(name.split("_")[1])].extend(accs)

        client_weights = [len(d) for d in client_datasets]
        aggregated = aggregate(
            client_states,
            aggregation_method,
            client_weights=client_weights,
            global_model=global_model,
            beta=cfg.fedcadr_beta,
            local_epochs=cfg.local_epochs,
        )
        global_model.load_state_dict(aggregated)

        acc = evaluate_model(global_model, test_loader, device)
        global_accuracies.append(acc)
        print(f"Global validation accuracy: {acc * 100:.2f}%")

    print("Federated learning complete.")
    return global_model, client_losses, client_accuracies, global_accuracies


def run_federated_learning_homogeneous(
    data_dir: str | Path,
    num_clients: int,
    cfg: FLConfig = DEFAULT_FL_CONFIG,
    model_factory: Callable = get_resnet18_model,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict, dict]:
    """
    Homogeneous FL via random_split (FD_Maize.ipynb).

    Splits a single centralized dataset into equal IID shards.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_data, test_dataset, actual_clients = get_homogeneous_client_data(
        data_dir, num_clients
    )
    print(f"num_clients (adjusted): {actual_clients}")

    global_model = model_factory(cfg.num_classes).to(device)
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size)

    client_losses = {i: [] for i in range(actual_clients)}
    client_accuracies = {i: [] for i in range(actual_clients)}

    for round_i in range(cfg.rounds):
        print(f"Round {round_i + 1}/{cfg.rounds}")
        client_states = []

        for i in range(actual_clients):
            client_model = model_factory(cfg.num_classes)
            client_model.load_state_dict(global_model.state_dict())
            state_dict, losses, accs = train_client_sync(
                client_model, client_data[i], device, cfg
            )
            client_states.append(state_dict)
            client_losses[i].extend(losses)
            client_accuracies[i].extend(accs)

        global_model.load_state_dict(federated_avg(client_states))
        acc = evaluate_model(global_model, test_loader, device)
        print(f"Global test accuracy: {acc * 100:.2f}%")

    return global_model, client_losses, client_accuracies


def run_federated_train_sync(
    client_data,
    global_model: nn.Module,
    test_loader: DataLoader,
    aggregation_method: Callable,
    client_weights: list,
    cfg: FLConfig = DEFAULT_FL_CONFIG,
    device=None,
    model_factory: Callable = None,
) -> nn.Module:
    """
    Synchronous FL loop for aggregation benchmarking (test_fed_agg_sync.ipynb).

    Uses the same architecture as `global_model` for each client
    (or `model_factory` if provided).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = global_model.to(device)
    if model_factory is None:
        # Rebuild clients with same class as global_model
        model_cls = type(global_model)

        def model_factory(nc):
            return model_cls(num_classes=nc)

    for round_num in range(cfg.rounds):
        print(f"Round {round_num + 1}/{cfg.rounds}")
        client_states = []

        for client_id, dataset in client_data.items():
            print(f"  Training client {client_id}...")
            client_model = model_factory(cfg.num_classes)
            client_model.load_state_dict(global_model.state_dict())
            state_dict, _, _ = train_client_sync(client_model, dataset, device, cfg)
            client_states.append(state_dict)

        aggregated = aggregate(
            client_states,
            aggregation_method,
            client_weights=client_weights,
            global_model=global_model,
            beta=cfg.fedcadr_beta,
            local_epochs=cfg.local_epochs,
        )
        global_model.load_state_dict(aggregated)

        acc = evaluate_model(global_model, test_loader, device)
        print(f"  [{aggregation_method.__name__}] validation accuracy: {acc * 100:.2f}%")

    return global_model


# Legacy alias (typo preserved for notebook compatibility)
StartFedLearnining = run_federated_learning_threaded
