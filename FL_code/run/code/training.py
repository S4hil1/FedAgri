"""
Local client training, evaluation, and plotting utilities.
"""

from __future__ import annotations

import gc
import queue
import time
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .config import FLConfig, DEFAULT_FL_CONFIG


def split_dataset(dataset, split_ratio: float = 0.8):
    """
    Split a client dataset into local train / local validation subsets.

    Default 80/20 matches the MaizeDNIID protocol in the paper.
    """
    total = len(dataset)
    train_len = int(split_ratio * total)
    val_len = total - train_len
    return random_split(dataset, [train_len, val_len])


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy on a DataLoader."""
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total else 0.0


def _build_optimizer(model: nn.Module, cfg: FLConfig) -> optim.Optimizer:
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.learning_rate)
    return optim.SGD(model.parameters(), lr=cfg.learning_rate)


def train_client_sync(
    client_model: nn.Module,
    client_dataset,
    device: torch.device,
    cfg: FLConfig = DEFAULT_FL_CONFIG,
) -> tuple[dict, list[float], list[float]]:
    """
    Train one client sequentially (no threading).

    Returns:
        (state_dict, epoch_losses, epoch_accuracies)
    """
    train_set, val_set = split_dataset(client_dataset, cfg.local_train_ratio)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)

    client_model.to(device)
    client_model.train()
    optimizer = _build_optimizer(client_model, cfg)
    criterion = nn.CrossEntropyLoss()

    epoch_losses, epoch_accuracies = [], []

    for _ in range(cfg.local_epochs):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_losses.append(running_loss / max(len(train_loader), 1))
        epoch_accuracies.append(correct / max(total, 1))

    val_acc = evaluate_model(client_model, val_loader, device)
    print(
        f"Client trained | samples={len(train_set)} | "
        f"epochs={cfg.local_epochs} | local val acc={val_acc:.3f}"
    )
    return client_model.state_dict(), epoch_losses, epoch_accuracies


def train_client_threaded(
    client_model: nn.Module,
    client_dataset,
    device: torch.device,
    cfg: FLConfig,
    result_queue_model: queue.Queue,
    result_queue_losses: queue.Queue,
    result_queue_accuracy: queue.Queue,
    client_name: str,
    model_path: str,
) -> None:
    """
    Thread worker for parallel client training (multi_processing notebook).

    Saves weights to disk and pushes metrics into thread-safe queues.
    """
    try:
        state_dict, losses, accs = train_client_sync(
            client_model, client_dataset, device, cfg
        )
        torch.save(state_dict, model_path)
        result_queue_model.put((client_name, model_path))
        result_queue_losses.put((client_name, losses))
        result_queue_accuracy.put((client_name, accs))
    finally:
        client_model.to("cpu")
        del client_model
        torch.cuda.empty_cache()
        gc.collect()


def plot_client_performance(client_losses: dict, client_accuracies: dict) -> None:
    """Plot per-client training loss curves across all local epochs."""
    plt.figure(figsize=(10, 5))
    for client_id, losses in client_losses.items():
        if losses:
            plt.plot(range(1, len(losses) + 1), losses, label=f"client_{client_id}")
    plt.title("Client Training Loss per Local Epoch")
    plt.xlabel("Local Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for client_id, accs in client_accuracies.items():
        if accs:
            plt.plot(range(1, len(accs) + 1), accs, label=f"client_{client_id}")
    plt.title("Client Training Accuracy per Local Epoch")
    plt.xlabel("Local Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()
