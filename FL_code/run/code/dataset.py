"""
MaizeDNIID dataset utilities.

- create_federated_dataset: build Non-IID client partitions (random class + sample assignment)
- FederatedDataset: PyTorch Dataset over per-client class folders
- Helpers for loading all clients and visualizing label skew
"""

from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import FEDERATED_ROOT, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, NUM_CLASSES


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp",
)


def set_seed(seed: Optional[int]) -> None:
    """Fix random seeds for reproducible Non-IID partitioning."""
    if seed is not None:
        random.seed(seed)


def get_fl_transform(image_size: tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """Standard preprocessing used during federated training."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_global_class_mapping(root_dir: str | Path) -> dict[str, int]:
    """
    Build a consistent class-name → index map from a folder of class subdirectories.

    Uses sorted class folder names so every client shares the same label encoding.
    """
    root = Path(root_dir)
    classes = sorted(
        d.name for d in root.iterdir() if d.is_dir()
    )
    return {name: idx for idx, name in enumerate(classes)}


def create_federated_dataset(
    dataset_dir: str | Path,
    num_clients: int,
    output_dir: str | Path,
    min_images_per_class: int = 5,
    seed: Optional[int] = None,
) -> None:
    """
    Create MaizeDNIID-style Non-IID client partitions.

    Protocol (see paper Section: Federated Partitioning):
      1. For each client, randomly pick 1..C disease classes.
      2. For each chosen class, randomly sample min_images..all images.
      3. Copy images into client_{id}/{class}/ folders.

    Notes:
      - This is NOT Dirichlet partitioning.
      - The same image may appear at multiple clients (overlap allowed).
      - Images are copied, not moved, so the central train set is preserved.

    Args:
        dataset_dir: Central train folder with one subfolder per class.
        num_clients: Number of simulated clients K.
        output_dir: Destination root (e.g. Federated_data_maize/client_32).
        min_images_per_class: Lower bound on samples drawn per assigned class.
        seed: Optional RNG seed for reproducibility.
    """
    set_seed(seed)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index all images by class name
    class_names = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    class_to_images: dict[str, list[Path]] = {}
    for class_name in class_names:
        class_path = dataset_dir / class_name
        class_to_images[class_name] = [
            class_path / f for f in os.listdir(class_path)
            if (class_path / f).is_file()
        ]

    num_classes = len(class_names)
    client_data: dict[int, dict] = {
        i: {"classes": [], "images": []} for i in range(num_clients)
    }

    # Assign random class subsets and image subsets to each client
    for client_id in range(num_clients):
        k_classes = random.randint(1, num_classes)
        selected = random.sample(class_names, k_classes)
        client_data[client_id]["classes"] = selected

        for class_name in selected:
            images = class_to_images[class_name]
            n_sample = random.randint(min(min_images_per_class, len(images)), len(images))
            client_data[client_id]["images"].extend(random.sample(images, n_sample))

    # Materialize client folders on disk
    for client_id, info in client_data.items():
        client_folder = output_dir / f"client_{client_id}"
        client_folder.mkdir(parents=True, exist_ok=True)

        for class_name in info["classes"]:
            class_folder = client_folder / class_name
            class_folder.mkdir(parents=True, exist_ok=True)

            for image_path in info["images"]:
                if class_name in str(image_path):
                    shutil.copy2(image_path, class_folder / image_path.name)

    print(f"Federated dataset created: {num_clients} clients → '{output_dir}'")


class FederatedDataset(Dataset):
    """
    Image dataset for one federated client.

    Expects client_folder/class_name/image.ext layout produced by
    create_federated_dataset().
    """

    def __init__(
        self,
        client_folder: str | Path,
        transform: transforms.Compose | None = None,
        class_to_idx: dict[str, int] | None = None,
    ):
        self.client_folder = Path(client_folder)
        self.transform = transform
        self.data: list[str] = []
        self.labels: list[str] = []

        for class_name in os.listdir(self.client_folder):
            class_folder = self.client_folder / class_name
            if not class_folder.is_dir():
                continue
            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith(IMAGE_EXTENSIONS):
                    self.data.append(str(class_folder / img_name))
                    self.labels.append(class_name)

        if class_to_idx is None:
            class_to_idx = {
                name: idx for idx, name in enumerate(sorted(set(self.labels)))
            }
        self.class_to_idx = class_to_idx
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = Image.open(self.data[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_client_dataset(
    client_folder: str | Path,
    class_to_idx: dict[str, int],
    transform: transforms.Compose | None = None,
) -> FederatedDataset:
    """Load a single client's dataset (returns Dataset, not DataLoader)."""
    if transform is None:
        transform = get_fl_transform()
    return FederatedDataset(client_folder, transform=transform, class_to_idx=class_to_idx)


def load_all_client_datasets(
    base_client_folder: str | Path,
    class_to_idx: dict[str, int],
    transform: transforms.Compose | None = None,
) -> list[FederatedDataset]:
    """
    Load datasets for every client under base_client_folder.

    Args:
        base_client_folder: e.g. Federated_data_maize/client_32
        class_to_idx: Shared label map (typically from global validation set).

    Returns:
        List of FederatedDataset objects, one per client subfolder.
    """
    base = Path(base_client_folder)
    client_folders = sorted(
        p for p in base.iterdir() if p.is_dir()
    )
    datasets = []
    for folder in client_folders:
        print(f"Loading {folder}...")
        datasets.append(load_client_dataset(folder, class_to_idx, transform))
    return datasets


def count_images_per_class(base_client_folder: str | Path) -> dict[str, dict[str, int]]:
    """
    Count images per class for each client.

    Returns:
        {client_folder_path: {class_name: count}}
    """
    base = Path(base_client_folder)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for client in sorted(p for p in base.iterdir() if p.is_dir()):
        for class_name in os.listdir(client):
            class_folder = client / class_name
            if class_folder.is_dir():
                n = sum(
                    1 for f in class_folder.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                counts[str(client)][class_name] = n
    return dict(counts)


def visualize_data_distribution(
    class_counts: dict,
    save_path: str | Path | None = None,
) -> None:
    """Bar chart of per-class image counts across clients."""
    fig, ax = plt.subplots(figsize=(12, 8))
    clients = list(class_counts.keys())
    classes = sorted({cls for c in class_counts.values() for cls in c.keys()})

    matrix = []
    for client in clients:
        row = [class_counts[client].get(cls, 0) for cls in classes]
        matrix.append(row)

    matrix = np.array(matrix)
    bar_width = 0.8 / max(len(clients), 1)
    index = np.arange(len(classes))

    for i, (client, row) in enumerate(zip(clients, matrix)):
        label = Path(client).name
        ax.bar(index + i * bar_width, row, bar_width, label=label)

    ax.set_xlabel("Classes")
    ax.set_ylabel("Number of Images")
    ax.set_title("Non-IID Data Distribution Across Clients")
    ax.set_xticks(index + bar_width * (len(clients) - 1) / 2)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def federated_output_path(num_clients: int) -> Path:
    """Standard output path for K-client partitions."""
    return FEDERATED_ROOT / f"client_{num_clients}"
