"""
Global configuration for FedAgri federated learning experiments.

Paths resolve relative to the project root (parent of ``new code/``) so
runnable scripts work whether launched from ``FL_P2/`` or ``new code/``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# new code/run/code/config.py → project root = parents[3], new code = parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_NEW_CODE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_root() -> Path:
    """Prefer project-root Dataset_Maize_DD; fall back to cwd / new code."""
    candidates = [
        _PROJECT_ROOT / "Dataset_Maize_DD",
        Path.cwd() / "Dataset_Maize_DD",
        _NEW_CODE_ROOT / "Dataset_Maize_DD",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _resolve_fed_root() -> Path:
    candidates = [
        _PROJECT_ROOT / "Federated_data_maize",
        Path.cwd() / "Federated_data_maize",
        _NEW_CODE_ROOT / "Federated_data_maize",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


DATASET_ROOT = _resolve_data_root()
TRAIN_DIR = DATASET_ROOT / "1_CREATED_NEW_SAMPLE_split" / "train"
VAL_DIR = DATASET_ROOT / "1_CREATED_NEW_SAMPLE_split" / "validation"
TEST_DIR = DATASET_ROOT / "1_CREATED_NEW_SAMPLE_split" / "test"
FEDERATED_ROOT = _resolve_fed_root()

HOMOGENEOUS_DATA_DIR = _PROJECT_ROOT / "Maize_Resized_Images"

NUM_CLASSES = 7
CLASS_NAMES = [
    "Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
    "MLN",
    "faw",
    "maize_streak",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = (128, 128)


@dataclass
class FLConfig:
    """Hyperparameters aligned with the FedAgri paper."""

    num_classes: int = NUM_CLASSES
    rounds: int = 50
    local_epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    local_train_ratio: float = 0.8  # client train/test split; notebook used 0.9
    client_participation: float = 1.0
    fedcadr_beta: float = 0.3
    mu: float = 0.01  # FedProx proximal coefficient (local + server)
    max_workers: int = 10  # ThreadPoolExecutor workers for parallel local train
    image_size: tuple = IMAGE_SIZE
    min_images_per_class: int = 5
    seed: Optional[int] = None


DEFAULT_FL_CONFIG = FLConfig()
PROJECT_ROOT = _PROJECT_ROOT
NEW_CODE_ROOT = _NEW_CODE_ROOT
