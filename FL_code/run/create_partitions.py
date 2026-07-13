"""Create MaizeDNIID Non-IID partitions for K ∈ {10, 32, 64} (or custom)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.paths import ensure_run_on_path  # noqa: E402

ensure_run_on_path()

from code import TRAIN_DIR, create_federated_dataset, federated_output_path  # noqa: E402
from code.dataset import count_images_per_class, visualize_data_distribution  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Create Non-IID federated partitions")
    p.add_argument("--clients", type=int, nargs="+", default=[10, 32, 64])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-images", type=int, default=5)
    p.add_argument("--plot", action="store_true", help="Save distribution plots")
    args = p.parse_args()

    if not Path(TRAIN_DIR).exists():
        raise FileNotFoundError(
            f"Train folder not found: {TRAIN_DIR}\n"
            "Place Dataset_Maize_DD at the project root first."
        )

    for k in args.clients:
        out = federated_output_path(k)
        print(f"Creating {k}-client partition → {out}")
        create_federated_dataset(
            dataset_dir=TRAIN_DIR,
            num_clients=k,
            output_dir=out,
            min_images_per_class=args.min_images,
            seed=args.seed,
        )
        if args.plot:
            counts = count_images_per_class(out)
            visualize_data_distribution(counts, save_path=f"distribution_{k}clients.png")


if __name__ == "__main__":
    main()
