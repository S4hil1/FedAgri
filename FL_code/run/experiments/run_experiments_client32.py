"""
Experimental / custom FL — 32 clients.

Custom models × experimental aggregators (paper ablation scale).

Run from ``new code/``::

    python run/experiments/run_experiments_client32.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import parse_common_args, run_experiment  # noqa: E402

NUM_CLIENTS = 32


def main():
    args = parse_common_args(f"Custom/experimental FL — {NUM_CLIENTS} clients")
    run_experiment(
        num_clients=NUM_CLIENTS,
        mode="experiment",
        models=args.models,
        aggregators=args.aggregators,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        seed=args.seed,
        device=args.device,
        threaded=not args.sequential,
        train_ratio=args.train_ratio,
        mu=args.mu,
        max_workers=args.max_workers,
        csv_prefix=args.csv_prefix,
    )


if __name__ == "__main__":
    main()
