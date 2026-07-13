"""
Experimental / custom FL — 10 clients.

Custom models (FedCAMNet, MD2, MiniNetPP, ContextAware, …)
× experimental aggregators (FedCADR, FedCAR, FedDCA, Krum, …).

Run from ``new code/``::

    python run/experiments/run_experiments_client10.py
    python run/experiments/run_experiments_client10.py --models fedcamnet md2 --aggregators fedcadr fedavg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import parse_common_args, run_experiment  # noqa: E402

NUM_CLIENTS = 10


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
