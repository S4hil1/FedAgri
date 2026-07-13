"""Shared CLI argument parser for all ``run/`` entry scripts."""

from __future__ import annotations

import argparse


def parse_common_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--beta", type=float, default=0.3, help="FedCADR β")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Client train/test split (notebook used 0.9)",
    )
    p.add_argument("--mu", type=float, default=0.01, help="FedProx μ")
    p.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Parallel local-train threads",
    )
    p.add_argument(
        "--csv-prefix",
        type=str,
        default="",
        help="Prefix for CSV filename (e.g. Context_EXP_)",
    )
    p.add_argument("--device", type=str, default=None, help="cuda / cpu / cuda:0")
    p.add_argument("--sequential", action="store_true", help="Disable threading")
    p.add_argument("--models", nargs="*", default=None, help="Override model list")
    p.add_argument(
        "--aggregators",
        nargs="*",
        default=None,
        help="Override aggregator list",
    )
    return p.parse_args()
