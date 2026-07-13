"""
Evaluation and history helpers matching FED_MDD_V3 ``Run_FED``.

Provides:
  - evaluate / evaluate_client  (global loss, accuracy, preds/labels)
  - compare_to_average          (Frobenius / cosine / global variance)
  - histories_to_df             (CSV export shape used in the notebooks)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """
    Global (or client) evaluation used by original ``Run_FED``.

    Returns:
        (mean_loss, accuracy, preds, labels)
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * y.size(0)
            p = torch.argmax(out, dim=1)
            correct += (p == y).sum().item()
            total += y.size(0)

            preds.extend(p.cpu().tolist())
            labels.extend(y.cpu().tolist())

    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if total == 0:
        return 0.0, 0.0, preds, labels
    return total_loss / total, correct / total, preds, labels


def evaluate_client(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Accuracy-only client eval (original ``evaluate_client``)."""
    model.eval()
    model.to(device)
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = torch.argmax(model(x), dim=1)
            correct += (p == y).sum().item()
            total += y.size(0)

    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return correct / total if total else 0.0


def flatten_state_dict(state_dict: dict) -> torch.Tensor:
    """Concatenate floating-point parameters into one vector."""
    return torch.cat([
        v.reshape(-1).float()
        for v in state_dict.values()
        if torch.is_floating_point(v)
    ])


def compare_to_average(
    avg_wts: dict,
    client_wts: list[dict],
) -> tuple[list[dict[str, Any]], float]:
    """
    Compare each client state to the (aggregated) global state.

    Returns:
        (per-client distance rows, global_variance)
    """
    avg_vec = flatten_state_dict(avg_wts)
    client_vecs = [flatten_state_dict(w) for w in client_wts]

    results = []
    avg_norm = torch.norm(avg_vec, p=2).clamp_min(1e-12)

    for i, vec in enumerate(client_vecs):
        frob_dist = torch.norm(vec - avg_vec, p=2)
        cos_sim = F.cosine_similarity(vec.unsqueeze(0), avg_vec.unsqueeze(0)).item()
        rel_dev = (frob_dist / avg_norm).item()
        results.append({
            "client": i,
            "frobenius_distance": frob_dist.item(),
            "cosine_similarity": cos_sim,
            "relative_deviation": rel_dev,
        })

    variance = torch.stack([
        torch.norm(vec - avg_vec, p=2) ** 2 for vec in client_vecs
    ]).mean().item()

    return results, variance


def histories_to_df(histories: dict[str, list[dict]]):
    """Flatten ``{aggregator: [round_records...]}`` into a pandas DataFrame."""
    import pandas as pd

    rows = []
    for agg, records in histories.items():
        for r in records:
            row = dict(r)
            row["aggregation"] = agg
            rows.append(row)
    return pd.DataFrame(rows)
