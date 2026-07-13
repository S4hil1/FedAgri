"""
Federated aggregation algorithms for FedAgri.

Baselines: FedAvg, weighted FedAvg, FedProx, FedSGD, Krum, Median,
           Trimmed Mean, FedAdam, FedNova, DP-FedAvg, FedAgri alignment.

Proposed:  FedCADR (fed_car / fed_car_plus) — class-wise residual classifier
           aggregation with magnitude normalization and optional cosine filtering.

FedDCA:    Divergence-/cosine-aware partial update (experimental).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helpers
# =============================================================================

def _clone_state(state: dict) -> dict:
    return {k: v.clone().detach() for k, v in state.items()}


def _get_state(model_or_state) -> dict:
    if isinstance(model_or_state, nn.Module):
        return model_or_state.state_dict()
    return model_or_state


def _classifier_weight_keys(state: dict) -> list[str]:
    """Detect shared classifier weight tensors for FedCADR."""
    keys = []
    for k in state:
        if not torch.is_floating_point(state[k]):
            continue
        if any(tok in k for tok in (
            "classifier.weight",
            "personal_layer.weight",
            "fc.weight",
            "fc2.weight",
        )):
            if state[k].ndim == 2:  # [num_classes, dim]
                keys.append(k)
    return keys


def _apply_to_model(global_model: nn.Module, new_state: dict) -> nn.Module:
    global_model.load_state_dict(new_state)
    return global_model


# =============================================================================
# Standard aggregators (return state_dict)
# =============================================================================

def federated_avg(client_models: list[dict]) -> dict:
    """FedAvg — unweighted mean of client state_dicts."""
    avg = {k: v.clone().float() for k, v in client_models[0].items()}
    for key in avg:
        if torch.is_floating_point(avg[key]):
            avg[key] = torch.stack([m[key].float() for m in client_models]).mean(0)
        else:
            avg[key] = client_models[0][key].clone()
    return avg


def weighted_federated_avg(client_models: list[dict], client_weights: list[float]) -> dict:
    """Sample-size-weighted FedAvg."""
    total = float(sum(client_weights))
    avg = {k: torch.zeros_like(v).float() if torch.is_floating_point(v) else v.clone()
           for k, v in client_models[0].items()}
    for i, model in enumerate(client_models):
        w = client_weights[i] / total
        for key in avg:
            if torch.is_floating_point(avg[key]):
                avg[key] = avg[key] + w * model[key].float()
    return avg


def federated_sgd(client_gradients: list[dict]) -> dict:
    """Mean of client gradient dictionaries."""
    avg = {k: v.clone().float() for k, v in client_gradients[0].items()}
    for key in avg:
        avg[key] = torch.stack([g[key].float() for g in client_gradients]).mean(0)
    return avg


def fedprox(client_models: list[dict], global_model: dict, mu: float = 0.1) -> dict:
    """FedProx-style proximal mix toward the previous global model."""
    g = _get_state(global_model)
    avg = federated_avg(client_models)
    for key in avg:
        if torch.is_floating_point(avg[key]):
            avg[key] = avg[key] + mu * (g[key].float() - avg[key])
    return avg


def differential_privacy_aggregation(
    client_models: list[dict],
    noise_scale: float = 0.1,
) -> dict:
    """FedAvg + isotropic Gaussian noise."""
    avg = federated_avg(client_models)
    for key in avg:
        if torch.is_floating_point(avg[key]):
            avg[key] = avg[key] + torch.randn_like(avg[key]) * noise_scale
    return avg


def krum(client_models: list[dict]) -> dict:
    """Krum — select the client closest (L1) to all others."""
    n = len(client_models)
    scores = []
    for i in range(n):
        dist = 0.0
        for j in range(n):
            if i == j:
                continue
            dist += sum(
                torch.abs(client_models[i][k].float() - client_models[j][k].float()).sum().item()
                for k in client_models[i]
                if torch.is_floating_point(client_models[i][k])
            )
        scores.append(dist)
    return _clone_state(client_models[int(np.argmin(scores))])


def median_aggregation(client_models: list[dict]) -> dict:
    """Coordinate-wise median."""
    result = {}
    for key in client_models[0]:
        if torch.is_floating_point(client_models[0][key]):
            stacked = torch.stack([m[key].float() for m in client_models])
            result[key] = torch.median(stacked, dim=0).values
        else:
            result[key] = client_models[0][key].clone()
    return result


def trimmed_mean(
    client_models: list[dict],
    trim_ratio: float = 0.1,
) -> dict:
    """
    Coordinate-wise trimmed mean.

    Drops the lowest/highest `trim_ratio` fraction of client values per coordinate.
    """
    n = len(client_models)
    trim = int(trim_ratio * n)
    if trim * 2 >= n:
        trim = max(0, (n - 1) // 2)

    result = {}
    for key in client_models[0]:
        if not torch.is_floating_point(client_models[0][key]):
            result[key] = client_models[0][key].clone()
            continue
        stacked = torch.stack([m[key].float() for m in client_models])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if trim > 0:
            result[key] = sorted_vals[trim:-trim].mean(dim=0)
        else:
            result[key] = sorted_vals.mean(dim=0)
    return result


def fednova(
    client_models: list[dict],
    client_weights: list[float],
    global_model: dict,
) -> dict:
    """FedNova — normalize client deltas before averaging."""
    g = _get_state(global_model)
    avg = federated_avg(client_models)
    for key in avg:
        if not torch.is_floating_point(avg[key]):
            continue
        normed = []
        for i, model in enumerate(client_models):
            delta = model[key].float() - g[key].float()
            norm = torch.norm(delta)
            if norm > 0:
                normed.append(client_weights[i] * delta / norm)
        if normed:
            avg[key] = g[key].float() + sum(normed) / len(normed)
    return avg


def federated_momentum(
    client_models: list[dict],
    client_weights: list[float],
    global_model: dict,
    momentum: float = 0.9,
    velocity: Optional[dict] = None,
) -> tuple[dict, dict]:
    """
    Momentum aggregation on weighted FedAvg delta.

    Returns (new_state, updated_velocity).
    """
    g = _get_state(global_model)
    avg = weighted_federated_avg(client_models, client_weights)
    if velocity is None:
        velocity = {k: torch.zeros_like(v).float() for k, v in avg.items()
                    if torch.is_floating_point(v)}
    new_state = {}
    for key in avg:
        if not torch.is_floating_point(avg[key]):
            new_state[key] = avg[key]
            continue
        delta = avg[key] - g[key].float()
        velocity[key] = momentum * velocity[key] + (1 - momentum) * delta
        new_state[key] = g[key].float() + velocity[key]
    return new_state, velocity


def divergence_aware_agg(
    client_models: list[dict],
    global_model: dict,
    gamma: float = 1.0,
) -> dict:
    """Weight clients by exp(-γ · ||θ_k − θ_g||²)."""
    g = _get_state(global_model)
    weights = []
    for state in client_models:
        dist = sum(
            torch.norm(state[k].float() - g[k].float()) ** 2
            for k in g if torch.is_floating_point(g[k])
        )
        weights.append(torch.exp(-gamma * dist))
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    return weighted_federated_avg(client_models, weights.tolist())


# =============================================================================
# FedAdam (stateful server optimizer)
# =============================================================================

def fedadam(
    global_model: nn.Module,
    client_states: list[dict],
    client_sizes: list[float],
    opt_state: Optional[dict] = None,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> dict:
    """
    FedAdam server aggregation.

    Updates `global_model` in place and returns opt_state for the next round.
    """
    if opt_state is None:
        opt_state = {"m": {}, "v": {}, "t": 0}
    opt_state["t"] += 1
    t = opt_state["t"]

    global_state = global_model.state_dict()
    total = float(sum(client_sizes))
    new_state = {}

    with torch.no_grad():
        for k, param in global_state.items():
            if not torch.is_floating_point(param):
                new_state[k] = param
                continue
            avg = sum(
                client_states[i][k].float() * (client_sizes[i] / total)
                for i in range(len(client_states))
            )
            delta = param.float() - avg
            m = opt_state["m"].get(k, torch.zeros_like(delta))
            v = opt_state["v"].get(k, torch.zeros_like(delta))
            m = beta1 * m + (1 - beta1) * delta
            v = beta2 * v + (1 - beta2) * (delta ** 2)
            opt_state["m"][k] = m
            opt_state["v"][k] = v
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            new_state[k] = param.float() - lr * m_hat / (torch.sqrt(v_hat) + eps)

    global_model.load_state_dict(new_state)
    return opt_state


def fedadagrad(
    global_model: nn.Module,
    client_updates: list[dict],
    g_prev: Optional[dict] = None,
    lr: float = 0.01,
    eps: float = 1e-8,
) -> dict:
    """FedAdagrad on mean client deltas."""
    if g_prev is None:
        g_prev = {}
    global_state = global_model.state_dict()
    mean_update = federated_avg(client_updates)
    new_state = {}
    for key, param in global_state.items():
        if not torch.is_floating_point(param):
            new_state[key] = param
            continue
        g = mean_update[key].float()
        acc = g_prev.get(key, torch.zeros_like(g)) + g ** 2
        g_prev[key] = acc
        new_state[key] = param.float() - lr * g / (torch.sqrt(acc) + eps)
    global_model.load_state_dict(new_state)
    return g_prev


# =============================================================================
# FedAgri alignment aggregator (experimental notebook version)
# =============================================================================

def global_alignment_loss(global_model: nn.Module, client_state_dicts: list[dict]) -> torch.Tensor:
    """Sum of L2 distances between global params and each client."""
    loss = torch.tensor(0.0, device=next(global_model.parameters()).device)
    for client_dict in client_state_dicts:
        for name, param in global_model.named_parameters():
            if name in client_dict:
                loss = loss + torch.norm(param - client_dict[name].to(param.device))
    return loss


def optimize_global_model(
    global_model: nn.Module,
    local_models: list[dict],
    lr: float = 0.001,
    steps: int = 10,
) -> nn.Module:
    """SGD steps minimizing alignment loss after FedAvg."""
    optimizer = torch.optim.SGD(global_model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = global_alignment_loss(global_model, local_models)
        loss.backward()
        optimizer.step()
    return global_model


def FedAgri(
    client_models: list[dict],
    global_model: Optional[nn.Module] = None,
    lr: float = 0.001,
    steps: int = 10,
) -> dict:
    """
    Experimental FedAgri notebook aggregator:
    FedAvg → optional alignment optimization on a live global_model.
    """
    avg = federated_avg(client_models)
    if global_model is None:
        return avg
    global_model.load_state_dict(avg)
    optimize_global_model(global_model, client_models, lr=lr, steps=steps)
    return global_model.state_dict()


# =============================================================================
# FedCADR — proposed class-aware residual aggregation
# =============================================================================

def fed_car(
    global_model: nn.Module,
    client_states: list[dict],
    client_sizes: list[float],
    beta: float = 0.3,
    local_epochs: int = 3,
    epsilon: float = 1e-6,
) -> nn.Module:
    """
    FedCADR (basic): feature FedAvg + class-wise residual classifier updates.

    For each classifier class row c:
      1. Skip clients with near-zero class weights (absent class)
      2. Magnitude-normalize residuals
      3. Weight by client data size
      4. Scale by β / E and apply to global classifier row

    Backbone (`features.*`) and remaining layers use sample-weighted FedAvg.
    """
    global_state = global_model.state_dict()
    new_state = {}
    total = float(sum(client_sizes))
    clf_keys = set(_classifier_weight_keys(global_state))

    for k, gw in global_state.items():
        if not torch.is_floating_point(gw):
            new_state[k] = gw
            continue

        # Backbone → weighted FedAvg
        if "features" in k or "backbone" in k:
            new_state[k] = sum(
                (client_sizes[i] / total) * client_states[i][k].float()
                for i in range(len(client_states))
            )
            continue

        # Classifier weight → class-wise residual aggregation
        if k in clf_keys:
            num_classes = gw.shape[0]
            updated = torch.zeros_like(gw).float()
            for c in range(num_classes):
                class_updates, weights = [], []
                for i in range(len(client_states)):
                    cw = client_states[i][k][c].float()
                    delta = cw - gw[c].float()
                    if cw.norm() < epsilon:
                        continue
                    class_updates.append(delta)
                    weights.append(client_sizes[i])

                if not class_updates:
                    updated[c] = gw[c]
                    continue

                updates = torch.stack(class_updates)
                flat = updates.view(len(updates), -1)
                norms = flat.norm(dim=1, keepdim=True) + 1e-8
                updates = (flat / norms).view_as(updates)

                w = torch.tensor(weights, dtype=torch.float32, device=updates.device)
                w = w / w.sum()
                beta_eff = beta / max(local_epochs, 1)
                update = sum(w[i] * updates[i] for i in range(len(updates)))
                updated[c] = gw[c].float() + beta_eff * update

            new_state[k] = updated
            continue

        # Fallback FedAvg
        new_state[k] = sum(
            (client_sizes[i] / total) * client_states[i][k].float()
            for i in range(len(client_states))
        )

    return _apply_to_model(global_model, new_state)


def fed_car_plus(
    global_model: nn.Module,
    client_states: list[dict],
    client_sizes: list[float],
    beta: float = 0.5,
    epsilon: float = 1e-6,
) -> nn.Module:
    """
    Full FedCADR (paper Algorithm 1):

      feature layers  → sample-weighted FedAvg
      classifier rows → residual updates with:
        (i)   near-zero skip (absent-class clients)
        (ii)  cosine-similarity filtering against class consensus
        (iii) magnitude-normalized adaptive weighting
        (iv)  scale by β
    """
    global_state = global_model.state_dict()
    new_state = {}
    total = float(sum(client_sizes))
    clf_keys = set(_classifier_weight_keys(global_state))

    for k, gw in global_state.items():
        if not torch.is_floating_point(gw):
            new_state[k] = gw
            continue

        if "features" in k or "backbone" in k:
            new_state[k] = sum(
                (client_sizes[i] / total) * client_states[i][k].float()
                for i in range(len(client_states))
            )
            continue

        if k in clf_keys:
            num_classes = gw.shape[0]
            updated = torch.zeros_like(gw).float()

            for c in range(num_classes):
                updates, weights = [], []
                for i in range(len(client_states)):
                    cw = client_states[i][k][c].float()
                    delta = cw - gw[c].float()
                    if delta.norm() < epsilon:
                        continue
                    updates.append(delta)
                    weights.append(client_sizes[i])

                if not updates:
                    updated[c] = gw[c]
                    continue

                updates_t = torch.stack(updates)

                # Cosine similarity filter vs consensus
                ref = updates_t.mean(dim=0, keepdim=True)
                sims = F.cosine_similarity(
                    updates_t.view(len(updates_t), -1),
                    ref.view(1, -1),
                    dim=1,
                )
                mask = sims > 0
                updates_t = updates_t[mask]
                if updates_t.shape[0] == 0:
                    updated[c] = gw[c]
                    continue

                w = torch.tensor(weights, dtype=torch.float32, device=updates_t.device)[mask]
                w = w / w.sum()

                # Confidence scaling by residual norms
                norms = updates_t.view(len(updates_t), -1).norm(dim=1)
                norms = norms / (norms.mean() + 1e-8)
                w = w * norms
                w = w / w.sum()

                update = sum(w[i] * updates_t[i] for i in range(len(updates_t)))
                updated[c] = gw[c].float() + beta * update

            new_state[k] = updated
            continue

        new_state[k] = sum(
            (client_sizes[i] / total) * client_states[i][k].float()
            for i in range(len(client_states))
        )

    return _apply_to_model(global_model, new_state)


# Paper-facing aliases
FedCADR = fed_car_plus
fedcadr = fed_car_plus


def fed_dca(
    global_model: nn.Module,
    client_states: list[dict],
    client_sizes: list[float],
    alpha: float = 0.8,
) -> nn.Module:
    """
    Experimental FedDCA: cosine-similarity-weighted classifier rows +
    partial FedAvg elsewhere.
    """
    global_state = global_model.state_dict()
    new_state = {}
    total = float(sum(client_sizes))
    n = len(client_states)

    for k in global_state:
        gw = global_state[k]
        if not torch.is_floating_point(gw):
            new_state[k] = gw
            continue

        # Special handling for KAN classifier / linear classifier weights
        if any(tok in k for tok in ("kan2.linear.weight", "classifier.weight", "personal_layer.weight", "fc.weight")) and gw.ndim == 2:
            agg_w = torch.zeros_like(gw).float()
            weight_sum = torch.zeros(gw.shape[0], device=gw.device)
            for i in range(n):
                client_w = client_states[i][k].float()
                sim = F.cosine_similarity(client_w, gw.float(), dim=1).clamp(min=0)
                w = sim * (client_sizes[i] / total)
                agg_w = agg_w + w.unsqueeze(1) * client_w
                weight_sum = weight_sum + w
            weight_sum = weight_sum.clamp(min=1e-6)
            agg_w = agg_w / weight_sum.unsqueeze(1)
            new_state[k] = alpha * agg_w + (1 - alpha) * gw.float()
            continue

        if any(tok in k for tok in ("kan2.linear.bias", "classifier.bias", "personal_layer.bias", "fc.bias")):
            new_state[k] = sum(
                (client_sizes[i] / total) * client_states[i][k].float() for i in range(n)
            )
            continue

        fedavg_w = sum(
            (client_sizes[i] / total) * client_states[i][k].float() for i in range(n)
        )
        new_state[k] = alpha * fedavg_w + (1 - alpha) * gw.float()

    return _apply_to_model(global_model, new_state)


# =============================================================================
# Model-mutating wrappers (FED_MDD notebook style: modify global_model in place)
# =============================================================================

def fedavg_inplace(global_model: nn.Module, client_states: list[dict], client_sizes: list[float]) -> nn.Module:
    """Weighted FedAvg that updates global_model in place."""
    total = float(sum(client_sizes))
    new_state = {
        k: sum((client_sizes[i] / total) * client_states[i][k] for i in range(len(client_states)))
        for k in global_model.state_dict()
    }
    return _apply_to_model(global_model, new_state)


def median_inplace(global_model: nn.Module, client_states: list[dict]) -> nn.Module:
    new_state = median_aggregation(client_states)
    return _apply_to_model(global_model, new_state)


def trimmed_mean_inplace(
    global_model: nn.Module,
    client_states: list[dict],
    trim_ratio: float = 0.2,
) -> nn.Module:
    return _apply_to_model(global_model, trimmed_mean(client_states, trim_ratio))


# =============================================================================
# Dispatcher
# =============================================================================

# Aggregators that only need client state_dicts
UNWEIGHTED = {federated_avg, federated_sgd, krum, median_aggregation, trimmed_mean}


def aggregate(
    client_models: list[dict],
    method: Callable,
    client_weights: Optional[list[float]] = None,
    global_model=None,
    **kwargs,
) -> dict:
    """
    Dispatch to the correct aggregation function.

    Returns a state_dict. For methods that mutate an nn.Module (FedCADR, FedAdam),
    pass global_model as nn.Module; the returned value is still a state_dict.
    """
    name = getattr(method, "__name__", str(method))

    # FedCADR family — mutate model, return state_dict
    if method in (fed_car, fed_car_plus, FedCADR, fedcadr, fed_dca):
        if global_model is None or not isinstance(global_model, nn.Module):
            raise ValueError(f"{name} requires global_model as nn.Module")
        if client_weights is None:
            raise ValueError(f"{name} requires client_weights (client sizes)")
        if method is fed_car:
            fed_car(
                global_model, client_models, client_weights,
                beta=kwargs.get("beta", 0.3),
                local_epochs=kwargs.get("local_epochs", 3),
                epsilon=kwargs.get("epsilon", 1e-6),
            )
        elif method is fed_dca:
            fed_dca(
                global_model, client_models, client_weights,
                alpha=kwargs.get("alpha", 0.8),
            )
        else:  # fed_car_plus / FedCADR
            fed_car_plus(
                global_model, client_models, client_weights,
                beta=kwargs.get("beta", 0.5),
                epsilon=kwargs.get("epsilon", 1e-6),
            )
        return global_model.state_dict()

    if method is FedAgri:
        return FedAgri(client_models, global_model if isinstance(global_model, nn.Module) else None, **kwargs)

    if method in (federated_avg, federated_sgd, krum, median_aggregation):
        return method(client_models)

    if method is trimmed_mean:
        return trimmed_mean(client_models, kwargs.get("trim_ratio", 0.1))

    if method is differential_privacy_aggregation:
        return differential_privacy_aggregation(client_models, kwargs.get("noise_scale", 0.1))

    if method is fedprox:
        if global_model is None:
            raise ValueError("fedprox requires global_model")
        return fedprox(client_models, _get_state(global_model), kwargs.get("mu", 0.1))

    if method is fednova:
        if global_model is None or client_weights is None:
            raise ValueError("fednova requires global_model and client_weights")
        return fednova(client_models, client_weights, _get_state(global_model))

    if method is weighted_federated_avg:
        if client_weights is None:
            raise ValueError("weighted_federated_avg requires client_weights")
        return weighted_federated_avg(client_models, client_weights)

    if method is divergence_aware_agg:
        if global_model is None:
            raise ValueError("divergence_aware_agg requires global_model")
        return divergence_aware_agg(client_models, _get_state(global_model), kwargs.get("gamma", 1.0))

    # Fallback: try common signatures
    if client_weights is not None:
        try:
            return method(client_models, client_weights)
        except TypeError:
            pass
    return method(client_models)


AGGREGATOR_REGISTRY = {
    "fedavg": federated_avg,
    "weighted_fedavg": weighted_federated_avg,
    "fedprox": fedprox,
    "fedsgd": federated_sgd,
    "krum": krum,
    "median": median_aggregation,
    "trimmed_mean": trimmed_mean,
    "fednova": fednova,
    "dp": differential_privacy_aggregation,
    "fedagri": FedAgri,
    "fedcadr": fed_car_plus,
    "fed_car": fed_car,
    "fed_car_plus": fed_car_plus,
    "fed_dca": fed_dca,
    "divergence_aware": divergence_aware_agg,
}


def get_aggregator(name: str) -> Callable:
    key = name.lower().replace("-", "_")
    if key not in AGGREGATOR_REGISTRY:
        raise ValueError(f"Unknown aggregator '{name}'. Choose from: {sorted(AGGREGATOR_REGISTRY)}")
    return AGGREGATOR_REGISTRY[key]
