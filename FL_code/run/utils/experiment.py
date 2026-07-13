"""Experiment orchestration — ``run_single`` / ``run_experiment``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from code import VAL_DIR, federated_output_path
from code import FLConfig
from code.run_fed import run_fed

from .config import build_config
from .io import save_json
from .models import model_factory
from .paths import results_dir
from .presets import (
    BASELINE_AGGREGATORS,
    BASELINE_MODELS,
    EXPERIMENT_AGGREGATORS,
    EXPERIMENT_MODELS,
    PROPOSED_AGG,
    PROPOSED_MODEL,
)


def _require_partitions(num_clients: int) -> Path:
    base = federated_output_path(num_clients)
    if not base.exists():
        raise FileNotFoundError(
            f"Missing partitions at {base}.\n"
            f"Run: python run/create_partitions.py --clients {num_clients}"
        )
    return base


def run_single(
    num_clients: int,
    model_name: str,
    agg_name: str,
    cfg: FLConfig,
    device: torch.device,
    out_dir: Path,
    threaded: bool = True,
    csv_prefix: str = "",
) -> dict:
    """Run one (model, aggregator) pair via ``run_fed`` (single-aggregator call)."""
    base = _require_partitions(num_clients)

    print("=" * 60)
    print(f" clients={num_clients}  model={model_name}  agg={agg_name}")
    print(
        f" rounds={cfg.rounds}  E={cfg.local_epochs}  "
        f"batch={cfg.batch_size}  β={cfg.fedcadr_beta}"
    )
    print(f" data={base}")
    print("=" * 60)

    histories, csv_path = run_fed(
        model_name=model_name,
        aggregator_names=[agg_name],
        num_clients=num_clients,
        cfg=cfg,
        model_factory=model_factory(model_name),
        device=device,
        out_dir=out_dir,
        base_client_folder=base,
        val_dir=VAL_DIR,
        csv_prefix=csv_prefix,
        threaded=threaded,
    )

    records = histories.get(agg_name, [])
    final = records[-1] if records else {}
    payload = {
        "num_clients": num_clients,
        "model": model_name,
        "aggregator": agg_name,
        "rounds": cfg.rounds,
        "local_epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "fedcadr_beta": cfg.fedcadr_beta,
        "csv_path": str(csv_path),
        "final_accuracy": final.get("global_acc"),
        "final_loss": final.get("global_loss"),
        "final_macro_f1": final.get("macro_f1"),
        "global_accuracies": [r["global_acc"] for r in records],
    }
    print(f"Final accuracy: {(payload['final_accuracy'] or 0) * 100:.2f}%")
    return payload


def run_experiment(
    num_clients: int,
    mode: str = "baseline",
    models: Optional[list] = None,
    aggregators: Optional[list] = None,
    rounds: int = 50,
    local_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.001,
    beta: float = 0.3,
    seed: Optional[int] = 42,
    device: Optional[str] = None,
    threaded: bool = True,
    train_ratio: float = 0.8,
    mu: float = 0.01,
    max_workers: int = 10,
    csv_prefix: str = "",
) -> list[dict]:
    """
    Run a full experiment matrix for one client count.

    mode:
      - baseline   → standard models × baseline aggregators
      - proposed   → FedCAMNet × FedCADR only
      - experiment → custom models × experimental aggregators

    For each model, ``run_fed`` is called once with the full aggregator list
    (original notebook nesting), producing one CSV per model.
    """
    mode = mode.lower()
    if models is None:
        if mode == "baseline":
            models = BASELINE_MODELS
        elif mode == "proposed":
            models = [PROPOSED_MODEL]
        else:
            models = EXPERIMENT_MODELS

    if aggregators is None:
        if mode == "baseline":
            aggregators = BASELINE_AGGREGATORS
        elif mode == "proposed":
            aggregators = [PROPOSED_AGG]
        else:
            aggregators = EXPERIMENT_AGGREGATORS

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    cfg = build_config(
        rounds, local_epochs, batch_size, lr, beta, seed,
        train_ratio, mu, max_workers,
    )
    out = results_dir(mode, num_clients)
    base = _require_partitions(num_clients)

    summary = []
    for model_name in models:
        print("=" * 60)
        print(f" clients={num_clients}  model={model_name}  aggs={aggregators}")
        print(f" rounds={cfg.rounds}  E={cfg.local_epochs}  batch={cfg.batch_size}")
        print(f" data={base}")
        print("=" * 60)
        try:
            histories, csv_path = run_fed(
                model_name=model_name,
                aggregator_names=aggregators,
                num_clients=num_clients,
                cfg=cfg,
                model_factory=model_factory(model_name),
                device=device_t,
                out_dir=out,
                base_client_folder=base,
                val_dir=VAL_DIR,
                csv_prefix=csv_prefix,
                threaded=threaded,
            )
            for agg_name, records in histories.items():
                final = records[-1] if records else {}
                summary.append({
                    "num_clients": num_clients,
                    "model": model_name,
                    "aggregator": agg_name,
                    "rounds": cfg.rounds,
                    "local_epochs": cfg.local_epochs,
                    "csv_path": str(csv_path),
                    "final_accuracy": final.get("global_acc"),
                    "final_loss": final.get("global_loss"),
                    "final_macro_f1": final.get("macro_f1"),
                    "mean_client_acc": final.get("mean_client_acc"),
                    "global_accuracies": [r["global_acc"] for r in records],
                })
        except Exception as exc:
            print(f"[ERROR] {model_name}: {exc}")
            summary.append({
                "model": model_name,
                "aggregators": aggregators,
                "error": str(exc),
            })

    summary_path = save_json(out / "summary.json", summary)
    print(f"\nExperiment summary → {summary_path}")
    return summary
