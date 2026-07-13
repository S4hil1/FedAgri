"""
Ablation experiments — 32 clients (paper ablation setting).

1) FedCAMNet component ablation (SE / KAN on-off) with FedCADR
2) FedCADR β sweep with full FedCAMNet
3) Local-epoch sweep {2, 5, 10}

Run from ``new code/``::

    python run/experiments/run_ablation_client32.py
    python run/experiments/run_ablation_client32.py --only beta
    python run/experiments/run_ablation_client32.py --only components --rounds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import build_config, results_dir, run_single, save_json  # noqa: E402
from code.models import FederatedMobileNetKAN_Ablation  # noqa: E402


NUM_CLIENTS = 32


def _ablation_factory(use_se: bool, use_kan: bool):
    def factory(num_classes: int):
        return FederatedMobileNetKAN_Ablation(
            num_classes=num_classes,
            use_se=use_se,
            use_kan=use_kan,
            personalization=False,
        )

    factory.__name__ = f"fedcamnet_se{int(use_se)}_kan{int(use_kan)}"
    return factory


def run_component_ablation(args, device, out):
    from code import VAL_DIR, federated_output_path, FedCADR
    from code.federated_loop import run_federated_learning_threaded

    combos = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    results = []
    for use_se, use_kan in combos:
        cfg = build_config(args.rounds, args.local_epochs, args.batch_size, args.lr, args.beta, args.seed)
        name = f"se{int(use_se)}_kan{int(use_kan)}"
        print(f"\n>>> Component ablation: SE={use_se} KAN={use_kan}")
        model, _, _, accs = run_federated_learning_threaded(
            base_client_folder=federated_output_path(NUM_CLIENTS),
            num_clients=NUM_CLIENTS,
            val_dir=VAL_DIR,
            cfg=cfg,
            aggregation_method=FedCADR,
            model_factory=_ablation_factory(use_se, use_kan),
            device=device,
        )
        row = {"use_se": use_se, "use_kan": use_kan, "final_accuracy": accs[-1], "global_accuracies": accs}
        results.append(row)
        torch.save(model.state_dict(), out / f"ablation_{name}.pth")
    path = save_json(out / "component_ablation.json", results)
    print(f"Saved {path}")


def run_beta_sweep(args, device, out):
    betas = [0.1, 0.3, 0.5, 0.85]
    results = []
    for beta in betas:
        print(f"\n>>> FedCADR β={beta}")
        cfg = build_config(args.rounds, args.local_epochs, args.batch_size, args.lr, beta, args.seed)
        payload = run_single(
            NUM_CLIENTS, "fedcamnet", "fedcadr", cfg, device, out, threaded=not args.sequential
        )
        results.append({"beta": beta, "final_accuracy": payload.get("final_accuracy")})
    path = save_json(out / "beta_sweep.json", results)
    print(f"Saved {path}")


def run_epoch_sweep(args, device, out):
    results = []
    for e in [2, 5, 10]:
        print(f"\n>>> Local epochs E={e}")
        cfg = build_config(args.rounds, e, args.batch_size, args.lr, args.beta, args.seed)
        payload = run_single(
            NUM_CLIENTS, "fedcamnet", "fedcadr", cfg, device, out, threaded=not args.sequential
        )
        results.append({"local_epochs": e, "final_accuracy": payload.get("final_accuracy")})
    path = save_json(out / "epoch_sweep.json", results)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description="FedAgri ablations — 32 clients")
    p.add_argument("--only", choices=["components", "beta", "epochs", "all"], default="all")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--sequential", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    out = results_dir("ablation", NUM_CLIENTS)

    if args.only in ("components", "all"):
        run_component_ablation(args, device, out)
    if args.only in ("beta", "all"):
        run_beta_sweep(args, device, out)
    if args.only in ("epochs", "all"):
        run_epoch_sweep(args, device, out)


if __name__ == "__main__":
    main()
