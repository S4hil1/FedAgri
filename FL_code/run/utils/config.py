"""Build ``FLConfig`` from CLI / experiment kwargs."""

from __future__ import annotations

from typing import Optional

from code import FLConfig


def build_config(
    rounds: int = 50,
    local_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.001,
    beta: float = 0.3,
    seed: Optional[int] = 42,
    train_ratio: float = 0.8,
    mu: float = 0.01,
    max_workers: int = 10,
) -> FLConfig:
    return FLConfig(
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer="adam",
        fedcadr_beta=beta,
        seed=seed,
        local_train_ratio=train_ratio,
        mu=mu,
        max_workers=max_workers,
    )
