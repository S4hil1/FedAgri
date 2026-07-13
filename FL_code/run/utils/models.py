"""Model factory helpers for runnable experiments."""

from __future__ import annotations

from typing import Callable

from code import get_model
from code.models import FedCAMNet


def model_factory(name: str) -> Callable:
    """Return a ``(num_classes) -> nn.Module`` factory for *name*."""

    def factory(num_classes: int):
        if name == "fedcamnet":
            return FedCAMNet(num_classes=num_classes, personalization=False)
        if name == "fedcamnet_ablation":
            return get_model(
                "fedcamnet_ablation",
                num_classes,
                use_se=True,
                use_kan=True,
            )
        return get_model(name, num_classes)

    factory.__name__ = name
    return factory
