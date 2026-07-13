"""Model / aggregator presets for baseline, proposed, and experiment modes."""

from __future__ import annotations

BASELINE_MODELS = [
    "resnet18",
    "mobilenet",
    "efficientnet",
    "squeezenet",
]

BASELINE_AGGREGATORS = [
    "fedavg",
    "weighted_fedavg",
    "fedprox",
    "median",
    "trimmed_mean",
    "krum",
]

# Paper proposed setup
PROPOSED_MODEL = "fedcamnet"
PROPOSED_AGG = "fedcadr"

EXPERIMENT_MODELS = [
    "fedcamnet",
    "fedcamnet_ablation",
    "md2",
    "mininetpp",
    "context_aware",
    "mobilenet",
]

EXPERIMENT_AGGREGATORS = [
    "fedavg",
    "fedcadr",
    "fed_car",
    "fed_dca",
    "median",
    "trimmed_mean",
    "krum",
    "fedprox",
]
