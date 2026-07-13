"""
Shared utilities for all runnable scripts under ``run/``.

Usage from an entry script::

    from utils import parse_common_args, run_experiment
    run_experiment(num_clients=10, mode="baseline")
"""

from __future__ import annotations

# Side-effect: put ``run/`` on sys.path so ``import code`` works
from .paths import NEW_CODE_ROOT, RUN_ROOT, ensure_new_code_on_path, ensure_run_on_path, results_dir
from .presets import (
    BASELINE_AGGREGATORS,
    BASELINE_MODELS,
    EXPERIMENT_AGGREGATORS,
    EXPERIMENT_MODELS,
    PROPOSED_AGG,
    PROPOSED_MODEL,
)
from .cli import parse_common_args
from .config import build_config
from .models import model_factory
from .io import save_json
from .experiment import run_experiment, run_single

__all__ = [
    "NEW_CODE_ROOT",
    "RUN_ROOT",
    "ensure_new_code_on_path",
    "ensure_run_on_path",
    "results_dir",
    "BASELINE_MODELS",
    "BASELINE_AGGREGATORS",
    "PROPOSED_MODEL",
    "PROPOSED_AGG",
    "EXPERIMENT_MODELS",
    "EXPERIMENT_AGGREGATORS",
    "parse_common_args",
    "build_config",
    "model_factory",
    "save_json",
    "run_single",
    "run_experiment",
]
