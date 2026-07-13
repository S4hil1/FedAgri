"""Path helpers and ``sys.path`` setup for runnable scripts under ``run/``."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# run/utils/paths.py → run/ → new code/
_UTILS_DIR = Path(__file__).resolve().parent
_RUN_DIR = _UTILS_DIR.parent
_NEW_CODE = _RUN_DIR.parent


def ensure_run_on_path() -> Path:
    """
    Insert ``run/`` on ``sys.path`` so ``import code`` and ``import utils`` work.

    The package lives at ``run/code/``.
    """
    run = str(_RUN_DIR)
    if run not in sys.path:
        sys.path.insert(0, run)
    return _RUN_DIR


def ensure_new_code_on_path() -> Path:
    """Insert ``new code/`` on ``sys.path`` (legacy; prefer ``ensure_run_on_path``)."""
    root = str(_NEW_CODE)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _NEW_CODE


# Apply on import so ``from utils import …`` / ``import code`` work from entry scripts.
ensure_run_on_path()

NEW_CODE_ROOT = _NEW_CODE
RUN_ROOT = _RUN_DIR


def results_dir(mode: str, num_clients: int) -> Path:
    """Create and return ``new code/results/<mode>/client_<K>/<timestamp>/``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = NEW_CODE_ROOT / "results" / mode / f"client_{num_clients}" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out
