"""Deterministic-run helpers and runtime dependency checks."""

from __future__ import annotations

import importlib.util
import os
import random
from typing import Mapping

import numpy as np


def set_global_seed(seed: int) -> dict[str, int | bool]:
    """Seed every supported random generator before model construction."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    report: dict[str, int | bool] = {"python": seed, "numpy": seed, "torch": False}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        report["torch"] = seed
    except ImportError:
        pass
    return report


def missing_packages(required: Mapping[str, str]) -> dict[str, str]:
    """Return package-to-purpose pairs for unavailable runtime requirements."""
    return {package: purpose for package, purpose in required.items() if importlib.util.find_spec(package) is None}


def require_packages(required: Mapping[str, str]) -> None:
    missing = missing_packages(required)
    if missing:
        details = ", ".join(f"{name} ({purpose})" for name, purpose in missing.items())
        raise RuntimeError(
            "Required runtime packages are unavailable: " + details + ". "
            "Install the pinned requirements before running the full experiment."
        )
