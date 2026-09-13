"""Publication-grade SPA and MCS wrappers backed by ``arch``."""

from __future__ import annotations

from typing import Any

import numpy as np


def spa_test(benchmark_loss: np.ndarray, model_losses: np.ndarray, reps: int = 1_000) -> dict[str, float]:
    """Run Hansen's SPA test using a stationary bootstrap.

    The benchmark is pre-specified by the caller (normally Random Walk),
    rather than selected ex post from the competing models.
    """
    try:
        from arch.bootstrap import SPA
    except ImportError as exc:  # pragma: no cover - requirements enforce this
        raise ImportError("arch is required for SPA testing") from exc
    benchmark = np.asarray(benchmark_loss, dtype=float)
    alternatives = np.asarray(model_losses, dtype=float)
    if alternatives.ndim != 2 or len(benchmark) != alternatives.shape[0]:
        raise ValueError("SPA losses must be T and T x M arrays on common dates")
    test = SPA(benchmark, alternatives, reps=reps, bootstrap="stationary", studentize=True)
    test.compute()
    return {str(key): float(value) for key, value in test.pvalues.items()}


def model_confidence_set(losses: np.ndarray, size: float = 0.05, reps: int = 1_000) -> tuple[list[int], dict[str, float]]:
    """Run Hansen--Lunde--Nason Model Confidence Set with block bootstrap.

    Exact duplicate loss columns are statistically indistinguishable and make
    ``arch``'s studentized range statistic undefined (its variance is zero).
    The MCS is therefore calculated once on the unique loss paths, then its
    inclusion decision and p-value are mapped back to every duplicate model.
    No model is dropped from the reported result.
    """
    try:
        from arch.bootstrap import MCS
    except ImportError as exc:  # pragma: no cover - requirements enforce this
        raise ImportError("arch is required for MCS testing") from exc
    matrix = np.asarray(losses, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("MCS requires T x M losses for at least two models")
    if not np.isfinite(matrix).all():
        raise ValueError("MCS losses must be finite")

    # Preserve first occurrence order so returned indices still refer to the
    # caller's full model matrix.  ``array_equal`` is deliberately exact: only
    # genuinely identical loss paths share a bootstrap inference result.
    representative_for: dict[int, int] = {}
    unique_indices: list[int] = []
    for candidate in range(matrix.shape[1]):
        representative = next(
            (index for index in unique_indices if np.array_equal(matrix[:, candidate], matrix[:, index])),
            None,
        )
        if representative is None:
            representative = candidate
            unique_indices.append(candidate)
        representative_for[candidate] = representative

    # A complete duplicate matrix contains no pairwise loss information.  It
    # remains a valid result: every labelled model belongs to the MCS with the
    # same non-rejection p-value.
    if len(unique_indices) == 1:
        return list(range(matrix.shape[1])), {str(index): 1.0 for index in range(matrix.shape[1])}

    test = MCS(matrix[:, unique_indices], size=size, reps=reps, method="R", bootstrap="stationary")
    test.compute()
    pvalues = test.pvalues
    # arch 7.x exposes MCS p-values as a one-column DataFrame indexed by
    # model position (unlike SPA's Series).  Normalize both forms to a small
    # JSON-safe mapping without discarding the model index.
    if hasattr(pvalues, "columns"):
        column = "Pvalue" if "Pvalue" in pvalues.columns else pvalues.columns[0]
        normalized_unique = {int(key): float(value) for key, value in pvalues[column].items()}
    else:
        normalized_unique = {int(key): float(value) for key, value in pvalues.items()}

    included_unique_positions = {int(index) for index in test.included}
    included_original = [
        original
        for original, representative in representative_for.items()
        if unique_indices.index(representative) in included_unique_positions
    ]
    normalized = {
        str(original): normalized_unique[unique_indices.index(representative)]
        for original, representative in representative_for.items()
    }
    return included_original, normalized
