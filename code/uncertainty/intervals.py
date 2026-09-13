"""Causal conformal prediction intervals."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def conformal_width_from_residuals(residuals: Sequence[float], alpha: float = 0.10) -> float:
    """Finite-sample split-conformal half-width from *prior* absolute errors."""
    errors = np.asarray(residuals, dtype=float)
    errors = np.abs(errors[np.isfinite(errors)])
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if len(errors) < 2:
        raise ValueError("At least two historical calibration residuals are required")
    # Conformal finite-sample correction: ceil((n + 1)(1-alpha)) / n.
    rank = min(len(errors), int(np.ceil((len(errors) + 1) * (1 - alpha))))
    return float(np.partition(errors, rank - 1)[rank - 1])


def conformal_interval(pred_cal: Sequence[float], actual_cal: Sequence[float], alpha: float = 0.10) -> float:
    """Compatibility wrapper returning a valid conformal half-width."""
    predicted = np.asarray(pred_cal, dtype=float)
    actual = np.asarray(actual_cal, dtype=float)
    if len(predicted) != len(actual):
        raise ValueError("Calibration predictions and actuals must have equal length")
    return conformal_width_from_residuals(actual - predicted, alpha=alpha)


def interval_metrics(actual: Sequence[float], forecast: Sequence[float], width: Sequence[float], alpha: float = 0.10) -> dict[str, float]:
    """Evaluate coverage, sharpness, calibration error, and Winkler score."""
    actual_values = np.asarray(actual, dtype=float)
    forecasts = np.asarray(forecast, dtype=float)
    widths = np.asarray(width, dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(forecasts) & np.isfinite(widths) & (widths >= 0)
    actual_values, forecasts, widths = actual_values[valid], forecasts[valid], widths[valid]
    if len(actual_values) == 0:
        return {"Coverage (%)": float("nan"), "Avg Width": float("nan"), "Sharpness": float("nan"), "Calibration Error": float("nan"), "Winkler": float("nan")}
    lower, upper = forecasts - widths, forecasts + widths
    inside = (actual_values >= lower) & (actual_values <= upper)
    coverage = float(np.mean(inside))
    full_width = upper - lower
    penalty = np.where(actual_values < lower, (2 / alpha) * (lower - actual_values), 0.0)
    penalty += np.where(actual_values > upper, (2 / alpha) * (actual_values - upper), 0.0)
    winkler = full_width + penalty
    return {
        "Coverage (%)": coverage * 100.0,
        "Avg Width": float(np.mean(full_width)),
        "Sharpness": float(np.mean(full_width)),
        "Calibration Error": abs(coverage - (1 - alpha)) * 100.0,
        "Winkler": float(np.mean(winkler)),
    }


def quantile_interval(pred: Sequence[float], lower_q: Sequence[float], upper_q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(lower_q, dtype=float), np.asarray(upper_q, dtype=float)


def bootstrap_interval(samples: np.ndarray, alpha: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    return np.quantile(samples, alpha / 2, axis=0), np.quantile(samples, 1 - alpha / 2, axis=0)
