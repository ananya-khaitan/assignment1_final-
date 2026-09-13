"""Forecast evaluation metrics and statistical tests."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t


def forecast_metrics(actual: Sequence[float], forecast: Sequence[float]) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    n = min(len(actual), len(forecast))
    actual = actual[:n]
    forecast = forecast[:n]
    valid = np.isfinite(actual) & np.isfinite(forecast)
    actual, forecast = actual[valid], forecast[valid]
    if len(actual) == 0:
        return {key: float("nan") for key in ("RMSE", "MAE", "MASE", "MAPE", "Directional Accuracy")}
    error = forecast - actual
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    mase_denom = np.mean(np.abs(np.diff(actual))) if len(actual) > 1 else np.nan
    mase = float(mae / (mase_denom + 1e-12)) if len(actual) > 1 else float("nan")
    mape = float(np.mean(np.abs(error / np.where(actual == 0, np.nan, actual))) * 100.0)
    da = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(forecast))) * 100.0) if len(actual) > 2 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MASE": mase, "MAPE": mape, "Directional Accuracy": da}


def dm_test(loss_diff: Sequence[float], horizon: int = 1) -> Tuple[float, float]:
    """Heteroskedasticity/autocorrelation-consistent Diebold--Mariano test.

    ``loss_diff`` is loss(model A) minus loss(model B); a negative statistic
    therefore favours model A.  The forecast horizon must match the actual
    forecast design.  This project uses only one-step forecasts.
    """
    d = np.asarray(loss_diff, dtype=float)
    d = d[np.isfinite(d)]
    t = len(d)
    if t < max(10, horizon + 2):
        return float("nan"), float("nan")
    if horizon < 1 or horizon >= t:
        raise ValueError("horizon must lie between 1 and len(loss_diff)-1")
    h = horizon
    mean_d = np.mean(d)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h + 1):
        cov = np.mean((d[:-lag] - mean_d) * (d[lag:] - mean_d))
        var_d += 2 * (1 - lag / (h + 1)) * cov
    var_d = max(var_d, 1e-12)
    dm = mean_d / np.sqrt(var_d / t)
    hln = np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    dm_hln = dm * hln
    p = 2 * (1 - student_t.cdf(abs(dm_hln), df=t - 1))
    return float(dm_hln), float(p)

