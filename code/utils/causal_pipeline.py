"""Core causal utilities for the in-place refactor.

The functions in this module are shared by the rebuilt forecasting,
uncertainty, and trading layers. They are designed around one rule:
for forecast date t, only information available at or before t may be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSnapshot:
    """All information available at a forecast date.

    `history` contains the series up to and including the forecast origin.
    `features` contains aligned exogenous predictors available at that date.
    """

    date: pd.Timestamp
    history: pd.Series
    features: pd.Series


@dataclass(frozen=True)
class WalkForwardSplit:
    """A one-step split whose forecast origin is ``train_end - 1``.

    ``train_end`` is exclusive, so ``series.iloc[:train_end]`` contains every
    observation available at the forecast origin and the scored target is
    ``series.iloc[train_end]``.  ``valid_end`` is retained for compatibility
    with older callers and is equal to ``train_end`` in the causal API.
    """

    train_end: int
    valid_end: int
    test_end: int


def clean_series(series: pd.Series) -> pd.Series:
    series = pd.Series(series).astype(float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def expanding_walk_forward(
    n_obs: int,
    min_train: int,
    valid_size: int = 0,
    test_size: int = 1,
    step_size: int = 1,
) -> List[WalkForwardSplit]:
    """Generate causally aligned expanding one-step splits.

    Validation is an *inner*, historical calibration activity; it must never
    create a gap between a forecast origin and the observation being scored.
    ``valid_size`` is consequently rejected when non-zero.  Use
    :func:`calibration_and_test_splits` for the standard separate calibration
    and final-test partition.
    """

    if valid_size:
        raise ValueError(
            "valid_size cannot be embedded between training and a one-step "
            "test target. Build validation forecasts inside the historical "
            "window instead."
        )
    if test_size != 1:
        raise ValueError("Only one-step walk-forward splits are supported.")
    if n_obs <= min_train + test_size:
        return []

    splits: List[WalkForwardSplit] = []
    train_end = min_train
    while train_end + test_size <= n_obs:
        test_end = train_end + test_size
        splits.append(WalkForwardSplit(train_end=train_end, valid_end=train_end, test_end=test_end))
        train_end += step_size
    return splits


def calibration_and_test_splits(
    n_obs: int,
    min_train: int,
    calibration_size: int,
    test_size: int,
) -> Tuple[List[WalkForwardSplit], List[WalkForwardSplit]]:
    """Create non-overlapping causal calibration and final-test splits.

    Every split predicts the immediately following observation.  Calibration
    forecasts are generated before the final test period and are available for
    model selection, conformal residuals, and threshold tuning.
    """

    if min_train < 1 or calibration_size < 1 or test_size < 1:
        raise ValueError("min_train, calibration_size, and test_size must be positive")
    test_start = n_obs - test_size
    calibration_start = test_start - calibration_size
    if calibration_start < min_train:
        raise ValueError(
            "Not enough observations for the requested training, calibration, "
            "and final-test windows."
        )

    def make(start: int, stop: int) -> List[WalkForwardSplit]:
        return [
            WalkForwardSplit(train_end=target, valid_end=target, test_end=target + 1)
            for target in range(start, stop)
        ]

    return make(calibration_start, test_start), make(test_start, n_obs)


def make_lag_frame(y: pd.Series, lags: int) -> pd.DataFrame:
    """Build a supervised frame from a single target series."""

    y = pd.Series(y).astype(float)
    data = {f"lag_{lag}": y.shift(lag) for lag in range(1, lags + 1)}
    frame = pd.DataFrame(data, index=y.index)
    frame["target"] = y
    return frame.dropna()


def make_exogenous_frame(y: pd.Series, x: pd.DataFrame, lags: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a t-1-information regression frame for forecasting ``y`` at t.

    The target's own lags and every exogenous predictor are shifted one
    observation.  This compatibility helper therefore cannot construct an
    accidental contemporaneous-feature design when reused by another script.
    """

    y = pd.Series(y).astype(float)
    x = pd.DataFrame(x).reindex(y.index).astype(float).shift(1)
    frame = pd.concat([y.rename("target"), x], axis=1)
    if lags > 0:
        for lag in range(1, lags + 1):
            frame[f"target_lag_{lag}"] = y.shift(lag)
    frame = frame.dropna()
    target = frame.pop("target")
    return frame, target


def directional_accuracy(actual: Sequence[float], forecast: Sequence[float]) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(actual) < 2 or len(forecast) < 2:
        return float("nan")
    actual_direction = np.sign(np.diff(actual))
    forecast_direction = np.sign(np.diff(forecast))
    return float(np.mean(actual_direction == forecast_direction) * 100.0)


def price_to_return(series: Sequence[float]) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    if len(series) < 2:
        return np.array([], dtype=float)
    return np.diff(series) / np.where(series[:-1] == 0, np.nan, series[:-1])


def max_drawdown(equity_curve: Sequence[float]) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / np.where(peak == 0, np.nan, peak)
    return float(np.nanmin(drawdown) * 100.0)
