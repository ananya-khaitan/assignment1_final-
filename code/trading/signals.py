"""Causally aligned trading signals and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # Support both `python code/...py` and package-based test imports.
    from ..utils.causal_pipeline import max_drawdown
except ImportError:  # pragma: no cover - direct script execution
    from utils.causal_pipeline import max_drawdown


@dataclass(frozen=True)
class TradingResult:
    equity_curve: np.ndarray
    returns: np.ndarray
    signal: np.ndarray
    predicted_returns: np.ndarray
    realized_returns: np.ndarray
    sharpe: float
    sortino: float
    calmar: float
    annualized_return: float
    max_drawdown: float
    turnover: float
    hit_rate: float


def forecast_returns(origin_price: Sequence[float], forecast_price: Sequence[float]) -> np.ndarray:
    origin = np.asarray(origin_price, dtype=float)
    forecast = np.asarray(forecast_price, dtype=float)
    if len(origin) != len(forecast):
        raise ValueError("Origin and forecast prices must have equal length")
    if np.any(origin <= 0):
        raise ValueError("Origin prices must be positive")
    return (forecast - origin) / origin


def uncertainty_adjusted_signal(
    predicted_return: Sequence[float],
    interval_width: Sequence[float],
    origin_price: Sequence[float],
    threshold: float | Sequence[float],
) -> np.ndarray:
    """Signal available at the forecast origin, scaled by return uncertainty."""
    predicted = np.asarray(predicted_return, dtype=float)
    width = np.asarray(interval_width, dtype=float)
    origin = np.asarray(origin_price, dtype=float)
    threshold_values = np.asarray(threshold, dtype=float)
    if threshold_values.ndim == 0:
        threshold_values = np.repeat(float(threshold_values), len(predicted))
    if not (len(predicted) == len(width) == len(origin) == len(threshold_values)):
        raise ValueError("Signal inputs must have equal length")
    uncertainty_return = width / np.where(origin > 0, origin, np.nan)
    score = predicted / uncertainty_return
    score = np.where(np.isfinite(score), score, 0.0)
    return np.where(score > threshold_values, 1, np.where(score < -threshold_values, -1, 0)).astype(int)


def directional_signal(predicted_return: Sequence[float]) -> np.ndarray:
    predicted = np.asarray(predicted_return, dtype=float)
    return np.where(predicted > 0, 1, np.where(predicted < 0, -1, 0)).astype(int)


def evaluate_trading(
    origin_price: Sequence[float],
    actual_price: Sequence[float],
    predicted_price: Sequence[float],
    interval_width: Sequence[float],
    threshold: float | Sequence[float] = 0.25,
    cost: float = 0.0005,
    use_uncertainty_filter: bool = True,
    annualization: int = 252,
) -> TradingResult:
    """Trade the t-1 forecast over the realized t-1→t return.

    Each row must contain the price at the forecast origin, the forecast made
    then, and the subsequently observed target price.  There is no use of a
    realized target in signal construction.
    """
    origin = np.asarray(origin_price, dtype=float)
    actual = np.asarray(actual_price, dtype=float)
    predicted = np.asarray(predicted_price, dtype=float)
    width = np.asarray(interval_width, dtype=float)
    if not (len(origin) == len(actual) == len(predicted) == len(width)):
        raise ValueError("Trading inputs must have equal length")
    valid = np.isfinite(origin) & np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(width) & (origin > 0)
    origin, actual, predicted, width = origin[valid], actual[valid], predicted[valid], width[valid]
    predicted_returns = forecast_returns(origin, predicted)
    realized_returns = forecast_returns(origin, actual)
    signal = (
        uncertainty_adjusted_signal(predicted_returns, width, origin, threshold)
        if use_uncertainty_filter
        else directional_signal(predicted_returns)
    )
    previous_signal = np.concatenate([[0], signal[:-1]]) if len(signal) else np.array([], dtype=int)
    turnover = float(np.mean(np.abs(signal - previous_signal))) if len(signal) else 0.0
    transaction_cost = np.abs(signal - previous_signal) * cost
    strategy_returns = signal * realized_returns - transaction_cost
    equity = np.cumprod(1.0 + strategy_returns)
    sharpe = float(np.mean(strategy_returns) / (np.std(strategy_returns, ddof=1) + 1e-12) * np.sqrt(annualization)) if len(strategy_returns) > 1 else float("nan")
    downside = np.minimum(strategy_returns, 0.0)
    sortino = float(np.mean(strategy_returns) / (np.sqrt(np.mean(downside**2)) + 1e-12) * np.sqrt(annualization)) if len(strategy_returns) else float("nan")
    annualized_return = float(equity[-1] ** (annualization / len(equity)) - 1.0) if len(equity) else float("nan")
    drawdown = max_drawdown(equity)
    calmar = float(annualized_return / (abs(drawdown) / 100.0)) if np.isfinite(drawdown) and drawdown < 0 else float("nan")
    hit_rate = float(np.mean((signal * realized_returns) > 0) * 100.0) if len(signal) else float("nan")
    return TradingResult(
        equity,
        strategy_returns,
        signal,
        predicted_returns,
        realized_returns,
        sharpe,
        sortino,
        calmar,
        annualized_return,
        drawdown,
        turnover,
        hit_rate,
    )
