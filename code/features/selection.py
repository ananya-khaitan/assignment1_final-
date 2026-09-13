"""Causal LASSO feature selection and ARX forecasting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SelectionResult:
    selected_columns: List[str]
    alpha: float
    scaler: StandardScaler


def build_feature_frame(target: pd.Series, exog: pd.DataFrame, lags: int) -> pd.DataFrame:
    """Build a t-1 information set for predicting the target at t."""
    target = pd.Series(target).astype(float)
    exog = pd.DataFrame(exog).reindex(target.index).astype(float)
    frame = pd.DataFrame(index=target.index)
    for lag in range(1, lags + 1):
        frame[f"target_lag_{lag}"] = target.shift(lag)
    # Daily exogenous series are known only after their own close, so use t-1.
    for col in exog.columns:
        frame[col] = exog[col].shift(1)
    frame["target"] = target
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _time_series_cv(n_rows: int, requested_folds: int) -> TimeSeriesSplit:
    # TimeSeriesSplit needs at least two folds and enough observations per fold.
    folds = min(requested_folds, max(2, n_rows // 20))
    if n_rows < 3 * folds:
        raise ValueError("Insufficient observations for time-series cross-validation")
    return TimeSeriesSplit(n_splits=folds)


def _select_penalty_temporally(x: pd.DataFrame, y: pd.Series, splitter: TimeSeriesSplit, model: str) -> tuple[float, float]:
    """Tune LASSO/Elastic Net with scalers fit separately in every CV fold."""
    preliminary_scaler = StandardScaler().fit(x)
    preliminary_x = preliminary_scaler.transform(x)
    alpha_max = float(np.max(np.abs(preliminary_x.T @ y.values)) / len(x))
    if not np.isfinite(alpha_max) or alpha_max <= 0:
        alpha_max = 1.0
    alphas = np.geomspace(alpha_max * 1e-4, alpha_max, num=30)
    l1_ratios = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0) if model == "elasticnet" else (1.0,)
    best: tuple[float, float, float] | None = None
    for l1_ratio in l1_ratios:
        for alpha in alphas:
            fold_errors = []
            for train_index, validation_index in splitter.split(x):
                fold_scaler = StandardScaler().fit(x.iloc[train_index])
                x_train = fold_scaler.transform(x.iloc[train_index])
                x_validation = fold_scaler.transform(x.iloc[validation_index])
                if l1_ratio == 1.0:
                    estimator = Lasso(alpha=float(alpha), max_iter=30_000)
                else:
                    estimator = ElasticNet(alpha=float(alpha), l1_ratio=l1_ratio, max_iter=30_000)
                estimator.fit(x_train, y.iloc[train_index])
                fold_errors.append(float(np.mean((estimator.predict(x_validation) - y.iloc[validation_index].values) ** 2)))
            candidate = (float(np.mean(fold_errors)), float(alpha), float(l1_ratio))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2]


def rolling_lasso_select(
    target: pd.Series,
    exog: pd.DataFrame,
    lags: int,
    train_end: int | None = None,
    valid_end: int | None = None,
    cv: int = 5,
    model: str = "lasso",
) -> SelectionResult:
    """Select predictors using only data available at the forecast origin.

    ``train_end`` and ``valid_end`` remain accepted for compatibility, but
    intentionally do not create an artificial validation gap.  Callers must
    pass only their historical slice; the selector then uses chronological
    cross-validation within that slice.
    """
    del train_end, valid_end
    frame = build_feature_frame(target, exog, lags)
    if len(frame) < 45:
        raise ValueError("Insufficient complete observations for LASSO selection")
    x = frame.drop(columns=["target"])
    y = frame["target"]
    splitter = _time_series_cv(len(x), cv)
    alpha, l1_ratio = _select_penalty_temporally(x, y, splitter, model)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    if model == "elasticnet" and l1_ratio < 1.0:
        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=30_000)
    else:
        estimator = Lasso(alpha=alpha, max_iter=30_000)
    estimator.fit(x_scaled, y.values)
    coefs = pd.Series(estimator.coef_, index=x.columns)
    selected = coefs[coefs.abs() > 1e-8].index.tolist()
    if not selected:
        # Retain own lags as an auditable deterministic fallback.
        selected = [col for col in x.columns if col.startswith("target_lag_")][: min(3, lags)]
    return SelectionResult(selected_columns=selected, alpha=alpha, scaler=scaler)


class SelectedARXForecaster:
    """A selected ARX model with an explicit fit/predict boundary.

    Selection, scaling, and coefficient estimation happen in :meth:`fit` on
    the history available at a scheduled re-estimation date.  Later calls to
    :meth:`predict` use only the current origin's observed lags and exogenous
    values; they do not refit or inspect future rows.
    """

    def __init__(self, lags: int = 5) -> None:
        self.lags = lags
        self.selection: SelectionResult | None = None
        self.columns: list[str] = []
        self.model: LinearRegression | None = None

    @property
    def selected_columns(self) -> list[str]:
        return list(self.columns)

    @property
    def alpha(self) -> float | None:
        return None if self.selection is None else self.selection.alpha

    def fit(self, target: pd.Series, exog: pd.DataFrame) -> "SelectedARXForecaster":
        self.selection = rolling_lasso_select(target, exog, lags=self.lags)
        frame = build_feature_frame(target, exog, self.lags)
        x = frame.drop(columns=["target"])
        self.columns = [column for column in self.selection.selected_columns if column in x.columns]
        if not self.columns:
            raise ValueError("Selected ARX has no usable predictor columns")
        self.model = LinearRegression().fit(x[self.columns].values, frame["target"].values)
        return self

    def predict(self, target: pd.Series, exog: pd.DataFrame, steps: int = 1) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SelectedARXForecaster.predict called before fit")
        current_target = pd.Series(target, dtype=float).copy()
        current_exog = pd.DataFrame(exog).reindex(current_target.index).astype(float).ffill()
        if len(current_target) < self.lags or current_exog.empty or current_exog.iloc[-1].isna().any():
            raise ValueError("Latest target/exogenous observations are unavailable for ARX forecast")
        predictions: list[float] = []
        for _ in range(steps):
            row: dict[str, float] = {}
            for lag in range(1, self.lags + 1):
                row[f"target_lag_{lag}"] = float(current_target.iloc[-lag])
            for column in current_exog.columns:
                row[column] = float(current_exog.iloc[-1][column])
            row_df = pd.DataFrame([row]).reindex(columns=self.columns)
            prediction = float(self.model.predict(row_df.values)[0])
            predictions.append(prediction)
            current_target = pd.concat([current_target, pd.Series([prediction])], ignore_index=True)
        return np.asarray(predictions, dtype=float)


def fit_selected_arx_forecast(
    target: pd.Series,
    exog: pd.DataFrame,
    selected_columns: List[str],
    lags: int,
    steps: int = 1,
) -> np.ndarray:
    """Fit selected ARX on historical data and recursively forecast forward.

    For horizons beyond one day, unavailable future exogenous data is held at
    its last *observed* value.  The production runner only permits one-step
    forecasts; keeping this behavior explicit avoids accidental future fills.
    """
    # Keep the historical function as a one-shot compatibility wrapper.  The
    # production engine uses ``SelectedARXForecaster`` directly so it can
    # retain a causal scheduled fit between daily origins.
    selector = SelectedARXForecaster(lags=lags)
    selector.selection = SelectionResult(list(selected_columns), alpha=float("nan"), scaler=StandardScaler())
    frame = build_feature_frame(target, exog, lags)
    x = frame.drop(columns=["target"])
    selector.columns = [column for column in selected_columns if column in x.columns]
    if not selector.columns:
        raise ValueError("Selected ARX has no usable predictor columns")
    selector.model = LinearRegression().fit(x[selector.columns].values, frame["target"].values)
    return selector.predict(target, exog, steps=steps)
