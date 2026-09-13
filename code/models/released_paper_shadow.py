"""Executable, provenance-preserving shadow of Liu et al.'s released code.

This module intentionally reproduces the *released implementation's* static
and non-causal mechanics for forensic comparison.  It is not part of the
critical-minerals forecasting model registry and must never be used for a
reportable forecast or trading claim.

The archived ``main_forecast.py`` is incomplete.  A repaired local copy
supplies the otherwise missing one-day horizon, five lags, 20% holdout,
``pre_process`` helper, and usable ARIMA order.  Where its repair changes the
archived execution, this module follows the raw archive: particularly,
ARIMA's test-target ``apply(...).fittedvalues`` path and the LSTM tensor
layout.  Every necessary operational repair is recorded by the runner.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StaticSplit:
    """Chronological 80:20 split equivalent to ``train_test_split(..., shuffle=False)``."""

    train_stop: int
    test_start: int
    test_size: int


@dataclass(frozen=True)
class ReleasedLassoSelection:
    """Result from the source archive's globally-scaled LASSO selection."""

    alpha: float
    validation_mse: float
    selected_columns: tuple[str, ...]
    coefficients: pd.Series
    used_empty_selection_fallback: bool


def static_holdout_split(n_rows: int, test_size: float = 0.20) -> StaticSplit:
    """Return the deterministic static holdout split used by the archive.

    Scikit-learn rounds a float ``test_size`` upward.  Matching that behavior
    matters for the final test dates, especially after the five initial lag
    rows are removed.
    """

    if n_rows < 10:
        raise ValueError("At least ten rows are needed for the released static split")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must lie strictly between zero and one")
    holdout = int(np.ceil(n_rows * test_size))
    train_stop = n_rows - holdout
    if train_stop < 5 or holdout < 1:
        raise ValueError("Static split leaves too little training or test data")
    return StaticSplit(train_stop=train_stop, test_start=train_stop, test_size=holdout)


def full_sample_vmd(values: Sequence[float], *, modes: int = 9) -> np.ndarray:
    """Run the archive's VMD call over the full series, including future data."""

    try:
        from vmdpy import VMD
    except ImportError as exc:  # pragma: no cover - validated by the runner.
        raise ImportError("vmdpy is required for the released-paper shadow") from exc
    series = np.asarray(values, dtype=float).reshape(-1)
    if len(series) < modes * 4:
        raise ValueError("Series is too short for the requested full-sample VMD")
    components, _spectrum, _omega = VMD(series, alpha=3000, tau=0, K=modes, DC=0, init=1, tol=1e-6)
    if components.shape != (modes, len(series)):
        raise RuntimeError(f"Unexpected VMD output shape: {components.shape}")
    return np.asarray(components, dtype=float)


def released_feature_frame(
    component: pd.Series,
    exogenous: pd.DataFrame,
    *,
    horizon: int = 1,
    lags: int = 5,
) -> pd.DataFrame:
    """Recreate ``load_var`` for one VMD component on project data.

    The source code makes each external series available only after shifting it
    by ``h`` rows.  The target component's own lags are likewise constructed
    before missing rows are removed.  This operation is deliberately applied
    to VMD components obtained from the *entire* sample.
    """

    if horizon != 1:
        raise ValueError("The released paper configuration only supports horizon=1")
    if lags < 1:
        raise ValueError("lags must be positive")
    target = pd.Series(component, dtype=float).copy()
    target.name = "target"
    frame = pd.DataFrame({"target": target}, index=target.index)
    aligned_exog = pd.DataFrame(exogenous).reindex(target.index).astype(float).ffill()
    for column in aligned_exog.columns:
        # Equivalent to the source's ffill followed by its ``[h:] = [: -h]``
        # assignment.  It prevents contemporaneous external values from being
        # used, but full-sample VMD and global LASSO scaling remain non-causal.
        frame[str(column)] = aligned_exog[column].shift(horizon)
    for lag in range(1, lags + 1):
        frame[f"target_{lag}"] = target.shift(lag)
    frame = frame.dropna(axis=0, how="any")
    if frame.empty:
        raise ValueError("No complete rows remain after released feature construction")
    return frame


def released_lasso_selection(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.20,
) -> ReleasedLassoSelection:
    """Apply the archive's full-sample scaler and static LASSO selection.

    The global scaler is intentional in this shadow because it is exactly the
    property that makes the protocol non-causal.  The selected LASSO is fitted
    only to the inner training slice, as in the released function.
    """

    if "target" not in frame.columns:
        raise ValueError("Released LASSO frame must contain a 'target' column")
    if frame.shape[1] < 2:
        raise ValueError("Released LASSO frame must contain at least one feature")
    outer = static_holdout_split(len(frame), 0.20)
    globally_scaled = StandardScaler().fit_transform(frame)
    scaled = pd.DataFrame(globally_scaled, index=frame.index, columns=frame.columns)
    outer_train = scaled.iloc[: outer.train_stop]
    inner = static_holdout_split(len(outer_train), validation_fraction)
    x_train = outer_train.drop(columns="target").iloc[: inner.train_stop]
    y_train = outer_train["target"].iloc[: inner.train_stop]
    x_validation = outer_train.drop(columns="target").iloc[inner.test_start :]
    y_validation = outer_train["target"].iloc[inner.test_start :]
    if min(len(x_train), len(x_validation)) < 5:
        raise ValueError("Released LASSO split leaves too few rows")

    candidates = np.arange(1e-3, 1e-1, 1e-4)
    scores: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for alpha in candidates:
            candidate = Lasso(alpha=float(alpha))
            candidate.fit(x_train, y_train)
            scores.append(float(mean_squared_error(y_validation, candidate.predict(x_validation))))
    best_index = int(np.argmin(scores))
    best_alpha = float(candidates[best_index])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = Lasso(alpha=best_alpha).fit(x_train, y_train)
    coefficients = pd.Series(model.coef_, index=x_train.columns, dtype=float)
    selected = tuple(coefficients[coefficients != 0.0].abs().sort_values(ascending=False).index.tolist())

    # The archive has no guard for a zero-feature LASSO result, although its
    # LSTM cannot accept a zero-width input.  Retaining the strongest own lag
    # is the narrowest repair and is written explicitly to the registry.
    fallback = False
    if not selected:
        fallback_column = "target_1" if "target_1" in frame.columns else str(frame.columns[1])
        selected = (fallback_column,)
        fallback = True
    return ReleasedLassoSelection(
        alpha=best_alpha,
        validation_mse=float(scores[best_index]),
        selected_columns=selected,
        coefficients=coefficients,
        used_empty_selection_fallback=fallback,
    )


def causal_lasso_selection(frame: pd.DataFrame, *, cv_folds: int = 5) -> ReleasedLassoSelection:
    """Apply train-only scaler and temporally-honest alpha selection.
    
    This function repairs the non-causal global scaling present in the released
    paper's LASSO implementation.
    """
    if "target" not in frame.columns:
        raise ValueError("Released LASSO frame must contain a 'target' column")
    if frame.shape[1] < 2:
        raise ValueError("Released LASSO frame must contain at least one feature")
        
    outer = static_holdout_split(len(frame), 0.20)
    train_frame = frame.iloc[: outer.train_stop]
    
    x_train_raw = train_frame.drop(columns="target")
    y_train_raw = train_frame["target"]
    
    # Fit scaler on train only
    x_scaler = StandardScaler().fit(x_train_raw)
    x_train_scaled = x_scaler.transform(x_train_raw)
    x_train = pd.DataFrame(x_train_scaled, index=x_train_raw.index, columns=x_train_raw.columns)
    
    # TimeSeriesSplit for alpha selection within train
    splitter = TimeSeriesSplit(n_splits=cv_folds)
    candidates = np.geomspace(1e-4, 1.0, num=30)
    scores: list[float] = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for alpha in candidates:
            fold_errors = []
            for train_idx, val_idx in splitter.split(x_train):
                candidate = Lasso(alpha=float(alpha))
                candidate.fit(x_train.iloc[train_idx], y_train_raw.iloc[train_idx])
                preds = candidate.predict(x_train.iloc[val_idx])
                fold_errors.append(float(mean_squared_error(y_train_raw.iloc[val_idx], preds)))
            scores.append(float(np.mean(fold_errors)))
            
    best_index = int(np.argmin(scores))
    best_alpha = float(candidates[best_index])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = Lasso(alpha=best_alpha).fit(x_train, y_train_raw)
        
    coefficients = pd.Series(model.coef_, index=x_train.columns, dtype=float)
    selected = tuple(coefficients[coefficients != 0.0].abs().sort_values(ascending=False).index.tolist())

    fallback = False
    if not selected:
        fallback_column = "target_1" if "target_1" in frame.columns else str(frame.columns[1])
        selected = (fallback_column,)
        fallback = True
        
    return ReleasedLassoSelection(
        alpha=best_alpha,
        validation_mse=float(scores[best_index]),
        selected_columns=selected,
        coefficients=coefficients,
        used_empty_selection_fallback=fallback,
    )


def released_pre_process(frame: pd.DataFrame, *, test_size: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StaticSplit]:
    """Use the repaired archive's train-fitted feature and target scalers."""

    if "target" not in frame.columns or frame.shape[1] < 2:
        raise ValueError("Pre-processing requires a target and at least one selected feature")
    split = static_holdout_split(len(frame), test_size)
    train, test = frame.iloc[: split.train_stop], frame.iloc[split.test_start :]
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(train.drop(columns="target"))
    x_test = x_scaler.transform(test.drop(columns="target"))
    y_train = y_scaler.fit_transform(train[["target"]])
    y_test = y_scaler.transform(test[["target"]])
    return x_train, y_train, x_test, y_test, y_scaler, split


def released_arima_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_scaler: StandardScaler,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Run the raw archive's auto-ARIMA order then ``apply`` test path.

    The target arrays are flattened solely to satisfy current pmdarima and
    statsmodels APIs; the source passes equivalent one-column arrays.
    """

    try:
        import pmdarima as pm
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:  # pragma: no cover - validated by the runner.
        raise ImportError("pmdarima and statsmodels are required for the released-paper shadow") from exc
    train_target = np.asarray(y_train, dtype=float).reshape(-1)
    test_target = np.asarray(y_test, dtype=float).reshape(-1)
    exogenous = np.asarray(x_train, dtype=float) if x_train.shape[1] else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        automatic = pm.auto_arima(train_target, X=exogenous, seasonal=False, with_intercept=True, error_action="raise")
        order = tuple(int(value) for value in automatic.order)
        fitted = ARIMA(train_target, order=order).fit()
        # The raw archive filters the observed static test targets through the
        # fitted ARIMA model instead of issuing a recursively out-of-sample
        # forecast.  Retain that behavior for an honest forensic shadow.
        filtered = fitted.apply(test_target, refit=False).fittedvalues
    predicted = target_scaler.inverse_transform(np.asarray(filtered, dtype=float).reshape(-1, 1)).reshape(-1)
    return predicted, order


def released_lstm_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    target_scaler: StandardScaler,
    *,
    hidden_size: int = 40,
    epochs: int = 100,
    batch_size: int = 128,
    dropout: float = 0.5,
) -> np.ndarray:
    """Run the archive's actual sequence-major LSTM shape and training loop."""

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - validated by the runner.
        raise ImportError("torch is required for the released-paper shadow") from exc
    if min(hidden_size, epochs, batch_size) < 1:
        raise ValueError("LSTM hidden size, epochs, and batch size must be positive")
    if len(x_train) < batch_size:
        raise ValueError("The released LSTM drops partial batches and needs one full training batch")

    class RegLSTM(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            # ``batch_first`` is deliberately false: shape (N, 1, features) is
            # interpreted as one chronological sequence, precisely as in the
            # source archive rather than as independent samples.
            # PyTorch ignores dropout for a single recurrent layer.  Passing
            # the archive's 0.5 value only emits thousands of warnings and has
            # no numerical effect, so record the parameter at the experiment
            # level but pass zero to the one-layer implementation.
            self.rnn = nn.LSTM(input_size, hidden_size, 1, dropout=0.0)
            self.reg = nn.Linear(hidden_size, 1)

        def forward(self, values: Any) -> Any:
            output = self.rnn(values)[0]
            sequence_length, batch, width = output.shape
            return self.reg(output.reshape(-1, width)).reshape(sequence_length, batch, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_tensor = torch.tensor(np.asarray(x_train)[:, np.newaxis, :], dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(np.asarray(y_train)[:, np.newaxis, :], dtype=torch.float32, device=device)
    model = RegLSTM(x_train.shape[1]).to(device)
    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())
    full_batches = int(x_train_tensor.shape[0] / batch_size)
    model.train()
    for _epoch in range(epochs):
        for batch_number in range(full_batches):
            start, stop = batch_number * batch_size, (batch_number + 1) * batch_size
            optimizer.zero_grad()
            output = model(x_train_tensor[start:stop])
            batch_loss = loss(output, y_train_tensor[start:stop])
            batch_loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(np.asarray(x_test), dtype=torch.float32, device=device).unsqueeze(1)
        predictions = model(test_tensor).detach().cpu().numpy().reshape(-1, 1)
    return target_scaler.inverse_transform(predictions).reshape(-1)
