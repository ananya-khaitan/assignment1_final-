"""Causal, auditable implementation of the paper-style VMD architecture.

This module borrows the *model family* from Liu et al. (2025), not the
released implementation.  In particular, VMD, LASSO, scalers, ARIMA-order
selection, and neural-network fitting consume only the history available at a
scheduled forecast origin.  The source paper's released code performs
full-sample VMD and global scaling, so it must not be used as a forecasting
baseline.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:  # Package imports for tests.
    from ..decomposition.causal import CausalComponentHistory, vmd_decompose_train_only
    from ..features.selection import SelectionResult, build_feature_frame, rolling_lasso_select
except ImportError:  # Direct execution from the ``code`` directory.
    from decomposition.causal import CausalComponentHistory, vmd_decompose_train_only
    from features.selection import SelectionResult, build_feature_frame, rolling_lasso_select

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover - reported pilots preflight this dependency.
    ARIMA = None


def next_origin_feature_row(
    target: pd.Series,
    exog: pd.DataFrame,
    columns: Sequence[str],
    *,
    lags: int,
) -> pd.DataFrame:
    """Build the feature row for the next unobserved target.

    ``target`` and ``exog`` end at the forecast origin.  The returned row
    therefore uses the observed origin target for ``target_lag_1`` and the
    observed origin exogenous values for a forecast of the next date.  It
    cannot read an unobserved target date.
    """

    history = pd.Series(target, dtype=float)
    available_exog = pd.DataFrame(exog).reindex(history.index).astype(float).ffill()
    if len(history) < lags:
        raise ValueError("Insufficient target history for next-origin feature construction")
    required_exogenous = [column for column in columns if not column.startswith("target_lag_")]
    if required_exogenous and (available_exog.empty or available_exog.iloc[-1][required_exogenous].isna().any()):
        raise ValueError("Latest selected exogenous values are unavailable for next-origin feature construction")

    row: dict[str, float] = {}
    for column in columns:
        if column.startswith("target_lag_"):
            try:
                lag = int(column.rsplit("_", maxsplit=1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid target lag column: {column}") from exc
            if lag < 1 or lag > lags:
                raise ValueError(f"Selected lag is outside the configured range: {column}")
            row[column] = float(history.iloc[-lag])
        elif column in available_exog.columns:
            row[column] = float(available_exog.iloc[-1][column])
        else:
            raise ValueError(f"Selected feature is not available at the forecast origin: {column}")
    return pd.DataFrame([row], columns=list(columns))


class CausalFeatureLSTM:
    """A train-window-only LASSO-selected multivariate LSTM for one VMD mode."""

    def __init__(
        self,
        *,
        lags: int = 5,
        window: int = 20,
        hidden: int = 64,
        epochs: int = 30,
        batch_size: int = 64,
        sequence_style: str = "temporal",
    ) -> None:
        self.lags = lags
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        if sequence_style not in {"temporal", "single_step"}:
            raise ValueError("sequence_style must be 'temporal' or 'single_step'")
        self.sequence_style = sequence_style
        self.selection: SelectionResult | None = None
        self.columns: list[str] = []
        self.feature_scaler: StandardScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self.network: Any | None = None

    @property
    def selected_columns(self) -> list[str]:
        return list(self.columns)

    @property
    def alpha(self) -> float | None:
        return None if self.selection is None else float(self.selection.alpha)

    def fit(self, target: pd.Series, exog: pd.DataFrame) -> "CausalFeatureLSTM":
        """Fit selection, scalers, and network on a historical slice only."""

        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - preflight handles this.
            raise ImportError("torch is required for the paper-style LSTM") from exc

        torch.set_num_threads(max(1, int(os.getenv("PIPELINE_TORCH_THREADS", "1"))))
        history = pd.Series(target, dtype=float)
        aligned_exog = pd.DataFrame(exog).reindex(history.index).astype(float).ffill()
        self.selection = rolling_lasso_select(history, aligned_exog, lags=self.lags)
        frame = build_feature_frame(history, aligned_exog, self.lags)
        self.columns = [column for column in self.selection.selected_columns if column in frame.columns]
        if not self.columns:
            raise ValueError("Paper-style LSTM has no selected features")

        x = frame[self.columns]
        y = frame["target"]
        self.feature_scaler = StandardScaler().fit(x)
        self.target_scaler = StandardScaler().fit(y.to_numpy().reshape(-1, 1))
        x_scaled = self.feature_scaler.transform(x)
        y_scaled = self.target_scaler.transform(y.to_numpy().reshape(-1, 1)).reshape(-1)
        required_rows = self.window + 20 if self.sequence_style == "temporal" else 20
        if len(x_scaled) < required_rows:
            # This branch is only an explicit early-history fallback.  It is
            # not reached in the configured long-history pilot.
            self.network = None
            return self

        if self.sequence_style == "temporal":
            sequence_end = np.arange(self.window - 1, len(x_scaled))
            x_sequences = np.asarray(
                [x_scaled[end - self.window + 1 : end + 1] for end in sequence_end], dtype=np.float32
            )
            y_sequences = y_scaled[sequence_end].astype(np.float32).reshape(-1, 1)
        else:
            # This deliberately reproduces the released source archive's LSTM
            # tensor shape: each row is a sequence of length one.  It is an
            # ablation diagnostic only; it cannot learn temporal recurrence
            # beyond the explicitly selected lagged inputs.
            x_sequences = x_scaled[:, None, :].astype(np.float32)
            y_sequences = y_scaled.astype(np.float32).reshape(-1, 1)
        x_tensor = torch.tensor(x_sequences, dtype=torch.float32)
        y_tensor = torch.tensor(y_sequences, dtype=torch.float32)

        class _Regressor(nn.Module):
            def __init__(network_self, input_size: int) -> None:
                super().__init__()
                network_self.rnn = nn.LSTM(input_size, self.hidden, num_layers=1, batch_first=True)
                network_self.head = nn.Linear(self.hidden, 1)

            def forward(network_self, values):
                outputs, _ = network_self.rnn(values)
                return network_self.head(outputs[:, -1, :])

        self.network = _Regressor(x_tensor.shape[2])
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        loss_function = nn.MSELoss()
        self.network.train()
        for _ in range(self.epochs):
            ordering = torch.randperm(len(x_tensor))
            for start in range(0, len(x_tensor), self.batch_size):
                indices = ordering[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_function(self.network(x_tensor[indices]), y_tensor[indices])
                loss.backward()
                optimizer.step()
        self.network.eval()
        return self

    def predict(self, target: pd.Series, exog: pd.DataFrame) -> float:
        """Forecast one mode one step ahead from origin-available values."""

        history = pd.Series(target, dtype=float)
        if self.network is None or self.feature_scaler is None or self.target_scaler is None:
            return float(history.iloc[-1])
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - fit already guards this.
            raise ImportError("torch is required for the paper-style LSTM") from exc

        aligned_exog = pd.DataFrame(exog).reindex(history.index).astype(float).ffill()
        frame = build_feature_frame(history, aligned_exog, self.lags)
        if self.sequence_style == "temporal" and len(frame) < self.window - 1:
            return float(history.iloc[-1])
        next_features = self.feature_scaler.transform(
            next_origin_feature_row(history, aligned_exog, self.columns, lags=self.lags)
        )
        if self.sequence_style == "temporal":
            prior_features = self.feature_scaler.transform(frame[self.columns].iloc[-(self.window - 1) :])
            sequence = np.vstack([prior_features, next_features]).astype(np.float32)
        else:
            sequence = next_features.astype(np.float32)
        with torch.no_grad():
            standardized = float(self.network(torch.tensor(sequence[None, :, :], dtype=torch.float32)).item())
        return float(self.target_scaler.inverse_transform(np.asarray([[standardized]])).item())


class BICARIMAForecaster:
    """ARIMA with order selection restricted to the current training history."""

    def __init__(self, *, max_order: int = 2) -> None:
        self.max_order = max_order
        self.order: tuple[int, int, int] | None = None
        self.params: Any | None = None

    def fit(self, values: Sequence[float]) -> "BICARIMAForecaster":
        series = np.asarray(values, dtype=float)
        self.order = None
        self.params = None
        if ARIMA is None or len(series) < 20:
            return self
        best: tuple[float, tuple[int, int, int], Any] | None = None
        for differencing in (0, 1):
            for ar_order in range(self.max_order + 1):
                for ma_order in range(self.max_order + 1):
                    order = (ar_order, differencing, ma_order)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fitted = ARIMA(
                                series,
                                order=order,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(method_kwargs={"warn_convergence": False})
                        candidate = (float(fitted.bic), order, fitted.params.copy())
                    except Exception:
                        continue
                    if np.isfinite(candidate[0]) and (best is None or candidate[0] < best[0]):
                        best = candidate
        if best is not None:
            _, self.order, self.params = best
        return self

    def predict(self, values: Sequence[float]) -> float:
        series = np.asarray(values, dtype=float)
        if self.order is None or self.params is None or ARIMA is None:
            return float(series[-1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filtered = ARIMA(
                    series,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).filter(self.params)
                return float(filtered.forecast(steps=1)[0])
        except Exception:
            return float(series[-1])


@dataclass(frozen=True)
class PaperStyleMetadata:
    fit_observations: int
    low_mode_orders: list[tuple[int, int, int] | None]
    high_mode_features: list[dict[str, Any]]


class CausalPaperStyleEngine:
    """Scheduled causal VMD-ARIMA/LASSO-LSTM forecasting engine."""

    proposed_label = "Paper-Style Causal VMD-ARIMA/LSTM"

    def __init__(
        self,
        *,
        vmd_k: int,
        low_modes: int,
        refit_interval: int,
        vmd_alpha: float,
        vmd_tau: float,
        vmd_dc: int,
        vmd_init: int,
        vmd_tol: float,
        lags: int = 5,
        lstm_window: int = 20,
        lstm_hidden: int = 64,
        lstm_epochs: int = 30,
        lstm_sequence_style: str = "temporal",
        proposed_label: str | None = None,
    ) -> None:
        if not 0 <= low_modes < vmd_k:
            raise ValueError("low_modes must be non-negative and smaller than vmd_k")
        self.vmd_k = vmd_k
        self.low_modes = low_modes
        self.refit_interval = refit_interval
        self.vmd_alpha = vmd_alpha
        self.vmd_tau = vmd_tau
        self.vmd_dc = vmd_dc
        self.vmd_init = vmd_init
        self.vmd_tol = vmd_tol
        self.lags = lags
        self.lstm_window = lstm_window
        self.lstm_hidden = lstm_hidden
        self.lstm_epochs = lstm_epochs
        self.lstm_sequence_style = lstm_sequence_style
        self.proposed_label = proposed_label or "Paper-Style Causal VMD-ARIMA/LSTM"
        self.component_history: CausalComponentHistory | None = None
        self.low_models: list[BICARIMAForecaster] = []
        self.high_models: list[CausalFeatureLSTM] = []
        self.level_arima = BICARIMAForecaster()
        self.last_fit_train_size: int | None = None
        self.refit_this_origin = False
        self.fit_count = 0
        self.metadata: PaperStyleMetadata | None = None

    def _fit(self, train_y: pd.Series, exog_train: pd.DataFrame) -> None:
        values = train_y.to_numpy(dtype=float)
        modes = vmd_decompose_train_only(
            values,
            self.vmd_alpha,
            self.vmd_tau,
            self.vmd_k,
            self.vmd_dc,
            self.vmd_init,
            self.vmd_tol,
            allow_approximation=False,
        )
        if modes.shape[0] != self.vmd_k:
            raise RuntimeError(f"Expected {self.vmd_k} VMD modes, received {modes.shape[0]}")
        self.component_history = CausalComponentHistory(modes, values)
        self.low_models = [BICARIMAForecaster().fit(mode) for mode in modes[: self.low_modes]]
        self.high_models = []
        high_metadata: list[dict[str, Any]] = []
        for mode_number, mode in enumerate(modes[self.low_modes :], start=self.low_modes + 1):
            model = CausalFeatureLSTM(
                lags=self.lags,
                window=self.lstm_window,
                hidden=self.lstm_hidden,
                epochs=self.lstm_epochs,
                sequence_style=self.lstm_sequence_style,
            ).fit(pd.Series(mode, index=train_y.index, dtype=float), exog_train)
            self.high_models.append(model)
            high_metadata.append(
                {
                    "mode": mode_number,
                    "route": "LASSO-selected multivariate LSTM",
                    "lstm_sequence_style": self.lstm_sequence_style,
                    "selected_features": model.selected_columns,
                    "alpha": model.alpha,
                }
            )
        self.level_arima = BICARIMAForecaster().fit(values)
        self.last_fit_train_size = len(train_y)
        self.fit_count += 1
        self.metadata = PaperStyleMetadata(
            fit_observations=len(train_y),
            low_mode_orders=[model.order for model in self.low_models],
            high_mode_features=high_metadata,
        )

    def forecast(self, train_y: pd.Series, exog_train: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
        due = self.last_fit_train_size is None or len(train_y) - self.last_fit_train_size >= self.refit_interval
        self.refit_this_origin = bool(due)
        if due:
            self._fit(train_y, exog_train)
        if self.component_history is None or self.metadata is None:
            raise RuntimeError("Paper-style engine has not been fit")

        modes = self.component_history.advance(train_y.to_numpy(dtype=float))
        if len(modes) != len(self.low_models) + len(self.high_models):
            raise RuntimeError("VMD component count changed between scheduled fits")
        low_predictions = [model.predict(mode) for model, mode in zip(self.low_models, modes[: self.low_modes])]
        high_predictions = [
            model.predict(pd.Series(mode, index=train_y.index, dtype=float), exog_train)
            for model, mode in zip(self.high_models, modes[self.low_modes :])
        ]
        predictions = {
            "Random Walk": float(train_y.iloc[-1]),
            "ARIMA": self.level_arima.predict(train_y.to_numpy(dtype=float)),
            self.proposed_label: float(sum(low_predictions) + sum(high_predictions)),
        }
        return (
            predictions,
            {
                self.proposed_label: {
                    "fit_observations": self.metadata.fit_observations,
                    "low_mode_orders": self.metadata.low_mode_orders,
                    "high_mode_features": self.metadata.high_mode_features,
                    "fit_count": self.fit_count,
                    "between_refit_component_update": "causal level-component innovation carry-forward",
                }
            },
        )
