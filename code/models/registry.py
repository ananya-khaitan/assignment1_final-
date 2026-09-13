"""Forecast-model registry with truthful availability labels."""

from __future__ import annotations

import os
import warnings
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover - preflight reports this clearly
    ARIMA = None

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover - preflight reports this clearly
    xgb = None

try:
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover - preflight reports this clearly
    lgb = None


def _model_threads() -> int:
    """Keep sequential rolling fits from oversubscribing the host CPU."""
    return max(1, int(os.getenv("PIPELINE_MODEL_THREADS", "1")))


def _safe_array(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Forecast inputs must be a non-empty finite one-dimensional series")
    return values


def make_lag_matrix(values: Sequence[float], lags: int) -> Tuple[np.ndarray, np.ndarray]:
    series = _safe_array(values)
    if len(series) <= lags:
        return np.empty((0, lags)), np.empty(0)
    x = np.asarray([series[index - lags : index] for index in range(lags, len(series))], dtype=float)
    return x, series[lags:]


def random_walk_forecast(train: Sequence[float], steps: int = 1) -> np.ndarray:
    series = _safe_array(train)
    return np.repeat(series[-1], steps)


def drift_forecast(train: Sequence[float], steps: int = 1) -> np.ndarray:
    series = _safe_array(train)
    if len(series) < 2:
        return random_walk_forecast(series, steps)
    slope = (series[-1] - series[0]) / (len(series) - 1)
    return np.asarray([series[-1] + slope * (step + 1) for step in range(steps)], dtype=float)


def arima_forecast(train: Sequence[float], steps: int = 1, order: Tuple[int, int, int] = (2, 1, 2)) -> np.ndarray:
    series = _safe_array(train)
    if ARIMA is None:
        raise ImportError("statsmodels is required for the ARIMA benchmark")
    if len(series) < max(10, order[0] + order[2] + 3):
        return random_walk_forecast(series, steps)
    try:
        fitted = ARIMA(series, order=order).fit(method_kwargs={"warn_convergence": False})
        return np.asarray(fitted.forecast(steps=steps), dtype=float)
    except Exception as exc:
        raise RuntimeError(f"ARIMA fitting failed: {exc}") from exc


def _temporal_cv(n_rows: int) -> TimeSeriesSplit:
    splits = min(5, max(2, n_rows // 20))
    return TimeSeriesSplit(n_splits=splits)


def linear_lasso_forecast(train: Sequence[float], steps: int = 1, lags: int = 5, model: str = "lasso") -> np.ndarray:
    series = _safe_array(train)
    x, target = make_lag_matrix(series, lags)
    if len(x) < 40:
        return random_walk_forecast(series, steps)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    if model == "elasticnet":
        estimator = ElasticNetCV(
            cv=_temporal_cv(len(x)),
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            max_iter=40_000,
            n_jobs=_model_threads(),
        )
    else:
        estimator = LassoCV(cv=_temporal_cv(len(x)), max_iter=40_000, n_jobs=_model_threads())
    estimator.fit(x_scaled, target)
    buffer = list(series[-lags:])
    output: list[float] = []
    for _ in range(steps):
        features = scaler.transform(np.asarray(buffer[-lags:], dtype=float).reshape(1, -1))
        prediction = float(estimator.predict(features)[0])
        output.append(prediction)
        buffer.append(prediction)
    return np.asarray(output, dtype=float)


def _recursive_regressor_forecast(regressor, train: Sequence[float], steps: int, lags: int) -> np.ndarray:
    series = _safe_array(train)
    x, target = make_lag_matrix(series, lags)
    if len(x) < max(40, lags + 5):
        return random_walk_forecast(series, steps)
    regressor.fit(x, target)
    buffer = list(series[-lags:])
    output: list[float] = []
    for _ in range(steps):
        prediction = float(regressor.predict(np.asarray(buffer[-lags:], dtype=float).reshape(1, -1))[0])
        output.append(prediction)
        buffer.append(prediction)
    return np.asarray(output, dtype=float)


def random_forest_forecast(train: Sequence[float], steps: int = 1, lags: int = 10) -> np.ndarray:
    return _recursive_regressor_forecast(
        RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=_model_threads()), train, steps, lags
    )


def histogram_gradient_boosting_forecast(train: Sequence[float], steps: int = 1, lags: int = 10) -> np.ndarray:
    return _recursive_regressor_forecast(
        HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=300, random_state=42), train, steps, lags
    )


def xgboost_forecast(train: Sequence[float], steps: int = 1, lags: int = 10) -> np.ndarray:
    if xgb is None:
        raise ImportError("xgboost is required for the XGBoost benchmark")
    return _recursive_regressor_forecast(
        xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=_model_threads(),
        ),
        train,
        steps,
        lags,
    )


def lightgbm_forecast(train: Sequence[float], steps: int = 1, lags: int = 10) -> np.ndarray:
    if lgb is None:
        raise ImportError("lightgbm is required for the LightGBM benchmark")
    return _recursive_regressor_forecast(
        lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=_model_threads(),
            verbosity=-1,
        ),
        train,
        steps,
        lags,
    )


class RecurrentForecaster:
    """A fitted causal LSTM/GRU that can score later observed histories.

    Parameters are fitted only when :meth:`fit` is called.  Calling
    :meth:`predict` with a longer history does not update them; it only uses
    observations that are available at that new forecast origin.  This makes
    scheduled re-estimation both computationally tractable and auditable.
    """

    def __init__(self, *, window: int = 20, hidden: int = 32, cell: str = "lstm", epochs: int = 20) -> None:
        self.window = window
        self.hidden = hidden
        self.cell = cell
        self.epochs = epochs
        self.network = None
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, train: Sequence[float]) -> "RecurrentForecaster":
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - preflight reports this clearly
            raise ImportError("torch is required for LSTM and GRU benchmarks") from exc

        # Small daily time-series batches are substantially faster and more
        # deterministic with one intra-op thread than with every host core.
        torch.set_num_threads(max(1, int(os.getenv("PIPELINE_TORCH_THREADS", "1"))))
        series = _safe_array(train)
        if len(series) < self.window + 30:
            self.network = None
            self.mean = None
            self.std = None
            return self
        self.mean = float(np.mean(series))
        self.std = float(np.std(series) + 1e-9)
        standardized = (series - self.mean) / self.std
        x = np.asarray([standardized[i : i + self.window] for i in range(len(standardized) - self.window)], dtype=float)
        target = np.asarray([standardized[i + self.window] for i in range(len(standardized) - self.window)], dtype=float)
        x_tensor = torch.tensor(x[:, :, None], dtype=torch.float32)
        y_tensor = torch.tensor(target[:, None], dtype=torch.float32)

        class RecurrentRegressor(nn.Module):
            def __init__(network_self) -> None:
                super().__init__()
                network_self.rnn = nn.GRU(1, self.hidden, batch_first=True) if self.cell == "gru" else nn.LSTM(1, self.hidden, batch_first=True)
                network_self.fc = nn.Linear(self.hidden, 1)

            def forward(network_self, inputs):
                outputs, _ = network_self.rnn(inputs)
                return network_self.fc(outputs[:, -1, :])

        self.network = RecurrentRegressor()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        loss_function = nn.MSELoss()
        self.network.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(len(x_tensor))
            for start in range(0, len(x_tensor), 64):
                index = permutation[start : start + 64]
                optimizer.zero_grad()
                loss = loss_function(self.network(x_tensor[index]), y_tensor[index])
                loss.backward()
                optimizer.step()
        self.network.eval()
        return self

    def predict(self, history: Sequence[float], steps: int = 1) -> np.ndarray:
        series = _safe_array(history)
        if self.network is None or self.mean is None or self.std is None:
            return random_walk_forecast(series, steps)
        if len(series) < self.window:
            return random_walk_forecast(series, steps)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - fit already guards this
            raise ImportError("torch is required for LSTM and GRU benchmarks") from exc
        standardized = (series - self.mean) / self.std
        buffer = list(standardized[-self.window :])
        output: list[float] = []
        with torch.no_grad():
            for _ in range(steps):
                inputs = torch.tensor(buffer[-self.window :], dtype=torch.float32).reshape(1, self.window, 1)
                prediction = float(self.network(inputs).item())
                output.append(prediction)
                buffer.append(prediction)
        return np.asarray(output, dtype=float) * self.std + self.mean


def _deep_forecast(train: Sequence[float], steps: int = 1, window: int = 20, hidden: int = 32, cell: str = "lstm") -> np.ndarray:
    series = _safe_array(train)
    fast_mode = os.getenv("PIPELINE_FAST_MODE", "0") == "1"
    if fast_mode:
        hidden = min(hidden, 16)
    epochs = 5 if fast_mode else 20
    return RecurrentForecaster(window=window, hidden=hidden, cell=cell, epochs=epochs).fit(series).predict(series, steps)


class StatefulForecastModel:
    """Fit a labelled benchmark once, then make causal predictions cheaply."""

    def __init__(self, name: str, *, lags: int = 10, rnn_epochs: int = 20) -> None:
        self.name = name
        self.lags = lags
        self.rnn_epochs = rnn_epochs
        self.estimator = None
        self.scaler = None
        self.params = None
        self.slope = 0.0

    def fit(self, train: Sequence[float]) -> "StatefulForecastModel":
        series = _safe_array(train)
        if self.name == "Random Walk":
            return self
        if self.name == "Random Walk + Drift":
            self.slope = 0.0 if len(series) < 2 else float((series[-1] - series[0]) / (len(series) - 1))
            return self
        if self.name == "ARIMA":
            if ARIMA is None:
                raise ImportError("statsmodels is required for the ARIMA benchmark")
            if len(series) < 10:
                self.params = None
            else:
                self.params = ARIMA(series, order=(2, 1, 2)).fit(method_kwargs={"warn_convergence": False}).params
            return self
        x, target = make_lag_matrix(series, self.lags)
        if len(x) < max(40, self.lags + 5):
            self.estimator = None
            self.scaler = None
            return self
        if self.name in {"LASSO", "Elastic Net"}:
            self.scaler = StandardScaler().fit(x)
            x = self.scaler.transform(x)
            self.estimator = (
                ElasticNetCV(cv=_temporal_cv(len(x)), l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], max_iter=40_000, n_jobs=_model_threads())
                if self.name == "Elastic Net"
                else LassoCV(cv=_temporal_cv(len(x)), max_iter=40_000, n_jobs=_model_threads())
            )
        elif self.name == "Random Forest":
            self.estimator = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=_model_threads())
        elif self.name == "Histogram Gradient Boosting":
            self.estimator = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=300, random_state=42)
        elif self.name == "XGBoost":
            if xgb is None:
                raise ImportError("xgboost is required for the XGBoost benchmark")
            self.estimator = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, objective="reg:squarederror", random_state=42, n_jobs=_model_threads())
        elif self.name == "LightGBM":
            if lgb is None:
                raise ImportError("lightgbm is required for the LightGBM benchmark")
            self.estimator = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=_model_threads(), verbosity=-1)
        elif self.name in {"LSTM", "GRU"}:
            self.estimator = RecurrentForecaster(cell="gru" if self.name == "GRU" else "lstm", epochs=self.rnn_epochs).fit(series)
            return self
        else:
            raise ValueError(f"Unknown stateful model: {self.name}")
        self.estimator.fit(x, target)
        return self

    def predict(self, history: Sequence[float], steps: int = 1) -> np.ndarray:
        series = _safe_array(history)
        if self.name == "Random Walk":
            return random_walk_forecast(series, steps)
        if self.name == "Random Walk + Drift":
            return np.asarray([series[-1] + self.slope * (offset + 1) for offset in range(steps)], dtype=float)
        if self.name == "ARIMA":
            if self.params is None or ARIMA is None:
                return random_walk_forecast(series, steps)
            try:
                return np.asarray(ARIMA(series, order=(2, 1, 2)).filter(self.params).forecast(steps=steps), dtype=float)
            except Exception as exc:
                raise RuntimeError(f"ARIMA filtering failed with fixed parameters: {exc}") from exc
        if self.name in {"LSTM", "GRU"}:
            if self.estimator is None:
                return random_walk_forecast(series, steps)
            return self.estimator.predict(series, steps)
        if self.estimator is None or len(series) < self.lags:
            return random_walk_forecast(series, steps)
        buffer = list(series[-self.lags :])
        output: list[float] = []
        for _ in range(steps):
            features = np.asarray(buffer[-self.lags :], dtype=float).reshape(1, -1)
            if self.scaler is not None:
                features = self.scaler.transform(features)
            # LightGBM 4.x emits this sklearn compatibility warning for its
            # internally generated feature names even though the ordered
            # numeric lag matrix is unchanged.  Suppress only that cosmetic
            # warning; any estimator error still propagates.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
                prediction = float(self.estimator.predict(features)[0])
            output.append(prediction)
            buffer.append(prediction)
        return np.asarray(output, dtype=float)


def stateful_candidate_models(*, rnn_epochs: int = 20) -> Dict[str, StatefulForecastModel]:
    """Construct one stateful instance for each benchmark label."""
    names = [
        "Random Walk",
        "Random Walk + Drift",
        "LASSO",
        "Elastic Net",
        "Random Forest",
        "Histogram Gradient Boosting",
        "ARIMA",
        "XGBoost",
        "LightGBM",
        "LSTM",
        "GRU",
    ]
    return {
        name: StatefulForecastModel(name, lags=5 if name in {"LASSO", "Elastic Net"} else 10, rnn_epochs=rnn_epochs)
        for name in names
    }


def lstm_forecast(train: Sequence[float], steps: int = 1, window: int = 20) -> np.ndarray:
    return _deep_forecast(train, steps=steps, window=window, cell="lstm")


def gru_forecast(train: Sequence[float], steps: int = 1, window: int = 20) -> np.ndarray:
    return _deep_forecast(train, steps=steps, window=window, cell="gru")


def forecast_candidates() -> Dict[str, Callable[[Sequence[float], int], np.ndarray]]:
    """Return only models that retain their truthful published labels."""
    candidates: Dict[str, Callable[[Sequence[float], int], np.ndarray]] = {
        "Random Walk": random_walk_forecast,
        "Random Walk + Drift": drift_forecast,
        "LASSO": lambda train, steps=1: linear_lasso_forecast(train, steps=steps, model="lasso"),
        "Elastic Net": lambda train, steps=1: linear_lasso_forecast(train, steps=steps, model="elasticnet"),
        "Random Forest": random_forest_forecast,
        "Histogram Gradient Boosting": histogram_gradient_boosting_forecast,
    }
    if ARIMA is not None:
        candidates["ARIMA"] = arima_forecast
    if xgb is not None:
        candidates["XGBoost"] = xgboost_forecast
    if lgb is not None:
        candidates["LightGBM"] = lightgbm_forecast
    try:
        import torch  # noqa: F401

        candidates["LSTM"] = lstm_forecast
        candidates["GRU"] = gru_forecast
    except ImportError:
        pass
    return candidates
