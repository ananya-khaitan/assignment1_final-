"""Development-locked adaptive decomposition and component routing.

This module is deliberately separate from the fixed Liu-style shadow.  It
keeps the agreed static/full-sample decomposition experiment, while removing
the unsupported assumption that the first two VMD modes must use ARIMA and
all later modes must use LSTM.  Component routes and convex stacking weights
are selected on the final 20% of the outer training sample; the final 20%
test sample is never used for those choices.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

try:
    from ..lithium_config import (
        ADAPTIVE_DEVELOPMENT_FRACTION,
        ADAPTIVE_FINAL_EPOCHS,
        ADAPTIVE_ROUTE_MODELS,
        ADAPTIVE_SEARCH_EPOCHS,
        LIU_LAGS,
        LIU_LSTM_BATCH_SIZE,
        LIU_LSTM_HIDDEN_SIZE,
        LIU_RANDOM_SEED,
    )
    from .released_paper_shadow import (
        released_feature_frame,
        causal_lasso_selection,
        released_lstm_predictions,
        static_holdout_split,
    )
except ImportError:  # pragma: no cover - direct execution from code/.
    from lithium_config import (
        ADAPTIVE_DEVELOPMENT_FRACTION,
        ADAPTIVE_FINAL_EPOCHS,
        ADAPTIVE_ROUTE_MODELS,
        ADAPTIVE_SEARCH_EPOCHS,
        LIU_LAGS,
        LIU_LSTM_BATCH_SIZE,
        LIU_LSTM_HIDDEN_SIZE,
        LIU_RANDOM_SEED,
    )
    from models.released_paper_shadow import (
        released_feature_frame,
        causal_lasso_selection,
        released_lstm_predictions,
        static_holdout_split,
    )


@dataclass(frozen=True)
class DecompositionSpec:
    method: str
    modes: int
    alpha: float | None = None

    @property
    def label(self) -> str:
        if self.method == "VMD":
            return f"VMD-K{self.modes}-A{int(self.alpha or 0)}"
        return f"CEEMDAN-M{self.modes}"


@dataclass(frozen=True)
class AdaptiveResult:
    dates: pd.DatetimeIndex
    forecast: np.ndarray
    best_candidate_forecast: np.ndarray
    search_rows: list[dict[str, Any]]
    route_rows: list[dict[str, Any]]
    weights: dict[str, float]
    selected_candidate: str


def development_and_test_slices(n_rows: int) -> tuple[slice, slice, slice]:
    """Return subtrain, development, and locked-test slices."""

    outer = static_holdout_split(n_rows, 0.20)
    inner = static_holdout_split(outer.train_stop, ADAPTIVE_DEVELOPMENT_FRACTION)
    return slice(0, inner.train_stop), slice(inner.test_start, outer.train_stop), slice(outer.test_start, n_rows)


def _seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return (LIU_RANDOM_SEED + int(digest[:8], 16)) % (2**31 - 1)


def _decompose(values: np.ndarray, spec: DecompositionSpec, train_stop: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if spec.method == "VMD":
        from vmdpy import VMD

        modes, _spectrum, _omega = VMD(
            values[:train_stop],
            alpha=float(spec.alpha),
            tau=0,
            K=int(spec.modes),
            DC=0,
            init=1,
            tol=1e-6,
        )
        components = np.asarray(modes, dtype=float)
    elif spec.method == "CEEMDAN":
        from PyEMD import CEEMDAN

        algorithm = CEEMDAN(trials=20, epsilon=0.01, parallel=False)
        algorithm.noise_seed(LIU_RANDOM_SEED)
        components = np.asarray(algorithm.ceemdan(values[:train_stop], max_imf=int(spec.modes)), dtype=float)
    else:
        raise ValueError(f"Unknown decomposition method {spec.method!r}")
    if components.ndim != 2 or components.shape[1] != train_stop:
        raise RuntimeError(f"Unexpected {spec.label} output shape {components.shape}")
    residual = values - components.sum(axis=0)
    if np.std(residual) > max(np.std(values) * 1e-8, 1e-10):
        components = np.vstack([components, residual])
    
    # Forward fill the unseen test period
    padded = np.full((components.shape[0], len(values)), np.nan)
    padded[:, :train_stop] = components
    for i in range(components.shape[0]):
        padded[i, train_stop:] = components[i, -1]
    return padded



def _scaled_arrays(frame: pd.DataFrame, train_slice: slice, eval_slice: slice) -> tuple[np.ndarray, ...]:
    train = frame.iloc[train_slice]
    evaluation = frame.iloc[eval_slice]
    x_scaler = StandardScaler().fit(train.drop(columns="target"))
    y_scaler = StandardScaler().fit(train[["target"]])
    return (
        x_scaler.transform(train.drop(columns="target")),
        y_scaler.transform(train[["target"]]),
        x_scaler.transform(evaluation.drop(columns="target")),
        y_scaler.transform(evaluation[["target"]]),
        y_scaler,
    )


def _arima_filter(y_train: np.ndarray, y_eval: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, str]:
    import pmdarima as pm

    train = np.asarray(y_train, dtype=float).reshape(-1)
    observed = np.asarray(y_eval, dtype=float).reshape(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        automatic = pm.auto_arima(
            train,
            seasonal=False,
            with_intercept=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        order = tuple(int(value) for value in automatic.order)
        attempted = [order, (1, 0, 0), (0, 1, 1), (0, 0, 0)]
        predicted = None
        used_order = None
        for candidate_order in dict.fromkeys(attempted):
            try:
                fitted = ARIMA(
                    train,
                    order=candidate_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(method_kwargs={"warn_convergence": False})
                candidate_prediction = fitted.apply(observed, refit=False).fittedvalues
                if np.isfinite(candidate_prediction).all():
                    predicted = candidate_prediction
                    used_order = candidate_order
                    break
            except (ValueError, np.linalg.LinAlgError):
                continue
        if predicted is None:
            # Stable observed-test persistence fallback for a degenerate mode.
            # It preserves the retrospective filtering information set while
            # preventing a numerically singular, near-zero component from
            # aborting the entire registered search.
            predicted = np.concatenate([[train[-1]], observed[:-1]])
            detail = "degenerate-mode persistence fallback"
        else:
            detail = str(used_order)
    values = scaler.inverse_transform(np.asarray(predicted).reshape(-1, 1)).reshape(-1)
    return values, detail


def _route_prediction(
    frame: pd.DataFrame,
    train_slice: slice,
    eval_slice: slice,
    route: str,
    *,
    epochs: int,
    seed_label: str,
) -> tuple[np.ndarray, str]:
    x_train, y_train, x_eval, y_eval, scaler = _scaled_arrays(frame, train_slice, eval_slice)
    seed = _seed(seed_label)
    np.random.seed(seed)
    if route == "ARIMA":
        return _arima_filter(y_train, y_eval, scaler)
    if route == "LSTM":
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:  # pragma: no cover - runner preflight covers torch.
            pass
        prediction = released_lstm_predictions(
            x_train,
            y_train,
            x_eval,
            scaler,
            hidden_size=LIU_LSTM_HIDDEN_SIZE,
            epochs=epochs,
            batch_size=LIU_LSTM_BATCH_SIZE,
        )
        return prediction, ""
    if route == "SVR":
        model = SVR(kernel="rbf", C=8.1, gamma="scale", epsilon=0.05)
    elif route == "RF":
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(route)
    model.fit(x_train, y_train.reshape(-1))
    scaled = model.predict(x_eval).reshape(-1, 1)
    return scaler.inverse_transform(scaled).reshape(-1), ""


def _loss(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    # The observed-test ARIMA path has a one-row state-initialization artifact;
    # score every route on the same rows after dropping that first row.
    actual = np.asarray(actual, dtype=float)[1:]
    predicted = np.asarray(predicted, dtype=float)[1:]
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    scale = max(float(np.std(actual)), 1e-12)
    return rmse, mae, rmse / scale + 0.25 * mae / scale


def _convex_weights(actual: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Fit nonnegative sum-to-one stacking weights on development data."""

    y = np.asarray(actual, dtype=float)[1:]
    matrix = np.asarray(predictions, dtype=float)[:, 1:].T
    # Optimize a dimensionless loss.  Without this common scaling, SLSQP can
    # fail on carbonate's six-figure price level and silently return the equal-
    # weight initializer even when candidate development errors differ by an
    # order of magnitude.
    scale = max(float(np.std(y)), float(np.mean(np.abs(y))), 1e-12)
    y_scaled = y / scale
    matrix_scaled = matrix / scale
    count = matrix.shape[1]
    initial = np.repeat(1.0 / count, count)
    result = minimize(
        lambda weights: float(np.mean((matrix_scaled @ weights - y_scaled) ** 2)),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        return initial
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def adaptive_forecast(
    price: pd.Series,
    exogenous: pd.DataFrame,
    specs: Iterable[DecompositionSpec],
    *,
    route_models: tuple[str, ...] = ADAPTIVE_ROUTE_MODELS,
    search_epochs: int = ADAPTIVE_SEARCH_EPOCHS,
    final_epochs: int = ADAPTIVE_FINAL_EPOCHS,
    progress_callback: Callable[[int, int, str], None] | None = None,
    checkpoint_dir: Path | None = None,
) -> AdaptiveResult:
    """Search decompositions/routes on development and score the locked test."""

    candidate_dev: list[np.ndarray] = []
    candidate_test: list[np.ndarray] = []
    candidate_labels: list[str] = []
    search_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    expected_dev_dates: pd.DatetimeIndex | None = None
    expected_test_dates: pd.DatetimeIndex | None = None

    specifications = list(specs)
    checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    for spec_number, spec in enumerate(specifications, start=1):
        checkpoint_npz = None if checkpoint_root is None else checkpoint_root / f"{spec.label}.npz"
        checkpoint_json = None if checkpoint_root is None else checkpoint_root / f"{spec.label}.json"
        if checkpoint_npz is not None and checkpoint_json is not None and checkpoint_npz.exists() and checkpoint_json.exists():
            saved = np.load(checkpoint_npz)
            metadata = json.loads(checkpoint_json.read_text(encoding="utf-8"))
            aggregate_dev = np.asarray(saved["development"], dtype=float)
            aggregate_test = np.asarray(saved["test"], dtype=float)
            dev_dates = pd.DatetimeIndex(pd.to_datetime(metadata["development_dates"]))
            test_dates = pd.DatetimeIndex(pd.to_datetime(metadata["test_dates"]))
            if expected_dev_dates is None:
                expected_dev_dates, expected_test_dates = dev_dates, test_dates
            elif not expected_dev_dates.equals(dev_dates) or not expected_test_dates.equals(test_dates):
                raise RuntimeError("Adaptive checkpoint dates are not aligned")
            candidate_dev.append(aggregate_dev)
            candidate_test.append(aggregate_test)
            candidate_labels.append(spec.label)
            search_rows.append(metadata["search_row"])
            route_rows.extend(metadata["route_rows"])
            if progress_callback is not None:
                progress_callback(spec_number, len(specifications), spec.label + " [checkpoint]")
            continue
        
        # Fix: ensure decomposition is mathematically causal.
        # We decompose ONLY up to the end of the Dev set (the 80% mark) and ffill the Test set.
        outer_split = static_holdout_split(len(price))
        components = _decompose(price.to_numpy(float), spec, outer_split.train_stop)

        dev_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for mode_index, component_values in enumerate(components, start=1):
            component = pd.Series(component_values, index=price.index, name="target", dtype=float)
            raw_frame = released_feature_frame(component, exogenous, lags=LIU_LAGS)
            selection = causal_lasso_selection(raw_frame)
            frame = raw_frame[["target", *selection.selected_columns]]
            subtrain, development, test = development_and_test_slices(len(frame))
            dev_dates = pd.DatetimeIndex(frame.index[development])
            test_dates = pd.DatetimeIndex(frame.index[test])
            if expected_dev_dates is None:
                expected_dev_dates = dev_dates
                expected_test_dates = test_dates
            elif not expected_dev_dates.equals(dev_dates) or not expected_test_dates.equals(test_dates):
                raise RuntimeError("Adaptive candidate dates are not aligned")

            actual_dev = frame.iloc[development]["target"].to_numpy(float)
            route_candidates: list[tuple[float, str, np.ndarray, str, float, float]] = []
            for route in route_models:
                prediction, detail = _route_prediction(
                    frame,
                    subtrain,
                    development,
                    route,
                    epochs=search_epochs,
                    seed_label=f"{spec.label}:{mode_index}:{route}:development",
                )
                rmse, mae, score = _loss(actual_dev, prediction)
                route_candidates.append((score, route, prediction, detail, rmse, mae))
            route_candidates.sort(key=lambda row: row[0])
            score, selected_route, dev_prediction, detail, dev_rmse, dev_mae = route_candidates[0]
            outer_train = slice(0, test.start)
            test_prediction, final_detail = _route_prediction(
                frame,
                outer_train,
                test,
                selected_route,
                epochs=final_epochs,
                seed_label=f"{spec.label}:{mode_index}:{selected_route}:test",
            )
            dev_parts.append(dev_prediction)
            test_parts.append(test_prediction)
            route_rows.append(
                {
                    "Candidate": spec.label,
                    "Method": spec.method,
                    "Configured Modes": spec.modes,
                    "Returned Component": mode_index,
                    "VMD Alpha": spec.alpha,
                    "Selected Route": selected_route,
                    "Development RMSE": dev_rmse,
                    "Development MAE": dev_mae,
                    "Selection Score": score,
                    "ARIMA Order": final_detail or detail,
                    "Selected Feature Count": len(selection.selected_columns),
                    "Selected Features": " | ".join(selection.selected_columns),
                }
            )

        aggregate_dev = np.sum(np.vstack(dev_parts), axis=0)
        aggregate_test = np.sum(np.vstack(test_parts), axis=0)
        actual_dev_price = price.reindex(expected_dev_dates).to_numpy(float)
        rmse, mae, score = _loss(actual_dev_price, aggregate_dev)
        candidate_dev.append(aggregate_dev)
        candidate_test.append(aggregate_test)
        candidate_labels.append(spec.label)
        search_rows.append(
            {
                "Candidate": spec.label,
                "Method": spec.method,
                "Configured Modes": spec.modes,
                "VMD Alpha": spec.alpha,
                "Returned Components": len(components),
                "Development RMSE": rmse,
                "Development MAE": mae,
                "Development Score": score,
            }
        )
        if checkpoint_npz is not None and checkpoint_json is not None:
            np.savez_compressed(checkpoint_npz, development=aggregate_dev, test=aggregate_test)
            checkpoint_json.write_text(
                json.dumps(
                    {
                        "development_dates": [value.isoformat() for value in expected_dev_dates],
                        "test_dates": [value.isoformat() for value in expected_test_dates],
                        "search_row": search_rows[-1],
                        "route_rows": route_rows[-len(components) :],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if progress_callback is not None:
            progress_callback(spec_number, len(specifications), spec.label)

    if expected_dev_dates is None or expected_test_dates is None:
        raise RuntimeError("No adaptive candidates were evaluated")
    dev_matrix = np.vstack(candidate_dev)
    test_matrix = np.vstack(candidate_test)
    actual_dev_price = price.reindex(expected_dev_dates).to_numpy(float)
    weights = _convex_weights(actual_dev_price, dev_matrix)
    ensemble = weights @ test_matrix
    best_index = int(np.argmin([row["Development Score"] for row in search_rows]))
    return AdaptiveResult(
        dates=expected_test_dates,
        forecast=np.asarray(ensemble, dtype=float),
        best_candidate_forecast=np.asarray(test_matrix[best_index], dtype=float),
        search_rows=search_rows,
        route_rows=route_rows,
        weights={label: float(weight) for label, weight in zip(candidate_labels, weights)},
        selected_candidate=candidate_labels[best_index],
    )


__all__ = [
    "AdaptiveResult",
    "DecompositionSpec",
    "adaptive_forecast",
    "development_and_test_slices",
]
