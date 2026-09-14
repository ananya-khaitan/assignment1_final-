"""Run the complete Liu-structured lithium forecasting paper experiment.

The experiment mirrors the reference paper's static 80:20 organization:
seven single models, four decomposition comparators, and an adaptive
VMD-LASSO-ARIMA/LSTM hybrid.  It deliberately preserves full-sample
decomposition and the released code's observed-test ARIMA filtering path, so
the resulting evidence is retrospective and noncausal.  Unlike the raw
archive, the corrected hybrid selects ARIMA orders without exogenous inputs
that are later discarded and explicitly forecasts the VMD reconstruction
residual.  A random walk is retained only as a robustness benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from decomposition.causal import approximate_entropy, vmd_decompose_train_only, ceemdan_decompose_train_only
from evaluation.metrics import dm_test, forecast_metrics
from lithium_config import (
    ADAPTIVE_CEEMDAN_MAX_IMF,
    ADAPTIVE_VMD_ALPHA,
    ADAPTIVE_VMD_K,
    FEATURE_BLOCKS,
    LIU_LAGS,
    LIU_LSTM_BATCH_SIZE,
    LIU_LSTM_EPOCHS,
    LIU_LSTM_HIDDEN_SIZE,
    LIU_RANDOM_SEED,
    EQUITY_TARGETS,
    TARGETS,
)
from models.adaptive_routing import DecompositionSpec, adaptive_forecast
from models.released_paper_shadow import (
    full_sample_vmd,
    released_feature_frame,
    causal_lasso_selection,
    released_lasso_selection,
    released_lstm_predictions,
    released_pre_process,
    static_holdout_split,
)
from trading.signals import evaluate_trading
from uncertainty.intervals import conformal_width_from_residuals, interval_metrics
from utils.reproducibility import set_global_seed


OUTPUT_ROOT = CODE_ROOT.parent / "outputs" / "lithium_paper"
SINGLE_MODELS = ("ES", "ARIMA", "SVR", "RF", "MLP", "ELM", "LSTM")
DECOMPOSITION_MODELS = (
    "CEEMDAN-ARIMA",
    "VMD-ARIMA",
    "CEEMDAN-LSTM",
    "VMD-LSTM",
    "Fixed VMD-LASSO-ARIMA/LSTM",
    "Proposed Adaptive Decomposition-Routing Ensemble",
)
ROBUSTNESS_MODELS = ("Random Walk",)


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    return float(np.nanmean(np.abs(predicted - actual) / np.where(denominator == 0, np.nan, denominator)) * 100)


def _metric_row(target: str, model: str, kind: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    metrics = forecast_metrics(actual, predicted)
    return {
        "Target": target,
        "Model": model,
        "Model Class": kind,
        "Scored Rows": len(actual),
        **metrics,
        "sMAPE": _smape(actual, predicted),
    }


def _load_common_panel(target: str, target_specs: dict[str, dict[str, Any]]) -> tuple[pd.Series, pd.DataFrame]:
    panel = pd.read_csv(target_specs[target]["processed"], index_col="Date", parse_dates=["Date"]).sort_index()
    selected = panel.dropna()
    if len(selected) % 2:
        selected = selected.iloc[1:]
    return selected.pop("Target_Close").astype(float), selected.astype(float)


def _univariate_arima_forecast(y_train: np.ndarray, y_test: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, tuple[int, int, int] | str]:
    import pmdarima as pm

    train = np.asarray(y_train, dtype=float).reshape(-1)
    test = np.asarray(y_test, dtype=float).reshape(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attempted: list[tuple[int, int, int]] = []
        try:
            automatic = pm.auto_arima(train, seasonal=False, with_intercept=True, error_action="ignore", suppress_warnings=True)
            attempted.append(tuple(int(value) for value in automatic.order))
        except (ValueError, np.linalg.LinAlgError):
            pass
        attempted.extend([(1, 0, 0), (0, 1, 1), (0, 0, 0)])
        forecasted = None
        used_order: tuple[int, int, int] | None = None
        for candidate_order in dict.fromkeys(attempted):
            try:
                fitted = ARIMA(
                    train,
                    order=candidate_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(method_kwargs={"warn_convergence": False})
                candidate_values = np.asarray(fitted.forecast(steps=len(test)), dtype=float)
                if np.isfinite(candidate_values).all():
                    forecasted = candidate_values
                    used_order = candidate_order
                    break
            except (ValueError, np.linalg.LinAlgError):
                continue
        if forecasted is None:
            # Near-zero decomposition modes can be singular under the
            # statespace optimizer.  Preserve the stable persistence fallback.
            forecasted = np.full(len(test), train[-1])
            used_order = None
    detail: tuple[int, int, int] | str = "degenerate-mode persistence fallback" if used_order is None else used_order
    return scaler.inverse_transform(np.asarray(forecasted).reshape(-1, 1)).reshape(-1), detail


def _elm_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, hidden: int = 20) -> np.ndarray:
    rng = np.random.default_rng(LIU_RANDOM_SEED)
    weights = rng.normal(scale=0.5, size=(x_train.shape[1], hidden))
    bias = rng.normal(scale=0.2, size=hidden)
    h_train = np.tanh(x_train @ weights + bias)
    h_test = np.tanh(x_test @ weights + bias)
    ridge = 1e-5 * np.eye(hidden)
    beta = np.linalg.solve(h_train.T @ h_train + ridge, h_train.T @ y_train)
    return h_test @ beta


def _direct_predictions(price: pd.Series, exogenous: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex, pd.DataFrame]:
    frame = released_feature_frame(price.rename("target"), exogenous, lags=LIU_LAGS)
    selection = causal_lasso_selection(frame)
    selected = frame[["target", *selection.selected_columns]]
    x_train, y_train, x_test, y_test, scaler_y, split = released_pre_process(selected)
    dates = pd.DatetimeIndex(selected.index[split.test_start:])
    target_test = scaler_y.inverse_transform(y_test).reshape(-1)
    predictions: dict[str, np.ndarray] = {}
    predictions["Random Walk"] = price.shift(1).reindex(dates).to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holt = Holt(scaler_y.inverse_transform(y_train).reshape(-1), initialization_method="estimated").fit(optimized=True)
        predictions["ES"] = np.asarray(holt.forecast(len(y_test)), dtype=float)
    predictions["ARIMA"], _ = _univariate_arima_forecast(y_train, y_test, scaler_y)
    for name, estimator in {
        "SVR": SVR(kernel="rbf", C=8.1, gamma=0.1),
        "RF": RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=1, random_state=LIU_RANDOM_SEED, n_jobs=1),
        "MLP": MLPRegressor(hidden_layer_sizes=(20,), activation="relu", max_iter=1000, random_state=LIU_RANDOM_SEED),
    }.items():
        estimator.fit(x_train, y_train.reshape(-1))
        scaled_prediction = estimator.predict(x_test).reshape(-1, 1)
        predictions[name] = scaler_y.inverse_transform(scaled_prediction).reshape(-1)
    elm_scaled = _elm_predict(x_train, y_train, x_test)
    predictions["ELM"] = scaler_y.inverse_transform(elm_scaled.reshape(-1, 1)).reshape(-1)
    predictions["LSTM"] = released_lstm_predictions(
        x_train,
        y_train,
        x_test,
        scaler_y,
        hidden_size=LIU_LSTM_HIDDEN_SIZE,
        epochs=LIU_LSTM_EPOCHS,
        batch_size=LIU_LSTM_BATCH_SIZE,
    )
    selection_frame = pd.DataFrame(
        {
            "Target": price.name,
            "Mode": "Undecomposed",
            "Selected Features": " | ".join(selection.selected_columns),
            "LASSO Alpha": selection.alpha,
            "Validation MSE": selection.validation_mse,
        },
        index=[0],
    )
    return predictions, dates, selection_frame


def _ceemdan(values: np.ndarray, modes: int) -> np.ndarray:
    from PyEMD import CEEMDAN

    algorithm = CEEMDAN(trials=20, epsilon=0.01, parallel=False)
    algorithm.noise_seed(LIU_RANDOM_SEED)
    components = np.asarray(algorithm.ceemdan(values, max_imf=modes), dtype=float)
    if components.ndim != 2:
        raise RuntimeError(f"Unexpected CEEMDAN output shape {components.shape}")
    return components


def _component_forecast(component: pd.Series, exogenous: pd.DataFrame, route: str) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, Any]]:
    frame = released_feature_frame(component.rename("target"), exogenous, lags=LIU_LAGS)
    selection = causal_lasso_selection(frame)
    selected = frame[["target", *selection.selected_columns]]
    x_train, y_train, x_test, y_test, scaler_y, split = released_pre_process(selected)
    dates = pd.DatetimeIndex(selected.index[split.test_start:])
    if route == "ARIMA":
        prediction, order = _univariate_arima_forecast(y_train, y_test, scaler_y)
    elif route == "LSTM":
        prediction = released_lstm_predictions(
            x_train,
            y_train,
            x_test,
            scaler_y,
            hidden_size=LIU_LSTM_HIDDEN_SIZE,
            epochs=LIU_LSTM_EPOCHS,
            batch_size=LIU_LSTM_BATCH_SIZE,
        )
        order = None
    else:
        raise ValueError(route)
    record = {
        "Selected Features": " | ".join(selection.selected_columns),
        "LASSO Alpha": selection.alpha,
        "Validation MSE": selection.validation_mse,
        "ARIMA Order": "" if order is None else str(order),
    }
    return prediction, dates, record


def _decomposition_predictions(
    target: str,
    price: pd.Series,
    exogenous: pd.DataFrame,
    target_specs: dict[str, dict[str, Any]],
    checkpoint_dir: Path | None = None,
) -> tuple[
    dict[str, np.ndarray],
    pd.DatetimeIndex,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    modes = int(target_specs[target]["modes"])
    values = price.to_numpy(float)
    split = static_holdout_split(len(values))
    train_values = values[: split.train_stop]
    
    vmd_train = vmd_decompose_train_only(train_values, alpha=3000, tau=0, k=modes, dc=0, init=1, tol=1e-6)
    ceemdan_train = ceemdan_decompose_train_only(train_values, trials=20, seed=LIU_RANDOM_SEED)
    
    # Pad to full length by carrying forward the last known state (ffill).
    # This prevents dropna() from destroying the test set while keeping it causal.
    def pad_components(components: np.ndarray) -> np.ndarray:
        padded = np.full((components.shape[0], len(values)), np.nan)
        padded[:, :split.train_stop] = components
        for i in range(components.shape[0]):
            padded[i, split.train_stop:] = components[i, -1]
        return padded
        
    decompositions = {
        "VMD": pad_components(vmd_train),
        "CEEMDAN": pad_components(ceemdan_train),
    }
    forecasts: dict[str, np.ndarray] = {}
    expected_dates: pd.DatetimeIndex | None = None
    selection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for method, components in decompositions.items():
        for route in ("ARIMA", "LSTM"):
            predicted_components: list[np.ndarray] = []
            for index, values_component in enumerate(components, start=1):
                component = pd.Series(values_component, index=price.index, dtype=float)
                prediction, dates, record = _component_forecast(component, pd.DataFrame(index=price.index), route)
                expected_dates = dates if expected_dates is None else expected_dates
                if not expected_dates.equals(dates):
                    raise RuntimeError(f"{target}/{method}-{route} component dates are not aligned")
                predicted_components.append(prediction)
                selection_rows.append({"Target": target, "Model": f"{method}-{route}", "Mode": index, **record})
            forecasts[f"{method}-{route}"] = np.sum(np.vstack(predicted_components), axis=0)

    vmd_components = decompositions["VMD"]
    vmd_residual = values - vmd_components.sum(axis=0)
    proposed_components = [*list(vmd_components), vmd_residual]
    proposed_predictions: list[np.ndarray] = []
    for index, values_component in enumerate(proposed_components, start=1):
        component = pd.Series(values_component, index=price.index, dtype=float)
        route = "ARIMA" if index <= 2 else "LSTM"
        prediction, dates, record = _component_forecast(component, exogenous, route)
        expected_dates = dates if expected_dates is None else expected_dates
        if not expected_dates.equals(dates):
            raise RuntimeError(f"{target}/proposed component dates are not aligned")
        proposed_predictions.append(prediction)
        selection_rows.append(
            {
                "Target": target,
                "Model": "Fixed VMD-LASSO-ARIMA/LSTM",
                "Mode": "Residual" if index > len(vmd_components) else index,
                "Route": route,
                **record,
            }
        )
    forecasts["Fixed VMD-LASSO-ARIMA/LSTM"] = np.sum(np.vstack(proposed_predictions), axis=0)

    # Search a deliberately finite, pre-specified candidate set.  K=9 and
    # alpha=3000 provide exact reference parity; neighboring values test
    # whether that fixed decomposition is appropriate for lithium.
    adaptive_specs = [
        DecompositionSpec("VMD", k, alpha)
        for k in ADAPTIVE_VMD_K
        for alpha in ADAPTIVE_VMD_ALPHA
    ] + [DecompositionSpec("CEEMDAN", modes) for modes in ADAPTIVE_CEEMDAN_MAX_IMF]
    adaptive_started = time.monotonic()

    def report_progress(completed: int, total: int, label: str) -> None:
        elapsed = time.monotonic() - adaptive_started
        remaining = elapsed / completed * (total - completed) if completed else float("nan")
        eta = datetime.now().astimezone().timestamp() + remaining
        eta_text = datetime.fromtimestamp(eta).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(
            f"  {target}/adaptive: {completed}/{total} candidates complete; "
            f"last={label}; elapsed={elapsed / 60:.1f}m; ETA={eta_text}",
            flush=True,
        )

    adaptive = adaptive_forecast(
        price,
        exogenous,
        adaptive_specs,
        progress_callback=report_progress,
        checkpoint_dir=checkpoint_dir,
    )
    if not expected_dates.equals(adaptive.dates):
        raise RuntimeError(f"{target}/adaptive test dates are not aligned with comparator dates")
    forecasts["Proposed Adaptive Decomposition-Routing Ensemble"] = adaptive.forecast
    adaptive_search = [{"Target": target, **row} for row in adaptive.search_rows]
    adaptive_routes = [{"Target": target, **row} for row in adaptive.route_rows]
    adaptive_metadata = {
        "target": target,
        "selected_candidate": adaptive.selected_candidate,
        "stacking_weights": adaptive.weights,
    }

    for method, components in decompositions.items():
        for index, component in enumerate(components, start=1):
            peaks, _ = find_peaks(component)
            summary_rows.append(
                {
                    "Target": target,
                    "Method": method,
                    "Mode": index,
                    "Frequency": len(peaks) / len(component),
                    "Average Period": len(component) / len(peaks) if len(peaks) else np.nan,
                    "Variance Ratio": np.var(component, ddof=1) / np.var(values, ddof=1),
                    "Correlation": np.corrcoef(values, component)[0, 1],
                    "Approximate Entropy": approximate_entropy(component),
                }
            )
    if expected_dates is None:
        raise RuntimeError("No decomposition forecasts were created")
    return (
        forecasts,
        expected_dates,
        selection_rows,
        summary_rows,
        adaptive_search,
        adaptive_routes,
        adaptive_metadata,
    )


def _forecast_rows(target: str, model: str, kind: str, dates: pd.DatetimeIndex, price: pd.Series, prediction: np.ndarray) -> list[dict[str, Any]]:
    # The released ARIMA fitted-value path has an initialization artifact in
    # its first holdout row; the source aggregate also drops that row.
    dates = dates[1:]
    prediction = np.asarray(prediction, dtype=float)[1:]
    actual = price.reindex(dates).to_numpy(float)
    origin = price.shift(1).reindex(dates).to_numpy(float)
    return [
        {
            "Target": target,
            "Model": model,
            "Model Class": kind,
            "Origin Date": price.index[price.index.get_loc(date) - 1],
            "Date": date,
            "Origin Price": origin_value,
            "Actual": actual_value,
            "Forecast": forecast_value,
        }
        for date, origin_value, actual_value, forecast_value in zip(dates, origin, actual, prediction)
    ]


def _inferential_and_trading_ledgers(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dm_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    interval_metric_rows: list[dict[str, Any]] = []
    trading_rows: list[dict[str, Any]] = []
    for target, target_frame in forecasts.groupby("Target"):
        pivot = target_frame.pivot(index="Date", columns="Model", values="Forecast")
        actual = target_frame.drop_duplicates("Date").set_index("Date")["Actual"].reindex(pivot.index)
        models = list(pivot.columns)
        for left_index, left in enumerate(models):
            for right in models[left_index + 1:]:
                common = pd.concat([actual, pivot[left], pivot[right]], axis=1).dropna()
                loss_diff = (common.iloc[:, 1] - common.iloc[:, 0]) ** 2 - (common.iloc[:, 2] - common.iloc[:, 0]) ** 2
                statistic, p_value = dm_test(loss_diff.to_numpy())
                dm_rows.append({"Target": target, "Row Model": left, "Column Model": right, "DM Statistic": statistic, "p-value": p_value})
        for model, model_frame in target_frame.groupby("Model"):
            model_frame = model_frame.sort_values("Date").reset_index(drop=True)
            calibration_count = max(20, int(math.ceil(len(model_frame) * 0.25)))
            calibration = model_frame.iloc[:calibration_count]
            evaluation = model_frame.iloc[calibration_count:].copy()
            residuals = calibration["Actual"].to_numpy() - calibration["Forecast"].to_numpy()
            widths: dict[float, float] = {}
            for alpha in (0.20, 0.10, 0.05):
                width = conformal_width_from_residuals(residuals, alpha=alpha)
                widths[alpha] = width
                metrics = interval_metrics(
                    evaluation["Actual"].to_numpy(),
                    evaluation["Forecast"].to_numpy(),
                    np.repeat(width, len(evaluation)),
                    alpha=alpha,
                )
                interval_metric_rows.append({"Target": target, "Model": model, "Nominal Coverage (%)": (1 - alpha) * 100, **metrics})
            width90 = widths[0.10]
            for row in evaluation.itertuples(index=False):
                interval_rows.append(
                    {
                        "Target": target,
                        "Model": model,
                        "Date": row.Date,
                        "Actual": row.Actual,
                        "Forecast": row.Forecast,
                        "Lower": row.Forecast - width90,
                        "Upper": row.Forecast + width90,
                        "Width": width90,
                    }
                )
            for scheme, cost, filtered in (
                ("Scheme 1", 0.0, False),
                ("Scheme 1'", 0.0, True),
                ("Scheme 2", 0.001, False),
                ("Scheme 2'", 0.001, True),
            ):
                result = evaluate_trading(
                    evaluation["Origin Price"].to_numpy(),
                    evaluation["Actual"].to_numpy(),
                    evaluation["Forecast"].to_numpy(),
                    np.repeat(width90, len(evaluation)),
                    threshold=1.0,
                    cost=cost,
                    use_uncertainty_filter=filtered,
                )
                trading_rows.append(
                    {
                        "Target": target,
                        "Model": model,
                        "Scheme": scheme,
                        "Transaction Cost": cost,
                        "Cumulative Return (%)": (result.equity_curve[-1] - 1) * 100,
                        "Average Daily Return (%)": np.mean(result.returns) * 100,
                        "Maximum Drawdown (%)": result.max_drawdown,
                        "Sharpe Ratio": result.sharpe,
                        "Number of Transactions": int(np.count_nonzero(np.diff(np.concatenate([[0], result.signal])))),
                        "Hit Rate (%)": result.hit_rate,
                    }
                )
    return pd.DataFrame(dm_rows), pd.DataFrame(interval_rows), pd.DataFrame(interval_metric_rows), pd.DataFrame(trading_rows)


def run(*, target_set: str = "prices") -> Path:
    os.environ.setdefault("PIPELINE_TORCH_THREADS", "1")
    set_global_seed(LIU_RANDOM_SEED)
    target_sets = {
        "prices": TARGETS,
        "equities": EQUITY_TARGETS,
        "all": {**TARGETS, **EQUITY_TARGETS},
    }
    if target_set not in target_sets:
        raise ValueError(f"Unknown target set: {target_set}")
    target_specs = target_sets[target_set]
    run_id = datetime.now(timezone.utc).strftime(f"lithium_{target_set}_%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    forecast_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    adaptive_search_rows: list[dict[str, Any]] = []
    adaptive_route_rows: list[dict[str, Any]] = []
    adaptive_metadata: list[dict[str, Any]] = []
    for target in target_specs:
        print(f"[{target}] loading aligned paper panel", flush=True)
        price, exogenous = _load_common_panel(target, target_specs)
        price.name = target
        direct, direct_dates, direct_selection = _direct_predictions(price, exogenous)
        selection_rows.extend(direct_selection.to_dict("records"))
        for model, prediction in direct.items():
            kind = "Robustness" if model in ROBUSTNESS_MODELS else "Single"
            forecast_rows.extend(_forecast_rows(target, model, kind, direct_dates, price, prediction))
            print(f"  {model} complete", flush=True)
        (
            decomposition,
            decomposition_dates,
            selections,
            summaries,
            target_search,
            target_routes,
            target_adaptive_metadata,
        ) = _decomposition_predictions(target, price, exogenous, target_specs, run_dir / "adaptive_checkpoints" / target)
        selection_rows.extend(selections)
        component_rows.extend(summaries)
        adaptive_search_rows.extend(target_search)
        adaptive_route_rows.extend(target_routes)
        selection_rows.extend(
            {
                "Target": row["Target"],
                "Model": "Proposed Adaptive Decomposition-Routing Ensemble",
                "Mode": f"{row['Candidate']} / C{row['Returned Component']}",
                "Route": row["Selected Route"],
                "Selected Features": row["Selected Features"],
                "LASSO Alpha": "development-locked",
                "Validation MSE": "",
                "ARIMA Order": row["ARIMA Order"],
            }
            for row in target_routes
        )
        adaptive_metadata.append(target_adaptive_metadata)
        for model, prediction in decomposition.items():
            forecast_rows.extend(_forecast_rows(target, model, "Decomposition", decomposition_dates, price, prediction))
            print(f"  {model} complete", flush=True)

    forecasts = pd.DataFrame(forecast_rows)
    metrics = pd.DataFrame(
        [
            _metric_row(target, model, str(frame["Model Class"].iloc[0]), frame["Actual"].to_numpy(), frame["Forecast"].to_numpy())
            for (target, model), frame in forecasts.groupby(["Target", "Model"])
        ]
    )
    dm, intervals, interval_summary, trading = _inferential_and_trading_ledgers(forecasts)
    outputs = {
        "forecast_ledger.csv": forecasts,
        "forecast_metrics.csv": metrics,
        "dm_tests.csv": dm,
        "interval_ledger.csv": intervals,
        "interval_metrics.csv": interval_summary,
        "trading_metrics.csv": trading,
        "mode_feature_selection.csv": pd.DataFrame(selection_rows),
        "component_summary.csv": pd.DataFrame(component_rows),
        "adaptive_candidate_search.csv": pd.DataFrame(adaptive_search_rows),
        "adaptive_component_routes.csv": pd.DataFrame(adaptive_route_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(run_dir / name, index=False)
    registry = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Reference-structured static 80:20 retrospective lithium experiment",
        "target_set": target_set,
        "classification": "noncausal; full-sample decomposition and observed-test ARIMA filtering",
        "targets": list(target_specs),
        "single_models": list(SINGLE_MODELS),
        "decomposition_models": list(DECOMPOSITION_MODELS),
        "robustness_models": list(ROBUSTNESS_MODELS),
        "lags": LIU_LAGS,
        "lstm_epochs": LIU_LSTM_EPOCHS,
        "lstm_hidden_size": LIU_LSTM_HIDDEN_SIZE,
        "lstm_batch_size": LIU_LSTM_BATCH_SIZE,
        "adaptive_selection": adaptive_metadata,
        "adaptive_protocol": {
            "route_and_stack_selection": "final 20% of the outer training sample",
            "locked_test": "final 20% of the complete aligned sample",
            "vmd_k": list(ADAPTIVE_VMD_K),
            "vmd_alpha": list(ADAPTIVE_VMD_ALPHA),
            "ceemdan_max_imf": list(ADAPTIVE_CEEMDAN_MAX_IMF),
        },
        "corrections_to_raw_archive": [
            "Low-mode ARIMA order selection is univariate because the final Statsmodels ARIMA is univariate.",
            "The VMD reconstruction residual is explicitly forecast and recombined in the proposed model.",
        ],
    }
    (run_dir / "experiment_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Saved complete lithium paper run to {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-set", choices=("prices", "equities", "all"), default="prices")
    args = parser.parse_args()
    run(target_set=args.target_set)


if __name__ == "__main__":
    main()
